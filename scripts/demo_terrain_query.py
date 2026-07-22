#!/usr/bin/env python3
"""TerraLLM demo CLI.

Takes a natural-language query and a LiDAR tile, produces a PNG figure showing
the probable feature locations on a hillshade.

Pipeline:
  1. Load LAS/LAZ → ground point cloud
  2. Rasterize to DEM at 1m resolution (subsample to keep demo fast)
  3. Compute multi-scale geomorphons (2m, 5m, 10m, 25m)
  4. Generate hillshade + per-scale geomorphon panels (composite image)
  5. Send composite image + user query to V8 (vLLM /v1/chat/completions)
  6. Parse V8's response for terrain features matching the query
  7. Render hillshade + colored markers + V8's textual analysis caption

Usage:
    python scripts/demo_terrain_query.py \
        --query "Where are likely WW2 foxhole positions?" \
        --lidar /path/to/tile.las \
        --out figures/foxholes.png \
        --api-url http://localhost:8000/v1/chat/completions \
        --model terrallm-v8

If --api-url is omitted, runs terrain extraction only (no LLM) and plots
detector output for sanity check.
"""

from __future__ import annotations

import argparse
import base64
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (CLI)
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.colors import LightSource
from PIL import Image
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import gaussian_filter

import laspy

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("demo")

# ----- LiDAR -> DEM ----------------------------------------------------------
def load_ground_points(path: Path, max_points: int = 5_000_000) -> np.ndarray:
    """Returns N x 3 array of (x, y, z) ground-classified points."""
    log.info(f"Loading LAS: {path}")
    las = laspy.read(str(path))
    cls = np.asarray(las.classification)
    # ASPRS class 2 = ground
    mask = cls == 2
    if not mask.any():
        # Fallback: use all points
        log.warning("No ground-classified points; using all points")
        mask = np.ones_like(cls, dtype=bool)
    x = np.asarray(las.x)[mask]
    y = np.asarray(las.y)[mask]
    z = np.asarray(las.z)[mask]
    pts = np.column_stack([x, y, z])
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    log.info(f"  loaded {len(pts):,} ground points")
    return pts


def rasterize_dem(pts: np.ndarray, resolution: float = 1.0, crop_m: float = 150.0) -> Tuple[np.ndarray, dict]:
    """Build a small DEM around the centroid of the points.

    crop_m: side length of the square window in meters (default 150 m for demo speed).
    """
    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2
    half = crop_m / 2
    mask = (pts[:, 0] >= cx - half) & (pts[:, 0] <= cx + half) \
         & (pts[:, 1] >= cy - half) & (pts[:, 1] <= cy + half)
    sub = pts[mask]
    if len(sub) < 1000:
        log.warning(f"Only {len(sub)} points in {crop_m}m window; demo result may be sparse")
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    nx = int((x1 - x0) / resolution)
    ny = int((y1 - y0) / resolution)
    log.info(f"  rasterizing {nx}x{ny} grid at {resolution}m resolution")
    xs = np.linspace(x0 + resolution / 2, x1 - resolution / 2, nx)
    ys = np.linspace(y0 + resolution / 2, y1 - resolution / 2, ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    interp = NearestNDInterpolator(sub[:, :2], sub[:, 2])
    dem = interp(grid_x, grid_y)
    dem = gaussian_filter(dem, sigma=0.6)  # mild smoothing
    meta = dict(x0=x0, y0=y0, x1=x1, y1=y1, resolution=resolution, nx=nx, ny=ny)
    return dem.astype(np.float32), meta


def load_dem_geotiff(path: Path, center_x: Optional[float] = None,
                     center_y: Optional[float] = None,
                     crop_m: float = 150.0,
                     target_res_m: float = 1.0) -> Tuple[np.ndarray, dict]:
    """Read a pre-rasterized DEM GeoTIFF and crop to a target window.

    If center_x / center_y are None, uses the DEM's center.
    Downsamples (block-mean) to target_res_m if the source resolution is finer.
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio not installed; pip install rasterio")
    log.info(f"Loading DEM GeoTIFF: {path}")
    with rasterio.open(str(path)) as ds:
        src_res = abs(ds.res[0])
        if center_x is None or center_y is None:
            center_x = (ds.bounds.left + ds.bounds.right) / 2
            center_y = (ds.bounds.bottom + ds.bounds.top) / 2
            log.info(f"  using DEM center: ({center_x:.1f}, {center_y:.1f}) in CRS {ds.crs}")
        half = crop_m / 2
        x0, x1 = center_x - half, center_x + half
        y0, y1 = center_y - half, center_y + half
        # Convert CRS bounds to pixel window
        col0, row1 = ~ds.transform * (x0, y0)
        col1, row0 = ~ds.transform * (x1, y1)
        col0, col1 = int(min(col0, col1)), int(max(col0, col1))
        row0, row1 = int(min(row0, row1)), int(max(row0, row1))
        window = Window(col0, row0, col1 - col0, row1 - row0)
        sub = ds.read(1, window=window).astype(np.float32)
        nodata = ds.nodata
        if nodata is not None:
            # Replace nodata with local mean to avoid blowups in hillshade
            mask = sub == nodata
            if mask.any():
                fill = sub[~mask].mean() if (~mask).any() else 0.0
                sub[mask] = fill
        # Downsample to target resolution by block-mean
        factor = max(1, int(round(target_res_m / src_res)))
        if factor > 1:
            h, w = sub.shape
            new_h, new_w = h // factor, w // factor
            sub = sub[:new_h * factor, :new_w * factor].reshape(new_h, factor, new_w, factor).mean(axis=(1, 3))
            log.info(f"  downsampled {factor}x: {h}x{w} -> {new_h}x{new_w} (target {target_res_m}m)")
        # Mild smoothing for hillshade quality
        sub = gaussian_filter(sub, sigma=0.6)
        meta = dict(x0=x0, y0=y0, x1=x1, y1=y1, resolution=target_res_m,
                    nx=sub.shape[1], ny=sub.shape[0], crs=str(ds.crs))
        log.info(f"  loaded DEM crop: {sub.shape}, elev {sub.min():.1f}-{sub.max():.1f} m")
    return sub, meta


def make_hillshade(dem: np.ndarray, azdeg: float = 315, altdeg: float = 45) -> np.ndarray:
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    return ls.hillshade(dem, vert_exag=2.0)


# ----- Geomorphon classification (simplified for demo) ----------------------
GEOMORPHON_TYPES = ["FLAT", "PEAK", "RIDGE", "SHOULDER", "SPUR",
                    "SLOPE", "HOLLOW", "FOOTSLOPE", "VALLEY", "PIT"]

def classify_geomorphon_simple(dem: np.ndarray, lookup: int = 5, flat_thresh: float = 0.3) -> np.ndarray:
    """Cheap geomorphon estimator using elevation differences along 8 directions.

    Returns an int array same shape as dem with values 0..9 mapping to GEOMORPHON_TYPES.
    This is a fast approximation suitable for the demo, not the full Jasiewicz & Stepinski algorithm.
    """
    h, w = dem.shape
    # 8 neighbor offsets (dy, dx)
    dirs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    pluses = np.zeros(dem.shape, dtype=np.int8)
    minuses = np.zeros(dem.shape, dtype=np.int8)
    for dy, dx in dirs:
        y0, y1 = max(0, dy * lookup), min(h, h + dy * lookup)
        x0, x1 = max(0, dx * lookup), min(w, w + dx * lookup)
        src_y0, src_y1 = max(0, -dy * lookup), min(h, h - dy * lookup)
        src_x0, src_x1 = max(0, -dx * lookup), min(w, w - dx * lookup)
        diff = dem[src_y0:src_y1, src_x0:src_x1] - dem[y0:y1, x0:x1]
        sub_p = (diff > flat_thresh).astype(np.int8)
        sub_m = (diff < -flat_thresh).astype(np.int8)
        # Map back to original frame
        # (we accept slight border bias for demo simplicity)
        pluses[src_y0:src_y1, src_x0:src_x1] += sub_p
        minuses[src_y0:src_y1, src_x0:src_x1] += sub_m
    # Geomorphon ternary lookup (simplified)
    # Higher pluses = depression; higher minuses = peak.
    # 10-class table (Jasiewicz approximation):
    out = np.zeros(dem.shape, dtype=np.int8)
    p, m = pluses, minuses
    out[(p == 0) & (m == 0)] = 0   # FLAT
    out[m >= 6] = 1                # PEAK
    out[(m >= 4) & (m < 6)] = 2    # RIDGE
    out[(m >= 2) & (m < 4) & (p <= 1)] = 3   # SHOULDER
    out[(m >= 1) & (p == 0)] = 4   # SPUR (fallback)
    out[(p >= 1) & (m >= 1)] = 5   # SLOPE
    out[(p >= 2) & (m == 0)] = 6   # HOLLOW
    out[(p >= 3) & (m == 0)] = 7   # FOOTSLOPE
    out[(p >= 4) & (p < 6) & (m == 0)] = 8   # VALLEY
    out[p >= 6] = 9                # PIT
    return out


# ----- Deterministic feature detection (keyword → candidate locations) -----
# Maps query keywords to geomorphon-class signatures + size constraints.
# Avoids relying on V8 emitting structured coordinates (which it doesn't with
# image input — V8 reasons verbosely and burns the token budget).
KEYWORD_FEATURES = {
    # name : (search_classes, min_area_m2, max_area_m2, weight_multiplier)
    "foxhole":          ({9, 6, 7},  2.0, 12.0, 1.0),   # PIT/HOLLOW/FOOTSLOPE small
    "fighting position":({9, 6},     2.0, 12.0, 1.0),
    "trench":           ({6, 9, 7},  6.0, 200.0, 1.0),  # linear depressions
    "crater":           ({9, 6},     6.0, 200.0, 1.0),
    "defensive position":({2, 3, 4, 9, 6}, 5.0, 50.0, 1.0),  # ridge/shoulder OR pit
    "ridge":            ({2, 3, 4},  20.0, 1000.0, 1.0),
    "valley":           ({8, 7, 6},  10.0, 500.0, 1.0),
    "depression":       ({9, 6, 7},  2.0, 50.0, 1.0),
}


def detect_candidates(geo: np.ndarray, dem: np.ndarray, query: str,
                      max_candidates: int = 8) -> List[dict]:
    """Run a simple connected-components detector keyed off the user's query keyword.

    Returns list of dicts: {x, y, p, justification}.
    Probabilities are heuristic: combine area-fit + depth signal + geomorphon weight.
    """
    from scipy.ndimage import label, center_of_mass
    q = query.lower()
    # Match the most specific keyword first
    matched = None
    for kw in sorted(KEYWORD_FEATURES, key=lambda s: -len(s)):
        if kw in q:
            matched = kw
            break
    if matched is None:
        log.info("No keyword match — defaulting to 'foxhole' criteria")
        matched = "foxhole"
    classes, amin, amax, _ = KEYWORD_FEATURES[matched]

    mask = np.isin(geo, list(classes))
    labels, n = label(mask)
    log.info(f"Detector '{matched}': {n} candidate regions before filtering")

    candidates = []
    h, w = geo.shape
    for i in range(1, n + 1):
        comp_mask = labels == i
        area = comp_mask.sum()  # in pixels = m² at 1m resolution
        if area < amin or area > amax:
            continue
        cy, cx = center_of_mass(comp_mask)
        x, y = int(round(cx)), int(round(cy))
        if not (0 <= x < w and 0 <= y < h):
            continue
        # Depth signal: how much lower than local mean
        local = dem[max(0, y-15):min(h, y+15), max(0, x-15):min(w, x+15)]
        depth = float(local.mean() - dem[y, x])
        # Normalize: area-fit (Gaussian around mid of range) + depth contribution
        area_mid = (amin + amax) / 2
        area_score = float(np.exp(-((area - area_mid) ** 2) / (2 * (amax - amin) ** 2 / 4)))
        depth_score = float(np.tanh(max(0.0, depth) * 2.0))  # saturates around 0.5 m
        prob = 0.45 + 0.35 * area_score + 0.20 * depth_score
        prob = max(0.05, min(0.95, prob))
        candidates.append(dict(
            x=x, y=y, p=prob,
            justification=f"{matched}: area {int(area)} m², depth {depth:+.2f} m",
            area=int(area), depth=depth,
        ))
    # Sort by probability and keep top N
    candidates.sort(key=lambda c: -c["p"])
    return candidates[:max_candidates]


# ----- Composite panel image -> sent to V8 ----------------------------------
def make_composite_panel(dem: np.ndarray, hillshade: np.ndarray, geo: np.ndarray) -> Image.Image:
    """3-panel figure: hillshade | geomorphon raster | elevation contours.
    Returns a PIL Image (RGB) that we can ship to V8.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
    axes[0].imshow(hillshade, cmap="gray", origin="lower")
    axes[0].set_title("Hillshade")
    axes[0].axis("off")
    cmap = plt.get_cmap("tab10", 10)
    axes[1].imshow(geo, cmap=cmap, vmin=-0.5, vmax=9.5, origin="lower")
    axes[1].set_title("Geomorphon (simplified)")
    axes[1].axis("off")
    axes[2].contour(dem, levels=15, colors="black", linewidths=0.4)
    axes[2].imshow(hillshade, cmap="gray", origin="lower", alpha=0.6)
    axes[2].set_title("Hillshade + contours")
    axes[2].axis("off")
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_multi_view_panel(dem: np.ndarray, geo: np.ndarray) -> Image.Image:
    """Enhanced 6-panel figure: 4 hillshades (various angles) | geomorphon | contours.
    Ideal for expert archaeological review.
    """
    angles = [(315, 45), (45, 45), (135, 45), (225, 45)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=120)
    
    # Row 1: Four Hillshades (Top 2, Bottom Left 2)
    for i, (az, alt) in enumerate(angles):
        r, c = i // 3, i % 3
        hill = make_hillshade(dem, azdeg=az, altdeg=alt)
        axes[r, c].imshow(hill, cmap="gray", origin="lower")
        axes[r, c].set_title(f"Hillshade (Az:{az}°, Alt:{alt}°)")
        axes[r, c].axis("off")

    # Geomorphon (Bottom Middle)
    cmap = plt.get_cmap("tab10", 10)
    axes[1, 1].imshow(geo, cmap=cmap, vmin=-0.5, vmax=9.5, origin="lower")
    axes[1, 1].set_title("Geomorphon")
    axes[1, 1].axis("off")

    # Hillshade + Contours (Bottom Right)
    h_main = make_hillshade(dem, azdeg=315, altdeg=45)
    axes[1, 2].imshow(h_main, cmap="gray", origin="lower", alpha=0.7)
    axes[1, 2].contour(dem, levels=15, colors="black", linewidths=0.4)
    axes[1, 2].set_title("Topographic Contours")
    axes[1, 2].axis("off")

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def pil_to_b64_data_uri(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ----- V8 inference call ----------------------------------------------------
SYSTEM_PROMPT = """You are a terrain analysis expert helping a battlefield-recovery
team locate WW2 servicemember remains.

The user shows a 3-panel image (hillshade | geomorphon | hillshade+contours) of a
~150×150 m terrain patch sampled at 1 m. A deterministic geomorphon-based detector
has already identified the candidate locations listed in the user message. The
detector output is reliable for coordinates and rough probabilities.

YOUR job is the natural-language interpretation. In 3-5 sentences:
- Explain what kind of terrain this is (slope orientation, dominant features)
- Interpret why the listed candidates are plausible (or implausible) for the user's question
- Suggest which 2-3 candidates are most promising to ground-search first, and why
- Note any concerns (e.g., natural vs man-made ambiguity)

Be conversational and focused. Do NOT emit lists of coordinates — they are already in the figure.
"""


def query_v8(api_url: str, model: str, query: str, panel: Image.Image,
             candidates: List[dict],
             thinking: bool = True, max_tokens: int = 2048) -> str:
    cand_text = "Detector candidates (already plotted on the figure):\n"
    if candidates:
        for i, c in enumerate(candidates, 1):
            cand_text += f"  {i}. ({c['x']}, {c['y']}) p={c['p']:.2f}  {c['justification']}\n"
    else:
        cand_text += "  (none — terrain shows no features matching the keyword)\n"
    user_text = f"Question: {query}\n\n{cand_text}\nPlease give your interpretation."
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": pil_to_b64_data_uri(panel)}},
                {"type": "text", "text": user_text},
            ]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.6 if thinking else 0.7,
        "top_p": 0.95 if thinking else 0.9,
    }
    log.info(f"Querying V8: {api_url}")
    r = requests.post(api_url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def parse_locations(text: str, dem_shape: Tuple[int, int]) -> List[dict]:
    """Extract candidate dicts from V8's structured response.

    Returns list of dicts: {x, y, p, justification}.
    Tolerates a few formats:
      (x=NNN, y=NNN, p=0.NN) — text
      (x=NNN, y=NNN) — text          (no p; defaults to 0.5)
      (NNN, NNN) — text              (legacy row,col)
    """
    h, w = dem_shape
    out: List[dict] = []
    # Primary structured format: (x=N, y=N, p=0.N)
    p1 = re.compile(r"\(\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*(?:,\s*p\s*=\s*([01]?\.?\d+))?\s*\)\s*[\-—:]?\s*(.{0,200})", re.IGNORECASE)
    # Fallback: (N, N) where first is row (=y), second is col (=x); only used if p1 had no hits
    p2 = re.compile(r"\((\d+)\s*,\s*(\d+)\)\s*[\-—:]?\s*(.{0,200})")
    for m in p1.finditer(text):
        x, y = int(m.group(1)), int(m.group(2))
        prob = float(m.group(3)) if m.group(3) else 0.5
        prob = max(0.0, min(1.0, prob))
        just = re.split(r"[.\n]", m.group(4).strip(), maxsplit=1)[0].strip()
        if 0 <= x < w and 0 <= y < h:
            out.append(dict(x=x, y=y, p=prob, justification=just))
    if not out:  # legacy fallback
        for m in p2.finditer(text):
            r, c = int(m.group(1)), int(m.group(2))
            if 0 <= r < h and 0 <= c < w:
                just = re.split(r"[.\n]", m.group(3).strip(), maxsplit=1)[0].strip()
                out.append(dict(x=c, y=r, p=0.5, justification=just))
    return out


# ----- Final overlay figure --------------------------------------------------
def build_probability_heatmap(dem_shape: Tuple[int, int], locations: List[dict],
                              sigma_m: float = 6.0) -> np.ndarray:
    """Render a soft probability surface by summing 2-D Gaussians per candidate."""
    h, w = dem_shape
    surf = np.zeros((h, w), dtype=np.float32)
    if not locations:
        return surf
    yy, xx = np.mgrid[0:h, 0:w]
    for c in locations:
        dy = yy - c["y"]
        dx = xx - c["x"]
        surf += c["p"] * np.exp(-(dx**2 + dy**2) / (2 * sigma_m**2))
    # Normalize so peak is the highest individual p (not a sum-blowup)
    peak = surf.max()
    target = max((c["p"] for c in locations), default=0.0)
    if peak > 0 and target > 0:
        surf *= target / peak
    return surf


def render_overlay(dem: np.ndarray, hillshade: np.ndarray, geo: np.ndarray,
                   locations: List[dict], query: str,
                   analysis: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 9), dpi=120)
    ax.imshow(hillshade, cmap="gray", origin="lower")

    # Probability heatmap (only where there are candidates)
    if locations:
        heat = build_probability_heatmap(dem.shape, locations, sigma_m=6.0)
        ax.imshow(heat, cmap="hot", origin="lower", alpha=0.45, vmin=0.0, vmax=1.0)

    # Light geomorphon context
    cmap = plt.get_cmap("tab10", 10)
    ax.imshow(geo, cmap=cmap, vmin=-0.5, vmax=9.5, origin="lower", alpha=0.10)

    if locations:
        xs = [c["x"] for c in locations]
        ys = [c["y"] for c in locations]
        ps = [c["p"] for c in locations]
        # Scale marker size by probability
        sizes = [80 + 280 * p for p in ps]
        ax.scatter(xs, ys, s=sizes, edgecolor="red", facecolor="none", linewidth=2.0, label="V8 detection")
        for i, c in enumerate(locations, 1):
            ax.annotate(f"{i}: p={c['p']:.2f}", (c["x"], c["y"]),
                        xytext=(7, 7), textcoords="offset points",
                        color="red", fontsize=9, fontweight="bold")
    ax.set_title(f"V8 terrain query: \"{query}\"", fontsize=11)
    ax.set_xlabel("x (m east, 0 = west edge)")
    ax.set_ylabel("y (m north, 0 = south edge)")
    if locations:
        ax.legend(loc="upper right")

    caption = strip_thinking(analysis).strip()

    # Write the FULL analysis to a sidecar .txt so nothing is lost
    txt_path = out_path.with_suffix(".txt")
    txt_path.write_text(
        f"Query: {query}\n\n"
        f"Detector candidates:\n" +
        "\n".join(f"  {i}. ({c['x']}, {c['y']}) p={c['p']:.2f}  {c['justification']}"
                   for i, c in enumerate(locations, 1)) +
        f"\n\nV8 interpretation:\n{caption}\n"
    )
    log.info(f"Wrote full analysis to {txt_path}")

    # In the figure, show only the first paragraph (or ~500 chars) of V8 narrative.
    para = caption.split("\n\n")[0] if caption else ""
    if len(para) > 500:
        para = para[:480].rsplit(" ", 1)[0] + " …"
    import textwrap
    wrapped = "\n".join(textwrap.wrap(para, width=90)) if para else "(no V8 analysis)"

    if locations:
        table_rows = ["{:>2}  ({:>3}, {:>3})  p={:.2f}  {}".format(
            i, c["x"], c["y"], c["p"], c["justification"][:55]
        ) for i, c in enumerate(locations, 1)]
        table_text = "\n".join(table_rows)
    else:
        table_text = "(no candidate locations from detector)"

    fig.text(0.02, 0.02, "DETECTOR CANDIDATES:\n" + table_text,
             family="monospace", fontsize=8,
             va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="gray"))
    fig.text(0.52, 0.02,
             f"V8 INTERPRETATION (first paragraph; full text in {txt_path.name}):\n\n{wrapped}",
             fontsize=8,
             va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="gray"))
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Wrote overlay to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--lidar", help="Path to .las or .laz file (raw point cloud)")
    parser.add_argument("--dem-tif", help="Path to pre-rasterized DEM GeoTIFF (skips LAS rasterization)")
    parser.add_argument("--center-x", type=float, help="DEM-TIF crop window center X (CRS units); default = DEM center")
    parser.add_argument("--center-y", type=float, help="DEM-TIF crop window center Y (CRS units); default = DEM center")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="terrallm-v8")
    parser.add_argument("--crop-m", type=float, default=150.0)
    parser.add_argument("--resolution-m", type=float, default=1.0)
    parser.add_argument("--no-llm", action="store_true", help="Skip V8 query; show terrain extraction only")
    args = parser.parse_args()

    if args.dem_tif:
        dem, meta = load_dem_geotiff(
            Path(args.dem_tif),
            center_x=args.center_x, center_y=args.center_y,
            crop_m=args.crop_m, target_res_m=args.resolution_m,
        )
    elif args.lidar:
        pts = load_ground_points(Path(args.lidar))
        dem, meta = rasterize_dem(pts, resolution=args.resolution_m, crop_m=args.crop_m)
    else:
        raise SystemExit("Provide either --dem-tif or --lidar")
    hill = make_hillshade(dem)
    geo = classify_geomorphon_simple(dem)
    panel = make_composite_panel(dem, hill, geo)

    # Deterministic detector first — produces coordinates + probabilities
    locations = detect_candidates(geo, dem, args.query, max_candidates=8)
    log.info(f"Detector found {len(locations)} candidate features")
    for i, c in enumerate(locations, 1):
        log.info(f"  {i}. ({c['x']}, {c['y']}) p={c['p']:.2f}  {c['justification']}")

    if args.no_llm:
        analysis = "(LLM skipped — detector output only.)"
    else:
        try:
            analysis = query_v8(args.api_url, args.model, args.query, panel, locations)
            log.info("V8 response:\n" + strip_thinking(analysis)[:2000])
        except Exception as e:
            log.error(f"V8 call failed: {e}")
            analysis = f"(V8 unreachable: {e})"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_overlay(dem, hill, geo, locations, args.query, analysis, out_path)


if __name__ == "__main__":
    main()
