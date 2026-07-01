"""Build a detection-review package for an AOI.

For each candidate detection (CNN peaks and/or shape-detector polygons) in an
AOI, fetch the bare-earth DEM, compute relief layers, and render a mound-scale
relief thumbnail. Writes:

    data/review/{aoi}/candidates.json   one record per candidate
    data/review/{aoi}/thumbs/{id}.png   relief thumbnail per candidate

The review web app (scripts/review_server.py) serves these and records verdicts.

Usage:
    python scripts/build_review.py esk027_icvillage_is winterville_recheck ...
    python scripts/build_review.py --all          # every AOI with peaks
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "src")
from earthwork_llm.ingestion.imageserver import Usgs3depImageServerSource, WindowRequest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

UTM = "EPSG:26915"
RES = 1.0
TILE = 1024
THUMB_M = 240          # thumbnail window side, metres (mound + context)
PAD_M = 220            # pad around candidate bbox when fetching the AOI DEM
DATA = Path("data")
TO_WGS = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)


def fetch_dem(xmin, ymax, W, H):
    src = Usgs3depImageServerSource()
    full = np.full((H, W), np.nan, dtype="float32")
    for py in range(0, H, TILE):
        for px in range(0, W, TILE):
            tcx = (xmin + px) + TILE / 2.0
            tcy = (ymax - py) - TILE / 2.0
            req = WindowRequest(center_x=tcx, center_y=tcy, utm_crs=UTM,
                                resolution_m=RES, size_px=TILE)
            for attempt in range(5):
                try:
                    arr = src.fetch_window(req); break
                except Exception:
                    if attempt == 4: raise
                    time.sleep(2 * (attempt + 1))
            h, w = min(TILE, H - py), min(TILE, W - px)
            full[py:py+h, px:px+w] = arr[:h, :w]
    return full


def _lrm(rel, mask, sigma_m):
    weight = mask.astype("float32")
    sm = gaussian_filter(rel * weight, sigma=max(sigma_m, 0.5), mode="reflect")
    nm = gaussian_filter(weight, sigma=max(sigma_m, 0.5), mode="reflect")
    return rel - np.where(nm > 1e-6, sm / nm, 0.0)


def _hillshade(rel, px_m=1.0):
    azs = [0, 45, 90, 135, 180, 225, 270, 315]
    alt = math.radians(20); cz = math.cos(math.pi/2 - alt); sz = math.sin(math.pi/2 - alt)
    gy, gx = np.gradient(rel, px_m)
    slp = np.arctan(np.hypot(gy, gx)); asp = np.arctan2(-gx, gy)
    out = np.zeros_like(rel)
    for a in azs:
        ar = math.radians(a)
        out += np.clip(cz*np.cos(slp) + sz*np.sin(slp)*np.cos(ar-asp), 0, 1)
    return out / len(azs)


def load_candidates(aoi_name: str):
    """Merge CNN peaks + shape detections for an AOI into one list."""
    cands = []
    cnn = DATA / "cnn_inference" / aoi_name / "mound_peaks.geojson"
    if cnn.exists():
        for f in json.loads(cnn.read_text())["features"]:
            p = f["properties"]
            cands.append({"id": p["feature_id"], "source": "cnn",
                          "score": round(float(p.get("score", 0)), 3),
                          "utm_x": float(p["utm_x"]), "utm_y": float(p["utm_y"])})
    shp = DATA / "features" / aoi_name / "mounds_shape_filtered.geojson"
    if shp.exists():
        for f in json.loads(shp.read_text())["features"]:
            p = f["properties"]
            cands.append({"id": p.get("feature_id"), "source": "shape",
                          "score": round(float(p.get("lrm_value", 0)), 3),
                          "utm_x": float(p["utm_x"]), "utm_y": float(p["utm_y"]),
                          "diameter_m": p.get("diameter_m"),
                          "circularity": p.get("circularity")})
    return cands


TO_UTM = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)


def load_terrallm(geojson_path: str, include_rejects: bool = False):
    """Load TerraLLM earthwork detections (WGS84) → UTM15N candidate records.

    Carries the LLM fields (prob, height, shield decision, justification,
    llm_analysis) so the reviewer sees the model's own reasoning.
    """
    doc = json.loads(Path(geojson_path).read_text())
    cands = []
    for i, f in enumerate(doc["features"], 1):
        p = f["properties"]
        dec = p.get("shield_decision", "")
        if not include_rejects and dec == "reject":
            continue
        lon, lat = f["geometry"]["coordinates"][:2]
        ux, uy = TO_UTM.transform(lon, lat)
        cands.append({
            "id": f"tllm_{i:03d}", "source": "terrallm",
            "score": round(float(p.get("prob", 0)), 3),
            "utm_x": float(ux), "utm_y": float(uy),
            "height_m": round(float(p.get("height_m", 0)), 2),
            "area_m2": p.get("area_m2"),
            "shield_decision": dec,
            "nlcd_class": p.get("nlcd_class", ""),
            "justification": p.get("justification", ""),
            "llm_analysis": p.get("llm_analysis", ""),
        })
    return cands


def build_aoi(name: str, candidates=None):
    out = DATA / "review" / name
    thumbs = out / "thumbs"; thumbs.mkdir(parents=True, exist_ok=True)
    cands = candidates if candidates is not None else load_candidates(name)
    if not cands:
        print(f"  {name}: no candidates"); return None
    xs = [c["utm_x"] for c in cands]; ys = [c["utm_y"] for c in cands]
    xmin = math.floor(min(xs) - PAD_M); xmax = math.ceil(max(xs) + PAD_M)
    ymin = math.floor(min(ys) - PAD_M); ymax = math.ceil(max(ys) + PAD_M)
    W = int(xmax - xmin); H = int(ymax - ymin)
    print(f"  {name}: {len(cands)} candidates · DEM {W}x{H} m", flush=True)
    dem = fetch_dem(xmin, ymax, W, H)
    mask = np.isfinite(dem)
    rel = np.where(mask, dem - np.nanmedian(dem), 0.0).astype("float32")
    lw = _lrm(rel, mask, 40.0); ln = _lrm(rel, mask, 10.0); hs = _hillshade(rel)
    half = int(THUMB_M / 2)
    for c in cands:
        pc = int(round(c["utm_x"] - xmin)); pr = int(round(ymax - c["utm_y"]))
        r0, r1 = max(0, pr-half), min(H, pr+half); c0, c1 = max(0, pc-half), min(W, pc+half)
        sub_hs = hs[r0:r1, c0:c1]; sub_lw = lw[r0:r1, c0:c1]; sub_ln = ln[r0:r1, c0:c1]
        lon, lat = TO_WGS.transform(c["utm_x"], c["utm_y"])
        c["lat"], c["lon"] = round(lat, 6), round(lon, 6)
        c["lrm_wide_m"] = round(float(lw[pr, pc]), 2) if (0 <= pr < H and 0 <= pc < W) else None
        c["thumb"] = f"thumbs/{c['id']}.png"
        fig, ax = plt.subplots(1, 3, figsize=(9, 3.1))
        for a, arr, t, cm in ((ax[0], sub_hs, "hillshade", "gray"),
                              (ax[1], sub_ln, "LRM narrow", "RdBu_r"),
                              (ax[2], sub_lw, "LRM wide", "RdBu_r")):
            if cm == "RdBu_r":
                v = np.nanpercentile(np.abs(arr), 98) if arr.size else 1; v = max(v, 0.3)
                a.imshow(arr, cmap=cm, vmin=-v, vmax=v)
            else:
                a.imshow(arr, cmap=cm)
            hh, ww = arr.shape
            a.plot(ww/2, hh/2, "+", color="lime", ms=14, mew=2)
            a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{c['id']} · {c['source']} · score {c['score']} · LRM {c['lrm_wide_m']} m · {THUMB_M} m",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(thumbs / f"{c['id']}.png", dpi=80, bbox_inches="tight"); plt.close(fig)
    (out / "candidates.json").write_text(json.dumps(
        {"aoi": name, "count": len(cands), "candidates": cands}, indent=2))
    print(f"    wrote {len(cands)} thumbnails → {out}")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--terrallm":
        # python build_review.py --terrallm <geojson> <aoi_name> [--include-rejects]
        gj = args[1]; name = args[2]
        inc = "--include-rejects" in args
        cands = load_terrallm(gj, include_rejects=inc)
        print(f"TerraLLM: {len(cands)} candidates from {gj}")
        build_aoi(name, candidates=cands)
    elif args == ["--all"]:
        names = sorted(p.parent.name for p in (DATA/"cnn_inference").glob("*/mound_peaks.geojson"))
        for n in names:
            try: build_aoi(n)
            except Exception: import traceback; traceback.print_exc()
    else:
        for n in args:
            try: build_aoi(n)
            except Exception: import traceback; traceback.print_exc()
