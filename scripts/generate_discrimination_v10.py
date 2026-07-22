"""V10 discrimination drills: universal age/authenticity training data.

The V9.1 ablations showed the model reads earthwork morphology but accepts
modern earthworks and natural rises; no prompt or voting scheme fixes it
(arms A-C). This generator builds the missing training signal, with no
target-region data:

  NEGATIVES  compact rises harvested beside OpenStreetMap-mapped linear
             features in upstate New York (canal spoil, road fill, rail
             embankment, ditch cleanout) -- real modern earthworks with
             mechanically derived labels.
  POSITIVES  synthetic eroded mounds (conical and platform, 15-60 m across,
             0.5-5 m high, smoothed flanks) injected into real New York
             terrain far from mapped features -- perfect labels, universal
             morphology.

Each example is ONE composite image (6-panel 160 m detail on top, 600 m wide
context view below -- the trainer takes one image per sample) plus the rubric
prompt used at evaluation, and a rubric-formatted response ending in
VERDICT / CONFIDENCE. Output: base64-inline JSONL for convert_jsonl_to_lf.py.
"""
from __future__ import annotations
import sys
import io
import json
import math
import time
import base64
import argparse
import random
import numpy as np
import requests
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, "scripts")
from demo_terrain_query import classify_geomorphon_simple, make_multi_view_panel
from earthwork_query import detect_earthworks
from earthwork_llm.ingestion.imageserver import fetch_dem
from scipy.ndimage import gaussian_filter
from pyproj import Transformer

EPSG = 26918          # UTM 18N, upstate New York
DETAIL_PX = 160
WIDE_M = 600
OUT = Path("data/v10_discrimination")
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]
HEADERS = {"User-Agent": "earthwork-llm-research/1.0 (training data; clipo@binghamton.edu)"}

# Upstate NY harvest boxes (lon/lat): Erie Canal corridor + rural drainage
BOXES = [
    (-75.65, 43.15, -75.05, 43.35),   # Rome / Oneida, canal + rail
    (-76.95, 42.95, -76.45, 43.15),   # Montezuma / Clyde, canal + drains
    (-78.15, 43.05, -77.55, 43.25),   # Brockport / Genesee canal stretch
    (-77.20, 42.85, -76.70, 43.00),   # Seneca River / rural roads
]

PROMPT = (
    "Analyse the terrain.\n\n"
    "You are given ONE composite image of a location. The TOP block is a 6-panel "
    "detail view (160 m across: 4 hillshades | geomorphon | contours) centred on a "
    "candidate feature. The BOTTOM block is a WIDE context view (600 m across: "
    "hillshade left, local relief right; candidate circled).\n"
    "Before deciding, answer these three checks in one line each:\n"
    "CHECK-SHAPE: is the central feature a compact, roughly symmetric positive rise? (yes/no + why)\n"
    "CHECK-CONTEXT: in the wide view, does it sit on or beside a linear modern feature "
    "(canal, ditch, drain, road, rail, levee) or a field edge whose construction or "
    "maintenance would produce a spoil pile of this size? (yes/no + why)\n"
    "CHECK-AGE: does its form suggest long erosion (softened, spreading flanks) or recent "
    "machine work (sharp, uniform, aligned with modern features)? (eroded/recent + why)\n"
    "A pre-European mound is compact and symmetric, stands apart from modern linear works, "
    "and looks eroded. A modern spoil pile or berm is beside the linear feature that produced "
    "it and looks recent. End with exactly two lines:\n"
    "VERDICT: MOUND        (pre-European mound)\n"
    "or VERDICT: NOT_MOUND (natural or modern)\n"
    "CONFIDENCE: <integer 0-100>"
)


def clean(dem):
    m = np.isfinite(dem)
    if not m.any():
        raise RuntimeError("empty dem")
    return np.where(m, dem, np.nanmedian(dem[m])).astype("float32"), float(m.mean())


def hillshade(dem, az=315):
    gy, gx = np.gradient(dem)
    sl = np.arctan(np.hypot(gy, gx))
    asp = np.arctan2(-gx, gy)
    z = math.radians(45)
    a = math.radians(az)
    return np.clip(np.cos(z) * np.cos(sl) + np.sin(z) * np.sin(sl) * np.cos(a - asp), 0, 1)


def wide_view(x, y, inject=None):
    dem = fetch_dem(x, y, WIDE_M, crs_epsg=EPSG, resolution_m=1.0)
    dem, frac = clean(dem)
    if frac < 0.7:
        raise RuntimeError("nodata")
    if inject is not None:
        dem = add_mound(dem, WIDE_M // 2, WIDE_M // 2, **inject)
    hs8 = (hillshade(dem) * 255).astype(np.uint8)
    rel = dem - gaussian_filter(dem, 45)
    v = np.nanpercentile(np.abs(rel), 98) or 1.0
    import matplotlib.cm as cm
    rel_rgb = (cm.RdBu_r(np.clip((rel / v + 1) / 2, 0, 1))[:, :, :3] * 255).astype(np.uint8)
    combo = np.concatenate([np.stack([hs8] * 3, axis=-1), rel_rgb], axis=1)
    img = Image.fromarray(combo)
    d = ImageDraw.Draw(img)
    for cx in (WIDE_M // 2, WIDE_M + WIDE_M // 2):
        d.ellipse([cx - 12, WIDE_M // 2 - 12, cx + 12, WIDE_M // 2 + 12],
                  outline=(255, 255, 0), width=3)
    return img


def add_mound(dem, cy, cx, diam=30.0, height=2.0, ellip=1.0, theta=0.0, platform=False, erode=2.5):
    """Inject a synthetic eroded mound into a DEM (in place on a copy)."""
    d = dem.copy()
    H, W = d.shape
    yy, xx = np.mgrid[0:H, 0:W].astype("float32")
    dy, dx = yy - cy, xx - cx
    ct, st = math.cos(theta), math.sin(theta)
    u = (ct * dx + st * dy) / (diam / 2 * ellip)
    v = (-st * dx + ct * dy) / (diam / 2)
    r = np.sqrt(u * u + v * v)
    prof = np.clip(np.cos(np.clip(r, 0, 1) * math.pi / 2), 0, 1) ** 1.5
    if platform:
        prof = np.clip(prof * 1.4, 0, 1)     # flat top
    bump = (height * prof).astype("float32")
    bump = gaussian_filter(bump, erode)       # erosion softening
    return d + bump


def detail_view(x, y, inject=None):
    dem = fetch_dem(x, y, DETAIL_PX, crs_epsg=EPSG, resolution_m=1.0)
    dem, frac = clean(dem)
    if frac < 0.7:
        raise RuntimeError("nodata")
    if inject is not None:
        dem = add_mound(dem, DETAIL_PX // 2, DETAIL_PX // 2, **inject)
    geo = classify_geomorphon_simple(dem)
    return make_multi_view_panel(dem, geo), dem, geo


def composite(detail_img, wide_img):
    w = max(detail_img.width, wide_img.width)
    scale_d = w / detail_img.width
    d2 = detail_img.resize((w, int(detail_img.height * scale_d)))
    scale_w = w / wide_img.width
    w2 = wide_img.resize((w, int(wide_img.height * scale_w)))
    out = Image.new("RGB", (w, d2.height + w2.height + 6), (20, 20, 24))
    out.paste(d2, (0, 0))
    out.paste(w2, (0, d2.height + 6))
    if out.width > 1100:
        s = 1100 / out.width
        out = out.resize((1100, int(out.height * s)))
    return out


def b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def overpass_ways(box):
    q = ('[out:json][timeout:60];('
         'way({s},{w},{n},{e})["waterway"~"canal|ditch|drain"];'
         'way({s},{w},{n},{e})["railway"~"rail|abandoned"];'
         'way({s},{w},{n},{e})["highway"~"secondary|tertiary|unclassified"];'
         ');out geom;').format(w=box[0], s=box[1], e=box[2], n=box[3])
    for i, url in enumerate(OVERPASS * 2):
        try:
            r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=90)
            r.raise_for_status()
            return r.json().get("elements", [])
        except Exception:
            time.sleep(8 * (i + 1))
    return []


def feature_label(tags):
    return (tags.get("waterway") or tags.get("railway") or
            ("road" if "highway" in tags else "feature"))


NEG_TMPL = ("CHECK-SHAPE: yes - there is a compact positive rise at the centre, but its form is "
            "uniform and bank-like.\n"
            "CHECK-CONTEXT: yes - the wide view shows it lying directly along a {feat} about "
            "{dist:.0f} m away; piles of this size are exactly what construction and periodic "
            "cleanout of such a feature leave behind.\n"
            "CHECK-AGE: recent - the flanks are sharp and the material is aligned with the "
            "adjacent {feat} rather than spreading evenly.\n"
            "VERDICT: NOT_MOUND\nCONFIDENCE: {conf}")

POS_TMPL = ("CHECK-SHAPE: yes - a compact, roughly symmetric rise about {diam:.0f} m across with "
            "{top} form.\n"
            "CHECK-CONTEXT: no - in the wide view the rise stands apart from any linear modern "
            "feature; there is no adjacent canal, ditch, road, or rail that would account for it "
            "as spoil.\n"
            "CHECK-AGE: eroded - the flanks are softened and spread smoothly into the surrounding "
            "surface, consistent with long weathering.\n"
            "VERDICT: MOUND\nCONFIDENCE: {conf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-neg", type=int, default=700)
    ap.add_argument("--n-pos", type=int, default=700)
    ap.add_argument("--out", default=str(OUT / "v10_discrimination.jsonl"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(101)
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)

    # ---- harvest feature geometries once per box ----
    feats = []   # (utm_x, utm_y, label) points along features
    for box in BOXES:
        els = overpass_ways(box)
        for el in els:
            lab = feature_label(el.get("tags", {}))
            geom = el.get("geometry", [])
            for g in geom[::4]:
                x, y = tf.transform(g["lon"], g["lat"])
                feats.append((x, y, lab))
        print(f"box {box}: cumulative feature points {len(feats)}", flush=True)
    if len(feats) < 500:
        raise SystemExit("too few OSM feature points harvested")
    fx = np.array([f[0] for f in feats])
    fy = np.array([f[1] for f in feats])

    fh = open(args.out, "w")

    # ---- negatives: compact rises beside features ----
    n_done, attempts = 0, 0
    while n_done < args.n_neg and attempts < args.n_neg * 12:
        attempts += 1
        i = rng.randrange(len(feats))
        x0, y0, lab = feats[i]
        ang = rng.uniform(0, 2 * math.pi)
        off = rng.uniform(8, 35)
        x, y = x0 + off * math.cos(ang), y0 + off * math.sin(ang)
        try:
            detail, dem, geo = detail_view(x, y)
            cands = detect_earthworks(geo, dem, "Find pre-European earthwork mounds")
            half = DETAIL_PX // 2
            near = [c for c in cands
                    if math.hypot(c["x"] - half, c["y"] - half) < 45
                    and 30 <= c.get("area", c.get("area_m2", 0)) <= 6000]
            if not near:
                continue
            wide = wide_view(x, y)
            comp = composite(detail, wide)
            dist = float(np.min(np.hypot(fx - x, fy - y)))
            resp = NEG_TMPL.format(feat=lab, dist=max(dist, 5), conf=rng.randint(82, 95))
            rec = {"messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": b64(comp)},
                    {"type": "text", "text": PROMPT}]},
                {"role": "assistant", "content": resp}],
                "sample_type": "discrimination_negative", "feature": lab}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_done += 1
            if n_done % 25 == 0:
                print(f"  negatives {n_done}/{args.n_neg} (attempts {attempts})", flush=True)
        except Exception:
            continue
    print(f"negatives done: {n_done}", flush=True)

    # ---- positives: synthetic mounds away from features ----
    p_done, attempts = 0, 0
    while p_done < args.n_pos and attempts < args.n_pos * 12:
        attempts += 1
        box = BOXES[rng.randrange(len(BOXES))]
        lon = rng.uniform(box[0], box[2])
        lat = rng.uniform(box[1], box[3])
        x, y = tf.transform(lon, lat)
        if float(np.min(np.hypot(fx - x, fy - y))) < 200:
            continue
        platform = rng.random() < 0.35
        inj = dict(diam=rng.uniform(15, 60), height=rng.uniform(0.5, 5.0),
                   ellip=rng.uniform(1.0, 1.5), theta=rng.uniform(0, math.pi),
                   platform=platform, erode=rng.uniform(1.5, 4.0))
        try:
            detail, dem, geo = detail_view(x, y, inject=inj)
            wide = wide_view(x, y, inject=inj)
            comp = composite(detail, wide)
            resp = POS_TMPL.format(diam=inj["diam"],
                                   top=("a flat-topped platform" if platform else "a rounded conical"),
                                   conf=rng.randint(80, 94))
            rec = {"messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": b64(comp)},
                    {"type": "text", "text": PROMPT}]},
                {"role": "assistant", "content": resp}],
                "sample_type": "discrimination_positive"}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            p_done += 1
            if p_done % 25 == 0:
                print(f"  positives {p_done}/{args.n_pos} (attempts {attempts})", flush=True)
        except Exception:
            continue
    fh.close()
    print(f"positives done: {p_done}")
    print(f"wrote {args.out}: {n_done} negatives + {p_done} positives")


if __name__ == "__main__":
    main()
