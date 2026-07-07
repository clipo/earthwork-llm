"""Model-only detection arm: can the vision-language model localize mounds?

Tests the model as an identification tool under the frozen Table 1 protocol:
a 300 m window centered on each hand-corrected mound center, up to ten
candidates per window, nearest-candidate distance to window center, scored at
10/15/20/25/30 m tolerances, with the identical procedure at the 135 viable
decoy windows of Appendix B.1 (same decoy set as Table 1).

The model receives the window as the landscape text it was trained to read:
a two-panel image (multidirectional hillshade | geomorphon classes), north up,
and returns pixel coordinates of candidate mounds. No detector, no shield.

Usage:
  python scripts/vlm_detection_arm.py --set mounds --out data/v10_eval/vlm_detect_mounds.csv
  python scripts/vlm_detection_arm.py --set decoys --out data/v10_eval/vlm_detect_decoys.csv

Env: VLM_API (default http://localhost:8001/v1/chat/completions),
     VLM_MODEL (default terrallm-v91), EARTHWORK_GOLD_LIST.
"""
from __future__ import annotations
import os, re, csv, io, sys, json, time, argparse
import numpy as np
import requests
from PIL import Image, ImageDraw

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from demo_terrain_query import classify_geomorphon_simple, make_hillshade, pil_to_b64_data_uri
from earthwork_llm.ingestion.imageserver import fetch_dem

API = os.environ.get("VLM_API", "http://localhost:8001/v1/chat/completions")
MODEL = os.environ.get("VLM_MODEL", "terrallm-v91")
GOLD = os.environ["EARTHWORK_GOLD_LIST"]  # restricted 35-mound reference set (see README data policy)
DECOY_CSV = os.environ.get("DECOY_CONTROL_CSV", "data/refind_utm/yazoo_decoy_control.csv")

WIN_M = 300          # window size in meters (Table 1 protocol)
SCALE = 2            # rendered pixels per meter
IMG_PX = WIN_M * SCALE

# tab10 palette (matplotlib), index -> (r, g, b), for the 10 geomorphon classes
TAB10 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
         (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
         (188, 189, 34), (23, 190, 207)]
CLASSES = ["FLAT", "PEAK", "RIDGE", "SHOULDER", "SPUR",
           "SLOPE", "HOLLOW", "FOOTSLOPE", "VALLEY", "PIT"]
LEGEND = ", ".join(f"{name}={col}" for name, col in zip(
    CLASSES, ["blue", "orange", "green", "red", "purple",
              "brown", "pink", "gray", "olive", "cyan"]))

PROMPT = f"""You are reading a bare-earth LiDAR terrain window rendered in the representation you were trained on. The image shows the same 300 m x 300 m area twice, north at the top. Left panel: multidirectional hillshade. Right panel: geomorphon landform classes ({LEGEND}). Each panel is {IMG_PX} x {IMG_PX} pixels, so 1 pixel = 0.5 m.

Task: identify every location that could be a pre-European earthen mound, a compact raised earthwork, conical or platform, roughly 10 to 100 m across. Use the geomorphon structure: a mound reads as a compact cluster of PEAK, RIDGE, SHOULDER, or SPUR cells surrounded by FLAT or SLOPE, not a long linear band.

Report up to ten candidate locations, most confident first, one per line, as pixel coordinates within a single panel (x rightward 0-{IMG_PX - 1}, y downward 0-{IMG_PX - 1}), in exactly this format:
CANDIDATE: x=<pixels>, y=<pixels>
If no location qualifies, output exactly: NONE"""


def clean(dem: np.ndarray) -> np.ndarray:
    d = dem.astype("float64")
    if np.isnan(d).all():
        raise ValueError("empty DEM")
    d[np.isnan(d)] = np.nanmedian(d)
    return d


def render(dem: np.ndarray) -> Image.Image:
    """Two-panel (hillshade | geomorphon) north-up image, SCALE px per meter."""
    hills = [make_hillshade(dem, azdeg=az, altdeg=45) for az in (45, 135, 225, 315)]
    hs = np.mean(hills, axis=0)
    hs = (255 * (hs - hs.min()) / max(float(np.ptp(hs)), 1e-9)).astype("uint8")
    left = Image.fromarray(hs).convert("RGB")
    geo = classify_geomorphon_simple(dem)
    pal = np.array(TAB10, dtype="uint8")
    right = Image.fromarray(pal[np.clip(geo, 0, 9)])
    left = left.resize((IMG_PX, IMG_PX), Image.NEAREST)
    right = right.resize((IMG_PX, IMG_PX), Image.NEAREST)
    img = Image.new("RGB", (IMG_PX * 2 + 8, IMG_PX), "white")
    img.paste(left, (0, 0))
    img.paste(right, (IMG_PX + 8, 0))
    return img


def query(img: Image.Image) -> str:
    payload = {"model": MODEL, "messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": pil_to_b64_data_uri(img)}},
            {"type": "text", "text": PROMPT}]}],
        "max_tokens": 4096, "temperature": 0.0}
    r = requests.post(API, json=payload, timeout=900)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse(txt: str) -> list[tuple[float, float]]:
    """Return list of (px_x, px_y). Reads only the text after any thinking trace."""
    tail = txt.split("</think>")[-1]
    out = []
    for m in re.finditer(r"CANDIDATE:\s*x\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*y\s*=\s*(-?\d+(?:\.\d+)?)", tail, re.I):
        out.append((float(m.group(1)), float(m.group(2))))
    return out[:10]


def px_to_offset_m(px_x: float, px_y: float) -> tuple[float, float]:
    """Pixel -> (dx east, dy north) meters from window center. Row 0 = north."""
    # strip the right panel back onto the shared grid
    if px_x >= IMG_PX + 8:
        px_x -= IMG_PX + 8
    dx = (px_x + 0.5) / SCALE - WIN_M / 2.0
    dy = WIN_M / 2.0 - (px_y + 0.5) / SCALE
    return dx, dy


def gold_sites() -> list[tuple[str, float, float]]:
    rows = list(csv.DictReader(open(GOLD)))
    return [(r["mound_id"], float(r["utm15n_easting_m"]), float(r["utm15n_northing_m"]))
            for r in rows if r["flag"] == "visible"]


def decoy_sites() -> list[tuple[str, float, float]]:
    centers = {sid: (x, y) for sid, x, y in gold_sites()}
    # the not-visible mound is also a decoy base in the released control; include all
    for r in csv.DictReader(open(GOLD)):
        centers.setdefault(r["mound_id"], (float(r["utm15n_easting_m"]),
                                           float(r["utm15n_northing_m"])))
    out = []
    for r in csv.DictReader(open(DECOY_CSV)):
        if r["status"] != "ok":
            continue
        m = re.match(r"^(.*)\+\(([+-]?\d+),([+-]?\d+)\)$", r["pt"])
        base, dx, dy = m.group(1), int(m.group(2)), int(m.group(3))
        x0, y0 = centers[base]
        out.append((r["pt"], x0 + dx, y0 + dy))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["mounds", "decoys"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-raw", default="")
    args = ap.parse_args()

    sites = gold_sites() if args.set == "mounds" else decoy_sites()
    if args.limit:
        sites = sites[:args.limit]
    rows = []
    for i, (sid, x, y) in enumerate(sites, 1):
        t0 = time.time()
        try:
            dem = clean(fetch_dem(x, y, WIN_M, crs_epsg=26915, resolution_m=1.0))
            txt = query(render(dem))
            cands = parse(txt)
            if args.save_raw:
                with open(args.save_raw, "a") as f:
                    f.write(json.dumps({"id": sid, "text": txt}) + "\n")
            best = None
            cand_m = []
            for px_x, px_y in cands:
                dx, dy = px_to_offset_m(px_x, px_y)
                d = (dx * dx + dy * dy) ** 0.5
                cand_m.append(round(d, 2))
                best = d if best is None or d < best else best
            rows.append(dict(id=sid, n=len(cands),
                             best_m=(round(best, 2) if best is not None else ""),
                             dists=";".join(map(str, cand_m))))
            print(f"[{i}/{len(sites)}] {sid}: n={len(cands)} best="
                  f"{rows[-1]['best_m']} {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            rows.append(dict(id=sid, n=-1, best_m="", dists=f"ERR {type(e).__name__}"))
            print(f"[{i}/{len(sites)}] {sid}: ERR {type(e).__name__}: {e}", flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "n", "best_m", "dists"])
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["n"] >= 0]
    print(f"\nwrote {args.out}  ({len(ok)}/{len(rows)} scored)")
    for tol in (10, 15, 20, 25, 30):
        hits = sum(1 for r in ok if r["best_m"] != "" and float(r["best_m"]) <= tol)
        print(f"  <= {tol} m: {hits}/{len(ok)} ({100*hits/max(len(ok),1):.0f}%)")


if __name__ == "__main__":
    main()
