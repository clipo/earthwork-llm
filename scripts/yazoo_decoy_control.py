"""Decoy-point control for the Table 1 recall protocol (Section 3.1, B.1).

Applies the standard the paper sets in Section 3.7 to the flagship Yazoo
recall: run the IDENTICAL frozen matching protocol (300 m window centered on
the point, single-scale geomorphon detector, up to ten candidates, hit if any
candidate centroid falls within the tolerance of window center) at decoy
points 500 m from each corrected mound center, and report the hit rate by
tolerance. If the mound recall is real, it must beat this decoy floor.

Env: EARTHWORK_GOLD_LIST (corrected centers CSV with utm15n_easting_m /
utm15n_northing_m). Output: data/refind_utm/yazoo_decoy_control.csv + summary.
"""
from __future__ import annotations
import os, sys, math, json
import numpy as np, pandas as pd

sys.path.insert(0, "/home/clipo/projects/terrallm")
sys.path.insert(0, "/home/clipo/projects/terrallm/scripts")
sys.path.insert(0, "/home/clipo/projects/earthwork-llm/src")
from demo_terrain_query import classify_geomorphon_simple
from scripts.earthwork_query import detect_earthworks
from earthwork_llm.ingestion.imageserver import fetch_dem

GOLD = os.environ.get(
    "EARTHWORK_GOLD_LIST",
    "/home/clipo/projects/yazoo/data/gold_correct/located_mounds_corrected.csv")
HALF = 150
RES = 1.0
TOLS = (10, 15, 20, 25, 30)
OFFS = ((500, 0), (-500, 0), (0, 500), (0, -500))


def probe(cx, cy):
    dem = fetch_dem(cx, cy, 2 * HALF, crs_epsg=26915, resolution_m=RES)
    m = np.isfinite(dem)
    if m.mean() < 0.6:
        return None
    dem = np.where(m, dem, np.nanmedian(dem[m])).astype("float32")
    if float(np.nanpercentile(dem, 95) - np.nanpercentile(dem, 5)) < 0.2:
        return None  # open water / featureless nodata fill
    geo = classify_geomorphon_simple(dem)
    cands = detect_earthworks(geo, dem, "Find pre-European earthwork mounds")[:10]
    if not cands:
        return dict(best=None)
    d = min(math.hypot(c["x"] - HALF, c["y"] - HALF) for c in cands)
    return dict(best=d * RES)


def main():
    gold = pd.read_csv(GOLD)
    rows = []
    for _, g in gold.iterrows():
        mid = g["mound_id"]
        cx, cy = float(g["utm15n_easting_m"]), float(g["utm15n_northing_m"])
        for dx, dy in OFFS:
            tag = f"{mid}+({dx:+d},{dy:+d})"
            try:
                r = probe(cx + dx, cy + dy)
            except Exception as e:
                print(f"  {tag}: ERR {type(e).__name__}", flush=True)
                continue
            if r is None:
                rows.append(dict(pt=tag, status="skipped"))
                continue
            rec = dict(pt=tag, status="ok", best_m=r["best"])
            rows.append(rec)
            print(f"  {tag}: best {r['best'] if r['best'] is not None else 'none'}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("data/refind_utm/yazoo_decoy_control.csv", index=False)
    ok = df[df.status == "ok"]
    summary = {}
    for t in TOLS:
        hits = int((ok.best_m.notna() & (ok.best_m <= t)).sum())
        summary[f"hit_at_{t}m"] = f"{hits}/{len(ok)} ({hits/len(ok)*100:.0f}%)"
    print("\n===== Yazoo decoy control (frozen Table 1 protocol at 500 m offsets) =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    with open("data/refind_utm/yazoo_decoy_summary.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
