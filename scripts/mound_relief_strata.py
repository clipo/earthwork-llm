"""Relief-stratified recall for B.1: local relief at each corrected mound
center, joined to the Table 1 recall run, reported by relief class.

Env: EARTHWORK_GOLD_LIST (corrected centers). Reads data/refind_utm/refind_utm.csv.
Writes data/refind_utm/mound_relief_strata.csv and prints the stratified table.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "/home/clipo/projects/earthwork-llm/src")
from earthwork_llm.ingestion.imageserver import fetch_dem

GOLD = os.environ.get(
    "EARTHWORK_GOLD_LIST",
    "/home/clipo/projects/yazoo/data/gold_correct/located_mounds_corrected.csv")
HALF = 150


def local_relief(cx, cy):
    dem = fetch_dem(cx, cy, 2 * HALF, crs_epsg=26915, resolution_m=1.0)
    m = np.isfinite(dem)
    dem = np.where(m, dem, np.nanmedian(dem[m])).astype("float32")
    rel = dem - gaussian_filter(dem, 45)
    yy, xx = np.mgrid[0:2 * HALF, 0:2 * HALF]
    near = (yy - HALF) ** 2 + (xx - HALF) ** 2 <= 40 ** 2
    return float(np.nanmax(rel[near]))


def main():
    gold = pd.read_csv(GOLD)
    res = pd.read_csv("/home/clipo/projects/terrallm/data/refind_utm/refind_utm.csv")
    res["dist_m"] = pd.to_numeric(res["dist_m"], errors="coerce")
    hit = {r.mound_id: (r.status == "ok" and r.dist_m <= 30) for r in res.itertuples()}
    rows = []
    for _, g in gold.iterrows():
        mid = g["mound_id"]
        try:
            rel = local_relief(float(g["utm15n_easting_m"]), float(g["utm15n_northing_m"]))
        except Exception as e:
            print(f"  {mid}: ERR {type(e).__name__}", flush=True)
            continue
        rows.append(dict(mound_id=mid, relief_m=round(rel, 2), recovered=bool(hit.get(mid, False))))
        print(f"  {mid}: relief {rel:.2f} m, recovered={hit.get(mid, False)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("/home/clipo/projects/terrallm/data/refind_utm/mound_relief_strata.csv", index=False)
    print("\n===== recall by local-relief class (30 m tolerance) =====")
    for lo, hi, lab in [(0, 0.5, "<0.5 m"), (0.5, 1.5, "0.5-1.5 m"), (1.5, 99, ">1.5 m")]:
        sub = df[(df.relief_m >= lo) & (df.relief_m < hi)]
        if len(sub):
            print(f"  {lab:10s}: {int(sub.recovered.sum())}/{len(sub)}")


if __name__ == "__main__":
    main()
