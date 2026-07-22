"""Simple LRM blob-detector baseline for the 35-mound recall protocol (B.1).

How much of the mound recall is attributable to the geomorphon representation?
Baseline: local relief model (DEM minus 40 m Gaussian), top-10 local maxima as
candidates, same 300 m centered window, same nearest-of-ten scoring.
"""
import os
import math
import csv
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from earthwork_llm.ingestion.imageserver import fetch_dem

GOLD = os.environ.get("EARTHWORK_GOLD_LIST", "data/reference/located_mounds.csv")
HALF = 150
rows = list(csv.DictReader(open(GOLD)))
dists = []
for r in rows:
    x, y = float(r["utm15n_easting_m"]), float(r["utm15n_northing_m"])
    dem = fetch_dem(x, y, 2 * HALF, crs_epsg=26915, resolution_m=1.0)
    m = np.isfinite(dem)
    dem = np.where(m, dem, np.nanmedian(dem[m])).astype("float32")
    rel = dem - gaussian_filter(dem, 40)
    mx = maximum_filter(rel, size=21)
    peaks = np.argwhere((rel == mx) & (rel > 0.1))
    peaks = sorted(peaks, key=lambda p: -rel[p[0], p[1]])[:10]   # top-10 candidates
    d = min((math.hypot(p[1] - HALF, p[0] - HALF) for p in peaks), default=1e9)
    dists.append(d)
    print(f"  {r['mound_id']:14} nearest LRM peak {d:6.1f} m", flush=True)
d = np.array(dists)
print("\n===== LRM blob baseline (same protocol as B.1) =====")
for tol in (10, 15, 20, 25, 30):
    print(f"  {tol:>3} m: {int((d<=tol).sum())}/{len(d)}")
rec = d[d <= 30]
print(f"  median offset of recovered: {np.median(rec):.1f} m")
import json  # noqa: E402  (deliberate late import: dump only runs after the summary prints)
json.dump({str(t): int((d<=t).sum()) for t in (10,15,20,25,30)},
    open("data/refind_utm/lrm_baseline.json","w"), indent=1)
