"""Recompute gold-list recall in UTM Zone 15N (true meters), with a recall-vs-
tolerance curve — the CRS correction requested in peer review.

Fetches each gold mound's DEM from the USGS 3DEP ImageServer in EPSG:26915
(UTM15N, true meters), runs the SAME single-scale geomorphon detector used
elsewhere (classify_geomorphon_simple + detect_earthworks), and matches the
nearest candidate to the catalogue point in true meters.
"""
from __future__ import annotations
import os
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from earthwork_llm.ingestion.imageserver import Usgs3depImageServerSource, WindowRequest
from demo_terrain_query import classify_geomorphon_simple
from earthwork_query import detect_earthworks

UTM = "EPSG:26915"
RES = 1.0
CROP = 150  # 150 m half-window -> 300 m tile (matches original crop_m=300)
GOLD = os.environ.get("EARTHWORK_GOLD_LIST", "data/reference/located_mounds.csv")  # RESTRICTED reference set, NOT shipped with this repo (see README, Data); set EARTHWORK_GOLD_LIST to reproduce Table 1
TOLS = [10, 15, 20, 25, 30]


def fetch_utm(cx, cy, half_m):
    src = Usgs3depImageServerSource()
    size = int(2 * half_m)
    # single request (<=1024 ok for 600 px); center on mound
    req = WindowRequest(center_x=cx, center_y=cy, utm_crs=UTM, resolution_m=RES, size_px=size)
    arr = src.fetch_window(req)  # returns (size,size), origin top-left = (cx-half, cy+half)
    return arr, cx - half_m, cy + half_m  # dem, xmin, ymax


def main():
    df = pd.read_csv(GOLD)
    rows = []
    for _, r in df.iterrows():
        mid = str(r["mound_id"])
        cx = float(r["utm15n_easting_m"])
        cy = float(r["utm15n_northing_m"])
        try:
            dem, xmin, ymax = fetch_utm(cx, cy, CROP)
        except Exception as e:
            rows.append(dict(mound_id=mid, status=f"fail:{type(e).__name__}"))
            continue
        dem = np.where(np.isfinite(dem), dem, np.nanmedian(dem[np.isfinite(dem)]) if np.isfinite(dem).any() else 0.0).astype("float32")
        geo = classify_geomorphon_simple(dem)
        cands = detect_earthworks(geo, dem, "Find pre-European earthwork mounds")
        # mound pixel: col = cx - xmin, row = ymax - cy  (y down)
        gpx = cx - xmin
        gpy = ymax - cy
        best = None
        for c in cands:
            d = math.hypot(c["x"] - gpx, c["y"] - gpy) * RES  # true meters (UTM 1 m grid)
            if best is None or d < best:
                best = d
        rows.append(dict(mound_id=mid, status="ok", n=len(cands),
                         dist_m=round(best, 2) if best is not None else None))
        print(f"{mid:12} nearest={best if best is None else round(best,1)} m  cands={len(cands)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("data/refind_utm/refind_utm.csv", index=False) if Path("data/refind_utm").exists() or Path("data/refind_utm").mkdir(parents=True) or True else None
    ok = out[out["status"] == "ok"].copy()
    ok["dist_m"] = pd.to_numeric(ok["dist_m"], errors="coerce")
    print(f"\n=== UTM recompute: {len(ok)}/{len(out)} tiles processed ===")
    for t in TOLS:
        hit = (ok["dist_m"] <= t).sum()
        print(f"  tol {t:>2} m: recall {hit}/{len(ok)} = {100*hit/len(ok):.1f}%")
    hits = ok[ok["dist_m"] <= 30]["dist_m"]
    print(f"  offsets (<=30 m): median {hits.median():.2f} m, mean {hits.mean():.2f} m, max {hits.max():.2f} m, n={len(hits)}")


if __name__ == "__main__":
    Path("data/refind_utm").mkdir(parents=True, exist_ok=True)
    main()
