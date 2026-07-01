"""Negative control for the shell-ring detector (Section 3.7 companion).

The containment metric scores a hit when the catalog point falls within the
nearest detected ring footprint, and the detector searches a window centered on
that point. The relevant null is: how often does the identical, frozen procedure
score a "hit" at an arbitrary coastal point in the same landscape? This script
runs the frozen detector at systematically offset decoy points (the ten catalog
points displaced 500 m in each of four cardinal directions, on land), which
samples the same landform population without touching the evaluated sites, and
reports the chance strict and containment rates.
"""
from __future__ import annotations
import os, sys, csv, math, json
import numpy as np
from pathlib import Path

from shell_ring_test import detect, HALF, RES  # frozen detector, identical code path
from earthwork_llm.ingestion.imageserver import fetch_dem
from pyproj import Transformer

GOLD = os.environ.get("SHELL_RING_GOLD", "data/reference/SC_Gold_List.csv")
OUT = Path("data/shell_rings")
RIM = 15
OFFSET_M = 500
DIRS = [(OFFSET_M, 0), (-OFFSET_M, 0), (0, OFFSET_M), (0, -OFFSET_M)]


def main():
    tf = Transformer.from_crs("EPSG:4326", "EPSG:26917", always_xy=True)
    pts = []
    with open(GOLD) as f:
        for r in csv.DictReader(f):
            if r.get("mound_id"):
                x, y = tf.transform(float(r["longitude"]), float(r["latitude"]))
                for dx, dy in DIRS:
                    pts.append((r["mound_id"].strip(), dx, dy, x + dx, y + dy))
    rows = []
    for sid, dx, dy, x, y in pts:
        tag = f"{sid}+({dx:+d},{dy:+d})"
        try:
            dem = fetch_dem(x, y, 2 * HALF, crs_epsg=26917, resolution_m=RES)
            m = np.isfinite(dem)
            if m.mean() < 0.6:
                rows.append(dict(pt=tag, status="skipped_nodata")); continue
            dem = np.where(m, dem, np.nanmedian(dem[m])).astype("float32")
            # skip open-water decoys (near-zero relief windows are not comparable land)
            if float(np.nanpercentile(dem, 95) - np.nanpercentile(dem, 5)) < 0.5:
                rows.append(dict(pt=tag, status="skipped_water")); continue
            rel, b = detect(dem)
            if b is None:
                rows.append(dict(pt=tag, status="ok", ring=False, strict=False, contain=False)); continue
            off = math.hypot(b[1] - HALF, b[0] - HALF) * RES
            R = b[2]
            rows.append(dict(pt=tag, status="ok", ring=True, offset=round(off, 1), R=R,
                             strict=bool(off <= R), contain=bool(off <= R + RIM)))
            print(f"  {tag:28} off {off:6.1f} R {R:>2} strict={off<=R} contain={off<=R+RIM}", flush=True)
        except Exception as e:
            rows.append(dict(pt=tag, status=f"ERROR:{type(e).__name__}"))
            print(f"  {tag:28} ERROR {type(e).__name__}", flush=True)
    ok = [r for r in rows if r.get("status") == "ok"]
    strict = sum(1 for r in ok if r.get("strict"))
    contain = sum(1 for r in ok if r.get("contain"))
    print(f"\n===== Negative control (frozen detector, decoy points 500 m off-site) =====")
    print(f"  valid decoy points: {len(ok)} (of {len(pts)}; rest water/nodata)")
    print(f"  chance STRICT hits:      {strict}/{len(ok)}  ({100*strict/max(len(ok),1):.0f}%)")
    print(f"  chance CONTAINMENT hits: {contain}/{len(ok)}  ({100*contain/max(len(ok),1):.0f}%)")
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "ring_negative_control.csv", index=False)
    (OUT / "ring_negative_control_summary.json").write_text(json.dumps(dict(
        n_valid=len(ok), strict=strict, contain=contain), indent=1))
    print("wrote", OUT / "ring_negative_control.csv")


if __name__ == "__main__":
    main()
