"""Figure A1: the multi-view ensemble (Section 2.5.1, Appendix A).

Renders the six-panel composite the interpretation layer receives, four
hillshades at different azimuths, the geomorphon classification, and contours,
for a public monument (Winterville Mound A by default, from data/reference/
published_sites.csv). Public data only.

Usage: python scripts/generate_multiview_fig.py [--site winterville_mound_a]
Writes: docs/figures/fig2_multiview_ensemble.png
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from earthwork_llm.ingestion.imageserver import fetch_dem
from demo_terrain_query import classify_geomorphon_simple, make_multi_view_panel

ROOT = Path(__file__).resolve().parent.parent
DETAIL_PX = 160  # the deployed 160 m detail window


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="winterville_mound_a")
    args = ap.parse_args()
    row = next(r for r in csv.DictReader(open(ROOT / "data/reference/published_sites.csv"))
               if r["site_id"] == args.site)
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:26915", always_xy=True).transform(
        float(row["longitude"]), float(row["latitude"]))
    dem = fetch_dem(x, y, DETAIL_PX, crs_epsg=26915, resolution_m=1.0).astype("float64")
    dem[np.isnan(dem)] = np.nanmedian(dem)
    img = make_multi_view_panel(dem, classify_geomorphon_simple(dem))
    out = ROOT / "docs/figures/fig2_multiview_ensemble.png"
    img.save(out)
    print("wrote", out, f"({row['site_name']})")


if __name__ == "__main__":
    main()
