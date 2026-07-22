"""Figure 8: Lake George Mound A, a canopy negative (Section 3.4).

Renders the corrected Lake George area of interest from the public 3DEP
ImageServer: multidirectional hillshade and wide local relief centered on the
published Mound A coordinate (Williams & Brain 1983; data/reference/
published_sites.csv). The figure shows the clean negative of Section 3.4: no
17 m platform is present in the seamless bare-earth product under closed
canopy. Public data only; no restricted inputs.

Usage: python scripts/generate_lake_george_fig.py
Writes: docs/figures/fig_lake_george_corrected.png
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from earthwork_llm.ingestion.imageserver import fetch_dem
from demo_terrain_query import make_hillshade

ROOT = Path(__file__).resolve().parent.parent
WIN_M = 1900          # matches the Section 3.4 area of interest (1.9 km)
RES_M = 2.0           # 2 m render keeps the request modest; relief is ~2 m total


def published_coordinate(site_id: str) -> tuple[float, float]:
    for r in csv.DictReader(open(ROOT / "data/reference/published_sites.csv")):
        if r["site_id"] == site_id:
            return float(r["latitude"]), float(r["longitude"])
    raise KeyError(site_id)


def main() -> None:
    lat, lon = published_coordinate("lake_george_mound_a")
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:26915", always_xy=True).transform(lon, lat)
    px = int(WIN_M / RES_M)
    dem = fetch_dem(x, y, px, crs_epsg=26915, resolution_m=RES_M).astype("float64")
    dem[np.isnan(dem)] = np.nanmedian(dem)

    hills = [make_hillshade(dem, azdeg=az, altdeg=45) for az in (45, 135, 225, 315)]
    hs = np.mean(hills, axis=0)
    rel = dem - gaussian_filter(dem, 45)
    v = np.nanpercentile(np.abs(rel), 98) or 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.4), dpi=200)
    ax1.imshow(hs, cmap="gray")
    ax1.set_title("(a) Multidirectional hillshade", fontsize=11, loc="left")
    ax2.imshow(rel, cmap="RdBu_r", vmin=-v, vmax=v)
    ax2.set_title("(b) Wide local relief", fontsize=11, loc="left")
    for ax in (ax1, ax2):
        ax.plot(px / 2, px / 2, marker="*", ms=16, mfc="none", mec="cyan", mew=2)
        ax.set_xticks([]), ax.set_yticks([])
        # 500 m scale bar
        bar = 500 / RES_M
        ax.plot([px * 0.05, px * 0.05 + bar], [px * 0.95] * 2, color="black", lw=3)
        ax.text(px * 0.05 + bar / 2, px * 0.93, "500 m", ha="center", fontsize=9)
    fig.suptitle("Lake George / Holly Bluff (22-Yz-557), published Mound A coordinate at center",
                 fontsize=11)
    fig.tight_layout()
    out = ROOT / "docs/figures/fig_lake_george_corrected.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
