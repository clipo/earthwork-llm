#!/usr/bin/env python3
"""Figure 9: Winterville Mound A, open-terrain positive control.

Two panels (multidirectional hillshade + wide local relief) over the public
3DEP bare earth, with the published coordinate starred and a scale bar. No
caption text is baked into the image.

    python scripts/generate_winterville_fig.py
"""

import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter

from earthwork_llm.ingestion.imageserver import fetch_dem

# Winterville Mounds State Park, Mound A (22-Ws-500): published coordinate
# from the seed list (UTM 15N).
CX, CY = 680252.0, 3706813.0
HALF = 500          # meters -> 1 km window
RES = 1.0
OUT = "docs/figures/fig_winterville.png"


def hillshade(dem, az):
    gy, gx = np.gradient(dem)
    sl = np.arctan(np.hypot(gy, gx))
    asp = np.arctan2(-gx, gy)
    z = math.radians(45)
    a = math.radians(az)
    return np.clip(np.cos(z) * np.cos(sl)
                   + np.sin(z) * np.sin(sl) * np.cos(a - asp), 0, 1)


def main():
    dem = fetch_dem(CX, CY, 2 * HALF, crs_epsg=26915, resolution_m=RES)
    m = np.isfinite(dem)
    dem = np.where(m, dem, np.nanmedian(dem[m])).astype("float32")

    mhs = 0.25 * sum(hillshade(dem, a) for a in (0, 90, 180, 270))
    rel = dem - gaussian_filter(dem, 45)
    v = np.nanpercentile(np.abs(rel), 98)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.4))
    ax1.imshow(mhs, cmap="gray")
    ax1.set_title("Multidirectional hillshade", fontsize=11)
    ax2.imshow(rel, cmap="RdBu_r", vmin=-v, vmax=v)
    ax2.set_title("Local relief (wide LRM)", fontsize=11)
    for ax in (ax1, ax2):
        ax.plot(HALF, HALF, marker="*", ms=18, mfc="yellow", mec="k", mew=1.2)
        ax.set_xticks([])
        ax.set_yticks([])
    # scale bar: 200 m at 1 m/px, lower left of the first panel
    x0, y0 = 60, 2 * HALF - 60
    ax1.plot([x0, x0 + 200], [y0, y0], color="black", lw=4,
             solid_capstyle="butt")
    ax1.text(x0 + 100, y0 - 18, "200 m", ha="center", fontsize=10,
             path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    fig.tight_layout()
    os.makedirs("docs/figures", exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
