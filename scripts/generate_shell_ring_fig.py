"""Generate the shell-ring manuscript figure (signature + recovery gallery).

The location map now lives in Figure 1 (generate_paper_map.py). Panel (a):
the ring structural signature at 38BU0300 in hillshade, local relief, and
geomorphons. (b): local relief at five cataloged rings, corrected center
marked.

No caption text is baked into the image; captions belong to the manuscript.
Site coordinates come from the restricted gold list (SHELL_RING_GOLD).
"""
from __future__ import annotations
import os
import sys
import math
import csv
import numpy as np
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/clipo/projects/earthwork-llm/src")
from demo_terrain_query import classify_geomorphon_simple
from earthwork_llm.ingestion.imageserver import fetch_dem

GOLD = os.environ.get("SHELL_RING_GOLD", "/home/clipo/projects/yazoo/data/shell_rings/SC_Gold_List.csv")
OUT = os.environ.get("OUT", "docs/figures/fig_shell_rings.png")
CLEAR = ["38BU0007", "Davis_32", "38BU0300", "38BU0301", "38BU0008"]

tf = Transformer.from_crs("EPSG:4326", "EPSG:26917", always_xy=True)


def get_dem(lat, lon, half):
    x, y = tf.transform(lon, lat)
    dem = fetch_dem(x, y, 2 * half, crs_epsg=26917, resolution_m=1.0)
    m = np.isfinite(dem)
    return np.where(m, dem, np.nanmedian(dem[m])).astype("float32")


def hillshade(dem, az):
    gy, gx = np.gradient(dem)
    sl = np.arctan(np.hypot(gy, gx))
    asp = np.arctan2(-gx, gy)
    z = math.radians(45)
    a = math.radians(az)
    return np.clip(np.cos(z) * np.cos(sl) + np.sin(z) * np.sin(sl) * np.cos(a - asp), 0, 1)


def multihs(dem):
    return 0.25 * sum(hillshade(dem, a) for a in (0, 90, 180, 270))


def lrm(dem, s=45):
    return dem - gaussian_filter(dem, s)


sites = []
with open(GOLD) as f:
    for r in csv.DictReader(f):
        if r.get("mound_id"):
            sites.append((r["mound_id"].strip(), float(r["latitude"]), float(r["longitude"])))

fig = plt.figure(figsize=(13, 5.6))
gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.0], hspace=0.18, wspace=0.10)

# ---- (a) structural signature at 38BU0300 -----------------------------------
H = 110
dem = get_dem(32.3409787, -80.7775707, H)
rel = lrm(dem)
geo = classify_geomorphon_simple(dem)
panels = [(multihs(dem), "gray", None, "hillshade"),
          (rel, "RdBu_r", np.nanpercentile(np.abs(rel), 98), "local relief"),
          (geo, "tab10", None, "geomorphons")]
for k, (im, cm, v, lab) in enumerate(panels):
    ax = fig.add_subplot(gs[0, k])
    if v is not None:
        ax.imshow(im, cmap=cm, vmin=-v, vmax=v)
    elif cm == "tab10":
        ax.imshow(im, cmap=cm, vmin=0, vmax=9)
    else:
        ax.imshow(im, cmap=cm)
    ax.set_title(lab, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if k == 0:
        ax.text(0.03, 0.97, "a", transform=ax.transAxes, fontsize=14, fontweight="bold",
                va="top", color="k", path_effects=[pe.withStroke(linewidth=2, foreground="w")])
# leave gs[0,3:] empty (white space keeps the row tidy)
for k in (3, 4):
    fig.add_subplot(gs[0, k]).axis("off")

# ---- (b) recovery gallery ----------------------------------------------------
by_id = {sid: (la, lo) for sid, la, lo in sites}
for k, sid in enumerate(CLEAR):
    la, lo = by_id[sid]
    ax = fig.add_subplot(gs[1, k])
    d = get_dem(la, lo, H)
    r = lrm(d)
    v = np.nanpercentile(np.abs(r), 98)
    ax.imshow(r, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.plot(H, H, "y+", ms=11, mew=2)
    ax.set_title(sid, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if k == 0:
        ax.text(0.03, 0.97, "b", transform=ax.transAxes, fontsize=14, fontweight="bold",
                va="top", path_effects=[pe.withStroke(linewidth=2, foreground="w")])

fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("wrote", OUT)
