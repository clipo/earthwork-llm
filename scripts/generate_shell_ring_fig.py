"""Generate the shell-ring manuscript figure (map + signature + recovery gallery).

Panel (a): location map, coastal South Carolina (Beaufort County) with the ten
catalogued shell-ring sites and a conterminous-US inset, in the style of the
Yazoo study-area map. (b): the ring structural signature at 38BU0300 in
hillshade, local relief, and geomorphons. (c): local relief at five catalogued
rings, catalogue coordinate marked.

No caption text is baked into the image; captions belong to the manuscript.
Site coordinates come from the restricted gold list (SHELL_RING_GOLD).
"""
from __future__ import annotations
import os, sys, math, csv
import numpy as np
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_terrain_query import classify_geomorphon_simple
from earthwork_llm.ingestion.imageserver import fetch_dem

GOLD = os.environ.get("SHELL_RING_GOLD", "data/reference/SC_Gold_List.csv")
OUT = os.environ.get("OUT", "figures/fig_shell_rings.png")
CLEAR = ["38BU0007", "Davis_32", "38BU0300", "38BU0301", "38BU0008"]

tf = Transformer.from_crs("EPSG:4326", "EPSG:26917", always_xy=True)


def get_dem(lat, lon, half):
    x, y = tf.transform(lon, lat)
    dem = fetch_dem(x, y, 2 * half, crs_epsg=26917, resolution_m=1.0)
    m = np.isfinite(dem)
    return np.where(m, dem, np.nanmedian(dem[m])).astype("float32")


def hillshade(dem, az):
    gy, gx = np.gradient(dem)
    sl = np.arctan(np.hypot(gy, gx)); asp = np.arctan2(-gx, gy)
    z = math.radians(45); a = math.radians(az)
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

fig = plt.figure(figsize=(13, 10.5))
gs = fig.add_gridspec(3, 5, height_ratios=[1.5, 1.0, 1.0], hspace=0.16, wspace=0.10)

# ---- (a) location map -------------------------------------------------------
proj = ccrs.PlateCarree()
axm = fig.add_subplot(gs[0, :], projection=proj)
lats = [la for _, la, _ in sites]; lons = [lo for _, _, lo in sites]
pad = 0.28
axm.set_extent([min(lons) - pad - 0.25, max(lons) + pad + 0.55,
                min(lats) - pad + 0.06, max(lats) + pad - 0.02], crs=proj)
axm.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f4efe6")
axm.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dceaf4")
axm.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7, edgecolor="#666")
axm.add_feature(cfeature.RIVERS.with_scale("10m"), edgecolor="#3b7ab5", linewidth=0.6)
states = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines", "10m")
axm.add_feature(states, facecolor="none", edgecolor="#7f8c8d", linewidth=0.9)
axm.plot(lons, lats, "o", ms=8, mfc="#d95f02", mec="k", mew=0.8, transform=proj, zorder=5)
LABEL_OFF = {"38BU0007": (6, -3), "38BU0008": (7, 2), "Davis_32": (-52, 2),
             "38BU0300": (-58, 6), "38BU0301": (7, -9)}
for sid, la, lo in sites:
    if sid in CLEAR:
        axm.annotate(sid, (lo, la), xytext=LABEL_OFF.get(sid, (5, 4)),
                     textcoords="offset points", fontsize=7.5,
                     path_effects=[pe.withStroke(linewidth=2, foreground="w")])
axm.text(-80.95, 32.44, "Beaufort\nCounty", fontsize=9, style="italic", color="#555",
         path_effects=[pe.withStroke(linewidth=2, foreground="w")])
axm.text(-80.62, 32.19, "Atlantic\nOcean", fontsize=9, style="italic", color="#33698f")
gl = axm.gridlines(draw_labels=True, linewidth=0.3, color="#bbb", linestyle=":")
gl.top_labels = gl.right_labels = False
gl.xlabel_style = gl.ylabel_style = {"size": 7}
axm.text(0.012, 0.965, "a", transform=axm.transAxes, fontsize=14, fontweight="bold", va="top")
# scale bar (~10 km at this latitude)
lat0 = min(lats) - pad + 0.12
km10 = 10 / (111.32 * math.cos(math.radians(32.2)))
x0 = min(lons) - pad - 0.15
axm.plot([x0, x0 + km10], [lat0, lat0], "k-", lw=2.5)
axm.text(x0 + km10 / 2, lat0 + 0.012, "10 km", ha="center", fontsize=7.5)
# conterminous-US inset
axi = fig.add_axes([0.755, 0.715, 0.16, 0.185], projection=ccrs.LambertConformal())
axi.set_extent([-120, -73, 23, 48], crs=proj)
axi.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f4efe6")
axi.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#dceaf4")
axi.add_feature(cfeature.STATES.with_scale("110m"), edgecolor="#aaa", linewidth=0.3)
axi.plot(np.mean(lons), np.mean(lats), "s", ms=6, mfc="#d95f02", mec="k", transform=proj)

# ---- (b) structural signature at 38BU0300 -----------------------------------
H = 110
dem = get_dem(32.3409787, -80.7775707, H)
rel = lrm(dem); geo = classify_geomorphon_simple(dem)
panels = [(multihs(dem), "gray", None, "hillshade"),
          (rel, "RdBu_r", np.nanpercentile(np.abs(rel), 98), "local relief"),
          (geo, "tab10", None, "geomorphons")]
for k, (im, cm, v, lab) in enumerate(panels):
    ax = fig.add_subplot(gs[1, k])
    if v is not None:
        ax.imshow(im, cmap=cm, vmin=-v, vmax=v, origin="lower")
    elif cm == "tab10":
        ax.imshow(im, cmap=cm, vmin=0, vmax=9, origin="lower")
    else:
        ax.imshow(im, cmap=cm, origin="lower")
    ax.set_title(lab, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if k == 0:
        ax.text(0.03, 0.97, "b", transform=ax.transAxes, fontsize=14, fontweight="bold",
                va="top", color="k", path_effects=[pe.withStroke(linewidth=2, foreground="w")])
# leave gs[1,3:] empty (white space keeps the row tidy)
for k in (3, 4):
    fig.add_subplot(gs[1, k]).axis("off")

# ---- (c) recovery gallery ----------------------------------------------------
by_id = {sid: (la, lo) for sid, la, lo in sites}
for k, sid in enumerate(CLEAR):
    la, lo = by_id[sid]
    ax = fig.add_subplot(gs[2, k])
    d = get_dem(la, lo, H); r = lrm(d)
    v = np.nanpercentile(np.abs(r), 98)
    ax.imshow(r, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
    ax.plot(H, H, "y+", ms=11, mew=2)
    ax.set_title(sid, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if k == 0:
        ax.text(0.03, 0.97, "c", transform=ax.transAxes, fontsize=14, fontweight="bold",
                va="top", path_effects=[pe.withStroke(linewidth=2, foreground="w")])

fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("wrote", OUT)
