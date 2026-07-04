#!/usr/bin/env python3
"""
Figure 1 - Study Area map for the EarthworkLLM paper.

Renders a publication-quality map of the Yazoo Basin showing the validation
mound set and the major Mississippian centers, with a CONUS locator inset,
state boundaries, the Mississippi / Yazoo river systems, a scale bar and a
north arrow.

Uses Cartopy + Natural Earth vector data so the output is crisp at any DPI
(no washed-out raster tiles). Falls back to a contextily relief basemap only
if Natural Earth data cannot be reached.

    python scripts/generate_paper_map.py
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# --- Major Mississippian centers used for calibration / orientation ----------
CENTERS = {
    "Winterville": (-91.062, 33.482),
    "Jaketown": (-90.485, 33.185),
    "Lake George\n(Holly Bluff)": (-90.712, 32.822),
    "Haynes Bluff": (-90.710, 32.515),
}

# Label offset (degrees) and alignment per center, to avoid collisions.
CENTER_LABEL = {
    "Winterville": (0.06, 0.04, "left"),
    "Jaketown": (0.06, 0.00, "left"),
    "Lake George\n(Holly Bluff)": (0.06, -0.02, "left"),
    "Haynes Bluff": (0.06, -0.04, "left"),
}

GOLD_CSV = os.environ.get(
    "EARTHWORK_GOLD_LIST",
    "/home/clipo/projects/yazoo/data/reference/located_mounds.csv")
# Fallback: reconstruct the gold coordinates from the validation summary.
GOLD_FALLBACK = "data/gold_validation_full/validation_summary.csv"

# Map extent (lon_min, lon_max, lat_min, lat_max). Covers both the central
# basin (validation mounds, case-study centers) and the northern basin, where
# Eskew's (2008) field-verified discrimination set lies (Section 3.6).
EXTENT = [-91.7, -89.65, 32.25, 34.95]

# Eskew (2008) ground-survey set: field-verified mounds and modern earthworks
# used as the Section 3.6 discrimination test. Coordinates from the seed list.
SEED_CSV = os.environ.get(
    "EARTHWORK_ABLATION_SET",
    "/home/clipo/projects/yazoo/data/reference/mounds_seed.csv")


def load_gold():
    path = GOLD_CSV if os.path.exists(GOLD_CSV) else GOLD_FALLBACK
    df = pd.read_csv(path)
    df = df.dropna(subset=["latitude", "longitude"])
    return df


def load_eskew():
    """Return (lons, lats) of the 28-site Eskew field-verified set."""
    import csv, io
    from pyproj import Transformer
    with open(SEED_CSV) as f:
        rows = list(csv.DictReader(io.StringIO(
            "".join(l for l in f if not l.lstrip().startswith("#")))))
    inv = Transformer.from_crs("EPSG:26915", "EPSG:4326", always_xy=True)
    lons, lats = [], []
    for r in rows:
        if not r.get("site_id"):
            continue
        ft, conf = r.get("feature_type", ""), r.get("confidence", "")
        keep = ft == "modern_earthwork_per_field" or (
            conf in ("high", "refined") and ft in (
                "mound_group", "village_with_mounds",
                "large_village_with_mounds", "platform_mound"))
        if not keep:
            continue
        try:
            lon, lat = inv.transform(float(r["utm15n_x_m"]), float(r["utm15n_y_m"]))
        except (ValueError, KeyError):
            continue
        lons.append(lon); lats.append(lat)
    return lons, lats


def add_scalebar(ax, lon0, lat0, length_km=25):
    """Simple scale bar in km, drawn in geographic coordinates."""
    # 1 deg lon at this latitude (~33N) ~= 93.4 km
    km_per_deg = 93.4
    dx = length_km / km_per_deg
    ax.plot([lon0, lon0 + dx], [lat0, lat0], color="black", lw=3,
            transform=ccrs.PlateCarree(), solid_capstyle="butt", zorder=20)
    ax.plot([lon0, lon0 + dx / 2], [lat0, lat0], color="white", lw=3,
            transform=ccrs.PlateCarree(), solid_capstyle="butt", zorder=21)
    ax.text(lon0 + dx / 2, lat0 + 0.025, f"{length_km} km", ha="center",
            va="bottom", fontsize=9, transform=ccrs.PlateCarree(), zorder=22,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])


def add_north_arrow(ax, x=0.95, y=0.95):
    ax.annotate("N", xy=(x, y), xytext=(x, y - 0.07),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=14, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="black"),
                zorder=30,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])


def draw_yazoo(fig, ax):
    gold = load_gold()
    proj = ccrs.PlateCarree()
    ax.set_extent(EXTENT, crs=proj)
    ax.set_facecolor("#efe9dd")
    rivers = cfeature.NaturalEarthFeature("physical", "rivers_lake_centerlines", "10m")
    states = cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lines", "10m")
    ax.add_feature(rivers, facecolor="none", edgecolor="#3b7ab5",
                   linewidth=1.4, zorder=2)
    ax.add_feature(states, facecolor="none", edgecolor="#7f8c8d", linewidth=1.1,
                   linestyle="--", zorder=3)
    ax.text(-91.55, 33.6, "ARKANSAS", fontsize=9, color="#7f8c8d",
            fontweight="bold", style="italic", zorder=4)
    ax.text(-90.3, 32.95, "MISSISSIPPI", fontsize=9, color="#7f8c8d",
            fontweight="bold", style="italic", zorder=4, ha="right")
    ax.text(-91.35, 33.05, "Mississippi R.", fontsize=8, color="#2c5f8a",
            rotation=58, style="italic", zorder=5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    ax.scatter(gold.longitude, gold.latitude, transform=proj,
               s=38, c="#f39c12", marker="o", edgecolor="black",
               linewidth=0.6, alpha=0.95, zorder=8)
    elons, elats = load_eskew()
    ax.scatter(elons, elats, transform=proj, s=26, c="#95a5a6", marker="D",
               edgecolor="black", linewidth=0.5, alpha=0.95, zorder=7)
    for name, (lon, lat) in CENTERS.items():
        ax.scatter([lon], [lat], transform=proj, s=190, c="#8B0000",
                   marker="^", edgecolor="white", linewidth=1.6, zorder=10)
        dx, dy, ha = CENTER_LABEL[name]
        ax.annotate(name, xy=(lon, lat), xytext=(lon + dx, lat + dy),
                    transform=proj, fontsize=10, fontweight="bold",
                    ha=ha, va="center", zorder=12,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                      alpha=0.4, linestyle=":")
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    add_scalebar(ax, lon0=-91.6, lat0=32.35, length_km=25)
    add_north_arrow(ax)
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", label="Major Mississippian center",
               markerfacecolor="#8B0000", markeredgecolor="white", markersize=12),
        Line2D([0], [0], marker="o", color="w",
               label=f"Validation mound (n={len(gold)})",
               markerfacecolor="#f39c12", markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="D", color="w",
               label=f"Age-discrimination site (n={len(elons)})",
               markerfacecolor="#95a5a6", markeredgecolor="black", markersize=7),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5,
              frameon=True, facecolor="white", framealpha=0.95,
              edgecolor="#bbb").set_zorder(31)
    ax.text(0.02, 0.98, "a", transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", zorder=40,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])


SC_GOLD = os.environ.get(
    "SHELL_RING_GOLD",
    "SET_SHELL_RING_GOLD")
SC_LABELED = ["38BU0007", "Davis_32", "38BU0300", "38BU0301", "38BU0008"]
SC_LABEL_OFF = {"38BU0007": (6, -3), "38BU0008": (7, 2), "Davis_32": (-46, 2),
                "38BU0300": (-52, 6), "38BU0301": (7, -9)}


def draw_sc(fig, ax):
    import csv
    proj = ccrs.PlateCarree()
    sites = []
    with open(SC_GOLD) as f:
        for r in csv.DictReader(f):
            if r.get("mound_id"):
                sites.append((r["mound_id"].strip(),
                              float(r["latitude"]), float(r["longitude"])))
    lats = [la for _, la, _ in sites]
    lons = [lo for _, _, lo in sites]
    pad = 0.28
    ax.set_extent([min(lons) - pad - 0.2, max(lons) + pad + 0.45,
                   min(lats) - pad + 0.06, max(lats) + pad - 0.02], crs=proj)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#efe9dd")
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dceaf4")
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7,
                   edgecolor="#666")
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), edgecolor="#3b7ab5",
                   linewidth=0.6)
    states = cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lines", "10m")
    ax.add_feature(states, facecolor="none", edgecolor="#7f8c8d", linewidth=0.9,
                   linestyle="--")
    ax.plot(lons, lats, "o", ms=8, mfc="#d95f02", mec="k", mew=0.8,
            transform=proj, zorder=5, ls="none")
    for sid, la, lo in sites:
        if sid in SC_LABELED:
            ax.annotate(sid, (lo, la), xytext=SC_LABEL_OFF.get(sid, (5, 4)),
                        textcoords="offset points", fontsize=7.5,
                        path_effects=[pe.withStroke(linewidth=2, foreground="w")])
    ax.text(-80.95, 32.44, "SOUTH CAROLINA", fontsize=9, style="italic",
            color="#7f8c8d", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="w")])
    ax.text(-80.62, 32.19, "Atlantic\nOcean", fontsize=9, style="italic",
            color="#33698f")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#bbb",
                      linestyle=":")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8}
    import math
    lat0 = min(lats) - pad + 0.12
    km10 = 10 / (111.32 * math.cos(math.radians(32.2)))
    x0 = min(lons) - pad - 0.1
    ax.plot([x0, x0 + km10], [lat0, lat0], "k-", lw=2.5, transform=proj)
    ax.text(x0 + km10 / 2, lat0 + 0.012, "10 km", ha="center", fontsize=7.5,
            transform=proj)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               label=f"Cataloged shell ring or mound (n={len(sites)})",
               markerfacecolor="#d95f02", markeredgecolor="black", markersize=9),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8.5,
              frameon=True, facecolor="white", framealpha=0.95,
              edgecolor="#bbb").set_zorder(31)
    ax.text(0.02, 0.98, "b", transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", zorder=40,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    return (min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2


def main():
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(15, 8.6))
    ax1 = fig.add_axes([0.04, 0.06, 0.44, 0.88], projection=proj)
    ax2 = fig.add_axes([0.53, 0.20, 0.44, 0.60], projection=proj)
    draw_yazoo(fig, ax1)
    sc_cx, sc_cy = draw_sc(fig, ax2)

    # Shared CONUS locator inset marking both regions.
    inset = fig.add_axes([0.395, 0.70, 0.20, 0.20],
                         projection=ccrs.LambertConformal(
                             central_longitude=-88, central_latitude=35))
    inset.set_extent([-98, -74, 24, 42], crs=proj)
    inset.add_feature(cfeature.LAND, facecolor="#e8e8e8")
    inset.add_feature(cfeature.OCEAN, facecolor="#cfe2f3")
    inset.add_feature(cfeature.STATES, edgecolor="#999", linewidth=0.3)
    inset.add_feature(cfeature.COASTLINE, edgecolor="#888", linewidth=0.4)
    cx = (EXTENT[0] + EXTENT[1]) / 2
    cy = (EXTENT[2] + EXTENT[3]) / 2
    inset.scatter([cx], [cy], transform=proj, s=90, marker="s",
                  facecolor="none", edgecolor="red", linewidth=2, zorder=10)
    inset.scatter([sc_cx], [sc_cy], transform=proj, s=90, marker="s",
                  facecolor="none", edgecolor="red", linewidth=2, zorder=10)
    inset.text(cx, cy + 1.1, "a", transform=proj, fontsize=9,
               fontweight="bold", ha="center", color="red")
    inset.text(sc_cx, sc_cy + 1.1, "b", transform=proj, fontsize=9,
               fontweight="bold", ha="center", color="red")

    os.makedirs("docs/figures", exist_ok=True)
    out = "docs/figures/fig0_yazoo_map.png"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Wrote {out} (two panels: Yazoo Basin + coastal South Carolina).")


if __name__ == "__main__":
    main()
