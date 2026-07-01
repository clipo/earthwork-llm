"""Build a coordinate-correction package: a clickable relief image per site.

For each site (published UTM15N coordinate), fetch a DEM window, render a
hillshade + wide-LRM relief PNG, and record the exact UTM bounds so a click in
the web tool maps back to a precise coordinate. The correction web app
(scripts/correct_server.py) serves these and records the corrected summit.

Usage:
    python scripts/build_correct.py            # default named sites
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, ".")
from earthwork_llm.ingestion.yazoo_downloader import YazooDownloader  # noqa (ensure repo import path)
from earthwork_llm.ingestion.imageserver import Usgs3depImageServerSource, WindowRequest
from pyproj import Transformer

UTM = "EPSG:26915"
RES = 1.0
HALF_M = 300        # 600 m window
TILE = 1024
OUT = Path("data/correct"); (OUT / "img").mkdir(parents=True, exist_ok=True)
TO_WGS = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)

# Named sites to correct: (id, label, published UTM15N x, y)
SITES = [
    ("Winterville_MoundA", "Winterville Mound A (22-Ws-500)", 680252.0, 3706813.0),
    ("Gates",              "Gates (Co-514)",                  743002.0, 3815532.0),
    ("IC_Village",         "I.C. Village (Co-672)",           728815.0, 3807777.0),
]


def fetch_dem(cx, cy, half_m):
    src = Usgs3depImageServerSource()
    xmin = math.floor(cx - half_m); ymax = math.ceil(cy + half_m)
    W = H = int(2 * half_m)
    full = np.full((H, W), np.nan, "float32")
    for py in range(0, H, TILE):
        for px in range(0, W, TILE):
            tcx = (xmin + px) + TILE / 2.0; tcy = (ymax - py) - TILE / 2.0
            req = WindowRequest(center_x=tcx, center_y=tcy, utm_crs=UTM, resolution_m=RES, size_px=TILE)
            for a in range(5):
                try: arr = src.fetch_window(req); break
                except Exception:
                    if a == 4: raise
                    time.sleep(2 * (a + 1))
            h, w = min(TILE, H - py), min(TILE, W - px)
            full[py:py+h, px:px+w] = arr[:h, :w]
    return full, xmin, ymax, W, H


def hillshade(rel, az_deg=315, alt=45):
    gy, gx = np.gradient(rel)
    slope = np.arctan(np.hypot(gy, gx)); aspect = np.arctan2(-gx, gy)
    az = math.radians(az_deg); zen = math.radians(90 - alt)
    hs = np.cos(zen)*np.cos(slope) + np.sin(zen)*np.sin(slope)*np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def main():
    sites_out = []
    for sid, label, cx, cy in SITES:
        print(f"== {label} ==", flush=True)
        dem, xmin, ymax, W, H = fetch_dem(cx, cy, HALF_M)
        mask = np.isfinite(dem); filled = np.where(mask, dem, np.nanmedian(dem))
        rel = filled - np.nanmedian(filled)
        lw = rel - gaussian_filter(filled, 40)
        # Crisp grayscale multidirectional hillshade, contrast-stretched.
        hs = 0.25*(hillshade(rel, 315) + hillshade(rel, 45) + hillshade(rel, 135) + hillshade(rel, 225))
        lo, hi = np.nanpercentile(hs[mask], [2, 98]) if mask.any() else (0, 1)
        hsn = np.clip((hs - lo) / max(hi - lo, 1e-6), 0, 1)
        base = np.dstack([hsn, hsn, hsn])
        # Overlay POSITIVE wide-LRM (mounds) as a warm tint; tight stretch so
        # 0.4-1.5 m rises pop. Negative relief (channels) is not overlaid.
        pcap = max(np.nanpercentile(lw[mask & (lw > 0)], 92) if (mask & (lw > 0)).any() else 1.0, 1.0)
        posn = np.clip(lw / pcap, 0, 1)
        orange = np.array([1.0, 0.55, 0.0])
        a = (0.75 * posn)[..., None]
        img = np.clip(base * (1 - a) + orange[None, None, :] * a, 0, 1)
        fig, ax = plt.subplots(figsize=(7, 7)); ax.imshow(img); ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(OUT / "img" / f"{sid}.png", dpi=100); plt.close(fig)
        # published pixel (fractional)
        pfx = (cx - xmin) / W; pfy = (ymax - cy) / H
        lon, lat = TO_WGS.transform(cx, cy)
        sites_out.append(dict(id=sid, label=label,
                              pub_utm_x=cx, pub_utm_y=cy, pub_lat=round(lat, 6), pub_lon=round(lon, 6),
                              pub_fx=round(pfx, 5), pub_fy=round(pfy, 5),
                              utm_xmin=xmin, utm_ymax=ymax, w=W, h=H, res=RES,
                              img=f"img/{sid}.png"))
        print(f"   rendered {sid}.png  ({W}x{H} m)")
    (OUT / "sites.json").write_text(json.dumps(sites_out, indent=2))
    print("wrote", OUT / "sites.json")


if __name__ == "__main__":
    main()
