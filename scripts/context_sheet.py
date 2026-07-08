#!/usr/bin/env python
"""Per-candidate context sheet: land-cover history + modern-feature proximity.

Given a point in EPSG:26915 (NAD83 / UTM 15N), queries public keyless
services and emits a short structured text block for use as LLM context:

  1. Annual NLCD land-cover history (1985-present) via the USGS/MRLC
     time-enabled WMS at dmsdata.cr.usgs.gov (GetFeatureInfo + TIME param,
     same request pattern as earthwork_llm.ingestion.yazoo_downloader).
     Falls back to the MRLC epoch layers (2001/2004/.../2021) on
     www.mrlc.gov if the annual service is unreachable.
  2. Distance to nearest building footprint: FEMA USA Structures public
     ArcGIS FeatureServer (Oak Ridge / FEMA, hosted on arcgis.com).
  3. Distance to nearest canal/ditch (NHD FCode 336xx) and nearest
     natural stream (FCode 460xx): USGS National Map NHD MapServer.
  4. Distance to nearest road: OSM Overpass (URL rotation + descriptive
     User-Agent, pattern reused from scripts/shield_eskew_proximity.py).

Every layer degrades to "unavailable" without crashing; the downstream
pipeline must treat missing context as incomplete, never as clean.

Usage:
    python scripts/context_sheet.py --x 733740 --y 3673542
    python scripts/context_sheet.py --csv candidates.csv --out sheets.jsonl
        (CSV columns: utm_x, utm_y, optional id)

Only stdlib + requests + pyproj.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time

import requests
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Endpoints (all public, keyless)
# ---------------------------------------------------------------------------

# Annual NLCD Collection 1 land cover, time-enabled WMS (one layer, TIME=year).
ANNUAL_NLCD_WMS = ("https://dmsdata.cr.usgs.gov/geoserver/"
                   "mrlc_Land-Cover-Native_conus_year_data/wms")
ANNUAL_NLCD_LAYER = "Land-Cover-Native_conus_year_data"

# MRLC epoch fallback (legacy NLCD releases). NOTE: the Esri Living Atlas
# ImageServer returns HTTP 499 "Token Required"; these GeoServer WMS layers
# are the token-free route (see earthwork-llm yazoo_downloader.py).
MRLC_WMS = "https://www.mrlc.gov/geoserver/mrlc_display/wms"
MRLC_EPOCH_YEARS = [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]
MRLC_EPOCH_LAYER = "NLCD_{year}_Land_Cover_L48"

# FEMA USA Structures (building footprints), public hosted FeatureServer.
USA_STRUCTURES = ("https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/"
                  "rest/services/USA_Structures_View/FeatureServer/0/query")

# USGS National Map NHD; layer 6 = "Flowline - Large Scale" (high res).
NHD_FLOWLINE = ("https://hydro.nationalmap.gov/arcgis/rest/services/"
                "nhd/MapServer/6/query")
FCODE_CANAL = "fcode >= 33600 AND fcode <= 33699"    # CanalDitch
FCODE_STREAM = "fcode >= 46000 AND fcode <= 46099"   # StreamRiver

# Overpass instances (rotated). A descriptive User-Agent is required; the
# default python-requests UA can be refused by public instances.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
HEADERS = {"User-Agent":
           "terrallm-context-sheet/1.0 (archaeological survey; "
           "clipo@binghamton.edu)"}

SEARCH_RADIUS_M = 1000  # envelope half-width for all proximity queries

NLCD_CLASSES = {
    11: "Open Water", 12: "Perennial Ice/Snow",
    21: "Developed, Open Space", 22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity", 24: "Developed, High Intensity",
    31: "Barren Land", 41: "Deciduous Forest", 42: "Evergreen Forest",
    43: "Mixed Forest", 52: "Shrub/Scrub", 71: "Grassland/Herbaceous",
    81: "Pasture/Hay", 82: "Cultivated Crops", 90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands", 250: "No Data",
}

_TO_LL = Transformer.from_crs("EPSG:26915", "EPSG:4326", always_xy=True)

# ---------------------------------------------------------------------------
# HTTP with retries/backoff
# ---------------------------------------------------------------------------


def _request(url, *, params=None, data=None, retries=3, timeout=30,
             throttle=0.2):
    """GET (or POST when data given) with backoff. Returns Response or None."""
    for attempt in range(retries):
        try:
            time.sleep(throttle)
            if data is not None:
                r = requests.post(url, data=data, headers=HEADERS,
                                  timeout=timeout)
            else:
                r = requests.get(url, params=params, headers=HEADERS,
                                 timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 - degrade, never crash
            if attempt == retries - 1:
                print(f"  [context_sheet] {url.split('/')[2]} failed: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                return None
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Geometry helpers (planar, meters — inputs already in EPSG:26915)
# ---------------------------------------------------------------------------


def _pt_seg(px, py, ax, ay, bx, by):
    """Distance from point to segment; returns (dist, qx, qy)."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return math.hypot(px - ax, py - ay), ax, ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy), qx, qy


def _nearest_on_paths(px, py, paths):
    """Min distance from point to a list of vertex paths [[(x,y),...],...]."""
    best = (None, None, None)
    for path in paths:
        for (ax, ay), (bx, by) in zip(path, path[1:]):
            d, qx, qy = _pt_seg(px, py, ax, ay, bx, by)
            if best[0] is None or d < best[0]:
                best = (d, qx, qy)
        if len(path) == 1:
            (ax, ay) = path[0]
            d = math.hypot(px - ax, py - ay)
            if best[0] is None or d < best[0]:
                best = (d, ax, ay)
    return best


def _inside_ring(px, py, ring):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > py) != (yj > py)) and \
                (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _bearing(px, py, qx, qy):
    """8-wind compass direction from point (px,py) toward (qx,qy)."""
    ang = math.degrees(math.atan2(qx - px, qy - py)) % 360
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[int((ang + 22.5) // 45) % 8]


def _fmt_near(dist, px, py, qx, qy, extra=""):
    tail = f" {extra}" if extra else ""
    if dist is None:
        return f">{SEARCH_RADIUS_M} m"
    if dist < 1:
        return f"0 m (point on feature){tail}"
    return f"{dist:.0f} m {_bearing(px, py, qx, qy)}{tail}"


# ---------------------------------------------------------------------------
# 1. Land-cover history (Annual NLCD, epoch fallback)
# ---------------------------------------------------------------------------


def _wms_class_at(url, layer, lon, lat, time_param=None):
    """One WMS GetFeatureInfo -> NLCD class int, or None on failure."""
    d = 0.0005
    params = {
        "service": "WMS", "version": "1.1.1", "request": "GetFeatureInfo",
        "layers": layer, "query_layers": layer, "srs": "EPSG:4326",
        "bbox": f"{lon - d},{lat - d},{lon + d},{lat + d}",
        "width": "3", "height": "3", "x": "1", "y": "1",
        "info_format": "application/json",
    }
    if time_param:
        params["time"] = time_param
    resp = _request(url, params=params, retries=2, timeout=20)
    if resp is None:
        return None
    try:
        feats = resp.json().get("features", [])
        if not feats:
            return None
        props = feats[0].get("properties", {})
        val = props.get("PALETTE_INDEX", props.get("GRAY_INDEX"))
        return int(val) if val is not None else None
    except Exception:  # noqa: BLE001
        return None


def _annual_nlcd_years():
    """Years advertised by the annual WMS TIME dimension; [] on failure."""
    resp = _request(ANNUAL_NLCD_WMS,
                    params={"service": "WMS", "request": "GetCapabilities"},
                    retries=2, timeout=45)
    if resp is None:
        return []
    m = re.search(r'<Dimension name="time"[^>]*>([^<]+)</Dimension>',
                  resp.text)
    if not m:
        return []
    return sorted({int(y) for y in re.findall(r"(\d{4})-01-01", m.group(1))})


def _compress_runs(pairs):
    """[(year, name), ...] -> 'y1-y2 Name; y3 Name; ...'."""
    parts = []
    run_start, run_name = None, None
    prev = None
    for year, name in pairs:
        if name != run_name:
            if run_name is not None:
                span = (f"{run_start}" if run_start == prev
                        else f"{run_start}-{prev}")
                parts.append(f"{span} {run_name}")
            run_start, run_name = year, name
        prev = year
    if run_name is not None:
        span = f"{run_start}" if run_start == prev else f"{run_start}-{prev}"
        parts.append(f"{span} {run_name}")
    return "; ".join(parts)


def landcover_history(x, y):
    """Annual NLCD history string for a UTM-15N point."""
    lon, lat = _TO_LL.transform(x, y)
    years = _annual_nlcd_years()
    pairs = []
    if years:
        for yr in years:
            val = _wms_class_at(ANNUAL_NLCD_WMS, ANNUAL_NLCD_LAYER, lon, lat,
                                time_param=f"{yr}-01-01T00:00:00.000Z")
            if val is not None and val != 250 and val != 0:
                pairs.append((yr, NLCD_CLASSES.get(val, f"class {val}")))
    source = "Annual NLCD"
    if not pairs:
        # Fall back to legacy epoch layers on www.mrlc.gov.
        for yr in MRLC_EPOCH_YEARS:
            val = _wms_class_at(MRLC_WMS, MRLC_EPOCH_LAYER.format(year=yr),
                                lon, lat)
            if val is not None and val != 250 and val != 0:
                pairs.append((yr, NLCD_CLASSES.get(val, f"class {val}")))
        source = "NLCD epochs; annual service unavailable"
    if not pairs:
        return "Land cover (NLCD): unavailable"
    names = {n for _, n in pairs}
    if len(names) == 1:
        tag = "stable"
    else:
        tag = f"{len(names)} classes, land-cover change"
    return f"Land cover ({source}): {_compress_runs(pairs)} ({tag})"


# ---------------------------------------------------------------------------
# 2-3. ArcGIS envelope queries (USA Structures, NHD)
# ---------------------------------------------------------------------------


def _arcgis_envelope_query(url, x, y, where="1=1", out_fields=""):
    """Query an ArcGIS REST layer with a UTM envelope; returns feature list
    (geometries in EPSG:26915) or None if the service is unavailable."""
    r = SEARCH_RADIUS_M
    env = {"xmin": x - r, "ymin": y - r, "xmax": x + r, "ymax": y + r,
           "spatialReference": {"wkid": 26915}}
    data = {
        "geometry": json.dumps(env),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "26915", "outSR": "26915",
        "spatialRel": "esriSpatialRelIntersects",
        "where": where, "outFields": out_fields,
        "returnGeometry": "true", "f": "json",
    }
    resp = _request(url, data=data, retries=3, timeout=45)
    if resp is None:
        return None
    try:
        body = resp.json()
        if "error" in body:
            print(f"  [context_sheet] ArcGIS error: {body['error']}",
                  file=sys.stderr)
            return None
        return body.get("features", [])
    except Exception:  # noqa: BLE001
        return None


def nearest_structure(x, y):
    """Distance to nearest FEMA USA Structures footprint."""
    feats = _arcgis_envelope_query(USA_STRUCTURES, x, y,
                                   out_fields="OBJECTID")
    if feats is None:
        return "Nearest mapped structure (FEMA USA Structures): unavailable"
    best = (None, None, None)
    for f in feats:
        rings = f.get("geometry", {}).get("rings", [])
        for ring in rings:
            if _inside_ring(x, y, ring):
                return ("Nearest mapped structure (FEMA USA Structures): "
                        "0 m (point inside a mapped footprint)")
        d, qx, qy = _nearest_on_paths(x, y, rings)
        if d is not None and (best[0] is None or d < best[0]):
            best = (d, qx, qy)
    return ("Nearest mapped structure (FEMA USA Structures): "
            + _fmt_near(best[0], x, y, best[1], best[2]))


def _nearest_flowline(x, y, where):
    feats = _arcgis_envelope_query(NHD_FLOWLINE, x, y, where=where,
                                   out_fields="fcode,gnis_name")
    if feats is None:
        return None, None
    best = (None, None, None)
    name = ""
    for f in feats:
        paths = f.get("geometry", {}).get("paths", [])
        d, qx, qy = _nearest_on_paths(x, y, paths)
        if d is not None and (best[0] is None or d < best[0]):
            best = (d, qx, qy)
            gnis = (f.get("attributes") or {}).get("gnis_name")
            name = gnis if gnis else ""
    return best, name


def nearest_canal(x, y):
    best, name = _nearest_flowline(x, y, FCODE_CANAL)
    if best is None:
        return "Nearest canal/ditch (NHD): unavailable"
    extra = f"({name})" if name else ""
    return "Nearest canal/ditch (NHD): " + _fmt_near(
        best[0], x, y, best[1], best[2], extra)


def nearest_stream(x, y):
    best, name = _nearest_flowline(x, y, FCODE_STREAM)
    if best is None:
        return "Nearest natural stream (NHD): unavailable"
    extra = f"({name})" if name else ""
    return "Nearest natural stream (NHD): " + _fmt_near(
        best[0], x, y, best[1], best[2], extra)


# ---------------------------------------------------------------------------
# 4. Roads via OSM Overpass
# ---------------------------------------------------------------------------

_FROM_LL = Transformer.from_crs("EPSG:4326", "EPSG:26915", always_xy=True)


def nearest_road(x, y):
    lon, lat = _TO_LL.transform(x, y)
    q = (f"[out:json][timeout:25];"
         f'way(around:{SEARCH_RADIUS_M},{lat},{lon})["highway"];'
         f"out geom;")
    els = None
    for attempt in range(4):
        url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        resp = _request(url, data={"data": q}, retries=1, timeout=60)
        if resp is not None:
            try:
                els = resp.json().get("elements", [])
                break
            except Exception:  # noqa: BLE001
                pass
        time.sleep(4 * (attempt + 1))
    if els is None:
        return "Nearest road (OSM): unavailable"
    best = (None, None, None)
    kind = ""
    for el in els:
        pts = [_FROM_LL.transform(g["lon"], g["lat"])
               for g in el.get("geometry", [])]
        d, qx, qy = _nearest_on_paths(x, y, [pts]) if pts else (None,) * 3
        if d is not None and (best[0] is None or d < best[0]):
            best = (d, qx, qy)
            kind = (el.get("tags") or {}).get("highway", "")
    extra = f"({kind})" if kind else ""
    return "Nearest road (OSM): " + _fmt_near(
        best[0], x, y, best[1], best[2], extra)


# ---------------------------------------------------------------------------
# Sheet assembly + CLI
# ---------------------------------------------------------------------------


def context_sheet(x, y):
    """Full context sheet for a point in EPSG:26915. Never raises."""
    lines = []
    for fn in (landcover_history, nearest_structure, nearest_canal,
               nearest_stream, nearest_road):
        try:
            lines.append(fn(x, y))
        except Exception as e:  # noqa: BLE001 - a sheet must always emit
            lines.append(f"{fn.__name__}: unavailable "
                         f"({type(e).__name__})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Per-candidate context sheet (EPSG:26915 input).")
    ap.add_argument("--x", type=float, help="UTM 15N easting (m)")
    ap.add_argument("--y", type=float, help="UTM 15N northing (m)")
    ap.add_argument("--csv", help="batch input CSV with utm_x, utm_y "
                                  "and optional id columns")
    ap.add_argument("--out", help="output JSONL path (batch mode); "
                                  "default stdout")
    args = ap.parse_args()

    if args.csv:
        sink = open(args.out, "w") if args.out else sys.stdout
        try:
            with open(args.csv, newline="") as fh:
                for i, row in enumerate(csv.DictReader(fh)):
                    try:
                        px, py = float(row["utm_x"]), float(row["utm_y"])
                    except (KeyError, ValueError) as e:
                        print(f"  [context_sheet] row {i} skipped: {e}",
                              file=sys.stderr)
                        continue
                    rec = {"id": row.get("id", str(i)),
                           "utm_x": px, "utm_y": py,
                           "sheet": context_sheet(px, py)}
                    sink.write(json.dumps(rec) + "\n")
                    sink.flush()
        finally:
            if args.out:
                sink.close()
    elif args.x is not None and args.y is not None:
        print(context_sheet(args.x, args.y))
    else:
        ap.error("provide --x/--y or --csv")


if __name__ == "__main__":
    main()
