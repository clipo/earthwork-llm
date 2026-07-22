"""Complete the shield test: proximity layer from open vector data (no OCR).

The modern-feature proximity layer was inactive in the first Eskew shield test
because no quad-derived noise map existed. The features that layer wants are,
however, available as open vector data: OpenStreetMap (waterway canals/ditches/
drains, highways, railways, dykes/embankments) via the Overpass API. This script
computes, for each of the 28 Eskew sites, the distance to the nearest mapped
modern linear feature, then re-evaluates Shield V2 with all three layers active
(linearity + NLCD from the earlier run, plus proximity), and reports the
completed confusion table.
"""
from __future__ import annotations
import time
import json
import math
import requests
import pandas as pd
from pathlib import Path

from earthwork_llm.surface.false_positive_shield import FalsePositiveShield
from pyproj import Transformer

PRIOR = "data/shield_eskew/shield_eskew_results.csv"
import os  # noqa: E402  (sys.path setup must precede repo-local imports)
SEED = os.environ.get("EARTHWORK_ABLATION_SET", "data/reference/mounds_seed.csv")
OUT = Path("data/shield_eskew")
# Overpass instances (tried in order). A descriptive User-Agent is required;
# the default python-requests UA can be refused by public instances.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
HEADERS = {"User-Agent": "earthwork-llm-research/1.0 (archaeological survey; clipo@binghamton.edu)"}
RADIUS = 300   # m search radius around each site

QUERY = """
[out:json][timeout:25];
(
  way(around:{r},{lat},{lon})["waterway"~"canal|ditch|drain"];
  way(around:{r},{lat},{lon})["highway"];
  way(around:{r},{lat},{lon})["railway"];
  way(around:{r},{lat},{lon})["man_made"~"dyke|embankment"];
  way(around:{r},{lat},{lon})["barrier"="ditch"];
);
out geom;
"""


def nearest_feature(lat, lon):
    """Return (distance_m, label) of nearest OSM modern linear feature."""
    q = QUERY.format(r=RADIUS, lat=lat, lon=lon)
    els = None
    for attempt in range(4):
        url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        try:
            r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=60)
            r.raise_for_status()
            els = r.json().get("elements", [])
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(8 * (attempt + 1))
    tf = Transformer.from_crs("EPSG:4326", "EPSG:26915", always_xy=True)
    px, py = tf.transform(lon, lat)
    best = (None, "")
    for el in els:
        tags = el.get("tags", {})
        label = (tags.get("waterway") or tags.get("man_made") or
                 tags.get("barrier") or tags.get("railway") or
                 ("road" if "highway" in tags else "feature"))
        for g in el.get("geometry", []):
            gx, gy = tf.transform(g["lon"], g["lat"])
            d = math.hypot(gx - px, gy - py)
            if best[0] is None or d < best[0]:
                best = (d, label)
    return best


def main():
    prior = pd.read_csv(PRIOR)
    # site -> coordinates from the seed list
    import csv
    import io
    rows = [r for r in csv.DictReader(io.StringIO("".join(
        ln for ln in open(SEED) if not ln.lstrip().startswith("#"))))]
    coords = {}
    inv = Transformer.from_crs("EPSG:26915", "EPSG:4326", always_xy=True)
    for r in rows:
        try:
            lon, lat = inv.transform(float(r["utm15n_x_m"]), float(r["utm15n_y_m"]))
            coords[r["site_name"]] = (lat, lon)
        except (ValueError, KeyError):
            pass
    shield = FalsePositiveShield(enclosure_query=False)
    # NLCD name -> value mapping for re-evaluation
    NLCD = {"Open Water": 11, "Developed": 21, "Developed, Open Space": 21,
            "Developed, Low Intensity": 22, "Developed, Medium Intensity": 23,
            "Developed, High Intensity": 24, "Barren Land": 31,
            "Deciduous Forest": 41, "Evergreen Forest": 42, "Mixed Forest": 43,
            "Shrub/Scrub": 52, "Herbaceous": 71, "Hay/Pasture": 81,
            "Pasture/Hay": 81, "Cultivated Crops": 82, "Woody Wetlands": 90,
            "Emergent Herbaceous Wetlands": 95}
    out = []
    for _, row in prior.iterrows():
        name = row["site"]
        if name not in coords or str(row["verdict"]).startswith("ERROR"):
            continue
        lat, lon = coords[name]
        try:
            dist, label = nearest_feature(lat, lon)
        except Exception as e:
            dist, label = None, f"ERR:{type(e).__name__}"
        nlcd_val = NLCD.get(str(row["nlcd"]).strip(), 0)
        aspect = row["aspect"] if pd.notna(row["aspect"]) else None
        v = shield.evaluate(base_score=0.6, aspect=aspect,
                            nlcd_value=nlcd_val, nlcd_name=str(row["nlcd"]),
                            nearest_noise_m=dist, nearest_noise_label=label)
        dec = str(getattr(v, "decision", v)).split(".")[-1]
        out.append(dict(site=name, label=int(row["label"]),
                        nearest_modern_m=round(dist, 1) if dist is not None else None,
                        feature=label, verdict_full=dec, verdict_prior=row["verdict"]))
        print(f"  {'MOUND ' if row['label'] else 'modern'} {name[:24]:24} "
              f"nearest {'none' if dist is None else round(dist):>5} m ({label[:14]:14}) -> {dec}", flush=True)
        time.sleep(2)   # be polite to Overpass
    df = pd.DataFrame(out)
    df.to_csv(OUT / "shield_eskew_proximity.csv", index=False)
    mo = df[df.label == 1]
    md = df[df.label == 0]
    print("\n===== Shield V2 with ALL layers (linearity + NLCD + OSM proximity) =====")
    print(f"  mounds  n={len(mo)}: {mo.verdict_full.value_counts().to_dict()}")
    print(f"  moderns n={len(md)}: {md.verdict_full.value_counts().to_dict()}")
    print(f"  mounds not rejected:        {int((mo.verdict_full!='REJECT').sum())}/{len(mo)}")
    print(f"  modern earthworks rejected: {int((md.verdict_full=='REJECT').sum())}/{len(md)}  (was 2/22 without proximity)")
    (OUT / "proximity_summary.json").write_text(json.dumps(dict(
        mounds_kept=int((mo.verdict_full != "REJECT").sum()), mounds_total=len(mo),
        modern_rejected=int((md.verdict_full == "REJECT").sum()), modern_total=len(md)), indent=1))


if __name__ == "__main__":
    main()
