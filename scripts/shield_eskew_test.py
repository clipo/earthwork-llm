"""Shield discrimination test on Eskew's field-verified set (Section 3.6 companion).

The paper's central architectural claim is that age discrimination belongs to the
context shield, not to morphology. Section 3.6 measured the morphology half (the
VLM). This script measures the shield half on the same 28 field-verified sites:
6 confirmed mounds and 22 modern earthworks. For each site we fetch the DEM, run
the detector, take the candidate nearest the coordinate (its aspect ratio feeds
the linearity layer), query NLCD at the point, and ask Shield V2 for a verdict.

Disclosure: the modern-feature proximity layer is INACTIVE (no noise map was
generated for the northern basin), so this tests the NLCD and linearity layers
only. Whatever the result, it is reported as measured.
"""
from __future__ import annotations
import os
import io
import csv
import math
import json
import numpy as np
from pathlib import Path

from demo_terrain_query import classify_geomorphon_simple
from earthwork_query import detect_earthworks
from earthwork_llm.ingestion.imageserver import fetch_dem
from earthwork_llm.surface.false_positive_shield import FalsePositiveShield
from earthwork_llm.ingestion.yazoo_downloader import YazooDownloader

SEED_CSV = os.environ.get("EARTHWORK_ABLATION_SET", "data/reference/mounds_seed.csv")
OUT = Path("data/shield_eskew")
UTM = 26915
WIN = 160


def load_sites():
    with open(SEED_CSV) as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    rows = list(csv.DictReader(io.StringIO("".join(lines))))
    out = []
    for r in rows:
        if not r.get("site_id"):
            continue
        ft, conf = r["feature_type"], r["confidence"]
        try:
            x, y = float(r["utm15n_x_m"]), float(r["utm15n_y_m"])
        except (ValueError, KeyError):
            continue
        if ft == "modern_earthwork_per_field":
            label = 0
        elif conf in ("high", "refined") and ft in (
            "mound_group", "village_with_mounds", "large_village_with_mounds", "platform_mound"):
            label = 1
        else:
            continue
        out.append((r["site_id"], r["site_name"], x, y, label))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from pyproj import Transformer
    inv = Transformer.from_crs("EPSG:26915", "EPSG:4326", always_xy=True)
    dl = YazooDownloader()
    shield = FalsePositiveShield(enclosure_query=False)
    sites = load_sites()
    print(f"sites: {sum(1 for s in sites if s[4]==1)} mounds, {sum(1 for s in sites if s[4]==0)} modern earthworks")
    rows = []
    for sid, name, x, y, label in sites:
        try:
            dem = fetch_dem(x, y, WIN, crs_epsg=UTM, resolution_m=1.0)
            m = np.isfinite(dem)
            dem = np.where(m, dem, np.nanmedian(dem[m])).astype("float32")
            geo = classify_geomorphon_simple(dem)
            cands = detect_earthworks(geo, dem, "Find pre-European earthwork mounds")
            half = WIN // 2
            best = min(cands, key=lambda c: math.hypot(c["x"] - half, c["y"] - half)) if cands else None
            aspect = float(best.get("aspect", 1.0)) if best else None
            base = float(best.get("probability", best.get("prob", 0.5))) if best else 0.5
            lon, lat = inv.transform(x, y)
            nlcd_val, nlcd_name = dl.get_nlcd_class(lon, lat)
            v = shield.evaluate(base_score=base, aspect=aspect,
                                nlcd_value=nlcd_val, nlcd_name=nlcd_name,
                                nearest_noise_m=None, nearest_noise_label="")
            dec = str(getattr(v, "decision", v)).split(".")[-1]
            rows.append(dict(site=name, label=label, aspect=round(aspect, 2) if aspect else None,
                             nlcd=nlcd_name, verdict=dec))
            print(f"  {'MOUND ' if label else 'modern'} {name[:24]:24} aspect={aspect if aspect else '-':>5} nlcd={nlcd_name[:22]:22} -> {dec}", flush=True)
        except Exception as e:
            rows.append(dict(site=name, label=label, aspect=None, nlcd="ERR", verdict=f"ERROR:{type(e).__name__}"))
            print(f"  ERR {name[:24]}: {type(e).__name__} {e}", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "shield_eskew_results.csv", index=False)
    ok = df[~df.verdict.str.startswith("ERROR")]
    mo = ok[ok.label == 1]
    md = ok[ok.label == 0]
    print("\n===== Shield V2 (NLCD + linearity; proximity layer inactive) =====")
    for nm, g in (("confirmed mounds", mo), ("modern earthworks", md)):
        c = g.verdict.value_counts().to_dict()
        print(f"  {nm:18} n={len(g)}: {c}")
    kept_mounds = int((mo.verdict != "REJECT").sum())
    rej_modern = int((md.verdict == "REJECT").sum())
    print(f"  mounds not rejected:      {kept_mounds}/{len(mo)}")
    print(f"  modern earthworks rejected: {rej_modern}/{len(md)}  (shield age-discrimination on this set)")
    (OUT / "summary.json").write_text(json.dumps(dict(
        mounds_total=len(mo), mounds_kept=kept_mounds,
        modern_total=len(md), modern_rejected=rej_modern,
        mound_verdicts=mo.verdict.value_counts().to_dict(),
        modern_verdicts=md.verdict.value_counts().to_dict()), indent=1))
    print("wrote", OUT / "shield_eskew_results.csv")


if __name__ == "__main__":
    main()
