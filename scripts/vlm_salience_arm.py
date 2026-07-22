"""Salience-rubric arm: the model as a structured reader of cultural evidence.

Instead of a binary MOUND/NOT-MOUND verdict, the model scores each candidate
0-5 on five axes of affirmative cultural evidence, the dimensions a human
surveyor uses: engineered form, built construction, cultural placement
(including orientation independent of the modern grid), distinctiveness from
the surrounding fabric, and an overall cultural-evidence score. Each candidate
is presented as the deployed composite (160 m six-panel detail + 600 m context).

Two evaluation sets with existing ground truth:
  --set jaketown   80 scan survivors; desk review labels 8 uncertain (plausible)
                   vs 72 rejected. Success = composite salience AUC > 0.43
                   (the verdict-share baseline that failed).
  --set eskew      28 field-verified sites; 6 confirmed mounds vs 22 confirmed
                   modern earthworks. Success = any axis separates the classes,
                   orientation/placement especially.

Usage:
  python scripts/vlm_salience_arm.py --set jaketown --out data/v10_eval/salience_jaketown.csv
  python scripts/vlm_salience_arm.py --set eskew    --out data/v10_eval/salience_eskew.csv

Env: VLM_API (default http://localhost:8001/v1/chat/completions),
     VLM_MODEL (default terrallm-v91).
"""
from __future__ import annotations
import os
import re
import csv
import sys
import json
import time
import argparse
import requests

sys.path.insert(0, "scripts")

sys.path.insert(0, "src")
import vlm_eval_v10 as base
from demo_terrain_query import pil_to_b64_data_uri

API = os.environ.get("VLM_API", "http://localhost:8001/v1/chat/completions")
MODEL = os.environ.get("VLM_MODEL", "terrallm-v91")
JAKETOWN_VERDICTS = os.environ.get("JAKETOWN_VERDICTS", "data/review/jaketown_verdicts.csv")

AXES = ["FORM", "CONSTRUCTION", "PLACEMENT", "DISTINCTIVENESS", "CULTURAL"]

PROMPT = """You are reading a bare-earth LiDAR composite of one candidate earthen feature. The upper block is a six-panel 160 m detail of the candidate (four hillshades, the geomorphon classification, and contours). The lower block is a 600 m context view (hillshade and local relief) with the candidate at center.

Your task is not to reject false positives. It is to assess the affirmative evidence that this feature is a deliberate cultural construction, a pre-industrial earthwork such as a mound, platform, ring, enclosure, or ditch. Weigh the qualities that make a feature stand out to an experienced human surveyor.

Score five axes from 0 to 5, each with a one-line justification:

FORM: engineered geometry. Circular or oval symmetry, a level summit platform, a coherent ring or enclosure plan. 0 = amorphous, 5 = clearly engineered.
CONSTRUCTION: built earthen mass. Raised volume with smoothed, eroded flanks consistent with old construction, not fresh spoil, plow residue, or natural deposition. 0 = no constructed mass, 5 = unmistakably built and weathered.
PLACEMENT: cultural siting. Positioned where a builder would place it, and oriented independently of the modern field grid, roads, and canals. 0 = placement and orientation follow the modern grid, 5 = siting reads as deliberate and independent of modern layout.
DISTINCTIVENESS: stands apart from both the natural fabric and the modern agricultural or engineering fabric of the surrounding 600 m. 0 = blends in, 5 = conspicuous anomaly.
CULTURAL: your overall reading of the evidence that this is a deliberate pre-industrial construction, weighing everything above. 0 = none, 5 = compelling.

Judge every anomaly relative to its terrain setting. In near-level floodplain, small rises matter; in dissected or hilly ground, judge relief against the local fabric rather than absolute height. Work across scales as well. Detail that looks noisy at 160 m may organize into a coherent feature at 600 m, and a large feature may register only as a broad, low pattern.

Most survivors in these landscapes are modern spoil, road fill, or farm features. Score conservatively and use the full 0 to 5 range, reserving 4 and 5 for evidence that would justify a field visit.

Respond with exactly five lines in this format and nothing else. Do not use any other report format.
FORM: <0-5> | <reason>
CONSTRUCTION: <0-5> | <reason>
PLACEMENT: <0-5> | <reason>
DISTINCTIVENESS: <0-5> | <reason>
CULTURAL: <0-5> | <reason>"""


AXES_V2 = ["FORM", "CONSTRUCTION", "PLACEMENT", "ISOLATION", "CULTURAL"]

PROMPT_V2 = """You are reading a bare-earth LiDAR composite of one candidate earthen feature. The upper block is a six-panel 160 m detail of the candidate (four hillshades, the geomorphon classification, and contours). The lower block is a context view (hillshade and local relief) with the candidate at center.

Your task is anomaly assessment. Cultural earthworks are features that cannot be accounted for by any visible process. Ask one organizing question: is this feature anomalous with respect to the natural fabric, the modern fabric, and the terrain setting, at any scale? A feature anomalous on all three counts has no explanation left but deliberate construction. Judge relief relative to the local terrain fabric, not absolute height, and work across scales, since detail that looks noisy up close may organize into a coherent feature in the wider view.

First name the plan form you see, then score five axes 0 to 5 with a one-line justification each.

PLAN: one of conical-mound | platform | ring | enclosure | linear-bank | ditch-circuit | amorphous | other.
FORM: geometry that excludes non-cultural origins at this scale. Circles, rings, squares, and right angles do not form naturally, but the modern landscape also builds them (center-pivot circles, tank pads, ring levees, stock ponds), so score high only when the geometry AND its size have no common natural or modern counterpart. A raised ring tens to hundreds of meters across enclosing a level interior, or a flat-topped platform with ramped access, scores high; a small crisp circle that could be a tank pad does not.
CONSTRUCTION: built earthen mass. Raised or excavated volume with smoothed, weathered flanks consistent with old earthmoving, not fresh spoil, plow residue, or natural deposition.
PLACEMENT: cultural siting. Positioned where a builder would place it, oriented independently of the modern field grid, roads, and canals.
ISOLATION: inexplicability. Can anything visible in the context, natural or modern, account for this feature: a levee or scroll bar that would shed it, a channel that would build it, a road, canal, structure, or field system that would explain it? Score high only when nothing visible explains the feature. An unexplained compact earthwork alone in open field or forest is itself evidence of cultural origin.
CULTURAL: your overall anomaly judgment that this is deliberate pre-industrial construction, weighing everything above.

Most candidates in these landscapes are modern spoil, road fill, or farm features. Use the full 0 to 5 range and reserve 4 and 5 for evidence that would justify a field visit.

Respond with exactly six lines in this format and nothing else. Do not use any other report format.
PLAN: <form>
FORM: <0-5> | <reason>
CONSTRUCTION: <0-5> | <reason>
PLACEMENT: <0-5> | <reason>
ISOLATION: <0-5> | <reason>
CULTURAL: <0-5> | <reason>"""

RETRY_SUFFIX = "\n\nIMPORTANT: your previous response used the wrong format. Respond ONLY with the five FORM/CONSTRUCTION/PLACEMENT/DISTINCTIVENESS/CULTURAL lines exactly as specified."


def query(img, prompt=None) -> str:
    payload = {"model": MODEL, "messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": pil_to_b64_data_uri(img)}},
            {"type": "text", "text": prompt or PROMPT}]}],
        "max_tokens": 4096, "temperature": 0.0}
    r = requests.post(API, json=payload, timeout=900)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def query_with_retry(img):
    txt = query(img)
    if len(parse(txt)) >= 2 * len(AXES) - 1:
        return txt, 1
    txt2 = query(img, PROMPT + RETRY_SUFFIX)
    return (txt2 if len(parse(txt2)) > len(parse(txt)) else txt), 2


def parse(txt: str) -> dict:
    tail = txt.split("</think>")[-1]
    out = {}
    m = re.search(r"PLAN\s*:\s*([A-Za-z-]+)", tail)
    if m:
        out["plan"] = m.group(1).lower()
    for ax in AXES:
        m = re.search(rf"{ax}\s*:\s*([0-5])(?:\s*/\s*5)?\s*\|?\s*(.*)", tail, re.I)
        if m:
            out[ax.lower()] = int(m.group(1))
            out[ax.lower() + "_note"] = m.group(2).strip()[:200]
    return out



def wide_view_at(x, y, epsg, ctx_m):
    """Context panel at an arbitrary window size (600 px render)."""
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.ndimage import gaussian_filter
    import matplotlib.cm as cm
    px = 600
    res = max(1.0, ctx_m / px)
    fetch_px = int(round(ctx_m / res))
    dem = base.clean(base.fetch_dem(x, y, fetch_px, crs_epsg=epsg, resolution_m=res))
    hs8 = (base.hillshade(dem) * 255).astype(np.uint8)
    rel = dem - gaussian_filter(dem, 45 * 1.0 / res if res < 1.0 else 45)
    v = np.nanpercentile(np.abs(rel), 98) or 1.0
    rel_rgb = (cm.RdBu_r(np.clip((rel / v + 1) / 2, 0, 1))[:, :, :3] * 255).astype(np.uint8)
    combo = np.concatenate([np.stack([hs8] * 3, axis=-1), rel_rgb], axis=1)
    img = Image.fromarray(combo)
    if fetch_px != px:
        img = img.resize((px * 2, px), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    w = img.size[0] // 2
    for cx in (w // 2, w + w // 2):
        d.ellipse([cx - 12, img.size[1] // 2 - 12, cx + 12, img.size[1] // 2 + 12],
                  outline=(255, 255, 0), width=3)
    return img


def make_composite_ctx(x, y, epsg, ctx_m):
    if ctx_m == 600:
        return base.make_composite(x, y, epsg)
    from generate_discrimination_v10 import composite as compose
    dem = base.clean(base.fetch_dem(x, y, base.DETAIL_PX, crs_epsg=epsg, resolution_m=1.0))
    from demo_terrain_query import classify_geomorphon_simple, make_multi_view_panel
    detail = make_multi_view_panel(dem, classify_geomorphon_simple(dem))
    return compose(detail, wide_view_at(x, y, epsg, ctx_m))


def jaketown_sites():
    out = []
    for r in csv.DictReader(open(JAKETOWN_VERDICTS)):
        label = 1 if r["verdict"] == "uncertain" else 0
        out.append((r["id"], float(r["utm_x"]), float(r["utm_y"]), 26915, label))
    return out


def eskew_sites():
    # (name, x, y, epsg, label) with label 1 = confirmed mound, 0 = modern
    return base.eskew_set()


def auc(scores, labels):
    pos = [s for s, lab in zip(scores, labels) if lab == 1 and s is not None]
    neg = [s for s, lab in zip(scores, labels) if lab == 0 and s is not None]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["jaketown", "eskew"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-raw", default="")
    ap.add_argument("--rubric", choices=["v1", "v2"], default="v1")
    ap.add_argument("--sheets", default="", help="JSONL of per-candidate context sheets to append to the prompt")
    ap.add_argument("--context-m", type=int, default=600)
    args = ap.parse_args()
    global PROMPT, AXES
    if args.rubric == "v2":
        PROMPT, AXES = PROMPT_V2, AXES_V2

    sheets = {}
    if args.sheets:
        for line in open(args.sheets):
            rec = json.loads(line)
            sheets[str(rec.get("id"))] = rec.get("sheet") or rec.get("text") or ""
    sites = jaketown_sites() if args.set == "jaketown" else eskew_sites()
    if args.limit:
        sites = sites[:args.limit]
    rows = []
    for i, (sid, x, y, epsg, label) in enumerate(sites, 1):
        t0 = time.time()
        try:
            img = make_composite_ctx(x, y, epsg, args.context_m)
            if sheets.get(str(sid)):
                site_prompt = PROMPT + "\n\nIndependent map and land-use records for this location:\n" + sheets[str(sid)] + "\nWeigh these records when scoring PLACEMENT and ISOLATION, and CULTURAL overall. A feature aligned with or adjacent to recorded modern infrastructure, or on land whose cover history matches modern construction, is explained; a feature these records leave unexplained is anomalous."
                txt = query(img, site_prompt)
                if len(parse(txt)) < 2 * len(AXES) - 1:
                    txt = query(img, site_prompt + RETRY_SUFFIX)
                attempts = 1
            else:
                txt, attempts = query_with_retry(img)
            if args.save_raw:
                with open(args.save_raw, "a") as f:
                    f.write(json.dumps({"id": sid, "label": label, "text": txt}) + "\n")
            d = parse(txt)
            scores = [d.get(ax.lower()) for ax in AXES]
            comp = (sum(s for s in scores if s is not None) / len([s for s in scores if s is not None])
                    if any(s is not None for s in scores) else None)
            row = dict(id=sid, label=label, plan=d.get("plan", ""),
                       composite=(round(comp, 2) if comp is not None else ""))
            for ax in AXES:
                row[ax.lower()] = d.get(ax.lower(), "")
                row[ax.lower() + "_note"] = d.get(ax.lower() + "_note", "")
            rows.append(row)
            print(f"[{i}/{len(sites)}] {sid} (label {label}): "
                  f"{'/'.join(str(row[ax.lower()]) for ax in AXES)} comp={row['composite']} "
                  f"{time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            rows.append(dict(id=sid, label=label, plan="", composite="",
                             **{ax.lower(): "" for ax in AXES},
                             **{ax.lower() + "_note": f"ERR {type(e).__name__}" for ax in AXES}))
            print(f"[{i}/{len(sites)}] {sid}: ERR {type(e).__name__}: {e}", flush=True)

    fieldnames = (["id", "label", "plan", "composite"] + [ax.lower() for ax in AXES]
                  + [ax.lower() + "_note" for ax in AXES])
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")
    labels = [r["label"] for r in rows]
    for key in [ax.lower() for ax in AXES] + ["composite"]:
        vals = [(float(r[key]) if r[key] != "" else None) for r in rows]
        a = auc(vals, labels)
        print(f"  AUC {key:16s}: {a:.2f}" if a is not None else f"  AUC {key:16s}: n/a")


if __name__ == "__main__":
    main()
