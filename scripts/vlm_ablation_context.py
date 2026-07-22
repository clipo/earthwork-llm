"""Context-conditioned VLM ablation (arm B of the Section 3.6 experiment).

Arm A (vlm_ablation.py) tested the model on the 160 m six-panel image alone and
found it accepts most modern earthworks. But the deployed pipeline supplies
context, and a human separates levee spoil from a mound by seeing the adjacent
canal — invisible in a 160 m crop. Arm B therefore gives the model the same
six-panel detail PLUS a wide 600 m context view (hillshade + local relief), and
asks the same MOUND / NOT_MOUND question. No labels leak: the extra input is
more of the same public DEM.

Compares directly against arm A (Table B1): mound recall and modern-earthwork
rejection, majority vote of three runs per site.
"""
from __future__ import annotations
import os
import io
import re
import csv
import json
import time
import math
import numpy as np
import requests
from pathlib import Path
from PIL import Image

from demo_terrain_query import classify_geomorphon_simple, make_multi_view_panel, pil_to_b64_data_uri
from earthwork_query import SYSTEM_PROMPT
from earthwork_llm.ingestion.imageserver import fetch_dem
from scipy.ndimage import gaussian_filter

API = os.environ.get("VLM_API", "http://localhost:8001/v1/chat/completions")
MODEL = os.environ.get("VLM_MODEL", "terrallm-v91")
UTM_EPSG = 26915
DETAIL_PX = 160          # same as arm A
WIDE_M = 600             # context window
SEED_CSV = os.environ.get("EARTHWORK_ABLATION_SET", "data/reference/mounds_seed.csv")
OUT_DIR = Path("data/vlm_ablation_context")
K = int(os.environ.get("VLM_RUNS", "3"))

VERDICT_INSTR = (
    "\n\nYou are given TWO images of the same location. Image 1 is a 6-panel detail "
    "view (160 m across: 4 hillshades | geomorphon | contours) centred on a candidate "
    "feature. Image 2 is a WIDE context view (600 m across: hillshade left, local "
    "relief right) with the candidate at the centre. Use the wide view to judge "
    "context: pre-European mounds usually stand apart from linear modern works, while "
    "levee spoil, canal cleanout piles, and berms sit along or beside the linear "
    "ditches, canals, channels, or field edges that produced them. Decide whether the "
    "central feature is a pre-European earthen mound or not (natural landform or "
    "modern earthwork). End your reply with exactly two lines:\n"
    "VERDICT: MOUND        (if most likely a pre-European mound)\n"
    "or VERDICT: NOT_MOUND (if natural or modern)\n"
    "CONFIDENCE: <integer 0-100>"
)


def load_eval_set():
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
            label, kind = 0, "modern_earthwork"
        elif conf in ("high", "refined") and ft in (
            "mound_group", "village_with_mounds", "large_village_with_mounds", "platform_mound"):
            label, kind = 1, "confirmed_mound"
        else:
            continue
        out.append(dict(site_id=r["site_id"], name=r["site_name"], x=x, y=y,
                        label=label, kind=kind))
    return out


def clean(dem):
    m = np.isfinite(dem)
    return np.where(m, dem, np.nanmedian(dem[m])).astype("float32")


def hillshade(dem, az=315):
    gy, gx = np.gradient(dem)
    sl = np.arctan(np.hypot(gy, gx))
    asp = np.arctan2(-gx, gy)
    z = math.radians(45)
    a = math.radians(az)
    return np.clip(np.cos(z) * np.cos(sl) + np.sin(z) * np.sin(sl) * np.cos(a - asp), 0, 1)


def wide_panel(x, y):
    dem = clean(fetch_dem(x, y, WIDE_M, crs_epsg=UTM_EPSG, resolution_m=1.0))
    hs = (hillshade(dem) * 255).astype(np.uint8)
    rel = dem - gaussian_filter(dem, 45)
    v = np.nanpercentile(np.abs(rel), 98) or 1.0
    rel8 = np.clip((rel / v + 1) * 127.5, 0, 255).astype(np.uint8)
    import matplotlib.cm as cm
    rel_rgb = (cm.RdBu_r(rel8 / 255.0)[:, :, :3] * 255).astype(np.uint8)
    hs_rgb = np.stack([hs] * 3, axis=-1)
    combo = np.concatenate([hs_rgb, rel_rgb], axis=1)
    img = Image.fromarray(combo)
    # centre marker
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for cx in (WIDE_M // 2, WIDE_M + WIDE_M // 2):
        d.ellipse([cx - 12, WIDE_M // 2 - 12, cx + 12, WIDE_M // 2 + 12], outline=(255, 255, 0), width=3)
    return img


def query(panel_detail, panel_wide):
    payload = {"model": MODEL, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": pil_to_b64_data_uri(panel_detail)}},
            {"type": "image_url", "image_url": {"url": pil_to_b64_data_uri(panel_wide)}},
            {"type": "text", "text": "Analyse the terrain." + VERDICT_INSTR}]}],
        "max_tokens": 4096, "temperature": 0}
    r = requests.post(API, json=payload, timeout=900)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_verdict(txt):
    v = re.findall(r"VERDICT:\s*(MOUND|NOT[_ ]?MOUND)", txt, re.I)
    if v:
        return "NOT_MOUND" if "NOT" in v[-1].upper() else "MOUND"
    ans = txt.split("</think>")[-1] if "</think>" in txt else txt
    tail = ans[-600:].lower()
    neg = sum(tail.count(k) for k in ("not a mound", "modern", "natural", "spoil", "levee",
              "not a pre-european", "unlikely to be a mound", "agricultural"))
    pos = sum(tail.count(k) for k in ("is a mound", "likely a mound", "pre-european mound",
              "platform mound", "conical mound", "appears to be a mound"))
    if neg > pos and neg > 0:
        return "NOT_MOUND"
    if pos > neg and pos > 0:
        return "MOUND"
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = load_eval_set()
    print(f"eval set: {sum(s['label']==1 for s in sites)} positives, "
          f"{sum(s['label']==0 for s in sites)} negatives | arm B: detail + 600 m context")
    results = []
    for i, s in enumerate(sites, 1):
        t0 = time.time()
        try:
            dem = clean(fetch_dem(s["x"], s["y"], DETAIL_PX, crs_epsg=UTM_EPSG, resolution_m=1.0))
            geo = classify_geomorphon_simple(dem)
            detail = make_multi_view_panel(dem, geo)
            wide = wide_panel(s["x"], s["y"])
            votes = []
            for _ in range(K):
                vv = parse_verdict(query(detail, wide))
                if vv:
                    votes.append(vv)
            nm, nn = votes.count("MOUND"), votes.count("NOT_MOUND")
            verdict = "MOUND" if nm > nn else ("NOT_MOUND" if nn > nm else None)
            pred = 1 if verdict == "MOUND" else (0 if verdict == "NOT_MOUND" else None)
            ok = "OK" if pred == s["label"] else ("?" if pred is None else "X")
            print(f"[{i}/{len(sites)}] {ok} {s['kind']:16} {s['name'][:26]:26} "
                  f"-> {verdict} (M{nm}/N{nn})  {time.time()-t0:.0f}s", flush=True)
            results.append({**{k: s[k] for k in ('site_id','name','kind','label')},
                            "pred": pred, "verdict": verdict, "votes_mound": nm,
                            "votes_notmound": nn, "runs": K})
        except Exception as e:
            print(f"[{i}/{len(sites)}] ERR {s['name'][:26]}: {type(e).__name__} {e}", flush=True)
            results.append({**{k: s[k] for k in ('site_id','name','kind','label')},
                            "pred": None, "verdict": "ERROR", "votes_mound": 0,
                            "votes_notmound": 0, "runs": K})
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "ablation_context_results.csv", index=False)
    ev = df[df.pred.notna()]
    y, p = ev.label.astype(int), ev.pred.astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    print("\n===== Arm B: VLM with 600 m context view =====")
    print(f"  confusion: TP={tp} FN={fn} TN={tn} FP={fp}")
    print(f"  mound recall:               {tp}/{tp+fn}")
    print(f"  modern-earthwork rejection: {tn}/{tn+fp}   (arm A was 3/22)")
    (OUT_DIR / "summary.json").write_text(json.dumps(dict(tp=tp, fn=fn, tn=tn, fp=fp), indent=1))


if __name__ == "__main__":
    main()
