"""Evaluate a served adapter with the V10 composite protocol.

Uses the exact format V10 was trained on: ONE composite image (6-panel 160 m
detail on top, 600 m wide context view below) + the rubric prompt, K votes,
majority. Two eval sets:

  --set eskew   28 Yazoo sites (6 confirmed mounds vs 22 modern earthworks)
  --set sc      20 SC points (10 hand-corrected ring centers vs 10 land decoys)

Env: VLM_API, VLM_MODEL, VLM_RUNS (default 5), VLM_TEMP (default 0.6).
"""
from __future__ import annotations
import sys
import os
import io
import re
import csv
import time
import math
import argparse
import numpy as np
import requests
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
from pyproj import Transformer

sys.path.insert(0, "scripts")
from demo_terrain_query import classify_geomorphon_simple, make_multi_view_panel, pil_to_b64_data_uri
from earthwork_llm.ingestion.imageserver import fetch_dem
from scipy.ndimage import gaussian_filter
from generate_discrimination_v10 import PROMPT, composite  # trained prompt + layout

API = os.environ.get("VLM_API", "http://localhost:8001/v1/chat/completions")
MODEL = os.environ.get("VLM_MODEL", "terrallm-v10")
K = int(os.environ.get("VLM_RUNS", "5"))
TEMP = float(os.environ.get("VLM_TEMP", "0.6"))
DETAIL_PX = 160
WIDE_M = 600


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


def wide_view(x, y, epsg):
    dem = clean(fetch_dem(x, y, WIDE_M, crs_epsg=epsg, resolution_m=1.0))
    hs8 = (hillshade(dem) * 255).astype(np.uint8)
    rel = dem - gaussian_filter(dem, 45)
    v = np.nanpercentile(np.abs(rel), 98) or 1.0
    import matplotlib.cm as cm
    rel_rgb = (cm.RdBu_r(np.clip((rel / v + 1) / 2, 0, 1))[:, :, :3] * 255).astype(np.uint8)
    combo = np.concatenate([np.stack([hs8] * 3, axis=-1), rel_rgb], axis=1)
    img = Image.fromarray(combo)
    d = ImageDraw.Draw(img)
    for cx in (WIDE_M // 2, WIDE_M + WIDE_M // 2):
        d.ellipse([cx - 12, WIDE_M // 2 - 12, cx + 12, WIDE_M // 2 + 12],
                  outline=(255, 255, 0), width=3)
    return img


def make_composite(x, y, epsg):
    dem = clean(fetch_dem(x, y, DETAIL_PX, crs_epsg=epsg, resolution_m=1.0))
    detail = make_multi_view_panel(dem, classify_geomorphon_simple(dem))
    return composite(detail, wide_view(x, y, epsg))


def query(img):
    payload = {"model": MODEL, "messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": pil_to_b64_data_uri(img)}},
            {"type": "text", "text": PROMPT}]}],
        "max_tokens": 3072, "temperature": TEMP}
    r = requests.post(API, json=payload, timeout=900)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse(txt):
    v = re.findall(r"VERDICT:\s*(MOUND|NOT[_ ]?MOUND)", txt, re.I)
    if v:
        return "NOT_MOUND" if "NOT" in v[-1].upper() else "MOUND"
    return None


def eskew_set():
    rows = [r for r in csv.DictReader(io.StringIO("".join(
        ln for ln in open(os.environ.get("EARTHWORK_ABLATION_SET", "data/reference/mounds_seed.csv"))
        if not ln.lstrip().startswith("#"))))]
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
            out.append((r["site_name"], x, y, 26915, 0))
        elif conf in ("high", "refined") and ft in (
            "mound_group", "village_with_mounds", "large_village_with_mounds", "platform_mound"):
            out.append((r["site_name"], x, y, 26915, 1))
    return out


def sc_set():
    tf = Transformer.from_crs("EPSG:4326", "EPSG:26917", always_xy=True)
    out = []
    with open(os.environ.get("SHELL_RING_GOLD", "data/reference/SC_Gold_List_corrected.csv")) as f:
        for r in csv.DictReader(f):
            if not r.get("mound_id"):
                continue
            x, y = tf.transform(float(r["longitude"]), float(r["latitude"]))
            out.append((f"ring:{r['mound_id'].strip()}", x, y, 26917, 1))
            for dx, dy in ((500, 0), (-500, 0), (0, 500), (0, -500)):
                try:
                    dem = fetch_dem(x + dx, y + dy, 100, crs_epsg=26917, resolution_m=1.0)
                    m = np.isfinite(dem)
                    if m.mean() < 0.7:
                        continue
                    d = np.where(m, dem, np.nanmedian(dem[m]))
                    if float(np.nanpercentile(d, 95) - np.nanpercentile(d, 5)) < 0.5:
                        continue
                    out.append((f"decoy:{r['mound_id'].strip()}", x + dx, y + dy, 26917, 0))
                    break
                except Exception:
                    continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["eskew", "sc"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sites = eskew_set() if args.set == "eskew" else sc_set()
    print(f"{args.set}: {len(sites)} points, model={MODEL}, K={K}, T={TEMP}")
    results = []
    for i, (name, x, y, epsg, label) in enumerate(sites, 1):
        t0 = time.time()
        try:
            img = make_composite(x, y, epsg)
            votes = [v for v in (parse(query(img)) for _ in range(K)) if v]
            nm, nn = votes.count("MOUND"), votes.count("NOT_MOUND")
            verdict = "MOUND" if nm > nn else ("NOT_MOUND" if nn > nm else None)
            pred = 1 if verdict == "MOUND" else (0 if verdict == "NOT_MOUND" else None)
            ok = "OK" if pred == label else ("?" if pred is None else "X")
            print(f"[{i}/{len(sites)}] {ok} {name[:26]:26} -> {verdict} (M{nm}/N{nn})  {time.time()-t0:.0f}s", flush=True)
            results.append(dict(name=name, label=label, pred=pred, verdict=verdict,
                                votes_mound=nm, votes_not=nn))
        except Exception as e:
            print(f"[{i}/{len(sites)}] ERR {name[:26]}: {type(e).__name__} {e}", flush=True)
            results.append(dict(name=name, label=label, pred=None, verdict="ERROR",
                                votes_mound=0, votes_not=0))
    df = pd.DataFrame(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    ev = df[df.pred.notna()]
    y, p = ev.label.astype(int), ev.pred.astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    print(f"\n===== {args.set} with {MODEL} =====")
    print(f"  confusion: TP={tp} FN={fn} TN={tn} FP={fp}")
    print(f"  positive recall:    {tp}/{tp+fn}")
    print(f"  negative rejection: {tn}/{tn+fp}")


if __name__ == "__main__":
    main()
