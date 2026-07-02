"""Strongest zero-shot arm (arm C of the Section 3.6 experiment).

Combines every no-training improvement identified by the earlier arms:
  - detail six-panel image (160 m) + wide context view (600 m), as in arm B
  - mapped-modern-feature context as TEXT (OpenStreetMap nearest feature +
    distance, from the shield proximity run) - the deployed pipeline's
    context-correlation input, exercised in the ablation for the first time
  - a structured rubric the model must answer before its verdict
  - nine votes per site at an explicit sampling temperature (0.6), majority
    rule - honest sampling instead of vLLM temperature-zero batching noise

Same 28-site Eskew set; comparable to arms A (image-only) and B (image+wide).

Env: VLM_API, VLM_MODEL, VLM_RUNS (default 9), VLM_TEMP (default 0.6),
EARTHWORK_ABLATION_SET (path to the restricted eval CSV; not distributed),
SHIELD_PROXIMITY_CSV (output of shield_eskew_proximity.py).
"""
from __future__ import annotations
import sys, os, io, re, csv, json, time, math
import numpy as np, requests, pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw

from demo_terrain_query import classify_geomorphon_simple, make_multi_view_panel, pil_to_b64_data_uri
from earthwork_query import SYSTEM_PROMPT
from earthwork_llm.ingestion.imageserver import fetch_dem
from scipy.ndimage import gaussian_filter

API = os.environ.get("VLM_API", "http://localhost:8001/v1/chat/completions")
MODEL = os.environ.get("VLM_MODEL", "terrallm-v91")
UTM_EPSG = 26915
DETAIL_PX = 160
WIDE_M = 600
K = int(os.environ.get("VLM_RUNS", "9"))
TEMP = float(os.environ.get("VLM_TEMP", "0.6"))
SEED_CSV = os.environ.get("EARTHWORK_ABLATION_SET", "data/reference/mounds_seed.csv")
PROX_CSV = os.environ.get("SHIELD_PROXIMITY_CSV", "data/shield_eskew/shield_eskew_proximity.csv")
OUT_DIR = Path("data/vlm_ablation_armc")

RUBRIC = (
    "\n\nYou are given TWO images of the same location. Image 1 is a 6-panel detail view "
    "(160 m across: 4 hillshades | geomorphon | contours) centred on a candidate feature. "
    "Image 2 is a WIDE context view (600 m across: hillshade left, local relief right; "
    "candidate circled). {ctx}\n"
    "Before deciding, answer these three checks in one line each:\n"
    "CHECK-SHAPE: is the central feature a compact, roughly symmetric positive rise? (yes/no + why)\n"
    "CHECK-CONTEXT: in the wide view, does it sit on or beside a linear modern feature "
    "(canal, ditch, drain, road, rail, levee) or a field edge whose construction or maintenance "
    "would produce a spoil pile of this size? (yes/no + why)\n"
    "CHECK-AGE: does its form suggest long erosion (softened, spreading flanks) or recent "
    "machine work (sharp, uniform, aligned with modern features)? (eroded/recent + why)\n"
    "A pre-European mound is compact and symmetric, stands apart from modern linear works, and "
    "looks eroded. A modern spoil pile or berm is beside the linear feature that produced it and "
    "looks recent. End with exactly two lines:\n"
    "VERDICT: MOUND        (pre-European mound)\n"
    "or VERDICT: NOT_MOUND (natural or modern)\n"
    "CONFIDENCE: <integer 0-100>"
)


def load_eval_set():
    with open(SEED_CSV) as f:
        lines = [l for l in f if not l.lstrip().startswith("#")]
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


def load_proximity():
    df = pd.read_csv(PROX_CSV)
    out = {}
    for _, r in df.iterrows():
        if pd.notna(r.get("nearest_modern_m")):
            out[r["site"]] = (float(r["nearest_modern_m"]), str(r["feature"]))
    return out


def clean(dem):
    m = np.isfinite(dem)
    return np.where(m, dem, np.nanmedian(dem[m])).astype("float32")


def hillshade(dem, az=315):
    gy, gx = np.gradient(dem)
    sl = np.arctan(np.hypot(gy, gx)); asp = np.arctan2(-gx, gy)
    z = math.radians(45); a = math.radians(az)
    return np.clip(np.cos(z) * np.cos(sl) + np.sin(z) * np.sin(sl) * np.cos(a - asp), 0, 1)


def wide_panel(x, y):
    dem = clean(fetch_dem(x, y, WIDE_M, crs_epsg=UTM_EPSG, resolution_m=1.0))
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


def query(detail, wide, ctx_text):
    payload = {"model": MODEL, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": pil_to_b64_data_uri(detail)}},
            {"type": "image_url", "image_url": {"url": pil_to_b64_data_uri(wide)}},
            {"type": "text", "text": "Analyse the terrain." + RUBRIC.format(ctx=ctx_text)}]}],
        "max_tokens": 4096, "temperature": TEMP}
    r = requests.post(API, json=payload, timeout=900)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse(txt):
    v = re.findall(r"VERDICT:\s*(MOUND|NOT[_ ]?MOUND)", txt, re.I)
    if v:
        return "NOT_MOUND" if "NOT" in v[-1].upper() else "MOUND"
    ans = txt.split("</think>")[-1] if "</think>" in txt else txt
    tail = ans[-500:].lower()
    neg = sum(tail.count(k) for k in ("not a mound", "modern", "spoil", "recent", "not_mound"))
    pos = sum(tail.count(k) for k in ("is a mound", "pre-european mound", "mound\n"))
    if neg > pos and neg > 0:
        return "NOT_MOUND"
    if pos > neg and pos > 0:
        return "MOUND"
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = load_eval_set()
    prox = load_proximity()
    print(f"arm C: {len(sites)} sites, K={K} votes at T={TEMP}, wide view + OSM context text + rubric")
    results = []
    for i, s in enumerate(sites, 1):
        t0 = time.time()
        try:
            dem = clean(fetch_dem(s["x"], s["y"], DETAIL_PX, crs_epsg=UTM_EPSG, resolution_m=1.0))
            detail = make_multi_view_panel(dem, classify_geomorphon_simple(dem))
            wide = wide_panel(s["x"], s["y"])
            if s["name"] in prox:
                d, f = prox[s["name"]]
                ctx = (f"Mapped modern features near this point (OpenStreetMap): "
                       f"nearest is a {f} at {d:.0f} m.")
            else:
                ctx = "Mapped modern features near this point (OpenStreetMap): none within 300 m."
            votes = []
            for _ in range(K):
                vv = parse(query(detail, wide, ctx))
                if vv:
                    votes.append(vv)
            nm, nn = votes.count("MOUND"), votes.count("NOT_MOUND")
            verdict = "MOUND" if nm > nn else ("NOT_MOUND" if nn > nm else None)
            pred = 1 if verdict == "MOUND" else (0 if verdict == "NOT_MOUND" else None)
            ok = "OK" if pred == s["label"] else ("?" if pred is None else "X")
            print(f"[{i}/{len(sites)}] {ok} {s['kind']:16} {s['name'][:24]:24} "
                  f"-> {verdict} (M{nm}/N{nn} of {K})  {time.time()-t0:.0f}s", flush=True)
            results.append({**{k: s[k] for k in ('site_id','name','kind','label')},
                            "pred": pred, "verdict": verdict, "votes_mound": nm,
                            "votes_notmound": nn, "runs": K})
        except Exception as e:
            print(f"[{i}/{len(sites)}] ERR {s['name'][:24]}: {type(e).__name__} {e}", flush=True)
            results.append({**{k: s[k] for k in ('site_id','name','kind','label')},
                            "pred": None, "verdict": "ERROR", "votes_mound": 0,
                            "votes_notmound": 0, "runs": K})
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "armc_results.csv", index=False)
    ev = df[df.pred.notna()]
    y, p = ev.label.astype(int), ev.pred.astype(int)
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    print("\n===== Arm C: wide view + OSM context text + rubric + 9-vote majority =====")
    print(f"  confusion: TP={tp} FN={fn} TN={tn} FP={fp}")
    print(f"  mound recall:               {tp}/{tp+fn}")
    print(f"  modern-earthwork rejection: {tn}/{tn+fp}   (arm A 3/22, arm B pooled 19/63)")
    (OUT_DIR / "summary.json").write_text(json.dumps(dict(tp=tp, fn=fn, tn=tn, fp=fp, K=K, temp=TEMP), indent=1))


if __name__ == "__main__":
    main()
