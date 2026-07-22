"""VLM ablation: does the fine-tuned interpretation layer separate real
pre-European mounds from the modern earthworks that a morphology-only detector
also flags? (Manuscript Section 3.6.)

Ground truth is Eskew's (2008) ground-survey set. Her contour-circularity
algorithm flagged candidate earthworks; field verification (Eskew, Lipo, Hunt,
June 2008) sorted them into confirmed prehistoric mounds and modern earthworks.
We add Winterville (open-terrain positive) and canopy-blind Lake George.

For each site we fetch a bare-earth DEM (UTM 15N), build the same six-panel
image the pipeline serves, ask the served model for a MOUND / NOT_MOUND verdict,
and score its calls against truth by majority vote over K runs.

The eval set carries site coordinates and is RESTRICTED (see docs/DATA_POLICY.md).
It is not shipped. Provide your own via the EARTHWORK_ABLATION_SET environment
variable (same schema as mounds_seed.csv: site_id, site_name, utm15n_x_m,
utm15n_y_m, feature_type, confidence). The published-name verdict results
(data/vlm_ablation/ablation_results.csv) contain no coordinates.

Requires a served model, e.g. via scripts/serve_yazoo_model.sh; point at it with
VLM_API / VLM_MODEL. Runs on CPU for the DEM/geomorphon work; the model needs a GPU.
"""
from __future__ import annotations
import sys
import os
import io
import re
import csv
import json
import time
import argparse
import numpy as np
import requests
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling scripts
from demo_terrain_query import classify_geomorphon_simple, make_multi_view_panel, pil_to_b64_data_uri
from earthwork_query import SYSTEM_PROMPT
from earthwork_llm.ingestion.imageserver import fetch_dem

API = os.environ.get("VLM_API", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("VLM_MODEL", "terrallm-v91")
UTM_EPSG = 26915
WIN_PX = 160          # ~160 m window centred on the feature
SEED_CSV = os.environ.get("EARTHWORK_ABLATION_SET", "data/reference/mounds_seed.csv")  # RESTRICTED, not shipped
OUT_DIR = Path("data/vlm_ablation")

VERDICT_INSTR = (
    "\n\nThis 6-panel image is centred on a single candidate feature "
    "(4 hillshades | geomorphon | contours). Decide whether the central feature "
    "is a pre-European earthen mound or not (natural landform or modern earthwork "
    "such as levee/spoil/road/field berm). End your reply with exactly two lines:\n"
    "VERDICT: MOUND        (if most likely a pre-European mound)\n"
    "or VERDICT: NOT_MOUND (if natural or modern)\n"
    "CONFIDENCE: <integer 0-100>"
)


def load_eval_set():
    """Positives = field-confirmed mounds; negatives = field-confirmed modern earthworks."""
    if not Path(SEED_CSV).exists():
        raise SystemExit(
            f"Eval set not found at {SEED_CSV}. This set is restricted and not shipped "
            "(see docs/DATA_POLICY.md); set EARTHWORK_ABLATION_SET to your copy.")
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
            continue  # skip 'uncertain' / ambiguous
        out.append(dict(site_id=r["site_id"], name=r["site_name"], x=x, y=y,
                        label=label, kind=kind))
    return out


def build_panel(x, y):
    dem = fetch_dem(x, y, WIN_PX, crs_epsg=UTM_EPSG, resolution_m=1.0)
    m = np.isfinite(dem)
    if m.sum() < 0.5 * dem.size:
        raise RuntimeError("DEM mostly nodata")
    dem = np.where(m, dem, np.nanmedian(dem[m])).astype("float32")
    geo = classify_geomorphon_simple(dem)
    return make_multi_view_panel(dem, geo)


def query_vlm(panel):
    payload = {"model": MODEL, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": pil_to_b64_data_uri(panel)}},
            {"type": "text", "text": "Analyse the terrain." + VERDICT_INSTR}]}],
        "max_tokens": 4096, "temperature": 0}
    r = requests.post(API, json=payload, timeout=900)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_verdict(txt):
    # primary: the explicit VERDICT line
    v = re.findall(r"VERDICT:\s*(MOUND|NOT[_ ]?MOUND)", txt, re.I)
    c = re.findall(r"CONFIDENCE:\s*(\d+)", txt, re.I)
    conf = int(c[-1]) if c else None
    if v:
        return ("NOT_MOUND" if "NOT" in v[-1].upper() else "MOUND"), conf
    # fallback: read the stated conclusion (answer after </think> if present)
    ans = txt.split("</think>")[-1] if "</think>" in txt else txt
    tail = ans[-600:].lower()
    neg = sum(tail.count(k) for k in ("not a mound", "not an earthwork", "modern", "natural",
              "not a pre-european", "not cultural", "unlikely to be a mound", "not likely a mound",
              "agricultural", "levee", "spoil", "no clear mound"))
    pos = sum(tail.count(k) for k in ("is a mound", "likely a mound", "pre-european mound",
              "platform mound", "conical mound", "cultural earthwork", "is an earthwork",
              "appears to be a mound", "consistent with a mound"))
    if neg > pos and neg > 0:
        return "NOT_MOUND", conf
    if pos > neg and pos > 0:
        return "MOUND", conf
    return None, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N sites (smoke test)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = load_eval_set()
    if args.limit:
        pos = [s for s in sites if s["label"] == 1][:max(1, args.limit // 2)]
        neg = [s for s in sites if s["label"] == 0][:max(1, args.limit // 2)]
        sites = pos + neg
    print(f"eval set: {sum(s['label']==1 for s in sites)} positives, "
          f"{sum(s['label']==0 for s in sites)} negatives")
    K = int(os.environ.get("VLM_RUNS", "3"))     # runs per site, majority vote
    results = []
    for i, s in enumerate(sites, 1):
        t0 = time.time()
        try:
            panel = build_panel(s["x"], s["y"])
            votes = []
            for _ in range(K):
                vv, _ = parse_verdict(query_vlm(panel))
                if vv:
                    votes.append(vv)
            nm, nn = votes.count("MOUND"), votes.count("NOT_MOUND")
            verdict = "MOUND" if nm > nn else ("NOT_MOUND" if nn > nm else None)
            pred = 1 if verdict == "MOUND" else (0 if verdict == "NOT_MOUND" else None)
            ok = "OK" if pred == s["label"] else ("?" if pred is None else "X")
            print(f"[{i}/{len(sites)}] {ok} {s['kind']:16} {s['name'][:26]:26} "
                  f"-> {verdict} (M{nm}/N{nn} of {K})  {time.time()-t0:.0f}s")
            results.append({**{k: s[k] for k in ('site_id','name','kind','label')},
                            "pred": pred, "verdict": verdict, "votes_mound": nm,
                            "votes_notmound": nn, "runs": K})
        except Exception as e:
            print(f"[{i}/{len(sites)}] ERR {s['name'][:26]}: {type(e).__name__} {e}")
            results.append({**{k: s[k] for k in ('site_id','name','kind','label')},
                            "pred": None, "verdict": "ERROR", "votes_mound": 0,
                            "votes_notmound": 0, "runs": K})
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "ablation_results.csv", index=False)
    score(df)


def score(df):
    ev = df[df.pred.notna()]
    y, p = ev.label.astype(int), ev.pred.astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    n = len(ev)
    acc = (tp + tn) / n if n else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")       # mound recall
    spec = tn / (tn + fp) if (tn + fp) else float("nan")      # modern-earthwork rejection
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    print("\n===== VLM as discriminator =====")
    print(f"scored {n} of {len(df)} sites ({len(df)-n} unparsed/error)")
    print(f"  confusion:  TP={tp}  FN={fn}  TN={tn}  FP={fp}")
    print(f"  accuracy                       {acc:.2f}")
    print(f"  mound recall  (TP/(TP+FN))     {rec:.2f}")
    print(f"  modern-earthwork rejection     {spec:.2f}   (specificity, TN/(TN+FP))")
    print(f"  precision     (TP/(TP+FP))     {prec:.2f}")
    summ = dict(n=n, unscored=int(len(df)-n), tp=tp, fn=fn, tn=tn, fp=fp,
                accuracy=acc, mound_recall=rec, modern_rejection=spec, precision=prec)
    (OUT_DIR / "ablation_summary.json").write_text(json.dumps(summ, indent=2))
    print("wrote", OUT_DIR / "ablation_results.csv", "and ablation_summary.json")


if __name__ == "__main__":
    main()
