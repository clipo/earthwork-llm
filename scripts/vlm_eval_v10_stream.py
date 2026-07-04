"""Evaluate the discrimination fine-tune on an arbitrary candidate stream (B.3).

Round-7 adversarial conditions 2 and 3: (a) does the fine-tune's flag-ordering
enrich the desk-review plausibles among the 80 Jaketown survivors (the
"ranking aid" claim, measured); (b) what does it do to the cropland
agricultural-island keeps (the island-rule collision test).

--csv     input with lat/lon columns (WGS84) and an id column
--id-col / --lat-col / --lon-col   column names
--out     per-candidate output (id, votes, share, verdict)

Env as vlm_eval_v10.py (VLM_MODEL, VLM_RUNS, VLM_TEMP). CRS: converts to
UTM 15N for the DEM fetch (both streams are in the Yazoo Basin).
"""
from __future__ import annotations
import sys, argparse, time
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, "/home/clipo/projects/terrallm")
sys.path.insert(0, "/home/clipo/projects/terrallm/scripts")
sys.path.insert(0, "/home/clipo/projects/earthwork-llm/src")
import vlm_eval_v10 as base

TF = Transformer.from_crs("EPSG:4326", "EPSG:26915", always_xy=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--lat-col", default="lat")
    ap.add_argument("--lon-col", default="lon")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    rows = []
    for i, r in enumerate(df.itertuples(), 1):
        cid = getattr(r, args.id_col)
        lat, lon = getattr(r, args.lat_col), getattr(r, args.lon_col)
        x, y = TF.transform(lon, lat)
        t0 = time.time()
        try:
            img = base.make_composite(x, y, 26915)
            votes = [v for v in (base.parse(base.query(img)) for _ in range(base.K)) if v]
            nm, nn = votes.count("MOUND"), votes.count("NOT_MOUND")
            verdict = "MOUND" if nm > nn else ("NOT_MOUND" if nn > nm else "TIE")
            share = nm / (nm + nn) if (nm + nn) else None
            print(f"[{i}/{len(df)}] {cid}: {verdict} (M{nm}/N{nn})  {time.time()-t0:.0f}s", flush=True)
            rows.append(dict(id=cid, votes_mound=nm, votes_not=nn, share=share, verdict=verdict))
        except Exception as e:
            print(f"[{i}/{len(df)}] {cid}: ERR {type(e).__name__}", flush=True)
            rows.append(dict(id=cid, votes_mound=0, votes_not=0, share=None, verdict="ERROR"))
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
