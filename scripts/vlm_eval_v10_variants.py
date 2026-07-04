"""Occlusion/transplant variants for the discrimination fine-tune (B.3).

Modes (--mode):
  detailmask  gray the 160 m six-panel DETAIL block, keep the 600 m context:
              if rejections persist, context is sufficient; the round-6
              positive demonstration.
  ctxquiet    every site gets the same QUIET donor context (Glover Place):
              rejections should vanish if verdicts follow context.
  ctxbusy     every site gets the same CONSTRUCTION-HEAVY donor context
              (Eskew Ground-Survey 01, unanimously rejected): rejections
              should spread to mounds if verdicts follow context.

Same env vars as vlm_eval_v10.py; --set eskew only.
"""
from __future__ import annotations
import sys, argparse

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from PIL import Image
import vlm_eval_v10 as base
from demo_terrain_query import classify_geomorphon_simple, make_multi_view_panel

DONORS = {"ctxquiet": "Glover Place", "ctxbusy": "Eskew Ground-Survey 01"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["detailmask", "ctxquiet", "ctxbusy"], required=True)
    ap.add_argument("--set", dest="which", default="eskew")
    ap.add_argument("--out", required=True)
    args, rest = ap.parse_known_args()
    from generate_discrimination_v10 import composite

    donor_wide = None
    if args.mode in DONORS:
        name = DONORS[args.mode]
        for n, x, y, epsg, label in base.eskew_set():
            if n == name:
                donor_wide = base.wide_view(x, y, epsg)
                break
        assert donor_wide is not None, f"donor {name} not found"

    def variant_composite(x, y, epsg):
        if args.mode == "detailmask":
            dem = base.clean(base.fetch_dem(x, y, base.DETAIL_PX, crs_epsg=epsg, resolution_m=1.0))
            detail = make_multi_view_panel(dem, classify_geomorphon_simple(dem))
            gray = Image.new("RGB", detail.size, (128, 128, 128))
            return composite(gray, base.wide_view(x, y, epsg))
        dem = base.clean(base.fetch_dem(x, y, base.DETAIL_PX, crs_epsg=epsg, resolution_m=1.0))
        detail = make_multi_view_panel(dem, classify_geomorphon_simple(dem))
        return composite(detail, donor_wide)

    base.make_composite = variant_composite
    sys.argv = [sys.argv[0], "--set", args.which, "--out", args.out]
    base.main()


if __name__ == "__main__":
    main()
