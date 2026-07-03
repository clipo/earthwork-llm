"""Occlusion test for the discrimination fine-tune (B.3).

Re-runs the composite-protocol Eskew evaluation with the 600 m context block
of the composite REPLACED by neutral gray. If the fine-tune's rejections
vanish without the context view, the learned filter is reading modern
construction in the wide view rather than anything intrinsic to the candidate,
which separates the "learned age" and "learned context" readings of the
fifth arm.

Same env vars as vlm_eval_v10.py. Writes --out CSV in the same format.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, "/home/clipo/projects/terrallm")
sys.path.insert(0, "/home/clipo/projects/terrallm/scripts")
sys.path.insert(0, "/home/clipo/projects/earthwork-llm/src")

import numpy as np
from PIL import Image
import vlm_eval_v10 as base


def occluded_composite(x, y, epsg):
    dem = base.clean(base.fetch_dem(x, y, base.DETAIL_PX, crs_epsg=epsg, resolution_m=1.0))
    from demo_terrain_query import classify_geomorphon_simple, make_multi_view_panel
    detail = make_multi_view_panel(dem, classify_geomorphon_simple(dem))
    wide = base.wide_view(x, y, epsg)
    gray = Image.new("RGB", wide.size, (128, 128, 128))
    from generate_discrimination_v10 import composite
    return composite(detail, gray)


base.make_composite = occluded_composite

if __name__ == "__main__":
    base.main()
