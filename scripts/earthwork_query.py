#!/usr/bin/env python3
"""TerraLLM Earthwork Query CLI.

Specialized for detecting pre-European earthworks in the Mississippi River Valley 
(Yazoo Basin). Uses zero-shot reasoning to distinguish between ancient earthworks 
and modern land modifications.

Usage:
    python scripts/earthwork_query.py \
        --query "Find pre-European mounds near the river" \
        --lidar /path/to/yazoo_tile.las \
        --out figures/yazoo_earthworks.png
"""

import argparse
import base64
import json
import logging
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from scipy.ndimage import gaussian_filter

# Import demo components (reusing loading/rasterization logic)
from demo_terrain_query import (
    load_ground_points, 
    rasterize_dem, 
    load_dem_geotiff, 
    make_hillshade, 
    classify_geomorphon_simple,
    make_composite_panel,
    make_multi_view_panel,
    pil_to_b64_data_uri,
    strip_thinking,
    render_overlay
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("earthwork")

# Specialized Earthwork Keyword Features for the deterministic detector
# Format: (geomorphon_classes, min_area_m2, max_area_m2, prob_base)
EARTHWORK_KEYWORDS = {
    "mound":            ({1, 2, 3, 4}, 20.0, 5000.0, 1.0),   # PEAK/RIDGE/SHOULDER/SPUR
    "platform mound":   ({0, 1, 3, 4}, 100.0, 10000.0, 1.0), # FLAT/PEAK/SHOULDER/SPUR
    "conical mound":    ({1, 3, 4},    10.0, 500.0, 1.0),    # PEAK/SHOULDER/SPUR
    "enclosure":        ({2, 3, 4},    100.0, 10000.0, 1.0), # RIDGE/SHOULDER/SPUR (linear)
    "earthwork":        ({1, 2, 3, 4}, 10.0, 10000.0, 1.0),
}

SYSTEM_PROMPT = """You are an expert archaeologist specializing in pre-European earthworks of the Mississippi River Valley, specifically the Yazoo Basin.

The user is providing an enhanced 6-panel terrain analysis image (4 multi-angle hillshades | geomorphon | contours). A deterministic detector has identified candidate locations.

YOUR TASK:
1. Analyze the terrain for pre-European earthworks (Mississippian culture platform mounds, conical burial mounds, or geometric enclosures).
2. Use the multi-angle hillshades to detect subtle features that might be hidden by specific sun angles.
3. DISTINGUISH ancient earthworks from modern/historic land modifications:
   - MODERN CANAL DREDGING: Often produces highly linear, regular spoil banks (RIDGE/PEAK) immediately adjacent to linear ditches (VALLEY).
   - ROAD WORK: Regular linear embankments (RIDGE) or cuts (VALLEY) following straight or smoothly curved paths.
   - LEVEES: Long, continuous, extremely regular linear RIDGE features along water bodies.
   - IRRIGATION/DRAINAGE: Small, regular, linear ditches and associated berms.
   - HISTORIC DISTURBANCE: Spoil piles from historic logging, borrow pits, or abandoned railroad grades.
   - ANCIENT EARTHWORKS: Often show subtle erosion, may be grouped in complexes, platform mounds are often rectangular with distinct flat tops, conical mounds are circular.

4. VISUAL CORRELATION:
   - For each candidate, check if its morphology matches its "justification" or any provided "nearby noise" context.
   - If a candidate is near a mapped canal or levee, look for the ditch-and-bank pattern.
   - If a candidate is near a road, check if it aligns with the road grade.

5. In 3-5 sentences:
   - Identify the most promising ancient earthwork candidates.
   - Explicitly assess whether detected features are more likely to be cultural (ancient) or related to modern/historic construction (dredging, roads, levees, irrigation).
   - Provide a confidence assessment based on morphological regularity and context.

Be professional and archaeological in your tone. Do NOT emit coordinate lists."""

def detect_earthworks(geo: np.ndarray, dem: np.ndarray, query: str, 
                      max_candidates: int = 10) -> List[dict]:
    """Similar to demo detector but with earthwork-specific parameters."""
    from scipy.ndimage import label, center_of_mass
    q = query.lower()
    matched = None
    for kw in sorted(EARTHWORK_KEYWORDS, key=lambda s: -len(s)):
        if kw in q:
            matched = kw
            break
    if matched is None:
        matched = "earthwork"
    
    classes, amin, amax, _ = EARTHWORK_KEYWORDS[matched]
    mask = np.isin(geo, list(classes))
    labels, n = label(mask)
    
    candidates = []
    h, w = geo.shape
    for i in range(1, n + 1):
        comp_mask = labels == i
        area = comp_mask.sum()
        if area < amin or area > amax:
            continue
        
        # Linearity Check (Eccentricity/Aspect Ratio)
        # Ancient mounds are generally circular or rectangular, not extremely long/thin
        rows, cols = np.where(comp_mask)
        if len(rows) < 5: continue
        
        # Calculate aspect ratio of bounding box as a simple linearity proxy
        width = cols.max() - cols.min() + 1
        height = rows.max() - rows.min() + 1
        aspect = max(width, height) / max(1.0, min(width, height))
        
        # If very linear (e.g., aspect > 4.0), it's likely a levee, road, or canal bank
        # unless specifically looking for enclosures.
        if aspect > 4.0 and matched != "enclosure":
            # Reduce probability significantly for linear noise
            p_base = 0.2
            just_prefix = "Linear noise? "
        else:
            p_base = 0.7
            just_prefix = ""

        cy, cx = center_of_mass(comp_mask)
        x, y = int(round(cx)), int(round(cy))
        
        local = dem[max(0, y-20):min(h, y+20), max(0, x-20):min(w, x+20)]
        height_val = float(dem[y, x] - local.mean())
        
        candidates.append(dict(
            x=x, y=y, p=p_base, 
            justification=f"{just_prefix}{matched}: area {int(area)} m², height {height_val:+.2f} m, aspect {aspect:.1f}",
            area=int(area), height=height_val, aspect=aspect
        ))
    
    # Sort by probability, then area
    candidates.sort(key=lambda c: (-c["p"], -c["area"]))
    return candidates[:max_candidates]

def query_earthwork_v8(api_url: str, model: str, query: str, panel: Image.Image,
                       candidates: List[dict], context_list: Optional[List[str]] = None) -> str:
    cand_text = "Detector candidates:\n"
    for i, c in enumerate(candidates, 1):
        context = f" [Nearby Noise: {context_list[i-1]}]" if context_list and i-1 < len(context_list) else ""
        cand_text += f"  {i}. ({c['x']}, {c['y']}) {c['justification']}{context}\n"
    
    user_text = f"Question: {query}\n\n{cand_text}\nPlease provide archaeological interpretation."
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": pil_to_b64_data_uri(panel)}},
                {"type": "text", "text": user_text},
            ]},
        ],
        "max_tokens": 1024,
        "temperature": 0.4,
    }
    
    r = requests.post(api_url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--lidar")
    parser.add_argument("--dem-tif")
    parser.add_argument("--center-x", type=float)
    parser.add_argument("--center-y", type=float)
    parser.add_argument("--out", required=True)
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="terrallm-v91")
    parser.add_argument("--noise-map", help="Optional GeoJSON noise map (modern features)")
    parser.add_argument("--crop-m", type=float, default=250.0) # Larger crop for earthworks
    parser.add_argument("--resolution-m", type=float, default=1.0)
    args = parser.parse_args()

    if args.dem_tif:
        dem, meta = load_dem_geotiff(
            Path(args.dem_tif),
            center_x=args.center_x, center_y=args.center_y,
            crop_m=args.crop_m, target_res_m=args.resolution_m,
        )
    elif args.lidar:
        pts = load_ground_points(Path(args.lidar))
        dem, meta = rasterize_dem(pts, resolution=args.resolution_m, crop_m=args.crop_m)
    else:
        # For demo purposes, if no data is provided, we can't do much. 
        # But I'll assume the user will provide data.
        print("Please provide --lidar or --dem-tif")
        return

    hill = make_hillshade(dem)
    geo = classify_geomorphon_simple(dem)
    
    # False Positive Shield setup
    from earthwork_llm.surface.false_positive_shield import FalsePositiveShield, nearest_noise_feature
    shield = FalsePositiveShield()
    noise_gdf = None
    if args.noise_map:
        import geopandas as gpd
        noise_gdf = gpd.read_file(args.noise_map)

    # Using the enhanced multi-view panel for the LLM
    panel = make_multi_view_panel(dem, geo)

    locations = detect_earthworks(geo, dem, args.query)
    
    # Apply shield to candidates and prepare context
    survivors = []
    contexts = []
    transformer_to_deg = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    
    for cand in locations:
        cand_x_m = meta['x0'] + (cand['x'] / meta['nx']) * (meta['x1'] - meta['x0'])
        cand_y_m = meta['y0'] + (cand['y'] / meta['ny']) * (meta['y1'] - meta['y0'])
        lon, lat = transformer_to_deg.transform(cand_x_m, cand_y_m)
        
        nearest_m, nearest_label = nearest_noise_feature(lon, lat, noise_gdf)
        verdict = shield.evaluate(
            base_score=cand['p'],
            aspect=cand.get('aspect'),
            nearest_noise_m=nearest_m,
            nearest_noise_label=nearest_label,
        )
        
        if verdict.decision != "reject":
            survivors.append(cand)
            ctx = f"{nearest_label} ({nearest_m:.1f}m away)" if nearest_label else "No mapped noise"
            contexts.append(ctx)

    try:
        analysis = query_earthwork_v8(args.api_url, args.model, args.query, panel, survivors, contexts)
        log.info("V8 Earthwork Interpretation:\n" + strip_thinking(analysis))
    except Exception as e:
        log.error(f"V8 call failed: {e}")
        analysis = f"(V8 unreachable: {e})"

    out_path = Path(args.out)
    render_overlay(dem, hill, geo, survivors, args.query, analysis, out_path)

if __name__ == "__main__":
    main()
