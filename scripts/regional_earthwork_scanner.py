#!/usr/bin/env python3
"""
Regional Earthwork Scanner for TerraLLM.

Tiles across a large geographic bounding box, fetches LiDAR/DEM data for each tile,
runs the detection pipeline, and aggregates results into a regional dataset.

Usage:
    python scripts/regional_earthwork_scanner.py \
        --bbox -91.1,33.4,-91.0,33.5 \
        --out-dir data/regional_scan_yazoo \
        --tile-size-m 500 \
        --overlap-m 50
"""

import argparse
import logging
import json
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from pyproj import Transformer

# Import project components
from earthwork_llm.ingestion.yazoo_downloader import YazooDownloader
from earthwork_llm.surface.false_positive_shield import (
    FalsePositiveShield,
    Decision,
    nearest_noise_feature,
)
from earthwork_llm.surface.triage import (
    scan_stats,
    score_a,
    score_b,
    rank_descending,
)
import context_sheet as cs
from earthwork_query import detect_earthworks, query_earthwork_v8
from demo_terrain_query import (
    load_dem_geotiff,
    make_hillshade,
    classify_geomorphon_simple,
    make_multi_view_panel,
    render_overlay
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("regional_scanner")

def generate_tiles(bbox_wgs84: Tuple[float, float, float, float], tile_size_m: float, overlap_m: float) -> List[Dict]:
    """
    Generates a grid of tiles in Web Mercator (meters) covering the WGS84 bbox.
    """
    transformer_to_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    transformer_to_deg = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    x0, y0 = transformer_to_m.transform(min_lon, min_lat)
    x1, y1 = transformer_to_m.transform(max_lon, max_lat)
    
    width_m = x1 - x0
    height_m = y1 - y0
    
    stride = tile_size_m - overlap_m
    nx = int(np.ceil(width_m / stride))
    ny = int(np.ceil(height_m / stride))
    
    tiles = []
    for i in range(nx):
        for j in range(ny):
            tx0 = x0 + i * stride
            ty0 = y0 + j * stride
            tx1 = tx0 + tile_size_m
            ty1 = ty0 + tile_size_m
            
            # Center of tile in meters
            cx_m, cy_m = (tx0 + tx1) / 2, (ty0 + ty1) / 2
            
            # Convert tile bounds back to degrees for the downloader
            tw0, ts0 = transformer_to_deg.transform(tx0, ty0)
            te1, tn1 = transformer_to_deg.transform(tx1, ty1)
            
            tiles.append({
                "id": f"tile_{i}_{j}",
                "bbox_wgs84": (tw0, ts0, te1, tn1),
                "center_m": (cx_m, cy_m),
                "grid_pos": (i, j)
            })
            
    return tiles

# ---------------------------------------------------------------------------
# Fast per-survivor context distances for Score B (Sections 3.8, 4; App. B.6).
# One FEMA USA Structures query + one NHD canal query per surviving candidate
# (never per raw candidate), reusing scripts/context_sheet.py endpoints.
# Convention (see earthwork_llm.surface.triage): None = service unavailable
# (Score B flagged incomplete); math.inf = queried, nothing within radius.
# ---------------------------------------------------------------------------

_dist_cache: Dict[Tuple[str, float, float], Optional[float]] = {}

def _structure_distance_m(x_utm: float, y_utm: float) -> Optional[float]:
    """Distance (m) to the nearest FEMA USA Structures footprint."""
    key = ("fema", round(x_utm, -1), round(y_utm, -1))
    if key in _dist_cache:
        return _dist_cache[key]
    feats = cs._arcgis_envelope_query(cs.USA_STRUCTURES, x_utm, y_utm,
                                      out_fields="OBJECTID")
    if feats is None:
        result = None
    else:
        result = math.inf
        for f in feats:
            rings = f.get("geometry", {}).get("rings", [])
            if any(cs._inside_ring(x_utm, y_utm, ring) for ring in rings):
                result = 0.0
                break
            d, _, _ = cs._nearest_on_paths(x_utm, y_utm, rings)
            if d is not None and d < result:
                result = d
    _dist_cache[key] = result
    return result

def _canal_distance_m(x_utm: float, y_utm: float) -> Optional[float]:
    """Distance (m) to the nearest NHD canal/ditch flowline."""
    key = ("nhd", round(x_utm, -1), round(y_utm, -1))
    if key in _dist_cache:
        return _dist_cache[key]
    best, _name = cs._nearest_flowline(x_utm, y_utm, cs.FCODE_CANAL)
    if best is None:
        result = None
    elif best[0] is None:
        result = math.inf
    else:
        result = best[0]
    _dist_cache[key] = result
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", required=True, help="Bounding box in WGS84: min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--out-dir", default="data/regional_scans", help="Output directory")
    parser.add_argument("--gcs-bucket", help="Optional GCS bucket for results")
    parser.add_argument("--tile-size-m", type=float, default=500.0, help="Size of each scan tile in meters")
    parser.add_argument("--overlap-m", type=float, default=50.0, help="Overlap between tiles in meters")
    parser.add_argument("--noise-map", help="Optional GeoJSON noise map (modern features) to cross-reference")
    parser.add_argument("--resolution-m", type=float, default=1.0, help="DEM resolution")
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="terrallm-v91")
    parser.add_argument("--llm-prob-threshold", type=float, default=0.7, help="Only run LLM if deterministic detector finds candidates with probability > X")
    parser.add_argument("--no-nlcd", action="store_true", help="Disable NLCD land-cover screening")
    parser.add_argument("--keep-rejected", action="store_true", help="Write shield-rejected candidates to output (flagged) instead of dropping them")
    parser.add_argument("--no-triage-queries", action="store_true",
                        help="Skip the per-survivor FEMA/NHD distance queries for Score B "
                             "(Score B is then computed from the shield context alone and flagged incomplete)")
    args = parser.parse_args()

    # Parse bbox
    try:
        bbox_wgs84 = tuple(map(float, args.bbox.split(",")))
        if len(bbox_wgs84) != 4:
            raise ValueError
    except Exception:
        log.error("Invalid bbox format. Use min_lon,min_lat,max_lon,max_lat")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    downloader = YazooDownloader(out_dir=str(out_dir / "dems"), gcs_bucket=args.gcs_bucket)
    tiles = generate_tiles(bbox_wgs84, args.tile_size_m, args.overlap_m)
    
    log.info(f"Generated {len(tiles)} tiles for regional scan")
    
    # Load noise map if provided
    noise_gdf = None
    if args.noise_map:
        import geopandas as gpd
        log.info(f"Loading noise map from {args.noise_map}")
        noise_gdf = gpd.read_file(args.noise_map)
        log.info(f"Noise map: {len(noise_gdf)} modern features loaded")
    else:
        log.warning("No --noise-map supplied: USGS/HTMC modern-feature screening is INACTIVE for this run")

    # Build the False Positive Shield once and reuse it across tiles.
    shield = FalsePositiveShield(enclosure_query=False)

    all_detections = []
    det_cands = []        # detector candidate dict for each row in all_detections
    scan_population = []  # every screened candidate: Score A is z-scored against these
    transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:26915", always_xy=True)
    shield_stats = {"kept": 0, "flagged": 0, "rejected": 0, "context_incomplete": 0}
    
    for tile in tiles:
        tile_id = tile["id"]
        log.info(f"Scanning {tile_id} at {tile['bbox_wgs84']}")
        
        # 1. Download
        dem_tif = downloader.download_3dep_dem_bbox(tile["bbox_wgs84"], tile_id, resolution=args.resolution_m)
        if not dem_tif:
            continue
            
        # 2. Load
        try:
            # Note: YazooDownloader now uses 3857, so we can use center_m directly
            dem, meta = load_dem_geotiff(dem_tif, center_x=tile["center_m"][0], center_y=tile["center_m"][1],
                                         crop_m=args.tile_size_m, target_res_m=args.resolution_m)
        except Exception as e:
            log.error(f"Failed to load tile {tile_id}: {e}")
            continue
            
        # 3. Detect
        hill = make_hillshade(dem)
        geo = classify_geomorphon_simple(dem)
        
        query = "Find pre-European earthwork mounds"
        candidates = detect_earthworks(geo, dem, query)

        if not candidates:
            continue

        # 4. False Positive Shield: screen every candidate against context layers
        #    BEFORE deciding whether to spend an LLM call. Rejected candidates are
        #    dropped (unless --keep-rejected); only survivors reach the model.
        transformer_to_deg = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        screened = []  # list of (cand, lon, lat, nlcd_name, verdict)

        for cand in candidates:
            cand_x_m = meta['x0'] + (cand['x'] / meta['nx']) * (meta['x1'] - meta['x0'])
            cand_y_m = meta['y0'] + (cand['y'] / meta['ny']) * (meta['y1'] - meta['y0'])
            lon, lat = transformer_to_deg.transform(cand_x_m, cand_y_m)

            if args.no_nlcd:
                nlcd_val, nlcd_name = 0, "NLCD disabled"
            else:
                nlcd_val, nlcd_name = downloader.get_nlcd_class(lon, lat)

            nearest_m, nearest_label = nearest_noise_feature(lon, lat, noise_gdf)

            verdict = shield.evaluate(
                base_score=cand['p'],
                aspect=cand.get('aspect'),
                nlcd_value=nlcd_val,
                nlcd_name=nlcd_name,
                nearest_noise_m=nearest_m,
                nearest_noise_label=nearest_label,
            )
            screened.append((cand, lon, lat, nlcd_name, nearest_label, nearest_m, verdict, nlcd_val))
            scan_population.append(cand)

            shield_stats["rejected" if verdict.decision == Decision.REJECT
                         else "flagged" if verdict.decision == Decision.FLAG
                         else "kept"] += 1
            if not verdict.context_complete:
                shield_stats["context_incomplete"] += 1

        survivors = [s for s in screened if s[6].decision != Decision.REJECT]

        # 5. Query LLM only on surviving candidates, only if any are strong enough.
        analysis = "N/A (Skipped LLM)"
        max_p = max((s[6].score for s in survivors), default=0.0)
        if survivors and max_p >= args.llm_prob_threshold and args.api_url:
            try:
                panel = make_multi_view_panel(dem, geo)
                surviving_cands = [s[0] for s in survivors]
                # Pass the nearby noise labels as context to help the LLM with false positive analysis
                surviving_context = [f"{s[4]} ({s[5]:.1f}m away)" if s[4] else "No mapped noise" for s in survivors]
                analysis = query_earthwork_v8(args.api_url, args.model, query, panel, surviving_cands, surviving_context)
                log.info(f"LLM Interpretation for {tile_id}:\n{analysis[:200]}...")
            except Exception as e:
                log.warning(f"LLM query failed for {tile_id}: {e}")

        # 6. Store detections (survivors, plus rejected ones if --keep-rejected)
        for cand, lon, lat, nlcd_name, nearest_label, nearest_m, verdict, nlcd_val in screened:
            if verdict.decision == Decision.REJECT and not args.keep_rejected:
                continue

            # Two-score triage (manuscript Sections 3.8, 4; Appendix B.6).
            # Score B here; Score A needs scan-wide statistics and is filled
            # in after all tiles are processed. The FEMA/NHD distance queries
            # run once per SURVIVING candidate only (never per raw candidate)
            # to keep the scan fast; rejected rows kept via --keep-rejected
            # get a partial Score B flagged incomplete.
            structure_m = canal_m = None
            if verdict.decision != Decision.REJECT and not args.no_triage_queries:
                x_utm, y_utm = transformer_to_utm.transform(lon, lat)
                structure_m = _structure_distance_m(x_utm, y_utm)
                canal_m = _canal_distance_m(x_utm, y_utm)
            sb = score_b(cand, {
                "nlcd_value": nlcd_val,
                "noise_map_available": noise_gdf is not None,
                "nearest_noise_m": nearest_m,
                "structure_m": structure_m,
                "canal_m": canal_m,
            })

            det = {
                "tile_id": tile_id,
                "latitude": lat,
                "longitude": lon,
                "prob": cand['p'],
                "shield_score": verdict.score,
                "shield_decision": verdict.decision.value,
                "context_complete": verdict.context_complete,
                "area_m2": cand.get('area', 0),
                "height_m": cand.get('height', 0),
                "aspect": cand.get('aspect', 0),
                "nlcd_class": nlcd_name,
                "nearby_noise": nearest_label or "None",
                "shield_reasons": "; ".join(verdict.reasons) if verdict.reasons else "none",
                "justification": cand['justification'],
                "llm_analysis": analysis if verdict.decision != Decision.REJECT else "N/A",
                # Two-score triage columns (additive; Sections 3.8, 4, B.6).
                "score_a": None,  # z-scored within the scan, filled post-loop
                "score_b": sb.score,
                "score_b_complete": sb.complete,
            }
            all_detections.append(det)
            det_cands.append(cand)

        # 7. Save a discovery visual when a surviving candidate is strong.
        if survivors and max_p >= args.llm_prob_threshold:
            out_png = out_dir / f"{tile_id}_discovery.png"
            render_overlay(dem, hill, geo, [s[0] for s in survivors], query, analysis, out_png)

    # Save final detections
    if all_detections:
        # Score A is z-scored within the scan, so it can only be computed once
        # every tile has been screened (Sections 3.8, 4; Appendix B.6).
        stats = scan_stats(scan_population)
        for det, cand in zip(all_detections, det_cands):
            det["score_a"] = score_a(cand, stats)
        for det, r in zip(all_detections,
                          rank_descending([d["score_a"] for d in all_detections])):
            det["rank_a"] = r
        for det, r in zip(all_detections,
                          rank_descending([d["score_b"] for d in all_detections])):
            det["rank_b"] = r

        df = pd.DataFrame(all_detections)
        report_path = out_dir / "regional_detections.csv"
        df.to_csv(report_path, index=False)
        log.info(f"Scan complete. Report: {report_path}")
        log.info(
            "False Positive Shield: "
            f"{shield_stats['kept']} kept, {shield_stats['flagged']} flagged, "
            f"{shield_stats['rejected']} rejected "
            f"({shield_stats['context_incomplete']} with incomplete context)"
        )
        
        # Optional: Save as GeoJSON for QGIS
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [d["longitude"], d["latitude"]]},
                    "properties": d
                } for d in all_detections
            ]
        }
        with open(out_dir / "regional_detections.geojson", "w") as f:
            json.dump(geojson, f)
    else:
        log.info("Scan complete. No candidates found.")

if __name__ == "__main__":
    main()
