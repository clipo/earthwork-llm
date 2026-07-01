#!/usr/bin/env python3
"""
Systematic Earthwork Search and Validation Script.

Iterates through a 'gold list' of known mounds, downloads LiDAR/DEM data for each,
runs the TerraLLM detection pipeline, and validates results.

Usage:
    python scripts/systematic_earthwork_search.py \
        --gold-list $EARTHWORK_GOLD_LIST  # restricted; not shipped \
        --out-dir data/validation_results
"""

import argparse
import logging
import pandas as pd
import numpy as np
import io
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from pyproj import Transformer

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

# Import project components
from earthwork_llm.ingestion.yazoo_downloader import YazooDownloader
from earthwork_llm.surface.false_positive_shield import FalsePositiveShield, Decision
from earthwork_query import detect_earthworks, query_earthwork_v8
from demo_terrain_query import (
    load_dem_geotiff, 
    make_hillshade, 
    classify_geomorphon_simple, 
    make_composite_panel, 
    render_overlay
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("systematic_search")

def check_match(gold_lat: float, gold_lon: float, candidates: List[dict], meta: dict, threshold_m: float = 30.0) -> Tuple[bool, Optional[dict]]:
    """
    Checks if any detected candidate matches the gold coordinates.
    Converts gold (lat, lon) to pixel (x, y) using the DEM metadata and projection.
    """
    # Project gold lat/lon to the DEM's CRS (assumed 3857 based on YazooDownloader)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    gold_x_m, gold_y_m = transformer.transform(gold_lon, gold_lat)
    
    # meta = dict(x0=x0, y0=y0, x1=x1, y1=y1, resolution=resolution, nx=nx, ny=ny)
    x0, y0 = meta['x0'], meta['y0']
    x1, y1 = meta['x1'], meta['y1']
    nx, ny = meta['nx'], meta['ny']
    
    # Gold pixel coordinates in the cropped DEM
    # load_dem_geotiff crops using center_x, center_y and half
    # so x0, y0 are the bounds of the CROP in meters (EPSG:3857)
    
    gold_px_x = (gold_x_m - x0) / (x1 - x0) * nx
    gold_px_y = (gold_y_m - y0) / (y1 - y0) * ny
    
    best_match = None
    min_dist = float('inf')
    
    for cand in candidates:
        # Distance in pixels
        dist_px = np.sqrt((cand['x'] - gold_px_x)**2 + (cand['y'] - gold_px_y)**2)
        # Distance in meters
        dist_m = dist_px * meta['resolution']
        
        if dist_m < threshold_m and dist_m < min_dist:
            min_dist = dist_m
            best_match = cand
            # Add dist_m to the candidate for recording
            best_match['dist_m'] = dist_m
            
    return (best_match is not None), best_match

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-list", required=True, help="Path to located_mounds.csv (local or gs://)")
    parser.add_argument("--out-dir", default="data/validation_results", help="Where to save results (local or gs://)")
    parser.add_argument("--gcs-bucket", help="GCS bucket name for data and results")
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="terrallm-v8")
    parser.add_argument("--limit", type=int, help="Limit number of mounds to process for testing")
    parser.add_argument("--crop-m", type=float, default=300.0)
    parser.add_argument("--resolution-m", type=float, default=1.0)
    parser.add_argument("--use-nlcd", action="store_true",
                        help="Apply NLCD land-cover screening in the shield (slower, network-dependent)")
    args = parser.parse_args()

    # The shield is applied to validation too, so we can report recall both
    # before and after false-positive screening (does the shield kill true mounds?).
    shield = FalsePositiveShield(enclosure_query=False)

    # Initialize GCS client if needed
    storage_client = None
    bucket = None
    if (args.gold_list.startswith("gs://") or args.out_dir.startswith("gs://") or args.gcs_bucket) and GCS_AVAILABLE:
        storage_client = storage.Client()
        if args.gcs_bucket:
            bucket = storage_client.bucket(args.gcs_bucket)

    # Resolve output directory
    out_dir_is_gcs = args.out_dir.startswith("gs://")
    out_dir_local = Path("/tmp/yazoo_validation") if out_dir_is_gcs else Path(args.out_dir)
    out_dir_local.mkdir(parents=True, exist_ok=True)
    
    downloader = YazooDownloader(out_dir=str(out_dir_local / "dems"), gcs_bucket=args.gcs_bucket)
    
    log.info(f"Loading gold list from {args.gold_list}")
    if args.gold_list.startswith("gs://"):
        # Download from GCS
        path_parts = args.gold_list[5:].split("/", 1)
        src_bucket = storage_client.bucket(path_parts[0])
        blob = src_bucket.blob(path_parts[1])
        csv_content = blob.download_as_text()
        df = pd.read_csv(io.StringIO(csv_content))
    else:
        df = pd.read_csv(args.gold_list)
    
    if args.limit:
        df = df.head(args.limit)
        
    results = []
    
    # Projection transformer for center point
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    for i, row in df.iterrows():
        mound_id = row['mound_id']
        lat, lon = row['latitude'], row['longitude']
        
        log.info(f"Processing mound {mound_id} at ({lat}, {lon})")
        
        # 1. Define bbox in WGS84 for download
        bbox = (lon - 0.0025, lat - 0.0025, lon + 0.0025, lat + 0.0025)
        
        # 2. Download DEM (now returns in EPSG:3857 meters)
        dem_tif = downloader.download_3dep_dem_bbox(bbox, f"mound_{mound_id}")
        if not dem_tif:
            log.error(f"Failed to download DEM for {mound_id}")
            results.append({"mound_id": mound_id, "status": "failed_download"})
            continue
            
        # 3. Load and Crop DEM (using meters)
        center_x, center_y = transformer.transform(lon, lat)
        try:
            dem, meta = load_dem_geotiff(dem_tif, center_x=center_x, center_y=center_y, 
                                         crop_m=args.crop_m, target_res_m=args.resolution_m)
        except Exception as e:
            log.error(f"Error loading DEM for {mound_id}: {e}")
            results.append({"mound_id": mound_id, "status": "failed_load"})
            continue
            
        # 4. Generate visual features
        hill = make_hillshade(dem)
        geo = classify_geomorphon_simple(dem)
        
        # 5. Detect Candidates
        query = "Find pre-European earthwork mounds"
        candidates = detect_earthworks(geo, dem, query)

        # 5b. Apply the False Positive Shield to each candidate. We keep the raw
        #     list (to measure detector recall) and a screened list (to measure
        #     how many true mounds the shield wrongly rejects).
        survivors = []
        survivor_contexts = []
        for cand in candidates:
            if args.use_nlcd:
                c_lon = lon  # NLCD at the mound point is a reasonable proxy here
                c_lat = lat
                nlcd_val, nlcd_name = downloader.get_nlcd_class(c_lon, c_lat)
            else:
                nlcd_val, nlcd_name = 0, "NLCD disabled"
            
            # Note: systematic_search currently doesn't load a noise map by default,
            # but we'll prepare the logic for context passing.
            verdict = shield.evaluate(
                base_score=cand['p'],
                aspect=cand.get('aspect'),
                nlcd_value=nlcd_val,
                nlcd_name=nlcd_name,
            )
            if verdict.decision != Decision.REJECT:
                survivors.append(cand)
                # Prepare a context string for the LLM
                reasons = "; ".join(verdict.reasons) if verdict.reasons else "morphology clean"
                context = f"Shield: {verdict.decision.value} ({reasons})"
                survivor_contexts.append(context)

        # 6. Validate Match (raw detector recall and post-shield recall)
        found, match = check_match(lat, lon, candidates, meta)
        found_shielded, _ = check_match(lat, lon, survivors, meta)
        
        # 7. Optional V8 Query
        analysis = "N/A"
        if args.api_url:
            try:
                panel = make_composite_panel(dem, hill, geo)
                # Pass survivors and their context to the LLM
                analysis = query_earthwork_v8(args.api_url, args.model, query, panel, survivors, survivor_contexts)
            except Exception as e:
                log.warning(f"V8 query failed for {mound_id}: {e}")
                analysis = f"Error: {e}"
                
        # 8. Save Overlay
        out_png = out_dir_local / f"mound_{mound_id}_validation.png"
        render_overlay(dem, hill, geo, candidates, query, analysis, out_png)
        
        # 9. Record Result
        results.append({
            "mound_id": mound_id,
            "latitude": lat,
            "longitude": lon,
            "status": "success",
            "found": found,
            "found_after_shield": found_shielded,
            "shield_dropped_true": bool(found and not found_shielded),
            "match_prob": match['p'] if found else 0.0,
            "match_dist": match.get('dist_m', 0.0) if found else -1.0,
            "candidates_count": len(candidates),
            "survivors_count": len(survivors)
        })

        log.info(f"Result for {mound_id}: Found={found} (after shield={found_shielded})")

    # Save summary report
    summary_df = pd.DataFrame(results)
    summary_path_local = out_dir_local / "validation_summary.csv"
    summary_df.to_csv(summary_path_local, index=False)
    
    # Upload results to GCS if requested
    if out_dir_is_gcs:
        dest_parts = args.out_dir[5:].split("/", 1)
        dest_bucket_name = dest_parts[0]
        dest_prefix = dest_parts[1] if len(dest_parts) > 1 else ""
        dest_bucket = storage_client.bucket(dest_bucket_name)
        
        log.info(f"Uploading results to gs://{dest_bucket_name}/{dest_prefix}")
        # Upload all files in out_dir_local
        for file_path in out_dir_local.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(out_dir_local)
                blob_path = os.path.join(dest_prefix, str(rel_path)).strip("/")
                blob = dest_bucket.blob(blob_path)
                blob.upload_from_filename(str(file_path))
                # log.info(f"Uploaded {file_path.name} to gs://{dest_bucket_name}/{blob_path}")

    if 'found' in summary_df.columns:
        hit_rate = summary_df['found'].mean()
        log.info(f"Systematic search complete. Hit rate: {hit_rate:.1%}")
    else:
        log.warning("No 'found' results recorded.")

    print(f"\nValidation Summary:")
    print(summary_df[['mound_id', 'found', 'match_prob']] if not summary_df.empty else "Empty results")

if __name__ == "__main__":
    main()
