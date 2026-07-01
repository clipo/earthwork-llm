#!/usr/bin/env python3
"""
Generate Yazoo Topo Noise Map.

Downloads modern and historical USGS topographic maps for a given AOI,
extracts potential modern noise features (roads, canals, levees, dredges),
and saves them as a GeoJSON for filtering mound candidates.

Usage:
    python scripts/generate_yazoo_noise_map.py \
        --bbox -91.1,33.4,-91.0,33.5 \
        --out data/yazoo_noise.geojson
"""

import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple

# Import TerraLLM ingestion tools
from earthwork_llm.ingestion.usgs_quad_downloader import USGSQuadDownloader
from earthwork_llm.ingestion.label_extractor import USGSLabelExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("noise_map")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", required=True, help="WGS84 bbox: min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--out", default="data/yazoo_noise_map.geojson")
    parser.add_argument("--gcs-bucket", default="lidar-data-for-llm")
    parser.add_argument("--historical", action="store_true", help="Also fetch historical maps (HTMC)")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # Parse bbox
    try:
        bbox_wgs84 = tuple(map(float, args.bbox.split(",")))
    except:
        log.error("Invalid bbox")
        return

    downloader = USGSQuadDownloader(gcs_bucket=args.gcs_bucket)
    extractor = USGSLabelExtractor(gcs_bucket=args.gcs_bucket, dpi=args.dpi)

    # 1. Query Quads
    quads = downloader.query_quads_for_bbox(bbox_wgs84)
    if args.historical:
        quads += downloader.query_quads_for_bbox(bbox_wgs84, historical=True)
    
    log.info(f"Found {len(quads)} quads for AOI")

    all_noise_features = []

    # 2. Process each quad
    # Note: This uses Google Vision API which has a cost.
    for quad in quads:
        log.info(f"Processing quad: {quad.title} ({quad.publication_date})")
        
        # Download (temporarily local if no GCS, or to GCS if available)
        local_path = Path(f"/tmp/{quad.source_id}.pdf")
        
        try:
            # Reusing downloader logic but keeping it local for extraction
            import requests
            resp = requests.get(quad.download_url, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract features
            # We pass the quad's bbox to the extractor for georeferencing
            result = extractor.extract_from_pdf(
                local_path, 
                quad_name=quad.title,
                geo_bounds=quad.bounding_box
            )
            
            # Combine all features that represent modern noise
            # We'll filter for roads, canals, levees, etc.
            noise = result.roads + result.streams # Note: Streams includes canals/dredge after my fix
            
            # Filter specifically for the keywords we added
            filtered_noise = []
            noise_keywords = [
                "canal", "levee", "ditch", "road", "st", "ave", "hwy", "dredge", 
                "drain", "embankment", "irrigation", "sluice", "pump", "borrow pit",
                "slough", "furrow", "railroad", "rr", "track", "trail"
            ]
            
            for f in noise:
                text = (f.name or "").lower()
                raw = (f.raw_text or "").lower()
                if any(k in text or k in raw for k in noise_keywords):
                    filtered_noise.append(f)
            
            all_noise_features.extend(filtered_noise)
            log.info(f"Extracted {len(filtered_noise)} potential noise features from {quad.title}")

        except Exception as e:
            log.error(f"Failed to process {quad.title}: {e}")
        finally:
            if local_path.exists():
                local_path.unlink()

    # 3. Save as GeoJSON
    if all_noise_features:
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": list(f.geo_coords)
                    } if f.geometry_type == "point" else {
                        "type": "LineString",
                        "coordinates": [list(f.metadata.get("start_geo")), list(f.metadata.get("end_geo"))]
                    } if f.geometry_type == "line" else None,
                    "properties": {
                        "type": f.feature_type,
                        "class": f.classification,
                        "name": f.name,
                        "source": f.source,
                        "quad": f.metadata.get("quad_name")
                    }
                } for f in all_noise_features if f.geo_coords or (f.metadata.get("start_geo") and f.metadata.get("end_geo"))
            ]
        }
        
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(geojson, f, indent=2)
            
        log.info(f"Saved {len(all_noise_features)} noise features to {out_path}")
    else:
        log.info("No noise features found.")

if __name__ == "__main__":
    main()
