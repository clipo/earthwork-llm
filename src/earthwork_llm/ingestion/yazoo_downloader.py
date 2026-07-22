"""
Yazoo Basin LiDAR Downloader for TerraLLM

Specialized downloader for Mississippi River Valley (Yazoo Basin) LiDAR data.
Supports multiple sources:
1. 3DEP ImageServer (Bare-earth DEM export)
2. National Map Access API (Raw LAZ tile discovery)
3. NOAA Digital Coast (USACE Delta Phase 1 dataset)
4. MARIS (Mississippi state portal)

This downloader enables zero-shot detection of pre-European earthworks
by fetching high-resolution data from the core Mississippian culture regions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import time

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Yazoo Basin Archaeological Site Bounds (WGS84) ---
# Approximate centroids and 1km bounding boxes for major earthwork sites
YAZOO_SITES = {
    "Jaketown": {
        "centroid": (33.185, -90.485),
        "description": "Poverty Point and Mississippian site near Belzoni, MS",
        "bbox": (-90.495, 33.175, -90.475, 33.195)
    },
    "Winterville": {
        "centroid": (33.482, -91.062),
        "description": "Major Mississippian mound complex near Greenville, MS",
        "bbox": (-91.075, 33.472, -91.050, 33.492)
    },
    "Holly Bluff": {
        "centroid": (32.822, -90.712),
        "description": "Lake George site, major Mississippian center",
        "bbox": (-90.725, 32.812, -90.700, 32.832)
    },
    "Haynes Bluff": {
        "centroid": (32.515, -90.710),
        "description": "Mound site on the Yazoo Bluffs",
        "bbox": (-90.725, 32.505, -90.695, 32.525)
    }
}

# --- API Endpoints ---
THREEDEP_IMAGE_SERVER = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
TNM_ACCESS_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
NOAA_DAV_QUERY = "https://maps.coast.noaa.gov/arcgis/rest/services/DAV/DAV_footprints/MapServer/0/query"
# MRLC public WMS for NLCD. Token-free, unlike the Esri Living Atlas
# ImageServer (which now returns HTTP 499 "Token Required" and was the cause
# of every NLCD lookup failing in earlier scans).
MRLC_WMS = "https://www.mrlc.gov/geoserver/mrlc_display/wms"
NLCD_WMS_LAYER = "NLCD_2021_Land_Cover_L48"

# NLCD Class Mappings (partial list relevant to filtering)
NLCD_CLASSES = {
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land (Rock/Sand/Clay)",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands"
}

class YazooDownloader:
    """
    Downloads Yazoo Basin LiDAR and DEM data from national and state sources.
    
    Example:
        >>> downloader = YazooDownloader()
        >>> # Get 3DEP DEM for Winterville Mounds
        >>> dem_path = downloader.download_3dep_dem("Winterville", out_dir="data/yazoo")
    """

    def __init__(self, out_dir: str = "data/yazoo", gcs_bucket: Optional[str] = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.gcs_bucket_name = gcs_bucket
        self._storage_client = None
        self._bucket = None
        
        # Throttling and Caching
        self.last_request_time = 0
        self.min_delay_sec = 1.0  # 1 second between requests
        self._nlcd_cache = {} # (round_lon, round_lat) -> (val, name)
        self._nlcd_failures = 0
        self._nlcd_disabled_until = 0
        
        if gcs_bucket and GCS_AVAILABLE:
            self._storage_client = storage.Client()
            self._bucket = self._storage_client.bucket(gcs_bucket)
            logger.info(f"Initialized YazooDownloader with GCS bucket: {gcs_bucket}")
        
        logger.info(f"Initialized YazooDownloader, output directory: {self.out_dir}")

    def download_3dep_dem(
        self, 
        site_name: str, 
        resolution: float = 1.0, 
        format: str = "tiff"
    ) -> Optional[Path]:
        """
        Download a bare-earth DEM from 3DEP ImageServer for a named site.
        """
        if site_name not in YAZOO_SITES:
            logger.error(f"Site {site_name} not found in YAZOO_SITES")
            return None
        
        bbox = YAZOO_SITES[site_name]["bbox"]
        return self.download_3dep_dem_bbox(bbox, site_name, resolution, format)

    def download_3dep_dem_bbox(
        self, 
        bbox: Tuple[float, float, float, float], 
        name: str,
        resolution: float = 1.0,
        format: str = "tiff"
    ) -> Optional[Path]:
        """
        Fetch DEM from 3DEP exportImage for a specific bounding box.
        bbox format: (min_lon, min_lat, max_lon, max_lat)
        """
        # Calculate pixel dimensions for the requested resolution
        # Approx meters per degree at this latitude (~33N) is ~93km/deg lat, ~111km/deg lon
        width_m = abs(bbox[2] - bbox[0]) * 93000
        height_m = abs(bbox[3] - bbox[1]) * 111000
        size_x = int(width_m / resolution)
        size_y = int(height_m / resolution)

        params = {
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "bboxSR": "4326", # WGS84
            "size": f"{size_x},{size_y}",
            "imageSR": "3857", # Web Mercator (meters)
            "format": format,
            "pixelType": "F32",
            "noDataInterpretation": "esriNoDataMatchAny",
            "interpolation": "RSP_BilinearInterpolation",
            "f": "json"
        }

        logger.info(f"Requesting 3DEP DEM for {name} ({size_x}x{size_y} pixels)")
        
        try:
            # First get the URL of the exported image
            resp = requests.get(THREEDEP_IMAGE_SERVER, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            
            image_url = data.get("href")
            if not image_url:
                logger.error(f"No image URL in 3DEP response: {data}")
                return None
            
            # Now download the actual image
            out_filename = f"{name}_3dep_dem.tif"
            out_path = self.out_dir / out_filename
            img_resp = requests.get(image_url, stream=True, timeout=120)
            img_resp.raise_for_status()
            
            with open(out_path, "wb") as f:
                for chunk in img_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded DEM to {out_path}")
            
            # Upload to GCS if configured
            if self._bucket:
                gcs_path = f"raw/yazoo/{out_filename}"
                blob = self._bucket.blob(gcs_path)
                blob.upload_from_filename(str(out_path))
                logger.info(f"Uploaded DEM to gs://{self.gcs_bucket_name}/{gcs_path}")

            return out_path

        except Exception as e:
            logger.error(f"Failed to download 3DEP DEM for {name}: {e}")
            return None

    def search_national_map_laz(
        self, 
        bbox: Tuple[float, float, float, float], 
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search for raw LAZ tiles overlapping a bbox via TNM Access API.
        """
        params = {
            "datasets": "Lidar Point Cloud (LPC)",
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "max": max_results,
            "outputFormat": "JSON"
        }

        logger.info(f"Searching National Map for LAZ tiles in bbox: {bbox}")
        
        try:
            resp = requests.get(TNM_ACCESS_API, params=params, timeout=60)
            resp.raise_for_status()
            items = resp.json().get("items", [])
            logger.info(f"Found {len(items)} LAZ products")
            return items
        except Exception as e:
            logger.error(f"National Map search failed: {e}")
            return []

    def download_noaa_usace_delta_info(self) -> List[Dict]:
        """
        Query metadata for the '2009-2010 USACE Delta Phase 1' dataset.
        This dataset covers most of our Yazoo sites.
        """
        # Search footprint by dataset name
        params = {
            "where": "DatasetName LIKE '%USACE Delta Phase 1%'",
            "outFields": "*",
            "f": "json"
        }
        
        try:
            resp = requests.get(NOAA_DAV_QUERY, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json().get("features", [])
        except Exception as e:
            logger.error(f"NOAA query failed: {e}")
            return []

    def get_nlcd_class(self, lon: float, lat: float, retries: int = 3) -> Tuple[int, str]:
        """
        Fetch the NLCD land cover class for a specific point.
        Includes point-based caching, throttling, and a circuit breaker for stability.
        """
        # 1. Circuit Breaker Check
        if time.time() < self._nlcd_disabled_until:
            return 0, "NLCD Lookup Paused (Circuit Breaker)"

        # 2. Check Cache (Round to ~30m resolution to group nearby points)
        cache_key = (round(lon, 4), round(lat, 4))
        if cache_key in self._nlcd_cache:
            return self._nlcd_cache[cache_key]

        # A small bbox around the point with a 3x3 query pixel; we read the
        # centre pixel via a WMS GetFeatureInfo call.
        d = 0.0005
        params = {
            "service": "WMS",
            "version": "1.1.1",
            "request": "GetFeatureInfo",
            "layers": NLCD_WMS_LAYER,
            "query_layers": NLCD_WMS_LAYER,
            "srs": "EPSG:4326",
            "bbox": f"{lon - d},{lat - d},{lon + d},{lat + d}",
            "width": "3",
            "height": "3",
            "x": "1",
            "y": "1",
            "info_format": "application/json",
        }

        for attempt in range(retries):
            # Throttling
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_delay_sec:
                time.sleep(self.min_delay_sec - elapsed)

            self.last_request_time = time.time()

            try:
                resp = requests.get(MRLC_WMS, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                # GeoServer returns a GeoJSON FeatureCollection; the NLCD class
                # code is in properties.PALETTE_INDEX (e.g. 82 = Cultivated Crops).
                features = data.get("features", [])
                if not features:
                    return 0, "No Data"

                props = features[0].get("properties", {})
                val = props.get("PALETTE_INDEX")
                if val is None:
                    val = props.get("GRAY_INDEX")
                if val is None:
                    return 0, "No Data"

                val = int(val)
                name = NLCD_CLASSES.get(val, f"Unknown ({val})")

                # Success: reset failures and update cache
                self._nlcd_failures = 0
                self._nlcd_cache[cache_key] = (val, name)
                return val, name
                
            except Exception as e:
                # Check for "No route to host" specifically
                if "[Errno 65]" in str(e) or "No route to host" in str(e):
                    logger.error(f"Host {MRLC_WMS} is unreachable. Skipping NLCD for this tile.")
                    self._nlcd_disabled_until = time.time() + 600 # 10 minute pause
                    return 0, "Host Unreachable"

                self._nlcd_failures += 1
                wait = (attempt + 1) * 5 # Aggressive backoff
                logger.warning(f"NLCD lookup attempt {attempt+1} failed at ({lon}, {lat}): {e}. Retrying in {wait}s...")
                
                # If too many failures, trip the circuit breaker for 5 minutes
                if self._nlcd_failures > 5:
                    logger.error("Too many NLCD lookup failures. Tripping circuit breaker for 5 minutes.")
                    self._nlcd_disabled_until = time.time() + 300
                    return 0, "NLCD Lookup Failed (Circuit Breaker Tripped)"
                
                time.sleep(wait)

        return 0, "Lookup Failed"

def main():
    """Download DEMs for key Yazoo sites if requested."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", help="Specific site to download (e.g., Winterville)")
    parser.add_argument("--all", action="store_true", help="Download all listed sites")
    parser.add_argument("--res", type=float, default=1.0, help="Resolution in meters")
    args = parser.parse_args()

    downloader = YazooDownloader()
    
    sites_to_download = []
    if args.site:
        sites_to_download = [args.site]
    elif args.all:
        sites_to_download = list(YAZOO_SITES.keys())
    
    for site in sites_to_download:
        downloader.download_3dep_dem(site, resolution=args.res)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
