"""
USGS 7.5-Minute Quadrangle Downloader for TerraLLM

Downloads USGS topographic quadrangles (US Topo) via the National Map API.
These maps provide pre-labeled features for automatic training data generation:
- Dashed black lines = Trails, dirt roads (MISSION CRITICAL for battlefield losses)
- Solid black/red lines = Paved roads
- Dashed blue lines = Intermittent streams
- Solid blue lines = Perennial streams
- Brown lines = Contour lines

The 7.5-minute (1:24,000 scale) quadrangles cover approximately 7.5 x 7.5 minutes
of latitude/longitude, roughly 49-70 square miles depending on latitude.
"""

import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

try:
    from google.cloud import storage

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logger = logging.getLogger(__name__)


# USGS National Map API configuration
USGS_API_BASE = "https://tnmaccess.nationalmap.gov/api/v1"
USGS_PRODUCTS_ENDPOINT = f"{USGS_API_BASE}/products"

# NY State FIPS code
NY_STATE_FIPS = "36"

# Adirondack region approximate bounds (WGS84)
ADIRONDACK_BOUNDS = {
    "min_lon": -75.5,
    "max_lon": -73.5,
    "min_lat": 43.5,
    "max_lat": 44.5,
}


@dataclass
class QuadInfo:
    """
    Information about a USGS 7.5-minute quadrangle.

    Attributes:
        title: Full quad title (e.g., "US Topo 7.5-minute map for Lake Placid, NY")
        source_id: USGS unique identifier
        download_url: Direct HTTPS URL to download the quad
        bounding_box: (min_lon, min_lat, max_lon, max_lat) in WGS84
        publication_date: Date the quad was published
        size_bytes: File size in bytes
        format: File format (typically "Geospatial PDF")
        quad_name: Short name extracted from title (e.g., "Lake Placid")
    """

    title: str
    source_id: str
    download_url: str
    bounding_box: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    publication_date: str
    size_bytes: int
    format: str
    quad_name: str = ""

    def __post_init__(self):
        # Extract quad name from title if not provided
        if not self.quad_name and self.title:
            # Title format: "US Topo 7.5-minute map for Lake Placid, NY"
            if " for " in self.title:
                name_part = self.title.split(" for ")[-1]
                # Remove state suffix
                if ", " in name_part:
                    name_part = name_part.split(", ")[0]
                self.quad_name = name_part

    @property
    def gcs_path(self) -> str:
        """GCS storage path for this quad"""
        safe_name = self.quad_name.replace(" ", "_").replace("/", "_")
        year = self.publication_date[:4] if self.publication_date else "unknown"
        return f"raw/usgs_quads/NY_{safe_name}_{year}.pdf"

    @property
    def size_mb(self) -> float:
        """File size in megabytes"""
        return self.size_bytes / (1024 * 1024)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "title": self.title,
            "source_id": self.source_id,
            "download_url": self.download_url,
            "bounding_box": self.bounding_box,
            "publication_date": self.publication_date,
            "size_bytes": self.size_bytes,
            "format": self.format,
            "quad_name": self.quad_name,
            "gcs_path": self.gcs_path,
        }

    @classmethod
    def from_api_response(cls, item: Dict) -> "QuadInfo":
        """Create QuadInfo from USGS API response item"""
        bbox = item.get("boundingBox", {})
        return cls(
            title=item.get("title", ""),
            source_id=item.get("sourceId", ""),
            download_url=item.get("downloadURL", ""),
            bounding_box=(
                bbox.get("minX", 0.0),
                bbox.get("minY", 0.0),
                bbox.get("maxX", 0.0),
                bbox.get("maxY", 0.0),
            ),
            publication_date=item.get("publicationDate", ""),
            size_bytes=item.get("sizeInBytes", 0),
            format=item.get("format", ""),
        )


@dataclass
class QuadDownloadProgress:
    """Tracks download progress for resumability"""

    downloaded: List[str] = field(default_factory=list)  # source_ids
    failed: List[str] = field(default_factory=list)
    total_bytes: int = 0
    last_updated: str = ""

    def to_dict(self) -> Dict:
        """Return the progress record as a JSON-serializable dict."""
        return {
            "downloaded": self.downloaded,
            "failed": self.failed,
            "total_bytes": self.total_bytes,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "QuadDownloadProgress":
        """Rebuild a progress record from :meth:`to_dict` output."""
        return cls(
            downloaded=data.get("downloaded", []),
            failed=data.get("failed", []),
            total_bytes=data.get("total_bytes", 0),
            last_updated=data.get("last_updated", ""),
        )


class USGSQuadDownloader:
    """
    Downloads USGS 7.5-minute quadrangles via the National Map API.

    These topographic maps contain pre-labeled features essential for
    training data generation:
    - TRAILS (dashed black) - Critical for battlefield loss locations
    - Roads (solid black/red) - Transportation network
    - Streams (blue, solid/dashed) - Water features
    - Contours (brown) - Elevation

    Example:
        >>> downloader = USGSQuadDownloader("lidar-data-for-llm")
        >>> # Query quads for Adirondack region
        >>> quads = downloader.query_quads_for_bbox((-75.0, 43.5, -74.0, 44.5))
        >>> print(f"Found {len(quads)} quads")
        >>> # Download quads
        >>> downloaded = downloader.download_batch(quads, max_quads=10)
    """

    PROGRESS_FILE = "metadata/usgs_quad_download_log.json"

    def __init__(self, gcs_bucket: str = "lidar-data-for-llm"):
        """
        Initialize USGS quad downloader.

        Args:
            gcs_bucket: Google Cloud Storage bucket name
        """
        self.gcs_bucket = gcs_bucket
        self._gcs_client = None
        self._bucket = None
        self.progress = QuadDownloadProgress()

        # Load existing progress
        self._load_progress()

        logger.info(f"Initialized USGSQuadDownloader with bucket: {gcs_bucket}")

    @property
    def gcs_client(self):
        """Lazy initialization of GCS client"""
        if self._gcs_client is None and GCS_AVAILABLE:
            self._gcs_client = storage.Client()
        return self._gcs_client

    @property
    def bucket(self):
        """Get GCS bucket"""
        if self._bucket is None and self.gcs_client:
            self._bucket = self.gcs_client.bucket(self.gcs_bucket)
        return self._bucket

    def query_quads_for_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        max_results: int = 100,
        historical: bool = False,
    ) -> List[QuadInfo]:
        """
        Query USGS API for quads overlapping a bounding box.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84
            max_results: Maximum number of results to return
            historical: If True, query HTMC (historical maps), else US Topo (modern)

        Returns:
            List of QuadInfo objects for matching quads
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        dataset = "Historical Topographic Map Collection (HTMC)" if historical else "US Topo"

        params = {
            "datasets": dataset,
            "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "max": max_results,
            "outputFormat": "JSON",
        }

        logger.info(f"Querying USGS API for bbox: {bbox}")

        try:
            response = requests.get(USGS_PRODUCTS_ENDPOINT, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            items = data.get("items", [])
            quads = [QuadInfo.from_api_response(item) for item in items]

            logger.info(f"Found {len(quads)} quads for bbox")
            return quads

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []

    def query_quads_for_state(
        self,
        state_fips: str = NY_STATE_FIPS,
        max_results: int = 500,
    ) -> List[QuadInfo]:
        """
        Query USGS API for all quads in a state.

        Args:
            state_fips: State FIPS code (NY = "36")
            max_results: Maximum number of results per query

        Returns:
            List of QuadInfo objects for all state quads
        """
        all_quads = []
        offset = 0

        logger.info(f"Querying all quads for state FIPS: {state_fips}")

        while True:
            params = {
                "datasets": "US Topo",
                "polyCode": state_fips,
                "polyType": "state",
                "max": max_results,
                "offset": offset,
                "outputFormat": "JSON",
            }

            try:
                response = requests.get(USGS_PRODUCTS_ENDPOINT, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()

                items = data.get("items", [])
                if not items:
                    break

                quads = [QuadInfo.from_api_response(item) for item in items]
                all_quads.extend(quads)

                logger.info(f"Retrieved {len(quads)} quads (total: {len(all_quads)})")

                # Check if more results available
                total = data.get("total", 0)
                if len(all_quads) >= total:
                    break

                offset += max_results

            except requests.RequestException as e:
                logger.error(f"API request failed at offset {offset}: {e}")
                break

        logger.info(f"Total quads for state: {len(all_quads)}")
        return all_quads

    def query_quads_for_adirondacks(self, max_results: int = 100) -> List[QuadInfo]:
        """
        Query quads specifically for the Adirondack region.

        Returns:
            List of QuadInfo objects for Adirondack area
        """
        bbox = (
            ADIRONDACK_BOUNDS["min_lon"],
            ADIRONDACK_BOUNDS["min_lat"],
            ADIRONDACK_BOUNDS["max_lon"],
            ADIRONDACK_BOUNDS["max_lat"],
        )
        return self.query_quads_for_bbox(bbox, max_results)

    def download_quad(
        self,
        quad: QuadInfo,
        skip_existing: bool = True,
    ) -> bool:
        """
        Download a single quad and upload to GCS.

        Args:
            quad: QuadInfo object
            skip_existing: Skip if already in GCS

        Returns:
            True if download successful
        """
        # Check if already downloaded
        if skip_existing and quad.source_id in self.progress.downloaded:
            logger.debug(f"Skipping {quad.quad_name} (already downloaded)")
            return True

        # Check if exists in GCS
        if skip_existing and self.bucket:
            blob = self.bucket.blob(quad.gcs_path)
            if blob.exists():
                logger.debug(f"Skipping {quad.quad_name} (exists in GCS)")
                self.progress.downloaded.append(quad.source_id)
                return True

        logger.info(f"Downloading: {quad.quad_name} ({quad.size_mb:.1f} MB)")

        try:
            # Download to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp_path = Path(tmp.name)

            response = requests.get(quad.download_url, stream=True, timeout=300)
            response.raise_for_status()

            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            actual_size = tmp_path.stat().st_size
            logger.info(f"  Downloaded {actual_size / (1024*1024):.1f} MB")

            # Upload to GCS
            if self.bucket:
                blob = self.bucket.blob(quad.gcs_path)
                blob.upload_from_filename(str(tmp_path))
                logger.info(f"  Uploaded to gs://{self.gcs_bucket}/{quad.gcs_path}")

            # Update progress
            self.progress.downloaded.append(quad.source_id)
            self.progress.total_bytes += actual_size
            self._save_progress()

            # Cleanup
            tmp_path.unlink()

            return True

        except Exception as e:
            logger.error(f"Failed to download {quad.quad_name}: {e}")
            self.progress.failed.append(quad.source_id)
            self._save_progress()
            return False

    def download_batch(
        self,
        quads: List[QuadInfo],
        max_quads: Optional[int] = None,
        max_size_gb: float = 5.0,
        skip_existing: bool = True,
    ) -> List[QuadInfo]:
        """
        Download multiple quads with constraints.

        Args:
            quads: List of QuadInfo objects to download
            max_quads: Maximum number of quads to download
            max_size_gb: Maximum total download size in GB
            skip_existing: Skip quads already downloaded

        Returns:
            List of successfully downloaded QuadInfo objects
        """
        downloaded = []
        total_size = 0.0
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024

        # Filter out already downloaded
        if skip_existing:
            quads = [q for q in quads if q.source_id not in self.progress.downloaded]

        # Apply max_quads limit
        if max_quads:
            quads = quads[:max_quads]

        logger.info(f"Starting batch download of {len(quads)} quads")
        logger.info(f"  Max size: {max_size_gb} GB")

        for i, quad in enumerate(quads):
            # Check size limit
            if total_size + quad.size_bytes > max_size_bytes:
                logger.info(f"Size limit reached ({total_size/(1024**3):.2f} GB)")
                break

            logger.info(f"\n[{i+1}/{len(quads)}] {quad.quad_name}")

            if self.download_quad(quad, skip_existing=skip_existing):
                downloaded.append(quad)
                total_size += quad.size_bytes

        logger.info(f"\n{'='*60}")
        logger.info("Batch complete:")
        logger.info(f"  Success: {len(downloaded)}")
        logger.info(f"  Failed: {len(self.progress.failed)}")
        logger.info(f"  Total size: {total_size/(1024**3):.2f} GB")
        logger.info(f"{'='*60}")

        return downloaded

    def get_quad_index(self) -> List[Dict]:
        """
        Get index of all downloaded quads with metadata.

        Returns:
            List of quad metadata dictionaries
        """
        index = []

        if not self.bucket:
            return index

        prefix = "raw/usgs_quads/"
        blobs = self.bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            if blob.name.endswith(".pdf"):
                index.append(
                    {
                        "gcs_path": blob.name,
                        "size_bytes": blob.size,
                        "updated": blob.updated.isoformat() if blob.updated else None,
                    }
                )

        return index

    def _load_progress(self):
        """Load download progress from GCS"""
        if not self.bucket:
            return

        try:
            blob = self.bucket.blob(self.PROGRESS_FILE)
            if blob.exists():
                data = json.loads(blob.download_as_text())
                self.progress = QuadDownloadProgress.from_dict(data)
                logger.info(f"Loaded progress: {len(self.progress.downloaded)} quads downloaded")
        except Exception as e:
            logger.debug(f"Could not load progress: {e}")

    def _save_progress(self):
        """Save download progress to GCS"""
        if not self.bucket:
            return

        self.progress.last_updated = datetime.utcnow().isoformat()

        try:
            blob = self.bucket.blob(self.PROGRESS_FILE)
            blob.upload_from_string(
                json.dumps(self.progress.to_dict(), indent=2), content_type="application/json"
            )
        except Exception as e:
            logger.error(f"Could not save progress: {e}")


def list_available_quads(
    bbox: Optional[Tuple[float, float, float, float]] = None,
    state_fips: str = NY_STATE_FIPS,
) -> List[QuadInfo]:
    """
    Convenience function to list available USGS quads.

    Args:
        bbox: Optional bounding box (min_lon, min_lat, max_lon, max_lat)
        state_fips: State FIPS code if no bbox provided

    Returns:
        List of QuadInfo objects
    """
    downloader = USGSQuadDownloader.__new__(USGSQuadDownloader)
    downloader._gcs_client = None
    downloader._bucket = None
    downloader.progress = QuadDownloadProgress()

    if bbox:
        return downloader.query_quads_for_bbox(bbox)
    else:
        return downloader.query_quads_for_state(state_fips)
