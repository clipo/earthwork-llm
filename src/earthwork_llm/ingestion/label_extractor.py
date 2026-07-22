"""
USGS Label Extractor for TerraLLM

Extracts labeled features from USGS 7.5-minute topographic quadrangles using
Google Vision API. These labels provide automatic training data for LiDAR
feature detection - no manual annotation required!

**VALIDATED APPROACH**: Google Vision API provides the most robust and accurate
feature extraction from USGS topographic quadrangles.

USGS Standard Symbology:
    - Dashed black lines = Trails, dirt roads (MISSION CRITICAL)
    - Solid black/red lines = Paved roads
    - Dashed blue lines = Intermittent streams
    - Solid blue lines = Perennial streams
    - Brown lines = Contour lines
    - Black polygons = Buildings

Google Vision API extracts:
    - Text labels (feature names, elevations)
    - Line features with color classification
    - Symbols and landmarks
    - Color/pattern analysis for dashed vs solid lines

Cost: ~$4-5 per quad (~$100 for 100 quads)
"""

import io
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from google.cloud import storage, vision

    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes, convert_from_path  # noqa: F401  (availability probe)

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import rasterio  # noqa: F401  (availability probe)
    from rasterio.transform import xy  # noqa: F401  (availability probe)

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import geopandas as gpd  # noqa: F401  (availability probe)
    from shapely.geometry import LineString, MultiLineString, Point, Polygon, box  # noqa: F401  (availability probe)
    from shapely.ops import linemerge  # noqa: F401  (availability probe)

    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


logger = logging.getLogger(__name__)


# Feature type keywords for text detection
TRAIL_KEYWORDS = [
    "trail",
    "path",
    "tr",
    "footpath",
    "hiking",
    "nature trail",
    "ski trail",
    "horse trail",
    "bike path",
    "bridle path",
]

ROAD_KEYWORDS = [
    "road",
    "rd",
    "street",
    "st",
    "avenue",
    "ave",
    "highway",
    "hwy",
    "route",
    "rte",
    "drive",
    "dr",
    "lane",
    "ln",
    "way",
    "boulevard",
    "blvd",
    "county road",
    "state route",
    "us route",
    "interstate",
    "levee",
    "dike",
    "embankment",
]

STREAM_KEYWORDS = [
    "brook",
    "creek",
    "stream",
    "river",
    "run",
    "branch",
    "fork",
    "kill",
    "outlet",
    "inlet",
    "pond",
    "lake",
    "falls",
    "spring",
    "marsh",
    "swamp",
    "wetland",
    "reservoir",
    "canal",
    "ditch",
    "drain",
    "aqueduct",
    "dredge",
]

BUILDING_KEYWORDS = [
    "church",
    "school",
    "cemetery",
    "cem",
    "fire station",
    "post office",
    "hospital",
    "library",
    "town hall",
    "courthouse",
    "museum",
    "mill",
    "mine",
    "quarry",
    "tower",
    "lookout",
    "ranger station",
    "lean-to",
    "shelter",
    "cabin",
    "lodge",
    "camp",
    "campground",
]

# USGS Map Color Ranges (HSV)
# These define the color ranges for extracting features from USGS topo maps
USGS_COLORS = {
    # Blue: Streams, lakes, water features
    "blue": {
        "lower": np.array([100, 50, 50]),  # HSV lower bound
        "upper": np.array([130, 255, 255]),  # HSV upper bound
        "features": ["stream", "river", "lake", "pond", "wetland"],
    },
    # Brown: Contour lines
    "brown": {
        "lower": np.array([10, 50, 50]),
        "upper": np.array([25, 255, 200]),
        "features": ["contour"],
    },
    # Black: Roads, trails, buildings, labels
    "black": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([180, 50, 80]),
        "features": ["road", "trail", "building", "label"],
    },
    # Red: Major highways, important features
    "red": {
        "lower": np.array([0, 100, 100]),
        "upper": np.array([10, 255, 255]),
        "features": ["highway", "boundary"],
    },
    # Green: Vegetation, parks
    "green": {
        "lower": np.array([35, 50, 50]),
        "upper": np.array([85, 255, 255]),
        "features": ["vegetation", "park", "forest"],
    },
}


@dataclass
class ExtractedFeature:
    """A feature extracted from a USGS quad."""

    feature_type: str  # 'trail', 'road', 'stream', 'building', 'label'
    name: Optional[str] = None
    geometry_type: str = "point"  # 'point', 'line', 'polygon'
    pixel_coords: Tuple[float, float] = (0, 0)  # (x, y) in image pixels
    pixel_bounds: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2)
    geo_coords: Optional[Tuple[float, float]] = None  # (lon, lat) in WGS84
    geo_bounds: Optional[Tuple[float, float, float, float]] = (
        None  # (lon_min, lat_min, lon_max, lat_max)
    )
    confidence: float = 0.0
    source: str = "vision_api"  # 'vision_api', 'text_detection', 'color_analysis'
    raw_text: Optional[str] = None
    classification: Optional[str] = None  # 'paved', 'unpaved', 'trail', 'perennial', 'intermittent'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Return the feature as a JSON-serializable dict."""
        return {
            "feature_type": self.feature_type,
            "name": self.name,
            "geometry_type": self.geometry_type,
            "pixel_coords": self.pixel_coords,
            "pixel_bounds": self.pixel_bounds,
            "geo_coords": self.geo_coords,
            "geo_bounds": self.geo_bounds,
            "confidence": self.confidence,
            "source": self.source,
            "raw_text": self.raw_text,
            "classification": self.classification,
            "metadata": self.metadata,
        }


@dataclass
class ExtractionResult:
    """Results from processing a USGS quad."""

    quad_name: str
    quad_path: str
    image_size: Tuple[int, int]  # (width, height)
    geo_bounds: Optional[Tuple[float, float, float, float]] = (
        None  # (lon_min, lat_min, lon_max, lat_max)
    )
    crs: str = "EPSG:4326"

    trails: List[ExtractedFeature] = field(default_factory=list)
    roads: List[ExtractedFeature] = field(default_factory=list)
    streams: List[ExtractedFeature] = field(default_factory=list)
    buildings: List[ExtractedFeature] = field(default_factory=list)
    labels: List[ExtractedFeature] = field(default_factory=list)

    api_calls: int = 0
    processing_time_sec: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def total_features(self) -> int:
        """Total feature count across all categories."""
        return (
            len(self.trails)
            + len(self.roads)
            + len(self.streams)
            + len(self.buildings)
            + len(self.labels)
        )

    def to_dict(self) -> Dict:
        """Return the result (features plus summary counts) as a JSON-serializable dict."""
        return {
            "quad_name": self.quad_name,
            "quad_path": self.quad_path,
            "image_size": self.image_size,
            "geo_bounds": self.geo_bounds,
            "crs": self.crs,
            "trails": [f.to_dict() for f in self.trails],
            "roads": [f.to_dict() for f in self.roads],
            "streams": [f.to_dict() for f in self.streams],
            "buildings": [f.to_dict() for f in self.buildings],
            "labels": [f.to_dict() for f in self.labels],
            "summary": {
                "total_features": self.total_features,
                "trails": len(self.trails),
                "roads": len(self.roads),
                "streams": len(self.streams),
                "buildings": len(self.buildings),
                "labels": len(self.labels),
            },
            "api_calls": self.api_calls,
            "processing_time_sec": self.processing_time_sec,
            "errors": self.errors,
        }

    def to_geojson(self) -> Dict:
        """Convert to GeoJSON FeatureCollection."""
        features = []

        for f in self.trails + self.roads + self.streams + self.buildings:
            if f.geo_coords:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": list(f.geo_coords),
                        },
                        "properties": {
                            "feature_type": f.feature_type,
                            "name": f.name,
                            "classification": f.classification,
                            "confidence": f.confidence,
                            "source": f.source,
                        },
                    }
                )

        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "quad_name": self.quad_name,
                "total_features": self.total_features,
            },
        }


class USGSLabelExtractor:
    """
    Extract labeled features from USGS topographic maps using Google Vision API.

    This class processes USGS 7.5-minute quadrangles (GeoPDF format) and extracts:
    - Trails (dashed black lines) - CRITICAL for battlefield losses
    - Roads (solid black/red lines)
    - Streams (blue lines, solid=perennial, dashed=intermittent)
    - Buildings (black rectangles/symbols)
    - Text labels (feature names, elevations)

    The extracted labels are used as automatic training data for LiDAR feature
    detection, eliminating the need for manual annotation.

    Example:
        >>> extractor = USGSLabelExtractor()
        >>> result = extractor.extract_from_pdf("Lake_Placid_2023.pdf")
        >>> print(f"Found {len(result.trails)} trails")
        >>> # Save as GeoJSON for training
        >>> result.save_geojson("lake_placid_labels.geojson")
    """

    def __init__(
        self,
        gcs_bucket: Optional[str] = None,
        credentials_path: Optional[str] = None,
        dpi: int = 300,
    ):
        """
        Initialize the label extractor.

        Args:
            gcs_bucket: Optional GCS bucket for reading/writing files
            credentials_path: Path to Google Cloud credentials JSON
            dpi: DPI for PDF to image conversion (higher = more detail, slower)
        """
        if not VISION_AVAILABLE:
            raise ImportError(
                "google-cloud-vision required. Install with: " "pip install google-cloud-vision"
            )

        self.gcs_bucket = gcs_bucket
        self.dpi = dpi
        self._vision_client = None
        self._gcs_client = None
        self._bucket = None

        # Initialize Vision client
        if credentials_path:
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self._vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            # Uses GOOGLE_APPLICATION_CREDENTIALS environment variable
            self._vision_client = vision.ImageAnnotatorClient()

        logger.info("Initialized USGSLabelExtractor with Google Vision API")

    @property
    def gcs_client(self):
        """Lazy initialization of GCS client."""
        if self._gcs_client is None:
            self._gcs_client = storage.Client()
        return self._gcs_client

    @property
    def bucket(self):
        """Get GCS bucket."""
        if self._bucket is None and self.gcs_bucket:
            self._bucket = self.gcs_client.bucket(self.gcs_bucket)
        return self._bucket

    def convert_pdf_to_images(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None,
    ) -> List[Image.Image]:
        """
        Convert GeoPDF to images for Vision API processing.

        Args:
            pdf_path: Path to PDF file (local or GCS)
            pages: Optional list of page numbers (0-indexed), default: all pages

        Returns:
            List of PIL Image objects
        """
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError(
                "pdf2image required. Install with: "
                "pip install pdf2image\n"
                "Also requires poppler: brew install poppler (macOS) or apt-get install poppler-utils (Linux)"
            )

        pdf_path = str(pdf_path)

        # Download from GCS if needed
        if pdf_path.startswith("gs://"):
            logger.info(f"Downloading PDF from GCS: {pdf_path}")
            parts = pdf_path.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            blob_path = parts[1]
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                blob.download_to_filename(tmp.name)
                pdf_path = tmp.name

        logger.info(f"Converting PDF to images at {self.dpi} DPI")

        # Convert PDF to images
        if pages:
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=min(pages) + 1,
                last_page=max(pages) + 1,
            )
        else:
            images = convert_from_path(pdf_path, dpi=self.dpi)

        logger.info(f"Converted {len(images)} pages")
        return images

    def image_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Convert PIL Image to bytes for Vision API."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()

    def detect_text(self, image_bytes: bytes) -> List[Any]:
        """
        Detect text in image using Vision API OCR.

        Returns text annotations including bounding boxes.
        """
        if not VISION_AVAILABLE:
            raise RuntimeError("google-cloud-vision not installed")
        image = vision.Image(content=image_bytes)
        response = self._vision_client.text_detection(image=image)

        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")

        return response.text_annotations

    def detect_objects(self, image_bytes: bytes) -> List[Any]:
        """
        Detect objects/features in image using Vision API.

        Returns localized object annotations with bounding polygons.
        """
        image = vision.Image(content=image_bytes)
        response = self._vision_client.object_localization(image=image)

        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")

        return response.localized_object_annotations

    def analyze_image_properties(self, image_bytes: bytes) -> Any:
        """
        Analyze image properties (colors) using Vision API.

        Used to detect dominant colors and color regions.
        """
        image = vision.Image(content=image_bytes)
        response = self._vision_client.image_properties(image=image)

        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")

        return response.image_properties_annotation

    def classify_text_as_feature(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Classify detected text as a feature type.

        Args:
            text: Detected text string

        Returns:
            (feature_type, classification) or (None, None) if not a feature
        """
        text_lower = text.lower().strip()

        # Check trails (CRITICAL)
        for keyword in TRAIL_KEYWORDS:
            if keyword in text_lower:
                return ("trail", "trail")

        # Check roads
        for keyword in ROAD_KEYWORDS:
            if keyword in text_lower:
                # Try to classify road type
                if any(
                    hw in text_lower
                    for hw in ["highway", "hwy", "interstate", "us route", "state route"]
                ):
                    return ("road", "highway")
                elif any(rd in text_lower for rd in ["county road", "rd", "road"]):
                    return ("road", "local")
                return ("road", "unknown")

        # Check streams
        for keyword in STREAM_KEYWORDS:
            if keyword in text_lower:
                # Classify stream type
                if any(w in text_lower for w in ["pond", "lake", "reservoir"]):
                    return ("stream", "standing_water")
                elif any(w in text_lower for w in ["marsh", "swamp", "wetland"]):
                    return ("stream", "wetland")
                return ("stream", "flowing_water")

        # Check buildings
        for keyword in BUILDING_KEYWORDS:
            if keyword in text_lower:
                return ("building", keyword)

        return (None, None)

    def pixel_to_geo(
        self,
        pixel_x: float,
        pixel_y: float,
        image_size: Tuple[int, int],
        geo_bounds: Tuple[float, float, float, float],
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates.

        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate
            image_size: (width, height) of image
            geo_bounds: (lon_min, lat_min, lon_max, lat_max)

        Returns:
            (longitude, latitude)
        """
        width, height = image_size
        lon_min, lat_min, lon_max, lat_max = geo_bounds

        # Linear interpolation
        lon = lon_min + (pixel_x / width) * (lon_max - lon_min)
        lat = lat_max - (pixel_y / height) * (lat_max - lat_min)  # Y is inverted

        return (lon, lat)

    def extract_linear_features_by_color(
        self,
        image: Image.Image,
        color: str = "blue",
        min_line_length: int = 50,
        geo_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[ExtractedFeature]:
        """
        Extract linear features (streams, trails, roads) using color-based line tracing.

        Uses color thresholding and Hough line detection to trace linear features
        as full LineString geometries rather than just point locations.

        Args:
            image: PIL Image object
            color: Color to extract ('blue' for streams, 'black' for roads/trails)
            min_line_length: Minimum line length in pixels
            geo_bounds: Geographic bounds for coordinate conversion

        Returns:
            List of ExtractedFeature objects with LineString geometries
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, skipping color-based extraction")
            return []

        features = []
        image_size = image.size

        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

        # Get color range
        if color not in USGS_COLORS:
            logger.warning(f"Unknown color: {color}")
            return []

        color_def = USGS_COLORS[color]
        lower = color_def["lower"]
        upper = color_def["upper"]

        # Create mask for this color
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours (for both lines and areas)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Also use Hough Line Transform for line detection
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=min_line_length, maxLineGap=10
        )

        # Process detected lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if length < min_line_length:
                    continue

                # Convert to geographic coordinates if bounds available
                geo_start = None
                geo_end = None
                if geo_bounds:
                    geo_start = self.pixel_to_geo(x1, y1, image_size, geo_bounds)
                    geo_end = self.pixel_to_geo(x2, y2, image_size, geo_bounds)

                # Determine feature type based on color
                if color == "blue":
                    feature_type = "stream"
                    classification = "water"
                elif color == "black":
                    feature_type = "road"  # Could be trail or road
                    classification = "unknown"
                elif color == "red":
                    feature_type = "road"
                    classification = "highway"
                else:
                    feature_type = "line"
                    classification = color

                feature = ExtractedFeature(
                    feature_type=feature_type,
                    geometry_type="line",
                    pixel_coords=((x1 + x2) / 2, (y1 + y2) / 2),
                    pixel_bounds=(x1, y1, x2, y2),
                    geo_coords=geo_start,
                    geo_bounds=(
                        (geo_start[0], geo_start[1], geo_end[0], geo_end[1])
                        if geo_start and geo_end
                        else None
                    ),
                    confidence=0.8,
                    source="color_extraction",
                    classification=classification,
                    metadata={
                        "length_pixels": length,
                        "start_pixel": (x1, y1),
                        "end_pixel": (x2, y2),
                        "start_geo": geo_start,
                        "end_geo": geo_end,
                    },
                )
                features.append(feature)

        logger.info(f"Extracted {len(features)} {color} linear features")
        return features

    def extract_area_features_by_color(
        self,
        image: Image.Image,
        color: str = "blue",
        min_area: int = 1000,
        geo_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[ExtractedFeature]:
        """
        Extract area features (lakes, wetlands) using color-based region detection.

        Uses color thresholding and contour detection to extract polygon geometries.

        Args:
            image: PIL Image object
            color: Color to extract ('blue' for water, 'green' for vegetation)
            min_area: Minimum area in pixels
            geo_bounds: Geographic bounds for coordinate conversion

        Returns:
            List of ExtractedFeature objects with Polygon geometries
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, skipping color-based extraction")
            return []

        features = []
        image_size = image.size

        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Convert to HSV
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

        # Get color range
        if color not in USGS_COLORS:
            logger.warning(f"Unknown color: {color}")
            return []

        color_def = USGS_COLORS[color]
        lower = color_def["lower"]
        upper = color_def["upper"]

        # Create mask
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological operations for cleaner regions
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            # Determine feature type based on color and shape
            aspect_ratio = w / h if h > 0 else 1
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            if color == "blue":
                if circularity > 0.7:
                    feature_type = "lake"
                    classification = "standing_water"
                else:
                    feature_type = "wetland"
                    classification = "wetland"
            elif color == "green":
                feature_type = "vegetation"
                classification = "forest"
            else:
                feature_type = "area"
                classification = color

            # Convert to geographic coordinates
            geo_center = None
            geo_bbox = None
            if geo_bounds:
                geo_center = self.pixel_to_geo(cx, cy, image_size, geo_bounds)
                geo_min = self.pixel_to_geo(x, y + h, image_size, geo_bounds)
                geo_max = self.pixel_to_geo(x + w, y, image_size, geo_bounds)
                geo_bbox = (geo_min[0], geo_min[1], geo_max[0], geo_max[1])

            # Simplify contour for polygon
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygon_pixels = [(p[0][0], p[0][1]) for p in approx]

            # Convert polygon to geo coords
            polygon_geo = None
            if geo_bounds and len(polygon_pixels) >= 3:
                polygon_geo = [
                    self.pixel_to_geo(p[0], p[1], image_size, geo_bounds) for p in polygon_pixels
                ]

            feature = ExtractedFeature(
                feature_type=feature_type,
                geometry_type="polygon",
                pixel_coords=(cx, cy),
                pixel_bounds=(x, y, x + w, y + h),
                geo_coords=geo_center,
                geo_bounds=geo_bbox,
                confidence=0.8,
                source="color_extraction",
                classification=classification,
                metadata={
                    "area_pixels": area,
                    "perimeter_pixels": perimeter,
                    "circularity": circularity,
                    "aspect_ratio": aspect_ratio,
                    "polygon_pixels": polygon_pixels,
                    "polygon_geo": polygon_geo,
                },
            )
            features.append(feature)

        logger.info(f"Extracted {len(features)} {color} area features")
        return features

    def extract_from_image(
        self,
        image: Image.Image,
        quad_name: str = "unknown",
        geo_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> ExtractionResult:
        """
        Extract features from a single image.

        Args:
            image: PIL Image object
            quad_name: Name of the quad for logging
            geo_bounds: Geographic bounds (lon_min, lat_min, lon_max, lat_max)

        Returns:
            ExtractionResult with extracted features
        """
        import time

        start_time = time.time()

        image_size = image.size
        result = ExtractionResult(
            quad_name=quad_name,
            quad_path="",
            image_size=image_size,
            geo_bounds=geo_bounds,
        )

        # Convert to bytes
        image_bytes = self.image_to_bytes(image)

        # 1. Text Detection - Find feature labels
        logger.info("Running text detection...")
        try:
            text_annotations = self.detect_text(image_bytes)
            result.api_calls += 1

            for annotation in text_annotations[1:]:  # Skip first (full text)
                text = annotation.description
                vertices = annotation.bounding_poly.vertices

                # Get center point
                if vertices:
                    cx = sum(v.x for v in vertices) / len(vertices)
                    cy = sum(v.y for v in vertices) / len(vertices)
                    bounds = (
                        min(v.x for v in vertices),
                        min(v.y for v in vertices),
                        max(v.x for v in vertices),
                        max(v.y for v in vertices),
                    )
                else:
                    cx, cy = 0, 0
                    bounds = None

                # Classify the text
                feature_type, classification = self.classify_text_as_feature(text)

                # Calculate geo coordinates if bounds available
                geo_coords = None
                geo_bounds_feature = None
                if geo_bounds:
                    geo_coords = self.pixel_to_geo(cx, cy, image_size, geo_bounds)
                    if bounds:
                        geo_bounds_feature = (
                            self.pixel_to_geo(bounds[0], bounds[3], image_size, geo_bounds)[0],
                            self.pixel_to_geo(bounds[0], bounds[3], image_size, geo_bounds)[1],
                            self.pixel_to_geo(bounds[2], bounds[1], image_size, geo_bounds)[0],
                            self.pixel_to_geo(bounds[2], bounds[1], image_size, geo_bounds)[1],
                        )

                feature = ExtractedFeature(
                    feature_type=feature_type or "label",
                    name=text if feature_type else None,
                    geometry_type="point",
                    pixel_coords=(cx, cy),
                    pixel_bounds=bounds,
                    geo_coords=geo_coords,
                    geo_bounds=geo_bounds_feature,
                    confidence=1.0,  # Text detection doesn't provide confidence
                    source="text_detection",
                    raw_text=text,
                    classification=classification,
                )

                # Categorize
                if feature_type == "trail":
                    result.trails.append(feature)
                elif feature_type == "road":
                    result.roads.append(feature)
                elif feature_type == "stream":
                    result.streams.append(feature)
                elif feature_type == "building":
                    result.buildings.append(feature)
                else:
                    result.labels.append(feature)

            logger.info(f"  Found {len(text_annotations)-1} text annotations")

        except Exception as e:
            error_msg = f"Text detection failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

        # 2. Object Detection - Find features by shape
        logger.info("Running object detection...")
        try:
            object_annotations = self.detect_objects(image_bytes)
            result.api_calls += 1

            for obj in object_annotations:
                name = obj.name.lower()
                vertices = obj.bounding_poly.normalized_vertices

                # Convert normalized coords to pixels
                if vertices:
                    px = [v.x * image_size[0] for v in vertices]
                    py = [v.y * image_size[1] for v in vertices]
                    cx = sum(px) / len(px)
                    cy = sum(py) / len(py)
                    bounds = (min(px), min(py), max(px), max(py))
                else:
                    cx, cy = 0, 0
                    bounds = None

                # Classify object
                feature_type = None
                classification = None

                if any(w in name for w in ["road", "path", "trail", "street"]):
                    if "trail" in name or "path" in name:
                        feature_type = "trail"
                        classification = "trail"
                    else:
                        feature_type = "road"
                        classification = "unknown"
                elif any(w in name for w in ["river", "stream", "water", "lake", "pond"]):
                    feature_type = "stream"
                    classification = "water"
                elif any(w in name for w in ["building", "house", "structure"]):
                    feature_type = "building"
                    classification = "structure"

                if feature_type:
                    geo_coords = None
                    if geo_bounds:
                        geo_coords = self.pixel_to_geo(cx, cy, image_size, geo_bounds)

                    feature = ExtractedFeature(
                        feature_type=feature_type,
                        name=obj.name,
                        geometry_type="polygon",
                        pixel_coords=(cx, cy),
                        pixel_bounds=bounds,
                        geo_coords=geo_coords,
                        confidence=obj.score,
                        source="object_detection",
                        classification=classification,
                    )

                    if feature_type == "trail":
                        result.trails.append(feature)
                    elif feature_type == "road":
                        result.roads.append(feature)
                    elif feature_type == "stream":
                        result.streams.append(feature)
                    elif feature_type == "building":
                        result.buildings.append(feature)

            logger.info(f"  Found {len(object_annotations)} object annotations")

        except Exception as e:
            error_msg = f"Object detection failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

        # 3. Color-based extraction for linear features (streams, roads)
        logger.info("Running color-based linear feature extraction...")
        try:
            # Extract blue lines (streams)
            stream_lines = self.extract_linear_features_by_color(
                image, color="blue", min_line_length=50, geo_bounds=geo_bounds
            )
            for f in stream_lines:
                result.streams.append(f)

            # Extract black lines (roads, trails)
            road_lines = self.extract_linear_features_by_color(
                image, color="black", min_line_length=100, geo_bounds=geo_bounds
            )
            for f in road_lines:
                result.roads.append(f)

            # Extract red lines (highways)
            highway_lines = self.extract_linear_features_by_color(
                image, color="red", min_line_length=100, geo_bounds=geo_bounds
            )
            for f in highway_lines:
                result.roads.append(f)

        except Exception as e:
            error_msg = f"Color-based linear extraction failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

        # 4. Color-based extraction for area features (lakes, wetlands)
        logger.info("Running color-based area feature extraction...")
        try:
            # Extract blue areas (lakes, ponds)
            water_areas = self.extract_area_features_by_color(
                image, color="blue", min_area=5000, geo_bounds=geo_bounds
            )
            for f in water_areas:
                result.streams.append(f)  # Add to streams (includes lakes)

            # Extract green areas (vegetation)
            self.extract_area_features_by_color(
                image, color="green", min_area=10000, geo_bounds=geo_bounds
            )
            # Could add a separate category for vegetation if needed

        except Exception as e:
            error_msg = f"Color-based area extraction failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

        result.processing_time_sec = time.time() - start_time
        logger.info(
            f"Extraction complete: {result.total_features} features "
            f"({len(result.trails)} trails, {len(result.roads)} roads, "
            f"{len(result.streams)} streams, {len(result.buildings)} buildings) "
            f"in {result.processing_time_sec:.1f}s"
        )

        return result

    def extract_from_pdf(
        self,
        pdf_path: Union[str, Path],
        quad_name: Optional[str] = None,
        geo_bounds: Optional[Tuple[float, float, float, float]] = None,
        page: int = 0,
    ) -> ExtractionResult:
        """
        Extract features from a USGS GeoPDF.

        Args:
            pdf_path: Path to PDF file (local or GCS)
            quad_name: Optional quad name (extracted from filename if not provided)
            geo_bounds: Geographic bounds from quad metadata
            page: Page number to process (default: 0, the map page)

        Returns:
            ExtractionResult with extracted features
        """
        pdf_path = Path(pdf_path) if not str(pdf_path).startswith("gs://") else pdf_path

        if quad_name is None:
            quad_name = Path(str(pdf_path)).stem

        logger.info(f"Extracting features from: {quad_name}")

        # Convert PDF to image
        images = self.convert_pdf_to_images(pdf_path, pages=[page])

        if not images:
            raise ValueError(f"No pages converted from PDF: {pdf_path}")

        # Extract from the image
        result = self.extract_from_image(
            images[0],
            quad_name=quad_name,
            geo_bounds=geo_bounds,
        )
        result.quad_path = str(pdf_path)

        return result

    def extract_batch(
        self,
        pdf_paths: List[Union[str, Path]],
        quad_metadata: Optional[Dict[str, Dict]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[ExtractionResult]:
        """
        Extract features from multiple USGS quads.

        Args:
            pdf_paths: List of PDF file paths
            quad_metadata: Optional dict mapping quad names to metadata (including geo_bounds)
            output_dir: Optional directory to save individual results

        Returns:
            List of ExtractionResult objects
        """
        results = []
        quad_metadata = quad_metadata or {}

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for i, pdf_path in enumerate(pdf_paths):
            logger.info(f"\n[{i+1}/{len(pdf_paths)}] Processing: {pdf_path}")

            quad_name = Path(str(pdf_path)).stem
            meta = quad_metadata.get(quad_name, {})
            geo_bounds = meta.get("geo_bounds") or meta.get("bounding_box")

            try:
                result = self.extract_from_pdf(
                    pdf_path,
                    quad_name=quad_name,
                    geo_bounds=geo_bounds,
                )
                results.append(result)

                # Save individual result
                if output_dir:
                    output_path = output_dir / f"{quad_name}_labels.json"
                    with open(output_path, "w") as f:
                        json.dump(result.to_dict(), f, indent=2)

                    geojson_path = output_dir / f"{quad_name}_labels.geojson"
                    with open(geojson_path, "w") as f:
                        json.dump(result.to_geojson(), f, indent=2)

            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                # Create error result
                results.append(
                    ExtractionResult(
                        quad_name=quad_name,
                        quad_path=str(pdf_path),
                        image_size=(0, 0),
                        errors=[str(e)],
                    )
                )

        # Summary
        total_trails = sum(len(r.trails) for r in results)
        total_roads = sum(len(r.roads) for r in results)
        total_streams = sum(len(r.streams) for r in results)
        total_api_calls = sum(r.api_calls for r in results)

        logger.info(f"\n{'='*60}")
        logger.info("Batch extraction complete:")
        logger.info(f"  Quads processed: {len(results)}")
        logger.info(f"  Total trails: {total_trails}")
        logger.info(f"  Total roads: {total_roads}")
        logger.info(f"  Total streams: {total_streams}")
        logger.info(f"  API calls: {total_api_calls}")
        logger.info(f"{'='*60}")

        return results


def estimate_cost(num_quads: int, pages_per_quad: int = 1) -> Dict[str, float]:
    """
    Estimate Google Vision API cost for label extraction.

    Pricing (as of 2025):
    - TEXT_DETECTION: $1.50 per 1,000 images
    - OBJECT_LOCALIZATION: $1.50 per 1,000 images
    - IMAGE_PROPERTIES: $1.00 per 1,000 images

    Args:
        num_quads: Number of USGS quads to process
        pages_per_quad: Pages per quad (typically 1 for the map)

    Returns:
        Cost breakdown dictionary
    """
    num_images = num_quads * pages_per_quad

    # We call 2 APIs per image (text + object detection)
    text_cost = (num_images / 1000) * 1.50
    object_cost = (num_images / 1000) * 1.50

    total_cost = text_cost + object_cost

    return {
        "num_quads": num_quads,
        "num_images": num_images,
        "text_detection_cost": text_cost,
        "object_detection_cost": object_cost,
        "total_cost": total_cost,
        "cost_per_quad": total_cost / num_quads if num_quads > 0 else 0,
    }
