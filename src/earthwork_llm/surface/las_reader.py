"""
LAS/LAZ File Reader for TerraLLM

Reads LAS and LAZ point cloud files using PDAL for robust processing.
PDAL provides powerful capabilities including:
- Built-in ground classification (SMRF, PMF algorithms)
- Noise filtering and outlier removal
- Pipeline-based processing for complex workflows
- Support for large point clouds with streaming

Bridges the gap between downloaded LiDAR tiles and the DEMGenerator/GeomorphonAnalyzer.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pdal

    PDAL_AVAILABLE = True
except ImportError:
    PDAL_AVAILABLE = False

# Fallback to laspy for basic reading if PDAL not available
try:
    import laspy

    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False

try:
    from google.cloud import storage

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logger = logging.getLogger(__name__)


# LAS point classification codes (ASPRS standard)
class PointClassification:
    """ASPRS LAS point classification codes"""

    CREATED_NEVER_CLASSIFIED = 0
    UNCLASSIFIED = 1
    GROUND = 2
    LOW_VEGETATION = 3
    MEDIUM_VEGETATION = 4
    HIGH_VEGETATION = 5
    BUILDING = 6
    LOW_POINT_NOISE = 7
    MODEL_KEY_POINT = 8
    WATER = 9
    RAIL = 10
    ROAD_SURFACE = 11
    OVERLAP = 12
    WIRE_GUARD = 13
    WIRE_CONDUCTOR = 14
    TRANSMISSION_TOWER = 15
    WIRE_STRUCTURE_CONNECTOR = 16
    BRIDGE_DECK = 17
    HIGH_NOISE = 18


@dataclass
class PointCloudMetadata:
    """
    Metadata extracted from LAS/LAZ file.

    Attributes:
        filename: Source file name
        point_count: Total number of points (after filtering)
        original_point_count: Total points in file before filtering
        bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
        crs_wkt: Coordinate reference system as WKT string
        las_version: LAS format version (e.g., "1.4")
        point_format: LAS point format ID
        classifications: Dict of classification code -> count
        scale: (scale_x, scale_y, scale_z) factors
        offset: (offset_x, offset_y, offset_z) values
    """

    filename: str
    point_count: int
    original_point_count: int
    bounds: Tuple[float, float, float, float, float, float]
    crs_wkt: Optional[str]
    las_version: str
    point_format: int
    classifications: Dict[int, int]
    scale: Tuple[float, float, float]
    offset: Tuple[float, float, float]


class LASReader:
    """
    Reads LAS/LAZ files using PDAL for robust point cloud processing.

    PDAL provides powerful capabilities beyond basic reading:
    - Ground classification using SMRF or PMF algorithms
    - Noise filtering and outlier removal
    - Height normalization
    - Pipeline-based processing

    Supports:
    - LAS 1.0-1.4 and LAZ (compressed) formats
    - Classification filtering (ground points, buildings, etc.)
    - Built-in ground classification for unclassified data
    - Point cloud thinning for memory efficiency
    - Local files and Google Cloud Storage paths
    - Coordinate reference system extraction

    Example:
        >>> reader = LASReader()
        >>> points, metadata = reader.read_las("terrain.laz", classification=2)
        >>> # points is (N, 3) array with [x, y, z] columns
        >>> print(f"Loaded {len(points)} ground points")

        >>> # Classify ground points using SMRF algorithm
        >>> points, meta = reader.read_with_ground_classification("unclassified.laz")

        >>> # Use with DEMGenerator
        >>> from earthwork_llm.surface import DEMGenerator
        >>> dem_gen = DEMGenerator()
        >>> dem, dem_meta = dem_gen.generate_dem(points)
    """

    # Ground classification algorithms
    GROUND_ALGORITHMS = ["smrf", "pmf", "csf"]

    def __init__(self, gcs_bucket: Optional[str] = None):
        """
        Initialize LAS reader.

        Args:
            gcs_bucket: Optional GCS bucket name for reading cloud files
        """
        if not PDAL_AVAILABLE:
            if not LASPY_AVAILABLE:
                raise ImportError(
                    "PDAL or laspy is required for reading LAS files. "
                    "Install PDAL with: conda install -c conda-forge pdal python-pdal "
                    "Or install laspy with: pip install laspy[lazrs]"
                )
            logger.warning(
                "PDAL not available, falling back to laspy. "
                "Install PDAL for ground classification: conda install -c conda-forge pdal python-pdal"
            )

        self.gcs_bucket = gcs_bucket
        self._gcs_client = None
        self.use_pdal = PDAL_AVAILABLE

        logger.info(f"Initialized LASReader (backend: {'PDAL' if self.use_pdal else 'laspy'})")

    @property
    def gcs_client(self):
        """Lazy initialization of GCS client"""
        if self._gcs_client is None and GCS_AVAILABLE:
            self._gcs_client = storage.Client()
        return self._gcs_client

    def read_las(
        self,
        file_path: Union[str, Path],
        classification: Optional[Union[int, List[int]]] = None,
        thin_factor: int = 1,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        max_points: Optional[int] = None,
    ) -> Tuple[np.ndarray, PointCloudMetadata]:
        """
        Read LAS/LAZ file and return point cloud.

        Args:
            file_path: Path to LAS/LAZ file (local or gs:// URI)
            classification: Filter to specific classification(s).
                - None: Return all points
                - 2: Return only ground points
                - [2, 6]: Return ground and building points
            thin_factor: Keep every Nth point (1=all, 2=half, etc.)
            bounds: Optional spatial filter (xmin, ymin, xmax, ymax)
            max_points: Maximum number of points to return

        Returns:
            Tuple of:
                - points: numpy array of shape (N, 3) with [x, y, z] columns
                - metadata: PointCloudMetadata with file information

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = str(file_path)

        # Handle GCS paths
        if file_path.startswith("gs://"):
            local_path = self._download_from_gcs(file_path)
            cleanup_after = True
        else:
            local_path = Path(file_path)
            cleanup_after = False

            if not local_path.exists():
                raise FileNotFoundError(f"LAS file not found: {file_path}")

        try:
            if self.use_pdal:
                return self._read_with_pdal(
                    local_path, file_path, classification, thin_factor, bounds, max_points
                )
            else:
                return self._read_with_laspy(
                    local_path, file_path, classification, thin_factor, bounds, max_points
                )
        finally:
            if cleanup_after and local_path.exists():
                os.unlink(local_path)

    def _read_with_pdal(
        self,
        local_path: Path,
        original_path: str,
        classification: Optional[Union[int, List[int]]],
        thin_factor: int,
        bounds: Optional[Tuple[float, float, float, float]],
        max_points: Optional[int],
    ) -> Tuple[np.ndarray, PointCloudMetadata]:
        """Read using PDAL pipeline"""
        logger.info(f"Reading LAS file with PDAL: {original_path}")

        # Build PDAL pipeline
        pipeline_stages = [str(local_path)]

        # Add spatial filter if bounds specified
        if bounds is not None:
            xmin, ymin, xmax, ymax = bounds
            pipeline_stages.append(
                {"type": "filters.crop", "bounds": f"([{xmin},{xmax}],[{ymin},{ymax}])"}
            )

        # Add classification filter
        if classification is not None:
            if isinstance(classification, int):
                classification = [classification]
            class_expr = " || ".join([f"Classification == {c}" for c in classification])
            pipeline_stages.append({"type": "filters.expression", "expression": class_expr})

        # Add thinning filter
        if thin_factor > 1:
            pipeline_stages.append({"type": "filters.decimation", "step": thin_factor})

        # Add sample filter for max_points
        if max_points is not None:
            pipeline_stages.append({"type": "filters.sample", "radius": 0})  # Random sampling

        # Create and execute pipeline
        pipeline_json = json.dumps(pipeline_stages)
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()

        # Get arrays
        arrays = pipeline.arrays
        if len(arrays) == 0:
            raise ValueError(f"No points read from {original_path}")

        arr = arrays[0]
        original_count = (
            pipeline.metadata.get("metadata", {}).get("readers.las", {}).get("count", len(arr))
        )

        # Apply max_points limit if needed
        if max_points is not None and len(arr) > max_points:
            indices = np.random.choice(len(arr), max_points, replace=False)
            arr = arr[indices]

        # Extract XYZ
        points = np.column_stack([arr["X"], arr["Y"], arr["Z"]])

        # Get classification counts
        classifications = {}
        if "Classification" in arr.dtype.names:
            unique, counts = np.unique(arr["Classification"], return_counts=True)
            classifications = dict(zip(unique.astype(int), counts.astype(int)))

        # Extract metadata from pipeline
        metadata = self._extract_pdal_metadata(
            pipeline, original_path, len(points), original_count, classifications
        )

        logger.info(f"Loaded {len(points):,} points from {Path(original_path).name}")
        return points, metadata

    def _extract_pdal_metadata(
        self,
        pipeline: "pdal.Pipeline",
        filename: str,
        point_count: int,
        original_count: int,
        classifications: Dict[int, int],
    ) -> PointCloudMetadata:
        """Extract metadata from PDAL pipeline"""
        try:
            meta = pipeline.metadata.get("metadata", {})
            reader_meta = meta.get("readers.las", {})

            bounds = (
                reader_meta.get("minx", 0),
                reader_meta.get("maxx", 0),
                reader_meta.get("miny", 0),
                reader_meta.get("maxy", 0),
                reader_meta.get("minz", 0),
                reader_meta.get("maxz", 0),
            )

            return PointCloudMetadata(
                filename=Path(filename).name,
                point_count=point_count,
                original_point_count=original_count,
                bounds=bounds,
                crs_wkt=reader_meta.get("srs", {}).get("wkt", None),
                las_version=f"{reader_meta.get('major_version', 1)}.{reader_meta.get('minor_version', 4)}",
                point_format=reader_meta.get("dataformat_id", 0),
                classifications=classifications,
                scale=(
                    reader_meta.get("scale_x", 0.001),
                    reader_meta.get("scale_y", 0.001),
                    reader_meta.get("scale_z", 0.001),
                ),
                offset=(
                    reader_meta.get("offset_x", 0),
                    reader_meta.get("offset_y", 0),
                    reader_meta.get("offset_z", 0),
                ),
            )
        except Exception as e:
            logger.warning(f"Could not extract full metadata: {e}")
            return PointCloudMetadata(
                filename=Path(filename).name,
                point_count=point_count,
                original_point_count=original_count,
                bounds=(0, 0, 0, 0, 0, 0),
                crs_wkt=None,
                las_version="1.4",
                point_format=0,
                classifications=classifications,
                scale=(0.001, 0.001, 0.001),
                offset=(0, 0, 0),
            )

    def read_with_ground_classification(
        self,
        file_path: Union[str, Path],
        algorithm: str = "smrf",
        return_all_classes: bool = False,
        **algorithm_params,
    ) -> Tuple[np.ndarray, PointCloudMetadata]:
        """
        Read LAS file and classify ground points using PDAL algorithms.

        This is useful when the input data is unclassified or poorly classified.
        PDAL provides robust ground classification algorithms:

        - **SMRF** (Simple Morphological Filter): Best for general terrain
        - **PMF** (Progressive Morphological Filter): Good for varied terrain
        - **CSF** (Cloth Simulation Filter): Fast, good for dense vegetation

        Args:
            file_path: Path to LAS/LAZ file
            algorithm: Ground classification algorithm ('smrf', 'pmf', 'csf')
            return_all_classes: If True, return all points with new classifications
                               If False (default), return only ground points
            **algorithm_params: Parameters for the classification algorithm
                SMRF params: cell (1.0), slope (0.15), scalar (1.25), threshold (0.5)
                PMF params: cell_size (1.0), max_window_size (33), slope (1.0)
                CSF params: resolution (0.5), rigidness (1), iterations (500)

        Returns:
            Tuple of (points, metadata)

        Example:
            >>> reader = LASReader()
            >>> # Classify ground using SMRF with custom parameters
            >>> points, meta = reader.read_with_ground_classification(
            ...     "unclassified.laz",
            ...     algorithm='smrf',
            ...     cell=0.5,      # Smaller cell for fine detail
            ...     slope=0.2      # More aggressive slope threshold
            ... )
        """
        if not self.use_pdal:
            raise RuntimeError(
                "Ground classification requires PDAL. "
                "Install with: conda install -c conda-forge pdal python-pdal"
            )

        if algorithm not in self.GROUND_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. Choose from: {self.GROUND_ALGORITHMS}"
            )

        file_path = str(file_path)

        # Handle GCS paths
        if file_path.startswith("gs://"):
            local_path = self._download_from_gcs(file_path)
            cleanup_after = True
        else:
            local_path = Path(file_path)
            cleanup_after = False

        try:
            logger.info(f"Classifying ground points using {algorithm.upper()}")

            # Build classification pipeline
            pipeline_stages = [str(local_path)]

            # Add noise filter first (outlier removal)
            pipeline_stages.append(
                {
                    "type": "filters.outlier",
                    "method": "statistical",
                    "mean_k": 12,
                    "multiplier": 2.2,
                }
            )

            # Add ground classification filter
            if algorithm == "smrf":
                ground_filter = {
                    "type": "filters.smrf",
                    "cell": algorithm_params.get("cell", 1.0),
                    "slope": algorithm_params.get("slope", 0.15),
                    "scalar": algorithm_params.get("scalar", 1.25),
                    "threshold": algorithm_params.get("threshold", 0.5),
                }
            elif algorithm == "pmf":
                ground_filter = {
                    "type": "filters.pmf",
                    "cell_size": algorithm_params.get("cell_size", 1.0),
                    "max_window_size": algorithm_params.get("max_window_size", 33),
                    "slope": algorithm_params.get("slope", 1.0),
                }
            elif algorithm == "csf":
                ground_filter = {
                    "type": "filters.csf",
                    "resolution": algorithm_params.get("resolution", 0.5),
                    "rigidness": algorithm_params.get("rigidness", 1),
                    "iterations": algorithm_params.get("iterations", 500),
                }

            pipeline_stages.append(ground_filter)

            # Filter to ground only if requested
            if not return_all_classes:
                pipeline_stages.append(
                    {"type": "filters.expression", "expression": "Classification == 2"}
                )

            # Execute pipeline
            pipeline_json = json.dumps(pipeline_stages)
            pipeline = pdal.Pipeline(pipeline_json)
            pipeline.execute()

            arrays = pipeline.arrays
            if len(arrays) == 0:
                raise ValueError(f"No points after classification from {file_path}")

            arr = arrays[0]

            # Extract XYZ
            points = np.column_stack([arr["X"], arr["Y"], arr["Z"]])

            # Get classification counts
            classifications = {}
            if "Classification" in arr.dtype.names:
                unique, counts = np.unique(arr["Classification"], return_counts=True)
                classifications = dict(zip(unique.astype(int), counts.astype(int)))

            metadata = self._extract_pdal_metadata(
                pipeline, file_path, len(points), len(arr), classifications
            )

            ground_count = classifications.get(2, 0)
            logger.info(f"Ground classification complete: {ground_count:,} ground points")

            return points, metadata

        finally:
            if cleanup_after and local_path.exists():
                os.unlink(local_path)

    def read_with_height_normalization(
        self, file_path: Union[str, Path], classification: Optional[Union[int, List[int]]] = None
    ) -> Tuple[np.ndarray, PointCloudMetadata]:
        """
        Read LAS file and normalize heights above ground.

        Computes Height Above Ground (HAG) for each point, useful for
        vegetation analysis and canopy height models.

        Args:
            file_path: Path to LAS/LAZ file
            classification: Filter to specific classification(s) after normalization

        Returns:
            Tuple of (points, metadata) where Z values are heights above ground
        """
        if not self.use_pdal:
            raise RuntimeError("Height normalization requires PDAL")

        file_path = str(file_path)

        if file_path.startswith("gs://"):
            local_path = self._download_from_gcs(file_path)
            cleanup_after = True
        else:
            local_path = Path(file_path)
            cleanup_after = False

        try:
            pipeline_stages = [
                str(local_path),
                {"type": "filters.hag_nn"},  # Height Above Ground using nearest neighbor
            ]

            if classification is not None:
                if isinstance(classification, int):
                    classification = [classification]
                class_expr = " || ".join([f"Classification == {c}" for c in classification])
                pipeline_stages.append({"type": "filters.expression", "expression": class_expr})

            pipeline_json = json.dumps(pipeline_stages)
            pipeline = pdal.Pipeline(pipeline_json)
            pipeline.execute()

            arr = pipeline.arrays[0]

            # Use HeightAboveGround if available, otherwise Z
            if "HeightAboveGround" in arr.dtype.names:
                z_values = arr["HeightAboveGround"]
            else:
                z_values = arr["Z"]

            points = np.column_stack([arr["X"], arr["Y"], z_values])

            classifications = {}
            if "Classification" in arr.dtype.names:
                unique, counts = np.unique(arr["Classification"], return_counts=True)
                classifications = dict(zip(unique.astype(int), counts.astype(int)))

            metadata = self._extract_pdal_metadata(
                pipeline, file_path, len(points), len(arr), classifications
            )

            return points, metadata

        finally:
            if cleanup_after and local_path.exists():
                os.unlink(local_path)

    def read_ground_points(
        self, file_path: Union[str, Path], classify_if_needed: bool = True, **kwargs
    ) -> Tuple[np.ndarray, PointCloudMetadata]:
        """
        Convenience method to read only ground-classified points.

        If classify_if_needed is True and no ground points exist in the file,
        will automatically run SMRF ground classification.

        Args:
            file_path: Path to LAS/LAZ file
            classify_if_needed: Run ground classification if no ground points exist
            **kwargs: Additional arguments passed to read_las()

        Returns:
            Tuple of (points, metadata)
        """
        # First try to read existing ground classification
        points, metadata = self.read_las(
            file_path, classification=PointClassification.GROUND, **kwargs
        )

        # Check if we got any points
        if len(points) == 0 and classify_if_needed and self.use_pdal:
            logger.warning("No ground points found, running SMRF classification...")
            return self.read_with_ground_classification(file_path, algorithm="smrf")

        return points, metadata

    def read_multiple(
        self,
        file_paths: List[Union[str, Path]],
        classification: Optional[Union[int, List[int]]] = None,
        thin_factor: int = 1,
        max_points_per_file: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[PointCloudMetadata]]:
        """
        Read multiple LAS/LAZ files and merge into single point cloud.

        Args:
            file_paths: List of file paths
            classification: Filter to specific classification(s)
            thin_factor: Keep every Nth point
            max_points_per_file: Limit points per file

        Returns:
            Tuple of:
                - points: Combined numpy array (N, 3)
                - metadatas: List of PointCloudMetadata for each file
        """
        all_points = []
        all_metadata = []

        for file_path in file_paths:
            try:
                points, metadata = self.read_las(
                    file_path,
                    classification=classification,
                    thin_factor=thin_factor,
                    max_points=max_points_per_file,
                )
                all_points.append(points)
                all_metadata.append(metadata)
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                continue

        if not all_points:
            raise ValueError("No LAS files could be read")

        combined = np.vstack(all_points)
        logger.info(f"Combined {len(file_paths)} files: {len(combined):,} total points")

        return combined, all_metadata

    def get_file_info(self, file_path: Union[str, Path]) -> PointCloudMetadata:
        """
        Get metadata without loading full point cloud.

        Args:
            file_path: Path to LAS/LAZ file

        Returns:
            PointCloudMetadata with file information
        """
        file_path = str(file_path)

        if file_path.startswith("gs://"):
            local_path = self._download_from_gcs(file_path)
            cleanup_after = True
        else:
            local_path = Path(file_path)
            cleanup_after = False

        try:
            if self.use_pdal:
                # Use PDAL info
                pipeline_json = json.dumps([str(local_path), {"type": "filters.stats"}])
                pipeline = pdal.Pipeline(pipeline_json)
                pipeline.execute()

                return self._extract_pdal_metadata(pipeline, file_path, 0, 0, {})
            else:
                # Use laspy
                return self._get_file_info_laspy(local_path, file_path)

        finally:
            if cleanup_after and local_path.exists():
                os.unlink(local_path)

    def _read_with_laspy(
        self,
        local_path: Path,
        original_path: str,
        classification: Optional[Union[int, List[int]]],
        thin_factor: int,
        bounds: Optional[Tuple[float, float, float, float]],
        max_points: Optional[int],
    ) -> Tuple[np.ndarray, PointCloudMetadata]:
        """Fallback reading using laspy"""
        logger.info(f"Reading LAS file with laspy: {original_path}")

        las = laspy.read(str(local_path))
        original_count = len(las.points)

        classifications = self._get_classification_counts_laspy(las)

        # Build point mask
        mask = np.ones(len(las.points), dtype=bool)

        if classification is not None:
            if isinstance(classification, int):
                classification = [classification]
            class_mask = np.zeros(len(las.points), dtype=bool)
            for cls in classification:
                class_mask |= las.classification == cls
            mask &= class_mask

        if bounds is not None:
            xmin, ymin, xmax, ymax = bounds
            bounds_mask = (las.x >= xmin) & (las.x <= xmax) & (las.y >= ymin) & (las.y <= ymax)
            mask &= bounds_mask

        indices = np.where(mask)[0]

        if thin_factor > 1:
            indices = indices[::thin_factor]

        if max_points is not None and len(indices) > max_points:
            np.random.seed(42)
            indices = np.random.choice(indices, max_points, replace=False)
            indices.sort()

        points = np.column_stack([las.x[indices], las.y[indices], las.z[indices]])

        metadata = self._extract_metadata_laspy(
            las, original_path, len(points), original_count, classifications
        )

        logger.info(f"Loaded {len(points):,} points from {Path(original_path).name}")
        return points, metadata

    def _get_classification_counts_laspy(self, las) -> Dict[int, int]:
        """Get count of points per classification code using laspy"""
        unique, counts = np.unique(las.classification, return_counts=True)
        return dict(zip(unique.astype(int), counts.astype(int)))

    def _extract_metadata_laspy(
        self,
        las,
        filename: str,
        point_count: int,
        original_count: int,
        classifications: Dict[int, int],
    ) -> PointCloudMetadata:
        """Extract metadata from laspy LasData"""
        header = las.header

        return PointCloudMetadata(
            filename=Path(filename).name,
            point_count=point_count,
            original_point_count=original_count,
            bounds=(
                float(header.x_min),
                float(header.x_max),
                float(header.y_min),
                float(header.y_max),
                float(header.z_min),
                float(header.z_max),
            ),
            crs_wkt=None,
            las_version=f"{header.version.major}.{header.version.minor}",
            point_format=header.point_format.id,
            classifications=classifications,
            scale=(float(header.x_scale), float(header.y_scale), float(header.z_scale)),
            offset=(float(header.x_offset), float(header.y_offset), float(header.z_offset)),
        )

    def _get_file_info_laspy(self, local_path: Path, original_path: str) -> PointCloudMetadata:
        """Get file info using laspy"""
        with laspy.open(str(local_path)) as las_file:
            header = las_file.header

            classifications = {}
            for chunk in las_file.chunk_iterator(1_000_000):
                for cls in np.unique(chunk.classification):
                    count = np.sum(chunk.classification == cls)
                    classifications[int(cls)] = classifications.get(int(cls), 0) + count

            return PointCloudMetadata(
                filename=Path(original_path).name,
                point_count=header.point_count,
                original_point_count=header.point_count,
                bounds=(
                    header.x_min,
                    header.x_max,
                    header.y_min,
                    header.y_max,
                    header.z_min,
                    header.z_max,
                ),
                crs_wkt=None,
                las_version=f"{header.version.major}.{header.version.minor}",
                point_format=header.point_format.id,
                classifications=classifications,
                scale=(header.x_scale, header.y_scale, header.z_scale),
                offset=(header.x_offset, header.y_offset, header.z_offset),
            )

    def _download_from_gcs(self, gcs_path: str) -> Path:
        """Download file from GCS to temporary location"""
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage required for GCS paths")

        parts = gcs_path.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        blob_path = parts[1] if len(parts) > 1 else ""

        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        suffix = Path(blob_path).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            temp_path = Path(f.name)

        logger.info(f"Downloading from GCS: {gcs_path}")
        blob.download_to_filename(str(temp_path))

        return temp_path


def read_las(
    file_path: Union[str, Path], classification: Optional[Union[int, List[int]]] = None, **kwargs
) -> Tuple[np.ndarray, PointCloudMetadata]:
    """
    Convenience function to read a LAS/LAZ file.

    Args:
        file_path: Path to LAS/LAZ file
        classification: Filter to specific classification(s)
        **kwargs: Additional arguments passed to LASReader.read_las()

    Returns:
        Tuple of (points, metadata)

    Example:
        >>> points, meta = read_las("terrain.laz", classification=2)
        >>> print(f"Loaded {len(points)} ground points")
    """
    reader = LASReader()
    return reader.read_las(file_path, classification=classification, **kwargs)


def read_ground_points(
    file_path: Union[str, Path], classify_if_needed: bool = True, **kwargs
) -> Tuple[np.ndarray, PointCloudMetadata]:
    """
    Convenience function to read ground-classified points from LAS/LAZ file.

    If no ground points exist in the file and PDAL is available,
    will automatically run SMRF ground classification.

    Args:
        file_path: Path to LAS/LAZ file
        classify_if_needed: Run ground classification if no ground points exist
        **kwargs: Additional arguments passed to LASReader.read_las()

    Returns:
        Tuple of (points, metadata)

    Example:
        >>> points, meta = read_ground_points("terrain.laz")
        >>> from earthwork_llm.surface import DEMGenerator
        >>> dem, dem_meta = DEMGenerator().generate_dem(points)
    """
    reader = LASReader()
    return reader.read_ground_points(file_path, classify_if_needed=classify_if_needed, **kwargs)


def classify_ground(
    file_path: Union[str, Path], algorithm: str = "smrf", **kwargs
) -> Tuple[np.ndarray, PointCloudMetadata]:
    """
    Classify ground points using PDAL algorithms.

    Args:
        file_path: Path to LAS/LAZ file
        algorithm: Ground classification algorithm ('smrf', 'pmf', 'csf')
        **kwargs: Algorithm-specific parameters

    Returns:
        Tuple of (ground_points, metadata)

    Example:
        >>> # Use SMRF with custom parameters
        >>> points, meta = classify_ground("raw_lidar.laz", algorithm='smrf', cell=0.5)

        >>> # Use CSF for fast classification
        >>> points, meta = classify_ground("raw_lidar.laz", algorithm='csf')
    """
    reader = LASReader()
    return reader.read_with_ground_classification(file_path, algorithm=algorithm, **kwargs)
