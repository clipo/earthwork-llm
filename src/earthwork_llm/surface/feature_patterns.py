"""
Feature Pattern Recognizer for TerraLLM

Identifies specific terrain and military features using multi-scale geomorphon
patterns. Assembles geomorphon signatures across scales to detect foxholes,
trenches, ridges, defensive positions, and other tactically significant features.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .geomorphons import GeomorphonType

logger = logging.getLogger(__name__)


class MilitaryFeature(Enum):
    """
    Military and tactical terrain features detectable from LiDAR.

    Features are identified by their multi-scale geomorphon signatures
    combined with geometric constraints (size, shape, orientation).
    """

    FOXHOLE = "foxhole"
    TRENCH = "trench"
    SHELL_CRATER = "shell_crater"
    BUNKER = "bunker"
    FIGHTING_POSITION = "fighting_position"
    DEFENSIVE_LINE = "defensive_line"
    COMMUNICATION_TRENCH = "communication_trench"
    RIDGE_POSITION = "ridge_position"
    DEFILADE_POSITION = "defilade_position"
    STREAM_CROSSING = "stream_crossing"


@dataclass
class FeatureSignature:
    """
    Multi-scale geomorphon pattern signature for a feature.

    Defines what geomorphon types should appear at each scale,
    along with spatial constraints.
    """

    feature_type: MilitaryFeature

    # Geomorphon types expected at each scale
    micro_types: Set[GeomorphonType]  # 2m scale
    meso_types: Set[GeomorphonType]  # 5m scale
    local_types: Set[GeomorphonType]  # 10m scale
    regional_types: Optional[Set[GeomorphonType]] = None  # 25m scale

    # Geometric constraints
    min_diameter: float = 0.0  # meters
    max_diameter: float = float("inf")  # meters
    min_length: Optional[float] = None  # for linear features
    max_length: Optional[float] = None
    min_depth: Optional[float] = None  # elevation change
    max_depth: Optional[float] = None

    # Shape constraints
    circularity_min: float = 0.0  # 0-1, 1 = perfect circle
    linearity_min: float = 0.0  # 0-1, 1 = perfect line

    # Tactical context (boosts confidence if true)
    prefer_high_ground: bool = False
    prefer_cover: bool = False
    prefer_defilade: bool = False

    description: str = ""


@dataclass
class DetectedFeature:
    """A detected feature with location and confidence"""

    feature_type: MilitaryFeature
    center: Tuple[int, int]  # (row, col) in DEM
    bounds: Tuple[int, int, int, int]  # (row_min, row_max, col_min, col_max)
    confidence: float  # 0-1
    properties: Dict  # Additional properties (size, orientation, etc.)


class FeaturePatternRecognizer:
    """
    Recognizes terrain and military features from multi-scale geomorphon patterns.

    Uses pattern matching combined with geometric and tactical constraints
    to identify features of interest. Operates on geomorphon stacks from
    multiple scales (2m, 5m, 10m, 25m).

    Example:
        >>> recognizer = FeaturePatternRecognizer()
        >>> features = recognizer.detect_features(geomorphons, dem)
        >>> foxholes = [f for f in features if f.feature_type == MilitaryFeature.FOXHOLE]
    """

    def __init__(self):
        """Initialize feature recognizer with predefined signatures"""
        self.signatures = self._define_signatures()
        logger.info(f"Initialized with {len(self.signatures)} feature signatures")

    def _define_signatures(self) -> Dict[MilitaryFeature, FeatureSignature]:
        """Define multi-scale geomorphon signatures for each feature type"""

        signatures = {}

        # FOXHOLE: Small circular depression with raised rim
        signatures[MilitaryFeature.FOXHOLE] = FeatureSignature(
            feature_type=MilitaryFeature.FOXHOLE,
            micro_types={GeomorphonType.PIT, GeomorphonType.HOLLOW},
            meso_types={GeomorphonType.PIT, GeomorphonType.HOLLOW, GeomorphonType.SHOULDER},
            local_types={GeomorphonType.FLAT, GeomorphonType.SLOPE},
            min_diameter=1.0,
            max_diameter=2.5,
            min_depth=0.8,
            max_depth=1.8,
            circularity_min=0.6,
            prefer_high_ground=True,
            prefer_cover=True,
            description="Circular depression 1-2.5m diameter with raised rim, often on tactical high ground",
        )

        # TRENCH: Linear depression, often zigzag pattern
        signatures[MilitaryFeature.TRENCH] = FeatureSignature(
            feature_type=MilitaryFeature.TRENCH,
            micro_types={GeomorphonType.VALLEY, GeomorphonType.HOLLOW},
            meso_types={GeomorphonType.VALLEY, GeomorphonType.HOLLOW},
            local_types={GeomorphonType.VALLEY, GeomorphonType.SLOPE},
            min_diameter=0.8,
            max_diameter=2.0,
            min_length=5.0,
            max_length=100.0,
            min_depth=1.0,
            max_depth=2.5,
            linearity_min=0.4,
            prefer_high_ground=True,
            prefer_defilade=True,
            description="Linear depression 0.8-2.0m wide, 5-100m long, often zigzag defensive pattern",
        )

        # SHELL CRATER: Larger circular depression with pronounced rim
        signatures[MilitaryFeature.SHELL_CRATER] = FeatureSignature(
            feature_type=MilitaryFeature.SHELL_CRATER,
            micro_types={GeomorphonType.PIT},
            meso_types={GeomorphonType.PIT, GeomorphonType.SHOULDER},
            local_types={GeomorphonType.PIT, GeomorphonType.SHOULDER},
            min_diameter=2.0,
            max_diameter=15.0,
            min_depth=0.5,
            max_depth=5.0,
            circularity_min=0.7,
            description="Circular depression 2-15m diameter with raised rim from explosive impact",
        )

        # BUNKER: Elevated structure with surrounding earthworks
        signatures[MilitaryFeature.BUNKER] = FeatureSignature(
            feature_type=MilitaryFeature.BUNKER,
            micro_types={GeomorphonType.PEAK, GeomorphonType.SHOULDER},
            meso_types={GeomorphonType.PEAK, GeomorphonType.SHOULDER, GeomorphonType.RIDGE},
            local_types={GeomorphonType.SHOULDER, GeomorphonType.SLOPE},
            min_diameter=3.0,
            max_diameter=10.0,
            prefer_high_ground=True,
            prefer_cover=True,
            description="Elevated structure 3-10m diameter with earthwork berms",
        )

        # FIGHTING POSITION: Small depression, may be part of larger system
        signatures[MilitaryFeature.FIGHTING_POSITION] = FeatureSignature(
            feature_type=MilitaryFeature.FIGHTING_POSITION,
            micro_types={GeomorphonType.PIT, GeomorphonType.HOLLOW, GeomorphonType.VALLEY},
            meso_types={GeomorphonType.HOLLOW, GeomorphonType.SLOPE},
            local_types={GeomorphonType.FLAT, GeomorphonType.SLOPE},
            min_diameter=1.0,
            max_diameter=3.0,
            min_depth=0.5,
            max_depth=1.5,
            prefer_high_ground=True,
            prefer_cover=True,
            description="Small fighting position 1-3m, often on high ground with cover",
        )

        # RIDGE POSITION: Tactical high ground (natural feature)
        signatures[MilitaryFeature.RIDGE_POSITION] = FeatureSignature(
            feature_type=MilitaryFeature.RIDGE_POSITION,
            micro_types={GeomorphonType.RIDGE, GeomorphonType.SHOULDER},
            meso_types={GeomorphonType.RIDGE, GeomorphonType.SPUR},
            local_types={GeomorphonType.RIDGE, GeomorphonType.SHOULDER},
            regional_types={GeomorphonType.RIDGE},
            min_length=10.0,
            linearity_min=0.5,
            prefer_high_ground=True,
            description="Linear elevated terrain providing observation and fields of fire",
        )

        # DEFILADE POSITION: Protected position (hull-down, turret-down)
        signatures[MilitaryFeature.DEFILADE_POSITION] = FeatureSignature(
            feature_type=MilitaryFeature.DEFILADE_POSITION,
            micro_types={GeomorphonType.HOLLOW, GeomorphonType.FOOTSLOPE},
            meso_types={GeomorphonType.HOLLOW, GeomorphonType.FOOTSLOPE},
            local_types={GeomorphonType.FOOTSLOPE, GeomorphonType.SLOPE},
            regional_types={GeomorphonType.RIDGE, GeomorphonType.SHOULDER},
            prefer_defilade=True,
            description="Protected position behind crest or terrain feature",
        )

        # DEFENSIVE LINE: Series of connected positions
        # (This is composite - detected by clustering individual features)
        signatures[MilitaryFeature.DEFENSIVE_LINE] = FeatureSignature(
            feature_type=MilitaryFeature.DEFENSIVE_LINE,
            micro_types={GeomorphonType.VALLEY, GeomorphonType.HOLLOW, GeomorphonType.PIT},
            meso_types={GeomorphonType.VALLEY, GeomorphonType.HOLLOW},
            local_types={GeomorphonType.VALLEY, GeomorphonType.RIDGE},
            min_length=20.0,
            linearity_min=0.3,
            description="Linear defensive system of connected positions",
        )

        # STREAM CROSSING: Valley with linear feature (tactical chokepoint)
        signatures[MilitaryFeature.STREAM_CROSSING] = FeatureSignature(
            feature_type=MilitaryFeature.STREAM_CROSSING,
            micro_types={GeomorphonType.VALLEY},
            meso_types={GeomorphonType.VALLEY},
            local_types={GeomorphonType.VALLEY, GeomorphonType.FOOTSLOPE},
            regional_types={GeomorphonType.VALLEY},
            min_length=10.0,
            linearity_min=0.6,
            description="Stream valley providing water source or obstacle",
        )

        return signatures

    def detect_features(
        self,
        geomorphons: Dict[str, np.ndarray],
        dem: np.ndarray,
        cell_size: float = 0.5,
        confidence_threshold: float = 0.5,
        feature_types: Optional[List[MilitaryFeature]] = None,
    ) -> List[DetectedFeature]:
        """
        Detect features in multi-scale geomorphon data.

        Args:
            geomorphons: Dict of geomorphon arrays from GeomorphonAnalyzer
            dem: Digital Elevation Model for geometric analysis
            cell_size: DEM resolution in meters
            confidence_threshold: Minimum confidence to return (0-1)
            feature_types: Optional list to limit detection to specific types

        Returns:
            List of detected features with locations and confidence scores
        """
        if feature_types is None:
            feature_types = list(MilitaryFeature)

        logger.info(f"Detecting {len(feature_types)} feature types...")

        all_features = []

        for feature_type in feature_types:
            if feature_type not in self.signatures:
                logger.warning(f"No signature defined for {feature_type}")
                continue

            signature = self.signatures[feature_type]
            logger.debug(f"Searching for {feature_type.value}...")

            features = self._detect_feature_type(
                signature, geomorphons, dem, cell_size, confidence_threshold
            )

            logger.info(f"Found {len(features)} {feature_type.value} candidates")
            all_features.extend(features)

        # Post-processing: remove duplicates, merge overlapping
        all_features = self._post_process_detections(all_features)

        logger.info(f"Total features detected: {len(all_features)}")

        return all_features

    def _detect_feature_type(
        self,
        signature: FeatureSignature,
        geomorphons: Dict[str, np.ndarray],
        dem: np.ndarray,
        cell_size: float,
        threshold: float,
    ) -> List[DetectedFeature]:
        """Detect all instances of a specific feature type"""

        features = []

        # Get geomorphon arrays for each scale
        micro = geomorphons.get("2m")
        meso = geomorphons.get("5m")
        local = geomorphons.get("10m")

        if micro is None or meso is None:
            logger.warning("Missing required geomorphon scales (2m, 5m)")
            return features

        height, width = micro.shape

        # Scan for matching patterns
        for i in range(height):
            for j in range(width):
                # Check if micro-scale matches
                if micro[i, j] not in signature.micro_types:
                    continue

                # Check meso-scale
                if meso[i, j] not in signature.meso_types:
                    continue

                # Check local-scale if available
                if local is not None and signature.local_types:
                    if local[i, j] not in signature.local_types:
                        continue

                # Potential match - compute detailed confidence
                confidence = self._compute_confidence(signature, i, j, geomorphons, dem, cell_size)

                if confidence >= threshold:
                    # Extract feature properties
                    properties = self._extract_properties(i, j, geomorphons, dem, cell_size)

                    # Check geometric constraints
                    if not self._check_constraints(signature, properties):
                        continue

                    # Create detected feature
                    feature = DetectedFeature(
                        feature_type=signature.feature_type,
                        center=(i, j),
                        bounds=self._compute_bounds(i, j, properties, height, width),
                        confidence=confidence,
                        properties=properties,
                    )

                    features.append(feature)

        return features

    def _compute_confidence(
        self,
        signature: FeatureSignature,
        row: int,
        col: int,
        geomorphons: Dict[str, np.ndarray],
        dem: np.ndarray,
        cell_size: float,
    ) -> float:
        """
        Compute confidence score for a potential feature.

        Considers:
        - Multi-scale pattern consistency
        - Geometric properties (size, shape)
        - Tactical context (if applicable)
        """
        confidence = 0.5  # Base confidence for pattern match

        # Multi-scale consistency bonus
        scales_matched = 2  # Already matched micro and meso
        if "10m" in geomorphons and signature.local_types:
            if geomorphons["10m"][row, col] in signature.local_types:
                scales_matched += 1
        if "25m" in geomorphons and signature.regional_types:
            if geomorphons["25m"][row, col] in signature.regional_types:
                scales_matched += 1

        confidence += 0.1 * scales_matched

        # Extract local properties
        props = self._extract_properties(row, col, geomorphons, dem, cell_size)

        # Geometric constraints bonus
        if signature.circularity_min > 0 and "circularity" in props:
            if props["circularity"] >= signature.circularity_min:
                confidence += 0.15

        if signature.linearity_min > 0 and "linearity" in props:
            if props["linearity"] >= signature.linearity_min:
                confidence += 0.15

        # Tactical context bonus
        if signature.prefer_high_ground:
            if self._is_high_ground(row, col, dem):
                confidence += 0.1

        if signature.prefer_defilade:
            if self._has_defilade(row, col, dem):
                confidence += 0.1

        return min(confidence, 1.0)

    def _extract_properties(
        self,
        row: int,
        col: int,
        geomorphons: Dict[str, np.ndarray],
        dem: np.ndarray,
        cell_size: float,
    ) -> Dict:
        """Extract geometric and morphological properties around a point"""

        # Simple local analysis (can be enhanced)
        window_size = 5
        row_min = max(0, row - window_size)
        row_max = min(dem.shape[0], row + window_size + 1)
        col_min = max(0, col - window_size)
        col_max = min(dem.shape[1], col + window_size + 1)

        local_dem = dem[row_min:row_max, col_min:col_max]

        properties = {
            "elevation": dem[row, col],
            "local_relief": local_dem.max() - local_dem.min(),
            "mean_elevation": local_dem.mean(),
            "std_elevation": local_dem.std(),
        }

        return properties

    def _check_constraints(self, signature: FeatureSignature, properties: Dict) -> bool:
        """Check if properties satisfy signature constraints"""

        # Size constraints
        if "diameter" in properties:
            if properties["diameter"] < signature.min_diameter:
                return False
            if properties["diameter"] > signature.max_diameter:
                return False

        # Depth constraints
        if signature.min_depth and "local_relief" in properties:
            if properties["local_relief"] < signature.min_depth:
                return False

        return True

    def _compute_bounds(
        self, row: int, col: int, properties: Dict, height: int, width: int
    ) -> Tuple[int, int, int, int]:
        """Compute bounding box for feature"""

        # Simple fixed-size bounds (can be enhanced with actual feature extent)
        radius = 5

        row_min = max(0, row - radius)
        row_max = min(height, row + radius)
        col_min = max(0, col - radius)
        col_max = min(width, col + radius)

        return (row_min, row_max, col_min, col_max)

    def _is_high_ground(self, row: int, col: int, dem: np.ndarray) -> bool:
        """Check if location is on high ground"""

        # Simple check: higher than local average
        window = 10
        row_min = max(0, row - window)
        row_max = min(dem.shape[0], row + window)
        col_min = max(0, col - window)
        col_max = min(dem.shape[1], col + window)

        local_mean = dem[row_min:row_max, col_min:col_max].mean()
        return dem[row, col] > local_mean + 1.0  # 1m above local average

    def _has_defilade(self, row: int, col: int, dem: np.ndarray) -> bool:
        """Check if location has defilade (protected from observation)"""

        # Simple check: behind a crest or rise
        # Look in 8 directions for higher terrain nearby
        directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

        center_z = dem[row, col]
        has_cover = False

        for dx, dy in directions:
            for dist in range(1, 10):
                i = row + dy * dist
                j = col + dx * dist

                if i < 0 or i >= dem.shape[0] or j < 0 or j >= dem.shape[1]:
                    break

                if dem[i, j] > center_z + 2.0:  # 2m higher nearby
                    has_cover = True
                    break

        return has_cover

    def _post_process_detections(self, features: List[DetectedFeature]) -> List[DetectedFeature]:
        """
        Post-process detected features.

        - Remove duplicates (same location, different types)
        - Merge overlapping features of same type
        - Filter false positives
        """

        if not features:
            return features

        # Simple duplicate removal based on proximity
        # Keep highest confidence feature at each location

        unique_features = []
        processed = set()

        for i, feat in enumerate(sorted(features, key=lambda f: f.confidence, reverse=True)):
            if i in processed:
                continue

            # Check for nearby features
            keep = True
            for j, other in enumerate(features):
                if i == j or j in processed:
                    continue

                # Calculate distance
                dist = np.sqrt(
                    (feat.center[0] - other.center[0]) ** 2
                    + (feat.center[1] - other.center[1]) ** 2
                )

                # If very close, mark lower confidence one as processed
                if dist < 5:  # 5 cells
                    processed.add(j)

            if keep:
                unique_features.append(feat)
                processed.add(i)

        logger.debug(f"Post-processing: {len(features)} -> {len(unique_features)} features")

        return unique_features

    def detect_features_fast(
        self,
        geomorphons: Dict[str, np.ndarray],
        dem: np.ndarray,
        cell_size: float = 0.5,
        confidence_threshold: float = 0.5,
        feature_types: Optional[List[MilitaryFeature]] = None,
        min_component_cells: int = 1,
    ) -> List[DetectedFeature]:
        """
        Fast feature detection using connected component analysis.

        Instead of scanning every pixel, this method:
        1. Creates a mask of candidate pixels (matching geomorphon patterns)
        2. Labels connected components
        3. Computes morphometric properties for each component
        4. Filters and scores based on signatures

        This is 10-20x faster than detect_features() for large DEMs.

        Args:
            geomorphons: Dict of geomorphon arrays from GeomorphonAnalyzer
            dem: Digital Elevation Model for geometric analysis
            cell_size: DEM resolution in meters
            confidence_threshold: Minimum confidence to return (0-1)
            feature_types: Optional list to limit detection to specific types
            min_component_cells: Minimum cells in a component to consider

        Returns:
            List of detected features with locations and confidence scores
        """
        from scipy import ndimage as ndi

        if feature_types is None:
            feature_types = list(MilitaryFeature)

        logger.info(f"Fast detection of {len(feature_types)} feature types...")

        all_features = []

        for feature_type in feature_types:
            if feature_type not in self.signatures:
                continue

            signature = self.signatures[feature_type]

            # Create candidate mask from multi-scale patterns
            candidate_mask = self._create_candidate_mask(signature, geomorphons)

            if candidate_mask.sum() == 0:
                continue

            # Label connected components
            labeled, n_components = ndi.label(candidate_mask)

            if n_components == 0:
                continue

            logger.debug(f"Found {n_components} {feature_type.value} candidate regions")

            # Process each component
            for comp_id in range(1, n_components + 1):
                component = labeled == comp_id
                n_cells = component.sum()

                if n_cells < min_component_cells:
                    continue

                # Compute morphometric properties
                props = compute_morphometric_properties(component, dem, cell_size)

                # Check if component matches signature constraints
                if not self._check_morphometric_constraints(signature, props, cell_size):
                    continue

                # Compute confidence
                confidence = self._compute_component_confidence(
                    signature, component, geomorphons, dem, props
                )

                if confidence >= confidence_threshold:
                    feature = DetectedFeature(
                        feature_type=signature.feature_type,
                        center=(props["centroid_row"], props["centroid_col"]),
                        bounds=(
                            props["bbox_row_min"],
                            props["bbox_row_max"],
                            props["bbox_col_min"],
                            props["bbox_col_max"],
                        ),
                        confidence=confidence,
                        properties=props,
                    )
                    all_features.append(feature)

            logger.info(
                f"Found {len([f for f in all_features if f.feature_type == feature_type])} {feature_type.value} features"
            )

        logger.info(f"Total features detected: {len(all_features)}")
        return all_features

    def _create_candidate_mask(
        self, signature: FeatureSignature, geomorphons: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Create a boolean mask of candidate pixels matching pattern."""

        micro = geomorphons.get("2m")
        meso = geomorphons.get("5m")

        if micro is None or meso is None:
            return np.zeros((1, 1), dtype=bool)

        # Convert signature types to int values for numpy
        micro_values = [int(t) for t in signature.micro_types]
        meso_values = [int(t) for t in signature.meso_types]

        # Create mask: must match at both scales
        mask = np.isin(micro, micro_values) & np.isin(meso, meso_values)

        # Optionally check local scale
        if signature.local_types and "10m" in geomorphons:
            local = geomorphons["10m"]
            local_values = [int(t) for t in signature.local_types]
            mask = mask & np.isin(local, local_values)

        return mask

    def _check_morphometric_constraints(
        self, signature: FeatureSignature, props: Dict, cell_size: float
    ) -> bool:
        """Check if morphometric properties satisfy signature constraints."""

        # Size constraints (diameter in meters)
        diameter = props.get("equivalent_diameter", 0) * cell_size

        if diameter < signature.min_diameter:
            return False
        if diameter > signature.max_diameter:
            return False

        # Length constraints (for linear features)
        if signature.min_length is not None:
            length = props.get("major_axis_length", 0) * cell_size
            if length < signature.min_length:
                return False

        if signature.max_length is not None:
            length = props.get("major_axis_length", 0) * cell_size
            if length > signature.max_length:
                return False

        # Circularity constraint
        if signature.circularity_min > 0:
            circularity = props.get("circularity", 0)
            if circularity < signature.circularity_min:
                return False

        # Linearity constraint
        if signature.linearity_min > 0:
            linearity = props.get("elongation", 0)
            if linearity < signature.linearity_min:
                return False

        # Depth constraint
        if signature.min_depth is not None:
            depth = props.get("depth", 0)
            if depth < signature.min_depth:
                return False

        return True

    def _compute_component_confidence(
        self,
        signature: FeatureSignature,
        component: np.ndarray,
        geomorphons: Dict[str, np.ndarray],
        dem: np.ndarray,
        props: Dict,
    ) -> float:
        """Compute confidence score for a connected component."""

        confidence = 0.5  # Base for pattern match

        # Multi-scale consistency bonus
        scales_matched = 2  # Already matched micro + meso

        rows, cols = np.where(component)
        center_row = int(np.mean(rows))
        center_col = int(np.mean(cols))

        if "10m" in geomorphons and signature.local_types:
            if geomorphons["10m"][center_row, center_col] in signature.local_types:
                scales_matched += 1

        if "25m" in geomorphons and signature.regional_types:
            if geomorphons["25m"][center_row, center_col] in signature.regional_types:
                scales_matched += 1

        confidence += 0.1 * scales_matched

        # Morphometric bonus
        if signature.circularity_min > 0:
            circ = props.get("circularity", 0)
            if circ >= signature.circularity_min:
                confidence += 0.15

        if signature.linearity_min > 0:
            elong = props.get("elongation", 0)
            if elong >= signature.linearity_min:
                confidence += 0.15

        # Tactical context bonus
        if signature.prefer_high_ground:
            if self._is_high_ground(center_row, center_col, dem):
                confidence += 0.1

        if signature.prefer_defilade:
            if self._has_defilade(center_row, center_col, dem):
                confidence += 0.1

        return min(confidence, 1.0)


# =============================================================================
# MORPHOMETRIC PROPERTY COMPUTATION
# =============================================================================


def compute_morphometric_properties(
    component: np.ndarray, dem: np.ndarray, cell_size: float = 1.0
) -> Dict:
    """
    Compute morphometric properties for a connected component.

    This provides shape descriptors critical for distinguishing:
    - Foxholes (circular, high circularity)
    - Trenches (linear, high elongation)
    - Craters (circular, larger diameter)

    Args:
        component: Boolean mask of the connected component
        dem: Digital Elevation Model
        cell_size: DEM resolution in meters

    Returns:
        Dict with morphometric properties:
        - centroid_row, centroid_col: Center of mass
        - area: Number of cells
        - equivalent_diameter: Diameter of circle with same area (in cells)
        - major_axis_length, minor_axis_length: PCA-based axes (in cells)
        - orientation: Angle of major axis (radians from horizontal)
        - circularity: 4π × area / perimeter² (1.0 = perfect circle)
        - elongation: major_axis / minor_axis (1.0 = circular)
        - depth: Elevation difference (center vs surrounding)
        - bbox_*: Bounding box coordinates
    """

    # Find component pixels
    rows, cols = np.where(component)

    if len(rows) == 0:
        return {}

    # Basic properties
    n_cells = len(rows)
    centroid_row = int(np.mean(rows))
    centroid_col = int(np.mean(cols))

    # Bounding box
    bbox_row_min = int(rows.min())
    bbox_row_max = int(rows.max())
    bbox_col_min = int(cols.min())
    bbox_col_max = int(cols.max())

    # Equivalent diameter (diameter of circle with same area)
    equivalent_diameter = np.sqrt(4 * n_cells / np.pi)

    # Compute perimeter (boundary length)
    # Use erosion to find boundary pixels
    from scipy.ndimage import binary_erosion

    eroded = binary_erosion(component)
    boundary = component & ~eroded
    perimeter = boundary.sum()

    # Circularity: 4π × area / perimeter²
    # 1.0 for perfect circle, lower for irregular shapes
    if perimeter > 0:
        circularity = 4 * np.pi * n_cells / (perimeter**2)
    else:
        circularity = 1.0

    # Clamp circularity to [0, 1] (can exceed 1 for small shapes)
    circularity = min(circularity, 1.0)

    # PCA for orientation and elongation
    if n_cells >= 3:
        # Center the coordinates
        coords = np.column_stack([rows - centroid_row, cols - centroid_col])

        # Compute covariance matrix
        cov = np.cov(coords.T)

        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Major/minor axis lengths (2 std deviations)
        major_axis_length = 4 * np.sqrt(max(eigenvalues[0], 0))
        minor_axis_length = 4 * np.sqrt(max(eigenvalues[1], 0))

        # Orientation (angle of major axis from horizontal)
        orientation = np.arctan2(eigenvectors[0, 0], eigenvectors[1, 0])

        # Elongation (aspect ratio)
        if minor_axis_length > 0:
            elongation = major_axis_length / minor_axis_length
        else:
            elongation = 1.0
    else:
        major_axis_length = equivalent_diameter
        minor_axis_length = equivalent_diameter
        orientation = 0.0
        elongation = 1.0

    # Depth: difference between center and surrounding elevation
    center_elevation = dem[centroid_row, centroid_col]
    component_elevations = dem[component]
    surrounding_elevation = np.nanmean(dem[boundary]) if boundary.any() else center_elevation
    depth = abs(surrounding_elevation - np.nanmean(component_elevations))

    # Local relief within component
    local_relief = np.nanmax(component_elevations) - np.nanmin(component_elevations)

    return {
        "centroid_row": centroid_row,
        "centroid_col": centroid_col,
        "area": n_cells,
        "area_m2": n_cells * cell_size * cell_size,
        "equivalent_diameter": equivalent_diameter,
        "equivalent_diameter_m": equivalent_diameter * cell_size,
        "major_axis_length": major_axis_length,
        "minor_axis_length": minor_axis_length,
        "major_axis_length_m": major_axis_length * cell_size,
        "minor_axis_length_m": minor_axis_length * cell_size,
        "orientation": orientation,
        "orientation_deg": np.degrees(orientation),
        "circularity": circularity,
        "elongation": elongation,
        "perimeter": perimeter,
        "perimeter_m": perimeter * cell_size,
        "depth": depth,
        "local_relief": local_relief,
        "bbox_row_min": bbox_row_min,
        "bbox_row_max": bbox_row_max,
        "bbox_col_min": bbox_col_min,
        "bbox_col_max": bbox_col_max,
        "center_elevation": center_elevation,
    }


def classify_feature_shape(props: Dict) -> str:
    """
    Classify feature shape based on morphometric properties.

    Returns:
        Shape classification: 'circular', 'elongated', 'irregular', 'linear'
    """
    circularity = props.get("circularity", 0)
    elongation = props.get("elongation", 1)

    if circularity > 0.7:
        return "circular"
    elif elongation > 3.0:
        return "linear"
    elif elongation > 1.5:
        return "elongated"
    else:
        return "irregular"


def filter_by_shape(
    features: List[DetectedFeature], shape: str, tolerance: float = 0.1
) -> List[DetectedFeature]:
    """
    Filter detected features by shape classification.

    Args:
        features: List of DetectedFeature objects
        shape: Target shape ('circular', 'elongated', 'irregular', 'linear')
        tolerance: Flexibility in circularity/elongation thresholds

    Returns:
        Filtered list of features matching the shape
    """
    filtered = []

    for feat in features:
        props = feat.properties
        feat_shape = classify_feature_shape(props)

        if feat_shape == shape:
            filtered.append(feat)

    return filtered
