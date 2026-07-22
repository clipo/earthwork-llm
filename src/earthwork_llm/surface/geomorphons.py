"""
Geomorphon Analyzer for TerraLLM

Multi-scale morphological terrain classification using geomorphon patterns.
Geomorphons classify terrain into 10 fundamental landform types based on
line-of-sight patterns in 8 directions.

This module provides two approaches:
1. **Direct elevation approach** (original): Computes zenith/nadir angles directly
2. **Openness approach** (Pingel): Uses terrain openness for more robust detection

The openness approach is particularly effective for detecting subtle linear
features like trails, which appear as shallow valleys in LiDAR data.

References:
- Jasiewicz, J., & Stepinski, T. F. (2013). Geomorphons—a pattern
  recognition approach to classification and mapping of landforms.
  Geomorphology, 182, 147-156.
- Yokoyama, R., Shirasawa, M., & Pike, R. J. (2002). Visualizing topography
  by openness: a new application of image processing to digital elevation
  models. Photogrammetric engineering and remote sensing, 68(3), 257-266.
- Pingel, T. neilpy: https://github.com/thomaspingel/neilpy
"""

import logging
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Lock for thread-safe lazy initialization of lookup tables
_TABLE_LOCK = threading.Lock()

# =============================================================================
# PRE-COMPUTED LOOKUP TABLES FOR FAST TERNARY OPERATIONS
# =============================================================================
# These are computed once at module import for O(1) encode/decode operations
# instead of O(8) loop-based computation. Provides ~10x speedup.

# Powers of 3 for encoding: [1, 3, 9, 27, 81, 243, 729, 2187]
_POWERS_OF_3 = np.array([3**i for i in range(8)], dtype=np.int32)

# Lookup table for decoding: ternary code -> 8-element pattern array
# Shape: (6561, 8) - covers all possible codes 0-6560
_DECODE_TABLE = np.zeros((6561, 8), dtype=np.int8)
for _code in range(6561):
    _temp = _code
    for _i in range(8):
        _DECODE_TABLE[_code, _i] = _temp % 3
        _temp //= 3

# Lookup table for geomorphon classification: code -> geomorphon type
# This will be populated lazily on first use
_GEOMORPHON_TABLE: Optional[np.ndarray] = None


def _build_geomorphon_table() -> np.ndarray:
    """Build lookup table mapping ternary codes to geomorphon types.

    This function is thread-safe and uses double-checked locking to ensure
    the table is only built once even in multi-threaded environments.

    Returns:
        numpy array mapping ternary codes (0-6560) to GeomorphonType values
    """
    global _GEOMORPHON_TABLE

    # Fast path: table already built
    if _GEOMORPHON_TABLE is not None:
        return _GEOMORPHON_TABLE

    # Slow path: acquire lock and build table
    with _TABLE_LOCK:
        # Double-check after acquiring lock
        if _GEOMORPHON_TABLE is not None:
            return _GEOMORPHON_TABLE

        table = np.zeros(6561, dtype=np.int8)
        for code in range(6561):
            pattern = _DECODE_TABLE[code]
            n_up = np.sum(pattern == 2)
            n_flat = np.sum(pattern == 1)
            n_down = np.sum(pattern == 0)

            # Classification logic from terrain_code_to_geomorphon
            if n_flat == 8:
                table[code] = GeomorphonType.FLAT
            elif n_down == 8:
                table[code] = GeomorphonType.PEAK
            elif n_up == 8:
                table[code] = GeomorphonType.PIT
            elif n_down >= 6 and n_up == 0:
                table[code] = GeomorphonType.RIDGE
            elif n_up >= 6 and n_down == 0:
                table[code] = GeomorphonType.VALLEY
            elif n_down >= 4 and n_up <= 1:
                table[code] = GeomorphonType.SHOULDER
            elif n_up >= 4 and n_down <= 1:
                table[code] = GeomorphonType.FOOTSLOPE
            elif n_down >= 3 and n_up <= 2:
                table[code] = GeomorphonType.SPUR
            elif n_up >= 3 and n_down <= 2:
                table[code] = GeomorphonType.HOLLOW
            else:
                table[code] = GeomorphonType.SLOPE

        _GEOMORPHON_TABLE = table
        return table


class GeomorphonType(IntEnum):
    """
    Ten fundamental geomorphon types based on ternary line-of-sight patterns.

    Each type represents a distinct morphological terrain configuration:
    - FLAT (1): Minimal relief, level terrain
    - PEAK (2): Elevated point, surrounded by lower terrain
    - RIDGE (3): Linear high, elongated crest
    - SHOULDER (4): Convex break in slope
    - SPUR (5): Projecting ridge from main high area
    - SLOPE (6): Planar inclined surface
    - HOLLOW (7): Concave depression, enclosed on some sides
    - FOOTSLOPE (8): Gently sloping base of steeper slope
    - VALLEY (9): Linear low, elongated trough
    - PIT (10): Enclosed depression, surrounded by higher terrain
    """

    FLAT = 1
    PEAK = 2
    RIDGE = 3
    SHOULDER = 4
    SPUR = 5
    SLOPE = 6
    HOLLOW = 7
    FOOTSLOPE = 8
    VALLEY = 9
    PIT = 10

    @classmethod
    def get_name(cls, value: int) -> str:
        """Get human-readable name for geomorphon type"""
        return cls(value).name.lower()

    @classmethod
    def get_description(cls, value: int) -> str:
        """Get description of geomorphon type"""
        descriptions = {
            cls.FLAT: "Level terrain with minimal relief",
            cls.PEAK: "Elevated point surrounded by lower terrain",
            cls.RIDGE: "Linear high, elongated crest line",
            cls.SHOULDER: "Convex break in slope",
            cls.SPUR: "Projecting ridge from main high area",
            cls.SLOPE: "Planar inclined surface",
            cls.HOLLOW: "Concave depression, partially enclosed",
            cls.FOOTSLOPE: "Gently sloping base of steeper slope",
            cls.VALLEY: "Linear low, elongated trough",
            cls.PIT: "Enclosed depression surrounded by higher terrain",
        }
        return descriptions[cls(value)]


@dataclass
class GeomorphonConfig:
    """Configuration for geomorphon computation"""

    search_radius: float  # meters
    flatness_threshold: float = 1.0  # degrees
    skip_radius: int = 0  # inner radius to skip
    n_directions: int = 8  # number of azimuth directions


class GeomorphonAnalyzer:
    """
    Computes multi-scale geomorphon classifications from DEMs.

    Uses line-of-sight analysis in 8 directions to classify each cell
    into one of 10 fundamental landform types. Computes at multiple
    scales to capture hierarchical terrain structure.

    Example:
        >>> analyzer = GeomorphonAnalyzer()
        >>> geomorphons = analyzer.compute_multiscale_geomorphons(dem, cell_size=0.5)
        >>> micro_scale = geomorphons['2m']  # Micro-features
        >>> regional_scale = geomorphons['25m']  # Regional patterns
    """

    # 8 directions (N, NE, E, SE, S, SW, W, NW)
    DIRECTIONS = [
        (0, 1),  # N
        (1, 1),  # NE
        (1, 0),  # E
        (1, -1),  # SE
        (0, -1),  # S
        (-1, -1),  # SW
        (-1, 0),  # W
        (-1, 1),  # NW
    ]

    # Ternary pattern to geomorphon lookup table
    # Pattern is 8-digit ternary: each digit is +1 (up), 0 (flat), -1 (down)
    # Converted to signature string for lookup
    PATTERN_LOOKUP = {
        # Format: pattern signature -> geomorphon type
        # All down: pit
        "--------": GeomorphonType.PIT,
        # All up: peak
        "++++++++": GeomorphonType.PEAK,
        # Linear patterns: ridge/valley
        "--++--++": GeomorphonType.RIDGE,
        "++++----": GeomorphonType.RIDGE,
        "++--++--": GeomorphonType.VALLEY,
        "----++++": GeomorphonType.VALLEY,
    }

    def __init__(self, flatness_threshold: float = 1.0, skip_radius: int = 0):
        """
        Initialize geomorphon analyzer.

        Args:
            flatness_threshold: Angle threshold (degrees) for flatness detection
            skip_radius: Inner radius (in cells) to skip from center
        """
        self.flatness_threshold = flatness_threshold
        self.skip_radius = skip_radius

    def compute_multiscale_geomorphons(
        self, dem: np.ndarray, cell_size: float = 0.5, scales: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute geomorphons at multiple scales.

        Args:
            dem: Digital Elevation Model array (height, width)
            cell_size: DEM cell size in meters
            scales: List of search radii in meters (default: [2, 5, 10, 25])

        Returns:
            Dictionary mapping scale name to geomorphon array
            Keys: '2m', '5m', '10m', '25m'
        """
        if scales is None:
            scales = [2.0, 5.0, 10.0, 25.0]

        logger.info(f"Computing geomorphons at {len(scales)} scales: {scales}")

        results = {}

        for radius in scales:
            scale_name = f"{int(radius)}m"
            logger.info(f"Computing {scale_name} scale geomorphons...")

            config = GeomorphonConfig(
                search_radius=radius,
                flatness_threshold=self.flatness_threshold,
                skip_radius=self.skip_radius,
            )

            geomorphons = self._compute_geomorphons(dem, cell_size, config)
            results[scale_name] = geomorphons

            # Log distribution
            unique, counts = np.unique(geomorphons[geomorphons > 0], return_counts=True)
            logger.debug(f"{scale_name} distribution:")
            for gtype, count in zip(unique, counts):
                pct = 100 * count / geomorphons.size
                logger.debug(f"  {GeomorphonType.get_name(gtype)}: {pct:.1f}%")

        logger.info("Multi-scale geomorphon computation complete")

        return results

    def _compute_geomorphons(
        self, dem: np.ndarray, cell_size: float, config: GeomorphonConfig
    ) -> np.ndarray:
        """
        Compute geomorphons for a single scale.

        Args:
            dem: DEM array
            cell_size: Cell size in meters
            config: Geomorphon configuration

        Returns:
            Geomorphon classification array (same shape as DEM)
        """
        height, width = dem.shape
        geomorphons = np.zeros((height, width), dtype=np.uint8)

        # Convert search radius to cells
        search_radius_cells = int(config.search_radius / cell_size)

        logger.debug(f"Search radius: {config.search_radius}m = {search_radius_cells} cells")

        # Process each cell
        for i in range(height):
            if i % 100 == 0 and i > 0:
                logger.debug(f"Processing row {i}/{height} ({100*i/height:.1f}%)")

            for j in range(width):
                # Get elevation at center
                z_center = dem[i, j]

                # Skip nodata
                if np.isnan(z_center) or z_center == -9999.0:
                    continue

                # Compute ternary pattern
                pattern = self._compute_ternary_pattern(
                    dem, i, j, z_center, search_radius_cells, cell_size, config
                )

                # Classify based on pattern
                geomorphon_type = self._classify_pattern(pattern)
                geomorphons[i, j] = geomorphon_type

        return geomorphons

    def _compute_ternary_pattern(
        self,
        dem: np.ndarray,
        row: int,
        col: int,
        z_center: float,
        search_radius: int,
        cell_size: float,
        config: GeomorphonConfig,
    ) -> List[int]:
        """
        Compute ternary line-of-sight pattern in 8 directions.

        For each direction, determines if line of sight is:
        - Ascending (+1): Looking up
        - Level (0): Flat
        - Descending (-1): Looking down

        Returns:
            List of 8 values: +1, 0, or -1 for each direction
        """
        pattern = []

        for dx, dy in self.DIRECTIONS:
            # Search along this direction
            zenith_angle = 0.0  # Maximum upward angle seen
            nadir_angle = 0.0  # Maximum downward angle seen

            for dist in range(config.skip_radius + 1, search_radius + 1):
                # Calculate position
                i = row + dy * dist
                j = col + dx * dist

                # Check bounds
                if i < 0 or i >= dem.shape[0] or j < 0 or j >= dem.shape[1]:
                    break

                # Get elevation
                z = dem[i, j]

                # Skip nodata
                if np.isnan(z) or z == -9999.0:
                    continue

                # Calculate angle (in degrees)
                horizontal_dist = dist * cell_size
                vertical_diff = z - z_center
                angle = np.degrees(np.arctan2(vertical_diff, horizontal_dist))

                # Update zenith/nadir angles
                if angle > zenith_angle:
                    zenith_angle = angle
                if angle < nadir_angle:
                    nadir_angle = angle

            # Determine ternary value based on angles
            if zenith_angle > config.flatness_threshold:
                ternary = +1  # Ascending
            elif abs(nadir_angle) > config.flatness_threshold:
                ternary = -1  # Descending
            else:
                ternary = 0  # Level

            pattern.append(ternary)

        return pattern

    def _classify_pattern(self, pattern: List[int]) -> int:
        """
        Classify ternary pattern into geomorphon type.

        Uses a combination of lookup table and heuristic rules.

        Args:
            pattern: List of 8 ternary values (+1, 0, -1)

        Returns:
            GeomorphonType value (1-10)
        """
        # Convert pattern to signature string
        signature = "".join(["+" if p > 0 else "-" if p < 0 else "0" for p in pattern])

        # Check lookup table
        if signature in self.PATTERN_LOOKUP:
            return self.PATTERN_LOOKUP[signature]

        # Count ups, downs, and flats
        n_up = sum(1 for p in pattern if p > 0)
        n_down = sum(1 for p in pattern if p < 0)
        n_flat = sum(1 for p in pattern if p == 0)

        # Heuristic classification rules
        if n_flat >= 6:
            return GeomorphonType.FLAT

        elif n_up >= 6:
            return GeomorphonType.PEAK

        elif n_down >= 6:
            return GeomorphonType.PIT

        # Check for linear patterns (ridge/valley)
        elif self._is_linear_high(pattern):
            return GeomorphonType.RIDGE

        elif self._is_linear_low(pattern):
            return GeomorphonType.VALLEY

        # Check for shoulder (convex break)
        elif n_up >= 4 and n_down >= 2:
            return GeomorphonType.SHOULDER

        # Check for footslope (gentle base)
        elif n_down >= 4 and n_up >= 2:
            return GeomorphonType.FOOTSLOPE

        # Check for spur (projecting ridge)
        elif self._is_spur(pattern):
            return GeomorphonType.SPUR

        # Check for hollow (concave depression)
        elif self._is_hollow(pattern):
            return GeomorphonType.HOLLOW

        # Default to slope
        else:
            return GeomorphonType.SLOPE

    def _is_linear_high(self, pattern: List[int]) -> bool:
        """Check if pattern represents a linear high (ridge)"""
        # Ridge: high along one axis, low perpendicular
        # Check N-S and E-W axes
        ns_axis = [pattern[0], pattern[4]]  # N, S
        ew_axis = [pattern[2], pattern[6]]  # E, W

        if all(p <= 0 for p in ns_axis) and any(p > 0 for p in ew_axis):
            return True
        if all(p <= 0 for p in ew_axis) and any(p > 0 for p in ns_axis):
            return True

        return False

    def _is_linear_low(self, pattern: List[int]) -> bool:
        """Check if pattern represents a linear low (valley)"""
        # Valley: low along one axis, high perpendicular
        ns_axis = [pattern[0], pattern[4]]  # N, S
        ew_axis = [pattern[2], pattern[6]]  # E, W

        if all(p >= 0 for p in ns_axis) and any(p < 0 for p in ew_axis):
            return True
        if all(p >= 0 for p in ew_axis) and any(p < 0 for p in ns_axis):
            return True

        return False

    def _is_spur(self, pattern: List[int]) -> bool:
        """Check if pattern represents a spur (projecting ridge)"""
        # Spur: high in one direction, mixed in others
        n_up = sum(1 for p in pattern if p > 0)
        # Look for concentration of ups in adjacent directions
        for i in range(len(pattern)):
            adjacent = [
                pattern[i],
                pattern[(i + 1) % len(pattern)],
                pattern[(i + 2) % len(pattern)],
            ]
            if sum(1 for p in adjacent if p > 0) >= 2 and n_up <= 4:
                return True
        return False

    def _is_hollow(self, pattern: List[int]) -> bool:
        """Check if pattern represents a hollow (concave depression)"""
        # Hollow: down in some directions but not all (partially enclosed)
        n_down = sum(1 for p in pattern if p < 0)
        return 3 <= n_down <= 5

    def get_geomorphon_stack(self, multiscale_results: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Stack multi-scale geomorphons into single array.

        Args:
            multiscale_results: Dict from compute_multiscale_geomorphons()

        Returns:
            Array of shape (n_scales, height, width)
        """
        # Get scales in order
        scale_order = ["2m", "5m", "10m", "25m"]
        available_scales = [s for s in scale_order if s in multiscale_results]

        if not available_scales:
            raise ValueError("No geomorphon results to stack")

        # Stack arrays
        stacked = np.stack([multiscale_results[s] for s in available_scales], axis=0)

        logger.debug(f"Stacked {len(available_scales)} scales: {stacked.shape}")

        return stacked

    # =========================================================================
    # OPENNESS-BASED GEOMORPHON METHODS (Pingel approach)
    # =========================================================================

    def compute_multiscale_geomorphons_openness(
        self,
        dem: np.ndarray,
        cell_size: float = 0.5,
        scales: Optional[List[float]] = None,
        threshold_angle: float = 1.0,
        use_negative_openness: bool = True,
        include_patterns: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Compute geomorphons using openness-based approach (Pingel method).

        This approach is more robust for detecting subtle terrain features
        like trails (shallow linear valleys) compared to direct elevation.

        Args:
            dem: Digital Elevation Model array (height, width)
            cell_size: DEM cell size in meters
            scales: List of search radii in meters (default: [2, 5, 10, 25])
            threshold_angle: Flatness threshold in degrees (default: 1.0)
            use_negative_openness: Use difference of positive and negative
                                   openness (recommended for geomorphons)
            include_patterns: If True, also return symbolic ternary patterns
                             (e.g., "++--++--") for LLM training

        Returns:
            Dictionary mapping:
            - '{scale}m': geomorphon array (1-10)
            - '{scale}m_patterns': symbolic pattern strings (if include_patterns=True)

        Example:
            >>> results = analyzer.compute_multiscale_geomorphons_openness(dem)
            >>> geom_2m = results['2m']  # Geomorphon classes
            >>> patterns_2m = results['2m_patterns']  # Symbolic strings
            >>> # Find PIT locations
            >>> pit_mask = patterns_2m == '++++++++'
        """
        if scales is None:
            scales = [2.0, 5.0, 10.0, 25.0]

        logger.info(f"Computing openness-based geomorphons at {len(scales)} scales")

        results = {}

        for radius in scales:
            scale_name = f"{int(radius)}m"
            logger.info(f"Computing {scale_name} scale (openness method)...")

            # Convert radius to pixels
            lookup_pixels = max(1, int(radius / cell_size))

            # Compute ternary pattern from openness
            ternary_codes = ternary_pattern_from_openness(
                dem,
                cell_size=cell_size,
                lookup_pixels=lookup_pixels,
                threshold_angle=threshold_angle,
                use_negative_openness=use_negative_openness,
            )

            # Convert to geomorphon types using loose classification
            geomorphons = terrain_code_to_geomorphon(ternary_codes, method="loose")
            results[scale_name] = geomorphons

            # Convert to symbolic pattern strings for LLM training
            if include_patterns:
                pattern_strings = ternary_codes_to_symbols(ternary_codes)
                results[f"{scale_name}_patterns"] = pattern_strings

            # Log distribution
            unique, counts = np.unique(geomorphons[geomorphons > 0], return_counts=True)
            logger.debug(f"{scale_name} distribution:")
            for gtype, count in zip(unique, counts):
                pct = 100 * count / geomorphons.size
                logger.debug(f"  {GeomorphonType.get_name(gtype)}: {pct:.1f}%")

        logger.info("Openness-based geomorphon computation complete")
        return results

    def compute_openness(
        self,
        dem: np.ndarray,
        cell_size: float = 0.5,
        lookup_pixels: int = 10,
        directions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute terrain openness (Yokoyama et al., 2002).

        Openness measures the degree to which a location is enclosed or
        exposed. It's the mean of the minimum zenith angles in each direction.

        Args:
            dem: Digital Elevation Model array
            cell_size: DEM cell size in meters
            lookup_pixels: Search distance in pixels
            directions: Which directions to compute (0-7, default: all 8)

        Returns:
            Openness array in degrees (0-90, higher = more open)
        """
        return compute_openness(
            dem,
            cell_size=cell_size,
            lookup_pixels=lookup_pixels,
            neighbors=directions,
        )


# =============================================================================
# STANDALONE OPENNESS FUNCTIONS (based on Pingel's neilpy)
# =============================================================================


def _ashift(arr: np.ndarray, direction: int, distance: int) -> np.ndarray:
    """
    Shift array in one of 8 directions by given distance.

    Directions (following image coordinates):
        0: East (+x)
        1: Southeast (+x, +y)
        2: South (+y)
        3: Southwest (-x, +y)
        4: West (-x)
        5: Northwest (-x, -y)
        6: North (-y)
        7: Northeast (+x, -y)

    Args:
        arr: Input array
        direction: Direction index (0-7)
        distance: Shift distance in pixels

    Returns:
        Shifted array (same shape, edge values are NaN)
    """
    result = np.full_like(arr, np.nan, dtype=np.float64)
    nrows, ncols = arr.shape

    # Direction offsets (dx, dy)
    offsets = [
        (1, 0),  # 0: E
        (1, 1),  # 1: SE
        (0, 1),  # 2: S
        (-1, 1),  # 3: SW
        (-1, 0),  # 4: W
        (-1, -1),  # 5: NW
        (0, -1),  # 6: N
        (1, -1),  # 7: NE
    ]

    dx, dy = offsets[direction]
    dx *= distance
    dy *= distance

    # Calculate source and destination slices
    if dx >= 0:
        src_x = slice(0, ncols - dx) if dx > 0 else slice(None)
        dst_x = slice(dx, None) if dx > 0 else slice(None)
    else:
        src_x = slice(-dx, None)
        dst_x = slice(0, ncols + dx)

    if dy >= 0:
        src_y = slice(0, nrows - dy) if dy > 0 else slice(None)
        dst_y = slice(dy, None) if dy > 0 else slice(None)
    else:
        src_y = slice(-dy, None)
        dst_y = slice(0, nrows + dy)

    result[dst_y, dst_x] = arr[src_y, src_x]

    return result


def compute_openness(
    dem: np.ndarray,
    cell_size: float = 1.0,
    lookup_pixels: int = 10,
    neighbors: Optional[np.ndarray] = None,
    fast: bool = False,
    how_fast: int = 20,
    pad_edges: bool = True,
) -> np.ndarray:
    """
    Compute terrain openness (Yokoyama et al., 2002).

    Openness is the mean of minimum zenith angles in each direction,
    measuring how exposed or enclosed a terrain location is.

    Args:
        dem: Digital Elevation Model array
        cell_size: DEM cell size in meters
        lookup_pixels: Maximum search distance in pixels
        neighbors: Array of direction indices to compute (default: all 8)
        fast: Use progressive sampling for speed (less accurate)
        how_fast: Number of sample points when fast=True
        pad_edges: Pad DEM edges to reduce boundary artifacts (default: True)

    Returns:
        Openness array in degrees (higher = more open/exposed)
    """
    if neighbors is None:
        neighbors = np.arange(8)

    # Pad DEM to reduce edge effects - use reflection for realistic boundaries
    if pad_edges:
        pad_width = lookup_pixels
        dem_padded = np.pad(dem, pad_width, mode="reflect")
    else:
        dem_padded = dem
        pad_width = 0

    nrows, ncols = dem_padded.shape
    opn = np.full((len(neighbors), nrows, ncols), np.inf, dtype=np.float64)

    # Distance multipliers for diagonal vs cardinal directions
    # Cardinal (N,S,E,W) = 1, Diagonal (NE,SE,SW,NW) = sqrt(2)
    dist_mult = np.array([1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)])

    # Determine which distances to test
    if fast:
        test_range = _progressive_window(1, lookup_pixels, how_fast)
    else:
        test_range = np.arange(1, lookup_pixels + 1)

    for L in test_range:
        for i, direction in enumerate(neighbors):
            # Calculate horizontal distance
            dist = cell_size * L * dist_mult[direction]

            # Get shifted elevation (using padded DEM)
            shifted = _ashift(dem_padded, direction, L)

            # Calculate zenith angle: angle from horizontal to line of sight
            # arctan(dz/dist) gives angle, subtract from 90 to get zenith
            dz = shifted - dem_padded
            angles = (np.pi / 2) - np.arctan(dz / dist)

            # Keep minimum angle seen so far
            this_layer = opn[i, :, :]
            mask = angles < this_layer
            this_layer[mask] = angles[mask]
            opn[i, :, :] = this_layer

    # Replace inf with NaN for proper averaging, then return in degrees
    opn[np.isinf(opn)] = np.nan
    result = np.rad2deg(np.nanmean(opn, axis=0))

    # Crop back to original size if we padded
    if pad_edges and pad_width > 0:
        result = result[pad_width:-pad_width, pad_width:-pad_width]

    return result


def _progressive_window(start: int, end: int, num_points: int) -> np.ndarray:
    """Generate progressively spaced sample points (denser near start)."""
    if num_points >= (end - start):
        return np.arange(start, end + 1)

    # Use logarithmic spacing for progressive sampling
    log_points = np.logspace(0, np.log10(end - start + 1), num_points)
    points = np.unique(np.round(log_points + start - 1).astype(int))
    return points[points <= end]


def _int2base(n: int, base: int, width: int = 8) -> str:
    """Convert integer to base-N string representation."""
    if n == 0:
        return "0" * width

    digits = []
    while n:
        digits.append(str(n % base))
        n //= base

    result = "".join(digits[::-1])
    return result.zfill(width)


def _get_lowest_equivalent(terrain_code: int) -> int:
    """
    Get rotationally and reflectionally equivalent minimum terrain code.

    This normalizes terrain codes so that patterns that differ only by
    rotation or reflection map to the same value.

    Args:
        terrain_code: Ternary terrain code (0 to 3^8-1)

    Returns:
        Minimum equivalent code
    """
    s = _int2base(terrain_code, 3, 8)
    min_val = int(s, 3)

    # Try all rotations and reflections
    for j in range(1, 16):
        # Rotate
        s = s[-1] + s[:-1]
        min_val = min(min_val, int(s, 3))

        # At halfway point, reflect
        if j == 7:
            s = s[::-1]

    return min_val


def ternary_pattern_from_openness(
    dem: np.ndarray,
    cell_size: float = 1.0,
    lookup_pixels: int = 10,
    threshold_angle: float = 1.0,
    use_negative_openness: bool = True,
    normalize: bool = False,
    vectorized: bool = True,
) -> np.ndarray:
    """
    Compute ternary terrain pattern from openness (Pingel approach).

    For each of 8 directions, computes openness (or difference between
    positive and negative openness) and classifies as:
    - 2: Above threshold (looking up)
    - 1: Within threshold (flat)
    - 0: Below threshold (looking down)

    The result is encoded as a base-3 number (0 to 3^8-1 = 6560).

    Args:
        dem: Digital Elevation Model array
        cell_size: DEM cell size in meters
        lookup_pixels: Maximum search distance in pixels
        threshold_angle: Flatness threshold in degrees
        use_negative_openness: If True, use difference of positive and
                               negative openness (more robust for geomorphons)
        normalize: If True, normalize codes to lowest rotational equivalent
        vectorized: If True, use faster vectorized computation (default: True)

    Returns:
        Array of ternary terrain codes (0-6560)
    """
    if vectorized:
        return _ternary_pattern_vectorized(
            dem, cell_size, lookup_pixels, threshold_angle, use_negative_openness, normalize
        )

    # Fallback to original method
    nrows, ncols = dem.shape
    tc = np.zeros((nrows, ncols), dtype=np.uint16)

    # Powers of 3 for encoding
    pows = 3 ** np.arange(8)

    for direction in range(8):
        # Compute openness in this direction
        opn = compute_openness(dem, cell_size, lookup_pixels, neighbors=np.array([direction]))

        if use_negative_openness:
            # Compute negative openness (openness of inverted terrain)
            O_neg = compute_openness(
                -dem, cell_size, lookup_pixels, neighbors=np.array([direction])
            )
            opn = opn - O_neg
        else:
            # Simple openness relative to horizontal
            opn = opn - 90.0

        # Classify into ternary values
        ternary = np.ones((nrows, ncols), dtype=np.uint16)  # Default: flat (1)
        ternary[opn > threshold_angle] = 2  # Above threshold: up
        ternary[opn < -threshold_angle] = 0  # Below threshold: down

        # Encode into terrain code
        tc = tc + ternary * pows[direction]

    # Optionally normalize to lowest equivalent
    if normalize:
        lookup_table = np.array([_get_lowest_equivalent(x) for x in range(3**8)])
        tc = lookup_table[tc]

    return tc


def _ternary_pattern_vectorized(
    dem: np.ndarray,
    cell_size: float = 1.0,
    lookup_pixels: int = 10,
    threshold_angle: float = 1.0,
    use_negative_openness: bool = True,
    normalize: bool = False,
) -> np.ndarray:
    """
    Vectorized computation of ternary patterns - computes all 8 directions simultaneously.

    This is 4-8x faster than the sequential version because:
    1. Computes all 8 direction openness values in one call
    2. Uses vectorized classification and encoding
    3. Avoids repeated function call overhead
    """
    nrows, ncols = dem.shape

    # Compute openness for all 8 directions at once
    # Returns shape (nrows, ncols) with mean of all directions
    # We need individual direction values, so we compute them in batches

    # Pad DEM once for all directions
    pad_width = lookup_pixels
    dem_padded = np.pad(dem, pad_width, mode="reflect")

    # Distance multipliers for diagonal vs cardinal directions
    dist_mult = np.array([1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)])

    # Pre-allocate arrays for all 8 directions
    nrows_pad, ncols_pad = dem_padded.shape
    openness_8dir = np.full((8, nrows_pad, ncols_pad), np.inf, dtype=np.float64)

    # Distance values to test
    test_range = np.arange(1, lookup_pixels + 1)

    # Compute openness for all directions in parallel loops
    for L in test_range:
        for direction in range(8):
            dist = cell_size * L * dist_mult[direction]

            # Get shifted elevation
            shifted = _ashift(dem_padded, direction, L)

            # Calculate zenith angle
            dz = shifted - dem_padded
            angles = (np.pi / 2) - np.arctan(dz / dist)

            # Keep minimum angle seen so far
            np.minimum(openness_8dir[direction], angles, out=openness_8dir[direction])

    # Replace inf with NaN and convert to degrees
    openness_8dir[np.isinf(openness_8dir)] = np.nan
    openness_8dir = np.rad2deg(openness_8dir)

    # Crop back to original size
    O_pos = openness_8dir[:, pad_width:-pad_width, pad_width:-pad_width]

    if use_negative_openness:
        # Compute negative openness (inverted terrain)
        neg_dem_padded = np.pad(-dem, pad_width, mode="reflect")
        openness_8dir_neg = np.full((8, nrows_pad, ncols_pad), np.inf, dtype=np.float64)

        for L in test_range:
            for direction in range(8):
                dist = cell_size * L * dist_mult[direction]
                shifted = _ashift(neg_dem_padded, direction, L)
                dz = shifted - neg_dem_padded
                angles = (np.pi / 2) - np.arctan(dz / dist)
                np.minimum(openness_8dir_neg[direction], angles, out=openness_8dir_neg[direction])

        openness_8dir_neg[np.isinf(openness_8dir_neg)] = np.nan
        openness_8dir_neg = np.rad2deg(openness_8dir_neg)
        O_neg = openness_8dir_neg[:, pad_width:-pad_width, pad_width:-pad_width]

        # Difference of positive and negative openness
        O_diff = O_pos - O_neg  # Shape: (8, nrows, ncols)
    else:
        O_diff = O_pos - 90.0

    # Vectorized classification into ternary values
    # Shape: (8, nrows, ncols), values: 0, 1, or 2
    ternary_8dir = np.ones((8, nrows, ncols), dtype=np.uint16)
    ternary_8dir[O_diff > threshold_angle] = 2  # Up
    ternary_8dir[O_diff < -threshold_angle] = 0  # Down

    # Encode all 8 directions into single terrain code using dot product
    # tc = sum(ternary_8dir[d] * 3^d for d in range(8))
    tc = np.tensordot(_POWERS_OF_3, ternary_8dir, axes=([0], [0]))

    # Optionally normalize
    if normalize:
        lookup_table = np.array([_get_lowest_equivalent(x) for x in range(3**8)], dtype=np.uint16)
        tc = lookup_table[tc]

    return tc.astype(np.uint16)


def terrain_code_to_geomorphon(
    terrain_code: Union[int, np.ndarray], method: str = "loose"
) -> Union[int, np.ndarray]:
    """
    Convert ternary terrain codes to geomorphon classifications.

    Args:
        terrain_code: Terrain code (0-6560) or array of codes
        method: Classification method:
            - 'strict': Only exact pattern matches (sparse classification)
            - 'loose': Use count-based lookup (complete classification)

    Returns:
        Geomorphon type (1-10) or array of types

    Geomorphon types:
        1: Flat, 2: Peak, 3: Ridge, 4: Shoulder, 5: Spur,
        6: Slope, 7: Hollow, 8: Footslope, 9: Valley, 10: Pit
    """
    # Build lookup table
    lookup_table = np.zeros(3**8, dtype=np.uint8)

    if method == "strict":
        # Only exact patterns (many cells will be unclassified)
        lookup_table[3280] = 1  # Flat (all 1s)
        lookup_table[0] = 2  # Peak (all 0s in original, means all up in openness)
        lookup_table[82] = 3  # Ridge
        lookup_table[121] = 4  # Shoulder
        lookup_table[26] = 5  # Spur
        lookup_table[160] = 6  # Slope
        lookup_table[242] = 7  # Hollow
        lookup_table[3293] = 8  # Footslope
        lookup_table[4346] = 9  # Valley
        lookup_table[6560] = 10  # Pit (all 2s)

    elif method == "loose":
        # Count-based classification (complete coverage)
        # Based on count of ups (2s) and downs (0s) in ternary pattern
        # Rows = count of 2s (up), Cols = count of 0s (down)
        strict_table = np.array(
            [
                [1, 1, 1, 8, 8, 9, 9, 9, 10],  # 0 ups
                [1, 1, 8, 8, 8, 9, 9, 9, 0],  # 1 up
                [1, 4, 6, 6, 7, 7, 9, 0, 0],  # 2 ups
                [4, 4, 6, 6, 6, 7, 0, 0, 0],  # 3 ups
                [4, 4, 5, 6, 6, 0, 0, 0, 0],  # 4 ups
                [3, 3, 5, 5, 0, 0, 0, 0, 0],  # 5 ups
                [3, 3, 3, 0, 0, 0, 0, 0, 0],  # 6 ups
                [3, 3, 0, 0, 0, 0, 0, 0, 0],  # 7 ups
                [2, 0, 0, 0, 0, 0, 0, 0, 0],  # 8 ups
            ],
            dtype=np.uint8,
        )

        for i in range(3**8):
            base = _int2base(i, 3, 8)
            n_up = base.count("2")  # Count of ups
            n_down = base.count("0")  # Count of downs
            if n_up < 9 and n_down < 9:
                lookup_table[i] = strict_table[n_up, n_down]

    else:
        raise ValueError(f"method must be 'strict' or 'loose', got '{method}'")

    # Apply lookup
    if isinstance(terrain_code, np.ndarray):
        return lookup_table[terrain_code]
    else:
        return int(lookup_table[terrain_code])


def compute_positive_openness(
    dem: np.ndarray,
    cell_size: float = 1.0,
    lookup_pixels: int = 10,
) -> np.ndarray:
    """
    Compute positive openness (skyview-like measure).

    Positive openness measures how open/exposed a location is,
    looking outward and upward from each point.

    Args:
        dem: Digital Elevation Model array
        cell_size: DEM cell size in meters
        lookup_pixels: Maximum search distance in pixels

    Returns:
        Positive openness in degrees (higher = more exposed)
    """
    return compute_openness(dem, cell_size, lookup_pixels)


def compute_negative_openness(
    dem: np.ndarray,
    cell_size: float = 1.0,
    lookup_pixels: int = 10,
) -> np.ndarray:
    """
    Compute negative openness (enclosure measure).

    Negative openness is computed on the inverted terrain,
    measuring how enclosed/protected a location is.

    Args:
        dem: Digital Elevation Model array
        cell_size: DEM cell size in meters
        lookup_pixels: Maximum search distance in pixels

    Returns:
        Negative openness in degrees (higher = more enclosed)
    """
    return compute_openness(-dem, cell_size, lookup_pixels)


# =============================================================================
# TERNARY PATTERN FEATURES (Exposed for LLM training)
# =============================================================================


@dataclass
class TernaryPatternResult:
    """
    Result containing ternary patterns, symbolic strings, and geomorphon classifications.

    The ternary pattern (0-6560) contains more information than the 10
    geomorphon classes. By exposing both numeric codes AND symbolic strings,
    we enable:
    - Custom pattern matching beyond standard geomorphons
    - Discovery of new patterns specific to military features
    - Richer, more interpretable features for LLM training
    - Semantic similarity in string space (similar patterns = similar strings)

    Symbolic Pattern Format:
        8 characters, one per direction (E, SE, S, SW, W, NW, N, NE):
        - '+' : Looking up (higher terrain in this direction)
        - '0' : Flat (within threshold)
        - '-' : Looking down (lower terrain in this direction)

        Examples:
        - "++++++++": PIT (surrounded by higher terrain)
        - "--------": PEAK (surrounded by lower terrain)
        - "++--++--": VALLEY (linear low)
        - "--++--++": RIDGE (linear high)

    Attributes:
        ternary_patterns: Raw ternary codes (0-6560 for each cell)
        pattern_strings: Symbolic pattern strings (e.g., "++--++--")
        geomorphons: Classified geomorphon types (1-10)
        pattern_counts: Distribution of patterns in the data
        scale: Scale name (e.g., '2m')
    """

    ternary_patterns: np.ndarray
    pattern_strings: np.ndarray  # NEW: Symbolic strings like "++--++--"
    geomorphons: np.ndarray
    pattern_counts: Dict[int, int]
    scale: str


def decode_ternary_pattern(code: int) -> Tuple[int, ...]:
    """
    Decode a ternary code into 8 directional values.

    Each direction is encoded as:
    - 0: Looking down (below threshold)
    - 1: Flat (within threshold)
    - 2: Looking up (above threshold)

    Directions (in order):
        0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE

    Args:
        code: Ternary terrain code (0-6560)

    Returns:
        Tuple of 8 values (0, 1, or 2) for each direction

    Example:
        >>> decode_ternary_pattern(6560)  # All ups (PIT)
        (2, 2, 2, 2, 2, 2, 2, 2)
        >>> decode_ternary_pattern(0)     # All downs (PEAK)
        (0, 0, 0, 0, 0, 0, 0, 0)
        >>> decode_ternary_pattern(3280)  # All flat (FLAT)
        (1, 1, 1, 1, 1, 1, 1, 1)
    """
    # Use pre-computed lookup table for O(1) operation
    return tuple(_DECODE_TABLE[code])


def decode_ternary_patterns_batch(codes: np.ndarray) -> np.ndarray:
    """
    Decode multiple ternary codes into pattern arrays (vectorized).

    Args:
        codes: Array of ternary terrain codes (0-6560)

    Returns:
        Array of shape (len(codes), 8) with pattern values

    Example:
        >>> decode_ternary_patterns_batch(np.array([0, 6560, 3280]))
        array([[0, 0, 0, 0, 0, 0, 0, 0],
               [2, 2, 2, 2, 2, 2, 2, 2],
               [1, 1, 1, 1, 1, 1, 1, 1]], dtype=int8)
    """
    return _DECODE_TABLE[codes]


def encode_ternary_pattern(pattern: Tuple[int, ...]) -> int:
    """
    Encode 8 directional values into a ternary code.

    Args:
        pattern: Tuple of 8 values (0, 1, or 2) for each direction

    Returns:
        Ternary terrain code (0-6560)

    Example:
        >>> encode_ternary_pattern((2, 2, 2, 2, 2, 2, 2, 2))  # All ups
        6560
        >>> encode_ternary_pattern((1, 1, 1, 1, 1, 1, 1, 1))  # All flat
        3280
    """
    # Use vectorized dot product for O(1) operation
    return int(np.dot(pattern, _POWERS_OF_3))


def encode_ternary_patterns_batch(patterns: np.ndarray) -> np.ndarray:
    """
    Encode multiple pattern arrays into ternary codes (vectorized).

    Args:
        patterns: Array of shape (N, 8) with pattern values (0, 1, or 2)

    Returns:
        Array of ternary terrain codes (0-6560)

    Example:
        >>> patterns = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
        ...                      [2, 2, 2, 2, 2, 2, 2, 2]])
        >>> encode_ternary_patterns_batch(patterns)
        array([   0, 6560])
    """
    return np.dot(patterns, _POWERS_OF_3)


def describe_ternary_pattern(code: int) -> str:
    """
    Generate human-readable description of a ternary pattern.

    Args:
        code: Ternary terrain code (0-6560)

    Returns:
        Description string

    Example:
        >>> describe_ternary_pattern(6560)
        'All 8 directions looking up (enclosed depression/PIT)'
    """
    pattern = decode_ternary_pattern(code)
    n_up = sum(1 for p in pattern if p == 2)
    n_flat = sum(1 for p in pattern if p == 1)
    n_down = sum(1 for p in pattern if p == 0)

    geomorph = terrain_code_to_geomorphon(code, method="loose")
    geomorph_name = GeomorphonType.get_name(geomorph) if geomorph > 0 else "unclassified"

    # Direction names
    dirs = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]

    up_dirs = [dirs[i] for i, p in enumerate(pattern) if p == 2]
    down_dirs = [dirs[i] for i, p in enumerate(pattern) if p == 0]
    flat_dirs = [dirs[i] for i, p in enumerate(pattern) if p == 1]

    parts = []
    if up_dirs:
        parts.append(f"up: {','.join(up_dirs)}")
    if flat_dirs:
        parts.append(f"flat: {','.join(flat_dirs)}")
    if down_dirs:
        parts.append(f"down: {','.join(down_dirs)}")

    return f"[{n_up}↑ {n_flat}= {n_down}↓] ({'; '.join(parts)}) → {geomorph_name.upper()}"


# =============================================================================
# SYMBOLIC PATTERN STRINGS (For LLM Training)
# =============================================================================
# These functions convert numeric ternary codes (0-6560) to symbolic strings
# like "++--++--" which are more interpretable for LLM training.
#
# Symbol meanings:
#   '+' : Looking up (higher terrain in this direction)
#   '0' : Flat (within threshold)
#   '-' : Looking down (lower terrain in this direction)
#
# Direction order: E, SE, S, SW, W, NW, N, NE (same as decode_ternary_pattern)
# =============================================================================

# Symbols for ternary values: 0=down(-), 1=flat(0), 2=up(+)
_TERNARY_SYMBOLS = ("-", "0", "+")


def ternary_code_to_symbol(code: int) -> str:
    """
    Convert a numeric ternary code to a symbolic pattern string.

    The string uses 8 characters, one per direction (E, SE, S, SW, W, NW, N, NE):
    - '+' : Looking up (surrounded by lower terrain)
    - '0' : Flat (within flatness threshold)
    - '-' : Looking down (surrounded by higher terrain)

    Args:
        code: Ternary terrain code (0-6560)

    Returns:
        8-character symbolic string like "++--++--"

    Examples:
        >>> ternary_code_to_symbol(6560)  # All ups (PIT)
        '++++++++'
        >>> ternary_code_to_symbol(0)     # All downs (PEAK)
        '--------'
        >>> ternary_code_to_symbol(3280)  # All flat (FLAT)
        '00000000'

    Note:
        - PIT (surrounded by higher terrain) shows '+' in all directions
          because from the center, you're looking UP to surrounding terrain
        - PEAK (surrounded by lower terrain) shows '-' in all directions
          because from the center, you're looking DOWN to surrounding terrain
    """
    pattern = []
    for _ in range(8):
        digit = code % 3
        pattern.append(_TERNARY_SYMBOLS[digit])
        code //= 3
    return "".join(pattern)


def symbol_to_ternary_code(symbol: str) -> int:
    """
    Convert a symbolic pattern string back to a numeric ternary code.

    Args:
        symbol: 8-character symbolic string like "++--++--"

    Returns:
        Ternary terrain code (0-6560)

    Example:
        >>> symbol_to_ternary_code("++++++++")  # All ups (PIT)
        6560
        >>> symbol_to_ternary_code("--------")  # All downs (PEAK)
        0
    """
    symbol_to_value = {"-": 0, "0": 1, "+": 2}
    code = 0
    for i, char in enumerate(symbol[:8]):
        code += symbol_to_value.get(char, 1) * (3**i)
    return code


def ternary_codes_to_symbols(codes: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion of ternary codes array to symbolic pattern strings.

    This is optimized for processing entire DEM arrays at once.

    Args:
        codes: Array of ternary codes (0-6560)

    Returns:
        Array of 8-character symbolic strings (dtype='U8')

    Example:
        >>> codes = np.array([[6560, 0], [3280, 1000]])
        >>> symbols = ternary_codes_to_symbols(codes)
        >>> symbols
        array([['++++++++', '--------'],
               ['00000000', '+0-+0-+0']], dtype='<U8')
    """
    # Flatten for processing
    flat_codes = codes.ravel()
    n = len(flat_codes)

    # Pre-allocate result
    result = np.empty(n, dtype="U8")

    # Convert each code
    for i in range(n):
        code = int(flat_codes[i])
        pattern = []
        for _ in range(8):
            pattern.append(_TERNARY_SYMBOLS[code % 3])
            code //= 3
        result[i] = "".join(pattern)

    return result.reshape(codes.shape)


def compute_multiscale_ternary_patterns(
    dem: np.ndarray,
    cell_size: float = 0.5,
    scales: Optional[List[float]] = None,
    threshold_angle: float = 1.0,
    use_negative_openness: bool = True,
) -> Dict[str, TernaryPatternResult]:
    """
    Compute ternary patterns at multiple scales, preserving the full pattern.

    Unlike compute_multiscale_geomorphons_openness which only returns the
    10 geomorphon classes, this function returns:
    - Raw ternary patterns (0-6560)
    - Symbolic pattern strings (e.g., "++--++--")
    - Geomorphon classifications (1-10)

    The ternary pattern and symbolic strings contain more information:
    - 6561 possible patterns vs 10 geomorphon classes
    - Directional information preserved
    - Custom pattern matching possible
    - Symbolic strings are LLM-interpretable and compositional

    Args:
        dem: Digital Elevation Model array (height, width)
        cell_size: DEM cell size in meters
        scales: List of search radii in meters (default: [2, 5, 10, 25])
        threshold_angle: Flatness threshold in degrees (default: 1.0)
        use_negative_openness: Use difference of positive and negative
                               openness (recommended for geomorphons)

    Returns:
        Dictionary mapping scale name to TernaryPatternResult

    Example:
        >>> results = compute_multiscale_ternary_patterns(dem, cell_size=0.5)
        >>> for scale, result in results.items():
        ...     print(f"{scale}: {len(result.pattern_counts)} unique patterns")
        ...     # Access symbolic strings for LLM training
        ...     print(f"  Sample pattern: {result.pattern_strings[100, 100]}")
        ...     # Find all PIT patterns
        ...     pit_mask = result.pattern_strings == '++++++++'
    """
    if scales is None:
        scales = [2.0, 5.0, 10.0, 25.0]

    logger.info(f"Computing ternary patterns at {len(scales)} scales")

    results = {}

    for radius in scales:
        scale_name = f"{int(radius)}m"
        logger.info(f"Computing {scale_name} scale ternary patterns...")

        # Convert radius to pixels
        lookup_pixels = max(1, int(radius / cell_size))

        # Compute ternary pattern from openness
        ternary_codes = ternary_pattern_from_openness(
            dem,
            cell_size=cell_size,
            lookup_pixels=lookup_pixels,
            threshold_angle=threshold_angle,
            use_negative_openness=use_negative_openness,
        )

        # Convert to geomorphon types
        geomorphons = terrain_code_to_geomorphon(ternary_codes, method="loose")

        # Convert to symbolic pattern strings for LLM training
        pattern_strings = ternary_codes_to_symbols(ternary_codes)

        # Count pattern distribution
        unique, counts = np.unique(ternary_codes, return_counts=True)
        pattern_counts = dict(zip(unique.astype(int), counts.astype(int)))

        results[scale_name] = TernaryPatternResult(
            ternary_patterns=ternary_codes,
            pattern_strings=pattern_strings,
            geomorphons=geomorphons,
            pattern_counts=pattern_counts,
            scale=scale_name,
        )

        logger.info(f"  {scale_name}: {len(pattern_counts)} unique patterns")

    return results


def find_pattern_matches(
    ternary_patterns: np.ndarray, target_pattern: int, tolerance: int = 0
) -> np.ndarray:
    """
    Find cells matching a specific ternary pattern.

    Args:
        ternary_patterns: Array of ternary codes from compute_multiscale_ternary_patterns
        target_pattern: Pattern to search for (0-6560)
        tolerance: Allow patterns within this Hamming distance (0 = exact match)

    Returns:
        Boolean mask of matching cells

    Example:
        >>> # Find all enclosed depressions (all directions looking up)
        >>> matches = find_pattern_matches(patterns, 6560)  # PIT pattern
        >>> print(f"Found {matches.sum()} potential foxholes")
    """
    if tolerance == 0:
        return ternary_patterns == target_pattern

    # For tolerance > 0, compute Hamming distance
    target_decoded = np.array(decode_ternary_pattern(target_pattern))

    # Decode all patterns (vectorized)
    height, width = ternary_patterns.shape
    patterns_flat = ternary_patterns.flatten()

    # Count differences for each pattern
    distances = np.zeros(len(patterns_flat), dtype=np.int32)
    for i in range(8):
        pattern_val = (patterns_flat // (3**i)) % 3
        target_val = target_decoded[i]
        distances += (pattern_val != target_val).astype(np.int32)

    return (distances <= tolerance).reshape(height, width)


# Key ternary patterns for military features
MILITARY_PATTERNS = {
    # Foxhole signatures at 2m scale
    "foxhole_perfect": 6560,  # All directions looking up (deep pit)
    "foxhole_partial": 5467,  # 6 ups, 2 flat (partially filled)
    # Trench signatures (linear valley)
    "trench_ns": encode_ternary_pattern((1, 2, 2, 2, 1, 0, 0, 0)),  # N-S trench
    "trench_ew": encode_ternary_pattern((0, 0, 1, 2, 2, 2, 1, 0)),  # E-W trench
    # Crater signatures (pit with raised rim)
    "crater_center": 6560,  # Same as foxhole but larger scale
    # Ridge/defensive position
    "ridge_ns": encode_ternary_pattern((1, 0, 0, 0, 1, 2, 2, 2)),  # N-S ridge
    "ridge_ew": encode_ternary_pattern((2, 2, 1, 0, 0, 0, 1, 2)),  # E-W ridge
}


def get_pattern_signature_for_feature(feature_name: str) -> List[int]:
    """
    Get expected ternary patterns for a given feature type.

    Args:
        feature_name: Feature to get patterns for ('foxhole', 'trench', 'crater', etc.)

    Returns:
        List of ternary codes that could indicate this feature
    """
    patterns = []

    if feature_name.lower() == "foxhole":
        # Foxhole: mostly looking up (enclosed depression)
        for code in range(6561):
            decoded = decode_ternary_pattern(code)
            n_up = sum(1 for p in decoded if p == 2)
            if n_up >= 6:  # At least 6 directions looking up
                patterns.append(code)

    elif feature_name.lower() == "trench":
        # Trench: linear valley pattern
        for code in range(6561):
            decoded = decode_ternary_pattern(code)
            # Check for linear pattern (opposite directions similar)
            if (
                decoded[0] == decoded[4]  # E-W same
                and decoded[2] == decoded[6]  # N-S same
                and decoded[0] != decoded[2]
            ):  # But E-W different from N-S
                patterns.append(code)

    elif feature_name.lower() == "pit":
        # Any enclosed depression
        for code in range(6561):
            decoded = decode_ternary_pattern(code)
            n_up = sum(1 for p in decoded if p == 2)
            if n_up >= 5:
                patterns.append(code)

    elif feature_name.lower() == "ridge":
        # Linear high
        for code in range(6561):
            decoded = decode_ternary_pattern(code)
            n_down = sum(1 for p in decoded if p == 0)
            # Linear pattern check
            if decoded[0] == decoded[4] and decoded[2] == decoded[6] and n_down >= 4:
                patterns.append(code)

    return patterns


# =============================================================================
# MULTI-SCALE PERSISTENCE SCORING
# =============================================================================


def compute_scale_persistence(
    multiscale_geomorphons: Dict[str, np.ndarray],
    target_types: Optional[Set[GeomorphonType]] = None,
) -> np.ndarray:
    """
    Compute persistence score for each cell across multiple scales.

    Features that persist across scales are more reliable. This function
    measures how consistently a geomorphon type appears across scales.

    Args:
        multiscale_geomorphons: Dict with scale keys ('2m', '5m', etc.)
                                and geomorphon type arrays as values
        target_types: Optional set of GeomorphonType to look for.
                     If None, measures consistency of ANY type across scales.

    Returns:
        Array of persistence scores (0.0 to 1.0) where:
        - 1.0 = same type at all scales (highly persistent)
        - 0.0 = different type at every scale (unstable)

    Example:
        >>> # Load multiscale geomorphons
        >>> analyzer = GeomorphonAnalyzer()
        >>> multiscale = analyzer.compute_multiscale_geomorphons_openness(dem)
        >>> # Find cells where PIT persists across scales (likely foxholes)
        >>> persistence = compute_scale_persistence(
        ...     multiscale,
        ...     target_types={GeomorphonType.PIT, GeomorphonType.HOLLOW}
        ... )
        >>> foxhole_candidates = persistence > 0.75  # At least 3/4 scales
    """
    # Get scale keys (exclude pattern keys)
    scale_keys = [
        k for k in multiscale_geomorphons.keys() if k.endswith("m") and not k.endswith("_patterns")
    ]

    if len(scale_keys) == 0:
        raise ValueError("No scale keys found in multiscale_geomorphons")

    # Get shape from first scale
    first_key = scale_keys[0]
    shape = multiscale_geomorphons[first_key].shape

    if target_types is not None:
        # Count how many scales have a target type
        target_set = {int(t) for t in target_types}
        matches = np.zeros(shape, dtype=np.float32)

        for key in scale_keys:
            geomorphons = multiscale_geomorphons[key]
            is_target = np.isin(geomorphons, list(target_set))
            matches += is_target.astype(np.float32)

        return matches / len(scale_keys)

    else:
        # Measure consistency: most common type at each cell
        # Stack all scales
        stacked = np.stack([multiscale_geomorphons[k] for k in scale_keys], axis=0)

        # For each cell, find the mode (most common value) and its frequency
        persistence = np.zeros(shape, dtype=np.float32)

        for i in range(shape[0]):
            for j in range(shape[1]):
                values = stacked[:, i, j]
                unique, counts = np.unique(values, return_counts=True)
                max_count = counts.max()
                persistence[i, j] = max_count / len(scale_keys)

        return persistence


def compute_multiscale_confidence(
    multiscale_geomorphons: Dict[str, np.ndarray],
    feature_signatures: Dict[str, Set[GeomorphonType]],
    scale_weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Compute feature detection confidence using multi-scale signatures.

    This combines geomorphon types across scales with optional weighting
    to produce a confidence map for detecting specific features.

    Args:
        multiscale_geomorphons: Dict with scale keys and geomorphon arrays
        feature_signatures: Dict mapping scale names to expected GeomorphonType sets
                           Example: {'2m': {PIT, HOLLOW}, '5m': {PIT, FLAT}}
        scale_weights: Optional weights per scale (default: equal weights)

    Returns:
        Confidence array (0.0 to 1.0)

    Example:
        >>> # Foxhole signature: PIT at 2m+5m, anything at 10m+25m
        >>> foxhole_sig = {
        ...     '2m': {GeomorphonType.PIT, GeomorphonType.HOLLOW},
        ...     '5m': {GeomorphonType.PIT, GeomorphonType.HOLLOW, GeomorphonType.FLAT},
        ... }
        >>> confidence = compute_multiscale_confidence(multiscale, foxhole_sig)
        >>> foxholes = confidence > 0.8
    """
    # Get available scales
    scale_keys = [
        k for k in multiscale_geomorphons.keys() if k.endswith("m") and not k.endswith("_patterns")
    ]

    # Default equal weights
    if scale_weights is None:
        scale_weights = {k: 1.0 for k in feature_signatures.keys()}

    # Normalize weights
    total_weight = sum(scale_weights.get(k, 0) for k in feature_signatures.keys())
    if total_weight == 0:
        total_weight = 1.0

    shape = multiscale_geomorphons[scale_keys[0]].shape
    confidence = np.zeros(shape, dtype=np.float32)

    for scale_key, expected_types in feature_signatures.items():
        if scale_key not in multiscale_geomorphons:
            continue

        geomorphons = multiscale_geomorphons[scale_key]
        weight = scale_weights.get(scale_key, 1.0) / total_weight

        # Check if geomorphon matches any expected type
        type_values = [int(t) for t in expected_types]
        matches = np.isin(geomorphons, type_values).astype(np.float32)

        confidence += matches * weight

    return confidence


def find_persistent_features(
    multiscale_geomorphons: Dict[str, np.ndarray],
    target_type: GeomorphonType,
    min_persistence: float = 0.5,
    min_cluster_size: int = 1,
) -> List[Tuple[int, int, float]]:
    """
    Find locations where a geomorphon type persists across scales.

    Args:
        multiscale_geomorphons: Dict with scale keys and geomorphon arrays
        target_type: GeomorphonType to search for
        min_persistence: Minimum persistence score (0.0-1.0)
        min_cluster_size: Minimum number of connected cells

    Returns:
        List of (row, col, persistence_score) for each detection

    Example:
        >>> # Find persistent PITs (potential foxholes)
        >>> pits = find_persistent_features(
        ...     multiscale,
        ...     GeomorphonType.PIT,
        ...     min_persistence=0.75
        ... )
        >>> print(f"Found {len(pits)} potential foxholes")
    """
    from scipy import ndimage as ndi

    # Compute persistence for target type
    persistence = compute_scale_persistence(multiscale_geomorphons, target_types={target_type})

    # Threshold
    mask = persistence >= min_persistence

    # Label connected components
    labeled, n_features = ndi.label(mask)

    # Extract features
    features = []
    for label_id in range(1, n_features + 1):
        component = labeled == label_id
        size = component.sum()

        if size >= min_cluster_size:
            # Find centroid
            rows, cols = np.where(component)
            center_row = int(np.mean(rows))
            center_col = int(np.mean(cols))

            # Average persistence in component
            avg_persistence = float(persistence[component].mean())

            features.append((center_row, center_col, avg_persistence))

    return features
