"""
Subtle Terrain Feature Detection for TerraLLM

Specialized algorithms for detecting subtle linear features like trails,
paths, old roads, and other movement corridors that are critical for
battlefield analysis.

Key techniques:
1. Curvature analysis - Profile and planform curvature for subtle depressions
2. Sky-View Factor (SVF) - Reveals subtle linear features in shadowed areas
3. Topographic Position Index (TPI) - Detects micro-ridges and valleys
4. Relative Elevation Model (REM) - Removes regional trend to highlight micro-features
5. Linear feature detection - Finds connected linear structures

These techniques are standard in archaeological LiDAR analysis and are
particularly effective for detecting:
- Old trails and footpaths (10-30cm depth)
- Historical roads (often appear as subtle terraces)
- Defensive trenches (partially filled over 80 years)
- Movement corridors between positions

References:
- Kokalj & Hesse (2017). Airborne laser scanning raster data visualization
- Doneus (2013). Openness as visualization technique for interpretative mapping
- Štular et al. (2012). Visualization of lidar-derived relief models
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, uniform_filter

logger = logging.getLogger(__name__)


# =============================================================================
# CURVATURE ANALYSIS
# =============================================================================


def compute_curvature(
    dem: np.ndarray,
    cell_size: float = 1.0,
    curvature_type: str = "profile",
    smoothing: float = 0.0,
) -> np.ndarray:
    """
    Compute terrain curvature for detecting subtle depressions and ridges.

    Curvature measures the rate of change of slope, revealing subtle terrain
    features that are invisible in slope or hillshade visualizations.

    Args:
        dem: Digital Elevation Model
        cell_size: DEM resolution in meters
        curvature_type: Type of curvature to compute:
            - 'profile': Curvature in direction of steepest slope (trails appear as negative)
            - 'planform': Curvature perpendicular to slope (convergent flow)
            - 'mean': Average of profile and planform
            - 'total': Total curvature (Gaussian curvature)
        smoothing: Gaussian smoothing sigma (0 = no smoothing)

    Returns:
        Curvature array (negative = concave/valley, positive = convex/ridge)

    Example:
        >>> curv = compute_curvature(dem, cell_size=0.5, curvature_type='profile')
        >>> trails = curv < -0.01  # Subtle concave features (potential trails)
    """
    if smoothing > 0:
        dem = gaussian_filter(dem, sigma=smoothing)

    # Compute first derivatives (slope components)
    dz_dx = np.gradient(dem, cell_size, axis=1)
    dz_dy = np.gradient(dem, cell_size, axis=0)

    # Compute second derivatives
    d2z_dx2 = np.gradient(dz_dx, cell_size, axis=1)
    d2z_dy2 = np.gradient(dz_dy, cell_size, axis=0)
    d2z_dxdy = np.gradient(dz_dx, cell_size, axis=0)

    # Slope magnitude squared
    p = dz_dx
    q = dz_dy
    p2 = p * p
    q2 = q * q
    pq = p * q

    # Avoid division by zero
    denom = p2 + q2
    denom[denom < 1e-10] = 1e-10

    if curvature_type == "profile":
        # Profile curvature: curvature in direction of steepest descent
        # Negative = concave (valleys, trails), Positive = convex (ridges)
        curv = -(p2 * d2z_dx2 + 2 * pq * d2z_dxdy + q2 * d2z_dy2) / (denom * np.sqrt(1 + denom))

    elif curvature_type == "planform":
        # Planform curvature: curvature perpendicular to slope
        # Shows convergent/divergent flow patterns
        curv = -(q2 * d2z_dx2 - 2 * pq * d2z_dxdy + p2 * d2z_dy2) / (denom**1.5)

    elif curvature_type == "mean":
        # Mean curvature: average of principal curvatures
        curv = -((1 + q2) * d2z_dx2 - 2 * pq * d2z_dxdy + (1 + p2) * d2z_dy2) / (
            2 * (1 + denom) ** 1.5
        )

    elif curvature_type == "total":
        # Total (Gaussian) curvature: product of principal curvatures
        curv = (d2z_dx2 * d2z_dy2 - d2z_dxdy**2) / (1 + denom) ** 2

    else:
        raise ValueError(f"Unknown curvature type: {curvature_type}")

    return curv


def compute_all_curvatures(
    dem: np.ndarray,
    cell_size: float = 1.0,
    smoothing: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Compute all curvature types at once.

    Returns:
        Dict with 'profile', 'planform', 'mean', 'total' curvature arrays
    """
    return {
        "profile": compute_curvature(dem, cell_size, "profile", smoothing),
        "planform": compute_curvature(dem, cell_size, "planform", smoothing),
        "mean": compute_curvature(dem, cell_size, "mean", smoothing),
        "total": compute_curvature(dem, cell_size, "total", smoothing),
    }


# =============================================================================
# SKY-VIEW FACTOR (SVF)
# =============================================================================


def compute_sky_view_factor(
    dem: np.ndarray,
    cell_size: float = 1.0,
    search_radius: int = 10,
    n_directions: int = 16,
) -> np.ndarray:
    """
    Compute Sky-View Factor for revealing subtle terrain features.

    SVF measures the proportion of visible sky hemisphere at each point.
    Lower values indicate enclosed areas (pits, valleys, trails).
    Higher values indicate exposed areas (ridges, peaks).

    SVF is particularly effective for:
    - Detecting subtle linear depressions (trails, old roads)
    - Visualizing features under forest canopy
    - Revealing archaeological features

    Args:
        dem: Digital Elevation Model
        cell_size: DEM resolution in meters
        search_radius: Maximum search distance in pixels
        n_directions: Number of azimuth directions (8, 16, or 32)

    Returns:
        Sky-View Factor array (0-1, lower = more enclosed)

    Reference:
        Kokalj et al. (2011). Application of sky-view factor for
        the visualization of historic landscape features in lidar-derived
        relief models.
    """
    nrows, ncols = dem.shape

    # Pad DEM to handle edges
    pad = search_radius
    dem_padded = np.pad(dem, pad, mode="reflect")

    # Direction angles (evenly spaced around horizon)
    azimuths = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)

    # Direction vectors (dx, dy for each azimuth)
    # Note: in image coordinates, y increases downward
    dx = np.sin(azimuths)  # East is positive x
    dy = -np.cos(azimuths)  # North is negative y (up in image)

    # Compute maximum elevation angle in each direction
    max_angles = np.zeros((n_directions, nrows, ncols))

    for d, (ddx, ddy) in enumerate(zip(dx, dy)):
        # Horizontal distance multiplier for this direction
        dist_mult = np.sqrt(ddx**2 + ddy**2)

        # Check each distance
        for r in range(1, search_radius + 1):
            # Offset in pixels
            offset_x = int(round(r * ddx))
            offset_y = int(round(r * ddy))

            # Horizontal distance
            horiz_dist = r * cell_size * dist_mult

            # Get shifted DEM (with padding offset)
            shifted = dem_padded[
                pad + offset_y : pad + offset_y + nrows, pad + offset_x : pad + offset_x + ncols
            ]

            # Elevation difference
            dz = shifted - dem

            # Elevation angle (radians from horizontal)
            angle = np.arctan2(dz, horiz_dist)

            # Keep maximum angle
            np.maximum(max_angles[d], angle, out=max_angles[d])

    # SVF = 1 - mean(sin(max_angle)) for all directions
    # This is an approximation; exact SVF integrates over hemisphere
    svf = 1 - np.mean(np.sin(np.maximum(max_angles, 0)), axis=0)

    return svf


# =============================================================================
# TOPOGRAPHIC POSITION INDEX (TPI)
# =============================================================================


def compute_tpi(
    dem: np.ndarray,
    inner_radius: int = 0,
    outer_radius: int = 10,
) -> np.ndarray:
    """
    Compute Topographic Position Index for micro-feature detection.

    TPI measures elevation relative to local mean, revealing:
    - Positive TPI: Ridges, peaks, raised features
    - Zero TPI: Flat areas, mid-slopes
    - Negative TPI: Valleys, depressions, trails

    Args:
        dem: Digital Elevation Model
        inner_radius: Inner radius of annulus (0 = use circular kernel)
        outer_radius: Outer radius of analysis window

    Returns:
        TPI array (negative = below local mean, positive = above)

    Example:
        >>> tpi = compute_tpi(dem, outer_radius=5)
        >>> subtle_valleys = tpi < -0.1  # Cells below local mean
    """
    # Create circular or annular kernel
    y, x = np.ogrid[-outer_radius : outer_radius + 1, -outer_radius : outer_radius + 1]
    dist = np.sqrt(x * x + y * y)

    if inner_radius > 0:
        # Annular kernel
        kernel = ((dist >= inner_radius) & (dist <= outer_radius)).astype(float)
    else:
        # Circular kernel
        kernel = (dist <= outer_radius).astype(float)

    # Normalize kernel
    kernel /= kernel.sum()

    # Compute local mean
    local_mean = ndimage.convolve(dem, kernel, mode="reflect")

    # TPI = elevation - local mean
    tpi = dem - local_mean

    return tpi


def compute_multiscale_tpi(
    dem: np.ndarray,
    radii: List[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute TPI at multiple scales for detecting features of different sizes.

    Args:
        dem: Digital Elevation Model
        radii: List of outer radii to use (default: [3, 5, 10, 20])

    Returns:
        Dict with radius keys and TPI arrays
    """
    if radii is None:
        radii = [3, 5, 10, 20]

    result = {}
    for r in radii:
        result[f"{r}px"] = compute_tpi(dem, outer_radius=r)

    return result


# =============================================================================
# RELATIVE ELEVATION MODEL (REM)
# =============================================================================


def compute_relative_elevation(
    dem: np.ndarray,
    window_size: int = 50,
    method: str = "mean",
) -> np.ndarray:
    """
    Compute Relative Elevation Model to highlight micro-features.

    REM removes the regional terrain trend, revealing subtle local features
    that would otherwise be hidden by larger-scale topography.

    Args:
        dem: Digital Elevation Model
        window_size: Size of window for computing regional trend
        method: Method for computing trend:
            - 'mean': Moving average (faster)
            - 'median': Moving median (more robust to outliers)
            - 'gaussian': Gaussian-weighted average (smoother)

    Returns:
        Relative elevation array (deviation from local trend)

    Example:
        >>> rem = compute_relative_elevation(dem, window_size=100)
        >>> subtle_features = np.abs(rem) > 0.1  # Deviations > 10cm
    """
    if method == "mean":
        trend = uniform_filter(dem, size=window_size, mode="reflect")
    elif method == "median":
        trend = ndimage.median_filter(dem, size=window_size, mode="reflect")
    elif method == "gaussian":
        sigma = window_size / 4  # Approximate equivalent
        trend = gaussian_filter(dem, sigma=sigma, mode="reflect")
    else:
        raise ValueError(f"Unknown method: {method}")

    return dem - trend


# =============================================================================
# LINEAR FEATURE DETECTION
# =============================================================================


def detect_linear_features(
    feature_map: np.ndarray,
    threshold: float = 0.0,
    min_length: int = 10,
    max_gap: int = 2,
    direction_tolerance: float = 30.0,
) -> List[Dict]:
    """
    Detect linear features (trails, trenches, roads) in a feature map.

    Uses connected component analysis with linearity filtering.

    Args:
        feature_map: Input array (e.g., negative curvature, low SVF, negative TPI)
        threshold: Threshold for binarizing feature map (cells below this are candidates)
        min_length: Minimum length in pixels to be considered a trail
        max_gap: Maximum gap to bridge in linear features
        direction_tolerance: Maximum deviation from straight line (degrees)

    Returns:
        List of detected linear features with properties:
        - 'pixels': List of (row, col) coordinates
        - 'length': Length in pixels
        - 'orientation': Primary orientation in degrees
        - 'linearity': How straight the feature is (0-1)
        - 'centroid': Center point (row, col)
    """
    from scipy.ndimage import binary_dilation, binary_erosion, label

    # Binarize
    binary = feature_map < threshold

    # Close small gaps
    if max_gap > 0:
        struct = np.ones((max_gap * 2 + 1, max_gap * 2 + 1))
        binary = binary_dilation(binary, structure=struct, iterations=1)
        binary = binary_erosion(binary, structure=struct, iterations=1)

    # Label connected components
    labeled, n_features = label(binary)

    # Analyze each component for linearity
    linear_features = []

    for feat_id in range(1, n_features + 1):
        component = labeled == feat_id
        rows, cols = np.where(component)

        if len(rows) < min_length:
            continue

        # Compute linearity using PCA
        coords = np.column_stack([rows, cols])
        centroid = coords.mean(axis=0)
        centered = coords - centroid

        # Covariance and eigendecomposition
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Linearity: ratio of eigenvalues (1 = perfectly linear)
        if eigenvalues[1] > 0:
            linearity = 1 - np.sqrt(eigenvalues[1] / eigenvalues[0])
        else:
            linearity = 1.0

        # Length: extent along major axis
        projections = centered @ eigenvectors[:, 0]
        length = projections.max() - projections.min()

        # Orientation (degrees from horizontal)
        orientation = np.degrees(np.arctan2(eigenvectors[0, 0], eigenvectors[1, 0]))

        # Only keep sufficiently linear features
        if linearity > 0.5 and length >= min_length:
            linear_features.append(
                {
                    "pixels": list(zip(rows.tolist(), cols.tolist())),
                    "n_pixels": len(rows),
                    "length": float(length),
                    "orientation": float(orientation),
                    "linearity": float(linearity),
                    "centroid": (float(centroid[0]), float(centroid[1])),
                    "eigenvalues": eigenvalues.tolist(),
                }
            )

    return linear_features


# =============================================================================
# TRAIL DETECTION PIPELINE
# =============================================================================


@dataclass
class TrailCandidate:
    """A detected trail or path candidate."""

    pixels: List[Tuple[int, int]]
    length_m: float
    width_m: float
    depth_m: float
    orientation_deg: float
    linearity: float
    confidence: float
    centroid: Tuple[float, float]
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]


def detect_trails(
    dem: np.ndarray,
    cell_size: float = 0.5,
    min_length_m: float = 5.0,
    max_width_m: float = 3.0,
    min_depth_m: float = 0.05,
    max_depth_m: float = 0.5,
    linearity_threshold: float = 0.5,  # Lowered from 0.6 - trails have width
) -> List[TrailCandidate]:
    """
    Detect trail-like features in DEM using multiple indicators.

    Uses TPI (Topographic Position Index) as the primary detector because
    it provides a cleaner signal than curvature for subtle linear depressions.
    Profile curvature is used as a secondary indicator.

    Combines:
    1. TPI (trails are below local mean) - primary detector
    2. Profile curvature (trails are concave) - secondary indicator
    3. Linear feature detection (trails are elongated)
    4. Width/depth constraints

    Args:
        dem: Digital Elevation Model
        cell_size: DEM resolution in meters
        min_length_m: Minimum trail length in meters
        max_width_m: Maximum trail width in meters
        min_depth_m: Minimum trail depth (below surrounding)
        max_depth_m: Maximum trail depth
        linearity_threshold: Minimum linearity score (0-1)

    Returns:
        List of TrailCandidate objects

    Example:
        >>> trails = detect_trails(dem, cell_size=0.5, min_length_m=10)
        >>> for t in trails:
        ...     print(f"Trail: {t.length_m:.1f}m, depth={t.depth_m:.2f}m, conf={t.confidence:.2f}")
    """
    logger.info("Detecting trails in DEM...")

    # Compute TPI as primary indicator (cleaner signal than curvature)
    tpi_radius = int(max_width_m / cell_size) + 3
    tpi = compute_tpi(dem, outer_radius=tpi_radius)

    # Compute curvature as secondary indicator (with smoothing to reduce noise)
    curvature = compute_curvature(dem, cell_size, "profile", smoothing=2.0)

    # Use TPI for linear feature detection (threshold based on min_depth)
    # TPI provides cleaner linear features than curvature
    min_length_px = int(min_length_m / cell_size)

    # Detect linear features using TPI (negative = below local mean)
    linear_features = detect_linear_features(
        tpi,
        threshold=-min_depth_m,  # Use min_depth as TPI threshold
        min_length=min_length_px,
        max_gap=3,  # Allow small gaps in trail
    )

    logger.info(f"Found {len(linear_features)} linear candidates from TPI")

    # Convert to TrailCandidate objects with additional filtering
    trails = []

    for feat in linear_features:
        if feat["linearity"] < linearity_threshold:
            continue

        pixels = feat["pixels"]
        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]

        # Compute width from minor axis
        width_px = 4 * np.sqrt(feat["eigenvalues"][1]) if feat["eigenvalues"][1] > 0 else 1
        width_m = width_px * cell_size

        if width_m > max_width_m:
            continue

        # Compute depth (how far below surrounding terrain)
        local_tpi = tpi[rows, cols]
        depth = -np.mean(local_tpi)  # Negative TPI = below surroundings

        if depth < min_depth_m or depth > max_depth_m:
            continue

        # Find endpoints (extremes along major axis)
        coords = np.column_stack([rows, cols])
        centroid = np.array(feat["centroid"])

        # Recalculate for endpoints
        centered = coords - centroid
        cov = np.cov(centered.T)
        _, evecs = np.linalg.eigh(cov)
        evecs = evecs[:, ::-1]  # Sort descending

        projections = centered @ evecs[:, 0]
        start_idx = np.argmin(projections)
        end_idx = np.argmax(projections)

        start_point = (rows[start_idx], cols[start_idx])
        end_point = (rows[end_idx], cols[end_idx])

        # Check curvature as secondary indicator (concave = negative curvature)
        mean_curvature = np.mean(curvature[rows, cols])
        is_concave = mean_curvature < -0.001

        # Confidence based on multiple factors
        confidence = 0.0
        confidence += min(feat["linearity"], 1.0) * 0.25  # Linearity
        confidence += min(depth / 0.2, 1.0) * 0.25  # Depth (optimal ~20cm)
        confidence += min(feat["length"] * cell_size / 20, 1.0) * 0.2  # Length
        confidence += (1 - min(width_m / max_width_m, 1.0)) * 0.15  # Narrowness
        confidence += 0.15 if is_concave else 0.0  # Curvature support

        trail = TrailCandidate(
            pixels=pixels,
            length_m=feat["length"] * cell_size,
            width_m=width_m,
            depth_m=depth,
            orientation_deg=feat["orientation"],
            linearity=feat["linearity"],
            confidence=confidence,
            centroid=feat["centroid"],
            start_point=start_point,
            end_point=end_point,
        )

        trails.append(trail)

    # Sort by confidence
    trails.sort(key=lambda t: t.confidence, reverse=True)

    logger.info(f"Detected {len(trails)} trail candidates")

    return trails


# =============================================================================
# MULTI-FEATURE VISUALIZATION
# =============================================================================


def compute_archaeological_visualization(
    dem: np.ndarray,
    cell_size: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Compute multiple visualization layers used in archaeological LiDAR analysis.

    Returns a dict of layers that can be combined or viewed separately:
    - svf: Sky-View Factor (reveals enclosed features)
    - slope: Slope in degrees
    - curvature_profile: Profile curvature (concave/convex in flow direction)
    - curvature_planform: Planform curvature (convergent/divergent)
    - tpi_fine: Fine-scale TPI (micro-features)
    - tpi_coarse: Coarse-scale TPI (larger features)
    - rem: Relative Elevation Model (deviation from trend)

    These can be combined using various blending modes for visualization.
    """
    logger.info("Computing archaeological visualization layers...")

    # Slope
    dz_dx = np.gradient(dem, cell_size, axis=1)
    dz_dy = np.gradient(dem, cell_size, axis=0)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    # All layers
    layers = {
        "slope": slope,
        "svf": compute_sky_view_factor(dem, cell_size, search_radius=10),
        "curvature_profile": compute_curvature(dem, cell_size, "profile"),
        "curvature_planform": compute_curvature(dem, cell_size, "planform"),
        "tpi_fine": compute_tpi(dem, outer_radius=3),
        "tpi_coarse": compute_tpi(dem, outer_radius=15),
        "rem": compute_relative_elevation(dem, window_size=50),
    }

    logger.info("Visualization layers computed")

    return layers
