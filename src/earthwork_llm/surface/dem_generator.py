"""
DEM Generator for TerraLLM

Converts LiDAR point clouds to Digital Elevation Models (DEMs) using various
interpolation methods. Produces regular grids suitable for geomorphon analysis
and terrain feature extraction.

IMPORTANT: For bare-earth terrain analysis, this generator expects GROUND-CLASSIFIED
points only (LAS classification = 2). Use LASReader.read_ground_points() or
classify_ground() to extract ground points before DEM generation.

For training data, we need bare-ground DEMs to detect terrain features like:
- Foxholes, trenches, craters (micro-features)
- Ridges, valleys, slopes (macro-features)

Vegetation and building points will corrupt the terrain surface.

Infill Methods (from Pingel's neilpy):
- Springs: Sparse matrix approach with spring connections between neighbors.
           Solves least-squares for missing pixels. Physically motivated.
- FDA: Finite Difference Approximation using Laplacian smoothing.
       Produces smooth interpolation across gaps.
- Nearest: Simple nearest neighbor (fast but blocky).
"""

import logging
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.interpolate import NearestNDInterpolator, griddata
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)


@dataclass
class DEMMetadata:
    """Metadata for generated DEM"""

    resolution: float  # Cell size in meters
    bounds: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    shape: Tuple[int, int]  # (height, width)
    crs: Optional[str] = None  # Coordinate reference system
    nodata_value: float = -9999.0
    interpolation_method: str = "idw"
    infill_method: str = "springs"
    roughness_restitution: bool = False  # Whether roughness was restored


# ============================================================================
# Infill Methods (from Pingel's neilpy)
# ============================================================================
# These methods fill NaN/nodata gaps in DEMs using different algorithms.
# Springs and FDA produce smoother, more physically-motivated results than
# simple nearest-neighbor interpolation.
# ============================================================================


def inpaint_nans_by_springs(A: np.ndarray, inplace: bool = False, neighbors: int = 4) -> np.ndarray:
    """
    Fill NaN values using a spring-based sparse matrix solver.

    This method treats each pixel as connected to its neighbors by springs.
    NaN pixels are solved by minimizing the total spring energy (least-squares).
    Produces smooth, physically-motivated interpolation.

    Algorithm from Pingel's neilpy (https://github.com/thomaspingel/neilpy)

    Args:
        A: 2D array with NaN values to fill
        inplace: If True, modify array in place
        neighbors: Number of neighbors (4 or 8)

    Returns:
        Array with NaN values filled
    """
    if not inplace:
        A = A.copy()

    n, m = A.shape
    nm = n * m

    # Find NaN locations
    nan_mask = np.isnan(A)
    nan_indices = np.where(nan_mask.ravel())[0]
    known_indices = np.where(~nan_mask.ravel())[0]

    if len(nan_indices) == 0:
        return A

    if len(known_indices) == 0:
        logger.warning("No valid data to interpolate from")
        return A

    # Build sparse matrix for spring connections
    # Each pixel is connected to its neighbors
    row_list = []
    col_list = []
    data_list = []

    for idx in nan_indices:
        i, j = divmod(idx, m)
        connections = 0

        # 4-connected neighbors
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Add diagonal neighbors for 8-connected
        if neighbors == 8:
            neighbor_offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

        for di, dj in neighbor_offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < m:
                neighbor_idx = ni * m + nj
                row_list.append(idx)
                col_list.append(neighbor_idx)
                data_list.append(-1.0)
                connections += 1

        # Diagonal element (sum of connections)
        row_list.append(idx)
        col_list.append(idx)
        data_list.append(float(connections))

    # Build sparse matrix
    spring_matrix = sparse.csr_matrix((data_list, (row_list, col_list)), shape=(nm, nm))

    # Extract submatrix for NaN locations
    # We need to solve: A_nan * x_nan = -A_known * x_known
    nan_to_idx = {old: new for new, old in enumerate(nan_indices)}
    known_to_idx = {old: new for new, old in enumerate(known_indices)}

    # Build reduced system
    n_nan = len(nan_indices)

    # Left-hand side: connections between NaN pixels
    lhs_row = []
    lhs_col = []
    lhs_data = []

    # Right-hand side: contributions from known pixels
    rhs = np.zeros(n_nan)

    for i, nan_idx in enumerate(nan_indices):
        row_start = spring_matrix.indptr[nan_idx]
        row_end = spring_matrix.indptr[nan_idx + 1]

        for ptr in range(row_start, row_end):
            col_idx = spring_matrix.indices[ptr]
            val = spring_matrix.data[ptr]

            if col_idx in nan_to_idx:
                # Connection to another NaN pixel
                lhs_row.append(i)
                lhs_col.append(nan_to_idx[col_idx])
                lhs_data.append(val)
            elif col_idx in known_to_idx:
                # Connection to known pixel - move to RHS
                rhs[i] -= val * A.ravel()[col_idx]

    lhs = sparse.csr_matrix((lhs_data, (lhs_row, lhs_col)), shape=(n_nan, n_nan))

    # Solve the system
    try:
        x_nan = spsolve(lhs, rhs)
        A.ravel()[nan_indices] = x_nan
    except Exception as e:
        logger.warning(f"Springs solver failed: {e}, falling back to nearest")
        A = inpaint_nearest(A if inplace else A.copy())

    return A


def inpaint_nans_by_fda(A: np.ndarray, fast: bool = True, inplace: bool = False) -> np.ndarray:
    """
    Fill NaN values using Finite Difference Approximation (Laplacian smoothing).

    Uses the discrete Laplacian operator to find values that minimize
    the second derivative across gaps. Produces very smooth results.

    Algorithm from Pingel's neilpy (https://github.com/thomaspingel/neilpy)

    Args:
        A: 2D array with NaN values to fill
        fast: Use faster approximation (recommended)
        inplace: If True, modify array in place

    Returns:
        Array with NaN values filled
    """
    if not inplace:
        A = A.copy()

    n, m = A.shape

    nan_mask = np.isnan(A)
    nan_indices = np.where(nan_mask.ravel())[0]
    known_indices = np.where(~nan_mask.ravel())[0]

    if len(nan_indices) == 0:
        return A

    if len(known_indices) == 0:
        logger.warning("No valid data to interpolate from")
        return A

    # Build Laplacian matrix
    # For each NaN pixel: -4*center + sum(neighbors) = 0
    # This minimizes the discrete Laplacian

    row_list = []
    col_list = []
    data_list = []

    nan_to_idx = {old: new for new, old in enumerate(nan_indices)}
    n_nan = len(nan_indices)
    rhs = np.zeros(n_nan)

    for local_idx, global_idx in enumerate(nan_indices):
        i, j = divmod(global_idx, m)

        # 4-connected Laplacian stencil: [-1, -1, 4, -1, -1]
        center_coeff = 0
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < m:
                neighbor_idx = ni * m + nj
                center_coeff += 1

                if neighbor_idx in nan_to_idx:
                    # Neighbor is also NaN
                    row_list.append(local_idx)
                    col_list.append(nan_to_idx[neighbor_idx])
                    data_list.append(-1.0)
                else:
                    # Neighbor is known - move to RHS
                    rhs[local_idx] += A.ravel()[neighbor_idx]

        # Diagonal element
        row_list.append(local_idx)
        col_list.append(local_idx)
        data_list.append(float(center_coeff))

    lhs = sparse.csr_matrix((data_list, (row_list, col_list)), shape=(n_nan, n_nan))

    # Solve the system
    try:
        x_nan = spsolve(lhs, rhs)
        A.ravel()[nan_indices] = x_nan
    except Exception as e:
        logger.warning(f"FDA solver failed: {e}, falling back to nearest")
        A = inpaint_nearest(A if inplace else A.copy())

    return A


def inpaint_nearest(A: np.ndarray) -> np.ndarray:
    """
    Fill NaN values using simple nearest neighbor interpolation.

    Fast but produces blocky results. Use springs or FDA for smoother output.

    Args:
        A: 2D array with NaN values to fill

    Returns:
        Array with NaN values filled
    """
    nan_mask = np.isnan(A)

    if not nan_mask.any():
        return A

    rows, cols = np.indices(A.shape)
    valid_mask = ~nan_mask

    if not valid_mask.any():
        logger.warning("No valid data to interpolate from")
        return A

    valid_points = np.column_stack([rows[valid_mask].ravel(), cols[valid_mask].ravel()])
    valid_values = A[valid_mask]

    nan_points = np.column_stack([rows[nan_mask].ravel(), cols[nan_mask].ravel()])

    interp = NearestNDInterpolator(valid_points, valid_values)
    A_filled = A.copy()
    A_filled[nan_mask] = interp(nan_points)

    return A_filled


# ============================================================================
# Roughness Restitution (from Crema et al. 2019)
# ============================================================================
# After inpainting, the filled areas are too smooth compared to natural terrain.
# This procedure restores surface texture by sampling residual topography from
# surrounding valid areas and adding it to the interpolated surface.
#
# Reference: Crema et al. (2019) "Can inpainting improve digital terrain analysis?
# Comparing techniques for void filling, surface reconstruction and geomorphometric
# analyses" - Earth Surface Processes and Landforms
# ============================================================================


def compute_residual_roughness(dem: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Compute residual roughness (texture) from a DEM.

    Calculates the difference between the original DEM and a moving-average
    smoothed version. The residuals represent local surface texture that
    would be lost during smooth interpolation.

    Based on Crema et al. (2019) roughness restitution approach.

    Args:
        dem: Input DEM array
        window_size: Size of moving window for smoothing (default: 5)
                     Should be odd number. Larger = more smoothing.

    Returns:
        Residual roughness array (original - smoothed)
    """
    from scipy.ndimage import uniform_filter

    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    # Handle NaN values by temporarily replacing with local mean
    nan_mask = np.isnan(dem)
    if nan_mask.any():
        dem_work = dem.copy()
        # Use nearest neighbor to fill NaN for smoothing
        dem_work = inpaint_nearest(dem_work)
    else:
        dem_work = dem

    # Apply moving average filter
    smoothed = uniform_filter(dem_work.astype(np.float64), size=window_size, mode="nearest")

    # Compute residuals (original - smoothed = texture)
    residuals = dem_work - smoothed

    # Restore NaN positions
    if nan_mask.any():
        residuals[nan_mask] = np.nan

    return residuals


def sample_surrounding_residuals(
    residuals: np.ndarray, void_mask: np.ndarray, search_radius: int = 10, min_samples: int = 50
) -> np.ndarray:
    """
    Sample residual values from areas surrounding voids.

    For each void region, samples residual (texture) values from nearby
    valid terrain that will be used to restore roughness.

    Args:
        residuals: Residual roughness array (from compute_residual_roughness)
        void_mask: Boolean mask where True = void/gap locations
        search_radius: Radius (in pixels) to search for residual samples
        min_samples: Minimum number of residual samples to collect

    Returns:
        Array of sampled residual values
    """
    from scipy.ndimage import binary_dilation

    # Create search zone around voids (dilation)
    struct = np.ones((2 * search_radius + 1, 2 * search_radius + 1))
    search_zone = binary_dilation(void_mask, structure=struct)

    # Valid sample locations: in search zone, not void, and has valid residual
    valid_residuals_mask = ~np.isnan(residuals)
    sample_mask = search_zone & ~void_mask & valid_residuals_mask

    if not sample_mask.any():
        # Fall back to any valid residual
        sample_mask = valid_residuals_mask

    if not sample_mask.any():
        logger.warning("No valid residuals to sample from")
        return np.array([0.0])

    # Extract residual values
    samples = residuals[sample_mask]

    # Ensure we have enough samples (with replacement if needed)
    if len(samples) < min_samples:
        samples = np.random.choice(samples, size=min_samples, replace=True)

    return samples


def apply_roughness_restitution(
    dem_inpainted: np.ndarray,
    original_dem: np.ndarray,
    void_mask: np.ndarray,
    window_size: int = 5,
    search_radius: int = 10,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply roughness restitution to an inpainted DEM.

    Restores surface texture to smoothly-filled void areas by sampling
    residual topography from surrounding terrain and adding it back.

    This addresses the key limitation of heat-diffusion inpainting: while it
    produces excellent trend surfaces, the filled areas are too smooth compared
    to natural terrain texture.

    Based on Crema et al. (2019) roughness restitution procedure.

    Algorithm:
        1. Compute residual roughness from valid surrounding terrain
        2. Sample residual values near void boundaries
        3. Randomly distribute residuals within filled void areas
        4. Add residuals to smooth inpainted surface

    Args:
        dem_inpainted: DEM with voids already filled by inpainting
        original_dem: Original DEM with voids (NaN or nodata)
        void_mask: Boolean mask where True = original void locations
        window_size: Window size for residual computation (default: 5)
        search_radius: Radius for sampling residuals (default: 10 pixels)
        seed: Random seed for reproducibility

    Returns:
        DEM with roughness restored in previously-void areas

    Example:
        >>> # After inpainting fills voids with smooth surface
        >>> dem_filled = inpaint_nans_by_springs(dem_with_voids)
        >>> # Restore texture to the filled areas
        >>> void_mask = np.isnan(dem_with_voids)
        >>> dem_textured = apply_roughness_restitution(
        ...     dem_filled, dem_with_voids, void_mask, window_size=5
        ... )
    """
    if seed is not None:
        np.random.seed(seed)

    # Count void pixels
    n_void = void_mask.sum()
    if n_void == 0:
        return dem_inpainted

    logger.debug(f"Applying roughness restitution to {n_void:,} void pixels")

    # Compute residual roughness from valid areas
    residuals = compute_residual_roughness(original_dem, window_size=window_size)

    # Sample residuals from surrounding areas
    sampled_residuals = sample_surrounding_residuals(
        residuals, void_mask, search_radius=search_radius, min_samples=max(100, n_void // 2)
    )

    # Create output array
    dem_result = dem_inpainted.copy()

    # Randomly select residuals for each void pixel
    void_residuals = np.random.choice(sampled_residuals, size=n_void, replace=True)

    # Add residuals to inpainted values
    dem_result[void_mask] += void_residuals

    # Compute statistics
    residual_mean = np.mean(sampled_residuals)
    residual_std = np.std(sampled_residuals)
    logger.debug(
        f"Roughness restitution: residual mean={residual_mean:.4f}, std={residual_std:.4f}"
    )

    return dem_result


def compute_point_density(points: np.ndarray, resolution: float = 1.0) -> Dict[str, float]:
    """
    Compute point cloud density statistics to guide DEM resolution selection.

    For reliable interpolation:
    - Need at least 1 point per cell on average
    - For micro-features (foxholes), need 4+ points per cell
    - Sparse areas will have holes requiring infill

    Args:
        points: Point cloud array (N, 2) or (N, 3) with x, y coordinates
        resolution: Proposed DEM resolution in meters

    Returns:
        Dict with density statistics:
        - points_per_m2: Average point density
        - points_per_cell: Points per grid cell at given resolution
        - recommended_resolution: Suggested resolution for good coverage
        - coverage_percent: Estimated % of cells with data
    """
    if points.shape[1] >= 2:
        x, y = points[:, 0], points[:, 1]
    else:
        raise ValueError("Points must have at least x, y coordinates")

    # Compute extent
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    area_m2 = x_range * y_range

    n_points = len(points)

    # Density metrics
    points_per_m2 = n_points / area_m2 if area_m2 > 0 else 0
    cell_area = resolution**2
    points_per_cell = points_per_m2 * cell_area

    # Estimate coverage using a grid sampling approach
    n_cells_x = int(x_range / resolution) + 1
    n_cells_y = int(y_range / resolution) + 1
    n_cells = n_cells_x * n_cells_y

    # Quick coverage estimate using histogram
    x_bins = np.linspace(x.min(), x.max(), n_cells_x + 1)
    y_bins = np.linspace(y.min(), y.max(), n_cells_y + 1)
    hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    cells_with_data = np.sum(hist > 0)
    coverage_percent = 100.0 * cells_with_data / n_cells if n_cells > 0 else 0

    # Recommend resolution based on density
    # Target: ~4 points per cell for reliable interpolation
    target_ppc = 4.0
    if points_per_m2 > 0:
        recommended_resolution = np.sqrt(target_ppc / points_per_m2)
        # Clamp to reasonable range
        recommended_resolution = max(0.25, min(5.0, recommended_resolution))
    else:
        recommended_resolution = 1.0

    return {
        "n_points": n_points,
        "area_m2": area_m2,
        "points_per_m2": points_per_m2,
        "points_per_cell": points_per_cell,
        "resolution": resolution,
        "grid_cells": n_cells,
        "cells_with_data": int(cells_with_data),
        "coverage_percent": coverage_percent,
        "recommended_resolution": round(recommended_resolution, 2),
    }


class DEMGenerator:
    """
    Generates Digital Elevation Models from LiDAR point clouds.

    IMPORTANT: For bare-earth terrain analysis (geomorphons, feature detection),
    input point cloud should contain ONLY ground-classified points (class 2).
    Use read_ground_points() or classify_ground() to filter first.

    Supports multiple interpolation methods optimized for different
    point cloud characteristics:
    - IDW: Fast, good for uniform density
    - Natural Neighbor: Best for irregular density
    - TIN: Accurate for complex terrain

    Example:
        >>> from earthwork_llm.surface import LASReader, DEMGenerator
        >>> # CRITICAL: Extract ground points only for bare-earth DEM
        >>> reader = LASReader()
        >>> ground_points, meta = reader.read_ground_points("terrain.laz")
        >>> # Generate DEM from ground points
        >>> generator = DEMGenerator(resolution=0.5)
        >>> dem, dem_meta = generator.generate_dem(ground_points, method="idw")
        >>> print(f"Bare-earth DEM shape: {dem.shape}")
    """

    def __init__(
        self,
        resolution: float = 0.5,
        nodata_value: float = -9999.0,
        fill_holes: bool = True,
        infill_method: Literal["springs", "fda", "nearest"] = "springs",
        roughness_restitution: bool = False,
        roughness_window: int = 5,
        roughness_search_radius: int = 10,
        remove_outliers: bool = True,
        outlier_std_threshold: float = 3.0,
    ):
        """
        Initialize DEM generator.

        Args:
            resolution: Grid cell size in meters (default: 0.5m)
            nodata_value: Value to use for missing data
            fill_holes: Whether to fill small gaps in the DEM
            infill_method: Algorithm for filling holes:
                - 'springs': Sparse matrix solver (smooth, physically-motivated)
                - 'fda': Finite Difference Approximation (very smooth)
                - 'nearest': Simple nearest neighbor (fast but blocky)
            roughness_restitution: Whether to restore surface texture after inpainting.
                Based on Crema et al. (2019). Recommended for gentle terrain where
                smooth inpainting loses important microtopographic detail.
            roughness_window: Window size for computing residual roughness (default: 5).
                Larger values = coarser texture sampling.
            roughness_search_radius: Radius (in pixels) for sampling residuals
                from surrounding terrain (default: 10).
            remove_outliers: Whether to remove elevation outliers
            outlier_std_threshold: Standard deviations for outlier detection
        """
        self.resolution = resolution
        self.nodata_value = nodata_value
        self.fill_holes = fill_holes
        self.infill_method = infill_method
        self.roughness_restitution = roughness_restitution
        self.roughness_window = roughness_window
        self.roughness_search_radius = roughness_search_radius
        self.remove_outliers = remove_outliers
        self.outlier_std_threshold = outlier_std_threshold

    def generate_dem(
        self,
        point_cloud: np.ndarray,
        method: Literal["idw", "natural_neighbor", "tin"] = "idw",
        power: float = 2.0,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        crs: Optional[str] = None,
    ) -> Tuple[np.ndarray, DEMMetadata]:
        """
        Generate DEM from point cloud.

        Args:
            point_cloud: Array of shape (N, 3) with [x, y, z] coordinates
            method: Interpolation method ('idw', 'natural_neighbor', 'tin')
            power: Power parameter for IDW (default: 2.0)
            bounds: Optional (xmin, ymin, xmax, ymax) bounds
            crs: Optional coordinate reference system string

        Returns:
            Tuple of (dem_array, metadata)
        """
        # Validate point cloud shape
        if point_cloud.ndim != 2:
            raise ValueError(f"Point cloud must be 2D array, got {point_cloud.ndim}D")
        if point_cloud.shape[0] == 0:
            raise ValueError("Point cloud is empty")
        if point_cloud.shape[1] != 3:
            raise ValueError(f"Point cloud must have shape (N, 3), got {point_cloud.shape}")

        # Check for invalid values
        n_nan = np.sum(np.isnan(point_cloud))
        n_inf = np.sum(np.isinf(point_cloud))
        if n_nan > 0:
            logger.warning(f"Point cloud contains {n_nan} NaN values, filtering...")
            valid_rows = ~np.any(np.isnan(point_cloud), axis=1)
            point_cloud = point_cloud[valid_rows]
            if len(point_cloud) == 0:
                raise ValueError("All points contain NaN values")
        if n_inf > 0:
            logger.warning(f"Point cloud contains {n_inf} Inf values, filtering...")
            valid_rows = ~np.any(np.isinf(point_cloud), axis=1)
            point_cloud = point_cloud[valid_rows]
            if len(point_cloud) == 0:
                raise ValueError("All points contain Inf values")

        logger.info(f"Generating DEM from {len(point_cloud):,} points using {method}")

        # Extract coordinates
        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]

        # Remove outliers if requested
        if self.remove_outliers:
            x, y, z = self._remove_outliers(x, y, z)
            logger.info(f"After outlier removal: {len(x):,} points")

        # Determine grid bounds
        if bounds is None:
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
        else:
            xmin, ymin, xmax, ymax = bounds

        # Create regular grid
        x_grid = np.arange(xmin, xmax + self.resolution, self.resolution)
        y_grid = np.arange(ymin, ymax + self.resolution, self.resolution)
        xi, yi = np.meshgrid(x_grid, y_grid)

        # Interpolate elevations
        logger.info(f"Interpolating to {xi.shape} grid")
        zi = self._interpolate(x, y, z, xi, yi, method, power)

        # Fill holes if requested
        if self.fill_holes:
            zi = self._fill_holes(
                zi,
                method=self.infill_method,
                apply_roughness=self.roughness_restitution,
                roughness_window=self.roughness_window,
                roughness_search_radius=self.roughness_search_radius,
            )

        # Create metadata
        metadata = DEMMetadata(
            resolution=self.resolution,
            bounds=(xmin, ymin, xmax, ymax),
            shape=zi.shape,
            crs=crs,
            nodata_value=self.nodata_value,
            interpolation_method=method,
            infill_method=self.infill_method if self.fill_holes else "none",
            roughness_restitution=self.roughness_restitution if self.fill_holes else False,
        )

        logger.info(f"DEM generated: {zi.shape} ({zi.shape[0]*zi.shape[1]:,} cells)")

        return zi, metadata

    def _remove_outliers(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove elevation outliers using z-score method"""
        z_mean = np.mean(z)
        z_std = np.std(z)
        z_scores = np.abs((z - z_mean) / z_std)
        mask = z_scores < self.outlier_std_threshold

        n_removed = (~mask).sum()
        if n_removed > 0:
            logger.debug(f"Removed {n_removed:,} outliers ({100*n_removed/len(z):.1f}%)")

        return x[mask], y[mask], z[mask]

    def _interpolate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        xi: np.ndarray,
        yi: np.ndarray,
        method: str,
        power: float,
    ) -> np.ndarray:
        """Perform interpolation using specified method"""
        points = np.column_stack([x, y])

        if method == "idw":
            # Inverse Distance Weighting
            zi = self._idw_interpolation(points, z, xi, yi, power)

        elif method == "natural_neighbor":
            # Natural neighbor interpolation (linear)
            zi = griddata(points, z, (xi, yi), method="linear", fill_value=self.nodata_value)

        elif method == "tin":
            # Triangulated Irregular Network (uses Delaunay triangulation)
            zi = griddata(points, z, (xi, yi), method="cubic", fill_value=self.nodata_value)
            # Fallback to linear for areas where cubic fails
            nan_mask = np.isnan(zi)
            if nan_mask.any():
                zi[nan_mask] = griddata(
                    points,
                    z,
                    (xi[nan_mask], yi[nan_mask]),
                    method="linear",
                    fill_value=self.nodata_value,
                )
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        return zi

    def _idw_interpolation(
        self,
        points: np.ndarray,
        values: np.ndarray,
        xi: np.ndarray,
        yi: np.ndarray,
        power: float,
        search_radius: Optional[float] = None,
        k_neighbors: int = 12,
    ) -> np.ndarray:
        """
        Inverse Distance Weighting interpolation using KDTree for fast queries.

        Uses scipy.spatial.cKDTree for O(log n) nearest neighbor queries instead
        of O(n) brute force distance calculations. This enables high-resolution
        DEM generation (0.5m) for micro-feature detection.

        Args:
            points: Source point coordinates (N, 2)
            values: Source point elevations (N,)
            xi, yi: Target grid coordinates
            power: IDW power parameter (typically 2.0)
            search_radius: Maximum search distance (default: 10 * resolution)
            k_neighbors: Number of neighbors to use for interpolation (default: 12)

        Returns:
            Interpolated elevation grid
        """
        from scipy.spatial import cKDTree

        # For large datasets, use a reasonable search radius
        if search_radius is None:
            search_radius = self.resolution * 10  # 10 cells radius

        # Build KDTree for fast spatial queries - O(n log n) build time
        logger.info(f"Building KDTree for {len(points):,} points...")
        tree = cKDTree(points)

        # Flatten grid
        xi_flat = xi.flatten()
        yi_flat = yi.flatten()
        grid_points = np.column_stack([xi_flat, yi_flat])
        n_grid = len(xi_flat)

        logger.info(f"Interpolating {n_grid:,} grid cells using KDTree IDW...")

        # Query all grid points at once for k nearest neighbors
        # This is MUCH faster than querying one at a time
        distances, indices = tree.query(
            grid_points,
            k=min(k_neighbors, len(points)),
            distance_upper_bound=search_radius,
            workers=-1,  # Use all CPU cores
        )

        # Handle case where k=1 returns 1D arrays
        if distances.ndim == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        # Initialize output
        zi_flat = np.full(n_grid, self.nodata_value, dtype=np.float64)

        # Vectorized IDW calculation
        # Valid points are those with at least one neighbor within search radius
        # cKDTree returns inf for points beyond distance_upper_bound
        valid_mask = np.isfinite(distances[:, 0])

        if valid_mask.any():
            # For valid grid points, compute IDW weights
            valid_distances = distances[valid_mask]
            valid_indices = indices[valid_mask]

            # Handle exact matches (distance ~= 0)
            exact_match = valid_distances[:, 0] < 1e-10
            if exact_match.any():
                zi_flat[np.where(valid_mask)[0][exact_match]] = values[
                    valid_indices[exact_match, 0]
                ]

            # For non-exact matches, compute weighted average
            non_exact = ~exact_match
            if non_exact.any():
                ne_distances = valid_distances[non_exact]
                ne_indices = valid_indices[non_exact]

                # Mask out invalid neighbors (beyond search radius)
                # cKDTree returns len(points) for no match, mark these as invalid
                out_of_bounds = ne_indices >= len(values)
                neighbor_valid = np.isfinite(ne_distances) & ~out_of_bounds

                # Compute weights: 1 / d^power
                # Add small epsilon to avoid division by zero
                weights = np.where(neighbor_valid, 1.0 / (ne_distances**power + 1e-10), 0.0)

                # Get neighbor values safely
                # Replace out-of-bounds indices with 0 (value won't be used due to weight=0)
                safe_indices = np.where(out_of_bounds, 0, ne_indices)
                neighbor_values = values[safe_indices]

                # Weighted average
                weighted_sum = np.sum(weights * neighbor_values * neighbor_valid, axis=1)
                weight_sum = np.sum(weights * neighbor_valid, axis=1)

                # Avoid division by zero
                valid_weights = weight_sum > 0
                result_indices = np.where(valid_mask)[0][non_exact][valid_weights]
                zi_flat[result_indices] = weighted_sum[valid_weights] / weight_sum[valid_weights]

        # Count results
        n_valid = np.sum(zi_flat != self.nodata_value)
        n_nodata = n_grid - n_valid
        logger.info(
            f"IDW complete: {n_valid:,} valid cells, {n_nodata:,} nodata ({100*n_nodata/n_grid:.1f}%)"
        )

        return zi_flat.reshape(xi.shape)

    def _fill_holes(
        self,
        dem: np.ndarray,
        method: Literal["springs", "fda", "nearest"] = "springs",
        apply_roughness: bool = False,
        roughness_window: int = 5,
        roughness_search_radius: int = 10,
    ) -> np.ndarray:
        """
        Fill holes in DEM using specified infill algorithm.

        Available methods (from Pingel's neilpy):
        - 'springs': Sparse matrix solver treating pixels as connected by springs.
                     Produces smooth, physically-motivated interpolation.
        - 'fda': Finite Difference Approximation using Laplacian smoothing.
                 Produces very smooth results, minimizes second derivative.
        - 'nearest': Simple nearest neighbor (fast but blocky).

        Optional roughness restitution (Crema et al. 2019):
        After smooth inpainting, can restore surface texture by sampling
        residuals from surrounding terrain and adding to filled areas.

        Args:
            dem: DEM array with nodata values
            method: Infill algorithm to use ('springs', 'fda', 'nearest')
            apply_roughness: Whether to apply roughness restitution after inpainting
            roughness_window: Window size for residual computation
            roughness_search_radius: Search radius for sampling residuals

        Returns:
            DEM with holes filled (and optionally texture restored)
        """
        # Find nodata cells and convert to NaN for infill functions
        nodata_mask = (dem == self.nodata_value) | np.isnan(dem)

        if not nodata_mask.any():
            return dem

        n_holes = nodata_mask.sum()
        valid_count = (~nodata_mask).sum()
        total_cells = dem.size
        hole_percent = 100.0 * n_holes / total_cells

        logger.info(f"Filling {n_holes:,} holes ({hole_percent:.1f}%) using {method} method")

        if valid_count == 0:
            logger.warning("No valid data to interpolate from")
            return dem

        # Create a copy with NaN for nodata (infill functions expect NaN)
        dem_work = dem.astype(np.float64).copy()
        dem_work[nodata_mask] = np.nan

        # Keep original for roughness restitution
        dem_original = dem_work.copy() if apply_roughness else None

        # Apply selected infill method
        if method == "springs":
            dem_filled = inpaint_nans_by_springs(dem_work, neighbors=4)
        elif method == "fda":
            dem_filled = inpaint_nans_by_fda(dem_work)
        elif method == "nearest":
            dem_filled = inpaint_nearest(dem_work)
        else:
            logger.warning(f"Unknown infill method '{method}', using springs")
            dem_filled = inpaint_nans_by_springs(dem_work, neighbors=4)

        logger.debug(f"Filled {n_holes:,} cells using {method}")

        # Apply roughness restitution if requested (Crema et al. 2019)
        if apply_roughness and dem_original is not None:
            logger.info("Applying roughness restitution to restore surface texture")
            dem_filled = apply_roughness_restitution(
                dem_inpainted=dem_filled,
                original_dem=dem_original,
                void_mask=nodata_mask,
                window_size=roughness_window,
                search_radius=roughness_search_radius,
            )

        return dem_filled

    def save_dem(
        self, dem: np.ndarray, metadata: DEMMetadata, output_path: str, format: str = "GTiff"
    ) -> None:
        """
        Save DEM to file using rasterio.

        Args:
            dem: DEM array
            metadata: DEM metadata
            output_path: Output file path
            format: GDAL format (default: GeoTiff)
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            logger.error("rasterio not installed. Cannot save DEM.")
            logger.info("Install with: pip install rasterio")
            return

        # Create affine transform
        transform = from_bounds(
            metadata.bounds[0],
            metadata.bounds[1],
            metadata.bounds[2],
            metadata.bounds[3],
            metadata.shape[1],
            metadata.shape[0],
        )

        # Write DEM
        with rasterio.open(
            output_path,
            "w",
            driver=format,
            height=metadata.shape[0],
            width=metadata.shape[1],
            count=1,
            dtype=dem.dtype,
            crs=metadata.crs,
            transform=transform,
            nodata=metadata.nodata_value,
        ) as dst:
            dst.write(dem, 1)

        logger.info(f"DEM saved to {output_path}")
