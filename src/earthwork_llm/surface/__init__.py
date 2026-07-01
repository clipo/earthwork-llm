"""
Surface reconstruction from LiDAR point clouds.

Includes LAS/LAZ reading (via PDAL), DEM generation, multi-scale geomorphon analysis,
pattern-based feature detection, and subtle feature detection for trails/paths.

Complete pipeline:
    1. LASReader - Read LAS/LAZ files using PDAL, with built-in ground classification
    2. DEMGenerator - Convert point clouds to digital elevation models
       - Infill methods from Pingel's neilpy: springs, FDA, nearest neighbor
       - Roughness restitution from Crema et al. (2019) for texture preservation
    3. GeomorphonAnalyzer - Multi-scale terrain classification (two methods)
    4. FeaturePatternRecognizer - Detect military and terrain features
    5. SubtleFeatures - Trail/path detection using curvature, SVF, TPI analysis

Two Geomorphon Approaches:
    - Direct elevation: Original Jasiewicz & Stepinski method
    - Openness-based: Pingel implementation, better for subtle features

Ternary Patterns:
    The ternary pattern (0-6560) contains more information than 10 geomorphon classes.
    Use compute_multiscale_ternary_patterns() for richer feature representation.

Example:
    >>> from earthwork_llm.surface import LASReader, DEMGenerator, GeomorphonAnalyzer
    >>> # Read LiDAR data (uses PDAL with fallback to laspy)
    >>> reader = LASReader()
    >>> points, meta = reader.read_ground_points("terrain.laz")
    >>> # Generate DEM
    >>> dem_gen = DEMGenerator()
    >>> dem, dem_meta = dem_gen.generate_dem(points)
    >>> # Analyze terrain with openness-based method
    >>> analyzer = GeomorphonAnalyzer()
    >>> geomorphons = analyzer.compute_multiscale_geomorphons_openness(dem)

    >>> # Get full ternary patterns (richer than geomorphon classes)
    >>> from earthwork_llm.surface import compute_multiscale_ternary_patterns
    >>> patterns = compute_multiscale_ternary_patterns(dem, cell_size=0.5)
    >>> for scale, result in patterns.items():
    ...     print(f"{scale}: {len(result.pattern_counts)} unique patterns")

    >>> # Use different infill methods for DEM generation
    >>> # 'springs' (default): Sparse matrix solver, smooth & physically-motivated
    >>> # 'fda': Finite Difference Approximation, very smooth
    >>> # 'nearest': Simple nearest neighbor, fast but blocky
    >>> dem_gen = DEMGenerator(resolution=0.5, infill_method='fda')
    >>> dem, meta = dem_gen.generate_dem(ground_points)
    >>> print(f"DEM filled using: {meta.infill_method}")

    >>> # Use roughness restitution for gentle terrain (Crema et al. 2019)
    >>> # Restores surface texture after smooth inpainting
    >>> dem_gen = DEMGenerator(
    ...     resolution=0.5,
    ...     infill_method='springs',
    ...     roughness_restitution=True,  # Enable texture restoration
    ...     roughness_window=5,          # Window for residual computation
    ...     roughness_search_radius=10   # Radius for sampling residuals
    ... )
    >>> dem, meta = dem_gen.generate_dem(ground_points)
    >>> print(f"Roughness restored: {meta.roughness_restitution}")

    >>> # Detect subtle trails and paths (critical for battlefield analysis)
    >>> from earthwork_llm.surface import detect_trails, compute_archaeological_visualization
    >>> trails = detect_trails(dem, cell_size=0.5, min_length_m=10)
    >>> for trail in trails:
    ...     print(f"Trail: {trail.length_m:.1f}m, depth={trail.depth_m:.2f}m")
    >>>
    >>> # Compute all archaeological visualization layers
    >>> layers = compute_archaeological_visualization(dem, cell_size=0.5)
    >>> print(f"Layers: {list(layers.keys())}")
    # ['slope', 'svf', 'curvature_profile', 'curvature_planform', 'tpi_fine', 'tpi_coarse', 'rem']
"""

from .dem_generator import (  # Point density analysis; Infill methods (from Pingel's neilpy); Roughness restitution (from Crema et al. 2019)
    DEMGenerator,
    DEMMetadata,
    apply_roughness_restitution,
    compute_point_density,
    compute_residual_roughness,
    inpaint_nans_by_fda,
    inpaint_nans_by_springs,
    inpaint_nearest,
    sample_surrounding_residuals,
)
from .feature_patterns import (  # Morphometric analysis
    DetectedFeature,
    FeaturePatternRecognizer,
    FeatureSignature,
    MilitaryFeature,
    classify_feature_shape,
    compute_morphometric_properties,
    filter_by_shape,
)
from .geomorphons import decode_ternary_patterns_batch  # Vectorized batch decode
from .geomorphons import encode_ternary_patterns_batch  # Vectorized batch encode
from .geomorphons import (  # Openness-based functions (Pingel approach); Ternary pattern features (richer than geomorphon classes); Symbolic pattern strings (for LLM training); Multi-scale persistence (for robust feature detection)
    MILITARY_PATTERNS,
    GeomorphonAnalyzer,
    GeomorphonType,
    TernaryPatternResult,
    compute_multiscale_confidence,
    compute_multiscale_ternary_patterns,
    compute_negative_openness,
    compute_openness,
    compute_positive_openness,
    compute_scale_persistence,
    decode_ternary_pattern,
    describe_ternary_pattern,
    encode_ternary_pattern,
    find_pattern_matches,
    find_persistent_features,
    get_pattern_signature_for_feature,
    symbol_to_ternary_code,
    ternary_code_to_symbol,
    ternary_codes_to_symbols,
    ternary_pattern_from_openness,
    terrain_code_to_geomorphon,
)
from .las_reader import classify_ground  # New PDAL-based ground classification
from .las_reader import (
    LASReader,
    PointClassification,
    PointCloudMetadata,
    read_ground_points,
    read_las,
)
from .subtle_features import (  # Curvature analysis for subtle depressions; Sky-View Factor for enclosed feature detection; Topographic Position Index for micro-features; Relative Elevation Model for trend removal; Linear feature detection; Trail detection pipeline; Archaeological visualization (all layers)
    TrailCandidate,
    compute_all_curvatures,
    compute_archaeological_visualization,
    compute_curvature,
    compute_multiscale_tpi,
    compute_relative_elevation,
    compute_sky_view_factor,
    compute_tpi,
    detect_linear_features,
    detect_trails,
)
from .terrain_grammar import (
    Detectability,
    DetectionMethod,
    FeatureCategory,
    FeatureScale,
    GeomorphonSignature,
    TermMapping,
    TerrainFeature,
    TerrainGrammar,
    get_geomorphon_signature,
    get_terrain_grammar,
    resolve_document_term,
)

__all__ = [
    # LAS reading (PDAL-based with laspy fallback)
    "LASReader",
    "PointCloudMetadata",
    "PointClassification",
    "read_las",
    "read_ground_points",
    "classify_ground",  # PDAL ground classification (SMRF/PMF/CSF)
    # DEM generation
    "DEMGenerator",
    "DEMMetadata",
    # Point density analysis
    "compute_point_density",
    # Infill methods (from Pingel's neilpy)
    "inpaint_nans_by_springs",
    "inpaint_nans_by_fda",
    "inpaint_nearest",
    # Roughness restitution (from Crema et al. 2019)
    "compute_residual_roughness",
    "sample_surrounding_residuals",
    "apply_roughness_restitution",
    # Geomorphons
    "GeomorphonAnalyzer",
    "GeomorphonType",
    # Openness-based functions (Pingel approach)
    "compute_openness",
    "compute_positive_openness",
    "compute_negative_openness",
    "ternary_pattern_from_openness",
    "terrain_code_to_geomorphon",
    # Ternary pattern features (richer than geomorphon classes)
    "TernaryPatternResult",
    "decode_ternary_pattern",
    "encode_ternary_pattern",
    "decode_ternary_patterns_batch",
    "encode_ternary_patterns_batch",
    "describe_ternary_pattern",
    "compute_multiscale_ternary_patterns",
    "find_pattern_matches",
    "get_pattern_signature_for_feature",
    "MILITARY_PATTERNS",
    # Symbolic pattern strings (for LLM training)
    "ternary_code_to_symbol",
    "symbol_to_ternary_code",
    "ternary_codes_to_symbols",
    # Multi-scale persistence (for robust feature detection)
    "compute_scale_persistence",
    "compute_multiscale_confidence",
    "find_persistent_features",
    # Feature detection
    "FeaturePatternRecognizer",
    "MilitaryFeature",
    "DetectedFeature",
    "FeatureSignature",
    # Morphometric analysis
    "compute_morphometric_properties",
    "classify_feature_shape",
    "filter_by_shape",
    # Terrain grammar
    "TerrainGrammar",
    "TerrainFeature",
    "FeatureCategory",
    "FeatureScale",
    "DetectionMethod",
    "Detectability",
    "GeomorphonSignature",
    "TermMapping",
    "get_terrain_grammar",
    "resolve_document_term",
    "get_geomorphon_signature",
    # Subtle features (trails, paths, movement corridors)
    "compute_curvature",
    "compute_all_curvatures",
    "compute_sky_view_factor",
    "compute_tpi",
    "compute_multiscale_tpi",
    "compute_relative_elevation",
    "detect_linear_features",
    "TrailCandidate",
    "detect_trails",
    "compute_archaeological_visualization",
]
