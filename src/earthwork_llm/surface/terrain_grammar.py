"""
Terrain Grammar for TerraLLM

Comprehensive geomorphon-based grammar for terrain feature recognition,
historical document parsing, and natural language understanding.

Defines:
- Complete terrain feature vocabulary (topographic, military, water, coastal, etc.)
- Multi-scale geomorphon signatures for each feature type
- Synonym mappings for historical document parsing (WW2 military terminology)
- Feature category hierarchies for semantic understanding

This grammar bridges:
- LiDAR-derived terrain features (geomorphons)
- Military terminology (KOCOA, defensive positions)
- Historical documents (after-action reports, hand-drawn maps)
- Natural language queries ("where are the foxholes?", "show me ridges")
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from .geomorphons import GeomorphonType

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE CATEGORIES
# =============================================================================


class FeatureCategory(Enum):
    """Top-level feature categories"""

    MILITARY = "military"
    TOPOGRAPHIC_ELEVATED = "topographic_elevated"
    TOPOGRAPHIC_DEPRESSION = "topographic_depression"
    WATER = "water"
    COASTAL = "coastal"
    TRANSPORTATION = "transportation"
    VEGETATION = "vegetation"
    MANMADE = "manmade"
    EARTHWORK = "earthwork"
    BATTLE_DAMAGE = "battle_damage"


class FeatureScale(Enum):
    """Scale at which features are typically detectable"""

    MICRO = "micro"  # 0.5-5m (geomorphon scale: 2m)
    MESO = "meso"  # 5-25m (geomorphon scale: 5m, 10m)
    MACRO = "macro"  # 25-100m+ (geomorphon scale: 10m, 25m)
    MULTI = "multi"  # Detectable across multiple scales


class DetectionMethod(Enum):
    """
    Data sources required to detect a feature.

    Features may require one or more detection methods. Geomorphons alone
    cannot detect all feature types - multimodal data is essential.
    """

    GEOMORPHON = "geomorphon"  # Detectable from terrain morphology alone
    GEOMORPHON_PLUS_CONTEXT = "geomorphon_plus_context"  # Needs geomorphon + tactical reasoning
    VEGETATION_RETURNS = "vegetation_returns"  # Requires full LiDAR (not just ground)
    INTENSITY = "intensity"  # Requires LiDAR intensity values
    RGB_IMAGERY = "rgb_imagery"  # Requires color imagery
    FLOW_ACCUMULATION = "flow_accumulation"  # Requires hydrological analysis
    ASPECT_SLOPE = "aspect_slope"  # Requires slope/aspect derivatives
    DOCUMENT_ONLY = "document_only"  # Only identifiable from historical documents
    MULTIMODAL = "multimodal"  # Requires multiple data sources combined


class Detectability(Enum):
    """How reliably a feature can be detected from LiDAR-derived data"""

    HIGH = "high"  # Strong, distinct signature - high confidence detection
    MEDIUM = "medium"  # Detectable but may have false positives
    LOW = "low"  # Weak signature, requires supporting evidence
    CONTEXTUAL = "contextual"  # Requires tactical/historical context to identify
    NOT_DETECTABLE = "not_detectable"  # Cannot be detected from LiDAR morphology


# =============================================================================
# TERRAIN FEATURE TYPES
# =============================================================================


class TerrainFeature(Enum):
    """
    Complete terrain feature vocabulary.

    Organized by category and mapped to geomorphon signatures.
    Each feature can be detected from LiDAR and/or mentioned in documents.
    """

    # ----- MILITARY: Defensive Positions -----
    FOXHOLE = "foxhole"
    SLIT_TRENCH = "slit_trench"
    SHELL_SCRAPE = "shell_scrape"
    SPIDER_HOLE = "spider_hole"
    RIFLE_PIT = "rifle_pit"
    MACHINE_GUN_NEST = "machine_gun_nest"
    MORTAR_PIT = "mortar_pit"
    ARTILLERY_POSITION = "artillery_position"
    OBSERVATION_BUNKER = "observation_bunker"
    AA_POSITION = "antiaircraft_position"

    # ----- MILITARY: Fortifications -----
    BUNKER = "bunker"
    PILLBOX = "pillbox"
    BLOCKHOUSE = "blockhouse"
    DUGOUT = "dugout"
    CASEMATE = "casemate"
    TOBRUK = "tobruk"

    # ----- MILITARY: Linear Defenses -----
    TRENCH = "trench"
    COMMUNICATION_TRENCH = "communication_trench"
    FIRE_TRENCH = "fire_trench"
    SUPPORT_TRENCH = "support_trench"
    RESERVE_TRENCH = "reserve_trench"
    SAP = "sap"

    # ----- MILITARY: Field Works -----
    BERM = "berm"
    PARAPET = "parapet"
    REVETMENT = "revetment"
    EMPLACEMENT = "emplacement"
    STRONGPOINT = "strongpoint"
    DEFENSIVE_LINE = "defensive_line"

    # ----- MILITARY: Tactical Terrain -----
    HIGH_GROUND = "high_ground"
    COMMANDING_POSITION = "commanding_position"
    CHOKEPOINT = "chokepoint"
    OBSERVATION_POST = "observation_post"
    VANTAGE_POINT = "vantage_point"
    REVERSE_SLOPE = "reverse_slope"
    DEAD_GROUND = "dead_ground"
    DEFILADE = "defilade"
    HULL_DOWN = "hull_down"
    AVENUE_OF_APPROACH = "avenue_of_approach"
    INFILTRATION_ROUTE = "infiltration_route"

    # ----- TOPOGRAPHIC: Elevated -----
    MOUNTAIN = "mountain"
    HILL = "hill"
    KNOLL = "knoll"
    HILLOCK = "hillock"
    MOUND = "mound"
    HUMMOCK = "hummock"
    RIDGE = "ridge"
    RIDGELINE = "ridgeline"
    SPUR = "spur"
    FINGER = "finger"
    HOGBACK = "hogback"
    SADDLE = "saddle"
    COL = "col"
    CREST = "crest"
    MILITARY_CREST = "military_crest"
    SHOULDER = "shoulder"
    BENCH = "bench"
    TERRACE = "terrace"
    SCARP = "scarp"
    ESCARPMENT = "escarpment"
    BLUFF = "bluff"

    # ----- TOPOGRAPHIC: Depressions -----
    VALLEY = "valley"
    CANYON = "canyon"
    GORGE = "gorge"
    RAVINE = "ravine"
    GULLY = "gully"
    DRAW = "draw"
    SWALE = "swale"
    COULEE = "coulee"
    DEPRESSION = "depression"
    HOLLOW = "hollow"
    BASIN = "basin"
    SINKHOLE = "sinkhole"
    CRATER = "crater"
    PIT = "pit"
    POTHOLE = "pothole"
    KETTLE = "kettle"
    PASS = "pass"
    GAP = "gap"
    NOTCH = "notch"
    DEFILE = "defile"
    CORRIDOR = "corridor"

    # ----- WATER: Flowing -----
    RIVER = "river"
    STREAM = "stream"
    CREEK = "creek"
    BROOK = "brook"
    TRIBUTARY = "tributary"
    CONFLUENCE = "confluence"
    HEADWATERS = "headwaters"

    # ----- WATER: Standing -----
    LAKE = "lake"
    POND = "pond"
    POOL = "pool"
    RESERVOIR = "reservoir"
    TARN = "tarn"

    # ----- WATER: Wetlands -----
    SWAMP = "swamp"
    MARSH = "marsh"
    BOG = "bog"
    FEN = "fen"
    WETLAND = "wetland"
    FLOODPLAIN = "floodplain"
    BOTTOMLAND = "bottomland"

    # ----- WATER: Crossings -----
    FORD = "ford"
    BRIDGE = "bridge"
    CULVERT = "culvert"
    CAUSEWAY = "causeway"
    FERRY_SITE = "ferry_site"
    STREAM_CROSSING = "stream_crossing"

    # ----- COASTAL -----
    BEACH = "beach"
    SHORELINE = "shoreline"
    STRAND = "strand"
    DUNE = "dune"
    BEACH_BERM = "beach_berm"
    BACKSHORE = "backshore"
    FORESHORE = "foreshore"
    CLIFF = "cliff"
    HEADLAND = "headland"
    PROMONTORY = "promontory"
    CAPE = "cape"
    POINT = "point"
    BAY = "bay"
    COVE = "cove"
    INLET = "inlet"
    HARBOR = "harbor"
    ESTUARY = "estuary"
    LAGOON = "lagoon"

    # ----- TRANSPORTATION -----
    HIGHWAY = "highway"
    ROAD = "road"
    LANE = "lane"
    TRACK = "track"
    TRAIL = "trail"
    PATH = "path"
    FOOTPATH = "footpath"
    FIREBREAK = "firebreak"
    ROAD_CUT = "road_cut"
    ROAD_FILL = "road_fill"
    EMBANKMENT = "embankment"
    SWITCHBACK = "switchback"
    HAIRPIN = "hairpin"
    RAILWAY = "railway"
    RAIL_BED = "rail_bed"
    RAIL_CUT = "rail_cut"

    # ----- VEGETATION -----
    FOREST = "forest"
    WOODS = "woods"
    GROVE = "grove"
    COPSE = "copse"
    THICKET = "thicket"
    WOODLOT = "woodlot"
    JUNGLE = "jungle"
    TREE_LINE = "tree_line"
    TIMBERLINE = "timberline"
    HEDGEROW = "hedgerow"
    WINDBREAK = "windbreak"
    CLEARING = "clearing"
    GLADE = "glade"
    FIELD = "field"
    ORCHARD = "orchard"
    VINEYARD = "vineyard"
    PASTURE = "pasture"
    MEADOW = "meadow"

    # ----- MANMADE -----
    BUILDING = "building"
    RUIN = "ruin"
    FOUNDATION = "foundation"
    WALL = "wall"
    FENCE_LINE = "fence_line"
    TOWER = "tower"
    LEVEE = "levee"
    DIKE = "dike"
    DAM = "dam"
    SPOIL_PILE = "spoil_pile"
    DITCH = "ditch"
    CANAL = "canal"
    QUARRY = "quarry"
    MINE = "mine"
    CELLAR_HOLE = "cellar_hole"
    CANAL_DREDGING = "canal_dredging"
    ROAD_WORK = "road_work"

    # ----- PRE-EUROPEAN EARTHWORKS -----
    PRE_EUROPEAN_MOUND = "pre_european_mound"
    PLATFORM_MOUND = "platform_mound"
    CONICAL_MOUND = "conical_mound"
    EFFIGY_MOUND = "effigy_mound"
    PRE_EUROPEAN_ENCLOSURE = "pre_european_enclosure"
    EARTHWORK_EMBANKMENT = "earthwork_embankment"

    # ----- BATTLE DAMAGE -----
    SHELL_CRATER = "shell_crater"
    BOMB_CRATER = "bomb_crater"
    MORTAR_CRATER = "mortar_crater"
    GRENADE_CRATER = "grenade_crater"
    DEBRIS_FIELD = "debris_field"
    BURNED_AREA = "burned_area"
    COLLAPSED_STRUCTURE = "collapsed_structure"
    VEHICLE_WRECK = "vehicle_wreck"
    CRATER_CLUSTER = "crater_cluster"


# =============================================================================
# GEOMORPHON SIGNATURES
# =============================================================================


@dataclass
class GeomorphonSignature:
    """
    Multi-scale geomorphon signature for a terrain feature.

    Defines which geomorphon types are expected at each analysis scale,
    along with geometric and contextual constraints.

    IMPORTANT: Not all features can be detected from geomorphons alone!
    Check the `detection_methods` and `detectability` fields to understand
    what data sources are required for reliable detection.
    """

    feature: TerrainFeature
    category: FeatureCategory
    scale: FeatureScale

    # Detection capability - CRITICAL for understanding what's possible
    detection_methods: Set[DetectionMethod] = field(
        default_factory=lambda: {DetectionMethod.GEOMORPHON}
    )
    detectability: Detectability = Detectability.MEDIUM

    # Geomorphon types at each scale (primary patterns)
    # NOTE: Empty sets mean this feature is NOT detectable via geomorphons at this scale
    micro_2m: Set[GeomorphonType] = field(default_factory=set)
    meso_5m: Set[GeomorphonType] = field(default_factory=set)
    local_10m: Set[GeomorphonType] = field(default_factory=set)
    regional_25m: Set[GeomorphonType] = field(default_factory=set)

    # Geometric constraints
    min_size_m: float = 0.0
    max_size_m: float = float("inf")
    min_depth_m: Optional[float] = None
    max_depth_m: Optional[float] = None

    # Shape descriptors
    is_linear: bool = False
    is_circular: bool = False
    aspect_ratio: Optional[Tuple[float, float]] = None  # (min, max)

    # Contextual patterns
    requires_high_ground: bool = False
    requires_water: bool = False
    requires_vegetation: bool = False

    description: str = ""


# =============================================================================
# DOCUMENT TERM MAPPINGS
# =============================================================================


@dataclass
class TermMapping:
    """
    Maps document terminology to terrain features.

    Supports:
    - Synonyms (multiple terms for same feature)
    - Historical terminology (WW2, Japanese, German)
    - Abbreviations (OP, LP, CP, MLR, etc.)
    """

    canonical: TerrainFeature
    synonyms: List[str] = field(default_factory=list)
    abbreviations: List[str] = field(default_factory=list)
    japanese_terms: List[str] = field(default_factory=list)
    german_terms: List[str] = field(default_factory=list)
    historical_terms: List[str] = field(default_factory=list)


# =============================================================================
# TERRAIN GRAMMAR CLASS
# =============================================================================


class TerrainGrammar:
    """
    Comprehensive terrain grammar for feature detection and document parsing.

    Provides:
    - Feature signatures for geomorphon pattern matching
    - Term mappings for historical document parsing
    - Category hierarchies for semantic queries
    - Synonym resolution for natural language understanding

    Example:
        >>> grammar = TerrainGrammar()
        >>> # Get signature for foxhole detection
        >>> sig = grammar.get_signature(TerrainFeature.FOXHOLE)
        >>> # Find what feature "Schützenloch" refers to
        >>> feature = grammar.resolve_term("Schützenloch")
        >>> # Get all military defensive features
        >>> features = grammar.get_features_by_category(FeatureCategory.MILITARY)
    """

    def __init__(self):
        """Initialize terrain grammar with predefined signatures and mappings"""
        self.signatures = self._build_signatures()
        self.term_mappings = self._build_term_mappings()
        self._term_index = self._build_term_index()

        logger.info(
            f"Initialized TerrainGrammar with {len(self.signatures)} signatures, "
            f"{len(self.term_mappings)} term mappings"
        )

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def get_signature(self, feature: TerrainFeature) -> Optional[GeomorphonSignature]:
        """Get geomorphon signature for a terrain feature"""
        return self.signatures.get(feature)

    def get_signatures_by_category(self, category: FeatureCategory) -> List[GeomorphonSignature]:
        """Get all signatures for a feature category"""
        return [sig for sig in self.signatures.values() if sig.category == category]

    def get_signatures_by_scale(self, scale: FeatureScale) -> List[GeomorphonSignature]:
        """Get all signatures detectable at a given scale"""
        return [
            sig
            for sig in self.signatures.values()
            if sig.scale == scale or sig.scale == FeatureScale.MULTI
        ]

    def resolve_term(self, term: str) -> Optional[TerrainFeature]:
        """
        Resolve a document term to its canonical terrain feature.

        Handles synonyms, abbreviations, and foreign terms.
        Case-insensitive matching.
        """
        term_lower = term.lower().strip()
        return self._term_index.get(term_lower)

    def get_all_terms(self, feature: TerrainFeature) -> List[str]:
        """Get all known terms for a terrain feature"""
        mapping = self.term_mappings.get(feature)
        if not mapping:
            return [feature.value]

        terms = [feature.value] + mapping.synonyms + mapping.abbreviations
        terms += mapping.japanese_terms + mapping.german_terms
        terms += mapping.historical_terms
        return terms

    def get_detectability_summary(self) -> Dict[str, List[TerrainFeature]]:
        """
        Summarize features by their detectability from geomorphons.

        Returns:
            Dict mapping detectability level to list of features:
            - 'high': Reliably detected from geomorphons alone
            - 'medium': Detectable but may have false positives
            - 'low': Weak signature, needs supporting evidence
            - 'contextual': Requires tactical/historical context
            - 'not_detectable': Cannot be detected from LiDAR morphology
        """
        summary = {"high": [], "medium": [], "low": [], "contextual": [], "not_detectable": []}

        for feature, sig in self.signatures.items():
            level = sig.detectability.value
            summary[level].append(feature)

        return summary

    def get_features_requiring_multimodal(self) -> Dict[str, List[TerrainFeature]]:
        """
        Get features that require additional data sources beyond geomorphons.

        Returns:
            Dict mapping data source to features requiring it
        """
        multimodal = {}

        for feature, sig in self.signatures.items():
            for method in sig.detection_methods:
                if method != DetectionMethod.GEOMORPHON:
                    if method.value not in multimodal:
                        multimodal[method.value] = []
                    multimodal[method.value].append(feature)

        return multimodal

    def get_geomorphon_only_features(self) -> List[TerrainFeature]:
        """
        Get features that CAN be reliably detected from geomorphons alone.

        These are the features where the geomorphon grammar is sufficient.
        """
        return [
            feature
            for feature, sig in self.signatures.items()
            if (
                sig.detectability in {Detectability.HIGH, Detectability.MEDIUM}
                and DetectionMethod.GEOMORPHON in sig.detection_methods
                and len(sig.detection_methods) == 1
            )
        ]

    def get_features_for_geomorphon(
        self, geomorphon: GeomorphonType, scale: str = "2m"
    ) -> List[TerrainFeature]:
        """
        Get features that match a geomorphon type at specified scale.

        Args:
            geomorphon: GeomorphonType to match
            scale: Scale string ("2m", "5m", "10m", "25m")

        Returns:
            List of matching TerrainFeature types
        """
        matches = []
        for feature, sig in self.signatures.items():
            scale_set = getattr(
                sig, f"{scale.replace('m', '')}m" if scale != "2m" else "micro_2m", set()
            )
            if scale == "2m":
                scale_set = sig.micro_2m
            elif scale == "5m":
                scale_set = sig.meso_5m
            elif scale == "10m":
                scale_set = sig.local_10m
            elif scale == "25m":
                scale_set = sig.regional_25m

            if geomorphon in scale_set:
                matches.append(feature)

        return matches

    def get_kocoa_features(self) -> Dict[str, List[TerrainFeature]]:
        """
        Get features organized by KOCOA/OAKOC military terrain analysis factors.

        K - Key terrain
        O - Observation and fields of fire
        C - Cover and concealment
        O - Obstacles
        A - Avenues of approach
        """
        return {
            "key_terrain": [
                TerrainFeature.HIGH_GROUND,
                TerrainFeature.COMMANDING_POSITION,
                TerrainFeature.CHOKEPOINT,
                TerrainFeature.RIDGE,
                TerrainFeature.HILL,
                TerrainFeature.PASS,
                TerrainFeature.BRIDGE,
                TerrainFeature.FORD,
            ],
            "observation_fields_of_fire": [
                TerrainFeature.OBSERVATION_POST,
                TerrainFeature.VANTAGE_POINT,
                TerrainFeature.RIDGE,
                TerrainFeature.MILITARY_CREST,
                TerrainFeature.HIGH_GROUND,
            ],
            "cover_concealment": [
                TerrainFeature.DEFILADE,
                TerrainFeature.HULL_DOWN,
                TerrainFeature.REVERSE_SLOPE,
                TerrainFeature.DEAD_GROUND,
                TerrainFeature.RAVINE,
                TerrainFeature.HOLLOW,
                (
                    TerrainFeature.SUNKEN_ROAD
                    if hasattr(TerrainFeature, "SUNKEN_ROAD")
                    else TerrainFeature.ROAD_CUT
                ),
            ],
            "obstacles": [
                TerrainFeature.RIVER,
                TerrainFeature.STREAM,
                TerrainFeature.CLIFF,
                TerrainFeature.ESCARPMENT,
                TerrainFeature.GORGE,
                TerrainFeature.SWAMP,
                TerrainFeature.MARSH,
            ],
            "avenues_of_approach": [
                TerrainFeature.CORRIDOR,
                TerrainFeature.VALLEY,
                TerrainFeature.DRAW,
                TerrainFeature.ROAD,
                TerrainFeature.TRAIL,
                TerrainFeature.INFILTRATION_ROUTE,
                TerrainFeature.AVENUE_OF_APPROACH,
            ],
        }

    # -------------------------------------------------------------------------
    # SIGNATURE DEFINITIONS
    # -------------------------------------------------------------------------

    def _build_signatures(self) -> Dict[TerrainFeature, GeomorphonSignature]:
        """Build complete signature database"""
        signatures = {}

        # ----- MILITARY: Individual Fighting Positions -----
        # HIGH detectability - strong geomorphon signature
        signatures[TerrainFeature.FOXHOLE] = GeomorphonSignature(
            feature=TerrainFeature.FOXHOLE,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MICRO,
            detection_methods={DetectionMethod.GEOMORPHON},
            detectability=Detectability.HIGH,  # Distinct PIT pattern
            micro_2m={GeomorphonType.PIT, GeomorphonType.HOLLOW},
            meso_5m={GeomorphonType.PIT, GeomorphonType.HOLLOW, GeomorphonType.SHOULDER},
            local_10m={GeomorphonType.FLAT, GeomorphonType.SLOPE},
            min_size_m=1.0,
            max_size_m=2.5,
            min_depth_m=0.8,
            max_depth_m=1.8,
            is_circular=True,
            requires_high_ground=True,
            description="Circular fighting position 1-2.5m diameter, 0.8-1.8m deep",
        )

        # LOW detectability - may be too degraded after 80 years
        signatures[TerrainFeature.SLIT_TRENCH] = GeomorphonSignature(
            feature=TerrainFeature.SLIT_TRENCH,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MICRO,
            detection_methods={DetectionMethod.GEOMORPHON},
            detectability=Detectability.LOW,  # Often filled in over time
            micro_2m={GeomorphonType.VALLEY, GeomorphonType.HOLLOW},
            meso_5m={GeomorphonType.HOLLOW, GeomorphonType.FLAT},
            local_10m={GeomorphonType.FLAT, GeomorphonType.SLOPE},
            min_size_m=0.5,
            max_size_m=2.0,
            is_linear=True,
            aspect_ratio=(3.0, 5.0),
            description="Narrow linear shelter 0.5m wide, 2m long",
        )

        # LOW detectability - too small, needs 10cm resolution
        signatures[TerrainFeature.SPIDER_HOLE] = GeomorphonSignature(
            feature=TerrainFeature.SPIDER_HOLE,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MICRO,
            detection_methods={DetectionMethod.GEOMORPHON},
            detectability=Detectability.LOW,  # Very small, often camouflaged
            micro_2m={GeomorphonType.PIT},
            meso_5m={GeomorphonType.FLAT, GeomorphonType.HOLLOW},
            local_10m={GeomorphonType.FLAT, GeomorphonType.SLOPE},
            min_size_m=0.6,
            max_size_m=1.0,
            is_circular=True,
            description="Small camouflaged fighting position 0.6-1m diameter",
        )

        signatures[TerrainFeature.MACHINE_GUN_NEST] = GeomorphonSignature(
            feature=TerrainFeature.MACHINE_GUN_NEST,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MICRO,
            micro_2m={GeomorphonType.PIT, GeomorphonType.HOLLOW},
            meso_5m={GeomorphonType.PIT, GeomorphonType.HOLLOW, GeomorphonType.VALLEY},
            local_10m={GeomorphonType.HOLLOW, GeomorphonType.SLOPE},
            min_size_m=2.0,
            max_size_m=3.0,
            min_depth_m=0.8,
            max_depth_m=1.5,
            requires_high_ground=True,
            description="Machine gun emplacement 2-3m diameter with approach trench",
        )

        signatures[TerrainFeature.MORTAR_PIT] = GeomorphonSignature(
            feature=TerrainFeature.MORTAR_PIT,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MICRO,
            micro_2m={GeomorphonType.PIT},
            meso_5m={GeomorphonType.PIT, GeomorphonType.HOLLOW},
            local_10m={GeomorphonType.HOLLOW, GeomorphonType.FLAT},
            min_size_m=2.0,
            max_size_m=3.0,
            min_depth_m=1.0,
            max_depth_m=2.0,
            is_circular=True,
            description="Mortar firing position 2-3m diameter, 1.5m deep",
        )

        # ----- MILITARY: Fortifications -----
        signatures[TerrainFeature.BUNKER] = GeomorphonSignature(
            feature=TerrainFeature.BUNKER,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.PEAK, GeomorphonType.SHOULDER},
            meso_5m={GeomorphonType.PEAK, GeomorphonType.SHOULDER, GeomorphonType.RIDGE},
            local_10m={GeomorphonType.SHOULDER, GeomorphonType.SLOPE},
            min_size_m=3.0,
            max_size_m=10.0,
            requires_high_ground=True,
            description="Reinforced shelter 3-10m with earthwork berms",
        )

        signatures[TerrainFeature.PILLBOX] = GeomorphonSignature(
            feature=TerrainFeature.PILLBOX,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.PEAK, GeomorphonType.SHOULDER},
            meso_5m={GeomorphonType.PEAK, GeomorphonType.FLAT},
            local_10m={GeomorphonType.FLAT, GeomorphonType.SHOULDER},
            min_size_m=2.0,
            max_size_m=5.0,
            description="Small concrete fortification, rectangular elevated",
        )

        signatures[TerrainFeature.TOBRUK] = GeomorphonSignature(
            feature=TerrainFeature.TOBRUK,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MICRO,
            micro_2m={GeomorphonType.PIT},
            meso_5m={GeomorphonType.SHOULDER, GeomorphonType.RIDGE},
            local_10m={GeomorphonType.SHOULDER, GeomorphonType.FLAT},
            min_size_m=2.0,
            max_size_m=4.0,
            is_circular=True,
            description="Circular bunker with raised rim and center pit",
        )

        # ----- MILITARY: Linear Defenses -----
        signatures[TerrainFeature.TRENCH] = GeomorphonSignature(
            feature=TerrainFeature.TRENCH,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.VALLEY, GeomorphonType.HOLLOW},
            meso_5m={GeomorphonType.VALLEY, GeomorphonType.HOLLOW},
            local_10m={GeomorphonType.VALLEY, GeomorphonType.SLOPE},
            min_size_m=0.8,
            max_size_m=2.0,
            min_depth_m=1.0,
            max_depth_m=2.5,
            is_linear=True,
            description="Linear defensive excavation 0.8-2m wide, 1.5-2m deep",
        )

        signatures[TerrainFeature.COMMUNICATION_TRENCH] = GeomorphonSignature(
            feature=TerrainFeature.COMMUNICATION_TRENCH,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.VALLEY, GeomorphonType.HOLLOW},
            meso_5m={GeomorphonType.VALLEY, GeomorphonType.HOLLOW},
            local_10m={GeomorphonType.VALLEY, GeomorphonType.FLAT},
            min_size_m=0.6,
            max_size_m=1.0,
            is_linear=True,
            description="Connecting trench 0.6-1m wide, often zigzag pattern",
        )

        # ----- MILITARY: Tactical Terrain -----
        signatures[TerrainFeature.HIGH_GROUND] = GeomorphonSignature(
            feature=TerrainFeature.HIGH_GROUND,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MACRO,
            micro_2m={GeomorphonType.PEAK, GeomorphonType.RIDGE, GeomorphonType.SHOULDER},
            meso_5m={GeomorphonType.PEAK, GeomorphonType.RIDGE, GeomorphonType.SHOULDER},
            local_10m={GeomorphonType.PEAK, GeomorphonType.RIDGE},
            regional_25m={GeomorphonType.PEAK, GeomorphonType.RIDGE},
            description="Elevated position with tactical advantage",
        )

        signatures[TerrainFeature.DEFILADE] = GeomorphonSignature(
            feature=TerrainFeature.DEFILADE,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.HOLLOW, GeomorphonType.FOOTSLOPE},
            meso_5m={GeomorphonType.HOLLOW, GeomorphonType.FOOTSLOPE},
            local_10m={GeomorphonType.FOOTSLOPE, GeomorphonType.SLOPE},
            regional_25m={GeomorphonType.RIDGE, GeomorphonType.SHOULDER},
            description="Position protected from enemy fire behind crest",
        )

        signatures[TerrainFeature.AVENUE_OF_APPROACH] = GeomorphonSignature(
            feature=TerrainFeature.AVENUE_OF_APPROACH,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MACRO,
            micro_2m={GeomorphonType.FLAT, GeomorphonType.VALLEY},
            meso_5m={GeomorphonType.FLAT, GeomorphonType.VALLEY},
            local_10m={GeomorphonType.FLAT, GeomorphonType.VALLEY},
            regional_25m={GeomorphonType.VALLEY, GeomorphonType.FLAT},
            is_linear=True,
            description="Route for movement toward objective",
        )

        # ----- TOPOGRAPHIC: Elevated -----
        signatures[TerrainFeature.HILL] = GeomorphonSignature(
            feature=TerrainFeature.HILL,
            category=FeatureCategory.TOPOGRAPHIC_ELEVATED,
            scale=FeatureScale.MACRO,
            micro_2m={GeomorphonType.PEAK, GeomorphonType.SHOULDER, GeomorphonType.SLOPE},
            meso_5m={GeomorphonType.PEAK, GeomorphonType.SHOULDER, GeomorphonType.SLOPE},
            local_10m={GeomorphonType.PEAK, GeomorphonType.SHOULDER},
            regional_25m={GeomorphonType.PEAK},
            min_size_m=30.0,
            max_size_m=300.0,
            description="Rounded elevation 30-300m prominence",
        )

        signatures[TerrainFeature.KNOLL] = GeomorphonSignature(
            feature=TerrainFeature.KNOLL,
            category=FeatureCategory.TOPOGRAPHIC_ELEVATED,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.PEAK, GeomorphonType.SHOULDER},
            meso_5m={GeomorphonType.PEAK, GeomorphonType.SHOULDER},
            local_10m={GeomorphonType.PEAK, GeomorphonType.SHOULDER},
            min_size_m=10.0,
            max_size_m=30.0,
            description="Small rounded hill 10-30m prominence",
        )

        signatures[TerrainFeature.RIDGE] = GeomorphonSignature(
            feature=TerrainFeature.RIDGE,
            category=FeatureCategory.TOPOGRAPHIC_ELEVATED,
            scale=FeatureScale.MULTI,
            micro_2m={GeomorphonType.RIDGE, GeomorphonType.SHOULDER},
            meso_5m={GeomorphonType.RIDGE, GeomorphonType.SPUR},
            local_10m={GeomorphonType.RIDGE, GeomorphonType.SHOULDER},
            regional_25m={GeomorphonType.RIDGE},
            is_linear=True,
            description="Extended elevation crest, linear pattern at multiple scales",
        )

        signatures[TerrainFeature.SPUR] = GeomorphonSignature(
            feature=TerrainFeature.SPUR,
            category=FeatureCategory.TOPOGRAPHIC_ELEVATED,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.SPUR, GeomorphonType.RIDGE},
            meso_5m={GeomorphonType.SPUR, GeomorphonType.RIDGE},
            local_10m={GeomorphonType.SPUR, GeomorphonType.RIDGE},
            is_linear=True,
            description="Ridge projecting from higher ground",
        )

        signatures[TerrainFeature.SADDLE] = GeomorphonSignature(
            feature=TerrainFeature.SADDLE,
            category=FeatureCategory.TOPOGRAPHIC_ELEVATED,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.FLAT, GeomorphonType.HOLLOW},
            meso_5m={GeomorphonType.FLAT, GeomorphonType.HOLLOW},
            local_10m={GeomorphonType.FLAT, GeomorphonType.RIDGE},
            description="Low point on ridge between two peaks",
        )

        signatures[TerrainFeature.SHOULDER] = GeomorphonSignature(
            feature=TerrainFeature.SHOULDER,
            category=FeatureCategory.TOPOGRAPHIC_ELEVATED,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.SHOULDER},
            meso_5m={GeomorphonType.SHOULDER},
            local_10m={GeomorphonType.SHOULDER, GeomorphonType.SLOPE},
            description="Convex slope break",
        )

        signatures[TerrainFeature.MILITARY_CREST] = GeomorphonSignature(
            feature=TerrainFeature.MILITARY_CREST,
            category=FeatureCategory.TOPOGRAPHIC_ELEVATED,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.SHOULDER},
            meso_5m={GeomorphonType.SHOULDER, GeomorphonType.RIDGE},
            local_10m={GeomorphonType.SHOULDER, GeomorphonType.RIDGE},
            requires_high_ground=True,
            description="Highest point with line of sight to base of slope",
        )

        # ----- TOPOGRAPHIC: Depressions -----
        signatures[TerrainFeature.VALLEY] = GeomorphonSignature(
            feature=TerrainFeature.VALLEY,
            category=FeatureCategory.TOPOGRAPHIC_DEPRESSION,
            scale=FeatureScale.MACRO,
            micro_2m={GeomorphonType.VALLEY, GeomorphonType.FOOTSLOPE},
            meso_5m={GeomorphonType.VALLEY, GeomorphonType.FOOTSLOPE},
            local_10m={GeomorphonType.VALLEY},
            regional_25m={GeomorphonType.VALLEY},
            is_linear=True,
            description="Low area between elevations",
        )

        signatures[TerrainFeature.RAVINE] = GeomorphonSignature(
            feature=TerrainFeature.RAVINE,
            category=FeatureCategory.TOPOGRAPHIC_DEPRESSION,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.VALLEY, GeomorphonType.HOLLOW},
            meso_5m={GeomorphonType.VALLEY},
            local_10m={GeomorphonType.VALLEY, GeomorphonType.SLOPE},
            is_linear=True,
            description="Small steep valley with significant depth",
        )

        signatures[TerrainFeature.GULLY] = GeomorphonSignature(
            feature=TerrainFeature.GULLY,
            category=FeatureCategory.TOPOGRAPHIC_DEPRESSION,
            scale=FeatureScale.MICRO,
            micro_2m={GeomorphonType.VALLEY},
            meso_5m={GeomorphonType.VALLEY, GeomorphonType.HOLLOW},
            local_10m={GeomorphonType.FLAT, GeomorphonType.SLOPE},
            is_linear=True,
            description="Small eroded channel, linear VALLEY at 2m, 5m",
        )

        signatures[TerrainFeature.HOLLOW] = GeomorphonSignature(
            feature=TerrainFeature.HOLLOW,
            category=FeatureCategory.TOPOGRAPHIC_DEPRESSION,
            scale=FeatureScale.MICRO,
            micro_2m={GeomorphonType.HOLLOW},
            meso_5m={GeomorphonType.HOLLOW, GeomorphonType.FLAT},
            local_10m={GeomorphonType.FLAT, GeomorphonType.FOOTSLOPE},
            description="Small concave depression",
        )

        signatures[TerrainFeature.CRATER] = GeomorphonSignature(
            feature=TerrainFeature.CRATER,
            category=FeatureCategory.TOPOGRAPHIC_DEPRESSION,
            scale=FeatureScale.MICRO,
            micro_2m={GeomorphonType.PIT},
            meso_5m={GeomorphonType.PIT, GeomorphonType.SHOULDER},
            local_10m={GeomorphonType.HOLLOW, GeomorphonType.SHOULDER},
            is_circular=True,
            description="Circular depression with raised rim (impact or natural)",
        )

        # ----- WATER FEATURES -----
        signatures[TerrainFeature.STREAM] = GeomorphonSignature(
            feature=TerrainFeature.STREAM,
            category=FeatureCategory.WATER,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.VALLEY},
            meso_5m={GeomorphonType.VALLEY},
            local_10m={GeomorphonType.VALLEY, GeomorphonType.FOOTSLOPE},
            is_linear=True,
            requires_water=True,
            description="Medium flowing water 3-30m wide",
        )

        signatures[TerrainFeature.CREEK] = GeomorphonSignature(
            feature=TerrainFeature.CREEK,
            category=FeatureCategory.WATER,
            scale=FeatureScale.MICRO,
            micro_2m={GeomorphonType.VALLEY},
            meso_5m={GeomorphonType.VALLEY, GeomorphonType.HOLLOW},
            local_10m={GeomorphonType.FLAT, GeomorphonType.FOOTSLOPE},
            is_linear=True,
            requires_water=True,
            description="Small flowing water 1-3m wide",
        )

        signatures[TerrainFeature.FORD] = GeomorphonSignature(
            feature=TerrainFeature.FORD,
            category=FeatureCategory.WATER,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.VALLEY, GeomorphonType.FLAT},
            meso_5m={GeomorphonType.VALLEY, GeomorphonType.FLAT},
            local_10m={GeomorphonType.VALLEY},
            requires_water=True,
            description="Shallow crossing point in stream",
        )

        signatures[TerrainFeature.STREAM_CROSSING] = GeomorphonSignature(
            feature=TerrainFeature.STREAM_CROSSING,
            category=FeatureCategory.WATER,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.VALLEY},
            meso_5m={GeomorphonType.VALLEY},
            local_10m={GeomorphonType.VALLEY, GeomorphonType.FOOTSLOPE},
            regional_25m={GeomorphonType.VALLEY},
            is_linear=True,
            requires_water=True,
            description="Stream valley providing water source or obstacle",
        )

        # ----- BATTLE DAMAGE -----
        signatures[TerrainFeature.SHELL_CRATER] = GeomorphonSignature(
            feature=TerrainFeature.SHELL_CRATER,
            category=FeatureCategory.BATTLE_DAMAGE,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.PIT},
            meso_5m={GeomorphonType.PIT, GeomorphonType.SHOULDER},
            local_10m={GeomorphonType.PIT, GeomorphonType.SHOULDER},
            min_size_m=2.0,
            max_size_m=15.0,
            min_depth_m=0.5,
            max_depth_m=5.0,
            is_circular=True,
            description="Artillery impact crater 2-15m diameter with raised rim",
        )

        signatures[TerrainFeature.BOMB_CRATER] = GeomorphonSignature(
            feature=TerrainFeature.BOMB_CRATER,
            category=FeatureCategory.BATTLE_DAMAGE,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.PIT},
            meso_5m={GeomorphonType.PIT, GeomorphonType.SHOULDER},
            local_10m={GeomorphonType.PIT, GeomorphonType.SHOULDER},
            regional_25m={GeomorphonType.PIT, GeomorphonType.HOLLOW},
            min_size_m=5.0,
            max_size_m=30.0,
            min_depth_m=2.0,
            max_depth_m=10.0,
            is_circular=True,
            description="Aerial bomb impact crater 5-30m diameter",
        )

        signatures[TerrainFeature.CRATER_CLUSTER] = GeomorphonSignature(
            feature=TerrainFeature.CRATER_CLUSTER,
            category=FeatureCategory.BATTLE_DAMAGE,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.PIT},
            meso_5m={GeomorphonType.PIT, GeomorphonType.HOLLOW},
            local_10m={GeomorphonType.HOLLOW},
            description="Multiple impact craters in concentrated area",
        )

        # ----- TRANSPORTATION -----
        signatures[TerrainFeature.ROAD] = GeomorphonSignature(
            feature=TerrainFeature.ROAD,
            category=FeatureCategory.TRANSPORTATION,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.FLAT, GeomorphonType.VALLEY},
            meso_5m={GeomorphonType.FLAT, GeomorphonType.VALLEY},
            local_10m={GeomorphonType.FLAT},
            is_linear=True,
            description="Vehicle route, linear FLAT or cut through terrain",
        )

        signatures[TerrainFeature.ROAD_CUT] = GeomorphonSignature(
            feature=TerrainFeature.ROAD_CUT,
            category=FeatureCategory.TRANSPORTATION,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.VALLEY},
            meso_5m={GeomorphonType.VALLEY},
            local_10m={GeomorphonType.VALLEY, GeomorphonType.RIDGE},
            is_linear=True,
            description="Excavated road passage through ridge",
        )

        signatures[TerrainFeature.TRAIL] = GeomorphonSignature(
            feature=TerrainFeature.TRAIL,
            category=FeatureCategory.TRANSPORTATION,
            scale=FeatureScale.MICRO,
            detection_methods={DetectionMethod.GEOMORPHON},
            detectability=Detectability.LOW,  # Faint - use curvature/TPI for better detection
            micro_2m={GeomorphonType.FLAT, GeomorphonType.VALLEY},
            meso_5m={GeomorphonType.FLAT},
            local_10m={GeomorphonType.FLAT, GeomorphonType.SLOPE},
            min_size_m=0.3,
            max_size_m=2.0,  # Trail width
            min_depth_m=0.05,
            max_depth_m=0.3,  # Subtle depression
            is_linear=True,
            aspect_ratio=(5.0, 100.0),  # Long and narrow
            description="Foot path, faint linear feature. Best detected via profile curvature (<-0.001) + TPI (<-0.05m). Use detect_trails() from subtle_features module for reliable detection.",
        )

        signatures[TerrainFeature.PATH] = GeomorphonSignature(
            feature=TerrainFeature.PATH,
            category=FeatureCategory.TRANSPORTATION,
            scale=FeatureScale.MICRO,
            detection_methods={DetectionMethod.GEOMORPHON},
            detectability=Detectability.LOW,
            micro_2m={GeomorphonType.FLAT, GeomorphonType.VALLEY},
            meso_5m={GeomorphonType.FLAT},
            local_10m={GeomorphonType.FLAT, GeomorphonType.SLOPE},
            min_size_m=0.5,
            max_size_m=1.5,
            min_depth_m=0.03,
            max_depth_m=0.2,
            is_linear=True,
            description="Walking path, similar to trail but may be less defined",
        )

        signatures[TerrainFeature.FOOTPATH] = GeomorphonSignature(
            feature=TerrainFeature.FOOTPATH,
            category=FeatureCategory.TRANSPORTATION,
            scale=FeatureScale.MICRO,
            detection_methods={DetectionMethod.GEOMORPHON},
            detectability=Detectability.LOW,
            micro_2m={GeomorphonType.FLAT, GeomorphonType.VALLEY},
            meso_5m={GeomorphonType.FLAT},
            local_10m={GeomorphonType.FLAT},
            min_size_m=0.3,
            max_size_m=1.0,
            is_linear=True,
            description="Narrow foot path, very subtle linear depression",
        )

        signatures[TerrainFeature.INFILTRATION_ROUTE] = GeomorphonSignature(
            feature=TerrainFeature.INFILTRATION_ROUTE,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MESO,
            detection_methods={DetectionMethod.GEOMORPHON, DetectionMethod.GEOMORPHON_PLUS_CONTEXT},
            detectability=Detectability.CONTEXTUAL,
            micro_2m={GeomorphonType.VALLEY, GeomorphonType.HOLLOW, GeomorphonType.FOOTSLOPE},
            meso_5m={GeomorphonType.VALLEY, GeomorphonType.HOLLOW},
            local_10m={GeomorphonType.VALLEY, GeomorphonType.SLOPE},
            is_linear=True,
            description="Concealed movement route using terrain for cover. Often trails through hollows/ravines. Critical for understanding troop movements.",
        )

        # =====================================================================
        # FEATURES REQUIRING NON-GEOMORPHON DETECTION METHODS
        # These CANNOT be reliably detected from geomorphons alone!
        # =====================================================================

        # ----- VEGETATION: Requires full LiDAR returns (not just ground) -----
        signatures[TerrainFeature.FOREST] = GeomorphonSignature(
            feature=TerrainFeature.FOREST,
            category=FeatureCategory.VEGETATION,
            scale=FeatureScale.MACRO,
            detection_methods={DetectionMethod.VEGETATION_RETURNS, DetectionMethod.RGB_IMAGERY},
            detectability=Detectability.NOT_DETECTABLE,  # From geomorphons
            # Empty geomorphon sets - not detectable this way
            micro_2m=set(),
            meso_5m=set(),
            local_10m=set(),
            requires_vegetation=True,
            description="Large wooded area - REQUIRES full LiDAR returns, not ground surface",
        )

        signatures[TerrainFeature.TREE_LINE] = GeomorphonSignature(
            feature=TerrainFeature.TREE_LINE,
            category=FeatureCategory.VEGETATION,
            scale=FeatureScale.MESO,
            detection_methods={DetectionMethod.VEGETATION_RETURNS},
            detectability=Detectability.NOT_DETECTABLE,  # From geomorphons
            micro_2m=set(),
            meso_5m=set(),
            local_10m=set(),
            is_linear=True,
            requires_vegetation=True,
            description="Forest edge - REQUIRES vegetation height analysis",
        )

        signatures[TerrainFeature.HEDGEROW] = GeomorphonSignature(
            feature=TerrainFeature.HEDGEROW,
            category=FeatureCategory.VEGETATION,
            scale=FeatureScale.MICRO,
            detection_methods={DetectionMethod.VEGETATION_RETURNS, DetectionMethod.RGB_IMAGERY},
            detectability=Detectability.NOT_DETECTABLE,  # From geomorphons
            micro_2m=set(),
            meso_5m=set(),
            local_10m=set(),
            is_linear=True,
            requires_vegetation=True,
            description="Linear vegetation band - REQUIRES vegetation returns",
        )

        signatures[TerrainFeature.CLEARING] = GeomorphonSignature(
            feature=TerrainFeature.CLEARING,
            category=FeatureCategory.VEGETATION,
            scale=FeatureScale.MESO,
            detection_methods={DetectionMethod.VEGETATION_RETURNS},
            detectability=Detectability.NOT_DETECTABLE,  # From geomorphons
            micro_2m={GeomorphonType.FLAT},  # May appear flat, but not diagnostic
            meso_5m={GeomorphonType.FLAT},
            local_10m={GeomorphonType.FLAT},
            requires_vegetation=True,
            description="Open area in forest - REQUIRES vegetation context",
        )

        # ----- MANMADE: Structures filtered out in ground extraction -----
        signatures[TerrainFeature.BUILDING] = GeomorphonSignature(
            feature=TerrainFeature.BUILDING,
            category=FeatureCategory.MANMADE,
            scale=FeatureScale.MESO,
            detection_methods={DetectionMethod.VEGETATION_RETURNS, DetectionMethod.RGB_IMAGERY},
            detectability=Detectability.NOT_DETECTABLE,  # Filtered out in ground extraction
            micro_2m=set(),
            meso_5m=set(),
            local_10m=set(),
            description="Enclosed structure - filtered out when extracting ground points",
        )

        signatures[TerrainFeature.RUIN] = GeomorphonSignature(
            feature=TerrainFeature.RUIN,
            category=FeatureCategory.MANMADE,
            scale=FeatureScale.MESO,
            detection_methods={DetectionMethod.GEOMORPHON, DetectionMethod.RGB_IMAGERY},
            detectability=Detectability.LOW,  # Collapsed structures may show as irregular terrain
            micro_2m={GeomorphonType.PEAK, GeomorphonType.SHOULDER},
            meso_5m={GeomorphonType.PEAK, GeomorphonType.SHOULDER, GeomorphonType.HOLLOW},
            local_10m={GeomorphonType.FLAT, GeomorphonType.SHOULDER},
            description="Collapsed structure - may show as irregular elevated debris",
        )

        signatures[TerrainFeature.FOUNDATION] = GeomorphonSignature(
            feature=TerrainFeature.FOUNDATION,
            category=FeatureCategory.MANMADE,
            scale=FeatureScale.MICRO,
            detection_methods={DetectionMethod.GEOMORPHON},
            detectability=Detectability.MEDIUM,  # Rectangular depressions often visible
            micro_2m={GeomorphonType.PIT, GeomorphonType.HOLLOW},
            meso_5m={GeomorphonType.HOLLOW, GeomorphonType.FLAT},
            local_10m={GeomorphonType.FLAT},
            description="Building base - rectangular depression or platform",
        )

        # ----- WATER: Standing water requires additional analysis -----
        signatures[TerrainFeature.WETLAND] = GeomorphonSignature(
            feature=TerrainFeature.WETLAND,
            category=FeatureCategory.WATER,
            scale=FeatureScale.MESO,
            detection_methods={DetectionMethod.GEOMORPHON, DetectionMethod.FLOW_ACCUMULATION},
            detectability=Detectability.LOW,  # Appears flat, needs hydrology context
            micro_2m={GeomorphonType.FLAT, GeomorphonType.HOLLOW},
            meso_5m={GeomorphonType.FLAT, GeomorphonType.FOOTSLOPE},
            local_10m={GeomorphonType.FLAT, GeomorphonType.VALLEY},
            requires_water=True,
            description="Saturated area - FLAT terrain, needs hydrological context",
        )

        # ----- TACTICAL: Requires context beyond morphology -----
        signatures[TerrainFeature.REVERSE_SLOPE] = GeomorphonSignature(
            feature=TerrainFeature.REVERSE_SLOPE,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MESO,
            detection_methods={
                DetectionMethod.GEOMORPHON_PLUS_CONTEXT,
                DetectionMethod.ASPECT_SLOPE,
            },
            detectability=Detectability.CONTEXTUAL,  # Need to know enemy direction
            micro_2m={GeomorphonType.SLOPE},
            meso_5m={GeomorphonType.SLOPE, GeomorphonType.FOOTSLOPE},
            local_10m={GeomorphonType.SLOPE},
            description="Slope facing away from enemy - REQUIRES tactical context",
        )

        signatures[TerrainFeature.DEAD_GROUND] = GeomorphonSignature(
            feature=TerrainFeature.DEAD_GROUND,
            category=FeatureCategory.MILITARY,
            scale=FeatureScale.MESO,
            detection_methods={DetectionMethod.GEOMORPHON_PLUS_CONTEXT},
            detectability=Detectability.CONTEXTUAL,  # Need viewpoint analysis
            micro_2m={GeomorphonType.HOLLOW, GeomorphonType.VALLEY},
            meso_5m={GeomorphonType.HOLLOW, GeomorphonType.VALLEY},
            local_10m={GeomorphonType.HOLLOW, GeomorphonType.VALLEY},
            description="Area not visible from enemy position - REQUIRES viewshed analysis",
        )

        # ----- PRE-EUROPEAN EARTHWORKS -----
        signatures[TerrainFeature.PLATFORM_MOUND] = GeomorphonSignature(
            feature=TerrainFeature.PLATFORM_MOUND,
            category=FeatureCategory.EARTHWORK,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.FLAT, GeomorphonType.SHOULDER},
            meso_5m={GeomorphonType.FLAT, GeomorphonType.SHOULDER},
            local_10m={GeomorphonType.PEAK, GeomorphonType.SHOULDER, GeomorphonType.FLAT},
            regional_25m={GeomorphonType.PEAK},
            min_size_m=10.0,
            max_size_m=100.0,
            description="Large, flat-topped pre-European platform mound",
        )

        signatures[TerrainFeature.CONICAL_MOUND] = GeomorphonSignature(
            feature=TerrainFeature.CONICAL_MOUND,
            category=FeatureCategory.EARTHWORK,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.PEAK, GeomorphonType.SHOULDER},
            meso_5m={GeomorphonType.PEAK, GeomorphonType.SHOULDER},
            local_10m={GeomorphonType.PEAK},
            min_size_m=5.0,
            max_size_m=50.0,
            is_circular=True,
            description="Circular pre-European conical burial mound",
        )

        signatures[TerrainFeature.PRE_EUROPEAN_ENCLOSURE] = GeomorphonSignature(
            feature=TerrainFeature.PRE_EUROPEAN_ENCLOSURE,
            category=FeatureCategory.EARTHWORK,
            scale=FeatureScale.MACRO,
            micro_2m={GeomorphonType.RIDGE, GeomorphonType.SHOULDER},
            meso_5m={GeomorphonType.RIDGE, GeomorphonType.SHOULDER},
            local_10m={GeomorphonType.RIDGE},
            is_linear=True,
            description="Linear embankment forming a pre-European enclosure",
        )

        # ----- MODERN MODIFICATIONS -----
        signatures[TerrainFeature.CANAL_DREDGING] = GeomorphonSignature(
            feature=TerrainFeature.CANAL_DREDGING,
            category=FeatureCategory.MANMADE,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.RIDGE, GeomorphonType.PEAK},
            meso_5m={GeomorphonType.RIDGE, GeomorphonType.SHOULDER},
            is_linear=True,
            description="Linear spoil piles from modern canal dredging",
        )

        signatures[TerrainFeature.ROAD_WORK] = GeomorphonSignature(
            feature=TerrainFeature.ROAD_WORK,
            category=FeatureCategory.MANMADE,
            scale=FeatureScale.MESO,
            micro_2m={GeomorphonType.FLAT, GeomorphonType.VALLEY, GeomorphonType.RIDGE},
            meso_5m={GeomorphonType.FLAT, GeomorphonType.VALLEY, GeomorphonType.RIDGE},
            is_linear=True,
            description="Regular linear features associated with modern road construction",
        )

        return signatures

    # -------------------------------------------------------------------------
    # TERM MAPPING DEFINITIONS
    # -------------------------------------------------------------------------

    def _build_term_mappings(self) -> Dict[TerrainFeature, TermMapping]:
        """Build document term mappings"""
        mappings = {}

        # ----- Military Positions -----
        mappings[TerrainFeature.FOXHOLE] = TermMapping(
            canonical=TerrainFeature.FOXHOLE,
            synonyms=["fighting hole", "hasty position", "hole", "fighting position"],
            abbreviations=["FH"],
            japanese_terms=["tako-tsubo"],
            german_terms=["schützenloch", "schutzenloch"],
            historical_terms=["rifle pit", "foxhole position"],
        )

        mappings[TerrainFeature.TRENCH] = TermMapping(
            canonical=TerrainFeature.TRENCH,
            synonyms=["ditch", "defensive trench", "entrenchment"],
            german_terms=["kampfgraben", "graben", "schützengraben"],
            japanese_terms=["jinchi"],
            historical_terms=["fire trench", "front-line trench"],
        )

        mappings[TerrainFeature.BUNKER] = TermMapping(
            canonical=TerrainFeature.BUNKER,
            synonyms=["concrete bunker", "fortification", "shelter"],
            german_terms=["bunker", "unterstand"],
            japanese_terms=["obei"],
            historical_terms=["strongpoint", "fortified position"],
        )

        mappings[TerrainFeature.OBSERVATION_POST] = TermMapping(
            canonical=TerrainFeature.OBSERVATION_POST,
            synonyms=["observation point", "lookout", "watch post"],
            abbreviations=["OP", "O.P."],
            german_terms=["beobachtungsposten", "B-Stelle"],
            historical_terms=["outpost", "forward observer position"],
        )

        mappings[TerrainFeature.MACHINE_GUN_NEST] = TermMapping(
            canonical=TerrainFeature.MACHINE_GUN_NEST,
            synonyms=["MG position", "machine gun position", "MG nest"],
            abbreviations=["MG"],
            german_terms=["MG-Nest", "MG-Stellung"],
            historical_terms=["automatic weapons position"],
        )

        mappings[TerrainFeature.SPIDER_HOLE] = TermMapping(
            canonical=TerrainFeature.SPIDER_HOLE,
            synonyms=["one-man hole", "sniper hole"],
            japanese_terms=["tako-tsubo", "tokko"],
            historical_terms=["camouflaged position", "hidden fighting hole"],
        )

        # ----- Military Terrain -----
        mappings[TerrainFeature.HIGH_GROUND] = TermMapping(
            canonical=TerrainFeature.HIGH_GROUND,
            synonyms=["elevated position", "commanding ground", "high terrain"],
            historical_terms=["key terrain", "vital ground", "decisive terrain"],
        )

        mappings[TerrainFeature.DEFILADE] = TermMapping(
            canonical=TerrainFeature.DEFILADE,
            synonyms=["protected position", "cover position"],
            historical_terms=["hull-down", "turret-down", "protected by terrain"],
        )

        mappings[TerrainFeature.DEFENSIVE_LINE] = TermMapping(
            canonical=TerrainFeature.DEFENSIVE_LINE,
            synonyms=["main line of resistance", "defensive position", "front line"],
            abbreviations=["MLR", "FEBA", "HKL"],
            german_terms=["hauptkampflinie", "HKL", "stellung"],
            historical_terms=["line of defense", "front-line positions"],
        )

        mappings[TerrainFeature.AVENUE_OF_APPROACH] = TermMapping(
            canonical=TerrainFeature.AVENUE_OF_APPROACH,
            synonyms=["approach route", "avenue", "corridor"],
            abbreviations=["AA", "AOA"],
            historical_terms=["axis of advance", "approach corridor"],
        )

        # ----- Topographic Features -----
        mappings[TerrainFeature.RIDGE] = TermMapping(
            canonical=TerrainFeature.RIDGE,
            synonyms=["ridgeline", "ridge line", "crest", "high ground"],
            historical_terms=["commanding ridge", "ridge position"],
        )

        mappings[TerrainFeature.HILL] = TermMapping(
            canonical=TerrainFeature.HILL,
            synonyms=["elevation", "rise", "high point", "height"],
            historical_terms=["Hill 101", "the hill", "objective hill"],
        )

        mappings[TerrainFeature.VALLEY] = TermMapping(
            canonical=TerrainFeature.VALLEY,
            synonyms=["low ground", "draw", "drainage"],
            historical_terms=["valley floor", "through the valley"],
        )

        mappings[TerrainFeature.RAVINE] = TermMapping(
            canonical=TerrainFeature.RAVINE,
            synonyms=["gully", "gulch", "canyon", "narrow valley"],
            historical_terms=["deep ravine", "steep ravine"],
        )

        # ----- Water Features -----
        mappings[TerrainFeature.STREAM] = TermMapping(
            canonical=TerrainFeature.STREAM,
            synonyms=["creek", "brook", "watercourse", "river"],
            historical_terms=["stream crossing", "water obstacle"],
        )

        mappings[TerrainFeature.FORD] = TermMapping(
            canonical=TerrainFeature.FORD,
            synonyms=["crossing", "ford crossing", "shallow crossing"],
            historical_terms=["fording site", "river ford"],
        )

        # ----- Battle Damage -----
        mappings[TerrainFeature.SHELL_CRATER] = TermMapping(
            canonical=TerrainFeature.SHELL_CRATER,
            synonyms=["crater", "shell hole", "artillery crater"],
            historical_terms=["impact crater", "explosion crater"],
        )

        mappings[TerrainFeature.BOMB_CRATER] = TermMapping(
            canonical=TerrainFeature.BOMB_CRATER,
            synonyms=["bomb hole", "aerial bomb crater", "large crater"],
            historical_terms=["bombing crater", "air raid crater"],
        )

        # ----- Transportation: Trails and Paths -----
        mappings[TerrainFeature.TRAIL] = TermMapping(
            canonical=TerrainFeature.TRAIL,
            synonyms=["path", "footpath", "track", "footway", "walking path"],
            historical_terms=["supply trail", "jungle trail", "mountain trail", "patrol route"],
        )

        mappings[TerrainFeature.INFILTRATION_ROUTE] = TermMapping(
            canonical=TerrainFeature.INFILTRATION_ROUTE,
            synonyms=["infiltration path", "covered approach", "hidden approach"],
            historical_terms=["sneak route", "approach under cover", "covered corridor"],
        )

        # ----- PRE-EUROPEAN EARTHWORKS -----
        mappings[TerrainFeature.PRE_EUROPEAN_MOUND] = TermMapping(
            canonical=TerrainFeature.PRE_EUROPEAN_MOUND,
            synonyms=["indian mound", "native american mound", "mississippian mound", "burial mound", "earthwork mound"],
            historical_terms=["tumulus", "mound"],
        )

        mappings[TerrainFeature.PLATFORM_MOUND] = TermMapping(
            canonical=TerrainFeature.PLATFORM_MOUND,
            synonyms=["temple mound", "flat-topped mound", "pyramidal mound"],
            historical_terms=["truncated pyramid", "platform"],
        )

        mappings[TerrainFeature.CONICAL_MOUND] = TermMapping(
            canonical=TerrainFeature.CONICAL_MOUND,
            synonyms=["burial mound", "rounded mound"],
        )

        mappings[TerrainFeature.PRE_EUROPEAN_ENCLOSURE] = TermMapping(
            canonical=TerrainFeature.PRE_EUROPEAN_ENCLOSURE,
            synonyms=["earthwork enclosure", "ceremonial enclosure", "geometric earthwork"],
        )

        # ----- MODERN MODIFICATIONS -----
        mappings[TerrainFeature.CANAL_DREDGING] = TermMapping(
            canonical=TerrainFeature.CANAL_DREDGING,
            synonyms=["dredge spoil", "canal bank", "spoil bank", "modern canal"],
        )

        mappings[TerrainFeature.ROAD_WORK] = TermMapping(
            canonical=TerrainFeature.ROAD_WORK,
            synonyms=["road embankment", "road cut", "highway construction"],
        )

        return mappings

    def _build_term_index(self) -> Dict[str, TerrainFeature]:
        """Build inverted index from all terms to features"""
        index = {}

        for feature, mapping in self.term_mappings.items():
            # Add canonical name
            index[feature.value.lower()] = feature

            # Add all synonyms and terms
            for term in mapping.synonyms:
                index[term.lower()] = feature
            for term in mapping.abbreviations:
                index[term.lower()] = feature
            for term in mapping.japanese_terms:
                index[term.lower()] = feature
            for term in mapping.german_terms:
                index[term.lower()] = feature
            for term in mapping.historical_terms:
                index[term.lower()] = feature

        return index


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_terrain_grammar() -> TerrainGrammar:
    """Get singleton instance of terrain grammar"""
    if not hasattr(get_terrain_grammar, "_instance"):
        get_terrain_grammar._instance = TerrainGrammar()
    return get_terrain_grammar._instance


def resolve_document_term(term: str) -> Optional[TerrainFeature]:
    """
    Resolve a term from historical documents to terrain feature.

    Args:
        term: Term from document (e.g., "Schützenloch", "MLR", "foxhole")

    Returns:
        TerrainFeature or None if not recognized
    """
    return get_terrain_grammar().resolve_term(term)


def get_geomorphon_signature(feature: TerrainFeature) -> Optional[GeomorphonSignature]:
    """Get geomorphon signature for terrain feature"""
    return get_terrain_grammar().get_signature(feature)
