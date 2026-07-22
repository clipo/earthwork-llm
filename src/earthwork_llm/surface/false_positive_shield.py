"""
False Positive Shield for EarthworkLLM.

Integrates three context layers into a single, auditable decision for each mound
candidate, so that modern landscape features are actually *rejected* rather than
merely annotated:

    1. Morphological linearity  - aspect ratio of the candidate footprint.
                                   Levees, roads and dredge spoil banks are long
                                   and thin; mounds are compact.
    2. NLCD land cover          - National Land Cover Database class at the
                                   candidate point. Developed land (21-24) and
                                   open water (11) are modern, not prehistoric.
    3. Modern feature proximity - distance to canals, ditches, levees and roads
                                   read from USGS topographic quads (modern + HTMC)
                                   and supplied as a GeoJSON "noise map".

Each layer returns a verdict. A single decisive layer (a clearly linear footprint,
developed land, open water, or a modern feature within the rejection radius) is
enough to REJECT a candidate. A near miss FLAGs the candidate and reduces its
score but keeps it for human review. Otherwise the candidate is KEPT.

A critical design choice: when a context layer cannot be evaluated (for example
the NLCD service is unreachable, as happened during the Jaketown scan), the
shield does NOT silently treat the candidate as clean. It records the layer as
``unavailable`` and surfaces that in the verdict, so downstream reporting can be
honest about which candidates were actually screened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import math


class Decision(str, Enum):
    """Screening outcome for a candidate: keep, flag for review, or reject."""

    KEEP = "keep"
    FLAG = "flag"
    REJECT = "reject"


# NLCD classes that are unambiguously modern / non-archaeological.
NLCD_DEVELOPED = {21, 22, 23, 24}  # Developed: open space -> high intensity
NLCD_WATER = {11}                  # Open water
NLCD_AGRICULTURE = {81, 82}        # Pasture/Hay, Cultivated Crops


@dataclass
class LayerResult:
    """Outcome of one context layer."""
    name: str
    decision: Decision
    available: bool
    detail: str


@dataclass
class ShieldVerdict:
    """Combined decision for a single candidate."""
    decision: Decision
    score: float                       # adjusted probability after screening
    base_score: float                  # detector probability before screening
    reasons: List[str] = field(default_factory=list)
    layers: List[LayerResult] = field(default_factory=list)
    context_complete: bool = True      # False if any layer was unavailable

    @property
    def rejected(self) -> bool:
        """True if the candidate was screened out."""
        return self.decision == Decision.REJECT

    @property
    def kept(self) -> bool:
        """True if the candidate survived screening (kept or flagged)."""
        return self.decision != Decision.REJECT


class FalsePositiveShield:
    """Screen mound candidates against modern-landscape context layers.

    Parameters
    ----------
    linear_reject_aspect:
        Footprints with bounding-box aspect ratio at or above this value are
        rejected outright as levees / roads / canal banks.
    linear_flag_aspect:
        Footprints above this value (but below ``linear_reject_aspect``) are
        flagged and penalised but retained.
    linear_penalty:
        Multiplier applied to the score for a flagged-linear candidate.
    noise_reject_m:
        Distance (metres) within which a modern mapped feature rejects a
        candidate.
    noise_flag_m:
        Distance (metres) within which a modern mapped feature flags a candidate.
    enclosure_query:
        When the active query targets geometric enclosures (which are themselves
        elongated), the linearity layer is disabled so that genuine embankment
        enclosures are not discarded.
    """

    def __init__(
        self,
        linear_reject_aspect: float = 4.5,
        linear_flag_aspect: float = 3.0,
        linear_penalty: float = 0.3,
        noise_reject_m: float = 25.0,
        noise_flag_m: float = 50.0,
        enclosure_query: bool = False,
    ):
        self.linear_reject_aspect = linear_reject_aspect
        self.linear_flag_aspect = linear_flag_aspect
        self.linear_penalty = linear_penalty
        self.noise_reject_m = noise_reject_m
        self.noise_flag_m = noise_flag_m
        self.enclosure_query = enclosure_query

    # ----- individual layers -------------------------------------------------

    def _linearity_layer(self, aspect: Optional[float], nlcd_value: Optional[int] = None) -> LayerResult:
        if self.enclosure_query:
            return LayerResult("linearity", Decision.KEEP, True,
                               "disabled for enclosure query")
        if aspect is None:
            return LayerResult("linearity", Decision.KEEP, False,
                               "aspect ratio not provided")
                               
        # Special rule for agricultural disturbance (plowing artifacts)
        if nlcd_value in NLCD_AGRICULTURE and aspect >= 2.5 and aspect < self.linear_reject_aspect:
            return LayerResult("linearity", Decision.FLAG, True,
                               f"aspect {aspect:.1f} >= 2.5 in agricultural land (possible plowing/ditch artifact)")
                               
        if aspect >= self.linear_reject_aspect:
            return LayerResult("linearity", Decision.REJECT, True,
                               f"aspect {aspect:.1f} >= {self.linear_reject_aspect} (levee/road/canal bank)")
        if aspect >= self.linear_flag_aspect:
            return LayerResult("linearity", Decision.FLAG, True,
                               f"aspect {aspect:.1f} >= {self.linear_flag_aspect} (elongated, possible linear noise)")
        return LayerResult("linearity", Decision.KEEP, True,
                           f"aspect {aspect:.1f} (compact)")

    def _nlcd_layer(self, nlcd_value: Optional[int], nlcd_name: str = "") -> LayerResult:
        # nlcd_value of 0 / None is the convention used by YazooDownloader for a
        # failed or unavailable lookup. Treat it as "unavailable", never "clean".
        if not nlcd_value:
            return LayerResult("nlcd", Decision.KEEP, False,
                               f"land cover unavailable ({nlcd_name or 'no data'})")
        if nlcd_value in NLCD_DEVELOPED:
            return LayerResult("nlcd", Decision.REJECT, True,
                               f"developed land ({nlcd_name})")
        if nlcd_value in NLCD_WATER:
            return LayerResult("nlcd", Decision.REJECT, True,
                               f"open water ({nlcd_name})")
        return LayerResult("nlcd", Decision.KEEP, True,
                           f"non-developed land ({nlcd_name})")

    def _noise_layer(self, nearest_m: Optional[float], nearest_label: str = "") -> LayerResult:
        if nearest_m is None:
            return LayerResult("noise_map", Decision.KEEP, False,
                               "no modern-feature map supplied")
        if nearest_m <= self.noise_reject_m:
            return LayerResult("noise_map", Decision.REJECT, True,
                               f"{nearest_label or 'modern feature'} {nearest_m:.0f} m away (<= {self.noise_reject_m:.0f} m)")
        if nearest_m <= self.noise_flag_m:
            return LayerResult("noise_map", Decision.FLAG, True,
                               f"{nearest_label or 'modern feature'} {nearest_m:.0f} m away (<= {self.noise_flag_m:.0f} m)")
        return LayerResult("noise_map", Decision.KEEP, True,
                           f"nearest modern feature {nearest_m:.0f} m away")

    # ----- combination -------------------------------------------------------

    def evaluate(
        self,
        base_score: float,
        aspect: Optional[float] = None,
        nlcd_value: Optional[int] = None,
        nlcd_name: str = "",
        nearest_noise_m: Optional[float] = None,
        nearest_noise_label: str = "",
    ) -> ShieldVerdict:
        """Combine all layers into a single verdict for one candidate."""
        layers = [
            self._linearity_layer(aspect, nlcd_value),
            self._nlcd_layer(nlcd_value, nlcd_name),
            self._noise_layer(nearest_noise_m, nearest_noise_label),
        ]

        reasons: List[str] = []
        decision = Decision.KEEP
        score = base_score
        context_complete = True

        for layer in layers:
            if not layer.available:
                context_complete = False
            if layer.decision == Decision.REJECT:
                decision = Decision.REJECT
                reasons.append(f"{layer.name}: {layer.detail}")
            elif layer.decision == Decision.FLAG and decision != Decision.REJECT:
                decision = Decision.FLAG
                score *= self.linear_penalty if layer.name == "linearity" else 0.5
                reasons.append(f"{layer.name}: {layer.detail}")

        if decision == Decision.REJECT:
            score = 0.0

        return ShieldVerdict(
            decision=decision,
            score=round(score, 3),
            base_score=base_score,
            reasons=reasons,
            layers=layers,
            context_complete=context_complete,
        )


def nearest_noise_feature(
    lon: float,
    lat: float,
    noise_gdf,
    max_consider_m: float = 200.0,
) -> Tuple[Optional[float], str]:
    """Return (distance_metres, label) to the closest modern feature in a GeoJSON
    noise map, or (None, "") if none is within ``max_consider_m``.

    Uses an approximate local metric conversion at the candidate latitude, which
    is accurate to better than 1% for the short distances that matter here.
    """
    if noise_gdf is None or len(noise_gdf) == 0:
        return None, ""

    try:
        from shapely.geometry import Point
    except ImportError:
        return None, ""

    # Degrees -> metres scale factors at this latitude.
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    # Convert the consider radius to degrees (use the smaller scale to be safe).
    deg_radius = max_consider_m / min(m_per_deg_lat, m_per_deg_lon)

    pt = Point(lon, lat)
    best_m: Optional[float] = None
    best_label = ""
    for _, row in noise_gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if geom.distance(pt) > deg_radius:
            continue
        # Approximate metric distance by sampling the geometry's nearest point.
        try:
            from shapely.ops import nearest_points
            near = nearest_points(geom, pt)[0]
            dx = (near.x - lon) * m_per_deg_lon
            dy = (near.y - lat) * m_per_deg_lat
            dist_m = math.hypot(dx, dy)
        except Exception:
            dist_m = geom.distance(pt) * min(m_per_deg_lat, m_per_deg_lon)
        if best_m is None or dist_m < best_m:
            best_m = dist_m
            label = row.get("type") or row.get("class") or "modern feature"
            name = row.get("name")
            best_label = f"{label} ({name})" if name else str(label)

    if best_m is not None and best_m <= max_consider_m:
        return best_m, best_label
    return None, ""
