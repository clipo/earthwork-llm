"""
Two-score triage ranking for EarthworkLLM.

Implements the two-score triage design of the manuscript (Sections 3.8 and 4,
measured in Appendix B.6). Every surviving candidate carries two independent
ranks so the analyst's queue can be ordered by both axes at once:

    Score A (artificial vs natural)  - how mound-like is the form? Computed
        local relief above the surrounding fabric is the backbone (relief
        alone ranks the Jaketown survivors at AUC 0.72; Appendix B.6), plus
        footprint compactness (the inverse of the bounding-box aspect ratio
        the shield's linearity layer already uses). Both components are
        z-scored within the scan, so A is a relative rank against the scan's
        own candidate population, not an absolute threshold. A RISES with
        relief and compactness. The model's feature-scale isolation reading
        (Section 3.8) fuses with relief at AUC 0.79 when the interpretation
        layer is running; at scan scale, with no model in the loop, Score A
        is the deterministic relief-plus-compactness backbone only.

    Score B (recent vs old)          - how strong is the modern association?
        A deterministic land-use-records score in the lineage of Davis et
        al. (2018), kept rule-based because the model does not exploit
        textual records (Appendix B.6). At scan scale it uses only the FAST
        layers: the single NLCD land-cover query and noise-map proximity the
        shield already computed, plus (optionally, per surviving candidate,
        never per raw candidate) one FEMA USA Structures and one NHD canal
        distance query. The full context sheet (scripts/context_sheet.py,
        1-2 min per point) is reserved for the review stage. B RISES with
        modern association: a high-B candidate is well explained by mapped
        modern activity, a low-B candidate is not.

Missing context never silently reads as clean, matching the False Positive
Shield's honest-reporting convention: an unavailable layer is excluded from
the Score B mean and the result is flagged ``complete=False``, so downstream
reporting can state which candidates were actually screened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import math


# NLCD class groups (mirrors false_positive_shield.py).
NLCD_DEVELOPED = {21, 22, 23, 24}   # Developed: open space -> high intensity
NLCD_WATER = {11}                   # Open water (borrow pits, impoundments)
NLCD_AGRICULTURE = {81, 82}         # Pasture/Hay, Cultivated Crops

# Distance (metres) beyond which a mapped modern feature contributes no
# modern-association evidence. Matches the 200 m consider radius of
# false_positive_shield.nearest_noise_feature.
MODERN_DECAY_M = 200.0


@dataclass
class ScanStats:
    """Scan-population mean/sd for Score A's z-scoring.

    Computed once over every screened candidate in the scan (kept, flagged
    AND rejected), so a candidate's Score A is its standing against the full
    fabric the detector surfaced, independent of shield filtering choices.
    """
    relief_mean: float = 0.0
    relief_sd: float = 0.0
    compactness_mean: float = 0.0
    compactness_sd: float = 0.0
    n: int = 0


@dataclass
class ScoreB:
    """Score B result for a single candidate.

    ``score`` is the mean of the available components (each 0..1, higher =
    stronger modern association), or None when no component was available.
    ``complete`` is False when any expected layer was missing, following the
    shield's ShieldVerdict.context_complete convention.
    """
    score: Optional[float]
    complete: bool
    components: Dict[str, Optional[float]] = field(default_factory=dict)


def _compactness(aspect: Optional[float]) -> Optional[float]:
    """Bounding-box compactness in (0, 1]: 1.0 = square footprint."""
    if aspect is None or aspect <= 0:
        return None
    return 1.0 / max(1.0, aspect)


def scan_stats(candidates: Sequence[dict]) -> ScanStats:
    """Compute the scan-population statistics Score A is z-scored against.

    Parameters
    ----------
    candidates:
        Every screened detector candidate in the scan (the dicts produced by
        ``detect_earthworks``, carrying ``height`` and ``aspect``).
    """
    reliefs = [c["height"] for c in candidates
               if c.get("height") is not None]
    compacts = [c for c in (_compactness(cand.get("aspect"))
                            for cand in candidates) if c is not None]

    def _mean_sd(vals: List[float]) -> tuple:
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return mean, math.sqrt(var)

    r_mean, r_sd = _mean_sd(reliefs)
    c_mean, c_sd = _mean_sd(compacts)
    return ScanStats(relief_mean=r_mean, relief_sd=r_sd,
                     compactness_mean=c_mean, compactness_sd=c_sd,
                     n=len(candidates))


def _z(value: float, mean: float, sd: float) -> float:
    return (value - mean) / sd if sd > 0 else 0.0


def score_a(candidate: dict, stats: ScanStats) -> Optional[float]:
    """Score A (artificial vs natural) for one candidate (Sections 3.8, 4;
    Appendix B.6).

    Sum of the z-scored local relief (``candidate['height']``, metres above
    the surrounding fabric) and the z-scored footprint compactness (inverse
    of ``candidate['aspect']``, the bounding-box aspect ratio). Z-scoring is
    within the scan (``stats`` from :func:`scan_stats`), so the score is a
    relative rank, not an absolute measure. Score A RISES with relief and
    with compactness: a tall, compact candidate looks built; a low or
    elongated one looks like natural or linear fabric.

    Returns None when relief or aspect is missing from the candidate record.
    """
    relief = candidate.get("height")
    compact = _compactness(candidate.get("aspect"))
    if relief is None or compact is None:
        return None
    return round(_z(relief, stats.relief_mean, stats.relief_sd)
                 + _z(compact, stats.compactness_mean, stats.compactness_sd),
                 3)


def _proximity_component(dist_m: Optional[float]) -> Optional[float]:
    """Linear decay: 1.0 on the feature, 0.0 at/beyond MODERN_DECAY_M.

    None means the layer was not evaluated (service down or query skipped)
    and is excluded rather than treated as clean. ``math.inf`` means the
    layer WAS queried and found nothing in range: evidence of no modern
    association, contributing 0.0.
    """
    if dist_m is None:
        return None
    return round(max(0.0, 1.0 - dist_m / MODERN_DECAY_M), 3)


def _landcover_component(nlcd_value: Optional[int]) -> Optional[float]:
    """Modern-association weight of the (single-epoch) NLCD class.

    The full B.6 records score uses land-cover change since 1985; at scan
    scale one epoch must stand in for the history, so the component grades
    the class itself: developed land is modern by definition (1.0), open
    water in the Delta is usually a borrow pit or impoundment (0.75),
    agriculture is actively worked ground (0.5), and undeveloped classes
    carry no modern association (0.0). The value 0/None is the downloader's
    convention for an unavailable lookup and yields None, never clean.
    """
    if not nlcd_value:
        return None
    if nlcd_value in NLCD_DEVELOPED:
        return 1.0
    if nlcd_value in NLCD_WATER:
        return 0.75
    if nlcd_value in NLCD_AGRICULTURE:
        return 0.5
    return 0.0


def score_b(candidate: dict, shield_context: dict) -> ScoreB:
    """Score B (recent vs old, i.e. modern association) for one candidate
    (Sections 3.8, 4; Appendix B.6).

    Deterministic land-use-evidence score built from the context the shield
    already gathered plus the optional fast per-survivor distance queries.
    Score B RISES with modern association. Components (each 0..1):

        land_cover          - single NLCD class at the point
                              (``shield_context['nlcd_value']``).
        noise_proximity     - distance to the nearest mapped modern feature
                              in the USGS-quad noise map
                              (``nearest_noise_m``; requires
                              ``noise_map_available``).
        structure_proximity - distance to the nearest FEMA USA Structures
                              footprint (``structure_m``).
        canal_proximity     - distance to the nearest NHD canal/ditch
                              (``canal_m``).

    Distance conventions: None = layer unavailable or not queried (excluded,
    flags the result incomplete); ``math.inf`` = queried, nothing within the
    search radius (contributes 0.0). The score is the mean of the available
    components, or None when none are available; ``complete`` is True only
    when all four layers were evaluated, matching the shield's
    honest-reporting convention.

    ``candidate`` is accepted for signature symmetry with :func:`score_a`
    and future form-conditional rules; the current rules use only
    ``shield_context``.
    """
    del candidate  # present for API symmetry; rules are context-only today

    noise_m = shield_context.get("nearest_noise_m")
    if noise_m is None and shield_context.get("noise_map_available"):
        # A noise map was supplied and found nothing within its consider
        # radius: that is evidence, not a gap.
        noise_m = math.inf

    components: Dict[str, Optional[float]] = {
        "land_cover": _landcover_component(shield_context.get("nlcd_value")),
        "noise_proximity": _proximity_component(noise_m),
        "structure_proximity": _proximity_component(
            shield_context.get("structure_m")),
        "canal_proximity": _proximity_component(
            shield_context.get("canal_m")),
    }

    available = [v for v in components.values() if v is not None]
    score = round(sum(available) / len(available), 3) if available else None
    return ScoreB(score=score,
                  complete=len(available) == len(components),
                  components=components)


def rank_descending(scores: Sequence[Optional[float]]) -> List[Optional[int]]:
    """1-based competition ranks, highest score first, None-safe.

    Candidates with a None score receive a None rank (never a default rank),
    so an unscored candidate cannot masquerade as a low-priority one. Ties
    share the smallest rank (competition / "min" ranking).
    """
    order = sorted((i for i, s in enumerate(scores) if s is not None),
                   key=lambda i: -scores[i])
    ranks: List[Optional[int]] = [None] * len(scores)
    rank, prev = 0, None
    for pos, i in enumerate(order, start=1):
        if scores[i] != prev:
            rank, prev = pos, scores[i]
        ranks[i] = rank
    return ranks
