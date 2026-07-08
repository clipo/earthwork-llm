#!/usr/bin/env python
"""Prototype negative-relief ditch / enclosure detector (geomorphon circuits).

Idea: an enclosure ditch shows up in a geomorphon layer as a connected circuit
of depression cells (VALLEY / HOLLOW / PIT) that wraps around a common
centroid; an embankment shows up as the positive-relief mirror. Linear
drainage also produces depression cells, but its cells lie along a line, so
their bearings from the component centroid concentrate in two opposite
sectors. Scoring angular coverage around the centroid therefore separates
closed (or near-closed) circuits from ordinary drainage.

Anchors are limited to coordinates traceable to public agency documents:

  spanish_fort  Spanish Fort Site (22SH500), Sharkey County, MS.
                USGS GNIS Feature ID 694850 ("Spanish Fort", Holly Bluff
                quad): 32.7595764 N, -90.7348198 W. NRHP ref 88000234
                (listed 1988; nomination address-restricted). Published
                descriptions (e.g., Jackson 1998, Midcontinental Journal of
                Archaeology 23(2):199-220, on the neighboring Little Spanish
                Fort) characterize these as semicircular embankment
                enclosures open to a waterway — expect an ARC, not a full
                360 circuit.

  marksville    Marksville Prehistoric Indian Site (NHL, 16AV1), Avoyelles
                Parish, LA. NPS NRHP public GIS point, NRIS refnum 66000372:
                31.125117663 N, -92.048133136 W. NHL Form 10-317 (Griffin,
                2/28/1964): "NE 1 mile from town of Marksville on bluffs
                overlooking Old River". Semicircular embankment (~3,300 ft
                long, 3-7 ft high per NHL/NRHP documentation), open on the
                bluff side — again expect an ARC.

Both sites are in UTM Zone 15N (EPSG:26915).

Classifier quirk (documented, not fixed here): classify_geomorphon_simple's
ternary table never emits RIDGE — positive linear relief comes out as SPUR
(and depressions spill into FOOTSLOPE). The embankment mode therefore selects
{PEAK, RIDGE, SHOULDER, SPUR} and the ditch mode {HOLLOW, FOOTSLOPE, VALLEY,
PIT} can be enabled with --wide-classes; the strict default is
{HOLLOW, VALLEY, PIT}.

Usage:
  python scripts/ditch_enclosure_query.py --site spanish_fort --mode ditch --decoys
  python scripts/ditch_enclosure_query.py --site marksville --mode embankment --decoys \
      --png-dir /tmp/out
  python scripts/ditch_enclosure_query.py --utm 712198.8 3626904.6 --size 900 --mode ditch
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from earthwork_llm.ingestion.imageserver import fetch_dem  # noqa: E402
from demo_terrain_query import (  # noqa: E402
    classify_geomorphon_simple,
    make_hillshade,
    GEOMORPHON_TYPES,
)

# Geomorphon class indices (see GEOMORPHON_TYPES)
FLAT, PEAK, RIDGE, SHOULDER, SPUR, SLOPE, HOLLOW, FOOTSLOPE, VALLEY, PIT = range(10)

NEGATIVE_CLASSES = {HOLLOW, VALLEY, PIT}          # strict ditch signature
NEGATIVE_CLASSES_WIDE = {HOLLOW, FOOTSLOPE, VALLEY, PIT}
POSITIVE_CLASSES = {PEAK, RIDGE, SHOULDER, SPUR}  # embankment signature

# Published-anchor presets (UTM 15N, EPSG:26915). Citations in module docstring.
SITES = {
    "spanish_fort": (712198.8, 3626904.6),
    "marksville": (590754.1, 3443857.9),
}

SECTOR_DEG = 10  # bearing-sector width for angular coverage


@dataclass
class Circuit:
    cx_px: float          # centroid, pixel coords (col)
    cy_px: float          # centroid, pixel coords (row)
    utm_x: float
    utm_y: float
    n_cells: int
    mean_radius_m: float
    radius_cv: float      # std/mean of cell radii — low for thin rings
    coverage_deg: float   # bearing sectors occupied around centroid
    verdict: str          # closed / arc / linear

    def describe(self) -> str:
        return (
            f"{self.verdict.upper():6s} centroid UTM ({self.utm_x:.0f}, {self.utm_y:.0f})  "
            f"r={self.mean_radius_m:.0f} m (cv {self.radius_cv:.2f})  "
            f"coverage={self.coverage_deg:.0f} deg  cells={self.n_cells}"
        )


def find_circuits(
    geo: np.ndarray,
    classes: set[int],
    origin_xy: tuple[float, float],  # UTM of pixel (row 0, col 0)
    resolution_m: float = 1.0,
    min_cells: int = 150,
    min_radius_m: float = 15.0,
    max_radius_frac: float = 0.45,
    dilate_iter: int = 2,
    closed_deg: float = 270.0,
    arc_deg: float = 135.0,
) -> list[Circuit]:
    """Connected components of the selected geomorphon classes, scored for
    closure by angular coverage of cell bearings around the component centroid.
    """
    from scipy.ndimage import binary_dilation, label

    h, w = geo.shape
    mask = np.isin(geo, list(classes))
    # Bridge small gaps so a dashed circuit labels as one component.
    grown = binary_dilation(mask, iterations=dilate_iter) if dilate_iter else mask
    labels, n = label(grown, structure=np.ones((3, 3), dtype=int))

    max_radius_m = max_radius_frac * min(h, w) * resolution_m
    ox, oy = origin_xy
    circuits: list[Circuit] = []
    for i in range(1, n + 1):
        # Score on the ORIGINAL (undilated) cells of this component.
        comp = mask & (labels == i)
        ys, xs = np.nonzero(comp)
        if len(ys) < min_cells:
            continue
        cx, cy = xs.mean(), ys.mean()
        dx, dy = xs - cx, ys - cy
        radii = np.hypot(dx, dy) * resolution_m
        mean_r = float(radii.mean())
        if not (min_radius_m <= mean_r <= max_radius_m):
            continue
        # Angular coverage: bearings binned into 10-degree sectors.
        bearings = np.degrees(np.arctan2(dy, dx)) % 360.0
        sectors = np.unique((bearings // SECTOR_DEG).astype(int))
        coverage = float(len(sectors) * SECTOR_DEG)
        if coverage >= closed_deg:
            verdict = "closed"
        elif coverage >= arc_deg:
            verdict = "arc"
        else:
            verdict = "linear"
        circuits.append(Circuit(
            cx_px=float(cx), cy_px=float(cy),
            utm_x=ox + cx * resolution_m, utm_y=oy - cy * resolution_m,
            n_cells=int(len(ys)),
            mean_radius_m=mean_r,
            radius_cv=float(radii.std() / mean_r) if mean_r else 99.0,
            coverage_deg=coverage,
            verdict=verdict,
        ))
    circuits.sort(key=lambda c: -c.coverage_deg)
    return circuits


def run_window(
    utm_x: float,
    utm_y: float,
    size_m: int,
    mode: str,
    wide_classes: bool = False,
    top_n: int = 5,
    png_path: Path | None = None,
    label_txt: str = "",
) -> list[Circuit]:
    dem = fetch_dem(utm_x, utm_y, size_m, crs_epsg=26915, resolution_m=1.0)
    dem = np.where(np.isnan(dem), np.nanmedian(dem), dem)
    geo = classify_geomorphon_simple(dem, lookup=5, flat_thresh=0.3)
    if mode == "embankment":
        classes = POSITIVE_CLASSES
    else:
        classes = NEGATIVE_CLASSES_WIDE if wide_classes else NEGATIVE_CLASSES
    origin = (utm_x - size_m / 2.0, utm_y + size_m / 2.0)  # UTM of pixel (0,0)
    circuits = find_circuits(geo, classes, origin)

    print(f"\n== {label_txt} window center UTM15N ({utm_x:.0f}, {utm_y:.0f}), "
          f"{size_m} m, mode={mode} "
          f"classes={sorted(GEOMORPHON_TYPES[c] for c in classes)}")
    if not circuits:
        print("   no candidate circuits (nothing above size/radius floor)")
    for c in circuits[:top_n]:
        print(f"   {c.describe()}")

    if png_path is not None:
        _render(dem, geo, classes, circuits[:top_n], png_path, label_txt)
    return circuits


def _render(dem, geo, classes, circuits, png_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = make_hillshade(dem)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2))
    axes[0].imshow(hs, cmap="gray")
    axes[0].set_title("hillshade")
    axes[1].imshow(hs, cmap="gray")
    sel = np.isin(geo, list(classes))
    overlay = np.zeros((*geo.shape, 4))
    overlay[sel] = (0.55, 0.15, 0.85, 0.8)
    axes[1].imshow(overlay)
    axes[1].set_title("selected geomorphon classes + circuits")
    for c in circuits:
        color = {"closed": "lime", "arc": "orange", "linear": "red"}[c.verdict]
        for ax in axes[1:]:
            circ = plt.Circle((c.cx_px, c.cy_px), c.mean_radius_m, fill=False,
                              color=color, lw=1.5)
            ax.add_patch(circ)
            ax.annotate(f"{c.verdict} {c.coverage_deg:.0f}°",
                        (c.cx_px, c.cy_px), color=color, fontsize=9,
                        ha="center")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    print(f"   overlay saved: {png_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    loc = ap.add_mutually_exclusive_group(required=True)
    loc.add_argument("--site", choices=sorted(SITES), help="published-anchor preset")
    loc.add_argument("--utm", nargs=2, type=float, metavar=("X", "Y"),
                     help="window center, UTM 15N meters (EPSG:26915)")
    ap.add_argument("--size", type=int, default=900, help="window size in m (default 900)")
    ap.add_argument("--mode", choices=("ditch", "embankment"), default="ditch")
    ap.add_argument("--wide-classes", action="store_true",
                    help="ditch mode: also include FOOTSLOPE (classifier quirk)")
    ap.add_argument("--decoys", action="store_true",
                    help="also run 4 windows 2 km N/E/S/W (false-alarm check)")
    ap.add_argument("--png-dir", type=Path, default=None,
                    help="write hillshade+overlay PNG per window here")
    args = ap.parse_args()

    x, y = SITES[args.site] if args.site else args.utm
    name = args.site or f"utm_{x:.0f}_{y:.0f}"

    def png(tag: str) -> Path | None:
        return args.png_dir / f"detector_{name}_{tag}_{args.mode}.png" if args.png_dir else None

    run_window(x, y, args.size, args.mode, args.wide_classes,
               png_path=png("anchor"), label_txt=f"{name} ANCHOR")
    if args.decoys:
        for tag, (dx, dy) in dict(N=(0, 2000), E=(2000, 0),
                                  S=(0, -2000), W=(-2000, 0)).items():
            run_window(x + dx, y + dy, args.size, args.mode, args.wide_classes,
                       png_path=png(f"decoy{tag}"), label_txt=f"{name} DECOY {tag} (+2 km)")


if __name__ == "__main__":
    main()
