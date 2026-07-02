#!/usr/bin/env python3
"""
Explanatory schematic figures for the EarthworkLLM paper.

Regenerates the conceptual diagrams that were too thin to be useful:

    fig_geomorphon_concepts.png  - how geomorphons are computed and what each
                                   landform means archaeologically
    fig_fp_shield.png            - the False Positive Shield decision flow

All values shown match the code: 1.0 degree flatness threshold, scales of
2/5/10/25 m, Qwen3-VL-30B-A3B-Thinking (V9), and the linearity / NLCD /
modern-feature layers with their KEEP / FLAG / REJECT outcomes.

    python scripts/generate_explanatory_figures.py
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "docs/figures"

INK = "#2c3e50"
ACCENT = "#8B0000"
GOOD = "#1e8449"
WARN = "#b9770e"
BAD = "#922b21"


# ---------------------------------------------------------------------------
# Figure: Geomorphon concepts
# ---------------------------------------------------------------------------

def _profile_icon(ax, kind):
    """Draw a small terrain cross-section illustrating one landform."""
    x = np.linspace(0, 1, 200)
    if kind == "PEAK":
        y = np.exp(-((x - 0.5) ** 2) / 0.02)
    elif kind == "RIDGE":
        y = 0.2 + 0.8 * np.exp(-((x - 0.5) ** 2) / 0.05)
        y = np.clip(y, 0, 1)
    elif kind == "FLAT":
        y = np.full_like(x, 0.5)
    elif kind == "VALLEY":
        y = 1 - 0.8 * np.exp(-((x - 0.5) ** 2) / 0.05)
    elif kind == "PIT":
        y = 1 - np.exp(-((x - 0.5) ** 2) / 0.02)
    elif kind == "SLOPE":
        y = x
    ax.fill_between(x, y, -0.1, color="#d7ccc8", zorder=1)
    ax.plot(x, y, color=INK, lw=1.8, zorder=2)
    # center cell marker
    ci = 100
    ax.plot([x[ci]], [y[ci]], "o", color=ACCENT, ms=6, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.15)
    ax.axis("off")


def geomorphon_figure():
    fig = plt.figure(figsize=(13, 9))

    # --- Panel A: line of sight on a cross-section --------------------------
    axA = fig.add_axes([0.06, 0.55, 0.52, 0.36])
    x = np.linspace(0, 10, 400)
    ground = (1.5 * np.exp(-((x - 5) ** 2) / 1.2)
              + 0.25 * np.sin(x) + 0.3 * x / 10)
    axA.fill_between(x, ground, -0.5, color="#d7ccc8", zorder=1)
    axA.plot(x, ground, color=INK, lw=2, zorder=2)
    ci = np.argmin(np.abs(x - 5))
    cx, cy = x[ci], ground[ci]
    axA.plot([cx], [cy], "o", color=ACCENT, ms=10, zorder=5)
    # two sample sight lines
    for tx in (1.0, 9.0):
        ti = np.argmin(np.abs(x - tx))
        axA.plot([cx, x[ti]], [cy, ground[ti]], "--", color="#555", lw=1.2, zorder=3)
    axA.annotate("look DOWN  (−)", xy=(1.0, ground[40]), xytext=(0.2, 2.4),
                 fontsize=10, color=GOOD, fontweight="bold")
    axA.annotate("look DOWN  (−)", xy=(9.0, ground[-40]), xytext=(7.0, 2.4),
                 fontsize=10, color=GOOD, fontweight="bold")
    axA.text(cx, cy + 0.25, "cell under test", ha="center", fontsize=9,
             color=ACCENT, fontweight="bold")
    axA.text(0.2, -0.35,
             "Each cell looks out in 8 directions. A sight line that rises is +1, "
             "that falls is −1,\nand one within the flatness angle (1.0°) is 0. "
             "The 8 ternary values form a code.",
             fontsize=9.5)
    axA.set_xlim(0, 10)
    axA.set_ylim(-0.5, 3.0)
    axA.axis("off")
    axA.set_title("A.  Eight-direction line-of-sight + 1.0° flatness threshold",
                  fontsize=12, fontweight="bold", loc="left")

    # --- Panel B: compass of the 8 ternary values ---------------------------
    axB = fig.add_axes([0.64, 0.55, 0.30, 0.36])
    axB.set_xlim(-1.4, 1.4)
    axB.set_ylim(-1.4, 1.4)
    axB.axis("off")
    axB.set_title("B.  Ternary code (this cell = PEAK)", fontsize=12,
                  fontweight="bold", loc="left")
    axB.plot([0], [0], "o", color=ACCENT, ms=12, zorder=5)
    for ang in range(0, 360, 45):
        a = np.radians(ang)
        dx, dy = np.cos(a), np.sin(a)
        axB.annotate("", xy=(dx, dy), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=GOOD, lw=2))
        axB.text(dx * 1.22, dy * 1.22, "−", ha="center", va="center",
                 fontsize=15, color=GOOD, fontweight="bold")
    axB.text(0, -1.32, "8 downs, 0 ups  →  PEAK", ha="center", fontsize=10,
             fontweight="bold")

    # --- Panel C: landform atlas + archaeology ------------------------------
    atlas = [
        ("PEAK", "8 down / 0 up", "Conical burial mound"),
        ("RIDGE", "6 down, linear", "Levee / embankment"),
        ("FLAT", "all level", "Alluvial plain; platform-mound top"),
        ("SLOPE", "graded", "Natural hillside / bluff face"),
        ("VALLEY", "6 up, linear", "Drainage ditch / canal"),
        ("PIT", "8 up / 0 down", "Borrow pit / sunken feature"),
    ]
    for i, (name, code, arch) in enumerate(atlas):
        col = i % 3
        row = i // 3
        x0 = 0.06 + col * 0.31
        y0 = 0.26 - row * 0.22
        ic = fig.add_axes([x0, y0, 0.12, 0.13])
        _profile_icon(ic, name)
        fig.text(x0 + 0.135, y0 + 0.10, name, fontsize=12, fontweight="bold",
                 color=ACCENT)
        fig.text(x0 + 0.135, y0 + 0.065, code, fontsize=9, color="#555")
        fig.text(x0 + 0.135, y0 + 0.025, arch, fontsize=9.5, style="italic",
                 color=INK, wrap=True)

    fig.text(0.06, 0.49,
             "C.  Six diagnostic landforms and their archaeological analogues "
             "(the morphology, not the height, drives classification)",
             fontsize=12, fontweight="bold")

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/fig_geomorphon_concepts.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig_geomorphon_concepts.png")


# ---------------------------------------------------------------------------
# Figure: False Positive Shield decision flow
# ---------------------------------------------------------------------------

def shield_figure():
    """Shield decision flow: three layers feed ONE combined verdict.

    The earlier drawing paired each layer with a different verdict arrow,
    which misread the logic; every candidate passes through all three layers
    and receives a single KEEP / FLAG / REJECT outcome.
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    # Candidate in
    ax.add_patch(FancyBboxPatch((2, 26), 14, 8, boxstyle="round,pad=0.4",
                                linewidth=2, edgecolor=INK, facecolor="#fdebd0"))
    ax.text(9, 30, "Geomorphon\ncandidate", ha="center", va="center",
            fontsize=10, fontweight="bold")

    layers = [
        ("LINEARITY", "aspect ratio of footprint", 46),
        ("NLCD LAND COVER", "class at candidate point", 30),
        ("MODERN FEATURES", "distance to mapped canal / levee / road", 14),
    ]
    for name, desc, y in layers:
        ax.add_patch(FancyBboxPatch((24, y - 4.5), 30, 9, boxstyle="round,pad=0.4",
                                    linewidth=2, edgecolor=INK, facecolor="#eef3f7"))
        ax.text(39, y + 1.7, name, ha="center", fontsize=11, fontweight="bold",
                color=INK)
        ax.text(39, y - 1.8, desc, ha="center", fontsize=8.5, style="italic",
                color="#555")
        ax.annotate("", xy=(24, y), xytext=(16, 30),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#888"))
        ax.annotate("", xy=(62, 30), xytext=(54, y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#888"))

    # Combined verdict node
    ax.add_patch(FancyBboxPatch((62, 25), 15, 10, boxstyle="round,pad=0.4",
                                linewidth=2.2, edgecolor=INK, facecolor="#f4ecf7"))
    ax.text(69.5, 31.5, "combined\nverdict", ha="center", va="center",
            fontsize=10, fontweight="bold", color=INK)
    ax.text(69.5, 27.0, "all three layers,\none outcome", ha="center",
            va="center", fontsize=7.8, style="italic", color="#555")

    outcomes = [
        ("KEEP", GOOD, "compact, non-developed,\nclear of modern features", 47),
        ("FLAG", WARN, "borderline: retained with a\nreduced score, for review", 30),
        ("REJECT", BAD, "a decisive layer fired;\ndropped, reason recorded", 13),
    ]
    for label, color, text, y in outcomes:
        ax.add_patch(FancyBboxPatch((84, y - 5), 14.5, 10, boxstyle="round,pad=0.4",
                                    linewidth=2, edgecolor=color, facecolor="white"))
        ax.text(91.2, y + 2.4, label, ha="center", va="center", fontsize=11,
                fontweight="bold", color=color)
        ax.text(91.2, y - 1.8, text, ha="center", va="center", fontsize=7.6,
                color=INK)
        ax.annotate("", xy=(84, y), xytext=(77, 30),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color=color))

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/fig_fp_shield.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig_fp_shield.png")


if __name__ == "__main__":
    geomorphon_figure()
    shield_figure()
