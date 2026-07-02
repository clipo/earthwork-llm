#!/usr/bin/env python3
"""Figure 2: conventional supervised detection versus label-free / zero-shot
transfer. No caption text is baked into the image; captions belong to the
manuscript.

    python scripts/generate_zero_shot_fig.py
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "docs/figures/fig_zero_shot_concept.png"

INK = "#2c3e50"
RED_EDGE = "#c0392b"
RED_FILL = "#fdecea"
GREEN_EDGE = "#1e8449"
GREEN_FILL = "#eafaf1"


def draw_chain(ax, y, boxes, edge, fill, note=None, note_color=None):
    n = len(boxes)
    w, gap, x0 = 21, 4.5, 2
    for i, text in enumerate(boxes):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y - 5.5), w, 11,
                                    boxstyle="round,pad=0.5,rounding_size=1.0",
                                    linewidth=1.8, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y, text, ha="center", va="center",
                fontsize=9.5, color=INK)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.7, y), xytext=(x + w + 0.7, y),
                        arrowprops=dict(arrowstyle="-|>", lw=2, color="#444"))
    if note:
        x = x0 + n * (w + gap)
        ax.text(x, y, note, ha="left", va="center", fontsize=9,
                color=note_color, wrap=True)


def main():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 60)
    ax.axis("off")

    ax.text(2, 55, "a  Conventional supervised detection",
            fontsize=13, fontweight="bold", color=INK)
    draw_chain(ax, 46,
               ["Labeled mounds\nfrom the target region\n(tens to hundreds needed)",
                "Train a\nregion-specific\ndetector",
                "Detect within the\nsame region"],
               RED_EDGE, RED_FILL,
               note="✗  bounded by known\nsites; transfers poorly\nto new regions",
               note_color=RED_EDGE)

    ax.plot([2, 108], [36, 36], ls="--", lw=1, color="#aaa")

    ax.text(2, 31, "b  Label-free / zero-shot transfer  (this work)",
            fontsize=13, fontweight="bold", color=INK)
    draw_chain(ax, 21,
               ["Abundant generic terrain\n(New York State LiDAR)\n+ deterministic geomorphons",
                "Train once, with\nno target-region\nlabels",
                "Apply to a NEW region\n(Yazoo Basin)\nwith no local labels",
                "Ranked candidates\nfor expert review"],
               GREEN_EDGE, GREEN_FILL)

    ax.text(55, 6,
            "✓  the training signal is decoupled from the target landscape, "
            "so it scales to a finite archaeological record",
            ha="center", fontsize=10.5, color=GREEN_EDGE)

    os.makedirs("docs/figures", exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
