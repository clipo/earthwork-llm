#!/usr/bin/env python3
"""Figure 3: the screening workflow as the three-stage pipeline argued in the
Discussion. Stage 1 is morphological and exhaustive (it flags earthworks of
any age), stage 2 is contextual and conservative (it flags rather than
deletes), stage 3 is manual evaluation of the concentrated candidate list.
No caption text is baked into the image.

    python scripts/generate_workflow_fig.py
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "docs/figures/fig4_workflow.png"

INK = "#2c3e50"
GOOD = "#1e8449"
WARN = "#b9770e"
BAD = "#922b21"

STAGES = [
    ("STAGE 1 · DETECT", "#eaf2f8", "#2471a3",
     "morphological, exhaustive",
     ["USGS 3DEP bare-earth DEM (public, 1 m)",
      "geomorphon landform classification",
      "structural combinations → compact candidates",
      "finds earthworks of ANY age:",
      "age is not present in shape"]),
    ("STAGE 2 · SCREEN", "#fdf6ec", "#b9770e",
     "contextual, conservative",
     ["False Positive Shield:",
      "NLCD land cover · footprint linearity ·",
      "mapped modern features (OSM / quads)",
      "optional model reading of earthwork form",
      "flags likely-modern; never silently deletes"]),
    ("STAGE 3 · EVALUATE", "#eafaf1", "#1e8449",
     "manual",
     ["analyst adjudicates the concentrated,",
      "ranked candidate list",
      "keep / downgrade / reject each reading",
      "field verification where possible"]),
]


def main():
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    w, gap, x0, y0, h = 28, 4.5, 1.5, 6, 28
    for i, (title, fill, edge, sub, lines) in enumerate(STAGES):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h,
                                    boxstyle="round,pad=0.7,rounding_size=1.2",
                                    linewidth=2.2, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y0 + h - 2.5, title, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=edge)
        ax.text(x + w / 2, y0 + h - 5.8, sub, ha="center", va="center",
                fontsize=9.5, style="italic", color="#555")
        for j, line in enumerate(lines):
            ax.text(x + w / 2, y0 + h - 9.5 - j * 3.4, line, ha="center",
                    va="center", fontsize=9, color=INK)
        if i < len(STAGES) - 1:
            ax.annotate("", xy=(x + w + gap - 0.6, y0 + h / 2),
                        xytext=(x + w + 0.6, y0 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=2.6, color=INK))

    # verdict chips under stage 2
    x2 = x0 + (w + gap)
    for k, (lab, col) in enumerate([("KEEP", GOOD), ("FLAG", WARN), ("REJECT", BAD)]):
        cx = x2 + 4 + k * 7.2
        ax.add_patch(FancyBboxPatch((cx, 1.0), 6.2, 3.2,
                                    boxstyle="round,pad=0.3", linewidth=1.4,
                                    edgecolor=col, facecolor="white"))
        ax.text(cx + 3.1, 2.6, lab, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=col)

    os.makedirs("docs/figures", exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
