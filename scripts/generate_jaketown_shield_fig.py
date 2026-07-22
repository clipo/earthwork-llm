#!/usr/bin/env python3
"""
Figure: False Positive Shield funnel for the controlled Jaketown rescan.

Reads the rescan output and renders how the raw geomorphon candidates are
reduced to survivors, broken out by rejection reason. Numbers come straight
from data/jaketown_rescan_shielded/regional_detections.csv so the figure
cannot drift from the run.

    python scripts/generate_jaketown_shield_fig.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

CSV = "data/jaketown_rescan_shielded/regional_detections.csv"
OUT = "docs/figures/fig_jaketown_shield_funnel.png"

INK = "#2c3e50"


def main():
    df = pd.read_csv(CSV)
    total = len(df)
    dec = df["shield_decision"].value_counts().to_dict()
    kept = dec.get("keep", 0)
    flagged = dec.get("flag", 0)
    rejected = dec.get("reject", 0)
    survivors = kept + flagged

    rej = df[df["shield_decision"] == "reject"]
    nlcd_dev = rej["shield_reasons"].str.contains("developed land").sum()
    nlcd_water = rej["shield_reasons"].str.contains("open water").sum()
    # A candidate can list two reasons; count linear-only to avoid double count.
    linear_only = rej["shield_reasons"].apply(
        lambda s: "aspect" in s and "developed" not in s and "water" not in s).sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={"width_ratios": [1, 1.2]})

    # --- Left: funnel bars ---------------------------------------------------
    stages = ["Raw geomorphon\ncandidates", "Survivors\n(kept + flagged)"]
    vals = [total, survivors]
    colors = ["#bdc3c7", "#1e8449"]
    bars = ax1.bar(stages, vals, color=colors, edgecolor=INK, width=0.6)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 4, str(v),
                 ha="center", fontsize=13, fontweight="bold")
    ax1.set_ylabel("candidates", fontsize=11)
    ax1.set_ylim(0, total * 1.15)
    ax1.set_title(f"Shield removed {rejected} of {total} "
                  f"({100*rejected/total:.0f}%)", fontsize=12, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)

    # --- Right: rejection reasons -------------------------------------------
    labels = ["Developed land\n(NLCD)", "Open water\n(NLCD)",
              "Linear footprint\n(levee / road)"]
    counts = [nlcd_dev, nlcd_water, linear_only]
    bar_colors = ["#922b21", "#2471a3", "#b9770e"]
    ybars = ax2.barh(labels[::-1], counts[::-1], color=bar_colors[::-1],
                     edgecolor=INK)
    for b, v in zip(ybars, counts[::-1]):
        ax2.text(v + 1, b.get_y() + b.get_height() / 2, str(v),
                 va="center", fontsize=12, fontweight="bold")
    ax2.set_xlabel("candidates rejected", fontsize=11)
    ax2.set_title("Why candidates were rejected", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, max(counts) * 1.18)
    ax2.spines[["top", "right"]].set_visible(False)

    # no baked figure title or caption: captions belong to the manuscript
    os.makedirs("docs/figures", exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")
    print(f"total={total} survivors={survivors} rejected={rejected} "
          f"(nlcd_dev={nlcd_dev} water={nlcd_water} linear_only={linear_only})")


if __name__ == "__main__":
    main()
