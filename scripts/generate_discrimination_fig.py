#!/usr/bin/env python3
"""Figure 11: the age-discrimination test on Eskew's field-verified set
(Section 3.6, Appendix B.3): four zero-shot arms plus the fine-tuned arm.

Panel (a): per test arm, the fraction of field-confirmed mounds kept and the
fraction of field-confirmed modern earthworks rejected. A discriminator needs
both to be high; every zero-shot arm keeps the mounds and fails to reject the
moderns, and only the fine-tuned V10 (rightmost pair) moves the red bar.
Panel (b): arm C MOUND-vote shares per site; the distributions overlap
completely (vote-share AUC 0.33), so the strongest zero-shot protocol carries
no age signal even below the verdict threshold. Panel (c): the same plot for
the fine-tuned V10 (AUC 0.62), the first above-chance age signal in the
series; its own-base control (v9.1 under the identical composite protocol)
stays at chance (AUC 0.48).

All numbers are read from the released run outputs so the figure cannot
drift from the data. No caption text is baked into the image.

    python scripts/generate_discrimination_fig.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = os.environ.get("TERRALLM_DATA_ROOT", ".")
OUT = "docs/figures/fig_discrimination.png"

INK = "#2c3e50"
MOUND_C = "#2471a3"
MODERN_C = "#922b21"


def shield_counts(csv, verdict_col):
    df = pd.read_csv(csv)
    df = df[~df[verdict_col].astype(str).str.startswith("ERROR")]
    mo = df[df.label == 1]
    md = df[df.label == 0]
    kept = (mo[verdict_col] != "REJECT").sum(), len(mo)
    rej = (md[verdict_col] == "REJECT").sum(), len(md)
    return kept, rej


def vlm_counts(csv):
    df = pd.read_csv(csv)
    df = df[df.pred.notna()]
    mo = df[df.label == 1]
    md = df[df.label == 0]
    return (int((mo.pred == 1).sum()), len(mo)), (int((md.pred == 0).sum()), len(md))


def main():
    arms = []  # (label, (mounds kept, n), (moderns rejected, n))

    k, r = shield_counts(f"{T}/data/shield_eskew/shield_eskew_results.csv", "verdict")
    arms.append(("Shield:\nNLCD+linearity", k, r))
    k, r = shield_counts(f"{T}/data/shield_eskew/shield_eskew_proximity.csv", "verdict_full")
    arms.append(("Shield:\n+OSM proximity", k, r))
    k, r = vlm_counts(f"{T}/data/vlm_ablation/ablation_results.csv")
    arms.append(("v9.1:\nimage only", k, r))
    for i in (1, 2, 3):
        k, r = vlm_counts(f"{T}/data/vlm_ablation_context/run{i}_results.csv")
        arms.append((f"v9.1: +context\nreplicate {i}", k, r))
    k, r = vlm_counts(f"{T}/data/vlm_ablation_armc/armc_results.csv")
    arms.append(("v9.1: strongest\nprompt, 9 votes", k, r))
    k, r = vlm_counts(f"{T}/data/v10_eval/eskew_v91same.csv")
    arms.append(("v9.1: composite\nprotocol", k, r))
    k, r = vlm_counts(f"{T}/data/v10_eval/eskew_v10.csv")
    arms.append(("V10: fine-tuned,\nsame protocol", k, r))

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(16.5, 5.2), gridspec_kw={"width_ratios": [2.1, 0.85, 0.85]})

    # --- (a) mounds kept vs moderns rejected, per arm ------------------------
    x = np.arange(len(arms))
    kept_f = [k / n for _, (k, n), _ in arms]
    rej_f = [r / n for _, _, (r, n) in arms]
    w = 0.38
    b1 = ax1.bar(x - w / 2, kept_f, w, color=MOUND_C, edgecolor=INK,
                 label="field-confirmed mounds kept")
    b2 = ax1.bar(x + w / 2, rej_f, w, color=MODERN_C, edgecolor=INK,
                 label="modern earthworks rejected")
    for b, (_, (k, n), _) in zip(b1, arms):
        ax1.text(b.get_x() + w / 2, b.get_height() + 0.02, f"{k}/{n}",
                 ha="center", fontsize=8.5, color=MOUND_C, fontweight="bold")
    for b, (_, _, (r, n)) in zip(b2, arms):
        ax1.text(b.get_x() + w / 2, b.get_height() + 0.02, f"{r}/{n}",
                 ha="center", fontsize=8.5, color=MODERN_C, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([a for a, _, _ in arms], fontsize=7.6, rotation=18, ha="right")
    ax1.set_ylabel("fraction of set", fontsize=10)
    ax1.set_ylim(0, 1.12)
    ax1.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.22, 0.98))
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.text(0.01, 1.06, "a", transform=ax1.transAxes, fontsize=15,
             fontweight="bold", va="top")

    # --- (b, c) vote-share strip plots: v9.1 strongest arm vs V10 -------------
    def share_panel(ax, csv, xlabel, letter):
        df = pd.read_csv(csv)
        df = df[(df.votes_mound + df.votes_notmound) > 0].copy() if "votes_notmound" in df else df
        notcol = "votes_notmound" if "votes_notmound" in df.columns else "votes_not"
        df = df[(df.votes_mound + df[notcol]) > 0].copy()
        df["share"] = df.votes_mound / (df.votes_mound + df[notcol])
        rng = np.random.default_rng(7)
        for label, sub, ypos, color in [
                (f"mounds (n={(df.label == 1).sum()})", df[df.label == 1], 1.0, MOUND_C),
                (f"moderns (n={(df.label == 0).sum()})", df[df.label == 0], 0.0, MODERN_C)]:
            y = ypos + rng.uniform(-0.13, 0.13, len(sub))
            ax.scatter(sub.share, y, s=48, c=color, edgecolor="k",
                       linewidth=0.6, alpha=0.85, zorder=5)
        ax.axvline(0.5, ls="--", lw=1.2, color="#888")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["modern\nearthworks", "confirmed\nmounds"], fontsize=8.5)
        ax.set_ylim(-0.5, 1.75)
        ax.set_xlim(-0.05, 1.08)
        ax.set_xlabel(xlabel, fontsize=9)
        pos = df[df.label == 1].share.values
        neg = df[df.label == 0].share.values
        auc = np.mean([(1.0 if p > n else 0.5 if p == n else 0.0)
                       for p in pos for n in neg])
        ax.text(0.02, -0.4, f"vote-share AUC = {auc:.2f}", fontsize=9, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.01, 1.06, letter, transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top")
        return auc

    auc_b = share_panel(ax2, f"{T}/data/vlm_ablation_armc/armc_results.csv",
                        "MOUND-vote share, v9.1\n(strongest prompt, 9 votes)", "b")
    auc_c = share_panel(ax3, f"{T}/data/v10_eval/eskew_v10.csv",
                        "MOUND-vote share, V10\n(fine-tuned, 5 votes)", "c")

    fig.tight_layout()
    os.makedirs("docs/figures", exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}  (v9.1 AUC {auc_b:.3f}, V10 AUC {auc_c:.3f})")


if __name__ == "__main__":
    main()
