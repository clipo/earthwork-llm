"""Figure 13: the structured reader and the fused triage ranking (Section 3.8).

Panel (a): composite and isolation AUC against desk labels as a function of
context-window size, with bootstrap 95% CIs, from the released sweep CSVs.
Panel (b): the 150 m fusion, standardized computed relief against the model's
isolation score, desk-review plausibles versus rejected candidates.

Reads: data/v10_eval/salience_v2_jaketown_{base,ctx*}.csv and the Jaketown
review verdicts (relief + labels). Writes docs/figures/fig_salience_ranking.png.
"""
from __future__ import annotations
import csv
import random
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
VERDICTS = __import__("os").environ.get("JAKETOWN_VERDICTS", "data/review/jaketown_verdicts.csv")

BLUE, ORANGE, GRAY = "#1f77b4", "#ff7f0e", "#8a8a8a"
WINDOWS = [(150, "ctx150"), (300, "ctx300"), (600, "base"), (1200, "ctx1200"), (2400, "ctx2400")]


def load(path):
    return [(r["id"], int(r["label"]),
             {k: (float(r[k]) if r[k] != "" else None) for k in ("isolation", "composite")})
            for r in csv.DictReader(open(path)) if r["composite"] != ""]


def auc(pairs, key):
    pos = [d[key] for _, lab, d in pairs if lab == 1 and d[key] is not None]
    neg = [d[key] for _, lab, d in pairs if lab == 0 and d[key] is not None]
    if not pos or not neg:
        return None
    return sum((1.0 if p > n else 0.5 if p == n else 0) for p in pos for n in neg) / (len(pos) * len(neg))


def boot_ci(pairs, key, n=3000):
    pos = [p for p in pairs if p[1] == 1]
    neg = [p for p in pairs if p[1] == 0]
    vals = sorted(a for a in (auc([random.choice(pos) for _ in pos] + [random.choice(neg) for _ in neg], key)
                              for _ in range(n)) if a is not None)
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def zscore(v):
    m, s = statistics.mean(v), statistics.pstdev(v) or 1.0
    return [(x - m) / s for x in v]


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=200)

    # ---- (a) AUC vs context window ----
    series = {"composite": (BLUE, "o", "-"), "isolation": (ORANGE, "s", "--")}
    for key, (color, marker, ls) in series.items():
        xs, ys, lo, hi = [], [], [], []
        for w, tag in WINDOWS:
            pairs = load(ROOT / f"data/v10_eval/salience_v2_jaketown_{tag}.csv")
            a = auc(pairs, key)
            ci_lo, h = boot_ci(pairs, key)
            xs.append(w)
            ys.append(a)
            lo.append(a - ci_lo)
            hi.append(h - a)
        ax1.errorbar(xs, ys, yerr=[lo, hi], color=color, marker=marker, ls=ls, lw=2,
                     ms=7, capsize=3, elinewidth=1, ecolor=color, alpha=0.95, label=key)
        ax1.annotate(key, (xs[0], ys[0]), textcoords="offset points",
                     xytext=(10, 8 if key == "isolation" else -14), color=color, fontsize=10)
    ax1.axhline(0.5, color=GRAY, ls=":", lw=1.2)
    ax1.text(2400, 0.505, "chance", color=GRAY, fontsize=9, ha="right", va="bottom")
    ax1.set_xscale("log")
    ax1.set_xticks([w for w, _ in WINDOWS])
    ax1.set_xticklabels([str(w) for w, _ in WINDOWS])
    ax1.set_xlabel("Context window (m)")
    ax1.set_ylabel("AUC against desk-review labels")
    ax1.set_ylim(0.25, 0.95)
    ax1.grid(alpha=0.25, lw=0.5)
    ax1.set_title("(a) Reading is a feature-scale activity", fontsize=11, loc="left")
    ax1.legend(frameon=False, fontsize=9, loc="lower left")

    # ---- (b) 150 m fusion scatter ----
    rev = {r["id"]: (1 if r["verdict"] == "uncertain" else 0, float(r["lrm_wide_m"]))
           for r in csv.DictReader(open(VERDICTS))}
    v = {rid: d for rid, _, d in load(ROOT / "data/v10_eval/salience_v2_jaketown_ctx150.csv")}
    ids = [i for i in rev if i in v and v[i]["isolation"] is not None]
    labels = [rev[i][0] for i in ids]
    zr = zscore([rev[i][1] for i in ids])
    zi = zscore([v[i]["isolation"] for i in ids])
    # isolation scores are integers 0-5: add small seeded vertical jitter for
    # legibility (disclosed in the caption); relief (x) is plotted as measured
    jit = [random.uniform(-0.07, 0.07) for _ in zi]
    zi_j = [a + b for a, b in zip(zi, jit)]
    rej_x = [a for a, lab in zip(zr, labels) if lab == 0]
    rej_y = [a for a, lab in zip(zi_j, labels) if lab == 0]
    pla_x = [a for a, lab in zip(zr, labels) if lab == 1]
    pla_y = [a for a, lab in zip(zi_j, labels) if lab == 1]
    ax2.scatter(rej_x, rej_y, s=26, facecolors="none", edgecolors=GRAY, lw=1.1,
                alpha=0.75, label="rejected (n=72)")
    ax2.scatter(pla_x, pla_y, s=75, marker="^", color=BLUE, edgecolors="white",
                lw=0.8, label="desk-review plausible (n=8)", zorder=3)
    ymax = max(zi_j)
    ax2.set_ylim(min(zi_j) - 0.35, ymax + 1.0)
    ax2.set_xlabel("Computed relief (z-score)")
    ax2.set_ylabel("Model isolation score at 150 m (z-score)")
    ax2.grid(alpha=0.25, lw=0.5)
    ax2.set_title("(b) Evidence streams stack", fontsize=11, loc="left")
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    ax2.text(0.98, 0.03, "relief 0.72   isolation 0.69   fused 0.79",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=9, color="#444444")

    fig.tight_layout()
    out = ROOT / "docs/figures/fig_salience_ranking.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
