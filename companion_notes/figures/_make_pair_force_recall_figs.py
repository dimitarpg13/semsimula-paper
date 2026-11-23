"""Figures for productization/Pair_Force_Recall_Split_FO_b_PMX_K16.md.

Generates three PNGs into figures/pair_force_recall/:

  pfr_where.png      -- the cost of removing the whole pair force, per token, at near recall, far recall, control
                        positions and over all positions, for SR2, PMX-K16 and FO-b-PMX-K16; beside it, the cost of
                        removing only the earlier occurrence from the pair force's sources (both layers summed).
  pfr_layers.png     -- the whole-pair-force cost split by layer, at far recall and at control positions.
  pfr_selection.png  -- how often the pair force's top-16 router selects the earlier occurrence, against the chance
                        rate, per layer and band.

Numbers are read from notebooks/conservative_arch/scaleup/results/recall_pathway/recall_pathway_
{sr2_owt_phisplit,pmxk16_owt,fobk16_owt}.json (debug/recall_pathway_probe.py, 2026-11-22).
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

BLUE = "#3B82F6"
PURPLE = "#8B5CF6"
GREEN = "#22C55E"
GREY = "#9CA3AF"

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent / "notebooks/conservative_arch/scaleup/results/recall_pathway"
OUT = HERE / "pair_force_recall"
OUT.mkdir(exist_ok=True)

MODELS = [("SR2", "sr2_owt_phisplit", GREY), ("PMX-K16", "pmxk16_owt", BLUE), ("FO-b-PMX-K16", "fobk16_owt", PURPLE)]
R = {name: json.loads((RES / f"recall_pathway_{stem}.json").read_text()) for name, stem, _ in MODELS}


def arm(name, a, c):
    return R[name]["arms"][a][c]          # [mean, se] in nats


def where():
    cls = [("near", "near recall"), ("far", "far recall"), ("control", "control"), ("all", "all positions")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.45, 1]})
    x = np.arange(len(cls)); w = 0.26
    for i, (name, _, col) in enumerate(MODELS):
        m = [arm(name, "phioff", c)[0] for c, _ in cls]; e = [arm(name, "phioff", c)[1] for c, _ in cls]
        ax.bar(x + (i - 1) * w, m, w, yerr=e, color=col, capsize=3, label=name)
    ax.set_xticks(x, [l for _, l in cls]); ax.set_ylabel("loss change, nats per token")
    ax.set_title("(a) Removing the whole pair force"); ax.legend(frameon=False, loc="upper right")
    ax.axhline(0, color="black", lw=0.6)
    # (b) at far recall: the whole pair force against the earlier occurrence only
    for i, (name, _, col) in enumerate(MODELS):
        whole = arm(name, "phioff", "far")[0]
        only_p = arm(name, "phi@0", "far")[0] + arm(name, "phi@1", "far")[0]
        bx.bar(i - 0.18, whole, 0.34, color=col)
        bx.bar(i + 0.18, only_p, 0.34, color=col, alpha=0.45, hatch="//", edgecolor="white")
        bx.text(i + 0.18, only_p + 0.0012, f"{100 * only_p / whole:.0f}%", ha="center", va="bottom", fontsize=9)
    bx.set_xticks(range(len(MODELS)), [n for n, _, _ in MODELS])
    bx.set_ylabel("loss change at far recall, nats per token")
    bx.set_title("(b) Far recall: whole force or one source")
    bx.legend(handles=[Patch(facecolor=GREY, label="whole pair force removed"),
                       Patch(facecolor=GREY, alpha=0.45, hatch="//", edgecolor="white",
                             label="only the earlier occurrence removed")], frameon=False, loc="upper left", fontsize=9)
    bx.set_ylim(0, 0.058)
    fig.tight_layout(); fig.savefig(OUT / "pfr_where.png"); plt.close(fig)


def layers():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, (c, title) in zip(axes, [("far", "(a) Far recall positions"), ("control", "(b) Control positions")]):
        x = np.arange(len(MODELS))
        l0 = [arm(n, "phioff@0", c)[0] for n, _, _ in MODELS]; l1 = [arm(n, "phioff@1", c)[0] for n, _, _ in MODELS]
        both = [arm(n, "phioff", c)[0] for n, _, _ in MODELS]
        ax.bar(x - 0.27, l0, 0.27, color=GREEN, label="layer 0 only")
        ax.bar(x, l1, 0.27, color=BLUE, label="layer 1 only")
        ax.bar(x + 0.27, both, 0.27, color=PURPLE, label="both layers")
        ax.set_xticks(x, [n for n, _, _ in MODELS]); ax.axhline(0, color="black", lw=0.6); ax.set_title(title)
    axes[0].set_ylabel("loss change, nats per token"); axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "pfr_layers.png"); plt.close(fig)


def selection():
    keys = [("near@0", "layer 0, near"), ("far@0", "layer 0, far"), ("near@1", "layer 1, near"), ("far@1", "layer 1, far")]
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    x = np.arange(len(keys)); w = 0.22
    for i, (name, _, col) in enumerate(MODELS):
        s = R[name]["selection"]
        ax.bar(x + (i - 1) * w, [s[k]["selected"][0] for k, _ in keys], w, yerr=[s[k]["selected"][1] for k, _ in keys],
               color=col, capsize=3, label=name)
    ch = [R["SR2"]["selection"][k]["chance"] for k, _ in keys]
    for xi, c in zip(x, ch):
        ax.plot([xi - 1.6 * w, xi + 1.6 * w], [c, c], color="black", lw=1.6, ls="--")
    ax.plot([], [], color="black", ls="--", label="chance: 16 / (sources the query can select)")
    ax.set_xticks(x, [l for _, l in keys]); ax.set_ylabel("share of recall pairs")
    ax.set_title("How often the router selects the earlier occurrence")
    ax.legend(frameon=False, loc="upper center", fontsize=9, ncol=2); ax.set_ylim(0, 0.34)
    fig.tight_layout(); fig.savefig(OUT / "pfr_selection.png"); plt.close(fig)


if __name__ == "__main__":
    where(); layers(); selection()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
