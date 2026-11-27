"""Figures for semantic_templates/Analogy_Test_Results.md (test 2, 2026-11-26).

Reads results/template_matcher/template_analogy_{np,g2,l4}.json (debug/template_analogy_analysis.py, primary development set).
Writes into figures/template_matcher/:
  at_foil.png    per model, the foil reading (restatement against its matched foil; chance 0.5) by level, with 95% intervals,
                 for token identity, the embedding, the early state, all states, the path and the wells; C2' as its own column
  at_gain.png    per model, every reading minus the embedding on the foil reading, by level; a dot marks an interval that
                 excludes zero
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY = "#1F2937", "#6B7280", "#9CA3AF"
HERE = Path(__file__).resolve().parent; OUT = HERE / "template_matcher"; RES = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/template_matcher"
MODELS = (("np", "SR2-NP"), ("g2", "G2"), ("l4", "L4"))
LV = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "C1", "C2", "C2'", "C3", "C4"]
READ = (("lex", "token identity", GREY), ("h0", "embedding", INK), ("h1", "early state (after layer 0)", "#3B82F6"),
        ("states", "all states", "#059669"), ("path", "path (displacements)", "#F59E0B"), ("wells", "wells (well frames)", "#DC2626"))
Q = {s: json.loads((RES / f"template_analogy_{s}.json").read_text())["original"] for s, _ in MODELS}
get = lambda o, L: o["C2"]["foil C2'"] if L == "C2'" else o[L]["foil"]
despine = lambda ax: ax.spines[["top", "right"]].set_visible(False)

fig, axs = plt.subplots(3, 1, figsize=(13, 11), sharex=True); x = np.arange(len(LV)); w = 0.13
for ax, (s, nm) in zip(axs, MODELS):
    for j, (k, lab, c) in enumerate(READ):
        v = np.array([get(Q[s][k], L) for L in LV]); xi = x + (j - 2.5) * w
        ax.errorbar(xi, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=4.5, color=c, ecolor=c, elinewidth=1, capsize=0, label=lab)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--"); ax.axvline(6.5, color=GREY, lw=0.6)
    ax.set_ylim(-0.03, 1.05); ax.set_ylabel("restatement beats\nits foil (share)"); ax.set_title(nm, loc="left"); despine(ax)
axs[0].text(3, 1.02, "analogy levels", ha="center", fontsize=9, color=MUTED); axs[0].text(9, 1.02, "in-context levels", ha="center", fontsize=9, color=MUTED)
axs[-1].set_xticks(x, LV); h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, ncol=6, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "at_foil.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(15, 6.2))
for ax, (s, nm) in zip(axs, MODELS):
    names = [k for k in Q[s] if k not in ("h0",)]
    D = np.array([[(Q[s][k]["C2"]["minus h0"]["foil C2'"] if L == "C2'" else Q[s][k][L]["minus h0"]["foil"])[0] for L in LV] for k in names])
    S = np.array([[(lambda d: d[1] > 0 or d[2] < 0)(Q[s][k]["C2"]["minus h0"]["foil C2'"] if L == "C2'" else Q[s][k][L]["minus h0"]["foil"]) for L in LV] for k in names])
    im = ax.imshow(D, cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="auto")
    yy, xx = np.nonzero(S); ax.scatter(xx, yy, s=10, color=INK)
    ax.set_xticks(range(len(LV)), LV, fontsize=8); ax.set_yticks(range(len(names)), names, fontsize=8); ax.set_title(nm, loc="left")
fig.colorbar(im, ax=axs, shrink=0.8, label="foil reading minus the embedding's (dot: interval excludes 0)")
fig.savefig(OUT / "at_gain.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote at_foil.png, at_gain.png")
