"""Figure for semantic_templates/Summary_State_Test.md (task A, 2026-11-27).

Reads results/template_matcher/template_summary_state_{np,g2,l4}.json (debug/template_summary_state.py). Writes into
figures/template_matcher/:
  sa_scores.png   per model, the rank of the right template on unseen wordings (chance 0.25) and the AUROC of valid forms
                  against their twins (chance 0.5), for the length, the embedding at "Therefore", each layer state and all states
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY, BLUE, GREEN = "#1F2937", "#6B7280", "#9CA3AF", "#3B82F6", "#059669"
HERE = Path(__file__).resolve().parent; OUT = HERE / "template_matcher"; RES = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/template_matcher"
MODELS = (("np", "SR2-NP"), ("g2", "G2"), ("l4", "L4"))
Q = {s: json.loads((RES / f"template_summary_state_{s}.json").read_text()) for s, _ in MODELS}
color = lambda k: GREY if k == "length" else INK if k == "h0" else GREEN if k == "states" else BLUE
label = lambda k: {"length": "length", "h0": "embedding", "states": "all states"}.get(k, f"state {k}")

fig, axs = plt.subplots(2, 3, figsize=(14, 7.5), sharey="row")
for col, (s, nm) in enumerate(MODELS):
    names = list(Q[s])
    for row, (key, chance, yl) in enumerate((("rank", 0.25, "right template ranked first\n(unseen wordings)"), ("twin", 0.5, "valid form against its twin\n(AUROC, known wordings)"))):
        ax = axs[row, col]; v = np.array([Q[s][k][key] for k in names]); x = np.arange(len(names))
        ax.bar(x, v[:, 0], 0.7, color=[color(k) for k in names])
        ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor=MUTED, elinewidth=1)
        ax.axhline(chance, color=MUTED, lw=0.8, ls="--"); ax.set_xticks(x, [label(k) for k in names], rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1); ax.spines[["top", "right"]].set_visible(False)
        for xi, vi in zip(x, v[:, 0]): ax.text(xi, vi + 0.02, f"{vi:.2f}", ha="center", fontsize=8, color=INK)
        if col == 0: ax.set_ylabel(yl)
        if row == 0: ax.set_title(nm, loc="left")
fig.tight_layout(); fig.savefig(OUT / "sa_scores.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote sa_scores.png")
