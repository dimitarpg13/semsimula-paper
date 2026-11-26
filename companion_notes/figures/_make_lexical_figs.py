"""Figures for semantic_templates/Lexical_Baseline.md (2026-11-26).

Reads results/template_matcher/template_lexical_{np,g2,l4}.json (debug/template_lexical_baseline.py). Writes into figures/template_matcher/:
  lex_scores.png     per model, the embedding against token identity (lex), token identity after lower-casing (lexn) and token
                     identity with the embedding's connective (lex+c): R1 on unseen wordings, and paraphrases caught at 1% near misses
  lex_wordings.png   detection of the valid forms by unseen wording, embedding against token identity (SR2-NP)
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY, BLUE, AMBER, GREEN = "#1F2937", "#6B7280", "#9CA3AF", "#3B82F6", "#F59E0B", "#059669"
HERE = Path(__file__).resolve().parent; OUT = HERE / "template_matcher"; RES = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/template_matcher"
MODELS = (("np", "SR2-NP"), ("g2", "G2"), ("l4", "L4")); READ = (("h0", "embedding", INK), ("lex", "token identity", GREY), ("lexn", "token identity, lower-cased", "#D1D5DB"), ("lex+c", "token identity + embedding's connective", AMBER))
Q = {s: json.loads((RES / f"template_lexical_{s}.json").read_text()) for s, _ in MODELS}
despine = lambda ax: ax.spines[["top", "right"]].set_visible(False)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6)); x = np.arange(len(MODELS)); w = 0.2
for ax, (key, title, lim) in zip(axs, (("R1 unseen", "Unseen wordings (R1, without \"unless\")", (0.8, 1.0)), ("para", "Paraphrases caught at 1% near misses", (0, 0.35)))):
    for j, (k, nm, c) in enumerate(READ):
        v = [Q[s][k]["para at near-miss rate"]["0.01"] if key == "para" else Q[s][k][key] for s, _ in MODELS]
        ax.bar(x + (j - 1.5) * w, v, w * 0.92, color=c, label=nm, edgecolor=INK if c == "#D1D5DB" else c, lw=0.5)
        for xi, vi in zip(x + (j - 1.5) * w, v): ax.text(xi, vi + (lim[1] - lim[0]) * 0.01, f"{vi:.3f}", ha="center", fontsize=7.5, color=INK, rotation=90)
    ax.set_xticks(x, [nm for _, nm in MODELS]); ax.set_ylim(*lim); ax.set_title(title); despine(ax)
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT / "lex_scores.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

W = list(Q["np"]["h0"]["unseen by wording"])
fig, ax = plt.subplots(figsize=(12, 4)); x = np.arange(len(W))
for j, (k, nm, c) in enumerate(READ[:2]):
    ax.bar(x + (j - 0.5) * 0.38, [Q["np"][k]["unseen by wording"][wd] for wd in W], 0.36, color=c, label=nm)
ax.set_xticks(x, W, rotation=0, fontsize=9); ax.set_ylim(0, 1.05); ax.set_ylabel("valid forms detected,\nwording left out of fitting"); ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.axvline(5.5, color=GREY, lw=0.8); ax.text(2.5, 1.0, "conditional", ha="center", fontsize=9, color=MUTED); ax.text(8, 1.0, "disjunctive", ha="center", fontsize=9, color=MUTED)
ax.set_title("SR2-NP: the embedding loses \"Q if P\" and \"provided\"; token identity is flat at 0.88 and 0.75"); despine(ax)
fig.savefig(OUT / "lex_wordings.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote lex_scores.png, lex_wordings.png")
