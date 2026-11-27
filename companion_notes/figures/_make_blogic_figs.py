"""Figure for semantic_templates/BLogic_Test.md (task B-logic, 2026-11-27).

Reads results/template_matcher/template_blogic_{np,g2,l4}.json (geometry matcher, primary development set) and
template_summary_state_blogic_{np,g2,l4}.json (the state at "Therefore"). Writes into figures/template_matcher/:
  bl_scores.png   per model, the foil and twin readings (chance 0.5) for the embedding, the early state, all states and the path
                  read by the geometry matcher, and the early state at "Therefore"; explicit contexts filled, natural hollow
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY = "#1F2937", "#6B7280", "#9CA3AF"
HERE = Path(__file__).resolve().parent; OUT = HERE / "template_matcher"; RES = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/template_matcher"
MODELS = (("np", "SR2-NP"), ("g2", "G2"), ("l4", "L4"))
G = {s: json.loads((RES / f"template_blogic_{s}.json").read_text())["original"] for s, _ in MODELS}
T = {s: json.loads((RES / f"template_summary_state_blogic_{s}.json").read_text()) for s, _ in MODELS}
READ = (("embedding", INK, lambda s, L, rd: G[s]["h0"][L][rd]), ("early state", "#3B82F6", lambda s, L, rd: G[s]["h1"][L][rd]),
        ("all states", "#059669", lambda s, L, rd: G[s]["states"][L][rd]), ("path", "#F59E0B", lambda s, L, rd: G[s]["path"][L][rd]),
        ('early state at "Therefore"', "#DC2626", lambda s, L, rd: T[s]["h1"]["analogy"][f"{L} {rd}"]))

fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for ax, (s, nm) in zip(axs, MODELS):
    for gi, rd in enumerate(("foil", "twin")):
        for j, (lab, c, get) in enumerate(READ):
            for k, (L, fill) in enumerate((("BX", True), ("BN", False))):
                v = get(s, L, rd); x = gi * 6 + j + (k - 0.5) * 0.32
                ax.errorbar(x, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", ms=6, color=c, mfc=c if fill else "white", ecolor=c, elinewidth=1,
                            label=(lab if (gi == 0 and k == 0) else None))
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--"); ax.set_xticks([2, 8], ["restatement against foil", "valid form against twin"])
    ax.set_ylim(0.25, 1.0); ax.set_title(nm, loc="left"); ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("valid version wins (share)")
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, ncol=5, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.06))
fig.text(0.99, 0.01, "filled: explicit contexts   hollow: natural contexts", ha="right", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT / "bl_scores.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote bl_scores.png")

# the control (2026-11-27): contexts that name both clauses, variants a and b
GC = {s: json.loads((RES / f"template_blogic_control_{s}.json").read_text())["original"] for s, _ in MODELS}
TC = {s: json.loads((RES / f"template_summary_state_blogic_control_{s}.json").read_text()) for s, _ in MODELS}
READC = (("early state (matcher)", "#3B82F6", lambda s, L, rd: GC[s]["h1"][L][rd]), ("all states (matcher)", "#059669", lambda s, L, rd: GC[s]["states"][L][rd]),
         ('early state at "Therefore"', "#DC2626", lambda s, L, rd: TC[s]["h1"]["analogy"][f"{L} {rd}"]),
         ('layer-1 state at "Therefore"', "#7C3AED", lambda s, L, rd: TC[s]["h2"]["analogy"][f"{L} {rd}"]))
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for ax, (s, nm) in zip(axs, MODELS):
    for gi, rd in enumerate(("foil", "twin")):
        for j, (lab, c, get) in enumerate(READC):
            for k, (L, fill) in enumerate((("BCa", True), ("BCb", False))):
                v = get(s, L, rd); x = gi * 5 + j + (k - 0.5) * 0.32
                ax.errorbar(x, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", ms=6, color=c, mfc=c if fill else "white", ecolor=c, elinewidth=1,
                            label=(lab if (gi == 0 and k == 0) else None))
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--"); ax.set_xticks([1.5, 6.5], ["restatement against foil", "valid form against twin"])
    ax.set_ylim(0.15, 1.02); ax.set_title(nm, loc="left"); ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("valid version wins (share)")
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, ncol=4, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.06))
fig.text(0.99, 0.01, "filled: variant a (bound clause first)   hollow: variant b (bound clause last)", ha="right", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT / "bl_control.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote bl_control.png")
