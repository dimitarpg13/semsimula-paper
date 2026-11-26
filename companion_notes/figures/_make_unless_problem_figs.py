"""Figures for semantic_templates/Connectives_With_a_Hidden_Negation.md (2026-11-25).

Reads results/template_matcher/: template_connective_position_sr2_2b.json, template_dynamics_readings_sr2_2b.json,
template_dynamics_without_unless_sr2_2b.json, template_dynamics_unless_own_sr2_2b.json. Writes into figures/template_matcher/:
  un_axis.png     where each wording's connective sits between the disjunctive (0) and conditional (1) centroids, per vector,
                  in the full fit and with the wording itself left out of the axis
  un_filings.png  three ways of filing "unless": R1 on the other eleven wordings, unseen-wording detection of MP, fires on the
                  denied antecedent
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY, BLUE, AMBER, RED, GREEN = "#1F2937", "#6B7280", "#9CA3AF", "#3B82F6", "#F59E0B", "#EF4444", "#22C55E"
HERE = Path(__file__).resolve().parent; OUT = HERE / "template_matcher"; RES = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/template_matcher"
L = lambda n: json.loads((RES / n).read_text())
POS, RD, NU, OW = L("template_connective_position_sr2_2b.json"), L("template_dynamics_readings_sr2_2b.json"), L("template_dynamics_without_unless_sr2_2b.json"), L("template_dynamics_unless_own_sr2_2b.json")
COND = ["if-then", "when", "whenever", "q-if-p", "assuming", "provided"]; DISJ = ["either-or", "or", "at-least-one", "or-else", "one-holds"]
despine = lambda ax: ax.spines[["top", "right"]].set_visible(False)

# 1. the axis
vec = [("h0", "embedding"), ("h1", "state after layer 0"), ("D1", "displacement, layer 1"), ("h2", "state after layer 1")]
fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True, gridspec_kw={"wspace": .06})
for ax, mode, ttl in ((axs[0], "fit", "Axis fitted on the eleven wordings"), (axs[1], "loo", "Each wording left out of the axis")):
    for i, (k, lab) in enumerate(vec):
        P = POS[k]; get = (lambda w: P[w][1]) if mode == "fit" else (lambda w: P["left out of the axis"][w])
        ax.scatter([get(w) for w in COND], [i] * 6, s=55, color=BLUE, alpha=.8, zorder=3, label="conditional wordings" if i == 0 else None)
        ax.scatter([get(w) for w in DISJ], [i] * 5, s=55, color=AMBER, alpha=.8, zorder=3, label="disjunctive wordings" if i == 0 else None)
        ax.scatter([get("unless")], [i], s=180, marker="*", color=RED, ec=INK, lw=.6, zorder=4, label='"unless"' if i == 0 else None)
        ax.axhline(i, color=GREY, lw=.5, zorder=1)
    ax.axvline(.5, color=INK, lw=.8, ls=":"); ax.set_xlim(-.3, 1.25); ax.set_title(ttl, fontsize=11.5); ax.set_xlabel("connective position: 0 = disjunctive centroid, 1 = conditional centroid"); despine(ax)
axs[0].set_yticks(range(len(vec)), [l for _, l in vec]); axs[0].invert_yaxis(); axs[0].legend(frameon=False, fontsize=9.5, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3)
fig.savefig(OUT / "un_axis.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# 2. three filings
mm = [("h0", "embedding"), ("D1+D2", "displacements\n(A)"), ("h1+h2", "states after\nboth layers")]; F = [("pooled with the disjunctions", RD, GREY), ("left out", NU, BLUE), ("clouds of its own", OW, RED)]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"wspace": .3})
def val(src, n, what):
    q = src[n]
    if what == "R1": return q["R1 unseen"]
    if what == "MP": v = q["unseen detect"]["MP"]; return v[0] if isinstance(v, list) else v
    v = q["unseen fires on"]["cond:DA"]; return v[0] if isinstance(v, list) else v
for ax, (what, ttl) in zip(axs, (("R1", "R1, unseen wordings"), ("MP", "MP detected, unseen wordings"), ("DA", "denied antecedent fires, unseen"))):
    x = np.arange(len(mm))
    for j, (lab, src, col) in enumerate(F):
        y = [val(src, n, what) for n, _ in mm]; ax.bar(x + (j - 1) * .26, y, width=.24, color=col, label=lab if what == "R1" else None)
        for a, b in zip(x + (j - 1) * .26, y): ax.text(a, b + .015, f"{b:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [l for _, l in mm], fontsize=9.5); ax.set_ylim(0, 1.1); ax.set_title(ttl, fontsize=11.5); despine(ax)
fig.legend(frameon=False, fontsize=9.5, loc="lower center", bbox_to_anchor=(.5, -.06), ncol=3); fig.text(.5, -.1, 'SR2-2B, 60 test clauses. R1 and the fallacy fires are over the eleven other wordings when "unless" is left out or has its own clouds.', ha="center", color=MUTED, fontsize=9.5)
fig.savefig(OUT / "un_filings.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
