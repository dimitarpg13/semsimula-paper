"""Figures for step 0 of the hybrid template matchers (semantic_templates/Hybrid_Step0_Complementarity.md), 2026-11-26.

Reads results/template_matcher/template_hybrid_check_{np,g2,l4}.json (debug/template_hybrid_check.py). Writes into figures/template_matcher/:
  hyb_overlap.png    which paraphrases each reading catches at its own 1% near-miss threshold: both, embedding only, layer only, neither
  hyb_residual.png   cross-fitted separation of each paraphrase kind from its near miss, and of the control pair, in the embedding's
                     coordinates and in the embedding's plus the joint states' (the gap is the residual separation)
Step 1a (semantic_templates/Hybrid_Step1a_Fitted_Gate.md), from template_gate_fit_*.json and template_gate_diag_*.json:
  hyb_gate_auroc.png      per pair, the matcher's AUROC against the fitted gate's held-out AUROC (embedding, embedding + all states)
  hyb_gate_separation.png the pooled-covariance squared separation against the separation the two kinds have along the direction
  hyb_gate_pass.png       the pooled gates: paraphrases caught at 1% near misses, and what else passes the same threshold
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY, BLUE, AMBER, LIGHT = "#1F2937", "#6B7280", "#9CA3AF", "#3B82F6", "#F59E0B", "#E5E7EB"
HERE = Path(__file__).resolve().parent; OUT = HERE / "template_matcher"; RES = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/template_matcher"
MODELS = (("np", "SR2-NP"), ("g2", "G2"), ("l4", "L4"))
Q = {s: json.loads((RES / f"template_hybrid_check_{s}.json").read_text()) for s, _ in MODELS}
despine = lambda ax: ax.spines[["top", "right"]].set_visible(False)
save = lambda fig, n: (fig.savefig(OUT / n, bbox_inches="tight", facecolor="white"), plt.close(fig))
SNAME = lambda k, L: f"state after layer {int(k[1:]) - 1}"

# 1. overlap
rows = []
for s, nm in MODELS:
    q = Q[s]; b = q["best_single"]
    for k, lab in ((b, SNAME(b, q["L"])), ("states", "all states"), ("path", "path")):
        rows.append((f"{nm}: {lab}", q["readings"][k]))
fig, ax = plt.subplots(figsize=(11, 5.6)); y = np.arange(len(rows))[::-1]
for yy, (lab, r) in zip(y, rows):
    left = 0
    for key, col in (("both", BLUE), ("emb_only", GREY), ("layer_only", AMBER), ("neither", LIGHT)):
        ax.barh(yy, r[key], left=left, color=col, height=.7, edgecolor="white", linewidth=2); left += r[key]
    ax.text(1.01, yy, f"either at 1%: {r['either_matched']:.2f}", va="center", fontsize=9.5, color=INK)
ax.set_yticks(y, [r[0] for r in rows], fontsize=10); ax.set_xlim(0, 1); ax.set_xlabel("share of the 1,320 test paraphrases")
for i in (3, 6): ax.axhline(len(rows) - i - .5, color=MUTED, lw=.8)
hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, GREY, AMBER, LIGHT)]
ax.legend(hs, ["caught by both", "embedding only", "layer reading only", "neither"], frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(.5, 1.0), fontsize=9.5)
despine(ax); ax.set_title("Which paraphrases each reading catches, each at its own 1% near-miss threshold", pad=28)
save(fig, "hyb_overlap.png")

# 2. residual separation, joint states
pairs = ("MP", "MT", "DS1", "DS2", "control MP vs P")
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, gridspec_kw={"wspace": .08}); x = np.arange(len(pairs))
for ax, (s, nm) in zip(axs, MODELS):
    rs = Q[s]["readings"]["states"]["residual_separation"]
    e = [rs[p]["emb"] for p in pairs]; j = [rs[p]["joint"] for p in pairs]
    ax.bar(x - .2, e, .38, color=GREY, label="embedding"); ax.bar(x + .2, j, .38, color=BLUE, label="embedding + all states")
    for xx, v in zip(x + .2, j): ax.text(xx, v + .8, f"{v:.0f}", ha="center", fontsize=8.5, color=INK)
    for xx, v in zip(x - .2, e): ax.text(xx, v + .8, f"{v:.0f}", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(x, ["MP", "MT", "DS1", "DS2", "control"], fontsize=10); ax.set_title(nm, fontsize=11.5); despine(ax)
axs[0].set_ylabel("squared separation (cross-fitted)"); axs[0].legend(frameon=False, fontsize=9.5, loc="upper left")
fig.suptitle("Paraphrase against near miss, and the control pair (MP against the affirmed disjunct)", fontweight="bold", y=1.02)
save(fig, "hyb_residual.png")

# 3. step 1a: per pair AUROC, matcher against fitted gate
G = {s: json.loads((RES / f"template_gate_fit_{s}.json").read_text()) for s, _ in MODELS}
D = {s: json.loads((RES / f"template_gate_diag_{s}.json").read_text()) for s, _ in MODELS}
PR = ("MP", "MT", "DS1", "DS2")
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, gridspec_kw={"wspace": .08}); x = np.arange(len(PR))
for ax, (s, nm) in zip(axs, MODELS):
    vals = ((GREY, "matcher (embedding)", [D[s]["matcher"][t]["auroc"] for t in PR]), (BLUE, "gate: embedding", [G[s]["readings"]["h0"]["pairs"][t]["held_auroc"] for t in PR]),
            (AMBER, "gate: embedding + all states", [G[s]["readings"]["h0+states"]["pairs"][t]["held_auroc"] for t in PR]))
    for i, (c, lab, v) in enumerate(vals):
        ax.bar(x + (i - 1) * .27, v, .25, color=c, label=lab)
        for xx, vv in zip(x + (i - 1) * .27, v): ax.text(xx, vv + .004, f"{vv:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, PR); ax.set_ylim(.7, 1.0); ax.set_title(nm, fontsize=11.5); despine(ax)
axs[0].set_ylabel("AUROC, paraphrase against its near miss"); axs[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("A fitted direction orders paraphrases against near misses better than the matcher (gate scored on held-out clauses)", fontweight="bold", y=1.02)
save(fig, "hyb_gate_auroc.png")

# 4. pooled-covariance separation against the separation along the direction
fig, ax = plt.subplots(figsize=(6.4, 5.4)); mk = {"np": "o", "g2": "s", "l4": "^"}
for s, nm in MODELS:
    for rn, r in D[s]["readings"].items():
        c = BLUE if rn == "h0" else (AMBER if rn == "h0+states" else GREY)
        ax.scatter([r[t]["sq_sep"] for t in PR], [r[t]["sq_sep_emp"] for t in PR], marker=mk[s], s=46, color=c, edgecolor="white", linewidth=1.5, zorder=3)
ax.plot([0, 32], [0, 32], color=MUTED, lw=1, ls="--"); ax.text(20, 21.5, "equal", color=MUTED, fontsize=9, rotation=36)
ax.set_xlim(0, 32); ax.set_ylim(0, 32); ax.set_xlabel("squared separation, pooled within-kind covariance"); ax.set_ylabel("squared separation along the direction")
hs = [plt.Line2D([], [], ls="", marker=mk[s], color=INK, label=nm) for s, nm in MODELS] + [plt.Line2D([], [], ls="", marker="o", color=c, label=l) for c, l in ((BLUE, "embedding"), (GREY, "one state, or embedding + one state"), (AMBER, "embedding + all states"))]
ax.legend(handles=hs, frameon=False, fontsize=8.5, loc="upper left"); despine(ax)
ax.set_title("The pooled covariance overstates how far apart the two kinds are", fontsize=11)
save(fig, "hyb_gate_separation.png")

# 5. pooled gates: what passes
ME = (("para_1pct", "paraphrases caught", BLUE), ("exact_pass", "exact forms pass", GREY), ("fallacy_pass", "fallacies pass", AMBER), ("other subject_pass", "other subject passes", INK))
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True, gridspec_kw={"wspace": .08}); GR = (("h0", "narrow"), ("h0", "broad"), ("h0+states", "narrow"), ("h0+states", "broad")); x = np.arange(len(GR))
for ax, (s, nm) in zip(axs, MODELS):
    for i, (k, lab, c) in enumerate(ME):
        v = [G[s]["readings"][rn]["pooled"][gt][k] for rn, gt in GR]; ax.bar(x + (i - 1.5) * .2, v, .19, color=c, label=lab)
    h0 = json.loads((RES / f"template_depth_{s}.json").read_text())["primary"]["h0"]["para at near-miss rate"]["0.01"]
    ax.axhline(max(h0, *(r["para_1pct"] for r in Q[s]["readings"].values())), color=MUTED, lw=.8, ls=":")      # the best matcher of step 0
    ax.set_xticks(x, ["embedding\nnarrow", "embedding\nbroad", "+ all states\nnarrow", "+ all states\nbroad"], fontsize=9); ax.set_ylim(0, 1.05); ax.set_title(nm, fontsize=11.5); despine(ax)
axs[0].set_ylabel("share, at 1% of near misses"); fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=9.5, ncol=4, loc="lower center", bbox_to_anchor=(.5, .99))
fig.suptitle("The narrow gate catches more paraphrases but lets most fallacies and other subjects through (dotted: the best matcher)", fontweight="bold", y=1.09)
save(fig, "hyb_gate_pass.png")
