"""Figures for semantic_templates/The_Embedding_Template_Matcher.md (2026-11-25).

Reads results/template_matcher/template_embedding_walkthrough_sr2_2b.json (debug/template_embedding_walkthrough.py) and writes
into figures/template_matcher/:
  te_gram.png      the particle cosines of one premise in the embedding: modus ponens, its paraphrase, its near miss
  te_coords.png    the three coordinates that carry the decision, per kind of premise, on the test clauses
  te_decision.png  the worked premises in the cloud subspace, with the cloud centres
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY = "#1F2937", "#6B7280", "#9CA3AF"
FC = {"MP": "#3B82F6", "MT": "#22C55E", "DS1": "#F59E0B", "DS2": "#8B5CF6"}
HERE = Path(__file__).resolve().parent; OUT = HERE / "template_matcher"
O = json.loads((HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/template_matcher/template_embedding_walkthrough_sr2_2b.json").read_text()); W = O["work"]
despine = lambda ax: ax.spines[["top", "right"]].set_visible(False)
save = lambda fig, n: (fig.savefig(OUT / n, bbox_inches="tight", facecolor="white"), plt.close(fig))

# 1. Gram matrices
fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"wspace": .35})
for ax, (k, ttl) in zip(axs, (("if-then:MP", "Modus ponens: fires MP"), ("if-then:MPsyn", "Paraphrase: no fire"), ("if-then:near", "Near miss: no fire"))):
    w = W[k]; G = np.array(w["G"]); n = len(G); im = ax.imshow(G, cmap="RdBu_r", vmin=-1, vmax=1); tk = [t.strip() or "·" for t in w["tokens"]]
    ax.set_xticks(range(n), tk, rotation=90, fontsize=9); ax.set_yticks(range(n), tk, fontsize=9); ax.set_title(ttl, fontsize=11.5)
    b = np.cumsum([0] + [w["roles"].count(r) for r in ("c", "a1", "a2", "p2", "th")]) - .5
    for x in b[1:-1]: ax.axhline(x, color=INK, lw=.8); ax.axvline(x, color=INK, lw=.8)
    p2, a1 = [i for i, r in enumerate(w["roles"]) if r == "p2"], [i for i, r in enumerate(w["roles"]) if r == "a1"]
    ax.add_patch(plt.Rectangle((a1[0] - .5, p2[0] - .5), len(a1), len(p2), fill=False, ec="#F59E0B", lw=2.2))
fig.colorbar(im, ax=axs, shrink=.75, label="cosine of whitened embeddings"); fig.text(.12, -.02, "Blocks, in order: connective, clause 1, clause 2, second sentence, \"Therefore\". Amber box: second sentence against clause 1.", color=MUTED, fontsize=10)
save(fig, "te_gram.png")

# 2. the coordinates that decide
D = O["test distributions"]; kinds = [("MP", "MP"), ("MPsyn", "MP paraphrase"), ("near", "near miss"), ("DA", "denied antecedent"), ("AC", "affirmed consequent"), ("MT", "MT"), ("U", "unrelated"), ("DS1", "DS1"), ("P", "affirmed disjunct")]
cols = {"MP": FC["MP"], "MT": FC["MT"], "DS1": FC["DS1"], "P": GREY, "MPsyn": FC["MP"]}
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, gridspec_kw={"wspace": .08})
for ax, (cn, ttl) in zip(axs, (("premise 2 to clause 1, worst 2", "second-worst match of premise 2 to clause 1"), ("premise 2 to clause 2, worst 2", "second-worst match of premise 2 to clause 2"), ("connective position", "connective's position (cond. +, disj. −)"))):
    for i, (k, lab) in enumerate(kinds):
        q = D[cn][k]; col = cols.get(k, MUTED); ax.plot([q[0], q[4]], [i, i], color=col, lw=1.2, alpha=.6); ax.plot([q[1], q[3]], [i, i], color=col, lw=5, solid_capstyle="butt"); ax.plot(q[2], i, "o", color="white", mec=col, ms=7, mew=2)
    ax.set_title(ttl, fontsize=11); despine(ax); ax.axvline(0, color=GREY, lw=.8, ls=":")
axs[0].set_yticks(range(len(kinds)), [l for _, l in kinds]); axs[0].invert_yaxis(); fig.text(.12, -.04, "Test clauses, all wordings. Thick: 25 to 75%; thin: 10 to 90%; circle: median.", color=MUTED, fontsize=10)
save(fig, "te_coords.png")

# 3. the decision in the cloud subspace
C = O["classes"]; th = np.linspace(0, 2 * np.pi, 100); show = [("if-then:MP", "MP premise", FC["MP"], "o"), ("if-then:MPsyn", "its paraphrase", FC["MP"], "D"), ("if-then:near", "its near miss", GREY, "s"), ("if-then:DA", "denied antecedent", GREY, "v"), ("either-or:P", "affirmed disjunct", GREY, "^"), ("unless:P", "\"unless\": affirmed disjunct", "#EF4444", "*")]
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"wspace": .25})
for ax, (a, b) in zip(axs, ((0, 1), (0, 4))):
    for k, v in C.items():
        c = v["centre"]
        if v["template"] and k.split(":")[1] in FC: ax.plot(c[a] + 2.45 * np.cos(th), c[b] + 2.45 * np.sin(th), color=FC[v["template"]], lw=1.3); ax.annotate(v["template"], (c[a], c[b]), xytext=(-16, 10) if v["template"] == "MT" and b == 1 else (16, 10) if v["template"] == "DS2" and b == 1 else (0, 13), textcoords="offset points", ha="center", fontsize=10, color=FC[v["template"]], fontweight="bold")
        elif k in ("cond:near", "cond:DA", "disj:P", "cond:AC", "cond:U", "text"): ax.plot(c[a], c[b], "x", color=GREY, ms=7); ax.annotate(k.split(":")[-1], (c[a], c[b]), xytext=(5, -11), textcoords="offset points", fontsize=8.5, color=MUTED)
    for k, lab, col, mk in show:
        y = W[k]["y"]; ax.plot(y[a], y[b], mk, color=col, ms=12 if mk == "*" else 8, mec=INK, mew=.8, label=lab if a == 0 and b == 1 else None)
    ax.set_xlabel(f"direction {a + 1}"); ax.set_ylabel(f"direction {b + 1}"); despine(ax); ax.set_aspect("equal")
fig.legend(frameon=False, fontsize=9.5, loc="lower center", bbox_to_anchor=(.5, -.04), ncol=6, columnspacing=1.2)
fig.suptitle("Worked premises in the cloud subspace (95% circles of the templates)", fontweight="bold", y=.98)
save(fig, "te_decision.png")
