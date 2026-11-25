"""Figures for semantic_templates/Logical_Forms_as_Geometric_Figures.md (2026-11-25).

Reads results/logic_figures/: logic_figures_*.npz, logic_controls_*.json and logic_analysis.json (debug/logic_figures.py,
logic_controls.py, logic_analysis.py) and writes PNGs into figures/logic_figures/:
  lf_frozen.png    the frozen figure of each form after layer 1 (SR2-NP), drawn from its mean inter-node distances
  lf_identify.png  held-out identification of the form by its frozen figure, and what is confused with what
  lf_dims.png      how many directions hold 90% of the spread as contents are added, against the two baselines
  lf_negation.png  a valid form against its fallacy twin, by state; the negation contrast and the validity contrast
  lf_restate.png   distances between a clause and its restatement, against unrelated clauses in the same frame
  lf_gain.png      what each argument adds to the four candidate conclusions
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY = "#1F2937", "#6B7280", "#9CA3AF"
FC = {"MP": "#3B82F6", "MT": "#22C55E", "DS": "#F59E0B", "AC": "#EF4444", "DA": "#8B5CF6"}
FNAME = {"MP": "modus ponens", "MT": "modus tollens", "DS": "disjunctive syllogism",
         "AC": "affirming the consequent (invalid)", "DA": "denying the antecedent (invalid)"}
MODELS = [("sr2np", "SR2-NP"), ("fobk16np", "FO-b-PMX-K16-NP"), ("sr2", "SR2"), ("sr2_2b", "SR2-2B")]
MC = {"sr2np": "#3B82F6", "fobk16np": "#22C55E", "sr2": "#F59E0B", "sr2_2b": "#8B5CF6"}
ROLES = ["c1", "a1", "c2", "a2", "p2", "th", "cn"]
FORMS = ["MP", "MT", "DS", "AC", "DA"]
ST = ["embedding", "after\nlayer 0", "after\nlayer 1"]
HERE = Path(__file__).resolve().parent
OUT = HERE / "logic_figures"; OUT.mkdir(exist_ok=True)
RES = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/logic_figures"
A = json.loads((RES / "logic_analysis.json").read_text())
C = {m: json.loads((RES / f"logic_controls_{m}.json").read_text()) for m, _ in MODELS}
Z = {m: np.load(RES / f"logic_figures_{m}.npz") for m, _ in MODELS}


def despine(ax):
    ax.spines[["top", "right"]].set_visible(False)


def unit(F):
    F = F - F.mean(-2, keepdims=True)
    return F / np.linalg.norm(F, axis=(-2, -1), keepdims=True)


def mean_dist(F):
    """Mean over instances of the inter-node distances of the unit-size figures."""
    U = unit(F)
    return np.linalg.norm(U[:, :, None] - U[:, None], axis=-1).mean(0)


def mds2(D):
    n = len(D); Jc = np.eye(n) - 1.0 / n
    B = -0.5 * Jc @ (D ** 2) @ Jc
    w, V = np.linalg.eigh(B); i = np.argsort(w)[::-1][:2]
    return V[:, i] * np.sqrt(np.maximum(w[i], 0))


def frozen():
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.4))
    sub = [ROLES.index(r) for r in ("a1", "a2", "p2", "cn")]
    for col, f in enumerate(FORMS):
        F = Z["sr2np"][f"nodes_h2_{f}"]
        for row, (idx, title) in enumerate(((list(range(7)), "all seven nodes"), (sub, "the four clause nodes alone"))):
            ax = axes[row, col]
            D = mean_dist(F[:, idx]); X = mds2(D); labs = [ROLES[i] for i in idx]
            ax.plot(X[:, 0], X[:, 1], "-", color=GREY, lw=1, zorder=1)
            ax.scatter(X[:, 0], X[:, 1], s=70, color=FC[f], zorder=3, edgecolor="white", linewidth=0.8)
            for (x, y), r in zip(X, labs):
                ax.annotate(r, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=10, color=INK)
            if row == 0:
                ax.set_title(FNAME[f].replace(" (invalid)", "\n(invalid)"), fontsize=10.5, loc="left")
            if col == 0:
                ax.set_ylabel(title, fontsize=10)
            ax.set_aspect("equal", adjustable="datalim"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GREY)
    fig.suptitle("The frozen figure of each form after layer 1 (SR2-NP): mean inter-node distances, drawn in two dimensions",
                 x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.01, "nodes: c1 first connective, a1 first clause, c2 second connective, a2 second clause, "
             "p2 second premise, th 'Therefore', cn conclusion; lines join the nodes in reading order", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "lf_frozen.png"); plt.close(fig)


def identify():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for m, lab in MODELS:
        for suf, ls in (("", "o-"), ("_c", "s--")):
            ax.plot(range(3), [A["frozen"][f"h{l}"][f"{m}->{m}{suf}"]["acc"] for l in range(3)], ls, color=MC[m], lw=1.8,
                    label=lab + (", all seven nodes" if not suf else ", four clause nodes"))
    ax.axhline(0.2, color=GREY, ls=":"); ax.text(2.05, 0.2, "chance", color=MUTED, fontsize=9, va="center")
    ax.set_xticks(range(3), ST); ax.set_ylim(0, 1.05); ax.set_xlim(-0.15, 2.35); ax.set_ylabel("five-way accuracy, content held out")
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2, bbox_to_anchor=(0, 0.23)); despine(ax)
    ax.set_title("Identifying the form from 21 distances", loc="left")
    K = sum(np.array(A["frozen"]["h2"][f"{m}->{m}"]["confusion"]) for m, _ in MODELS); K = K / K.sum(1, keepdims=True)
    ax = axes[1]; ax.imshow(K, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{K[i, j]:.0%}", ha="center", va="center", fontsize=10, color="white" if K[i, j] > 0.5 else INK)
    ax.set_xticks(range(5), FORMS); ax.set_yticks(range(5), FORMS); ax.set_xlabel("read as"); ax.set_ylabel("true form")
    ax.set_title("After layer 1, four models pooled", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "lf_identify.png"); plt.close(fig)


def dims():
    fig, ax = plt.subplots(figsize=(8, 4.4)); xs = [5, 10, 20, 40]
    for k, lab, c in (("TXT", "ordinary text in the same layout", GREY), ("NR", "same frame, unrelated clauses", "#F59E0B"),
                      ("MP", "modus ponens", "#3B82F6")):
        Y = np.array([[A[m]["h2"][f"desc_{k}"]["form_d90"][str(x)] for x in xs] for m, _ in MODELS])
        ax.plot(xs, Y.mean(0), "o-", color=c, lw=2, label=lab); ax.fill_between(xs, Y.min(0), Y.max(0), color=c, alpha=0.2, lw=0)
    ax.set_xscale("log"); ax.set_xticks(xs, xs); ax.minorticks_off(); ax.set_xlabel("number of contents")
    ax.set_ylabel("directions holding 90% of the spread"); ax.legend(frameon=False); despine(ax)
    ax.set_title("After layer 1 (mean and range of four models)", loc="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / "lf_dims.png"); plt.close(fig)


def negation():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3)); ax = axes[0]
    for m, lab in MODELS:
        for key, ls, kl in (("cn", "o-", "conclusion state"), ("shape", "s--", "figure only")):
            ax.plot(range(3), [np.mean([A[m][f"h{l}"][p][key] for p in ("MP-DA", "MT-AC")]) for l in range(3)], ls, color=MC[m], lw=1.8,
                    label=f"{lab}, {kl}")
    ax.axhline(0.5, color=GREY, ls=":"); ax.text(0.0, 0.515, "chance", color=MUTED, fontsize=9)
    ax.set_xticks(range(3), ST); ax.set_ylim(0.45, 1.03); ax.set_ylabel("accuracy, content held out")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 0.12)); despine(ax)
    ax.set_title("A valid form against its fallacy twin", loc="left")
    ax = axes[1]; x = np.arange(4)
    for i, (k, lab, c) in enumerate((("N_norm", "negation contrast", "#3B82F6"), ("V_norm", "validity contrast", "#EF4444"))):
        v = [A[m]["h2"]["validity_contrast"]["cn"][k] for m, _ in MODELS]
        ax.bar(x + (i - .5) * .36, v, .36, color=c, label=lab)
        for a, b in zip(x + (i - .5) * .36, v):
            ax.text(a, b + .06, f"{b:.2f}", ha="center", fontsize=9)
    for a, (m, _) in zip(x, MODELS):
        ax.text(a + .18, 1.0, f"p {A[m]['h2']['validity_contrast']['cn']['p']:.2f}", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(x, [l.replace("FO-b-PMX-", "FO-b-PMX-\n") for _, l in MODELS]); ax.set_ylabel("norm of the mean contrast"); ax.set_ylim(0, 5.2)
    ax.legend(frameon=False, ncol=2, loc="upper center"); despine(ax); ax.set_title("Conclusion state after layer 1", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "lf_negation.png"); plt.close(fig)


def restate():
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True); P = ["p2_a1", "p2_a2", "cn_a1", "cn_a2"]
    for ax, (m, lab) in zip(axes, MODELS):
        for i, (f, c) in enumerate([(f, FC[f]) for f in ("MP", "DA", "MT", "AC")] + [("NR", GREY)]):
            ax.bar(np.arange(4) + (i - 2) * .16, [A[m]["h2"]["restate"][f][q] for q in P], .16, color=c, edgecolor="white",
                   label=(FNAME.get(f, "same frame, unrelated clauses")) if ax is axes[0] else None)
        ax.set_xticks(range(4), ["p2 to a1", "p2 to a2", "cn to a1", "cn to a2"], fontsize=9); ax.set_title(lab, loc="left"); despine(ax)
    axes[0].set_ylabel("distance in the clause figure")
    fig.legend(frameon=False, fontsize=9, loc="lower center", ncol=5); fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(OUT / "lf_restate.png"); plt.close(fig)


def gain():
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3)); OKC = {"MP": 1, "MT": 2, "DS": 1}
    for ax, (m, lab) in zip(axes, MODELS):
        G = np.array([C[m]["gain"][f]["mean"] for f in FORMS]); ax.imshow(G, cmap="Blues", vmin=1.5, vmax=5.6, aspect="auto")
        for i, f in enumerate(FORMS):
            for j in range(4):
                ax.text(j, i, f"{G[i, j]:.1f}\ntop {C[m]['gain'][f]['top'][j]}", ha="center", va="center", fontsize=8.5, color="white" if G[i, j] > 3.9 else INK)
            if f in OKC:
                ax.add_patch(plt.Rectangle((OKC[f] - .47, i - .47), .94, .94, fill=False, ec="#EF4444", lw=2.2))
        ax.set_xticks(range(4), ["P", "Q", "not P", "not Q"]); ax.set_yticks(range(5), FORMS); ax.set_title(lab, loc="left")
    fig.text(0.01, 0.01, "nats per token gained from the argument, and in how many of 40 contents the candidate gains most; red frame: the valid conclusion (AC and DA have none)", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(OUT / "lf_gain.png"); plt.close(fig)


if __name__ == "__main__":
    for old in ("lf_recurrence", "lf_separation", "lf_reaction", "lf_wellframe"):
        (OUT / f"{old}.png").unlink(missing_ok=True)
    frozen(); identify(); dims(); negation(); restate(); gain()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
