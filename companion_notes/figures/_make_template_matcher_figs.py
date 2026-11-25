"""Figures for semantic_templates/Template_Matcher_From_Geometry.md (2026-11-25).

Reads results/template_matcher/ (template_geometry*.json, template_cloud_sr2_2b.json, example_premises_sr2_2b.json, written by
debug/template_geometry_{analysis,detail,frame,export}.py) and writes PNGs into figures/template_matcher/:
  tm_figure.png    one premise as the geometry the matcher sees: the cosines between its particles after layer 0
  tm_clouds.png    the template clouds and their rivals in the subspace that encloses them (leading four directions)
  tm_dims.png      development score against the number of particles and the subspace dimension
  tm_firing.png    how often the right template fires: known and unseen wordings, as fixed in advance and with the position
  tm_rivals.png    false fires per rival kind, known and unseen wordings
  tm_wordings.png  each wording left out in turn: detection and false fires on the fallacies
  tm_check.png     predicted against measured pair errors, and the squared distance to the cloud centre against chi-squared
  tm_training.png  trained models against their step-500 checkpoints
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY = "#1F2937", "#6B7280", "#9CA3AF"
FC = {"MP": "#3B82F6", "MT": "#22C55E", "DS1": "#F59E0B", "DS2": "#8B5CF6"}
SRC = {"h1+h2": "#3B82F6", "h0": "#9CA3AF", "h1": "#22C55E", "h2": "#F59E0B"}
HERE = Path(__file__).resolve().parent
OUT = HERE / "template_matcher"; OUT.mkdir(exist_ok=True)
RES = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/template_matcher"
L = lambda n: json.loads((RES / n).read_text())
GEO, DET, FRM, CLD, EX = L("template_geometry.json"), L("template_geometry_detail.json"), L("template_geometry_frame.json"), L("template_cloud_sr2_2b.json"), L("example_premises_sr2_2b.json")


def despine(ax):
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white"); plt.close(fig)


# 1. one premise as a geometry
fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
lab = {"c": "connective", "a1": "clause 1", "a2": "clause 2", "p2": "premise 2", "th": "Therefore"}
for ax, (k, t) in zip(axs, (("MP", "modus ponens"), ("U", "unrelated premise 2"), ("DS1", "disjunctive syllogism"))):
    G, ro = np.array(EX[k]["G"]), EX[k]["roles"]; n = len(ro); im = ax.imshow(G, cmap="RdBu_r", vmin=-1, vmax=1)
    cuts = [i for i in range(1, n) if ro[i] != ro[i - 1]]
    for c in cuts:
        ax.axhline(c - .5, color=INK, lw=.8); ax.axvline(c - .5, color=INK, lw=.8)
    e = [0] + cuts + [n]; mid = [(a + b - 1) / 2 for a, b in zip(e[:-1], e[1:])]; nm = [lab[ro[a]] for a in e[:-1]]
    ax.set_xticks(mid, nm, rotation=45, ha="right", fontsize=8.5); ax.set_yticks(mid, nm if ax is axs[0] else [], fontsize=8.5); ax.set_title(f"{t}\n{n} particles", fontsize=11)
fig.colorbar(im, ax=axs, shrink=.75, label="cosine between two particles after layer 0")
save(fig, "tm_figure.png")

# 2. clouds in the leading directions
ids = {k: i for i, k in enumerate(CLD["main"])}; C = {k: np.array(CLD["classes"][k]["mean"]) for k in CLD["classes"]}; P = {k: np.array(v) for k, v in CLD["points"].items()}
show = [("cond:MP", "MP", "modus ponens"), ("cond:MT", "MT", "modus tollens"), ("disj:DS1", "DS1", "disjunctive syllogism, first denied"), ("disj:DS2", "DS2", "disjunctive syllogism, second denied")]
riv = [("cond:DA", "X", "denying the antecedent"), ("disj:P", "^", "affirmed disjunct"), ("cond:near", "s", "near miss"), ("text", "D", "ordinary text")]
fig, axs = plt.subplots(1, 2, figsize=(13, 5.4))
for ax, (i, j) in zip(axs, ((0, 1), (2, 3))):
    for k, m, n in riv: ax.scatter(P[k][:, i], P[k][:, j], s=12, marker=m, c=MUTED, alpha=.35, lw=0, label=n)
    for k, t, n in show:
        ax.scatter(P[k][:, i], P[k][:, j], s=10, c=FC[t], alpha=.45, lw=0)
        ax.add_patch(plt.Circle(C[k][[i, j]], np.sqrt(stats.chi2.ppf(.95, 2)), fill=False, ec=FC[t], lw=2)); ax.plot(*C[k][[i, j]], "o", color=FC[t], mec="white", ms=9, label=n)
    ax.set_xlabel(f"direction {i + 1}"); ax.set_ylabel(f"direction {j + 1}"); despine(ax); ax.set_aspect("equal", adjustable="datalim")
axs[0].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(-.02, -.16), ncol=4, columnspacing=1)
fig.suptitle("Template clouds in their subspace (circles: 95% of a unit Gaussian cloud in the two directions shown)", fontsize=12, fontweight="bold")
save(fig, "tm_clouds.png")

# 3. how many particles, how many dimensions
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for st in ("h0", "h1", "h2", "h1+h2"):
    q = DET[f"sr2_2b {st}"]; ms = sorted(int(k) for k in q["particles"]); axs[0].plot(ms, [q["particles"][str(k)][0] for k in ms], "o-", color=SRC[st], lw=2, ms=5, label=st)
    dq = {int(k): v[0] for k, v in q["dims"].items()}; ds = sorted(dq); axs[1].plot(ds, [dq[k] for k in ds], "o-", color=SRC[st], lw=2, ms=5, label=st)
axs[0].set_xlabel("particles of premise 2 used, worst matched first"); axs[1].set_xlabel("dimension of the cloud subspace"); axs[0].set_ylabel("development score (held-out clauses)")
for ax in axs: despine(ax); ax.grid(axis="y", color="#E5E7EB")
axs[0].legend(frameon=False, title="state", fontsize=9); save(fig, "tm_dims.png")

# 4. firing
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True); TPL = ["MP", "MT", "DS1", "DS2"]; w = .2
for ax, (wk, ttl) in zip(axs, (("known", "wordings known"), ("unseen", "wording left out"))):
    for j, (st, fr, al) in enumerate((("h1+h2", "geo", .45), ("h1+h2", "pos", 1.), ("h0", "pos", 1.))):
        v = []
        for t in TPL:
            if fr == "geo": v.append(DET["sr2_2b h1+h2"][wk]["detect by kind"][t][t][0])
            else: v.append(FRM[f"sr2_2b {st}"][wk]["detect"][t][0])
        x = np.arange(4) + (j - 1) * (w + .02); ax.bar(x, v, w, color=SRC[st] if fr == "pos" else "#93C5FD", label={0: "states after both layers, as fixed in advance", 1: "states after both layers, connective position added", 2: "embedding, position added (reference)"}[j])
        for xi, vi in zip(x, v): ax.text(xi, vi + .012, ("%.2f" % vi).lstrip("0"), ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(4), ["modus\nponens", "modus\ntollens", "disj. syll.,\nfirst denied", "disj. syll.,\nsecond denied"], fontsize=9); ax.set_title(ttl); despine(ax); ax.set_ylim(0, 1.15)
axs[0].set_ylabel("the right template fires"); axs[0].legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(1.0, -.18), ncol=3); save(fig, "tm_firing.png")

# 5. rivals
RK = ["false AC", "false DA", "false affirmed disjunct", "false near miss", "false other subject", "false unrelated", "false ordinary text"]
NM = ["affirming the\nconsequent", "denying the\nantecedent", "affirmed\ndisjunct", "near\nmiss", "other\nsubject", "unrelated\nclause", "ordinary\ntext"]
fig, ax = plt.subplots(figsize=(11, 4.3)); x = np.arange(len(RK)); w = .38
for j, (wk, c) in enumerate((("known", "#9CA3AF"), ("unseen", "#3B82F6"))):
    r = FRM["sr2_2b h1+h2"][wk]["report"]; v = [100 * r[k][0] for k in RK]; ax.bar(x + (j - .5) * w, v, w - .03, color=c, label=f"wordings {'known' if wk == 'known' else 'left out'}")
    for xi, vi in zip(x + (j - .5) * w, v): ax.text(xi, vi + .25, f"{vi:.1f}", ha="center", fontsize=8.5)
ax.set_xticks(x, NM, fontsize=9); ax.set_ylabel("false fires, %"); ax.legend(frameon=False); despine(ax); ax.set_title("Rivals on which a template fires by mistake (states after both layers, position added)", fontsize=11.5); save(fig, "tm_rivals.png")

# 6. wordings
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8)); W12 = list(FRM["sr2_2b h1+h2"]["unseen"]["by wording"])
for ax, (st, t) in zip(axs, (("h1+h2", "states after both layers"), ("h0", "embedding (reference)"))):
    q = FRM[f"sr2_2b {st}"]["unseen"]["by wording"]; d = [q[k][0] for k in W12]; f = [q[k][1] for k in W12]; y = np.arange(12)[::-1]
    ax.barh(y + .2, d, .38, color="#3B82F6", label="right template fires"); ax.barh(y - .2, f, .38, color="#EF4444", label="fires on a fallacy")
    ax.set_yticks(y, W12, fontsize=9); ax.set_xlim(0, 1.02); ax.set_title(t); despine(ax)
axs[0].legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(1.0, -.2), ncol=2); save(fig, "tm_wordings.png")

# 7. theory check
fig, axs = plt.subplots(1, 2, figsize=(13, 5)); pr = CLD["pairs"]; ax = axs[0]; y = np.arange(len(pr))[::-1]; fl = 1e-5
nmk = {"cond:MP": "MP", "cond:MT": "MT", "cond:DA": "DA", "cond:AC": "AC", "cond:near": "near miss", "cond:nearQn": "near miss", "cond:swapQn": "other subject", "disj:DS1": "DS1", "disj:DS2": "DS2", "disj:P": "affirmed P", "disj:Q": "affirmed Q", "disj:nearn": "near miss"}
for yi, (a_, b_, dl, pe, me) in zip(y, pr):
    ax.plot([max(pe, fl), max(me, fl)], [yi, yi], color="#D1D5DB", lw=2, zorder=1)
    ax.plot(max(pe, fl), yi, "o", ms=8, mfc="white", mec="#3B82F6", mew=2, zorder=3); ax.plot(max(me, fl), yi, "o", ms=8, color="#EF4444", mec="white", zorder=4)
ax.set_yticks(y, [f"{nmk[a_]} / {nmk[b_]}  ({dl:.1f})" for a_, b_, dl, pe, me in pr], fontsize=9); ax.set_xscale("log"); ax.set_xlim(fl * .7, .2); ax.set_xlabel("chance of taking one cloud for the other (floor 1e-5)"); ax.set_title("Predicted from the centres (ring), measured (red)", fontsize=11); despine(ax)
ax.text(.99, -.2, "in brackets: distance of the centres, in cloud widths", transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)
ax = axs[1]; qs = np.array([.1, .25, .5, .75, .9, .99]); d_ = CLD["d"]; th = stats.chi2.ppf(qs, d_); ax.plot([0, 60], [0, 60], "--", color=MUTED, lw=1)
for k, n in (("cond:MP", "MP"), ("cond:MT", "MT"), ("disj:DS1", "DS1"), ("disj:DS2", "DS2")): ax.plot(th, CLD["chi2"][k]["q"], "o-", color=FC[n], lw=2, ms=6, label=n)
ax.set_xlabel(f"quantile of chi-squared with {d_} degrees of freedom"); ax.set_ylabel("quantile of the squared distance"); ax.set_title("Distances on fresh clauses against their law", fontsize=11); ax.legend(frameon=False); despine(ax); ax.set_xlim(0, 25); save(fig, "tm_check.png")

# 8. training
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4)); runs = [("sr2_2b", "SR2-2B"), ("sr2_2b_s500", "SR2-2B,\nstep 500"), ("sr2", "SR2"), ("sr2_s500", "SR2,\nstep 500")]; x = np.arange(4)
def val(s, f): q = FRM[f"{s} h1+h2"]; return f(q)
a1 = [np.mean([val(s, lambda q: q["known"]["detect"][t][0]) for t in TPL]) for s, _ in runs]; a2 = [np.mean([val(s, lambda q: q["unseen"]["detect"][t][0]) for t in TPL]) for s, _ in runs]
a3 = [np.mean([val(s, lambda q: q["known"]["para"][t][0]) for t in TPL]) for s, _ in runs]; a4 = [np.mean(list(FRM[f"{s} h1+h2"]["known"]["AUROC para / near miss"].values())) for s, _ in runs]
for ax, (v, t, yl) in zip(axs, ((list(zip(a1, a2)), "Exact premises: right template fires", "mean of the four forms"), ([(v,) for v in a3], "Paraphrased premise: right template fires", "mean of the four forms"), ([(v,) for v in a4], "Paraphrase against its near miss", "AUROC (0.5 is chance)"))):
    c = ["#3B82F6", "#93C5FD"]
    for j in range(len(v[0])): ax.bar(x + (j - .5 * (len(v[0]) - 1)) * .36, [q[j] for q in v], .34, color=c[j], label=["wordings known", "wordings left out"][j] if len(v[0]) > 1 else None)
    for i, q in enumerate(v):
        for j, qq in enumerate(q): ax.text(x[i] + (j - .5 * (len(q) - 1)) * .36, qq + .01, f"{qq:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [n for _, n in runs], fontsize=9); ax.set_title(t, fontsize=10); ax.set_ylabel(yl, fontsize=9.5); despine(ax); ax.set_ylim(0, 1.1 if len(v[0]) > 1 else (1.0 if "AUROC" in yl else .5))
axs[0].legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2); axs[2].axhline(.5, color=MUTED, ls="--", lw=1); save(fig, "tm_training.png")
print("ok")
