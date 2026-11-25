"""Figures for semantic_templates/Template_Matchers_on_the_Dynamics_SR2_2B.md (2026-11-25).

Reads results/template_matcher/template_dynamics*_sr2_2b.json (written by debug/template_dynamics_{analysis,readings,rates,
without_unless}.py) and writes PNGs into figures/template_matcher/:
  td_contrasts.png   every matcher against the embedding on R1 and R2, with intervals and the marks of the calls
  td_wordings.png    each wording left out in turn: detection of the valid forms per matcher
  td_unless.png      R1 and the fallacy fires with and without the wording "unless"
  td_paraphrase.png  paraphrases caught against near misses fired, threshold moved
  td_words.png       the last word of the second sentence against clause 1, per particle vector
  td_clouds.png      the cloud centres of the displacement matcher in its leading directions
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY = "#1F2937", "#6B7280", "#9CA3AF"
BLUE, GREEN, AMBER, VIOLET, RED = "#3B82F6", "#22C55E", "#F59E0B", "#8B5CF6", "#EF4444"
HERE = Path(__file__).resolve().parent
OUT = HERE / "template_matcher"
RES = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/template_matcher"
L = lambda n: json.loads((RES / n).read_text())
DYN, RD, RT, NU = L("template_dynamics_sr2_2b.json")["sr2_2b"], L("template_dynamics_readings_sr2_2b.json"), L("template_dynamics_rates_sr2_2b.json"), L("template_dynamics_without_unless_sr2_2b.json")
NAME = {"h0": "embedding", "D1": "displacement, layer 0", "D2": "displacement, layer 1", "D1+D2": "A: both displacements", "W1": "well frame, layer 0", "W2": "well frame, layer 1",
        "W1+W2": "B: both well frames", "h1+h2": "states after both layers", "h1": "state after layer 0", "h2": "state after layer 1"}
despine = lambda ax: ax.spines[["top", "right"]].set_visible(False)
save = lambda fig, n: (fig.savefig(OUT / n, bbox_inches="tight", facecolor="white"), plt.close(fig))

# 1. contrasts to the embedding
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, gridspec_kw={"wspace": .08}); order = ["D1+D2", "W1+W2", "h1+h2", "D1", "D2", "W1", "W2", "h1", "h2"][::-1]
for ax, key, mark, ttl in ((axs[0], "R1 unseen minus h0", .03, "R1: hard set, unseen wordings"), (axs[1], "R2 known minus h0", .10, "R2: paraphrases, known wordings")):
    for i, n in enumerate(order):
        m, lo, hi = DYN[n][key]; col = BLUE if n in ("D1+D2", "W1+W2") else GREY
        ax.plot([lo, hi], [i, i], color=col, lw=2); ax.plot(m, i, "o", ms=8, color=col, mec="white", mew=1.5); ax.text(1.0, i, f"{m:+.3f}", va="center", ha="right", color=INK, fontsize=10, transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK, lw=1); ax.axvline(mark, color=RED, lw=1.2, ls="--"); ax.text(mark, len(order) - .35, f"call's mark {mark:+.2f}", color=MUTED, fontsize=9.5, ha="center", va="bottom")
    ax.set_title(ttl, pad=22); ax.set_xlabel("difference to the embedding"); despine(ax); ax.set_ylim(-.6, len(order) - .4)
axs[0].set_yticks(range(len(order)), [NAME[n] for n in order]); axs[0].set_xlim(-.42, .16); axs[1].set_xlim(-.05, .27); axs[1].tick_params(axis="y", length=0)
save(fig, "td_contrasts.png")

# 2. wordings left out
ws = ["if-then", "when", "whenever", "q-if-p", "assuming", "provided", "either-or", "or", "at-least-one", "or-else", "one-holds", "unless"]; mm = ["h0", "D1", "D1+D2", "W1+W2", "h1+h2"]
Z = np.array([[RD[n]["unseen by wording"][w]["detect"] for w in ws] for n in mm]); Fz = np.array([[RD[n]["unseen by wording"][w]["fallacy"] for w in ws] for n in mm])
fig, axs = plt.subplots(2, 1, figsize=(12.5, 6.2), sharex=True, gridspec_kw={"hspace": .28})
for ax, M, cm, ttl in ((axs[0], Z, "Blues", "The right template fires on valid premises"), (axs[1], Fz, "Oranges", "Any template fires on a fallacy (affirming the consequent, denying the antecedent, affirmed disjunct)")):
    ax.imshow(M, aspect="auto", cmap=cm, vmin=0, vmax=1 if cm == "Blues" else .6); ax.set_title(ttl, loc="left"); ax.set_yticks(range(len(mm)), [NAME[n] for n in mm]); ax.axvline(5.5, color="white", lw=4)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=10, color="white" if v > (.6 if cm == "Blues" else .4) else INK)
    ax.tick_params(length=0)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
axs[1].set_xticks(range(len(ws)), ws, rotation=30, ha="right"); axs[1].set_xlabel("wording left out of fitting (six conditional, six disjunctive), per cent of premises", labelpad=10)
save(fig, "td_wordings.png")

# 3. unless
order = ["h0", "h1", "D1", "h1+h2", "D1+D2", "W1", "W1+W2", "h2", "D2", "W2"][::-1]; fig, axs = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"wspace": .5})
ax = axs[0]
for i, n in enumerate(order):
    a, b = DYN[n]["R1 unseen"], NU[n]["R1 unseen"]; ax.plot([a, b], [i, i], color=GREY, lw=2); ax.plot(a, i, "o", ms=8, color=GREY, mec="white"); ax.plot(b, i, "o", ms=8, color=BLUE, mec="white"); ax.text(max(a, b) + .012, i, f"{b:.3f}", va="center", fontsize=10, color=INK)
ax.set_yticks(range(len(order)), [NAME[n] for n in order]); ax.set_xlim(.5, 1.05); ax.set_xlabel("R1: hard set, unseen wordings"); ax.set_title('All twelve wordings (grey), "unless" left out (blue)', fontsize=11); despine(ax)
ax = axs[1]; keys = [("cond:DA", "denying the\nantecedent"), ("cond:DAnot", "the same,\nreworded"), ("disj:P", "affirmed\ndisjunct (first)")]; x = np.arange(3); rows = ["h0", "D1+D2", "W1+W2", "h1+h2"]; cols = [GREY, BLUE, AMBER, GREEN]
for j, n in enumerate(rows):
    a = [RD[n]["unseen fires on"][k][0] for k, _ in keys]; b = [NU[n]["unseen fires on"][k] for k, _ in keys]; off = (j - 1.5) * .19
    ax.bar(x + off, a, width=.17, color=cols[j], alpha=.35); ax.bar(x + off, b, width=.17, color=cols[j], label=NAME[n])
ax.set_xticks(x, [k for _, k in keys], fontsize=10); ax.set_ylabel("any template fires, unseen wordings"); ax.set_title('Fallacy fires: all twelve (light), without "unless" (solid)', fontsize=11); ax.legend(frameon=False, fontsize=9.5); despine(ax); ax.set_ylim(0, .33)
save(fig, "td_unless.png")

# 4. paraphrases against near misses
fig, ax = plt.subplots(figsize=(7.6, 4.8)); rates = [.0025, .01, .02, .05]
for n, col in (("h0", GREY), ("D1+D2", BLUE), ("W1+W2", AMBER), ("h1+h2", GREEN)):
    y = [RT[n]["at near-miss rate"][str(r)]["paraphrase"] for r in rates]; ax.plot(rates, y, color=col, lw=2, marker="o", ms=7, mec="white", label=NAME[n])
    nx = RT[n]["near miss gates"]["both"]; ax.plot(nx, RT[n]["paraphrase gates"]["all"], "D", ms=9, color=col, mec=INK, mew=1.2)
ax.set_xscale("symlog", linthresh=.0025, linscale=.8); ax.set_xticks([0] + rates, ["0", "0.25%", "1%", "2%", "5%"]); ax.minorticks_off(); ax.set_xlim(-.0003, .07); ax.set_ylim(0, .55); ax.set_xlabel("near misses that fire"); ax.set_ylabel("paraphrases caught (right template)"); ax.set_title("A and the embedding rank alike, and stop at different points")
ax.text(.0001, .2, "diamonds: the rule's\nown threshold", color=MUTED, fontsize=9.5, ha="left"); ax.legend(frameon=False, fontsize=9.5, loc="upper left"); despine(ax)
save(fig, "td_paraphrase.png")

# 5. last word of the second sentence against clause 1
fig, ax = plt.subplots(figsize=(9.5, 4.8)); vec = ["h0", "D1", "h1", "D2", "h2", "W1", "W2"]; kinds = [("MP", "repeated", BLUE), ("MPsyn", "paraphrased", GREEN), ("near", "near miss", AMBER), ("U", "unrelated", GREY)]
for j, (k, lab, col) in enumerate(kinds):
    for i, v in enumerate(vec):
        lo, m, hi = RD["last particle cosine"][v][k]; xx = i + (j - 1.5) * .17; ax.plot([xx, xx], [lo, hi], color=col, lw=2); ax.plot(xx, m, "o", ms=7, color=col, mec="white", label=lab if i == 0 else None)
ax.set_xticks(range(len(vec)), ["embedding", "displ. 0", "state 1", "displ. 1", "state 2", "wells 0", "wells 1"]); ax.set_ylabel("cosine with the last word of clause 1"); ax.set_title("The displacements carry word identity, the well frame does less"); ax.legend(frameon=False, fontsize=9.5, ncol=4, loc="upper right"); ax.set_ylim(-.05, 1.2); despine(ax)
save(fig, "td_words.png")

# 6. cloud centres of matcher A
C = RD["D1+D2"]["cloud"]["classes"]; fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"wspace": .25}); col = {"MP": BLUE, "MT": GREEN, "DS1": AMBER, "DS2": VIOLET}; riv = ["cond:AC", "cond:DA", "cond:DAnot", "disj:P", "disj:Q", "cond:U", "text"]; th = np.linspace(0, 2 * np.pi, 100)
for ax, (a, b) in zip(axs, ((0, 1), (0, 3))):
    for k, v in C.items():
        c = v["centre"]
        if v["template"]:
            if k.split(":")[1] in col: ax.plot(c[a] + 2.45 * np.cos(th), c[b] + 2.45 * np.sin(th), color=col[v["template"]], lw=1.5); ax.plot(c[a], c[b], "o", color=col[v["template"]], ms=9, mec="white"); ax.annotate(v["template"], (c[a], c[b]), xytext=(-14, -22) if v["template"] == "MT" and b == 1 else (8, 8), textcoords="offset points", fontsize=10, color=INK)
        elif k in riv: ax.plot(c[a], c[b], "X", color=GREY, ms=9, mec="white"); ax.annotate(k.split(":")[-1], (c[a], c[b]), xytext=(-14, -12) if k.endswith(":AC") and b == 1 else (6, 5) if k.endswith(":P") and b == 1 else (6, -12), textcoords="offset points", fontsize=9, color=MUTED)
    ax.set_xlabel(f"direction {a + 1}"); ax.set_ylabel(f"direction {b + 1}"); ax.set_aspect("equal"); despine(ax)
fig.suptitle("Displacement matcher: clouds of the four templates (95% circles) and some rivals", fontweight="bold", fontsize=12)
save(fig, "td_clouds.png")
