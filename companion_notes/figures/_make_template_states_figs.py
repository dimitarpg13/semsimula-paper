"""Figure for semantic_templates/What_the_Matchers_Read.md (2026-11-25).

Reads results/template_matcher/template_state_readings_sr2_2b.json (written by debug/template_state_readings.py) and writes
figures/template_matcher/tm_states.png: the repeated word across the states, the length of the states, and the radius to the well.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY, BLUE, AMBER = "#1F2937", "#6B7280", "#9CA3AF", "#3B82F6", "#F59E0B"
HERE = Path(__file__).resolve().parent
D = json.loads((HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/template_matcher/template_state_readings_sr2_2b.json").read_text())
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1.1, 1.1], "wspace": 0.3})

ax = axs[0]; x = [0, 1, 2]
for k, col, lab in (("MP", BLUE, "repeated word"), ("U", GREY, "unrelated word")):
    y = [D[f"cos_h{l}"][k][1] for l in x]; ax.plot(x, y, color=col, marker="o", ms=8, lw=2)
    for a, b in zip(x, y): ax.annotate(f"{b:.2f}" if b < .99 else f"{b:.3f}", (a, b), xytext=(0, 9), textcoords="offset points", ha="center", color=INK, fontsize=10)
    ax.text(1.0, y[1] - .12, lab, color=MUTED, ha="center", va="center", fontsize=10)
ax.set_xticks(x, ["$h^{(0)}$", "$h^{(1)}$", "$h^{(2)}$"]); ax.set_ylim(-.12, 1.15); ax.set_xlim(-.25, 2.25); ax.set_ylabel("cosine (whitened states)"); ax.set_title("Word identity fades")

ax = axs[1]; nm = ["h0", "pre1", "h1", "pre2", "h2"]; y = [D[f"norm_{k}"] for k in nm]
ax.bar(range(5), y, width=.62, color=[GREY, BLUE, GREY, BLUE, GREY])
for a, b in enumerate(y): ax.text(a, b + .4, f"{b:.1f}", ha="center", color=INK, fontsize=10)
ax.set_xticks(range(5), ["$h^{(0)}$", "after\nstep 0", "$h^{(1)}$", "after\nstep 1", "$h^{(2)}$"], fontsize=10); ax.set_ylim(0, 24); ax.set_ylabel("median length of the state"); ax.set_title("The LayerNorm sets the length")

ax = axs[2]
for l, col in ((0, BLUE), (1, AMBER)):
    w = D[f"well_{l}"]
    for j, key in enumerate(("entry", "out_pre")):
        lo, m, hi = w[key]; xx = 3 * l + j; ax.plot([xx, xx], [lo, hi], color=col, lw=2); ax.plot(xx, m, "o", ms=9, color=col, mec="white", mew=1.5)
        ax.text(xx + (.14 if j else -.14), m, f"{m:.2f}", va="center", ha="left" if j else "right", color=INK, fontsize=10)
    ax.annotate("", (3 * l + .9, w["out_pre"][1]), (3 * l + .1, w["entry"][1]), arrowprops=dict(arrowstyle="-", color=col, lw=1, ls=":"))
ax.set_yscale("log"); ax.set_yticks([.2, .5, 1, 2, 5, 10], ["0.2", "0.5", "1", "2", "5", "10"]); ax.minorticks_off(); ax.set_ylim(.16, 18); ax.set_xlim(-.8, 4.8)
ax.set_xticks([0, 1, 3, 4], ["entry\nlayer 0", "after\nlayer 0", "entry\nlayer 1", "after\nlayer 1"], fontsize=10); ax.set_ylabel("radius to the entry well (widths)"); ax.set_title("Particles move away from the well")
for ax in axs: ax.spines[["top", "right"]].set_visible(False)
fig.savefig(HERE / "template_matcher/tm_states.png", bbox_inches="tight", facecolor="white")
