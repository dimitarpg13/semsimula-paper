"""Figure for semantic_templates/Early_State_Matcher_Runner_Up.md (2026-11-26): every reading against the embedding, on both tasks.

Reads results/template_matcher/template_depth_{np,g2,l4}.json (debug/template_depth_analysis.py, primary readings: "unless" left out).
Writes into figures/template_matcher/:
  es_standings.png   per model, each reading's difference to the embedding on R1 (unseen wordings) against its difference on
                     paraphrases caught at 1% near misses, with 95% intervals over clauses; families by colour
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY, BLUE, AMBER, RED = "#1F2937", "#6B7280", "#9CA3AF", "#3B82F6", "#F59E0B", "#DC2626"
HERE = Path(__file__).resolve().parent; OUT = HERE / "template_matcher"; RES = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/template_matcher"
MODELS = (("np", "SR2-NP, two layers"), ("g2", "G2, two layers"), ("l4", "L4, four layers"))
EARLY = {"np": ("h1",), "g2": ("h1",), "l4": ("h1", "h2")}              # the states up to the middle of the depth
FAMILY = [("early state", BLUE), ("later state", RED), ("displacement", GREY), ("well frame", MUTED), ("joint (path, wells, states)", AMBER)]


def family(s, k):
    if k in EARLY[s]: return "early state"
    if k[0] == "h": return "later state"
    if k[0] == "D": return "displacement"
    if k[0] == "W": return "well frame"
    return "joint (path, wells, states)"


def label(k):
    if k[0] == "h": return f"after layer {int(k[1:]) - 1}"
    return {"D": "displacement", "W": "well frame"}.get(k[0], k) + (f" {k[1:]}" if k[0] in "DW" else "")


fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
for ax, (s, nm) in zip(axes, MODELS):
    P = json.loads((RES / f"template_depth_{s}.json").read_text())["primary"]
    ax.axhline(0, color=GREY, lw=0.8, zorder=0); ax.axvline(0, color=GREY, lw=0.8, zorder=0)
    for k, v in P.items():
        if k == "h0": continue
        f = family(s, k); c = dict(FAMILY)[f]; (x, xl, xh), (y, yl, yh) = v["minus h0"]["r1"], v["minus h0"]["p0.01"]
        ax.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], fmt="o", ms=7 if f == "early state" else 5, color=c,
                    ecolor=c, elinewidth=1, alpha=0.95 if f == "early state" else 0.75, zorder=3 if f == "early state" else 2)
        if f in ("early state", "joint (path, wells, states)"):
            dx, dy = {("l4", "h1"): (-58, -14), ("g2", "path"): (-30, -12)}.get((s, k), (6, 4))
            ax.annotate(label(k), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK if f == "early state" else MUTED)
    ax.set_title(nm); ax.set_xlabel("R1, unseen wordings: reading minus embedding")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("paraphrases caught at 1% near misses:\nreading minus embedding")
fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAMILY], loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "es_standings.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote es_standings.png")
