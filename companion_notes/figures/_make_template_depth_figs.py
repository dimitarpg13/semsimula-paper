"""Figures for the depth arms of the template matchers (semantic_templates/Template_Matchers_and_Depth_<arm>.md), 2026-11-25.

Reads results/template_matcher/template_depth_<stem>.json and flow_<stem>.json (debug/template_depth_{test,analysis}.py) and,
for the comparison with SR2-2B, template_dynamics_without_unless_sr2_2b.json. Usage: python3 _make_template_depth_figs.py stem
Writes into figures/template_matcher/:
  tdp_<stem>_flow.png       the gap of each layer's trained step to its 16-substep flow, against the number of substeps
  tdp_<stem>_contrasts.png  every matcher against the embedding: R1 (without "unless") and paraphrases at 1% near misses
  tdp_<stem>_vs_sr2_2b.png  R1 per reading, this model and SR2-2B (same clauses, both without "unless")
"""
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY, BLUE, AMBER, GREEN, VIOLET = "#1F2937", "#6B7280", "#9CA3AF", "#3B82F6", "#F59E0B", "#22C55E", "#8B5CF6"
HERE = Path(__file__).resolve().parent; OUT = HERE / "template_matcher"; RES = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/template_matcher"
s = sys.argv[1] if len(sys.argv) > 1 else "g2"; L = lambda n: json.loads((RES / n).read_text()); Q, FL = L(f"template_depth_{s}.json"), L(f"flow_{s}.json"); NU = L("template_dynamics_without_unless_sr2_2b.json"); Lq = Q["L"]
despine = lambda ax: ax.spines[["top", "right"]].set_visible(False)
save = lambda fig, n: (fig.savefig(OUT / n, bbox_inches="tight", facecolor="white"), plt.close(fig))
NAME = {"h0": "embedding", "path": "path: all displacements", "wells": "all well frames", "states": "all states after layers"}
for l in range(Lq + 1): NAME[f"h{l}"] = "embedding" if l == 0 else f"state after layer {l - 1}"
for l in range(Lq): NAME[f"D{l}"] = f"displacement, layer {l}"; NAME[f"W{l}"] = f"well frame, layer {l}"

# 1. flow gap
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True, gridspec_kw={"wspace": .08}); cols = [BLUE, AMBER, GREEN, VIOLET, GREY, INK, "#EF4444", "#14B8A6"]
for ax, key, ttl in ((axs[0], "text", "ordinary text"), (axs[1], "premises", "premises")):
    for r in FL[key]:
        k = [1, 2, 4, 8]; y = [r["gap"][str(x)] for x in k]; ax.plot(k, y, marker="o", color=cols[r["layer"]], lw=2, label=f"layer {r['layer']}")
        ax.annotate(f"{y[0]:.2f}", (1, y[0]), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=9.5, color=INK)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8], ["1 (trained step)", "2", "4", "8"]); ax.set_xlim(.6, 10); ax.set_title(ttl, fontsize=11.5); ax.set_xlabel("substeps"); despine(ax)
axs[0].set_ylabel("gap to the 16-substep flow"); axs[0].legend(frameon=False, fontsize=9.5); fig.suptitle(f"How far each layer's step is from its own flow ({s.upper()})", fontweight="bold", y=1.02)
save(fig, f"tdp_{s}_flow.png")

# 2. contrasts
P = Q["primary"]; order = [n for n in P if n != "h0"][::-1]
fig, axs = plt.subplots(1, 2, figsize=(13, .45 * len(order) + 1.6), sharey=True, gridspec_kw={"wspace": .08})
for ax, key, ttl in ((axs[0], "r1", "R1, unseen wordings, without \"unless\""), (axs[1], "p0.01", "paraphrases caught at 1% near misses")):
    for i, n in enumerate(order):
        m, lo, hi = P[n]["minus h0"][key]; col = BLUE if n in ("path", "states", "wells") else GREY
        ax.plot([lo, hi], [i, i], color=col, lw=2); ax.plot(m, i, "o", ms=8, color=col, mec="white", mew=1.5); ax.text(1.0, i, f"{m:+.3f}", va="center", ha="right", fontsize=9.5, color=INK, transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK, lw=1); ax.set_title(ttl, fontsize=11.5); ax.set_xlabel("difference to the embedding"); despine(ax)
axs[0].set_yticks(range(len(order)), [NAME.get(n, n) for n in order]); axs[0].set_xlim(-.5, .12); axs[1].set_xlim(-.35, .2)
save(fig, f"tdp_{s}_contrasts.png")

# 3. against SR2-2B (two-layer readings only)
pairs = [("embedding", "h0", "h0"), ("state after layer 0", "h1", "h1"), ("displacement, layer 0", "D0", "D1"), ("displacement, layer 1", "D1", "D2"), ("all displacements", "path", "D1+D2"), ("all states", "states", "h1+h2"), ("state after layer 1", "h2", "h2"), ("well frame, layer 0", "W0", "W1"), ("well frame, layer 1", "W1", "W2"), ("all well frames", "wells", "W1+W2")]
if Lq == 2:
    fig, ax = plt.subplots(figsize=(11, 4.4)); x = np.arange(len(pairs))
    a = [P[k]["R1 unseen"] for _, k, _ in pairs]; b = [NU[k]["R1 unseen"] for _, _, k in pairs]
    ax.bar(x - .2, a, .38, color=BLUE, label=s.upper()); ax.bar(x + .2, b, .38, color=GREY, label="SR2-2B")
    for xx, v in zip(x - .2, a): ax.text(xx, v + .01, f"{v:.2f}", ha="center", fontsize=8.5)
    for xx, v in zip(x + .2, b): ax.text(xx, v + .01, f"{v:.2f}", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(x, [p[0] for p in pairs], rotation=30, ha="right", fontsize=9.5); ax.set_ylim(.4, 1.05); ax.set_ylabel("R1, unseen wordings, without \"unless\""); ax.legend(frameon=False); despine(ax); ax.set_title("Two two-layer models on the same clauses")
    save(fig, f"tdp_{s}_vs_sr2_2b.png")
