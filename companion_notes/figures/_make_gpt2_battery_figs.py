"""Figure for semantic_templates/GPT2_Template_Battery_Results.md (the matched GPT-2 W2 read by the template tests, 2026-11-27).

Reads results/template_matcher/template_summary_state_{np,g2,l4,w2}.json (task A) and
template_summary_state_blogic_control_{np,g2,l4,w2}.json (the B-logic control read at "Therefore"). Writes into figures/template_matcher/:
  gp_depth.png   per state (0 = the embedding, l = after block or layer l): task A's rank and the control's twin reading averaged
                 over the two word orders, for the three ladder models and GPT-2; and GPT-2's twin reading in each order
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY = "#1F2937", "#6B7280", "#9CA3AF"
HERE = Path(__file__).resolve().parent; OUT = HERE / "template_matcher"; RES = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/template_matcher"
MODELS = (("np", "SR2-NP (2 layers)", "#3B82F6"), ("g2", "G2 (2 layers)", "#DC2626"), ("l4", "L4 (4 layers)", "#059669"), ("w2", "GPT-2 W2 (8 blocks)", "#F59E0B"))
A = {s: json.loads((RES / f"template_summary_state_{s}.json").read_text()) for s, _, _ in MODELS}
B = {s: json.loads((RES / f"template_summary_state_blogic_control_{s}.json").read_text()) for s, _, _ in MODELS}
states = lambda q: sorted((k for k in q if k[0] == "h" and k[1:].isdigit()), key=lambda k: int(k[1:]))

fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
for s, nm, c in MODELS:
    ks = states(A[s]); x = [int(k[1:]) for k in ks]; r = [A[s][k]["rank"] for k in ks]
    axs[0].errorbar(x, [v[0] for v in r], yerr=[[v[0] - v[1] for v in r], [v[2] - v[0] for v in r]], fmt="o-", color=c, ms=5, lw=1.6, capsize=0, label=nm)
    ks = states(B[s]); x = [int(k[1:]) for k in ks]
    tw = [0.5 * (B[s][k]["analogy"]["BCa twin"][0] + B[s][k]["analogy"]["BCb twin"][0]) for k in ks]
    axs[1].plot(x, tw, "o-", color=c, ms=5, lw=1.6, label=nm)
axs[0].axhline(0.25, color=MUTED, lw=0.8, ls="--"); axs[0].text(8, 0.26, "chance", color=MUTED, fontsize=9, ha="right")
axs[0].axhline(A["np"]["length"]["rank"][0], color=GREY, lw=0.8, ls=":"); axs[0].text(8, A["np"]["length"]["rank"][0] + 0.01, "length alone", color=MUTED, fontsize=9, ha="right")
axs[0].set_title('task A: the form at "Therefore"', loc="left"); axs[0].set_ylabel("right template ranked first (share)")
axs[1].axhline(0.5, color=MUTED, lw=0.8, ls="--"); axs[1].set_title('B-logic control: the binding at "Therefore"', loc="left"); axs[1].set_ylabel("valid form beats its twin (share, both orders)")
ks = states(B["w2"]); x = [int(k[1:]) for k in ks]
for L, lab, mfc in (("BCa", "bound clause named first", "#F59E0B"), ("BCb", "bound clause named last", "white")):
    v = [B["w2"][k]["analogy"][f"{L} twin"] for k in ks]
    axs[2].errorbar(x, [u[0] for u in v], yerr=[[u[0] - u[1] for u in v], [u[2] - u[0] for u in v]], fmt="o-", color="#F59E0B", mfc=mfc, ms=6, lw=1.4, capsize=0, label=lab)
axs[2].axhline(0.5, color=MUTED, lw=0.8, ls="--"); axs[2].set_title("GPT-2: mention first, binding from block 3", loc="left"); axs[2].set_ylabel("valid form beats its twin (share)")
for ax in axs:
    ax.set_xlabel("state (0 = embedding, l = after block or layer l)"); ax.set_xticks(range(9)); ax.spines[["top", "right"]].set_visible(False)
axs[0].legend(frameon=False, fontsize=9, loc="center right"); axs[2].legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "gp_depth.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote gp_depth.png")
