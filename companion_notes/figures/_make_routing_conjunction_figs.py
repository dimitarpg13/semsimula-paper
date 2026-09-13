"""Figure for Gradient_Spikes_as_Routing_Conjunctions.md.

One PNG. Every number is measured on an A100 (2026-09-13) against the
step-87196/86201/90360 spikebatch captures -- nothing here is synthetic.

  rc_conjunction.png  -- Panel A: each captured spike collapses to the
                         ordinary baseline when the RNG is reset so every
                         microbatch draws the same routing noise. Panel B:
                         the spiking batch of step 87196 replayed under 20
                         fresh routing draws, against the draw training
                         actually gave it (harness validated to 0.05%).

Run:  python3 _make_routing_conjunction_figs.py
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.titleweight": "bold", "figure.dpi": 150})
RED, BLUE, GREY, DARK, AMBER = "#EF4444", "#3B82F6", "#9CA3AF", "#111827", "#F59E0B"

STEPS = [(87196, 2539.20, 2.71), (86201, 685.56, 1.19), (90360, 567.26, 2.15)]
# 20 alternative routing draws, rescaled to the in-context (loss/4) convention
DRAWS = np.array([31.93, 62.07, 7.50, 10.97, 4.12, 2.77, 1.76, 4.36, 1.99,
                  2.83, 1.43, 2.77, 1.53, 1.89, 4.94, 9.88, 2.96, 1.66,
                  3.40, 4.89]) / 4.0
REAL = 2538.96

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.5))

x = np.arange(len(STEPS))
axA.bar(x - 0.2, [s[1] for s in STEPS], 0.38, color=RED, alpha=0.85,
        edgecolor=DARK, lw=0.6, label="as trained")
axA.bar(x + 0.2, [s[2] for s in STEPS], 0.38, color=BLUE, alpha=0.85,
        edgecolor=DARK, lw=0.6, label="RNG reset per microbatch")
axA.set_yscale("log"); axA.set_ylim(0.5, 1e4)
axA.set_xticks(x); axA.set_xticklabels([f"step {s[0]}" for s in STEPS])
axA.set_ylabel("pre-clip gradient norm")
axA.set_title("A. Every captured spike is a routing draw")
axA.legend(fontsize=9, loc="upper right")
for xi, (_, a, b) in zip(x, STEPS):
    axA.text(xi - 0.2, a * 1.3, f"{a:,.0f}", ha="center", fontsize=8.5)
    axA.text(xi + 0.2, b * 1.3, f"{b:.2f}", ha="center", fontsize=8.5)
    axA.annotate("", xy=(xi + 0.2, b * 1.9), xytext=(xi - 0.2, a * 0.75),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    axA.text(xi + 0.30, b * 3.4, f"{a/b:.0f}x", ha="left", fontsize=10,
             color=RED, fontweight="bold")

axB.scatter(np.arange(len(DRAWS)), DRAWS, s=34, color=GREY,
            edgecolor=DARK, lw=0.5, zorder=3, label="20 alternative draws")
axB.axhline(np.median(DRAWS), color=GREY, ls=":", lw=1.4)
axB.scatter([len(DRAWS) / 2], [REAL], s=130, marker="*", color=RED,
            edgecolor=DARK, lw=0.7, zorder=4, label="the draw training gave it")
axB.set_yscale("log"); axB.set_ylim(0.1, 1e4)
axB.set_xlabel("routing draw (same batch, same weights)")
axB.set_ylabel("pre-clip gradient norm")
axB.set_title("B. The real draw sits far outside the distribution")
axB.legend(fontsize=9, loc="upper left")
axB.annotate(f"{REAL / DRAWS.max():.0f}x the worst\nof twenty",
             xy=(len(DRAWS) / 2, REAL), xytext=(11.5, 160),
             fontsize=9.2, color=DARK,
             arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.8))
axB.text(0.02, 0.04, f"median {np.median(DRAWS):.2f}   max {DRAWS.max():.1f}"
         f"   spread {DRAWS.max()/DRAWS.min():.0f}x",
         transform=axB.transAxes, fontsize=8.6, color=GREY)

fig.suptitle("A gradient spike is a coincidence between a batch and a "
             "routing draw", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("rc_conjunction.png", bbox_inches="tight")
print(f"draws: median {np.median(DRAWS):.2f}  max {DRAWS.max():.2f}  "
      f"real/max {REAL/DRAWS.max():.0f}x  real/median {REAL/np.median(DRAWS):.0f}x")
