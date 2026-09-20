"""Figures for Eliminating_the_Autograd_Dependency_in_VPhi.md.

Four PNGs into figures/vphi_analytic/. Panels A of the first figure are
MEASURED values quoted from Fock_Inference_Productionization_Plan.md SS8a.3
(A100 80GB, 2026-09-19); everything else is exact arithmetic on the
deployed shapes (d=384, d_l=32, K=16, T=512, TOP_K=16) or a labelled
schematic of a decision.

  vpa_blocker.png    -- why this blocks more than its 3.85% FLOP share
                        suggests. A: measured forward time, Fock vs GPT-2,
                        essentially flat in T (O(T^0.04)) -- the signature
                        of overhead, not arithmetic. B: measured inference
                        peak memory against a consumer-GPU line.
  vpa_hinge.png      -- the analyticity hinge. Which factors of each
                        potential depend on the LIVE h_t, and therefore
                        how many product-rule terms a closed-form gradient
                        carries.
  vpa_cost.png       -- the softmax-gate covariance term, costed three
                        ways: naive pairwise, projected to d_l, and
                        reduced to moments. Memory per layer at T=512.
  vpa_options.png    -- the five options on effort against expected
                        quality risk, with what each unblocks.

Run:  python3 _make_vphi_analytic_figs.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.titleweight": "bold", "figure.dpi": 150})
GREEN, BLUE, RED = "#22C55E", "#3B82F6", "#EF4444"
PURPLE, GREY, ORANGE = "#8B5CF6", "#9CA3AF", "#F59E0B"

D, DL, K, T, TOPK = 384, 32, 16, 512, 16


def fig_blocker():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.5))
    Ts = np.array([128, 256, 512, 1024])
    fock = np.array([334.1, 358.1, 346.9, 373.4])     # measured, SS8a.3
    gpt2 = np.array([3.62, 3.68, 3.91, 4.16])
    axA.plot(Ts, fock, "-o", color=RED, lw=2.2, ms=8, label="Fock L=8")
    axA.plot(Ts, gpt2, "-o", color=BLUE, lw=2.2, ms=8, label="GPT-2")
    for t, f, g in zip(Ts, fock, gpt2):
        axA.annotate(f"{f/g:.0f}x", (t, f), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=9.5,
                     color=RED, fontweight="bold")
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xticks(Ts); axA.set_xticklabels([str(t) for t in Ts])
    axA.set_yticks([3, 10, 30, 100, 300])
    axA.set_yticklabels(["3", "10", "30", "100", "300"])
    axA.xaxis.set_minor_formatter(NullFormatter())
    axA.yaxis.set_minor_formatter(NullFormatter())
    axA.set_xlabel("context length T"); axA.set_ylabel("forward, ms (batch 1)")
    axA.set_title("A. Measured: flat in T, so it is NOT arithmetic")
    axA.annotate("Fock scales as $T^{0.04}$:\noverhead, not MACs",
                 (300, 350), xytext=(140, 90), fontsize=10, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    axA.legend(fontsize=10, loc="lower right"); axA.grid(alpha=.25, which="major")

    lbl = ["b1\nT=128", "b4\nT=1024"]
    mem = [32.8, 49.7]
    axB.bar(lbl, mem, color=RED, alpha=.85, width=.5, zorder=3)
    for i, m in enumerate(mem):
        axB.text(i, m + 1.2, f"{m:.1f} GB", ha="center",
                 fontsize=11, fontweight="bold")
    axB.axhline(24, color=GREEN, ls="--", lw=2, zorder=2)
    axB.text(1.42, 25, "24 GB consumer GPU", color=GREEN, fontsize=9.5,
             ha="right", fontweight="bold")
    axB.axhline(80, color=GREY, ls=":", lw=1.6, zorder=2)
    axB.text(1.42, 81, "A100 80 GB", color=GREY, fontsize=9.5, ha="right")
    axB.set_ylim(0, 92); axB.set_ylabel("inference peak memory (GB)")
    axB.set_title("B. Measured: INFERENCE memory, no no&#95;grad path".replace("&#95;", "_"))
    axB.grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig("vphi_analytic/vpa_blocker.png", bbox_inches="tight")
    print("wrote vpa_blocker.png")


def fig_hinge():
    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    rows = [
        ("$V_\\theta$  anisotropic Gaussian",
         [("mu,a,w,B\nfrom xi", 0), ("quadratic\nin h", 1)], 1,
         "params from xi ONLY  ->  exactly Gaussian in h"),
        ("$V_\\phi$  structural competitive",
         [("1/r", 1), ("$\\Theta$\naligner", 1), ("$\\Phi$\ngate", 1),
          ("softmax\nrow", 1)], 4,
         "gate params depend on h_t  ->  4 product-rule terms"),
        ("family A  xi-routed attention",
         [("alpha\ndetached", 0), ("$\\phi$\nkernel", 1)], 1,
         "alpha from DETACHED xi  ->  hinge restored"),
    ]
    y = 0
    for name, factors, nterm, note in rows:
        x = 0.0
        for lab, live in factors:
            c = RED if live else GREEN
            ax.barh(y, 1.0, left=x, height=.72, color=c, alpha=.85,
                    edgecolor="white", lw=2, zorder=3)
            ax.text(x + .5, y, lab, ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold",
                    linespacing=1.15)
            x += 1.0
        ax.text(-0.12, y, name, ha="right", va="center", fontsize=11)
        ax.text(x + .15, y, f"{nterm} term" + ("s" if nterm > 1 else "") +
                f"   |   {note}", ha="left", va="center", fontsize=9.5,
                color="#374151")
        y -= 1
    ax.set_xlim(-2.6, 8.4); ax.set_ylim(-2.7, .7)
    ax.axis("off")
    ax.set_title("The analyticity hinge: RED factors depend on the live "
                 "$h_t$", loc="left", fontsize=12.5)
    ax.text(-2.55, -2.45, "green = depends only on context or detached "
            "state  |  red = depends on the live h_t and so enters the "
            "product rule", fontsize=9.5, color=GREY)
    fig.tight_layout()
    fig.savefig("vphi_analytic/vpa_hinge.png", bbox_inches="tight")
    print("wrote vpa_hinge.png")


def fig_cost():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.4))
    naive = T * T * D * 4 / 2**20            # (T,T,d) fp32, MB
    proj = T * T * DL * 4 / 2**20            # (T,T,d_l)
    moment = (T * T + T * DL) * 4 / 2**20    # scores + per-token moments
    names = ["naive\n$(T,T,d)$", f"projected\n$(T,T,d_l)$",
             "moments\n$(T,T) + (T,d_l)$"]
    vals = [naive, proj, moment]
    cols = [RED, ORANGE, GREEN]
    axA.bar(names, vals, color=cols, alpha=.85, zorder=3)
    for i, v in enumerate(vals):
        axA.text(i, v * 1.28, (f"{v:.1f} MB" if v >= 1 else f"{v*1024:.0f} KB"),
                 ha="center", fontsize=10.5, fontweight="bold")
    axA.set_yscale("log"); axA.set_ylabel("memory per layer (MB, fp32)")
    axA.set_ylim(0.6, naive * 4.0)
    axA.set_title(f"A. Softmax-gate term at T={T}, d={D}, $d_l$={DL}")
    axA.yaxis.set_minor_formatter(NullFormatter())
    axA.grid(alpha=.25, axis="y", which="major")
    axA.annotate(f"{naive/moment:,.0f}x", xy=(2, moment), xytext=(1.1, naive*0.5),
                 fontsize=13, color=GREEN, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))

    fl = {"forward scores\n$O(T^2 d_l)$": T*T*DL,
          "radial term\n$O(T^2 d)$ -> top-k": T*TOPK*D,
          "gate covariance\n$O(T^2 d_l)$": T*T*DL,
          "aligner term\n$O(T^2 K)$": T*T*K}
    axB.bar(list(fl), [v/1e6 for v in fl.values()],
            color=[GREY, BLUE, GREEN, PURPLE], alpha=.85, zorder=3)
    for i, v in enumerate(fl.values()):
        axB.text(i, v/1e6*1.04, f"{v/1e6:.1f}", ha="center", fontsize=10)
    axB.set_ylabel("MMAC per token per layer")
    axB.set_title("B. The analytic gradient costs about one more forward")
    axB.tick_params(axis="x", labelsize=9)
    axB.grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig("vphi_analytic/vpa_cost.png", bbox_inches="tight")
    print("wrote vpa_cost.png")


def fig_options():
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    opts = [
        ("A  ablate V_phi at eval", 0.3, 7.0, RED,
         "minutes; may cost real PPL"),
        ("C  bilinear Theta,\n    constant c", 2.2, 3.0, ORANGE,
         "config flags exist; needs retrain"),
        ("B  full analytic grad", 6.5, 0.4, GREEN,
         "exact; no quality change at all"),
        ("D  family A replaces V_phi", 7.5, 5.5, PURPLE,
         "trained alternative, own hinge"),
        ("E  keep autograd,\n    narrow the graph", 3.0, 0.2, GREY,
         "halves memory; export STILL blocked"),
    ]
    for name, x, y, c, note in opts:
        ax.scatter([x], [y], s=520, color=c, alpha=.85, zorder=3,
                   edgecolors="white", lw=2)
        ax.annotate(name, (x, y), xytext=(0, 26), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color=c)
        ax.annotate(note, (x, y), xytext=(0, -30),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color="#374151")
    ax.set_xlim(-1.2, 9.6); ax.set_ylim(-1.6, 9.2)
    ax.set_xlabel("implementation effort  ->")
    ax.set_ylabel("risk to perplexity  ->")
    ax.set_title("Five ways out, and what each actually buys")
    ax.axhline(1.0, color=GREEN, ls="--", lw=1.4, alpha=.7)
    ax.text(9.4, 1.25, "below: quality provably unchanged", color=GREEN,
            fontsize=9, ha="right")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig("vphi_analytic/vpa_options.png", bbox_inches="tight")
    print("wrote vpa_options.png")


if __name__ == "__main__":
    fig_blocker(); fig_hinge(); fig_cost(); fig_options()
