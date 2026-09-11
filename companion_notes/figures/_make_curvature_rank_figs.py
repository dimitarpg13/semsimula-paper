"""Figures for Curvature_Diagnostics_and_Rank_Selection_for_Aniso_Gaussian_Vtheta.md.

Four PNGs into this folder. Every panel is either an exact evaluation of a
formula stated in that note, or a clearly-labelled synthetic illustration of
a decision rule -- nothing is fit to unpublished data, and the measured
numbers that do appear are cited from the companion notes:

  cr_two_channels.png      -- the diagonal/low-rank asymmetry. Panel A: the
                              reachable curvature of each channel on a log
                              axis (diagonal capped at precision_max = 2/d
                              = 0.0052; low-rank ambient sigma_max(B_k)^2
                              percentiles 283/670/1054/6362 from the
                              Mitigations note SS42.4's bracket table), i.e.
                              the ~54,000x gap. Panel B: which channel each
                              integrator treats exactly vs by explicit kick.
  cr_spectrum_pr.png       -- participation ratio as the effective-rank
                              metric: four example spectra at rank 4 with
                              their exact PR, plus the decision bands used
                              by spectrum_across_checkpoints().
  cr_rank_budget.png       -- under a BINDING Frobenius cap, sum_i s_i^2 is
                              pinned, so rank is a redistribution knob:
                              sigma_max^2 vs rank for flat vs degenerate
                              spectra, and the resulting worst-case
                              parameter-gradient proxy (SS3: ~sigma_max^2).
  cr_rank_selection.png    -- the proposed rank-selection procedure: PR(r)
                              knee curves for three intrinsic dimensionality
                              regimes, and a per-well PR histogram showing
                              why one global rank over-serves some wells
                              while starving others.

Run:  python3 _make_curvature_rank_figs.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

GREEN = "#22C55E"
BLUE = "#3B82F6"
RED = "#EF4444"
PURPLE = "#8B5CF6"
GREY = "#9CA3AF"
DARK = "#111827"
AMBER = "#F59E0B"

D_MODEL = 384
PRECISION_MAX = 2.0 / D_MODEL          # Cell 5: _prec_max = 2.0 / d
A_INIT = np.log1p(np.exp(-np.log(D_MODEL))) + 1e-4   # softplus(-log d) + 1e-4
# Mitigations note SS42.4, healthy step-27,000 column of the bracket table.
LR_PCTILES = {"p50": 283.2, "p90": 669.9, "p99": 1054.0, "max": 6362.1}


def participation_ratio(s):
    """PR = (sum s_i^2)^2 / sum s_i^4, in [1, r]. s = singular values."""
    s2 = np.asarray(s, dtype=float) ** 2
    denom = (s2 ** 2).sum()
    return (s2.sum() ** 2) / denom if denom > 0 else 1.0


# ---------------------------------------------------------------------------
def fig_two_channels():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.5))

    # -- Panel A: reachable curvature, log axis -----------------------------
    labels = ["diagonal\ninit  $a_k$", "diagonal\nceiling $2/d$",
              "low-rank\n$p_{50}$", "low-rank\n$p_{90}$",
              "low-rank\n$p_{99}$", "low-rank\nmax"]
    vals = [A_INIT, PRECISION_MAX, LR_PCTILES["p50"], LR_PCTILES["p90"],
            LR_PCTILES["p99"], LR_PCTILES["max"]]
    colors = [GREEN, GREEN, RED, RED, RED, RED]
    bars = axA.bar(range(len(vals)), vals, color=colors, alpha=0.85,
                   edgecolor=DARK, linewidth=0.6)
    axA.set_yscale("log")
    axA.set_xticks(range(len(vals)))
    axA.set_xticklabels(labels, fontsize=8.6)
    axA.set_ylabel("reachable curvature  (log scale)")
    axA.set_title("A. All the curvature lives in the dangerous channel")
    for b, v in zip(bars, vals):
        axA.text(b.get_x() + b.get_width() / 2, v * 1.5, f"{v:,.4g}",
                 ha="center", fontsize=8.2, color=DARK)
    ratio = LR_PCTILES["p50"] / PRECISION_MAX
    axA.annotate("", xy=(2, LR_PCTILES["p50"]), xytext=(1, PRECISION_MAX),
                 arrowprops=dict(arrowstyle="<->", color=AMBER, lw=2.0))
    axA.text(1.5, np.sqrt(PRECISION_MAX * LR_PCTILES["p50"]) * 1.8,
             f"~{ratio:,.0f}x", color=AMBER, fontsize=11, fontweight="bold",
             ha="center")
    axA.set_ylim(1e-3, 5e4)
    axA.grid(alpha=0.25, axis="y")
    axA.legend(handles=[
        Line2D([], [], color=GREEN, lw=7, alpha=0.85,
               label="diagonal: integrated EXACTLY (safe)"),
        Line2D([], [], color=RED, lw=7, alpha=0.85,
               label="low-rank: explicit kick (has the wall)"),
    ], fontsize=8.6, loc="upper left")

    # -- Panel B: integrator treatment schematic ---------------------------
    axB.axis("off")
    axB.set_title("B. Which channel each integrator treats exactly")
    rows = [
        ("verlet",            "explicit", "explicit"),
        ("baoab_cfc",         "EXACT",    "explicit"),
        ("baoab_cfc_lowrank", "EXACT",    "EXACT"),
    ]
    y0, dy = 0.74, 0.17
    axB.text(0.04, y0 + 0.13, "integrator", fontsize=9.6, fontweight="bold")
    axB.text(0.46, y0 + 0.13, "diagonal\n$\\mathrm{diag}(a_k)$", fontsize=9.6,
             fontweight="bold", ha="center")
    axB.text(0.80, y0 + 0.13, "low-rank\n$B_kB_k^{\\top}$", fontsize=9.6,
             fontweight="bold", ha="center")
    for i, (name, diag, lr) in enumerate(rows):
        y = y0 - i * dy
        weight = "bold" if name == "baoab_cfc" else "normal"
        axB.text(0.04, y, name, fontsize=9.6, family="monospace",
                 fontweight=weight)
        for x, val in ((0.46, diag), (0.80, lr)):
            c = GREEN if val == "EXACT" else RED
            axB.add_patch(Rectangle((x - 0.10, y - 0.035), 0.20, 0.075,
                                     facecolor=c, alpha=0.22,
                                     edgecolor=c, linewidth=1.1))
            axB.text(x, y, val, fontsize=9.0, ha="center", color=DARK)
    axB.text(0.04, 0.20,
             "Production runs `baoab_cfc`: the diagonal no longer has a\n"
             "stability wall, yet it is the channel still capped at the\n"
             "Verlet-era $2/d$. The low-rank channel keeps the wall and\n"
             "was uncapped entirely until step 47,121.",
             fontsize=9.0, color=DARK, va="top")
    axB.set_xlim(0, 1)
    axB.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig("cr_two_channels.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_spectrum_pr():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.4))
    r = 4
    cases = [
        ("flat  (1, 1, 1, 1)",          [1.0, 1.0, 1.0, 1.0],   GREEN),
        ("mild  (1, .8, .6, .4)",       [1.0, 0.8, 0.6, 0.4],   BLUE),
        ("steep (1, .3, .1, .03)",      [1.0, 0.3, 0.1, 0.03],  AMBER),
        ("rank-1 (1, 0, 0, 0)",         [1.0, 0.0, 0.0, 0.0],   RED),
    ]
    idx = np.arange(1, r + 1)
    for name, s, c in cases:
        pr = participation_ratio(s)
        axA.plot(idx, np.array(s) / s[0], "o-", color=c, lw=2.0, ms=6,
                 label=f"{name}   PR = {pr:.2f}")
    axA.set_xticks(idx)
    axA.set_xlabel("singular value index $i$")
    axA.set_ylabel(r"$\sigma_i / \sigma_1$")
    axA.set_title("A. Four spectra at rank 4, and their exact PR")
    axA.grid(alpha=0.25)
    axA.legend(fontsize=8.6, loc="upper right")
    axA.set_ylim(-0.05, 1.25)

    # -- Panel B: decision bands -------------------------------------------
    prs = [participation_ratio(s) for _, s, _ in cases]
    colors = [c for _, _, c in cases]
    names = [n.split()[0] for n, _, _ in cases]
    axB.barh(range(len(prs)), prs, color=colors, alpha=0.85,
             edgecolor=DARK, linewidth=0.6)
    axB.axvspan(0.75 * r, r, color=GREEN, alpha=0.13)
    axB.axvspan(1.0, 0.5 * r, color=RED, alpha=0.13)
    axB.axvline(0.5 * r, color=RED, ls="--", lw=1.4)
    axB.axvline(0.75 * r, color=GREEN, ls="--", lw=1.4)
    axB.set_ylim(-0.62, 4.35)
    axB.text(0.5 * r * 0.52, 4.02, "budget NOT used\nrank 8 wasted",
             fontsize=8.4, color=RED, ha="center", va="center")
    axB.text((0.75 * r + r) / 2, 4.02, "SATURATED\nrank 8 justified",
             fontsize=8.4, color="#15803D", ha="center", va="center")
    axB.text(0.625 * r, 4.02, "ambiguous", fontsize=8.4, color=DARK,
             ha="center", va="center")
    axB.set_yticks(range(len(prs)))
    axB.set_yticklabels(names, fontsize=9)
    axB.set_xlim(0, r + 0.15)
    axB.set_xlabel(f"participation ratio   PR $\\in$ [1, {r}]")
    axB.set_title("B. The decision bands the tooling applies")
    axB.grid(alpha=0.25, axis="x")
    for i, p in enumerate(prs):
        axB.text(p + 0.06, i, f"{p:.2f}", va="center", fontsize=8.6)

    fig.tight_layout()
    fig.savefig("cr_spectrum_pr.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_rank_budget():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ranks = np.arange(1, 17)
    budget = 1.0                       # ||B_k||_F^2 <= PRECISION_LR_MAX = 1.0

    flat = budget / ranks              # sigma_max^2 when spectrum is flat
    degen = np.full_like(ranks, budget, dtype=float)

    axA.plot(ranks, flat, "o-", color=GREEN, lw=2.2, ms=5,
             label=r"flat spectrum:  $\sigma_{\max}^2 = b/r$")
    axA.plot(ranks, degen, "s--", color=RED, lw=2.2, ms=5,
             label=r"degenerate (rank-1):  $\sigma_{\max}^2 = b$")
    axA.fill_between(ranks, flat, degen, color=GREY, alpha=0.18)
    axA.text(11.0, 0.52, "the range rank BUYS you\n(only if the spectrum spreads)",
             fontsize=8.8, color=DARK, ha="center")
    for rr, lbl in ((4, "current"), (8, "proposed")):
        axA.axvline(rr, color=PURPLE, ls=":", lw=1.5)
        axA.text(rr, 1.06, lbl, color=PURPLE, fontsize=8.6, ha="center")
    axA.set_xlabel("rank $r$")
    axA.set_ylabel(r"$\sigma_{\max}(B_k)^2$   at a binding Frobenius cap")
    axA.set_title("A. Under a binding cap, rank redistributes -- it does not add")
    axA.set_xticks(ranks[::2])
    axA.set_ylim(0, 1.15)
    axA.grid(alpha=0.25)
    axA.legend(fontsize=8.8, loc="center left", framealpha=0.95)

    # -- Panel B: spike-magnitude proxy -------------------------------------
    axB.plot(ranks, flat, "o-", color=GREEN, lw=2.2, ms=5,
             label="flat spectrum")
    axB.plot(ranks, degen, "s--", color=RED, lw=2.2, ms=5,
             label="degenerate spectrum")
    axB.set_yscale("log")
    axB.set_xlabel("rank $r$")
    axB.set_ylabel("worst-case parameter-gradient proxy\n"
                   r"$\|\nabla_\theta \mathcal{L}\| \sim \sigma_{\max}(B_k)^2$")
    axB.set_title("B. Same quantity sets the spike magnitude (SS3)")
    axB.set_xticks(ranks[::2])
    axB.grid(alpha=0.25, which="both")
    axB.set_ylim(0.04, 3.2)
    axB.annotate(f"rank 4 -> 8 halves it\n({flat[3]:.3f} -> {flat[7]:.3f})",
                 xy=(8, flat[7]), xytext=(9.8, 0.42),
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6),
                 fontsize=8.8, color="#15803D")
    axB.text(8.5, 1.45, "degenerate: rank buys nothing at all",
             fontsize=8.8, color=RED, ha="center")
    axB.legend(fontsize=8.8, loc="lower left")

    fig.tight_layout()
    fig.savefig("cr_rank_budget.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_rank_selection():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ranks = np.arange(1, 13)

    # Synthetic illustration: a well whose data-optimal intrinsic
    # dimensionality is k* saturates PR(r) ~ min(r, k*), softened.
    def pr_curve(k_star, sharpness=2.2):
        return k_star * np.tanh((ranks / k_star) ** sharpness) ** (1 / sharpness)

    for k_star, c, name in ((2, RED, "intrinsic dim $\\approx$ 2"),
                            (6, BLUE, "intrinsic dim $\\approx$ 6"),
                            (11, GREEN, "intrinsic dim $\\approx$ 11")):
        axA.plot(ranks, pr_curve(k_star), "o-", color=c, lw=2.1, ms=5,
                 label=name)
        knee = k_star
        axA.plot([knee], [pr_curve(k_star)[knee - 1]], "*", color=c, ms=16,
                 markeredgecolor=DARK, markeredgewidth=0.5)
    axA.plot(ranks, ranks, "--", color=GREY, lw=1.5,
             label="PR = r  (budget fully used)")
    axA.set_xlabel("rank $r$ the model is given")
    axA.set_ylabel("measured PR at that rank")
    axA.set_title("A. Rank selection: find the knee where PR stops tracking $r$")
    axA.set_xticks(ranks)
    axA.grid(alpha=0.25)
    axA.legend(fontsize=8.6, loc="upper left")
    axA.text(7.4, 2.0, "stars mark the knee =\nproposed rank for that well",
             fontsize=8.6, color=DARK, ha="center")

    # -- Panel B: per-well heterogeneity ------------------------------------
    rng = np.random.default_rng(0)
    # Illustrative mixture: many near-degenerate wells, a tail that uses rank 4.
    pr_wells = np.clip(
        np.concatenate([
            rng.normal(1.35, 0.22, 150),
            rng.normal(2.40, 0.45, 110),
            rng.normal(3.55, 0.30, 60),
        ]), 1.0, 4.0)
    axB.hist(pr_wells, bins=28, range=(1, 4), color=BLUE, alpha=0.78,
             edgecolor=DARK, linewidth=0.5)
    axB.axvline(np.median(pr_wells), color=DARK, lw=2.0,
                label=f"median PR = {np.median(pr_wells):.2f}")
    axB.axvspan(1.0, 2.0, color=RED, alpha=0.10)
    axB.axvspan(3.0, 4.0, color=GREEN, alpha=0.10)
    axB.set_ylim(0, 41)
    axB.text(1.5, 37.0, "over-served\n(rank could drop)",
             fontsize=8.4, color=RED, ha="center", va="center")
    axB.text(3.5, 37.0, "starved\n(wants more rank)",
             fontsize=8.4, color="#15803D", ha="center", va="center")
    axB.set_xlabel("per-well participation ratio at rank 4")
    axB.set_ylabel("number of (layer, channel, well) triples")
    axB.set_title("B. Why a single global rank is the wrong knob\n"
                  "(illustrative distribution)")
    axB.legend(fontsize=8.8, loc="center right", framealpha=0.95)
    axB.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig("cr_rank_selection.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_two_channels()
    fig_spectrum_pr()
    fig_rank_budget()
    fig_rank_selection()
    print("wrote cr_two_channels.png, cr_spectrum_pr.png, "
          "cr_rank_budget.png, cr_rank_selection.png")
