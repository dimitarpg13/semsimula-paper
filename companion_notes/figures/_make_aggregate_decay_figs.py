"""Figures for Pairwise_vs_Aggregate_Decay_in_Semantic_Space.md.

Writes four PNGs to figures/aggregate_decay/:

  agg_two_notions.png     -- the two notions side by side: the per-pair force
      falls, the partner count rises, and their product is what decides
      convergence.
  agg_gravity_shells.png  -- the Newtonian L = 3 case, where the two factors
      cancel exactly and every shell contributes 4*pi per unit thickness.
  agg_locality.png        -- the bounded-region reading: where the force comes
      from, how fast near-field truncation fails with L, and why "thin shell"
      means dr << r/L in high dimension.
  agg_well_vs_parf.png    -- the framework's own contrast: the Gaussian well
      has aggregate decay in every dimension, PARF's inverse square has none.

Every panel is exact; high-dimensional quantities are computed in log space
to avoid overflow. Run:  python3 _make_aggregate_decay_figs.py
"""
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "figure.dpi": 150,
})
GREEN, BLUE, RED = "#22C55E", "#3B82F6", "#EF4444"
PURPLE, GREY, ORANGE = "#8B5CF6", "#9CA3AF", "#F59E0B"
INK, MUTED = "#1F2937", "#6B7280"
OUT = Path(__file__).parent / "aggregate_decay"
OUT.mkdir(exist_ok=True)

KAPPA = math.sqrt(2.0)          # note convention: sigma = 0.25, kappa^2 = 2
L_SEM = 768                     # a representative semantic dimensionality


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- figure 1

def panel_per_pair(ax):
    r = np.linspace(0.6, 6.0, 400)
    ax.plot(r, 1.0 / r**2, color=RED, lw=2.6)
    ax.annotate("", xy=(4.0, 1 / 16), xytext=(2.0, 1 / 4),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    ax.text(3.0, 0.30, "double $r$,\nquarter the pull", color=MUTED,
            fontsize=10, ha="center")
    ax.set_xlabel("separation $r$")
    ax.set_ylabel("force from one partner")
    ax.set_title("A.  Pairwise: one term falls as $1/r^2$")
    ax.set_xlim(0.6, 6.0)
    ax.set_ylim(0, 1.2)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def panel_partner_count(ax):
    r = np.linspace(0.6, 6.0, 400)
    for L, col in ((1, GREY), (2, GREEN), (3, BLUE), (5, PURPLE)):
        ax.plot(r, r ** (L - 1), color=col, lw=2.3, label=f"$L$ = {L}")
    ax.set_xlabel("radius $r$")
    ax.set_ylabel(r"partners per unit thickness, $\propto r^{L-1}$")
    ax.set_title("B.  The factor people forget: the count rises")
    ax.set_xlim(0.6, 6.0)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 2e3)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.set_axisbelow(True)


def panel_integrand(ax):
    r = np.linspace(0.6, 6.0, 400)
    for L, col in ((1, GREY), (2, GREEN), (3, BLUE), (5, PURPLE), (10, RED)):
        ax.plot(r, r ** (L - 3), color=col, lw=2.3, label=f"$L$ = {L}")
    ax.axhline(1.0, color=INK, lw=1.0, ls=":")
    ax.text(5.9, 1.25, "$L = 3$: exact cancellation", color=BLUE, fontsize=9.5,
            ha="right")
    ax.text(0.72, 4e-2, "converges\n($L < 2$)", color=MUTED, fontsize=9.5)
    ax.text(3.6, 2e2, "diverges\n($L \\geq 2$)", color=MUTED, fontsize=9.5)
    ax.set_xlabel("radius $r$")
    ax.set_ylabel(r"shell contribution, $\propto r^{L-3}$")
    ax.set_title("C.  Aggregate: the product decides")
    ax.set_xlim(0.6, 6.0)
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e4)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9, ncol=2)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.set_axisbelow(True)


def fig_two_notions():
    fig = plt.figure(figsize=(16.0, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.30)
    panel_per_pair(fig.add_subplot(gs[0, 0]))
    panel_partner_count(fig.add_subplot(gs[0, 1]))
    panel_integrand(fig.add_subplot(gs[0, 2]))
    fig.suptitle("Pairwise decay is a property of one term; aggregate decay is "
                 "a property of the sum",
                 fontsize=13, fontweight="bold", y=1.03)
    save(fig, "agg_two_notions.png")


# ---------------------------------------------------------------- figure 2

SHELLS = [(1.0, 2.0), (10.0, 11.0), (100.0, 101.0)]


def shell_stats(a, b):
    count = 4.0 * math.pi / 3.0 * (b**3 - a**3)
    mid = 0.5 * (a + b)
    per = 1.0 / mid**2
    return count, per, count * per


def panel_shell_bars(ax):
    labels = [f"{int(a)} to {int(b)}" for a, b in SHELLS]
    counts, pers, totals = zip(*(shell_stats(a, b) for a, b in SHELLS))
    x = np.arange(len(SHELLS))
    w = 0.34
    ax.bar(x - w / 2, counts, w, color=BLUE, label="partners in the shell")
    ax.bar(x + w / 2, pers, w, color=RED, label="force from each partner")
    for xi, (c, p) in enumerate(zip(counts, pers)):
        ax.text(xi - w / 2, c * 1.5, f"{c:,.0f}", ha="center", fontsize=9,
                color=BLUE)
        ax.text(xi + w / 2, p * 1.5, f"{p:.2g}", ha="center", fontsize=9,
                color=RED)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("shell, unit thickness")
    ax.set_ylabel("magnitude (log scale)")
    ax.set_title("A.  The two factors move in step ($L = 3$)")
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.6, axis="y", which="both")
    ax.set_axisbelow(True)
    ax.axhline(3e7, color=GREY, lw=0.8, ls=":")
    ax.text(0.015, 0.955, "count $\\times$ force from each:", fontsize=9.5,
            color=MUTED, ha="left", va="top", transform=ax.transAxes)
    for xi, t in enumerate(totals):
        ax.text(xi, 2.5e8, f"{t:.1f}", ha="center", fontsize=11.5,
                color=INK, fontweight="bold")


def panel_running_total(ax):
    R = np.linspace(0.0, 100.0, 500)
    ax.plot(R, 4.0 * math.pi * R, color=PURPLE, lw=2.6)
    ax.axhline(0, color=GREY, lw=0.8)
    for a, b in SHELLS:
        ax.plot([b], [4.0 * math.pi * b], marker="o", color=PURPLE, ms=6,
                zorder=5)
    ax.text(52, 4 * math.pi * 52 * 0.55,
            r"total $\approx 4\pi n \mathcal{C} R$" "\n"
            "grows without bound",
            color=PURPLE, fontsize=10.5)
    ax.set_xlabel("radius $R$ of the occupied region")
    ax.set_ylabel(r"running total of $\|\vec{F}\|$")
    ax.set_title("B.  Every shell adds the same amount")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 4 * math.pi * 105)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def fig_gravity_shells():
    fig = plt.figure(figsize=(12.6, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.26)
    panel_shell_bars(fig.add_subplot(gs[0, 0]))
    panel_running_total(fig.add_subplot(gs[0, 1]))
    fig.suptitle("The Newtonian case: pairwise decay with no aggregate decay "
                 "at all", fontsize=13, fontweight="bold", y=1.03)
    save(fig, "agg_gravity_shells.png")


# ---------------------------------------------------------------- figure 3

DIMS = ((3, GREEN), (10, BLUE), (100, PURPLE), (768, RED))


def panel_cumulative(ax):
    x = np.linspace(0.0, 1.0, 500)
    for L, col in DIMS:
        ax.plot(x, x ** (L - 2), color=col, lw=2.4, label=f"$L$ = {L}")
    ax.axvline(0.5, color=GREY, lw=1.0, ls=":")
    ax.text(0.47, 0.55, "inner half", color=MUTED, fontsize=9.5, rotation=90,
            ha="right")
    ax.set_xlabel("fraction of the region's radius, $r/R$")
    ax.set_ylabel(r"fraction of $\|\vec{F}_i\|$ from inside $r$")
    ax.set_title("A.  In a bounded region, the force sits at the rim")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def panel_inner_half(ax):
    L = np.arange(3, 300)
    ax.semilogy(L, 2.0 ** -(L - 2.0), color=RED, lw=2.5)
    for Lv, col in DIMS:
        if Lv <= 299:
            ax.plot([Lv], [2.0 ** -(Lv - 2)], marker="o", color=col, ms=7,
                    zorder=5)
    ax.annotate("$L = 3$:  50%", xy=(3, 0.5), xytext=(28, 2e-8),
                fontsize=10, color=GREEN,
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2))
    ax.annotate("$L = 10$:  0.4%", xy=(10, 2.0 ** -8), xytext=(60, 1e-16),
                fontsize=10, color=BLUE,
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
    ax.annotate("$L = 100$:  $3\\times 10^{-30}$", xy=(100, 2.0 ** -98),
                xytext=(130, 1e-20), fontsize=10, color=PURPLE,
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.2))
    ax.text(0.97, 0.06, "$L = 768$:  $3\\times 10^{-231}$, far below the axis",
            fontsize=10, color=RED, ha="right", transform=ax.transAxes)
    ax.set_xlabel("semantic dimensionality $L$")
    ax.set_ylabel(r"fraction from $r < R/2$,  $2^{-(L-2)}$")
    ax.set_title("B.  Near-field truncation keeps nothing")
    ax.set_xlim(0, 300)
    ax.set_ylim(1e-90, 8)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.set_axisbelow(True)


def thin_shell_ratio(L, eps):
    """Exact shell volume over its linearisation, computed in log space."""
    log_exact = L * np.log1p(eps) + np.log1p(-np.exp(-L * np.log1p(eps)))
    return np.exp(log_exact - np.log(L * eps))


def panel_thin_shell(ax):
    prod = np.logspace(-2.2, 1.2, 400)          # the product L * dr/r
    for L, col in ((3, GREEN), (10, BLUE), (100, PURPLE), (768, RED)):
        eps = prod / L
        ax.loglog(prod, thin_shell_ratio(L, eps), color=col, lw=2.3,
                  label=f"$L$ = {L}")
    ax.axhline(1.0, color=INK, lw=1.0, ls=":")
    ax.axvline(1.0, color=GREY, lw=1.0, ls="--")
    ax.text(1.25, 3.5e3, "$L\\cdot dr/r = 1$", color=MUTED, fontsize=9.5)
    ax.text(0.014, 1.35, "linearisation good", color=MUTED, fontsize=9.5)
    ax.set_xlabel(r"$L \cdot dr/r$")
    ax.set_ylabel("exact shell volume / linearisation")
    ax.set_title(r"C.  'Thin' means $dr \ll r/L$, not $dr \ll r$")
    ax.set_ylim(0.8, 1e4)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.set_axisbelow(True)


def fig_locality():
    fig = plt.figure(figsize=(16.0, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.30)
    panel_cumulative(fig.add_subplot(gs[0, 0]))
    panel_inner_half(fig.add_subplot(gs[0, 1]))
    panel_thin_shell(fig.add_subplot(gs[0, 2]))
    fig.suptitle("In a bounded region the failure is not divergence but the "
                 "loss of locality", fontsize=13, fontweight="bold", y=1.03)
    save(fig, "agg_locality.png")


# ---------------------------------------------------------------- figure 4

def log_integrand_power(r, L):
    return (L - 3.0) * np.log(r)


def log_integrand_well(r, L, kappa=KAPPA):
    return L * np.log(r) - (kappa * r) ** 2


def panel_contrast(ax, L, rmax, title, parf_note, note_x=0.55):
    r = np.linspace(0.05, rmax, 900)
    lw_ = log_integrand_well(r, L)
    lp_ = log_integrand_power(r, L)
    ax.plot(r, lw_ - lw_.max(), color=GREEN, lw=2.6,
            label="Gaussian well, $r^{L-1}\\cdot r e^{-\\kappa^2 r^2}$")
    ax.plot(r, lp_ - lp_.max(), color=RED, lw=2.4, ls="--",
            label="PARF, $r^{L-1}\\cdot r^{-2}$")
    peak = math.sqrt(L) / (KAPPA * math.sqrt(2.0))
    ax.axvline(peak, color=GREEN, lw=1.0, ls=":")
    ax.annotate(f"well band, $r \\approx {peak:.1f}$",
                xy=(peak, 0.0), xytext=(peak + 0.08 * rmax, -4.2),
                fontsize=10, color=GREEN,
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2))
    ax.text(note_x * rmax, -1.6, parf_note, color=RED, fontsize=10,
            ha="center")
    ax.set_xlabel("radius $r$")
    ax.set_ylabel("log shell contribution (peak at 0)")
    ax.set_title(title)
    ax.set_xlim(0, rmax)
    ax.set_ylim(-18, 2)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def fig_well_vs_parf():
    fig = plt.figure(figsize=(12.6, 4.6))
    gs = fig.add_gridspec(1, 2, wspace=0.26)
    panel_contrast(fig.add_subplot(gs[0, 0]), 3, 6.0, "A.  $L = 3$",
                   "PARF is flat:\nevery shell contributes equally", 0.62)
    panel_contrast(fig.add_subplot(gs[0, 1]), L_SEM, 30.0,
                   f"B.  $L = {L_SEM}$",
                   "PARF is a spike at the rim:\nall of it is far away", 0.72)
    fig.suptitle("The well confines in every dimension; the inverse square "
                 "never does", fontsize=13, fontweight="bold", y=1.03)
    save(fig, "agg_well_vs_parf.png")


def main():
    fig_two_notions()
    fig_gravity_shells()
    fig_locality()
    fig_well_vs_parf()

    print("\nshell table (L = 3, unit density):")
    for a, b in SHELLS:
        c, p, t = shell_stats(a, b)
        print(f"  r = {int(a):3d} to {int(b):3d}:  partners {c:12,.0f}   "
              f"each {p:.6f}   total {t:.4f}")
    print(f"  4*pi = {4 * math.pi:.4f}")

    print("\nfraction of |F| from r < R/2:")
    for Lv, _ in DIMS:
        print(f"  L = {Lv:4d}: {2.0 ** -(Lv - 2):.3e}")

    print("\nthin-shell ratio at r = 10, L = 768:")
    for dr in (1.0, 0.1, 0.01, 0.001):
        print(f"  dr = {dr:<6} L*dr/r = {768 * dr / 10:8.3f}   "
              f"ratio = {thin_shell_ratio(768, dr / 10.0):.4g}")

    print("\nGaussian band centre sqrt(L)/(kappa*sqrt2):")
    for Lv in (3, 768):
        print(f"  L = {Lv:4d}: {math.sqrt(Lv) / (KAPPA * math.sqrt(2)):.3f}")


if __name__ == "__main__":
    main()
