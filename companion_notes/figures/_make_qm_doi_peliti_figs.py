#!/usr/bin/env python3
"""Figures for Quantum_Formalism_Behind_Doi_Peliti_in_Semantic_Simulation.md.

    python3 _make_qm_doi_peliti_figs.py        # writes figures/qm_doi_peliti/*.png

Everything is computed in the script (no data files). Lengths are in units of the
inflection radius x* of the Gaussian well, with the framework's width sigma = x*/2
(single-particle note, equation (1.3)), so that the overlap of two modes a distance d
apart is exp(-kappa^2 d^2) with kappa^2 = 1 / (2 x*^2).

Colour roles are fixed across all six figures: quantum mechanics = PURPLE,
Doi-Peliti = GREEN, a reference mixture = GREY dashed, a cross term = ORANGE.
"""
from math import factorial, sqrt, exp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN, BLUE, RED, PURPLE, GREY, ORANGE = "#22C55E", "#3B82F6", "#EF4444", "#8B5CF6", "#9CA3AF", "#F59E0B"
INK, MUTED = "#1F2937", "#6B7280"
QM, DP = PURPLE, GREEN
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})

HERE = Path(__file__).parent
OUT = HERE / "qm_doi_peliti"
OUT.mkdir(exist_ok=True)

XSTAR = 1.0
SIGMA = XSTAR / 2
KAPPA2 = 1.0 / (2 * XSTAR ** 2)


def phi(x, mu):
    """Root density of the framework's Gaussian mode, one dimension, unit L2 norm."""
    return (2 * np.pi * SIGMA ** 2) ** (-0.25) * np.exp(-(x - mu) ** 2 / (4 * SIGMA ** 2))


def poisson(alpha, nmax):
    return np.array([exp(-alpha) * alpha ** n / factorial(n) for n in range(nmax + 1)])


# ---------------------------------------------------------------- Figure 1
def two_conventions():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    # A: the ladder of number states, with the matrix elements of a-dagger (up) and a (down)
    N = 4
    for n in range(N + 1):
        a.plot([0.12, 0.42], [n, n], color=INK, lw=1.2)
        a.plot([0.58, 0.88], [n, n], color=INK, lw=1.2)
        a.text(0.0, n, rf"$|{n}\rangle$", ha="right", va="center", fontsize=10)
    for n in range(N):
        # quantum: up sqrt(n+1), down sqrt(n+1) (acting on |n+1>)
        a.annotate("", xy=(0.20, n + 0.93), xytext=(0.20, n + 0.07), arrowprops=dict(arrowstyle="->", color=QM, lw=1.6))
        a.text(0.215, n + 0.5, rf"$\sqrt{{{n + 1}}}$", color=QM, fontsize=9, va="center")
        a.annotate("", xy=(0.34, n + 0.07), xytext=(0.34, n + 0.93), arrowprops=dict(arrowstyle="->", color=QM, lw=1.0, ls="--"))
        a.text(0.355, n + 0.5, rf"$\sqrt{{{n + 1}}}$", color=QM, fontsize=9, va="center", alpha=0.75)
        # Doi-Peliti: up 1, down n+1 (acting on |n+1>)
        a.annotate("", xy=(0.66, n + 0.93), xytext=(0.66, n + 0.07), arrowprops=dict(arrowstyle="->", color=DP, lw=1.6))
        a.text(0.675, n + 0.5, "1", color=DP, fontsize=9, va="center")
        a.annotate("", xy=(0.80, n + 0.07), xytext=(0.80, n + 0.93), arrowprops=dict(arrowstyle="->", color=DP, lw=1.0, ls="--"))
        a.text(0.815, n + 0.5, f"{n + 1}", color=DP, fontsize=9, va="center", alpha=0.85)
    a.text(0.27, N + 0.55, "quantum mechanics", color=QM, ha="center", fontsize=10)
    a.text(0.73, N + 0.55, "Doi–Peliti", color=DP, ha="center", fontsize=10)
    a.text(0.27, -0.75, "solid: create, dashed: annihilate\nnumbers are the matrix elements", color=MUTED,
           ha="center", fontsize=8.5)
    a.text(0.73, -0.75, "same operators, same commutator\nonly the basis normalisation differs", color=MUTED,
           ha="center", fontsize=8.5)
    a.set_xlim(-0.12, 1.0)
    a.set_ylim(-1.1, N + 0.9)
    a.axis("off")
    a.set_title("A. one ladder, two normalisations", loc="left", fontsize=11)
    # B: the squared norm of the n-particle state
    n = np.arange(0, 9)
    w = 0.38
    b.bar(n - w / 2, np.ones_like(n), w, color=QM, label="quantum: ⟨n|n⟩ = 1")
    b.bar(n + w / 2, [factorial(k) for k in n], w, color=DP, label="Doi–Peliti: ⟨n|n⟩ = n!")
    b.set_yscale("log")
    b.set_xlabel("number of particles n")
    b.set_ylabel("squared norm of |n⟩ (log scale)")
    b.set_xticks(n)
    b.legend(frameon=False, fontsize=9, loc="upper left")
    b.set_title("B. what each convention calls the length of |n⟩", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "qdp_two_conventions.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def coherent_readout():
    mean = 0.7
    alpha_q = sqrt(mean)          # quantum amplitude whose Born-rule distribution has mean 0.7
    nmax = 6
    n = np.arange(nmax + 1)
    amp = np.array([exp(-alpha_q ** 2 / 2) * alpha_q ** k / sqrt(factorial(k)) for k in n])
    born = amp ** 2
    dp = poisson(mean, nmax)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    a.bar(n, amp, 0.6, color=QM)
    a.set_title("A. quantum: amplitudes ⟨n|α⟩", loc="left", fontsize=11)
    a.set_xlabel("n")
    a.set_ylabel("amplitude")
    a.text(0.97, 0.95, rf"$\alpha=\sqrt{{0.7}}$" + "\nsum of amplitudes " + f"{amp.sum():.3f}" + "\nsum of squares 1.000",
           transform=a.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)
    b.bar(n, born, 0.6, color=QM, alpha=0.55)
    b.set_title("B. quantum: squared, the Born rule", loc="left", fontsize=11)
    b.set_xlabel("n")
    b.set_ylabel("probability")
    b.text(0.97, 0.95, "Poisson, mean |α|² = 0.7", transform=b.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)
    c.bar(n, dp, 0.6, color=DP)
    c.set_title("C. Doi–Peliti: the coefficients themselves", loc="left", fontsize=11)
    c.set_xlabel("n")
    c.set_ylabel("probability")
    c.text(0.97, 0.95, "Poisson, mean α = 0.7\nno squaring step", transform=c.transAxes, ha="right", va="top",
           fontsize=8.5, color=MUTED)
    for ax, vals in ((b, born), (c, dp)):
        for k, v in zip(n[:4], vals[:4]):
            ax.text(k, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, color=INK)
        ax.set_ylim(0, 0.6)
    a.set_ylim(0, 0.8)
    fig.tight_layout()
    fig.savefig(OUT / "qdp_coherent_readout.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def lowdin_modes():
    d = 0.8 * XSTAR
    g = exp(-KAPPA2 * d ** 2)                  # overlap of the two modes
    x = np.linspace(-2.5, 3.3, 1200)
    pa, pb = phi(x, 0.0), phi(x, d)
    # Loewdin: G^{-1/2} for G = [[1,g],[g,1]]
    lp, lm = (1 + g) ** -0.5, (1 - g) ** -0.5
    s, t = 0.5 * (lp + lm), 0.5 * (lp - lm)   # G^{-1/2} = [[s,t],[t,s]]
    ta, tb = s * pa + t * pb, t * pa + s * pb
    dx = x[1] - x[0]
    gram_t = np.array([[np.sum(ta * ta), np.sum(ta * tb)], [np.sum(tb * ta), np.sum(tb * tb)]]) * dx
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.9))
    a.plot(x, pa, color=BLUE, lw=2, label="φ cat")
    a.plot(x, pb, color=RED, lw=2, label="φ kitten")
    a.fill_between(x, 0, np.minimum(pa, pb), color=ORANGE, alpha=0.35, lw=0, label="where both are large")
    a.axhline(0, color=GREY, lw=0.8)
    a.set_xlabel("position in semantic space, units of x*")
    a.set_ylabel("root density")
    a.set_title(f"A. two Gaussian modes, overlap ⟨φ cat, φ kitten⟩ = {g:.3f}", loc="left", fontsize=11)
    a.legend(frameon=False, fontsize=9)
    b.plot(x, ta, color=BLUE, lw=2, label="φ̃ cat (Löwdin)")
    b.plot(x, tb, color=RED, lw=2, label="φ̃ kitten (Löwdin)")
    b.plot(x, pa, color=BLUE, lw=1, ls=":", alpha=0.7)
    b.plot(x, pb, color=RED, lw=1, ls=":", alpha=0.7)
    b.axhline(0, color=GREY, lw=0.8)
    b.set_xlabel("position in semantic space, units of x*")
    b.set_ylabel("orthonormalised mode")
    b.set_title("B. the orthonormal pair closest to them", loc="left", fontsize=11)
    b.legend(frameon=False, fontsize=9, loc="upper right")
    b.text(0.02, 0.97, f"G^(−1/2): diagonal {s:.3f}, off-diagonal {t:.3f}"
           + "\n" + f"check: ⟨φ̃a,φ̃a⟩ = {gram_t[0, 0]:.4f}, ⟨φ̃a,φ̃b⟩ = {gram_t[0, 1]:+.4f}",
           transform=b.transAxes, ha="left", va="top", fontsize=8.5, color=MUTED)
    b.set_ylim(-0.3, 1.25)
    fig.tight_layout()
    fig.savefig(OUT / "qdp_lowdin_modes.png", dpi=150)
    plt.close(fig)
    return g, s, t


# ---------------------------------------------------------------- Figure 4
def length_vs_sum():
    fig, a = plt.subplots(figsize=(5.6, 5.4))
    th = np.linspace(0, np.pi / 2, 200)
    a.plot(np.cos(th), np.sin(th), color=QM, lw=2.2, label="quantum: unit circle, c₁² + c₂² = 1")
    a.plot([1, 0], [0, 1], color=DP, lw=2.2, label="Doi–Peliti: simplex, c₁ + c₂ = 1, c ≥ 0")
    for k in (0.4, 0.7, 1.3):
        a.plot([k, 0], [0, k], color=DP, lw=0.8, ls=":", alpha=0.6)
    a.annotate("", xy=(0.52, 0.52), xytext=(0.02, 0.02), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    a.text(0.30, 0.19, "⟨1| = (1, 1): the functional\nthat sums the coefficients", color=MUTED, fontsize=8.5)
    q = 1 / sqrt(2)
    a.plot([q], [q], "o", color=QM, ms=8)
    a.text(q + 0.03, q + 0.03, "quantum 'bank'\n(1/√2, 1/√2)", color=QM, fontsize=9)
    a.plot([0.5], [0.5], "o", color=DP, ms=8)
    a.text(0.53, 0.40, "Doi–Peliti 'bank'\n(1/2, 1/2)", color=DP, fontsize=9)
    a.plot([0.5, q], [0.5, q], color=GREY, lw=0.8)
    a.set_xlim(-0.05, 1.4)
    a.set_ylim(-0.05, 1.4)
    a.set_aspect("equal")
    a.set_xlabel("coefficient on mode 1 (river)")
    a.set_ylabel("coefficient on mode 2 (finance)")
    a.set_title("length against sum, in the one-particle sector of two modes", loc="left", fontsize=11)
    a.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "qdp_length_vs_sum.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5
def birth_death():
    c, d = 1.4, 2.0
    t = np.linspace(0, 3, 400)
    alpha = (c / d) * (1 - np.exp(-d * t))
    snaps = [(0.25, BLUE), (0.75, PURPLE), (3.0, GREEN)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1, 1.2]})
    a.plot(t, alpha, color=INK, lw=2)
    a.axhline(c / d, color=GREY, lw=1, ls="--")
    a.text(2.95, c / d + 0.02, "c/d = 0.7", ha="right", fontsize=9, color=MUTED)
    for ts, col in snaps:
        a.plot([ts], [(c / d) * (1 - exp(-d * ts))], "o", color=col, ms=8)
    a.set_xlabel("time t")
    a.set_ylabel("Poisson mean α(t)")
    a.set_title("A. birth at rate c = 1.4, death at rate d = 2 per particle", loc="left", fontsize=11)
    a.text(0.03, 0.95, "α(t) = (c/d)(1 − e^{−dt})\nstarting from the vacuum", transform=a.transAxes, va="top",
           fontsize=9, color=MUTED)
    nmax = 5
    n = np.arange(nmax + 1)
    w = 0.26
    for i, (ts, col) in enumerate(snaps):
        al = (c / d) * (1 - exp(-d * ts))
        b.bar(n + (i - 1) * w, poisson(al, nmax), w * 0.92, color=col, label=f"t = {ts:g}, α = {al:.3f}")
    b.set_xticks(n)
    b.set_xlabel("number of particles n")
    b.set_ylabel("P(n, t)")
    b.set_title("B. the state stays a coherent state at every time", loc="left", fontsize=11)
    b.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "qdp_birth_death.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 6
def superposition_vs_mixture():
    d = 0.6 * XSTAR
    x = np.linspace(-2.2, 2.8, 1200)
    pa, pb = phi(x, 0.0), phi(x, d)
    g = exp(-KAPPA2 * d ** 2)
    mix = 0.5 * pa ** 2 + 0.5 * pb ** 2
    plus = ((pa + pb) / sqrt(2)) ** 2 / (1 + g)      # normalised to a density
    minus = ((pa - pb) / sqrt(2)) ** 2 / (1 - g)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
    for ax, q, lab, sign in ((a, plus, "(φa + φb)² / (2(1 + g))", "+"), (b, minus, "(φa − φb)² / (2(1 − g))", "−")):
        ax.plot(x, mix, color=DP, lw=2.2, ls="--", label="Doi–Peliti one-particle state: ½ pa + ½ pb")
        ax.plot(x, q, color=QM, lw=2.2, label=f"quantum superposition with sign {sign}: {lab}")
        ax.fill_between(x, mix, q, color=ORANGE, alpha=0.3, lw=0, label="difference: the cross term")
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xlabel("position in semantic space, units of x*")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    a.set_ylabel("density")
    a.set_title(f"A. same coefficients, same sign (overlap g = {g:.2f})", loc="left", fontsize=11)
    b.set_title("B. same coefficients, opposite sign: a node appears", loc="left", fontsize=11)
    b.text(0.5 * d, 0.02, "a sign has no meaning\nfor a probability", ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "qdp_superposition_vs_mixture.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    two_conventions()
    coherent_readout()
    g, s, t = lowdin_modes()
    length_vs_sum()
    birth_death()
    superposition_vs_mixture()
    print(f"wrote 6 figures to {OUT}; Löwdin example: overlap {g:.4f}, G^-1/2 diag {s:.4f} offdiag {t:.4f}")
