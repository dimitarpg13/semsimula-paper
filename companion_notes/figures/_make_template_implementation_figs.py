"""Figure for Implementing_Semantic_Templates_A_Path.md.

Writes one PNG to figures/template_implementation/:

  tpl_rate_annealing.png -- the learnable firing rate of (4.4) in the templates
      note, seen three ways:
      A  the rate against the matching overlap, as the sharpness beta anneals
         toward the hard gate of the manuscript's threshold rule;
      B  the same rate against semantic distance, where the gate is a binding
         radius in units of the inflection radius (equation (3.4) there);
      C  the training cost of sharpness: the 10-90% transition width shrinks
         as 1/beta, so the gradient that trains Theta vanishes with it.

Conventions follow the companion notes: kappa^2 = 2, so x* = 1/(kappa sqrt2)
= 0.5 and BC = exp(-u^2/2) with u = d/x*.

Run:  python3 _make_template_implementation_figs.py
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
OUT = Path(__file__).parent / "template_implementation"
OUT.mkdir(exist_ok=True)

THETA = 0.5                      # matching threshold used in panels A and B
BETAS = ((4, GREY), (16, BLUE), (64, PURPLE), (256, GREEN))


def u_of_theta(theta):
    """Binding radius in inflection radii: equation (3.4)."""
    return math.sqrt(2.0 * math.log(1.0 / theta))


def bc_of_u(u):
    return np.exp(-0.5 * u * u)


def rate(bc, beta, theta=THETA):
    return 1.0 / (1.0 + np.exp(-beta * (bc - theta)))


def panel_rate_vs_bc(ax):
    bc = np.linspace(0.0, 1.0, 600)
    for beta, col in BETAS:
        ax.plot(bc, rate(bc, beta), color=col, lw=2.3,
                label=f"$\\beta$ = {beta}")
    ax.plot([0, THETA, THETA, 1], [0, 0, 1, 1], color=RED, lw=2.0, ls="--",
            label=r"$\beta\to\infty$ (hard gate)")
    ax.axvline(THETA, color=GREY, lw=1.0, ls=":")
    ax.text(THETA + 0.02, 0.17, r"$\Theta$", color=MUTED, fontsize=12)
    ax.set_xlabel("matching overlap  BC")
    ax.set_ylabel(r"firing rate  $r_\mathcal{T}/r_0$")
    ax.set_title("A.  The rate is differentiable until it is not")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.04, 1.08)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def panel_rate_vs_distance(ax):
    u = np.linspace(0.0, 3.2, 600)
    bc = bc_of_u(u)
    for beta, col in BETAS:
        ax.plot(u, rate(bc, beta), color=col, lw=2.3,
                label=f"$\\beta$ = {beta}")
    uth = u_of_theta(THETA)
    ax.axvline(uth, color=RED, lw=1.6, ls="--")
    ax.annotate(f"binding radius\n$d/x^*$ = {uth:.2f}",
                xy=(uth, 0.5), xytext=(uth + 0.35, 0.70),
                fontsize=10, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
    ax.set_xlabel("semantic distance  $d/x^*$")
    ax.set_ylabel(r"firing rate  $r_\mathcal{T}/r_0$")
    ax.set_title("B.  In space, the gate is a binding radius")
    ax.set_xlim(0, 3.2)
    ax.set_ylim(-0.04, 1.08)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def transition_width(beta, theta):
    """10-90% width of the gate, in units of x*.

    sigma(beta*(BC-Theta)) spans 0.1 to 0.9 over Delta_BC = 2 ln9 / beta.
    With BC = exp(-u^2/2), |dBC/du| = u*BC, evaluated at the threshold radius.
    """
    uth = u_of_theta(theta)
    return 2.0 * math.log(9.0) / (beta * uth * theta)


def panel_cost(ax):
    beta = np.logspace(0.3, 3.0, 400)
    for theta, col in ((0.9, GREEN), (0.5, BLUE), (0.1, PURPLE)):
        w = [transition_width(b, theta) for b in beta]
        ax.loglog(beta, w, color=col, lw=2.3,
                  label=f"$\\Theta$ = {theta}  ($d/x^*$ = {u_of_theta(theta):.2f})")
    for b, _ in BETAS:
        ax.plot([b], [transition_width(b, THETA)], marker="o", color=INK,
                ms=6, zorder=5)
    ax.text(4.6, transition_width(4, THETA) * 1.35,
            r"$\beta$ = 4, 16, 64, 256", color=INK, fontsize=9.5)
    ax.axhline(0.01, color=GREY, lw=1.0, ls=":")
    ax.text(2.2, 0.0125, "one hundredth of an inflection radius",
            color=MUTED, fontsize=9.5)
    ax.set_xlabel(r"sharpness  $\beta$")
    ax.set_ylabel(r"10-90% gate width  ($d/x^*$)")
    ax.set_title(r"C.  Sharpness costs gradient: width $\propto 1/\beta$")
    ax.set_xlim(2, 1000)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.set_axisbelow(True)


def main():
    fig = plt.figure(figsize=(16.0, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.30)
    panel_rate_vs_bc(fig.add_subplot(gs[0, 0]))
    panel_rate_vs_distance(fig.add_subplot(gs[0, 1]))
    panel_cost(fig.add_subplot(gs[0, 2]))
    fig.suptitle("A learnable template fires through a soft gate; the hard "
                 "threshold is its zero-temperature limit",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.savefig(OUT / "tpl_rate_annealing.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "tpl_rate_annealing.png")

    print("\nbinding radius d/x* by threshold (equation 3.4):")
    for th in (0.9, 0.5, 0.1, 0.031):
        print(f"  Theta = {th:<6} -> {u_of_theta(th):.2f}")
    print("\n10-90% gate width in x*, at Theta = 0.5:")
    for b, _ in BETAS:
        print(f"  beta = {b:<5} -> {transition_width(b, 0.5):.4f}")


if __name__ == "__main__":
    main()
