"""Supplementary figure for Single_Particle_Hilbert_Space_in_Semantic_Simulation.md §7.4.

Adds one PNG to figures/single_particle_space/, alongside the eight produced by
_make_single_particle_space_figs.py (which this script does not touch).

  sps_lowdin_construction.png -- why G^(-1/2) is the right orthonormalisation:
      A  the two-mode geometry, with Loewdin rotating both modes equally and
         Gram-Schmidt pinning the first and moving all the distortion onto the
         second;
      B  the total distortion of the two schemes against the overlap, both in
         closed form;
      C  the price: the Loewdin coefficients and the Gram condition number
         diverge as the modes merge.

Every panel is exact. The marked point is the toy vocabulary's cat-kitten pair,
overlap c = 0.9753 (note §2.3).

Run:  python3 _make_lowdin_figs.py
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
OUT = Path(__file__).parent / "single_particle_space"
OUT.mkdir(exist_ok=True)

C_CK = 0.9753          # cat-kitten overlap, note §2.3


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def lowdin_distortion(c):
    """Exact: 2 [ 2 - sqrt(1+c) - sqrt(1-c) ] = tr (I - G^(1/2))^2 for two modes."""
    return 2.0 * (2.0 - np.sqrt(1.0 + c) - np.sqrt(1.0 - c))


def gs_distortion(c):
    """Exact: 2 - 2 sqrt(1 - c^2); the first mode is left untouched."""
    return 2.0 - 2.0 * np.sqrt(1.0 - c * c)


def panel_geometry(ax):
    """A: the two-mode picture at the cat-kitten overlap."""
    c = C_CK
    th = math.acos(c)                      # angle between the two modes
    a = np.array([math.cos(-th / 2), math.sin(-th / 2)])
    b = np.array([math.cos(th / 2), math.sin(th / 2)])
    # Loewdin: symmetry forces the orthonormal pair symmetric about the axis
    la = np.array([math.cos(-math.pi / 4), math.sin(-math.pi / 4)])
    lb = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4)])
    # Gram-Schmidt: a fixed, b rotated to the orthogonal complement
    gb = (b - c * a) / math.sqrt(1.0 - c * c)

    for v, col, lab, off in ((a, INK, r"$\phi_a$", (0.06, -0.10)),
                             (b, INK, r"$\phi_b$", (0.06, 0.05))):
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.8))
        ax.text(v[0] + off[0], v[1] + off[1], lab, color=col, fontsize=12,
                fontweight="bold")
    for v, lab, off in ((la, r"$\tilde\phi_a$", (0.05, -0.12)),
                        (lb, r"$\tilde\phi_b$", (0.04, 0.07))):
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.4,
                                    linestyle="--"))
        ax.text(v[0] + off[0], v[1] + off[1], lab, color=BLUE, fontsize=12,
                fontweight="bold")
    ax.annotate("", xy=gb, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.4,
                                linestyle=":"))
    ax.text(gb[0] + 0.07, gb[1] - 0.02, "Gram–\nSchmidt", color=ORANGE,
            fontsize=10.5, fontweight="bold", ha="left", va="top")

    arc = np.linspace(-th / 2, th / 2, 60)
    ax.plot(0.33 * np.cos(arc), 0.33 * np.sin(arc), color=GREY, lw=1.4)
    ax.text(0.40, 0.0, rf"$\theta = {math.degrees(th):.1f}^\circ$", color=MUTED,
            fontsize=10, va="center")

    ax.text(0.475, -1.04, "Löwdin turns both modes outward equally;\n"
            "Gram–Schmidt pins one and moves it all onto the other",
            color=MUTED, fontsize=9.5, ha="center")
    ax.set_xlim(-0.35, 1.30)
    ax.set_ylim(-1.12, 1.15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(rf"A.  Two modes at overlap $c$ = {c}")


def panel_distortion(ax):
    """B: the minimal-distortion property, in closed form."""
    c = np.linspace(0.0, 0.995, 500)
    ax.plot(c, lowdin_distortion(c), color=BLUE, lw=2.4,
            label=r"Löwdin, $2(2-\sqrt{1+c}-\sqrt{1-c})$")
    ax.plot(c, gs_distortion(c), color=ORANGE, lw=2.2, ls="--",
            label=r"Gram–Schmidt, $2-2\sqrt{1-c^2}$")
    ax.fill_between(c, lowdin_distortion(c), gs_distortion(c), color=GREY,
                    alpha=0.16)
    for f, col in ((lowdin_distortion, BLUE), (gs_distortion, ORANGE)):
        ax.plot([C_CK], [f(C_CK)], marker="o", color=col, ms=7, zorder=5)
    ax.axvline(C_CK, color=GREY, lw=1.0, ls=":")
    ax.text(C_CK - 0.02, 0.12, "cat–kitten", color=MUTED, fontsize=9.5,
            rotation=90, ha="right")
    ax.text(0.06, 1.22, rf"at $c$ = {C_CK}:  Löwdin "
            rf"{lowdin_distortion(C_CK):.3f},  G–S {gs_distortion(C_CK):.3f}",
            color=INK, fontsize=10)
    ax.set_xlabel(r"overlap $c$")
    ax.set_ylabel(r"total distortion $\sum_v\|\tilde\phi_v-\phi_v\|^2$")
    ax.set_title("B.  Löwdin distorts least, at every overlap")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 2.0)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def panel_conditioning(ax):
    """C: what orthonormalising near-synonyms costs."""
    c = np.linspace(0.0, 0.995, 500)
    alpha = 0.5 * ((1 + c) ** -0.5 + (1 - c) ** -0.5)
    beta = 0.5 * ((1 - c) ** -0.5 - (1 + c) ** -0.5)
    cond = (1 + c) / (1 - c)
    ax.semilogy(c, alpha, color=BLUE, lw=2.3, label=r"$\alpha$")
    ax.semilogy(c, beta, color=RED, lw=2.3, ls="--", label=r"$|\beta|$")
    ax.semilogy(c, cond, color=PURPLE, lw=2.0, ls="-.",
                label=r"$\mathrm{cond}(G) = (1+c)/(1-c)$")
    ax.axvline(C_CK, color=GREY, lw=1.0, ls=":")
    a0 = 0.5 * ((1 + C_CK) ** -0.5 + (1 - C_CK) ** -0.5)
    k0 = (1 + C_CK) / (1 - C_CK)
    ax.plot([C_CK], [a0], marker="o", color=BLUE, ms=7, zorder=5)
    ax.plot([C_CK], [k0], marker="o", color=PURPLE, ms=7, zorder=5)
    ax.annotate(rf"$\alpha$ = {a0:.2f}", xy=(C_CK, a0), xytext=(0.46, 5.2),
                fontsize=10, color=BLUE,
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
    ax.annotate(rf"cond = {k0:.0f}", xy=(C_CK, k0), xytext=(0.40, 160),
                fontsize=10, color=PURPLE,
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.2))
    ax.set_xlabel(r"overlap $c$")
    ax.set_ylabel("magnitude")
    ax.set_title(r"C.  The price: coefficients diverge as $c\to 1$")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.3, 1e3)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.set_axisbelow(True)


def main():
    fig = plt.figure(figsize=(16.0, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.86, 1.07, 1.07], wspace=0.28)
    panel_geometry(fig.add_subplot(gs[0, 0]))
    panel_distortion(fig.add_subplot(gs[0, 1]))
    panel_conditioning(fig.add_subplot(gs[0, 2]))
    fig.suptitle("Löwdin orthonormalisation: the least-distorting way to make "
                 "overlapping modes orthonormal",
                 fontsize=13, fontweight="bold", y=1.03)
    save(fig, "sps_lowdin_construction.png")


if __name__ == "__main__":
    main()
    print(f"  cat-kitten c = {C_CK}")
    print(f"  Loewdin distortion   = {lowdin_distortion(C_CK):.4f}")
    print(f"  Gram-Schmidt         = {gs_distortion(C_CK):.4f}")
    a = 0.5 * ((1 + C_CK) ** -0.5 + (1 - C_CK) ** -0.5)
    b = 0.5 * ((1 - C_CK) ** -0.5 - (1 + C_CK) ** -0.5)
    print(f"  alpha = {a:.3f}  |beta| = {b:.3f}  cond(G) = "
          f"{(1 + C_CK) / (1 - C_CK):.1f}")
