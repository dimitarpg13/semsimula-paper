"""Figures for Semantic_Templates_as_Doi_Peliti_Reactions.md.

Four PNGs into figures/semantic_templates_dp/. Every panel is an exact
computation from the note's own formulas and the toy embeddings shared with
Single_Particle_Hilbert_Space_in_Semantic_Simulation.md
(cat = (0.90, 0.40), kitten = (0.85, 0.50), tax = (-0.30, 0.95), sigma = 0.25,
so kappa^2 = 2 and x* = 0.5). Nothing is fit to model data.

  std_matching_functional.png -- what the manuscript's Eqs. (17)-(20) compute:
                              the arc-length average of the template density
                              along a probe path, for a path through the core
                              and a path clipping the edge (S1.4).
  std_linear_vs_bilinear.png  -- the gap: probe-as-path (linear in f_p) against
                              probe-as-cloud (bilinear in the roots), and the
                              three candidate functionals of (3.2) against
                              centroid separation (S3.1-S3.3).
  std_threshold_radius.png    -- the payoff: the matching threshold is a binding
                              radius, d/x* = sqrt(2 ln(1/Theta)), with the toy
                              vocabulary's binding circles (S3.4).
  std_control_hopping.png     -- control flow as a hopping term: the executive
                              particle is a Markov chain on its own atom tree,
                              with the arc significance vectors as rates (S7.2).

Run:  python3 _make_semantic_templates_dp_figs.py
"""
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "figure.dpi": 150,
})
GREEN, BLUE, RED = "#22C55E", "#3B82F6", "#EF4444"
PURPLE, GREY, ORANGE = "#8B5CF6", "#9CA3AF", "#F59E0B"
INK, MUTED = "#1F2937", "#6B7280"
OUT = Path(__file__).parent / "semantic_templates_dp"
OUT.mkdir(exist_ok=True)

EMB = {"cat": np.array([0.90, 0.40]), "kitten": np.array([0.85, 0.50]),
       "tax": np.array([-0.30, 0.95])}
COL = {"cat": BLUE, "kitten": GREEN, "tax": RED}
SIGMA = 0.25
KAPPA2 = 1.0 / (8 * SIGMA ** 2)          # = 2
XSTAR = 1.0 / math.sqrt(2 * KAPPA2)      # = 0.5 = 2 sigma


def density(X, mu, s=SIGMA):
    """p(x) = (2 pi s^2)^(-L/2) exp(-|x - mu|^2 / (2 s^2)), L = 2 here."""
    r2 = ((X - mu) ** 2).sum(-1)
    return (2 * np.pi * s * s) ** (-1.0) * np.exp(-r2 / (2 * s * s))


def bc(d, s=SIGMA):
    """Bhattacharyya coefficient of two equal-width Gaussians, Eq. (3.3)."""
    return np.exp(-d ** 2 / (8 * s * s))


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def _grid(xlim, ylim, n=320):
    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    XX, YY = np.meshgrid(xs, ys)
    return xs, ys, np.stack([XX, YY], axis=-1)


# ---------------------------------------------------------------------------
def fig_matching_functional():
    """S1.4: the arc-length average the manuscript's (17)-(20) actually compute."""
    mu = np.array([0.0, 0.0])
    paths = [("through the core", 0.0, BLUE, "-"),
             ("clipping the edge", 0.35, ORANGE, "--")]
    x0, x1 = -0.8, 0.8

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.3))

    xs, ys, P = _grid((-0.9, 0.9), (-0.6, 0.75))
    Z = density(P, mu)
    axL.contourf(xs, ys, Z, levels=12, cmap="Blues", alpha=0.55)
    axL.contour(xs, ys, Z, levels=6, colors=GREY, linewidths=0.6, alpha=0.8)
    axL.plot(*mu, marker="+", color=INK, ms=11, mew=2)
    axL.annotate(r"template cloud $f_p$", xy=(0.02, -0.12), color=INK,
                 fontsize=10, ha="left")

    for label, b, col, ls in paths:
        axL.plot([x0, x1], [b, b], color=col, lw=2.2, ls=ls)
        axL.annotate("", xy=(x1, b), xytext=(x1 - 0.22, b),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
        axL.text(x0 + 0.02, b + 0.055, label, color=col, fontsize=10,
                 fontweight="bold")
    axL.set_xlim(-0.9, 0.9)
    axL.set_ylim(-0.6, 0.75)
    axL.set_xlabel(r"$\Sigma$, first coordinate")
    axL.set_ylabel(r"$\Sigma$, second coordinate")
    axL.set_title("A probe crosses the binding region")
    axL.set_aspect("equal")

    s = np.linspace(0.0, x1 - x0, 600)
    for label, b, col, ls in paths:
        pts = np.stack([x0 + s, np.full_like(s, b)], axis=-1)
        w = density(pts, mu)
        wbar = np.trapezoid(w, s) / (x1 - x0)
        axR.plot(s, w, color=col, lw=2.0, ls=ls)
        axR.axhline(wbar, color=col, lw=1.2, ls=":", alpha=0.9)
        axR.text(s[-1], wbar + 0.06, rf"$\bar w$ = {wbar:.2f}", color=col,
                 fontsize=10, ha="right", fontweight="bold")
        axR.text(0.02, max(w) + 0.06, label, color=col, fontsize=10,
                 fontweight="bold")

    axR.axhline(0.6, color=GREY, lw=1.4, ls="-.")
    axR.text(0.02, 0.63, r"threshold $\Theta$", color=MUTED, fontsize=10)
    axR.set_xlabel(r"arc length $\ell$ along the path")
    axR.set_ylabel(r"$f_p$ along the path   [$(\mathrm{length})^{-2}$]")
    axR.set_title(r"$\bar w = \int f_p\,d\ell\ /\int d\ell$")
    axR.set_xlim(0, x1 - x0)
    axR.set_ylim(0, 2.9)
    axR.grid(alpha=0.25, lw=0.6)
    axR.set_axisbelow(True)

    fig.suptitle("Matching as written: an arc-length average of the template density",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.text(0.5, -0.06, "The score carries the units of a density, not of a pure "
             "number: both integrals contribute one factor of length (note §1.4).",
             ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    save(fig, "std_matching_functional.png")


# ---------------------------------------------------------------------------
def fig_linear_vs_bilinear():
    """S3: probe as a path (linear in f_p) against probe as a cloud (bilinear)."""
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4))
    (axA, axB), (axC, axD) = axes
    mu_p = np.array([0.0, 0.0])
    mu_q = np.array([0.42, 0.0])

    xs, ys, P = _grid((-0.75, 1.15), (-0.7, 0.7))
    Zp = density(P, mu_p)

    for ax, title in ((axA, "Probe as a path"), (axB, "Probe as a cloud")):
        ax.contourf(xs, ys, Zp, levels=12, cmap="Blues", alpha=0.5)
        ax.contour(xs, ys, Zp, levels=5, colors=GREY, linewidths=0.6, alpha=0.8)
        ax.plot(*mu_p, marker="+", color=INK, ms=11, mew=2)
        ax.text(-0.70, 0.56, r"$f_p$", color=BLUE, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.75, 1.15)
        ax.set_ylim(-0.7, 0.7)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    axA.plot([-0.6, 1.0], [0.0, 0.0], color=ORANGE, lw=2.4)
    axA.annotate("", xy=(1.0, 0.0), xytext=(0.78, 0.0),
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.4))
    axA.text(0.32, 0.10, "trajectory", color=ORANGE, fontsize=10, fontweight="bold")
    axA.text(-0.70, -0.62, r"linear in $f_p$, one number per path",
             color=MUTED, fontsize=9.5)

    Zq = density(P, mu_q)
    axB.contour(xs, ys, Zq, levels=5, colors=PURPLE, linewidths=1.3)
    axB.plot(*mu_q, marker="+", color=PURPLE, ms=11, mew=2)
    axB.text(0.82, 0.40, r"$f_q$", color=PURPLE, fontsize=12, fontweight="bold")
    axB.text(-0.70, -0.62, r"bilinear in the roots, one number per pair",
             color=MUTED, fontsize=9.5)

    d = np.linspace(0.0, 3.0, 400) * XSTAR          # separation in units of x*
    dx = d / XSTAR
    axC.plot(dx, bc(d), color=BLUE, lw=2.2)
    axC.plot(dx, np.exp(-d ** 2 / (4 * SIGMA ** 2)), color=PURPLE, lw=2.0, ls="--")
    axC.text(1.52, 0.56, r"$\int\sqrt{f_pf_q}$  (Bhattacharyya)", color=BLUE,
             fontsize=10, fontweight="bold")
    axC.text(0.30, 0.11, r"$\int f_pf_q$, normalised", color=PURPLE,
             fontsize=10, fontweight="bold")
    axC.set_xlabel(r"centroid separation  $d/x^{*}$")
    axC.set_ylabel("score")
    axC.set_title("Bounded, symmetric, in [0, 1]")
    axC.set_ylim(-0.03, 1.05)
    axC.set_xlim(0, 3)
    axC.grid(alpha=0.25, lw=0.6)
    axC.set_axisbelow(True)

    axD.plot(dx, d ** 2 / (2 * SIGMA ** 2), color=RED, lw=2.2)
    axD.text(1.05, 6.4, r"$\int f_q\log(f_q/f_p)$", color=RED, fontsize=10,
             fontweight="bold")
    axD.text(1.05, 4.6, "unbounded, asymmetric", color=MUTED, fontsize=9.5)
    axD.set_xlabel(r"centroid separation  $d/x^{*}$")
    axD.set_ylabel("divergence  (nats)")
    axD.set_title("Kullback–Leibler, for contrast")
    axD.set_xlim(0, 3)
    axD.grid(alpha=0.25, lw=0.6)
    axD.set_axisbelow(True)

    fig.suptitle("The bilinearity gap, and why the root is the right choice",
                 fontsize=12.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "std_linear_vs_bilinear.png")


# ---------------------------------------------------------------------------
def fig_threshold_radius():
    """S3.4: the matching threshold is a binding radius in inflection radii."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.6))

    th = np.linspace(0.01, 0.99, 500)
    axL.plot(th, np.sqrt(2 * np.log(1 / th)), color=BLUE, lw=2.4)
    marks = [(0.9, "near-synonyms only", "right"), (0.5, "about one $x^{*}$", "right"),
             (0.1, "the whole basin", "left"), (0.031, "cat–tax overlap", "left")]
    for t, lab, side in marks:
        r = math.sqrt(2 * math.log(1 / t))
        axL.plot([t], [r], marker="o", color=RED, ms=7, zorder=5)
        dx = 0.05 if side == "left" else -0.05
        axL.annotate(rf"$\Theta$={t:g} $\to$ {r:.2f}   {lab}",
                     xy=(t, r), xytext=(t + dx, r + 0.13), fontsize=9.5,
                     color=INK, ha=side)
    axL.set_xlabel(r"matching threshold  $\Theta$")
    axL.set_ylabel(r"binding radius  $d/x^{*}$")
    axL.set_title(r"$d/x^{*} = \sqrt{2\ln(1/\Theta)}$")
    axL.set_xlim(0, 1.0)
    axL.set_ylim(0, 3.2)
    axL.grid(alpha=0.25, lw=0.6)
    axL.set_axisbelow(True)

    # each circle labelled on its own arc, at a distinct angle, so the three
    # labels never collide with each other or with the three types
    for t, col, ang in ((0.9, PURPLE, -62), (0.5, ORANGE, -35), (0.031, GREY, -18)):
        r = math.sqrt(2 * math.log(1 / t)) * XSTAR
        axR.add_patch(Circle(EMB["cat"], r, fill=False, ec=col, lw=1.8,
                             ls="--", zorder=2))
        a = math.radians(ang)
        axR.annotate(rf"$\Theta$={t:g}",
                     xy=(EMB["cat"][0] + (r + 0.07) * math.cos(a),
                         EMB["cat"][1] + (r + 0.07) * math.sin(a)),
                     color=col, fontsize=10, fontweight="bold", zorder=4,
                     ha="left", va="center")
    offsets = {"cat": (-0.10, -0.12), "kitten": (0.10, 0.10), "tax": (-0.10, 0.12)}
    for name, p in EMB.items():
        ox, oy = offsets[name]
        axR.plot(*p, marker="o", color=COL[name], ms=10, zorder=5)
        axR.annotate(name, xy=p, xytext=(p[0] + ox, p[1] + oy),
                     color=COL[name], fontsize=11, fontweight="bold", zorder=5,
                     ha="right" if ox < 0 else "left")
    axR.set_xlim(-0.85, 2.95)
    axR.set_ylim(-1.15, 2.05)
    axR.set_aspect("equal")
    axR.set_xlabel(r"$\Sigma$, first coordinate")
    axR.set_ylabel(r"$\Sigma$, second coordinate")
    axR.set_title("Binding circles around cat")
    axR.grid(alpha=0.22, lw=0.6)
    axR.set_axisbelow(True)

    d_ct = float(np.linalg.norm(EMB["cat"] - EMB["tax"]))
    fig.text(0.5, -0.04, "The toy vocabulary is a consistency check: cat–tax sits at "
             rf"$d/x^{{*}}$ = {d_ct / XSTAR:.2f}, and the $\Theta$ = 0.031 circle "
             "passes through tax, because 0.031 is exactly their overlap.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    save(fig, "std_threshold_radius.png")


# ---------------------------------------------------------------------------
def fig_control_hopping():
    """S7.2: delta as a hopping term for a control particle on the atom tree."""
    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 4.9)
    ax.axis("off")

    def atom(x, y, label, occupied=False, w=1.5, h=0.74):
        ec = BLUE if occupied else GREY
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.06",
                                    fc="#EFF6FF" if occupied else "#F9FAFB",
                                    ec=ec, lw=2.0 if occupied else 1.3,
                                    zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=11,
                color=INK, zorder=4,
                fontweight="bold" if occupied else "normal")

    atom(2.0, 3.8, r"$\alpha$", occupied=True)
    ax.plot([2.0], [3.8 + 0.52], marker="o", color=BLUE, ms=12, zorder=6)
    ax.annotate("control particle", xy=(2.0, 4.32), xytext=(0.15, 4.52),
                fontsize=10, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.2))

    children = [(5.6, 4.1, r"$\alpha_1$", r"$r_1$", PURPLE),
                (5.6, 2.7, r"$\alpha_2$", r"$r_2$", ORANGE),
                (5.6, 1.3, r"$\alpha_3$", r"$r_3$", GREY)]
    tot = 3
    for (cx, cy, lab, rlab, col) in children:
        atom(cx, cy, lab)
        ax.annotate("", xy=(cx - 0.80, cy), xytext=(2.78, 3.8),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0,
                                    connectionstyle="arc3,rad=0.12"))
        mx, my = (2.78 + cx - 0.80) / 2, (3.8 + cy) / 2
        ax.text(mx, my + 0.16, rlab, color=col, fontsize=11, fontweight="bold",
                ha="center")

    ax.text(7.05, 4.1, r"$\mathfrak{S}$: products into $\Sigma$", fontsize=10,
            color=MUTED, va="center")
    ax.text(7.05, 2.7, r"$\mathfrak{E}$: products onward in $\mathbf{E}$",
            fontsize=10, color=MUTED, va="center")
    ax.text(7.05, 1.3, r"$\mu$ rejects: no firing", fontsize=10, color=MUTED,
            va="center")

    ax.text(0.15, 0.72,
            r"$\mathcal{L}_\delta=\sum_n r_n\,"
            r"(\tilde a^{\dagger}_{\alpha_n}-\tilde a^{\dagger}_{\alpha})\,"
            r"\tilde a_{\alpha}$",
            fontsize=13, color=INK)
    ax.text(0.15, 0.20,
            rf"child $n$ receives control with probability "
            rf"$r_n/\sum_m r_m$;  a deterministic $\delta$ is the "
            rf"zero-temperature limit", fontsize=10, color=MUTED)

    ax.set_title("Control flow is a hopping term: the executive particle is a "
                 "Markov chain on its own tree", fontsize=12.5,
                 fontweight="bold", loc="left")
    fig.tight_layout()
    save(fig, "std_control_hopping.png")


if __name__ == "__main__":
    fig_matching_functional()
    fig_linear_vs_bilinear()
    fig_threshold_radius()
    fig_control_hopping()
    print(f"\nsigma={SIGMA}  kappa^2={KAPPA2}  x*={XSTAR}")
