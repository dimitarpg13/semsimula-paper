"""Figures for Hamiltonian_Flow_in_the_Semantic_Simulation_Framework.md.

Three PNGs into figures/hamiltonian_flow/. Every panel is an exact
computation; the only schematic element is the labelling in hf_constraint.png.

  hf_gradient_vs_flow.png -- the role of the symplectic matrix J: the same
        Hamiltonian generates descent under -grad H and circulation under
        J grad H (S2.1).
  hf_properties.png       -- the two structural properties and what damping
        costs: phase-space area under the Hamiltonian flow against the damped
        flow, and the shadow-Hamiltonian behaviour of a symplectic integrator
        against explicit Euler (S3.2, S3.3, S5.2).
  hf_constraint.png       -- the LayerNorm surface as a holonomic-shaped
        manifold: at d = 3 it is a circle, the step leaves it, the projection
        returns the position but not the velocity, and the retraction does
        work on the potential (S4.4).

Run:  python3 _make_hamiltonian_flow_figs.py
"""
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "figure.dpi": 150,
})
GREEN, BLUE, RED = "#22C55E", "#3B82F6", "#EF4444"
PURPLE, GREY, ORANGE = "#8B5CF6", "#9CA3AF", "#F59E0B"
INK, MUTED = "#1F2937", "#6B7280"
OUT = Path(__file__).parent / "hamiltonian_flow"
OUT.mkdir(exist_ok=True)

KAPPA2 = 2.0          # the framework's toy value


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# --- the 1-DOF toy: one Gaussian well, as in the framework -----------------
def V(h):
    return -np.exp(-KAPPA2 * h * h)


def dV(h):
    return 2.0 * KAPPA2 * h * np.exp(-KAPPA2 * h * h)


def H(h, p):
    return 0.5 * p * p + V(h)


# ---------------------------------------------------------------------------
def fig_gradient_vs_flow():
    """S2.1: -grad H descends; J grad H circulates. Same H."""
    hs = np.linspace(-1.6, 1.6, 420)
    ps = np.linspace(-1.5, 1.5, 420)
    HH, PP = np.meshgrid(hs, ps)
    E = H(HH, PP)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.7))
    for ax, title in ((axL, r"Gradient descent:  $\dot\Gamma = -\nabla H$"),
                      (axR, r"Hamiltonian flow:  $\dot\Gamma = J\nabla H$")):
        ax.contour(hs, ps, E, levels=16, colors=GREY, linewidths=0.7,
                   alpha=0.85)
        ax.set_xlabel(r"position $h$")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(title)
    axL.set_ylabel(r"momentum $p$")

    hq = np.linspace(-1.5, 1.5, 19)
    pq = np.linspace(-1.4, 1.4, 17)
    HQ, PQ = np.meshgrid(hq, pq)
    # -grad H = (-dV/dh, -p)
    axL.streamplot(hq, pq, -dV(HQ), -PQ, color=RED, density=0.9,
                   linewidth=1.0, arrowsize=1.0)
    # J grad H = (dH/dp, -dH/dh) = (p, -dV/dh)
    axR.streamplot(hq, pq, PQ, -dV(HQ), color=BLUE, density=0.9,
                   linewidth=1.0, arrowsize=1.0)

    axL.plot([0], [0], marker="o", color=INK, ms=8, zorder=5)
    axL.annotate("every path ends at the\nminimum: energy is lost",
                 xy=(0.0, 0.0), xytext=(-1.5, 1.05), fontsize=10, color=INK,
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    axR.annotate("paths follow level sets\nof $H$: energy is conserved",
                 xy=(0.62, 0.52), xytext=(-1.52, 1.05), fontsize=10, color=INK,
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))

    fig.suptitle(r"One Hamiltonian, two vector fields: $J$ rotates the gradient"
                 r" by a quarter turn", fontsize=12.5, fontweight="bold",
                 y=1.01)
    fig.text(0.5, -0.045, "Grey contours are level sets of "
             r"$H = p^2/2 + V(h)$ with the framework's Gaussian well "
             r"$V(h) = -e^{-\kappa^2h^2}$, $\kappa^2 = 2$. Both panels show the "
             "same $H$.", ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    save(fig, "hf_gradient_vs_flow.png")


# ---------------------------------------------------------------------------
def _shoelace(x, y):
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _verlet(h, p, dt, gamma=0.0, n=1):
    """Damped velocity-Verlet, the framework's integrator, vectorised."""
    for _ in range(n):
        p = p - 0.5 * dt * dV(h)
        h = h + dt * p
        p = p - 0.5 * dt * dV(h)
        if gamma:
            p = p * math.exp(-gamma * dt)
    return h, p


def panel_volume(ax):
    """S3.3: Liouville preserves area; damping contracts it by exp(-gamma t)."""
    th = np.linspace(0, 2 * np.pi, 400)
    h0, p0 = 0.45 + 0.17 * np.cos(th), 0.17 * np.sin(th)
    a0 = _shoelace(h0, p0)
    dt, T = 0.01, 3.0
    n = int(T / dt)

    ax.plot(h0, p0, color=GREY, lw=2.0, ls=":")
    ax.fill(h0, p0, color=GREY, alpha=0.18)
    ax.text(0.45, 0.0, "t = 0", ha="center", va="center", color=INK,
            fontsize=9.5)

    hH, pH = _verlet(h0.copy(), p0.copy(), dt, 0.0, n)
    hD, pD = _verlet(h0.copy(), p0.copy(), dt, 0.35, n)
    aH, aD = _shoelace(hH, pH), _shoelace(hD, pD)

    ax.plot(hH, pH, color=BLUE, lw=2.2)
    ax.fill(hH, pH, color=BLUE, alpha=0.16)
    ax.plot(hD, pD, color=RED, lw=2.2)
    ax.fill(hD, pD, color=RED, alpha=0.16)

    ax.text(0.02, 0.24, rf"area ratio, $\gamma = 0$:  {aH / a0:.4f}",
            transform=ax.transAxes, color=BLUE, fontsize=10,
            fontweight="bold", va="top")
    ax.text(0.02, 0.15,
            rf"area ratio, $\gamma = 0.35$:  {aD / a0:.4f}"
            "\n" rf"predicted $e^{{-\gamma T}} = {math.exp(-0.35 * T):.4f}$",
            transform=ax.transAxes, color=RED, fontsize=10,
            fontweight="bold", va="top")
    ax.set_xlabel(r"position $h$")
    ax.set_ylabel(r"momentum $p$")
    ax.set_title("A.  Phase-space area after  t = 3")
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def panel_shadow(ax):
    """S5.2: a symplectic integrator has no secular energy drift.

    Harmonic well, which is exactly the stiff part the CfC substep
    propagates in closed form.
    """
    dt, n, omega = 0.05, 4000, 1.0
    h_v, p_v = 0.8, 0.0
    h_e, p_e = 0.8, 0.0
    E0 = 0.5 * p_v ** 2 + 0.5 * omega ** 2 * h_v ** 2
    E_v, E_e, ts = [], [], []
    for k in range(n):
        p_v = p_v - 0.5 * dt * omega ** 2 * h_v          # velocity-Verlet
        h_v = h_v + dt * p_v
        p_v = p_v - 0.5 * dt * omega ** 2 * h_v
        h_new = h_e + dt * p_e                            # explicit Euler
        p_e = p_e - dt * omega ** 2 * h_e
        h_e = h_new
        if k % 8 == 0:
            ts.append(k * dt)
            E_v.append((0.5 * p_v ** 2 + 0.5 * omega ** 2 * h_v ** 2) / E0)
            E_e.append((0.5 * p_e ** 2 + 0.5 * omega ** 2 * h_e ** 2) / E0)
    ax.semilogy(ts, E_v, color=BLUE, lw=2.0,
                label="velocity-Verlet (symplectic)")
    ax.semilogy(ts, E_e, color=RED, lw=2.0, ls="--", label="explicit Euler")
    ax.axhline(1.0, color=GREY, lw=1.2, ls=":")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$H/H_0$")
    ax.set_title("B.  Energy over 4000 steps, harmonic well")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.set_axisbelow(True)
    ax.text(0.97, 0.30, "Verlet stays in a band:\nit exactly conserves a\n"
            "nearby shadow Hamiltonian", transform=ax.transAxes, color=BLUE,
            fontsize=9.5, ha="right")


def fig_properties():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    panel_volume(axA)
    panel_shadow(axB)
    fig.suptitle("What the Hamiltonian structure guarantees, and what damping "
                 "costs", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "hf_properties.png")


# ---------------------------------------------------------------------------
def _project(x):
    """F.layer_norm with no affine: zero mean, unit variance, so norm = sqrt(d)."""
    x = x - x.mean(-1, keepdims=True)
    return x / np.sqrt((x * x).mean(-1, keepdims=True))


def panel_circle(ax):
    """S4.4: the position is projected back; the velocity is not."""
    R = 1.0
    t = np.linspace(-0.95, 1.25, 400)
    ax.plot(R * np.cos(t), R * np.sin(t), color=BLUE, lw=2.6, zorder=3)
    ax.text(0.66, -0.30, r"the surface $\mathcal{S}$", color=BLUE,
            fontsize=11, fontweight="bold")

    h = np.array([1.0, 0.0])
    tang = np.array([0.0, 1.0])          # tangent at h
    norm = np.array([1.0, 0.0])          # outward normal at h
    v = 0.52 * tang + 0.30 * norm        # the integrator's velocity

    ax.plot([h[0] - 0.02 * tang[0], h[0] + 0.72 * tang[0]],
            [h[1] - 0.30, h[1] + 0.72], color=GREY, lw=1.4, ls="--", zorder=2)
    ax.text(1.03, -0.36, "tangent space at $h$", color=MUTED, fontsize=9.5)

    off = h + v
    pr = R * off / np.linalg.norm(off)

    ax.annotate("", xy=off, xytext=h,
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.6))
    ax.text(1.30, 0.40, "free step", color=ORANGE, fontsize=10.5,
            fontweight="bold")
    ax.annotate("", xy=pr, xytext=off,
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.4,
                                linestyle="--"))
    ax.text(1.06, 0.60, "projection\nreturns $h$", color=GREEN, fontsize=10.5,
            fontweight="bold")

    # the velocity decomposition at h
    ax.annotate("", xy=h + 0.52 * tang, xytext=h,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.0))
    ax.annotate("", xy=h + 0.30 * norm, xytext=h,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.4))
    ax.text(0.84, 0.30, "tangential\npart", color=INK, fontsize=10, ha="right")
    ax.text(1.17, 0.015, "normal part of $v$", color=RED, fontsize=10.5,
            fontweight="bold")
    ax.text(1.17, -0.085, "never removed", color=RED, fontsize=10.5,
            fontweight="bold")

    ax.scatter(*h, color=INK, s=58, zorder=6)
    ax.text(0.90, -0.10, "$h$", color=INK, fontsize=13, fontweight="bold",
            ha="right")
    ax.scatter(*off, color=ORANGE, s=48, zorder=6)
    ax.scatter(*pr, color=GREEN, s=48, zorder=6)

    ax.set_xlim(0.62, 1.95)
    ax.set_ylim(-0.46, 0.95)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("A.  Position is projected, velocity is not")


def panel_work(ax):
    """S4.4: the retraction is not workless -- it changes V."""
    rng = np.random.default_rng(0)
    d, n = 384, 4000
    kappa2 = 1.0 / (2.0 * d)
    mu = _project(rng.normal(size=(1, d)))

    def Vd(x):
        r2 = ((x - mu) ** 2).sum(-1)
        return -np.exp(-kappa2 * r2)

    sizes = np.linspace(0.2, 6.0, 22)
    med, p90 = [], []
    h0 = _project(rng.normal(size=(n, d)))
    for s in sizes:
        step = rng.normal(size=(n, d))
        step /= np.sqrt((step ** 2).sum(-1, keepdims=True))
        off = h0 + s * step
        dv = np.abs(Vd(_project(off)) - Vd(off))
        med.append(np.median(dv))
        p90.append(np.quantile(dv, 0.9))
    ax.plot(sizes, med, color=PURPLE, lw=2.3, label="median")
    ax.plot(sizes, p90, color=PURPLE, lw=1.6, ls="--", label="90th percentile")
    ax.fill_between(sizes, med, p90, color=PURPLE, alpha=0.14)
    ax.set_xlabel("step length before projection")
    ax.set_ylabel(r"$|\Delta V|$ caused by the projection")
    ax.set_title(r"B.  The retraction does work  ($d = 384$)")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(0.97, 0.06, "a constraint force would\nsit flat at zero",
            transform=ax.transAxes, color=MUTED, fontsize=9.5, ha="right")


def fig_constraint():
    fig = plt.figure(figsize=(11.6, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.22)
    panel_circle(fig.add_subplot(gs[0, 0]))
    panel_work(fig.add_subplot(gs[0, 1]))
    fig.suptitle("The LayerNorm surface is holonomic-shaped, but the "
                 "projection is not a constraint force",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "hf_constraint.png")


if __name__ == "__main__":
    fig_gradient_vs_flow()
    fig_properties()
    fig_constraint()
