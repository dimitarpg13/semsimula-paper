"""Figures for the FDT appendix of
Langevin_dynamics_reformulation_of_classical_damped_Lagrangian_flow.md.

One PNG into figures/fdt_ostep/. Every panel is an exact computation of the
O-step actually implemented in ``cfc_baoab.ou_step``; nothing is schematic
except the middle panel, which is a labelled diagram of the substep order in
``model_parf_multixi._layer_step_langevin``.

  fdt_the_lock.png -- why the noise amplitude is not free:
      A  the velocity variance under three noise amplitudes, only the
         FDT-locked one is stationary at the equipartition value T/m;
      B  where temperature enters the A-B-O-A step: the O substep alone;
      C  what the lock buys: the sampled configuration matches the Boltzmann
         weight of the framework's own two-Gaussian-well potential, which is
         what the readout softmax consumes.

Run:  python3 _make_fdt_ostep_figs.py
"""
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "figure.dpi": 150,
})
GREEN, BLUE, RED = "#22C55E", "#3B82F6", "#EF4444"
PURPLE, GREY, ORANGE = "#8B5CF6", "#9CA3AF", "#F59E0B"
INK, MUTED = "#1F2937", "#6B7280"
OUT = Path(__file__).parent / "fdt_ostep"
OUT.mkdir(exist_ok=True)

# The live SemSimula setting: gamma = 0.1 per layer at dt = 1.
GAMMA, DT, MASS, TEMP = 0.1, 1.0, 1.0, 1.0
C1 = math.exp(-GAMMA * DT)                      # exact friction factor
FDT_STD = math.sqrt((TEMP / MASS) * (1.0 - C1 * C1))   # ou_step, verbatim


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------
def panel_lock(ax):
    """A: only the FDT amplitude leaves the equipartition variance invariant."""
    n_steps, n_walk = 120, 40_000
    rng = np.random.default_rng(0)
    cases = [(0.5, RED, "-.", r"noise $0.5\times$ FDT  (cools)"),
             (1.0, BLUE, "-", r"noise $=$ FDT  (stationary)"),
             (2.0, ORANGE, "--", r"noise $2\times$ FDT  (heats)")]

    for scale, col, ls, lab in cases:
        std = scale * FDT_STD
        # start AT equipartition, so the FDT case has nothing to relax to
        v = rng.normal(0.0, math.sqrt(TEMP / MASS), size=n_walk)
        var_sim = [v.var()]
        for _ in range(n_steps):
            v = C1 * v + std * rng.normal(size=n_walk)
            var_sim.append(v.var())
        # exact recursion Var_{n+1} = c1^2 Var_n + std^2
        var_ex, x = [TEMP / MASS], [TEMP / MASS]
        for _ in range(n_steps):
            x[0] = C1 * C1 * x[0] + std * std
            var_ex.append(x[0])
        ax.plot(var_ex, color=col, lw=2.3, ls=ls, zorder=3)
        ax.plot(np.arange(0, n_steps + 1, 6), np.array(var_sim)[::6],
                ls="none", marker="o", ms=3.4, color=col, alpha=0.55, zorder=2)
        ax.text(n_steps * 0.99, var_ex[-1], f"  {scale:g}×", color=col,
                fontsize=10, fontweight="bold", va="center")

    ax.axhline(TEMP / MASS, color=GREY, lw=1.4, ls=":")
    ax.text(3, TEMP / MASS + 0.12, r"equipartition  $\langle v^2\rangle = T/m$",
            color=MUTED, fontsize=10)
    ax.set_xlabel("O-step applications")
    ax.set_ylabel(r"velocity variance  $\langle v^2\rangle$")
    ax.set_title("A.  The amplitude is not free")
    ax.set_xlim(0, n_steps * 1.09)
    ax.set_ylim(0, 4.4)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(44, 2.45, "lines: exact recursion\ndots: simulated ensemble",
            color=MUTED, fontsize=9)


# ---------------------------------------------------------------------------
def panel_where(ax):
    """B: the substep order actually implemented, with O as sole T carrier."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    boxes = [(1.15, "A", r"half drift", GREY, "#F9FAFB"),
             (3.45, "B", r"full kick", GREY, "#F9FAFB"),
             (5.75, "O", r"friction $+$ noise", BLUE, "#EFF6FF"),
             (8.05, "A", r"half drift", GREY, "#F9FAFB")]
    for x, tag, sub, ec, fc in boxes:
        hot = tag == "O"
        ax.add_patch(FancyBboxPatch((x - 0.92, 3.5), 1.84, 1.5,
                                    boxstyle="round,pad=0.06", fc=fc, ec=ec,
                                    lw=2.4 if hot else 1.3, zorder=3))
        ax.text(x, 4.52, tag, ha="center", fontsize=15, color=INK,
                fontweight="bold", zorder=4)
        ax.text(x, 3.92, sub, ha="center", fontsize=8.0,
                color=BLUE if hot else MUTED, zorder=4)
    for x0, x1 in ((2.07, 2.53), (4.37, 4.83), (6.67, 7.13)):
        ax.annotate("", xy=(x1, 4.25), xytext=(x0, 4.25),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8))

    ax.annotate("", xy=(5.75, 3.34), xytext=(5.75, 2.52),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.0))
    ax.text(5.75, 2.28, "the only substep that contains $T$", ha="center",
            color=BLUE, fontsize=10.5, fontweight="bold")
    ax.text(5.75, 1.80,
            r"$v \leftarrow c_1 v + \sqrt{(T/m)(1-c_1^2)}\,\xi,"
            r"\qquad c_1 = e^{-\gamma\Delta t}$",
            ha="center", color=INK, fontsize=11.5)

    ax.text(0.15, 5.62, "A and B carry conservative mechanics only",
            color=MUTED, fontsize=10)
    rows = [(r"$T>0$", "Langevin, samples Gibbs", BLUE),
            (r"$T=0$", "deterministic damped flow  (the live runs)", INK),
            (r"$T=0,\ \gamma=0$", "symplectic Verlet", MUTED)]
    for i, (lhs, rhs, col) in enumerate(rows):
        ax.text(1.55, 1.08 - 0.46 * i, lhs, color=col, fontsize=10.5,
                fontweight="bold", ha="right")
        ax.text(1.75, 1.08 - 0.46 * i, rhs, color=col, fontsize=10.5)
    ax.set_title("B.  Where temperature enters the step")


# ---------------------------------------------------------------------------
def _V(x, kappa2=2.0, a=1.0, mu=1.0):
    """Two of the framework's Gaussian wells, as in PM1 eq. (1.3)."""
    return -a * (np.exp(-kappa2 * (x - mu) ** 2)
                 + np.exp(-kappa2 * (x + mu) ** 2))


def _dV(x, kappa2=2.0, a=1.0, mu=1.0):
    return -a * (-2 * kappa2 * (x - mu) * np.exp(-kappa2 * (x - mu) ** 2)
                 - 2 * kappa2 * (x + mu) * np.exp(-kappa2 * (x + mu) ** 2))


def panel_gibbs(ax):
    """C: with the lock, the sampled configuration IS the Boltzmann weight."""
    T, gamma, dt, m = 0.25, 0.5, 0.02, 1.0
    c1 = math.exp(-gamma * dt)
    std = math.sqrt((T / m) * (1.0 - c1 * c1))
    rng = np.random.default_rng(1)
    n_walk, n_steps, burn = 12_000, 16_000, 5_000

    x = rng.normal(0.0, 0.5, size=n_walk)
    v = rng.normal(0.0, math.sqrt(T / m), size=n_walk)
    keep = []
    for n in range(n_steps):
        x = x + 0.5 * dt * v                      # A
        v = v - (dt / m) * _dV(x)                 # B
        v = c1 * v + std * rng.normal(size=n_walk)  # O
        x = x + 0.5 * dt * v                      # A
        if n >= burn and n % 20 == 0:
            keep.append(x.copy())
    samples = np.concatenate(keep)

    grid = np.linspace(-2.2, 2.2, 600)
    boltz = np.exp(-_V(grid) / T)
    boltz /= np.trapezoid(boltz, grid)

    ax.hist(samples, bins=140, range=(-2.2, 2.2), density=True,
            color=BLUE, alpha=0.38, label=r"sampled, FDT-locked $T$")
    ax.plot(grid, boltz, color=INK, lw=2.2, ls="--",
            label=r"$e^{-V(x)/T}/Z$")

    # T = 0: the same step with the noise switched off collapses to a minimum
    x0 = rng.normal(0.0, 0.5, size=2000)
    v0 = np.zeros_like(x0)
    for _ in range(n_steps):
        x0 = x0 + 0.5 * dt * v0
        v0 = v0 - (dt / m) * _dV(x0)
        v0 = c1 * v0
        x0 = x0 + 0.5 * dt * v0
    for xm, lab in ((np.median(x0[x0 > 0]), True), (np.median(x0[x0 < 0]), False)):
        ax.axvline(xm, color=RED, lw=2.0, ls="-")
    ax.text(0.5, 0.09, r"$T=0$: collapse onto the minima",
            transform=ax.transAxes, color=RED, fontsize=10,
            fontweight="bold", ha="center")

    ax.set_xlabel("configuration coordinate")
    ax.set_ylabel("density")
    ax.set_title("C.  What the lock buys")
    ax.set_xlim(-2.2, 2.2)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
def main():
    fig = plt.figure(figsize=(16.2, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.0], wspace=0.26)
    panel_lock(fig.add_subplot(gs[0, 0]))
    panel_where(fig.add_subplot(gs[0, 1]))
    panel_gibbs(fig.add_subplot(gs[0, 2]))
    fig.suptitle("Friction and noise are one coupling: the fluctuation"
                 "–dissipation lock in the O-step",
                 fontsize=13.5, fontweight="bold", y=1.03)
    fig.text(0.5, -0.07,
             rf"Panel A at the live setting $\gamma={GAMMA}$, $\Delta t={DT}$, "
             rf"so $c_1=e^{{-\gamma\Delta t}}={C1:.4f}$ and the FDT amplitude is "
             rf"$\sqrt{{(T/m)(1-c_1^2)}}={FDT_STD:.4f}$. Panel C uses two of the "
             r"framework's Gaussian wells, $\kappa^2=2$.",
             ha="center", fontsize=9.5, color=MUTED)
    save(fig, "fdt_the_lock.png")


if __name__ == "__main__":
    main()
    print(f"c1 = {C1:.6f}   FDT std = {FDT_STD:.6f}   T/m = {TEMP/MASS}")
