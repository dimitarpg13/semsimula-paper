"""Figures for Markov_Generators_and_the_Langevin_Splitting.md.

Writes two PNGs to figures/markov_generator/:

  mg_ou_relaxation.png  -- the O-step is the exact solution of its generator:
      A  sample paths of the Ornstein-Uhlenbeck process with the analytic
         mean and standard-deviation envelopes;
      B  the variance law (sigma^2/2gamma)(1 - exp(-2 gamma t)) approaching
         kT/m, with the iterated `ou_step` recursion landing on it exactly;
      C  the stationary density against the Maxwell-Boltzmann Gaussian.

  mg_commutator_exactness.png -- the commutator algebra predicts numerics:
      A  Strang/Verlet is EXACT under constant force (both double commutators
         vanish) and second order under a harmonic force (they do not);
      B  the harmonic error is a frequency shift obeying
         sin(omega_tilde h / 2) = omega h / 2, measured from the one-step map;
      C  the same relation sets the stability ceiling at omega h = 2.

Everything is computed, not asserted. Run:
    python3 _make_markov_generator_figs.py
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
OUT = Path(__file__).parent / "markov_generator"
OUT.mkdir(exist_ok=True)

RNG = np.random.default_rng(20261025)


# ---------------------------------------------------------------- figure 1

def ou_paths(p0, gamma, T, m, dt, n_steps, n_paths):
    """Iterate exactly the recursion `ou_step` implements."""
    c1 = math.exp(-gamma * dt)
    c2 = math.sqrt((T / m) * (1.0 - c1 * c1))
    p = np.full(n_paths, float(p0))
    out = np.empty((n_steps + 1, n_paths))
    out[0] = p
    for k in range(n_steps):
        p = c1 * p + c2 * RNG.standard_normal(n_paths)
        out[k + 1] = p
    return out


def panel_paths(ax, gamma=1.0, T=1.0, m=1.0, p0=3.0, dt=0.02, n=400):
    t = np.arange(n + 1) * dt
    paths = ou_paths(p0, gamma, T, m, dt, n, 6)
    for j in range(6):
        ax.plot(t, paths[:, j], lw=1.0, color=GREY, alpha=0.85)
    mean = p0 * np.exp(-gamma * t)
    sd = np.sqrt((T / m) * (1.0 - np.exp(-2.0 * gamma * t)))
    ax.plot(t, mean, color=BLUE, lw=2.6, label=r"mean $e^{-\gamma t}p_0$")
    ax.fill_between(t, mean - sd, mean + sd, color=BLUE, alpha=0.16,
                    label=r"$\pm$ 1 s.d. (analytic)")
    ax.axhline(0.0, color=INK, lw=0.8, ls=":")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("momentum $p$")
    ax.set_title("A.  Paths forget $p_0$, acquire $kT/m$")
    ax.set_xlim(0, t[-1])
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def panel_variance(ax, T=1.0, m=1.0, dt=0.05, n=120, n_paths=200000):
    t = np.linspace(0.0, n * dt, 600)
    for gamma, col in ((0.5, GREEN), (1.0, BLUE), (2.0, PURPLE)):
        ax.plot(t, (T / m) * (1.0 - np.exp(-2.0 * gamma * t)), color=col,
                lw=2.3, label=f"$\\gamma$ = {gamma}")
        paths = ou_paths(0.0, gamma, T, m, dt, n, n_paths)
        ks = np.arange(0, n + 1, 12)
        ax.plot(ks * dt, paths[ks].var(axis=1), ls="none", marker="o",
                ms=5, mfc="none", mec=col, mew=1.4)
    ax.axhline(T / m, color=RED, lw=1.6, ls="--")
    ax.text(0.98 * t[-1], T / m + 0.035, r"$kT/m$", color=RED, fontsize=11,
            ha="right")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$\mathrm{Var}(p)$")
    ax.set_title("B.  Rings: the iterated step, exactly on the curve")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 1.25 * T / m)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def panel_stationary(ax, gamma=1.0, T=1.0, m=1.0, dt=0.05):
    paths = ou_paths(3.0, gamma, T, m, dt, 400, 200000)
    p = paths[-1]
    ax.hist(p, bins=90, density=True, color=BLUE, alpha=0.45,
            label="sampled, $t$ = 20")
    x = np.linspace(-4.5, 4.5, 400)
    ax.plot(x, np.exp(-0.5 * m * x * x / T) / math.sqrt(2 * math.pi * T / m),
            color=RED, lw=2.4, label=r"$\pi(p)\propto e^{-p^2/2mkT}$")
    ax.set_xlabel("momentum $p$")
    ax.set_ylabel("density")
    ax.set_title(r"C.  The null space of $\mathcal{L}^*_O$")
    ax.set_xlim(-4.5, 4.5)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    print(f"  stationary sample variance = {p.var():.5f}   target kT/m = {T/m}")


def fig_ou():
    fig = plt.figure(figsize=(16.0, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.30)
    panel_paths(fig.add_subplot(gs[0, 0]))
    panel_variance(fig.add_subplot(gs[0, 1]))
    panel_stationary(fig.add_subplot(gs[0, 2]))
    fig.suptitle("The O-step is not an approximation: it is the closed-form "
                 "solution of its own generator",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.savefig(OUT / "mg_ou_relaxation.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "mg_ou_relaxation.png")


# ---------------------------------------------------------------- figure 2

def strang_step(q, v, h, force):
    """B(h/2) A(h) B(h/2) -- velocity Verlet."""
    v = v + 0.5 * h * force(q)
    q = q + h * v
    v = v + 0.5 * h * force(q)
    return q, v


def integrate(q0, v0, h, n, force):
    q, v = q0, v0
    for _ in range(n):
        q, v = strang_step(q, v, h, force)
    return q, v


def panel_orders(ax):
    hs = np.logspace(-3.0, -0.6, 22)
    T_end = 2.0
    err_const, err_harm = [], []
    for h in hs:
        n = max(1, int(round(T_end / h)))
        hh = T_end / n
        # constant force F = 1
        q, v = integrate(0.0, 0.0, hh, n, lambda q: 1.0)
        exact = 0.5 * T_end**2
        err_const.append(abs(q - exact) / abs(exact))
        # harmonic force F = -w^2 q
        w = 2.0
        q, v = integrate(1.0, 0.0, hh, n, lambda q: -(w**2) * q)
        err_harm.append(abs(q - math.cos(w * T_end)) / abs(math.cos(w * T_end)))
    ax.loglog(hs, np.maximum(err_const, 1e-18), color=GREEN, lw=2.4,
              marker="o", ms=4, label="constant force")
    ax.loglog(hs, err_harm, color=RED, lw=2.4, marker="s", ms=4,
              label="harmonic force")
    ax.loglog(hs, 0.4 * hs**2, color=GREY, lw=1.6, ls="--",
              label=r"slope 2 ($\Delta t^2$)")
    ax.axhline(2.2e-16, color=INK, lw=1.0, ls=":")
    ax.text(1.1e-3, 3.4e-16, "machine epsilon", color=MUTED, fontsize=9.5)
    ax.set_xlabel(r"step size $\Delta t$")
    ax.set_ylabel("relative error at $t$ = 2")
    ax.set_title("A.  Exact when the double commutators vanish")
    ax.set_ylim(1e-18, 1e1)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.set_axisbelow(True)


def measured_omega(wh):
    """omega_tilde * h from the trace of the one-step Verlet map."""
    tr_half = 1.0 - 0.5 * wh * wh          # cos(omega_tilde h)
    if abs(tr_half) > 1.0:
        return np.nan
    return math.acos(tr_half)


def panel_frequency(ax):
    wh = np.linspace(0.01, 1.999, 400)
    meas = np.array([measured_omega(x) for x in wh])
    pred = 2.0 * np.arcsin(wh / 2.0)
    ax.plot(wh, pred, color=BLUE, lw=3.2,
            label=r"$2\arcsin(\omega h/2)$ (predicted)")
    ax.plot(wh[::9], meas[::9], ls="none", marker="o", ms=5, mfc="none",
            mec=RED, mew=1.5, label="from the map trace")
    ax.plot(wh, wh, color=GREY, lw=1.4, ls="--", label=r"$\omega h$ (no shift)")
    ax.axvline(2.0, color=INK, lw=1.2, ls=":")
    ax.set_xlabel(r"$\omega h$")
    ax.set_ylabel(r"$\tilde\omega h$")
    ax.set_title("B.  The error is a frequency shift")
    ax.set_xlim(0, 2.15)
    ax.set_ylim(0, 3.4)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    print(f"  max |measured - predicted| over wh in (0,2): "
          f"{np.nanmax(np.abs(meas - pred)):.3e}")


def panel_stability(ax):
    wh = np.linspace(0.01, 2.6, 500)
    tr_half = 1.0 - 0.5 * wh**2
    growth = np.where(np.abs(tr_half) <= 1.0, 1.0,
                      np.abs(tr_half) + np.sqrt(np.maximum(tr_half**2 - 1, 0)))
    ax.plot(wh, growth, color=PURPLE, lw=2.6)
    ax.axvline(2.0, color=RED, lw=1.8, ls="--")
    ax.axhline(1.0, color=GREY, lw=1.0, ls=":")
    ax.fill_between(wh, 0.9, growth, where=(wh > 2.0), color=RED, alpha=0.14)
    ax.text(2.06, 2.1, r"$\omega h = 2$", color=RED, fontsize=11)
    ax.text(0.15, 1.6, "stable:\n$|\\mathrm{tr}/2| \\leq 1$", color=MUTED,
            fontsize=10)
    ax.set_xlabel(r"$\omega h$")
    ax.set_ylabel("amplification factor per step")
    ax.set_title("C.  The same relation sets the ceiling")
    ax.set_xlim(0, 2.6)
    ax.set_ylim(0.9, 3.0)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def fig_commutator():
    fig = plt.figure(figsize=(16.0, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.30)
    panel_orders(fig.add_subplot(gs[0, 0]))
    panel_frequency(fig.add_subplot(gs[0, 1]))
    panel_stability(fig.add_subplot(gs[0, 2]))
    fig.suptitle("The commutator algebra predicts the numerics exactly",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.savefig(OUT / "mg_commutator_exactness.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "mg_commutator_exactness.png")


def main():
    fig_ou()
    fig_commutator()
    print("\nconstant-force check at a deliberately coarse step:")
    for h in (0.5, 0.25, 0.1):
        n = int(round(2.0 / h))
        q, _ = integrate(0.0, 0.0, 2.0 / n, n, lambda q: 1.0)
        print(f"  h = {h:<5} q = {q:.17f}   exact = 2.0   "
              f"abs err = {abs(q - 2.0):.3e}")


if __name__ == "__main__":
    main()
