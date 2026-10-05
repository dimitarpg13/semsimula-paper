"""Figures for Doi_Peliti_Dynamics_of_Semantic_Particles_and_Registers.md.

Theory PNGs into figures/doi_peliti/. Every panel is an exact computation or
an exact stochastic simulation (Gillespie, or BAOAB Langevin) of the stated
process. Nothing is fit to model data; the DP1-DP3 results figure is written
by notebooks/conservative_arch/scaleup/debug/dp_register_statistics.py.

  dp_boson_vs_exclusion.png -- two processes with the SAME mean equation:
                               bosonic creation/decay and the two-state
                               (exclusion) register. Gillespie means against
                               the rate equation; stationary Poisson against
                               Bernoulli; variances over time (S2).
  dp_layer_chain.png        -- the code's per-layer salience update as the
                               mean of a two-state chain: trajectories from
                               the full start, and the share of the initial
                               value still present after L layers (S3, S6).
  dp_phase_plane.png        -- Hamilton's equations of the coherent-state
                               action for creation/decay: the invariant line
                               phi~ = 1 is the rate equation; the WKB
                               large-deviation function against the exact
                               Poisson law (S4).
  dp_kramers_vs_smoluchowski.png -- an ensemble in the Gaussian well at the
                               ladder's damping ratio (0.05) and overdamped
                               (5): what position-only Doi-Peliti misses (S5).

Run:  python3 _make_doi_peliti_figs.py
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
OUT = Path(__file__).parent / "doi_peliti"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261005)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------
def gillespie_counts(rate_fn, n0, t_grid, n_runs):
    """Exact SSA for a one-species birth-death process.
    rate_fn(n) -> (birth_rate, death_rate). Returns (n_runs, len(t_grid))."""
    out = np.zeros((n_runs, len(t_grid)), dtype=int)
    for r in range(n_runs):
        t, n, k = 0.0, n0, 0
        while k < len(t_grid):
            b, d = rate_fn(n)
            tot = b + d
            dt = RNG.exponential(1 / tot) if tot > 0 else np.inf
            while k < len(t_grid) and t_grid[k] < t + dt:
                out[r, k] = n
                k += 1
            t += dt
            if tot > 0:
                n += 1 if RNG.random() < b / tot else -1
    return out


def fig_boson_vs_exclusion():
    k_on, k_off = 1.5, 1.0
    rho_star = k_on / (k_on + k_off)               # 0.6
    c, d = k_on, k_on + k_off                      # bosonic rates with the same mean equation
    t = np.linspace(0, 4, 121)
    bos = gillespie_counts(lambda n: (c, d * n), 0, t, 4000)
    exc = gillespie_counts(lambda n: (k_on * (n == 0), k_off * n), 0, t, 4000)
    mean_eq = rho_star * (1 - np.exp(-(k_on + k_off) * t))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(t, bos.mean(0), color=BLUE, lw=2, label="bosonic: creation c, decay d per particle")
    ax[0].plot(t, exc.mean(0), color=RED, lw=2, ls="--", label="exclusion: on k_on if empty, off k_off")
    ax[0].plot(t, mean_eq, color="k", lw=1, ls=":", label="shared rate equation")
    ax[0].set_xlabel("time"); ax[0].set_ylabel("mean occupation")
    ax[0].set_title("Same mean (4000 Gillespie runs each)")
    ax[0].legend(fontsize=8.5, frameon=False, loc="lower right")
    n = np.arange(0, 5)
    pois = np.array([math.exp(-rho_star) * rho_star ** m / math.factorial(m) for m in n])
    bern = np.array([1 - rho_star, rho_star, 0, 0, 0])
    ax[1].bar(n - 0.18, pois, 0.36, color=BLUE, label=f"Poisson, mean {rho_star}")
    ax[1].bar(n + 0.18, bern, 0.36, color=RED, label=f"Bernoulli, mean {rho_star}")
    emp_b = np.bincount(bos[:, -1], minlength=5)[:5] / bos.shape[0]
    emp_e = np.bincount(exc[:, -1], minlength=5)[:5] / exc.shape[0]
    ax[1].plot(n - 0.18, emp_b, "k.", ms=8, label="Gillespie at t = 4")
    ax[1].plot(n + 0.18, emp_e, "k.", ms=8)
    ax[1].set_xlabel("occupation n"); ax[1].set_title("Different statistics")
    ax[1].text(2.1, 0.32, f"P(n >= 2) = {1 - pois[0] - pois[1]:.3f}\nfor the boson", fontsize=9)
    ax[1].legend(fontsize=8.5, frameon=False)
    ax[2].plot(t, bos.var(0), color=BLUE, lw=2, label="bosonic variance (= mean)")
    ax[2].plot(t, exc.var(0), color=RED, lw=2, ls="--", label="exclusion variance (= mean (1 - mean))")
    ax[2].plot(t, mean_eq, color=BLUE, lw=0.8, ls=":")
    ax[2].plot(t, mean_eq * (1 - mean_eq), color=RED, lw=0.8, ls=":")
    ax[2].set_xlabel("time"); ax[2].set_title("Fluctuations tell them apart")
    ax[2].legend(fontsize=8.5, frameon=False, loc="lower right")
    save(fig, "dp_boson_vs_exclusion.png")


# ---------------------------------------------------------------------------
def fig_layer_chain():
    lam = 0.5
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
    L = 8
    for m, g, c in ((0.9, 0.0, BLUE), (0.3, 0.0, GREEN), (0.05, 0.0, ORANGE), (0.3, 0.3, RED)):
        s = [1.0]
        for _ in range(L):
            s.append((lam * s[-1] + (1 - lam) * m) * (1 - g))
        fix = (1 - lam) * m * (1 - g) / (1 - lam * (1 - g))
        ax[0].plot(range(L + 1), s, "o-", color=c, ms=4, label=f"creation weight {m}, destruction {g}")
        ax[0].axhline(fix, color=c, lw=0.7, ls=":")
    ax[0].axhline(0.005, color="k", lw=0.8, ls="--")
    ax[0].text(8, 0.02, "activity threshold 0.005", ha="right", fontsize=8.5)
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("salience")
    ax[0].set_title("Salience as a two-state occupancy, from the full start")
    ax[0].legend(fontsize=8, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.68))
    gs = np.linspace(0, 0.6, 7)
    Ls = np.arange(1, 9)
    for g, c in zip((0.0, 0.2, 0.4), (BLUE, GREEN, RED)):
        ax[1].plot(Ls, (lam * (1 - g)) ** Ls, "o-", color=c, ms=4, label=f"destruction g = {g}")
    ax[1].set_yscale("log")
    ax[1].axvspan(1.6, 2.4, color=GREY, alpha=0.2); ax[1].text(2, 1.3e-3, "L = 2", ha="center", fontsize=9)
    ax[1].axvspan(3.6, 4.4, color=GREY, alpha=0.2); ax[1].text(4, 1.3e-3, "L = 4", ha="center", fontsize=9)
    ax[1].set_xlabel("depth L"); ax[1].set_ylabel("share of the initial salience left")
    ax[1].set_title("How much the full start is remembered")
    ax[1].legend(fontsize=9, frameon=False)
    save(fig, "dp_layer_chain.png")


# ---------------------------------------------------------------------------
def fig_phase_plane():
    c, d = 3.0, 1.0
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))
    ph, pt = np.meshgrid(np.linspace(0, 6, 25), np.linspace(0.2, 2.2, 25))
    dph = c - d * ph                    # dphi/dt  = dH/dphi~
    dpt = d * (pt - 1)                  # dphi~/dt = -dH/dphi
    ax[0].streamplot(ph, pt, dph, dpt, color=GREY, density=1.1, linewidth=0.7, arrowsize=0.8)
    ax[0].axhline(1, color=BLUE, lw=2.2, label="phi~ = 1: the rate equation")
    ax[0].axvline(c / d, color=RED, lw=2.2, label="phi = c/d: the activation line")
    ax[0].plot([c / d], [1], "ko")
    ax[0].set_xlabel("phi (Doi field)"); ax[0].set_ylabel("phi~ (response field)")
    ax[0].set_xlim(0, 6); ax[0].set_ylim(0.2, 2.2)
    ax[0].set_title("Hamilton flow of H = (phi~ - 1)(c - d phi)")
    ax[0].legend(fontsize=8.5, frameon=True, loc="upper right")
    n = np.arange(0, 16)
    alpha = c / d
    exact = np.array([-alpha + m * math.log(alpha) - math.lgamma(m + 1) for m in n])
    S = np.array([alpha if m == 0 else m * math.log(m / alpha) - m + alpha for m in n])
    ax[1].plot(n, exact, "o", color=BLUE, label="exact log P(n), Poisson mean 3")
    ax[1].plot(n, -S, "--", color=RED, lw=1.2, label="WKB exponent only:  - S(n)")
    pref = np.array([0.0 if m == 0 else -0.5 * math.log(2 * math.pi * m) for m in n])
    ax[1].plot(n[1:], (-S + pref)[1:], "-", color=RED, lw=2,
               label="WKB with Gaussian prefactor:  - S(n) - log(2 pi n)/2")
    ax[1].set_xlabel("occupation n"); ax[1].set_ylabel("log probability")
    ax[1].set_title("Large deviations from the Hamiltonian")
    ax[1].legend(fontsize=8.5, frameon=False, loc="lower left")
    save(fig, "dp_phase_plane.png")


# ---------------------------------------------------------------------------
def fig_kramers():
    # Gaussian well V = m u^2 (1 - exp(-k^2 x^2)), m = u = 1, k^2 = 2 (x* = 0.5)
    k2, T = 2.0, 0.02
    force = lambda x: -2 * k2 * x * np.exp(-k2 * x * x)
    omega = math.sqrt(2 * k2)                    # harmonic frequency at the bottom
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
    for zeta, col, name in ((0.05, BLUE, "underdamped, zeta 0.05 (the ladder)"), (5.0, RED, "overdamped, zeta 5")):
        gam = 2 * zeta * omega
        dt = 0.01 if zeta < 1 else 0.002
        nstep = int(12 / dt)
        x = np.full(4000, 0.4); v = np.zeros(4000)
        c1 = math.exp(-gam * dt); c2 = math.sqrt((1 - c1 * c1) * T)
        ts, mean, sd = [], [], []
        for s in range(nstep + 1):
            if s % int(0.05 / dt) == 0:
                ts.append(s * dt); mean.append(x.mean()); sd.append(x.std())
            v += 0.5 * dt * force(x); x += 0.5 * dt * v                  # B, A
            v = c1 * v + c2 * RNG.standard_normal(x.size)               # O
            x += 0.5 * dt * v; v += 0.5 * dt * force(x)                  # A, B
        ax[0].plot(ts, mean, color=col, lw=2, label=name)
        ax[1].plot(ts, sd, color=col, lw=2, label=name)
    ax[0].axhline(0, color=GREY, lw=0.6)
    ax[0].set_xlabel("time"); ax[0].set_ylabel("ensemble mean position")
    ax[0].set_title("Relaxation in the Gaussian well")
    ax[0].legend(fontsize=8.5, frameon=False)
    ax[1].axhline(math.sqrt(T / (2 * k2)), color="k", lw=0.8, ls=":")
    ax[1].text(12, math.sqrt(T / (2 * k2)) * 1.05, "Boltzmann width", ha="right", fontsize=8.5)
    ax[1].set_xlabel("time"); ax[1].set_ylabel("ensemble spread")
    ax[1].set_title("Spread: same endpoint, different path")
    ax[1].legend(fontsize=8.5, frameon=False, loc="lower right")
    save(fig, "dp_kramers_vs_smoluchowski.png")


if __name__ == "__main__":
    fig_boson_vs_exclusion()
    fig_layer_chain()
    fig_phase_plane()
    fig_kramers()
