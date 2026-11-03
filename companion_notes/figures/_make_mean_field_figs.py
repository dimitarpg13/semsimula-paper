#!/usr/bin/env python3
"""Figures for Mean_Field_Theory_for_Particle_Language_Models.md (2026-11-03).

    python3 _make_mean_field_figs.py          # writes figures/mean_field/*.png

One figure reads the temperature sweeps of vtheta_weight_logits.py (results/dead_wells) and marks each layer's trained
operating point; it is skipped when the files are absent. The other four are synthetic and say so in their titles.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN, BLUE, RED, PURPLE, GREY, ORANGE = "#22C55E", "#3B82F6", "#EF4444", "#8B5CF6", "#9CA3AF", "#F59E0B"
INK, MUTED = "#1F2937", "#6B7280"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 150})

HERE = Path(__file__).parent
OUT = HERE / "mean_field"
OUT.mkdir(exist_ok=True)
DW = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/dead_wells"
rng = np.random.default_rng(20261103)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------------------------------------------------
def _parse_probe(path):
    """The per-layer table (|x| median, trained n_eff median) and the temperature sweep (r, n_eff per layer)."""
    txt = path.read_text().splitlines()
    layers = {}
    for line in txt:
        p = line.split()
        if len(p) == 8 and p[0].isdigit():
            try:
                layers[int(p[0])] = (float(p[1]), float(p[5]))
            except ValueError:
                pass
    i = next((k for k, l in enumerate(txt) if l.startswith("temperature sweep")), None)
    if i is None:
        return layers, None
    rows = []
    for line in txt[i + 2:]:
        p = line.split()
        if not p:
            break
        rows.append([float(v) for v in p])
    return layers, np.array(rows)


def fig_entropy_vs_temperature():
    """The well bank's entropy S = ln n_eff against the input norm r (the inverse temperature), per layer and model,
    with each layer's trained operating point (its median |x|, its trained n_eff) marked."""
    specs = [("L8-CODE (step 18,000)", "vtheta_weight_logits_l8code_code_step18000.txt", BLUE),
             ("SO-CODE", "vtheta_weight_logits_so_code.txt", GREEN),
             ("SO-OWT", "vtheta_weight_logits_so_owt.txt", ORANGE)]
    data = [(n, *_parse_probe(DW / f), c) for n, f, c in specs if (DW / f).exists()]
    data = [d for d in data if d[2] is not None]
    if not data:
        print("skip entropy_vs_temperature: no sweep files"); return
    fig, axes = plt.subplots(1, len(data), figsize=(4.2 * len(data) + 1, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, layers, sweep, col) in zip(axes, data):
        r = sweep[:, 0]; L = sweep.shape[1] - 1
        for g in range(L):
            shade = 0.25 + 0.75 * g / max(L - 1, 1)
            ax.plot(r, np.log(sweep[:, 1 + g]), color=col, alpha=shade, lw=1.8, label=f"layer {g}" if L <= 2 or g in (0, L - 1) else None)
            if g in layers:
                xm, ne = layers[g]
                ax.scatter([xm], [np.log(ne)], s=46, color=INK, zorder=4, marker="o" if g == 0 else "s")
                print(f"{name} layer {g}: |x| median {xm:.2f}, trained n_eff {ne:.2f}")
        ax.axhline(np.log(8), color=GREY, ls="--", lw=1); ax.text(0.27, np.log(8) + 0.04, "ln 8: uniform over 8 wells", color=MUTED, fontsize=8.5)
        ax.set_xscale("log"); ax.set_xlabel("bank input norm r  (an inverse temperature)")
        ax.set_title(name, fontsize=11); ax.legend(frameon=False, fontsize=9, loc="lower left")
    axes[0].set_ylabel("entropy of the well weights  S = ln n_eff")
    axes[0].scatter([], [], s=46, color=INK, marker="o", label="trained operating point, layer 0")
    axes[0].scatter([], [], s=46, color=INK, marker="s", label="trained operating point, deeper layers")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("Measured: every token's bank input rescaled to norm r; median over tokens (vtheta_weight_logits.py sweep=1)", fontsize=10, color=MUTED, y=1.02)
    save(fig, "entropy_vs_temperature.png")


# ---------------------------------------------------------------------------------------------------------------------
def fig_curie_weiss():
    """Synthetic: the Curie-Weiss free energy per spin and the self-consistent magnetisation."""
    J = 1.0
    m = np.linspace(-0.999, 0.999, 999)
    ent = -((1 + m) / 2 * np.log((1 + m) / 2) + (1 - m) / 2 * np.log((1 - m) / 2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for T, col in ((1.5, BLUE), (1.0, PURPLE), (0.6, RED)):
        f = -0.5 * J * m * m - T * ent
        ax.plot(m, f - f.min(), color=col, lw=2, label=f"T/J = {T}")
    ax.set_xlabel("magnetisation m (the order parameter)"); ax.set_ylabel("free energy per spin  f(m) − min f")
    ax.set_title("(a) Free energy: one minimum above $T_c = J$, two below", fontsize=11); ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    Ts = np.linspace(0.05, 2.0, 200)
    for h, col, lab in ((0.0, RED, "h = 0"), (0.05, ORANGE, "h = 0.05 J (a small external field)")):
        ms = []
        for T in Ts:
            x = 0.9
            for _ in range(500):
                x = np.tanh((J * x + h) / T)
            ms.append(x)
        ax.plot(Ts, ms, color=col, lw=2, label=lab)
    ax.axvline(1.0, color=GREY, ls="--", lw=1); ax.text(1.03, 0.9, "$T_c$", color=MUTED)
    ax.set_xlabel("temperature T / J"); ax.set_ylabel("self-consistent m = tanh((J m + h)/T)")
    ax.set_title("(b) The order parameter against temperature", fontsize=11); ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Synthetic illustration: the Curie–Weiss model (every spin coupled to the mean of all the others)", fontsize=10, color=MUTED, y=1.02)
    save(fig, "curie_weiss.png")


# ---------------------------------------------------------------------------------------------------------------------
def fig_desai_zwanzig():
    """Synthetic: N particles in a double well, each attracted to the mean of all (Desai-Zwanzig), at temperature T.
    The order parameter |mean| against T, with the particle density at two temperatures."""
    N, theta, dt = 2000, 1.0, 0.005
    Ts = np.linspace(0.05, 1.2, 16)
    order, snaps = [], {}
    for T in Ts:
        x = 1.0 + 0.1 * rng.standard_normal(N)
        acc = []
        for s in range(6000):
            m = x.mean()
            x = x - (x ** 3 - x) * dt - theta * (x - m) * dt + np.sqrt(2 * T * dt) * rng.standard_normal(N)
            if s >= 4000:
                acc.append(abs(x.mean()))
        order.append(np.mean(acc))
        if abs(T - 0.2) < 0.04 or abs(T - 1.05) < 0.04:
            snaps[round(T, 2)] = x.copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for (T, xs), col in zip(sorted(snaps.items()), (RED, BLUE)):
        h, e = np.histogram(xs, bins=60, range=(-2.2, 2.2), density=True)
        ax.fill_between(0.5 * (e[1:] + e[:-1]), 0, h, color=col, alpha=0.4, label=f"T = {T:g}")
    z = np.linspace(-2.2, 2.2, 300); ax.plot(z, 0.25 * (z * z - 1) ** 2, color=INK, lw=1.5, label="the double well U(x) (scaled)")
    ax.set_xlabel("state x"); ax.set_ylabel("particle density"); ax.set_title("(a) Cold: one well occupied. Hot: both.", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    ax.plot(Ts, order, "o-", color=PURPLE, lw=2)
    ax.set_xlabel("temperature T"); ax.set_ylabel("order parameter  |mean of x|")
    ax.set_title("(b) The mean field's phase transition", fontsize=11)
    fig.suptitle("Synthetic illustration: the Desai–Zwanzig model, 2,000 double-well particles with a mean-field attraction", fontsize=10, color=MUTED, y=1.02)
    save(fig, "desai_zwanzig.png")


# ---------------------------------------------------------------------------------------------------------------------
def fig_potts_selfconsistency():
    """Synthetic: K = 8 states whose weights are p = softmax(beta (J p + h)): a bank whose context field is made of
    the wells it selects. The effective count of the stable fixed point against beta J, with and without a bias."""
    K = 8
    betas = np.geomspace(0.25, 64, 200)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for h_scale, col, lab in ((0.0, RED, "no bias: a sharp transition at βJ = K"),
                              (0.3, ORANGE, "a fixed bias h (an external field): a crossover")):
        h = h_scale * np.array([1.0, 0.8, 0.5, 0.2, 0, -0.2, -0.5, -0.8])
        ne = []
        p = np.full(K, 1.0 / K) + 1e-3 * rng.standard_normal(K); p /= p.sum()
        for b in betas:
            for _ in range(2000):
                z = b * (p + h); z -= z.max()
                q = np.exp(z); q /= q.sum()
                p = 0.5 * p + 0.5 * q
            ne.append(np.exp(-(p * np.log(p.clip(1e-300))).sum()))
        ax.plot(betas, ne, color=col, lw=2.2, label=lab)
    ax.axvline(K, color=GREY, ls="--", lw=1); ax.text(K * 1.05, 7.2, "βJ = K", color=MUTED)
    ax.set_xscale("log"); ax.set_xlabel("coupling βJ  (the gain of the context field)"); ax.set_ylabel("effective count of occupied states")
    ax.set_title("Synthetic: self-consistent occupancy p = softmax(β(Jp + h)), K = 8", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    save(fig, "potts_selfconsistency.png")


# ---------------------------------------------------------------------------------------------------------------------
def fig_free_energy_descent():
    """Synthetic: a density in a double well with a mean-field attraction evolves by the McKean-Vlasov (nonlinear
    Fokker-Planck) equation; its free energy F[rho] = int V rho + 1/2 int int W rho rho + T int rho ln rho decreases."""
    x = np.linspace(-2.5, 2.5, 401); dx = x[1] - x[0]
    V = (x * x - 1) ** 2; theta, T, dt = 0.8, 0.15, 2e-4
    rho = np.exp(-((x + 0.3) ** 2) / (2 * 0.5 ** 2)); rho /= rho.sum() * dx
    W = lambda r: 0.5 * theta * (x[:, None] - x[None, :]) ** 2
    Wmat = W(None)
    def free_energy(r):
        return float((V * r).sum() * dx + 0.5 * (r @ Wmat @ r) * dx * dx + T * (r * np.log(r.clip(1e-300))).sum() * dx)
    ts, Fs, snaps = [], [], {}
    for s in range(60001):
        if s % 200 == 0:
            ts.append(s * dt); Fs.append(free_energy(rho))
        if s in (0, 4000, 20000, 60000):
            snaps[s * dt] = rho.copy()
        mu = V + Wmat @ rho * dx + T * np.log(rho.clip(1e-300))       # the variational derivative of F
        flux = rho * np.gradient(mu, dx)
        rho = rho + dt * np.gradient(flux, dx)
        rho = rho.clip(1e-12); rho /= rho.sum() * dx
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for (t, r), col in zip(sorted(snaps.items()), (BLUE, PURPLE, ORANGE, RED)):
        ax.plot(x, r, color=col, lw=2, label=f"t = {t:g}")
    ax.plot(x, V / V.max() * 1.2, color=INK, lw=1, ls="--", label="V(x) (scaled)")
    ax.set_xlabel("state x"); ax.set_ylabel("density ρ(x, t)"); ax.set_title("(a) The density moves to the nearer well and sharpens", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    ax.plot(ts, Fs, color=GREEN, lw=2.2)
    ax.set_xlabel("time"); ax.set_ylabel("free energy F[ρ]"); ax.set_title("(b) F[ρ] never increases: the dynamics is its gradient flow", fontsize=11)
    fig.suptitle("Synthetic illustration: a McKean–Vlasov density in a double well with a mean-field attraction (T = 0.15)", fontsize=10, color=MUTED, y=1.02)
    save(fig, "free_energy_descent.png")


if __name__ == "__main__":
    fig_entropy_vs_temperature()
    fig_curie_weiss()
    fig_desai_zwanzig()
    fig_potts_selfconsistency()
    fig_free_energy_descent()
