#!/usr/bin/env python3
"""Figures for Macroscopic_Limits_of_the_Conservative_Step.md (2026-11-03).

    python3 _make_macroscopic_figs.py          # writes figures/macroscopic/*.png

Three figures are synthetic experiments (the mean-field convergence, the Kramers populations, the shared basis) and
say so in their titles; one reads measured numbers typed in from the profile notes; one parses the L8-CODE probes
in notebooks/conservative_arch/scaleup/results/dead_wells and is skipped when they are absent.
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
OUT = HERE / "macroscopic"
OUT.mkdir(exist_ok=True)
DW = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/dead_wells"
rng = np.random.default_rng(20261103)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------------------------------------------------
def fig_particles_vs_field():
    """Left: the force on one token from its 16 routed neighbours. Right: the same token in the field of a density."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    centres = np.array([[-1.6, 0.9], [1.4, 1.1], [0.3, -1.5]])
    weights = np.array([0.5, 0.3, 0.2])
    sd = 0.45
    q = np.array([0.2, 0.3])
    eps = 0.35

    def force(x, ys, w=None):
        d = x - ys
        r3 = (np.sum(d * d, -1) + eps ** 2) ** 1.5
        f = -d / r3[:, None]
        return (f * (w[:, None] if w is not None else 1.0 / len(ys))).sum(0)

    # left: particles
    ax = axes[0]
    comp = rng.choice(3, size=16, p=weights)
    ys = centres[comp] + sd * rng.standard_normal((16, 2))
    fs = np.array([-(q - y) / (np.sum((q - y) ** 2) + eps ** 2) ** 1.5 for y in ys])
    fmax = np.linalg.norm(fs, axis=1).max()
    for fy in fs:                                   # each neighbour's pull, scaled to the strongest one
        ax.annotate("", xy=q + 0.7 * fy / fmax, xytext=q, arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0))
    ax.scatter(ys[:, 0], ys[:, 1], s=36, color=BLUE, zorder=3, label="16 routed neighbours $h_j$")
    f = force(q, ys)
    ax.annotate("", xy=q + 1.0 * f / np.linalg.norm(f), xytext=q, arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.5))
    ax.scatter(*q, s=90, color=RED, zorder=4, label="the token $h_t$")
    ax.set_title("Particle picture: a sum over neighbours", fontsize=11)
    ax.text(0.02, 0.02, r"$f_t=\sum_{j\in K(t)}\nabla V_\varphi(h_t,h_j)$", transform=ax.transAxes, fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    # right: field
    ax = axes[1]
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    dens = np.zeros_like(xx)
    for c, w in zip(centres, weights):
        dens += w * np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * sd ** 2))
    ax.contourf(xx, yy, dens, levels=12, cmap="Blues", alpha=0.85)
    ax.scatter(centres[:, 0], centres[:, 1], s=weights * 400, color=INK, marker="x", zorder=3,
               label="M = 3 components $(\\pi_m,\\mu_m)$")
    big = centres[rng.choice(3, size=20000, p=weights)] + sd * rng.standard_normal((20000, 2))
    f_field = force(q, big)
    ax.annotate("", xy=q + 1.0 * f_field / np.linalg.norm(f_field), xytext=q, arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.5))
    ax.scatter(*q, s=90, color=RED, zorder=4, label="the token $h_t$")
    ax.set_title("Field picture: an integral against a density", fontsize=11)
    ax.text(0.02, 0.02, r"$f_t=\int\nabla V_\varphi(h_t,h')\,\rho_t(dh')$", transform=ax.transAxes, fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "particles_vs_field.png")


# ---------------------------------------------------------------------------------------------------------------------
def fig_cost_anatomy():
    """Measured shares: GPU time per microbatch (Cell 6c profile, FOCK-CODE build, 2026-10-27) and the bytes one
    training forward saves for the backward (CPU count on SO-CODE at K = 2, analytic force, 2026-11-02)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    time_parts = [("element-wise kernels\n(mul, add, copy, sum)", 50, BLUE), ("matrix products\n(bmm, mm)", 27, GREEN),
                  ("float64 Gram SVD\n(stiff modes)", 10.6, ORANGE), ("other", 12.4, GREY)]
    mem_parts = [("analytic pair force\nintermediates (B,T,k,H)", 75, BLUE), ("Fock register path", 18, GREEN),
                 ("V_theta well bank (K = 2)", 6, ORANGE), ("other", 1, GREY)]
    for ax, parts, title in ((axes[0], time_parts, "GPU time, one microbatch (0.52 s CUDA)"),
                             (axes[1], mem_parts, "Activation memory saved for the backward, per layer")):
        left = 0
        for name, v, col in parts:
            ax.barh(0, v, left=left, color=col, edgecolor="white", height=0.6, label=f"{name.replace(chr(10), ' ')}: {v:g}%")
            if v >= 25:
                ax.text(left + v / 2, 0, f"{name}\n{v:g}%", ha="center", va="center", fontsize=8.5, color="white")
            left += v
        ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel("% of the total"); ax.set_title(title, fontsize=11)
        ax.spines["left"].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    save(fig, "cost_anatomy.png")


# ---------------------------------------------------------------------------------------------------------------------
def fig_scaling():
    """Pair or component evaluations per token and layer, against the context length T. Exact counts, not times."""
    T = np.array([128, 256, 512, 1024, 2048, 4096, 8192])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.loglog(T, T / 2, color=RED, lw=2.2, label="score head (router): every causal pair, T/2 per token")
    ax.loglog(T, np.full_like(T, 16), color=BLUE, lw=2.2, label="gathered pair force: k = 16 neighbours")
    ax.loglog(T, np.full_like(T, 8), color=GREEN, lw=2.2, ls="--", label="field force: M = 8 components")
    ax.loglog(T, np.full_like(T, 2), color=GREEN, lw=2.2, ls=":", label="field force: M = 2 components")
    ax.axvline(512, color=GREY, lw=1, ls="--"); ax.text(530, 30, "T = 512\n(every arm so far)", color=MUTED, fontsize=9)
    ax.set_xlabel("context length T (tokens)"); ax.set_ylabel("pair or component evaluations\nper token and layer")
    ax.set_title("What grows with context: the router is the only O(T) term per token", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, "scaling.png")


# ---------------------------------------------------------------------------------------------------------------------
def _gmm_em(x, M, iters=60, restarts=5):
    """A plain EM for a Gaussian mixture with spherical components, in numpy (illustration only): the best of
    `restarts` random initialisations by log-likelihood, the usual guard against EM's local optima."""
    n = len(x)
    best = None
    for _ in range(restarts):
        mu = x[rng.choice(n, M, replace=False)].copy()
        var = np.full(M, x.var())
        pi = np.full(M, 1.0 / M)
        for _ in range(iters):
            d2 = ((x[:, None, :] - mu[None]) ** 2).sum(-1)                  # (n, M)
            logp = np.log(pi)[None] - d2 / (2 * var[None]) - np.log(2 * np.pi * var[None])
            m = logp.max(1, keepdims=True)
            r = np.exp(logp - m); tot = r.sum(1, keepdims=True); r /= tot
            nk = r.sum(0) + 1e-9
            mu = (r.T @ x) / nk[:, None]
            var = ((r * ((x[:, None, :] - mu[None]) ** 2).sum(-1)).sum(0) / nk / 2).clip(1e-4)
            pi = nk / n
        ll = float((m[:, 0] + np.log(tot[:, 0])).sum())
        if best is None or ll > best[0]:
            best = (ll, pi, mu, var)
    return best[1:]


def fig_meanfield_convergence():
    """Synthetic: sources from a 3-component mixture in the plane; the softened inverse-distance force on 200 query
    points. (a) The N-particle average against the N -> infinity field: RMS error ~ N^(-1/2). (b) The field of an
    M-component mixture fitted by EM to 4,096 samples, against the same limit."""
    centres = np.array([[-1.6, 0.9], [1.4, 1.1], [0.3, -1.5]]); weights = np.array([0.5, 0.3, 0.2]); sd = 0.45
    eps = 0.35
    queries = rng.uniform(-2.5, 2.5, size=(200, 2))

    def sample(n, pi=weights, mu=centres, var=None):
        comp = rng.choice(len(pi), size=n, p=pi)
        s = sd if var is None else np.sqrt(var[comp])[:, None]
        return mu[comp] + s * rng.standard_normal((n, 2))

    def field(ys, chunk=20000):
        f = np.zeros_like(queries)
        for i in range(0, len(ys), chunk):
            y = ys[i:i + chunk]
            d = queries[:, None, :] - y[None]
            r3 = ((d * d).sum(-1) + eps ** 2) ** 1.5
            f += (-d / r3[..., None]).sum(1)
        return f / len(ys)

    truth = field(sample(400000))
    scale = np.sqrt((truth ** 2).sum(1).mean())
    Ns = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    err = []
    for N in Ns:
        e = [np.sqrt(((field(sample(N)) - truth) ** 2).sum(1).mean()) / scale for _ in range(12)]
        err.append((np.mean(e), np.std(e)))
    err = np.array(err)
    Ms = [1, 2, 3, 4, 6, 8]
    errM = []
    for M in Ms:
        e = []
        for _ in range(6):
            pi, mu, var = _gmm_em(sample(4096), M)
            e.append(np.sqrt(((field(sample(60000, pi, mu, var)) - truth) ** 2).sum(1).mean()) / scale)
        errM.append((np.mean(e), np.std(e)))
    errM = np.array(errM)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ax.errorbar(Ns, err[:, 0], yerr=err[:, 1], fmt="o-", color=BLUE, lw=2, capsize=3, label="N sampled particles")
    ref = err[0, 0] * np.sqrt(Ns[0] / Ns)
    ax.loglog(Ns, ref, color=GREY, ls="--", lw=1.5, label=r"$\propto N^{-1/2}$")
    ax.axvline(16, color=RED, lw=1, ls=":"); ax.text(17.5, err[:, 0].max() * 0.22, "k = 16\n(the routed\nneighbours)", color=RED, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of particles N"); ax.set_ylabel("relative RMS force error\nagainst the N → ∞ field")
    ax.set_title("(a) Particle sum: the error falls like $N^{-1/2}$", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    ax.errorbar(Ms, errM[:, 0], yerr=errM[:, 1], fmt="s-", color=GREEN, lw=2, capsize=3, label="M-component mixture, EM on 4,096 samples")
    ax.axhline(err[0, 0], color=BLUE, lw=1.5, ls="--", label="16 particles (panel a)")
    ax.axhline(err[-1, 0], color=BLUE, lw=1.5, ls=":", label="4,096 particles (panel a)")
    ax.set_yscale("log"); ax.set_xlabel("mixture components M"); ax.set_ylabel("relative RMS force error")
    ax.set_title("(b) Field of a fitted mixture: enough components match the limit", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Synthetic illustration (softened 1/r kernel, 3-component source density in the plane)", fontsize=10, color=MUTED, y=1.02)
    save(fig, "meanfield_convergence.png")


# ---------------------------------------------------------------------------------------------------------------------
def fig_kramers():
    """Synthetic: overdamped Langevin particles in a double well U(x) = (x^2 - 1)^2, started in the left well, at
    temperature T = 0.25. The fraction in the right well against the master equation with the Kramers rate."""
    Temp, dt, steps, n = 0.25, 0.01, 15000, 5000
    x = -1.0 + 0.15 * rng.standard_normal(n)
    U = lambda z: (z * z - 1) ** 2
    dU = lambda z: 4 * z * (z * z - 1)
    ts, pR, snaps = [], [], {}
    for s in range(steps + 1):
        if s % 50 == 0:
            ts.append(s * dt); pR.append((x > 0).mean())
        if s in (0, 3000, 15000):
            snaps[s * dt] = x.copy()
        x = x - dU(x) * dt + np.sqrt(2 * Temp * dt) * rng.standard_normal(n)
    ts, pR = np.array(ts), np.array(pR)
    k = np.sqrt(8 * 4) / (2 * np.pi) * np.exp(-1.0 / Temp)                  # Kramers, overdamped: sqrt(U''_min |U''_b|)/(2 pi) e^{-dU/T}
    theory = 0.5 * (1 - np.exp(-2 * k * ts))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    z = np.linspace(-1.8, 1.8, 400)
    ax.plot(z, U(z), color=INK, lw=2)
    for (t, xs), col in zip(sorted(snaps.items()), (BLUE, PURPLE, GREEN)):
        h, edges = np.histogram(xs, bins=60, range=(-1.8, 1.8), density=True)
        ax.fill_between(0.5 * (edges[1:] + edges[:-1]), 0, 0.6 * h, color=col, alpha=0.35, label=f"particles at t = {t:g}")
    ax.set_xlabel("state x"); ax.set_ylabel("U(x), and the particle density")
    ax.set_title("(a) Many particles in a double well", fontsize=11); ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    ax.plot(ts, pR, color=BLUE, lw=2.2, label="simulation: fraction in the right well")
    ax.plot(ts, theory, color=RED, lw=2, ls="--", label=r"master equation, Kramers rate $k=\frac{\sqrt{U''_{\min}|U''_b|}}{2\pi}e^{-\Delta U/T}$")
    ax.set_xlabel("time"); ax.set_ylabel("population $p_R(t)$")
    ax.set_title("(b) One macroscopic variable replaces 5,000 trajectories", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Synthetic illustration (overdamped Langevin, T = 0.25, barrier 1)", fontsize=10, color=MUTED, y=1.02)
    save(fig, "kramers_populations.png")


# ---------------------------------------------------------------------------------------------------------------------
def fig_shared_basis():
    """Synthetic: per-token generators G_t = G_0 + s E_t (d = 64, P = 8). How well the top-P subspace of the pooled
    Gram matrix captures each token's own: the largest principal angle and the energy fraction, against s."""
    d, P, n = 64, 8, 256
    G0 = rng.standard_normal((d, P))
    ss = np.linspace(0, 2.0, 21)
    ang, energy = [], []
    for s in ss:
        E = rng.standard_normal((n, d, P))
        G = G0[None] + s * E
        pooled = np.einsum("ndp,nqp->dq", G, G) / n
        Up = np.linalg.eigh(pooled)[1][:, -P:]                               # top-P eigenvectors
        a, en = [], []
        for t in range(n):
            Ut = np.linalg.svd(G[t], full_matrices=False)[0]
            sv = np.linalg.svd(Up.T @ Ut, compute_uv=False)
            a.append(np.degrees(np.arccos(np.clip(sv.min(), -1, 1))))
            en.append(np.sum((Up.T @ G[t]) ** 2) / np.sum(G[t] ** 2))
        ang.append((np.mean(a), np.std(a))); energy.append((np.mean(en), np.std(en)))
    ang, energy = np.array(ang), np.array(energy)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].errorbar(ss, ang[:, 0], yerr=ang[:, 1], fmt="o-", color=PURPLE, lw=2, capsize=2)
    axes[0].set_xlabel("per-token spread s (relative to the shared part)"); axes[0].set_ylabel("largest principal angle (degrees)")
    axes[0].set_title("(a) How far a token's stiff subspace is from the pooled one", fontsize=11)
    axes[1].errorbar(ss, energy[:, 0], yerr=energy[:, 1], fmt="s-", color=GREEN, lw=2, capsize=2)
    axes[1].axhline(1.0, color=GREY, ls="--", lw=1)
    axes[1].set_xlabel("per-token spread s"); axes[1].set_ylabel("fraction of $|G_t|^2$ inside the pooled subspace")
    axes[1].set_title("(b) What a shared basis keeps", fontsize=11)
    fig.suptitle("Synthetic illustration (d = 64, P = 8, 256 tokens): the measurable quantity is the per-token spread s", fontsize=10, color=MUTED, y=1.02)
    save(fig, "shared_basis.png")


# ---------------------------------------------------------------------------------------------------------------------
def fig_slow_variables():
    """L8-CODE at step 18,000: per layer, the median effective well count n_eff (vtheta_weight_logits) and the force
    share of the two strongest wells (well_probe). The evidence that deep layers have few slow variables."""
    probe = DW / "well_probe_l8code_code_step18000.txt"
    logits = DW / "vtheta_weight_logits_l8code_code_step18000.txt"
    if not (probe.exists() and logits.exists()):
        print("skip slow_variables: probe files missing"); return
    top2 = {}
    for m in re.finditer(r"^  layer (\d+): force\s+(.*?)\s+\|", probe.read_text(), re.M):
        sh = sorted((float(v) for v in m.group(2).split()), reverse=True)
        top2[int(m.group(1))] = sum(sh[:2]) / max(sum(sh), 1e-12)
    neff = {}
    for line in logits.read_text().splitlines():
        p = line.split()
        if len(p) == 8 and p[0].isdigit():
            try:
                neff[int(p[0])] = (float(p[5]), float(p[6]), float(p[7]))
            except ValueError:
                pass
    if not top2 or not neff:
        print("skip slow_variables: parse failed"); return
    L = sorted(set(top2) & set(neff))
    fig, ax1 = plt.subplots(figsize=(7.6, 4.2))
    ax1.bar([l - 0.2 for l in L], [neff[l][0] for l in L], width=0.4, color=BLUE, label="median n_eff (effective wells used)")
    ax1.errorbar([l - 0.2 for l in L], [neff[l][0] for l in L],
                 yerr=[[neff[l][0] - neff[l][1] for l in L], [neff[l][2] - neff[l][0] for l in L]], fmt="none", ecolor=INK, capsize=2)
    ax1.set_ylabel("n_eff (p10–p90 bars)"); ax1.set_xlabel("layer"); ax1.set_xticks(L); ax1.set_ylim(0, 8.4)
    ax2 = ax1.twinx()
    ax2.bar([l + 0.2 for l in L], [100 * top2[l] for l in L], width=0.4, color=ORANGE, label="force share of the two strongest wells")
    ax2.set_ylabel("% of the force weight"); ax2.set_ylim(0, 105); ax2.spines["top"].set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax1.set_title("L8-CODE (step 18,000): layer 0 uses many wells, the deep layers two", fontsize=11)
    save(fig, "slow_variables.png")


if __name__ == "__main__":
    fig_particles_vs_field()
    fig_cost_anatomy()
    fig_scaling()
    fig_meanfield_convergence()
    fig_kramers()
    fig_shared_basis()
    fig_slow_variables()
