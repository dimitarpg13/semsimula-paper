"""Figures for Poisson_Mode_Registers_PM1.md.

PNGs into figures/poisson_modes/. Every theory panel is an exact computation
or an exact simulation of the stated process in a two-dimensional toy space;
nothing is fit to model data. The probe panel replots the logged evals of the
three runs (values copied below from their training logs).

  pm_slots_vs_modes.png  -- schematic: the v2.1 slot bank (M always-occupied
                            exclusion slots, read by the reverse channel)
                            against PM1 (K shared modes, unbounded occupation,
                            read by a well force) (S1, S5).
  pm_bank_example.png    -- worked example: two contexts ending in the
                            ambiguous token "bank". Creation rates, occupations
                            and the potential the token feels; its damped
                            trajectory ends in the context's well (S2).
  pm_poisson_exact.png   -- the occupation is exactly a Poisson mean:
                            Monte Carlo of the token-time immigration-death
                            process against phi, the law at the peak, and
                            variance/mean over time (S3).
  pm_conservativity.png  -- PM1's force field against a toy reverse channel:
                            streamlines, curl, and the circulation around
                            closed loops against loop radius (S4).
  pm_probe.png           -- the 8,000-step probe: PM1 against F3.1 and G2, and
                            the pm_ group's pre-clip gradient norm (S6).
  pm_full_run.png        -- the full run: PM1 against F3.1 and G2 over 32,500
                            steps, and the refinement trade-off of the wells
                            (S6). Reads the three runs' logs and PM1's
                            localization outputs from the results folders.

Run:  python3 _make_poisson_mode_figs.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from scipy.stats import poisson

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "figure.dpi": 150,
})
GREEN, BLUE, RED = "#22C55E", "#3B82F6", "#EF4444"
PURPLE, GREY, ORANGE = "#8B5CF6", "#9CA3AF", "#F59E0B"
MODE_COL = [BLUE, GREEN, ORANGE]
OUT = Path(__file__).parent / "poisson_modes"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261006)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------
# The PM1 mechanism, exactly as in model_parf_multixi.py, in d dimensions.
def overlaps(h, mu, k2):
    """E[s, v] = exp(-k2_v |h_s - mu_v|^2)."""
    d2 = ((h[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
    return np.exp(-k2[None, :] * d2)


def occupation(E, lam):
    """phi[t, v] = sum_{s<t} lam_v^(t-1-s) E[s, v]  (strict past)."""
    T, K = E.shape
    phi = np.zeros((T, K))
    for t in range(1, T):
        phi[t] = lam * phi[t - 1] + E[t - 1]
    return phi


def pm_potential(h, phi_t, a, mu, k2):
    """U(h) = -sum_v phi_v a_v exp(-k2_v |h - mu_v|^2), h of shape (..., d)."""
    d2 = ((h[..., None, :] - mu) ** 2).sum(-1)
    return -(phi_t * a * np.exp(-k2 * d2)).sum(-1)


def pm_force(h, phi_t, a, mu, k2):
    """F = -grad U, written as in poisson_mode_force: -(sum w) h + sum w mu."""
    d2 = ((h[..., None, :] - mu) ** 2).sum(-1)
    w = phi_t * a * np.exp(-k2 * d2) * 2.0 * k2
    return -(w.sum(-1, keepdims=True) * h - w @ mu)


# ---------------------------------------------------------------------------
def fig_slots_vs_modes():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    # ---- left: the v2.1 slot bank ------------------------------------
    ax = axes[0]
    ax.set_title("Fock v2.1: M slot registers + reverse channel", pad=8)
    sal = [0.98, 0.03, 0.55, 0.01, 0.80, 0.12]
    ang = [20, 110, 200, 300, 60, 160]
    for i, (s, th) in enumerate(zip(sal, ang)):
        y = 8.6 - i * 1.05
        ax.add_patch(FancyBboxPatch((0.6, y - 0.4), 3.2, 0.8,
                                    boxstyle="round,pad=0.02", fc="#EFF6FF", ec=BLUE, lw=1.4))
        ax.text(0.85, y, f"r$_{i+1}$", va="center", fontsize=10)
        c, sn = np.cos(np.radians(th)), np.sin(np.radians(th))
        ax.add_patch(FancyArrowPatch((2.0 - 0.32 * c, y - 0.32 * sn), (2.0 + 0.32 * c, y + 0.32 * sn),
                                     arrowstyle="-|>", mutation_scale=11, color=BLUE, lw=1.6))
        ax.add_patch(Rectangle((2.7, y - 0.15), 0.9, 0.3, fc="white", ec=GREY))
        ax.add_patch(Rectangle((2.7, y - 0.15), 0.9 * s, 0.3, fc=PURPLE, ec="none"))
    ax.text(2.2, 9.45, "one content vector per slot", ha="center", fontsize=9.5)
    ax.text(3.15, 9.2, "salience", ha="center", fontsize=8.5, color=PURPLE)
    notes = [
        "every slot occupied, always (DP1: active 0.9995-1.000)",
        "at most one vector per slot: exclusion",
        "salience = probability old content is kept (DP3)",
        "destruction gate resets content each layer",
        "time axis: depth (2 or 4 layer steps per position)",
    ]
    for j, t in enumerate(notes):
        ax.text(4.2, 8.6 - j * 0.8, "- " + t, fontsize=9.5, va="center")
    ax.add_patch(FancyBboxPatch((0.4, 0.5), 9.2, 1.9, boxstyle="round,pad=0.05",
                                fc="#FEF2F2", ec=RED, lw=1.4))
    ax.text(5.0, 1.95, r"read: $Q_i=\mathrm{RMS}\!\left(\sum_k \mathrm{softmax}_k(q_i\cdot k_k)\,v_k\right)$",
            ha="center", fontsize=11)
    ax.text(5.0, 1.0, "not the gradient of any potential: curl $\\neq 0$  (non-conservative)",
            ha="center", fontsize=10, color=RED)

    # ---- right: PM1 modes --------------------------------------------
    ax = axes[1]
    ax.set_title("PM1: K shared Poisson modes + well force", pad=8)
    x = np.linspace(0.5, 9.5, 600)
    centres = [1.6, 3.6, 5.6, 7.8]
    occ = [0, 4, 1, 2]
    base = 6.0
    U = np.zeros_like(x)
    for c, n in zip(centres, occ):
        U -= 0.55 * n * np.exp(-((x - c) / 0.55) ** 2)
    ax.plot(x, base + U, color="k", lw=1.6)
    for c, n in zip(centres, occ):
        for k in range(n):
            ax.add_patch(plt.Circle((c - 0.28 + 0.28 * (k % 3), base + 0.45 + 0.42 * (k // 3)),
                                    0.13, color=GREEN))
        ax.text(c, base + 2.05, r"$\mu_{%d}$:  $n=%d$" % (centres.index(c) + 1, n), ha="center", fontsize=9.5)
    ax.text(5.0, 9.45, "occupation = number of particles in a mode; deeper well when more occupied",
            ha="center", fontsize=9.5)
    notes = [
        "arrivals: each token adds Poisson($E_v$) particles, $E_v=e^{-\\kappa_v^2\\|h_s-\\mu_v\\|^2}$",
        "survival: each particle lives on with prob. $\\lambda_v$ per token",
        "no cap: bosonic, many tokens share a mode",
        "time axis: tokens (strict past, $s<t$)",
    ]
    for j, t in enumerate(notes):
        ax.text(0.5, 3.35 - j * 0.55, "- " + t, fontsize=9.5, va="center")
    ax.add_patch(FancyBboxPatch((0.4, 0.15), 9.2, 1.05, boxstyle="round,pad=0.05",
                                fc="#F0FDF4", ec=GREEN, lw=1.4))
    ax.text(5.0, 0.85, r"read: $F_t=-\nabla_{h_t}U_t,\;\;U_t(h)=-\sum_v\phi_v(t)\,a_v\,e^{-\kappa_v^2\|h-\mu_v\|^2}$",
            ha="center", fontsize=10.5)
    ax.text(5.0, 0.38, "an exact gradient in the token's own state: curl $=0$  (conservative)",
            ha="center", fontsize=10, color="#15803D")
    fig.tight_layout()
    save(fig, "pm_slots_vs_modes.png")


# ---------------------------------------------------------------------------
MU = np.array([[-1.3, 0.9], [1.3, 0.9], [0.0, -1.3]])     # finance, river, weather
MODE_NAMES = ["finance", "river", "weather"]
K2 = np.array([1.0, 1.0, 1.0])
LAM = np.full(3, 2 ** (-1 / 6))                            # half-life 6 tokens
A = np.full(3, 1.5)                                        # equal depths
BANK = np.array([0.0, 0.7])

CONTEXTS = {
    "finance context": (
        ["investors", "moved", "their", "cash", "to", "the", "bank"],
        np.array([[-1.1, 1.1], [0.3, -0.3], [0.1, 0.2], [-1.5, 0.7], [0.0, 0.1], [0.1, 0.0], BANK])),
    "river context": (
        ["the", "boat", "drifted", "down", "river", "to", "the", "bank"],
        np.array([[0.1, 0.0], [1.0, 1.3], [0.8, 0.4], [0.3, -0.4], [1.4, 0.9], [0.0, 0.1], [0.1, 0.0], BANK])),
}


def damped_path(phi_t, h0, steps=400, dt=0.05, gamma=1.2):
    h, v = h0.copy(), np.zeros(2)
    path = [h.copy()]
    for _ in range(steps):
        v = v + 0.5 * dt * pm_force(h, phi_t, A, MU, K2)
        h = h + dt * v
        v = v + 0.5 * dt * pm_force(h, phi_t, A, MU, K2)
        v = v * np.exp(-gamma * dt)
        path.append(h.copy())
    return np.array(path)


def fig_bank_example():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.4), gridspec_kw={"width_ratios": [1.15, 1.0, 1.2]})
    gx, gy = np.meshgrid(np.linspace(-2.6, 2.6, 220), np.linspace(-2.4, 2.2, 200))
    grid = np.stack([gx, gy], -1)
    for row, (name, (words, H)) in enumerate(CONTEXTS.items()):
        E = overlaps(H, MU, K2)
        phi = occupation(E, LAM)
        T = len(words)
        idx = np.arange(T)

        ax = axes[row, 0]
        bottom = np.zeros(T)
        for v in range(3):
            ax.bar(idx, E[:, v], bottom=bottom, color=MODE_COL[v], label=MODE_NAMES[v], width=0.7)
            bottom += E[:, v]
        ax.set_xticks(idx)
        ax.set_xticklabels(words, rotation=35, ha="right")
        ax.set_ylabel(r"creation rate $E_v(s)$")
        ax.set_title(f"{name}: what each token creates")
        if row == 0:
            ax.legend(fontsize=9, loc="upper right")

        ax = axes[row, 1]
        for v in range(3):
            ax.step(idx, phi[:, v], where="mid", color=MODE_COL[v], lw=2, label=MODE_NAMES[v])
        ax.axvline(T - 1, color=GREY, ls=":", lw=1)
        ax.text(T - 1.05, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1, '"bank"\nfeels this',
                ha="right", fontsize=9, color="k")
        ax.set_xticks(idx)
        ax.set_xticklabels(words, rotation=35, ha="right")
        ax.set_ylabel(r"occupation $\phi_v(t)$ (strict past)")
        ax.set_title("occupations, half-life 6 tokens")

        ax = axes[row, 2]
        pt = phi[T - 1]
        U = pm_potential(grid, pt, A, MU, K2)
        cs = ax.contourf(gx, gy, U, levels=24, cmap="viridis")
        sub = (slice(None, None, 14), slice(None, None, 14))
        Fg = pm_force(grid, pt, A, MU, K2)
        ax.quiver(gx[sub], gy[sub], Fg[..., 0][sub], Fg[..., 1][sub], color="white", alpha=0.75, scale=28)
        path = damped_path(pt, BANK)
        ax.plot(path[:, 0], path[:, 1], color=RED, lw=2.4)
        ax.plot(*BANK, "o", color=RED, ms=8, mec="white")
        ax.text(BANK[0] + 0.12, BANK[1] - 0.32, '"bank" starts here', color="white", fontsize=9)
        for v in range(3):
            ax.plot(*MU[v], "x", color=MODE_COL[v], ms=10, mew=2.5)
            ax.text(MU[v, 0], MU[v, 1] + 0.22, MODE_NAMES[v], color="white", ha="center", fontsize=9.5)
        ax.set_aspect("equal")
        ax.set_title(r'potential $U_t(h)$ felt by "bank", and its path')
        fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.03, label=r"$U_t$")
    fig.tight_layout()
    save(fig, "pm_bank_example.png")


# ---------------------------------------------------------------------------
def fig_poisson_exact():
    T = 70
    E = np.full(T, 0.08)
    E[6:18] = 0.9
    E[38:44] = 1.6
    lam = 2 ** (-1 / 8)
    phi = np.zeros(T + 1)
    for t in range(T):
        phi[t + 1] = lam * phi[t] + E[t]
    n_runs = 20000
    n = np.zeros((n_runs, T + 1), dtype=int)
    for t in range(T):
        n[:, t + 1] = RNG.binomial(n[:, t], lam) + RNG.poisson(E[t], n_runs)
    mean, var = n.mean(0), n.var(0)
    tt = np.arange(T + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for r in range(25):
        ax.step(tt, n[r], where="post", color=GREY, alpha=0.35, lw=0.9)
    ax.plot(tt, mean, color=BLUE, lw=2.4, label="Monte Carlo mean (20,000 runs)")
    ax.plot(tt, phi, color=RED, lw=1.6, ls="--", label=r"$\phi(t)$, the code's formula")
    ax2 = ax.twinx()
    ax2.bar(np.arange(T) + 0.5, E, color=ORANGE, alpha=0.25, width=1.0)
    ax2.set_ylim(0, 6)
    ax2.set_yticks([])
    ax.set_xlabel("token $t$")
    ax.set_ylabel("particles in the mode")
    ax.set_title("integer paths, their mean, and $\\phi$")
    ax.legend(fontsize=9, loc="upper right")
    ax.text(27, 0.6, "shaded: arrivals $E(s)$", color="#B45309", fontsize=9, ha="center")

    ax = axes[1]
    tp = int(np.argmax(phi))
    ks = np.arange(0, n[:, tp].max() + 1)
    freq = np.bincount(n[:, tp], minlength=len(ks)) / n_runs
    ax.bar(ks, freq, color=BLUE, alpha=0.6, label="simulated")
    ax.plot(ks, poisson.pmf(ks, phi[tp]), "o-", color=RED, ms=5, label=f"Poisson, mean {phi[tp]:.2f}")
    ax.axvspan(1.5, ks[-1] + 0.5, color=GREY, alpha=0.12)
    ax.text(ks[-1] * 0.62, freq.max() * 0.85,
            f"{100 * (n[:, tp] >= 2).mean():.1f}% of the mass\nat 2 or more:\na slot cannot hold it",
            fontsize=9, ha="center")
    ax.set_xlabel("particles in the mode at the peak")
    ax.set_ylabel("probability")
    ax.set_title(f"the law at $t={tp}$ is Poisson")
    ax.legend(fontsize=9)

    ax = axes[2]
    ok = mean > 0.05
    ax.plot(tt[ok], (var / np.maximum(mean, 1e-12))[ok], color=BLUE, lw=2, label="variance / mean")
    ax.axhline(1.0, color=RED, ls="--", lw=1.2, label="Poisson: 1")
    rel = np.abs(mean - phi)[ok].max() / phi.max()
    ax.set_ylim(0.8, 1.2)
    ax.set_xlabel("token $t$")
    ax.set_title("Poisson at every token")
    ax.legend(fontsize=9, loc="upper right")
    ax.text(0.03, 0.06, f"max |mean - $\\phi$| / max $\\phi$ = {rel:.4f}", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    save(fig, "pm_poisson_exact.png")


# ---------------------------------------------------------------------------
# A toy reverse channel in d = 2 with M = 3 fixed registers, as ReverseChannel:
# Q(h) = sum_k softmax_k(q . k_k / sqrt(dk)) v_k, q = W_Q h, k_k = W_K r_k,
# v_k = W_V r_k, optionally soft-RMS-normalised Q / sqrt(mean(Q^2) + 1).
R_REG = np.array([[1.2, 0.4], [-0.8, 1.0], [0.1, -1.3]])
WQ = np.array([[1.4, 0.6], [-0.5, 1.1]])
WK = np.array([[0.9, -0.7], [0.8, 1.2]])
WV = np.array([[0.3, -1.3], [1.5, 0.4]])
DK = 2.0


def rc_force(h, kind="rc"):
    q = h @ WQ.T                                       # (..., dk)
    keys = R_REG @ WK.T                                # (M, dk)
    logits = q @ keys.T / np.sqrt(DK)
    logits -= logits.max(-1, keepdims=True)
    al = np.exp(logits)
    al /= al.sum(-1, keepdims=True)
    if kind == "lse":                                  # v_k = W_Q^T k_k: grad of sqrt(dk)*LSE
        vals = keys @ WQ
    else:
        vals = R_REG @ WV.T
    Q = al @ vals
    if kind == "rc_norm":
        Q = Q / np.sqrt((Q ** 2).mean(-1, keepdims=True) + 1.0)
    return Q


def circulation(field, c, r, n=4096):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = c + r * np.stack([np.cos(th), np.sin(th)], -1)
    tang = r * np.stack([-np.sin(th), np.cos(th)], -1)
    return (field(pts) * tang).sum(-1).mean() * 2 * np.pi     # periodic trapezoid


def curl(field, X, Y, eps=1e-5):
    P = np.stack([X, Y], -1)
    dQy_dx = (field(P + [eps, 0])[..., 1] - field(P - [eps, 0])[..., 1]) / (2 * eps)
    dQx_dy = (field(P + [0, eps])[..., 0] - field(P - [0, eps])[..., 0]) / (2 * eps)
    return dQy_dx - dQx_dy


def fig_conservativity():
    phi_t = np.array([2.2, 1.4, 0.9])
    pm = lambda P: pm_force(P, phi_t, A, MU, K2)
    gx, gy = np.meshgrid(np.linspace(-2.6, 2.6, 160), np.linspace(-2.4, 2.2, 150))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))

    ax = axes[0]
    U = pm_potential(np.stack([gx, gy], -1), phi_t, A, MU, K2)
    ax.contourf(gx, gy, U, levels=22, cmap="viridis", alpha=0.9)
    F = pm(np.stack([gx, gy], -1))
    ax.streamplot(gx, gy, F[..., 0], F[..., 1], color="white", density=1.1, linewidth=0.8)
    c_pm = curl(pm, gx, gy)
    ax.set_title(f"PM1: $F=-\\nabla U$ at fixed $\\phi$\nmax |curl| = {np.abs(c_pm).max():.1e} (finite-difference noise)")
    ax.set_aspect("equal")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(-0.4 + 0.7 * np.cos(th), 0.2 + 0.7 * np.sin(th), color=RED, lw=1.8)
    ax.text(-0.4, 1.05, "closed loop", color=RED, ha="center", fontsize=9)

    ax = axes[1]
    c_rc = curl(lambda P: rc_force(P, "rc_norm"), gx, gy)
    lim = np.abs(c_rc).max()
    im = ax.pcolormesh(gx, gy, c_rc, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    Qg = rc_force(np.stack([gx, gy], -1), "rc_norm")
    ax.streamplot(gx, gy, Qg[..., 0], Qg[..., 1], color="k", density=1.0, linewidth=0.7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=r"curl $\partial_xQ_y-\partial_yQ_x$")
    ax.set_title("toy reverse channel (3 registers, RMS-normed)\ncurl $\\neq 0$: work depends on the path")
    ax.set_aspect("equal")

    ax = axes[2]
    radii = np.logspace(-2, 0, 12)
    centres = RNG.uniform([-1.8, -1.5], [1.8, 1.5], size=(40, 2))
    curves = {
        "PM1 well force": (pm, GREEN),
        r"attention as $\nabla\,$LSE ($v_k=W_Q^\top k_k$)": (lambda P: rc_force(P, "lse"), BLUE),
        "reverse channel, raw": (lambda P: rc_force(P, "rc"), ORANGE),
        "reverse channel, RMS-normed": (lambda P: rc_force(P, "rc_norm"), RED),
    }
    for lab, (fld, col) in curves.items():
        circ = [np.median([abs(circulation(fld, c, r)) for c in centres]) for r in radii]
        ax.loglog(radii, np.maximum(circ, 1e-17), "o-", color=col, label=lab, ms=4)
    ax.loglog(radii, 0.3 * radii ** 2, color=GREY, ls=":", lw=1)
    ax.text(0.12, 0.3 * 0.12 ** 2 * 2.2, r"$\propto r^2$ (curl $\times$ area)", color=GREY, fontsize=9)
    ax.set_xlabel("loop radius $r$")
    ax.set_ylabel(r"$|\oint F\cdot dh|$, median over 40 loops")
    ax.set_title("work around closed loops")
    ax.legend(fontsize=8.5, loc="center right")
    fig.tight_layout()
    save(fig, "pm_conservativity.png")


# ---------------------------------------------------------------------------
STEPS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
PM1_03 = [459.92, 238.54, 177.5, 148.12, 129.76, 122.34, 114.04, 109.33, 100.02, 97.23, 91.21, 89.73,
          86.56, 84.45, 81.49, 79.78]
F31 = [463.41, 241.67, 179.65, 151.89, 134.33, 127.73, 122.02, 113.51, 109.51, 106.71, 102.12, 103.0,
       100.06, 99.0, 93.69, 93.57]
G2 = [461.14, 240.45, 176.53, 148.01, 129.63, 123.59, 115.08, 111.61, 103.08, 101.17, 95.34, 93.11,
      90.99, 86.98, 84.71, 82.92]
# pm_ group pre-clip norm on the logged steps where it was the largest group
# (138 of 160), PM1 probe at clip 0.3, from the Cell 6 log.
PM_NORM = [0.3, 0.3, 0.3, 0.3, 0.8, 1.4, 0.7, 0.6, 0.7, 0.7, 0.6, 0.7, 0.7, 0.4, 0.5, 0.5, 0.7, 0.5, 0.6,
           0.5, 0.4, 0.5, 0.6, 0.5, 0.5, 0.6, 0.6, 0.6, 0.4, 0.4, 0.5, 0.5, 0.6, 0.5, 0.5, 0.4, 0.5, 0.4,
           0.5, 0.4, 0.6, 0.4, 0.6, 0.6, 0.5, 0.4, 0.4, 0.6, 0.4, 0.3, 0.6, 0.4, 0.6, 0.4, 0.5, 0.4, 0.7,
           0.5, 0.3, 0.4, 0.6, 0.6, 0.7, 0.5, 0.6, 0.6, 0.9, 0.4, 0.4, 0.7, 0.5, 0.4, 0.8, 0.4, 0.4, 0.6,
           0.4, 0.5, 0.4, 1.0, 0.7, 0.6, 0.5, 0.7, 0.3, 0.4, 0.3, 0.3, 0.6, 0.4, 0.8, 0.4, 0.6, 0.5, 0.4,
           0.6, 0.5, 0.3, 0.3, 0.3, 0.6, 1.0, 0.5, 1.5, 0.4, 0.5, 1.6, 0.5, 0.5, 0.5, 0.7, 0.9, 0.5, 0.3,
           0.8, 0.5, 0.4, 0.4, 0.4, 0.5, 0.4, 0.5, 0.3, 0.5, 0.8, 0.6, 0.6, 0.8, 0.6, 0.6, 0.7, 0.9, 0.4,
           0.3, 1.4, 0.4, 0.6, 0.6]
PM_STEP = [50, 150, 200, 250, 750, 950, 1100, 1150, 1300, 1350] + list(range(1650, 8001, 50))


def fig_probe():
    s = np.array(STEPS)
    f, g, p = map(np.array, (F31, G2, PM1_03))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))
    ax = axes[0]
    ax.plot(s, f, "o-", color=GREY, ms=4, label="F3.1 (conservative-only)")
    ax.plot(s, g, "o-", color=PURPLE, ms=4, label="G2 (slot registers, non-conservative)")
    ax.plot(s, p, "o-", color=GREEN, ms=4, lw=2.2, label="PM1, pm_ clip 0.3 (conservative)")
    ax.set_yscale("log")
    ax.set_ylim(70, 500)
    ax.set_xlabel("step")
    ax.set_ylabel("validation PPL")
    ax.set_title("the 8,000-step probe")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.axhspan(-2, 2, color=GREY, alpha=0.15, label="±2% eval scatter")
    ax.plot(s, 100 * (g / f - 1), "o-", color=PURPLE, ms=4, label="G2 vs F3.1")
    ax.plot(s, 100 * (p / f - 1), "o-", color=GREEN, ms=4, lw=2.2, label="PM1 vs F3.1")
    ax.axhline(-3, color=RED, ls="--", lw=1)
    ax.text(4200, -1.4, "probe gate at 8,000:\n3% below F3.1 (dashed)", color=RED, fontsize=9)
    ax.set_xlabel("step")
    ax.set_ylabel("% against F3.1 at the same step")
    ax.set_title("what each memory buys over the base")
    ax.legend(fontsize=9, loc="lower left")

    ax = axes[2]
    pn = np.array(PM_NORM)
    ax.plot(PM_STEP, pn, ".", color=GREEN, ms=6)
    for thr, col in ((0.3, RED), (1.0, BLUE)):
        late = pn[np.array(PM_STEP) > 2000]
        ax.axhline(thr, color=col, ls="--", lw=1.3)
        ax.text(8150, thr + 0.05, f"clip {thr:g}: {100 * (late > thr).mean():.0f}%", color=col, va="bottom", fontsize=9)
    ax.set_xlim(0, 9600)
    ax.set_xlabel("step")
    ax.set_ylabel("pm_ group pre-clip gradient norm")
    ax.set_title("the 0.3 clip throttled the modes")
    ax.text(200, 1.5, "share of steps 2k-8k\nabove each threshold", fontsize=8.5, color=GREY)
    fig.tight_layout()
    save(fig, "pm_probe.png")


RES = Path(__file__).resolve().parents[2] / "notebooks/conservative_arch/scaleup/results"
_RUN = "cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_{}vplive_xilive_{}L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn"
LOG_F31 = RES / _RUN.format("norc_", "") / "L2_idt4_lr0p0012_norc_vplive_xilive_noattn_32500_result.txt"
LOG_G2 = RES / _RUN.format("", "") / "L2_arm_none_live_grads_output.txt"
DIR_PM1 = RES / _RUN.format("norc_", "pm64_")


def evals_from_printout(path):
    import re
    out = {}
    for line in open(path):
        m = re.search(r"EVAL step ([\d,]+)\s+val_loss=[\d.]+\s+val_ppl=([\d.]+)", line)
        if m:
            out[int(m.group(1).replace(",", ""))] = float(m.group(2))
    return out


def evals_from_jsonl(path):
    import json
    out = {}
    for line in open(path):
        r = json.loads(line)
        if "val_ppl" in r:
            out[int(r["step"])] = float(r["val_ppl"])
    return out


def fig_full_run():
    import json
    f31, g2 = evals_from_printout(LOG_F31), evals_from_printout(LOG_G2)
    pm = evals_from_jsonl(DIR_PM1 / "training_log.jsonl")
    s = np.array(sorted(set(f31) & set(g2) & set(pm)))
    f, g, p = (np.array([d[k] for k in s]) for d in (f31, g2, pm))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))

    ax = axes[0]
    ax.plot(s, f, "-", color=GREY, lw=1.6, label=f"F3.1 (conservative-only), settled {f[-3:].mean():.2f}")
    ax.plot(s, g, "-", color=PURPLE, lw=1.6, label=f"G2 (slot registers), settled {g[-3:].mean():.2f}")
    ax.plot(s, p, "-", color=GREEN, lw=2.2, label=f"PM1, clip 0.3 (conservative), settled {p[-3:].mean():.2f}")
    ax.axvspan(21125, 32500, color=GREY, alpha=0.12)
    ax.text(21600, 135, "WSD decay", color=GREY, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(48, 160)
    ax.set_xlabel("step")
    ax.set_ylabel("validation PPL")
    ax.set_title("the full run, 32,500 steps")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.axhspan(-2, 2, color=GREY, alpha=0.15, label="±2% eval scatter")
    ax.plot(s, 100 * (g / f - 1), "-", color=PURPLE, lw=1.6, label="G2 vs F3.1")
    ax.plot(s, 100 * (p / f - 1), "-", color=GREEN, lw=2.2, label="PM1 vs F3.1")
    ax.axvspan(21125, 32500, color=GREY, alpha=0.12)
    ax.set_xlabel("step")
    ax.set_ylabel("% against F3.1 at the same step")
    ax.set_title("PM1's lead peaks by step 9,000; G2's lasts longer")
    ax.legend(fontsize=9, loc="upper right")

    ax = axes[2]
    loc = json.load(open(DIR_PM1 / "pm1_refinement_localization.json"))["refinement"]
    cap = json.load(open(DIR_PM1 / "pm1_refinement_localization_cap0p3.json"))["refinement"]["1.0"]
    pts = [(f"wells x{float(a):g}", r) for a, r in loc.items()] + [("capped at 0.3\n(post hoc)", cap)]
    for name, r in pts:
        x, y = r["2"], 100 * (r["4"] / r["2"] - 1)
        col = ORANGE if "capped" in name else GREEN
        ax.plot(x, y, "o", color=col, ms=8)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(7, 4), fontsize=8.5, color=col)
    ax.plot(56.30, 374, "s", color=GREY, ms=8)
    ax.annotate("F3.1", (56.30, 374), textcoords="offset points", xytext=(7, -12), fontsize=8.5, color=GREY)
    ax.set_yscale("log")
    ax.set_xlim(48, 140)
    ax.set_xlabel("PPL at the trained step count, N = 2")
    ax.set_ylabel("Gate 3 at N = 4: % above N = 2")
    ax.set_title("the wells trade perplexity for refinement")
    ax.text(0.97, 0.96, "PM1's trained weights, wells scaled or capped\nafter training; not trained arms",
            transform=ax.transAxes, fontsize=8, color=GREY, ha="right", va="top")
    fig.tight_layout()
    save(fig, "pm_full_run.png")


if __name__ == "__main__":
    fig_slots_vs_modes()
    fig_bank_example()
    fig_poisson_exact()
    fig_conservativity()
    fig_probe()
    fig_full_run()
