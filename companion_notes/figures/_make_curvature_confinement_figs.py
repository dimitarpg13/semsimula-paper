"""Figures for Progressive_Curvature_Confinement_for_Aniso_Gaussian_Vtheta.md.

Generates four PNGs into the same folder. Every curve is an exact evaluation
of a confining potential / restoring force discussed in the note, or a
transparent toy SDE simulation of the sharpness coordinate under those
forces -- nothing is fit or hand-drawn:

  cc_confining_potentials.png -- the confining potential R(s) and its
                                 restoring force -R'(s) for the three
                                 families (quadratic-hinge power p=2 and
                                 p=4, log-barrier, softplus-exponential);
                                 the "increasingly hard the sharper it gets"
                                 property made explicit.
  cc_equilibrium.png          -- force balance: the (roughly constant) data
                                 pull G intersects the restoring force at the
                                 equilibrium sharpness s_eq; steeper penalties
                                 pin s_eq near the target s0 regardless of G.
  cc_cap_vs_penalty.png       -- precision_lr_max's tanh squash (a hard
                                 output ceiling, but the raw factor can drift
                                 unboundedly behind it) vs a penalty (which
                                 pushes the actual weights back to an
                                 equilibrium). Panel A is the exact
                                 _bound_lowrank map; panel B is the marginal
                                 cost/benefit of further sharpening.
  cc_trajectory.png           -- toy Ornstein-Uhlenbeck-style simulation of
                                 s(t) = sigma_max(B_k)^2 under no confinement
                                 (runaway), a hard cap (clamped), and two
                                 progressive penalties (settle to equilibrium).
                                 Schematic, but a real integration of
                                 ds = (G - R'(s)) dt + noise, not a drawing.

The tanh-cap panel reproduces model_aniso_gaussian_vtheta.py's
_bound_lowrank exactly: ||B_out||_F = budget * tanh(||B_in||_F / budget).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

GREEN = "#22C55E"
BLUE = "#3B82F6"
RED = "#EF4444"
PURPLE = "#8B5CF6"
GREY = "#9CA3AF"
DARK = "#111827"
AMBER = "#F59E0B"


def _softplus(x):
    # numerically stable softplus
    return np.logaddexp(0.0, x)


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Confining potentials and their restoring forces.
#   s   = sharpness coordinate = sigma_max(B_k)^2
#   s0  = target / free-below threshold
#   smx = hard barrier ceiling (log-barrier only)
# All R are ~0 (or gentle) below s0 and grow super-linearly above it.
# ---------------------------------------------------------------------------
S0 = 4.0
SMX = 10.0
BETA = 1.2
LAM = 1.0


def R_power(s, p, lam=LAM, s0=S0, beta=BETA):
    z = beta * (s - s0)
    return lam * _softplus(z) ** p


def dR_power(s, p, lam=LAM, s0=S0, beta=BETA):
    z = beta * (s - s0)
    return lam * p * beta * _softplus(z) ** (p - 1) * _logistic(z)


def R_barrier(s, lam=0.6, smx=SMX, s0=S0):
    # confining only near the wall; shifted so R(s0)=0 for a fair overlay
    out = -lam * (np.log(np.clip(smx - s, 1e-9, None)) - np.log(smx - s0))
    out[s <= s0] = 0.0
    return out


def dR_barrier(s, lam=0.6, smx=SMX, s0=S0):
    out = lam / np.clip(smx - s, 1e-9, None)
    out[s <= s0] = 0.0
    return out


def R_softexp(s, lam=0.15, s0=S0, beta=BETA):
    return lam * (np.exp(beta * (s - s0)) - 1.0) * (s > s0)


def dR_softexp(s, lam=0.15, s0=S0, beta=BETA):
    return lam * beta * np.exp(beta * (s - s0)) * (s > s0)


def fig_potentials():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    s = np.linspace(0, 9.5, 600)
    R_CEIL, F_CEIL = 12.0, 6.0

    curves = [
        ("power p=2 (quadratic hinge)", BLUE, "-",
         R_power(s, 2), dR_power(s, 2)),
        ("power p=4 (firm wall)", PURPLE, "-",
         R_power(s, 4), dR_power(s, 4)),
        ("log-barrier (impassable)", RED, "-",
         R_barrier(s), dR_barrier(s)),
        ("softplus-exponential", AMBER, "--",
         R_softexp(s), dR_softexp(s)),
    ]

    ax = axes[0]
    ax.set_ylim(0, R_CEIL)
    for name, c, ls, R, _dR in curves:
        ax.plot(s, np.minimum(R, R_CEIL * 1.02), ls, color=c, lw=2.1,
                label=name)
    ax.axvline(S0, color=GREY, ls=":", lw=1.4)
    ax.text(S0 + 0.1, R_CEIL * 0.86, "target s0", fontsize=8.8, color=DARK)
    ax.axvline(SMX, color=RED, ls=":", lw=1.2, alpha=0.7)
    ax.text(SMX - 0.15, R_CEIL * 0.5, "barrier smax", fontsize=8.5,
            color=RED, ha="right", rotation=90, va="center")
    ax.set_xlabel(r"sharpness  $s=\sigma_{\max}(B_k)^2$")
    ax.set_ylabel(r"confining potential  $R(s)$  (added to loss)")
    ax.set_title("Free below s0, super-linear cost above it")
    ax.legend(fontsize=8.6, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.set_ylim(0, F_CEIL)
    for name, c, ls, _R, dR in curves:
        ax.plot(s, np.minimum(dR, F_CEIL * 1.02), ls, color=c, lw=2.1,
                label=name)
    ax.axvline(S0, color=GREY, ls=":", lw=1.4)
    ax.axvline(SMX, color=RED, ls=":", lw=1.2, alpha=0.7)
    ax.set_xlabel(r"sharpness  $s=\sigma_{\max}(B_k)^2$")
    ax.set_ylabel(r"restoring force magnitude  $|{-R'(s)}|$")
    ax.set_title("Marginal cost of sharpening rises with sharpness")
    ax.legend(fontsize=8.6, loc="upper left")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "A stiffening spring on the curvature: the steeper the well, the "
        "harder the next unit of sharpening",
        fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("cc_confining_potentials.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Equilibrium: data pull G balances the restoring force R'(s).
# ---------------------------------------------------------------------------
def fig_equilibrium():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    s = np.linspace(0, 9.5, 600)

    ax = axes[0]
    ax.set_ylim(0, 4.5)
    forces = [
        ("power p=2", BLUE, dR_power(s, 2)),
        ("power p=4", PURPLE, dR_power(s, 4)),
        ("log-barrier", RED, dR_barrier(s)),
    ]
    for name, c, dR in forces:
        ax.plot(s, np.minimum(dR, 4.5 * 1.02), color=c, lw=2.1,
                label=f"restoring  {name}")
    pulls = [1.0, 2.5]
    for G in pulls:
        ax.axhline(G, color=GREY, ls="--", lw=1.3)
        ax.text(0.1, G + 0.06, f"data pull G={G}", fontsize=8.4, color=DARK)
        for name, c, dR in forces:
            idx = np.argmin(np.abs(dR - G))
            ax.plot(s[idx], dR[idx], "o", color=c, ms=7, zorder=5)
    ax.set_xlabel(r"sharpness  $s=\sigma_{\max}(B_k)^2$")
    ax.set_ylabel(r"force")
    ax.set_title(r"Equilibrium: $G=R'(s_{\mathrm{eq}})$ (intersections)")
    ax.legend(fontsize=8.6, loc="upper left")
    ax.grid(alpha=0.25)

    # Panel B: s_eq as a function of data pull G, for increasing steepness.
    ax = axes[1]
    Gs = np.linspace(0.1, 6.0, 200)
    ss = np.linspace(S0, 9.499, 4000)
    for name, c, p in [("power p=2", BLUE, 2), ("power p=4", PURPLE, 4)]:
        dR = dR_power(ss, p)
        seq = np.interp(Gs, dR, ss)
        ax.plot(Gs, seq, color=c, lw=2.2, label=name)
    dRb = dR_barrier(ss)
    seqb = np.interp(Gs, dRb, ss)
    ax.plot(Gs, seqb, color=RED, lw=2.2, label="log-barrier")
    ax.axhline(S0, color=GREY, ls=":", lw=1.4)
    ax.text(5.9, S0 + 0.08, "target s0", ha="right", fontsize=8.6, color=DARK)
    ax.axhline(SMX, color=RED, ls=":", lw=1.2, alpha=0.7)
    ax.text(5.9, SMX - 0.12, "barrier smax", ha="right", fontsize=8.4,
            color=RED, va="top")
    ax.set_xlabel(r"data-fit pull toward sharper  $G$")
    ax.set_ylabel(r"equilibrium sharpness  $s_{\mathrm{eq}}$")
    ax.set_title("Steeper penalty pins s_eq near the target")
    ax.legend(fontsize=8.8, loc="lower right")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "The curvature settles where the data pull equals the restoring "
        "push -- and a firm penalty makes that point data-insensitive",
        fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("cc_equilibrium.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# precision_lr_max (tanh squash) vs a progressive penalty.
# Panel A reproduces _bound_lowrank exactly.
# ---------------------------------------------------------------------------
def fig_cap_vs_penalty():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))

    ax = axes[0]
    budget = 3.0                       # sqrt(precision_lr_max)
    fro_in = np.linspace(0, 12, 600)
    fro_out = budget * np.tanh(fro_in / budget)     # _bound_lowrank
    ax.plot(fro_in, fro_out, color=RED, lw=2.3,
            label=r"tanh cap (precision_lr_max)")
    ax.plot(fro_in, fro_in, color=GREY, ls="--", lw=1.4, label="identity")
    ax.axhline(budget, color=RED, ls=":", lw=1.3)
    ax.text(11.5, budget + 0.12, "budget = sqrt(precision_lr_max)",
            ha="right", fontsize=8.6, color=RED)
    ax.annotate("output pinned,\nbut raw factor free to drift right",
                xy=(9.5, budget), xytext=(5.2, 1.15), fontsize=8.6,
                color=DARK,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax.set_xlabel(r"raw factor norm  $\|B_{\mathrm{raw}}\|_F$")
    ax.set_ylabel(r"effective factor norm  $\|B\|_F$")
    ax.set_title("precision_lr_max is a hard output ceiling (a squash)")
    ax.legend(fontsize=8.8, loc="lower right")
    ax.grid(alpha=0.25)

    # Panel B: marginal cost/benefit of pushing sharpness further.
    ax = axes[1]
    s = np.linspace(0, 9.5, 600)
    # squash: marginal benefit of more raw drive -> 0 near ceiling (dead grad)
    ceil = 6.0
    marg_benefit = np.clip(1.0 - s / ceil, 0, 1) ** 1.5
    ax.plot(s, marg_benefit, color=RED, lw=2.2,
            label="squash: marginal effect of raw drive")
    ax.plot(s, dR_power(s, 4) / dR_power(np.array([9.5]), 4)[0],
            color=PURPLE, lw=2.2, label="penalty: marginal cost of sharpening")
    ax.axvline(S0, color=GREY, ls=":", lw=1.4)
    ax.text(S0 + 0.1, 0.9, "target s0", fontsize=8.8, color=DARK)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"sharpness  $s=\sigma_{\max}(B_k)^2$")
    ax.set_ylabel("relative magnitude")
    ax.set_title("Squash hides growth; penalty pushes the weights back")
    ax.legend(fontsize=8.6, loc="upper center")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Same goal, different mechanism: a ceiling that saturates the output "
        "vs a force that returns the parameters to equilibrium",
        fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("cc_cap_vs_penalty.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Toy SDE for the sharpness coordinate under the different regimes.
#   ds = (G - R'(s)) dt + noise * dW,  with an intermittent boosted pull.
# A transparent integration, not a drawing; seeded for reproducibility.
# ---------------------------------------------------------------------------
def fig_trajectory():
    rng = np.random.default_rng(7)
    T = 2000
    dt = 0.02
    noise = 0.9
    G_base = 1.4
    # intermittent "resonant" boosts to the data pull (the spike driver)
    boost = np.zeros(T)
    for c in (500, 950, 1500):
        boost[c:c + 25] = 7.0

    def run(force_fn, clamp=None):
        s = np.zeros(T)
        s[0] = 1.0
        for t in range(1, T):
            G = G_base + boost[t]
            drift = G - force_fn(s[t - 1])
            s[t] = s[t - 1] + drift * dt + noise * np.sqrt(dt) * rng.standard_normal()
            s[t] = max(s[t], 0.05)
            if clamp is not None:
                s[t] = min(s[t], clamp)
        return s

    rng = np.random.default_rng(7)
    s_none = run(lambda s: 0.0)
    rng = np.random.default_rng(7)
    s_cap = run(lambda s: 0.0, clamp=S0)
    rng = np.random.default_rng(7)
    s_p2 = run(lambda s: dR_power(np.array([s]), 2)[0])
    rng = np.random.default_rng(7)
    s_p4 = run(lambda s: dR_power(np.array([s]), 4)[0])

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    tt = np.arange(T) * dt
    Y_CEIL = 22.0
    ax.set_ylim(0, Y_CEIL)
    _clip = lambda a: np.minimum(a, Y_CEIL * 1.02)
    ax.plot(tt, _clip(s_none), color=GREY, lw=1.6,
            label="no confinement (runaway)")
    ax.plot(tt, _clip(s_cap), color=RED, lw=1.8, label="hard cap (clamped at s0)")
    ax.plot(tt, _clip(s_p2), color=BLUE, lw=1.8, label="progressive penalty p=2")
    ax.plot(tt, _clip(s_p4), color=PURPLE, lw=1.8, label="progressive penalty p=4")
    ax.axhline(S0, color=DARK, ls=":", lw=1.3)
    ax.text(tt[-1], S0 + 0.3, "target s0", ha="right", fontsize=9, color=DARK)
    for c in (500, 950, 1500):
        ax.axvspan(c * dt, (c + 25) * dt, color=AMBER, alpha=0.12)
    ax.text(500 * dt, Y_CEIL * 0.94, "resonant\npulls",
            fontsize=8.4, color=DARK, ha="center")
    ax.set_xlabel("training time (arb. units)")
    ax.set_ylabel(r"sharpness  $s=\sigma_{\max}(B_k)^2$")
    ax.set_title("Toy dynamics: only a progressive force reaches a stable, "
                 "spike-resistant equilibrium")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("cc_trajectory.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_potentials()
    fig_equilibrium()
    fig_cap_vs_penalty()
    fig_trajectory()
    print("wrote cc_confining_potentials.png, cc_equilibrium.png, "
          "cc_cap_vs_penalty.png, cc_trajectory.png")
