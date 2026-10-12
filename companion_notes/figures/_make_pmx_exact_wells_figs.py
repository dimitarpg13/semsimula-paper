"""Figures for Exact_Integration_of_Poisson_Mode_Wells_PMX.md.

PNGs into figures/pmx_exact_wells/. The theory panels are computed with the
integrator's own functions (parf/cfc_baoab.py: damped_mode_coefficients,
_cholqr_basis); the measured panels read the run's recorded numbers (protocol
SS5.15) and the trained spring spectrum (debug/pmx_spring_spectrum.py).

  pmx_well_curvature.png   -- one Gaussian well along its radial direction:
                              potential, force and the two Hessian eigenvalues
                              (transverse w, radial w(1 - 2 kappa^2 r^2)), the
                              inflection where the radial one turns negative
  pmx_exact_vs_explicit.png -- why exact: a stable mode at omega dt = 2.1 and
                              an unstable one at rate dt = 0.97 under the exact
                              damped flow and under the explicit split step;
                              the coefficients E12, Q across omega0^2 = 0
  pmx_operator.png          -- the spring K = alpha I + U0 k0 U0^T - R dW R^T on
                              R^d, and the route that diagonalises it
  pmx_step.png              -- one layer step under PM1, SR2 and PMX
  pmx_hybrid.png            -- the 32-wide hybrid: force missed against the wells
                              kept in the rank part; the kick share; second-order
                              consistency with the explicit wells
  pmx_spectrum.png          -- the trained PMX spring per token: signed omega dt
                              of the 32 modes and of the complement, per layer

Run:  MPLBACKEND=Agg python3 _make_pmx_exact_wells_figs.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
GREEN, BLUE, RED, PURPLE, GREY, ORANGE = "#22C55E", "#3B82F6", "#EF4444", "#8B5CF6", "#9CA3AF", "#F59E0B"
OUT = Path(__file__).parent / "pmx_exact_wells"
OUT.mkdir(exist_ok=True)
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "notebooks/conservative_arch/parf"))
from cfc_baoab import damped_mode_coefficients  # noqa: E402
RES = REPO / "notebooks/conservative_arch/scaleup/results"
PMX_RUN = ("cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64_pmx16_sr2_"
           "L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn")


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- 1. one well
def fig_well():
    kappa, depth = 1.0, 1.0
    r = np.linspace(0.0, 2.6, 400)
    E = np.exp(-kappa ** 2 * r ** 2)
    rstar = 1.0 / (np.sqrt(2.0) * kappa)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for a, col, lab in ((depth, BLUE, "attractive, $a > 0$"), (-depth, RED, "repulsive, $a < 0$")):
        ax[0].plot(r, -a * E, color=col, lw=2, label=f"potential $U$, {lab}")
        ax[0].plot(r, -2 * kappa ** 2 * a * E * r, color=col, lw=1.3, ls="--", label=f"radial force, {lab}")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].axvline(rstar, color=GREY, ls=":")
    ax[0].text(rstar + 0.04, 0.85, "inflection\n$r^\\ast = 1/(\\sqrt{2}\\,\\kappa)$", fontsize=9)
    ax[0].set_xlabel("distance from the mode centre, $r = \\Vert h - \\mu \\Vert$")
    ax[0].set_title("One well at fixed occupation ($\\phi a = \\pm 1$, $\\kappa = 1$)")
    ax[0].legend(fontsize=8, loc="lower right")
    w = 2 * kappa ** 2 * depth * E                                   # the weight w_v for a > 0
    ax[1].plot(r, w, color=GREEN, lw=2, label="transverse eigenvalue: $w$ (from $\\alpha$)")
    ax[1].plot(r, w * (1 - 2 * kappa ** 2 * r ** 2), color=PURPLE, lw=2,
               label="radial eigenvalue: $w\\,(1 - 2\\kappa^2 r^2)$")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].axvspan(rstar, r[-1], color=RED, alpha=0.08)
    ax[1].text(1.3, 0.9, "radial direction unstable:\nhyperbolic, $\\omega^2 < 0$", color=RED, fontsize=9)
    ax[1].axvline(rstar, color=GREY, ls=":")
    ax[1].set_xlabel("$r$")
    ax[1].set_ylabel("curvature of $U$ (attractive well)")
    ax[1].set_title("Hessian $H = \\alpha I - \\sum_v dW_v\\, r_v r_v^{\\top}$, one well")
    ax[1].legend(fontsize=8)
    save(fig, "pmx_well_curvature.png")


# ------------------------------------------------- 2. exact against explicit
def damped_exact_step(x, v, w2, g, dt):
    """Exact flow of x'' + g x' + w2 x = 0 over dt (unit mass), via the integrator's coefficients."""
    E12, E22, P, Q = (float(c) for c in damped_mode_coefficients(torch.tensor([w2]), g, dt))
    # damped_mode_coefficients works in displacement from the step start with a constant force a = -w2 x0
    a = -w2 * x
    return x + E12 * v + Q * a, E22 * v + P * a


def explicit_split_step(x, v, w2, g, dt):
    """Explicit kick-drift-kick with exact friction in the middle (the explicit wells' treatment)."""
    v = v - 0.5 * dt * w2 * x
    x = x + 0.5 * dt * v
    v = v * np.exp(-g * dt)
    x = x + 0.5 * dt * v
    v = v - 0.5 * dt * w2 * x
    return x, v


def true_solution(t, w2, g, x0=1.0, v0=0.0):
    E12, E22, P, Q = (float(c) for c in damped_mode_coefficients(torch.tensor([w2]), g, t))
    a = -w2 * x0
    return x0 + E12 * v0 + Q * a


def fig_exact_vs_explicit():
    g, dt, n = 0.1, 1.0, 12
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    for k, (w2, title) in enumerate(((2.1 ** 2, "stable mode, $\\omega\\,\\Delta t = 2.1$"),
                                      (-(0.97 ** 2), "unstable mode, rate $\\times\\,\\Delta t = 0.97$"))):
        xe, ve, xs, vs = 1.0, 0.0, 1.0, 0.0
        te, Xe, Xs = [0], [xe], [xs]
        for i in range(n):
            xe, ve = damped_exact_step(xe, ve, w2, g, dt)
            xs, vs = explicit_split_step(xs, vs, w2, g, dt)
            te.append((i + 1) * dt); Xe.append(xe); Xs.append(xs)
        tt = np.linspace(0, n * dt, 600)
        ax[k].plot(tt, [true_solution(t, w2, g) for t in tt], color=GREY, lw=3, alpha=0.6, label="the ODE's solution")
        ax[k].plot(te, Xe, "o", color=BLUE, label="exact damped flow, per step")
        ax[k].plot(te, Xs, "s--", color=RED, ms=4, label="explicit split step")
        ax[k].set_yscale("symlog", linthresh=1.0)
        ax[k].set_xlabel("time (steps of $\\Delta t$)")
        ax[k].set_title(title)
        ax[k].legend(fontsize=8, loc="upper left")
    ax[0].set_ylabel("mode coordinate $x$ (symlog)")
    w2 = np.linspace(-1.0, 3.0, 801)
    c = damped_mode_coefficients(torch.tensor(w2), 0.1, 2.0)
    ax[2].plot(w2, c[0].numpy(), color=BLUE, lw=2, label="$E_{12}$: position from initial velocity")
    ax[2].plot(w2, c[3].numpy(), color=ORANGE, lw=2, label="$Q$: position from a unit force")
    ax[2].plot(w2, c[1].numpy(), color=PURPLE, lw=1.5, ls="--", label="$E_{22}$: velocity from velocity")
    ax[2].axvline(0, color="k", lw=0.6)
    ax[2].axvspan(-1.0, 0.0, color=RED, alpha=0.06)
    ax[2].text(-0.95, 2.45, "hyperbolic\n($\\omega_0^2 < 0$)", color=RED, fontsize=9)
    ax[2].set_xlabel("$\\omega_0^2$ (with $t = 2$, $\\gamma = 0.1$: PMX's half step)")
    ax[2].set_title("Coefficients, continuous through $\\omega_0^2 = 0$")
    ax[2].legend(fontsize=8, loc="upper right")
    save(fig, "pmx_exact_vs_explicit.png")


# -------------------------------------------------------------- 3. operator
def box(ax, xy, w, h, text, col, fs=9):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", fc=col, ec="k", lw=0.8, alpha=0.9))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.2))


def fig_operator():
    fig, ax = plt.subplots(2, 1, figsize=(13, 7.2), gridspec_kw={"height_ratios": [1, 1.25]})
    a0 = ax[0]
    a0.set_xlim(0, 384); a0.set_ylim(0, 1); a0.axis("off")
    a0.add_patch(Rectangle((0, 0.35), 50, 0.4, fc=BLUE, ec="k"))
    a0.add_patch(Rectangle((50, 0.35), 50, 0.4, fc=GREEN, ec="k"))
    a0.add_patch(Rectangle((100, 0.35), 284, 0.4, fc="#E5E7EB", ec="k"))
    a0.text(25, 0.55, "16 V$_\\theta$ modes", ha="center", fontsize=9, color="white")
    a0.text(75, 0.55, "16 well directions", ha="center", fontsize=9)
    a0.text(242, 0.55, "complement, 352 directions: stiffness $\\alpha$\n(isotropic, every well), friction $\\gamma$", ha="center", fontsize=10)
    a0.annotate("", xy=(100, 0.25), xytext=(0, 0.25), arrowprops=dict(arrowstyle="<->", lw=1))
    a0.text(0, 0.05, "span$[U_0, R]$, 32 wide: diagonalised, stiffness $\\kappa_i + \\alpha$ (either sign)", ha="left", fontsize=9)
    a0.text(384, 0.05, "(widths not to scale)", ha="right", fontsize=8, color=GREY)
    a0.set_title("$K = \\alpha I + U_0\\,\\mathrm{diag}(\\kappa_0)\\,U_0^{\\top} - R\\,\\mathrm{diag}(dW)\\,R^{\\top}$ on $\\mathbb{R}^{384}$, per token")
    a1 = ax[1]
    a1.set_xlim(0, 13); a1.set_ylim(0, 3.2); a1.axis("off")
    steps = [("$R = [r_v]$\n$d \\times 16$", GREEN),
             ("$R_\\perp = (I - U_0 U_0^{\\top}) R$\nremove span$(U_0)$", "#D1FAE5"),
             ("Cholesky QR, twice\n$Q_r = R_\\perp L^{-\\top}$", "#D1FAE5"),
             ("$Q = [U_0, Q_r]$\nrestriction\n$M = Q^{\\top} K_{\\mathrm{rk}} Q$\n$32 \\times 32$", "#FEF3C7"),
             ("shift $M + cI$\n$c \\geq \\rho(M)$\nJacobi SVD", "#FEF3C7"),
             ("$\\kappa = \\lambda - c$\n$U = Q\\,Y$", BLUE)]
    xs = np.linspace(0.15, 11.0, len(steps))
    for i, (t, col) in enumerate(steps):
        box(a1, (xs[i], 1.0), 1.75, 1.5, t, col, fs=8.5)
        if i:
            arrow(a1, (xs[i - 1] + 1.78, 1.75), (xs[i] - 0.03, 1.75))
    a1.text(6.5, 0.35, "$U_0$ (V$_\\theta$'s retained modes) is already orthonormal from SR2's solve: only the 16 well directions are orthonormalised.\n"
            "Inert directions ($\\Vert r_\\perp \\Vert^2 \\leq$ floor) get a large distinct diagonal, so their columns come out exactly zero.",
            ha="center", fontsize=8.5)
    a1.set_title("Diagonalising the indefinite spring (indefinite_lowrank_modes_chol)")
    save(fig, "pmx_operator.png")


# ------------------------------------------------------------------ 4. step
def fig_step():
    fig, ax = plt.subplots(figsize=(13.8, 5.6))
    ax.set_xlim(0, 13.8); ax.set_ylim(0, 6.2); ax.axis("off")
    rows = [("PM1\n(explicit wells)", [("A  half\nexact on V$_\\theta$ modes,\nundamped", "#DBEAFE"),
                                       ("B  kick\nV$_\\theta$ rest + V$_\\phi$\n+ ALL wells", "#FEE2E2"),
                                       ("O  friction\nall of $\\mathbb{R}^d$", "#F3F4F6"),
                                       ("A  half", "#DBEAFE")]),
            ("SR2", [("A  half\nexact damped flow\non V$_\\theta$ modes", "#BFDBFE"),
                     ("B  kick\nV$_\\theta$ rest + V$_\\phi$", "#FEE2E2"),
                     ("O  friction\ncomplement only", "#F3F4F6"),
                     ("A  half", "#BFDBFE")]),
            ("PMX", [("A  half\nexact damped flow on ALL of $\\mathbb{R}^d$:\nV$_\\theta$ modes + wells' quadratic", "#BBF7D0"),
                     ("B  kick\nthe remainder only", "#FEE2E2"),
                     ("(no O-step)", "white"),
                     ("A  half, same\nfrozen spring", "#BBF7D0")])]
    for k, (name, parts) in enumerate(rows):
        y = 4.6 - 2.0 * k
        ax.text(0.1, y + 0.55, name, fontsize=11, fontweight="bold", va="center")
        x = 1.7
        widths = [2.9, 2.4, 2.0, 2.4]
        for (t, col), wdt in zip(parts, widths):
            box(ax, (x, y), wdt, 1.1, t, col, fs=8.5)
            x += wdt + 0.25
        box(ax, (x, y), 0.9, 1.1, "LN", "#E9D5FF", fs=9)
    ax.text(6.9, 0.15, "The spring is frozen at the step's start $h_0$; the kick uses the true forces at $h_{\\mathrm{mid}}$ minus what the flow carries. "
            "PMX's flow carries friction on every direction, so no separate O-step remains.", ha="center", fontsize=9)
    ax.set_title("One layer step: what is integrated exactly, and what is kicked")
    save(fig, "pmx_step.png")


# ---------------------------------------------------------------- 5. hybrid
def fig_hybrid():
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    M = np.array([0, 8, 16, 24, 64]); miss = np.array([0.272, 0.182, 0.132, 0.093, 0.047])
    ax[0].plot(M, miss, "o-", color=GREEN, lw=2)
    ax[0].plot([16], [0.291], "x", color=RED, ms=9, mew=2)
    ax[0].annotate("top 16 wells whole,\nno $\\alpha$ for the rest: 0.291", (16, 0.291), (27, 0.255), fontsize=8,
                   arrowprops=dict(arrowstyle="->", lw=0.8))
    ax[0].axvline(16, color=BLUE, ls="--")
    ax[0].text(17, 0.06, "PMX: 16 + 16 = 32,\ncuSOLVER's batched\nJacobi limit", color=BLUE, fontsize=8.5)
    ax[0].set_xlabel("wells whose rank part enters the flow (all keep force and $\\alpha$)")
    ax[0].set_ylabel("force missed at the kick point / force")
    ax[0].set_title("Phase 0: what the hybrid gives up (layer 1)")
    labels = ["layer 0", "layer 1"]; xx = np.arange(2); w = 0.26
    for j, (vals, col, lab) in enumerate((([0.586, 0.938], RED, "PM1, explicit wells"),
                                           ([0.319, 0.119], GREEN, "PMX design, on PM1's weights"),
                                           ([0.319, 0.288], BLUE, "PMX trained (FLOW-R)"))):
        ax[1].bar(xx + (j - 1) * w, vals, w, color=col, label=lab)
    ax[1].axhline(0.3, color="k", ls=":", lw=1)
    ax[1].text(-0.45, 0.31, "CX6 threshold 0.3", fontsize=8)
    ax[1].set_xticks(xx, labels); ax[1].set_ylabel("kick share, median")
    ax[1].set_title("The explicit part of the step")
    ax[1].set_ylim(0, 1.25)
    ax[1].legend(fontsize=8, loc="upper left")
    dts = np.array([0.5, 0.25]); dev = np.array([1.85e-2, 4.50e-3])
    ax[2].loglog(dts, dev, "o-", color=PURPLE, lw=2, label="PMX against explicit wells")
    ref = dev[0] * (dts / dts[0]) ** 2
    ax[2].loglog(dts, ref, "--", color=GREY, label="slope 2")
    ax[2].set_xlabel("$\\Delta t$"); ax[2].set_ylabel("$|\\Delta h|$ / step")
    ax[2].set_title("Same ODE: the difference falls at second order")
    ax[2].legend(fontsize=8)
    save(fig, "pmx_hybrid.png")


# -------------------------------------------------------------- 6. spectrum
def fig_spectrum():
    f = RES / PMX_RUN / "pmx_spring_spectrum.npz"
    if not f.exists():
        print("skip pmx_spectrum.png: run debug/pmx_spring_spectrum.py first")
        return
    d = np.load(f)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), sharey=False)
    bins = np.linspace(-3.5, 6.0, 96)
    for li in (0, 1):
        m = d[f"l{li}_modes"].ravel(); m = m[np.isfinite(m)]
        c = d[f"l{li}_comp"].ravel()
        ax[li].hist(m, bins=bins, color=GREEN, alpha=0.75, density=True, label="the 32 modes, $\\kappa_i + \\alpha$")
        ax[li].hist(c, bins=bins, color=BLUE, alpha=0.6, density=True, label="complement, $\\alpha$")
        ax[li].axvline(0, color="k", lw=0.8)
        for s in (-2, 2):
            ax[li].axvline(s, color=RED, ls=":", lw=1)
        ax[li].set_xlabel("signed $\\omega\\,\\Delta t$  (negative: hyperbolic, rate $\\times\\,\\Delta t$)")
        ax[li].set_title(f"Trained PMX, layer {li} (1 x 256 tokens)")
        ax[li].legend(fontsize=8)
    ax[0].text(-3.3, ax[0].get_ylim()[1] * 0.8, "weak, repulsive wells:\n$\\alpha < 0$ for every token", fontsize=8.5)
    ax[1].text(2.3, ax[1].get_ylim()[1] * 0.8, "explicit wall\n$\\omega\\Delta t = 2$", color=RED, fontsize=8.5)
    save(fig, "pmx_spectrum.png")


if __name__ == "__main__":
    fig_well(); fig_exact_vs_explicit(); fig_operator(); fig_step(); fig_hybrid(); fig_spectrum()
