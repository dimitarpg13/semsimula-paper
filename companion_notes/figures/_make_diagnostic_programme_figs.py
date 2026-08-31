"""Figures for Diagnostic_Programme_in_CfC_BAOAB_Integrator.md.

Generates five PNGs into the same folder, all from either closed-form
evaluation of the anisotropic-Gaussian well or the real replay/Phase-0
numbers recorded in the companion note's diagnostic programme:

  dp_well_landscape.png       -- isotropic vs low-rank-sharpened well: how a
                                 single off-diagonal factor B_k B_k^T turns a
                                 round basin into a razor ridge (contours +
                                 gradient-magnitude heatmap).
  dp_force_profile.png        -- phi(t) = lambda*t*exp(-lambda t^2/2), the
                                 along-sharp-direction force of one well, for a
                                 sweep of curvatures lambda = sigma_max(B_k)^2;
                                 the crux of the spike-susceptibility argument.
  dp_mode_profiles.png        -- per-layer h-gradient of the four Phase-1/2
                                 replayed captures (log y): smooth cascade vs
                                 localized layer-0-2 blowup (real replay data).
  dp_perrow_falsification.png -- per-row depth_code top-1 share for the four
                                 captures vs the flat-batch baseline: localized
                                 events are the FLATTER ones (real SS39 data).
  dp_exponent_occupancy.png   -- V_theta exponent live-fraction per bank across
                                 the four captures: no separation, and >99.9%
                                 of well-token pairs numerically dead (real
                                 SS39 data).

Nothing here is fit or hand-drawn: the well/force panels are exact
evaluations of model_aniso_gaussian_vtheta.py's energy/force expressions,
and the three data panels are the literal numbers logged in the note.
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

SMOOTH = BLUE
LOCAL = RED


# ---------------------------------------------------------------------------
# Figure 1 -- the well: isotropic vs low-rank-sharpened
# ---------------------------------------------------------------------------
def _well_V(H1, H2, mu, P):
    """V(h) = -exp(-0.5 (h-mu)^T P (h-mu)) on a 2D grid (single unit well)."""
    d1 = H1 - mu[0]
    d2 = H2 - mu[1]
    quad = P[0, 0] * d1 * d1 + 2 * P[0, 1] * d1 * d2 + P[1, 1] * d2 * d2
    return -np.exp(-0.5 * quad)


def _well_gradmag(H1, H2, mu, P):
    """|grad_h V| for the same well; grad V = g * P (h-mu), g = -V."""
    d1 = H1 - mu[0]
    d2 = H2 - mu[1]
    quad = P[0, 0] * d1 * d1 + 2 * P[0, 1] * d1 * d2 + P[1, 1] * d2 * d2
    g = np.exp(-0.5 * quad)
    f1 = g * (P[0, 0] * d1 + P[0, 1] * d2)
    f2 = g * (P[0, 1] * d1 + P[1, 1] * d2)
    return np.sqrt(f1 * f1 + f2 * f2)


def fig_well():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    x = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, x)
    mu = np.array([0.0, 0.0])

    a = 1.0                                  # diagonal precision
    # Isotropic: P = a I.
    P_iso = np.array([[a, 0.0], [0.0, a]])
    # Low-rank sharpened: P = a I + b b^T with b along a 30-deg ridge.
    theta = np.deg2rad(30.0)
    bvec = 3.2 * np.array([np.cos(theta), np.sin(theta)])
    P_lr = P_iso + np.outer(bvec, bvec)

    # Panel A: isotropic contour.
    ax = axes[0]
    V = _well_V(X, Y, mu, P_iso)
    ax.contourf(X, Y, V, levels=25, cmap="Blues_r")
    ax.contour(X, Y, V, levels=8, colors="white", linewidths=0.5, alpha=0.6)
    ax.set_title(r"Isotropic well:  $P=\mathrm{diag}(a)$")
    ax.set_xlabel(r"$h_1$"); ax.set_ylabel(r"$h_2$")
    ax.set_aspect("equal")

    # Panel B: low-rank-sharpened contour + the sharp eigendirection.
    ax = axes[1]
    V = _well_V(X, Y, mu, P_lr)
    ax.contourf(X, Y, V, levels=25, cmap="Reds_r")
    ax.contour(X, Y, V, levels=8, colors="white", linewidths=0.5, alpha=0.6)
    evals, evecs = np.linalg.eigh(P_lr)
    v = evecs[:, -1]                          # sharpest direction
    lam = evals[-1]
    tstar = 1.0 / np.sqrt(lam)                # peak-force radius
    ax.annotate("", xy=1.6 * v, xytext=-1.6 * v,
                arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.8))
    ax.plot(*(tstar * v), "o", color=AMBER, ms=8, zorder=5)
    ax.plot(*(-tstar * v), "o", color=AMBER, ms=8, zorder=5)
    ax.text(0.03, 0.03,
            rf"$\sigma_{{\max}}(B_k)^2={lam - a:.1f}$" + "\n"
            rf"$t^*=1/\sqrt{{\lambda}}={tstar:.2f}$",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.85))
    ax.set_title(r"Low-rank sharpened:  $P=\mathrm{diag}(a)+B_kB_k^{\top}$")
    ax.set_xlabel(r"$h_1$"); ax.set_ylabel(r"$h_2$")
    ax.set_aspect("equal")

    # Panel C: gradient-magnitude heatmap of the sharpened well.
    ax = axes[2]
    G = _well_gradmag(X, Y, mu, P_lr)
    pc = ax.contourf(X, Y, G, levels=25, cmap="magma")
    # the ring of maximum |grad| sits at Mahalanobis radius 1 -- a thin shell
    # along the sharp direction, wide along the soft one.
    ax.contour(X, Y, G, levels=[0.9 * G.max()], colors="cyan", linewidths=1.2)
    fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04, label=r"$|\nabla_h V|$")
    ax.set_title(r"Force magnitude: peak on a thin shell $\|h-\mu\|_P\!\approx\!1$")
    ax.set_xlabel(r"$h_1$"); ax.set_ylabel(r"$h_2$")
    ax.set_aspect("equal")

    fig.suptitle(
        "A single low-rank factor $B_k$ turns a round basin into a razor ridge; "
        "the largest forces live on a thin shell, not at the centre",
        fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("dp_well_landscape.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 -- along-direction force profile phi(t) and its peak scaling
# ---------------------------------------------------------------------------
def fig_force_profile():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    t = np.linspace(0, 4, 500)
    lambdas = [1.0, 4.0, 16.0, 64.0]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(lambdas)))

    ax = axes[0]
    for lam, c in zip(lambdas, cmap):
        phi = lam * t * np.exp(-0.5 * lam * t * t)
        ax.plot(t, phi, color=c, lw=2, label=rf"$\lambda={lam:.0f}$")
        tstar = 1.0 / np.sqrt(lam)
        ax.plot(tstar, np.sqrt(lam / np.e), "o", color=c, ms=7, zorder=5)
    ax.set_xlabel(r"displacement along sharp direction  $t=v^{\top}(h-\mu)$")
    ax.set_ylabel(r"along-direction force  $\phi(t)=\lambda t\,e^{-\lambda t^2/2}$")
    ax.set_title(r"One well's force peaks at $t^*=1/\sqrt{\lambda}$")
    ax.legend(title=r"curvature $\lambda=\sigma_{\max}(P_k)$", fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    lam = np.linspace(0.5, 100, 400)
    ax.plot(lam, np.sqrt(lam / np.e), color=RED, lw=2.2,
            label=r"peak force $\propto\sqrt{\lambda}$")
    ax.plot(lam, lam / np.e, color=PURPLE, lw=2.2, ls="--",
            label=r"peak param-grad $\propto\lambda=\sigma_{\max}(B_k)^2$")
    ax.axvline(30.0, color=GREY, ls=":", lw=1.4)
    ax.text(31, ax.get_ylim()[1] * 0.5,
            "typical\nprecision_lr_max\nbudget", fontsize=8.5, color=DARK)
    ax.set_xlabel(r"low-rank curvature  $\lambda=\sigma_{\max}(B_k)^2$")
    ax.set_ylabel("worst-case magnitude (arb. units)")
    ax.set_title("Why a drifting $B_k$ is a spike generator")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle(
        r"Sharper low-rank direction $\Rightarrow$ larger peak force AND a "
        r"narrower, closer active shell $t^*\!\to\!0$",
        fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("dp_force_profile.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 -- the two failure modes, per-layer h-gradient (real replay data)
# ---------------------------------------------------------------------------
LAYER_GRAD = {
    # step: (per-layer h-grad norms layers 0..7, mode)
    37763: ([0.169, 0.102, 0.044, 0.027, 0.017, 0.011, 0.008, 0.005], "smooth"),
    41318: ([0.054, 0.042, 0.028, 0.021, 0.016, 0.011, 0.008, 0.004], "smooth"),
    39983: ([16.45, 12.20, 6.08, 0.33, 0.030, 0.020, 0.010, 0.006], "localized"),
    41837: ([31.07, 12.61, 1.455, 0.175, 0.0167, 0.0109, 0.0077, 0.0046],
            "localized"),
}


def fig_mode_profiles():
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    layers = np.arange(8)
    for step, (prof, mode) in LAYER_GRAD.items():
        c = SMOOTH if mode == "smooth" else LOCAL
        ls = "-" if mode == "localized" else "--"
        lw = 2.4 if mode == "localized" else 1.6
        ax.plot(layers, prof, ls, color=c, lw=lw, marker="o", ms=5,
                label=f"step {step:,}  ({mode})")
    ax.set_yscale("log")
    ax.set_xlabel("layer index (gradient flows $L\\!\\to\\!0$)")
    ax.set_ylabel(r"$\|\nabla_h\|$ into each layer boundary")
    ax.set_title("Two failure modes, one discriminator: the per-layer profile")
    ax.axvspan(-0.4, 2.4, color=AMBER, alpha=0.10)
    ax.text(1.0, ax.get_ylim()[1] * 0.4, "layers 0-2\n(all salience lives here)",
            ha="center", fontsize=8.5, color=DARK)
    ax.set_xticks(layers)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig("dp_mode_profiles.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 -- per-row falsification (real SS39 data)
# ---------------------------------------------------------------------------
PERROW = {
    # step: (depth_code top1 share, mode, pre-clip, layer0/layer3 ratio)
    37763: (0.292, "smooth", 160.4, 6.3),
    41318: (0.387, "smooth", 432.8, 2.6),
    39983: (0.217, "localized", 235.5, 50.0),
    41837: (0.095, "localized", 528.0, 177.0),
}
BASELINE = 1.0 / 32.0


def fig_perrow():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))

    ax = axes[0]
    steps = list(PERROW)
    shares = [PERROW[s][0] for s in steps]
    modes = [PERROW[s][1] for s in steps]
    colors = [SMOOTH if m == "smooth" else LOCAL for m in modes]
    xs = np.arange(len(steps))
    ax.bar(xs, shares, color=colors, width=0.6)
    ax.axhline(BASELINE, color=DARK, ls="--", lw=1.4)
    ax.text(len(steps) - 0.5, BASELINE + 0.006, "flat-batch baseline (1/32)",
            ha="right", fontsize=8.5, color=DARK)
    for x, s in zip(xs, steps):
        ax.text(x, shares[list(PERROW).index(s)] + 0.008,
                f"{PERROW[s][0]:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s:,}\n{PERROW[s][1]}" for s in steps], fontsize=9)
    ax.set_ylabel(r"top-1 row share of $\nabla_{\mathrm{depth\_code}}$")
    ax.set_title("The token-minority conjecture, falsified")
    legend = [Line2D([0], [0], color=SMOOTH, lw=8, label="smooth cascade"),
              Line2D([0], [0], color=LOCAL, lw=8, label="localized")]
    ax.legend(handles=legend, fontsize=9)

    # Panel B: the anti-correlation -- more layer-localized => flatter across rows.
    ax = axes[1]
    for s in steps:
        share, mode, _pre, ratio = PERROW[s]
        c = SMOOTH if mode == "smooth" else LOCAL
        ax.scatter(ratio, share, color=c, s=90, zorder=5)
        ax.annotate(f"{s:,}", (ratio, share), textcoords="offset points",
                    xytext=(6, 6), fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"layer-0 / layer-3 $h$-gradient ratio (localization severity)")
    ax.set_ylabel(r"top-1 row share of $\nabla_{\mathrm{depth\_code}}$")
    ax.set_title("More layer-localized $\\Rightarrow$ more batch-uniform")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig("dp_perrow_falsification.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 -- exponent occupancy (real SS39 data)
# ---------------------------------------------------------------------------
LIVE_FRAC = {
    # step: [bank0..bank4] live fraction (exponent > -10)
    37763: [7.6e-6, 1.5e-5, 1.4e-4, 1.4e-3, 2.4e-3],
    41318: [3.8e-5, 3.1e-5, 1.4e-4, 8.8e-4, 1.1e-3],
    39983: [3.1e-5, 7.6e-6, 1.4e-4, 9.3e-4, 1.6e-3],
    41837: [3.1e-5, 2.3e-5, 1.4e-4, 8.7e-4, 8.6e-4],
}
MODES = {37763: "smooth", 41318: "smooth", 39983: "localized",
         41837: "localized"}


def fig_occupancy():
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    banks = np.arange(5)
    w = 0.2
    for i, step in enumerate(LIVE_FRAC):
        c = SMOOTH if MODES[step] == "smooth" else LOCAL
        hatch = "" if MODES[step] == "smooth" else "//"
        ax.bar(banks + (i - 1.5) * w, LIVE_FRAC[step], w,
               color=c, hatch=hatch, edgecolor="white", linewidth=0.5,
               label=f"{step:,} ({MODES[step]})")
    ax.set_yscale("log")
    ax.set_xlabel(r"$V_\theta$ bank (context channel) index")
    ax.set_ylabel(r"live fraction  (exponent $> -10$)")
    ax.set_title("Exponent occupancy does not separate the modes\n"
                 r"($>99.9\%$ of well-token pairs numerically dead in every capture)")
    ax.set_xticks(banks)
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(alpha=0.25, which="both", axis="y")
    fig.tight_layout()
    fig.savefig("dp_exponent_occupancy.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_well()
    fig_force_profile()
    fig_mode_profiles()
    fig_perrow()
    fig_occupancy()
    print("wrote dp_well_landscape.png, dp_force_profile.png, "
          "dp_mode_profiles.png, dp_perrow_falsification.png, "
          "dp_exponent_occupancy.png")
