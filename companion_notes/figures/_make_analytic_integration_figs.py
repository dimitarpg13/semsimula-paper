"""Figures for Analytic_Multi_Channel_Integration_in_Structured_Vtheta.md.

Three PNGs into this folder:

  ami_fusion_1d.png       -- additive (mixture / OR) vs product (fusion / AND)
                             of two per-channel Gaussian factors in 1-D h.
                             Exact evaluation of the completing-the-square
                             identity: the product collapses to a single well
                             at the precision-weighted mean mu_star with
                             precision P = p1 + p2 (stiffer than either).
  ami_curvature_dial.png  -- fused precision (and sigma_max proxy) vs number
                             of fused channels for product fusion (sum, grows
                             linearly -> spike surface) vs convex fusion
                             (bounded by the stiffest channel). The knob that
                             trades conjunction strength against stiffness.
  ami_channel_hessian.png -- the channel-input Hessian |d2V / dxi_i dxi_j|
                             computed from the actual model: additive is
                             block-diagonal (separable, no cross-horizon
                             term); joint is dense (cross-horizon coupling).
                             This is the mechanism the whole note is about.

Figs 1-2 are pure-numpy exact evaluations. Fig 3 imports the real
model classes and differentiates them with autograd.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

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


# ---------------------------------------------------------------------------
# Figure 1: additive (OR) vs product fusion (AND) of two Gaussian factors.
# ---------------------------------------------------------------------------
def fig_fusion_1d():
    h = np.linspace(-4, 4, 800)
    mu1, mu2 = -1.2, 1.8
    p1, p2 = 1.6, 0.7          # per-channel precisions (curvatures)
    w = 1.0

    # additive: sum of two attractive bumps -> two wells (OR)
    bump1 = np.exp(-0.5 * p1 * (h - mu1) ** 2)
    bump2 = np.exp(-0.5 * p2 * (h - mu2) ** 2)
    V_add = -w * (bump1 + bump2)

    # product: multiply the two Gaussian factors -> one fused well (AND)
    # exponent adds: p1(h-mu1)^2 + p2(h-mu2)^2 = P(h-mu*)^2 + c
    P = p1 + p2
    mu_star = (p1 * mu1 + p2 * mu2) / P
    c = p1 * mu1 ** 2 + p2 * mu2 ** 2 - P * mu_star ** 2
    V_prod = -w * np.exp(-0.5 * c) * np.exp(-0.5 * P * (h - mu_star) ** 2)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=False)

    axA.plot(h, V_add, color=BLUE, lw=2.6)
    axA.axvline(mu1, color=GREY, ls=":", lw=1.3)
    axA.axvline(mu2, color=GREY, ls=":", lw=1.3)
    axA.text(mu1, -0.08, "mu1", color=DARK, ha="center", va="top", fontsize=9)
    axA.text(mu2, -0.08, "mu2", color=DARK, ha="center", va="top", fontsize=9)
    axA.set_title("A. Additive channels: V = -(g1 + g2)  (mixture -> OR)")
    axA.set_xlabel("h (one dimension)")
    axA.set_ylabel("potential V")
    axA.text(0.03, 0.06,
             "two separate wells:\nh is drawn to channel 1 OR channel 2",
             transform=axA.transAxes, va="bottom", ha="left", fontsize=9,
             bbox=dict(boxstyle="round", fc="#EFF6FF", ec=BLUE))
    axA.grid(alpha=0.25)

    axB.plot(h, V_prod, color=RED, lw=2.6)
    axB.axvline(mu1, color=GREY, ls=":", lw=1.3)
    axB.axvline(mu2, color=GREY, ls=":", lw=1.3)
    axB.axvline(mu_star, color=GREEN, ls="--", lw=1.8)
    axB.text(mu_star, -0.02, "  mu_star", color=GREEN, ha="left", va="top",
             fontsize=9)
    axB.set_title("B. Product fusion: V = -(g1 x g2)  (product -> AND)")
    axB.set_xlabel("h (one dimension)")
    axB.set_ylabel("potential V")
    axB.text(0.03, 0.06,
             f"one fused well at the\nprecision-weighted mean\n"
             f"mu_star = (p1 mu1 + p2 mu2) / P\n"
             f"P = p1 + p2 = {P:.1f}  (stiffer than either)",
             transform=axB.transAxes, va="bottom", ha="left", fontsize=9,
             bbox=dict(boxstyle="round", fc="#FEF2F2", ec=RED))
    axB.grid(alpha=0.25)

    fig.tight_layout()
    out = "ami_fusion_1d.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: curvature accumulation vs the confinement dial.
# ---------------------------------------------------------------------------
def fig_curvature_dial():
    Ks = np.arange(1, 9)
    # per-channel precisions drawn once, reused (illustrative but fixed).
    rng = np.random.default_rng(0)
    p = 0.6 + 0.8 * rng.random(8)           # per-channel curvatures

    prod = np.array([p[:k].sum() for k in Ks])          # product: sum
    convex = np.array([p[:k].mean() for k in Ks])       # convex: weighted mean
    # additive mixture: each well keeps its own precision; the *max* well
    # curvature does not grow, but the number of wells (spike sites) does.
    mixture_maxcurv = np.array([p[:k].max() for k in Ks])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.6))

    axA.plot(Ks, prod, "-^", color=RED, lw=2.4, ms=6,
             label="product fusion  P = sum p_m  (AND, stiff)")
    axA.plot(Ks, convex, "-o", color=GREEN, lw=2.4, ms=6,
             label="convex fusion  P = sum beta_m p_m  (bounded)")
    axA.plot(Ks, mixture_maxcurv, "-s", color=BLUE, lw=2.0, ms=5,
             label="additive mixture  max well curvature (flat)")
    axA.set_xlabel("number of channels fused into one well")
    axA.set_ylabel("fused well curvature (precision)")
    axA.set_title("A. Fusion adds curvature; convex fusion caps it")
    axA.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    axA.grid(alpha=0.25)
    axA.annotate("spike surface\ngrows with K", xy=(Ks[-1], prod[-1]),
                 xytext=(4.4, prod[-1] * 0.72), color=RED, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=RED))

    # Panel B: the stability wall. Explicit Verlet is stable while
    # omega*dt < 2, omega = sqrt(P/m). Product fusion pushes omega up.
    m, dt = 1.0, 1.0
    omega_prod = np.sqrt(prod / m)
    omega_convex = np.sqrt(convex / m)
    axB.plot(Ks, omega_prod * dt, "-^", color=RED, lw=2.4, ms=6,
             label="product fusion")
    axB.plot(Ks, omega_convex * dt, "-o", color=GREEN, lw=2.4, ms=6,
             label="convex fusion")
    axB.axhline(2.0, color=DARK, ls="--", lw=1.6)
    axB.text(1.1, 2.05, "explicit-Verlet stability wall  omega dt = 2",
             color=DARK, fontsize=9, va="bottom")
    axB.set_xlabel("number of channels fused into one well")
    axB.set_ylabel("omega dt  (explicit-step stiffness)")
    axB.set_title("B. Why fusion needs CfC-BAOAB or a precision cap")
    axB.legend(loc="upper left", fontsize=9, framealpha=0.95)
    axB.grid(alpha=0.25)

    fig.tight_layout()
    out = "ami_curvature_dial.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3: the channel-input Hessian, computed from the real model.
# ---------------------------------------------------------------------------
def fig_channel_hessian():
    parf = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..",
        "notebooks", "conservative_arch", "parf",
    ))
    sys.path.insert(0, parf)
    import torch
    from model_aniso_gaussian_vtheta import (
        AnisotropicMultiContextGaussianVTheta,
        JointContextAnisotropicGaussianVTheta,
    )

    d, K, n_ctx, rank = 6, 4, 3, 2

    def channel_hessian(bank, seed=2):
        torch.manual_seed(seed)
        h = torch.randn(1, 1, d)
        xi0 = torch.randn(n_ctx * d)

        def f(xi_flat):
            xis = xi_flat.view(1, 1, n_ctx, d)
            return bank(xis, h).sum()

        H = torch.autograd.functional.hessian(f, xi0)
        return H.abs().numpy()

    torch.manual_seed(1)
    joint = JointContextAnisotropicGaussianVTheta(d=d, K=K, n_ctx=n_ctx, rank=rank)
    with torch.no_grad():
        torch.nn.init.normal_(joint.bank.a_proj.weight, std=0.1)
        torch.nn.init.normal_(joint.bank.w_proj.weight, std=0.1)
        torch.nn.init.normal_(joint.bank.B_proj.weight, std=0.1)
    Hj = channel_hessian(joint)

    additive = AnisotropicMultiContextGaussianVTheta(d=d, K=K, n_ctx=n_ctx, rank=rank)
    with torch.no_grad():
        for b in additive.banks:
            torch.nn.init.normal_(b.a_proj.weight, std=0.1)
            torch.nn.init.normal_(b.w_proj.weight, std=0.1)
            torch.nn.init.normal_(b.B_proj.weight, std=0.1)
    Ha = channel_hessian(additive)

    vmax = max(Hj.max(), Ha.max())
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 5.0))

    for ax, Hm, title, cross in (
        (axA, Ha, "A. Additive banks: block-diagonal (separable)",
         Ha[0:d, d:2 * d].max()),
        (axB, Hj, "B. Joint bank: dense (cross-horizon coupling)",
         Hj[0:d, d:2 * d].max()),
    ):
        im = ax.imshow(Hm, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(title)
        for b in range(1, n_ctx):
            ax.axhline(b * d - 0.5, color="white", lw=1.0, alpha=0.6)
            ax.axvline(b * d - 0.5, color="white", lw=1.0, alpha=0.6)
        ticks = [i * d + d / 2 - 0.5 for i in range(n_ctx)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"xi {i+1}" for i in range(n_ctx)])
        ax.set_yticklabels([f"xi {i+1}" for i in range(n_ctx)])
        ax.set_xlabel("context input (by channel)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # mark the channel-1 x channel-2 off-block (rows xi1, cols xi2)
        ax.add_patch(plt.Rectangle((d - 0.5, -0.5), d, d, fill=False,
                                   edgecolor=GREEN, lw=2.2))
        # annotate inside the (xi3, xi1) off-block, which is dark in both panels
        ax.text(d / 2 - 0.5, 2 * d + d / 2 - 0.5,
                f"cross block\nmax = {cross:.1e}",
                color="white", ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round", fc="#0b0b0b", ec=GREEN, alpha=0.8))

    fig.suptitle("Channel-input Hessian  |d2V / dxi_i dxi_j|  (from the model)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = "ami_channel_hessian.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_fusion_1d()
    fig_curvature_dial()
    fig_channel_hessian()
