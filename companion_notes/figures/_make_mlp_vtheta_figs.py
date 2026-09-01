"""Figures for MLP_VTheta_Fock-PARFLM_on_OWT_d384.md.

Generates three PNGs into the same folder. Every curve is either an exact
parameter-count evaluation of the two V_theta parameterisations, an exact
evaluation of the causal-EMA kernels for the deployed xi-channel alphas, or
a transparent schematic of the additive-vs-joint potential structure -- all
computed, none hand-drawn:

  mlpvt_param_scaling.png     -- V_theta parameter count vs number of K-EMA
                                 context channels for the shared-MLP
                                 (ScalarPotentialMultiXi) vs the additive
                                 per-channel Gaussian banks (isotropic rank-0
                                 and anisotropic rank-4). Panel B breaks the
                                 anisotropic per-bank cost into mu/a/w/B_proj
                                 to show B_proj (the low-rank curvature
                                 machinery, i.e. the spike surface) dominates.
  mlpvt_channel_redundancy.png-- exact cosine similarity between the causal
                                 EMA kernels of the deployed 5long alphas
                                 (Panel A), and the mean off-diagonal kernel
                                 similarity + effective channel count
                                 (participation ratio) as more channels are
                                 packed into the same horizon band (Panel B).
  mlpvt_expressivity.png      -- schematic of why an ADDITIVE per-channel
                                 potential (sum of two 1-D wells -> a
                                 separable "cross" salience) cannot represent
                                 a cross-horizon CONJUNCTION (a product-shaped
                                 corner blob) that a joint MLP can. Both
                                 panels are exact evaluations of the toy
                                 salience fields; only the fields are toys.

Exact model facts reproduced here:
  ScalarPotentialMultiXi:  in_dim = (K+1)*d ; net = Linear(in_dim,H),GELU,
                           (depth-1) x [Linear(H,H),GELU], Linear(H,1).
  AnisotropicMixtureGaussianVTheta bank (per context channel):
                           mu_proj: Linear(d, W*d)
                           a_proj : Linear(d, W*d)
                           w_proj : Linear(d, W)
                           B_proj : Linear(d, W*d*rank)   (rank>0 only)
  AnisotropicMultiContextGaussianVTheta: one bank per context channel,
                           V = sum_m V^(m)(xi^(m), h)     (additive).
Deployed OWT d384 config: d=384, H=2048, depth=3, W=8 wells, rank=4,
  xi '5long' alphas = [0.50, 0.75, 0.95, 0.99, 0.995].
"""

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

# Deployed OWT d384 configuration.
D = 384
H = 2048
DEPTH = 3
W = 8        # wells per bank (V_THETA_WELLS_PER_HEAD)
RANK = 4     # ANISO_RANK
L = 16       # Verlet layers (for depth_code count)
ALPHAS_5LONG = [0.50, 0.75, 0.95, 0.99, 0.995]


# ---------------------------------------------------------------------------
# Exact parameter-count formulas.
# ---------------------------------------------------------------------------
def mlp_vtheta_params(K, d=D, hh=H, depth=DEPTH):
    """ScalarPotentialMultiXi: (K+1)*d -> H -> ... -> 1."""
    in_dim = (K + 1) * d
    p = in_dim * hh + hh                 # first Linear + bias
    p += (depth - 1) * (hh * hh + hh)    # hidden Linears
    p += hh * 1 + 1                      # head
    return p


def aniso_bank_params(d=D, w=W, rank=RANK):
    """One AnisotropicMixtureGaussianVTheta bank (in_d = d)."""
    mu = d * (w * d) + w * d
    a = d * (w * d) + w * d
    ww = d * w + w
    b = (d * (w * d * rank) + w * d * rank) if rank > 0 else 0
    return dict(mu=mu, a=a, w=ww, B=b, total=mu + a + ww + b)


def gaussian_vtheta_params(K, d=D, w=W, rank=RANK, n_layers=L):
    """AnisotropicMultiContextGaussianVTheta: K banks + depth_code (L,K,d)."""
    bank = aniso_bank_params(d, w, rank)["total"]
    depth_code = n_layers * K * d
    return K * bank + depth_code


def mlp_marginal(d=D, hh=H):
    """Params added to the MLP per extra context channel (first-layer only)."""
    return d * hh


# ---------------------------------------------------------------------------
# Figure 1: parameter scaling with the number of context channels.
# ---------------------------------------------------------------------------
def fig_param_scaling():
    Ks = np.arange(1, 9)
    mlp = np.array([mlp_vtheta_params(k) for k in Ks], dtype=float)
    iso = np.array([gaussian_vtheta_params(k, rank=0) for k in Ks], dtype=float)
    ani = np.array([gaussian_vtheta_params(k, rank=RANK) for k in Ks], dtype=float)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.6))

    axA.plot(Ks, mlp / 1e6, "-o", color=BLUE, lw=2.4, ms=6,
             label="MLP  (shared net, H=2048 x depth 3)")
    axA.plot(Ks, iso / 1e6, "-s", color=GREEN, lw=2.4, ms=6,
             label="Gaussian isotropic  (rank 0)")
    axA.plot(Ks, ani / 1e6, "-^", color=RED, lw=2.4, ms=6,
             label="Gaussian anisotropic  (rank 4)")
    axA.axvline(4, color=GREY, ls=":", lw=1.3)
    axA.axvline(5, color=DARK, ls="--", lw=1.3)
    axA.text(5.05, axA.get_ylim()[1] * 0.02, "5long (deployed)",
             rotation=90, va="bottom", ha="left", color=DARK, fontsize=9)

    # marginal-cost annotations
    mlp_slope = mlp_marginal() / 1e6
    iso_slope = aniso_bank_params(rank=0)["total"] / 1e6
    ani_slope = aniso_bank_params(rank=RANK)["total"] / 1e6
    axA.text(1.1, mlp[-1] / 1e6 * 0.60,
             f"per extra channel:\n"
             f"MLP  +{mlp_slope:.2f}M  (shared depth)\n"
             f"iso  +{iso_slope:.2f}M  (a whole bank)\n"
             f"aniso  +{ani_slope:.2f}M  (a whole bank)",
             fontsize=9, color=DARK,
             bbox=dict(boxstyle="round", fc="#F9FAFB", ec=GREY))

    axA.set_xlabel("number of K-EMA context channels")
    axA.set_ylabel("V_theta parameters  (millions)")
    axA.set_title("A. Cost of one more horizon")
    axA.legend(loc="upper left", fontsize=9, framealpha=0.95)
    axA.grid(alpha=0.25)

    # Panel B: per-bank breakdown for the anisotropic bank.
    bk = aniso_bank_params(rank=RANK)
    parts = ["B_proj\n(low-rank\ncurvature)", "mu_proj", "a_proj", "w_proj"]
    vals = np.array([bk["B"], bk["mu"], bk["a"], bk["w"]]) / 1e6
    colors = [RED, PURPLE, BLUE, GREY]
    bars = axB.bar(parts, vals, color=colors, edgecolor=DARK, lw=0.6)
    for b, v in zip(bars, vals):
        axB.text(b.get_x() + b.get_width() / 2, v + 0.05,
                 f"{v:.2f}M", ha="center", va="bottom", fontsize=9)
    frac = 100.0 * bk["B"] / bk["total"]
    axB.set_ylabel("parameters per bank  (millions)")
    axB.set_title("B. Where the anisotropic bank spends its budget")
    axB.text(0.98, 0.95,
             f"B_proj = {frac:.0f}% of each bank\n"
             f"= the low-rank curvature machinery\n"
             f"(the spike surface, see the diagnostic note)",
             transform=axB.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#FEF2F2", ec=RED))
    axB.set_ylim(0, vals.max() * 1.25)
    axB.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    out = "mlpvt_param_scaling.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Causal EMA kernel utilities.
#   xi^(k)_t = (1-a_k) sum_{s<=t} a_k^{t-s} h_s  (normalised geometric weights)
#   Kernel at the final position over the window is  w_k[j] ~ a_k^{j}.
# ---------------------------------------------------------------------------
def ema_kernel(alpha, T=512):
    j = np.arange(T)                      # lag from the query position
    w = (1.0 - alpha) * alpha ** j
    return w / np.linalg.norm(w)


def cosine_matrix(alphas, T=512):
    ker = np.stack([ema_kernel(a, T) for a in alphas])
    return ker @ ker.T


def participation_ratio(alphas, T=512):
    """Effective number of independent channels from the kernel Gram spectrum."""
    ker = np.stack([ema_kernel(a, T) for a in alphas])
    G = ker @ ker.T
    ev = np.linalg.eigvalsh(G)
    ev = np.clip(ev, 0, None)
    return (ev.sum() ** 2) / (np.square(ev).sum() + 1e-12)


def fig_channel_redundancy():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.6))

    # Panel A: cosine similarity of the deployed 5long kernels.
    C = cosine_matrix(ALPHAS_5LONG)
    im = axA.imshow(C, cmap="magma", vmin=0, vmax=1)
    horizons = [f"a={a}\n(~{round(1/(1-a))} tok)" for a in ALPHAS_5LONG]
    axA.set_xticks(range(len(ALPHAS_5LONG)))
    axA.set_yticks(range(len(ALPHAS_5LONG)))
    axA.set_xticklabels(horizons, fontsize=8)
    axA.set_yticklabels(horizons, fontsize=8)
    for i in range(len(ALPHAS_5LONG)):
        for jx in range(len(ALPHAS_5LONG)):
            axA.text(jx, i, f"{C[i, jx]:.2f}", ha="center", va="center",
                     color="white" if C[i, jx] < 0.6 else "black", fontsize=8)
    axA.set_title("A. 5long kernel cosine similarity")
    fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
    # highlight the redundant long pair
    axA.add_patch(plt.Rectangle((2.5, 2.5), 2, 2, fill=False,
                                edgecolor=GREEN, lw=2.4))
    axA.text(3.5, 1.5, "long horizons\nnearly duplicate\n(0.94)",
             color=GREEN, fontsize=8, ha="center", va="center",
             bbox=dict(boxstyle="round", fc="#0b0b0b", ec=GREEN, alpha=0.75))

    # Panel B: redundancy grows as more channels pack the same horizon band.
    Ks = np.arange(2, 13)
    mean_off = []
    pr = []
    for K in Ks:
        # log-uniform horizons from ~2 to ~200 tokens -> alphas
        taus = np.geomspace(2.0, 200.0, K)
        al = 1.0 - 1.0 / taus
        C = cosine_matrix(list(al))
        off = C[~np.eye(K, dtype=bool)]
        mean_off.append(off.mean())
        pr.append(participation_ratio(list(al)))
    mean_off = np.array(mean_off)
    pr = np.array(pr)

    axB.plot(Ks, mean_off, "-o", color=RED, lw=2.2, ms=5,
             label="mean off-diagonal kernel similarity")
    axB.set_xlabel("number of channels packed into ~2-200 tok")
    axB.set_ylabel("mean pairwise cosine similarity", color=RED)
    axB.tick_params(axis="y", labelcolor=RED)
    axB.set_ylim(0, 1)
    axB.grid(alpha=0.25)

    ax2 = axB.twinx()
    ax2.plot(Ks, pr, "-s", color=BLUE, lw=2.2, ms=5,
             label="effective channel count")
    ax2.plot(Ks, Ks, ":", color=GREY, lw=1.5, label="ideal (all independent)")
    ax2.set_ylabel("effective # channels (participation ratio)", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax2.set_ylim(0, Ks.max() + 1)

    axB.axvline(5, color=DARK, ls="--", lw=1.3, label="5 channels (5long)")
    axB.set_title("B. More channels -> more overlap, fewer new dimensions")
    lines = (axB.get_lines() + ax2.get_lines())
    axB.legend(lines, [l.get_label() for l in lines],
               loc="upper center", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    out = "mlpvt_channel_redundancy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3: additive superposition vs cross-horizon conjunction (schematic).
# ---------------------------------------------------------------------------
def fig_expressivity():
    u = np.linspace(-3, 3, 300)
    v = np.linspace(-3, 3, 300)
    U, Vv = np.meshgrid(u, v)
    c = 1.2
    gu = np.exp(-((U - c) ** 2) / 0.5)
    gv = np.exp(-((Vv - c) ** 2) / 0.5)

    add = gu + gv          # additive: what per-channel Gaussian banks produce
    conj = gu * gv         # conjunction: what a joint MLP can also represent

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.8))

    for ax in (axA, axB):
        ax.set_xlabel("short-horizon feature  (channel 1)")
        ax.set_ylabel("long-horizon feature  (channel 2)")

    im1 = axA.contourf(U, Vv, add, levels=20, cmap="viridis")
    axA.set_title("A. Additive per-channel salience  s = g1 + g2")
    fig.colorbar(im1, ax=axA, fraction=0.046, pad=0.04)
    axA.text(0.03, 0.97,
             "responds to channel 1 OR channel 2\n"
             "(a separable 'cross'); no term couples them",
             transform=axA.transAxes, va="top", ha="left", fontsize=9,
             color="white")

    im2 = axB.contourf(U, Vv, conj, levels=20, cmap="viridis")
    axB.set_title("B. Cross-horizon conjunction  s = g1 x g2")
    fig.colorbar(im2, ax=axB, fraction=0.046, pad=0.04)
    axB.text(0.03, 0.97,
             "fires only where channel 1 AND channel 2 agree\n"
             "(a localised corner); MLP can, additive banks cannot",
             transform=axB.transAxes, va="top", ha="left", fontsize=9,
             color="white")
    axB.plot(c, c, "*", color=RED, ms=16, mec="white")

    fig.tight_layout()
    out = "mlpvt_expressivity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    # quick sanity print of the numbers the note quotes
    print("MLP V_theta params:")
    for k in (4, 5, 6, 8):
        print(f"  K={k}: {mlp_vtheta_params(k):,}")
    print("Gaussian aniso V_theta params:")
    for k in (4, 5, 6, 8):
        print(f"  K={k}: {gaussian_vtheta_params(k):,}")
    print("aniso per-bank breakdown:", {k: f"{v:,}" for k, v in
          aniso_bank_params().items()})
    print(f"MLP marginal per channel: {mlp_marginal():,}")
    print(f"5long participation ratio: {participation_ratio(ALPHAS_5LONG):.3f}")

    fig_param_scaling()
    fig_channel_redundancy()
    fig_expressivity()
