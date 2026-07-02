"""
Generate the illustrative figure for the register-collapse / repulsion analysis
in Improving_the_Fock_Mechanism_to_match_Attention.md (section 20).

register_collapse_repulsion.png -- three panels:

  (A) Effective register count N_eff(rho) = M / (1 + (M-1) rho^2), the closed
      form derived in section 20.1 for a bank whose active registers share a
      mean pairwise cosine similarity rho.  As rho -> 1 the M-register pool
      collapses to a single effective direction (N_eff -> 1), wasting capacity.
      Plotted for M = 16 and M = 32; the M = 32 curve collapses faster,
      matching the empirical Q7 M = 32 divergence.

  (B) Geometry of the fix.  Left: Gaussian / shared-key registers cluster into
      one direction (high rho, collapsed).  Right: orthogonal init (B3) places
      them apart, per-register keys (B2) give each an independent handle, and a
      pairwise repulsion force (arrows) supplies the continuous restoring force
      that keeps them apart throughout training.

  (C) NTP reducible-loss floor vs effective rank.  Treating the register pool
      as a rank-r_eff Gaussian channel, the reducible cross-entropy is
      L_red(r) = (1/2) sum_{i=1}^{r} log2(1 + SNR_i) nats-per-symbol of context
      the tokens can recover.  A collapsed pool (r_eff = 1) leaves most of this
      on the table; raising r_eff (via the three fixes) lowers the achievable
      NTP floor with diminishing returns.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BLUE_DARK = "#1565C0"
BLUE_MED = "#1E88E5"
TEAL_DARK = "#00695C"
ORANGE = "#EF6C00"
RED = "#E53935"
PURPLE = "#6A1B9A"
GREEN = "#2E7D32"
GREY = "#616161"
BG = "#FAFAFA"


def n_eff(rho, M):
    """Effective register count for equicorrelated unit rows (section 20.1)."""
    return M / (1.0 + (M - 1) * rho ** 2)


def main():
    fig = plt.figure(figsize=(15.4, 5.4), facecolor=BG)
    gs = GridSpec(1, 3, width_ratios=[1.05, 1.15, 1.05], wspace=0.32,
                  left=0.055, right=0.985, top=0.80, bottom=0.14)

    # ---- Panel A: effective register count vs cosine similarity ----
    axA = fig.add_subplot(gs[0, 0])
    axA.set_facecolor(BG)
    rho = np.linspace(0.0, 1.0, 400)
    for M, c in [(16, BLUE_DARK), (32, ORANGE)]:
        axA.plot(rho, n_eff(rho, M), color=c, lw=2.4, label=f"M = {M}")
    # anchor points: healthy vs collapsed
    axA.scatter([0.05], [n_eff(0.05, 16)], s=90, color=GREEN, zorder=5,
                edgecolor="white", linewidth=1.4)
    axA.annotate("healthy\nrho ~ 0.05", xy=(0.05, n_eff(0.05, 16)),
                 xytext=(0.14, 12.6), fontsize=9.0, color=GREEN,
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
    axA.scatter([0.7], [n_eff(0.7, 16)], s=90, color=RED, zorder=5,
                edgecolor="white", linewidth=1.4)
    axA.annotate("collapsed\nrho ~ 0.7", xy=(0.7, n_eff(0.7, 16)),
                 xytext=(0.55, 6.0), fontsize=9.0, color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    axA.set_title("(A)  Capacity collapse\n"
                  r"$N_{\mathrm{eff}} = M / (1 + (M-1)\rho^2)$",
                  fontsize=11.5, color=BLUE_DARK, fontweight="bold", pad=8)
    axA.set_xlabel(r"mean pairwise cosine similarity  $\rho$", fontsize=10)
    axA.set_ylabel("effective register count  N_eff", fontsize=10)
    axA.set_xlim(0, 1)
    axA.set_ylim(0, 33)
    axA.legend(fontsize=9.5, frameon=False, loc="upper right")
    axA.grid(True, alpha=0.3, linestyle=":")
    for spine in ["top", "right"]:
        axA.spines[spine].set_visible(False)

    # ---- Panel B: geometry of the fix ----
    axB = fig.add_subplot(gs[0, 1])
    axB.set_facecolor(BG)
    axB.set_xlim(-1.35, 1.35)
    axB.set_ylim(-1.25, 1.35)
    axB.set_aspect("equal")
    axB.axis("off")
    axB.set_title("(B)  Init spread (B3) + private keys (B2)\n"
                  "+ repulsion force keep registers apart",
                  fontsize=11.5, color=TEAL_DARK, fontweight="bold", pad=8)

    # left cluster: collapsed registers (near-parallel)
    cx, cy = -0.72, 0.05
    base = np.array([0.62, 0.42])
    base = base / np.linalg.norm(base)
    rng = np.random.default_rng(3)
    for _ in range(6):
        d = base + rng.normal(0, 0.05, size=2)
        d = d / np.linalg.norm(d) * 0.52
        axB.annotate("", xy=(cx + d[0], cy + d[1]), xytext=(cx, cy),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.8,
                                     alpha=0.85))
    axB.text(cx, cy - 0.72, "collapsed\n(shared keys,\nGaussian init)",
             ha="center", va="top", fontsize=9.0, color=RED, fontweight="bold")
    axB.text(cx, cy + 0.82, "rho -> 1", ha="center", fontsize=9.0, color=RED)

    # right cluster: repelled registers (spread) with repulsion arrows
    dx, dy = 0.68, 0.05
    n = 6
    angs = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.3
    tips = []
    for a in angs:
        d = np.array([np.cos(a), np.sin(a)]) * 0.55
        tips.append((dx + d[0], dy + d[1]))
        axB.annotate("", xy=(dx + d[0], dy + d[1]), xytext=(dx, dy),
                     arrowprops=dict(arrowstyle="->", color=TEAL_DARK, lw=1.9))
    # small repulsion force hints between neighbouring tips
    for i in range(n):
        x0, y0 = tips[i]
        x1, y1 = tips[(i + 1) % n]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        push = np.array([mx - dx, my - dy])
        push = push / (np.linalg.norm(push) + 1e-9) * 0.12
        axB.annotate("", xy=(mx + push[0], my + push[1]), xytext=(mx, my),
                     arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.3,
                                     alpha=0.8))
    axB.text(dx, dy - 0.72, "spread\n(ortho init + private keys\n+ repulsion)",
             ha="center", va="top", fontsize=9.0, color=TEAL_DARK,
             fontweight="bold")
    axB.text(dx, dy + 0.82, "rho -> 0", ha="center", fontsize=9.0,
             color=TEAL_DARK)
    axB.annotate("", xy=(0.02, 0.05), xytext=(-0.30, 0.05),
                 arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2.2))
    axB.text(-0.14, 0.20, "fixes", ha="center", fontsize=9.0, color=GREY,
             style="italic")
    axB.text(dx + 0.62, dy + 0.30, "repulsion\nforce", fontsize=8.2,
             color=PURPLE, fontweight="bold", ha="left")

    # ---- Panel C: NTP reducible-loss floor vs effective rank ----
    axC = fig.add_subplot(gs[0, 2])
    axC.set_facecolor(BG)
    r = np.arange(1, 17)
    # per-mode SNR with mild decay across register modes (illustrative)
    snr0, decay = 3.0, 0.82
    snr = snr0 * decay ** (r - 1)
    red = 0.5 * np.log2(1 + snr)          # reducible nats-per-symbol per mode
    cum = np.cumsum(red)                    # total recoverable context info
    H_y = cum[-1] + 1.6                     # illustrative base entropy floor
    floor = H_y - cum                       # achievable NTP floor vs r_eff

    axC.plot(r, floor, color=PURPLE, lw=2.4, marker="o", ms=4,
             markerfacecolor="white", markeredgecolor=PURPLE)
    axC.scatter([1], [floor[0]], s=110, color=RED, zorder=5,
                edgecolor="white", linewidth=1.4)
    axC.annotate("collapsed\n(r_eff = 1)", xy=(1, floor[0]),
                 xytext=(2.2, floor[0] + 0.02), fontsize=9.0, color=RED,
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    r_star = 9
    axC.scatter([r_star], [floor[r_star - 1]], s=110, color=GREEN, zorder=5,
                edgecolor="white", linewidth=1.4)
    axC.annotate("healthy\n(r_eff ~ 9)", xy=(r_star, floor[r_star - 1]),
                 xytext=(r_star - 1.5, floor[r_star - 1] - 0.9),
                 fontsize=9.0, color=GREEN, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
    axC.annotate("", xy=(r_star, floor[r_star - 1] + 0.05),
                 xytext=(1, floor[0] - 0.05),
                 arrowprops=dict(arrowstyle="->", color=GREY, lw=1.7,
                                 linestyle=(0, (4, 3))))
    axC.text(5.0, (floor[0] + floor[r_star - 1]) / 2 + 0.35,
             "raise r_eff ->\nlower NTP floor", fontsize=8.8, color=GREY,
             ha="center", va="center", style="italic")
    axC.set_title("(C)  NTP floor drops as r_eff grows\n"
                  r"$L_{\mathrm{red}}(r)=\frac{1}{2}\sum_{i\leq r}\log_2(1+\mathrm{SNR}_i)$",
                  fontsize=11.5, color=PURPLE, fontweight="bold", pad=8)
    axC.set_xlabel("effective register rank  r_eff", fontsize=10)
    axC.set_ylabel("achievable NTP floor  (nats, lower better)", fontsize=10)
    axC.set_xlim(0.4, 16.6)
    axC.grid(True, alpha=0.3, linestyle=":")
    for spine in ["top", "right"]:
        axC.spines[spine].set_visible(False)

    fig.suptitle("Register-content collapse and the three fixes: initial spread "
                 "(B3), private keys (B2), and a repulsion restoring force",
                 fontsize=13.2, fontweight="bold", color="#212121", y=0.965)

    out = "register_collapse_repulsion.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    print("wrote", out)


if __name__ == "__main__":
    main()
