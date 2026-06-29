"""
Generate the illustrative figure for the reverse-channel benefit analysis in
Improving_the_Fock_Mechanism_to_match_Attention.md (section 10.8-10.11).

reverse_channel_benefit.png -- three panels:
  (A) Without the reverse channel: the effective token-token coupling that
      survives integrating out the registers is SYMMETRIC (A = A^T). Only
      undirected associations are representable.
  (B) With the reverse channel: A_ij ~ sum_k beta_ik alpha_kj is ASYMMETRIC.
      A directed "induction / copy" band appears in the upper part that the
      symmetric model cannot represent. This directed component is exactly
      what next-token prediction needs (copy the token that followed a prior
      occurrence of the current token).
  (C) Empirical anchor: across the two causal-fixed v2.1 runs the better
      model invests a far larger reverse-channel scale. |s*|^2 is the
      curvature-relevant reliance measure; the better model uses ~16x more.

All data in panel (C) are real numbers pulled from the v2.1 training logs.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# colour palette (kept consistent with the other companion figures)
BLUE_DARK = "#1565C0"
BLUE_MED = "#1E88E5"
TEAL_DARK = "#00695C"
ORANGE = "#EF6C00"
RED = "#E53935"
PURPLE = "#6A1B9A"
GREY = "#616161"
BG = "#FAFAFA"

rng = np.random.default_rng(7)

T = 14  # toy sequence length for the coupling heatmaps


def symmetric_coupling(n):
    """Undirected, reciprocal coupling: A = A^T (visibly mirror-symmetric)."""
    base = rng.gamma(1.4, 0.5, size=(n, n))
    sym = 0.5 * (base + base.T)
    # local-decay envelope so it reads as a sensible interaction kernel
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :])
    env = np.exp(-dist / 3.5)
    A = sym * env
    np.fill_diagonal(A, A.max() * 0.9)
    return A / A.max()


def asymmetric_coupling(n):
    """Directed coupling: symmetric part + a directed induction/copy band that
    lives only below the diagonal (token i reads from earlier token j<i),
    making A != A^T visibly obvious."""
    A = symmetric_coupling(n) * 0.45
    # directed induction band: token i reads strongly from i-2 / i-1 (copy what
    # followed a previous occurrence) -- a one-sided, non-reciprocal stripe
    for i in range(n):
        if i - 2 >= 0:
            A[i, i - 2] += 0.9
        if i - 1 >= 0:
            A[i, i - 1] += 0.4
    # sparse long-range directed links (induction-head style), strictly i>j
    A[11, 3] += 0.85
    A[9, 2] += 0.7
    A[12, 6] += 0.6
    return A / A.max()


def main():
    fig = plt.figure(figsize=(15.2, 5.5), facecolor=BG)
    gs = GridSpec(1, 4, width_ratios=[1.0, 1.0, 0.05, 1.18], wspace=0.45,
                  left=0.05, right=0.985, top=0.78, bottom=0.14)

    A_sym = symmetric_coupling(T)
    A_asym = asymmetric_coupling(T)

    cmap = "magma"

    # ---- Panel A: symmetric (no reverse channel) ----
    axA = fig.add_subplot(gs[0, 0])
    axA.set_facecolor(BG)
    im = axA.imshow(A_sym, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    axA.set_title("(A)  No reverse channel\neffective coupling is symmetric  "
                  r"$A=A^{\mathsf{T}}$",
                  fontsize=11.5, color=BLUE_DARK, fontweight="bold", pad=8)
    axA.set_xlabel("source token  j", fontsize=10)
    axA.set_ylabel("target token  i", fontsize=10)
    axA.text(0.5, -0.27, "only undirected associations\n(reciprocity preserved)",
             transform=axA.transAxes, ha="center", va="top",
             fontsize=9.5, color=GREY)

    # ---- Panel B: asymmetric (with reverse channel) ----
    axB = fig.add_subplot(gs[0, 1])
    axB.set_facecolor(BG)
    axB.imshow(A_asym, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    axB.set_title("(B)  With reverse channel\n"
                  r"$A_{ij}\sim\sum_k \beta_{ik}\,\alpha_{kj}$  is asymmetric",
                  fontsize=11.5, color=TEAL_DARK, fontweight="bold", pad=8)
    axB.set_xlabel("source token  j", fontsize=10)
    axB.set_ylabel("target token  i", fontsize=10)
    # highlight the directed induction band
    axB.annotate("directed\ninduction /\ncopy band",
                 xy=(9.0, 11.0), xytext=(2.2, 12.6),
                 fontsize=9.0, color=RED, fontweight="bold", ha="left",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    axB.text(0.5, -0.27, "directed routing -> copy / induction\n"
                         "(impossible for any symmetric A)",
             transform=axB.transAxes, ha="center", va="top",
             fontsize=9.5, color=GREY)

    # shared colourbar in its own slim gridspec column
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("coupling strength", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # ---- Panel C: empirical reliance vs PPL ----
    axC = fig.add_subplot(gs[0, 3])
    axC.set_facecolor(BG)

    # real numbers from the causal-fixed v2.1 training logs
    runs = ["v2.1\ntau-only", "v2.1\ntau+perK+ortho"]
    s_star = np.array([0.056, -0.227])      # learned reverse-channel scale
    ppl = np.array([11.18, 9.30])           # final TinyStories val PPL
    reliance = s_star ** 2                   # curvature-relevant reliance

    colors = [ORANGE, PURPLE]
    sizes = np.array([260.0, 1100.0])  # fixed, readable bubble sizes

    axC.set_xscale("log")
    axC.set_xlim(reliance[0] * 0.30, reliance[1] * 4.6)
    axC.set_ylim(8.4, 12.0)

    for k in range(2):
        axC.scatter(reliance[k], ppl[k], s=sizes[k], color=colors[k],
                    edgecolor="white", linewidth=1.6, zorder=3, alpha=0.92)

    axC.annotate(f"{runs[0]}\n" + r"$s^{*}$ = " + f"{s_star[0]:+.3f}\nPPL = {ppl[0]:.2f}",
                 xy=(reliance[0], ppl[0]), xytext=(reliance[0] * 1.45, ppl[0] - 0.05),
                 ha="left", va="center", fontsize=9.0, color=ORANGE, fontweight="bold")
    axC.annotate(f"{runs[1]}\n" + r"$s^{*}$ = " + f"{s_star[1]:+.3f}\nPPL = {ppl[1]:.2f}",
                 xy=(reliance[1], ppl[1]), xytext=(reliance[1] * 0.80, ppl[1] + 0.62),
                 ha="right", va="bottom", fontsize=9.0, color=PURPLE, fontweight="bold")

    # guide arrow: more reliance -> lower PPL
    axC.annotate("", xy=(reliance[1] * 0.62, ppl[1] + 0.18),
                 xytext=(reliance[0] * 1.7, ppl[0] - 0.18),
                 arrowprops=dict(arrowstyle="->", color=GREY, lw=1.8,
                                 linestyle=(0, (4, 3))))
    axC.text(reliance.mean() * 0.5, ppl.mean() + 0.05,
             "more reverse-channel\nreliance -> lower PPL",
             fontsize=8.8, color=GREY, ha="center", va="center", style="italic")

    ratio = reliance[1] / reliance[0]
    axC.set_title("(C)  The model invests more in the reverse\n"
                  f"channel when it helps more  (~{ratio:.0f}x reliance)",
                  fontsize=11.5, color=PURPLE, fontweight="bold", pad=8)
    axC.set_xlabel(r"reverse-channel reliance  $|s^{*}|^{2}$", fontsize=10)
    axC.set_ylabel("TinyStories val PPL  (lower is better)", fontsize=10)
    axC.grid(True, alpha=0.3, linestyle=":")
    axC.invert_yaxis()
    for spine in ["top", "right"]:
        axC.spines[spine].set_visible(False)

    fig.suptitle("What the reverse channel buys: directed routing the conservative "
                 "core cannot represent",
                 fontsize=13.5, fontweight="bold", color="#212121", y=0.965)

    out = "reverse_channel_benefit.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    print("wrote", out)


if __name__ == "__main__":
    main()
