"""
Generate Figure 10.2 of Improving_the_Fock_Mechanism_to_match_Attention.md (section 10.9).

reverse_channel_benefit.png -- three panels (redrawn 2026-10-16; the first version showed the
coupling without the channel as symmetric, which a causal model cannot have):
  (A) Without the reverse channel the tokens feel only V_phi: a causal (strictly lower-triangular),
      sparse coupling, already directed by the mask, whose transferred content is locked to its
      strength (the gradient of a pair potential). The registers are written but never read.
  (B) With the reverse channel: A_ij = sum_k beta_ik alpha_kj through M registers -- causal (the bank
      read at position i is built from tokens <= i), rank <= M, content-routed with a decoupled value.
      A planted copy: one register stores what followed the first occurrence of a token; a later
      occurrence reads it.
  (C) Empirical anchor: across the two causal-fixed v2.1 runs the better model invests a far larger
      reverse-channel scale (real numbers from the training logs; pending re-certification, see the
      note's banner).
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


def vphi_coupling(n, k=3):
    """V_phi alone: each token is pulled by up to k earlier tokens (strictly j < i), sparse, directed."""
    A = np.zeros((n, n))
    for i in range(1, n):
        js = np.arange(i)
        w = rng.gamma(1.5, 0.5, size=i) * np.exp(-(i - js) / 4.0)
        top = js[np.argsort(w)[-k:]]
        A[i, top] = w[np.isin(js, top)]
    return A / A.max()


def register_coupling(n, M=4):
    """A_ij = sum_k beta_ik alpha^(i)_kj: the bank read at i is written from tokens j <= i."""
    S = rng.normal(0, 1.0, size=(M, n))          # write scores of register k for token j
    S[0, 4] = 6.0                                 # register 0 stores the token after the first "x" (j = 4)
    Bl = rng.normal(0, 1.0, size=(n, M))         # read scores of token i for register k
    Bl[10, 0] = 6.0                               # the second "x" (i = 10) reads register 0
    A = np.zeros((n, n))
    for i in range(n):
        al = np.exp(S[:, :i + 1]); al /= al.sum(1, keepdims=True)   # prefix softmax over j <= i
        be = np.exp(Bl[i]); be /= be.sum()
        A[i, :i + 1] = be @ al
    return A / A.max()


def main():
    fig = plt.figure(figsize=(15.2, 5.5), facecolor=BG)
    gs = GridSpec(1, 4, width_ratios=[1.0, 1.0, 0.05, 1.18], wspace=0.45,
                  left=0.05, right=0.985, top=0.78, bottom=0.14)

    A_v = vphi_coupling(T)
    A_r = register_coupling(T)

    cmap = "magma"

    # ---- Panel A: V_phi only (no reverse channel) ----
    axA = fig.add_subplot(gs[0, 0])
    axA.set_facecolor(BG)
    im = axA.imshow(A_v, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    axA.set_title("(A)  No reverse channel: tokens feel only\n"
                  r"$V_\phi$ (causal, already directed)",
                  fontsize=11.5, color=BLUE_DARK, fontweight="bold", pad=8)
    axA.set_xlabel("source token  j", fontsize=10)
    axA.set_ylabel("target token  i", fontsize=10)
    axA.text(0.5, -0.24, "content locked to the coupling strength;\n"
                         "registers written but never read",
             transform=axA.transAxes, ha="center", va="top", fontsize=9.5, color=GREY)

    # ---- Panel B: the register path (with reverse channel) ----
    axB = fig.add_subplot(gs[0, 1])
    axB.set_facecolor(BG)
    axB.imshow(A_r, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    axB.set_title("(B)  With reverse channel: through M registers\n"
                  r"$A_{ij}=\sum_k \beta_{ik}\,\alpha_{kj}$  (rank $\leq M$)",
                  fontsize=11.5, color=TEAL_DARK, fontweight="bold", pad=8)
    axB.set_xlabel("source token  j", fontsize=10)
    axB.set_ylabel("target token  i", fontsize=10)
    axB.annotate("copy via a register:\n2nd \"x\" (i = 10) reads what\nfollowed the 1st (j = 4)",
                 xy=(4.0, 10.0), xytext=(5.6, 1.6),
                 fontsize=8.6, color="white", fontweight="bold", ha="left",
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.6))
    axB.text(0.5, -0.24, "register content reaches the tokens; value\n"
                         "decoupled from key; softmax over registers",
             transform=axB.transAxes, ha="center", va="top", fontsize=9.5, color=GREY)

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
    axC.set_xlabel(r"reverse-channel reliance  $|s^{*}|^{2}$" + "\n(PPLs pending re-certification)", fontsize=10)
    axC.set_ylabel("TinyStories val PPL  (lower is better)", fontsize=10)
    axC.grid(True, alpha=0.3, linestyle=":")
    axC.invert_yaxis()
    for spine in ["top", "right"]:
        axC.spines[spine].set_visible(False)

    fig.suptitle("What the reverse channel buys: a read port that brings register content, "
                 "routed by content, into the tokens",
                 fontsize=13.5, fontweight="bold", color="#212121", y=0.965)

    out = "reverse_channel_benefit.png"
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
