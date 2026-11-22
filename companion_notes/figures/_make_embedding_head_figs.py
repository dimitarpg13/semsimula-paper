"""Figures for productization/Embedding_Head_Alignment_Test.md.

Generates five PNGs into figures/embedding_head/:

  eh_roles.png          -- the two roles of the token matrix: the input embedding places a token at its starting
                           position, the head reads the next token out of the final state; untied, two matrices;
                           tied, one.
  eh_alignment.png      -- (a) per frequency fifth, the cosine between a token's input and output rows in the
                           untied model: raw, after the best shared linear map (unweighted and frequency-weighted
                           fits), and the shuffled-row null; (b) the relative residual of the two fits, with the
                           no-map reference at 1.
  eh_distributions.png  -- per-token distributions of the mapped cosine for the rarest and the most frequent fifth,
                           against the shuffled-row null.
  eh_role_bands.png     -- (a) how well the tied model's shared rows are predicted from the untied input embedding
                           and from the untied head, by frequency band; (b) the tied model's loss change per fifth.
  eh_norms.png          -- median row norms per fifth: untied input embedding, untied head, tied shared matrix.

Numbers are read from results/frequency_quintiles/embedding_head_alignment_sr2np_vs_tied.json and
frequency_quintiles_sr2np_tied_gpt2.json (debug/embedding_head_alignment.py, debug/frequency_quintile_loss.py).
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

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
ORANGE = "#F59E0B"
INK = "#1F2937"
MUTED = "#6B7280"

HERE = Path(__file__).resolve().parent
OUT = HERE / "embedding_head"
RES = HERE.parent.parent / "notebooks/conservative_arch/scaleup/results/frequency_quintiles"
A = json.loads((RES / "embedding_head_alignment_sr2np_vs_tied.json").read_text())
Q = json.loads((RES / "frequency_quintiles_sr2np_tied_gpt2.json").read_text())
FIFTHS = ["Q0\nrarest", "Q1", "Q2", "Q3", "Q4\nmost frequent"]


def _spines(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def med(key):
    return [r["median"] for r in A["fifths"][key]]


def box(ax, x, y, w, h, text, fc, ec=INK, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, x0, y0, x1, y1, col=INK, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=col, lw=1.4,
                                 linestyle=ls))


def fig_roles():
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6))
    for ax, tied in zip(axes, (False, True)):
        ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
        box(ax, 0.1, 1.1, 1.4, 0.8, "token x", "#F3F4F6")
        box(ax, 2.2, 1.1, 2.3, 0.8, "starting position\nh0 = E[x] + P[t]", "#DBEAFE")
        box(ax, 5.3, 1.1, 2.5, 0.8, "two damped layer steps\n(potential, velocity)", "#EDE9FE")
        box(ax, 8.6, 1.1, 1.6, 0.8, "final state hL", "#DBEAFE")
        box(ax, 10.6, 1.1, 1.3, 0.8, "logits\nhL W^T + b", "#DCFCE7")
        for x0, x1 in ((1.5, 2.2), (4.5, 5.3), (7.8, 8.6), (10.2, 10.6)):
            arrow(ax, x0, 1.5, x1, 1.5)
        if not tied:
            box(ax, 2.35, 2.25, 2.0, 0.6, "input embedding E", "white", BLUE)
            box(ax, 10.35, 2.25, 1.6, 0.6, "head W", "white", GREEN)
            arrow(ax, 3.35, 2.25, 3.35, 1.92, BLUE); arrow(ax, 11.15, 2.25, 11.15, 1.92, GREEN)
            ax.set_title("Untied (SR2-NP): two matrices, each free to take its own geometry", loc="left")
        else:
            box(ax, 5.3, 2.25, 2.5, 0.6, "one shared matrix S = E = W", "white", PURPLE)
            arrow(ax, 5.3, 2.55, 3.35, 1.92, PURPLE); arrow(ax, 7.8, 2.55, 11.15, 1.92, PURPLE)
            ax.set_title("Tied (SR2-NP-T): one matrix places the token and reads the next one out", loc="left")
            ax.text(6.55, 0.45, "if S keeps the input geometry, the layer steps must carry the state further,\n"
                    "from an input-like start to the output direction of the next token",
                    ha="center", va="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "eh_roles.png", bbox_inches="tight")
    plt.close(fig)


def fig_alignment():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(5); w = 0.2
    series = [("cos(E, W) raw", "raw", GREY),
              ("cos(EA, W) mapped", "after the best shared map", BLUE),
              ("cos(EA, W) mapped, freq-weighted fit", "after a map fitted by frequency", PURPLE),
              ("null: cos(E_i A, W_j) shuffled", "shuffled rows (null)", INK)]
    for j, (key, label, col) in enumerate(series):
        a.bar(x + (j - 1.5) * w, med(key), w * 0.92, color=col, label=label)
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xticks(x, FIFTHS, fontsize=9)
    a.set_ylabel("median cosine over the fifth's token types")
    a.set_title("(a) Input row against output row, SR2-NP")
    a.set_ylim(-0.05, 0.75)
    a.legend(fontsize=8.5, frameon=False, loc="upper left")
    _spines(a)

    for key, label, col in (("residual mapped", "best shared map, every type alike", BLUE),
                            ("residual mapped, freq-weighted fit", "map fitted by training frequency", PURPLE)):
        b.plot(x, med(key), "-o", color=col, lw=2, label=label)
    b.axhline(1.0, color=RED, ls="--", lw=1.2)
    b.text(4.15, 1.02, "no map at all", color=RED, fontsize=9, ha="right", va="bottom")
    b.set_xticks(x, FIFTHS, fontsize=9)
    b.set_ylabel("median relative residual |EA − W| / |W|")
    b.set_title("(b) What no shared map can reconcile")
    b.set_ylim(0.3, 1.25)
    b.legend(fontsize=8.5, frameon=False, loc="lower left")
    _spines(b)
    fig.tight_layout()
    fig.savefig(OUT / "eh_alignment.png", bbox_inches="tight")
    plt.close(fig)


def fig_distributions():
    h = A["hist"]; bins = np.array(h["bins"]); c = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for key, label, col, ls in (("mapped_Q0", "rarest fifth (Q0)", ORANGE, "-"),
                                ("mapped_Q4", "most frequent fifth (Q4)", BLUE, "-"),
                                ("null_Q4", "shuffled rows (null, Q4)", INK, "--")):
        y = np.array(h[key], float); y /= y.sum() * (bins[1] - bins[0])
        ax.plot(c, y, ls, color=col, lw=2, label=label)
    ax.set_xlabel("cosine between E_x A and W_x (rows mean-centred)")
    ax.set_ylabel("density over the fifth's token types")
    ax.set_title("Per-token alignment after the best shared linear map, SR2-NP")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    _spines(ax)
    fig.tight_layout()
    fig.savefig(OUT / "eh_distributions.png", bbox_inches="tight")
    plt.close(fig)


def fig_role_bands():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    bands = A["bands"]
    labels = ["top 100", "100 to 1,000", "1,000 to 10,051", "10,051 to 50,257"]
    x = np.arange(len(bands)); w = 0.36
    a.bar(x - w / 2, [r["r2_from_E"] for r in bands], w, color=BLUE, label="from the untied input embedding E")
    a.bar(x + w / 2, [r["r2_from_W"] for r in bands], w, color=GREEN, label="from the untied head W")
    for xi, r in zip(x, bands):
        a.text(xi, max(r["r2_from_E"], r["r2_from_W"]) + 0.02, f"{100 * r['share']:.0f}% of\ntraining tokens",
               ha="center", va="bottom", fontsize=8.5, color=MUTED)
    a.set_xticks(x, labels, fontsize=9)
    a.set_xlabel("frequency rank of the token type")
    a.set_ylabel("R² of the tied rows, pooled over the band")
    a.set_title("(a) Which role the shared matrix took")
    a.set_ylim(0, 0.6)
    a.legend(fontsize=8.5, frameon=False, loc="upper right")
    _spines(a)

    pr = next(iter(Q["paired"].values()))["quintiles"]
    d = [r["diff"] for r in pr]; se = [r["se"] for r in pr]
    cols = [GREEN if v < 0 else RED for v in d]
    b.bar(np.arange(5), d, 0.6, color=cols, yerr=se, capsize=4, ecolor=INK)
    b.axhline(0, color=MUTED, lw=0.8)
    b.set_xticks(np.arange(5), FIFTHS, fontsize=9)
    b.set_ylabel("tied minus untied mean loss (nats)")
    b.set_title("(b) Where the tie gains and where it pays")
    b.text(3.55, d[4] + 0.005, f"+{d[4]:.3f}, 89% of targets", ha="right", va="center", fontsize=8.5, color=INK)
    b.set_ylim(-0.58, 0.12)
    _spines(b)
    fig.tight_layout()
    fig.savefig(OUT / "eh_role_bands.png", bbox_inches="tight")
    plt.close(fig)


def fig_norms():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(5)
    for vals, label, col in ((med("norm E"), "untied input embedding E", BLUE),
                             (med("norm W"), "untied head W", GREEN),
                             (A["norm_shared"], "tied shared matrix S", PURPLE)):
        ax.plot(x, vals, "-o", color=col, lw=2, label=label)
    ax.set_xticks(x, FIFTHS, fontsize=9)
    ax.set_ylabel("median row norm (rows mean-centred)")
    ax.set_title("Row norms by frequency fifth")
    ax.set_ylim(0, 3.4)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    _spines(ax)
    fig.tight_layout()
    fig.savefig(OUT / "eh_norms.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_roles()
    fig_alignment()
    fig_distributions()
    fig_role_bands()
    fig_norms()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
