"""Figures for productization/Training_Prefill_and_Decoding.md.

Generates PNGs into figures/training_prefill_decoding/:

  tpd_pipeline.png  -- the three phases: training on batches of whole sequences (forward over every position at once,
                       loss, backward, update), then, deployed, the prompt read once (prefill) and one token per decode
                       step; how many tokens each call processes and what bounds it.
  tpd_shapes.png    -- the shape of the context-channel computation in each phase: the weight tensor each builds.
  tpd_read.png      -- one decode step's read of the cached states at context n: how often each form streams the
                       states (broadcast over the 5 channels, 30 heads of 64, one product), with the megabytes at 8,192.
  tpd_levers.png    -- every measured lever against every phase: the paired time ratio, or n/a, or not measured.
  tpd_training.png  -- for training: (a) the memory of the explicit channel weights against the sequence length, and
                       (b) the float32 / bfloat16 error scale of the folded distance key, 3 u lambda T, with the
                       measured points.

Numbers: Stage2b_Decode_Step_Levers.md (decode, the prefill checks, the training estimate), Stage1_CPU_Levers.md §8
(30 heads of 64 against the folded form, forward-level), Stage1b_Context_Channels_as_30_Heads.md §7 (error),
Incremental_Decoder.md (cache), all 2026-11-17 to 2026-11-23.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})

GREEN, BLUE, RED, PURPLE, GREY, ORANGE = "#22C55E", "#3B82F6", "#EF4444", "#8B5CF6", "#9CA3AF", "#F59E0B"
INK, MUTED = "#1F2937", "#6B7280"

HERE = Path(__file__).resolve().parent
OUT = HERE / "training_prefill_decoding"
OUT.mkdir(exist_ok=True)


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color=INK, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.3,
                                 connectionstyle=f"arc3,rad={rad}"))


def pipeline():
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 5.6); ax.axis("off")
    # training lane
    ax.add_patch(FancyBboxPatch((0.15, 3.05), 12.2, 2.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#EFF6FF", ec=BLUE, lw=1.0, ls="--"))
    ax.text(0.35, 5.05, "1  Training (GPU, float32)", fontsize=11, fontweight="bold", color=BLUE)
    box(ax, 0.4, 3.35, 1.9, 1.2, "corpus\n→ batch of B\nsequences of\nT = 512 tokens")
    box(ax, 2.8, 3.35, 2.6, 1.2, "forward over every\nposition at once;\nthe causal mask hides\nlater tokens", fc="white", ec=BLUE)
    box(ax, 5.9, 3.35, 1.8, 1.2, "loss: next-token\ncross-entropy\nat every position")
    box(ax, 8.2, 3.35, 1.8, 1.2, "backward:\ngradients of\nall parameters")
    box(ax, 10.5, 3.35, 1.6, 1.2, "optimizer\nupdate")
    for x0, x1 in ((2.3, 2.8), (5.4, 5.9), (7.7, 8.2), (10.0, 10.5)):
        arrow(ax, x0, 3.95, x1, 3.95)
    arrow(ax, 11.3, 4.55, 1.35, 4.55, color=MUTED, rad=0.06)
    ax.text(6.3, 4.98, "repeat for 32,500 steps", fontsize=8.5, color=MUTED, ha="center")
    # deployment lane
    ax.add_patch(FancyBboxPatch((0.15, 0.2), 12.2, 2.45, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#F0FDF4", ec=GREEN, lw=1.0, ls="--"))
    ax.text(0.35, 2.35, "Deployed (CPU here, one sequence)", fontsize=11, fontweight="bold", color=GREEN)
    box(ax, 0.4, 0.55, 1.9, 1.2, "prompt\n→ token ids")
    box(ax, 2.8, 0.55, 2.6, 1.2, "2  Prefill: one forward\nover the prompt,\nfills the cache", fc="white", ec=GREEN, bold=True)
    box(ax, 5.9, 0.55, 2.4, 1.2, "3  Decode step: the\nnew token alone,\nreads the cache", fc="white", ec=GREEN, bold=True)
    box(ax, 8.8, 0.55, 1.6, 1.2, "logits →\nchoose a token")
    box(ax, 10.9, 0.55, 1.2, 1.2, "token ids\n→ text")
    for x0, x1 in ((2.3, 2.8), (5.4, 5.9), (8.3, 8.8), (10.4, 10.9)):
        arrow(ax, x0, 1.15, x1, 1.15)
    arrow(ax, 9.6, 1.75, 7.1, 1.75, color=MUTED, rad=0.35)
    ax.text(8.35, 2.28, "one decode step per new token", fontsize=8.5, color=MUTED, ha="center")
    # the trained weights flow down
    arrow(ax, 11.3, 3.35, 11.3, 2.65, color=PURPLE)
    ax.text(11.42, 2.85, "trained weights", fontsize=8.5, color=PURPLE, va="center")
    fig.suptitle("Three phases: tokens per call  B × 512  →  T (the prompt)  →  1", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(OUT / "tpd_pipeline.png"); plt.close(fig)


def shapes():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5))
    specs = [("1  Training", "B × 5 × T × T weights per layer,\nkept for the backward;\nB sequences of T = 512", BLUE, 512, 512, 3),
             ("2  Prefill", "5 × T × T weights per layer;\nT the prompt length\n(here up to 8,192)", GREEN, 512, 512, 1),
             ("3  Decode step", "5 × 1 × n weights per layer:\none row per channel over\nthe n cached tokens", ORANGE, 512, 512, 1)]
    for ax, (title, sub, col, H, W, depth) in zip(axes, specs):
        ax.set_xlim(-60, 640); ax.set_ylim(-150, 640); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, loc="left")
        for k in range(5 if title != "3  Decode step" else 5):
            off = 14 * k
            if title == "3  Decode step":
                ax.add_patch(Rectangle((off, 470 + off), W, 22, fc=col, ec="white", lw=1.0, alpha=0.85))
            else:
                ax.add_patch(Rectangle((off, off), W, H, fc=col, ec="white", lw=1.0, alpha=0.30 + 0.12 * k))
                tri = plt.Polygon([(off + W, off), (off + W, off + H), (off, off + H)], closed=True, fc="white", alpha=0.55, ec="none")
                ax.add_patch(tri)                                     # the causal mask: only s <= t
        if title == "1  Training":
            for b in range(1, depth):
                ax.add_patch(Rectangle((56 + 22 * b, 56 + 22 * b), W, H, fc="none", ec=col, lw=0.8, ls=":", zorder=0))
            ax.text(W - 40, H + 125, "× B sequences", fontsize=9, color=col)
        if title == "3  Decode step":
            ax.text(0, 440, "query: the new token", fontsize=9, color=INK)
            ax.annotate("", xy=(W + 56, 585), xytext=(0, 585), arrowprops=dict(arrowstyle="<->", color=MUTED))
            ax.text(W / 2 + 28, 600, "n cached tokens", fontsize=9, color=MUTED, ha="center")
        else:
            ax.text(W / 2 + 28, -40, "earlier token s", fontsize=9, color=MUTED, ha="center")
            ax.text(-45, H / 2 + 28, "token t", fontsize=9, color=MUTED, rotation=90, va="center")
        ax.text(0, -140 if title != "3  Decode step" else 330, sub, fontsize=9.2, color=INK, va="bottom" if title == "3  Decode step" else "top")
    fig.suptitle("The context-channel weights each phase builds (5 channels; white: masked, s > t)", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(OUT / "tpd_shapes.png"); plt.close(fig)


def read():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    a.set_xlim(0, 10); a.set_ylim(0, 6.2); a.axis("off")
    rows = [("broadcast (the original read)", 5, ORANGE, "5 row × matrix products,\neach streams all states"),
            ("30 heads of 64 (folded keys)", 5, RED, "each channel's 6 heads stream\nits 6 blocks: all states, 5 times"),
            ("one product, (5 × n)(n × 384)", 1, GREEN, "the 5 weight rows together:\nthe states streamed once")]
    for i, (lab, reads, col, note) in enumerate(rows):
        y = 4.6 - 2.0 * i
        a.text(0.0, y + 0.95, lab, fontsize=10, fontweight="bold", color=INK)
        a.add_patch(Rectangle((0.0, y - 0.25), 2.2, 0.9, fc="#F3F4F6", ec=MUTED))
        a.text(1.1, y + 0.2, "cached states\nn × 384", ha="center", va="center", fontsize=8.5)
        for r in range(reads):
            yy = y + 0.55 - 0.18 * r
            a.annotate("", xy=(4.0, yy if reads > 1 else y + 0.2), xytext=(2.25, yy if reads > 1 else y + 0.2),
                       arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4))
        a.add_patch(Rectangle((4.05, y - 0.25), 1.6, 0.9, fc="white", ec=col))
        a.text(4.85, y + 0.2, "weighted\nsum", ha="center", va="center", fontsize=8.5)
        a.text(6.0, y + 0.2, note, fontsize=8.8, color=INK, va="center")
    a.set_title("(a) How a decode step reads the cached states, per layer", loc="left")
    n, d = 8192, 384
    mb = [5 * n * d * 4 / 1e6, 5 * n * d * 4 / 1e6, n * d * 4 / 1e6]
    b.barh([2, 1, 0], mb, color=[ORANGE, RED, GREEN], height=0.55)
    b.set_yticks([2, 1, 0], ["broadcast", "30 heads", "one product"])
    for y, v in zip([2, 1, 0], mb):
        b.text(v + 1, y, f"{v:.1f} MB" if v < 20 else f"{v:.0f} MB", va="center", fontsize=10)
    b.set_xlim(0, 80); b.set_xlabel("states read per layer per step at n = 8,192 (MB)")
    b.spines[["top", "right"]].set_visible(False)
    b.set_title("(b) States read per layer and step", loc="left")
    b.text(0, -0.85, "measured with one product: 24.4 to 16.7 ms per step at 8,192", fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "tpd_read.png"); plt.close(fig)


def levers():
    cols = ["training\nstep", "prefill\n512", "prefill\n8,192", "decode\n512", "decode\n8,192"]
    # value: a time ratio (B/A, lower is faster); 'na' = cannot act there; 'nm' = not measured
    rows = [("series only where used (step 1)", ["na", 1.01, 0.99, 0.71, 0.81]),
            ("12-term series", [0.99, "nm", "nm", 0.90, 0.92]),
            ("products as multiply-and-sum (step 2)", ["na", 1.00, 1.00, 0.90, 0.94]),
            ("compiled core (step 3)", ["nm", 0.85, 0.92, 0.77, 0.86]),
            ("cache read as one product (step 4)", ["na", "na", "na", 0.99, 0.69]),
            ("folded read, one fused call", ["nm", "nm", "nm", 1.00, 1.05]),
            ("30 heads of 64 (against the folded form)", ["nm", 0.99, 0.85, "nm", "nm"])]
    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(rows)); ax.invert_yaxis(); ax.axis("off")
    for j, c in enumerate(cols):
        ax.text(j + 0.5, -0.12, c, ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for i, (lab, vals) in enumerate(rows):
        ax.text(-0.08, i + 0.5, lab, ha="right", va="center", fontsize=9.5)
        for j, v in enumerate(vals):
            if v == "na":
                fc, txt, tc = "white", "n/a", MUTED
            elif v == "nm":
                fc, txt, tc = "#F3F4F6", "not\nmeasured", MUTED
            else:
                fc = (GREEN if v <= 0.80 else "#86EFAC" if v <= 0.95 else "#E5E7EB" if v < 1.03 else "#FCA5A5")
                txt, tc = f"{v:.2f}", INK
            ax.add_patch(Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, fc=fc, ec="white"))
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=9 if v in ("na", "nm") else 11, color=tc)
    ax.text(0, len(rows) + 0.3, "Paired time ratio of each lever against the configuration before it (lower is faster).\n"
            "Dark green: at least 20% faster; light green: 5 to 20%; grey: within the noise; red: slower.",
            fontsize=8.5, color=MUTED, va="top")
    fig.suptitle("Which lever acts in which phase", fontweight="bold", x=0.02, ha="left")
    fig.subplots_adjust(left=0.33, right=0.98, top=0.86, bottom=0.12)
    fig.savefig(OUT / "tpd_levers.png"); plt.close(fig)


def training():
    T = np.array([256, 512, 1024, 2048, 4096, 8192])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    for B, col, mk in ((16, BLUE, "o"), (32, PURPLE, "s")):
        mem = 2 * B * 5 * T.astype(float) ** 2 * 4 / 2 ** 30         # logits and weights, float32, per layer
        a.plot(T, mem, color=col, marker=mk, lw=2, ms=6)
        a.annotate(f"explicit, B = {B}", (T[-1], mem[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    fused = 32 * 30 * T * 4 / 2 ** 30                               # a fused kernel keeps one statistic per row and head
    a.plot(T, fused, color=GREEN, marker="D", lw=2, ms=6)
    a.annotate("fused kernel, B = 32:\nrow statistics only", (T[-1], fused[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    a.axvline(512, color=MUTED, lw=0.8, ls=":"); a.text(560, 30, "trained at T = 512", fontsize=8.5, color=MUTED)
    a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xticks(T, [f"{t:,}" for t in T]); a.set_xlim(200, 40000)
    a.set_xlabel("sequence length T"); a.set_ylabel("GB per layer (float32)")
    a.set_title("(a) Channel weights kept for the backward", loc="left"); a.spines[["top", "right"]].set_visible(False)
    lam = 0.69                                                      # the fastest channel, decay about 0.5
    for u, col, lab in ((2 ** -24, BLUE, "float32"), (2 ** -9, RED, "bfloat16")):
        b.plot(T, 3 * u * lam * T, color=col, lw=2)
        b.annotate(lab, (T[-1], 3 * u * lam * T[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9.5)
    b.plot([2048, 8192, 8192], [2.3e-4, 7.8e-4, 9.4e-4], ls="none", marker="o", color=INK, ms=6)
    b.annotate("measured, channel outputs\n(Stage1b, float32)", (8192, 9.4e-4), xytext=(-150, 25), textcoords="offset points",
               fontsize=8.5, arrowprops=dict(arrowstyle="-", color=MUTED))
    b.plot([512, 2048, 8192], [1.1e-4, 3.9e-4, 1.9e-3], ls="none", marker="s", color=ORANGE, ms=6)
    b.annotate("measured, decode logits\n(step 4, centred)", (512, 1.1e-4), xytext=(-20, 120), textcoords="offset points",
               fontsize=8.5, arrowprops=dict(arrowstyle="-", color=MUTED))
    b.set_xscale("log", base=2); b.set_yscale("log"); b.set_xticks(T, [f"{t:,}" for t in T]); b.set_xlim(200, 40000)
    b.set_xlabel("sequence length T"); b.set_ylabel("error scale 3 u λ T")
    b.set_title("(b) Error of the folded distance key", loc="left"); b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "tpd_training.png"); plt.close(fig)


if __name__ == "__main__":
    pipeline(); shapes(); read(); levers(); training()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
