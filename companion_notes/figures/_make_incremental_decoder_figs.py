"""Figures for productization/Incremental_Decoder.md.

Generates PNGs into figures/incremental_decoder/:

  dec_pipeline.png  -- where the decoder sits in text generation: the prompt is read once (prefill), which fills the
                       cache; then each new token is one decode step that reads the cache and appends to it.
  dec_step.png      -- one decode step of the conservative core (two layers) beside one of the matched GPT-2: what
                       reads the cache and what acts within the new token's own position.
  dec_cost.png      -- the cache per token and the time per generated token against the context length, the
                       conservative core against the matched GPT-2.

Numbers in dec_cost.png: cache sizes from parf/incremental_decoder.py (cache_numbers_per_token); times per token from
results/productization/decode_bench_d_gpt2_{1,2,3}.txt (GPT-2, mean of three, 2026-11-17) and
decode_ab_s1_masked_pruned.txt (SR2-NP, layer 1 pruned, before and after Stage 2b step 1, 2026-11-23). The GPT-2 runs
and the SR2-NP runs come from different sessions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

GREEN, BLUE, RED, PURPLE, GREY, ORANGE = "#22C55E", "#3B82F6", "#EF4444", "#8B5CF6", "#9CA3AF", "#F59E0B"
INK, MUTED = "#1F2937", "#6B7280"

HERE = Path(__file__).resolve().parent
OUT = HERE / "incremental_decoder"
OUT.mkdir(exist_ok=True)


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=10, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color=INK, ls="-", lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))


def pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 4.6); ax.axis("off")
    box(ax, 0.2, 2.9, 1.7, 0.9, "prompt text\n→ token ids")
    box(ax, 2.5, 2.9, 2.4, 0.9, "prefill\none forward over\nthe whole prompt", fc="#EFF6FF", ec=BLUE, bold=True)
    box(ax, 5.6, 2.9, 1.9, 0.9, "logits of the\nnext token")
    box(ax, 8.1, 2.9, 1.5, 0.9, "choose a\ntoken")
    box(ax, 10.0, 2.9, 1.3, 0.9, "token ids\n→ text")
    box(ax, 4.6, 0.35, 2.9, 1.1, "decode step\nthe new token alone,\nat its position", fc="#F0FDF4", ec=GREEN, bold=True)
    box(ax, 0.6, 0.35, 2.9, 1.1, "cache\nper layer: each earlier\ntoken's key and state", fc="#F9FAFB", ec=MUTED)
    arrow(ax, 1.9, 3.35, 2.5, 3.35); arrow(ax, 4.9, 3.35, 5.6, 3.35); arrow(ax, 7.5, 3.35, 8.1, 3.35)
    arrow(ax, 9.6, 3.35, 10.0, 3.35)
    arrow(ax, 8.85, 2.9, 7.5, 0.95, rad=-0.15)                       # chosen token -> decode step
    arrow(ax, 6.05, 1.45, 6.55, 2.9)                                 # decode step -> logits
    arrow(ax, 3.2, 2.9, 2.4, 1.45, color=BLUE)                       # prefill fills the cache
    ax.text(2.25, 2.25, "fills", color=BLUE, fontsize=9, ha="right")
    arrow(ax, 3.5, 1.05, 4.6, 1.05, color=MUTED)                     # cache -> decode step (read)
    arrow(ax, 4.6, 0.65, 3.5, 0.65, color=GREEN)                     # decode step -> cache (append)
    ax.text(4.05, 1.13, "read", color=MUTED, fontsize=9, ha="center", va="bottom")
    ax.text(4.05, 0.57, "append", color=GREEN, fontsize=9, ha="center", va="top")
    ax.text(9.55, 1.45, "repeat until\nthe text ends", fontsize=9, color=MUTED, ha="left")
    ax.text(3.7, 4.25, "time to the first token: the prefill", fontsize=9.5, color=BLUE, ha="center")
    ax.text(6.05, 0.05, "time per further token: one decode step", fontsize=9.5, color=GREEN, ha="center")
    ax.set_title("Text generation with an incremental decoder", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "dec_pipeline.png"); plt.close(fig)


def step():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), gridspec_kw={"width_ratios": [1.35, 1]})
    # left: the conservative core
    ax = axes[0]; ax.set_xlim(0, 7.2); ax.set_ylim(0, 7.0); ax.axis("off")
    ax.set_title("Conservative core (SR2-NP), one decode step", loc="left")
    box(ax, 1.6, 0.2, 4.0, 0.6, "embedding of the new token at position t")
    for li, y0 in enumerate((1.25, 3.85)):
        ax.add_patch(FancyBboxPatch((0.15, y0 - 0.1), 6.9, 2.35, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc="#FAFAFA", ec=GREY, lw=1.0, ls="--"))
        ax.text(0.3, y0 + 2.05, f"layer {li}", fontsize=10, color=MUTED, fontweight="bold")
        box(ax, 0.35, y0 + 0.1, 2.0, 1.65, "context channels ξ\n5 channels:\nsoftmax over the\ncached tokens", fc="#FFF7ED", ec=ORANGE)
        box(ax, 2.75, y0 + 0.95, 1.9, 0.8, "wells Vθ read\nh_t and ξ_t", fc="#F5F3FF", ec=PURPLE)
        box(ax, 2.75, y0 + 0.1, 1.9, 0.7, "stiff modes\n(eigendecomposition)", fc="#F5F3FF", ec=PURPLE, fs=9)
        box(ax, 4.95, y0 + 0.1, 1.95, 1.65, "integrator:\nexact damped flow\nof the stiff modes,\nkick, friction\n→ h_t at the next layer", fc="#F5F3FF", ec=PURPLE, fs=9)
        arrow(ax, 2.35, y0 + 1.0, 2.75, y0 + 1.3); arrow(ax, 3.7, y0 + 0.95, 3.7, y0 + 0.8)
        arrow(ax, 4.65, y0 + 0.45, 4.95, y0 + 0.9); arrow(ax, 4.65, y0 + 1.35, 4.95, y0 + 1.1)
    arrow(ax, 3.6, 0.8, 3.6, 1.15); arrow(ax, 3.6, 3.5, 3.6, 3.75)   # embedding -> layer 0 -> layer 1
    ax.text(3.7, 3.62, "h_t", fontsize=9, color=INK, va="center")
    box(ax, 4.95, 6.35, 1.95, 0.55, "output head → logits")
    arrow(ax, 5.92, 6.1, 5.92, 6.35)
    ax.text(0.35, 6.55, "orange: reads the cache (keys and states\nof all earlier tokens, per layer)\npurple: the new token's own position only",
            fontsize=8.5, color=INK, va="center")
    # right: the matched GPT-2
    ax = axes[1]; ax.set_xlim(0, 5.0); ax.set_ylim(0, 7.0); ax.axis("off")
    ax.set_title("Matched GPT-2, one decode step", loc="left")
    box(ax, 0.9, 0.2, 3.2, 0.6, "token + position embedding")
    ax.add_patch(FancyBboxPatch((0.25, 1.15), 4.5, 3.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#FAFAFA", ec=GREY, lw=1.0, ls="--"))
    ax.text(0.4, 4.15, "each of 8 layers", fontsize=10, color=MUTED, fontweight="bold")
    box(ax, 0.45, 2.55, 2.0, 1.3, "attention:\nnew query against\ncached keys and\nvalues (6 heads)", fc="#FFF7ED", ec=ORANGE)
    box(ax, 2.75, 2.55, 1.8, 1.3, "MLP\n(own position)", fc="#F5F3FF", ec=PURPLE)
    box(ax, 0.45, 1.35, 4.1, 0.85, "LayerNorm, residual additions (own position)", fc="#F5F3FF", ec=PURPLE, fs=9)
    arrow(ax, 2.45, 3.2, 2.75, 3.2); arrow(ax, 2.5, 0.8, 2.5, 1.35); arrow(ax, 2.5, 4.45, 2.5, 5.1)
    box(ax, 1.25, 5.1, 2.5, 0.6, "output head → logits")
    ax.text(0.3, 6.35, "cache per token: keys and values,\n2 × 384 numbers in each of 8 layers = 6,144\n(the core: 5 × 48 keys + 384 state,\nin each of 2 layers = 1,248)",
            fontsize=8.5, color=INK, va="center")
    fig.tight_layout(); fig.savefig(OUT / "dec_step.png"); plt.close(fig)


def cost():
    ctx = [512, 2048, 8192]
    gpt2_ms = [1e3 / 122, 1e3 / 107, 1e3 / 68]          # decode_bench_d_gpt2_{1,2,3}, mean tokens per second
    core_ms = [27.97, 29.38, 37.55]                     # decode_ab_s1_masked_pruned.txt, A (before step 1)
    step1_ms = [19.82, 21.43, 30.33]                    # the same, B (after step 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    a.bar([0, 1], [1248, 6144], 0.55, color=[GREEN, BLUE])
    a.set_xticks([0, 1], ["conservative core\n(2 layers)", "matched GPT-2\n(8 layers)"])
    for x, v in ((0, 1248), (1, 6144)):
        a.text(x, v + 120, f"{v:,}", ha="center", fontsize=10)
    a.set_ylabel("numbers cached per token"); a.set_title("(a) Cache per token")
    a.spines[["top", "right"]].set_visible(False)
    for ys, col, lab, mk in ((core_ms, GREY, "core, before step 1", "o"), (step1_ms, GREEN, "core, after step 1", "s"),
                             (gpt2_ms, BLUE, "matched GPT-2", "D")):
        b.plot(ctx, ys, color=col, marker=mk, lw=2, ms=7)
        b.annotate(lab, (ctx[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9.5)
    b.set_xscale("log", base=2); b.set_xticks(ctx, ["512", "2,048", "8,192"]); b.set_xlim(400, 30000)
    b.set_ylim(0, 42); b.set_xlabel("context length (tokens)"); b.set_ylabel("ms per generated token")
    b.set_title("(b) Time per generated token (CPU, batch 1)"); b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "dec_cost.png"); plt.close(fig)


if __name__ == "__main__":
    pipeline(); step(); cost()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
