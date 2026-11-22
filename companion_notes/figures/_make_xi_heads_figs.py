"""Figures for productization/Stage1b_Context_Channels_as_30_Heads.md.

Generates five PNGs into figures/xi_heads/:

  xh_layout.png     -- the two layouts of the folded context channels: five
                       heads with queries and keys of width 49 and values of
                       width 384 (the fused call falls back to the path that
                       builds the score matrix), against 30 heads of width 64
                       (the tiled kernel).
  xh_time_memory.png -- (a) seconds per call against the length, every layout
                       and one layer of the matched GPT-2's attention;
                       (b) peak memory above the loaded checkpoint, with the
                       size of one score tensor.
  xh_widths.png     -- (a) seconds per call against the head width at 4,096
                       and 8,192 tokens; (b) multiply-adds per layer, split
                       into the useful score work, the redundant score work
                       (padding and repetition), the value work and, for the
                       fallback path, the half above the diagonal.
  xh_error.png      -- (a) the largest difference from the float64 reference
                       per channel against the length; (b) the same against
                       the rounding scale u * lambda_c * T.
  xh_breakdown.png  -- the forward pass at 8,192 tokens by component: folded
                       without and with the subnormal flush, then 30 heads.

Numbers are read from results/productization/xi_head_width_bench.json and the
cpu_bench_*_timers.json files named below.
"""

import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

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
CHANNEL = ["#1D4ED8", "#3B82F6", "#8B5CF6", "#F59E0B", "#9CA3AF"]   # fastest to slowest decay

HERE = Path(__file__).resolve().parent
OUT = HERE / "xi_heads"
RES = HERE.parent.parent / "notebooks/conservative_arch/scaleup/results/productization"
U = 2.0 ** -24                    # float32 unit roundoff

B = json.loads((RES / "xi_head_width_bench.json").read_text())
K, DK, D = B["K"], B["d_k"], B["d"]
LAM = [-x for x in B["log_alpha"]]


def _spines(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def timing(variant):
    rows = sorted((r for r in B["timing"] if r["variant"] == variant), key=lambda r: r["T"])
    base = next(r["peak_rss_gb"] for r in rows if r["T"] == 0)
    rows = [r for r in rows if r["T"] > 0]
    return (np.array([r["T"] for r in rows]), np.array([r["median_s"] for r in rows]),
            np.array([r["peak_rss_gb"] - base for r in rows]))


def fig_layout():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    ax = axes[0]
    ax.set_title("Folded form: 5 heads, widths 49 and 384")
    for c in range(K):
        y = 8.6 - 1.55 * c
        ax.add_patch(Rectangle((0.3, y), 0.48 * 2, 0.9, fc=CHANNEL[c], ec="white", alpha=0.85))
        ax.add_patch(Rectangle((0.3 + 0.96, y), 0.04 * 2, 0.9, fc=INK, ec="white"))
        ax.add_patch(Rectangle((1.75, y), 3.84 * 1.2, 0.9, fc=CHANNEL[c], ec="white", alpha=0.35))
        ax.text(0.2, y + 0.45, f"channel {c}", ha="right", va="center", fontsize=9, color=MUTED)
    ax.text(0.8, 9.95, "query, key\n48 + 1", ha="center", va="center", fontsize=9, color=INK)
    ax.text(4.05, 9.75, "value: the state, 384", ha="center", va="center", fontsize=9, color=INK)
    ax.text(5.0, 0.55, "unequal widths: the fused call falls back to\nthe path that builds the T x T score matrix",
            ha="center", va="center", fontsize=10, color=RED)

    ax = axes[1]
    ax.set_title("Equal widths: 30 heads of 64")
    for c in range(K):
        y = 8.6 - 1.55 * c
        for b in range(6):
            x = 0.3 + b * 1.6
            ax.add_patch(Rectangle((x, y), 0.48, 0.9, fc=CHANNEL[c], ec="white", alpha=0.85))
            ax.add_patch(Rectangle((x + 0.48, y), 0.02, 0.9, fc=INK, ec=INK, lw=0.5))
            ax.add_patch(Rectangle((x + 0.5, y), 0.14, 0.9, fc="white", ec=GREY, lw=0.6, hatch="////"))
            ax.add_patch(Rectangle((x + 0.66, y), 0.64, 0.9, fc=CHANNEL[c], ec="white", alpha=0.35))
    ax.text(0.62, 9.95, "query, key\n48 + 1 + 15 zeros", ha="center", va="center", fontsize=9, color=INK)
    ax.text(5.4, 9.75, "value block b = 1, ..., 6: 64 of the 384 state components",
            ha="center", va="center", fontsize=9, color=INK)
    ax.text(5.0, 0.55, "every head 64 wide: the tiled kernel, no score matrix;\n"
            "each channel's scores are computed once per value block",
            ha="center", va="center", fontsize=10, color=BLUE)
    fig.tight_layout()
    fig.savefig(OUT / "xh_layout.png", bbox_inches="tight")
    plt.close(fig)


def fig_time_memory():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    styles = [("fold", "folded, widths 49 / 384", RED, "-", "o"),
              ("eq64", "30 heads of 64", BLUE, "-", "o"),
              ("eq96", "20 heads of 96", GREY, "--", "s"),
              ("eq128", "15 heads of 128", GREY, ":", "^"),
              ("eq384", "5 heads of 384", GREY, "-.", "D"),
              ("gpt2layer", "matched GPT-2, one layer (6 heads of 64)", INK, "--", "x")]
    for v, label, col, ls, mk in styles:
        T, s, _ = timing(v)
        a.plot(T, s, ls=ls, marker=mk, color=col, lw=2, ms=6, label=label)
    a.set_xscale("log", base=2); a.set_yscale("log")
    a.set_xlabel("length T (tokens)"); a.set_ylabel("seconds per call, one layer")
    a.set_title("(a) Time per call (subnormals flushed)")
    a.legend(fontsize=8.5, frameon=False, loc="upper left")
    a.grid(True, which="major", color="#E5E7EB", lw=0.8)
    _spines(a)

    T, _, m_fold = timing("fold")
    _, _, m_eq = timing("eq64")
    Tf = np.geomspace(T[0], T[-1], 50)
    b.plot(T, m_fold, "-o", color=RED, lw=2, label="folded, widths 49 / 384")
    b.plot(T, m_eq, "-o", color=BLUE, lw=2, label="30 heads of 64")
    b.plot(Tf, 4 * K * Tf ** 2 / 2 ** 30, ":", color=MUTED, lw=1.5, label="one score tensor, 4 K T squared bytes")
    b.set_xscale("log", base=2); b.set_yscale("log")
    b.set_ylim(1e-2, 10)
    b.set_xlabel("length T (tokens)"); b.set_ylabel("peak memory above the checkpoint (GB)")
    b.set_title("(b) Peak memory per call")
    b.legend(fontsize=8.5, frameon=False, loc="upper left")
    b.grid(True, which="major", color="#E5E7EB", lw=0.8)
    _spines(b)
    fig.tight_layout()
    fig.savefig(OUT / "xh_time_memory.png", bbox_inches="tight")
    plt.close(fig)


def fig_widths():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})
    widths = [64, 96, 128, 192, 384]
    x = np.arange(len(widths))
    for j, (Tq, col) in enumerate([(4096, PURPLE), (8192, BLUE)]):
        s = [timing(f"eq{p}")[1][list(timing(f"eq{p}")[0]).index(Tq)] for p in widths]
        f = timing("fold")[1][list(timing("fold")[0]).index(Tq)]
        a.bar(x + (j - 0.5) * 0.38, s, 0.36, color=col, label=f"equal widths, T = {Tq:,}")
        a.axhline(f, color=col, ls="--", lw=1.4)
        a.text(len(widths) - 0.35, f, f"folded,\nT = {Tq:,}", color=col, fontsize=8.5, va="center", ha="left")
    a.set_xticks(x, [f"{p}\n{K * D // p} heads" for p in widths], fontsize=9)
    a.set_xlabel("head width p"); a.set_ylabel("seconds per call, one layer")
    a.set_title("(a) Time against the head width")
    a.set_ylim(0, 1.55); a.set_xlim(-0.6, len(widths) + 0.5)
    a.legend(fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.86))
    _spines(a)

    # multiply-adds per layer in units of T^2 (large T; the tiled kernel does the causal half)
    useful = K * (DK + 1)
    rows = [("fallback path\n(folded)", useful, 0.0, K * D, True),
            ("tiled, any p\n(equal widths)", useful / 2, (K * D - useful) / 2, K * D / 2, False),
            ("one score pass\nper channel (ideal)", useful / 2, 0.0, K * D / 2, False)]
    y = np.arange(len(rows))[::-1]
    for yi, (label, us, red, val, full) in zip(y, rows):
        if full:
            b.barh(yi, us / 2, color=PURPLE, height=0.55)
            b.barh(yi, val / 2, left=us / 2, color=GREEN, height=0.55)
            b.barh(yi, (us + val) / 2, left=(us + val) / 2, color="white", ec=GREY, hatch="////", height=0.55)
            tot = us + val
        else:
            b.barh(yi, us, color=PURPLE, height=0.55)
            b.barh(yi, red, left=us, color=ORANGE, height=0.55)
            b.barh(yi, val, left=us + red, color=GREEN, height=0.55)
            tot = us + red + val
        b.text(tot + 30, yi, f"{tot:,.0f}", va="center", fontsize=9, color=INK)
    b.set_yticks(y, [r[0] for r in rows], fontsize=9)
    b.set_xlim(0, 2600)
    b.set_xlabel("multiply-adds per layer, in units of T squared")
    b.set_title("(b) Where the arithmetic goes")
    b.legend(handles=[Patch(color=PURPLE, label="scores, needed"),
                      Patch(color=ORANGE, label="scores, padding and repetition"),
                      Patch(color=GREEN, label="values"),
                      Patch(fc="white", ec=GREY, hatch="////", label="above the diagonal, computed then masked")],
             fontsize=8, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.17))
    _spines(b)
    fig.tight_layout()
    fig.savefig(OUT / "xh_widths.png", bbox_inches="tight")
    plt.close(fig)


def fig_error():
    err = B["errors"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for c in range(K):
        for v, ls, mk in [("eq64", "-", "o"), ("naive", ":", None)]:
            r = sorted((e for e in err if e["variant"] == v), key=lambda e: e["T"])
            a.plot([e["T"] for e in r], [e["max_abs_per_channel"][c] for e in r], ls=ls, marker=mk, ms=5,
                   color=CHANNEL[c], lw=2 if v == "eq64" else 1.2,
                   label=(f"channel {c}, decay rate {LAM[c]:.3f}" if v == "eq64" else None))
    a.plot([], [], ":", color=MUTED, label="trained code path in float32 (dotted)")
    a.set_xscale("log", base=2); a.set_yscale("log")
    a.set_xlabel("length T (tokens)"); a.set_ylabel("largest absolute difference")
    a.set_title("(a) 30 heads of 64 against float64")
    a.legend(fontsize=8, frameon=False, loc="upper left")
    a.grid(True, which="major", color="#E5E7EB", lw=0.8)
    _spines(a)

    for v, col, mk, label in [("fold", RED, "s", "folded"), ("eq64", BLUE, "o", "30 heads of 64"),
                              ("eq64c", GREEN, "^", "30 heads of 64, key term centred")]:
        xs, ys = [], []
        for e in err:
            if e["variant"] != v:
                continue
            for c in range(K):
                xs.append(U * LAM[c] * e["T"]); ys.append(e["max_abs_per_channel"][c])
        b.scatter(xs, ys, color=col, marker=mk, s=28, label=label, alpha=0.85, zorder=3)
    xx = np.geomspace(1e-7, 1e-3, 50)
    b.plot(xx, 3 * xx, "--", color=INK, lw=1.2, label="3 u lambda T")
    b.plot(xx, 1.5 * xx, ":", color=INK, lw=1.2, label="1.5 u lambda T")
    floor = np.median([e["max_abs_per_channel"] for e in err if e["variant"] == "naive"])
    b.axhline(floor, color=MUTED, lw=1, ls="-.")
    b.text(9e-4, floor * 0.7, "float32 floor of the\ntrained code path", color=MUTED, fontsize=8.5, ha="right", va="top")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlim(1e-7, 1e-3); b.set_ylim(1e-6, 3e-3)
    b.set_xlabel("rounding scale u lambda T (u = 2 to the -24)")
    b.set_ylabel("largest absolute difference")
    b.set_title("(b) The error follows the rounding scale")
    b.legend(fontsize=8, frameon=False, loc="upper left")
    b.grid(True, which="major", color="#E5E7EB", lw=0.8)
    _spines(b)
    fig.tight_layout()
    fig.savefig(OUT / "xh_error.png", bbox_inches="tight")
    plt.close(fig)


def fig_breakdown():
    runs = [("folded,\nnot flushed", "cpu_bench_np_lean_fold_timers.json"),
            ("folded,\nflushed", "cpu_bench_k_np_fold_flush_timers.json"),
            ("30 heads of 64,\nflushed", "cpu_bench_i_np_fold64_flush_timers.json")]
    parts = [("context channels (xi)", "context channels", RED),
             ("well bank: context_components", "well bank", BLUE),
             ("well bank: harmonic_terms_lowrank", "bank: quadratic and force", PURPLE),
             ("well bank: analytical_grad", None, PURPLE),
             ("stiff-mode decomposition (lowrank_modes)", "stiff-mode decomposition", ORANGE)]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    y = np.arange(len(runs))[::-1]
    for yi, (label, fn) in zip(y, runs):
        comp = json.loads((RES / fn).read_text())["lengths"]["8192"]["components"]
        txt = (RES / fn).with_suffix(".txt").read_text()            # the timed forward the components come from
        total = float(re.search(r"components at 8192 \(one timed forward, ([0-9.]+) s", txt).group(1))
        left = 0.0
        for key, name, col in parts:
            w = comp.get(key, {}).get("s", 0.0)
            ax.barh(yi, w, left=left, color=col, height=0.55, ec="white", lw=1.5,
                    label=name if yi == y[0] and name else None)
            if key.startswith("context"):
                ax.text(left + w / 2, yi, f"{w:.1f} s", ha="center", va="center", color="white", fontsize=9)
            left += w
        ax.barh(yi, total - left, left=left, color=GREY, height=0.55, ec="white", lw=1.5,
                label="the rest" if yi == y[0] else None)
        ax.text(total + 0.15, yi, f"{total:.1f} s, {8192 / total:,.0f} tokens/s", va="center", fontsize=9.5)
    ax.set_yticks(y, [r[0] for r in runs])
    ax.set_xlim(0, 17.5)
    ax.set_xlabel("seconds for one forward pass of 8,192 tokens (SR2-NP, every other lever on)")
    ax.set_title("Where the forward pass spends its time")
    ax.legend(fontsize=8.5, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _spines(ax)
    fig.tight_layout()
    fig.savefig(OUT / "xh_breakdown.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_layout()
    fig_time_memory()
    fig_widths()
    fig_error()
    fig_breakdown()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
