#!/usr/bin/env python3
"""Figures for Refinement_Experiments_and_Variable_Step_Training.md.

    python3 _make_refinement_figs.py        # writes figures/refinement/*.png

Data (copied from the run outputs named beside each block, so the script needs no checkpoints):
- Cell 6b-7 Gates 2 and 3 of SO-CODE: results/cfc_baoab_code_<SO-CODE tag>/
  Cell-6b-7_flow_or_maps_noattn_SR2_on_Code_19250steps_output.txt
- FLOW-R penalties: results/<arm>/refinement_flow_confirmation_<kind>_seed20261009_output.txt
  (12 x 4 x 512 tokens at batch seed 20261009; the whole-stack value is FLOW-R's own standard refinement).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

GREEN, BLUE, RED, PURPLE, GREY, ORANGE = "#22C55E", "#3B82F6", "#EF4444", "#8B5CF6", "#9CA3AF", "#F59E0B"
INK, MUTED = "#1F2937", "#6B7280"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})

HERE = Path(__file__).parent
OUT = HERE / "refinement"
OUT.mkdir(exist_ok=True)
LAYER_COL = {0: BLUE, 1: GREEN}


def hold(j, N, L=2):
    return min((j * L) // N, L - 1)


def walk_row(ax, y, N, dt, layers, label, note="", reset=False, h=0.56):
    """One walk: N steps of size dt, each box coloured by the trained layer it uses."""
    for j in range(N):
        ax.add_patch(Rectangle((j * dt, y - h / 2), dt, h, facecolor=LAYER_COL[layers[j]], edgecolor="white", lw=2, alpha=0.9))
        if dt >= 0.9:
            ax.text(j * dt + dt / 2, y, f"{layers[j]}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        if reset and j > 0:
            ax.text(j * dt, y + h / 2 + 0.08, "v ← 0", ha="center", va="bottom", fontsize=8.5, color=RED)
    ax.text(-0.35, y, label, ha="right", va="center", fontsize=10)
    if note:
        ax.text(N * dt + 0.3, y, note, ha="left", va="center", fontsize=9, color=MUTED)


# ---------------------------------------------------------------------------------------------
def gates():
    fig, ax = plt.subplots(figsize=(12, 6.3))
    rows = [
        ("trained walk (Gate 0)", 2, 4.0, [0, 1], "N = L = 2, Δt = 4, T = 8: must reproduce the model exactly", False),
        ("Gate 1: velocity reset", 2, 4.0, [0, 1], "same walk, velocity zeroed between steps: what inertia buys", True),
        ("Gate 2: extension, N = 3", 3, 4.0, [0, 1, 0], "Δt = 4 kept, T grows to 12; layers cycle 0,1,0", False),
        ("Gate 2: extension, N = 4", 4, 4.0, [0, 1, 0, 1], "T = 16", False),
        ("Gate 3: refinement, N = 3", 3, 8 / 3, [hold(j, 3) for j in range(3)], "T = 8 kept, Δt = 8/3; 'hold': layer 0 takes two steps", False),
        ("Gate 3: refinement, N = 4", 4, 2.0, [hold(j, 4) for j in range(4)], "Δt = 2", False),
        ("Gate 3: refinement, N = 8", 8, 1.0, [hold(j, 8) for j in range(8)], "Δt = 1", False),
    ]
    ys = np.arange(len(rows))[::-1] * 1.0
    for y, (lab, N, dt, lay, note, rs) in zip(ys, rows):
        walk_row(ax, y, N, dt, lay, lab, note, reset=rs)
    ax.axvline(8, color=INK, lw=1.2, ls="--")
    ax.text(8, ys[0] + 0.62, "T = 8: the time the output head was trained to read", ha="center", fontsize=9)
    ax.axhspan(ys[-1] - 0.5, ys[4] + 0.5, color=PURPLE, alpha=0.06, lw=0)
    ax.text(16.2, ys[5], "the framework test:\na flow gives the same\nanswer for every N", fontsize=9.5, color=PURPLE, va="center")
    ax.set_xlim(-0.2, 22)
    ax.set_ylim(ys[-1] - 0.7, ys[0] + 0.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("integration time τ  (one box = one layer step; the number in it = the trained layer whose weights it uses)")
    ax.set_xticks([0, 2, 4, 8, 12, 16])
    for c, t in ((BLUE, "layer 0"), (GREEN, "layer 1")):
        ax.add_patch(Rectangle((0, 0), 0, 0, facecolor=c, label=t))
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    ax.set_title("The four gates of Cell 6b-7 on an L = 2 model (Δt = 4, T = 8)", loc="left", fontsize=12)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.92, bottom=0.11)
    fig.savefig(OUT / "gates.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
def box(ax, x, y, w, h, text, col, fs=9.5, alpha=0.16):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=col, alpha=alpha,
                                edgecolor=col, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1, col=INK):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))


def anatomy():
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.6), gridspec_kw={"height_ratios": [0.85, 0.85, 1.15]})
    # A: one trained step
    a = axes[0]
    xs = [0.0, 2.05, 4.1, 6.15, 8.2]
    items = [("context ξ from h\n(5 decaying channels)", RED), ("Vθ wells: occupancies\nand the quadratic", RED),
             ("exact damped flow\nof the stiff modes", GREEN), ("kick: all other\nforces (Vφ, rest)", RED),
             ("LayerNorm:\nproject to the sphere", RED)]
    for x, (t, c) in zip(xs, items):
        box(a, x, 0.25, 1.8, 0.9, t, c)
    for x in xs[:-1]:
        arrow(a, x + 1.82, 0.7, x + 2.03, 0.7)
    a.text(-0.2, 0.7, "h, v in", ha="right", va="center", fontsize=10)
    a.text(10.1, 0.7, "h′, v′ out", ha="left", va="center", fontsize=10)
    a.text(0, 1.45, "A. One layer step, as trained (schematic; the integrator interleaves these). Green: a function of elapsed time\n"
                    "    (composes exactly for any N, Prop. 47). Red: done once per step.", fontsize=10, ha="left", va="bottom")
    a.set_xlim(-1.2, 11.2); a.set_ylim(0, 1.7); a.axis("off")

    # B: whole-stack refinement, N = 4 on L = 2
    b = axes[1]
    for j in range(4):
        x = j * 2.55
        box(b, x, 0.2, 2.35, 0.95, f"step {j}: layer {hold(j, 4)}, step size 2\nξ, wells, flow, kick,\nLayerNorm", LAYER_COL[hold(j, 4)], fs=9)
        if j:
            arrow(b, x - 0.2, 0.67, x, 0.67)
    b.text(0, 1.45, "B. Whole-stack refinement (Gate 3), N = 4: every red operation now happens 4 times instead of 2, "
                    "at states the trained walk never visited.", fontsize=10)
    b.set_xlim(-1.2, 11.2); b.set_ylim(0, 1.7); b.axis("off")

    # C: per-layer flow, k = 4 substeps per layer
    c = axes[2]
    for ell in range(2):
        x0 = ell * 5.4
        box(c, x0, 1.05, 1.15, 0.55, "ξ, wells\nfrozen at entry", LAYER_COL[ell], fs=8.5, alpha=0.3)
        for s in range(4):
            box(c, x0 + 1.3 + s * 0.83, 0.2, 0.75, 0.6, f"flow\n+kick", LAYER_COL[ell], fs=8)
        box(c, x0 + 4.65, 0.2, 0.6, 0.6, "LN\nonce", RED, fs=8)
        arrow(c, x0 + 0.6, 1.05, x0 + 1.5, 0.82, MUTED)
        c.text(x0 + 2.85, 0.03, f"layer {ell}: k = 4 substeps of Δt/4", ha="center", fontsize=9, color=MUTED)
    arrow(c, 5.28, 0.5, 5.55, 1.05, MUTED)
    c.text(0, 1.85, "C. Per-layer flow (FLOW-R), k = 4: inside each trained layer, context, wells and Vφ's choices are held at the "
                    "layer's entry\n    and LayerNorm runs once, at the layer's end, as in training. Only the integration is refined. "
                    "At k = 1 it is the trained model, bit for bit.", fontsize=10, va="top")
    c.set_xlim(-1.2, 11.2); c.set_ylim(-0.1, 1.95); c.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "anatomy.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
K = [1.5, 2, 3, 4, 6, 8]
FLOWR = {  # per-layer flow penalty ln(PPL_kL / PPL_L) at k = 1.5 ... 8, then whole stack at k = 4
    "SR2 (OpenWebText)":         ([0.286, 0.206, 0.084, 0.103, 0.104, 0.104], 2.004, BLUE, "-"),
    "SO-TS (SR2, TinyStories)":  ([0.255, 0.151, 0.086, 0.100, 0.098, 0.097], 1.988, GREEN, "-"),
    "SO-CODE (SR2, code)":       ([0.970, 0.526, 0.300, 0.384, 0.376, 0.369], 2.858, RED, "-"),
    "L4-CODE (SO-CODE at L = 4)": ([0.429, 0.520, 0.282, 0.320, 0.309, 0.306], 2.960, ORANGE, "-"),
    "PMX (wells, OpenWebText)":  ([1.711, 0.841, 0.292, 0.415, 0.386, 0.387], 2.464, PURPLE, "-"),
    "FO-b-OWT (first order)":    ([0.548, 1.049, 1.014, 0.877, 0.856, 0.904], 1.404, GREY, "--"),
}


def flowr():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [1.45, 1]})
    for name, (pen, ws, col, ls) in FLOWR.items():
        a.plot(K, pen, ls, color=col, lw=2, marker="o", ms=5, label=name)
    a.axvspan(4, 8, color=GREY, alpha=0.12, lw=0)
    a.text(5.6, 0.56, "criterion window:\nmoves ≤ 0.02 from k = 4 to 6\nand ≤ 0.01 from k = 6 to 8", ha="center", fontsize=8.5, color=MUTED)
    a.set_xscale("log", base=2)
    a.set_xticks(K); a.set_xticklabels(["1.5", "2", "3", "4", "6", "8"])
    a.set_xlabel("k = substeps per trained layer  (N = k·L steps over the same T)")
    a.set_ylabel("penalty ln(PPL_N / PPL_trained), nats")
    a.set_ylim(0, 1.8)
    a.legend(frameon=False, fontsize=8.5, loc="upper right")
    a.set_title("A. Per-layer flow: second-order models settle, FO-b does not", loc="left", fontsize=11)

    names = list(FLOWR)
    y = np.arange(len(names))[::-1]
    ws = [FLOWR[n][1] for n in names]
    pl = [FLOWR[n][0][-1] for n in names]
    b.barh(y + 0.19, ws, 0.36, color=INK, alpha=0.75, label="whole stack refined, k = 4")
    b.barh(y - 0.19, pl, 0.36, color=[FLOWR[n][2] for n in names], label="per-layer flow, k = 8 (model's colour)")
    for yi, w_, p_ in zip(y, ws, pl):
        b.text(w_ + 0.04, yi + 0.19, f"{w_:.2f}", va="center", fontsize=8.5)
        b.text(p_ + 0.04, yi - 0.19, f"{p_:.2f}", va="center", fontsize=8.5)
    b.set_yticks(y); b.set_yticklabels([n.split(" (")[0] for n in names], fontsize=9.5)
    b.set_xlabel("penalty, nats", labelpad=2)
    b.set_xlim(0, 3.4)
    b.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2)
    b.set_title("B. The same models, both tests", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "flowr.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
def aliasing():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    phi0 = 0.3
    cases = [("A. SO-CODE, trained: N = 2, θ = ωΔt = 7.55", 7.55, 2, RED),
             ("B. SO-CODE refined: N = 8, θ/4 = 1.89 per step", 7.55 / 4, 8, RED),
             ("C. L4-CODE, trained: N = 4, θ = 2.20", 2.20, 4, ORANGE)]
    for ax, (title, th, n, col) in zip(axes, cases):
        ax.add_patch(Circle((0, 0), 1, fill=False, color=GREY, lw=1.2))
        tt = np.linspace(0, th * n, 400)
        r = 1 - 0.035 * tt / (2 * np.pi)          # a slow inward drift, only so the turns stay visible
        ax.plot(r * np.cos(phi0 + tt), r * np.sin(phi0 + tt), color=col, lw=1, alpha=0.45)
        for j in range(n + 1):
            p = phi0 + j * th
            rr = 1 - 0.035 * j * th / (2 * np.pi)
            ax.plot(rr * np.cos(p), rr * np.sin(p), "o", ms=9, color=col if j else INK, mec="white", mew=1.5, zorder=3)
            ax.text(1.18 * np.cos(p), 1.18 * np.sin(p), str(j), ha="center", va="center", fontsize=9)
        ax.set_aspect("equal"); ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.axis("off")
        ax.set_title(title, fontsize=10, loc="left")
    axes[0].text(0, -1.42, "2.4 turns between samples: the readout sees\nonly the phases 0, 1, 2 (an apparent 1.27 rad per step)",
                 ha="center", fontsize=8.5, color=MUTED)
    axes[1].text(0, -1.42, "the same mode, now stopped at 8 phases;\nthe per-step operations run at all of them",
                 ha="center", fontsize=8.5, color=MUTED)
    axes[2].text(0, -1.42, "a third of a turn per step:\nthe trained samples already cover the circle",
                 ha="center", fontsize=8.5, color=MUTED)
    fig.suptitle("A stiff mode's phase at each step (median stiffest mode; damping omitted)", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "aliasing.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
def sr4a():
    fig = plt.figure(figsize=(12.5, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], width_ratios=[1.1, 1], hspace=0.55, wspace=0.25)
    a = fig.add_subplot(gs[0, :])
    rng = np.random.default_rng(0 + 4_000_000)
    draws = [int(rng.choice((2, 3, 4))) for _ in range(60)]
    cols = {2: BLUE, 3: PURPLE, 4: ORANGE}
    a.bar(np.arange(1, 61), draws, color=[cols[d] for d in draws], width=0.8)
    a.set_xlim(0.3, 60.7); a.set_ylim(0, 4.6); a.set_yticks([2, 3, 4])
    a.set_xlabel("optimizer step (the first 60 draws of the run's own generator, seed SEED + 4,000,000)")
    a.set_ylabel("N this step")
    a.set_title("A. Each training step draws N from {2, 3, 4}; every microbatch of the step uses it", loc="left", fontsize=11)

    b = fig.add_subplot(gs[1, 0])
    for y, N in zip((2, 1, 0), (2, 3, 4)):
        walk_row(b, y, N, 8 / N, [hold(j, N) for j in range(N)], f"N = {N}", f"Δt = {8 / N:.2f}", h=0.6)
    b.axvline(8, color=INK, lw=1, ls="--")
    b.set_xlim(-0.2, 10.5); b.set_ylim(-0.6, 2.7); b.set_yticks([]); b.spines["left"].set_visible(False)
    b.set_xlabel("integration time τ (T = 8 always)")
    b.set_title("B. The three walks the model trains on", loc="left", fontsize=11)

    c = fig.add_subplot(gs[1, 1])
    steps = np.arange(0, 19_251, 500)
    c.vlines(steps[1:], 0.55, 0.95, color=INK, lw=0.8)
    c.vlines(np.arange(2_000, 19_251, 2_000), 1.25, 1.75, color=RED, lw=2)
    c.text(0, 1.0, "evaluation at N = L = 2, every 500 steps (paired with SO-CODE's)", fontsize=9, va="bottom")
    c.text(0, 1.8, "refinement probe at N = 2, 3, 4, 6, 8, every 2,000 steps", fontsize=9, va="bottom", color=RED)
    c.set_xlim(0, 19_400); c.set_ylim(0.3, 2.3); c.set_yticks([]); c.spines["left"].set_visible(False)
    c.set_xlabel("training step")
    c.set_title("C. What gets measured during training", loc="left", fontsize=11)
    fig.savefig(OUT / "sr4a.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
def calls():
    N = [1, 2, 3, 4, 6, 8]
    so = np.array([4.4395, 1.6221, 2.2611, 3.0405, 3.9309, 4.3489]) - 1.6221      # SO-CODE, Cell 6b-7 Gate 3
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    a.plot(N, so, "-o", color=RED, lw=2, label="SO-CODE, measured (Cell 6b-7 Gate 3)")
    a.add_patch(Rectangle((2.75, 0), 1.5, 0.095, facecolor=GREEN, alpha=0.35, lw=0))
    a.annotate("VS1: ≤ 0.095 nats (+10%)\nat N = 3 and 4", xy=(3.5, 0.095), xytext=(3.3, 0.8), fontsize=9,
               arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    a.add_patch(Rectangle((7.6, 0), 0.8, 0.50, facecolor=PURPLE, alpha=0.3, lw=0))
    a.annotate("VS4: ≤ 0.50 nats\nat N = 8", xy=(8, 0.5), xytext=(6.1, 1.25), fontsize=9,
               arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    a.axvspan(2, 4, color=GREY, alpha=0.12, lw=0)
    a.text(3, 2.05, "trained\nrange", ha="center", fontsize=9, color=MUTED)
    for n_, v in zip(N[2:], so[2:]):
        a.text(n_ + 0.12, v + 0.06, f"{v:.2f}", fontsize=8.5)
    a.set_xticks(N); a.set_xlabel("N, steps over T = 8"); a.set_ylabel("penalty ln(PPL_N / PPL_2), nats")
    a.set_ylim(0, 3.1); a.legend(frameon=False, fontsize=9, loc="upper center")
    a.set_title("A. The baseline the calls are scored against", loc="left", fontsize=11)

    sc = {"no effect (SO-CODE's curve)": (so, GREY, ":"),
          "three maps: works where trained, not between or beyond": ([1.6, 0, 0.05, 0.06, 1.3, 2.2], ORANGE, "--"),
          "a flow: one answer for every N": ([0.9, 0, 0.04, 0.05, 0.07, 0.08], GREEN, "-")}
    for name, (v, col, ls) in sc.items():
        b.plot(N, v, ls, color=col, lw=2.2, marker="o", ms=4, label=name)
    b.axvspan(2, 4, color=GREY, alpha=0.12, lw=0)
    b.set_xticks(N); b.set_xlabel("N, steps over T = 8"); b.set_ylabel("penalty, nats")
    b.set_ylim(0, 3.75); b.legend(frameon=False, fontsize=8.8, loc="upper right")
    b.set_title("B. Three outcomes SR4a-CODE could show (illustration, not data)", loc="left", fontsize=11)
    b.text(6.2, 1.62, "VS1 hits, VS4 misses", color=ORANGE, fontsize=9)
    b.text(5.3, 0.25, "VS1 and VS4 both hit", color=GREEN, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "calls.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    gates(); anatomy(); flowr(); aliasing(); sr4a(); calls()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
