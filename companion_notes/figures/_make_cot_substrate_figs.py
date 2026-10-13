"""Figures for Chain_of_Thought_Substrate_in_Lagrangian_Models.md (the working-memory note).

PNGs into figures/cot_substrate/. The measured panels carry the ladder's
recorded numbers (protocol SS5.10, SS5.15, SS5.19; the generations registry);
the toy panel simulates the Poisson-mode occupation recursion of
Poisson_Mode_Registers_PM1.md eq. 1.2 on a planted sequence.

  cot_three_needs.png     -- the four things chain-of-thought needs, order-dependent
                             composition first (the classical definition, Feng et
                             al. 2023), and which mechanism supplies each
  cot_extra_compute.png   -- the two ways to spend more compute on a trained
                             model: extension past the trained depth (Gate 2,
                             catastrophic in every arm) against refinement of
                             the per-layer flow (converges in SR2 and PMX)
  cot_scratchpad.png      -- the Poisson-mode occupation as a scratchpad: a
                             planted sequence with two "thought" tokens, the
                             occupation of the modes they overlap, and the well
                             depth every later token then feels
  cot_route.png           -- the route: emit intermediate particles into the
                             registers, read them back through the wells,
                             monitor them with the per-step energy; what exists
                             and what is missing

Run:  MPLBACKEND=Agg python3 _make_cot_substrate_figs.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
GREEN, BLUE, RED, PURPLE, GREY, ORANGE = "#22C55E", "#3B82F6", "#EF4444", "#8B5CF6", "#9CA3AF", "#F59E0B"
OUT = Path(__file__).parent / "cot_substrate"
OUT.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def box(ax, xy, w, h, text, col, fs=9, ec="k"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", fc=col, ec=ec, lw=0.8, alpha=0.95))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


# ------------------------------------------------------------- 1. the grid
def fig_three_needs():
    needs = ["order-dependent\ncomposition", "extra compute\non a hard token", "memory across\nthe computation", "readback into\nlater predictions",
             "conservative\n(per token)", "refinement-ready\n(at 0.5B tokens)"]
    mechs = ["Fock\nregisters", "Poisson-mode\nregisters", "pair potential\n$V_\\phi$", "per-layer flow\n(SR2 / PMX)"]
    # 2 = yes, 1 = partial, 0 = no
    M = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 2],
                  [2, 2, 0, 0],
                  [2, 2, 1, 0],
                  [0, 2, 2, 2],
                  [0, 0, 0, 1]])
    labels = [["no: additive\nupdates commute", "no: occupations\nare sums", "no: additive", "no: a flow of\none fixed step"],
              ["fixed step", "fixed step", "fixed step", "substeps_per_layer\n= k converges"],
              ["slot contents\n(two-hop attention)", "occupations $\\bar n_v$\nover the strict past", "none: pairwise,\nno persistent state", "none"],
              ["reverse channel", "the wells' force", "selected sources\n(top-16)", "—"],
              ["no: gradient\nof nothing", "yes", "yes", "yes"],
              ["no: +1,274%", "no: CX3 miss\n(0.39 nats)", "—", "SR2 yes:\nwithin 0.10 nats"]]
    cols = {2: "#BBF7D0", 1: "#FEF3C7", 0: "#FEE2E2"}
    fig, ax = plt.subplots(figsize=(12.5, 7.8))
    ax.set_xlim(0, 4.6); ax.set_ylim(0, 7.0); ax.axis("off")
    for j, m in enumerate(mechs):
        ax.text(0.95 + j * 0.9 + 0.42, 6.72, m, ha="center", va="center", fontsize=10, fontweight="bold")
    for i, n in enumerate(needs):
        y = 5.6 - i * 0.95
        ax.text(0.47, y + 0.4, n, ha="center", va="center", fontsize=9.5, fontweight="bold")
        for j in range(4):
            box(ax, (0.95 + j * 0.9, y), 0.85, 0.82, labels[i][j], cols[M[i, j]], fs=8, ec="#6B7280")
    ax.plot([0.9, 4.6], [1.74, 1.74], color="k", lw=0.8)
    ax.text(0.47, 1.52, "properties", ha="center", fontsize=8.5, color=GREY, style="italic")
    ax.text(0.47, 5.43, "the defining one:\nno mechanism\nsupplies it", ha="center", va="top", fontsize=8, color=RED, style="italic")
    ax.text(0.47, 6.72, "what CoT needs", ha="center", va="center", fontsize=9, color=GREY, style="italic")
    ax.set_title("Four things chain-of-thought needs, in order, and which mechanism supplies each (green yes, amber partial, red no)")
    save(fig, "cot_three_needs.png")


# ----------------------------------------------- 2. extension vs refinement
def fig_extra_compute():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    arms = ["SR2", "L=4 Fock", "G2", "F3.1", "G3", "G3′", "PM1", "PMX"]
    g2 = [35, 41, 51, 92, 325, 441, 951, 956]
    colors = [BLUE, GREY, GREY, GREY, GREY, GREY, GREEN, GREEN]
    ax[0].bar(arms, g2, color=colors)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("perplexity rise, one extra layer step (%)")
    ax[0].set_title("Extension past the trained depth (Gate 2)")
    for i, v in enumerate(g2):
        ax[0].text(i, v * 1.15, f"+{v}%", ha="center", fontsize=8)
    ax[0].text(0.02, 0.93, "catastrophic in every arm; worst where the\nmemory lives (the registers' wells)", transform=ax[0].transAxes, fontsize=8.5, va="top")
    ks = [1, 2, 3, 4, 6, 8]
    sr2 = [55.04, 67.64, 59.83, 61.00, 61.08, 61.09]
    pmx = [52.26, 121.19, 69.96, 79.16, 76.90, 76.95]
    pmx_std = [52.26, None, None, 614.07, None, None]
    ax[1].plot(ks, sr2, "o-", color=BLUE, lw=2, label="SR2, per-layer flow (FLOW-R)")
    ax[1].plot(ks, pmx, "s-", color=GREEN, lw=2, label="PMX, per-layer flow (FLOW-R)")
    ax[1].plot([1, 4], [52.26, 614.07], "x--", color=RED, ms=9, mew=2, label="PMX, standard refinement (everything recomputed)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("substeps per layer, k (same total integration time)")
    ax[1].set_ylabel("validation perplexity")
    ax[1].set_title("Refinement of the per-layer flow converges")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].text(0.35, 0.72, "a knob, not a scratchpad: more accuracy\nof the same step, not more reasoning", transform=ax[1].transAxes, fontsize=8.5)
    save(fig, "cot_extra_compute.png")


# ------------------------------------------------------- 3. the scratchpad
def fig_scratchpad():
    """PM1 note eq. 1.1-1.3 on a planted sequence: 64 modes, two thought tokens
    that overlap modes 20 and 41 strongly; occupation over the strict past."""
    rng = np.random.default_rng(3)
    T, K = 40, 64
    lam = np.full(K, 0.90)                                  # half-life ~ 6.6 tokens
    E = rng.uniform(0.0, 0.08, size=(T, K))                 # background overlaps
    thought = {12: 20, 26: 41}                              # position -> mode it overlaps
    for t, v in thought.items():
        E[t, v] = 0.95
    phi = np.zeros((T + 1, K))
    for t in range(T):
        phi[t + 1] = lam * phi[t] + E[t]                    # phi seen by token t+1: strict past
    depth = 0.3
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.3), gridspec_kw={"width_ratios": [1.3, 1, 1], "wspace": 0.32})
    im = ax[0].imshow(phi[1:].T, aspect="auto", cmap="Greens", origin="lower")
    ax[0].set_xlabel("token position t"); ax[0].set_ylabel("mode v")
    ax[0].set_title("Occupation $\\bar n_v(t)$ seen by each token")
    for (t, v), (dx, dy) in zip(thought.items(), ((-9, 11), (-2, 12))):
        ax[0].annotate(f"thought token at t = {t}\nraises mode {v}", (t, v), (t + dx, v + dy), fontsize=8,
                       arrowprops=dict(arrowstyle="->", lw=0.8))
    plt.colorbar(im, ax=ax[0], fraction=0.04)
    ts = np.arange(1, T + 1)
    ax[1].plot(ts, phi[1:, 20], color=GREEN, lw=2, label="mode 20")
    ax[1].plot(ts, phi[1:, 41], color=BLUE, lw=2, label="mode 41")
    ax[1].plot(ts, phi[1:, 5], color=GREY, lw=1.2, label="a background mode")
    for t in thought:
        ax[1].axvline(t, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("token position t"); ax[1].set_ylabel("$\\bar n_v(t)$")
    ax[1].set_title("A thought token's trace decays")
    ax[1].legend(fontsize=8)
    r = np.linspace(0, 2.5, 300)
    for t_read, col, lab in ((14, GREEN, "token 14, two after the thought"), (24, ORANGE, "token 24, twelve after"), (10, GREY, "token 10, before it")):
        U = -phi[t_read, 20] * depth * np.exp(-r ** 2)
        ax[2].plot(r, U, color=col, lw=2, label=lab)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xlabel("distance from the centre of mode 20"); ax[2].set_ylabel("well the token feels, $V_{\\mathrm{pm}}$ (mode 20)")
    ax[2].set_title("Readback: the well later tokens feel")
    ax[2].legend(fontsize=8, loc="lower right")
    save(fig, "cot_scratchpad.png")


# ------------------------------------------------------------- 4. the route
def fig_route():
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 6); ax.axis("off")
    have, miss = "#BBF7D0", "#FEE2E2"
    steps = [("1. trigger\ncurvature $\\mathcal{K}_{\\max}$ above\na learned threshold", miss, "designed (book §37.4),\nnever trained or tested"),
             ("2. emit intermediate\nparticles\n(thought tokens)", miss, "needs a reasoning\ncorpus and a training\nsignal: the real cost"),
             ("3. write them into the\nPoisson-mode registers\n(occupations $\\bar n_v$)", have, "exists: PMX, published;\nconservative, inspectable"),
             ("4. read back through\nthe wells' force on\nevery later token", have, "exists: the wells carry\nthe model (1.06 nats)"),
             ("5. monitor the trace:\nper-step energy and\nlandscape work", have, "exists in principle\n(Definition 36); the\ndetector is unbuilt")]
    xs = np.linspace(0.2, 10.9, 5)
    for i, (t, col, note) in enumerate(steps):
        box(ax, (xs[i], 3.2), 2.3, 1.7, t, col, fs=8.5)
        ax.text(xs[i] + 1.15, 2.55, note, ha="center", va="top", fontsize=8, color="#374151")
        if i:
            ax.annotate("", xy=(xs[i] - 0.03, 4.05), xytext=(xs[i - 1] + 2.33, 4.05), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(xs[2] + 1.15, 4.95), xytext=(xs[4] + 1.15, 4.95), arrowprops=dict(arrowstyle="->", lw=1.0, connectionstyle="arc3,rad=0.35", color=PURPLE))
    ax.text(xs[3] + 1.15, 5.65, "a flagged thought can be re-emitted: self-correction (not designed yet)", ha="center", fontsize=8.5, color=PURPLE)
    ax.text(6.75, 0.9, "What the framework adds over a transformer's scratchpad: the trace is a per-token conservative memory\n"
            "(a 64-vector of Poisson means per position, readable and removable), its effect on later tokens is a gradient force,\n"
            "and each step of the trace has an energy whose deviation from its expected value is a hallucination signal on the reasoning itself.",
            ha="center", fontsize=8.8)
    ax.text(6.75, 0.15, "What it does not add: a way to think longer without emitting. Extension fails (Figure 2); refinement converges but reasons no more.",
            ha="center", fontsize=8.8, color=RED)
    ax.set_title("The route: emitted steps held in a conservative working memory (green exists, red missing)")
    save(fig, "cot_route.png")


if __name__ == "__main__":
    fig_three_needs(); fig_extra_compute(); fig_scratchpad(); fig_route()
