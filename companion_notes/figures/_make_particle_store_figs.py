#!/usr/bin/env python3
"""Figures for the particle store (Stage 0'' of Implementing_Semantic_Templates_A_Path.md).

    python3 _make_particle_store_figs.py        # writes figures/particle_store/*.png

Data: particle_store/live_count_init_data.json, dumped from the PARTICLE_STORE = 4, rank 8
model at initialisation on 2 x 256 real validation tokens (the readings of
debug/verify_particle_store.py), one arm with half-lives 1-16 tokens and one without
spontaneous death.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN, BLUE, RED, PURPLE, GREY, ORANGE = "#22C55E", "#3B82F6", "#EF4444", "#8B5CF6", "#9CA3AF", "#F59E0B"
INK, MUTED = "#1F2937", "#6B7280"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})

HERE = Path(__file__).parent
OUT = HERE / "particle_store"
OUT.mkdir(exist_ok=True)
D = json.load(open(OUT / "live_count_init_data.json"))


def live_count():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
    t = np.arange(256)
    for arm, col, lab in (("decay", BLUE, "half-lives 1–16 tokens"), ("nodecay", RED, "no spontaneous death")):
        for layer, ls in (("layer0", "-"), ("layer1", "--")):
            n = np.array(D[arm][layer]["n"])
            a.plot(t, n, ls, color=col, lw=2, label=f"{lab}, {layer.replace('layer', 'layer ')}")
    a.plot(t, 1.17 * t, ":", color=GREY, lw=1.5, label="1.17 t")
    a.set_xlabel("token position t")
    a.set_ylabel("live particles seen by token t")
    a.set_title("A. the store's live count, at initialisation", loc="left", fontsize=11)
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    a.annotate("saturates at Σ_j ē_j / (1 − λ_j)", xy=(200, D["decay"]["layer1"]["n"][200]), xytext=(120, 60),
               fontsize=9, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    # B: existence per type, both layers
    K = len(D["decay"]["layer0"]["E_mean_per_type"])
    x = np.arange(K)
    w = 0.38
    for i, (layer, col) in enumerate((("layer0", GREEN), ("layer1", PURPLE))):
        m = D["decay"][layer]["E_mean_per_type"]
        cv = D["decay"][layer]["E_cv_per_type"]
        b.bar(x + (i - 0.5) * w, m, w, color=col, label=f"{layer.replace('layer', 'layer ')}")
        for xi, mi, ci in zip(x + (i - 0.5) * w, m, cv):
            b.text(xi, mi + 0.02, f"CV {ci:.2f}", ha="center", fontsize=7.5, color=MUTED, rotation=90, va="bottom")
    b.set_xticks(x)
    b.set_xticklabels([f"type {j}" for j in range(K)])
    b.set_ylabel("mean existence E_j across tokens")
    b.set_ylim(0, 1.0)
    b.set_title("B. what each type creates, per token", loc="left", fontsize=11)
    b.legend(frameon=False, fontsize=9)
    s0, s1 = D["decay"]["layer0"]["stats"][2], D["decay"]["layer1"]["stats"][2]
    b.text(0.98, 0.97, f"reach at layer 1 carried by\nlayer-0 particles: {s1:.0%}", transform=b.transAxes,
           ha="right", va="top", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "ps_live_count_init.png", dpi=150)
    plt.close(fig)


def short_train():
    """The three-arm CPU run: loss curves, and what the model does with the store."""
    import json as _json
    D = {a: _json.load(open(OUT / f"ps_short_train_{a}.json")) for a in ("off", "decay", "nodecay")}
    COL = {"off": GREY, "decay": BLUE, "nodecay": RED}
    LAB = {"off": "no store", "decay": "store, half-lives 1-16", "nodecay": "store, no death"}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.8))

    for arm, d in D.items():
        st, ls = zip(*d["loss_log"])
        a.plot(st, ls, color=COL[arm], lw=2, label=LAB[arm])
    a.set_xlabel("training step")
    a.set_ylabel("training loss")
    a.set_title("A. the three arms train alike", loc="left", fontsize=11)
    a.legend(frameon=False, fontsize=9)
    a.text(0.97, 0.93, "held-out PPL\n" + "\n".join(f"{LAB[k]}: {D[k]['heldout_ppl']:.0f}" for k in D),
           transform=a.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)

    arms = ["decay", "nodecay"]
    x = np.arange(len(arms))
    cost = [D[k]["removal_cost_pct"] for k in arms]
    err = [D[k]["removal_spread_pct"] / np.sqrt(8) for k in arms]
    b.bar(x, cost, 0.5, color=[COL[k] for k in arms], yerr=err, capsize=5,
          error_kw=dict(ecolor=INK, lw=1))
    b.set_xticks(x); b.set_xticklabels(["half-lives\n1-16", "no\ndeath"])
    b.set_ylabel("removal cost, % PPL")
    b.set_title("B. does the model use the store?", loc="left", fontsize=11)
    for xi, ci in zip(x, cost):
        b.text(xi, ci + 0.35, f"{ci:+.2f}%", ha="center", fontsize=9, color=INK)
    b.axhline(0, color=GREY, lw=0.8)
    b.set_ylim(0, max(cost) * 1.45)

    w = 0.38
    for i, l in enumerate(("layer0", "layer1")):
        c.bar(x + (i - 0.5) * w, [D[k]["layers"][l[-1]]["position_R2"] for k in arms], w,
              color=[GREEN, PURPLE][i], label=f"layer {i}")
    c.axhline(0, color=INK, lw=0.9)
    c.set_xticks(x); c.set_xticklabels(["half-lives\n1-16", "no\ndeath"])
    c.set_ylabel("existences explained by position, $R^2$")
    c.set_ylim(-0.25, 1.0)
    c.set_title("C. content, not position", loc="left", fontsize=11)
    c.legend(frameon=False, fontsize=9, loc="upper left")
    c.axhspan(0.94, 1.0, color=ORANGE, alpha=0.30, zorder=0)
    c.text(0.97, 0.965, "Poisson occupations", transform=c.transAxes, ha="right",
           va="top", fontsize=8, color=ORANGE)

    fig.tight_layout()
    fig.savefig(OUT / "ps_short_train.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    live_count()
    short_train()
    print("wrote", OUT / "ps_live_count_init.png", "and", OUT / "ps_short_train.png")
