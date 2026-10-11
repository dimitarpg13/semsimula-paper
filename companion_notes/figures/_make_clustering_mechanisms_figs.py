"""Figures for Two_Clustering_Mechanisms_Saturation_and_Synchronization.md.

PNGs into figures/clustering_mechanisms/. The theory panels are exact
simulations of the stated toy systems; the measured panels read the D-sync
diagnostic's JSON from the results folders (debug/dsync_clustering_diagnostics.py).

  cm_saturation.png       -- mechanism 1, hard assignment: a softmax over K
                             wells hardening with the inverse temperature,
                             the effective number of active wells, and the
                             participation-ratio rank of G = [sqrt(g_k) B_k]
                             that follows (the 2026-09-17 degeneracy)
  cm_kuramoto.png         -- mechanism 2, synchronization: Kuramoto
                             oscillators, the order parameter against the
                             coupling (the threshold) and in time
  cm_causal_attention.png -- Karagodin et al. 2024 on the circle: causal
                             attention with V = I, the first token frozen,
                             a metastable multi-cluster phase, convergence to
                             the first token; the share of tokens within a
                             small angle of token 1 against depth
  cm_measured.png         -- the trained arms: effective wells of the two
                             factors of g, the rank of G, the pairwise-cosine
                             histograms of token states per layer

Run:  MPLBACKEND=Agg python3 _make_clustering_mechanisms_figs.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
GREEN, BLUE, RED, PURPLE, GREY, ORANGE = "#22C55E", "#3B82F6", "#EF4444", "#8B5CF6", "#9CA3AF", "#F59E0B"
OUT = Path(__file__).parent / "clustering_mechanisms"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261011)
RES = Path(__file__).resolve().parents[2] / "notebooks/conservative_arch/scaleup/results"
_RUN = ("cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_{}"
        "L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn")
ARMS = {"F3.1": ("", GREY), "SR2": ("sr2_", BLUE), "PM1": ("pm64_", GREEN)}


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_saturation():
    K, rank, d = 8, 4, 64
    scores = RNG.normal(size=(200, K))                       # 200 tokens' well scores
    Bs = RNG.normal(size=(K, d, rank)) / np.sqrt(d)
    betas = np.logspace(-1, 1.5, 30)
    eff_wells, pr_rank, n_floor = [], [], []
    for beta in betas:
        w = np.exp(beta * scores); w /= w.sum(1, keepdims=True)
        H = np.exp(-(w * np.log(w + 1e-300)).sum(1))
        prs, nf = [], []
        for wi in w:
            G = np.concatenate([np.sqrt(wi[k]) * Bs[k] for k in range(K)], axis=1)
            lam = np.linalg.eigvalsh(G.T @ G)
            prs.append(lam.sum() ** 2 / (lam ** 2).sum()); nf.append((lam > 1e-10).sum())
        eff_wells.append(np.median(H)); pr_rank.append(np.median(prs)); n_floor.append(np.median(nf))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    for beta, col in ((0.3, GREEN), (2.0, ORANGE), (8.0, RED)):
        w = np.exp(beta * scores[0]); w /= w.sum()
        ax.bar(np.arange(K) + {0.3: -0.27, 2.0: 0.0, 8.0: 0.27}[beta], w, width=0.26, color=col, label=f"beta = {beta:g}")
    ax.set_xlabel("well k"); ax.set_ylabel("softmax weight w_k"); ax.set_title("one token's gating hardens with beta"); ax.legend(fontsize=9)
    ax = axes[1]
    ax.semilogx(betas, eff_wells, "-", color=BLUE, lw=2)
    ax.set_xlabel("inverse temperature beta"); ax.set_ylabel("effective number of wells, exp H(w)")
    ax.set_title("saturation: active wells fall to one"); ax.axhline(1, color=GREY, ls="--", lw=1)
    ax = axes[2]
    ax.semilogx(betas, pr_rank, "-", color=RED, lw=2, label="participation ratio of eig(G^T G)")
    ax.semilogx(betas, n_floor, "-", color=GREY, lw=1.5, label="eigenvalues above the 1e-10 floor")
    ax.axhline(rank, color=GREEN, ls="--", lw=1); ax.text(betas[0] * 1.1, rank + 0.8, "one well x rank 4", color=GREEN, fontsize=9)
    ax.set_xlabel("inverse temperature beta"); ax.set_ylabel("rank of G (nominal 32)")
    ax.set_title("the low-rank factor degenerates with it"); ax.legend(fontsize=8.5, loc="center left")
    fig.tight_layout(); save(fig, "cm_saturation.png")


def kuramoto(Kc, n=200, T=60.0, dt=0.02, omega_sd=1.0, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n); om = rng.normal(0, omega_sd, n)
    R = []
    for _ in range(int(T / dt)):
        z = np.exp(1j * th).mean()
        R.append(abs(z))
        th = th + dt * (om + Kc * abs(z) * np.sin(np.angle(z) - th))
    return np.array(R)


def fig_kuramoto():
    Ks = np.linspace(0, 4, 33)
    Rinf = [kuramoto(k)[-500:].mean() for k in Ks]
    Kc = 2 / (np.pi * (1 / np.sqrt(2 * np.pi)))               # Kuramoto's critical coupling for N(0,1) frequencies
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    ax.plot(Ks, Rinf, "o-", color=PURPLE, ms=4)
    ax.axvline(Kc, color=RED, ls="--", lw=1); ax.text(Kc + 0.05, 0.05, f"K_c = {Kc:.2f}", color=RED, fontsize=9)
    ax.set_xlabel("coupling K"); ax.set_ylabel("order parameter R, long-time mean"); ax.set_title("a threshold, not a slope")
    ax = axes[1]
    for k, col in ((1.0, GREY), (2.0, ORANGE), (3.0, PURPLE)):
        ax.plot(np.arange(0, 60, 0.02), kuramoto(k), color=col, lw=1.5, label=f"K = {k:g}")
    ax.set_xlabel("time"); ax.set_ylabel("R(t)"); ax.set_title("locking in time"); ax.legend(fontsize=9)
    ax = axes[2]
    rng = np.random.default_rng(1)
    th = rng.uniform(0, 2 * np.pi, 60); om = rng.normal(0, 1, 60)
    for k, col, lab in ((1.0, GREY, "K = 1: incoherent"), (3.0, PURPLE, "K = 3: locked cluster")):
        t = th.copy()
        for _ in range(3000):
            z = np.exp(1j * t).mean(); t = t + 0.02 * (om + k * abs(z) * np.sin(np.angle(z) - t))
        ax.scatter(np.cos(t), np.sin(t), s=14, color=col, label=lab, alpha=0.8)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=GREY, lw=0.8)); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("final phases on the circle"); ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout(); save(fig, "cm_kuramoto.png")


def causal_attention_circle(n=200, beta=64.0, T=500.0, dt=0.05, seed=3):
    """Karagodin et al. (CSA) on S^1 with Q = K = V = I: angle dynamics."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    mask = np.tril(np.ones((n, n), dtype=bool))
    snaps, share_near_first, ts = {}, [], []
    want = {0.0, 5.0, 10.0, 75.0, 150.0, 500.0}
    t = 0.0
    for i in range(int(T / dt) + 1):
        if round(t, 6) in want:
            snaps[round(t, 6)] = th.copy()
        if i % 20 == 0:
            d = np.abs((th - th[0] + np.pi) % (2 * np.pi) - np.pi)
            share_near_first.append((d < 0.1).mean()); ts.append(t)
        c = np.cos(th[:, None] - th[None, :])
        Wt = np.exp(beta * (c - 1.0)) * mask
        Wt /= Wt.sum(1, keepdims=True)
        # tangential component of sum_j W_kj x_j at x_k: sum_j W_kj sin(th_j - th_k)
        th = th + dt * (Wt * np.sin(th[None, :] - th[:, None])).sum(1)
        t += dt
    return snaps, np.array(ts), np.array(share_near_first)


def fig_causal_attention():
    snaps, ts, share = causal_attention_circle()
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, tt in zip(axes[:3], (0.0, 75.0, 500.0)):
        th = snaps[tt]
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=GREY, lw=0.8))
        ax.scatter(np.cos(th[1:]), np.sin(th[1:]), s=10, color=BLUE, alpha=0.7, label="tokens 2..n")
        ax.scatter([np.cos(th[0])], [np.sin(th[0])], s=90, color=RED, marker="*", zorder=5, label="token 1 (frozen)")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"depth t = {tt:g}")
        if tt == 0.0:
            ax.legend(fontsize=8.5, loc="upper left")
    ax = axes[3]
    ax.plot(ts, 100 * share, color=RED, lw=2)
    ax.set_xlabel("depth t"); ax.set_ylabel("% of tokens within 0.1 rad of token 1")
    ax.set_title("metastable clusters, then the first token wins")
    fig.suptitle("Causal attention on the circle (Karagodin et al. 2024): n = 200, beta = 64, Q = K = V = I", y=1.03, fontsize=11)
    fig.tight_layout(); save(fig, "cm_causal_attention.png")


def fig_measured():
    data = {}
    for name, (tag, col) in ARMS.items():
        p = RES / _RUN.format(tag) / f"dsync_clustering_{name.lower().replace('.', '')}.json"
        if p.exists():
            data[name] = (json.load(open(p)), col)
    if not data:
        print("no D-sync JSON yet; skipping cm_measured.png"); return
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    names = list(data)
    ax = axes[0]
    groups = [(n, l) for n in names for l in ("0", "1")]
    xs = np.arange(len(groups)); wdt = 0.26
    for j, (lab, key, col) in enumerate((("context gating w", "H_w", BLUE), ("proximity exp(-e/2)", "H_prox", ORANGE), ("responsibility g", "H_g", RED))):
        vals = [data[n][0]["layers"][l][key][1] for n, l in groups]
        ax.bar(xs + (j - 1) * wdt, vals, width=wdt, color=col, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels([f"{n}\nlayer {l}" for n, l in groups], fontsize=9)
    ax.set_ylabel("effective number of wells (median of tokens)"); ax.axhline(1, color=GREY, ls="--", lw=1)
    ax.set_title("the two factors of g, by layer"); ax.legend(fontsize=8.5)
    ax = axes[1]
    for j, n in enumerate(names):
        for l, hatch in (("0", ""), ("1", "//")):
            r = data[n][0]["layers"][l]
            ax.bar(j + (-0.2 if l == "0" else 0.2), r["pr_rank"][1], width=0.38, color=data[n][1], alpha=0.9 if l == "1" else 0.5, hatch=hatch)
            ax.text(j + (-0.2 if l == "0" else 0.2), r["pr_rank"][1] + 0.4, f'{int(r["n_above_floor"][1])}', ha="center", fontsize=8, color=GREY)
    ax.axhline(32, color=GREY, ls="--", lw=1); ax.text(-0.4, 32.6, "nominal 32", color=GREY, fontsize=9)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names); ax.set_ylabel("participation-ratio rank of G (median)")
    ax.set_title("rank of G (light: layer 0, dark: layer 1;\nnumbers: eigenvalues above the floor)")
    ax = axes[2]
    for n in names:
        L = data[n][0]["layers"]
        ys = [L["0"]["tokens_in"]["cos_mean"], L["0"]["tokens_out"]["cos_mean"], L["1"]["tokens_out"]["cos_mean"]]
        ax.plot([0, 1, 2], ys, "o-", color=data[n][1], lw=2, label=n)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["embeddings", "after layer 0", "after layer 1"])
    ax.set_ylabel("mean pairwise cosine of token states"); ax.set_title("coherence with depth: transient, then undone"); ax.legend(fontsize=9)
    ax = axes[3]
    for n in names:
        r = data[n][0]["layers"]["1"]["tokens_in"]
        e = np.array(r["cos_edges"]); c = np.array(r["cos_hist"], float); c /= c.sum()
        ax.step(0.5 * (e[1:] + e[:-1]), c, where="mid", color=data[n][1], lw=1.8, label=f"{n} (mean {r['cos_mean']:+.2f})")
    ax.set_xlabel("pairwise cosine of token states entering layer 1"); ax.set_ylabel("share of pairs")
    ax.set_title("after layer 0: one shared direction, no clusters"); ax.legend(fontsize=9)
    fig.tight_layout(); save(fig, "cm_measured.png")


if __name__ == "__main__":
    fig_saturation()
    fig_kuramoto()
    fig_causal_attention()
    fig_measured()
