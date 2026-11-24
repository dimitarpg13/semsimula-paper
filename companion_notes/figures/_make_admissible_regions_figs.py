"""Figures for semantic_templates/Admissible_Regions_for_Template_Search.md.

Reads results/well_settling/settling_sr2np.json (debug/well_settling_map.py on SR2-NP's three checkpoints,
2026-11-24) and writes PNGs into figures/admissible_regions/:

  ar_settling.png  -- the dominant well's four drift measures (Jaccard, radial W1 drift, Spearman of the radius and
                      of the stiff coordinate) for the early pair (500 -> 22,500) and the late pair (22,500 -> best),
                      per layer; the dead wells' Spearman shown as the reference they set.
  ar_spread.png    -- the context spread of every well at the best checkpoint, in the well metric against Euclidean,
                      per layer; the dominant well marked.
  ar_dominance.png -- the median over tokens of the largest dominance share, per checkpoint and layer, with the
                      dominant well's share of tokens.
  ar_radial.png    -- the dominant well's radial law r_5 = sqrt(q_5) at the three checkpoints, and the same tokens'
                      r_5 at step 22,500 against the best checkpoint (the late pair's rank agreement), per layer;
                      reads settling_sr2np.npz.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
GREEN, BLUE, RED, PURPLE, GREY, ORANGE = "#22C55E", "#3B82F6", "#EF4444", "#8B5CF6", "#9CA3AF", "#F59E0B"
INK, MUTED = "#1F2937", "#6B7280"

HERE = Path(__file__).resolve().parent
OUT = HERE / "admissible_regions"
OUT.mkdir(exist_ok=True)
J = json.loads((HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/well_settling/settling_sr2np.json").read_text())
T = J["tables"]; CK = J["ckpts"]; LAYERS = (0, 1); KD = 5          # the dominant well of SR2-NP at both layers
PAIRS = [(f"{a}_to_{b}", f"{a.replace('step', '')} → {b.replace('step', '')}") for a, b in zip(CK[:-1], CK[1:])]


def settling():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    meas = [("jaccard", "Jaccard J"), ("w1", "radial drift Δ"), ("rho_r", "Spearman ρ_r"), ("rho_u", "Spearman ρ_u")]
    for ax, l in zip(axes, LAYERS):
        xs = np.arange(len(meas)); wd = 0.36
        for i, (key, lab) in enumerate(PAIRS):
            rows = T[f"L{l}_{key}"]; d = rows[KD]
            vals = [d[m] for m, _ in meas]
            ax.bar(xs + (i - 0.5) * wd, vals, wd, color=[GREY, GREEN][i], label=f"pair {lab}", edgecolor="white")
            for x_, v in zip(xs + (i - 0.5) * wd, vals):
                ax.text(x_, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
        dead = [np.nanmedian([r["rho_r"] for k, r in enumerate(T[f"L{l}_{PAIRS[1][0]}"]) if k != KD])]
        ax.hlines(dead, xs[2] - 0.5, xs[3] + 0.5, color=PURPLE, ls="--", lw=1.3)
        ax.text(xs[3] + 0.5, dead[0] + 0.02, "dead wells,\nlate pair", color=PURPLE, fontsize=8.5, ha="right", va="bottom")
        ax.set_xticks(xs, [lab for _, lab in meas]); ax.set_ylim(-0.15, 1.3)
        ax.axhline(0, color=INK, lw=0.8); ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"layer {l}, dominant well (k = {KD})", loc="left")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].set_ylabel("value (ideal: J → 1, Δ → 0, ρ → 1)")
    fig.suptitle("Training drift of the dominant well between SR2-NP's checkpoints", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "ar_settling.png"); plt.close(fig)


def spread():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, l in zip(axes, LAYERS):
        rows = T[f"L{l}_best"]
        for r in rows:
            k = r["k"]; dom = k == KD
            ax.scatter(r["s_eu"], r["s"], s=70 if dom else 40, color=GREEN if dom else GREY, zorder=3,
                       edgecolor="white", linewidth=0.8)
            ax.annotate(f"k{k}", (r["s_eu"], r["s"]), xytext=(6, 2), textcoords="offset points", fontsize=8.5,
                        color=INK if dom else MUTED)
        ax.axhline(1, color=ORANGE, lw=1.2, ls="--"); ax.axvline(1, color=ORANGE, lw=1.2, ls="--")
        ax.text(0.03, 1.03, "one width", color=ORANGE, fontsize=8.5, ha="left", va="bottom", transform=ax.get_yaxis_transform())
        ax.set_xlabel("Euclidean spread  s_eu  (centre motion / |mean centre|)")
        ax.set_ylabel("context spread  s  (well metric, widths)")
        ax.set_title(f"layer {l}, best checkpoint", loc="left"); ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, max(r["s_eu"] for r in rows) * 1.15); ax.set_ylim(0, max(r["s"] for r in rows) * 1.15)
    axes[0].text(0.02, 0.97, "green: the dominant well", transform=axes[0].transAxes, fontsize=9, color=GREEN, va="top")
    fig.suptitle("How far the context moves each well: in its own metric against in Euclidean terms", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "ar_spread.png"); plt.close(fig)


def dominance():
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    xs = np.arange(len(CK)); wd = 0.36
    for i, l in enumerate(LAYERS):
        share = [T[f"L{l}_{c}"][KD]["share"] for c in CK]
        pi = [np.nanmedian(T[f"L{l}_{c}"][KD]["pi"]) for c in CK]
        ax.bar(xs + (i - 0.5) * wd, share, wd, color=[GREEN, BLUE][i], label=f"layer {l}: share of tokens well {KD} dominates", edgecolor="white")
        ax.plot(xs + (i - 0.5) * wd, pi, "o", color=INK, ms=6, zorder=4)
        for x_, s_, p_ in zip(xs + (i - 0.5) * wd, share, pi):
            ax.text(x_, s_ + 0.03, f"{s_:.2f}", ha="center", fontsize=8.5, color=INK)
            ax.text(x_, p_ - 0.09, f"π {p_:.2f}", ha="center", fontsize=8, color=INK)
    ax.plot([], [], "o", color=INK, label=f"median dominance π of well {KD} on those tokens")
    ax.set_xticks(xs, [c.replace("step", "step ") if c != "best" else "best (31,000)" for c in CK])
    ax.set_ylim(0, 1.25); ax.set_ylabel("share"); ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("When the live set forms: the dominant well per checkpoint", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "ar_dominance.png"); plt.close(fig)


def radial():
    Z = np.load(HERE.parents[1] / "notebooks/conservative_arch/scaleup/results/well_settling/settling_sr2np.npz")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), gridspec_kw={"width_ratios": [1.25, 1]})
    cols = {CK[0]: GREY, CK[1]: BLUE, CK[2]: GREEN}
    rng = np.random.default_rng(0)
    for row, l in enumerate(LAYERS):
        ax = axes[row, 0]
        rs = {c: np.sqrt(Z[f"{c}_L{l}_q"][:, KD]) for c in CK}
        hi = max(np.percentile(r, 99.5) for r in rs.values())
        bins = np.linspace(0, hi, 70)
        for c in CK:
            lab = c.replace("step", "step ") if c != "best" else "best (31,000)"
            ax.hist(rs[c], bins=bins, histtype="step", lw=2, color=cols[c], density=True,
                    label=f"{lab}: median {np.median(rs[c]):.2f}")
        ax.set_xlabel("radius r_5 in widths of well 5"); ax.set_ylabel("density")
        ax.set_title(f"layer {l}: radial law of the dominant well", loc="left")
        ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
        ax = axes[row, 1]
        r0, r1 = rs[CK[1]], rs[CK[2]]
        idx = rng.choice(r0.size, 2500, replace=False)
        ax.scatter(r0[idx], r1[idx], s=5, color=GREEN, alpha=0.35, linewidths=0)
        lo_, hi_ = np.percentile(np.r_[r0, r1], [0.5, 99.5])
        ax.plot([lo_, hi_], [lo_, hi_], color=MUTED, lw=1, ls="--")
        ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_)
        rho = T[f"L{l}_{PAIRS[1][0]}"][KD]["rho_r"]
        ax.text(0.03, 0.96, f"Spearman ρ_r = {rho:.2f}\n(2,500 of 8,192 tokens shown)", transform=ax.transAxes,
                fontsize=9, va="top", color=INK)
        ax.set_xlabel("r_5 at step 22,500"); ax.set_ylabel("r_5 at best (31,000)")
        ax.set_title(f"layer {l}: the same tokens, late pair", loc="left"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "ar_radial.png"); plt.close(fig)


if __name__ == "__main__":
    settling(); spread(); dominance(); radial()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
