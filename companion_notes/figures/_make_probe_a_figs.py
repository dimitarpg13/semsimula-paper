"""Figures for mean_field/Probe_A_Field_Pair_Force.md.

Generates PNGs into figures/mean_field_probe_a/:

  pa_rho.png    -- the fraction of the pair force's removal cost the field recovers, rho_M, against the number of
                   components M, per model: the causal field (EMA weights, solid) and the context-free control
                   (fixed centres at fixed weights, dashed).
  pa_where.png  -- where the field's remaining cost falls (EMA weights, M = 8): share of the substitution cost
                   against share of the tokens, per copy-split category, one panel per model.
  pa_cos.png    -- the median cosine between the field force and the model's force, per layer (EMA, M = 8).

Numbers are read from notebooks/conservative_arch/scaleup/results/mean_field_probe_a/mean_field_probe_a_<name>.json
(debug/mean_field_probe_a.py, 2026-11-22). A model whose file is missing is skipped.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent / "notebooks/conservative_arch/scaleup/results/mean_field_probe_a"
OUT = HERE / "mean_field_probe_a"
OUT.mkdir(exist_ok=True)

MODELS = [("F3.1", "f31_owt", "#16A34A", "o"), ("SO-CODE", "so_code", "#D97706", "s"),
          ("L8-CODE", "l8_code", "#0891B2", "D"), ("FO-b-PMX-K16", "fobk16_owt", "#7C3AED", "^"),
          ("SR2", "sr2_owt", "#6B7280", "v"), ("PM1", "pm1_owt", "#DB2777", "P"),
          ("PMX-K16", "pmxk16_owt", "#2563EB", "X")]
WHERE = ["F3.1", "SO-CODE", "L8-CODE", "FO-b-PMX-K16", "SR2"]   # pa_where.png keeps these five
R = {}
for name, stem, _, _ in MODELS:
    p = RES / f"mean_field_probe_a_{stem}.json"
    if p.exists():
        R[name] = json.loads(p.read_text())
MODELS = [m for m in MODELS if m[0] in R]
CATS = [("first occurrence", "first\noccurrence"), ("repeat (not induction)", "repeat"), ("induction", "induction")]


def rho():
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for name, _, col, mk in MODELS:
        Ms = R[name]["Ms"]
        for vr, ls in (("ema", "-"), ("global", "--")):
            y = [R[name]["substitution"][f"{vr}|{M}"]["rho"] for M in Ms]
            ax.plot(Ms, y, ls=ls, color=col, marker=mk, ms=6, lw=2, mfc=col if vr == "ema" else "white")
        ax.annotate(name, (Ms[-1], R[name]["substitution"][f"ema|{Ms[-1]}"]["rho"]), xytext=(8, {"PM1": 6, "F3.1": -7}.get(name, 0)),
                    textcoords="offset points", va="center", fontsize=10, color="black")
    ax.axhline(0.75, color="black", lw=0.8, ls=":")
    ax.text(1, 0.755, "A1 threshold (F3.1, M = 8)", fontsize=9, va="bottom")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_xlim(0.8, 70); ax.set_ylim(0.35, 1.0)
    ax.set_xlabel("components M"); ax.set_ylabel(r"fraction recovered, $\rho_M$")
    ax.plot([], [], color="black", lw=2, label="causal field (EMA weights)")
    ax.plot([], [], color="black", lw=2, ls="--", label="context-free control")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("How much of the pair force a field reproduces")
    fig.tight_layout(); fig.savefig(OUT / "pa_rho.png"); plt.close(fig)


def where():
    ms = [m for m in MODELS if m[0] in WHERE]
    fig, axes = plt.subplots(1, len(ms), figsize=(2.9 * len(ms), 3.9), sharey=True)
    for ax, (name, _, col, _) in zip(axes, ms):
        s = R[name]["substitution"]["ema|8"]["cost_share"]; t = R[name]["token_share"]
        x = np.arange(len(CATS))
        ax.bar(x - 0.19, [t[c] for c, _ in CATS], 0.36, color="#D1D5DB", label="share of tokens")
        ax.bar(x + 0.19, [s[c] for c, _ in CATS], 0.36, color=col, label="share of the field's cost")
        ax.set_xticks(x, [l for _, l in CATS], fontsize=9); ax.axhline(0, color="black", lw=0.6)
        ax.set_title(name, fontsize=11)
    axes[0].set_ylabel("share"); axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.suptitle("Where the field's remaining cost falls (EMA weights, M = 8)", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "pa_where.png"); plt.close(fig)


def cos():
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for name, _, col, mk in MODELS:
        f = R[name]["fidelity"]
        ls_ = sorted(int(k.split("|")[-1]) for k in f if k.startswith("ema|8|"))
        y = [f[f"ema|8|{l}"]["cos_p10_p50_p90"][1] for l in ls_]
        if name == "SR2":                  # layer 1's pair force is near zero (scale 0.006): its cosine is not measurable
            ax.plot(ls_[:1], y[:1], color=col, marker=mk, ms=7, lw=2, label=name)
            ax.annotate("SR2 layer 1: force near zero,\ncosine not measurable", (0.1, 0.735), ha="left", va="center", fontsize=9)
            continue
        ax.plot(ls_, y, color=col, marker=mk, ms=7, lw=2, label=name)
    ax.set_xlabel("layer"); ax.set_ylabel("median cosine, field against model")
    ax.set_xticks(range(max(len(R[n]["layers"]) for n, *_ in MODELS)))
    ax.axhline(0.9, color="black", lw=0.8, ls=":"); ax.text(7, 0.905, "A2 threshold", ha="right", va="bottom", fontsize=9)
    ax.set_ylim(0.5, 1.02); ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Force direction per layer (EMA weights, M = 8)")
    fig.tight_layout(); fig.savefig(OUT / "pa_cos.png"); plt.close(fig)


if __name__ == "__main__":
    rho(); where(); cos()
    print("models", [m[0] for m in MODELS], "wrote", sorted(p.name for p in OUT.glob("*.png")))
