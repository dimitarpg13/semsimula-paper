#!/usr/bin/env python3
"""Figures for Tech_Report_Position_Dependent_Damping_CfC_BAOAB.md (2026-10-21).

    python3 _make_tr_damping_figs.py      # writes figures/tr_damping/*.png

  tr_layer_step.png              -- schematic: the A-B-O-A layer step of SR2 and where the friction of the
                                    stiff modes and of the complement enters (no data)
  tr_parameterisation_family.png -- the damping rate of a stiff mode, and the amplitude it keeps over a layer
                                    step, against its stiffness, for every parameterisation of the report;
                                    computed from the formulas, with the learned arm's settled medians
                                    (0.36 and 0.56, protocol 5.13) as the band
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "tr_damping"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150})

# ---------------------------------------------------------------- 1. the layer step
fig, ax = plt.subplots(figsize=(13, 5.6))
ax.set_xlim(0, 13.7); ax.set_ylim(0, 5.6); ax.axis("off")
cols = [("A", "Δt/2", 1.3), ("B", "Δt", 4.3), ("O", "Δt", 7.3), ("A", "Δt/2", 10.3)]
txt_U = ["exact damped oscillator\nper mode, rate γᵢ\n(the friction is here)",
         "kick: the force the\nA sub-steps do not carry",
         "undone: the friction\nwas applied in A",
         "exact damped oscillator\nper mode, rate γᵢ\n(the same γᵢ)"]
txt_C = ["free drift\nh ← h + (Δt/2) v",
         "kick: the full force\nv ← v + (Δt/m) f",
         "friction γ₀ = 0.1\nv ← e^(−γ₀Δt) v",
         "free drift\nh ← h + (Δt/2) v"]
for (name, dur, x), tu, tc in zip(cols, txt_U, txt_C):
    ax.add_patch(FancyBboxPatch((x, 0.55), 2.3, 3.75, boxstyle="round,pad=0.05", fc="0.97", ec="0.3"))
    ax.text(x + 1.15, 4.0, f"{name}   ({dur})", ha="center", fontsize=12, weight="bold")
    ax.add_patch(FancyBboxPatch((x + 0.1, 2.25), 2.1, 1.45, boxstyle="round,pad=0.03", fc="#fde0c5", ec="none"))
    ax.add_patch(FancyBboxPatch((x + 0.1, 0.7), 2.1, 1.4, boxstyle="round,pad=0.03", fc="#d9e8f5", ec="none"))
    ax.text(x + 1.15, 2.97, tu, ha="center", va="center", fontsize=8.5)
    ax.text(x + 1.15, 1.4, tc, ha="center", va="center", fontsize=8.5)
for x0, x1 in ((3.6, 4.3), (6.6, 7.3), (9.6, 10.3)):
    ax.add_patch(FancyArrowPatch((x0, 2.4), (x1, 2.4), arrowstyle="-|>", mutation_scale=14, color="0.3"))
ax.text(0.65, 2.97, "on span(U):\nthe q = 16\nstiff modes", ha="center", va="center", fontsize=8.5, color="#a6611a")
ax.text(0.65, 1.4, "on the\ncomplement\nof span(U)", ha="center", va="center", fontsize=8.5, color="#2166ac")
ax.text(13.25, 2.4, "→ h_new\n(LayerNorm\nprojection)", ha="center", va="center", fontsize=8.5)
# the damping head, feeding both A sub-steps
ax.add_patch(FancyBboxPatch((4.6, 4.75), 3.8, 0.6, boxstyle="round,pad=0.05", fc="#d9f0d3", ec="#1b7837"))
ax.text(6.5, 5.05, "layer input h → ζ(h) (head or constant) → γᵢ = γ₀ + 2ζ ωᵢ", ha="center", va="center", fontsize=9)
for x0, xa in ((4.55, 2.45), (8.45, 11.45)):
    ax.add_patch(FancyArrowPatch((x0, 4.95), (xa, 4.35), arrowstyle="-|>", mutation_scale=12, color="#1b7837"))
ax.text(6.85, 0.15, "U and the stiffnesses κᵢ are computed once, at the layer input, and held fixed through the step; "
        "so is ζ. Velocity enters decoded from (h, h_prev) and leaves encoded in h_prev.",
        ha="center", fontsize=8.5, color="0.3")
fig.tight_layout(); fig.savefig(OUT / "tr_layer_step.png"); plt.close(fig)

# ---------------------------------------------------------------- 2. the family of parameterisations
dt, g0 = 4.0, 0.1
wdt = np.linspace(0.05, 14, 400); w = wdt / dt
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
fams = [("SR2: γ₀ for every mode", np.full_like(w, g0), "k", "-"),
        ("SR3-crit: γ₀ + 2ωᵢ (ζ = 1)", g0 + 2 * w, "#b2182b", "-"),
        ("critical rate 2ωᵢ (no floor)", 2 * w, "0.5", ":")]
for lab, gam, c, ls in fams:
    ax[0].plot(wdt, gam, color=c, ls=ls, lw=1.6, label=lab)
    if not lab.startswith("critical"):
        ax[1].plot(wdt, np.exp(-gam * dt / 2), color=c, ls=ls, lw=1.6, label=lab)
lo, hi = g0 + 2 * 0.36 * w, g0 + 2 * 0.56 * w
ax[0].fill_between(wdt, lo, hi, color="#1b9e77", alpha=0.3, label="learned ζ(h): settled medians 0.36 to 0.56")
ax[1].fill_between(wdt, np.exp(-hi * dt / 2), np.exp(-lo * dt / 2), color="#1b9e77", alpha=0.3,
                   label="learned ζ(h): settled medians 0.36 to 0.56")
rng = np.random.default_rng(3)
wd_pm = np.sort(rng.uniform(1.0, 13.0, 16)); z_pm = np.exp(rng.normal(np.log(0.3), 1.0, 16))
ax[0].plot(wd_pm, g0 + 2 * z_pm * wd_pm / dt, "o", color="#7570b3", ms=4, label="per-mode ζᵢ(h), D5: one free ratio per mode (illustrative)")
for a in ax:
    a.axvline(7.9, color="0.6", ls="--", lw=0.8)
ax[0].text(8.1, 0.125, "median stiff mode (SR2, γ(h))", fontsize=8, color="0.4")
ax[0].set_yscale("log"); ax[0].set_ylim(0.01, 10)
ax[0].set(xlabel="stiffness of the mode, ωᵢΔt (Δt = 4)", ylabel="damping rate γᵢ", title="A. The damping rate of stiff mode i")
ax[0].legend(fontsize=7.5, loc="lower right")
ax[0].text(0.3, 5.0, "FO-b, the overdamped limit (no velocity in any direction),\nlies above this panel: γᵢ → ∞ on every direction", fontsize=8, color="0.3")
ax[1].set_yscale("log"); ax[1].set_ylim(1e-5, 1.5)
ax[1].set(xlabel="stiffness of the mode, ωᵢΔt (Δt = 4)", ylabel="amplitude kept over one layer step",
          title="B. What the mode keeps: e^(−γᵢΔt/2)")
ax[1].legend(fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "tr_parameterisation_family.png"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
