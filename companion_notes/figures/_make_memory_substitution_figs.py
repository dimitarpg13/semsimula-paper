#!/usr/bin/env python3
"""Figures for the memory-substitution hypothesis (2026-10-22): the future note §7, the protocol §5.13 (MS)
and paper v6 §17h.

    python3 _make_memory_substitution_figs.py      # writes figures/memory_substitution/*.png

  ms_two_memories.png            -- schematic: the implicit memory of a second-order step (the velocity between
                                    layers, the rotation inside a step) and the explicit memory of the Poisson-mode
                                    wells (an exponentially decaying trace of past tokens); retention curves from
                                    the trained constants (gamma_0 = 0.1, dt = 4; trained half-lives 3-20 tokens)
  ms_substitution_evidence.png   -- recorded: the 2 x 2 of order against wells (settled perplexities), Gate 1
                                    reliance by arm, and the first-order gap through training on both bases
                                    (from the run logs in ~/Downloads)
  ms_wells_compare.png           -- recorded: Cell 6b-15 on PMX and FO-b-PMX: well depths, half-lives, and the
                                    force share by half-life quartile
  ms_dose_response_plan.png      -- the pre-registered prediction of MS2: the second-order gap against the number
                                    of modes, under substitution and under the alternative (illustrative)
"""
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "memory_substitution"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150})
C2, C1, CW = "#1b7837", "#d95f02", "#2166ac"      # second order, first order, wells

# ---------------------------------------------------------------- 1. the two memories (schematic)
fig = plt.figure(figsize=(13, 7.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.45, wspace=0.28)

# A. the second-order step as a two-term recurrence, and its retention
ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.set_title("A. The implicit memory of a second-order step", loc="left", weight="bold")
for i, (x, lab) in enumerate(((1.2, r"$h_{\ell-1}$"), (4.5, r"$h_{\ell}$"), (7.8, r"$h_{\ell+1}$"))):
    ax.add_patch(plt.Circle((x, 4.3), 0.42, fc="#d9f0d3" if i < 2 else "white", ec=C2, lw=1.5))
    ax.text(x, 4.3, lab, ha="center", va="center", fontsize=11)
ax.add_patch(FancyArrowPatch((1.65, 4.3), (4.05, 4.3), arrowstyle="-|>", mutation_scale=14, color="0.3"))
ax.add_patch(FancyArrowPatch((4.95, 4.3), (7.35, 4.3), arrowstyle="-|>", mutation_scale=14, color="0.3"))
ax.add_patch(FancyArrowPatch((1.2, 3.85), (7.8, 3.85), arrowstyle="-|>", mutation_scale=14, color=C2,
                             connectionstyle="arc3,rad=0.35", lw=1.6))
ax.text(4.5, 2.05, r"$v_{\ell} = (h_{\ell} - h_{\ell-1})/\Delta t$: the step reads two past states", ha="center", fontsize=9, color=C2)
ax.text(4.5, 5.35, "second order: a two-term recurrence", ha="center", fontsize=9.5)
ax.text(4.5, 1.3, "first order: one term, no memory of where the token came from", ha="center", fontsize=9, color=C1)
ax.text(4.5, 0.55, "inside a step: the stiff modes rotate; area kept $e^{-\\gamma_i \\Delta t}$ (distinctions kept)", ha="center", fontsize=9)

# B. the wells as an explicit memory: a decaying trace of past tokens
ax = fig.add_subplot(gs[0, 1]); ax.set_title("B. The explicit memory of the Poisson-mode wells", loc="left", weight="bold")
lam = 0.5 ** (1 / 8.4)                                     # the trained median half-life, 8.4 tokens
s = np.arange(-16, 0); rng = np.random.default_rng(5)
E = rng.uniform(0.0, 1.0, len(s)) * (rng.uniform(0, 1, len(s)) < 0.45)
w = lam ** (-s - 1) * E
ax.bar(s, E, color="0.82", width=0.8, label="overlap $E_v(s)$ of token $s$ with mode $v$")
ax.bar(s, w, color=CW, width=0.8, label="what survives to token $t$: $\\lambda_v^{\\,t-1-s} E_v(s)$")
ax.plot(s, lam ** (-s - 1), color=CW, ls="--", lw=0.9)
ax.text(-10.3, 0.50, "survival $\\lambda_v^{\\,t-1-s}$,\nhalf-life 8.4 tokens\n(trained median)", fontsize=8, color=CW)
ax.set(xlabel="position $s$ relative to the current token $t$", ylabel="deposit", xlim=(-16.6, 0.6), ylim=(0, 1.08))
ax.text(-0.3, 1.0, f"$\\bar n_v(t) = \\sum_s \\lambda_v^{{t-1-s}} E_v(s) = {w.sum():.2f}$\n→ a well of depth $\\bar n_v \\varepsilon_{{l,v}}$",
        ha="right", va="top", fontsize=9)
ax.legend(fontsize=7.5, loc="upper left")

# C. retention of the implicit memory, through the stack
ax = fig.add_subplot(gs[1, 0]); ax.set_title("C. How long each memory lasts", loc="left", weight="bold")
t = np.linspace(0, 8, 200); g0 = 0.1
ax.plot(t, np.exp(-g0 * t), color=C2, lw=1.8, label="velocity on the complement: $e^{-\\gamma_0 t}$")
ax.plot(t, np.exp(-g0 * t / 2), color=C2, lw=1.2, ls="--", label="amplitude of a stiff mode: $e^{-\\gamma_0 t/2}$")
for xx in (4, 8):
    ax.axvline(xx, color="0.7", lw=0.7, ls=":")
ax.text(4.05, 0.95, "layer 1", fontsize=8, color="0.4"); ax.text(8.05, 0.95, "output", fontsize=8, color="0.4", ha="left")
ax.text(8.0, np.exp(-0.8) - 0.07, f"{np.exp(-0.8):.0%}", fontsize=8, color=C2, ha="right")
ax.text(8.0, np.exp(-0.4) + 0.03, f"{np.exp(-0.4):.0%}", fontsize=8, color=C2, ha="right")
ax.set(xlabel="integration time $t$ (one stack: $T = L\\,\\Delta t = 8$)", ylabel="fraction kept", xlim=(0, 8.6), ylim=(0, 1.08))
ax.legend(fontsize=7.5, loc="lower left")
ax.text(0.15, 0.36, "the implicit memory: the token's own past,\none layer boundary at $L = 2$", fontsize=8, color=C2)

ax = fig.add_subplot(gs[1, 1])
k = np.arange(0, 41)
for hl, c, lab in ((3.2, "#9ecae1", "p05: 3.2 tokens"), (8.4, CW, "median: 8.4 tokens"), (19.8, "#08306b", "p95: 19.8 tokens")):
    ax.plot(k, 0.5 ** (k / hl), color=c, lw=1.6, label=lab)
ax.axhline(0.5, color="0.7", lw=0.7, ls=":")
ax.set(xlabel="tokens since the deposit", ylabel="fraction kept", xlim=(0, 40), ylim=(0, 1.08))
ax.legend(fontsize=7.5, loc="upper right", title="trained half-lives (FO-b-PMX)", title_fontsize=7.5)
ax.text(26, 0.15, "the explicit memory:\nother tokens' pasts,\nacross the sequence", fontsize=8, color=CW)
fig.savefig(OUT / "ms_two_memories.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 2. the evidence (recorded)
D = Path.home() / "Downloads"
def ev_txt(f):
    t = open(f, errors="replace").read()
    return {int(m.group(1).replace(",", "")): float(m.group(2)) for m in re.finditer(r">>> EVAL step ([\d,]+)\s+val_loss=[\d.]+\s+val_ppl=([\d.]+)", t)}
def ev_jsonl(f):
    d = {}
    for l in open(f):
        if "val_ppl" in l:
            r = json.loads(l)
            if r.get("val_ppl") is not None: d[int(r["step"])] = float(r["val_ppl"])
    return d
SR2F = D / ("semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_L2probe_ob_untied_wsd_e5c_"
            "plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/results/L2_SR2_for_noattn_none_32500steps_output.txt")
PMXF = D / ("semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64_pmx16_sr2_L2probe_ob_untied_wsd_e5c_"
            "plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/results/training_log.jsonl")
have = SR2F.exists() and PMXF.exists() and (D / "L2_FO-b-SR2_OWT_32500steps_output.txt").exists() and (D / "semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64_pmx16_sr2_fob_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn" / "results" / "L2_FO-b-PMX_OWT_32500steps_output.txt").exists()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
# A. the 2 x 2
x = [0, 1]
ax[0].plot(x, [58.32, 54.63], "o-", color=C2, lw=2, ms=8, label="second order (SR2, PMX)")
ax[0].plot(x, [61.16, 54.02], "s-", color=C1, lw=2, ms=8, label="first order (FO-b-OWT, FO-b-PMX)")
for xx, y2, y1, lab in ((0, 58.32, 61.16, "+4.9%"), (1, 54.63, 54.02, "−1.1%")):
    ax[0].annotate("", xy=(xx, y1), xytext=(xx, y2), arrowprops=dict(arrowstyle="<->", color="0.3"))
    ax[0].text(xx + (0.05 if xx else -0.07), (y1 + y2) / 2, f"first order\n{lab}", fontsize=8.5, va="center", ha="left" if xx else "right")
for xx, y, v in ((0, 58.32, "58.32"), (0, 61.16, "61.16"), (1, 54.63, "54.63"), (1, 54.02, "54.02")):
    ax[0].text(xx - 0.05, y, v, fontsize=8, ha="right", va="center")
ax[0].set_xticks(x, ["no wells\n(SR2 base)", "64 Poisson-mode wells\n(PMX base)"]); ax[0].set_xlim(-0.45, 1.6); ax[0].set_ylim(53, 62.5)
ax[0].set_ylabel("settled perplexity, OpenWebText"); ax[0].set_title("A. Order against wells (recorded)")
ax[0].legend(fontsize=8, loc="upper right")
ax[0].text(0.5, 53.4, "the wells help first order by 11.7%, second order by 6.3%", ha="center", fontsize=8, color="0.3")
# B. Gate 1 reliance
arms = ["SR2", "FO-b-OWT", "PMX", "FO-b-PMX"]; g1 = [3.9, 0.0, 91.3, 0.0]; cols = [C2, C1, C2, C1]
ax[1].bar(arms, g1, color=cols); ax[1].set_yscale("symlog", linthresh=5)
for i, v in enumerate(g1): ax[1].text(i, v + (3 if v > 0 else 0.5), f"+{v:.1f}%" if v else "0", ha="center", fontsize=8.5)
ax[1].set_ylabel("perplexity cost of resetting the velocity (Gate 1)"); ax[1].set_title("B. Reliance on inertia, not need"); ax[1].set_ylim(0, 300)
ax[1].text(1.5, 150, "PMX leans on its velocity 23× more than SR2,\nyet its first-order twin matches it", ha="center", fontsize=8.5)
# C. the first-order gap through training, both bases
if have:
    E = {"SR2": ev_txt(SR2F), "FOB": ev_txt(D / "L2_FO-b-SR2_OWT_32500steps_output.txt"), "PMX": ev_jsonl(PMXF), "FOBPMX": ev_txt(D / "semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64_pmx16_sr2_fob_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn" / "results" / "L2_FO-b-PMX_OWT_32500steps_output.txt")}
    for a, b, c, lab in (("FOB", "SR2", C1, "no wells: FO-b-OWT against SR2"), ("FOBPMX", "PMX", CW, "wells: FO-b-PMX against PMX")):
        ks = sorted(set(E[a]) & set(E[b]))
        ax[2].plot(ks, [100 * (E[a][k] / E[b][k] - 1) for k in ks], color=c, lw=1.5, label=lab)
ax[2].axhline(0, color="k", lw=0.8); ax[2].axvspan(21125, 32500, color="0.92"); ax[2].text(21500, 7.0, "decay", fontsize=8)
ax[2].set(xlabel="training step", ylabel="first-order model against its second-order twin (%)", xlim=(0, 32500), ylim=(-8, 8))
ax[2].set_title("C. The first-order gap through training"); ax[2].legend(fontsize=8, loc="upper left")
ax[2].text(1500, -7.2, "below 0: the first-order model is ahead", fontsize=8, color="0.4")
fig.tight_layout(); fig.savefig(OUT / "ms_substitution_evidence.png"); plt.close(fig)

# ---------------------------------------------------------------- 3. the wells, PMX against FO-b-PMX (Cell 6b-15)
R = {"PMX": dict(depth={0: (-0.072, -0.001, 0.045), 1: (-0.537, -0.009, 0.546)}, pos={0: 50, 1: 48}, hl=(3.1, 9.1, 21.2),
                 share={0: (9.3, 44.6), 1: (16.1, 37.9)}, pmf={0: 0.812, 1: 4.793}),
     "FO-b-PMX": dict(depth={0: (-0.095, 0.031, 0.121), 1: (-0.217, 0.071, 0.583)}, pos={0: 64, 1: 61}, hl=(3.2, 8.4, 19.8),
                      share={0: (13.7, 33.3), 1: (38.0, 19.1)}, pmf={0: 0.605, 1: 3.323})}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for j, (arm, c) in enumerate((("PMX", C2), ("FO-b-PMX", C1))):
    for li in (0, 1):
        lo, md, hi = R[arm]["depth"][li]; xx = li + (j - 0.5) * 0.3
        ax[0].plot([xx, xx], [lo, hi], color=c, lw=2.5, solid_capstyle="round"); ax[0].plot(xx, md, "o", color="white", mec=c, ms=8, mew=2)
        ax[0].text(xx, hi + 0.03, f"{R[arm]['pos'][li]}% > 0", ha="center", fontsize=8, color=c)
    ax[0].plot([], [], color=c, lw=2.5, label=arm)
ax[0].axhline(0, color="k", lw=0.8); ax[0].set_xticks([0, 1], ["layer 0", "layer 1"]); ax[0].set_xlim(-0.6, 1.6)
ax[0].set_ylabel("well depth $\\varepsilon_{l,v}$ (p05 – median – p95 over the 64 modes)"); ax[0].set_title("A. Well depths: attractive or repulsive")
ax[0].legend(fontsize=8, loc="upper left"); ax[0].text(0.5, -0.5, "> 0 attractive, < 0 repulsive", ha="center", fontsize=8, color="0.4")
for j, (arm, c) in enumerate((("PMX", C2), ("FO-b-PMX", C1))):
    lo, md, hi = R[arm]["hl"]
    ax[1].plot([lo, hi], [j, j], color=c, lw=2.5, solid_capstyle="round"); ax[1].plot(md, j, "o", color="white", mec=c, ms=8, mew=2)
    ax[1].text(hi + 0.5, j, f"median {md}", va="center", fontsize=8.5, color=c)
ax[1].axvspan(4, 128, color="0.93"); ax[1].text(60, 1.45, "initial range 4–128", fontsize=8, color="0.4", ha="center")
ax[1].set_yticks([0, 1], ["PMX", "FO-b-PMX"]); ax[1].set_xscale("log"); ax[1].set_xlim(2, 140); ax[1].set_ylim(-0.6, 1.8)
ax[1].set_xlabel("half-life of a mode (tokens), p05 – median – p95"); ax[1].set_title("B. Half-lives: the same horizons")
w = 0.18
for j, (arm, c) in enumerate((("PMX", C2), ("FO-b-PMX", C1))):
    for li in (0, 1):
        sh, lg = R[arm]["share"][li]; base = li * 2.2
        ax[2].bar(base + j * 0.9 - 0.2, sh, w * 2, color=c, alpha=0.45); ax[2].bar(base + j * 0.9 + 0.2, lg, w * 2, color=c)
        ax[2].text(base + j * 0.9, max(sh, lg) + 1.5, arm, ha="center", fontsize=7.5, color=c)
ax[2].axhline(25, color="0.6", ls=":", lw=0.8); ax[2].text(3.3, 25.7, "balanced", fontsize=7.5, color="0.4", ha="right")
ax[2].set_xticks([0.45, 2.65], ["layer 0", "layer 1"]); ax[2].set_ylabel("share of the wells' force (%)")
ax[2].set_title("C. Force share by half-life: short (pale), long (solid)")
ax[2].text(2.65, 47, "PMX: long-lived modes carry the force at both layers\nFO-b-PMX: long-lived at layer 0, short-lived at layer 1", ha="center", fontsize=8)
ax[2].set_ylim(0, 54)
fig.tight_layout(); fig.savefig(OUT / "ms_wells_compare.png"); plt.close(fig)

# ---------------------------------------------------------------- 4. the pre-registered dose-response (illustrative)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
K = np.array([0, 16, 64, 256]); xk = np.arange(len(K))
ax.plot(xk, [4.9, 2.0, -1.1, -1.5], "o-", color=C2, lw=2, label="MS2, substitution: the gap closes as the explicit memory grows")
ax.plot(xk, [4.9, 4.5, 4.9, 4.7], "s--", color="0.5", lw=1.5, label="the alternative: the wells help both orders alike, the gap stays")
ax.plot(0, 4.9, "o", color="k", ms=9, zorder=5); ax.plot(2, -1.1, "o", color="k", ms=9, zorder=5)
ax.text(0.08, 5.3, "recorded: SR2 / FO-b-OWT", fontsize=8); ax.text(2.08, -0.7, "recorded: PMX / FO-b-PMX", fontsize=8)
ax.axhline(0, color="k", lw=0.8); ax.axhspan(-2, 2, color="0.94"); ax.text(3.0, 1.3, "seed-spread band (to be measured, MS1)", fontsize=8, color="0.4", ha="right")
ax.set_xticks(xk, [f"K = {k}" for k in K]); ax.set_xlabel("number of Poisson modes (the size of the explicit memory)")
ax.set_ylabel("second-order advantage over the first-order twin (%)"); ax.set_ylim(-3, 6.5)
ax.set_title("The pre-registered dose-response (MS2): two points recorded, shapes illustrative", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "ms_dose_response_plan.png"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")), "| run logs found:", have)
