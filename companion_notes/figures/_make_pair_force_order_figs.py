"""Figures for depth_ladder/Pair_Force_in_the_First_Order_Step.md (2026-11-24).

Reads the three runs' evaluation logs (PMX-K16, FO-b-PMX-K16, FO-b-PMX-K16-NP) and the step probe's JSON files
(results/pair_force_order/, debug/pair_force_order_probe.py), and writes PNGs into figures/pair_force_order/:

  pfo_training_gap.png  the paired validation-loss gaps along training: the model trained without the pair force
                        against FO-b-PMX-K16, and FO-b-PMX-K16 against its second-order twin PMX-K16
  pfo_step.png          the pair force at the kick point, median over tokens: its cosine with the rest of the force
                        and its displacement against the layer input, per layer, at step 500 and at the best checkpoint
  pfo_position.png      by position: the cosine at layer 0 and the cost of switching the pair force off (reliance)
  pfo_takeover.png      what took the pair force's work: the cost of removing the wells and of swapping the context
                        channels, the stiffness of V_theta at layer 1, and the share of layer 1's step made by its
                        explicit kick, in the three models
"""
import json, re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
BLUE, ORANGE, GREEN, GREY = "#3B82F6", "#F59E0B", "#22C55E", "#9CA3AF"
INK, MUTED = "#1F2937", "#6B7280"
COL = {"PMX-K16": BLUE, "FO-b-PMX-K16": ORANGE, "FO-b-PMX-K16-NP": GREEN}

HERE = Path(__file__).resolve().parent
OUT = HERE / "pair_force_order"; OUT.mkdir(exist_ok=True)
RES = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results"
PFX = "cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_"
SFX = "_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn"
LOGS = {"PMX-K16": RES / f"{PFX}vplive_xilive_pm16_pmx16_sr2{SFX}/L2_PM16-PMX16_on_OWT_32500steps_output.txt",
        "FO-b-PMX-K16": RES / f"{PFX}vplive_xilive_pm16_pmx16_sr2_fob{SFX}/L2_FO-b-PM16-PMX16_on_OWT_32500steps_output.txt",
        "FO-b-PMX-K16-NP": RES / f"{PFX}nophi_xilive_pm16_pmx16_sr2_fob{SFX}/L2_FO-b-PMX-K16-NP_0.5B_OWT_32500steps_output.txt"}
PROBE = RES / "pair_force_order"


def evals(p):
    out = {}
    for line in p.read_text().splitlines():
        m = re.search(r">>> EVAL step ([\d,]+)\s+val_loss=([\d.]+)", line)
        if m:
            out[int(m.group(1).replace(",", ""))] = float(m.group(2))
    return out


def probe(name):
    return json.loads((PROBE / f"pair_force_order_{name}.json").read_text())


def training_gap():
    ev = {k: evals(p) for k, p in LOGS.items()}
    steps = sorted(set.intersection(*(set(v) for v in ev.values())))
    s = np.array(steps)
    d_np = np.array([ev["FO-b-PMX-K16-NP"][t] - ev["FO-b-PMX-K16"][t] for t in steps])
    d_fo = np.array([ev["FO-b-PMX-K16"][t] - ev["PMX-K16"][t] for t in steps])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axvspan(0, 1625, color=GREY, alpha=0.15, lw=0); ax.axvspan(21125, 32500, color=GREY, alpha=0.15, lw=0)
    ax.text(800, 0.021, "warm-up", ha="center", fontsize=8.5, color=MUTED)
    ax.text(26800, 0.021, "decay", ha="center", fontsize=8.5, color=MUTED)
    ax.axhline(0, color=INK, lw=0.8)
    ax.plot(s, d_np, "o-", ms=3, lw=1.6, color=GREEN, label="trained without the pair force − FO-b-PMX-K16")
    ax.plot(s, d_fo, "o-", ms=3, lw=1.6, color=ORANGE, label="FO-b-PMX-K16 − PMX-K16 (first order − second order)")
    ax.set_xlabel("training step"); ax.set_ylabel("validation loss gap (nats)")
    ax.set_ylim(-0.035, 0.025); ax.set_xlim(0, 32700)
    ax.legend(frameon=False, fontsize=9, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Paired validation-loss gaps along training (same batches, same seed)", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "pfo_training_gap.png"); plt.close(fig)


def step_readings():
    runs = [("PMX-K16\nstep 500", "pmxk16_step500", BLUE, 0.45), ("PMX-K16\nbest", "pmxk16_best", BLUE, 1.0),
            ("FO-b-PMX-K16\nstep 500", "fobk16_step500", ORANGE, 0.45), ("FO-b-PMX-K16\nbest", "fobk16_best", ORANGE, 1.0)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, layer in enumerate(("0", "1")):
        for row, (key, lab) in enumerate((("cos", "cosine of the pair force\nwith the rest of the force"),
                                          ("disp_phi_rel_h", "pair-force displacement\n|c f_phi| / |h_in|"))):
            ax = axes[row, col]
            vals = [probe(n)["readings_p05_p50_p95_mean"][layer][key][1] for _, n, _, _ in runs]
            xs = np.arange(len(runs))
            ax.bar(xs, vals, 0.62, color=[c for *_, c, _ in runs], alpha=1.0, edgecolor="white")
            for b, (_, _, _, a) in zip(ax.patches, runs):
                b.set_alpha(a)
            top = 0.75 if key == "cos" else max(vals) * 1.15
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.015 * top, f"{v:.3f}", ha="center", fontsize=9, color=INK)
            ax.set_xticks(xs, [r[0] for r in runs], fontsize=8.5); ax.axhline(0, color=INK, lw=0.8)
            ax.set_ylabel(lab, fontsize=9.5); ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(f"layer {layer}", loc="left")
            ax.set_ylim(-0.1 if key == "cos" else 0, top)
    fig.suptitle("The pair force at the kick point, median over tokens (h_in at layer 0 is the embedding)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "pfo_step.png"); plt.close(fig)


def position():
    fo, so = probe("fobk16_best"), probe("pmxk16_best")
    b = fo["position_buckets"]; labs = [f"{lo}" if hi - lo == 1 else f"{lo}–{hi - 1}" for lo, hi in zip(b[:-1], b[1:])]
    xs = np.arange(len(labs))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    for name, d, c in (("FO-b-PMX-K16", fo, ORANGE), ("PMX-K16", so, BLUE)):
        ax.plot(xs[1:], d["readings_by_position"]["0"]["cos"][1:], "o-", color=c, lw=1.8, label=name)
    ax.set_title("layer 0: cosine with the rest of the force", loc="left"); ax.set_ylim(0, 0.75)
    ax = axes[1]
    for name, d, c in (("FO-b-PMX-K16", fo, ORANGE), ("PMX-K16", so, BLUE)):
        lp = d["loss_by_position"]
        ax.plot(xs, np.array(lp["vphi_off"]) - np.array(lp["base"]), "o-", color=c, lw=1.8, label=name)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_title("cost of switching the pair force off (nats)", loc="left")
    for ax in axes:
        ax.set_xticks(xs, labs, rotation=45, fontsize=8.5); ax.set_xlabel("position in the window")
        ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "pfo_position.png"); plt.close(fig)


def takeover():
    # sources: mechanism_ablations_*_output.txt, vtheta_omega_dt_*_output.txt, refinement_flow_confirmation_*_output.txt
    names = ["PMX-K16", "FO-b-PMX-K16", "FO-b-PMX-K16-NP"]
    panels = [("removing the wells (%)", [155.53, 343.44, 382.54]),
              ("swapping the context channels (%)", [333.91, 337.46, 349.36]),
              ("V_θ stiffest mode at layer 1\n(median ω·Δt)", [4.554, 5.311, 6.432]),
              ("layer 1: share of the step\nmade by the explicit kick", [0.390, 0.417, 0.585])]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, (lab, vals) in zip(axes, panels):
        xs = np.arange(3)
        ax.bar(xs, vals, 0.62, color=[COL[n] for n in names], edgecolor="white")
        for x, v in zip(xs, vals):
            ax.text(x, v * 1.02, f"{v:g}", ha="center", fontsize=9, color=INK)
        ax.set_xticks(xs, ["PMX-K16", "FO-b-\nPMX-K16", "FO-b-PMX-\nK16-NP"], fontsize=8.5)
        ax.set_title(lab, loc="left", fontsize=10); ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.text(0.99, 0.01, "PMX-K16's kick share counts the position part only", ha="right", fontsize=7.5, color=MUTED)
    fig.suptitle("What took the pair force's work (best checkpoints, inference readings)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "pfo_takeover.png"); plt.close(fig)


if __name__ == "__main__":
    training_gap(); step_readings(); position(); takeover()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
