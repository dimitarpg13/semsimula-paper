"""Figures for mean_field/Pair_Force_By_Layer_L8_CODE.md.

Generates figures/pair_force_layers/pfl_layers.png: four panels over the eight layers of L8-CODE (step 18,000):
  (a) the cost of switching the pair force off in that layer alone, all positions, with the standard error over
      windows (debug/recall_pathway_probe.py arms=phioff_layers);
  (b) the median cosine between Probe A's causal field and the model's force at that layer (EMA weights, M = 8;
      debug/mean_field_probe_a.py);
  (c) the trained per-layer scale of the pair force, softplus(raw_v_phi_scale), read from the checkpoint;
  (d) how often the layer's router selects the earlier occurrence at far recall, as a multiple of chance.

Numbers are read from notebooks/conservative_arch/scaleup/results/recall_pathway/recall_pathway_l8_code_phioff_layers.json,
results/recall_pathway/recall_pathway_l8_code_phioff_deep.json (layers 1-7 off together, arms=phioff_deep) and
results/mean_field_probe_a/mean_field_probe_a_l8_code.json; the scales are the checkpoint's (2026-11-22).
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

TEAL = "#0891B2"
GREY = "#6B7280"

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent / "notebooks/conservative_arch/scaleup/results"
OUT = HERE / "pair_force_layers"
OUT.mkdir(exist_ok=True)

RP = json.loads((RES / "recall_pathway/recall_pathway_l8_code_phioff_layers.json").read_text())
PA = json.loads((RES / "mean_field_probe_a/mean_field_probe_a_l8_code.json").read_text())
DP = json.loads((RES / "recall_pathway/recall_pathway_l8_code_phioff_deep.json").read_text())
SCALE = [0.0241, 0.0446, 0.0416, 0.0508, 0.0895, 0.1027, 0.2713, 1.6425]   # softplus(raw_v_phi_scale), step 18,000
NL = 8
X = np.arange(NL)


def layers():
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), sharex=True)
    a, b, c, d = axes.flat
    cost = [RP["arms"][f"phioff@{L}"]["all"] for L in range(NL)]
    a.bar(X, [m for m, _ in cost], 0.6, yerr=[e for _, e in cost], color=TEAL, capsize=3)
    a.axhline(0, color="black", lw=0.6)
    a.set_ylabel("loss change, nats per token"); a.set_title("(a) Pair force off in this layer only")
    whole = RP["arms"]["phioff"]["all"][0]
    deep = DP["arms"]["phioff@1+"]["all"][0]
    a.text(0.98, 0.95, f"all eight layers off: +{whole:.3f}\nlayers 1-7 off together: +{deep:.3f}\nsum of the eight single layers: +{sum(m for m, _ in cost):.3f}",
           transform=a.transAxes, ha="right", va="top", fontsize=9)
    cos = [PA["fidelity"][f"ema|8|{L}"]["cos_p10_p50_p90"][1] for L in range(NL)]
    b.plot(X, cos, color=TEAL, marker="D", lw=2, ms=7)
    b.axhline(0.9, color="black", lw=0.8, ls=":"); b.text(7, 0.905, "A2 threshold", ha="right", va="bottom", fontsize=9)
    b.set_ylim(0.5, 1.0); b.set_ylabel("median cosine"); b.set_title("(b) How field-like the force is (Probe A)")
    c.bar(X, SCALE, 0.6, color=GREY); c.set_yscale("log")
    c.set_ylabel("softplus of the raw scale"); c.set_title("(c) Trained pair-force scale")
    sel = [RP["selection"][f"far@{L}"] for L in range(NL)]
    d.bar(X, [s["selected"][0] / s["chance"] for s in sel], 0.6, color=GREY)
    d.axhline(1, color="black", lw=0.8, ls="--"); d.text(7.3, 1.02, "chance", ha="right", va="bottom", fontsize=9)
    d.set_ylim(0, 1.2); d.set_ylabel("selected / chance"); d.set_title("(d) Router picks the earlier occurrence (far recall)")
    for ax in (c, d):
        ax.set_xlabel("layer"); ax.set_xticks(X)
    fig.tight_layout(); fig.savefig(OUT / "pfl_layers.png"); plt.close(fig)


if __name__ == "__main__":
    layers()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
