"""Three-way side-by-side comparison of shared-V_psi fits:
SPLM vs matched-parameter GPT-2-style baseline vs pretrained GPT-2 small.

This is the step-3 headline figure.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RES = Path(__file__).parent / "results"


def load(tag: str):
    z = np.load(RES / f"sharedV_{tag}_results.npz")
    return {k: z[k] for k in z.files}


def main():
    splm    = load("shakespeare_ckpt_latest")
    matched = load("matched_baseline")
    gpt2    = load("gpt2_baseline")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))

    panels = [
        (splm,    "SPLM (conservative-by-construction)\n7.1 M params, Shakespeare ppl 287",
         "tab:blue"),
        (matched, "Matched GPT-2-style (same params, same data)\n8.0 M params, Shakespeare ppl 142",
         "tab:purple"),
        (gpt2,    "Pretrained GPT-2 small (reference)\n124 M params, WebText-trained",
         "tab:red"),
    ]

    for ax, (data, name, colour) in zip(axes, panels):
        layers = data["layers"]
        y_null = data["r2_null_test"]
        y_vo   = data["r2_vo_test"]
        y_shv  = data["r2_shv_test"]
        ax.plot(layers, y_null, marker="x", linewidth=0.8, color="tab:gray",
                label="A. static null")
        ax.plot(layers, y_vo,   marker="s", linewidth=1.2, color="tab:orange",
                label="B. velocity-only")
        ax.plot(layers, y_shv,  marker="o", linewidth=1.8, color=colour,
                label="C. vel + shared $V_\\psi$")
        ax.fill_between(layers, y_vo, y_shv, alpha=0.12, color=colour)
        ax.axhline(0,   color="gray", linewidth=0.5)
        ax.axhline(0.5, color="gray", linewidth=0.3, linestyle=":")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("layer $\\ell$")
        ax.set_ylabel("per-layer TEST $R^2$")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower center", fontsize=8)

        med = float(np.median(y_shv))
        mn  = float(np.min(y_shv))
        mx  = float(np.max(y_shv))
        n_pass = int(np.sum(y_shv >= 0.5))
        ax.text(0.02, 0.98,
                f"median $R^2$  = {med:+.2f}\n"
                f"min $R^2$     = {mn:+.2f}\n"
                f"max $R^2$     = {mx:+.2f}\n"
                f"$R^2\\geq 0.5$: {n_pass}/{len(y_shv)}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))

    fig.suptitle("Step 3 -- shared-$V_\\psi$ fit, three-way architectural comparison  "
                 "(does a single scalar $V_\\psi(h)$ explain the whole network?)",
                 fontsize=11)
    fig.tight_layout()
    out = RES / "sharedV_three_way.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved -> {out}")

    # ---------- summary stats table ----------
    print("\nSummary (per-layer TEST R^2):")
    for (data, name, _) in panels:
        y = data["r2_shv_test"]
        print(f"  {name.split(chr(10))[0]:<55}  "
              f"median {np.median(y):+.3f}  "
              f"min {np.min(y):+.3f}  "
              f"max {np.max(y):+.3f}  "
              f"n_pass_0.5 {int(np.sum(y >= 0.5))}/{len(y)}")


if __name__ == "__main__":
    main()
