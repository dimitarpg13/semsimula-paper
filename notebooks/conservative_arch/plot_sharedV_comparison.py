"""Side-by-side comparison of shared-V fits on SPLM vs GPT-2."""
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
    splm = load("shakespeare_ckpt_latest")
    gpt2 = load("gpt2_baseline")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    for ax, data, name, colour in zip(
        axes,
        [splm, gpt2],
        ["Scalar-Potential LM  (conservative-by-construction)",
         "GPT-2 small  (attention baseline)"],
        ["tab:blue", "tab:red"],
    ):
        layers = data["layers"]
        y_null = data["r2_null_test"]
        y_vo   = data["r2_vo_test"]
        y_shv  = data["r2_shv_test"]
        ax.plot(layers, y_null, marker="x", linewidth=0.8, color="tab:gray",
                label="A. static null")
        ax.plot(layers, y_vo,   marker="s", linewidth=1.2, color="tab:orange",
                label="B. velocity-only")
        ax.plot(layers, y_shv,  marker="o", linewidth=1.8, color=colour,
                label="C. velocity + shared $V_\\psi$")
        ax.fill_between(layers, y_vo, y_shv, alpha=0.12, color=colour,
                        label="gain from shared $V_\\psi$")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axhline(0.5, color="gray", linewidth=0.3, linestyle=":")
        ax.set_title(name)
        ax.set_xlabel("layer $\\ell$")
        ax.set_ylabel("per-layer TEST $R^2$")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

        min_sh = float(np.min(y_shv))
        med_sh = float(np.median(y_shv))
        ax.text(0.02, 0.96,
                f"median $R^2$ (shared V)  = {med_sh:+.2f}\n"
                f"min $R^2$ (shared V)     = {min_sh:+.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))

    fig.suptitle("Shared-potential fit:  does a SINGLE scalar $V_\\psi(h)$ "
                 "explain the per-layer dynamics of the whole network?",
                 fontsize=11)
    fig.tight_layout()
    out = RES / "sharedV_splm_vs_gpt2.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
