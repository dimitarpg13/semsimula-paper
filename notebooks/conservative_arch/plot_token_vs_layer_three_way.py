"""
Three-way + two-direction comparison plot.

Compares the shared-V_psi per-layer TEST R^2 for SPLM, Matched-GPT, and
Pretrained GPT-2, side-by-side along (a) the LAYER direction (step-2
default) and (b) the TOKEN direction (new).

Saves results/sharedV_layer_vs_token.png.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"

# Layer-direction files (step 2).
LAYER_FILES = {
    "SPLM":           RESULTS_DIR / "sharedV_shakespeare_ckpt_latest_results.npz",
    "Matched GPT-style": RESULTS_DIR / "sharedV_matched_baseline_results.npz",
    "Pretrained GPT-2":  RESULTS_DIR / "sharedV_gpt2_baseline_results.npz",
}
TOKEN_FILES = {
    "SPLM":              RESULTS_DIR / "tokdir_splm_shakespeare_results.npz",
    "Matched GPT-style": RESULTS_DIR / "tokdir_matched_baseline_results.npz",
    "Pretrained GPT-2":  RESULTS_DIR / "tokdir_gpt2_baseline_results.npz",
}
COLORS = {
    "SPLM":              "tab:blue",
    "Matched GPT-style": "tab:green",
    "Pretrained GPT-2":  "tab:red",
}


def load(npz_path: Path, token: bool):
    data = np.load(str(npz_path))
    if token:
        layers = data["layers_shv"]
        r2     = data["r2_shv_test"]
    else:
        layers = data["layers"]
        r2     = data["r2_shv_test"]
    return layers, r2


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    # Panel 1: LAYER direction.
    ax = axes[0]
    for name, p in LAYER_FILES.items():
        if not p.exists():
            print(f"skip missing {p}")
            continue
        layers, r2 = load(p, token=False)
        ax.plot(layers, r2, marker="o", color=COLORS[name], label=name)
        med = np.median(r2)
        print(f"[layer] {name:22s} median R^2 = {med:+.3f}  range {r2.min():+.3f}..{r2.max():+.3f}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(r"layer $\ell$")
    ax.set_ylabel(r"shared-$V_\psi$ TEST $R^2$")
    ax.set_title(r"(a) LAYER direction  (fix token, vary depth)")
    ax.set_ylim(-0.3, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 2: TOKEN direction.
    ax = axes[1]
    for name, p in TOKEN_FILES.items():
        if not p.exists():
            print(f"skip missing {p}")
            continue
        layers, r2 = load(p, token=True)
        ax.plot(layers, r2, marker="o", color=COLORS[name], label=name)
        med = np.median(r2)
        print(f"[token] {name:22s} median R^2 = {med:+.3f}  range {r2.min():+.3f}..{r2.max():+.3f}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(r"layer $\ell$  (separate fit at each $\ell$)")
    ax.set_title(r"(b) TOKEN direction  (fix layer, vary token)")
    ax.set_ylim(-0.3, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("Shared scalar-potential fit: two coordinate systems",
                 y=1.02, fontsize=13)
    fig.tight_layout()

    out = RESULTS_DIR / "sharedV_layer_vs_token.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
