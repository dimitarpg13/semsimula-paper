"""
Build a single side-by-side comparison figure across the three runs:
  (Euler L=8 dyn, Verlet L=16 dt=0.5 dyn, Verlet L=16 dt=0.5 gradient).

For each prompt produce a row: scoreboard table of top decoded tokens
per attractor, sized by basin size.  Saves
  results/attractors_comparison.png
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES_DIR = Path(__file__).parent / "results"

RUNS = [
    ("Euler L=8 dt=1.0  (dynamics)",     "euler_L8_dt1_dyn"),
    ("Verlet L=16 dt=0.5  (dynamics)",   "verlet_L16_dt05_dyn"),
    ("Verlet L=16 dt=0.5  (V_theta GD)", "verlet_L16_dt05_grad"),
]


def fmt_attr(toks: List[List], top_k: int = 5) -> str:
    parts = []
    for tok, prob in toks[:top_k]:
        t = tok.replace("\\n", "\\n").replace("\\t", "\\t")
        parts.append(f"{t.strip() or '·'}·{float(prob):.2f}")
    return "  ".join(parts)


def main() -> None:
    runs_data = []
    for label, tag in RUNS:
        path = RES_DIR / f"attractors_{tag}_results.json"
        with path.open() as f:
            runs_data.append((label, tag, json.load(f)))

    prompts = [p["prompt_name"] for p in runs_data[0][2]["prompts"]]
    n_prompts = len(prompts)

    fig, axes = plt.subplots(
        n_prompts, len(RUNS),
        figsize=(6.0 * len(RUNS), 1.6 * n_prompts + 0.6),
        squeeze=False,
    )
    for col, (label, tag, data) in enumerate(runs_data):
        for row, prompt_data in enumerate(data["prompts"]):
            ax = axes[row][col]
            ax.set_xlim(0, 1)
            n_tot = prompt_data["n_total"]
            attractors = sorted(prompt_data["attractors"],
                                key=lambda a: -a["size"])
            n_show = min(len(attractors), 6)
            ax.set_ylim(0, n_show + 0.5)
            for i, a in enumerate(attractors[:n_show]):
                y = n_show - i
                width = a["size"] / n_tot
                ax.barh(y, width, height=0.65, color="steelblue", alpha=0.55)
                ax.text(width + 0.01, y, fmt_attr(a["top_tokens"], 5),
                        fontsize=7.5, va="center")
                ax.text(0.005, y, f"A{a['id']} ({a['size']:3d})",
                        fontsize=7, va="center", color="black",
                        fontweight="bold")
            ax.set_yticks([])
            ax.set_xticks([])
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_alpha(0.3)
            if col == 0:
                real_top = prompt_data["real_top_tokens"][:5]
                real_str = fmt_attr(real_top, 5)
                ax.set_ylabel(
                    f"{prompt_data['prompt_name']}\n"
                    f"\"{prompt_data['prompt_text']}\"\n"
                    f"real next: {real_str}",
                    rotation=0, ha="right", va="center",
                    fontsize=7.5, labelpad=4,
                )
            if row == 0:
                ax.set_title(label, fontsize=10, fontweight="bold")
            ax.text(0.99, 0.02,
                    f"K* = {prompt_data['K_star']}   "
                    f"({prompt_data['n_converged']}/{prompt_data['n_total']} "
                    f"converged)",
                    transform=ax.transAxes,
                    fontsize=7, ha="right", va="bottom", color="gray")

    fig.suptitle(
        "Semantic-attractor extraction from $V_\\theta$:  "
        "Euler vs Verlet, dynamical vs gradient descent",
        fontsize=12, y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path = RES_DIR / "attractors_comparison.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
