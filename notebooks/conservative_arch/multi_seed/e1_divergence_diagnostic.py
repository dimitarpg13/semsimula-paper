"""Divergence-rate diagnostic for the E1 multi-seed sweep.

Compares the three E1 models (`splm_sarfmass_logfreq`, `splm_em_ln`,
`matched_baseline`) along two axes:

* training NLL trajectory (with NaN onset markers)
* gradient-norm trajectory (log-scale, to expose blow-up vs absorption)

The figure tells the architectural story behind the divergence-rate
column of the multi-seed report:

* `sarfmass` (per-token mass + Euler, **no** LayerNorm-after-step)
  diverges on 2 of 3 seeds despite modest grad-norm magnitudes
  (max ~13).  The instability is **state-space drift**: the Euler step
  pushes the integrator state outside the basin from which the
  per-token-mass schedule is well-conditioned.

* `em_ln` (per-token mass + Euler, **with** LayerNorm-after-step) on
  the *same* corpus and identical optimiser config still emits
  catastrophic gradient transients (max ~6.9e17 on seed 3) but
  **never diverges**.  The LayerNorm-after-step renormalises the state
  to the unit sphere after every Euler update, so a single bad batch
  cannot launch the trajectory into a runaway regime; the optimiser's
  grad-clip then absorbs the residual.

* `matched_baseline` (GPT-2 micro) sits on a tightly-bounded grad-norm
  trajectory (max ~2.5) -- the textbook well-conditioned LM.

Usage::

    python3 notebooks/conservative_arch/multi_seed/e1_divergence_diagnostic.py \
        --tag E1_shakespeare
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parent.parent.parent


MODEL_CONFIGS: List[Tuple[str, str, str]] = [
    ("matched_baseline", "matched_shakespeare_training_log.jsonl",
     "matched_baseline (GPT-2 micro)"),
    ("splm_em_ln", "em_ln_shakespeare_training_log.jsonl",
     "splm_em_ln (Euler + per-token mass + LN-after-step)"),
    ("splm_sarfmass_logfreq",
     "splm_sarfmass_logfreq_shakespeare_training_log.jsonl",
     "splm_sarfmass_logfreq (Euler + per-token mass, no LN)"),
]


@dataclass
class Trace:
    seed: int
    steps: List[int]
    train_loss: List[float]
    grad_norm: List[float]
    diverged_at: Optional[int]


def _load(jsonl: Path) -> Trace:
    steps: List[int] = []
    tloss: List[float] = []
    gnorm: List[float] = []
    diverged_at: Optional[int] = None
    seed = -1
    if not jsonl.exists():
        return Trace(seed=seed, steps=steps, train_loss=tloss,
                     grad_norm=gnorm, diverged_at=None)
    seed = int(jsonl.parent.name.rsplit("_", 1)[-1])
    with jsonl.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            st = r.get("step")
            tl = r.get("train_loss")
            gn = r.get("grad_norm")
            if st is None:
                continue
            if (isinstance(tl, float) and not math.isfinite(tl)
                    and diverged_at is None):
                diverged_at = int(st)
            steps.append(int(st))
            tloss.append(tl if tl is not None else float("nan"))
            gnorm.append(gn if gn is not None else float("nan"))
    return Trace(seed=seed, steps=steps, train_loss=tloss,
                 grad_norm=gnorm, diverged_at=diverged_at)


def _model_traces(run_root: Path, model_dir: str,
                  log_name: str) -> List[Trace]:
    out: List[Trace] = []
    mdir = run_root / model_dir
    if not mdir.exists():
        return out
    for sdir in sorted(p for p in mdir.iterdir()
                       if p.is_dir() and p.name.startswith("seed_")):
        out.append(_load(sdir / log_name))
    return out


def _plot_panel(ax, traces: List[Trace], series: str, title: str,
                log_y: bool = False) -> None:
    n_div = sum(1 for t in traces if t.diverged_at is not None)
    n_total = len(traces)
    for t in traces:
        ys: List[float] = (t.train_loss if series == "train_loss"
                           else t.grad_norm)
        finite = [(s, y) for s, y in zip(t.steps, ys)
                  if y is not None and math.isfinite(y)]
        if not finite:
            continue
        ss, vs = zip(*finite)
        diverged = t.diverged_at is not None
        marker = " (diverged)" if diverged else ""
        ls = ":" if diverged else "-"
        ax.plot(ss, vs, lw=1.2, alpha=0.55, linestyle=ls,
                label=f"seed {t.seed}{marker}")
        if diverged and t.diverged_at is not None:
            ax.axvline(t.diverged_at, color="red", lw=0.6,
                       alpha=0.4, linestyle="--")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(f"{title}\n(diverged {n_div} / {n_total})", fontsize=10)
    ax.set_xlabel("training step")
    if series == "train_loss":
        ax.set_ylabel("train NLL")
    else:
        ax.set_ylabel("grad norm")
    ax.legend(fontsize=7, ncol=1, loc="best")


def _emit_table(model_summaries: List[Dict]) -> str:
    lines: List[str] = []
    lines.append("## Divergence-rate diagnostic (per-model)")
    lines.append("")
    lines.append("| model | seeds run | diverged | divergence rate | "
                 "first-NaN steps | grad_norm max (per seed) |")
    lines.append("|---|---:|---:|---:|---|---|")
    for s in model_summaries:
        diverged_steps = ", ".join(
            str(d) if d is not None else "-" for d in s["diverged_at"]
        ) or "-"
        gn_maxes = ", ".join(
            f"{m:.2g}" if (m is not None and math.isfinite(m)) else "-"
            for m in s["grad_max"]
        )
        rate = (s["diverged_count"] / s["n_total"]
                if s["n_total"] else float("nan"))
        lines.append(
            f"| `{s['label']}` | {s['n_total']} | "
            f"{s['diverged_count']} | {rate:.2%} | "
            f"{diverged_steps} | {gn_maxes} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", type=str, required=True)
    args = ap.parse_args()

    run_root = RESULTS_ROOT / args.tag
    if not run_root.exists():
        print(f"[diagnostic] no run root: {run_root}")
        return 1

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), dpi=140,
                             sharex=True)
    summaries: List[Dict] = []

    for row, (mdir, log_name, pretty) in enumerate(MODEL_CONFIGS):
        traces = _model_traces(run_root, mdir, log_name)
        log_y_grad = (mdir == "splm_em_ln")
        _plot_panel(axes[row, 0], traces, "train_loss",
                    f"{pretty}\ntrain NLL", log_y=False)
        _plot_panel(axes[row, 1], traces, "grad_norm",
                    f"{pretty}\ngrad norm"
                    + ("  (log y; LN absorbs spikes)" if log_y_grad
                       else ""),
                    log_y=log_y_grad)
        diverged_at = [t.diverged_at for t in traces]
        diverged_count = sum(1 for d in diverged_at if d is not None)
        finite_grad_max = [
            max((g for g in t.grad_norm
                 if g is not None and math.isfinite(g)),
                default=float("nan"))
            for t in traces
        ]
        summaries.append({
            "label": mdir,
            "pretty": pretty,
            "n_total": len(traces),
            "diverged_count": diverged_count,
            "diverged_at": diverged_at,
            "grad_max": finite_grad_max,
        })

    fig.suptitle(f"E1 divergence diagnostic ({args.tag})",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_png = run_root / f"{args.tag}_divergence_diagnostic.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[diagnostic] figure -> {out_png.relative_to(REPO_ROOT)}")

    table_md = _emit_table(summaries)
    out_md = run_root / f"{args.tag}_divergence_diagnostic.md"
    out_md.write_text(table_md)
    print(f"[diagnostic] table  -> {out_md.relative_to(REPO_ROOT)}")
    print()
    print(table_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
