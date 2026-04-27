"""Aggregate per-seed training logs into a multi-seed report and overlay plot.

For a given ``--tag`` directory under
``notebooks/conservative_arch/multi_seed/results/<tag>/``, this script:

1. Discovers each ``<model_label>/seed_<s>/`` subdirectory and the
   ``*_training_log.jsonl`` it contains.
2. Extracts every eval point (step, train_loss_eval, val_loss, val_ppl)
   per seed and the final-eval row.
3. Computes mean / std / min / max of final val ppl and val loss across
   seeds, separately per model.
4. If multiple models are present, runs Welch's t-test on the
   final-val-ppl distributions (no assumption of equal variance) and
   reports the 95% CI of the difference of means.
5. Writes a multi-seed loss-curve overlay plot per model
   (light per-seed lines + bold mean line).
6. Writes a markdown report ``<tag>/<tag>_report.md``.

Usage::

    python3 notebooks/conservative_arch/multi_seed/multi_seed_aggregator.py \\
        --tag E1_shakespeare
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
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


@dataclass
class EvalPoint:
    step: int
    train_loss_eval: Optional[float]
    val_loss: Optional[float]
    val_ppl: Optional[float]


@dataclass
class SeedRun:
    model_label: str
    seed: int
    eval_points: List[EvalPoint]
    final: Optional[EvalPoint]


def _parse_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _extract_eval_points(rows: List[Dict]) -> List[EvalPoint]:
    pts: List[EvalPoint] = []
    for r in rows:
        if "val_loss" in r:
            pts.append(EvalPoint(
                step=int(r.get("step", 0)),
                train_loss_eval=r.get("train_loss_eval"),
                val_loss=r.get("val_loss"),
                val_ppl=r.get("val_ppl"),
            ))
    return pts


def _discover(run_root: Path) -> Dict[str, List[SeedRun]]:
    by_model: Dict[str, List[SeedRun]] = {}
    if not run_root.exists():
        raise FileNotFoundError(f"run root does not exist: {run_root}")
    for model_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        seeds: List[SeedRun] = []
        for seed_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()
                               and p.name.startswith("seed_")):
            seed = int(seed_dir.name.split("_", 1)[1])
            jsonls = sorted(seed_dir.glob("*_training_log.jsonl"))
            if not jsonls:
                continue
            rows = _parse_jsonl(jsonls[0])
            pts = _extract_eval_points(rows)
            final = pts[-1] if pts else None
            seeds.append(SeedRun(
                model_label=model_dir.name, seed=seed,
                eval_points=pts, final=final,
            ))
        if seeds:
            by_model[model_dir.name] = seeds
    return by_model


def _summary_stats(values: List[float]) -> Dict[str, float]:
    """Summary stats over finite (non-NaN, non-inf) values only.

    Returns ``n_total`` (input length), ``n_ok`` (finite count) and
    ``n_diverged`` (count of NaN / inf entries) so divergent seeds do
    not poison the headline mean/std.
    """
    arr_all = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(arr_all)
    arr = arr_all[finite_mask]
    n_total = int(arr_all.size)
    n_ok = int(arr.size)
    n_diverged = n_total - n_ok
    if n_ok == 0:
        return {
            "n": n_total, "n_ok": 0, "n_diverged": n_diverged,
            "mean": float("nan"), "std": float("nan"),
            "min": float("nan"), "max": float("nan"),
            "median": float("nan"),
        }
    return {
        "n": n_total,
        "n_ok": n_ok,
        "n_diverged": n_diverged,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if n_ok > 1 else float("nan"),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


def _welch_t_ci(a: List[float], b: List[float], conf: float = 0.95
                ) -> Tuple[float, float, float, float]:
    """Welch's t-test for difference of means.

    NaN / inf entries are dropped before the test; if either group has
    fewer than 2 finite values we return all-NaN.

    Returns ``(mean_diff, t, dof, half_width)`` where the 95% CI for the
    difference of means is ``mean_diff +/- half_width``.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    arr_a = arr_a[np.isfinite(arr_a)]
    arr_b = arr_b[np.isfinite(arr_b)]
    na, nb = arr_a.size, arr_b.size
    if na < 2 or nb < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    var_a = arr_a.var(ddof=1)
    var_b = arr_b.var(ddof=1)
    se = math.sqrt(var_a / na + var_b / nb)
    if se == 0.0:
        return (float(arr_a.mean() - arr_b.mean()),
                float("inf"), float("nan"), 0.0)
    mean_diff = float(arr_a.mean() - arr_b.mean())
    t_stat = mean_diff / se
    dof = (var_a / na + var_b / nb) ** 2 / (
        (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
    )
    try:
        from scipy.stats import t as student_t
        half_width = float(student_t.ppf(0.5 + conf / 2.0, dof) * se)
    except Exception:
        half_width = 1.96 * se
    return mean_diff, float(t_stat), float(dof), half_width


def _plot_overlay(model_label: str, runs: List[SeedRun], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    finite_steps: List[List[int]] = []
    finite_vals: List[List[float]] = []
    n_diverged = 0
    for r in runs:
        pts = [(p.step, p.val_loss) for p in r.eval_points
               if p.val_loss is not None and math.isfinite(p.val_loss)]
        is_diverged = (
            r.final is None
            or r.final.val_loss is None
            or not math.isfinite(r.final.val_loss)
        )
        if is_diverged:
            n_diverged += 1
        if not pts:
            continue
        steps = [s for s, _ in pts]
        vals = [v for _, v in pts]
        marker = " (diverged)" if is_diverged else ""
        ls = ":" if is_diverged else "-"
        ax.plot(steps, vals, alpha=0.45, lw=1.2, linestyle=ls,
                label=f"seed {r.seed}{marker}")
        finite_steps.append(steps)
        finite_vals.append(vals)
    n_total = len(runs)
    n_ok = n_total - n_diverged
    if finite_vals:
        max_n = max(len(v) for v in finite_vals)
        finite_only = [
            (s, v) for s, v, run in zip(finite_steps, finite_vals, runs)
            if run.final is not None and run.final.val_loss is not None
            and math.isfinite(run.final.val_loss) and len(v) == max_n
        ]
        if len(finite_only) >= 2:
            steps_arr = np.asarray(finite_only[0][0])
            vals_arr = np.asarray([v for _, v in finite_only])
            vals_mean = vals_arr.mean(axis=0)
            ax.plot(steps_arr, vals_mean, color="black", lw=2.0,
                    label=f"mean (n_ok={len(finite_only)})")
    ax.set_xlabel("training step")
    ax.set_ylabel("val loss (NLL)")
    title = (f"{model_label}  multi-seed val-loss overlay "
             f"(n_ok={n_ok}/{n_total}, diverged={n_diverged})")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _emit_markdown(
    tag: str, run_root: Path,
    by_model: Dict[str, List[SeedRun]],
    plots: Dict[str, Path],
) -> Path:
    md_path = run_root / f"{tag}_report.md"
    lines: List[str] = []
    lines.append(f"# Multi-seed report: `{tag}`")
    lines.append("")
    lines.append(f"Aggregator: "
                 f"`notebooks/conservative_arch/multi_seed/multi_seed_aggregator.py`")
    lines.append(f"Run root: `{run_root.relative_to(REPO_ROOT)}/`")
    lines.append("")

    def _fmt(x: float, fmt: str) -> str:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return "n/a"
        return format(x, fmt)

    lines.append("## Per-model summary (final val loss / val ppl)")
    lines.append("")
    lines.append("Stats are computed over **finite** seeds only; the "
                 "`diverged` column reports seeds whose final eval was "
                 "NaN / inf (these are excluded from mean/std/min/max).")
    lines.append("")
    lines.append("| model | n seeds | diverged | val loss mean | "
                 "val loss std | val ppl mean | val ppl std | "
                 "val ppl min | val ppl max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    final_ppl: Dict[str, List[float]] = {}
    final_loss: Dict[str, List[float]] = {}
    for label, runs in by_model.items():
        ppl_vals = [r.final.val_ppl for r in runs
                    if r.final is not None and r.final.val_ppl is not None]
        loss_vals = [r.final.val_loss for r in runs
                     if r.final is not None and r.final.val_loss is not None]
        if not ppl_vals:
            continue
        final_ppl[label] = ppl_vals
        final_loss[label] = loss_vals
        s_loss = _summary_stats(loss_vals) if loss_vals else None
        s_ppl = _summary_stats(ppl_vals)
        lines.append(
            f"| `{label}` | {s_ppl['n']} | {s_ppl['n_diverged']} | "
            f"{_fmt(s_loss['mean'], '.4f')} | "
            f"{_fmt(s_loss['std'], '.4f')} | "
            f"{_fmt(s_ppl['mean'], '.2f')} | "
            f"{_fmt(s_ppl['std'], '.2f')} | "
            f"{_fmt(s_ppl['min'], '.2f')} | "
            f"{_fmt(s_ppl['max'], '.2f')} |"
        )
    lines.append("")

    lines.append("## Per-seed final eval points")
    lines.append("")
    lines.append("| model | seed | step | train loss eval | val loss | val ppl |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, runs in by_model.items():
        for r in sorted(runs, key=lambda x: x.seed):
            if r.final is None:
                continue
            f = r.final
            lines.append(
                f"| `{label}` | {r.seed} | {f.step} | "
                f"{_fmt(f.train_loss_eval, '.4f')} | "
                f"{_fmt(f.val_loss, '.4f')} | "
                f"{_fmt(f.val_ppl, '.2f')} |"
            )
    lines.append("")

    if len(final_ppl) >= 2:
        lines.append("## Pairwise gap (Welch's t-test on final val ppl)")
        lines.append("")
        lines.append("Welch's t-test is applied to the **finite** final-ppl "
                     "values only; pairs with fewer than 2 finite seeds in "
                     "either group are reported as `n/a`.")
        lines.append("")
        lines.append("| model A | model B | n_A | n_B | "
                     "A mean - B mean | 95% CI half-width | t | dof |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        labels = list(final_ppl.keys())
        for i, la in enumerate(labels):
            for lb in labels[i + 1:]:
                a_finite = [v for v in final_ppl[la] if math.isfinite(v)]
                b_finite = [v for v in final_ppl[lb] if math.isfinite(v)]
                mean_diff, t_stat, dof, hw = _welch_t_ci(
                    final_ppl[la], final_ppl[lb], conf=0.95,
                )
                lines.append(
                    f"| `{la}` | `{lb}` | {len(a_finite)} | {len(b_finite)} | "
                    f"{_fmt(mean_diff, '+.2f')} | {_fmt(hw, '.2f')} | "
                    f"{_fmt(t_stat, '+.2f')} | {_fmt(dof, '.1f')} |"
                )
        lines.append("")

    lines.append("## Loss-curve overlays")
    lines.append("")
    for label, plot in plots.items():
        rel = plot.relative_to(run_root.parent.parent.parent.parent
                              if plot.is_absolute() else run_root)
        lines.append(f"### `{label}`")
        lines.append("")
        try:
            rel = plot.relative_to(REPO_ROOT)
        except ValueError:
            rel = plot
        lines.append(f"![{label}]({plot.name})")
        lines.append("")

    lines.append("## Interpretation (manual)")
    lines.append("")
    lines.append("> **TODO (human reviewer):** Inspect the table and "
                 "overlay plots and answer:")
    lines.append(">")
    lines.append("> 1. Does the previously-reported single-seed perplexity "
                 "fall within one std of the multi-seed mean? If not, "
                 "was that single run an outlier and the headline number "
                 "needs to be revised.")
    lines.append("> 2. Is the SPLM-vs-baseline gap statistically meaningful "
                 "at this n? Compare the 95% CI half-width of the "
                 "difference of means against the absolute gap.")
    lines.append("> 3. Are any seeds catastrophic (val ppl >> mean+3*std)? "
                 "If so, investigate before reporting; do not silently "
                 "discard.")
    lines.append("")

    md_path.write_text("\n".join(lines))
    return md_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", type=str, required=True,
                    help="Run-tag subdirectory under results/.")
    args = ap.parse_args()

    run_root = RESULTS_ROOT / args.tag
    if not run_root.exists():
        print(f"[aggregator] no run root: {run_root}")
        return 1

    print(f"[aggregator] reading {run_root.relative_to(REPO_ROOT)}")
    by_model = _discover(run_root)
    if not by_model:
        print("[aggregator] no per-seed training logs discovered.")
        return 1

    plots: Dict[str, Path] = {}
    for label, runs in by_model.items():
        out = run_root / f"{args.tag}_loss_curves_{label}.png"
        _plot_overlay(label, runs, out)
        plots[label] = out
        print(f"  [plot] {out.relative_to(REPO_ROOT)}  "
              f"({len(runs)} seeds)")

    md = _emit_markdown(args.tag, run_root, by_model, plots)
    print(f"[aggregator] report -> {md.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
