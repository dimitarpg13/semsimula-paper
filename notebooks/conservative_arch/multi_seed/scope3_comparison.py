"""Scope-3 comparison aggregator: v2 buggy vs v4 leak-free PPL table.

Reads a multi-seed run directory (the same layout
``results/<tag>/<model_label>/seed_<s>/`` that
``multi_seed_aggregator.py`` consumes) and emits a focused one-page
markdown report tailored to the **Scope-3 retrain** of `paper_v4`.

For each Scope-3 model cell, the report shows:

  * v2 buggy reference PPL (single-seed, frozen literature value).
  * v4 leak-free PPL: mean / std / min / max / divergence count over the
    seeds present in the run directory.
  * Inflation factor = v4 mean PPL / v2 reference PPL.
  * Qualitative direction (does the v4 ordering still match the v2
    ordering of (B) > (A) > SARF > fixed-xi?).

This script does **not** replace ``multi_seed_aggregator.py``: it is the
Scope-3-specific narrative layer on top of it.  Run the standard
aggregator first to get the per-seed report and overlay plots, then run
this script to get the Scope-3 headline table.

Usage::

    python3 notebooks/conservative_arch/multi_seed/scope3_comparison.py \\
        --tag scope3_shakespeare
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parent.parent.parent


# -----------------------------------------------------------------------------
# v2 reference numbers (frozen, single-seed, buggy integrator).
#
# Sources:
#   * splm_baseline / splm_sarf / splm_sarfmass_embed_head /
#     splm_sarfmass_logfreq:
#       notebooks/conservative_arch/sarf_mass_variant/comparison_report.md
#       (compare.py output, single-seed run, May 2025).
#   * matched_baseline:
#       paper_v4/sections/15_conservative_architectures.tex,
#       Table tab:e1-headline (5-seed Welch reading from v2).
# -----------------------------------------------------------------------------
V2_REFERENCE: Dict[str, Dict[str, object]] = {
    "splm_baseline": {
        "label": "fixed-xi SPLM (sec 1 baseline)",
        "v2_ppl": 287.43,
        "v2_n_seeds": 1,
        "v2_std": None,
        "leak_immune_by_construction": True,
        "rank_v2": 5,
    },
    "splm_sarf": {
        "label": "SARF-faithful SPLM (no per-token mass)",
        "v2_ppl": 192.21,
        "v2_n_seeds": 1,
        "v2_std": None,
        "leak_immune_by_construction": False,
        "rank_v2": 3,
    },
    "splm_sarfmass_embed_head": {
        "label": "SARF + embed-head mass (variant A)",
        "v2_ppl": 222.91,
        "v2_n_seeds": 1,
        "v2_std": None,
        "leak_immune_by_construction": False,
        "rank_v2": 4,
    },
    "splm_sarfmass_logfreq": {
        "label": "SARF + logfreq mass (variant B)",
        "v2_ppl": 160.55,
        "v2_n_seeds": 1,
        "v2_std": None,
        "leak_immune_by_construction": False,
        "rank_v2": 2,
    },
    "matched_baseline": {
        "label": "matched GPT-2-micro baseline",
        "v2_ppl": 149.80,
        "v2_n_seeds": 5,
        "v2_std": 7.21,
        "leak_immune_by_construction": True,
        "rank_v2": 1,
    },
}


# -----------------------------------------------------------------------------
# Reuse the parser from multi_seed_aggregator without importing it (keeps
# this module self-contained for the Colab notebook).
# -----------------------------------------------------------------------------


@dataclass
class FinalEval:
    step: int
    val_loss: Optional[float]
    val_ppl: Optional[float]


def _parse_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
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


def _last_eval(rows: List[Dict]) -> Optional[FinalEval]:
    eval_rows = [r for r in rows if "val_loss" in r]
    if not eval_rows:
        return None
    r = eval_rows[-1]
    val_loss = r.get("val_loss")
    val_ppl = r.get("val_ppl")
    if val_ppl is None and val_loss is not None:
        try:
            val_ppl = math.exp(val_loss)
        except OverflowError:
            val_ppl = float("inf")
    return FinalEval(
        step=int(r.get("step", -1)),
        val_loss=val_loss,
        val_ppl=val_ppl,
    )


def _discover_seed_logs(model_dir: Path) -> List[Tuple[int, Path]]:
    """Return (seed, training_log_path) tuples sorted by seed."""
    out: List[Tuple[int, Path]] = []
    if not model_dir.exists():
        return out
    for seed_dir in sorted(model_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        log_candidates = list(seed_dir.glob("*_training_log.jsonl"))
        if not log_candidates:
            continue
        out.append((seed, log_candidates[0]))
    out.sort(key=lambda x: x[0])
    return out


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    diverged = len(values) - len(finite)
    if not finite:
        return {
            "n": len(values),
            "n_finite": 0,
            "n_diverged": diverged,
            "mean": None, "std": None,
            "min": None, "max": None,
        }
    return {
        "n": len(values),
        "n_finite": len(finite),
        "n_diverged": diverged,
        "mean": statistics.fmean(finite),
        "std": statistics.stdev(finite) if len(finite) >= 2 else 0.0,
        "min": min(finite),
        "max": max(finite),
    }


def _fmt(x: Optional[float], fmt: str) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return format(x, fmt)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True,
                    help="Run-tag subdirectory under results/.")
    ap.add_argument("--results-root", default=str(RESULTS_ROOT),
                    help=f"Override the results root (default: {RESULTS_ROOT}).")
    ap.add_argument("--out", default=None,
                    help="Override the report output path (default: "
                         "<results-root>/<tag>/scope3_comparison.md).")
    args = ap.parse_args()

    run_root = Path(args.results_root) / args.tag
    if not run_root.exists():
        print(f"[scope3] ERROR: run root not found: {run_root}")
        return 2

    out_path = Path(args.out) if args.out else run_root / "scope3_comparison.md"

    # Discover the v4 leak-free PPLs cell-by-cell.
    cells: List[Tuple[str, Dict[str, object], Dict[str, Optional[float]],
                       List[Tuple[int, Optional[float]]]]] = []
    for label, ref in V2_REFERENCE.items():
        model_dir = run_root / label
        seed_logs = _discover_seed_logs(model_dir)
        per_seed_ppl: List[Tuple[int, Optional[float]]] = []
        for seed, log_path in seed_logs:
            fe = _last_eval(_parse_jsonl(log_path))
            per_seed_ppl.append((seed, fe.val_ppl if fe is not None else None))
        ppl_only = [v for _, v in per_seed_ppl]
        cells.append((label, ref, _stats(ppl_only), per_seed_ppl))

    # Determine v4 ordering by mean PPL (ascending = better).
    v4_ranked = sorted(
        [(label, stats) for label, _, stats, _ in cells if stats["mean"] is not None],
        key=lambda x: x[1]["mean"],
    )
    v4_rank: Dict[str, int] = {label: i + 1 for i, (label, _) in enumerate(v4_ranked)}

    lines: List[str] = []
    lines.append(f"# Scope-3 retrain comparison (`{args.tag}`)")
    lines.append("")
    lines.append("Re-runs the v2 SPLM-family experiments under the v4 leak-free")
    lines.append("integrator (`cfg.causal_force = True`, the post-fix default).")
    lines.append("All cells share the v2 hyperparameters (Tiny Shakespeare,")
    lines.append("`d=128, L=8, max_len=512, block_size=128, batch_size=16,`")
    lines.append("`lr=5e-4, 4000 steps`); the only systematic change vs v2 is the")
    lines.append("integrator's causal-honesty flag.")
    lines.append("")
    try:
        run_root_display = str(run_root.relative_to(REPO_ROOT)) + "/"
    except ValueError:
        run_root_display = str(run_root)
    lines.append("Run root: `" + run_root_display + "`")
    lines.append("")
    lines.append("## Headline table -- v2 buggy vs v4 leak-free")
    lines.append("")
    lines.append("Inflation factor = v4 mean PPL / v2 reference PPL. A factor")
    lines.append("near `1.00` for `splm_baseline` and `matched_baseline` is the")
    lines.append("leak-immunity sanity check (those two cells are leak-free by")
    lines.append("construction); for the SARF cells the inflation quantifies how")
    lines.append("much the v2 numbers were under-reported by the bug.")
    lines.append("")
    lines.append("| cell | v2 ref PPL | v4 mean ± std PPL | n_finite / n_seeds | "
                 "diverged | inflation x | leak-immune? |")
    lines.append("|---|---:|---:|---:|---:|---:|:--:|")
    for label, ref, stats, _ in cells:
        v2_ppl = ref["v2_ppl"]
        if stats["mean"] is None:
            v4_str = "*(no runs)*"
            inflation_str = "n/a"
        else:
            v4_str = f"{_fmt(stats['mean'], '.2f')} ± {_fmt(stats['std'], '.2f')}"
            inflation_str = f"{(stats['mean'] / v2_ppl):.2f}x"
        leak_immune = "yes" if ref["leak_immune_by_construction"] else "no"
        seeds_present = (f"{stats['n_finite']} / {stats['n']}"
                         if stats["n"] > 0 else "0 / 0")
        v2_str = f"{v2_ppl:.2f}"
        if ref["v2_std"] is not None:
            v2_str += f" ± {ref['v2_std']:.2f}"
        if ref["v2_n_seeds"] > 1:
            v2_str += f" ({ref['v2_n_seeds']}s)"
        lines.append(
            f"| `{label}` | {v2_str} | {v4_str} | {seeds_present} | "
            f"{stats['n_diverged']} | {inflation_str} | {leak_immune} |"
        )
    lines.append("")

    lines.append("## Per-seed PPL ladder")
    lines.append("")
    lines.append("Empty cells = seed not run yet.")
    lines.append("")
    all_seeds = sorted({
        s for _, _, _, per in cells for s, _ in per
    })
    if all_seeds:
        header = "| cell | " + " | ".join(f"seed {s}" for s in all_seeds) + " | mean | std |"
        sep = "|---|" + "---:|" * (len(all_seeds) + 2)
        lines.append(header)
        lines.append(sep)
        for label, _, stats, per in cells:
            per_map = dict(per)
            row = [f"`{label}`"]
            for s in all_seeds:
                v = per_map.get(s)
                if v is None:
                    row.append("")
                elif not math.isfinite(v):
                    row.append("**NaN**")
                else:
                    row.append(f"{v:.2f}")
            row.append(_fmt(stats["mean"], ".2f"))
            row.append(_fmt(stats["std"], ".2f"))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    else:
        lines.append("*(no seeds discovered)*")
        lines.append("")

    lines.append("## Qualitative-direction check")
    lines.append("")
    lines.append("The v2 `compare.py` reading (single-seed) ordered the four")
    lines.append("SPLM cells from best to worst as:")
    lines.append("")
    lines.append("> matched (1) -> sarfmass_logfreq (2) -> sarf (3) -> "
                 "sarfmass_embed_head (4) -> splm_baseline (5)")
    lines.append("")
    lines.append("This corresponds to the framework-internal prediction that")
    lines.append("(a) SARF-faithful $\\xi$ recomputation beats fixed-$\\xi$,")
    lines.append("(b) the surprisal-prior mass beats the unconstrained-MLP mass,")
    lines.append("and (c) per-token mass with a useful prior beats global mass.")
    lines.append("If the v4 leak-free retrain preserves this order, the")
    lines.append("framework's qualitative claims survive causal honesty.")
    lines.append("")
    lines.append("| cell | v2 rank | v4 rank | Δ rank | preserved? |")
    lines.append("|---|---:|---:|---:|:--:|")
    for label, ref, stats, _ in cells:
        rank_v4 = v4_rank.get(label)
        rank_v4_str = str(rank_v4) if rank_v4 is not None else "n/a"
        delta = (rank_v4 - ref["rank_v2"]) if rank_v4 is not None else None
        delta_str = f"{delta:+d}" if delta is not None else "n/a"
        if rank_v4 is None:
            preserved = "n/a"
        elif rank_v4 == ref["rank_v2"]:
            preserved = "yes"
        else:
            preserved = f"shifted ({delta:+d})"
        lines.append(
            f"| `{label}` | {ref['rank_v2']} | {rank_v4_str} | "
            f"{delta_str} | {preserved} |"
        )
    lines.append("")

    lines.append("## Decision rules for `paper_v4`")
    lines.append("")
    lines.append("1. **Leak-immunity controls**. If `splm_baseline` and")
    lines.append("   `matched_baseline` reproduce v2 PPL within seed noise")
    lines.append("   (inflation factor in `[0.95, 1.05]`), the v4 retrain is")
    lines.append("   self-consistent and the SARF-cell inflations are")
    lines.append("   attributable to the leak fix, not to confound.")
    lines.append("2. **SARF asymmetric inflation**. Report each SARF cell's")
    lines.append("   inflation factor in `paper_v4` §15 (replacing the")
    lines.append("   `~2x asymmetric inflation` placeholder). Inflation is")
    lines.append("   expected to be `>1` because the leak gave the v2 SARF")
    lines.append("   models access to future tokens through $\\xi$.")
    lines.append("3. **Qualitative direction**. If the v4 ordering matches the")
    lines.append("   v2 ordering across the 4 SPLM cells, all `v2-historical`")
    lines.append("   caveat blocks in `paper_v4` §15.17 / §15.19 can be")
    lines.append("   retired. Any rank inversion warrants a footnote in the")
    lines.append("   relevant subsection.")
    lines.append("4. **Matched-baseline pairing**. The `matched_baseline` 5-seed")
    lines.append("   PPL is the new pairing target for the SPLM-vs-baseline")
    lines.append("   absolute-quality claim previously retired in §15.")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    try:
        out_display = str(out_path.relative_to(REPO_ROOT))
    except ValueError:
        out_display = str(out_path)
    print(f"[scope3] wrote {out_display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
