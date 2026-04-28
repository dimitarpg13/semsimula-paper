"""Robustness sweep — 4 function classes x 3 PCA dims x 2 architectures = 24 cells.

Implements the §6.5 / §6.6 audit of the pre-registered protocol
(`docs/first_order_ODE_rejection_pre-registered_protocol.md`).

Per cell we record (rho_12, p_12_two_sided, rho_23, p_23_two_sided, decision)
and a Bonferroni-corrected pass/fail flag at p < 1e-3 / 24 ≈ 4.2e-5.

Usage
-----
    python robustness_sweep.py \
        --gpt2_quads results/gpt2/quadruples.npz \
        --pythia_quads results/pythia/quadruples.npz \
        --output_path results/robustness_grid.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from markov_order_regression import (
    RNG_SEED,
    aggregate_residuals,
    cluster_bootstrap_diff,
    decide_outcome,
    run_loso,
)


CLASSES = ["kernel", "linear", "poly2", "mlp"]
PCA_DIMS = [20, 50, 100]
N_CELLS = len(CLASSES) * len(PCA_DIMS) * 2  # 2 architectures
BONFERRONI_THRESHOLD = 1e-3 / N_CELLS


def _load_quads(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return {
        "H_tm2": data["H_tm2"],
        "H_tm1": data["H_tm1"],
        "H_t": data["H_t"],
        "H_tp1": data["H_tp1"],
        "sentence_idx": data["sentence_idx"],
    }


def run_cell(
    arch: str,
    quads: dict,
    p: int,
    function_class: str,
    n_jobs: int,
    seed: int,
    n_bootstrap: int,
) -> dict:
    t0 = time.time()
    folds = run_loso(
        quads, p=p, k_values=[1, 2, 3],
        function_class=function_class, n_jobs=n_jobs, seed=seed, verbose=0,
    )
    runtime = time.time() - t0
    agg = aggregate_residuals(folds, [1, 2, 3])
    r = agg["residuals"]
    sids = agg["sentence_idx"]

    rho_12 = float(r[1].mean() / r[2].mean())
    rho_23 = float(r[2].mean() / r[3].mean())

    try:
        p_12_two = float(wilcoxon(r[1], r[2], alternative="two-sided").pvalue)
        p_12_gt = float(wilcoxon(r[1], r[2], alternative="greater").pvalue)
    except ValueError:
        p_12_two = p_12_gt = float("nan")
    try:
        p_23_two = float(wilcoxon(r[2], r[3], alternative="two-sided").pvalue)
        p_23_gt = float(wilcoxon(r[2], r[3], alternative="greater").pvalue)
    except ValueError:
        p_23_two = p_23_gt = float("nan")

    ci12 = cluster_bootstrap_diff(r[1] - r[2], sids, n_bootstrap, seed)
    ci23 = cluster_bootstrap_diff(r[2] - r[3], sids, n_bootstrap, seed)

    decision = decide_outcome(rho_12, p_12_two, rho_23, p_23_two)
    bonferroni_rejects_first_order = (rho_12 >= 1.20) and (p_12_two < BONFERRONI_THRESHOLD)

    return dict(
        arch=arch,
        p=p,
        function_class=function_class,
        n_quads=int(sids.size),
        R1_mean=float(r[1].mean()),
        R2_mean=float(r[2].mean()),
        R3_mean=float(r[3].mean()),
        rho_12=rho_12,
        rho_23=rho_23,
        wilcoxon_p_12_two_sided=p_12_two,
        wilcoxon_p_12_one_sided_R1_gt_R2=p_12_gt,
        wilcoxon_p_23_two_sided=p_23_two,
        wilcoxon_p_23_one_sided_R2_gt_R3=p_23_gt,
        bootstrap_ci_R1_minus_R2_lo=float(ci12[0]),
        bootstrap_ci_R1_minus_R2_hi=float(ci12[1]),
        bootstrap_ci_R2_minus_R3_lo=float(ci23[0]),
        bootstrap_ci_R2_minus_R3_hi=float(ci23[1]),
        decision_per_protocol_6_4=decision,
        bonferroni_rejects_first_order=bool(bonferroni_rejects_first_order),
        runtime_seconds=float(runtime),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpt2_quads", required=True)
    ap.add_argument("--pythia_quads", required=True)
    ap.add_argument("--output_path", required=True,
                    help="Where to write robustness_grid.csv")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--n_bootstrap", type=int, default=10_000)
    args = ap.parse_args()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    archs = [
        ("gpt2", _load_quads(Path(args.gpt2_quads))),
        ("pythia-160m", _load_quads(Path(args.pythia_quads))),
    ]

    results: list[dict] = []
    cell_idx = 0
    total = N_CELLS
    print(f"[robust] sweeping {total} cells "
          f"({len(CLASSES)} classes x {len(PCA_DIMS)} PCA dims x {len(archs)} archs)")
    print(f"[robust] Bonferroni threshold p < {BONFERRONI_THRESHOLD:.4g}")
    sweep_t0 = time.time()
    for arch_name, quads in archs:
        for p in PCA_DIMS:
            for fc in CLASSES:
                cell_idx += 1
                msg = f"[{cell_idx}/{total}] arch={arch_name} p={p} class={fc}"
                print(msg, flush=True)
                row = run_cell(
                    arch=arch_name, quads=quads, p=p, function_class=fc,
                    n_jobs=args.n_jobs, seed=args.seed, n_bootstrap=args.n_bootstrap,
                )
                results.append(row)
                print(f"   -> rho_12={row['rho_12']:.4f}  "
                      f"p_12_two={row['wilcoxon_p_12_two_sided']:.3g}  "
                      f"rho_23={row['rho_23']:.4f}  "
                      f"p_23_two={row['wilcoxon_p_23_two_sided']:.3g}  "
                      f"decision={row['decision_per_protocol_6_4']}  "
                      f"({row['runtime_seconds']:.1f}s)", flush=True)

    print(f"\n[robust] sweep complete in {(time.time()-sweep_t0)/60:.1f} min")

    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[robust] wrote {out_path}")

    # Summary block (decision counts)
    decisions = [r["decision_per_protocol_6_4"] for r in results]
    rej_count = sum(1 for r in results if r["bonferroni_rejects_first_order"])
    summary_payload = {
        "n_cells": len(results),
        "decisions": {x: decisions.count(x) for x in ["A", "B", "C", "D"]},
        "bonferroni_rejecting_cells": rej_count,
        "bonferroni_threshold": BONFERRONI_THRESHOLD,
    }
    sum_path = out_path.with_name(out_path.stem + "_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"[robust] wrote {sum_path}")
    print(f"[robust] decision counts: {summary_payload['decisions']}")
    print(f"[robust] cells rejecting first-order at Bonferroni "
          f"({BONFERRONI_THRESHOLD:.2g}): {rej_count}/{N_CELLS}")


if __name__ == "__main__":
    main()
