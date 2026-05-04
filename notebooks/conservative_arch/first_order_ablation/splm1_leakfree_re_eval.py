"""Leak-free re-evaluation driver for the SPLM-1 vs SPLM-2 TinyShakespeare
6-cell sweep (3 seeds × {SPLM-1 first-order, SPLM-2 em_ln γ=0.30}).

Background
----------
The 6-cell sweep that produced `results/RESULTS.md` was trained against the
*pre-fix* SPLM `integrate()` loop, in which `ξ` was computed from `h` without
detaching. That autograd path leaks future-token information into the
gradient at past positions (see `docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`).
Both arms — SPLM-1 (first-order, gradient-flow) and SPLM-2 (second-order,
em_ln γ=0.30) — were affected: the bug-fix doc lists both in §1.3.

This driver loads each of the 6 saved `.pt` checkpoints under
`cfg.causal_force ∈ {False, True}` and reports paired PPL on the SAME
random val batches under each. The ratio `PPL_fixed / PPL_buggy` is the
forensic inflation factor for that ckpt: a value > 1 means the model's
training-time perplexity was leak-amplified.

The output is forensic, not the same as a leak-free retraining: the
weights are still those produced by training under the buggy gradient.
But it is sufficient to bound the fraction of the published 23.18-PPL
gap that is leak-amplification rather than genuine architectural lift.

Usage
-----
    python3 notebooks/conservative_arch/first_order_ablation/splm1_leakfree_re_eval.py

    # Custom out path / seed:
    python3 notebooks/conservative_arch/first_order_ablation/splm1_leakfree_re_eval.py \
        --out notebooks/conservative_arch/first_order_ablation/results/leakfree_re_eval.json \
        --seed 0

Outputs
-------
- `<results-dir>/leakfree_re_eval.json` — per-ckpt (buggy, fixed) PPL.
- `<results-dir>/LEAKFREE_RE_EVAL.md` — markdown summary table with
  per-seed paired comparison and arm-level paired-t / Cohen's d_z.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


THIS_DIR = Path(__file__).parent
RESULTS_ROOT = THIS_DIR / "results"
EVAL_SCRIPT = THIS_DIR.parent / "eval_ppl_under_fix.py"


@dataclass
class CkptCell:
    arm: str          # "splm1" or "splm2"
    seed: int
    path: Path


# Eval params matched to the original trainer's `_evaluate` call site
# (notebooks/conservative_arch/first_order_ablation/train_splm_first_order.py
#  shakespeare mode: batch_size=16, block_size=128, eval_iters=40).
DEFAULT_N_BATCHES = 40
DEFAULT_BATCH = 16
DEFAULT_BLOCK = 128
DEFAULT_CORPUS = "shakespeare"
DEFAULT_DEVICE = "cpu"   # CPU so we coexist with concurrent MPS training jobs


def discover_ckpts() -> List[CkptCell]:
    cells: List[CkptCell] = []
    for arm_dir, arm in [("splm1", "splm1"), ("splm2_gamma0p30", "splm2")]:
        for seed in (0, 1, 2):
            seed_dir = RESULTS_ROOT / arm_dir / f"seed{seed}"
            ckpts = sorted(seed_dir.glob("*ckpt_latest.pt"))
            if len(ckpts) != 1:
                raise FileNotFoundError(
                    f"expected exactly one ckpt_latest.pt under {seed_dir}, "
                    f"found {len(ckpts)}: {ckpts}"
                )
            cells.append(CkptCell(arm=arm, seed=seed, path=ckpts[0]))
    return cells


def run_one(cell: CkptCell, n_batches: int, batch: int, block: int,
            corpus: str, seed: int, device: str) -> dict:
    """Run eval_ppl_under_fix.py on one ckpt and parse stdout for the
    paired (buggy, fixed) PPL pair plus inflation factor."""
    cmd = [
        sys.executable, str(EVAL_SCRIPT), str(cell.path),
        "--n-batches", str(n_batches),
        "--batch", str(batch),
        "--block", str(block),
        "--corpus", corpus,
        "--seed", str(seed),
        "--device", device,
    ]
    print(f"[driver] arm={cell.arm} seed={cell.seed}  ckpt={cell.path.name}")
    print(f"[driver]   {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    stdout = proc.stdout
    stderr = proc.stderr
    if proc.returncode != 0:
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)
        raise RuntimeError(
            f"eval_ppl_under_fix.py failed for {cell.path} "
            f"(exit {proc.returncode}); see stderr above."
        )

    # Parse e.g.:
    #   [ppl-inflation] buggy: loss=4.7012  ppl=110.13
    #   [ppl-inflation] fixed: loss=5.7894  ppl=327.39
    #   [ppl-inflation] inflation factor PPL_fixed / PPL_buggy = 2.97x
    pat = re.compile(
        r"\[ppl-inflation\]\s+(buggy|fixed):\s+loss=([0-9.]+)\s+ppl=([0-9.]+)"
    )
    found = {m.group(1): (float(m.group(2)), float(m.group(3)))
             for m in pat.finditer(stdout)}
    if "buggy" not in found or "fixed" not in found:
        sys.stdout.write(stdout)
        raise RuntimeError(
            f"could not parse buggy/fixed PPL from eval output for {cell.path}"
        )
    loss_buggy, ppl_buggy = found["buggy"]
    loss_fixed, ppl_fixed = found["fixed"]
    inflation = ppl_fixed / ppl_buggy
    print(f"[driver]   -> buggy ppl={ppl_buggy:.2f}  "
          f"fixed ppl={ppl_fixed:.2f}  inflation={inflation:.2f}x")
    return {
        "arm": cell.arm,
        "seed": cell.seed,
        "ckpt": str(cell.path),
        "loss_buggy": loss_buggy,
        "ppl_buggy": ppl_buggy,
        "loss_fixed": loss_fixed,
        "ppl_fixed": ppl_fixed,
        "inflation": inflation,
        "n_batches": n_batches,
        "batch": batch,
        "block": block,
        "corpus": corpus,
        "seed_eval": seed,
        "device": device,
    }


def _paired_t(deltas: List[float]) -> dict:
    """Two-sided paired-t for the mean of `deltas` against zero, plus
    Cohen's d_z. Returns a dict with mean, std, t, df, and d_z. We do
    not import scipy here; with df=2 the user can look up the p-value
    from the t-statistic directly (one-sided thresholds: t≥2.92 for
    p≤0.05, t≥6.97 for p≤0.01)."""
    n = len(deltas)
    if n < 2:
        return {"n": n, "mean": deltas[0] if deltas else 0.0, "std": 0.0,
                "t": float("nan"), "df": 0, "d_z": float("nan")}
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n) if std > 0 else float("nan")
    t = mean / se if se == se and se > 0 else float("nan")
    d_z = mean / std if std > 0 else float("nan")
    return {"n": n, "mean": mean, "std": std, "t": t, "df": n - 1, "d_z": d_z}


def write_markdown_summary(results: List[dict], out_path: Path) -> None:
    by_arm: dict = {"splm1": [], "splm2": []}
    for r in results:
        by_arm[r["arm"]].append(r)
    for arm in by_arm:
        by_arm[arm].sort(key=lambda r: r["seed"])

    s1 = by_arm["splm1"]
    s2 = by_arm["splm2"]

    delta_buggy = [a["ppl_buggy"] - b["ppl_buggy"] for a, b in zip(s1, s2)]
    delta_fixed = [a["ppl_fixed"] - b["ppl_fixed"] for a, b in zip(s1, s2)]
    stat_buggy = _paired_t(delta_buggy)
    stat_fixed = _paired_t(delta_fixed)

    lines: List[str] = []
    lines.append("# SPLM-1 vs SPLM-2 — leak-free re-evaluation of the "
                 "TinyShakespeare 6-cell sweep")
    lines.append("")
    lines.append("> **Forensic re-evaluation only.** The trained weights are "
                 "those produced by training under the *buggy* SPLM "
                 "`integrate()` loop. We evaluate each ckpt under the leak-"
                 "free integrator (`cfg.causal_force = True`) on the same "
                 "random val batches as the buggy integrator. The resulting "
                 "(`PPL_buggy`, `PPL_fixed`) pair quantifies how much of "
                 "the trained-time PPL claim was an autograd-leak artefact "
                 "for *this* checkpoint. A definitive replication requires "
                 "leak-free retraining of the 6-cell sweep.")
    lines.append("")
    lines.append("## 1. Eval configuration")
    lines.append("")
    if results:
        cfg = results[0]
        lines.append(f"- corpus: `{cfg['corpus']}`")
        lines.append(f"- n_batches: {cfg['n_batches']}")
        lines.append(f"- batch: {cfg['batch']}")
        lines.append(f"- block: {cfg['block']}")
        lines.append(f"- device: `{cfg['device']}`")
        lines.append(f"- val-batch RNG seed: {cfg['seed_eval']} (same seed "
                     f"used for the buggy and fixed evaluators, so they "
                     f"see the identical val tokens for a paired comparison).")
    lines.append("")

    lines.append("## 2. Per-ckpt paired (buggy, fixed) PPL")
    lines.append("")
    lines.append("Inflation factor = `PPL_fixed / PPL_buggy`. Values > 1 "
                 "mean the buggy training-time integrator overstated this "
                 "ckpt's val performance.")
    lines.append("")
    lines.append("| arm | seed | ckpt | PPL (buggy) | PPL (fixed) | "
                 "inflation × |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for r in s1 + s2:
        lines.append(
            f"| {r['arm']} | {r['seed']} | `{Path(r['ckpt']).name}` | "
            f"{r['ppl_buggy']:.2f} | {r['ppl_fixed']:.2f} | "
            f"{r['inflation']:.2f}× |"
        )
    lines.append("")

    lines.append("## 3. Paired SPLM-1 − SPLM-2 deltas")
    lines.append("")
    lines.append("| seed | Δ_PPL (buggy) | Δ_PPL (fixed) |")
    lines.append("|---:|---:|---:|")
    for i, sd in enumerate([0, 1, 2]):
        lines.append(
            f"| {sd} | {delta_buggy[i]:+.2f} | {delta_fixed[i]:+.2f} |"
        )
    lines.append(
        f"| **mean** | **{stat_buggy['mean']:+.2f}** | "
        f"**{stat_fixed['mean']:+.2f}** |"
    )
    lines.append(
        f"| **std**  | {stat_buggy['std']:.2f} | "
        f"{stat_fixed['std']:.2f} |"
    )
    lines.append("")
    lines.append(
        f"- **buggy integrator** (matches published RESULTS.md headline of "
        f"+23.18 PPL): paired-t = {stat_buggy['t']:.2f}, df = "
        f"{stat_buggy['df']}, Cohen's d_z = {stat_buggy['d_z']:.2f}."
    )
    lines.append(
        f"- **fixed integrator** (leak-free): paired-t = "
        f"{stat_fixed['t']:.2f}, df = {stat_fixed['df']}, Cohen's d_z = "
        f"{stat_fixed['d_z']:.2f}."
    )
    lines.append("")

    lines.append("## 4. Interpretation guide")
    lines.append("")
    lines.append("Three coarse outcomes are possible:")
    lines.append("")
    lines.append(
        "- **(a) Both arms re-evaluate close to their published numbers** "
        "(inflation ≈ 1× for both). The TinyShakespeare comparison is "
        "robust to the leak-fix; the published 23.18-PPL gap is a clean "
        "second-order-vs-first-order architectural lift. Paper v3 §15 may "
        "stand as written, with a footnote citing this re-evaluation.")
    lines.append(
        "- **(b) Both arms inflate roughly equally**, the paired delta "
        "under the fixed integrator is similar in sign and magnitude. "
        "Published gap is *qualitatively* preserved but the absolute PPLs "
        "are not safe to claim. Paper v3 §15 needs the absolute numbers "
        "replaced (after a leak-free retrain) but the qualitative "
        "conclusion stands.")
    lines.append(
        "- **(c) Asymmetric inflation** — one arm exploits the leak more "
        "than the other, so Δ(fixed) ≠ Δ(buggy). If SPLM-2's lead shrinks "
        "or inverts under the fix, the published paper-v3 §15 conclusion "
        "is at risk. Definitive resolution requires a 6-cell leak-free "
        "retrain.")
    lines.append("")
    lines.append("These outcome categories are pre-registered to remove "
                 "post-hoc reading of the data. Inspect the table in §3 "
                 "and assign one of (a), (b), (c) before drawing further "
                 "implications.")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"[driver] wrote markdown summary -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", type=Path,
                    default=RESULTS_ROOT / "leakfree_re_eval.json",
                    help="Path for machine-readable JSON output.")
    ap.add_argument("--out-md", type=Path,
                    default=RESULTS_ROOT / "LEAKFREE_RE_EVAL.md",
                    help="Path for human-readable markdown summary.")
    ap.add_argument("--n-batches", type=int, default=DEFAULT_N_BATCHES)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for val batches (default 0).")
    ap.add_argument("--device", choices=("auto", "cpu", "mps"),
                    default=DEFAULT_DEVICE,
                    help="Compute device. Default 'cpu' so the driver "
                         "coexists with concurrent MPS training jobs.")
    args = ap.parse_args()

    cells = discover_ckpts()
    print(f"[driver] discovered {len(cells)} ckpts:")
    for c in cells:
        print(f"  - {c.arm} seed={c.seed}  {c.path}")
    print()

    results: List[dict] = []
    for c in cells:
        r = run_one(c, args.n_batches, args.batch, args.block,
                    args.corpus, args.seed, args.device)
        results.append(r)
        # Stream-write JSON after each ckpt so a partial run is recoverable.
        args.out_json.write_text(
            json.dumps({"results": results}, indent=2)
        )

    print()
    print(f"[driver] all {len(results)} ckpts evaluated.")
    print(f"[driver] wrote JSON -> {args.out_json}")
    write_markdown_summary(results, args.out_md)


if __name__ == "__main__":
    main()
