"""
Aggregator for H0 + H1 layer-split sweep results.

Reads:
  hybrid/results/h1_sweep/k{k}_m{m}*/seed{S}/hybrid_*_summary.md
  hybrid/results/h1_sweep/k{k}_m{m}*/seed{S}/hybrid_*_ckpt_latest.pt

Writes:
  hybrid/results/h1_sweep/H1_RESULTS.md   summary table

Comparison anchors (recorded in
the v4 title-justification rule §3):
  - All-attention baseline (matched_baseline_model)         val PPL ~150
  - All-SPLM em_ln, leak-free, free gamma                   val PPL ~173.59
  - Pre-registered title-justification rule                 §6.5

Usage:
  python3 aggregate_h1.py [--sweep-dir hybrid/results/h1_sweep]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch


@dataclass
class CellResult:
    n_attn: int
    n_splm: int
    seed: int
    fixed_gamma: Optional[float]
    n_params: int
    final_train: Optional[float]
    final_val: Optional[float]
    final_val_ppl: Optional[float]
    final_gamma: Optional[float]
    elapsed_s: Optional[float]
    summary_path: Path


def parse_summary(path: Path) -> CellResult:
    """Parse the trainer's _summary.md to extract reported numbers."""
    text = path.read_text()
    n_attn = n_splm = -1
    fixed_gamma: Optional[float] = None
    seed = -1
    n_params = 0
    final_train = final_val = final_val_ppl = final_gamma = None
    elapsed_s: Optional[float] = None

    # Tag-based fields parse out of the path name; trainer writes
    # tag = hybrid_k{n_attn}_m{n_splm}[_g{fixed_gamma}]_<mode>_seed{seed}
    name = path.stem    # hybrid_kK_mM[_gGAMMA]_<mode>_seedS_summary
    parts = name.split("_")
    for p in parts:
        if p.startswith("k") and p[1:].isdigit():
            n_attn = int(p[1:])
        elif p.startswith("m") and p[1:].isdigit():
            n_splm = int(p[1:])
        elif p.startswith("g"):
            try:
                fixed_gamma = float(p[1:])
            except ValueError:
                pass
        elif p.startswith("seed") and p[4:].isdigit():
            seed = int(p[4:])

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- Parameters:"):
            n_params = int(line.split("**")[1].replace(",", ""))
        elif line.startswith("- Final train loss:"):
            final_train = float(line.split(":")[1].strip())
        elif line.startswith("- Final val loss:"):
            try:
                lhs, rhs = line.split("(ppl")
                final_val = float(lhs.split(":")[1].strip())
                final_val_ppl = float(rhs.replace(")", "").strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("- Final gamma:"):
            final_gamma = float(line.split(":")[1].strip())
        elif line.startswith("- Wall-clock time:"):
            try:
                elapsed_s = float(line.split(":")[1].strip().rstrip("s"))
            except ValueError:
                pass

    return CellResult(
        n_attn=n_attn, n_splm=n_splm, seed=seed,
        fixed_gamma=fixed_gamma, n_params=n_params,
        final_train=final_train, final_val=final_val,
        final_val_ppl=final_val_ppl, final_gamma=final_gamma,
        elapsed_s=elapsed_s, summary_path=path,
    )


def gather(sweep_dir: Path) -> List[CellResult]:
    pattern = "*/seed*/hybrid_*_summary.md"
    cells = sorted(sweep_dir.glob(pattern))
    results: List[CellResult] = []
    for s in cells:
        try:
            results.append(parse_summary(s))
        except Exception as exc:
            print(f"[agg] WARN failed to parse {s}: {exc}")
    return results


def render(results: List[CellResult], out_path: Path) -> None:
    if not results:
        out_path.write_text("# H1 layer-split sweep — no results parsed yet.\n")
        print(f"[agg] no results found; wrote {out_path}")
        return

    # Sort by (n_attn, n_splm, seed)
    results.sort(key=lambda r: (r.n_attn, r.n_splm, r.fixed_gamma or -1, r.seed))

    lines = [
        "# H1 — Layer-split sweep results (Variant A two-stage hybrid)\n",
        "",
        "Architecture: `k` attention blocks (front) + `m` SPLM integration",
        "steps (back), tied embeddings, single shared V_theta,",
        "`causal_force=True`, `ln_after_step=True`, `mass_mode='logfreq'`,",
        "Tiny Shakespeare 4000 steps at d=128.",
        "",
        "## Per-cell results",
        "",
        "| (k, m) | seed | γ_mode | val PPL | val loss | final γ | params | elapsed |",
        "|--------|------|--------|---------|----------|---------|--------|---------|",
    ]
    for r in results:
        gmode = (f"fixed γ={r.fixed_gamma:.3f}"
                 if r.fixed_gamma is not None else "free γ")
        ppl = f"{r.final_val_ppl:.2f}" if r.final_val_ppl is not None else "—"
        vl = f"{r.final_val:.4f}" if r.final_val is not None else "—"
        fg = f"{r.final_gamma:.3f}" if r.final_gamma is not None else "—"
        params_m = f"{r.n_params / 1e6:.2f} M" if r.n_params else "—"
        elapsed = (f"{r.elapsed_s/60:.1f} min"
                   if r.elapsed_s is not None else "—")
        lines.append(
            f"| ({r.n_attn}, {r.n_splm}) | {r.seed} | {gmode} | {ppl} | "
            f"{vl} | {fg} | {params_m} | {elapsed} |"
        )

    lines += [
        "",
        "## Anchors (already on disk, leak-free)",
        "",
        "| arm | val PPL | source |",
        "|-----|---------|--------|",
        "| All-attention (matched GPT-2, n_layer=8)        | ~150     | `multi_seed/results/` |",
        "| All-SPLM em_ln (free-γ, leak-free)              | 173.59   | `energetic_minima/results/` |",
        "| All-SPLM em_ln (γ=0.10, leak-free)              | ~178–181 | `ln_damping_sweep/results/RESULTS_LEAKFREE_GAMMA_SWEEP.md` |",
        "",
        "## Pre-registered decision (Phase 2 gate)",
        "",
        "Per `the v4 title-justification rule` §6.5:",
        "**\"Efficient\" is justified iff** some hybrid (k, m) achieves val PPL",
        "within +5 PPL of the all-attention baseline AND its analytical decode-FLOP",
        "cost at T=1024 is ≥ 30% lower than all-attention, both at S=3 with sign 3/3.",
        "",
        "**Phase 1 → Phase 2 gate:** if best (k, m) at S=1 (this sweep) is",
        "within +10 PPL of all-attention, proceed to H2 (S=3 confirmation).",
        "If gap is > 15 PPL, soften the title (Option 2 fallback) and",
        "document the hybrid as Future Work.",
        "",
        "## Best-of-sweep summary",
        "",
    ]

    finite = [r for r in results if r.final_val_ppl is not None]
    if finite:
        best = min(finite, key=lambda r: r.final_val_ppl or float("inf"))
        gap_to_attn = (best.final_val_ppl or 0.0) - 150.0
        gap_to_splm = (best.final_val_ppl or 0.0) - 173.59
        lines += [
            f"Best cell: **(k={best.n_attn}, m={best.n_splm})** "
            f"at seed {best.seed} with val PPL **{best.final_val_ppl:.2f}**.",
            f"Gap to all-attention (~150): **{gap_to_attn:+.2f} PPL**.",
            f"Gap to all-SPLM em_ln free-γ (~173.59): {gap_to_splm:+.2f} PPL.",
            "",
            ("Phase 2 gate: " + (
                "**PASS** (within +10 PPL of all-attention)."
                if gap_to_attn <= 10.0
                else ("**MARGINAL** (+10 to +15 PPL of all-attention; "
                      "consider one extra reconnaissance cell or proceed "
                      "to H2 cautiously).")
                if gap_to_attn <= 15.0
                else "**FAIL** (gap > 15 PPL of all-attention; soften title)."
            )),
        ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"[agg] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir",
                    default=str(Path(__file__).parent / "results" / "h1_sweep"))
    ap.add_argument("--out",
                    default=None,
                    help="default: <sweep-dir>/H1_RESULTS.md")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_path = (Path(args.out) if args.out is not None
                else sweep_dir / "H1_RESULTS.md")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results = gather(sweep_dir)
    render(results, out_path)
    print(f"[agg] {len(results)} cell summaries parsed")


if __name__ == "__main__":
    main()
