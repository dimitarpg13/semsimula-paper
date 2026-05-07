"""
Aggregator for the Helmholtz (Q9d) H0 + H1 schedule sweep.

Reads:
  helmholtz/results/h1_sweep/<schedule>*/seed{S}/helm_*_summary.md

Writes:
  helmholtz/results/h1_sweep/H1_RESULTS.md   summary table

Comparison anchors (recorded in
companion_notes/HSPLM_Path_Forward_and_Experiments.md):
  - All-attention baseline (matched_baseline_model)         val PPL ~150
  - All-SPLM em_ln, leak-free, free gamma                   val PPL ~173.59
  - Variant A HSPLM (k=4, m=4) at S=1                       val PPL ~133.01
  - Variant A HSPLM best-of-sweep                           ~133–147 across (k,m)
  - Pre-registered title-justification rule                 §6.5 of
      the v4 title-justification rule

Usage:
  python3 aggregate_h1.py [--sweep-dir helmholtz/results/h1_sweep]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CellResult:
    schedule: str
    n_S: int
    n_A: int
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

    schedule = "?"
    n_S = n_A = -1
    seed = -1
    fixed_gamma: Optional[float] = None
    n_params = 0
    final_train = final_val = final_val_ppl = final_gamma = None
    elapsed_s: Optional[float] = None

    # Tag-based fields parse out of the path name; trainer writes
    # tag = helm_<schedule>[_g<fixed_gamma>]_<mode>_seed{seed}_summary
    name = path.stem
    parts = name.split("_")
    if parts and parts[0] == "helm" and len(parts) >= 2:
        schedule = parts[1]
    for p in parts:
        if p.startswith("g") and p != "g":
            try:
                fixed_gamma = float(p[1:])
            except ValueError:
                pass
        elif p.startswith("seed") and p[4:].isdigit():
            seed = int(p[4:])

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- Schedule:"):
            try:
                # "- Schedule: `AAAASSSS` (n_S=4, n_A=4, L=8)"
                inside = line.split("`")[1]
                schedule = inside
                lhs = line.split("(", 1)[1].rstrip(")")
                for kv in lhs.split(","):
                    k, v = kv.strip().split("=")
                    if k == "n_S":
                        n_S = int(v)
                    elif k == "n_A":
                        n_A = int(v)
            except (IndexError, ValueError):
                pass
        elif line.startswith("- Parameters:"):
            try:
                n_params = int(line.split("**")[1].replace(",", ""))
            except (IndexError, ValueError):
                pass
        elif line.startswith("- Final train loss:"):
            try:
                final_train = float(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("- Final val loss:"):
            try:
                lhs, rhs = line.split("(ppl")
                final_val = float(lhs.split(":")[1].strip())
                final_val_ppl = float(rhs.replace(")", "").strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("- Final gamma:"):
            try:
                final_gamma = float(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("- Wall-clock time:"):
            try:
                elapsed_s = float(line.split(":")[1].strip().rstrip("s"))
            except ValueError:
                pass

    if n_S < 0 or n_A < 0:
        # Fallback: count from the schedule string we did parse.
        n_S = sum(1 for c in schedule if c.upper() == "S")
        n_A = sum(1 for c in schedule if c.upper() == "A")

    return CellResult(
        schedule=schedule, n_S=n_S, n_A=n_A, seed=seed,
        fixed_gamma=fixed_gamma, n_params=n_params,
        final_train=final_train, final_val=final_val,
        final_val_ppl=final_val_ppl, final_gamma=final_gamma,
        elapsed_s=elapsed_s, summary_path=path,
    )


def gather(sweep_dir: Path) -> List[CellResult]:
    pattern = "*/seed*/helm_*_summary.md"
    cells = sorted(sweep_dir.glob(pattern))
    results: List[CellResult] = []
    for s in cells:
        try:
            results.append(parse_summary(s))
        except Exception as exc:
            print(f"[helm-agg] WARN failed to parse {s}: {exc}")
    return results


def render(results: List[CellResult], out_path: Path) -> None:
    if not results:
        out_path.write_text(
            "# H1 Helmholtz schedule sweep — no results parsed yet.\n"
        )
        print(f"[helm-agg] no results found; wrote {out_path}")
        return

    # Sort by (n_A, n_S, schedule, seed) so attention-heavy and
    # SPLM-heavy schedules cluster naturally.
    results.sort(key=lambda r: (r.n_A, r.n_S, r.schedule,
                                r.fixed_gamma or -1, r.seed))

    lines = [
        "# H1 — Helmholtz (Q9d) schedule sweep results\n",
        "",
        "Architecture: layer-type Helmholtz hybrid -- single shared",
        "`V_theta` on every S-block, per-layer attention on every A-block,",
        "velocity-Verlet damped Euler-Lagrange S-step.  Schedule "
        "`sigma : {0..L-1} -> {S, A}` controls the architectural shape.",
        "",
        "Tied embeddings, `causal_force=True`, `ln_after_s_step=True`,",
        "`mass_mode='logfreq'`, Tiny Shakespeare 4000 steps at d=128.",
        "",
        "## Per-schedule results",
        "",
        "| schedule | n_S | n_A | seed | γ_mode | val PPL | val loss | "
        "final γ | params | elapsed |",
        "|----------|-----|-----|------|--------|---------|----------|"
        "---------|--------|---------|",
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
            f"| `{r.schedule}` | {r.n_S} | {r.n_A} | {r.seed} | {gmode} | "
            f"{ppl} | {vl} | {fg} | {params_m} | {elapsed} |"
        )

    lines += [
        "",
        "## Anchors (already on disk, leak-free)",
        "",
        "| arm | val PPL | source |",
        "|-----|---------|--------|",
        "| All-attention (matched GPT-2, n_layer=8)        | ~150       | `multi_seed/results/` |",
        "| All-SPLM em_ln (free-γ, leak-free)              | 173.59     | `energetic_minima/results/` |",
        "| Variant A HSPLM (k=4, m=4) at S=1               | **133.01** | `hybrid/results/h1_sweep/H1_RESULTS.md` |",
        "| Variant A HSPLM best across (k, m)              | 133.01–147.28 | `hybrid/results/h1_sweep/H1_RESULTS.md` |",
        "",
        "## Q9d-vs-Variant-A comparison",
        "",
        "The Helmholtz `bottom_a_LA4` schedule (`AAAASSSS`) is the closest",
        "Q9d analogue of Variant A HSPLM (k=4, m=4): same param count,",
        "same compute, same n_S/n_A.  Two structural differences are",
        "expected to drive any divergence in PPL:",
        "",
        "1. The velocity proxy `h_ell - h_{ell-1}` passes *through* the",
        "   attention stack rather than being reset to v=0 at the SPLM",
        "   boundary.",
        "2. `xi` is re-derived from the running `h.detach()` at every",
        "   S-block, instead of being fixed at `h_k.detach()` once after",
        "   the attention stack.",
        "",
        "If `bottom_a_LA4` matches Variant A's 133.01 within seed noise,",
        "the velocity-passing and xi-re-derivation are kinematically",
        "equivalent at this scale.  The interesting cells are the",
        "schedules unreachable by Variant A: `sandwich_k1` (boundary-case",
        "S-blocks, doc §A.5 mechanism), `interleaved` (block-type-indexed",
        "step-function R²_ψ test, doc §4.1), `top_a_LA1` (single-attention",
        "hybrid, the cleanest narrative cell of doc §6).",
        "",
        "## Pre-registered decision (Phase 2 gate)",
        "",
        "Per `the v4 title-justification rule` §6.5:",
        "**\"Efficient\" is justified iff** some hybrid achieves val PPL",
        "within +5 PPL of the all-attention baseline AND its analytical",
        "decode-FLOP cost at T=1024 is ≥ 30% lower than all-attention,",
        "both at S=3 with sign-consistency 3/3.",
        "",
        "**Phase 1 → Phase 2 gate:** if best schedule at S=1 (this sweep)",
        "is within +10 PPL of all-attention, proceed to H2 (S=3",
        "confirmation).  If gap is > 15 PPL, soften the title (Option 2",
        "fallback) and document the schedule sweep as Future Work.",
        "",
        "## Best-of-sweep summary",
        "",
    ]

    finite = [r for r in results if r.final_val_ppl is not None]
    if finite:
        best = min(finite, key=lambda r: r.final_val_ppl or float("inf"))
        gap_to_attn = (best.final_val_ppl or 0.0) - 150.0
        gap_to_va_best = (best.final_val_ppl or 0.0) - 133.01
        lines += [
            f"Best schedule: **`{best.schedule}`** "
            f"(n_S={best.n_S}, n_A={best.n_A}) at seed {best.seed} "
            f"with val PPL **{best.final_val_ppl:.2f}**.",
            f"Gap to all-attention (~150): **{gap_to_attn:+.2f} PPL**.",
            f"Gap to Variant A best (133.01): {gap_to_va_best:+.2f} PPL.",
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
    print(f"[helm-agg] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir",
                    default=str(Path(__file__).parent / "results" / "h1_sweep"))
    ap.add_argument("--out", default=None,
                    help="default: <sweep-dir>/H1_RESULTS.md")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_path = (Path(args.out) if args.out is not None
                else sweep_dir / "H1_RESULTS.md")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results = gather(sweep_dir)
    render(results, out_path)
    print(f"[helm-agg] {len(results)} schedule summaries parsed")


if __name__ == "__main__":
    main()
