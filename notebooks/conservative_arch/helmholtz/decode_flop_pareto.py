"""
Helmholtz (Q9d) -- Analytical decode-FLOP Pareto across schedules.

The per-token AR decode cost for the layer-type Helmholtz hybrid
depends only on the *count* of A-blocks (KV cached, O(T) each) and
S-blocks (streaming-xi, O(1) each), not the order of blocks in the
schedule.  The composition is therefore:

    per_token = embed
              + sum over A-blocks: attn_decode_block_only_flops(T)
              + sum over S-blocks: splm_step_only_flops()
              + final logits

where the per-block helpers are imported from the existing Variant A
Pareto (`hybrid/decode_flop_pareto.py`) so the two paths report
identical numbers for matched (n_A, n_S) cells.

Output: appends a "Decode-FLOP Pareto" section to
  notebooks/conservative_arch/helmholtz/results/h1_sweep/H1_RESULTS.md
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
HYBRID_DIR = PARENT_DIR / "hybrid"

sys.path.insert(0, str(PARENT_DIR))

from inference_efficiency.flop_counter import (  # noqa: E402
    AttnFLOPParams,
    SPLMFLOPParams,
)

# Load hybrid/decode_flop_pareto.py under a distinct module name to
# avoid a self-import collision when *this* module (also named
# `decode_flop_pareto`) is imported by name (e.g. by aggregate_h1p5.py).
_HYBRID_FLOP_PATH = HYBRID_DIR / "decode_flop_pareto.py"
_spec = importlib.util.spec_from_file_location(
    "_hybrid_decode_flop_pareto", _HYBRID_FLOP_PATH,
)
_hybrid_flop_mod = importlib.util.module_from_spec(_spec)
sys.modules["_hybrid_decode_flop_pareto"] = _hybrid_flop_mod
_spec.loader.exec_module(_hybrid_flop_mod)
_attn_decode_block_only_flops = _hybrid_flop_mod._attn_decode_block_only_flops
_splm_step_only_flops = _hybrid_flop_mod._splm_step_only_flops

sys.path.insert(0, str(SCRIPT_DIR))
from model_helmholtz import (  # noqa: E402
    parse_schedule,
    schedule_counts,
)


def helmholtz_decode_token_flops(
    d: int,
    schedule: str,
    n_head: int,
    mlp_mult: int,
    v_hidden: int,
    v_depth: int,
    vocab_size: int,
    ln_after_step: bool,
    T: int,
) -> Dict:
    """Per-token AR decode FLOPs for a Helmholtz schedule at context T."""
    sigma = parse_schedule(schedule)
    nS, nA = schedule_counts(sigma)

    p_attn = AttnFLOPParams(d=d, L=nA, n_head=n_head,
                            mlp_mult=mlp_mult, vocab_size=vocab_size)
    p_splm = SPLMFLOPParams(d=d, L=nS, v_hidden=v_hidden,
                            v_depth=v_depth, vocab_size=vocab_size,
                            ln_after_step=ln_after_step)

    flops = 0
    flops += d                                                # embed + pos
    flops += nA * _attn_decode_block_only_flops(p_attn, T)
    if ln_after_step and nS > 0:
        flops += 5 * d                                        # initial ln
    flops += nS * _splm_step_only_flops(p_splm)
    flops += 2 * d * vocab_size                               # final logits
    return {
        "per_token": int(flops),
        "T": T,
        "n_S": nS,
        "n_A": nA,
        "schedule": schedule,
    }


def make_pareto_table(
    cells: List[Tuple[str, float]],
    *,
    d: int = 128,
    n_head: int = 4,
    mlp_mult: int = 4,
    v_hidden: int = 512,
    v_depth: int = 3,
    vocab_size: int = 50257,
    ln_after_step: bool = True,
    T: int = 1024,
) -> Tuple[List[Dict], Dict]:
    """Cells: list of (schedule_string, val_ppl).  Returns (rows, refs)."""

    # All-attention reference at T (length-8 attention transformer).
    attn_full = helmholtz_decode_token_flops(
        d=d, schedule="A" * 8, n_head=n_head,
        mlp_mult=mlp_mult, v_hidden=v_hidden, v_depth=v_depth,
        vocab_size=vocab_size, ln_after_step=ln_after_step, T=T,
    )["per_token"]
    splm_full = helmholtz_decode_token_flops(
        d=d, schedule="S" * 8, n_head=n_head,
        mlp_mult=mlp_mult, v_hidden=v_hidden, v_depth=v_depth,
        vocab_size=vocab_size, ln_after_step=ln_after_step, T=T,
    )["per_token"]

    rows = []
    for schedule, ppl in cells:
        f = helmholtz_decode_token_flops(
            d=d, schedule=schedule, n_head=n_head,
            mlp_mult=mlp_mult, v_hidden=v_hidden, v_depth=v_depth,
            vocab_size=vocab_size, ln_after_step=ln_after_step, T=T,
        )
        rows.append({
            "schedule": schedule, "n_S": f["n_S"], "n_A": f["n_A"],
            "val_ppl": ppl,
            "decode_flops_per_token": f["per_token"],
            "ratio_vs_attn_full": f["per_token"] / attn_full,
            "reduction_vs_attn_full_pct":
                (1.0 - f["per_token"] / attn_full) * 100.0,
            "T": T,
        })
    return rows, {"attn_full_per_token": attn_full,
                  "splm_full_per_token": splm_full,
                  "T": T}


def main():
    """Append the Pareto table to H1_RESULTS.md.

    The cells list is hard-coded to the canonical schedule sweep
    (the same shapes the H0 / H1 launcher trains).  Update val PPL
    placeholders after the actual runs land.
    """
    out_path = (SCRIPT_DIR / "results" / "h1_sweep" / "H1_RESULTS.md")
    if not out_path.exists():
        raise SystemExit(
            f"missing {out_path}; run aggregate_h1.py first"
        )

    # Read the existing H1_RESULTS.md to extract per-schedule val PPL.
    # If a schedule has no recorded PPL, we list it with "—" so the
    # FLOP arm is still reported.
    text = out_path.read_text()
    cells: List[Tuple[str, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("| `"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 7:
            continue
        try:
            sched = parts[0].strip("`")
            ppl_str = parts[5]
            if ppl_str == "—":
                continue
            ppl = float(ppl_str)
            cells.append((sched, ppl))
        except (ValueError, IndexError):
            continue

    if not cells:
        # Fall back to the canonical 7-schedule placeholder list with no PPL,
        # so users at least see the FLOP-arm column for every schedule.
        from model_helmholtz import canonical_schedules
        cells = [(sigma, float("nan"))
                 for _, sigma in canonical_schedules(L=8)]

    print(f"[helm-flop] composing decode-FLOP Pareto for "
          f"{len(cells)} schedules at T=1024")

    section_lines = [
        "", "## Decode-FLOP Pareto (analytical)",
        "",
        "Per-token AR decode FLOPs at context length T "
        "(KV-cached attention; streaming-ξ SPLM).  Cost depends only on "
        "the *count* (n_A, n_S) of each block type, so schedules with "
        "the same (n_A, n_S) report identical FLOP cost — only val PPL "
        "differentiates them.",
        "",
    ]

    for T in (256, 1024, 4096):
        rows, refs = make_pareto_table(cells, T=T)
        section_lines += [
            f"",
            f"### T = {T}",
            f"",
            f"All-attention reference (`AAAAAAAA`): "
            f"**{refs['attn_full_per_token'] / 1e6:.3f} MFLOPs/tok**.",
            f"All-SPLM reference (`SSSSSSSS`): "
            f"**{refs['splm_full_per_token'] / 1e6:.3f} MFLOPs/tok**.",
            f"",
            f"| schedule | n_S | n_A | val PPL (S=1) | decode FLOPs/tok | "
            f"vs all-attn | reduction |",
            f"|----------|-----|-----|---------------|------------------|"
            f"-------------|-----------|",
        ]
        for r in rows:
            ppl_str = (f"{r['val_ppl']:.2f}" if r['val_ppl'] == r['val_ppl']
                       else "—")  # NaN check
            section_lines.append(
                f"| `{r['schedule']}` | {r['n_S']} | {r['n_A']} | "
                f"{ppl_str} | "
                f"{r['decode_flops_per_token'] / 1e6:.3f} MFLOPs | "
                f"{r['ratio_vs_attn_full']:.3f}× | "
                f"{r['reduction_vs_attn_full_pct']:+.1f}% |"
            )

    section_lines += [
        "",
        "### Pre-registered rule check",
        "",
        "Per `the v4 title-justification rule` §6.5:",
        "**\"Efficient\" is justified iff** some hybrid achieves val PPL",
        "within +5 PPL of the all-attention baseline (~150) AND decode-FLOP",
        "cost at T=1024 is ≥ 30% lower than all-attention, both at S=3 with",
        "sign 3/3.",
        "",
        "Note: at the prototype `v_hidden = 512` the SPLM step costs",
        "~3.94 MFLOPs/tok, which is ~4× per-step heavier than an attention",
        "block at T=1024 (~0.94 MFLOPs/tok).  Schedules with high n_S",
        "are therefore *more* expensive at T=1024 -- the same FLOP-arm",
        "failure mode the Variant A H1 sweep documents.  The H1.5 fix is",
        "to narrow `v_hidden`; in Q9d it applies identically since V_theta",
        "is the same module.",
        "",
        "_(Pareto computed by",
        "`notebooks/conservative_arch/helmholtz/decode_flop_pareto.py`",
        "from `inference_efficiency/flop_counter.py` and the per-block",
        "helpers in `hybrid/decode_flop_pareto.py`.)_",
        "",
    ]

    with out_path.open("a") as f:
        f.write("\n".join(section_lines) + "\n")
    print(f"[helm-flop] appended decode-FLOP Pareto to {out_path}")


if __name__ == "__main__":
    main()
