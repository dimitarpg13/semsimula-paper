"""
H1 / H4 — Analytical decode-FLOP Pareto for the hybrid layer-split sweep.

Reuses notebooks/conservative_arch/inference_efficiency/flop_counter.py
to compute per-token AR decode FLOPs at context length T for:

  - all-attention baseline (k=8, m=0)  -- KV cached, O(T) per-token
  - all-SPLM baseline      (k=0, m=8)  -- streaming-xi, O(1) per-token
  - hybrid (k, m)          k attn + m SPLM, shared embedding/logits

For the hybrid the per-token decode cost is:
  embed + pos
  + sum over k attention-block decode steps         (each O(T))
  + boundary LN
  + sum over m SPLM integration steps               (each O(1))
  + final logits

This is the second prong of the pre-registered title-justification rule
(`the v4 title-justification rule` §6.5):
  decode-FLOP cost at T=1024 must be >= 30% lower than all-attention,
  AT iso-quality (val PPL within +5 PPL of all-attention).

Output: appends a "Decode-FLOP Pareto" section to
  notebooks/conservative_arch/hybrid/results/h1_sweep/H1_RESULTS.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from inference_efficiency.flop_counter import (  # noqa: E402
    AttnFLOPParams,
    SPLMFLOPParams,
    _v_theta_forward_flops,
    _v_theta_grad_flops,
)


def _attn_decode_block_only_flops(p: AttnFLOPParams, T: int) -> int:
    """Per-token AR decode FLOPs for a SINGLE attention block at context T.

    Excludes embedding, final LN, and final logits projection (counted
    separately in the composite).
    """
    head_dim = p.d // p.n_head
    block = 0
    block += 5 * p.d                        # ln1
    block += 2 * p.d * (3 * p.d)            # QKV projection
    block += 2 * p.n_head * 1 * head_dim * T   # Q (new) @ K^T (cached)
    block += 5 * p.n_head * T               # softmax
    block += 2 * p.n_head * T * head_dim    # softmax @ V (cached)
    block += 2 * p.d * p.d                  # output projection
    block += 5 * p.d                        # ln2
    block += 2 * p.d * (p.mlp_mult * p.d)   # MLP fc1
    block += p.mlp_mult * p.d               # GELU
    block += 2 * (p.mlp_mult * p.d) * p.d   # MLP fc2
    return int(block)


def _splm_step_only_flops(p: SPLMFLOPParams) -> int:
    """Per-token AR decode FLOPs for a SINGLE SPLM integration step.

    Streaming-xi: cost is O(1) in T. Excludes embedding, initial LN
    on h0, final logits.
    """
    step = 0
    step += 2 * p.d                         # xi running-sum + divide
    step += _v_theta_forward_flops(p, 1)
    step += _v_theta_grad_flops(p, 1)
    step += 4 * p.d                         # 2nd-order velocity + position update
    if p.ln_after_step:
        step += 5 * p.d
    return int(step)


def hybrid_decode_token_flops(
    d: int,
    k_attn: int,
    m_splm: int,
    n_head: int,
    mlp_mult: int,
    v_hidden: int,
    v_depth: int,
    vocab_size: int,
    ln_after_step: bool,
    T: int,
) -> Dict:
    """Per-token AR decode FLOPs for the two-stage hybrid (k attn + m splm).

    Composition:
      embed + pos
      + k * attn_decode_block_only_flops(T)
      + boundary_LN(d)
      + initial post-attn LN  (only if ln_after_step in SPLM stage; we use LN here)
      + m * splm_step_only_flops()
      + final_logits(d, vocab_size)
    """
    p_attn = AttnFLOPParams(d=d, L=k_attn, n_head=n_head,
                            mlp_mult=mlp_mult, vocab_size=vocab_size)
    p_splm = SPLMFLOPParams(d=d, L=m_splm, v_hidden=v_hidden,
                            v_depth=v_depth, vocab_size=vocab_size,
                            ln_after_step=ln_after_step)

    flops = 0
    flops += d                               # embed lookup + pos add
    flops += k_attn * _attn_decode_block_only_flops(p_attn, T)
    flops += 5 * d                           # boundary LN before SPLM stage
    if ln_after_step and m_splm > 0:
        flops += 5 * d                       # _project on h_k
    flops += m_splm * _splm_step_only_flops(p_splm)
    flops += 2 * d * vocab_size              # final logits
    return {
        "per_token": int(flops),
        "T": T,
        "k_attn": k_attn,
        "m_splm": m_splm,
    }


def make_pareto_table(
    cells: List[Tuple[int, int, float]],
    *,
    d: int = 128,
    n_head: int = 4,
    mlp_mult: int = 4,
    v_hidden: int = 512,
    v_depth: int = 3,
    vocab_size: int = 50257,
    ln_after_step: bool = True,
    T: int = 1024,
) -> List[Dict]:
    """Cells: list of (k_attn, m_splm, val_ppl).  Returns Pareto rows."""
    p_attn_full = AttnFLOPParams(d=d, L=8, n_head=n_head,
                                 mlp_mult=mlp_mult, vocab_size=vocab_size)
    p_splm_full = SPLMFLOPParams(d=d, L=8, v_hidden=v_hidden,
                                 v_depth=v_depth, vocab_size=vocab_size,
                                 ln_after_step=ln_after_step)

    # All-attention reference at T (k=8, m=0): use the full attn decode flop.
    # We can construct it via hybrid_decode_token_flops(k=8, m=0).
    attn_full = hybrid_decode_token_flops(
        d=d, k_attn=8, m_splm=0, n_head=n_head,
        mlp_mult=mlp_mult, v_hidden=v_hidden, v_depth=v_depth,
        vocab_size=vocab_size, ln_after_step=ln_after_step, T=T,
    )["per_token"]
    # All-SPLM reference at T (k=0, m=8)
    splm_full = hybrid_decode_token_flops(
        d=d, k_attn=0, m_splm=8, n_head=n_head,
        mlp_mult=mlp_mult, v_hidden=v_hidden, v_depth=v_depth,
        vocab_size=vocab_size, ln_after_step=ln_after_step, T=T,
    )["per_token"]

    rows = []
    for k, m, ppl in cells:
        f = hybrid_decode_token_flops(
            d=d, k_attn=k, m_splm=m, n_head=n_head,
            mlp_mult=mlp_mult, v_hidden=v_hidden, v_depth=v_depth,
            vocab_size=vocab_size, ln_after_step=ln_after_step, T=T,
        )["per_token"]
        rows.append({
            "k_attn": k, "m_splm": m, "val_ppl": ppl,
            "decode_flops_per_token": f,
            "ratio_vs_attn_full": f / attn_full,
            "reduction_vs_attn_full_pct": (1.0 - f / attn_full) * 100.0,
            "T": T,
        })
    return rows, {"attn_full_per_token": attn_full,
                  "splm_full_per_token": splm_full,
                  "T": T}


def main():
    # H1 sweep results (val PPL @ S=1).
    cells = [
        (2, 6, 147.28),
        (3, 5, 139.29),
        (4, 4, 133.01),
        (5, 3, 136.48),
        (6, 2, 135.08),
    ]

    out_path = (SCRIPT_DIR / "results" / "h1_sweep" / "H1_RESULTS.md")
    if not out_path.exists():
        raise SystemExit(f"missing {out_path}; run aggregate_h1.py first")

    print(f"[h1-flop] composing decode-FLOP Pareto for "
          f"{len(cells)} hybrid cells at T=1024")

    section_lines = ["", "## Decode-FLOP Pareto (analytical, T = 1024)",
                     "",
                     "Per-token AR decode FLOPs at context length 1024 "
                     "(KV-cached attention; streaming-ξ SPLM):",
                     ""]

    for T in (256, 1024, 4096):
        rows, refs = make_pareto_table(cells, T=T)
        section_lines += [
            f"",
            f"### T = {T}",
            f"",
            f"All-attention reference (k=8, m=0): "
            f"**{refs['attn_full_per_token'] / 1e6:.3f} MFLOPs/tok**.",
            f"All-SPLM reference (k=0, m=8): "
            f"**{refs['splm_full_per_token'] / 1e6:.3f} MFLOPs/tok**.",
            f"",
            f"| (k, m) | val PPL (S=1) | decode FLOPs/tok | vs all-attn | reduction |",
            f"|--------|---------------|------------------|-------------|-----------|",
        ]
        for r in rows:
            section_lines.append(
                f"| ({r['k_attn']}, {r['m_splm']}) | "
                f"{r['val_ppl']:.2f} | "
                f"{r['decode_flops_per_token'] / 1e6:.3f} MFLOPs | "
                f"{r['ratio_vs_attn_full']:.3f}× | "
                f"{r['reduction_vs_attn_full_pct']:+.1f}% |"
            )

    section_lines += [
        "",
        "### Pre-registered rule check",
        "",
        "Per `the v4 title-justification rule` §6.5:",
        "**\"Efficient\" is justified iff** some hybrid (k, m) achieves",
        "val PPL within +5 PPL of the all-attention baseline (~150) AND",
        "decode-FLOP cost at T = 1024 is ≥ 30% lower than all-attention,",
        "both at S=3 with sign 3/3.",
        "",
        "At S=1 (this sweep) every cell already passes the *quality* arm",
        "(in fact every cell BEATS all-attention by 2–17 PPL). For the",
        "*FLOP* arm at T=1024 the table above is the relevant comparison.",
        "Cells that satisfy ≥ 30% decode-FLOP reduction at T=1024 simultaneously",
        "with within-+5-PPL-of-all-attn are the candidates that earn",
        "the title word.  Phase 2 (H2) confirms at S=3.",
        "",
        "_(Pareto computed by `notebooks/conservative_arch/hybrid/decode_flop_pareto.py`",
        "from `inference_efficiency/flop_counter.py`.)_",
        "",
    ]

    with out_path.open("a") as f:
        f.write("\n".join(section_lines) + "\n")
    print(f"[h1-flop] appended decode-FLOP Pareto to {out_path}")


if __name__ == "__main__":
    main()
