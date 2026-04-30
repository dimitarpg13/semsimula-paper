r"""Analytical FLOP counters for SPLM and the matched attention baseline.

Both forward / decode FLOP counts follow the conventions used in
the paper Appendix A2:

  - A matrix multiply X (m, n) @ W (n, p) costs 2 * m * n * p FLOPs
    (one multiply + one add per element of the output).
  - An elementwise activation costs O(N) FLOPs (counted as 1 * N).
  - A LayerNorm costs ~5 * N (mean + variance + scale + shift).
  - Softmax costs ~5 * N.

The counts are *per sequence of length T* unless noted otherwise.

Two operating modes are reported:

  1. **Full-prefill forward** (`forward_flops`): the cost of running
     model(input_ids) on a length-T prefix (training-style). Both
     architectures grow O(T) in this mode for the per-token cost; SPLM
     also has a quadratic term from the cumulative-mean recomputation,
     and attention has a quadratic term from the QK^T product.

  2. **Streaming AR decode** (`decode_token_flops`): the cost of
     generating ONE additional token given a length-(T-1) prefix. SPLM
     in streaming-xi mode is O(1) in T (only L * v_hidden * d work per
     token); attention with KV cache is O(T) in T.

Use `crossover_T(d, L, n_head, mlp_mult, v_hidden, v_depth)` to compute
the predicted FLOP-crossover context length T* where SPLM's per-token
streaming cost equals attention's per-token cached cost.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SPLMFLOPParams:
    d: int
    L: int
    v_hidden: int
    v_depth: int
    vocab_size: int = 50257
    ln_after_step: bool = True


@dataclass
class AttnFLOPParams:
    d: int
    L: int
    n_head: int
    mlp_mult: int = 4
    vocab_size: int = 50257


def _v_theta_forward_flops(p: SPLMFLOPParams, n: int) -> int:
    """FLOPs to evaluate V_theta on n input rows of dimension 2d.

    V_theta is an MLP: 2d -> v_hidden -> v_hidden -> ... -> 1
    with v_depth hidden layers (so the architecture is
    Linear(2d, v_h) GELU Linear(v_h, v_h) GELU ... Linear(v_h, 1)).
    """
    flops = 0
    flops += 2 * n * (2 * p.d) * p.v_hidden  # input layer
    flops += n * p.v_hidden  # GELU
    for _ in range(max(p.v_depth - 1, 0)):
        flops += 2 * n * p.v_hidden * p.v_hidden
        flops += n * p.v_hidden
    flops += 2 * n * p.v_hidden * 1  # output layer to scalar
    return flops


def _v_theta_grad_flops(p: SPLMFLOPParams, n: int) -> int:
    """FLOPs to compute -nabla_h V_theta on n rows.

    Backward of V_theta over its 2d -> v_hidden -> ... -> 1 stack,
    keeping only the gradient w.r.t. h (the second half of the input).
    A reasonable approximation is that backward is ~2x the forward cost.
    """
    return 2 * _v_theta_forward_flops(p, n)


def splm_forward_flops(p: SPLMFLOPParams, T: int, B: int = 1) -> dict:
    """Total forward FLOPs for SPLM on a (B, T) input sequence.

    Counts the embedding + integration + output projection. Used as the
    baseline for non-streaming inference (i.e., re-running the full
    forward at every new token).
    """
    n = B * T
    flops = 0
    flops += B * T * p.d  # embedding lookup (table read; no MUL/ADD)
    flops += B * T * p.d  # position add

    if p.ln_after_step:
        flops += 5 * n * p.d  # initial _project

    cumsum_flops = 2 * n * p.d  # running sum + division for xi
    integration_step_flops = (
        cumsum_flops
        + _v_theta_forward_flops(p, n)
        + _v_theta_grad_flops(p, n)
        + 4 * n * p.d
        + (5 * n * p.d if p.ln_after_step else 0)
    )
    flops += p.L * integration_step_flops
    flops += 2 * n * p.d * p.vocab_size  # final projection to logits
    return {
        "total": int(flops),
        "per_token": int(flops / max(B * T, 1)),
        "per_sequence": int(flops / max(B, 1)),
        "T": T,
        "B": B,
    }


def splm_decode_token_flops(p: SPLMFLOPParams, T: int) -> dict:
    """FLOPs for one streaming-xi AR decode step at context length T.

    Only the new token's column is integrated, and xi_t is computed by
    incrementally updating a per-layer running sum. Cost is O(L *
    v_hidden * d), independent of T.
    """
    flops = 0
    flops += p.d  # embed lookup + position add
    if p.ln_after_step:
        flops += 5 * p.d
    integration_step_flops = (
        2 * p.d  # running sum increment + divide
        + _v_theta_forward_flops(p, 1)
        + _v_theta_grad_flops(p, 1)
        + 4 * p.d  # 1st-order or 2nd-order update (small constant)
        + (5 * p.d if p.ln_after_step else 0)
    )
    flops += p.L * integration_step_flops
    flops += 2 * p.d * p.vocab_size  # final projection
    return {
        "per_token": int(flops),
        "T": T,
    }


def attn_forward_flops(p: AttnFLOPParams, T: int, B: int = 1) -> dict:
    """Forward FLOPs for the matched-attention baseline on a (B, T) input.

    Counts the embedding, per-block (LN, QKV proj, attention, output proj,
    LN, MLP), and final logits projection.
    """
    n = B * T
    flops = 0
    flops += B * T * p.d  # embedding
    flops += B * T * p.d  # position add

    head_dim = p.d // p.n_head
    per_block = 0
    per_block += 5 * n * p.d  # ln1
    per_block += 2 * n * p.d * (3 * p.d)  # QKV projection
    per_block += 2 * B * p.n_head * T * head_dim * T  # QK^T
    per_block += 5 * B * p.n_head * T * T  # softmax
    per_block += 2 * B * p.n_head * T * T * head_dim  # softmax @ V
    per_block += 2 * n * p.d * p.d  # output projection
    per_block += 5 * n * p.d  # ln2
    per_block += 2 * n * p.d * (p.mlp_mult * p.d)  # mlp fc1
    per_block += n * p.mlp_mult * p.d  # gelu
    per_block += 2 * n * (p.mlp_mult * p.d) * p.d  # mlp fc2

    flops += p.L * per_block
    flops += 5 * n * p.d  # final ln
    flops += 2 * n * p.d * p.vocab_size  # logits
    return {
        "total": int(flops),
        "per_token": int(flops / max(B * T, 1)),
        "per_sequence": int(flops / max(B, 1)),
        "T": T,
        "B": B,
    }


def attn_decode_token_flops(p: AttnFLOPParams, T: int) -> dict:
    """KV-cached AR decode FLOPs to generate ONE token at context length T.

    The new token is queried against T cached keys/values (after this
    token, T+1 cached entries). Cost grows ~linear in T from the
    QK^T and softmax-V terms.
    """
    head_dim = p.d // p.n_head
    flops = 0
    flops += p.d  # embed + position
    per_block = 0
    per_block += 5 * p.d
    per_block += 2 * p.d * (3 * p.d)
    per_block += 2 * p.n_head * 1 * head_dim * T  # Q (new) @ K^T (cached, length T)
    per_block += 5 * p.n_head * T  # softmax
    per_block += 2 * p.n_head * T * head_dim  # softmax @ V (cached)
    per_block += 2 * p.d * p.d
    per_block += 5 * p.d
    per_block += 2 * p.d * (p.mlp_mult * p.d)
    per_block += p.mlp_mult * p.d
    per_block += 2 * (p.mlp_mult * p.d) * p.d
    flops += p.L * per_block
    flops += 5 * p.d
    flops += 2 * p.d * p.vocab_size
    return {
        "per_token": int(flops),
        "T": T,
    }


def splm_decode_full_token_flops(p: SPLMFLOPParams, T: int) -> dict:
    """Non-streaming SPLM AR decode: re-run the full model(prefix) per token.

    Used as a quality-baseline against streaming-xi (which is approximate
    -- see splm_streaming_decode.py docstring). FLOPs scale ~linearly
    in T per generated token.
    """
    full = splm_forward_flops(p, T, B=1)
    return {
        "per_token": full["total"],
        "T": T,
    }


def crossover_T(p_splm: SPLMFLOPParams, p_attn: AttnFLOPParams) -> dict:
    """Find the smallest integer T where SPLM streaming per-token FLOPs
    are <= attention KV-cached per-token FLOPs.
    """
    T = 1
    while T < 1_000_000:
        s = splm_decode_token_flops(p_splm, T)["per_token"]
        a = attn_decode_token_flops(p_attn, T)["per_token"]
        if s <= a:
            return {
                "T_crossover": T,
                "splm_streaming_per_token_flops_at_T": int(s),
                "attn_kv_per_token_flops_at_T": int(a),
            }
        T = max(T + 1, int(T * 1.5))
    return {
        "T_crossover": None,
        "note": "no crossover up to T=1e6; SPLM streaming dominates throughout",
    }


def _smoke_test():
    """Sanity prints for the matched config used in the SPLM-1 ablation
    arm B vs the matched attention baseline (E8 Phase 1)."""
    p_splm = SPLMFLOPParams(d=128, L=8, v_hidden=512, v_depth=3,
                            vocab_size=50257, ln_after_step=True)
    p_attn = AttnFLOPParams(d=128, L=8, n_head=4, mlp_mult=4,
                            vocab_size=50257)

    print("=== Forward (training-style) per token ===")
    for T in [16, 64, 128, 256, 512, 1024, 2048]:
        s = splm_forward_flops(p_splm, T)["per_token"]
        a = attn_forward_flops(p_attn, T)["per_token"]
        print(f"  T={T:5d}  SPLM={s/1e6:8.2f} MFLOPs/tok  "
              f"ATTN={a/1e6:8.2f} MFLOPs/tok  "
              f"ratio (SPLM/ATTN)={s/a:.2f}")

    print("\n=== AR decode per generated token ===")
    print(f"  context  SPLM_stream  SPLM_full  ATTN_kv  cross_at")
    for T in [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        s_str = splm_decode_token_flops(p_splm, T)["per_token"]
        s_full = splm_decode_full_token_flops(p_splm, T)["per_token"]
        a = attn_decode_token_flops(p_attn, T)["per_token"]
        marker = " <-- crossover" if s_str < a else ""
        print(f"  T={T:5d}  {s_str/1e6:8.2f} M  "
              f"{s_full/1e6:8.2f} M  {a/1e6:8.2f} M{marker}")

    cross = crossover_T(p_splm, p_attn)
    print(f"\n=== Predicted crossover ===")
    print(f"  T*={cross.get('T_crossover')}")
    if cross.get("T_crossover") is not None:
        print(f"  at T*: SPLM stream = {cross['splm_streaming_per_token_flops_at_T']/1e6:.2f} MF, "
              f"ATTN kv = {cross['attn_kv_per_token_flops_at_T']/1e6:.2f} MF")


if __name__ == "__main__":
    _smoke_test()
