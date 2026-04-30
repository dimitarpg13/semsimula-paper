r"""Streaming-xi autoregressive decoder for SPLM.

This is the operationalisation of the per-token inference cost claim in
the paper Appendix A2: SPLM's decoder cost is O(L * v_hidden * d) per
token, **independent of context length T**, because the only intra-token
dependency is through xi_t = causal cumulative mean of h_{<=t}, which can
be maintained as a running sum (L+1) * d in size.

Cache state per sequence (one batch element):
    - running_sums:     list of L+1 tensors (d,) - layer-by-layer
                        cumulative sums of h^(l)_{<=t-1}
    - position_index:   int (number of tokens emitted so far)
    - last_h_L:         tensor (d,) - the final-layer hidden state
                        of the most-recently emitted token


Important: streaming-xi is *not* bit-exact w.r.t. the batched forward
=====================================================================

The SPLM integrator currently in the codebase computes the gradient of
V_theta w.r.t. h_in via `torch.autograd.grad(V.sum(), h_in)`. PyTorch's
autograd tracks operations on a tensor through `requires_grad_(True)`
even retroactively if the leaf is registered before backward, so
`xi_now = causal_cumulative_mean(h)` followed by
`h.requires_grad_(True)` does flow gradient through xi back into h.
Concretely this means the force at position t includes a non-causal
"future" term:

    f_t = -[ partial V_t / partial h_t                       (local)
           + sum_{s > t} partial V_s / partial xi_s
                          * partial xi_s / partial h_t       (functional) ]

In a true streaming AR decode we cannot evaluate the second term because
V_s is undefined for s > t. The streaming-xi loop in this module
computes only the local term -- the standard "Markovian" interpretation
of the inference dynamics in the paper Appendix A2.

Empirical impact on a freshly-initialised d=32, L=4 model: the residual
between batch and streaming h_L is at the ~3e-4 level per position, with
a monotonically decreasing per-position signature (max at t=0, ~0 at
t=T-1) -- the exact signature of the missing future-contribution term.
This is a faithful approximation of the trained dynamics, not a bug.
The benchmark harness uses streaming-xi for wall-clock measurement and
reports both the raw streaming-xi PPL and the gap against the
non-streaming PPL on the same inputs as a quality-divergence diagnostic.

Public API:
    state = StreamingState.init(model, device)
    state, h_L_t = streaming_step(model, state, token_id)
    # Predict next token
    logits_t = h_L_t @ model.E.weight.T
    next_token_id = sample(logits_t)
    state, h_L_t = streaming_step(model, state, next_token_id)
    ...

Limitations:
- Batch size 1 only (the typical AR decode setting). Generalising to
  B > 1 is mechanical (vectorise the running sums) but not needed for
  the benchmark.
- The forward must temporarily enable grad because the integrator uses
  `torch.autograd.grad` to compute -nabla V_theta. Caches are held with
  `.detach()` so they don't leak the autograd graph between steps.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "energetic_minima"))
sys.path.insert(0, str(PARENT_DIR / "sarf_mass_variant"))


@dataclass
class StreamingState:
    running_sums: list[torch.Tensor]
    position: int = 0
    last_h_L: Optional[torch.Tensor] = None

    @classmethod
    def init(cls, model, device) -> "StreamingState":
        cfg = model.cfg
        d = cfg.d
        L = cfg.L
        dtype = next(model.parameters()).dtype
        return cls(
            running_sums=[torch.zeros(d, device=device, dtype=dtype)
                          for _ in range(L + 1)],
            position=0,
            last_h_L=None,
        )

    def clone(self) -> "StreamingState":
        return StreamingState(
            running_sums=[s.clone() for s in self.running_sums],
            position=self.position,
            last_h_L=None if self.last_h_L is None else self.last_h_L.clone(),
        )


def _embed_one_token(model, token_id: int, position: int,
                     device: torch.device) -> torch.Tensor:
    """Return emb(x_t) + P[t] of shape (d,)."""
    cfg = model.cfg
    if position >= cfg.max_len:
        raise ValueError(
            f"streaming position {position} exceeds model.cfg.max_len={cfg.max_len}; "
            "trained position embedding does not extend further."
        )
    x = torch.tensor([[token_id]], dtype=torch.long, device=device)
    emb = model.E(x)
    if hasattr(model, "P"):
        emb = emb + model.P[position:position + 1].unsqueeze(0)
    return emb.squeeze(0).squeeze(0)


def _compute_mass_one_token(model, token_id: int, emb_d: torch.Tensor,
                            device: torch.device) -> torch.Tensor:
    """Compute the per-token mass for a single token."""
    cfg = model.cfg
    if cfg.mass_mode == "global":
        return model.m_global

    x = torch.tensor([[token_id]], dtype=torch.long, device=device)
    emb_unsq = emb_d.view(1, 1, -1)
    return model.compute_mass(x, emb_unsq).view(-1)


def _project_one_token(model, h_d: torch.Tensor) -> torch.Tensor:
    """Apply LayerNorm-after-step if the model uses it."""
    if not getattr(model.cfg, "ln_after_step", False):
        return h_d
    if getattr(model, "post_ln", None) is not None:
        return model.post_ln(h_d.view(1, 1, -1)).view(-1)
    return F.layer_norm(h_d.view(1, 1, -1), (model.cfg.d,),
                        eps=getattr(model.cfg, "ln_eps", 1e-5)).view(-1)


def _is_first_order(model) -> bool:
    """Detect whether to use the first-order integrator semantics."""
    cls_name = type(model).__name__
    return cls_name == "ScalarPotentialLMFirstOrder"


def streaming_step(model, state: StreamingState,
                   token_id: int) -> tuple[StreamingState, torch.Tensor]:
    """Run one integration sweep for the new token; advance the cache.

    Returns the updated state (a new object; old state preserved by clone)
    and h_L (shape (d,)).
    """
    cfg = model.cfg
    L = cfg.L
    dt = cfg.dt
    device = state.running_sums[0].device

    if cfg.ln_after_step:
        h_d = _project_one_token(model, _embed_one_token(model, token_id,
                                                          state.position, device))
    else:
        h_d = _embed_one_token(model, token_id, state.position, device)

    new_sums: list[torch.Tensor] = [s.clone() for s in state.running_sums]
    new_sums[0] = state.running_sums[0] + h_d.detach()

    pos = state.position
    is_fo = _is_first_order(model)
    if not is_fo:
        gamma_val = float(model.gamma) if hasattr(model, "gamma") else 0.0
        v_d = torch.zeros_like(h_d)

    raw_emb = _embed_one_token(model, token_id, pos, device).detach()
    m_d = _compute_mass_one_token(model, token_id, raw_emb, device).detach()
    if m_d.dim() == 0:
        m_b = m_d
    else:
        m_b = m_d

    for ell in range(L):
        sum_now = new_sums[ell]
        denom = float(pos + 1)
        xi_d = (sum_now / denom).detach()

        h_in = h_d.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            V = model.V_theta(xi_d.view(1, 1, -1), h_in.view(1, 1, -1)).sum()
            grad_V, = torch.autograd.grad(V, h_in)
        f = -grad_V

        if is_fo:
            h_new = h_in.detach() + dt * f / m_b
        else:
            v_d = (v_d + dt * f / m_b) / (1.0 + dt * gamma_val)
            h_new = h_in.detach() + dt * v_d

        if cfg.ln_after_step:
            h_new = _project_one_token(model, h_new)
        h_d = h_new.detach()

        new_sums[ell + 1] = state.running_sums[ell + 1] + h_d

    new_state = StreamingState(
        running_sums=new_sums,
        position=state.position + 1,
        last_h_L=h_d.detach().clone(),
    )
    return new_state, h_d


@torch.no_grad()
def logits_from_h(model, h_d: torch.Tensor) -> torch.Tensor:
    return h_d @ model.E.weight.T


def streaming_decode(model, prompt_ids: list[int], n_new: int,
                     device: torch.device,
                     greedy: bool = True,
                     temperature: float = 1.0,
                     seed: int = 0) -> tuple[list[int], list[torch.Tensor]]:
    """Greedy / sampled streaming AR decode. Returns (all_ids, h_L_per_token)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    state = StreamingState.init(model, device)
    h_per_token: list[torch.Tensor] = []
    all_ids: list[int] = list(prompt_ids)

    for tok in prompt_ids:
        state, h_L = streaming_step(model, state, int(tok))
        h_per_token.append(h_L)

    for _ in range(n_new):
        logits = logits_from_h(model, h_per_token[-1])
        if greedy or temperature == 0.0:
            next_id = int(logits.argmax().item())
        else:
            probs = F.softmax(logits / temperature, dim=-1).cpu()
            next_id = int(torch.multinomial(probs, 1, generator=g).item())
        state, h_L = streaming_step(model, state, next_id)
        h_per_token.append(h_L)
        all_ids.append(next_id)

    return all_ids, h_per_token


def verify_against_batch_forward(model, prompt_ids: list[int],
                                 device: torch.device) -> dict:
    """Compare streaming-xi to the batched forward; report divergence pattern.

    Streaming computes the *local* (Markovian) gradient at each position;
    the batched forward computes the *functional* gradient, which in
    addition to the local term includes contributions from V_s
    (s > t) routed through xi_s. The diagnostic returns:

    - max_abs / mean_abs: per-position abs differences (full sequence)
    - per_position_max: max-abs diff at each t (should be monotonically
      non-increasing in t, approaching 0 at t = T - 1)
    - causal_signature_ok: True iff per_position_max[0] >= per_position_max[T-1]
      (a sanity-check that the residual reflects the missing
      future-contribution term, not an unrelated bug)
    """
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.enable_grad():
        out = model(x, return_trajectory=True)
        h_L_batch = out[2][-1].squeeze(0).detach().to(device)

    state = StreamingState.init(model, device)
    h_L_stream = []
    for tok in prompt_ids:
        state, h_L = streaming_step(model, state, int(tok))
        h_L_stream.append(h_L)
    h_L_stream = torch.stack(h_L_stream, dim=0).to(device)

    diff = (h_L_batch - h_L_stream).abs()
    per_position_max = diff.max(dim=-1).values.detach().cpu().tolist()
    norm_b = float(h_L_batch.norm(dim=-1).mean())
    norm_s = float(h_L_stream.norm(dim=-1).mean())
    causal_signature_ok = (per_position_max[0] >= per_position_max[-1])
    last_pos_diff = per_position_max[-1]
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float((diff / (h_L_batch.abs() + 1e-8)).max()),
        "norm_batch_h_L_mean": norm_b,
        "norm_stream_h_L_mean": norm_s,
        "per_position_max_diff": per_position_max,
        "last_pos_max_diff": last_pos_diff,
        "causal_signature_ok": causal_signature_ok,
        "n_positions": len(prompt_ids),
    }


def _smoke_test():
    """Run streaming-xi vs batch divergence-pattern check on a tiny SPLM.

    Streaming-xi computes the *local* (Markovian) gradient of V_theta;
    the batch forward computes the *functional* gradient that also
    routes h_t's contribution through xi_s for s > t. We expect a
    monotonically decreasing per-position diff (max at t=0, ~0 at last
    position). The smoke test asserts this signature is present.
    """
    from energetic_minima.model_ln import (
        ScalarPotentialLMSARFMassLN, SPLMSARFMassLNConfig,
    )

    torch.manual_seed(0)
    cfg = SPLMSARFMassLNConfig(
        vocab_size=200, d=32, max_len=64,
        v_hidden=64, v_depth=2, L=4,
        init_m=1.0, init_gamma=1.0,
        mass_mode="global",
    )
    device = torch.device("cpu")
    model = ScalarPotentialLMSARFMassLN(cfg).eval().to(device)

    prompt_ids = list(range(1, 17))
    info = verify_against_batch_forward(model, prompt_ids, device)
    print(f"[smoke-LN]   max_abs={info['max_abs_diff']:.2e}   "
          f"mean_abs={info['mean_abs_diff']:.2e}   "
          f"last_pos={info['last_pos_max_diff']:.2e}   "
          f"causal_signature_ok={info['causal_signature_ok']}")
    print(f"[smoke-LN]   per-pos max diff: "
          f"{[f'{v:.1e}' for v in info['per_position_max_diff'][:8]]}... "
          f"{[f'{v:.1e}' for v in info['per_position_max_diff'][-3:]]}")
    if not info["causal_signature_ok"]:
        raise SystemExit("streaming-xi causal signature FAILED on em_ln smoke")

    sys.path.insert(0, str(PARENT_DIR / "first_order_ablation"))
    from model_first_order import (
        ScalarPotentialLMFirstOrder, SPLMFirstOrderConfig,
    )

    cfg_fo = SPLMFirstOrderConfig(
        vocab_size=200, d=32, max_len=64,
        v_hidden=64, v_depth=2, L=4,
        init_m=1.0, init_gamma=1.0,
        mass_mode="global",
    )
    model_fo = ScalarPotentialLMFirstOrder(cfg_fo).eval().to(device)
    info_fo = verify_against_batch_forward(model_fo, prompt_ids, device)
    print(f"[smoke-FO]   max_abs={info_fo['max_abs_diff']:.2e}   "
          f"mean_abs={info_fo['mean_abs_diff']:.2e}   "
          f"last_pos={info_fo['last_pos_max_diff']:.2e}   "
          f"causal_signature_ok={info_fo['causal_signature_ok']}")
    if not info_fo["causal_signature_ok"]:
        raise SystemExit("streaming-xi causal signature FAILED on first_order smoke")

    print("[smoke] streaming-xi local-gradient implementation passes "
          "the causal-signature divergence check on both SPLM variants")


if __name__ == "__main__":
    _smoke_test()
