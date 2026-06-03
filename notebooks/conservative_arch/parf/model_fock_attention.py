"""
FockAttentionPARFLM — literal §5.1 Feynman diagram as a non-conservative force.

Implements the Section 5.1 "Attention as Virtual Particle Exchange" Feynman
diagram directly: each token j emits a virtual photon carrying key k_j and
payload v_j; each token i absorbs with query q_i.  The coupling is
α_ij = softmax_j(q_i · k_j / √d_k).  The exchange force on token i is
F_i = Σ_j α_ij · v_j.

This is the λ=0 (instantaneous exchange) limit — no registers, no persistence,
no creation/destruction gates.  The force is injected post-Verlet using the
same pattern as FockPARFLM_v2's reverse channel:

    h_new += (dt² / m_b) · tanh(scale) · F_exchange

The model extends MultiXiPARFLM so it inherits the full K-EMA ξ + sparse PARF
dynamics.  The exchange force is the only non-conservative addition.

Inheritance
-----------
    FockAttentionPARFLM  →  MultiXiPARFLM  →  SparsePARFLM  →  PARFLM
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

_THIS_DIR = Path(__file__).parent
_PARENT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(_PARENT_DIR / "multixi"))

from model_parf_multixi import (  # noqa: E402
    MultiXiPARFConfig,
    MultiXiPARFLM,
)


# ---------------------------------------------------------------------------
# Direct token-to-token exchange force (literal §5.1 Feynman diagram)
# ---------------------------------------------------------------------------
class DirectExchangeForce(nn.Module):
    """Literal §5.1: direct token-to-token virtual photon exchange.

    Token j emits (key k_j = W_K h_j, payload v_j = W_V h_j).
    Token i absorbs (query q_i = W_Q h_i).
    Coupling α_ij = softmax_j(q_i · k_j / √d_k).
    Force on token i: F_i = Σ_j α_ij · v_j.

    Supports multi-head exchange: each head operates on a d_k-dim subspace.
    """

    def __init__(
        self,
        d: int,
        d_k: int,
        n_heads: int = 1,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.d = d
        self.d_k = d_k
        self.n_heads = n_heads
        total_k = d_k * n_heads

        self.W_Q = nn.Linear(d, total_k, bias=False)
        self.W_K = nn.Linear(d, total_k, bias=False)
        self.W_V = nn.Linear(d, total_k, bias=False)

        nn.init.normal_(self.W_Q.weight, std=init_scale)
        nn.init.normal_(self.W_K.weight, std=init_scale)
        nn.init.normal_(self.W_V.weight, std=init_scale)

        if n_heads > 1:
            self.W_O = nn.Linear(total_k, d, bias=False)
            nn.init.normal_(self.W_O.weight, std=init_scale)
        else:
            self.W_O = nn.Linear(d_k, d, bias=False)
            nn.init.normal_(self.W_O.weight, std=init_scale)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute the exchange force on each token.

        Args:
            h: (B, T, d) — token hidden states.

        Returns:
            F_exchange: (B, T, d) — non-conservative exchange force.
        """
        B, T, d = h.shape
        nh, dk = self.n_heads, self.d_k

        q = self.W_Q(h)  # (B, T, nh*dk)
        k = self.W_K(h)  # (B, T, nh*dk)
        v = self.W_V(h)  # (B, T, nh*dk)

        # Reshape for multi-head: (B, nh, T, dk)
        q = q.view(B, T, nh, dk).transpose(1, 2)
        k = k.view(B, T, nh, dk).transpose(1, 2)
        v = v.view(B, T, nh, dk).transpose(1, 2)

        # Scaled dot-product scores: (B, nh, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (dk ** 0.5)

        # Causal mask: token i can only absorb from j <= i
        causal = torch.tril(
            torch.ones(T, T, device=h.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e9)

        alpha = F.softmax(scores, dim=-1)  # (B, nh, T, T)

        # Exchange: weighted sum of values
        out = torch.matmul(alpha, v)  # (B, nh, T, dk)

        # Reshape back: (B, T, nh*dk)
        out = out.transpose(1, 2).contiguous().view(B, T, nh * dk)

        # Output projection → (B, T, d)
        return self.W_O(out)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class FockAttentionConfig(MultiXiPARFConfig):
    """MultiXi PARF config extended with direct exchange force parameters.

    Exchange-specific knobs:

      exchange_n_heads      : int — Number of attention heads in the exchange.
      exchange_d_k          : int — Per-head key/query dimension.
      exchange_scale_init   : float — Initial value of the learnable gate
                              (0.0 → force starts off, model learns to open).
      exchange_init_scale   : float — Weight initialisation std.
    """
    exchange_n_heads: int = 1
    exchange_d_k: int = 64
    exchange_scale_init: float = 0.0
    exchange_init_scale: float = 0.02


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FockAttentionPARFLM(MultiXiPARFLM):
    """MultiXi PARFLM + direct token-to-token exchange force (§5.1 Feynman).

    Overrides _stack_forward to inject a non-conservative exchange force
    after each conservative Verlet layer step.  No registers, no gates,
    no creation/destruction — just the literal Feynman diagram.
    """

    cfg: FockAttentionConfig

    def __init__(self, cfg: FockAttentionConfig):
        if not isinstance(cfg, FockAttentionConfig):
            raise TypeError(
                f"FockAttentionPARFLM requires a FockAttentionConfig, "
                f"got {type(cfg)!r}."
            )
        super().__init__(cfg)

        self.exchange_force = DirectExchangeForce(
            d=cfg.d,
            d_k=cfg.exchange_d_k,
            n_heads=cfg.exchange_n_heads,
            init_scale=cfg.exchange_init_scale,
        )

        # Learnable gate initialised to exchange_scale_init (default 0).
        # tanh(0) = 0 → force is off at init; model learns to open it.
        self.exchange_scale = nn.Parameter(
            torch.tensor(float(cfg.exchange_scale_init))
        )

    # ------------------------------------------------------------------
    def _baseline_layer_step(
        self,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        m_b: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One layer step: conservative Verlet + post-Verlet exchange force.

        Returns (h_new, h_for_prev).
        """
        # Conservative dynamics (MultiXi PARF Verlet step)
        h_new = super()._layer_step(
            h, h_prev, m_b, gamma, dt, layer_idx=layer_idx,
        )

        # Non-conservative exchange force (§5.1 Feynman diagram)
        F_ex = self.exchange_force(h_new)
        scale = torch.tanh(self.exchange_scale)
        denom = 1.0 + dt * gamma
        h_new = h_new + (dt * dt / (m_b * denom)) * scale * F_ex

        if self.cfg.ln_after_step:
            h_new = self._project(h_new)

        return h_new, h

    # ------------------------------------------------------------------
    def _stack_forward(
        self,
        h0: torch.Tensor,
        x: torch.Tensor,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Walk L layers with conservative dynamics + exchange force."""
        cfg = self.cfg
        B, T, d = h0.shape
        gamma, dt = self.gamma, cfg.dt
        m_b = self.compute_mass(x)

        h = h0
        h_prev = h0

        traj: Optional[List[torch.Tensor]] = None
        if return_trajectory:
            traj = [h.detach().cpu()]

        for ell in range(cfg.L):
            if cfg.use_layer_checkpoint and self.training:
                def _ckpt_step(
                    _h, _h_prev, _m_b, _gamma,
                    _dt=dt, _ell=ell,
                ):
                    return self._baseline_layer_step(
                        _h, _h_prev, _m_b, _gamma, _dt, _ell,
                    )

                h_new, h_prev_out = torch.utils.checkpoint.checkpoint(
                    _ckpt_step,
                    h, h_prev, m_b, gamma,
                    use_reentrant=False,
                )
            else:
                h_new, h_prev_out = self._baseline_layer_step(
                    h, h_prev, m_b, gamma, dt, layer_idx=ell,
                )

            h_prev = h_prev_out
            h = h_new
            if traj is not None:
                traj.append(h.detach().cpu())

        return h, traj

    # ------------------------------------------------------------------
    @torch.no_grad()
    def exchange_diagnostics(self) -> dict:
        """Diagnostic scalars for logging."""
        return {
            "exchange_scale": torch.tanh(self.exchange_scale).item(),
        }

    # ------------------------------------------------------------------
    def get_exchange_overhead(self) -> int:
        """Count parameters specific to the exchange force augmentation."""
        overhead = sum(
            p.numel() for p in self.exchange_force.parameters()
        )
        overhead += self.exchange_scale.numel()
        return overhead


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke():
    """Minimal forward+backward for single-head and multi-head modes."""
    for n_heads in (1, 4):
        for layer_ckpt in (False, True):
            for gathered in (False, True):
                tag_parts = [f"h{n_heads}"]
                if layer_ckpt:
                    tag_parts.append("lc")
                if gathered:
                    tag_parts.append("gv")
                tag = "+".join(tag_parts)

                d_k = 64 if n_heads == 1 else 16
                cfg = FockAttentionConfig(
                    vocab_size=257, d=16, max_len=64, L=4,
                    v_hidden=32, v_depth=2,
                    v_phi_d_type=4, v_phi_d_angle=2,
                    v_phi_phi_hidden=8, v_phi_theta_hidden=8,
                    v_phi_mlp_hidden=16,
                    mass_mode="global",
                    top_k=4,
                    score_head_hidden=8,
                    xi_channels=4,
                    xi_alpha_inits=[0.0, 0.5, 0.9, 0.99],
                    xi_learnable=True,
                    use_layer_checkpoint=layer_ckpt,
                    use_gathered_v_phi=gathered,
                    exchange_n_heads=n_heads,
                    exchange_d_k=d_k,
                    exchange_scale_init=0.0,
                )
                torch.manual_seed(0)
                net = FockAttentionPARFLM(cfg)
                total = sum(p.numel() for p in net.parameters())
                ex_oh = net.get_exchange_overhead()
                alpha_str = ", ".join(
                    f"{a:.3f}" for a in net.xi_alpha_values()
                )
                print(
                    f"[fock-attention-smoke/{tag}] "
                    f"params={total:,}  exchange_oh={ex_oh:,} "
                    f"({100*ex_oh/total:.1f}%)  "
                    f"K={cfg.xi_channels} \u03b1=[{alpha_str}]  "
                    f"n_heads={n_heads}  d_k={d_k}"
                )

                x = torch.randint(0, cfg.vocab_size, (2, 12))
                y = torch.randint(0, cfg.vocab_size, (2, 12))

                net.train()
                logits, loss = net(x, targets=y)
                print(
                    f"[fock-attention-smoke/{tag}] "
                    f"forward: logits {tuple(logits.shape)} "
                    f"loss {loss.item():.4f}"
                )
                loss.backward()

                alpha_grad = net.xi_module.raw_alpha.grad
                assert alpha_grad is not None, "raw_alpha got no gradient"

                ex_scale_grad = net.exchange_scale.grad
                assert ex_scale_grad is not None, "exchange_scale got no gradient"
                print(
                    f"[fock-attention-smoke/{tag}] "
                    f"raw_\u03b1 grad norm: {alpha_grad.norm().item():.3e}  "
                    f"exchange_scale grad: {ex_scale_grad.item():.3e}  "
                    f"backward OK."
                )
                net.zero_grad()

    print("\n\u2713 All FockAttention smoke tests passed.")


if __name__ == "__main__":
    _smoke()
