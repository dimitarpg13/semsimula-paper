"""
Multi-channel ξ PARF-augmented SPLM — K-EMA × sparse PARF hybrid.

This module combines the two strongest SPLM extensions:

  1. **Multi-channel K-EMA ξ** from `multixi/model_multixi.py`:
     replaces the rank-1 causal cumulative mean with K learnable
     exponential moving averages at multiple decay scales, giving
     V_θ a multi-resolution summary of the past.

  2. **Sparse PARF pair-interactions** from `model_parf_sparse.py`:
     the Gumbel-softmax top-k pair routing that adds V_φ(h_t, h_s)
     particle-exchange forces on top of V_θ.

Architecture (per layer)
------------------------

    ξ^{(k)}_t  =  Σ_{s ≤ t} W_k[t, s] · h_s       (K causal EMAs, learnable α_k)
    V_θ       :  ℝ^{(K+1)·d} → ℝ                   (wide MLP on [ξ_1..ξ_K, h])
    V_φ       :  ℝ^d × ℝ^d → ℝ                     (unchanged structural/competitive pair potential)
    U_t       =  V_θ(ξ_t, h_t)  +  Σ_{s<t} ~m_{ts} · V_φ(h_t, h_s)
    f_t       =  -∇_{h_t} U_t
    h_new     =  velocity-Verlet(h, f, m, γ, dt)

The only change vs SparsePARFLM is that `causal_cumulative_mean` is
replaced by `MultiChannelXi` and V_theta is widened from 2d→1 to
(K+1)d→1.  Everything else — V_φ, score head, sparse routing,
mass model, LN-after-step, causal detach — is inherited unchanged.

Inheritance
-----------
    MultiXiPARFLM  →  SparsePARFLM  →  PARFLM

The model works with any V_φ variant (structural, competitive,
MLP) and all P8 patches (LN-before-distance, per-layer scale,
softsign, bilinear Θ).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).parent
_PARENT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(_PARENT_DIR / "multixi"))

from model_parf_sparse import (  # noqa: E402
    SparsePARFConfig,
    SparsePARFLM,
    _has_analytical_grad,
)
from model_multixi import (  # noqa: E402
    MultiChannelXi,
    ScalarPotentialMultiXi,
    log_spaced_alpha_inits,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class MultiXiPARFConfig(SparsePARFConfig):
    """Sparse PARF config extended with multi-channel K-EMA ξ parameters.

    Defaults give a 4-channel hand-picked multi-resolution past
    (matching the R6.h.0 K-EMA pilot):
      α₁ = 0.0   → ξ^(1) = h_t      (no past)
      α₂ = 0.5   → effective horizon ~2 tokens
      α₃ = 0.9   → effective horizon ~10 tokens
      α₄ = 0.99  → effective horizon ~100 tokens
    """
    xi_channels: int = 4
    xi_alpha_inits: List[float] = field(
        default_factory=lambda: [0.0, 0.5, 0.9, 0.99]
    )
    xi_learnable: bool = True
    xi_alpha_init_mode: str = "explicit"   # "explicit" | "log_spaced"
    xi_tau_max: float = 100.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MultiXiPARFLM(SparsePARFLM):
    """Sparse PARF with multi-channel K-EMA ξ replacing causal cumulative mean.

    Overrides two components of SparsePARFLM:
      1. V_theta: ScalarPotential(2d → 1) → ScalarPotentialMultiXi((K+1)d → 1)
      2. _layer_step: causal_cumulative_mean → MultiChannelXi

    All other layers (V_phi, score head, mass, LN, etc.) are inherited.
    """

    cfg: MultiXiPARFConfig

    def __init__(self, cfg: MultiXiPARFConfig):
        if not isinstance(cfg, MultiXiPARFConfig):
            raise TypeError(
                f"MultiXiPARFLM requires a MultiXiPARFConfig, "
                f"got {type(cfg)!r}."
            )
        # Resolve α-init before super().__init__ so the config is
        # fully populated when the parent stores it.
        if cfg.xi_alpha_init_mode == "log_spaced":
            alpha_inits = log_spaced_alpha_inits(
                cfg.xi_channels, cfg.xi_tau_max,
            )
            cfg.xi_alpha_inits = alpha_inits
        elif cfg.xi_alpha_init_mode == "explicit":
            alpha_inits = cfg.xi_alpha_inits
            if len(alpha_inits) != cfg.xi_channels:
                raise ValueError(
                    f"len(xi_alpha_inits)={len(alpha_inits)} != "
                    f"xi_channels={cfg.xi_channels}"
                )
        else:
            raise ValueError(
                f"unknown xi_alpha_init_mode={cfg.xi_alpha_init_mode!r} "
                "(expected 'explicit' or 'log_spaced')"
            )

        super().__init__(cfg)

        # ── Replace V_theta with the multi-xi version ──
        self.V_theta = ScalarPotentialMultiXi(
            d=cfg.d,
            hidden=cfg.v_hidden,
            depth=cfg.v_depth,
            K=cfg.xi_channels,
        )

        # ── K causal-EMA channels ──
        self.xi_module = MultiChannelXi(
            K=cfg.xi_channels,
            max_len=cfg.max_len,
            alpha_inits=alpha_inits,
            learnable=cfg.xi_learnable,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def xi_alpha_values(self) -> List[float]:
        """Current α_k values (diagnostic)."""
        return [float(a) for a in self.xi_module.alpha.detach().cpu().tolist()]

    # ------------------------------------------------------------------
    def _layer_step(
        self,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        m_b: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """One velocity-Verlet step with K-EMA ξ + sparse PARF routing.

        Identical to SparsePARFLM._layer_step except:
          - causal_cumulative_mean(xi_input) → self.xi_module(xi_input)
          - V_theta(xi_now, h_in) → V_theta(xis, h_in) with xis: (B, T, K, d)
        """
        cfg = self.cfg
        B, T, d = h.shape
        delta = h - h_prev

        # ── Multi-channel ξ (replaces causal_cumulative_mean) ──
        xi_input = h.detach() if cfg.causal_force else h
        xis = self.xi_module(xi_input)                           # (B, T, K, d)

        h_in = h
        if not h_in.requires_grad:
            h_in = h_in.requires_grad_(True)

        h_src = h_in.detach() if cfg.causal_force else h_in
        h_src_for_score = (
            h_in.detach() if cfg.score_head_use_detached_h_src else h_in
        )

        # ── V_theta on multi-channel ξ ──
        V_th_per_token = self.V_theta(xis, h_in)                # (B, T, 1)

        # ── Score head → routing ──
        pi = self.score_head(h_in, h_src_for_score)              # (B, T, T)
        causal = self._pair_mask_for(T, h_in.device)

        # ── Pair potential — Stage-1.5b (gathered) or Stage-1.5a (dense) ──
        if cfg.use_gathered_v_phi:
            idx, m_g = self._sparse_topk_indices(pi, causal, T)  # (B,T,k), (B,T,k)
            idx_for_gather = idx.unsqueeze(-1).expand(-1, -1, -1, d)
            h_src_g = h_src.unsqueeze(1).expand(-1, T, -1, -1).gather(
                2, idx_for_gather,
            )                                                    # (B, T, k, d)
            V_phi_g = self.V_phi.forward_gathered(h_in, h_src_g) # (B, T, k)
            U_pair = (V_phi_g * m_g).sum()
        else:
            tilde_m = self._sparse_mask(pi, causal, T)           # (B, T, T)
            if cfg.use_grad_checkpoint and self.training:
                P = torch.utils.checkpoint.checkpoint(
                    self.V_phi, h_in, h_src, use_reentrant=False,
                )
            else:
                P = self.V_phi(h_in, h_src)                      # (B, T, T)
            U_pair = (P * tilde_m).masked_fill(~causal, 0.0).sum()

        s_ell = self.per_layer_scale(layer_idx)
        if s_ell is not None:
            U_pair = U_pair * s_ell

        # ── Force computation ──
        U = V_th_per_token.sum() + U_pair
        grad_U, = torch.autograd.grad(
            U, h_in,
            create_graph=self.training,
            retain_graph=True,
        )
        f = -grad_U

        denom = 1.0 + dt * gamma
        h_new = h_in + delta / denom + (dt * dt / (m_b * denom)) * f

        if cfg.ln_after_step:
            h_new = self._project(h_new)
        return h_new

    # ------------------------------------------------------------------
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke():
    """Minimal round-trip on CPU."""
    for layer_ckpt in (False, True):
        for gathered in (False, True):
            tag_parts = []
            if layer_ckpt:
                tag_parts.append("layer_ckpt")
            if gathered:
                tag_parts.append("gathered")
            tag = "+".join(tag_parts) or "baseline"
            cfg = MultiXiPARFConfig(
                vocab_size=257, d=16, max_len=64, L=4,
                v_hidden=32, v_depth=2,
                v_phi_d_type=4, v_phi_d_angle=2,
                v_phi_phi_hidden=8, v_phi_theta_hidden=8,
                v_phi_mlp_hidden=16,
                mass_mode="global",
                top_k=8,
                score_head_hidden=8,
                xi_channels=4,
                xi_alpha_inits=[0.0, 0.5, 0.9, 0.99],
                xi_learnable=True,
                use_layer_checkpoint=layer_ckpt,
                use_gathered_v_phi=gathered,
            )
            torch.manual_seed(0)
            net = MultiXiPARFLM(cfg)
            n = net.num_params()
            alpha_str = ", ".join(f"{a:.3f}" for a in net.xi_alpha_values())
            print(f"[multixi-parf-smoke/{tag}] params: {n:,}")
            print(f"[multixi-parf-smoke/{tag}] K={cfg.xi_channels}  "
                  f"\u03b1=[{alpha_str}]")

            x = torch.randint(0, cfg.vocab_size, (2, 16))
            y = torch.randint(0, cfg.vocab_size, (2, 16))

            net.train()
            logits, loss = net(x, targets=y)
            print(f"[multixi-parf-smoke/{tag}] forward: logits "
                  f"{tuple(logits.shape)} loss {loss.item():.4f}")
            loss.backward()

            alpha_grad = net.xi_module.raw_alpha.grad
            assert alpha_grad is not None, "raw_alpha got no gradient"
            print(f"[multixi-parf-smoke/{tag}] raw_\u03b1 grad norm: "
                  f"{alpha_grad.norm().item():.3e}")
            print(f"[multixi-parf-smoke/{tag}] backward OK.")


if __name__ == "__main__":
    _smoke()
