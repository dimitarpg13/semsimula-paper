"""
PARF-augmented SPLM (Q9c) — Algorithm-A reference prototype.

Reference
---------
companion_notes/PARF_Augmented_SPLM_Architecture.md

Architecture (one-paragraph summary)
------------------------------------
A depth-L stack of velocity-Verlet integrators, each layer applying
the gradient of a SHARED effective scalar

    U^{(ℓ)}_t = V_θ(ξ_t, h_t) + Σ_{s<t} V_φ(h_t, h_s)

to advance every token's hidden state.  V_θ is the SPLM single-particle
external field (4-layer GELU MLP, identical to em_ln-leakfree SPLM and
to the Helmholtz Q9d S-block).  V_φ is a NEW pair-interaction scalar
shared across all (ℓ, t, s).  Past tokens are treated as fixed external
sources (the causal reduction of design-doc §3) by .detach()-ing the
source slice when forming the pair-potential matrix; this severs the
back-reaction force on past tokens and makes the per-token force
strictly causal.

Two V_φ variants ship in this prototype:

  1. Structural (default) — the §5.1-faithful pair potential

         V_φ(h_t, h_s) = -C · Θ_φ(θ(h_t), θ(h_s)) · Φ_φ(l(h_t), l(h_s))
                              / sqrt(||h_t - h_s||^2 + ε^2)

     with l(h) = W_l h ∈ R^{d_l} (type vector), θ(h) = W_θ h ∈ R^K
     (value angles), Φ_φ a learned Gaussian type-matcher and Θ_φ a
     small bounded-tanh MLP value-aligner.  The 1/r factor is softened
     by ε to avoid the s≈t singularity (with the s<t causal mask, exact
     s=t is excluded but nearby tokens can be very close in h-space).

  2. Unstructured MLP (`parf_v_phi='mlp'`) — V_φ(h_t, h_s; φ) is a
     learned MLP applied to concat(h_t, h_s).  This is the design-doc
     OQ-1 ablation: if structural matches MLP the §5.1 prior is
     pedagogical; if structural outperforms, the prior is empirically
     active.

The pair force on h_t is

    F_pair^{(ℓ)}_t = -∇_{h_t} Σ_{s<t} V_φ(h_t, h_s).

We compute it by building the pair-potential matrix P[b, t, s] with
the source slice detached, masking it strictly lower-triangular
(s < t; the diagonal s == t is excluded so the soft 1/r factor never
sees zero distance), summing to a scalar U, and taking its gradient
w.r.t. h.  This vectorises across (B, T, T) and matches attention's
O(T^2) per layer.

The full per-layer force is V_θ-force + V_φ-force; the velocity-Verlet
step is identical in form to the Helmholtz Q9d S-block.

Causal-leak fix
---------------
Two .detach() points preserve causality:

  - ξ is re-derived from h.detach() at every layer (the `causal_force`
    flag of the SPLM family; same as model_sarf_mass.py).
  - The source slice {h_s} of the pair-potential matrix is detached
    so ∇_{h_t} sees a frozen-past field (the §3 causal reduction).

A `causal_force=False` mode is exposed for parity with the SPLM
family's causal-probe forensics; the production default is True.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401  -- explicit submodule import for grad-checkpoint path


_THIS_DIR = Path(__file__).parent
_PARENT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_PARENT_DIR))

from sarf_mass_variant.model_sarf_mass import (  # noqa: E402
    ScalarPotential,
    causal_cumulative_mean,
    _raw_from_positive,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class PARFConfig:
    """PARF-augmented SPLM configuration.

    Defaults match the H1.5 vh=128 cell-shape of the Helmholtz / Variant A
    family for direct comparability:

      d=128, L=8, T<=256, v_hidden=128, v_depth=3, mass_mode='logfreq',
      causal_force=True, ln_after_step=True.

    PARF-specific knobs:

      v_phi_kind        : 'structural' (default) or 'mlp'.
      v_phi_d_type      : type-vector dimension d_l (structural).
      v_phi_d_angle     : value-angle dimension K (structural).
      v_phi_phi_hidden  : hidden width of the Gaussian-gate inverse-bandwidth
                          MLP (structural).
      v_phi_theta_hidden: hidden width of the value-aligner MLP (structural).
      v_phi_mlp_hidden  : hidden width of the unstructured V_φ MLP.
      v_phi_C           : strength constant C (structural).
      v_phi_eps         : Plummer softening for the 1/r factor.
      v_phi_init_scale  : initial weight scale for V_φ.  Smaller than V_θ
                          so the pair force starts as a small perturbation
                          on the SPLM dynamics.
    """
    vocab_size: int = 50257
    d: int = 128
    max_len: int = 256
    L: int = 8

    # SPLM (V_θ) parameters — same as em_ln leakfree / Q9d S-block.
    v_hidden: int = 128
    v_depth: int = 3
    dt: float = 1.0
    init_m: float = 1.0
    init_gamma: float = 0.15
    learn_mgamma: bool = True
    fixed_gamma: Optional[float] = None

    # PARF (V_φ) parameters.
    v_phi_kind: str = "structural"      # 'structural' | 'mlp'
    v_phi_d_type: int = 16               # d_l in design doc
    v_phi_d_angle: int = 8               # K in design doc
    v_phi_phi_hidden: int = 32           # Φ_φ inverse-bandwidth MLP width
    v_phi_theta_hidden: int = 32         # Θ_φ value-aligner MLP width
    v_phi_mlp_hidden: int = 64           # MLP V_φ hidden width
    v_phi_C: float = 1.0                 # strength constant
    v_phi_eps: float = 1e-2              # Plummer softening for 1/r
    v_phi_init_scale: float = 0.02       # init weights small so V_φ starts
                                         # as a perturbation on V_θ dynamics

    # Per-token mass.
    mass_mode: str = "logfreq"           # 'logfreq' | 'global'
    logfreq_init_alpha: float = 0.1
    logfreq_path: Optional[str] = None

    # Stability / parity.
    ln_after_step: bool = True           # LayerNorm after each PARF step
    ln_eps: float = 1e-5
    causal_force: bool = True            # ξ.detach() AND pair-source.detach()
    tie_embeddings: bool = True

    # Performance: gradient-checkpoint the V_φ pair sum.  When True,
    # the V_φ forward at each layer is wrapped in
    # torch.utils.checkpoint.checkpoint(use_reentrant=False), which
    # discards V_φ's intermediate activations after forward and
    # recomputes them during the backward pass.  Trades ~15-25% extra
    # wall-clock for ~50% lower per-layer activation memory; the
    # gradient flow into V_φ's parameters is mathematically unchanged.
    # Recommended ON for the MLP V_φ variant (which OOMs at B=16 on
    # 16 GB MPS without it) and any deeper-stack / longer-T runs.
    # Default OFF so the structural V_φ at the prototype scale (where
    # memory is not the binding constraint) keeps the cheaper path.
    use_grad_checkpoint: bool = False


def _load_npy(p: str) -> np.ndarray:
    return np.load(p)


# ---------------------------------------------------------------------------
# V_φ — structural §5.1-faithful variant
# ---------------------------------------------------------------------------
class StructuralVPhi(nn.Module):
    """§5.1-faithful pair potential

        V_φ(h_t, h_s) = -C · Θ_φ(θ(h_t), θ(h_s)) · Φ_φ(l(h_t), l(h_s))
                             / sqrt(||h_t - h_s||^2 + ε^2)

    Components
    ----------
      l(h) = W_l h            type vector ∈ R^{d_l}
      θ(h) = W_θ h            value angles ∈ R^K
      Φ_φ(l_t, l_s)           = exp(-c · ||l_t - l_s||^2),
                                  c = softplus(Linear(|l_t-l_s|^2, 1) → ())
                                  -- a learned per-pair inverse bandwidth
      Θ_φ(θ_t, θ_s)           = tanh( v · MLP([θ_t, θ_s, θ_t-θ_s]) )
                                  -- bounded value-aligner; for K=2 this
                                  reproduces the design-doc canonical
                                  Θ = -sin(θ_t - θ_s) up to a learnable
                                  parameterisation.
      C                        = scalar strength
      ε                        = Plummer softening, prevents s≈t blow-up

    Forward contract
    ----------------
      forward(h, h_src) -> P of shape (B, T, T)
        h     : (B, T, d) — query side, requires_grad
        h_src : (B, T, d) — source side, .detach()-ed if causal_force=True
        P[b, t, s] = V_φ(h[b, t], h_src[b, s])

    The caller is responsible for masking out s >= t before summing.
    """

    def __init__(self, cfg: PARFConfig):
        super().__init__()
        d, dl, K = cfg.d, cfg.v_phi_d_type, cfg.v_phi_d_angle
        self.K = K
        self.theta_hidden = cfg.v_phi_theta_hidden

        self.W_l = nn.Linear(d, dl, bias=False)
        self.W_theta = nn.Linear(d, K, bias=False)
        # Φ_φ inverse bandwidth: a small MLP that maps the squared type
        # distance to a positive scalar c, broadcast across the pair.
        self.phi_c_net = nn.Sequential(
            nn.Linear(1, cfg.v_phi_phi_hidden), nn.GELU(),
            nn.Linear(cfg.v_phi_phi_hidden, 1),
        )
        # Θ_φ value-aligner.  Conceptually a small MLP on
        # cat([θ_t, θ_s, θ_t-θ_s]) ∈ R^{3K} -> R^{theta_hidden} -> R^1.
        # We split the first linear into separate query, source and
        # difference weight blocks so we can apply them BEFORE the
        # (T, T) outer-product broadcast — this avoids the (B, T, T, 3K)
        # intermediate that otherwise dominates the structural V_φ
        # wall-clock under autograd.  The second layer (theta_hidden -> 1)
        # is a vanilla Linear applied to the (B, T, T, theta_hidden)
        # post-broadcast hidden; that intermediate is unavoidable
        # because the GELU non-linearity in between is not bilinear.
        H = cfg.v_phi_theta_hidden
        self.theta_w_q = nn.Linear(K, H, bias=False)
        self.theta_w_s = nn.Linear(K, H, bias=False)
        self.theta_w_d = nn.Linear(K, H, bias=False)
        self.theta_b1  = nn.Parameter(torch.zeros(H))
        self.theta_w2  = nn.Linear(H, 1)

        self.eps2 = cfg.v_phi_eps ** 2
        self.C = cfg.v_phi_C

        # Initialise: small weights so the pair force starts as a
        # perturbation on the V_θ dynamics.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=cfg.v_phi_init_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _pair_dist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Squared pairwise distance ||a_t - b_s||^2 in O(B·T·T) memory.

        Standard squared-norm expansion:
            ||a-b||^2 = ||a||^2 + ||b||^2 - 2·<a, b>.
        Crucially this avoids materialising the full (B, T, T, d) diff
        tensor (which at training scale is 16·128·128·128·4 B ≈ 130 MB
        per layer and dominates the structural V_φ wall-clock under
        autograd's `create_graph=True`).  At T=128 the saving is ~64×
        in intermediate memory (d / 2) and ~3× in step time on MPS.
        """
        a2 = (a * a).sum(dim=-1, keepdim=True)         # (B, T_q, 1)
        b2 = (b * b).sum(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, T_s)
        ab = torch.matmul(a, b.transpose(1, 2))         # (B, T_q, T_s)
        # clamp to >= 0 to defend against fp negatives near zero.
        return (a2 + b2 - 2.0 * ab).clamp_min(0.0)

    def forward(self, h: torch.Tensor, h_src: torch.Tensor) -> torch.Tensor:
        B, T, d = h.shape
        # Type and angle projections for both sides.
        l_q = self.W_l(h)             # (B, T, dl)
        l_s = self.W_l(h_src)         # (B, T, dl)
        th_q = self.W_theta(h)        # (B, T, K)
        th_s = self.W_theta(h_src)    # (B, T, K)

        # Pairwise type distance (squared) -> Φ_φ Gaussian gate.
        # Squared-norm expansion avoids the (B, T, T, dl) intermediate.
        l_dist2 = self._pair_dist2(l_q, l_s)             # (B, T, T)
        # Per-pair inverse bandwidth c via a small MLP on d^2.
        c = F.softplus(
            self.phi_c_net(l_dist2.unsqueeze(-1)).squeeze(-1)
        )                                                # (B, T, T), positive
        Phi = torch.exp(-c * l_dist2)                    # (B, T, T)

        # Pairwise angle vectors -> Θ_φ bounded-tanh MLP.
        # The first linear layer of the (3K -> H) MLP has been split
        # into per-input weight blocks (theta_w_q, theta_w_s, theta_w_d)
        # so we can apply each before the (T, T) outer-product
        # broadcast.  This avoids materialising the (B, T, T, 3K)
        # intermediate that the naive cat-then-Linear formulation would
        # otherwise dominate the structural V_φ wall-clock with.
        # Mathematical equivalence:
        #   Linear_3K->H( cat([θ_q, θ_s, θ_d]) )
        #     = θ_q W_q + θ_s W_s + θ_d W_d + b
        #   where W = [W_q; W_s; W_d] is the original block-rows split.
        proj_q = self.theta_w_q(th_q)             # (B, T, H)
        proj_s = self.theta_w_s(th_s)             # (B, T, H)
        proj_qd = self.theta_w_d(th_q)            # (B, T, H)  (θ_q part of θ_d)
        proj_sd = self.theta_w_d(th_s)            # (B, T, H)  (θ_s part)
        # Broadcast-add to (B, T, T, H):
        #   hidden[b, t, s, h] = proj_q[b, t, h] + proj_s[b, s, h]
        #                      + proj_qd[b, t, h] - proj_sd[b, s, h]
        #                      + b[h]
        proj_t = proj_q + proj_qd + self.theta_b1     # (B, T, H)
        proj_u = proj_s - proj_sd                     # (B, T, H)
        hidden = proj_t.unsqueeze(2) + proj_u.unsqueeze(1)   # (B, T, T, H)
        hidden = F.gelu(hidden)
        Theta = torch.tanh(self.theta_w2(hidden).squeeze(-1))  # (B, T, T)

        # Distance kernel with Plummer softening (avoids singularity).
        # Squared-norm expansion again to keep the (B, T, T, d) diff
        # tensor out of the autograd graph.
        h_dist2 = self._pair_dist2(h, h_src)             # (B, T, T)
        r = torch.sqrt(h_dist2 + self.eps2)              # (B, T, T)

        # V_φ = -C · Θ · Φ / r  (sign matches the design doc convention:
        # the negative sign makes attractive Θ·Φ = +1 a binding well).
        return -self.C * Theta * Phi / r


# ---------------------------------------------------------------------------
# V_φ — unstructured MLP ablation
# ---------------------------------------------------------------------------
class MLPVPhi(nn.Module):
    """V_φ(h_t, h_s; φ) as an unstructured MLP on concat(h_t, h_s, h_t-h_s).

    Uses the same shape contract as `StructuralVPhi`: returns P of
    shape (B, T, T) with P[b, t, s] = V_φ(h[b, t], h_src[b, s]).
    """

    def __init__(self, cfg: PARFConfig):
        super().__init__()
        d = cfg.d
        self.net = nn.Sequential(
            nn.Linear(3 * d, cfg.v_phi_mlp_hidden), nn.GELU(),
            nn.Linear(cfg.v_phi_mlp_hidden, cfg.v_phi_mlp_hidden), nn.GELU(),
            nn.Linear(cfg.v_phi_mlp_hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=cfg.v_phi_init_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, h_src: torch.Tensor) -> torch.Tensor:
        # Note: the MLP V_φ inherently materialises a (B, T, T, 3d)
        # intermediate -- there is no squared-norm shortcut that
        # preserves the unstructured MLP's non-bilinear dependence on
        # (h_t, h_s).  This is a deliberate cost asymmetry with the
        # structural variant, and is part of the OQ-1 trade-off:
        # unstructured MLPs are slower and have more capacity; the
        # structural form is cheaper and biased.
        B, T, d = h.shape
        h_q = h.unsqueeze(2).expand(B, T, T, d)
        h_k = h_src.unsqueeze(1).expand(B, T, T, d)
        feats = torch.cat([h_q, h_k, h_q - h_k], dim=-1)       # (B, T, T, 3d)
        return self.net(feats).squeeze(-1)                     # (B, T, T)


# ---------------------------------------------------------------------------
# PARF model
# ---------------------------------------------------------------------------
class PARFLM(nn.Module):
    """PARF-augmented SPLM language model (Q9c) — Algorithm-A reference.

    Forward contract
    ----------------
      forward(x, targets=None, return_trajectory=False)
        -> (logits, loss[, traj])

    where traj is the per-layer list of hidden states (length L+1) on
    CPU when return_trajectory=True.
    """

    def __init__(self, cfg: PARFConfig):
        super().__init__()
        self.cfg = cfg

        # Embeddings (token + position), tied output.
        self.E = nn.Embedding(cfg.vocab_size, cfg.d)
        self.P = nn.Parameter(torch.zeros(cfg.max_len, cfg.d))
        nn.init.normal_(self.E.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.P, mean=0.0, std=0.02)

        # ----- Single shared V_theta -----
        self.V_theta = ScalarPotential(cfg.d, cfg.v_hidden, cfg.v_depth)

        # ----- Single shared V_phi -----
        if cfg.v_phi_kind == "structural":
            self.V_phi: nn.Module = StructuralVPhi(cfg)
        elif cfg.v_phi_kind == "mlp":
            self.V_phi = MLPVPhi(cfg)
        else:
            raise ValueError(
                f"unknown v_phi_kind={cfg.v_phi_kind!r}; "
                "expected 'structural' or 'mlp'."
            )

        # ----- Per-token mass + global gamma -----
        self.raw_m_bias = nn.Parameter(
            torch.tensor(_raw_from_positive(cfg.init_m)),
            requires_grad=cfg.learn_mgamma,
        )
        if cfg.fixed_gamma is not None:
            self.raw_gamma = nn.Parameter(
                torch.tensor(0.0), requires_grad=False,
            )
            self._gamma_value: Optional[float] = float(cfg.fixed_gamma)
        else:
            self.raw_gamma = nn.Parameter(
                torch.tensor(_raw_from_positive(cfg.init_gamma)),
                requires_grad=cfg.learn_mgamma,
            )
            self._gamma_value = None

        if cfg.mass_mode == "logfreq":
            if cfg.logfreq_path is None:
                raise ValueError(
                    "mass_mode='logfreq' requires cfg.logfreq_path "
                    "(.npy with one surprisal value per vocab id)."
                )
            surprisal = torch.from_numpy(_load_npy(cfg.logfreq_path)).float()
            if surprisal.numel() != cfg.vocab_size:
                raise ValueError(
                    f"logfreq vector length {surprisal.numel()} != "
                    f"vocab_size {cfg.vocab_size}"
                )
            self.register_buffer("logfreq_surprisal", surprisal)
            self.raw_logfreq_alpha = nn.Parameter(
                torch.tensor(
                    _raw_from_positive(max(cfg.logfreq_init_alpha, 1e-3))
                ),
                requires_grad=True,
            )
        elif cfg.mass_mode == "global":
            pass
        else:
            raise ValueError(
                f"unknown mass_mode for PARF: {cfg.mass_mode!r}. "
                "Supported: 'logfreq', 'global'."
            )

    # ------------------------------------------------------------------
    @property
    def gamma(self) -> torch.Tensor:
        if self._gamma_value is not None:
            return torch.full(
                (), self._gamma_value,
                device=self.raw_gamma.device, dtype=self.raw_gamma.dtype,
            )
        return F.softplus(self.raw_gamma)

    @property
    def m_global(self) -> torch.Tensor:
        return F.softplus(self.raw_m_bias) + 1e-3

    def compute_mass(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.mass_mode == "global":
            return self.m_global
        if cfg.mass_mode == "logfreq":
            surprisal = self.logfreq_surprisal[x]                  # (B, T)
            alpha = F.softplus(self.raw_logfreq_alpha)             # ()
            scaled = alpha * surprisal.unsqueeze(-1)               # (B, T, 1)
            return F.softplus(self.raw_m_bias + scaled) + 1e-3
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------
    def _project(self, h: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(h, (self.cfg.d,), eps=self.cfg.ln_eps)

    def _embed(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        B, T = x.shape
        pos = self.P[position_offset:position_offset + T].unsqueeze(0)
        return self.E(x) + pos

    # ------------------------------------------------------------------
    def _pair_mask_for(self, T: int, device: torch.device) -> torch.Tensor:
        """Cache the strict-lower-triangular mask (s < t)."""
        if not hasattr(self, "_pair_mask") \
                or self._pair_mask.shape[0] != T \
                or self._pair_mask.device != device:
            self._pair_mask = torch.tril(
                torch.ones(T, T, device=device, dtype=torch.bool),
                diagonal=-1,
            )
        return self._pair_mask

    def _layer_step(
        self,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        m_b: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """One velocity-Verlet step of the PARF dynamics.

        delta = h - h_prev
        ξ     = causal_cumulative_mean(h.detach())   # leak-fix invariant
        U     = V_θ(ξ, h) + Σ_{s<t} V_φ(h_t, h_s.detach())
        f     = -∇_h U
        h_new = h + delta / (1+dt·γ)
                  + (dt^2 / (m·(1+dt·γ))) · f

        Performance note: V_θ and V_φ both depend on the same h, so we
        sum them into a single scalar U and take a single
        `autograd.grad` call.  This halves the backward-pass cost
        relative to the two-grad-call version (each call would
        otherwise walk through the full per-layer graph).  The
        strict-causal mask (s < t, diagonal excluded) is applied to
        the pair-potential matrix BEFORE summation; this preserves
        causality and avoids the s≈t Plummer-softened 1/r near-zero.
        """
        cfg = self.cfg
        B, T, d = h.shape
        delta = h - h_prev

        xi_input = h.detach() if cfg.causal_force else h
        xi_now = causal_cumulative_mean(xi_input)

        h_in = h
        if not h_in.requires_grad:
            h_in = h_in.requires_grad_(True)

        # Causal reduction: source slice is .detach()-ed so the
        # gradient of U_pair w.r.t. h sees h_src as a frozen external
        # field.
        h_src = h_in.detach() if cfg.causal_force else h_in

        V_th_per_token = self.V_theta(xi_now, h_in)              # (B, T, 1)

        # V_φ pair sum.  Optionally checkpointed: when on, the per-layer
        # V_φ activations (the (B, T, T, H) Theta hidden state in
        # particular) are not retained for backward; PyTorch
        # re-computes the V_φ forward during the outer backward to
        # recover them.  `use_reentrant=False` is required so that the
        # inner `autograd.grad(U, h_in, create_graph=True)` call below
        # can build a 2nd-order graph through the recomputation.
        if cfg.use_grad_checkpoint and self.training:
            P = torch.utils.checkpoint.checkpoint(
                self.V_phi, h_in, h_src, use_reentrant=False,
            )
        else:
            P = self.V_phi(h_in, h_src)                          # (B, T, T)
        mask = self._pair_mask_for(T, h_in.device)
        P_masked = P.masked_fill(~mask, 0.0)
        U = V_th_per_token.sum() + P_masked.sum()

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
    def _stack_forward(
        self,
        h0: torch.Tensor,
        x: torch.Tensor,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Walk the L PARF layers."""
        cfg = self.cfg
        gamma, dt = self.gamma, cfg.dt
        m_b = self.compute_mass(x)

        h = h0
        h_prev = h0   # velocity proxy starts at 0

        traj: Optional[List[torch.Tensor]] = None
        if return_trajectory:
            traj = [h.detach().cpu()]

        for _ in range(cfg.L):
            h_new = self._layer_step(h, h_prev, m_b, gamma, dt)
            h_prev = h
            h = h_new
            if traj is not None:
                traj.append(h.detach().cpu())

        return h, traj

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        position_offset: int = 0,
    ):
        h0 = self._embed(x, position_offset=position_offset)
        h_L, traj = self._stack_forward(
            h0, x, return_trajectory=return_trajectory,
        )

        logits = h_L @ self.E.weight.T

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
            )

        out = [logits, loss]
        if return_trajectory:
            out.append(traj)
        return tuple(out)


# ---------------------------------------------------------------------------
# Smoke entry point (cheap sanity check, not the real smoke_test.py)
# ---------------------------------------------------------------------------
def _smoke():
    """Minimal one-step round-trip on CPU.  Not the real smoke_test.py."""
    cfg = PARFConfig(
        vocab_size=257, d=16, max_len=64, L=4,
        v_hidden=32, v_depth=2,
        v_phi_d_type=4, v_phi_d_angle=2,
        v_phi_phi_hidden=8, v_phi_theta_hidden=8,
        v_phi_mlp_hidden=16,
        mass_mode="global",
    )
    torch.manual_seed(0)
    net = PARFLM(cfg)
    print(f"[parf-smoke] params: {sum(p.numel() for p in net.parameters()):,}")
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    net.train()
    logits, loss = net(x, targets=y)
    print(f"[parf-smoke] forward: logits {tuple(logits.shape)} "
          f"loss {loss.item():.4f}")
    loss.backward()
    print("[parf-smoke] backward OK; no exceptions.")


if __name__ == "__main__":
    _smoke()
