"""
PARF-augmented SPLM (Q9c) — Algorithm-A reference prototype.

Reference
---------
docs/PARF_Augmented_SPLM_Architecture_v2.md

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

# Structured V_theta support (Phase 2): imported lazily so that the base
# model_parf.py does not hard-depend on model_structured_vtheta.py.
# _has_analytical_grad(m) returns True when m implements analytical_grad.
def _has_analytical_grad(module: "nn.Module") -> bool:
    return callable(getattr(module, "analytical_grad", None))

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
    v_phi_kind: str = "structural"      # 'structural' | 'mlp' | 'structural_competitive'
    v_phi_d_type: int = 16               # d_l in design doc
    v_phi_d_angle: int = 8               # K in design doc
    v_phi_phi_hidden: int = 32           # Φ_φ inverse-bandwidth MLP width
    v_phi_theta_hidden: int = 32         # Θ_φ value-aligner MLP width
    v_phi_mlp_hidden: int = 64           # MLP V_φ hidden width
    v_phi_C: float = 1.0                 # strength constant
    v_phi_eps: float = 1e-2              # Plummer softening for 1/r
    v_phi_init_scale: float = 0.02       # init weights small so V_φ starts
                                         # as a perturbation on V_θ dynamics

    # ----- Lever 3: competitive (softmax-normalised) Φ_φ -----
    # Active only when v_phi_kind == "structural_competitive".
    # Replaces the unnormalised Gaussian type-gate
    #     Φ_φ(l_t, l_s) = exp(-c · ||l_t - l_s||²)
    # with a row-softmax over the causal sources s < t:
    #     Φ̃_φ(l_t, l_s) = scale(t) · softmax_{s<t}(-c · ||l_t - l_s||² / τ),
    # so Σ_{s<t} Φ̃_φ = scale(t) per query token t.  This imports
    # softmax attention's competitive selectivity into the structural
    # V_φ while preserving (i) the AR sign decomposition through Θ_φ
    # and (ii) the gravity-like 1/r distance kernel.  See design doc
    # PARF_Augmented_SPLM_Architecture_v2.md §10 (Lever 3).
    v_phi_competitive_temp: float = 1.0   # τ in the softmax denominator;
                                          # smaller τ ⇒ sharper selectivity.
    v_phi_competitive_scale: str = "row"  # 'row' | 'mean' | 'none':
                                          #   'row'  : multiply by row-causal-count
                                          #            (Σ Φ̃ ≈ unnormalised sum scale).
                                          #   'mean' : leave Σ Φ̃ = 1 (mean-of-pairs scale).
                                          #   'none' : skip post-softmax rescale.

    # ----- P8 patches: scale-balance + saturation-resilient Θ -----
    # Motivated by the diagnostic findings on the P1 dense ckpt:
    #   F-Layer1: R(ℓ=1) ≈ 3 (V_φ dominates V_θ at the embedding-adjacent
    #             layer because ‖h_t-h_s‖ is small there, the Plummer
    #             softening ε=1e-2 is irrelevant, and 1/r blows up).
    #   F-Θsat:   |Θ_φ| saturated at ±1 in layers 2–8 because deep
    #             layers have small 1/r per pair, so the optimiser drives
    #             the value-aligner to its tanh rails to amplify V_φ.
    # All four flags default OFF so the existing `structural` baseline
    # is byte-identical when none are set.  See design-doc §10.9 (P8
    # cell) for predictions and decision rules.
    ln_before_distance: bool = False     # Patch A: replace ‖h_t-h_s‖ with
                                         # ‖LN(h_t)-LN(h_s)‖ inside V_φ.
                                         # Equalises the radial scale across
                                         # layers, kills the Layer-1 1/r
                                         # blowup driven by ‖h‖ growth.
    per_layer_v_phi_scale: bool = False  # Patch B: add a learnable scalar
                                         # s_ℓ = softplus(σ_ℓ) per layer that
                                         # multiplies the V_φ contribution to
                                         # U.  Init σ_ℓ = per_layer_scale_init
                                         # so s_ℓ starts ~ 0.05 ⇒ V_φ enters
                                         # as a perturbation; the optimiser
                                         # may down-weight Layer 1 and
                                         # up-weight middle layers freely.
    per_layer_scale_init: float = -3.0   # softplus(-3) ≈ 0.0486
    theta_activation: str = "tanh"       # Patch C: 'tanh' (default) or
                                         # 'softsign'.  softsign(x) = x/(1+|x|)
                                         # has codomain [-1, 1] like tanh but
                                         # saturates polynomially, so the
                                         # saturation-zone gradient is ~1000×
                                         # larger at logit magnitude 5.
    theta_form: str = "mlp"              # Patch D: 'mlp' (default 3K→H→1
                                         # MLP) or 'bilinear' (θ_t^T W θ_s + b).
                                         # Bilinear has gradient-bounded
                                         # backward and recovers the §5.2
                                         # canonical Θ = -sin(θ_t-θ_s) when
                                         # K=2 and W is skew-symmetric.

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

    # Level-2 per-layer-step checkpointing: wraps each _layer_step call
    # in checkpoint(use_reentrant=False), discarding ALL per-layer
    # intermediates (V_φ forward activations AND 2nd-order graph from
    # create_graph=True) and recomputing them one layer at a time during
    # backward.  Reduces peak V_φ activation memory from O(L) to O(1).
    # Wall-clock cost: ~50% slower per step.  Requires PyTorch >= 2.0.
    # See companion_notes/Gradient_Checkpointing_for_PARF.md.
    use_layer_checkpoint: bool = False


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

        # ----- P8 patch flags (read once, stashed as instance attrs) -----
        self.ln_before_distance = bool(cfg.ln_before_distance)
        self.theta_activation = str(cfg.theta_activation).lower()
        self.theta_form = str(cfg.theta_form).lower()
        if self.theta_activation not in {"tanh", "softsign"}:
            raise ValueError(
                f"theta_activation must be 'tanh' or 'softsign'; "
                f"got {self.theta_activation!r}."
            )
        if self.theta_form not in {"mlp", "bilinear"}:
            raise ValueError(
                f"theta_form must be 'mlp' or 'bilinear'; "
                f"got {self.theta_form!r}."
            )

        self.W_l = nn.Linear(d, dl, bias=False)
        self.W_theta = nn.Linear(d, K, bias=False)
        # Φ_φ inverse bandwidth: a small MLP that maps the squared type
        # distance to a positive scalar c, broadcast across the pair.
        self.phi_c_net = nn.Sequential(
            nn.Linear(1, cfg.v_phi_phi_hidden), nn.GELU(),
            nn.Linear(cfg.v_phi_phi_hidden, 1),
        )
        if self.theta_form == "mlp":
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
        else:  # 'bilinear' — Patch D
            # Θ_φ(θ_t, θ_s) = act(θ_t^T W θ_s + b), implemented as
            #   score[b, t, s] = (θ_q @ W) @ θ_s.transpose(-2, -1) + b.
            # K^2 + 1 params instead of (3K+1)·H + H + 1 of the MLP variant
            # (e.g. K=8, H=32 → 65 vs 169).  Recovers the §5.2 canonical
            # Θ = -sin(θ_t-θ_s) at K=2, W skew-symmetric, exactly.
            self.theta_W = nn.Parameter(torch.zeros(K, K))
            self.theta_b = nn.Parameter(torch.zeros(()))

        self.eps2 = cfg.v_phi_eps ** 2
        self.C = cfg.v_phi_C

        # Initialise: small weights so the pair force starts as a
        # perturbation on the V_θ dynamics.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=cfg.v_phi_init_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.theta_form == "bilinear":
            # Match the per-element scale of the MLP path so initial Θ
            # logits are O(init_scale · K).  Bias starts at 0.
            nn.init.normal_(self.theta_W, std=cfg.v_phi_init_scale)

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

    def _theta_act(self, x: torch.Tensor) -> torch.Tensor:
        """Bounded value-aligner activation.

        'tanh' (default): exponential approach to ±1; gradient (1-tanh²)
        vanishes exponentially fast in the saturation zone.

        'softsign' (Patch C): x / (1+|x|); polynomial approach to ±1;
        gradient 1/(1+|x|)² vanishes only polynomially, ~1000× larger
        than tanh' at logit magnitude 5.  Targets the F-Θsat finding
        that Θ_φ saturates at ±1 in deep PARF layers.
        """
        if self.theta_activation == "tanh":
            return torch.tanh(x)
        return F.softsign(x)

    def _compute_theta(
        self, th_q: torch.Tensor, th_s: torch.Tensor,
    ) -> torch.Tensor:
        """Θ_φ in either MLP form (default) or bilinear form (Patch D)."""
        if self.theta_form == "mlp":
            # Pre-broadcast linear blocks (avoids the (B, T, T, 3K) intermediate).
            proj_q = self.theta_w_q(th_q)             # (B, T, H)
            proj_s = self.theta_w_s(th_s)             # (B, T, H)
            proj_qd = self.theta_w_d(th_q)            # (B, T, H)
            proj_sd = self.theta_w_d(th_s)            # (B, T, H)
            proj_t = proj_q + proj_qd + self.theta_b1     # (B, T, H)
            proj_u = proj_s - proj_sd                     # (B, T, H)
            hidden = proj_t.unsqueeze(2) + proj_u.unsqueeze(1)   # (B, T, T, H)
            hidden = F.gelu(hidden)
            score = self.theta_w2(hidden).squeeze(-1)            # (B, T, T)
        else:  # 'bilinear'
            # score[b, t, s] = θ_q[b, t, :] @ W @ θ_s[b, s, :]^T + b
            tmp = th_q @ self.theta_W                            # (B, T, K)
            score = tmp @ th_s.transpose(-2, -1) + self.theta_b  # (B, T, T)
        return self._theta_act(score)

    def _radial_distance(
        self, h: torch.Tensor, h_src: torch.Tensor,
    ) -> torch.Tensor:
        """Compute r = sqrt(‖h_t − h_s‖² + ε²), with optional LN-before-distance.

        When `ln_before_distance` (Patch A) is on, the inputs to the
        squared-norm expansion are LN-normalised first (no affine).  This
        equalises the radial scale across layers and removes the strong
        dependence of 1/r on the absolute hidden-state norm — the
        mechanism behind the F-Layer1 finding (R(ℓ=1) ≈ 3 in the
        diagnostic).
        """
        if self.ln_before_distance:
            h_for_dist = F.layer_norm(h, (h.shape[-1],))
            hs_for_dist = F.layer_norm(h_src, (h_src.shape[-1],))
        else:
            h_for_dist = h
            hs_for_dist = h_src
        h_dist2 = self._pair_dist2(h_for_dist, hs_for_dist)   # (B, T, T)
        return torch.sqrt(h_dist2 + self.eps2)

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

        # Θ_φ value-aligner — MLP (default) or bilinear (Patch D),
        # bounded by tanh (default) or softsign (Patch C).
        Theta = self._compute_theta(th_q, th_s)          # (B, T, T)

        # Distance kernel with Plummer softening; optional LN-before-
        # distance (Patch A) decouples r from ‖h‖ growth.
        r = self._radial_distance(h, h_src)              # (B, T, T)

        # V_φ = -C · Θ · Φ / r  (sign matches the design doc convention:
        # the negative sign makes attractive Θ·Φ = +1 a binding well).
        return -self.C * Theta * Phi / r

    # ---- Stage-1.5b gathered form ----------------------------------------
    def forward_gathered(
        self,
        h: torch.Tensor,           # (B, T, d)
        h_src_g: torch.Tensor,     # (B, T, k, d)
    ) -> torch.Tensor:
        """Gathered-eval V_φ: intermediates are (B,T,k,H) not (B,T,T,H).

        Mathematically identical to ``forward(h, h_src).gather(-1, idx)``
        where idx produced h_src_g, but uses O(T·k) memory instead of O(T²).
        See companion_notes/PARF_Stage_1_5b_design.md for the equivalence
        proof and memory analysis.
        """
        B, T, d = h.shape
        k = h_src_g.shape[2]

        # ---- type projections ----
        l_q = self.W_l(h)                                      # (B, T, dl)
        l_s = self.W_l(h_src_g)                                # (B, T, k, dl)

        # ---- Φ_φ: Gaussian type-gate (gathered) ----
        diff_l = l_q.unsqueeze(2) - l_s                        # (B, T, k, dl)
        l_dist2 = (diff_l * diff_l).sum(dim=-1)               # (B, T, k)
        c = F.softplus(
            self.phi_c_net(l_dist2.unsqueeze(-1)).squeeze(-1)
        )                                                      # (B, T, k)
        Phi = torch.exp(-c * l_dist2)                          # (B, T, k)

        # ---- Θ_φ: value-aligner (gathered) ----
        th_q = self.W_theta(h)                                 # (B, T, K)
        th_s = self.W_theta(h_src_g)                           # (B, T, k, K)

        if self.theta_form == "mlp":
            H = self.theta_hidden
            proj_q = self.theta_w_q(th_q)                      # (B, T, H)
            proj_s = self.theta_w_s(th_s)                      # (B, T, k, H)
            proj_qd = self.theta_w_d(th_q)                     # (B, T, H)
            proj_sd = self.theta_w_d(th_s)                     # (B, T, k, H)
            proj_t = (proj_q + proj_qd + self.theta_b1).unsqueeze(2)  # (B,T,1,H)
            proj_u = proj_s - proj_sd                          # (B, T, k, H)
            hidden = F.gelu(proj_t + proj_u)                   # (B, T, k, H)
            Theta = self._theta_act(
                self.theta_w2(hidden).squeeze(-1)              # (B, T, k)
            )
        else:  # 'bilinear'
            tmp = th_q @ self.theta_W                          # (B, T, K)
            score = (tmp.unsqueeze(2) * th_s).sum(-1) + self.theta_b  # (B, T, k)
            Theta = self._theta_act(score)

        # ---- distance kernel (gathered), with optional LN-before (Patch A) ----
        if self.ln_before_distance:
            h_for_dist = F.layer_norm(h, (d,))
            hs_for_dist = F.layer_norm(h_src_g, (d,))
        else:
            h_for_dist = h
            hs_for_dist = h_src_g
        h_diff = h_for_dist.unsqueeze(2) - hs_for_dist        # (B, T, k, d)
        h_dist2 = (h_diff * h_diff).sum(dim=-1)               # (B, T, k)
        r = torch.sqrt(h_dist2 + self.eps2)                    # (B, T, k)

        return -self.C * Theta * Phi / r                       # (B, T, k)


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
# V_φ — Lever 3: competitive (softmax-normalised) structural V_φ
# ---------------------------------------------------------------------------
class StructuralCompetitiveVPhi(StructuralVPhi):
    """Structural V_φ with a softmax-competitive type-gate (Lever 3).

    Replaces the unnormalised Gaussian gate
        Φ_φ(l_t, l_s) = exp(-c · ||l_t - l_s||²)
    with a row-softmax across the causal sources s < t:
        Φ̃_φ(l_t, l_s) = scale(t) · softmax_{s<t}(-c · ||l_t - l_s||² / τ)

    Motivation
    ----------
    The diagnostic on the dense P1.6 cell (see
    parf/diagnostics/diagnose_v_phi_channels.py) localises the
    binding constraint on dense PARF to two failure modes:
      (a) Φ_φ saturates near 1 in d=128 type-projection space, i.e.
          the type-gate provides no per-pair selectivity, and
      (b) the dense aggregation across O(T²) pairs interferes
          destructively (signed sum << per-pair magnitude × pair
          count), washing out the directional pair force.

    Lever 3 imports softmax attention's competitive-and-zero-sum
    selectivity (Σ_s w_ts = 1) into the structural V_φ, while
    preserving (i) the AR sign decomposition through Θ_φ ∈ [-1, 1]
    and (ii) the gravity-like 1/r distance kernel.  The result is a
    "PARF-attention hybrid" whose force law is

        F_t  =  -∇_h_t  Σ_{s<t} V_φ(h_t, h_s)
             ∝   Σ_{s<t}  Θ_φ(t, s) · Φ̃_φ(t, s) · (1/r(t, s)) · r̂_{ts}

    with Φ̃_φ a row-stochastic gate.  The architecture remains
    framework-native (every layer is still a velocity-Verlet step
    under a single shared scalar U = V_θ + Σ V_φ); the only change
    is how Φ_φ allocates its mass across past tokens.

    Causality
    ---------
    The strict-causal mask (s < t) is enforced inside V_φ.forward,
    BEFORE the softmax, by setting non-causal logits to -1e9 (a
    large finite negative; -inf would NaN the backward through an
    all-(-inf) row at t = 0).  The outer _layer_step of PARFLM
    re-applies the same mask multiplicatively, so the contract
    "P[b, t, s] = 0 for s ≥ t" is preserved end-to-end.

    Scale options
    -------------
    cfg.v_phi_competitive_scale ∈ {'row', 'mean', 'none'}:
      'row'  : multiply Φ̃ by the per-row causal count t (so Σ Φ̃ ≈ t).
               Approximates the magnitude of the unnormalised dense
               sum, lets the existing C and learning-rate schedules
               transfer with minimal retuning.  This is the default.
      'mean' : leave Σ Φ̃ = 1 per row (mean-of-pairs scale).  Forces
               the model to learn a larger C; exposes the average-
               pair-strength interpretation cleanly.
      'none' : no rescale.  Diagnostic only; expect very small forces.
    """

    def __init__(self, cfg: PARFConfig):
        super().__init__(cfg)
        self.competitive_temp = cfg.v_phi_competitive_temp
        self.competitive_scale = cfg.v_phi_competitive_scale
        if self.competitive_scale not in {"row", "mean", "none"}:
            raise ValueError(
                f"v_phi_competitive_scale must be 'row', 'mean' or 'none'; "
                f"got {self.competitive_scale!r}."
            )

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """Bool mask of shape (T, T) with True for s < t (the causal pair slots)."""
        return torch.tril(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=-1
        )

    def forward(self, h: torch.Tensor, h_src: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        # ----- type and angle projections (unchanged from base class) -----
        l_q = self.W_l(h)
        l_s = self.W_l(h_src)
        th_q = self.W_theta(h)
        th_s = self.W_theta(h_src)

        # ----- competitive Φ̃_φ via causal row-softmax of -c · ||l_t-l_s||² -----
        l_dist2 = self._pair_dist2(l_q, l_s)             # (B, T, T)
        c = F.softplus(
            self.phi_c_net(l_dist2.unsqueeze(-1)).squeeze(-1)
        )                                                # (B, T, T), positive
        # logit = -c · d² / τ  (negative of the squared-distance penalty,
        # which is the Gaussian-gate analogue of an attention score).
        logit = -(c * l_dist2) / max(self.competitive_temp, 1e-6)
        # Causal mask: replace s >= t with a large finite negative so
        # softmax assigns ~0 to those positions AND the backward stays
        # finite (cf. the Gumbel-sparse module's same trick).
        causal = self._causal_mask(T, logit.device)      # (T, T) bool
        logit = logit.masked_fill(~causal[None, ...], -1e9)
        Phi_norm = torch.softmax(logit, dim=-1)          # (B, T, T)

        # Optional rescale.  At t=0 the row is all-(-inf) so softmax
        # returns uniform 1/T; row_has_valid zeros that out cleanly.
        row_has_valid = causal.any(dim=-1)               # (T,)
        Phi_norm = Phi_norm * row_has_valid[None, :, None].to(Phi_norm.dtype)
        if self.competitive_scale == "row":
            # row_count[t] = number of valid causal sources for query t
            #              = t  (because s ranges in [0, t-1])
            row_count = causal.sum(dim=-1).to(Phi_norm.dtype)  # (T,)
            Phi_norm = Phi_norm * row_count[None, :, None]
        elif self.competitive_scale == "mean":
            pass  # Σ Φ̃ = 1 per row; nothing to do.
        else:  # 'none'
            pass

        # ----- Θ_φ value-aligner (P8-aware via base-class helper) -----
        Theta = self._compute_theta(th_q, th_s)          # (B, T, T)

        # ----- distance kernel (P8-aware via base-class helper) -----
        r = self._radial_distance(h, h_src)              # (B, T, T)

        # V_φ = -C · Θ · Φ̃ / r  with the competitive Φ̃ in place of Φ.
        return -self.C * Theta * Phi_norm / r

    def forward_gathered(
        self,
        h: torch.Tensor,           # (B, T, d)
        h_src_g: torch.Tensor,     # (B, T, k, d)
    ) -> torch.Tensor:
        """Gathered-eval of the competitive V_φ.

        In the gathered form the Gumbel-softmax routing has already
        performed competitive source selection, so the per-row softmax
        normalization of the dense forward() is redundant.  We fall back
        to the base-class (unnormalized Gaussian Φ) gathered evaluation.
        """
        return super().forward_gathered(h, h_src_g)


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
        elif cfg.v_phi_kind == "structural_competitive":
            self.V_phi = StructuralCompetitiveVPhi(cfg)
        elif cfg.v_phi_kind == "mlp":
            self.V_phi = MLPVPhi(cfg)
        else:
            raise ValueError(
                f"unknown v_phi_kind={cfg.v_phi_kind!r}; "
                "expected 'structural', 'structural_competitive', or 'mlp'."
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

        # ----- P8 patch B: per-layer learnable V_φ scale -----
        # When enabled, U^(ℓ)_t = V_θ + s_ℓ · Σ_s V_φ with
        # s_ℓ = softplus(σ_ℓ).  Init σ_ℓ = per_layer_scale_init so
        # s_ℓ ≈ 0.05 ⇒ V_φ enters every layer as a perturbation,
        # eliminating the Layer-1 R≈3 force imbalance at random init.
        if cfg.per_layer_v_phi_scale:
            self.raw_v_phi_scale: Optional[nn.Parameter] = nn.Parameter(
                torch.full(
                    (cfg.L,),
                    float(cfg.per_layer_scale_init),
                    dtype=torch.float32,
                )
            )
        else:
            self.raw_v_phi_scale = None

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

    def per_layer_scale(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Return softplus(σ_ℓ) at this layer, or None if Patch B is off.

        Used by both `PARFLM._layer_step` and `SparsePARFLM._layer_step`
        to apply a single per-layer multiplier to the V_φ contribution.
        """
        if self.raw_v_phi_scale is None:
            return None
        return F.softplus(self.raw_v_phi_scale[layer_idx])

    def _layer_step(
        self,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        m_b: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
        layer_idx: int = 0,
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
        # ----- P8 patch B: per-layer scale on the V_φ contribution -----
        s_ell = self.per_layer_scale(layer_idx)
        if s_ell is not None:
            P_masked = P_masked * s_ell

        # ── Phase-2 force computation ────────────────────────────────────
        # When V_theta implements analytical_grad (i.e. is a structured
        # StructuredVThetaBase subclass), we compute ∇_h V_theta analytically
        # and call autograd.grad only on U_phi = V_phi.sum() — a much
        # smaller graph that does NOT require create_graph on V_theta.
        # This eliminates the dominant create_graph overhead in training.
        #
        # When V_theta is the legacy MLP (no analytical_grad), we fall
        # back to the original single-call autograd.grad(U_total, h_in),
        # preserving full backward compatibility.
        if _has_analytical_grad(self.V_theta):
            # Analytical ∇_h V_theta (one matvec, no autograd)
            f_theta = -self.V_theta.analytical_grad(xi_now, h_in)  # (B, T, d)
            # autograd only on the V_phi graph (cheaper: no V_theta terms)
            U_phi = P_masked.sum()
            grad_phi, = torch.autograd.grad(
                U_phi, h_in,
                create_graph=self.training,
                retain_graph=True,
            )
            f = f_theta - grad_phi
        else:
            # Legacy path: single joint autograd.grad over U_total
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

        for ell in range(cfg.L):
            if cfg.use_layer_checkpoint and self.training:
                h_new = torch.utils.checkpoint.checkpoint(
                    self._layer_step,
                    h, h_prev, m_b, gamma, dt, ell,
                    use_reentrant=False,
                )
            else:
                h_new = self._layer_step(
                    h, h_prev, m_b, gamma, dt, layer_idx=ell,
                )
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
    for layer_ckpt in (False, True):
        tag = "layer_ckpt" if layer_ckpt else "no_ckpt"
        cfg = PARFConfig(
            vocab_size=257, d=16, max_len=64, L=4,
            v_hidden=32, v_depth=2,
            v_phi_d_type=4, v_phi_d_angle=2,
            v_phi_phi_hidden=8, v_phi_theta_hidden=8,
            v_phi_mlp_hidden=16,
            mass_mode="global",
            use_layer_checkpoint=layer_ckpt,
        )
        torch.manual_seed(0)
        net = PARFLM(cfg)
        print(f"[parf-smoke/{tag}] params: "
              f"{sum(p.numel() for p in net.parameters()):,}")
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))
        net.train()
        logits, loss = net(x, targets=y)
        print(f"[parf-smoke/{tag}] forward: logits {tuple(logits.shape)} "
              f"loss {loss.item():.4f}")
        loss.backward()
        print(f"[parf-smoke/{tag}] backward OK.")


if __name__ == "__main__":
    _smoke()
