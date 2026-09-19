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
from cfc_baoab import (  # noqa: E402
    cfc_substep,
    decode_velocity,
    encode_velocity,
    lowrank_cfc_substep,
    lowrank_modes,
    ou_step,
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

    # ── Pair-potential family (Gate 0, 2026-09-18) ───────────────────
    # 'sparse_topk'  : the deployed Gumbel straight-through top-k V_phi.
    #                  DEFAULT -- every run predating this option is
    #                  bit-identical under it, and nothing below is built.
    # 'xi_attention' : family A of paper_v5 sections/17f -- all-to-all soft
    #                  routing read off the DETACHED xi summary, so
    #                  alpha(t,s) is constant w.r.t. h_t and the induced
    #                  force remains the gradient of a scalar potential
    #                  (lem:cm-detached-routing).  Implemented by
    #                  model_xi_attention.XiRoutedConservativeAttention,
    #                  imported lazily because that module imports from
    #                  THIS one -- a module-level import would be circular.
    #                  Selecting it drops V_phi and score_head entirely.
    # ── Content-addressed xi routing (A2, 2026-09-18) ────────────────
    # The existing EMA is exactly softmax_s[(t-s)*log alpha_j], so adding a
    # content term inside that same softmax generalises it rather than
    # replacing it:
    #     alpha_j(t,s) = softmax_s[(t-s)*log alpha_j + q_j(h_t).k_j(h_s)/sqrt(d_k)]
    # W_q is zero-initialised, so with this off OR on at step 0 the model is
    # bit-identical -- which is what lets a probe warm-start from a trained
    # checkpoint exactly.  This targets the xi -> V_theta path (87.6% of
    # compute) rather than the pair path that `pair_potential` governs.
    xi_content_route: bool = False
    xi_content_d_k: int = 48
    xi_content_init_scale: float = 0.02

    # ── Relaxed force law (conservativity-price probe, 2026-09-19) ────
    # Measures what the gradient constraint costs, per
    # companion_notes/Measuring_the_Price_of_Conservativity.md.
    #
    # 'none'            : DEFAULT.  f = -grad U exactly.  Nothing is built
    #                     and every run predating this option is unchanged.
    # 'nonconservative' : Arm N.  f = -grad U + lambda * g_psi(xi, h), with
    #                     g_psi vector-valued, so the field need not be
    #                     integrable and the defect kappa is free to grow.
    # 'potential'       : Arm C.  The SAME parameter budget routed through a
    #                     scalar: f = -grad(U + lambda * Phi_psi(xi, h)).
    #                     The force is still a gradient, so kappa == 0 by
    #                     construction.  This is the control that separates
    #                     "the model wanted non-conservativity" from "the
    #                     model wanted more parameters".
    #
    # lambda is ALWAYS zero-initialised, making step 0 bit-identical and the
    # warm start exact.  psi is NOT: the force depends on the product, so
    # zeroing both makes both gradients vanish (see the note's SS3.3).
    # 'attention'       : Arm N-attn.  The added force is the §5.1 direct
    #                     exchange -- standard causal multi-head attention,
    #                     softmax(qk/sqrt(d_k)) @ v with an output
    #                     projection, nothing detached, injected rather than
    #                     derived from a potential.  Its conservative
    #                     counterpart is pair_potential='xi_attention'
    #                     (family A), which is the SAME mechanism made
    #                     conservative by the framework's own rules: routing
    #                     read off detached xi and the force taken as the
    #                     gradient of a scalar.  The pair measures what
    #                     conservativity costs using a mechanism already
    #                     known to work -- GPT-2 beats this model 1.49x with
    #                     it -- rather than a random field that may simply
    #                     have failed to find the useful direction.
    force_relaxation: str = "none"
    relax_hidden: int = 128
    relax_init_scale: float = 0.02
    relax_lambda_per_layer: bool = True
    # How the added term is held at zero on step 0.
    #
    # 'zero_readout' : DEFAULT.  The field's OUTPUT layer is zeroed and its
    #                  input layer is random, so the term is identically
    #                  zero at init while the readout carries a live,
    #                  well-conditioned gradient from step 1 -- the standard
    #                  zero-init-the-output-projection trick.
    # 'scalar'       : a single learnable lambda in front of a fixed random
    #                  field, zero-initialised.  SUPERSEDED: lambda can
    #                  scale that field but not orient it, and a random
    #                  direction in d dimensions overlaps the useful one by
    #                  only about 1/sqrt(d) with arbitrary per-batch sign,
    #                  so lambda random-walks instead of growing.  Raising
    #                  relax_init_scale does not help: it scales signal and
    #                  noise together, and Adam is scale-invariant in the
    #                  gradient.  Kept to reproduce the 2026-09-19 run.
    relax_gate: str = "zero_readout"
    relax_attn_d_k: int = 48        # attention modes only
    relax_attn_heads: int = 4
    # Routing source for 'attention_potential'. Defaults to 'h' so the
    # conservative arm is parameter-matched against 'attention', which
    # routes from h. Both are conservative -- the lemma requires the
    # routing be detached, not that it come from xi -- so charging xi's
    # extra width to "conservativity" would confound the measurement.
    relax_attn_route_from: str = "h"
    # Fix the gate instead of learning it. None learns lambda (or holds it
    # at 1 under zero_readout); a float pins every layer there and freezes
    # it, which is how the PPL-versus-kappa curve is swept. Two probes have
    # now shown a learned gate reports as much about the optimiser as about
    # the architecture: a scalar gate diffused as sqrt(t), and a zero
    # readout surged to 50% force share and raised the loss.
    relax_lambda_fixed: Optional[float] = None

    pair_potential: str = "sparse_topk"
    attn_n_heads: int = 4
    attn_d_k: int = 48
    attn_d_v: int = 48
    attn_kernel: str = "dot"               # 'dot' | 'rbf'
    attn_init_scale: float = 0.02          # small: enters as a perturbation
    attn_rbf_log_sigma_init: float = 0.0
    # Zero the query read-out W_uq so phi -- and hence the added potential
    # -- is identically zero at init.  Without it family A is NOT
    # bit-identical and cannot warm-start exactly.  'dot' kernel only.
    attn_zero_readout: bool = False

    # Stability: force clamping and LN-before-V_theta.
    force_clamp_max: Optional[float] = None   # clamp force to [-F, F] per dim
    ln_before_vtheta: bool = False            # LN(h) before V_theta evaluation

    # ── Integrator (see cfc_baoab.py) ────────────────────────────────
    # 'verlet'     : damped velocity-Verlet, friction folded into the
    #                1/(1+dt*gamma) coefficient.  The historical default;
    #                bit-identical to every run before this option existed.
    # 'baoab'      : palindromic splitting with an exact OU friction
    #                substep, exp(-gamma*dt), and a genuine velocity.
    # 'baoab_cfc'  : as 'baoab', but the stiff diagonal part of V_theta is
    #                propagated by its closed-form harmonic solution
    #                instead of an explicit kick -- unconditionally stable
    #                however sharp the wells become.  Requires a V_theta
    #                exposing ``harmonic_terms`` (anisotropic Gaussian).
    # 'baoab_cfc_lowrank' : as 'baoab_cfc', but the anisotropic *off*-diagonal
    #                coupling (the ``B_k B_k^T`` part) is ALSO integrated
    #                exactly, on the modes of the aggregate PSD low-rank
    #                operator ``L = sum_k g_k B_k B_k^T`` (mitigation "#1"
    #                of the CfC/BAOAB companion note).  This removes the last
    #                explicitly-integrated stiff channel, so an anisotropic
    #                well no longer has an ``omega dt < 2`` wall on any axis.
    #                Requires a V_theta exposing ``harmonic_terms_lowrank``.
    #                An A-substep Strang-splits the diagonal spring and the
    #                low-rank rotation (2nd order in their commutator, both
    #                factors unconditionally stable).
    #                STATUS (2026-08-30): mathematically correct and stable
    #                (see cfc_baoab.py's module docstring for the three bugs
    #                fixed getting it there), but NOT production-feasible at
    #                L=8/d=384/OWT scale -- the batched per-token SVD costs
    #                ~120 s/step at full width, ~50 s/step even restricted to
    #                the 2 stiffest layers via ``lowrank_layers`` (vs.
    #                ~10-15 s/step for 'baoab_cfc'), and the companion note's
    #                stiffness bracket (S:33) shows the curvature it bounds,
    #                sigma_max(B_k)^2, is only a weak correlate of the
    #                observed gradient spikes, not their driver.  Retained
    #                for completeness / smaller-scale future use; production
    #                training uses 'baoab_cfc'.
    integrator: str = "verlet"

    # Cap on the number of low-rank modes rotated exactly by the
    # 'baoab_cfc_lowrank' A-substep: keep only the ``lowrank_max_modes``
    # stiffest eigenmodes of ``L`` (the rest fall back to the explicit
    # kick, which is fine for the soft modes).  ``None`` keeps all
    # ``n_ctx * K * rank`` of them; a small cap bounds the per-token
    # ``P x P`` eigensolve when that aggregate is large.
    lowrank_max_modes: Optional[int] = None

    # Which driver extracts those eigenmodes.  'svd' is the historical path
    # (full torch.linalg.svd, or a randomised range-finder when truncating).
    # 'gram' eigendecomposes the P x P Gram G^T G instead -- measured 241x
    # faster at this model's shapes on an A100, because the randomised path
    # spends 99.6% of its time in three batched (d x q) QRs, and it returns
    # all P modes so truncation becomes unnecessary.  See _gram_eigh in
    # cfc_baoab.py for the condition-number tradeoff it accepts.
    lowrank_driver: str = "svd"

    # Restrict the (expensive) 'baoab_cfc_lowrank' exact off-diagonal
    # integration to a subset of layers -- the batched per-token SVD is the
    # whole cost of this arm, so running it on only the stiffest layers cuts
    # wall-clock roughly proportionally.  Layers not listed fall back to the
    # diagonal CfC substep ('baoab_cfc'), which is cheap and unconditionally
    # stable on its own diagonal spring.  ``None`` = every layer (original
    # behaviour).  Accepts any container of 0-based layer indices.
    lowrank_layers: Optional[frozenset] = None

    # Randomised-SVD cost knobs for the low-rank modes (see cfc_baoab.lowrank_modes):
    # subspace iterations and probe oversampling.  Fewer iterations / probes =
    # cheaper but slightly less accurate top-modes; the demoted soft modes go
    # to the (stable) explicit kick anyway, so modest values are safe.
    lowrank_niter: int = 2
    lowrank_oversample: int = 4

    # Compute -grad V_theta from its closed form instead of autograd.
    # This is what removes V_theta from the second-order `create_graph`
    # chain; orthogonal to the integrator choice, and forced on by the
    # BAOAB family (which needs the force split).  Ignored when V_theta
    # has no ``analytical_grad`` or when ln_before_vtheta is set.
    vtheta_analytic_force: bool = False

    # Thermostat temperature for the O-step.  0.0 = deterministic friction
    # only, which keeps a BAOAB run directly comparable to a Verlet one.
    langevin_T: float = 0.0
    langevin_noise_eval: bool = False         # sample noise in eval too


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class _RelaxationField(nn.Module):
    """``(xi, h) -> R^out``.  out = d for Arm N, out = 1 for Arm C.

    Deliberately the same shape for both arms so the only difference is
    whether the output is a force or a potential.  Weights are drawn at
    ``init_scale``; the gate ``lambda`` in front of this module is what
    starts at zero.
    """

    def __init__(self, d: int, K: int, hidden: int, out_dim: int,
                 init_scale: float, gate: str = "zero_readout"):
        super().__init__()
        self.in_dim = (K + 1) * d
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=init_scale)
                nn.init.zeros_(m.bias)
        if gate == "zero_readout":
            # Output layer at zero: the field is identically zero at init,
            # so step 0 stays bit-identical, but dL/dW2 = delta (x)
            # GELU(W1 z) is non-zero and points where the loss wants to go.
            # W1 stays random and unlocks once W2 leaves zero.  Contrast the
            # 'scalar' gate, whose single coefficient can only rescale a
            # fixed random direction.
            nn.init.zeros_(self.net[2].weight)
            nn.init.zeros_(self.net[2].bias)

    def forward(self, xis: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        B, T, K, d = xis.shape
        return self.net(torch.cat([xis.reshape(B, T, K * d), h], dim=-1))


def _matched_hidden(in_dim: int, d: int, hidden_n: int) -> int:
    """Hidden width for Arm C that matches Arm N's parameter count.

    Arm N:  in*H  + H + H*d + d
    Arm C:  in*H' + H' + H'  + 1
    """
    target = hidden_n * (in_dim + 1 + d) + d
    return max(1, round((target - 1) / (in_dim + 2)))


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
            content_route=getattr(cfg, "xi_content_route", False),
            d=cfg.d,
            content_d_k=getattr(cfg, "xi_content_d_k", 48),
            content_init_scale=getattr(cfg, "xi_content_init_scale", 0.02),
        )

        # ── Optional LN before V_theta (bounds force input range) ──
        if cfg.ln_before_vtheta:
            self.ln_before_v = nn.LayerNorm(cfg.d, eps=cfg.ln_eps)
        else:
            self.ln_before_v = None

        # ── Optional xi-routed conservative attention (family A) ──
        # getattr, not cfg.pair_potential: a checkpoint or notebook built
        # against an older config still loads and still takes the sparse
        # path, which is the whole point of the default.
        # ── Optional relaxed force law ──
        self.relax_field = None
        self.relax_lambda = None
        _relax = getattr(cfg, "force_relaxation", "none")
        if _relax != "none":
            _VALID = ("nonconservative", "potential", "attention",
                      "attention_potential")
            if _relax not in _VALID:
                raise ValueError(
                    f"force_relaxation must be 'none' or one of {_VALID}, "
                    f"got {_relax!r}")
            _H = getattr(cfg, "relax_hidden", 128)
            _in = (cfg.xi_channels + 1) * cfg.d
            if _relax == "nonconservative":
                _out, _hid = cfg.d, _H
            else:
                _out, _hid = 1, _matched_hidden(_in, cfg.d, _H)
            _gate = getattr(cfg, "relax_gate", "zero_readout")
            if _gate not in ("zero_readout", "scalar"):
                raise ValueError(
                    f"relax_gate must be 'zero_readout' or 'scalar', "
                    f"got {_gate!r}")
            if _relax == "attention_potential":
                # The conservative twin of 'attention'. The SAME xi-routed
                # attention, but entering as a scalar potential so the force
                # stays a gradient.
                #
                # It must ADD, not replace. pair_potential='xi_attention'
                # sets V_phi and score_head to None, discarding a trained
                # component -- measured at 6.2e-05 even on a randomly
                # initialised model, and far larger on a real checkpoint.
                # Arm N adds its term, so the control must too, or the pair
                # differs by more than conservativity.
                from model_xi_attention import XiRoutedConservativeAttention
                self.relax_field = XiRoutedConservativeAttention(
                    d=cfg.d, xi_channels=cfg.xi_channels,
                    n_heads=getattr(cfg, "relax_attn_heads", 4),
                    d_k=getattr(cfg, "relax_attn_d_k", 48),
                    d_v=getattr(cfg, "relax_attn_d_k", 48),
                    kernel="dot",
                    init_scale=getattr(cfg, "relax_init_scale", 0.02),
                    zero_readout=(_gate == "zero_readout"),
                    route_from=getattr(cfg, "relax_attn_route_from", "h"),
                )
                self._relax_takes_h_only = False
            elif _relax == "attention":
                # Lazy: model_fock_attention imports from THIS module.
                from model_fock_attention import DirectExchangeForce
                self.relax_field = DirectExchangeForce(
                    d=cfg.d,
                    d_k=getattr(cfg, "relax_attn_d_k", 48),
                    n_heads=getattr(cfg, "relax_attn_heads", 4),
                    init_scale=getattr(cfg, "relax_init_scale", 0.02),
                )
                if _gate == "zero_readout":
                    # W_O is the output projection, so zeroing it holds the
                    # exchange force at exactly zero while leaving Q/K/V
                    # random and giving W_O a live gradient -- the same
                    # discipline that fixed the scalar gate.
                    nn.init.zeros_(self.relax_field.W_O.weight)
                self._relax_takes_h_only = True
            else:
                self.relax_field = _RelaxationField(
                    d=cfg.d, K=cfg.xi_channels, hidden=_hid, out_dim=_out,
                    init_scale=getattr(cfg, "relax_init_scale", 0.02),
                    gate=_gate,
                )
                self._relax_takes_h_only = False
            _n_lam = cfg.L if getattr(cfg, "relax_lambda_per_layer", True) else 1
            _fixed = getattr(cfg, "relax_lambda_fixed", None)
            if _fixed is not None:
                del self.relax_lambda
                self.register_buffer(
                    "relax_lambda", torch.full((_n_lam,), float(_fixed)))
            elif _gate == "scalar":
                self.relax_lambda = nn.Parameter(torch.zeros(_n_lam))
            elif True:
                # The readout already holds the field at zero, so the
                # coefficient is a fixed 1 and carries no parameters.  It
                # stays a buffer so the force path is identical in both
                # modes and so the resume block still sees it as new.
                # `self.relax_lambda = None` above put the name in __dict__,
                # and register_buffer refuses an existing attribute.
                del self.relax_lambda
                self.register_buffer("relax_lambda", torch.ones(_n_lam))
            # Per-layer share of the force carried by the added term,
            # ||lambda g|| / ||conservative force||.  With 'zero_readout'
            # this replaces lambda as the measurement.
            self.register_buffer(
                "relax_share", torch.zeros(cfg.L), persistent=False)

        self.V_attn = None
        if getattr(cfg, "pair_potential", "sparse_topk") == "xi_attention":
            # Lazy import: model_xi_attention imports MultiXiPARFConfig
            # from this module, so a top-level import is circular.
            from model_xi_attention import XiRoutedConservativeAttention
            self.V_attn = XiRoutedConservativeAttention(
                d=cfg.d,
                xi_channels=cfg.xi_channels,
                n_heads=getattr(cfg, "attn_n_heads", 4),
                d_k=getattr(cfg, "attn_d_k", 48),
                d_v=getattr(cfg, "attn_d_v", 48),
                kernel=getattr(cfg, "attn_kernel", "dot"),
                init_scale=getattr(cfg, "attn_init_scale", 0.02),
                rbf_log_sigma_init=getattr(cfg, "attn_rbf_log_sigma_init", 0.0),
                zero_readout=getattr(cfg, "attn_zero_readout", False),
            )
            # Retire the sparse machinery so its parameters leave the
            # optimiser and the state_dict (same choice XiAttnPARFLM makes).
            self.V_phi = None
            self.score_head = None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def relax_share_values(self) -> List[float]:
        """Per-layer ||added force|| / ||conservative force||, or [].

        Under ``relax_gate='zero_readout'`` this replaces lambda as the
        probe's primary measurement: the fraction of the dynamics the model
        has chosen to take outside the conservative class.
        """
        if getattr(self, "relax_share", None) is None:
            return []
        return [float(v) for v in self.relax_share.detach().cpu().tolist()]

    @torch.no_grad()
    def relax_lambda_values(self) -> List[float]:
        """Per-layer gate values, or [] when the relaxation is off."""
        if self.relax_lambda is None:
            return []
        return [float(v) for v in self.relax_lambda.detach().cpu().tolist()]

    @torch.no_grad()
    def xi_alpha_values(self) -> List[float]:
        """Current α_k values (diagnostic)."""
        return [float(a) for a in self.xi_module.alpha.detach().cpu().tolist()]

    # ------------------------------------------------------------------
    def _use_analytic_vtheta(self) -> bool:
        """Whether -∇V_theta can and should be taken from its closed form.

        Requires (a) the caller to have asked for it or an integrator that
        needs the force split, (b) a V_theta that implements
        ``analytical_grad``, and (c) no LayerNorm between h and V_theta
        (``ln_before_vtheta``), whose Jacobian the closed form does not
        include.
        """
        cfg = self.cfg
        wants = (
            getattr(cfg, "vtheta_analytic_force", False)
            or getattr(cfg, "integrator", "verlet") != "verlet"
        )
        return (
            wants
            and self.ln_before_v is None
            and _has_analytical_grad(self.V_theta)
        )

    # ------------------------------------------------------------------
    def _relax_gate(self, layer_idx: int) -> torch.Tensor:
        """The per-layer gate, or the single shared one."""
        n = self.relax_lambda.numel()
        return self.relax_lambda[layer_idx % n]

    def _add_relax_potential(self, U_pair, xis, h_in, layer_idx):
        """Arm C: add lambda * Phi_psi to the potential, so the force it
        induces is still a gradient.  Added AFTER the per-layer V_phi scale
        so that lambda is the only gate in front of it."""
        if self.relax_field is None:
            return U_pair
        _mode = getattr(self.cfg, "force_relaxation", "none")
        if _mode not in ("potential", "attention_potential"):
            return U_pair
        if xis is None:
            raise RuntimeError(
                "force_relaxation='potential' needs xis; callers must pass "
                "xis=... to _pair_potential (see _layer_forces).")
        lam = self._relax_gate(layer_idx)
        if _mode == "attention_potential":
            B, T, _ = h_in.shape
            h_src = h_in.detach() if self.cfg.causal_force else h_in
            _route = (h_in.detach()
                      if getattr(self.relax_field, "route_from", "xi") == "h"
                      else xis.detach())
            add = self.relax_field.potential(
                h_in, h_src, _route,
                self._pair_mask_for(T, h_in.device))
        else:
            add = self.relax_field(xis, h_in).sum()
        return U_pair + lam * add

    def _pair_potential(
        self, h_in: torch.Tensor, layer_idx: int,
        xis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scalar pair potential at ``h_in``.

        Sparse top-k V_φ by default; the xi-routed conservative attention
        potential when ``cfg.pair_potential == 'xi_attention'``.  ``xis`` is
        required only by the latter and is ignored by the former, so the
        sparse path keeps its original two-argument behaviour.
        """
        cfg = self.cfg
        B, T, d = h_in.shape

        if self.V_attn is not None:
            if xis is None:
                raise RuntimeError(
                    "pair_potential='xi_attention' needs the xi tensor for "
                    "routing, but _pair_potential was called without it. "
                    "Callers must pass xis=... (see _layer_forces)."
                )
            h_src = h_in.detach() if cfg.causal_force else h_in
            # Detach again unconditionally. xis is already built from
            # h.detach() whenever causal_force is set, but conservativity
            # must not silently depend on that flag: alpha has to be a
            # constant w.r.t. h_in for the force to stay a gradient.
            xi_route = xis.detach()
            causal = self._pair_mask_for(T, h_in.device)   # strict s < t
            U_pair = self.V_attn.potential(h_in, h_src, xi_route, causal)
            s_ell = self.per_layer_scale(layer_idx)
            if s_ell is not None:
                U_pair = U_pair * s_ell
            return self._add_relax_potential(U_pair, xis, h_in, layer_idx)

        h_src = h_in.detach() if cfg.causal_force else h_in
        h_src_for_score = (
            h_in.detach() if cfg.score_head_use_detached_h_src else h_in
        )

        pi = self.score_head(h_in, h_src_for_score)              # (B, T, T)
        causal = self._pair_mask_for(T, h_in.device)

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
            if cfg.use_grad_checkpoint and torch.is_grad_enabled():
                P = torch.utils.checkpoint.checkpoint(
                    self.V_phi, h_in, h_src, use_reentrant=False,
                )
            else:
                P = self.V_phi(h_in, h_src)                      # (B, T, T)
            U_pair = (P * tilde_m).masked_fill(~causal, 0.0).sum()

        s_ell = self.per_layer_scale(layer_idx)
        if s_ell is not None:
            U_pair = U_pair * s_ell
        return self._add_relax_potential(U_pair, xis, h_in, layer_idx)

    # ------------------------------------------------------------------
    def _layer_forces(
        self,
        h_in: torch.Tensor,
        xis: torch.Tensor,
        layer_idx: int,
        *,
        split: bool = False,
        vtheta_comps=None,
    ):
        """Conservative force ``f = -∇_h (V_theta + V_φ)`` evaluated at ``h_in``.

        Returns ``f`` (default) or the pair ``(f_theta, f_phi)`` when
        ``split=True``, which the BAOAB/CfC integrator needs so it can
        route the two contributions through different substeps.

        ``vtheta_comps`` optionally carries well parameters already
        derived from ``xis`` by the caller (see
        ``AnisotropicMixtureGaussianVTheta.context_components``), so the
        CfC step does not re-derive them after having built them for its
        harmonic linearisation.

        Force computation is fp32-guarded for bf16 stability: under bf16
        autocast the potential may be bf16, so it is cast to fp32 before
        differentiation, keeping the gradient (which compounds across L
        layers) in full precision.  The cast is in-graph, so autograd
        still traces back through the bf16 V_theta / V_φ ops to their
        parameters.  No-op when already fp32.
        """
        cfg = self.cfg
        U_pair = self._pair_potential(h_in, layer_idx, xis=xis)

        if self._use_analytic_vtheta():
            # Closed-form V_theta force: no autograd, so V_theta never
            # enters the second-order create_graph chain at all.  Only the
            # (much smaller) V_φ graph is differentiated twice.
            if vtheta_comps is None:
                f_theta = -self.V_theta.analytical_grad(xis, h_in)
            else:
                f_theta = -self.V_theta.analytical_grad(
                    xis, h_in, comps=vtheta_comps,
                )
            with torch.autocast(device_type="cuda", enabled=False):
                grad_phi, = torch.autograd.grad(
                    U_pair.float(), h_in,
                    create_graph=self.training,
                    retain_graph=self.training,
                )
            f_phi = -grad_phi
        else:
            if split:
                raise RuntimeError(
                    "The BAOAB/CfC integrators need the V_theta force in "
                    "closed form, but this V_theta has no analytical_grad "
                    "(or ln_before_vtheta is set, whose Jacobian the "
                    "closed form omits). Use integrator='verlet', or an "
                    "anisotropic-Gaussian V_theta."
                )
            h_for_v = (
                self.ln_before_v(h_in) if self.ln_before_v is not None else h_in
            )
            V_th_per_token = self.V_theta(xis, h_for_v)           # (B, T, 1)
            U = V_th_per_token.sum() + U_pair
            with torch.autocast(device_type="cuda", enabled=False):
                grad_U, = torch.autograd.grad(
                    U.float(), h_in,
                    create_graph=self.training,
                    retain_graph=self.training,
                )
            f_theta, f_phi = None, -grad_U

        # Arm N: an unconstrained field added to the force itself, so the
        # total need not be a gradient.  It goes into f_phi because that is
        # the term the CfC/BAOAB integrator treats as a plain kick -- only
        # V_theta's harmonic part is subtracted from f_kick and propagated
        # exactly, and this field has no harmonic decomposition.
        if (self.relax_field is not None
                and getattr(cfg, "force_relaxation", "none")
                in ("nonconservative", "attention")):
            _field = (self.relax_field(h_in)
                      if getattr(self, "_relax_takes_h_only", False)
                      else self.relax_field(xis, h_in))
            _add = self._relax_gate(layer_idx) * _field
            # Record the share of the force carried outside the conservative
            # class.  Under 'zero_readout' this IS the measurement, lambda
            # being fixed at 1.  Three norms per layer, negligible.
            with torch.no_grad():
                _cons = f_phi if f_theta is None else f_theta + f_phi
                self.relax_share[layer_idx] = (
                    _add.norm() / (_cons.norm() + 1e-12))
            f_phi = f_phi + _add

        if split:
            return f_theta, f_phi

        f = f_phi if f_theta is None else f_theta + f_phi
        if cfg.force_clamp_max is not None:
            f = f.clamp(-cfg.force_clamp_max, cfg.force_clamp_max)
        return f

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
        """One damped velocity-Verlet step with K-EMA ξ + sparse PARF routing.

        Identical to SparsePARFLM._layer_step except:
          - causal_cumulative_mean(xi_input) → self.xi_module(xi_input)
          - V_theta(xi_now, h_in) → V_theta(xis, h_in) with xis: (B, T, K, d)

        The force itself is computed by :meth:`_layer_forces`, whose
        ``autograd.grad`` call passes ``retain_graph=self.training``:
        retain_graph is only needed when create_graph=True, because only
        then does the gradient carry a grad_fn back into a graph that the
        *outer* ``loss.backward()`` will walk a second time.  In eval
        (create_graph=False) it would instead keep every layer's buffers
        alive with no outer backward ever around to free them -- across L
        layers that is exactly the eval-time OOM in forward_gathered.

        This method integrates with the historical Verlet update and is
        bit-identical to the pre-integrator-option behaviour.  The BAOAB /
        CfC integrators live in :meth:`_layer_step_langevin` and are
        reached through :meth:`_layer_step_ex`, because they need to
        return an outgoing velocity as well as a position.
        """
        cfg = self.cfg
        delta = h - h_prev

        # ── Multi-channel ξ (replaces causal_cumulative_mean) ──
        xi_input = h.detach() if cfg.causal_force else h
        xis = self.xi_module(xi_input)                           # (B, T, K, d)

        h_in = h
        if not h_in.requires_grad:
            h_in = h_in.requires_grad_(True)

        f = self._layer_forces(h_in, xis, layer_idx)

        denom = 1.0 + dt * gamma
        h_new = h_in + delta / denom + (dt * dt / (m_b * denom)) * f

        if cfg.ln_after_step:
            h_new = self._project(h_new)
        return h_new

    # ------------------------------------------------------------------
    def _layer_step_langevin(
        self,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        m_b: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
        layer_idx: int = 0,
    ) -> tuple:
        """One BAOAB-family step, returning ``(h_new, h_prev_out)``.

        Palindromic position-first (ABOBA) ordering, one force evaluation
        per layer to match the cost of the Verlet step it replaces::

            A  half substep   drift, or the exact harmonic flow under CfC
            B  full kick      everything the A substep did not integrate
            O  friction       exact exp(-gamma*dt) (+ FDT noise if T > 0)
            A  half substep   drift / harmonic flow again

        Under ``integrator='baoab_cfc'`` the A substeps propagate the
        stiff diagonal part of V_theta *exactly* (see
        ``cfc_baoab.cfc_substep``) and the B kick carries only the
        remainder ``f_theta - f_harm + f_phi``.  The two parts sum to the
        unmodified total force, so this changes how the dynamics is
        integrated without changing the force field being integrated --
        which is what makes a Verlet-vs-CfC comparison interpretable.

        The outgoing velocity is encoded back into ``h_prev_out`` so the
        ``(h, h_prev)`` state signature, the checkpoint layout and the
        inference path are all unchanged.
        """
        cfg = self.cfg
        use_cfc = cfg.integrator == "baoab_cfc"
        use_lowrank = cfg.integrator == "baoab_cfc_lowrank"
        # Per-layer opt-out: run the costly exact off-diagonal integration only
        # on the configured (stiffest) layers; elsewhere fall back to the cheap
        # diagonal CfC substep.  The two share the same frozen-force split, so
        # this only changes how the low-rank part is propagated on this layer.
        if use_lowrank:
            lr_layers = getattr(cfg, "lowrank_layers", None)
            if lr_layers is not None and layer_idx not in lr_layers:
                use_lowrank = False
                use_cfc = True
        half = 0.5 * dt

        xi_input = h.detach() if cfg.causal_force else h
        xis = self.xi_module(xi_input)                           # (B, T, K, d)

        h_in = h
        if not h_in.requires_grad:
            h_in = h_in.requires_grad_(True)

        v = decode_velocity(h_in, h_prev, dt)

        # Cached, frozen linearisation of V_theta at h_in (reused by both A
        # substeps and the kick subtraction below), and the well parameters
        # so the force evaluation does not re-derive the bank.
        k_diag = s_lin = None
        lr_U = lr_kappa = lr_sL = lr_G = None
        vtheta_comps = None
        if use_cfc or use_lowrank:
            if use_lowrank and not hasattr(self.V_theta, "harmonic_terms_lowrank"):
                raise RuntimeError(
                    "integrator='baoab_cfc_lowrank' needs a V_theta exposing "
                    "harmonic_terms_lowrank(xis, h) -- e.g. the anisotropic "
                    "Gaussian family in model_aniso_gaussian_vtheta.py. "
                    "Use integrator='baoab_cfc' or 'baoab' otherwise."
                )
            if use_cfc and not hasattr(self.V_theta, "harmonic_terms"):
                raise RuntimeError(
                    "integrator='baoab_cfc' needs a V_theta exposing "
                    "harmonic_terms(xis, h) -- e.g. the anisotropic "
                    "Gaussian family in model_aniso_gaussian_vtheta.py. "
                    "Use integrator='baoab' for other V_theta variants."
                )
            # The well parameters depend only on xis, so derive them once
            # here and hand them to the force evaluation below: without
            # this the bank (whose low-rank factor alone is K*d*rank
            # floats per token) is built twice per layer, which is most of
            # the CfC arm's activation footprint.
            if hasattr(self.V_theta, "context_components"):
                vtheta_comps = self.V_theta.context_components(xis)

        if use_lowrank:
            # Impulse / RESPA scheme: the stiff PSD low-rank part L = G G^T
            # is put in the exact fast flow (lowrank_cfc_substep), which
            # carries the drift; the clamped diagonal spring, V_phi and the
            # nonlinear V_theta residual are demoted to the explicit kick.
            _, _, lr_G, lr_Gmu = self.V_theta.harmonic_terms_lowrank(
                xis, h_in, comps=vtheta_comps,
            )
            lr_U, lr_kappa = lowrank_modes(
                lr_G, max_modes=getattr(cfg, "lowrank_max_modes", None),
                niter=getattr(cfg, "lowrank_niter", 2),
                oversample=getattr(cfg, "lowrank_oversample", 4),
                driver=getattr(cfg, "lowrank_driver", "svd"),
            )
            lr_sL = torch.einsum('...dp,...p->...d', lr_G, lr_Gmu)
            f_L = lr_sL - self._lowrank_matvec(lr_G, h_in)
            h_mid, v_mid = lowrank_cfc_substep(
                h_in, v, lr_U, lr_kappa, f_L, m_b, half,
            )
        elif use_cfc:
            # Frozen over the layer step, as in any exponential
            # integrator: the linearisation is taken once, at h.
            k_diag, s_lin = self.V_theta.harmonic_terms(
                xis, h_in, comps=vtheta_comps,
            )
            h_mid, v_mid = cfc_substep(
                h_in, v, s_lin - k_diag * h_in, k_diag, m_b, half,
            )
        else:
            h_mid, v_mid = h_in + half * v, v

        if not h_mid.requires_grad:
            h_mid = h_mid.requires_grad_(True)

        # ── B: kick with whatever the A substeps did not already carry ──
        f_theta, f_phi = self._layer_forces(
            h_mid, xis, layer_idx, split=True, vtheta_comps=vtheta_comps,
        )
        f_kick = f_theta + f_phi
        if use_cfc:
            f_kick = f_kick - (s_lin - k_diag * h_mid)
        elif use_lowrank:
            # Subtract only the retained-mode projection P_U f_L of the
            # low-rank harmonic force -- exactly the part the A substeps
            # integrate exactly.  With no truncation span(lr_U) = range(L),
            # so P_U is the identity on f_L and this equals the full
            # subtraction (the step is unchanged); with lowrank_max_modes
            # set, the dropped soft modes stay in this explicit kick (stable,
            # being the sub-threshold ones) instead of being silently
            # cancelled out of the dynamics.
            f_L_mid = lr_sL - self._lowrank_matvec(lr_G, h_mid)
            f_L_mid = torch.einsum(
                '...dq,...q->...d', lr_U,
                torch.einsum('...dq,...d->...q', lr_U, f_L_mid),
            )
            f_kick = f_kick - f_L_mid
        if cfg.force_clamp_max is not None:
            f_kick = f_kick.clamp(-cfg.force_clamp_max, cfg.force_clamp_max)
        v_mid = v_mid + (dt / m_b) * f_kick

        # ── O: exact friction, optionally FDT-thermostatted ──
        v_mid = ou_step(
            v_mid, gamma, dt, m=m_b,
            T=getattr(cfg, "langevin_T", 0.0),
            training=self.training,
            noise_eval=getattr(cfg, "langevin_noise_eval", False),
        )

        # ── A: second half substep ──
        if use_lowrank:
            f_L = lr_sL - self._lowrank_matvec(lr_G, h_mid)
            h_new, v_new = lowrank_cfc_substep(
                h_mid, v_mid, lr_U, lr_kappa, f_L, m_b, half,
            )
        elif use_cfc:
            h_new, v_new = cfc_substep(
                h_mid, v_mid, s_lin - k_diag * h_mid, k_diag, m_b, half,
            )
        else:
            h_new, v_new = h_mid + half * v_mid, v_mid

        if cfg.ln_after_step:
            h_new = self._project(h_new)
        return h_new, encode_velocity(h_new, v_new, dt)

    @staticmethod
    def _lowrank_matvec(G: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``(G G^T) x`` without forming the d x d operator: G @ (G^T x)."""
        return torch.einsum(
            '...dp,...p->...d', G, torch.einsum('...dp,...d->...p', G, x),
        )

    # ------------------------------------------------------------------
    def _layer_step_ex(
        self,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        m_b: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
        layer_idx: int = 0,
    ) -> tuple:
        """Dispatch to the configured integrator; see the base-class docstring."""
        if getattr(self.cfg, "integrator", "verlet") == "verlet":
            return self._layer_step(h, h_prev, m_b, gamma, dt, layer_idx), h
        return self._layer_step_langevin(
            h, h_prev, m_b, gamma, dt, layer_idx=layer_idx,
        )

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
