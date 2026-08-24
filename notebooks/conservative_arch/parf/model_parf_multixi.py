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
    integrator: str = "verlet"

    # Cap on the number of low-rank modes rotated exactly by the
    # 'baoab_cfc_lowrank' A-substep: keep only the ``lowrank_max_modes``
    # stiffest eigenmodes of ``L`` (the rest fall back to the explicit
    # kick, which is fine for the soft modes).  ``None`` keeps all
    # ``n_ctx * K * rank`` of them; a small cap bounds the per-token
    # ``P x P`` eigensolve when that aggregate is large.
    lowrank_max_modes: Optional[int] = None

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

        # ── Optional LN before V_theta (bounds force input range) ──
        if cfg.ln_before_vtheta:
            self.ln_before_v = nn.LayerNorm(cfg.d, eps=cfg.ln_eps)
        else:
            self.ln_before_v = None

    # ------------------------------------------------------------------
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
    def _pair_potential(
        self, h_in: torch.Tensor, layer_idx: int,
    ) -> torch.Tensor:
        """Scalar V_φ pair sum at ``h_in`` (Stage-1.5b gathered or dense)."""
        cfg = self.cfg
        B, T, d = h_in.shape

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
        return U_pair

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
        U_pair = self._pair_potential(h_in, layer_idx)

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
            f_kick = f_kick - (lr_sL - self._lowrank_matvec(lr_G, h_mid))
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
