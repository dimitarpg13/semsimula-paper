"""
Gaussian mixture-PDF V_theta: structurally bounded scalar potential.

Replaces the SQ3 (log-sum-exp mixture) V_theta with a sum of negative
Gaussian bumps whose potential and force are bounded by construction:

    V(xi, h) = -sum_k w_k(xi) exp(-||a_k^{1/2}(h - mu_k(xi))||^2 / 2)

Properties (contrast with SQ3):
    - Range:           [-sum_k w_k, 0]     (SQ3: (-inf, +inf))
    - Force max:       0.607 w_k / sigma_k  (SQ3: unbounded)
    - V^2 penalty:     bounded by (sum w_k)^2
    - Jacobi metric:   positive-definite everywhere (no degeneracy)

This directly eliminates the "Blowup 1" failure mode (penalty dominance)
and structurally caps the force magnitude that drove Blowups 2-3.

Two variants:
    MixtureGaussianVTheta    -- learned mu_k(xi) centres (general)
    SARFGaussianVTheta       -- frozen PMI-peak anchor centres (SARF)

Both implement the StructuredVThetaBase interface (forward + analytical_grad).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_structured_vtheta import StructuredVThetaBase


class MixtureGaussianVTheta(StructuredVThetaBase):
    """K-component mixture of Gaussian wells (bounded potential).

      V(xi, h) = -sum_k w_k(xi) exp(-0.5 a_k(xi)^T (h - mu_k(xi))^2)

    where w_k > 0 (softmax weights), a_k > 0 (diagonal precision via
    softplus), and mu_k are context-conditioned centres.

    Analytical gradient:
      grad_h V = sum_k w_k * a_k ⊙ (h - mu_k) * exp(-0.5 a_k^T (h-mu_k)^2)

    The force -grad_h V is bounded because the (h-mu_k) * exp(-r^2/2)
    product peaks at r = sigma_k and decays exponentially beyond.
    """

    def __init__(self, d: int, K: int = 8, w_scale: float = 1.0,
                 xi_d: Optional[int] = None,
                 init_log_precision: Optional[float] = None,
                 precision_max: Optional[float] = None):
        """
        Parameters
        ----------
        d : int
            Well dimension (must match h dimension).
        K : int
            Number of Gaussian wells.
        w_scale : float
            Scale factor for softmax weights.
        xi_d : int or None
            Context input dimension.  Defaults to d.  Set to K_xi * d
            when used with the multi-xi adapter.
        init_log_precision : float or None
            Initial value for the log-precision bias of ``a_proj``.
            Controls the initial effective well width:
            ``sigma_eff = 1 / sqrt(softplus(init_log_precision))``.
            When ``ln_after_step=True`` is active, hidden states live in
            a LayerNorm-normalised space with ``||h_L|| ≈ sqrt(d)``.
            Set this to ``-math.log(d)`` so that ``sigma_eff ≈ sqrt(d)``
            and the wells are active from step 1.
            None keeps the default (``softplus(0) ≈ 0.693``, very narrow).
        precision_max : float or None
            Hard upper bound on per-dimension precision ``a_k``.  Equivalent
            to a floor on ``sigma_k = 1/sqrt(a_k)``.  Without this cap the
            optimizer can drive ``a_k → ∞``, collapsing wells into
            delta-function spikes whose force diverges (same instability
            mechanism as SQ3 Blowup 2, arriving via precision).
            Recommended value: ``1.0 / d`` → ``sigma_min = sqrt(d)``.
            A looser ``2.0 / d`` allows sigma down to ``sqrt(d/2)``.
            None disables the constraint.
        """
        super().__init__()
        self.d = d
        self.K = K
        self.w_scale = w_scale
        self._init_log_precision = init_log_precision
        self._precision_max: Optional[float] = precision_max
        in_d = xi_d if xi_d is not None else d
        self.mu_proj = nn.Linear(in_d, K * d)
        self.a_proj = nn.Linear(in_d, K * d)
        self.w_proj = nn.Linear(in_d, K)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.mu_proj.weight, std=0.02)
        nn.init.zeros_(self.mu_proj.bias)
        nn.init.normal_(self.a_proj.weight, std=0.02)
        if self._init_log_precision is not None:
            # Bias initialised so softplus(bias) ≈ exp(bias) ≈ exp(init_log_precision),
            # giving sigma_eff = 1/sqrt(a_k) ≈ 1/sqrt(exp(init_log_precision)).
            # For init_log_precision = -log(d): sigma_eff ≈ sqrt(d).
            nn.init.constant_(self.a_proj.bias, self._init_log_precision)
        else:
            nn.init.zeros_(self.a_proj.bias)
        nn.init.normal_(self.w_proj.weight, std=0.02)
        nn.init.zeros_(self.w_proj.bias)

    def _components(self, xi: torch.Tensor):
        """Parse context xi into well parameters."""
        lead = xi.shape[:-1]
        mu = self.mu_proj(xi).view(*lead, self.K, self.d)
        a = (F.softplus(self.a_proj(xi)) + 1e-4).view(*lead, self.K, self.d)
        if self._precision_max is not None:
            a = a.clamp(max=self._precision_max)
        w = F.softmax(self.w_proj(xi), dim=-1) * self.w_scale   # (..., K)
        return mu, a, w

    def forward(self, xi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        mu, a, w = self._components(xi)                          # (...,K,d), (...,K,d), (...,K)
        h_e = h.unsqueeze(-2)                                    # (...,1,d)
        diff = h_e - mu                                          # (...,K,d)
        exponent = -0.5 * (a * diff * diff).sum(dim=-1)          # (...,K)
        bumps = w * torch.exp(exponent)                          # (...,K)
        V = -bumps.sum(dim=-1, keepdim=True)                     # (...,1)
        return V

    def analytical_grad(
        self, xi: torch.Tensor, h: torch.Tensor,
    ) -> torch.Tensor:
        mu, a, w = self._components(xi)
        h_e = h.unsqueeze(-2)
        diff = h_e - mu
        exponent = -0.5 * (a * diff * diff).sum(dim=-1)
        g = w * torch.exp(exponent)                              # (...,K)
        per_comp = a * diff * g.unsqueeze(-1)                    # (...,K,d)
        return per_comp.sum(dim=-2)                              # (...,d)

    def attractor_centres(self, xi: torch.Tensor) -> torch.Tensor:
        lead = xi.shape[:-1]
        return self.mu_proj(xi).view(*lead, self.K, self.d)


class SARFGaussianVTheta(StructuredVThetaBase):
    """Gaussian wells centred on frozen SARF anchor positions.

      V(xi, h) = -sum_j w_j(xi) exp(-||h - a_j||^2 / (2 sigma_j^2))

    where a_j are static PMI-peak anchor embeddings (registered as
    non-trainable buffers) and sigma_j are learned per-anchor widths.

    This structurally prevents escape: PMI-extremal anchors span the
    semantic space, so every h is near at least one well centre.

    Analytical gradient:
      grad_h V = sum_j w_j / sigma_j^2 * (h - a_j) * exp(...)
    """

    def __init__(
        self,
        d: int,
        anchor_positions: torch.Tensor,
        xi_d: int,
        w_scale: float = 1.0,
        init_log_sigma: float = 0.0,
        log_sigma_max: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        d : int
            Model hidden dimension.
        anchor_positions : (N_S, d)
            Frozen PMI-peak anchor embeddings.
        xi_d : int
            Dimension of the flattened xi input (K_xi * d for multi-xi).
        w_scale : float
            Scale factor for softmax weights.
        init_log_sigma : float
            Initial log(sigma) for per-anchor widths.
        log_sigma_max : float or None
            Hard upper bound on log(sigma).  Prevents wells from widening
            into flatness during training.  Recommended value:
            ``0.5 * math.log(d) + 1.0``  (one e-fold above sqrt(d)).
            None disables the constraint.
        """
        super().__init__()
        self.d = d
        N_S = anchor_positions.shape[0]
        self.N_S = N_S
        self.w_scale = w_scale
        self._log_sigma_max: Optional[float] = log_sigma_max

        self.register_buffer('anchors', anchor_positions.detach().clone())

        self.log_sigma = nn.Parameter(
            torch.full((N_S,), init_log_sigma)
        )
        self.w_proj = nn.Linear(xi_d, N_S)
        nn.init.normal_(self.w_proj.weight, std=0.02)
        nn.init.zeros_(self.w_proj.bias)

    @property
    def sigma(self) -> torch.Tensor:
        """Effective sigma, clamped to [1e-3, exp(log_sigma_max)] if set."""
        ls = self.log_sigma
        if self._log_sigma_max is not None:
            ls = ls.clamp(max=self._log_sigma_max)
        return ls.exp().clamp(min=1e-3)

    def clamp_params(self) -> None:
        """Project log_sigma into the feasible set after each optimizer step.

        Call this immediately after ``optimizer.step()`` to enforce the
        log_sigma_max constraint via projected gradient descent.  Without this,
        Adam can push log_sigma past log_sigma_max even though the effective
        sigma is clamped in the forward pass (the optimizer sees a flat gradient
        surface and keeps accumulating momentum past the constraint boundary).
        """
        if self._log_sigma_max is not None:
            self.log_sigma.data.clamp_(max=self._log_sigma_max)

    def forward(self, xi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma                                       # (N_S,)
        w = F.softmax(self.w_proj(xi), dim=-1) * self.w_scale   # (..., N_S)
        h_e = h.unsqueeze(-2)                                    # (..., 1, d)
        diff = h_e - self.anchors                                # (..., N_S, d)
        dist_sq = (diff * diff).sum(dim=-1)                      # (..., N_S)
        exponent = -dist_sq / (2.0 * sigma * sigma)
        bumps = w * torch.exp(exponent)
        return -bumps.sum(dim=-1, keepdim=True)

    def analytical_grad(
        self, xi: torch.Tensor, h: torch.Tensor,
    ) -> torch.Tensor:
        sigma = self.sigma
        inv_sigma_sq = 1.0 / (sigma * sigma)                    # (N_S,)
        w = F.softmax(self.w_proj(xi), dim=-1) * self.w_scale
        h_e = h.unsqueeze(-2)
        diff = h_e - self.anchors
        dist_sq = (diff * diff).sum(dim=-1)
        exponent = -dist_sq / (2.0 * sigma * sigma)
        g = w * torch.exp(exponent)                              # (..., N_S)
        per_anchor = diff * (g * inv_sigma_sq).unsqueeze(-1)     # (..., N_S, d)
        return per_anchor.sum(dim=-2)

    def attractor_centres(self, xi: torch.Tensor) -> torch.Tensor:
        lead = xi.shape[:-1]
        return self.anchors.unsqueeze(0).expand(*lead, -1, -1)


# ---------------------------------------------------------------------------
# Multi-Xi adapter (same interface as SQ3 adapter)
# ---------------------------------------------------------------------------
class GaussianVThetaMultiXiAdapter(nn.Module):
    """Adapts Gaussian V_theta classes to Multi-Xi interface.

    (xis: (B, T, K, d), h: (B, T, d)) -> V: (B, T, 1)
    """

    def __init__(self, inner: StructuredVThetaBase, K: int, d: int):
        super().__init__()
        self.inner = inner
        self.K = K
        self.d = d

    def forward(self, xis: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        B, T, K, d = xis.shape
        xi_flat = xis.reshape(B, T, K * d)
        return self.inner(xi_flat, h)

    def analytical_grad(self, xis: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        B, T, K, d = xis.shape
        xi_flat = xis.reshape(B, T, K * d)
        return self.inner.analytical_grad(xi_flat, h)

    def attractor_centres(self, xis: torch.Tensor) -> torch.Tensor:
        B, T, K, d = xis.shape
        xi_flat = xis.reshape(B, T, K * d)
        return self.inner.attractor_centres(xi_flat)


# ---------------------------------------------------------------------------
# Multi-context Gaussian V_theta (Bottleneck-2 cure)
# ---------------------------------------------------------------------------
class MultiContextGaussianVTheta(nn.Module):
    """Per-context Gaussian well banks: V = sum_m V^(m)(xi^(m), h).

    The concat baseline (``MixtureGaussianVTheta`` + ``GaussianVThetaMultiXiAdapter``)
    flattens the K multi-resolution xi-channels into one ``K*d`` summary and
    feeds a *single* well bank, so the model cannot resolve which horizon a
    given attractor responds to.  This variant instead gives **each xi-channel
    (context view) its own Gaussian well bank** and sums their potentials:

        V(xis, h) = sum_{m=1}^{n_ctx}  V_m(xi^(m), h),
        V_m(xi^(m), h) = -sum_k w_k^m(xi^(m)) exp(-0.5 a_k^m (h - mu_k^m)^2)

    Properties
    ----------
    - Parameter-neutral vs the concat baseline: each bank's projection is
      ``d -> K*d`` and there are ``n_ctx`` banks, i.e. the same total
      ``(n_ctx*d) -> (n_ctx*K*d)`` mapping as one ``K*d -> K*d`` bank, but now
      block-diagonal by horizon instead of dense.
    - Bounded: V in ``[-n_ctx * sum_k w_k, 0]`` (sum of bounded banks).
    - Conservative: ``-grad_h V`` is the gradient of a scalar.
    - Expressivity: ``n_ctx * K`` distinct attractors, each conditioned on a
      single resolution, instead of ``K`` attractors on a blurred summary.

    Interface matches ``GaussianVThetaMultiXiAdapter``:
        forward(xis: (B,T,n_ctx,d), h: (B,T,d)) -> V: (B,T,1)
    """

    def __init__(
        self,
        d: int,
        K: int,
        n_ctx: int,
        w_scale: float = 1.0,
        init_log_precision: Optional[float] = None,
        precision_max: Optional[float] = None,
    ):
        super().__init__()
        self.d = d
        self.K = K
        self.n_ctx = n_ctx
        self.banks = nn.ModuleList(
            MixtureGaussianVTheta(
                d=d, K=K, w_scale=w_scale, xi_d=d,
                init_log_precision=init_log_precision,
                precision_max=precision_max,
            )
            for _ in range(n_ctx)
        )

    def forward(self, xis: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        out = self.banks[0](xis[..., 0, :], h)
        for m in range(1, self.n_ctx):
            out = out + self.banks[m](xis[..., m, :], h)
        return out

    def analytical_grad(self, xis: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        out = self.banks[0].analytical_grad(xis[..., 0, :], h)
        for m in range(1, self.n_ctx):
            out = out + self.banks[m].analytical_grad(xis[..., m, :], h)
        return out

    def attractor_centres(self, xis: torch.Tensor) -> torch.Tensor:
        """Per-context attractor centres concatenated: (B, T, n_ctx*K, d)."""
        cs = [
            self.banks[m].attractor_centres(xis[..., m, :])
            for m in range(self.n_ctx)
        ]
        return torch.cat(cs, dim=-2)


# ---------------------------------------------------------------------------
# Depth-conditioned multi-context Gaussian V_theta (cheap per-layer untying)
# ---------------------------------------------------------------------------
class DepthConditionedMultiContextGaussianVTheta(nn.Module):
    """One shared multi-context well bank + a learned per-layer code.

    Motivation
    ----------
    Fully untying ``V_theta`` across the ``L`` Verlet layers (``G=L``
    independent ``MultiContextGaussianVTheta`` copies) gives each layer its
    own potential landscape but multiplies the (already dominant) well-bank
    parameter count by ``L`` — e.g. ~12M -> ~190M at ``L=16``.  Almost all of
    that cost lives in the two dense ``d -> K*d`` projections of every bank.

    This module keeps a *single* shared ``MultiContextGaussianVTheta`` bank and
    instead gives each layer a small learned **depth code** ``e_g in R^{n_ctx x d}``
    that is added to the per-channel context ``xi`` before the bank projections:

        xi_g^(m) = xi^(m) + e_g^(m),
        V_g(xis, h) = sum_m V_m(xi_g^(m), h).

    Each layer therefore sees a *shifted* view of the same well bank, producing
    a distinct effective potential per layer at the cost of only
    ``L * n_ctx * d`` extra parameters (e.g. 16*5*384 ~ 31k) instead of ~178M.

    Properties
    ----------
    - Conservative: ``e_g`` is constant w.r.t. ``h``, so the per-step force is
      still ``-grad_h V`` of a scalar potential (the gradient is unchanged by
      the additive constant shift of the *input* context).
    - Parameter cost: shared bank (~12M at L=16, d=384, K=8, n_ctx=5) plus a
      tiny ``(L, n_ctx, d)`` code table.
    - Per-layer specialisation: shift-only (the well *shapes* are tied; only
      the conditioning differs).  Use full per-layer untying / LoRA-across-depth
      if shift-only conditioning proves too weak.

    Layer routing
    -------------
    The bank is invoked once per Verlet layer.  Because the standard interface
    is ``forward(xis, h)`` with no layer argument, the *current* layer index is
    read from the mutable ``_active_layer`` attribute, which the model's
    per-layer step is expected to set immediately before the call (see
    ``install_depth_routing`` below).  This is safe under gradient checkpointing
    because the attribute is set synchronously within the same layer-step call
    that consumes it, on both the forward pass and the recomputation pass.

    Interface matches ``MultiContextGaussianVTheta`` / the adapters:
        forward(xis: (B,T,n_ctx,d), h: (B,T,d)) -> V: (B,T,1)
    """

    def __init__(
        self,
        d: int,
        K: int,
        n_ctx: int,
        n_layers: int,
        w_scale: float = 1.0,
        init_log_precision: Optional[float] = None,
        precision_max: Optional[float] = None,
        code_init_std: float = 0.02,
    ):
        super().__init__()
        self.d = d
        self.K = K
        self.n_ctx = n_ctx
        self.n_layers = n_layers
        self.bank = MultiContextGaussianVTheta(
            d=d, K=K, n_ctx=n_ctx, w_scale=w_scale,
            init_log_precision=init_log_precision,
            precision_max=precision_max,
        )
        # Per-layer context shift: (n_layers, n_ctx, d).  Small init so the
        # model starts ~tied (all layers share one bank) and learns to
        # differentiate.
        self.depth_code = nn.Parameter(
            torch.randn(n_layers, n_ctx, d) * code_init_std
        )
        # Mutable routing pointer set by the model's layer step.  Not a
        # buffer/parameter: it carries no state to persist.
        self._active_layer: int = 0

    @property
    def banks(self) -> nn.ModuleList:
        """Expose the underlying per-channel banks (clamp_params compat)."""
        return self.bank.banks

    def set_active_layer(self, layer_idx: int) -> None:
        self._active_layer = int(layer_idx)

    def _shift(self, xis: torch.Tensor) -> torch.Tensor:
        """Add the active layer's depth code to xis: (..., n_ctx, d)."""
        g = self._active_layer
        if not (0 <= g < self.n_layers):
            g = g % self.n_layers
        code = self.depth_code[g]                                # (n_ctx, d)
        # Broadcast over all leading (batch/time) dims of xis.
        lead = xis.dim() - 2
        code = code.view(*([1] * lead), self.n_ctx, self.d)
        return xis + code

    def forward(self, xis: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.bank(self._shift(xis), h)

    def analytical_grad(self, xis: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # The depth code is constant w.r.t. h, so grad_h is the bank's grad
        # evaluated at the shifted context.
        return self.bank.analytical_grad(self._shift(xis), h)

    def attractor_centres(self, xis: torch.Tensor) -> torch.Tensor:
        return self.bank.attractor_centres(self._shift(xis))


def install_depth_routing(model) -> None:
    """Monkey-patch ``model._fock_layer_step`` to broadcast the layer index.

    The depth-conditioned ``V_theta`` reads its active layer from
    ``model.V_theta._active_layer``.  This wrapper sets that attribute at the
    start of every per-layer step so the shared bank receives the correct
    depth code, including during gradient-checkpoint recomputation (the
    attribute is set inside the same call that consumes it).

    No-op if ``model.V_theta`` is not depth-conditioned.
    """
    import types

    vt = getattr(model, "V_theta", None)
    if not isinstance(vt, DepthConditionedMultiContextGaussianVTheta):
        return
    if getattr(model, "_depth_routing_installed", False):
        return

    if not hasattr(model, "_fock_layer_step"):
        raise AttributeError(
            "install_depth_routing expects model._fock_layer_step "
            "(FockMultiXiPARFLM)."
        )

    _orig = model._fock_layer_step.__func__   # unbound original

    def _routed_fock_layer_step(self, h, h_prev, r, salience, m_b, gamma,
                                dt, layer_idx, *args, **kwargs):
        vt_local = getattr(self, "V_theta", None)
        if isinstance(vt_local, DepthConditionedMultiContextGaussianVTheta):
            vt_local.set_active_layer(layer_idx)
        return _orig(self, h, h_prev, r, salience, m_b, gamma, dt,
                     layer_idx, *args, **kwargs)

    model._fock_layer_step = types.MethodType(_routed_fock_layer_step, model)
    model._depth_routing_installed = True


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke():
    from model_structured_vtheta import validate_analytical_grad

    d = 16
    print(f"--- Gaussian V_theta validation (d={d}) ---")
    validate_analytical_grad(MixtureGaussianVTheta(d=d, K=4), d=d)
    validate_analytical_grad(MixtureGaussianVTheta(d=d, K=8), d=d)

    anchors = torch.randn(6, d)
    validate_analytical_grad(
        SARFGaussianVTheta(d=d, anchor_positions=anchors, xi_d=d),
        d=d,
    )

    print(f"\n--- Multi-Xi adapter (K_xi=4) ---")
    K_xi = 4
    xi_d = K_xi * d

    inner_g = MixtureGaussianVTheta(d=d, K=8, xi_d=xi_d)
    adapter_g = GaussianVThetaMultiXiAdapter(inner_g, K=K_xi, d=d)

    B, T = 2, 8
    xis = torch.randn(B, T, K_xi, d)
    h = torch.randn(B, T, d, requires_grad=True)
    V = adapter_g(xis, h)
    assert V.shape == (B, T, 1), f"Expected (B,T,1), got {V.shape}"

    inner_s = SARFGaussianVTheta(d=d, anchor_positions=anchors, xi_d=xi_d)
    adapter_s = GaussianVThetaMultiXiAdapter(inner_s, K=K_xi, d=d)
    V_s = adapter_s(xis, h)
    assert V_s.shape == (B, T, 1)

    print(f"  MixtureGaussian adapter V shape: {tuple(V.shape)}")
    print(f"  SARFGaussian adapter V shape:    {tuple(V_s.shape)}")

    print(f"\n--- Boundedness check ---")
    h_far = torch.randn(2, 8, d) * 100.0
    xi_test = torch.randn(2, 8, d)
    mod = MixtureGaussianVTheta(d=d, K=4)
    V_far = mod(xi_test, h_far)
    print(f"  V at ||h||~100: min={V_far.min().item():.6f}  "
          f"max={V_far.max().item():.6f}  (should be in [-1, 0])")

    print(f"\n--- Parameter counts at d=128, K=8 ---")
    d_big = 128
    sq3 = __import__('model_structured_vtheta').MixtureQuadraticVTheta(d=d_big, K=8)
    gauss = MixtureGaussianVTheta(d=d_big, K=8)
    sarf_a = torch.randn(64, d_big)
    sarf = SARFGaussianVTheta(d=d_big, anchor_positions=sarf_a, xi_d=d_big)
    for name, m in [("SQ3 K=8", sq3), ("Gaussian K=8", gauss),
                    ("SARF N_S=64", sarf)]:
        n = sum(p.numel() for p in m.parameters())
        n_buf = sum(b.numel() for b in m.buffers())
        print(f"  {name:<20}: {n:>8,} params  {n_buf:>6,} buffer")

    print(f"\n--- Depth-conditioned multi-context V_theta ---")
    n_ctx, L = 4, 6
    dcv = DepthConditionedMultiContextGaussianVTheta(
        d=d, K=8, n_ctx=n_ctx, n_layers=L,
    )
    mcv = MultiContextGaussianVTheta(d=d, K=8, n_ctx=n_ctx)
    xis = torch.randn(2, 8, n_ctx, d)
    h = torch.randn(2, 8, d, requires_grad=True)
    # Different layers must produce different potentials.
    dcv.set_active_layer(0)
    V0 = dcv(xis, h)
    dcv.set_active_layer(L - 1)
    VL = dcv(xis, h)
    assert V0.shape == (2, 8, 1)
    assert not torch.allclose(V0, VL), "depth code did not differentiate layers"
    # analytical_grad must match autograd at the shifted context.
    dcv.set_active_layer(2)
    g_ana = dcv.analytical_grad(xis, h)
    V = dcv(xis, h).sum()
    g_auto, = torch.autograd.grad(V, h)
    max_err = (g_ana - g_auto).abs().max().item()
    print(f"  depth code differentiates layers: V0!=VL  OK")
    print(f"  analytical_grad vs autograd max_err: {max_err:.2e}")
    assert max_err < 1e-4, f"analytical_grad mismatch: {max_err}"
    n_dcv = sum(p.numel() for p in dcv.parameters())
    n_mcv = sum(p.numel() for p in mcv.parameters())
    n_code = dcv.depth_code.numel()
    print(f"  shared-bank+code params: {n_dcv:,}  "
          f"(bank {n_mcv:,} + code {n_code:,})")
    print(f"  vs {L} untied copies:    {n_mcv * L:,}  "
          f"({n_mcv * L / max(n_dcv,1):.1f}x more)")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke()
