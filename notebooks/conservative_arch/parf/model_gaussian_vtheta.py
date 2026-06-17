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

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke()
