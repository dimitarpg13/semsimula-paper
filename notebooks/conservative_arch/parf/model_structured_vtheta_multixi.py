"""
Multi-Xi adapter for structured V_theta parameterisations.

Wraps any StructuredVThetaBase subclass (which expects a single xi vector)
to accept the Multi-Xi interface used by ScalarPotentialLMSARFMassLNMultiXi:

    (xis: (B, T, K, d), h: (B, T, d))  ->  V: (B, T, 1)

The K xi-channels are flattened into a single (B, T, K*d) vector that serves
as the structured V_theta's 'xi' input.  The structured V_theta's own linear
projections learn to map this K*d-dimensional context summary to attractors,
precisions, and mixing weights.

Usage
-----
    from model_structured_vtheta import MixtureQuadraticVTheta
    from model_structured_vtheta_multixi import StructuredVThetaMultiXiAdapter

    inner = MixtureQuadraticVTheta(d=K*d_model, K=8, tau=1.0)
    V_theta = StructuredVThetaMultiXiAdapter(inner, K=4, d=256)
    model.V_theta = V_theta  # drop-in replacement
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_structured_vtheta import StructuredVThetaBase


class StructuredVThetaMultiXiAdapter(nn.Module):
    """Adapts a structured V_theta (single-xi interface) to the Multi-Xi
    V_theta interface: (xis: (B,T,K,d), h: (B,T,d)) -> V: (B,T,1).

    The K xi-channels are flattened and concatenated with h, then split
    so the structured V_theta sees xi_flat as its 'xi' input.
    This means the structured V_theta operates on dim = (K+1)*d total
    input, where the 'xi' portion is K*d and the 'h' portion is d.
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


def _smoke():
    """Quick validation of the Multi-Xi adapter."""
    from model_structured_vtheta import (
        MixtureQuadraticVTheta,
        validate_analytical_grad,
    )

    K_xi, d = 4, 16
    K_mix = 4
    xi_d = K_xi * d

    inner = MixtureQuadraticVTheta(d=xi_d, K=K_mix, tau=1.0)
    adapter = StructuredVThetaMultiXiAdapter(inner, K=K_xi, d=d)

    B, T = 2, 8
    xis = torch.randn(B, T, K_xi, d)
    h = torch.randn(B, T, d, requires_grad=True)

    V = adapter(xis, h)
    assert V.shape == (B, T, 1), f"Expected (B,T,1), got {V.shape}"

    grad = adapter.analytical_grad(xis, h)
    assert grad.shape == (B, T, d), f"Expected (B,T,d), got {grad.shape}"

    centres = adapter.attractor_centres(xis)
    assert centres.shape == (B, T, K_mix, xi_d), \
        f"Expected (B,T,K_mix,xi_d), got {centres.shape}"

    print(f"StructuredVThetaMultiXiAdapter smoke test passed.")
    print(f"  V shape: {tuple(V.shape)}")
    print(f"  grad shape: {tuple(grad.shape)}")
    print(f"  centres shape: {tuple(centres.shape)}")


if __name__ == "__main__":
    _smoke()
