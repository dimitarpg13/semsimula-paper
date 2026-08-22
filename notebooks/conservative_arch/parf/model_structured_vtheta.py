"""
Structured V_theta parameterisations with analytical gradients
(Option 5 of the autograd.grad cost-reduction plan).

Rationale
---------
The current PARFLM / FockPARFLM / SPLM V_theta is an unrestricted MLP
(xi, h) in R^(2d) -> scalar V.  Computing the force f = -grad_h V_theta
requires either torch.autograd.grad (~ 2x the forward cost) or a
manually-implemented backward.  With V_theta regularisation active,
the learned V_theta is empirically near-constant (range 0.26-20 across
all regularised cells), suggesting that a low-capacity *structured*
parameterisation should suffice -- and would admit a closed-form
gradient evaluable in a single matvec.

Four variants are provided, each implementing the same interface as
the standard ScalarPotential MLP plus an additional `analytical_grad`
method that returns grad_h V *without* invoking autograd:

  QuadraticWellVTheta          -- diagonal precision, 1 attractor
                                  per context.  Cheapest, simplest.
  LowRankQuadraticVTheta       -- low-rank + diagonal precision,
                                  1 attractor with off-diagonal
                                  correlations.
  MixtureQuadraticVTheta       -- K-component mixture of diagonal
                                  quadratic wells, soft-mixed via
                                  log-sum-exp.  K attractors per
                                  context; matches the K* = 4 basin
                                  structure observed in the PR2 PARF
                                  regularisation sweep.
  HybridQuadraticVTheta        -- quadratic backbone + small MLP
                                  residual.  Safety net for cases
                                  where pure quadratic underfits.

All four classes return the per-token scalar V with shape (..., 1)
from `forward(xi, h)` and the per-token gradient with shape (..., d)
from `analytical_grad(xi, h)`.  Both methods are first-class
differentiable PyTorch tensors -- training-time parameter gradients
flow through them via the cross-entropy chain.

Interpretability bonus
----------------------
Attractor centres are *explicit* in these parameterisations:
  - QuadraticWell / LowRankQuadratic: the attractor is mu(xi).
  - Mixture: the K attractors are mu_k(xi).
No 1500-step gradient-descent extraction is required -- the basin
structure is read directly from the model parameters.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class StructuredVThetaBase(nn.Module):
    """Base class for structured V_theta with analytical gradient.

    Subclasses MUST implement both `forward(xi, h)` returning V with
    shape (..., 1), and `analytical_grad(xi, h)` returning grad_h V
    with shape (..., d).  Both must be differentiable through xi, h,
    and self.parameters() for training-time backpropagation.
    """

    def forward(self, xi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def analytical_grad(self, xi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def attractor_centres(self, xi: torch.Tensor) -> torch.Tensor:
        """Return the analytical attractor centres for the given xi.

        Shape is (..., K, d) where K is the number of attractors
        (K=1 for single-well variants, K=cfg.K for mixtures).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1.  Diagonal quadratic well
# ---------------------------------------------------------------------------
class QuadraticWellVTheta(StructuredVThetaBase):
    """Diagonal quadratic potential, ξ-conditioned.

      V(xi, h) = 0.5 * a(xi)^T (h - mu(xi))^2 + b(xi)
      grad_h V = a(xi) ⊙ (h - mu(xi))

    Single attractor per context at mu(xi); precision a(xi) > 0 is
    diagonal.  Cheapest variant: 3 linear projections (xi -> d, d, 1).
    """

    def __init__(self, d: int, init_a_bias: float = 0.0):
        super().__init__()
        self.d = d
        self.mu_proj = nn.Linear(d, d)
        self.a_proj = nn.Linear(d, d)
        self.b_proj = nn.Linear(d, 1)
        self._init_weights(init_a_bias)

    def _init_weights(self, init_a_bias: float) -> None:
        nn.init.normal_(self.mu_proj.weight, std=0.02)
        nn.init.zeros_(self.mu_proj.bias)
        nn.init.normal_(self.a_proj.weight, std=0.02)
        nn.init.constant_(self.a_proj.bias, init_a_bias)
        nn.init.normal_(self.b_proj.weight, std=0.002)
        nn.init.zeros_(self.b_proj.bias)

    def _mu(self, xi: torch.Tensor) -> torch.Tensor:
        return self.mu_proj(xi)

    def _a(self, xi: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.a_proj(xi)) + 1e-4

    def forward(self, xi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        mu = self._mu(xi)
        a = self._a(xi)
        diff = h - mu
        quad = 0.5 * (a * diff * diff).sum(dim=-1, keepdim=True)
        return quad + self.b_proj(xi)

    def analytical_grad(self, xi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        mu = self._mu(xi)
        a = self._a(xi)
        return a * (h - mu)

    def harmonic_terms(
        self, xi: torch.Tensor, h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact global harmonic model.

        This potential already *is* a single diagonal quadratic well, so
        unlike the Gaussian-bump families there is no local-linearisation
        approximation here: ``k_diag`` and ``s`` reproduce ``-grad_h V``
        for *every* ``h``, not just at the point it was evaluated at.
        """
        mu = self._mu(xi)
        a = self._a(xi)
        return a, a * mu

    def attractor_centres(self, xi: torch.Tensor) -> torch.Tensor:
        return self._mu(xi).unsqueeze(-2)


# ---------------------------------------------------------------------------
# 2.  Low-rank + diagonal quadratic well
# ---------------------------------------------------------------------------
class LowRankQuadraticVTheta(StructuredVThetaBase):
    """Low-rank-plus-diagonal quadratic potential, ξ-conditioned.

      A(xi) = U(xi) U(xi)^T + diag(lambda(xi)),   U in R^{d x r}
      V(xi, h) = 0.5 (h - mu)^T A (h - mu) + b(xi)
               = 0.5 ||U^T (h - mu)||^2 + 0.5 lambda^T (h - mu)^2 + b(xi)
      grad_h V = U (U^T (h - mu)) + lambda ⊙ (h - mu)

    Captures off-diagonal correlations at controllable rank r << d.
    """

    def __init__(self, d: int, rank: int = 4, init_a_bias: float = 0.0):
        super().__init__()
        self.d = d
        self.rank = rank
        self.mu_proj = nn.Linear(d, d)
        self.lam_proj = nn.Linear(d, d)
        self.U_proj = nn.Linear(d, d * rank)
        self.b_proj = nn.Linear(d, 1)
        self._init_weights(init_a_bias)

    def _init_weights(self, init_a_bias: float) -> None:
        nn.init.normal_(self.mu_proj.weight, std=0.02)
        nn.init.zeros_(self.mu_proj.bias)
        nn.init.normal_(self.lam_proj.weight, std=0.02)
        nn.init.constant_(self.lam_proj.bias, init_a_bias)
        nn.init.normal_(self.U_proj.weight, std=0.01)
        nn.init.zeros_(self.U_proj.bias)
        nn.init.normal_(self.b_proj.weight, std=0.002)
        nn.init.zeros_(self.b_proj.bias)

    def _mu(self, xi: torch.Tensor) -> torch.Tensor:
        return self.mu_proj(xi)

    def _lam(self, xi: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.lam_proj(xi)) + 1e-4

    def _U(self, xi: torch.Tensor) -> torch.Tensor:
        u_flat = self.U_proj(xi)
        return u_flat.view(*xi.shape[:-1], self.d, self.rank)

    def forward(self, xi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        mu = self._mu(xi)
        lam = self._lam(xi)
        U = self._U(xi)
        diff = h - mu
        ut_diff = torch.einsum('...dr,...d->...r', U, diff)
        qf = (ut_diff * ut_diff).sum(dim=-1, keepdim=True)
        qf = qf + (lam * diff * diff).sum(dim=-1, keepdim=True)
        return 0.5 * qf + self.b_proj(xi)

    def analytical_grad(
        self, xi: torch.Tensor, h: torch.Tensor,
    ) -> torch.Tensor:
        mu = self._mu(xi)
        lam = self._lam(xi)
        U = self._U(xi)
        diff = h - mu
        ut_diff = torch.einsum('...dr,...d->...r', U, diff)
        u_ut_diff = torch.einsum('...dr,...r->...d', U, ut_diff)
        return u_ut_diff + lam * diff

    def harmonic_terms(
        self, xi: torch.Tensor, h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Diagonal part of this well's precision ``A = U U^T + diag(lam)``.

        Exact for the diagonal contribution (``diag(A) = lam +
        rowsum(U^2)``); the residual is the off-diagonal ``U U^T``
        coupling -- exactly analogous to the anisotropic Gaussian
        family's ``rank > 0`` residual
        (``model_aniso_gaussian_vtheta.AnisotropicMixtureGaussianVTheta.harmonic_terms``).
        """
        mu = self._mu(xi)
        lam = self._lam(xi)
        U = self._U(xi)
        k_diag = lam + (U * U).sum(dim=-1)
        return k_diag, k_diag * mu

    def attractor_centres(self, xi: torch.Tensor) -> torch.Tensor:
        return self._mu(xi).unsqueeze(-2)


# ---------------------------------------------------------------------------
# 3.  Mixture of K diagonal quadratic wells
# ---------------------------------------------------------------------------
class MixtureQuadraticVTheta(StructuredVThetaBase):
    """K-component soft mixture of diagonal quadratic wells.

      E_k(xi, h) = 0.5 a_k(xi)^T (h - mu_k(xi))^2     (per-component well)
      V(xi, h)   = -tau * logsumexp_k(-E_k(xi, h)/tau + log pi_k(xi))
                   + b(xi)

    The K basins are the K minima of E_k; V(xi, h) approaches min_k E_k
    as tau -> 0 and the mean of E_k as tau -> infty.  At tau = 1
    (default) V is the negative log marginal likelihood of a K-Gaussian
    mixture (up to a constant), so the framework recovers the
    Gaussian-mixture motivation of Sec. 4 of the paper in a
    xi-conditioned form.

    Analytical gradient:
      q_k(xi, h) = softmax_k(-E_k/tau + log pi_k)        (responsibilities)
      grad_h V    = sum_k q_k * grad_h E_k
                  = sum_k q_k * a_k(xi) ⊙ (h - mu_k(xi))

    The K attractor centres mu_k(xi) are directly readable from the
    model parameters -- no GD extraction needed for interpretability.
    """

    def __init__(self, d: int, K: int = 4, tau: float = 1.0,
                 init_a_bias: float = 0.0, xi_d: int | None = None):
        super().__init__()
        self.d = d
        self.K = K
        self.tau = tau
        in_d = xi_d if xi_d is not None else d
        self.mu_proj = nn.Linear(in_d, K * d)
        self.a_proj = nn.Linear(in_d, K * d)
        self.pi_proj = nn.Linear(in_d, K)
        self.b_proj = nn.Linear(in_d, 1)
        self._init_weights(init_a_bias)

    def _init_weights(self, init_a_bias: float) -> None:
        nn.init.normal_(self.mu_proj.weight, std=0.02)
        nn.init.zeros_(self.mu_proj.bias)
        nn.init.normal_(self.a_proj.weight, std=0.02)
        nn.init.constant_(self.a_proj.bias, init_a_bias)
        nn.init.normal_(self.pi_proj.weight, std=0.02)
        nn.init.zeros_(self.pi_proj.bias)
        nn.init.normal_(self.b_proj.weight, std=0.002)
        nn.init.zeros_(self.b_proj.bias)

    def _components(self, xi: torch.Tensor):
        lead = xi.shape[:-1]
        mu = self.mu_proj(xi).view(*lead, self.K, self.d)
        a = (F.softplus(self.a_proj(xi)) + 1e-4).view(*lead, self.K, self.d)
        log_pi = F.log_softmax(self.pi_proj(xi), dim=-1)
        return mu, a, log_pi

    def forward(self, xi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        mu, a, log_pi = self._components(xi)
        h_e = h.unsqueeze(-2)
        diff = h_e - mu
        E = 0.5 * (a * diff * diff).sum(dim=-1)
        log_terms = -E / self.tau + log_pi
        log_marg = torch.logsumexp(log_terms, dim=-1, keepdim=True)
        return -self.tau * log_marg + self.b_proj(xi)

    def analytical_grad(
        self, xi: torch.Tensor, h: torch.Tensor,
    ) -> torch.Tensor:
        mu, a, log_pi = self._components(xi)
        h_e = h.unsqueeze(-2)
        diff = h_e - mu
        E = 0.5 * (a * diff * diff).sum(dim=-1)
        log_terms = -E / self.tau + log_pi
        q = F.softmax(log_terms, dim=-1).unsqueeze(-1)
        per_comp_grad = a * diff
        return (q * per_comp_grad).sum(dim=-2)

    def harmonic_terms(
        self, xi: torch.Tensor, h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact linearisation of the mixture force at ``h``.

        Like the Gaussian-mixture families, the gradient is a convex
        (softmax) combination of per-well linear springs, so ``k_diag``/
        ``s`` reproduce ``-grad_h V`` exactly at ``h``. Unlike the
        anisotropic Gaussian families this mixture has no low-rank term,
        so there is no residual at all -- this is exact everywhere the
        Gaussian ``rank=0`` case is, for the same reason.
        """
        mu, a, log_pi = self._components(xi)
        h_e = h.unsqueeze(-2)
        diff = h_e - mu
        E = 0.5 * (a * diff * diff).sum(dim=-1)
        q = F.softmax(-E / self.tau + log_pi, dim=-1).unsqueeze(-1)
        qa = q * a                                                # (...,K,d)
        k_diag = qa.sum(dim=-2)
        s = (qa * mu).sum(dim=-2)
        return k_diag, s

    def attractor_centres(self, xi: torch.Tensor) -> torch.Tensor:
        lead = xi.shape[:-1]
        return self.mu_proj(xi).view(*lead, self.K, self.d)


# ---------------------------------------------------------------------------
# 4.  Hybrid quadratic backbone + small MLP residual
# ---------------------------------------------------------------------------
class HybridQuadraticVTheta(StructuredVThetaBase):
    """Quadratic backbone + small MLP correction.

      V(xi, h) = V_quad(xi, h) + alpha * V_MLP(xi, h)

    The backbone gives the analytical gradient; the MLP provides a
    learnable correction.  alpha is a learnable scalar (initialised
    small) that can be optionally regularised toward 0 by an external
    penalty.

    The MLP portion still requires autograd for its gradient.  The
    overall speedup is partial: the backbone is free, but the MLP
    contributes a (small) autograd cost.  Use this variant when the
    pure quadratic underfits and a small flexible correction is
    needed.
    """

    def __init__(
        self,
        d: int,
        v_hidden: int = 32,
        v_depth: int = 2,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.quad = QuadraticWellVTheta(d)
        layers = [nn.Linear(2 * d, v_hidden), nn.GELU()]
        for _ in range(v_depth - 1):
            layers += [nn.Linear(v_hidden, v_hidden), nn.GELU()]
        layers += [nn.Linear(v_hidden, 1)]
        self.mlp = nn.Sequential(*layers)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(last.weight, std=0.002)
        nn.init.zeros_(last.bias)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, xi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        v_q = self.quad(xi, h)
        v_m = self.mlp(torch.cat([xi, h], dim=-1))
        return v_q + self.alpha * v_m

    def analytical_grad(
        self, xi: torch.Tensor, h: torch.Tensor,
    ) -> torch.Tensor:
        """Combines analytical quadratic gradient with autograd MLP gradient.

        Note: the MLP portion still uses autograd internally because no
        analytical gradient is available for an arbitrary MLP.  Use a
        small v_hidden to keep the cost low.
        """
        grad_quad = self.quad.analytical_grad(xi, h)
        h_in = h if h.requires_grad else h.detach().requires_grad_(True)
        v_mlp = self.mlp(torch.cat([xi, h_in], dim=-1)).sum()
        grad_mlp, = torch.autograd.grad(
            v_mlp, h_in, create_graph=self.training, retain_graph=True,
        )
        return grad_quad + self.alpha * grad_mlp

    def harmonic_terms(
        self, xi: torch.Tensor, h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backbone-only linearisation.

        This is the quadratic backbone's harmonic model, **not** an exact
        linearisation of the full potential: the ``alpha * MLP`` residual
        has no closed-form curvature and is not included. Any stiffness
        read off this for a Hybrid checkpoint is therefore a lower bound,
        not a ceiling -- the true ``omega*dt`` including the MLP's
        contribution could be higher than what this reports.
        """
        return self.quad.harmonic_terms(xi, h)

    def attractor_centres(self, xi: torch.Tensor) -> torch.Tensor:
        return self.quad.attractor_centres(xi)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def _max_abs(t: torch.Tensor) -> float:
    return float(t.abs().max().item())


def validate_analytical_grad(
    module: StructuredVThetaBase,
    d: int,
    batch_shape: tuple = (3, 5),
    seed: int = 0,
    tol: float = 1e-5,
    device=None,
) -> tuple:
    """Compare analytical_grad against torch.autograd.grad on V.sum().

    Returns (max_abs_diff, max_rel_diff).  Both should be ~< 1e-5
    for the structured variants (floating-point noise) and ~< 1e-4
    for the hybrid (autograd through the MLP).

    `device` defaults to the device of the module's first parameter so that
    the validation tensors are always co-located with the module weights.
    """
    if device is None:
        try:
            device = next(module.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
    torch.manual_seed(seed)
    xi = torch.randn(*batch_shape, d, device=device)
    h = torch.randn(*batch_shape, d, requires_grad=True, device=device)
    V = module(xi, h)
    assert V.shape == (*batch_shape, 1), \
        f"Expected V shape {(*batch_shape, 1)}, got {V.shape}"
    grad_auto, = torch.autograd.grad(V.sum(), h, create_graph=False)
    grad_ana = module.analytical_grad(xi, h)
    assert grad_ana.shape == h.shape, \
        f"Expected grad shape {h.shape}, got {grad_ana.shape}"
    diff = grad_auto - grad_ana
    max_abs = _max_abs(diff)
    denom = grad_auto.abs().clamp_min(1e-8)
    max_rel = float((diff.abs() / denom).max().item())
    status = "OK" if max_abs < tol else "FAIL"
    print(
        f"[{type(module).__name__:<32}] "
        f"max_abs_diff = {max_abs:.3e}  "
        f"max_rel_diff = {max_rel:.3e}  "
        f"[{status}]"
    )
    return max_abs, max_rel


def _smoke():
    """Validate all four structured V_theta classes against autograd."""
    d = 16
    print(f"--- Structured V_theta validation (d={d}) ---")
    validate_analytical_grad(QuadraticWellVTheta(d=d), d=d)
    validate_analytical_grad(LowRankQuadraticVTheta(d=d, rank=4), d=d)
    validate_analytical_grad(MixtureQuadraticVTheta(d=d, K=4), d=d)
    validate_analytical_grad(MixtureQuadraticVTheta(d=d, K=8, tau=0.5), d=d)
    validate_analytical_grad(
        HybridQuadraticVTheta(d=d, v_hidden=32, v_depth=2), d=d, tol=1e-4,
    )
    print()
    print("--- harmonic_terms: exact linearisation (quadratic family) ---")
    torch.manual_seed(0)
    xi_h = torch.randn(3, 5, d)
    h_h = torch.randn(3, 5, d)

    for name, mod in [
        ("QuadraticWellVTheta", QuadraticWellVTheta(d=d)),
        ("MixtureQuadraticVTheta(K=6)", MixtureQuadraticVTheta(d=d, K=6)),
    ]:
        k_diag, s = mod.harmonic_terms(xi_h, h_h)
        f_harm = s - k_diag * h_h
        f_true = -mod.analytical_grad(xi_h, h_h)
        err = (f_harm - f_true).abs().max().item()
        print(f"  {name:<32}: |f_harm - f_true|_max = {err:.2e}  (must be ~0)")
        assert err < 1e-5, err
        assert (k_diag >= 0).all(), f"{name}: stiffness must be non-negative"

    # rank>0 has a genuine off-diagonal U U^T residual by construction
    # (documented in harmonic_terms' docstring), so it is NOT exact here --
    # only report it, don't assert exactness.
    lr4 = LowRankQuadraticVTheta(d=d, rank=4)
    k_diag, s = lr4.harmonic_terms(xi_h, h_h)
    f_harm = s - k_diag * h_h
    f_true = -lr4.analytical_grad(xi_h, h_h)
    resid4 = (f_true - f_harm).abs().max().item()
    print(f"  {'LowRankQuadraticVTheta(rank=4)':<32}: residual "
          f"(off-diagonal U U^T) max = {resid4:.2e}  (nonzero by design)")
    assert (k_diag >= 0).all()

    # LowRankQuadraticVTheta at rank=0 has no off-diagonal coupling either,
    # so it should also be globally exact like the diagonal-only variants.
    lr0 = LowRankQuadraticVTheta(d=d, rank=0)
    k_diag, s = lr0.harmonic_terms(xi_h, h_h)
    f_harm = s - k_diag * h_h
    f_true = -lr0.analytical_grad(xi_h, h_h)
    err0 = (f_harm - f_true).abs().max().item()
    print(f"  {'LowRankQuadraticVTheta(rank=0)':<32}: "
          f"|f_harm - f_true|_max = {err0:.2e}  (must be ~0)")
    assert err0 < 1e-5, err0

    hyb = HybridQuadraticVTheta(d=d, v_hidden=32, v_depth=2)
    k_diag, s = hyb.harmonic_terms(xi_h, h_h)
    f_harm = s - k_diag * h_h
    f_true = -hyb.analytical_grad(xi_h, h_h)
    resid = (f_true - f_harm).abs().max().item()
    print(f"  {'HybridQuadraticVTheta':<32}: residual (MLP contribution) "
          f"max = {resid:.2e}  (nonzero by design -- backbone-only, see "
          f"docstring)")
    assert (k_diag >= 0).all()
    print(f"  harmonic_terms available for the whole structured-V_theta "
          f"family: OK (needed by scaf_checkpoint_analysis.ipynb's "
          f"stiffness audit for SQ1-4 recipes)")

    print()
    print("--- Attractor centres demo (Mixture K=3) ---")
    torch.manual_seed(0)
    mod = MixtureQuadraticVTheta(d=d, K=3)
    xi = torch.randn(2, 4, d)
    centres = mod.attractor_centres(xi)
    print(f"  xi shape:           {tuple(xi.shape)}")
    print(f"  centres shape:      {tuple(centres.shape)}  (expected (2,4,3,d))")
    print(f"  centres[0,0,:,:3]:  {centres[0, 0, :, :3].tolist()}")
    print()
    print("--- Parameter counts at d=128, MLP baseline at v_hidden=128 ---")
    d = 128
    baseline_mlp = nn.Sequential(
        nn.Linear(2 * d, 128), nn.GELU(),
        nn.Linear(128, 128), nn.GELU(),
        nn.Linear(128, 128), nn.GELU(),
        nn.Linear(128, 1),
    )
    print(f"  MLP baseline (v_hidden=128, v_depth=3): "
          f"{sum(p.numel() for p in baseline_mlp.parameters()):>8,} params")
    for cls, kw in [
        (QuadraticWellVTheta, dict(d=d)),
        (LowRankQuadraticVTheta, dict(d=d, rank=8)),
        (MixtureQuadraticVTheta, dict(d=d, K=4)),
        (MixtureQuadraticVTheta, dict(d=d, K=8)),
        (HybridQuadraticVTheta, dict(d=d, v_hidden=32, v_depth=2)),
    ]:
        m = cls(**kw)
        n = sum(p.numel() for p in m.parameters())
        name = f"{cls.__name__}({', '.join(f'{k}={v}' for k, v in kw.items() if k != 'd')})"
        print(f"  {name:<48}: {n:>8,} params")


if __name__ == "__main__":
    _smoke()
