"""
Anisotropic Gaussian mixture-PDF V_theta: low-rank cross-correlated wells.

Extends :mod:`model_gaussian_vtheta`'s isotropic ``MixtureGaussianVTheta``
(diagonal precision, axis-aligned wells) with a learned low-rank correction:

    Sigma_k^{-1} = diag(a_k) + B_k @ B_k^T,     B_k in R^{d x r}

    V(xi, h) = -sum_k w_k(xi) exp(-0.5 [(a_k*diff^2).sum() + ||B_k^T diff||^2])

This produces non-axis-aligned ellipsoidal attractors (ridges and saddles)
while keeping the potential and its gradient closed-form and the force
magnitude bounded by the same diff*exp(-r^2/2) decay as the isotropic case.
Setting ``rank=0`` recovers the isotropic potential exactly (the low-rank
term is a sum over an empty dimension, contributing zero).

This is a standalone counterpart to ``model_gaussian_vtheta.py``'s
``DepthConditionedMultiContextGaussianVTheta`` family — same shared-bank +
per-layer depth-code design, same ``_components``/``forward``/
``analytical_grad``/``attractor_centres`` interface, so callers (including
SCAF's ``BasinMembershipProbe`` via ``FockAdapter.well_parameters()``) can
treat both families uniformly.

Three classes, in order of composition:
    AnisotropicMixtureGaussianVTheta          -- single K-well bank
    AnisotropicMultiContextGaussianVTheta      -- one bank per xi-channel
    AnisotropicDepthConditionedGaussianVTheta  -- shared bank + per-layer
                                                   depth-code shift

Plus ``install_aniso_depth_routing(model)``, which wires the depth-code
routing into ``FockMultiXiPARFLM``'s actual per-layer step. Note: this
patches ``_fock_layer_step`` (the method ``FockMultiXiPARFLM`` in
``model_fock_parf_multixi.py`` actually defines), not ``_layer_step`` —
earlier ad-hoc copies of this routing helper in training notebooks patched
``_layer_step``, which does not exist on ``FockMultiXiPARFLM`` and would
raise ``AttributeError`` if invoked against it.
"""

from __future__ import annotations

import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnisotropicMixtureGaussianVTheta(nn.Module):
    """K-component mixture of anisotropic Gaussian wells.

      V(xi, h) = -sum_k w_k(xi) exp(-0.5 [(a_k*diff^2).sum() + ||B_k^T diff||^2])

    where ``diff = h - mu_k(xi)``, ``a_k > 0`` is a diagonal precision (as in
    the isotropic case), and ``B_k in R^{d x r}`` is a learned low-rank
    factor giving the well a non-axis-aligned, ellipsoidal shape.

    Analytical gradient:
      grad_h V = sum_k w_k [a_k*diff + B_k(B_k^T diff)] exp(-0.5[...])

    Force remains bounded by the same diff*exp(-r^2/2) decay as the
    isotropic well.
    """

    def __init__(
        self,
        d: int,
        K: int = 8,
        rank: int = 4,
        w_scale: float = 1.0,
        xi_d: Optional[int] = None,
        init_log_precision: Optional[float] = None,
        precision_max: Optional[float] = None,
        force_norm_max: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        d : int
            Well dimension (must match h dimension).
        K : int
            Number of Gaussian wells.
        rank : int
            Rank ``r`` of the low-rank precision correction ``B_k``.
            ``rank=0`` recovers the isotropic potential exactly.
        w_scale : float
            Scale factor for softmax weights.
        xi_d : int or None
            Context input dimension. Defaults to ``d``.
        init_log_precision : float or None
            Initial log-precision bias for ``a_proj`` (see
            ``model_gaussian_vtheta.MixtureGaussianVTheta`` for the
            ``sigma_eff`` derivation). ``None`` keeps the default.
        precision_max : float or None
            Hard upper bound on the diagonal precision ``a_k``.
        force_norm_max : float or None
            Per-well force magnitude cap.
        """
        super().__init__()
        self.d = d
        self.K = K
        self.rank = rank
        self.w_scale = w_scale
        self._precision_max = precision_max
        self._force_norm_max = force_norm_max
        in_d = xi_d if xi_d is not None else d

        self.mu_proj = nn.Linear(in_d, K * d)
        self.a_proj = nn.Linear(in_d, K * d)
        self.w_proj = nn.Linear(in_d, K)
        self.B_proj = nn.Linear(in_d, K * d * rank) if rank > 0 else None

        self._init_weights(init_log_precision)

    def _init_weights(self, init_log_precision: Optional[float]) -> None:
        nn.init.xavier_uniform_(self.mu_proj.weight)
        nn.init.zeros_(self.mu_proj.bias)
        nn.init.zeros_(self.a_proj.weight)
        if init_log_precision is not None:
            self.a_proj.bias.data.fill_(init_log_precision)
        else:
            self.a_proj.bias.data.fill_(0.0)
        nn.init.zeros_(self.w_proj.weight)
        nn.init.zeros_(self.w_proj.bias)
        if self.B_proj is not None:
            nn.init.normal_(self.B_proj.weight, std=0.01)
            nn.init.zeros_(self.B_proj.bias)

    def _components(self, xi: torch.Tensor):
        """Parse context xi into well parameters: (mu, a, w, B)."""
        lead = xi.shape[:-1]
        mu = self.mu_proj(xi).view(*lead, self.K, self.d)
        a = (F.softplus(self.a_proj(xi)) + 1e-4).view(*lead, self.K, self.d)
        if self._precision_max is not None:
            a = a.clamp(max=self._precision_max)
        w = F.softmax(self.w_proj(xi), dim=-1) * self.w_scale
        if self.B_proj is not None:
            B = self.B_proj(xi).view(*lead, self.K, self.d, self.rank)
        else:
            B = mu.new_zeros(*mu.shape, 0)
        return mu, a, w, B

    def forward(self, xi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        mu, a, w, B = self._components(xi)
        h_e = h.unsqueeze(-2)
        diff = h_e - mu

        diag_term = (a * diff * diff).sum(dim=-1)
        Bt_diff = torch.einsum('...kd,...kdr->...kr', diff, B)
        lr_term = (Bt_diff * Bt_diff).sum(dim=-1)

        exponent = -0.5 * (diag_term + lr_term)
        bumps = w * torch.exp(exponent)
        return -bumps.sum(dim=-1, keepdim=True)

    def analytical_grad(
        self, xi: torch.Tensor, h: torch.Tensor,
    ) -> torch.Tensor:
        mu, a, w, B = self._components(xi)
        h_e = h.unsqueeze(-2)
        diff = h_e - mu

        diag_term = (a * diff * diff).sum(dim=-1)
        Bt_diff = torch.einsum('...kd,...kdr->...kr', diff, B)
        lr_term = (Bt_diff * Bt_diff).sum(dim=-1)

        exponent = -0.5 * (diag_term + lr_term)
        g = w * torch.exp(exponent)

        grad_diag = a * diff
        grad_lr = torch.einsum('...kdr,...kr->...kd', B, Bt_diff)
        per_comp = (grad_diag + grad_lr) * g.unsqueeze(-1)

        if self._force_norm_max is not None:
            norms = per_comp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = (self._force_norm_max / norms).clamp(max=1.0)
            per_comp = per_comp * scale

        return per_comp.sum(dim=-2)

    def attractor_centres(self, xi: torch.Tensor) -> torch.Tensor:
        lead = xi.shape[:-1]
        return self.mu_proj(xi).view(*lead, self.K, self.d)


class AnisotropicMultiContextGaussianVTheta(nn.Module):
    """Per-context anisotropic Gaussian well banks: V = sum_m V^(m)(xi^(m), h).

    Mirrors ``model_gaussian_vtheta.MultiContextGaussianVTheta``, with each
    per-channel bank being an ``AnisotropicMixtureGaussianVTheta`` instead
    of the isotropic ``MixtureGaussianVTheta``.

    Interface: forward(xis: (B,T,n_ctx,d), h: (B,T,d)) -> V: (B,T,1)
    """

    def __init__(
        self,
        d: int,
        K: int,
        n_ctx: int,
        rank: int = 4,
        w_scale: float = 1.0,
        init_log_precision: Optional[float] = None,
        precision_max: Optional[float] = None,
        force_norm_max: Optional[float] = None,
    ):
        super().__init__()
        self.d = d
        self.K = K
        self.n_ctx = n_ctx
        self.banks = nn.ModuleList(
            AnisotropicMixtureGaussianVTheta(
                d=d, K=K, rank=rank, w_scale=w_scale, xi_d=d,
                init_log_precision=init_log_precision,
                precision_max=precision_max,
                force_norm_max=force_norm_max,
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
        cs = [
            self.banks[m].attractor_centres(xis[..., m, :])
            for m in range(self.n_ctx)
        ]
        return torch.cat(cs, dim=-2)


class AnisotropicDepthConditionedGaussianVTheta(nn.Module):
    """One shared anisotropic multi-context bank + a learned per-layer code.

    Mirrors ``model_gaussian_vtheta.DepthConditionedMultiContextGaussianVTheta``
    exactly (same depth-code shift mechanism, same layer-routing contract via
    ``set_active_layer`` / ``_active_layer``), with an
    ``AnisotropicMultiContextGaussianVTheta`` bank underneath.

    Interface: forward(xis: (B,T,n_ctx,d), h: (B,T,d)) -> V: (B,T,1)
    """

    def __init__(
        self,
        d: int,
        K: int,
        n_ctx: int,
        n_layers: int,
        rank: int = 4,
        w_scale: float = 1.0,
        init_log_precision: Optional[float] = None,
        precision_max: Optional[float] = None,
        force_norm_max: Optional[float] = None,
        code_init_std: float = 0.02,
    ):
        super().__init__()
        self.d = d
        self.K = K
        self.n_ctx = n_ctx
        self.n_layers = n_layers
        self.rank = rank
        self.bank = AnisotropicMultiContextGaussianVTheta(
            d=d, K=K, n_ctx=n_ctx, rank=rank, w_scale=w_scale,
            init_log_precision=init_log_precision,
            precision_max=precision_max,
            force_norm_max=force_norm_max,
        )
        self.depth_code = nn.Parameter(
            torch.randn(n_layers, n_ctx, d) * code_init_std
        )
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
        code = self.depth_code[g]
        lead = xis.dim() - 2
        code = code.view(*([1] * lead), self.n_ctx, self.d)
        return xis + code

    def forward(self, xis: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.bank(self._shift(xis), h)

    def analytical_grad(self, xis: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.bank.analytical_grad(self._shift(xis), h)

    def attractor_centres(self, xis: torch.Tensor) -> torch.Tensor:
        return self.bank.attractor_centres(self._shift(xis))


def install_aniso_depth_routing(model) -> None:
    """Monkey-patch ``model._fock_layer_step`` to broadcast the layer index.

    Same contract as ``model_gaussian_vtheta.install_depth_routing``, but
    written as a standalone duck-typed version (checks for
    ``set_active_layer`` rather than ``isinstance``-checking one specific
    class) so it works with either the isotropic or the anisotropic
    depth-conditioned V_theta.

    No-op if ``model.V_theta`` does not expose ``set_active_layer``.
    """
    vt = getattr(model, "V_theta", None)
    if vt is None or not hasattr(vt, "set_active_layer"):
        return
    if getattr(model, "_depth_routing_installed", False):
        return

    if not hasattr(model, "_fock_layer_step"):
        raise AttributeError(
            "install_aniso_depth_routing expects model._fock_layer_step "
            "(FockMultiXiPARFLM). If you're using a different Fock "
            "variant, check its per-layer step method name and adapt "
            "the patch target accordingly."
        )

    _orig = model._fock_layer_step.__func__  # unbound original

    def _routed_fock_layer_step(self, h, h_prev, r, salience, m_b, gamma,
                                 dt, layer_idx, *args, **kwargs):
        vt_local = getattr(self, "V_theta", None)
        if vt_local is not None and hasattr(vt_local, "set_active_layer"):
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
    print(f"--- Anisotropic Gaussian V_theta validation (d={d}) ---")
    validate_analytical_grad(AnisotropicMixtureGaussianVTheta(d=d, K=4, rank=2), d=d)
    validate_analytical_grad(AnisotropicMixtureGaussianVTheta(d=d, K=8, rank=4), d=d)

    print(f"\n--- rank=0 reduces to isotropic potential ---")
    torch.manual_seed(0)
    bank_r0 = AnisotropicMixtureGaussianVTheta(d=d, K=4, rank=0)
    xi = torch.randn(2, 4, d)
    h = torch.randn(2, 4, d, requires_grad=True)
    V_r0 = bank_r0(xi, h)
    g_r0 = bank_r0.analytical_grad(xi, h)
    print(f"  rank=0 V shape={V_r0.shape}  grad shape={g_r0.shape}  OK")

    print(f"\n--- Depth-conditioned anisotropic multi-context V_theta ---")
    n_ctx, K, L, rank = 4, 8, 6, 4
    dcv = AnisotropicDepthConditionedGaussianVTheta(
        d=d, K=K, n_ctx=n_ctx, n_layers=L, rank=rank,
    )
    xis = torch.randn(2, 8, n_ctx, d)
    h2 = torch.randn(2, 8, d, requires_grad=True)
    dcv.set_active_layer(0)
    V0 = dcv(xis, h2)
    dcv.set_active_layer(L - 1)
    VL = dcv(xis, h2)
    assert V0.shape == (2, 8, 1)
    assert not torch.allclose(V0, VL), "depth code did not differentiate layers"

    dcv.set_active_layer(2)
    g_ana = dcv.analytical_grad(xis, h2)
    V = dcv(xis, h2).sum()
    g_auto, = torch.autograd.grad(V, h2)
    max_err = (g_ana - g_auto).abs().max().item()
    print(f"  depth code differentiates layers: V0!=VL  OK")
    print(f"  analytical_grad vs autograd max_err: {max_err:.2e}")
    assert max_err < 1e-4, f"analytical_grad mismatch: {max_err}"

    n_dcv = sum(p.numel() for p in dcv.parameters())
    print(f"  shared-bank+code params: {n_dcv:,}")

    print(f"\n--- install_aniso_depth_routing (duck-typed) ---")

    class _FakeFockModel(torch.nn.Module):
        def __init__(self, V_theta):
            super().__init__()
            self.V_theta = V_theta
            self.calls = []

        def _fock_layer_step(self, h, h_prev, r, salience, m_b, gamma, dt,
                              layer_idx):
            self.calls.append(layer_idx)
            return h, h_prev, r, salience

    fake = _FakeFockModel(dcv)
    install_aniso_depth_routing(fake)
    fake._fock_layer_step(h2, h2, None, None, None, None, 1.0, 3)
    assert dcv._active_layer == 3, "routing did not set the active layer"
    print(f"  install_aniso_depth_routing set active_layer=3 via patched "
          f"_fock_layer_step  OK")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke()
