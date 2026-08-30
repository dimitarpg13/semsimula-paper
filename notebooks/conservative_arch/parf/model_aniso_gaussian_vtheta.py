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
        precision_lr_max: Optional[float] = None,
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
        precision_lr_max : float or None
            Smooth upper bound on the low-rank precision contribution:
            caps the curvature added by the off-diagonal factor at
            ``sigma_max(B_k)^2 <= precision_lr_max``. Implemented as a
            differentiable Frobenius-norm cap on ``B_k`` (mitigation
            "#2 / smooth bounded curvature" of the CfC/BAOAB companion
            note): since ``sigma_max(B_k) <= ||B_k||_F``, bounding
            ``||B_k||_F <= sqrt(precision_lr_max)`` guarantees the
            spectral bound. ``None`` (default) leaves ``B_k`` unbounded
            (current behaviour). The bound is conservative by up to a
            factor ``rank`` when the low-rank energy is spread across
            singular values; tune it against the SCAF Phase 7b/7c Weyl
            audit rather than as a literal ``sigma_max^2`` target.
        force_norm_max : float or None
            Per-well force magnitude cap.
        """
        super().__init__()
        self.d = d
        self.K = K
        self.rank = rank
        self.w_scale = w_scale
        self._precision_max = precision_max
        self._precision_lr_max = precision_lr_max
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
            B = self._bound_lowrank(B)
        else:
            B = mu.new_zeros(*mu.shape, 0)
        return mu, a, w, B

    def _bound_lowrank(self, B: torch.Tensor) -> torch.Tensor:
        """Smoothly cap the low-rank factor's spectral norm (mitigation #2).

        Bounds ``sigma_max(B_k)^2 <= precision_lr_max`` by capping the
        Frobenius norm of each well's factor at ``sqrt(precision_lr_max)``
        with a ``tanh`` soft cap (identity for small norms, asymptotic to
        the budget for large ones, strictly below it everywhere). Because
        ``sigma_max(B_k) <= ||B_k||_F`` the spectral bound is guaranteed;
        the cap is differentiable and never divides by zero.

        No-op (returns ``B`` unchanged) when ``precision_lr_max`` is None
        or ``rank == 0``.

        ``B`` : (..., K, d, rank)
        """
        if self._precision_lr_max is None or self.rank == 0:
            return B
        budget = self._precision_lr_max ** 0.5
        fro = B.flatten(-2, -1).norm(dim=-1).clamp(min=1e-12)     # (..., K)
        scale = budget * torch.tanh(fro / budget) / fro          # (..., K)
        return B * scale.unsqueeze(-1).unsqueeze(-1)

    def context_components(self, xi: torch.Tensor):
        """Well parameters for ``xi``, reusable across several ``h``.

        The well parameters depend only on the context, so a caller that
        evaluates this bank at more than one position for the *same* xi
        (the CfC integrator does: ``harmonic_terms`` at h and the force
        at the drifted h_mid) can compute them once and pass them back
        in via the ``comps`` argument.  ``B`` alone is
        ``K * d * rank`` floats per token, so re-deriving it per
        evaluation dominates the layer's activation footprint.
        """
        return self._components(xi)

    def forward(
        self, xi: torch.Tensor, h: torch.Tensor, *, comps=None,
    ) -> torch.Tensor:
        mu, a, w, B = self._components(xi) if comps is None else comps
        h_e = h.unsqueeze(-2)
        diff = h_e - mu

        diag_term = (a * diff * diff).sum(dim=-1)
        Bt_diff = torch.einsum('...kd,...kdr->...kr', diff, B)
        lr_term = (Bt_diff * Bt_diff).sum(dim=-1)

        exponent = -0.5 * (diag_term + lr_term)
        bumps = w * torch.exp(exponent)
        return -bumps.sum(dim=-1, keepdim=True)

    def analytical_grad(
        self, xi: torch.Tensor, h: torch.Tensor, *, comps=None,
    ) -> torch.Tensor:
        mu, a, w, B = self._components(xi) if comps is None else comps
        h_e = h.unsqueeze(-2)
        diff = h_e - mu

        diag_term = (a * diff * diff).sum(dim=-1)
        Bt_diff = torch.einsum('...kd,...kdr->...kr', diff, B)
        lr_term = (Bt_diff * Bt_diff).sum(dim=-1)

        exponent = -0.5 * (diag_term + lr_term)
        g = w * torch.exp(exponent)

        # Kept in this per-well form deliberately: contracting the well
        # axis inside einsums looks leaner but measures ~40% *worse*,
        # because autograd then has to save both einsum operands instead
        # of reusing the buffers this expression already holds.
        grad_lr = torch.einsum('...kdr,...kr->...kd', B, Bt_diff)
        per_comp = (a * diff + grad_lr) * g.unsqueeze(-1)

        if self._force_norm_max is not None:
            norms = per_comp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = (self._force_norm_max / norms).clamp(max=1.0)
            per_comp = per_comp * scale

        return per_comp.sum(dim=-2)

    def harmonic_terms(
        self, xi: torch.Tensor, h: torch.Tensor, *, comps=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Frozen-coefficient diagonal harmonic model of the force at ``h``.

        The force of this potential is *exactly* a sum of linear springs
        with state-dependent coefficients:

            f(h) = -grad_h V = -sum_k g_k P_k (h - mu_k),
            g_k  = w_k exp(-0.5 diff_k^T P_k diff_k) > 0,
            P_k  = diag(a_k) + B_k B_k^T.

        Keeping only the diagonal of each ``P_k`` gives a per-dimension
        spring whose exact flow the CfC propagator can integrate
        (``cfc_baoab.cfc_substep``):

            k_diag = sum_k g_k diag(P_k),      diag(P_k) = a_k + rowsum(B_k^2)
            s      = sum_k g_k diag(P_k) mu_k
            f_harm(h') = s - k_diag * h'

        At ``h' = h`` this reproduces the diagonal part of the true force
        exactly, so the caller can form the residual
        ``f - f_harm`` and integrate it numerically without changing the
        total force field at all: the split is exact, only the *way the
        two parts are propagated* differs.

        Returns
        -------
        (k_diag, s) : both (..., d)
            ``k_diag >= 0`` elementwise (all wells are attractive), so the
            induced frequency ``sqrt(k_diag/m)`` is always real and the
            propagator is always a bounded rotation.
        """
        mu, a, w, B = self._components(xi) if comps is None else comps
        h_e = h.unsqueeze(-2)
        diff = h_e - mu

        diag_term = (a * diff * diff).sum(dim=-1)
        Bt_diff = torch.einsum('...kd,...kdr->...kr', diff, B)
        lr_term = (Bt_diff * Bt_diff).sum(dim=-1)

        g = w * torch.exp(-0.5 * (diag_term + lr_term))          # (..., K)

        p_diag = a + (B * B).sum(dim=-1)                          # (..., K, d)
        gp = g.unsqueeze(-1) * p_diag                             # (..., K, d)

        k_diag = gp.sum(dim=-2)                                   # (..., d)
        s = (gp * mu).sum(dim=-2)                                 # (..., d)
        return k_diag, s

    def harmonic_terms_lowrank(
        self, xi: torch.Tensor, h: torch.Tensor, *, comps=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the frozen force into a pure-diagonal spring + a PSD low-rank part.

        Unlike :meth:`harmonic_terms` (which keeps the *diagonal* of the
        full precision and leaves everything off-diagonal to the explicit
        kick), this returns the pieces of a split in which the entire
        low-rank correction is integrated exactly.  The frozen-coefficient
        force factors as

            f(h') = (s_a - k_diag_a * h')  +  (G @ (G^T mu) - (G G^T) h')
                  = f_a(h')                +  f_L(h'),

        where the low-rank operator ``L = G G^T`` is symmetric PSD with

            G = [sqrt(g_1) B_1, ..., sqrt(g_K) B_K]   (d x K*rank).

        The diagonal spring ``f_a`` uses only the diagonal precision
        ``a_k`` (never ``diag(B_k B_k^T)``, which now lives entirely in
        ``L``), so the two channels do not double-count.  At ``h' = h`` the
        sum reproduces the exact analytical force, i.e.
        ``f_a(h) + f_L(h) == -analytical_grad(h)``.

        This is mitigation "#1 / low-rank exponential integration": the
        caller (``model_parf_multixi._layer_step_langevin`` under
        ``integrator='baoab_cfc_lowrank'``) integrates ``f_a`` with the
        per-dimension ``cfc_substep`` and ``f_L`` with ``lowrank_cfc_substep``
        on the modes of ``L``, both unconditionally stable.

        Returns
        -------
        (k_diag_a, s_a, G, Gmu) :
            ``k_diag_a`` (..., d):  ``sum_k g_k a_k`` (diagonal precision).
            ``s_a``      (..., d):  ``sum_k g_k a_k * mu_k``.
            ``G``        (..., d, K*rank):  aggregate low-rank factor.
            ``Gmu``      (..., K*rank):     ``G^T`` applied to the wells'
                centres, so ``s_L = G @ Gmu``.
            When ``rank == 0`` the last two are zero-width (no low-rank part).
        """
        mu, a, w, B = self._components(xi) if comps is None else comps
        h_e = h.unsqueeze(-2)
        diff = h_e - mu

        diag_term = (a * diff * diff).sum(dim=-1)
        Bt_diff = torch.einsum('...kd,...kdr->...kr', diff, B)
        lr_term = (Bt_diff * Bt_diff).sum(dim=-1)

        g = w * torch.exp(-0.5 * (diag_term + lr_term))           # (..., K)

        # Diagonal-precision spring (excludes the B contribution).
        ga = g.unsqueeze(-1) * a                                  # (..., K, d)
        k_diag_a = ga.sum(dim=-2)                                 # (..., d)
        s_a = (ga * mu).sum(dim=-2)                               # (..., d)

        # Aggregate low-rank factor G with columns sqrt(g_k) * B_k[:, r].
        lead = g.shape[:-1]
        if self.rank == 0:
            G = h.new_zeros(*lead, self.d, 0)
            Gmu = h.new_zeros(*lead, 0)
        else:
            # sqrt(g) with a SMOOTH gradient.  g = w*exp(-0.5*e) underflows to
            # exactly 0 for any well far from h (common), where g.sqrt() has an
            # infinite derivative 1/(2 sqrt g); autograd then forms inf * (dg/de
            # = g = 0) = NaN and poisons the whole backward, even though the
            # forward is finite.  Computing sqrt(g) = sqrt(w) * exp(-0.25*e)
            # directly is identical in value but analytic in e (no 1/sqrt(0)),
            # so the low-rank arm's gradient stays finite.  (Plain harmonic_terms
            # never hits this -- it uses g itself, not sqrt(g).)
            sqrt_g = w.clamp(min=0.0).sqrt() * torch.exp(
                -0.25 * (diag_term + lr_term)
            )                                                     # (..., K)
            Gk = sqrt_g[..., None, None] * B                      # (..., K, d, r)
            G = Gk.movedim(-3, -2).reshape(*lead, self.d, self.K * self.rank)
            Btmu = torch.einsum('...kd,...kdr->...kr', mu, B)     # (..., K, r)
            Gmu = (sqrt_g[..., None] * Btmu).reshape(
                *lead, self.K * self.rank,
            )
        return k_diag_a, s_a, G, Gmu

    def attractor_centres(self, xi: torch.Tensor) -> torch.Tensor:
        lead = xi.shape[:-1]
        return self.mu_proj(xi).view(*lead, self.K, self.d)


def _ctx(comps, m):
    """Select channel ``m`` from a cached component list, if there is one."""
    return None if comps is None else comps[m]


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
        precision_lr_max: Optional[float] = None,
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
                precision_lr_max=precision_lr_max,
                force_norm_max=force_norm_max,
            )
            for _ in range(n_ctx)
        )

    def context_components(self, xis: torch.Tensor) -> list:
        """Per-channel well parameters; see the per-bank docstring."""
        return [
            self.banks[m].context_components(xis[..., m, :])
            for m in range(self.n_ctx)
        ]

    def forward(
        self, xis: torch.Tensor, h: torch.Tensor, *, comps=None,
    ) -> torch.Tensor:
        out = self.banks[0](xis[..., 0, :], h, comps=_ctx(comps, 0))
        for m in range(1, self.n_ctx):
            out = out + self.banks[m](xis[..., m, :], h, comps=_ctx(comps, m))
        return out

    def analytical_grad(
        self, xis: torch.Tensor, h: torch.Tensor, *, comps=None,
    ) -> torch.Tensor:
        out = self.banks[0].analytical_grad(
            xis[..., 0, :], h, comps=_ctx(comps, 0),
        )
        for m in range(1, self.n_ctx):
            out = out + self.banks[m].analytical_grad(
                xis[..., m, :], h, comps=_ctx(comps, m),
            )
        return out

    def harmonic_terms(
        self, xis: torch.Tensor, h: torch.Tensor, *, comps=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sum the per-channel harmonic models (the potentials add, so do
        their linearisations)."""
        k_diag, s = self.banks[0].harmonic_terms(
            xis[..., 0, :], h, comps=_ctx(comps, 0),
        )
        for m in range(1, self.n_ctx):
            k_m, s_m = self.banks[m].harmonic_terms(
                xis[..., m, :], h, comps=_ctx(comps, m),
            )
            k_diag = k_diag + k_m
            s = s + s_m
        return k_diag, s

    def harmonic_terms_lowrank(
        self, xis: torch.Tensor, h: torch.Tensor, *, comps=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate the per-channel split of :meth:`.harmonic_terms_lowrank`.

        The channels' potentials add, so their diagonal springs add and
        their low-rank operators add: ``L = sum_m G^(m) G^(m)T`` is exactly
        ``G_agg G_agg^T`` with ``G_agg = [G^(1) | ... | G^(n_ctx)]``.  So
        the aggregate factor is the *concatenation* of the per-channel
        factors along the mode axis (and likewise for ``Gmu``).
        """
        k_diag_a, s_a, G0, Gmu0 = self.banks[0].harmonic_terms_lowrank(
            xis[..., 0, :], h, comps=_ctx(comps, 0),
        )
        Gs, Gmus = [G0], [Gmu0]
        for m in range(1, self.n_ctx):
            k_m, s_m, G_m, Gmu_m = self.banks[m].harmonic_terms_lowrank(
                xis[..., m, :], h, comps=_ctx(comps, m),
            )
            k_diag_a = k_diag_a + k_m
            s_a = s_a + s_m
            Gs.append(G_m)
            Gmus.append(Gmu_m)
        G = torch.cat(Gs, dim=-1)
        Gmu = torch.cat(Gmus, dim=-1)
        return k_diag_a, s_a, G, Gmu

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
        precision_lr_max: Optional[float] = None,
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
            precision_lr_max=precision_lr_max,
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

    def context_components(self, xis: torch.Tensor) -> list:
        """Per-channel well parameters for the active layer's shifted xis."""
        return self.bank.context_components(self._shift(xis))

    def _maybe_shift(self, xis: torch.Tensor, comps):
        """Skip the shift when components are supplied: xis is then unused,
        and the shift would otherwise allocate a full copy of it."""
        return xis if comps is not None else self._shift(xis)

    def forward(
        self, xis: torch.Tensor, h: torch.Tensor, *, comps=None,
    ) -> torch.Tensor:
        return self.bank(self._maybe_shift(xis, comps), h, comps=comps)

    def analytical_grad(
        self, xis: torch.Tensor, h: torch.Tensor, *, comps=None,
    ) -> torch.Tensor:
        return self.bank.analytical_grad(
            self._maybe_shift(xis, comps), h, comps=comps,
        )

    def harmonic_terms(
        self, xis: torch.Tensor, h: torch.Tensor, *, comps=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.bank.harmonic_terms(
            self._maybe_shift(xis, comps), h, comps=comps,
        )

    def harmonic_terms_lowrank(
        self, xis: torch.Tensor, h: torch.Tensor, *, comps=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.bank.harmonic_terms_lowrank(
            self._maybe_shift(xis, comps), h, comps=comps,
        )

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

    print(f"\n--- harmonic_terms: exact diagonal linearisation of the force ---")
    torch.manual_seed(0)
    xi_h = torch.randn(3, 5, d)
    h_h = torch.randn(3, 5, d)

    # rank=0: the precision is exactly diagonal, so the harmonic model must
    # reproduce the FULL analytical force, not just part of it.
    bank_iso = AnisotropicMixtureGaussianVTheta(d=d, K=6, rank=0)
    k_diag, s = bank_iso.harmonic_terms(xi_h, h_h)
    f_harm = s - k_diag * h_h
    f_true = -bank_iso.analytical_grad(xi_h, h_h)
    err = (f_harm - f_true).abs().max().item()
    print(f"  rank=0: |f_harm - f_true|_max = {err:.2e}  (must be ~0)")
    assert err < 1e-5, err
    assert (k_diag >= 0).all(), "stiffness must be non-negative (all wells attract)"

    # rank>0: the harmonic model captures the diagonal; the residual is the
    # off-diagonal low-rank coupling, which the caller integrates separately.
    bank_aniso = AnisotropicMixtureGaussianVTheta(d=d, K=6, rank=4)
    k_diag, s = bank_aniso.harmonic_terms(xi_h, h_h)
    f_harm = s - k_diag * h_h
    f_true = -bank_aniso.analytical_grad(xi_h, h_h)
    resid = (f_true - f_harm).abs().max().item()
    print(f"  rank=4: residual (off-diagonal) max = {resid:.2e}  "
          f"(nonzero by design; total force is preserved by construction)")
    assert (k_diag >= 0).all()

    print(f"\n--- precision_lr_max: smooth spectral bound on B_k (#2) ---")
    torch.manual_seed(0)
    budget = 0.05
    bank_capped = AnisotropicMixtureGaussianVTheta(
        d=d, K=6, rank=4, precision_lr_max=budget,
    )
    # Drive B_proj to produce large low-rank factors, then check the bound.
    with torch.no_grad():
        bank_capped.B_proj.weight.mul_(50.0)
        bank_capped.B_proj.bias.add_(5.0)
    xi_c = torch.randn(4, 7, d, requires_grad=True)
    _, _, _, B_c = bank_capped._components(xi_c)
    # sigma_max(B_k) via the r x r Gram (small, only for the assertion).
    gram = torch.einsum('...dr,...ds->...rs', B_c, B_c)
    sig_max_sq = torch.linalg.eigvalsh(gram)[..., -1].max().item()
    print(f"  sigma_max(B_k)^2 (max over wells/tokens) = {sig_max_sq:.4f}  "
          f"(budget precision_lr_max = {budget})")
    assert sig_max_sq <= budget + 1e-5, (sig_max_sq, budget)
    # Differentiable: a scalar of B flows a finite grad back to the context.
    B_c.pow(2).sum().backward()
    assert torch.isfinite(xi_c.grad).all()
    # No-op when precision_lr_max is None (default): B passes through raw.
    torch.manual_seed(0)
    bank_free = AnisotropicMixtureGaussianVTheta(d=d, K=6, rank=4)
    with torch.no_grad():
        bank_free.B_proj.weight.copy_(bank_capped.B_proj.weight)
        bank_free.B_proj.bias.copy_(bank_capped.B_proj.bias)
    _, _, _, B_free = bank_free._components(xi_c.detach())
    gram_f = torch.einsum('...dr,...ds->...rs', B_free, B_free)
    sig_free = torch.linalg.eigvalsh(gram_f)[..., -1].max().item()
    assert sig_free > budget, "unbounded bank should exceed the budget here"
    print(f"  unbounded bank sigma_max(B_k)^2 = {sig_free:.4f}  "
          f"(> budget, as expected)  OK")

    print(f"\n--- harmonic_terms_lowrank: exact diag + PSD low-rank split (#1) ---")
    torch.manual_seed(0)
    bank_lr = AnisotropicMixtureGaussianVTheta(d=d, K=6, rank=4)
    xi_lr = torch.randn(3, 5, d)
    h_lr_pt = torch.randn(3, 5, d)
    k_diag_a, s_a, G, Gmu = bank_lr.harmonic_terms_lowrank(xi_lr, h_lr_pt)
    # f_a(h) + f_L(h) must equal the exact analytical force at h.
    f_a = s_a - k_diag_a * h_lr_pt
    s_L = torch.einsum('...dp,...p->...d', G, Gmu)
    Gt_h = torch.einsum('...dp,...d->...p', G, h_lr_pt)
    f_L = s_L - torch.einsum('...dp,...p->...d', G, Gt_h)
    f_true = -bank_lr.analytical_grad(xi_lr, h_lr_pt)
    err = (f_a + f_L - f_true).abs().max().item()
    print(f"  |(f_a + f_L) - f_true|_max = {err:.2e}  (must be ~0)")
    assert err < 1e-4, err
    # L = G G^T must be PSD (its Gram spectrum is non-negative).
    Lmat = torch.einsum('...dp,...ep->...de', G, G)
    eig_min = torch.linalg.eigvalsh(Lmat).min().item()
    print(f"  min eigenvalue of L = G G^T: {eig_min:.2e}  (PSD, >= 0)")
    assert eig_min > -1e-4, eig_min
    # rank=0 must give a zero-width low-rank part (pure isotropic).
    bank_lr0 = AnisotropicMixtureGaussianVTheta(d=d, K=6, rank=0)
    _, _, G0, Gmu0 = bank_lr0.harmonic_terms_lowrank(xi_lr, h_lr_pt)
    assert G0.shape[-1] == 0 and Gmu0.shape[-1] == 0
    print(f"  rank=0: low-rank part is zero-width  OK")

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
