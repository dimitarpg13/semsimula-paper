"""SPLM em_ln + per-token non-conservative force: Stage 1 of SP-HSPLM.

Pre-registered protocol
-----------------------
docs/SP_HSPLM_Stage_1_pre-registered_protocol.md

This module implements the five Stage 1 ablation cells (E1-fix through
E5-fix) plus the matched Cell 0 baseline. Each cell augments the SPLM
em_ln (LayerNorm-after-step) integrator with a per-token non-conservative
force g_l, defined locally from h_l and the pre-update velocity v_l so
that the existing causal-leak invariant (cfg.causal_force=True) is
preserved by construction.

Cells
-----
- e0_baseline         : g_l = 0 (matched Cell 0 baseline at the 16k schedule).
- e1_const_skew       : g_l = (J - J^T) v_l, J in R^{d x d}.
- e2_affine_rank1     : g_l = (u h^T - h u^T) v_l, u in R^d.
- e3_lowrank_rank2    : g_l = (U H(h)^T - H(h) U^T) v_l, U in R^{d x 2},
                        H(h) = W h reshaped to (d x 2), W in R^{d x d x 2}.
- e4_solenoidal_rank4 : g_l = (U V(h)^T - V(h) U^T) rho(h), U in R^{d x 4},
                        V(h) = W h reshaped to (d x 4), rho a small MLP.
                        Class C (position-only solenoidal, no v coupling).
- e5_lowrank_rank4    : g_l = (U V(h)^T - V(h) U^T) v_l, U in R^{d x 4},
                        V(h) = W h reshaped to (d x 4). The richer
                        rank-ablation companion of e3_lowrank_rank2.

Design notes
------------
The integrator is identical to model_ln.py except for one extra line per
step:
    g = self.nonconservative(h_in, v)
    f_total = -grad_V + g
    v = (v + dt * f_total / m) / (1 + dt * gamma)
    h = h_in + dt * v

The non-conservative force is computed from the local h_in (the current
layer's hidden state, before the integrator update) and v (the
pre-update velocity at this layer). No cross-token information enters
the force, so the existing .detach() points in xi computation continue
to enforce zero anti-causal leak.

All five non-conservative force modules use small initialisation
(default std=0.002) so the initial g/f norm ratio is below 0.05; this
is the structural mitigation against the velocity-coupled E5 with s=1
divergence pathology of the v3 paper section 15.5.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).parent
_PARENT_DIR = _THIS_DIR.parent
_EM_DIR = _PARENT_DIR / "energetic_minima"
_SARF_MASS_DIR = _PARENT_DIR / "sarf_mass_variant"
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(_EM_DIR))
sys.path.insert(0, str(_SARF_MASS_DIR))

from model_ln import (  # type: ignore  # noqa: E402
    SPLMSARFMassLNConfig,
    ScalarPotentialLMSARFMassLN,
)
from sarf_mass_variant.model_sarf_mass import (  # type: ignore  # noqa: E402
    causal_cumulative_mean,
)


# ---------------------------------------------------------------------------
# Stage 1 cell registry
# ---------------------------------------------------------------------------

CELLS: Tuple[str, ...] = (
    "e0_baseline",
    "e1_const_skew",
    "e2_affine_rank1",
    "e3_lowrank_rank2",
    "e4_solenoidal_rank4",
    "e5_lowrank_rank4",
)


# ---------------------------------------------------------------------------
# Non-conservative force modules
# ---------------------------------------------------------------------------

class NonConservativeForce(nn.Module):
    """Base class: returns zero. Cell 0 (e0_baseline) uses this directly.

    Subclasses implement forward(h, v) -> g, where:
        h : (B, T, d)  current-layer hidden state (with grad if training)
        v : (B, T, d)  pre-update velocity at this layer
        g : (B, T, d)  non-conservative force, same shape as h.
    """

    needs_velocity: bool = False

    def forward(self, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(h)


class ConstantSkewForce(NonConservativeForce):
    """E1-fix: g = (J - J^T) v with J shared across all (b, t, layers).

    The skew matrix Omega = J - J^T is divergence-free in h: tr(Omega) = 0
    by construction.
    """

    needs_velocity = True

    def __init__(self, d: int, init_std: float = 0.002):
        super().__init__()
        self.J = nn.Parameter(torch.randn(d, d) * init_std)

    def forward(self, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        Omega = self.J - self.J.transpose(-1, -2)
        return torch.einsum("ij,btj->bti", Omega, v)


class AffineRank1SkewForce(NonConservativeForce):
    """E2-fix: g = Omega(h) v with Omega(h) = u h^T - h u^T.

    Skew is automatic: Omega(h)^T = h u^T - u h^T = -Omega(h). The trace
    of Omega(h) is u . h - h . u = 0 for every h, so the term is
    divergence-free in h at every point.
    """

    needs_velocity = True

    def __init__(self, d: int, init_std: float = 0.002):
        super().__init__()
        self.u = nn.Parameter(torch.randn(d) * init_std)

    def forward(self, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # g = (u h^T - h u^T) v = u (h . v) - h (u . v)
        h_dot_v = (h * v).sum(dim=-1, keepdim=True)        # (B, T, 1)
        u_dot_v = (self.u * v).sum(dim=-1, keepdim=True)    # (B, T, 1)
        return self.u * h_dot_v - h * u_dot_v


class LowRankSkewForce(NonConservativeForce):
    """E3 / E5 family: g = (U V(h)^T - V(h) U^T) v with V(h) low-rank in h.

    V(h) is computed as r separate linear maps from R^d to R^d, parameterised
    by W in R^{d x d x r}. The result V(h) in R^{d x r} per token is then
    combined with U in R^{d x r} into a rank-2r skew matrix Omega(h) =
    U V(h)^T - V(h) U^T.

    Used for two cells:
        e3_lowrank_rank2 : r = 2
        e5_lowrank_rank4 : r = 4
    """

    needs_velocity = True

    def __init__(self, d: int, r: int, init_std: float = 0.002):
        super().__init__()
        self.d = d
        self.r = r
        self.U = nn.Parameter(torch.randn(d, r) * init_std)
        # W : (d, d, r) maps h in R^d to V(h) in R^{d x r} per token.
        self.W = nn.Parameter(torch.randn(d, d, r) * init_std)

    def forward(self, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # V(h) : (B, T, d, r), V(h)[b,t,j,k] = sum_i W[j,i,k] * h[b,t,i]
        V = torch.einsum("jir,bti->btjr", self.W, h)
        # g = U V^T v - V U^T v
        Vv = torch.einsum("btjr,btj->btr", V, v)            # (B, T, r)
        Uv = torch.einsum("jr,btj->btr", self.U, v)         # (B, T, r)
        term1 = torch.einsum("jr,btr->btj", self.U, Vv)     # (B, T, d)
        term2 = torch.einsum("btjr,btr->btj", V, Uv)        # (B, T, d)
        return term1 - term2


class LowRankSolenoidalForce(NonConservativeForce):
    """E4-fix: g = (J_+(h) - J_+(h)^T) rho(h), J_+(h) = U V(h)^T.

    Class C (position-only, no velocity coupling). The skew construction
    J_+ - J_+^T makes the term divergence-free in h at every point. The
    rho(h) MLP provides the position-dependent direction the per-token
    solenoidal field couples to (the analogue of the SP-HSPLM C-block's
    velocity difference, evaluated locally for the per-token Stage 1
    cell).
    """

    needs_velocity = False

    def __init__(
        self, d: int, r: int = 4, h_rho: int = 64, init_std: float = 0.002,
    ):
        super().__init__()
        self.d = d
        self.r = r
        self.U = nn.Parameter(torch.randn(d, r) * init_std)
        self.W = nn.Parameter(torch.randn(d, d, r) * init_std)

        self.rho = nn.Sequential(
            nn.Linear(d, h_rho),
            nn.GELU(),
            nn.Linear(h_rho, d),
        )
        # Init rho input head with std 0.02 (standard), output head small
        # so the initial g/f norm ratio stays below 0.05.
        for m in self.rho.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.rho[-1].weight, std=init_std)
        nn.init.zeros_(self.rho[-1].bias)

    def forward(self, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # V(h) : (B, T, d, r)
        V = torch.einsum("jir,bti->btjr", self.W, h)
        rho_h = self.rho(h)                                 # (B, T, d)
        # g = U V^T rho - V U^T rho
        Vrho = torch.einsum("btjr,btj->btr", V, rho_h)      # (B, T, r)
        Urho = torch.einsum("jr,btj->btr", self.U, rho_h)   # (B, T, r)
        term1 = torch.einsum("jr,btr->btj", self.U, Vrho)   # (B, T, d)
        term2 = torch.einsum("btjr,btr->btj", V, Urho)      # (B, T, d)
        return term1 - term2


def make_force(
    cell: str,
    d: int,
    init_std: float = 0.002,
    h_rho: int = 64,
) -> NonConservativeForce:
    """Build the non-conservative force module for a Stage 1 cell."""
    if cell == "e0_baseline":
        return NonConservativeForce()
    if cell == "e1_const_skew":
        return ConstantSkewForce(d, init_std=init_std)
    if cell == "e2_affine_rank1":
        return AffineRank1SkewForce(d, init_std=init_std)
    if cell == "e3_lowrank_rank2":
        return LowRankSkewForce(d, r=2, init_std=init_std)
    if cell == "e4_solenoidal_rank4":
        return LowRankSolenoidalForce(d, r=4, h_rho=h_rho, init_std=init_std)
    if cell == "e5_lowrank_rank4":
        return LowRankSkewForce(d, r=4, init_std=init_std)
    raise ValueError(
        f"unknown cell: {cell!r}. valid cells: {CELLS}"
    )


# ---------------------------------------------------------------------------
# Config and model
# ---------------------------------------------------------------------------

@dataclass
class SPLMNonConservativeConfig(SPLMSARFMassLNConfig):
    """Extends SPLMSARFMassLNConfig with the Stage 1 cell selector."""
    cell: str = "e0_baseline"
    nonconservative_init_std: float = 0.002
    nonconservative_h_rho: int = 64


class ScalarPotentialLMNonConservative(ScalarPotentialLMSARFMassLN):
    """SPLM em_ln + per-token non-conservative force per Stage 1 protocol.

    The integrator is structurally identical to ScalarPotentialLMSARFMassLN
    except that the conservative force f = -grad V_theta is augmented with
    g = self.nonconservative(h_in, v) at every layer, then
        v_new = (v + dt * (f + g) / m) / (1 + dt * gamma)
        h_new = h_in + dt * v_new
        h_new = layer_norm(h_new)   # if cfg.ln_after_step

    Cell selection lives in cfg.cell (one of CELLS). Cell e0_baseline uses
    NonConservativeForce (the no-op base class), reproducing the em_ln
    baseline at the 16k schedule.
    """

    def __init__(self, cfg: SPLMNonConservativeConfig):
        super().__init__(cfg)
        self.cfg: SPLMNonConservativeConfig = cfg
        if cfg.cell not in CELLS:
            raise ValueError(
                f"cfg.cell = {cfg.cell!r} not in {CELLS}"
            )
        self.nonconservative = make_force(
            cfg.cell,
            cfg.d,
            init_std=cfg.nonconservative_init_std,
            h_rho=cfg.nonconservative_h_rho,
        )

    # ------------------------------------------------------------------
    def integrate(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        return_trajectory: bool = False,
        return_xi_trajectory: bool = False,
        return_g_norms: bool = False,
    ) -> Tuple[torch.Tensor,
               Optional[List[torch.Tensor]],
               Optional[List[torch.Tensor]],
               Optional[List[float]]]:
        cfg = self.cfg
        h = self._project(emb) if cfg.ln_after_step else emb
        v = torch.zeros_like(h)
        gamma, dt = self.gamma, cfg.dt

        m = self.compute_mass(x, emb)
        m_b = m

        traj_h: Optional[List[torch.Tensor]] = None
        traj_xi: Optional[List[torch.Tensor]] = None
        g_norms: Optional[List[float]] = None
        if return_trajectory:
            traj_h = [h.detach().cpu()]
        if return_xi_trajectory:
            traj_xi = []
        if return_g_norms:
            g_norms = []

        for _ in range(cfg.L):
            xi_input = h.detach() if cfg.causal_force else h
            xi_now = causal_cumulative_mean(xi_input)
            if return_xi_trajectory:
                assert traj_xi is not None
                traj_xi.append(xi_now.detach().cpu())

            h_in = h
            if not h_in.requires_grad:
                h_in = h_in.requires_grad_(True)
            V = self.V_theta(xi_now, h_in).sum()
            grad_V, = torch.autograd.grad(
                V, h_in,
                create_graph=self.training,
                retain_graph=True,
            )
            f_conservative = -grad_V
            # Stage 1 augmentation: per-token non-conservative force.
            # g is computed from the LOCAL h_in and the pre-update v at
            # this layer. No cross-token information enters g, so the
            # existing causal-leak invariant (cfg.causal_force) is
            # preserved by construction.
            g = self.nonconservative(h_in, v)
            f_total = f_conservative + g

            if return_g_norms:
                assert g_norms is not None
                g_norms.append(float(g.detach().norm().item()))

            v = (v + dt * f_total / m_b) / (1.0 + dt * gamma)
            h_new = h_in + dt * v
            if cfg.ln_after_step:
                h_new = self._project(h_new)
            h = h_new
            if return_trajectory:
                assert traj_h is not None
                traj_h.append(h.detach().cpu())

        return h, traj_h, traj_xi, g_norms

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        return_xi_trajectory: bool = False,
        return_g_norms: bool = False,
    ):
        emb = self._embed(x)
        h_L, traj_h, traj_xi, g_norms = self.integrate(
            x, emb,
            return_trajectory=return_trajectory,
            return_xi_trajectory=return_xi_trajectory,
            return_g_norms=return_g_norms,
        )
        logits = h_L @ self.E.weight.T

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
            )
        out: List = [logits, loss]
        if return_trajectory:
            out.append(traj_h)
        if return_xi_trajectory:
            out.append(traj_xi)
        if return_g_norms:
            out.append(g_norms)
        return tuple(out) if len(out) > 2 else (out[0], out[1])

    # ------------------------------------------------------------------
    def nonconservative_norm_stats(
        self, x: torch.Tensor, n_samples: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Diagnostic: per-layer ||g_l|| / ||f_l|| ratio averaged over a batch.

        Returns a dict with keys 'g_norms', 'f_norms', 'ratio' each a list of
        cfg.L floats.

        Uses model.eval() with grad-enabled to compute the conservative
        force, then probes the non-conservative module per layer. Caller
        is responsible for restoring the prior train/eval mode if needed.
        """
        cfg = self.cfg
        was_training = self.training
        self.eval()
        emb = self._embed(x)
        if n_samples is not None:
            emb = emb[:n_samples]
            x = x[:n_samples]
        with torch.enable_grad():
            h = self._project(emb) if cfg.ln_after_step else emb
            v = torch.zeros_like(h)
            gamma, dt = self.gamma, cfg.dt
            m = self.compute_mass(x, emb)
            m_b = m

            g_norms: List[float] = []
            f_norms: List[float] = []
            ratios: List[float] = []

            for _ in range(cfg.L):
                xi_input = h.detach() if cfg.causal_force else h
                xi_now = causal_cumulative_mean(xi_input)
                h_in = h.detach().requires_grad_(True)
                V = self.V_theta(xi_now, h_in).sum()
                grad_V, = torch.autograd.grad(V, h_in, create_graph=False)
                f_c = -grad_V
                g = self.nonconservative(h_in, v)

                f_n = float(f_c.detach().norm().item())
                g_n = float(g.detach().norm().item())
                f_norms.append(f_n)
                g_norms.append(g_n)
                ratios.append(g_n / max(f_n, 1e-12))

                v = (v + dt * (f_c + g) / m_b) / (1.0 + dt * gamma)
                h_new = h_in + dt * v
                if cfg.ln_after_step:
                    h_new = self._project(h_new)
                h = h_new.detach()

        if was_training:
            self.train()
        return {"g_norms": g_norms, "f_norms": f_norms, "ratio": ratios}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _smoke_test():
    """Forward + backward on every cell at a tiny config."""
    torch.manual_seed(0)
    V = 257
    base_kw = dict(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
        mass_mode="global",
    )
    for cell in CELLS:
        cfg = SPLMNonConservativeConfig(cell=cell, **base_kw)
        net = ScalarPotentialLMNonConservative(cfg)
        x = torch.randint(0, V, (2, 16))
        y = torch.randint(0, V, (2, 16))
        logits, loss = net(x, y)
        loss.backward()
        n_params = net.num_params()
        n_extra = sum(
            p.numel() for p in net.nonconservative.parameters()
            if p.requires_grad
        )
        print(
            f"[non_conservative smoke] cell={cell:24s} "
            f"loss={loss.item():.4f} params={n_params:,} "
            f"force_params={n_extra:,}"
        )


if __name__ == "__main__":
    _smoke_test()
