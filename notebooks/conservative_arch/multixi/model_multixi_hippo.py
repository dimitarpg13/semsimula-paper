"""
HiPPO-based multi-channel ξ extension of the SARF-faithful SPLM with LN-after-step.

Architectural change vs `multixi/model_multixi.py`
--------------------------------------------------
The K-EMA bank of `MultiChannelXi` produces ξ as K parallel causal weighted
exponential moving averages at learnable decays α_k:

    ξ^{(k)}_t  =  Σ_{s ≤ t} α_k^{t-s} · h_s / Z_{k,t}                  (K-EMA bank)

This module replaces that bank with the **HiPPO/Legendre projection of the
past trajectory** (Gu, Dao, Ermon, Rudra, Re 2020), embedded inside the SPLM
Lagrangian framework. The K "channels" become the K Legendre-polynomial
coefficients of the past, computed via the structured state-space ODE

    dc(t)/dt  =  A · c(t) + B · h(t)                                    (continuous time)
    c_{t+1}   =  Â · c_t + B̂ · h_t                                     (bilinear-discretised)

with A ∈ ℝ^{K×K} and B ∈ ℝ^{K×1} the **non-diagonal** (HiPPO-LegT) or
**time-varying** (HiPPO-LegS) Legendre transition matrices.

Diagonalising A ⇒ K-EMA bank; the K-EMA bank is therefore the diagonal
restriction of the HiPPO ODE, with strictly higher pairwise channel
correlation than the orthogonal Legendre coefficients (see
`docs/Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md` §5.3 for
the derivation, §5.4 for the quantitative MI gap).

Why this matters for SPLM
-------------------------
Empirically the leak-corrected K=4 multi-channel ξ landed at val_ppl 14.78
on TinyStories (`docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md` §4.6).
The K=4 EMA bank is heavily redundant: pairwise correlations between
adjacent channels are ≈0.94 (≈1.07 nats of mutual information), leaving
the bank with ≈2.16 effective independent channels rather than the nominal
four. HiPPO-LegT reduces pairwise correlation to ≈0 by construction
(orthogonal Legendre basis), recovering the full K effective channels and
increasing the predictive information available to V_θ.

Causal-leak compatibility
-------------------------
Identical to `model_multixi.py`. The integrator computes the HiPPO state
from `h.detach()` whenever `cfg.causal_force = True` (the default), severing
the autograd path from c_t back to h_t. The HiPPO recurrence is strictly
causal by construction (c_t depends only on h_1, …, h_t), so under the fix
the conservative-flow Euler-Lagrange equation

    m · ḧ_t = -∂V_θ(c_t, h_t) / ∂h_t

is preserved exactly. See `docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`
for the full bug-and-fix writeup.

Parameter cost
--------------
- HiPPO state matrices A, B: stored as buffers (no learnable params) by
  default. Optionally a learnable per-channel discretisation step Δ_k can
  be enabled (S4-style); by default disabled to keep the prototype minimal.
- V_θ first layer widened from 2·d → (K+1)·d, identical to the K-EMA variant.
- Net: same parameter count as `model_multixi.py` minus the K α_k scalars.

References
----------
- Gu, A., Dao, T., Ermon, S., Rudra, A., Re, C. (2020).
  HiPPO: Recurrent Memory with Optimal Polynomial Projections.
- Gu, A., Goel, K., Re, C. (2021). Efficiently Modeling Long Sequences with
  Structured State Spaces.
- docs/Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md §5.
- docs/HiPPO_S4_SPLM_Analysis.docx.
- docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).parent
_PARENT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(_PARENT_DIR / "energetic_minima"))
sys.path.insert(0, str(_PARENT_DIR / "sarf_mass_variant"))
sys.path.insert(0, str(_THIS_DIR))

from model_ln import (  # noqa: E402
    ScalarPotentialLMSARFMassLN,
    SPLMSARFMassLNConfig,
)
from model_multixi import ScalarPotentialMultiXi  # noqa: E402


# ---------------------------------------------------------------------------
# HiPPO matrix builders (numpy; converted to torch buffers downstream)
# ---------------------------------------------------------------------------

def make_hippo_legt(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """HiPPO-LegT matrices (translated Legendre, fixed-window measure).

    Defined on a sliding window; the window length is folded into the
    discretisation step Δ = 1/θ at discretisation time.

    Reference: HiPPO (Gu et al. 2020), and the §5.3 derivation in
    `docs/Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md`:

        A_{nk} = -√((2n+1)(2k+1)) · σ_{nk},   σ_{nk} = 1 if n ≥ k else (-1)^{n-k}
        B_n    = √(2n+1)

    Returns
    -------
    A : (N, N) np.ndarray — continuous-time transition matrix
    B : (N, 1) np.ndarray — input projection
    """
    n = np.arange(N, dtype=np.float64)
    P = np.sqrt(2.0 * n + 1.0)
    PP = P[:, None] * P[None, :]                    # (N, N)
    diff = n[:, None] - n[None, :]                  # i - j
    sign = np.ones_like(PP)
    upper = diff < 0
    # σ_{nk} = (-1)^{n-k} for n < k  (equivalently (-1)^{|n-k|})
    sign[upper] = (-1.0) ** (-diff[upper])
    A = -PP * sign
    B = P.reshape(N, 1)
    return A, B


def make_hippo_legs(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """HiPPO-LegS matrices (scaled Legendre, full-history measure with 1/t scaling).

    Reference: HiPPO (Gu et al. 2020), Algorithm 1; the lower-triangular
    form used by S4 implementations (Gu, Goel, Re 2021):

        A_{nk} = - √((2n+1)(2k+1))  for n > k
                  - (n + 1)         for n = k
                  0                 for n < k

        B_n   = √(2n+1)

    Returns
    -------
    A : (N, N) np.ndarray — continuous-time transition matrix (constant; the
                            1/t time-scaling is applied at discretisation time
                            in `MultiChannelHiPPO._forward_legs`).
    B : (N, 1) np.ndarray — input projection.
    """
    n = np.arange(N, dtype=np.float64)
    P = np.sqrt(2.0 * n + 1.0)
    A = P[:, None] * P[None, :]
    A = np.tril(A) - np.diag(n)
    A = -A
    B = P.reshape(N, 1)
    return A, B


def bilinear_discretize(
    A: np.ndarray, B: np.ndarray, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinear (Tustin) discretisation of  ċ = A c + B u  at step Δt = dt.

        Â = (I - dt/2 · A)^{-1} (I + dt/2 · A)
        B̂ = (I - dt/2 · A)^{-1} · dt · B

    Tustin preserves stability (continuous-time A Hurwitz ⇒ Â Schur), which
    is the property we need here: HiPPO-LegT continuous A is Hurwitz, so Â
    has spectral radius ≤ 1 and the recurrence is numerically stable to T → ∞.
    """
    I = np.eye(A.shape[0])
    inv = np.linalg.inv(I - 0.5 * dt * A)
    Abar = inv @ (I + 0.5 * dt * A)
    Bbar = inv @ (B * dt)
    return Abar, Bbar


# ---------------------------------------------------------------------------
# Multi-channel HiPPO ξ module
# ---------------------------------------------------------------------------

class MultiChannelHiPPO(nn.Module):
    """K HiPPO/Legendre coefficients summarising the causal past of h.

    Forward:  h (B, T, d)  →  c (B, T, K, d).

    Each scalar feature dim h[..., j] is treated as an independent input
    signal; HiPPO produces K Legendre-polynomial coefficients per feature.
    The K basis functions are orthogonal under the chosen measure
    (LegT: uniform on [t-θ, t]; LegS: scaled-uniform on [0, t]).

    Parameters
    ----------
    K : int
        Number of Legendre orders / channels.  K=4 matches the existing
        K-EMA bank for direct comparison.
    max_len : int
        Maximum sequence length (used to size precomputed kernels).
    basis : str
        "legt" — HiPPO-LegT (translated Legendre, fixed-window).
        "legs" — HiPPO-LegS (scaled Legendre, full history).
    theta : float
        LegT only: effective window length. The discretisation step is
        Δ = 1/theta (continuous-time ODE rescaled so τ ∈ [0, 1] over the
        window).  Default 200 — chosen for TinyStories per
        `Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md` §5.7.
    learnable_dt : bool
        If True (LegT only), expose log(dt) as a learnable parameter and
        rebuild Â, B̂ on every forward pass. S4-style. Default False.
    """

    def __init__(
        self,
        K: int,
        max_len: int,
        basis: str = "legt",
        theta: float = 200.0,
        learnable_dt: bool = False,
    ):
        super().__init__()
        if basis not in ("legt", "legs"):
            raise ValueError(f"unknown basis: {basis!r} (expected 'legt' or 'legs')")
        self.K = K
        self.max_len = max_len
        self.basis = basis
        self.theta = theta
        self.learnable_dt = learnable_dt

        if basis == "legt":
            A_np, B_np = make_hippo_legt(K)
            self.register_buffer(
                "A_cont", torch.tensor(A_np, dtype=torch.float32)
            )
            self.register_buffer(
                "B_cont", torch.tensor(B_np, dtype=torch.float32)
            )
            dt_init = 1.0 / float(theta)
            if learnable_dt:
                # log(dt) parameterisation — keeps dt > 0.
                self.log_dt = nn.Parameter(
                    torch.tensor(math.log(dt_init), dtype=torch.float32)
                )
                # No precomputed kernel; rebuild each forward pass.
            else:
                Abar_np, Bbar_np = bilinear_discretize(A_np, B_np, dt_init)
                self.register_buffer(
                    "Abar", torch.tensor(Abar_np, dtype=torch.float32)
                )
                self.register_buffer(
                    "Bbar",
                    torch.tensor(Bbar_np, dtype=torch.float32).squeeze(-1),
                )
                M_np = self._compute_kernel(Abar_np, Bbar_np.squeeze(-1), max_len)
                self.register_buffer(
                    "M_kernel", torch.tensor(M_np, dtype=torch.float32)
                )

        elif basis == "legs":
            # LegS continuous-time ODE (in our convention with A_0 = -A_paper):
            #     ċ(t) = (1/t) A_0 c(t) + (1/t) B_0 h(t)
            # A_0 is Hurwitz (all eigenvalues have negative real parts), so
            # the continuous-time ODE is asymptotically stable. We discretise
            # with **backward Euler** (A-stable: stable for any step size,
            # any stable ODE), which is essential here because the effective
            # step Δ_t = 1/(t+1) is large for small t and forward Euler
            # would amplify by a factor of (K + 1) per step in the first few
            # tokens. Backward-Euler form:
            #
            #     (I - (1/(t+1)) A_0)  c_{t+1} = c_t + (1/(t+1)) B_0 h_{t+1}
            #     c_{t+1} = M_{t+1}^{-1} ( c_t + (1/(t+1)) B_0 h_{t+1} )
            #
            # M_t^{-1} depends only on t and A_0, both of which are constants,
            # so we precompute the (max_len, K, K) buffer once at init time.
            A0_np, B0_np = make_hippo_legs(K)
            I = np.eye(K)
            Minv_np = np.zeros((max_len, K, K), dtype=np.float64)
            for t in range(max_len):
                inv_t = 1.0 / float(t + 1)
                Minv_np[t] = np.linalg.inv(I - inv_t * A0_np)
            self.register_buffer(
                "A0", torch.tensor(A0_np, dtype=torch.float32)
            )
            self.register_buffer(
                "B0", torch.tensor(B0_np, dtype=torch.float32).squeeze(-1)
            )
            self.register_buffer(
                "M_inv", torch.tensor(Minv_np, dtype=torch.float32)
            )

    # ------------------------------------------------------------------
    # Kernel computation (LegT only)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_kernel(Abar: np.ndarray, Bbar: np.ndarray, T: int) -> np.ndarray:
        """M[k] = Â^k · B̂  for k = 0, …, T-1.

        Used as the convolution kernel of the time-invariant LegT recurrence
        c_{t+1} = Â c_t + B̂ h_t   →   c_{t+1} = Σ_{s=0}^{t} Â^{t-s} B̂ · h_s
        i.e.  c_out[t] := c_{t+1} = Σ_{s=0}^{t} M[t-s] · h_s.
        """
        K = Abar.shape[0]
        M = np.zeros((T, K), dtype=Abar.dtype)
        cur = Bbar.astype(Abar.dtype).copy()
        for k in range(T):
            M[k] = cur
            cur = Abar @ cur
        return M

    def _kernel_for_dt(self, dt: torch.Tensor, T: int) -> torch.Tensor:
        """Build M kernel in torch (allows grad through dt). LegT only.

        Implementation note. The K×K bilinear inverse and the T-step
        Â/B̂ iteration are run on CPU even when dt lives on MPS:
        `torch.linalg.inv` of a small autograd-tracked matrix on the MPS
        backend triggers an `MPSNDArrayDescriptor sliceDimension:` shader
        assertion on current PyTorch builds, and the per-step matmul of
        (4,4) × (4,) for T = 512 iterations dispatches an MPS kernel-launch
        per iteration that is dominated by overhead anyway. Doing the
        whole construction on CPU costs ≪ 1 ms at K = 4, T ≤ 1024 and
        autograd through `dt` is preserved by `.to('cpu')` / `.to(device)`.
        Only the final (T, K) kernel is moved back to the MPS side.
        """
        device = dt.device
        dtype = dt.dtype
        dt_c = dt.to(device="cpu")
        I_c = torch.eye(self.K, dtype=dtype, device="cpu")
        A_c = self.A_cont.to(device="cpu", dtype=dtype)
        B_c = self.B_cont.to(device="cpu", dtype=dtype).squeeze(-1)  # (K,)
        inv_c = torch.linalg.inv(I_c - 0.5 * dt_c * A_c)
        Abar_c = inv_c @ (I_c + 0.5 * dt_c * A_c)
        Bbar_c = (inv_c @ (B_c.unsqueeze(-1) * dt_c)).squeeze(-1)
        rows = [Bbar_c]
        for _ in range(1, T):
            rows.append(Abar_c @ rows[-1])
        M_cpu = torch.stack(rows, dim=0)              # (T, K)
        return M_cpu.to(device=device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, d = h.shape
        if T > self.max_len:
            raise ValueError(
                f"MultiChannelHiPPO: T={T} > max_len={self.max_len}"
            )
        if self.basis == "legt":
            return self._forward_legt(h)
        return self._forward_legs(h)

    def _forward_legt(self, h: torch.Tensor) -> torch.Tensor:
        # Apply the K kernels one-at-a-time and stack. This matches the
        # `MultiChannelXi` (K-EMA) memory pattern: each k-step is a single
        # (T, T) × (B, T, d) batched matmul, costing ≈ B·T²·d FLOPs and
        # peaking at ≈ B·T·d activation memory. Doing all K kernels in a
        # single einsum would materialise a (B, T, T, K, d) intermediate
        # (≈ 8 GB at pilot config B=16 T=512 K=4 d=256), OOM-ing on MPS.
        B, T, d = h.shape
        K = self.K
        device, dtype = h.device, h.dtype
        if self.learnable_dt:
            M = self._kernel_for_dt(torch.exp(self.log_dt), T)  # (T, K)
        else:
            M = self.M_kernel[:T].to(dtype=dtype, device=device)
        # Index matrix for the lower-triangular Toeplitz layout:
        #   W_k[t, s] = M[t - s, k]  for s ≤ t,  else 0.
        ix = (
            torch.arange(T, device=device).view(T, 1)
            - torch.arange(T, device=device).view(1, T)
        )                                            # (T, T)
        mask = (ix >= 0).to(dtype)
        ix_clamped = ix.clamp(min=0)
        cs: List[torch.Tensor] = []
        for k in range(K):
            W_k = M[ix_clamped, k] * mask            # (T, T)
            c_k = W_k.unsqueeze(0) @ h               # (1, T, T) @ (B, T, d) → (B, T, d)
            cs.append(c_k)
        return torch.stack(cs, dim=2)                # (B, T, K, d)

    def _forward_legs(self, h: torch.Tensor) -> torch.Tensor:
        # Backward-Euler discretisation (A-stable). See __init__ for the
        # derivation. Recurrence (1-indexed; c_0 := 0):
        #
        #   c_t = M_t^{-1} ( c_{t-1} + (1/t) · B_0 · h_t )
        #
        # where M_t^{-1} = (I - (1/t) A_0)^{-1} is precomputed in self.M_inv.
        B, T, d = h.shape
        K = self.K
        device, dtype = h.device, h.dtype
        c = torch.zeros(B, K, d, device=device, dtype=dtype)
        B0 = self.B0.to(dtype=dtype, device=device)
        M_inv = self.M_inv[:T].to(dtype=dtype, device=device)  # (T, K, K)
        outs: List[torch.Tensor] = []
        for t in range(T):
            inv_t = 1.0 / float(t + 1)
            rhs = c + (inv_t * B0).view(1, K, 1) * h[:, t, :].unsqueeze(1)
            c = torch.einsum("Kk,bkd->bKd", M_inv[t], rhs)
            outs.append(c)
        return torch.stack(outs, dim=1)                       # (B, T, K, d)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def discretisation_step(self) -> float:
        """Current Δ = 1/θ (LegT only). Returns the learned value if learnable_dt."""
        if self.basis != "legt":
            return float("nan")
        if self.learnable_dt:
            return float(torch.exp(self.log_dt).item())
        return 1.0 / float(self.theta)


# ---------------------------------------------------------------------------
# SPLM model class with HiPPO multi-channel ξ
# ---------------------------------------------------------------------------

@dataclass
class SPLMSARFMassLNMultiHiPPOConfig(SPLMSARFMassLNConfig):
    """Config for the HiPPO-based multi-channel-ξ SPLM em_ln model.

    Defaults are HiPPO-LegT with K = 4 Legendre orders and θ = 200, matching
    the discussion in
    `docs/Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md` §5.7
    for TinyStories-scale corpora.
    """

    xi_channels: int = 4
    xi_basis: str = "legt"               # "legt" | "legs"
    xi_theta: float = 200.0              # LegT only
    xi_learnable_dt: bool = False


class ScalarPotentialLMSARFMassLNMultiHiPPO(ScalarPotentialLMSARFMassLN):
    """SARF-faithful SPLM with LN-after-step and HiPPO-based multi-channel ξ.

    Replaces the K-EMA bank of `ScalarPotentialLMSARFMassLNMultiXi` with the
    HiPPO/Legendre projection. V_θ takes the K-channel × d-dim ξ stack as
    input — interface-compatible with the K-EMA variant — but the K channels
    are now orthogonal Legendre coefficients of the past trajectory rather
    than redundant exponential averages.

    The integration loop is identical to `model_multixi.py`'s, including the
    `h.detach()` causal-leak fix when `cfg.causal_force = True`.
    """

    def __init__(self, cfg: SPLMSARFMassLNMultiHiPPOConfig):
        super().__init__(cfg)
        self.cfg: SPLMSARFMassLNMultiHiPPOConfig = cfg

        self.V_theta = ScalarPotentialMultiXi(
            d=cfg.d,
            hidden=cfg.v_hidden,
            depth=cfg.v_depth,
            K=cfg.xi_channels,
        )

        self.xi_module = MultiChannelHiPPO(
            K=cfg.xi_channels,
            max_len=cfg.max_len,
            basis=cfg.xi_basis,
            theta=cfg.xi_theta,
            learnable_dt=cfg.xi_learnable_dt,
        )

    def integrate(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        return_trajectory: bool = False,
        return_xi_trajectory: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        cfg = self.cfg
        h = self._project(emb) if cfg.ln_after_step else emb
        v = torch.zeros_like(h)
        gamma, dt = self.gamma, cfg.dt

        m = self.compute_mass(x, emb)
        m_b = m

        traj_h: Optional[List[torch.Tensor]] = None
        traj_xi: Optional[List[torch.Tensor]] = None
        if return_trajectory:
            traj_h = [h.detach().cpu()]
        if return_xi_trajectory:
            traj_xi = []

        for _ in range(cfg.L):
            xi_input = h.detach() if cfg.causal_force else h
            xis = self.xi_module(xi_input)                  # (B, T, K, d)
            if return_xi_trajectory:
                assert traj_xi is not None
                traj_xi.append(xis.detach().cpu())

            h_in = h
            if not h_in.requires_grad:
                h_in = h_in.requires_grad_(True)
            V = self.V_theta(xis, h_in).sum()
            (grad_V,) = torch.autograd.grad(
                V,
                h_in,
                create_graph=self.training,
                retain_graph=True,
            )
            f = -grad_V
            v = (v + dt * f / m_b) / (1.0 + dt * gamma)
            h_new = h_in + dt * v
            if cfg.ln_after_step:
                h_new = self._project(h_new)
            h = h_new
            if return_trajectory:
                assert traj_h is not None
                traj_h.append(h.detach().cpu())

        return h, traj_h, traj_xi

    @torch.no_grad()
    def hippo_diagnostics(self) -> dict:
        """Diagnostic summary of the HiPPO state-space configuration."""
        return {
            "K": self.cfg.xi_channels,
            "basis": self.cfg.xi_basis,
            "theta": self.cfg.xi_theta if self.cfg.xi_basis == "legt" else None,
            "dt": self.xi_module.discretisation_step(),
            "learnable_dt": self.cfg.xi_learnable_dt,
        }


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def _smoke_test_matrices():
    print("=" * 60)
    print("[hippo] HiPPO matrix construction smoke test")

    A_t, B_t = make_hippo_legt(4)
    A_s, B_s = make_hippo_legs(4)
    print(f"  LegT A.shape={A_t.shape}  B.shape={B_t.shape}")
    print(f"  LegS A.shape={A_s.shape}  B.shape={B_s.shape}")
    # LegT: A diagonal entries should be -(2n+1) = [-1, -3, -5, -7]
    diag_t = np.diag(A_t)
    expected = np.array([-1.0, -3.0, -5.0, -7.0])
    assert np.allclose(diag_t, expected), f"LegT diag mismatch: {diag_t} vs {expected}"
    print(f"  LegT diag = {diag_t.tolist()} (expected -(2n+1))")
    # LegS: lower-triangular and zero strictly above the diagonal
    upper_s = np.triu(A_s, k=1)
    assert np.allclose(upper_s, 0.0), "LegS A is not lower-triangular"
    print(f"  LegS strictly-upper-triangle ≡ 0  (lower-triangular ✓)")
    # LegS diagonal: A_{nn} = -(n+1)
    diag_s = np.diag(A_s)
    expected_s = -np.arange(1, 5)
    assert np.allclose(diag_s, expected_s), f"LegS diag mismatch: {diag_s} vs {expected_s}"
    print(f"  LegS diag = {diag_s.tolist()} (expected -(n+1))")

    # Bilinear discretisation: stability check on LegT.
    Abar, Bbar = bilinear_discretize(A_t, B_t, 1.0 / 200.0)
    eigs = np.linalg.eigvals(Abar)
    assert np.all(np.abs(eigs) <= 1.0 + 1e-9), (
        f"bilinear-discretised A is not Schur: |eigs|max = {np.abs(eigs).max()}"
    )
    print(f"  LegT bilinear @ θ=200: max|eig(Â)| = {np.abs(eigs).max():.6f}  (≤ 1 ✓)")
    print("  HiPPO matrix construction: PASS")


def _smoke_test_forward():
    print("=" * 60)
    print("[hippo] MultiChannelHiPPO forward smoke test")
    torch.manual_seed(0)
    B, T, d, K = 2, 32, 8, 4

    for basis in ("legt", "legs"):
        h = torch.randn(B, T, d)
        mod = MultiChannelHiPPO(K=K, max_len=64, basis=basis, theta=64.0)
        c = mod(h)
        assert c.shape == (B, T, K, d), f"{basis}: got {c.shape}"
        finite = torch.isfinite(c).all().item()
        print(f"  {basis}: shape={tuple(c.shape)}  finite={finite}  "
              f"|c|_max={c.abs().max().item():.3e}")
        assert finite, f"{basis} produced non-finite values"


def _smoke_test_causality():
    print("=" * 60)
    print("[hippo] MultiChannelHiPPO strict-causality test")
    torch.manual_seed(0)
    B, T, d, K = 2, 32, 8, 4
    t_pert = 20

    for basis in ("legt", "legs"):
        h_a = torch.randn(B, T, d)
        h_b = h_a.clone()
        h_b[:, t_pert, :] += torch.randn(B, d) * 5.0  # large perturbation at t_pert
        mod = MultiChannelHiPPO(K=K, max_len=T, basis=basis, theta=64.0)
        c_a = mod(h_a)
        c_b = mod(h_b)
        diffs = (c_a - c_b).abs().reshape(B, T, -1).max(dim=-1).values
        pre = float(diffs[:, :t_pert].max().item())
        at = float(diffs[:, t_pert].max().item())
        post = float(diffs[:, t_pert + 1:].max().item())
        print(f"  {basis}: max|Δc| pre={pre:.2e}  @ t_pert={at:.2e}  post={post:.2e}")
        assert pre < 1e-6, (
            f"{basis} not causal: pre-perturbation max |Δc| = {pre:.2e}"
        )


def _smoke_test_orthogonality():
    """Check pairwise channel correlation on white-noise input.

    Per `Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md` §5.4,
    the K-EMA bank at the canonical α = (0, 0.5, 0.9, 0.99) grid has
    pairwise correlations on white noise of {0.943, 0.575, 0.198, 0.745,
    0.281, 0.575} (six pairs, mean ≈ 0.57). HiPPO-LegT should drop these
    substantially (Legendre orthogonality holds exactly only in the
    continuous-time limit; finite T and bilinear discretisation introduce
    a small bias, but the qualitative gap should be very large).
    """
    print("=" * 60)
    print("[hippo] LegT pairwise channel correlation on white noise")
    torch.manual_seed(0)
    B, T, d, K = 8, 1024, 1, 4
    h = torch.randn(B, T, d)
    mod = MultiChannelHiPPO(K=K, max_len=T, basis="legt", theta=200.0)
    c = mod(h)                                   # (B, T, K, 1)
    # Drop the last 100 timesteps to give the integrator's transient time
    # to die out before measuring correlations.
    c_flat = c[:, 100:, :, 0].reshape(-1, K)     # (B·(T-100), K)
    c_flat = c_flat - c_flat.mean(dim=0, keepdim=True)
    cov = c_flat.T @ c_flat / c_flat.shape[0]
    std = cov.diag().sqrt().clamp(min=1e-8)
    corr = cov / (std[:, None] * std[None, :])
    np.set_printoptions(precision=3, suppress=True)
    print(f"  pairwise corr matrix (K={K} HiPPO-LegT channels):")
    print("  ", str(corr.cpu().numpy()).replace("\n", "\n   "))

    # Compare against the K-EMA bank with the same α-init for context.
    from model_multixi import MultiChannelXi
    ema = MultiChannelXi(K=K, max_len=T,
                        alpha_inits=[0.0, 0.5, 0.9, 0.99], learnable=False)
    xi = ema(h)                                  # (B, T, K, 1)
    xi_flat = xi[:, 100:, :, 0].reshape(-1, K)
    xi_flat = xi_flat - xi_flat.mean(dim=0, keepdim=True)
    cov_e = xi_flat.T @ xi_flat / xi_flat.shape[0]
    std_e = cov_e.diag().sqrt().clamp(min=1e-8)
    corr_e = cov_e / (std_e[:, None] * std_e[None, :])
    print(f"  pairwise corr matrix (K={K} K-EMA channels α=(0, 0.5, 0.9, 0.99)):")
    print("  ", str(corr_e.cpu().numpy()).replace("\n", "\n   "))

    off_h = corr - torch.eye(K)
    off_e = corr_e - torch.eye(K)
    mean_off_h = off_h.abs().mean().item() * (K * K) / max(K * K - K, 1)
    mean_off_e = off_e.abs().mean().item() * (K * K) / max(K * K - K, 1)
    print(f"  mean |off-diagonal corr|: HiPPO={mean_off_h:.3f}   "
          f"K-EMA={mean_off_e:.3f}   gain factor = {mean_off_e/max(mean_off_h, 1e-6):.1f}×")


def _smoke_test_full_model():
    print("=" * 60)
    print("[hippo] full SPLM-MultiHiPPO model smoke test")
    torch.manual_seed(0)
    V = 257
    cfg = SPLMSARFMassLNMultiHiPPOConfig(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
        mass_mode="global", ln_after_step=True,
        xi_channels=4, xi_basis="legt", xi_theta=32.0,
    )
    net = ScalarPotentialLMSARFMassLNMultiHiPPO(cfg)
    print(f"  params  : {net.num_params():,}")
    print(f"  hippo   : {net.hippo_diagnostics()}")
    x = torch.randint(0, V, (2, 16))
    y = torch.randint(0, V, (2, 16))
    logits, loss = net(x, y)
    print(f"  loss    : {loss.item():.4f}")
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()
    print("  full-model forward + backward: PASS")


if __name__ == "__main__":
    _smoke_test_matrices()
    _smoke_test_forward()
    _smoke_test_causality()
    _smoke_test_orthogonality()
    _smoke_test_full_model()
    print("=" * 60)
    print("[hippo] All smoke tests passed.")
