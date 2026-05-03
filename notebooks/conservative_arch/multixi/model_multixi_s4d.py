"""
S4D-based multi-channel ξ extension of the SARF-faithful SPLM with LN-after-step.

Architectural change vs `multixi/model_multixi_hippo.py`
-------------------------------------------------------
The HiPPO-LegT recurrence in `MultiChannelHiPPO` constructs ξ from a
**fixed structured A** (the Legendre transition matrix, Hurwitz, with
specific eigenvalue spectrum determined by the orthonormal basis). The
basis is therefore non-trainable: the spectrum of A is what it is.

This module replaces that structured A with a **learnable diagonal
complex A** — the S4D parametrisation (Gu, Goel, Gu, Re 2022, "On the
Parameterization and Initialization of Diagonal State Space Models"):

    dc(t)/dt  =  A · c(t) + B · h(t),    A = diag(λ_1, …, λ_K),  λ_k ∈ ℂ

with Re(λ_k) < 0 enforced structurally. The K complex eigenvalues
{λ_k} ARE the basis — they are gradient-trained jointly with the rest
of the model. Every K-channel linear-time-invariant context summary
with a stable diagonal A is reachable in this parametrisation, and
the K-EMA bank (real eigenvalues) and the diagonalised HiPPO-LegT
(specific complex eigenvalues) are both special cases.

Why this matters for SPLM
-------------------------
R6.a (HiPPO-LegT, fixed θ = 200) trailed the K-EMA pilot by 34 % on
val_ppl (19.82 vs 14.78) despite delivering 2.9× lower mean off-diagonal
channel correlation (0.24 vs 0.69) and 1.75× higher effective channels
(K_eff = 3.47 / 4 vs 1.98 / 4). The R6.c diagnostic established that
HiPPO's orthogonal modes ARE more informative in the §5.4 sense, but
the orthogonality did not transfer to val PPL. R6.e (HiPPO-LegT with
learnable Δt) closes ~half of that gap by giving the model one
adaptive horizon scalar; the residual gap is plausibly structural —
the Legendre eigenvalue spectrum is the wrong inductive bias for
TinyStories at K = 4.

S4D answers this directly: instead of asking "is Legendre the right
basis?" or "is exponential the right basis?", we ask "what is the
basis the gradient-descent process discovers when initialised from
HiPPO-LegT and given freedom to drift?" The answer is empirically
testable, and the optimisation cost is trivial (8 extra real
parameters at K = 4: 4 Re + 4 Im of the diagonal eigenvalues).

Discretisation: zero-order hold (ZOH)
-------------------------------------
For diagonal A the ZOH discrete-time matrices are closed-form, with no
matrix inverse:

    Â_kk  =  exp(λ_k · Δt)
    B̂_k   =  ((Â_kk − 1) / λ_k) · B_k

Convolution kernel:  M[t, k]  =  Â_kk^t · B̂_k  for t = 0, …, T − 1.
The K basis functions are 2 · Re(M[t, k]) — pure decaying exponentials
when Im(λ_k) = 0, damped sinusoids when Im(λ_k) ≠ 0. The factor 2
absorbs the conjugate-pair symmetry implicit in the Re-only output, so
K complex eigenvalues correspond to K real basis functions (no double
counting). This is exactly the S4D output convention.

Causal-leak compatibility
-------------------------
Identical to `model_multixi_hippo.py` and `model_multixi.py`. The
integrator computes the S4D state from `h.detach()` whenever
`cfg.causal_force = True` (the default), severing the autograd path
from c_t back to h_t. The S4D recurrence is strictly causal by
construction. See `docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`.

Parameter cost (K = 4 vs HiPPO-LegT R6.e baseline)
--------------------------------------------------
- log_neg_re: K real        → +4
- imag      : K real        → +4
- B_proj    : K real        → +4
- log_dt    : 1 real        → +1 (only if learnable_dt=True; same as R6.e)

Net: +8 trainable real scalars over R6.a (fixed-everything HiPPO-LegT),
+12 over R6.e if also learnable_dt. Total model size still ≈ 16.54 M
params (rounding error on the V_θ MLP).

References
----------
- Gu, A., Goel, K., Gu, A., Re, C. (2022). On the Parameterization and
  Initialization of Diagonal State Space Models. NeurIPS 2022.
- Gu, A., Goel, K., Re, C. (2022). Efficiently Modeling Long Sequences
  with Structured State Spaces. ICLR 2022.
- Gu, A., Dao, T., Ermon, S., Rudra, A., Re, C. (2020). HiPPO: Recurrent
  Memory with Optimal Polynomial Projections.
- docs/Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md §11.
- docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

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
from model_multixi_hippo import make_hippo_legt  # noqa: E402


# ---------------------------------------------------------------------------
# S4D eigenvalue / B initialisation strategies
# ---------------------------------------------------------------------------

def s4d_init_legt(K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Initialisation: K complex eigenvalues from HiPPO-LegT spectrum.

    This makes R6.i a strict generalisation of R6.a/R6.e: the model starts
    from the same continuous-time dynamics as HiPPO-LegT, then gradient
    descent is free to drift the eigenvalues.

    Returns
    -------
    eigvals : (K,) np.complex64 — initial diagonal of A (Hurwitz).
    B_init  : (K,) np.float32   — initial input projection.
    """
    A_legt, B_legt = make_hippo_legt(K)
    eigvals_full = np.linalg.eigvals(A_legt)
    # Some eigenvalues come as complex conjugate pairs; we keep all K.
    # Sort by descending real part so the most slowly-decaying mode is
    # channel 0 (matches the K-EMA convention of "α=0 first").
    order = np.argsort(-eigvals_full.real)
    eigvals = eigvals_full[order].astype(np.complex64)
    # Numerical safety: if any eigenvalue rounded to Re ≥ 0, push it
    # slightly negative (shouldn't trigger for well-conditioned LegT).
    eigvals.real[eigvals.real >= 0.0] = -1e-3
    B_init = B_legt.squeeze(-1)[order].astype(np.float32)
    return eigvals, B_init


def s4d_init_lin(K: int) -> Tuple[np.ndarray, np.ndarray]:
    """S4D-Lin initialisation (Gu, Goel, Gu, Re 2022): Re=−1/2, Im=π·(k+1/2).

    Provided as an alternative for ablation against the HiPPO-LegT init.
    """
    re = -0.5 * np.ones(K, dtype=np.float32)
    im = (np.pi * (np.arange(K, dtype=np.float32) + 0.5)).astype(np.float32)
    eigvals = (re + 1j * im).astype(np.complex64)
    B_init = np.ones(K, dtype=np.float32)
    return eigvals, B_init


# ---------------------------------------------------------------------------
# MultiChannelS4D module
# ---------------------------------------------------------------------------

class MultiChannelS4D(nn.Module):
    """K-channel context summary via diagonal complex-valued state-space ODE.

    Continuous-time:

        ċ(t)  =  A · c(t) + B · h(t),
        A     =  diag(λ_1, …, λ_K),   λ_k = −exp(log_neg_re_k) + i · imag_k,
        B     =  diag(B_proj_1, …, B_proj_K)  (per-channel input gain).

    Discretisation (ZOH, no matrix inverse — A is diagonal):

        Â_kk  =  exp(λ_k · Δt),
        B̂_k   =  ((Â_kk − 1) / λ_k) · B_proj_k.

    Output (real-valued ξ):

        ξ^k_t  =  2 · Re(c_k(t))  =  2 · Re( Σ_{s ≤ t} M[t − s, k] · h_s ),

    where  M[Δ, k] = Â_kk^Δ · B̂_k  (taken with Re-doubling).

    Shape contract is identical to MultiChannelHiPPO and MultiChannelXi:
    input (B, T, d) → output (B, T, K, d).

    Parameters
    ----------
    K : int
        Number of context channels (≡ number of complex eigenvalues).
    max_len : int
        Cached for the output ξ. Forward asserts T ≤ max_len.
    theta : float
        Initial discretisation horizon (Δt = 1 / theta). Carried over from
        the HiPPO trainer for compatibility; only used at init.
    learnable_dt : bool
        If True, expose log(Δt) as a trainable scalar (S4-style). Defaults
        to True since the entire point of S4D is end-to-end basis learning.
    eigval_init : {"legt", "s4d_lin"}
        How to initialise the diagonal eigenvalues of A.
    learnable_B : bool
        If True, B_proj is a Parameter; if False, a buffer (rare; default True).
    """

    def __init__(
        self,
        K: int,
        max_len: int,
        theta: float = 200.0,
        learnable_dt: bool = True,
        eigval_init: str = "legt",
        learnable_B: bool = True,
    ):
        super().__init__()
        self.K = K
        self.max_len = max_len
        self.theta = theta
        self.learnable_dt = learnable_dt
        self.eigval_init = eigval_init
        self.learnable_B = learnable_B

        if eigval_init == "legt":
            eigvals, B_init = s4d_init_legt(K)
        elif eigval_init == "s4d_lin":
            eigvals, B_init = s4d_init_lin(K)
        else:
            raise ValueError(
                f"unknown eigval_init: {eigval_init!r} "
                "(expected 'legt' or 's4d_lin')"
            )

        re_init = np.real(eigvals).astype(np.float32)
        im_init = np.imag(eigvals).astype(np.float32)
        # Hurwitz constraint: parametrise Re(λ_k) = -exp(log_neg_re_k) so it
        # is structurally negative under any value of log_neg_re_k. To avoid
        # log(0) at init, clamp re_init from above by a small negative number.
        re_init = np.minimum(re_init, -1e-3)
        log_neg_re_init = np.log(-re_init).astype(np.float32)

        self.log_neg_re = nn.Parameter(torch.tensor(log_neg_re_init))
        self.imag = nn.Parameter(torch.tensor(im_init))

        if learnable_B:
            self.B_proj = nn.Parameter(torch.tensor(B_init))
        else:
            self.register_buffer("B_proj", torch.tensor(B_init))

        dt_init = 1.0 / float(theta)
        if learnable_dt:
            self.log_dt = nn.Parameter(
                torch.tensor(math.log(dt_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "log_dt_buf",
                torch.tensor(math.log(dt_init), dtype=torch.float32),
            )

    # ------------------------------------------------------------------
    # Helpers: structurally Hurwitz λ; current Δt
    # ------------------------------------------------------------------

    def _eig_re(self) -> torch.Tensor:
        """Re(λ_k) = -exp(log_neg_re_k); structurally negative."""
        return -torch.exp(self.log_neg_re)

    def eigvals_cpu(self) -> torch.Tensor:
        """Build the diagonal of A as a length-K complex64 tensor on CPU.

        Why CPU. `torch.complex(...)` is not implemented for the MPS
        backend (PyTorch issue #77764). We therefore move the real and
        imaginary parts to CPU before constructing the complex view. The
        autograd graph through `log_neg_re` and `imag` is preserved by
        `.to('cpu')`.
        """
        re_cpu = self._eig_re().to(device="cpu")
        im_cpu = self.imag.to(device="cpu")
        return torch.complex(re_cpu, im_cpu)

    @torch.no_grad()
    def eigvals(self) -> torch.Tensor:
        """Public read-only accessor for the current eigenvalues (on CPU)."""
        return self.eigvals_cpu()

    def dt(self) -> torch.Tensor:
        if self.learnable_dt:
            return torch.exp(self.log_dt)
        return torch.exp(self.log_dt_buf)

    # ------------------------------------------------------------------
    # Convolution-kernel construction (CPU; complex; small)
    # ------------------------------------------------------------------

    def _kernel(self, T: int, device, dtype) -> torch.Tensor:
        """Build the S4D real-valued convolution kernel M ∈ ℝ^{T × K}.

        The construction is performed on CPU because (i) MPS support for
        complex64 ops is partial — `torch.complex`, `torch.exp(complex)`,
        and complex × complex elementwise multiplication are not all
        available — and (ii) the K × T = 4 × 1024 complex tensor is tiny,
        so the CPU detour costs ≪ 1 ms per forward pass. Autograd through
        `log_neg_re`, `imag`, `B_proj`, and `log_dt` is preserved by
        `.to('cpu')`.
        """
        eig_cpu = self.eigvals_cpu()                           # (K,) complex
        B_cpu = self.B_proj.to(device="cpu", dtype=torch.float32)  # (K,) real
        dt_cpu = self.dt().to(device="cpu", dtype=torch.float32)   # () real

        Abar = torch.exp(eig_cpu * dt_cpu)                     # (K,) complex
        # B̂_k = (Â_kk − 1) / λ_k · B_proj_k
        # For very small λ·dt the (Â − 1) / λ form is numerically fine
        # because |λ·dt| ≈ 0.005 in our regime; no special-case needed.
        Bbar = ((Abar - 1.0) / eig_cpu) * B_cpu                # (K,) complex

        # M[t, k] = Â_kk^t · B̂_k, computed via the closed-form
        #     log Â_kk = λ_k · dt   ⇒   Â_kk^t = exp(λ_k · dt · t)
        # This is O(T · K) flops, no inner loop, no matrix powers.
        t_idx = torch.arange(T, dtype=torch.float32)           # (T,)
        log_Abar = eig_cpu * dt_cpu                            # (K,) complex
        # Outer product: (T, 1) × (1, K) → (T, K) complex
        powers = torch.exp(log_Abar.unsqueeze(0) * t_idx.unsqueeze(1))
        kernel_complex = powers * Bbar.unsqueeze(0)            # (T, K) complex

        # Real-valued output absorbs conjugate-pair symmetry: 2·Re(·) gives
        # a length-K basis of decaying exponentials (Im λ = 0) and damped
        # sinusoids (Im λ ≠ 0). This is the standard S4D output convention.
        kernel_real = 2.0 * kernel_complex.real                # (T, K)
        return kernel_real.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, d = h.shape
        if T > self.max_len:
            raise ValueError(
                f"MultiChannelS4D: T={T} > max_len={self.max_len}"
            )
        K = self.K
        device, dtype = h.device, h.dtype

        M = self._kernel(T, device, dtype)                     # (T, K)

        # Lower-triangular Toeplitz convolution, identical layout to the
        # MultiChannelHiPPO LegT path so the memory pattern matches K-EMA:
        # K iterations, each one (T, T) × (B, T, d) → (B, T, d).
        ix = (
            torch.arange(T, device=device).view(T, 1)
            - torch.arange(T, device=device).view(1, T)
        )
        mask = (ix >= 0).to(dtype)
        ix_clamped = ix.clamp(min=0)
        cs: List[torch.Tensor] = []
        for k in range(K):
            W_k = M[ix_clamped, k] * mask                      # (T, T)
            c_k = W_k.unsqueeze(0) @ h                         # (B, T, d)
            cs.append(c_k)
        return torch.stack(cs, dim=2)                          # (B, T, K, d)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def discretisation_step(self) -> float:
        return float(self.dt().item())

    @torch.no_grad()
    def eigvals_numpy(self) -> np.ndarray:
        return self.eigvals_cpu().detach().cpu().numpy()

    @torch.no_grad()
    def b_proj_numpy(self) -> np.ndarray:
        return self.B_proj.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# SPLM model class with S4D multi-channel ξ
# ---------------------------------------------------------------------------

@dataclass
class SPLMSARFMassLNMultiS4DConfig(SPLMSARFMassLNConfig):
    """Config for the S4D-based multi-channel-ξ SPLM em_ln model.

    Defaults: K = 4 channels, eigenvalues initialised from the HiPPO-LegT
    spectrum, learnable Δt and B (smallest delta from R6.e). See
    `docs/Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md` §11.
    """

    xi_channels: int = 4
    xi_theta: float = 200.0                # init Δt = 1/theta
    xi_eigval_init: str = "legt"           # "legt" | "s4d_lin"
    xi_learnable_dt: bool = True
    xi_learnable_B: bool = True


class ScalarPotentialLMSARFMassLNMultiS4D(ScalarPotentialLMSARFMassLN):
    """SARF-faithful SPLM with LN-after-step and S4D-based multi-channel ξ.

    Replaces the structured-A HiPPO module of ScalarPotentialLMSARFMassLNMultiHiPPO
    with the diagonal-complex-A S4D module, exposing the K eigenvalues and
    the per-channel B as trainable parameters. The outer integrator and
    V_θ are unchanged; the K-channel ξ stack still feeds V_θ as a
    (B, T, K, d) tensor, identical interface to the K-EMA and HiPPO variants.

    The integration loop preserves the `h.detach()` causal-leak fix
    inherited from `ScalarPotentialLMSARFMassLNMultiHiPPO`.
    """

    def __init__(self, cfg: SPLMSARFMassLNMultiS4DConfig):
        super().__init__(cfg)
        self.cfg: SPLMSARFMassLNMultiS4DConfig = cfg

        self.V_theta = ScalarPotentialMultiXi(
            d=cfg.d,
            hidden=cfg.v_hidden,
            depth=cfg.v_depth,
            K=cfg.xi_channels,
        )

        self.xi_module = MultiChannelS4D(
            K=cfg.xi_channels,
            max_len=cfg.max_len,
            theta=cfg.xi_theta,
            learnable_dt=cfg.xi_learnable_dt,
            eigval_init=cfg.xi_eigval_init,
            learnable_B=cfg.xi_learnable_B,
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
    def s4d_diagnostics(self) -> dict:
        """Diagnostic summary of the S4D state-space configuration."""
        eigvals = self.xi_module.eigvals_numpy()
        return {
            "K": self.cfg.xi_channels,
            "eigval_init": self.cfg.xi_eigval_init,
            "dt": self.xi_module.discretisation_step(),
            "learnable_dt": self.cfg.xi_learnable_dt,
            "learnable_B": self.cfg.xi_learnable_B,
            "eigvals_re": eigvals.real.tolist(),
            "eigvals_im": eigvals.imag.tolist(),
            "b_proj": self.xi_module.b_proj_numpy().tolist(),
        }


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def _smoke_test_init():
    print("=" * 60)
    print("[s4d] init smoke test")
    eig_l, b_l = s4d_init_legt(4)
    eig_s, b_s = s4d_init_lin(4)
    print(f"  legt: eig.real = {eig_l.real.tolist()}")
    print(f"        eig.imag = {eig_l.imag.tolist()}")
    print(f"        B_init   = {b_l.tolist()}")
    print(f"  s4d_lin: eig.real = {eig_s.real.tolist()}")
    print(f"           eig.imag = {eig_s.imag.tolist()}")
    print(f"           B_init   = {b_s.tolist()}")
    assert (eig_l.real < 0).all(), "LegT init produced non-Hurwitz eigenvalue"
    assert (eig_s.real < 0).all(), "S4D-Lin init produced non-Hurwitz eigenvalue"
    print("  init: PASS")


def _smoke_test_forward():
    print("=" * 60)
    print("[s4d] MultiChannelS4D forward smoke test")
    torch.manual_seed(0)
    B, T, d, K = 2, 32, 8, 4
    for init in ("legt", "s4d_lin"):
        h = torch.randn(B, T, d)
        mod = MultiChannelS4D(
            K=K, max_len=64, theta=64.0,
            learnable_dt=True, eigval_init=init,
        )
        c = mod(h)
        assert c.shape == (B, T, K, d), f"{init}: got shape {c.shape}"
        finite = torch.isfinite(c).all().item()
        print(f"  init={init}: shape={tuple(c.shape)}  finite={finite}  "
              f"|c|_max={c.abs().max().item():.3e}")
        assert finite, f"{init}: produced non-finite values"
        # Backward sanity.
        loss = c.pow(2).mean()
        loss.backward()
        for n, p in mod.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (
                f"{init}: bad grad on {n}"
            )
        print(f"  init={init}: backward OK (all params have finite grads)")


def _smoke_test_causality():
    print("=" * 60)
    print("[s4d] MultiChannelS4D strict-causality test")
    torch.manual_seed(0)
    B, T, d, K = 2, 32, 8, 4
    t_pert = 20

    h_a = torch.randn(B, T, d)
    h_b = h_a.clone()
    h_b[:, t_pert, :] += torch.randn(B, d) * 5.0
    mod = MultiChannelS4D(K=K, max_len=T, theta=64.0, learnable_dt=True)
    c_a = mod(h_a)
    c_b = mod(h_b)
    diffs = (c_a - c_b).abs().reshape(B, T, -1).max(dim=-1).values
    pre = float(diffs[:, :t_pert].max().item())
    at = float(diffs[:, t_pert].max().item())
    post = float(diffs[:, t_pert + 1:].max().item())
    print(f"  max|Δc| pre={pre:.2e}  @ t_pert={at:.2e}  post={post:.2e}")
    assert pre < 1e-6, f"S4D not causal: pre-perturbation max |Δc| = {pre:.2e}"


def _smoke_test_full_model():
    print("=" * 60)
    print("[s4d] full SPLM-MultiS4D model smoke test")
    torch.manual_seed(0)
    V = 257
    cfg = SPLMSARFMassLNMultiS4DConfig(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
        mass_mode="global", ln_after_step=True,
        xi_channels=4, xi_theta=32.0, xi_eigval_init="legt",
        xi_learnable_dt=True, xi_learnable_B=True,
    )
    net = ScalarPotentialLMSARFMassLNMultiS4D(cfg)
    diag = net.s4d_diagnostics()
    print(f"  params : {net.num_params():,}")
    print(f"  s4d    : K={diag['K']}  init={diag['eigval_init']}  "
          f"dt={diag['dt']:.4g}  learnable_dt={diag['learnable_dt']}")
    print(f"          eigvals.re = {[f'{x:.3f}' for x in diag['eigvals_re']]}")
    print(f"          eigvals.im = {[f'{x:.3f}' for x in diag['eigvals_im']]}")
    print(f"          B_proj     = {[f'{x:.3f}' for x in diag['b_proj']]}")
    x = torch.randint(0, V, (2, 16))
    y = torch.randint(0, V, (2, 16))
    logits, loss = net(x, y)
    print(f"  forward: logits.shape={tuple(logits.shape)}  loss={float(loss.item()):.3f}")
    loss.backward()
    print("  backward: OK")


if __name__ == "__main__":
    _smoke_test_init()
    _smoke_test_forward()
    _smoke_test_causality()
    _smoke_test_full_model()
    print("=" * 60)
    print("ALL S4D SMOKE TESTS PASSED")
