"""
Multi-channel ξ extension of the SARF-faithful SPLM with LN-after-step.

Architectural change vs `energetic_minima/model_ln.py`
------------------------------------------------------
The baseline SPLM (E9) computes a single causal cumulative-mean context

    ξ_t = (h_1 + h_2 + ... + h_t) / t                               (baseline)

at every integration step and feeds (ξ_t, h_t) into a shared scalar
potential V_θ.  This gives the energy potential a *rank-1 summary of the
past*: any two prefixes with the same arithmetic mean are
indistinguishable to V_θ.  Comparing E9 to a matched-attention baseline at
the TinyStories scale-up produced a Δ ≈ −1.04 PPL gap; the dominant
candidate explanation is that this rank-1 summary is the structural
information bottleneck.

This module replaces the single ξ with **K causal weighted EMAs at
multiple decay scales**, parameterised by α₁, …, α_K ∈ (0, 1):

    ξ^{(k)}_t  =  Σ_{s ≤ t} W_k[t, s] · h_s

where the weights are the *normalised* causal exponential kernel

    W_k[t, s]  =  α_k^{(t-s)}  /  Σ_{r ≤ t} α_k^{(t-r)}        for s ≤ t.

With this normalisation:
  - α_k → 0   ⇒   ξ^{(k)}_t = h_t                       (instant, no past)
  - α_k → 1   ⇒   ξ^{(k)}_t = (h_1 + ... + h_t) / t     (causal cumulative mean — recovers baseline)
  - in between ⇒   weighted causal mean with effective horizon ~ 1/(1-α_k).

The K resulting ξ-channels (each of shape (B, T, d)) are concatenated
with h and fed into a wider V_θ:

    V_θ : ℝ^{(K+1)·d} → ℝ.

All other components (per-token logfreq mass, fixed γ, semi-implicit
damped Euler, LN-after-step) are unchanged.

Parameter cost
--------------
  - V_θ first layer: (K+1)·d → v_hidden   instead of   2·d → v_hidden
    (K=4, d=256, v_hidden=1024 ⇒  +0.79 M params for the standard config)
  - α₁,…,α_K: K scalars (negligible)
  - Total: ~ +0.8 M params on top of the 15.75 M E9 SPLM ⇒ ~16.55 M.
    Still below MatchedGPT's 19.45 M.

Numerical stability
-------------------
Weights are constructed in log-space and normalised via logsumexp per
row, then exponentiated.  This handles α near 0 and near 1 cleanly and
avoids the α^{-T} blow-up of the naïve cumsum trick.

Backward compatibility
----------------------
This is a *new* model class.  The locked E9 model class
`ScalarPotentialLMSARFMassLN` is untouched; old checkpoints continue to
load with the old class.

Pre-registration
----------------
Companion protocol: `docs/SPLM_multichannel_xi_pre-registered_protocol.md`
(experiment E11).  Pre-registration commit will be recorded there.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).parent
_PARENT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(_PARENT_DIR / "energetic_minima"))
sys.path.insert(0, str(_PARENT_DIR / "sarf_mass_variant"))

from model_ln import (  # noqa: E402
    ScalarPotentialLMSARFMassLN,
    SPLMSARFMassLNConfig,
)


_ALPHA_EPS = 1e-6


# ---------------------------------------------------------------------------
# Multi-channel ξ EMA
# ---------------------------------------------------------------------------

def causal_ema_weights(
    T: int,
    alpha: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return the (T, T) causal weighted-EMA weight matrix for one channel.

    W[t, s] = α^(t-s) / Z_t   for s ≤ t,
              0               otherwise,
    where Z_t = Σ_{r ≤ t} α^(t-r) ensures each row sums to 1.

    Parameters
    ----------
    T : int
        Sequence length.
    alpha : torch.Tensor
        0-d tensor in (0, 1).  Differentiable; clamped into [eps, 1-eps] for
        numerical stability of the log.
    dtype, device : passed through to the constructed tensor.

    Returns
    -------
    W : torch.Tensor of shape (T, T), float dtype, on `device`.
    """
    alpha_safe = alpha.clamp(min=_ALPHA_EPS, max=1.0 - _ALPHA_EPS)

    s_idx = torch.arange(T, dtype=dtype, device=device)
    diffs = s_idx.view(T, 1) - s_idx.view(1, T)              # (T, T): t - s
    causal = (diffs >= 0)                                    # mask s ≤ t

    log_alpha = torch.log(alpha_safe)
    log_W = log_alpha * diffs.clamp(min=0.0)
    log_W = log_W.masked_fill(~causal, float("-inf"))
    log_Z = torch.logsumexp(log_W, dim=1, keepdim=True)      # (T, 1)
    return torch.exp(log_W - log_Z)


class MultiChannelXi(nn.Module):
    """K parallel causal weighted EMAs over h with learnable decays.

    Forward:  h (B, T, d)  →  xis (B, T, K, d).

    Decays are parameterised as α_k = sigmoid(raw_α_k) so that they live in
    (0, 1) by construction.  Initialisation comes from `alpha_inits`.
    """

    def __init__(
        self,
        K: int,
        max_len: int,
        alpha_inits: List[float],
        learnable: bool = True,
    ):
        super().__init__()
        if K != len(alpha_inits):
            raise ValueError(
                f"alpha_inits length {len(alpha_inits)} != K={K}"
            )
        self.K = K
        self.max_len = max_len

        raw = torch.tensor(
            [
                math.log(max(a, _ALPHA_EPS) / max(1.0 - a, _ALPHA_EPS))
                for a in alpha_inits
            ],
            dtype=torch.float32,
        )
        if learnable:
            self.raw_alpha = nn.Parameter(raw)
        else:
            self.register_buffer("raw_alpha", raw)

    @property
    def alpha(self) -> torch.Tensor:
        """Effective decays α_k ∈ (0, 1)."""
        return torch.sigmoid(self.raw_alpha)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, d)  →  xis: (B, T, K, d)."""
        B, T, d = h.shape
        alphas = self.alpha
        xis = []
        for k in range(self.K):
            W_k = causal_ema_weights(T, alphas[k], h.dtype, h.device)  # (T, T)
            # (1, T, T) @ (B, T, d) → (B, T, d) via broadcasting
            xi_k = W_k.unsqueeze(0) @ h
            xis.append(xi_k)
        return torch.stack(xis, dim=2)                       # (B, T, K, d)


# ---------------------------------------------------------------------------
# V_theta over (xi_1, ..., xi_K, h)
# ---------------------------------------------------------------------------

class ScalarPotentialMultiXi(nn.Module):
    """V_θ : ℝ^{(K+1)·d} → ℝ.

    Same depth/hidden/MLP structure as `ScalarPotential` but accepts the
    concatenation of K xi-channels and h as input.
    """

    def __init__(self, d: int, hidden: int, depth: int, K: int):
        super().__init__()
        self.K = K
        self.d = d
        in_dim = (K + 1) * d
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        last = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(last.weight, std=0.002)
        nn.init.zeros_(last.bias)

    def forward(
        self, xis: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """xis: (B, T, K, d), h: (B, T, d)  →  V: (B, T, 1)."""
        B, T, K, d = xis.shape
        if K != self.K or d != self.d:
            raise ValueError(
                f"xis shape {tuple(xis.shape)} incompatible with "
                f"K={self.K} d={self.d}"
            )
        flat_xis = xis.reshape(B, T, K * d)
        cat = torch.cat([flat_xis, h], dim=-1)
        return self.net(cat)


# ---------------------------------------------------------------------------
# Multi-channel SPLM model
# ---------------------------------------------------------------------------

def log_spaced_alpha_inits(K: int, tau_max: float) -> List[float]:
    """Log-spaced α-init from §4.2 of `docs/Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md`.

    Returns alpha_k = 1 - 1 / tau_max^(k / (K - 1))  for k = 0, …, K - 1.

    Effective horizon spans tau_min = 1 (alpha_0 = 0) to tau_max (alpha_{K-1}).
    For K = 4, tau_max = 100: α ≈ (0, 0.785, 0.954, 0.99) — same shape as
    the hand-picked (0, 0.5, 0.9, 0.99) but log-uniform in horizon space.
    Used by R6.h.1 (Fix 2 only) to test whether the K-EMA bank's win over
    HiPPO-LegT is sensitive to α-init placement.
    """
    if K < 2:
        raise ValueError(f"log_spaced_alpha_inits requires K >= 2, got {K}")
    if tau_max <= 1.0:
        raise ValueError(f"tau_max must be > 1.0, got {tau_max}")
    return [
        float(1.0 - tau_max ** (-(k / (K - 1))))
        for k in range(K)
    ]


@dataclass
class SPLMSARFMassLNMultiXiConfig(SPLMSARFMassLNConfig):
    """Config for the multi-channel-ξ SPLM em_ln model.

    Two α-init modes:
      - "explicit"   : use `xi_alpha_inits` directly (default, matches all
                       pre-R6.h.1 ckpts including the K-EMA pilot at
                       α = (0, 0.5, 0.9, 0.99)).
      - "log_spaced" : compute α_k = 1 - 1 / xi_tau_max^(k / (K - 1)) at
                       __init__ from `xi_channels` and `xi_tau_max`,
                       overriding `xi_alpha_inits`. This is the §4.2 Fix 2
                       initialisation used by R6.h.1.

    Defaults give a 4-channel hand-picked multi-resolution past:
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
    xi_tau_max: float = 100.0              # only used when mode == "log_spaced"


class ScalarPotentialLMSARFMassLNMultiXi(ScalarPotentialLMSARFMassLN):
    """SARF-faithful SPLM with LN-after-step and **multi-channel ξ**.

    Inherits per-token mass, fixed/free γ, LN projection from
    `ScalarPotentialLMSARFMassLN`; replaces the single causal cumulative
    mean with K parallel weighted EMAs and widens V_θ accordingly.
    """

    def __init__(self, cfg: SPLMSARFMassLNMultiXiConfig):
        super().__init__(cfg)
        self.cfg: SPLMSARFMassLNMultiXiConfig = cfg

        # Resolve α-init: under "log_spaced" we ignore cfg.xi_alpha_inits and
        # compute from K + tau_max, then write the resolved list back to the
        # cfg so it serialises into the ckpt verbatim.
        if cfg.xi_alpha_init_mode == "log_spaced":
            alpha_inits = log_spaced_alpha_inits(
                cfg.xi_channels, cfg.xi_tau_max,
            )
            cfg.xi_alpha_inits = alpha_inits
        elif cfg.xi_alpha_init_mode == "explicit":
            alpha_inits = cfg.xi_alpha_inits
        else:
            raise ValueError(
                f"unknown xi_alpha_init_mode={cfg.xi_alpha_init_mode!r} "
                "(expected 'explicit' or 'log_spaced')"
            )

        # Replace the parent's V_theta with the multi-xi version.
        self.V_theta = ScalarPotentialMultiXi(
            d=cfg.d,
            hidden=cfg.v_hidden,
            depth=cfg.v_depth,
            K=cfg.xi_channels,
        )

        # K causal-EMA channels with their own (learnable) decays.
        self.xi_module = MultiChannelXi(
            K=cfg.xi_channels,
            max_len=cfg.max_len,
            alpha_inits=alpha_inits,
            learnable=cfg.xi_learnable,
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
            xis = self.xi_module(xi_input)                   # (B, T, K, d)
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
    def xi_alpha_values(self) -> List[float]:
        """Diagnostic: current α_k values."""
        return [float(a) for a in self.xi_module.alpha.detach().cpu().tolist()]


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def _smoke_test_ema_weights():
    print("=" * 60)
    print("[multixi] EMA weight construction smoke test")
    T = 8

    # α = 0  ⇒  identity
    W0 = causal_ema_weights(T, torch.tensor(0.0), torch.float32, torch.device("cpu"))
    err0 = (W0 - torch.eye(T)).abs().max().item()
    print(f"  α=0    : max |W - I| = {err0:.2e}   (should be ~1e-6)")
    assert err0 < 1e-3

    # α → 1  ⇒  causal cumulative mean
    W1 = causal_ema_weights(T, torch.tensor(0.999), torch.float32, torch.device("cpu"))
    expected_cumulative = torch.tril(torch.ones(T, T) / torch.arange(1, T + 1).view(T, 1).float())
    err1 = (W1 - expected_cumulative).abs().max().item()
    print(f"  α≈1    : max |W - cumulative_mean| = {err1:.2e}  (should be ~1e-3)")
    assert err1 < 1e-2

    # α = 0.5  ⇒  hand-computed for t=2
    Wh = causal_ema_weights(T, torch.tensor(0.5), torch.float32, torch.device("cpu"))
    # Row t=2: [0.25, 0.5, 1] / 1.75 = [0.1429, 0.2857, 0.5714]
    expected = torch.tensor([0.1429, 0.2857, 0.5714])
    errh = (Wh[2, :3] - expected).abs().max().item()
    print(f"  α=0.5  : max row-2 deviation from hand-computed = {errh:.2e}")
    assert errh < 1e-3

    # rows sum to 1
    row_sums = W1.sum(dim=1)
    print(f"  row sums (α≈1): mean={row_sums.mean().item():.6f}  "
          f"max_dev={(row_sums - 1).abs().max().item():.2e}")
    assert (row_sums - 1).abs().max().item() < 1e-4
    print("  EMA weight matrix construction: PASS")


def _smoke_test_multi_channel_xi():
    print("=" * 60)
    print("[multixi] MultiChannelXi forward smoke test")
    torch.manual_seed(0)
    B, T, d, K = 2, 12, 16, 4
    h = torch.randn(B, T, d)
    xi_mod = MultiChannelXi(
        K=K, max_len=T, alpha_inits=[0.0, 0.5, 0.9, 0.99], learnable=True,
    )
    xis = xi_mod(h)
    assert xis.shape == (B, T, K, d), f"got {xis.shape}"
    # channel 0 (α≈0) should equal h
    err_ch0 = (xis[:, :, 0, :] - h).abs().max().item()
    print(f"  shape={tuple(xis.shape)}   "
          f"channel-0 deviation from h = {err_ch0:.2e}  (α≈0)")
    assert err_ch0 < 1e-2
    # channel 3 (α=0.99) at t=T-1 should approximately match cumulative mean
    cum_mean_last = h.cumsum(dim=1)[:, -1, :] / float(T)
    err_ch3 = (xis[:, -1, 3, :] - cum_mean_last).abs().mean().item()
    print(f"  channel-3 last token vs cumulative mean: mean abs dev = {err_ch3:.2e}")
    print("  MultiChannelXi forward: PASS")


def _smoke_test_full_model():
    print("=" * 60)
    print("[multixi] full model smoke test (no logfreq)")
    torch.manual_seed(0)
    V = 257
    cfg = SPLMSARFMassLNMultiXiConfig(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
        mass_mode="global",
        xi_channels=4, xi_alpha_inits=[0.0, 0.5, 0.9, 0.99],
        xi_learnable=True,
    )
    net = ScalarPotentialLMSARFMassLNMultiXi(cfg)
    print(f"  params  : {net.num_params():,}")
    print(f"  α_k init: {net.xi_alpha_values()}")
    x = torch.randint(0, V, (2, 16))
    y = torch.randint(0, V, (2, 16))
    logits, loss = net(x, y)
    print(f"  loss    : {loss.item():.4f}")
    loss.backward()

    # gradient should reach raw_alpha
    grad = net.xi_module.raw_alpha.grad
    assert grad is not None, "raw_alpha got no gradient"
    print(f"  raw_α grad norm: {grad.norm().item():.3e}  (should be > 0)")
    assert grad.norm().item() > 0.0
    print("  full-model forward + backward: PASS")


if __name__ == "__main__":
    _smoke_test_ema_weights()
    _smoke_test_multi_channel_xi()
    _smoke_test_full_model()
    print("=" * 60)
    print("[multixi] All smoke tests passed.")
