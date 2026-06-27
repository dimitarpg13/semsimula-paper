"""
Xi-routed conservative attention PARFLM — all-to-all context mixing that
stays inside the conservative (potential-derived force) framework.

Motivation
----------
The sparse top-k V_phi mechanism of MultiXiPARFLM routes each query token to
only ``top_k`` past tokens through a pairwise scalar potential.  This is an
information bottleneck: a token can never receive a weighted contribution from
*every* past token the way softmax attention can.  The companion technical
report

    companion_notes/Context_Mixing_Mechanisms_in_the_Conservative_Framework.md
    §4  "Alternative A: xi-routed conservative attention"

shows how to import attention's all-to-all mixing WITHOUT breaking the
potential-derived force law.  This module implements that proposal.

The idea
--------
Define a scalar *attention potential* over the hidden state

    V_attn(h_t, H_{<t}) = - sum_{s<t} alpha(t, s) * phi(h_t, h_s)

and let the force be its (negative) gradient, exactly as for V_theta / V_phi:

    f_t = -grad_{h_t} V_attn.

The routing weights ``alpha(t, s)`` are computed from the **detached EMA
context** xi (which is itself derived from ``h.detach()`` when
``causal_force=True``):

    alpha(t, s) = softmax_{s<t} ( q(xi_t) . k(xi_s) / sqrt(d_k) ).

Because xi is detached, ``alpha`` is *constant* with respect to ``h_t``, so the
second-order routing term ``phi * grad_{h_t} alpha`` of standard attention
vanishes and the force is the gradient of a genuine scalar potential.  The
source slice ``h_s`` is detached too (the same causal reduction the sparse
V_phi already uses), so the per-token force is strictly causal:

    f_t = sum_{s<t} alpha(t, s) * grad_{h_t} phi(h_t, h_s).

Conservativity (Theorem, report §4.7): each ``alpha(t, s)`` is a constant
scalar w.r.t. ``h_t`` and ``phi`` is a scalar function of ``h_t``, so V_attn is
a finite linear combination of scalar functions of ``h_t`` — its gradient is by
construction a conservative force.  The model therefore passes the same
"force == -grad of a scalar U" contract as the rest of the SPLM/PARF family.

Kernels (report §4.4)
---------------------
  'dot' : phi(h_t, h_s) = (U h_t) . (W h_s) / sqrt(d_v)        [default]
          Force term  = alpha * U^T (W h_s) / sqrt(d_v)  — attention-like
          value transport along a learned direction.
  'rbf' : phi(h_t, h_s) = -||h_t - h_s||^2 / (2 sigma^2)
          Force term  = alpha * (h_s - h_t) / sigma^2  — spring-like pull
          toward attended tokens.

This is a *fully conservative* model: there is no Fock register pool, no
reverse channel, and no non-conservative exchange force.  The only context
mixing is the conservative attention potential plus the (self-energy)
multi-context V_theta.

Inheritance
-----------
    XiAttnPARFLM  ->  MultiXiPARFLM  ->  SparsePARFLM  ->  PARFLM

The sparse V_phi pair potential and its Gumbel-softmax score head are retired
(set to ``None``) and replaced by ``self.V_attn``.  Everything else — K-EMA xi
channels, multi-context V_theta, velocity-Verlet integration, mass model,
read-out head — is inherited unchanged.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

_THIS_DIR = Path(__file__).parent
_PARENT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(_PARENT_DIR / "multixi"))

from model_parf_multixi import (  # noqa: E402
    MultiXiPARFConfig,
    MultiXiPARFLM,
)

# Depth-conditioned V_theta routing support (optional): if the active
# V_theta is depth-conditioned, the layer step broadcasts the current
# layer index to it.  Imported lazily-safe (only an isinstance check).
try:
    from model_gaussian_vtheta import (  # noqa: E402
        DepthConditionedMultiContextGaussianVTheta,
    )
except Exception:  # pragma: no cover - keep the model importable standalone
    DepthConditionedMultiContextGaussianVTheta = ()  # type: ignore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class XiAttnPARFConfig(MultiXiPARFConfig):
    """Multi-Xi config extended with xi-routed conservative attention knobs.

    Inherits all multi-xi fields (xi_channels, xi_alpha_inits, ...) and the
    PARF stability/read-out knobs (causal_force, ln_after_step, use_output_bias,
    tie_embeddings, per_layer_v_phi_scale, ...).  The sparse-V_phi-specific
    fields (top_k, v_phi_*, score_head_*, gumbel_*) are inherited but UNUSED:
    the sparse pair potential is replaced by the attention potential below.

    Attention knobs:

      attn_n_heads        : int   — number of conservative-attention heads.
      attn_d_k            : int   — per-head routing (query/key) dimension.
      attn_d_v            : int   — per-head value/read-out dimension (dot kernel).
      attn_kernel         : str   — 'dot' (bilinear value transport) or
                            'rbf' (squared-distance spring).
      attn_init_scale     : float — std of the projection weight init.  Small so
                            the attention force enters as a perturbation on the
                            V_theta dynamics (mirrors v_phi_init_scale).
      attn_rbf_log_sigma_init : float — initial log(sigma) per head (rbf kernel).
      attn_route_detach_xi : bool — route from detached xi (True ⇒ conservative).
                            Leave True; exposed only for ablation.
    """
    attn_n_heads: int = 4
    attn_d_k: int = 48
    attn_d_v: int = 48
    attn_kernel: str = "dot"            # 'dot' | 'rbf'
    attn_init_scale: float = 0.02
    attn_rbf_log_sigma_init: float = 0.0
    attn_route_detach_xi: bool = True


# ---------------------------------------------------------------------------
# Xi-routed conservative attention potential
# ---------------------------------------------------------------------------
class XiRoutedConservativeAttention(nn.Module):
    """Scalar attention potential with xi-derived (h-independent) routing.

        V_attn(h) = - sum_t sum_{s<t} sum_head alpha^head(t,s) * phi^head(h_t, h_s)

    ``potential(h_in, h_src, xi_route, causal)`` returns the scalar
    ``V_attn`` summed over the batch / heads / token pairs.  The caller takes
    ``-autograd.grad(V_attn + ..., h_in)`` to obtain the conservative force, so
    this module never builds the force itself — it only assembles the potential.

    Shapes
    ------
      h_in     : (B, T, d)   query side, requires_grad.
      h_src    : (B, T, d)   source side, detached when causal_force.
      xi_route : (B, T, n_ctx, d)  detached EMA context for routing.
      causal   : (T, T) bool, True where s < t (strict lower-tri).
    """

    def __init__(
        self,
        d: int,
        xi_channels: int,
        n_heads: int = 4,
        d_k: int = 48,
        d_v: int = 48,
        kernel: str = "dot",
        init_scale: float = 0.02,
        rbf_log_sigma_init: float = 0.0,
    ):
        super().__init__()
        if kernel not in {"dot", "rbf"}:
            raise ValueError(f"attn kernel must be 'dot' or 'rbf', got {kernel!r}")
        self.d = d
        self.H = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.kernel = kernel
        xi_d = xi_channels * d

        # ── Routing projections from the (detached) xi summary ──
        self.W_q = nn.Linear(xi_d, n_heads * d_k, bias=False)
        self.W_k = nn.Linear(xi_d, n_heads * d_k, bias=False)
        nn.init.normal_(self.W_q.weight, std=init_scale)
        nn.init.normal_(self.W_k.weight, std=init_scale)

        if kernel == "dot":
            # Bilinear value-transport kernel phi = (U h_t).(W h_s)/sqrt(d_v).
            self.W_uq = nn.Linear(d, n_heads * d_v, bias=False)   # query read-out
            self.W_v = nn.Linear(d, n_heads * d_v, bias=False)    # source value
            nn.init.normal_(self.W_uq.weight, std=init_scale)
            nn.init.normal_(self.W_v.weight, std=init_scale)
            self.log_sigma = None
        else:  # 'rbf'
            # Spring kernel phi = -||h_t - h_s||^2 / (2 sigma_head^2).
            self.log_sigma = nn.Parameter(
                torch.full((n_heads,), float(rbf_log_sigma_init))
            )
            self.W_uq = None
            self.W_v = None

    # ------------------------------------------------------------------
    def _routing(
        self, xi_route: torch.Tensor, causal: torch.Tensor,
    ) -> torch.Tensor:
        """Attention weights alpha (B, H, T, T) from detached xi.

        alpha is constant w.r.t. h (xi_route is detached upstream), which is
        what keeps the induced force conservative.
        """
        B, T, n_ctx, d = xi_route.shape
        xi_flat = xi_route.reshape(B, T, n_ctx * d)
        q = self.W_q(xi_flat).view(B, T, self.H, self.d_k).transpose(1, 2)
        k = self.W_k(xi_flat).view(B, T, self.H, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (self.d_k ** -0.5)
        scores = scores.masked_fill(~causal.view(1, 1, T, T), float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        # t=0 row (and any fully-masked row) softmaxes all -inf -> nan; zero it.
        alpha = torch.nan_to_num(alpha, nan=0.0)
        return alpha

    # ------------------------------------------------------------------
    def potential(
        self,
        h_in: torch.Tensor,
        h_src: torch.Tensor,
        xi_route: torch.Tensor,
        causal: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar attention potential V_attn (summed over B, H, pairs)."""
        B, T, d = h_in.shape
        alpha = self._routing(xi_route, causal)              # (B, H, T, T)

        if self.kernel == "dot":
            uq = self.W_uq(h_in).view(B, T, self.H, self.d_v).transpose(1, 2)
            vs = self.W_v(h_src).view(B, T, self.H, self.d_v).transpose(1, 2)
            phi = torch.matmul(uq, vs.transpose(-1, -2)) * (self.d_v ** -0.5)
            # V_attn = - sum alpha * phi  (binding when alpha*phi > 0).
            return -(alpha * phi).sum()

        # 'rbf': phi = -||h_t - h_s||^2 / (2 sigma_head^2).
        # ||h_t - h_s||^2 = ||h_t||^2 + ||h_s||^2 - 2 h_t.h_s (B,T,T).
        a2 = (h_in * h_in).sum(-1, keepdim=True)             # (B, T, 1)
        b2 = (h_src * h_src).sum(-1, keepdim=True).transpose(1, 2)  # (B, 1, T)
        ab = torch.matmul(h_in, h_src.transpose(1, 2))       # (B, T, T)
        dist2 = (a2 + b2 - 2.0 * ab).clamp_min(0.0)          # (B, T, T)
        inv = (0.5 * torch.exp(-2.0 * self.log_sigma)).view(1, self.H, 1, 1)
        phi = -dist2.unsqueeze(1) * inv                      # (B, H, T, T)
        return -(alpha * phi).sum()


# ---------------------------------------------------------------------------
# Xi-routed conservative attention PARFLM
# ---------------------------------------------------------------------------
class XiAttnPARFLM(MultiXiPARFLM):
    """Multi-Xi PARFLM with the sparse V_phi replaced by conservative attention.

    The inherited ``_stack_forward`` (from PARFLM) drives ``self._layer_step``,
    which is overridden here so the per-layer potential is

        U^(l)_t = V_theta(xi_t, h_t) + s_l * V_attn(h_t, H_{<t}),

    and the force is the single autograd gradient ``-grad_h U`` — identical
    machinery to the sparse-V_phi model, only the pair term changed.
    """

    cfg: XiAttnPARFConfig

    def __init__(self, cfg: XiAttnPARFConfig):
        if not isinstance(cfg, XiAttnPARFConfig):
            raise TypeError(
                f"XiAttnPARFLM requires an XiAttnPARFConfig, got {type(cfg)!r}."
            )
        super().__init__(cfg)

        # Retire the sparse pair-potential machinery (unused here).  Setting
        # them to None drops their parameters from the optimiser/state_dict.
        self.V_phi = None
        self.score_head = None

        # Conservative attention potential (replaces sparse V_phi).
        self.V_attn = XiRoutedConservativeAttention(
            d=cfg.d,
            xi_channels=cfg.xi_channels,
            n_heads=cfg.attn_n_heads,
            d_k=cfg.attn_d_k,
            d_v=cfg.attn_d_v,
            kernel=cfg.attn_kernel,
            init_scale=cfg.attn_init_scale,
            rbf_log_sigma_init=cfg.attn_rbf_log_sigma_init,
        )
        self._attn_mask: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def _attn_mask_for(self, T: int, device: torch.device) -> torch.Tensor:
        """Cached strict-lower-triangular (s < t) bool mask."""
        if (
            self._attn_mask is None
            or self._attn_mask.shape[0] != T
            or self._attn_mask.device != device
        ):
            self._attn_mask = torch.tril(
                torch.ones(T, T, device=device, dtype=torch.bool), diagonal=-1
            )
        return self._attn_mask

    # ------------------------------------------------------------------
    def _layer_step(
        self,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        m_b: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """One velocity-Verlet step: V_theta self-energy + conservative attention."""
        cfg = self.cfg
        B, T, d = h.shape
        delta = h - h_prev

        # ── Multi-channel xi (detached when causal_force) ──
        xi_input = h.detach() if cfg.causal_force else h
        xis = self.xi_module(xi_input)                          # (B, T, K, d)

        h_in = h
        if not h_in.requires_grad:
            h_in = h_in.requires_grad_(True)

        # Causal reduction: source side frozen so the force is strictly causal.
        h_src = h_in.detach() if cfg.causal_force else h_in

        # ── Depth-conditioned V_theta routing (no-op otherwise) ──
        if isinstance(self.V_theta, DepthConditionedMultiContextGaussianVTheta):
            self.V_theta.set_active_layer(layer_idx)

        # ── V_theta self-energy (optional LN on h before eval) ──
        h_for_v = self.ln_before_v(h_in) if self.ln_before_v is not None else h_in
        V_th_per_token = self.V_theta(xis, h_for_v)            # (B, T, 1)

        # ── Conservative attention potential (replaces sparse V_phi) ──
        causal = self._attn_mask_for(T, h_in.device)
        xi_route = xis.detach() if cfg.attn_route_detach_xi else xis
        U_attn = self.V_attn.potential(h_in, h_src, xi_route, causal)

        s_ell = self.per_layer_scale(layer_idx)
        if s_ell is not None:
            U_attn = U_attn * s_ell

        # ── Single-call force: f = -grad_h (V_theta + V_attn) ──
        U = V_th_per_token.sum() + U_attn
        grad_U, = torch.autograd.grad(
            U, h_in,
            create_graph=self.training,
            retain_graph=True,
        )
        f = -grad_U

        if cfg.force_clamp_max is not None:
            f = f.clamp(-cfg.force_clamp_max, cfg.force_clamp_max)

        denom = 1.0 + dt * gamma
        h_new = h_in + delta / denom + (dt * dt / (m_b * denom)) * f

        if cfg.ln_after_step:
            h_new = self._project(h_new)
        return h_new

    # ------------------------------------------------------------------
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke():
    """Forward+backward + conservativity check across kernels / checkpoint."""
    for kernel in ("dot", "rbf"):
        for layer_ckpt in (False, True):
            tag = f"{kernel}{'+lc' if layer_ckpt else ''}"
            cfg = XiAttnPARFConfig(
                vocab_size=257, d=16, max_len=64, L=4,
                v_hidden=32, v_depth=2,
                mass_mode="global",
                xi_channels=4,
                xi_alpha_inits=[0.0, 0.5, 0.9, 0.99],
                xi_learnable=True,
                causal_force=True,
                ln_after_step=True,
                per_layer_v_phi_scale=True,
                use_layer_checkpoint=layer_ckpt,
                attn_n_heads=2,
                attn_d_k=8,
                attn_d_v=8,
                attn_kernel=kernel,
            )
            torch.manual_seed(0)
            net = XiAttnPARFLM(cfg)
            n = net.num_params()
            n_attn = sum(p.numel() for p in net.V_attn.parameters())
            alpha_str = ", ".join(f"{a:.3f}" for a in net.xi_alpha_values())
            print(
                f"[xi-attn-smoke/{tag}] params={n:,}  "
                f"V_attn={n_attn:,}  K={cfg.xi_channels} a=[{alpha_str}]"
            )
            assert net.V_phi is None and net.score_head is None

            x = torch.randint(0, cfg.vocab_size, (2, 12))
            y = torch.randint(0, cfg.vocab_size, (2, 12))

            net.train()
            logits, loss = net(x, targets=y)
            print(
                f"[xi-attn-smoke/{tag}] forward: logits {tuple(logits.shape)} "
                f"loss {loss.item():.4f}"
            )
            loss.backward()
            alpha_grad = net.xi_module.raw_alpha.grad
            assert alpha_grad is not None, "raw_alpha got no gradient"
            attn_has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in net.V_attn.parameters()
            )
            assert attn_has_grad, "V_attn received no gradient"
            print(
                f"[xi-attn-smoke/{tag}] backward OK  "
                f"(raw_a grad {alpha_grad.norm().item():.2e})"
            )
            net.zero_grad()

    # ── Conservativity check: force == -grad of a scalar potential ──
    # Build U(h) for a single layer and verify autograd force matches a
    # finite-difference gradient of the same scalar (symmetric Jacobian is
    # then automatic because the force IS a gradient).
    torch.manual_seed(1)
    cfg = XiAttnPARFConfig(
        vocab_size=257, d=8, max_len=32, L=1,
        v_hidden=16, v_depth=2, mass_mode="global",
        xi_channels=3, xi_alpha_inits=[0.0, 0.5, 0.9],
        causal_force=True, ln_after_step=False,
        attn_n_heads=2, attn_d_k=4, attn_d_v=4, attn_kernel="dot",
        attn_init_scale=0.3,
    )
    net = XiAttnPARFLM(cfg).eval()
    net.double()
    B, T = 1, 6
    h0 = torch.randn(B, T, cfg.d, dtype=torch.float64)
    causal = net._attn_mask_for(T, h0.device)
    # Routing (alpha) and source slice are detached constants under the
    # causal-force force law, so freeze them at h0 and vary ONLY the query.
    xis0 = net.xi_module(h0).detach()
    src0 = h0.detach()

    h = h0.clone().requires_grad_(True)
    U = net.V_attn.potential(h, src0, xis0, causal)
    g_auto, = torch.autograd.grad(U, h, create_graph=False)

    eps = 1e-6
    g_fd = torch.zeros_like(h0)
    for b in range(B):
        for t in range(T):
            for c in range(cfg.d):
                hp = h0.clone(); hp[b, t, c] += eps
                hm = h0.clone(); hm[b, t, c] -= eps
                Up = net.V_attn.potential(hp, src0, xis0, causal)
                Um = net.V_attn.potential(hm, src0, xis0, causal)
                g_fd[b, t, c] = (Up - Um) / (2 * eps)
    max_err = (g_auto - g_fd).abs().max().item()
    print(f"[xi-attn-smoke/conservativity] autograd vs finite-diff "
          f"max_err={max_err:.2e}")
    assert max_err < 1e-5, f"force is not -grad of a scalar (err {max_err})"

    print("\n\u2713 All xi-attention smoke tests passed.")


if __name__ == "__main__":
    _smoke()
