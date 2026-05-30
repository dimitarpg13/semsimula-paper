"""
Sparse PARF-augmented SPLM (Q9c, Stage 1.5) — Gumbel-softmax top-k routing.

Reference
---------
docs/parf/On_Gumbel_softmax_sparsity_applied_to_V_phi.md

Architecture (one-paragraph summary)
------------------------------------
This module is the Stage 1.5 add-on to the dense Algorithm-A PARF model
of `model_parf.py`.  Where the dense model evaluates and aggregates the
pair-interaction scalar over every causal source (s < t) at every layer
-- O(B T^2) pair contributions per layer -- this sparse variant routes
through the framework's §5.2 prescribed top-k cutoff using a learned
score head plus a straight-through Gumbel-softmax mask.

For each query position t at every layer ell, the sparse layer computes:

  1. A score logit  pi(h_t, h_s)    for every causal source s < t,
     produced by a small MLP (the *score head*) on the same
     (h_q, h_s, h_q - h_s) features as the unstructured V_phi MLP.
  2. A Gumbel-perturbed score      z = (pi + g) / tau,
                                   g ~ Gumbel(0, 1),
     enforcing the strict-causal mask (s >= t -> z := -inf).
  3. A hard top-k mask              m_hard = top-k(z) over s < t,
     scattered to the (T, T) shape, then re-multiplied by the strict
     causal mask to defend against scatter into invalid positions when
     top-k temporarily exceeds the number of valid sources at small t.
  4. A soft mask                    y       = softmax(z, dim=s),
     used as the differentiable proxy for m_hard during backward.
  5. A composite straight-through  ~m_ts = stop_grad(m_hard - k * y) + k * y,
     equal to m_hard in forward and to k * y in backward.
  6. The pair sum                  U_pair = sum_{ts} V_phi(h_t, h_s) * ~m_ts,
     evaluated densely (P[b,t,s] for all s) in this Stage-1.5a prototype;
     a gathered O(T*k) form is sketched in the design doc as the
     Stage-1.5b optimisation.
  7. The total per-layer scalar U  =  sum_t V_theta(xi_t, h_t) + U_pair,
     and the velocity-Verlet step proceeds as in dense PARF.

Causality
---------
The strict-causal mask is enforced in TWO places:
  - On the Gumbel-perturbed scores via masked_fill(~causal, -inf), which
    guarantees softmax(z) places exactly zero weight on any s >= t.
  - On the hard mask via element-wise multiplication with the causal
    bool, which catches the edge case where top-k scattered into
    -inf indices (this happens whenever top_k > t valid causal sources
    on a given row; at row t=0 it always happens).

Both .detach() points of the dense PARF model are preserved
unchanged:
  - xi  = causal_cumulative_mean(h.detach())   (the SPLM xi-pool detach)
  - h_src = h.detach()                         (the PARF causal reduction)

The score head also receives the detached h_src for its source-side
input, so the routing decision -- like the pair force -- is computed
against a frozen-past field.  This keeps the gradient picture identical
to dense PARF: routing learns to exploit the PRESENT query at every
layer, but does not back-react on the past representations.

Bit-identity to dense PARF
--------------------------
With top_k >= T-1, gumbel_noise = False, and a fixed score_head, the
forward pass of SparsePARFLM is mathematically identical to PARFLM
modulo the score head's compute and the multiplication of P by the
constant 1.0 mask.  This is verified by the smoke test
`smoke_test_sparse.py`, and is the design-time guarantee that the
Stage-1.5 add-on is a strict superset of the Stage-1 baseline.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from model_parf import (  # noqa: E402
    PARFConfig,
    PARFLM,
    causal_cumulative_mean,
    _has_analytical_grad,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SparsePARFConfig(PARFConfig):
    """Sparse PARF configuration extending the dense PARFConfig.

    Sparse-specific knobs:

      top_k                : int — number of past tokens kept per query
                                   per layer.  Capped at T-1 internally.
      score_head_hidden    : int — hidden width of the per-pair score MLP.
      score_head_init_scale: float — Gaussian std for score-head weights.
                                     Small (default 0.02) so initial
                                     scores are near-uniform and the
                                     hard top-k acts like a random
                                     sample over the first few steps.
      gumbel_tau_init      : float — initial Gumbel-softmax temperature.
                                     The trainer is responsible for
                                     annealing this toward gumbel_tau_min
                                     over the course of training; this
                                     class only exposes set_gumbel_tau().
      gumbel_tau_min       : float — temperature floor used by the
                                     trainer; not consulted by the model
                                     directly.
      gumbel_noise         : bool  — enable/disable Gumbel noise during
                                     forward.  Always False in .eval();
                                     ignored at training time when False.
      score_head_use_detached_h_src : bool — preserve the dense PARF
                                     causal reduction for the score
                                     head's source-side input.
                                     Production default True.
    """

    # Sparse routing.
    top_k: int = 16
    score_head_hidden: int = 32
    score_head_init_scale: float = 0.02

    # Gumbel-softmax + STE.
    gumbel_tau_init: float = 1.0
    gumbel_tau_min: float = 0.1
    gumbel_noise: bool = True

    # Causality / parity.
    score_head_use_detached_h_src: bool = True

    # Stage-1.5b gathered V_phi: evaluate V_phi only at the top-k
    # indices instead of densely at all O(T^2) pairs.  Reduces V_phi
    # intermediates from (B,T,T,H) to (B,T,k,H) — a T/k reduction.
    # Gradients are bit-identical to Stage-1.5a for V_phi params and
    # h_in, and equivalent for score-head params in the limit tau->0.
    # See companion_notes/PARF_Stage_1_5b_design.md.
    use_gathered_v_phi: bool = False


# ---------------------------------------------------------------------------
# Score head
# ---------------------------------------------------------------------------
class ScoreHead(nn.Module):
    """Per-pair score logit pi(h_t, h_s) for the Gumbel-softmax routing.

    A small two-layer MLP with the same per-pair feature triplet as
    `MLPVPhi` ([h_q, h_s, h_q - h_s]), evaluated at the same (B, T, T)
    shape contract.  The first linear is split into per-input weight
    blocks (W_q, W_s, W_d) so we can apply each before the (T, T)
    outer-product broadcast -- this avoids the (B, T, T, 3d)
    intermediate that the naive cat-then-Linear formulation would
    otherwise dominate the score head's wall-clock with.

    Mathematical equivalence:
      Linear_3d->H( cat([h_q, h_s, h_q - h_s]) )
        = h_q W_q + h_s W_s + (h_q - h_s) W_d + b
        = h_q (W_q + W_d) + h_s (W_s - W_d) + b

    The output is a single scalar logit per pair, in (B, T, T).  No
    activation on the readout: the logits feed straight into the
    Gumbel-softmax temperature scaling.

    Initialisation
    --------------
    Weights are drawn from N(0, score_head_init_scale^2) so initial
    logits are near-uniform; the hard top-k at large tau is therefore
    essentially a uniform random sample over past tokens, which lets
    the score head's gradient signal drive the routing decision
    rather than the random-init asymmetry.
    """

    def __init__(self, cfg: SparsePARFConfig):
        super().__init__()
        d, H = cfg.d, cfg.score_head_hidden
        self.w_q = nn.Linear(d, H, bias=False)
        self.w_s = nn.Linear(d, H, bias=False)
        self.w_d = nn.Linear(d, H, bias=False)
        self.b1 = nn.Parameter(torch.zeros(H))
        self.w2 = nn.Linear(H, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=cfg.score_head_init_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_q: torch.Tensor, h_s: torch.Tensor) -> torch.Tensor:
        """Return the (B, T, T) logit tensor pi[b, t, s].

        h_q : (B, T, d) — query side, requires_grad in training.
        h_s : (B, T, d) — source side, .detach()-ed if cfg.score_head_use_detached_h_src.
        """
        proj_q = self.w_q(h_q)             # (B, T, H)
        proj_s = self.w_s(h_s)             # (B, T, H)
        proj_qd = self.w_d(h_q)            # (B, T, H)  — h_q part of (h_q - h_s)
        proj_sd = self.w_d(h_s)            # (B, T, H)  — h_s part of (h_q - h_s)

        proj_t = proj_q + proj_qd + self.b1     # (B, T, H)
        proj_u = proj_s - proj_sd                # (B, T, H)
        hidden = proj_t.unsqueeze(2) + proj_u.unsqueeze(1)  # (B, T, T, H)
        hidden = F.gelu(hidden)
        return self.w2(hidden).squeeze(-1)                   # (B, T, T)


# ---------------------------------------------------------------------------
# Sparse PARF language model
# ---------------------------------------------------------------------------
class SparsePARFLM(PARFLM):
    """PARF-augmented SPLM (Q9c) with Gumbel-softmax top-k pair routing.

    This subclass adds a `ScoreHead` and overrides `_layer_step` to apply
    the straight-through Gumbel-softmax mask described in
    `docs/parf/On_Gumbel_softmax_sparsity_applied_to_V_phi.md` §3-§4.
    All other layers (embedding, V_theta, V_phi, mass head, etc.) are
    inherited unchanged from `PARFLM`.

    Mutable state
    -------------
    The Gumbel-softmax temperature `tau` is held in `self._gumbel_tau`
    and is mutated by the trainer (or the user) via `set_gumbel_tau`.
    This avoids re-instantiating the model when annealing tau across
    training steps.  The initial value is `cfg.gumbel_tau_init`.
    """

    cfg: SparsePARFConfig  # narrow the type hint

    def __init__(self, cfg: SparsePARFConfig):
        if not isinstance(cfg, SparsePARFConfig):
            raise TypeError(
                f"SparsePARFLM requires a SparsePARFConfig, got {type(cfg)!r}."
            )
        super().__init__(cfg)
        self.score_head = ScoreHead(cfg)
        self.register_buffer(
            "_gumbel_tau",
            torch.tensor(float(cfg.gumbel_tau_init)),
            persistent=False,
        )

    # ------------------------------------------------------------------
    def set_gumbel_tau(self, tau: float) -> None:
        """Mutate the Gumbel-softmax temperature in place.

        Called by the trainer between optimiser steps.  The model itself
        does not anneal tau — the trainer owns the schedule (linear,
        cosine, log, etc.) and just pushes the current value here.
        """
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}.")
        self._gumbel_tau.fill_(float(tau))

    @property
    def gumbel_tau(self) -> float:
        return float(self._gumbel_tau.item())

    # ------------------------------------------------------------------
    def _sparse_mask(
        self,
        pi: torch.Tensor,
        causal: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """Build the straight-through composite mask `~m` of shape (B, T, T).

        Forward value equals m_hard (the top-k hard 0/1 mask, restricted
        to causal positions); backward gradient flows through k * y
        (the Gumbel-softmax soft mask), per the standard STE formula

            ~m = stop_grad(m_hard - k * y) + k * y .

        The strict-causal mask is enforced in two places (see module
        docstring).
        """
        cfg = self.cfg
        # Cap k at the maximum number of causal sources reachable on the
        # last row (T-1).  At smaller t there are even fewer valid sources;
        # the post-scatter `m_hard *= causal` handles those rows safely.
        k_eff = max(1, min(cfg.top_k, T - 1))
        tau = float(self._gumbel_tau)

        # Top-k path uses a -inf mask (topk doesn't backprop through
        # values directly, so -inf is safe here).
        pi_topk_masked = pi.masked_fill(~causal, float("-inf"))

        gumbel_active = self.training and cfg.gumbel_noise
        if gumbel_active:
            # Standard Gumbel(0, 1) draw via -log(-log(U)), U ~ U(0, 1).
            # Clamp the inner log to defend against the U = 0 corner.
            # Draw on the unmasked logits and re-apply the mask separately
            # to the top-k and soft branches so each branch sees the
            # right kind of mask (-inf for top-k, large-finite for soft).
            u = torch.rand_like(pi).clamp_min_(1e-9)
            g = -torch.log(-torch.log(u))
            z_unmasked = (pi + g) / tau
        else:
            # Eval / no-noise: deterministic top-k of the raw scores.
            # tau still controls the soft-mask sharpness, but the hard
            # selection is invariant to a positive scalar tau.
            z_unmasked = pi / tau

        # 3. Hard top-k selection along the source axis.  At rows where
        # the number of valid causal sources is < k_eff, topk returns
        # k_eff entries that include some -inf positions; step 3a strips
        # them (in particular row t = 0 has zero valid sources, so
        # m_hard collapses to all zeros there).
        z_topk = z_unmasked.masked_fill(~causal, float("-inf"))
        _, topk_idx = z_topk.topk(k_eff, dim=-1)                     # (B, T, k)
        m_hard = torch.zeros_like(pi).scatter(
            -1, topk_idx, 1.0,
        )                                                            # (B, T, T)
        # 3a. Re-apply causal mask.  Zeros out non-causal positions that
        # received scatter weight on rows with fewer than k_eff valid
        # sources.
        m_hard = m_hard * causal.to(m_hard.dtype)

        # 4. Soft mask via row softmax of the Gumbel-perturbed scores.
        # We mask non-causal positions with a LARGE FINITE negative
        # value (-1e9) rather than -inf, because the backward of
        # softmax through an all-(-inf) row produces NaN gradients
        # (0 / 0 in the Jacobian).  At -1e9, exp(-1e9) underflows to
        # exactly 0 in fp32 -> the masked positions contribute 0 weight
        # in softmax (matching the -inf behaviour in forward) AND the
        # backward is well-defined.  Rows with no valid causal source
        # (only row t = 0) become uniform 1/T after softmax; we zero
        # those rows post-hoc via `row_has_valid`.
        z_soft = z_unmasked.masked_fill(~causal, -1e9)
        y = torch.softmax(z_soft, dim=-1)
        row_has_valid = causal.any(dim=-1, keepdim=True).to(y.dtype)
        y = y * row_has_valid

        # 5. Composite straight-through mask.
        kf = float(k_eff)
        return (m_hard - kf * y).detach() + kf * y

    # ------------------------------------------------------------------
    def _sparse_topk_indices(
        self,
        pi: torch.Tensor,        # (B, T, T) — score logits
        causal: torch.Tensor,    # (B, T, T) — strict-causal bool mask
        T: int,
    ) -> tuple:
        """Top-k Gumbel-softmax routing in gathered form.

        Returns
        -------
        idx  : (B, T, k_eff) int64  — index of each top-k source per query
        m_g  : (B, T, k_eff) float  — straight-through STE composite mask
                                       gathered at the top-k positions
        """
        cfg = self.cfg
        k_eff = max(1, min(cfg.top_k, T - 1))
        tau = float(self._gumbel_tau)

        if self.training and cfg.gumbel_noise:
            u = torch.rand_like(pi).clamp_min_(1e-9)
            g = -torch.log(-torch.log(u))
            z_unmasked = (pi + g) / tau
        else:
            z_unmasked = pi / tau

        z_topk = z_unmasked.masked_fill(~causal, float("-inf"))
        _, idx = z_topk.topk(k_eff, dim=-1)                   # (B, T, k_eff)

        m_hard_g = torch.ones_like(idx, dtype=pi.dtype)        # (B, T, k_eff)

        z_soft = z_unmasked.masked_fill(~causal, -1e9)
        y = torch.softmax(z_soft, dim=-1)                      # (B, T, T)
        y_g = y.gather(-1, idx)                                # (B, T, k_eff)

        causal_exp = causal.unsqueeze(0).expand_as(y) if causal.dim() == 2 else causal
        causal_g = causal_exp.gather(-1, idx)                  # (B, T, k_eff)
        m_hard_g = m_hard_g * causal_g.to(m_hard_g.dtype)
        y_g = y_g * causal_g.to(y_g.dtype)

        kf = float(k_eff)
        m_g = (m_hard_g - kf * y_g).detach() + kf * y_g       # (B, T, k_eff)

        return idx, m_g

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
        """One velocity-Verlet step with Gumbel-softmax sparse routing.

        Mirrors `PARFLM._layer_step` step-for-step, with the dense
        pair-mask 1{s < t} replaced by the straight-through composite
        ~m of `_sparse_mask`.  When `top_k >= T - 1` and
        `gumbel_noise = False`, ~m collapses to the dense causal mask
        and this step is bit-equivalent to `PARFLM._layer_step` modulo
        the score-head computation (see smoke_test_sparse.py for the
        verification).
        """
        cfg = self.cfg
        B, T, d = h.shape
        delta = h - h_prev

        xi_input = h.detach() if cfg.causal_force else h
        xi_now = causal_cumulative_mean(xi_input)

        h_in = h
        if not h_in.requires_grad:
            h_in = h_in.requires_grad_(True)

        h_src = h_in.detach() if cfg.causal_force else h_in
        h_src_for_score = (
            h_in.detach() if cfg.score_head_use_detached_h_src else h_in
        )

        # 1. V_theta evaluation (unchanged from dense PARF).
        V_th_per_token = self.V_theta(xi_now, h_in)              # (B, T, 1)

        # 2. Score head -> routing.
        pi = self.score_head(h_in, h_src_for_score)              # (B, T, T)
        causal = self._pair_mask_for(T, h_in.device)             # (B, T, T) bool

        # 3. Pair potential — Stage-1.5b (gathered) or Stage-1.5a (dense).
        if cfg.use_gathered_v_phi:
            idx, m_g = self._sparse_topk_indices(pi, causal, T)  # (B,T,k), (B,T,k)
            idx_for_gather = idx.unsqueeze(-1).expand(-1, -1, -1, d)
            h_src_g = h_src.unsqueeze(1).expand(-1, T, -1, -1).gather(
                2, idx_for_gather,
            )                                                    # (B, T, k, d)
            V_phi_g = self.V_phi.forward_gathered(h_in, h_src_g) # (B, T, k)
            U_pair = (V_phi_g * m_g).sum()
        else:
            tilde_m = self._sparse_mask(pi, causal, T)           # (B, T, T)
            if cfg.use_grad_checkpoint and self.training:
                P = torch.utils.checkpoint.checkpoint(
                    self.V_phi, h_in, h_src, use_reentrant=False,
                )
            else:
                P = self.V_phi(h_in, h_src)                      # (B, T, T)
            U_pair = (P * tilde_m).masked_fill(~causal, 0.0).sum()

        # P8 patch B: per-layer V_φ scale.
        s_ell = self.per_layer_scale(layer_idx)
        if s_ell is not None:
            U_pair = U_pair * s_ell

        # ── Phase-2 force computation ────────────────────────────────────
        if _has_analytical_grad(self.V_theta):
            f_theta = -self.V_theta.analytical_grad(xi_now, h_in)  # (B, T, d)
            grad_phi, = torch.autograd.grad(
                U_pair, h_in,
                create_graph=self.training,
                retain_graph=True,
            )
            f = f_theta - grad_phi
        else:
            U = V_th_per_token.sum() + U_pair
            grad_U, = torch.autograd.grad(
                U, h_in,
                create_graph=self.training,
                retain_graph=True,
            )
            f = -grad_U

        denom = 1.0 + dt * gamma
        h_new = h_in + delta / denom + (dt * dt / (m_b * denom)) * f

        if cfg.ln_after_step:
            h_new = self._project(h_new)
        return h_new


# ---------------------------------------------------------------------------
# Smoke entry point (cheap sanity check, not the real smoke_test_sparse.py)
# ---------------------------------------------------------------------------
def _smoke():
    """Minimal one-step round-trip on CPU.  Not the real smoke test."""
    for layer_ckpt in (False, True):
        for gathered in (False, True):
            tag_parts = []
            if layer_ckpt:
                tag_parts.append("layer_ckpt")
            if gathered:
                tag_parts.append("gathered")
            tag = "+".join(tag_parts) or "baseline"
            cfg = SparsePARFConfig(
                vocab_size=257, d=16, max_len=64, L=4,
                v_hidden=32, v_depth=2,
                v_phi_d_type=4, v_phi_d_angle=2,
                v_phi_phi_hidden=8, v_phi_theta_hidden=8,
                v_phi_mlp_hidden=16,
                mass_mode="global",
                top_k=8,
                score_head_hidden=8,
                use_layer_checkpoint=layer_ckpt,
                use_gathered_v_phi=gathered,
            )
            torch.manual_seed(0)
            net = SparsePARFLM(cfg)
            print(f"[parf-sparse-smoke/{tag}] params: "
                  f"{sum(p.numel() for p in net.parameters()):,}")
            x = torch.randint(0, cfg.vocab_size, (2, 16))
            y = torch.randint(0, cfg.vocab_size, (2, 16))
            net.train()
            logits, loss = net(x, targets=y)
            print(f"[parf-sparse-smoke/{tag}] forward: logits "
                  f"{tuple(logits.shape)} loss {loss.item():.4f}")
            loss.backward()
            print(f"[parf-sparse-smoke/{tag}] backward OK.")


if __name__ == "__main__":
    _smoke()
