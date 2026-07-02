"""
FockPARFLM v2 — Q/K/V-structured creation protocol for latent registers.

Reference
---------
docs/Improving_the_Fock_Mechanism_to_match_Attention.md  §§9–10

Architecture (one-paragraph summary)
-------------------------------------
This module replaces the mean-conditioned creation gate of FockPARFLM v1
(model_fock_parf.py) with a Q/K/V-structured creation protocol.  Each
latent register carries a persistent query probe that attends over the
input tokens via scaled dot-product attention to (a) determine its
content (weighted sum of values) and (b) drive salience (max attention
weight).  Active registers then participate in PARF dynamics AND inject
a non-conservative generalised force Q_i on the token particles via a
reverse-channel attention readout.

Three missing properties of attention restored:
  1. Asymmetry  — register→token coupling ≠ token→register coupling
  2. Q/K/V decoupling — coupling strength (Q·K) ≠ content (V)
  3. Competitive normalisation — softmax budget Σ_j α_kj = 1

The reverse channel (§10.1) adds a non-conservative force term:
    Q_i = Σ_{k∈active} softmax_k(q_i · k_k^reg / √d) · v_k^reg
which breaks Newton's Third Law by design and cannot be derived from
any scalar potential.

What is novel vs standard attention: temporal persistence — registers
carry content across layers with exponential decay λ, providing
cross-layer working memory.  Standard attention is λ=0 (instantaneous).
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

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from model_parf_sparse import (  # noqa: E402
    SparsePARFConfig,
    SparsePARFLM,
)
from model_parf import causal_cumulative_mean  # noqa: E402


# ---------------------------------------------------------------------------
# Causal creation readout — shared by QKVCreationGate and QKVCreationGate_v21
# ---------------------------------------------------------------------------
def _causal_creation_readout(
    scores: torch.Tensor,
    V: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Position-dependent register content via cumulative softmax.

    The creation gate's attention scores are position-independent (the
    register query does not depend on token position), but the causal
    constraint requires that register content seen by position t only
    reflects tokens 1…t.  This is computed efficiently as a cumulative
    softmax: at position t, the normaliser is the prefix-sum of
    exp-scores up to t.

    Args:
        scores: (B, M, T) — temperature-scaled attention scores.
        V:      (B, T, d) — token values.

    Returns:
        r_new:     (B, M, d)    — register content at the last position
                                   (equivalent to full-sequence softmax).
        r_causal:  (B, T, M, d) — position-dependent register content.
        alpha_max: (B, M)       — max attention weight (from the full-
                                   sequence softmax, for salience update).
    """
    s_max = scores.max(dim=-1, keepdim=True).values          # (B, M, 1)
    exp_s = torch.exp(scores - s_max)                        # (B, M, T)
    Z = torch.cumsum(exp_s, dim=-1)                          # (B, M, T)

    # Cumulative weighted sum of V: Σ_{j≤t} exp_s[j] · V[j]
    weighted_V = exp_s.unsqueeze(-1) * V.unsqueeze(1)        # (B, M, T, d)
    numerator = torch.cumsum(weighted_V, dim=2)              # (B, M, T, d)
    r_causal_mt = numerator / Z.unsqueeze(-1).clamp(min=1e-8)  # (B, M, T, d)

    r_new = r_causal_mt[:, :, -1, :]                         # (B, M, d)
    r_causal = r_causal_mt.permute(0, 2, 1, 3)               # (B, T, M, d)

    # Full-sequence alpha for salience (= causal alpha at last position)
    alpha = exp_s / Z[:, :, -1:].clamp(min=1e-8)             # (B, M, T)
    alpha_max = alpha.max(dim=-1).values                     # (B, M)

    return r_new, r_causal, alpha_max


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class FockPARFConfig_v2(SparsePARFConfig):
    """Fock-space v2 config with Q/K/V-structured creation.

    v2-specific knobs:

      n_registers             : int — Pool size M (maximum latent particles).
      d_k                     : int — Key/query projection dimension.
      register_salience_decay : float — Exponential decay λ of register
                                salience per layer.  Memory lifetime ≈ 1/(1-λ).
      register_salience_threshold : float — σ_k must exceed this for register
                                k to participate in dynamics.
      register_init_scale     : float — Std of the learnable vacuum embeddings.
      stack_discipline        : bool — LIFO (salience-ordered) activation.
      destruction_gate_hidden : int — Hidden width of the destruction MLP.
      reverse_channel         : bool — When True, add the non-conservative
                                force Q_i (tokens read from active registers
                                via attention-like coupling).  §10 of the
                                design doc.
      tau_create_init         : float — Initial value for the learnable
                                creation-attention temperature τ.  Scores
                                are divided by τ instead of √d_k.  Small τ
                                → peaked (selective) attention; the model
                                learns to relax it if needed.  None means
                                fall back to the fixed 1/√d_k scaling.
    """
    n_registers: int = 16
    d_k: int = 64
    tau_create_init: Optional[float] = 0.1
    register_salience_decay: float = 0.5
    register_salience_threshold: float = 0.005
    register_init_scale: float = 0.02
    stack_discipline: bool = True
    destruction_gate_hidden: int = 64
    reverse_channel: bool = True


# ---------------------------------------------------------------------------
# Q/K/V-structured creation gate
# ---------------------------------------------------------------------------
class QKVCreationGate(nn.Module):
    """Per-register Q/K/V attention readout over input tokens.

    Each register k has a persistent query probe q_k = r_k @ W_Q[k].
    At each layer, registers attend over tokens to determine:
      - content:  r_k_new = Σ_j α_kj · v_j
      - salience signal: max_j(α_kj)

    The softmax enforces Σ_j α_kj = 1, importing the competitive
    budget constraint that independent sigmoid gates lack.
    """

    def __init__(
        self,
        d: int,
        d_k: int,
        M: int,
        init_scale: float = 0.02,
        tau_create_init: Optional[float] = None,
    ):
        super().__init__()
        self.M = M
        self.d_k = d_k

        self.W_Q = nn.Parameter(torch.randn(M, d, d_k) * init_scale)
        self.W_K = nn.Linear(d, d_k, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)

        nn.init.normal_(self.W_K.weight, std=init_scale)
        nn.init.normal_(self.W_V.weight, std=init_scale)

        if tau_create_init is not None:
            self.log_tau = nn.Parameter(
                torch.tensor(tau_create_init).log()
            )
        else:
            self.log_tau = None

        # Diagnostic capture (zero cost when off).
        self.capture_stats = False
        self.last_entropy = None   # mean creation-attention entropy (nats)

    def forward(
        self,
        h_tokens: torch.Tensor,
        register_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q/K/V-structured creation event with causal readout.

        Args:
            h_tokens: (B, T, d) — current token hidden states.
            register_states: (B, M, d) — current register states
                             (used to derive per-register queries).

        Returns:
            r_new:     (B, M, d)    — register content (full-sequence).
            r_causal:  (B, T, M, d) — position-dependent causal content.
            alpha_max: (B, M)       — max attention weight (salience signal).
        """
        B, T, d = h_tokens.shape
        M = self.M

        K = self.W_K(h_tokens)     # (B, T, d_k)
        V = self.W_V(h_tokens)     # (B, T, d)

        Q = torch.einsum("bmd,mdk->bmk", register_states, self.W_Q)  # (B, M, d_k)

        scores = torch.bmm(
            Q.reshape(B * M, 1, self.d_k),
            K.unsqueeze(1).expand(B, M, T, self.d_k).reshape(B * M, self.d_k, T),
        ).reshape(B, M, T)

        if self.log_tau is not None:
            tau = self.log_tau.exp().clamp(min=1e-4)
            scores = scores / tau
        else:
            scores = scores / (self.d_k ** 0.5)

        if self.capture_stats:
            with torch.no_grad():
                a = F.softmax(scores, dim=-1).clamp(min=1e-9)   # (B, M, T)
                self.last_entropy = float(-(a * a.log()).sum(dim=-1).mean())

        return _causal_creation_readout(scores, V)


# ---------------------------------------------------------------------------
# Q/K/V-structured creation gate v2.1 — fixes temperature collapse
# ---------------------------------------------------------------------------
class QKVCreationGate_v21(nn.Module):
    """Improved creation gate addressing the temperature collapse diagnostic.

    Three fixes over QKVCreationGate (v2):

      B1 — Per-register learnable temperature: each register k has its own
           log_tau[k], allowing the model to learn different selectivity
           levels per register.  Initialised at tau_init (default sqrt(d_k))
           for standard-scale attention.

      B2 — Per-register key subspaces: W_K is (M, d, d_k) instead of
           (d, d_k).  Each register projects tokens into a different key
           space, encouraging diverse attention patterns (analogous to
           per-head K projections in multi-head attention).

      B3 — (External) Orthogonal register embedding init is handled in
           FockMultiXiPARFLM.__init__, not here.

    Backward-compatible: when per_register_keys=False and M=1, this
    reduces to the original QKVCreationGate behaviour.
    """

    def __init__(
        self,
        d: int,
        d_k: int,
        M: int,
        init_scale: float = 0.02,
        tau_create_init: Optional[float] = None,
        per_register_keys: bool = False,
    ):
        super().__init__()
        self.M = M
        self.d_k = d_k
        self.per_register_keys = per_register_keys

        self.W_Q = nn.Parameter(torch.randn(M, d, d_k) * init_scale)
        self.W_V = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.W_V.weight, std=init_scale)

        if per_register_keys:
            self.W_K = nn.Parameter(torch.randn(M, d, d_k) * init_scale)
        else:
            self.W_K = nn.Linear(d, d_k, bias=False)
            nn.init.normal_(self.W_K.weight, std=init_scale)

        # Per-register learnable temperature (B1)
        if tau_create_init is not None:
            self.log_tau = nn.Parameter(
                torch.full((M,), math.log(tau_create_init))
            )
        else:
            self.log_tau = None

        # Diagnostic capture (zero cost when off).
        self.capture_stats = False
        self.last_entropy = None   # mean creation-attention entropy (nats)

    def forward(
        self,
        h_tokens: torch.Tensor,
        register_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q/K/V-structured creation with per-register temperature and keys.

        Args:
            h_tokens: (B, T, d)
            register_states: (B, M, d)

        Returns:
            r_new:     (B, M, d)    — register content (full-sequence).
            r_causal:  (B, T, M, d) — position-dependent causal content.
            alpha_max: (B, M)       — max attention weight per register.
        """
        B, T, d = h_tokens.shape
        M = self.M

        V = self.W_V(h_tokens)  # (B, T, d)

        Q = torch.einsum("bmd,mdk->bmk", register_states, self.W_Q)  # (B, M, d_k)

        if self.per_register_keys:
            K = torch.einsum("btd,mdk->bmtk", h_tokens, self.W_K)
            scores = torch.einsum("bmk,bmtk->bmt", Q, K)
        else:
            K = self.W_K(h_tokens)  # (B, T, d_k)
            scores = torch.bmm(
                Q.reshape(B * M, 1, self.d_k),
                K.unsqueeze(1).expand(B, M, T, self.d_k).reshape(B * M, self.d_k, T),
            ).reshape(B, M, T)

        if self.log_tau is not None:
            tau = self.log_tau.exp().clamp(min=1e-4)  # (M,)
            scores = scores / tau.unsqueeze(0).unsqueeze(-1)  # (B, M, T)
        else:
            scores = scores / (self.d_k ** 0.5)

        if self.capture_stats:
            with torch.no_grad():
                a = F.softmax(scores, dim=-1).clamp(min=1e-9)   # (B, M, T)
                self.last_entropy = float(-(a * a.log()).sum(dim=-1).mean())

        return _causal_creation_readout(scores, V)


# ---------------------------------------------------------------------------
# Reverse-channel: non-conservative force Q_i (§10)
# ---------------------------------------------------------------------------
class ReverseChannel(nn.Module):
    """Non-conservative Fock exchange force: tokens read from active registers.

    Implements §10.1:
        Q_i = Σ_{k∈active} softmax_k(q_i · k_k^reg / √d) · v_k^reg

    This force is non-conservative because:
      - It depends on relative inner products across all registers (softmax)
      - Q_i ≠ Q_j in general (asymmetry)
      - No scalar potential can generate it

    Stabilised variant (``stable=True``, §10.12 of the design doc / E5c)
    -------------------------------------------------------------------
    The vanilla readout is prone to gradient blow-ups because it feeds raw,
    unbounded hidden/register states into an unnormalised attention readout
    and injects an unnormalised force.  ``stable=True`` adds three bounding
    devices that preserve the directed/asymmetric routing (and hence the
    non-conservativity) while removing the explosion pathways:

      1. QK-normalisation — queries and keys are L2-normalised and the
         logits are scaled by a learnable temperature ``logit_scale``
         (clamped), so logits stay bounded regardless of ‖q‖, ‖k‖ and the
         softmax cannot saturate into spiky-gradient regimes.
      2. Output RMS-normalisation — the force ``Q_force`` is RMS-normalised
         per token, so its magnitude is bounded by construction (only its
         *direction* is learned), capping ∂L/∂W_V_rev.  With ``soft_norm=True``
         the hard floor (eps=1e-6) is replaced by a soft floor (eps=1.0):
         ``Q / sqrt(mean(Q²)+1)``.  This is the identity for small forces
         (backward Jacobian ~1, no 1/‖Q‖ blow-up) and a soft cap at ~unit RMS
         for large ones, so the *pre-clip* projection gradient stays O(1).
      3. Optional pre-LayerNorm (``pre_ln=True``) — q/k/v inputs are
         LayerNorm-ed, bounding magnitudes at the source.
    """

    def __init__(
        self,
        d: int,
        d_k: int,
        init_scale: float = 0.02,
        stable: bool = False,
        pre_ln: bool = True,
        soft_norm: bool = False,
    ):
        super().__init__()
        self.d_k = d_k
        self.stable = stable
        self.pre_ln = bool(stable and pre_ln)
        # Output normalisation floor.  Hard RMS-norm (eps=1e-6) pins the force to
        # unit RMS but its 1/‖Q‖ Jacobian blows up the *pre-clip* gradient when
        # the natural force is small.  Soft-floored norm (eps=1.0) divides by
        # sqrt(mean(Q²)+1): identity for small forces (Jacobian ~1, no blow-up)
        # and a soft cap at ~unit RMS for large ones.
        self.soft_norm = bool(stable and soft_norm)
        self.out_norm_eps = 1.0 if self.soft_norm else 1e-6
        self.W_Q_rev = nn.Linear(d, d_k, bias=False)
        self.W_K_rev = nn.Linear(d, d_k, bias=False)
        self.W_V_rev = nn.Linear(d, d, bias=False)

        nn.init.normal_(self.W_Q_rev.weight, std=init_scale)
        nn.init.normal_(self.W_K_rev.weight, std=init_scale)
        nn.init.normal_(self.W_V_rev.weight, std=init_scale)

        if self.pre_ln:
            self.ln_h = nn.LayerNorm(d)
            self.ln_r = nn.LayerNorm(d)
        else:
            self.ln_h = None
            self.ln_r = None

        if self.stable:
            # CLIP-style learnable, clamped logit temperature for QK-norm.
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
            self.logit_scale_max = 100.0
        else:
            self.logit_scale = None
            self.logit_scale_max = None

        # Diagnostic capture (zero cost when off).  When capture_stats is True
        # the forward pass stashes the reverse-attention entropy and peak
        # weight so a probe can attribute how focused / diffuse the token->
        # register readout is without changing the return signature.
        self.capture_stats = False
        self.last_entropy = None      # mean readout entropy (nats)
        self.last_alpha_max = None    # mean peak attention weight

    def forward(
        self,
        h_tokens: torch.Tensor,
        r_active: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the non-conservative force on each token.

        Accepts either global registers (B, M, d) or position-dependent
        causal registers (B, T, M, d).  The causal variant ensures that
        the force applied to token t only reflects register content
        derived from tokens 1…t, preventing future-token information
        from leaking through the reverse channel.

        Args:
            h_tokens:    (B, T, d)           — token hidden states.
            r_active:    (B, M, d)           — global register states, OR
                         (B, T, M, d)        — position-dependent causal registers.
            active_mask: (B, M)              — boolean; inactive registers masked.

        Returns:
            Q_force: (B, T, d) — non-conservative Fock exchange force.
        """
        B, T, d = h_tokens.shape
        position_dependent = r_active.dim() == 4

        # (Optional) pre-LayerNorm bounds magnitudes at the source.
        h_in = self.ln_h(h_tokens) if self.pre_ln else h_tokens
        r_in = self.ln_r(r_active) if self.pre_ln else r_active

        q = self.W_Q_rev(h_in)                        # (B, T, d_k)
        if position_dependent:
            M = r_active.shape[2]
            k = self.W_K_rev(r_in)                    # (B, T, M, d_k)
            v = self.W_V_rev(r_in)                    # (B, T, M, d)
        else:
            M = r_active.shape[1]
            k = self.W_K_rev(r_in)                    # (B, M, d_k)
            v = self.W_V_rev(r_in)                    # (B, M, d)

        if self.stable:
            # QK-norm: unit-normalise q, k and use a clamped learnable
            # temperature so logits cannot grow with ‖q‖, ‖k‖.
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            logit_scale = self.logit_scale.exp().clamp(max=self.logit_scale_max)
            if position_dependent:
                scores = torch.einsum("btk,btmk->btm", q, k) * logit_scale
            else:
                scores = torch.matmul(q, k.transpose(-2, -1)) * logit_scale
        else:
            if position_dependent:
                scores = torch.einsum("btk,btmk->btm", q, k) / (self.d_k ** 0.5)
            else:
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        mask_expanded = active_mask.unsqueeze(1).expand(B, T, M)
        scores = scores.masked_fill(~mask_expanded, -1e9)

        has_active = active_mask.any(dim=-1, keepdim=True).unsqueeze(1)
        alpha = F.softmax(scores, dim=-1)             # (B, T, M)
        alpha = alpha * has_active.float()

        if self.capture_stats:
            with torch.no_grad():
                valid = has_active.squeeze(-1).squeeze(1).float()  # (B,) 1 if any active
                denom = valid.sum().clamp(min=1.0) * float(T)
                a = alpha.clamp(min=1e-9)
                ent = -(a * a.log()).sum(dim=-1)          # (B, T)
                amax = alpha.max(dim=-1).values           # (B, T)
                w = valid.unsqueeze(-1)                    # (B, 1)
                self.last_entropy = float((ent * w).sum() / denom)
                self.last_alpha_max = float((amax * w).sum() / denom)

        if position_dependent:
            Q_force = torch.einsum("btm,btmd->btd", alpha, v)
        else:
            Q_force = torch.matmul(alpha, v)          # (B, T, d)

        if self.stable:
            # Output norm: bound the injected force magnitude per token.  With
            # the hard floor (eps=1e-6) this is unit-RMS (direction only); with
            # the soft floor (eps=1.0) small forces pass through ~unchanged and
            # only large ones are capped, keeping the backward Jacobian O(1).
            # Zero force on tokens with no active register stays exactly zero
            # (has_active gate above).
            rms = Q_force.pow(2).mean(dim=-1, keepdim=True).add(self.out_norm_eps).sqrt()
            Q_force = Q_force / rms

        return Q_force


# ---------------------------------------------------------------------------
# Destruction gate (reused from v1 with minor refinement)
# ---------------------------------------------------------------------------
class DestructionGate_v2(nn.Module):
    """Per-register destruction gate: r_k → destruction probability ∈ [0, 1]."""

    def __init__(self, d: int, hidden: int, init_scale: float = 0.02):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=init_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """r: (B, M, d) → (B, M) in [0, 1]."""
        return torch.sigmoid(self.net(r).squeeze(-1))


# ---------------------------------------------------------------------------
# FockPARFLM v2
# ---------------------------------------------------------------------------
class FockPARFLM_v2(SparsePARFLM):
    """PARFLM with Q/K/V-structured Fock creation and non-conservative exchange.

    Inherits the full sparse PARF dynamics and replaces the v1 mean-conditioned
    creation gate with a Q/K/V attention readout.  Optionally adds a reverse
    channel (non-conservative force Q_i) where tokens read from active
    registers.

    Register lifecycle per layer ℓ:
      1. Q/K/V creation gate: each register attends over tokens to update
         its content and salience.
      2. Active mask derived from salience (optionally with LIFO discipline).
      3. Active registers concatenated to token hidden states.
      4. PARF dynamics (V_θ + sparse V_φ + damped Verlet) on extended state.
      5. (Optional) Reverse channel: tokens read from active registers via
         attention-like coupling → non-conservative force Q_i.
      6. Destruction gate fires per active register → decays salience.
      7. Token and register states split apart for next layer.
    """

    cfg: FockPARFConfig_v2

    def __init__(self, cfg: FockPARFConfig_v2):
        if not isinstance(cfg, FockPARFConfig_v2):
            raise TypeError(
                f"FockPARFLM_v2 requires a FockPARFConfig_v2, got {type(cfg)!r}."
            )
        super().__init__(cfg)
        M, d, L = cfg.n_registers, cfg.d, cfg.L

        # Learnable vacuum embedding for each register slot.
        self.register_embed = nn.Parameter(
            torch.randn(M, d) * cfg.register_init_scale
        )

        # Q/K/V creation gate (shared across layers — the query comes
        # from the evolving register state, so layer-specificity is
        # implicit in the register content).
        self.creation_gate = QKVCreationGate(
            d, cfg.d_k, M,
            init_scale=cfg.register_init_scale,
            tau_create_init=cfg.tau_create_init,
        )

        # Per-layer destruction gates.
        self.destruction_gates = nn.ModuleList([
            DestructionGate_v2(d, cfg.destruction_gate_hidden,
                               init_scale=cfg.register_init_scale)
            for _ in range(L)
        ])

        # Reverse channel (non-conservative force Q_i).
        if cfg.reverse_channel:
            self.reverse_ch = ReverseChannel(
                d, cfg.d_k, init_scale=cfg.register_init_scale
            )
            # Learnable gate on the reverse channel magnitude, initialised to
            # zero so the force starts fully off.  The model learns when to
            # open it.  Applied as tanh(scale) ∈ (-1, +1), keeping training
            # stable while all registers fire from step 0.
            self.reverse_channel_scale = nn.Parameter(torch.zeros(1))
        else:
            self.reverse_ch = None
            self.reverse_channel_scale = None

    # ------------------------------------------------------------------
    def _init_registers(
        self, B: int, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialise register states and salience for a new forward pass."""
        M, d = self.cfg.n_registers, self.cfg.d
        r = self.register_embed.unsqueeze(0).expand(B, M, d).clone()
        # Start fully active (salience=1) so registers fire from step 0
        # and the destruction gate learns when to annihilate them.
        # This avoids the cold-start problem at short-sequence scale where
        # max_j(alpha_kj) ~ 1/T is too small to build salience from zero.
        salience = torch.ones(B, M, device=device)
        return r, salience

    # ------------------------------------------------------------------
    def _active_mask(self, salience: torch.Tensor) -> torch.Tensor:
        """Derive the boolean active mask from salience, optionally with LIFO."""
        cfg = self.cfg
        above_thresh = salience > cfg.register_salience_threshold

        if not cfg.stack_discipline:
            return above_thresh

        # LIFO: sort by salience descending; only contiguous prefix active.
        sorted_sal, sort_idx = salience.sort(dim=-1, descending=True)
        sorted_above = sorted_sal > cfg.register_salience_threshold
        sorted_active = torch.cumprod(sorted_above.float(), dim=-1).bool()

        active = torch.zeros_like(sorted_active)
        active.scatter_(1, sort_idx, sorted_active)
        return active

    # ------------------------------------------------------------------
    def _fock_v2_layer_step(
        self,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        r: torch.Tensor,
        salience: torch.Tensor,
        m_b: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One Fock v2 layer step with Q/K/V creation and optional reverse channel.

        1. Q/K/V creation gate → update register content and salience.
        2. Build active mask → select registers.
        3. Concatenate [tokens, active_registers] and run Verlet step.
        4. (Optional) Reverse channel: non-conservative force Q_i on tokens.
        5. Destruction gate → decay salience.
        6. Split states back.

        Returns updated (h_new, h, r_new, salience_new).
        """
        cfg = self.cfg
        B, T, d = h.shape
        M = cfg.n_registers
        decay = cfg.register_salience_decay

        # --- 1. Q/K/V creation gate ---
        r_new_content, r_causal, alpha_max = self.creation_gate(h, r)

        blend = salience.unsqueeze(-1)  # (B, M, 1), in [0, ~1]
        r = blend * r + (1.0 - blend) * r_new_content

        salience = salience * decay + alpha_max * (1.0 - decay)

        # --- 2. Active mask ---
        active = self._active_mask(salience)  # (B, M) bool

        # --- 3. Concatenate active registers ---
        active_float = active.float().unsqueeze(-1)  # (B, M, 1)
        r_gated = r * active_float

        h_ext = torch.cat([h, r_gated], dim=1)        # (B, T+M, d)
        h_prev_ext = torch.cat([h_prev, r_gated], dim=1)

        # Extended mass: registers share the global mass.
        if isinstance(m_b, torch.Tensor) and m_b.dim() >= 2:
            m_reg = self.m_global.expand(B, M, 1)
            m_ext = torch.cat([m_b, m_reg], dim=1)
        else:
            m_ext = m_b

        # --- 4. Run standard PARF dynamics on extended state ---
        h_ext_new = super()._layer_step(
            h_ext, h_prev_ext, m_ext, gamma, dt, layer_idx=layer_idx,
        )

        # --- 5. Split back ---
        h_new = h_ext_new[:, :T, :]
        r_new = h_ext_new[:, T:, :]
        r_new = torch.where(active_float.bool(), r_new, r)

        # --- 6. Reverse channel (non-conservative force Q_i) ---
        if self.reverse_ch is not None and active.any():
            # Use position-dependent causal register content so that
            # the force on token t only reflects tokens 1…t.
            Q_force = self.reverse_ch(h_new, r_causal, active)
            scale = torch.tanh(self.reverse_channel_scale)
            h_new = h_new + (dt * dt / m_b) * scale * Q_force

            if cfg.ln_after_step:
                h_new = self._project(h_new)

        # --- 7. Destruction gate ---
        g_destroy = self.destruction_gates[layer_idx](r_new)  # (B, M)
        salience = salience * (1.0 - g_destroy * active.float())

        return h_new, h, r_new, salience

    # ------------------------------------------------------------------
    def _stack_forward(
        self,
        h0: torch.Tensor,
        x: torch.Tensor,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Walk the L Fock v2 layers with Q/K/V register lifecycle."""
        cfg = self.cfg
        B, T, d = h0.shape
        gamma, dt = self.gamma, cfg.dt
        m_b = self.compute_mass(x)

        r, salience = self._init_registers(B, h0.device)

        h = h0
        h_prev = h0

        traj: Optional[List[torch.Tensor]] = None
        if return_trajectory:
            traj = [h.detach().cpu()]

        for ell in range(cfg.L):
            h_new, h_prev_out, r, salience = self._fock_v2_layer_step(
                h, h_prev, r, salience, m_b, gamma, dt, layer_idx=ell,
            )
            h_prev = h_prev_out
            h = h_new
            if traj is not None:
                traj.append(h.detach().cpu())

        return h, traj

    # ------------------------------------------------------------------
    def get_fock_v2_overhead(self) -> int:
        """Count parameters specific to the Fock v2 augmentation."""
        overhead = self.register_embed.numel()
        overhead += sum(p.numel() for p in self.creation_gate.parameters())  # includes log_tau
        for gate in self.destruction_gates:
            overhead += sum(p.numel() for p in gate.parameters())
        if self.reverse_ch is not None:
            overhead += sum(p.numel() for p in self.reverse_ch.parameters())
            overhead += self.reverse_channel_scale.numel()  # 1 scalar gate
        return overhead


# ---------------------------------------------------------------------------
# Smoke entry point
# ---------------------------------------------------------------------------
def _smoke():
    """Minimal forward+backward on CPU."""
    cfg = FockPARFConfig_v2(
        vocab_size=257, d=64, max_len=64, L=4,
        v_hidden=64, v_depth=2,
        v_phi_d_type=4, v_phi_d_angle=2,
        v_phi_phi_hidden=8, v_phi_theta_hidden=8,
        v_phi_mlp_hidden=16,
        mass_mode="global",
        top_k=8,
        score_head_hidden=8,
        # Fock v2 specific
        n_registers=16,
        d_k=32,
        destruction_gate_hidden=32,
        stack_discipline=True,
        reverse_channel=True,
    )
    torch.manual_seed(0)
    net = FockPARFLM_v2(cfg)
    total_params = sum(p.numel() for p in net.parameters())
    fock_overhead = net.get_fock_v2_overhead()
    print(f"[fock-v2-smoke] total params: {total_params:,}")
    print(f"[fock-v2-smoke] fock v2 overhead: {fock_overhead:,} "
          f"({100*fock_overhead/total_params:.1f}%)")
    print(f"[fock-v2-smoke] base params: {total_params - fock_overhead:,}")
    print(f"[fock-v2-smoke] M={cfg.n_registers}, d_k={cfg.d_k}, "
          f"reverse_channel={cfg.reverse_channel}")

    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    net.train()
    logits, loss = net(x, targets=y)
    print(f"[fock-v2-smoke] forward: logits {tuple(logits.shape)} "
          f"loss {loss.item():.4f}")
    loss.backward()
    print("[fock-v2-smoke] backward OK; no exceptions.")

    # Verify eval-mode forward.
    net.eval()
    with torch.enable_grad():
        logits_eval, _ = net(x)
    print(f"[fock-v2-smoke] eval forward OK: {tuple(logits_eval.shape)}")


def _budget():
    """Print parameter budget at P10f scale (d=256, L=8, M=16)."""
    cfg = FockPARFConfig_v2(
        vocab_size=50257, d=256, max_len=256, L=8,
        v_hidden=128, v_depth=3,
        v_phi_d_type=16, v_phi_d_angle=8,
        v_phi_phi_hidden=32, v_phi_theta_hidden=32,
        v_phi_mlp_hidden=64,
        mass_mode="global",
        top_k=16,
        score_head_hidden=32,
        # Fock v2 specific at P10f scale
        n_registers=16,
        d_k=64,
        destruction_gate_hidden=64,
        stack_discipline=True,
        reverse_channel=True,
    )
    torch.manual_seed(0)
    net = FockPARFLM_v2(cfg)
    total = sum(p.numel() for p in net.parameters())
    overhead = net.get_fock_v2_overhead()
    base = total - overhead

    print("=" * 60)
    print("FockPARFLM v2 Parameter Budget (P10f scale)")
    print("=" * 60)
    print(f"  d={cfg.d}, L={cfg.L}, M={cfg.n_registers}, d_k={cfg.d_k}")
    print(f"  reverse_channel={cfg.reverse_channel}")
    print(f"  stack_discipline={cfg.stack_discipline}")
    print("-" * 60)
    print(f"  Base PARFLM params:          {base:>12,}")
    print(f"  Fock v2 overhead:            {overhead:>12,}")

    reg_params = cfg.n_registers * cfg.d
    print(f"    register_embed ({cfg.n_registers}x{cfg.d}): "
          f"{reg_params:>8,}")

    cg_params = sum(p.numel() for p in net.creation_gate.parameters())
    print(f"    creation_gate (Q/K/V):      {cg_params:>8,}")

    dg_params = sum(
        sum(p.numel() for p in g.parameters())
        for g in net.destruction_gates
    )
    print(f"    destruction_gates ({cfg.L} layers): {dg_params:>8,}")

    if net.reverse_ch is not None:
        rc_params = sum(p.numel() for p in net.reverse_ch.parameters())
        print(f"    reverse_channel:            {rc_params:>8,}")

    print("-" * 60)
    print(f"  TOTAL:                       {total:>12,}")
    print(f"  Overhead fraction:           {100*overhead/total:>11.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "--budget":
        _budget()
    else:
        _smoke()
