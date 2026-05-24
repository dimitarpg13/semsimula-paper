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

    def forward(
        self,
        h_tokens: torch.Tensor,
        register_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Q/K/V-structured creation event.

        Args:
            h_tokens: (B, T, d) — current token hidden states.
            register_states: (B, M, d) — current register states
                             (used to derive per-register queries).

        Returns:
            r_new: (B, M, d) — new register content from attention readout.
            alpha_max: (B, M) — max attention weight per register (salience signal).
        """
        B, T, d = h_tokens.shape
        M = self.M

        K = self.W_K(h_tokens)     # (B, T, d_k)
        V = self.W_V(h_tokens)     # (B, T, d)

        # Per-register query via batched matmul:
        # register_states: (B, M, d), W_Q: (M, d, d_k)
        # q_k = register_states[:, k, :] @ W_Q[k]  for each k
        # Expand register_states to (B, M, 1, d) and W_Q to (1, M, d, d_k)
        # then batched matmul gives (B, M, 1, d_k), squeeze to (B, M, d_k)
        Q = torch.einsum("bmd,mdk->bmk", register_states, self.W_Q)  # (B, M, d_k)

        # Scaled dot-product attention scores: (B, M, T)
        scores = torch.bmm(
            Q.reshape(B * M, 1, self.d_k),
            K.unsqueeze(1).expand(B, M, T, self.d_k).reshape(B * M, self.d_k, T),
        ).reshape(B, M, T)

        if self.log_tau is not None:
            tau = self.log_tau.exp().clamp(min=1e-4)
            scores = scores / tau
        else:
            scores = scores / (self.d_k ** 0.5)

        alpha = F.softmax(scores, dim=-1)  # (B, M, T), sums to 1 over j

        # Content: r_k_new = Σ_j α_kj · v_j
        # alpha: (B, M, T), V: (B, T, d) → r_new: (B, M, d)
        r_new = torch.bmm(
            alpha.reshape(B * M, 1, T),
            V.unsqueeze(1).expand(B, M, T, d).reshape(B * M, T, d),
        ).reshape(B, M, d)

        # Salience signal: max attention weight per register
        alpha_max = alpha.max(dim=-1).values  # (B, M)

        return r_new, alpha_max


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
    """

    def __init__(self, d: int, d_k: int, init_scale: float = 0.02):
        super().__init__()
        self.d_k = d_k
        self.W_Q_rev = nn.Linear(d, d_k, bias=False)
        self.W_K_rev = nn.Linear(d, d_k, bias=False)
        self.W_V_rev = nn.Linear(d, d, bias=False)

        nn.init.normal_(self.W_Q_rev.weight, std=init_scale)
        nn.init.normal_(self.W_K_rev.weight, std=init_scale)
        nn.init.normal_(self.W_V_rev.weight, std=init_scale)

    def forward(
        self,
        h_tokens: torch.Tensor,
        r_active: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the non-conservative force on each token.

        Args:
            h_tokens: (B, T, d) — token hidden states (queries).
            r_active: (B, M, d) — register states (keys/values).
            active_mask: (B, M) — boolean mask; inactive registers zeroed.

        Returns:
            Q_force: (B, T, d) — non-conservative Fock exchange force per token.
        """
        B, T, d = h_tokens.shape
        M = r_active.shape[1]

        q = self.W_Q_rev(h_tokens)   # (B, T, d_k)
        k = self.W_K_rev(r_active)   # (B, M, d_k)
        v = self.W_V_rev(r_active)   # (B, M, d)

        # Attention scores: (B, T, M)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Mask inactive registers with large negative before softmax
        mask_expanded = active_mask.unsqueeze(1).expand(B, T, M)  # (B, T, M)
        scores = scores.masked_fill(~mask_expanded, -1e9)

        # If no registers are active, skip the computation
        has_active = active_mask.any(dim=-1, keepdim=True).unsqueeze(1)  # (B, 1, 1)

        alpha = F.softmax(scores, dim=-1)  # (B, T, M)
        alpha = alpha * has_active.float()

        # Force: Q_i = Σ_k α_ik · v_k^reg
        Q_force = torch.matmul(alpha, v)  # (B, T, d)

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
        r_new_content, alpha_max = self.creation_gate(h, r)

        # Blend new content into existing register state.
        # Registers that are already active get their content refreshed;
        # inactive registers receive fresh content from the readout.
        # Use salience-weighted blending: high-salience registers keep
        # more of their existing state (temporal persistence).
        blend = salience.unsqueeze(-1)  # (B, M, 1), in [0, ~1]
        r = blend * r + (1.0 - blend) * r_new_content

        # Salience update: exponential decay + creation signal
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
            Q_force = self.reverse_ch(h_new, r_new, active)
            # Gate the reverse force by a learnable scalar initialised to 0:
            # tanh(0)=0 → force off at init; the model learns to open it.
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
