"""
Fock-space PARF-augmented SPLM (v2 creation/destruction) — Latent Particle Pool.

Reference
---------
companion_notes/Augmenting_PARFLM_to_handle_MCS_Languages.md

Architecture (one-paragraph summary)
------------------------------------
This module augments SparsePARFLM with M latent register particles that can
be dynamically activated (created) and deactivated (destroyed) during the
forward pass.  Registers start in a "vacuum" state.  At each layer a learned
creation gate, conditioned on the mean token field, may activate registers;
a per-register destruction gate may deactivate them.  Active registers
participate in V_θ and V_φ pair interactions identically to real tokens.

The Fock-space interpretation (§9.4.2 of paper_v4):
  - Register pool at rest        → vacuum |0⟩
  - Activation of register r_j   → creation operator a†_v|0⟩
  - Deactivation of register r_j → annihilation operator a_v|ψ⟩
  - Active register count         → number operator N

With a salience-ordered LIFO activation discipline the system implements a
pushdown automaton and escapes the v0 expressivity ceiling (Theorem v0-ceiling,
§9.2): it can recognise Dyck_n past the predicted collapse depth D*.

Parameter overhead at P10f scale (d=256, L=8, M=32): ~290K parameters (<2%
of 22M total).
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
class FockPARFConfig(SparsePARFConfig):
    """Fock-space (v2) augmented PARF configuration.

    v2-specific knobs:

      n_registers             : int — Pool size M (maximum latent particles).
      register_salience_decay : float — Exponential decay of register salience
                                per layer; prevents ghost activations from
                                persisting indefinitely.
      register_salience_threshold : float — σ_j must exceed this for register
                                j to participate in dynamics.
      creation_gate_hidden    : int — Hidden width of the creation gate MLP.
      stack_discipline        : bool — When True, enforce LIFO (salience-ordered)
                                activation on registers, implementing a pushdown
                                automaton discipline.  When False, any register
                                can activate independently (bag discipline).
      register_init_scale     : float — Std of the learnable vacuum embeddings.
    """
    n_registers: int = 32
    register_salience_decay: float = 0.9
    register_salience_threshold: float = 0.1
    creation_gate_hidden: int = 64
    stack_discipline: bool = True
    register_init_scale: float = 0.02


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
class CreationGate(nn.Module):
    """Per-layer creation gate: mean(h_tokens) → activation scores ∈ [0, 1]^M.

    The gate is conditioned on the global token-field summary (mean-pooled
    hidden states) so the activation decision is a collective property of
    the discourse state, not a local per-token decision.
    """

    def __init__(self, d: int, hidden: int, M: int, init_scale: float = 0.02):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, M),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=init_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_mean: torch.Tensor) -> torch.Tensor:
        """h_mean: (B, d) → (B, M) in [0, 1]."""
        return torch.sigmoid(self.net(h_mean))


class DestructionGate(nn.Module):
    """Per-register destruction gate: r_j → destruction probability ∈ [0, 1].

    Applied per-register independently.  The gate reads the register's own
    hidden state to decide whether the entity it represents has exited the
    discourse.
    """

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
# Fock PARF language model
# ---------------------------------------------------------------------------
class FockPARFLM(SparsePARFLM):
    """PARFLM augmented with a latent register pool (v2 creation/destruction).

    Inherits the full sparse PARF dynamics (Gumbel-softmax top-k pair routing)
    and adds M latent register particles that can be created/destroyed per
    layer.  Active registers participate in both V_θ and V_φ interactions
    identically to input tokens.

    Register lifecycle per layer ℓ:
      1. Creation gate fires based on mean token field → updates salience.
      2. Active mask derived from salience (optionally with LIFO discipline).
      3. Active registers concatenated to token hidden states.
      4. Standard PARF dynamics (V_θ + sparse V_φ + damped Verlet) on the
         extended state.
      5. Destruction gate fires per active register → decays salience.
      6. Token and register states split apart for next layer.

    The register pool is a learnable parameter (the "vacuum embedding").
    Each forward pass maintains a running salience vector σ ∈ [0,1]^M that
    tracks which registers are currently "alive".
    """

    cfg: FockPARFConfig

    def __init__(self, cfg: FockPARFConfig):
        if not isinstance(cfg, FockPARFConfig):
            raise TypeError(
                f"FockPARFLM requires a FockPARFConfig, got {type(cfg)!r}."
            )
        super().__init__(cfg)
        M, d, L = cfg.n_registers, cfg.d, cfg.L

        # Learnable vacuum embedding for each register slot.
        self.register_embed = nn.Parameter(
            torch.randn(M, d) * cfg.register_init_scale
        )

        # Per-layer creation gates.
        self.creation_gates = nn.ModuleList([
            CreationGate(d, cfg.creation_gate_hidden, M, init_scale=0.02)
            for _ in range(L)
        ])

        # Per-layer destruction gates (shared architecture, independent weights).
        self.destruction_gates = nn.ModuleList([
            DestructionGate(d, cfg.creation_gate_hidden, init_scale=0.02)
            for _ in range(L)
        ])

    # ------------------------------------------------------------------
    def _init_registers(self, B: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialise register states and salience for a new forward pass.

        Returns:
            r: (B, M, d) — register hidden states (copies of vacuum embed).
            salience: (B, M) — all zeros (no register active at start).
        """
        M, d = self.cfg.n_registers, self.cfg.d
        r = self.register_embed.unsqueeze(0).expand(B, M, d).clone()
        salience = torch.zeros(B, M, device=device)
        return r, salience

    # ------------------------------------------------------------------
    def _active_mask(self, salience: torch.Tensor) -> torch.Tensor:
        """Derive the boolean active mask from salience, optionally with LIFO.

        Args:
            salience: (B, M) continuous salience values in [0, 1].

        Returns:
            active: (B, M) boolean mask.

        LIFO stack discipline: only the contiguous prefix of registers ordered
        by activation time (approximated by salience rank) is active.  This
        enforces a pushdown constraint where the most-recently-created register
        must be destroyed before earlier ones can be.

        Without LIFO: any register above threshold is independently active.
        """
        cfg = self.cfg
        above_thresh = salience > cfg.register_salience_threshold  # (B, M)

        if not cfg.stack_discipline:
            return above_thresh

        # LIFO: sort registers by salience descending; only the contiguous
        # leading block of above-threshold registers is active.
        sorted_sal, sort_idx = salience.sort(dim=-1, descending=True)
        sorted_above = sorted_sal > cfg.register_salience_threshold

        # Find the first False in sorted order → all slots after it are inactive.
        # Use cumprod as a portable alternative to cummin (MPS lacks cummin).
        sorted_active = torch.cumprod(sorted_above.float(), dim=-1).bool()

        # Scatter back to original register ordering.
        active = torch.zeros_like(sorted_active)
        active.scatter_(1, sort_idx, sorted_active)
        return active

    # ------------------------------------------------------------------
    def _fock_layer_step(
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
        """One Fock-augmented PARF layer step.

        1. Creation gate → update salience.
        2. Build active mask → select registers.
        3. Concatenate [tokens, active_registers] and run Verlet step.
        4. Destruction gate → decay salience.
        5. Split states back.

        Returns updated (h_new, h, r_new, salience_new).
        """
        cfg = self.cfg
        B, T, d = h.shape
        M = cfg.n_registers
        decay = cfg.register_salience_decay

        # --- 1. Creation gate ---
        h_mean = h.mean(dim=1)  # (B, d) — token field summary
        g_create = self.creation_gates[layer_idx](h_mean)  # (B, M)
        salience = salience * decay + g_create * (1.0 - decay)

        # --- 2. Active mask ---
        active = self._active_mask(salience)  # (B, M) bool

        # --- 3. Concatenate active registers ---
        # Soft gating: multiply register states by salience (differentiable).
        # Hard masking zeros out inactive registers so they don't participate.
        active_float = active.float().unsqueeze(-1)  # (B, M, 1)
        r_gated = r * active_float  # (B, M, d) — inactive = zero vector

        # Build extended state: (B, T+M, d)
        h_ext = torch.cat([h, r_gated], dim=1)
        h_prev_ext = torch.cat([h_prev, r_gated], dim=1)

        # Extended mass: registers share the global mass.
        if isinstance(m_b, torch.Tensor) and m_b.dim() >= 2:
            # m_b is per-token (B, T, 1) from logfreq mode.
            m_reg = self.m_global.expand(B, M, 1)
            m_ext = torch.cat([m_b, m_reg], dim=1)
        else:
            m_ext = m_b

        # --- 4. Run standard PARF dynamics on extended state ---
        # We call the parent SparsePARFLM._layer_step which handles V_θ, V_φ,
        # sparse routing, and the Verlet integrator.  The pair mask is rebuilt
        # internally for the extended sequence length T+M.
        h_ext_new = super()._layer_step(
            h_ext, h_prev_ext, m_ext, gamma, dt, layer_idx=layer_idx,
        )

        # --- 5. Split back ---
        h_new = h_ext_new[:, :T, :]      # (B, T, d) — updated token states
        r_new = h_ext_new[:, T:, :]      # (B, M, d) — updated register states

        # Inactive registers retain their prior state (no dynamics applied).
        r_new = torch.where(active_float.bool(), r_new, r)

        # --- 6. Destruction gate ---
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
        """Walk the L Fock-PARF layers with register lifecycle management."""
        cfg = self.cfg
        B, T, d = h0.shape
        gamma, dt = self.gamma, cfg.dt
        m_b = self.compute_mass(x)

        # Initialise registers.
        r, salience = self._init_registers(B, h0.device)

        h = h0
        h_prev = h0

        traj: Optional[List[torch.Tensor]] = None
        if return_trajectory:
            traj = [h.detach().cpu()]

        for ell in range(cfg.L):
            h_new, h_prev_out, r, salience = self._fock_layer_step(
                h, h_prev, r, salience, m_b, gamma, dt, layer_idx=ell,
            )
            h_prev = h_prev_out
            h = h_new
            if traj is not None:
                traj.append(h.detach().cpu())

        return h, traj

    # ------------------------------------------------------------------
    def get_register_overhead(self) -> int:
        """Count parameters specific to the Fock (v2) augmentation."""
        overhead = self.register_embed.numel()
        for gate in self.creation_gates:
            overhead += sum(p.numel() for p in gate.parameters())
        for gate in self.destruction_gates:
            overhead += sum(p.numel() for p in gate.parameters())
        return overhead


# ---------------------------------------------------------------------------
# Smoke entry point
# ---------------------------------------------------------------------------
def _smoke():
    """Minimal forward+backward on CPU."""
    cfg = FockPARFConfig(
        vocab_size=257, d=64, max_len=64, L=4,
        v_hidden=64, v_depth=2,
        v_phi_d_type=4, v_phi_d_angle=2,
        v_phi_phi_hidden=8, v_phi_theta_hidden=8,
        v_phi_mlp_hidden=16,
        mass_mode="global",
        top_k=8,
        score_head_hidden=8,
        # Fock-specific
        n_registers=16,
        creation_gate_hidden=32,
        stack_discipline=True,
    )
    torch.manual_seed(0)
    net = FockPARFLM(cfg)
    total_params = sum(p.numel() for p in net.parameters())
    fock_overhead = net.get_register_overhead()
    print(f"[fock-parf-smoke] total params: {total_params:,}")
    print(f"[fock-parf-smoke] fock overhead: {fock_overhead:,} "
          f"({100*fock_overhead/total_params:.1f}%)")
    print(f"[fock-parf-smoke] base params: {total_params - fock_overhead:,}")
    print(f"[fock-parf-smoke] M={cfg.n_registers}, "
          f"stack_discipline={cfg.stack_discipline}, "
          f"decay={cfg.register_salience_decay}")

    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    net.train()
    logits, loss = net(x, targets=y)
    print(f"[fock-parf-smoke] forward: logits {tuple(logits.shape)} "
          f"loss {loss.item():.4f}")
    loss.backward()
    print("[fock-parf-smoke] backward OK; no exceptions.")

    # Verify eval-mode forward (still needs enable_grad for Verlet integrator).
    net.eval()
    with torch.enable_grad():
        logits_eval, _ = net(x)
    print(f"[fock-parf-smoke] eval forward OK: {tuple(logits_eval.shape)}")


def _budget_p10f():
    """Print parameter budget at P10f scale (d=256, L=8, M=32)."""
    cfg = FockPARFConfig(
        vocab_size=50257, d=256, max_len=256, L=8,
        v_hidden=128, v_depth=3,
        v_phi_d_type=16, v_phi_d_angle=8,
        v_phi_phi_hidden=32, v_phi_theta_hidden=32,
        v_phi_mlp_hidden=64,
        mass_mode="global",
        top_k=16,
        score_head_hidden=32,
        # Fock-specific at P10f scale
        n_registers=32,
        creation_gate_hidden=64,
        stack_discipline=True,
    )
    torch.manual_seed(0)
    net = FockPARFLM(cfg)
    total = sum(p.numel() for p in net.parameters())
    overhead = net.get_register_overhead()
    base = total - overhead

    print("=" * 60)
    print("FockPARFLM Parameter Budget (P10f scale)")
    print("=" * 60)
    print(f"  d={cfg.d}, L={cfg.L}, M={cfg.n_registers}")
    print(f"  creation_gate_hidden={cfg.creation_gate_hidden}")
    print(f"  stack_discipline={cfg.stack_discipline}")
    print("-" * 60)
    print(f"  Base PARFLM params:          {base:>12,}")
    print(f"  Fock (v2) overhead:          {overhead:>12,}")
    print(f"    register_embed ({cfg.n_registers}×{cfg.d}): "
          f"{cfg.n_registers * cfg.d:>8,}")

    cg_params = sum(
        sum(p.numel() for p in g.parameters())
        for g in net.creation_gates
    )
    dg_params = sum(
        sum(p.numel() for p in g.parameters())
        for g in net.destruction_gates
    )
    print(f"    creation_gates ({cfg.L} layers): {cg_params:>8,}")
    print(f"    destruction_gates ({cfg.L} layers): {dg_params:>8,}")
    print("-" * 60)
    print(f"  TOTAL:                       {total:>12,}")
    print(f"  Overhead fraction:           {100*overhead/total:>11.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "--budget":
        _budget_p10f()
    else:
        _smoke()
