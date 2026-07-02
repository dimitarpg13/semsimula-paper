"""
Fock-space Multi-Xi PARF-augmented SPLM — K-EMA × sparse PARF × latent registers.

Combines three extensions of the base SPLM:

  1. **Multi-channel K-EMA ξ** (from model_parf_multixi.py):
     K learnable causal EMAs at multiple decay scales, widening
     V_θ input from 2d to (K+1)d.

  2. **Sparse PARF pair-interactions** (from model_parf_sparse.py):
     Gumbel-softmax top-k pair routing with V_φ(h_t, h_s).

  3. **Fock-space latent register pool** (from model_fock_parf.py / v2):
     M latent register particles with creation/destruction gates
     that escape the v0 expressivity ceiling (Dyck_n recognition).

Memory optimisations:
  - Level-2 per-layer gradient checkpointing (use_reentrant=False)
  - Stage-1.5b gathered V_φ (O(T·k) instead of O(T²))

Supports both Fock gate variants via ``fock_version``:
  - ``'v1'``: mean-conditioned creation gate (FockPARFLM-style)
  - ``'v2'``: Q/K/V-structured creation + optional reverse channel

Inheritance
-----------
    FockMultiXiPARFLM  →  MultiXiPARFLM  →  SparsePARFLM  →  PARFLM
"""

from __future__ import annotations

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
from model_fock_parf import (  # noqa: E402
    CreationGate,
    DestructionGate,
)
from model_fock_parf_v2 import (  # noqa: E402
    QKVCreationGate,
    QKVCreationGate_v21,
    ReverseChannel,
    DestructionGate_v2,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class FockMultiXiPARFConfig(MultiXiPARFConfig):
    """Multi-Xi PARF config extended with Fock register-pool parameters.

    Inherits all multi-xi fields (xi_channels, xi_alpha_inits, etc.)
    and all sparse PARF fields (top_k, use_gathered_v_phi, etc.).

    Fock-specific knobs:

      fock_version          : str — 'v1' (mean-conditioned) or 'v2' (Q/K/V).
      n_registers           : int — Pool size M.
      register_salience_decay : float — Per-layer exponential decay of salience.
      register_salience_threshold : float — σ_j must exceed this for register
                              j to participate in dynamics.
      creation_gate_hidden  : int — Hidden width of the v1 creation gate MLP.
      stack_discipline      : bool — LIFO (salience-ordered) activation.
      register_init_scale   : float — Std of the learnable vacuum embeddings.

    v2-only knobs:

      d_k                   : int — Key/query projection dimension.
      tau_create_init       : float|None — Learnable creation temperature.
      destruction_gate_hidden : int — Hidden width of the v2 destruction MLP.
      reverse_channel       : bool — Non-conservative force Q_i on tokens.
    """
    fock_version: str = "v1"
    n_registers: int = 16
    register_salience_decay: float = 0.9
    register_salience_threshold: float = 0.1
    creation_gate_hidden: int = 64
    stack_discipline: bool = True
    register_init_scale: float = 0.02
    # v2-only
    d_k: int = 64
    tau_create_init: Optional[float] = 0.1
    destruction_gate_hidden: int = 64
    reverse_channel: bool = True
    # v2.1 creation gate improvements (§6 of Fock-PARFLM_Next_Steps.md)
    per_register_tau: bool = False      # B1: per-register learnable temperature
    per_register_keys: bool = False     # B2: per-register key subspaces
    ortho_register_init: bool = False   # B3: orthogonal register embed init
    # Reverse-channel stabilisation (§10.12 of
    # Improving_the_Fock_Mechanism_to_match_Attention.md; experiment E5c)
    reverse_channel_stable: bool = False   # QK-norm + output RMS-norm readout
    reverse_channel_pre_ln: bool = True    # pre-LayerNorm on q/k/v (stable only)
    reverse_channel_soft_norm: bool = False  # soft-floored output norm (eps=1.0)
                                           # instead of hard unit-RMS (eps=1e-6);
                                           # removes the 1/‖Q‖ Jacobian blow-up
    reverse_channel_warmup_steps: int = 0  # linear gate warmup over this many
                                           # training forward passes; 0 = off


# ---------------------------------------------------------------------------
# Fock Multi-Xi PARFLM
# ---------------------------------------------------------------------------
class FockMultiXiPARFLM(MultiXiPARFLM):
    """Multi-channel ξ PARFLM augmented with a Fock-space register pool.

    Inherits the full multi-xi sparse PARF dynamics (K-EMA ξ + widened V_θ +
    Gumbel-softmax top-k V_φ) and adds M latent register particles with
    creation/destruction gates.

    Register lifecycle per layer ℓ (identical to FockPARFLM/v2 but running
    on the multi-xi _layer_step):
      1. Creation gate fires → updates salience.
      2. Active mask derived from salience (optionally LIFO).
      3. Active registers concatenated to token hidden states.
      4. Multi-xi PARF dynamics on the extended (T+M) state.
      5. (v2 only, optional) Reverse channel force Q_i on tokens.
      6. Destruction gate → decay salience.
      7. Split token and register states for next layer.
    """

    cfg: FockMultiXiPARFConfig

    def __init__(self, cfg: FockMultiXiPARFConfig):
        if not isinstance(cfg, FockMultiXiPARFConfig):
            raise TypeError(
                f"FockMultiXiPARFLM requires a FockMultiXiPARFConfig, "
                f"got {type(cfg)!r}."
            )
        super().__init__(cfg)
        self._fock_cfg = cfg
        M, d, L = cfg.n_registers, cfg.d, cfg.L

        if cfg.ortho_register_init and M <= d:
            U, _, _ = torch.linalg.svd(torch.randn(d, d))
            self.register_embed = nn.Parameter(
                U[:M] * cfg.register_init_scale
            )
        else:
            self.register_embed = nn.Parameter(
                torch.randn(M, d) * cfg.register_init_scale
            )

        use_v21_gate = cfg.per_register_tau or cfg.per_register_keys

        if cfg.fock_version == "v1":
            self.creation_gates = nn.ModuleList([
                CreationGate(d, cfg.creation_gate_hidden, M, init_scale=0.02)
                for _ in range(L)
            ])
            self.destruction_gates = nn.ModuleList([
                DestructionGate(d, cfg.creation_gate_hidden, init_scale=0.02)
                for _ in range(L)
            ])
            self.creation_gate_qkv = None
            self.reverse_ch = None
            self.reverse_channel_scale = None
        elif cfg.fock_version == "v2":
            self.creation_gates = nn.ModuleList()  # unused for v2/v2.1
            if use_v21_gate:
                self.creation_gate_qkv = QKVCreationGate_v21(
                    d, cfg.d_k, M,
                    init_scale=cfg.register_init_scale,
                    tau_create_init=cfg.tau_create_init,
                    per_register_keys=cfg.per_register_keys,
                )
            else:
                self.creation_gate_qkv = QKVCreationGate(
                    d, cfg.d_k, M,
                    init_scale=cfg.register_init_scale,
                    tau_create_init=cfg.tau_create_init,
                )
            self.destruction_gates = nn.ModuleList([
                DestructionGate_v2(
                    d, cfg.destruction_gate_hidden,
                    init_scale=cfg.register_init_scale,
                )
                for _ in range(L)
            ])
            if cfg.reverse_channel:
                self.reverse_ch = ReverseChannel(
                    d, cfg.d_k, init_scale=cfg.register_init_scale,
                    stable=cfg.reverse_channel_stable,
                    pre_ln=cfg.reverse_channel_pre_ln,
                    soft_norm=cfg.reverse_channel_soft_norm,
                )
                self.reverse_channel_scale = nn.Parameter(torch.zeros(1))
            else:
                self.reverse_ch = None
                self.reverse_channel_scale = None
        else:
            raise ValueError(
                f"fock_version must be 'v1' or 'v2', got {cfg.fock_version!r}"
            )

        # Warmup counter for the reverse-channel gate (incremented once per
        # training forward pass in _stack_forward).  Persisted so resumed
        # runs continue the schedule.
        self.register_buffer(
            "reverse_warmup_step",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )

        # Diagnostic capture buffer.  When not None, _fock_layer_step appends
        # one dict of per-layer health scalars per layer per forward pass.
        # Left None (and sub-module capture off) during normal training so it
        # is exactly zero cost; a probe flips it on for a single eval forward.
        self._fock_capture: Optional[List[dict]] = None

    # ------------------------------------------------------------------
    def _init_registers(
        self, B: int, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        M, d = self.cfg.n_registers, self.cfg.d
        r = self.register_embed.unsqueeze(0).expand(B, M, d).clone()
        if self.cfg.fock_version == "v1":
            salience = torch.zeros(B, M, device=device)
        else:
            salience = torch.ones(B, M, device=device)
        return r, salience

    # ------------------------------------------------------------------
    def _active_mask(self, salience: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        above_thresh = salience > cfg.register_salience_threshold

        if not cfg.stack_discipline:
            return above_thresh

        sorted_sal, sort_idx = salience.sort(dim=-1, descending=True)
        sorted_above = sorted_sal > cfg.register_salience_threshold
        sorted_active = torch.cumprod(sorted_above.float(), dim=-1).bool()

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
        """One Fock-augmented multi-xi PARF layer step.

        Dispatches to v1 or v2 creation protocol based on cfg.fock_version.
        The inner dynamics call super()._layer_step which is the multi-xi
        version (K-EMA ξ + widened V_θ + gathered V_φ).
        """
        cfg = self.cfg
        B, T, d = h.shape
        M = cfg.n_registers
        decay = cfg.register_salience_decay

        # --- Creation ---
        alpha_max = None
        if cfg.fock_version == "v1":
            h_mean = h.mean(dim=1)
            g_create = self.creation_gates[layer_idx](h_mean)
            salience = salience * decay + g_create * (1.0 - decay)
            r_causal = None
        else:
            r_new_content, r_causal, alpha_max = self.creation_gate_qkv(h, r)
            blend = salience.unsqueeze(-1)
            r = blend * r + (1.0 - blend) * r_new_content
            salience = salience * decay + alpha_max * (1.0 - decay)

        # --- Active mask ---
        active = self._active_mask(salience)
        active_float = active.float().unsqueeze(-1)
        r_gated = r * active_float

        # --- Extend state ---
        h_ext = torch.cat([h, r_gated], dim=1)
        h_prev_ext = torch.cat([h_prev, r_gated], dim=1)

        if isinstance(m_b, torch.Tensor) and m_b.dim() >= 2:
            m_reg = self.m_global.expand(B, M, 1)
            m_ext = torch.cat([m_b, m_reg], dim=1)
        else:
            m_ext = m_b

        # --- Multi-xi PARF dynamics on extended state ---
        h_ext_new = super()._layer_step(
            h_ext, h_prev_ext, m_ext, gamma, dt, layer_idx=layer_idx,
        )

        # --- Split back ---
        h_new = h_ext_new[:, :T, :]
        r_new = h_ext_new[:, T:, :]
        r_new = torch.where(active_float.bool(), r_new, r)

        # --- Reverse channel (v2 only) ---
        _cap_qforce_ratio = 0.0
        _cap_rev_scale = 0.0
        if (
            cfg.fock_version == "v2"
            and self.reverse_ch is not None
            and active.any()
        ):
            # Use position-dependent causal register content so that
            # the force on token t only reflects tokens 1…t (no leak).
            r_rev = r_causal if r_causal is not None else r_new
            Q_force = self.reverse_ch(h_new, r_rev, active)
            scale = torch.tanh(self.reverse_channel_scale)
            if cfg.reverse_channel_warmup_steps > 0:
                # Linear gate warmup: open the non-conservative force only as
                # the conservative backbone settles, avoiding the early-training
                # regime where it fires at random and fights V_theta/V_phi.
                warm = (
                    self.reverse_warmup_step.float()
                    / float(cfg.reverse_channel_warmup_steps)
                ).clamp(max=1.0)
                scale = scale * warm
            increment = (dt * dt / m_b) * scale * Q_force
            if self._fock_capture is not None:
                with torch.no_grad():
                    h_rms = h_new.pow(2).mean().sqrt().clamp(min=1e-8)
                    _cap_qforce_ratio = float(
                        increment.pow(2).mean().sqrt() / h_rms
                    )
                    _cap_rev_scale = float(scale.reshape(-1)[0])
            h_new = h_new + increment
            if cfg.ln_after_step:
                h_new = self._project(h_new)

        # --- Destruction ---
        g_destroy = self.destruction_gates[layer_idx](r_new)
        salience = salience * (1.0 - g_destroy * active.float())

        if self._fock_capture is not None:
            self._fock_capture.append(
                self._fock_layer_stats(
                    layer_idx, r_new, active, salience, g_destroy,
                    alpha_max, _cap_qforce_ratio, _cap_rev_scale,
                )
            )

        return h_new, h, r_new, salience

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _fock_layer_stats(
        self,
        layer_idx: int,
        r_new: torch.Tensor,
        active: torch.Tensor,
        salience: torch.Tensor,
        g_destroy: torch.Tensor,
        alpha_max: Optional[torch.Tensor],
        qforce_ratio: float,
        rev_scale: float,
    ) -> dict:
        """Snapshot per-layer Fock health scalars for the diagnostic probe.

        All quantities are cheap reductions over the batch; only called when
        ``self._fock_capture`` is not None (i.e. during a diagnostic forward).
        """
        active_f = active.float()
        active_frac = float(active_f.mean())

        # Register content diversity: mean pairwise cosine similarity among the
        # ACTIVE registers.  ~1.0 => registers have collapsed to one direction
        # (wasted capacity); ~0.0 => a diverse, well-used register bank.
        r_norm = F.normalize(r_new, dim=-1)                 # (B, M, d)
        gram = torch.bmm(r_norm, r_norm.transpose(1, 2))    # (B, M, M)
        M = gram.shape[-1]
        eye = torch.eye(M, device=gram.device).bool()
        pair_mask = active.unsqueeze(2) & active.unsqueeze(1) & ~eye  # (B,M,M)
        n_pairs = pair_mask.float().sum().clamp(min=1.0)
        reg_diversity = float((gram * pair_mask.float()).sum() / n_pairs)

        stats = {
            "layer": layer_idx,
            "active_frac": active_frac,
            "salience_mean": float(salience.mean()),
            "salience_std": float(salience.std()),
            "reg_cos_sim": reg_diversity,
            "destroy_mean": float((g_destroy * active_f).sum()
                                  / active_f.sum().clamp(min=1.0)),
            "qforce_ratio": qforce_ratio,
            "rev_scale": rev_scale,
        }
        if alpha_max is not None:
            stats["create_alpha_max"] = float(alpha_max.mean())
        if self.creation_gate_qkv is not None:
            stats["create_entropy"] = self.creation_gate_qkv.last_entropy
        if self.reverse_ch is not None:
            stats["rev_entropy"] = self.reverse_ch.last_entropy
            stats["rev_alpha_max"] = self.reverse_ch.last_alpha_max
        return stats

    # ------------------------------------------------------------------
    def _stack_forward(
        self,
        h0: torch.Tensor,
        x: torch.Tensor,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Walk L Fock multi-xi layers with register lifecycle.

        Supports Level-2 per-layer gradient checkpointing when
        cfg.use_layer_checkpoint is True.
        """
        cfg = self.cfg
        B, T, d = h0.shape
        gamma, dt = self.gamma, cfg.dt
        m_b = self.compute_mass(x)

        # Advance the reverse-channel warmup once per training forward pass
        # (kept outside the per-layer checkpoint so it counts forwards, not
        # recomputations).
        if (
            self.training
            and self.reverse_channel_scale is not None
            and cfg.reverse_channel_warmup_steps > 0
        ):
            self.reverse_warmup_step += 1

        r, salience = self._init_registers(B, h0.device)

        h = h0
        h_prev = h0

        traj: Optional[List[torch.Tensor]] = None
        if return_trajectory:
            traj = [h.detach().cpu()]

        for ell in range(cfg.L):
            if cfg.use_layer_checkpoint and self.training:
                def _ckpt_step(
                    _h, _h_prev, _r, _sal, _m_b, _gamma,
                    _dt=dt, _ell=ell,
                ):
                    h_n, h_p, r_n, s_n = self._fock_layer_step(
                        _h, _h_prev, _r, _sal, _m_b, _gamma, _dt, _ell,
                    )
                    return h_n, h_p, r_n, s_n

                h_new, h_prev_out, r, salience = (
                    torch.utils.checkpoint.checkpoint(
                        _ckpt_step,
                        h, h_prev, r, salience, m_b, gamma,
                        use_reentrant=False,
                    )
                )
            else:
                h_new, h_prev_out, r, salience = self._fock_layer_step(
                    h, h_prev, r, salience, m_b, gamma, dt, layer_idx=ell,
                )

            h_prev = h_prev_out
            h = h_new
            if traj is not None:
                traj.append(h.detach().cpu())

        return h, traj

    # ------------------------------------------------------------------
    @torch.no_grad()
    def fock_diagnostics(self) -> dict:
        """Return a dict of Fock-specific diagnostic scalars for logging."""
        diag: dict = {}
        cfg = self._fock_cfg
        if cfg.fock_version == "v2" and self.creation_gate_qkv is not None:
            log_tau = self.creation_gate_qkv.log_tau
            if log_tau is not None:
                tau_vals = log_tau.exp().clamp(min=1e-4)
                if tau_vals.dim() == 0:
                    diag["fock_tau_create"] = tau_vals.item()
                else:
                    diag["fock_tau_create_mean"] = tau_vals.mean().item()
                    diag["fock_tau_create_min"] = tau_vals.min().item()
                    diag["fock_tau_create_max"] = tau_vals.max().item()
        if cfg.fock_version == "v2" and self.reverse_channel_scale is not None:
            diag["fock_rev_scale"] = torch.tanh(
                self.reverse_channel_scale
            ).item()
            if cfg.reverse_channel_warmup_steps > 0:
                diag["fock_rev_warmup"] = min(
                    1.0,
                    float(self.reverse_warmup_step)
                    / float(cfg.reverse_channel_warmup_steps),
                )
            rc = self.reverse_ch
            if rc is not None and getattr(rc, "logit_scale", None) is not None:
                diag["fock_rev_logit_scale"] = (
                    rc.logit_scale.exp().clamp(max=rc.logit_scale_max).item()
                )
        return diag

    # ------------------------------------------------------------------
    def set_fock_capture(self, enabled: bool) -> None:
        """Toggle per-layer Fock health capture on the model and sub-modules.

        When enabled, the next forward pass records one stats dict per layer
        into ``self._fock_capture`` (cleared here on each enable).  Keep it on
        for a single eval forward, then read ``fock_component_report()`` and
        turn it off — it adds a small amount of work and memory per layer.
        """
        self._fock_capture = [] if enabled else None
        for mod in (self.creation_gate_qkv, self.reverse_ch):
            if mod is not None:
                mod.capture_stats = enabled

    # ------------------------------------------------------------------
    def fock_component_report(self, reset: bool = True) -> dict:
        """Aggregate captured per-layer stats into a component-health report.

        Returns a dict with:
          - ``per_layer``: list of the raw per-layer stat dicts.
          - ``summary``:   mean of each scalar across layers, plus derived
                           health flags that point at the weakest link.

        Run a diagnostic forward with ``set_fock_capture(True)`` first.
        """
        cap = self._fock_capture or []
        report: dict = {"per_layer": list(cap), "summary": {}}
        if not cap:
            if reset:
                self.set_fock_capture(False)
            return report

        keys = [
            "active_frac", "salience_mean", "salience_std", "reg_cos_sim",
            "destroy_mean", "qforce_ratio", "rev_scale", "create_alpha_max",
            "create_entropy", "rev_entropy", "rev_alpha_max",
        ]
        summ = {}
        for k in keys:
            vals = [d[k] for d in cap if d.get(k) is not None]
            if vals:
                summ[k] = sum(vals) / len(vals)
        report["summary"] = summ

        # Derived, decision-oriented health flags (heuristic thresholds).
        import math as _m
        flags = []
        M = self.cfg.n_registers
        T_hint = getattr(self.cfg, "max_len", None)
        if summ.get("reg_cos_sim", 0.0) > 0.6:
            flags.append(
                f"register bank COLLAPSING (cos_sim={summ['reg_cos_sim']:.2f}"
                f" > 0.6): registers are redundant -> B2 per-register keys /"
                f" B3 orthogonal init / stronger repulsion may add capacity"
            )
        if summ.get("active_frac", 1.0) < 1.5 / max(M, 1):
            flags.append(
                f"register UTILISATION low (active_frac={summ['active_frac']:.2f}"
                f", ~{summ['active_frac']*M:.1f}/{M} active): most of the pool"
                f" is idle -> raise salience_threshold headroom / rebalance"
                f" creation vs destruction, or M is oversized"
            )
        if summ.get("create_entropy") is not None and T_hint:
            frac = summ["create_entropy"] / _m.log(max(T_hint, 2))
            if frac > 0.85:
                flags.append(
                    f"creation attention near-UNIFORM (entropy"
                    f"={summ['create_entropy']:.2f}, {frac*100:.0f}% of max):"
                    f" registers aren't selecting tokens -> lower tau_create /"
                    f" per-register tau (B1) may sharpen routing"
                )
            elif frac < 0.15:
                flags.append(
                    f"creation attention near-DEGENERATE (entropy"
                    f"={summ['create_entropy']:.2f}, {frac*100:.0f}% of max):"
                    f" each register locks to ~1 token -> temperature may have"
                    f" collapsed; raise tau_create floor"
                )
        if summ.get("rev_scale") is not None and abs(summ["rev_scale"]) < 1e-3:
            flags.append(
                f"reverse channel effectively OFF (|scale|"
                f"={abs(summ['rev_scale']):.1e}): the non-conservative force is"
                f" not being used -> either it has no headroom here, or warmup"
                f" is still ramping / gate init needs help"
            )
        if summ.get("qforce_ratio") is not None and summ["qforce_ratio"] > 0.5:
            flags.append(
                f"reverse force DOMINATES token update (qforce_ratio"
                f"={summ['qforce_ratio']:.2f}): the exchange force is large vs"
                f" the conservative step -> watch for instability / tighten gate"
            )
        if summ.get("destroy_mean") is not None and summ["destroy_mean"] > 0.5:
            flags.append(
                f"destruction AGGRESSIVE (destroy_mean="
                f"{summ['destroy_mean']:.2f}): registers are annihilated fast,"
                f" shortening cross-layer memory -> lower destruction or raise"
                f" salience decay lambda for longer-lived memory"
            )
        report["flags"] = flags

        if reset:
            self.set_fock_capture(False)
        return report

    # ------------------------------------------------------------------
    def get_register_overhead(self) -> int:
        """Count parameters specific to the Fock augmentation."""
        overhead = self.register_embed.numel()
        for gate in self.creation_gates:
            overhead += sum(p.numel() for p in gate.parameters())
        for gate in self.destruction_gates:
            overhead += sum(p.numel() for p in gate.parameters())
        if self.creation_gate_qkv is not None:
            overhead += sum(
                p.numel() for p in self.creation_gate_qkv.parameters()
            )
        if self.reverse_ch is not None:
            overhead += sum(p.numel() for p in self.reverse_ch.parameters())
        if self.reverse_channel_scale is not None:
            overhead += self.reverse_channel_scale.numel()
        return overhead


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke():
    """Minimal forward+backward for v1 and v2 across checkpoint/gathered modes."""
    for version in ("v1", "v2"):
        for layer_ckpt in (False, True):
            for gathered in (False, True):
                tag_parts = [version]
                if layer_ckpt:
                    tag_parts.append("lc")
                if gathered:
                    tag_parts.append("gv")
                tag = "+".join(tag_parts) or version

                cfg = FockMultiXiPARFConfig(
                    vocab_size=257, d=16, max_len=64, L=4,
                    v_hidden=32, v_depth=2,
                    v_phi_d_type=4, v_phi_d_angle=2,
                    v_phi_phi_hidden=8, v_phi_theta_hidden=8,
                    v_phi_mlp_hidden=16,
                    mass_mode="global",
                    top_k=4,
                    score_head_hidden=8,
                    xi_channels=4,
                    xi_alpha_inits=[0.0, 0.5, 0.9, 0.99],
                    xi_learnable=True,
                    use_layer_checkpoint=layer_ckpt,
                    use_gathered_v_phi=gathered,
                    # Fock
                    fock_version=version,
                    n_registers=8,
                    creation_gate_hidden=16,
                    stack_discipline=True,
                    register_salience_decay=(
                        0.9 if version == "v1" else 0.5
                    ),
                    register_salience_threshold=(
                        0.1 if version == "v1" else 0.005
                    ),
                    # v2-only
                    d_k=16,
                    destruction_gate_hidden=16,
                    reverse_channel=(version == "v2"),
                )
                torch.manual_seed(0)
                net = FockMultiXiPARFLM(cfg)
                total = sum(p.numel() for p in net.parameters())
                fock_oh = net.get_register_overhead()
                alpha_str = ", ".join(
                    f"{a:.3f}" for a in net.xi_alpha_values()
                )
                print(
                    f"[fock-multixi-smoke/{tag}] "
                    f"params={total:,}  fock_oh={fock_oh:,} "
                    f"({100*fock_oh/total:.1f}%)  "
                    f"K={cfg.xi_channels} α=[{alpha_str}]  "
                    f"M={cfg.n_registers}"
                )

                x = torch.randint(0, cfg.vocab_size, (2, 12))
                y = torch.randint(0, cfg.vocab_size, (2, 12))

                net.train()
                logits, loss = net(x, targets=y)
                print(
                    f"[fock-multixi-smoke/{tag}] "
                    f"forward: logits {tuple(logits.shape)} "
                    f"loss {loss.item():.4f}"
                )
                loss.backward()

                alpha_grad = net.xi_module.raw_alpha.grad
                assert alpha_grad is not None, "raw_alpha got no gradient"
                print(
                    f"[fock-multixi-smoke/{tag}] "
                    f"raw_α grad norm: {alpha_grad.norm().item():.3e}  "
                    f"backward OK."
                )
                net.zero_grad()

    print("\n✓ All Fock Multi-Xi smoke tests passed.")


if __name__ == "__main__":
    _smoke()
