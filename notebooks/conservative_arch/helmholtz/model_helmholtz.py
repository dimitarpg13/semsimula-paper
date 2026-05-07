"""
Helmholtz SPLM + Attention language model (Q9d, the layer-type Helmholtz hybrid).

Reference
---------
companion_notes/Scalar_Potential_based_Helmholtz_Architecture.md

Architecture
------------
A depth-L stack whose layers are indexed by a schedule
sigma : {0, 1, ..., L-1} -> {S, A}.  S-blocks are autonomous,
conservative, single-shared-V_theta integration steps; A-blocks are
standard pre-LN attention + MLP residual blocks with per-layer
parameters.  The kinematic memory (h_prev) is carried across both
block types so the velocity proxy (h_ell - h_{ell-1}) remains well-
defined regardless of how many A-blocks separate two S-blocks.

S-block update (velocity-Verlet, doc Eq. of section 2.2 with dt=1):

    delta = h_ell - h_{ell-1}                       # velocity proxy
    xi    = causal_cumulative_mean(h_ell.detach())  # leak-fix invariant
    f     = -grad_h V_theta(xi, h_ell)
    h_new = h_ell + delta / (1 + dt*gamma)
                  + (dt^2 / (m * (1 + dt*gamma))) * f
    h_new = LayerNorm(h_new)                        # if ln_after_s_step

A-block update (canonical pre-LN GPT-2 block):

    h_new = h_ell + Attn_{theta_ell}(LayerNorm(h_ell))
                  + MLP_{theta_ell}(LayerNorm(...))

Single shared V_theta
---------------------
EXACTLY ONE ScalarPotential is allocated and used by every S-block in
the stack -- including non-contiguous S-blocks under interleaved or
sandwich schedules.  This is the strongest version of the SPLM
"single energy field" commitment: any contiguous run of S-blocks
passes the strict shared-V_psi separator test by construction (see
companion_notes/Scalar_Potential_based_Helmholtz_Architecture.md section 4.1).

Causal-leak fix
---------------
xi is re-derived from h_ell.detach() at every S-block, severing the
autograd path from xi back to h.  This preserves the v3 causal-honesty
invariant (companion_notes/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md).

Schedule registry
-----------------
The 7 canonical patterns of section 6 of the doc are exposed via
`make_schedule(name, L, ...)`:

    "all_s"               -> S^L
    "all_a"               -> A^L
    "sandwich_k"          -> S^k A^(L-2k) S^k
    "inverse_sandwich_k"  -> A^k S^(L-2k) A^k
    "interleaved"         -> (SA)^(L/2)   (requires L even)
    "top_a_LA"            -> S^(L-LA) A^LA
    "bottom_a_LA"         -> A^LA S^(L-LA)

`bottom_a_LA` with LA=k (and L-LA=m) reproduces the layout of the
existing two-stage HSPLM (Variant A) at matched (k, m) -- but the
dynamics differ in two ways: (i) the velocity proxy passes through
the attention stack instead of being reset to 0 at the SPLM boundary,
and (ii) xi is re-derived each S-block from the *running* h.detach()
rather than fixed at h_k.detach().
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

from matched_baseline_model import (  # noqa: E402
    Block as AttnBlock,
    MatchedConfig,
)
from sarf_mass_variant.model_sarf_mass import (  # noqa: E402
    ScalarPotential,
    causal_cumulative_mean,
    _raw_from_positive,
)


# ---------------------------------------------------------------------------
# Schedule parsing and registry
# ---------------------------------------------------------------------------
_VALID_BLOCK_TYPES = ("S", "A")


def parse_schedule(schedule: str) -> List[str]:
    """Validate and tokenise a schedule string.

    Accepts case-insensitive strings made of 'S' and 'A' characters.
    Returns a list ['S', 'A', 'S', ...] of length L = len(schedule).
    """
    if not isinstance(schedule, str) or len(schedule) == 0:
        raise ValueError(f"schedule must be a non-empty string, got {schedule!r}")
    sigma = [c.upper() for c in schedule]
    bad = [c for c in sigma if c not in _VALID_BLOCK_TYPES]
    if bad:
        raise ValueError(
            f"schedule {schedule!r} contains invalid block types {bad}; "
            f"only 'S' and 'A' are allowed."
        )
    return sigma


def make_schedule(name: str, L: int = 8, k: int = 1, LA: int = 1) -> str:
    """Construct a canonical schedule string of length L.

    Parameters
    ----------
    name : str
        One of {"all_s", "all_a", "sandwich", "inverse_sandwich",
                "interleaved", "top_a", "bottom_a"}.  The
        "sandwich"/"inverse_sandwich" patterns use `k` S-blocks at each
        end; "top_a"/"bottom_a" patterns use `LA` A-blocks at the
        designated end.
    L : int
        Total stack depth.
    k : int
        Sandwich half-width (only used by sandwich / inverse_sandwich).
    LA : int
        Number of A-blocks (only used by top_a / bottom_a).

    Returns
    -------
    str
        A length-L schedule string.
    """
    name = name.lower()
    if name == "all_s":
        return "S" * L
    if name == "all_a":
        return "A" * L
    if name == "sandwich":
        if 2 * k > L:
            raise ValueError(f"sandwich requires 2*k <= L; got k={k}, L={L}")
        return "S" * k + "A" * (L - 2 * k) + "S" * k
    if name == "inverse_sandwich":
        if 2 * k > L:
            raise ValueError(
                f"inverse_sandwich requires 2*k <= L; got k={k}, L={L}"
            )
        return "A" * k + "S" * (L - 2 * k) + "A" * k
    if name == "interleaved":
        if L % 2 != 0:
            raise ValueError(f"interleaved requires even L; got L={L}")
        return "SA" * (L // 2)
    if name == "top_a":
        if LA > L:
            raise ValueError(f"top_a requires LA <= L; got LA={LA}, L={L}")
        return "S" * (L - LA) + "A" * LA
    if name == "bottom_a":
        if LA > L:
            raise ValueError(f"bottom_a requires LA <= L; got LA={LA}, L={L}")
        return "A" * LA + "S" * (L - LA)
    raise ValueError(
        f"unknown schedule name {name!r}; valid: all_s, all_a, sandwich, "
        f"inverse_sandwich, interleaved, top_a, bottom_a."
    )


def schedule_counts(sigma: List[str]) -> Tuple[int, int]:
    """Return (n_S_blocks, n_A_blocks) for a parsed schedule."""
    nS = sum(1 for c in sigma if c == "S")
    nA = sum(1 for c in sigma if c == "A")
    return nS, nA


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class HelmholtzConfig:
    """Helmholtz hybrid configuration.

    The architectural shape is set by `schedule` (a string of 'S' and
    'A' characters).  All other defaults match the leak-free Variant A
    HSPLM cells of `notebooks/conservative_arch/hybrid/`:
      d=128, max_len=256, v_hidden=512, v_depth=3, n_head=4, mlp_mult=4,
      mass_mode='logfreq', causal_force=True, ln_after_s_step=True.
    """
    vocab_size: int = 50257
    d: int = 128
    max_len: int = 256

    # Schedule (the design space of section 6 of the doc).
    schedule: str = "AAAASSSS"   # bottom_a with LA=4, matches existing HSPLM (k=4, m=4)

    # Attention (per-A-block) parameters.
    n_head: int = 4
    mlp_mult: int = 4
    dropout: float = 0.0

    # SPLM (shared-V_theta) parameters.
    v_hidden: int = 512
    v_depth: int = 3
    dt: float = 1.0
    init_m: float = 1.0
    init_gamma: float = 0.15
    learn_mgamma: bool = True
    fixed_gamma: Optional[float] = None

    # Per-token mass (matches leak-free Variant A HSPLM cells).
    mass_mode: str = "logfreq"   # supported: 'logfreq', 'global'
    logfreq_init_alpha: float = 0.1
    logfreq_path: Optional[str] = None

    # Stability / parity controls.
    ln_after_s_step: bool = True   # LayerNorm after each S-block step
    ln_eps: float = 1e-5
    causal_force: bool = True      # leak-fix invariant; should always be True
    tie_embeddings: bool = True


def _load_npy(p) -> np.ndarray:
    return np.load(p)


def _attn_cfg_from(cfg: HelmholtzConfig, n_attn: int) -> MatchedConfig:
    return MatchedConfig(
        vocab_size=cfg.vocab_size,
        d=cfg.d,
        max_len=cfg.max_len,
        n_layer=n_attn,        # not used by AttnBlock directly
        n_head=cfg.n_head,
        mlp_mult=cfg.mlp_mult,
        dropout=cfg.dropout,
        tie_embeddings=cfg.tie_embeddings,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class HelmholtzLM(nn.Module):
    """Layer-type Helmholtz hybrid LM (Q9d).

    Forward contract
    ----------------
      forward(x, targets=None, return_trajectory=False) -> (logits, loss[, traj])

    where traj is the per-layer list of hidden states (length L+1) on
    CPU when return_trajectory=True.
    """

    def __init__(self, cfg: HelmholtzConfig):
        super().__init__()
        self.cfg = cfg

        sigma = parse_schedule(cfg.schedule)
        self.sigma: List[str] = sigma
        nS, nA = schedule_counts(sigma)
        self.L: int = len(sigma)
        self.n_s_blocks: int = nS
        self.n_a_blocks: int = nA

        # Embeddings (token + position), tied output.
        self.E = nn.Embedding(cfg.vocab_size, cfg.d)
        self.P = nn.Parameter(torch.zeros(cfg.max_len, cfg.d))
        nn.init.normal_(self.E.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.P, mean=0.0, std=0.02)

        # ----- A-blocks -----
        # Build n_A independent attention blocks; the i-th 'A' in sigma
        # uses self.attn_blocks[i].
        attn_cfg = _attn_cfg_from(cfg, nA)
        self.attn_blocks = nn.ModuleList(
            [AttnBlock(attn_cfg) for _ in range(nA)]
        )
        self.attn_blocks.apply(self._gpt2_init)

        # Build the per-layer (block_type, sub_index) routing list once.
        # sub_index for an 'A' is the index into self.attn_blocks.
        # sub_index for an 'S' is unused (single shared V_theta).
        self._layer_routing: List[Tuple[str, int]] = []
        a_cursor = 0
        for c in sigma:
            if c == "A":
                self._layer_routing.append(("A", a_cursor))
                a_cursor += 1
            else:
                self._layer_routing.append(("S", -1))
        assert a_cursor == nA

        # ----- S-block (single shared V_theta) -----
        # Allocated even when nS == 0 so checkpoint state-dict shape is
        # stable across schedules; the param count for nS == 0 schedules
        # is therefore inflated by O(v_hidden^2) -- callers who want a
        # bit-exact param-matched all-attn baseline should use
        # matched_baseline_model.MatchedGPT directly.
        self.V_theta = ScalarPotential(cfg.d, cfg.v_hidden, cfg.v_depth)

        # ----- Per-token mass + global gamma -----
        self.raw_m_bias = nn.Parameter(
            torch.tensor(_raw_from_positive(cfg.init_m)),
            requires_grad=cfg.learn_mgamma,
        )
        if cfg.fixed_gamma is not None:
            self.raw_gamma = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self._gamma_value: Optional[float] = float(cfg.fixed_gamma)
        else:
            self.raw_gamma = nn.Parameter(
                torch.tensor(_raw_from_positive(cfg.init_gamma)),
                requires_grad=cfg.learn_mgamma,
            )
            self._gamma_value = None

        if cfg.mass_mode == "logfreq":
            if cfg.logfreq_path is None:
                raise ValueError(
                    "mass_mode='logfreq' requires cfg.logfreq_path "
                    "(.npy with one surprisal value per vocabulary id)."
                )
            surprisal = torch.from_numpy(_load_npy(cfg.logfreq_path)).float()
            if surprisal.numel() != cfg.vocab_size:
                raise ValueError(
                    f"logfreq vector length {surprisal.numel()} != "
                    f"vocab_size {cfg.vocab_size}"
                )
            self.register_buffer("logfreq_surprisal", surprisal)
            self.raw_logfreq_alpha = nn.Parameter(
                torch.tensor(
                    _raw_from_positive(max(cfg.logfreq_init_alpha, 1e-3))
                ),
                requires_grad=True,
            )
        elif cfg.mass_mode == "global":
            pass
        else:
            raise ValueError(
                f"unknown mass_mode for helmholtz: {cfg.mass_mode!r}. "
                "Supported: 'logfreq', 'global'."
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _gpt2_init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @property
    def gamma(self) -> torch.Tensor:
        if self._gamma_value is not None:
            return torch.full(
                (), self._gamma_value,
                device=self.raw_gamma.device, dtype=self.raw_gamma.dtype,
            )
        return F.softplus(self.raw_gamma)

    @property
    def m_global(self) -> torch.Tensor:
        return F.softplus(self.raw_m_bias) + 1e-3

    def compute_mass(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.mass_mode == "global":
            return self.m_global
        if cfg.mass_mode == "logfreq":
            surprisal = self.logfreq_surprisal[x]                     # (B, T)
            alpha = F.softplus(self.raw_logfreq_alpha)                # ()
            scaled = alpha * surprisal.unsqueeze(-1)                  # (B, T, 1)
            return F.softplus(self.raw_m_bias + scaled) + 1e-3
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------
    def _project(self, h: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(h, (self.cfg.d,), eps=self.cfg.ln_eps)

    def _embed(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        B, T = x.shape
        pos = self.P[position_offset:position_offset + T].unsqueeze(0)
        return self.E(x) + pos

    # ------------------------------------------------------------------
    def _s_block_step(
        self,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        m_b: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """One velocity-Verlet damped Euler-Lagrange step under V_theta.

        Implements the schedule-S branch of doc section 2.2:

            delta = h - h_prev
            xi    = causal_cumulative_mean(h.detach())     # leak-fix
            f     = -grad_h V_theta(xi, h)
            h_new = h + delta / (1 + dt*gamma)
                      + (dt^2 / (m * (1 + dt*gamma))) * f
        """
        cfg = self.cfg

        delta = h - h_prev

        xi_input = h.detach() if cfg.causal_force else h
        xi_now = causal_cumulative_mean(xi_input)

        h_in = h
        if not h_in.requires_grad:
            h_in = h_in.requires_grad_(True)
        V = self.V_theta(xi_now, h_in).sum()
        grad_V, = torch.autograd.grad(
            V, h_in,
            create_graph=self.training,
            retain_graph=True,
        )
        f = -grad_V

        denom = 1.0 + dt * gamma
        h_new = h_in + delta / denom + (dt * dt / (m_b * denom)) * f

        if cfg.ln_after_s_step:
            h_new = self._project(h_new)
        return h_new

    def _a_block_step(
        self,
        h: torch.Tensor,
        a_idx: int,
        kv_cache_in: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """One pre-LN attention-and-MLP residual block."""
        return self.attn_blocks[a_idx](
            h, kv_cache=kv_cache_in, use_cache=use_cache,
        )

    # ------------------------------------------------------------------
    def _stack_forward(
        self,
        h0: torch.Tensor,
        x: torch.Tensor,
        return_trajectory: bool = False,
        kv_caches: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List]]:
        """Walk the schedule, dispatching each layer to S or A.

        Returns (h_final, traj_or_none, new_caches_or_none).  KV caches
        are maintained only over the A-block subset (length n_a_blocks).
        """
        cfg = self.cfg
        gamma, dt = self.gamma, cfg.dt
        m_b = self.compute_mass(x)

        h = h0
        h_prev = h0   # velocity proxy starts at 0 (delta = h - h_prev = 0)

        traj: Optional[List[torch.Tensor]] = None
        if return_trajectory:
            traj = [h.detach().cpu()]

        use_cache = kv_caches is not None
        new_caches: Optional[List] = [] if use_cache else None

        a_idx = 0
        for layer_kind, route in self._layer_routing:
            if layer_kind == "S":
                h_new = self._s_block_step(h, h_prev, m_b, gamma, dt)
            else:
                cache_in = kv_caches[a_idx] if use_cache else None
                h_new, new_cache = self._a_block_step(
                    h, route, kv_cache_in=cache_in, use_cache=use_cache,
                )
                if new_caches is not None:
                    new_caches.append(new_cache)
                a_idx += 1

            h_prev = h
            h = h_new

            if traj is not None:
                traj.append(h.detach().cpu())

        return h, traj, new_caches

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        kv_caches: Optional[List] = None,
        position_offset: int = 0,
    ):
        """Forward pass.

        Training: kv_caches=None, position_offset=0.
        AR-decode: pass kv_caches (list of length n_a_blocks, one entry
        per A-block in schedule order) and position_offset = T_past.
        """
        h0 = self._embed(x, position_offset=position_offset)
        h_L, traj, new_caches = self._stack_forward(
            h0, x,
            return_trajectory=return_trajectory,
            kv_caches=kv_caches,
        )

        logits = h_L @ self.E.weight.T

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
            )

        out = [logits, loss]
        if return_trajectory:
            out.append(traj)
        if new_caches is not None:
            out.append(new_caches)
        return tuple(out)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Naive AR generation (no KV cache)."""
        self.eval()
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.cfg.max_len:]
            with torch.enable_grad():
                logits, _ = self.forward(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, nxt], dim=1)
        return x

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Param-match table for canonical 7-schedule sweep at L=8
# ---------------------------------------------------------------------------
def canonical_schedules(L: int = 8) -> List[Tuple[str, str]]:
    """Return [(name, schedule_string), ...] for the section-6 patterns.

    For L=8 the registry is:
      bottom_a_LA4   AAAASSSS   (matches existing HSPLM (k=4, m=4))
      bottom_a_LA1   ASSSSSSS
      top_a_LA1      SSSSSSSA
      sandwich_k1    SAAAAAAS
      sandwich_k2    SSAAAASS
      inverse_sandwich_k1   ASSSSSSA
      interleaved    SASASASA
      all_s          SSSSSSSS
      all_a          AAAAAAAA
    """
    out = [
        ("bottom_a_LA4", make_schedule("bottom_a", L=L, LA=4)),
        ("bottom_a_LA1", make_schedule("bottom_a", L=L, LA=1)),
        ("top_a_LA1",    make_schedule("top_a",    L=L, LA=1)),
        ("sandwich_k1",  make_schedule("sandwich", L=L, k=1)),
        ("sandwich_k2",  make_schedule("sandwich", L=L, k=2)),
        ("inverse_sandwich_k1",
                          make_schedule("inverse_sandwich", L=L, k=1)),
    ]
    if L % 2 == 0:
        out.append(("interleaved", make_schedule("interleaved", L=L)))
    out.append(("all_s", make_schedule("all_s", L=L)))
    out.append(("all_a", make_schedule("all_a", L=L)))
    return out


def param_count_table(
    vocab_size: int = 50257,
    d: int = 128,
    max_len: int = 256,
    n_head: int = 4,
    mlp_mult: int = 4,
    v_hidden: int = 512,
    v_depth: int = 3,
    L: int = 8,
    logfreq_path: Optional[str] = None,
) -> List[Tuple[str, str, int, int, int]]:
    """Return [(name, sigma, n_S, n_A, num_params), ...] for canonical schedules.

    Uses mass_mode='global' if logfreq_path is None (cleaner counting).
    """
    rows: List[Tuple[str, str, int, int, int]] = []
    for name, sigma_str in canonical_schedules(L=L):
        nS, nA = schedule_counts(parse_schedule(sigma_str))
        cfg = HelmholtzConfig(
            vocab_size=vocab_size, d=d, max_len=max_len,
            schedule=sigma_str,
            n_head=n_head, mlp_mult=mlp_mult,
            v_hidden=v_hidden, v_depth=v_depth,
            mass_mode="global" if logfreq_path is None else "logfreq",
            logfreq_path=logfreq_path,
        )
        net = HelmholtzLM(cfg)
        rows.append((name, sigma_str, nS, nA, net.num_params()))
    return rows


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def smoke_test():
    """Tiny forward+backward over multiple schedules."""
    torch.manual_seed(0)
    V = 257

    base_kw = dict(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2,
        n_head=2, mlp_mult=2, mass_mode="global",
    )

    schedules_to_smoke = [
        ("bottom_a_LA2", "AASSSS"),    # L=6, mirrors existing HSPLM shape
        ("top_a_LA1",    "SSSSSA"),
        ("sandwich_k1",  "SAAAAS"),
        ("interleaved",  "SASASA"),
        ("all_s",        "SSSSSS"),
        ("all_a",        "AAAAAA"),
    ]
    for name, sigma_str in schedules_to_smoke:
        cfg = HelmholtzConfig(schedule=sigma_str, **base_kw)
        net = HelmholtzLM(cfg)
        x = torch.randint(0, V, (2, 16))
        y = torch.randint(0, V, (2, 16))
        logits, loss = net(x, y)
        loss.backward()
        nS, nA = net.n_s_blocks, net.n_a_blocks
        v_last = [p for p in net.V_theta.net.parameters()][-2]
        v_grad_str = (f"{v_last.grad.norm().item():.4e}"
                      if v_last.grad is not None else "None (no S-blocks)")
        print(f"[helm smoke]  schedule={sigma_str!r:>10s}  "
              f"({name})  n_S={nS} n_A={nA}  "
              f"params={net.num_params():,}  "
              f"loss={loss.item():.4f}  "
              f"E.grad_norm={net.E.weight.grad.norm().item():.4f}  "
              f"V_theta.last_grad={v_grad_str}")

    # Trajectory extraction sanity check on the interleaved schedule.
    cfg = HelmholtzConfig(schedule="SASASA", **base_kw)
    net = HelmholtzLM(cfg)
    net.eval()
    x = torch.randint(0, V, (2, 16))
    y = torch.randint(0, V, (2, 16))
    with torch.enable_grad():
        _, _, traj = net(x, y, return_trajectory=True)
    assert len(traj) == net.L + 1, (len(traj), net.L + 1)
    assert traj[0].shape == (2, 16, cfg.d)
    print(f"[helm smoke]  trajectory ok: "
          f"{len(traj)} layers x {tuple(traj[0].shape)}  on schedule SASASA")

    # Bottom-A reproducibility check: two-stage schedule = HSPLM layout
    # (param count should equal hybrid Variant A (k, m) cell at matched
    # config).
    print(f"[helm smoke]  bottom_a (AAAASSSS-equivalent at smoke scale) "
          f"is the natural Q9d analogue of HSPLM (k=4, m=4).")


def print_param_match_table():
    print()
    print("=" * 78)
    print("Helmholtz param-match table at d=128, max_len=256, vocab=50257, "
          "L=8,")
    print("n_head=4, mlp_mult=4, v_hidden=512, v_depth=3 "
          "(mass_mode='global')")
    print("=" * 78)
    rows = param_count_table()
    print(f"{'name':>22}  {'schedule':>10}  {'n_S':>4}  {'n_A':>4}  "
          f"{'params':>12}  {'vs 8.0 M':>10}")
    for name, sigma_str, nS, nA, n_params in rows:
        delta = (n_params - 8_000_000) / 1e6
        print(f"{name:>22}  {sigma_str:>10}  {nS:>4d}  {nA:>4d}  "
              f"{n_params:>12,}  {delta:>+8.3f} M")
    print("=" * 78)
    print()
    print("Notes:")
    print("  - 'all_a' has nS=0 but still allocates V_theta (~700k);")
    print("    use matched_baseline_model.MatchedGPT directly for a")
    print("    bit-exact param-matched all-attn baseline.")
    print("  - 'bottom_a_LA4' is the closest Q9d analogue of HSPLM (k=4, m=4);")
    print("    same params + same compute, but the velocity proxy passes")
    print("    through the attention stack and xi is re-derived each S-block.")
    print()


if __name__ == "__main__":
    smoke_test()
    print_param_match_table()
