"""
Hybrid SPLM + Attention language model (HSPLM, Variant A).

Architecture (two-stage)
------------------------
  h_0 = E[x] + P
  for i = 1..n_attn:
      h_i = AttnBlock_i(h_{i-1})            # k distinct attention blocks
  h_k = LayerNorm(h_{n_attn})
  xi  = causal_cumulative_mean(h_k.detach())  # leak-safe re-derivation of xi
  for j = 1..n_splm:
      f = -grad_h V_theta(xi, h)
      v = (v + dt * f / m) / (1 + dt * gamma)
      h = h + dt * v
      h = LayerNorm(h)                       # if ln_after_step
  logits = h @ E^T                           # tied embeddings

Rationale
---------
The attention stack does what attention does best (gather global context
across positions); the SPLM stack does what scalar-potential dynamics do
best (refine each position deterministically through a learned energy
field).  At decode time, the n_attn blocks pay O(T·d) per new token (with
KV cache); the n_splm steps pay O(d^2) per new token, *independent of T*.
At long T the SPLM tail is essentially free — this is the FLOP-efficiency
hypothesis the hybrid is meant to test (see
the v4 title-justification rule §6.5 for the
pre-registered title-justification rule).

Causal-leak fix
---------------
xi is re-derived from h_k.detach() (when cfg.causal_force=True) so that
∂V[t']/∂h[t] = 0 for t' > t.  This preserves the v3 causal-honesty
invariant introduced in:
  notebooks/conservative_arch/sarf_mass_variant/model_sarf_mass.py
  notebooks/conservative_arch/energetic_minima/model_ln.py
  companion_notes/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md

Single shared V_theta
---------------------
A single V_theta is shared across all n_splm integration steps, exactly
as in the canonical SPLM em_ln variant.  This preserves the
"single energy field" interpretation that grounds the Lagrangian-
mechanics framing of paper_v3.  An ablation with per-step V_thetas can
be added later if (i) is empirically necessary.
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


# Reuse the canonical attention block + the SPLM scalar potential MLP.
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


@dataclass
class HSPLMConfig:
    """Hybrid SPLM + Attention configuration.

    The architectural budget is `n_attn + n_splm` "layers" (where an
    attention block and an SPLM integration step are counted as one
    layer slot each).  Defaults match the leak-free em_ln cell:
    d=128, max_len=256, n_attn=4, n_splm=4, v_hidden=512, v_depth=3,
    n_head=4, mlp_mult=4, mass_mode="logfreq", fixed_gamma=None.
    """
    vocab_size: int = 50257
    d: int = 128
    max_len: int = 256

    # Attention stage
    n_attn: int = 4
    n_head: int = 4
    mlp_mult: int = 4
    dropout: float = 0.0

    # SPLM stage
    n_splm: int = 4
    v_hidden: int = 512
    v_depth: int = 3
    dt: float = 1.0
    init_m: float = 1.0
    init_gamma: float = 0.15
    learn_mgamma: bool = True
    fixed_gamma: Optional[float] = None

    # Per-token mass (matches leak-free SPLM em_ln cells)
    mass_mode: str = "logfreq"  # currently only "logfreq" supported in hybrid
    logfreq_init_alpha: float = 0.1
    logfreq_path: Optional[str] = None

    # Misc
    ln_after_step: bool = True
    ln_eps: float = 1e-5
    causal_force: bool = True   # leak-fix invariant; should always be True
    tie_embeddings: bool = True


def _load_npy(p) -> np.ndarray:
    return np.load(p)


def _attn_cfg_from(cfg: HSPLMConfig) -> MatchedConfig:
    """Build a MatchedConfig view that AttnBlock accepts."""
    return MatchedConfig(
        vocab_size=cfg.vocab_size,
        d=cfg.d,
        max_len=cfg.max_len,
        n_layer=cfg.n_attn,         # not used by AttnBlock directly
        n_head=cfg.n_head,
        mlp_mult=cfg.mlp_mult,
        dropout=cfg.dropout,
        tie_embeddings=cfg.tie_embeddings,
    )


class HybridSPLM(nn.Module):
    """Hybrid SPLM + Attention language model (Variant A: two-stage).

    Forward contract
    ----------------
      forward(x, targets=None, return_trajectory=False) -> (logits, loss[, traj])

    where traj is the per-step list of hidden states at SPLM positions
    (length n_splm + 1) on CPU, returned only when return_trajectory=True.
    """

    def __init__(self, cfg: HSPLMConfig):
        super().__init__()
        self.cfg = cfg

        # Embeddings (token + position).  Tied output.
        self.E = nn.Embedding(cfg.vocab_size, cfg.d)
        self.P = nn.Parameter(torch.zeros(cfg.max_len, cfg.d))
        nn.init.normal_(self.E.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.P, mean=0.0, std=0.02)

        # Attention stack.
        attn_cfg = _attn_cfg_from(cfg)
        self.attn_blocks = nn.ModuleList(
            [AttnBlock(attn_cfg) for _ in range(cfg.n_attn)]
        )
        # Boundary LayerNorm after attention stack (clean transition into
        # the SPLM stage; mirrors the GPT-2 ln_f conventions).
        self.ln_boundary = nn.LayerNorm(cfg.d, eps=cfg.ln_eps)

        # Apply GPT-2 init to attention blocks for parity with
        # matched_baseline_model.MatchedGPT.
        self.attn_blocks.apply(self._gpt2_init)

        # SPLM stage: shared scalar potential.
        self.V_theta = ScalarPotential(cfg.d, cfg.v_hidden, cfg.v_depth)

        # Per-token mass parameters (logfreq mode).
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
                torch.tensor(_raw_from_positive(max(cfg.logfreq_init_alpha, 1e-3))),
                requires_grad=True,
            )
        elif cfg.mass_mode == "global":
            # Allowed for diagnostic runs; mostly we want logfreq.
            pass
        else:
            raise ValueError(
                f"unknown mass_mode for hybrid: {cfg.mass_mode!r}. "
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
        """Per-token mass tensor m of shape (B, T, 1) or 0-d for global."""
        cfg = self.cfg
        if cfg.mass_mode == "global":
            return self.m_global
        if cfg.mass_mode == "logfreq":
            surprisal = self.logfreq_surprisal[x]                      # (B, T)
            alpha = F.softplus(self.raw_logfreq_alpha)                 # ()
            scaled = alpha * surprisal.unsqueeze(-1)                   # (B, T, 1)
            return F.softplus(self.raw_m_bias + scaled) + 1e-3
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------
    def _project(self, h: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(h, (self.cfg.d,), eps=self.cfg.ln_eps)

    def _embed(self, x: torch.Tensor,
               position_offset: int = 0) -> torch.Tensor:
        B, T = x.shape
        pos = self.P[position_offset:position_offset + T].unsqueeze(0)
        return self.E(x) + pos

    # ------------------------------------------------------------------
    def _attn_stack(
        self,
        h: torch.Tensor,
        kv_caches: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """Run the n_attn attention blocks; optionally maintain KV caches."""
        use_cache = kv_caches is not None
        new_caches: Optional[List] = [] if use_cache else None
        for i, blk in enumerate(self.attn_blocks):
            cache_in = kv_caches[i] if use_cache else None
            h, new_cache = blk(h, kv_cache=cache_in, use_cache=use_cache)
            if new_caches is not None:
                new_caches.append(new_cache)
        return h, new_caches

    def _splm_integrate(
        self,
        h_k: torch.Tensor,
        x: torch.Tensor,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Run the n_splm shared-V damped integration steps starting from h_k.

        xi is re-derived from h_k.detach() (causal-leak fix) and held fixed
        across the integration loop, mirroring the SARF-faithful
        single-pool prescription of model_sarf_mass.py.
        """
        cfg = self.cfg
        h = self._project(h_k) if cfg.ln_after_step else h_k
        v = torch.zeros_like(h)
        gamma, dt = self.gamma, cfg.dt

        m_b = self.compute_mass(x)

        traj: Optional[List[torch.Tensor]] = None
        if return_trajectory:
            traj = [h.detach().cpu()]

        for _ in range(cfg.n_splm):
            # Causal-leak fix: re-derive xi from current h.detach() each step
            # (matches model_ln.py / model_sarf_mass.py canonical behaviour).
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
            v = (v + dt * f / m_b) / (1.0 + dt * gamma)
            h_new = h_in + dt * v
            if cfg.ln_after_step:
                h_new = self._project(h_new)
            h = h_new
            if return_trajectory:
                assert traj is not None
                traj.append(h.detach().cpu())

        return h, traj

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                        # (B, T) int64
        targets: Optional[torch.Tensor] = None,  # (B, T) int64
        return_trajectory: bool = False,
        kv_caches: Optional[List] = None,
        position_offset: int = 0,
    ):
        """Forward.

        Training mode: kv_caches=None, position_offset=0.
        AR-decode mode: pass kv_caches (list of length n_attn) and
        position_offset = T_past.  The function returns the updated
        kv_caches as the last tuple element.
        """
        h0 = self._embed(x, position_offset=position_offset)
        h_attn, new_caches = self._attn_stack(h0, kv_caches=kv_caches)
        h_k = self.ln_boundary(h_attn)
        h_L, traj = self._splm_integrate(
            h_k, x, return_trajectory=return_trajectory,
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
        if len(out) == 2:
            return tuple(out)
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
# Parameter-matching analysis.  At d=128, max_len=256, vocab=50257 we want
# total trainable params close to the matched-attention 8.0M baseline so
# the hybrid is a fair architectural ablation.
# ---------------------------------------------------------------------------
def param_count_table(
    vocab_size: int = 50257,
    d: int = 128,
    max_len: int = 256,
    n_head: int = 4,
    mlp_mult: int = 4,
    v_hidden: int = 512,
    v_depth: int = 3,
    pairs: Optional[List[Tuple[int, int]]] = None,
    logfreq_path: Optional[str] = None,
) -> List[Tuple[Tuple[int, int], int]]:
    """Return a list of ((n_attn, n_splm), num_params) tuples.

    If logfreq_path is None, falls back to mass_mode='global' so we can
    report params without needing the surprisal file (only ±1 param
    relative to logfreq).
    """
    if pairs is None:
        # n_attn + n_splm = 8 budget (matches matched_baseline L=8).
        pairs = [(0, 8), (1, 7), (2, 6), (3, 5), (4, 4),
                 (5, 3), (6, 2), (7, 1), (8, 0)]
    out: List[Tuple[Tuple[int, int], int]] = []
    for k, m in pairs:
        if k == 0 and m == 0:
            continue
        cfg = HSPLMConfig(
            vocab_size=vocab_size, d=d, max_len=max_len,
            n_attn=k, n_splm=m,
            n_head=n_head, mlp_mult=mlp_mult,
            v_hidden=v_hidden, v_depth=v_depth,
            mass_mode="global" if logfreq_path is None else "logfreq",
            logfreq_path=logfreq_path,
        )
        if cfg.n_attn == 0:
            net = _NullAttnHybrid(cfg)
        else:
            net = HybridSPLM(cfg)
        out.append(((k, m), net.num_params()))
    return out


class _NullAttnHybrid(HybridSPLM):
    """Edge case for k=0: no attention blocks at all (pure SPLM stack).

    Defined here only so the param-count table can compare against pure
    SPLM cleanly without instantiating attention blocks of size 0.
    """
    def __init__(self, cfg: HSPLMConfig):
        # Bypass the parent __init__'s attention-stack construction by
        # temporarily setting n_head to 1 and n_attn to 0; nn.ModuleList
        # of length 0 is the natural way to skip the stack.
        nn.Module.__init__(self)
        self.cfg = cfg
        self.E = nn.Embedding(cfg.vocab_size, cfg.d)
        self.P = nn.Parameter(torch.zeros(cfg.max_len, cfg.d))
        nn.init.normal_(self.E.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.P, mean=0.0, std=0.02)
        self.attn_blocks = nn.ModuleList()
        self.ln_boundary = nn.LayerNorm(cfg.d, eps=cfg.ln_eps)
        self.V_theta = ScalarPotential(cfg.d, cfg.v_hidden, cfg.v_depth)
        self.raw_m_bias = nn.Parameter(
            torch.tensor(_raw_from_positive(cfg.init_m)),
            requires_grad=cfg.learn_mgamma,
        )
        if cfg.fixed_gamma is not None:
            self.raw_gamma = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self._gamma_value = float(cfg.fixed_gamma)
        else:
            self.raw_gamma = nn.Parameter(
                torch.tensor(_raw_from_positive(cfg.init_gamma)),
                requires_grad=cfg.learn_mgamma,
            )
            self._gamma_value = None
        if cfg.mass_mode == "global":
            pass
        elif cfg.mass_mode == "logfreq" and cfg.logfreq_path is not None:
            surprisal = torch.from_numpy(_load_npy(cfg.logfreq_path)).float()
            self.register_buffer("logfreq_surprisal", surprisal)
            self.raw_logfreq_alpha = nn.Parameter(
                torch.tensor(_raw_from_positive(max(cfg.logfreq_init_alpha, 1e-3))),
                requires_grad=True,
            )


# ---------------------------------------------------------------------------
# Self-test: forward/backward, trajectory shape, parameter count vs (k, m).
# Run with: python3 model_hybrid.py
# ---------------------------------------------------------------------------
def smoke_test():
    torch.manual_seed(0)
    V = 257
    cfg = HSPLMConfig(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2,
        n_attn=2, n_splm=3, n_head=2, mlp_mult=2,
        mass_mode="global",   # avoid loading logfreq for smoke
    )
    net = HybridSPLM(cfg)
    print(f"[hybrid smoke]  params={net.num_params():,}  "
          f"k={cfg.n_attn}, m={cfg.n_splm}  "
          f"gamma={net.gamma.item():.3f}")

    x = torch.randint(0, V, (2, 16))
    y = torch.randint(0, V, (2, 16))
    logits, loss = net(x, y)
    loss.backward()
    print(f"[hybrid smoke]  forward+backward ok: "
          f"loss={loss.item():.4f}  logits={tuple(logits.shape)}  "
          f"E.grad_norm={net.E.weight.grad.norm().item():.4f}")

    net.eval()
    with torch.enable_grad():
        _, _, traj = net(x, y, return_trajectory=True)
    assert len(traj) == cfg.n_splm + 1, (len(traj), cfg.n_splm + 1)
    assert traj[0].shape == (2, 16, cfg.d)
    print(f"[hybrid smoke]  trajectory ok: "
          f"{len(traj)} layers x {tuple(traj[0].shape)}")


def print_param_match_table():
    print()
    print("=" * 64)
    print("Parameter-match table at d=128, max_len=256, vocab=50257, "
          "n_head=4, mlp_mult=4, v_hidden=512, v_depth=3")
    print("(mass_mode='global' for fair counting; logfreq adds 1 param)")
    print("=" * 64)
    rows = param_count_table()
    print(f"{'(n_attn, n_splm)':>20}   {'params':>12}    "
          f"{'vs 8.0 M':>10}")
    for (k, m), n in rows:
        delta = (n - 8_000_000) / 1_000_000
        sign = "+" if delta >= 0 else ""
        print(f"{('(' + str(k) + ', ' + str(m) + ')'):>20}   "
              f"{n:>12,}    {sign}{delta:>+8.3f} M")
    print("=" * 64)
    print()
    print("Notes:")
    print("  - All-attention (8, 0) corresponds to matched_baseline_model "
          "(the leak-immune ATTN baseline, val PPL ~150 on Tiny "
          "Shakespeare at this width).")
    print("  - All-SPLM (0, 8) corresponds to em_ln (val PPL ~178-181 "
          "leak-free; ~88 buggy v2).")
    print("  - The hybrid cells (k>=1, m>=1) at this width are within "
          "+/- ~1 M params of 8.0 M; param-matching is dominated by the "
          "embedding (~6.4 M) which is shared.")
    print()


if __name__ == "__main__":
    smoke_test()
    print_param_match_table()
