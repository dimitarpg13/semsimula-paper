"""
Hybrid Attention + Fock-PARF language model.

Architecture (two-stage)
------------------------
  h_0 = E[x] + P
  for i = 1..n_attn:
      h_i = AttnBlock_i(h_{i-1})             # learned attention front-end
  h_k = LayerNorm(h_{n_attn})
  for j = 1..L:
      h_j = FockPARF_layer_step(h_{j-1}, registers, ...)
          creation gate   → update register salience
          active mask     → gate register participation
          V_θ + V_φ       → conservative force on [tokens, registers]
          velocity-Verlet → advance hidden states
          destruction gate → decay register salience
  logits = h_L @ E^T                         # tied embeddings

Rationale
---------
Combines the two highest-performing architectural strategies:
  1. Attention front-end (from HybridSPLM): O(T·d) per new token with
     KV cache; provides the fast global-context gathering that closes
     the PPL gap to attention baselines.
  2. FockPARF back-end (from FockPARFLM): V_θ + sparse V_φ pair
     interactions + latent register lifecycle (v0+v1.5+v2).  Provides
     three expressivity channels beyond what plain SPLM integration
     offers in HybridSPLM.

The prediction is that this architecture should outperform both:
  - HybridSPLM (which uses plain V_θ-only SPLM as its back-end)
  - FockPARFLM (which lacks the attention front-end)

Causal-leak fix
---------------
xi is re-derived from h.detach() at each FockPARF layer step (when
cfg.causal_force=True), preserving the causal-honesty invariant.
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
_PARENT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(_THIS_DIR))

from matched_baseline_model import (  # noqa: E402
    Block as AttnBlock,
    MatchedConfig,
)
from model_fock_parf import (  # noqa: E402
    FockPARFConfig,
    FockPARFLM,
)


@dataclass
class HybridFockPARFConfig(FockPARFConfig):
    """Hybrid Attention + FockPARF configuration.

    Extends FockPARFConfig with attention-stage parameters.  The total
    layer budget is n_attn (attention blocks) + L (FockPARF integration
    steps).  For a fair comparison with HybridSPLM at (k=4, m=4), use
    n_attn=4, L=4.
    """
    n_attn: int = 4
    n_head: int = 4
    mlp_mult: int = 4
    dropout: float = 0.0


def _attn_cfg_from(cfg: HybridFockPARFConfig) -> MatchedConfig:
    return MatchedConfig(
        vocab_size=cfg.vocab_size,
        d=cfg.d,
        max_len=cfg.max_len,
        n_layer=cfg.n_attn,
        n_head=cfg.n_head,
        mlp_mult=cfg.mlp_mult,
        dropout=cfg.dropout,
        tie_embeddings=cfg.tie_embeddings,
    )


class HybridFockPARF(FockPARFLM):
    """Hybrid Attention + FockPARF language model.

    Inherits the full FockPARF dynamics (V_θ + sparse V_φ + register
    lifecycle with creation/destruction gates) and prepends an attention
    front-end.  The attention blocks are identical to those in HybridSPLM
    (reused from matched_baseline_model.Block).

    Forward contract
    ----------------
      forward(x, targets=None, return_trajectory=False)
        -> (logits, loss[, traj])
    """

    cfg: HybridFockPARFConfig

    def __init__(self, cfg: HybridFockPARFConfig):
        if not isinstance(cfg, HybridFockPARFConfig):
            raise TypeError(
                f"HybridFockPARF requires HybridFockPARFConfig, "
                f"got {type(cfg)!r}."
            )
        super().__init__(cfg)

        attn_cfg = _attn_cfg_from(cfg)
        self.attn_blocks = nn.ModuleList(
            [AttnBlock(attn_cfg) for _ in range(cfg.n_attn)]
        )
        self.ln_boundary = nn.LayerNorm(cfg.d, eps=cfg.ln_eps)
        self.attn_blocks.apply(self._gpt2_init)

    @staticmethod
    def _gpt2_init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _attn_stack(
        self,
        h: torch.Tensor,
        kv_caches: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        use_cache = kv_caches is not None
        new_caches: Optional[List] = [] if use_cache else None
        for i, blk in enumerate(self.attn_blocks):
            cache_in = kv_caches[i] if use_cache else None
            h, new_cache = blk(h, kv_cache=cache_in, use_cache=use_cache)
            if new_caches is not None:
                new_caches.append(new_cache)
        return h, new_caches

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        kv_caches: Optional[List] = None,
        position_offset: int = 0,
    ):
        # 1. Embed
        h0 = self._embed(x, position_offset=position_offset)

        # 2. Attention front-end
        h_attn, new_caches = self._attn_stack(h0, kv_caches=kv_caches)
        h_k = self.ln_boundary(h_attn)

        # 3. FockPARF back-end (V_θ + sparse V_φ + register lifecycle)
        h_L, traj = self._stack_forward(
            h_k, x, return_trajectory=return_trajectory,
        )

        # 4. Logits (tied embeddings)
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

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
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

    def get_attn_overhead(self) -> int:
        """Count parameters specific to the attention front-end."""
        count = sum(p.numel() for blk in self.attn_blocks
                    for p in blk.parameters())
        count += sum(p.numel() for p in self.ln_boundary.parameters())
        return count


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke():
    """Minimal forward+backward on CPU."""
    cfg = HybridFockPARFConfig(
        vocab_size=257, d=64, max_len=64, L=4,
        v_hidden=64, v_depth=2,
        v_phi_d_type=4, v_phi_d_angle=2,
        v_phi_phi_hidden=8, v_phi_theta_hidden=8,
        v_phi_mlp_hidden=16,
        mass_mode="global",
        top_k=8,
        score_head_hidden=8,
        n_registers=16,
        creation_gate_hidden=32,
        stack_discipline=True,
        n_attn=2, n_head=4, mlp_mult=2,
    )
    torch.manual_seed(0)
    net = HybridFockPARF(cfg)
    total_params = sum(p.numel() for p in net.parameters())
    attn_overhead = net.get_attn_overhead()
    fock_overhead = net.get_register_overhead()
    base = total_params - attn_overhead - fock_overhead
    print(f"[hybrid-fock-smoke] total params: {total_params:,}")
    print(f"[hybrid-fock-smoke] attn overhead: {attn_overhead:,} "
          f"({100*attn_overhead/total_params:.1f}%)")
    print(f"[hybrid-fock-smoke] fock overhead: {fock_overhead:,} "
          f"({100*fock_overhead/total_params:.1f}%)")
    print(f"[hybrid-fock-smoke] base PARF params: {base:,}")
    print(f"[hybrid-fock-smoke] n_attn={cfg.n_attn}, L={cfg.L}, "
          f"M={cfg.n_registers}")

    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    net.train()
    logits, loss = net(x, targets=y)
    print(f"[hybrid-fock-smoke] forward: logits {tuple(logits.shape)} "
          f"loss {loss.item():.4f}")
    loss.backward()
    print("[hybrid-fock-smoke] backward OK; no exceptions.")

    net.eval()
    with torch.enable_grad():
        _, _, traj = net(x, targets=y, return_trajectory=True)
    assert len(traj) == cfg.L + 1, (len(traj), cfg.L + 1)
    assert traj[0].shape == (2, 16, cfg.d)
    print(f"[hybrid-fock-smoke] trajectory ok: "
          f"{len(traj)} layers x {tuple(traj[0].shape)}")


if __name__ == "__main__":
    _smoke()
