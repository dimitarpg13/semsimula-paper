"""
Matched-parameter GPT-2-style decoder -- the step-3 negative control.

Purpose
-------
The Shakespeare-trained `ScalarPotentialLM` (7.1 M params) is compared
in step 1 + step 2 against *pretrained* GPT-2 small (124 M params, trained
on WebText).  A sceptic will rightly ask whether the observed gap --
median per-layer shared-V_psi TEST R^2 of +0.90 for SPLM vs +0.45 for
GPT-2, with GPT-2's middle layers 6-10 collapsing to mean R^2 = +0.09 --
is explained by parameter count or pretraining data rather than
architecture.

This module defines a matched-parameter GPT-2-style transformer that
we train on the *same* Tiny Shakespeare data with the *same* token
budget as SPLM.  The per-block structure is the canonical GPT-2 block
(pre-LN, multi-head causal attention, GELU MLP at 4x width) without
any of the prescriptive structure of SPLM (no weight-tied integration,
no shared scalar potential).  All other differences from SPLM are
controlled: tokenizer, vocabulary, embedding dim, depth, and training
setup match.

Config
------
Default shakespeare-mode matched baseline:
  d = 128, max_len = 256, n_layer = 8, n_head = 4, mlp_mult = 4
  tied embedding (same as SPLM)
  ~8.0 M parameters (vs SPLM 7.1 M)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
@dataclass
class MatchedConfig:
    vocab_size: int = 50257
    d: int = 128
    max_len: int = 256
    n_layer: int = 8
    n_head: int = 4
    mlp_mult: int = 4
    dropout: float = 0.0
    tie_embeddings: bool = True


# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: MatchedConfig):
        super().__init__()
        assert cfg.d % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d // cfg.n_head
        self.qkv    = nn.Linear(cfg.d, 3 * cfg.d, bias=True)
        self.proj   = nn.Linear(cfg.d, cfg.d, bias=True)
        self.drop   = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x)                                          # (B,T,3D)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B,H,T,Dh)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, cfg: MatchedConfig):
        super().__init__()
        hidden = cfg.mlp_mult * cfg.d
        self.fc1 = nn.Linear(cfg.d, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, cfg.d, bias=True)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: MatchedConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
class MatchedGPT(nn.Module):
    """Canonical tiny GPT-2 decoder.  Matched with SPLM on vocabulary,
    embedding dim, max_len, and tied embedding -- but structurally
    nothing like SPLM internally."""

    def __init__(self, cfg: MatchedConfig):
        super().__init__()
        self.cfg = cfg
        self.E = nn.Embedding(cfg.vocab_size, cfg.d)
        self.P = nn.Parameter(torch.zeros(cfg.max_len, cfg.d))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d)

        # Init matching SPLM conventions.
        nn.init.normal_(self.E.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.P, mean=0.0, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        if self.cfg.tie_embeddings:
            pass
        return n

    # -----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                          # (B, T) int64
        targets: Optional[torch.Tensor] = None,   # (B, T) int64
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = x.shape
        h = self.drop(self.E(x) + self.P[:T])

        traj = [h] if return_trajectory else None
        for block in self.blocks:
            h = block(h)
            if traj is not None:
                traj.append(h)

        h = self.ln_f(h)

        if self.cfg.tie_embeddings:
            logits = h @ self.E.weight.T
        else:
            logits = h @ self.E.weight.T    # (fallback: still tied since no lm_head was defined)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return (logits, loss) if traj is None else (logits, loss, traj)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = MatchedConfig()
    m = MatchedGPT(cfg)
    print(f"MatchedGPT params: {m.num_params():,}")
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    y = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = m(x, y)
    print(f"logits: {tuple(logits.shape)}   loss: {loss.item():.4f}")
