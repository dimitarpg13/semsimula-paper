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

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Causal self-attention with optional KV-cache.

        Args:
            x:         (B, T_new, D)
            kv_cache:  optional (K_cached, V_cached) of shape
                       (B, H, T_past, D_h); pass None for the first
                       step of decode mode (new_cache will still be
                       returned).
            use_cache: if True, return new_cache; if False, return None
                       (training-style behaviour).
        Returns:
            out:           (B, T_new, D)
            new_kv_cache:  updated KV cache, or None if use_cache=False.
        """
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        if kv_cache is not None:
            K_past, V_past = kv_cache
            k = torch.cat([K_past, k], dim=2)
            v = torch.cat([V_past, v], dim=2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        elif use_cache:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        new_cache = (k, v) if use_cache else None
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.drop(self.proj(out)), new_cache


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

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache,
                                        use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


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

    def forward(
        self,
        x: torch.Tensor,                          # (B, T) int64
        targets: Optional[torch.Tensor] = None,   # (B, T) int64
        return_trajectory: bool = False,
        kv_caches: Optional[list] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward.

        Args:
            x:               (B, T_new) integer token ids.
            targets:         optional (B, T_new) for loss.
            return_trajectory: if True, return per-layer hidden states.
            kv_caches:       optional list of length n_layer of (K, V)
                             cached pairs from previous decode steps.
                             If provided, the position embedding starts
                             at `position_offset` and the new KV is
                             concatenated to the cache; the function
                             returns the updated cache as a third tuple
                             element. This is the AR-decode mode.
            position_offset: position offset for the position embedding
                             (used with kv_caches to index P[T_past:T_past+T_new]).
        """
        B, T = x.shape
        pos = self.P[position_offset:position_offset + T]
        h = self.drop(self.E(x) + pos)

        use_cache = kv_caches is not None
        traj = [h] if return_trajectory else None
        new_caches: list = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            cache_in = kv_caches[i] if use_cache else None
            h, new_cache = block(h, kv_cache=cache_in, use_cache=use_cache)
            if new_caches is not None:
                new_caches.append(new_cache)
            if traj is not None:
                traj.append(h)

        h = self.ln_f(h)
        logits = h @ self.E.weight.T

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        out = [logits, loss]
        if traj is not None:
            out.append(traj)
        if new_caches is not None:
            out.append(new_caches)
        if len(out) == 2:
            return tuple(out)
        return tuple(out)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = MatchedConfig()
    m = MatchedGPT(cfg)
    print(f"MatchedGPT params: {m.num_params():,}")
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    y = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = m(x, y)
    print(f"logits: {tuple(logits.shape)}   loss: {loss.item():.4f}")
