"""Incremental (token-by-token) decoding for the Gen 3 conservative core, and a key-value-cache decoder for the
matched GPT-2 to compare it with (2026-11-17; stage 2 of
companion_notes/productization/Conservative_Core_Productization_Plan.md, lever N4).

With the pair force disabled and the unread Fock registers skipped, the only pathway between positions in a ladder
model is the context channels xi: at layer l, position t reads

    xi_{k,t} = sum_{s <= t} softmax_s((t - s) log a_k + q_{k,t} . kappa_{k,s} / sqrt(d_k)) h_s

over that layer's input states h_s (MultiChannelXi.forward). Everything else -- the well bank, the stiff-mode
decomposition, the integrator, the mass, the output head -- acts within one position. So the decoder runs the model's
own forward on the new token alone (position_offset = t) and swaps in a context-channel module that reads a cache of
the earlier keys kappa_s and states h_s, one cache per layer: K * d_k + d numbers per token and layer (624 for the
ladder's K = 5, d_k = 48, d = 384). The prefill is one ordinary forward that fills the cache on the way.

The decoder refuses configurations in which anything else crosses positions (the routed pair force, an exchange
field, Poisson-mode or particle-store terms, a reverse channel, read registers, a variable-step or substep walk).
debug/cpu_throughput_benchmark.py decode_check= compares its logits with the full forward's.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class LadderDecoder:
    """Token-by-token decoding of a ladder model (FockMultiXiPARFLM) with the pair force disabled.

    drop_pair_force=True replaces the closed-form pair force by zero on the model (the configuration of the
    productization plan's bound); it is required when the model has a pair force, so that the change is explicit.
    """

    def __init__(self, model, max_len: int, batch: int = 1, drop_pair_force: bool = False):
        m, cfg = model, model.cfg
        if getattr(m, "V_phi", None) is not None:
            if not drop_pair_force:
                raise NotImplementedError("the routed pair force crosses positions and has no decoder cache here; "
                                          "pass drop_pair_force=True to decode without it")
            if not cfg.vphi_analytic_force:
                raise ValueError("drop_pair_force replaces the closed-form pair force: set cfg.vphi_analytic_force")
            m._pair_force_analytic = (lambda h_in, layer_idx, h_src_live=None, q_slice=None:
                                      torch.zeros_like(h_in))
        for name in ("relax_field", "pm_mu", "pstore_mu", "V_attn", "reverse_ch"):
            if getattr(m, name, None) is not None:
                raise NotImplementedError(f"{name} couples positions; the decoder does not cache it")
        if not (getattr(cfg, "fock_version", None) == "v2" and getattr(cfg, "prefix_causal_registers", False)
                and getattr(cfg, "skip_unread_registers", False)):
            raise NotImplementedError("the decoder needs prefix-causal, unread registers with "
                                      "cfg.skip_unread_registers = True")
        if (getattr(cfg, "substeps_per_layer", 1) or 1) != 1 or getattr(m, "stack_steps", None) is not None:
            raise NotImplementedError("the decoder assumes the trained walk: one step per layer")
        xm = m.xi_module
        self.m, self.xm, self.L = m, xm, int(cfg.L)
        self.K, self.d = int(xm.K), int(cfg.d)
        self.content = xm.W_q is not None
        self.dk = int(xm.content_d_k) if self.content else 0
        dt = m.P.dtype
        self.keys = [torch.zeros(batch, self.K, max_len, self.dk, dtype=dt) for _ in range(self.L)]
        self.states = [torch.zeros(batch, max_len, self.d, dtype=dt) for _ in range(self.L)]
        self.max_len, self.t, self._layer = max_len, 0, 0
        self._xi_full = xm.forward              # the module's own forward, used for the prefill
        xm.forward = self._xi

    def cache_numbers_per_token(self) -> int:
        return self.L * (self.K * self.dk + self.d)

    def close(self) -> None:
        """Give the context-channel module its own forward back (the pair force stays dropped)."""
        self.xm.forward = self._xi_full

    def _xi(self, h: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        li, t0 = self._layer, self.t
        self._layer += 1
        if t0 + T > self.max_len:
            raise RuntimeError(f"decoder cache full ({self.max_len} positions)")
        xm = self.xm
        self.states[li][:B, t0:t0 + T] = h
        if self.content:
            kk = xm.W_k(h).view(B, T, self.K, self.dk).permute(0, 2, 1, 3)
            self.keys[li][:B, :, t0:t0 + T] = kk
        if T > 1:                               # the prefill: the ordinary computation over the prompt
            if t0 != 0:
                raise NotImplementedError("a multi-token step is supported only as the prefill")
            return self._xi_full(h)
        n = t0 + 1                              # positions 0..t0, the new token included
        la = torch.log(xm.alpha.clamp(min=1e-6, max=1.0 - 1e-6))                   # as causal_ema_logits
        dist = (t0 - torch.arange(n, dtype=h.dtype, device=h.device))              # t - s >= 0
        logits = (la.view(self.K, 1) * dist.view(1, n)).unsqueeze(0).unsqueeze(2)  # (1, K, 1, n)
        if self.content:
            q = xm.W_q(h).view(B, 1, self.K, self.dk).permute(0, 2, 1, 3)        # (B, K, 1, dk)
            logits = logits + (q @ self.keys[li][:B, :, :n].transpose(-1, -2)) * (self.dk ** -0.5)
        w = torch.softmax(logits, dim=-1)                                          # (B, K, 1, n)
        out = w @ self.states[li][:B, :n].unsqueeze(1)                             # (B, K, 1, d)
        return out.permute(0, 2, 1, 3).contiguous()                                # (B, 1, K, d)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits for the tokens x (B, T): the prompt when the cache is empty, else one new token."""
        self._layer = 0
        logits = self.m(x, position_offset=self.t)[0]
        if self._layer != self.L:
            raise RuntimeError(f"expected {self.L} context-channel calls per forward, saw {self._layer}")
        self.t += x.shape[1]
        return logits


class GPT2Decoder:
    """Key-value-cache decoding for the matched GPT-2 (colab_matched_gpt2_baseline_openwebtext.ipynb's class),
    with F.scaled_dot_product_attention: causal over the prompt, the new query against the cache after it."""

    def __init__(self, model, max_len: int, batch: int = 1):
        self.m = model
        b0 = model.blocks[0].attn
        H, dh = b0.n_heads, b0.d_head
        dt = model.tok_emb.weight.dtype
        self.k = [torch.zeros(batch, H, max_len, dh, dtype=dt) for _ in model.blocks]
        self.v = [torch.zeros(batch, H, max_len, dh, dtype=dt) for _ in model.blocks]
        self.max_len, self.t = max_len, 0

    def cache_numbers_per_token(self) -> int:
        return 2 * len(self.k) * self.k[0].shape[1] * self.k[0].shape[3]

    def _attn(self, a, x, li):
        B, T, C = x.shape
        q, k, v = a.qkv(x).reshape(B, T, 3, a.n_heads, a.d_head).permute(2, 0, 3, 1, 4)
        t0 = self.t
        self.k[li][:B, :, t0:t0 + T] = k
        self.v[li][:B, :, t0:t0 + T] = v
        if T > 1:
            if t0 != 0:
                raise NotImplementedError("a multi-token step is supported only as the prefill")
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, self.k[li][:B, :, :t0 + 1], self.v[li][:B, :, :t0 + 1])
        return a.proj(out.transpose(1, 2).reshape(B, T, C))

    @torch.no_grad()
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        m, (B, T) = self.m, idx.shape
        if self.t + T > self.max_len:
            raise RuntimeError(f"decoder cache full ({self.max_len} positions)")
        pos = torch.arange(self.t, self.t + T, device=idx.device).unsqueeze(0)
        x = m.tok_emb(idx) + m.pos_emb(pos)
        for li, blk in enumerate(m.blocks):
            x = x + self._attn(blk.attn, blk.ln1(x), li)
            x = x + blk.mlp(blk.ln2(x))
        x = m.ln_f(x)
        logits = x @ m.tok_emb.weight.T if m.tie_embeddings else m.lm_head(x)
        self.t += T
        return logits
