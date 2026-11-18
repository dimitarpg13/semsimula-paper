#!/usr/bin/env python3
"""The content-routed context channels alone: can a fused attention kernel compute them faster on the CPU, with the
same result? (2026-11-17; lever N1 of companion_notes/productization/Conservative_Core_Productization_Plan.md.)

    python3 xi_sdpa_microbench.py OUT FOLDER [lengths=512,2048,8192] [reps=3] [threads=8] [chunk=512] [hnorm=1]

The channels compute, per channel k, softmax_s((t - s) log a_k + q_t . k_s / sqrt(d_k)) h_s over s <= t: attention
with a distance bias and the states as values (MultiChannelXi.forward). FOLDER's checkpoint supplies the trained decays
and query and key maps; the states are standard Gaussian, scaled by hnorm= (timing does not depend on them; the
agreement figures are a first check, not the model-level one). Variants, each against `naive`:

  naive          the trained code path: the (K, T, T) bias and (B, K, T, T) scores in full
  chunk          the inference-only switch already in the model: blocks of `chunk` query rows
  sdpa_mask      F.scaled_dot_product_attention with the bias as an additive float mask, in full
  sdpa_mask_ch   the same, a block of query rows at a time
  sdpa_fold      the bias folded into the keys: (t - s) log a = t log a - s log a, and the row constant t log a
                 cancels in the softmax, so -s log a is a per-key term: one extra query component sqrt(d_k) and one
                 extra key component -s log a; then plain causal attention (is_causal=True), no mask
  sdpa_fold_h48  sdpa_fold with the value width split into heads of 48, so query, key and value widths match

Per variant and length: median seconds over `reps` after one warm-up, max |difference| from naive (lengths up to 2,048;
naive's tensors at 8,192 are timed but not kept), and the process's peak resident memory. Writes OUT/xi_sdpa_microbench.txt.
"""
import resource, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
LENGTHS = [int(t) for t in OPTS.get('lengths', '512,2048,8192').split(',')]
REPS = int(OPTS.get('reps', 3)); CH = int(OPTS.get('chunk', 512))
torch.set_num_threads(int(OPTS.get('threads', 8)))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'multixi'))
from model_multixi import causal_ema_logits, causal_ema_logits_rows      # noqa: E402

ckd = Path.home() / 'Downloads' / FOLDER / 'checkpoints'
ck = torch.load(next(ckd.glob('*_best.pt')), map_location='cpu', weights_only=False)['model_state_dict']
Wq, Wk = ck['xi_module.W_q.weight'], ck['xi_module.W_k.weight']
alphas = torch.sigmoid(ck['xi_module.raw_alpha'])
K = alphas.numel(); d = Wq.shape[1]; dk = Wq.shape[0] // K
log_a = torch.log(alphas.clamp(1e-6, 1 - 1e-6))


def qkv(h):
    B, T, _ = h.shape
    q = (h @ Wq.T).view(B, T, K, dk).permute(0, 2, 1, 3)
    k = (h @ Wk.T).view(B, T, K, dk).permute(0, 2, 1, 3)
    return q, k


def naive(h):
    q, k = qkv(h); T = h.shape[1]
    base = torch.stack([causal_ema_logits(T, alphas[j], h.dtype, h.device) for j in range(K)])
    w = torch.softmax(base.unsqueeze(0) + (q @ k.transpose(-1, -2)) * dk ** -0.5, dim=-1)
    return w @ h.unsqueeze(1)                                          # (B, K, T, d)


def chunk(h):
    q, k = qkv(h); T = h.shape[1]; hs = h.unsqueeze(1); out = []
    for t0 in range(0, T, CH):
        t1 = min(t0 + CH, T)
        bc = torch.stack([causal_ema_logits_rows(t0, t1, T, alphas[j], h.dtype, h.device) for j in range(K)])
        out.append(torch.softmax(bc.unsqueeze(0) + (q[:, :, t0:t1] @ k.transpose(-1, -2)) * dk ** -0.5, -1) @ hs)
    return torch.cat(out, 2)


def sdpa_mask(h):
    q, k = qkv(h); T = h.shape[1]
    base = torch.stack([causal_ema_logits(T, alphas[j], h.dtype, h.device) for j in range(K)])
    v = h.unsqueeze(1).expand(-1, K, -1, -1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=base.unsqueeze(0), scale=dk ** -0.5)


def sdpa_mask_ch(h):
    q, k = qkv(h); T = h.shape[1]; v = h.unsqueeze(1).expand(-1, K, -1, -1); out = []
    for t0 in range(0, T, CH):
        t1 = min(t0 + CH, T)
        bc = torch.stack([causal_ema_logits_rows(t0, t1, t1, alphas[j], h.dtype, h.device) for j in range(K)])
        out.append(F.scaled_dot_product_attention(q[:, :, t0:t1], k[:, :, :t1], v[:, :, :t1],
                                                  attn_mask=bc.unsqueeze(0), scale=dk ** -0.5))
    return torch.cat(out, 2)


def _fold(h):
    q, k = qkv(h); B, T, _ = h.shape
    s = torch.arange(T, dtype=h.dtype)
    qe = torch.full((B, K, T, 1), dk ** 0.5, dtype=h.dtype)
    ke = (-s.view(1, 1, T, 1) * log_a.view(1, K, 1, 1)).expand(B, K, T, 1)
    return torch.cat([q, qe], -1), torch.cat([k, ke], -1)


def sdpa_fold(h):
    q2, k2 = _fold(h)
    v = h.unsqueeze(1).expand(-1, K, -1, -1)
    return F.scaled_dot_product_attention(q2, k2, v, is_causal=True, scale=dk ** -0.5)


def sdpa_fold_h48(h):
    q2, k2 = _fold(h); B, T, _ = h.shape; nh = d // dk
    v = h.view(B, T, nh, dk).permute(0, 2, 1, 3)                       # (B, nh, T, dk)
    v = v.unsqueeze(1).expand(B, K, nh, T, dk).reshape(B, K * nh, T, dk)
    qq = q2.unsqueeze(2).expand(B, K, nh, T, dk + 1).reshape(B, K * nh, T, dk + 1)
    kk = k2.unsqueeze(2).expand(B, K, nh, T, dk + 1).reshape(B, K * nh, T, dk + 1)
    o = F.scaled_dot_product_attention(qq, kk, v, is_causal=True, scale=dk ** -0.5)   # (B, K*nh, T, dk)
    return o.view(B, K, nh, T, dk).permute(0, 1, 3, 2, 4).reshape(B, K, T, d)


VARIANTS = [naive, chunk, sdpa_mask, sdpa_mask_ch, sdpa_fold, sdpa_fold_h48]
torch.manual_seed(0)
lines = [f'context channels alone, {FOLDER}', f'K = {K} channels, key width {dk}, state width {d}; decays '
         + ', '.join(f'{a:.4f}' for a in alphas.tolist()) + f'; torch {torch.__version__}, {torch.get_num_threads()} '
         f'threads, float32, one sequence; chunk {CH}', '',
         f'{"length":>6}  {"variant":<14} {"median s":>9}  {"vs naive":>8}  {"max |diff|":>10}  {"peak RSS GB":>11}']
print('\n'.join(lines), flush=True)
with torch.no_grad():
    for T in LENGTHS:
        h = torch.randn(1, T, d) * float(OPTS.get('hnorm', 1.0))
        ref = naive(h) if T <= 2048 else None
        tn = None
        for f in VARIANTS:
            try:
                f(h)
                ts = []
                for _ in range(REPS):
                    t0 = time.perf_counter(); o = f(h); ts.append(time.perf_counter() - t0)
                med = float(np.median(ts)); tn = tn or med
                dif = f'{(o - ref).abs().max().item():10.2e}' if ref is not None else f'{"":>10}'
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
                lines.append(f'{T:>6}  {f.__name__:<14} {med:9.4f}  {tn / med:7.2f}x  {dif}  {rss:11.2f}')
            except Exception as e:                                       # a kernel the build lacks
                lines.append(f'{T:>6}  {f.__name__:<14} failed: {type(e).__name__}: {str(e)[:80]}')
            print(lines[-1], flush=True)
            o = None
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'xi_sdpa_microbench.txt').write_text('\n'.join(lines) + '\n')
