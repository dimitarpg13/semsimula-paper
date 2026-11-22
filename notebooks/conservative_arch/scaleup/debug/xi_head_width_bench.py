#!/usr/bin/env python3
"""The folded context channels as heads of equal width: time, memory and error (2026-11-22, for
companion_notes/productization/Stage1b_Context_Channels_as_30_Heads.md).

    python3 xi_head_width_bench.py OUT FOLDER [lengths=256,512,1024,2048,4096,8192] [reps=5] [threads=8]
                                   [errlengths=512,1024,2048,4096,8192]

The channels compute, per channel k, softmax_s((t - s) log a_k + q_t . k_s / sqrt(d_k)) h_s over s <= t
(MultiChannelXi.forward). FOLDER's checkpoint supplies the trained decays and query and key maps; the states are
standard Gaussian. Subnormals are flushed (lever N6) unless a variant's name ends in `_noflush`.

Time and memory. Each variant runs in its own process, lengths ascending, so the peak resident memory read after a
length is that length's. Variants:

  fold        the folded form of lever N1: 5 heads, queries and keys of width d_k + 1 = 49, values of width 384
  eq<p>       queries and keys zero-padded to p, values split into 384 / p blocks of width p: 5 * 384 / p heads of
              width p (p = 64 is xi=fold64 of cpu_throughput_benchmark.py)
  gpt2layer   one layer of the matched GPT-2's attention for scale: 6 heads of width 64, random inputs
  fold_noflush, eq64_noflush   as fold and eq64, subnormals not flushed

Error (`errlengths`). In one process, against the same channels computed in float64 by the trained code path: the
trained code path in float32 (naive), fold, eq64, and eq64c (eq64 with the key term centred, -(s - s0) log a with
s0 = (T - 1) / 2, a constant that cancels in the softmax). Per channel, the largest absolute difference.

Writes OUT/xi_head_width_bench.txt and .json.
"""
import json, resource, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
LENGTHS = [int(t) for t in OPTS.get('lengths', '256,512,1024,2048,4096,8192').split(',')]
ERRLENGTHS = [int(t) for t in OPTS.get('errlengths', '512,1024,2048,4096,8192').split(',')]
REPS = int(OPTS.get('reps', 5))
torch.set_num_threads(int(OPTS.get('threads', 8)))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'multixi'))
from model_multixi import causal_ema_logits      # noqa: E402

ckd = Path.home() / 'Downloads' / FOLDER / 'checkpoints'
ck = torch.load(next(ckd.glob('*_best.pt')), map_location='cpu', weights_only=False)['model_state_dict']
Wq, Wk = ck['xi_module.W_q.weight'], ck['xi_module.W_k.weight']
alphas = torch.sigmoid(ck['xi_module.raw_alpha'])
K = alphas.numel(); d = Wq.shape[1]; dk = Wq.shape[0] // K
log_a = torch.log(alphas.clamp(1e-6, 1 - 1e-6))


def qk(h):
    B, T, _ = h.shape
    q = (h @ Wq.T.to(h.dtype)).view(B, T, K, dk).permute(0, 2, 1, 3)
    k = (h @ Wk.T.to(h.dtype)).view(B, T, K, dk).permute(0, 2, 1, 3)
    return q, k


def naive(h):
    q, k = qk(h); T = h.shape[1]
    base = torch.stack([causal_ema_logits(T, alphas[j].to(h.dtype), h.dtype, h.device) for j in range(K)])
    w = torch.softmax(base.unsqueeze(0) + (q @ k.transpose(-1, -2)) * dk ** -0.5, dim=-1)
    return w @ h.unsqueeze(1)                                          # (B, K, T, d)


def folded(h, width, s0=0.0):
    """Queries and keys with the distance term folded in, zero-padded to `width`."""
    q, k = qk(h); B, T, _ = h.shape
    s = torch.arange(T, dtype=h.dtype) - s0
    qe = torch.full((B, K, T, 1), dk ** 0.5, dtype=h.dtype)
    ke = (-s.view(1, 1, T, 1) * log_a.to(h.dtype).view(1, K, 1, 1)).expand(B, K, T, 1)
    pad = torch.zeros((B, K, T, width - dk - 1), dtype=h.dtype)
    return torch.cat([q, qe, pad], -1), torch.cat([k, ke, pad], -1)


def fold(h):
    q, k = folded(h, dk + 1)
    return F.scaled_dot_product_attention(q, k, h.unsqueeze(1).expand(-1, K, -1, -1), is_causal=True,
                                          scale=dk ** -0.5)


def eq(h, p, s0=0.0):
    B, T, _ = h.shape; nb = d // p
    q, k = folded(h, p, s0)
    q = q.unsqueeze(2).expand(B, K, nb, T, p).reshape(B, K * nb, T, p)
    k = k.unsqueeze(2).expand(B, K, nb, T, p).reshape(B, K * nb, T, p)
    v = h.view(B, T, nb, p).permute(0, 2, 1, 3).unsqueeze(1).expand(B, K, nb, T, p).reshape(B, K * nb, T, p)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=dk ** -0.5)
    return o.view(B, K, nb, T, p).permute(0, 1, 3, 2, 4).reshape(B, K, T, d)


def gpt2layer(h):
    B, T, _ = h.shape
    q, k, v = (torch.randn(B, 6, T, 64) for _ in range(3))
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def make(name):
    if name.startswith('eq'):
        p = int(name[2:].split('_')[0])
        return lambda h: eq(h, p)
    return {'fold': fold, 'gpt2layer': gpt2layer}[name.split('_')[0]]


def worker(name):
    """One variant, lengths ascending; prints one JSON line per length."""
    if not name.endswith('_noflush'):
        torch.set_flush_denormal(True)
    f = make(name)
    torch.manual_seed(0)
    print(json.dumps({'variant': name, 'T': 0, 'peak_rss_gb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30}),
          flush=True)                                                   # before any call: the loaded checkpoint
    with torch.no_grad():
        for T in LENGTHS:
            h = torch.randn(1, T, d)
            f(h)
            ts = []
            for _ in range(REPS):
                t0 = time.perf_counter(); f(h); ts.append(time.perf_counter() - t0)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30   # bytes on macOS
            print(json.dumps({'variant': name, 'T': T, 'median_s': float(np.median(ts)), 'min_s': min(ts),
                              'peak_rss_gb': rss}), flush=True)


def errors():
    torch.set_flush_denormal(True)
    torch.manual_seed(1)
    rows = []
    with torch.no_grad():
        for T in ERRLENGTHS:
            h = torch.randn(1, T, d)
            ref = naive(h.double())
            outs = {'naive': naive(h), 'fold': fold(h), 'eq64': eq(h, 64), 'eq64c': eq(h, 64, (T - 1) / 2)}
            for name, o in outs.items():
                e = (o.double() - ref).abs().amax(dim=(0, 2, 3))           # per channel
                rows.append({'variant': name, 'T': T, 'max_abs_per_channel': e.tolist()})
                print(json.dumps(rows[-1]), flush=True)
            ref = outs = None
    return rows


if __name__ == '__main__':
    if OPTS.get('worker'):
        worker(OPTS['worker'])
        sys.exit()
    OUT.mkdir(parents=True, exist_ok=True)
    variants = ['fold', 'eq64', 'eq96', 'eq128', 'eq192', 'eq384', 'gpt2layer', 'fold_noflush', 'eq64_noflush']
    timing = []
    for v in variants:
        p = subprocess.run([sys.executable, __file__, str(OUT), FOLDER, f'worker={v}']
                           + [a for a in sys.argv[3:] if not a.startswith('worker=')],
                           capture_output=True, text=True)
        for ln in p.stdout.splitlines():
            if ln.startswith('{'):
                timing.append(json.loads(ln)); print(ln, flush=True)
        if p.returncode:
            print(v, 'failed:', p.stderr[-400:], flush=True)
    err = errors()

    lines = [f'context channels as heads of equal width, {FOLDER}',
             f'K = {K} channels, key width {dk}, state width {d}; decays '
             + ', '.join(f'{a:.4f}' for a in alphas.tolist()) + f'; |log a| '
             + ', '.join(f'{-x:.4f}' for x in log_a.tolist())
             + f'; torch {torch.__version__}, {torch.get_num_threads()} threads, float32, one sequence', '',
             'median seconds per call (peak resident GB of the variant\'s process after that length)',
             f'{"variant":<14}' + ''.join(f'{T:>18}' for T in LENGTHS)]
    for v in variants:
        r = {x['T']: x for x in timing if x['variant'] == v}
        base = r.pop(0, {}).get('peak_rss_gb', float('nan'))
        lines.append(f'{v:<14}' + ''.join(f'{r[T]["median_s"]:>10.4f} ({r[T]["peak_rss_gb"]:4.2f})' if T in r
                                          else f'{"-":>18}' for T in LENGTHS) + f'   (before any call {base:.2f})')
    lines += ['', 'largest absolute difference from the float64 trained code path, per channel (decays as above)',
              f'{"variant":<8}{"T":>6}  ' + ''.join(f'{"ch" + str(j):>10}' for j in range(K))]
    for r in err:
        lines.append(f'{r["variant"]:<8}{r["T"]:>6}  ' + ''.join(f'{x:>10.2e}' for x in r['max_abs_per_channel']))
    (OUT / 'xi_head_width_bench.txt').write_text('\n'.join(lines) + '\n')
    (OUT / 'xi_head_width_bench.json').write_text(json.dumps(
        {'folder': FOLDER, 'K': K, 'd_k': dk, 'd': d, 'alphas': alphas.tolist(), 'log_alpha': log_a.tolist(),
         'torch': torch.__version__, 'timing': timing, 'errors': err}, indent=1))
    print('\n'.join(lines))
