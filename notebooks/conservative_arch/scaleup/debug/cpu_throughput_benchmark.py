#!/usr/bin/env python3
"""CPU inference throughput: the conservative model against the matched GPT-2, and where the conservative model's
time goes (2026-11-17; the author's question: does the closed-form CfC + BAOAB design pay on a CPU?).

    python3 cpu_throughput_benchmark.py OUT MODEL FOLDER [lengths=512,2048,8192] [reps=3] [threads=8] [profile=512,2048]
                                        [name=<stem>]

MODEL is `ladder` (FOLDER a ladder run in ~/Downloads, built from its tag as the probes build it), `gpt2` (the matched
GPT-2 notebook's own class and checkpoint, naive attention as trained) or `gpt2_sdpa` (the same weights with
F.scaled_dot_product_attention, the fused kernel a deployed transformer would use). One sequence per forward, float32,
both under no_grad (the ladder model's analytic V_phi force needs no graph; grad=1 times it under enable_grad,
as the training notebook's evaluate() runs it). Timing only: past the trained context, the position tables are extended by tiling (the speed does
not depend on those values; the outputs there mean nothing). Per length: median wall time over `reps` forwards after
one warm-up, tokens per second, and the process's peak resident memory so far (lengths run in increasing order).
`profile=` lengths get one profiled forward (ladder only): the top operators by self CPU time and a summary by kind.
Writes OUT/cpu_bench_<stem>.txt and .json.
"""
import json, resource, sys, time
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve(); MODEL = sys.argv[2]; FOLDER = sys.argv[3]
OPTS = dict(a.split('=', 1) for a in sys.argv[4:] if '=' in a)
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))
LENGTHS = [int(t) for t in OPTS.get('lengths', '512,2048,8192').split(',')]
REPS = int(OPTS.get('reps', 3)); THREADS = int(OPTS.get('threads', 8))
PROF = [int(t) for t in OPTS.get('profile', '512,2048').split(',') if t] if MODEL == 'ladder' else []
TMAX = max(LENGTHS)

src = (DBG / 'copy_split.py').read_text()
src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', 'windows=1', f'threads={THREADS}']
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)
torch.set_num_threads(THREADS)


def tile(w, n):
    reps = -(-n // w.shape[0])
    return w.detach().repeat(reps, 1)[:n].clone()


if MODEL == 'ladder':
    import verify_pm_switch as P
    _build = P.build; got = {}
    def build(*a, **k):
        r = _build(*a, **k); got['model'] = r[0]; return r
    P.build = build
    _, desc = CS['ladder_model'](FOLDER, OPTS.get('ckpt'))
    m = got['model']
    if m.P.shape[0] < TMAX:
        m.P = torch.nn.Parameter(tile(m.P, TMAX)); m.cfg.max_len = TMAX
        if hasattr(m, 'xi_module'):
            m.xi_module.max_len = TMAX
    # 2026-11-17: the tag-based build leaves VPHI_ANALYTIC_FORCE at its notebook default (False), and the autograd
    # V_phi force silently drops out under no_grad (loss +0.016 nats on SR2). analytic=1 (default) restores the
    # trained configuration. chunk=<n> sets the inference-only query chunking of V_phi and xi; skipreg=1 skips the
    # unread Fock registers (debug/verify_inference_chunking.py).
    m.cfg.vphi_analytic_force = OPTS.get('analytic', '1') == '1'
    if OPTS.get('chunk'):
        m.cfg.vphi_query_chunk = int(OPTS['chunk']); m.xi_module.query_chunk = int(OPTS['chunk'])
    m.cfg.skip_unread_registers = OPTS.get('skipreg', '0') == '1'
    desc += (f"; analytic V_phi {m.cfg.vphi_analytic_force}; query chunk {OPTS.get('chunk', 'off')}; "
             f"skip unread registers {m.cfg.skip_unread_registers}")
    if OPTS.get('novphi', '0') == '1':
        # 2026-11-17: V_phi removed entirely (no score network, top-k, router or pair force): the upper bound on
        # what dropping or replacing it saves. Timing only; the loss reading below shows what it costs as trained.
        assert m.cfg.vphi_analytic_force, 'novphi replaces the analytic pair force'
        m._pair_force_analytic = lambda h_in, layer_idx, h_src_live=None, q_slice=None: torch.zeros_like(h_in)
        desc += '; V_phi removed'
    GRAD = OPTS.get('grad', '0') == '1'
    def run(x):                             # (verify_vphi_analytic_output.txt: no_grad, max |dlogit| 0)
        with (torch.enable_grad() if GRAD else torch.no_grad()):
            return m(x)[0]
else:
    import contextlib, io
    nb = DBG.parent / 'colab_matched_gpt2_baseline_openwebtext.ipynb'
    cells = [''.join(c['source']) for c in json.load(open(nb))['cells']]
    ck = torch.load(Path.home() / 'Downloads' / FOLDER / 'checkpoints' / 'gpt2_baseline_best.pt', map_location='cpu',
                    weights_only=False)
    g = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec('import math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np', g)
        exec(cells[1], g); exec(cells[4], g)
    m = g['GPT2'](vocab_size=g['VOCAB_SIZE'], d=g['D_MODEL'], n_layers=g['N_LAYERS'], n_heads=g['N_HEADS'],
                  d_ff=g['D_FF'], max_len=g['MAX_LEN'], dropout=0.0, tie_embeddings=g['TIE_EMBEDDINGS'])
    m.load_state_dict(ck['model_state_dict']); m.eval()
    desc = f"gpt2_baseline_best.pt (step {ck['step']}, val {ck['val_ppl']:.3f}); L {g['N_LAYERS']}, d {g['D_MODEL']}"
    if m.pos_emb.weight.shape[0] < TMAX:
        pe = torch.nn.Embedding(TMAX, m.pos_emb.weight.shape[1]); pe.weight.data = tile(m.pos_emb.weight, TMAX)
        m.pos_emb = pe
        mask = torch.tril(torch.ones(TMAX, TMAX)).view(1, 1, TMAX, TMAX)
        for b in m.blocks:
            b.attn.mask = mask                      # one shared buffer
    if MODEL == 'gpt2_sdpa':
        F = torch.nn.functional
        def fwd(self, x):
            B, T, C = x.shape
            q, k, v = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
            return self.proj(out)
        for b in m.blocks:
            b.attn.forward = fwd.__get__(b.attn)
        desc += '; attention by F.scaled_dot_product_attention'
    def run(x):
        with torch.no_grad():
            return m(x)[0]

m.eval()
n_par = sum(p.numel() for p in m.parameters())
print('model:', desc, f'{n_par:,} parameters', flush=True)
rng = np.random.default_rng(0)
res = {'model': MODEL, 'folder': FOLDER, 'desc': desc, 'params': n_par, 'threads': THREADS, 'lengths': {}}
lines = [f'CPU inference benchmark: {MODEL}  {FOLDER}', f'{desc}; {n_par:,} parameters; {THREADS} threads; float32; '
         'one sequence per forward', '', f'{"length":>6}  {"median s":>9}  {"tokens/s":>9}  {"peak RSS GB":>11}  times']
for T in LENGTHS:
    x = torch.from_numpy(rng.integers(0, 50257, size=(1, T)).astype(np.int64))
    run(x)                                           # warm-up
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); out = run(x); ts.append(time.perf_counter() - t0)
        assert torch.isfinite(out).all()
    med = float(np.median(ts)); rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30   # macOS: bytes
    res['lengths'][T] = {'times': ts, 'median_s': med, 'tokens_per_s': T / med, 'peak_rss_gb': rss}
    lines.append(f'{T:>6}  {med:9.3f}  {T / med:9.1f}  {rss:11.2f}  ' + ' '.join(f'{t:.3f}' for t in ts))
    print(lines[-1], flush=True)

if OPTS.get('loss'):
    # 2026-11-17: the validation loss of the configuration as timed, on evenly spaced windows of 512 tokens
    vv = np.load(OPTS['loss']).astype(np.int64); nw = int(OPTS.get('loss_windows', 16))
    stl = np.linspace(0, len(vv) - 513, nw).astype(np.int64)
    xyl = torch.from_numpy(np.stack([vv[s_:s_ + 513] for s_ in stl]))
    ces = []
    for j in range(0, nw, 4):
        lgl = run(xyl[j:j + 4, :512])
        ces.append(torch.nn.functional.cross_entropy(lgl.reshape(-1, lgl.shape[-1]).float(), xyl[j:j + 4, 1:].reshape(-1)))
    lv = float(torch.stack(ces).mean())
    res['val_loss'] = lv
    lines += ['', f'validation loss on {nw} windows of 512 ({Path(OPTS["loss"]).name}): {lv:.4f} (ppl {np.exp(lv):.2f})']
    print(lines[-1], flush=True)

KINDS = [('decompositions (svd, eigh, qr, cholesky, solve)', ('svd', 'eigh', 'qr', 'cholesky', 'solve', 'lu')),
         ('backward (autograd)', ('backward',)),
         ('matrix products (mm, bmm, matmul, einsum, addmm, linear)', ('mm', 'bmm', 'matmul', 'einsum', 'linear', 'addmm', 'baddbmm')),
         ('softmax', ('softmax',)), ('top-k, sort, gather, scatter', ('topk', 'sort', 'gather', 'scatter', 'index'))]
for T in PROF:
    x = torch.from_numpy(rng.integers(0, 50257, size=(1, T)).astype(np.int64))
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        run(x)
    ka = prof.key_averages()
    tot = sum(e.self_cpu_time_total for e in ka)
    by = {k: 0.0 for k, _ in KINDS}; other = 0.0
    for e in ka:
        nm = e.key.lower()
        for k, pats in KINDS:
            if any(p in nm for p in pats):
                by[k] += e.self_cpu_time_total; break
        else:
            other += e.self_cpu_time_total
    by['other (elementwise, copies, reductions)'] = other
    res.setdefault('profile', {})[T] = {k: v / tot for k, v in by.items()}
    lines += ['', f'profile at length {T}: self CPU time {tot / 1e6:.2f} s summed over threads and operators'] + \
             [f'  {100 * v / tot:5.1f}%  {k}' for k, v in by.items()] + ['', '  top operators by self CPU time:'] + \
             ['  ' + l for l in ka.table(sort_by='self_cpu_time_total', row_limit=20).split('\n')]
stem = OPTS.get('name', MODEL)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f'cpu_bench_{stem}.txt').write_text('\n'.join(lines) + '\n')
(OUT / f'cpu_bench_{stem}.json').write_text(json.dumps(res, indent=1, default=str))
print('\n'.join(lines[-40:]))
