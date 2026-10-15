#!/usr/bin/env python3
"""
SY5: does extra depth over-synchronize the tokens? (protocol §5.22, "SY5 and SY6:
amendment and additions", fixed 2026-10-15 before this script was built.)

    python3 sy5_depth.py ARM [n_seq=200]      ARM in F3.1 SR2 G2 PM1 PMX L4F

Gate 2, exactly as Cell 6b-7 defines it: N steps at the trained step size with the
"hold" layer schedule, run through the cell's own _fom_stack (executed verbatim from
the notebook). Two-layer arms at N = 2 (trained), 3, 4; the L = 4 arm at N = 4, 6, 8.

On the final states (what the readout reads), per N:
  S_N        pairwise contraction against the trained N = L: mean over pairs of
             ln(distance at N = L / distance at N); > 0 when the tokens drew together
  erank      exp of the entropy of the normalized squared singular values of the
             centred final states, per sequence
  PPL        perplexity of the next-token prediction
  B          noun-chunk and head-dependent binding contrast (descriptive)
Over-synchronization holds in an arm if, from N = L to the first extra N, S is at least
0.05 with its interval excluding 0, or the effective rank falls by at least 20%, while
perplexity rises. SY-G2 is a HIT if this holds in PMX and in another two-layer arm.

Check first: at N = L the patched stack reproduces the trained logits bit for bit
(Cell 6b-7's gate 0).
"""
import ast, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ARM = sys.argv[1]
N_SEQ = next((int(a[6:]) for a in sys.argv[2:] if a.startswith('n_seq=')), 200)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sy_replay import ARMS, P, CACHE
import sy_instruments as S
from sy_groupings import Groupings, bin_stats_rel

lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
folder, repl, _, _ = ARMS[ARM]
model, g, _ = P.build(folder, repl)
model.to('cpu'); model.eval()
L, DT = model.cfg.L, model.cfg.dt
NS = (2, 3, 4) if L == 2 else (4, 6, 8)

nb = json.load(open(P.G.NB))
src = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('# == Cell 6b-7:')][0]
keep = [n for n in ast.parse(src).body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
exec(compile(ast.Module(body=keep, type_ignores=[]), 'Cell6b7_defs', 'exec'), g)
_fom_stack = g['_fom_stack']

calls = []
orig_step = model._fock_layer_step
def spy(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0, **kw):
    out = orig_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx, **kw)
    calls.append((layer_idx, out[0].detach()))
    return out
model._fock_layer_step = spy

TOK = np.load(CACHE / 'sy_tokens.npz'); X = TOK['x'][:N_SEQ]; Y = TOK['y'][:N_SEQ]
GR = Groupings(np.load(CACHE / 'sy_annotation.npz'), X)

def run(xb, N):
    calls.clear()
    with _fom_stack(model, N, DT, 'hold'), torch.enable_grad():
        logits, _ = model(xb)
    return logits.detach(), calls[-1][1].double().numpy(), [c[0] for c in calls]

def plain(xb):
    calls.clear()
    with torch.enable_grad():
        logits, _ = model(xb)
    return logits.detach()

say(f'SY5, Gate 2 axis: {ARM}  ...{folder[-60:]}  (L = {L}, trained dt = {DT:g}, N in {NS})')
xb = torch.from_numpy(X[:2])
ref = plain(xb)
lg, _, sched = run(xb, L)
d = float((lg - ref).abs().max())
say(f'CHECK  N = L through Cell 6b-7\'s _fom_stack vs the trained forward: max |d logits| {d:.1e}; layer schedule {sched[:L]}')
assert d == 0.0
for N in NS[1:]:
    _, _, sched = run(xb, N)
    say(f'       N = {N}: layer schedule {sched[:N]}')

def erank(H):
    Hc = H - H.mean(0)
    sv = np.linalg.svd(Hc, compute_uv=False) ** 2
    p = sv / sv.sum(); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))

store = {}
def put(k, v): store.setdefault(k, []).append(v)
rng = np.random.default_rng(20261015)
t0 = time.time()
for b0 in range(0, N_SEQ, 2):
    xb = torch.from_numpy(X[b0:b0 + 2]); yb = torch.from_numpy(Y[b0:b0 + 2])
    res = {N: run(xb, N) for N in NS}
    for j in range(xb.shape[0]):
        i = b0 + j
        G, _ = GR(i, rng)
        H_L = res[L][1][j]
        for N in NS:
            lg, H = res[N][0][j], res[N][1][j]
            put(f'N{N}/nll', float(F.cross_entropy(lg.float(), yb[j], reduction='mean')))
            put(f'N{N}/erank', erank(H))
            if N != L:
                put(f'N{N}/S', S.coherence(S.contraction_matrix(H, H_L)))
            cos = S.cos_matrix(S.centred(H))
            for name in ('chunk', 'dep'):
                put(f'N{N}/B/{name}', bin_stats_rel(cos, *G[name]))
    if (b0 // 2) % 20 == 0:
        print(f'   {b0 + 2}/{N_SEQ} sequences, {(time.time() - t0) / 60:.1f} min', flush=True)
model._fock_layer_step = orig_step
A = {k: np.asarray(v) for k, v in store.items()}
np.savez_compressed(HERE / f'sy5_depth_{ARM}.npz', **A)

def boot_ratio(a, b, seed=0):
    """mean(a)/mean(b) - 1 with a cluster bootstrap over sequences."""
    I = np.random.default_rng(seed).integers(0, len(a), (S.N_BOOT, len(a)))
    bs = a[I].mean(1) / b[I].mean(1) - 1
    return float(a.mean() / b.mean() - 1), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

say()
say(f'RESULTS ({N_SEQ} sequences; 95% cluster-bootstrap intervals)')
say(f'   {"N":>3}  {"PPL":>9}  {"S vs N=L":>30}  {"eff. rank":>9}  {"rank change":>30}  {"chunk B":>9}  {"dep B":>9}')
out = {}
for N in NS:
    ppl = float(np.exp(A[f'N{N}/nll'].mean()))
    s_ = S.boot_mean(A[f'N{N}/S']) if N != L else (0.0, 0.0, 0.0)
    rk = boot_ratio(A[f'N{N}/erank'], A[f'N{L}/erank'])
    bc = S.boot_contrast(A[f'N{N}/B/chunk']); bd = S.boot_contrast(A[f'N{N}/B/dep'])
    out[N] = {'ppl': ppl, 'S': s_, 'erank': float(A[f'N{N}/erank'].mean()), 'rank_change': rk, 'B_chunk': bc, 'B_dep': bd}
    say(f'   {N:>3}  {ppl:>9.2f}  {S.fmt(s_):>30}  {A[f"N{N}/erank"].mean():>9.2f}  {S.fmt(rk):>30}  {bc[0]:>+9.4f}  {bd[0]:>+9.4f}')
N1 = NS[1]
s1, r1 = out[N1]['S'], out[N1]['rank_change']
over = (s1[1] > 0 and s1[0] >= 0.05) or (r1[0] <= -0.20)
rises = out[N1]['ppl'] > out[L]['ppl']
say(f'   from N = {L} to N = {N1}: perplexity {out[L]["ppl"]:.2f} -> {out[N1]["ppl"]:.2f} ({100 * (out[N1]["ppl"] / out[L]["ppl"] - 1):+.0f}%); '
    f'S {s1[0]:+.4f} (criterion >= 0.05); effective rank {100 * r1[0]:+.1f}% (criterion <= -20%)')
say(f'   => over-synchronization {"HOLDS" if (over and rises) else "does not hold"} in this arm')
(HERE / f'sy5_depth_{ARM}_output.txt').write_text('\n'.join(lines) + '\n')
(HERE / f'sy5_depth_{ARM}.json').write_text(json.dumps({'NS': NS, 'results': {str(k): v for k, v in out.items()}, 'over': bool(over and rises)}, indent=1))
print(f'wrote sy5_depth_{ARM}_output.txt')
