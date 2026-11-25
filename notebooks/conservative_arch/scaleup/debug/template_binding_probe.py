#!/usr/bin/env python3
"""Which reading of premise 2 tells an exact repetition from a near miss, and a paraphrase from a near miss? (2026-11-25,
design probe for the robust template matcher; exploratory)
    python3 template_binding_probe.py OUT FOLDER name=<stem>
Token states of the required clause and of premise 2, on the 60 test clauses, two conditional and one disjunctive wording.
Readings (larger = more like a restatement), at h0, h1, h2:
  mean      cosine of the mean states (centred) -- what the first matcher used
  white     the same after whitening with the token covariance of ordinary text (shared directions damped)
  last      cosine of the states at the two clauses' last tokens
  maxsim    each premise-2 token's best cosine with a clause token, averaged both ways (whitened)
  worst     the worst such match over premise-2 and clause tokens, negation words and articles excluded (whitened)
and from the model's predictions:
  gain      loss of premise 2 after a premise 1 in which the required clause is replaced by an unrelated one, minus its
            loss after the true premise 1 (nats per token): what the required clause contributes
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
src = (Path(__file__).parent / 'template_matcher_test.py').read_text(); a, b = src.index("base = list(items)"), src.index("# model\n")
OPT_OUT, argv = Path(sys.argv[1]).resolve(), list(sys.argv)
exec(compile(src[:a], 'tm', 'exec')); pre = [int(v) for v in val[1_000_000:1_000_048]] + enc.encode("\n\n")
code = src[b:]; exec(compile(code[:code.index("rec = {}; os_ =")], 'tm', 'exec'))
rec = {}; os_ = m._layer_step_langevin
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    o = os_(h, h_prev, m_b, gamma, dt_, layer_idx)
    if int(layer_idx) == 0: rec[0] = h.detach().float()
    rec[int(layer_idx) + 1] = o[0].detach().float(); return o
m._layer_step_langevin = step
def run(seqs):
    L = len(pre) + max(map(len, seqs)); x = torch.tensor([pre + s + [50256] * (L - len(pre) - len(s)) for s in seqs])
    with torch.enable_grad(): lo = m(x)[0].detach().float()
    return lo, {l: rec[l].clone() for l in rec}
# whitening from ordinary text
ws = torch.tensor(np.stack([val[300_000 + 700 * i:300_000 + 700 * i + 160] for i in range(64)])); Wt = {}
H = {0: [], 1: [], 2: []}
for i in range(0, 64, 16):
    with torch.enable_grad(): m(ws[i:i + 16])
    for l in H: H[l].append(rec[l][:, 32:].reshape(-1, 384).numpy())
for l in H:
    x = np.concatenate(H[l]); mu = x.mean(0); e, U = np.linalg.eigh(np.cov((x - mu).T)); e = np.maximum(e, 0)
    Wt[l] = (mu, U / np.sqrt(e + .1 * e.mean()))
SKIP = set(' the| a| an| not| does| do| has| is| are| no| it| true| that| have| been|The|It|A|The| did'.split('|')) | {' The', ' It', ' A'}
KS = ['c:MP', 'c:near+', 'c:paraMP', 'c:unrel+', 'c:MT', 'c:near-', 'c:paraMT', 'c:unrel-', 'c:notMT', 'c:DA', 'd:DS1', 'd:near-', 'd:paraDS', 'd:unrel-', 'd:notDS']
sel = [it for it in items if it['split'] == 'test' and it['wording'] in ('if-then', 'q-if-p', 'either-or') and it['kind'] in KS]
REQ = lambda k: 'a2' if k in ('c:MT', 'c:near-', 'c:paraMT', 'c:unrel-', 'c:notMT') else 'a1'
def unrel(it):
    """premise 1 with the required clause replaced by one that appears nowhere in the argument"""
    r = REQ(it['kind']); key = 'P' if r == 'a1' else 'Q'; c = next(t['aff'] for t in TEST[(it['content'] * 7 + 11) % 60:] + TEST if t['aff'] not in (it['P'], it['Q']) and t['aff'] not in it['p2'])
    head, rest = it['text'].split('.', 1); old = it[key]
    h2 = head.replace(old, c) if old in head else head.replace(cap(old), cap(c)); ids = enc.encode(h2 + '.') ; p2 = enc.encode(' ' + cap(it['p2']))
    return ids, p2
def cs(a, b): return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
R = {}; t0 = time.time()
for b_ in range(0, len(sel), 16):
    B = sel[b_:b_ + 16]; lo, st = run([i['ids'] for i in B]); alt = [unrel(i) for i in B]; lo2, _ = run([i + p for i, p in alt]) 
    for r, it in enumerate(B):
        P2 = [j for j, q in enumerate(it['role']) if q == 'p2']; CLm = [j for j, q in enumerate(it['role']) if q == REQ(it['kind'])]; out = {}
        ce = lambda L, pos, ids: float(np.mean([-torch.log_softmax(L[len(pre) + j - 1], -1)[t].item() for j, t in zip(pos, ids)]))
        i2, p2 = alt[r]; out['gain'] = ce(lo2[r], range(len(i2), len(i2) + len(p2)), p2) - ce(lo[r], P2, [it['ids'][j] for j in P2])
        keep = lambda J: [j for j in J if enc.decode([it['ids'][j]]) not in SKIP] or J
        for l in (0, 1, 2):
            s = st[l][r].numpy(); mu, W_ = Wt[l]; z = lambda J: s[[len(pre) + j for j in J]] - mu; w = lambda J: z(J) @ W_
            out[f'h{l} mean'] = cs(z(P2).mean(0), z(CLm).mean(0)); out[f'h{l} white'] = cs(w(P2).mean(0), w(CLm).mean(0)); out[f'h{l} last'] = cs(z(P2[-1:])[0], z(CLm[-1:])[0])
            u, v = w(keep(P2)), w(keep(CLm)); u /= np.linalg.norm(u, axis=1, keepdims=True); v /= np.linalg.norm(v, axis=1, keepdims=True); M = u @ v.T
            out[f'h{l} maxsim'] = float((M.max(1).mean() + M.max(0).mean()) / 2); out[f'h{l} worst'] = float(min(M.max(1).min(), M.max(0).min()))
        R.setdefault(it['kind'], []).append(dict(content=it['content'], wording=it['wording'], **out))
    if b_ % 320 == 0: print(b_, len(sel), f'{time.time() - t0:.0f}s', flush=True)
OPT_OUT.mkdir(parents=True, exist_ok=True); (OPT_OUT / f'binding_probe_{OPTS["name"]}.json').write_text(json.dumps(R))
au = lambda a, b: (lambda r: float((r[:len(a)].sum() - len(a) * (len(a) - 1) / 2) / (len(a) * len(b))))(np.argsort(np.argsort(np.r_[a, b] + 1e-9 * np.arange(len(a) + len(b)))))
T = [('exact / near', [('c:MP', 'c:near+'), ('c:MT', 'c:near-'), ('d:DS1', 'd:near-')]), ('para / near', [('c:paraMP', 'c:near+'), ('c:paraMT', 'c:near-'), ('d:paraDS', 'd:near-')]),
     ('para / unrel', [('c:paraMP', 'c:unrel+'), ('c:paraMT', 'c:unrel-'), ('d:paraDS', 'd:unrel-')]), ('exact / unrel', [('c:MP', 'c:unrel+'), ('c:MT', 'c:unrel-'), ('d:DS1', 'd:unrel-')]), ('reworded denial / near', [('c:notMT', 'c:near-'), ('d:notDS', 'd:near-')])]
print('AUROC: ' + ' | '.join(n for n, _ in T))
for f in [k for k in R['c:MP'][0] if k not in ('content', 'wording')]:
    print(f'{f:11s} ' + ' | '.join(' '.join(f'{au(np.array([x[f] for x in R[a]]), np.array([x[f] for x in R[b]])):.2f}' for a, b in ps) for _, ps in T))
