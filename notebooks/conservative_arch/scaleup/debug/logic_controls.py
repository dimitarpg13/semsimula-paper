#!/usr/bin/env python3
"""Controls for logic_figures.py (2026-11-25), added after its first numbers were read.

    python3 logic_controls.py OUT FOLDER [name=<stem>] [threads=8]

  NR   modus ponens' frame with four unrelated clauses ("If P_i, then Q_j. P_k. Therefore, Q_l."): the connectives
       without any recurrence
  TXT  ordinary OpenWebText validation tokens in the token layout of the 40 modus ponens arguments
  Node states (h0, h1, h2) and dominant wells of both, saved next to the forms'.
  Reaction: after "... Therefore," each of the four clauses P, Q, not P, not Q is scored. The gain of a clause is its
  loss per token after "<passage> Therefore," with no argument, minus its loss after the argument: what the argument
  adds, with the clause's own difficulty and length removed.
"""
import sys
from pathlib import Path
src = (Path(__file__).parent / 'logic_figures.py').read_text()
src = src[:src.index("\nt0 = time.time()")]
exec(compile(src, 'logic_figures.py', 'exec'))
import json
n = len(C); rng = np.random.default_rng(1)
def run(ids):
    with torch.enable_grad():
        return m(torch.tensor(ids))[0].detach().float()
def nodes_of(ids, role):
    rec.clear(); cur['on'] = True; run([ids]); cur['on'] = False
    ix = [[i for i, r in enumerate(role) if r == ro] for ro in ROLES]
    return ([np.stack([rec[('h', l)][0][0][j].mean(0).numpy() for j in ix]) for l in (0, 1, 2)],
            [[int(torch.mode(rec[('k', l)][0][0][j]).values) for j in ix] for l in (0, 1)])
save, wells = {}, {}
for kind in ('NR', 'TXT'):
    H, K = [], []
    for i in range(n):
        if kind == 'NR':
            o = (i + np.array([0, 7, 14, 21])) % n
            segs = segments('MP', C[o[0]][0], C[o[1]][1], '', '')
            segs = [(" " + cap(C[o[2]][0]), r) if r == 'p2' else (" " + C[o[3]][1], r) if r == 'cn' else (t, r) for t, r in segs]
            ids, role = list(prefix), [None] * len(prefix)
            for t, r in segs:
                e = enc.encode(t); ids += e; role += [r] * len(e)
        else:
            it = items[i]; s = 1_200_000 + 600 * i
            ids = list(prefix) + [int(v) for v in val[s:s + len(it['ids']) - len(prefix)]]; role = it['role']
        a, b = nodes_of(ids, role); H.append(a); K.append(b)
    for l in (0, 1, 2):
        save[f'nodes_h{l}_{kind}'] = np.stack([h[l] for h in H])
    wells[kind] = [[round(float((np.array([k[l] for k in K]) == v).mean()), 3) for v in range(8)] for l in (0, 1)]
# reaction on four candidates
NAMES = ['P', 'Q', 'notP', 'notQ']
def score(ctx, cands):
    L = max(len(ctx) + len(c) for c in cands)
    lo = run([ctx + c + [50256] * (L - len(ctx) - len(c)) for c in cands]); out = []
    for j, c in enumerate(cands):
        lp = torch.log_softmax(lo[j, len(ctx) - 1:len(ctx) + len(c) - 1], -1)
        out.append(float(-lp[torch.arange(len(c)), torch.tensor(c)].mean()))
    return out
cand = [[enc.encode(" " + c) for c in row] for row in C]
base = np.array([score(list(prefix) + enc.encode("Therefore,"), cand[i]) for i in range(n)])
gain = {}
for f in FORMS:
    G_ = []
    for i in range(n):
        it = next(t for t in items if t['form'] == f and t['content'] == i)
        G_.append(base[i] - np.array(score(it['ids'][:it['cn'][0]], cand[i])))
    gain[f] = np.array(G_)
out = {'wells': wells, 'gain': {}, 'model': NAME}
print(NAME, 'gain per candidate (nats per token; se), and how often each is the largest gain')
for f, g_ in gain.items():
    top = np.bincount(g_.argmax(1), minlength=4)
    print(f'  {f}: ' + '  '.join(f'{c} {g_[:, j].mean():+.2f} ({g_[:, j].std(ddof=1) / n ** .5:.2f}) top {top[j]}' for j, c in enumerate(NAMES)))
    out['gain'][f] = dict(mean=g_.mean(0).tolist(), se=(g_.std(0, ddof=1) / n ** .5).tolist(), top=top.tolist(), raw=g_.tolist())
print('  dominant-well shares over nodes, per layer:', wells)
(OUT / f'logic_controls_{NAME}.json').write_text(json.dumps(out, indent=1)); np.savez_compressed(OUT / f'logic_controls_{NAME}.npz', **save)
