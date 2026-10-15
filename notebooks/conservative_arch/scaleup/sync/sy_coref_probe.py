#!/usr/bin/env python3
"""
SY decoding probe: does a later mention carry information about its
antecedent, whether or not their states become alike? (protocol §5.22;
criteria fixed 2026-10-14 before the first run.)

    python3 sy_coref_probe.py ARM [n_seq=200]      ARM in F3.1 SR2 G2 PM1 PMX L4F

Every SY state measure asks whether related tokens become MORE ALIKE. A
transformer binds differently: attention copies information from the
antecedent into the later token without making the two states similar. This
probe tests for that kind of binding.

Task. For each pronoun mention (its last token) that fastcoref links to an
earlier non-pronoun mention of the same chain, rank candidates: the last token
of the nearest such antecedent mention, and the last tokens of up to 7
preceding mentions of OTHER chains (items with fewer than 2 distractors are
dropped).

Probe. score(m, c) = (U h_m) . (V h_c), U, V in R^{32 x d} (one attention
head's query-key form), softmax over the candidates, trained on 150 sequences
(Adam, 400 steps, weight decay 1e-3), tested on the other 50; one probe per
layer state: the embedding (layer-0 input) and every layer's output.

Baselines: chance (1 / candidates), distance only (pick the nearest candidate),
and the probe on the embeddings, which can learn purely lexical compatibility
("she" with names) but nothing the dynamics adds.

Reading. The dynamics MAKE binding information if the final-layer probe's test
accuracy exceeds the embedding probe's by at least 5 points, with the 95%
bootstrap interval over test sequences excluding 0, and exceeds the distance
baseline. Replicated if this holds in PMX and in at least one other arm.

Coupling ablation (added 2026-10-14 with the subspace test, before any probe run):
the probe trained on the trained model's last-layer states is applied, on the
test sequences, to last-layer states from replays of that step with each
coupling replaced by its common-mode reference (the one sy_measure adopted),
switched off, and with xi given per token alone ("own") or as the sequence
average ("common"). The readout RELIES on a coupling if the accuracy falls by
at least 5 points under its common-mode reference, with the 95% interval
excluding 0 (meaningful only where the dynamics make binding information).

The final-layer probe's subspace (the rows of U and V, mapped back from the
standardised coordinates; 64 directions) is saved for sy_subspace.py.

Output beside this script: sy_coref_probe_<ARM>_output.txt, .json, _subspace.npz.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

ARM = sys.argv[1]
N_SEQ = next((int(a[6:]) for a in sys.argv[2:] if a.startswith('n_seq=')), 200)
HERE = Path(__file__).resolve().parent
CACHE = HERE / 'sy_cache'
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / 'debug'))   # shared model-building helpers (verify_pm_switch, ...)

from sy_replay import Replayer
RANK, STEPS, WD, MAX_DIST = 32, 400, 1e-3, 7
N_TRAIN = 150 if N_SEQ >= 200 else int(0.75 * N_SEQ)        # smaller only in smoke tests
lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))

ANN = np.load(CACHE / 'sy_annotation.npz')
X = np.load(CACHE / 'sy_tokens.npz')['x'][:N_SEQ]
PRON = list(ANN['pos_tags']).index('PRON')


def items_for(i):
    """[(m, [candidates], index of the true antecedent in the list)] for sequence i."""
    mn = ANN['mention'][i]; cr = ANN['coref'][i]; pos = ANN['pos'][i]
    ments = {}
    for t in range(len(mn)):
        if mn[t] >= 0:
            ments.setdefault(int(mn[t]), []).append(t)
    M = sorted((max(ts), min(ts), int(cr[ts[0]]), all(pos[u] == PRON for u in ts if pos[u] >= 0)) for ts in ments.values())
    out = []
    for k, (last, first, ch, is_pron) in enumerate(M):
        if not is_pron:
            continue
        prev = M[:k]
        ante = [p for p in prev if p[2] == ch and not p[3]]
        if not ante:
            continue
        a = ante[-1][0]
        distr = [p[0] for p in prev if p[2] != ch][-MAX_DIST:]
        if len(distr) < 2:
            continue
        cands = sorted(distr + [a])
        out.append((last, cands, cands.index(a)))
    return out


ITEMS = [items_for(i) for i in range(N_SEQ)]
n_items = sum(len(x) for x in ITEMS)

R = Replayer(ARM)
L = R.L
say(f'SY coreference decoding probe: {ARM} (L = {L}); {n_items} pronoun items in {N_SEQ} sequences')
R.check(torch.from_numpy(X[:2]), say)
STATES = {k: [] for k in ['emb'] + list(range(L))}
t0 = time.time()
for b0 in range(0, N_SEQ, 4):
    R.forward(torch.from_numpy(X[b0:b0 + 4]))
    STATES['emb'].append(R.records[0]['args'][0].float())
    for l in range(L):
        STATES[l].append(R.records[l]['out'].float())
STATES = {k: torch.cat(v) for k, v in STATES.items()}
say(f'   states captured in {(time.time() - t0) / 60:.1f} min')

rng = np.random.default_rng(20261014)
perm = rng.permutation(N_SEQ); TR, TE = perm[:N_TRAIN], perm[N_TRAIN:]


def batch(H, seqs):
    Q, C, mask, y, sid = [], [], [], [], []
    for i in seqs:
        for (m, cands, a) in ITEMS[i]:
            Q.append(H[i, m]); C.append(H[i, cands]); y.append(a); sid.append(i)
    K = max(len(c) for c in C)
    Cp = torch.zeros(len(C), K, H.shape[-1]); Mk = torch.zeros(len(C), K, dtype=torch.bool)
    for j, c in enumerate(C):
        Cp[j, :len(c)] = c; Mk[j, :len(c)] = True
    return torch.stack(Q), Cp, Mk, torch.tensor(y), np.array(sid)


def fit_eval(H):
    torch.manual_seed(0)
    q, c, mk, y, _ = batch(H, TR)
    mu, sd = H.reshape(-1, H.shape[-1]).mean(0), H.reshape(-1, H.shape[-1]).std(0) + 1e-6
    nz = lambda z: (z - mu) / sd
    d = H.shape[-1]
    U = torch.nn.Parameter(torch.randn(RANK, d) / d ** 0.5); V = torch.nn.Parameter(torch.randn(RANK, d) / d ** 0.5)
    opt = torch.optim.Adam([U, V], lr=3e-3, weight_decay=WD)
    for _ in range(STEPS):
        s = torch.einsum('nr,nkr->nk', nz(q) @ U.T, nz(c) @ V.T).masked_fill(~mk, -1e9)
        loss = torch.nn.functional.cross_entropy(s, y)
        opt.zero_grad(); loss.backward(); opt.step()
    probe = (U.detach(), V.detach(), mu, sd)
    return evaluate(probe, H), probe


def evaluate(probe, H):
    U, V, mu, sd = probe
    q, c, mk, y, sid = batch(H, TE)
    with torch.no_grad():
        s = torch.einsum('nr,nkr->nk', ((q - mu) / sd) @ U.T, ((c - mu) / sd) @ V.T).masked_fill(~mk, -1e9)
    return (s.argmax(1) == y).numpy().astype(float), sid


def by_seq(correct, sid):
    seqs = np.unique(sid)
    return seqs, np.array([correct[sid == s].sum() for s in seqs]), np.array([(sid == s).sum() for s in seqs])


def boot_diff(c1, c0, sid, n=1000):
    seqs = np.unique(sid); I = np.random.default_rng(0).integers(0, len(seqs), (n, len(seqs)))
    a1 = np.array([c1[sid == s].sum() for s in seqs]); a0 = np.array([c0[sid == s].sum() for s in seqs])
    nn = np.array([(sid == s).sum() for s in seqs])
    b = np.array([(a1[ii].sum() - a0[ii].sum()) / nn[ii].sum() for ii in I])
    return float((a1.sum() - a0.sum()) / nn.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


_, _, _, y_te, sid_te = batch(STATES['emb'], TE)
cand_te = [cands for i in TE for (_, cands, _) in ITEMS[i]]
nearest = np.array([float(len(c) - 1 == a) for c, a in zip(cand_te, y_te.numpy())])      # nearest = last candidate
chance = float(np.mean([1.0 / len(c) for c in cand_te]))
say(f'   test items {len(y_te)} in {len(np.unique(sid_te))} sequences; chance {chance:.3f}; distance only (nearest) {nearest.mean():.3f}')
acc = {}
probes = {}
for k in ['emb'] + list(range(L)):
    (corr, sid), probes[k] = fit_eval(STATES[k])
    acc[k] = corr
    say(f'   probe on {"the embeddings" if k == "emb" else f"layer {k} output":<16} test accuracy {corr.mean():.3f}')
d_emb = boot_diff(acc[L - 1], acc['emb'], sid_te)
d_near = boot_diff(acc[L - 1], nearest, sid_te)
made = d_emb[0] >= 0.05 and d_emb[1] > 0 and d_near[1] > 0
say(f'   final layer minus embeddings {d_emb[0]:+.3f} [{d_emb[1]:+.3f}, {d_emb[2]:+.3f}];  minus distance-only {d_near[0]:+.3f} [{d_near[1]:+.3f}, {d_near[2]:+.3f}]')
say(f'   => the dynamics {"MAKE" if made else "do not make"} decodable binding information in this arm')

# ---- the final-layer probe's subspace, for sy_subspace.py
U, V, mu, sd = probes[L - 1]
basis = torch.cat([U / sd, V / sd]).double().numpy().T                 # d x 64
np.savez(HERE / f'sy_coref_probe_{ARM}_subspace.npz', basis=basis)

# ---- coupling ablation: the trained probe on last-layer states from replays (test sequences)
say()
say('ABLATION: the final-layer probe (trained on the trained model) applied to the last step replayed under each change')
versions = [(c, kind) for c in R.couplings for kind in ('common', 'off')] + [('xi', 'own'), ('xi', 'common')]
ABL = {v: STATES[L - 1].clone() for v in versions}
for b0 in range(0, N_SEQ, 2):
    seqs = [i for i in range(b0, min(b0 + 2, N_SEQ)) if i in set(TE.tolist())]
    if not seqs:
        continue
    R.forward(torch.from_numpy(X[b0:b0 + 2]))
    on = R.replay(L - 1, None)
    for (c, kind) in versions:
        if c == 'xi':
            post = R.replay(L - 1, ('xi', kind))[1]
        elif kind == 'common':
            post = R.common_ref(L - 1, c, on)[1]
        else:
            post = R.replay(L - 1, ('off', c))[1]
        for i in seqs:
            ABL[(c, kind)][i] = torch.from_numpy(post[i - b0]).float()
abl = {}
for v in versions:
    corr, _ = evaluate(probes[L - 1], ABL[v])
    dd = boot_diff(acc[L - 1], corr, sid_te)
    abl[f'{v[0]}/{v[1]}'] = {'acc': float(corr.mean()), 'drop': dd}
    say(f'   {v[0]:<5} {v[1]:<7} accuracy {corr.mean():.3f}   drop {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}]'
        + ('   RELIES' if (v[1] == 'common' and dd[0] >= 0.05 and dd[1] > 0) else ''))
(HERE / f'sy_coref_probe_{ARM}_output.txt').write_text('\n'.join(lines) + '\n')
(HERE / f'sy_coref_probe_{ARM}.json').write_text(json.dumps({
    'n_items': n_items, 'n_test': int(len(y_te)), 'chance': chance, 'nearest': float(nearest.mean()),
    'acc': {str(k): float(v.mean()) for k, v in acc.items()}, 'final_minus_emb': d_emb, 'final_minus_nearest': d_near, 'made': made,
    'ablation': abl}, indent=1))
print(f'wrote sy_coref_probe_{ARM}_output.txt')
