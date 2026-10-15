#!/usr/bin/env python3
"""
SY6 addition: the coreference decoding probe on the matched GPT-2 (protocol §5.22,
fixed 2026-10-15 before this script was built). Task, probe, split and criterion are
sy_coref_probe.py's, with its code reused verbatim: rank each pronoun's true antecedent
against up to 7 preceding mentions of other chains with a rank-32 bilinear probe,
trained on 150 sequences and tested on 50. The dynamics make decodable binding
information if the final states' probe beats the embedding probe by at least 5 points,
with the interval excluding 0, and beats the nearest-mention baseline.

    python3 sy6_gpt2_probe.py
"""
import json, sys, time, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

N_SEQ = 200
HERE = Path(__file__).resolve().parent
CACHE = HERE / 'sy_cache'
lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
ANN = np.load(CACHE / 'sy_annotation.npz')
X = np.load(CACHE / 'sy_tokens.npz')['x'][:N_SEQ]
PRON = list(ANN['pos_tags']).index('PRON')
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

NB = HERE.parent / 'colab_matched_gpt2_baseline_openwebtext.ipynb'
CKPT = Path.home() / 'Downloads/semsimula_matched_gpt2_baseline_owt/checkpoints/gpt2_baseline_best.pt'
nb = json.load(open(NB))
cell3 = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).lstrip().startswith('# ── Cell 3')][0]
g = {'torch': torch, 'nn': nn, 'F': F, 'math': math, '__name__': 'cell3'}
exec(compile(cell3, 'Cell3', 'exec'), g)
ck = torch.load(CKPT, map_location='cpu', weights_only=False); cfg = ck['config']
model = g['GPT2'](cfg['vocab_size'], cfg['d'], cfg['n_layers'], cfg['n_heads'], cfg['d_ff'], cfg['max_len'],
                  dropout=0.0, tie_embeddings=cfg['tie_embeddings'])
model.load_state_dict(ck['model_state_dict'], strict=True); model.eval()
L = len(model.blocks)
say(f'SY6 coreference probe on the matched GPT-2 ({L} layers); {n_items} pronoun items in {N_SEQ} sequences')

KEYS = ['emb'] + list(range(L)) + ['final']
STATES = {k: [] for k in KEYS}
with torch.no_grad():
    for b0 in range(0, N_SEQ, 4):
        xb = torch.from_numpy(X[b0:b0 + 4]); T = xb.shape[1]
        x = model.tok_emb(xb) + model.pos_emb(torch.arange(T).unsqueeze(0))
        STATES['emb'].append(x.float())
        for l, blk in enumerate(model.blocks):
            x = blk(x); STATES[l].append(x.float())
        STATES['final'].append(model.ln_f(x).float())
STATES = {k: torch.cat(v) for k, v in STATES.items()}
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
nearest = np.array([float(len(c) - 1 == a) for c, a in zip(cand_te, y_te.numpy())])
chance = float(np.mean([1.0 / len(c) for c in cand_te]))
say(f'   test items {len(y_te)} in {len(np.unique(sid_te))} sequences; chance {chance:.3f}; distance only (nearest) {nearest.mean():.3f}')
acc = {}
for k in KEYS:
    (corr, sid), _ = fit_eval(STATES[k])
    acc[k] = corr
    say(f'   probe on {"the embeddings" if k == "emb" else ("the final states" if k == "final" else f"layer {k} output"):<18} test accuracy {corr.mean():.3f}')
d_emb = boot_diff(acc['final'], acc['emb'], sid_te); d_near = boot_diff(acc['final'], nearest, sid_te)
best = max(KEYS[1:], key=lambda k: acc[k].mean())
d_best = boot_diff(acc[best], nearest, sid_te)
made = d_emb[0] >= 0.05 and d_emb[1] > 0 and d_near[1] > 0
say(f'   final minus embeddings {d_emb[0]:+.3f} [{d_emb[1]:+.3f}, {d_emb[2]:+.3f}];  final minus distance-only {d_near[0]:+.3f} [{d_near[1]:+.3f}, {d_near[2]:+.3f}]')
say(f'   best layer ({best}) minus distance-only {d_best[0]:+.3f} [{d_best[1]:+.3f}, {d_best[2]:+.3f}] (descriptive)')
say(f'   => the dynamics {"MAKE" if made else "do not make"} decodable binding information in GPT-2')
(HERE / 'sy6_gpt2_probe_output.txt').write_text('\n'.join(lines) + '\n')
(HERE / 'sy6_gpt2_probe.json').write_text(json.dumps({'n_items': n_items, 'chance': chance, 'nearest': float(nearest.mean()),
    'acc': {str(k): float(v.mean()) for k, v in acc.items()}, 'final_minus_emb': d_emb, 'final_minus_nearest': d_near,
    'best_layer': str(best), 'best_minus_nearest': d_best, 'made': made}, indent=1))
