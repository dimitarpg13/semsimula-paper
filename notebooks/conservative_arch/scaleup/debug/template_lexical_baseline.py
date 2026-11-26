#!/usr/bin/env python3
"""Lexical baseline for the template matchers (2026-11-26; written before any result of it was seen).
    python3 template_lexical_baseline.py RUNDIR OUTDIR stem[,stem...]
RUNDIR holds depth_<stem>.pkl of template_depth_test.py. The question: how much of the embedding matcher's score is string
matching? The matcher is unchanged (template_depth_analysis.py: relational coordinates, connective position, clouds, the 1% rule,
the same search for m and d, "unless" left out). Only the particle geometry is replaced:
  lex       the cosine of two particles is 1 if they are the same token and 0 otherwise, every length is 1, and the connective's
            vector is the mean one-hot vector of its tokens (vocabulary: the tokens of the template connectives; an ordinary-text
            connective outside it is the zero vector). No learned vector enters.
  lexn      as lex, with tokens compared after stripping the leading space and lower-casing ("The" = " the")
  lex+c     lex, with the connective's vector taken from the model's embedding (its own whitened h0): string identity for the
            clauses, the learned word vector for the connective only
  h0        the embedding matcher, recomputed so the differences are paired clause by clause
The token ids are rebuilt by the item construction of template_depth_test.py (deterministic) and checked against the saved texts.
Expectations, stated before the run:
  L1  on known wordings lex detects MP, MT, DS1 and DS2 at a mean within 0.05 of the embedding's on every model, and fires on at
      most 2% of the fallacies (the valid forms are string patterns)
  L2  on unseen wordings (R1, without "unless") lex is below the embedding by at least 0.05 on every model (a connective with no
      token in common with a fitted one has no position on the axis)
  L3  lex+c is within 0.02 of the embedding on R1 on every model (the embedding's R1 is string identity plus the connective's word
      vector)
  L4  lex catches at most 0.05 of the paraphrases at 1% near misses, and its AUROC of paraphrase against near miss is at most 0.60
Writes OUTDIR/template_lexical_<stem>.json and prints a table."""
import json, pickle, sys
from pathlib import Path
import numpy as np
DBG = Path(__file__).resolve().parent; sys.path.insert(0, str(DBG))
RUN_, OUT_, STEMS_ = sys.argv[1:4]
_S = (DBG / 'template_depth_analysis.py').read_text(); _S = _S[:_S.rindex("if __name__ == '__main__':")]; exec(_S)
RUN, OUT = Path(RUN_), Path(OUT_)
# rebuild the items (token ids and roles) exactly as the extraction did
_T = (DBG / 'template_depth_test.py').read_text(); _T = _T[_T.index('from template_robust_clauses import DEV'):_T.index("if 'limit' in OPTS")]
OPTS = {}; exec(_T)
ITEMS = items


def particle_ids(it):
    return [t for ro in ROLES for t, q_ in zip(it['ids'], it['role']) if q_ == ro]


def lexical(ids_list, norm_tok, cvocab, Cemb=None):
    out = []
    for n_, ids in enumerate(ids_list):
        ids, ro = ids
        k = np.array([norm_tok(t) for t in ids], object); G = (k[:, None] == k[None]).astype(np.float64); c = np.zeros(len(cvocab))
        cs = [cvocab[x] for x, r in zip(k, ro) if r == 'c' and x in cvocab]
        for j in cs: c[j] += 1 / max(1, sum(r == 'c' for r in ro))
        out.append(dict(G=G, norm=np.ones(len(ids)), c=c if Cemb is None else Cemb[n_]))
    return out


res_all = {}
for s in STEMS_.split(','):
    M = pickle.load(open(RUN / f'depth_{s}.pkl', 'rb')); meta, res = M['meta'], M['res']; R = [x['roles'] for x in res]
    assert len(meta) == len(ITEMS) and all(a['text'] == b['text'] and a['kind'] == b['kind'] for a, b in zip(meta, ITEMS)), 'rebuilt items differ'
    ids_list = [(particle_ids(it), R[i]) for i, it in enumerate(ITEMS)]
    assert all(len(a) == len(r) for a, r in ids_list), 'particle count differs'
    K = np.array([x['kind'] for x in meta]); want = np.array([x['want'] for x in meta], object); wi = np.array([x['wording'] for x in meta]); sp = np.array([x['split'] for x in meta]); fr = np.array([x['frame'] for x in meta]); g = np.array([x['content'] for x in meta])
    lab = np.array(['text' if k == 'text' else f'{f}:{k}' for k, f in zip(K, fr)]); L_meta = (K, want, wi, sp, fr, g, lab)
    raw = lambda t: t; nrm = lambda t: enc.decode([int(t)]).strip().lower()
    P = {'h0': [x['st']['h0'] for x in res]}; Ch0 = [x['st']['h0']['c'] for x in res]
    for name, f in (('lex', raw), ('lexn', nrm)):
        voc = sorted({f(t) for (ids, ro), k_ in zip(ids_list, K) if k_ != 'text' for t, r in zip(ids, ro) if r == 'c'}, key=str); cv = {v: j for j, v in enumerate(voc)}
        P[name] = lexical(ids_list, f, cv)
        if name == 'lex': P['lex+c'] = lexical(ids_list, f, cv, Ch0)
    del M, res
    q, vec = {}, {}
    for name in ('h0', 'lex', 'lexn', 'lex+c'):
        q[name], vec[name] = score(name, P, (name,), R, L_meta); fx = (q[name]['dev'], q[name]['m'], q[name]['d'])
        q[name + '|12'], vec[name + '|12'] = score(name, P, (name,), R, L_meta, fx, drop=None)
        p = q[name]; print(f"{s} {name:6s} m{p['m']} d{p['d']:2d} dev {p['dev']:.3f} | R1 {p['R1 unseen']:.3f} (12 wordings {q[name + '|12']['R1 unseen']:.3f}) | known det " + ' '.join(f'{v:.2f}' for v in p['known detect'].values())
              + ' fires ' + ' '.join(f'{v:.2f}' for v in p['known fires on'].values()) + f" | unseen det " + ' '.join(f'{v:.2f}' for v in p['unseen detect'].values()) + ' fires ' + ' '.join(f'{v:.2f}' for v in p['unseen fires on'].values())
              + f" | para 1% {p['para at near-miss rate']['0.01']:.3f} AUROC {p['AUROC para/near']:.3f} | by wording unseen " + ' '.join(f'{w}:{v:.2f}' for w, v in p['unseen by wording'].items()), flush=True)
        if name != 'h0':
            q[name]['minus h0'] = {k: boot(vec[name][k] - vec['h0'][k]) for k in ('r1', 'r2', *(f'p{a_}' for a_ in RATES))}
            q[name + '|12']['minus h0'] = {k: boot(vec[name + '|12'][k] - vec['h0|12'][k]) for k in ('r1', 'r2')}
            print(f"   minus h0: R1 {q[name]['minus h0']['r1']}  para 1% {q[name]['minus h0']['p0.01']}", flush=True)
    res_all[s] = q; (OUT / f'template_lexical_{s}.json').write_text(json.dumps(q, indent=1))
