#!/usr/bin/env python3
"""
Examination of the one replicated subspace hit on coreference (sy_subspace.py,
2026-10-14): xi, coreference, top-16 principal components, layer 0, in PMX,
PM1, F3.1, G2 and SR2. Not pre-registered: it asks what the hit is made of.

xi is a running average that includes each token's own state, and many
coreferent pairs are the same word repeated. The hit could therefore be
lexical identity routed through xi's self-term rather than reference. Fixed
before running: the hit is LEXICAL if B_spec,xi in the top-16 subspace falls
below the floor when identical-token pairs are excluded (from related and
unrelated pairs alike) and on pronoun-antecedent pairs alone; it is
REFERENCE if it passes the floor, with the shuffled control null and above the
matched random subspace, on the pronoun-antecedent pairs.

Revision 2 (after PMX's first run, before any other arm): identical TOKEN ids are
the wrong unit, since GPT-2 spells one word with different ids (" Zhang" and
"Zhang"; "Trump" and "Trump's" share the word). A word-level variant excludes
pairs whose words match after lower-casing, stripping and removing a possessive
"'s". Fixed before the rerun: the hit is LEXICAL if it falls below the floor
without same-word pairs; if it survives, it is similarity between DIFFERENT words
referring to one entity, which is not reference resolution. First run of PMX:
sy_coref_subspace_check_PMX_v1_output.txt.

    python3 sy_coref_subspace_check.py ARM [n_seq=200]

Same subspace fit (layer 0 outputs of sequences 0-49), same measurement split
(50-199), same common-xi replay as sy_subspace.py; the full coreference
grouping is recomputed and must reproduce sy_subspace's value.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ARM = sys.argv[1]
N_SEQ = next((int(a[6:]) for a in sys.argv[2:] if a.startswith('n_seq=')), 200)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / 'debug'))   # shared model-building helpers (verify_pm_switch, ...)
from sy_replay import Replayer, CACHE
import sy_instruments as S
from sy_groupings import Groupings, bin_stats_rel

N_FIT, L0 = 50, 0
lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
R = Replayer(ARM)
m = R.model
TOK = np.load(CACHE / 'sy_tokens.npz'); X = TOK['x'][:N_SEQ]
ANN = np.load(CACHE / 'sy_annotation.npz')
GR = Groupings(ANN, X)
PRON = list(ANN['pos_tags']).index('PRON')
import json as _json
_TXT = _json.loads((CACHE / 'sy_annotation_text.json').read_text())
def words_of(i):
    t, sp = _TXT['texts'][i], _TXT['spans'][i]
    ws = ANN['word'][i]
    # the word a token belongs to: concatenate the token pieces of each spaCy word
    full = {}
    for k, (a, b) in enumerate(sp):
        if ws[k] >= 0: full.setdefault(int(ws[k]), []).append(t[a:b])
    norm = {w: ''.join(v).strip().lower().removesuffix("'s").removesuffix("’s") for w, v in full.items()}
    return np.array([norm.get(int(w), f'<none{k}>') if w >= 0 else f'<none{k}>' for k, w in enumerate(ws)], dtype=object)

fit = []
for b0 in range(0, N_FIT, 4):
    R.forward(torch.from_numpy(X[b0:min(b0 + 4, N_FIT)]))
    fit.append(R.records[L0]['out'].double().numpy().reshape(-1, m.cfg.d))
Hf = np.concatenate(fit); Hf -= Hf.mean(0)
Q = np.linalg.svd(Hf, full_matrices=False)[2][:16].T
Qr = np.linalg.qr(np.random.default_rng(7).normal(size=(m.cfg.d, 16)))[0]   # same draw order as sy_subspace's rand16
proj = lambda H, B: S.cos_matrix(S.centred(H @ B))

store = {}
put = lambda k, v: store.setdefault(k, []).append(v)
rng = np.random.default_rng(20261014)
n_pairs = {'all': 0, 'nonident': 0, 'nonword': 0, 'pron': 0}
for b0 in range(N_FIT, N_SEQ, 2):
    R.forward(torch.from_numpy(X[b0:b0 + 2]))
    on = R.replay(L0, None)[1]; com = R.replay(L0, ('xi', 'common'))[1]
    for j in range(on.shape[0]):
        i = b0 + j
        G, Gs = GR(i, rng)
        ids = X[i]; ident = ids[:, None] == ids[None, :]
        wd = words_of(i); sameword = wd[:, None] == wd[None, :]
        pron = ANN['pos'][i] == PRON
        onep = pron[:, None] ^ pron[None, :]
        variants = {}
        for tag, (rel, unr) in (('', G['coref']), ('shuf/', Gs['coref'])):
            variants[tag + 'all'] = (rel, unr)
            variants[tag + 'nonident'] = (rel & ~ident, unr & ~ident)
            variants[tag + 'nonword'] = (rel & ~sameword, unr & ~sameword)
            variants[tag + 'pron'] = (rel & ~ident & onep, unr & ~ident)
        if True:
            iu = np.triu_indices(len(ids), 1)
            for k in n_pairs: n_pairs[k] += int(variants[k][0][iu].sum())
        for B, bname in ((Q, 'pc16'), (Qr, 'rand16')):
            c_on, c_com = proj(on[j], B), proj(com[j], B)
            for v, (rel, unr) in variants.items():
                put(f'{bname}/{v}/on', bin_stats_rel(c_on, rel, unr))
                put(f'{bname}/{v}/com', bin_stats_rel(c_com, rel, unr))
A = {k: np.asarray(v) for k, v in store.items()}
D = lambda b, v: S.boot_contrast_diff(A[f'{b}/{v}/on'], A[f'{b}/{v}/com'])

say(f'Coreference subspace hit, examined: {ARM}, layer 0, xi common-mode, top-16 PCs ({N_SEQ - N_FIT} sequences)')
say(f'   related coreference pairs: all {n_pairs["all"]}, identical tokens excluded {n_pairs["nonident"]}, same words excluded {n_pairs["nonword"]}, pronoun-antecedent {n_pairs["pron"]}')
out = {}
for v, label in (('all', 'all coreferent pairs (reproduces sy_subspace)'), ('nonident', 'identical-token pairs excluded'),
                 ('nonword', 'same-word pairs excluded'), ('pron', 'pronoun-antecedent pairs only')):
    b = D('pc16', v); sh = D('pc16', 'shuf/' + v); rnd = D('rand16', v)
    hit = b[1] > 0 and b[0] >= S.FLOOR and S.is_null(sh) and b[0] - rnd[0] >= S.FLOOR
    out[v] = {'B_spec': b, 'shuffled': sh, 'rand16': rnd, 'passes': hit}
    say(f'   {label:<46} B_spec {S.fmt(b)}  shuffled {S.fmt(sh)}  rand16 {rnd[0]:+.4f}  {"PASSES" if hit else "-"}')
verdict = ('REFERENCE' if out['pron']['passes'] else
           'LEXICAL' if not out['nonword']['passes'] else
           'DIFFERENT WORDS OF ONE ENTITY (survives without same-word pairs; not pronoun resolution)')
say(f'   => {verdict}')
(HERE / f'sy_coref_subspace_check_{ARM}_output.txt').write_text('\n'.join(lines) + '\n')
(HERE / f'sy_coref_subspace_check_{ARM}.json').write_text(json.dumps({'n_pairs': n_pairs, 'results': out, 'verdict': verdict}, default=float, indent=1))
