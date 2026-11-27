#!/usr/bin/env python3
"""Test 2 scoring: the template matchers on the analogy and in-context premises (2026-11-26; written before any result of test 2).
    python3 template_analogy_analysis.py DEPTHDIR ANALOGYDIR OUTDIR stem[,stem...] [gates]
"gates" scores token identity only, on the primary development set, and checks the extraction (gates G1 to G3 of the calls),
so the item set can be checked before any model reading is computed; it writes template_analogy_gates_<stem>.json.
DEPTHDIR holds depth_<stem>.pkl (template_depth_test.py), ANALOGYDIR analogy_<stem>.pkl (template_analogy_test.py); OUTDIR holds
template_depth_<stem>.json and template_lexical_<stem>.json, whose primary m and d are reused (nothing is tuned here).
Design: companion_notes/semantic_templates/Analogy_and_Context_Test_Set.md; items: analogy_items/; calls: Analogy_Test_Calls.md.

The matcher is the depth analysis's, unchanged (relational coordinates, connective position, one cloud per kind, discriminant
subspace), and the clouds are fitted on development premises only, never on an analogy premise:
  original  (primary) the depth analysis's own development set: the 60 development clauses, verbatim kinds only (that set has no
            paraphrase kind), "unless" left out, ordinary text included (5,790 clause premises and 600 of text)
  clean     (secondary, a check on shared nouns) the development and earlier test premises of the depth extraction, verbatim kinds
            only, "unless" left out, and only premises none of whose clause subjects (P, Q, and the subject of the second sentence)
            has a head noun that occurs anywhere in the analogy items (404 development and 3,957 earlier test premises), and all
            1,200 premises of ordinary text. The clouds read relations between particles, never which word a particle is, so a
            shared noun gives them nothing to remember; this reading checks that.
Matchers: lex (token identity, template_lexical_baseline.py), h0 (the embedding), every h, D, W, and path, wells, states.
Readings per level, each a mean over the level's 20 clauses with a 95% interval from 4,000 resamples of the clauses:
  rank      the right template has the largest density among the four (valid kinds); chance 0.25, but a reader that sees only
            "not" already decides two of the four choices, so the floor that matters is lex's score
  foil      matched pairs (restatement against its foil, within clause and wording): the share of pairs in which the restatement
            has the larger log odds of the restatement's template (ties count half); chance 0.5
  twin      the same, a valid form against its fallacy twin (MP and AC, MT and DA, DS1 and the affirmed P, DS2 and the affirmed Q)
  foil C2'  C2 only: the paraphrase in the clause's sense against the false friend
Log odds of a template: log of the summed densities of its clouds over the summed densities of all other clouds.
Secondary: the same with the clean development set; detection and false fires at the 1% rule's threshold.
Differences to h0 and to lex are paired by clause."""
import json, pickle, re, sys
from pathlib import Path
import numpy as np
DBG = Path(__file__).resolve().parent; sys.path.insert(0, str(DBG))
GATES = 'gates' in sys.argv[5:]; DEPTH_, AN_, OUT_, STEMS_ = sys.argv[1:5]; sys.argv = [sys.argv[0], DEPTH_, OUT_, STEMS_]
_S = (DBG / 'template_depth_analysis.py').read_text(); _S = _S[:_S.rindex("if __name__ == '__main__':")]; exec(_S)
from template_analogy_items import PAIRS, LEVEL_ORDER
from template_robust_clauses import DEV
from template_fresh_clauses import FRESH
RUN, AN, OUT = Path(DEPTH_), Path(AN_), Path(OUT_)
_T = (DBG / 'template_depth_test.py').read_text(); _T = _T[_T.index('from template_robust_clauses import DEV'):_T.index("if 'limit' in OPTS")]
OPTS = {}; exec(_T); DITEMS = items                                     # the depth premises with token ids, for lex
T4 = ['MP', 'MT', 'DS1', 'DS2']; B_ = 4000; RNG = np.random.default_rng(3)
WC.clear(); WC.update({k: [w for w in v if w != 'unless'] for k, v in ALLW.items()})
lse = lambda L, m: np.logaddexp.reduce(np.where(m[None], L, -np.inf), axis=1)


def bootc(per):
    """per: (n_clauses,) -> [mean, lo, hi]"""
    i = RNG.integers(len(per), size=(B_, len(per))); b = per[i].mean(1); return [float(per.mean()), *[float(x) for x in np.percentile(b, [2.5, 97.5])]]


def clean_mask(meta):
    import template_analogy_clauses as A
    words = set(re.findall(r"[a-z']+", ' '.join(v for bank in A.LEVELS.values() for f in bank for v in f.values() if isinstance(v, str)).lower()))
    subj = sorted({x['subj'] for x in DEV + FRESH}, key=len, reverse=True)
    def ok(m):
        if m['kind'] == 'text': return True
        for s in (m['P'], m['Q'], m['p2'].lower()):
            h = next((x for x in subj if s.startswith(x)), None)
            if h is not None and h.split()[-1] in words: return False
        return True
    return np.array([ok(m) for m in meta])


def lexical(ids_list, cvocab):
    out = []
    for ids, ro in ids_list:
        k = np.array(ids); c = np.zeros(len(cvocab)); nc = max(1, sum(r == 'c' for r in ro))
        for x, r in zip(ids, ro):
            if r == 'c' and x in cvocab: c[cvocab[x]] += 1 / nc
        out.append(dict(G=(k[:, None] == k[None]).astype(np.float64), norm=np.ones(len(ids)), c=c))
    return out


def fit_score(Pd, Rd, Pt, Rt_, ks, m, d, dmeta, tmeta):
    """clouds on the development premises, read on the test premises: (log odds per template, rank template, fire at rho)"""
    K, want, wi, fr, g, lab = dmeta
    Cn = lambda P, k: (lambda C: C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9))(np.stack([x['c'] for x in P[k]]).astype(np.float64))
    Xd = np.concatenate([np.stack([coords(x, r, m) for x, r in zip(Pd[k], Rd)]) for k in ks], 1); Cd = [Cn(Pd, k) for k in ks]
    Xt = np.concatenate([np.stack([coords(x, r, m) for x, r in zip(Pt[k], Rt_)]) for k in ks], 1); Ct = [Cn(Pt, k) for k in ks]
    F_ = Frame(Cd, wi, fr); c = Clouds(np.c_[Xd, F_(Cd)], lab, want, d); rho = cvf(Xd, Cd, lab, want, g, wi, fr, d)[0]
    Y = np.c_[Xt, F_(Ct)]; Yp = ((Y - c.mu0) / c.sd) @ c.B; L = -.5 * ((Yp[:, None] - c.C[None]) ** 2).sum(2); tpl = np.array([c.tpl[x] for x in c.cl], object)
    S = np.stack([lse(L, tpl == t) for t in T4], 1); LO = np.stack([S[:, j] - lse(L, tpl != t) for j, t in enumerate(T4)], 1)
    return LO, np.array(T4, object)[S.argmax(1)], c.fire(Y, rho), rho


def readings(LO, top, fire, tm):
    lv, g, wi, K, want = tm; out, per = {}, {}
    for L_ in LEVEL_ORDER:
        q = lv == L_; out[L_], per[L_] = {}, {}
        v = q & (want != ''); per[L_]['rank'] = np.array([(top[v & (g == c)] == want[v & (g == c)]).mean() for c in range(20)])
        for rd in sorted({p[0] for p in PAIRS[L_]}):
            w = {c: [] for c in range(20)}
            for _, a, b, t in [p for p in PAIRS[L_] if p[0] == rd]:
                j = T4.index(t); ia = {(g[i], wi[i]): i for i in np.flatnonzero(q & (K == a))}; ib = {(g[i], wi[i]): i for i in np.flatnonzero(q & (K == b))}
                for key, i in ia.items():
                    x, y = LO[i, j], LO[ib[key], j]; w[key[0]].append(1. if x > y + 1e-9 else .5 if abs(x - y) <= 1e-9 else 0.)
            per[L_][rd] = np.array([np.mean(w[c]) for c in range(20)])
        for rd, x in per[L_].items(): out[L_][rd] = bootc(x)
        out[L_]['detect at rho'] = float((fire[v] == want[v]).mean()); nv = q & (want == ''); out[L_]['false fire at rho'] = float((fire[nv] != '').mean())
    return out, per


if __name__ == '__main__':
    allq = {}
    for s in STEMS:
        M = pickle.load(open(RUN / f'depth_{s}.pkl', 'rb')); dmeta_, dres, L_ = M['meta'], M['res'], M['L']; kinds = M['kinds']; del M
        A_ = pickle.load(open(AN / f'analogy_{s}.pkl', 'rb')); ameta, ares = A_['meta'], A_['res']; del A_
        assert len(dmeta_) == len(DITEMS) and all(a['text'] == b['text'] for a, b in zip(dmeta_, DITEMS)), 'rebuilt depth items differ'
        chk = {}
        for m_, r_ in zip(ameta, ares):                                  # the check premises against the stored depth premises
            if m_['split'] != 'check': continue
            r0 = dres[m_['depth_index']]; assert dmeta_[m_['depth_index']]['text'] == m_['text']
            for k in kinds: chk[k] = max(chk.get(k, 0.), float(np.abs(r_['st'][k]['G'].astype(float) - r0['st'][k]['G'].astype(float)).max()))
        q = allq[s] = dict(check_max_abs_cosine_diff=chk); assert max(chk.values()) < 1e-3, chk
        ti = [i for i, m_ in enumerate(ameta) if m_['split'] == 'analogy']; tmeta = [ameta[i] for i in ti]; tres = [ares[i] for i in ti]; del ares
        tm = tuple(np.array([m_[k] for m_ in tmeta], object if k == 'want' else None) for k in ('level', 'content', 'wording', 'kind', 'want'))
        K = np.array([x['kind'] for x in dmeta_]); want = np.array([x['want'] for x in dmeta_], object); wi = np.array([x['wording'] for x in dmeta_]); sp = np.array([x['split'] for x in dmeta_]); fr = np.array([x['frame'] for x in dmeta_])
        g = np.array([x['content'] + (100 if x['split'] == 'test' and x['kind'] != 'text' else 2000 if x['split'] == 'test' else 0) for x in dmeta_]); lab = np.array(['text' if k == 'text' else f'{f}:{k}' for k, f in zip(K, fr)])
        syn = np.isin(K, list(SYN_V) + list(SYN_F)); cm = clean_mask(dmeta_)
        DEVS = {'original': (sp == 'dev') & (wi != 'unless'), 'clean': (wi != 'unless') & ~syn & cm}
        q['development premises'] = {k: int(v.sum()) for k, v in DEVS.items()}; q['clean clauses'] = dict(dev=int(len(set(g[DEVS['clean'] & (sp == 'dev') & (K != 'text')]))), test=int(len(set(g[DEVS['clean'] & (sp == 'test') & (K != 'text')]))))
        print(s, 'check', {k: round(v, 5) for k, v in chk.items()}, '| development premises', q['development premises'], 'clean clauses', q['clean clauses'], flush=True)
        Rd_all = [x['roles'] for x in dres]; Rt_ = [x['roles'] for x in tres]
        J = json.loads((OUT / f'template_depth_{s}.json').read_text())['primary']; JL = json.loads((OUT / f'template_lexical_{s}.json').read_text())['lex']
        pid = lambda it: [t for ro in ROLES for t, q_ in zip(it['ids'], it['role']) if q_ == ro]
        dev_ids = [(pid(it), Rd_all[i]) for i, it in enumerate(DITEMS)]; voc = sorted({t for (ids, ro), k_ in zip(dev_ids, K) if k_ != 'text' for t, r in zip(ids, ro) if r == 'c'}); cv = {v: j for j, v in enumerate(voc)}
        Pd_full = {k: [x['st'][k] for x in dres] for k in kinds}; Pd_full['lex'] = lexical(dev_ids, cv); del dres
        Pt = {k: [x['st'][k] for x in tres] for k in kinds}; Pt['lex'] = lexical([(m_['pids'], r) for m_, r in zip(tmeta, Rt_)], cv); del tres
        MATCH = [('lex', ('lex',), JL['m'], JL['d']), ('h0', ('h0',), None, None)] + [(k, (k,), None, None) for k in kinds if k != 'h0'] + [('path', tuple(f'D{l}' for l in range(L_)), None, None), ('wells', tuple(f'W{l}' for l in range(L_)), None, None), ('states', tuple(f'h{l}' for l in range(1, L_ + 1)), None, None)]
        if GATES: MATCH, DEVS = MATCH[:1], {'original': DEVS['original']}
        for dv, msk in DEVS.items():
            ix = np.flatnonzero(msk); Pd = {k: [v[i] for i in ix] for k, v in Pd_full.items()}; Rd = [Rd_all[i] for i in ix]; dm = (K[ix], want[ix], wi[ix], fr[ix], g[ix], lab[ix])
            q[dv] = {}; per = {}
            for name, ks, m, d in MATCH:
                m, d = (m, d) if m is not None else (J[name]['m'], J[name]['d'])
                LO, top, fire, rho = fit_score(Pd, Rd, Pt, Rt_, ks, m, d, dm, tm); q[dv][name], per[name] = readings(LO, top, fire, tm); q[dv][name]['m, d, rho'] = [m, d, rho]
                print(f'{s} {dv:8s} {name:7s} m{m} d{d:2d} | ' + ' '.join(f"{L}: " + ','.join(f"{q[dv][name][L][rd][0]:.2f}" for rd in ('rank', 'foil', 'twin') if rd in q[dv][name][L]) for L in LEVEL_ORDER), flush=True)
            for name in per:
                for ref in ('h0', 'lex'):
                    if name == ref or ref not in per: continue
                    for L in LEVEL_ORDER: q[dv][name][L][f'minus {ref}'] = {rd: bootc(per[name][L][rd] - per[ref][L][rd]) for rd in per[name][L]}
            # the gain of each reading over the embedding, analogy levels against L0 and L1 (call S3); levels resampled independently
            ANL, BAS = ('L2', 'L3', 'L4', 'L5', 'L6', 'C1', 'C2', 'C3'), ('L0', 'L1')
            for name in per:
                if name in ('h0', 'lex') or GATES: continue
                dd = {L: per[name][L]['foil'] - per['h0'][L]['foil'] for L in ANL + BAS}; bs = []
                for _ in range(B_): bs.append(np.mean([dd[L][RNG.integers(20, size=20)].mean() for L in ANL]) - np.mean([dd[L][RNG.integers(20, size=20)].mean() for L in BAS]))
                pt = np.mean([dd[L].mean() for L in ANL]) - np.mean([dd[L].mean() for L in BAS]); q[dv][name]['S3 gain over h0, analogy minus L0-L1 (foil)'] = [float(pt), *[float(x) for x in np.percentile(bs, [2.5, 97.5])]]
        (OUT / f'template_analogy_{"gates_" if GATES else ""}{s}.json').write_text(json.dumps(q, indent=1))
