"""Premises of test 2, the analogy and in-context test set (2026-11-26), built from template_analogy_clauses.py.
Shared by template_analogy_test.py (extraction) and template_analogy_analysis.py (scoring), so both see the same items and pairs.

A premise is as in template_depth_test.py: connective and clauses in one of six known wordings, the second sentence (p2), then
" Therefore". The particles have the same roles (c, a1, a2, p2, th). In C1 the context sentence and a blank line come first; its
tokens are not particles. Clause-level banks (L0, L1, L2 from BASE; L3, L5, L6) take Q from another frame of the same bank, rolled
by 17 as in the earlier tests; L4, C1 to C4 carry their own Q.

Kinds of second sentence (x is the restatement of the level, xf its matched foil):
  cond  MP  affirms P (want MP)       MPf  its foil     MT  denies Q (want MT)    MTf  its foil
        AC  affirms Q (twin of MP)    DA   denies P (twin of MT)                 MPp  the C2' paraphrase of P (want MP)
  disj  DS1 denies P (want DS1)       DS1f its foil     DS2 denies Q (want DS2)   DS2f its foil
        Paff affirms P (twin of DS1)  Qaff affirms Q (twin of DS2)
PAIRS[level] lists the comparisons (reading, kind that should score higher, kind it is compared with, template whose log odds
are compared), within one frame, clause and wording."""
import numpy as np
import template_analogy_clauses as A

cap = lambda s: s[0].upper() + s[1:]
pq = lambda lead: [(lead, 'c'), (" {P}", 'a1'), (",", None), (" {Q}", 'a2'), (".", None)]
WORDINGS = {'cond': {'if-then': [("If", 'c'), (" {P}", 'a1'), (",", None), (" then", 'c'), (" {Q}", 'a2'), (".", None)], 'when': pq("When"), 'provided': pq("Provided that")},
            'disj': {'either-or': [("Either", 'c'), (" {P}", 'a1'), (" or", 'c'), (" {Q}", 'a2'), (".", None)], 'or': [("{P^}", 'a1'), (" or", 'c'), (" {Q}", 'a2'), (".", None)],
                     'or-else': [("{P^}", 'a1'), (",", None), (" or else", 'c'), (" {Q}", 'a2'), (".", None)]}}
WANT = {'MP': 'MP', 'MT': 'MT', 'DS1': 'DS1', 'DS2': 'DS2', 'MPp': 'MP'}
FOIL = [('foil', 'MP', 'MPf', 'MP'), ('foil', 'MT', 'MTf', 'MT'), ('foil', 'DS1', 'DS1f', 'DS1'), ('foil', 'DS2', 'DS2f', 'DS2')]
TWIN = [('twin', 'MP', 'AC', 'MP'), ('twin', 'MT', 'DA', 'MT'), ('twin', 'DS1', 'Paff', 'DS1'), ('twin', 'DS2', 'Qaff', 'DS2')]


def generic(P, Q, aP, nP, aQ, nQ):
    """aP, nP: (restatement, foil) affirming and denying P; aQ, nQ the same for Q"""
    return P, Q, dict(cond=dict(MP=aP[0], MPf=aP[1], MT=nQ[0], MTf=nQ[1], AC=aQ[0], DA=nP[0]),
                      disj=dict(DS1=nP[0], DS1f=nP[1], DS2=nQ[0], DS2f=nQ[1], Paff=aP[0], Qaff=aQ[0]))


def frames():
    """level -> list of (P, Q, context or '', {frame: {kind: second sentence}})"""
    out = {}
    roll = lambda n: np.roll(np.arange(n), 17)
    B = A.BASE; q = roll(len(B))
    for lv, (a, af, n, nf) in {'L0': ('aff', 'near', 'neg', 'near_neg'), 'L1': ('syn', 'near', 'syn_neg', 'near_neg'), 'L2': ('para', 'pfoil', 'para_neg', 'pfoil_neg')}.items():
        out[lv] = [(*generic(x['aff'], B[q[i]]['aff'], (x[a], x[af]), (x[n], x[nf]), (B[q[i]][a], B[q[i]][af]), (B[q[i]][n], B[q[i]][nf])), '') for i, x in enumerate(B)]
    for lv, bank, (a, af, n, nf) in (('L3', A.L3, ('case', 'case_foil', 'ncase', 'ncase_foil')), ('L5', A.L5, ('res', 'foil', 'res_neg', 'foil_neg')), ('L6', A.L6, ('para', 'pfoil', 'para_neg', 'pfoil_neg'))):
        q = roll(len(bank))
        out[lv] = [(*generic(x['aff'], bank[q[i]]['aff'], (x[a], x[af]), (x[n], x[nf]), (bank[q[i]][a], bank[q[i]][af]), (bank[q[i]][n], bank[q[i]][nf])), '') for i, x in enumerate(bank)]
    out['L4'] = [(*generic(x['P'], x['Q'], (x['res'], x['foil']), (x['res_neg'], x['foil_neg']), (x['resQ'], x['foilQ']), (x['resQ_neg'], x['foilQ_neg'])), '') for x in A.L4]
    out['C1'] = [(x['P'], x['Q'], dict(cond=dict(MP=x['res'], MPf=x['foil'], DA=x['res_neg']), disj=dict(DS1=x['res_neg'], DS1f=x['foil_neg'], Paff=x['res'])), x['context']) for x in A.C1]
    out['C2'] = [(x['P'], x['Q'], dict(cond=dict(MP=x['res'], MPf=x['foil'], MPp=x['para'], DA=x['res_neg']), disj=dict(DS1=x['res_neg'], DS1f=x['foil_neg'], Paff=x['res'])), '') for x in A.C2]
    # C3: the foil is the other clause's pronoun, so "She does." affirms Q (AC) and "She does not." denies Q (MT, DS2)
    out['C3'] = [(x['P'], x['Q'], dict(cond=dict(MP=x['res'], AC=x['foil'], MT=x['foil_neg'], DA=x['res_neg']), disj=dict(DS1=x['res_neg'], DS2=x['foil_neg'], Paff=x['res'], Qaff=x['foil'])), '') for x in A.C3]
    out['C4'] = [(x['P'], x['Q'], dict(cond=dict(MT=x['Q_inc'], MTf=x['Q_foil'], DA=x['P_inc']), disj=dict(DS1=x['P_inc'], DS1f=x['P_foil'], DS2=x['Q_inc'], DS2f=x['Q_foil'])), '') for x in A.C4]
    return out


PAIRS = {lv: FOIL + TWIN for lv in ('L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6')}
PAIRS['C1'] = [FOIL[0], FOIL[2], TWIN[2]]
PAIRS['C2'] = [FOIL[0], FOIL[2], TWIN[2], ("foil C2'", 'MPp', 'MPf', 'MP')]
PAIRS['C3'] = [('foil', 'MP', 'AC', 'MP'), ('foil', 'MT', 'DA', 'MT'), ('foil', 'DS1', 'DS2', 'DS1'), ('foil', 'DS2', 'DS1', 'DS2')] + TWIN
PAIRS['C4'] = [FOIL[1], FOIL[2], FOIL[3], TWIN[1]]
LEVEL_ORDER = ('L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'C1', 'C2', 'C3', 'C4')


def build(enc):
    """the premises, as dicts with ids, role, text and the meta fields the scorer uses"""
    items = []
    for lv, rows in frames().items():
        for i, (P, Q, kinds, ctx) in enumerate(rows):
            sub = lambda s: s.replace('{P}', P).replace('{Q}', Q).replace('{P^}', cap(P)).replace('{Q^}', cap(Q))
            for frame, wd in WORDINGS.items():
                for wn, pcs in wd.items():
                    for kind, p2 in kinds[frame].items():
                        segs = ([(ctx, None), ("\n\n", None)] if ctx else []) + [(sub(t), ro) for t, ro in pcs] + [(" " + cap(p2), 'p2'), (".", None), (" Therefore", 'th')]; ids, role = [], []
                        for t, ro in segs:
                            e = enc.encode(t); ids += e; role += [ro] * len(e)
                        items.append(dict(split='analogy', level=lv, content=i, frame=frame, wording=wn, kind=kind, want=WANT.get(kind, ''), ids=ids, role=role, text=''.join(t for t, _ in segs), P=P, Q=Q, p2=p2))
    return items


if __name__ == '__main__':
    import tiktoken
    from collections import Counter
    it = build(tiktoken.get_encoding('gpt2'))
    print(len(it), Counter(x['level'] for x in it))
    for lv in LEVEL_ORDER:
        x = next(y for y in it if y['level'] == lv and y['kind'] in ('MP', 'MT'))
        print(lv, repr(x['text']))
    # every pair's two kinds exist for every clause and wording of its frame
    have = {(x['level'], x['content'], x['wording'], x['kind']) for x in it}
    for lv, prs in PAIRS.items():
        for _, a, b, t in prs:
            fr = 'cond' if t in ('MP', 'MT') else 'disj'
            for i in range(20):
                for wn in WORDINGS[fr]:
                    assert (lv, i, wn, a) in have and (lv, i, wn, b) in have, (lv, i, wn, a, b)
    print('pairs complete')
