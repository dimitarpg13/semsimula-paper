"""Premises of task B, meaning set only by context (2026-11-27), built from template_bcontext_clauses.py, for template_analogy_test.py
(bank=bcontext) and the scorers (items=template_bcontext_items). Levels (content = the frame, 0 to 19):
  SX, SN, SCa, SCb   word sense: explicit, natural, control with both senses named (bound one first, last)
  PX, PN, PCa, PCb   pronoun: explicit, natural, control with both people named (bound one first, last)
Every premise of a frame and wording is the same string; only the context sentence before it differs. As in task B-logic, the
ordinary text before the context is shortened by the context's length (filler from the validation tokens at FILL), so that the
premise starts at the same position in every version: its particles, position vectors included, are identical, and token
identity and the embedding tie. Pair readings: 'twin' (MP against AC, MT against DA), 'other' (DS1 against DS2 and back) and,
on the pronoun levels, 'foil' (the valid form against the third person)."""
import numpy as np
import template_bcontext_clauses as B
from template_analogy_items import WORDINGS, cap

LEVELS = {'SX': ('sense', 'explicit'), 'SN': ('sense', 'natural'), 'SCa': ('sense', 'both_a'), 'SCb': ('sense', 'both_b'),
          'PX': ('pronoun', 'explicit'), 'PN': ('pronoun', 'natural'), 'PCa': ('pronoun', 'both_a'), 'PCb': ('pronoun', 'both_b')}
LEVEL_ORDER = tuple(LEVELS)
WANT = {'MP': 'MP', 'MT': 'MT', 'DS1': 'DS1', 'DS2': 'DS2'}
TWIN = [('twin', 'MP', 'AC', 'MP'), ('twin', 'MT', 'DA', 'MT')]
OTHER = [('other', 'DS1', 'DS2', 'DS1'), ('other', 'DS2', 'DS1', 'DS2')]
FOIL = [('foil', 'MP', 'foil', 'MP'), ('foil', 'MT', 'foiln', 'MT'), ('foil', 'DS1', 'foiln', 'DS1'), ('foil', 'DS2', 'foiln', 'DS2')]
PAIRS = {lv: TWIN + OTHER + (FOIL if sub == 'pronoun' else []) for lv, (sub, _) in LEVELS.items()}
FILL = 1_300_000


def contexts(lv):
    """per frame: (P, Q, S, {meaning: context sentence}) for level lv"""
    sub, fl = LEVELS[lv]
    out = []
    for i in range(20):
        P, Q, S, ctx = (B.sense_frame if sub == 'sense' else B.pronoun_frame)(i)
        out.append((P, Q, S, {k: cap(v) + '.' for k, v in ctx[fl].items()}))
    return out


def build(enc, val=None, levels=LEVEL_ORDER):
    if val is None:
        import verify_pm_switch as P_
        val = np.load(P_.G.VAL).astype(np.int64)
    nl = enc.encode("\n\n"); items = []
    for lv in levels:
        rows = contexts(lv)
        cmax = max(len(enc.encode(c)) for _, _, _, cx in rows for c in cx.values()) + 2 * len(nl) + 2   # context + blank lines + filler
        for i, (P, Q, S, cx) in enumerate(rows):
            for frame, wd in WORDINGS.items():
                for wn, pcs in wd.items():
                    for (pol, m), kind in B.KIND[frame].items():
                        if m not in cx:
                            continue                                    # no third person in word-sense frames
                        c = enc.encode(cx[m]); k = cmax - len(c) - 2 * len(nl); assert k >= 2, (lv, i, m, k)
                        s2 = S if pol == 'aff' else B.neg(S)
                        sub = lambda s_: s_.replace('{P}', P).replace('{Q}', Q).replace('{P^}', cap(P)).replace('{Q^}', cap(Q))
                        ids = [int(v) for v in val[FILL:FILL + k]] + nl + c + nl; role = [None] * len(ids)
                        for t, ro in [(sub(t), ro) for t, ro in pcs] + [(" " + cap(s2), 'p2'), (".", None), (" Therefore", 'th')]:
                            e = enc.encode(t); ids += e; role += [ro] * len(e)
                        items.append(dict(split='analogy', level=lv, content=i, frame=frame, wording=wn, kind=kind, want=WANT.get(kind, ''),
                                          ids=ids, role=role, text=enc.decode(ids), P=P, Q=Q, p2=s2, context=cx[m]))
    return items


if __name__ == '__main__':
    import sys, tempfile, tiktoken
    from collections import Counter
    enc = tiktoken.get_encoding('gpt2')
    it = build(enc, val=np.zeros(FILL + 200, dtype=np.int64))       # a stand-in filler; the check is about the premise tokens
    print(len(it), dict(Counter(x['level'] for x in it)), dict(Counter(x['kind'] for x in it)))
    # within a frame, level, wording and polarity of the second sentence, every version has the same particle tokens at the same positions
    key = lambda x: (x['level'], x['content'], x['wording'], x['p2'])
    groups = {}
    for x in it: groups.setdefault(key(x), []).append(x)
    for g in groups.values():
        sig = {tuple((j, t) for j, (t, r) in enumerate(zip(x['ids'], x['role'])) if r is not None) for x in g}; assert len(sig) == 1, g[0]['text']
        assert len({len(x['ids']) for x in g}) == 1
    # every pair's two kinds exist for every frame and wording of its frame type
    have = {(x['level'], x['content'], x['wording'], x['kind']) for x in it}
    for lv, prs in PAIRS.items():
        for _, a, b, t in prs:
            fr = 'cond' if t in ('MP', 'MT') else 'disj'
            for i in range(20):
                for wn in WORDINGS[fr]:
                    assert (lv, i, wn, a) in have and (lv, i, wn, b) in have, (lv, i, wn, a, b)
    print('particles identical within every group:', len(groups), 'groups; pairs complete')
    for lv in LEVEL_ORDER:
        x = next(y for y in it if y['level'] == lv and y['kind'] == 'MP')
        print(lv, repr(x['context']), '|', repr(x['text'].split('\n\n')[-1]))
