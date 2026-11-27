"""Premises of task B-logic (2026-11-27), built from template_blogic_clauses.py, for template_analogy_test.py (bank=blogic) and the
scorers (items=template_blogic_items). Levels: BX (explicit contexts) and BN (natural contexts); content = the frame (0 to 19).
Every premise of a frame and wording is the same string; the context sentence before it differs. The ordinary text before the
context is shortened by the context's length (filler from the validation tokens at FILL), so that the premise starts at the same
position in every version and its particles, position vectors included, are identical: the embedding and token identity tie."""
import numpy as np
import template_blogic_clauses as B
from template_analogy_items import WORDINGS, cap

LEVEL_ORDER = ('BX', 'BN'); FLAVOUR = {'BX': 'explicit', 'BN': 'natural'}; WANT = {'MP': 'MP', 'MT': 'MT', 'DS1': 'DS1', 'DS2': 'DS2'}
FOIL = [('foil', 'MP', 'foil', 'MP'), ('foil', 'MT', 'foil', 'MT'), ('foil', 'DS1', 'foil', 'DS1'), ('foil', 'DS2', 'foil', 'DS2')]
TWIN = [('twin', 'MP', 'AC', 'MP'), ('twin', 'MT', 'DA', 'MT'), ('twin', 'DS1', 'Paff', 'DS1'), ('twin', 'DS2', 'Qaff', 'DS2')]
PAIRS = {lv: FOIL + TWIN for lv in LEVEL_ORDER}
FILL = 1_300_000; CMAX = 26                       # context + its two blank lines + filler = CMAX tokens


def build(enc, val=None):
    if val is None:
        import verify_pm_switch as P_
        val = np.load(P_.G.VAL).astype(np.int64)
    nl = enc.encode("\n\n"); items = []
    for lv in LEVEL_ORDER:
        for i, (P, Q, S, U) in enumerate(B.FRAMES):
            ctx = B.contexts(i, FLAVOUR[lv])
            for frame, wd in WORDINGS.items():
                for wn, pcs in wd.items():
                    for m, kind in B.KIND[frame].items():
                        c = enc.encode(ctx[m]); k = CMAX - len(c) - 2 * len(nl); assert k >= 2, (i, m, k)
                        sub = lambda s_: s_.replace('{P}', P).replace('{Q}', Q).replace('{P^}', cap(P)).replace('{Q^}', cap(Q))
                        ids = [int(v) for v in val[FILL:FILL + k]] + nl + c + nl; role = [None] * len(ids)
                        for t, ro in [(sub(t), ro) for t, ro in pcs] + [(" " + cap(S), 'p2'), (".", None), (" Therefore", 'th')]:
                            e = enc.encode(t); ids += e; role += [ro] * len(e)
                        items.append(dict(split='analogy', level=lv, content=i, frame=frame, wording=wn, kind=kind, want=WANT.get(kind, ''), ids=ids, role=role,
                                          text=enc.decode(ids), P=P, Q=Q, p2=S, context=ctx[m]))
    return items


if __name__ == '__main__':
    import sys, tempfile, tiktoken
    sys.argv.append(tempfile.mkdtemp())             # verify_pm_switch reads an output folder from argv
    from collections import Counter
    enc = tiktoken.get_encoding('gpt2'); it = build(enc)
    print(len(it), Counter(x['level'] for x in it), Counter(x['kind'] for x in it))
    # within a frame, level and wording, every version has the same particle tokens at the same positions
    key = lambda x: (x['level'], x['content'], x['wording'])
    groups = {}
    for x in it: groups.setdefault(key(x), []).append(x)
    for g in groups.values():
        sig = {tuple((j, t) for j, (t, r) in enumerate(zip(x['ids'], x['role'])) if r is not None) for x in g}; assert len(sig) == 1, g[0]['text']
        assert len({len(x['ids']) for x in g}) == 1
    print('particles identical within every group:', len(groups), 'groups')
    print(repr(it[0]['text']))
