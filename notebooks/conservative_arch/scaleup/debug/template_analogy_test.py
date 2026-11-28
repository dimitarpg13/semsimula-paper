#!/usr/bin/env python3
"""Test 2 extraction: the analogy and in-context premises (2026-11-26).
    python3 template_analogy_test.py OUTDIR FOLDER name=<stem> [ckpt=best|step500] [threads=8] [vec=1] [limit=N] [levels=L3,C2] [bank=blogic]
bank=blogic extracts the premises of task B-logic (template_blogic_items.py) instead of Test 2's; bank=blogic_control its control.
bank=bcontext extracts task B, meaning set only by context (template_bcontext_items.py; 2026-11-27, awaiting the author's review).
levels= extracts only those levels (plus the check premises), for re-extracting a level after its items change; merge the
result into the full file with template_analogy_merge.py. Every premise is extracted alone in its batch row, so its values do not
depend on which other premises are run (the check premises confirm it).
Runs template_depth_test.py with its premise list replaced by the 6,720 premises of template_analogy_items.py, plus 96 of its own
development and test premises rebuilt the same way (split 'check', with 'depth_index' their place in depth_<stem>.pkl), so that the
scorer can confirm that the premises stored by the depth extraction are what this run gives. Everything else is the depth
extraction unchanged: the same 48 tokens of ordinary text before every premise, the same whitening by 12,288 ordinary-text tokens,
the cosines, norms and connective vector per kind (h, D, W at every layer). No flow replay. vec=1 also keeps each particle's
whitened state vector (float16, kinds h0 ... hL), for a later fitted-metric reading.
Writes OUTDIR/analogy_<stem>.pkl (meta, res, L, kinds; meta carries level, content, frame, wording, kind, want and the particle
token ids 'pids')."""
import sys
from pathlib import Path
DBG = Path(__file__).resolve().parent
src = (DBG / 'template_depth_test.py').read_text()
R = [("if 'limit' in OPTS:", "items = _analogy(items)\nif 'limit' in OPTS:"),                                   # the premise list
     ("    if i == 0: keep['args'] = {}\n", ""), ("    if i == 0: flow['text'] = flow_rows(pos); keep['args'] = None\n", ""),   # no flow replay
     ("    if b == 0 and len(order) >= 32: keep['args'] = {}\n", ""),
     ("(OUT / f'flow_{OPTS[\"name\"]}.json').write_text(json.dumps(flow, indent=1)); pickle.dump(dict(meta=meta, res=res, L=L_, kinds=KINDS), open(OUT / f'depth_{OPTS[\"name\"]}.pkl', 'wb'))",
      "pickle.dump(dict(meta=meta, res=res, L=L_, kinds=KINDS), open(OUT / f'analogy_{OPTS[\"name\"]}.pkl', 'wb'))"),
     ("meta = [{k: v for k, v in it.items() if k not in ('ids', 'role')} for it in items]",
      "meta = [{**{k: v for k, v in it.items() if k not in ('ids', 'role')}, 'pids': [t for ro in ('c', 'a1', 'a2', 'p2', 'th') for t, q_ in zip(it['ids'], it['role']) if q_ == ro]} for it in items]")]
if 'vec=1' in sys.argv:
    R.append(("res[i] = dict(roles=ro_, st=st)", "res[i] = dict(roles=ro_, st=st, vec={k: Z[k][a:e_].astype(np.float16) for k in KINDS if k[0] == 'h'})"))
for a, b in R:
    assert src.count(a) == 1, a
    src = src.replace(a, b)


def _analogy(depth_items):
    import numpy as np, tiktoken
    bank = next((x.split('=', 1)[1] for x in sys.argv if x.startswith('bank=')), 'analogy')
    from importlib import import_module
    build = import_module({'analogy': 'template_analogy_items', 'blogic': 'template_blogic_items', 'blogic_control': 'template_blogic_control_items', 'bcontext': 'template_bcontext_items'}[bank]).build
    rng = np.random.default_rng(5); pick = sorted(rng.choice([j for j, x in enumerate(depth_items) if x['kind'] != 'text'], 96, replace=False))
    chk = [{**depth_items[j], 'split': 'check', 'level': 'check', 'depth_index': int(j)} for j in pick]
    lv = next((a.split('=', 1)[1].split(',') for a in sys.argv if a.startswith('levels=')), None)
    return chk + [x for x in build(tiktoken.get_encoding('gpt2')) if lv is None or x['level'] in lv]


exec(compile(src, str(DBG / 'template_depth_test.py'), 'exec'), {'__name__': '__main__', '__file__': str(DBG / 'template_depth_test.py'), '_analogy': _analogy})
