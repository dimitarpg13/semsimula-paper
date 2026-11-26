#!/usr/bin/env python3
"""template_depth_test.py with the particles' state vectors kept (2026-11-26), for template_metric_fit.py.
    python3 template_vectors_extract.py OUT FOLDER name=<stem> [threads=8]
Runs template_depth_test.py unchanged and adds, per premise, the whitened vector of every particle for each state h0 ... hL
(float16, whitened by ordinary text as the stored cosines are). Writes OUT/depth_<stem>.pkl with a 'vec' entry per premise."""
from pathlib import Path
DBG = Path(__file__).resolve().parent
src = (DBG / 'template_depth_test.py').read_text()
old = "res[i] = dict(roles=ro_, st=st)"
assert src.count(old) == 1
src = src.replace(old, "res[i] = dict(roles=ro_, st=st, vec={k: Z[k][a:e_].astype(np.float16) for k in KINDS if k[0] == 'h'})")
exec(compile(src, str(DBG / 'template_depth_test.py'), 'exec'), {'__name__': '__main__', '__file__': str(DBG / 'template_depth_test.py')})
