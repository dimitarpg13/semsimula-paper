"""Build a code-corpus ladder arm for the CPU readings (2026-11-10, for the C5 pair PM2-CODE and FO-b-PM2-CODE).

    model, g, ckpt = code_arm.build(FOLDER, OUT, ckpt=None)

FOLDER is a ladder run folder in ~/Downloads (semsimula_fock_cfc_baoab_code_<tag>). Its Cell 0 is rebuilt from
the tag's parts by copy_split.tag_repl, as copy_split.py and refinement_flow_confirmation.py's code kind do; the
default checkpoint is the folder's <prefix>_best.pt; g['val_ids'] becomes the code validation tokens (copy_split's
VAL) in place of the harness's OpenWebText. The caller has set sys.argv for verify_pm_switch before importing it.
"""
import sys
from pathlib import Path

import numpy as np

DBG = Path(__file__).resolve().parent


def copy_split_defs(out):
    """copy_split's definitions without its command dispatch."""
    import types
    src = (DBG / 'copy_split.py').read_text()
    argv = list(sys.argv); sys.argv = ['copy_split.py', str(Path(out) / 'harness'), 'none']
    g = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
    try:
        exec(compile(src[:src.index("\nif MODE == 'nll':")], 'copy_split.py', 'exec'), g)
    finally:
        sys.argv = argv
    return types.SimpleNamespace(**g)


def build(folder, out, ckpt=None):
    import verify_pm_switch as P
    CS = copy_split_defs(out)
    repl, tag = CS.tag_repl(folder)
    if not ckpt:
        ck = [p.name for p in (Path.home() / 'Downloads' / folder / 'checkpoints').glob('fock_cfc_*_best.pt')
              if '_step' not in p.name]
        assert len(ck) == 1, ck
        ckpt = ck[0]
    model, g, _ = P.build(folder, repl, ckpt=ckpt)
    assert g.get('_variant_tag', tag) == tag, (g.get('_variant_tag'), tag)
    g['val_ids'] = np.load(CS.VAL)
    print(f'checkpoint: {ckpt}  (code validation tokens {CS.VAL})', flush=True)
    return model, g, ckpt
