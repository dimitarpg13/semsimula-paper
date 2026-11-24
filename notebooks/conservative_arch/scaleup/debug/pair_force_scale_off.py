"""(2026-11-22; moved from a scratch script 2026-11-23.) Independent check of the pair force's removal cost: zero its trained per-layer scale (raw -> -30, softplus ~1e-13)
instead of wrapping its output. Same 16 fresh windows as recall_pathway_probe.py."""
import sys
from pathlib import Path
import numpy as np, torch
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))
OUT, FOLDER = Path(sys.argv[1]), sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:])
src = (DBG / 'copy_split.py').read_text(); src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', 'windows=1', 'threads=8'] + ([f"val={OPTS['val']}"] if 'val' in OPTS else [])
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)
import verify_pm_switch as P
_b = P.build; got = {}
def build(*a, **k):
    r = _b(*a, **k); got['m'] = r[0]; return r
P.build = build
run, desc = CS['ladder_model'](FOLDER, OPTS.get('ckpt'))
m = got['m']
v = np.load(CS['VAL']).astype(np.int64); W, T = 16, 512
sp = (len(v) - (T + 1)) / W; st = (np.arange(W) * sp + sp / 2).astype(np.int64)
xy = np.stack([v[s:s + T + 1] for s in st])
def ce():
    out = []
    for i in range(0, W, 4):
        lg = run(xy[i:i + 4, :T])
        out.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), torch.from_numpy(xy[i:i + 4, 1:]).reshape(-1)).item())
    return float(np.mean(out))
base = ce()
raw = m.raw_v_phi_scale.data.clone()
print('scales (softplus):', torch.nn.functional.softplus(raw).tolist())
m.raw_v_phi_scale.data.fill_(-30.0); off = ce(); m.raw_v_phi_scale.data.copy_(raw)
print(f'{desc[:60]}  base {base:.4f}  pair force scale zeroed {off:.4f}  change {off - base:+.4f} nats ({100 * (np.exp(off - base) - 1):+.1f}% ppl)')
