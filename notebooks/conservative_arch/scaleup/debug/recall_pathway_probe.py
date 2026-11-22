#!/usr/bin/env python3
"""Which pathway carries an earlier token to its recall? Cut one pathway at a layer and read the loss where the
token is recalled (2026-11-17, protocol p-bank-memory-conjecture: the recall pathway probe, LR2 and LR3).

    python3 recall_pathway_probe.py OUT FOLDER [ckpt=<file>] [windows=16] [threads=8] [val=<tokens.npy>]
                                    [name=<stem>] [maxfreq=8] [check=1]

Recall pairs, per window: a position q whose target last occurred at an earlier input position p, at distance
q + 1 - p in a band (near 1-16, far 17-256); tokens occurring more than `maxfreq` times in the window are excluded.
Control positions: targets that do not occur earlier in the window, with the same frequency filter. Cuts, each at
one layer, applied to every recall pair at once:
  - vel:  the velocity entering the layer is set to zero (h_prev = h); a second-order model only;
  - xi:   for each pair, p is removed from q's content-routed xi channels (its logit set to -inf);
  - phi:  for each pair, p is removed from q's V_phi sources (the causal mask at (q, p) set to False);
  - xiphi: both.
Readings: the change of the per-token loss (nats) at near and far recall positions and at control positions,
averaged within each window and then over windows, with the standard error over windows. The windows are evenly
spaced over the validation set and offset by half a spacing, so they are not the windows of the earlier probes.
Writes OUT/recall_pathway_<stem>.txt and .json.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))
W, T, MAXF = int(OPTS.get('windows', 16)), 512, int(OPTS.get('maxfreq', 8))

src = (DBG / 'copy_split.py').read_text()
src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', 'windows=1', f"threads={OPTS.get('threads', 8)}"] \
    + ([f"val={OPTS['val']}"] if 'val' in OPTS else [])
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)
import verify_pm_switch as P
_build = P.build; got = {}
def build(*a, **k):
    r = _build(*a, **k); got['model'] = r[0]; return r
P.build = build
_, desc = CS['ladder_model'](FOLDER, OPTS.get('ckpt'))
m = got['model']; m.eval()
print('model:', desc, flush=True)
FOB = bool(getattr(m.cfg, 'first_order_overdamped', False))
xm = m.xi_module
assert getattr(xm, 'W_q', None) is not None, 'needs content-routed xi channels'
MX = sys.modules[type(xm).__module__]

# fresh windows: evenly spaced, offset by half a spacing
v = np.load(CS['VAL']).astype(np.int64)
sp = (len(v) - (T + 1)) / W
starts = (np.arange(W) * sp + sp / 2).astype(np.int64)
xy = np.stack([v[s:s + T + 1] for s in starts]); X, Y = xy[:, :T], xy[:, 1:]


def classes(xr, yr):
    """near / far recall pairs (q, p) and control positions q"""
    cnt = np.bincount(xr, minlength=50257); last = {}; out = {'near': [], 'far': [], 'control': []}
    for q in range(len(xr)):
        last[int(xr[q])] = q
        t = int(yr[q]); p = last.get(t)
        if cnt[t] > MAXF:
            continue
        if p is None:
            out['control'].append((q, -1)); continue
        if p == q:
            continue
        dist = q + 1 - p
        if 1 <= dist <= 16:
            out['near'].append((q, p))
        elif 17 <= dist <= 256:
            out['far'].append((q, p))
    return out


st = {'layer': None, 'cut': set(), 'cut_layer': None, 'REM': None}
orig_step = m._layer_step_langevin
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    st['layer'] = int(layer_idx)
    if 'vel' in st['cut'] and int(layer_idx) == st['cut_layer']:
        h_prev = h                                      # zero velocity into this layer
    return orig_step(h, h_prev, m_b, gamma, dt_, layer_idx)
m._layer_step_langevin = step

orig_xi = xm.forward
def xi_masked(h):
    """the content-routed branch of MultiChannelXi.forward, with -inf at the removed (q, p) entries"""
    B, T_, d = h.shape; alphas = xm.alpha; dk = xm.content_d_k
    base = torch.stack([MX.causal_ema_logits(T_, alphas[k], h.dtype, h.device) for k in range(xm.K)], dim=0)
    q = xm.W_q(h).view(B, T_, xm.K, dk).permute(0, 2, 1, 3)
    kk = xm.W_k(h).view(B, T_, xm.K, dk).permute(0, 2, 1, 3)
    logits = base.unsqueeze(0) + (q @ kk.transpose(-1, -2)) * (dk ** -0.5)
    if st['REM'] is not None:
        logits = logits.masked_fill(st['REM'].unsqueeze(1), float('-inf'))
    w = torch.softmax(logits, dim=-1)
    return (w @ h.unsqueeze(1)).permute(0, 2, 1, 3).contiguous()
def xi_forward(h):
    if 'xi' in st['cut'] and st['layer'] == st['cut_layer']:
        return xi_masked(h)
    return orig_xi(h)
xm.forward = xi_forward

orig_mask = m._pair_mask_for
def pair_mask(T_, device):
    mk = orig_mask(T_, device)
    if 'phi' in st['cut'] and st['layer'] == st['cut_layer'] and st['REM'] is not None:
        return mk.unsqueeze(0) & ~st['REM']
    return mk
m._pair_mask_for = pair_mask

ARMS = [('base', set(), None)] + ([] if FOB else [('vel@1', {'vel'}, 1)]) + \
       [('xi@1', {'xi'}, 1), ('phi@1', {'phi'}, 1), ('xiphi@1', {'xi', 'phi'}, 1), ('xiphi@0', {'xi', 'phi'}, 0)]
if getattr(m, 'V_phi', None) is None:              # SR2-NP (2026-11-21): no pair force, so only the xi cuts
    ARMS = [a for a in ARMS if a[0] in ('base', 'vel@1', 'xi@1')] + [('xi@0', {'xi'}, 0)]
per = {a: {c: [] for c in ('near', 'far', 'control')} for a, _, _ in ARMS}
n_cls = {c: 0 for c in ('near', 'far', 'control')}
checks = []
for i in range(0, W, 4):
    xb, yb = X[i:i + 4], Y[i:i + 4]; B = len(xb)
    cl = [classes(xb[b], yb[b]) for b in range(B)]
    REM = torch.zeros(B, T, T, dtype=torch.bool)
    for b in range(B):
        for c in ('near', 'far'):
            for q, p in cl[b][c]:
                REM[b, q, p] = True
    for c in n_cls:
        n_cls[c] += sum(len(cl[b][c]) for b in range(B))
    ce = {}
    for a, cut, L in ARMS:
        st.update(cut=cut, cut_layer=L, REM=REM if cut & {'xi', 'phi'} else None)
        with torch.enable_grad():                       # the V_phi pair force is an autograd gradient here
            lg = m(torch.from_numpy(xb))[0].detach()
        ce[a] = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(),
                                                  torch.from_numpy(yb).reshape(-1), reduction='none').view(B, T)
    if OPTS.get('check', '0') == '1' and i == 0:
        _cut = 'xiphi@1' if 'xiphi@1' in ce else 'xi@1'
        # the re-implemented xi branch reproduces the model's own with nothing removed; a cut leaves earlier
        # positions than the first recall position unchanged (causality)
        st.update(cut={'xi'}, cut_layer=1, REM=torch.zeros(B, T, T, dtype=torch.bool))
        with torch.enable_grad():
            lg_e = m(torch.from_numpy(xb))[0].detach()
        ce_e = torch.nn.functional.cross_entropy(lg_e.reshape(-1, lg_e.shape[-1]).float(),
                                                 torch.from_numpy(yb).reshape(-1), reduction='none').view(B, T)
        checks.append(f'xi re-implementation with nothing removed: max |d loss| {float((ce_e - ce["base"]).abs().max()):.2e}')
        q0 = min(q for b in range(B) for c in ('near', 'far') for q, _ in cl[b][c])
        checks.append(f'cut {_cut} before the first recall position ({q0}): max |d loss| '
                      f'{float((ce[_cut][:, :q0] - ce["base"][:, :q0]).abs().max()):.2e}')
    for a, _, _ in ARMS:
        d_ = ce[a] - ce['base']
        for b in range(B):
            for c in ('near', 'far', 'control'):
                qs = [q for q, _ in cl[b][c]]
                per[a][c].append(float(d_[b, qs].mean()) if qs else float('nan'))
    st.update(cut=set(), cut_layer=None, REM=None)
    print(f'windows {i + 4}/{W}', flush=True)

res = {'n': n_cls, 'arms': {}, 'checks': checks, 'first_order': FOB}
lines = [f'Recall pathway probe: {FOLDER}', f'model {desc}',
         f'{W} fresh validation windows x {T} ({Path(CS["VAL"]).name}; offset half a spacing); first order {FOB}; '
         f'tokens over {MAXF} occurrences per window excluded; positions: {n_cls}',
         'change of the per-token loss (nats) at the class, mean over windows (se over windows)', '',
         f'{"cut":>9}  {"near recall":>18}  {"far recall":>18}  {"control":>18}  {"far / control":>13}']
for a, _, _ in ARMS:
    r = {}
    for c in ('near', 'far', 'control'):
        x_ = np.array(per[a][c]); x_ = x_[np.isfinite(x_)]
        r[c] = [float(x_.mean()), float(x_.std(ddof=1) / np.sqrt(len(x_))) if len(x_) > 1 else float('nan')]
    r['per_window'] = per[a]
    res['arms'][a] = r
    ratio = r['far'][0] / r['control'][0] if r['control'][0] != 0 else float('nan')
    lines.append(f'{a:>9}  {r["near"][0]:+9.4f} ({r["near"][1]:.4f})  {r["far"][0]:+9.4f} ({r["far"][1]:.4f})  '
                 f'{r["control"][0]:+9.4f} ({r["control"][1]:.4f})  {ratio:13.2f}')
if checks:
    lines += ['', 'checks:'] + ['  ' + c for c in checks]
stem = OPTS.get('name', FOLDER[-40:])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f'recall_pathway_{stem}.txt').write_text('\n'.join(lines) + '\n')
(OUT / f'recall_pathway_{stem}.json').write_text(json.dumps({'folder': FOLDER, 'model': desc, **res}, indent=1))
print('\n'.join(lines))
