#!/usr/bin/env python3
"""How compressible are V_theta's context projections? Truncate each to rank m and read the loss (2026-11-13).

    python3 vtheta_projection_svd.py OUT FOLDER [ckpt=<file>] [windows=16] [threads=8] [val=<tokens.npy>] [name=<stem>]
                                     [ranks=64,128,256,512,1024] [joint=256,512]

Writes OUT/projection_svd_<stem>.txt. The joint bank maps the concatenated context channels (in_d = n_ctx * d = 1,920 at
d = 384) to the wells' parameters with dense layers: B_proj (to K * d * rank), mu_proj and a_proj (to K * d), w_proj (to K).
A bottleneck of width m (in_d -> m -> out) has m * (in_d + out) weights. Its best case is bounded from below by
truncating the trained weight to rank m, without retraining:

  - plain: the rank-m SVD truncation of W;
  - input-aware: the rank-m truncation of W C^(1/2), mapped back with C^(-1/2), where C = E[x x^T] is the second moment of
    the bank's input x (the context channels with the layer's depth code added, both layers pooled, since the bank is
    shared), estimated on the even windows of 2 x windows evenly spaced ones. This is the rank-m map closest to W on
    the inputs the bank actually sees, so it is the better bound.

The loss is read on the odd (held-out) windows, the bias kept, one projection truncated at a time, then all three
together at the ranks in joint=. A small change at rank m says a trained bottleneck of width m will do at least as
well; a large one does not rule a bottleneck out, since training adapts the rest of the model to it.
Also reported: each projection's share of the squared singular values in its top m, plain and input-aware.
"""
import sys
from pathlib import Path

import torch

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
W = int(OPTS.get('windows', 16))
RANKS = [int(v) for v in OPTS.get('ranks', '64,128,256,512,1024').split(',')]
JOINT = [int(v) for v in OPTS.get('joint', '256,512').split(',') if v]
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))

# copy_split's own Cell-0-from-tag build and validation windows, without its command dispatch (as vtheta_well_probe.py)
src = (DBG / 'copy_split.py').read_text()
src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', f'windows={2 * W}', f"threads={OPTS.get('threads', 8)}"] \
    + ([f"val={OPTS['val']}"] if 'val' in OPTS else [])
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)

import verify_pm_switch as P
_build = P.build; got = {}
def build(*a, **k):
    r = _build(*a, **k); got['model'] = r[0]; return r
P.build = build
_, desc = CS['ladder_model'](FOLDER, OPTS.get('ckpt'))
model = got['model']
print('model:', desc, flush=True)

VT = model.V_theta
bank = VT.bank.bank if hasattr(VT.bank, 'bank') else VT.bank      # depth-conditioned -> joint -> mixture
PROJ = [n for n in ('B_proj', 'mu_proj', 'a_proj') if getattr(bank, n, None) is not None]

x, y = CS['windows']()
xe, xv, yv = x[0::2], x[1::2], y[1::2]

def loss_on(xs, ys, label):
    nll = []
    for i in range(0, len(xs), 4):
        with torch.enable_grad():
            lg = model(torch.from_numpy(xs[i:i + 4]))[0].detach()
        if ys is not None:
            nll.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), torch.from_numpy(ys[i:i + 4]).reshape(-1)))
        print(f'{label}: windows {i + 4}/{len(xs)}', flush=True)
    return float(torch.stack(nll).mean()) if nll else None

# -- second moment of the bank input, both layers pooled (every call of _components sees the same xi within a layer
#    step, so count each distinct input tensor once)
acc = {'C': 0.0, 'n': 0}; seen = set()
orig_comp = bank._components
def comp_acc(xi):
    if id(xi) not in seen:
        seen.add(id(xi))
        with torch.no_grad():
            z = xi.detach().double().reshape(-1, xi.shape[-1])
            acc['C'] = acc['C'] + z.T @ z; acc['n'] += z.shape[0]
    return orig_comp(xi)
bank._components = comp_acc
loss_on(xe, None, 'input moment')
bank._components = orig_comp
C = acc['C'] / acc['n']
lam, V = torch.linalg.eigh(C)
lam = lam.clamp(min=lam.max() * 1e-8)
Ch, Cih = (V * lam.sqrt()) @ V.T, (V * lam.rsqrt()) @ V.T

orig_w = {n: getattr(bank, n).weight.detach().clone() for n in PROJ}
svd = {}
for n in PROJ:
    Wd = orig_w[n].double()
    U, S, Vh = torch.linalg.svd(Wd, full_matrices=False)
    Ua, Sa, Vha = torch.linalg.svd(Wd @ Ch, full_matrices=False)
    svd[n] = dict(plain=(U, S, Vh), aware=(Ua, Sa, Vha))

def truncated(n, m, how):
    U, S, Vh = svd[n][how]
    Wm = (U[:, :m] * S[:m]) @ Vh[:m]
    return (Wm if how == 'plain' else Wm @ Cih).to(orig_w[n].dtype)

def set_w(n, Wt):
    with torch.no_grad():
        getattr(bank, n).weight.copy_(Wt)

base = loss_on(xv, yv, 'trained')
rows = []
for n in PROJ:
    out_d, in_d = orig_w[n].shape
    for how in ('plain', 'aware'):
        for m in RANKS:
            if m >= min(out_d, in_d):
                continue
            set_w(n, truncated(n, m, how))
            rows.append((n, how, m, m * (in_d + out_d), out_d * in_d, loss_on(xv, yv, f'{n} {how} {m}')))
            set_w(n, orig_w[n])
jrows = []
for m in JOINT:
    for n in PROJ:
        if m < min(orig_w[n].shape):
            set_w(n, truncated(n, m, 'aware'))
    jrows.append((m, loss_on(xv, yv, f'all three aware {m}')))
    for n in PROJ:
        set_w(n, orig_w[n])

ch = lambda v: 100 * (float(torch.tensor(v - base).exp()) - 1)
L = [f'projection SVD truncation, {FOLDER}', desc,
     f'input moment on {W} even windows, loss on {W} held-out odd windows of {2 * W} evenly spaced x 512 tokens, validation '
     f'{CS["VAL"]}; K = {bank.K}, rank = {bank.rank}; bank input {C.shape[0]} wide, {acc["n"]:,} input vectors',
     f'trained: loss {base:.4f}  ppl {float(torch.tensor(base).exp()):.2f}', '',
     'share of the squared singular values in the top m (plain | input-aware):']
for n in PROJ:
    S, Sa = svd[n]['plain'][1], svd[n]['aware'][1]
    e, ea = (S ** 2).cumsum(0) / (S ** 2).sum(), (Sa ** 2).cumsum(0) / (Sa ** 2).sum()
    L.append(f'  {n:<8} ' + '  '.join(f'm={m}: {100 * float(e[m - 1]):5.1f} | {100 * float(ea[m - 1]):5.1f}' for m in RANKS if m <= len(S)))
L += ['', 'one projection truncated to rank m (a bottleneck of width m has m * (in + out) weights):',
      f'  {"projection":<10} {"method":<6} {"m":>5} {"weights":>12} {"of dense":>9} {"loss":>8} {"ppl":>8} {"change":>9}']
for n, how, m, wb, wd, v in rows:
    L.append(f'  {n:<10} {how:<6} {m:5d} {wb:12,d} {100 * wb / wd:8.1f}% {v:8.4f} {float(torch.tensor(v).exp()):8.2f} {ch(v):+8.2f}%')
L += ['', 'all three projections truncated together (input-aware):']
for m, v in jrows:
    L.append(f'  m = {m:5d}: loss {v:.4f}  ppl {float(torch.tensor(v).exp()):.2f}  {ch(v):+.2f}%')
txt = '\n'.join(L); print(txt)
stem = OPTS.get('name') or FOLDER.split('_')[5]
(OUT / f'projection_svd_{stem}.txt').write_text(txt + '\n')
