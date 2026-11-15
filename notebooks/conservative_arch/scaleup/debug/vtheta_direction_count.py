#!/usr/bin/env python3
"""How many directions does each layer's V_theta precision need per token? (2026-11-14, minimum-well-count note)

    python3 vtheta_direction_count.py OUT FOLDER [ckpt=<file>] [windows=16] [threads=8] [val=<tokens.npy>] [name=<stem>]
                                      [s=4,6,8,12,16,20,24] [tol=0.5]

Writes OUT/direction_count_<stem>.txt. The bank's precision term B_k B_k^T (h - mu_k) pushes the state along the R
columns of each well's B_k, so a bank of K wells can push along at most K * R directions at one token. The question is
how many of them a layer uses. Per token, stack the wells' factors weighted by their context weights,
C = [sqrt(w_k) B_k] (d x K*R), and take the top s left singular vectors U_s of C. On one layer at a time, every well's
B_k is replaced by U_s U_s^T B_k, rescaled to its own trained norm (as shared_oracle's 'sub8' in
vtheta_precision_ablation.py, with s swept), and the loss is read on the validation windows. r*(layer) = the smallest s
whose loss is within tol% (perplexity) of the trained model's. Also reported: the median share of sum_k w_k ||B_k||_F^2
in the top s directions.

The weights are the context weights w_k, not the force weights g_k = w_k * occupancy: the bank's context components do
not see h. Removal from a trained model overstates what a model trained with fewer directions needs, so r* is an upper
bound on the direction budget K * R. FOLDER and the build are as in vtheta_well_probe.py.
"""
import sys
from pathlib import Path

import torch

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
W = int(OPTS.get('windows', 16)); TOL = float(OPTS.get('tol', 0.5))
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))

# copy_split's own Cell-0-from-tag build and validation windows, without its command dispatch (as vtheta_well_probe.py)
src = (DBG / 'copy_split.py').read_text()
src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', f'windows={W}', f"threads={OPTS.get('threads', 8)}"] \
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
K, d, R = bank.K, VT.d, bank.rank
assert R > 0, 'rank-0 bank: no B to project'
SS = [s for s in (int(v) for v in OPTS.get('s', '4,6,8,12,16,20,24').split(',')) if s < min(d, K * R)]

cur = {'layer': 0}
orig_step = model._layer_step_langevin
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    cur['layer'] = int(layer_idx)
    return orig_step(h, h_prev, m_b, gamma, dt_, layer_idx)
model._layer_step_langevin = step

mode = {'s': None, 'layer': None}
ENERGY = {}

@torch.no_grad()
def project(B, w, s):
    lead = B.shape[:-3]
    Bf = B.detach().float().reshape(-1, K, d, R)
    wf = w.detach().float().reshape(-1, K)
    C = (wf.clamp(min=0).sqrt()[:, :, None, None] * Bf).permute(0, 2, 1, 3).reshape(-1, d, K * R)
    U, S, _ = torch.linalg.svd(C, full_matrices=False)
    e = (S ** 2).cumsum(-1) / (S ** 2).sum(-1, keepdim=True).clamp(min=1e-30)
    ENERGY.setdefault(cur['layer'], []).append(e)
    Us = U[:, :, :s]
    Bp = torch.einsum('nds,nes,nker->nkdr', Us, Us, Bf)
    nrm = Bf.flatten(-2).norm(dim=-1)
    Bn = Bp * (nrm / Bp.flatten(-2).norm(dim=-1).clamp(min=1e-30))[:, :, None, None]
    return Bn.reshape(*lead, K, d, R).to(B.dtype)

orig_comp = bank._components
def comp(xi):
    mu, a, w, B = orig_comp(xi)
    if mode['s'] is not None and cur['layer'] == mode['layer']:
        B = project(B, w, mode['s'])
    return mu, a, w, B
bank._components = comp

x, y = CS['windows']()
def loss_on(label):
    nll = []
    for i in range(0, len(x), 4):
        with torch.enable_grad():
            lg = model(torch.from_numpy(x[i:i + 4]))[0].detach()
        nll.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), torch.from_numpy(y[i:i + 4]).reshape(-1)))
    print(f'{label}: done', flush=True)
    return float(torch.stack(nll).mean())

base = loss_on('trained')
n_layers = model.cfg.L
rows = {}
for l in range(n_layers):
    for s in SS:
        mode.update(s=s, layer=l)
        rows[(l, s)] = loss_on(f'layer {l}, s = {s}')
mode.update(s=None, layer=None)
bank._components = orig_comp

ch = lambda v: 100 * (float(torch.tensor(v - base).exp()) - 1)
L = [f'direction count, {FOLDER}', desc,
     f'{W} evenly spaced validation windows x 512 tokens, validation {CS["VAL"]}; K = {K}, rank = {R}, K * R = {K * R}; '
     f'tolerance {TOL}% in perplexity',
     f'trained: loss {base:.4f}  ppl {float(torch.tensor(base).exp()):.2f}', '',
     'every well\'s B_k projected onto the top s shared directions of C = [sqrt(w_k) B_k] per token, on one layer:',
     f'  {"layer":>5} {"s":>4} {"loss":>8} {"ppl":>8} {"change":>9}  {"energy in top s (median, p10)":>30}']
rstar = {}
for l in range(n_layers):
    e = torch.cat(ENERGY[l]) if l in ENERGY else None
    for s in SS:
        v = rows[(l, s)]
        en = f'{100 * float(e[:, s - 1].median()):5.1f}%  {100 * float(e[:, s - 1].quantile(0.1)):5.1f}%' if e is not None else ''
        L.append(f'  {l:5d} {s:4d} {v:8.4f} {float(torch.tensor(v).exp()):8.2f} {ch(v):+8.2f}%  {en:>30}')
        if l not in rstar and ch(v) <= TOL:
            rstar[l] = s
L += ['', 'r* (smallest s within tolerance; "> s_max" if none): '
      + '  '.join(f'layer {l}: {rstar.get(l, f"> {SS[-1]}")}' for l in range(n_layers))]
txt = '\n'.join(L); print(txt)
stem = OPTS.get('name') or FOLDER.split('_')[5]
(OUT / f'direction_count_{stem}.txt').write_text(txt + '\n')
