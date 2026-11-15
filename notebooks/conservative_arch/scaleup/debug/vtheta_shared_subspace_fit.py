#!/usr/bin/env python3
"""Can a shared-subspace map replace B_proj? Fit one to the trained bank, swap it in, read the loss (2026-11-14).

    python3 vtheta_shared_subspace_fit.py OUT FOLDER [ckpt=<file>] [val=<tokens.npy>] train=<tokens.npy> [s=8]
                                          [fit_windows=64] [iters=1500] [batch=2048] [lr=1e-3] [frob=0.1]
                                          [windows=16] [threads=8] [seed=0] [name=<stem>]

Writes OUT/shared_subspace_<stem>_s<s>.txt. The trained bank makes each well's low-rank precision factor from the
context, B_k(xi) = cap(B_proj(xi))_k, a dense 1,920 -> K * d * r map (23.6M weights at K = 8, d = 384, r = 4).
The shared-subspace head makes, per token, one basis U(xi) (d x s) shared by the wells and a small weighting
C_k(xi) (s x r) per well, B^_k = cap(U C_k): two dense maps, 1,920 -> d * s and 1,920 -> K * s * r
(6.4M weights at s = 8). vtheta_precision_ablation.py's oracle (each B_k projected onto the top-s shared directions
of that token's B's) cost +121% at layer 0 and nothing at layer 1 at s = 8; this asks whether a map the model could
actually compute does as well. Without retraining the rest of the model:

  1. Collection, on fit_windows random windows of 512 tokens from train= (seeded): per layer and token, the bank
     input xi (with the layer's depth code), the state h and the force weights g_k = w_k occ_k, at the first
     harmonic call of each layer step (as vtheta_well_probe.py).
  2. Fit (Adam, both layers pooled, since the bank is shared): the head's low-rank force g_k B^_k B^_k^T (h - mu_k)
     against the trained one, g_k B_k B_k^T (h - mu_k), relative to the trained force's mean square, plus frob= times
     the mean over wells of ||B^_k B^_k^T - B_k B_k^T||_F^2, which also shapes the wells away from the visited states.
  3. Evaluation, on windows= evenly spaced validation windows: the trained model, and the head in place of B_proj
     on both layers, on layer 0 only and on layer 1 only. mu_k, a_k and w_k are the trained bank's throughout.
"""
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
W = int(OPTS.get('windows', 16)); SUB = int(OPTS.get('s', 8)); NFIT = int(OPTS.get('fit_windows', 64))
ITERS = int(OPTS.get('iters', 1500)); BATCH = int(OPTS.get('batch', 2048)); LR = float(OPTS.get('lr', 1e-3))
FROB = float(OPTS.get('frob', 0.1)); SEED = int(OPTS.get('seed', 0))
assert 'train' in OPTS, 'train=<tokens.npy> is required (the fit must not see the validation set)'
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
for p_ in model.parameters():
    p_.requires_grad_(False)

VT = model.V_theta
bank = VT.bank.bank if hasattr(VT.bank, 'bank') else VT.bank      # depth-conditioned -> joint -> mixture
K, d, R = bank.K, VT.d, bank.rank
IN = bank.B_proj.in_features

cur = {'layer': 0, 'lr': False}
orig_step = model._layer_step_langevin
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    cur.update(layer=int(layer_idx), lr=False)
    return orig_step(h, h_prev, m_b, gamma, dt_, layer_idx)
model._layer_step_langevin = step

def run(xs, ys, label):
    nll = []
    for i in range(0, len(xs), 4):
        with torch.enable_grad():
            lg = model(torch.from_numpy(xs[i:i + 4]))[0].detach()
        if ys is not None:
            nll.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), torch.from_numpy(ys[i:i + 4]).reshape(-1)))
        print(f'{label}: windows {i + 4}/{len(xs)}', flush=True)
    return float(torch.stack(nll).mean()) if nll else None

# -- 1. collection on training windows
col = {'xi': [], 'h': [], 'g': [], 'layer': []}
orig_lr = bank.harmonic_terms_lowrank
def lr_col(xi, h, *, comps=None):
    out = orig_lr(xi, h, comps=comps)
    if not cur['lr']:
        cur['lr'] = True
        with torch.no_grad():
            mu, a, w, B = (c.detach().float() for c in (comps if comps is not None else bank._components(xi)))
            diff = h.detach().float().unsqueeze(-2) - mu
            quad = (a * diff * diff).sum(-1) + (torch.einsum('...kd,...kdr->...kr', diff, B) ** 2).sum(-1)
            g = (w * torch.exp(-0.5 * quad)).reshape(-1, K)
            col['xi'].append(xi.detach().float().reshape(-1, IN)); col['h'].append(h.detach().float().reshape(-1, d))
            col['g'].append(g); col['layer'].append(torch.full((g.shape[0],), cur['layer'], dtype=torch.int8))
    return out
tr = np.load(OPTS['train'], mmap_mode='r')
T = CS['windows']()[0].shape[1]
st = np.sort(np.random.default_rng(SEED).integers(0, len(tr) - T - 1, NFIT))
xt = np.stack([np.asarray(tr[s:s + T], dtype=np.int64) for s in st])
bank.harmonic_terms_lowrank = lr_col
run(xt, None, 'collection')
bank.harmonic_terms_lowrank = orig_lr
XI, H, G, LAY = (torch.cat(col[k]) for k in ('xi', 'h', 'g', 'layer'))
N = XI.shape[0]
print(f'collected {N:,} (token, layer) inputs', flush=True)

# trained targets, in chunks: B_k, and the low-rank force g_k B_k B_k^T (h - mu_k)
with torch.no_grad():
    BT, FT, DF = [], [], []
    for i in range(0, N, 4096):
        xi = XI[i:i + 4096]
        B = bank._bound_lowrank(bank.B_proj(xi).view(-1, K, d, R))
        diff = H[i:i + 4096].unsqueeze(1) - bank.mu_proj(xi).view(-1, K, d)
        BT.append(B); DF.append(diff)
        FT.append(G[i:i + 4096, :, None] * torch.einsum('nkdr,nkr->nkd', B, torch.einsum('nkd,nkdr->nkr', diff, B)))
    BT, FT, DF = torch.cat(BT), torch.cat(FT), torch.cat(DF)
    FSCALE = float((FT ** 2).sum(-1).mean())

# -- 2. the head and its fit
class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.U = torch.nn.Linear(IN, d * SUB)
        self.C = torch.nn.Linear(IN, K * SUB * R)
    def forward(self, xi):
        lead = xi.shape[:-1]
        U = self.U(xi).view(*lead, d, SUB)
        C = self.C(xi).view(*lead, K, SUB, R)
        return bank._bound_lowrank(torch.einsum('...ds,...ksr->...kdr', U, C))

torch.manual_seed(SEED)
head = Head()
opt = torch.optim.Adam(head.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
gen = torch.Generator().manual_seed(SEED)
for it in range(ITERS):
    idx = torch.randint(0, N, (BATCH,), generator=gen)
    Bh = head(XI[idx])
    Fh = G[idx, :, None] * torch.einsum('nkdr,nkr->nkd', Bh, torch.einsum('nkd,nkdr->nkr', DF[idx], Bh))
    lf = ((Fh - FT[idx]) ** 2).sum(-1).mean() / FSCALE
    Bt = BT[idx]
    # ||Bh Bh^T - Bt Bt^T||_F^2 = ||Bh^T Bh||^2 + ||Bt^T Bt||^2 - 2 ||Bh^T Bt||^2 (r x r products)
    fr = ((torch.einsum('nkdr,nkds->nkrs', Bh, Bh) ** 2).sum((-2, -1)) + (torch.einsum('nkdr,nkds->nkrs', Bt, Bt) ** 2).sum((-2, -1))
          - 2 * (torch.einsum('nkdr,nkds->nkrs', Bh, Bt) ** 2).sum((-2, -1))).mean()
    loss = lf + FROB * fr
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 100 == 0 or it == ITERS - 1:
        print(f'fit {it:5d}/{ITERS}  force err {float(lf):.4f}  frob {float(fr):.4f}', flush=True)

with torch.no_grad():                                   # fit quality per layer, over all collected inputs
    fit = {}
    for l in sorted(set(LAY.tolist())):
        m = LAY == l; ef = 0.0; efr = 0.0; n = int(m.sum())
        for i in range(0, N, 4096):
            mm = m[i:i + 4096]
            if not mm.any():
                continue
            Bh = head(XI[i:i + 4096][mm]); Bt = BT[i:i + 4096][mm]
            Fh = G[i:i + 4096][mm][:, :, None] * torch.einsum('nkdr,nkr->nkd', Bh, torch.einsum('nkd,nkdr->nkr', DF[i:i + 4096][mm], Bh))
            ef += float(((Fh - FT[i:i + 4096][mm]) ** 2).sum()) ; efr += float(((FT[i:i + 4096][mm]) ** 2).sum())
        fit[l] = ef / max(efr, 1e-30)
del BT, FT, DF

# -- 3. evaluation on validation windows
mode = {'layers': ()}
orig_comp = bank._components
def comp(xi):
    mu, a, w, B = orig_comp(xi)
    if cur['layer'] in mode['layers']:
        with torch.no_grad():
            B = head(xi.float()).to(B.dtype)
    return mu, a, w, B
bank._components = comp
xv, yv = CS['windows']()
LAYERS = sorted(fit)
res = []
for label, ls in [('trained', ())] + [('head, both layers', tuple(LAYERS))] + [(f'head, layer {l} only', (l,)) for l in LAYERS]:
    mode['layers'] = ls
    res.append((label, run(xv, yv, label)))
bank._components = orig_comp

nh = sum(p_.numel() for p_ in head.parameters()); nb = sum(p_.numel() for p_ in bank.B_proj.parameters())
base = res[0][1]
L = [f'shared-subspace head in place of B_proj, s = {SUB}, {FOLDER}', desc,
     f'fit on {NFIT} random training windows x {T} from {OPTS["train"]} ({N:,} token-layer inputs; seed {SEED}), {ITERS} Adam steps '
     f'of {BATCH}, lr {LR} (cosine), frob weight {FROB}; scored on {W} evenly spaced validation windows from {CS["VAL"]}',
     f'head: U 1,920 -> {d} x {SUB}, C 1,920 -> {K} x {SUB} x {R}: {nh:,} weights against B_proj\'s {nb:,} ({100 * nh / nb:.1f}%)',
     'relative low-rank force error on the collected inputs (sum ||F^ - F||^2 / sum ||F||^2): '
     + '  '.join(f'layer {l} {100 * v:.1f}%' for l, v in fit.items()), '',
     f'  {"condition":<22} {"loss":>8} {"ppl":>8} {"change":>9}']
for label, v in res:
    L.append(f'  {label:<22} {v:8.4f} {float(np.exp(v)):8.2f} {100 * (float(np.exp(v - base)) - 1):+8.2f}%')
txt = '\n'.join(L); print(txt)
stem = OPTS.get('name') or FOLDER.split('_')[5]
(OUT / f'shared_subspace_{stem}_s{SUB}.txt').write_text(txt + '\n')
