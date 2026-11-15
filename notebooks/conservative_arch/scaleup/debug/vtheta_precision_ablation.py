#!/usr/bin/env python3
"""Does V_theta use its precision as a function of the context? Replace it by a fixed per-well average and read the loss (2026-11-13).

    python3 vtheta_precision_ablation.py OUT FOLDER [ckpt=<file>] [windows=16] [threads=8] [val=<tokens.npy>] [name=<stem>]
                                         [weight=g|none]

Writes OUT/precision_ablation_<stem>.txt. The bank's precision is P_k = diag(a_k) + B_k B_k^T, with a_k and B_k computed
from the context channels by a_proj and B_proj (5.9M and 23.6M parameters at K = 8, rank 4). The question is whether a
context-free precision (one learned a_k and B_k per well) would do. Without retraining:

  1. Estimation, on the even windows of 2 x windows evenly spaced ones: per layer and well, the weighted means
     M_k = E[B_k B_k^T] and abar_k = E[a_k], weighted by g_k = w_k exp(-1/2 (h - mu_k)^T P_k (h - mu_k)), the weight
     the token gives the well (weight=none: every token alike). Bbar_k = the rank-r part of M_k (top r eigenvectors
     times the root of their eigenvalues), so Bbar_k Bbar_k^T is the best rank-r fit to M_k. Its squared Frobenius
     norm is the top-r share of E||B_k||_F^2, so where B_k's orientation varies with context it is also weaker than
     the trained factor; 'trained norm' rescales it to sqrt E||B_k||_F^2 (inside the PRECISION_LR_MAX cap) so that
     only the orientation is fixed.
  2. Evaluation, on the odd windows: the loss of the trained model, then with Bbar_k (as fitted, or at the trained
     norm) in place of B_k(xi), with B_k = 0 (diagonal precision only, for scale), with one Bbar_k per well pooled over
     the layers (what a single context-free B shared by the layers would be), with abar_k in place of a_k(xi), with
     both, with every well given the B_k(xi) of the layer's strongest well (by force share: what one context-dependent
     B shared by the wells would be), with the per-token shared directions of all wells' B_k (shared_oracle: one B for
     all wells, or each B_k projected onto a shared 4 or 8 dimensional subspace), and each replacement on one layer
     only. mu_k and w_k stay context-dependent throughout. conds=shared runs only the shared-B conditions.

Also reported per layer and well: the share of the force weight sum g_k; the share of trace(M_k) in its top r
eigenvalues (100% if B_k B_k^T did not vary with context); and the share of the second moment E||B_k B_k^T||_F^2
carried by the mean, ||M_k||_F^2 / E||B_k B_k^T||_F^2 (and the same for a_k), i.e. 1 minus the variance across
tokens relative to the second moment: 100% means the shape does not depend on the context at all.

FOLDER and the build are as in vtheta_well_probe.py (copy_split.ladder_model rebuilds Cell 0 from the tag).
A small change in the loss says the trained model barely uses context-dependent precision, which supports a
context-free twin; a large one says it uses it as trained, which a twin trained from scratch may or may not recover.
"""
import sys
from pathlib import Path

import torch

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
W = int(OPTS.get('windows', 16)); WEIGHT = OPTS.get('weight', 'g')
assert WEIGHT in ('g', 'none'), WEIGHT
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
K, d, R = bank.K, VT.d, bank.rank
assert R > 0, 'rank-0 bank: no B to ablate'

cur = {'layer': 0, 'lr': False}
orig_step = model._layer_step_langevin
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    cur.update(layer=int(layer_idx), lr=False)
    return orig_step(h, h_prev, m_b, gamma, dt_, layer_idx)
model._layer_step_langevin = step

# -- 1. estimation: weighted moments of B B^T and a per layer and well (first harmonic call of each layer step, as the probe)
acc = {}
orig_lr = bank.harmonic_terms_lowrank
def lr_acc(xi, h, *, comps=None):
    out = orig_lr(xi, h, comps=comps)
    if not cur['lr']:
        cur['lr'] = True
        with torch.no_grad():
            mu, a, w, B = (c.detach().float() for c in (comps if comps is not None else bank._components(xi)))
            diff = h.detach().float().unsqueeze(-2) - mu
            quad = (a * diff * diff).sum(-1) + (torch.einsum('...kd,...kdr->...kr', diff, B) ** 2).sum(-1)
            g = (w * torch.exp(-0.5 * quad)).reshape(-1, K)
            wt = g if WEIGHT == 'g' else torch.ones_like(g)
            B = B.reshape(-1, K, d, R); a = a.reshape(-1, K, d)
            BtB = torch.einsum('nkdr,nkds->nkrs', B, B)                      # ||B B^T||_F = ||B^T B||_F
            r = acc.setdefault(cur['layer'], {k: 0.0 for k in ('S', 'G', 'M', 'A', 'QB', 'QA')})
            r['S'] = r['S'] + wt.sum(0).double()
            r['G'] = r['G'] + g.sum(0).double()
            r['M'] = r['M'] + torch.einsum('nk,nkdr,nker->kde', wt, B, B).double()
            r['A'] = r['A'] + torch.einsum('nk,nkd->kd', wt, a).double()
            r['QB'] = r['QB'] + (wt * (BtB ** 2).sum((-2, -1))).sum(0).double()
            r['QA'] = r['QA'] + (wt * (a ** 2).sum(-1)).sum(0).double()
    return out

x, y = CS['windows']()
xe, ye, xv, yv = x[0::2], y[0::2], x[1::2], y[1::2]

def loss_on(xs, ys, label):
    nll = []
    for i in range(0, len(xs), 4):
        with torch.enable_grad():
            lg = model(torch.from_numpy(xs[i:i + 4]))[0].detach()
        nll.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), torch.from_numpy(ys[i:i + 4]).reshape(-1)))
        print(f'{label}: windows {i + 4}/{len(xs)}', flush=True)
    return float(torch.stack(nll).mean())

bank.harmonic_terms_lowrank = lr_acc
loss_est = loss_on(xe, ye, 'estimation')
bank.harmonic_terms_lowrank = orig_lr

def rank_r(M):
    """Best rank-R factor of a symmetric PSD (K, d, d) stack: V_R sqrt(lam_R), plus the top-R share of the trace."""
    lam, V = torch.linalg.eigh(M)
    lam = lam.clamp(min=0)
    top = lam[..., -R:]
    return (V[..., -R:] * top.sqrt().unsqueeze(-2)).float(), (top.sum(-1) / lam.sum(-1).clamp(min=1e-30))

fixed, stats = {}, {}
for l in sorted(acc):
    r = acc[l]; S = r['S'].clamp(min=1e-30)
    M = r['M'] / S[:, None, None]; abar = (r['A'] / S[:, None]).float()
    Bbar, topshare = rank_r(M)
    fro = M.diagonal(dim1=-2, dim2=-1).sum(-1).sqrt()                        # sqrt E||B_k||_F^2
    # the same directions at the trained mean size: the rank-R fit keeps only the top-R share of the trace, so on its
    # own it also weakens the extra stiffness; this variant separates orientation from size
    Bnorm = Bbar * (fro / Bbar.flatten(-2).norm(dim=-1).clamp(min=1e-30).double()).float()[:, None, None]
    fixed[l] = dict(fixed=Bbar, fixed_norm=Bnorm, a=abar)
    stats[l] = dict(gshare=(r['G'] / r['G'].sum()), topshare=topshare,
                    mshareB=(M ** 2).sum((-2, -1)) / (r['QB'] / S).clamp(min=1e-30),
                    mshareA=(r['A'] / S[:, None]).pow(2).sum(-1) / (r['QA'] / S).clamp(min=1e-30), froB=fro)
pooledM = sum(acc[l]['M'] for l in acc) / sum(acc[l]['S'] for l in acc).clamp(min=1e-30)[:, None, None]
Bpool, toppool = rank_r(pooledM)
LAYERS = sorted(acc)
KSTAR = {l: int(stats[l]['gshare'].argmax()) for l in LAYERS}   # the well with the largest force share

# -- 2. evaluation on the held-out (odd) windows. A condition: how B is replaced and on which layers, which layers get abar
mode = {'B': None, 'B_layers': (), 'a_layers': ()}
ENERGY = {}

@torch.no_grad()
def shared_oracle(B, w, how):
    """Per token, the directions the wells share: C = [sqrt(w_k) B_k] (d x K*R), weighted by the context weights w_k
    (the force weight g_k needs h, which the bank's context components do not see). U_s = the top s left singular
    vectors of C. 'oracle': one B for every well, U_R S_R normalised; 'sub4'/'sub8': each well keeps its own B_k
    projected onto the shared s = 4 / 8 dimensional subspace. Every well's factor is rescaled to its own trained
    norm. These are not maps the trained model can produce: they bound what one shared B could carry."""
    lead = B.shape[:-3]
    Bf = B.detach().float().reshape(-1, K, d, R)
    wf = w.detach().float().reshape(-1, K)
    C = (wf.clamp(min=0).sqrt()[:, :, None, None] * Bf).permute(0, 2, 1, 3).reshape(-1, d, K * R)
    U, S, _ = torch.linalg.svd(C, full_matrices=False)
    s = 8 if how == 'sub8' else R
    e = ((S[:, :s] ** 2).sum(-1) / (S ** 2).sum(-1).clamp(min=1e-30))
    ENERGY.setdefault((cur['layer'], s), []).append(e)
    nrm = Bf.flatten(-2).norm(dim=-1)                                        # (N, K): each well's trained ||B_k||_F
    if how == 'oracle':
        Bs = U[:, :, :R] * (S[:, :R] / S[:, :R].norm(dim=-1, keepdim=True).clamp(min=1e-30))[:, None, :]
        Bn = Bs[:, None] * nrm[:, :, None, None]
    else:
        Us = U[:, :, :s]
        Bp = torch.einsum('nds,nes,nker->nkdr', Us, Us, Bf)
        Bn = Bp * (nrm / Bp.flatten(-2).norm(dim=-1).clamp(min=1e-30))[:, :, None, None]
    return Bn.reshape(*lead, K, d, R).to(B.dtype)
orig_comp = bank._components
def comp(xi):
    mu, a, w, B = orig_comp(xi)
    l = cur['layer']
    if l in mode['B_layers']:
        if mode['B'] == 'zero':
            B = torch.zeros_like(B)
        elif mode['B'] == 'shared':                     # every well takes the layer's strongest well's B(xi)
            ks = KSTAR[l]
            B = B[..., ks:ks + 1, :, :].expand_as(B)
        elif mode['B'] in ('oracle', 'sub4', 'sub8'):
            B = shared_oracle(B, w, mode['B'])
        elif mode['B'] == 'pooled':
            B = Bpool.to(B.dtype).expand_as(B)
        else:
            B = fixed[l][mode['B']].to(B.dtype).expand_as(B)
    if l in mode['a_layers']:
        a = fixed[l]['a'].to(a.dtype).expand_as(a)
    return mu, a, w, B
bank._components = comp

ALL = tuple(LAYERS)
CONDS = [('trained', None, (), ()),
         ('B fixed per layer', 'fixed', ALL, ()),
         ('B fixed, trained norm', 'fixed_norm', ALL, ()),
         ('B zero (diagonal only)', 'zero', ALL, ()),
         ('B fixed, pooled over layers', 'pooled', ALL, ()),
         ('B shared from strongest well', 'shared', ALL, ()),
         ('a fixed per layer', None, (), ALL),
         ('a and B fixed per layer', 'fixed', ALL, ALL)]
for l in LAYERS:
    CONDS += [(f'B fixed, layer {l} only', 'fixed', (l,), ()), (f'B fixed norm, layer {l} only', 'fixed_norm', (l,), ()),
              (f'a fixed, layer {l} only', None, (), (l,)), (f'B shared, layer {l} only', 'shared', (l,), ())]
for l in LAYERS:
    CONDS += [(f'B one shared (oracle), layer {l} only', 'oracle', (l,), ()),
              (f'B in shared 4-dim subspace, layer {l}', 'sub4', (l,), ()),
              (f'B in shared 8-dim subspace, layer {l}', 'sub8', (l,), ())]
CONDS += [('B one shared (oracle)', 'oracle', ALL, ()), ('B in shared 4-dim subspace', 'sub4', ALL, ()),
          ('B in shared 8-dim subspace', 'sub8', ALL, ())]
if OPTS.get('conds') == 'shared':                    # only the trained model and the shared-B conditions
    CONDS = [c for c in CONDS if c[1] in (None, 'shared', 'oracle', 'sub4', 'sub8') and not c[3]]
res = []
for label, mb, bl, al in CONDS:
    mode.update(B=mb, B_layers=bl, a_layers=al)
    res.append((label, loss_on(xv, yv, label)))
bank._components = orig_comp

base = res[0][1]
L = [f'precision ablation, {FOLDER}', desc,
     f'{W} estimation windows (even) and {W} evaluation windows (odd) of {2 * W} evenly spaced x 512 tokens, '
     f'validation {CS["VAL"]}; K = {K}, rank = {R}, PRECISION_LR_MAX = {bank._precision_lr_max}; '
     f'moments weighted by {"g_k (the force weight)" if WEIGHT == "g" else "nothing (every token alike)"}',
     f'estimation windows, trained model: loss {loss_est:.4f}  ppl {torch.tensor(loss_est).exp():.2f}', '',
     'evaluation windows (mu_k and w_k context-dependent throughout):',
     f'  {"condition":<40} {"loss":>8} {"ppl":>8} {"change":>9}']
for label, v in res:
    L.append(f'  {label:<40} {v:8.4f} {float(torch.tensor(v).exp()):8.2f} {100 * (float(torch.tensor(v - base).exp()) - 1):+7.2f}%')
L += ['', 'per layer and well: share of the force weight; top-R share of trace E[B B^T]; mean share of the second moment '
      'for B B^T and for a (100% = no dependence on the context); sqrt E||B||_F^2']
for l in sorted(stats):
    s = stats[l]
    L.append(f'  layer {l}:')
    for name, key, fmt in (('force share %', 'gshare', 100), ('top-R share %', 'topshare', 100), ('mean share B %', 'mshareB', 100),
                           ('mean share a %', 'mshareA', 100), ('sqrt E|B|_F^2', 'froB', 1)):
        L.append(f'    {name:<15} ' + ' '.join(f'{fmt * float(v):6.1f}' if fmt == 100 else f'{float(v):6.3f}' for v in s[key]))
for (l, sdim), v in sorted(ENERGY.items()):
    e = torch.cat(v)
    L.append(f'  layer {l}: share of sum_k w_k ||B_k||_F^2 in the top {sdim} shared directions per token, median {100 * float(e.median()):5.1f}%, '
             f'p10 {100 * float(e.quantile(0.1)):5.1f}%')
L.append(f'  pooled over layers, top-R share %: ' + ' '.join(f'{100 * float(v):6.1f}' for v in toppool))
txt = '\n'.join(L); print(txt)
stem = OPTS.get('name') or FOLDER.split('_')[5]
(OUT / f'precision_ablation_{stem}{"" if WEIGHT == "g" else "_unweighted"}.txt').write_text(txt + '\n')
