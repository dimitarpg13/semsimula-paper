#!/usr/bin/env python3
"""Where are V_theta's wells relative to the states? Per layer, on validation windows (2026-10-28, protocol p-dead-wells).

    python3 vtheta_well_probe.py OUT FOLDER [ckpt=<file>] [windows=8] [threads=8] [val=<tokens.npy>] [name=<stem>]
                                 [ablate=<channels>] [keep=<wells> keep_layers=<layers>]

Writes OUT/well_probe_<stem>.txt (the report) and .npz (per token and layer: nearest-well ratio, occupancy;
per well: force-weight share).

FOLDER is a ladder run folder in ~/Downloads (its Cell 0 is rebuilt from the tag by copy_split.ladder_model);
val defaults to the code validation set. Reports, per layer:
  - omega*dt = sqrt(lambda_max(G G^T) / m) * dt_kick and its factors: G = [sqrt(g_k) B_k], g_k = w_k * occ_k,
    occ_k = exp(-1/2 (h-mu_k)^T P_k (h-mu_k)), ||B_k||_F^2 <= PRECISION_LR_MAX, sum_k w_k = W_SCALE, so
    lambda_max <= sum_k g_k ||B_k||_F^2; the learned mass m; the nearest well's distance and occupancy;
  - the context channels at the bank input (norms, cosines) and the bank weights reading each channel;
  - |mu_k| / |h| over all wells, the share of the force weight sum g_k by well, the nearest-well ratio's tail;
  - the split mu = W (xi - code_l) + W code_l + b of the layer's well centres (channels, depth code, bias);
  - the validation loss, optionally with channels zeroed at the bank input (ablate=) or wells removed (keep=,
    weight w_k = 0 without renormalisation, on the layers in keep_layers=).
"""
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))

# copy_split's own Cell-0-from-tag build and validation windows, without its command dispatch
src = (DBG / 'copy_split.py').read_text()
src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', f"windows={OPTS.get('windows', 8)}", f"threads={OPTS.get('threads', 8)}"] \
    + ([f"val={OPTS['val']}"] if 'val' in OPTS else [])
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)

import verify_pm_switch as P
_build = P.build; got = {}
def build(*a, **k):
    r = _build(*a, **k); got['model'], got['g'] = r[0], r[1]; return r
P.build = build
_, desc = CS['ladder_model'](FOLDER, OPTS.get('ckpt'))
model, g = got['model'], got['g']
print('model:', desc, flush=True)

VT = model.V_theta
bank = VT.bank.bank if hasattr(VT.bank, 'bank') else VT.bank      # depth-conditioned -> joint -> mixture
n_ctx, d = VT.n_ctx, VT.d
cap = bank._precision_lr_max
mod = sys.modules[type(model)._layer_step_langevin.__module__]

rec = {}; cur = {'layer': 0, 'ctx': False, 'lr': False, 'sub': False}
orig_step = model._layer_step_langevin
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    cur.update(layer=int(layer_idx), ctx=False, lr=False, sub=False)
    return orig_step(h, h_prev, m_b, gamma, dt_, layer_idx)
ABL = [int(c) for c in OPTS.get('ablate', '').split(',') if c != '']     # channels replaced by zero at the bank input
KEEP = [int(c) for c in OPTS.get('keep', '').split(',') if c != '']      # wells kept; the others get weight w_k = 0 (no renormalisation)
KEEP_L = [int(c) for c in OPTS.get('keep_layers', '0,1').split(',')]      # layers the mask applies to
if KEEP:
    _orig_comp = bank._components
    _drop = torch.tensor([k not in KEEP for k in range(bank.K)])
    def _comp(xi):
        mu, a, w, B = _orig_comp(xi)
        return (mu, a, w.masked_fill(_drop.to(w.device), 0.0), B) if cur['layer'] in KEEP_L else (mu, a, w, B)
    bank._components = _comp
orig_cc = bank.context_components
def cc(xi):
    if ABL:
        xi = xi.reshape(*xi.shape[:-1], n_ctx, d).clone(); xi[..., ABL, :] = 0; xi = xi.reshape(*xi.shape[:-2], n_ctx * d)
    out = orig_cc(xi)
    if not cur['ctx']:
        cur['ctx'] = True
        x = xi.detach().float().reshape(-1, n_ctx, d)
        r = rec.setdefault(cur['layer'], {}); r.setdefault('xi', []).append(x)
        Wm = bank.mu_proj.weight.detach().float().reshape(bank.K, d, n_ctx, d)      # well k, out dim, channel, in dim
        contrib = torch.einsum('kocd,ncd->nkco', Wm, x).norm(dim=-1).mean(1)        # (tokens, channel): |W_c xi_c| averaged over wells
        r.setdefault('mu_c', []).append(contrib)
    return out
orig_lr = bank.harmonic_terms_lowrank
def lr(xi, h, *, comps=None):
    out = orig_lr(xi, h, comps=comps)
    if not cur['lr']:
        cur['lr'] = True
        with torch.no_grad():
            mu, a, w, B = (c.detach().float() for c in (comps if comps is not None else bank._components(xi)))
            diff = h.detach().float().unsqueeze(-2) - mu
            quad = (a * diff * diff).sum(-1) + (torch.einsum('...kd,...kdr->...kr', diff, B) ** 2).sum(-1)
            occ = torch.exp(-0.5 * quad); gk = w * occ
            fro2 = (B * B).sum((-2, -1))
            G = (gk.sqrt()[..., None, None] * B).movedim(-3, -2).reshape(*gk.shape[:-1], d, -1)
            # float64: tokens outside every well give G = 0 exactly, where float32 eigh fails to converge
            lam = torch.linalg.eigvalsh((G.transpose(-2, -1) @ G).double())[..., -1].clamp(min=0).float()
            ks = gk.argmax(-1, keepdim=True)
            r0 = rec.setdefault(cur['layer'], {})
            r0.setdefault('hn', []).append(h.detach().float().norm(dim=-1).flatten())
            r0.setdefault('mun', []).append(mu.norm(dim=-1).mean(-1).flatten())
            r0.setdefault('dmin', []).append(diff.norm(dim=-1).min(-1).values.flatten())
            r0.setdefault('ratio', []).append((mu.norm(dim=-1) / h.detach().float().norm(dim=-1, keepdim=True)).flatten())   # |mu_k| / |h|, every well
            r0.setdefault('gw', []).append(gk.flatten())
            r0.setdefault('near_k', []).append(diff.norm(dim=-1).argmin(-1).flatten())
            r = rec.setdefault(cur['layer'], {})
            for k, v in dict(lam=lam, wmax=w.max(-1).values, w_star=w.gather(-1, ks)[..., 0],
                             occ_star=occ.gather(-1, ks)[..., 0], gsum=gk.sum(-1), fro2_star=fro2.gather(-1, ks)[..., 0],
                             conc=lam / (gk * fro2).sum(-1).clamp(min=1e-30),
                             neff=torch.exp(-(w.clamp(min=1e-30) * w.clamp(min=1e-30).log()).sum(-1)),
                             dist=diff.norm(dim=-1).gather(-1, ks)[..., 0]).items():
                r.setdefault(k, []).append(v.flatten())
    return out
orig_sub = mod.lowrank_damped_substep
def sub(h, v, U, kappa, f_lr, m, gamma, dt_):
    if not cur['sub']:
        cur['sub'] = True
        r = rec.setdefault(cur['layer'], {})
        r.setdefault('m', []).append(m.detach().float().flatten() if torch.is_tensor(m) else torch.tensor([float(m)]))
        r['dt_kick'] = 2.0 * float(dt_)
    return orig_sub(h, v, U, kappa, f_lr, m, gamma, dt_)

orig_csub = mod.lowrank_cfc_substep                 # arms without SR2 (F3.1, G2): same mass and step, no friction
def csub(h, v, U, kappa, f_lr, m, dt_):
    if not cur['sub']:
        cur['sub'] = True
        r = rec.setdefault(cur['layer'], {})
        r.setdefault('m', []).append(m.detach().float().flatten() if torch.is_tensor(m) else torch.tensor([float(m)]))
        r['dt_kick'] = 2.0 * float(dt_)
    return orig_csub(h, v, U, kappa, f_lr, m, dt_)

model._layer_step_langevin = step; bank.context_components = cc; bank.harmonic_terms_lowrank = lr
mod.lowrank_damped_substep = sub; mod.lowrank_cfc_substep = csub
x, y = CS['windows'](); nll = []
for i in range(0, len(x), 4):
    with torch.enable_grad():
        lg = model(torch.from_numpy(x[i:i + 4]))[0].detach()
    nll.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), torch.from_numpy(y[i:i + 4]).reshape(-1)))
    print(f'windows {i + 4}/{len(x)}', flush=True)

q = lambda t: f'{float(t.median()):8.3f}'
L = [f'ablate channels {ABL}: loss {float(torch.stack(nll).mean()):.4f}  ppl {float(torch.stack(nll).mean().exp()):.2f}\n{FOLDER}\n{desc}\n{len(x)} code validation windows x 512; cap PRECISION_LR_MAX = {cap}; n_ctx = {n_ctx}; '
     f'K = {bank.K}, rank = {bank.rank}\n',
     'medians per layer (well k* = the largest g_k at the token):',
     f'  {"layer":>5} {"omega*dt":>8} {"lam_max":>8} {"m":>8} {"w_max":>8} {"n_eff":>8} {"w(k*)":>8} {"occ(k*)":>8} '
     f'{"sum g":>8} {"fro2(k*)":>8} {"conc":>8} {"|h-mu|":>8} {"over 2":>7}']
for l in sorted(rec):
    r = {k: (torch.cat(v) if isinstance(v, list) and k != 'xi' else v) for k, v in rec[l].items()}
    m = r['m'] if r['m'].numel() == r['lam'].numel() else r['m'].expand_as(r['lam'])
    wdt = (r['lam'] / m.clamp(min=1e-30)).sqrt() * r['dt_kick']
    L.append(f'  {l:5d} {q(wdt)} {q(r["lam"])} {q(m)} {q(r["wmax"])} {q(r["neff"])} {q(r["w_star"])} {q(r["occ_star"])} '
             f'{q(r["gsum"])} {q(r["fro2_star"])} {q(r["conc"])} {q(r["dist"])} {100 * float((wdt > 2).float().mean()):6.1f}%')
L.append('\ncontext channels at the bank input (depth code added), per layer: norm median, cosine with the last channel')
for l in sorted(rec):
    xi = torch.cat(rec[l]['xi']); nrm = xi.norm(dim=-1)
    cos = torch.nn.functional.cosine_similarity(xi, xi[:, -1:, :].expand_as(xi), dim=-1)
    L.append(f'  layer {l}: norm ' + ' '.join(f'{float(v):7.2f}' for v in nrm.median(0).values)
             + ' | cos ' + ' '.join(f'{float(v):6.3f}' for v in cos.median(0).values))
L.append('\nwell centres per layer: median |h|, mean-over-wells |mu|, nearest-well distance; |W_c xi_c| of mu per channel')
for l in sorted(rec):
    r = rec[l]; mc = torch.cat(r['mu_c']).median(0).values
    L.append(f'  layer {l}: |h| {q(torch.cat(r["hn"]))} |mu| {q(torch.cat(r["mun"]))} nearest {q(torch.cat(r["dmin"]))} | per channel '
             + ' '.join(f'{float(v):7.2f}' for v in mc))
L.append('\n|mu_k| / |h| over all (token, well) pairs: quantiles; share above c; share of the force weight sum_k g_k carried by wells above c')
for l in sorted(rec):
    rt = torch.cat(rec[l]['ratio']); gw = torch.cat(rec[l]['gw']); s_ = rt[torch.randperm(rt.numel())[:1 << 20]]
    qs = ' '.join(f'{float(s_.quantile(p)):6.2f}' for p in (.05, .25, .5, .75, .95, .99))
    L.append(f'  layer {l}: p05/25/50/75/95/99 {qs} | ' + '  '.join(
        f'>{c}: {100 * float((rt > c).float().mean()):5.1f}% wells, {100 * float(gw[rt > c].sum() / gw.sum().clamp(min=1e-30)):5.1f}% force'
        for c in (2, 3, 4, 6)))
L.append('\nper well k = 0..K-1: share of the total force weight sum g_k; share of tokens whose nearest well it is')
for l in sorted(rec):
    gw = torch.cat(rec[l]['gw']).reshape(-1, bank.K); gsh = gw.sum(0) / gw.sum().clamp(min=1e-30)
    near = torch.cat(rec[l]['near_k']); nsh = torch.bincount(near, minlength=bank.K).float() / near.numel()
    L.append(f'  layer {l}: force ' + ' '.join(f'{100 * float(v):5.1f}' for v in gsh) + '  | nearest ' + ' '.join(f'{100 * float(v):5.1f}' for v in nsh))
L.append('\nnearest-well distance / |h| per token: quantiles; share of tokens above rho')
for l in sorted(rec):
    nr = torch.cat(rec[l]['dmin']) / torch.cat(rec[l]['hn']).clamp(min=1e-6)
    L.append(f'  layer {l}: p50/90/99/99.9/max ' + ' '.join(f'{float(nr.quantile(p)):6.2f}' for p in (.5, .9, .99, .999)) + f' {float(nr.max()):6.2f} | '
             + '  '.join(f'>{c}: {100 * float((nr > c).float().mean()):6.2f}%' for c in (1.5, 2, 2.5, 3)))
L.append('\nwell centres per layer split as mu = W (xi - code_l) + W code_l + b (median over tokens of the mean over wells of each norm; cos = alignment of the code term with the total)')
Wm_ = bank.mu_proj.weight.detach().float(); b_ = bank.mu_proj.bias.detach().float()
for l in sorted(rec):
    x_ = torch.cat(rec[l]['xi']).reshape(-1, n_ctx * d); code = VT.depth_code[l % VT.n_layers].detach().float().reshape(-1)
    tot = (x_ @ Wm_.T + b_).reshape(-1, bank.K, d); cterm = (code @ Wm_.T).reshape(bank.K, d)
    xterm = ((x_ - code) @ Wm_.T).reshape(-1, bank.K, d); bterm = b_.reshape(bank.K, d)
    cs_ = torch.nn.functional.cosine_similarity(tot, cterm.expand_as(tot), dim=-1).mean(-1)
    L.append(f'  layer {l}: |mu| {q(tot.norm(dim=-1).mean(-1))}  channels {q(xterm.norm(dim=-1).mean(-1))}  depth code {float(cterm.norm(dim=-1).mean()):8.3f}'
             f'  bias {float(bterm.norm(dim=-1).mean()):8.3f}  cos(code term, mu) {q(cs_)}  |code_l| {float(code.norm()):.2f}')
L.append('\nbank projections: Frobenius norm of the weight columns reading each channel')
for name in ('B_proj', 'mu_proj', 'a_proj', 'w_proj'):
    W = getattr(bank, name).weight.detach().float().reshape(-1, n_ctx, d)
    L.append(f'  {name:7s} ' + ' '.join(f'{float(W[:, c].norm()):8.3f}' for c in range(n_ctx))
             + f'   cos(last two blocks) {float(torch.nn.functional.cosine_similarity(W[:, -2].flatten(), W[:, -1].flatten(), dim=0)):+.3f}')
txt = '\n'.join(L); print(txt)
stem = OPTS.get('name') or f'{FOLDER.split("_")[5]}_{"fock" if "_norc_" not in FOLDER else "so"}'
(OUT / f'well_probe_{stem}.txt').write_text(txt + '\n')
# per token, per layer: nearest-well distance / |h|, the largest-g well's occupancy; per well: force-weight share
arrs = {}
for l in sorted(rec):
    arrs[f'L{l}_near_ratio'] = (torch.cat(rec[l]['dmin']) / torch.cat(rec[l]['hn']).clamp(min=1e-6)).numpy()
    arrs[f'L{l}_occ'] = torch.cat(rec[l]['occ_star']).numpy()
    gw = torch.cat(rec[l]['gw']).reshape(-1, bank.K); arrs[f'L{l}_force_share'] = (gw.sum(0) / gw.sum().clamp(min=1e-30)).numpy()
np.savez_compressed(OUT / f'well_probe_{stem}.npz', **arrs)
