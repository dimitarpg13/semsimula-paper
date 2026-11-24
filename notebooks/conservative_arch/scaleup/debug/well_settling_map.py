#!/usr/bin/env python3
"""Well settling map: the admissible-region quantities of companion_notes/semantic_templates/
Admissible_Regions_for_Template_Search.md on the checkpoints of one ladder run (2026-11-24).

    python3 well_settling_map.py OUT FOLDER [ckpts=step500,step22500,best] [windows=16] [threads=8]
                                 [val=<tokens.npy>] [name=<stem>]

Every checkpoint is run on the same validation windows. Per layer, token and well it records the
frame-free quantities of the note: the squared well radius q_k = (h - mu_k)^T P_k (h - mu_k), the force
weight g_k = w_k exp(-q_k / 2) and the dominance share pi_k = g_k / sum_j g_j, the stiff coordinates
u_k = B_k^T (h - mu_k) (rank r), and the context spread of the well's centre in the well metric,
c_k(t) = (mu_k(xi_t) - mean_t mu_k)^T P_k(xi_t) (mu_k(xi_t) - mean_t mu_k). Between consecutive checkpoints
it then reports, per well: the Jaccard overlap of the token sets the well dominates, the 1-D Wasserstein
drift of the radial law r_k = sqrt(q_k) (normalised by the later median), and the Spearman correlation
over the same tokens of r_k and of |u_k|.

Writes OUT/settling_<stem>.txt (report), .json (the per-well tables) and .npz (per checkpoint, layer and
token: q, pi, |u|, c, and the dominant well).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))
OUT.mkdir(parents=True, exist_ok=True)

# copy_split's own Cell-0-from-tag build and validation windows, without its command dispatch (as vtheta_well_probe.py)
src = (DBG / 'copy_split.py').read_text()
src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', f"windows={OPTS.get('windows', 16)}", f"threads={OPTS.get('threads', 8)}"] \
    + ([f"val={OPTS['val']}"] if 'val' in OPTS else [])
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)

import verify_pm_switch as P
_build = P.build; got = {}
def build(*a, **k):
    r = _build(*a, **k); got['model'], got['g'] = r[0], r[1]; return r
P.build = build

CK_DIR = Path.home() / 'Downloads' / FOLDER / 'checkpoints'
def ckpt_file(tag):
    if tag == 'best':
        c = [p.name for p in CK_DIR.glob('fock_cfc_*_best.pt') if '_step' not in p.name]
    else:
        c = [p.name for p in CK_DIR.glob(f'fock_cfc_*_{tag}_best.pt')]
    assert len(c) == 1, (tag, c)
    return c[0]

CKPTS = OPTS.get('ckpts', 'step500,step22500,best').split(',')
x, y = CS['windows']()
N = x.shape[0] * x.shape[1]


def collect(tag):
    """One checkpoint on the windows: per layer the bank's components at every token and the state h."""
    _, desc = CS['ladder_model'](FOLDER, ckpt_file(tag))
    model = got['model']
    VT = model.V_theta
    bank = VT.bank.bank if hasattr(VT.bank, 'bank') else VT.bank
    rec = {}; cur = {'layer': 0, 'lr': False}
    orig_step = model._layer_step_langevin
    def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
        cur.update(layer=int(layer_idx), lr=False)
        return orig_step(h, h_prev, m_b, gamma, dt_, layer_idx)
    orig_lr = bank.harmonic_terms_lowrank
    def lr(xi, h, *, comps=None):
        out = orig_lr(xi, h, comps=comps)
        if not cur['lr']:                                # the first evaluation of the layer: at h, not at h_mid
            cur['lr'] = True
            with torch.no_grad():
                mu, a, w, B = (c.detach().float() for c in (comps if comps is not None else bank._components(xi)))
                r = rec.setdefault(cur['layer'], {})
                for k_, v in dict(h=h.detach().float().reshape(-1, h.shape[-1]), mu=mu.reshape(-1, *mu.shape[-2:]),
                                  a=a.reshape(-1, *a.shape[-2:]), w=w.reshape(-1, w.shape[-1]),
                                  B=B.reshape(-1, *B.shape[-3:])).items():
                    r.setdefault(k_, []).append(v)
        return out
    model._layer_step_langevin = step; bank.harmonic_terms_lowrank = lr
    nll = []
    for i in range(0, len(x), 4):
        with torch.enable_grad():
            lg = model(torch.from_numpy(x[i:i + 4]))[0].detach()
        nll.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), torch.from_numpy(y[i:i + 4]).reshape(-1)))
        print(f'  {tag}: windows {i + 4}/{len(x)}', flush=True)
    out = {}
    for l in sorted(rec):
        r = {k_: torch.cat(v) for k_, v in rec[l].items()}
        h, mu, a, w, B = r['h'], r['mu'], r['a'], r['w'], r['B']          # (N,d) (N,K,d) (N,K,d) (N,K) (N,K,d,r)
        diff = h.unsqueeze(1) - mu
        u = torch.einsum('nkd,nkdr->nkr', diff, B)                          # stiff coordinates
        q = (a * diff * diff).sum(-1) + (u * u).sum(-1)                     # squared well radius
        g = w * torch.exp(-0.5 * q); pi = g / g.sum(-1, keepdim=True).clamp_min(1e-30)
        mbar = mu.mean(0, keepdim=True); dm = mu - mbar
        c = (a * dm * dm).sum(-1) + (torch.einsum('nkd,nkdr->nkr', dm, B) ** 2).sum(-1)   # context spread, well metric
        ceu = (dm * dm).sum(-1) / (mbar * mbar).sum(-1).clamp_min(1e-30)                # Euclidean spread / |mean centre|^2
        out[l] = dict(q=q.numpy(), pi=pi.numpy(), un=u.norm(dim=-1).numpy(), c=c.numpy(), ceu=ceu.numpy(),
                      kstar=pi.argmax(-1).numpy(), hn=h.norm(dim=-1).numpy(), w=w.numpy(), g=g.numpy())
    del model; got.clear()
    return out, desc, float(torch.stack(nll).mean().exp())


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64); rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def w1(a, b):
    return float(np.abs(np.sort(a) - np.sort(b)).mean())


data = {}; descs = {}; ppls = {}
for tag in CKPTS:
    print(f'checkpoint {tag}', flush=True)
    data[tag], descs[tag], ppls[tag] = collect(tag)
    print(f'  {tag}: {descs[tag]}  ppl {ppls[tag]:.2f}', flush=True)

layers = sorted(data[CKPTS[0]]); K = data[CKPTS[0]][layers[0]]['q'].shape[1]
L = [f'{FOLDER}', f'{len(x)} validation windows x {x.shape[1]} ({Path(CS["VAL"]).name}); K = {K}; checkpoints {CKPTS}']
for tag in CKPTS:
    L.append(f'  {tag}: {descs[tag]}  ppl {ppls[tag]:.2f}')
J = {}
med = lambda v: float(np.median(v)) if v.size else float('nan')
for l in layers:
    L.append(f'\n=== layer {l} ===')
    L.append('per checkpoint and well: share of tokens the well dominates (k* = argmax pi); median pi on them; median radius r = sqrt(q) '
             'on them; median |u|; context spread s = sqrt(median c) in the well metric; Euclidean spread sqrt(median ceu) / |mean centre|')
    for tag in CKPTS:
        D = data[tag][l]; rows = []
        for k in range(K):
            on = D['kstar'] == k
            rows.append(dict(k=k, share=float(on.mean()), pi=med(D['pi'][on, k]), r=med(np.sqrt(D['q'][on, k])),
                             un=med(D['un'][on, k]), s=float(np.sqrt(np.median(D['c'][:, k]))),
                             s_eu=float(np.sqrt(np.median(D['ceu'][:, k]))), r_all=med(np.sqrt(D['q'][:, k]))))
        J[f'L{l}_{tag}'] = rows
        L.append(f'  {tag:10s}' + ' '.join(f'| k{k}: share {r["share"]:5.3f} pi {r["pi"]:5.3f} r {r["r"]:6.2f} |u| {r["un"]:5.2f} s {r["s"]:6.2f} s_eu {r["s_eu"]:5.2f}'
                                          for k, r in enumerate(rows)))
    L.append('between consecutive checkpoints, per well: Jaccard of the dominated token sets; W1 drift of the radial law r_k over all tokens '
             '/ later median; Spearman over the same tokens of r_k and of |u_k|')
    for t0, t1 in zip(CKPTS[:-1], CKPTS[1:]):
        D0, D1 = data[t0][l], data[t1][l]; rows = []
        for k in range(K):
            s0, s1 = D0['kstar'] == k, D1['kstar'] == k
            uni = (s0 | s1).sum()
            r0, r1 = np.sqrt(D0['q'][:, k]), np.sqrt(D1['q'][:, k])
            rows.append(dict(k=k, jaccard=float((s0 & s1).sum() / uni) if uni else float('nan'),
                             w1=w1(r0, r1) / max(float(np.median(r1)), 1e-12),
                             rho_r=spearman(r0, r1), rho_u=spearman(D0['un'][:, k], D1['un'][:, k])))
        J[f'L{l}_{t0}_to_{t1}'] = rows
        L.append(f'  {t0} -> {t1}: ' + ' '.join(f'| k{k}: J {r["jaccard"]:5.3f} W1 {r["w1"]:6.3f} rho_r {r["rho_r"]:+5.3f} rho_u {r["rho_u"]:+5.3f}'
                                                for k, r in enumerate(rows)))
    L.append('frame check per checkpoint: median |h|; median of the largest pi over tokens')
    for tag in CKPTS:
        D = data[tag][l]
        L.append(f'  {tag:10s} |h| {med(D["hn"]):8.3f}  max pi {med(D["pi"].max(-1)):5.3f}')

txt = '\n'.join(L); print(txt)
stem = OPTS.get('name', 'run')
(OUT / f'settling_{stem}.txt').write_text(txt + '\n')
(OUT / f'settling_{stem}.json').write_text(json.dumps(dict(folder=FOLDER, ckpts=CKPTS, desc=descs, ppl=ppls, tables=J), indent=1))
np.savez_compressed(OUT / f'settling_{stem}.npz', **{f'{tag}_L{l}_{k_}': v for tag in CKPTS for l in layers
                                                     for k_, v in data[tag][l].items() if k_ in ('q', 'pi', 'un', 'c', 'kstar')})
