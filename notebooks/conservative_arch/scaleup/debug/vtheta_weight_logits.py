#!/usr/bin/env python3
"""How alike are V_theta's well weights from layer to layer? (2026-11-03)

    python3 vtheta_weight_logits.py OUT FOLDER [ckpt=<file>] [windows=16] [threads=8] [name=<stem>] [val=<tokens.npy>] [sweep=1]

The joint bank's well weights are w_l = softmax(z_l), z_l = W_w x_l + b, x_l = [xi_l + c_l] (the context channels at
layer l plus that layer's depth code, concatenated); vtheta_well_probe.py's n_eff is exp(H(w_l)), per token. They
depend on the bank input only, not on the state h. For a token at layers i and j, with delta = z_i - z_j,

    |ln n_eff(i) - ln n_eff(j)| <= 1/2 * s_K * (max_k delta_k - min_k delta_k),

where s_K is the largest square root of the varentropy Var_p(-ln p) over the K-simplex (the gradient of
H(softmax z) is -p_k (ln p_k + H): it sums to zero, so a constant shift of delta is free, and its l1 norm is at most
the square root of the varentropy by Cauchy-Schwarz; the mean value theorem does the rest). s_K is found numerically.

Reports, on the same windows as vtheta_well_probe.py: per layer the bank input's norm, the logits' spread and the
median n_eff (it must reproduce the probe's); for every pair of layers the largest ratio of the actual change to the
bound (at most 1 if the bound holds), how tight it is, and the split of delta's spread into its context part
W_w (xi_i - xi_j) and its depth-code part W_w (c_i - c_j); and the scale test: layer 0's input rescaled to the deep
layers' median norm, with and without its depth code. sweep=1 adds, per layer, the median n_eff with every token's bank
input rescaled to norm r on a log grid (2026-11-03): the softmax temperature curve.
"""
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))
src = (DBG / 'copy_split.py').read_text()
src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', f"windows={OPTS.get('windows', 16)}", f"threads={OPTS.get('threads', 8)}"] \
    + ([f"val={OPTS['val']}"] if 'val' in OPTS else [])
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)

run, desc = CS['ladder_model'](FOLDER, OPTS.get('ckpt'))
import gc
model = next(o for o in gc.get_objects() if isinstance(o, torch.nn.Module) and hasattr(o, 'V_theta') and hasattr(o.V_theta, 'depth_code'))
vt = model.V_theta
bank = vt.bank.bank                      # the joint bank's single AnisotropicMixtureGaussianVTheta
L, K = model.cfg.L, bank.K

cap, cur = {}, {}
orig_shift = vt._shift
def shift(xis):
    g = vt._active_layer % vt.n_layers
    if g not in cur:                         # the first call per layer and forward (later calls repeat it)
        cur[g] = xis.detach().float().reshape(-1, *xis.shape[-2:])
    return orig_shift(xis)
vt._shift = shift

x, _ = CS['windows']()
for i in range(0, len(x), 4):
    cur.clear()
    run(x[i:i + 4])
    for g, v in cur.items():
        cap.setdefault(g, []).append(v)
vt._shift = orig_shift
xi = {g: torch.cat(v) for g, v in cap.items()}                    # (tokens, n_ctx, d), without the depth code
code = {g: vt.depth_code[g].detach().float() for g in range(L)}
Ww, b = bank.w_proj.weight.detach().float(), bank.w_proj.bias.detach().float()
flat = lambda t: t.reshape(t.shape[0], -1) if t.dim() == 3 else t.reshape(-1)
z = {g: flat(xi[g] + code[g]) @ Ww.T + b for g in range(L)}
ent = lambda zz: -(torch.softmax(zz, -1) * torch.log_softmax(zz, -1)).sum(-1)
neff = {g: torch.exp(ent(z[g])) for g in range(L)}

# s_K: the largest square-rooted varentropy over the K-simplex, by random restarts of a local search
def varent(zz):
    lp = torch.log_softmax(zz, -1); p = lp.exp()
    return (p * lp * lp).sum(-1) - (p * lp).sum(-1) ** 2
torch.manual_seed(0)
zz = (torch.randn(4096, K, dtype=torch.float64) * 4).requires_grad_(True)
opt = torch.optim.Adam([zz], lr=0.05)
for _ in range(3000):
    opt.zero_grad(); (-varent(zz).sum()).backward(); opt.step()
sK = float(varent(zz).max().detach().sqrt())

lines = [f'V_theta well-weight logits per layer: {desc}', f'{len(x)} windows x {x.shape[1]} tokens; K = {K}; s_K = {sK:.4f} (bound factor {sK / 2:.4f})', '']
lines.append(f'{"layer":>5} {"|x| med":>8} {"|xi| med":>8} {"|c|":>6} {"spread z med":>12} {"n_eff med":>9} {"n_eff p10":>9} {"n_eff p90":>9}')
for g in range(L):
    sp = z[g].max(-1).values - z[g].min(-1).values
    q = lambda t, a: float(t.quantile(a))
    lines.append(f'{g:5d} {q(flat(xi[g] + code[g]).norm(dim=-1), .5):8.2f} {q(flat(xi[g]).norm(dim=-1), .5):8.2f} {float(code[g].norm()):6.2f} '
                 f'{q(sp, .5):12.2f} {q(neff[g], .5):9.3f} {q(neff[g], .1):9.3f} {q(neff[g], .9):9.3f}')

lines += ['', 'pairs of layers, per token: d = |ln n_eff(i) - ln n_eff(j)|, B = the bound; ratio d/B must be <= 1',
          f'{"i":>2} {"j":>2} {"d med":>7} {"B med":>7} {"max d/B":>8} {"med d/B":>8} {"spread ctx":>10} {"spread code":>11} {"spread all":>10}']
ok = True
for i in range(L):
    for j in range(i + 1, L):
        dz = z[i] - z[j]
        osc = lambda t: t.max(-1).values - t.min(-1).values
        B = 0.5 * sK * osc(dz)
        d = (neff[i].log() - neff[j].log()).abs()
        r = d / B.clamp_min(1e-12)
        ok &= bool((d <= B + 1e-5).all())
        ctx = flat(xi[i] - xi[j]) @ Ww.T
        cod = flat(code[i] - code[j]) @ Ww.T
        lines.append(f'{i:2d} {j:2d} {float(d.median()):7.3f} {float(B.median()):7.3f} {float(r.max()):8.3f} {float(r.median()):8.3f} '
                     f'{float(osc(ctx).median()):10.2f} {float(osc(cod[None])[0]):11.2f} {float(osc(dz).median()):10.2f}')
lines.append(f'-> bound holds for every token and pair: {"YES" if ok else "NO"}')

deep = torch.cat([flat(xi[g] + code[g]).norm(dim=-1) for g in range(1, L)]).median()
x0 = flat(xi[0]); c0 = flat(code[0])
lines += ['', f'scale test: layer 0 rescaled to the deep layers\' median input norm {float(deep):.2f}']
for nm, xx in (('layer 0 as trained (xi + code)', x0 + c0),
               ('layer 0, xi + code rescaled', (x0 + c0) * (deep / (x0 + c0).norm(dim=-1, keepdim=True))),
               ('layer 0, xi rescaled, code kept', x0 * (deep / x0.norm(dim=-1, keepdim=True)) + c0)):
    ne = torch.exp(ent(xx @ Ww.T + b))
    lines.append(f'  {nm:34s} n_eff median {float(ne.median()):.3f}  (p10 {float(ne.quantile(.1)):.3f}, p90 {float(ne.quantile(.9)):.3f})')
for g in (1, L - 1):
    xx = flat(xi[g] + code[g]); sm = xx * (flat(xi[0] + code[0]).norm(dim=-1).median() / xx.norm(dim=-1, keepdim=True))
    ne = torch.exp(ent(sm @ Ww.T + b))
    lines.append(f'  layer {g} shrunk to layer 0\'s median norm  n_eff median {float(ne.median()):.3f}  (p10 {float(ne.quantile(.1)):.3f}, p90 {float(ne.quantile(.9)):.3f})')

if OPTS.get('sweep') == '1':                 # 2026-11-03: n_eff as a function of the bank input's norm (the softmax temperature)
    grid = np.geomspace(0.25, 64, 25)
    lines += ['', 'temperature sweep: every token\'s bank input (xi + code) rescaled to norm r, per layer; median n_eff',
              f'{"r":>7} ' + ' '.join(f'{"L" + str(g):>6}' for g in range(L))]
    for r in grid:
        row = []
        for g in range(L):
            xx = flat(xi[g] + code[g])
            row.append(float(torch.exp(ent((xx * (r / xx.norm(dim=-1, keepdim=True))) @ Ww.T + b)).median()))
        lines.append(f'{r:7.3f} ' + ' '.join(f'{v:6.3f}' for v in row))

txt = '\n'.join(lines)
print(txt)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"vtheta_weight_logits_{OPTS.get('name', 'run')}.txt").write_text(txt + '\n')
