#!/usr/bin/env python3
"""
Verify the learned gamma(h) switch (Cell 0 LEARNED_DAMPING_RATIO, cfg.learned_damping_ratio; protocol
SS5.13, 2026-10-20): zeta(h) = softplus(head(h)) per token and layer, applied as gamma_0 + 2 zeta(h) omega_i
to the stiff modes inside SR2's exact damped flow, initialised at zeta = 1.

    python3 verify_gzh_switch.py OUT check REF.npz [threads=1]   # REF: verify_sr3_switch.py's HEAD reference

On SR2's trained weights. Checks:
  1. off is HEAD, bit for bit (the reference written by verify_sr3_switch.py ref).
  2. on, at initialisation, equals SR3-crit bit for bit: zeta = 1 exactly, so the logits, the loss and every
     shared parameter's gradient coincide with STIFF_MODE_DAMPING_RATIO = 1.0.
  3. the head receives a gradient (finite, non-zero on its weight and bias), the other parameters receive the
     same gradients as SR3-crit, and the head is the only addition to the state_dict.
  4. zeta_stats are recorded per layer in a training forward: median 1, p10 1, p90 1, fraction above 1 equal
     to 0 at initialisation; and after a perturbation of the head's bias the recorded median moves with it.
  5. the clip-group name 'damping_head' matches exactly the head's parameters and nothing else.
"""
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); MODE = sys.argv[2]
REF = Path(sys.argv[3])
HERE = Path(__file__).resolve().parent
_argv = list(sys.argv)
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(HERE))
import verify_pm_switch as P
sys.argv = _argv
G = P.G
torch.set_num_threads(int(next((a[8:] for a in sys.argv[2:] if a.startswith('threads=')), 8)))

SR2 = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
SR3 = (("STIFF_MODE_DAMPING_RATIO = None", "STIFF_MODE_DAMPING_RATIO = 1.0"),)
GZH = (("LEARNED_DAMPING_RATIO = False", "LEARNED_DAMPING_RATIO = True"),)
FOLDER = ('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
          'sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn')
REPL = P.CONFIGS['F3.1'][1] + SR2

val = np.load(G.VAL)
xy = torch.from_numpy(val[:2 * 257].astype(np.int64)).view(2, 257)
x, y = xy[:, :256], xy[:, 1:]


def run(model):
    model.eval()
    with torch.enable_grad():
        lo = model(x)[0].detach().clone()
    model.train(); model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    _, loss = model(x, targets=y)
    loss.backward()
    grads = {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in model.named_parameters()}
    model.eval()
    return lo, loss.detach().clone(), grads


lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
ok = []
say("Learned gamma(h) switch verification on SR2's trained weights (2 x 256 validation tokens)")

off, _, _ = P.build(FOLDER, REPL); off.to('cpu')
lo0, loss0, gr0 = run(off)
ref = np.load(REF)
mine = {n for n, v in gr0.items() if v is not None}
gk = {k[3:] for k in ref.files if k.startswith('g::')}
d_lo = float(np.abs(lo0.numpy() - ref['logits']).max()); d_loss = abs(float(loss0) - float(ref['loss']))
d_g = max(float(np.abs(gr0[n].numpy() - ref[f'g::{n}']).max()) for n in gk & mine)
c1 = d_lo == 0.0 and d_loss == 0.0 and gk == mine and d_g == 0.0
ok.append(c1)
say(f'1. off vs HEAD: max |d logit| {d_lo:.1e}; |d loss| {d_loss:.1e}; same parameters with a gradient {gk == mine} '
    f'({len(mine)}); max |d grad| {d_g:.1e}  -> {"PASS" if c1 else "FAIL"}')

sr3, _, _ = P.build(FOLDER, REPL + SR3); sr3.to('cpu')
lo3, loss3, gr3 = run(sr3)
on, g_on, _ = P.build(FOLDER, REPL + GZH); on.to('cpu')
assert on.damping_head is not None and on.cfg.learned_damping_ratio is True
lo1, loss1, gr1 = run(on)
shared = {n for n, v in gr1.items() if v is not None and not n.startswith('damping_head')}
head = {n for n, v in gr1.items() if n.startswith('damping_head')}
d_lo = float((lo1 - lo3).abs().max()); d_loss = abs(float(loss1) - float(loss3))
d_g = max(float((gr1[n] - gr3[n]).abs().max()) for n in shared)
c2 = d_lo == 0.0 and d_loss == 0.0 and d_g == 0.0 and shared == {n for n, v in gr3.items() if v is not None}
ok.append(c2)
say(f'2. on at initialisation vs SR3-crit: max |d logit| {d_lo:.1e}; |d loss| {d_loss:.1e}; max |d grad| over the '
    f'{len(shared)} shared parameters {d_g:.1e}  -> {"PASS" if c2 else "FAIL"}')

hg = {n: gr1[n] for n in head}
fin = all(v is not None and torch.isfinite(v).all() for v in hg.values())
nz = {n: float(v.abs().max()) for n, v in hg.items() if v is not None}
last_w, last_b = nz.get('damping_head.2.weight', 0.0), nz.get('damping_head.2.bias', 0.0)
sd0, sd1 = off.state_dict(), on.state_dict()
added = set(sd1) - set(sd0)
c3 = fin and last_w > 0 and last_b > 0 and all(k.startswith('damping_head') for k in added) and set(sd0) <= set(sd1)
ok.append(c3)
say(f'3. the head: gradients finite {fin}; |grad| on its last weight {last_w:.2e}, bias {last_b:.2e} (both must be > 0; '
    f'the first layer\'s gradient is zero at initialisation because the last weight is zero); added tensors {sorted(added)}'
    f'  -> {"PASS" if c3 else "FAIL"}')

zs = on.zeta_stats.clone()
init_ok = bool(torch.allclose(zs[:, :3], torch.ones_like(zs[:, :3])) and torch.all(zs[:, 3] == 0))
with torch.no_grad():
    on.damping_head[2].bias.fill_(float(np.log(np.exp(2.0) - 1.0)))      # softplus^-1(2)
on.train(); on.zero_grad(set_to_none=True)
with torch.enable_grad():
    on(x, targets=y)
zs2 = on.zeta_stats.clone(); on.eval()
moved_ok = bool(torch.allclose(zs2[:, 0], torch.full_like(zs2[:, 0], 2.0), atol=1e-5) and torch.all(zs2[:, 3] == 1.0))
c4 = init_ok and moved_ok
ok.append(c4)
say(f'4. zeta_stats per layer at initialisation [median, p10, p90, frac > 1]: {[[round(float(v), 3) for v in r] for r in zs]}; '
    f'with the bias set to softplus^-1(2): {[[round(float(v), 3) for v in r] for r in zs2]}  -> {"PASS" if c4 else "FAIL"}')

names = [n for n, _ in on.named_parameters()]
grp = [n for n in names if 'damping_head' in n]
c5 = set(grp) == head and len(grp) == 4
ok.append(c5)
say(f'5. clip group \'damping_head\' matches {len(grp)} parameters: {grp}  -> {"PASS" if c5 else "FAIL"}')
say(f'\nALL {"PASS" if all(ok) else "FAIL"}')
(HERE / 'verify_gzh_switch_output.txt').write_text('\n'.join(lines) + '\n')
