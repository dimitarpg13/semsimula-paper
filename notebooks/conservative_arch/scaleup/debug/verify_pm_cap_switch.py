"""PM1-cap switch check: bounded well depths (protocol SS5.15, 2026-10-08).

OFF (POISSON_DEPTH_CAP = None) must be bit-identical to HEAD on the PM1
configuration (F3.1's Cell 0 + POISSON_MODES = 64) on PM1's trained weights:
eval logits, train-mode loss and every parameter gradient. HEAD is taken from
`git archive HEAD`, so the working tree is not touched.

  python3 verify_pm_cap_switch.py OUT_DIR --dump     # run from the HEAD copy
  python3 verify_pm_cap_switch.py OUT_DIR            # working copy: compare + ON checks

ON (POISSON_DEPTH_CAP = 0.3), on PM1's trained weights:
  1. tags: every existing arm's tag unchanged against HEAD; the capped arm's
     tag carries 'pmcap0p3'; Cell 5b built the model with the cap
  2. effective depths: a = cap tanh(a_raw / cap), |a| < cap, and below the cap
     the trained depths are only mildly changed
  3. the force equals -autograd.grad of U with the capped depths at fixed phi
  4. depths at 0: logits bit-identical to the model without modes
  5. causality with the cap: tokens replaced at t >= 256 leave t < 256 unchanged
  6. gradients reach pm_depth through the tanh
"""
import contextlib, io, json, re, sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); DUMP = '--dump' in sys.argv
sys.argv = [sys.argv[0], str(OUT / 'harness')]
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import verify_pm_switch as P

PM1 = ('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64'
       '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn')
_, F31_REPL = P.CONFIGS['F3.1']
CAP_ON = (("POISSON_DEPTH_CAP    = None", "POISSON_DEPTH_CAP    = 0.3"),)


def record(model, x, y):
    model.eval()
    with torch.enable_grad():
        ev, _ = model(x)
    model.train(); torch.manual_seed(7)
    tr, loss = model(x, y)
    loss.backward()
    grads = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return {'eval': ev.detach(), 'train': tr.detach(), 'loss': loss.item(), 'grads': grads}


x, y = P.tokens(n_seq=2, T=256)
if DUMP:
    model, g, _ = P.build(PM1, F31_REPL + P.PM_ON)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(record(model, x, y), OUT / 'head_pm1.pt')
    print(f'dumped HEAD PM1 outputs to {OUT}/head_pm1.pt')
    raise SystemExit(0)

lines = []
say = lambda s='': (print(s), lines.append(s))
head = torch.load(OUT / 'head_pm1.pt', weights_only=False)

# ---- OFF: bit-identical to HEAD ----
model, g, _ = P.build(PM1, F31_REPL + P.PM_ON)
assert getattr(model.cfg, 'poisson_depth_cap', 'missing') is None
cur = record(model, x, y)
d_eval = float((cur['eval'] - head['eval']).abs().max())
d_train = float((cur['train'] - head['train']).abs().max())
d_grad = max(float((cur['grads'][k] - head['grads'][k]).abs().max()) for k in head['grads'])
same_keys = set(cur['grads']) == set(head['grads'])
off_ok = d_eval == 0 and d_train == 0 and d_grad == 0 and cur['loss'] == head['loss'] and same_keys
say('OFF (POISSON_DEPTH_CAP = None): working copy vs HEAD, PM1 config on PM1\'s trained weights')
say(f'   eval {d_eval:.1e}  train {d_train:.1e}  grads {d_grad:.1e}  ({len(head["grads"])} params, same set: {same_keys})'
    f'  -> {"IDENTICAL" if off_ok else "DIFFERS"}')

# ---- ON ----
say('\nON (F3.1 Cell 0 + POISSON_MODES = 64 + POISSON_DEPTH_CAP = 0.3), on PM1\'s trained weights')
model, g, out5b = P.build(PM1, F31_REPL + P.PM_ON + CAP_ON)
tag = g['_variant_tag']
ok1 = 'pmcap0p3' in tag.split('_') and model.cfg.poisson_depth_cap == 0.3 and 'well depths capped at 0.3' in out5b
say(f'1. tag ...{tag[tag.find("pm64"):tag.find("_ob_")]}; model cap {model.cfg.poisson_depth_cap}; 5b banner: '
    f'{"well depths capped at 0.3" in out5b} -> {ok1}')

raw = model.pm_depth.detach().float()
eff = torch.stack([model.pm_effective_depth(l) for l in range(model.cfg.L)]).detach()
ok2 = torch.allclose(eff, 0.3 * torch.tanh(raw / 0.3)) and float(eff.abs().max()) < 0.3
say(f'2. effective depths: max |a| {float(eff.abs().max()):.4f} < 0.3; raw max |a| {float(raw.abs().max()):.4f}; '
    f'layer-1 median raw {float(raw[1].median()):+.3f} -> capped {float(eff[1].median()):+.3f} -> {ok2}')

torch.manual_seed(0)
h = (model.pm_mu.detach()[None, :64] + 0.5 * torch.randn(1, 64, model.cfg.d)).requires_grad_(True)
E, phi = model.poisson_mode_occupation(h.detach())
k2 = model.pm_log_kappa2.detach().float().exp()
U = -(phi.detach() * eff[1] * torch.exp(-k2 * ((h[0, :, None, :] - model.pm_mu.detach()[None]) ** 2).sum(-1))[None]).sum()
gU, = torch.autograd.grad(U, h)
F = model.poisson_mode_force(h.detach(), 1).float()
rel = float((F + gU).norm() / gU.norm())
ok3 = rel < 1e-5
say(f'3. force vs -grad U with the capped depths at fixed phi: relative error {rel:.1e} -> {ok3}')

# 4. one set of weights (PM1's): the capped model at zero depth against the same
#    non-mode weights loaded into a model built without modes
model.eval()
keep = model.pm_depth.detach().clone()
model_nopm, _, _ = P.build(P.CONFIGS['F3.1'][0], F31_REPL)
model_nopm.eval()
r = model_nopm.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith('pm_')}, strict=False)
with torch.no_grad():
    model.pm_depth.zero_()
with torch.enable_grad():
    lo_zero, _ = model(x)
    lo_nopm, _ = model_nopm(x)
with torch.no_grad():
    model.pm_depth.copy_(keep)
d0 = float((lo_zero - lo_nopm).abs().max())
ok4 = d0 == 0.0 and not r.missing_keys and not r.unexpected_keys
say(f'4. depths at 0 with the cap on vs the same weights in a model without modes: max |d logit| {d0:.1e} -> {ok4}')

with torch.enable_grad():
    base, _ = model(x)
xc = x.clone(); xc[:, 256:] = torch.randint(0, model.cfg.vocab_size, xc[:, 256:].shape, generator=torch.Generator().manual_seed(1))
with torch.enable_grad():
    pert, _ = model(xc)
d5 = float((pert[:, :256] - base[:, :256]).abs().max())
ok5 = d5 == 0.0
say(f'5. causality with the cap: tokens replaced at t >= 256, max |d logit| at t < 256 = {d5:.1e} -> {ok5}')

model.train(); torch.manual_seed(7)
_, loss = model(x, y); loss.backward()
gd = model.pm_depth.grad
ok6 = gd is not None and float(gd.abs().max()) > 0
say(f'6. gradient reaches pm_depth through the tanh: norm {float(gd.norm()) if gd is not None else 0:.3e} -> {ok6}')
model.zero_grad(set_to_none=True)

allok = off_ok and ok1 and ok2 and ok3 and ok4 and ok5 and ok6
say(f'\n-> {"ALL PASS" if allok else "FAILURES ABOVE"}')
(HERE / 'verify_pm_cap_switch_output.txt').write_text('\n'.join(lines) + '\n')
