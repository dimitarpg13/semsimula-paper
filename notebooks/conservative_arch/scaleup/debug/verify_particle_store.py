#!/usr/bin/env python3
"""Verification of the particle store (PARTICLE_STORE, 2026-10-31; Stage 0'' of
Implementing_Semantic_Templates_A_Path.md SS7).

    python3 verify_particle_store.py OUT_DIR

Bit-identity with the switch OFF is checked separately, by verify_pm_match.py's
fingerprint on a HEAD checkout and on the working copy (four configurations: F3.1,
PM1-K64, PMX-K16, FO-b-PMX-K16); see verify_particle_store_output.txt.

Checks here (PARTICLE_STORE = 4, rank 8, F3.1 base + SR2, no Poisson modes), at
initialisation, real validation tokens, CPU:
  1. tag 'ps4r8'; the new parameters are exactly pstore_{mu, P, log_c, depth, delta,
     logit_lambda}; the clip group 'pstore_' catches them and nothing else;
  2. depths at 0: eval and train logits bit-identical to the model without the store,
     and the gradient reaches pstore_depth;
  3. the first training forward sets mu_l, P_l, D2_l from the layer's states; the
     existences are finite and in (0, 1);
  4. depths set non-zero: the force equals -autograd.grad of U at fixed sources, at
     layer 0 (own particles only) and at layer 1 (layer-0 and own particles);
  5. causality with non-zero depths: tokens replaced at t >= 128 leave the logits at
     t < 128 unchanged;
  6. layer checkpointing on and off give the same loss and the same gradients, bit
     for bit (the store is read across checkpoint regions); the same for PM1-K16
     pmlr8 (Stage 0'), whose matcher shares _lowrank_match -- before 2026-10-31 the
     backward recompute re-ran the running median (gradients off by 6.6e-5, D2
     moved twice per step) under the production default LAYER_CHECKPOINT = True;
  7. every pstore_ parameter has a finite, non-zero gradient;
  8. readings for criterion 2 of Stage 0'' (no pass/fail): mean existence and its CV
     across tokens per layer, live particles at t = 32 / 128 / 255 with and without
     spontaneous death, and the share of layer 1's potential carried by layer-0
     particles (the cross-layer reach).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(Path(__file__).parent))
import verify_pm_switch as P
torch.set_num_threads(8)

F31 = P.CONFIGS['F3.1'][1]
SR2 = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
PS = (("PARTICLE_STORE       = 0", "PARTICLE_STORE       = 4"),)
ND = (("PARTICLE_STORE_DECAY = True", "PARTICLE_STORE_DECAY = False"),)
NEW = ['pstore_P', 'pstore_delta', 'pstore_depth', 'pstore_log_c', 'pstore_logit_lambda', 'pstore_mu']


class _Stop(Exception):
    pass


def build_init(repl):
    load = torch.load
    torch.load = lambda *a, **k: (_ for _ in ()).throw(_Stop())
    torch.manual_seed(0); np.random.seed(0)
    try:
        P.build(P.F31_F, repl)
    except _Stop:
        tb = sys.exc_info()[2]; m = g = None
        while tb:
            fl = tb.tb_frame.f_locals
            if tb.tb_frame.f_code.co_name == 'build' and 'model' in fl:
                m, g = fl['model'], fl['g']
            tb = tb.tb_next
        m.to('cpu'); m.cfg.use_layer_checkpoint = False
        return m, g
    finally:
        torch.load = load


def tokens(g, n=2, T=64, off=0):
    x = torch.from_numpy(g['val_ids'][off:off + n * (T + 1)].astype('int64')).view(n, T + 1)
    return x[:, :T], x[:, 1:]


def record(model, x, y, seed=7):
    model.eval()
    with torch.enable_grad():
        ev, _ = model(x)
    model.train(); torch.manual_seed(seed)
    tr, loss = model(x, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return {'eval': ev.detach(), 'train': tr.detach(), 'loss': loss.item(), 'grads': grads}


ok_all = True
def say(ok, msg):
    global ok_all
    ok_all &= bool(ok)
    print(f'  [{"PASS" if ok else "FAIL"}] {msg}', flush=True)


print('=== PARTICLE_STORE = 4, rank 8, F3.1 base + SR2, explicit kick ===')
off_model, g_off = build_init(F31 + SR2)
model, g = build_init(F31 + SR2 + PS)
tag, tag_off = g['_variant_tag'], g_off['_variant_tag']
L, d = model.cfg.L, model.cfg.d

# 1. structure
new = sorted(set(dict(model.named_parameters())) - set(dict(off_model.named_parameters())))
say('ps4r8' in tag.split('_') and tag == tag_off.replace('_sr2_L2probe', '_ps4r8_sr2_L2probe'), f'tag ...{tag[tag.find("ps4r8"):]}')
say(new == NEW and tuple(model.pstore_mu.shape) == (L, 4, d) and tuple(model.pstore_P.shape) == (L, 8, d)
    and tuple(model.pstore_depth.shape) == (L, 4) and tuple(model.pstore_delta.shape) == (4, d)
    and float(model.pstore_depth.abs().sum()) == 0 and float(model.pstore_delta.abs().sum()) == 0,
    f'new parameters {new}; depths and delta at 0; {sum(p.numel() for p in model.parameters()):,} parameters')
c6 = [''.join(c['source']) for c in json.load(open(P.G.NB))['cells']
      if ''.join(c['source']).startswith('# == Cell 6: Training loop')][0]
i0 = c6.index('GRAD_CLIP_OVERRIDES = {')
end = "    GRAD_CLIP_OVERRIDES['pstore_'] = PARTICLE_STORE_CLIP\n"
i1 = c6.index(end) + len(end)
from grad_clip_utils import GradClipConfig, assign_clip_group
ns = {'GRAD_CLIP_VPHI': 0.3, 'RELAX_FIELD_CLIP': None, 'POISSON_MODES': 0, 'POISSON_MODE_CLIP': 0.3,
      'PARTICLE_STORE': 4, 'PARTICLE_STORE_CLIP': 0.3}
exec(c6[i0:i1], ns)
ccfg = GradClipConfig(default_clip=1.0, overrides=ns['GRAD_CLIP_OVERRIDES'])
grp = {n_: assign_clip_group(n_, ccfg)[0] for n_, _ in model.named_parameters()}
caught = sorted(n_ for n_, gk in grp.items() if gk == 'override:pstore_')
say(caught == NEW, f"clip group 'pstore_' catches {caught}")

# 2. depths at 0: identical to the model without the store
xb, yb = tokens(g)
a = record(model, xb, yb); b = record(off_model, xb, yb)
d0 = max((a['eval'] - b['eval']).abs().max().item(), (a['train'] - b['train']).abs().max().item())
gdep = a['grads'].get('pstore_depth')
say(d0 == 0.0 and gdep is not None and bool(torch.isfinite(gdep).all()) and float(gdep.abs().sum()) > 0,
    f'depths 0: max |dlogit| vs no store {d0:.1e}; grad reaches pstore_depth (norm {float(gdep.norm()) if gdep is not None else float("nan"):.2e})')
say(float(model.pstore_ready.sum()) == L, f'the first training forward initialised every layer\'s matcher (ready {model.pstore_ready.tolist()})')

# 3. existences on 2 x 256 real tokens
xl, yl = tokens(g, n=2, T=256, off=100_000)
model.train(); torch.manual_seed(7)
with torch.enable_grad():
    _, loss = model(xl, yl)
Es = {l: model._pstore[l][0].detach() for l in range(L)}
fin = all(bool(torch.isfinite(E).all()) and float(E.min()) > 0 and float(E.max()) < 1 for E in Es.values())
say(fin and bool(torch.isfinite(loss)), 'existences finite and in (0, 1) at every layer; loss finite')
D2 = model.pstore_D2.clone()
say(bool((D2 > 0).all()) and bool((model.pstore_P.detach().norm(dim=-1) > 0.99).all()),
    f'D2 per layer {[round(float(v), 3) for v in D2]}; P rows unit-norm')

# 4. force = -grad U at fixed sources, depths and delta non-zero
torch.manual_seed(3)
with torch.no_grad():
    model.pstore_depth.normal_(0, 0.5)
    model.pstore_delta.normal_(0, 0.05)
model.eval(); model._in_stack_forward = False          # read the forward's store; record nothing
for l in range(L):
    h = model._pstore[l][1].detach().clone().requires_grad_(True)
    U = model.particle_store_potential(h, l)
    gU, = torch.autograd.grad(U, h)
    f = model.particle_store_force(h, l)
    rel = ((f + gU).norm() / gU.norm()).item()
    say(rel < 1e-5, f'layer {l}: force vs -grad U at fixed sources, relative error {rel:.1e} (|F| {f.norm():.3e}, '
                    f'{"own particles only" if l == 0 else "layer-0 and own particles"})')
model._in_stack_forward = True

# 5. causality
rng = np.random.default_rng(0)
x2 = xl.clone(); x2[:, 128:] = torch.from_numpy(rng.integers(0, 50257, size=(2, xl.shape[1] - 128)))
model.eval()
with torch.enable_grad():
    l1, _ = model(xl); l2, _ = model(x2)
dc = (l1[:, :128] - l2[:, :128]).abs().max().item()
say(dc == 0.0, f'causality, depths non-zero: tokens replaced at t >= 128, max |dlogit| at t < 128 = {dc:.1e}')

# 6. checkpoint on / off, from the same state (pstore_D2 is a running median that a
#    training forward moves, so it is restored between the two runs)
D2_0 = model.pstore_D2.clone()
r_off = record(model, xb, yb)
D2_off = model.pstore_D2.clone()
with torch.no_grad():
    model.pstore_D2.copy_(D2_0)
model.cfg.use_layer_checkpoint = True
r_on = record(model, xb, yb)
D2_on = model.pstore_D2.clone()
model.cfg.use_layer_checkpoint = False
same = (r_on['loss'] == r_off['loss'] and set(r_on['grads']) == set(r_off['grads'])
        and all(torch.equal(r_on['grads'][k], r_off['grads'][k]) for k in r_off['grads']))
say(same, f'layer checkpointing on vs off from the same state: loss {r_on["loss"]:.6f} vs {r_off["loss"]:.6f}; '
          f'max grad diff {max((r_on["grads"][k] - r_off["grads"][k]).abs().max().item() for k in r_off["grads"]):.1e}')
say(torch.equal(D2_off, D2_on), f'D2 after the step the same with checkpointing on and off: {[round(float(v), 4) for v in D2_on]}')

# 6b. the same for the Stage 0' matcher (PM1-K16 pmlr8), which shares _lowrank_match
K16 = (("POISSON_MODES        = 0", "POISSON_MODES        = 16"),)
MATCH = (("POISSON_MATCH_RANK   = 0", "POISSON_MATCH_RANK   = 8"),)
mp, gp = build_init(F31 + SR2 + K16 + MATCH)
xp, yp = tokens(gp)
record(mp, xp, yp)                       # matcher initialised at depth 0
torch.manual_seed(3)
with torch.no_grad():
    mp.pm_depth.normal_(0, 0.5)          # the states now move between layers, so cur != D2
pD2_0 = mp.pm_D2.clone()
pa = record(mp, xp, yp); pD2_off = mp.pm_D2.clone()
with torch.no_grad():
    mp.pm_D2.copy_(pD2_0)
mp.cfg.use_layer_checkpoint = True
pb = record(mp, xp, yp); pD2_on = mp.pm_D2.clone()
mp.cfg.use_layer_checkpoint = False
pdg = max((pa['grads'][k] - pb['grads'][k]).abs().max().item() for k in pa['grads'])
say(pa['loss'] == pb['loss'] and pdg == 0.0 and torch.equal(pD2_off, pD2_on),
    f'PM1-K16 pmlr8 (Stage 0\'): checkpointing on vs off, loss diff {abs(pa["loss"] - pb["loss"]):.1e}, max grad diff {pdg:.1e}, '
    f'D2 {[round(float(v), 3) for v in pD2_on]} both ways')
del mp

# 7. gradients
gn = {k: r_off['grads'][k].norm().item() for k in r_off['grads'] if k.startswith('pstore_')}
say(sorted(gn) == NEW and all(np.isfinite(v) and v > 0 for v in gn.values()),
    'grads, depths non-zero: ' + ', '.join(f'{k[7:]} {v:.2e}' for k, v in gn.items()))

# 8. readings for criterion 2
print('\n--- readings (criterion 2 of Stage 0\'\'): 2 x 256 real tokens, matcher at initialisation ---')
def live_count(E, logit_lam):
    B, T, K = E.shape
    t = torch.arange(T); lag = t[:, None] - t[None, :] - 1; past = lag >= 0
    if logit_lam is None:
        Lam = past.float().unsqueeze(-1).expand(T, T, K)
    else:
        ll = torch.nn.functional.logsigmoid(logit_lam.detach().float())
        Lam = torch.where(past, torch.exp(lag.clamp_min(0)[None].float() * ll[:, None, None]), torch.zeros(())).permute(1, 2, 0)
    return torch.einsum('tsk,bsk->btk', Lam, E).sum(-1).mean(0)       # (T,) live particles seen by token t
for arm, extra in (('decay 1-16 tokens', ()), ('no decay', ND)):
    m2, g2 = build_init(F31 + SR2 + PS + extra)
    m2.train(); torch.manual_seed(7)
    with torch.enable_grad():
        m2(xl, yl)
    st = m2.pstore_stats.clone()
    print(f'  {arm}: tag ...{g2["_variant_tag"][g2["_variant_tag"].find("ps4r8"):g2["_variant_tag"].find("_L2probe")]}')
    for l in range(L):
        E = m2._pstore[l][0].detach()
        n = live_count(E, m2.pstore_logit_lambda)
        reach = (E > 0.5).float().mean(dim=(0, 1))
        print(f'    layer {l}: mean existence {st[l, 0]:.3f}, CV across tokens {st[l, 1]:.2f}, '
              f'share of the reach from earlier layers\' particles {st[l, 2]:.3f}; '
              f'live particles at t = 32 / 128 / 255: {n[32]:.1f} / {n[128]:.1f} / {n[255]:.1f}; '
              f'tokens with E > 0.5 per type {[round(float(v), 2) for v in reach]}')
    del m2

print(f'\n-> {"ALL PASS" if ok_all else "FAIL"}')
sys.exit(0 if ok_all else 1)
