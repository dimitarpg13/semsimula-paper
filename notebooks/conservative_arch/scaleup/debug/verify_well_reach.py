#!/usr/bin/env python3
"""Well-reachability switch check (protocol p-well-reach, 2026-10-28).

    python3 verify_well_reach.py ROOT OUT dump      # ROOT = a HEAD worktree (git worktree add): reference records
    python3 verify_well_reach.py ROOT OUT compare   # ROOT = the working copy: OFF must be bit-identical
    WR_VAL=<code_val_2M.npy> WR_NSEQ=8 python3 verify_well_reach.py ROOT OUT on   # ON checks (default: 2 x 256 OWT tokens)

Built through the ladder notebook's own Cells 0-5b (verify_pm_switch.build) on trained weights,
2 x 256 OpenWebText validation tokens; records eval logits, train-mode logits (same seed), the
loss and every parameter gradient (verify_pm_switch.record).
"""
import sys, time
from pathlib import Path

import torch

ROOT, OUT, MODE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(ROOT / 'notebooks/conservative_arch/scaleup/debug'))
import verify_pm_switch as P
assert Path(P.__file__).resolve().is_relative_to(ROOT.resolve()), P.__file__
torch.set_num_threads(8)

_T = ('topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_'
      'fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn')
SO = 'semsimula_fock_cfc_baoab_code_xi5long_' + _T
XI6 = 'semsimula_fock_cfc_baoab_code_xi6long_' + _T.replace('dcvt5x8', 'dcvt6x8')
FOCK = 'semsimula_fock_cfc_baoab_code_xi5long_' + _T.replace('_norc', '')
CODE = P.LIVE + (('LOWRANK_DAMPED_FLOW  = False', 'LOWRANK_DAMPED_FLOW  = True'),
                 ('LADDER_STEPS     = 32_500', 'LADDER_STEPS     = 16_250'))
NORC = (('REVERSE_CHANNEL              = True', 'REVERSE_CHANNEL              = False'),)
CONFIGS = {   # name: (weights folder, checkpoint, Cell 0 replacements)
    'SO-CODE': (SO, SO[10:].replace('baoab_code', 'code') + '_best.pt', CODE + NORC),
    'XI6-CODE': (XI6, XI6[10:].replace('baoab_code', 'code') + '_best.pt',
                 CODE + NORC + (("XI_OVERRIDE     = '5long'", "XI_OVERRIDE     = '6long'"),)),
    'FOCK-CODE': (FOCK, FOCK[10:].replace('baoab_code', 'code') + '_best.pt', CODE),
    'F3.1': (P.CONFIGS['F3.1'][0], None, P.CONFIGS['F3.1'][1]),
    "G3'": (P.CONFIGS["G3'"][0], None, P.CONFIGS["G3'"][1]),
}
WR = (('WELL_REACH_RHO    = None', 'WELL_REACH_RHO    = 3'),)
import os
if os.environ.get('WR_VAL'):            # e.g. the code validation set, for the code arms' penalty readings
    P.G.VAL = Path(os.environ['WR_VAL'])
x, y = P.tokens(n_seq=int(os.environ.get('WR_NSEQ', 2)))
ref = OUT / 'wr_ref.pt'

if MODE == 'dump':
    recs = {}
    for name, (f, ck, repl) in CONFIGS.items():
        t0 = time.time(); model, g, _ = P.build(f, repl, ckpt=ck)
        recs[name] = P.record(model, x, y)
        print(f"  dumped {name}: tag ...{g['_variant_tag'][-40:]}  loss {recs[name]['loss']:.6f}  ({time.time() - t0:.0f} s)", flush=True)
    torch.save(recs, ref)
    sys.exit(0)

if MODE == 'compare':
    h = torch.load(ref); ok = []
    print('OFF (WELL_REACH_RHO = None): HEAD vs working copy, trained weights')
    for name, (f, ck, repl) in CONFIGS.items():
        model, g, _ = P.build(f, repl, ckpt=ck)
        assert 'wr' not in ''.join(t[:2] for t in g['_variant_tag'].split('_'))
        rec = P.record(model, x, y)
        assert set(h[name]['grads']) == set(rec['grads']), name
        de = (h[name]['eval'] - rec['eval']).abs().max().item()
        dt = (h[name]['train'] - rec['train']).abs().max().item()
        dg = max((h[name]['grads'][k] - rec['grads'][k]).abs().max().item() for k in rec['grads'])
        ok.append(de == dt == dg == 0 and h[name]['loss'] == rec['loss'])
        print(f"  {name:9s} eval {de:.1e}  train {dt:.1e}  grads {dg:.1e}  loss {rec['loss']:.6f}  ({len(rec['grads'])} params)  -> "
              f"{'IDENTICAL' if ok[-1] else 'DIFFERS'}", flush=True)
    print('ALL IDENTICAL' if all(ok) else 'SOME DIFFER')
    sys.exit(0 if all(ok) else 1)

# ON checks, on the three code arms
ok = []
def check(cond, msg):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {msg}", flush=True)

for name in ('SO-CODE', 'XI6-CODE', 'FOCK-CODE'):
    f, ck, repl = CONFIGS[name]
    print(f'\n{name}')
    m0, g0, _ = P.build(f, repl, ckpt=ck)
    keys0 = set(m0.state_dict()); params0 = {k: p.shape for k, p in m0.named_parameters()}
    model, g, out = P.build(f, repl + WR, ckpt=ck)
    tag = g['_variant_tag']
    check(tag == g0['_variant_tag'].replace('_L2probe_', '_wr3l0p1_L2probe_'), f"tag gains exactly 'wr3l0p1': ...{tag[tag.find('st16250'):][:40]}")
    check('WELL REACHABILITY: rho = 3' in out and '[reach] WELL_REACH_RHO=3' in out, 'Cell 0 banner and Cell 5b guard printed')
    check(set(model.state_dict()) == keys0 and {k: p.shape for k, p in model.named_parameters()} == params0,
          'no new parameters or state-dict keys (the stats buffer is not persistent)')
    # eval: identical to OFF, nothing recorded
    model.eval(); m0.eval()
    with torch.enable_grad():
        e1 = model(x)[0].detach(); e0 = m0(x)[0].detach()
    check(torch.equal(e1, e0) and not model._well_reach_inputs, 'eval logits bit-identical to OFF; nothing recorded in eval')
    # train: forward identical, one term per layer, values as the probe measured
    model.train(); m0.train(); model.cfg.use_layer_checkpoint = m0.cfg.use_layer_checkpoint = False
    torch.manual_seed(7); t1, l1 = model(x, y)
    torch.manual_seed(7); t0_, l0 = m0(x, y)
    terms = dict(model._well_reach_inputs)
    check(torch.equal(t1.detach(), t0_.detach()) and l1.item() == l0.item(), 'train-mode logits and NTP loss bit-identical to OFF')
    check(sorted(terms) == list(range(model.cfg.L)), f'inputs recorded for layers {sorted(terms)} (L = {model.cfg.L})')
    pen = model.pop_well_reach_penalty()
    st = model.well_reach_stats_values()
    print(f"      per layer [median nearest ratio, fraction above 3, term]: {st}")
    check(not model._well_reach_inputs and torch.isfinite(pen), f'pop drains the inputs; penalty {pen.item():.6f}')
    # gradient of the penalty alone: only the well centres' generator and the depth code
    model.zero_grad(set_to_none=True)
    pen.backward()
    got = sorted(k for k, p in model.named_parameters() if p.grad is not None and p.grad.abs().max() > 0)
    allowed = lambda k: k.startswith('V_theta.') and ('mu_proj.' in k or k.endswith('depth_code'))
    print(f"      parameters with a penalty gradient: {got}")
    if pen.item() > 0:
        check(got and all(allowed(k) for k in got), 'penalty gradient reaches only mu_proj and the depth code')
    else:
        check(not got, 'zero penalty, no gradient')
    model.zero_grad(set_to_none=True)
    # layer checkpointing: recorded once per layer (not on the recompute), backward runs
    model.cfg.use_layer_checkpoint = True
    torch.manual_seed(7); _, lc = model(x, y)
    n_ck = len(model._well_reach_inputs); pc = model.pop_well_reach_penalty()
    (lc + 0.1 * pc).backward()
    check(n_ck == model.cfg.L and abs(pc.item() - pen.item()) < 1e-5 and len(model._well_reach_inputs) == 0,
          f'under layer checkpointing: {n_ck} layers recorded, penalty {pc.item():.6f} (as without), backward runs, nothing recorded by its recompute')
    model.cfg.use_layer_checkpoint = False
    del model, m0

print('\nALL PASS' if all(ok) else '\nSOME FAIL')
sys.exit(0 if all(ok) else 1)
