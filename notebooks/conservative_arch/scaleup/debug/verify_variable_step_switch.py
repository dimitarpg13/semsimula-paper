#!/usr/bin/env python3
"""Variable-step training switch check (SR4a, protocol SS5.9 and p-sr4a-code, 2026-11-01).

    python3 verify_variable_step_switch.py OUT [probe]

OFF (VARIABLE_STEP_COUNTS = None) bit-identity is checked by verify_well_reach.py's dump (HEAD
worktree) and compare (working copy) modes, which record eval logits, train logits, the loss and every
gradient on five trained arms. This script checks ON, on SO-CODE's trained weights built through the
notebook's Cells 0-5b, 2 x 256 code validation tokens:
  1. tag: SO-CODE's plus exactly 'vs234'; the Cell 0 banner; no new parameters or state-dict keys
  2. stack_steps = L (and None) is the trained walk, bit for bit
  3. eval, N = 1, 3, 4, 6, 8: the model's walk equals Cell 6b-7's _fom_stack ('hold', dt = T/N), bit for bit
  4. train, N = 3: loss and every gradient equal _fom_stack's, with layer checkpointing off and on;
     the repulsion terms number N (one per step); _stack_j is None after the walk
  5. layer checkpointing on vs off at N = 3: same loss and gradients
  6. stack_steps with substeps_per_layer > 1 is refused
  7. Cell 6 fragments: the step-count draw (own generator, uniform, resumable from its saved state,
     leaves the batch generator untouched) and the per-N log entry
  8. with 'probe': Cell 6's vs_refinement_probe equals Cell 6b-7's evaluation on the same 12 x 4 batches
     at N = 2, 3 and 8 (about 15 min on CPU)
"""
import contextlib, io, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
PROBE = 'probe' in sys.argv[2:]
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(Path(__file__).parent))
import verify_pm_switch as P
torch.set_num_threads(8)

P.G.VAL = Path.home() / 'Downloads/semsimula_code_gpt2/data/code_val_2M.npy'
_T = ('topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_'
      'fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn')
SO = 'semsimula_fock_cfc_baoab_code_xi5long_' + _T
SO_CK = SO[10:].replace('baoab_code', 'code') + '_best.pt'
CODE = P.LIVE + (('LOWRANK_DAMPED_FLOW  = False', 'LOWRANK_DAMPED_FLOW  = True'),
                 ('LADDER_STEPS     = 32_500', 'LADDER_STEPS     = 16_250'),
                 ('REVERSE_CHANNEL              = True', 'REVERSE_CHANNEL              = False'))
VS = (('VARIABLE_STEP_COUNTS      = None', 'VARIABLE_STEP_COUNTS      = (2, 3, 4)'),)

ok = []
def check(cond, msg):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {msg}", flush=True)

t0 = time.time()
m0, g0, _ = P.build(SO, CODE, ckpt=SO_CK)
model, g, out = P.build(SO, CODE + VS, ckpt=SO_CK)
L, T = model.cfg.L, model.cfg.L * model.cfg.dt
print(f'built SO-CODE twice ({time.time() - t0:.0f} s); L = {L}, dt = {model.cfg.dt:g}, T = {T:g}')

# 1. tag, banner, parameters
tag = g['_variant_tag']
check(tag == g0['_variant_tag'].replace('_sr2_st16250_', '_sr2_vs234_st16250_'), f"tag ...{tag[tag.find('sr2'):][:40]}")
check('[SR4a] VARIABLE_STEP_COUNTS=(2, 3, 4)' in out, 'Cell 0 banner printed')
check(set(model.state_dict()) == set(m0.state_dict())
      and {k: p.shape for k, p in model.named_parameters()} == {k: p.shape for k, p in m0.named_parameters()},
      'no new parameters or state-dict keys')
check(model.stack_steps is None and model._stack_j is None, 'built with stack_steps = None')

x, y = P.tokens(n_seq=2)

# Cell 6b-7's harness, executed from the notebook (definitions only)
c67 = [''.join(c['source']) for c in json.load(open(P.G.NB))['cells']
       if ''.join(c['source']).startswith('# == Cell 6b-7: flow or maps?')][0]
ns67 = {'CKPT_DIR': Path('.'), 'CKPT_PREFIX': 'x'}
exec(compile(c67[:c67.index('_f7_saved = ')], 'Cell6b7', 'exec'), ns67)
fom = ns67['_fom_stack']


def logits_eval(n=None, fom_n=None):
    model.eval()
    with torch.enable_grad():
        if fom_n is not None:
            with fom(model, fom_n, T / fom_n, 'hold'):
                return model(x)[0].detach()
        model.stack_steps = n
        try:
            return model(x)[0].detach()
        finally:
            model.stack_steps = None


def train_rec(n=None, fom_n=None, ckpt=False):
    model.train(); model.cfg.use_layer_checkpoint = ckpt
    model.zero_grad(set_to_none=True)
    torch.manual_seed(7)
    ctx = fom(model, fom_n, T / fom_n, 'hold') if fom_n is not None else contextlib.nullcontext()
    model.stack_steps = n
    try:
        with ctx:
            lg, loss = model(x, y)
    finally:
        model.stack_steps = None
    nrep = len(model._repulsion_terms)
    loss.backward()
    gr = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True); model._repulsion_terms = []
    return {'logits': lg.detach(), 'loss': loss.item(), 'grads': gr, 'nrep': nrep, 'j': model._stack_j}


def gdiff(a, b):
    assert set(a['grads']) == set(b['grads'])
    return max((a['grads'][k] - b['grads'][k]).abs().max().item() for k in a['grads'])


# 2. N = L is the trained walk
base = logits_eval()
check(torch.equal(logits_eval(n=L), base), f'eval: stack_steps = {L} (= L) bit-identical to None')
with torch.enable_grad():
    m0.eval(); check(torch.equal(m0(x)[0].detach(), base), 'eval: the ON-built model at N = L equals the OFF-built model')

# 3. eval against Cell 6b-7
for n in (1, 3, 4, 6, 8):
    a, b = logits_eval(n=n), logits_eval(fom_n=n)
    dl = (a - b).abs().max().item()
    check(dl == 0.0 and not torch.equal(a, base), f'eval N = {n}: max |d logit| vs Cell 6b-7 _fom_stack {dl:.1e} (and differs from N = L)')

# 4./5. train against Cell 6b-7, checkpointing off and on
rec = {}
for ck in (False, True):
    a, b = train_rec(n=3, ckpt=ck), train_rec(fom_n=3, ckpt=ck)
    rec[ck] = a
    check(a['loss'] == b['loss'] and torch.equal(a['logits'], b['logits']) and gdiff(a, b) == 0.0,
          f'train N = 3, layer checkpointing {"on" if ck else "off"}: loss {a["loss"]:.6f} vs {b["loss"]:.6f}, '
          f'max |d grad| {gdiff(a, b):.1e} ({len(a["grads"])} params)')
    if getattr(model.cfg, 'register_repulsion', False):
        check(a['nrep'] == 3, f'  repulsion terms: {a["nrep"]} (one per step; Cell 6b-7\'s loop, resetting at each layer-0 step, keeps {b["nrep"]})')
    check(a['j'] is None, '  _stack_j is None after the walk')
dck = gdiff(rec[False], rec[True])
check(rec[False]['loss'] == rec[True]['loss'] and dck < 1e-6, f'N = 3: checkpointing on vs off, loss equal, max |d grad| {dck:.1e}')
r2 = train_rec(n=None)
check(r2['nrep'] == (L if getattr(model.cfg, 'register_repulsion', False) else 0), f'trained walk: {r2["nrep"]} repulsion terms (L = {L})')

# 6. refused combination
model.eval(); model.cfg.substeps_per_layer = 2; model.stack_steps = 3
try:
    with torch.enable_grad():
        model(x); refused = False
except ValueError:
    refused = True
finally:
    model.cfg.substeps_per_layer = 1; model.stack_steps = None
check(refused, 'stack_steps with substeps_per_layer > 1 raises ValueError')

# 7. Cell 6 fragments
c6 = [''.join(c['source']) for c in json.load(open(P.G.NB))['cells']
      if ''.join(c['source']).startswith('# == Cell 6: Training loop')][0]
i0 = c6.index('rng = np.random.default_rng(SEED)\n')
i1 = c6.index('_vs_run = {}') ; i1 = c6.index('\n', i1) + 1
for counts in (None, (2, 3, 4)):
    ns = {'np': np, 'SEED': 0, 'VARIABLE_STEP_COUNTS': counts}
    exec(c6[i0:i1], ns)
    if counts is None:
        check(ns['_vs_counts'] is None, 'OFF: _vs_counts is None')
        continue
    ref_batches = np.random.default_rng(0).integers(0, 10**9, size=5)
    draws = [int(ns['_vs_rng'].choice(ns['_vs_counts'])) for _ in range(19_250)]
    check(np.array_equal(ns['rng'].integers(0, 10**9, size=5), ref_batches), 'the batch generator is untouched by the draws')
    cnt = {n: draws.count(n) for n in (2, 3, 4)}
    check(set(draws) == {2, 3, 4} and all(abs(c - 19_250 / 3) < 4 * math.sqrt(19_250 * 2 / 9) for c in cnt.values()),
          f'19,250 draws: {cnt}, mean N {np.mean(draws):.3f}')
    st = ns['_vs_rng'].bit_generator.state
    nxt = [int(ns['_vs_rng'].choice(ns['_vs_counts'])) for _ in range(20)]
    ns2 = {'np': np, 'SEED': 0, 'VARIABLE_STEP_COUNTS': counts}
    exec(c6[i0:i1], ns2)
    ns2['_vs_rng'].bit_generator.state = json.loads(json.dumps(st))
    check([int(ns2['_vs_rng'].choice(ns2['_vs_counts'])) for _ in range(20)] == nxt, 'resume: the saved state continues the draw')
j0 = c6.index('            # SR4a: the training loss by step count')
j1 = c6.index('            _vs_run.clear()\n') + len('            _vs_run.clear()\n')
frag = '\n'.join(l[12:] for l in c6[j0:j1].splitlines())
ns = {'_vs_counts': (2, 3, 4), '_vs_run': {3: [6.0, 3], 2: [4.5, 3]}}
exec(frag, ns)
check(ns['_vs_log'] == {'vs_ntp_by_n': {'2': 1.5, '3': 2.0}, 'vs_steps_by_n': {'2': 3, '3': 3}} and ns['_vs_str'] == 'vs=[N2:1.5 N3:2.0]  '
      and ns['_vs_run'] == {}, f"log entry {ns['_vs_log']}, print '{ns['_vs_str']}'")
ns = {'_vs_counts': None, '_vs_run': {}}
exec(frag, ns)
check(ns['_vs_log'] == {} and ns['_vs_str'] == '', 'OFF: no log keys, nothing printed')

# 8. the refinement probe against Cell 6b-7
if PROBE:
    k0 = c6.index('_VS_PROBE_BATCHES = []')
    k1 = c6.index('def run_causal_probe(step_num):')
    logged = []
    ns = {'np': np, 'torch': torch, 'math': math, 'json': json, 'gc': __import__('gc'), 'model': model,
          'get_batch': g['get_batch'], 'val_ids': np.load(P.G.VAL), 'BLOCK_SIZE': g['BLOCK_SIZE'], 'DEVICE': 'cpu',
          'VARIABLE_STEP_PROBE_N': (1, 1.5, 4), '_log_write': logged.append}
    exec(c6[k0:k1], ns)
    ck_before = model.cfg.use_layer_checkpoint
    model.train(); model.stack_steps = None
    t1 = time.time(); ns['vs_refinement_probe'](19_000)
    after = (model.stack_steps, model.training, model.cfg.use_layer_checkpoint)
    rec = json.loads(logged[-1])
    print(f'  probe {time.time() - t1:.0f} s: {rec}')
    batches = ns['_VS_PROBE_BATCHES']
    ns67['_f7_batches'] = batches
    r67 = np.random.default_rng(20260920)
    same = all(np.array_equal(a[0], b[0]) for a, b in zip(batches, [g['get_batch'](ns['val_ids'], 4, g['BLOCK_SIZE'], r67) for _ in range(12)]))
    check(same and len(batches) == 12, "probe batches are Cell 6b-7's (seed 20260920, 12 x 4)")
    model.eval()
    for n in (2, 3, 8):
        with fom(model, n, T / n, 'hold'):
            ls = []
            for xb, yb in batches:
                with torch.enable_grad():
                    ls.append(model(torch.from_numpy(xb), torch.from_numpy(yb))[1].item())
        v = float(np.mean(ls))
        check(abs(round(v, 5) - rec['loss_by_n'][str(n)]) < 1e-9, f'probe N = {n}: {rec["loss_by_n"][str(n)]:.5f} vs Cell 6b-7 {v:.5f}')
    check(after == (None, True, ck_before), f'right after the probe: stack_steps None, train mode, layer checkpointing restored {after}')

print(f'\nALL PASS ({len(ok)} checks, {time.time() - t0:.0f} s)' if all(ok) else f'\nSOME FAIL ({ok.count(False)} of {len(ok)})')
sys.exit(0 if all(ok) else 1)
