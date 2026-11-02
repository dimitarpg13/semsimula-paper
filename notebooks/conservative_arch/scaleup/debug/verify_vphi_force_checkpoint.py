#!/usr/bin/env python3
"""V_phi-force checkpoint switch check (Cell 0 VPHI_FORCE_CHECKPOINT, 2026-11-02).

    python3 verify_vphi_force_checkpoint.py ROOT OUT dump      # ROOT = a HEAD worktree: reference records
    python3 verify_vphi_force_checkpoint.py ROOT OUT compare   # ROOT = the working copy: OFF must be bit-identical
    python3 verify_vphi_force_checkpoint.py ROOT OUT on        # ON checks on the three code arms

Built through the ladder notebook's own Cells 0-5b (verify_pm_switch.build) on trained weights.
OFF: five arms, each with the notebook defaults and with the speed switches (VPHI_ANALYTIC_FORCE = True,
LAYER_CHECKPOINT = False), 2 x 256 OpenWebText validation tokens; eval logits, train-mode logits (same
seed), the loss and every gradient (verify_pm_switch.record). ON: the tag, the Cell 0 guard, the switch
reaching the model; eval/train logits, loss, gradients and the RNG state after the backward against the
switch off, with layer checkpointing off and on; pair-force calls per training step; saved-for-backward
bytes on 2 x 512 code tokens; Cell 6b-17 run on CPU at a small batch.
"""
import contextlib, io, json, sys, time
from pathlib import Path

import numpy as np
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
SPEED = (('VPHI_ANALYTIC_FORCE = False', 'VPHI_ANALYTIC_FORCE = True'), ('LAYER_CHECKPOINT = True', 'LAYER_CHECKPOINT = False'))
VFC = (('VPHI_FORCE_CHECKPOINT = False', 'VPHI_FORCE_CHECKPOINT = True'),)
CODE_VAL = Path.home() / 'Downloads/semsimula_code_gpt2/data/code_val_2M.npy'
x, y = P.tokens(n_seq=2)
ref = OUT / 'vfc_ref.pt'


def record_or_none(model):
    try:
        return P.record(model, x, y)
    except ValueError as e:          # e.g. the analytic force refuses an exchange potential (G3')
        model.zero_grad(set_to_none=True)
        return str(e)[:80]


if MODE == 'dump':
    recs = {}
    for name, (f, ck, repl) in CONFIGS.items():
        for setting, extra in (('default', ()), ('speed', SPEED)):
            t0 = time.time(); model, g, _ = P.build(f, repl + extra, ckpt=ck)
            recs[name, setting] = record_or_none(model)
            r = recs[name, setting]
            print(f"  dumped {name} {setting}: " + (f"loss {r['loss']:.6f}" if isinstance(r, dict) else f'n/a ({r})')
                  + f'  ({time.time() - t0:.0f} s)', flush=True)
            del model, g
    torch.save(recs, ref)
    sys.exit(0)

if MODE == 'compare':
    h = torch.load(ref); ok = []
    print('OFF (VPHI_FORCE_CHECKPOINT = False): HEAD vs working copy, trained weights')
    for name, (f, ck, repl) in CONFIGS.items():
        for setting, extra in (('default', ()), ('speed', SPEED)):
            model, g, _ = P.build(f, repl + extra, ckpt=ck)
            assert model.cfg.vphi_force_checkpoint is False
            rec, hr = record_or_none(model), h[name, setting]
            if not isinstance(rec, dict) or not isinstance(hr, dict):
                ok.append(rec == hr)
                print(f"  {name:9s} {setting:7s} n/a on both sides: {ok[-1]}  ({rec})", flush=True)
                continue
            assert set(hr['grads']) == set(rec['grads']), name
            de = (hr['eval'] - rec['eval']).abs().max().item()
            dt = (hr['train'] - rec['train']).abs().max().item()
            dg = max((hr['grads'][k] - rec['grads'][k]).abs().max().item() for k in rec['grads'])
            ok.append(de == dt == dg == 0 and hr['loss'] == rec['loss'])
            print(f"  {name:9s} {setting:7s} eval {de:.1e}  train {dt:.1e}  grads {dg:.1e}  loss {rec['loss']:.6f}  "
                  f"({len(rec['grads'])} params)  -> {'IDENTICAL' if ok[-1] else 'DIFFERS'}", flush=True)
            del model, g
    print('ALL IDENTICAL' if all(ok) else 'SOME DIFFER')
    sys.exit(0 if all(ok) else 1)

# ON checks
ok = []
def check(cond, msg):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {msg}", flush=True)


def same(a, b):
    return (a['loss'] == b['loss'] and torch.equal(a['eval'], b['eval']) and torch.equal(a['train'], b['train'])
            and set(a['grads']) == set(b['grads']) and all(torch.equal(a['grads'][k], b['grads'][k]) for k in a['grads']))


def record_rng(model):
    torch.manual_seed(123)
    rec = P.record(model, x, y)
    return rec, torch.get_rng_state()


print('Cell 0 guard: VPHI_FORCE_CHECKPOINT = True without VPHI_ANALYTIC_FORCE')
try:
    P.build(SO, CONFIGS['SO-CODE'][2] + VFC, ckpt=CONFIGS['SO-CODE'][1]); raised = ''
except AssertionError as e:
    raised = str(e)
check('VPHI_ANALYTIC_FORCE' in raised, f'Cell 0 refuses it ({raised[:70]}...)')

for name in ('SO-CODE', 'XI6-CODE', 'FOCK-CODE'):
    f, ck, repl = CONFIGS[name]
    print(f'\n{name}')
    off, g_off, _ = P.build(f, repl + SPEED, ckpt=ck)
    on, g_on, log = P.build(f, repl + SPEED + VFC, ckpt=ck)
    check(g_on['_variant_tag'] == g_off['_variant_tag'], 'tag unchanged')
    check('VPHI_FORCE_CHECKPOINT=True' in log, 'Cell 0 banner printed')
    check(on.cfg.vphi_force_checkpoint is True and off.cfg.vphi_force_checkpoint is False, 'Cell 5 passes the switch to the model')
    check(on.cfg.use_layer_checkpoint is False and on.cfg.vphi_analytic_force is True, 'speed switches in place')
    a, ra = record_rng(off)
    b, rb = record_rng(on)
    check(same(a, b), f'layer ckpt off: eval and train logits, loss, all {len(a["grads"])} gradients bit-identical')
    check(torch.equal(ra, rb), 'RNG state after forward + backward identical')
    off.cfg.use_layer_checkpoint = on.cfg.use_layer_checkpoint = True
    c, _ = record_rng(off)
    d, _ = record_rng(on)
    check(same(c, d), 'layer ckpt on (nested checkpoints): bit-identical to the switch off')
    md = max((a['grads'][k] - d['grads'][k]).abs().max().item() for k in a['grads'])
    print(f'      (layer ckpt on against off, both switch settings: max |d grad| {md:.1e})')
    off.cfg.use_layer_checkpoint = on.cfg.use_layer_checkpoint = False
    n = {'calls': 0}
    _orig = on._pair_force_analytic
    def _count(*a_, **k_):
        n['calls'] += 1; return _orig(*a_, **k_)
    on._pair_force_analytic = _count
    on.train(); torch.manual_seed(7); _, loss = on(x, y)
    fwd = n['calls']; loss.backward(); on.zero_grad(set_to_none=True)
    check(fwd == on.cfg.L and n['calls'] == 2 * on.cfg.L, f'pair force: {fwd} calls in the forward, {n["calls"] - fwd} recomputes in the backward (L = {on.cfg.L})')
    on.eval(); n['calls'] = 0
    with torch.no_grad():
        e_on = on(x)[0]
    off.eval()
    with torch.no_grad():
        e_off = off(x)[0]
    check(torch.equal(e_on, e_off) and n['calls'] == on.cfg.L, 'under no_grad: no checkpoint, logits identical')
    on._pair_force_analytic = _orig
    if name == 'SO-CODE':
        val = np.load(CODE_VAL)
        xb = torch.from_numpy(val[:2 * 513].astype(np.int64)).view(2, 513)
        def saved(m):
            seen = {}
            def pack(t):
                try: s = t.untyped_storage(); seen[s.data_ptr()] = s.nbytes()
                except Exception: pass
                return t
            m.train(); torch.manual_seed(7)
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                m(xb[:, :-1], xb[:, 1:])
            return sum(seen.values()) / 2**30
        s_off, s_on = saved(off), saved(on)
        print(f'      saved for backward, 2 x 512 code tokens, L = 2, K = 8: off {s_off:.3f} GiB, on {s_on:.3f} GiB')
        check(s_on < 0.5 * s_off, 'saved tensors at least halved')
        # Cell 6b-17 on CPU at a small batch: the new row runs, the switch is restored
        nbc = [''.join(c['source']) for c in json.load(open(P.G.NB))['cells']]
        c17 = next(s for s in nbc if s.startswith('# == Cell 6b-17:'))
        g_on.update(V17_BATCH=1, V17_BLOCK=128, CKPT_DIR=Path('/nonexistent'))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(P.G.strip(c17), 'Cell6b17', 'exec'), g_on)
        row = [l for l in out.getvalue().splitlines() if 'V_phi ckpt' in l]
        print('      6b-17: ' + (row[0].strip() if row else 'no row'))
        check(bool(row) and 'out of memory' not in row[0], 'Cell 6b-17 times the new row')
        check(on.cfg.vphi_force_checkpoint is True and on.cfg.use_layer_checkpoint is False
              and on.cfg.vphi_analytic_force is True, 'Cell 6b-17 restores the switches')
    del off, on, g_off, g_on

print(f'\n{sum(ok)}/{len(ok)} PASS')
sys.exit(0 if all(ok) else 1)
