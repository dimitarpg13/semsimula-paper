#!/usr/bin/env python3
"""
Verify the FO-a switch (Cell 0 FIRST_ORDER_MEMORYLESS, cfg.first_order_memoryless; protocol SS5.13,
2026-10-18) on SR2's trained weights, the FO series' second-order anchor.

    python3 verify_fo_switch.py OUT ref [arm=pmx]            # in a copy of HEAD: write the reference
    python3 verify_fo_switch.py OUT check REF.npz [arm=pmx]  # in the working tree: compare and verify

arm: sr2 (default; the FO 2x2's anchor) or pmx (the exploratory FO-PMX twin, added 2026-10-18).

Checks (working tree):
  1. off is HEAD, bit for bit: eval logits, the training loss and every parameter gradient
     (train mode) against the reference written by this script in a copy of HEAD.
  2. on is Cell 6b-7's Gate 1: eval logits equal the notebook cell's own _fom_stack(N = L,
     policy 'hold', reset_prev=True) on the same weights, bit for bit, and differ from off.
  3. on trains: the loss is finite, every gradient is finite, the same parameters receive a
     gradient as when off, and the gradients differ from off.
  4. the switch adds no parameters: identical state_dict keys and shapes on and off.
"""
import contextlib, io, json, sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); MODE = sys.argv[2]
REF = Path(sys.argv[3]) if MODE == 'check' else None
ARM = next((a[4:] for a in sys.argv[2:] if a.startswith('arm=')), 'sr2')
HERE = Path(__file__).resolve().parent
_argv = list(sys.argv)
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(HERE))
import verify_pm_switch as P
sys.argv = _argv
G = P.G
# threads=1 makes PMX's backward deterministic: at 8 threads one gradient (pm_mu) varies run to run by
# about 5e-10, in HEAD as much as in the working tree, so bit-identity is checked single-threaded there.
torch.set_num_threads(int(next((a[8:] for a in sys.argv[2:] if a.startswith('threads=')), 8)))

SR2 = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
FO = (("FIRST_ORDER_MEMORYLESS = False", "FIRST_ORDER_MEMORYLESS = True"),)
PMXS = (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),)
_T = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
_E = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
FOLDER, REPL = {'sr2': (_T + 'sr2' + _E, P.CONFIGS['F3.1'][1] + SR2),
                'pmx': (_T + 'pm64_pmx16_sr2' + _E, P.CONFIGS['F3.1'][1] + P.PM_ON + SR2 + PMXS)}[ARM]

val = np.load(G.VAL)
xy = torch.from_numpy(val[:2 * 257].astype(np.int64)).view(2, 257)
x, y = xy[:, :256], xy[:, 1:]


def run(model):
    """eval logits; train-mode loss and every gradient (name -> array, None if no gradient)."""
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


off, g_off, _ = P.build(FOLDER, REPL)
off.to('cpu')
lo0, loss0, gr0 = run(off)

if MODE == 'ref':
    np.savez(OUT / f'fo_ref_{ARM}.npz', logits=lo0.numpy(), loss=loss0.numpy(),
             **{f'g::{n}': v.numpy() for n, v in gr0.items() if v is not None},
             nograd=np.array([n for n, v in gr0.items() if v is None], dtype=object))
    print(f'reference written: {OUT / f"fo_ref_{ARM}.npz"}  (loss {float(loss0):.6f}, {len(gr0)} parameters)')
    sys.exit(0)

lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
ok = []
say(f'FO-a switch verification on {ARM.upper()}\'s trained weights (2 x 256 validation tokens)')

# 1. off == HEAD
ref = np.load(REF, allow_pickle=True)
d_lo = float(np.abs(lo0.numpy() - ref['logits']).max()); d_loss = abs(float(loss0) - float(ref['loss']))
gk = {k[3:] for k in ref.files if k.startswith('g::')}
mine = {n for n, v in gr0.items() if v is not None}
d_g = max(float(np.abs(gr0[n].numpy() - ref[f'g::{n}']).max()) for n in gk & mine) if gk & mine else float('nan')
c1 = d_lo == 0.0 and d_loss == 0.0 and gk == mine and d_g == 0.0
ok.append(c1)
say(f'1. off vs HEAD: max |d logit| {d_lo:.1e}; |d loss| {d_loss:.1e}; same parameters with a gradient: '
    f'{gk == mine} ({len(mine)}); max |d grad| {d_g:.1e}  -> {"PASS" if c1 else "FAIL"}')

# 2. on == Cell 6b-7 Gate 1
on, g_on, _ = P.build(FOLDER, REPL + FO)
on.to('cpu')
assert on.cfg.first_order_memoryless is True and off.cfg.first_order_memoryless is False
lo1, loss1, gr1 = run(on)
c7 = next(''.join(c['source']) for c in json.load(open(G.NB))['cells']
          if ''.join(c['source']).startswith('# == Cell 6b-7'))
defs = c7.split('_f7_saved = {', 1)[0]                     # the cell's definitions, before it touches any model
g7 = dict(g_off)
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(G.strip(defs), 'Cell6b-7-defs', 'exec'), g7)
off.eval()
with g7['_fom_stack'](off, off.cfg.L, off.cfg.dt, 'hold', reset_prev=True):
    with torch.enable_grad():
        lo_g1 = off(x)[0].detach().clone()
d_g1 = float((lo1 - lo_g1).abs().max()); d_onoff = float((lo1 - lo0).abs().max())
c2 = d_g1 == 0.0 and d_onoff > 0.0
ok.append(c2)
say(f'2. on vs Cell 6b-7 Gate 1 (_fom_stack, N = L, reset_prev=True): max |d logit| {d_g1:.1e}; '
    f'on vs off: {d_onoff:.2e} (must be > 0)  -> {"PASS" if c2 else "FAIL"}')
nll = lambda lo: float(torch.nn.functional.cross_entropy(lo.reshape(-1, lo.shape[-1]), y.reshape(-1)))
say(f'   eval loss: off {nll(lo0):.4f}, on {nll(lo1):.4f} (perplexity x{np.exp(nll(lo1) - nll(lo0)):.3f} on this batch)')

# 3. on trains
fin = all(torch.isfinite(v).all() for v in gr1.values() if v is not None) and bool(torch.isfinite(loss1))
same_set = {n for n, v in gr1.items() if v is not None} == mine
moved = max(float((gr1[n] - gr0[n]).abs().max()) for n in mine)
c3 = fin and same_set and moved > 0
ok.append(c3)
say(f'3. on, train mode: loss {float(loss1):.4f} (off {float(loss0):.4f}); all finite {fin}; same parameters with a '
    f'gradient {same_set}; max |grad on - grad off| {moved:.2e}  -> {"PASS" if c3 else "FAIL"}')

# 4. no parameters added
sd0, sd1 = off.state_dict(), on.state_dict()
c4 = sd0.keys() == sd1.keys() and all(sd0[k].shape == sd1[k].shape for k in sd0)
ok.append(c4)
say(f'4. state_dict keys and shapes identical on and off ({len(sd0)} tensors)  -> {"PASS" if c4 else "FAIL"}')
say(f'\nALL {"PASS" if all(ok) else "FAIL"}')
(HERE / ('verify_fo_switch_output.txt' if ARM == 'sr2' else f'verify_fo_switch_{ARM}_output.txt')).write_text('\n'.join(lines) + '\n')
