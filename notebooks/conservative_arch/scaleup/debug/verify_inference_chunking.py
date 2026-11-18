#!/usr/bin/env python3
"""Verify the inference-only memory switches (2026-11-17):
  - cfg.vphi_query_chunk        the analytic V_phi force built a block of query rows at a time;
  - xi_module.query_chunk       the content-routed xi channels built a block of query rows at a time;
  - cfg.skip_unread_registers   the Fock registers skipped when nothing reads them (no reverse channel).

    python3 verify_inference_chunking.py OUT FOLDER [chunk=256] [windows=4] [T=1024] [threads=8] [val=<tokens.npy>]

Checks, on a trained ladder model (FOLDER in ~/Downloads, built from its tag) and real validation windows of T tokens:
  1. causal_ema_logits_rows equals the matching rows of causal_ema_logits, bit for bit;
  2. under no_grad, each switch alone and all three together give the logits of the model with every switch off,
     to float rounding (max |d logit| and the loss difference are printed; threshold 1e-4 and 1e-5);
  3. with gradients enabled the two chunking switches are inert: all three on, enable_grad, bit-identical logits;
  4. a short window (T <= chunk) is bit-identical with the switches on (the chunked path is not entered).
Writes OUT/verify_inference_chunking_output.txt.
"""
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))
CHUNK, W, T = int(OPTS.get('chunk', 256)), int(OPTS.get('windows', 4)), int(OPTS.get('T', 1024))

src = (DBG / 'copy_split.py').read_text()
src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', 'windows=1', f"threads={OPTS.get('threads', 8)}"] \
    + ([f"val={OPTS['val']}"] if 'val' in OPTS else [])
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)
import verify_pm_switch as P
_build = P.build; got = {}
def build(*a, **k):
    r = _build(*a, **k); got['model'] = r[0]; return r
P.build = build
_, desc = CS['ladder_model'](FOLDER, OPTS.get('ckpt'))
m = got['model']; m.eval()
# The tag-based build leaves VPHI_ANALYTIC_FORCE at its notebook default (False); the ladder arms trained with it
# on (Cell 0), and the V_phi chunking lives in the analytic force, so it is switched on here as trained.
m.cfg.vphi_analytic_force = True
MX = sys.modules[type(m.xi_module).__module__]          # the module the model's xi channels come from
# engagement counters: a pass of 0.0 must not come from a path that was never entered
CNT = {'vphi_chunks': 0, 'xi_row_blocks': 0, 'register_creations': 0}
_pf = m._pair_force_analytic
def _pf_count(*a, **k):
    if k.get('q_slice') is not None:
        CNT['vphi_chunks'] += 1
    return _pf(*a, **k)
m._pair_force_analytic = _pf_count
_rows = MX.causal_ema_logits_rows
def _rows_count(*a, **k):
    CNT['xi_row_blocks'] += 1
    return _rows(*a, **k)
MX.causal_ema_logits_rows = _rows_count
_fp = m.creation_gate_qkv.forward_prefix
def _fp_count(*a, **k):
    CNT['register_creations'] += 1
    return _fp(*a, **k)
m.creation_gate_qkv.forward_prefix = _fp_count
lines = [f'Inference-only memory switches: {FOLDER}', f'model {desc}', f'{W} windows x {T} tokens, chunk {CHUNK}', '']
ok = []

# 1. the row-block EMA logits
a = torch.tensor(0.95)
full = MX.causal_ema_logits(T, a, torch.float32, torch.device('cpu'))
rows = torch.cat([MX.causal_ema_logits_rows(t0, min(t0 + CHUNK, T), T, a, torch.float32, torch.device('cpu'))
                  for t0 in range(0, T, CHUNK)])
c1 = torch.equal(full, rows); ok.append(c1)
lines.append(f'1. causal_ema_logits_rows against causal_ema_logits: bit-identical {c1}  -> {"PASS" if c1 else "FAIL"}')

v = np.load(CS['VAL']).astype(np.int64)
st = np.linspace(0, len(v) - (T + 1), W).astype(np.int64)
xy = torch.from_numpy(np.stack([v[s:s + T + 1] for s in st]))
x, y = xy[:, :T], xy[:, 1:]


def setsw(vphi=False, xi=False, reg=False):
    m.cfg.vphi_query_chunk = CHUNK if vphi else None
    m.xi_module.query_chunk = CHUNK if xi else None
    m.cfg.skip_unread_registers = bool(reg)


def run(grad=False, xx=x):
    with (torch.enable_grad() if grad else torch.no_grad()):
        lg = m(xx)[0].detach()
    ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), y[:, :xx.shape[1]].reshape(-1))
    return lg, float(ce)


setsw(); CNT.update({k_: 0 for k_ in CNT}); ref, ce_ref = run()
lines.append(f'reference (all off, no_grad): loss {ce_ref:.6f}; counters {dict(CNT)}')
for name, kw in (('V_phi query chunking', dict(vphi=True)), ('xi query chunking', dict(xi=True)),
                 ('skip unread registers', dict(reg=True)), ('all three', dict(vphi=True, xi=True, reg=True))):
    setsw(**kw); CNT.update({k_: 0 for k_ in CNT}); lg, ce = run()
    dmax = float((lg - ref).abs().max()); dce = abs(ce - ce_ref)
    eng = ((not kw.get('vphi') or CNT['vphi_chunks'] > 0) and (not kw.get('xi') or CNT['xi_row_blocks'] > 0)
           and (not kw.get('reg') or CNT['register_creations'] == 0))
    c = dmax <= 1e-4 and dce <= 1e-5 and eng; ok.append(c)
    lines.append(f'2. {name:>22}: max |d logit| {dmax:.2e}, |d loss| {dce:.2e}; engaged {eng} {dict(CNT)}  -> {"PASS" if c else "FAIL"}')

setsw(); refg, _ = run(grad=True)
setsw(vphi=True, xi=True, reg=True); lgg, _ = run(grad=True)
# the register skip is not gated on grad (it is exact); the two chunking switches must not engage
c3 = float((lgg - refg).abs().max()) <= 1e-4; ok.append(c3)
setsw(vphi=True, xi=True); lgg2, _ = run(grad=True)
c3b = torch.equal(lgg2, refg); ok.append(c3b)
lines.append(f'3. enable_grad, chunking switches on: bit-identical {c3b}; with the register skip too, max |d logit| '
             f'{float((lgg - refg).abs().max()):.2e}  -> {"PASS" if c3 and c3b else "FAIL"}')

xs = x[:, :CHUNK]
setsw(); rs, _ = run(xx=xs)
setsw(vphi=True, xi=True); ls_, _ = run(xx=xs)
c4 = torch.equal(rs, ls_); ok.append(c4)
lines.append(f'4. a window of {CHUNK} tokens (no chunk entered): bit-identical {c4}  -> {"PASS" if c4 else "FAIL"}')
setsw()
# 5. the guard: the autograd pair force under no_grad must raise, not return zeros; enable_grad still matches
m.cfg.vphi_analytic_force = False
try:
    run(xx=xs); c5a = False; msg = 'no error'
except RuntimeError as e:
    c5a = 'autograd is disabled' in str(e); msg = str(e)[:70]
lg_ag, _ = run(grad=True, xx=xs)
m.cfg.vphi_analytic_force = True
lg_an, _ = run(xx=xs)
c5b = float((lg_ag - lg_an).abs().max()) <= 1e-3; ok += [c5a, c5b]
lines.append(f'5. guard: autograd force under no_grad raises {c5a} ("{msg}..."); under enable_grad it matches the '
             f'analytic force, max |d logit| {float((lg_ag - lg_an).abs().max()):.2e}  -> {"PASS" if c5a and c5b else "FAIL"}')
lines += ['', f'ALL {"PASS" if all(ok) else "FAIL"}']
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'verify_inference_chunking_output.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
