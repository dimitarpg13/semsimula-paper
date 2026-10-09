"""Per-layer flow inference mode check: cfg.substeps_per_layer (protocol SS5.19, 2026-10-09).

k = 1 (the default) must be bit-identical to HEAD on SR2's and PM1's
configurations, on their trained weights: eval logits, train-mode loss and
every parameter gradient. HEAD comes from `git archive HEAD`, so the working
tree is not touched.

  python3 verify_substeps_switch.py OUT_DIR --dump     # run from the HEAD copy
  python3 verify_substeps_switch.py OUT_DIR            # working copy: compare + k > 1 checks

k > 1, on the trained weights:
  1. SR2: logits at k = 4 and k = 8 equal the FLOW-C / FLOW-R harness
     (refinement_ln_linearisation_split.split_stack, every switch on) at
     N = 8 and N = 16, and the loss on FLOW-R's first batch at k = 8 equals
     the harness's
  2. PM1: logits at k = 4 equal the harness with phi frozen at N = 8
  3. causality at k = 4: tokens replaced at t >= 128 leave t < 128 unchanged
  4. k = 1 set explicitly equals the default; _flow_ctx is None after every
     forward
  5. guards: k > 1 refuses training mode and k < 1
"""
import ast, json, sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); DUMP = '--dump' in sys.argv
sys.argv = [sys.argv[0], str(OUT / 'harness')]
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import verify_pm_switch as P

_B = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
_E = 'L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
SR2_ON = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
ARMS = {'SR2': (_B + 'sr2_' + _E, P.CONFIGS['F3.1'][1] + SR2_ON),
        'PM1': (_B + 'pm64_' + _E, P.CONFIGS['F3.1'][1] + P.PM_ON)}


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
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (folder, repl) in ARMS.items():
        model, g, _ = P.build(folder, repl)
        torch.save(record(model, x, y), OUT / f'head_{name}.pt')
        print(f'dumped HEAD {name} outputs to {OUT}/head_{name}.pt')
    raise SystemExit(0)

from refinement_ln_linearisation_split import split_stack
lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
ok_all = True

# ---- k = 1: bit-identical to HEAD ----
say('k = 1 (default): working copy vs HEAD, on the trained weights')
models = {}
for name, (folder, repl) in ARMS.items():
    model, g, _ = P.build(folder, repl)
    assert model.cfg.substeps_per_layer == 1
    head = torch.load(OUT / f'head_{name}.pt', weights_only=False)
    cur = record(model, x, y)
    d_eval = float((cur['eval'] - head['eval']).abs().max())
    d_train = float((cur['train'] - head['train']).abs().max())
    d_grad = max(float((cur['grads'][k] - head['grads'][k]).abs().max()) for k in head['grads'])
    same = set(cur['grads']) == set(head['grads'])
    ok = d_eval == 0 and d_train == 0 and d_grad == 0 and cur['loss'] == head['loss'] and same
    ok_all &= ok
    say(f'   {name}: eval {d_eval:.1e}  train {d_train:.1e}  grads {d_grad:.1e}  ({len(head["grads"])} params, same set: {same})'
        f'  -> {"IDENTICAL" if ok else "DIFFERS"}')
    models[name] = (model, g)

# Cell 6b-7's policy function, verbatim, for the harness
model, g = models['SR2']
nb = json.load(open(P.G.NB))
src = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('# == Cell 6b-7:')][0]
tree = ast.parse(src)
keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
exec(compile(ast.Module(body=keep, type_ignores=[]), 'Cell6b7_defs', 'exec'), g)
policy_index = g['_fom_policy_index']


def logits(m, k=None, ctx=None, xin=x):
    m.eval()
    old = m.cfg.substeps_per_layer
    if k is not None:
        m.cfg.substeps_per_layer = k
    try:
        with (ctx if ctx is not None else torch.enable_grad()), torch.enable_grad():
            out = m(xin)[0].detach()
    finally:
        m.cfg.substeps_per_layer = old
    assert getattr(m, '_flow_ctx', None) is None
    return out


def harness(m, N, phi):
    T = m.cfg.L * m.cfg.dt
    return split_stack(m, N, T / N, policy_index, ln_once=True, freeze=True, xi_freeze=True, phi_freeze=phi)


say('\nk > 1, on the trained weights')
L = model.cfg.L
for k in (4, 8):
    d = float((logits(model, k) - logits(model, ctx=harness(model, k * L, False))).abs().max())
    ok_all &= d == 0.0
    say(f'1. SR2 k = {k} vs the FLOW-C harness at N = {k * L}: max |d logit| {d:.1e}')
rng = np.random.default_rng(20261009)
xb, yb = g['get_batch'](g['val_ids'], 4, 512, rng)
def loss_of(ctx=None, k=None):
    old = model.cfg.substeps_per_layer
    if k is not None:
        model.cfg.substeps_per_layer = k
    try:
        with (ctx if ctx is not None else torch.enable_grad()), torch.enable_grad():
            return float(model(torch.from_numpy(xb), torch.from_numpy(yb))[1])
    finally:
        model.cfg.substeps_per_layer = old
l_mode, l_harn = loss_of(k=8), loss_of(ctx=harness(model, 8 * L, False))
ok_all &= l_mode == l_harn
say(f'   SR2 loss on FLOW-R\'s first batch at k = 8: mode {l_mode:.6f}  harness {l_harn:.6f}  (PPL {np.exp(l_mode):.2f})')

pm, _ = models['PM1']
d = float((logits(pm, 4) - logits(pm, ctx=harness(pm, 4 * L, True))).abs().max())
ok_all &= d == 0.0
say(f'2. PM1 k = 4 vs the harness with phi frozen at N = 8: max |d logit| {d:.1e}')

base = logits(model, 4)
xc = x.clone()
xc[:, 256 // 2:] = torch.randint(0, model.cfg.vocab_size, xc[:, 128:].shape, generator=torch.Generator().manual_seed(1))
d = float((logits(model, 4, xin=xc)[:, :128] - base[:, :128]).abs().max())
ok_all &= d == 0.0
say(f'3. causality at k = 4: tokens replaced at t >= 128, max |d logit| at t < 128 = {d:.1e}')

d = float((logits(model, 1) - logits(model)).abs().max())
ok_all &= d == 0.0
say(f'4. k = 1 set explicitly vs the default: max |d logit| {d:.1e}; _flow_ctx None after every forward: True')

refused = []
model.train(); model.cfg.substeps_per_layer = 4
try:
    model(x)
    refused.append(False)
except RuntimeError:
    refused.append(True)
model.eval(); model.cfg.substeps_per_layer = 0
try:
    model(x)
    refused.append(False)
except ValueError:
    refused.append(True)
model.cfg.substeps_per_layer = 1
ok_all &= all(refused) and getattr(model, '_flow_ctx', None) is None
say(f'5. guards: refuses training mode {refused[0]}, refuses k = 0 {refused[1]}')

say(f'\n-> {"ALL PASS" if ok_all else "FAILURES ABOVE"}')
(HERE / 'verify_substeps_switch_output.txt').write_text('\n'.join(lines) + '\n')
