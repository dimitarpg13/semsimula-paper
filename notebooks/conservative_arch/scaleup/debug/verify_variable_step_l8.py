#!/usr/bin/env python3
"""Variable-step training at L = 8 with step counts above 9 (K2-L8-R6 at N in {8, 12, 16}; 2026-11-05).

    python3 verify_variable_step_l8.py OUT

verify_variable_step_switch.py checks the switch at L = 2 on SO-CODE. This script checks what changes at L = 8
and for two-digit counts, on K2-L8-R6-CODE's trained weights (step 18,000) built through the notebook's Cells 0-5b:
  1. tags: SO-CODE, L8-CODE, K2-L8-CODE, K2-L8-R6-CODE and SR4a-CODE ('vs234') unchanged; with
     VARIABLE_STEP_COUNTS = (8, 12, 16) the tag is K2-L8-R6-CODE's plus 'vs8-12-16'; copy_split.tag_repl round
     trip for all six; Cell 0 refuses one count and a fractional count
  2. the walk: which trained layer each step uses, and each layer's share of T, at N = 8 to 16
  3. eval, N = 4, 6, 9-16, 24, 32: the model's walk equals Cell 6b-7's _fom_stack ('hold', dt = T/N), bit for bit;
     N = 8 equals the trained walk
  4. train, N = 12 and 16: loss and every gradient equal _fom_stack's (layer checkpointing off); N = 12 with
     layer checkpointing on gives the same loss and gradients
  5. the probe's step counts for the Cell 0 VARIABLE_STEP_PROBE_N of the run
  6. memory: bytes saved for the backward by one training forward at N = 8, 12 and 16 (2 x 512 tokens, the GPU
     run's switches), scaled to the GPU at 16 x 2 by K2-L8-R6-CODE's measured peak at N = 8
"""
import contextlib, io, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
DBG = Path(__file__).resolve().parent
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(DBG))
import verify_pm_switch as P
torch.set_num_threads(8)
P.G.VAL = Path.home() / 'Downloads/semsimula_code_gpt2/data/code_val_2M.npy'

ok = []
def check(cond, msg):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {msg}", flush=True)

t0 = time.time()
# ---- 1. tags ------------------------------------------------------------------------------------------
SO9 = [("REVERSE_CHANNEL              = True", "REVERSE_CHANNEL              = False"),
       ("VPHI_GRAD_PATH         = 'default'", "VPHI_GRAD_PATH         = 'live'"),
       ("XI_GRAD_PATH           = 'default'", "XI_GRAD_PATH           = 'live'"),
       ("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),
       ("VPHI_ANALYTIC_FORCE = False", "VPHI_ANALYTIC_FORCE = True"),
       ("LAYER_CHECKPOINT = True", "LAYER_CHECKPOINT = False"),
       ("CORPUS           = 'openwebtext'", "CORPUS           = 'code'"),
       ("LADDER_STEPS     = 32_500", "LADDER_STEPS     = 16_250"),
       ("EXTRA_FLOOR_STEPS = 0", "EXTRA_FLOOR_STEPS = 3_000")]
L8 = [("LADDER_L         = 2", "LADDER_L         = 8")]
K2 = [("V_THETA_WELLS_PER_HEAD      = 8", "V_THETA_WELLS_PER_HEAD      = 2")]
SNAP = [("SNAPSHOT_STEPS = []", "SNAPSHOT_STEPS = [1_000, 1_500, 2_000, 2_500, 3_000]")]
VFC = [("VPHI_FORCE_CHECKPOINT = False", "VPHI_FORCE_CHECKPOINT = True")]
R6 = [("ANISO_RANK                  = 4", "ANISO_RANK                  = 6")]
vs = lambda counts: [("VARIABLE_STEP_COUNTS      = None", f"VARIABLE_STEP_COUNTS      = {counts}")]
c0 = P.G.cells['Cell 0:']


def cell0(repl):
    s = c0
    for a, b in repl:
        line = next(l for l in s.splitlines() if l.startswith(a))
        s = s.replace(line, line.replace(a, b), 1)
    g = {'__name__': '__main__'}; out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(P.G.strip(s), 'Cell0', 'exec'), g)
    return g, out.getvalue()


SO_TAG = ('xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_L2probe_ob_untied_wsd_e5c_'
          'plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn')
L8_TAG = SO_TAG.replace('L2probe', 'L8probe').replace('_idt4', '')
K2L8_TAG = L8_TAG.replace('dcvt5x8', 'dcvt5x2')
R6_TAG = K2L8_TAG.replace('dcvt5x2_', 'dcvt5x2_ar6_')
VS_TAG = R6_TAG.replace('_sr2_st16250_', '_sr2_vs8-12-16_st16250_')
K2L8 = SO9 + L8 + K2 + SNAP + VFC
no_lc = [r for r in SO9 if not r[0].startswith('LAYER_CHECKPOINT')]
arms = (("SO-CODE", SO9, SO_TAG), ("L8-CODE", no_lc + L8 + SNAP, L8_TAG), ("K2-L8-CODE", K2L8, K2L8_TAG),
        ("K2-L8-R6-CODE", K2L8 + R6, R6_TAG), ("SR4a-CODE", SO9 + vs('(2, 3, 4)'), SO_TAG.replace('_sr2_', '_sr2_vs234_')),
        ("K2-L8-R6, N in {8, 12, 16}", K2L8 + R6 + vs('(8, 12, 16)'), VS_TAG))
for name, repl, want in arms:
    g, out = cell0(repl)
    check(g['_variant_tag'] == want, f"{name}: tag {'as expected' if g['_variant_tag'] == want else g['_variant_tag']}")
    if 'N in' in name:
        check('[SR4a] VARIABLE_STEP_COUNTS=(8, 12, 16)' in out, '  Cell 0 banner printed')
_src = (DBG / 'copy_split.py').read_text()
_argv = sys.argv; sys.argv = ['copy_split.py', str(OUT / 'harness'), 'none']
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(_src[:_src.index("\nif MODE == 'nll':")], 'copy_split.py', 'exec'), CS)
sys.argv = _argv
for _, _, tag in arms:
    repl, t = CS['tag_repl']('semsimula_fock_cfc_baoab_code_' + tag)
    g, _ = cell0(repl)
    check(g['_variant_tag'] == tag == t, f"tag_repl round trip ...{tag[tag.find('dcvt'):][:12]}...{tag[tag.find('sr2'):][:20]}"
          f" -> VARIABLE_STEP_COUNTS {g['VARIABLE_STEP_COUNTS']}")
for bad in ('(8,)', '(8, 12.5)'):
    try:
        cell0(K2L8 + R6 + vs(bad)); refused = False
    except AssertionError:
        refused = True
    check(refused, f'Cell 0 refuses VARIABLE_STEP_COUNTS = {bad}')

# ---- 2. the walk ---------------------------------------------------------------------------------------
L, T = 8, 8.0
print('\n  the walk (step j uses trained layer min(j*L // N, L-1)); share of T per layer, in units of the trained dt:')
for n in range(8, 17):
    lay = [min(j * L // n, L - 1) for j in range(n)]
    share = [lay.count(l) * L / n for l in range(L)]
    print(f'    N = {n:2d}: layers {lay}\n            share  {[round(s, 2) for s in share]}')
check(all(min(j * L // 16, L - 1) == j // 2 for j in range(16)), 'N = 16: every layer twice, at half the trained step')

# ---- build the trained model with the switch on ---------------------------------------------------------
FOLDER = 'semsimula_fock_cfc_baoab_code_' + R6_TAG
CK = 'fock_cfc_code_' + R6_TAG + '_step18000_best.published.pt'
repl, _ = CS['tag_repl']('semsimula_fock_cfc_baoab_code_' + VS_TAG)
model, g, out = P.build(FOLDER, repl, ckpt=CK)
check(g['_variant_tag'] == VS_TAG and model.cfg.L == 8 and model.V_theta.bank.bank.rank == 6,
      f'built K2-L8-R6-CODE (step 18,000) with the switch on ({time.time() - t0:.0f} s): L = {model.cfg.L}, '
      f'rank {model.V_theta.bank.bank.rank}, K = {model.V_theta.bank.bank.K}')
x, y = P.tokens(n_seq=2)
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
    loss.backward()
    gr = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True); model._repulsion_terms = []
    return {'logits': lg.detach(), 'loss': loss.item(), 'grads': gr, 'j': model._stack_j}


def gdiff(a, b):
    assert set(a['grads']) == set(b['grads'])
    return max((a['grads'][k] - b['grads'][k]).abs().max().item() for k in a['grads'])


# ---- 3. eval walk against Cell 6b-7 -----------------------------------------------------------------------
base = logits_eval()
check(torch.equal(logits_eval(n=8), base), 'eval: stack_steps = 8 (= L) bit-identical to the trained walk')
for n in (4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32):
    a, b = logits_eval(n=n), logits_eval(fom_n=n)
    dl = (a - b).abs().max().item()
    check(dl == 0.0 and not torch.equal(a, base), f'eval N = {n:2d}: max |d logit| vs Cell 6b-7 _fom_stack {dl:.1e}')

# ---- 4. training against Cell 6b-7 ----------------------------------------------------------------------
rec = {}
for n in (12, 16):
    a, b = train_rec(n=n), train_rec(fom_n=n)
    rec[n] = a
    check(a['loss'] == b['loss'] and torch.equal(a['logits'], b['logits']) and gdiff(a, b) == 0.0 and a['j'] is None,
          f'train N = {n}: loss {a["loss"]:.6f} vs {b["loss"]:.6f}, max |d grad| {gdiff(a, b):.1e} ({len(a["grads"])} params)')
c = train_rec(n=12, ckpt=True)
dck = gdiff(rec[12], c)
check(c['loss'] == rec[12]['loss'] and dck < 1e-6, f'N = 12: layer checkpointing on vs off, loss equal, max |d grad| {dck:.1e}')

# ---- 5. the probe's step counts -------------------------------------------------------------------------
PROBE_N = (0.5, 0.75, 1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2, 3, 4)
ns = sorted({int(round(f * L)) for f in PROBE_N} | {L})
check(ns == [4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32], f'VARIABLE_STEP_PROBE_N = {PROBE_N} -> N = {ns}')

# ---- 6. memory --------------------------------------------------------------------------------------------
model.cfg.vphi_analytic_force = True; model.cfg.vphi_force_checkpoint = True; model.cfg.use_layer_checkpoint = False
val = np.load(P.G.VAL)
xb = torch.from_numpy(val[:2 * 513].astype(np.int64)).view(2, 513)


def saved(n):
    seen = {}
    def pack(t):
        try:
            s = t.untyped_storage(); seen[s.data_ptr()] = s.nbytes()
        except Exception:
            pass
        return t
    model.train(); torch.manual_seed(7); model.stack_steps = None if n == L else n
    try:
        with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = model(xb[:, :-1], xb[:, 1:])
    finally:
        model.stack_steps = None
    del out; model._repulsion_terms = []
    return sum(seen.values()) / 2**30


gib = {n: saved(n) for n in (8, 12, 16)}
GPU_N8 = 37.1                              # K2-L8-R6-CODE's logged training peak at 16 x 2, N = 8
fixed = GPU_N8 - 8.59 * gib[8]             # 8.59 = (batch 16 / 2) x (GiB -> GB)
print('\n  saved for the backward, one training forward, 2 x 512 tokens:')
for n, v in gib.items():
    print(f'    N = {n:2d}: {v:6.3f} GiB  ->  GPU at 16 x 2 about {fixed + 8.59 * v:5.1f} GB')
print(f'    (fixed part {fixed:.1f} GB, from the N = 8 peak)')
check(fixed + 8.59 * gib[16] < 80.0, f'N = 16 fits an 80 GB card without layer checkpointing ({fixed + 8.59 * gib[16]:.1f} GB)')

print(f'\nALL PASS ({len(ok)} checks, {time.time() - t0:.0f} s)' if all(ok) else f'\nSOME FAIL ({ok.count(False)} of {len(ok)})')
sys.exit(0 if all(ok) else 1)
