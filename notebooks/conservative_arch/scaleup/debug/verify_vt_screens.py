#!/usr/bin/env python3
"""Dead-wells screen switches: checks (protocol p-dead-wells, 2026-10-29).

    python3 verify_vt_screens.py OUT

Screen A (VTHETA_INPUT_UNIT_NORM, tag 'vtun') and screen B (VTHETA_LR_MULT, tag 'vtlr<m>'), on XI6-CODE-WR's
step-500 weights, built through the ladder notebook's own Cells 0-5b (verify_pm_switch.build); Cell 6's optimizer
block and its schedule line are taken from the notebook and run as written. The OFF bit-identity of Cells 0-5b
is checked separately with verify_well_reach.py dump/compare (HEAD worktree against the working copy).
"""
import json, os, re, sys
from pathlib import Path

import torch

OUT = Path(sys.argv[1])
sys.argv = [sys.argv[0], str(OUT / 'harness')]
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))
import verify_pm_switch as P
torch.set_num_threads(8)
P.G.VAL = Path(os.path.expanduser('~/Downloads/semsimula_code_gpt2/data/code_val_2M.npy'))

T = ('xi6long_topk16_dt32da16_mh4_aniso_dcvt6x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_wr3l0p1_L2probe_ob_untied_wsd_e5c_'
     'plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn')
FOLDER, CK = 'semsimula_fock_cfc_baoab_code_' + T, 'fock_cfc_code_' + T + '_step500_best.pt'
BASE = P.LIVE + (('LOWRANK_DAMPED_FLOW  = False', 'LOWRANK_DAMPED_FLOW  = True'), ('LADDER_STEPS     = 32_500', 'LADDER_STEPS     = 16_250'),
                 ('REVERSE_CHANNEL              = True', 'REVERSE_CHANNEL              = False'),
                 ("XI_OVERRIDE     = '5long'", "XI_OVERRIDE     = '6long'"),
                 ('VPHI_ANALYTIC_FORCE = False', 'VPHI_ANALYTIC_FORCE = True'), ('LAYER_CHECKPOINT = True', 'LAYER_CHECKPOINT = False'),
                 ('WELL_REACH_RHO    = None', 'WELL_REACH_RHO    = 3'))
A = (('VTHETA_INPUT_UNIT_NORM = False', 'VTHETA_INPUT_UNIT_NORM = True'),)
B = (('VTHETA_LR_MULT         = None', 'VTHETA_LR_MULT         = 0.25'),)
x, y = P.tokens(n_seq=2)
x, y = torch.as_tensor(x), torch.as_tensor(y)

ok = []
def check(cond, msg):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {msg}", flush=True)

# Cell 6's optimizer block and schedule line, as written in the notebook
nb = json.load(open(P.G.REPO / 'notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb'))
c6 = next(''.join(c['source']) for c in nb['cells'] if 'def split_decay_params' in ''.join(c['source']))
i0 = c6.index('def split_decay_params'); i1 = c6.index('\n\n\n', i0)
CONSTS = ''.join(m.group(0) + '\n' for m in re.finditer(r'(?m)^(LR|WEIGHT_DECAY|NO_DECAY_1D)\s*=.*$', c6))
OPT_SRC = CONSTS + c6[i0:i1] + '\n\n_trainable = [p for p in model.parameters() if p.requires_grad]\n' + \
    c6[c6.index("if OPTIMIZER == 'adamw':"):c6.index("if DEVICE == 'cuda':\n    _free, _total = torch.cuda.mem_get_info()")]
LR_LINE = re.search(r"\n( +)for g in optim\.param_groups:\n\1    (g\['lr'\] = .+)\n", c6).group(2)
print('Cell 6 schedule line:', LR_LINE)

def build(repl):
    model, g, out = P.build(FOLDER, repl, ckpt=CK)
    model.cfg.use_layer_checkpoint = False
    return model, g, out

WD = [None]
def optimizer(model, g):
    g = dict(g); g['model'] = model
    exec(compile(OPT_SRC, 'Cell6-optimizer', 'exec'), g)
    WD[0] = g['WEIGHT_DECAY']
    return g['optim']

def bank_inputs(model, layer):
    """At one layer: the context channels the layer computed, and the channels as they enter the bank
    (the depth-conditioned V_theta's shift, with its depth code taken off again)."""
    VT = model.V_theta; got = {}; orig = VT._shift
    def shift(xis):
        out = orig(xis)
        if VT._active_layer == layer and 'raw' not in got:
            got['raw'] = xis.detach(); got['in'] = out.detach() - VT.depth_code[layer].detach()
            got['exact'] = torch.equal(out.detach(), xis.detach() + VT.depth_code[layer].detach())
        return out
    VT._shift = shift
    try:
        model.eval()
        with torch.enable_grad():
            model(x)
    finally:
        del VT._shift
    return got['raw'], got['in'], got['exact']

print('\nOFF (both switches at their defaults)')
m0, g0, out0 = build(BASE)
tag0 = g0['_variant_tag']
check('vtun' not in tag0.split('_') and not any(t.startswith('vtlr') for t in tag0.split('_')), f'tag has neither part: ...{tag0[tag0.find("st16250"):][:40]}')
check(m0.V_theta.input_unit_norm is False, 'V_theta.input_unit_norm is False')
o0 = optimizer(m0, g0)
d_ref, nd_ref = [p for p in m0.parameters() if p.requires_grad and p.ndim >= 2], [p for p in m0.parameters() if p.requires_grad and p.ndim < 2]
check(len(o0.param_groups) == 2 and [id(p) for p in o0.param_groups[0]['params']] == [id(p) for p in d_ref]
      and [id(p) for p in o0.param_groups[1]['params']] == [id(p) for p in nd_ref]
      and [pg['weight_decay'] for pg in o0.param_groups] == [WD[0], 0.0]
      and not any('lr_mult' in pg for pg in o0.param_groups),
      'optimizer: the two groups of HEAD ([>= 2-D decayed] + [1-D]), same parameter order, no lr_mult')
vals = [1.2e-3, 7.39e-05, 6e-05, 1.1873e-3]
check(all(eval(LR_LINE.split('=', 1)[1], {'lr_now': v, 'g': pg}) == v for v in vals for pg in o0.param_groups),
      'schedule line gives exactly lr_now to every group')
for l in (0, 1):
    raw, ent, exact = bank_inputs(m0, l)
    check(exact, f'layer {l}: the shift is exactly channels + depth code (bit for bit); channel norms (median) '
          f'{[round(float(v), 2) for v in ent.norm(dim=-1).flatten(0, -2).median(0).values]}')

print('\nSCREEN A (VTHETA_INPUT_UNIT_NORM = True)')
mA, gA, outA = build(BASE + A)
tagA = gA['_variant_tag']
check(tagA == tag0.replace('_wr3l0p1_', '_wr3l0p1_vtun_'), f"tag gains exactly 'vtun': ...{tagA[tagA.find('st16250'):][:40]}")
check('[vtun]' in outA and 'SCREEN A' in outA, 'Cell 0 banner and Cell 5b guard printed')
check(mA.V_theta.input_unit_norm is True, 'V_theta.input_unit_norm is True')
check(set(mA.state_dict()) == set(m0.state_dict()), 'no new parameters or state-dict keys')
for l in (0, 1):
    raw, xi, _ = bank_inputs(mA, l); n = xi.norm(dim=-1)
    cosd = float((torch.nn.functional.cosine_similarity(xi, raw, dim=-1) - 1).abs().max())
    check(float((n - 1).abs().max()) < 1e-4 and cosd < 1e-4,
          f'layer {l}: every channel enters the bank at norm 1 (max |norm - 1| {float((n - 1).abs().max()):.1e}), direction of the computed channel kept (max 1 - cos {cosd:.1e})')
mA.train(); torch.manual_seed(7)
_, lossA = mA(x, y); penA = mA.pop_well_reach_penalty()
(lossA + 0.1 * penA).backward()
gA_ok = all(torch.isfinite(p.grad).all() for p in mA.parameters() if p.grad is not None)
check(torch.isfinite(lossA) and gA_ok, f'train step: loss {lossA.item():.4f} (trained on raw inputs, so higher than off), every gradient finite')
oA = optimizer(mA, gA)
check(len(oA.param_groups) == 2, 'screen A alone leaves the optimizer groups as HEAD')

print('\nSCREEN B (VTHETA_LR_MULT = 0.25)')
mB, gB, outB = build(BASE + B)
tagB = gB['_variant_tag']
check(tagB == tag0.replace('_wr3l0p1_', '_wr3l0p1_vtlr0p25_'), f"tag gains exactly 'vtlr0p25': ...{tagB[tagB.find('st16250'):][:40]}")
check('[vtlr]' in outB and 'SCREEN B' in outB, 'Cell 0 banner and Cell 5b guard printed')
oB = optimizer(mB, gB)
vt = {id(p) for p in mB.V_theta.parameters()}
grp = [(pg.get('lr_mult'), pg['weight_decay'], {id(p) for p in pg['params']}) for pg in oB.param_groups]
allp = {id(p) for p in mB.parameters() if p.requires_grad}
check(sum(len(s) for *_, s in grp) == len(allp) and set().union(*(s for *_, s in grp)) == allp,
      f'every trainable tensor in exactly one of {len(grp)} groups')
check(all((m == 0.25) == (s <= vt) and (m is None) == (not (s & vt)) for m, _, s in grp)
      and set().union(*(s for m, _, s in grp if m == 0.25)) == vt,
      f'the lr_mult groups hold exactly V_theta\'s {len(vt)} tensors (bank + depth code)')
dk = {id(p) for p in mB.parameters() if p.requires_grad and p.ndim >= 2}
check(all(wd == (WD[0] if s <= dk else 0.0) for _, wd, s in grp), 'weight decay as HEAD: >= 2-D decayed, 1-D not')
lrs = {}
for pg in oB.param_groups:
    pg['lr'] = eval(LR_LINE.split('=', 1)[1], {'lr_now': 1.2e-3, 'g': pg})
    lrs[pg.get('lr_mult')] = pg['lr']
check(lrs == {None: 1.2e-3, 0.25: 1.2e-3 * 0.25}, f'schedule line: lr {lrs}')
before = {n: p.detach().clone() for n, p in mB.named_parameters()}
mB.train(); torch.manual_seed(7)
_, lossB = mB(x, y); penB = mB.pop_well_reach_penalty()
oB.zero_grad(set_to_none=True); (lossB + 0.1 * penB).backward(); oB.step()
moved = {n: float((p.detach() - before[n]).abs().max()) for n, p in mB.named_parameters()}
vt_names = {f'V_theta.{n}' for n, _ in mB.V_theta.named_parameters()}
# first Adam step: |update| = lr (+ lr * wd * |p|) for every coordinate with a nonzero gradient
big = max(v for n, v in moved.items() if n not in vt_names)
small = max(v for n, v in moved.items() if n in vt_names)
check(abs(small / big - 0.25) < 0.02, f'one AdamW step: largest V_theta move {small:.3e} vs largest elsewhere {big:.3e} (ratio {small / big:.3f}, expected 0.25)')

print('\nALL PASS' if all(ok) else '\nSOME FAIL')
sys.exit(0 if all(ok) else 1)
