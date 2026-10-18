#!/usr/bin/env python3
"""
Verify selective bf16 (Cell 0 AMP_BF16, 2026-10-18) on trained weights.

    python3 verify_bf16_switch.py OUT ref [arm=sr2|pmx|g3p] [threads=1]            # in a copy of HEAD
    python3 verify_bf16_switch.py OUT check REF.npz [arm=sr2|pmx|g3p] [threads=1]  # in the working tree

Checks (working tree):
  1. autocast off is HEAD, bit for bit: eval logits, the training loss and every parameter gradient
     (this also covers the analytic-V_phi edits of the same day with VPHI_ANALYTIC_FORCE off);
  2. under bf16 autocast (CPU autocast here; the notebook uses CUDA's): train and eval run, the loss and
     every gradient are finite, the well generator's outputs are bf16, and every input to the
     eigensolves and the exact substeps is float32 -- the selective design;
  3. descriptive: how far bf16 moves the eval logits, the training loss and the gradient.
"""
import contextlib, sys
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
torch.set_num_threads(int(next((a[8:] for a in sys.argv[2:] if a.startswith('threads=')), 8)))

SR2 = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
PMXS = (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),)
QK = (("LADDER_MECHANISM = 'none'", "LADDER_MECHANISM = 'attention_potential'"),
      ("RELAX_GRAD_PATH        = 'default'", "RELAX_GRAD_PATH        = 'live'"),
      ("RELAX_ATTN_QK_NORM          = False", "RELAX_ATTN_QK_NORM          = True"),
      ("RELAX_FIELD_CLIP            = None", "RELAX_FIELD_CLIP            = 0.3"))
_T = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
_E = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
FOLDER, REPL = {'sr2': (_T + 'sr2' + _E, P.CONFIGS['F3.1'][1] + SR2),
                'pmx': (_T + 'pm64_pmx16_sr2' + _E, P.CONFIGS['F3.1'][1] + P.PM_ON + SR2 + PMXS),
                'g3p': (P._P + 'rglive_vplive_xilive_rfqk_rfclip0p3_L2probe' + P._S + 'attnpot', P.LIVE + QK)}[ARM]

val = np.load(G.VAL)
xy = torch.from_numpy(val[:2 * 257].astype(np.int64)).view(2, 257)
x, y = xy[:, :256], xy[:, 1:]


def run(model, amp=False):
    ctx = (lambda: torch.autocast('cpu', dtype=torch.bfloat16)) if amp else contextlib.nullcontext
    model.eval()
    with torch.enable_grad(), ctx():
        lo = model(x)[0].detach().float().clone()
    model.train(); model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    with ctx():
        _, loss = model(x, targets=y)
    loss.backward()
    grads = {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in model.named_parameters()}
    model.eval()
    return lo, loss.detach().float().clone(), grads


model, g, _ = P.build(FOLDER, REPL)
model.to('cpu')
lo0, loss0, gr0 = run(model)

if MODE == 'ref':
    np.savez(OUT / f'bf16_ref_{ARM}.npz', logits=lo0.numpy(), loss=loss0.numpy(),
             **{f'g::{n}': v.numpy() for n, v in gr0.items() if v is not None})
    print(f'reference written ({ARM}): loss {float(loss0):.6f}')
    sys.exit(0)

lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
ok = []
say(f'Selective bf16 verification on {ARM.upper()}\'s trained weights (2 x 256 validation tokens)')
ref = np.load(REF)
gk = {k[3:] for k in ref.files if k.startswith('g::')}
mine = {n for n, v in gr0.items() if v is not None}
d_lo = float(np.abs(lo0.numpy() - ref['logits']).max()); d_loss = abs(float(loss0) - float(ref['loss']))
d_g = max(float(np.abs(gr0[n].numpy() - ref[f'g::{n}']).max()) for n in gk & mine)
c1 = d_lo == 0 and d_loss == 0 and gk == mine and d_g == 0
ok.append(c1)
say(f'1. autocast off vs HEAD: max |d logit| {d_lo:.1e}; |d loss| {d_loss:.1e}; same parameters with a gradient '
    f'{gk == mine} ({len(mine)}); max |d grad| {d_g:.1e}  -> {"PASS" if c1 else "FAIL"}')

# 2. under bf16: record dtypes at the generator's output and at the eigensolve / substep inputs
import model_parf_multixi as MPX
seen = {'gen': set(), 'core': set()}
wrapped = []
def wrap(owner, name, kind, outputs=False):
    if not hasattr(owner, name):
        return
    f0 = getattr(owner, name)
    def w(*a, **k):
        out = f0(*a, **k)
        ts = out if outputs else list(a) + list(k.values())
        flat = []
        def walk(v):
            if torch.is_tensor(v) and v.is_floating_point():
                flat.append(v.dtype)
            elif isinstance(v, (list, tuple)):
                for u in v: walk(u)
        walk(ts)
        seen[kind].update(flat)
        return out
    setattr(owner, name, w); wrapped.append((owner, name, f0))
wrap(model.V_theta, 'context_components', 'gen', outputs=True)
for fn in ('lowrank_modes', 'indefinite_lowrank_modes_chol', 'lowrank_damped_substep', 'lowrank_iso_damped_substep',
           'lowrank_cfc_substep', 'cfc_substep', 'ou_step'):
    wrap(MPX, fn, 'core')
try:
    lo1, loss1, gr1 = run(model, amp=True)
finally:
    for o, n, f0 in wrapped:
        setattr(o, n, f0)
fin = bool(torch.isfinite(loss1)) and all(torch.isfinite(v).all() for v in gr1.values() if v is not None)
c2 = fin and seen['gen'] == {torch.bfloat16} and seen['core'] == {torch.float32}
ok.append(c2)
say(f'2. bf16 autocast: train and eval run; loss and gradients finite {fin}; well generator outputs {sorted(map(str, seen["gen"]))}; '
    f'eigensolve and substep inputs {sorted(map(str, seen["core"]))}  -> {"PASS" if c2 else "FAIL"}')
D = lambda t: t.double()                                     # accumulate in float64: ~30M-element sums
gt = torch.sqrt(sum(D(gr0[n]).norm() ** 2 for n in mine))
dg = torch.sqrt(sum((D(gr1[n]) - D(gr0[n])).norm() ** 2 for n in mine if gr1[n] is not None))
cos = float(sum((D(gr1[n]) * D(gr0[n])).sum() for n in mine if gr1[n] is not None) /
            (gt * torch.sqrt(sum(D(gr1[n]).norm() ** 2 for n in mine if gr1[n] is not None))))
say(f'3. (descriptive) bf16 against float32: eval logits max |d| {float((lo1 - lo0).abs().max()):.2e}; '
    f'eval loss {float(torch.nn.functional.cross_entropy(lo0.reshape(-1, lo0.shape[-1]), y.reshape(-1))):.4f} -> '
    f'{float(torch.nn.functional.cross_entropy(lo1.reshape(-1, lo1.shape[-1]), y.reshape(-1))):.4f}; '
    f'train loss {float(loss0):.4f} -> {float(loss1):.4f}; gradient |d| / |g| {float(dg / gt):.2e}, cosine {cos:.4f}')
say(f'\nALL {"PASS" if all(ok) else "FAIL"}')
(HERE / f'verify_bf16_switch_{ARM}_output.txt').write_text('\n'.join(lines) + '\n')
