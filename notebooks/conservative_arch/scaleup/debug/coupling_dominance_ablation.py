"""CD: what does each coupling cost when it is removed? (protocol SS5.20, pre-registered 2026-10-11).

6b-9 shows V_phi carrying the non-V_theta part of the step only in F3.1 and
moving it by < 0.02 wherever a competitor exists (the Fock reverse channel,
the PM1 wells, SR2's exact V_theta flow). 6b-9 measures displacement. This
measures reliance, in one unit for every mechanism: the perplexity cost of
removing it at inference from the trained weights,

  c = ln(PPL_removed / PPL_full)   nats, paired over Cell 6b-7's 12 batches.

Switches, all reversible on the same weights:
  V_phi off : _pair_potential returns the constant zero of the
              pair_potential='none' path, so f_phi = 0 (the wells' force and
              the Fock increment are added elsewhere and stay)
  Fock off  : reverse_channel_scale = 0, so tanh(0) = 0 and the increment is 0
  wells off : pm_depth = 0
  exch off  : relax_lambda = 0, the exchange field's gate (added 2026-10-18, for G3, G3' and XRB-F)

Checks first, all exact: the identity patch and every switch restored give
bit-identical logits; the patch is called on every layer step.

Usage: python3 coupling_dominance_ablation.py OUT_DIR [ARM ...]
  ARM from F3.1 SR2 G2 PM1 PMX G3 G3p XRBF (default: the four pre-registered arms;
  PMX added 2026-10-11, after CD was scored, as a descriptive extension; G3, G3p and
  XRBF added 2026-10-18, descriptive, to complete the token-to-token couplings:
  V_phi, the exchange field and the register read, alone and in every combination).
  A subset writes coupling_dominance_ablation_<ARMS>_output.txt.
"""
import json, sys, types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path(sys.argv[1])
SELECT = sys.argv[2:]
sys.argv = [sys.argv[0], str(OUT / 'harness')]
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import verify_pm_switch as P

N_BATCH, BATCH, BLOCK = 12, 4, 512
SR2 = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
PMXS = (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),)
_T = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
_E = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
ATT = (("LADDER_MECHANISM = 'none'", "LADDER_MECHANISM = 'attention_potential'"),
       ("RELAX_GRAD_PATH        = 'default'", "RELAX_GRAD_PATH        = 'live'"))
QK = (("RELAX_ATTN_QK_NORM          = False", "RELAX_ATTN_QK_NORM          = True"),
      ("RELAX_FIELD_CLIP            = None", "RELAX_FIELD_CLIP            = 0.3"))
ALL3 = ['vphi', 'exch', 'fock', 'vphi+exch', 'vphi+fock', 'exch+fock', 'vphi+exch+fock']
ARMS = {   # name: (folder, Cell 0 replacements, removals)
    'F3.1': (P.F31_F, P.CONFIGS['F3.1'][1], ['vphi']),
    'SR2': (_T + 'sr2' + _E, P.CONFIGS['F3.1'][1] + SR2, ['vphi']),
    'G2': (P.G2_F, P.CONFIGS['G2'][1], ['vphi', 'fock', 'vphi+fock']),
    'PM1': (_T + 'pm64' + _E, P.CONFIGS['F3.1'][1] + P.PM_ON, ['vphi', 'wells', 'vphi+wells']),
    'PMX': (_T + 'pm64_pmx16_sr2' + _E, P.CONFIGS['F3.1'][1] + P.PM_ON + SR2 + PMXS, ['vphi', 'wells', 'vphi+wells']),
    # 2026-10-18 (descriptive): the exchange-field arms
    'G3': (P.G3_F, P.LIVE + ATT, ALL3),
    'G3p': (P._P + 'rglive_vplive_xilive_rfqk_rfclip0p3_L2probe' + P._S + 'attnpot', P.LIVE + ATT + QK, ALL3),
    'XRBF': (P._P + 'norc_rglive_vplive_xilive_rfqk_rfclip0p3_L2probe' + P._S + 'attnpot',
             P.CONFIGS['F3.1'][1] + ATT + QK, ['vphi', 'exch', 'vphi+exch']),
}
RUN = SELECT or ['F3.1', 'SR2', 'G2', 'PM1']
assert all(a in ARMS for a in RUN), RUN

lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))


class Switches:
    """Reversible removals on one built model."""

    def __init__(self, model):
        self.m = model
        self.orig_pair = model._pair_potential          # bound method of the class
        self.calls = 0

    def _count(self, *a, **k):
        self.calls += 1
        return self.orig_pair(*a, **k)

    def _zero(self, h_in, layer_idx, xis=None, h_src_live=None):
        self.calls += 1
        return self.m._add_relax_potential(h_in.new_zeros(()), xis, h_in, layer_idx)

    def set(self, removal, identity=False):
        m = self.m
        self.saved = {}
        parts = removal.split('+') if removal else []
        if identity:
            m._pair_potential = self._count
        elif 'vphi' in parts:
            m._pair_potential = self._zero
        if 'fock' in parts:
            self.saved['fock'] = m.reverse_channel_scale.detach().clone()
            with torch.no_grad():
                m.reverse_channel_scale.zero_()
        if 'wells' in parts:
            self.saved['wells'] = m.pm_depth.detach().clone()
            with torch.no_grad():
                m.pm_depth.zero_()
        if 'exch' in parts:
            self.saved['exch'] = m.relax_lambda.detach().clone()
            with torch.no_grad():
                m.relax_lambda.zero_()

    def reset(self):
        m = self.m
        if '_pair_potential' in m.__dict__:
            del m.__dict__['_pair_potential']
        with torch.no_grad():
            if 'fock' in self.saved:
                m.reverse_channel_scale.copy_(self.saved['fock'])
            if 'wells' in self.saved:
                m.pm_depth.copy_(self.saved['wells'])
            if 'exch' in self.saved:
                m.relax_lambda.copy_(self.saved['exch'])
        self.saved = {}


def logits(model, x):
    model.eval()
    with torch.enable_grad():
        out, _ = model(x)
    return out.detach()


def nll(model, batches):
    """Mean next-token loss per batch."""
    return np.array([float(F.cross_entropy(logits(model, torch.from_numpy(xb.astype(np.int64))).flatten(0, 1),
                                           torch.from_numpy(yb.astype(np.int64)).flatten()))
                     for xb, yb in batches])


results = {}
ok_all = True
for name in RUN:
    folder, repl, removals = ARMS[name]
    model, g, _ = P.build(folder, repl)
    model.to('cpu'); model.eval()
    sw = Switches(model)
    L = model.cfg.L
    assert ('fock' in ' '.join(removals)) == (model.reverse_ch is not None), name
    assert ('wells' in ' '.join(removals)) == (getattr(model, 'pm_mu', None) is not None), name
    assert ('exch' in ' '.join(removals)) == (getattr(model, 'relax_field', None) is not None), name
    assert bool(getattr(model.cfg, 'lowrank_damped_flow', False)) == (name in ('SR2', 'PMX')), name
    assert bool(getattr(model.cfg, 'poisson_wells_exact', False)) == (name == 'PMX'), name

    rng = np.random.default_rng(20260920)                       # Cell 6b-7's batches
    batches = [g['get_batch'](g['val_ids'], BATCH, BLOCK, rng) for _ in range(12)][:N_BATCH]

    # ---- checks, on a short slice ----
    x = torch.from_numpy(batches[0][0][:2, :256].astype(np.int64))
    ref = logits(model, x)
    sw.set('', identity=True); sw.calls = 0
    idl = logits(model, x); n_calls = sw.calls; sw.reset()
    chk = [(idl - ref).abs().max().item() == 0.0 and n_calls >= L]
    say(f'{name}  ...{folder[-90:]}')
    say(f'  check: identity patch bit-identical, called {n_calls}x for L = {L} -> {chk[-1]}')
    for rem in removals:
        sw.set(rem); sw.calls = 0
        ab = logits(model, x); n_ab = sw.calls; sw.reset()
        back = logits(model, x)
        moved = (ab - ref).abs().max().item()
        chk.append((back - ref).abs().max().item() == 0.0 and moved > 0 and ('vphi' not in rem or n_ab >= L))
        say(f'  check: {rem:11s} moves logits by max {moved:.2e}; restored bit-identical -> {chk[-1]}')
    ok_all &= all(chk)

    # ---- the measurement ----
    base = nll(model, batches)
    r = {'ppl_full': float(np.exp(base.mean())), 'checks': all(chk)}
    say(f'  full model: PPL {r["ppl_full"]:.2f} on {N_BATCH} x {BATCH} x {BLOCK} tokens')
    for rem in removals:
        sw.set(rem)
        try:
            ab = nll(model, batches)
        finally:
            sw.reset()
        d = ab - base
        c, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        r[rem] = {'cost_nats': c, 'se': se, 'ppl': float(np.exp(ab.mean())),
                  'per_batch_min': float(d.min()), 'per_batch_max': float(d.max())}
        say(f'  remove {rem:11s}: PPL {r[rem]["ppl"]:8.2f}   c = {c:+.4f} +- {se:.4f} nats '
            f'({100 * np.expm1(c):+.1f}% PPL; batches {d.min():+.4f} .. {d.max():+.4f})')
    results[name] = r
    say()
    del model

say(f'-> checks {"ALL PASS" if ok_all else "FAILURES ABOVE"}')
stem = 'coupling_dominance_ablation' + ('' if not SELECT else '_' + '_'.join(RUN).replace('.', ''))
(HERE / f'{stem}_output.txt').write_text('\n'.join(lines) + '\n')
(HERE / f'{stem}.json').write_text(json.dumps(results, indent=1))
