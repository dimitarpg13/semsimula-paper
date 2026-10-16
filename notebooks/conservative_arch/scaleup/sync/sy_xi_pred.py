#!/usr/bin/env python3
"""
SY follow-up: is the cross-arm pattern binding through the shared context xi,
or the geometry of next-token prediction? (protocol §5.22, "SY on the L = 4
Fock arm ... and a pattern common to every arm"; criteria fixed 2026-10-14
before the first run.)

    python3 sy_xi_pred.py ARM [n_seq=200]      ARM in F3.1 SR2 G2 PM1 PMX L4F

The pattern: in every arm the final states put noun-chunk members closer
(B +0.032 to +0.047) and heads further from their dependents (-0.009 to
-0.018), from identical embedding values; V_phi and the Fock channel do not
carry it, the wells do where they exist.

TEST A, binding through xi. xi (the leaky running averages of earlier states)
is computed inside every step from the step's input; the step is replayed
from its captured input with xi replaced:
  own     each token's xi computed from that token alone (no past): removes every
          other token's influence through xi  -> B_tot_xi = B(on) - B(own)
  common  every token given the sequence-average xi: removes what is
          token-specific about the context  -> B_spec_xi = B(on) - B(common)
  Reading 1 holds if B_spec_xi passes the floor at the LAST layer for noun
  chunks (positive) and for dependencies (negative), with the shuffled control
  null, in PMX and in at least one other arm.

TEST B, prediction geometry (last layer's output, which the readout reads).
  B_target  the contrast on the cosine of the true next tokens' output
            embeddings (centred): does the data's continuation structure have
            the pattern?
  B_pred    the contrast on the Bhattacharyya overlap of the model's predicted
            distributions, sum_v sqrt(p_t,v p_s,v)
  B_resid   the state contrast after removing, per sequence, the linear fit of
            state cosine on prediction overlap (over all pairs)
  Reading 2 holds if B_resid is null for noun chunks in at least five of the
  six arms while B_on passes; it is contradicted if B_resid passes for noun
  chunks in PMX and in at least one other arm.

Checks first: the "on" replay reproduces the trained step bit for bit; each xi
patch moves the step and restores; logits rebuilt from the last step's output
equal the model's.

Output beside this script: sy_xi_pred_<ARM>.npz and _output.txt.
"""
import sys, time
from pathlib import Path

import numpy as np
import torch

ARM = sys.argv[1]
N_SEQ = next((int(a[6:]) for a in sys.argv[2:] if a.startswith('n_seq=')), 200)
HERE = Path(__file__).resolve().parent
CACHE = HERE / 'sy_cache'
sys.argv = [sys.argv[0], str(CACHE / 'harness')]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / 'debug'))   # shared model-building helpers (verify_pm_switch, ...)
import verify_pm_switch as P
import sy_instruments as S
from sy_groupings import GROUPINGS, Groupings, bin_stats_rel

SR2 = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
PMXS = (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),)
_T = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
_E = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
F31 = P.CONFIGS['F3.1'][1]
ARMS = {
    'F3.1': (P.F31_F, F31), 'SR2': (_T + 'sr2' + _E, F31 + SR2), 'G2': (P.G2_F, P.CONFIGS['G2'][1]),
    'PM1': (_T + 'pm64' + _E, F31 + P.PM_ON), 'PMX': (_T + 'pm64_pmx16_sr2' + _E, F31 + P.PM_ON + SR2 + PMXS),
    'L4F': (P._P + 'vplive_xilive_L4probe' + P._S.replace('idt4_', 'idt2_') + 'noattn',
            P.CONFIGS['G2'][1] + (("LADDER_L         = 2 ", "LADDER_L         = 4 "),)),
    'G3': (P._P + 'rglive_vplive_xilive_L2probe' + P._S + 'attnpot',
           P.LIVE + (("LADDER_MECHANISM = 'none'", "LADDER_MECHANISM = 'attention_potential'"),
                     ("RELAX_GRAD_PATH        = 'default'", "RELAX_GRAD_PATH        = 'live'"))),
    'G3p': (P._P + 'rglive_vplive_xilive_rfqk_rfclip0p3_L2probe' + P._S + 'attnpot',
            P.LIVE + (("LADDER_MECHANISM = 'none'", "LADDER_MECHANISM = 'attention_potential'"),
                      ("RELAX_GRAD_PATH        = 'default'", "RELAX_GRAD_PATH        = 'live'"),
                      ("RELAX_ATTN_QK_NORM          = False", "RELAX_ATTN_QK_NORM          = True"),
                      ("RELAX_FIELD_CLIP            = None", "RELAX_FIELD_CLIP            = 0.3"))),
}
lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))

folder, repl = ARMS[ARM]
model, g, _ = P.build(folder, repl)
model.to('cpu'); model.eval()
L = model.cfg.L
xim = model.xi_module
orig_xi_fwd = xim.forward
orig_step = model._fock_layer_step
orig_project = model._project

def xi_own(x):
    B, T, d = x.shape
    out = orig_xi_fwd(x.reshape(B * T, 1, d))
    return out.reshape(B, T, *out.shape[2:])

def xi_common(x):
    out = orig_xi_fwd(x)
    return out.mean(dim=1, keepdim=True).expand_as(out)

class XiPatch:
    def __init__(self, kind): self.kind = kind
    def __enter__(self):
        if self.kind == 'own': xim.forward = xi_own
        elif self.kind == 'common': xim.forward = xi_common
    def __exit__(self, *e):
        xim.__dict__.pop('forward', None)

proj_in = []
def _proj(h):
    proj_in.append(h.detach())
    return orig_project(h)
model._project = _proj

records = {}
def spy_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
    out = orig_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
    cl = lambda a: a.detach().clone() if torch.is_tensor(a) else a
    records[layer_idx] = {'args': [cl(a) for a in (h, h_prev, r, sal, m_b, gamma, dt)], 'out': out[0].detach().clone()}
    return out
model._fock_layer_step = spy_step

def replay(rec, l, kind):
    args = [a.clone() if torch.is_tensor(a) else a for a in rec['args']]
    args[0] = args[0].requires_grad_(True)
    proj_in.clear()
    with XiPatch(kind), torch.enable_grad():
        out = orig_step(*args, layer_idx=l)
    return proj_in[-1].double().numpy(), out[0].detach().double().numpy()

TOK = np.load(CACHE / 'sy_tokens.npz'); X = TOK['x'][:N_SEQ]; Y = TOK['y'][:N_SEQ]
GR = Groupings(np.load(CACHE / 'sy_annotation.npz'), X)
W_out = (model.lm_head.weight if model.lm_head is not None else model.E.weight).detach().double().numpy()

# ---------------------------------------------------------------- checks
say(f'SY xi / prediction tests: {ARM}  ...{folder[-60:]}  (L = {L})')
xb = torch.from_numpy(X[:2])
with torch.enable_grad():
    logits_ref, _ = model(xb)
for l in range(L):
    pre, post = replay(records[l], l, None)
    d0 = float(np.abs(post - records[l]['out'].double().numpy()).max())
    moved = {k: float(np.abs(replay(records[l], l, k)[1] - post).max()) for k in ('own', 'common')}
    say(f'CHECK  layer {l}: replay "on" max |d| {d0:.1e}; xi own moves the output by {moved["own"]:.3e}, common by {moved["common"]:.3e}')
    assert d0 == 0.0 and moved['own'] > 0 and moved['common'] > 0
    assert np.array_equal(replay(records[l], l, None)[1], post)
with torch.no_grad():
    lg = model.compute_logits(records[L - 1]['out'])
dl = float((lg - logits_ref.detach()).abs().max())
say(f'CHECK  logits rebuilt from the last step\'s output vs the model\'s: max |d| {dl:.1e}')
assert dl < 1e-4
with torch.enable_grad():
    logits_after, _ = model(xb)
assert torch.equal(logits_after, logits_ref)
say('CHECK  every patch restored (logits identical after all replays)')

# ---------------------------------------------------------------- measurement
store = {}
def put(k, v): store.setdefault(k, []).append(v)
rng = np.random.default_rng(20261014)
t0 = time.time()
for b0 in range(0, N_SEQ, 2):
    xb = torch.from_numpy(X[b0:b0 + 2])
    with torch.enable_grad():
        model(xb)
    vers = {(l, k): replay(records[l], l, k) for l in range(L) for k in (None, 'own', 'common')}
    with torch.no_grad():
        logp = torch.log_softmax(model.compute_logits(records[L - 1]['out']).float(), -1)
    for j in range(xb.shape[0]):
        i = b0 + j
        G, Gs = GR(i, rng)
        for l in range(L):
            Hin = records[l]['args'][0][j].double().numpy()
            pre_on, post_on = vers[(l, None)][0][j], vers[(l, None)][1][j]
            cos_on = S.cos_matrix(S.centred(post_on))
            for k in ('own', 'common'):
                pre_r, post_r = vers[(l, k)][0][j], vers[(l, k)][1][j]
                cos_r = S.cos_matrix(S.centred(post_r))
                put(f'L{l}/{k}/Gamma', S.coherence(S.contraction_matrix(pre_on, pre_r)))
                for name in GROUPINGS:
                    put(f'L{l}/{name}/{k}/B_ref', bin_stats_rel(cos_r, *G[name]))
                    put(f'L{l}/{name}/{k}/shuf/B_ref', bin_stats_rel(cos_r, *Gs[name]))
            for name in GROUPINGS:
                put(f'L{l}/{name}/B_on', bin_stats_rel(cos_on, *G[name]))
                put(f'L{l}/{name}/shuf/B_on', bin_stats_rel(cos_on, *Gs[name]))
                put(f'L{l}/{name}/B_in', bin_stats_rel(S.cos_matrix(S.centred(Hin)), *G[name]))
        # test B on the last layer's output
        post_L = vers[(L - 1, None)][1][j]
        cosS = S.cos_matrix(S.centred(post_L))
        q = torch.exp(0.5 * logp[j]).double()
        BC = (q @ q.T).numpy()
        tgt = S.cos_matrix(S.centred(W_out[Y[i]]))
        iu = np.triu_indices(len(BC), 1)
        A = np.stack([np.ones(len(iu[0])), BC[iu]], 1)
        coef, *_ = np.linalg.lstsq(A, cosS[iu], rcond=None)
        R = cosS - (coef[0] + coef[1] * BC)
        put('fit', np.array([coef[0], coef[1], np.corrcoef(BC[iu], cosS[iu])[0, 1]]))
        for name in GROUPINGS:
            for tag, rel in (('', G[name]), ('shuf/', Gs[name])):
                put(f'pred/{name}/{tag}B_state', bin_stats_rel(cosS, *rel))
                put(f'pred/{name}/{tag}B_pred', bin_stats_rel(BC, *rel))
                put(f'pred/{name}/{tag}B_resid', bin_stats_rel(R, *rel))
                put(f'pred/{name}/{tag}B_target', bin_stats_rel(tgt, *rel))
    if (b0 // 2) % 10 == 0:
        print(f'   {b0 + xb.shape[0]}/{N_SEQ} sequences, {(time.time() - t0) / 60:.1f} min', flush=True)
model._fock_layer_step = orig_step
arrs = {k: np.asarray(v) for k, v in store.items()}
np.savez_compressed(HERE / f'sy_xi_pred_{ARM}.npz', **arrs)

# ---------------------------------------------------------------- summary
D = lambda a, b: S.boot_contrast_diff(arrs[a], arrs[b])
say()
say(f'TEST A: xi as a coupling ({N_SEQ} sequences; 95% cluster-bootstrap intervals; floor {S.FLOOR})')
for l in range(L):
    say(f'layer {l}: Gamma vs own-xi {S.fmt(S.boot_mean(arrs[f"L{l}/own/Gamma"]))}   vs common-xi {S.fmt(S.boot_mean(arrs[f"L{l}/common/Gamma"]))}')
    for name in GROUPINGS:
        bo = S.boot_contrast(arrs[f'L{l}/{name}/B_on']); bi = S.boot_contrast(arrs[f'L{l}/{name}/B_in'])
        bt = D(f'L{l}/{name}/B_on', f'L{l}/{name}/own/B_ref')
        bs = D(f'L{l}/{name}/B_on', f'L{l}/{name}/common/B_ref')
        bss = D(f'L{l}/{name}/shuf/B_on', f'L{l}/{name}/common/shuf/B_ref')
        say(f'   {name:<6} B_in {bi[0]:+.4f}  B_on {bo[0]:+.4f}   B_tot_xi {S.fmt(bt)}   B_spec_xi {S.fmt(bs)}  (shuffled {S.fmt(bss)})')
say()
f = arrs['fit']
say(f'TEST B: prediction geometry at the last layer\'s output (per-sequence fit of state cosine on prediction overlap: '
    f'median slope {np.median(f[:, 1]):+.3f}, median correlation {np.median(f[:, 2]):+.3f})')
for name in GROUPINGS:
    vals = {k: S.boot_contrast(arrs[f'pred/{name}/{k}']) for k in ('B_state', 'B_pred', 'B_resid', 'B_target')}
    sh = {k: S.boot_contrast(arrs[f'pred/{name}/shuf/{k}']) for k in ('B_resid', 'B_target')}
    say(f'   {name:<6} state {S.fmt(vals["B_state"])}   prediction overlap {S.fmt(vals["B_pred"])}   '
        f'residual {S.fmt(vals["B_resid"])} (shuffled {S.fmt(sh["B_resid"])})   true next tokens {S.fmt(vals["B_target"])} (shuffled {S.fmt(sh["B_target"])})')
(HERE / f'sy_xi_pred_{ARM}_output.txt').write_text('\n'.join(lines) + '\n')
print(f'wrote sy_xi_pred_{ARM}.npz and _output.txt')
