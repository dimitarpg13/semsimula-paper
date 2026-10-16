#!/usr/bin/env python3
"""
SY2 route-specific readouts (protocol §5.22 level 1 (a) and (b); design note §3.4).

    python3 sy_routes.py ARM [n_seq=200]      ARM in F3.1 SR2 G2 PM1 PMX L4F (the L = 4 Fock arm, live V_phi and xi)

(a) V_phi, pair by pair, at every layer step of the trained forward pass. For
    each token t and each of its top-16 selected sources s, the pair's radial force
        a_ts = f_ts . (h_s - h_t) / |h_s - h_t|,   f_ts = -grad_{h_t} V_phi(h_t, h_s),
    positive when the pair attracts (the potential's gradient with the source
    held fixed, as the design note defines it; the trained force also carries
    the straight-through selection term, whose share is reported).
      attraction contrast   mean a_ts over related selected pairs minus
                            distance-matched unrelated selected pairs, divided by
                            the mean |a_ts| over all selected pairs (a force has
                            no cosine units; fixed 2026-10-13 before the scored
                            run, after a 2-sequence mechanics smoke test)
      selection enrichment  related sources selected into the top 16, divided by
                            the number expected if they were selected at the
                            unrelated rate for their distance
(b) The wells (PM1, PMX), at each layer's input and output states. Token t's
    force share over the 64 modes, s_tv proportional to |w_tv| |h_t - mu_v| with
    w = phi E 2 kappa^2 a (the explicit Poisson-mode force; for PMX the same
    formula read at the state, though PMX integrates it exactly), and its
    dominant mode m_t.
      co-assignment contrast  related pairs sharing a dominant mode minus
                              distance-matched unrelated pairs, output minus input
      NMI gain                NMI(m_t, part of speech), output minus input

Groupings and shuffled controls as in sy_measure.py (shared code: sy_groupings.py).
Checks first: the rebuilt V_phi branch reproduces the model's pair potential and
its total V_phi force; the per-pair forces sum to the potential-only part.

Output beside this script: sy_routes_<ARM>.npz, sy_routes_<ARM>_output.txt.
"""
import sys, time
from pathlib import Path

import numpy as np
import torch

sys.argv_in = list(sys.argv)
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
WELLS = getattr(model, 'pm_mu', None) is not None
assert model.cfg.use_gathered_v_phi, 'readout written for the gathered V_phi path'

records = {}
orig_step = model._fock_layer_step
def spy_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
    out = orig_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
    records[layer_idx] = (h.detach().clone(), out[0].detach().clone())
    return out
model._fock_layer_step = spy_step


def vphi_pairs(h, l):
    """Rebuilt gathered V_phi branch at the states h (B, T, d): per-pair radial forces,
    selection, and the check quantities."""
    B, T, d = h.shape
    ht = h.detach().clone().requires_grad_(True)
    hs = h.detach()
    with torch.enable_grad():
        pi = model.score_head(ht, hs)
        causal = model._pair_mask_for(T, h.device)
        idx, m_g = model._sparse_topk_indices(pi, causal, T)
        k = idx.shape[-1]
        h_src_g = hs.gather(1, idx.reshape(B, T * k, 1).expand(-1, -1, d)).view(B, T, k, d)
        Vg = model.V_phi.forward_gathered(ht, h_src_g)                  # (B, T, k)
        s_ell = model.per_layer_scale(l)
        sc = 1.0 if s_ell is None else s_ell
        U = (Vg * m_g).sum() * sc
        f_full, = torch.autograd.grad(U, ht, retain_graph=True)
        fk = []
        hard = m_g.detach()                                              # value: 1 on selected causal sources
        for j in range(k):
            gj, = torch.autograd.grad((Vg[:, :, j] * hard[:, :, j]).sum() * sc, ht, retain_graph=True)
            fk.append(-gj)
    fk = torch.stack(fk, 2).detach()                                     # (B, T, k, d)
    dvec = h_src_g.detach() - h.detach().unsqueeze(2)
    a = (fk * dvec).sum(-1) / dvec.norm(dim=-1).clamp_min(1e-12)         # (B, T, k)
    return dict(U=U.detach(), f_full=-f_full.detach(), f_pot=fk.sum(2), a=a, idx=idx, sel=hard > 0)


def wells_dominant(h, l):
    with torch.no_grad():
        E, phi = model.poisson_mode_occupation(h)
        k2 = model.pm_log_kappa2.float().exp(); a = model.pm_effective_depth(l)
        w = phi * E * (2.0 * k2 * a)                                     # (B, T, K)
        dist = torch.cdist(h.float(), model.pm_mu.float().unsqueeze(0).expand(h.shape[0], -1, -1))
        share = w.abs() * dist
    return share.argmax(-1).numpy()                                      # (B, T)


def nmi(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
    C = np.zeros((len(ua), len(ub))); np.add.at(C, (ia, ib), 1); C /= C.sum()
    pa, pb = C.sum(1), C.sum(0); nz = C > 0
    I = (C[nz] * np.log(C[nz] / np.outer(pa, pb)[nz])).sum()
    Ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum(); Hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return 2 * I / (Ha + Hb) if Ha + Hb > 0 else 0.0


TOK = np.load(CACHE / 'sy_tokens.npz'); X = TOK['x'][:N_SEQ]
GR = Groupings(np.load(CACHE / 'sy_annotation.npz'), X)
T_ = X.shape[1]

# ---------------------------------------------------------------- checks
say(f'SY2 route readouts: {ARM}  ...{folder[-60:]}  (L = {L}; wells: {WELLS})')
with torch.enable_grad():
    model(torch.from_numpy(X[:2]))
for l in range(L):
    h = records[l][0]
    R = vphi_pairs(h, l)
    with torch.enable_grad():
        ha = h.detach().clone().requires_grad_(True)
        Um = model._pair_potential(ha, l, xis=model.xi_module(ha.detach()), h_src_live=None)   # xi is required by the exchange-field arms, unused by V_phi
        fm, = torch.autograd.grad(Um, ha)
    dU = float((R['U'] - Um.detach()).abs()); dF = float((R['f_full'] + fm).abs().max())
    share = float(R['f_pot'].norm() / R['f_full'].norm())
    say(f'CHECK  layer {l}: rebuilt pair potential vs model |dU| {dU:.1e}; total V_phi force max |d| {dF:.1e}; '
        f'potential-only share of the force {share:.3f}')
    assert dU <= 1e-4 * max(1.0, float(Um.abs())) and dF < 1e-4

# ---------------------------------------------------------------- measurement
store = {}
def put(k, v): store.setdefault(k, []).append(v)
pos_out, pos_in, pos_lab = {l: [] for l in range(L)}, {l: [] for l in range(L)}, []
rng = np.random.default_rng(20261011)
t0 = time.time()
for b0 in range(0, N_SEQ, 2):
    xb = torch.from_numpy(X[b0:b0 + 2])
    with torch.enable_grad():
        model(xb)
    per_layer = {}
    for l in range(L):
        h_in, h_out = records[l]
        R = vphi_pairs(h_in, l)
        dom = (wells_dominant(h_in, l), wells_dominant(h_out, l)) if WELLS else None
        per_layer[l] = (R, dom)
    for j in range(xb.shape[0]):
        i = b0 + j
        G, Gs = GR(i, rng)
        for l in range(L):
            R, dom = per_layer[l]
            idx = R['idx'][j].numpy(); sel = R['sel'][j].numpy(); a = R['a'][j].numpy()
            A = np.zeros((T_, T_)); Sel = np.zeros((T_, T_), bool)
            tt = np.repeat(np.arange(T_)[:, None], idx.shape[1], 1)
            m = sel
            A[idx[m], tt[m]] = a[m]; Sel[idx[m], tt[m]] = True             # stored at [source, target], source < target
            cand = np.triu(np.ones((T_, T_), bool), 1)                      # every causal pair is a candidate source
            put(f'L{l}/abs_a', np.array([np.abs(a[m]).sum(), m.sum()]))      # normaliser: mean |a| over selected pairs
            for name in GROUPINGS:
                for tag, (rel, unr) in (('', G[name]), ('shuf/', Gs[name])):
                    put(f'L{l}/{name}/{tag}attr', bin_stats_rel(A, rel & Sel, unr & Sel))
                    put(f'L{l}/{name}/{tag}sel', bin_stats_rel(Sel.astype(float), rel & cand, unr & cand))
            if dom is not None:
                for which, dm in (('in', dom[0][j]), ('out', dom[1][j])):
                    Co = (dm[:, None] == dm[None, :]).astype(float)
                    for name in GROUPINGS:
                        for tag, (rel, unr) in (('', G[name]), ('shuf/', Gs[name])):
                            put(f'L{l}/{name}/{tag}coassign_{which}', bin_stats_rel(Co, rel, unr))
                lab = GR.pos_labels(i)
                keep = lab >= 0
                pos_in[l].append(dom[0][j][keep]); pos_out[l].append(dom[1][j][keep])
                if l == 0: pos_lab.append(lab[keep])
    if (b0 // 2) % 10 == 0:
        print(f'   {b0 + xb.shape[0]}/{N_SEQ} sequences, {(time.time() - t0) / 60:.1f} min', flush=True)
model._fock_layer_step = orig_step
arrs = {k: np.asarray(v) for k, v in store.items()}
if WELLS:
    for l in range(L):
        arrs[f'L{l}/pos_dom_in'] = np.concatenate(pos_in[l]); arrs[f'L{l}/pos_dom_out'] = np.concatenate(pos_out[l])
    arrs['pos_lab'] = np.concatenate(pos_lab)
    arrs['pos_seq_len'] = np.array([len(x) for x in pos_lab])
np.savez_compressed(HERE / f'sy_routes_{ARM}.npz', **arrs)


# ---------------------------------------------------------------- summary
def enrichment(Ssum):
    """related selections / selections expected at the unrelated rate for their distances."""
    ok = Ssum[:, 3] > 0
    exp = (Ssum[ok, 1] * Ssum[ok, 2] / Ssum[ok, 3]).sum()
    return Ssum[ok, 0].sum() / exp if exp > 0 else np.nan

def boot_enrich(lst, seed=0):
    Sx = np.asarray(lst); I = np.random.default_rng(seed).integers(0, len(Sx), (S.N_BOOT, len(Sx)))
    b = np.array([enrichment(Sx[ii].sum(0)) for ii in I])
    return float(enrichment(Sx.sum(0))), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))

say()
say(f'SUMMARY ({N_SEQ} sequences; 95% cluster-bootstrap intervals; floor {S.FLOOR}; enrichment threshold 1.2)')
for l in range(L):
    say(f'layer {l}')
    for name in GROUPINGS:
        nrm = arrs[f'L{l}/abs_a'].sum(0); nrm = nrm[0] / nrm[1]
        at = tuple(v / nrm for v in S.boot_contrast(arrs[f'L{l}/{name}/attr']))
        ats = tuple(v / nrm for v in S.boot_contrast(arrs[f'L{l}/{name}/shuf/attr']))
        en = boot_enrich(arrs[f'L{l}/{name}/sel']); ens = boot_enrich(arrs[f'L{l}/{name}/shuf/sel'])
        s = f'   {name:<6} V_phi attraction (relative) {S.fmt(at)} (shuffled {S.fmt(ats)})   enrichment {en[0]:.3f} [{en[1]:.3f}, {en[2]:.3f}] (shuffled {ens[0]:.3f})'
        say(s)
        if WELLS:
            ca = S.boot_contrast_diff(arrs[f'L{l}/{name}/coassign_out'], arrs[f'L{l}/{name}/coassign_in'])
            cas = S.boot_contrast_diff(arrs[f'L{l}/{name}/shuf/coassign_out'], arrs[f'L{l}/{name}/shuf/coassign_in'])
            say(f'          wells co-assignment gain {S.fmt(ca)} (shuffled {S.fmt(cas)})')
    if WELLS:
        lab = arrs['pos_lab']; o = arrs[f'L{l}/pos_dom_out']; n_in = arrs[f'L{l}/pos_dom_in']
        cut = np.concatenate([[0], np.cumsum(arrs['pos_seq_len'])])
        g_pt = nmi(o, lab) - nmi(n_in, lab)
        I = np.random.default_rng(0).integers(0, len(cut) - 1, (200, len(cut) - 1))
        bs = []
        for ii in I:
            sl = np.concatenate([np.arange(cut[q], cut[q + 1]) for q in ii])
            bs.append(nmi(o[sl], lab[sl]) - nmi(n_in[sl], lab[sl]))
        say(f'   wells NMI(dominant mode, POS): input {nmi(n_in, lab):.4f}  output {nmi(o, lab):.4f}  gain {g_pt:+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] (200 resamples)')
(HERE / f'sy_routes_{ARM}_output.txt').write_text('\n'.join(lines) + '\n')
print(f'wrote sy_routes_{ARM}.npz and _output.txt')
