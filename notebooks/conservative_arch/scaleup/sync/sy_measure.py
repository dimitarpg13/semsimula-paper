#!/usr/bin/env python3
"""
SY1 and SY2, state level: does each coupling synchronize tokens, and along
linguistic structure? (protocol §5.22; design note §3-4; instruments from
sy_instruments.py as admitted by SY0 run 3).

    python3 sy_measure.py ARM [n_seq=200]      ARM in F3.1 SR2 G2 PM1 PMX L4F (the L = 4 Fock arm, live V_phi and xi)

For every layer step of the trained model, on the 200 SY evaluation sequences
(sy_cache/sy_tokens.npz) and their annotation (sy_cache/sy_annotation.npz),
the step is replayed from its captured input in these versions:

  on        as trained (must reproduce the forward pass bit for bit)
  off_c     coupling c switched off (the CD switches: V_phi, wells, Fock)
  fcom_c    coupling c's FORCE replaced by its cross-token mean (V_phi, PM1's
            explicit wells, the Fock reverse-channel force); not available for
            PMX's wells, which are integrated inside the exact flow
  dcom_c    coupling c's DISPLACEMENT replaced by its cross-token mean:
            off_c + mean over tokens of (on - off_c), before the closing
            LayerNorm; available for every coupling

SY0 validated the common-mode reference at the force level. The displacement
form is what PMX's wells allow, so it is checked against the force form on
V_phi, a plain kick on every arm, before it is used: the two must agree on
Gamma_spec and on every grouping's B_spec to within a quarter of the floor
(0.005), with the same sign (fixed before the first run).

Measures (per layer, per coupling; Gamma on the pre-LayerNorm state, B on the
state the next layer reads, C on the pre-LayerNorm displacement of the layer):
  Gamma_tot, Gamma_spec        global pairwise contraction, on vs off / on vs common
  B_spec, Gb_spec, C_spec      binding, contraction-binding and co-movement contrasts,
                               on minus common, per grouping
  B_in, B_on                   the binding contrast at the layer's input and output (descriptive)
Groupings: coref (chains; pairs within one mention excluded), dep (head-dependent
word pairs, punctuation excluded), chunk (noun chunks), ent (entity spans), pos
(part of speech, punctuation excluded), repeat (repeated token ids, a mechanical
reference). Pairs of tokens within one word are excluded throughout. Every grouping
has a shuffled control: labels permuted within the sequence (dep: each dependent
paired with a word at another dependent's head offset).

Output beside this script: sy_measure_<ARM>.npz (per-sequence bin statistics and
global scalars; scoring and the level calls are done across arms by sy_score.py)
and sy_measure_<ARM>_output.txt (checks and a per-arm summary).
"""
import json, sys, time
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

SR2 = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
PMXS = (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),)
_T = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
_E = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
F31 = P.CONFIGS['F3.1'][1]
ARMS = {   # name: (folder, Cell 0 replacements, couplings)
    'F3.1': (P.F31_F, F31, ['vphi']),
    'SR2': (_T + 'sr2' + _E, F31 + SR2, ['vphi']),
    'G2': (P.G2_F, P.CONFIGS['G2'][1], ['vphi', 'fock']),
    'PM1': (_T + 'pm64' + _E, F31 + P.PM_ON, ['vphi', 'wells']),
    'PMX': (_T + 'pm64_pmx16_sr2' + _E, F31 + P.PM_ON + SR2 + PMXS, ['vphi', 'wells']),
    'L4F': (P._P + 'vplive_xilive_L4probe' + P._S.replace('idt4_', 'idt2_') + 'noattn',
            P.CONFIGS['G2'][1] + (("LADDER_L         = 2 ", "LADDER_L         = 4 "),), ['vphi', 'fock']),
}
GROUPINGS = ('coref', 'dep', 'chunk', 'ent', 'pos', 'repeat')
AGREE_TOL = S.FLOOR / 4

lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))


# ---------------------------------------------------------------- the model and its switches
folder, repl, COUPLINGS = ARMS[ARM]
model, g, _ = P.build(folder, repl)
model.to('cpu'); model.eval()
L = model.cfg.L
assert ('fock' in COUPLINGS) == (model.reverse_ch is not None), ARM
assert ('wells' in COUPLINGS) == (getattr(model, 'pm_mu', None) is not None), ARM
PMX_WELLS = bool(getattr(model.cfg, 'poisson_wells_exact', False))
FORCE_COMMON = {c for c in COUPLINGS if not (c == 'wells' and PMX_WELLS)}

orig_pair = model._pair_potential
orig_layer_forces = model._layer_forces
orig_pm_force = getattr(model, 'poisson_mode_force', None)
orig_project = model._project
orig_step = model._fock_layer_step

def _zero_pair(h_in, layer_idx, xis=None, h_src_live=None):
    return model._add_relax_potential(h_in.new_zeros(()), xis, h_in, layer_idx)

def _tokmean(F):
    return F.mean(dim=1, keepdim=True).expand_as(F)

class Patch:
    """One version of the step: 'on', ('off', c) or ('fcom', c). Reversible."""
    def __init__(self, kind=None, c=None):
        self.kind, self.c = kind, c
    def __enter__(self):
        self.saved = {}
        k, c = self.kind, self.c
        if k == 'off':
            if c == 'vphi':
                model._pair_potential = _zero_pair
            elif c == 'wells':
                self.saved['w'] = model.pm_depth.detach().clone()
                with torch.no_grad(): model.pm_depth.zero_()
            elif c == 'fock':
                self.saved['f'] = model.reverse_channel_scale.detach().clone()
                with torch.no_grad(): model.reverse_channel_scale.zero_()
        elif k == 'fcom':
            if c == 'vphi':
                def lf(h_in, xis, layer_idx, *, split=False, vtheta_comps=None):
                    ft, fp = orig_layer_forces(h_in, xis, layer_idx, split=True, vtheta_comps=vtheta_comps)
                    model._pair_potential = _zero_pair
                    try:
                        _, fp0 = orig_layer_forces(h_in, xis, layer_idx, split=True, vtheta_comps=vtheta_comps)
                    finally:
                        del model.__dict__['_pair_potential']
                    fp = fp0 + _tokmean(fp - fp0)
                    return (ft, fp) if split else (fp if ft is None else ft + fp)
                model._layer_forces = lf
            elif c == 'wells':
                model.poisson_mode_force = lambda h, layer_idx: _tokmean(orig_pm_force(h, layer_idx))
            elif c == 'fock':
                rc = model.reverse_ch
                self.saved['rc_fwd'] = rc.forward
                orig_fwd = rc.forward
                rc.forward = lambda *a, **kw: _tokmean(orig_fwd(*a, **kw))
        return self
    def __exit__(self, *exc):
        for name in ('_pair_potential', '_layer_forces', 'poisson_mode_force'):
            model.__dict__.pop(name, None)
        with torch.no_grad():
            if 'w' in self.saved: model.pm_depth.copy_(self.saved['w'])
            if 'f' in self.saved: model.reverse_channel_scale.copy_(self.saved['f'])
        if 'rc_fwd' in self.saved:
            del model.reverse_ch.__dict__['forward']
        return False


# ---------------------------------------------------------------- capture and replay
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

def replay(rec, layer_idx, version):
    args = [a.clone() if torch.is_tensor(a) else a for a in rec['args']]
    args[0] = args[0].requires_grad_(True)
    proj_in.clear()
    with Patch(*version), torch.enable_grad():
        out = orig_step(*args, layer_idx=layer_idx)
    return proj_in[-1].double().numpy(), out[0].detach().double().numpy()

def ln_np(H):
    Hn = torch.from_numpy(H).float()
    return orig_project(Hn).double().numpy()


# ---------------------------------------------------------------- groupings
TOK = np.load(CACHE / 'sy_tokens.npz')
ANN = np.load(next((a[4:] for a in sys.argv_in if a.startswith('ann=')), CACHE / 'sy_annotation.npz'))   # ann=PATH: smoke tests only
X = TOK['x'][:N_SEQ]
POS = np.arange(X.shape[1])
EXCL_PUNCT = {list(ANN['pos_tags']).index(t) for t in ('PUNCT', 'SPACE', 'X', 'SYM')}

def rel_gid(gid):
    lab = gid >= 0
    same = (gid[:, None] == gid[None, :])
    return same & lab[:, None] & lab[None, :], (~same) & lab[:, None] & lab[None, :]

def groupings(i, rng):
    """{name: (related, unrelated)} boolean N x N, for the real labels and a shuffled control."""
    word = ANN['word'][i]; posc = ANN['pos'][i]; head = ANN['head'][i]; punct = ANN['punct'][i] == 1
    same_word = (word[:, None] == word[None, :]) & (word[:, None] >= 0)
    content = (word >= 0) & ~punct
    out, shuf = {}, {}
    def put(name, rel, unr, rel_s, unr_s):
        out[name] = (rel & ~same_word, unr & ~same_word)
        shuf[name] = (rel_s & ~same_word, unr_s & ~same_word)
    def permuted(gid):
        g2 = gid.copy(); m = gid >= 0
        g2[m] = rng.permutation(gid[m]); return g2
    # coreference: chains, pairs within one mention excluded
    cr = ANN['coref'][i]; mn = ANN['mention'][i]
    same_m = (mn[:, None] == mn[None, :]) & (mn[:, None] >= 0)
    r, u = rel_gid(cr); rs, us = rel_gid(permuted(cr))
    put('coref', r & ~same_m, u, rs & ~same_m, us)
    # dependencies: head-dependent word pairs among content words
    wid_ok = content
    def dep_rel(head_of_word):
        hw = np.full(word.max() + 2 if word.max() >= 0 else 1, -1)
        for t in range(len(word)):
            if word[t] >= 0: hw[word[t]] = head_of_word.get(word[t], -1)
        wt, ws = word[:, None], word[None, :]
        rel = ((hw[np.maximum(wt, 0)] == ws) | (hw[np.maximum(ws, 0)] == wt)) & (wt >= 0) & (ws >= 0) & (wt != ws)
        both = wid_ok[:, None] & wid_ok[None, :]
        return rel & both, (~rel) & both
    heads = {int(word[t]): int(head[t]) for t in range(len(word)) if word[t] >= 0 and content[t] and head[t] != word[t]}
    r, u = dep_rel(heads)
    ws = list(heads); offs = [heads[w] - w for w in ws]; perm = rng.permutation(len(ws))
    heads_s = {w: w + offs[perm[k]] for k, w in enumerate(ws)}
    rs, us = dep_rel(heads_s)
    put('dep', r, u, rs, us)
    for name, key in (('chunk', 'chunk'), ('ent', 'ent')):
        gid = ANN[key][i]
        r, u = rel_gid(gid); rs, us = rel_gid(permuted(gid))
        put(name, r, u, rs, us)
    gp = np.where(content, posc, -1)
    gp[np.isin(gp, list(EXCL_PUNCT))] = -1
    r, u = rel_gid(gp); rs, us = rel_gid(permuted(gp))
    put('pos', r, u, rs, us)
    ids = X[i]; vals, cnt = np.unique(ids, return_counts=True)
    rep = np.where(np.isin(ids, vals[cnt >= 2]), ids, -1)
    r, u = rel_gid(rep); rs, us = rel_gid(permuted(rep))
    put('repeat', r, u, rs, us)
    return out, shuf

IU = np.triu_indices(len(POS), 1)
BIN = np.searchsorted(S.EDGES, np.abs(POS[IU[0]] - POS[IU[1]]), side='right') - 1

def bin_stats_rel(M, rel, unr):
    r = rel[IU]; u = unr[IU]; b = BIN
    v = M[IU]
    out = np.zeros((len(S.EDGES) - 1, 4))
    np.add.at(out[:, 0], b[r], v[r]); np.add.at(out[:, 1], b[r], 1)
    np.add.at(out[:, 2], b[u], v[u]); np.add.at(out[:, 3], b[u], 1)
    return out


# ---------------------------------------------------------------- checks
say(f'SY1/SY2 state level: {ARM}  ...{folder[-60:]}  (L = {L}, couplings {COUPLINGS}; force-level common for {sorted(FORCE_COMMON)})')
rng0 = np.random.default_rng(0)
gid_test = np.where(rng0.random(512) < 0.6, rng0.integers(0, 7, 512), -1)
Mtest = rng0.normal(size=(512, 512)); Mtest = Mtest + Mtest.T
dev = np.abs(bin_stats_rel(Mtest, *rel_gid(gid_test)) - S.bin_stats(Mtest, gid_test, POS)).max()
say(f'CHECK  pair-relation bin statistics equal the validated gid form: max |d| {dev:.1e}')
assert dev < 1e-9

model._fock_layer_step = spy_step
xb = torch.from_numpy(X[:2])
with torch.enable_grad():
    logits_ref, _ = model(xb)
for l in range(L):
    pre, post = replay(records[l], l, (None, None))
    d_out = float(np.abs(post - records[l]['out'].double().numpy()).max())
    d_ln = float(np.abs(ln_np(pre) - post).max())
    say(f'CHECK  layer {l}: replay "on" vs the forward pass, max |d| {d_out:.1e}; LayerNorm of the captured pre-LN state vs the output {d_ln:.1e}')
    assert d_out == 0.0, 'replay does not reproduce the trained step'
    assert d_ln < 1e-5
    for c in COUPLINGS:
        for kind in ('off', 'fcom'):
            if kind == 'fcom' and c not in FORCE_COMMON:
                continue
            _, p2 = replay(records[l], l, (kind, c))
            say(f'       {kind}_{c}: output moved by {float(np.abs(p2 - post).max()):.3e} (must be > 0)')
            assert np.abs(p2 - post).max() > 0
    pre_b, post_b = replay(records[l], l, (None, None))
    assert np.array_equal(post_b, post), 'patches did not restore'
with torch.enable_grad():
    logits_after, _ = model(xb)
say(f'CHECK  every patch restored: logits after all replays vs before, max |d| {float((logits_after - logits_ref).abs().max()):.1e}')
assert torch.equal(logits_after, logits_ref)


# ---------------------------------------------------------------- measurement
store = {}                      # key -> list over sequences (bin stats or scalars)
def put(key, v):
    store.setdefault(key, []).append(v)

t0 = time.time()
rng = np.random.default_rng(20261011)
for b0 in range(0, N_SEQ, 2):
    xb = torch.from_numpy(X[b0:b0 + 2])
    with torch.enable_grad():
        model(xb)
    vers = {}
    for l in range(L):
        vers[(l, 'on')] = replay(records[l], l, (None, None))
        for c in COUPLINGS:
            vers[(l, 'off', c)] = replay(records[l], l, ('off', c))
            if c in FORCE_COMMON:
                vers[(l, 'fcom', c)] = replay(records[l], l, ('fcom', c))
    for j in range(xb.shape[0]):
        i = b0 + j
        G, Gs = groupings(i, rng)
        for l in range(L):
            Hin = records[l]['args'][0][j].double().numpy()
            pre_on, post_on = vers[(l, 'on')][0][j], vers[(l, 'on')][1][j]
            cos_in = S.cos_matrix(S.centred(Hin)); cos_on = S.cos_matrix(S.centred(post_on))
            C_on = S.cos_matrix(S.centred(pre_on - Hin))
            for name in GROUPINGS:
                put(f'L{l}/{name}/B_in', bin_stats_rel(cos_in, *G[name]))
                put(f'L{l}/{name}/B_on', bin_stats_rel(cos_on, *G[name]))
                put(f'L{l}/{name}/C_on', bin_stats_rel(C_on, *G[name]))
                put(f'L{l}/{name}/shuf/B_on', bin_stats_rel(cos_on, *Gs[name]))
                put(f'L{l}/{name}/shuf/C_on', bin_stats_rel(C_on, *Gs[name]))
            for c in COUPLINGS:
                pre_off = vers[(l, 'off', c)][0][j]
                refs = {'off': pre_off, 'dcom': pre_off + (pre_on - pre_off).mean(0, keepdims=True)}
                if c in FORCE_COMMON:
                    refs['fcom'] = vers[(l, 'fcom', c)][0][j]
                for rname, pre_ref in refs.items():
                    post_ref = ln_np(pre_ref) if rname == 'dcom' else (vers[(l, rname, c)][1][j])
                    Gm = S.contraction_matrix(pre_on, pre_ref)
                    put(f'L{l}/{c}/{rname}/Gamma', S.coherence(Gm))
                    cos_ref = S.cos_matrix(S.centred(post_ref)); C_ref = S.cos_matrix(S.centred(pre_ref - Hin))
                    for name in GROUPINGS:
                        put(f'L{l}/{name}/{c}/{rname}/B_ref', bin_stats_rel(cos_ref, *G[name]))
                        put(f'L{l}/{name}/{c}/{rname}/C_ref', bin_stats_rel(C_ref, *G[name]))
                        put(f'L{l}/{name}/{c}/{rname}/Gb', bin_stats_rel(Gm, *G[name]))
                        put(f'L{l}/{name}/{c}/{rname}/shuf/B_ref', bin_stats_rel(cos_ref, *Gs[name]))
                        put(f'L{l}/{name}/{c}/{rname}/shuf/C_ref', bin_stats_rel(C_ref, *Gs[name]))
                        put(f'L{l}/{name}/{c}/{rname}/shuf/Gb', bin_stats_rel(Gm, *Gs[name]))
    if (b0 // 2) % 10 == 0:
        el = time.time() - t0
        print(f'   {b0 + xb.shape[0]}/{N_SEQ} sequences, {el / 60:.1f} min', flush=True)
model._fock_layer_step = orig_step

arrs = {k: np.asarray(v) for k, v in store.items()}
np.savez_compressed(HERE / f"sy_measure_{ARM}{'' if not any(a.startswith('ann=') for a in sys.argv_in) else '_SMOKE'}.npz", **arrs)


# ---------------------------------------------------------------- per-arm summary (calls are made by sy_score.py)
def diff(k_on, k_ref):
    return S.boot_contrast_diff(arrs[k_on], arrs[k_ref])

say()
say(f'SUMMARY ({N_SEQ} sequences; 95% cluster-bootstrap intervals; floor {S.FLOOR}). spec = on minus common-mode reference.')
for l in range(L):
    say(f'layer {l}')
    for c in COUPLINGS:
        g_tot = S.boot_mean(arrs[f'L{l}/{c}/off/Gamma'])
        g_d = S.boot_mean(arrs[f'L{l}/{c}/dcom/Gamma'])
        s = f'   {c:<5}  Gamma_tot {S.fmt(g_tot)}   Gamma_spec(disp) {S.fmt(g_d)}'
        if c in FORCE_COMMON:
            s += f'   Gamma_spec(force) {S.fmt(S.boot_mean(arrs[f"L{l}/{c}/fcom/Gamma"]))}'
        say(s)
    for name in GROUPINGS:
        b_in = S.boot_contrast(arrs[f'L{l}/{name}/B_in']); b_on = S.boot_contrast(arrs[f'L{l}/{name}/B_on'])
        say(f'   {name:<6}  B_in {S.fmt(b_in)}   B_on {S.fmt(b_on)}')
        for c in COUPLINGS:
            bs = diff(f'L{l}/{name}/B_on', f'L{l}/{name}/{c}/dcom/B_ref')
            bss = diff(f'L{l}/{name}/shuf/B_on', f'L{l}/{name}/{c}/dcom/shuf/B_ref')
            gb = S.boot_contrast(arrs[f'L{l}/{name}/{c}/dcom/Gb'])
            cs = diff(f'L{l}/{name}/C_on', f'L{l}/{name}/{c}/dcom/C_ref')
            say(f'      {c:<5} B_spec {S.fmt(bs)}  (shuffled {S.fmt(bss)})   Gb_spec {S.fmt(gb)}   C_spec {S.fmt(cs)}')

say()
say(f'CHECK  displacement-level vs force-level common reference, on V_phi (adopted for every coupling only if |d| < {AGREE_TOL} with the same sign)')
agree = True
for l in range(L):
    a = S.boot_mean(arrs[f'L{l}/vphi/dcom/Gamma'])[0]; b = S.boot_mean(arrs[f'L{l}/vphi/fcom/Gamma'])[0]
    ok = abs(a - b) < AGREE_TOL and (np.sign(a) == np.sign(b) or max(abs(a), abs(b)) < AGREE_TOL)
    agree &= ok
    say(f'   layer {l} Gamma_spec: disp {a:+.4f}  force {b:+.4f}  {"agree" if ok else "DISAGREE"}')
    for name in GROUPINGS:
        a = diff(f'L{l}/{name}/B_on', f'L{l}/{name}/vphi/dcom/B_ref')[0]
        b = diff(f'L{l}/{name}/B_on', f'L{l}/{name}/vphi/fcom/B_ref')[0]
        ok = abs(a - b) < AGREE_TOL and (np.sign(a) == np.sign(b) or max(abs(a), abs(b)) < AGREE_TOL)
        agree &= ok
        say(f'   layer {l} B_spec {name:<6}: disp {a:+.4f}  force {b:+.4f}  {"agree" if ok else "DISAGREE"}')
say(f'   => displacement-level reference {"ADOPTED" if agree else "NOT ADOPTED: use the force level where it exists, report the rest as descriptive"}')
SUF = '' if not any(a.startswith('ann=') for a in sys.argv_in) else '_SMOKE'
(HERE / f'sy_measure_{ARM}{SUF}_output.txt').write_text('\n'.join(lines) + '\n')
print(f'wrote sy_measure_{ARM}{SUF}.npz and _output.txt')
