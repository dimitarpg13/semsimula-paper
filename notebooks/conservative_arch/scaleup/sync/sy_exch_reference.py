#!/usr/bin/env python3
"""
Which common-mode reference for Gamma_spec: the mechanism replay (protocol §5.22, "Which common-mode
reference for Gamma_spec", criteria fixed 2026-10-15 before this ran; extended after SY0 check 6
failed, also before this ran).

In the Fock arms a layer step is: integrator (V_phi and the exchange field push here) -> LayerNorm
-> + Fock increment -> [sy_measure reads Gamma here] -> LayerNorm. This script reads Gamma at two points:
    A  the integrator's output, before any normalization (the first LayerNorm's input);
    B  sy_measure's readout (the last LayerNorm's input).
and replays each layer with the coupling c under these versions:
    on      as trained
    off     switched off
    fcom    its force replaced by its token mean (the shared push only)
    fdev    its force replaced by its deviation from the token mean (the specific part only, no push)
    exch only: fcom with the first LayerNorm's per-token mean and scale frozen at the "off" replay's
            values (frozen); fcom with the mean force scaled by 0.5 and 2 (linearity)

    Gamma_tot  = Gamma(on vs off)          (= the displacement-form Gamma_spec)
    Gamma_spec = Gamma(on vs fcom)         (the force form)
    Gamma_push = Gamma(fcom vs off)        so that Gamma_tot = Gamma_push + Gamma_spec exactly
    Gamma_dev  = Gamma(fdev vs off)        the specific part acting without the shared push

Decision for the exchange route's level 0 (fixed before this ran): at A, if Gamma_spec and Gamma_dev
agree within 0.01 in both arms, their mean decides by the floor; otherwise level 0 stays undecided.
M1 (check): B reproduces sy_measure's per-sequence values. M2: |Gamma_push| at A below the floor.
M3: Gamma_push at B with the first LayerNorm frozen below the floor. V_phi is read the same way in
every Fock arm (G2, L4F, G3, G3p), to show whether its earlier scores depended on the readout point.

    python3 sy_exch_reference.py ARM [n_seq=50]

Arms without a reverse channel (XRBF, added 2026-10-18) have one LayerNorm per step: integrator -> LayerNorm.
There sy_measure's readout already is the integrator's output, so stages A and B coincide, and the
frozen-LayerNorm replay at B reduces to the push at A (M3 = M2).
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ARM = sys.argv[1]
N_SEQ = next((int(a[6:]) for a in sys.argv[2:] if a.startswith('n_seq=')), 50)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sy_replay import Replayer, CACHE, _tokmean
import sy_instruments as S

lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
R = Replayer(ARM)
m = R.model
COUP = [c for c in ('vphi', 'exch') if c in R.couplings]
ONE_LN = m.reverse_ch is None
X = np.load(CACHE / 'sy_tokens.npz')['x'][:N_SEQ]
REF = np.load(HERE / f'sy_measure_{ARM}.npz')


def _tokdev(Fv):
    return Fv - _tokmean(Fv)


def patch_extra(version):
    """versions the Replayer lacks: ('fdev', c), ('fscale', 'exch', a). Returns an undo callable."""
    k, c = version[0], version[1]
    if c == 'exch':
        rf = m.relax_field; fl0 = rf.force_live
        if k == 'fdev':
            rf.force_live = lambda *a, **kw: _tokdev(fl0(*a, **kw))
        elif k == 'fscale':
            al = version[2]
            rf.force_live = lambda *a, **kw: al * _tokmean(fl0(*a, **kw))
        return lambda: rf.__dict__.pop('force_live')
    if c == 'vphi' and k == 'fdev':
        lf0, zp = R._lf, R._zero_pair
        def lf(h_in, xis, layer_idx, *, split=False, vtheta_comps=None):
            ft, fp = lf0(h_in, xis, layer_idx, split=True, vtheta_comps=vtheta_comps)
            m._pair_potential = zp
            try:
                _, fp0 = lf0(h_in, xis, layer_idx, split=True, vtheta_comps=vtheta_comps)
            finally:
                del m.__dict__['_pair_potential']
            fp = fp0 + _tokdev(fp - fp0)
            return (ft, fp) if split else (fp if ft is None else ft + fp)
        m._layer_forces = lf
        return lambda: m.__dict__.pop('_layer_forces')
    raise ValueError(version)


def run(l, version, frozen=None):
    """(stage A, stage B) states of layer l's step, float64 numpy, (B, T, d)."""
    rec = R.records[l]
    args = [a.clone() if torch.is_tensor(a) else a for a in rec['args']]
    args[0] = args[0].requires_grad_(True)
    R.proj_in.clear()
    undo = None; saved = {}
    if version is not None and version[0] in ('fdev', 'fscale'):
        undo = patch_extra(version)
    else:
        saved = R._patch(version)
    if frozen is not None:                     # first LayerNorm with per-token statistics frozen at "off"
        mu, sd = frozen; calls = [0]; proj0 = R._spy_proj
        def proj(h):
            calls[0] += 1
            if calls[0] == 1:
                R.proj_in.append(h.detach())
                return (h - mu) / sd
            return proj0(h)
        m._project = proj
    try:
        with torch.enable_grad():
            R._step(*args, layer_idx=l)
    finally:
        if undo: undo()
        R._unpatch(saved)
        if frozen is not None:
            m._project = R._spy_proj
    if ONE_LN:                                 # no reverse channel: one LayerNorm, so B is A
        assert len(R.proj_in) == 1, f'expected one LayerNorm per step (no reverse channel), got {len(R.proj_in)}'
        a = R.proj_in[0].double().numpy()
        return a, a
    assert len(R.proj_in) == 2, f'expected two LayerNorms per step (Fock arm), got {len(R.proj_in)}'
    return R.proj_in[0].double().numpy(), R.proj_in[1].double().numpy()


def G(a, b):
    return S.coherence(S.contraction_matrix(a, b))


say(f'SY reference test: {ARM} (L = {R.L}), couplings {COUP}, {N_SEQ} sequences')
store = {}
put = lambda k, v: store.setdefault(k, []).append(v)
t0 = time.time(); maxd_on = 0.0
for b0 in range(0, N_SEQ, 2):
    xb = torch.from_numpy(X[b0:b0 + 2]).long()
    R.forward(xb)
    for l in range(R.L):
        on = run(l, None)
        rebuilt = F.layer_norm(torch.from_numpy(on[1]).float(), (on[1].shape[-1],), eps=m.cfg.ln_eps)
        maxd_on = max(maxd_on, float((rebuilt - R.records[l]['out']).abs().max()))
        for c in COUP:
            V = {'off': run(l, ('off', c)), 'fcom': run(l, ('fcom', c)), 'fdev': run(l, ('fdev', c))}
            if c == 'exch':
                A_off = torch.from_numpy(V['off'][0]).float()
                mu = A_off.mean(-1, keepdim=True); sd = torch.sqrt(A_off.var(-1, unbiased=False, keepdim=True) + m.cfg.ln_eps)
                V['frozen_off'] = run(l, ('off', c), frozen=(mu, sd))
                V['frozen_fcom'] = run(l, ('fcom', c), frozen=(mu, sd))
                V['f0.5'] = run(l, ('fscale', c, 0.5)); V['f2'] = run(l, ('fscale', c, 2.0))
            for j in range(xb.shape[0]):
                for st, s in (('A', 0), ('B', 1)):
                    o, f_, fc, dv = on[s][j], V['off'][s][j], V['fcom'][s][j], V['fdev'][s][j]
                    put(f'L{l}/{c}/{st}/tot', G(o, f_)); put(f'L{l}/{c}/{st}/spec', G(o, fc))
                    put(f'L{l}/{c}/{st}/push', G(fc, f_)); put(f'L{l}/{c}/{st}/dev', G(dv, f_))
                if c == 'exch':
                    put(f'L{l}/exch/B/push_frozen', G(V['frozen_fcom'][1][j], V['frozen_off'][1][j]))
                    put(f'L{l}/exch/check_frozen_off', float(np.abs(V['frozen_off'][1][j] - V['off'][1][j]).max()))
                    ln1 = lambda A: F.layer_norm(torch.from_numpy(A).float(), (A.shape[-1],), eps=m.cfg.ln_eps).double().numpy()
                    put(f'L{l}/exch/LN1/push', G(ln1(V['fcom'][0][j]), ln1(V['off'][0][j])))
                    for al, key in ((0.5, 'f0.5'), (2.0, 'f2')):
                        put(f'L{l}/exch/B/push_x{al}', G(V[key][1][j], V['off'][1][j]))
                        dA = V[key][0][j] - V['off'][0][j]; d1 = V['fcom'][0][j] - V['off'][0][j]
                        put(f'L{l}/exch/A/lin_x{al}', float(np.linalg.norm(dA - al * d1) / max(np.linalg.norm(al * d1), 1e-30)))
                    d1 = V['fcom'][0][j] - V['off'][0][j]; offA = V['off'][0][j]
                    put(f'L{l}/exch/A/push_size', float(np.linalg.norm(d1.mean(0)) / np.sqrt(((offA - offA.mean(0)) ** 2).sum(1).mean())))
                    put(f'L{l}/exch/A/push_tokspec', float(np.linalg.norm(d1 - d1.mean(0)) / (np.sqrt(len(d1)) * np.linalg.norm(d1.mean(0)))))
    if b0 == 0:
        say(f'   first batch {time.time() - t0:.0f} s')
say(f'   {N_SEQ} sequences, {(time.time() - t0) / 60:.1f} min')

# ---------------------------------------------------------------- checks
say(f'CHECK     "on" replay: last LayerNorm of stage B vs the recorded step output, max |d| {maxd_on:.1e}')
for l in range(R.L):
    for c in COUP:
        for rname, key in (('off', 'tot'), ('fcom', 'spec')):
            ref = REF[f'L{l}/{c}/{rname}/Gamma'][:N_SEQ]
            d = float(np.abs(np.array(store[f'L{l}/{c}/B/{key}']) - ref).max())
            say(f'CHECK M1  layer {l} {c} {rname}: stage-B Gamma vs sy_measure per sequence, max |d| {d:.1e}')
        idt = float(np.abs(np.array(store[f'L{l}/{c}/B/tot']) - np.array(store[f'L{l}/{c}/B/push']) - np.array(store[f'L{l}/{c}/B/spec'])).max())
        say(f'CHECK M1  layer {l} {c}: Gamma_tot - Gamma_push - Gamma_spec, max |d| {idt:.1e}')
    if 'exch' in COUP:
        say(f'CHECK     layer {l}: frozen-LayerNorm "off" replay vs the plain "off" replay, max |d| {max(store[f"L{l}/exch/check_frozen_off"]):.1e}')

# ---------------------------------------------------------------- summary
out = {}
bm = lambda k: S.boot_mean(store[k])
say(f'\nSUMMARY ({N_SEQ} sequences; 95% cluster-bootstrap intervals; floor 0.02)')
say('   A = integrator output, before any normalization;  B = sy_measure readout ' +
    ('(no reverse channel: the same point as A)' if ONE_LN else '(after the first LayerNorm + Fock increment)'))
for l in range(R.L):
    say(f'layer {l}')
    for c in COUP:
        for st in ('A', 'B'):
            v = {k: bm(f'L{l}/{c}/{st}/{k}') for k in ('tot', 'spec', 'push', 'dev')}
            out.update({f'L{l}/{c}/{st}/{k}': x for k, x in v.items()})
            say(f'   {c:5s} {st}: tot (= disp) {S.fmt(v["tot"])}  spec (force) {S.fmt(v["spec"])}  '
                f'push {S.fmt(v["push"])}  dev (specific only) {S.fmt(v["dev"])}')
    if 'exch' in COUP:
        for k in ('B/push_frozen', 'LN1/push', 'B/push_x0.5', 'B/push_x2.0'):
            out[f'L{l}/exch/{k}'] = bm(f'L{l}/exch/{k}')
        say(f'   exch  push at B with the first LayerNorm frozen {S.fmt(out[f"L{l}/exch/B/push_frozen"])};  '
            f'push after the first LayerNorm, no Fock increment {S.fmt(out[f"L{l}/exch/LN1/push"])}')
        say(f'   exch  push at B scaled x0.5 {S.fmt(out[f"L{l}/exch/B/push_x0.5"])}, x2 {S.fmt(out[f"L{l}/exch/B/push_x2.0"])};  '
            f'linearity at A: rel. error x0.5 {np.median(store[f"L{l}/exch/A/lin_x0.5"]):.1e}, x2 {np.median(store[f"L{l}/exch/A/lin_x2.0"]):.1e}')
        say(f'   exch  shared displacement at A / RMS spread of the states: median {np.median(store[f"L{l}/exch/A/push_size"]):.3f};  '
            f'token-specific part of the push displacement at A, relative: median {np.median(store[f"L{l}/exch/A/push_tokspec"]):.3f}')

# ---------------------------------------------------------------- readings (decided in the protocol, both arms together)
if 'exch' in COUP:
    say('\nREADINGS for this arm (the decision needs both G3 and G3p)')
    for l in range(R.L):
        a, b = out[f'L{l}/exch/A/spec'], out[f'L{l}/exch/A/dev']
        agree = abs(a[0] - b[0]) <= 0.01
        say(f'   layer {l}: at A, spec {a[0]:+.4f} and dev {b[0]:+.4f} {"agree" if agree else "DISAGREE"} (within 0.01); '
            f'mean {0.5 * (a[0] + b[0]):+.4f}')
        say(f'            M2 |push at A| < floor: {"holds" if abs(out[f"L{l}/exch/A/push"][0]) < S.FLOOR else "fails"} '
            f'({out[f"L{l}/exch/A/push"][0]:+.4f});  M3 push at B, LayerNorm frozen, < floor: '
            f'{"holds" if abs(out[f"L{l}/exch/B/push_frozen"][0]) < S.FLOOR else "fails"} ({out[f"L{l}/exch/B/push_frozen"][0]:+.4f})')
(HERE / f'sy_exch_reference_{ARM}_output.txt').write_text('\n'.join(lines) + '\n')
(HERE / f'sy_exch_reference_{ARM}.json').write_text(json.dumps(out, indent=1))
