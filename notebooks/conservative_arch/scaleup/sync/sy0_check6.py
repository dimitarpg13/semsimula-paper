#!/usr/bin/env python3
"""
SY0 check 6: which common-mode reference reads Gamma_spec correctly when a LayerNorm sits between
the coupling's push and the state Gamma is read on (protocol §5.22, "Which common-mode reference
for Gamma_spec", criteria fixed 2026-10-15 before this ran).

In the Fock arms (G2, L4F, G3, G3') a layer step is: integrator (couplings push here) -> LayerNorm
-> + Fock increment computed on the normalized state -> [Gamma is read here] -> LayerNorm.
The toy step copies that order, as one step:

    h1 = h + 0.2 * f(h) + coupling(h)        f a shared per-token map
    z  = LN(h1)
    r  = z + 0.1 * tanh(z W2)                 a small per-token increment on the normalized state
    Gamma read on r (before the closing LayerNorm)

References, as in sy_measure.py:
    force (fcom):        the coupling's force on every token replaced by its token mean, step re-run
    displacement (dcom): r_off + token mean of (r_on - r_off); for Gamma this equals switch-off

  6a  a pure common push c (|c| = s * RMS spread of the states, s in 0.3, 1, 3). PASSES if the
      force-form Gamma_spec is null at every size; informative if the displacement form passes the
      floor at some size.
  6b  a genuine grouping pull (toward the mean of the token's planted group; its token mean is 0)
      plus a common push of size s in the same coupling. PASSES if the force-form Gamma_spec is within
      0.01 of the pull's Gamma_spec without the push, at every size, and that one passes the floor.
  6c  (descriptive) a common push through a per-token linear response (token t moves c / (1 + k_t),
      k_t ~ U[0.2, 2], as stiffer modes move less), no LayerNorm.
  40 draws, N = 128, d = 16 and 384; the verdict is read at d = 384, as for SY0's checks 3 and 4.

Run 1 (sy0_check6_v1_output.txt) FAILED 6b: the LayerNorm before the readout makes the pull's
effect depend on the shared push, so the parts do not add. Extension, fixed before this run:
  6b'  6b read on h1, before the LayerNorm. PASSES if the force form is within 0.01 of the pull alone
       at every push size.
  For the pull cases a third reference is reported, "specific-only": the force replaced by its
  deviation from the token mean (no shared push), read as Gamma(specific-only vs off).

    python3 sy0_check6.py [n_draws=40]
"""
import json, sys
from pathlib import Path
import numpy as np
from sy_instruments import contraction_matrix, coherence, boot_mean, is_null, passes_floor, fmt, _unit, FLOOR

N_DRAWS = next((int(a[8:]) for a in sys.argv[1:] if a.startswith('n_draws=')), 40)
HERE = Path(__file__).resolve().parent
N, NG, SIZES = 128, 4, (0.3, 1.0, 3.0)
lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))


def LN(H):
    Z = H - H.mean(1, keepdims=True)
    return Z / np.maximum(Z.std(1, keepdims=True), 1e-12)


def gamma(A, Ref):
    return coherence(contraction_matrix(A, Ref))


def draw(rng, d, kind, s):
    H = rng.normal(size=(N, d))
    spread = np.sqrt(((H - H.mean(0)) ** 2).sum(1).mean())
    gid = rng.integers(0, NG, N)
    W = rng.normal(0, 1.5 / np.sqrt(d), (d, d)); W2 = rng.normal(0, 1.5 / np.sqrt(d), (d, d))
    c = s * spread * _unit(rng.normal(size=d))
    k = rng.uniform(0.2, 2.0, N)[:, None]
    G = np.stack([H[gid == g].mean(0) for g in range(NG)])

    def coupling(h):
        if kind == 'push':
            return np.broadcast_to(c, h.shape)
        if kind in ('pull', "pull'"):
            return 0.5 * (G[gid] - h)
        if kind in ('pull+push', "pull+push'"):
            return 0.5 * (G[gid] - h) + c
        if kind == 'linresp':
            return np.broadcast_to(c, h.shape)
        raise ValueError(kind)

    def step(mode):
        F = coupling(H)
        Fm = np.broadcast_to(F.mean(0, keepdims=True), F.shape)
        F = {'off': 0 * F, 'fcom': Fm, 'fdev': F - Fm}.get(mode, F)
        if kind.endswith("'"):                                     # 6b': read before the LayerNorm
            return H + 0.2 * (np.tanh(H @ W.T) - H) + F
        if kind == 'linresp':                                      # 6c: per-token response, no LayerNorm
            return H + 0.2 * (np.tanh(H @ W.T) - H) + c / (1 + k) * (mode != 'off')
        z = LN(H + 0.2 * (np.tanh(H @ W.T) - H) + F)
        return z + 0.1 * np.tanh(z @ W2.T)

    on, off = step('on'), step('off')
    # a uniform push is its own token mean, so for 6a and 6c the fcom replay equals "on" exactly and the
    # force form reads 0 by construction; what these cases test is the displacement form
    fcom = on if kind == 'linresp' else step('fcom')
    dcom = off + (on - off).mean(0, keepdims=True)
    gdev = gamma(step('fdev'), off) if kind.startswith('pull') else np.nan
    return gamma(on, fcom), gamma(on, dcom), gdev


res = {}
for d in (16, 384):
    for kind, sizes in (('push', SIZES), ('pull', (0.0,)), ('pull+push', SIZES), ('linresp', SIZES),
                        ("pull'", (0.0,)), ("pull+push'", SIZES)):
        for s in sizes:
            rng = np.random.default_rng(20261015 + d)
            G = np.array([draw(rng, d, kind, s) for _ in range(N_DRAWS)])
            res[(d, kind, s)] = [boot_mean(G[:, j]) if np.isfinite(G[:, j]).all() else None for j in range(3)]

say(f'SY0 check 6: the common-mode reference with a LayerNorm before the readout ({N_DRAWS} draws, N = {N})')
for d in (16, 384):
    say(f'\nd = {d}')
    say('   case                     Gamma_spec (force)            Gamma_spec (displacement = switch-off)   specific-only vs off')
    for (dd, kind, s), (gf, gd, gv) in res.items():
        if dd == d:
            say(f'   {kind:11s} s = {s:3.1f}      {fmt(gf)}     {fmt(gd)}          {fmt(gv) if gv else ""}')

d = 384
ok_a = all(is_null(res[(d, 'push', s)][0]) for s in SIZES)
inf_a = any(passes_floor(res[(d, 'push', s)][1]) for s in SIZES)
base = res[(d, 'pull', 0.0)][0]
ok_b = passes_floor(base) and all(abs(res[(d, 'pull+push', s)][0][0] - base[0]) <= 0.01 for s in SIZES)
say('\nVERDICT (d = 384)')
say(f'   6a force form null under a pure common push at every size: {"PASS" if ok_a else "FAIL"}'
    f'; displacement form passes the floor at some size (misreads the push): {"yes" if inf_a else "no"}')
say(f'   6b force form recovers the genuine pull (base {base[0]:+.4f}) within 0.01 at every push size: '
    f'{"PASS" if ok_b else "FAIL"}  (' + ', '.join(f's = {s}: {res[(d, "pull+push", s)][0][0]:+.4f}' for s in SIZES) + ')')
PP = "pull+push'"
basep = res[(d, "pull'", 0.0)][0]
ok_bp = passes_floor(basep) and all(abs(res[(d, "pull+push'", s)][0][0] - basep[0]) <= 0.01 for s in SIZES)
say(f"   6b' (read before the LayerNorm) force form recovers the pull (base {basep[0]:+.4f}) within 0.01: "
    f'{"PASS" if ok_bp else "FAIL"}  (' + ', '.join(f's = {s}: {res[(d, PP, s)][0][0]:+.4f}' for s in SIZES) + ')')
say(f'   => check 6 as first fixed {"PASSES" if ok_a and ok_b else "FAILS"}; '
    f"6b' {'PASSES' if ok_bp else 'FAILS'}")
(HERE / 'sy0_check6_output.txt').write_text('\n'.join(lines) + '\n')
(HERE / 'sy0_check6.json').write_text(json.dumps({f'{k[0]}/{k[1]}/{k[2]}': v for k, v in res.items()}
                                                 | {'pass_6a': ok_a, 'pass_6b': ok_b, "pass_6b'": ok_bp, 'informative_6a': inf_a}, indent=1))
