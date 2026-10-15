#!/usr/bin/env python3
"""
SY scoring across arms: levels 0 and 1 and the calls that rest on them
(protocol §5.22 with the SY0 amendment and the fixes recorded beside it).

    python3 sy_score.py

Reads sy_measure_<ARM>.npz and sy_routes_<ARM>.npz for the arms present.

Rules (fixed before any scored data, 2026-10-13):
  passes   95% interval excludes 0 and the point estimate is at least the floor (0.02), positive
  null     interval includes 0, or |point| below the floor
  level 0  Gamma_spec passes for some coupling at some layer, in PMX and in another arm with the same
           route (V_phi: F3.1 or PM1; wells: PM1; Fock: the L = 4 Fock arm, not yet available)
  level 1  for a linguistic grouping (coref, dep, chunk, ent, pos; 'repeat' is a mechanical reference
           and never decides), at one layer:
             V_phi route: (a) relative attraction contrast passes, enrichment >= 1.2, shuffled attraction null
             wells route: (b) co-assignment gain passes with the shuffled gain null, or (pos only) the NMI gain
                          passes the floor with its interval excluding 0
             and (c) B_spec of the same coupling and grouping passes with the shuffled B_spec null;
           replicated: the same route and grouping in PMX and in another arm with that route
  common reference: the displacement form where sy_measure adopted it for the arm, else the force form
  SY-pair   level 1 (a) alone, for coref or dep, replicated
  SY-wells  the wells route passes level 1 (b) (replicated) and the V_phi route does not
"""
import re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / 'debug'))   # shared model-building helpers (verify_pm_switch, ...)
import sy_instruments as S
from sy_routes_stats import enrichment_ci, nmi_gain_ci

ARMS = [a for a in ('PMX', 'PM1', 'F3.1', 'G2', 'SR2') if (HERE / f'sy_measure_{a}.npz').exists()]
LING = ('coref', 'dep', 'chunk', 'ent', 'pos')
ROUTE_ARMS = {'vphi': ('PMX', 'F3.1', 'PM1'), 'wells': ('PMX', 'PM1'), 'fock': ()}
lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))

def passes(ci):
    return ci[1] > 0 and ci[0] >= S.FLOOR

meas = {a: dict(np.load(HERE / f'sy_measure_{a}.npz')) for a in ARMS}
routes = {a: dict(np.load(HERE / f'sy_routes_{a}.npz')) for a in ARMS if (HERE / f'sy_routes_{a}.npz').exists()}

def adopted(a):
    txt = (HERE / f'sy_measure_{a}_output.txt').read_text()
    return 'displacement-level reference ADOPTED' in txt

def couplings(a):
    return sorted({k.split('/')[1] for k in meas[a] if re.match(r'L\d/(vphi|wells|fock)/', k)})

def ref(a, c):
    return 'dcom' if (adopted(a) or f'L0/{c}/fcom/Gamma' not in meas[a]) else 'fcom'

def layers(a):
    return sorted({int(k[1]) for k in meas[a] if k.startswith('L')})

say(f'SY scoring: arms present {ARMS}; route readouts for {sorted(routes)}')
say()

# ---------------------------------------------------------------- level 0
say('LEVEL 0  Gamma_spec (global contraction attributable to the coupling\'s token-specific action)')
L0 = {}
for a in ARMS:
    for c in couplings(a):
        for l in layers(a):
            ci = S.boot_mean(meas[a][f'L{l}/{c}/{ref(a, c)}/Gamma'])
            ok = passes(ci)
            L0.setdefault((c, a), []).append(ok)
            say(f'   {a:<5} {c:<5} layer {l}  Gamma_spec {S.fmt(ci)}  ({ref(a, c)})  {"PASSES" if ok else "-"}')
l0_routes = {}
for c, others in ROUTE_ARMS.items():
    in_pmx = any(L0.get((c, 'PMX'), [False]))
    rep = [a for a in others if a != 'PMX' and any(L0.get((c, a), [False]))]
    l0_routes[c] = in_pmx and bool(rep)
    say(f'   route {c:<5}: PMX {"yes" if in_pmx else "no"}; replicated in {rep or "none"}  =>  {"HOLDS" if l0_routes[c] else "does not hold"}')
level0 = any(l0_routes.values())
say(f'   => level 0 {"HOLDS" if level0 else "FAILS"}')
say()

# ---------------------------------------------------------------- level 1
say('LEVEL 1  per grouping and layer: route readout (a)/(b), and (c) B_spec, each against its shuffled control')
hits = {}            # (route, grouping) -> set of arms where it holds at some layer
a_hits = {}          # V_phi (a) alone
b_hits = {}          # wells (b) alone
for a in ARMS:
    R = routes.get(a)
    for l in layers(a):
        for name in LING:
            for c in couplings(a):
                if c == 'fock':
                    continue
                k_on = f'L{l}/{name}/B_on'; k_ref = f'L{l}/{name}/{c}/{ref(a, c)}/B_ref'
                bs = S.boot_contrast_diff(meas[a][k_on], meas[a][k_ref])
                bss = S.boot_contrast_diff(meas[a][f'L{l}/{name}/shuf/B_on'], meas[a][f'L{l}/{name}/{c}/{ref(a, c)}/shuf/B_ref'])
                c_ok = passes(bs) and S.is_null(bss)
                route_ok, desc = False, ''
                if R is not None and c == 'vphi':
                    nrm = R[f'L{l}/abs_a'].sum(0); nrm = nrm[0] / nrm[1]
                    at = tuple(v / nrm for v in S.boot_contrast(R[f'L{l}/{name}/attr']))
                    ats = tuple(v / nrm for v in S.boot_contrast(R[f'L{l}/{name}/shuf/attr']))
                    en = enrichment_ci(R[f'L{l}/{name}/sel'])
                    route_ok = passes(at) and en[0] >= 1.2 and S.is_null(ats)
                    desc = f'attraction {S.fmt(at)} enrichment {en[0]:.2f}'
                    if route_ok: a_hits.setdefault(name, set()).add(a)
                elif R is not None and c == 'wells':
                    ca = S.boot_contrast_diff(R[f'L{l}/{name}/coassign_out'], R[f'L{l}/{name}/coassign_in'])
                    cas = S.boot_contrast_diff(R[f'L{l}/{name}/shuf/coassign_out'], R[f'L{l}/{name}/shuf/coassign_in'])
                    route_ok = passes(ca) and S.is_null(cas)
                    desc = f'co-assignment gain {S.fmt(ca)}'
                    if name == 'pos':
                        ng = nmi_gain_ci(R, l)
                        desc += f'  NMI gain {S.fmt(ng)}'
                        route_ok = route_ok or passes(ng)
                    if route_ok: b_hits.setdefault(name, set()).add(a)
                ok = route_ok and c_ok
                if ok:
                    hits.setdefault((c, name), set()).add(a)
                say(f'   {a:<5} L{l} {name:<6} {c:<5}  {desc}  | B_spec {S.fmt(bs)} (shuffled {S.fmt(bss)})  {"HOLDS" if ok else "-"}')
def replicated(arms_set, route):
    return 'PMX' in arms_set and any(x in arms_set for x in ROUTE_ARMS[route] if x != 'PMX')
level1 = [(c, n) for (c, n), s in hits.items() if replicated(s, c)]
say(f'   => level 1 {"HOLDS" if level1 else "FAILS"}' + (f'  (route, grouping): {level1}' if level1 else ''))
sy_pair = any(replicated(a_hits.get(n, set()), 'vphi') for n in ('coref', 'dep'))
wells_b = any(replicated(s, 'wells') for s in b_hits.values())
vphi_l1 = any(c == 'vphi' for c, _ in level1)
say()
say('CALLS (protocol §5.22)')
say(f'   SY-L0     level 0 holds (75%)                                   -> {"HIT" if level0 else "MISS"}')
say(f'   SY-L1     level 1 holds (45%)                                   -> {"HIT" if level1 else "MISS"}')
say(f'   SY-pair   V_phi attracts coreferent or head-dependent pairs (50%) -> {"HIT" if sy_pair else "MISS"}')
say(f'   SY-wells  wells pass 1 (b) and V_phi does not (25%)             -> {"HIT" if (wells_b and not vphi_l1) else "MISS"}')
say('   SY-L2, SY-L3, SY-G2, SY-GPT2: need SY3-SY6 (not yet run)')
(HERE / 'sy_score_output.txt').write_text('\n'.join(lines) + '\n')
print('wrote sy_score_output.txt')
