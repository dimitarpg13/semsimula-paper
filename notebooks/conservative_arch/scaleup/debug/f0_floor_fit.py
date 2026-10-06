"""F0: is there a shared floor near 50 PPL? Stable-phase extrapolation (protocol SS5.16).

Pre-registered 2026-10-06 before any fit. Fits L(t) = L_inf + A t^(-alpha) in
validation loss (nats) to each model's evals in the constant-learning-rate
stable phase, steps 3,000-21,000 inclusive (the WSD decay starts at 21,125),
with a 90% interval on L_inf from 2,000 residual-bootstrap refits. A fit is
unidentified if its L_inf interval is wider than 0.3 nats or alpha hits a bound.

Models: F3.1, G2, G3, G3' and L=4 Fock (identical WSD schedule, LR, batches,
tokens). The matched GPT-2 used a cosine schedule with no constant-LR phase; it
is fitted on the same window for description only and is NOT compared.

Usage: python3 f0_floor_fit.py
"""
import json, math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

DL = Path.home() / 'Downloads'
P = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_'
S = '_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_'
MODELS = {
    'F3.1 (L=2, conservative-only)': P + 'norc_vplive_xilive_L2probe' + S + 'idt4_lr0p0012_noattn',
    'G2 (L=2, registers)': P + 'vplive_xilive_L2probe' + S + 'idt4_lr0p0012_noattn',
    'G3 (L=2, + field)': P + 'rglive_vplive_xilive_L2probe' + S + 'idt4_lr0p0012_attnpot',
    "G3' (L=2, + hardened field)": P + 'rglive_vplive_xilive_rfqk_rfclip0p3_L2probe' + S + 'idt4_lr0p0012_attnpot',
    'L=4 Fock': P + 'vplive_xilive_L4probe' + S + 'idt2_lr0p0012_noattn',
}
GPT2 = DL / 'semsimula_matched_gpt2_baseline_owt' / 'results' / 'training_log.jsonl'
LO, HI, NB, SEED = 3000, 21000, 2000, 20261006


def evals_fock(folder):
    rows = [json.loads(l) for l in open(DL / folder / 'results' / 'training_log.jsonl')]
    ev = {}
    for r in rows:
        if 'val_loss' in r and r.get('val_loss') is not None:
            ev[int(r['step'])] = float(r['val_loss'])          # last write wins (resumes)
    return ev


def evals_gpt2():
    ev = {}
    for l in open(GPT2):
        r = json.loads(l)
        if r.get('val_loss_512') is not None:
            ev[int(r['step'])] = float(r['val_loss_512'])
    return ev


def law(t, linf, a, alpha):
    return linf + a * (t / 1000.0) ** (-alpha)


def fit(t, y):
    p0 = (y.min() - 0.3, (y.max() - y.min()) * 3, 0.5)
    bounds = ((0.0, 0.0, 0.01), (y.min(), 1e3, 5.0))
    p, _ = curve_fit(law, t, y, p0=p0, bounds=bounds, maxfev=20000)
    return p


def analyse(name, ev, rng):
    t = np.array(sorted(s for s in ev if LO <= s <= HI), dtype=float)
    y = np.array([ev[int(s)] for s in t])
    p = fit(t, y)
    resid = y - law(t, *p)
    boots = []
    for _ in range(NB):
        yb = law(t, *p) + rng.choice(resid, size=len(resid), replace=True)
        try:
            boots.append(fit(t, yb))
        except RuntimeError:
            pass
    boots = np.array(boots)
    lo, hi = np.quantile(boots[:, 0], [0.05, 0.95])
    at_bound = p[2] <= 0.0101 or p[2] >= 4.999 or p[0] <= 1e-6
    ident = (hi - lo) <= 0.3 and not at_bound
    return dict(n=len(t), linf=float(p[0]), A=float(p[1]), alpha=float(p[2]), ci=(float(lo), float(hi)),
                ppl_inf=math.exp(p[0]), ppl_ci=(math.exp(lo), math.exp(hi)), rmse=float(np.sqrt((resid ** 2).mean())),
                identified=bool(ident), y_end=float(y[-1]), ppl_end=math.exp(y[-1]))


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    res = {}
    print(f"F0: L(t) = L_inf + A (t/1000)^-alpha on val loss, steps {LO:,}-{HI:,}; 90% interval from {NB} bootstrap refits\n")
    print(f"{'model':<32} {'n':>3} {'L_inf':>7} {'90% interval':>17} {'PPL_inf':>8} {'PPL interval':>17} {'alpha':>6} "
          f"{'rmse':>6} {'PPL@21k':>8} {'identified':>10}")
    for name, folder in list(MODELS.items()) + [('GPT-2, cosine (descriptive only)', None)]:
        ev = evals_gpt2() if folder is None else evals_fock(folder)
        r = analyse(name, ev, rng)
        res[name] = r
        print(f"{name:<32} {r['n']:3d} {r['linf']:7.3f} [{r['ci'][0]:6.3f}, {r['ci'][1]:6.3f}] {r['ppl_inf']:8.2f} "
              f"[{r['ppl_ci'][0]:6.2f}, {r['ppl_ci'][1]:6.2f}] {r['alpha']:6.2f} {r['rmse']:6.3f} {r['ppl_end']:8.2f} "
              f"{str(r['identified']):>10}")
    fock = {k: v for k, v in res.items() if 'GPT-2' not in k}
    l2f = [res[k] for k in ('G2 (L=2, registers)', 'G3 (L=2, + field)', "G3' (L=2, + hardened field)")]
    l4 = res['L=4 Fock']; f31 = res['F3.1 (L=2, conservative-only)']
    spread = max(r['linf'] for r in l2f) - min(r['linf'] for r in l2f)
    p1 = spread <= 0.05
    p2 = all(r['ci'][0] > l4['ci'][1] for r in l2f)
    p3 = all(f31['ci'][0] > r['ci'][1] for r in l2f)
    p4 = any(not r['identified'] for r in fock.values())
    print(f"\nP1  L=2 Fock L_inf spread {spread:.3f} nats (<= 0.05): {'HIT' if p1 else 'MISS'}  (65%)")
    print(f"P2  every L=2 Fock interval above L=4's (depth-set floor): {'HIT' if p2 else 'MISS'}  (45%)")
    print(f"P3  F3.1's interval above every L=2 Fock interval: {'HIT' if p3 else 'MISS'}  (60%)")
    print(f"P4  at least one fit unidentified: {'HIT' if p4 else 'MISS'}  (50%)")
    res['scores'] = dict(P1=p1, P2=p2, P3=p3, P4=p4, l2_spread=spread)
    out = Path(__file__).with_name('f0_floor_fit.json')
    out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {out}")
