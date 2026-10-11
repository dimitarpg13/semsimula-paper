"""Where does PMX's step time go? (protocol SS5.15, 2026-10-10)

Profiles one training step (forward + backward) of PMX (pmx16 on SR2) against
the same model with explicit wells (PM1 + SR2), on PM1's weights, CPU. Labelled
wrappers time the suspected components (inclusive, per call site); the
profiler's operator table shows what dominates inside them. CPU timings are
not GPU timings, but the split between eigensolves, occupation and tensor work
indicates where a faster implementation pays.

Usage: python3 profile_pmx_step.py OUT_DIR [B] [T]
"""
import sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

if __name__ == '__main__':
    OUT = Path(sys.argv[1])
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    sys.argv = [sys.argv[0], str(OUT / 'harness')]
    sys.path.insert(0, str(Path(__file__).parent))
    import verify_pm_switch as P
    PM1F = ('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64'
            '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn')
    SR2 = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
    PMX = (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),)
    model, g, _ = P.build(PM1F, P.CONFIGS['F3.1'][1] + P.PM_ON + SR2 + PMX)
    model.to('cpu')
    import model_parf_multixi as MPX, cfc_baoab as C
    xb, yb = g['get_batch'](g['val_ids'], B, T, np.random.default_rng(20260920))
    x, y = torch.from_numpy(xb), torch.from_numpy(yb)

    timers = defaultdict(float); counts = defaultdict(int)
    def wrap(owner, name, label):
        orig = getattr(owner, name)
        def w(*a, **k):
            t0 = time.perf_counter()
            with torch.profiler.record_function(label):
                out = orig(*a, **k)
            timers[label] += time.perf_counter() - t0; counts[label] += 1
            return out
        setattr(owner, name, w)
        return orig
    restore = []
    for owner, name, label in ((MPX, 'lowrank_modes', 'Vtheta lowrank_modes (SVD)'),
                               (MPX, 'indefinite_lowrank_modes', 'PMX indefinite_lowrank_modes'),
                               (C, '_gram_eigh', '  _gram_eigh (inside both)'),
                               (MPX, 'lowrank_iso_damped_substep', 'PMX exact substep'),
                               (MPX, 'lowrank_damped_substep', 'SR2 exact substep'),
                               (type(model), 'poisson_mode_occupation', 'occupation phi (O(K T^2))'),
                               (type(model), 'poisson_mode_quadratic', 'PMX quadratic'),
                               (type(model), 'poisson_mode_force', 'wells force (kick)')):
        restore.append((owner, name, wrap(owner, name, label)))

    def step():
        model.train(); torch.manual_seed(7)
        t0 = time.perf_counter()
        _, loss = model(x, y)
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        model.zero_grad(set_to_none=True)
        return t1 - t0, t2 - t1

    lines = [f'PMX step profile, CPU, {B} x {T} tokens, PM1 weights, train mode (forward + backward)']
    res = {}
    for arm, flag in (('explicit wells on SR2', False), ('PMX pmx16 on SR2', True)):
        model.cfg.poisson_wells_exact = flag
        step()                                            # warm-up
        timers.clear(); counts.clear()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            tf, tb = step()
        res[arm] = (tf, tb, dict(timers), dict(counts), prof)
        lines += ['', f'== {arm}: forward {tf:.2f} s, backward {tb:.2f} s, total {tf + tb:.2f} s',
                  '   component (inclusive wall time in the forward pass and the checkpoint recompute; calls)']
        for k in sorted(timers, key=lambda k: -timers[k]):
            lines.append(f'   {k:<34} {timers[k]:7.2f} s  ({counts[k]} calls)')
        lines.append('   top operators by self CPU time:')
        tab = prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=14)
        lines += ['   ' + l for l in tab.split('\n')]
    for owner, name, orig in restore:
        setattr(owner, name, orig)
    a, b_ = res['explicit wells on SR2'], res['PMX pmx16 on SR2']
    lines += ['', f'PMX / explicit: forward {b_[0] / a[0]:.2f}x, backward {b_[1] / a[1]:.2f}x, total {(b_[0] + b_[1]) / (a[0] + a[1]):.2f}x']
    txt = '\n'.join(lines)
    print(txt)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'profile_pmx_step_output.txt').write_text(txt + '\n')
