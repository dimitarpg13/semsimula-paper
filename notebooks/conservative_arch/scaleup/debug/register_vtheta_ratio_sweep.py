"""Register vs V_theta gradient-ratio sweep over training_log.jsonl.

Companion to Post_100K_CfC_BAOAB_Analysis_Checklist.md SS2.6.

Discriminates "register is riding a V_theta cascade" (Mitigations SS42's
mechanism B on mechanism A) from "register broke away on its own". Reads
only the JSONL -- no bundles, no GPU, no checkpoint loads -- so it covers
every spike in the whole run, including ones whose bundles have rotated out.

    python3 register_vtheta_ratio_sweep.py <RESULTS_DIR>/training_log.jsonl
"""
import json, math, statistics
from pathlib import Path

def register_vtheta_sweep(log_path, reg_key='override:register', vth_key='V_theta'):
    # A hard-trigger step emits BOTH a grad_spike (top-8) and a
    # watchdog_hard_reload (top-5) record. Keep one row per step, preferring
    # the grad_spike -- its top-8 is strictly richer, and counting the step
    # twice would skew the median the decision rule is measured against.
    best = {}
    for line in Path(log_path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        tg = r.get('top_groups')
        if not tg:
            continue
        step = r.get('step')
        prev = best.get(step)
        if prev is not None and prev.get('event') == 'grad_spike':
            continue
        best[step] = r

    rows, partial = [], []
    for step in sorted(best):
        r = best[step]
        tg = r['top_groups']
        tot = r.get('pre_clip_grad_norm') or r.get('raw_grad_norm')
        vth, reg = tg.get(vth_key), tg.get(reg_key)
        if vth and reg:
            rows.append((step, r.get('event'), tot, vth, reg, reg / vth))
        elif vth:
            # register absent => it ranked below the retained top-N,
            # so we only get an UPPER bound on the ratio
            rows.append((step, r.get('event'), tot, vth, None, None))
            partial.append((step, min(tg.values()) / vth))
    return rows, partial

def report(log_path):
    rows, partial = register_vtheta_sweep(log_path)
    print(f'{"step":>7} {"event":>20} {"total":>9} {"V_theta":>9} {"register":>9} {"reg/Vth":>8}')
    for step, ev, tot, vth, reg, ratio in rows:
        rs = f'{ratio:8.2f}' if ratio else '     n/a'
        rg = f'{reg:9.1f}'   if reg   else '      n/a'
        print(f'{step:>7} {str(ev):>20} {tot:9.1f} {vth:9.1f} {rg} {rs}')
    vals = [r[5] for r in rows if r[5]]
    if not vals:
        print('\nno event had both register and V_theta in top_groups')
        return
    med = statistics.median(vals)
    print(f'\nn={len(vals)} events with both groups present')
    print(f'median reg/V_theta = {med:.2f}   range {min(vals):.2f} - {max(vals):.2f}')

    # The ratio is log-normally distributed with a fat right tail, so a
    # multiple-of-the-median rule is NOT an outlier test: measured over 88
    # real spikes (steps 50-56,300) the distribution had geometric mean 0.81
    # and log-sd 0.82, which puts "3x median" (1.95) between p75 (1.16) and
    # p90 (2.50) -- it flagged 14% of all events. Score in log space instead.
    if len(vals) < 8:
        print('\n(too few events for a distribution fit; showing the '
              'top ratios only)')
        for s, v in sorted(((r[0], r[5]) for r in rows if r[5]),
                           key=lambda t: -t[1])[:5]:
            print(f'  step {s}: {v:.2f}')
        return
    logs = [math.log(v) for v in vals]
    mu, sd = statistics.mean(logs), statistics.stdev(logs)
    t2, t3 = math.exp(mu + 2 * sd), math.exp(mu + 3 * sd)
    print(f'log-space fit: geometric mean {math.exp(mu):.2f}, '
          f'sd {sd:.2f}  ->  +2sd = {t2:.2f}, +3sd = {t3:.2f}')

    def z(v):
        return (math.log(v) - mu) / sd

    out = sorted(((r[0], r[5]) for r in rows if r[5] and r[5] > t2),
                 key=lambda t: -t[1])
    print(f'\nDECOUPLED (>+2sd in log space) -> register-specific candidates:')
    for s, v in out:
        mark = '  <-- >3sd, strong' if v > t3 else ''
        print(f'  step {s}: {v:.2f}  (z={z(v):+.2f}){mark}')
    if not out:
        print('  none - every spike is V_theta-slaved, no register-specific event')
    if partial:
        print(f'\n{len(partial)} event(s) had V_theta but no register in top_groups')
        print('  (register ranked below the cut; upper bound on its ratio shown)')
        for s, ub in partial:
            print(f'  step {s}: reg/V_theta < {ub:.2f}')

if __name__ == '__main__':
    import sys
    report(sys.argv[1])
