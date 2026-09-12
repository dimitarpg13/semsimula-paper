"""Register vs V_theta gradient-ratio sweep over training_log.jsonl.

Companion to Post_100K_CfC_BAOAB_Analysis_Checklist.md SS2.6.

Discriminates "register is riding a V_theta cascade" (Mitigations SS42's
mechanism B on mechanism A) from "register broke away on its own". Reads
only the JSONL -- no bundles, no GPU, no checkpoint loads -- so it covers
every spike in the whole run, including ones whose bundles have rotated out.

    python3 register_vtheta_ratio_sweep.py <RESULTS_DIR>/training_log.jsonl
"""
import json, statistics
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
    out = [(r[0], r[5]) for r in rows if r[5] and r[5] > 3 * med]
    print(f'\nDECOUPLED (>3x median) -> register-specific candidates:')
    for s, v in out:
        print(f'  step {s}: {v:.2f}  ({v/med:.1f}x median)')
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
