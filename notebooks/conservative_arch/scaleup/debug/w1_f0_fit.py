"""W1.3-W1.4: the F0 stable-phase fit on the matched GPT-2 trained on WSD (protocol SS5.17).

Same law, window (steps 3,000-21,000), bootstrap and identification rule as
f0_floor_fit.py, whose functions are reused unchanged. Evals are parsed from
the run's printed output (loss@512, 4 decimals), since the WSD run's
training_log.jsonl was not downloaded.

Usage: python3 w1_f0_fit.py [OUTPUT_TXT]
"""
import json, math, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import f0_floor_fit as F

TXT = Path(sys.argv[1]) if len(sys.argv) > 1 else F.DL / 'gpt2_matched_baseline_wsd_schedule_output.txt'

if __name__ == '__main__':
    ev = {}
    for m in re.finditer(r'EVAL step ([\d,]+)\s+loss@1024=[\d.]+ ppl@1024=[\d.]+\s+loss@512=([\d.]+)', TXT.read_text()):
        ev[int(m.group(1).replace(',', ''))] = float(m.group(2))
    f0 = json.loads(Path(__file__).with_name('f0_floor_fit.json').read_text())
    l4 = f0['L=4 Fock']
    rng = np.random.default_rng(F.SEED)
    r = F.analyse('GPT-2, WSD', ev, rng)
    rc = F.analyse('GPT-2, cosine (descriptive)', F.evals_gpt2(), np.random.default_rng(F.SEED))
    print(f"{len(ev)} evals parsed, steps {min(ev):,}-{max(ev):,}; window {F.LO:,}-{F.HI:,}: {r['n']} evals\n")
    for name, x in (('GPT-2, WSD (W1)', r), ('GPT-2, cosine (descriptive)', rc), ('L=4 Fock (F0)', l4)):
        print(f"{name:<28} L_inf {x['linf']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]  PPL_inf {x['ppl_inf']:.2f} "
              f"[{x['ppl_ci'][0]:.2f}, {x['ppl_ci'][1]:.2f}]  alpha {x['alpha']:.2f}  rmse {x['rmse']:.4f}  "
              f"identified {x['identified']}")
    w13 = r['identified']
    w14 = r['identified'] and r['linf'] < l4['ci'][0]
    print(f"\nW1.3 the fit is identified (interval <= 0.3 nats, alpha off its bounds): {'HIT' if w13 else 'MISS'}  (75%)")
    print(f"W1.4 its L_inf lies below L=4 Fock's 90% interval (< {l4['ci'][0]:.3f} nats): "
          f"{'HIT' if w14 else 'MISS'}  (50%)")
    out = Path(__file__).with_name('w1_f0_fit.json')
    out.write_text(json.dumps({'wsd': r, 'cosine_descriptive': rc, 'L4': l4, 'W1.3': w13, 'W1.4': w14}, indent=1))
