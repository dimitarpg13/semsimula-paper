#!/usr/bin/env python3
"""How many wells does a layer need? Keep only its k strongest wells, k = 1..K-1, and read the loss (2026-11-03).

    python3 keep_k_curves.py OUT FOLDER NAME LAYER [ckpt=<file>] [windows=16] [threads=8]

The empirical K_needed of one layer: vtheta_well_probe.py is run with keep=<the k wells with the largest force-weight
share in that layer, from OUT/well_probe_<NAME>.txt> and keep_layers=<LAYER>, for k = 1 .. K-1 (k = K is the trained
model, whose report is the base). Greedy by force share, so each k's set contains the previous one; the best k-subset
could differ. Writes OUT/well_probe_<NAME>_L<LAYER>_keep<k>.txt (runs already there are kept) and the curve to
OUT/keep_k_<NAME>_L<LAYER>.txt: k, wells kept, ppl, change against the base.
"""
import re, subprocess, sys
from pathlib import Path

OUT, FOLDER, NAME, LAYER = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3], int(sys.argv[4])
OPTS = [a for a in sys.argv[5:] if '=' in a and not a.startswith(('keep', 'name'))]
RN = 'r' if 'renorm=1' in OPTS else ''     # renorm=1: kept weights rescaled to the trained sum (vtheta_well_probe.py)
base = OUT / f'well_probe_{NAME}.txt'
txt = base.read_text()
ppl = lambda t: float(re.search(r'ppl\s+([\d.]+)', t.splitlines()[0]).group(1))
m = re.search(rf'^  layer {LAYER}: force\s+(.*?)\s+\|', txt, re.M)
share = [float(v) for v in m.group(1).split()]
order = sorted(range(len(share)), key=lambda k: -share[k])
p0 = ppl(txt)
rows = [f'keep-k curve{" (kept weights renormalised)" if RN else ""}, {NAME} layer {LAYER}: wells by force share {order} (shares {share}); base ppl {p0:.2f} on these windows',
        f'{"k":>2} {"wells kept":<24} {"ppl":>9} {"change":>9}']
for k in range(1, len(share)):
    keep = sorted(order[:k])
    out = OUT / f'well_probe_{NAME}_L{LAYER}_keep{k}{RN}.txt'
    if not out.exists():
        cmd = [sys.executable, str(Path(__file__).resolve().parent / 'vtheta_well_probe.py'), str(OUT), FOLDER,
               f'name={NAME}_L{LAYER}_keep{k}{RN}', f'keep={",".join(map(str, keep))}', f'keep_layers={LAYER}', *OPTS]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'probe failed at k = {k}:\n{r.stderr[-2000:]}')
    pk = ppl(out.read_text())
    rows.append(f'{k:2d} {str(keep):<24} {pk:9.2f} {100 * (pk / p0 - 1):+8.1f}%')
    print(rows[-1], flush=True)
rows.append(f'{len(share):2d} {"all (trained)":<24} {p0:9.2f} {"":>9}')
(OUT / f'keep_k_{NAME}_L{LAYER}{"_renorm" if RN else ""}.txt').write_text('\n'.join(rows) + '\n')
print('\n'.join(rows[:1] + rows[-1:]))
