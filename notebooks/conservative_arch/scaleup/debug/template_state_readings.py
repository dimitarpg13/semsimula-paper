#!/usr/bin/env python3
"""What a template matcher reads: the embedding, the layer states and the wells, on one frozen checkpoint (2026-11-25).
    python3 template_state_readings.py ROBUST.pkl DYNTEXT.npz OUT_STEM

ROBUST.pkl  premise geometry of template_robust_test.py (whitened cosines between particles, states h0, h1, h2)
DYNTEXT.npz ordinary text of template_dynamics_test.py: S = h0, pre1, h1, pre2, h2 (pre = before the layer-end LayerNorm),
            W = per layer w 8, q in 8, u in 32, q pre 8, u pre 32, q out 8 (the bank read inside the forward pass)
Prints and saves OUT_STEM.txt / .json. Reads stored files only; runs no model."""
import sys, json, pickle
import numpy as np
P, T, OUT = sys.argv[1:4]
out, lines = {}, []
def say(*a): s = ' '.join(str(x) for x in a); print(s); lines.append(s)
med = lambda a: float(np.median(a))

d = pickle.load(open(P, 'rb')); M, R = d['meta'], d['res']
say('1. whitened cosines, test clauses, if-then wording: last word of premise 2 against last word of clause 1')
for l in (0, 1, 2):
    v = {'MP': [], 'U': []}
    for x, r in zip(M, R):
        if x['split'] == 'test' and x['wording'] == 'if-then' and x['kind'] in v:
            ro = np.array(r['roles']); a, p = np.flatnonzero(ro == 'a1'), np.flatnonzero(ro == 'p2'); v[x['kind']].append(r['st'][l]['G'][p[-1], a[-1]])
    out[f'cos_h{l}'] = {k: [len(b), med(b), float(min(b)), float(max(b))] for k, b in v.items()}
    say(f'  h{l}  repeated (MP, n={len(v["MP"])}) {med(v["MP"]):.3f} [{min(v["MP"]):.3f}, {max(v["MP"]):.3f}]   unrelated (U, n={len(v["U"])}) {med(v["U"]):.3f}')
idx = [i for i, x in enumerate(M) if x['split'] == 'test' and x['kind'] in ('MP', 'MT', 'DS1', 'DS2', 'AC', 'DA')]
idx = np.random.default_rng(0).choice(idx, 400, replace=False)
say('2. correlation of the particle cosines of one premise between two states (400 test premises, median [10%, 90%])')
for a, b in ((0, 1), (0, 2), (1, 2)):
    c = []
    for i in idx:
        A, B = R[i]['st'][a]['G'].astype(float), R[i]['st'][b]['G'].astype(float); t = np.triu_indices(len(A), 1); c.append(np.corrcoef(A[t], B[t])[0, 1])
    q = np.percentile(c, [10, 90]); out[f'corr_{a}{b}'] = [med(c), *map(float, q)]; say(f'  h{a} vs h{b}  {med(c):.3f} [{q[0]:.3f}, {q[1]:.3f}]')

z = np.load(T); S, W = z['S'].astype(np.float64), z['W'].astype(np.float64); n = lambda a: np.linalg.norm(a, axis=1); cos = lambda a, b: (a * b).sum(1) / n(a) / n(b)
h = dict(zip(('h0', 'pre1', 'h1', 'pre2', 'h2'), [S[:, 384 * i:384 * i + 384] for i in range(5)]))
say(f'3. lengths and displacements, {len(S)} ordinary tokens (raw states, medians)')
for k, v in h.items(): out[f'norm_{k}'] = med(n(v)); say(f'  |{k}| {med(n(v)):.2f}')
for l, (a, b) in enumerate((('h0', 'pre1'), ('h1', 'pre2'))):
    D = h[b] - h[a]; e = [med(n(D)), med(n(D) / n(h[a])), med(cos(D, h[a]))]; out[f'disp_{l}'] = e
    say(f'  layer {l}: |displacement| {e[0]:.2f}, over |state on entry| {e[1]:.2f}, cosine with the state on entry {e[2]:.2f}')
for a, b in (('h0', 'h1'), ('h1', 'h2'), ('h0', 'h2')):
    X = np.c_[h[a], np.ones(len(S))]; tr = np.arange(len(S)) % 2 == 0; B = np.linalg.lstsq(X[tr], h[b][tr], rcond=None)[0]
    r2 = 1 - ((h[b][~tr] - X[~tr] @ B) ** 2).sum() / ((h[b][~tr] - h[b][tr].mean(0)) ** 2).sum()
    out[f'lin_{a}_{b}'] = [med(cos(h[a], h[b])), float(r2)]; say(f'  {a} -> {b}: cosine {med(cos(h[a], h[b])):.2f}, linear fit explains {r2:.3f} of the held-out variance')
say('4. the wells: radius sqrt(q) to the well that dominates on entry, in the layer\'s own context')
for l in (0, 1):
    b = W[:, l * 96:(l + 1) * 96]; w, qi, qo, qp = b[:, :8], b[:, 8:16], b[:, 48:56], b[:, 88:96]; i = np.arange(len(b))
    share = lambda q: w * np.exp(-q / 2) / (w * np.exp(-q / 2)).sum(1, keepdims=True)
    k, kp = share(qi).argmax(1), share(qo).argmax(1); r = [np.sqrt(q[i, k]) for q in (qi, qo, qp)]
    e = dict(entry=np.percentile(r[0], [10, 50, 90]).tolist(), out_pre=np.percentile(r[1], [10, 50, 90]).tolist(), out_post=np.percentile(r[2], [10, 50, 90]).tolist(),
             closer=float((r[1] < r[0]).mean()), same=float((k == kp).mean()), new_r=med(np.sqrt(qo[i, kp])), nearest=med(np.sqrt(qo.min(1))),
             share_in=med(share(qi).max(1)), occ_in=med(np.exp(-qi[i, k] / 2)), occ_out=med(np.exp(-qo[i, kp] / 2)), dominant_in=np.bincount(k, minlength=8).tolist(), dominant_out=np.bincount(kp, minlength=8).tolist())
    out[f'well_{l}'] = e; p = lambda a: '/'.join(f'{x:.2f}' for x in a)
    say(f'  layer {l}: radius 10/50/90% on entry {p(e["entry"])}, after the step {p(e["out_pre"])}, after its LayerNorm {p(e["out_post"])}; closer after the step {100 * e["closer"]:.2f}%')
    say(f'    dominant wells on entry {e["dominant_in"]}, after the step {e["dominant_out"]}; same well {100 * e["same"]:.1f}%; radius to the new dominant well {e["new_r"]:.2f}, to the nearest {e["nearest"]:.2f}')
    say(f'    occupancy exp(-q/2) of the dominant well on entry {e["occ_in"]:.3f}, after the step {e["occ_out"]:.2e}; median share on entry {e["share_in"]:.2f}')
json.dump(out, open(OUT + '.json', 'w'), indent=1); open(OUT + '.txt', 'w').write('\n'.join(lines) + '\n')
