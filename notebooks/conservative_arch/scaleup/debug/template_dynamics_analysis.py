#!/usr/bin/env python3
"""Templates read from the dynamics: displacement and well-frame matchers against the embedding (2026-11-25; written before
any result of these matchers was seen).
    python3 template_dynamics_analysis.py RUNDIR OUTDIR stem[,stem...]

What differs between matchers is only the vector that stands for a particle. Everything after that is identical to
template_geometry_analysis.py / template_geometry_frame.py: whitening by ordinary text, the cosines and norms between
particles, the relational coordinates, one Gaussian cloud per kind, the discriminant subspace, the 1% false-fire rule, plus
the connective's position along the disjunctive-conditional axis. Particle vectors:
  h0, h1, h2     the states (h0 the reference the matchers must beat; h1 and h2 carried over from the earlier test)
  D1 = pre1 - h0, D2 = pre2 - h1     the displacement of each layer, before the layer-end LayerNorm
  W1, W2         the well-frame reading of each layer: per well log(w q-exponent) share, radius in, radius change, stiff
                 coordinates in, stiff-coordinate change (the bank's own frame at the context the layer used)
  combinations D1+D2, W1+W2, h1+h2 are concatenations of their relational coordinates (one axis each)
(m, d) is the grid point with the best content-wise five-fold development score, the same search for every matcher:
m in 1..8 and d in 6, 10, 15, 20, 25, 31. Nothing else is tuned. Test clauses are the 60 of template_fresh_clauses.py.

Primary readings (see Template_Matchers_on_the_Dynamics.md):
  A = D1+D2 against h0, B = W1+W2 against h0
  R1 hard-set balanced score on unseen wordings: half the mean detection of MP, MT, DS1, DS2 (right template), half the mean
     silence on AC, DA, DAnot, affirmed disjunct (P, Q)
  R2 paraphrase detection on known wordings: mean over MPsyn, MTsyn, DS1syn, DS2syn (subject kept, predicate paraphrased)
Differences to h0 are paired over clauses (bootstrap, 1000 resamples of the content index)."""
import sys
from pathlib import Path
_A = Path(__file__).with_name('template_geometry_analysis.py').read_text(); _A = _A[:_A.index("res = {}; np.random.seed(0)")]; exec(_A)
_F = Path(__file__).with_name('template_geometry_frame.py').read_text(); exec(_F[_F.index("class Frame:"):_F.index("\nout = {}\nfor s in STEMS")])
VAL = {'MP': ('cond', 'MP'), 'MT': ('cond', 'MT'), 'DS1': ('disj', 'DS1'), 'DS2': ('disj', 'DS2')}; SY = {t: t + 'syn' for t in VAL}
GRID_M, GRID_D = range(1, 9), (6, 10, 15, 20, 25, 31); ROLES = ('c', 'a1', 'a2', 'p2', 'th')
def wellframe(W, l):
    b = W[:, l * 96:(l + 1) * 96]; w, qi, ui, qo, uo = b[:, :8], b[:, 8:16], b[:, 16:48], b[:, 48:56], b[:, 56:88]
    lg = np.log(np.maximum(w, 1e-12)) - .5 * qi; lp = lg - np.logaddexp.reduce(lg, axis=1, keepdims=True); ri, ro = np.sqrt(qi), np.sqrt(qo)
    return np.c_[np.maximum(lp, -20), ri, ro - ri, ui, uo - ui]
def vectors(S, W):
    h0, p1, h1, p2, h2 = (S[:, 384 * i:384 * (i + 1)] for i in range(5)); return {'h0': h0, 'h1': h1, 'h2': h2, 'D1': p1 - h0, 'D2': p2 - h1, 'W1': wellframe(W, 0), 'W2': wellframe(W, 1)}
def whiten(x):
    mu = x.mean(0); e, U = np.linalg.eigh(np.cov((x - mu).T)); e = np.maximum(e, 0); return mu, U / np.sqrt(e + .1 * e.mean())
def particle_stats(V, R, off, T):
    """per item and vector kind: cosines, norms and the connective's mean whitened vector."""
    out = {}
    for k in V:
        mu, U = whiten(T[k]); Z = (V[k] - mu) @ U; nm = np.linalg.norm(Z, axis=1); Zu = Z / (nm[:, None] + 1e-9); st = []
        for i in range(len(R)):
            a, b = off[i], off[i + 1]; c = np.array([q == 'c' for q in R[i]]); st.append(dict(G=Zu[a:b] @ Zu[a:b].T, norm=nm[a:b], c=Z[a:b][c].mean(0)))
        out[k] = st
    return out
def tune(Xm, Cs, dev, lab, want, g, wi, fr):
    best = None
    for m in GRID_M:
        for d in GRID_D:
            if d > Xm[m].shape[1]: continue
            s = cvf(Xm[m][dev], [C[dev] for C in Cs], lab[dev], want[dev], g[dev], wi[dev], fr[dev], d)[1]
            if best is None or s > best[0] + 1e-12: best = (s, m, d)
    return best
PAIRED = 1000
def paired(vA, v0, g):
    """vA, v0: per content (n_contents,) arrays; mean difference and 95% CI over contents"""
    d = vA - v0; rng = np.random.default_rng(0); i = rng.integers(len(d), size=(PAIRED, len(d))); b = d[i].mean(1); return [float(d.mean())] + [float(x) for x in np.percentile(b, [2.5, 97.5])]
if __name__ == '__main__':
    out = {}
    for s in STEMS:
        M = pickle.load(open(RUN / f'dynmeta_{s}.pkl', 'rb')); Sx, Wx = np.load(RUN / f'dynS_{s}.npy', mmap_mode='r'), np.load(RUN / f'dynW_{s}.npy', mmap_mode='r'); Tx = np.load(RUN / f'dyntext_{s}.npz')
        meta, R, off = M['meta'], M['roles'], M['off']; V, T = vectors(np.asarray(Sx), np.asarray(Wx)), vectors(Tx['S'], Tx['W']); P = particle_stats(V, R, off, T); del V
        K = np.array([x['kind'] for x in meta]); want = np.array([x['want'] for x in meta], object); wi = np.array([x['wording'] for x in meta]); sp = np.array([x['split'] for x in meta]); fr = np.array([x['frame'] for x in meta]); g = np.array([x['content'] for x in meta])
        lab = np.array(['text' if k == 'text' else f'{f}:{k}' for k, f in zip(K, fr)]); dev, te = sp == 'dev', sp == 'test'; stw = wi == 'text'
        fire_un, fire_kn, q = {}, {}, out.setdefault(s, {})
        for name, ks in (('h0', ('h0',)), ('D1', ('D1',)), ('D2', ('D2',)), ('D1+D2', ('D1', 'D2')), ('W1', ('W1',)), ('W2', ('W2',)), ('W1+W2', ('W1', 'W2')), ('h1+h2', ('h1', 'h2')), ('h1', ('h1',)), ('h2', ('h2',))):
            Cs = []
            for k in ks: C = np.stack([x['c'] for x in P[k]]); Cs.append(C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9))
            Xm = {m: np.concatenate([np.stack([coords(x, r, m) for x, r in zip(P[k], R)]) for k in ks], 1) for m in GRID_M}; sc, m, d = tune(Xm, Cs, dev, lab, want, g, wi, fr); X = Xm[m]; r = q[name] = dict(dev=sc, m=m, d=d)
            for un in (False, True):
                F, L = np.full(len(K), '', object), np.zeros(len(K))
                for h in (sum(WC.values(), []) if un else [None]):
                    a = dev & (wi != h); F_ = Frame([C[a] for C in Cs], wi[a], fr[a]); c = Clouds(np.c_[X[a], F_([C[a] for C in Cs])], lab[a], want[a], d); rho = cvf(X[a], [C[a] for C in Cs], lab[a], want[a], g[a], wi[a], fr[a], d)[0]; b = te & ((wi == h) | (stw & (h == 'if-then')) if un else True); Y = np.c_[X[b], F_([C[b] for C in Cs])]; L[b] = c.rates(Y)[3]; F[b] = c.fire(Y, rho)
                (fire_un if un else fire_kn)[name] = F
                cond = lambda t, kk: te & (K == kk) & (fr == VAL[t][0]); rp = report(F, K, want, g, L, te)
                r['unseen' if un else 'known'] = dict(report=rp, detect={t: ci(F[cond(t, t)] == t, g[cond(t, t)]) for t in VAL}, para={t: ci(F[cond(t, SY[t])] == t, g[cond(t, SY[t])]) for t in VAL}, near_false=float((F[te & np.isin(K, DIFF['near miss'])] != '').mean()))
        # primary contrasts, paired over test contents: detection/silence per content, averaged over the wordings
        contents = np.unique(g[te & ~stw]); hs = [('AC', 'cond', 'AC'), ('DA', 'cond', 'DA'), ('DAnot', 'cond', 'DAnot'), ('P', 'disj', 'P'), ('Q', 'disj', 'Q')]
        def r1(F):
            out_ = []
            for c in contents:
                m = te & (g == c); det = np.mean([(F[m & (K == t) & (fr == VAL[t][0])] == t).mean() for t in VAL]); sil = np.mean([(F[m & (K == kk) & (fr == f)] == '').mean() for _, f, kk in hs]); out_.append(.5 * det + .5 * sil)
            return np.array(out_)
        def r2(F): return np.array([np.mean([(F[te & (g == c) & (K == SY[t]) & (fr == VAL[t][0])] == t).mean() for t in VAL]) for c in contents])
        for nm in ('D1+D2', 'W1+W2', 'h1+h2', 'D1', 'D2', 'W1', 'W2', 'h1', 'h2'):
            q[nm]['R1 unseen'] = float(r1(fire_un[nm]).mean()); q[nm]['R1 unseen minus h0'] = paired(r1(fire_un[nm]), r1(fire_un['h0']), g); q[nm]['R2 known'] = float(r2(fire_kn[nm]).mean()); q[nm]['R2 known minus h0'] = paired(r2(fire_kn[nm]), r2(fire_kn['h0']), g)
        q['h0']['R1 unseen'], q['h0']['R2 known'] = float(r1(fire_un['h0']).mean()), float(r2(fire_kn['h0']).mean())
        fm = lambda v: f'{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]'
        for nm in q:
            r = q[nm]; p = r['known']['report']; u = r['unseen']; print(f"{s} {nm:6s} m{r['m']} d{r['d']:2d} dev {r['dev']:.3f} | known det " + ' '.join(f"{t} {r['known']['detect'][t][0]:.2f}" for t in VAL) + f" para {r['R2 known']:.2f} false any {p['any false fire'][0]:.3f} near {r['known']['near_false']:.3f} | unseen det " + ' '.join(f"{t} {u['detect'][t][0]:.2f}" for t in VAL) + f" false AC {u['report']['false AC'][0]:.2f} DA {u['report']['false DA'][0]:.2f} disj {u['report']['false affirmed disjunct'][0]:.2f} wrong {u['report']['wrong template']:.2f} | R1 unseen {r['R1 unseen']:.3f}" + (f" vs h0 R1 {fm(r['R1 unseen minus h0'])} R2 {fm(r['R2 known minus h0'])}" if nm != 'h0' else ''), flush=True)
        json.dump(out, open(OUT / 'template_dynamics.json', 'w'), indent=1)
