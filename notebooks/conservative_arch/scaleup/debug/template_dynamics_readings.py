#!/usr/bin/env python3
"""Secondary readings of the dynamics test, after its calls were scored (2026-11-25, POST HOC, no call rests on them).
    python3 template_dynamics_readings.py RUNDIR OUTDIR stem DYNAMICS_JSON
Refits every matcher of template_dynamics_analysis.py at the (m, d) it chose, checks that R1 and R2 come out as stored, keeps the
decision of every test premise, and reads: detection and fallacy fires per unseen wording; the paraphrase gain per form;
threshold-free paraphrase against near miss; the matchers against each other (not only against h0); three concatenations with the
embedding; the clouds of matcher A written out; the cosine between the repeated, paraphrased or unrelated last words, per vector."""
import sys
from pathlib import Path
_D = Path(__file__).with_name('template_dynamics_analysis.py').read_text(); _D = _D[:_D.index("if __name__ == '__main__':")]; STORED = sys.argv[4]; exec(_D)
s = STEMS[0]; J = json.load(open(STORED))[s]
M = pickle.load(open(RUN / f'dynmeta_{s}.pkl', 'rb')); Sx, Wx = np.load(RUN / f'dynS_{s}.npy', mmap_mode='r'), np.load(RUN / f'dynW_{s}.npy', mmap_mode='r'); Tx = np.load(RUN / f'dyntext_{s}.npz')
meta, R, off = M['meta'], M['roles'], M['off']; V, T = vectors(np.asarray(Sx), np.asarray(Wx)), vectors(Tx['S'], Tx['W']); P = particle_stats(V, R, off, T); del V
K = np.array([x['kind'] for x in meta]); want = np.array([x['want'] for x in meta], object); wi = np.array([x['wording'] for x in meta]); sp = np.array([x['split'] for x in meta]); fr = np.array([x['frame'] for x in meta]); g = np.array([x['content'] for x in meta])
lab = np.array(['text' if k == 'text' else f'{f}:{k}' for k, f in zip(K, fr)]); dev, te = sp == 'dev', sp == 'test'; stw = wi == 'text'; contents = np.unique(g[te & ~stw])
HS = [('cond', 'AC'), ('cond', 'DA'), ('cond', 'DAnot'), ('disj', 'P'), ('disj', 'Q')]; WS = sum(WC.values(), [])
def r1(F, C=contents): return np.array([.5 * np.mean([(F[te & (g == c) & (K == t) & (fr == VAL[t][0])] == t).mean() for t in VAL]) + .5 * np.mean([(F[te & (g == c) & (K == k) & (fr == f)] == '').mean() for f, k in HS]) for c in C])
def r2(F): return np.array([np.mean([(F[te & (g == c) & (K == SY[t]) & (fr == VAL[t][0])] == t).mean() for t in VAL]) for c in contents])
def boot(d, B=4000):
    i = np.random.default_rng(1).integers(len(d), size=(B, len(d))); b = d[i].mean(1); return [float(d.mean()), *[float(x) for x in np.percentile(b, [2.5, 97.5])], float((b >= .1).mean()), float((b >= .03).mean())]
MATCH = [('h0', ('h0',)), ('D1', ('D1',)), ('D2', ('D2',)), ('D1+D2', ('D1', 'D2')), ('W1', ('W1',)), ('W2', ('W2',)), ('W1+W2', ('W1', 'W2')), ('h1+h2', ('h1', 'h2')), ('h1', ('h1',)), ('h2', ('h2',)), ('h0+D1+D2', ('h0', 'D1', 'D2')), ('h0+W1+W2', ('h0', 'W1', 'W2')), ('h0+h1+h2', ('h0', 'h1', 'h2'))]
out, fire = {}, {}
for name, ks in MATCH:
    Cs = []
    for k in ks: C = np.stack([x['c'] for x in P[k]]); Cs.append(C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9))
    X_ = lambda m: np.concatenate([np.stack([coords(x, r, m) for x, r in zip(P[k], R)]) for k in ks], 1)
    if name in J: sc, m, d = J[name]['dev'], J[name]['m'], J[name]['d']; X = X_(m)
    else: Xm = {m: X_(m) for m in GRID_M}; sc, m, d = tune(Xm, Cs, dev, lab, want, g, wi, fr); X = Xm[m]; del Xm
    q = out[name] = dict(dev=sc, m=m, d=d, posthoc=name not in J); Fs = {}
    for un in (False, True):
        F, L, rhos = np.full(len(K), '', object), np.zeros(len(K)), []
        for h in (WS if un else [None]):
            a = dev & (wi != h); F_ = Frame([C[a] for C in Cs], wi[a], fr[a]); c = Clouds(np.c_[X[a], F_([C[a] for C in Cs])], lab[a], want[a], d); rho = cvf(X[a], [C[a] for C in Cs], lab[a], want[a], g[a], wi[a], fr[a], d)[0]; rhos.append(rho)
            b = te & ((wi == h) | (stw & (h == 'if-then')) if un else True); Y = np.c_[X[b], F_([C[b] for C in Cs])]; L[b] = c.rates(Y)[3]; F[b] = c.fire(Y, rho)
        Fs[un] = F; tag = 'unseen' if un else 'known'; q[f'{tag} rho'] = rhos; cond = lambda t, kk: te & (K == kk) & (fr == VAL[t][0])
        q[f'{tag} detect'] = {t: ci(F[cond(t, t)] == t, g[cond(t, t)]) for t in VAL}; q[f'{tag} paraphrase'] = {t: ci(F[cond(t, SY[t])] == t, g[cond(t, SY[t])]) for t in VAL}
        q[f'{tag} fires on'] = {f'{f}:{kk}': ci(F[te & (K == kk) & (fr == f)] != '', g[te & (K == kk) & (fr == f)]) for f, kk in HS}
        q[f'{tag} by wording'] = {w: dict(detect=float((F[te & (wi == w) & np.isin(K, list(VAL))] == want[te & (wi == w) & np.isin(K, list(VAL))]).mean()), fallacy=float((F[te & (wi == w) & np.isin(K, ['AC', 'DA', 'DAnot', 'P', 'Q'])] != '').mean()), each={t: float((F[te & (wi == w) & (K == t)] == t).mean()) for t in VAL if (te & (wi == w) & (K == t)).any()}) for w in WS}
        sy, nr = te & np.isin(K, SYN_V), te & np.isin(K, DIFF['near miss']); q[f'{tag} AUROC paraphrase / near miss'] = auroc(L[sy], L[nr]); q[f'{tag} near fires'] = float((F[nr] != '').mean())
        rng = np.random.default_rng(2); bs = []
        for _ in range(200): gi = rng.choice(contents, len(contents)); bs.append(auroc(np.concatenate([L[sy & (g == x)] for x in gi]), np.concatenate([L[nr & (g == x)] for x in gi])))
        q[f'{tag} AUROC paraphrase / near miss ci'] = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
        if name == 'D1+D2' and not un:
            fn = [f'{w_} {n_}' for w_ in ('step 0', 'step 1') for n_ in names(m)] + ['connective position, step 0', 'connective position, step 1']; Xf = np.c_[X[dev], F_([C[dev] for C in Cs])]; Yd = ((Xf - c.mu0) / c.sd) @ c.B
            corr = np.array([[np.corrcoef(Xf[:, j], Yd[:, i])[0, 1] for j in range(Xf.shape[1])] for i in range(d)]); Yt = ((np.c_[X[te], F_([C[te] for C in Cs])] - c.mu0) / c.sd) @ c.B
            q['cloud'] = dict(rho=rho, singular=c.sv.tolist(), classes={k: dict(template=c.tpl[k], centre=c.C[i].tolist(), reach=float(c.reach[k])) for i, k in enumerate(c.cl)}, directions=[[(fn[j], float(corr[i, j])) for j in np.argsort(-abs(corr[i]))[:6]] for i in range(d)], D=int(Xf.shape[1]))
            q['cloud']['test sq distance to own centre'] = {k: [float(x) for x in np.percentile(((Yt[lab[te] == k] - c.C[c.cl.index(k)]) ** 2).sum(1), [25, 50, 75])] for k in c.cl if (lab[te] == k).any()}
            q['cloud']['centre distances'] = {f'{a_}|{b_}': float(np.linalg.norm(c.C[c.cl.index(a_)] - c.C[c.cl.index(b_)])) for a_, b_ in (('cond:MP', 'cond:AC'), ('cond:MP', 'cond:U'), ('cond:MP', 'cond:near'), ('cond:MT', 'cond:DA'), ('cond:MT', 'cond:DAnot'), ('cond:MP', 'cond:MT'), ('disj:DS1', 'disj:P'), ('disj:DS2', 'disj:Q'), ('disj:DS1', 'disj:DS2'), ('disj:DS1', 'cond:MT'), ('cond:MP', 'disj:P')) if a_ in c.cl and b_ in c.cl}
            np.savez(OUT / f'template_cloud_dynamics_{s}.npz', mu0=c.mu0, sd=c.sd, B=c.B, C=c.C, classes=np.array(c.cl), template=np.array([c.tpl[k] for k in c.cl]), reach=np.array([c.reach[k] for k in c.cl]), coordinate_names=np.array(fn))
    fire[name] = Fs; q['R1 unseen'], q['R2 known'] = float(r1(Fs[True]).mean()), float(r2(Fs[False]).mean())
    if name in J: assert abs(q['R1 unseen'] - J[name]['R1 unseen']) < 1e-9 and abs(q['R2 known'] - J[name]['R2 known']) < 1e-9, (name, q['R1 unseen'], q['R2 known'])
    print(name, m, d, round(sc, 3), 'R1', round(q['R1 unseen'], 4), 'R2', round(q['R2 known'], 4), 'AUROC para/near known', round(q['known AUROC paraphrase / near miss'], 3), q['known AUROC paraphrase / near miss ci'], flush=True)
    json.dump(out, open(OUT / f'template_dynamics_readings_{s}.json', 'w'), indent=1)
pair = out['pairs'] = {}
for a_, b_ in (('D1+D2', 'h0'), ('W1+W2', 'h0'), ('D1+D2', 'h1+h2'), ('D1+D2', 'h2'), ('h2', 'h0'), ('W1+W2', 'h1+h2'), ('h0+D1+D2', 'h0'), ('h0+W1+W2', 'h0'), ('h0+h1+h2', 'h0')):
    pair[f'{a_} minus {b_}'] = dict(R1=boot(r1(fire[a_][True]) - r1(fire[b_][True])), R2=boot(r2(fire[a_][False]) - r2(fire[b_][False]))); print(a_, '-', b_, pair[f'{a_} minus {b_}'])
sy = te & np.isin(K, SYN_V); hit = {n: fire[n][False][sy] == want[sy] for n, _ in MATCH}; out['paraphrase overlap'] = {f'{a_}|{b_}': dict(a=int(hit[a_].sum()), b=int(hit[b_].sum()), both=int((hit[a_] & hit[b_]).sum()), n=int(sy.sum())) for a_, b_ in (('D1+D2', 'h2'), ('D1+D2', 'h0'), ('D1+D2', 'h1+h2'))}
# which clauses carry the paraphrase gain: per content, A's and h0's hits over the 24 paraphrase premises of the content
out['paraphrase per content'] = {n: (np.array([hit[n][g[sy] == c].mean() for c in contents])).tolist() for n in ('h0', 'D1+D2', 'h2')}; out['contents'] = [int(c) for c in contents]
# the repeated, paraphrased, near-miss and unrelated second sentence: cosine of its last particle to the last particle of the clause it repeats
cos = out['last particle cosine'] = {}
for k in ('h0', 'D1', 'D2', 'W1', 'W2', 'h1', 'h2'):
    cos[k] = {}
    for kk, fr_, cl in (('MP', 'cond', 'a1'), ('MPsyn', 'cond', 'a1'), ('near', 'cond', 'a1'), ('swap', 'cond', 'a1'), ('U', 'cond', 'a1')):
        v = [];
        for i in np.flatnonzero(te & (K == kk) & (fr == fr_)):
            r = np.array(R[i]); p_, c_ = np.flatnonzero(r == 'p2')[-1], np.flatnonzero(r == cl)[-1]; v.append(P[k][i]['G'][p_, c_])
        cos[k][kk] = [float(x) for x in np.percentile(v, [25, 50, 75])]
    a_ = te & (K == 'MPsyn') & (fr == 'cond'); b_ = te & (K == 'near') & (fr == 'cond'); val_ = lambda idx: np.array([P[k][i]['G'][np.flatnonzero(np.array(R[i]) == 'p2')[-1], np.flatnonzero(np.array(R[i]) == 'a1')[-1]] for i in np.flatnonzero(idx)]); cos[k]['AUROC paraphrase over near miss'] = auroc(val_(a_), val_(b_))
np.savez_compressed(OUT / f'template_dynamics_fires_{s}.npz', **{f'{n}|{"unseen" if u else "known"}': fire[n][u][te].astype(str) for n in fire for u in (False, True)}, kind=K[te], wording=wi[te], frame=fr[te], content=g[te])
json.dump(out, open(OUT / f'template_dynamics_readings_{s}.json', 'w'), indent=1); print('done')
