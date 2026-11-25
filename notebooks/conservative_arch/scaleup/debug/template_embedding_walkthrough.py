#!/usr/bin/env python3
"""The embedding-based template matcher, one premise at a time (2026-11-25).
    python3 template_embedding_walkthrough.py RUNDIR OUTDIR stem
The matcher of template_geometry_frame.py on the embedding h0 (m = 8 particles, d = 10, connective's position included), fitted on
all development premises (the known-wording fit). Written out: the clouds (centres, reach, threshold, what each direction reads), and
for a set of test premises on one clause pair ("the coffee is cold", "the carpet is filthy") every step of the decision: the tokens of
each role, the particle cosines and norms, the relational coordinates, the projection, the distances to the clouds, the rates, the
decision; for the modus ponens premise also which single coordinates, swapped for its near miss's or its paraphrase's, move the
decision most. Writes OUTDIR/template_embedding_walkthrough_<stem>.json."""
import sys
from pathlib import Path
_A = Path(__file__).with_name('template_geometry_analysis.py').read_text(); _A = _A[:_A.index("res = {}; np.random.seed(0)")]; exec(_A)
_F = Path(__file__).with_name('template_geometry_frame.py').read_text(); exec(_F[_F.index("class Frame:"):_F.index("\nout = {}\nfor s in STEMS")])
import tiktoken
enc = tiktoken.get_encoding('gpt2'); s = STEMS[0]; m_, d_ = 8, 10
D_ = pickle.load(open(RUN / f'robust_{s}.pkl', 'rb')); M, Rr = D_['meta'], D_['res']; K = np.array([x['kind'] for x in M]); want = np.array([x['want'] for x in M], object); wi = np.array([x['wording'] for x in M]); sp = np.array([x['split'] for x in M]); fr = np.array([x['frame'] for x in M]); g = np.array([x['content'] for x in M])
lab = np.array(['text' if k == 'text' else f'{f}:{k}' for k, f in zip(K, fr)]); dev, te = sp == 'dev', sp == 'test'
X = np.stack([coords(r['st'][0], r['roles'], m_) for r in Rr]); C0 = np.stack([r['st'][0]['c'].astype(np.float64) for r in Rr]); C0 /= np.linalg.norm(C0, axis=1, keepdims=True)
F_ = Frame([C0[dev]], wi[dev], fr[dev]); XF = np.c_[X, F_([C0])]; c = Clouds(XF[dev], lab[dev], want[dev], d_); rho = cvf(X[dev], [C0[dev]], lab[dev], want[dev], g[dev], wi[dev], fr[dev], d_)[0]
nms = names(m_) + ['connective position']; Y = ((XF - c.mu0) / c.sd) @ c.B; ids = {k: i for i, k in enumerate(c.cl)}
corr = np.array([[np.corrcoef(XF[dev][:, j], Y[dev][:, i])[0, 1] for j in range(XF.shape[1])] for i in range(d_)])
out = dict(stem=s, m=m_, d=d_, rho=rho, coordinates=nms, mu0=c.mu0.tolist(), sd=c.sd.tolist(), singular=c.sv[:d_].tolist(), classes={k: dict(template=c.tpl[k], centre=c.C[ids[k]].tolist(), reach=float(c.reach[k])) for k in c.cl},
           directions=[[(nms[j], float(corr[i, j])) for j in np.argsort(-abs(corr[i]))[:6]] for i in range(d_)], axis_of_position=dict(note='connective unit vector minus the midpoint of the two centroids, along the cond - disj axis'))
def decide(xf):
    y = ((xf - c.mu0) / c.sd) @ c.B; Dq = ((y - c.C) ** 2).sum(1); P = np.exp(-.5 * (Dq - Dq.min())); P /= P.sum(); rt = {t: float(P[[ids[k] for k in c.cl if c.tpl[k] == t]].sum()) for t in ('MP', 'MT', 'DS1', 'DS2')}
    b = int(Dq.argmin()); inside = bool(Dq[b] <= c.reach[c.cl[b]]); t = max(rt, key=rt.get); fire = t if rt[t] >= rho and inside else ''
    return dict(y=y.tolist(), nearest=[(c.cl[i], float(Dq[i])) for i in np.argsort(Dq)[:5]], rates=rt, inside_reach=inside, nearest_reach=float(c.reach[c.cl[b]]), fires=fire)
cap = lambda x: x[0].upper() + x[1:]; pq = lambda lead: [(lead, 'c'), (" {P}", 'a1'), (",", None), (" {Q}", 'a2'), (".", None)]
LAY = {'if-then': [("If", 'c'), (" {P}", 'a1'), (",", None), (" then", 'c'), (" {Q}", 'a2'), (".", None)], 'unless': pq("Unless"), 'either-or': [("Either", 'c'), (" {P}", 'a1'), (" or", 'c'), (" {Q}", 'a2'), (".", None)], 'when': pq("When")}
def tokens(it):
    segs = [(t.replace('{P}', it['P']).replace('{Q}', it['Q']), ro) for t, ro in LAY[it['wording']]] + [(" " + cap(it['p2']), 'p2'), (".", None), (" Therefore", 'th')]; tk = {ro: [] for ro in ('c', 'a1', 'a2', 'p2', 'th')}
    for t, ro in segs:
        if ro: tk[ro] += [enc.decode([e]) for e in enc.encode(t)]
    return sum((tk[ro] for ro in ('c', 'a1', 'a2', 'p2', 'th')), [])
work = {}
for i in np.flatnonzero(te & (np.array([x['P'] for x in M]) == 'the coffee is cold') & np.isin(wi, list(LAY)) & np.isin(K, ['MP', 'MT', 'AC', 'DA', 'near', 'MPsyn', 'U', 'swap', 'DS1', 'DS2', 'P', 'Q'])):
    it = M[i]; st = Rr[i]['st'][0]; tk = tokens(it); assert len(tk) == len(Rr[i]['roles']), (tk, Rr[i]['roles'])
    work[f"{it['wording']}:{it['kind']}"] = dict(text=it['text'], want=it['want'], tokens=tk, roles=Rr[i]['roles'], G=np.round(st['G'].astype(float), 3).tolist(), norm=np.round(st['norm'].astype(float), 2).tolist(), r=XF[i].tolist(), **decide(XF[i]))
    print(f"{it['wording']:9s} {it['kind']:5s} fires {work[f'{it[chr(119)+chr(111)+chr(114)+chr(100)+chr(105)+chr(110)+chr(103)]}:{it[chr(107)+chr(105)+chr(110)+chr(100)]}']['fires'] or '-':4s}", it['text'][:110], flush=True)
idx = {f"{M[i]['wording']}:{M[i]['kind']}": i for i in np.flatnonzero(te & (np.array([x['P'] for x in M]) == 'the coffee is cold') & np.isin(wi, list(LAY)))}
def llr(xf):
    y = ((xf - c.mu0) / c.sd) @ c.B; L = -.5 * ((y - c.C) ** 2).sum(1); P = np.exp(L - L.max()); P /= P.sum(); return float(np.log(P[[ids[k] for k in c.cl if c.tpl[k] == 'MP']].sum() + 1e-300))
base = XF[idx['if-then:MP']]; out['swaps'] = {}
for other in ('near', 'MPsyn', 'DA', 'U'):
    o = XF[idx[f'if-then:{other}']]; ch = []
    for j in range(len(base)):
        x = base.copy(); x[j] = o[j]; ch.append((nms[j], float(base[j]), float(o[j]), llr(x) - llr(base)))
    out['swaps'][other] = sorted(ch, key=lambda v: v[3])[:8]; print('swap from', other, [(a, round(e, 1)) for a, _, _, e in out['swaps'][other][:5]])
# without the position: refit and decide the same premises
c2 = Clouds(X[dev], lab[dev], want[dev], d_); rho2 = cv(X[dev], lab[dev], want[dev], g[dev], d_)[0]; out['without position'] = dict(rho=rho2, fires={k: str(c2.fire(X[[v]], rho2)[0]) for k, v in idx.items() if k in work})
out['work'] = work; (OUT / f'template_embedding_walkthrough_{s}.json').write_text(json.dumps(out, indent=1)); print('rho', rho, 'without position', out['without position'])
for i, l in enumerate(out['directions']): print(i + 1, round(c.sv[i], 1), [(n, round(x, 2)) for n, x in l[:4]])
