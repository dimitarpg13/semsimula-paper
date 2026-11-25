#!/usr/bin/env python3
"""Where do short logical arguments end up geometrically? A preliminary, unregistered look (2026-11-25, author's idea).

    python3 logic_figures.py OUT FOLDER [name=<stem>] [ckpt=<file>] [threads=8] [prefix=256]

Three valid argument forms and two invalid controls, each written on the same 40 contents:
  MP  modus ponens          If P, then Q. P. Therefore, Q.
  MT  modus tollens         If P, then Q. not Q. Therefore, not P.
  DS  disjunctive syllogism Either P or Q. not P. Therefore, Q.
  AC  affirming the consequent (invalid)  If P, then Q. Q. Therefore, P.
  DA  denying the antecedent (invalid)    If P, then Q. not P. Therefore, not Q.
Every argument follows the same fixed passage of OpenWebText validation text (`prefix` tokens and a blank line), so it
sits in the middle of a window, not at its start. Each argument has seven role nodes, the same in every form:
  c1 ("If" / "Either"), a1 (the first clause), c2 ("then" / "or"), a2 (the second clause), p2 (the second premise),
  th ("Therefore"), cn (the conclusion clause).
A node's state is the mean of its tokens' states. Recorded at every layer boundary: h0 (the embedding), h1 (after layer
0), h2 (after layer 1); and per layer the dominant well of the V_theta bank and the token's stiff coordinates in it.

Readings per model, form and state:
  - the instance figure: the seven node states, centred; its singular values, and the dimensions holding 90% and 99%;
  - the form's subspace: all 40 centred figures stacked; the dimensions holding 90%, 95%, 99% of their spread;
  - shape consistency: generalised Procrustes alignment of the 40 figures, each centred and scaled to unit size, then
    rotated (an orthogonal map) onto the mean; the mean shape and the residual, the mean squared distance to it; the same with the node labels permuted at random in
    every instance (a figure with no consistent shape), and each instance against the other forms' mean shapes;
  - the frozen figure: the mean shape in its own principal coordinates, saved;
  - the reaction: the loss on the conclusion clause, and on an alternative conclusion (MP: P; MT: P without the
    negation; DS: P; AC and DA: Q), the model's preference being the difference.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
NAME = OPTS.get('name', 'model'); PREFIX = int(OPTS.get('prefix', 256))
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))
OUT.mkdir(parents=True, exist_ok=True)

src = (DBG / 'copy_split.py').read_text(); src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT / 'harness'), 'probe', 'windows=1', f"threads={OPTS.get('threads', 8)}"]
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)
import verify_pm_switch as P
G = P.G
_build = P.build; got = {}
def build(*a, **k):
    r = _build(*a, **k); got['model'] = r[0]; return r
P.build = build
_, DESC = CS['ladder_model'](FOLDER, OPTS.get('ckpt'))
DESC = DESC.split(' (built tag')[0][-60:]
m = got['model']; m.eval()
import tiktoken
enc = tiktoken.get_encoding('gpt2')

# ---- the contents: (P, Q, not P, not Q), lower case, no final period ----
C = [
 ("the build fails", "the deployment aborts", "the build does not fail", "the deployment does not abort"),
 ("the service is overloaded", "latency exceeds 500 ms", "the service is not overloaded", "latency does not exceed 500 ms"),
 ("the cache is stale", "the database query is slow", "the cache is not stale", "the database query is not slow"),
 ("it rains", "the streets get wet", "it does not rain", "the streets do not get wet"),
 ("the battery is empty", "the phone shuts down", "the battery is not empty", "the phone does not shut down"),
 ("the bridge is closed", "the traffic is diverted", "the bridge is not closed", "the traffic is not diverted"),
 ("the store is open", "the lights are on", "the store is not open", "the lights are not on"),
 ("the patient has a fever", "the doctor orders a test", "the patient does not have a fever", "the doctor does not order a test"),
 ("the river floods", "the farm is damaged", "the river does not flood", "the farm is not damaged"),
 ("the password is wrong", "the login fails", "the password is not wrong", "the login does not fail"),
 ("the team wins", "the fans celebrate", "the team does not win", "the fans do not celebrate"),
 ("the price rises", "sales fall", "the price does not rise", "sales do not fall"),
 ("the alarm rings", "the guards respond", "the alarm does not ring", "the guards do not respond"),
 ("the engine overheats", "the car stops", "the engine does not overheat", "the car does not stop"),
 ("the bill passes", "the law changes", "the bill does not pass", "the law does not change"),
 ("the snow melts", "the lake rises", "the snow does not melt", "the lake does not rise"),
 ("the server crashes", "the website goes down", "the server does not crash", "the website does not go down"),
 ("the student studies", "the exam goes well", "the student does not study", "the exam does not go well"),
 ("the dog barks", "the baby wakes up", "the dog does not bark", "the baby does not wake up"),
 ("the meeting is cancelled", "the report is delayed", "the meeting is not cancelled", "the report is not delayed"),
 ("the pipe leaks", "the floor gets wet", "the pipe does not leak", "the floor does not get wet"),
 ("the flight is late", "we miss the connection", "the flight is not late", "we do not miss the connection"),
 ("the test passes", "the code is merged", "the test does not pass", "the code is not merged"),
 ("the market crashes", "investors panic", "the market does not crash", "investors do not panic"),
 ("the wind is strong", "the ferry is cancelled", "the wind is not strong", "the ferry is not cancelled"),
 ("the oven is hot", "the bread rises", "the oven is not hot", "the bread does not rise"),
 ("the vote is close", "a recount is held", "the vote is not close", "a recount is not held"),
 ("the disk is full", "the backup fails", "the disk is not full", "the backup does not fail"),
 ("the coach resigns", "the club hires a new manager", "the coach does not resign", "the club does not hire a new manager"),
 ("the road is icy", "the bus is late", "the road is not icy", "the bus is not late"),
 ("the plant gets water", "the plant grows", "the plant does not get water", "the plant does not grow"),
 ("the contract is signed", "the work begins", "the contract is not signed", "the work does not begin"),
 ("the signal is weak", "the call drops", "the signal is not weak", "the call does not drop"),
 ("the fire spreads", "the town is evacuated", "the fire does not spread", "the town is not evacuated"),
 ("the light is red", "the cars stop", "the light is not red", "the cars do not stop"),
 ("the loan is approved", "the house is bought", "the loan is not approved", "the house is not bought"),
 ("the sensor fails", "the machine halts", "the sensor does not fail", "the machine does not halt"),
 ("the actor is ill", "the show is postponed", "the actor is not ill", "the show is not postponed"),
 ("the glass falls", "the glass breaks", "the glass does not fall", "the glass does not break"),
 ("the payment clears", "the order ships", "the payment does not clear", "the order does not ship"),
]
cap = lambda s: s[0].upper() + s[1:]
ROLES = ['c1', 'a1', 'c2', 'a2', 'p2', 'th', 'cn']
FORMS = ('MP', 'MT', 'DS', 'AC', 'DA')     # the last two are the invalid controls


def segments(form, p, q, np_, nq):
    """(text, role) pieces; role None for punctuation. Pieces after the first start with a space where words do."""
    if form == 'MP':
        return [("If", 'c1'), (" " + p, 'a1'), (",", None), (" then", 'c2'), (" " + q, 'a2'), (".", None),
                (" " + cap(p), 'p2'), (".", None), (" Therefore", 'th'), (",", None), (" " + q, 'cn'), (".", None)]
    if form == 'MT':
        return [("If", 'c1'), (" " + p, 'a1'), (",", None), (" then", 'c2'), (" " + q, 'a2'), (".", None),
                (" " + cap(nq), 'p2'), (".", None), (" Therefore", 'th'), (",", None), (" " + np_, 'cn'), (".", None)]
    if form == 'AC':                                   # affirming the consequent (invalid): MP's words, roles swapped
        return [("If", 'c1'), (" " + p, 'a1'), (",", None), (" then", 'c2'), (" " + q, 'a2'), (".", None),
                (" " + cap(q), 'p2'), (".", None), (" Therefore", 'th'), (",", None), (" " + p, 'cn'), (".", None)]
    if form == 'DA':                                   # denying the antecedent (invalid): MT's words, roles swapped
        return [("If", 'c1'), (" " + p, 'a1'), (",", None), (" then", 'c2'), (" " + q, 'a2'), (".", None),
                (" " + cap(np_), 'p2'), (".", None), (" Therefore", 'th'), (",", None), (" " + nq, 'cn'), (".", None)]
    return [("Either", 'c1'), (" " + p, 'a1'), (" or", 'c2'), (" " + q, 'a2'), (".", None),
            (" " + cap(np_), 'p2'), (".", None), (" Therefore", 'th'), (",", None), (" " + q, 'cn'), (".", None)]


def alt_conclusion(form, p, q, np_, nq):
    return {'MP': p, 'MT': p, 'DS': p, 'AC': q, 'DA': q}[form]


val = np.load(G.VAL).astype(np.int64)
prefix = list(val[1_000_000:1_000_000 + PREFIX]) + enc.encode("\n\n")
items = []
for form in FORMS:
    for ci, (p, q, np_, nq) in enumerate(C):
        ids, role = list(prefix), [None] * len(prefix)
        for text, r in segments(form, p, q, np_, nq):
            t = enc.encode(text); ids += t; role += [r] * len(t)
        assert enc.decode(ids[len(prefix):]).startswith(("If", "Either")), enc.decode(ids[len(prefix):])
        cn = [i for i, r in enumerate(role) if r == 'cn']
        # the alternative conclusion: same context, the other clause in the conclusion slot
        alt = list(ids[:cn[0]]) + enc.encode(" " + alt_conclusion(form, p, q, np_, nq))
        items.append(dict(form=form, content=ci, ids=ids, role=role, cn=cn, alt=alt, alt_start=cn[0]))
print(f'{DESC}\n{len(items)} arguments; prefix {PREFIX} tokens; longest {max(len(i["ids"]) for i in items)}', flush=True)

# ---- hooks: the state at each layer boundary, and the bank's components at the layer input ----
VT = m.V_theta
bank = VT.bank.bank if hasattr(VT.bank, 'bank') else VT.bank
cur = {'layer': 0, 'lr': False}; rec = {}
orig_step = m._layer_step_langevin
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    cur.update(layer=int(layer_idx), lr=False)
    out = orig_step(h, h_prev, m_b, gamma, dt_, layer_idx)
    if cur.get('on'):
        rec.setdefault(('h', int(layer_idx) + 1), []).append(out[0].detach().float())
        if int(layer_idx) == 0:
            rec.setdefault(('h', 0), []).append(h.detach().float())
    return out
orig_lr = bank.harmonic_terms_lowrank
def lr(xi, h, *, comps=None):
    out = orig_lr(xi, h, comps=comps)
    if cur.get('on') and not cur['lr']:
        cur['lr'] = True
        with torch.no_grad():
            mu, a, w, B = (c_.detach().float() for c_ in (comps if comps is not None else bank._components(xi)))
            hh = h.detach().float()
            diff = hh.unsqueeze(-2) - mu                                     # (b, T, K, d)
            u = torch.einsum('btkd,btkdr->btkr', diff, B)
            q = (a * diff * diff).sum(-1) + (u * u).sum(-1)
            g = w * torch.exp(-0.5 * q); pi = g / g.sum(-1, keepdim=True).clamp_min(1e-30)
            ks = pi.argmax(-1)
            us = torch.gather(u, 2, ks[..., None, None].expand(*ks.shape, 1, u.shape[-1])).squeeze(2)
            rec.setdefault(('k', cur['layer']), []).append(ks)
            rec.setdefault(('u', cur['layer']), []).append(us)
            rec.setdefault(('r', cur['layer']), []).append(torch.gather(q, 2, ks[..., None]).squeeze(-1).clamp_min(0).sqrt())
    return out
m._layer_step_langevin = step; bank.harmonic_terms_lowrank = lr

t0 = time.time()
node = {}            # (level, form) -> (40, 7, d) node states
kst = {}; ust = {}   # dominant well and stiff coordinates of the nodes, per layer
react = {f: [] for f in FORMS}
for it in items:
    rec.clear(); cur['on'] = True
    x = torch.tensor([it['ids']])
    with torch.enable_grad():
        lo = m(x)[0].detach().float()
    cur['on'] = False
    # loss on the conclusion clause (predicting each of its tokens from the one before)
    tgt = torch.tensor(it['ids'])
    nll = torch.nn.functional.cross_entropy(lo[0, :-1], tgt[1:], reduction='none')
    l_val = float(nll[[i - 1 for i in it['cn']]].mean())
    xa = torch.tensor([it['alt']])
    with torch.enable_grad():
        la = m(xa)[0].detach().float()
    na = torch.nn.functional.cross_entropy(la[0, :-1], xa[0, 1:], reduction='none')
    l_alt = float(na[it['alt_start'] - 1:].mean())
    react[it['form']].append((l_val, l_alt))
    for lev in (0, 1, 2):
        H = rec[('h', lev)][0][0]                                            # (T, d)
        nodes = torch.stack([H[[i for i, r in enumerate(it['role']) if r == ro]].mean(0) for ro in ROLES])
        node.setdefault((lev, it['form']), []).append(nodes.numpy())
    for lay in (0, 1):
        K_ = rec[('k', lay)][0][0]; U_ = rec[('u', lay)][0][0]
        kst.setdefault((lay, it['form']), []).append([int(torch.mode(K_[[i for i, r in enumerate(it['role']) if r == ro]]).values) for ro in ROLES])
        ust.setdefault((lay, it['form']), []).append(torch.stack([U_[[i for i, r in enumerate(it['role']) if r == ro]].mean(0) for ro in ROLES]).numpy())
print(f'forward passes done ({time.time() - t0:.0f}s)', flush=True)
node = {k: np.stack(v) for k, v in node.items()}; ust = {k: np.stack(v) for k, v in ust.items()}; kst = {k: np.array(v) for k, v in kst.items()}


def dims(sv, fr):
    e = np.cumsum(sv ** 2) / (sv ** 2).sum()
    return int(np.searchsorted(e, fr) + 1)


def center(X):
    return X - X.mean(-2, keepdims=True)


def procrustes(A, B):
    """Rotate centred, unit-size B onto centred, unit-size A (orthogonal map, size kept); return the rotated B."""
    # B^T A has rank at most 7: work in the two figures' own row spaces (thin SVDs), which gives the same B R
    Ub, sb, Vbt = np.linalg.svd(B, full_matrices=False)
    Ua, sa, Vat = np.linalg.svd(A, full_matrices=False)
    Uc, sc, Vct = np.linalg.svd((sb[:, None] * (Ub.T @ Ua)) * sa[None, :])
    return (Ub * sb) @ Uc @ Vct @ Vat


def gpa(F, iters=20):
    """Generalised Procrustes: F (n, 7, d), centred. Returns the mean shape, aligned figures, residual share."""
    Z = np.stack([f / np.linalg.norm(f) for f in F])
    M = Z[0].copy()
    for _ in range(iters):
        Z = np.stack([procrustes(M, z) for z in Z]); M = Z.mean(0); M /= np.linalg.norm(M)
    res = np.mean([np.linalg.norm(z - M) ** 2 for z in Z])                # each z has (about) unit size
    return M, Z, float(res)


rng = np.random.default_rng(0)
lines = [f'LOGIC FIGURES: {NAME}  ({DESC})', f'   {len(C)} contents x 3 forms; prefix {PREFIX} tokens of OpenWebText validation text',
         '   nodes: ' + ', '.join(ROLES)]
J = {'model': NAME, 'desc': DESC, 'roles': ROLES, 'levels': {}, 'reaction': {}}
frozen = {}
lines.append('\nREACTION: loss on the conclusion clause (nats per token), the valid one against the alternative')
for f, v in react.items():
    v = np.array(v); d = v[:, 1] - v[:, 0]
    lines.append(f'   {f}: valid {v[:, 0].mean():.3f}  alternative {v[:, 1].mean():.3f}  preference {d.mean():+.3f} '
                 f'(se {d.std(ddof=1) / np.sqrt(len(d)):.3f}; valid preferred in {int((d > 0).sum())} of {len(d)})')
    J['reaction'][f] = dict(valid=float(v[:, 0].mean()), alt=float(v[:, 1].mean()), pref=float(d.mean()),
                            se=float(d.std(ddof=1) / np.sqrt(len(d))), n_pref=int((d > 0).sum()))
for lev in (0, 1, 2):
    lines.append(f'\nSTATE h{lev}' + (' (the embedding)' if lev == 0 else f' (after layer {lev - 1})'))
    J['levels'][lev] = {}
    means = {}
    for f in FORMS:
        F = center(node[(lev, f)])                                            # (40, 7, d)
        sv_inst = np.stack([np.linalg.svd(x, compute_uv=False) for x in F])  # (40, 7)
        d90 = [dims(s, 0.90) for s in sv_inst]; d99 = [dims(s, 0.99) for s in sv_inst]
        sv_form = np.linalg.svd(F.reshape(-1, F.shape[-1]), compute_uv=False)
        fd = {fr: dims(sv_form, fr) for fr in (0.90, 0.95, 0.99)}
        M, Z, res = gpa(F)
        Fp = np.stack([x[rng.permutation(7)] for x in F]); _, _, res_perm = gpa(Fp)
        means[f] = (M, res)
        # the frozen figure: the mean shape in its own principal coordinates
        Um, sm, Vm = np.linalg.svd(M, full_matrices=False)
        k = dims(sm, 0.99); frozen[f'h{lev}_{f}'] = Um[:, :k] * sm[:k]
        lines.append(f'   {f}: one figure spans {np.median(d90):.0f} dims at 90%, {np.median(d99):.0f} at 99% (median of 40; at most 6); '
                     f'all 40 span {fd[0.9]} / {fd[0.95]} / {fd[0.99]} dims at 90 / 95 / 99%; '
                     f'shape residual {res:.3f} (labels permuted {res_perm:.3f}); frozen figure in {k} dims')
        J['levels'][lev][f] = dict(inst_d90=float(np.median(d90)), inst_d99=float(np.median(d99)), form_dims=fd,
                                   res=res, res_perm=res_perm, frozen_dims=k, frozen_sv=sm.tolist())
    # each instance against the other forms' mean shapes
    cross = {}
    for f in FORMS:
        F = center(node[(lev, f)])
        for g_ in FORMS:
            Mg = means[g_][0]
            r_ = np.mean([np.linalg.norm(procrustes(Mg, x / np.linalg.norm(x)) - Mg) ** 2 for x in F])
            cross[f'{f}->{g_}'] = float(r_)
    lines.append('   residual of each form\'s figures against each mean shape (rows: figures; columns: mean shape of ' + ', '.join(FORMS) + '):')
    for f in FORMS:
        lines.append(f'      {f}: ' + '  '.join(f'{cross[f"{f}->{g_}"]:.3f}' for g_ in FORMS))
    J['levels'][lev]['cross'] = cross
    # node distances in the mean shape (unit size), for reading the figure
    for f in FORMS:
        M = means[f][0]; D = np.linalg.norm(M[:, None] - M[None], axis=-1)
        J['levels'][lev][f]['node_dist'] = D.tolist()
lines.append('\nWELLS: dominant well of each node, share of the 40 instances on the most common well (per layer)')
for lay in (0, 1):
    for f in FORMS:
        Kf = kst[(lay, f)]
        s = []
        for j, ro in enumerate(ROLES):
            vals, cnt = np.unique(Kf[:, j], return_counts=True)
            s.append(f'{ro} k{vals[cnt.argmax()]} {cnt.max() / len(Kf):.0%}')
        lines.append(f'   layer {lay} {f}: ' + ', '.join(s))
        U4 = center(ust[(lay, f)])                                            # (40, 7, r): the figure in the stiff coordinates
        _, _, res_u = gpa(U4); Up = np.stack([x[rng.permutation(7)] for x in U4]); _, _, res_up = gpa(Up)
        lines.append(f'      in the dominant well\'s {U4.shape[-1]} stiff coordinates: shape residual {res_u:.3f} (labels permuted {res_up:.3f})')
        J.setdefault('wells', {})[f'L{lay}_{f}'] = dict(res_u=res_u, res_u_perm=res_up, kstar=Kf.tolist())
lines.append(f'\n({time.time() - t0:.0f}s)')
print('\n'.join(lines))
(OUT / f'logic_figures_{NAME}.txt').write_text('\n'.join(lines) + '\n')
(OUT / f'logic_figures_{NAME}.json').write_text(json.dumps(J, indent=1))
np.savez_compressed(OUT / f'logic_figures_{NAME}.npz', **{f'nodes_h{lev}_{f}': node[(lev, f)] for lev in (0, 1, 2) for f in FORMS},
                    **{f'frozen_{k}': v for k, v in frozen.items()},
                    **{f'u_L{lay}_{f}': ust[(lay, f)] for lay in (0, 1) for f in FORMS})
