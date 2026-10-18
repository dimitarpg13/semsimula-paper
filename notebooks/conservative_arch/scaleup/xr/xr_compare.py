#!/usr/bin/env python3
"""
XR part A: the exchange field against the reverse channel, on the trained checkpoints
(protocol §5.25, pre-registered 2026-10-17 before any measurement).

    python3 xr_compare.py ARM [n_rem=100] [n_ker=20] [n_cap=20]      ARM in G3, G3p, G2, XRBF

  1. removal at inference, 2 x 2: field off (gate 0), read off (read scale 0), both, neither;
  2. the routes' kernels: the field's routing weights (head average) and the register path's
     A_ij = sum_k beta_ik (1 - sigma^(i)_k) alpha^(i)_kj, row-normalised, with distance, entropy,
     rank, previous-token / same-token / induction scores and the row overlap of the two kernels;
  3. each route's removal cost by target class (induction, repeated, novel; frequent vs rest);
  4. capacity: single-head and single-register removal costs, then keep the top k (and a random k).
G2 has no exchange field: only the read is measured there. XRBF (XR part B, the field-only model) has no
reverse channel: only the field is measured there (added 2026-10-18).
Outputs: xr_<ARM>_output.txt, xr_<ARM>.json, xr_<ARM>.npz (per-sequence statistics; git-ignored).
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ARM = sys.argv[1]
kw = dict(a.split('=') for a in sys.argv[2:] if '=' in a)
N_REM, N_KER, N_CAP = int(kw.get('n_rem', 100)), int(kw.get('n_ker', 20)), int(kw.get('n_cap', 20))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'sync'))
from sy_replay import Replayer, CACHE
torch.set_num_threads(max(1, torch.get_num_threads()))

lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
R = Replayer(ARM); m = R.model; m.eval()
HAS_EX = m.relax_field is not None
HAS_RC = m.reverse_ch is not None
rf, rc, cg = m.relax_field, m.reverse_ch, m.creation_gate_qkv
X = np.load(CACHE / 'sy_tokens.npz')['x']
NB = 1000
rng0 = np.random.default_rng(20261017)


def boot(v, seed=0):
    v = np.asarray(v, float); I = np.random.default_rng(seed).integers(0, len(v), (NB, len(v)))
    b = v[I].mean(1); return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fmt(c): return f'{c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]'


@torch.no_grad()
def read_beta(h, r, active):
    hi = rc.ln_h(h) if rc.pre_ln else h; ri = rc.ln_r(r) if rc.pre_ln else r
    q = rc.W_Q_rev(hi); k = rc.W_K_rev(ri)
    if rc.stable:
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        sc = torch.einsum('btk,btmk->btm', q, k) * rc.logit_scale.exp().clamp(max=rc.logit_scale_max)
    else:
        sc = torch.einsum('btk,btmk->btm', q, k) / (rc.d_k ** 0.5)
    sc = sc.masked_fill(~active, -1e9)
    return torch.softmax(sc, -1) * active.any(-1, keepdim=True).float()



# ---------------------------------------------------------------- switches
lam0 = m.relax_lambda.detach().clone() if HAS_EX else None
s0 = m.reverse_channel_scale.detach().clone() if HAS_RC else None
fl0 = rf.force_live if HAS_EX else None
rcf0 = rc.forward if HAS_RC else None


def set_state(field=True, read=True, heads=None, regs=None, renorm=False):
    """field/read on or off; heads: keep-mask over exchange heads; regs: keep-mask over registers."""
    with torch.no_grad():
        if HAS_EX:
            m.relax_lambda.copy_(lam0 if field else torch.zeros_like(lam0))
        if HAS_RC:
            m.reverse_channel_scale.copy_(s0 if read else torch.zeros_like(s0))
    rf.__dict__.pop('force_live', None) if HAS_EX else None
    rc.__dict__.pop('forward', None) if HAS_RC else None
    if HAS_EX and heads is not None:
        hm = torch.as_tensor(heads, dtype=torch.float32)
        def fl(h_in, xi_route, causal):
            B, T, d = h_in.shape
            alpha = rf._routing(xi_route, causal)
            vs = rf.W_v(h_in).view(B, T, rf.H, rf.d_v).transpose(1, 2)
            agg = torch.matmul(alpha, vs) * (rf.d_v ** -0.5) * hm.view(1, rf.H, 1, 1)
            return agg.transpose(1, 2).reshape(B, T, rf.H * rf.d_v) @ rf.W_uq.weight
        rf.force_live = fl
    if regs is not None:
        km = torch.as_tensor(regs, dtype=torch.bool)
        if renorm:                         # descriptive: mask the registers out, softmax renormalised over the rest
            def rcf(h, r, active):
                return rcf0(h, r, active & km.view(*([1] * (active.dim() - 1)), -1))
        else:                              # scored: drop their terms, the others keep their trained read weights
            def rcf(h, r, active):
                be = read_beta(h, r, active) * km.view(1, 1, -1).float()
                ri = rc.ln_r(r) if rc.pre_ln else r
                Q = torch.einsum('btm,btmd->btd', be, rc.W_V_rev(ri))
                if rc.stable:
                    Q = Q / Q.pow(2).mean(-1, keepdim=True).add(rc.out_norm_eps).sqrt()
                return Q
        rc.forward = rcf


def nll(xb):
    with torch.enable_grad():
        logits = R.forward(torch.from_numpy(xb[:, :-1]).long())
    y = torch.from_numpy(xb[:, 1:]).long()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction='none').view(y.shape).double().numpy()


def run_all(seqs, **state):
    set_state(**state)
    out = [nll(X[b:b + 2]) for b in range(0, len(seqs), 2)]
    set_state()
    return np.concatenate(out, 0)                                   # (n, 511)


# ---------------------------------------------------------------- checks
xb = X[:2]
a = nll(xb); set_state(field=False, read=False); set_state(); b = nll(xb)
say(f'XR: {ARM} (exchange field: {"yes" if HAS_EX else "no"}; reverse channel: {"yes" if HAS_RC else "no"}); '
    f'{N_REM} / {N_KER} / {N_CAP} sequences')
say(f'CHECK  switches restore the model: max |d loss| after off/on {float(np.abs(a - b).max()):.1e}')
if HAS_RC:
    set_state(read=False); c = nll(xb); set_state()
    say(f'CHECK  the read switch acts: mean |d loss| with the read off {float(np.abs(c - a).mean()):.3f}')
if HAS_EX:
    set_state(field=False); c = nll(xb); set_state()
    say(f'CHECK  the field switch acts: mean |d loss| with the field off {float(np.abs(c - a).mean()):.3f}')
    set_state(heads=[1] * rf.H); d = nll(xb); set_state()
    say(f'CHECK  the head-mask rewrite with all heads kept reproduces the model: max |d| {float(np.abs(d - a).max()):.1e}')
if HAS_RC:
    set_state(regs=[True] * cg.M); e = nll(xb); set_state()
    say(f'CHECK  the rewritten read with all registers kept reproduces the model: max |d| {float(np.abs(e - a).max()):.1e}')
    set_state(regs=[True] * cg.M, renorm=True); e2 = nll(xb); set_state()
    say(f'CHECK  the masked read with all registers kept reproduces the model: max |d| {float(np.abs(e2 - a).max()):.1e}')
t0 = time.time()

# ---------------------------------------------------------------- 1. removal 2 x 2 and 3. target classes
seqs = list(range(N_REM))
full = run_all(seqs)
offs = {'read': run_all(seqs, read=False)} if HAS_RC else {}
if HAS_EX:
    offs['field'] = run_all(seqs, field=False)
if HAS_EX and HAS_RC:
    offs['both'] = run_all(seqs, field=False, read=False)
say(f'   removal runs done, {(time.time() - t0) / 60:.1f} min')
Xs = X[:N_REM]
cur, nxt = Xs[:, :-1], Xs[:, 1:]
cls = np.zeros(cur.shape, dtype='<U9')
freq = np.bincount(X.reshape(-1), minlength=int(X.max()) + 1)
top = set(np.argsort(freq)[::-1][:max(1, int(0.01 * (freq > 0).sum()))].tolist())
isfreq = np.vectorize(lambda v: v in top)(nxt)
for i in range(len(Xs)):
    last_follow = {}                       # token -> the token that followed its last earlier occurrence
    seen = set()
    for t in range(cur.shape[1]):
        x, y = int(cur[i, t]), int(nxt[i, t])
        seen.add(x)                        # the text up to and including the current token
        if x in last_follow and last_follow[x] == y:
            cls[i, t] = 'induction'
        elif y in seen:
            cls[i, t] = 'repeated'
        else:
            cls[i, t] = 'novel'
        if t > 0:
            last_follow[int(cur[i, t - 1])] = x
res = {'removal': {}, 'classes': {}}
say('\n1. REMOVAL AT INFERENCE (cost in nats per token; 95% cluster-bootstrap intervals)')
say(f'   full model: mean loss {full.mean():.4f} (perplexity {np.exp(full.mean()):.2f})')
per_seq = {k: (v - full).mean(1) for k, v in offs.items()}
for k in offs:
    c = boot(per_seq[k]); res['removal'][k] = c
    say(f'   {k:5s} off: cost {fmt(c)}  (perplexity x{np.exp(c[0]):.2f})')
if HAS_EX and HAS_RC:
    ov = 1 - per_seq['both'] / (per_seq['field'] + per_seq['read'])
    tot = 1 - per_seq['both'].sum() / (per_seq['field'].sum() + per_seq['read'].sum())
    c = boot(ov); res['removal']['overlap'] = (float(tot), c[1], c[2])
    say(f'   overlap 1 - cost(both) / (cost(field) + cost(read)): {tot:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}] (per-sequence interval)')
say('\n3. COST BY TARGET CLASS (nats per token)')
masks = {'induction': cls == 'induction', 'repeated': cls == 'repeated', 'novel': cls == 'novel',
         'frequent': isfreq, 'rare': ~isfreq}
say('   class share: ' + ', '.join(f'{k} {v.mean():.3f}' for k, v in masks.items()))
for k, v in offs.items():
    row = {}
    for name, mk in masks.items():
        ps = np.array([(v[i] - full[i])[mk[i]].mean() if mk[i].any() else np.nan for i in range(len(v))])
        ps = ps[np.isfinite(ps)]; row[name] = boot(ps)
    res['classes'][k] = row
    say(f'   {k:5s} off: ' + '; '.join(f'{n} {row[n][0]:+.4f}' for n in masks))
    ind, nov = row['induction'][0], row['novel'][0]
    say(f'         induction / novel ratio: {ind / nov:.2f}' if nov > 0 else '         induction / novel ratio: n/a (novel cost <= 0)')
    res['classes'][k]['ratio_induction_novel'] = (ind / nov) if nov > 0 else None

# ---------------------------------------------------------------- 2. kernels
cap = {}
def capture_hooks():
    cap.clear()
    st0 = R._step
    def step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
        cap.setdefault('sal', {})[layer_idx] = sal.detach().clone(); cap['layer'] = layer_idx
        return st0(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
    m._fock_layer_step = step
    cf0 = rcf_ = None
    if HAS_RC:
        cf0 = cg.forward_prefix
        def cf(h, r):
            cap.setdefault('cg', {})[cap['layer']] = (h.detach().clone(), r.detach().clone()); return cf0(h, r)
        cg.forward_prefix = cf
        rcf_ = rc.forward
        def rcf(h, r, active):
            cap.setdefault('rc', {})[cap['layer']] = (h.detach().clone(), r.detach().clone(), active.detach().clone()); return rcf_(h, r, active)
        rc.forward = rcf
    if HAS_EX:
        fl_ = rf.force_live
        def fl(h_in, xi_route, causal):
            with torch.no_grad():
                cap.setdefault('ex', {})[cap['layer']] = rf._routing(xi_route.detach(), causal).clone()
            return fl_(h_in, xi_route, causal)
        rf.force_live = fl
    return st0, cf0, rcf_, fl_ if HAS_EX else None


def release(st0, cf0, rcf_, fl_):
    m._fock_layer_step = R._spy_step
    if HAS_RC: cg.__dict__.pop('forward_prefix', None); rc.__dict__.pop('forward', None)
    if HAS_EX: rf.__dict__.pop('force_live', None)


@torch.no_grad()
def write_alpha(h, r_state):
    """alpha^(i)_kj for j <= i: (B, M, T_i, T_j)."""
    B, T, d = h.shape
    Q = torch.einsum('btmd,mdk->btmk', r_state, cg.W_Q)
    if Q.shape[1] == 1: Q = Q.expand(B, T, cg.M, cg.d_k)
    K = torch.einsum('btd,mdk->btmk', h, cg.W_K) if cg.per_register_keys else cg.W_K(h).unsqueeze(2).expand(B, T, cg.M, cg.d_k)
    Q, K = cg._apply_qk_norm(Q, K)
    s = cg._scale_scores((Q * K).sum(-1).permute(0, 2, 1))           # (B, M, T)
    w = torch.exp(s - s.amax(-1, keepdim=True)).unsqueeze(2).expand(B, cg.M, T, T)
    tri = torch.tril(torch.ones(T, T, dtype=torch.bool))
    w = w * tri
    return w / w.sum(-1, keepdim=True).clamp_min(1e-30)


def kernel_stats(Kr, toks, T0=64):
    """Kr: (T, T) row-normalised, causal. Returns per-row-averaged statistics over rows t >= T0."""
    T = Kr.shape[0]; out = {}
    dist = (np.arange(T)[:, None] - np.arange(T)[None, :]).astype(float)
    rows = np.arange(T0, T)
    K = Kr[rows]; D = dist[rows]
    out['far64'] = float((K * (D > 64)).sum(1).mean())
    out['mean_dist'] = float((K * D).sum(1).mean())
    ent = -(K * np.log(np.clip(K, 1e-12, None))).sum(1)
    out['eff_sources'] = float(np.exp(ent).mean())
    out['prev_tok'] = float(K[np.arange(len(rows)), rows - 1].mean())
    same, ind = [], []
    for ii, t in enumerate(rows):
        occ = np.where(toks[:t] == toks[t])[0]
        same.append(K[ii, occ].sum() if len(occ) else np.nan)
        nxt = occ + 1; nxt = nxt[nxt < t]
        ind.append(K[ii, nxt].sum() if len(nxt) else np.nan)
    out['same_tok'] = float(np.nanmean(same)); out['induction'] = float(np.nanmean(ind))
    sv = np.linalg.svd(Kr, compute_uv=False) ** 2; sv = sv / sv.sum()
    out['eff_rank'] = float(np.exp(-(sv * np.log(np.clip(sv, 1e-30, None))).sum()))
    return out


def rownorm(K):
    s = K.sum(-1, keepdims=True); return np.where(s > 0, K / np.where(s > 0, s, 1), 0.0)


say('\n2. THE ROUTES\' KERNELS (rows t >= 64; head-averaged field routing; register path A = sum_k beta (1 - sigma) alpha)')
kst = {'field': {0: [], 1: []}, 'reg': {0: [], 1: []}, 'overlap_cos': {0: [], 1: []}, 'overlap_js': {0: [], 1: []}}
hooks = capture_hooks()
for b in range(0, N_KER, 2):
    xbk = X[b:b + 2]
    with torch.enable_grad():
        R.forward(torch.from_numpy(xbk[:, :-1]).long())
    for l in range(m.cfg.L):
        if HAS_RC:
            hcg, rst = cap['cg'][l]; hrc, rr, act = cap['rc'][l]; sal = cap['sal'][l]
            al = write_alpha(hcg, rst)                                       # (B, M, T, T)
            be = read_beta(hrc, rr, act)                                     # (B, T, M)
            sal_b = sal.expand(hcg.shape[0], hcg.shape[1], cg.M) if sal.shape[1] == 1 else sal
            A = torch.einsum('bim,bmij->bij', be * (1 - sal_b), al).numpy()   # (B, T, T)
        if HAS_EX:
            E = cap['ex'][l].mean(1).numpy()                             # (B, T, T)
        for j in range(xbk.shape[0]):
            toks = xbk[j, :-1]
            if HAS_RC:
                Ar = rownorm(A[j])
                if A[j].sum() > 0:
                    kst['reg'][l].append(kernel_stats(Ar, toks))
            if HAS_EX:
                Er = rownorm(E[j]); kst['field'][l].append(kernel_stats(Er, toks))
                if not HAS_RC or A[j].sum() == 0:
                    continue
                rows = np.arange(64, Ar.shape[0])
                a_, e_ = Ar[rows], Er[rows]
                cos = (a_ * e_).sum(1) / (np.linalg.norm(a_, axis=1) * np.linalg.norm(e_, axis=1) + 1e-12)
                mid = 0.5 * (a_ + e_)
                kl = lambda p, q: (p * (np.log(np.clip(p, 1e-12, None)) - np.log(np.clip(q, 1e-12, None)))).sum(1)
                js = 0.5 * kl(a_, mid) + 0.5 * kl(e_, mid)
                kst['overlap_cos'][l].append(float(cos.mean())); kst['overlap_js'][l].append(float(js.mean()))
release(*hooks)
res['kernels'] = {}
for l in range(m.cfg.L):
    for route in (('reg',) if HAS_RC else ()) + (('field',) if HAS_EX else ()):
        rows = kst[route][l]
        if not rows:
            say(f'   layer {l} {"register path" if route == "reg" else "exchange field":15s}: no token-to-token transfer '
                f'(the write is a no-op: salience 1 keeps the learned embeddings)')
            res['kernels'][f'{route}/L{l}'] = None
            continue
        agg = {k: boot([r[k] for r in rows]) for k in rows[0]}
        res['kernels'][f'{route}/L{l}'] = agg
        say(f'   layer {l} {"register path" if route == "reg" else "exchange field":15s}: '
            + '; '.join(f'{k} {v[0]:.3f}' for k, v in agg.items()))
    if HAS_EX and kst['overlap_cos'][l]:
        c, j = boot(kst['overlap_cos'][l]), boot(kst['overlap_js'][l])
        res['kernels'][f'overlap/L{l}'] = {'cos': c, 'js': j}
        say(f'   layer {l} overlap of the two kernels: row cosine {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]; Jensen-Shannon {j[0]:.3f} nats')

# ---------------------------------------------------------------- 4. capacity
say('\n4. CAPACITY (20 sequences; retained value = (cost removed - cost keeping k) / cost removed)')
cseq = list(range(N_CAP))
fullc = run_all(cseq).mean()
res['capacity'] = {}
def cost(**st): return run_all(cseq, **st).mean() - fullc
routes = ([('read', cg.M, [1, 2, 4, 8, 16, 32])] if HAS_RC else []) + ([('field', rf.H, [1, 2, 4, 6, 8])] if HAS_EX else [])
for route, n, ks in routes:
    off = cost(read=False) if route == 'read' else cost(field=False)
    single = []
    for u in range(n):
        keep = [True] * n; keep[u] = False
        single.append(cost(regs=keep) if route == 'read' else cost(heads=[float(x) for x in keep]))
    order = list(np.argsort(single)[::-1])                                   # most important first
    rnd = list(rng0.permutation(n))
    rec = {'off': off, 'single': single, 'order': [int(o) for o in order], 'top': {}, 'random': {}}
    for k in ks:
        for name, ordr in (('top', order), ('random', rnd)):
            keep = [False] * n
            for u in ordr[:k]: keep[u] = True
            ck = cost(regs=keep) if route == 'read' else cost(heads=[float(x) for x in keep])
            rec[name][k] = (off - ck) / off if off > 0 else float('nan')
    if route == 'read':
        rec['top_renorm'] = {}
        for k in ks:
            keep = [False] * n
            for u in order[:k]: keep[u] = True
            rec['top_renorm'][k] = (off - cost(regs=keep, renorm=True)) / off if off > 0 else float('nan')
    res['capacity'][route] = rec
    say(f'   {route}: cost removed {off:+.4f}; single-unit removal costs (largest 4) '
        + ', '.join(f'{single[u]:+.4f}' for u in order[:4]))
    say(f'          retained value keeping the top k:   ' + ', '.join(f'k={k} {rec["top"][k]:.2f}' for k in ks))
    say(f'          retained value keeping a random k: ' + ', '.join(f'k={k} {rec["random"][k]:.2f}' for k in ks))
    if route == 'read':
        say(f'          (descriptive) top k, read renormalised over the kept registers: ' + ', '.join(f'k={k} {rec["top_renorm"][k]:.2f}' for k in ks))
say(f'\n   total {(time.time() - t0) / 60:.1f} min')
(HERE / f'xr_{ARM}_output.txt').write_text('\n'.join(lines) + '\n')
(HERE / f'xr_{ARM}.json').write_text(json.dumps(res, indent=1, default=float))
np.savez(HERE / f'xr_{ARM}.npz', full=full, cls=cls, isfreq=isfreq, **{f'off_{k}': v for k, v in offs.items()})
