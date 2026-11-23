#!/usr/bin/env python3
"""Probe A of the mean-field programme: can a content-gated field stand in for the trained pair force V_phi?
(2026-11-22; Mean_Field_PARFLM_Programme.md, §5, calls A0-A5.)

    python3 mean_field_probe_a.py OUT FOLDER [ckpt=<file>] [val=<tokens.npy>] [windows=64] [threads=8]
                                  [name=<stem>] [Ms=1,2,4,8,16,32] [variants=ema,uniform] [rss_gb=40]

The field, per layer l and query token t, replaces the routed sources by M component means of the earlier states:

    f_t = -lambda_l s_l grad_{h_t} sum_m pi_{m,t} V_phi(h_t, mu_{m,t}),
    pi_{m,t} = sum_{s<t} W_ts r_{s,m},   mu_{m,t} = sum_{s<t} W_ts r_{s,m} h_s / pi_{m,t},
    r_{s,m} = softmax_m(-|h_s - c_m|^2 / tau),

with prototypes c_m from k-means on the layer's input states over all windows (a probe of representability, not
of training), tau set so that the median effective number of components of r is 1.5, W the strictly causal
average with decay 0.995 ('ema') or the uniform causal mean ('uniform'), s_l the model's trained per-layer scale,
and lambda_l one scalar per layer fitted by least squares to the model's own force. The heads V_phi are the
model's (value aligner, type gate, inverse distance after LayerNorm), evaluated at the component means (the
monopole approximation); the sources are detached, so the force is the gradient of a potential in h_t.

A control variant, 'global' (added 2026-11-22 after the first smoke test), ignores the context: the sources are the
k-means centres themselves at their overall weights, the same for every token. If it scores as well as the causal
field, the pair force acts mostly as a function of the token's own state, not of its context.
The report gives the paired difference between the causal field's cost and this control's (the context gap).

Readings, per layer, M and variant: the cosine between the field force and the model's force per token, and the
relative error after lambda_l (medians and 10th-90th percentiles). Then the substitution loss: the field force in
every layer, against the trained model and against the pair force off, on the same windows, with
rho_M = 1 - c_field / c_off (c = paired cost in nats, all tokens). Costs are also split by copy-split category
(first occurrence: the target does not occur earlier in the window; repeat; induction: the pair (current token,
target) occurred earlier) and by distance to the last occurrence.

Windows: `windows` of 512 tokens evenly spaced over the validation set (copy_split.windows), so the categories
exist on every corpus. Checks before any reading: the closed-form force gives the same loss as the autograd force;
the re-implemented heads reproduce the model's closed-form force on its own top-16 sources (router term made
negligible); the field at t does not depend on states after t. Memory: one batch of 4 windows at a time; the
script stops if its peak resident memory passes rss_gb.
Writes OUT/mean_field_probe_a_<name>.txt and .json.
"""
import json, resource, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))
W_, T, BS = int(OPTS.get('windows', 64)), 512, 4
MS = [int(x) for x in OPTS.get('Ms', '1,2,4,8,16,32').split(',')]
VARIANTS = OPTS.get('variants', 'ema,uniform,global').split(',')
RSS_GB = float(OPTS.get('rss_gb', 40))
t0 = time.time()


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 30 if sys.platform == 'darwin' else r / 2 ** 20


def guard(where):
    if rss_gb() > RSS_GB:
        raise SystemExit(f'peak resident memory {rss_gb():.1f} GB passed the limit {RSS_GB} GB at {where}')


# ---- the model, built from its folder tag exactly as recall_pathway_probe.py does
src = (DBG / 'copy_split.py').read_text()
src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', 'windows=1', f"threads={OPTS.get('threads', 8)}"] \
    + ([f"val={OPTS['val']}"] if 'val' in OPTS else [])
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)
import verify_pm_switch as P
_build = P.build; got = {}
def build(*a, **k):
    r = _build(*a, **k); got['model'] = r[0]; return r
P.build = build
_, desc = CS['ladder_model'](FOLDER, OPTS.get('ckpt'))
m = got['model']; m.eval()
m.cfg.vphi_force_checkpoint = False
assert getattr(m, 'V_phi', None) is not None, 'the model has no pair force'
print('model:', desc, flush=True)

v = np.load(CS['VAL']).astype(np.int64)
starts = np.linspace(0, len(v) - (T + 1), W_).astype(np.int64)
XY = np.stack([v[s:s + T + 1] for s in starts]); X, Y = XY[:, :T], XY[:, 1:]


def categories(x, y):
    """copy_split.categories without the tokenizer: 0 first occurrence, 1 repeat, 2 induction; distance."""
    kind = np.zeros(x.shape, np.int8); dist = np.zeros(x.shape, np.int32)
    for w in range(x.shape[0]):
        last, pairs = {}, set()
        for i in range(x.shape[1]):
            last[x[w, i]] = i
            if i:
                pairs.add((x[w, i - 1], x[w, i]))
            t = y[w, i]
            if t in last:
                dist[w, i] = i + 1 - last[t]
                kind[w, i] = 2 if (x[w, i], t) in pairs else 1
    return kind, dist


KIND, DIST = categories(X, Y)
GROUPS = {'all': np.ones_like(KIND, bool), 'first occurrence': KIND == 0, 'repeat (not induction)': KIND == 1,
          'induction': KIND == 2, 'recall, distance 1-16': (KIND > 0) & (DIST <= 16),
          'recall, distance 17-256': (KIND > 0) & (DIST > 16) & (DIST <= 256)}

# ---- hooks: record or replace the pair force at each layer
heads = list(m.V_phi.heads) if hasattr(m.V_phi, 'heads') else [m.V_phi]
orig_pf, orig_pp = m._pair_force_analytic, m._pair_potential
st = {'mode': 'base', 'rec': None, 'calls': 0, 'field': None}


def pf(h_in, layer_idx, *a, **k):
    st['calls'] += 1
    if st['mode'] == 'off':
        return torch.zeros_like(h_in)
    if st['mode'] == 'field':
        return st['field'](h_in, int(layer_idx))
    f = orig_pf(h_in, layer_idx, *a, **k)
    if st['rec'] is not None:
        st['rec'].append((int(layer_idx), h_in.detach().clone(), f.detach().clone()))
    return f
m._pair_force_analytic = pf


def ce_of(xb, yb):
    with torch.enable_grad():
        lg = m(torch.from_numpy(xb))[0].detach()
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), torch.from_numpy(yb).reshape(-1),
                           reduction='none').view(len(xb), -1)


checks = []
# check 1: the closed-form force gives the model's autograd loss
m.cfg.vphi_analytic_force = False
st['calls'] = 0
ce_auto = ce_of(X[:BS], Y[:BS])
assert st['calls'] == 0
m.cfg.vphi_analytic_force = True
ce_closed = ce_of(X[:BS], Y[:BS])
n_layers = st['calls']
assert n_layers >= 1, 'the closed-form pair force is not on the forward path of this model'
checks.append(f'closed-form against autograd pair force, first batch: max |d loss| {float((ce_closed - ce_auto).abs().max()):.2e} '
              f'({n_layers} pair-force evaluations per forward)')


def v_heads(h_q, srcs):
    """sum over heads of V_phi(h_t, src) = -C Theta Phi / r, exactly as _pair_force_analytic builds V_tot.
    h_q (B, T, d), srcs (B, T, M, d) -> (B, T, M)."""
    d = h_q.shape[-1]
    if heads[0].ln_before_distance:
        u_q = F.layer_norm(h_q, (d,)); u_s = F.layer_norm(srcs, (d,))
    V = 0.0
    for hd in heads:
        diff_l = hd.W_l(h_q).unsqueeze(2) - hd.W_l(srcs)
        D = (diff_l * diff_l).sum(-1)
        lin1, lin2 = hd.phi_c_net[0], hd.phi_c_net[2]
        c = F.softplus(lin2(F.gelu(lin1(D.unsqueeze(-1)))).squeeze(-1))
        Phi = torch.exp(-c * D)
        th_q = hd.W_theta(h_q); th_s = hd.W_theta(srcs)
        hid = (hd.theta_w_q(th_q) + hd.theta_w_d(th_q) + hd.theta_b1).unsqueeze(2) + (hd.theta_w_s(th_s) - hd.theta_w_d(th_s))
        Theta = torch.tanh(hd.theta_w2(F.gelu(hid)).squeeze(-1))
        diff_u = (u_q.unsqueeze(2) - u_s) if hd.ln_before_distance else (h_q.unsqueeze(2) - srcs)
        r = torch.sqrt((diff_u * diff_u).sum(-1) + hd.eps2)
        V = V + (-hd.C) * Theta * Phi / r
    return V


def s_layer(l):
    s = m.per_layer_scale(l)
    return 1.0 if s is None else s.detach()


def force_of(h_in, l, srcs, weights):
    """-grad_{h_t} s_l sum_j weights_tj V_phi(h_t, srcs_tj), sources detached."""
    with torch.enable_grad():
        h_q = h_in.detach().requires_grad_(True)
        U = (s_layer(l) * weights * v_heads(h_q, srcs.detach())).sum()
        g, = torch.autograd.grad(U, h_q)
    return -g


# check 2: the re-implemented heads reproduce the model's pair force on its own top-16 sources
rec_idx = {}
orig_topk = m._sparse_topk_indices
def topk(*a, **k):
    r = orig_topk(*a, **k); rec_idx['idx'], rec_idx['m'] = r[0].detach(), r[1].detach(); return r
m._sparse_topk_indices = topk
st['rec'] = []
ce_of(X[:2], Y[:2])
l0, h0, _ = st['rec'][0]; st['rec'] = None
tau_save = m._gumbel_tau.detach().clone()
m._gumbel_tau.fill_(1e12)                          # router term ~ 1/tau: negligible; the hard top-k is unchanged
f_model = orig_pf(h0, l0)
m._sparse_topk_indices = orig_topk
idx, mg = rec_idx['idx'], rec_idx['m']
B0, _, d = h0.shape; kk = idx.shape[-1]
h_src_g = h0.gather(1, idx.reshape(B0, T * kk, 1).expand(-1, -1, d)).view(B0, T, kk, d)
f_mine = force_of(h0, l0, h_src_g, mg)
m._gumbel_tau.copy_(tau_save)
rel = float((f_mine - f_model).norm() / f_model.norm())
checks.append(f'heads re-implemented against the closed-form force on its own top-{kk} sources (layer {l0}, router term off): '
              f'relative difference {rel:.2e}')
assert rel < 1e-3, rel


# ---- the base pass: record each layer's input states and the model's force; the trained and pair-force-off losses
H = {}; Fm = {}; ce_base = np.zeros((W_, T)); ce_off = np.zeros((W_, T))
for i in range(0, W_, BS):
    st['mode'] = 'base'; st['rec'] = []
    ce_base[i:i + BS] = ce_of(X[i:i + BS], Y[i:i + BS]).numpy()
    for l, h, f in st['rec']:
        H.setdefault(l, []).append(h); Fm.setdefault(l, []).append(f)
    st['rec'] = None; st['mode'] = 'off'
    ce_off[i:i + BS] = ce_of(X[i:i + BS], Y[i:i + BS]).numpy()
    guard(f'base pass, windows {i + BS}')
H = {l: torch.cat(x) for l, x in H.items()}; Fm = {l: torch.cat(x) for l, x in Fm.items()}
LAYERS = sorted(H)
print(f'base pass done ({time.time() - t0:.0f} s, peak {rss_gb():.1f} GB); layers {LAYERS}', flush=True)


def causal_weights(variant):
    t_ = torch.arange(T, dtype=torch.float32)
    if variant == 'ema':
        Wt = 0.995 ** (t_[:, None] - 1 - t_[None, :]).clamp_min(0)
    else:
        Wt = torch.ones(T, T)
    Wt = Wt * (t_[None, :] < t_[:, None])          # s < t only: no token in its own field
    return Wt / Wt.sum(-1, keepdim=True).clamp_min(1e-12)


WTS = {vr: causal_weights(vr) for vr in VARIANTS if vr != 'global'}


def kmeans(x, M, iters=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randint(len(x), (1,), generator=g)]
    for _ in range(1, M):                          # k-means++ seeding
        d2 = torch.cdist(x, c).min(1).values ** 2
        c = torch.cat([c, x[torch.multinomial(d2 / d2.sum(), 1, generator=g)]])
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(M):
            sel = a == j
            if sel.any():
                c[j] = x[sel].mean(0)
    return c


def tau_for(x, c, target=1.5):
    d2 = torch.cdist(x, c) ** 2
    if c.shape[0] == 1:
        return 1.0
    lo, hi = np.log(float(d2.median()) * 1e-6), np.log(float(d2.median()) * 1e6)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        r = torch.softmax(-d2 / np.exp(mid), -1)
        neff = float((1.0 / (r * r).sum(-1)).median())
        lo, hi = (mid, hi) if neff < target else (lo, mid)
    return float(np.exp(0.5 * (lo + hi)))


PROTO = {}
for l in LAYERS:
    x = H[l].reshape(-1, H[l].shape[-1])
    sub = x[torch.randperm(len(x), generator=torch.Generator().manual_seed(1))[:16384]]
    for M_ in MS:
        c = kmeans(x, M_)
        tau = tau_for(sub, c)
        frac = torch.softmax(-(torch.cdist(sub, c) ** 2) / tau, -1).mean(0)   # the components' overall weights
        PROTO[(l, M_)] = (c, tau, frac)
print(f'prototypes done ({time.time() - t0:.0f} s)', flush=True)


def field_force(h_in, l, M_, variant):
    """the field's force before lambda_l; h_in (B, T, d)"""
    c, tau, frac = PROTO[(l, M_)]
    h = h_in.detach()
    if variant == 'global':                        # control: no context at all, the centres at their overall weights
        B_, T_ = h.shape[:2]
        return force_of(h, l, c.expand(B_, T_, -1, -1), frac.expand(B_, T_, -1))
    r = torch.softmax(-(torch.cdist(h, c.unsqueeze(0).expand(len(h), -1, -1)) ** 2) / tau, -1)   # (B, T, M)
    Wt = WTS[variant]
    pi = torch.einsum('ts,bsm->btm', Wt, r)
    num = torch.einsum('ts,bsm,bsd->btmd', Wt, r, h)
    mu = num / pi.clamp_min(1e-12).unsqueeze(-1)
    return force_of(h, l, mu, pi)


# check 3: the field at t does not depend on states after t
hp = H[LAYERS[0]][:1].clone(); hq = hp.clone(); hq[:, 300:] += torch.randn_like(hq[:, 300:])
fa = field_force(hp, LAYERS[0], MS[-1], VARIANTS[0]); fb = field_force(hq, LAYERS[0], MS[-1], VARIANTS[0])
checks.append(f'causality: field force at positions 0-299 after perturbing positions 300-511: max |d f| '
              f'{float((fa[:, :300] - fb[:, :300]).abs().max()):.2e}')


def pct(a):
    return [float(np.percentile(a, q)) for q in (10, 50, 90)]


# ---- force fidelity and lambda_l on the recorded states
FID = {}; LAM = {}
for vr in VARIANTS:
    for M_ in MS:
        for l in LAYERS:
            ff = torch.cat([field_force(H[l][i:i + BS], l, M_, vr) for i in range(0, W_, BS)])
            fm = Fm[l]
            lam = float((ff * fm).sum() / (ff * ff).sum().clamp_min(1e-30))
            cos = F.cosine_similarity(ff.reshape(-1, ff.shape[-1]), fm.reshape(-1, fm.shape[-1]), dim=-1).numpy()
            rel = ((lam * ff - fm).norm(dim=-1) / fm.norm(dim=-1).clamp_min(1e-30)).reshape(-1).numpy()
            keep = fm.norm(dim=-1).reshape(-1).numpy() > 0
            LAM[(vr, M_, l)] = lam
            FID[(vr, M_, l)] = {'lambda': lam, 'cos_p10_p50_p90': pct(cos[keep]), 'relerr_p10_p50_p90': pct(rel[keep])}
            guard(f'fidelity {vr} M={M_} layer {l}')
print(f'fidelity done ({time.time() - t0:.0f} s, peak {rss_gb():.1f} GB)', flush=True)


# ---- substitution: the field force in every layer
def win_mean(ce, mask):
    out = []
    for w in range(W_):
        sel = mask[w]
        out.append(float(ce[w, sel].mean()) if sel.any() else np.nan)
    return np.array(out)


def summary(d_ce):
    res = {}
    for g, mk in GROUPS.items():
        x = win_mean(d_ce, mk); x = x[np.isfinite(x)]
        res[g] = [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))]
    return res


c_off = summary(ce_off - ce_base)
SUB = {}
for vr in VARIANTS:
    for M_ in MS:
        st['mode'] = 'field'
        st['field'] = lambda h_in, l, M_=M_, vr=vr: LAM[(vr, M_, l)] * field_force(h_in, l, M_, vr)
        ce_f = np.zeros((W_, T))
        for i in range(0, W_, BS):
            ce_f[i:i + BS] = ce_of(X[i:i + BS], Y[i:i + BS]).numpy()
            guard(f'substitution {vr} M={M_}')
        c_f = summary(ce_f - ce_base)
        dpaired = win_mean(ce_f - ce_off, GROUPS['all'])
        rho = 1.0 - c_f['all'][0] / c_off['all'][0]
        share = {g: float((ce_f - ce_base)[mk].sum() / (ce_f - ce_base).sum()) for g, mk in GROUPS.items() if g != 'all'}
        SUB[(vr, M_)] = {'cost': c_f, 'rho': rho, 'field_minus_off': [float(dpaired.mean()), float(dpaired.std(ddof=1) / np.sqrt(W_))],
                         'cost_share': share, 'win_all': win_mean(ce_f - ce_base, GROUPS['all']).tolist()}
        print(f'  substitution {vr:7s} M={M_:2d}: cost {c_f["all"][0]:+.4f} ({c_f["all"][1]:.4f}) nats, rho {rho:.3f} '
              f'[{time.time() - t0:.0f} s, peak {rss_gb():.1f} GB]', flush=True)
st['mode'] = 'base'

# ---- report
token_share = {g: float(mk.mean()) for g, mk in GROUPS.items() if g != 'all'}
stem = OPTS.get('name', FOLDER[-40:])
L = [f'Mean-field Probe A: {FOLDER}', f'model {desc}',
     f'{W_} windows x {T} evenly spaced over {Path(CS["VAL"]).name}; M in {MS}; variants {VARIANTS}; '
     f'{len(LAYERS)} layers; token shares {json.dumps({k: round(v, 3) for k, v in token_share.items()})}', '',
     'checks:'] + ['  ' + c for c in checks] + ['',
     f'pair force off: cost {c_off["all"][0]:+.4f} ({c_off["all"][1]:.4f}) nats per token (all), '
     + ', '.join(f'{g} {c_off[g][0]:+.4f}' for g in GROUPS if g != 'all'), '',
     'force fidelity per layer: cosine (p10 / median / p90), relative error after lambda (median), lambda']
for vr in VARIANTS:
    for M_ in MS:
        row = [f'  {vr:7s} M={M_:2d}']
        for l in LAYERS:
            f_ = FID[(vr, M_, l)]
            row.append(f'L{l}: cos {f_["cos_p10_p50_p90"][0]:+.2f}/{f_["cos_p10_p50_p90"][1]:+.3f}/{f_["cos_p10_p50_p90"][2]:+.2f} '
                       f'err {f_["relerr_p10_p50_p90"][1]:.2f} lam {f_["lambda"]:.3g}')
        L.append('  '.join(row))
L += ['', 'substitution: the field force (times lambda) in every layer; cost against the trained model, nats per token '
      '(se over windows); rho = 1 - cost / cost of the pair force off']
hdr = ['all', 'first occurrence', 'repeat (not induction)', 'induction', 'recall, distance 17-256']
L.append(f'  {"variant":7s} {"M":>3s}  {"rho":>6s}  ' + '  '.join(f'{h[:22]:>22s}' for h in hdr))
for vr in VARIANTS:
    for M_ in MS:
        s_ = SUB[(vr, M_)]
        L.append(f'  {vr:7s} {M_:3d}  {s_["rho"]:6.3f}  ' + '  '.join(f'{s_["cost"][h][0]:+.4f} ({s_["cost"][h][1]:.4f})'.rjust(22) for h in hdr))
L += ['', 'share of the substitution cost against share of the tokens (A4 reads the induction row)']
for vr in VARIANTS:
    s8 = SUB.get((vr, 8))
    if s8:
        L.append(f'  {vr} M=8: ' + '; '.join(f'{g}: cost {s8["cost_share"][g]:.2f} / tokens {token_share[g]:.2f}'
                                          for g in ('first occurrence', 'repeat (not induction)', 'induction')))
if 'global' in VARIANTS and len(VARIANTS) > 1:   # context gap: causal field against the context-free control, paired
    L += ['', 'context gap: cost of the causal field minus cost of the context-free control, paired over windows '
          '(negative: the context helps)']
    for vr in [v for v in VARIANTS if v != 'global']:
        L.append(f'  {vr:7s} ' + '  '.join(
            f'M={M_}: {d.mean():+.4f} ({d.std(ddof=1) / np.sqrt(len(d)):.4f})' for M_ in MS
            for d in [np.array(SUB[(vr, M_)]['win_all']) - np.array(SUB[('global', M_)]['win_all'])]))
L += ['', f'time {time.time() - t0:.0f} s; peak resident memory {rss_gb():.1f} GB']
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f'mean_field_probe_a_{stem}.txt').write_text('\n'.join(L) + '\n')
(OUT / f'mean_field_probe_a_{stem}.json').write_text(json.dumps({
    'folder': FOLDER, 'model': desc, 'windows': W_, 'Ms': MS, 'variants': VARIANTS, 'layers': LAYERS, 'checks': checks,
    'token_share': token_share, 'off': c_off,
    'fidelity': {f'{vr}|{M_}|{l}': v_ for (vr, M_, l), v_ in FID.items()},
    'substitution': {f'{vr}|{M_}': v_ for (vr, M_), v_ in SUB.items()},
    'tau': {f'{l}|{M_}': PROTO[(l, M_)][1] for (l, M_) in PROTO}, 'peak_rss_gb': rss_gb()}, indent=1))
print('\n'.join(L))
