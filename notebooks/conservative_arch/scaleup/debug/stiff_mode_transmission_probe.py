#!/usr/bin/env python3
"""Does a difference along the stiff directions survive the layer step and reach the prediction? (2026-11-16,
protocol p-bank-memory-conjecture: the measurement that does not go through the loss.)

    python3 stiff_mode_transmission_probe.py OUT FOLDER [ckpt=<file>] [layers=0,1] [modes=4] [rand=4] [eps=2.5e-3] [pos=96,288] [fut=64]
                                             [windows=16] [threads=8] [val=<tokens.npy>] [name=<stem>] [check=1] [double=1]
                                             [recall=1 slots=2 maxfreq=8 offset=1]   (code: read at the positions that recall the moved token)

Per layer L and position p (one token per window moves, so the readings are not mixed by the context channels),
the input state of token p is moved by eps along a unit direction, in position only: h and h_prev move together,
so the velocity (h - h_prev) / dt is unchanged. The directions are token p's own stiffest retained modes at that
layer (columns of U from lowrank_modes at the unperturbed state, the `modes` largest stiffnesses), or `rand` random
unit directions orthogonal to all of U (the complement, fixed seed). Readings, as finite differences over eps
against the unperturbed run, root mean square over windows, positions and directions:
  - local, token p after the layer step: its position, the part on its U, its velocity (h_new - h_prev_out) / dt;
  - now:   the change of token p's logits, centred over the vocabulary (softmax ignores a common shift);
  - next64, after64: the same over the next 64 tokens and the 64 after those (what later predictions keep);
  - before: the tokens before p, which must not change (causality check).
The ratios stiff over complement say how well a stiff-direction difference is kept relative to a soft one,
within one model. check=1 repeats the first stiff and first random direction at eps / 2 (linearity).
Writes OUT/transmission_<stem>.txt and .json.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))

# copy_split's own Cell-0-from-tag build and validation windows, without its command dispatch (as vtheta_well_probe)
src = (DBG / 'copy_split.py').read_text()
src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', f"windows={OPTS.get('windows', 16)}", f"threads={OPTS.get('threads', 8)}"] \
    + ([f"val={OPTS['val']}"] if 'val' in OPTS else [])
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)

import verify_pm_switch as P
_build = P.build; got = {}
def build(*a, **k):
    r = _build(*a, **k); got['model'] = r[0]; return r
P.build = build
_, desc = CS['ladder_model'](FOLDER, OPTS.get('ckpt'))
model = got['model']
print('model:', desc, flush=True)
if OPTS.get('double', '1') == '1':     # float64: the changes in later predictions are small enough for float32 rounding
    model.double()
FOB = bool(getattr(model.cfg, 'first_order_overdamped', False))
assert getattr(model.cfg, 'lowrank_damped_flow', False), 'needs the exact low-rank flow (SR2, SR3-crit, FO-b)'

LAYERS = [int(t) for t in OPTS.get('layers', '0,1').split(',')]
NM, NR, EPS = int(OPTS.get('modes', 4)), int(OPTS.get('rand', 4)), float(OPTS.get('eps', 2.5e-3))
CHECK = OPTS.get('check', '0') == '1'
DT = float(model.cfg.dt)

mod = sys.modules[type(model)._layer_step_langevin.__module__]
orig_modes = mod.lowrank_modes
st = {'layer': None, 'modes': {}, 'n_calls': {}, 'out': {}, 'pert': None}
N_RETRY = [0]
def modes(G, *a, **k):
    try:
        U, kappa = orig_modes(G, *a, **k)
    except RuntimeError:
        # 2026-11-17: a CPU SVD in the code models' 'gram' driver can fail to converge on one element (seen once in
        # float64). As the model's own _svd_stable does, retry with a deterministic jitter 1e-10 relative to G; counted.
        N_RETRY[0] += 1
        gj = torch.Generator().manual_seed(N_RETRY[0])
        sc = G.detach().abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        U, kappa = orig_modes(G + 1e-10 * sc * torch.randn(G.shape, generator=gj, dtype=G.dtype), *a, **k)
    L = st['layer']
    st['n_calls'][L] = st['n_calls'].get(L, 0) + 1
    if st['pert'] is None:              # the unperturbed run: keep this layer's modes
        st['modes'][L] = (U.detach(), kappa.detach())
    return U, kappa
mod.lowrank_modes = modes

orig_step = model._layer_step_langevin
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    L = int(layer_idx); st['layer'] = L
    if st['pert'] is not None and st['pert'][0] == L:
        D = st['pert'][1].to(h.dtype)
        h, h_prev = h + EPS_NOW[0] * D, h_prev + EPS_NOW[0] * D
    h_new, h_prev_out = orig_step(h, h_prev, m_b, gamma, dt_, layer_idx)
    st['out'][L] = (h_new.detach(), h_prev_out.detach())
    return h_new, h_prev_out
model._layer_step_langevin = step
EPS_NOW = [EPS]


def forward(xb):
    st['out'] = {}; st['n_calls'] = {}
    with torch.enable_grad():         # the V_phi pair force is an autograd gradient
        lg = model(torch.from_numpy(xb))[0].detach()
    return lg - lg.mean(-1, keepdim=True)


def rms(t):                            # root mean square of the per-token norm over the leading dimensions
    return float(t.norm(dim=-1).pow(2).mean().sqrt())


RECALL = OPTS.get('recall', '0') == '1'
if RECALL:
    # 2026-11-17, code corpus: move the state of a token's earlier occurrence p and read the change at the later
    # positions q that must recall it (target y_q == x_p, p its last occurrence at or before q), in distance bands.
    BANDS = [(1, 16, 'near'), (17, 256, 'far')]
    NSLOT, MAXF = int(OPTS.get('slots', 2)), int(OPTS.get('maxfreq', 8))
    xw, yw = CS['windows']()
    if OPTS.get('offset', '0') == '1':
        # 2026-11-17 (LR1): fresh windows, evenly spaced and offset by half a spacing (the recall pathway probe's)
        _v = np.load(CS['VAL']).astype(np.int64); _W, _T = len(xw), xw.shape[1]
        _sp = (len(_v) - (_T + 1)) / _W
        _xy = np.stack([_v[int(k * _sp + _sp / 2):int(k * _sp + _sp / 2) + _T + 1] for k in range(_W)])
        xw, yw = _xy[:, :_T], _xy[:, 1:]
    rng = np.random.default_rng(20261117)
    gen = torch.Generator().manual_seed(20261116)
    def pairs(xr, yr):
        """per band: {p: [q, ...]} for targets whose last occurrence (at or before q) is p, distance q + 1 - p"""
        last, out = {}, {b[2]: {} for b in BANDS}
        cnt = np.bincount(xr, minlength=50257)
        for q in range(len(xr)):
            last[int(xr[q])] = q
            t = int(yr[q]); pp = last.get(t)
            if pp is None or pp == q or cnt[t] > MAXF:
                continue
            dist = q + 1 - pp
            for lo, hi, nm in BANDS:
                if lo <= dist <= hi:
                    out[nm].setdefault(pp, []).append(q)
        return out
    acc = {}; npairs = {b[2]: 0 for b in BANDS}
    for i in range(0, len(xw), 4):
        xb, yb = xw[i:i + 4], yw[i:i + 4]; B = len(xb)
        st['pert'] = None; lg0 = forward(xb); out0 = dict(st['out']); md = dict(st['modes'])
        assert all(st['n_calls'].get(L, 0) == 1 for L in LAYERS), st['n_calls']
        P_ = [pairs(xb[b], yb[b]) for b in range(B)]
        slots = []
        for lo, hi, nm in BANDS:
            for j in range(NSLOT):
                sel = []
                for b in range(B):
                    cand = sorted(P_[b][nm])
                    cand = [c for c in cand if c not in [ss[b][0] for _, ss in slots if ss[b]]]
                    sel.append((int(c := rng.choice(cand)), P_[b][nm][int(c)]) if cand else None)
                slots.append((nm, sel))
        for nm, sel in slots:
            npairs[nm] += sum(len(t[1]) for t in sel if t)
        for L in LAYERS:
            U, kappa = md[L]; h0, hp0 = out0[L]
            for nm, sel in slots:
                live = [b for b in range(B) if sel[b]]
                if not live:
                    continue
                pv = torch.tensor([sel[b][0] if sel[b] else 0 for b in range(B)])
                Up, kp = U[torch.arange(B), pv], kappa[torch.arange(B), pv]
                idx = kp.argsort(-1, descending=True)[:, :NM]
                Us = torch.gather(Up, -1, idx.unsqueeze(-2).expand(*Up.shape[:-1], NM))
                dirs = [('stiff', Us[..., j]) for j in range(NM)]
                for j in range(NR):
                    g = torch.randn(Up.shape[:-1], generator=gen, dtype=Up.dtype)
                    g = g - torch.einsum('bdq,bq->bd', Up, torch.einsum('bdq,bd->bq', Up, g))
                    dirs.append(('comp', g / g.norm(dim=-1, keepdim=True)))
                for n, (kind, Dp) in enumerate(dirs):
                    D = torch.zeros(U.shape[:-1], dtype=U.dtype)
                    for b in live:
                        D[b, sel[b][0]] = Dp[b]
                    reps = [EPS] + ([EPS / 2] if CHECK and n in (0, NM) else [])
                    for e in reps:
                        EPS_NOW[0] = e; st['pert'] = (L, D)
                        dl = (forward(xb) - lg0) / e
                        h1, hp1 = st['out'][L]
                        r = {k: [] for k in ('recall', 'recall_tgt', 'now', 'before', 'posU')}
                        for b in live:
                            pp, qs = sel[b]; qs = torch.tensor(qs)
                            r['recall'].append(dl[b, qs].norm(dim=-1).pow(2).mean())
                            r['recall_tgt'].append(dl[b, qs, torch.from_numpy(yb[b])[qs]].pow(2).mean())
                            r['now'].append(dl[b, pp].norm().pow(2))
                            r['before'].append(dl[b, :pp].norm(dim=-1).pow(2).mean() if pp > 0 else torch.tensor(0.0))
                            r['posU'].append(torch.einsum('dq,d->q', Up[b].to(h1.dtype), (h1 - h0)[b, pp] / e).norm().pow(2))
                        r = {k: float(torch.stack(v).mean().sqrt()) for k, v in r.items()}
                        keys = ([kind] + ([kind + '_first'] if len(reps) > 1 else [])) if e == EPS else [kind + '_half']
                        for kk in keys:
                            for k, v in r.items():
                                acc.setdefault((L, nm, kk), {}).setdefault(k, []).append(v)
                    EPS_NOW[0] = EPS
        st['pert'] = None
        print(f'windows {i + 4}/{len(xw)}', flush=True)
    res = {'pairs': npairs}
    COLS = ['recall', 'recall_tgt', 'now', 'posU', 'before']
    lines = [f'Stiff-direction transmission to recall positions: {FOLDER}', f'model {desc}',
             f'{len(xw)} validation windows x 512 ({Path(CS["VAL"]).name}); first order {FOB}; eps {EPS}; per window and band '
             f'{NSLOT} earlier occurrences p moved, one at a time; tokens occurring more than {MAXF} times in the window '
             f'excluded; (p, q) pairs read: {npairs}; mode-solver retries {N_RETRY[0]}',
             'recall = centred-logit change at the positions q whose target last occurred at p; recall_tgt = the change of '
             'the target\'s own centred logit there; now = at p; posU = p after the step, on its U; before = check, must be 0', '',
             f'{"layer":>5}  {"band":>5}  {"kind":>11}' + ''.join(f'  {c:>10}' for c in COLS)]
    for (L, nm, kind), a in sorted(acc.items()):
        r = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in a.items()}
        res[f'{L}:{nm}:{kind}'] = r
        lines.append(f'{L:>5}  {nm:>5}  {kind:>11}' + ''.join(f'  {r[c]:10.4f}' for c in COLS))
    lines.append('')
    for L in LAYERS:
        for nm in ('near', 'far'):
            s_, c_ = res.get(f'{L}:{nm}:stiff'), res.get(f'{L}:{nm}:comp')
            if s_ and c_:
                for k in ('recall', 'recall_tgt', 'now'):
                    res[f'{L}:{nm}:R_{k}'] = s_[k] / c_[k]
                lines.append(f'layer {L}, {nm}: stiff over complement: recall {res[f"{L}:{nm}:R_recall"]:.3f}, recall_tgt '
                             f'{res[f"{L}:{nm}:R_recall_tgt"]:.3f}, now {res[f"{L}:{nm}:R_now"]:.3f}; posU (stiff) {s_["posU"]:.3f}')
            if CHECK:
                for kind in ('stiff', 'comp'):
                    h_, f1 = res.get(f'{L}:{nm}:{kind}_half'), res.get(f'{L}:{nm}:{kind}_first')
                    if h_ and f1:
                        lines.append(f'  linearity, {nm} {kind} (first direction): recall {f1["recall"]:.4f} at eps, '
                                     f'{h_["recall"]:.4f} at eps/2; recall_tgt {f1["recall_tgt"]:.4f}, {h_["recall_tgt"]:.4f}')
    stem = OPTS.get('name', FOLDER[-40:])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'transmission_recall_{stem}.txt').write_text('\n'.join(lines) + '\n')
    (OUT / f'transmission_recall_{stem}.json').write_text(json.dumps({'folder': FOLDER, 'model': desc, 'first_order': FOB,
                                                                       'eps': EPS, 'readings': res}, indent=1))
    print('\n'.join(lines))
    sys.exit(0)

POS = [int(t) for t in OPTS.get('pos', '96,288').split(',')]
FUT = int(OPTS.get('fut', 64))
x, _ = CS['windows']()
assert max(POS) + 2 * FUT < x.shape[1], (POS, FUT)
gen = torch.Generator().manual_seed(20261116)
acc = {}                                # (layer, kind) -> readings, one per (batch, position, direction)
shapes = {}
for i in range(0, len(x), 4):
    xb = x[i:i + 4]
    st['pert'] = None; lg0 = forward(xb); out0 = dict(st['out']); md = dict(st['modes'])
    assert all(st['n_calls'].get(L, 0) == 1 for L in LAYERS), st['n_calls']
    for L in LAYERS:
        U, kappa = md[L]
        shapes[L] = (tuple(U.shape), tuple(kappa.shape))
        h0, hp0 = out0[L]
        for p in POS:
            Up, kp = U[:, p], kappa[:, p]                                                  # (B, d, q), (B, q)
            idx = kp.argsort(-1, descending=True)[:, :NM]
            Us = torch.gather(Up, -1, idx.unsqueeze(-2).expand(*Up.shape[:-1], NM))         # (B, d, NM)
            dirs = [('stiff', Us[..., j]) for j in range(NM)]
            for j in range(NR):
                g = torch.randn(Up.shape[:-1], generator=gen, dtype=Up.dtype)
                g = g - torch.einsum('bdq,bq->bd', Up, torch.einsum('bdq,bd->bq', Up, g))
                dirs.append(('comp', g / g.norm(dim=-1, keepdim=True)))
            for n, (kind, Dp) in enumerate(dirs):
                D = torch.zeros(U.shape[:-1], dtype=U.dtype); D[:, p] = Dp                  # only token p moves
                reps = [EPS] + ([EPS / 2] if CHECK and n in (0, NM) else [])
                for e in reps:
                    EPS_NOW[0] = e; st['pert'] = (L, D)
                    dl = (forward(xb) - lg0) / e
                    h1, hp1 = st['out'][L]
                    dh = (h1 - h0)[:, p] / e
                    dv = ((h1 - hp1) - (h0 - hp0))[:, p] / (e * DT)
                    r = {'now': rms(dl[:, p]), 'next64': rms(dl[:, p + 1:p + 1 + FUT]),
                         'after64': rms(dl[:, p + 1 + FUT:p + 1 + 2 * FUT]), 'before': rms(dl[:, :p]),
                         'pos': rms(dh), 'posU': rms(torch.einsum('bdq,bd->bq', Up.to(dh.dtype), dh)), 'vel': rms(dv)}
                    keys = ([kind] + ([kind + '_first'] if len(reps) > 1 else [])) if e == EPS else [kind + '_half']
                    for kk in keys:
                        for k, v in r.items():
                            acc.setdefault((L, kk), {}).setdefault(k, []).append(v)
                EPS_NOW[0] = EPS
    st['pert'] = None
    print(f'windows {i + 4}/{len(x)}', flush=True)

res = {}
COLS = ['now', 'next64', 'after64', 'pos', 'posU', 'vel', 'before']
lines = [f'Stiff-direction transmission: {FOLDER}', f'model {desc}',
         f'{len(x)} validation windows x 512 ({Path(CS["VAL"]).name}); first order {FOB}; eps {EPS}; one token moved '
         f'per window at positions {POS}; {NM} stiffest modes and {NR} complement directions; dt {DT}; '
         f'U, kappa shapes {shapes}; mode-solver retries {N_RETRY[0]}',
         'now = centred-logit change at the moved token; next64 / after64 = RMS over the next 64 / the 64 after those '
         '(memory); pos, posU, vel = the moved token after the layer step (all, on its U, velocity); before = check, '
         'must be 0 (causality)', '',
         f'{"layer":>5}  {"kind":>11}' + ''.join(f'  {c:>8}' for c in COLS)]
for (L, kind), a in sorted(acc.items()):
    r = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in a.items()}
    res[f'{L}:{kind}'] = r
    lines.append(f'{L:>5}  {kind:>11}' + ''.join(f'  {r[c]:8.4f}' for c in COLS))
lines.append('')
for L in LAYERS:
    s_, c_ = res[f'{L}:stiff'], res[f'{L}:comp']
    for k in ('now', 'next64', 'after64'):
        res[f'{L}:R_{k}'] = s_[k] / c_[k]
    lines.append(f'layer {L}: stiff over complement: now {res[f"{L}:R_now"]:.3f}, next64 {res[f"{L}:R_next64"]:.3f}, '
                 f'after64 {res[f"{L}:R_after64"]:.3f}; position kept on U (stiff input) {s_["posU"]:.3f}')
    if CHECK:
        for kind in ('stiff', 'comp'):
            h, f1 = res.get(f'{L}:{kind}_half'), res.get(f'{L}:{kind}_first')
            if h and f1:
                lines.append(f'  linearity, {kind} (first direction): next64 {f1["next64"]:.4f} at eps, '
                             f'{h["next64"]:.4f} at eps/2; now {f1["now"]:.4f}, {h["now"]:.4f}; posU {f1["posU"]:.4f}, '
                             f'{h["posU"]:.4f}')
stem = OPTS.get('name', FOLDER[-40:])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f'transmission_{stem}.txt').write_text('\n'.join(lines) + '\n')
(OUT / f'transmission_{stem}.json').write_text(json.dumps({'folder': FOLDER, 'model': desc, 'first_order': FOB,
                                                            'eps': EPS, 'readings': res}, indent=1))
print('\n'.join(lines))
