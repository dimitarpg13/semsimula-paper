"""RF: do the registers carry forward-looking state? (protocol SS5.23, pre-registered 2026-10-12)

At every position of 200 validation sequences (512 tokens, seed 20261012):
the full model's per-token loss, the layer-1 input state h_t (d = 384) and
the layer-1 register state: for the Poisson-mode arms the occupation n(t)
(K = 64, the strict-past Poisson mean, poisson_mode_occupation); for G2 the
per-position causal register bank as the reverse channel reads it
(M x d, flattened). Target: the mean loss over the next w tokens, w in
{1, 4, 16}. Ridge probes (regularisation chosen on 25 of the 150 training
sequences), R^2 on 50 held-out sequences, bootstrap intervals over
sequences (1,000 resamples).

  probes:   n alone; h alone; n and h; controls: n shuffled across
            positions (within each sequence; keeps the marginal, breaks the
            trace); position alone; h reduced to 64 PCs (capacity-matched)
  primary:  dR2 = R2(n, h) - R2(h) at w = 16

Checks first, exact: the captured layer-1 input reproduces the model's
occupation bit for bit when fed to poisson_mode_occupation; the captured
losses equal a direct evaluation of the model on the same batch.

Usage: python3 register_forward_probe.py OUT_DIR {pmx|pm1|g2} [n_seq=200] [past]
  past (descriptive follow-up, 2026-10-12): add the mean loss over the PREVIOUS
  16 tokens as a feature, alone and with n, h and n+h, to ask whether n's
  forward signal is content or local difficulty (a frequency proxy).
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path(sys.argv[1]); KIND = sys.argv[2]
N_SEQ = next((int(a[6:]) for a in sys.argv[3:] if a.startswith('n_seq=')), 200)
PAST = 'past' in sys.argv[3:]
assert KIND in ('pmx', 'pm1', 'g2'), KIND
sys.argv = [sys.argv[0], str(OUT / 'harness')]
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import verify_pm_switch as P

T, W, SEED = 512, (1, 4, 16), 20261012
N_TRAIN, N_VAL, N_TEST = 125, 25, 50
if N_SEQ < 200:                                              # smoke runs
    N_TRAIN, N_VAL, N_TEST = max(2, N_SEQ // 2), max(1, N_SEQ // 8), max(1, N_SEQ - N_SEQ // 2 - max(1, N_SEQ // 8))
SR2 = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
PMXS = (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),)
_T = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
_E = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
ARMS = {'pmx': (_T + 'pm64_pmx16_sr2' + _E, P.CONFIGS['F3.1'][1] + P.PM_ON + SR2 + PMXS),
        'pm1': (_T + 'pm64' + _E, P.CONFIGS['F3.1'][1] + P.PM_ON),
        'g2': (P.G2_F, P.CONFIGS['G2'][1])}
folder, repl = ARMS[KIND]
model, g, _ = P.build(folder, repl)
model.to('cpu'); model.eval()
L = model.cfg.L
has_pm = getattr(model, 'pm_mu', None) is not None
assert has_pm == (KIND != 'g2') and (model.reverse_ch is not None) == (KIND == 'g2')

lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
say(f'RF, register forward probe: {KIND}  ...{folder[-70:]}')

# ---- sequences ----
val = np.load(P.G.VAL)
rng = np.random.default_rng(SEED)
starts = rng.integers(0, len(val) - T - 1, size=N_SEQ)
X = np.stack([val[s:s + T] for s in starts]).astype(np.int64)
Y = np.stack([val[s + 1:s + T + 1] for s in starts]).astype(np.int64)

# ---- capture: layer-1 input state and register state, per sequence ----
cap = {}
orig_step = model._fock_layer_step
def spy_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
    if layer_idx == L - 1:
        cap['h'] = h.detach().clone()
        if KIND == 'g2':
            cap['r_in'] = r.detach().clone()
    return orig_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
hook = None
if KIND == 'g2':
    def rc_hook(mod, inp, out):
        if cap.get('want_rc', False):
            cap['r_rev'] = inp[1].detach().clone(); cap['want_rc'] = False
    hook = model.reverse_ch.register_forward_hook(rc_hook)

H, R, LOSS = [], [], []
model._fock_layer_step = spy_step
try:
    BS = 4
    for b0 in range(0, N_SEQ, BS):
        xb = torch.from_numpy(X[b0:b0 + BS]); yb = torch.from_numpy(Y[b0:b0 + BS])
        cap.clear()
        if KIND == 'g2':
            cap['want_rc'] = True
            # the reverse channel is called once per layer; we want layer L-1's call
            calls = []
            def rc_hook2(mod, inp, out):
                calls.append(inp[1].detach().clone())
            hook.remove(); hook = model.reverse_ch.register_forward_hook(rc_hook2)
        with torch.enable_grad():
            logits, _ = model(xb)
        loss = F.cross_entropy(logits.flatten(0, 1), yb.flatten(), reduction='none').view(len(xb), T).detach().numpy()
        LOSS.append(loss)
        H.append(cap['h'].numpy())
        if KIND == 'g2':
            r = calls[-1]                                        # (B, T, M, d) prefix-causal bank at the last layer
            assert r.dim() == 4 and r.shape[1] == T, r.shape
            R.append(r.flatten(2).numpy())
        else:
            with torch.no_grad():
                _, phi = model.poisson_mode_occupation(cap['h'])
            R.append(phi.numpy())
        if b0 == 0:
            # checks: occupation reproduces; losses match a direct evaluation
            with torch.enable_grad():
                lg2, _ = model(xb)
            d_loss = float((F.cross_entropy(lg2.flatten(0, 1), yb.flatten(), reduction='none').view(len(xb), T).detach().numpy() - loss).max())
            say(f'CHECK  captured losses vs direct evaluation, max |d| {d_loss:.1e}')
            if has_pm:
                with torch.no_grad():
                    _, phi2 = model.poisson_mode_occupation(torch.from_numpy(H[0]))
                say(f'CHECK  occupation from the captured layer-{L-1} input vs the model\'s, max |d| {float((phi2.numpy() - R[0]).max()):.1e}')
            else:
                say(f'CHECK  register bank captured at the reverse channel: shape {tuple(calls[-1].shape)}, {len(calls)} calls per forward (one per layer: {L})')
        if (b0 // BS) % 10 == 0:
            print(f'   captured {b0 + len(xb)}/{N_SEQ}', flush=True)
finally:
    model._fock_layer_step = orig_step
    if hook is not None: hook.remove()
assert model._fock_layer_step is orig_step
H = np.concatenate(H); R = np.concatenate(R); LOSS = np.concatenate(LOSS)       # (N, T, .)
say(f'   {N_SEQ} x {T} tokens; register state width {R.shape[-1]}; state width {H.shape[-1]}; mean loss {LOSS.mean():.4f} (PPL {np.exp(LOSS.mean()):.2f})')

# ---- targets: mean loss over the next w tokens ----
def target(w):
    cs = np.concatenate([np.zeros((N_SEQ, 1)), np.cumsum(LOSS, axis=1)], axis=1)       # (N, T+1)
    t = np.arange(T)
    hi = np.minimum(t + 1 + w, T)
    y = (cs[:, hi] - cs[:, t + 1]) / np.maximum(hi - (t + 1), 1)
    valid = (t + 1 + w) <= T                                                            # full window available
    return y, valid

# ---- features ----
pos = (np.arange(T) / T)[None, :, None].repeat(N_SEQ, 0)
rng2 = np.random.default_rng(SEED + 1)
R_shuf = np.stack([R[i][rng2.permutation(T)] for i in range(N_SEQ)])                   # modes kept, positions shuffled
idx = rng.permutation(N_SEQ)
tr, va, te = idx[:N_TRAIN], idx[N_TRAIN:N_TRAIN + N_VAL], idx[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]
# capacity-matched h: 64 PCs fit on the training split
Hm = H[tr].reshape(-1, H.shape[-1]).mean(0)
U, S, Vt = np.linalg.svd(H[tr].reshape(-1, H.shape[-1])[::8] - Hm, full_matrices=False)
H64 = (H - Hm) @ Vt[:R.shape[-1] if R.shape[-1] <= 64 else 64].T

def ridge_fit(Xtr, ytr, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Z = (Xtr - mu) / sd
    ym = ytr.mean()
    A = Z.T @ Z + lam * np.eye(Z.shape[1]); b = Z.T @ (ytr - ym)
    w = np.linalg.solve(A, b)
    return mu, sd, ym, w

def ridge_pred(p, Xte):
    mu, sd, ym, w = p
    return ((Xte - mu) / sd) @ w + ym

def r2(y, yhat):
    return 1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def probe(feat, w, seqs_tr, seqs_te, lam=None):
    y, valid = target(w)
    Xtr = feat[seqs_tr][:, valid].reshape(-1, feat.shape[-1]); ytr = y[seqs_tr][:, valid].ravel()
    if lam is None:                                     # choose on the validation split
        Xva = feat[va][:, valid].reshape(-1, feat.shape[-1]); yva = y[va][:, valid].ravel()
        best = max((r2(yva, ridge_pred(ridge_fit(Xtr, ytr, l), Xva)), l) for l in (1e-2, 1e-1, 1, 10, 100, 1000))
        lam = best[1]
    p = ridge_fit(Xtr, ytr, lam)
    per_seq = np.array([r2(y[s][valid], ridge_pred(p, feat[s][valid])) for s in seqs_te])
    pooled = r2(y[seqs_te][:, valid].ravel(), ridge_pred(p, feat[seqs_te][:, valid].reshape(-1, feat.shape[-1])))
    return pooled, per_seq, lam

FEATS = {'n': R, 'h': H, 'n+h': np.concatenate([R, H], -1), 'n shuffled': R_shuf, 'position': pos, 'h, 64 PCs': H64}
if PAST:
    csum = np.concatenate([np.zeros((N_SEQ, 1)), np.cumsum(LOSS, axis=1)], axis=1)
    t = np.arange(T); lo = np.maximum(t - 16, 0)
    past16 = ((csum[:, t] - csum[:, lo]) / np.maximum(t - lo, 1))[..., None]           # mean loss over the previous 16 tokens
    FEATS = {'past16': past16, 'n': R, 'n+past16': np.concatenate([R, past16], -1), 'h+past16': np.concatenate([H, past16], -1),
             'n+h+past16': np.concatenate([R, H, past16], -1), 'h': H, 'n+h': np.concatenate([R, H], -1)}
res = {}
say('\nR^2 on the held-out sequences (pooled), target = mean loss over the next w tokens')
say('   feature          ' + '  '.join(f'w={w:<7d}' for w in W))
for name, feat in FEATS.items():
    row = {}
    for w in W:
        pooled, per_seq, lam = probe(feat, w, tr, te)
        row[w] = dict(r2=float(pooled), per_seq=per_seq.tolist(), lam=float(lam))
    res[name] = row
    say(f'   {name:<16} ' + '  '.join(f'{row[w]["r2"]:+.4f}   ' for w in W))

# ---- primary statistic and bootstrap ----
boot = np.random.default_rng(SEED + 2)
def boot_ci(a, b=None, n=1000):
    a = np.asarray(a); b = None if b is None else np.asarray(b)
    stats = []
    for _ in range(n):
        s = boot.integers(0, len(a), len(a))
        stats.append(a[s].mean() if b is None else (a[s] - b[s]).mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
if PAST:
    say('\nFOLLOW-UP  does n\'s increment survive the past-16 difficulty feature? per held-out sequence, 95% bootstrap CI')
    for w in W:
        lo1, hi1 = boot_ci(np.array(res['n+h'][w]['per_seq']), np.array(res['h'][w]['per_seq']))
        lo2, hi2 = boot_ci(np.array(res['n+h+past16'][w]['per_seq']), np.array(res['h+past16'][w]['per_seq']))
        lo3, hi3 = boot_ci(np.array(res['n+past16'][w]['per_seq']), np.array(res['past16'][w]['per_seq']))
        say(f'   w={w:<3d} n over h: [{lo1:+.4f}, {hi1:+.4f}]   n over h+past16: [{lo2:+.4f}, {hi2:+.4f}]   n over past16 alone: [{lo3:+.4f}, {hi3:+.4f}]')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'register_forward_probe_{KIND}_past_output.txt').write_text('\n'.join(lines) + '\n')
    print('wrote', OUT / f'register_forward_probe_{KIND}_past_output.txt'); raise SystemExit(0)
say('\nPRIMARY  dR2 = R2(n+h) - R2(h), per held-out sequence (mean, 95% bootstrap CI over sequences)')
calls = {}
for w in W:
    d = np.array(res['n+h'][w]['per_seq']) - np.array(res['h'][w]['per_seq'])
    lo, hi = boot_ci(np.array(res['n+h'][w]['per_seq']), np.array(res['h'][w]['per_seq']))
    say(f'   w={w:<3d} dR2 {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]')
    calls[w] = dict(dR2=float(d.mean()), lo=lo, hi=hi)
say('\nCONTROLS  n alone against the shuffle and against position, per sequence')
rfa = True
for w in W:
    for ctrl in ('n shuffled', 'position'):
        lo, hi = boot_ci(np.array(res['n'][w]['per_seq']), np.array(res[ctrl][w]['per_seq']))
        ok = lo > 0
        rfa &= ok
        say(f'   w={w:<3d} n - {ctrl:<11s} [{lo:+.4f}, {hi:+.4f}] -> {"disjoint from 0" if ok else "NOT"}')
lo64, hi64 = boot_ci(np.array(res['h, 64 PCs'][16]['per_seq']), np.array(res['n'][16]['per_seq']))
d16 = calls[16]['dR2']
say('\nCALLS (protocol SS5.23)')
say(f'   RF-a  n beats shuffle and position at every w            -> {"HIT" if rfa else "MISS"}')
say(f'   RF-b  dR2 >= 0.02 at w = 16                                 -> {"HIT" if d16 >= 0.02 else "MISS"} ({d16:+.4f})')
say(f'   RF-c  dR2 increases from w = 1 to w = 16                    -> {"HIT" if calls[16]["dR2"] > calls[1]["dR2"] else "MISS"} ({calls[1]["dR2"]:+.4f} -> {calls[16]["dR2"]:+.4f})')
say(f'   RF-d  h (64 PCs) beats n alone at w = 16                    -> {"HIT" if lo64 > 0 else "MISS"} ([{lo64:+.4f}, {hi64:+.4f}])')
OUT.mkdir(parents=True, exist_ok=True)
stem = f'register_forward_probe_{KIND}' + ('' if N_SEQ == 200 else f'_n{N_SEQ}')
(OUT / f'{stem}_output.txt').write_text('\n'.join(lines) + '\n')
(OUT / f'{stem}.json').write_text(json.dumps({'r2': {k: {str(w): {'r2': v[w]['r2'], 'lam': v[w]['lam']} for w in W} for k, v in res.items()}, 'primary': {str(w): v for w, v in calls.items()}}, indent=1))
print(f'\nwrote {OUT}/{stem}_output.txt')
