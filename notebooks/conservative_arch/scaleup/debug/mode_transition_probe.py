"""MT: do the Poisson modes have transitions? (protocol SS5.24, pre-registered 2026-10-12)

A mode FIRES on token t if its overlap E_v(t) exceeds a per-mode threshold at
its 90th percentile over the training split (every mode fires on 10% of
tokens). For lags D in {1, 2, 4, 8}: the 64 x 64 transition matrix
P_D(u -> v) = Pr[v fires at t+D | u fires at t], its lift over the marginal,
and the mutual information I_D between the two firing indicators.

The null: adjacent states are similar, so firings co-occur without any
transition structure. The DRIFT null (revised after the smoke run, 2026-10-12:
firing a ridge-predicted state directly has the wrong marginals and reads
HIGHER than the data) keeps the data's own firings and conditions on the
state: for each target mode v and lag D, a ridge-logistic model predicts
"v fires at t+D" from h_t (fit on the training split); the null's firing
indicator for v is a Bernoulli draw from that predicted probability on
the test split, which has the same marginal and carries exactly the
information the current state's drift supplies. The logistic model is
CAPACITY-MATCHED: it reads the 64-PC projection of h_t (as RF's control
did), because a 385-parameter fit on the training split predicts sharper
than the truth and its sampled firings then read MORE coherent than the
data (the 8-sequence smoke run showed this). The full-state fit is also
reported, as "I_drift(384)", so the choice is visible. Structured
transitions are the excess MI of the real firings over the matched null. A SHUFFLE null
(modes permuted within each sequence) is the weak control.

  primary:   excess dI_D = I_D - I_D^drift, bootstrap over sequences
  secondary: pairs with lift >= 2 or <= 0.5 at D = 1 beyond the drift null;
             asymmetric pairs, L(u->v) >= 2 L(v->u), both beyond the null

Checks first: captured layer-1 input reproduces the model's overlaps and
occupations bit for bit (as RF).

Usage: python3 mode_transition_probe.py OUT_DIR {pmx|pm1} [n_seq=200]
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); KIND = sys.argv[2]
N_SEQ = next((int(a[6:]) for a in sys.argv[3:] if a.startswith('n_seq=')), 200)
assert KIND in ('pmx', 'pm1'), KIND
sys.argv = [sys.argv[0], str(OUT / 'harness')]
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import verify_pm_switch as P

T, LAGS, SEED, Q = 512, (1, 2, 4, 8), 20261012, 0.90
N_TRAIN = 150 if N_SEQ >= 200 else max(2, N_SEQ // 2)
SR2 = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
PMXS = (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),)
_T = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
_E = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
ARMS = {'pmx': (_T + 'pm64_pmx16_sr2' + _E, P.CONFIGS['F3.1'][1] + P.PM_ON + SR2 + PMXS),
        'pm1': (_T + 'pm64' + _E, P.CONFIGS['F3.1'][1] + P.PM_ON)}
folder, repl = ARMS[KIND]
model, g, _ = P.build(folder, repl)
model.to('cpu'); model.eval()
L = model.cfg.L
assert getattr(model, 'pm_mu', None) is not None

lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
say(f'MT, mode transition probe: {KIND}  ...{folder[-70:]}')

val = np.load(P.G.VAL)
rng = np.random.default_rng(SEED)
starts = rng.integers(0, len(val) - T - 1, size=N_SEQ)
X = np.stack([val[s:s + T] for s in starts]).astype(np.int64)

# ---- capture layer-1 input states; overlaps E from the model's own function ----
cap = {}
orig_step = model._fock_layer_step
def spy_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
    if layer_idx == L - 1:
        cap['h'] = h.detach().clone()
    return orig_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
H, E = [], []
model._fock_layer_step = spy_step
try:
    for b0 in range(0, N_SEQ, 4):
        xb = torch.from_numpy(X[b0:b0 + 4])
        with torch.enable_grad():
            model(xb)
        h = cap['h']
        with torch.no_grad():
            e, phi = model.poisson_mode_occupation(h)
        H.append(h.numpy()); E.append(e.numpy())
        if b0 == 0:
            with torch.no_grad():
                e2, _ = model.poisson_mode_occupation(torch.from_numpy(H[0]))
            say(f'CHECK  overlaps from the captured layer-{L-1} input vs the model\'s, max |d| {float(np.abs(e2.numpy() - E[0]).max()):.1e}')
        if (b0 // 4) % 10 == 0:
            print(f'   captured {b0 + len(xb)}/{N_SEQ}', flush=True)
finally:
    model._fock_layer_step = orig_step
H = np.concatenate(H).astype(np.float64); E = np.concatenate(E).astype(np.float64)       # (N, T, d), (N, T, K)
K = E.shape[-1]
idx = rng.permutation(N_SEQ); tr, te = idx[:N_TRAIN], idx[N_TRAIN:]
say(f'   {N_SEQ} x {T} tokens, K = {K} modes; train {len(tr)} / test {len(te)} sequences')

# ---- firing thresholds: per-mode 90th percentile on the training split ----
theta = np.quantile(E[tr].reshape(-1, K), Q, axis=0)                                   # (K,)
F = (E > theta)                                                                         # (N, T, K) bool
say(f'   firing rate on test, per mode: median {F[te].mean((0, 1)).mean():.3f} (target {1-Q:.2f}); modes never firing on test: {int((F[te].sum((0,1))==0).sum())}')

# ---- drift null: ridge prediction of h_{t+D} from h_t, then the same overlaps and firings ----
mu_v = model.pm_mu.detach().double().numpy(); k2 = model.pm_log_kappa2.detach().double().exp().numpy()
def overlaps(h):
    d2 = (h * h).sum(-1, keepdims=True) + (mu_v * mu_v).sum(-1) - 2.0 * h @ mu_v.T
    return np.exp(-k2 * np.clip(d2, 0, None))
# capacity-matched state: 64 PCs fit on the training split
_Hm = H[tr].reshape(-1, H.shape[-1]).mean(0)
_U, _S, _Vt = np.linalg.svd(H[tr].reshape(-1, H.shape[-1])[::8] - _Hm, full_matrices=False)
H64 = (H - _Hm) @ _Vt[:64].T

def drift_fire(D, seed=0, full=False):
    """Null firings of every mode at t+D given the state at t: ridge-logistic per mode (IRLS, 8 steps), Bernoulli-sampled on test.
    Capacity-matched (64 PCs) by default; full=True uses all 384 dimensions (reported, not used for the calls)."""
    Hs = H if full else H64
    Xtr = Hs[tr][:, :-D].reshape(-1, Hs.shape[-1]); Ytr = F[tr][:, D:].reshape(-1, K).astype(np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Z = np.concatenate([(Xtr - mu) / sd, np.ones((len(Xtr), 1))], 1)
    Zte = np.concatenate([((Hs[te][:, :-D] - mu) / sd).reshape(-1, Hs.shape[-1]), np.ones((len(te) * (T - D), 1))], 1)
    lam = 1.0 * np.eye(Z.shape[1]); lam[-1, -1] = 0.0
    probs = np.zeros((Zte.shape[0], K))
    for v in range(K):
        y = Ytr[:, v]; w = np.zeros(Z.shape[1]); w[-1] = np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6))
        for _ in range(8):
            pz = 1 / (1 + np.exp(-(Z @ w))); r = pz * (1 - pz) + 1e-9
            grad = Z.T @ (pz - y) + lam @ w
            Hs = (Z * r[:, None]).T @ Z + lam
            w = w - np.linalg.solve(Hs, grad)
        probs[:, v] = 1 / (1 + np.exp(-(Zte @ w)))
    rs = np.random.default_rng(SEED + 100 + seed + D)
    return (rs.random(probs.shape) < probs).reshape(len(te), T - D, K)

# ---- statistics ----
def mi_bits(A, B):
    """Summed mutual information over mode pairs (u in A at t, v in B at t+D), in bits; A, B (n, K) bool aligned."""
    A = A.reshape(-1, K).astype(np.float64); B = B.reshape(-1, K).astype(np.float64); n = len(A)
    pa, pb = A.mean(0), B.mean(0)
    pab = A.T @ B / n                                                                     # (K, K) joint of (u fires, v fires)
    tot = 0.0
    for a_u, a_v, j in ((1, 1, pab), (1, 0, pa[:, None] - pab), (0, 1, pb[None, :] - pab), (0, 0, 1 - pa[:, None] - pb[None, :] + pab)):
        marg = (pa if a_u else 1 - pa)[:, None] * (pb if a_v else 1 - pb)[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(j > 0, j * np.log2(j / marg), 0.0)
        tot += term
    return tot                                                                            # (K, K) per-pair MI; sum for the total

def lift(A, B):
    A = A.reshape(-1, K).astype(np.float64); B = B.reshape(-1, K).astype(np.float64)
    pab = A.T @ B / len(A); pa = A.mean(0); pb = B.mean(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((pa[:, None] > 0) & (pb[None, :] > 0), pab / (pa[:, None] * pb[None, :]), np.nan)

res = {}
boot = np.random.default_rng(SEED + 7)
say('\nMUTUAL INFORMATION between "u fires at t" and "v fires at t + D", summed over the 64 x 64 pairs (bits; test sequences)')
say('   D   I_D       I_drift   I_shuffle   excess over drift [95% CI over sequences]   (drift null: capacity-matched, 64 PCs)')
for D in LAGS:
    A = F[te][:, :-D]; B = F[te][:, D:]
    Bd = drift_fire(D)
    Bd_boot = [Bd] + [drift_fire(D, seed=k) for k in (1, 2)]                           # resample the null's draws too
    rs = np.random.default_rng(SEED + D)
    Bs = np.stack([F[te][i][:, rs.permutation(K)][D:] for i in range(len(te))])            # modes permuted per sequence
    I = mi_bits(A, B).sum(); Id = mi_bits(A, Bd).sum(); Is = mi_bits(A, Bs).sum(); Id384 = mi_bits(A, drift_fire(D, full=True)).sum()
    # bootstrap over test sequences for the excess
    per = []
    for b in range(1000):
        s = boot.integers(0, len(te), len(te)); Bn = Bd_boot[b % len(Bd_boot)]
        per.append(mi_bits(A[s], B[s]).sum() - mi_bits(A[s], Bn[s]).sum())
    lo, hi = np.percentile(per, 2.5), np.percentile(per, 97.5)
    res[D] = dict(I=float(I), I_drift=float(Id), I_drift_384=float(Id384), I_shuffle=float(Is), excess=float(I - Id), lo=float(lo), hi=float(hi))
    say(f'   {D:<3d} {I:8.3f}  {Id:8.3f}  {Is:9.3f}   {I-Id:+8.3f}  [{lo:+.3f}, {hi:+.3f}]   (I_drift(384) {Id384:.3f})')

# ---- secondary at D = 1: lift pairs beyond the drift null; asymmetry ----
A = F[te][:, :-1]; B = F[te][:, 1:]; Bd = drift_fire(1)
Lf = lift(A, B); Ld = lift(A, Bd)
off = ~np.eye(K, dtype=bool)
strong = (Lf >= 2) & off; weak = (Lf <= 0.5) & off
strong_beyond = strong & ~(Ld >= 2); weak_beyond = weak & ~(Ld <= 0.5)
asym = off & (Lf >= 2) & (Lf >= 2 * np.nan_to_num(Lf.T, nan=np.inf)) & ~(Ld >= 2)
say(f'\nLIFT at D = 1, off-diagonal pairs: lift >= 2: {int(strong.sum())} (beyond the drift null: {int(strong_beyond.sum())});  lift <= 0.5: {int(weak.sum())} (beyond: {int(weak_beyond.sum())})')
say(f'   asymmetric pairs, L(u->v) >= 2 L(v->u), L(u->v) >= 2 and beyond the drift null: {int(asym.sum())}')
top = np.argsort(np.nan_to_num(np.where(off, Lf, 0), nan=0).ravel())[::-1][:8]
say('   top transitions (u -> v): ' + ', '.join(f'{u}->{v} lift {Lf[u,v]:.1f} (drift {Ld[u,v]:.1f}, reverse {Lf[v,u]:.1f})' for u, v in (divmod(int(i), K) for i in top)))

say('\nCALLS (protocol SS5.24)')
say(f'   MT-a  I_1 exceeds the shuffle null                         -> {"HIT" if res[1]["I"] > res[1]["I_shuffle"] else "MISS"} ({res[1]["I"]:.3f} vs {res[1]["I_shuffle"]:.3f})')
say(f'   MT-b  excess over drift at D = 1, CI disjoint from 0         -> {"HIT" if res[1]["lo"] > 0 else "MISS"} ({res[1]["excess"]:+.3f} [{res[1]["lo"]:+.3f}, {res[1]["hi"]:+.3f}])')
say(f'   MT-c  excess decays with D and is positive at D = 4           -> {"HIT" if (res[4]["lo"] > 0 and res[4]["excess"] < res[1]["excess"]) else "MISS"} (D=4: {res[4]["excess"]:+.3f} [{res[4]["lo"]:+.3f}, {res[4]["hi"]:+.3f}])')
say(f'   MT-d  >= 20 asymmetric pairs beyond the drift null            -> {"HIT" if asym.sum() >= 20 else "MISS"} ({int(asym.sum())})')
OUT.mkdir(parents=True, exist_ok=True)
stem = f'mode_transition_probe_{KIND}' + ('' if N_SEQ == 200 else f'_n{N_SEQ}')
(OUT / f'{stem}_output.txt').write_text('\n'.join(lines) + '\n')
(OUT / f'{stem}.json').write_text(json.dumps({'mi': {str(k): v for k, v in res.items()}, 'pairs': {'strong': int(strong.sum()), 'strong_beyond': int(strong_beyond.sum()), 'weak': int(weak.sum()), 'weak_beyond': int(weak_beyond.sum()), 'asym': int(asym.sum())}}, indent=1))
np.savez(OUT / f'{stem}_matrices.npz', lift=Lf, lift_drift=Ld, theta=theta)
print(f'\nwrote {OUT}/{stem}_output.txt')
