#!/usr/bin/env python3
"""
SY instruments and SY0: the measures of the synchronization programme and
their validation on systems where the answer is known.

Design: companion_notes/Spontaneous_Synchronization_and_Semantic_Binding_Test_Programme.md
(measures §3, SY0 §4); pre-registration: protocol §5.22, which makes SY0 a
prerequisite of every model measurement ("a failure is fixed and re-run,
never waived").

    python3 sy_instruments.py [n_draws=40]

Run 1 (sy0_v1_output.txt) tested the measures as the design note specifies them and rejected most
of them; this is run 2, with the instruments fixed as protocol §5.22 requires. What run 1 found:
  - raw (uncentred) measures read a shared drift as synchronization (+0.75 in coherence);
  - centred coherence R peaks at the Kuramoto threshold and then falls as synchronization completes
    (+0.08 at 1.25 K_c, +0.03 at 3 K_c with order parameter 0.98): for centred vectors y_t,
    sum over t != s of y_t . y_s = -sum |y_t|^2, so their mean pairwise cosine is pinned near
    -1/(N-1) as the spread vanishes. It passed run 1's letter, which had no monotonicity
    requirement, and is unusable;
  - "coupling switched off" removes the coupling's common push along with its token-specific part,
    and a common push aligns states indirectly, through the per-token nonlinearity and through the
    closing LayerNorm (Gamma +0.56 and +1.82 under pure drift).

Run 2 (sy0_v2_output.txt) admitted Gamma_spec, C_spec, and passed the binding measures everywhere
except check 3, where a pull toward each token's running mean lowered B by 0.033 at both widths.
A diagnostic (scratchpad) showed this is a real effect, not an artefact: a pull toward one shared
target gives exactly 0 (-0.0000 without the per-token map), while per-token targets dilute tight
clusters by the same amount with or without the map and at either width. A coupling with a
different target for every token is token-specific, so it was the wrong null. Run 3 (this one) makes
check 3 the shared-target pull, decided two-sided, and keeps the running-mean pull as check 3' on
record (descriptive: its sign is the check that dilution reads as dilution, not as binding).

The fix. Every attribution has two references: the switch-off replay (total) and a common-mode
replay, in which the coupling's force on every token is replaced by its cross-token mean
(specific). Synchronization is what the coupling does differently to different tokens; a shared
push is reported but cannot count. For Kuramoto the common part is exactly 0, so nothing is lost.

Measures (run 2):
  global    Gamma_tot, Gamma_spec   pairwise contraction of the pre-LayerNorm step output against the
                                    switch-off / common-mode replay: mean over pairs of
                                    log(|h_t - h_s| at reference / |h_t - h_s|), > 0 when the coupling
                                    brings tokens together; translation-invariant
            R_tot, R_spec           centred coherence, kept on record (rejected in run 1)
  binding   B_tot, B_spec           centred cosine, related minus distance-matched unrelated, on the
                                    states the next layer reads (after any LayerNorm), on minus reference
            Gb_tot, Gb_spec         the binding form of Gamma: contraction of related pairs minus
                                    distance-matched unrelated ones
            C_tot, C_spec           the binding contrast on the centred pre-LayerNorm displacements

Checks and criteria (run 2; fixed before it runs):
  1  Kuramoto, K swept through K_c: null at K <= 0.5 K_c, passes the floor at K >= 2 K_c, and point
     estimates non-decreasing in K (added after run 1, which lacked it).
  2  Causal attention with V = I, groups planted only in the attention weights: the on-state B passes
     the floor at the deepest point and rises with depth; the attribution passes the floor; a
     shuffled-label control is null.
  3  Planted clusters (similar inputs), shared per-token map, and a pull toward one shared target (the
     mean of all tokens) as the coupling: the on-state B passes the floor; every binding attribution
     is null. 3' (descriptive): the same with a pull toward each token's running mean.
  4  Common-mode drift (a) without and (b) with a closing LayerNorm: every attribution is null.
  5  (descriptive) a common external attractor, no interaction.
  Checks 3 and 4 run at d = 16 (as run 1) and at d = 384, the width of every model in the programme.
  The second-order distortions run 1 found shrink with the width; the verdict for checks 3 and 4 is
  read at d = 384, the system under test, and d = 16 is reported beside it.

"Null": the 95% interval includes 0, or the point estimate is below the floor in magnitude.
"Passes the floor": the interval excludes 0 and the point estimate is at least the floor
(protocol §5.22: 0.02; Gamma, in log-distance units, takes the same floor).

Library use by the model scripts: from sy_instruments import centred, coherence, contraction_matrix,
cos_matrix, bin_stats, contrast, boot_mean, boot_contrast_diff.
"""
import json, sys
from pathlib import Path
import numpy as np

FLOOR = 0.02
EDGES = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 10 ** 9])
N_BOOT = 1000
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- state versions and pair matrices
def _unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)

def raw(H):
    return _unit(H)

def centred_v0(H):                       # design note §3.1 as written
    U = _unit(H)
    return _unit(U - U.mean(0))

def centred(H):                          # translation-invariant: subtract the cross-token mean, then normalise
    return _unit(H - H.mean(0))

VERSIONS = {"raw": raw, "v0": centred_v0, "c": centred}

def cos_matrix(X):
    return X @ X.T

def _dist2(H):
    n = (H * H).sum(1)
    return np.maximum(n[:, None] + n[None, :] - 2 * H @ H.T, 0.0)

def contraction_matrix(H, Href, eps=1e-12):
    """log(|h_t - h_s| at reference / |h_t - h_s|) for every pair; diagonal 0."""
    M = 0.5 * (np.log(_dist2(Href) + eps) - np.log(_dist2(H) + eps))
    np.fill_diagonal(M, 0.0)
    return M

def coherence(M):
    """mean off-diagonal entry of a pair matrix (R for a cosine matrix, Gamma for a contraction matrix)."""
    N = len(M)
    return (M.sum() - np.trace(M)) / (N * (N - 1))


# ---------------------------------------------------------------- distance-matched contrasts
def bin_stats(M, gid, pos, exclude=None):
    """Per distance bin: [sum over related pairs, count, sum over unrelated pairs, count].
    Related: same group id (>= 0). Unrelated: both labelled, different ids. Pairs t < s only;
    `exclude` (N x N bool) removes pairs, e.g. two BPE pieces of one word."""
    gid = np.asarray(gid); pos = np.asarray(pos)
    iu = np.triu_indices(len(gid), 1)
    lab = (gid[iu[0]] >= 0) & (gid[iu[1]] >= 0)
    rel = lab & (gid[iu[0]] == gid[iu[1]])
    unr = lab & ~rel
    if exclude is not None:
        keep = ~exclude[iu]; rel &= keep; unr &= keep
    b = np.searchsorted(EDGES, np.abs(pos[iu[0]] - pos[iu[1]]), side="right") - 1
    v = M[iu]
    S = np.zeros((len(EDGES) - 1, 4))
    np.add.at(S[:, 0], b[rel], v[rel]); np.add.at(S[:, 1], b[rel], 1)
    np.add.at(S[:, 2], b[unr], v[unr]); np.add.at(S[:, 3], b[unr], 1)
    return S

def contrast(S):
    """Related mean minus unrelated mean reweighted to the related pairs' distance distribution."""
    ok = (S[:, 1] > 0) & (S[:, 3] > 0)
    if not ok.any():
        return np.nan
    w = S[ok, 1] / S[ok, 1].sum()
    return S[ok, 0].sum() / S[ok, 1].sum() - (w * S[ok, 2] / S[ok, 3]).sum()


# ---------------------------------------------------------------- cluster bootstrap over draws/sequences
def _idx(n, seed):
    return np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n))

def boot_mean(x, seed=0):
    x = np.asarray(x, float); I = _idx(len(x), seed)
    b = x[I].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def boot_contrast(S_list, seed=0):
    S = np.asarray(S_list); I = _idx(len(S), seed)
    b = np.array([contrast(S[i].sum(0)) for i in I])
    return float(contrast(S.sum(0))), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))

def boot_contrast_diff(S_on, S_off, seed=0):
    A = np.asarray(S_on); Bo = np.asarray(S_off); I = _idx(len(A), seed)
    b = np.array([contrast(A[i].sum(0)) - contrast(Bo[i].sum(0)) for i in I])
    pt = contrast(A.sum(0)) - contrast(Bo.sum(0))
    return float(pt), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))

def is_null(ci):
    pt, lo, hi = ci
    return (lo <= 0 <= hi) or abs(pt) < FLOOR

def passes_floor(ci):
    pt, lo, hi = ci
    return (lo > 0 and pt >= FLOOR) or (hi < 0 and pt <= -FLOOR)

def fmt(ci):
    return f"{ci[0]:+.4f} [{ci[1]:+.4f}, {ci[2]:+.4f}]"


# ---------------------------------------------------------------- all measures for one draw
def measures(H_on, H_off, H_in, gid, pos):
    """Per-draw quantities: global scalars (on/off) and binding bin stats (on/off) for every version."""
    out = {"g": {}, "S": {}}
    for k, f in VERSIONS.items():
        out["g"][f"R_{k}_on"] = coherence(cos_matrix(f(H_on)))
        out["g"][f"R_{k}_off"] = coherence(cos_matrix(f(H_off)))
    G = contraction_matrix(H_on, H_off)
    out["g"]["Gamma"] = coherence(G)
    if gid is not None:
        for k, f in VERSIONS.items():
            out["S"][f"B_{k}_on"] = bin_stats(cos_matrix(f(H_on)), gid, pos)
            out["S"][f"B_{k}_off"] = bin_stats(cos_matrix(f(H_off)), gid, pos)
        for k, f in (("raw", raw), ("c", centred)):
            out["S"][f"C_{k}_on"] = bin_stats(cos_matrix(f(H_on - H_in)), gid, pos)
            out["S"][f"C_{k}_off"] = bin_stats(cos_matrix(f(H_off - H_in)), gid, pos)
        out["S"]["Gb"] = bin_stats(G, gid, pos)
    return out

def summarise(draws, with_binding=True, seed=0):
    """Bootstrap intervals for every attribution (on minus off) and the on-state binding values."""
    r = {}
    for k in VERSIONS:
        r[f"dR_{k}"] = boot_mean([d["g"][f"R_{k}_on"] - d["g"][f"R_{k}_off"] for d in draws], seed)
    r["Gamma"] = boot_mean([d["g"]["Gamma"] for d in draws], seed)
    if with_binding:
        for k in VERSIONS:
            r[f"B_{k}"] = boot_contrast([d["S"][f"B_{k}_on"] for d in draws], seed)
            r[f"dB_{k}"] = boot_contrast_diff([d["S"][f"B_{k}_on"] for d in draws], [d["S"][f"B_{k}_off"] for d in draws], seed)
        for k in ("raw", "c"):
            r[f"C_{k}"] = boot_contrast([d["S"][f"C_{k}_on"] for d in draws], seed)
            r[f"dC_{k}"] = boot_contrast_diff([d["S"][f"C_{k}_on"] for d in draws], [d["S"][f"C_{k}_off"] for d in draws], seed)
        r["Gb"] = boot_contrast([d["S"]["Gb"] for d in draws], seed)
    return r


# ---------------------------------------------------------------- toy systems, each run under three couplings
# mode "on": the coupling as given; "off": switched off; "common": its force on every token replaced by the
# cross-token mean of that force (the coupling's common push, without its token-specific part)
def _apply(F, mode):
    if mode == "off":
        return 0.0 * F
    if mode == "common":
        return np.broadcast_to(F.mean(0, keepdims=True), F.shape)
    return F

KC = 2 * np.sqrt(2 * np.pi) / np.pi          # Kuramoto threshold for unit-variance Gaussian frequencies

def kuramoto(K, rng, N=256, T=40.0, dt=0.05):
    th0 = rng.uniform(0, 2 * np.pi, N); om = rng.normal(0, 1, N)
    S2 = lambda a: np.stack([np.cos(a), np.sin(a)], 1)
    out = {}
    for mode in ("on", "off", "common"):
        def f(th):
            z = np.exp(1j * th).mean()
            return om + _apply(K * np.abs(z) * np.sin(np.angle(z) - th), mode)
        th = th0.copy()
        for _ in range(int(T / dt)):
            k1 = f(th); k2 = f(th + dt / 2 * k1); k3 = f(th + dt / 2 * k2); k4 = f(th + dt * k3)
            th = th + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[mode] = (S2(th), S2(th))      # (pre, post): no LayerNorm
        if mode == "on":
            r_on = float(np.abs(np.exp(1j * th).mean()))
    return out, (S2(th0), S2(th0)), r_on

def attention_flow(rng, N=128, d=16, n_groups=4, w_out=0.1, beta=1.0, dt=0.1, checkpoints=(1, 2, 4, 8)):
    """Causal attention with V = I on the sphere; groups enter only through the attention weights."""
    X0 = _unit(rng.normal(size=(N, d))); gid = rng.integers(0, n_groups, N)
    logw = np.where(gid[:, None] == gid[None, :], 0.0, np.log(w_out))
    mask = np.tril(np.ones((N, N), bool))
    snaps = {}
    for mode in ("on", "off", "common"):
        x = X0.copy(); t = 0.0
        for T in checkpoints:
            while t < T - 1e-9:
                L = np.where(mask, beta * x @ x.T + logw, -np.inf)
                A = np.exp(L - L.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
                v = A @ x
                v -= (v * x).sum(1, keepdims=True) * x
                x = _unit(x + dt * _apply(v, mode)); t += dt
            snaps[(mode, T)] = (x.copy(), x.copy())
    return snaps, (X0, X0), gid

def _shared_map(d, rng):
    W = rng.normal(0, 1.5 / np.sqrt(d), (d, d))
    return lambda h: np.tanh(h @ W.T) - h

def _layernorm(H):
    Z = H - H.mean(1, keepdims=True)
    return Z / np.maximum(Z.std(1, keepdims=True), 1e-12)

def clustered_step(rng, coupling, N=128, d=16, n_groups=4, T=2.0, dt=0.1, ln=False):
    """Planted clusters (similar inputs) evolved by a shared per-token map plus `coupling`, in the three modes.
    Returns {mode: (pre-LayerNorm state, state the next layer reads)}, the input in the same form, labels."""
    C = _unit(rng.normal(size=(n_groups, d)))
    gid = rng.integers(0, n_groups, N)
    H0 = C[gid] + rng.normal(0, 0.3 / np.sqrt(d), (N, d))
    f = _shared_map(d, rng); g = coupling(rng, d)
    post = _layernorm if ln else (lambda h: h)
    out = {}
    for mode in ("on", "off", "common"):
        h = H0.copy()
        for _ in range(int(T / dt)):
            h = h + dt * (f(h) + _apply(g(h), mode))
        out[mode] = (h, post(h))
    return out, (H0, post(H0)), gid

def shared_target_pull(rng, d, kappa=0.5):   # group-blind: toward one shared target, the mean of all tokens
    return lambda h: kappa * (h.mean(0, keepdims=True) - h)

def mean_field_pull(rng, d, kappa=0.5):      # causal: toward the running mean of earlier tokens (a different target per token)
    def g(h):
        m = np.cumsum(h, 0) / np.arange(1, len(h) + 1)[:, None]
        return kappa * (m - h)
    return g

def common_drift(rng, d, size=3.0):         # state-independent shared displacement
    v = size * _unit(rng.normal(size=d))
    return lambda h: np.broadcast_to(v, h.shape).copy()

def external_attractor(rng, d, kappa=0.5):  # a fixed point pulling every token, no interaction
    mu = _unit(rng.normal(size=d))
    return lambda h: kappa * (mu - h)


# ---------------------------------------------------------------- run-2 measures for one draw
def measures2(S, Sin, gid, pos):
    """S: {mode: (pre, post)}; Sin: (pre, post) of the input. Per-draw global scalars and binding bin stats."""
    g, B = {}, {}
    for ref, tag in (("off", "tot"), ("common", "spec")):
        G = contraction_matrix(S["on"][0], S[ref][0])
        g[f"Gamma_{tag}"] = coherence(G)
        g[f"R_{tag}"] = coherence(cos_matrix(centred(S["on"][1]))) - coherence(cos_matrix(centred(S[ref][1])))
        if gid is not None:
            B[f"Gb_{tag}"] = bin_stats(G, gid, pos)
    if gid is not None:
        for mode in ("on", "off", "common"):
            B[f"B_{mode}"] = bin_stats(cos_matrix(centred(S[mode][1])), gid, pos)
            B[f"C_{mode}"] = bin_stats(cos_matrix(centred(S[mode][0] - Sin[0])), gid, pos)
    return {"g": g, "S": B}

def summarise2(draws, seed=0):
    r = {}
    for k in ("Gamma_tot", "Gamma_spec", "R_tot", "R_spec"):
        r[k] = boot_mean([d["g"][k] for d in draws], seed)
    if draws[0]["S"]:
        St = lambda k: [d["S"][k] for d in draws]
        r["B_on"] = boot_contrast(St("B_on"), seed)
        r["C_on"] = boot_contrast(St("C_on"), seed)
        for ref, tag in (("off", "tot"), ("common", "spec")):
            r[f"B_{tag}"] = boot_contrast_diff(St("B_on"), St(f"B_{ref}"), seed)
            r[f"C_{tag}"] = boot_contrast_diff(St("C_on"), St(f"C_{ref}"), seed)
            r[f"Gb_{tag}"] = boot_contrast(St(f"Gb_{tag}"), seed)
    return r


# ---------------------------------------------------------------- SY0, run 2
def main():
    n_draws = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    lines = []; P = lambda s="": (print(s), lines.append(s))
    res = {"run": 3, "floor": FLOOR, "n_draws": n_draws, "checks": {}}
    verdict = {}
    def mark(m, check, ok):
        verdict.setdefault(m, []).append((check, bool(ok)))
    ATTR = ("tot", "spec")

    P(f"SY0 run 3: the SY instruments with the common-mode replay ({n_draws} independent draws per condition; 95% cluster-bootstrap intervals; floor {FLOOR})")
    P("   tot = on minus switch-off replay; spec = on minus common-mode replay (the coupling's force replaced by its cross-token mean)")
    P()

    # ---- check 1
    P(f"CHECK 1  Kuramoto, N = 256, unit-variance Gaussian frequencies, K_c = {KC:.3f}")
    P(f"   {'K/K_c':>6}  {'r on':>6}  {'Gamma_tot':>30}  {'Gamma_spec':>30}  {'R_spec (centred, on record)':>30}")
    c1 = {}
    for frac in (0.0, 0.25, 0.5, 0.75, 1.25, 2.0, 3.0):
        rng = np.random.default_rng(1000 + int(frac * 100))
        dr = []; ro = []
        for _ in range(n_draws):
            S, Sin, r_on = kuramoto(frac * KC, rng)
            dr.append(measures2(S, Sin, None, None)); ro.append(r_on)
        sm = summarise2(dr); c1[frac] = {**sm, "r_on": float(np.mean(ro))}
        P(f"   {frac:6.2f}  {np.mean(ro):6.3f}  {fmt(sm['Gamma_tot']):>30}  {fmt(sm['Gamma_spec']):>30}  {fmt(sm['R_spec']):>30}")
    for m in ("Gamma_tot", "Gamma_spec", "R_tot", "R_spec"):
        pts = [c1[f][m][0] for f in sorted(c1)]
        ok = (all(is_null(c1[f][m]) for f in (0.0, 0.25, 0.5)) and all(passes_floor(c1[f][m]) and c1[f][m][0] > 0 for f in (2.0, 3.0))
              and all(np.diff(pts) >= -1e-9))
        mark(m, "1", ok)
    res["checks"]["1_kuramoto"] = {str(k): v for k, v in c1.items()}
    P()

    # ---- check 2
    P("CHECK 2  causal attention, V = I, on S^15; N = 128, 4 groups planted only in the attention weights (within 1, between 0.1)")
    rng = np.random.default_rng(2000); runs = [attention_flow(rng) for _ in range(n_draws)]
    pos = np.arange(128); c2 = {}
    P(f"   {'T':>3}  {'B on':>30}  {'B_tot':>30}  {'B_spec':>30}  {'Gb_spec':>30}  {'C_spec (descr.)':>30}  {'Gamma_spec':>30}")
    for T in (1, 2, 4, 8):
        dr = [measures2({m: snaps[(m, T)] for m in ("on", "off", "common")}, Sin, g, pos) for snaps, Sin, g in runs]
        sm = summarise2(dr); c2[T] = sm
        P(f"   {T:3d}  {fmt(sm['B_on']):>30}  {fmt(sm['B_tot']):>30}  {fmt(sm['B_spec']):>30}  {fmt(sm['Gb_spec']):>30}  {fmt(sm['C_spec']):>30}  {fmt(sm['Gamma_spec']):>30}")
    rs = np.random.default_rng(2001)
    sh = summarise2([measures2({m: snaps[(m, 8)] for m in ("on", "off", "common")}, Sin, rs.permutation(g), pos) for snaps, Sin, g in runs])
    c2["shuffled_T8"] = sh
    P(f"   shuffled labels, T = 8:  B on {fmt(sh['B_on'])}   B_tot {fmt(sh['B_tot'])}   B_spec {fmt(sh['B_spec'])}   Gb_spec {fmt(sh['Gb_spec'])}")
    rising = lambda k: all(np.diff([c2[T][k][0] for T in (1, 2, 4, 8)]) > 0)
    for a in ATTR:
        mark(f"B_{a}", "2", passes_floor(c2[8]["B_on"]) and rising("B_on") and passes_floor(c2[8][f"B_{a}"]) and is_null(sh[f"B_{a}"]))
        mark(f"Gb_{a}", "2", passes_floor(c2[8][f"Gb_{a}"]) and rising(f"Gb_{a}") and is_null(sh[f"Gb_{a}"]))
    res["checks"]["2_attention"] = {str(k): v for k, v in c2.items()}
    P()

    # ---- checks 3, 4a, 4b at d = 16 and d = 384; check 5 descriptive
    specs = [("3", shared_target_pull, False, "planted clusters, shared per-token map, pull toward one shared target (group-blind)"),
             ("3'", mean_field_pull, False, "(descriptive) the same, pull toward each token's running mean (a different target per token)"),
             ("4a", common_drift, False, "common-mode drift (state-independent shared displacement, 3x the state scale)"),
             ("4b", common_drift, True, "common-mode drift, with a closing LayerNorm"),
             ("5", external_attractor, False, "(descriptive) a common external attractor, no interaction")]
    for tag, coup, ln, desc in specs:
        P(f"CHECK {tag}  {desc}")
        res["checks"][tag] = {}
        for d in (16, 384):
            rng = np.random.default_rng(3000 + 100 * int(tag[0]) + (7 if ln else 0) + d)
            dr = []
            for _ in range(n_draws):
                S, Sin, g = clustered_step(rng, coup, d=d, ln=ln)
                dr.append(measures2(S, Sin, g, np.arange(128)))
            sm = summarise2(dr); res["checks"][tag][str(d)] = sm
            P(f"   d = {d:<3}  B on {fmt(sm['B_on'])}   Gamma_tot {fmt(sm['Gamma_tot'])}   Gamma_spec {fmt(sm['Gamma_spec'])}")
            P(f"            B_tot {fmt(sm['B_tot'])}   B_spec {fmt(sm['B_spec'])}   Gb_tot {fmt(sm['Gb_tot'])}   Gb_spec {fmt(sm['Gb_spec'])}")
            P(f"            C_tot {fmt(sm['C_tot'])}   C_spec {fmt(sm['C_spec'])}   R_tot {fmt(sm['R_tot'])}   R_spec {fmt(sm['R_spec'])}")
            if tag in ("5", "3'"):
                continue
            ck = f"{tag}@{d}"
            for a in ATTR:
                if tag == "3":
                    mark(f"B_{a}", ck, passes_floor(sm["B_on"]) and is_null(sm[f"B_{a}"]))
                else:
                    mark(f"B_{a}", ck, is_null(sm[f"B_{a}"]))
                    mark(f"Gamma_{a}", ck, is_null(sm[f"Gamma_{a}"]))
                    mark(f"R_{a}", ck, is_null(sm[f"R_{a}"]))
                mark(f"Gb_{a}", ck, is_null(sm[f"Gb_{a}"]))
                mark(f"C_{a}", ck, is_null(sm[f"C_{a}"]))
        P()

    P("VERDICT per measure. Checks 3 and 4 decide at d = 384 (the models' width); d = 16 is shown for the record.")
    adm = {}
    for m, v in verdict.items():
        deciding = [(c, ok) for c, ok in v if not c.endswith("@16")]
        a = all(ok for _, ok in deciding)
        adm[m] = a
        P(f"   {m:<11} {'ADMITTED' if a else 'REJECTED':<9} " + "  ".join(f"{c}:{'pass' if ok else 'FAIL'}" for c, ok in v))
    res["verdict"] = {m: [list(x) for x in v] for m, v in verdict.items()}
    res["admitted"] = adm
    (HERE / "sy0_output.txt").write_text("\n".join(lines) + "\n")
    (HERE / "sy0.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"\nwrote {HERE / 'sy0_output.txt'} and sy0.json")


if __name__ == "__main__":
    main()
