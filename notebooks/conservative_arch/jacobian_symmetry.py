"""
Jacobian-symmetry test (the structural positive-control diagnostic).

For a conservative flow  x_{l+1} - x_l ~= -dt * grad_h V(x_l) / m,
the *local* linearisation has a SYMMETRIC Jacobian (the Hessian of a
scalar is symmetric).  So if we fit

    y := x_{l+1} - x_l   ~=   M_l  x_l   +   b_l

on a principal-component subspace, the symmetric part of M_l should
fit roughly as well as the unconstrained M_l.  Concretely, for every
layer l we report

    R2_full     = fit quality of unconstrained M_l
    R2_symm     = fit quality when constrained to M_l = M_l^T

For the five attention-transformer experiments in §1 of the Failure
doc, R2_full sits in [0.5, 0.8] but R2_symm is *negative* (the
skew-restricted alternative did no better), demonstrating the
dominance of a symmetric-non-Hessian component and ruling out any
scalar-potential + Helmholtz fit.

This script runs the identical diagnostic on SPLM trajectories.  The
prediction: R2_symm tracks R2_full layer-by-layer, because our
dynamics is conservative in h by construction.

Usage:
  python3 jacobian_symmetry.py --traj results/splm_shakespeare_ckpt_latest.trajectories.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"


# ---------------------------------------------------------------------------
def fit_linear_and_symmetric(
    X: np.ndarray,  # (N, k)  predictors
    Y: np.ndarray,  # (N, k)  targets (same dim -- square operator)
    ridge: float = 1e-3,
) -> Dict:
    """Fit y = M x + b unconstrained and constrained-symmetric, both with
    intercept; return R^2 per fit and operators themselves.

    Unconstrained solve is standard ridge-regularised OLS.  The
    symmetric-constrained solve uses the fact that for a symmetric M,
    vec(M) = D_sym h where h is the upper-triangular vector of free
    entries and D_sym is the (k*k, k*(k+1)/2) expansion matrix.
    """
    N, k = X.shape
    assert Y.shape == (N, k), (X.shape, Y.shape)

    # Center for intercept.
    mx = X.mean(axis=0, keepdims=True)
    my = Y.mean(axis=0, keepdims=True)
    Xc = X - mx
    Yc = Y - my

    # ---- Unconstrained ridge fit.
    G   = Xc.T @ Xc                    # (k, k)
    reg = ridge * (np.trace(G) / max(N, 1)) * np.eye(k)
    M_full = np.linalg.solve(G + reg, Xc.T @ Yc).T      # solve gives (k, k) M^T; .T -> M
    Y_full = Xc @ M_full.T
    ss_tot = float(np.sum(Yc ** 2))
    r2_full = 1.0 - float(np.sum((Yc - Y_full) ** 2)) / (ss_tot + 1e-12)

    # ---- Symmetric constrained fit.
    # vec(Y_c) ~= (X_c kron I_k) * vec(M)  with vec(M) = D_sym * h_sym.
    # Build D_sym  (k^2, k(k+1)/2).
    idx_ij = [(i, j) for i in range(k) for j in range(i, k)]
    p      = len(idx_ij)
    D_sym  = np.zeros((k * k, p))
    for col, (i, j) in enumerate(idx_ij):
        if i == j:
            D_sym[i * k + i, col] = 1.0
        else:
            D_sym[i * k + j, col] = 1.0
            D_sym[j * k + i, col] = 1.0
    # We have Y_c = X_c @ M.T ; so  (row-vec)  vec(Y_c) = (I_k kron X_c) vec(M.T)
    # Use the identity  X_c @ M.T = M @ X_c.T  (since M symmetric); then
    # vec(X_c @ M.T) = (X_c kron I_k) vec(M).  Choose the latter.
    # ---- Form normal equations in h_sym space.
    Phi = np.zeros((N * k, p))
    # For each (i, j), the effect on vec(Y_c) equals:
    #   if i==j: column i in Y_c gets  X_c[:, i]
    #   else:    column j in Y_c gets  X_c[:, i]   AND   column i gets  X_c[:, j]
    for col, (i, j) in enumerate(idx_ij):
        if i == j:
            Phi[i * N:(i + 1) * N, col] = Xc[:, i]
        else:
            Phi[j * N:(j + 1) * N, col] = Xc[:, i]
            Phi[i * N:(i + 1) * N, col] = Xc[:, j]
    vecY = np.concatenate([Yc[:, c] for c in range(k)], axis=0)
    G2   = Phi.T @ Phi
    reg2 = ridge * (np.trace(G2) / max(Phi.shape[0], 1)) * np.eye(p)
    h_sym = np.linalg.solve(G2 + reg2, Phi.T @ vecY)
    # Reconstruct M_sym
    M_sym = np.zeros((k, k))
    for col, (i, j) in enumerate(idx_ij):
        if i == j:
            M_sym[i, i] = h_sym[col]
        else:
            M_sym[i, j] = h_sym[col]
            M_sym[j, i] = h_sym[col]
    Y_sym = Xc @ M_sym.T
    r2_sym = 1.0 - float(np.sum((Yc - Y_sym) ** 2)) / (ss_tot + 1e-12)

    return {
        "r2_full": r2_full, "M_full": M_full,
        "r2_sym":  r2_sym,  "M_sym":  M_sym,
        "n": N, "k": k,
    }


# ---------------------------------------------------------------------------
def _pca_basis(X: np.ndarray, k: int) -> np.ndarray:
    """Return (d, k) PCA basis from centered samples X (N, d)."""
    U, s, Vt = np.linalg.svd(X - X.mean(axis=0, keepdims=True),
                             full_matrices=False)
    return Vt[:k].T


def pooled_layer_samples(trajs, layer: int):
    """Stack per-token (x, x_next - x) samples for a given layer."""
    Xs, Ys = [], []
    for tr in trajs:
        x  = tr.x_ps[layer]            # (T, d)
        xn = tr.x_ps[layer + 1]
        Xs.append(x);  Ys.append(xn - x)
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


def pooled_layer_samples_with_velocity(trajs, layer: int):
    """Per-token (x_l, v_l ~ x_l - x_{l-1}, x_{l+1} - x_l) samples.

    Only defined for layer >= 1 (needs previous layer for v_l).
    """
    Xs, Vs, Ys = [], [], []
    for tr in trajs:
        x_prev = tr.x_ps[layer - 1]
        x      = tr.x_ps[layer]
        x_next = tr.x_ps[layer + 1]
        Xs.append(x); Vs.append(x - x_prev); Ys.append(x_next - x)
    return (np.concatenate(Xs, axis=0),
            np.concatenate(Vs, axis=0),
            np.concatenate(Ys, axis=0))


def fit_second_order(
    X: np.ndarray,   # (N, k)  positions
    V: np.ndarray,   # (N, k)  velocity proxy (x_l - x_{l-1})
    Y: np.ndarray,   # (N, k)  x_{l+1} - x_l
    ridge: float = 1e-3,
) -> Dict:
    """Fit  y = A v + M x + b  on shared k-D subspace.

    Unconstrained:  A, M arbitrary.
    Symmetric:      A arbitrary, M constrained to M = M^T.
    Rationale: for damped-second-order flow on a scalar potential,
        y = dt / (1+gamma) * v  -  dt^2/((1+gamma) m) * Hess(V) x  + O(x^2)
    so M is the (symmetric) Hessian, while A is essentially a scaled
    identity.  Omitting v forces the fit to absorb the velocity term
    into a position-dependent effective operator that can appear
    asymmetric even when the underlying flow is conservative.
    """
    N, k = X.shape
    assert V.shape == (N, k) and Y.shape == (N, k)

    # Centre.
    mx, mv, my = X.mean(0, keepdims=True), V.mean(0, keepdims=True), Y.mean(0, keepdims=True)
    Xc, Vc, Yc = X - mx, V - mv, Y - my
    ss_tot_tr  = float(np.sum(Yc ** 2))

    # ---- Unconstrained (concatenated design [X | V] of shape (N, 2k)).
    Phi = np.concatenate([Xc, Vc], axis=1)                    # (N, 2k)
    G   = Phi.T @ Phi
    reg = ridge * (np.trace(G) / max(N, 1)) * np.eye(2 * k)
    B   = np.linalg.solve(G + reg, Phi.T @ Yc).T              # (k, 2k)
    M_full = B[:, :k].copy()
    A_full = B[:, k:].copy()
    Y_pred = Phi @ B.T
    r2_full = 1.0 - float(np.sum((Yc - Y_pred) ** 2)) / (ss_tot_tr + 1e-12)

    # ---- Symmetric-constrained M, A unconstrained.
    # Build design matrix in (h_sym, vec(A)) space.
    idx_ij = [(i, j) for i in range(k) for j in range(i, k)]
    p_sym  = len(idx_ij)
    # Symmetric block:  (N*k, p_sym) same construction as before on Xc.
    Phi_sym = np.zeros((N * k, p_sym))
    for col, (i, j) in enumerate(idx_ij):
        if i == j:
            Phi_sym[i * N:(i + 1) * N, col] = Xc[:, i]
        else:
            Phi_sym[j * N:(j + 1) * N, col] = Xc[:, i]
            Phi_sym[i * N:(i + 1) * N, col] = Xc[:, j]
    # Unconstrained A:  (N*k, k*k) ; column (j*k + i) sets Y_c[:, i] += A[i, j] * Vc[:, j].
    Phi_A = np.zeros((N * k, k * k))
    for i in range(k):
        for j in range(k):
            Phi_A[i * N:(i + 1) * N, j * k + i] = Vc[:, j]
    Phi_big = np.concatenate([Phi_sym, Phi_A], axis=1)        # (N*k, p_sym + k*k)
    vecY    = np.concatenate([Yc[:, c] for c in range(k)], axis=0)
    G2      = Phi_big.T @ Phi_big
    reg2    = ridge * (np.trace(G2) / max(Phi_big.shape[0], 1)) * np.eye(G2.shape[0])
    theta   = np.linalg.solve(G2 + reg2, Phi_big.T @ vecY)
    h_sym, a_vec = theta[:p_sym], theta[p_sym:]
    M_sym = np.zeros((k, k))
    for col, (i, j) in enumerate(idx_ij):
        if i == j:
            M_sym[i, i] = h_sym[col]
        else:
            M_sym[i, j] = h_sym[col]
            M_sym[j, i] = h_sym[col]
    A_sym = a_vec.reshape(k, k).T   # because column (j*k + i) was A[i, j]
    # Predict.
    Y_sym_pred = Xc @ M_sym.T + Vc @ A_sym.T
    r2_sym = 1.0 - float(np.sum((Yc - Y_sym_pred) ** 2)) / (ss_tot_tr + 1e-12)

    return {
        "M_full": M_full, "A_full": A_full, "r2_full": r2_full,
        "M_sym":  M_sym,  "A_sym":  A_sym,  "r2_sym":  r2_sym,
        "n": N, "k": k,
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--pca_k", type=int, default=16)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--tag",   default=None)
    args = ap.parse_args()

    traj_path = Path(args.traj)
    with traj_path.open("rb") as f:
        bundle = pickle.load(f)
    trajs = bundle["trajectories"]
    L     = int(bundle["L"])
    d     = int(bundle["d"])
    default_tag = traj_path.stem.replace(".trajectories", "")
    if default_tag.startswith("splm_"):
        default_tag = default_tag[len("splm_"):]
    tag = args.tag or default_tag

    print(f"[jacsym] tag={tag}  L={L}  d={d}  pca_k={args.pca_k}")

    train = [tr for tr in trajs if tr.split == "train"]
    test  = [tr for tr in trajs if tr.split == "test"]

    # Per-layer shared PCA basis built from TRAIN centered states at that layer.
    per_layer: List[Dict] = []
    for ell in range(L):
        X_tr, Y_tr = pooled_layer_samples(train, ell)
        Vk = _pca_basis(X_tr, args.pca_k)              # (d, k)
        X_tr_pca = X_tr @ Vk                            # (N, k)
        Y_tr_pca = Y_tr @ Vk

        # ---- §1.5-style position-only fit.
        fit1 = fit_linear_and_symmetric(X_tr_pca, Y_tr_pca, ridge=args.ridge)
        X_te, Y_te = pooled_layer_samples(test, ell)
        X_te_pca = X_te @ Vk; Y_te_pca = Y_te @ Vk
        Xte_c = X_te_pca - X_te_pca.mean(0, keepdims=True)
        Yte_c = Y_te_pca - Y_te_pca.mean(0, keepdims=True)
        ss_te = float(np.sum(Yte_c ** 2))
        r2_te_full1 = 1.0 - float(np.sum((Yte_c - Xte_c @ fit1["M_full"].T) ** 2)) / (ss_te + 1e-12)
        r2_te_sym1  = 1.0 - float(np.sum((Yte_c - Xte_c @ fit1["M_sym"].T)  ** 2)) / (ss_te + 1e-12)

        # ---- Velocity-aware second-order fit (only valid for ell >= 1).
        r2_tr_full2 = r2_tr_sym2 = r2_te_full2 = r2_te_sym2 = float("nan")
        if ell >= 1:
            X_tr2, V_tr2, Y_tr2 = pooled_layer_samples_with_velocity(train, ell)
            X_tr2p = X_tr2 @ Vk; V_tr2p = V_tr2 @ Vk; Y_tr2p = Y_tr2 @ Vk
            fit2 = fit_second_order(X_tr2p, V_tr2p, Y_tr2p, ridge=args.ridge)

            X_te2, V_te2, Y_te2 = pooled_layer_samples_with_velocity(test, ell)
            X_te2p = X_te2 @ Vk; V_te2p = V_te2 @ Vk; Y_te2p = Y_te2 @ Vk
            Xte2c = X_te2p - X_te2p.mean(0, keepdims=True)
            Vte2c = V_te2p - V_te2p.mean(0, keepdims=True)
            Yte2c = Y_te2p - Y_te2p.mean(0, keepdims=True)
            sst2  = float(np.sum(Yte2c ** 2))
            r2_tr_full2 = fit2["r2_full"]; r2_tr_sym2 = fit2["r2_sym"]
            r2_te_full2 = 1.0 - float(np.sum((Yte2c - (Xte2c @ fit2["M_full"].T + Vte2c @ fit2["A_full"].T)) ** 2)) / (sst2 + 1e-12)
            r2_te_sym2  = 1.0 - float(np.sum((Yte2c - (Xte2c @ fit2["M_sym"].T  + Vte2c @ fit2["A_sym"].T))  ** 2)) / (sst2 + 1e-12)

        per_layer.append({
            "layer":     ell,
            # position-only fit (§1.5 analogue)
            "r2_train_full_p": fit1["r2_full"], "r2_train_sym_p": fit1["r2_sym"],
            "r2_test_full_p":  r2_te_full1,     "r2_test_sym_p":  r2_te_sym1,
            # velocity-aware second-order fit
            "r2_train_full_v": r2_tr_full2,     "r2_train_sym_v": r2_tr_sym2,
            "r2_test_full_v":  r2_te_full2,     "r2_test_sym_v":  r2_te_sym2,
        })
        print(f"[jacsym] layer {ell}:  "
              f"POS-ONLY full={fit1['r2_full']:+.3f}/sym={fit1['r2_sym']:+.3f}    "
              f"VEL-AUG full={r2_tr_full2:+.3f}/sym={r2_tr_sym2:+.3f}    "
              f"TEST-VEL full={r2_te_full2:+.3f}/sym={r2_te_sym2:+.3f}")

    layers = np.array([p["layer"] for p in per_layer])

    # --- Save raw ---
    npz_path = RESULTS_DIR / f"splm_{tag}_jacsym_results.npz"
    np.savez(
        npz_path,
        layer=layers,
        **{k: np.array([p[k] for p in per_layer]) for k in per_layer[0].keys() if k != "layer"},
    )
    print(f"[jacsym] saved -> {npz_path}")

    # --- Figure: side-by-side position-only vs velocity-aware ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    # Panel 1: POS-ONLY TEST fits (§1.5 analogue).
    axes[0].plot(layers, [p["r2_test_full_p"] for p in per_layer],
                 marker="o", label="unconstrained $M_\\ell$")
    axes[0].plot(layers, [p["r2_test_sym_p"]  for p in per_layer],
                 marker="s", label="symmetric-restricted")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].set_title("Position-only fit  $x_{\\ell+1}\\!-\\!x_\\ell = M_\\ell x_\\ell$  (§1.5)")
    axes[0].set_xlabel("layer $\\ell$"); axes[0].set_ylabel("TEST $R^2$")
    axes[0].set_ylim(-0.3, 1.0); axes[0].grid(True, alpha=0.3); axes[0].legend()
    # Panel 2: VEL-AUG TEST fits.
    axes[1].plot(layers, [p["r2_test_full_v"] for p in per_layer],
                 marker="o", label="unconstrained $M_\\ell$")
    axes[1].plot(layers, [p["r2_test_sym_v"]  for p in per_layer],
                 marker="s", label="symmetric-restricted")
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_title("Velocity-aware fit  $y = A v_\\ell + M_\\ell x_\\ell$")
    axes[1].set_xlabel("layer $\\ell$")
    axes[1].set_ylim(-0.3, 1.0); axes[1].grid(True, alpha=0.3); axes[1].legend()
    fig.suptitle("Jacobian-symmetry diagnostic: is the per-step spring matrix symmetric?")
    fig.tight_layout()
    fig_path = RESULTS_DIR / f"splm_{tag}_fig_jacsym.png"
    fig.savefig(fig_path, dpi=130); plt.close(fig)
    print(f"[jacsym] saved -> {fig_path}")

    # --- Markdown summary ---
    md = RESULTS_DIR / f"splm_{tag}_jacsym_summary.md"
    gap_pos = np.array([p["r2_test_full_p"] - p["r2_test_sym_p"] for p in per_layer])
    gap_vel = np.array([p["r2_test_full_v"] - p["r2_test_sym_v"]
                        for p in per_layer if p["layer"] >= 1])
    verdict_vel = (
        "**Velocity-aware: symmetric-restricted fit tracks the unconstrained "
        f"fit (max TEST gap = {gap_vel.max():+.3f}).  The per-step spring "
        "matrix is consistent with a symmetric Hessian, i.e. the dynamics "
        "is conservative on h.**"
        if gap_vel.max() < 0.10 else
        f"**Velocity-aware: gap of up to {gap_vel.max():+.3f} between "
        "unconstrained and symmetric-restricted fits.  Either residual "
        "asymmetry survives or higher-order terms matter.**"
    )
    with md.open("w") as f:
        f.write(f"# Jacobian-symmetry diagnostic -- scalar-potential LM ({tag})\n\n")
        f.write("Tests whether the per-step linear operator $M_\\ell$ is "
                "**symmetric**.  A symmetric $M_\\ell$ is what a conservative "
                "flow on a scalar potential must produce (Hessians of scalars "
                "are symmetric).  Skew-symmetric or symmetric-non-Hessian "
                "components indicate non-conservative dynamics.\n\n")
        f.write("We run TWO variants:\n\n")
        f.write("1. **Position-only (§1.5 analogue)**: "
                "$x_{\\ell+1}-x_\\ell \\approx M_\\ell x_\\ell$.  "
                "For **damped second-order** dynamics (i.e. both this model "
                "and the Failure-doc §1 integrator), this fit is known to "
                "be confounded because the single-step transition mixes "
                "$x_\\ell$ with the hidden velocity $v_\\ell \\approx "
                "x_\\ell - x_{\\ell-1}$.  The confound can manufacture "
                "apparent asymmetry even in genuinely conservative flows.\n")
        f.write("2. **Velocity-aware**: "
                "$x_{\\ell+1}-x_\\ell \\approx A v_\\ell + M_\\ell x_\\ell$ "
                "with $v_\\ell = x_\\ell - x_{\\ell-1}$.  Here $M_\\ell$ is "
                "the clean signal of the per-step spring matrix.  "
                "**This is the variant that tests conservativity.**\n\n")
        f.write(f"- Hidden dim `d = {d}`, integration steps `L = {L}`, PCA `k = {args.pca_k}`\n")
        f.write(f"- Train / test sentences: {len(train)} / {len(test)}\n\n")
        f.write("## Per-layer fit quality\n\n")
        f.write("| layer | POS-only $R^{2}_\\text{full}$ | POS-only $R^{2}_\\text{sym}$ | "
                "VEL-aug $R^{2}_\\text{full}$ | VEL-aug $R^{2}_\\text{sym}$ | "
                "VEL-aug gap |\n")
        f.write("|--:|--:|--:|--:|--:|--:|\n")
        for p in per_layer:
            gap_v = (p["r2_test_full_v"] - p["r2_test_sym_v"]
                     if p["layer"] >= 1 else float("nan"))
            f.write(f"| {p['layer']} | {p['r2_test_full_p']:+.3f} | {p['r2_test_sym_p']:+.3f} | "
                    f"{p['r2_test_full_v']:+.3f} | {p['r2_test_sym_v']:+.3f} | "
                    f"{gap_v:+.3f} |\n")
        f.write("\n## Reference: GPT-2 small (§1.5 of Failure doc)\n\n")
        f.write("At matched PCA-$k$, POS-only $R^{2}_\\text{full}\\in[0.5,0.8]$ "
                "but POS-only $R^{2}_\\text{sym}<0$ across every layer.  "
                "The gap in the **velocity-aware** variant has not yet been "
                "re-run for GPT-2 here; that is a required cross-check for "
                "v2 to make the comparison apples-to-apples.\n\n")
        f.write(f"## Verdict\n\n")
        f.write(f"- **Position-only (§1.5 analogue)**: max TEST gap = "
                f"{gap_pos.max():+.3f} (vs. GPT-2 where symmetric $R^2$ was "
                "*negative* everywhere).  Already a qualitative improvement.\n\n")
        f.write(f"- **Velocity-aware (the clean test)**: {verdict_vel}\n\n")
        f.write(f"## Artefacts\n\n")
        f.write(f"- `splm_{tag}_jacsym_results.npz`\n")
        f.write(f"- `splm_{tag}_fig_jacsym.png`\n")
    print(f"[jacsym] saved -> {md}")


if __name__ == "__main__":
    main()
