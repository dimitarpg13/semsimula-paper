"""Markov-order regression with leave-one-sentence-out cross-validation.

This is **phase 2** of the first-order ODE rejection experiment, implementing
§4 (function class), §5 (LOSO + nested CV), and the per-quadruple residual
pipeline of the pre-registered protocol
(`docs/first_order_ODE_rejection_pre-registered_protocol.md`).

For each Markov order k in {1, 2, 3}, we predict h_{t+1} from
(h_t, h_{t-1}, ..., h_{t-k+1}) using the **primary** function class
(Gaussian RBF kernel ridge regression) preceded by a per-fold PCA projection
to dimension `p` (basis fit on the *training* fold's h_{t+1} values).

Outputs per LOSO fold:
- selected (alpha_k, gamma_k) per k from the inner 5-fold CV grid,
- per-quadruple squared residuals on the held-out sentence's quadruples.

Statistical decision rule (§6) is implemented in-line via `decide_outcome`:
  rho_12, rho_23 effect-size ratios, paired Wilcoxon, sentence-cluster bootstrap.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Singular-matrix and convergence warnings are expected for over-parameterised
# poly2 features at small alpha and for early-stopped MLPs. The Ridge
# pseudoinverse fallback is documented behaviour and not a numerical concern
# at the precision we report.
warnings.filterwarnings("ignore", message="Singular matrix in solving dual problem")
warnings.filterwarnings("ignore", message="Ill-conditioned matrix")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# -----------------------------------------------------------------------------
# Pre-registered hyperparameter grids (do not change without a DEVIATIONS.md note)
# -----------------------------------------------------------------------------

ALPHA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
GAMMA_HEURISTIC_MULTIPLIERS = [0.5, 1.0, 2.0]
INNER_FOLDS = 5
RNG_SEED = 42


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass
class FoldResult:
    held_out_sentence_idx: int
    n_test: int
    selected_alpha: dict   # {k: alpha}
    selected_gamma: dict   # {k: gamma}
    residuals_per_quad: dict  # {k: ndarray (n_test,) of ||h_{t+1} - F_k(...)||^2}


# -----------------------------------------------------------------------------
# Kernel + lag helpers
# -----------------------------------------------------------------------------


def median_pairwise_distance(X: np.ndarray, max_samples: int = 500, seed: int = 0) -> float:
    """Median pairwise euclidean distance, sub-sampled if needed for cost."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    diffs = Xs[:, None, :] - Xs[None, :, :]
    d = np.sqrt((diffs ** 2).sum(axis=-1))
    iu = np.triu_indices(d.shape[0], k=1)
    return float(np.median(d[iu]))


def gamma_from_median(med: float) -> float:
    """Convert median pairwise euclidean distance to RBF gamma.

    The convention sklearn uses: K(x, y) = exp(-gamma * ||x - y||^2). The
    "median heuristic" in this convention sets gamma = 1 / (2 * median^2).
    Multipliers in {0.5, 1, 2} thus correspond to bandwidths around the median.
    """
    return 1.0 / (2.0 * (med ** 2 + 1e-12))


def make_X(H_t: np.ndarray, H_tm1: np.ndarray, H_tm2: np.ndarray, k: int) -> np.ndarray:
    """Concatenate (h_t, h_{t-1}, ..., h_{t-k+1}) into shape (n, k*p)."""
    if k == 1:
        return H_t
    if k == 2:
        return np.concatenate([H_t, H_tm1], axis=1)
    if k == 3:
        return np.concatenate([H_t, H_tm1, H_tm2], axis=1)
    raise ValueError(f"unsupported k={k}")


# -----------------------------------------------------------------------------
# Inner CV: select (alpha, gamma) per k, given a fixed PCA basis
# -----------------------------------------------------------------------------


def _inner_cv_score(
    X: np.ndarray,
    Y: np.ndarray,
    sentence_idx: np.ndarray,
    alphas: list[float],
    gammas: list[float],
    n_folds: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    """Pick (alpha, gamma) minimising mean MSE across n_folds inner KFold splits.

    Splits are by **quadruple index** within the outer training set. We do not
    nest LOSO inside LOSO (would be too expensive at 50x49 folds); the outer
    LOSO already provides sentence-level independence for the *primary* test.
    Returns (best_alpha, best_gamma, mse_grid_shape (n_alphas, n_gammas)).
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    grid = np.zeros((len(alphas), len(gammas)))
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]
        for gi, gamma in enumerate(gammas):
            for ai, alpha in enumerate(alphas):
                kr = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
                kr.fit(X_tr, Y_tr)
                pred = kr.predict(X_va)
                mse = float(((Y_va - pred) ** 2).sum(axis=1).mean())
                grid[ai, gi] += mse / n_folds
    ai, gi = np.unravel_index(np.argmin(grid), grid.shape)
    return float(alphas[ai]), float(gammas[gi]), grid


# -----------------------------------------------------------------------------
# Per-fold work: PCA basis on training H_{t+1}, project all lags + target,
# inner CV, refit on full training, evaluate on held-out
# -----------------------------------------------------------------------------


def process_one_fold(
    held_out_sent: int,
    quads: dict,
    p: int,
    k_values: list[int],
    seed: int,
    use_kernel: bool = True,
    function_class: str | None = None,
    mlp_hidden: int = 128,
) -> FoldResult:
    """Run the LOSO-out-fold for one held-out sentence.

    function_class overrides use_kernel when set:
      - "kernel": Gaussian RBF kernel ridge (primary).
      - "linear": linear ridge with cross-validated alpha.
      - "poly2":  degree-2 polynomial ridge with cross-validated alpha.
      - "mlp":    2-layer MLP, fixed alpha 1e-4, early stopping.
    """
    if function_class is None:
        function_class = "kernel" if use_kernel else "linear"

    sentence_idx = quads["sentence_idx"]
    train_mask = sentence_idx != held_out_sent
    test_mask = ~train_mask

    H_tm2_tr, H_tm2_te = quads["H_tm2"][train_mask], quads["H_tm2"][test_mask]
    H_tm1_tr, H_tm1_te = quads["H_tm1"][train_mask], quads["H_tm1"][test_mask]
    H_t_tr, H_t_te = quads["H_t"][train_mask], quads["H_t"][test_mask]
    H_tp1_tr, H_tp1_te = quads["H_tp1"][train_mask], quads["H_tp1"][test_mask]

    # PCA basis fitted on training fold's h_{t+1}, applied to all four time steps.
    pca = PCA(n_components=p, random_state=seed)
    pca.fit(H_tp1_tr)
    Z_tm2_tr = pca.transform(H_tm2_tr)
    Z_tm1_tr = pca.transform(H_tm1_tr)
    Z_t_tr = pca.transform(H_t_tr)
    Z_tp1_tr = pca.transform(H_tp1_tr)
    Z_tm2_te = pca.transform(H_tm2_te)
    Z_tm1_te = pca.transform(H_tm1_te)
    Z_t_te = pca.transform(H_t_te)
    Z_tp1_te = pca.transform(H_tp1_te)

    selected_alpha: dict[int, float] = {}
    selected_gamma: dict[int, float] = {}
    residuals_per_quad: dict[int, np.ndarray] = {}

    for k in k_values:
        X_tr = make_X(Z_t_tr, Z_tm1_tr, Z_tm2_tr, k)
        X_te = make_X(Z_t_te, Z_tm1_te, Z_tm2_te, k)
        Y_tr = Z_tp1_tr
        Y_te = Z_tp1_te

        if function_class == "kernel":
            med = median_pairwise_distance(X_tr, max_samples=500, seed=seed)
            gamma_center = gamma_from_median(med)
            gammas = [m * gamma_center for m in GAMMA_HEURISTIC_MULTIPLIERS]
            sub_sent_tr = sentence_idx[train_mask]
            best_alpha, best_gamma, _ = _inner_cv_score(
                X_tr, Y_tr, sub_sent_tr, ALPHA_GRID, gammas, INNER_FOLDS, seed
            )
            model = KernelRidge(alpha=best_alpha, kernel="rbf", gamma=best_gamma)
            model.fit(X_tr, Y_tr)
            pred = model.predict(X_te)
            selected_alpha[k] = best_alpha
            selected_gamma[k] = best_gamma

        elif function_class == "linear":
            best_alpha = _select_alpha_ridge(X_tr, Y_tr, ALPHA_GRID, INNER_FOLDS, seed)
            model = Ridge(alpha=best_alpha, random_state=seed)
            model.fit(X_tr, Y_tr)
            pred = model.predict(X_te)
            selected_alpha[k] = best_alpha
            selected_gamma[k] = float("nan")

        elif function_class == "poly2":
            from sklearn.preprocessing import PolynomialFeatures

            poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
            X_tr_p = poly.fit_transform(X_tr)
            X_te_p = poly.transform(X_te)
            best_alpha = _select_alpha_ridge(X_tr_p, Y_tr, ALPHA_GRID, INNER_FOLDS, seed)
            model = Ridge(alpha=best_alpha, random_state=seed)
            model.fit(X_tr_p, Y_tr)
            pred = model.predict(X_te_p)
            selected_alpha[k] = best_alpha
            selected_gamma[k] = float("nan")

        elif function_class == "mlp":
            from sklearn.neural_network import MLPRegressor

            model = MLPRegressor(
                hidden_layer_sizes=(mlp_hidden,),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                early_stopping=True,
                validation_fraction=0.1,
                max_iter=1000,
                random_state=seed,
                n_iter_no_change=30,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, Y_tr)
            pred = model.predict(X_te)
            selected_alpha[k] = 1e-4
            selected_gamma[k] = float("nan")

        else:
            raise ValueError(f"unknown function_class={function_class}")

        residuals_per_quad[k] = ((Y_te - pred) ** 2).sum(axis=1)

    return FoldResult(
        held_out_sentence_idx=int(held_out_sent),
        n_test=int(Y_te.shape[0]),
        selected_alpha=selected_alpha,
        selected_gamma=selected_gamma,
        residuals_per_quad=residuals_per_quad,
    )


def _select_alpha_ridge(X, Y, alphas, n_folds, seed):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = np.zeros(len(alphas))
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]
        for ai, alpha in enumerate(alphas):
            r = Ridge(alpha=alpha).fit(X_tr, Y_tr)
            pred = r.predict(X_va)
            scores[ai] += float(((Y_va - pred) ** 2).sum(axis=1).mean()) / n_folds
    return float(alphas[int(np.argmin(scores))])


# -----------------------------------------------------------------------------
# Top-level LOSO driver
# -----------------------------------------------------------------------------


def run_loso(
    quads: dict,
    p: int,
    k_values: list[int],
    use_kernel: bool = True,
    function_class: str | None = None,
    n_jobs: int = -1,
    seed: int = RNG_SEED,
    verbose: int = 1,
) -> list[FoldResult]:
    sent_ids = sorted(set(quads["sentence_idx"].tolist()))
    fc = function_class or ("kernel" if use_kernel else "linear")
    if verbose:
        print(f"[loso] {len(sent_ids)} sentences  p={p}  "
              f"k_values={k_values}  function_class={fc}  n_jobs={n_jobs}")
    fold_results = Parallel(n_jobs=n_jobs, verbose=10 * verbose)(
        delayed(process_one_fold)(
            s, quads, p, k_values, seed, use_kernel, fc
        )
        for s in sent_ids
    )
    return fold_results


# -----------------------------------------------------------------------------
# Statistical decision (Wilcoxon + cluster bootstrap)
# -----------------------------------------------------------------------------


def aggregate_residuals(folds: list[FoldResult], k_values: list[int]) -> dict:
    """Concatenate per-quadruple residuals across folds, with sentence-id labels."""
    r = {k: [] for k in k_values}
    sids: list[int] = []
    for fr in folds:
        for k in k_values:
            r[k].append(fr.residuals_per_quad[k])
        sids.extend([fr.held_out_sentence_idx] * fr.n_test)
    return {
        "residuals": {k: np.concatenate(r[k]) for k in k_values},
        "sentence_idx": np.asarray(sids, dtype=np.int32),
    }


def cluster_bootstrap_diff(
    diff: np.ndarray,
    sentence_idx: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    """BCa-ish cluster bootstrap for mean diff (resample sentences with replacement).

    For simplicity we report the percentile 95 % CI rather than full BCa; with
    n_resamples = 10k and balanced clusters this is within rounding distance
    and avoids an extra implementation surface that could harbour bugs in the
    pre-registered cell.
    """
    rng = np.random.default_rng(seed)
    sids = np.array(sorted(set(sentence_idx.tolist())))
    sent_to_idx = {s: np.where(sentence_idx == s)[0] for s in sids}
    n_sent = len(sids)
    boot = np.empty(n_resamples)
    for b in range(n_resamples):
        sample_ids = rng.choice(sids, size=n_sent, replace=True)
        # Concatenate per-sentence residual diffs and take their mean.
        chunks = [diff[sent_to_idx[s]] for s in sample_ids]
        boot[b] = float(np.concatenate(chunks).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def decide_outcome(
    rho_12: float,
    p_12: float,
    rho_23: float,
    p_23: float,
) -> str:
    """Apply the §6.4 pre-registered decision matrix.

    A: first-order rejected, second-order sufficient.
    B: first-order rejected, second-order ALSO insufficient.
    C: first-order not rejected.
    D: inconclusive / boundary.
    """
    rejects_12 = (rho_12 >= 1.20) and (p_12 < 1e-3)
    fails_12 = (rho_12 < 1.10) or (p_12 > 0.01)
    if fails_12:
        return "C"
    if not rejects_12:
        return "D"
    # rejects 12; now check 23
    if rho_23 <= 1.05 and p_23 > 0.05:
        return "A"
    if rho_23 > 1.10 and p_23 < 0.05:
        return "B"
    return "D"


def summarise(
    folds: list[FoldResult],
    k_values: list[int],
    n_bootstrap: int = 10_000,
    seed: int = RNG_SEED,
) -> dict:
    agg = aggregate_residuals(folds, k_values)
    r = agg["residuals"]
    sids = agg["sentence_idx"]
    summary: dict = {"n_quadruples": int(sids.size), "k_values": list(k_values)}
    for k in k_values:
        summary[f"R{k}_mean"] = float(r[k].mean())
        summary[f"R{k}_sem"] = float(r[k].std(ddof=1) / np.sqrt(len(r[k])))
    if 1 in k_values and 2 in k_values:
        d12 = r[1] - r[2]
        rho_12 = float(r[1].mean() / r[2].mean())
        try:
            # Protocol §6.2 specifies two-sided. We additionally record the
            # two one-sided p-values for direction interpretation.
            w12_two = wilcoxon(r[1], r[2], alternative="two-sided", zero_method="wilcox")
            w12_gt = wilcoxon(r[1], r[2], alternative="greater", zero_method="wilcox")
            w12_lt = wilcoxon(r[1], r[2], alternative="less", zero_method="wilcox")
            p_12 = float(w12_two.pvalue)
            p_12_greater = float(w12_gt.pvalue)
            p_12_less = float(w12_lt.pvalue)
        except ValueError:
            p_12 = p_12_greater = p_12_less = float("nan")
        ci12 = cluster_bootstrap_diff(d12, sids, n_bootstrap, seed)
        summary["rho_12"] = rho_12
        summary["wilcoxon_p_12"] = p_12  # two-sided per §6.2
        summary["wilcoxon_p_12_one_sided_R1_gt_R2"] = p_12_greater
        summary["wilcoxon_p_12_one_sided_R1_lt_R2"] = p_12_less
        summary["bootstrap_ci_diff_R1_R2"] = list(ci12)
    if 2 in k_values and 3 in k_values:
        d23 = r[2] - r[3]
        rho_23 = float(r[2].mean() / r[3].mean())
        try:
            w23_two = wilcoxon(r[2], r[3], alternative="two-sided", zero_method="wilcox")
            w23_gt = wilcoxon(r[2], r[3], alternative="greater", zero_method="wilcox")
            w23_lt = wilcoxon(r[2], r[3], alternative="less", zero_method="wilcox")
            p_23 = float(w23_two.pvalue)
            p_23_greater = float(w23_gt.pvalue)
            p_23_less = float(w23_lt.pvalue)
        except ValueError:
            p_23 = p_23_greater = p_23_less = float("nan")
        ci23 = cluster_bootstrap_diff(d23, sids, n_bootstrap, seed)
        summary["rho_23"] = rho_23
        summary["wilcoxon_p_23"] = p_23  # two-sided per §6.2
        summary["wilcoxon_p_23_one_sided_R2_gt_R3"] = p_23_greater
        summary["wilcoxon_p_23_one_sided_R2_lt_R3"] = p_23_less
        summary["bootstrap_ci_diff_R2_R3"] = list(ci23)
    if "rho_12" in summary and "rho_23" in summary:
        summary["decision"] = decide_outcome(
            summary["rho_12"], summary["wilcoxon_p_12"],
            summary["rho_23"], summary["wilcoxon_p_23"],
        )
    return summary


# -----------------------------------------------------------------------------
# CLI: load quadruples, run primary cell, write outputs
# -----------------------------------------------------------------------------


def _load_quads(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return {
        "H_tm2": data["H_tm2"],
        "H_tm1": data["H_tm1"],
        "H_t": data["H_t"],
        "H_tp1": data["H_tp1"],
        "sentence_idx": data["sentence_idx"],
    }


def _save_residuals(folds: list[FoldResult], k_values: list[int], out: Path):
    payload: dict = {
        "n_folds": len(folds),
        "k_values": list(k_values),
        "folds": [
            {
                "held_out_sentence_idx": fr.held_out_sentence_idx,
                "n_test": fr.n_test,
                "selected_alpha": fr.selected_alpha,
                "selected_gamma": fr.selected_gamma,
            }
            for fr in folds
        ],
    }
    np.savez_compressed(
        out,
        meta=json.dumps(payload),
        sentence_idx=np.concatenate(
            [np.full(fr.n_test, fr.held_out_sentence_idx, dtype=np.int32) for fr in folds]
        ),
        **{
            f"residuals_k{k}": np.concatenate(
                [fr.residuals_per_quad[k] for fr in folds]
            )
            for k in k_values
        },
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quads", required=True, help="Path to quadruples.npz")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--p", type=int, default=50, help="PCA dim (default 50)")
    ap.add_argument("--k", default="1,2,3", help="Comma-sep k values, default 1,2,3")
    ap.add_argument("--class", dest="cls", default="kernel",
                    choices=["kernel", "linear"],
                    help="Function class (primary = kernel)")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--n_bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    quads = _load_quads(Path(args.quads))
    k_values = [int(x) for x in args.k.split(",")]
    use_kernel = (args.cls == "kernel")

    t0 = time.time()
    folds = run_loso(
        quads, p=args.p, k_values=k_values,
        use_kernel=use_kernel, n_jobs=args.n_jobs, seed=args.seed,
    )
    runtime = time.time() - t0

    summary = summarise(folds, k_values, n_bootstrap=args.n_bootstrap, seed=args.seed)
    summary["runtime_seconds"] = float(runtime)
    summary["p"] = args.p
    summary["function_class"] = args.cls
    summary["n_jobs"] = args.n_jobs

    print("\n[summary]")
    for k, v in summary.items():
        print(f"  {k} = {v}")

    with open(out / "primary_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _save_residuals(folds, k_values, out / "primary_residuals.npz")
    print(f"[done] runtime={runtime:.1f}s  decision={summary.get('decision')}")


if __name__ == "__main__":
    main()
