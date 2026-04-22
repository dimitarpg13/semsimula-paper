"""
E-init validation for the scalar-potential conservative-by-construction LM.

Applies the same Gaussian-well + damped symplectic-Euler pipeline as the
Failure-doc §1.1 experiment (`notebooks/e_init/e_init_validation.ipynb`)
and its follow-ups, but to trajectories extracted from our
scalar-potential LM rather than from GPT-2.

The central question:  on the same test distribution, same metric
(median layer-L relative residual), same gamma sweep, does a
conservative-by-construction circuit admit a scalar-potential fit that
beats the static-null baseline -- or does it show the same null-floor
collapse that every attention-transformer fit did in §1?

Usage:
  python3 e_init_validation.py --traj results/splm_smoke_ckpt_latest.trajectories.pkl

Outputs:
  results/splm_<tag>_e_init_summary.md
  results/splm_<tag>_e_init_results.npz
  results/splm_<tag>_fig_residual_vs_gamma.png
  results/splm_<tag>_fig_residual_vs_layer.png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Gaussian well: V(r) = a * (1 - exp(-b r^2))
# ---------------------------------------------------------------------------
def gaussian_well(r: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * (1.0 - np.exp(-b * r ** 2))


def fit_well_for_layer(x_pool: np.ndarray, e_pool: np.ndarray) -> Dict:
    """Fit Gaussian well V(r) = a(1 - exp(-b r^2)) to (r, energy) data."""
    r = np.linalg.norm(x_pool, axis=1)
    mask = np.isfinite(r) & np.isfinite(e_pool)
    r, e = r[mask], e_pool[mask]
    if len(r) < 10:
        return {"a": 0.0, "b": 0.0, "r2": -np.inf, "n": len(r)}
    p0 = [max(e.max(), 1e-3), 1.0 / (r.std() ** 2 + 1e-8)]
    try:
        popt, _ = curve_fit(gaussian_well, r, e, p0=p0, maxfev=20000)
        pred = gaussian_well(r, *popt)
        ss_res = np.sum((e - pred) ** 2)
        ss_tot = np.sum((e - e.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        return {"a": float(popt[0]), "b": float(popt[1]),
                "r2": float(r2), "n": len(r)}
    except RuntimeError:
        return {"a": 0.0, "b": 0.0, "r2": -np.inf, "n": len(r)}


# ---------------------------------------------------------------------------
# Integrator: same damped symplectic Euler used in all §1 experiments.
# ---------------------------------------------------------------------------
def symplectic_step(x: np.ndarray, v: np.ndarray,
                    a_well: float, b_well: float, m: float,
                    gamma: float, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    r2 = float(np.dot(x, x))
    f_over_m = -2.0 * (a_well / max(m, 1e-8)) * b_well * x * np.exp(-b_well * r2)
    v_new = (v + dt * f_over_m) / (1.0 + dt * gamma)
    x_new = x + dt * v_new
    return x_new, v_new


def integrate(x0: np.ndarray, v0: np.ndarray, m: float,
              well_params: Dict[int, Dict], L: int,
              gamma: float, dt: float = 1.0,
              r2_gate: float = 0.02) -> np.ndarray:
    x, v = x0.copy(), v0.copy()
    traj = [x.copy()]
    for ell in range(1, L + 1):
        p = well_params[ell]
        a_w, b_w = (p["a"], p["b"]) if p["r2"] > r2_gate else (0.0, 0.0)
        x, v = symplectic_step(x, v, a_w, b_w, m, gamma, dt)
        traj.append(x.copy())
    return np.stack(traj, axis=0)


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------
def residuals_from_fit(trajs, well_params: Dict[int, Dict],
                       L: int, gamma: float) -> np.ndarray:
    """Median layer-L relative residual per token, stacked."""
    out = []
    for tr in trajs:
        mu_ps = tr.mu_ps[:, 0, :]
        for ti in range(tr.hs.shape[1]):
            x0 = tr.x_ps[0, ti, :].astype(np.float64)
            v0 = tr.x_ps[1, ti, :] - tr.x_ps[0, ti, :]
            v0 = v0.astype(np.float64)
            m = 1.0     # uniform mass (see trajectory_extraction.py)
            pred_x = integrate(x0, v0, m, well_params, L, gamma)
            pred_h = pred_x + mu_ps
            obs_h  = tr.hs[:, ti, :]
            denom  = np.linalg.norm(obs_h, axis=1) + 1e-8
            out.append(np.linalg.norm(pred_h - obs_h, axis=1) / denom)
    return np.stack(out, axis=0)


def residuals_static(trajs, L: int) -> np.ndarray:
    """Residual of the 'freeze at h_0' baseline."""
    out = []
    for tr in trajs:
        mu_ps = tr.mu_ps[:, 0, :]
        for ti in range(tr.hs.shape[1]):
            x0 = tr.x_ps[0, ti, :]
            pred_h = np.tile(x0, (L + 1, 1)) + mu_ps
            obs_h  = tr.hs[:, ti, :]
            denom  = np.linalg.norm(obs_h, axis=1) + 1e-8
            out.append(np.linalg.norm(pred_h - obs_h, axis=1) / denom)
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True,
                    help="pickle from trajectory_extraction.py")
    ap.add_argument("--tag", default=None,
                    help="label used in output filenames (defaults to traj stem)")
    args = ap.parse_args()

    traj_path = Path(args.traj)
    with traj_path.open("rb") as f:
        bundle = pickle.load(f)
    trajs      = bundle["trajectories"]
    L          = int(bundle["L"])
    d          = int(bundle["d"])
    default_tag = traj_path.stem.replace(".trajectories", "")
    if default_tag.startswith("splm_"):
        default_tag = default_tag[len("splm_"):]
    tag = args.tag or default_tag

    print(f"[e_init] tag={tag}   L={L}   d={d}   n_sentences={len(trajs)}")

    train_traj = [tr for tr in trajs if tr.split == "train"]
    test_traj  = [tr for tr in trajs if tr.split == "test"]
    print(f"[e_init] train={len(train_traj)}   test={len(test_traj)}")

    # --- Per-layer Gaussian well fit on TRAIN ---
    well_params: Dict[int, Dict] = {}
    for ell in range(1, L + 1):
        x_pool = np.concatenate([tr.x_ps[ell, :-1, :] for tr in train_traj
                                 if tr.hs.shape[1] > 1], axis=0)
        e_pool = np.concatenate([tr.ptl for tr in train_traj
                                 if tr.hs.shape[1] > 1], axis=0)
        well_params[ell] = fit_well_for_layer(x_pool, e_pool)
        print(f"[e_init]  well layer {ell}: a={well_params[ell]['a']:.3g}  "
              f"b={well_params[ell]['b']:.3g}  R^2={well_params[ell]['r2']:.3f}  "
              f"n={well_params[ell]['n']}")

    # --- Static-null baseline ---
    rho_static_train = residuals_static(train_traj, L)
    rho_static_test  = residuals_static(test_traj,  L)
    STATIC_TRAIN = float(np.median(rho_static_train[:, -1]))
    STATIC_TEST  = float(np.median(rho_static_test[:, -1]))
    print(f"[e_init] static null:  TRAIN {STATIC_TRAIN:.4f}   TEST {STATIC_TEST:.4f}")

    # --- Gamma sweep of the damped E-init integrator ---
    gammas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    results: List[Dict] = []
    for g in gammas:
        rho_tr = residuals_from_fit(train_traj, well_params, L, g)
        rho_te = residuals_from_fit(test_traj,  well_params, L, g)
        med_tr = float(np.median(rho_tr[:, -1]))
        med_te = float(np.median(rho_te[:, -1]))
        results.append({
            "gamma": g, "train": med_tr, "test": med_te,
            "per_layer_train": np.median(rho_tr, axis=0),
            "per_layer_test":  np.median(rho_te, axis=0),
        })
        print(f"[e_init] gamma={g:>4.2f}   TRAIN {med_tr:.4f}   TEST {med_te:.4f}"
              f"   (null: {STATIC_TRAIN:.4f}/{STATIC_TEST:.4f})")

    # --- Save raw ---
    npz_path = RESULTS_DIR / f"splm_{tag}_e_init_results.npz"
    np.savez(
        npz_path,
        gammas=np.array(gammas),
        train=np.array([r["train"] for r in results]),
        test =np.array([r["test"]  for r in results]),
        per_layer_train=np.stack([r["per_layer_train"] for r in results]),
        per_layer_test =np.stack([r["per_layer_test"]  for r in results]),
        static_train=np.array([STATIC_TRAIN]),
        static_test =np.array([STATIC_TEST]),
        well_a=np.array([well_params[l]["a"]  for l in range(1, L + 1)]),
        well_b=np.array([well_params[l]["b"]  for l in range(1, L + 1)]),
        well_r2=np.array([well_params[l]["r2"] for l in range(1, L + 1)]),
    )
    print(f"[e_init] saved -> {npz_path}")

    # --- Figure A: residual vs gamma ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key, static in zip(axes, ["train", "test"], [STATIC_TRAIN, STATIC_TEST]):
        ys = [r[key] for r in results]
        ax.plot(gammas, ys, marker="o", label="Gaussian-well E-init")
        ax.axhline(static, linestyle="--", color="tab:gray",
                   label=f"static null ({static:.3f})")
        ax.set_xlabel(r"damping $\gamma$")
        ax.set_title(f"median layer-L residual -- {key.upper()}")
        ax.grid(True, alpha=0.3); ax.legend()
    axes[0].set_ylabel("median residual")
    fig.tight_layout()
    fig_a = RESULTS_DIR / f"splm_{tag}_fig_residual_vs_gamma.png"
    fig.savefig(fig_a, dpi=130)
    plt.close(fig)
    print(f"[e_init] saved -> {fig_a}")

    # --- Figure B: per-layer residual at gamma* (best TEST) ---
    best_idx = int(np.argmin([r["test"] for r in results]))
    best_g   = gammas[best_idx]
    fig, ax  = plt.subplots(figsize=(8, 4.5))
    layers   = np.arange(L + 1)
    ax.plot(layers, results[best_idx]["per_layer_test"],
            marker="o", label=f"Gaussian-well E-init  (γ*={best_g})")
    ax.plot(layers, np.median(rho_static_test, axis=0),
            linestyle="--", color="tab:gray", label="static null")
    ax.set_xlabel(r"layer $\ell$")
    ax.set_ylabel("median relative residual")
    ax.set_title(r"Per-layer TEST residual at $\gamma^{*}$")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig_b = RESULTS_DIR / f"splm_{tag}_fig_residual_vs_layer.png"
    fig.savefig(fig_b, dpi=130)
    plt.close(fig)
    print(f"[e_init] saved -> {fig_b}")

    # --- Summary ---
    md_path = RESULTS_DIR / f"splm_{tag}_e_init_summary.md"
    best = results[best_idx]
    delta_te = best["test"] - STATIC_TEST
    verdict  = (
        "**Positive result: conservative E-init beats static null on held-out data.**"
        if delta_te < -0.005 else
        "**Matches static null (no clean scalar-potential fit).**"
        if abs(delta_te) <= 0.005 else
        "**Negative result: scalar-potential fit is worse than static null.**"
    )
    with md_path.open("w") as f:
        f.write(f"# E-init validation -- scalar-potential LM ({tag})\n\n")
        f.write("Same protocol as `notebooks/e_init/` §1 experiments, but on "
                "trajectories of a conservative-by-construction circuit rather "
                "than GPT-2.\n\n")
        f.write(f"- Hidden dim `d = {d}`; integration steps `L = {L}`\n")
        f.write(f"- Corpus: 40 train / 10 test sentences "
                f"(same as `notebooks/e_init/e_init_corpus.py`)\n")
        f.write(f"- Tokens: {sum(tr.hs.shape[1] for tr in trajs)} across {len(trajs)} sentences\n\n")
        f.write("## Gaussian-well fit quality (TRAIN)\n\n")
        f.write("| layer | a | b | $R^{2}$ | n |\n|--:|--:|--:|--:|--:|\n")
        for l in range(1, L + 1):
            p = well_params[l]
            f.write(f"| {l} | {p['a']:.3g} | {p['b']:.3g} | {p['r2']:.3f} | {p['n']} |\n")
        f.write("\n## TRAIN / TEST residual vs damping\n\n")
        f.write("| $\\gamma$ | TRAIN | TEST |\n|--:|--:|--:|\n")
        for r in results:
            f.write(f"| {r['gamma']} | {r['train']:.4f} | {r['test']:.4f} |\n")
        f.write(f"\nStatic-null baseline: TRAIN {STATIC_TRAIN:.4f}, TEST {STATIC_TEST:.4f}.\n\n")
        f.write(f"## Verdict\n\n")
        f.write(f"Best TEST residual: {best['test']:.4f} at $\\gamma^{{*}} = {best_g}$  "
                f"(Δ vs null = {delta_te:+.4f}).\n\n")
        f.write(f"{verdict}\n\n")
        f.write("Comparison reference: every attention-transformer fit in §1 "
                "of the Failure doc matched the static null on held-out data "
                "(Δ ≈ 0 at best). A negative Δ here of order $-0.01$ or more "
                "would constitute the quantitative positive control that v2 "
                "needs.\n\n")
        f.write(f"## Artefacts\n\n")
        f.write(f"- `splm_{tag}_e_init_results.npz`\n")
        f.write(f"- `splm_{tag}_fig_residual_vs_gamma.png`\n")
        f.write(f"- `splm_{tag}_fig_residual_vs_layer.png`\n")
    print(f"[e_init] saved -> {md_path}")


if __name__ == "__main__":
    main()
