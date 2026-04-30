"""E5 LN damping sweep — per-cell diagnostic driver.

Identical pipeline to damping_sweep/run_diagnostics.py but adapted for the
em_ln (LayerNorm-after-step) model:

  1. Energy-state extraction using --variant em_ln.
  2. Energy-drift summary (drift slope + oscillation bandwidth).
  3. LN-hidden quadruple extraction (extract_ln_quadruples.py).
  4. Markov-order regression primary cell (PCA-50, kernel-ridge LOSO).
  5. §14 acceleration statistics on per-token last-layer hidden states.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT_DIR         = Path(__file__).resolve().parent
PARENT_DIR         = SCRIPT_DIR.parent
SARF_MASS_DIR      = PARENT_DIR / "sarf_mass_variant"
ENERGY_DRIFT_DIR   = PARENT_DIR / "energy_drift"
DYNAMICS_ORDER_DIR = PARENT_DIR.parent / "dynamics_order_test"

DEFAULT_TAGS = (
    "gamma0p00", "gamma0p10", "gamma0p30",
    "gamma0p85", "gamma2p00", "gamma5p00",
)
DEFAULT_LOGFREQ = SARF_MASS_DIR / "results" / "logfreq_surprisal.npy"


def _run(cmd: list[str], cwd: Optional[Path] = None) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    rc = subprocess.call(cmd, cwd=cwd)
    if rc != 0:
        print(f"!! command exited with status {rc}")
    return rc


def _ckpt_path(cell_dir: Path, tag: str) -> Path:
    return cell_dir / f"splm_em_ln_shakespeare_{tag}_ckpt_latest.pt"


def _energy_drift_summary(npz_path: Path) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    H = z["H"]
    K = z["kinetic"]
    V = z["potential"]
    L = H.shape[1] - 1
    layers = np.arange(L + 1) / max(L, 1)

    H_mean = H.mean(axis=0)
    H0 = float(np.abs(H_mean[0]))
    H_norm = H_mean - H_mean[0]
    if H0 > 0:
        H_norm = H_norm / H0

    x = layers
    y = H_norm
    x_c = x - x.mean()
    y_c = y - y.mean()
    sxx = float((x_c ** 2).sum())
    slope = float((x_c * y_c).sum() / sxx) if sxx > 0 else 0.0

    if H_mean.size >= 5:
        win = 3
        roll = np.array([
            H_mean[max(0, i - win): min(H_mean.size, i + win + 1)].std()
            for i in range(H_mean.size)
        ])
    else:
        roll = np.array([H_mean.std()])
    bandwidth = float(roll.mean() / max(H0, 1e-9))

    return {
        "L": int(L),
        "H0": float(H_mean[0]),
        "HL": float(H_mean[-1]),
        "delta_H": float(H_mean[-1] - H_mean[0]),
        "drift_slope_per_normalised_layer": slope,
        "drift_slope_per_layer": slope / max(L, 1),
        "rolling_std_mean_normalised": bandwidth,
        "kinetic_layer0_mean": float(K.mean(axis=0)[0]),
        "kinetic_layerL_mean": float(K.mean(axis=0)[-1]),
        "potential_layer0_mean": float(V.mean(axis=0)[0]),
        "potential_layerL_mean": float(V.mean(axis=0)[-1]),
    }


def _acceleration_stats(traj_npz_path: Path,
                        n_perm: int = 50,
                        seed: int = 0) -> dict:
    z = np.load(traj_npz_path, allow_pickle=True)
    H = z["H"]
    T_lens = z["T_lens"]
    rng = np.random.default_rng(seed)

    a_par_all: list[np.ndarray] = []
    a_perp_all: list[np.ndarray] = []
    natural_a2: list[float] = []
    perm_a2: list[list[float]] = []
    for s_idx, T in enumerate(T_lens):
        if T < 4:
            continue
        hs = H[s_idx, :T, :]
        d1 = hs[1:-1] - hs[:-2]
        d2 = hs[2:] - hs[1:-1]
        a = d2 - d1
        d1_norm = np.linalg.norm(d1, axis=-1)
        eps = 1e-9
        u = d1 / np.maximum(d1_norm[:, None], eps)
        a_par = (a * u).sum(axis=-1)
        a_perp_vec = a - a_par[:, None] * u
        a_perp = np.linalg.norm(a_perp_vec, axis=-1)
        a_par_all.append(a_par)
        a_perp_all.append(a_perp)
        natural_a2.append(float((np.linalg.norm(a, axis=-1) ** 2).mean()))

        sent_perms: list[float] = []
        for _ in range(n_perm):
            perm = rng.permutation(T)
            hs_p = hs[perm]
            d1p = hs_p[1:-1] - hs_p[:-2]
            d2p = hs_p[2:] - hs_p[1:-1]
            a_p = d2p - d1p
            sent_perms.append(float((np.linalg.norm(a_p, axis=-1) ** 2).mean()))
        perm_a2.append(sent_perms)

    if not a_par_all:
        return {"n_triplets": 0}

    a_par_flat  = np.concatenate(a_par_all)
    a_perp_flat = np.concatenate(a_perp_all)
    nz = a_perp_flat > 1e-9
    ratio = np.full_like(a_par_flat, np.nan)
    ratio[nz] = np.abs(a_par_flat[nz]) / a_perp_flat[nz]

    natural_arr = np.asarray(natural_a2)
    perm_arr    = np.asarray(perm_a2)
    natural_mean = float(natural_arr.mean())
    perm_mean    = float(perm_arr.mean())
    perm_std     = float(perm_arr.mean(axis=1).std()) if perm_arr.size > 1 else 0.0
    perm_z = (
        (perm_mean - natural_mean) / perm_std if perm_std > 1e-12 else float("nan")
    )

    return {
        "n_triplets": int(a_par_flat.size),
        "frac_a_par_negative": float((a_par_flat < 0).mean()),
        "mean_ratio_apar_aperp": (
            float(np.nanmean(ratio)) if np.isfinite(ratio).any() else float("nan")
        ),
        "median_ratio_apar_aperp": (
            float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else float("nan")
        ),
        "natural_mean_a_squared": natural_mean,
        "perm_mean_a_squared": perm_mean,
        "permutation_z": perm_z,
        "n_perm_per_sentence": int(n_perm),
    }


def run_one_cell(tag: str, results_root: Path,
                 logfreq_path: Path,
                 force_rerun: bool = False,
                 n_jobs: int = -1) -> dict:
    cell_dir = results_root / tag
    if not cell_dir.exists():
        print(f"[E5-diag] cell {tag}: {cell_dir} does not exist; skipping.")
        return {"tag": tag, "status": "missing"}
    ckpt = _ckpt_path(cell_dir, tag)
    if not ckpt.exists():
        print(f"[E5-diag] cell {tag}: checkpoint {ckpt} missing; "
              f"did training fail?")
        return {"tag": tag, "status": "no_checkpoint"}

    label = f"E5_ln_{tag}"
    energy_npz = cell_dir / "energy_states.npz"
    if force_rerun or not energy_npz.exists():
        rc = _run([
            sys.executable,
            str(ENERGY_DRIFT_DIR / "extract_energy_states.py"),
            "--variant", "em_ln",
            "--ckpt", str(ckpt),
            "--out_npz", str(energy_npz),
            "--label", label,
            "--logfreq", str(logfreq_path),
        ])
        if rc != 0:
            return {"tag": tag, "status": "energy_extract_failed"}
    drift = _energy_drift_summary(energy_npz)
    with open(cell_dir / "energy_drift_summary.json", "w") as f:
        json.dump(drift, f, indent=2)

    mo_dir = cell_dir / "markov_order"
    mo_dir.mkdir(exist_ok=True)
    quad_path = mo_dir / "quadruples.npz"
    traj_path = mo_dir / "trajectories.npz"
    if force_rerun or not (quad_path.exists() and traj_path.exists()):
        rc = _run([
            sys.executable,
            str(SCRIPT_DIR / "extract_ln_quadruples.py"),
            "--ckpt", str(ckpt),
            "--logfreq", str(logfreq_path),
            "--out_dir", str(mo_dir),
        ])
        if rc != 0:
            return {"tag": tag, "status": "quadruple_extract_failed"}
        if traj_path.exists():
            os.replace(traj_path, cell_dir / "trajectories.npz")

    primary_summary_path = mo_dir / "primary_summary.json"
    if force_rerun or not primary_summary_path.exists():
        rc = _run([
            sys.executable,
            str(DYNAMICS_ORDER_DIR / "markov_order_regression.py"),
            "--quads", str(quad_path),
            "--output_dir", str(mo_dir),
            "--p", "50",
            "--n_jobs", str(n_jobs),
        ])
        if rc != 0:
            return {"tag": tag, "status": "markov_regression_failed"}
    with open(primary_summary_path) as f:
        markov = json.load(f)

    accel_path = cell_dir / "acceleration_stats.json"
    if force_rerun or not accel_path.exists():
        traj_for_accel = cell_dir / "trajectories.npz"
        if not traj_for_accel.exists():
            traj_for_accel = mo_dir / "trajectories.npz"
        accel = _acceleration_stats(traj_for_accel)
        with open(accel_path, "w") as f:
            json.dump(accel, f, indent=2)
    else:
        with open(accel_path) as f:
            accel = json.load(f)

    return {
        "tag": tag, "status": "ok",
        "drift": drift, "markov": markov, "accel": accel,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_root", type=Path, default=SCRIPT_DIR / "results")
    ap.add_argument("--logfreq", type=Path, default=DEFAULT_LOGFREQ)
    ap.add_argument("--cell", type=str, default=None,
                    help="If set, only run diagnostics for this tag.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run all sub-steps even if outputs exist.")
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    if not args.logfreq.exists():
        print(f"[E5-diag] missing logfreq table at {args.logfreq}; "
              "run sarf_mass_variant/compute_unigram_frequencies.py first.",
              file=sys.stderr)
        return 1

    tags = (args.cell,) if args.cell else DEFAULT_TAGS
    args.results_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows: list[dict] = []
    for tag in tags:
        print(f"\n========== cell {tag} ==========")
        rows.append(
            run_one_cell(tag, args.results_root, args.logfreq,
                         force_rerun=args.force, n_jobs=args.n_jobs)
        )
    elapsed = time.time() - t0
    print(f"\n[E5-diag] all cells done in {elapsed:.0f}s")

    summary_path = args.results_root / "diagnostics_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"elapsed_s": elapsed, "rows": rows}, f, indent=2, default=str)
    print(f"[E5-diag] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
