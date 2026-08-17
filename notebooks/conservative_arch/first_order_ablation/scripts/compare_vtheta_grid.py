#!/usr/bin/env python3
"""3x3 matched-OWT V_theta sensitivity grid: width (d) x damping (gamma).

Purpose
-------
Extends compare_vtheta_across_scale.py from a single-gamma width trend to
a full d x gamma grid, to rank how much each factor perturbs the learned
aniso-Gaussian potential:

    training order (FO vs SO, see compare_vtheta_profiles.py)  -- null
    damping gamma (within a fixed d)                            -- modest
    width d (within a fixed gamma)                               -- large

All nine checkpoints are matched on corpus (OpenWebText), v_theta variant
(aniso_gaussian + fock-reg), depth (L=16), aniso_rank (4), and sweep
duration (3,000 steps) -- only d and gamma vary. Because raw-parameter
cosine similarity requires identical tensor shapes, this script reports
only dimensionless / scale-free summary statistics across d
(anisotropy_ratio, entropy fraction of max) plus lambda_max as a
supplementary (not strictly scale-free, but partially self-normalising
because precision_max = 2/d) reference.

Usage
-----
    python compare_vtheta_grid.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare_vtheta_profiles import load_vtheta, summarize_checkpoint  # noqa: E402

HOME = Path.home()

DS = [384, 768, 1024]
GAMMAS = ["0.050", "0.100", "0.300"]

HEADS = 5
WELLS = 8
RANK = 4
LAYERS = 16


def ckpt_path(d: int, gamma: str) -> Path:
    base = HOME / f"Downloads/semsimula_fock_gamma_sweep_aniso_gaussian_fockreg_d{d}/gamma_sweep/gamma_{gamma}"
    for cand in (base / "checkpoints" / "ckpt_best.pt", base / "ckpt_best.pt"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no checkpoint for d={d}, gamma={gamma} under {base}")


def main():
    cells = {}
    for d in DS:
        for gamma in GAMMAS:
            path = ckpt_path(d, gamma)
            precision_max = 2.0 / d
            vtheta, meta = load_vtheta(str(path), d, WELLS, HEADS, LAYERS, RANK, precision_max)
            stats = summarize_checkpoint(vtheta, n_probe=20, sigma=1.0, seed=0)
            cells[(d, gamma)] = (meta, stats)
            print(f"=== d={d}, gamma={gamma} ===  {path}")
            print(f"  meta: step={meta.get('step')} val_ppl={meta.get('val_ppl')}")

    def grid(metric_fn, fmt="{:>10.4g}"):
        header = "d\\gamma".ljust(10) + "".join(g.rjust(12) for g in GAMMAS)
        lines = [header]
        for d in DS:
            row = str(d).ljust(10)
            for gamma in GAMMAS:
                _, stats = cells[(d, gamma)]
                row += fmt.format(metric_fn(stats)).rjust(12)
            lines.append(row)
        return "\n".join(lines)

    print("\n--- anisotropy_ratio (mean) ---")
    print(grid(lambda s: s["anisotropy_ratio"]["mean"]))
    print("\n--- anisotropy_ratio (max, worst-case well) ---")
    print(grid(lambda s: s["anisotropy_ratio"]["max"]))
    print("\n--- well_weight_entropy_frac_of_max (scale-free, in [0,1]) ---")
    print(grid(lambda s: s["well_weight_entropy_frac_of_max"], fmt="{:>10.4f}"))
    print("\n--- lambda_max (mean) [supplementary; partially self-normalising via precision_max=2/d] ---")
    print(grid(lambda s: s["lambda_max"]["mean"]))
    print("\n--- trace (mean) [supplementary; same caveat] ---")
    print(grid(lambda s: s["trace"]["mean"]))
    print("\n--- depth_code_norm_variation_across_layers [supplementary] ---")
    print(grid(lambda s: s["depth_code_norm_variation_across_layers"], fmt="{:>10.4f}"))

    # Sensitivity-budget summary: relative spread along each axis.
    def rel_spread(vals):
        vals = list(vals)
        lo, hi = min(vals), max(vals)
        return (hi - lo) / abs(lo) if lo else float("nan")

    aniso_by_gamma_at_d = {d: [cells[(d, g)][1]["anisotropy_ratio"]["mean"] for g in GAMMAS] for d in DS}
    aniso_by_d_at_gamma = {g: [cells[(d, g)][1]["anisotropy_ratio"]["mean"] for d in DS] for g in GAMMAS}
    print("\n--- Sensitivity budget (relative spread of mean anisotropy_ratio) ---")
    for d in DS:
        print(f"  across gamma at d={d}: {rel_spread(aniso_by_gamma_at_d[d]) * 100:.1f}%")
    for g in GAMMAS:
        print(f"  across d at gamma={g}: {rel_spread(aniso_by_d_at_gamma[g]) * 100:.1f}%")

    report = {f"d={d},gamma={g}": {"meta": cells[(d, g)][0], "stats": cells[(d, g)][1]}
              for d in DS for g in GAMMAS}
    out_path = Path(__file__).parent / "vtheta_grid_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
