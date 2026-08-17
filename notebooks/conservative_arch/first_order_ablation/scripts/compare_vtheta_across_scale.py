#!/usr/bin/env python3
"""Cross-scale (same-order) V_theta curvature/anisotropy comparison.

Purpose
-------
Reuses the summarizer from compare_vtheta_profiles.py to ask a narrower
question than the SPLM-1 vs SPLM-2 comparator: within second-order
(damped-Verlet) training only, does the learned aniso-Gaussian V_theta's
curvature/anisotropy/entropy profile already trend toward a more
"scarred" shape as width grows on OpenWebText (d=384 -> d=768), i.e.
before any first-order counterpart exists at this scale?

This is a same-order, cross-scale probe -- a cheap proxy for whether the
"second-order gradient cascade" documented in
companion_notes/Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md
(Sec 23-25) and paper_v5/sections/17e_scaling_up.tex leaves a visible
imprint on V_theta's own shape even in runs that never hit the watchdog
(the local OWT gamma-sweep checkpoints here are short, ~3k-step runs,
well short of the ~33-37k-step onset window documented for the full
d=768 run).

Usage
-----
    python compare_vtheta_across_scale.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare_vtheta_profiles import load_vtheta, summarize_checkpoint  # noqa: E402

HOME = Path.home()

CONFIGS = [
    dict(label="d=384, gamma=0.30 (OWT sweep)", d=384, heads=5, wells=8, rank=4, layers=16,
         ckpt=HOME / "Downloads/semsimula_fock_gamma_sweep_aniso_gaussian_fockreg_d384/gamma_sweep/gamma_0.300/checkpoints/ckpt_best.pt"),
    dict(label="d=768, gamma=0.30 (OWT sweep)", d=768, heads=5, wells=8, rank=4, layers=16,
         ckpt=HOME / "Downloads/semsimula_fock_gamma_sweep_aniso_gaussian_fockreg_d768/gamma_sweep/gamma_0.300/checkpoints/ckpt_best.pt"),
    dict(label="d=1024, gamma=0.30 (OWT sweep)", d=1024, heads=5, wells=8, rank=4, layers=16,
         ckpt=HOME / "Downloads/semsimula_fock_gamma_sweep_aniso_gaussian_fockreg_d1024/gamma_sweep/gamma_0.300/ckpt_best.pt"),
    dict(label="d=1024, gamma=0.05 (OWT sweep, preview)", d=1024, heads=5, wells=8, rank=4, layers=16,
         ckpt=HOME / "Downloads/semsimula_fock_gamma_sweep_aniso_gaussian_fockreg_d1024/gamma_sweep/gamma_0.050/ckpt_best.pt"),
    dict(label="d=256, gamma=0.30 (TinyStories anchor)", d=256, heads=4, wells=8, rank=4, layers=8,
         ckpt=HOME / "Downloads/semsimula_fock_aniso_gaussian_fockreg_tinystories/results/seed0_gamma=0.3/ckpt_best.pt"),
]
CONFIGS = [c for c in CONFIGS if c["ckpt"].exists()]


def main():
    rows = []
    for cfg in CONFIGS:
        precision_max = 2.0 / cfg["d"]
        vtheta, meta = load_vtheta(str(cfg["ckpt"]), cfg["d"], cfg["wells"], cfg["heads"],
                                    cfg["layers"], cfg["rank"], precision_max)
        stats = summarize_checkpoint(vtheta, n_probe=20, sigma=1.0, seed=0)
        rows.append((cfg["label"], cfg, meta, stats))
        print(f"=== {cfg['label']} ===  {cfg['ckpt']}")
        print(f"  meta: step={meta.get('step')} val_ppl={meta.get('val_ppl')} "
              f"best_val_ppl={meta.get('best_val_ppl')} gamma={meta.get('gamma')}")

    print(f"\n{'metric':38s}" + "".join(f"{r[0]:>34s}" for r in rows))
    print("-" * (38 + 34 * len(rows)))
    for key in ["lambda_min", "lambda_max", "trace", "anisotropy_ratio",
                "well_weight_entropy", "nearest_neighbour_centre_dist"]:
        vals = [r[3][key]["mean"] for r in rows]
        print(f"{key + ' (mean)':38s}" + "".join(f"{v:34.5g}" for v in vals))
    print(f"{'well_weight_entropy_frac_of_max':38s}" +
          "".join(f"{r[3]['well_weight_entropy_frac_of_max']:34.4f}" for r in rows))
    print(f"{'anisotropy_ratio (max)':38s}" +
          "".join(f"{r[3]['anisotropy_ratio']['max']:34.5g}" for r in rows))
    print(f"{'lambda_max (max)':38s}" +
          "".join(f"{r[3]['lambda_max']['max']:34.5g}" for r in rows))
    print(f"{'depth_code_norm_variation':38s}" +
          "".join(f"{r[3]['depth_code_norm_variation_across_layers']:34.5g}" for r in rows))

    report = {r[0]: {"config": {k: str(v) for k, v in r[1].items()}, "meta": r[2], "stats": r[3]}
              for r in rows}
    out_path = Path(__file__).parent / "cross_scale_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
