# PARF V_phi channel diagnostic — `parf_structural_competitive_vphi64_lnD-pls-θss-θbl_sparse_k4_scaleup_scaleup_seed0_ckpt_latest.pt`

- variant: `parf_structural_competitive_vphi64_lnD-pls-θss-θbl_sparse_k4_scaleup_scaleup_seed0`
- v_phi_kind: `structural_competitive`
- L: 8
- batches × batch_size × block_size: 4 × 8 × 128

## Channel summary table

| L | ‖h_t-h_s‖ med | Φ med | Φ p95 | |Θ| med | |V_φ| med | Σ_s V_φ |·| | ‖∇V_θ‖ | ‖∇ΣV_φ‖ | R(ℓ) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 22.146 | 1.002 | 1.008 | 0.134 | 6.280e-03 | 8.755e-01 | 4.901e+00 | 1.056e-01 | 0.019 |
| 2 | 19.742 | 0.000 | 5.424 | 0.102 | 1.170e-13 | 1.663e+01 | 8.335e+00 | 1.519e+00 | 0.156 |
| 3 | 18.759 | 0.000 | 5.099 | 0.121 | 4.298e-09 | 1.099e+01 | 5.398e+00 | 1.467e+00 | 0.234 |
| 4 | 17.638 | 0.000 | 5.126 | 0.282 | 1.126e-08 | 6.094e+00 | 4.679e+00 | 4.659e-01 | 0.135 |
| 5 | 17.542 | 0.000 | 5.680 | 0.302 | 6.683e-16 | 6.839e+00 | 4.129e+00 | 4.457e-01 | 0.139 |
| 6 | 18.263 | 0.000 | 5.649 | 0.231 | 8.976e-25 | 5.287e+00 | 4.708e+00 | 8.192e-01 | 0.244 |
| 7 | 19.350 | 0.000 | 5.850 | 0.114 | 3.845e-34 | 3.582e+00 | 6.179e+00 | 5.554e-01 | 0.126 |
| 8 | 20.275 | 0.000 | 5.800 | 0.166 | 3.997e-34 | 3.521e+00 | 9.296e+00 | 4.144e-01 | 0.055 |

## Failure-mode read-off

- Φ_φ has working dynamic range (median 0.125, p95 4.954); selectivity is at least partially active.
- Θ_φ retains nontrivial sign structure (mean |Θ_φ| median 0.181).
- ‖h_t-h_s‖ has healthy spread (rel (p95-p05)/median = 0.880).
- Signed pair-sum magnitudes are consistent with constructive interference; aggregation is *not* obviously destructive.
- Force-norm ratio R(ℓ) = 0.139 is in the perturbation-but-non-trivial regime; the pair force is plausibly active but not dominant.
