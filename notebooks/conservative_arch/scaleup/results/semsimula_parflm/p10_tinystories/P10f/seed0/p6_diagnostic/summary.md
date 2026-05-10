# PARF V_phi channel diagnostic — `parf_structural_competitive_vphi64_lnD-pls-θss-θbl_sparse_k4_scaleup_scaleup_seed0_ckpt_latest.pt`

- variant: `parf_structural_competitive_vphi64_lnD-pls-θss-θbl_sparse_k4_scaleup_scaleup_seed0`
- v_phi_kind: `structural_competitive`
- L: 8
- batches × batch_size × block_size: 4 × 8 × 128

## Channel summary table

| L | ‖h_t-h_s‖ med | Φ med | Φ p95 | |Θ| med | |V_φ| med | Σ_s V_φ |·| | ‖∇V_θ‖ | ‖∇ΣV_φ‖ | R(ℓ) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 22.219 | 0.784 | 3.029 | 0.184 | 6.228e-03 | 1.175e+00 | 3.628e+00 | 1.773e-01 | 0.063 |
| 2 | 21.342 | 0.000 | 6.993 | 0.009 | 2.245e-08 | 4.761e+00 | 1.305e+01 | 2.245e-01 | 0.019 |
| 3 | 20.997 | 0.000 | 7.039 | 0.032 | 1.754e-08 | 3.866e+00 | 4.926e+00 | 3.850e-01 | 0.086 |
| 4 | 20.461 | 0.000 | 5.997 | 0.300 | 8.741e-08 | 4.650e+00 | 3.559e+00 | 5.372e-01 | 0.159 |
| 5 | 20.175 | 0.000 | 4.512 | 0.408 | 8.885e-07 | 5.758e+00 | 3.121e+00 | 8.056e-01 | 0.274 |
| 6 | 20.189 | 0.000 | 4.225 | 0.370 | 2.807e-06 | 5.738e+00 | 3.545e+00 | 8.866e-01 | 0.313 |
| 7 | 20.457 | 0.000 | 4.097 | 0.288 | 4.311e-06 | 4.986e+00 | 4.673e+00 | 5.657e-01 | 0.156 |
| 8 | 20.935 | 0.000 | 3.399 | 0.169 | 5.780e-06 | 4.138e+00 | 6.985e+00 | 8.370e-01 | 0.114 |

## Failure-mode read-off

- Φ_φ has working dynamic range (median 0.098, p95 4.911); selectivity is at least partially active.
- Θ_φ retains nontrivial sign structure (mean |Θ_φ| median 0.220).
- ‖h_t-h_s‖ has healthy spread (rel (p95-p05)/median = 0.724).
- Signed pair-sum magnitudes are consistent with constructive interference; aggregation is *not* obviously destructive.
- Force-norm ratio R(ℓ) = 0.148 is in the perturbation-but-non-trivial regime; the pair force is plausibly active but not dominant.
