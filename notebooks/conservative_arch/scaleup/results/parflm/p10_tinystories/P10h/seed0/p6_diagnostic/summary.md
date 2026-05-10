# PARF V_phi channel diagnostic — `parf_structural_competitive_vphi64_lnD-pls-θss-θbl_sparse_k4_scaleup_scaleup_seed0_ckpt_latest.pt`

- variant: `parf_structural_competitive_vphi64_lnD-pls-θss-θbl_sparse_k4_scaleup_scaleup_seed0`
- v_phi_kind: `structural_competitive`
- L: 8
- batches × batch_size × block_size: 4 × 8 × 128

## Channel summary table

| L | ‖h_t-h_s‖ med | Φ med | Φ p95 | |Θ| med | |V_φ| med | Σ_s V_φ |·| | ‖∇V_θ‖ | ‖∇ΣV_φ‖ | R(ℓ) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 21.571 | 1.001 | 1.007 | 0.107 | 5.166e-03 | 5.232e-01 | 5.018e+00 | 7.047e-02 | 0.013 |
| 2 | 19.918 | 0.683 | 2.816 | 0.099 | 8.824e-03 | 1.542e+00 | 1.315e+01 | 1.205e-01 | 0.011 |
| 3 | 19.663 | 0.762 | 2.713 | 0.135 | 9.411e-03 | 1.289e+00 | 4.399e+00 | 1.362e-01 | 0.029 |
| 4 | 19.508 | 0.567 | 2.982 | 0.134 | 7.006e-03 | 1.330e+00 | 3.213e+00 | 2.587e-01 | 0.078 |
| 5 | 19.379 | 0.212 | 3.693 | 0.136 | 2.813e-03 | 2.433e+00 | 3.268e+00 | 5.813e-01 | 0.166 |
| 6 | 19.344 | 0.129 | 4.012 | 0.159 | 1.710e-03 | 2.725e+00 | 4.147e+00 | 5.646e-01 | 0.129 |
| 7 | 19.451 | 0.153 | 3.903 | 0.202 | 2.023e-03 | 2.540e+00 | 7.451e+00 | 4.320e-01 | 0.060 |
| 8 | 19.887 | 0.227 | 3.838 | 0.235 | 2.960e-03 | 2.104e+00 | 1.471e+01 | 2.346e-01 | 0.017 |

## Failure-mode read-off

- Φ_φ has working dynamic range (median 0.467, p95 3.121); selectivity is at least partially active.
- Θ_φ retains nontrivial sign structure (mean |Θ_φ| median 0.151).
- ‖h_t-h_s‖ has healthy spread (rel (p95-p05)/median = 0.798).
- Signed pair-sum magnitudes are consistent with constructive interference; aggregation is *not* obviously destructive.
- Force-norm ratio R(ℓ) = 0.063 is in the perturbation-but-non-trivial regime; the pair force is plausibly active but not dominant.
