# PARF V_phi channel diagnostic — `parf_structural_competitive_vphi64_lnD-pls-θss-θbl_sparse_k4_scaleup_scaleup_seed0_ckpt_latest.pt`

- variant: `parf_structural_competitive_vphi64_lnD-pls-θss-θbl_sparse_k4_scaleup_scaleup_seed0`
- v_phi_kind: `structural_competitive`
- L: 8
- batches × batch_size × block_size: 4 × 8 × 128

## Channel summary table

| L | ‖h_t-h_s‖ med | Φ med | Φ p95 | |Θ| med | |V_φ| med | Σ_s V_φ |·| | ‖∇V_θ‖ | ‖∇ΣV_φ‖ | R(ℓ) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 21.838 | 1.103 | 1.731 | 0.075 | 4.089e-03 | 4.332e-01 | 4.340e+00 | 3.381e-01 | 0.039 |
| 2 | 19.784 | 0.876 | 1.505 | 0.202 | 2.250e-02 | 4.767e+00 | 2.186e+01 | 1.188e-01 | 0.008 |
| 3 | 20.145 | 0.894 | 1.696 | 0.244 | 2.082e-02 | 3.985e+00 | 6.607e+00 | 5.361e-01 | 0.079 |
| 4 | 20.348 | 0.903 | 1.810 | 0.184 | 1.937e-02 | 2.712e+00 | 4.233e+00 | 4.481e-01 | 0.096 |
| 5 | 20.421 | 0.945 | 1.921 | 0.218 | 1.744e-02 | 2.034e+00 | 3.989e+00 | 3.980e-01 | 0.099 |
| 6 | 20.495 | 0.971 | 1.945 | 0.340 | 2.042e-02 | 1.792e+00 | 5.482e+00 | 3.070e-01 | 0.063 |
| 7 | 20.769 | 0.948 | 2.093 | 0.382 | 2.001e-02 | 1.704e+00 | 9.133e+00 | 1.245e-01 | 0.011 |
| 8 | 21.236 | 0.922 | 2.099 | 0.307 | 1.546e-02 | 1.604e+00 | 1.359e+01 | 8.186e-01 | 0.035 |

## Failure-mode read-off

- **[Φ_φ saturated near 1]** Median Φ across layers is 0.945 (p95 = 1.850).  The type-gate is *not* selecting; nearly every pair contributes close to its full distance-and-Θ value.  Lever 3 (softmax-normalised Φ) directly targets this.
- Θ_φ retains nontrivial sign structure (mean |Θ_φ| median 0.244).
- ‖h_t-h_s‖ has healthy spread (rel (p95-p05)/median = 0.770).
- Signed pair-sum magnitudes are consistent with constructive interference; aggregation is *not* obviously destructive.
- Force-norm ratio R(ℓ) = 0.054 is in the perturbation-but-non-trivial regime; the pair force is plausibly active but not dominant.
