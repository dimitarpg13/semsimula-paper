# PARF V_phi channel diagnostic — `parf_structural_competitive_vphi64_lnD-pls-θss-θbl_sparse_k8_scaleup_scaleup_seed0_ckpt_latest.pt`

- variant: `parf_structural_competitive_vphi64_lnD-pls-θss-θbl_sparse_k8_scaleup_scaleup_seed0`
- v_phi_kind: `structural_competitive`
- L: 8
- batches × batch_size × block_size: 4 × 8 × 128

## Channel summary table

| L | ‖h_t-h_s‖ med | Φ med | Φ p95 | |Θ| med | |V_φ| med | Σ_s V_φ |·| | ‖∇V_θ‖ | ‖∇ΣV_φ‖ | R(ℓ) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 21.914 | 1.014 | 1.850 | 0.043 | 2.069e-03 | 3.811e-01 | 4.942e+00 | 1.295e-01 | 0.017 |
| 2 | 19.479 | 0.859 | 2.037 | 0.003 | 1.587e-02 | 3.386e+00 | 2.000e+01 | 2.815e-01 | 0.015 |
| 3 | 19.945 | 0.824 | 1.954 | 0.001 | 1.374e-02 | 1.801e+00 | 6.723e+00 | 2.187e-01 | 0.031 |
| 4 | 19.940 | 0.811 | 2.049 | 0.078 | 1.204e-02 | 2.296e+00 | 5.524e+00 | 3.114e-01 | 0.052 |
| 5 | 20.035 | 0.859 | 2.047 | 0.171 | 1.210e-02 | 1.549e+00 | 5.945e+00 | 2.678e-01 | 0.047 |
| 6 | 20.300 | 0.931 | 1.926 | 0.207 | 1.200e-02 | 1.099e+00 | 7.086e+00 | 2.163e-01 | 0.039 |
| 7 | 20.522 | 0.985 | 1.778 | 0.255 | 1.266e-02 | 1.280e+00 | 9.223e+00 | 2.282e-01 | 0.026 |
| 8 | 20.961 | 1.017 | 1.720 | 0.224 | 1.159e-02 | 1.159e+00 | 1.365e+01 | 2.086e-01 | 0.017 |

## Failure-mode read-off

- **[Φ_φ saturated near 1]** Median Φ across layers is 0.912 (p95 = 1.920).  The type-gate is *not* selecting; nearly every pair contributes close to its full distance-and-Θ value.  Lever 3 (softmax-normalised Φ) directly targets this.
- Θ_φ retains nontrivial sign structure (mean |Θ_φ| median 0.123).
- ‖h_t-h_s‖ has healthy spread (rel (p95-p05)/median = 0.774).
- Signed pair-sum magnitudes are consistent with constructive interference; aggregation is *not* obviously destructive.
- **[V_φ force negligible vs V_θ]** Mean R(ℓ) = 0.030 << 1; the pair force is not contributing to the dynamics in a meaningful way.  Either Θ has collapsed (see above) or C is too small.  Lever 6 (warm-up curriculum) targets this.
