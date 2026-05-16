# PARF V_phi channel diagnostic — `parf_structural_competitive_vphi64_lnD-pls-θss-θbl_sparse_k16_scaleup_scaleup_seed0_ckpt_latest.pt`

- variant: `parf_structural_competitive_vphi64_lnD-pls-θss-θbl_sparse_k16_scaleup_scaleup_seed0`
- v_phi_kind: `structural_competitive`
- L: 8
- batches × batch_size × block_size: 4 × 8 × 128

## Channel summary table

| L | ‖h_t-h_s‖ med | Φ med | Φ p95 | |Θ| med | |V_φ| med | Σ_s V_φ |·| | ‖∇V_θ‖ | ‖∇ΣV_φ‖ | R(ℓ) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 21.942 | 0.974 | 1.597 | 0.061 | 2.784e-03 | 3.836e-01 | 5.135e+00 | 9.373e-02 | 0.014 |
| 2 | 20.663 | 0.435 | 4.865 | 0.123 | 6.024e-03 | 1.122e+00 | 1.713e+01 | 3.703e-02 | 0.003 |
| 3 | 19.690 | 0.643 | 3.836 | 0.006 | 8.172e-03 | 1.191e+00 | 6.400e+00 | 6.279e-02 | 0.009 |
| 4 | 19.348 | 0.804 | 2.758 | 0.069 | 9.464e-03 | 1.597e+00 | 5.852e+00 | 1.485e-01 | 0.027 |
| 5 | 19.864 | 0.813 | 2.702 | 0.111 | 9.643e-03 | 9.524e-01 | 6.595e+00 | 1.561e-01 | 0.032 |
| 6 | 20.296 | 0.834 | 2.524 | 0.115 | 9.338e-03 | 6.630e-01 | 8.990e+00 | 1.821e-01 | 0.029 |
| 7 | 20.447 | 0.935 | 1.858 | 0.111 | 1.019e-02 | 5.773e-01 | 1.339e+01 | 1.163e-01 | 0.014 |
| 8 | 20.816 | 0.943 | 1.844 | 0.107 | 1.037e-02 | 6.626e-01 | 2.061e+01 | 1.305e-01 | 0.008 |

## Failure-mode read-off

- Φ_φ has working dynamic range (median 0.798, p95 2.748); selectivity is at least partially active.
- Θ_φ retains nontrivial sign structure (mean |Θ_φ| median 0.088).
- ‖h_t-h_s‖ has healthy spread (rel (p95-p05)/median = 0.777).
- Signed pair-sum magnitudes are consistent with constructive interference; aggregation is *not* obviously destructive.
- **[V_φ force negligible vs V_θ]** Mean R(ℓ) = 0.017 << 1; the pair force is not contributing to the dynamics in a meaningful way.  Either Θ has collapsed (see above) or C is too small.  Lever 6 (warm-up curriculum) targets this.
