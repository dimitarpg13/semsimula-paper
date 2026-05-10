# PARF V_phi channel diagnostic — `parf_structural_vphi16_sparse_k4_scaleup_scaleup_seed0_ckpt_latest.pt`

- variant: `parf_q9c_sparse_stage1.5`
- v_phi_kind: `structural`
- L: 8
- batches × batch_size × block_size: 4 × 4 × 128

## Channel summary table

| L | ‖h_t-h_s‖ med | Φ med | Φ p95 | |Θ| med | |V_φ| med | Σ_s V_φ |·| | ‖∇V_θ‖ | ‖∇ΣV_φ‖ | R(ℓ) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.059 | 0.772 | 0.989 | 0.001 | 7.814e-04 | 5.339e-02 | 3.460e+00 | 5.337e-01 | 0.204 |
| 2 | 16.949 | 0.000 | 0.144 | 0.013 | 0.000e+00 | 6.483e-02 | 7.301e+00 | 1.870e-01 | 0.025 |
| 3 | 15.802 | 0.000 | 0.126 | 0.047 | 0.000e+00 | 1.100e-01 | 4.820e+00 | 2.559e-01 | 0.057 |
| 4 | 14.994 | 0.000 | 0.107 | 0.146 | 0.000e+00 | 2.637e-01 | 4.057e+00 | 3.827e-01 | 0.096 |
| 5 | 15.447 | 0.000 | 0.011 | 0.269 | 0.000e+00 | 1.825e-01 | 3.684e+00 | 2.570e-01 | 0.065 |
| 6 | 16.750 | 0.000 | 0.000 | 0.360 | 0.000e+00 | 9.539e-02 | 4.358e+00 | 1.387e-01 | 0.028 |
| 7 | 18.427 | 0.000 | 0.000 | 0.354 | 0.000e+00 | 4.285e-02 | 6.009e+00 | 7.147e-02 | 0.010 |
| 8 | 20.166 | 0.000 | 0.000 | 0.198 | 0.000e+00 | 8.009e-03 | 6.640e+00 | 1.730e-02 | 0.002 |

## Failure-mode read-off

- Φ_φ has working dynamic range (median 0.096, p95 0.172); selectivity is at least partially active.
- Θ_φ retains nontrivial sign structure (mean |Θ_φ| median 0.174).
- ‖h_t-h_s‖ has healthy spread (rel (p95-p05)/median = 1.034).
- Signed pair-sum magnitudes are consistent with constructive interference; aggregation is *not* obviously destructive.
- Force-norm ratio R(ℓ) = 0.061 is in the perturbation-but-non-trivial regime; the pair force is plausibly active but not dominant.
