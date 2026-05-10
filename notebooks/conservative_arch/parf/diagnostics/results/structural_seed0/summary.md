# PARF V_phi channel diagnostic — `parf_structural_shakespeare_seed0_ckpt_latest.pt`

- variant: `parf_q9c`
- v_phi_kind: `structural`
- L: 8
- batches × batch_size × block_size: 2 × 4 × 64

## Channel summary table

| L | ‖h_t-h_s‖ med | Φ med | Φ p95 | |Θ| med | |V_φ| med | Σ_s V_φ |·| | ‖∇V_θ‖ | ‖∇ΣV_φ‖ | R(ℓ) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.568 | 0.964 | 0.996 | 0.738 | 4.546e-01 | 1.738e+01 | 7.097e+00 | 1.905e+01 | 2.946 |
| 2 | 8.474 | 0.225 | 0.734 | 1.000 | 2.604e-02 | 1.301e+00 | 6.126e+00 | 6.437e-01 | 0.106 |
| 3 | 9.148 | 0.216 | 0.766 | 1.000 | 2.318e-02 | 1.226e+00 | 5.720e+00 | 5.393e-01 | 0.094 |
| 4 | 9.934 | 0.226 | 0.807 | 1.000 | 2.244e-02 | 1.188e+00 | 8.006e+00 | 4.368e-01 | 0.058 |
| 5 | 9.017 | 0.430 | 0.838 | 0.913 | 3.685e-02 | 1.387e+00 | 9.295e+00 | 6.447e-01 | 0.069 |
| 6 | 6.440 | 0.634 | 0.908 | 1.000 | 9.985e-02 | 3.213e+00 | 6.312e+00 | 1.107e+00 | 0.208 |
| 7 | 3.574 | 0.786 | 0.948 | 1.000 | 2.206e-01 | 6.834e+00 | 9.515e+00 | 2.707e+00 | 0.300 |
| 8 | 6.864 | 0.271 | 0.832 | 1.000 | 3.794e-02 | 2.159e+00 | 1.571e+01 | 9.893e-01 | 0.061 |

## Failure-mode read-off

- Φ_φ has working dynamic range (median 0.469, p95 0.854); selectivity is at least partially active.
- Θ_φ retains nontrivial sign structure (mean |Θ_φ| median 0.956).
- ‖h_t-h_s‖ has healthy spread (rel (p95-p05)/median = 1.164).
- Signed pair-sum magnitudes are consistent with constructive interference; aggregation is *not* obviously destructive.
- Force-norm ratio R(ℓ) = 0.480 is in the perturbation-but-non-trivial regime; the pair force is plausibly active but not dominant.
