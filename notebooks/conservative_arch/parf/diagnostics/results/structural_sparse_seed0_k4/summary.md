# PARF V_phi channel diagnostic — `parf_structural_vphi128_sparse_k4_shakespeare_seed0_ckpt_latest.pt`

- variant: `parf_q9c_sparse_stage1.5`
- v_phi_kind: `structural`
- L: 8
- batches × batch_size × block_size: 2 × 4 × 64

## Channel summary table

| L | ‖h_t-h_s‖ med | Φ med | Φ p95 | |Θ| med | |V_φ| med | Σ_s V_φ |·| | ‖∇V_θ‖ | ‖∇ΣV_φ‖ | R(ℓ) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.654 | 0.977 | 0.997 | 0.001 | 2.029e-03 | 7.957e-02 | 1.547e+00 | 9.017e-01 | 0.758 |
| 2 | 10.756 | 0.000 | 0.700 | 0.317 | 2.622e-06 | 2.769e-01 | 2.540e+00 | 4.099e-01 | 0.161 |
| 3 | 10.944 | 0.000 | 0.679 | 0.305 | 1.364e-06 | 2.884e-01 | 2.450e+00 | 4.895e-01 | 0.188 |
| 4 | 11.609 | 0.000 | 0.552 | 0.252 | 7.722e-08 | 2.210e-01 | 2.435e+00 | 4.050e-01 | 0.151 |
| 5 | 12.590 | 0.000 | 0.335 | 0.184 | 3.318e-10 | 1.153e-01 | 2.544e+00 | 1.885e-01 | 0.073 |
| 6 | 13.571 | 0.000 | 0.178 | 0.130 | 5.772e-13 | 8.072e-02 | 2.640e+00 | 1.445e-01 | 0.047 |
| 7 | 14.399 | 0.000 | 0.109 | 0.096 | 4.303e-15 | 8.101e-02 | 2.616e+00 | 1.482e-01 | 0.042 |
| 8 | 15.026 | 0.000 | 0.112 | 0.070 | 1.752e-15 | 7.411e-02 | 2.651e+00 | 1.369e-01 | 0.031 |

## Failure-mode read-off

- Φ_φ has working dynamic range (median 0.122, p95 0.458); selectivity is at least partially active.
- Θ_φ retains nontrivial sign structure (mean |Θ_φ| median 0.169).
- ‖h_t-h_s‖ has healthy spread (rel (p95-p05)/median = 0.795).
- Signed pair-sum magnitudes are consistent with constructive interference; aggregation is *not* obviously destructive.
- Force-norm ratio R(ℓ) = 0.181 is in the perturbation-but-non-trivial regime; the pair force is plausibly active but not dominant.
