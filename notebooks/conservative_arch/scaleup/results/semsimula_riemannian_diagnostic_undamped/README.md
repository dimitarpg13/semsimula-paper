# Riemannian Geometry Diagnostic Battery — Undamped Framework

**Experiment:** `riemannian_geometry_diagnostic_battery` (original undamped formulation)
**Date:** June 2026
**Notebook:** [`colab_riemannian_diagnostic.ipynb`](../../colab_riemannian_diagnostic.ipynb) (original version)
**Paper section:** Section 18 of the mega-paper (v4/v5)

## Models tested

| Model | Integrator | γ |
|---|---|---|
| Multi-Xi SPLM | Semi-implicit damped Euler | 0.30 |
| Fock v2.1 PARFLM | Damped velocity-Verlet | 0.30 |
| Fock Attention PARFLM | Damped velocity-Verlet + direct exchange | 0.30 |

## Configuration

- Batches: 3 × 4 sequences, block_size = 128
- Power iterations (for λ_max): 20
- Sample positions per sequence: 32
- Sample sequences for Arm 3: 3
- Device: CUDA (Google Colab)

## Five-arm battery

| Arm | Test | Key metric |
|---|---|---|
| 1 | Metric validity (Ω² = 2(E−V) positivity) | frac_positive per layer |
| 2 | Geodesic compliance (R² and cosine vs Christoffel prediction) | mean compliance, mean cosine |
| 3 | Curvature proxy (K_max = λ_max / Ω²) vs entropy | Spearman ρ, p-value |
| 4 | Energy conservation profile | total drift, relative drift |
| 5 | Conservativity separator (R²_full, R²_sym) | mean R²_full, mean R²_sym |

## Key findings

- **Arm 1:** Ω² > 0 at 100% of positions for all models (metric valid everywhere).
- **Arm 2:** Moderate directional cosine similarity (SPLM: 0.73, Fock v2.1: 0.48, Fock Attn: 0.44). R² compliance is negative for all.
- **Arm 3:** K_max well-defined but no significant entropy correlation (|ρ| < 0.15, p > 0.16).
- **Arm 4:** Significant energy drift (SPLM: 172% relative, Fock: 55×/41× relative) — consistent with damping, not energy conservation.
- **Arm 5:** R²_full ≈ 0.78–0.83 (predominantly linear dynamics), R²_sym ≪ 0 (strongly asymmetric).

## Note

This is the **original undamped** formulation of the diagnostic battery. The Arm 4 results (large energy drift) motivated the reframing to a **damped Riemannian** framework, where energy dissipation is expected and the anomaly signal ΔE_anomaly = |ΔE_obs − ΔE_expected| replaces energy conservation as the diagnostic. See `semsimula_riemannian_diagnostic/` for the damped version.
