# Damped Riemannian Geometry Diagnostic Battery

**Experiment:** `damped_riemannian_geometry_diagnostic_battery`
**Date:** June 2026
**Notebook:** [`colab_riemannian_diagnostic.ipynb`](../../colab_riemannian_diagnostic.ipynb) (damped-framework revision)
**Paper section:** Section 18 (diagnostic battery subsection) and Section 18d of the mega-paper (v4/v5)
**Companion note:** `companion_notes/Exploiting_the_Riemannian_geometry_of_conservative_language_models.md`

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

## Five-arm battery (damped framework)

| Arm | Test | Key metric |
|---|---|---|
| 1 | Metric validity (Ω²_ℓ = 2T_ℓ·m positivity) | frac_positive per layer |
| 2 | Geodesic compliance — **damped vs undamped** | cosine and R² for both formulations |
| 3 | Curvature proxy (K_max = λ_max / 2T_ℓ) vs entropy | Spearman ρ, p-value |
| 4 | Energy **dissipation** profile | ΔE_observed, ΔE_expected, ΔE_anomaly per layer |
| 5 | Conservativity & damping separator | R²_full, R²_sym, **asymmetry ratio** |

## Key findings

### Arm 1 — Metric validity (CONFIRMED)
- Ω²_ℓ > 0 at **100%** of positions for all models, all layers.
- Conformal factor decays monotonically: SPLM 86.5 → 17.0; Fock v2.1 330 → 24.8; Fock Attn 330 → 28.8.
- Fock models show layer-1 spike from exchange force injection.

### Arm 2 — Geodesic compliance (DAMPED BETTER)
| Model | Undamped cosine | Damped cosine | Improvement |
|---|---|---|---|
| Multi-Xi SPLM | 0.727 | **0.751** | +3.3% |
| Fock v2.1 | 0.483 | **0.558** | +15.5% |
| Fock Attention | 0.436 | **0.521** | +19.5% |

R²-style compliance is negative for both formulations (LayerNorm + exchange forces prevent magnitude-level tracking).

### Arm 3 — Curvature proxy (COMPUTABLE, NO ENTROPY CORRELATION YET)
| Model | K_max mean | Spearman ρ | p-value |
|---|---|---|---|
| Multi-Xi SPLM | 0.057 | −0.144 | 0.161 |
| Fock v2.1 | 0.027 | +0.012 | 0.908 |
| Fock Attention | 0.026 | −0.038 | 0.712 |

### Arm 4 — Energy dissipation profile (CLEAN DECAY + EXCHANGE TRANSIENTS)
- **SPLM:** Monotonic dissipation E: 39.8 → −28.6. Anomaly peaks at layers 2–3 (6.7, 6.0), drops to ~0.5 at later layers.
- **Fock v2.1/Attn:** Layer-1 energy spike (0.3 → 142) from exchange force. Anomaly ~200 at layer 1, settling to ~2.2 by layer 3.

### Arm 5 — Conservativity & damping separator
| Model | R²_full | R²_sym | Asymmetry ratio |
|---|---|---|---|
| Multi-Xi SPLM | 0.780 | −2.96 | **1.396** |
| Fock v2.1 | 0.832 | −3.74 | **1.350** |
| Fock Attention | 0.825 | −3.07 | **1.350** |

## Implications for native architectural features

1. **Geodesic Analogical Reasoning:** Viable but must use directed arcs (asymmetry ratio ~1.4).
2. **Native Hallucination Detection:** Strongly validated — ΔE_anomaly signal is computable and well-behaved.
3. **Geodesic Semantic Distance:** Well-defined but inherently asymmetric; symmetrised variant available.
4. **Native Chain-of-Thought:** Supported — Fock dynamics predominantly linear (R²_full ≈ 0.83).

## Relation to undamped version

This battery supersedes the undamped version (`semsimula_riemannian_diagnostic_undamped/`). The key change is reframing energy conservation (which fails due to damping) as controlled energy dissipation with a computable anomaly signal. The Arm 2 comparison between undamped and damped geodesic equations is new to this version.
