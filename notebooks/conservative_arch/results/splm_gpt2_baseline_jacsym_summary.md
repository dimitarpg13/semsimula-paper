# Jacobian-symmetry diagnostic -- scalar-potential LM (gpt2_baseline)

Tests whether the per-step linear operator $M_\ell$ is **symmetric**.  A symmetric $M_\ell$ is what a conservative flow on a scalar potential must produce (Hessians of scalars are symmetric).  Skew-symmetric or symmetric-non-Hessian components indicate non-conservative dynamics.

We run TWO variants:

1. **Position-only (§1.5 analogue)**: $x_{\ell+1}-x_\ell \approx M_\ell x_\ell$.  For **damped second-order** dynamics (i.e. both this model and the Failure-doc §1 integrator), this fit is known to be confounded because the single-step transition mixes $x_\ell$ with the hidden velocity $v_\ell \approx x_\ell - x_{\ell-1}$.  The confound can manufacture apparent asymmetry even in genuinely conservative flows.
2. **Velocity-aware**: $x_{\ell+1}-x_\ell \approx A v_\ell + M_\ell x_\ell$ with $v_\ell = x_\ell - x_{\ell-1}$.  Here $M_\ell$ is the clean signal of the per-step spring matrix.  **This is the variant that tests conservativity.**

- Hidden dim `d = 768`, integration steps `L = 12`, PCA `k = 16`
- Train / test sentences: 40 / 10

## Per-layer fit quality

| layer | POS-only $R^{2}_\text{full}$ | POS-only $R^{2}_\text{sym}$ | VEL-aug $R^{2}_\text{full}$ | VEL-aug $R^{2}_\text{sym}$ | VEL-aug gap |
|--:|--:|--:|--:|--:|--:|
| 0 | +0.861 | +0.694 | +nan | +nan | +nan |
| 1 | +0.997 | +0.995 | +0.998 | +0.997 | +0.000 |
| 2 | +1.000 | +0.997 | +1.000 | +1.000 | +0.000 |
| 3 | +0.981 | +0.979 | +0.981 | +0.980 | +0.000 |
| 4 | +0.975 | +0.970 | +0.976 | +0.972 | +0.003 |
| 5 | +0.938 | +0.935 | +0.940 | +0.939 | +0.001 |
| 6 | +0.832 | +0.812 | +0.836 | +0.828 | +0.008 |
| 7 | +0.609 | +0.538 | +0.608 | +0.586 | +0.022 |
| 8 | +0.438 | +0.356 | +0.451 | +0.432 | +0.019 |
| 9 | +0.594 | +0.409 | +0.604 | +0.558 | +0.046 |
| 10 | +0.616 | +0.441 | +0.631 | +0.552 | +0.079 |
| 11 | +0.999 | +0.999 | +0.999 | +0.999 | +0.000 |

## Reference: GPT-2 small (§1.5 of Failure doc)

At matched PCA-$k$, POS-only $R^{2}_\text{full}\in[0.5,0.8]$ but POS-only $R^{2}_\text{sym}<0$ across every layer.  The gap in the **velocity-aware** variant has not yet been re-run for GPT-2 here; that is a required cross-check for v2 to make the comparison apples-to-apples.

## Verdict

- **Position-only (§1.5 analogue)**: max TEST gap = +0.185 (vs. GPT-2 where symmetric $R^2$ was *negative* everywhere).  Already a qualitative improvement.

- **Velocity-aware (the clean test)**: **Velocity-aware: symmetric-restricted fit tracks the unconstrained fit (max TEST gap = +0.079).  The per-step spring matrix is consistent with a symmetric Hessian, i.e. the dynamics is conservative on h.**

## Artefacts

- `splm_gpt2_baseline_jacsym_results.npz`
- `splm_gpt2_baseline_fig_jacsym.png`
