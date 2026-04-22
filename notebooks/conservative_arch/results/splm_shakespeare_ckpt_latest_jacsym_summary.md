# Jacobian-symmetry diagnostic -- scalar-potential LM (shakespeare_ckpt_latest)

Tests whether the per-step linear operator $M_\ell$ is **symmetric**.  A symmetric $M_\ell$ is what a conservative flow on a scalar potential must produce (Hessians of scalars are symmetric).  Skew-symmetric or symmetric-non-Hessian components indicate non-conservative dynamics.

We run TWO variants:

1. **Position-only (§1.5 analogue)**: $x_{\ell+1}-x_\ell \approx M_\ell x_\ell$.  For **damped second-order** dynamics (i.e. both this model and the Failure-doc §1 integrator), this fit is known to be confounded because the single-step transition mixes $x_\ell$ with the hidden velocity $v_\ell \approx x_\ell - x_{\ell-1}$.  The confound can manufacture apparent asymmetry even in genuinely conservative flows.
2. **Velocity-aware**: $x_{\ell+1}-x_\ell \approx A v_\ell + M_\ell x_\ell$ with $v_\ell = x_\ell - x_{\ell-1}$.  Here $M_\ell$ is the clean signal of the per-step spring matrix.  **This is the variant that tests conservativity.**

- Hidden dim `d = 128`, integration steps `L = 8`, PCA `k = 16`
- Train / test sentences: 40 / 10

## Per-layer fit quality

| layer | POS-only $R^{2}_\text{full}$ | POS-only $R^{2}_\text{sym}$ | VEL-aug $R^{2}_\text{full}$ | VEL-aug $R^{2}_\text{sym}$ | VEL-aug gap |
|--:|--:|--:|--:|--:|--:|
| 0 | +0.812 | +0.481 | +nan | +nan | +nan |
| 1 | +0.963 | +0.778 | +0.983 | +0.982 | +0.001 |
| 2 | +0.923 | +0.676 | +0.954 | +0.946 | +0.008 |
| 3 | +0.890 | +0.199 | +0.931 | +0.907 | +0.024 |
| 4 | +0.735 | +0.314 | +0.819 | +0.779 | +0.039 |
| 5 | +0.917 | +0.655 | +0.932 | +0.903 | +0.029 |
| 6 | +0.909 | +0.706 | +0.974 | +0.964 | +0.010 |
| 7 | +0.934 | +0.732 | +0.960 | +0.952 | +0.008 |

## Reference: GPT-2 small (§1.5 of Failure doc)

At matched PCA-$k$, POS-only $R^{2}_\text{full}\in[0.5,0.8]$ but POS-only $R^{2}_\text{sym}<0$ across every layer.  The gap in the **velocity-aware** variant has not yet been re-run for GPT-2 here; that is a required cross-check for v2 to make the comparison apples-to-apples.

## Verdict

- **Position-only (§1.5 analogue)**: max TEST gap = +0.691 (vs. GPT-2 where symmetric $R^2$ was *negative* everywhere).  Already a qualitative improvement.

- **Velocity-aware (the clean test)**: **Velocity-aware: symmetric-restricted fit tracks the unconstrained fit (max TEST gap = +0.039).  The per-step spring matrix is consistent with a symmetric Hessian, i.e. the dynamics is conservative on h.**

## Artefacts

- `splm_shakespeare_ckpt_latest_jacsym_results.npz`
- `splm_shakespeare_ckpt_latest_fig_jacsym.png`
