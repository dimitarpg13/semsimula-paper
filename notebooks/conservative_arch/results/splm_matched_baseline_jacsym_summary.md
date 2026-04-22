# Jacobian-symmetry diagnostic -- scalar-potential LM (matched_baseline)

Tests whether the per-step linear operator $M_\ell$ is **symmetric**.  A symmetric $M_\ell$ is what a conservative flow on a scalar potential must produce (Hessians of scalars are symmetric).  Skew-symmetric or symmetric-non-Hessian components indicate non-conservative dynamics.

We run TWO variants:

1. **Position-only (§1.5 analogue)**: $x_{\ell+1}-x_\ell \approx M_\ell x_\ell$.  For **damped second-order** dynamics (i.e. both this model and the Failure-doc §1 integrator), this fit is known to be confounded because the single-step transition mixes $x_\ell$ with the hidden velocity $v_\ell \approx x_\ell - x_{\ell-1}$.  The confound can manufacture apparent asymmetry even in genuinely conservative flows.
2. **Velocity-aware**: $x_{\ell+1}-x_\ell \approx A v_\ell + M_\ell x_\ell$ with $v_\ell = x_\ell - x_{\ell-1}$.  Here $M_\ell$ is the clean signal of the per-step spring matrix.  **This is the variant that tests conservativity.**

- Hidden dim `d = 128`, integration steps `L = 8`, PCA `k = 16`
- Train / test sentences: 40 / 10

## Per-layer fit quality

| layer | POS-only $R^{2}_\text{full}$ | POS-only $R^{2}_\text{sym}$ | VEL-aug $R^{2}_\text{full}$ | VEL-aug $R^{2}_\text{sym}$ | VEL-aug gap |
|--:|--:|--:|--:|--:|--:|
| 0 | +0.733 | +0.496 | +nan | +nan | +nan |
| 1 | +0.774 | +0.646 | +0.798 | +0.787 | +0.011 |
| 2 | +0.801 | +0.694 | +0.840 | +0.830 | +0.010 |
| 3 | +0.645 | +0.481 | +0.746 | +0.723 | +0.023 |
| 4 | +0.691 | +0.506 | +0.770 | +0.734 | +0.036 |
| 5 | +0.673 | +0.534 | +0.757 | +0.721 | +0.036 |
| 6 | +0.568 | +0.439 | +0.687 | +0.617 | +0.070 |
| 7 | +0.642 | +0.428 | +0.714 | +0.644 | +0.070 |

## Reference: GPT-2 small (§1.5 of Failure doc)

At matched PCA-$k$, POS-only $R^{2}_\text{full}\in[0.5,0.8]$ but POS-only $R^{2}_\text{sym}<0$ across every layer.  The gap in the **velocity-aware** variant has not yet been re-run for GPT-2 here; that is a required cross-check for v2 to make the comparison apples-to-apples.

## Verdict

- **Position-only (§1.5 analogue)**: max TEST gap = +0.237 (vs. GPT-2 where symmetric $R^2$ was *negative* everywhere).  Already a qualitative improvement.

- **Velocity-aware (the clean test)**: **Velocity-aware: symmetric-restricted fit tracks the unconstrained fit (max TEST gap = +0.070).  The per-step spring matrix is consistent with a symmetric Hessian, i.e. the dynamics is conservative on h.**

## Artefacts

- `splm_matched_baseline_jacsym_results.npz`
- `splm_matched_baseline_fig_jacsym.png`
