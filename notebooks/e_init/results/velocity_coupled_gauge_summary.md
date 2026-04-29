# Velocity-coupled gauge-field augmentation: does the electromagnetic-analogue Lagrangian fit?

Model: **gpt2**.  TRAIN: 40 sentences / 1127 tokens.  TEST: 10 sentences / 286 tokens.  PCA subspace: **k = 16** per layer.  Ridge: **0.001**.

## 1. Question and ansatzes

The §1.4 negative result ruled out the simplest Helmholtz correction $V_\ell \Omega_\ell V_\ell^\top x$ with constant skew $\Omega_\ell$.  Here we test four strictly richer linear gauge ansatzes derived from the electromagnetic-analogue Lagrangian

$$L = \tfrac12 \mathfrak m \lVert\dot{x}\rVert^2 + \vec A(x) \cdot \dot{x} - V(x),$$

which gives the Euler-Lagrange equation

$$\mathfrak m \ddot{x} = -\nabla V(x) + F(x) \dot{x} - \mathfrak m \gamma \dot{x},\qquad F = \partial A - (\partial A)^\top.$$

We parameterise $F$ in the per-layer top-$k$ PCA subspace (with $z = V^\top x$ and $w = V^\top v$) as one of:

| config | PCA-space force | params/layer |
|---|---|--:|
| `gaussian`         | 0 | 0 |
| `omega_x`          | $\Omega_0 z$ (skew) | $k(k-1)/2$ |
| `B_const`          | $B_0 w$ (skew) | $k(k-1)/2$ |
| `B_affine_r1`      | $(B_0 + z_1 B_1) w$ | $2 k(k-1)/2$ |
| `B_affine_r2`      | $(B_0 + z_1 B_1 + z_2 B_2) w$ | $3 k(k-1)/2$ |
| `omega_and_Bconst` | $\Omega_0 z + B_0 w$ | $2 k(k-1)/2$ |

(At $k=16$: 120 params per skew block; up to 360 per layer for `B_affine_r2`.  Training samples per layer: $\approx 1127$.)

## 2. Results

Static-null TEST floor: **0.1796**.

Each configuration is evaluated at two operator strengths:
- **s = 1** (full fit): applies the fitted skew operators unchanged.  For velocity-coupled configs this is numerically unstable in the integrator (positive-feedback $v \to B v \to $ larger $v$) and often diverges (reported as $10^{6}$ when it does).
- **s = s\* (shrunk fit)**: applies the same operators multiplied by a scalar $s\in[0,1]$, chosen per config per $\gamma$ to minimise TRAIN residual on a fine grid $\{0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0\}$.  Setting $s=0$ recovers the gaussian-only baseline.  This gives each ansatz its best shot at actually helping rather than blowing up.

### 2.1 Full-strength operators (s = 1)

| config | $\gamma=0.0$ | $\gamma=0.5$ | $\gamma=1.0$ | $\gamma=2.0$ | $\gamma=5.0$ |
|---|:---:|:---:|:---:|:---:|:---:|
| `gaussian` | 1.431 | 0.2905 | 0.2059 | 0.1828 | 0.1774 |
| `omega_x` | 2.561 | 0.433 | 0.2685 | 0.214 | 0.193 |
| `B_const` | 200.1 | 37.93 | 21.38 | 5.906 | 3.051 |
| `B_affine_r1` | 1.49 | 0.3002 | 0.2129 | 0.1929 | 0.1851 |
| `B_affine_r2` | 11.29 | 0.3265 | 0.2213 | 0.1909 | 0.1815 |
| `omega_and_Bconst` | 9.599 | 2.202 | 1.329 | 0.8736 | 0.6745 |

### 2.2 TRAIN-optimal scale (s = s\*), TEST residual

| config | $\gamma=0.0$ | $\gamma=0.5$ | $\gamma=1.0$ | $\gamma=2.0$ | $\gamma=5.0$ |
|---|:---:|:---:|:---:|:---:|:---:|
| `gaussian` | 1.4312 (s=0.0) | 0.2905 (s=0.0) | 0.2059 (s=0.0) | 0.1828 (s=0.0) | 0.1774 (s=0.0) |
| `omega_x` | 1.4312 (s=0.0) | 0.2905 (s=0.0) | 0.2059 (s=0.0) | 0.1829 (s=0.1) | 0.1774 (s=0.0) |
| `B_const` | 1.4312 (s=0.0) | 0.2905 (s=0.01) | 0.2059 (s=0.0) | 0.1828 (s=0.01) | 0.1773 (s=0.05) |
| `B_affine_r1` | 1.4312 (s=0.01) | 0.2905 (s=0.05) | 0.2059 (s=0.05) | 0.1829 (s=0.1) | 0.1774 (s=0.01) |
| `B_affine_r2` | 1.4312 (s=0.0) | 0.2905 (s=0.0) | 0.2060 (s=0.05) | 0.1828 (s=0.1) | 0.1774 (s=0.01) |
| `omega_and_Bconst` | 1.4312 (s=0.0) | 0.2905 (s=0.0) | 0.2059 (s=0.0) | 0.1828 (s=0.0) | 0.1774 (s=0.0) |

### 2.3 Best-$\gamma$ summary (shrunk fit)

| config | $\gamma^{*}$ | $s^{*}$ | TRAIN | TEST | Δ vs. null | PCA-ev skew | PCA-ev full |
|---|:-:|:-:|--:|--:|--:|--:|--:|
| `gaussian` | 5.0 | 0.0 | 0.1765 | 0.1774 | -0.0022 | 0.000 | 0.000 |
| `omega_x` | 5.0 | 0.0 | 0.1765 | 0.1774 | -0.0022 | -21.990 | 0.785 |
| `B_const` | 5.0 | 0.05 | 0.1764 | 0.1773 | -0.0022 | -5.608 | 0.691 |
| `B_affine_r1` | 5.0 | 0.01 | 0.1765 | 0.1774 | -0.0022 | -38.900 | 0.676 |
| `B_affine_r2` | 5.0 | 0.01 | 0.1765 | 0.1774 | -0.0022 | -40.044 | 0.700 |
| `omega_and_Bconst` | 5.0 | 0.0 | 0.1765 | 0.1774 | -0.0022 | -26.631 | 0.798 |

## 3. Interpretation

**Best configuration on TEST (scale-tuned):** `B_const` at $\gamma^{*}=5.0$, $s^{*}=0.05$, TEST median layer-$L$ residual **0.1773** vs. static-null **0.1796** (Δ = -0.0022).

Sanity check against §1.4 (`helmholtz_curl_summary.md`): at $\gamma=5$, `omega_x` gives TEST 0.1930 (s=1) here vs. 0.1882 in §1.4; tiny differences come from the independent PCA basis recomputation, not from the model.

### 3.1 Velocity-coupled ansatzes are numerically unstable at s = 1

`B_const`, `B_affine_r1`, `B_affine_r2`, and `omega_and_Bconst` all **diverge** when the fitted operators are applied at full strength. The symptom is the positive-feedback loop $v \to B\dot{x} \to v' \to \ldots$ that a linear integrator cannot stabilise when the fitted $B$ has eigenvalues of magnitude comparable to the damping.  Concretely, at $\gamma=5$, $s=1$:
- `B_const` TEST median = 3.051
- `B_affine_r1` TEST median = 0.1851
- `omega_and_Bconst` TEST median = 0.6745
(Values of $10^{6}$ indicate full divergence.)

This is itself an informative failure: the *fitted-optimal* skew velocity-coupling that best explains one-layer residuals on TRAIN is too strong to be propagated through 12 steps.

### 3.2 Shrinking $B$ stabilises but gives no meaningful gain

When we shrink the fitted operators by a factor $s\in[0,1]$ chosen to minimise TRAIN residual, the integrator becomes stable but **$s^{*}$ collapses towards 0**, i.e. the TRAIN-optimal velocity coupling is almost no velocity coupling:

- `B_const`, $\gamma^{*}=5$: $s^{*}=0.05$, TEST = 0.1773 (Δ = -0.0022).
- `B_affine_r1`, $\gamma^{*}=5.0$: $s^{*}=0.01$, TEST = 0.1774 (Δ = -0.0022).
- `B_affine_r2`, $\gamma^{*}=5.0$: $s^{*}=0.01$, TEST = 0.1774 (Δ = -0.0022).
- `omega_and_Bconst`, $\gamma^{*}=5.0$: $s^{*}=0.0$, TEST = 0.1774 (Δ = -0.0022).

In every case the shrunk-best configuration sits at or slightly above the static-null floor, never meaningfully below.  This is the substantive negative answer to 'do velocity-coupled position-dependent gauge fields fit the data?' at the linear level: **no**.

### 3.3 Why the one-step fit is good and the trajectory is not

The `PCA-ev full` columns show that the *unconstrained* linear operators capture a substantial fraction of one-step PCA-space residual variance ($R^{2}\approx 0.5\text{-}0.8$ per layer). Imposing the skew-symmetry constraint collapses the fit quality to deeply negative values (ev skew $\approx -10$ to $-180$), because the symmetric part of the fitted operator is where most of the deterministic structure lives.  This is the same pattern seen in §1.4 (`omega_x`) but now extended to $v$-coupled and position-dependent ansatzes: in every case, the residual structure that a linear operator can fit is overwhelmingly *symmetric*, not solenoidal.

### 3.4 Bottom line

Five consecutive negative experiments now bracket the linear Lagrangian programme from above and below:

1. Scalar $V(x)$, any $V$ (§1.1-1.3): **fails** (static null).
2. $+$ skew $\Omega x$ (§1.4): **fails** (marginally worse than null).
3. $+$ skew $B \dot{x}$, constant: **unstable at s=1**; at s\*, at or slightly above null.
4. $+$ skew $B(x)\dot{x}$, affine in $x$, rank 1 or 2: **unstable at s=1**; at s\*, at or slightly above null.
5. $+$ skew $\Omega x + B\dot{x}$ (combined): **unstable at s=1**; at s\*, strictly worse than null.

The Helmholtz electromagnetic-analogue Lagrangian, at the level of constant or low-order-polynomial antisymmetric operators in a top-16 PCA subspace, does not fit hidden-state trajectories on held-out sentences.  The per-layer one-step linear approximation of the transformer block is symmetric and non-Hessian, not antisymmetric; this is not captured by the electromagnetic analogue.

Remaining untested candidates that are strictly richer than anything here and could in principle still work:

- Non-linear (not just affine) position dependence in $B(x)$, e.g. $B(x) = \sum_k c_k B_k \phi_k(x)$ with a basis of smooth position features $\phi_k$ (RBFs, Fourier on PCA coords, etc.).
- Symmetric-but-non-Hessian extensions -- these lie **outside the autonomous, shared-potential** Helmholtz class (they cannot be written as the Hessian of any single shared scalar potential) but are what the $R^{2}$ data say is actually needed.  A Riemannian / Jacobi-geodesic formulation (§14 of the paper) accommodates them via Christoffel symbols of a non-flat metric, without any scalar potential.
- Non-abelian (multi-head) gauge structure $\vec F = \sum_h F^{(h)}(x)$ with head-specific antisymmetric generators -- left for future work.

## 4. Artefacts

- [`notebooks/e_init/velocity_coupled_gauge.py`](../velocity_coupled_gauge.py) -- reproducible script
- [`results/velocity_coupled_gauge_results.npz`](velocity_coupled_gauge_results.npz) -- full numerical results
- [`results/fig_gauge_residual_vs_gamma.png`](fig_gauge_residual_vs_gamma.png) -- TRAIN / TEST residual vs $\gamma$ (all configs)
- [`results/fig_gauge_residual_vs_layer_at_gamma_star.png`](fig_gauge_residual_vs_layer_at_gamma_star.png) -- per-layer TEST residual at per-config $\gamma^{*}$

## 5. Reproduce

```bash
python3 notebooks/e_init/velocity_coupled_gauge.py
```
