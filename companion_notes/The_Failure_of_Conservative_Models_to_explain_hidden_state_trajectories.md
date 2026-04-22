# The Failure of Conservative Models to Explain Hidden-State Trajectories

*A theoretical post-mortem on the E-init programme and its functional-form
extensions.  Targeted at paper v2.  Status: working document synthesising
empirical and theoretical observations, not a submitted result.*

*Companion experiments (scripts / notebook):*
- `notebooks/e_init/e_init_validation.ipynb` &nbsp;--- v1 E-init with the
  Gaussian well (§1.1 below)
- `notebooks/e_init/extended_gamma_and_first_order.py` &nbsp;--- extended
  $\gamma$ sweep and pure first-order overdamped integrator (§1.2)
- `notebooks/e_init/well_functional_form_comparison.py` &nbsp;--- seven-form
  functional comparison and integrator rerun (§1.3)
- `notebooks/e_init/helmholtz_curl_augmented.py` &nbsp;--- Helmholtz
  augmentation: add a linear solenoidal term $V_\ell\,\Omega_\ell\,V_\ell^{\top}x$
  to the 2nd-order integrator (§1.4)
- `notebooks/e_init/velocity_coupled_gauge.py` &nbsp;--- electromagnetic-
  analogue Lagrangian: add a skew-symmetric velocity-coupled term
  $B(x)\,\dot x$, with constant $B$ and with affine position-dependent
  $B(x)$ (§1.5)

*Companion result files:* all artefacts under
`notebooks/e_init/results/`.  A complete, navigable index appears in §9.

### How to reproduce

From the repository root, with a Python environment that has
`torch`, `transformers`, `numpy`, `scipy`, and `matplotlib` installed:

```bash
# (1) v1 E-init  -- open the notebook and run all cells:
jupyter nbconvert --execute --to notebook --inplace \
    notebooks/e_init/e_init_validation.ipynb

# (2) Extended gamma sweep + first-order integrator (~1 min on MPS):
python3 notebooks/e_init/extended_gamma_and_first_order.py

# (3) Functional-form sweep + per-form integrator (~2 min on MPS):
python3 notebooks/e_init/well_functional_form_comparison.py

# (4) Helmholtz-augmented integrator: add skew-symmetric linear
#     solenoidal term; train/test on 40/10 split (~45 s on MPS):
python3 notebooks/e_init/helmholtz_curl_augmented.py

# (5) Velocity-coupled gauge augmentation: constant skew B*v and
#     affine-in-x B(x)*v, plus scale-tuned stability sweep (~6 min on MPS):
python3 notebooks/e_init/velocity_coupled_gauge.py
```

Each stage writes its summary Markdown into `notebooks/e_init/results/`
alongside raw `.npz`, `.json`, and `.png` artefacts. The five summary
files
[`extended_gamma_first_order_summary.md`][sum2],
[`well_form_comparison_summary.md`][sum3],
[`helmholtz_curl_summary.md`][sum4],
[`velocity_coupled_gauge_summary.md`][sum5],
and the v1 notebook itself are the primary sources for the results
quoted throughout this document.

[sum2]: ../notebooks/e_init/results/extended_gamma_first_order_summary.md
[sum3]: ../notebooks/e_init/results/well_form_comparison_summary.md
[sum4]: ../notebooks/e_init/results/helmholtz_curl_summary.md
[sum5]: ../notebooks/e_init/results/velocity_coupled_gauge_summary.md

---

## 0. TL;DR

Five successive empirical experiments on GPT-2 small have now established,
with increasing breadth, that **no linear Lagrangian in the top-16
per-layer PCA subspace reproduces GPT-2 hidden-state trajectories at the
layer-$L$ level better than the trivial "freeze at $h_0$" baseline.**
Specifically:

1. Scalar potential $V(x)$, any bounded-attractive or power-law form
   (§1.1--1.3): **fails** (ties static null).
2. $+$ linear skew-symmetric $\Omega x$ in the top-$k$ PCA subspace
   (§1.4): **fails** (marginally worse than null).
3. $+$ linear skew-symmetric $B\,\dot x$, constant per layer (§1.5):
   **unstable at full strength; at TRAIN-optimal shrinkage factor
   $s^{*}\approx 0.05$, ties null**.
4. $+$ affine-in-$x$ skew $B(x)\,\dot x$ with rank 1 or 2 position
   dependence (§1.5): **unstable at full strength; at $s^{*}\approx 0.01$,
   ties null**.
5. $+$ both $\Omega x$ and $B\,\dot x$ simultaneously (§1.5): **unstable
   at full strength; at $s^{*}=0$ (i.e. the optimiser turns the gauge
   term off), ties null**.

The combined evidence is a signature of a structural property of the
transformer: **the layer-wise flow on hidden states is neither a
conservative (gradient) flow nor any linear gauge field on top of one.**
The best unconstrained constant *linear* per-layer operator on $(x, v)$
in the top-16 PCA subspace captures ~70--80 % of one-step variance --
but its *antisymmetric* (skew / solenoidal) projection is essentially
orthogonal to the observed residual, and its full unconstrained variant
blows up across 12 integration steps because its symmetric part is not a
Hessian. The residual structure that a linear per-layer operator can
fit is overwhelmingly **symmetric and non-Hessian** -- outside the
Helmholtz (grad + curl) decomposition altogether.

What remains consistent with the data is a flow that is
**state-dependent in a non-polynomial way**, rank-deficient by layer,
and whose best natural formalism is the Riemannian / Jacobi-geodesic
framework of §14 (non-flat metric with Christoffel symbols encoding both
non-integrable rotation *and* non-Hessian stretching without requiring
a scalar potential).

The paper's local STP-acceleration identity (Result 1) and the per-phrase
Jacobi-metric programme are *not* invalidated by this finding. What is
invalidated is the stronger claim that a single scalar potential -- or
any scalar-plus-linear-gauge ansatz at constant or affine order -- drives
the global multi-layer trajectory. This document makes the argument
explicit and lists what should replace it in v2.

---

## 1. The empirical case

Five experiments have now been run, in increasing generality, against the
same GPT-2 corpus (50 sentences, 1,413 per-token trajectories) with
per-sentence per-layer centering:

### 1.1 E-init with Gaussian well (paper v1, §11.7 and §13)

The Euler--Lagrange equation

$$\mathfrak{m}\,\ddot{\vec x} = -2ab\,\vec x\,e^{-b\|\vec x\|^2}
  -\mathfrak{m}\gamma\,\dot{\vec x} \tag{1.1}$$

integrated forward from $(\vec x_0, \vec v_0 = \vec x_1 - \vec x_0)$ with
per-layer fitted Gaussian $(a,b)$, under-performs the static null at
$\gamma^{*}=1.0$ (median layer-$L$ residual $0.2064$ vs. $0.1773$). This
was reported in v1 as the E-init negative result.

**Reproducibility trail for §1.1**

- Source notebook (all stages including well fit, integrator, and
  residual plots): [`notebooks/e_init/e_init_validation.ipynb`][nb1].
  The symplectic Euler integrator that solves eq. (1.1) is the function
  `symplectic_euler_step` in Stage 5 of the notebook; the per-sentence
  residual sweep is Stage 10.
- Fitted per-sentence Gaussian-well parameters
  $\{(a_\ell, b_\ell, R^2_\ell)\}$:
  [`notebooks/e_init/results/well_params_ps.json`][wpps].
- Raw per-token residuals, $\gamma$ sweep, and summary arrays:
  [`notebooks/e_init/results/e_init_results_ps.npz`][nv1]
  (keys `rho_einit_star_ps`, `rho_static_star_ps`, `rho_linear_star_ps`,
  `gammas_ps`, `median_final_ps`, `gamma_star_ps`).
- Figures: [`fig_A_residual_vs_layer_ps.png`][fa1] and
  [`fig_B_residual_vs_logw_ps.png`][fb1] in the results directory; the
  per-layer $R^2$ curve is [`stage10_well_r2_vs_layer_ps.png`][s10].

[nb1]: ../notebooks/e_init/e_init_validation.ipynb
[wpps]: ../notebooks/e_init/results/well_params_ps.json
[nv1]: ../notebooks/e_init/results/e_init_results_ps.npz
[fa1]: ../notebooks/e_init/results/fig_A_residual_vs_layer_ps.png
[fb1]: ../notebooks/e_init/results/fig_B_residual_vs_logw_ps.png
[s10]: ../notebooks/e_init/results/stage10_well_r2_vs_layer_ps.png

### 1.2 Extended $\gamma$ sweep and first-order overdamped limit

Extending the v1 sweep from $\gamma \in \{0, \dots, 1\}$ to
$\gamma \in [0, 50]$ on the same integrator (1.1), and running a
parallel pure first-order flow

$$\vec x_{\ell+1} \;=\; \vec x_\ell \;-\; \eta\,\nabla V(\vec x_\ell)
  \tag{1.2}$$

with the *same* per-sentence Gaussian well, yields two findings:

1. The correct $\gamma^{*} = 5.0$ (not $\gamma^{*}=1.0$) reduces the median
   layer-$L$ residual to $0.1768$ -- _exactly_ the static-null floor
   $0.1773$ to three decimal places. At $\gamma^{*}=5.0$ the 2nd-order
   integrator beats the static null on $68.44\%$ of tokens, but by tiny
   margins (median improvement $0.0011$).
2. The first-order flow (1.2) reproduces the static null to
   floating-point precision at *every* $\eta \in \{0.01, 0.05, \dots, 25\}$
   tried: $\max_t |\rho^{(L)}_{t,\mathrm{1st}} - \rho^{(L)}_{t,\mathrm{static}}| = 0$.
   The mechanism is quantitative (see §4.2).

Both integrators asymptote to "predict $h_0$ everywhere", not to the
observed trajectory.

**Reproducibility trail for §1.2**

- Script (self-contained; re-extracts GPT-2, refits wells, runs both
  sweeps): [`notebooks/e_init/extended_gamma_and_first_order.py`][s1].
  The second-order integrator is the function
  `integrate_second_order`; the first-order flow is
  `integrate_first_order`.
- Results archive: [`extended_gamma_first_order_results.npz`][r1]
  (keys `gammas_full`, `median_layerL_second`, `etas`,
  `median_layerL_first`, `rho_static_layerL`, `rho_second_star_layerL`,
  `rho_first_star_layerL`, `gamma_star_full`, `eta_star`,
  plus per-layer medians).
- Figures: [`fig_extended_gamma.png`][f1]
  (residual vs. $\gamma$, with static-null reference line) and
  [`fig_first_order_eta.png`][f2] (residual vs. $\eta$, with both the
  static-null and best-2nd-order reference lines).
- Human-readable report: [`extended_gamma_first_order_summary.md`][sum2].

[s1]: ../notebooks/e_init/extended_gamma_and_first_order.py
[r1]: ../notebooks/e_init/results/extended_gamma_first_order_results.npz
[f1]: ../notebooks/e_init/results/fig_extended_gamma.png
[f2]: ../notebooks/e_init/results/fig_first_order_eta.png

### 1.3 Non-Gaussian functional-form sweep

Seven functional forms -- **harmonic** $a r^2$,
**Gaussian** $a(1-e^{-br^2})$, **Morse** $a(1-e^{-br})^2$,
**rational / Lorentzian** $a b r^2/(1+b r^2)$,
**log-saturation** $a\log(1+br^2)$,
**Weibull / stretched-exponential** $a(1-e^{-br^\alpha})$,
and **power** $a r^p$ -- were fit layer-by-layer to the
$(\|x\|,\text{NTP loss})$ scatter with per-sentence centering, and each
was plugged (via its analytic gradient) into the damped 2nd-order
integrator (1.1). Two empirical findings:

- **On fit quality** the forms are nearly indistinguishable on raw
  pooled scatter ($R^2 \approx 0.107\text{-}0.108$ at middle layers
  2-11, differing only in the fourth decimal). On the 15 equal-count
  radially binned medians, which isolate the deterministic trend, $R^2$
  jumps to $\approx 0.82\text{-}0.90$ at layers 3-7 -- again
  essentially the same across all seven forms.
- **On integrator behaviour** every form, and the AIC-selected mixed
  map, produces median layer-$L$ residual $0.1768$ at $\gamma^{*}=5.0$,
  identical to Gaussian to four decimals.

The mechanism is quantitative: at a typical per-sentence-centered radius
$r\approx 300$ (middle layers), the fitted force-per-displacement
$k(r) = V'(r)/r$ is pinned in a narrow band $\sim (0.7\text{-}3.1)\times 10^{-6}$
across all seven forms, set by the data geometry -- well depth
$a\approx 8$ nats divided by squared data scale $\|x\|^2\sim 10^{5}$.
See §4.3 for the calculation.

**Reproducibility trail for §1.3**

- Script (self-contained; fits all seven forms per layer and reruns
  the integrator under each):
  [`notebooks/e_init/well_functional_form_comparison.py`][s2]. The
  seven forms and their analytic gradients are in the `FORMS`
  dictionary at the top of the file; the integrator is
  `integrate_general` using per-layer `grad_coeff` functions.
- All fitted parameters (form $\times$ layer):
  [`well_form_comparison_params.json`][pj] &nbsp;(each entry gives
  `params`, `r2`, `aic`, `n`).
- Machine-readable $R^2$ and AIC table:
  [`well_form_comparison_r2.csv`][rc].
- Integrator residuals per form / $\gamma$:
  [`well_form_integrator_results.npz`][ni] &nbsp;(keys
  `gammas_sweep`, `static_null_layerL`, one `{label}_layerL` and
  `{label}_med_per_gamma` per configuration).
- Figures:
  [`fig_well_form_r2_vs_layer.png`][fwr] (raw $R^2$ per layer, one
  curve per form),
  [`fig_well_form_r2_vs_layer_binned.png`][fwb] (binned-median $R^2$),
  and scatter plots with all seven fits overlaid at three
  representative layers:
  [layer 3][fs3], [layer 6][fs6], [layer 9][fs9].
- Human-readable report: [`well_form_comparison_summary.md`][sum3].

[s2]: ../notebooks/e_init/well_functional_form_comparison.py
[pj]: ../notebooks/e_init/results/well_form_comparison_params.json
[rc]: ../notebooks/e_init/results/well_form_comparison_r2.csv
[ni]: ../notebooks/e_init/results/well_form_integrator_results.npz
[fwr]: ../notebooks/e_init/results/fig_well_form_r2_vs_layer.png
[fwb]: ../notebooks/e_init/results/fig_well_form_r2_vs_layer_binned.png
[fs3]: ../notebooks/e_init/results/fig_well_form_scatter_layer3.png
[fs6]: ../notebooks/e_init/results/fig_well_form_scatter_layer6.png
[fs9]: ../notebooks/e_init/results/fig_well_form_scatter_layer9.png

### 1.4 Helmholtz-augmented integrator: adding a linear solenoidal term

Because §1.1-1.3 tests all scalar potentials, we now follow the
Helmholtz programme and *add* a divergence-free correction to the force.
The simplest choice is a layer-local linear solenoidal field in the
top-$k$ PCA subspace of training hidden states,

$$\mathfrak m\,\ddot x
  \;=\; -\nabla V(x)
  \;+\; V_\ell\,\Omega_\ell\,V_\ell^{\top}x
  \;-\; \mathfrak m\,\gamma\,\dot x,\qquad
  \Omega_\ell=-\Omega_\ell^{\top}\in\mathbb R^{k\times k}, \tag{1.3}$$

with $V_\ell\in\mathbb R^{d\times k}$ the per-layer top-$k$ PCA basis.
The skew constraint makes the extra term divergence-free in the linear
sense ($\nabla\cdot (V\Omega V^{\top}x)=\operatorname{tr}\Omega=0$). We
split the corpus 40 train / 10 test (balanced across the five domains),
fit $\Omega_\ell$ per layer by weighted OLS on observed
$f_\ell/m=(1+\gamma)v_{\ell+1}-v_\ell$ after subtracting the Gaussian
conservative force, then integrate on both folds with $k=16$ and ridge
$10^{-3}$. As an *upper bound*, and to bracket how much of any change
comes specifically from the skew sub-component, we also fit an
unconstrained linear operator $M_\ell\in\mathbb R^{k\times k}$ (which
contains both a skew part and a symmetric part with no Hessian
constraint).

Three empirical findings (raw numbers in
[`helmholtz_curl_results.npz`][hm1] and
[`helmholtz_curl_summary.md`][sum4]):

1. **The skew-symmetric $\Omega$ is orthogonal to the observed residual**
   at every layer. Its per-layer fit-quality $R^{2}$ averages **−13 to
   −21** across the five $\gamma$ values tested (the negative
   $R^{2}$ means applying $\Omega$ to $x$ produces a vector with _larger_
   magnitude than the residual and essentially perpendicular to it).
2. **The unconstrained linear $M$ captures ~78 % of per-layer residual
   variance** in PCA space at $\gamma^*=5$ -- a genuinely strong linear
   approximation of what each transformer block does in one layer.
3. **Neither augmentation beats the static null on the held-out test
   set.** At the best $\gamma^*=5$ the TEST medians are
   $0.1774$ (scalar-only), $0.1882$ (+skew $\Omega$), $0.2890$
   (+full linear $M$), vs. static null $0.1796$.

The fit-quality-to-trajectory-quality gap is the signature of
compounding extrapolation error in a non-linear system: a linear
approximation that is locally excellent can, when iterated across 12
layers, diverge catastrophically because LayerNorm and softmax make the
true block Jacobian state-dependent.

The key *structural* lesson is the asymmetry between the skew and
symmetric parts of $M$:

> The per-layer hidden-state dynamics are dominated by a
> **symmetric-linear stretching** component (which is not the Hessian
> of any scalar potential), and *not* by a rotational one.

So although §3 below shows that transformer attention _does_ generate a
non-zero curl in principle, in practice the observed layer-wise
displacement field is not well approximated as a sum of a gradient and
a linear rotation. The rotational / solenoidal component exists, but if
it is to fit the data it must be (a) position-dependent (non-linear in
$x$), (b) coupled to velocity (as in the electromagnetic-analogue
Lagrangian of §6), and/or (c) rank-split across attention heads in a way
that cannot be captured by a single antisymmetric $\Omega_\ell$.

**Reproducibility trail for §1.4**

- Script (self-contained; 40/10 train/test split, per-layer PCA and
  $\Omega_\ell$ fit, augmented symplectic Euler, $\gamma$ sweep):
  [`notebooks/e_init/helmholtz_curl_augmented.py`][s3]. Key functions:
  `fit_linear_in_pca` (the $\Omega$ / $M$ fit) and
  `integrate_helmholtz` (the augmented integrator).
- Results archive: [`helmholtz_curl_results.npz`][hm1]
  (keys `gammas`, `base_test`, `omega_test`, `mfull_test`,
  `static_test`, `mean_frac_ev_omega`, `mean_frac_ev_mfull`,
  per-layer medians for each of the three configurations).
- Figures: [`fig_helmholtz_residual_vs_gamma.png`][hmf1]
  (TRAIN / TEST median residual vs. $\gamma$ for all three
  configurations), and
  [`fig_helmholtz_residual_vs_layer_at_gamma_star.png`][hmf2]
  (per-layer TEST residual at $\gamma^{*}=5$).
- Human-readable report: [`helmholtz_curl_summary.md`][sum4].

[s3]:   ../notebooks/e_init/helmholtz_curl_augmented.py
[hm1]:  ../notebooks/e_init/results/helmholtz_curl_results.npz
[hmf1]: ../notebooks/e_init/results/fig_helmholtz_residual_vs_gamma.png
[hmf2]: ../notebooks/e_init/results/fig_helmholtz_residual_vs_layer_at_gamma_star.png

### 1.5 Velocity-coupled electromagnetic-analogue gauge (and position-dependence)

Because §1.4 tested the simplest *position*-coupled linear solenoidal
term and it failed, we now test the richer *velocity*-coupled class
derived from the electromagnetic-analogue Lagrangian

$$L = \tfrac{1}{2}\mathfrak m\,\lVert\dot x\rVert^{2}
  + \vec A(\vec x)\cdot\dot x - V(\vec x),\qquad
  F_{ij}(\vec x) = \partial_i A_j - \partial_j A_i, \tag{1.4}$$

whose Euler--Lagrange equation is
$\mathfrak m\,\ddot x = -\nabla V + F(\vec x)\,\dot x - \mathfrak m\gamma\,\dot x$.
In the per-layer top-$k$ PCA subspace (same split and basis as §1.4)
we parameterise $F$ as skew-symmetric and try four progressively richer
variants, with $z=V^{\top}x$, $w=V^{\top}v$:

| config | PCA-space extra force | extra parameters per layer |
|---|---|--:|
| `B_const`          | $B_0\,w$                              | $k(k-1)/2$ |
| `B_affine_r1`      | $(B_0 + z_1 B_1)\,w$                  | $k(k-1)$ |
| `B_affine_r2`      | $(B_0 + z_1 B_1 + z_2 B_2)\,w$        | $3k(k-1)/2$ |
| `omega_and_Bconst` | $\Omega_0\,z + B_0\,w$                | $k(k-1)$ |

All coefficient matrices are skew-symmetric $k\times k$; fitting uses
the same weighted OLS of eq.~(1.3) after subtracting the Gaussian
conservative part, with ridge $10^{-3}$.

Three quantitative findings (full numbers in
[`velocity_coupled_gauge_summary.md`][sum5], raw data in
[`velocity_coupled_gauge_results.npz`][vc1]):

1. **At full strength ($s=1$) the velocity-coupled integrators
   diverge.** The fitted skew-symmetric $B$ has eigenvalues whose
   magnitudes, after $12$ symplectic-Euler steps, drive positive
   feedback $v\to B\,\dot x\to v'\to\dots$. At $\gamma=5$, $s=1$, TEST
   medians are $3.05$ (`B_const`), $0.185$ (`B_affine_r1`), $0.18$
   (`B_affine_r2`), $0.67$ (`omega_and_Bconst`) -- at the low-$\gamma$
   end many reach $10^{1}$--$10^{2}$ or overflow to NaN.
2. **Shrinking the fitted operators by a TRAIN-optimal factor
   $s^{*}\in[0,1]$ stabilises the integrator -- and $s^{*}$
   collapses towards 0 for every config.** The TRAIN-optimal $s^{*}$
   at $\gamma=5$ is
   $s^{*}_{\mathrm{B\_const}}=0.05$,
   $s^{*}_{\mathrm{B\_affine\_r1}}=0.01$,
   $s^{*}_{\mathrm{B\_affine\_r2}}=0.01$,
   $s^{*}_{\mathrm{omega\_and\_Bconst}}=0$.
   That is, when asked to pick how much of the fitted gauge field to
   apply, the optimiser prefers *almost none of it*.
3. **At the TRAIN-optimal shrinkage, TEST residual equals the
   gaussian-only baseline (and hence the static null) to within
   $\sim 10^{-4}$.** Concretely:

   | config | $\gamma^{*}$ | $s^{*}$ | TEST layer-$L$ | Δ vs. null |
   |---|:-:|:-:|--:|--:|
   | `gaussian`         | 5 | 0    | 0.1774 | −0.0022 |
   | `omega_x`          | 5 | 0    | 0.1774 | −0.0022 |
   | `B_const`          | 5 | 0.05 | 0.1773 | −0.0023 |
   | `B_affine_r1`      | 5 | 0.01 | 0.1774 | −0.0022 |
   | `B_affine_r2`      | 5 | 0.01 | 0.1774 | −0.0022 |
   | `omega_and_Bconst` | 5 | 0    | 0.1774 | −0.0022 |

All six rows agree to four decimals. The "improvement" that any gauge
term delivers is indistinguishable from zero on held-out data.

The fit-quality vs. trajectory-quality gap is as in §1.4, slightly
worse: the *unconstrained* linear operator on $(x,v)$ explains
$R^{2}\approx 0.5$--$0.8$ of one-step PCA-space residual variance per
layer, but its skew projection has $R^{2}\in[-180,-5]$ -- the
symmetric, non-Hessian part is doing all the real work, and no
electromagnetic-analogue Lagrangian captures that part.

**Reproducibility trail for §1.5**

- Script: [`notebooks/e_init/velocity_coupled_gauge.py`][s4]. Key
  functions: `fit_gauge_in_pca` (joint fit of $\Omega_0$, $B_0$, $B_i$),
  `gauge_force_over_m` (applies any subset of the fitted operators with
  optional scalar shrinkage $s$), `integrate` (symplectic step),
  `residuals` (per-sentence per-token per-layer residual).
- Raw data: [`velocity_coupled_gauge_results.npz`][vc1] -- keys
  `gammas`, `train_s1`, `test_s1`, `train_star`, `test_star`, `best_s`,
  `ev_skew`, `ev_full`, `per_layer_test_s1`, `per_layer_test_star`,
  `config_names` (a (6, 5) grid per metric).
- Figures: [`fig_gauge_residual_vs_gamma.png`][vcf1] (TRAIN / TEST
  panels; log y-axis because divergent configs would otherwise squash
  the plot) and
  [`fig_gauge_residual_vs_layer_at_gamma_star.png`][vcf2] (per-layer
  TEST residual at per-config $\gamma^{*}$ with the shrinkage-tuned
  operators).
- Report: [`velocity_coupled_gauge_summary.md`][sum5].

[s4]:   ../notebooks/e_init/velocity_coupled_gauge.py
[vc1]:  ../notebooks/e_init/results/velocity_coupled_gauge_results.npz
[vcf1]: ../notebooks/e_init/results/fig_gauge_residual_vs_gamma.png
[vcf2]: ../notebooks/e_init/results/fig_gauge_residual_vs_layer_at_gamma_star.png

### 1.6 What all five experiments have in common

Three of them (§1.1-1.3) assume, explicitly, that the layer-wise force
on the hidden state is the gradient of some scalar function of
position: $\vec F(\vec x) = -\nabla V(\vec x)$. The fourth (§1.4)
adds a linear solenoidal correction $V\Omega V^{\top}x$. The fifth
(§1.5) adds the velocity-coupled analogue $F(x)\,\dot x$ from the
electromagnetic Lagrangian, including position-dependent $F(x)$ at
rank 1 and rank 2 and the combined $\Omega x + B\dot x$ ansatz. **All
five produce the same TEST residual floor, to four decimal places, and
that floor equals the static null.** This is the empirical fingerprint
of a structural limitation, not of any particular choice of $V$, $A$,
or $B(x)$. Moreover the §1.5 fit-quality ceiling tells us that the
one-step residual **can** be captured with $R^{2}\sim 0.8$ by a linear
operator on $(x,v)$ -- but only if we allow that operator a **symmetric
non-Hessian** component, which is outside the Helmholtz decomposition.
The next natural formalism is therefore a non-flat Riemannian
(Jacobi-metric) one in which Christoffel symbols of a generic metric
encode both rotational and non-Hessian stretching without any scalar
potential (§14 of the paper).

---

## 2. What "conservative" means here

A force field $\vec F(\vec x)$ on hidden-state space is **conservative**
if and only if

$$\oint_{\mathcal{C}} \vec F \cdot d\vec x = 0$$

for every closed loop $\mathcal{C}$, or equivalently
$\nabla \times \vec F = 0$ (zero curl in every 2-plane), or equivalently
there exists a scalar potential $V$ such that $\vec F = -\nabla V$.
A dynamical system $\mathfrak{m}\,\ddot{\vec x} = \vec F(\vec x) + \text{damping}$
with conservative $\vec F$ is fully described by the two scalar
quantities $V$ and $T=\tfrac{1}{2}\mathfrak{m}\lVert\dot x\rVert^2$ and
conserves their sum along undamped trajectories.

The paper's current Lagrangian construction
$$L = T - V,\qquad V = V(x),$$
is of this type. **Any integrator derived from it, at any damping or
any order, cannot produce motion that has non-zero circulation -- that
is, it cannot generate rotational trajectories around a point where
$V$ is locally flat, and it cannot reach layer-$L$ states that a pure
gradient flow from $(\vec x_0, \vec v_0)$ would never reach.**

The Helmholtz decomposition of an arbitrary (smooth, sufficiently
decaying) vector field on $\mathbb{R}^d$ is

$$\vec F(\vec x) \;=\; \underbrace{-\nabla V(\vec x)}_{\text{conservative}}
  \;+\; \underbrace{\vec F_{\mathrm s}(\vec x)}_{\text{solenoidal}},
  \qquad \nabla \cdot \vec F_{\mathrm s} = 0,\;
  \nabla \times \vec F_{\mathrm s} \neq 0 \text{ in general}.$$

Our experiments effectively fit the best scalar $V$ (in a broad
functional class) to the observed displacement data. What remains --
the $\vec F_{\mathrm s}$ component -- is by construction invisible to
that fitting procedure, and by dynamics is not reproduced by any
integrator driven by $V$ alone. In our data, the $\vec F_{\mathrm s}$
component is not a small correction; it **carries most of the
observed layer-to-layer motion**.

---

## 3. Why transformer attention is non-conservative

The hidden-state update between consecutive layers is, schematically,

$$h_t^{(\ell+1)} \;=\; h_t^{(\ell)} \;+\;
  \underbrace{\mathrm{FFN}^{(\ell)}\!\bigl(h_t^{(\ell)}\bigr)}_{\text{per-token, potentially gradient-like}}
  \;+\;
  \underbrace{\sum_{j\le t}\alpha^{(\ell)}_{t,j}(h^{(\ell)}_{\le t})\,V^{(\ell)} h_j^{(\ell)}}_{\text{attention update}}.$$

The attention term alone is already sufficient to destroy conservativity
at the layer-wise level. Three separate properties, any one of them
fatal to the gradient-field reading:

### 3.1 Path dependence

The attention update at position $t$ is a function of the **full prefix**
$h_{<t}$, not of $h_t$ alone:

$$\Delta h_t \;=\; f\bigl(h_t,\,h_{t-1},\,\dots,\,h_0\bigr).$$

A conservative force on $h_t$ must depend on $h_t$ only (or on $h_t$ and
$\dot h_t$, for dissipative systems). Any dependence on the history
$h_{<t}$ means that two different _paths_ arriving at the same point
$h_t$ receive, in general, different updates. That is precisely the
defining property of a **path-dependent** (hence non-conservative) force.

In the paper's layer-as-time convention this translates to: the force
field driving the radial coordinate depends on the trajectory of _all
other tokens_ in the sentence, not just on the state of the token whose
trajectory we are tracking. No scalar potential $V(\vec x_t)$ can
encode that.

### 3.2 Non-symmetry

The attention matrix $\alpha^{(\ell)}$ is asymmetric for two reasons:
causal masking ($\alpha_{t,j}=0$ for $j>t$) and the fact that the
query-key inner product $(Q_t)^\top K_j$ is _not_ symmetric in
$(t, j)$ because $Q$ and $K$ are distinct linear maps of the same
$h^{(\ell)}$.

As a consequence: the infinitesimal contribution of moving token A
towards token B's semantic region is **not** equal and opposite to the
contribution of moving B towards A. In a force-field language,

$$\vec F_{A\to B} \;\neq\; -\vec F_{B\to A}.$$

Gradient fields always satisfy this reciprocity (it is the integrability
condition $\partial_i F_j = \partial_j F_i$). Attention does not.
Moving from semantic region A to semantic region B is not
undone by the reverse force field; the asymmetry is encoded in the
$(Q,K)$-pair and survives the softmax.

### 3.3 Rank-deficient per-head "torques"

Multi-head attention writes, for each head $h$,

$$\Delta h_t^{(\ell,h)} \;=\; \sum_j \alpha^{(\ell,h)}_{t,j}\,V^{(\ell,h)}\,h_j^{(\ell)},$$

and then projects through $W^O$. Each head operates in its own
rank-$(d/H)$ subspace, with its own $(Q_h,K_h,V_h)$ triple. The heads'
contributions are not, in general, gradients of a common scalar -- they
are _linearly independent_ directional updates. Geometrically this is
a collection of rank-deficient "torques" on the residual stream; the
sum is a high-dimensional field with no reason at all to be curl-free.

Even if each head _individually_ were to happen to be gradient-like
(which there is no reason to expect), the multi-head sum is conservative
only if the heads' individual potentials are consistent -- a highly
non-generic coincidence that nothing in training enforces.

---

## 4. Consequences: the solenoidal component and semantic rotation

Taking the three mechanisms in §3 together, the layer-wise effective
force on hidden states must have a non-zero curl in general. By
Helmholtz,

$$\vec F = -\nabla V + \vec F_{\mathrm s},\qquad \vec F_{\mathrm s}\neq 0.$$

The solenoidal component $\vec F_{\mathrm s}$ represents
**semantic rotation** -- trajectories that curve through meaning-space
**without** being attracted to, or repelled from, any fixed point. A
decoder processing, say, a subordinate clause is rotating the hidden
state through syntactic space before returning to the main semantic
thread. No scalar potential captures this.

This is the correct physical reading of our four experiments. The
conservative part of the flow is real but small: §1.3 shows that a
weak, nearly-harmonic, roughly $\sim 10^{-6}$-strength radial force is
what all seven functional forms converge to. The $\sim 0.18$ residual
that remains is the trajectory's **rotational displacement** -- the
part that a gradient-only integrator cannot reach from
$(\vec x_0, \vec v_0)$ regardless of $\gamma$, $\eta$, or the shape of
$V$.

The Helmholtz-augmented experiment of §1.4 refines this statement: the
residual is rotational in the sense of having non-zero curl, but it is
**not** well described as the action of a constant linear skew operator
on position. In the top-16 PCA subspace, the best constant skew operator
has $R^{2}\!\ll 0$ at the per-layer fit -- i.e. it cannot even explain
the *one-step* residual, let alone generate the integrated trajectory.
What the data support is the stronger claim:

> *The rotational component of the layer-wise force is non-linear in
> $x$ (or coupled to $\dot x$), rank-deficient and head-specific, and
> state-dependent in a way consistent with the LayerNorm + softmax
> nonlinearities in each transformer block.*

The simple Ohm's-law-style reading "torque = $\Omega\,x$" does not hold
for this data; any minimally adequate solenoidal ansatz must therefore
live in the function class of §6 (velocity-coupled vector potential with
position dependence) or the connection-based formulation of §14's
Jacobi-metric programme.

### 4.1 Why damping "wins" at $\gamma^*\approx 5$

The $\gamma$ sweep of §1.2 (raw data in
[`extended_gamma_first_order_results.npz`][r1], visualised in
[`fig_extended_gamma.png`][f1]) shows a monotone decrease from
$\gamma=0$ (trajectories wildly over-shoot because $\vec v_0$ launches
them in unphysical directions, median residual $\approx 1.44$) down to a
plateau at $\gamma^{*}\approx 5$ (median residual $0.1768$, i.e. the
static-null floor $0.1773$ to three decimal places). The interpretation
now is clean: **what $\gamma$ is doing is killing the initial velocity
component that would otherwise inject energy into the wrong rotational
modes.** Once the velocity is damped out the integrator predicts
$\vec x \approx \vec x_0$ -- the static null. It cannot do better because
the rotational motion that GPT-2 performs is not generated by $V$.

### 4.2 Why the first-order flow gives _exactly_ the static null

A pure first-order flow with a very weak $V$ moves the state by a
negligible amount per layer: the fitted $b_\ell \sim 10^{-5}$ and
typical $r^2 \sim 10^5$ at middle layers give a gradient
$2 a b\,x\,e^{-b r^2} \sim 6\times 10^{-5}\,x$, which even at $\eta=25$
over 12 layers accumulates to $\lesssim 0.02\|x\|$ -- well below the
scale of the observed inter-layer motion. Because there is no velocity
variable to carry rotational information, the first-order integrator
has no mechanism for anything but tiny radial displacement, and that
displacement is swamped by numerical precision. It _literally_ does
not move the hidden state, which is why it ties the static null to
floating-point precision
($\max_t|\rho^{(L)}_{t,\mathrm{1st}} - \rho^{(L)}_{t,\mathrm{static}}| = 0$)
across all $\eta \in \{0.01,\dots,25\}$ tried in
[`extended_gamma_first_order_results.npz`][r1].

### 4.3 Dimensional picture

The "work" that GPT-2's layer stack performs on a hidden state is
overwhelmingly **rotational work** -- in the elementary sense that
the path integral $\int \vec F \cdot d\vec x$ along the observed
trajectory, as decomposed by Helmholtz, is dominated by the
$\vec F_{\mathrm s}$ contribution. The conservative part
$\int -\nabla V \cdot d\vec x = V(\vec x_0) - V(\vec x_L)$ is small --
at most a few nats per sentence -- which is consistent with NTP loss
being the only scalar that $V$ tracks, and NTP loss changing by at
most $\mathcal{O}(1)$ between layers.

Quantitatively, the force-per-displacement $k(r) := V'(r)/r$ at a
typical per-sentence-centered radius $r\approx 300$ (middle layers),
computed from the fitted parameters in
[`well_form_comparison_params.json`][pj], is pinned in the narrow band

| form | $k(r=300)$ at layer 4 (units $r^{-2}$) |
|---|---:|
| harmonic | $1.51\times 10^{-6}$ |
| gaussian | $1.67\times 10^{-6}$ |
| morse | $1.61\times 10^{-6}$ |
| rational | $1.79\times 10^{-6}$ |
| log-saturation | $1.69\times 10^{-6}$ |
| Weibull | $3.07\times 10^{-6}$ |
| power | $0.66\times 10^{-6}$ |

across every functional form we tested (raw values backed by
[`well_form_integrator_results.npz`][ni] and derived in detail in
[`well_form_comparison_summary.md`][sum3] §4). This common $\sim 10^{-6}$
force scale is set by the data geometry --
well depth $a\approx 8$ nats divided by squared data scale $\|x\|^2\sim 10^{5}$ --
and is the reason all eight integrator configurations collapse onto
the static-null floor. It is not a property of $V$'s shape.

---

## 5. What this does and does not invalidate in the paper

### 5.1 Still intact

The paper's core claims _at local microscopic scales_ are untouched by
this argument:

- **Result 1 (STP-acceleration identity).** The local identity
  between the semantic-mass formulation of attention and the
  per-phrase tangential/normal acceleration decomposition holds
  _pointwise_ and does not require a global scalar potential. In the
  Helmholtz decomposition it is an algebraic identity on
  $\vec F$ itself; the part of $\vec F$ captured by it is
  path-independent by construction (a local derivative), but it does
  not claim path-independence of the _multi-layer integrated_
  trajectory.
- **Semantic mass formalism.** $\mathfrak{m}_t^{(\ell)} = w_t^{(\ell)}$
  (attention column sum) remains a valid per-layer scalar. Nothing
  in §3 depends on removing it.
- **Per-phrase attractive wells.** Per-phrase centering shrinks the
  relevant $\lVert x\rVert$ scale by an order of magnitude, and the
  binned-median $R^{2}\approx 0.90$ at middle layers (§1.3) shows the
  conservative part is _real_ at phrase-local resolution. Prediction
  P4 in §10.3 still has empirical content: it is testing the
  conservative sub-component of the full field.
- **Jacobi-metric / Riemannian programme (§14).** Geodesic motion in
  a curved space is _not_ intrinsically gradient-flow. A Riemannian
  connection can encode rotational dynamics (via Christoffel symbols
  and holonomy) without any scalar potential. The §14 programme is
  therefore **the correct next formalism**, not a failed alternative.

### 5.2 What must be softened in v2

- **"E-init should reproduce the trajectory."** This claim must be
  retired. Any language in §11.7 and §13.7 that presents E-init as a
  test the framework is expected to pass should be rewritten in the
  form "we tested whether the scalar-potential reading sufficed; it
  does not; the deficit is interpretable as the solenoidal / rotational
  component of the layer-wise flow".
- **"Gaussian vs. other functional forms of the well."** §14.6
  Outcome C should be narrowed: the functional-form discrimination
  has been attempted (Morse, rational, log-saturation, Weibull,
  power law, harmonic) and the class is empirically indistinguishable
  on integrator residual. The open question is _not_ which $V(r)$
  is "the" right well; it is whether and how to add a non-gradient
  term to the Lagrangian.
- **"Layer-as-time evolution of the residual stream is Lagrangian."**
  True only for the conservative component. The v2 framing should say
  that the residual stream decomposes, at the flow level, into a
  small gradient-flow part (captured by a scalar well) and a dominant
  solenoidal / rotational part (not captured by it).

---

## 6. A minimal extension: gauge / vector-potential term

The simplest Lagrangian extension that accommodates a non-zero curl
is the electromagnetic-analogue form:

$$L \;=\; \tfrac{1}{2}\mathfrak{m}\,\lVert\dot{\vec x}\rVert^{2}
  \;+\; \vec A(\vec x)\!\cdot\!\dot{\vec x}
  \;-\; V(\vec x),$$

where $\vec A$ is a vector potential. The Euler--Lagrange equation
becomes

$$\mathfrak{m}\,\ddot{\vec x} \;=\; -\nabla V(\vec x)
  \;+\; \bigl[\dot{\vec x}\!\times\!B(\vec x)\bigr]
  \;-\; \mathfrak{m}\gamma\,\dot{\vec x},$$

with $B = \nabla\times\vec A$ in 3D and with the obvious $d$-dimensional
antisymmetric generalisation $F_{ij} = \partial_i A_j - \partial_j A_i$
in higher $d$. The $B$-term is velocity-dependent and orthogonal to the
velocity: it does no work, but it **rotates** the velocity, producing
exactly the semantic-rotation phenomenon that attention imposes.

In this formulation:

- The scalar $V$ captures per-token NTP-loss-coupled attraction --
  the tiny conservative component our experiments saw.
- The vector $\vec A$ captures the attention-driven non-integrable
  part. Its curl $F_{ij}$ is the "semantic magnetic field".
- Multi-head attention maps naturally onto **multiple** such
  antisymmetric 2-forms $F^{(h)}_{ij}$, one per head, which sum in
  the equation of motion.

This is not proposed as the v2 framework -- it is a sketch of the
kind of Lagrangian that would be needed. It is presented here for
three reasons:

1. To show that the failure of the conservative reading is
   _expected_ rather than mysterious from a classical-mechanics
   standpoint.
2. To indicate that the Jacobi-metric programme of §14 is the
   right level of generality: a Riemannian connection with
   non-zero torsion or an affine connection encoding $F_{ij}$
   produces the same rotational phenomena without committing to a
   specific electromagnetic analogy.
3. To clarify that the §1.4 negative result, while empirically closing
   the *simplest* Helmholtz augmentation ($\vec F_{\mathrm s}=\Omega\,x$
   with constant $\Omega$), does **not** close the electromagnetic
   analogue: the force $\dot{\vec x}\times B(\vec x)$ is *velocity*-
   dependent and $B(\vec x)$ is *position*-dependent, so the Lagrangian
   (6.1) is strictly richer than the ansatz tested in §1.4. Section
   §1.5 has now closed the next natural step of this programme --
   constant skew $B\,\dot x$ and affine-in-$x$ skew $B(x)\,\dot x$ with
   rank 1 and rank 2 position dependence, both alone and in combination
   with $\Omega x$ -- and the outcome is uniformly negative on held-out
   data. The path forward, if one wishes to stay inside a gauge
   Lagrangian, is either (a) fully non-polynomial position dependence
   of $B(x)$ via a rich basis of position features (RBFs, Fourier on
   PCA coordinates), or (b) a non-abelian multi-head decomposition
   $F(x) = \sum_h F^{(h)}(x)$. Both are substantial research
   undertakings.

   A cleaner path is to drop the gauge ansatz entirely and move to the
   Riemannian / Jacobi-geodesic formulation of §14. In that language
   the Christoffel symbols of a generic metric tensor encode both
   antisymmetric *and* non-Hessian symmetric parts of the effective
   local flow, directly accommodating the fit-quality signal from §1.5
   (PCA-ev full $\approx 0.8$) without forcing it into either a scalar
   potential or a skew gauge field.

---

## 7. Testable predictions that follow

If the hypothesis of §4 is correct -- that the residual the
conservative models cannot close is semantic rotation -- several
concrete predictions follow that v2 experiments can pursue:

- **P-rot-1 (non-zero curl, position-dependent).** Estimate the
  discrete curl of the layer-wise displacement field
  $\Delta h^{(\ell)} = h^{(\ell+1)} - h^{(\ell)}$ on a local chart
  fitted by PCA at each layer. §1.4 already shows the *global linear*
  curl in the top-16 PCA subspace is not what drives the residual --
  the curl must therefore be **position-dependent**: the antisymmetric
  part of the Jacobian $\partial_i \Delta h_j - \partial_j \Delta h_i$
  should vary substantially with $x$ (and with neighbouring tokens),
  and in particular should grow with attention-head rank-deficiency.
  The right diagnostic is *local* curl (estimated from small
  neighbourhoods on the token/layer manifold), not global linear curl.
- **P-rot-2 (loop-integral asymmetry).** Pick two semantically
  related phrases (e.g. "the cat sat" and "sat the cat did"); compare
  the integrated $\int \Delta h \cdot d\vec x$ along the two paths.
  For a conservative field the difference would vanish; for
  transformers we predict a measurable, direction-dependent gap.
- **P-rot-3 (head ablation removes rotation).** Ablating individual
  attention heads should reduce the solenoidal component of the
  effective field in the subspace spanned by that head's $V^{(\ell,h)}$
  projection, without strongly affecting the conservative component.
- **P-rot-4 (per-phrase wells recover the conservative part).**
  Prediction P4 of §10.3, restricted to _within-phrase_ motion,
  should succeed where the whole-sentence well fails: the solenoidal
  contribution is chiefly inter-phrase and near-phrase-boundary, not
  within a single phrase.
- **P-rot-5 (FFN-only integrator).** An integrator that uses only
  the FFN per-layer update (stripping attention) should behave more
  like a gradient flow, and the scalar-potential reading should work
  better on that sub-system. This is a clean mechanistic
  decomposition test.
- **~~P-rot-6 (velocity-coupled gauge).~~** _Tested and closed in §1.5._
  The electromagnetic-analogue Lagrangian at its constant and
  affine-in-$x$ linear level does not beat the static null on held-out
  data (see §1.5 and
  [`velocity_coupled_gauge_summary.md`][sum5]). The next falsifiable
  form of this prediction is **P-rot-6' (non-polynomial $B(x)$)**:
  parameterise $B(x)$ via a basis of smooth position features
  (RBFs centred on training-data clusters, or Fourier modes on the top
  few PCA coordinates), and repeat the §1.5 fitting / stability /
  shrinkage protocol. A positive result here would be the first
  Lagrangian-family model that meaningfully beats the static null on
  held-out sentences.
- **P-rot-7 (per-layer Jacobian symmetry test).** Directly compare,
  layer-by-layer, the symmetric vs. skew spectra of the fitted
  $M_\ell$ in [`helmholtz_curl_results.npz`][hm1]. §1.5 now *confirms*
  this prediction in a preliminary way (PCA-ev skew $\approx -10$ to
  $-180$; PCA-ev full $\approx 0.5$--$0.8$), but the quantitative
  question -- what fraction of the symmetric part of $M_\ell$ is
  Hessian of a scalar (Helmholtz-conservative-linear) vs. non-Hessian
  (neither conservative nor solenoidal) -- has not been split out yet.
  The split uses $\operatorname{sym}(M) = \operatorname{sym}(M)_{\text{Hess}} + \operatorname{sym}(M)_{\text{non-Hess}}$ via the constraint
  that the Hessian part must be a second derivative of some scalar
  field; per layer this is a small linear-algebra exercise and would
  turn the §1.5 fit-quality signal into a positive empirical
  characterisation of what the transformer block is doing.
- **P-rot-8 (Riemannian-connection fit).** On the same 40/10 split,
  fit a local Riemannian metric $g_{ij}(x)$ per layer (e.g. by
  parameterising $g$ as low-rank perturbation of the Euclidean metric
  in PCA coordinates) and run the geodesic equation
  $\ddot x^i + \Gamma^i_{jk}\,\dot x^j\,\dot x^k = 0$ as integrator.
  This is the §14 Jacobi programme made operational. Unlike §1.5 the
  Christoffel term $\Gamma^i_{jk}\dot x^j\dot x^k$ is
  quadratic in $\dot x$ and naturally encodes both rotational and
  stretching contributions. A positive result here is the strongest
  evidence for the Riemannian reading of the paper.

None of these predictions is cheap, but none is more expensive than
the experiments already conducted, and they can be staged for v2.

---

## 8. What changes in the paper

A minimal v2 rewrite implied by this document:

1. **Abstract / §1 Introduction.** Downgrade any sentence claiming
   the framework predicts global trajectories from a scalar
   potential. Replace with "a scalar potential captures the
   conservative component of the layer-wise flow; the remaining
   solenoidal component is the subject of the Riemannian / gauge
   extension in §14".
2. **§10.3 P4 (Per-phrase attractive well).** Keep as-is -- it is
   already softened to "bounded attractive well", and its natural
   interpretation in light of §4 is as a **conservative-part
   discriminator**.
3. **§11.7 and §13 (E-init).** Update with the extended-$\gamma$ and
   functional-form results, and with the Helmholtz decomposition
   reading. The E-init negative result becomes a structural finding
   rather than a numerical embarrassment.
4. **§14.6 Outcome C.** Cross-reference this document. Narrow Outcome
   C from "the Gaussian form is wrong; try Morse / harmonic / LJ" to
   "the conservative functional form is approximately harmonic with
   $\sim 10^{-6}$-strength force-per-displacement; the unexplained
   residual is rotational motion generated by attention, which
   requires adding a non-gradient term to the Lagrangian (sketch in
   §6 of this note) or a connection-based formulation in the Jacobi
   programme."
5. **New §14.X "Semantic rotation".** Introduce the Helmholtz-
   decomposition picture and the P-rot predictions of §7 above. This
   becomes the positive reframing of the E-init work.

---

## 9. Artefact index

A complete listing of every file referenced by this document, grouped
by experiment.  All paths are relative to the repository root.

### 9.1 v1 E-init with Gaussian well (§1.1)

Source:
- [`notebooks/e_init/e_init_validation.ipynb`][nb1]
  &nbsp;--- full extraction + well fit + integrator pipeline.

Fitted parameters:
- [`notebooks/e_init/results/well_params.json`][wpg]
  &nbsp;--- global centering.
- [`notebooks/e_init/results/well_params_ps.json`][wpps]
  &nbsp;--- per-sentence centering (used in this document).

Raw data:
- [`notebooks/e_init/results/e_init_results.npz`][nvg]
  &nbsp;--- global-centering residuals and $\gamma$ sweep.
- [`notebooks/e_init/results/e_init_results_ps.npz`][nv1]
  &nbsp;--- per-sentence-centering residuals (used in this document).

Figures:
- [`fig_A_residual_vs_layer.png`][fag] / [`fig_A_residual_vs_layer_ps.png`][fa1]
  &nbsp;--- median residual per layer, global / per-sentence.
- [`fig_B_residual_vs_logw.png`][fbg] / [`fig_B_residual_vs_logw_ps.png`][fb1]
  &nbsp;--- residual vs. log attention mass.
- [`stage4_well_r2_vs_layer.png`][s4] / [`stage10_well_r2_vs_layer_ps.png`][s10]
  &nbsp;--- per-layer $R^2$ of the Gaussian well fit.

### 9.2 Extended $\gamma$ sweep and pure first-order (§1.2)

Source:
- [`notebooks/e_init/extended_gamma_and_first_order.py`][s1]
  &nbsp;--- self-contained; re-extracts GPT-2, refits wells, runs both
  sweeps (~1 min on MPS).

Raw data:
- [`notebooks/e_init/results/extended_gamma_first_order_results.npz`][r1]
  &nbsp;--- $\gamma$ and $\eta$ sweeps, per-token layer-$L$ residuals,
  per-layer medians.

Figures:
- [`fig_extended_gamma.png`][f1]
  &nbsp;--- median residual vs. $\gamma$ (12-point sweep) with static-null
  reference.
- [`fig_first_order_eta.png`][f2]
  &nbsp;--- median residual vs. $\eta$ (10-point sweep) with static-null
  and best-2nd-order reference lines.

Report:
- [`notebooks/e_init/results/extended_gamma_first_order_summary.md`][sum2]
  &nbsp;--- tables, per-layer breakdown, interpretation.

### 9.3 Functional-form comparison (§1.3)

Source:
- [`notebooks/e_init/well_functional_form_comparison.py`][s2]
  &nbsp;--- self-contained; fits seven forms per layer, reruns the
  damped 2nd-order integrator under each (~2 min on MPS).

Raw data and fit parameters:
- [`notebooks/e_init/results/well_form_comparison_r2.csv`][rc]
  &nbsp;--- $R^2$ and AIC per form per layer (machine readable).
- [`notebooks/e_init/results/well_form_comparison_params.json`][pj]
  &nbsp;--- all fitted parameters (form $\times$ layer).
- [`notebooks/e_init/results/well_form_integrator_results.npz`][ni]
  &nbsp;--- integrator residuals per form per $\gamma$.

Figures:
- [`fig_well_form_r2_vs_layer.png`][fwr]
  &nbsp;--- raw $R^2$ per layer, one curve per form.
- [`fig_well_form_r2_vs_layer_binned.png`][fwb]
  &nbsp;--- binned-median $R^2$ (the *deterministic-trend* plot).
- [`fig_well_form_scatter_layer3.png`][fs3],
  [`fig_well_form_scatter_layer6.png`][fs6],
  [`fig_well_form_scatter_layer9.png`][fs9]
  &nbsp;--- pooled scatter with all seven fits overlaid.

Report:
- [`notebooks/e_init/results/well_form_comparison_summary.md`][sum3]
  &nbsp;--- tables, force-scale derivation, interpretation.

### 9.4 Helmholtz-augmented integrator (§1.4)

Source:
- [`notebooks/e_init/helmholtz_curl_augmented.py`][s3]
  &nbsp;--- self-contained; 40/10 train/test split, per-layer PCA
  ($k=16$), weighted OLS fit of skew $\Omega_\ell$ and full $M_\ell$,
  augmented symplectic Euler, $\gamma$ sweep (~45 s on MPS).

Raw data:
- [`notebooks/e_init/results/helmholtz_curl_results.npz`][hm1]
  &nbsp;--- $\gamma$ sweep, median residual per configuration / fold,
  mean fit $R^2$ of skew and full operator, per-layer TEST median
  residual arrays.

Figures:
- [`fig_helmholtz_residual_vs_gamma.png`][hmf1]
  &nbsp;--- median residual vs. $\gamma$, TRAIN and TEST panels, with
  the three integrator configurations and the static-null reference.
- [`fig_helmholtz_residual_vs_layer_at_gamma_star.png`][hmf2]
  &nbsp;--- per-layer TEST residual at $\gamma^{*}=5$.

Report:
- [`notebooks/e_init/results/helmholtz_curl_summary.md`][sum4]
  &nbsp;--- full tables, PCA fit-quality columns, detailed
  interpretation.

### 9.5 Velocity-coupled gauge augmentation (§1.5)

Source:
- [`notebooks/e_init/velocity_coupled_gauge.py`][s4]
  &nbsp;--- self-contained; same 40/10 train/test split and top-16
  PCA basis as §1.4; six configurations (`gaussian`, `omega_x`,
  `B_const`, `B_affine_r1`, `B_affine_r2`, `omega_and_Bconst`) crossed
  with five $\gamma$ values and an eight-point shrinkage grid
  $s\in\{0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1\}$; per-layer skew /
  unconstrained OLS with ridge $10^{-3}$.
  Runtime ~6 min on MPS (because of the scale sweep).

Raw data:
- [`notebooks/e_init/results/velocity_coupled_gauge_results.npz`][vc1]
  &nbsp;--- full $\gamma$ sweep, per-config per-scale TRAIN/TEST median
  layer-$L$ residuals, per-layer TEST medians at $s=1$ and at the
  TRAIN-optimal $s^{*}$, and per-layer fit-quality $R^{2}$ for both
  skew-projected and unconstrained operators.

Figures:
- [`fig_gauge_residual_vs_gamma.png`][vcf1]
  &nbsp;--- TRAIN / TEST median residual vs $\gamma$, log-scale
  y-axis, shrinkage-tuned operators, six-config comparison with
  static-null reference.
- [`fig_gauge_residual_vs_layer_at_gamma_star.png`][vcf2]
  &nbsp;--- per-layer TEST residual at per-config $\gamma^{*}$ with the
  TRAIN-optimal shrinkage applied.

Report:
- [`notebooks/e_init/results/velocity_coupled_gauge_summary.md`][sum5]
  &nbsp;--- all tables (full-strength and shrunk variants),
  per-config best-$\gamma$ summary, structural interpretation,
  "Five negative experiments" bottom line.

### 9.6 Related conceptual documents

- [`docs/E_init_execution_plan.md`][plan] §18 --- "Would a non-Gaussian
  well profile flip the E-init result?"
- `companion_notes/On_the_Interpretation_of_Semantic_Mass.md` --- the
  semantic-mass formalism referenced in §5.1.
- `companion_notes/On_The_Existence_of_Acceleration_in_Semantic_Structures.md`
  --- the STP-acceleration derivation referenced in §5.1.

[wpg]: ../notebooks/e_init/results/well_params.json
[nvg]: ../notebooks/e_init/results/e_init_results.npz
[fag]: ../notebooks/e_init/results/fig_A_residual_vs_layer.png
[fbg]: ../notebooks/e_init/results/fig_B_residual_vs_logw.png
[s4]: ../notebooks/e_init/results/stage4_well_r2_vs_layer.png
[plan]: ./E_init_execution_plan.md
