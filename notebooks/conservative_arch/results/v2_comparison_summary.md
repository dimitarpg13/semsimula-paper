# v2 positive-control vs. negative-control comparison

**Status:** steps 1--4 deliverable.  Shakespeare convergence run,
velocity-aware Jacobian symmetry test, strict shared-potential test,
matched-parameter baseline, $V_\psi$ capacity sweep, SPLM oracle, and
token-direction coordinate-system robustness check are all in.

**Headline separator (step 2 -- shared-potential fit, layer direction).**
A single 2-layer scalar MLP $V_\psi(h)$ (hidden 256) whose gradient
reproduces per-layer $\Delta x_\ell$ across all layers and held-out
sentences yields a **median per-layer TEST $R^2$ of +0.90 on SPLM vs.
+0.45 on GPT-2**, and — more tellingly — **mean R² over GPT-2's middle
layers 6-10 is +0.09** while SPLM has no such collapse (middle-layers
mean $R^2$ = +0.86).  On 6 of 7 SPLM layers, the fit exceeds
$R^2 = 0.5$; on 5 of 11 GPT-2 middle layers, it fails completely
(gain over velocity-only baseline $\leq 0.05$).

**Step-4 robustness check: token-direction replication.**  Running the
same diagnostic with **sequence-time** as the integration axis (fixed
layer, vary token position) reproduces the separator and sharpens it:
median gain over velocity-only is **+0.27 for SPLM, $-$0.08 for
Matched GPT-style, +0.03 for Pretrained GPT-2**.  Only SPLM's
held-out token-direction forces derive from any global scalar; the
attention baselines' learned forces don't transfer from train to test
in that coordinate.  The separator is robust to the choice of time
axis.  Full write-up: [`sharedV_comparison_summary.md`](sharedV_comparison_summary.md)
and [`token_direction_summary.md`](token_direction_summary.md).
Figure:

![sharedV](sharedV_splm_vs_gpt2.png)

The rest of this document walks through the supporting diagnostics
and motivates the step-2 test.

This document records the side-by-side output of our shared trajectory
diagnostics applied to two architectures on the same 50-sentence test
corpus:

- **Positive control (PC)**: the scalar-potential conservative-by-construction
  language model (`ScalarPotentialLM`, $d=128, L=8$, 7.1 M params),
  trained on Tiny Shakespeare to val ppl 287.
- **Negative control (NC)**: GPT-2 small (124 M params, pretrained,
  $d=768, L=12$), evaluated on the identical corpus.

## 1. Training convergence of the positive control

The scalar-potential LM in `shakespeare` mode converges cleanly in
~33 min on MPS:

- Initial uniform-distribution loss: $\log 50257 \approx 10.82$
- Final training loss: 5.07 (eval), val loss 5.66, **val ppl 287**
- Learned $m = 0.980$, $\gamma = 0.961$
- Artefacts: [`splm_shakespeare_summary.md`](splm_shakespeare_summary.md),
  [`splm_shakespeare_loss_curve.png`](splm_shakespeare_loss_curve.png)

## 2. Diagnostic 1: Gaussian-well E-init (§1.1 ansatz)

**Fits $V_\ell(r) = a_\ell(1 - e^{-b_\ell r^2})$ per layer to (radial distance,
PTL) pairs, then integrates with symplectic Euler + damping.  Reports
median layer-L relative residual vs. static null.**

| quantity | PC (SPLM, shakespeare) | NC (GPT-2, §1.5 ref) |
|---|--:|--:|
| static-null TEST residual | 0.437 | 0.180 |
| best Gaussian-well TEST | 0.438 (γ=0.25) | 0.177 (γ=5.0) |
| $\Delta$ vs. null | $-0.000$ (null) | $-0.003$ (null) |
| per-layer well $R^2$ on TRAIN | 0.000–0.010 | similar |

**Verdict:** The radial-Gaussian ansatz *fails on both architectures*
alike.  This is a limitation of the ansatz (one 2-parameter radial
well per layer is too restrictive to capture the learned potential),
not a specific signal of non-conservativeness in either model.  In
particular, the smoke-run "positive result" on the 300-step SPLM
was partly an under-training artefact: tiny trajectory motion made
a weak well look good.

## 3. Diagnostic 2: Jacobian-symmetry test

**Fits the per-step linear operator $y = x_{\ell+1}-x_\ell$ on a
shared PCA-16 subspace, both unconstrained and with $M_\ell = M_\ell^\top$.
A small full-vs.-symmetric gap means the per-step dynamics is
consistent with a (local) scalar-potential flow.**

### 3.1 Position-only (§1.5 analogue): $y \approx M_\ell x_\ell$

TEST $R^2$ (unconstrained / symmetric-restricted), max-gap across layers:

| architecture | layer range of full $R^2$ | layer range of sym $R^2$ | max gap |
|---|:--:|:--:|--:|
| PC (SPLM) | [0.74, 0.97] | [0.18, 0.78] | 0.69 (layer 3) |
| NC (GPT-2) | [0.45, 1.00] | [0.36, 1.00] | 0.18 (layer 10) |

**Observation:** Surprisingly, *GPT-2 looks as good or better* than
the conservative-by-construction model on this test.  This is the
same symmetric-operator pattern §1.5 of the Failure doc already
identified ("the per-layer one-step linear approximation of the
transformer block is symmetric and non-Hessian").  What §1.5 called
"symmetric and non-Hessian" our re-analysis makes concrete:

- **Symmetric**: $R^2_\mathrm{sym} \approx R^2_\mathrm{full}$ layer-by-layer.
- **Non-Hessian**: the symmetric $M_\ell$ changes with $\ell$ (and with
  input distribution) in a way that no single radial $V$ can reproduce
  -- hence the failure of §1.1--1.3 and §2 above.

### 3.2 Velocity-aware (the clean test): $y \approx A v_\ell + M_\ell x_\ell$

With $v_\ell := x_\ell - x_{\ell-1}$, the linear model now matches the
analytic form of the damped second-order integrator exactly; only
the residual symmetry of $M_\ell$ remains meaningful.

TEST $R^2$ (unconstrained / symmetric-restricted), max-gap across layers:

| architecture | layer range of full $R^2$ | layer range of sym $R^2$ | max gap |
|---|:--:|:--:|--:|
| PC (SPLM) | [0.82, 0.98] | [0.78, 0.98] | **0.040** (layer 4) |
| NC (GPT-2) | [0.45, 1.00] | [0.43, 1.00] | **0.079** (layer 10) |

**Observation:** Both architectures pass the local-symmetry test
cleanly -- full and symmetric-restricted fits track each other to
within $\leq 0.08$ across every layer.  Figures:

- PC: [`splm_shakespeare_ckpt_latest_fig_jacsym.png`](splm_shakespeare_ckpt_latest_fig_jacsym.png)
- NC: [`splm_gpt2_baseline_fig_jacsym.png`](splm_gpt2_baseline_fig_jacsym.png)

## 4. Reconciliation with the Failure doc narrative

The Failure doc §1.5 conclusion ("GPT-2 has **symmetric, non-Hessian**
per-step linear operators") is **confirmed** by our re-analysis.  What
requires refinement is the *interpretation*:

- What the local Jacobian test rules out for GPT-2 is **the presence
  of a non-negligible skew-symmetric (curl / solenoidal) force
  component**.  Both §1.5's "Helmholtz curl does not help" and our
  new finding that $R^2_\mathrm{sym} \approx R^2_\mathrm{full}$ say
  this clearly.
- What the local Jacobian test does **not** rule out is the existence
  of *some* scalar potential $V$ whose Hessian matches the per-layer
  $M_\ell$.  The *layer-varying* symmetric $M_\ell$ is **compatible**
  with such a $V$ -- what fails is the restriction to a *shared,
  simple, radial* form of $V$ across all layers.

This sharpens the v2 narrative:

- **Where the separation lives is not at the local linear level.** At
  that level both architectures exhibit symmetric per-step Jacobians.
- **Where the separation lives is at the global shared-potential level.**
  For PC, all per-layer $M_\ell$ derive from a single learned scalar
  $V_\theta(\xi, h)$ *by construction*.  For NC, no single $V$ from a
  tractable family has yet been found that reproduces the per-layer
  operators.

## 5. Step 2: the strict shared-potential test

The sharp question left open by §3-§4: does *some* smooth scalar
$V$ reproduce all per-layer operators?  We answered it by joint
optimisation of a **single** neural scalar $V_\psi$ plus per-layer
scalars $\alpha_\ell,\beta_\ell$ minimising

$$\left\lVert \Delta x_\ell \;-\; \alpha_\ell v_\ell \;+\; \beta_\ell\, \nabla_h V_\psi(x_\ell) \right\rVert^2$$

pooled across all layers and all training sentences, with per-layer
TEST $R^2$ evaluated on held-out sentences.  $V_\psi$ is a 2-layer MLP
with hidden 256 and GELU activations; the same architecture is used
for both PC and NC.  Details and full tables in
[`sharedV_comparison_summary.md`](sharedV_comparison_summary.md).

### 5.1 Per-layer TEST $R^2$ with shared $V_\psi$ (the step-2 headline)

| model | layers | median | min | max | layers with $R^2 \ge 0.5$ |
|---|--:|--:|--:|--:|--:|
| PC (SPLM, shakespeare) | 7  | **+0.90** | +0.28 | +0.97 | **6 / 7**   |
| NC (GPT-2 small)        | 11 | **+0.19** | +0.04 | +0.99 | **4 / 11**  |

### 5.2 Per-layer gain over velocity-only baseline (shared-$V_\psi$ - vel-only)

| model | median | max | layers with gain $\ge 0.3$ |
|---|--:|--:|--:|
| PC (SPLM)  | **+0.46** | +0.85 | **6 / 7**  |
| NC (GPT-2) | **+0.08** | +0.98 | **3 / 11** |

### 5.3 Why this is the real separator

The velocity-aware Jacobian test of §3.2 allowed an *independently chosen
symmetric $M_\ell$ per layer* -- effectively no structural constraint at
all beyond local symmetry.  The shared-$V_\psi$ test forces all per-layer
Jacobians to arise as slices of **one** Hessian field:
$M_\ell \approx -\beta_\ell \nabla^2 V_\psi(h_\ell)$.

GPT-2 passes the §3.2 test but fails §5: each of its 11 layers admits a
locally-symmetric $M_\ell$, but those $M_\ell$'s cannot be jointly
described by the Hessian of any single smooth scalar.  The conservative-
by-construction SPLM passes both, as expected.

GPT-2's shared-$V_\psi$ fit succeeds only at its *boundary layers*
(embedding-to-semantic and pre-logit transitions, layers 1-4 and 11)
and collapses on the middle of the stack (layers 5-10, median
$R^2 \le 0.2$).  This matches the intuition that attention + MLP
composition in the bulk of the network is performing *heterogeneous*
transformations that cannot be collapsed into a single semantic energy
landscape.

## 6. Remaining follow-ups (step 3 candidates)

1. **Matched-parameter GPT-2-style baseline** (7 M transformer on the
   same Tiny Shakespeare data and compute budget as SPLM).  Re-run
   the step-1 and step-2 diagnostics.  Isolates architecture from
   parameter count / pretraining-scale effects.
2. **Capacity sweep of $V_\psi$** on GPT-2 (hidden in
   $\{256, 512, 1024, 2048\}$, depth up to 4).  Does the middle-layer
   failure eventually disappear, or is it structural?  Under what
   capacity does GPT-2 reach SPLM's median?
3. **Oracle reference for SPLM** -- use $V_\theta(\xi, h)$ as the
   shared potential; gives the achievable upper bound for the
   $V_\psi(h)$-only ansatz and explains the weak layer-4 residual
   (likely due to dropped $\xi$ dependence).
4. **Scale sweep on PC** ($d \in \{64, 128, 192\}$, $L \in \{4, 8, 12\}$)
   to confirm step-2 robustness.

## 7. Provisional v2 summary sentence

"The per-step hidden-state dynamics of both attention transformers and
scalar-potential LMs admit a symmetric linear approximation at the PCA-16
level (per-layer Jacobian test; max TEST gap 0.04 for SPLM vs. 0.07–0.08
for GPT-2 and a matched-parameter GPT-2-style baseline).  But when the
stronger structural constraint is imposed -- that all per-layer operators
derive from **one** shared scalar potential -- a small 2-layer MLP $V_\psi$
reproduces the SPLM dynamics at median TEST $R^2 = 0.90$ across every layer,
while a scale- and data-matched attention baseline drops to $R^2 = 0.56$
(monotonic decay 0.81 → 0.27) and pretrained GPT-2 small drops to $R^2 = 0.45$
with a bathtub pattern whose middle layers 6–10 mean $R^2 = 0.09$.  Capacity
of $V_\psi$ is not the bottleneck: a 6-config sweep (up to 1.8 M V_ψ params)
recovers the identical middle-layer R² values to 3 decimals.  The
conservative-by-construction architecture is thus empirically distinguishable
from attention baselines at exactly the structural level the Semantic
Simulation framework prescribes: a shared, tractable semantic energy
landscape."
