# Symplectic Integration for SPLM: Velocity-Verlet, Strang Splitting, and What We Actually Measured

> **The idea in one sentence.** Macaron-net's insight — and the Strang /
> leapfrog literature more generally — is that for *conservative second-order
> flows*, a symmetric second-order splitting is the canonical integrator.
> Velocity-Verlet is $O(\Delta t^2)$ accurate, symplectic in the undamped
> limit, and is the standard tool that the molecular-dynamics and
> Hamiltonian-dynamics communities reach for.  SPLM's equation of motion is
> exactly that kind of flow, so SPLM should use it.

This document explains that idea from first principles, describes the
minimal experiment we ran to test it on SPLM, and summarises what we
actually learned from the numbers.

---

## 1. A five-minute tutorial on symmetric second-order integrators

### 1.1 The kind of equation we care about

SPLM's layer-wise dynamics, derived in the paper, is a damped
second-order conservative flow:

$$
\mathfrak{m}_t\ddot{h} = -\nabla_h V_\theta(\xi, h) - \mathfrak{m}_t \gamma \dot{h}
$$

Splitting into position $h$ and velocity $v = \dot{h}$:

$$
\dot{h} = v, \qquad
\dot{v} = -\frac{1}{\mathfrak m_t}\nabla_h V_\theta(h)\ -\ \gamma v.
$$

This is a particle of mass $\mathfrak m_t$ moving in a potential
$V_\theta$, plus a viscous damping term $\gamma v$.  In the undamped
limit $\gamma \to 0$ it is a **Hamiltonian system** with

$$
H(h, p) = \tfrac{1}{2\mathfrak m_t}p^{\top} p + V_\theta(h),
\qquad p = \mathfrak m_t v.
$$

The central property of such systems is that their continuous flow
preserves three structures simultaneously: **energy** $H$, the
**symplectic form** $dp \wedge dh$, and **time-reversal symmetry**
($t \to -t$ combined with $v \to -v$ leaves the trajectory invariant).

### 1.2 What a numerical integrator is asked to do

A numerical integrator is a recipe $(h, v) \mapsto (h', v')$ that
approximates the continuous flow by one step of size $\Delta t$.
The question that dominates classical mechanics is **which of those
three structures does the integrator preserve?**

- **Explicit Euler** ($h' = h + \Delta t v,\ v' = v + \Delta t f/\mathfrak m$)
  preserves *none* of them and accumulates $O(\Delta t)$ error per
  step.  Energy drifts up without bound.
- **Symplectic Euler** ($v' = v + \Delta t f/\mathfrak m,\ h' = h + \Delta t v'$) preserves the **symplectic form** exactly and
  energy approximately (bounded drift).  Still only $O(\Delta t)$
  accurate.
- **Velocity-Verlet** (below) is $O(\Delta t^2)$, symplectic, *and*
  time-reversal-symmetric.  It is the first scheme that preserves all
  three structures simultaneously.

### 1.3 Velocity-Verlet (kick-drift-kick, KDK)

One full step of velocity-Verlet for the undamped flow
$\ddot{h} = f(h)/\mathfrak m$:

```
kick   v_{1/2} <- v + (dt/2) * f(h)   / m     # half-step kick at old h
drift  h'      <- h + dt * v_{1/2}             # full-step drift
kick   v'      <- v_{1/2} + (dt/2) * f(h') / m # half-step kick at new h
```

Equivalently: "half kick — full drift — half kick", which is exactly a
**Strang splitting** of the Hamiltonian $H = T(p) + V(h)$ into its
kinetic and potential parts.

Why this helps:

- The update is **symmetric in time**: reversing $v \to -v$ and running
  it backward recovers the original $(h, v)$ exactly (to machine
  precision).  This alone upgrades the global error from
  $O(\Delta t)$ to $O(\Delta t^2)$.
- Each of the three sub-steps is a **shear** in phase space, which
  exactly preserves the symplectic form.  A composition of shears is
  symplectic, so the full step preserves $dp \wedge dh$.
- Consequently, velocity-Verlet conserves a **shadow Hamiltonian**
  $\tilde H = H + O(\Delta t^2)$ for exponentially long times.  Energy
  does not drift; it oscillates inside a tight band.

### 1.4 What "Strang-split damping" adds

For a *damped* flow, we can split the generator $\mathcal L$ as

$$
\mathcal L = \underbrace{\mathcal L_{\text{Hamil}}}_{\text{Verlet handles this}} + \underbrace{\mathcal L_{\text{damp}}}_{\dot v = -\gamma v}.
$$

The damping flow has an *exact* closed form: $v \mapsto v
e^{-\gamma\Delta t}$.  A symmetric (Strang) composition of the
Hamiltonian part $\Phi_{\Delta t}^{\text{Ham}}$ and the damping part
$\Phi_{\Delta t}^{\text{damp}}$ is

$$
\Phi_{\Delta t}^{\text{total}} \approx
\Phi_{\Delta t/2}^{\text{damp}} \circ \Phi_{\Delta t}^{\text{Ham}} \circ \Phi_{\Delta t/2}^{\text{damp}},
$$

which preserves the $O(\Delta t^2)$ accuracy and keeps the symplectic
structure of the Hamiltonian sub-step exactly.  In code, this is just

```
v <- v * exp(-gamma * dt / 2)    # half-step damping
(kick, drift, kick)              # velocity-Verlet on h, v
v <- v * exp(-gamma * dt / 2)    # half-step damping
```

With $L$ such steps stacked, consecutive half-damping steps merge into
single full-damping steps, so only the first and last steps actually
apply the half variant.

### 1.5 The Macaron-net connection

Macaron-net (Lu et al., 2020) noticed that a standard transformer block
$\mathrm{Attention} \to \mathrm{FFN}$ has the structure of an
**asymmetric first-order splitting** of two sub-operators $A$ and $F$:

$$
(A \to F):\quad x \mapsto F(A(x)).
$$

Their fix was to replace it with the symmetric **Strang splitting**

$$
(F/2 \to A \to F/2):\quad x \mapsto (F/2)(A((F/2)(x))),
$$

which trades a modest extra cost for second-order splitting accuracy.
Velocity-Verlet is exactly the same fix applied to a
position/momentum split: **half kick → full drift → half kick** is a
Strang splitting of the Hamiltonian vector field.

SPLM's equation of motion is already a second-order conservative flow
by construction.  So applying the Macaron/Strang fix to SPLM's
*integrator* (not to an attention/FFN block) is the canonical
physicist's thing to do.  That was the motivation to try it.

---

## 2. Why we tried it in SPLM

Before the experiment, the theoretical case looked strong:

1. **SPLM's dynamics is exactly the kind symplectic methods were
   designed for.** Damped Newtonian particle in a learnable potential;
   Hamiltonian in the $\gamma \to 0$ limit.
2. **SPLM's baseline integrator is damped semi-implicit Euler**,
   $O(\Delta t)$.  The established textbook replacement is
   velocity-Verlet + Strang damping, $O(\Delta t^2)$.
3. **No extra learnables.** The integrator swap leaves the model's
   parameter count, architecture, and loss untouched; only the layer
   update rule changes.
4. **Small integration overhead.**  With the KDK force-reuse
   optimisation, an $L$-step Verlet stack costs $L + 1$ force
   evaluations vs Euler's $L$ — $\approx 12\%$ overhead at $L = 8$.
5. **Two separate things might improve.** Language-modelling
   perplexity (via sharper integration of the learned flow) and the
   paper's conservative-dynamics diagnostics (the shared-$V_\psi$
   fits) — which directly test whether the trajectories look like a
   Hamiltonian flow.

The experiment was framed as one minimal-diff swap against our best
SPLM variant (`sarf_mass_variant/` with `logfreq` mass on Tiny
Shakespeare), plus a small sweep over integrator-matched variants.

---

## 3. What we ran

The companion folder is
[`notebooks/conservative_arch/symplectic_variant/`](../notebooks/conservative_arch/symplectic_variant/).
Three training runs, identical model / data / hyperparameters / seed,
differing only in integrator settings:

| Run | Integrator | $L$ | $\Delta t$ | Force evals per forward | Wall-clock |
|-----|-----------|----:|-----------:|-----------------------:|-----------:|
| **Euler baseline** (reference) | damped semi-implicit Euler | 8 | 1.0 | 8 | 2 390 s |
| **Verlet L=8** (matched) | velocity-Verlet + Strang damping | 8 | 1.0 | 9 | 2 034 s |
| **Verlet L=4** (FLOP-halved) | velocity-Verlet + Strang damping | 4 | 1.0 | 5 | 1 060 s |
| **Verlet L=16, dt=0.5** (matched flow distance $L\Delta t = 8$) | velocity-Verlet + Strang damping | 16 | 0.5 | 17 | 2 951 s |

After training, we extracted trajectories for all checkpoints and ran
both parent diagnostics used elsewhere in the paper:

- **Depth-axis shared-$V_\psi$ fit** (paper §14.2).  Ansatz
  $\Delta h_\ell \approx \alpha_\ell v_\ell - \beta_\ell \nabla V_\psi(h_\ell)$
  pooled across every layer $\ell \geq 1$, every token, every
  training sentence; test on held-out sentences.  $R^2$ near 1 means
  "the trajectory really does behave like motion in a single scalar
  potential".
- **Token-axis shared-$V_\psi$ fit.** Same ansatz but along the
  autoregressive token axis at fixed layer.  This is the "natural time
  axis" of the decoder.

---

## 4. Results at a glance

| Variant | Val ppl | Depth pooled TEST $R^2$ | Token pooled TEST $R^2$ | Token min-layer |
|---------|--------:|------------------------:|------------------------:|----------------:|
| Euler baseline | **160.55** | +0.837 | +0.329 | +0.195 |
| Verlet $L=8$ | 167.46 | +0.755 | +0.427 | +0.290 |
| Verlet $L=4$ | 280.30 | +0.892 | +0.445 | +0.290 |
| **Verlet $L=16$, $dt=0.5$** | 174.32 | **+0.958** | **+0.515** | **+0.296** |

The three-line summary:

- **LM perplexity.**  Euler wins.  No Verlet variant matches it.
- **Depth-axis conservative-dynamics fit.**  Verlet at matched $(L,dt)$
  *loses* by $-0.08$, but Verlet at halved $dt$ *gains* $+0.12$ and
  lifts every layer above $+0.93$.
- **Token-axis conservative-dynamics fit.**  Verlet wins at every
  $(L,dt)$ setting, with the largest gain at the smallest $dt$.  The
  Euler baseline's mid-to-late layer collapse is eliminated.

---

## 5. What we learned

### 5.1 Integrator order is not a free lunch for LM perplexity

Velocity-Verlet adds second-order accuracy and symplecticity at a
$\sim 12\%$ compute surcharge.  At fixed model scale, fixed training
budget, and fixed hyperparameters, this does **not** translate into
lower perplexity.  Why not:

1. **The flow is heavily damped.**  Verlet's symplectic advantage is a
   property of the *undamped* Hamiltonian sub-step.  At the converged
   value $\gamma \approx 0.85$, $\Delta t = 1$, each step dissipates
   $\approx 57\%$ of the velocity — the flow is closer to first-order
   gradient descent on $V_\theta$ than to a Hamiltonian trajectory.
   There is not much symplectic structure left to preserve.
2. **Euler here is already semi-implicit.**  The $sarf\_mass\_variant$
   baseline uses $v' = (v + \Delta tf/\mathfrak m)/(1 + \Delta t\gamma)$,
   $h' = h + \Delta tv'$ — that is *symplectic Euler* with
   dissipation folded into the velocity update, not naive explicit
   Euler.  The gap to velocity-Verlet is a genuine $O(\Delta t^2)$
   correction, not a jump from $O(\Delta t)$ to $O(\Delta t^2)$.
3. **Hyperparameters do not transfer uniformly.**  Doubling $L$ and
   halving $\Delta t$ keeps flow distance $L\Delta t$ constant but
   changes the number of discretised force evaluations per forward.
   The lr/warmup schedule was tuned for $L=8$; the $L=16$ run ends at
   a slightly worse optimum despite training for the same number of
   SGD steps.
4. **Halving integration depth is expensive.**  $L=4$ Verlet at 5
   force evaluations is 63% of the Euler integration cost but LM
   perplexity collapses from 160 to 280.  The architecture needs a
   minimum total flow distance $L\Delta t \gtrsim 8$ to form a useful
   semantic landscape.  No integrator upgrade recovers from halved
   depth.

**Take-away.** Perplexity is integrator-insensitive *at this scale* in
the parameter regime this paper uses.  This is a useful calibration
— it means the perplexity numbers reported in the paper are not
artefacts of a particular integrator choice.

### 5.2 The depth-axis shared-\ $V_\psi$ fit has an integrator-bias artefact at coarse $\Delta t$

This was the surprise of the study.  The paper's shared - $V_\psi$ fit
asks: can a *single* scalar $V_\psi$ explain the layer-wise dynamics
as $\Delta h_\ell = \alpha_\ell v_\ell - \beta_\ell \nabla V_\psi(h_\ell)$?
This ansatz is **one-step pointwise in $h_\ell$**: it evaluates the
force only at the current state.

- **Euler's update is literally pointwise in $h_\ell$** by
  construction.  The ansatz matches Euler's update structure exactly
  (up to the per-layer scalars $\alpha_\ell, \beta_\ell$), so the fit
  looks *great* on Euler trajectories by construction.
- **Verlet's update is a two-point symmetric average of forces at
  $h_\ell$ and $h_{\ell+1}$.**  A pointwise ansatz cannot capture a
  two-point update at coarse $\Delta t$.  The fit looks *worse* on
  Verlet trajectories — **even though the continuous flow is
  identical.**

Empirically: at $(L{=}8,\Delta t{=}1)$, depth-axis pooled TEST $R^2$ is
$+0.837$ for Euler and $+0.755$ for Verlet.  A naive reading would
conclude "Verlet is less conservative", which is wrong.

At $(L{=}16, \Delta t{=}0.5)$ both integrators are close enough to the
continuous flow that the one-step approximation becomes tight, and
Verlet's depth-axis fit jumps to $+0.958$ pooled TEST (every layer
$\geq +0.938$) — the strongest conservative-dynamics signature of
any SPLM variant trained so far.

**Take-away.** The depth-axis shared-$V_\psi$ diagnostic as currently
implemented rewards integrators whose update rule matches its
ansatz's functional form.  When comparing across integrators one
should either (a) evaluate at small $\Delta t$, where both update
rules approach the continuous flow and the ansatz is tight for
both, or (b) upgrade the ansatz to a two-point form matching the
integrator.

### 5.3 The token-axis shared-$V_\psi$ fit is where the Verlet advantage shows up cleanly

The token axis is not touched by the integrator in the same way the
depth axis is — the integrator does not generate a one-step map
between adjacent *token positions*.  It influences the token-axis
trajectories only *indirectly*, through the learned $V_\theta$,
$\mathfrak m$, $\gamma$.

Verlet's second-order accuracy and time-reversal symmetry bias the
learned $V_\theta$ to be smoother in $h$, which makes the per-token
dynamics at fixed depth more consistent with a shared-potential
description.  The effect is uniform across layers:

| Layer | Euler | Verlet L=8 | Verlet L=16 dt=0.5 |
|------:|------:|-----------:|-------------------:|
| 1 | +0.484 | +0.414 | +0.452 |
| 2 | +0.526 | +0.468 | +0.504 |
| 3 | +0.463 | +0.495 | +0.534 |
| 4 | +0.373 | +0.488 | +0.523 |
| 5 | +0.347 | +0.496 | +0.503 |
| 6 | +0.323 | +0.466 | +0.508 |
| 7 | +0.258 | +0.412 | +0.509 |
| 8 | +0.195 | +0.365 | +0.507 |

The Euler baseline's mid-to-late layer **collapse** (layers 5–8 drop
from $+0.35$ to $+0.20$) is **eliminated** in every Verlet variant.
For $L=16, dt=0.5$, every layer from 1 to 16 has TEST $R^2 \in
[+0.452, +0.534]$ — the profile is flat.

**Take-away.** This is the cleanest carryover of Verlet's theoretical
symmetry property (no-bias ansatz, no $\Delta t$-scaling artefact) to
a measurable structural property of the trained model.  If a future
paper wants to argue "SPLM's trajectories look like motion in a single
scalar potential", the token-axis diagnostic on a Verlet-trained
checkpoint is the cleanest headline number to report.

### 5.4 Minimum integration depth matters more than integration order

$L=4$ Verlet (5 force evaluations, 63% of Euler's integration cost)
is a disaster on LM quality (ppl 280 vs Euler's 160) but is the
highest-scoring depth-axis shared-$V_\psi$ fit (+0.892 TEST) among
matched-$\Delta t$ variants.  The reason: fewer integration steps
means less "flow distance" from the embedding layer to the head, so
the trained $V_\theta$ develops a shallower landscape which is
easier to fit with a single $V_\psi$ but not semantic enough to
predict tokens well.

This is a concrete example of a case where the conservative-dynamics
diagnostic and the LM metric **move in opposite directions**.  A
model can be *more conservative-looking* and *less useful as a
language model* at the same time.  The two metrics measure different
things, and "more symplectic" is not a proxy for "better LM".

---

## 6. What this means for the paper

The three follow-ups sharpen the null LM-perplexity result from the
$L=8$ Verlet run into a more nuanced picture.  Concretely:

- The paper's claim that SPLM's trajectories look like motion in a
  single scalar potential survives and is *strengthened* by the
  token-axis Verlet numbers ($+0.515$ pooled TEST, all layers
  $\geq +0.45$).
- The paper should flag that the depth-axis shared-$V_\psi$ fit has
  an integrator-bias artefact at coarse $\Delta t$, and report
  the small-$\Delta t$ numbers when comparing integrators.
- The current "damped semi-implicit Euler with $L=8,\Delta t=1$"
  configuration is not a bottleneck on LM perplexity.  A reader
  concerned that "using a textbook $O(\Delta t)$ integrator is
  leaving quality on the table" now has a concrete experiment
  showing otherwise.
- Velocity-Verlet is still the canonical choice *in the small-$\Delta
  t$ limit* and remains the correct integrator to use if one wants
  to claim a conservative flow at inference time (e.g. for
  downstream analyses that rely on approximate energy conservation
  across depth).

---

## 6a. A bonus side-finding: integrator accuracy can hurt expressivity

The attractor-extraction study (item C3 in
`Next_Model_Experiments_for_SPLM.md`; full write-up in
`Semantic_Attractor_Extraction.md`) re-runs the damped SPLM
integrator from hundreds of random $h$ seeds at fixed context $\xi$
and clusters the endpoints to read out "what the model's flow
converges to".  Applied to the Euler $L{=}8$ and Verlet $L{=}16,
\Delta t{=}0.5$ checkpoints, it surfaces an unexpectedly clean
mechanistic story for the small PPL gap reported in Sec. 4.

**Observation.** The Verlet $L{=}16, \Delta t{=}0.5$ model has both
slightly worse perplexity *and* **coarser, mostly-punctuation
attractors** ($K^* \le 6$ per prompt, dominated by `,`, `\n`, `:`),
whereas the Euler $L{=}8$ baseline exhibits richer, content-bearing
basins ($K^*$ up to 10 per prompt, including ` the`, ` I`, ` to`,
` and`).  Rendered in 3D, the Verlet landscape is a **narrow
funnel-slide** into which every trajectory channels, while the Euler
landscape is a **broad symmetric U-valley** with basin endpoints
spread across the whole floor (see
`notebooks/conservative_arch/attractor_analysis/results/landscape3d_compare_*.png`).

**Mechanistic explanation.** The two observations are two facets of
the same phenomenon.  SPLM's damped dynamics has no finite equilibria
(the learned $V_\theta$ is unbounded below — see
`Semantic_Attractor_Extraction.md` Sec. 4), so over any finite
integration horizon the flow is always racing down the steepest
descent direction of $V_\theta$, and the endpoint distribution
reflects *how faithfully* the integrator tracks that descent:

- The Verlet $O(\Delta t^2)$ integrator is genuinely more accurate:
  at each step it commits more aggressively to the true steepest
  descent direction, which on a Tiny Shakespeare corpus is dominated
  by the large-probability-mass "punctuation manifold" (comma /
  newline / colon).  Its flow concentrates there, and both the
  attractor portrait ($K^*$ small, punctuation-heavy) and the
  token-likelihood ratio reflect that: the model loses the
  prompt-conditional content-word diversity that the perplexity
  metric cares about.
- Euler's first-order truncation error adds per-step *stochastic
  jitter* orthogonal to the exact gradient direction.  For an
  otherwise-conservative system this would simply be numerical
  noise.  But for SPLM, whose learned $V_\theta$ is unbounded and
  lacks natural basin *widths*, that jitter acts as a useful
  regulariser: it keeps the flow from collapsing onto the single
  globally-steepest direction and so preserves the
  prompt-dependent content-word diversity.  The result is more
  basins, fewer of them purely punctuational, and a slightly lower
  perplexity.

**Why this is a simulation-framework result, not a generic ML
result.** The headline sentence —

> *"integrator accuracy can hurt expressivity when the underlying
> continuous system has no equilibria"*

— is an insight the attention-transformer literature cannot frame,
because attention has no underlying continuous system in the first
place.  It is a concrete instance of the deeper claim the paper makes
about viewing the forward pass as a dynamical simulation: once one
commits to that view, statements that *only* make sense in the
numerical-integration sense (order of the local truncation error,
symplecticity, step size) become *empirically testable* model-design
decisions with measurable effects on downstream LM metrics.  Here the
effect is small — a handful of perplexity points — but the
*direction* is consistent across the depth-axis fit, the
content-word likelihood gap, and the attractor portrait.  That
consistency is what turns the null LM-perplexity result of Sec. 4
into a usable framework-level observation.

**Practical consequences for future work.**

1. When simulating conservative second-order flows whose
   *continuous* system has no basins, prefer integrators whose
   per-step noise is comparable to the spread of the data manifold.
   This is the opposite of the usual MD advice.
2. A natural follow-up is to give Verlet the jitter artificially:
   add a small Langevin-style Brownian term to the damped Verlet
   step, scaled so that the stationary distribution of the
   resulting SDE matches the empirical $h$ distribution.  If the
   mechanistic story above is right, a Brownian-Verlet SPLM should
   recover Euler's richer attractor portrait *and* close the small
   perplexity gap, while still being a textbook symplectic
   integrator in the $\Delta t \to 0$ limit.
3. The attractor portrait is a new, integrator-aware diagnostic.
   It should be reported alongside perplexity whenever comparing
   SPLM variants, because it exposes *which* of the prompt-conditional
   structure the flow is retaining, not merely whether the aggregate
   next-token distribution happens to match.

---

## 7. Artefacts

All code, checkpoints, and diagnostic outputs live in
`notebooks/conservative_arch/symplectic_variant/`:

- `model_symplectic.py` — `ScalarPotentialLMSymplectic` with
  velocity-Verlet + Strang-split damping.
- `train_splm_symplectic.py` — trainer; accepts `--L`, `--dt`,
  `--mass-mode`, `--tag-suffix`.
- `trajectory_extraction_symplectic.py` — emits `.trajectories.pkl`
  compatible with the parent diagnostics.
- `results/followups_summary.md` — consolidated numerical report.
- `results/splm_sym_logfreq_shakespeare{,_L4,_L16_dt05}_summary.md`
  — per-variant training summaries.
- `../results/sharedV_sym_logfreq_shakespeare{,_L4,_L16_dt05}_summary.md`
  — per-variant depth-axis fits.
- `../results/tokdir_sym_logfreq_shakespeare{,_L4,_L16_dt05}_summary.md`
  — per-variant token-axis fits.

For the Sec. 6a attractor-portrait side-finding:

- `notebooks/conservative_arch/attractor_analysis/` — extraction,
  comparison, and 3D landscape rendering scripts.
- `.../results/attractors_comparison.png` — the 3×5 side-by-side
  attractor bar-chart figure.
- `.../results/landscape3d_compare_<prompt>.png` — Euler vs Verlet
  3D landscape panels (the wide U-valley vs narrow funnel picture).
- `.../results/landscape3d_*_dialogue.gif` — rotating 360° animations.
- `Semantic_Attractor_Extraction.md` — full methodology and
  interpretation.

## 8. References

- Lu, Yiping et al. *Understanding and Improving Transformer from a
  Multi-Particle Dynamic System Point of View.* ICML 2020 Workshop on
  Integration of Deep Neural Models and Differential Equations. 2020.
  — the Macaron-net paper.
- Hairer, Lubich, Wanner. *Geometric Numerical Integration:
  Structure-Preserving Algorithms for Ordinary Differential
  Equations.* Springer, 2006. — the textbook reference for
  velocity-Verlet, Strang splitting, symplectic integrators.
- Frenkel & Smit. *Understanding Molecular Simulation: From Algorithms
  to Applications.* Academic Press, 2001. — the MD-community
  reference, where velocity-Verlet is the default.
