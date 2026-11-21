# Clamping the Low-Rank Precision Factor, and Principled Directions Beyond It

**What this note is.** §28–§29 of the training-instabilities record: the proposed (deferred) clamp on the low-rank precision factor B_k, and the principled directions beyond it. It was split on 2026-11-21 from [`CfC_BAOAB_Integrator_and_Mitigations.md`](https://github.com/dimitarpg13/semsimula-paper/blob/5551315/companion_notes/CfC_BAOAB_Integrator_and_Mitigations.md) (written August to October 2026). The text is that document's, unchanged except for its links, and keeps its section numbers, so a reference to "§N" anywhere in the record or in other notes finds its note through [the index](README.md).

## Contents

- [28. Proposed (Deferred) Mitigation: Clamping the Low-Rank Precision Factor B_k](#28-proposed-deferred-mitigation-clamping-the-low-rank-precision-factor-b_k)
- [29. Principled Directions Beyond the B_k Clamp](#29-principled-directions-beyond-the-b_k-clamp)

---

## 28. Proposed (Deferred) Mitigation: Clamping the Low-Rank Precision Factor $B_k$

> **Status: PROPOSED — DEFERRED until further notice (23-24 August 2026).**
> §28.1-28.2 record a hypothesis; §28.2b upgrades it to a direct
> measurement on this run's own checkpoints. No code has been changed and
> no run has been launched — the clamp itself is still queued behind the
> ongoing gamma=0.10 CfC/BAOAB run and the $L=8$ depth probe.

### 28.1 The observation: $a_k$ is clamped, $B_k$ is not

The anisotropic well's precision is $P_k = \mathrm{diag}(a_k) + B_k B_k^T$
with $B_k \in \mathbb{R}^{d \times r}$ ($r = 4$ in the production runs). Only
the diagonal part is bounded. In
[`model_aniso_gaussian_vtheta.py`](../../notebooks/conservative_arch/parf/model_aniso_gaussian_vtheta.py),
`_components` applies `precision_max` (set to $2/d \approx 0.0052$ at $d=384$)
to $a_k$:

```python
a = (F.softplus(self.a_proj(xi)) + 1e-4).view(*lead, self.K, self.d)
if self._precision_max is not None:
    a = a.clamp(max=self._precision_max)
...
B = self.B_proj(xi).view(*lead, self.K, self.d, self.rank)  # no clamp
```

There is no corresponding bound on $B_k$, and — unlike the isotropic sibling
`MixtureGaussianVTheta` in `model_gaussian_vtheta.py`, which defines a
`clamp_params()` method — the anisotropic class defines none. The factor is
initialised small (`nn.init.normal_(self.B_proj.weight, std=0.01)`) but is
otherwise free to grow throughout training.

### 28.2 Why this is the prime suspect for "curvature exceeding 2"

The SCAF `StiffnessProbe` Weyl bound (Phase 7b/7c) is, per well,

$$K_{\text{Weyl}}(h) = \sum_k g_k \left( \max_i a_k[i] + \sigma_{\max}(B_k)^2 \right),$$

where $g_k = w_k \exp(-\tfrac{1}{2} \delta_k^T P_k \delta_k)$, which is positive and $\delta_k = h - \mu_k$.
The diagonal contribution $\max_i a_k[i]$ is capped at $2/d \approx 0.0052$,
which is tiny; the only term in that per-well bracket that can plausibly push
$\omega \Delta t$ past 2 is $\sigma_{\max}(B_k)^2$ — the squared largest
singular value of the *unclamped* low-rank factor. In other words, the
off-diagonal curvature the Phase 7b/7c audit flagged on the Verlet runs is,
by construction, carried almost entirely by the one quantity that has no
upper bound. This is orthogonal to the well count $K$ (see §27's companion
discussion and the well-count analysis: doubling $K$ adds more equally
unclamped $B_k$ factors, giving *more* opportunities for an outlier, not
fewer).

### 28.2b Confirmed: $B_k$ is already large and still growing in the actual g0.1 CfC/BAOAB run

§28.1-28.2 argue the case from the model definition and from the Verlet-run
Phase 7b/7c Weyl-bound audits (§25-§26). Those audits never inspected
`B_proj` directly — they measured the *aggregate* curvature functional
$K_{\text{Weyl}}(h)$, not its per-term decomposition. To close that gap,
the `B_proj.weight` (and, for reference, `a_proj.weight`/`a_proj.bias`)
tensors were pulled directly from the six pre-spike-burst CfC/BAOAB
checkpoints of this exact g0.1 run (steps 500, 1500, 2500, 3500, 4500,
5500 — the same run whose spike burst starts at step 6297, §27.1), one per
xi-channel of the depth-conditioned bank ($n_{\text{ctx}}=5$ channels).
For each channel, `torch.linalg.svdvals(W_B)[0]` gives the spectral norm
of `B_proj.weight`, a direct proxy for $\sigma_{\max}(B_k)$ at
representative context scale (the bias term is comparatively small and
omitted for brevity):

| step | ch 0 | ch 1 | ch 2 | ch 3 | ch 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500  | 1.4  | 1.6  | 2.3  | 1.3  | 1.7  |
| 1500 | 3.6  | 4.0  | 5.2  | 3.4  | 4.1  |
| 2500 | 5.8  | 6.3  | 7.9  | 5.5  | 6.4  |
| 3500 | 7.9  | 8.5  | 10.1 | 7.4  | 8.4  |
| 4500 | 9.6  | 10.3 | 12.0 | 9.0  | 10.1 |
| 5500 | 10.8 | 11.6 | 14.1 | 10.2 | 11.4 |

(spectral norm $\sigma_{\max}$ of `B_proj.weight` per xi-channel; higher
channel indices correspond to longer context windows in the multi-xi
bank.)

Three observations upgrade §28.1-28.2 from a plausible mechanism to a
confirmed, live one in this run:

1. **Growth is uniform and monotonic across all five channels**, roughly
   6-10x from step 500 to step 5500, with no channel plateauing before the
   window ends. This contrasts with the `depth_code` trajectory of §27.1
   (layer 15 grows explosively then plateaus by step ~2500); $B_k$ shows
   no such saturation over the same span.
2. **The unclamped term already dwarfs the clamped one.** $\sigma_{\max}(B_k)^2$
   at step 5500 is on the order of $10^2$ (squaring ~10-14), while the
   diagonal cap contributes at most $a_{\max} = 2/d \approx 0.0052$ per
   §28.2 — a gap of three to four orders of magnitude that only widens as
   training continues, since one side is architecturally bounded and the
   other is not.
3. **The growth window matches the spike-burst window.** The measured
   checkpoints span exactly the steps leading up to the first hard-trigger
   spikes (step 6297 onward, §27.1's log). This does not prove causation
   on its own, but it rules out the alternative explanation that $B_k$
   growth is a slow, background effect unrelated to the timing of the
   instability.

**Caveat.** `svdvals(W_B)` measures the weight matrix's own spectral norm,
not $\sigma_{\max}(B_k(\xi))$ evaluated at the specific $\xi$ context seen
by any single token — the two differ by whatever gain `B_proj`'s input
activations carry. Since `B_proj` has no output nonlinearity between the
matrix multiply and $B_k$, and the context vector's norm is $O(1)$ by
construction (pre-LN blocks), the weight-matrix spectral norm is a
reasonable order-of-magnitude proxy, but a Phase 7c-style native-trajectory
probe restricted to this run's own checkpoints (rather than this offline
weight inspection) would be the rigorous next step if the clamp is ever
un-deferred.

This measurement is the empirical bridge between §28.2's structural
argument and §28.6's amplification mechanism: it confirms the
$P_k$ that gates $\delta f \approx g_k P_k \delta\mu_k$ in §28.6 is not a
hypothetical worst case but the actual, still-growing operator this run is
integrating against.

### 28.3 Proposed implementation

Mirror the existing `precision_max` pattern with a `precision_lr_max`
(spectral clamp on the low-rank contribution), applied in `_components`
after `B` is formed, so it flows uniformly through `forward`,
`analytical_grad`, `harmonic_terms`, and hence the Weyl bound. Two candidate
forms, cheapest first:

1. **Per-column norm clamp (cheap, elementwise):** rescale each of the $r$
   columns of $B_k$ so its squared norm does not exceed the budget. Bounds
   $\lVert B_k \rVert_F^2$ and therefore $\sigma_{\max}(B_k)^2 \le \lVert B_k \rVert_F^2$, at $O(Kdr)$ cost with no eigensolve.
2. **True spectral clamp (tighter, costlier):** clamp $\sigma_{\max}(B_k)$
   directly via the $r \times r$ Gram eigenvalues (the same
   `eigvalsh(B_k^T B_k)` the Weyl bound already computes), rescaling $B_k$
   when the top singular value exceeds the budget.

Both are pure functions of the freshly-projected $B_k$ (a per-forward
activation, not an `nn.Parameter`), so this is a forward-pass clamp like
`precision_max`, not a projected-gradient step on stored weights — no
optimiser-state interaction, safe under gradient checkpointing.

### 28.4 Validation protocol

1. Add `precision_lr_max` (default `None` = current behaviour) to
   `AnisotropicMixtureGaussianVTheta` / `...MultiContext...` /
   `...DepthConditioned...` and thread it through the notebook config.
2. Pick the budget so the intended max off-diagonal curvature lands safely
   below the $\omega \Delta t = 2$ bound at the production $\Delta t$ and
   mass; start from the observed Phase 7b `eig_max` distribution.
3. Train a short segment (a few thousand steps) from a fixed init or a
   pre-burst checkpoint, with and without the clamp.
4. Re-run the SCAF stiffness audit (Phase 7b/7c) on the resulting
   checkpoints and compare `Weyl frac(>2)`, `eig_max`, and
   `frac_unstable_ci95`. Success = the clamped run's `Weyl frac(>2)` is
   materially lower with no PPL regression attributable to the clamp.

### 28.5 Why deferred, and why still worth queuing

Under CfC/BAOAB the diagonal harmonic force is already integrated exactly
regardless of stiffness (§24), so this clamp is **not** on the critical path
for the current run's spikes (which come from `depth_code`, `creation_gate`,
`destruction_gate`, `register`, `reverse_ch`, `E`/`P` — not the well
curvature). Its value is (a) as a targeted, near-zero-cost hardening of the
*Verlet* failure mode this whole note diagnoses, should the aniso-Gaussian
family ever be run under an explicit integrator again, and (b) as a clean
falsification test of the "unclamped $B_k$ is the curvature culprit"
hypothesis. It is therefore documented now and deferred, rather than
implemented, pending the outcome of the current CfC/BAOAB and $L=8$ runs.
§28.6 below sharpens this: the diagonal/off-diagonal split means $B_k$ is not
implicated in *causing* the current spikes, but it is a live candidate for
*amplifying their severity* through the one channel CfC/BAOAB leaves
unprotected — worth keeping in view rather than fully dismissing.

### 28.6 Mechanism: why well curvature amplifies, rather than merely coexists with, the §27 spikes

This connects §27's `depth_code` finding and §28.1-28.2's curvature finding
into a single causal chain, and sharpens "recovery is deeper/slower" into a
falsifiable, structural claim rather than an intuition.

**The amplification is derivable, not just plausible.** The mixture force is

$$f(h) = \sum_k g_k P_k (\mu_k - h), \qquad P_k = \mathrm{diag}(a_k) + B_k B_k^T,$$

and every well centre is itself a function of the (possibly depth-shifted)
context, $\mu_k = \mu_k(\xi)$. A perturbation to that context therefore
propagates to a force perturbation in two multiplicative stages:

$$\delta \mu_k = \frac{\partial \mu_k}{\partial \xi} \delta \xi \qquad \Longrightarrow \qquad \delta f \approx g_k P_k \delta \mu_k = g_k P_k \left(\frac{\partial \mu_k}{\partial \xi}\right) \delta \xi.$$

The first stage's gain is fixed by `mu_proj`'s weights and has nothing to do
with stiffness. The second stage is scaled by $P_k$ itself: **the same-sized
upstream context perturbation is amplified into a state perturbation in
direct proportion to the local curvature**, and that larger $\delta h$
propagates straight through to the read-out logits as a larger PPL
excursion. Crucially, for the depth-conditioned bank the context *is*
literally $\xi = \xi_{\text{base}} + e_g$ (§14.3), so a `depth_code` step
$\delta e_g$ **is** a $\delta \xi$ here — §27's finding and this section's
finding are not two independent stories, they are two ends of the same
mechanism: `depth_code` supplies the perturbation, $P_k$'s curvature sets
its gain.

**But the gain only bites through the channel CfC/BAOAB leaves explicit.**
Splitting $P_k$ along the same diagonal/off-diagonal line as §28.1-§28.2:

- On the **diagonal part** ($k_{\text{diag}}$, integrated by `cfc_substep`),
  amplification does not translate into *slower recovery*: the A-substep is
  an exact rotation (energy-conserving, `cos`/`sinc`, Jacobian determinant
  exactly 1) and all damping comes from the O-step's $e^{-\gamma \Delta t}$
  factor, which is independent of $\omega = \sqrt{k_{\text{diag}}/m}$ by
  construction (`cfc_baoab.py` composes them as separate substeps
  precisely so damping timescale does not depend on stiffness). A stiffer
  diagonal well rotates the perturbed state faster; it does not, on its
  own, leave it displaced longer.
- On the **off-diagonal residual** ($f_{\text{kick}}$, carrying
  $\sigma_{\max}(B_k)^2$ per §28.2), there is no such protection: it is an
  ordinary explicit velocity kick (`v_mid = v_mid + (dt/m) * f_kick` in
  `model_parf_multixi.py`), with none of the bounded-rotation structure
  that makes the diagonal part immune to its own stiffness. A large,
  depth-code-amplified perturbation routed through this residual can push
  $h$ further away over a step with nothing structurally pulling it back
  within that same step — this is the specific, falsifiable sense in which
  "recovery is deeper/slower": not a property of CfC/BAOAB's core design,
  but of the one piece ($B_k$'s off-diagonal contribution) that design
  deliberately leaves outside it.

**Net reading.** This does not overturn §27's conclusion that `depth_code`,
`creation_gate`, `register`, and the embeddings are the *proximate* sources
of the current burst — they are, by grad-norm rank, and the diagonal V_theta
channel is provably not the bottleneck under CfC/BAOAB. What it adds is a
concrete reason `V_theta`'s own group still appears mid-pack in every spike
(§27.1's `V_theta=502.0` at the worst step): the off-diagonal residual gives
`depth_code`/embedding perturbations a second-order route to amplify their
own functional impact, gated by exactly the unclamped quantity §28.1-§28.2
already flag. It is a plausible *severity multiplier* riding on top of §27's
proximate causes, not an independent trigger — consistent with deferring
§28.3's clamp, but a reason not to fully write it off as Verlet-only
hardening.

---

## 29. Principled Directions Beyond the $B_k$ Clamp

The §28.3 clamp is a gain-attenuator: it shrinks the magnitude of the
amplification multiplier, but leaves the actual pathology in place — a stiff
operator that is integrated **explicitly**. It also does not touch the
driving term. This section records more principled options, organized by
which factor of the instability they attack. All of it is design/roadmap;
nothing here is implemented.

### 29.1 The two orthogonal levers

The instability is a driven stiff system. Schematically, a PPL excursion is

$$\underbrace{\text{driving perturbation}}_{\text{E/P/depth-code spikes}} \times \underbrace{\text{amplification gain}}_{P_k \text{ and integration scheme}} = \text{h-excursion} \to \text{PPL swing}.$$

The clamp shrinks one factor of the gain (the size of $B_k B_k^\top$). It
changes neither *how* that gain is integrated (the real source of blow-up)
nor the driving term. So there are three principled families: **fix the
integration**, **bound the curvature by construction**, and **condition the
source**.

### 29.2 Remove the unstable channel, don't shrink it — low-rank exponential integration

> **Implemented** as `integrator='baoab_cfc_lowrank'`
> (`cfc_baoab.lowrank_modes` / `lowrank_cfc_substep`, wired in
> `model_parf_multixi._layer_step_langevin`; tests in `test_cfc_baoab.py`).
> The construction below is the corrected, code-accurate version; two claims
> in the original sketch turned out to be wrong and are flagged inline.

This is the direction that eliminates the pathology rather than attenuating
it. Recall *why* the diagonal part is safe under CfC/BAOAB (§24): it is
integrated **exactly** by the harmonic propagator, which has no
$\omega \Delta t \lt 2$ restriction *as a standalone flow*. The off-diagonal
is dangerous **only because it is demoted to an explicit $f_{\text{kick}}$**
(§28.6). This connects to §24.6's result that no *explicit* second-order
symplectic integrator beats $\omega \Delta t \lt 2$ — but putting the stiff
linear part in an *exact* flow moves the wall off the stiff frequency.

**The correct split is PSD, not "off-diagonal".** The first instinct — "the
diagonal is already exact, so rotate only the *off-diagonal* remainder
$G G^\top - \mathrm{diag}(G G^\top)$" — **does not work**: that off-diagonal
operator is **indefinite** (it has negative eigenvalues), and an indefinite
"spring" gives a *hyperbolic* flow that amplifies, not a bounded rotation. Any
exact rotation must act on a **positive semidefinite** operator. The clean
split that keeps both parts PSD is

$$H = \underbrace{\mathrm{diag}\Big(\textstyle\sum_k g_k a_k\Big)}_{D_a,\ \text{diagonal precision, PSD}} + \underbrace{\textstyle\sum_k g_k B_k B_k^\top}_{L = G G^\top,\ \text{low-rank, PSD}},$$

with $G = [\sqrt{g_1} B_1, \dots, \sqrt{g_K} B_K] \in \mathbb{R}^{d \times Kr}$.
Note $L$ is the **full** low-rank term including its own diagonal
$\mathrm{diag}(G G^\top)$ — that diagonal is *removed* from the diagonal
channel here (which now carries only $a_k$) so the two channels do not
double-count. `harmonic_terms_lowrank()` returns exactly this
$(D_a, s_a, G, G^\top\mu)$ split, and $f_{D_a}(h) + f_L(h) = -\nabla V(h)$ to
machine precision (tested).

The structural fact that makes absorbing $L$ affordable:

> $L = G G^\top$ has rank at most $Kr$ — fixed by architecture ($K$ wells,
> rank $r =$ `ANISO_RANK`), **independent of how large $\sigma_{\max}(B_k)$
> grows**. For the d=384 run $Kr \approx 8 \times 4 = 32 \ll d = 384$ (and
> $n_{\text{ctx}} Kr$ if the channels are aggregated).

**Impulse / RESPA, not a second exact rotation composed with the diagonal.**
The second wrong instinct is to flow $T + V_{D_a}$ exactly (the current
`cfc_substep`) *and* $T + V_L$ exactly and compose them. That double-counts
the kinetic term $T$ (each factor carries a full drift), integrating the wrong
mass — an $O(1)$ error. Splitting the kinetic term ($\tfrac12 T$ each) fixes
the mass but re-introduces a stability wall, because **composing two
non-commuting harmonic rotations at large angles is itself hyperbolic** — this
is exactly why Verlet has an $\omega \Delta t \lt 2$ wall (it *is* a splitting
method). The working scheme is the **impulse / multiple-time-stepping (RESPA)**
construction: put the *single* stiff operator $L$ in one exact fast flow that
also carries the drift, and demote everything soft to the explicit kick:

- **A-substep** $=$ exact flow of $T + V_L$ (`lowrank_cfc_substep`): free
  drift on the whole state, plus an exact bounded rotation on the $\le Kr$
  modes of $L$. No $\omega_L \Delta t \lt 2$ wall on those modes.
- **B-kick** carries the clamped diagonal spring $D_a$ (bounded curvature —
  `precision_max`, so safe explicitly), $V_\phi$, and the nonlinear V_theta
  residual.

The stiff channel is therefore integrated exactly, so the
amplification-into-blow-up channel from §28.6 is gone for the low-rank modes
**for any $\sigma_{\max}(B_k)$**, clamped or not. §30 works this out concretely.

**Correction to the original A-stability claim.** The impulse scheme is *not*
unconditionally A-stable end-to-end. The fast flow alone is a bounded rotation
for any curvature, but the impulse method has known **isolated resonance
instabilities** at $\omega_L \Delta t \approx k\pi$. Between resonances it is
stable regardless of stiffness, and the O-step damping ($\gamma$) plus the
nonlinear residual attenuate the resonances in practice — but "no wall at all"
overstates it. The honest claim is: the *hard* $\omega \Delta t \lt 2$ wall on
the stiff modes is replaced by *narrow, damped* resonance bands. This is a
large improvement, not a total elimination, which is why §29.3 (bounding
$\omega_L$) remains a genuine complement — see §29.7.

### 29.3 Bound the curvature by construction, smoothly (architectural)

> **Implemented** as the `precision_lr_max` argument on the anisotropic
> Gaussian V_theta classes (`model_aniso_gaussian_vtheta.py`,
> `_bound_lowrank`), mirroring the existing `precision_max` pattern.

The clamp is non-smooth and its threshold is somewhat arbitrary. The
principled version makes the low-rank curvature bound an **invariant of the
parameterization** instead of something monitored:

- **Spectral-normalize `B_proj`** (SN-GAN style): $B_k = s \cdot B_{\text{raw}} / \sigma_{\max}(B_{\text{raw}})$ with $s$ a bounded learnable scale. Differentiable, no discontinuity.
- Or parameterize the whole precision through a bounded spectral map (matrix-sigmoid / Cayley form) so $\mathrm{spec}(P_k) \subseteq [0, a_{\max}]$ is guaranteed, with the real CFL value $a_{\max} = 4m / \Delta t^2$ as the ceiling.

**What was implemented (and its one caveat).** `precision_lr_max` uses a
smooth **Frobenius** cap: it rescales each well's $B_k$ by
$\mathrm{budget} \cdot \tanh(\lVert B_k \rVert_F / \mathrm{budget}) / \lVert B_k \rVert_F$
with $\mathrm{budget} = \sqrt{c}$, where $c$ is the `precision_lr_max`
config value. Since $\sigma_{\max}(B_k) \le \lVert B_k \rVert_F$, this
**guarantees** $\sigma_{\max}(B_k)^2 \lt c$ — differentiable, with
no eigensolve (so no `eigvalsh`-backward degeneracy) and no division by zero.
It is *conservative*: when the low-rank energy is spread across up to $r$
singular values the true $\sigma_{\max}^2$ can be forced below the budget by
up to a factor $r$, so tune `precision_lr_max` against the SCAF Phase 7b/7c
Weyl audit rather than as a literal $\sigma_{\max}^2$ target. A tighter
spectral (SN-style) variant is a natural follow-up.

This attacks "unbounded growth" directly (the §28.2b finding) and pairs
naturally with §29.2 — but **not** by reducing the mode count (that is fixed
at $\le Kr$ regardless, per §29.2). Its role is to bound each mode's
*magnitude* $\omega_L$, which (a) keeps §29.2's impulse resonances shallow and
narrow by keeping $\omega_L \Delta t$ from running away, and (b) keeps the
frozen-coefficient linearization error over $\Delta t$ bounded. See §29.7 for
the division of labor.

### 29.4 Precondition the dynamics so stiffness is uniform (mass/metric)

Set the per-well mass to track curvature, $m_k \propto P_k$ (or a
diagonal/low-rank approximation via the Woodbury identity), so that
$\omega = \sqrt{P/m}$ stays $O(1)$ everywhere regardless of how large $B_k$
grows. This is the Riemannian / natural-dynamics view (Hamiltonian Monte
Carlo with a learned metric): curvature can grow, but the metric absorbs it
so the effective step never destabilizes. More elegant, but it changes the
semantics of inertia and needs an SPD, cheaply invertible metric — the
low-rank structure is exactly what makes the inverse tractable.

### 29.5 Regularize the actual stability quantity (soft, learned, near-free)

The SCAF `StiffnessProbe` already *computes* the Weyl curvature
$K_{\text{Weyl}}(h)$ (§28.2). Add a soft penalty to the loss,

$$\mathcal{L}_{\text{stiff}} = \lambda \cdot \mathrm{relu}\big(K_{\text{Weyl}}(h) - c\big)^2,$$

which trains the model to keep $\omega \Delta t$ away from 2. This turns the
ad-hoc clamp into a differentiable constraint tied to the *true* stability
criterion, requires no integrator change, and is a good cheap hedge to run
alongside §29.2 / §29.3.

### 29.6 Condition the source, not the raw gradients (functional trust region)

The clamp and watchdog bound raw per-group gradient norms — an arbitrary
proxy. The principled version bounds the change in the **induced potential**
per optimizer step: a KL / trust-region on $V_\theta$ (natural gradient in
function space) rather than in parameter space. This limits the
E/P/depth-code perturbations *by their effect on the dynamics*, which is
exactly the quantity that matters, instead of by a unitless norm.

### 29.7 Recommendation: §29.2 + §29.3 together, staged

The preferred fix is **§29.2 and §29.3 combined**, because they do genuinely
different jobs — one is not a weaker version of the other, and neither alone
is sufficient:

| | §29.2 low-rank exponential (impulse/RESPA) | §29.3 smooth bounded curvature |
| --- | --- | --- |
| Fixes | **stability** of forward integration: moves the stiff $L$ into an exact fast flow, so the hard $\omega_L \Delta t \lt 2$ wall becomes narrow damped resonances at $\omega_L \Delta t \approx k\pi$ | **conditioning**: bounds each mode's $\omega_L$, keeping resonances shallow and the linearization error bounded |
| Mechanism | exact flow of $T + V_L$ on the rank-$Kr$ PSD subspace of $G G^\top$, as the A-substep; $D_a$ + residual demoted to the explicit kick | Frobenius cap on $B_k$ so $\sigma_{\max}(B_k)^2 \lt$ `precision_lr_max` |
| Alone gives | stiff modes stable, but resonances deepen and the frozen linearization degrades as $\sigma_{\max}(B_k)$ runs away | a smooth ceiling — but the off-diagonal is **still integrated explicitly** (essentially §28.3's clamp made smooth) |
| Cost | eig of $G^\top G$, $Kr \times Kr$, per token | eltwise Frobenius rescale, no eigensolve |

In words: **§29.2 alone** keeps the stiff modes stable but its resonances
deepen under unbounded growth; **§29.3 alone** caps curvature but leaves the
explicit channel in place; **§29.2 + §29.3** moves the stiff channel into the
exact flow *and* keeps $\omega_L \Delta t$ small enough that the impulse
resonances stay shallow.

**Staging (both landed): §29.3 first, then §29.2.**

1. §29.3 — a small, low-risk change: a smooth Frobenius spectral bound in `_components` right after `B` is formed, mirroring the existing `precision_max` pattern. Immediately validatable through the SCAF Phase 7b/7c Weyl audit (`Weyl frac(>2)` should drop), and the smooth upgrade of the deferred §28.3 clamp.
2. §29.2 on the now-bounded landscape, so its Gram eigendecomposition never sees pathological or near-degenerate singular values.

- **§29.5 (Weyl soft-reg)** remains a near-free immediate hedge that needs no integrator change and can run alongside either step.
- **§29.4 / §29.6** are higher-risk research bets worth noting but not the first move.

**Status: IMPLEMENTED (opt-in), not yet run at scale.** Both §29.3
(`precision_lr_max`) and §29.2 (`integrator='baoab_cfc_lowrank'`,
`lowrank_max_modes`) are implemented and unit-tested; the pre-existing
`baoab_cfc` path is byte-for-byte unchanged, so switching is opt-in via the
notebook config. The next step is an OWT d=384 g0.1 run comparing
`baoab_cfc_lowrank` (with a `precision_lr_max` tuned against the Weyl audit)
to the current `baoab_cfc` baseline, watching whether the §27 E/P/depth-code
spike bursts shrink.
