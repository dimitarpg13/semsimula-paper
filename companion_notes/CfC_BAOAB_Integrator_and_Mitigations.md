# CfC/BAOAB Propagator and Stiffness Mitigations for Fock-PARFLM with Structured $V_{\theta}$

Companion to `Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md`.
This note collects the CfC/BAOAB-integrator analysis, the empirical
depth-code and curvature findings observed under CfC/BAOAB, and the proposed
(deferred) plus forward-looking stiffness mitigations. It was split out of the
parent note (which had grown past 4,700 lines) purely for maintainability; the
§24-§28 content below is unchanged from the parent's former sections of the
same numbers.

**Section numbering.** Sections keep their original numbers from the parent
document (§24 onward) so that every existing cross-reference stays valid.
Cross-references to §1-§23 refer to the parent note
`Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md`; both files
live in the same `companion_notes/` folder.

## Table of Contents

24. [BAOAB + CfC Propagator: Eliminating the Force Cascade at Source](#24-baoab--cfc-propagator-eliminating-the-force-cascade-at-source)
25. [Late-Training Spike Emergence: The Cascade is Universal, Not Depth-Specific](#25-late-training-spike-emergence-the-cascade-is-universal-not-depth-specific)
26. [The Damping Hypothesis: Is Low γ the Dominant Cause of the Cascade?](#26-the-damping-hypothesis-is-low-γ-the-dominant-cause-of-the-cascade)
27. [Empirical Depth-Code Growth: Boundary Layers Dominate in Both Integrators](#27-empirical-depth-code-growth-boundary-layers-dominate-in-both-integrators)
28. [Proposed (Deferred) Mitigation: Clamping the Low-Rank Precision Factor $B_k$](#28-proposed-deferred-mitigation-clamping-the-low-rank-precision-factor-b_k)
29. [Principled Directions Beyond the $B_k$ Clamp](#29-principled-directions-beyond-the-b_k-clamp)
30. [Concrete Sketch: The Low-Rank Exponential Substep](#30-concrete-sketch-the-low-rank-exponential-substep)
31. [SCAF Phase 7b/7c Audit Plan for Tuning precision_lr_max (L=16, and now L=8)](#31-scaf-phase-7b7c-audit-plan-for-tuning-precision_lr_max-l16-and-now-l8)
32. [L=8 baoab_cfc Baseline: Extended Trajectory and the Decision to Switch Mid-Run](#32-l8-baoab_cfc-baseline-extended-trajectory-steps-2700039867-and-the-decision-to-switch-mid-run)
33. [The Bracketing Result Is Modest and Non-Escalating: $B_k$ Is Not the Primary Driver, and a Root-Cause Workflow for the Non-$V_\theta$ Spikes](#33-the-bracketing-result-is-modest-and-non-escalating-b_k-is-not-the-primary-driver-and-a-root-cause-workflow-for-the-non-v_theta-spikes)
34. [`baoab_cfc_lowrank` at L=8 Scale: Correct and Stable, but Not Production-Feasible](#34-baoab_cfc_lowrank-at-l8-scale-correct-and-stable-but-not-production-feasible)
35. [Phase 1/2 Validated: First `replay_spike_batch` Result (Step 37,763) and a Clean Cascade Signature](#35-phase-12-validated-first-replay_spike_batch-result-step-37763-and-a-clean-cascade-signature)
36. [Decoupling Spike Capture from the Reload Trigger: the Plateau, Not Just the Rare Crisis, Is the Real Target](#36-decoupling-spike-capture-from-the-reload-trigger-the-plateau-not-just-the-rare-crisis-is-the-real-target)
37. [Making Cell 6 Resumable In-Place, and Extracting `grad_clip_utils.py`](#37-making-cell-6-resumable-in-place-and-extracting-grad_clip_utilspy)
38. [Seven Replays In: Two Distinct Failure Modes, Not One -- and Only the Localized One Has Crossed 500](#38-seven-replays-in-two-distinct-failure-modes-not-one----and-only-the-localized-one-has-crossed-500)
39. [The Token-Minority Conjecture Is Falsified Twice Over: the Localized Mode Is Batch-Wide, Not Batch-Specific](#39-the-token-minority-conjecture-is-falsified-twice-over-the-localized-mode-is-batch-wide-not-batch-specific)
40. [Does `baoab_cfc_lowrank` Address the Localized Mode? Reconciling §33's Verdict, and a Targeted Ablation Test](#40-does-baoab_cfc_lowrank-address-the-localized-mode-reconciling-33s-verdict-and-a-targeted-ablation-test)
41. [Chronic, Not Transient: Three New Replays Refine §33 and §38, and Motivate Turning On `precision_lr_max`](#41-chronic-not-transient-three-new-replays-refine-33-and-38-and-motivate-turning-on-precision_lr_max)
42. [Step 1 Validated: `precision_lr_max` and `baoab_cfc_lowrank` Both Collapse All Three Replays, a Hook Bug and a Checkpoint-Loading Pitfall Found Along the Way, and the Cap Switched On](#42-step-1-validated-precision_lr_max-and-baoab_cfc_lowrank-both-collapse-all-three-replays-a-hook-bug-and-a-checkpoint-loading-pitfall-found-along-the-way-and-the-cap-switched-on)
43. [Four Repeats of the Same Eval-Time OOM at Step 47,500: `gc.collect()` Was Never Going to Fix It, and Why](#43-four-repeats-of-the-same-eval-time-oom-at-step-47500-gccollect-was-never-going-to-fix-it-and-why)

---

## 24. BAOAB + CfC Propagator: Eliminating the Force Cascade at Source

This section analyses how replacing the Verlet-style integrator with the BAOAB + CfC propagator (from [Closed\_Form\_and\_Hybrid\_Integration\_Strategies\_for\_Fock-PARFLM.md](Closed_Form_and_Hybrid_Integration_Strategies_for_Fock-PARFLM.md) §10) would address the $d=1024$ instability — not merely limit it (as the Tier 1–2 mitigations do) but **structurally eliminate** the second-order gradient cascade.

### 24.1 Why the O-Step Alone Does Not Help

The O-step in BAOAB is the Ornstein-Uhlenbeck friction/noise step:

$$p \leftarrow e^{-\gamma \Delta t}  p + \sigma \sqrt{1 - e^{-2\gamma \Delta t}}  \xi$$

This is the **exact closed-form solution** of the velocity damping equation. It is already perfectly stable by construction and contains **no force evaluation** — it simply rescales the momentum and adds noise.

In the current Verlet implementation, friction enters as the implicit factor $1/(1 + \Delta t \gamma)$ in the denominator:

$$h\_{\ell+1} = h\_\ell + \frac{\delta\_\ell}{1 + \Delta t \gamma} + \frac{\Delta t^2}{m(1 + \Delta t \gamma)} f\_\ell$$

This is a first-order approximation to $e^{-\gamma \Delta t}$. The difference is negligible: at $\gamma = 0.05$, $1/(1+0.05) = 0.952$ vs $e^{-0.05} = 0.951$. **The O-step upgrade is essentially cosmetic for stability.**

The instability lives in the **B-steps** (force kicks), not the O-step. Any intervention that targets only the friction/noise handling leaves the cascade untouched.

### 24.2 The CfC Propagator Removes the Second-Order Chain

The CfC (Closed-form Continuous-time) propagator replaces the B-step force evaluation with an analytical matrix-exponential propagator. Near each Gaussian well centroid $\mu\_k$, the potential is well-approximated by a harmonic oscillator with frequency $\omega\_k = \sqrt{2 V\_0 \kappa\_k^2}$. The exact solution for the undamped harmonic oscillator (the B-step in BAOAB is purely conservative, with damping handled by the O-step) is:

$$\Phi\_k^{\text{B}}(\Delta t) = \begin{pmatrix}\cos(\omega\_k \Delta t) & \frac{\sin(\omega\_k \Delta t)}{\omega\_k}\\ -\omega\_k \sin(\omega\_k \Delta t) & \cos(\omega\_k \Delta t)\end{pmatrix}$$

The blended CfC propagator uses the Gaussian envelope $\alpha\_k(h) = \exp(-\kappa\_k^2 \lVert h - \mu\_k \rVert^2)$ to interpolate between the harmonic propagator (near centroids) and a free-particle ballistic step (far from all wells).

**The key point for stability:** this propagator is a **forward-mode analytical computation**. It requires no `autograd.grad` call and no `create_graph=True`. The well parameters ($\mu\_k$, $\kappa\_k$, $V\_0$) enter through $\omega\_k$ and $\alpha\_k$ in a standard differentiable computation graph. PyTorch's first-order autograd handles parameter gradients naturally via the chain rule through $\Phi\_k$.

The consequence for the gradient chain:

| | Verlet (current) | BAOAB + CfC |
|---|---|---|
| $V\_\theta$ force computation | `autograd.grad(U, h, create_graph=True)` | Analytical propagator $\Phi\_k$ (forward pass) |
| Gradient chain through $V\_\theta$ | **Second-order** ($\nabla^2 U$ at every layer) | **First-order** (standard backprop through $\Phi\_k$) |
| Spectral radius of per-layer Jacobian | Contains $\nabla^2 U$ — can be $> 1$ | Propagator $\Phi\_k$ has spectral radius $\leq 1$ |
| Cascade over $L$ layers | Exponential amplification of Hessian eigenvalues | Bounded (norm-contractive propagator) |

The $V\_\theta$ second-order cascade is **eliminated entirely** — replaced by first-order backprop through a norm-bounded matrix. This is not tweaking a coefficient; it is removing the structural source of the exponential amplification.

The propagator $\Phi\_k$ is norm-bounded because the undamped harmonic propagator is a rotation matrix (spectral radius exactly 1), and the blending weights $\alpha\_k \in [0, 1]$ ensure the convex combination preserves this bound. Over $L=24$ layers, a product of norm-1 matrices remains norm-1 — in stark contrast to the product of Hessian-containing Jacobians that grows exponentially.

### 24.3 Residual Cascade from $V\_\phi$

The current force computation combines $V\_\theta$ and $V\_\phi$ in a single `autograd.grad` call:

```python
U = V_th_per_token.sum() + U_pair
grad_U, = torch.autograd.grad(U.float(), h_in, create_graph=True, ...)
```

In the BAOAB + CfC framework, $V\_\theta$'s contribution is handled analytically, but $V\_\phi$ (the pairwise register interaction) still requires a numerical force evaluation via `autograd.grad`. The question is whether $V\_\phi$ alone — without $V\_\theta$ amplifying the cascade — can produce the O($10^4$) gradient norms observed at $d=1024$.

**Assessment: probably not.** Several structural facts suggest $V\_\phi$'s cascade contribution is much smaller:

1. **Simpler function:** $V\_\phi$ is a pairwise interaction between token–register pairs (MLP-based or attention-based), without $V\_\theta$'s multi-well Gaussian bank with exponential envelopes and depth conditioning. The Hessian of $V\_\phi$ w.r.t. $h$ is correspondingly smaller.

2. **Sparse routing:** the top-$k$ gathered $V\_\phi$ evaluation (`use_gathered_v_phi=True`) restricts the pairwise computation to $k$ neighbours, limiting the rank of the Hessian.

3. **Per-layer scaling:** `per_layer_v_phi_scale` provides a learned attenuation factor $s\_\ell$ that reduces the pair potential's contribution in early layers (where the registers are not yet populated), partially decoupling the cascade.

4. **Strang splitting:** the BAOAB + CfC Strang splitting (§10.3 of the companion note) puts $V\_\phi$'s numerical kicks in half-step sub-intervals, further limiting their cascade contribution.

If empirical testing confirms that $V\_\phi$-only cascades are manageable, the BAOAB + CfC propagator would **fully resolve** the $d=1024$ instability.

### 24.4 Relationship to §23 Mitigations

The Tier 1–3 mitigations of §23 and the BAOAB + CfC propagator attack the same problem from opposite ends:

| Approach | Strategy | What it does to the cascade | Invasiveness |
|---|---|---|---|
| Tier 1 (§23.3) | **Clip the consequence** | Limits parameter gradients after the cascade amplifies | Config-only |
| Tier 2 (§23.4) | **Shorten the cascade** | Reduces $L$, $\Delta t$, or increases $m$ | Config change |
| Tier 3 (§23.5, items 7 & 9) | **Segment the cascade** | Detach boundaries every $K$ layers | Code change |
| **CfC propagator** (this section) | **Remove the cascade at source** | Replaces second-order force chain with first-order analytical propagator | Architectural refactor |

The recommended strategy is **sequential**: apply Tier 1 immediately (already implemented), test whether it stabilises training, and pursue the CfC propagator as the long-term solution — both for stability and for the inference-speed gains documented in [Closed\_Form\_and\_Hybrid\_Integration\_Strategies\_for\_Fock-PARFLM.md](Closed_Form_and_Hybrid_Integration_Strategies_for_Fock-PARFLM.md) §12.

**Update (July 17, 2026):** Full training runs at both d=768 (L=12) and d=1024 (L=16) demonstrated that Tier 1 (per-group clipping) and Tier 2 (reduce $L$) **delay but do not prevent** the cascade from emerging. The d=768 model, which was perfectly stable during the 3,000-step sweep and the first 33,000 steps, developed catastrophic spikes (up to grad=81,019) at step ≈37,000. See §25 for the full analysis. The CfC propagator is now the **only known mitigation that addresses the root cause** and is needed for any training run exceeding ≈30K steps at scale.

**Update (August 23–24, 2026) — Tier 2 re-tested under `baoab_cfc`, on a different failure mode.** The July 17 update above is about the Verlet-era `create_graph=True` autograd chain — a mechanism `baoab_cfc` already removes (`vtheta_analytic_force=True`). But the g0.1/d=384/L=16 OWT run under `baoab_cfc` itself hit a burst of large, uncaught grad-clip spikes at steps 6,297–6,676 (pre-clip totals up to 3,337, dominated by `creation_gate`/`destruction_gate`/`register`/`reverse_ch`/`depth_code`/`V_theta`), moving val PPL from 176.88 to 207.11 across the 6,000→6,500 eval. Because `depth_code` is a per-layer `nn.Parameter` (shape `[L, n_ctx, d]`) and `creation_gate`/`destruction_gate` are per-layer `nn.ModuleList`s while `reverse_ch` is a single weight-tied module reused at every layer, a smaller $L$ shortens the chain a spike has to propagate through, both forward (activation state) and backward (Jacobian-product depth) — a **different** cascade mechanism than the July 17 one, so the "delays but does not prevent" verdict does not automatically transfer.

A single-variable depth probe (same `d=384`, `dt=1`, `integrator=baoab_cfc`, identical V_theta bank/xi/V_phi/schedule/batch — only $L$: 16 → 8) ran **clean for the full 8,000-step slice tested**, including through the exact 6,297–6,676 window that broke the $L=16$ run: zero `[spike]` events, monotonically improving PPL (1,476.67 → 136.06). This supports depth $L$ itself as an **independent contributor** to this burst, on top of (not instead of) the $B_k$/off-diagonal curvature story that motivated §29's `baoab_cfc_lowrank` + `precision_lr_max`. Given the July 17 precedent, this 8,000-step window is **not yet enough to call it resolved rather than delayed**; the probe has been extended (`PROBE_MAX_STEPS: 8_000 -> None`) to run further and determine which it is. The $L=16$ curvature-side mitigation (§29) is being pursued independently and is not gated on this result.

**Decision (August 25, 2026): reactive, not proactive, mitigation for L=8.** §29's curvature-bound (`precision_lr_max`) and low-rank-exponential (`baoab_cfc_lowrank`) mitigations are integrator/V_theta-level and apply at any $L$, so they *could* be added to this probe. They are deliberately **not** being added while it stays clean: doing so would confound "did shortening $L$ alone delay/resolve the burst" (the question this probe exists to answer) with "did bounding curvature also help," destroying the single-variable control against the L=16 baseline. Two mechanical notes for when/if this run does spike:
- `precision_lr_max` alone is not part of the Drive variant tag (it lives in `_bound_lowrank()`, which runs regardless of integrator), so it can be enabled **in place** on this exact checkpoint lineage with no fork.
- `INTEGRATOR='baoab_cfc_lowrank'` **is** part of the variant tag, so switching it always forks to a new Drive folder; warm-starting from this probe's progress needs the desired checkpoint copied into that new folder manually (Cell 2's auto-resume won't find it otherwise).

The plan is to let this run stand as-is until it produces a real turbulence event (or clearly outlasts the L=16-equivalent horizon), then fork from the last good checkpoint / `_prereload` snapshot at that point into a `baoab_cfc_lowrank` + `precision_lr_max` lineage tuned via the same §31.2–31.4 audit procedure — testing whether the curvature fix rescues exactly the failure depth-shortening couldn't prevent, which is a sharper result than testing it pre-emptively on a run that hasn't failed yet. Combined with §31's planned L=16 A/B, this builds toward a 2×2 ($L \in \{8, 16\} \times$ integrator) without committing extra compute until each cell is actually needed.

**Cross-references:**
- [Closed\_Form\_and\_Hybrid\_Integration\_Strategies\_for\_Fock-PARFLM.md](Closed_Form_and_Hybrid_Integration_Strategies_for_Fock-PARFLM.md) — full derivation of the CfC propagator, blending weights, error bounds, and BAOAB integration (§10).
- [Fock-PARFLM\_Scale-Up\_Gamma\_Sweep\_Results\_and\_Damping\_Regime\_Analysis.md](Fock-PARFLM_Scale-Up_Gamma_Sweep_Results_and_Damping_Regime_Analysis.md) §4.5 — the empirical evidence that motivated this analysis.
- [Blended\_CfC\_BAOAB\_Deep\_Dive.md](Blended_CfC_BAOAB_Deep_Dive.md) — fully worked-out construction of the 7-sub-step B̃AOAB̃ scheme.

### 24.5 A second dividend: the propagator unlocks a *safe* position-dependent damping

The propagator's justification in §24.1–24.4 is stability at scale. There
is a second, less obvious payoff that becomes the anchor of a small
implementation roadmap: **the BAOAB/CfC integrator is also the enabling
step for position-dependent damping $\gamma(h)$**
([Position\_Dependent\_Damping\_and\_Reinforcement\_Field.md](Position_Dependent_Damping_and_Reinforcement_Field.md) §9.7).

The reasoning is a direct consequence of §24.1's own observation that
"damping is handled by the O-step." In the current Verlet integrator
friction is baked into the force coefficients $\rho, \beta$, so promoting
$\gamma \to \gamma(h)$ contaminates the `create_graph=True` force step and
adds a new position-dependent term to a backward pass already sitting near
its spectral-radius margin. BAOAB moves all friction into the standalone
O-step; CfC removes the `create_graph` chain from the B-step. Together they
change $\gamma(h)$ from a term *inside* the second-order cascade into a
plain first-order, elementwise rescaling of the momentum in the O-step:

$$v \leftarrow e^{-\gamma(h)\Delta t} v + \sqrt{1-e^{-2\gamma(h)\Delta t}} \sigma \xi,$$

with $\gamma(h)$ evaluated at the position held fixed within the sub-step.
Three consequences follow, developed in full in the companion note's §9.7:

1. **$\gamma(h)$ leaves the second-order chain.** Its backward pass is
   ordinary first-order autograd, and with the $V_\theta$ cascade already
   gone (§24.2) there is nothing left for it to amplify — dissolving the
   "self-defeating" objection that makes $\gamma(h)$ risky on the Verlet
   integrator.
2. **The correct control signal comes for free.** A spike- or
   curvature-aware $\gamma(h)$ wants the local curvature; CfC already
   computes the local harmonic frequency $\omega_k = \sqrt{2V_0\kappa_k^2}$
   (§24.2), so $\gamma(h)=\gamma_0 + \kappa \phi(\omega_k)$ is an analytic,
   first-order-differentiable byproduct of the B-step.
3. **Strong spatial variation is safe.** The O-step is the exact OU
   solution for any $\gamma\ge0$, so a sharply varying $\gamma(h)$ never
   destabilises the integrator.

Deeper still: once CfC removes the cascade at source, the *reason* one
would reach for $\gamma(h)$ changes. The whole tension flagged in
`Corpus_Statistics...md` §13.5 — that the fine-settling parameterisation
starves damping exactly where the cascade Jacobian is largest — assumes a
cascade to be starved. Remove it and that danger evaporates, so $\gamma(h)$
reverts from a *stability governor* (the §9.6 framing) to its original role
as an inference-geometry / fine-settling knob.

**Implementation roadmap (the ordering is causal).**

| Step | Item | Why it must come first |
| ---: | --- | --- |
| 1 | **CfC/BAOAB propagator** (§24) | Removes the cascade at source; keeps second-order forward geodesics; stabilises turbulent corpora. Load-bearing on its own. |
| 2 | **Position-dependent damping $\gamma(h)$** on top | Only *after* CfC: it removes the obstacle that makes $\gamma(h)$ dangerous, creates the O-step as its physically faithful home, supplies the $\omega_k$ control signal, and lets a constant-$\gamma$ CfC baseline de-risk attribution. |

Verify a constant-$\gamma$ CfC run first (geodesics reproduced, spikes
gone), then layer $\gamma(h)$ on and judge it by reload-and-geometry
diagnostics rather than by spike suppression, which CfC already owns.

**Mixed-corpus corollary.** This roadmap also answers how to run the
turbulence-prone second-order corpora (e.g. OpenWebText) that cannot be
reduced to first order without losing inference-time geodesic realism. The
first-order-sufficiency analysis
([Corpus\_Statistics\_and\_the\_First\_vs\_Second\_Order\_Well\_Gap.md](Corpus_Statistics_and_the_First_vs_Second_Order_Well_Gap.md) §12)
lets a corpus be partitioned by its anharmonicity: where $A_i \ll 1$
first-order dynamics is a certified substitute and carries no cascade;
where it is not (high predictive information, long-range dependence), the
CfC/BAOAB second-order propagator keeps the geodesics genuine while
removing the spikes. The training process is therefore *corpus-partitioned*
— first-order where sufficiency holds, CfC-second-order where it does not —
rather than a single global integrator choice.

### 24.6 Is there a different explicit symplectic integrator that tolerates stiffer wells than Verlet?

Before committing to the CfC rewrite it is worth asking whether a
smaller change — swapping Verlet for some other explicit,
Euler-family, symplectic integrator — could push the
$\omega\Delta t \lt 2$ bound of §24.2/§4.1 of
[PyTorch Implementation of CfC/BAOAB](https://github.com/dimitarpg13/semantic_simulation/blob/main/docs/BAOAB/PyTorch_Implementation_of_CfC_BAOAB_in_Fock-PARFLM.md#41-the-verlet-stability-bound)
higher and avoid the second-order-cascade removal described in
§24.2 above. It cannot, for three separate reasons, each of which
rules out one natural candidate:

1. **Symplectic (semi-implicit) Euler has the identical bound.** One
   step is $v \to v - \omega^2\Delta t h$ then $h \to h + \Delta t v$
   using the updated $v$; as a $2\times2$ map this has $\det=1$ and
   $\mathrm{tr}=2-\omega^2\Delta t^2$, giving the same
   $\omega\Delta t \le 2$ threshold as Verlet's characteristic
   equation. Not a coincidence — velocity-Verlet on the harmonic
   oscillator is algebraically two symplectic-Euler half-steps glued
   together, so both inherit the same bound.
2. **Higher-order explicit symplectic composition (Yoshida,
   Forest-Ruth) generally *shrinks* the bound, not raises it.**
   Reaching 4th order via composed Verlet substeps requires at least
   one negative substep (Suzuki-Sheng theorem, for any symmetric
   composition of order $\ge3$), and a negative substep is a
   destabilising direction for a stiff linear mode. Composing for
   accuracy and composing for stability margin move in opposite
   directions for exactly the linear-well regime that produces the
   $d=768$/$1024$ spikes documented in §23–25.
3. **General barrier: explicit stability functions are polynomials,
   and polynomials are unbounded.** For the harmonic model, any
   one-step method is $y\_{n+1}=R(i\omega\Delta t)y\_n$; any explicit
   method (fixed number of force evaluations, no matrix inverse per
   step) has $R$ polynomial in $i\omega\Delta t$, which cannot stay
   bounded by 1 as $\omega\Delta t\to\infty$ — the imaginary-axis
   specialisation of the standard fact that no explicit Runge-Kutta
   method is A-stable. There is therefore no finite-stage, explicit,
   Euler-family redesign that removes the crossing documented
   empirically in §25.3-25.4's per-well curvature growth; the ceiling
   is structural, not a Verlet-specific artefact.

Two designs do escape the bound, and CfC (§24.2) is deliberately the
second, not the first: **implicit midpoint** / Gauss-Legendre
collocation makes $R$ a rational Padé approximant to $e^{z}$ —
unconditionally stable, but a genuinely implicit nonlinear solve per
layer for a non-quadratic $V\_\theta$, and wrong-phase at large
$\omega\Delta t$ (right energy, wrong oscillation frequency); or
**exact propagation of the locally-frozen linear part** — the
$\Phi\_k$ rotation of §24.2 — which is both unconditionally stable
and phase-exact, with no implicit solve because only $\omega\_k$
itself changes step to step, not the equation being integrated
within a step. Put plainly: the CfC/BAOAB rewrite is not one
adequate fix chosen among several comparably good alternatives to
Verlet — within "explicit, one global $\Delta t$" the bound in
§24.2 is close to the ceiling, and CfC is the cheaper and more
accurate of the only two ways past it.

Full derivation of both the symplectic-Euler bound and the general
polynomial-stability argument: [PyTorch Implementation of CfC/BAOAB, §4.3](https://github.com/dimitarpg13/semantic_simulation/blob/main/docs/BAOAB/PyTorch_Implementation_of_CfC_BAOAB_in_Fock-PARFLM.md#43-why-not-a-different-explicit-symplectic-integrator).
Same argument in the SPLM tutorial framing:
[Symplectic Integration for SPLM, §1.6](Symplectic_Integration_for_SPLM.md#16-how-far-can-an-explicit-symplectic-integrator-be-pushed).

## 25. Late-Training Spike Emergence: The Cascade is Universal, Not Depth-Specific

### 25.1 Background

Sections 23–24 attributed the catastrophic gradient spikes at d=1024 to the depth of the second-order gradient cascade: L=24 produced a 24-deep chain of `autograd.grad(create_graph=True)` calls, with exponential amplification causing gradient norms up to 7,870 (L=24) and 63,949 (L=16 at lr=1.5e-4). The working hypothesis was that reducing $L$ would proportionally reduce the cascade severity.

### 25.2 d=768 at L=12: The delayed cascade

Full training of d=768 (L=12, 137M params, gamma=0.05) at lr=2e-4 on a single H100 revealed that **the same catastrophic spike pattern emerges after ≈37,000 steps** — despite L=12 producing zero spikes during the 3,000-step gamma sweep and the first ≈33,000 steps of full training.

#### Observed spikes (step 37,576–38,070):

| Step | Pre-clip grad | Top groups |
|:----:|:------------:|------------|
| 37,700 | 128.9 | P=91, E=91 |
| 37,715 | 785.0 | P=534, E=534, creation_gate=207 |
| 37,763 | **14,988.6** | P=10,049, E=10,042, creation_gate=4,738 |
| 37,766 | **20,704.2** | P=14,281, E=14,280, register=3,355 |
| 37,829 | **4,697.9** | E=3,181, P=3,181, reverse_channel_scale=2,089 |
| 37,840 | **81,019.2** | **P=78,417**, E=16,547, creation_gate=9,928 |
| 37,975 | 1,265.5 | P=1,237, E=220, creation_gate=113 |
| 38,054 | 1,159.1 | P=761, E=761, destruction_gate=325 |
| 38,070 | **9,378.7** | P=6,463, E=6,456, creation_gate=2,035 |

**The worst spike at d=768 (grad=81,019 at step 37,840) exceeds the worst spike at d=1024 (grad=63,949 at step 10,041).** The same parameter groups dominate: `P` (positional embedding), `E` (input embedding), and `creation_gate`.

#### Key comparison:

| | d=768 (L=12) | d=1024 (L=16, lr=1.5e-4) |
|---|---|---|
| Spike onset | Step ≈37,000 | Step ≈4,000 |
| Worst spike | **81,019** | 63,949 |
| Top spike group | `P` = 78,417 | `E` = 42,247 |
| PPL at onset | ≈93 | ≈260 |
| Model still learning? | Yes (PPL improving) | Stalling |
| Watchdog reloads | 0 (as of step 38,000) | 1 |

### 25.3 Revised understanding

The original framing — that L=24 is "too deep" while L=12 is stable — was **correct for short sweeps but wrong for full training**. The second-order gradient cascade is a function of both depth ($L$) and training duration:

1. **Early training:** The force field $-\nabla_h U$ is weak (the potential surface is approximately flat) and the Hessian eigenvalues are small. The cascade amplification factor is close to 1.0 per layer, so even L=24 would be stable.

2. **Mid training:** As the potential landscape develops sharper features (deeper wells, steeper barriers), the Hessian eigenvalues grow. The per-layer amplification factor exceeds 1.0, and the cascade begins to compound. Deeper models ($L=24$) reach this threshold first ($\sim$step 4K) because the cascade compounds over more layers.

3. **Late training:** Even shallower models ($L=12$) eventually develop potential landscapes with large enough Hessian eigenvalues that the 12-layer cascade amplifies to catastrophic levels ($\sim$step 37K). The cascade is **delayed, not prevented**, by reducing $L$.

This can be expressed as a rough scaling law for the cascade onset step:

$$\text{step}\_{\text{onset}} \propto \frac{1}{L} \cdot \frac{1}{\lambda\_{\max}(H_0)}$$

where $\lambda\_{\max}(H_0)$ is the initial rate of Hessian eigenvalue growth, which depends on $d$, the learning rate, and the corpus difficulty.

### 25.4 Implications for the mitigation tiers

The finding invalidates the **Tier 2 mitigation (reduce $L$)** as a long-term solution. The updated assessment:

| Tier | Strategy | Short-term | Long-term | Status |
|:----:|----------|:----------:|:---------:|:------:|
| 1 | Per-group clip + force clamp | Effective | **Degrades** (clip fraction grows) | Implemented |
| 2 | Reduce $L$ | **Delays onset** | Does not prevent | Applied (L=24→16) |
| 3 | CfC propagator | N/A | **Only root-cause fix** | Not yet implemented |
| — | Reduce LR | **Delays onset** | Delays but does not prevent | Being tested (d=1024) |

The **BAOAB + CfC propagator (§24)** is now the only known mitigation that can prevent the cascade from emerging at any training length, because it removes the `create_graph=True` chain entirely.

### 25.5 Why the models survive (for now)

Despite spikes reaching 81,019 at d=768 and 63,949 at d=1024, both models continue to learn (d=768 PPL=93.17, still improving). This is because:

1. **Spikes are intermittent**, not sustained — perhaps 1 in 20 steps triggers a spike, and the remaining steps receive clean gradients.
2. **Per-group clipping** truncates the spike direction but preserves some gradient signal. The clipped gradient is not zero — it points in a direction that still has a component of the true gradient.
3. **AdamW's momentum** smooths out spike steps. A single spike step has limited impact on the exponential moving averages of the first and second moments.
4. **The watchdog** rolls back to the best checkpoint if sustained instability is detected, preventing catastrophic divergence.

However, as training progresses and the potential landscape sharpens further, the spike fraction is expected to grow. At some point, the fraction of useful (non-clipped) gradient steps will drop below the threshold needed for continued learning, and PPL will stall. This is likely what happened to d=1024 at lr=1.5e-4, where PPL stalled at ~258 between steps 6,500 and 9,000.

### 25.6 Practical recommendations

1. **For ongoing runs (d=768, d=1024):** Reduce LR when spikes become frequent. The current d=1024 run resumed from step 9,000 with lr=5e-5 (3× reduction) and grad_clip=0.5. The d=768 run may need a similar LR reduction if PPL stalls.

2. **For the paper:** The spike onset timing and severity should be documented as empirical evidence that the `create_graph=True` force computation has a fundamental scalability limit. This motivates the CfC propagator as a necessary architectural evolution, not just an optional optimization.

3. **For future architectures:** The CfC propagator should be implemented before attempting scale-ups beyond d=1024 or training runs beyond ~50K steps at any scale. The 3,000-step gamma sweep protocol is validated for finding the optimal $\gamma$ but cannot predict whether a full training run will be stable.

---

## 26. The Damping Hypothesis: Is Low γ the Dominant Cause of the Cascade?

### 26.1 Observation

Sections 23–25 attributed the catastrophic gradient spikes at d≥768 to the `create_graph=True` second-order chain through $L$ layers of force computation. However, a striking confound has been overlooked: **the two stable runs and the two unstable runs differ not only in $d$ and $L$, but also in γ**.

| Run | $d$ | $L$ | $\gamma$ | Max grad | Watchdog reloads | Regime |
|-----|:---:|:---:|:--------:|:--------:|:----------------:|--------|
| d=384 Phase 1 | 384 | 16 | **0.30** | 757 | 0 | Stable |
| d=384 Phase 2 | 384 | 16 | **0.30** | 1,703 | 0 | Stable |
| d=384 Phase 3 | 384 | 16 | **0.30** | 7,427 | 0 | Stable |
| d=768 | 768 | **12** | **0.05** | 5,158,336 | 2 | Catastrophic |
| d=1024 | 1024 | 16 | **0.05** | 63,949 | multiple | Catastrophic |

Crucially, d=768 has **fewer layers** ($L=12$) than d=384 ($L=16$) — yet its worst spike is **7 orders of magnitude** larger. If the cascade depth $L$ were the primary driver, d=384 should be worse, not better. This points to $\gamma$ as the dominant variable.

### 26.2 Mechanistic argument: damping controls cascade amplification

The Velocity-Verlet integrator with Langevin friction updates each layer as:

$$v_{l+1} = (1 - \gamma   dt)   v_l + F(h_l)   dt, \qquad h_{l+1} = h_l + v_{l+1}   dt$$

The training gradient $\partial \mathcal{L}/\partial \theta$ must differentiate through the force $F = -\nabla_h V$ via `create_graph=True`, producing second-order terms $\partial^2 V / \partial h   \partial \theta$. The severity of this cascade depends on how far perturbations in $h$ propagate across layers — which is controlled by the **per-layer velocity attenuation factor** $(1 - \gamma)$:

| Property | Low $\gamma$ (0.05) | High $\gamma$ (0.30) |
|----------|:-------------------:|:--------------------:|
| Per-layer velocity attenuation | $(1 - 0.05) = 0.95$ | $(1 - 0.30) = 0.70$ |
| Residual velocity after $L=12$ | $(0.95)^{12} \approx 0.54$ | $(0.70)^{12} \approx 0.014$ |
| Residual velocity after $L=16$ | $(0.95)^{16} \approx 0.44$ | $(0.70)^{16} \approx 0.003$ |
| Effective memory horizon | ≈20 layers | ≈3 layers |
| Dynamical regime | Nearly conservative (ballistic) | Overdamped (gradient-descent-like) |
| Jacobian spectral radius | $\approx 1$ (perturbations persist) | $\ll 1$ (perturbations decay) |

**The key number: at $\gamma=0.05$, a velocity perturbation retains 54% of its magnitude after 12 layers. At $\gamma=0.30$, it retains only 1.4%.** This is a **38× difference** in how much perturbation energy survives to compound in the backward pass.

The backward pass through `autograd.grad(create_graph=True)` computes the chain:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \sum_{l=1}^{L} \frac{\partial \mathcal{L}}{\partial h_L} \cdot \prod_{k=l}^{L-1} J_k \cdot \frac{\partial F_l}{\partial \theta}$$

where $J_k = \partial(h_{k+1}, v_{k+1}) / \partial(h_k, v_k)$ is the per-layer Jacobian. The spectral radius $\rho(J_k)$ determines whether the product $\prod J_k$ grows or decays:

- **$\gamma = 0.05$:** $\rho(J_k) \approx 1 - 0.05 + \mathcal{O}(\lambda_{\max}(H_V))$. When the Hessian eigenvalue $\lambda_{\max}(H_V)$ exceeds $\gamma/dt$, the spectral radius exceeds 1.0 and the product grows exponentially with $L$. This is the **cascade onset condition**.

- **$\gamma = 0.30$:** $\rho(J_k) \approx 1 - 0.30 + \mathcal{O}(\lambda_{\max}(H_V))$. The Hessian eigenvalue must exceed a **6× larger threshold** before the spectral radius exceeds 1.0. This dramatically raises the bar for cascade onset.

In other words, **$\gamma$ sets the stability margin**: the gap between the current Hessian eigenvalues and the critical threshold for exponential gradient amplification. Low $\gamma$ leaves almost no margin; high $\gamma$ provides a large buffer.

### 26.3 Why the gamma sweep missed this

The 3,000-step gamma sweep correctly identified $\gamma = 0.05$ as the PPL-optimal value at that horizon. But the catastrophic spike regime does not onset until step ~50K at d=768 (§25.2) — the sweep runs for only 6% of that window.

This is a **short-horizon optimisation trap**:

| Training stage | Low $\gamma$ (0.05) | High $\gamma$ (0.20–0.30) |
|:--------------:|:-------------------:|:-------------------------:|
| Steps 0–3K (sweep window) | ✅ Best PPL (ballistic exploration is aggressive) | ❌ Slightly worse PPL (dynamics are more conservative) |
| Steps 3K–50K | ✅ Still fine (Hessian eigenvalues below threshold) | ✅ Fine |
| Steps 50K+ | ❌ **Catastrophic spikes** (Hessian exceeds stability margin) | ✅ Likely stable (large stability margin) |
| Steps 65K–100K (WSD decay) | ❌ **Watchdog reloads**, wasted steps, regression | ✅ Smooth decay, full PPL compression |

The sweep selects $\gamma$ that is optimal **conditional on the dynamics remaining stable**, which they do not. The long-run optimal $\gamma$ may be substantially higher.

### 26.4 The confound with $d$

One might argue that the instability is driven by $d$ (larger hidden dimension → sharper potential landscape → larger Hessian eigenvalues), not $\gamma$. This is partially true: the Hessian eigenvalue growth rate $\lambda_{\max}(H_V)$ almost certainly increases with $d$ as the model develops more refined representations. However, $\gamma$ controls the **tolerance** for those eigenvalues:

- At $\gamma = 0.30$, the stability margin is large enough to absorb the Hessian growth at d=384 across 500K steps without a single cascade event.
- At $\gamma = 0.05$, the stability margin is so thin that even the moderate Hessian growth at d=768 triggers catastrophic cascades by step 50K.

**The hypothesis is not that $d$ is irrelevant, but that $\gamma$ modulates the cascade severity multiplicatively**, and the gamma sweep inadvertently selected a $\gamma$ that minimises the stability margin.

### 26.5 Expected effect of $\gamma = 0.20$ at d=768

At $\gamma = 0.20$, the per-layer attenuation is $(1 - 0.20) = 0.80$, giving:

| Metric | $\gamma = 0.05$ | $\gamma = 0.20$ | Ratio |
|--------|:---------------:|:---------------:|:-----:|
| Residual velocity ($L=12$) | $(0.95)^{12} = 0.54$ | $(0.80)^{12} = 0.069$ | 7.8× more damping |
| Stability margin ($\gamma / dt$) | 0.05 | 0.20 | 4× higher threshold |
| Jacobian product decay | Near-neutral | Exponentially decaying | Qualitatively different |

The 7.8× increase in velocity damping and 4× increase in stability margin should:

1. **Prevent the exponential cascade**: Hessian eigenvalues that trigger cascades at $\gamma = 0.05$ remain safely below threshold at $\gamma = 0.20$.
2. **Eliminate or drastically reduce spike severity**: The multiplicative amplification across 12 layers is cut from near-neutral to strongly decaying.
3. **Produce d=384-like training stability**: The dynamical regime at $\gamma = 0.20$ is qualitatively similar to d=384's $\gamma = 0.30$ — overdamped, with rapid perturbation decay.

**The cost** is some PPL sacrifice in early training (the gamma sweep showed higher $\gamma$ → higher short-run PPL). However, the **net long-run PPL** may actually be *better* because:

- No watchdog reloads wasting ~10K effective steps each
- No post-reload regression and multi-thousand-step recovery periods
- Smooth WSD decay phase delivering full PPL compression
- No risk of cascade re-escalation during the critical decay window

### 26.6 Hints at a γ–d scaling law

The optimal-stable $\gamma$ may follow a dimension-dependent pattern:

| $d$ | $\gamma$ (PPL-sweep optimal) | $\gamma$ (geodesic-optimal) | $\gamma$ (training-stable, empirical) |
|:---:|:----------------------------:|:---------------------------:|:-------------------------------------:|
| 384 | 0.25 | 0.05 | 0.30 ✓ (500K steps, zero reloads) |
| 768 | 0.05 | 0.05 | 0.05 ✗ (catastrophic at 50K) |
| 1024 | 0.05 | — | 0.05 ✗ (catastrophic at 4K) |

At d=384, the training-stable $\gamma$ (0.30) is **close to the PPL-optimal** (0.25) and far above the geodesic-optimal (0.05). At d≥768, the PPL-sweep optimal and geodesic-optimal happen to **coincide** at 0.05 — but this value is training-unstable.

The PPL-geodesic coincidence at d=768 (documented in the gamma sweep analysis) may be a red herring for training: it selects a $\gamma$ that produces beautiful near-geodesic dynamics in the short run but catastrophic gradient cascades in the long run. The **training-stable $\gamma$ at d=768 likely lies in the range 0.15–0.25**, similar to d=384 — in the overdamped regime where PPL and geodesic optimality diverge.

### 26.6b Counter-evidence: the hypothesis reverses on the aniso-Gaussian $V_\theta$ family (August 20, 2026)

**Every run in §26.1's table, including the d=384/γ=0.30 "stable" anchor, uses the SQ3 (structured-quadratic-mixture) $V_\theta$** (this note's header). A same-width test on a *different* $V_\theta$ family — the bounded anisotropic-Gaussian + Fock-reg configuration of `Determining_optimal_gamma_for_Fock-PARFLM.md` §12.5, `d=384`, `L=16`, two full 100K-step runs differing only in `FIXED_GAMMA` — gives the **opposite** ordering:

![Validation-PPL trajectory for the two d=384, L=16 aniso-Gaussian + Fock-reg full runs, gamma_train=0.10 versus gamma_train=0.30, with watchdog reload steps marked as vertical dashed lines](images/gamma_d384_ppl_comparison.png)

Note the red dashed lines (γ=0.30's watchdog reloads, now five: steps 7,124; 7,891; 8,093; 8,421; 9,477) landing on a curve that goes flat as early as step 6,000 and only breaks through at step 10,000, versus the green dashed lines (γ=0.10's two reloads at 8,925 and 10,697) landing on a curve that is still descending — a visual restatement of "reloads that interrupt real progress" versus "reloads that interrupt a plateau nobody would have missed." The γ=0.30 run disconnected once mid-training and was resumed from a step-7,500 checkpoint; the re-pulled log now runs to step 11,206, within 300 steps of γ=0.10's 11,500-step horizon, so the table below is reported at this near-matched endpoint rather than the original ~8,039-step read.

| Run | $d$ | $L$ | $\gamma$ | $V_\theta$ family | Watchdog reloads (by ~matched endpoint) | Max pre-clip grad |
|---|:---:|:---:|:---:|---|:---:|---:|
| d=384, aniso-Gaussian | 384 | 16 | **0.10** | anisotropic Gaussian | **2** (by step 11,500) | 3,899 |
| d=384, aniso-Gaussian | 384 | 16 | **0.30** | anisotropic Gaussian | **5** (by step 11,206) | 37,229 |
| d=384, Phase 1–3 (§26.1) | 384 | 16 | 0.30 | SQ3 (structured quadratic) | 0 (over 500K cumulative steps) | 7,427 |

At $d=384$, on aniso-Gaussian, $\gamma=0.30$ now shows $2.5\times$ as many watchdog reloads as $\gamma=0.10$ over a near-matched horizon and a worst spike nearly $10\times$ larger — a wider gap than the first (~8,039-step) read, not a narrower one — the opposite of what §26.2's constant-Hessian argument predicts, and the opposite of what the SQ3 row of this very table shows at the identical $(d, L, \gamma)=(384, 16, 0.30)$ triple. Since §26.2's mechanism (per-layer velocity attenuation $(1-\gamma)^L$ setting the cascade margin) is a property of the shared Verlet integrator and should not depend on which $V_\theta$ is plugged into the force term, the reversal implies the constant-Hessian toy model is missing a $\gamma$-*dependent* term that differs between the two $V_\theta$ families — most plausibly a $\gamma$-dependent change in $\lambda_{\max}(\nabla^2_h U)$ for the bounded Gaussian well (whose curvature saturates away from well centres, unlike SQ3's unbounded quadratic), or an interaction specific to the aniso-Gaussian run's depth-conditioning and register-gate machinery (its largest spikes are dominated by `depth_code`, `creation_gate`, and `register` groups, none of which SQ3 has). See `Determining_optimal_gamma_for_Fock-PARFLM.md` §12.5 for the full comparison and open questions #8–#9.

**Revised statement of the Damping Hypothesis.** "Raising $\gamma_{\mathrm{train}}$ increases the cascade stability margin" is confirmed for the SQ3 $V_\theta$ family at $d=384$ and **falsified** for the aniso-Gaussian family at the same $d$. The hypothesis should therefore be read as **architecture-conditional**, not as a universal property of the `create_graph` second-order chain — §26.6's γ–$d$ scaling-law table (and by extension the §26.5 recommendation to raise $\gamma$ for stability at $d\ge768$) is validated only within the SQ3 family until re-tested on aniso-Gaussian at those widths.

### 26.7 Experimental plan

The validation strategy is designed around **information-efficient sequencing**: spend the minimum compute to resolve the key uncertainty before committing to expensive full runs.

#### Why Phase 1a (fresh-init comparison) was dropped

An earlier version of this plan included a 10K-step fresh-init run at $\gamma = 0.20$ to compare gradient profiles against the $\gamma = 0.05$ Phase 1 logs. This was abandoned for two reasons:

1. **No baseline data:** The d=768 $\gamma = 0.05$ training log (JSONL) records `grad_norm` every 50 steps as a point sample, but does not capture the maximum gradient norm within each window. The terminal output showing between-step catastrophic spikes (e.g., grad=5.16M at step 51,898) is not persisted — early-step terminal output is lost to scrollback. There is no reliable gradient-norm baseline to compare against.

2. **The first 10K steps are not discriminating:** Even at $\gamma = 0.05$, the first 10K steps were clean (JSONL max grad ≈395–482, only 1 spike > 100). The cascade does not onset until step ≈37K–50K, when the Hessian eigenvalues exceed the thin stability margin. A 10K fresh-init comparison would show **both** runs looking clean — it cannot distinguish the two $\gamma$ values.

Both problems are solved by the **cross-gamma Phase 2 test** (below), which starts from a checkpoint whose Hessian has *already* exceeded the $\gamma = 0.05$ stability margin.

#### Prerequisite: Improved gradient logging

Before running the validation, `train_fock.py` should be patched to log `max_grad_in_window` — the maximum gradient norm seen across all steps within each 50-step JSONL logging interval. This ensures that future runs capture spike severity at full resolution, enabling fair cross-run comparisons. See `train_fock.py` for the implementation.

#### Step 1: Complete d=768 Phase 1 at $\gamma = 0.05$ (~18 hours remaining)

The current run is at step 81,500 / 100,000. The WSD decay is actively compressing PPL (best 84.04 at step 81,500, with 4 consecutive new bests). Let it finish — every remaining step is valuable, and the final Phase 1 PPL at $\gamma = 0.05$ becomes the **baseline** for comparison.

**Deliverable:** Phase 1 best checkpoint and final PPL at $\gamma = 0.05$.

#### Step 2: Cross-gamma Phase 2 test (10K steps, ~10 hours) — THE CRITICAL EXPERIMENT

Run 10K steps of Phase 2 starting from the **$\gamma = 0.05$ Phase 1 best checkpoint** but using $\gamma = 0.20$ (with `FRESH_SCHEDULE=True`, `SKIP_OPTIMIZER_STATE=True`).

**Why this is the most discriminating test:** The Phase 1 best checkpoint contains a model at PPL ~65–70, whose potential landscape is sharp enough that $\gamma = 0.05$ produced catastrophic spikes (grad > 5M) in the second half of Phase 1. By resuming from this checkpoint at $\gamma = 0.20$, we test the hypothesis at the exact model state where it matters — a model whose Hessian has **already exceeded** the $\gamma = 0.05$ stability margin. If $\gamma = 0.20$ tames it, the hypothesis is confirmed. If it doesn't, the hypothesis is wrong.

This is analogous to the d=384 Phase 2→3 transition, which changed peak LR by 2× (from $3 \times 10^{-4}$ to $1.5 \times 10^{-4}$) — a similarly dramatic dynamics change — and the model adapted within ~500 steps. Switching $\gamma$ may be equally recoverable.

| Metric | Expected at $\gamma = 0.05$ (Phase 2) | Expected at $\gamma = 0.20$ (cross-gamma) |
|--------|:--:|:--:|
| Gradient profile (steps 0–10K) | Catastrophic spikes within 5–10K steps (Hessian already above $\gamma = 0.05$ margin) | Clean (if hypothesis correct) |
| Warm-restart regression depth | ~15% (based on d=384) | Possibly deeper (regime change) |
| Recovery time to Phase 1 best PPL | ~25K steps (based on d=384) | ? |
| `max_grad_in_window` (from improved logging) | Expect > 10,000 | Expect < 1,000 |

**Decision gate:**

| Outcome | Recommended path |
|---------|-----------------|
| Cross-gamma works (PPL recovering, clean gradients) | **Continue this run as Phase 2** — no Phase 1 rerun needed (Path A) |
| Cross-gamma PPL stalls (regime mismatch) | **Full Phase 1 rerun at $\gamma = 0.20$**, then Phases 2+3 (Path B) |
| Cross-gamma still spiky (hypothesis wrong) | **Continue with $\gamma = 0.05$ Phase 2**, accept instabilities (Path C) |

#### Step 3: Commit to full Phase 2 (140K remaining steps)

Based on Step 2 results, commit to one of three paths:

| Path | Scenario | Total H100 time (from now) | Expected outcome |
|:----:|----------|:----------------:|-----------------|
| **A** | Cross-gamma works | 18h (finish P1) + 10h (test) + 130h (P2 remainder) = **158h** | Best efficiency — preserves all Phase 1 compute |
| **B** | Regime mismatch | 18h + 10h + 93h (P1 rerun at $\gamma = 0.20$) + 140h (P2) = **261h** | Clean foundation, higher cost |
| **C** | Hypothesis wrong | 18h + 10h + 140h (P2 at $\gamma = 0.05$) = **168h** | Accept instabilities |

**Path A is the most attractive:** a single 10-hour experiment either preserves all Phase 1 compute and stabilises Phases 2+3, or fails cheaply.

#### Step 4: d=1024 validation (contingent on Step 2 success)

If $\gamma = 0.20$ stabilises d=768, extend the validation to d=1024:

1. **10K steps at d=1024, $\gamma = 0.20$** — the $\gamma = 0.05$ run had catastrophic spikes by step 4K, so 10K steps is a decisive test.
2. **10K steps at d=1024, $\gamma = 0.25$** — given the earlier cascade onset at d=1024 (step ≈4K vs ≈50K at d=768), a higher $\gamma$ may be needed. $\gamma = 0.25$ provides a 5× stability margin increase and is closest to d=384's proven-stable regime.

| Metric | $\gamma = 0.05$ (from prior run) | $\gamma = 0.20$ | $\gamma = 0.25$ |
|--------|:--:|:--:|:--:|
| Cascade onset | Step ~4K | ? (expect > 10K if hypothesis correct) | ? (expect > 10K) |
| Max gradient (0–10K) | 63,949 | ? | ? |
| PPL at 10K | ~260 (stalled) | ? | ? |

If either value produces clean training, commit to a full 100K-step Phase 1 at d=1024 (~350M params, directly comparable to GPT-2 Medium). This would transform the d=1024 narrative from "unstable, cannot train" to "stability resolved via damping hypothesis."

**Cost:** ≈20 hours for both 10K-step tests. If successful, a full d=1024 Phase 1 would cost ≈130–160 hours on 1×H100 (slower per step due to larger model).

#### Future scales: Stability-aware gamma sweep protocol

For d=2048 and beyond, replace the current 3K-step PPL-only gamma sweep with a **20K–30K-step stability-aware sweep** that monitors both PPL *and* gradient norm statistics. The selection criterion becomes:

$$\gamma^{\ast} = \arg\min_\gamma \text{PPL}_{20K} \quad \text{subject to} \quad \max_{t \leq 20K} \lVert g_t \rVert \lt \tau_{\text{spike}}$$

where $\tau_{\text{spike}}$ is a spike severity threshold (e.g., 10,000). This trades sweep cost (7× longer per candidate) for much higher confidence that the selected $\gamma$ will remain stable through full training.

#### Summary timeline

Assuming a single LambdaLabs 1×H100 instance:

```
Day 1       (18h):  Finish d=768 Phase 1 at γ=0.05 → bank checkpoint
Day 2       (10h):  Step 2 — 10K steps cross-gamma Phase 2 test (γ=0.20 from γ=0.05 ckpt)
Day 2              Decision gate: commit to Path A, B, or C
Day 2–3     (20h):  Step 4 — d=1024 validation (γ=0.20 and γ=0.25, 10K each)
Day 3+             Full Phase 2 (d=768) and/or Phase 1 (d=1024) at validated γ
```

**Total validation cost before any full-run commitment: ≈28 hours (≈1 day).** This resolves the damping hypothesis at the most informative model state (post-Phase 1 checkpoint where $\gamma = 0.05$ was already catastrophically unstable), at minimal compute cost.

### 26.8 Implications for the mitigation tier table

The damping hypothesis adds a new mitigation tier — potentially more practical than the CfC propagator (§24) because it requires zero code changes:

| Tier | Strategy | Mechanism | Long-term | Status |
|:----:|----------|-----------|:---------:|:------:|
| 1 | Per-group clip + force clamp | Truncate spike direction | Degrades | Implemented |
| 2 | Reduce $L$ | Shorten cascade chain | Delays onset | Applied |
| 3 | CfC propagator | Remove `create_graph` chain | **Root-cause fix** | Not implemented |
| **4** | **Increase $\gamma$ (overdamped regime)** | **Raise stability margin** | **May prevent onset entirely** | **Proposed** |
| — | Reduce LR | Slow Hessian eigenvalue growth | Delays onset | Being tested |

**Tier 4 is uniquely attractive** because:
- It is a single hyperparameter change (no code modification)
- It has a clear mechanistic justification (exponential perturbation damping)
- It can be validated cheaply (10K-step comparison run)
- It may render Tiers 1–2 unnecessary if the stability margin is large enough
- It works within the existing Velocity-Verlet framework, unlike the CfC propagator which requires a new integrator

The main risk is that overdamped dynamics ($\gamma \gg \gamma_{\text{geodesic}}$) sacrifice some modelling capacity — the ballistic regime captures long-range token interactions that the overdamped regime may miss. This would manifest as a higher *floor* PPL even with unlimited training steps. However, d=384's excellent PPL results at $\gamma = 0.30$ (well into the overdamped regime, far from the geodesic-optimal $\gamma = 0.05$) demonstrate that the overdamped regime retains substantial modelling power at least up to d=384.

### 26.9 Open questions

1. **Is there a sweet spot?** Can we find a $\gamma$ at d=768 that is stable *and* retains some ballistic character — e.g., $\gamma = 0.15$ — or does stability require full overdamping?

2. **Does the stability threshold shift with training?** The Hessian eigenvalues grow throughout training. A $\gamma$ that is stable at step 50K may become unstable at step 200K. If so, an **annealing schedule** for $\gamma$ (increasing $\gamma$ during training) might be needed.

3. **Interaction with LR:** Both $\gamma$ and LR affect the stability margin. The current d=768 run uses LR $= 2 \times 10^{-4}$ (vs d=384's $3 \times 10^{-4}$). A combined ($\gamma$, LR) stability sweep would map the full stable region.

4. **Does the CfC propagator become unnecessary?** If overdamped $\gamma$ stabilises training at all scales, the CfC propagator may be an overengineered solution. However, the CfC propagator also has efficiency benefits (no `create_graph` memory overhead), so it may still be desirable for memory-constrained scale-ups.

---

## 27. Empirical Depth-Code Growth: Boundary Layers Dominate in Both Integrators

### 27.1 Context

The first CfC/BAOAB production run (gamma=0.10, d=384, $L=16$, `aniso_dcvt5x8`,
same architecture as the Verlet runs in §23-§26) hit its first grad-clip spike
burst at step 6,297 (pre-clip total grad up to 3,336.8 at step 6,676), and
val_ppl visibly worsened over the corresponding eval window (176.88 -> 207.11
between steps 6,000 and 6,500) before the watchdog's slow EMA (`alpha=0.05`,
`patience=200`) caught it. The top contributing groups at every spike were
`depth_code`, `creation_gate`, `E`/`P` (token/positional embeddings),
`register`, and `reverse_ch` -- i.e. this is the same embedding-spike /
force-cascade family documented in §18-§20 and §23, now reappearing under
CfC/BAOAB despite §24 having removed the second-order force-cascade term the
Verlet integrator was suffering from. Two fixes were applied in response
(tightening `depth_code`'s per-group clip override from 0.5 to 0.25, and
adding a `GRAD_NORM_HARD_TRIGGER=500.0` fast path that reloads immediately on
any single-step raw grad norm above threshold, independent of the slow EMA);
both are implemented in
`colab_fock_cfc_baoab_aniso_gaussian_openwebtext_d384.ipynb`.

`depth_code`'s prominence in every spike prompted a direct empirical check of
what this parameter — the per-layer additive shift $e_g$ of
[`Structured_VTheta_Design_and_Theory.md`](./Structured_VTheta_Design_and_Theory.md)
§14.3 — actually does once trained, using six early CfC/BAOAB checkpoints
(steps 500-5,500, i.e. before the burst) and a mid-run Verlet checkpoint
(steps 9,000-15,000) from the sibling gamma=0.10 experiment.

### 27.2 Finding: boundary layers ($g=0$, $g=L-1$) move; middle layers do not

The full per-layer trajectories, numbers, and the tie-back to the §14.4
conservativity proof are in
[`Structured_VTheta_Design_and_Theory.md`](./Structured_VTheta_Design_and_Theory.md)
§14.7. In summary, for both integrators layers $g=1,\dots,L-2$ stay
statistically at their random-init norm throughout the windows checked, while
the layer(s) adjacent to a boundary move in a clearly structured, non-random
way:

- **Verlet:** only $g=0$ deviates from init (shrinks to ≈0.66x at step 9,000,
  recovers to ≈0.89x by step 15,000).
- **CfC/BAOAB:** $g=15$ ($=L-1$) grows explosively in the first 2,500 steps
  (1.14x -> 3.41x init) then plateaus (3.41x -> 3.65x over the next 3,000
  steps); $g=1$ climbs more slowly and has not plateaued by step 5,500
  (1.03x -> 1.91x); $g=0$ peaks at 1.72x (step 3,500) then relaxes back to
  1.17x (step 5,500) -- non-monotonic, same as the Verlet run's $g=0$, but
  with the opposite sign.

Critically, $g=15$'s growth is already flat by step 2,500-3,500 -- more than
2,500 steps before the step-6,297 spike burst -- and val_ppl improves
smoothly and monotonically the entire time (1369 -> 171). So an elevated
$\lVert e_{L-1}\rVert$ is not, by itself, sufficient to trigger the cascade;
it establishes a *precondition* (per §14.7's argument, a large per-layer
shift can move that layer's read of the shared well bank into a
higher-curvature region than the bank's precision matrices were tuned for),
consistent with §26's curvature-based reversal of the naive damping
argument, but something else — plausibly continued drift in $g=1$, or in one
of the other implicated groups (`creation_gate`, `destruction_gate`,
`register`, `reverse_ch`) — has to change further before the burst itself
fires. No checkpoint spanning the burst is available yet to settle this.

### 27.3 Relationship to §23-§26

This finding sits alongside, rather than replacing, the mitigation ladder of
§23 and the damping/curvature discussion of §26: it identifies which
*parameter group* inside the shared V_theta bank is the structural conduit
for the boundary-layer sensitivity that both integrators exhibit, and gives
a concrete, checkpoint-verifiable quantity (`depth_code` per-layer norm) to
track alongside the Weyl-bound stiffness audit (SCAF `StiffnessProbe`, see
the `semsimula-scaf` docs) when diagnosing future bursts. The pending $L=8$
depth-probe run is the natural next test of whether the same boundary
pattern reappears at $g=7$ ($=L-1$ for $L=8$) on a similar step-relative
timeline, which would support an $L$-independent boundary effect, versus a
markedly different timeline or magnitude, which would point to a
depth-dependent mechanism instead.

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
[`model_aniso_gaussian_vtheta.py`](../notebooks/conservative_arch/parf/model_aniso_gaussian_vtheta.py),
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

where $g_k = w_k \exp(-\tfrac{1}{2} \delta_k^T P_k \delta_k) > 0$ and $\delta_k = h - \mu_k$.
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

*(spectral norm $\sigma_{\max}$ of `B_proj.weight` per xi-channel; higher
channel indices correspond to longer context windows in the multi-xi
bank.)*

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

---

## 30. Concrete Sketch: The Low-Rank Exponential Substep

This is the **as-built** description of `integrator='baoab_cfc_lowrank'`
(`cfc_baoab.lowrank_modes` / `lowrank_cfc_substep`,
`model_parf_multixi._layer_step_langevin`). It follows the code's
**aggregated-spring** structure and the PSD/impulse corrections of §29.2.

Freezing the coefficients $g_k, \mu_k, P_k$ at the current $h$ (the same
frozen-coefficient model `harmonic_terms` uses), the aggregate local force is
$f(h') = s - H h'$ with $H = D_a + L$, $D_a = \mathrm{diag}(\sum_k g_k a_k)$
(clamped diagonal precision) and $L = G G^\top$ the PSD low-rank part. Only
$L$ is stiff (its $\sigma_{\max}$ is unbounded, §28.2b); $D_a$ is bounded by
`precision_max`. So $L$ is the part to integrate exactly.

### 30.1 Why the exactly-integrated operator must be PSD

This is the correctness point the earlier sketch glossed, and the reason the
split is written the specific way it is. The frozen aggregate Hessian is

$$H = \sum_k g_k P_k = \underbrace{\mathrm{diag}\Big(\textstyle\sum_k g_k a_k\Big)}_{\text{from the diagonal precision } a_k} + \underbrace{\sum_k g_k B_k B_k^\top}_{= G G^\top},\qquad g_k \ge 0,$$

and $H$ is symmetric positive semidefinite (PSD): a nonnegative-weighted sum
of the PSD terms $P_k = \mathrm{diag}(a_k) + B_k B_k^\top$. So *every mode of
$H$ itself is a genuine oscillator* — the question is only how to **split**
$H$ into pieces cheap enough to integrate exactly.

**The tempting split, and why it is wrong.** `harmonic_terms` already
integrates $\mathrm{diag}(H)$ exactly (per dimension) and demotes the rest to
the kick, so the obvious next move is "also rotate the leftover off-diagonal."
That leftover is

$$H_{\text{off}} = H - \mathrm{diag}(H) = G G^\top - \mathrm{diag}(G G^\top),$$

where the pure-$a$ diagonal has cancelled. One line of trace algebra kills the
idea. A matrix and its diagonal have equal trace, so

$$\mathrm{tr}(H_{\text{off}}) = \mathrm{tr}(G G^\top) - \mathrm{tr}\big(\mathrm{diag}(G G^\top)\big) = 0.$$

A nonzero symmetric matrix whose eigenvalues sum to zero must have **both a
strictly positive and a strictly negative eigenvalue** — so $H_{\text{off}}$
is **indefinite** whenever the coupling is nonzero (i.e. whenever the wells are
genuinely anisotropic). A minimal $2 \times 2$ witness with $G = (1, 1)^\top$:

$$L = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix},\ \mathrm{spec}(L) = \{2, 0\} \succeq 0 \quad\text{(PSD)};\qquad L_{\text{off}} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix},\ \mathrm{spec}(L_{\text{off}}) = \{+1, -1\}\quad\text{(indefinite)}.$$

**Why an indefinite mode defeats the whole point.** A mode with curvature
(eigenvalue) $\lambda$ obeys $m \ddot\eta = -\lambda \eta$:

| mode curvature | equation of motion | exact solution | behaviour |
| --- | --- | --- | --- |
| $\lambda \gt 0$ | $m\ddot\eta = -\lambda \eta$ | $\eta_0 \cos(\omega t) + \dots,\ \omega = \sqrt{\lambda/m}$ | bounded rotation |
| $\lambda \lt 0$ | $m\ddot\eta = \lvert\lambda\rvert \eta$ | $\eta_0 \cosh\big(t\sqrt{\lvert\lambda\rvert/m}\big) + \dots$ | exponential blow-up |

So "exactly rotating" the modes of $H_{\text{off}}$ would *exactly integrate a
repeller* on its negative eigenvalues — amplifying, not taming, precisely the
failure CfC was built to avoid. In code the symptom is immediate:
`omega = (kappa / m).sqrt()` with $\kappa \lt 0$ is a `NaN`, or, if floored to
zero, silently mis-integrates a repeller as free drift.

**The principle.** PSD-ness of the *sum* $H$ does **not** imply stability of a
*split*. Once an indefinite factor is carved off and integrated on its own,
the composition inherits that factor's hyperbolic growth. Integrating $H$
*whole* would be unconditionally stable (all its modes are $\ge 0$), but that
needs a full $d \times d$ eigendecomposition, $O(d^3)$ per token — infeasible.
The low-rank structure only helps on $L = G G^\top$ (rank $\le Kr$, so the
Gram eigendecomposition is $O((Kr)^3)$); a diagonal-plus-low-rank matrix has no
cheap exact matrix function. Hence we must split — and **a split is stable
only if every exactly-integrated factor is individually PSD.**

**The PSD split (what the code does).** Keep both factors PSD:

$$H = D' + L,\qquad D' = \mathrm{diag}\Big(\textstyle\sum_k g_k a_k\Big) \succeq 0,\qquad L = G G^\top \succeq 0.$$

This moves the **entire** $B_k B_k^\top$ — *including its own diagonal*
$\mathrm{diag}(G G^\top)$ — into $L$, and correspondingly removes that piece
from the diagonal channel, which now carries only $a_k$. That is exactly why
`harmonic_terms_lowrank` returns $k_{\text{diag}} = \sum_k g_k a_k$ (pure $a$),
**not** the $\sum_k g_k\big(a_k + \mathrm{rowsum}(B_k^2)\big)$ that the
diagonal-only `harmonic_terms` returns. The bookkeeping has two jobs and does
both at once:

- **No double-counting.** If the diagonal channel kept the full $\mathrm{diag}(H)$ *and* $L$ carried its own diagonal, $\mathrm{diag}(G G^\top)$ would be integrated twice. Giving that diagonal entirely to $L$ counts it exactly once.
- **No indefinite remainder.** Because the off-diagonal is never separated from its stabilising diagonal, no indefinite operator is ever formed; each exactly-integrated factor ($D'$ diagonal-nonnegative, $L$ a Gram) is PSD, so each sub-flow is a bounded rotation.

$D'$ is bounded (clamped by `precision_max`), so it is safe as the cheap
explicit/diagonal channel; $L$ is the stiff part, integrated exactly on its
$\le Kr$ modes. Both PSD, so the impulse composition (§29.2) has **no**
hyperbolic factor — the only residual stability concern is the mild resonance
effect discussed under "Stability" below, not blow-up.

### 30.2 The four steps, as implemented

**Step 1 — expose the low-rank subspace.** `harmonic_terms_lowrank()` stacks
the per-well factors, weighted by $\sqrt{g_k}$, into
$G = [\sqrt{g_1} B_1, \dots, \sqrt{g_K} B_K] \in \mathbb{R}^{d \times Kr}$ (all
xi-channels aggregated), so $L = G G^\top$. `lowrank_modes` eigendecomposes
the small Gram $G^\top G = W \Lambda W^\top$ ($Kr \times Kr$); the eigenvalues
$\lambda_j$ **are** the curvatures of $L$ and the reconstructed left vectors
$u_j = G w_j / \sqrt{\lambda_j}$ are its modes. (Gram-eig rather than a raw SVD
of $G$ is the numerically stable route; the geometry $U, \kappa$ is
**detached** — a frozen Jacobian — because a rank-deficient Gram has
degenerate near-zero eigenvalues whose `eigh` backward is singular. The
substep stays differentiable in $h, v$ and the frozen force.)

**Step 2 — per-mode frequency.** Because $L$ (not the full $H$) is what the
fast flow integrates, the mode stiffness is simply
$\kappa_j = \lambda_j$, $\omega_j = \sqrt{\lambda_j / m}$ — **no** Rayleigh
quotient of $H$ and **no** mixing of the diagonal into the mode curvature
(that mixing was the "residual coupling" caveat of the original sketch; the
PSD split removes it). `lowrank_max_modes` optionally keeps only the stiffest
$q$ modes; modes with $\lambda_j$ below a floor are made inert.

**Step 3 — the exact fast flow (A-substep).** `lowrank_cfc_substep` advances
$T + V_L$ over the substep: a **free drift on the whole state** plus, on each
mode, the exact undamped rotation the diagonal channel already uses,

$$\begin{pmatrix} \eta_j' \\ \zeta_j' \end{pmatrix} = \begin{pmatrix} \cos(\omega_j \Delta t) & \sin(\omega_j \Delta t)/\omega_j \\ -\omega_j \sin(\omega_j \Delta t) & \cos(\omega_j \Delta t) \end{pmatrix} \begin{pmatrix} \eta_j \\ \zeta_j \end{pmatrix},$$

with $\eta_j = u_j^\top h$, $\zeta_j = u_j^\top v$. The increments are written
back force-based (via the frozen force $f_L = s_L - L h$, $s_L = G (G^\top\mu)$)
so no fixed point $h_\ast = L^{-1} s_L$ or division by $L$ is formed —
identical to `cfc_substep`'s treatment of the diagonal. The complement of
$\mathrm{span}(U)$, where $L$ exerts no force, is left to the free drift.

**Step 4 — everything soft goes to the explicit kick.** The B-kick carries
$f_\theta + f_\phi - f_L(h_{\text{mid}})$: the full V_theta force **minus** the
frozen low-rank force the A-substep already integrates. What remains is the
clamped diagonal spring $D_a$, $V_\phi$, and the nonlinear V_theta residual
(the variation of $g_k, \mu_k, P_k$ with $h$) — all of bounded curvature, so
none is a blow-up channel. The total force field is preserved exactly (tested
to $O(\Delta t^3)$ agreement with plain BAOAB).

**Stability (impulse / RESPA, corrected).** The fast flow is a bounded
rotation on the modes for any $\omega_j$ — as a *standalone* map there is no
$\omega_j \Delta t \lt 2$ wall. But this is an **impulse / multiple-time-step**
composition (fast flow $\Vert$ soft kick), so it is *not* unconditionally
A-stable: it has narrow **resonance instabilities** at $\omega_j \Delta t
\approx k\pi$. Between resonances it is stable at any stiffness; the O-step
friction $e^{-\gamma \Delta t}$ and the nonlinear residual damp the resonances
in practice. §29.3's `precision_lr_max` keeps $\omega_j \Delta t$ from running
away, so the resonances stay shallow — the two mitigations are complementary
for exactly this reason. A unit test (`test_cfc_baoab.py`) confirms the scheme
survives a curvature that overflows the explicit step, at a non-resonant
$\omega \Delta t = 4.7$.

**Cost.** One eig of the $Kr \times Kr$ Gram matrix ($O((Kr)^3)$, with the
$O(d (Kr)^2)$ Gram formation dominating) plus a few rank-1 projections, per
token; with $Kr \approx 32$ (or $n_{\text{ctx}} Kr$ aggregated) and $d = 384$
this is small next to the forward pass. `lowrank_max_modes` bounds it further.

**Caveats.**

1. Freezing $g_k, \mu_k, P_k$ at the current $h$ makes this a local exponential (Rosenbrock-type) step: it integrates the frozen linearization exactly, and the leftover nonlinearity (Step 4's residual) stays explicit but non-stiff.
2. The frozen mode geometry ($U, \kappa$) is detached, so parameter gradients flow through the frozen *force* magnitude but not through the eigendecomposition. This is the standard exponential-integrator treatment and is why the `eigh`-degeneracy backward is never hit; the force field (and hence the loss) is still exact.
3. The fast flow is the A (drift) substep and the O (thermostat) and B (kick) are unchanged, so the BAOAB splitting structure — and its $T=0$ sampling properties — are preserved.

---

## 31. SCAF Phase 7b/7c Audit Plan for Tuning precision_lr_max (L=16, and now L=8)

**Status: §31.3 IMPLEMENTED (27 August 2026); §31.2/§31.4/§31.5 not yet
run.** This section captures the audit methodology for choosing a
`precision_lr_max` value on the live g0.1/d=384 `baoab_cfc` runs, ahead of
the `baoab_cfc_lowrank` + `precision_lr_max` vs. `baoab_cfc` baseline A/B
flagged as the next step in §29.7. It is independent of the depth-probe
track in §24.4's August update: that track asks whether shortening $L$
suppresses the burst; this track asks whether bounding $B_k$'s curvature
suppresses it *without* touching $L$.

**Scope broadened to L=8.** §31 was originally scoped to the live L=16
run only, treating the L=8 depth probe as a separate axis (§24.4). The
L=8 probe has since produced its own escalating hard-watchdog evidence
that the same mechanism is at work there too: a step-32,139 trigger
(pre-clip grad norm 701.1; top groups `E`/`P`=413.2, `creation_gate`=259.0,
`reverse_channel_scale`=243.3, `register`=200.9, `depth_code`=158.5) was
followed, only ~2,000 steps after reload, by a step-34,091 trigger nearly
an order of magnitude worse (5,864.9; `P`/`E`=3,106.9, `depth_code`=2,805.1,
`reverse_channel_scale`=2,560.5, `creation_gate`=2,354.4, and — notably
larger than the §27.1 "mid-pack" reference value of 502.0 — `V_theta`=1,073.0).
Both reloaded to the step-27,000 checkpoint (PPL 100.47), with no net
progress across the ~8,000 intervening steps. The escalating severity
(701 → 5,865 at the *same* reload point) is the qualitative signature
§28.6 predicts: `depth_code`/embedding perturbations get a second-order
amplification route through $B_k$'s still-growing curvature, so the same
proximate trigger produces a larger excursion the longer training
continues past it. §31.2's bracketing protocol therefore now has a
concrete L=8 candidate pair on hand: the step-27,000 best checkpoint
(healthy) against the `..._step32139_prereload.pt` /
`..._step34091_prereload.pt` snapshots (spike-regime) that the watchdog's
`_reload_best()` already wrote to Drive — no separate capture run needed.
A lighter-weight, no-`scaf`-dependency route to the same
`sigma_max(B_k)^2` percentiles is now also available directly in the
training notebook (`sigma_lr_report`, Cell 6b-2, added alongside
`stiffness_report`), for a quick in-Colab check against these exact
checkpoints ahead of, or instead of, the full offline SCAF audit below.

### 31.1 Goal

Pick a `precision_lr_max` value (the $\sigma^2$ budget on $B_k$'s
spectral norm enforced by `_bound_lowrank()` in
`model_aniso_gaussian_vtheta.py`, §29.3) from the *empirical*
$\sigma_{\max}(B_k)^2$ distribution actually reached on the g0.1/L=16
run, rather than guessing a number. Too tight a budget flattens the well
geometry and costs modelling capacity; too loose a budget doesn't touch
the runaway tail that caused the burst.

### 31.2 Checkpoints to bracket: healthy vs. spike-regime

A single checkpoint's stiffness distribution is not enough — the budget
should sit **between** the healthy bulk and the runaway tail, so both
ends need to be measured on the same run:

- **Healthy end.** A checkpoint from before the burst — e.g. the
  step-5,500/6,000 periodic or best checkpoint (val PPL 171–188, grad
  norm 0.8–1.4 in `training_log.jsonl`).
- **Spike-regime end.** A checkpoint from *during* the burst (steps
  6,297–6,676). The ordinary "best" and periodic-grid checkpoints are
  unreliable for this: `best_val_ppl` was regressing through that window
  (176.88 → 207.11), so no new "best" checkpoint was written there, and
  the periodic grid may not land inside a ~380-step window. The training
  loop's `_prereload` snapshots (`_reload_best()` in the notebook's Cell
  6, `tag_suffix='_prereload'`, capped by `PRERELOAD_SNAPSHOT_MAX_KEEP=5`)
  exist for exactly this reason — `GRAD_NORM_HARD_TRIGGER=500.0` fires
  unconditionally on any raw pre-clip grad norm above 500, and steps
  6,407 / 6,435 / 6,676 (pre-clip totals 1,816 / 2,402 / 3,337) all cross
  it — so the run's Drive `checkpoints/` folder should contain
  `..._step64xx_prereload.pt` / `..._step66xx_prereload.pt` files
  capturing the model state at (or immediately after) the worst moments
  of the burst. **Check for these first**, before falling back to the
  nearest periodic checkpoint.

Audit both ends with the same `StiffnessProbe` configuration so the
percentile ladders are directly comparable.

### 31.3 Required SCAF change: raw sigma_max(B_k)^2 percentiles

> **Status: IMPLEMENTED (27 August 2026).** All three items below are
> done, on the `stiffness_audit` branch of `semsimula-scaf`; the full
> suite (188 passed, 7 skipped, including the new test) is green.

`StiffnessProbe` (`semsimula-scaf/src/scaf/probes/stiffness.py`,
`stiffness_audit` branch) already computed what's needed internally, in
`weyl_upper_bound()`: `sigma_max_sq = torch.linalg.eigvalsh(gram)[..., -1]`
is the per-well, per-token $\sigma_{\max}(B_k)^2$. But it was only ever
folded into the aggregate Weyl bound
`k_weyl = sum_k g_k * (max_i a_k[i] + sigma_max_sq)` and reported as
`eig_median` / `eig_p90` / `eig_p99` / `eig_p999` / `eig_max` **after**
conversion to $\omega \Delta t$ — never as a raw, un-aggregated,
un-converted $\sigma^2$ value. `precision_lr_max` is exactly that raw
quantity, so the probe needed a small, additive change, now made:

1. `weyl_upper_bound()` takes a new `return_sigma_lr: bool = False`
   argument; when `True` it returns `(k_weyl, sigma_max_sq)` instead of
   just `k_weyl`, exposing the per-well, pre-aggregation
   $\sigma_{\max}(B_k)^2$ (shape `(..., K)`) alongside the existing
   aggregate bound, with the default-`False` call site unchanged.
   `StiffnessProbe.run()` now collects it into its own `sigma_lr_blocks`
   list alongside `eig_omega_dt_blocks`.
2. The `detail` dict now emits `sigma_lr_p50` / `sigma_lr_p90` /
   `sigma_lr_p99` / `sigma_lr_p999` / `sigma_lr_max` (reusing the existing
   `_quantiles()` helper) — the same percentile ladder already used for
   `omega_dt` and `eig_omega_dt`, just in raw $\sigma^2$ units instead of
   $\omega \Delta t$ units.
3. `tests/test_stiffness_probe.py` gained
   `test_return_sigma_lr_reproduces_planted_spectral_norm`, a synthetic
   two-well, rank-2 case with a diagonal Gram (so the top singular value
   is exact by inspection: $\sigma_{\max}^2 = 25$ and $9$ for the two
   wells) asserting `weyl_upper_bound(..., return_sigma_lr=True)`
   reproduces both planted values and leaves the aggregate `k_weyl`
   bit-for-bit unchanged from the non-`return_sigma_lr` call.

This was purely additive (new `detail` keys and an opt-in return-value
change only) — no existing probe output changed, so nothing downstream
(the OWT g0.1/g0.3 Phase 7b/7c reports already published on HF) needs
re-validation.

### 31.4 From percentiles to a precision_lr_max budget

Once both checkpoints report `sigma_lr_*`:

1. Confirm the qualitative story first: `sigma_lr_p99` / `sigma_lr_max`
   should be visibly larger on the spike-regime checkpoint than the
   healthy one, and the gap between `eig_p99` and `p99` (Weyl vs.
   diagonal-only $\omega \Delta t$) should be the dominant contributor to
   instability on the spike-regime side. If it isn't, $B_k$ growth isn't
   actually the driver of *this particular* burst and `precision_lr_max`
   is the wrong lever for it (the depth-cascade track of §24.4 would be
   the more relevant explanation).
2. Set the budget **above** the healthy checkpoint's `sigma_lr_p95`–`p99`
   (not its median — clamping there would flatten the well geometry
   every well relies on) and **below** the spike-regime checkpoint's
   `sigma_lr_p99` / `max`.
3. Remember the cap is on $\lVert B_k \rVert_F^2 \ge \sigma_{\max}(B_k)^2$
   (conservative by up to a factor `rank` — 4 for this run's
   `ANISO_RANK`), so the *effective* spectral cap achieved is somewhat
   tighter than the nominal `precision_lr_max` number; bias the choice
   slightly upward from step 2's interval to compensate.

### 31.5 The A/B run

With a `precision_lr_max` value in hand, run the comparison already
flagged as the next step in §29.7's status line: `baoab_cfc_lowrank` +
tuned `precision_lr_max` vs. the current `baoab_cfc` baseline, at
g0.1/d=384, watching whether the burst signature shrinks or disappears —
the 6,297–6,676-style E/P/`depth_code` bursts on **L=16**, or the
27,000-onward 32,139/34,091-style bursts on **L=8** (§31 preamble; no
longer treated as a separate, independently-tracked axis from §24.4's
depth-probe track now that it has produced its own escalating
hard-watchdog evidence of the same mechanism).

### 31.6 Status

**§31.3 done; L=8's §31.2/§31.4 done (§31.7); §31.5 and L=16's §31.2/§31.4
not yet started.** The SCAF probe change (`sigma_lr_*` percentiles, §31.3)
landed 27 August 2026, and an equivalent no-`scaf`-dependency diagnostic
(`sigma_lr_report`) was added directly to the training notebook (Cell
6b-2) for a quicker in-Colab check. Two candidate runs were in scope to
bracket:

- **L=16** (the original scope): still needs the g0.1/L=16 Drive folder
  inspected for `_prereload` snapshots in the 6,297–6,676 range.
- **L=8** (added this update, §31 preamble): the bracket pair was already
  on hand and needed no separate capture run —
  `..._best.pt` at step 27,000 (healthy) against
  `..._step32139_prereload.pt` / `..._step34091_prereload.pt`
  (spike-regime), all on the live `..._L8probe_..._g0.1_baoab_cfc` Drive
  folder. **This bracket has now been run through §31.4's logic — see
  §31.7 — and the answer is "no valid budget window; leave
  `PRECISION_LR_MAX = None`."**

§31.5's A/B (`baoab_cfc_lowrank` + a tuned budget, vs. the `baoab_cfc`
baseline) remains available as an opt-in notebook config change
(`INTEGRATOR`, `PRECISION_LR_MAX` in Cell 0) once/if the L=16 bracket
produces a workable window, or as a diagnostic falsification run on L=8
using the non-binding value discussed in §31.7.

### 31.7 L=8 budget selection: the recipe's own preconditions fail — leave `PRECISION_LR_MAX = None`

Running the three checkpoints (§33.1) through §31.4's two-step recipe:

**Step 1 (qualitative gate) fails.** §31.4 step 1 requires `sigma_lr_p99` /
`sigma_lr_max` to be *visibly* larger on the spike-regime checkpoints than
on the healthy one. They are not: the largest gap is +24% (p50, and only
at step 32,139) and the *p99/max* gap that step 2 actually keys off is a
near-flat +1.0% to +14.5% (§33.1's table). Per the recipe's own written
rule, this means "$B_k$ growth isn't actually the driver of *this
particular* burst and `precision_lr_max` is the wrong lever for it" — i.e.
the recipe self-terminates here, matching §33's independent conclusion via
the per-group grad log and the depth-cascade mechanism.

**Step 2 (the interval) is vacuous even if forced.** Attempting it anyway
for completeness: the lower bound ("above healthy's p95–p99") is
$\approx 1{,}050$–$1{,}100$; the upper bound ("below spike-regime's
p99/max") should be the *tighter* of the two spike checkpoints, and here
is the decisive number — **the healthy checkpoint's own `max` (6,364.81)
is essentially identical to spike_34091's `max` (6,427.16, +1.0%) and only
14.5% below spike_32139's `max`.** The long tail of $\sigma_{\max}(B_k)^2$
is not a spike-exclusive phenomenon; it is already present at the best
checkpoint the run has ever produced (PPL 100.47). Any budget tight enough
to bind on the spike tail (roughly below 6,400–7,300) necessarily clips
the healthy checkpoint's own tail too — there is no interval that
isolates "runaway" from "healthy," because on this evidence there isn't
one.

**Recommendation for the live L=8 run: leave `PRECISION_LR_MAX = None`.**
Picking any value in the only mathematically available window
(roughly 1,100–6,400, and recall §31.4 step 3: the true budget should be
biased *upward* from a naive read of that window because the Frobenius
cap is conservative by up to `rank`$=4$) would flatten well geometry that
the healthy checkpoint is actively using — a real cost in modelling
capacity — for a mechanism §33.2 has already shown is not the bottleneck.
This is consistent with, and sharpens, §33.2's "sound hygiene but not the
lever" framing: it isn't just that `precision_lr_max` is a weak lever
here, it's that this particular run's data contains no value that would
act as a *targeted* one.

**If a falsification A/B (§31.5) is still wanted**, run it with the
understanding that any chosen value is a blunt, non-targeted probe rather
than a tuned fix — e.g. `PRECISION_LR_MAX` $\approx 2{,}500$ (comfortably
above every checkpoint's `p99`, so the ordinary bulk is untouched, while
still meaningfully compressing the `p99.9`/`max` tail on *all three*
checkpoints, healthy included) — and expect, per §33, little to no change
in the burst rate. A clean negative result there would be further
confirmation, not a surprise.

---

## 32. L=8 `baoab_cfc` Baseline: Extended Trajectory (steps 27,000–39,867) and the Decision to Switch Mid-Run

This section records the L=8 probe's behaviour for ~13,000 further steps
past the step-27,000 best (extending §31's preamble, which covered only
the two hard-trigger events themselves), and the resulting decision to
switch this run to `baoab_cfc_lowrank` before it reaches `TOTAL_STEPS`
rather than let the plain-`baoab_cfc` arm run to completion.

### 32.1 The extended trajectory is a noisy plateau, not a decreasing trend

Across steps 30,000–39,867 (all still within the WSD **stable** phase,
LR pinned at 3e-4 until step 65,000):

- **No new best.** Every eval PPL from step 30,500 through 39,500 landed
  in a 101–113 band; the running best has stayed frozen at 100.47 (step
  27,000) the entire time.
- **Spikes did not slow down.** 33 pre-clip spikes >100 occurred over
  ~8,600 steps (steps 31,272–39,867) — one every ~260 steps on average —
  and the rate is essentially unchanged before vs. after the second
  hard-trigger (18 of the 33 in the 5,776 steps following step 34,091,
  max 265.4). Two of the 33 (701.1 at step 32,139; 5,864.9 at step
  34,091) were severe enough to trigger `GRAD_NORM_HARD_TRIGGER` and
  force a full reload to the step-27,000 checkpoint, each discarding
  several thousand steps of intervening optimizer state.
- **A faint downward drift is visible underneath the noise, though.**
  The five evals immediately after the second reload (34,500–36,500)
  average 107.71; the five most recent at the time of this note
  (37,500–39,500) average 103.48, briefly touching 101.46 at step 38,000
  before another spike pushed it back up. So this is not simply frozen
  noise around a fixed level — there is slow, bumpy net progress — but
  it has not yet recovered past the pre-spike best, ~13,000 steps later.

**Reading.** This is consistent with, and adds a second, independent
line of evidence for, §28.6's amplification mechanism and §28.2b's
"$B_k$ keeps growing, unbounded, no plateau" measurement: nothing in
plain `baoab_cfc` bounds the off-diagonal channel, so there is no
structural reason for the spike rate to decay on its own, and it hasn't,
over an interval nearly 2.5x longer than the one analysed in §31's
preamble.

### 32.2 Will it converge below 100 PPL by step 100,000?

Two effects pull in opposite directions, and neither is resolved by the
data in hand:

- **Against:** the driving mechanism (unbounded $B_k$ curvature) has no
  reason to weaken with further training under plain `baoab_cfc` — if
  anything §28.2b's measurements suggest it should compound, not fade.
- **For:** the run has not yet reached the WSD **decay** phase
  (65,000→100,000, LR 3e-4 → floor 1.5e-5). WSD-style schedules
  typically produce much of their net improvement during this anneal —
  a shrinking LR directly shrinks the magnitude of *every* gradient
  step, spike or not — so both the instability and the underlying loss
  could improve substantially once decay starts, independent of whether
  the off-diagonal channel itself is ever fixed.

Net: a soft landing somewhere near, but probably not dramatically below,
100 PPL by step 100,000 is plausible; a clean, convincing win is not
supported by the trend so far. Not enough of the stable-phase budget
has gone to net progress (vs. fighting and recovering from spikes) to
be confident either way from trend extrapolation alone.

### 32.3 Decision: switch to `baoab_cfc_lowrank` now rather than complete this arm to 100,000

**Decided 27 August 2026, given single-GPU compute (no parallel session
available for a simultaneous baseline).** Reasoning:

1. §32.1's ~13,000 steps already constitute a solid, well-characterized
   "before" picture (frozen best, ~260-step average spike interval, two
   forced reloads) — completing the remaining ~60,000 steps under the
   same integrator mostly re-confirms this rather than adding new
   information, since the driving mechanism is not expected to resolve
   itself (§32.1's reading).
2. The comparison that actually matters — whether `baoab_cfc_lowrank`
   changes the trajectory — only requires resuming from the *same*
   step-27,000 checkpoint under the new integrator; it does not require
   first finishing the plain-`baoab_cfc` arm to `TOTAL_STEPS`.
3. On a single GPU, finishing a run already showing this signature is a
   worse use of the one available compute slot than testing the
   implemented-and-ready fix (§29.2/§29.3, §29.7) sooner.
4. Nothing here is destroyed: the plain-`baoab_cfc`-to-100,000 baseline
   can still be run later from the same step-27,000 checkpoint whenever
   a second GPU/session is available, using
   `RESUME_VARIANT_TAG_OVERRIDE` (Cell 1b) to redirect back to this run's
   own checkpoint folder.

**Early read to watch for after switching:** compare the spike
frequency/magnitude over the first 5,000–10,000 steps past 27,000 under
`baoab_cfc_lowrank` against §32.1's ~260-step/spike, ≤265 (soft) /
up to 5,865 (hard) baseline from the same step range under plain
`baoab_cfc`. A visibly lower spike rate and/or the absence of a new
hard-trigger over a comparable window would be the first concrete
evidence the fix is doing what §29.2's theory predicts, ahead of any
`precision_lr_max` tuning (§31).

---

## 33. The Bracketing Result Is Modest and Non-Escalating: $B_k$ Is Not the Primary Driver, and a Root-Cause Workflow for the Non-$V_\theta$ Spikes

§31.4 step 1 wrote down an explicit escape hatch before any numbers were in
hand: if the raw low-rank curvature $\sigma_{\max}(B_k)^2$ does **not** grow
visibly from a healthy checkpoint to a spike-regime one, then "$B_k$ growth
isn't actually the driver of *this particular* burst and `precision_lr_max`
is the wrong lever for it." This section records that the L=8 bracket pair
(§31.6) was measured on 28 August 2026 — and the escape hatch fired.

### 33.1 The measurement: healthy vs. both spike-regime snapshots are elevated by +1% to +24%, non-monotonically

All three checkpoints from §31.6 were run through the Cell 6b-2
`sigma_lr_report` diagnostic (the dependency-free notebook mirror of SCAF's
`sigma_lr_*` percentiles, §31.3), each on the same fixed seed-0 probe batch,
pooling 1,310,720 samples over layers, wells, and xi-channels:

| percentile | healthy (step 27,000 best, PPL 100.47) | spike-regime (step 32,139 prereload, 1st hard trigger) | delta vs. healthy | spike-regime (step 34,091 prereload, 2nd hard trigger) | delta vs. healthy |
|:---|---:|---:|---:|---:|---:|
| p50 | 282.11 | 350.07 | +24.1% | 305.59 | +8.3% |
| p90 | 663.61 | 758.32 | +14.3% | 700.53 | +5.6% |
| p99 | 1047.00 | 1145.95 | +9.5% | 1082.80 | +3.4% |
| p99.9 | 2322.33 | 2514.92 | +8.3% | 2499.07 | +7.6% |
| max | 6364.81 | 7285.88 | +14.5% | 6427.16 | +1.0% |

![Grouped bar chart of sigma_max(B_k)^2 percentiles for the healthy step-27000 checkpoint versus both spike-regime prereload snapshots (step 32139 and step 34091), on a log y-axis, showing both spike checkpoints modestly and non-monotonically elevated above healthy, annotated with the conclusion that B_k is at most a weak correlate, not the driver, of these bursts](images/scaf_spike_diag_sigma_lr_bracket_result.png)

With the second bracket point in hand, the picture is more nuanced than a
flat null result, but the qualitative conclusion is unchanged. Both
spike-regime snapshots sit consistently *above* healthy at every percentile
(ten comparisons, ten positive deltas) — this is a real, repeatable effect,
not measurement noise straddling zero. But two features argue against $B_k$
being the driver of the bursts rather than a weak correlate of them:

- **The magnitude is far too small.** The largest deltas (p50 +24%, max
  +14–15%) correspond to at most a $\sqrt{1.24}\approx 1.11\times$ increase
  in the well's own frequency $\omega\propto\sqrt{\kappa}$ — nowhere near
  the scale of the recorded pre-clip grad-norms at these two hard triggers
  (701.1 at step 32,139; 5,864.9 at step 34,091; §32.1), both an order of
  magnitude or more above the `GRAD_NORM_HARD_TRIGGER=500.0` threshold and
  far above the sub-100 norms typical of untroubled steps. A curvature
  effect of at most $1.11\times$ cannot by itself produce spikes of that
  size.
- **It does not escalate monotonically.** Step 32,139 fired *first* and is
  the *more* elevated of the two snapshots at four of five percentiles
  (p50, p90, p99, max); step 34,091, which fired ~2,000 steps later, is
  closer to healthy. If $B_k$ were progressively drifting toward the
  crisis, later triggers should show more elevation than earlier ones, not
  less. The pattern is consistent instead with each hard trigger being an
  **independent excursion** from the same step-27,000 reload point — two
  separate draws of "how large $B_k$'s bulk happens to be by the time some
  other mechanism trips the watchdog," not a single escalating trend.

Both snapshots are finite; the `PPL=nan` recorded in each prereload
snapshot's metadata (an expected artefact of a snapshot taken at the instant
the loss went non-finite, not a fresh validation pass) did not contaminate
the $B_k$ measurement.

**Caveat on the probe batch.** `sigma_lr_report` evaluates $B_k$ on a fixed
generic batch, so it measures the *weights'* capacity to produce large
$\sigma_{\max}(B_k)^2$ on a typical input, not what happened on the specific
batch that tripped the watchdog at each crisis step ($B_k$ is
context-dependent, `context_components(xis)`). The modest-and-non-monotonic
result therefore rules out a *drifting baseline* — the parameters defining
$B_k$ did not migrate into a permanently, progressively stiffer regime going
into the crisis — but does not by itself rule out a transient, batch-specific
$B_k$ excursion on the offending step. The per-group gradient log at the
crisis step (§33.3) is what closes that remaining gap.

### 33.2 What this rules in and out

Three independent lines of evidence now point the same way — away from
$V_\theta$'s low-rank correction as the *primary driver* of the L=8 bursts:

1. **The bracket is elevated but not escalating, and far too small in
   magnitude (§33.1).** Both crises sit +1% to +24% above healthy,
   non-monotonically between the two triggers, versus the $>100\times$ scale
   of the observed grad-norm spikes. `precision_lr_max` caps a real but
   minor and non-escalating quantity.
2. **The per-group grad log points elsewhere.** The recorded spike groups for
   this architecture are `depth_code`, `E`, `P`, `creation_gate`, `register`,
   `reverse_channel_scale` — the **non-$V_\theta$** groups (§23.1's d=1024
   table shows exactly this cast of characters; the L=8 bursts are the same
   family). $V_\theta$'s own group is not the one that spikes.
3. **The mechanism is already documented as depth-driven, not curvature-driven
   for these groups** — the second-order force cascade of §23.2 and the
   boundary-layer depth-code growth of §27 are gradient-topology effects, not
   $B_k$-magnitude effects.

The practical conclusion: **`baoab_cfc_lowrank` + `precision_lr_max` remains
sound hygiene** (unbounded-by-construction curvature is a real hazard, the
+24% bulk shift and +14% tail shift at the crisis are genuine and worth
trimming, and it is good that the channel is now removable), but it is **not
the primary lever that will stop these specific bursts** — a $\lesssim
1.1\times$ frequency effect cannot explain a $>100\times$ gradient-norm
spike. The A/B run of §31.5 is still worth doing as a falsification check,
but §33.1 lowers its prior: expect the burst signature to largely persist,
because the bulk of the burst was never coming through $B_k$.

### 33.3 A root-cause workflow for the non-$V_\theta$ spikes

The bursts live in the non-$V_\theta$ groups, so the diagnostic has to target
those groups directly. The good news is that the training loop already emits
most of the raw material; the workflow below is ordered cheapest-first, and
each phase unlocks the next.

```mermaid
flowchart TD
    P0["Phase 0 - mine the existing training log grad spike events"]
    P1["Phase 1 (implemented) - capture the offending batch on a hard trigger"]
    P2["Phase 2 (implemented) - replay one isolated forward plus backward and instrument it"]
    P3["Phase 3 - productionize as a SCAF GradientSpikeProbe"]
    V{"which group layer op leads"}
    LV["V theta or B k implicated"]
    NV["non V theta group implicated"]
    RV["bracket sigma lr and tune precision lr max"]
    RN["targeted per group clip or op level fix"]

    P0 --> P1
    P1 --> P2
    P2 --> P3
    P2 --> V
    V -->|V theta| LV
    V -->|other group| NV
    LV --> RV
    NV --> RN
```

**Phase 0 — mine what is already logged. Status: DONE (28 August 2026),
findings below.** With `GRAD_SPIKE_DEBUG=True` the loop already writes an
`event: grad_spike` record (top-8 pre-clip groups + loss breakdown) to
`RESULTS_DIR/training_log.jsonl`, plus a watchdog record on every reload.
Mining the L=8 run's log (964 lines, steps 50–40,900, downloaded from
Drive) answers the first-order question — *which group leads each burst,
and is it always the same one* — with no new run, no code change, and no
Phase 1/2 capture needed:

| population | n | lead: `depth_code` | lead: E/P (tied) | other |
|:---|---:|---:|---:|---:|
| `grad_spike` events, pre-clip < 200 | 42 | 32 (76%) | 7 (17%) | 3 (7%) |
| `grad_spike` events, pre-clip ≥ 200 | 9 | 4 (44%) | 5 (56%) | 0 |
| `watchdog_hard_reload` events (32,139 / 34,091) | 2 | 0 (0%) | 2 (100%) | 0 |

(51 `grad_spike` events total, threshold 100, plus the 2 hard triggers;
zero EMA `watchdog_reload` events occurred in this run — every reload was
a hard trigger, consistent with §32.1.) Two findings fall out of this:

1. **`depth_code` dominates the small/frequent end** (76% lead-share
   below pre-clip 200, and in the top-3 breakdown 94% of the time across
   all 51 events) — it is the single most informative "leading indicator"
   group for the run's baseline spikiness.
2. **But at the severe end, leadership flips to `E`/`P`, and they are
   *always* tied for first at both hard triggers** — `E`=413.24/`P`=413.24
   at step 32,139, `P`=3106.93/`E`=3106.92 at step 34,091 (agreement to
   4 significant figures at the larger of the two). This is not a
   coincidence: `model_parf.py` defines
   `h0 = self.E(x) + self.P[position_offset:position_offset+T]` — the
   token embedding and the *additively combined* positional embedding.
   Both receive gradient as different linear reductions
   (scatter-by-token-id vs. sum-over-batch-at-each-position) of the exact
   same upstream tensor $\partial L/\partial h_0$, which is why their
   *norms* track so closely without the parameters being tied. (`E` is
   *also* weight-tied to the output logits projection,
   `logits = h_L @ self.E.weight.T`, so it has a second, direct gradient
   source P entirely lacks — the fact that E and P still agree this
   closely even so implies that direct output-side contribution is small
   next to the embedding-boundary one, i.e. **the crisis signal reaching
   `E`/`P` is overwhelmingly the one that has been backpropagated through
   all $L$ layers back to $h_0$, not a locally-large output-layer
   gradient.**)

**Reading the two findings together points at a cascade, not two
independent culprits.** The severity-band split (44%→56% for `depth_code`
vs. E/P moving from the <200 to the ≥200 band, then 0%→100% at the two
hard triggers) looks like a single mechanism crossing a threshold, not a
population of unrelated causes: `depth_code`'s own gradient (present at
every layer, per §33.4) is the visible signal while a disturbance is
still small and local; once it is large enough, the same disturbance
propagates backward through the $L=8$ stack and is amplified layer over
layer (the second-order force cascade of §23.2, the boundary-layer
`depth_code` growth of §27), arriving at the embedding boundary the
*largest* it will ever be simply because that is the far end of the
backward pass. This yields a sharp, falsifiable prediction for Phase 2,
once it has a capture to work with: **the per-layer $h$-gradient-norm
profile at a hard trigger should show growth from layer $L-1$ toward
layer $0$ (cascade amplification), not an isolated spike confined to one
interior layer** — and `depth_code`'s own per-layer grad norm should be
large at whichever layer the amplification *starts*, even if it is no
longer the largest single number by the time gradient reaches $E$/$P$.
This refines candidate hypothesis 4 in §33.4 below: it is not necessarily
that `depth_code` pushes $V_\theta$ into a stiff regime specifically, but
that it (or something correlated with it) seeds a disturbance the
existing $L$-layer cascade amplifies regardless of which downstream
mechanism carries it.

One subtlety must be corrected for at this stage: the scalar `grad_norm` the
watchdog thresholds on is `sqrt(sum of gn_k^2)` over groups **excluding**
`reverse_channel_scale` and `reverse_ch` (`WATCHDOG_EXCLUDE_GROUPS`). So if
the reverse channel is the true instigator, the aggregate under-reports it and
the hard trigger only fires once the disturbance bleeds into an *included*
group. Read the per-group breakdown (`_last_pg_norms`), never the single
aggregate, when attributing a burst. (`reverse_channel_scale` is in fact in
the top-8 breakdown of 50 of the 53 events mined above, including both hard
triggers — always behind `depth_code`/E/P, never masking a trigger outright
in this run, but consistently present.)

**Phase 1 — capture the offending batch. Status: IMPLEMENTED (28 August
2026), notebook `Cell 6` (config block + training loop).** The `_prereload`
snapshot preserves the *weights* at the crisis but not the *token batch*
that caused it — which is exactly why the §33.1 fixed-seed probe could not
reproduce the event, and it is *not* the pre-step weights either: by the
time `_reload_best` runs, `optim.step()` has already applied the crisis
update (§33.1's own bracket measured that post-update state, which is
useful for a different question — "how stiff did $B_k$ get" — but the
wrong state for replaying the forward+backward that *produced* the
gradient). The implemented fix does not touch `_reload_best` itself;
instead it moves the capture *earlier*, into the training loop, right
after `grad_norm` is computed but **before** the `optim.step()` call that
would mutate the weights:

- A new `CAPTURE_SPIKE_BATCH` / `SPIKEBATCH_SNAPSHOT_MAX_KEEP` config pair
  (mirroring `PRERELOAD_SNAPSHOT_MAX_KEEP`'s rotation policy) guards a new
  branch that fires exactly when `GRAD_NORM_HARD_TRIGGER` is about to.
- Two cheap, *unconditional* per-step additions feed it: the torch CPU/CUDA
  RNG state is captured right after `optim.zero_grad()` (the Langevin
  thermostat noise draw inside the forward pass consumes it, so replay
  needs it to reproduce that draw bit-for-bit), and each microbatch's raw
  `(xb, yb)` arrays are appended to a per-step list inside the
  `GRAD_ACCUM` loop (bypassing `get_batch`'s own RNG entirely — the saved
  arrays are replayed directly, so nothing needs to reproduce *which*
  windows `get_batch` would have drawn).
- Only in the rare case the hard trigger is about to fire does the
  (comparatively expensive) full `model.state_dict()` CPU clone happen,
  bundled with the batches/RNG state into a new
  `{CKPT_PREFIX}_step{step}_spikebatch.pt` sidecar — a new, additive
  artifact alongside (not replacing) `_prereload.pt`.
- **Scope decision: hard-trigger only.** The slow EMA-based watchdog path
  is a sustained drift across ~200 steps, not a single anomalous step, so
  there is no one well-defined "offending batch" for it — that path is
  left to Phase 0 log-mining (the `grad_spike` time series) rather than
  forced into a single-batch replay it doesn't fit. Both of the run's
  documented crises (701.1 at step 32,139; 5,864.9 at step 34,091, §32.1)
  are hard triggers, so this scope covers the evidence in hand.

**Phase 2 — op- and layer-resolved forensics on the replay. Status:
IMPLEMENTED (28 August 2026), new notebook `Cell 6d`
(`replay_spike_batch(step_tag)`).** Given a `_spikebatch.pt` bundle, the
cell loads the pinned pre-step weights and RNG state into the live model,
replays every captured microbatch through one isolated `backward()`, and
instruments it with:

- **per-parameter** grad norms (finer than per-group) to pinpoint the exact
  tensor — e.g. `creation_gate_qkv.log_tau` vs. `W_Q`, or a single
  `reverse_channel_scale[k]`;
- **per-layer** grad attribution via a tensor hook (`h_new.register_hook`)
  installed by temporarily wrapping whatever `_fock_layer_step` is
  currently bound to (composing with the aniso depth-routing patch rather
  than clobbering it) — there is no per-layer `nn.Module` to attach
  `register_full_backward_hook` to (the $L$ layers share most submodules
  through a plain Python loop, routed by `layer_idx`), so a tensor hook on
  $h$ at each layer boundary is the mechanism that actually applies here;
- **forward-activation extremes** at the numerically risky ops per group —
  the creation-gate softmax temperature
  $\tau = \exp(\log\tau).\mathrm{clamp}(10^{-4})$ and its cumulative-softmax
  weight (a direct hook on `forward_prefix`, which bypasses `__call__` and
  so cannot be reached by a standard forward hook), the reverse-channel
  `logit_scale.exp().clamp(max=100)` and the $Q_{\mathrm{force}}$ it
  injects (a standard forward hook on `reverse_ch`), the destruction-gate
  output, and the $V_\theta$ quadratic-form exponent and the depth-code-shifted
  xi norm (recomputed from each per-channel bank's own public
  `_components()` on the exact `(xi, h)` the forward call saw, via a
  `with_kwargs=True` forward hook — no source-file changes to
  `model_fock_parf_v2.py` / `model_fock_parf_multixi.py` /
  `model_aniso_gaussian_vtheta.py` were needed).
- **A fidelity check is built in:** the replayed matching-groups total is
  compared against the `pre_clip_grad_norm` recorded at capture time; a gap
  under a few percent is the signal that the RNG-state/batch capture
  actually reproduced the crisis bit-for-bit (or close to it), and a large
  gap is a warning that something about the replay is not exact before any
  attribution conclusions are drawn from it. The replay also reports the
  matching-groups total *and* the all-groups total side by side, so the
  reverse-channel contribution the watchdog aggregate is blind to (§33.3
  Phase 0's caveat) is visible directly rather than inferred.
- **Non-pollution:** the cell saves the live model's weights, every
  parameter's `.grad`, and the RNG state up front and restores all three
  in a `finally` block, so resuming the training loop afterward is safe —
  mirroring the SCAF `GradientSpikeProbe` design's save/zero/restore
  contract before that probe exists.
- **Caveat.** (a) and (b) above only touch already-used, documented
  training-loop machinery and are exact. (c)'s hooks were derived from
  reading the three model source files, not exercised against the live
  model yet; each is independently try/except-guarded, so a
  `[replay][WARN] could not instrument ...` line for any one of them
  narrows down what to fix without sinking the rest of the report.

![Schematic of the Fock-PARFLM forward integrator drawn top to bottom, with the embedding, creation gate, PARF dynamics, reverse channel, and destruction gate stages in the centre column, the second-order backward gradient flow arrow on the left, and a right-hand column annotating the numerically risky op and per-group clip ceiling for each spiking group, marking the reverse-channel groups as excluded from the watchdog aggregate](images/scaf_spike_diag_forward_backward_map.png)

This turns "group X is big" into a mechanism, e.g. "a near-degenerate
creation-gate temperature at layer $k$ produced a huge softmax Jacobian" or "a
large $Q_{\mathrm{force}}$ at layer $k$ hit the non-conservative reverse-channel
injection $(\Delta t^2/\mathfrak{m}).\tanh(s).Q_{\mathrm{force}}$."

**Phase 3 — productionize as a SCAF probe.** Once the manual harness proves
out, fold it into SCAF as a reusable, unit-tested `GradientSpikeProbe`. The
full requirements, interface, and design — including how it stays inside
SCAF's "probing must not pollute training `.grad`" invariant as the library's
first backward-aware probe — are in the SCAF design document
[`Gradient_Spike_Probe_Requirements_and_Design.md`](https://github.com/dimitarpg13/semsimula-scaf/blob/spike_diagnostic/docs/Gradient_Spike_Probe_Requirements_and_Design.md)
(branch `spike_diagnostic`).

### 33.4 Candidate hypotheses the workflow discriminates between

The instrumentation above is designed to separate these, which the raw
grad-norm log alone cannot:

- **Cascade amplification through the $L=8$ stack (favoured by §33.3
  Phase 0's findings).** `depth_code` (or something correlated with it)
  seeds a disturbance that is small and local while it is small, then
  gets amplified layer-over-layer on the way back to $h_0$ (the
  second-order force cascade of §23.2, the boundary-layer growth of
  §27) — consistent with `depth_code` leading 76% of smaller spikes while
  `E`/`P` (which only ever see the *cumulative* signal that has
  backpropagated through every layer, §33.3 Phase 0) take over 100% of
  the time at the two severe hard triggers. Predicts a per-layer
  $h$-gradient-norm profile that *grows* from layer $L-1$ toward layer
  $0$, not an isolated single-layer spike.
- **Sharp-softmax Jacobian** in the creation gate or reverse channel (small
  $\tau$ / saturated logits) — batch-triggered, transient.
- **Non-conservative reverse-channel kick** — a large $Q_{\mathrm{force}}$ on
  particular tokens injected via $\tanh(s).Q_{\mathrm{force}}$; note this
  channel is the one masked from the watchdog aggregate (§33.3, Phase 0),
  though the mined log shows it present (if never leading) in 50/53 events.
- **Hard salience/threshold gating** in the register path creating
  near-discontinuous gradients at the 0.005 boundary.
- **`depth_code` pushing $V_\theta$ into a stiff regime** for particular token
  contexts — this would re-implicate $V_\theta$ *indirectly* (via the shifted
  xi, not via $B_k$ magnitude) and would show up as correlated `depth_code`
  and $V_\theta$ grads at the same boundary layer. (A special case of the
  cascade hypothesis above if $V_\theta$'s own layer-local amplification
  turns out to be the specific mechanism carrying the disturbance forward;
  a distinct, competing mechanism if the amplification instead runs
  through the dynamics independent of $V_\theta$'s curvature.)

### 33.5 Status and next step

**§33.1 done (both bracket points); §33.3 Phase 0 done — cascade
hypothesis identified from the mined log; Phases 1 and 2 done (code
landed, not yet run against a real crisis); a fresh GRAD_NORM_HARD_TRIGGER
event (to actually exercise Phases 1/2 and test the per-layer growth
prediction) is the immediate next step.** The bracket measurement is
complete: $B_k$ is modestly and non-monotonically elevated at both crises
but far too small in magnitude to be the primary driver. Phase 0 mining of
the L=8 run's `training_log.jsonl` (downloaded from Drive, 964 lines,
steps 50–40,900) found `depth_code` leading 76% of smaller (<200) spikes
but `E`/`P` — tied, to 4 significant figures, at both severe events —
leading 100% of the two hard triggers, which reads as a single
cascade-amplification mechanism crossing a severity threshold rather than
two unrelated culprits (§33.3, §33.4). Phase 1 (spike-batch capture) and
Phase 2 (`replay_spike_batch`, per-parameter/per-layer/activation-extreme
forensics) are implemented in the training notebook (`Cell 6`'s config
and loop, and the new `Cell 6d`) — but they were added *after* the two
existing hard triggers (steps 32,139 / 34,091), which therefore have no
`_spikebatch.pt` sidecar to replay; the harness needs the run to hit a
*new* `GRAD_NORM_HARD_TRIGGER` (expected roughly every ~1,000–2,000 steps
per §32.1's rate) before it has real data to run on. Phase 0 log-mining
against the live L=8 run's `training_log.jsonl` (already on Drive) remains
available immediately, with no new run needed, and may resolve the
leading-group question before Phase 2 even gets a capture to work with.
The SCAF `GradientSpikeProbe` (Phase 3) is specified in the companion
design doc but not yet implemented; Phases 1/2 landing first in the
notebook is intentional (§33.3's "manual harness proves out" precondition
for productionising it).

![Schematic of the GradientSpikeProbe data flow left to right: pinned checkpoint weights and a captured offending batch feed into one isolated forward plus backward run under a save-zero-restore grad invariant, producing per-group per-parameter per-layer grad-norm quantiles and forward-activation extremes, which yield an attribution verdict that branches to either the precision-lr-max lever or a targeted per-group fix](images/scaf_spike_diag_probe_pipeline.png)

---

## 34. `baoab_cfc_lowrank` at L=8 Scale: Correct and Stable, but Not Production-Feasible

While §33's Phase 1/2 harness was being built and awaiting a fresh
`GRAD_NORM_HARD_TRIGGER` to fire against, the still-open §29.2/§30 low-rank
exponential substep — `integrator='baoab_cfc_lowrank'`, which integrates
the anisotropic Gaussian's off-diagonal $B_k B_k^\top$ coupling exactly
rather than explicitly — was finally exercised end-to-end at the live
L=8, $d=384$, OWT scale, resuming from the §32 `baoab_cfc` run's step
37,500 checkpoint. It surfaced three implementation bugs (below), all now
fixed and unit-tested, before landing on the negative verdict that gives
this section its title.

### 34.1 Three bugs surfaced getting it running at scale

**1. `torch.svd_lowrank` is not safe under gradient checkpointing.**
Re-running the training loop with `LOG_INTERVAL=1` after an in-session
edit reproduced

```
CheckpointError: torch.utils.checkpoint: Recomputed values for the
following tensors have different metadata than during the forward pass.
```

with dozens of shape/dtype/device mismatches. `torch.svd_lowrank`'s
internal projection draws from the *global* RNG and silently falls back
through several code paths (including a CPU last resort) depending on
conditioning, so PyTorch's activation-checkpoint recompute — which must
take the *same* branch the forward pass took, tensor-for-tensor — has no
guarantee of doing so. `lowrank_modes` (`cfc_baoab.py`) was rewritten to
call a new `_randomised_svd_det`: a local-`torch.Generator`, branch-free,
GPU-resident randomised SVD whose output shape/dtype/device is a pure
function of its inputs, with no internal fallback.

**2. NaN modes from a degenerate GPU SVD poisoned the whole gradient.**
Even after fix 1, `grad=nan` appeared from the very first training step.
A fully degenerate token (all wells numerically zero-weight at that
position) can make cusolver return `NaN` singular vectors/values
*without raising an exception*; the old masking, `U * keep`, computed
`NaN * 0 = NaN` and injected it into the differentiable substep. Fixed
with `torch.where`-based masking, which picks the zero branch regardless
of the NaN in the discarded branch, plus a final `nan_to_num` scrub on
`U` and `kappa`.

**3. `sqrt(g)` in `harmonic_terms_lowrank` had an infinite backward at
$g=0$.** The same bug class as the `_OMEGA_SQ_FLOOR` fix already
documented at the top of `cfc_baoab.py`, but for the per-well Gaussian
weight $g = w\exp(-\tfrac12 e)$ rather than the stiffness: any well far
enough from $h$ underflows $g\to0$ forward-safely, but
`g.clamp(min=0).sqrt()`'s backward, $1/(2\sqrt g)$, times the upstream
$dg/de = g = 0$, evaluates as $\infty \cdot 0 = \mathrm{NaN}$ in
floating point. Fixed in `model_aniso_gaussian_vtheta.py` by computing

$$\sqrt g = \sqrt w \cdot \exp\left(-\tfrac14\left(e_{\mathrm{diag}} + e_{\mathrm{lr}}\right)\right)$$

directly — identical value to $\sqrt{w e^{-e/2}}$, but analytic in the
exponent $e$, so it has no $1/\sqrt{\cdot}$ node to differentiate through.

### 34.2 Cost controls added

Two config knobs were added to `model_parf_multixi.PARFConfig` so the
(now-correct) exact arm's cost could be tuned down before being measured:
`lowrank_layers` (restrict the expensive path to a subset of layers,
falling back to the cheap diagonal `baoab_cfc` substep elsewhere) and
`lowrank_niter` / `lowrank_oversample` (subspace-iteration and
probe-oversampling knobs for `_randomised_svd_det`, cheaper at the cost of
slightly less accurate top modes — safe here because demoted modes still
go to the stable explicit kick). `lowrank_max_modes` (§30, already
existing) truncates to the $k$ stiffest modes per token via the same
randomised path.

### 34.3 Measured cost: 4-12x `baoab_cfc`, at any layer-count setting

| Configuration | s/step (measured) | vs. `baoab_cfc` (≈10-15 s/step) |
|---|---|---|
| All 8 layers, `lowrank_max_modes=16`, `niter=2`, `oversample=4` | ≈120 s | ≈8-12x |
| 2 stiffest layers only, `niter=1`, `oversample=2` | ≈50 s | ≈4-5x |

The batched per-token SVD is the entire extra cost; it does not amortize
with fewer layers as cheaply as hoped because the remaining SVDs are per
*token* (not per layer) and dominate even a 2-layer configuration. At
~50 s/step, the remaining 62,500 steps of a 100,000-step run would take
roughly **36 days** of wall-clock on a single GPU — not a run that
finishes.

### 34.4 Verdict: retained for completeness, not used in production

**`baoab_cfc_lowrank` is mathematically correct and unconditionally
stable** (no NaNs, no divergence, in every configuration tested) **but is
not production-feasible at L=8/$d$=384/OWT scale**, on cost grounds alone.
It is also, independently, aimed at a target that §33's bracket
measurement already showed to be a weak lever: $\sigma_{\max}(B_k)^2$ —
the exact quantity this arm integrates without approximation — was found
elevated by only +1% to +24%, non-monotonically, between the healthy
checkpoint and both hard-trigger snapshots (§33.1), while `E`/`P` and
`depth_code`, not $V_\theta$'s off-diagonal curvature, lead the actual
gradient spikes (§33.3 Phase 0). Fixing the off-diagonal integration
exactly, even if it were free, would not be expected to address the
dominant spike mechanism.

The code is kept in the repository — `cfc_baoab.py`'s module docstring
now carries this status note alongside the three-bug writeup, and
`model_parf_multixi.PARFConfig.integrator`'s docstring carries a shorter
pointer — because the underlying idea (exact rather than explicit
integration of stiff channels) remains architecturally sound and may
become practical at smaller scale ($d$, $L$, or context length), or if the
per-token batched SVD cost is amortized differently in the future (e.g.
sharing a decomposition across nearby tokens, or a cheaper structured
approximation to $B_k B_k^\top$). **Production training reverts to
`integrator='baoab_cfc'`** — resuming the §32 trajectory from step 37,500
— and the §33 root-cause workflow (Phases 1/2, already landed in the
notebook) remains the active line of investigation for the non-$V_\theta$
spikes.

---

## 35. Phase 1/2 Validated: First `replay_spike_batch` Result (Step 37,763) and a Clean Cascade Signature

### 35.1 A bug fixed first: `torch.load`'s `map_location=DEVICE` broke RNG restore

Running `replay_spike_batch` for the first time against a real capture failed
with `TypeError: RNG state must be a torch.ByteTensor`.
`torch.load(path, map_location=DEVICE, weights_only=False)` remaps every
tensor in the bundle to `DEVICE` (cuda) -- including `rng_state_cpu` and
`rng_state_cuda`'s per-GPU tensors, which `torch.set_rng_state()` /
`torch.cuda.set_rng_state_all()` both require to stay plain CPU
`ByteTensor`s. The model-weights tensors never needed this either:
`replay_spike_batch` already re-maps each one explicitly
(`{k: v.to(DEVICE) for k, v in bundle['model_state_dict'].items()}`) a few
lines later. Fixed by loading with `map_location='cpu'` instead.

### 35.2 Fidelity check: 0.0012% -- bit-exact reproduction confirmed

With the fix in place, the Stage-1 smoke test (`GRAD_NORM_HARD_TRIGGER=110.0`,
§33.5/§34) produced its first `_spikebatch.pt` at step 37,763 (pre-clip total
grad 160.4), and `replay_spike_batch(37763)` reproduced it with:

```
pre_clip_grad_norm_recorded  = 160.39
pre_clip_grad_norm_replayed  = 160.39   (matching groups)
fidelity_gap_pct             = 0.0012%
```

This settles the open question from §33.5: the RNG/microbatch/weight capture
and the isolated replay reproduce a real training step's forward+backward
essentially exactly, not just "close enough." Phase 1 and Phase 2 are
validated end to end.

### 35.3 The result: a clean cascade-amplification signature, not an isolated spike

The per-layer hidden-state gradient norm (the gradient flowing into each
layer boundary -- i.e. how much of the backward pass's magnitude has reached
that point) is monotone across all 8 layers:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| grad norm | 0.169 | 0.102 | 0.044 | 0.027 | 0.019 | 0.013 | 0.008 | 0.005 |

That is roughly a 37x growth from layer 7 down to layer 0, with no bump or
plateau anywhere in the middle. This is exactly the signature §33.4's
cascade-amplification hypothesis predicted (a per-layer $h$-gradient-norm
profile that grows from layer $L-1$ toward layer 0, not an isolated
single-layer spike), and it argues against a single-layer localized cause
(one bad gate or one saturated softmax firing at one specific depth).

### 35.4 `depth_code`'s structural role

The largest single per-parameter gradient at this event is
`V_theta.depth_code` (88.3), narrowly ahead of `E.weight` (81.0) and `P`
(80.9). `depth_code` is used at every layer, so it directly sums the
disturbance from all 8 layers as it accumulates during backprop, rather than
having to survive the full backward pass the way `E`/`P` do (both of which
only see the *cumulative* post-cascade signal once it reaches $h_0$ --
consistent with them landing close behind, not ahead). This is compatible
with either reading: `depth_code` as the stack-wide cascade's natural
aggregation point, or the more specific §33.4 variant where `depth_code`
pushes $V_\theta$ into a locally stiff regime that itself carries the
cascade forward.

### 35.5 Secondary signals: mostly clean, nothing else implicated at this severity

- `creation_gate.tau` sits at 5.6-7.5 (not small/sharp), so the
  "sharp-softmax Jacobian" candidate does not look active here -- though
  `creation_gate.alpha_max` reaching exactly 1.0 for at least one token in
  the batch shows some individual token is winner-take-all even at this
  $\tau$.
- `reverse_ch.Q_force` stays inside [-9.4, 7.6], no outlier blow-up -- the
  "non-conservative reverse-channel kick" candidate is not implicated in
  this event either. `reverse_ch.logit_scale` is a fixed 17.93 across every
  token (it is a single global scalar); worth a follow-up check on whether
  that sits at or near its configured clamp ceiling.
- Several `destruction_gates` swing from near 0 to near 1 across the batch
  (e.g. banks 2, 4, 5, 6), consistent with -- but not confirming -- the
  "hard salience/threshold gating" candidate.
- `V_theta.bank[k].exponent` reaching into the $-10^4$ to $-2\times10^5$
  range is the already-documented underflow regime (most wells are far from
  most tokens); expected, not new information.

### 35.6 Caveat: this event is not yet the target population

Step 37,763's pre-clip total (160.4) sits in the "<200" severity band, where
Phase 0's log-mining (§33.3) found `depth_code` leading 76% of the time --
not the `E`/`P`-dominated severe (≥500) `watchdog_hard_reload` regime the
whole investigation exists to explain. The three top groups here (88.3 /
81.0 / 80.9) are close enough to read as a blend right at that boundary,
which is itself informative (§34's Stage-1 design rationale), but §35.3's
cascade signature and §35.4's `depth_code` attribution should be treated as
a first bit-exact look, not a closed case, until a genuine severe event is
replayed the same way.

### 35.7 Status and next step

**Phase 1/2 fully validated (§35.1-§35.2); one modest-severity replay
analyzed (§35.3-§35.5), with a caveat (§35.6).** `GRAD_NORM_HARD_TRIGGER` is
being restored to `500.0` and `baoab_cfc` production training resumed, per
the two-stage plan (§33.5). The next actionable step is unchanged from
before: replay the next genuine `watchdog_hard_reload` (pre-clip ≥500, the
`E`/`P`-dominated regime per §33.3 Phase 0) the same way, and check whether
the per-layer growth profile holds or sharpens further, and whether `E`/`P`
overtake `depth_code` at the top the way the mined log predicts. (See §36:
the plan below is revised once more real training data came in.)

---

## 36. Decoupling Spike Capture from the Reload Trigger: the Plateau, Not Just the Rare Crisis, Is the Real Target

### 36.1 A ~1,800-step window with zero hard triggers, still stuck on the plateau

After §35's Stage-1 validation, `GRAD_NORM_HARD_TRIGGER` was restored to
`500.0` and `baoab_cfc` production training resumed from step 37,500.
Over the following ≈1,800 steps (37,501-39,350), evals came back 107.85 /
105.59 / 105.84 -- no new best, still stuck on the §32 plateau -- while
**zero** `watchdog_hard_reload` events fired. Four ordinary `[spike]`
events did fire in that window (`GRAD_SPIKE_THRESHOLD=100`, pre-clip 198 to
449), none of them close to the 500 reload line. This directly
demonstrates that the rare ≥500 events are not what is capping this run's
convergence -- whatever is stalling it at ~105 is happening through the far
more frequent, individually-clipped 100-450 band instead (or through
something unrelated to spikes at all, e.g. the WSD schedule not entering
its decay phase until step 65,000 -- §36.4 flags this as still open).

That reframes the diagnostic goal. §33-§35 were built to explain the rare
catastrophic reloads (question A); the run's actual practical problem --
what is capping convergence (question B) -- may have a different answer,
and this window is evidence that (A) does not answer (B).

### 36.2 Why Phase 1 could not see this band: capture and reload shared one knob

Cell 6's Phase 1 capture condition was gated on the same variable as the
watchdog-hard reload:

```python
# capture (old):
if CAPTURE_SPIKE_BATCH and GRAD_NORM_HARD_TRIGGER is not None
        and _raw_gn_pre > GRAD_NORM_HARD_TRIGGER: ...
# reload:
elif GRAD_NORM_HARD_TRIGGER is not None and _raw_gn > GRAD_NORM_HARD_TRIGGER:
    _reload_best(step + 1)
```

Lowering that one threshold to harvest the frequent moderate band would
also have made the watchdog reload on every one of those events --
discarding progress constantly, which is the opposite of what a
convergence-focused investigation wants. This coupling, not a deliberate
choice, is the only reason Phase 1 had only ever captured rare, possibly
unrepresentative catastrophic events.

### 36.3 The fix: a separate `CAPTURE_SPIKE_THRESHOLD`, plus a batch-replay helper

Three changes to
`colab_fock_cfc_baoab_aniso_gaussian_openwebtext_d384.ipynb`:

1. **New `CAPTURE_SPIKE_THRESHOLD = 200.0`** (Cell 6), independent of
   `GRAD_NORM_HARD_TRIGGER = 500.0`. The capture condition now reads
   `_raw_gn_pre > CAPTURE_SPIKE_THRESHOLD`; the reload condition is
   untouched. Lowering the capture bar can no longer cause an extra
   reload -- the two questions (A) and (B) from §36.1 now have
   independently tunable instrumentation.
2. **`SPIKEBATCH_SNAPSHOT_MAX_KEEP` raised 5 -> 12**, so the run
   accumulates a small *library* of moderate-band captures to replay
   across, rather than only ever holding the single latest one.
3. **New `replay_all_captures()` in Cell 6d**, which replays every
   `_spikebatch.pt` currently on disk and prints one summary table (step,
   pre-clip norm, fidelity gap, leading group, next two groups) instead of
   requiring a separate `replay_spike_batch(step_tag)` call per capture.
   `verbose=True` still prints each individual replay's full per-parameter
   / per-layer / activation-extreme report.

### 36.4 What this buys, and the honest caveat

A consistent leading-group and per-layer signature across many
moderate-band replays is materially stronger evidence than the single
step-37,763 data point in §35 -- and, being harvested every ~1-2K steps
instead of waiting an unpredictable number of hours for a ≥500 event, it
arrives fast enough to actually inform this run rather than only
postmortem it.

**Caveat.** It is not yet established that these moderate spikes (rather
than, say, the stable-phase learning rate not having entered its WSD decay
until step 65,000) are what is capping convergence -- each one is clipped,
so any single update is bounded. `replay_all_captures()`'s output is what
will discriminate this: a coherent, fixable mechanism across the library
(the cascade profile sharpening, a specific gate saturating, etc.) argues
for (B) being spike-driven; captures that look like benign clipped noise
argue for the LR-schedule explanation instead.

### 36.5 Status and next step

**Implemented, not yet run against real data.** The notebook changes
landed in this section; the live run has `CAPTURE_SPIKE_THRESHOLD=200.0`
and `SPIKEBATCH_SNAPSHOT_MAX_KEEP=12` armed going forward, but no capture
under the new, lower threshold has fired yet as of this writing. Next
step: once several moderate-band captures accumulate, run
`replay_all_captures()` and compare the resulting per-event leading-group
column against §35's single depth_code-led data point and the severity
trend noted after §35 (`depth_code` leading the smaller of a set of
spikes, `E`/`P` overtaking it in the largest) -- looking specifically for
whether that trend holds up over a larger n.

## 37. Making Cell 6 Resumable In-Place, and Extracting `grad_clip_utils.py`

### 37.1 The problem: inspecting captures cost a full restart

§36's `replay_all_captures()` needs Cell 6d's functions, which (before this
section) required Cell 6 to have already run at least once, and running
`replay_all_captures()` at all meant *interrupting* Cell 6's training loop.
Two practical problems followed directly from Cell 6 being a bare
top-level `for step in range(resume_step, TOTAL_STEPS):` loop:

1. **Resuming training after an interrupt required a full restart.** Cell 5
   unconditionally rebuilds `model` from scratch and Cell 6's own resume
   block always reloads from whatever checkpoint is on disk, so continuing
   training after an inspection break meant: force a manual checkpoint of
   the in-memory state (`save_checkpoint(step + 1, ..., tag_suffix='_manual')`
   from a scratch cell), restart the Colab runtime, re-run Cells 0-5, add a
   one-off override cell to point the resume logic at the manual
   checkpoint, then re-run Cell 6 -- several minutes of ceremony every time,
   purely to look at a JSON file.
2. **Interrupting reliably caused a `CUDA out of memory` on the very next
   `replay_all_captures()` call.** A `KeyboardInterrupt` landing mid-step
   left that iteration's `x`, `y`, `loss`, and the full forward/backward
   graph alive as ordinary top-level notebook globals -- nothing ever went
   out of scope, since there was no enclosing function frame to tear down.
   On this run that pinned 78+ GiB of *allocated* (not just
   reserved-but-cached) GPU memory until a manual
   `gc.collect()`+`torch.cuda.empty_cache()` cell cleared it -- see the
   error/fix pair earlier in this session's log; every one of the first
   `replay_all_captures()` attempts against a freshly-interrupted run OOM'd
   before that fix.

### 37.2 The fix: `run_training()`, a real function instead of a bare loop

The training loop in `colab_fock_cfc_baoab_aniso_gaussian_openwebtext_d384.ipynb`'s
Cell 6 is now:

```python
def run_training(start_step, total_steps):
    global run_ntp, run_vreg, run_fock_reg, n_run, n_skipped, best_val_ppl
    global _grad_norm_ema, _grad_norm_above_thresh, steps_this_session
    global _last_pg_norms, _last_spike_step, step
    for step in range(start_step, total_steps):
        ...  # unchanged body
    _log_fh[0].close()
    print(f'\nTraining complete. Best PPL: {best_val_ppl:.2f}')
    return total_steps


try:
    next_step = run_training(resume_step, TOTAL_STEPS)
except KeyboardInterrupt:
    print(f'\n[run_training] interrupted at step {step + 1:,}. Call '
          f'run_training({step + 1}, TOTAL_STEPS) to resume in-place ...')
    next_step = step + 1
```

Two things fall out of this shape almost for free:

* **Interrupting no longer OOMs the next replay.** `x`, `y`, `loss`, `xb`,
  `yb`, and every other per-step temporary are now locals of
  `run_training`'s stack frame. When a `KeyboardInterrupt` propagates out
  of that frame to the `try/except` around the call, the frame is torn
  down and those locals go with it -- the bare `except KeyboardInterrupt:`
  (no `as e:`, nothing stored) means CPython drops the exception's
  traceback, and every frame it referenced, as soon as the except block
  finishes, typically well before the very next cell (a
  `replay_all_captures()` call) runs.
* **Resuming after an interrupt is just calling the function again.**
  `run_ntp`, `best_val_ppl`, `_grad_norm_ema`, and friends are declared
  `global` and initialized exactly once, outside and before the function
  (unchanged from before); `run_training()` only mutates them, so a second
  call picks up exactly where the first left off -- watchdog EMA state,
  best-PPL tracking, and the open log file handle all carry over. The new
  contract is:

  ```python
  next_step = run_training(next_step, TOTAL_STEPS)   # re-run this line to resume
  ```

  with no rebuild, no checkpoint reload, and no runtime restart, unless you
  actually want to pick up new *code* (e.g. a fresh `git pull` of this
  notebook) -- `save_manual_checkpoint(next_step)` (a thin wrapper over the
  existing `save_checkpoint(..., tag_suffix='_manual')`) is still there for
  that case.

`evaluate()` got the same treatment for a narrower failure mode: its
`model.eval()`/`model.train()` pair is now a `try/finally`, so an interrupt
landing mid-eval can no longer leave `model` stuck in eval mode for
whatever `run_training()` call resumes after it.

### 37.3 `grad_clip_utils.py`: the first functionality actually pushed to a module

`assign_clip_group`, `per_group_grad_norms`, and `clip_grads_per_group`
moved out of Cell 6 into
`notebooks/conservative_arch/scaleup/grad_clip_utils.py`, parameterized by
a small `GradClipConfig` dataclass (`default_clip`, `overrides`,
`watchdog_exclude_groups`) instead of reading Cell 6's globals directly.
This was the one piece of Cell 6 judged low-risk enough to extract in this
pass: the three functions are pure functions of `(model, config)` with no
hidden state, and moving them fixes a real, pre-existing wart --
Cell 6d's `replay_spike_batch`/`replay_all_captures` silently depended on
Cell 6 having already executed at least once, purely so these three names
existed as globals. `grad_clip_utils.py` ships with
`test_grad_clip_utils.py` (4 CPU-only tests against a toy module, run via
`python test_grad_clip_utils.py` or `pytest`), and Cell 6d now moves to
*before* Cell 6 in the notebook's cell order, since its functions no
longer need Cell 6 to have run first to be *defined* (only to be *called*,
once `_GRAD_CLIP_CFG` and a handful of other Cell-6 config globals exist).

**Deliberately not extracted in this pass:** `evaluate`, the two probes
(`run_causal_probe`, `run_trained_leak_probe`), `save_checkpoint`,
`forward_with_vreg`, and the watchdog/reload logic all stay inline in
Cell 6. They are tightly coupled to `model`/`model_cfg` and several dozen
live-tunable config constants, and there is no way to validate a full
extraction against the actual GPU+OpenWebText pipeline outside Colab --
moving them now would be a much larger, harder-to-verify change for
comparatively little benefit to the actual pain point (§37.1). Candidate
for a later, separate pass once this run's current phase is less
time-critical.

### 37.4 Corollary bug found the same day: `replay_all_captures()` had the identical leak

Confirmed live on this run: interrupting Cell 6 to inspect the run's first
genuine `watchdog-hard` reload (step 41,837, pre-clip 528.0 -- analysis
pending, to be added as a future section once replayed) reproduced the
exact §37.1 OOM symptom (`allocated` stuck near 80 GB even
after the memory-cleanup cell) from *inside* `replay_all_captures()`
itself, not just from interrupting Cell 6's training step. Root cause:
`KeyboardInterrupt` derives from `BaseException`, not `Exception`, so
`replay_all_captures`'s per-capture `except Exception as e:` never caught
it -- an interrupt landing mid-replay propagated uncaught to IPython's
top-level handler, which pins the interrupted replay's forward/backward
graph via `sys.last_traceback` exactly like an interrupted training step
does. Fixed by adding an explicit `except KeyboardInterrupt:` clause that
stops the loop (keeping whatever reports already replayed) instead of
letting the exception escape. `replay_spike_batch`'s own `finally` block
was never affected by this -- `finally` runs for any `BaseException`, so
the live model's weights/grads/hooks were always correctly restored even
when this bug pinned memory; only the *cleanup of the replay's own
temporaries* was missing.

### 37.5 Status

**Implemented, syntax-checked, not yet exercised against a live GPU
session.** Every notebook cell was re-parsed with `ast.parse()` after the
edit (no `SyntaxError`s), `grad_clip_utils.py`'s 4 unit tests pass, and the
full CfC/BAOAB test suite (`test_cfc_baoab.py`) still passes unchanged.
`git diff` on the notebook is limited to Cell 6d's relocation, Cell 6's
edits, and a trailing-newline addition -- no unrelated cells were touched.
Next real test is the first live interrupt-and-resume cycle on the running
L=8 job.

## 38. Seven Replays In: Two Distinct Failure Modes, Not One -- and Only the Localized One Has Crossed 500

### 38.1 The first genuine `watchdog-hard` reload, replayed

Step 41,837 (pre-clip 528.0) is this run's first `GRAD_NORM_HARD_TRIGGER`
event since Phase 1/2 went live in §35, replayed alongside the also-newly-
captured step 41,824 (265.8, under the reload line but over
`CAPTURE_SPIKE_THRESHOLD`). Both replayed with the same excellent fidelity
as the earlier five (all seven now <0.0013% gap), so the comparison below
draws on n=7, not n=1:

| step | pre-clip | leader (norm) | layer 0 -> 3 ($h$-grad) | layer0/layer3 | profile |
|---|---|---|---|---|---|
| 37,763 | 160.4 | `depth_code`=88.3 | 0.169, 0.102, 0.044, 0.027 | 6.3x | smooth cascade |
| 40,043 | 379.8 | `reverse_channel_scale`=274.9 | 0.081, 0.064, 0.040, 0.025 | 3.2x | smooth cascade |
| 40,387 | 369.3 | `depth_code`=258.8 | 0.082, 0.058, 0.033, 0.021 | 3.9x | smooth cascade |
| 41,318 | 432.8 | `E`=266.7 | 0.054, 0.042, 0.028, 0.021 | 2.6x | smooth cascade |
| 41,824 | 265.8 | `register`=231.8 | 0.087, 0.070, 0.047, 0.032 | 2.7x | smooth cascade |
| 39,983 | 235.5 | `depth_code`=208.9 | 16.45, 12.20, 6.08, 0.33 | 50x | **localized** |
| 41,837 | 528.0 | `depth_code`=424.6 | 31.07, 12.61, 1.46, 0.17 | 177x | **localized** |

### 38.2 Two failure modes, discriminated cleanly by the per-layer profile

Five of the seven replays reproduce the smooth, monotone cascade signature
from §33.4/§35: layer 0's $h$-gradient is only 2.6-6.3x layer 3's, decaying
gently across all 8 layers, consistent with a single gain compounding as
it backpropagates the full stack. The other two -- 39,983 and 41,837 -- are
categorically different: layers 0-2 are **50-300x larger** than the same
layers in every one of the other five replays, with a sharp cliff by layer
3, after which they rejoin the same gentle tail everyone else has. This is
not a one-off artifact of a single replay (the possibility flagged when
39,983 was first seen): it has now recurred, at roughly double the
severity (layer 0: 16.45 -> 31.07; total: 235.5 -> 528.0), a second
independent instance of the same distinctive shape.

**Leadership does not predict the mode.** `depth_code` leads both localized
events, but it also leads 40,387 (258.8, its second-largest reading of all
seven) -- a perfectly smooth cascade, not localized. The per-layer profile,
not which group tops the leaderboard, is the reliable discriminator between
the two modes.

**`register` leading 41,824 (231.8) is itself a new observation** -- no
prior replay in this run had `register` anywhere near the top of the
per-group breakdown; it is otherwise a normal smooth-cascade event.

### 38.3 Working hypothesis: the localized mode is what actually forces a reload

Of the five smooth-cascade events, none exceeds 432.8 pre-clip. The one
event that has crossed `GRAD_NORM_HARD_TRIGGER=500` in this run's entire
Phase-1/2-instrumented history is a localized event (41,837). Read
together with 39,983 being the second-most-severe localized case and also
comfortably under the reload line at the time, the working hypothesis is:
the smooth cascade self-limits under per-group clipping (each group's own
threshold caps its own contribution before the aggregate can compound much
past ~400), while the layer-0-2 localized blowup does not follow the same
per-group-clip-bounded dynamics and is the actual driver of this run's rare
genuine reloads.

**Caveat: this is n=2 for the localized mode.** It is a real, recurring,
now-twice-confirmed signature, and the only one of the two modes to have
reached the reload threshold so far -- but two events is not enough to
rule out coincidence in which mode happens to cross 500 first. The
falsification test is straightforward: keep harvesting captures via
`replay_all_captures()`; if a *smooth-cascade* event eventually also
crosses 500, or a third localized event stays comfortably under it, the
hypothesis needs revising.

### 38.4 What is not yet known

Nothing in the per-parameter or activation-extreme breakdown (V_theta bank
exponents, `xi_shifted_norm`, destruction-gate outputs, creation-gate tau)
cleanly separates the two localized events from the five smooth-cascade
ones on inspection so far -- the layer-0-2 $h$-gradient profile is the only
discriminator found. What specifically differs about the forward pass at
layers 0-2 on a localized-mode step (a particular token pattern, a
register-creation event, a V_theta well boundary crossing, ...) is open;
answering it would need comparing the *forward* activations (not just
backward gradients) between a localized and a smooth-cascade capture at
matched layers, which the current instrumentation does not yet do.

### 38.5 Status

**Descriptive finding, not yet a mitigation.** Documents the pattern
across the 7 captures replayed to date; no code change proposed here.
Next step: keep accumulating captures under `CAPTURE_SPIKE_THRESHOLD=200`
and re-run this comparison at larger n to firm up (or break) both the
mode-discrimination claim (§38.2) and the reload-driver hypothesis (§38.3).

### 38.6 Deepening forensics on the two existing localized captures, without waiting for new ones

§38.4 flagged that nothing in the *current* forward-activation breakdown
discriminates the two modes. Before spending wall-clock time waiting for a
third localized event, it's worth exploiting a fact already true today:
every `_spikebatch.pt` bundle pins the exact pre-step weights, the exact
offending microbatch tokens, and the RNG state, and Cell 6d's replay is
already proven bit-exact (<0.0013% fidelity gap on all 7). That means
steps 39,983 and 41,837 can be re-interrogated with *more* instrumentation
right now, with zero new training time -- so Cell 6d gained three
additions the same day, rather than waiting on more captures:

1. **Fixed a real per-layer attribution bug.** `creation_gate_qkv`,
   `reverse_ch`, and each `V_theta` bank are single shared modules called
   once per `_fock_layer_step` invocation (i.e. once per layer, L times
   per forward) -- unlike `destruction_gates`, a genuine per-layer
   `ModuleList`. The old `_record()` hook overwrote the same dict key on
   every layer's call, so `activation_extremes` silently only ever
   reflected **layer 7's** values -- the layers *not* implicated in the
   localized mode. `activation_extremes` is now `{name: {layer: stats}}`,
   layer-resolved for every hooked op. This plausibly explains why §38.4
   found "nothing" in the existing breakdown: it was looking at the wrong
   layers the whole time.
2. **Wired in the model's own per-layer diagnostic buffer.**
   `FockMultiXiPARFLM.set_fock_capture(True)` (the same API the eval-time
   causal/register diagnostics already use) makes `_fock_layer_step`
   append a cheap `@torch.no_grad()` dict per layer per forward:
   `active_frac`, `salience_mean/std`, `reg_cos_sim` (register-collapse
   diagnostic), `destroy_mean`, `create_alpha_max`, `create_entropy`,
   `rev_entropy`, `rev_alpha_max`, `rev_scale`, `qforce_ratio`. `replay_spike_batch`
   now enables it for the replayed forward+backward and reports
   `fock_capture_per_microbatch` (full per-layer table) in the returned
   report, printed as a table in verbose mode.
3. **Added `inspect_spike_tokens(step_tag)`**, pure CPU bookkeeping (no
   model, GPU, or RNG state touched) that decodes the offending
   microbatch's GPT-2 tokens via Cell 3's `tok`, ranks rows by longest
   same-token repeat run and unique-token ratio, and prints a decoded
   snippet of the most degenerate rows -- testing whether a boilerplate
   OpenWebText passage (long whitespace/punctuation/header repeats) is
   what triggers the localized layers-0-2 blowup, independent of anything
   on the gradient side.

**Status: implemented, syntax-checked (`ast.parse` on every notebook
cell), existing `grad_clip_utils`/CfC-BAOAB test suites still pass
unchanged, not yet run against a live GPU session.** Next step: run
`replay_spike_batch(39983)` / `replay_spike_batch(41837)` and
`inspect_spike_tokens(39983)` / `inspect_spike_tokens(41837)` against the
already-captured bundles and see whether either new signal (per-layer
register/gate health, or offending-token degeneracy) separates from the
five smooth-cascade replays.

### 38.7 The deepened forensics discriminate nothing -- but a number already sitting in the Phase-0 data does

§38.6's three additions were run against the two localized captures
(39,983 / 41,837), then, for a genuine control group, against two of the
five smooth-cascade captures (37,763 / 41,318). Both localized events
replayed with the same 0.0% fidelity as before.

**Negative result 1 -- token degeneracy, ruled out.** `inspect_spike_tokens`
found nothing: across all 64 rows examined (32 per capture), every single
row has `max_repeat_run = 2` -- the floor value for ordinary English text
(no run of 3+ identical tokens anywhere) -- and unique-token ratios in the
unremarkable 0.29-0.63 range. The decoded snippets are routine OpenWebText
prose (and, incidentally, one Solidity smart-contract snippet mixed into
the 41,837 batch, but that's normal corpus composition, not a cause --
grad norms aggregate over the whole 32-row batch, so nothing here singles
out *which* row would even matter). Boilerplate/degenerate input text is
not the trigger.

**Negative result 2 -- the layer-resolved activation extremes and
`set_fock_capture` gate-health table, ruled out.** Comparing all four
deep replays side by side, the numbers are essentially indistinguishable
between modes. Layer 0's `create_alpha_max` mean is 0.0036-0.0041 and its
`create_entropy` is 6.215-6.224 in *all four* events, localized or smooth
-- to three significant figures, regardless of which failure mode
happened. `V_theta.bank[*].exponent` is in the tens-of-thousands-negative
range in all four. `reverse_ch.Q_force` and `destruction_gates[*].output`
per-layer curves are likewise the same shape in all four. This appears to
be a property of the current model weights at this point in training, not
of the batch or the outcome -- so this round of instrumentation, while a
useful sanity check, was measuring the wrong thing.

**Correction to §38.6 item 1:** `V_theta` banks turned out *not* to share
the per-layer-overwrite bug after all. Re-inspecting real replay output
showed every `V_theta.bank[*]` stat reported at "layer 7" in *every*
capture, localized or smooth -- because `V_theta` is called once per
forward pass on the full depth-stacked `xi` tensor, not once per
`_fock_layer_step`. `_record()` now takes an explicit `layer=-1` override
for these three stats (matching the existing `reverse_ch.logit_scale`
convention) instead of trusting `_current_layer`, which would otherwise
report whichever layer happened to run last.

**Positive result -- it was already sitting in the Phase-0 data,
uncomputed.** Going back to the original 7-event `spike_replay_reports`
JSON (no new replay needed) and computing one ratio per event --
`override:depth_code`'s captured group norm divided by the next-largest
group's norm ("`dc_ratio`") -- produces a clean split:

| step | mode | pre-clip | `depth_code` | 2nd-largest group | `dc_ratio` | L0/L3 |
|---|---|---|---|---|---|---|
| 37,763 | smooth | 160.4 | 88.3 | E (81.0) | 1.09 | 6.2 |
| 40,043 | smooth | 379.8 | 163.2 | reverse_channel_scale (274.9) | 0.59 | 3.2 |
| 40,387 | smooth | 369.3 | 258.8 | P (144.8) | 1.79 | 3.9 |
| 41,318 | smooth | 432.8 | 31.2 | E (266.7) | 0.12 | 2.6 |
| 41,824 | smooth | 265.8 | 73.1 | register (231.8) | 0.32 | 2.7 |
| 39,983 | **localized** | 235.5 | 208.9 | P (67.5) | **3.09** | 50.1 |
| 41,837 | **localized** | 528.0 | 424.6 | E (190.7) | **2.23** | 177.6 |

Every smooth-cascade event has `dc_ratio < 1.8`; both localized events
have `dc_ratio > 2.2`. `dc_ratio` is computable straight from data the
watchdog already gathers every step (`per_group_grad_norms`/
`clip_grads_per_group`'s per-group breakdown) with zero replay cost, and
it is mechanistically plausible as the actual signature of the localized
mode: `depth_code` is what shifts `xi` per depth position, so a runaway
gradient there concentrates in the early layers where that shift is
applied, rather than spreading evenly like an ordinary cascade -- which
is exactly the L0/L3 pattern that originally defined the split.

**Caveat -- this cannot yet be validated as a *leading* indicator.**
`training_log.jsonl`'s periodic entries only ever carried the aggregate
`grad_norm`, never the per-group breakdown; the breakdown was only ever
persisted to disk at the 7 spike-capture moments. So there is no way to
retroactively check what `dc_ratio` looks like on ordinary, uneventful
steps, or whether it climbs for several steps before a localized-mode
hard-trigger (a real early-warning signal) versus jumping simultaneously
with no lead time (useful for post hoc labeling, not prevention). Cell
6's periodic `LOG_INTERVAL` log line now also writes `dc_ratio` (the
per-group breakdown needed for it is already computed in memory every
step when `PER_GROUP_CLIP` is on, for clipping itself -- this just
persists one extra derived number, no new computation), so a future
mining pass over an extended run has the data to answer this.

**Status: negative results (token degeneracy, layer-resolved activation
extremes) and the `dc_ratio` correlation established from the existing 7
captures; `V_theta` layer-label correction and `dc_ratio` logging landed
in Cell 6/6d; not yet validated against live per-step data.** Next step:
let training accumulate steps with `dc_ratio` now being logged, then mine
`training_log.jsonl` for its distribution on ordinary steps and its
behavior in the run-up to any future hard-trigger.

## 39. The Token-Minority Conjecture Is Falsified Twice Over: the Localized Mode Is Batch-Wide, Not Batch-Specific

§38.4 established that nothing in the forward-activation breakdown
discriminates the two modes, but flagged that *density*-style measurements
(as opposed to the min/max/mean extremes already tried) had not been ruled
out, and that comparing forward activations at matched layers between the
two modes had not been done. Two such measurements were designed and run
against all four Phase-1/2-instrumented events with a per-layer $h$-gradient
profile on record (37,763 / 41,318 smooth; 39,983 / 41,837 localized). Both
came back negative -- one of them decisively in the *opposite* direction
from the working conjecture.

### 39.1 The conjecture, stated before either result came back

Per-group clipping caps `depth_code` at 0.25 on *every* step, quiet or
spiking (§28's clamp, tightened 0.5 → 0.25 on 2026-08-23 precisely because
`depth_code` was already saturating its old ceiling on every quiet step --
see the `GRAD_CLIP_OVERRIDES` comment in Cell 6). So a localized event's
*applied* `depth_code` update is exactly the same size as a quiet step's --
0.25 either way, whether the pre-clip norm was 88 or 425. Its damage
therefore cannot be magnitude; it has to be direction. The conjecture: the
direction is dictated by a small minority of rows in the batch whose tokens
land near a sharp `V_theta` well at layers 0-2 (the only layers carrying
meaningful salience, §38.7), while the other ~30 rows contribute little.
Two tests were designed to check this *before* either was run, so the
result couldn't be retrofitted to the story:

1. **Per-row gradient attribution** (`attribute_spike_rows`, new in Cell
   6d): replay the captured batch one row at a time (RNG reset to the
   capture's pinned state before every row, so all rows see an identical
   noise draw and are comparable to each other) and measure what fraction
   of the summed `depth_code` row-norms a single row accounts for. Flat
   batch of 32 rows -> baseline 1/32 = 0.031; conjecture predicts the two
   localized events sit well above that and the two smooth events sit near
   it.
2. **`V_theta` exponent occupancy histogram** (added to `replay_spike_batch`):
   per-bank counts of how many token-slots have exponent > -10 (i.e.
   `exp()` >= 4.5e-5, the well still contributes gradient) versus
   underflowed to numerically dead. Conjecture predicts localized events
   show a denser live band than smooth events.

### 39.2 Result 1: per-row attribution -- concentration is *higher* in the smooth mode

| step | mode | pre-clip | `depth_code` top-1-row share | top-3-rows share | layer-0 $h$-grad top-1 share |
|---|---|---|---|---|---|
| 37,763 | smooth | 160.4 | 0.292 | 0.558 | 0.238 |
| 41,318 | smooth | 432.8 | **0.387** | 0.585 | **0.429** |
| 39,983 | localized | 235.5 | 0.217 | 0.424 | 0.149 |
| 41,837 | localized | 528.0 | **0.095** | 0.248 | **0.115** |

(uniform-batch baseline for 32 rows: 0.031)

This is the opposite ranking from the conjecture. The two localized events
are the *flattest* of the four -- 41,837, the single most extreme event on
record (layer0/layer3 $h$-gradient ratio 177x, the only one to cross the
500 hard-trigger), has its `depth_code` gradient spread almost evenly
across all 32 rows, barely 3x the flat-batch baseline. The two smooth
events are the more row-concentrated ones, with 41,318 putting 58% of the
`depth_code` gradient on three rows. There is also a clean monotonic
anti-correlation: as layer-0-2 localization gets more severe (39,983's 50x
-> 41,837's 177x layer0/layer3 ratio), the per-row concentration gets
*less* severe (0.217 -> 0.095 top-1 share) in lockstep.

### 39.3 Result 2: exponent occupancy -- no separation in either direction

| bank | 37,763 (smooth) | 41,318 (smooth) | 39,983 (localized) | 41,837 (localized) |
|---|---|---|---|---|
| 0 | 7.6e-6 | 3.8e-5 | 3.1e-5 | 3.1e-5 |
| 1 | 1.5e-5 | 3.1e-5 | 7.6e-6 | 2.3e-5 |
| 2 | 1.4e-4 | 1.4e-4 | 1.4e-4 | 1.4e-4 |
| 3 | 1.4e-3 | 8.8e-4 | 9.3e-4 | 8.7e-4 |
| 4 | 2.4e-3 | 1.1e-3 | 1.6e-3 | 8.6e-4 |

(`live_frac`: fraction of token-slots per bank with exponent > -10)

Bank 2 is 1.4e-4 in all four events; the others are within the same order
of magnitude across modes with no consistent direction. Worth noting on
its own, independent of the mode question: **at every capture, regardless
of mode, over 99.9% of token-well pairs are numerically dead** (exponent
< -10, contributing exactly zero gradient) -- the anisotropic-Gaussian
`V_theta` wells operate in this extremely sparse regime universally, not
just during spikes.

### 39.4 Revised picture: localized is layer-localized but batch-wide

"Localized" in §38 was always a statement about *layers* (0-2 blow up, 3-7
don't), never about *rows*. §39.2 shows those are not the same thing: the
localized mode is layer-localized but **batch-wide** -- something makes
nearly every row in the batch simultaneously hypersensitive at layers 0-2,
rather than one bad token dominating. The smooth mode, perhaps counter to
its name, is the one with real row-level structure.

Combined with §38.4/§38.6/§38.7's prior negative results (token
degeneracy, layer-resolved activation extremes, per-layer gate health all
failed to discriminate the modes), the pattern is now consistent across
five independent forward-pass-side measurements: **nothing about what is
in this particular batch, on this particular forward pass, discriminates
the two modes.** The only thing that ever has is a backward/parameter-
magnitude quantity -- `dc_ratio` (§38.7) -- which is really a statement
about where in *parameter* space the gradient concentrates
(`depth_code`+`V_theta` vs. everywhere else), not about the data.

A batch-wide effect that gets *more* uniform across rows as it gets *more*
severe by layer smells like a property of the shared weights/dynamics
state the model is in going into that step, not of which tokens happen to
be present. This reopens, in sharper form, the `sigma_max(B_k)` /
precision-matrix stiffness line from §28/§31 (previously bracketed and
downgraded as "not the primary driver" for *overall* run stability, a
different question). The sharper form: does a raw weight-space stiffness
proxy -- e.g. `V_theta.bank.banks[k].B_proj`'s spectral or Frobenius norm,
computed from the *parameters themselves*, not their per-token forward
evaluation (which §39.3 just showed doesn't discriminate) -- drift upward
over the steps immediately preceding a localized event, before any
particular batch is even drawn? The `_spikebatch.pt` bundles are single-
instant snapshots and cannot answer this; it needs a trajectory.

### 39.5 Status

**Two forward-pass hypotheses falsified (token-minority via per-row
attribution; well-density via exponent occupancy), both tested against
all four Phase-1/2-instrumented events with a recorded layer profile
before either was run.** Sharpens rather than resolves the open question
from §38.4: the discriminator is confirmed to live on the parameter side,
not the data side. Next step, mirroring the `dc_ratio` pattern of §38.7:
add a weight-space stiffness proxy for the `V_theta` low-rank precision
factor to Cell 6's periodic `LOG_INTERVAL` logging, then mine it
retroactively once more localized events accumulate, checking specifically
whether it trends upward in the steps before a localized-mode hard-trigger
(a real leading indicator) or only coincides with one (useful for labeling,
not prevention).

## 40. Does `baoab_cfc_lowrank` Address the Localized Mode? Reconciling §33's Verdict, and a Targeted Ablation Test

§34.4 closed with a negative verdict on `baoab_cfc_lowrank`: not
production-feasible on cost, and independently "aimed at a target §33's
bracket measurement already showed to be a weak lever." §39 leaves the
localized mode pointing at exactly that target -- weight-space
`sigma_max(B_k)` stiffness. This section asks directly whether the earlier
verdict actually rules that out, and lays out the cheap offline test that
would settle it without touching the cost question at all.

### 40.1 The old verdict was measured against a different-looking crisis

§33's bracket used a fixed, generic probe batch (not the batches that
tripped the two triggers it measured) and found `sigma_max(B_k)^2` elevated
by only +1% to +24%, non-monotonically, between a healthy checkpoint and the
step-32,139 / step-34,091 hard-trigger snapshots -- far too small to explain
grad-norm spikes two orders of magnitude larger. The mechanism §33.3
identified instead was a cascade that amplifies through all eight layers and
lands equally on `E` and `P` at the embedding boundary.

That signature -- gradual amplification across the whole stack, landing on
the embedding groups -- is structurally the **smooth-cascade** mode of §38,
not the **localized** L0-2-cliff mode that motivates §39's weight-space
hypothesis and is the only mode ever observed to cross the 500 hard trigger.
Two gaps mean §33's verdict does not automatically transfer:

1. §33.1 flags its own limitation: a generic probe batch "does not by itself
   rule out a transient, batch-specific `B_k` excursion on the offending
   step." §38-§39's Phase 1/2 replays are the first measurement made
   directly on the actual offending batches.
2. §33's two triggers (32,139 / 34,091) predate every localized-mode capture
   on record (37,763 onward, all after the step-37,500 reload that
   preceded §34's `baoab_cfc_lowrank` trial). The bracket was never run
   against the failure mode this section is asking about.

### 40.2 What the existing `baoab_cfc_lowrank` trial already hints at

§34's own end-to-end trial ran from that same step-37,500 checkpoint and
reported the exact arm "mathematically correct and unconditionally stable --
no NaNs, no divergence, in every configuration tested." That window
temporally brackets where localized events start appearing under plain
`baoab_cfc`. No instability of the localized kind was reported while running
the exact arm -- weak evidence (the mode taxonomy did not exist yet, so
nobody was explicitly looking for it), but a real, standing absence-of-signal
worth stating plainly rather than re-discovering from scratch.

### 40.3 The cost floor is unchanged, but a layer-restricted deployment was never tried

§34.3 measured cost for two configurations only -- all 8 layers
(8-12x `baoab_cfc`) and 2 layers picked generically (4-5x). §38.7's salience
profile now says *which* layers the localized mode actually lives in:
0-2, identically across every replay (salience roughly 0.32 / 0.14-0.22 /
0.06-0.15 at layers 0-2 versus 0.001-0.0001 at layers 5-6). Restricting
`lowrank_layers` to `frozenset({0, 1, 2})` -- informed by that finding rather
than a generic pick -- has not been measured, and per-token SVD cost does
not amortize cleanly with layer count (§34.3), so it should be treated as an
open number, not assumed to land near the 4-5x point.

### 40.4 The direct test: an integrator-ablation replay against the existing bundles

The question can be answered without a training run at all. The
`_spikebatch.pt` bundles already pin weights, batch, and RNG state for every
captured event; `replay_spike_batch`'s snapshot/restore invariant already
proves this kind of ablation is safe to run against a live session. The only
new ingredient is swapping `model.cfg.integrator` between arms before
replaying the same pinned bundle, since `model_parf_multixi.py` reads
`cfg.integrator` and `cfg.lowrank_layers` as plain runtime branches
(`use_cfc = cfg.integrator == "baoab_cfc"`, `use_lowrank = cfg.integrator ==
"baoab_cfc_lowrank"`) -- the weights themselves (`B_proj`, `mu_proj`, etc.)
are identical either way, so no new checkpoint or retraining is needed.

```python
def replay_integrator_ablation(step_tag, lowrank_layers=frozenset({0, 1, 2}),
                                mdl=None, verbose=True):
    """SS40. Re-run a captured spike batch under two integrator configs --
    the recorded 'baoab_cfc' and 'baoab_cfc_lowrank' restricted to
    lowrank_layers -- with weights, batch, and RNG state held bit-identical
    to the capture, and compare the resulting pre-clip gradient norm and
    per-layer h-gradient profile (SS38's discriminator). Same non-pollution
    invariant as replay_spike_batch: weights, .grad tensors, RNG state, and
    now also cfg.integrator / cfg.lowrank_layers are snapshotted up front
    and restored in a finally block.
    """
    mdl = mdl if mdl is not None else model
    path = CKPT_DIR / f'{CKPT_PREFIX}_step{step_tag}_spikebatch.pt'
    bundle = torch.load(path, map_location='cpu', weights_only=False)

    saved_sd = _copy.deepcopy(mdl.state_dict())
    saved_grads = _isolated_grad_snapshot(mdl)
    saved_rng_cpu = torch.get_rng_state()
    saved_rng_cuda = torch.cuda.get_rng_state_all() if DEVICE == 'cuda' else None
    saved_integrator = mdl.cfg.integrator
    saved_lowrank_layers = getattr(mdl.cfg, 'lowrank_layers', None)

    layer_grad = {}
    orig_layer_step = mdl._fock_layer_step

    def instrumented_layer_step(h, h_prev, r, salience, m_b, gamma, dt,
                                 layer_idx, *args, **kwargs):
        out = orig_layer_step(h, h_prev, r, salience, m_b, gamma, dt,
                               layer_idx, *args, **kwargs)
        h_new = out[0]
        if torch.is_tensor(h_new) and h_new.requires_grad:
            h_new.register_hook(
                lambda g, li=layer_idx: layer_grad.setdefault(li, float(g.detach().norm())))
        return out

    arms = [
        ('baoab_cfc as captured', 'baoab_cfc', None),
        (f'baoab_cfc_lowrank layers {sorted(lowrank_layers)}',
         'baoab_cfc_lowrank', lowrank_layers),
    ]
    results = {}
    try:
        mdl.load_state_dict(
            {k: v.to(DEVICE) for k, v in bundle['model_state_dict'].items()},
            strict=False)
        mdl._fock_layer_step = instrumented_layer_step
        grad_accum = bundle.get('grad_accum', len(bundle['batches']))
        mdl.train()
        for label, integrator, layers in arms:
            mdl.cfg.integrator = integrator
            mdl.cfg.lowrank_layers = layers
            layer_grad.clear()
            for p in mdl.parameters():
                p.grad = None
            torch.set_rng_state(bundle['rng_state_cpu'])
            if bundle.get('rng_state_cuda') is not None and DEVICE == 'cuda':
                torch.cuda.set_rng_state_all(bundle['rng_state_cuda'])
            for xb, yb in bundle['batches']:
                x = torch.from_numpy(xb).to(DEVICE)
                y = torch.from_numpy(yb).to(DEVICE)
                loss, *_ = forward_with_vreg(
                    x, y, LAMBDA_V, LAMBDA_FOCK_REG, FOCK_REG_EPS)
                (loss / grad_accum).backward()
            total_sq = sum(float(p.grad.detach().norm()) ** 2
                           for p in mdl.parameters() if p.grad is not None)
            results[label] = {
                'pre_clip_grad_norm': total_sq ** 0.5,
                'per_layer_h_grad': dict(sorted(layer_grad.items())),
            }
            if verbose:
                print(f'[ablation] {label}: pre_clip_grad_norm='
                      f'{results[label]["pre_clip_grad_norm"]:.2f}  '
                      f'per_layer={results[label]["per_layer_h_grad"]}')
    finally:
        mdl._fock_layer_step = orig_layer_step
        mdl.cfg.integrator = saved_integrator
        mdl.cfg.lowrank_layers = saved_lowrank_layers
        mdl.load_state_dict(saved_sd)
        _isolated_grad_restore(mdl, saved_grads)
        torch.set_rng_state(saved_rng_cpu)
        if saved_rng_cuda is not None and DEVICE == 'cuda':
            torch.cuda.set_rng_state_all(saved_rng_cuda)
    return results
```

Cost note: even a `lowrank_layers={0,1,2}` replay pays §34.3's per-token SVD
cost, so a handful of forward/backward passes offline against an existing
bundle is a several-second-to-minute affair, not a multi-day commitment --
the sketch above is deliberately cheap to actually run, unlike the question
of deploying it live.

| Outcome | Localized pre-clip norm under the exact arm | Per-layer profile | Reading |
|---|---|---|---|
| Confirms the hypothesis | drops toward the sub-100 healthy range | loses the L0-2 cliff, flattens toward the smooth-mode shape | the approximate off-diagonal kick in plain `baoab_cfc` was generating the spike; exact integration removes it |
| Falsifies the hypothesis | stays comparable to the `baoab_cfc` replay | keeps the same L0-2 cliff shape | the localized mode survives exact dynamics; `sigma_max(B_k)` growth (or something else) acts through the true force too, not through an integration artifact |

```mermaid
flowchart LR
    A["captured localized event<br>step 39983 or 41837"] --> B["replay under baoab&#95;cfc<br>as captured baseline norm"]
    A --> C["replay under baoab&#95;cfc&#95;lowrank<br>layers 0 1 2 only"]
    B --> D{"compare pre clip norm<br>and per layer profile"}
    C --> D
    D -->|lowrank suppresses cliff| E["mechanism confirmed<br>try a live layer restricted trial"]
    D -->|no change| F["mechanism falsified<br>search a non V theta cause"]
```

### 40.5 Decision rule

If the exact arm suppresses both the pre-clip norm and the L0-2 cliff on the
localized captures (39,983 / 41,837), the mechanism is confirmed for this
mode specifically -- contradicting nothing in §33, which was scoped to a
different crisis signature -- and the next real experiment is measuring
`lowrank_layers={0,1,2}`'s actual per-step cost live, since §34.3 never
measured that configuration. If it does not, the hypothesis is falsified for
the localized mode too, in which case the search needs a localized-mode
analogue of §33.3's non-`V_theta` root-cause workflow: something other than
`V_theta`'s off-diagonal curvature is producing a layer-0-2-concentrated
cliff that plain `baoab_cfc`'s explicit off-diagonal kick and the exact
`baoab_cfc_lowrank` arm reproduce equally.

---

## 41. Chronic, Not Transient: Three New Replays Refine §33 and §38, and Motivate Turning On `precision_lr_max`

Following the `dc_ratio` / `b_proj_sigma_max` logging added in §38.7/§39.5,
the run produced a fresh `GRAD_NORM_HARD_TRIGGER` (step 47,116, pre-clip
13,139.5 -- by a wide margin the largest single-step gradient norm recorded
in this run's history) plus two moderate captures either side of it
(48,507 and 48,917). All three were replayed with `replay_spike_batch`, and
47,116 was additionally run through `attribute_spike_rows`. Together they
refine three of this note's earlier claims and point at a concrete,
already-implemented fix.

### 41.1 The three replays at a glance

| step | pre-clip | leader (norm) | `dc_ratio` | L0 to L3 h-grad | L0/L3 ratio | mean lr_term_share (banks 0-4) | fidelity gap |
|---|---|---|---|---|---|---|---|
| 47,116 | 13,139.5 | depth_code (8,611.4) | 1.50 | 0.0662, 0.0568, 0.0398, 0.0292 | 2.27x | 0.9993 / 0.9992 / 0.9986 / 0.9966 / 0.9964 | 0.0000092 percent |
| 48,507 | 203.1 | depth_code (128.2) | 1.35 | 0.0446, 0.0379, 0.0264, 0.0196 | 2.28x | 0.9993 / 0.9992 / 0.9984 / 0.9952 / 0.9952 | 0.0017 percent |
| 48,917 | 202.0 | reverse_channel_scale (132.3) | 0.94 | 19.15, 13.44, 6.39, 0.14 | 142x | 0.9993 / 0.9993 / 0.9984 / 0.9956 / 0.9959 | 0.0019 percent |

All three replayed at essentially perfect fidelity (worst case 0.0019
percent), so every number below is a bit-exact reproduction of what
actually happened at that step, not an approximation.

### 41.2 Finding 1: the low-rank channel is chronically dominant, not a transient excursion -- closing §33.1's own caveat, but not the way it expected

`lr_term_share` -- the fraction of $V_\theta$'s exponent contributed by
$B_kB_k^{\top}$ rather than $\mathrm{diag}(a_k)$ -- sits at 0.995-0.9999 in
every bank of every one of these three replays, whether the pre-clip norm
is 202 or 13,139. So does the exponent-occupancy structure: bank 0's live
fraction is 3.0e-5 at 47,116 and identically 3.0e-5 at 48,507 (a
65x-smaller event); its exponent tail is actually *slightly more* extreme
at 48,507 (minimum -197,368) than at 47,116 (minimum -174,097).

§33.1 flagged this exact gap in its own bracket measurement: a fixed
generic probe batch "does not by itself rule out a transient,
batch-specific $B_k$ excursion on the offending step." These three
replays are the first direct, batch-specific measurement of that
excursion -- and the answer is not "yes, transient": the low-rank channel
is not doing anything special at the crisis moment. It looks the same, at
the same order of severity, whether the resulting gradient is unremarkable
or catastrophic. This does not contradict §33's magnitude argument (its
bracket measured how much $\sigma_{\max}(B_k)^2$ shifts between
*checkpoints* on a generic batch, a different question); it says the term
§33 called "at most a weak correlate" is, on the batches that actually
matter, essentially the entire computation, all the time -- a standing
property of the current weights (`PRECISION_LR_MAX=None`, unbounded), not
something that switches on during a spike.

### 41.3 Finding 2: 47,116 and 48,507 are smooth-cascade-shaped, and 47,116 falsifies §38.3's working hypothesis outright

By both of the taxonomy's own discriminators -- `dc_ratio` (1.50, 1.35:
both comfortably under the 1.8 smooth-cascade ceiling of §38.7) and the
per-layer h-gradient ratio (2.27x, 2.28x: both *below* the smooth-cascade
floor of 2.6x recorded across the original five smooth replays) --
47,116 and 48,507 are smooth-cascade events, not localized ones.

§38.3 proposed, with an explicit falsification test attached, that the
smooth cascade "self-limits under per-group clipping... caps its own
contribution before the aggregate can compound much past ~400," and that
only the localized mode had ever crossed the 500 hard trigger, with the
caveat that "if a smooth-cascade event eventually also crosses 500... the
hypothesis needs revising." 47,116 is exactly that event: a smooth-shaped
cascade at pre-clip 13,139.5 -- not just past 500, but 26x past it, and
30x past the previous smooth-cascade ceiling of 432.8 (§38.1's table).
**§38.3's working hypothesis is falsified.** Both modes can cross the hard
trigger; the smooth cascade is not self-limiting, it is simply less
frequent at extreme severity.

The per-group ratios between 47,116 and 48,507 support one mechanism
scaling, not two: `depth_code` 67.2x, `E` 63.6x, `P` 60.6x, `creation_gate`
58.7x, `V_theta` 78.7x, `reverse_channel_scale` 31.9x -- all within a
factor of 2.5 of the aggregate's own 64.7x ratio. Whatever varies batch to
batch to produce this range, it varies the *same* chronic mechanism's
output roughly uniformly across every group it touches, rather than
turning on a qualitatively different one at the severe end.

**Caveat on the h-gradient magnitudes specifically (not the
classification).** Both captures have `grad_accum=4`, and the per-layer
hook in `replay_spike_batch` overwrites rather than accumulates across the
four microbatches' backward calls (`_layer_grad_norms[ell] = ...` on every
hook firing), so the reported profile reflects only the last of four
microbatches. The *classification* above does not depend on this being
exact -- the pre-clip total (13,139.5, 203.1) is measured independently
and exactly (0.002 percent fidelity gap) by the parameter-level sum, and
it is that number, not the per-layer hook, that falsifies §38.3. But the
h-gradient numbers themselves should be treated as indicative of shape,
not a trustworthy magnitude, until the hook is fixed to accumulate
(sum-of-squares or track-max) across microbatches -- see §41.7.

### 41.4 Finding 3: 48,917 is a genuine localized event, and `dc_ratio` alone would have missed it

48,917 tells the opposite story. Its per-layer profile -- 19.15, 13.44,
6.39 at layers 0-2, then a hard cliff to 0.14 at layer 3 (L0/L3 = 142x) --
sits squarely in the localized band (50-177x, §38.2), and unlike
47,116/48,507 these numbers are large enough that the grad_accum-overwrite
caveat above cannot explain them away as noise.

But its leading group is `reverse_channel_scale` (132.3), not `depth_code`
(124.3, a close second), and `dc_ratio` -- `depth_code`'s norm over the
*next-largest* group's norm -- comes out to 0.94, comfortably in the
smooth range by §38.7's rule. **`dc_ratio` classifies this event as
smooth. The per-layer profile says it is localized.** This is the first
disagreement between the two discriminators since `dc_ratio` was
discovered, and it happens precisely in the one case where the leading
group is not `depth_code`.

The parameter list confirms this is mechanistically distinct from
47,116/48,507: `reverse_ch.W_V_rev.weight` (20.1) appears in the top-12
for the first time across all replays to date. Combined with §33.3's own
count that `reverse_channel_scale` was present, if never leading, in 50
of 53 mined Phase-0 events, the working picture is now **two coexisting
mechanisms**, not one taxonomy with two shapes:

- **Mechanism A -- chronic $V_\theta$/low-rank stiffness (§41.2).**
  Present in every replay to date, feeds `depth_code`/`E`/`P`/
  `creation_gate` roughly proportionally to how much of the batch happens
  to sit in the razor-thin live band, and is what makes an ordinary
  smooth cascade capable of reaching 13,139.5 at 47,116.
- **Mechanism B -- an episodic reverse-channel-driven early-layer
  cascade.** A real, structured h-boundary gradient concentrated at
  layers 0-2, led by `reverse_channel_scale`/`W_V_rev`, which is what
  actually produces the L0-2 cliff -- i.e. plausibly what "localized" in
  §38 has been all along, independent of `depth_code`'s chronic
  background contribution.

Because `dc_ratio` is keyed specifically to `depth_code`, it is blind to
mechanism B whenever `reverse_channel_scale` leads instead of
`depth_code`. Given `dc_ratio` is the metric now logged every
`LOG_INTERVAL` step as the leading-indicator candidate (§38.7/§39.5), this
is a real gap: a future localized-mode event led by `reverse_channel_scale`
(48,917's own pattern) could climb toward a hard trigger without
`dc_ratio` ever leaving the "smooth" range. §41.7 proposes a fix.

### 41.5 Finding 4: `attribute_spike_rows` cannot be trusted on 47,116 -- the isolated-row replay reconstructs less than 0.03 percent of the real gradient

`attribute_spike_rows(47116)` ranked row (microbatch 3, row 2) as the top
single contributor to `depth_code`'s gradient (26.4 percent of the summed
row-norms, 8.4x the flat-batch baseline of 3.1 percent), with the same row
also leading `layer0_h_grad` and `total_grad_norm`.

That ranking cannot be trusted at face value. Summing the 32 isolated-row
gradients as vectors gives `norm_of_summed_grad = 2.35` for `depth_code`;
the actual full-batch value from `replay_spike_batch` is 8,611.4 -- a
3,663x gap. Neither documented caveat in `attribute_spike_rows` (RNG/noise
realism, which is moot with `LANGEVIN_T=0.0`; vector cancellation, which
only accounts for the 7.60-to-2.35 gap, about 3x) comes close to covering
this. The most likely explanation: `V_phi`'s top-k routing uses
Gumbel-softmax, which draws noise sized to the actual batch shape;
isolating one row to batch-size-1 shifts that row's noise realization
relative to what it experienced inside the batch of 8, and given
`lr_term_share`'s razor-thin live occupancy (as low as 3e-5, §41.2), a
different draw is more than enough to flip which handful of tokens land
in the numerically catastrophic regime.

**Implication.** `attribute_spike_rows`, as currently built, is unreliable
for attributing this specific (mechanism-A) failure mode to individual
rows -- isolating a row changes which computational regime it lands in,
not just which noise realization it sees. §41.7 proposes a fix.

### 41.6 A concrete, already-implemented fix: turn on `precision_lr_max`

`model_aniso_gaussian_vtheta.py` already implements exactly the cap
§41.2's finding calls for, and this run has it switched off:

```python
def _bound_lowrank(self, B: torch.Tensor) -> torch.Tensor:
    """Smoothly cap the low-rank factor's spectral norm (mitigation #2).
    Bounds sigma_max(B_k)^2 <= precision_lr_max via a tanh soft cap on the
    Frobenius norm of each well's factor -- identity for small norms,
    strictly below the budget for large ones, differentiable everywhere,
    never divides by zero. No-op when precision_lr_max is None."""
    if self._precision_lr_max is None or self.rank == 0:
        return B
    budget = self._precision_lr_max ** 0.5
    fro = B.flatten(-2, -1).norm(dim=-1).clamp(min=1e-12)
    scale = budget * torch.tanh(fro / budget) / fro
    return B * scale.unsqueeze(-1).unsqueeze(-1)
```

`PRECISION_LR_MAX = None` in Cell 0 leaves it a no-op for this entire run;
`precision_max` (the *diagonal* term's hard cap, $2/d\approx0.0052$) is
active, but the off-diagonal channel §41.2 just found responsible for
essentially the entire exponent has been unbounded throughout.

This directly targets both symptoms found above: the exponent blow-up
($\lVert B^{\top}\mathrm{diff}\rVert^2$ unbounded) and the
parameter-gradient blow-up (d(exponent)/dB scales with B itself, so an
unbounded B lets even a barely-alive token produce an enormous
`B_proj`/`mu_proj` gradient). Because the cap is applied to `B_proj(xi)`'s
*output* at forward time, not to the stored weights, it can be enabled on
the live model and resumed from any existing checkpoint immediately -- no
retraining from scratch and no checkpoint surgery, though Adam's
per-parameter step-size history for `B_proj`/`mu_proj` will have adapted
to the old, much-larger gradients and may need a few hundred steps to
re-equilibrate.

**Suggested starting point.** `baoab_cfc`'s explicit off-diagonal kick has
an $\omega \Delta t \lt 2$ stability wall (§29); with $\Delta t\approx1$ that is
$\sigma_{\max}(B_k)\le2$, i.e. `precision_lr_max` $\approx4$. Start
tighter for margin -- `precision_lr_max = 1.0` -- and relax if PPL
suffers; the exponents actually observed (`lr_term` up to roughly 400,000
against a `diff` norm on the order of a few hundred, implying an effective
$\sigma_{\max}(B_k)^2$ in the thousands) are so far past any reasonable
value that even a loose cap is a drastic tightening from the current,
effectively-unbounded state.

**How this relates to §40.** §40 asks whether `baoab_cfc_lowrank` (an
*integrator* choice: exact propagation of whatever $B_k$ the model has)
addresses the localized mode. `precision_lr_max` is a different,
complementary lever: it changes the *model* (bounds how sharp a well can
ever get), applies regardless of integrator, costs nothing extra per step
(it is already in the forward pass, just currently a no-op), and is now
backed by direct evidence from all three of this section's replays, not
just the theoretical argument of §28-§29. It is not expected to touch
mechanism B (§41.4's reverse-channel cascade) at all -- that needs its own
lever.

### 41.7 Status and next steps

1. **Offline validation, no live run needed. Status: implemented (31
   August 2026), `Cell 6d`, `replay_precision_cap_ablation(step_tag,
   budgets=(1.0, 4.0, None))`.** Reuses the existing `*_spikebatch.pt`
   bundles, mirrors `replay_integrator_ablation`'s snapshot/restore
   invariant, and swaps `bank._precision_lr_max` across arms (a plain
   Python attribute `_bound_lowrank` reads fresh on every forward call, so
   no new checkpoint is needed) instead of `cfg.integrator`. Reports, per
   budget, the matching-groups pre-clip norm, per-group breakdown,
   per-layer $h$-gradient profile, and each bank's exponent minimum --
   the last of these is the direct check that a given budget is actually
   biting. Syntax-checked (`ast.parse` on every notebook cell); not yet
   run against a live GPU session with real captures.
2. **Fix the per-layer hook to accumulate, not overwrite, across
   `grad_accum` microbatches** (sum-of-squares or track-max) in
   `replay_spike_batch` / `attribute_spike_rows` -- needed before
   trusting per-layer *magnitudes* (not shapes) on any `grad_accum>1`
   capture, including a re-check of whether this changes anything about
   the original seven §38 replays.
3. **Investigate mechanism B (reverse-channel-led localized cascade) on
   its own track.** Add a `reverse_ch`-side stiffness proxy (spectral
   norm of `W_V_rev.weight` or similar) to Cell 6's `LOG_INTERVAL`
   logging, the same way `b_proj_sigma_max` tracks mechanism A, and widen
   the Phase-0 leading-indicator logic beyond `dc_ratio` alone (e.g. also
   track `reverse_channel_scale`'s own group norm against its post-warmup
   baseline) so a mechanism-B event does not hide from the cheap per-step
   monitor the way 48,917 just did.
4. **Do not trust `attribute_spike_rows`'s row ranking on mechanism-A
   events as literal attribution.** A more surgical tool -- replay the
   full 8-row microbatch unchanged but zero out 7 of the 8 rows' *loss*
   contribution before `.backward()`, rather than isolating a row to
   batch-size-1 -- would preserve batch-shape-dependent RNG/routing
   behaviour and give a trustworthy per-row breakdown; not yet built.
5. **If step 1 validates, resume live training with `PRECISION_LR_MAX` set
   to the chosen value.** No checkpoint surgery required.

## 42. Step 1 Validated: `precision_lr_max` and `baoab_cfc_lowrank` Both Collapse All Three Replays, a Hook Bug and a Checkpoint-Loading Pitfall Found Along the Way, and the Cap Switched On

§41.7 item 1's offline ablation ran the same evening against all three
captures (47,116 / 48,507 / 48,917), alongside `replay_integrator_ablation`
(§40) on the same three. The result is unambiguous, and one part of it is
a genuine surprise that reframes §41.4's two-mechanism picture.

### 42.1 A latent bug in both new ablation helpers, found before any real data came out

Both `replay_precision_cap_ablation` and `replay_integrator_ablation`
crashed on first use with `TypeError: expected Variable, but hook returned
'float'`. Their per-layer hook was written as

```python
h_new.register_hook(
    lambda g, li=layer_idx: _layer_grad.setdefault(
        li, float(g.detach().norm())))
```

`dict.setdefault(key, value)` *returns* the resulting value (the new one,
or the existing one if `key` was already set) -- unlike the
no-`return`-statement hook in `replay_spike_batch`'s `_make_layer_hook`
that this was modelled on. PyTorch treats any non-`None` value returned
from a tensor hook as a proposed replacement gradient; a bare `float`
fails that check immediately. Fixed by moving the `setdefault` call into a
named function's body as a statement, so it returns `None`:

```python
def _layer_hook(g, li=layer_idx):
    _layer_grad.setdefault(li, float(g.detach().norm()))
h_new.register_hook(_layer_hook)
```

Semantics (first microbatch per budget/arm wins, per §41.3's caveat) are
unchanged; only the crash is gone. Worth remembering for any future
hook-based instrumentation in this notebook: a tensor hook must return
`None` or an actual replacement `Tensor`, never a bookkeeping value.

### 42.2 The ablation result: both budgets, and the low-rank integrator, collapse all three replays -- including the reverse-channel-led one

| step | recorded pre-clip | replayed (as trained) | `precision_lr_max=1.0` | `precision_lr_max=4.0` | `baoab_cfc_lowrank` L0-2 |
|---|---|---|---|---|---|
| 47,116 | 13,139.5 | 13,033.0 | 3.98 | 4.09 | 2.65 |
| 48,507 | 203.1 | 203.9 | 2.14 | 2.95 | 1.42 |
| 48,917 | 202.0 | 199.6 | 2.79 | 1.90 | 1.34 |

All three "as trained" replays reproduce the recorded pre-clip norm within
about 1 percent, so this is trustworthy, not an artefact of drift between
capture and replay. Both `precision_lr_max` budgets and the low-rank
integrator collapse every one of the three captures down to a pre-clip
norm of about 1-4 -- squarely in quiet-step territory -- regardless of
whether the starting severity was 202 or 13,139.5.

**The surprise is 48,917.** §41.4 classified it as mechanism B -- an
episodic, `reverse_channel_scale`-led cascade, mechanistically distinct
from mechanism A's chronic $V_\theta$ stiffness, on the grounds that
neither `precision_lr_max` nor the low-rank integrator touches
`reverse_channel_scale` or `reverse_ch` at all. Yet both fully suppress
it too. The leading groups collapse in lockstep even though only the
$V_\theta$ low-rank term was touched:

| group | uncapped | `precision_lr_max=1.0` | ratio |
|---|---|---|---|
| `reverse_channel_scale` | 130.7 | 1.94 | 67x |
| `depth_code` | 122.8 | 0.87 | 141x |

**Finding: mechanism B rides on mechanism A, it is not independent of
it.** The most coherent reading is that `reverse_channel_scale`'s cascade
was never an independent trigger -- it was amplifying the same
catastrophically sharp $V_\theta$ force mechanism A produces. Once that
force is bounded, whatever propagates through `reverse_ch` in the early
layers has nothing extreme left to amplify. §41.4's "two coexisting
mechanisms" framing should be read as two *symptoms* of one root cause
(unbounded low-rank curvature), not two independent failure modes that
happen to share a coincidental remediation.

Per-bank exponent minima confirm the cap is doing exactly what §41.6
predicted, at both budgets, e.g. at 47,116: uncapped bank minima range
from -257,630 to -150,076 (numerically annihilated); at
`precision_lr_max=1.0` they range from -146.8 to -51.8; at
`precision_lr_max=4.0`, a looser cap as expected, from -579.3 to -186.6 --
both drastic improvements, with `1.0` giving a firmer margin.

### 42.3 Corroborating evidence: the low-rank integrator's own SVD chokes on the uncapped matrices

`replay_integrator_ablation`'s `baoab_cfc_lowrank` arm triggered a PyTorch
warning during the 47,116 replay: 119 of its batched $B_k$ matrices (5
named plus "114 batches" more) failed to converge under the standard
cuSOLVER driver and fell back to a slower, more accurate method. This is
independent numerical evidence, from a completely different code path
(exact SVD rather than the exponent computation), that the raw
(uncapped) $B_k$ at this snapshot are severely ill-conditioned -- not
just numerically large in the exponent sense, but pathological enough
that a generic dense-SVD routine chokes on them. It is also a fresh, very
concrete illustration of why `baoab_cfc_lowrank` remains
production-infeasible (§34) even though it is the single best-performing
arm in the table above: it is exposed to these same matrices via a more
expensive, less numerically robust code path.

### 42.4 Bracketing $\sigma_{\max}(B_k)^2$ directly: healthy and spike-regime checkpoints look statistically similar under a neutral batch

A companion tool, `bracket_precision_lr_max` (`Cell 6b-3`), implements
§31.2-§31.4's bracketing protocol properly: rather than reading
`sigma_max(B_k)^2` off "whatever's currently loaded," it loads the
healthy `_best.pt` (step 27,000) and each spike bundle's own
`model_state_dict` in turn, on a fixed generic batch, restoring the live
model afterward.

| state | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| healthy (step 27,000) | 283.2 | 669.9 | 1054.0 | 2329.5 | 6362.1 |
| spike step 47,116 | 310.6 | 714.0 | 1136.1 | 2900.6 | 8065.6 |
| spike step 48,507 | 284.2 | 683.4 | 1058.3 | 2305.3 | 6386.8 |
| spike step 48,917 | 281.6 | 687.4 | 1054.9 | 2426.9 | 6682.8 |

**Finding: on a neutral batch, all four states look almost the same.**
This refines §33's own bracket (which also found the shift between
checkpoints "modest") with a wider, four-way comparison: the ambient
sigma_max(B_k)^2 distribution has apparently been elevated since at
least step 27,000 and stayed roughly flat since, rather than escalating
toward the spikes. §31.4's own rule -- "set the budget above the healthy
row's p90-p99, below the spike rows' p99/max" -- does not resolve cleanly
here, because the healthy and spike percentile bands overlap almost
entirely.

This means there is no tight, tail-only budget available: §41.2's
"chronic, not transient" finding already implied this, and this
four-checkpoint bracket confirms it directly. What actually
distinguishes a quiet forward pass from a catastrophic one is not
`sigma_max(B_k)^2` alone but its *product* with $\lVert h - \mu_k
\rVert^2$ on the specific tokens in play -- which is exactly why the
direct ablation in §42.2 (replaying the real offending batch) is more
decisive evidence than this bracket, and why the chosen budget (§42.6)
follows the ablation, not the bracket.

### 42.5 A checkpoint-loading pitfall: Cell 5 never loads a checkpoint, and an over-eager interrupt on Cell 6 can silently leave weights at random init

While chasing an apparent contradiction (`Cell 6b-2`'s live-model reading
of `sigma_max(B_k)^2` was 400-600x smaller than the same quantity measured
moments later on the same nominal checkpoint via `bracket_precision_lr_max`),
the actual root cause turned out to be a workflow hazard, not a
measurement bug. `Cell 5`'s own `_rebuild_model()` docstring says so
explicitly: *"Weights are still random at this point (the training cell
loads any checkpoint), so this does not discard trained state."* The
checkpoint load is a top-level block inside `Cell 6`, executed
synchronously before `run_training()` is even called:

```python
if resume_ckpt is not None and resume_step < TOTAL_STEPS:
    ckpt_data = torch.load(resume_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt_data['model_state_dict'], strict=False)
    ...
```

`torch.load` on a large checkpoint over Google Drive's FUSE mount is not
instantaneous. If `Cell 6` is interrupted immediately after being run
(the documented pattern for defining its globals without committing to a
long training run, §37), the interrupt can land *before* this block
finishes, leaving `model` on fresh random-init weights while still
printing as if the session were live. A direct parameter-norm check
caught this cleanly:

```python
_live_w = dict(model.named_parameters())['V_theta.bank.banks.0.B_proj.weight']
_bundle_w = bundle['model_state_dict']['V_theta.bank.banks.0.B_proj.weight']
# broken session: live norm 21.72 vs bundle norm 118.90 (random-init magnitude)
# after waiting for the "Model loaded." print before interrupting:
# live norm 118.9039 vs bundle norm 118.8997 (diff 0.0012 -- correct)
```

`bracket_precision_lr_max`'s own four measurements were never affected --
it reloads each state's weights from disk internally, independent of
whatever `model` held beforehand -- so §42.4's table stands as reported.
Only a bare, no-reload reading of "the live model" was ever at risk.

**Operational rule going forward.** Before calling
`run_training(next_step, TOTAL_STEPS)` to resume real training, always
run a quick parameter-norm check like the one above against the
checkpoint that was supposed to load. Silently resuming from random-init
weights would be a far worse failure than anything else diagnosed in this
note, and the only symptom is a live-model diagnostic reading that looks
implausibly tame -- exactly what happened here before the cause was
understood.

### 42.6 The decision: `PRECISION_LR_MAX = 1.0`, switched on, training resumed from step 47,121

Both evidenced-safe budgets (`1.0` and `4.0`) fully suppress all three
replays; `1.0` was chosen as the live value: it is the more conservative
of the two (firmer exponent-tail margin, §42.2), and `_precision_lr_max`
is a live, hot-swappable Python attribute rather than part of
`state_dict()`, so it is cheap to loosen later and expensive to have
picked too loose a value now (another catastrophic reload). `Cell 0`'s
default was updated from `None` to `1.0` for any future fresh session,
and the live session (already confirmed, via §42.5's check, to be
correctly holding the step-47,116 weights) was patched in place:

```python
for b in model.V_theta.banks:
    b._precision_lr_max = 1.0
next_step = run_training(47121, TOTAL_STEPS)
```

No checkpoint surgery, no rebuild, no restart -- exactly as §41.6
anticipated. `PRECISION_LR_MAX` is not expected to touch anything the
low-rank $V_\theta$ channel does not feed (§42.2's finding says that is,
in practice, everything captured to date, mechanism B included), so this
single change is the resumption strategy in full, pending the monitoring
in §42.7.

### 42.7 Status and next steps

1. **Done.** `replay_precision_cap_ablation` / `replay_integrator_ablation`
   hook bug fixed (§42.1); both validated against all three captures
   (§42.2); `bracket_precision_lr_max` added and run (§42.4); the
   checkpoint-loading pitfall found and worked around (§42.5);
   `PRECISION_LR_MAX = 1.0` switched on in both `Cell 0`'s default and the
   live session; training resumed from step 47,121.
2. **Monitor the resumed run.** Watch `dc_ratio` and `b_proj_sigma_max` in
   the periodic log for a plateau rather than continued growth, watch for
   the absence of further `[watchdog-hard]` reloads near the old
   47,116-49,095 territory, and watch val_ppl against the restored best
   of 100.47 (step 27,000) to confirm the cap is not visibly hampering
   learning. Loosen to `4.0` if it is; no rebuild needed either way.
3. **§41.7 items 2-4 remain open**: the per-layer hook's overwrite-not-accumulate
   behaviour across `grad_accum` microbatches, a `reverse_ch`-side
   stiffness proxy now that §42.2 suggests it is downstream of mechanism
   A rather than an independent trigger, and a surgical row-attribution
   fix for `attribute_spike_rows`.
4. **Fix the two hook-registration call sites' habit of re-registering a
   hook on every microbatch's forward** (`replay_precision_cap_ablation`
   and `replay_integrator_ablation` both do this, inherited from
   `replay_spike_batch`'s pattern) if a future variant needs to
   distinguish per-microbatch contributions rather than "first microbatch
   per arm wins."

---

## 43. Four Repeats of the Same Eval-Time OOM at Step 47,500: `gc.collect()` Was Never Going to Fix It, and Why

Resuming training after §42.6 hit a **fourth** `CUDA OutOfMemoryError`, at
the identical step (~47,500) and inside the identical call
(`evaluate()`'s first iteration), across four independent attempts —
including one made *after* patching `evaluate()` to call `gc.collect()` +
`torch.cuda.empty_cache()` on every iteration (not just every 10, the
weaker version that shipped first). The live per-iteration memory print
added alongside that patch read:

```
[evaluate] iter 0/40  mem_alloc=47.47GB  mem_resv=48.12GB
```

bit-for-bit identical, to two decimal places, both before and after the
`gc.collect()`-every-iteration change. That identity is the whole
diagnosis: if `mem_alloc` were a garbage-collectible Python reference
cycle (the theory the fix was built on), adding `gc.collect()` would have
to change the number. It changed nothing, which means the memory was
never Python garbage in the first place — it was live, validly
referenced, un-freed CUDA tensor data that no amount of `del` +
`gc.collect()` + `empty_cache()` was ever going to touch.

### 43.1 The `create_graph=True`-in-eval theory was falsified by the code itself

The fix that shipped with the original OOM patch assumed the CfC-BAOAB
analytic $V_\theta$ force "almost certainly" builds a `create_graph=True`
second-order graph even at eval time, and that `gc.collect()` is the
documented remedy for the reference cycle such graphs leave behind. Both
halves turn out to be checkable directly against `MultiXiPARFLM._layer_forces`
in `model_parf_multixi.py` — the class this run actually instantiates
(`FockMultiXiPARFLM` on top of it), *not* the superficially similar but
structurally different, unused `model_parf.PARFLM._layer_forces` (§42's
`Cell 5` regression guard exists specifically to catch that
stale-reference trap). `MultiXiPARFLM._layer_forces` contains the *only*
`torch.autograd.grad` call in the analytic-$V_\theta$ path (the one that
recovers $\nabla_h V_\phi$, since $V_\phi$ has no closed form):

```336:355:notebooks/conservative_arch/parf/model_parf_multixi.py
        if self._use_analytic_vtheta():
            # Closed-form V_theta force: no autograd, so V_theta never
            # enters the second-order create_graph chain at all.  Only the
            # (much smaller) V_φ graph is differentiated twice.
            if vtheta_comps is None:
                f_theta = -self.V_theta.analytical_grad(xis, h_in)
            else:
                f_theta = -self.V_theta.analytical_grad(
                    xis, h_in, comps=vtheta_comps,
                )
            with torch.autocast(device_type="cuda", enabled=False):
                grad_phi, = torch.autograd.grad(
                    U_pair.float(), h_in,
                    create_graph=self.training,
                    retain_graph=self.training,
                )
            f_phi = -grad_phi
```

`evaluate()` calls `model.eval()` first, so `self.training` is `False`
for the whole eval pass — `create_graph=False` for this call, exactly as
the comment two lines above it says is the point ("V_theta never enters
the second-order create_graph chain at all"). There is no hidden
`create_graph=True` anywhere in this call. `gc.collect()` had nothing
collectible to find, which is the direct, mechanical reason the number
never moved.

### 43.2 The real graph that survives eval: the analytic force itself, not the `V_phi` grad call

`f_theta = -self.V_theta.analytical_grad(xis, h_in)` is **not** a discrete
`autograd.grad()` call — it is ordinary differentiable tensor arithmetic
(the closed-form Gaussian-well gradient, built from matmuls against
$V_\theta$'s own parameters and against `h_in`). Ordinary tensor ops are
never gated by a `create_graph` flag; they stay connected to whatever
they were computed from — `h_in` and $V_\theta$'s parameters — for as
long as grad-tracking is enabled, in train mode or eval mode alike. Since
`h_new` of layer $\ell$ feeds `h_in` of layer $\ell+1$, this graph chains
across all $L$ layers:

```mermaid
flowchart LR
    H0["h in, layer 0"]
    FT0["f theta, layer 0, closed form gradient"]
    FP0["f phi, layer 0, detached since create graph is false"]
    HN0["h new, layer 0"]
    H1["h in, layer 1"]
    FT1["f theta, layer 1"]
    HN1["h new, layers 1 through 7"]
    HL["h at layer L"]
    LOSS["loss"]
    STUCK["checkpoint segments never freed"]

    subgraph EvalForward [One evaluate forward pass, grad enabled]
        H0
        FT0
        FP0
        HN0
        H1
        FT1
        HN1
        HL
        LOSS
    end

    H0 --> FT0
    H0 --> FP0
    FT0 --> HN0
    FP0 --> HN0
    HN0 --> H1
    H1 --> FT1
    FT1 --> HN1
    HN1 --> HL
    HL --> LOSS
    LOSS -->|no backward ever called| STUCK
```

`use_layer_checkpoint=True` (set in `Cell 5`'s `make_config`) wraps each
layer in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`,
correctly gated on `torch.is_grad_enabled()` rather than `self.training`
(§42's own regression guard in `Cell 5` checks exactly this, and it
passed — this is *not* a repeat of that old bug). Checkpointing's promise
is "discard intermediates now, recompute them later if a backward ever
walks through here." During training, `loss.backward()` immediately
follows the forward and that promise is redeemed layer by layer: each
checkpoint segment is recomputed, differentiated, and freed as backward
walks from layer $L-1$ down to layer $0$. That is exactly why the
training loop's own periodic log line —

```
step 47500/100000  ...  mem_alloc=1.7GB  mem_resv=57.6GB  mem_peak=57.2GB  ...
```

— shows `mem_peak=57.2GB` (the transient forward+backward cost of one
`batch=8` microbatch, before backward frees it) alongside
`mem_alloc=1.7GB` (the steady-state cost of parameters + optimiser state
+ populated `.grad`, measured *after* backward has already run). Eval
reaches the same order of peak — `47.47GB` at `batch=8`, comparable to
training's `57.2GB` peak at the same microbatch size — for the same
underlying reason (the same per-layer analytic-force graph, at the same
batch size), but `evaluate()` never calls `.backward()`. The checkpoint
segments' recompute obligation is never redeemed, the per-layer graph
that chains all the way to `loss` is never walked, and the memory is
simply never reclaimed. No Python object leaks; `del` + `gc.collect()`
correctly find nothing to do, because from Python's perspective the
graph rooted at `loss` is exactly as reachable, and exactly as intended
to be reachable, as it always is until something actually backpropagates
through it.

### 43.3 Fix: give eval a real (but weight-inert) `backward()`

The one-line fix is to let PyTorch's own teardown mechanism run, exactly
as it does during training, then throw the resulting gradients away
before they can do anything:

```python
with torch.enable_grad():
    _, loss = model(x, y)
    losses.append(loss.item())
    loss.backward()
model.zero_grad(set_to_none=True)   # discard; optimizer.step() is never called
del loss, x, y
gc.collect()
torch.cuda.empty_cache()
```

`model.eval()` is still in effect (`self.training` stays `False`), so
`create_graph`/`retain_graph` for the `V_phi` grad call are still `False`
and nothing here is any more expensive than a normal training step's
forward+backward at the same batch size — the exact regime the training
loop already survives every step. One correctness footnote: because
`grad_phi` is detached (`create_graph=False`), `V_\phi`'s own parameters
receive no gradient contribution through this path during eval's
backward — harmless here, since every eval gradient is discarded
immediately and never touches the optimiser.

### 43.4 Status

Patched in `Cell 6`'s `evaluate()` in
`colab_fock_cfc_baoab_aniso_gaussian_openwebtext_d384.ipynb`. Not yet
re-validated live (the session that surfaced the fourth OOM had already
been interrupted at step 47,500 with no manual checkpoint saved, so the
next `run_training(47500, TOTAL_STEPS)` call — or a fresh-session resume,
which per §42.5/42.6 will resolve to the `_prereload`/`_manual` checkpoint
at or near step 47,116/47,121 — is the first real test of this fix).
Watch the first post-fix `evaluate()` call's `[evaluate] iter .../40
mem_alloc=...` trend line: it should now stay flat near the pre-eval
training baseline (`mem_alloc≈1.7GB` scale) rather than jumping to the
`~47GB` scale at iteration 0.

---

Companion note to `Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md`.
The CfC/BAOAB propagator is implemented in
`notebooks/conservative_arch/parf/cfc_baoab.py` and wired into the layer step
in `notebooks/conservative_arch/parf/model_parf_multixi.py`; the production
training notebook is
`notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_aniso_gaussian_openwebtext_d384.ipynb`.
The anisotropic Gaussian V_theta is in
`notebooks/conservative_arch/parf/model_aniso_gaussian_vtheta.py`, and the
SCAF stiffness audit (Phase 7b/7c Weyl bound) in the `stiffness_audit` branch
of `semsimula-scaf` (`src/scaf/probes/stiffness.py`).

Last updated: 6 September 2026 (adds §43: a fourth repeat of the
step-47,500 eval-time OOM, this time *after* patching `evaluate()` to
call `gc.collect()` every iteration, hit the identical `mem_alloc=47.47GB`
at iteration 0, falsifying the "uncollected `create_graph=True` reference
cycle" theory behind that patch -- the code confirms `create_graph` is
correctly gated `False` in eval. The real cause is the analytic
$V_\theta$ force's ordinary (non-`autograd.grad`, hence
create_graph-flag-immune) tensor graph chaining across all $L$
checkpointed layers, which only gets torn down by an actual
`.backward()` call -- exactly what training does every step and eval
never does. Fix: give `evaluate()` a real, weight-inert
`loss.backward()` + `zero_grad(set_to_none=True)` per iteration so
PyTorch's own checkpoint-teardown machinery runs; not yet re-validated
live). Previously updated 5 September 2026 (adds §42: the §41.7 item-1 ablation ran
against all three captures and validates `precision_lr_max` at both
1.0 and 4.0, and `baoab_cfc_lowrank`, against all of them -- including
the reverse-channel-led 48,917, which implies mechanism B rides on
mechanism A rather than being independent of it; fixes a hook-return-value
bug in the two new ablation helpers; adds `bracket_precision_lr_max` and
finds healthy and spike-regime checkpoints look statistically similar
under a neutral batch; finds and works around a checkpoint-loading
pitfall where `Cell 5` never loads a checkpoint and an over-eager
interrupt on `Cell 6` can silently leave weights at random init; switches
`PRECISION_LR_MAX` on, at 1.0, in both `Cell 0`'s default and the live
session, and resumes training from step 47,121). Previously updated 31
August 2026 (§41.7 item 1 implemented: `Cell 6d`'s
`replay_precision_cap_ablation(step_tag, budgets=(1.0, 4.0, None))`,
mirroring §40's `replay_integrator_ablation` but swapping
`bank._precision_lr_max` across arms; not yet run against a live GPU
session). Previously updated 31 August 2026 (adds §41: three new replays -- steps 47,116,
48,507, 48,917 -- show the low-rank channel is chronically, not
transiently, dominant across every capture, refining §33.1's own caveat;
falsify §38.3's smooth-cascade self-limiting hypothesis outright (47,116
crosses the hard trigger 26x over while smooth-shaped); expose a `dc_ratio`
blind spot for a reverse-channel-led second mechanism at 48,917; flag
`attribute_spike_rows` as unreliable on this failure mode; and motivate
turning on the already-implemented `precision_lr_max` cap, with a starting
value and an offline ablation plan). Previously updated 31 August 2026
(adds §40: does `baoab_cfc_lowrank` address the
localized mode; reconciles §33's negative verdict, which was measured against
a structurally different, earlier crisis signature, with the new
layer-0-2-cliff taxonomy, and proposes a targeted `lowrank_layers={0,1,2}`
offline ablation test against the existing spikebatch bundles rather than a
full-cost live retraining trial). Previously updated 31 August 2026 (adds §39: two forward-pass hypotheses for
the localized mode's mechanism -- a token-minority driving `depth_code`'s
direction, and denser `V_theta` well occupancy -- were each designed
before being tested and both came back falsified against all four
Phase-1/2-instrumented events with a recorded layer profile; per-row
attribution shows the localized events are the *flattest* across rows,
the opposite of the conjecture, with a clean monotonic anti-correlation
between layer-0-2 severity and row concentration; exponent occupancy
shows no separation at all, and additionally reveals over 99.9% of
token-well pairs are numerically dead in every capture regardless of
mode. Revises the working picture: the localized mode is layer-localized
but batch-wide, not batch-specific, reopening the `sigma_max(B_k)`
precision-matrix stiffness line in sharper, trajectory-based form. Next
step: add a weight-space stiffness proxy to Cell 6's periodic logging,
same pattern as `dc_ratio`.)

Previously updated 30 August 2026 (adds §38.7: the §38.6 deep-dive
against the two localized captures, and a matching control-group replay
of two smooth-cascade captures, found that the layer-resolved activation
extremes and `set_fock_capture` gate-health table are statistically
indistinguishable between modes (a property of current model weights, not
batch/outcome) and token degeneracy is ruled out entirely -- but a number
already present in the original 7-event Phase-0 data, `dc_ratio`
(`depth_code`'s group norm over the next-largest group's), splits cleanly:
<1.8 for every smooth-cascade event, >2.2 for both localized events.
Corrects §38.6: `V_theta` banks don't share the per-layer-overwrite bug,
they're called once per forward pass on the depth-stacked `xi` tensor, now
reported at `layer=-1`. `dc_ratio` logging added to Cell 6's periodic log
line so a future mining pass can check whether it leads a hard-trigger or
only coincides with one).
Previously updated the same day (adds §38.6: rather than waiting on
new localized captures, Cell 6d's replay was deepened to re-interrogate
the two already-on-disk localized bundles (steps 39,983/41,837) for free
-- fixes a real bug where `activation_extremes` silently only ever
reflected the LAST layer (7) for `creation_gate_qkv`/`reverse_ch`/`V_theta`
banks, since they're shared modules called once per layer and the old hook
overwrote its own dict key each time; wires in the model's own
`set_fock_capture` per-layer register/gate health buffer; and adds
`inspect_spike_tokens()` to decode and rank the offending microbatch's
tokens by degeneracy, independent of the gradient side entirely).
Previously updated the same day (adds §38: with 7 spike replays now
on hand -- including this run's first genuine `watchdog-hard` reload,
step 41,837 at 528.0 pre-clip -- the per-layer $h$-gradient profile splits
cleanly into two failure modes, a smooth 8-layer cascade (5 of 7 events,
layer0/layer3 ratio 2.6-6.3x) and a localized layer-0-2 blowup with a sharp
cliff by layer 3 (2 of 7, ratio 50x and 177x); leadership by parameter
group does not discriminate the two modes (`depth_code` leads one of each),
but only the localized mode has crossed 500 so far, giving a working
(n=2, not yet confirmed) hypothesis that it -- not the self-limiting smooth
cascade -- is what actually forces this run's rare reloads. Previously
updated the same day, earlier (adds §37: Cell 6's training loop is
now a real function, `run_training(start_step, total_steps)`, instead of a
bare top-level `for` loop -- interrupting it to inspect spike captures no
longer OOMs the next `replay_all_captures()` call (per-step locals now die
with the function's stack frame instead of lingering as notebook globals)
and no longer requires a checkpoint-save/restart/override dance to resume
(`run_training(next_step, TOTAL_STEPS)` just picks up where it left off,
since its accumulator/watchdog state is `global` and initialized once
outside the function). `evaluate()` got a `try/finally` so an interrupt
mid-eval can't leave `model` stuck in eval mode. `assign_clip_group` /
`per_group_grad_norms` / `clip_grads_per_group` moved out to a new
`grad_clip_utils.py` module (with its own unit tests), the first
Cell-6 functionality actually pushed to a module rather than left inline;
Cell 6d moved before Cell 6 in the notebook's cell order now that its
functions no longer need Cell 6 to have run first just to be defined.
Same day, confirmed live on the run: `replay_all_captures()` had the
identical leak for the identical reason -- `except Exception` doesn't
catch `KeyboardInterrupt` (a `BaseException`), so an interrupt landing
mid-replay pinned ~80GB via `sys.last_traceback` just like an interrupted
training step did; fixed with an explicit `except KeyboardInterrupt:`
that stops the loop instead of letting it escape uncaught).
Previously updated 30 August 2026, later (adds §36: after §35's fix,
`GRAD_NORM_HARD_TRIGGER` was restored to 500.0 and production resumed --
over the next ≈1,800 steps (37,501-39,350) zero hard reloads fired, yet
the run stayed stuck on the same ~105 plateau, with four ordinary
100-450 `[spike]` events in that window. Since Phase 1 capture had been
gated on the same `GRAD_NORM_HARD_TRIGGER` as the reload, it could never
harvest that far more frequent moderate band. Fix: a new, independent
`CAPTURE_SPIKE_THRESHOLD=200.0` that gates capture without touching the
reload condition, `SPIKEBATCH_SNAPSHOT_MAX_KEEP` raised 5->12, and a new
`replay_all_captures()` helper in Cell 6d that replays every captured
`_spikebatch.pt` and prints one summary attribution table instead of one
call per capture. Reframes the diagnostic goal from "explain the rare
catastrophic reload" to "explain the plateau," with the caveat that the
moderate band's causal role vs. the not-yet-decayed WSD learning-rate
schedule is still open, to be discriminated once captures accumulate).
Previously updated 30 August 2026 (adds §35: fixed a `TypeError` in
`replay_spike_batch` -- `torch.load`'s `map_location=DEVICE` was remapping
`rng_state_cpu`/`rng_state_cuda` off the CPU, which `torch.set_rng_state()`
rejects; fixed via `map_location='cpu'`. With the fix, the Stage-1 smoke
test's first capture (step 37,763, pre-clip 160.4) replayed with a
0.0012% fidelity gap -- Phase 1/2 fully validated end to end. The
per-layer $h$-gradient profile came back monotone across all 8 layers
(≈37x growth from layer 7 to layer 0, no isolated bump), matching §33.4's
cascade-amplification signature rather than a single-layer cause;
`depth_code` led the per-parameter attribution narrowly ahead of `E`/`P`.
Flagged as a first look, not a closed case, since this event sits in the
"<200" band rather than the `E`/`P`-dominated severe regime; next step is
the same replay against a genuine ≥500 `watchdog_hard_reload`).
Previously updated 30 August 2026 (adds §34: `baoab_cfc_lowrank` was run
end-to-end at the live L=8/$d$=384/OWT scale from the §32 checkpoint,
surfacing and fixing three bugs — a `CheckpointError` from
`torch.svd_lowrank`'s non-deterministic branching under gradient
checkpointing (fixed via the new branch-free `_randomised_svd_det`), NaN
modes from a degenerate GPU SVD poisoning the gradient (fixed via
`torch.where` masking + `nan_to_num`), and an infinite backward in
`harmonic_terms_lowrank`'s `sqrt(g)` at $g=0$ (fixed via the analytic
$\sqrt w\exp(-e/4)$ form) — before measuring ≈120 s/step at full width and
≈50 s/step even restricted to 2 layers (vs. ≈10-15 s/step for
`baoab_cfc`), which would take ≈36 days to finish the run. Verdict: correct
and stable but not production-feasible at this scale, and independently
aimed at a quantity §33 already showed to be a weak driver; the code is
retained for future/smaller-scale use but production reverts to
`baoab_cfc`, resuming the §32 trajectory. `cfc_baoab.py`'s module
docstring and `model_parf_multixi.PARFConfig.integrator`'s docstring now
carry the same status note). Previously updated 28 August 2026, latest
(§33.3 Phase 0 done: mined the L=8
run's `training_log.jsonl` (964 lines, steps 50-40,900, downloaded from
Drive) for `grad_spike`/`watchdog_hard_reload` events. `depth_code` leads
76% of smaller (pre-clip < 200) spikes but 0% of the two severe hard
triggers, where `E`/`P` (token/positional embedding, tied via
$h_0 = E(x) + P$) lead 100% of the time, agreeing to 4 significant figures
at the larger trigger -- read together as one cascade-amplification
mechanism crossing a severity threshold, not two independent culprits.
§33.3/§33.4/§33.5 updated with the finding, the E/P mechanism (via
`model_parf.py`), and the resulting per-layer-growth prediction for Phase
2 to test against the run's next hard trigger). Previously updated 28
August 2026, later night (implements §33.3 Phases 1 and
2 in `colab_fock_cfc_baoab_aniso_gaussian_openwebtext_d384.ipynb`: `Cell 6`
gains `CAPTURE_SPIKE_BATCH`/`SPIKEBATCH_SNAPSHOT_MAX_KEEP` plus a
pre-`optim.step()` capture of RNG state, exact microbatches, and (only on a
`GRAD_NORM_HARD_TRIGGER`) a CPU clone of the pre-step weights, into a new
`_spikebatch.pt` sidecar; new `Cell 6d` adds `replay_spike_batch(step_tag)`,
a non-polluting isolated replay with per-parameter grad norms, a per-layer
`h`-tensor hook composed with the existing depth-routing patch, best-effort
forward-activation-extreme hooks on the creation gate/reverse
channel/V_theta/destruction gates, and a built-in fidelity check against
the originally recorded `pre_clip_grad_norm`. Scoped to hard triggers only
— the EMA path has no single offending batch. §33.3/§33.5 text and the
Phase-0-3 mermaid diagram updated to mark Phases 1/2 implemented; no
`_spikebatch.pt` exists yet for the two already-passed hard triggers
(32,139/34,091), so the harness awaits the run's next one). Previously
updated 28 August 2026, night (adds §31.7: ran the L=8 bracket
through §31.4's own budget-selection recipe and it self-terminates at
step 1 — no valid `precision_lr_max` window exists, because the
"runaway" tail is already present in the best/healthy checkpoint
(`max` 6,364.81, within 1.0% of spike_34091's 6,427.16 and only 14.5%
below spike_32139's 7,285.88). Recommendation: leave
`PRECISION_LR_MAX = None` on the live run; a non-targeted diagnostic
value (~2,500) is noted for §31.5's optional falsification A/B only.
§31.6's status line updated to match). Previously updated 28 August
2026, evening (§33.1 revised with the second bracket point: the
step-32,139 prereload snapshot came back +3% to +24% above healthy —
more elevated than step-34,091, despite firing first — so the bracket is
a modest, non-monotonic elevation rather than a perfectly flat null
result. The magnitude ($\lesssim 1.1\times$ in $\omega$) still rules out
$B_k$ as the primary driver of the $>100\times$ grad-norm spikes;
§33.1/§33.2 text, table, and figure updated accordingly). Previously
updated 28 August 2026 (adds §33: the L=8 `precision_lr_max` bracket pair
was measured — $\sigma_{\max}(B_k)^2$ was within +8% at every percentile
between the healthy step-27,000 checkpoint and the step-34,091 spike-regime
prereload snapshot — confirming §31.4 step 1's escape hatch that $B_k$
growth is not the primary driver of these bursts; `precision_lr_max` is not
the primary lever, and §33 lays out the cheapest-first Phase 0–3 root-cause
workflow for the non-$V_\theta$ spike groups plus a pointer to the new SCAF
`GradientSpikeProbe` design doc. Also fixes `lowrank_modes` to decompose $G$
by SVD instead of eigh on the Gram $G^\top G$, removing an
ill-conditioned-Gram convergence failure surfaced while running the
diagnostic). Previously updated 27
August 2026, evening (adds §32: the L=8 probe's
extended 27,000–39,867 trajectory is a noisy plateau — no new best,
spike rate not decaying — and the resulting decision to switch to
`baoab_cfc_lowrank` mid-run on single-GPU compute rather than complete
the plain-`baoab_cfc` arm to 100,000). Earlier the same day: implements
§31.3's SCAF `sigma_lr_*` percentiles and adds the equivalent
`sigma_lr_report` notebook diagnostic; broadens §31's scope to the L=8
probe after its own escalating 32,139/34,091 hard-watchdog bursts, with a
bracket checkpoint pair already on hand. Previously updated 24 August
2026 (adds §24.4's depth-probe update and §31's SCAF audit plan). Split
out of the parent note (former §24-§28,
content unchanged) for maintainability. Sections 29-30 record the principled
directions beyond the $B_k$ clamp and the concrete low-rank exponential
substep. This revision marks §29.2 (`integrator='baoab_cfc_lowrank'`) and
§29.3 (`precision_lr_max`) as **implemented and unit-tested** (the pre-existing
`baoab_cfc` path is byte-for-byte unchanged; both are opt-in via notebook
config) and corrects two errors uncovered during implementation:
(1) the exact rotation must act on the **PSD** low-rank $L = G G^\top$ with the
diagonal precision $D_a$ split off, not on the **indefinite** off-diagonal
$G G^\top - \mathrm{diag}$ (§30.1 proves the off-diagonal is indefinite via a
zero-trace argument and shows why PSD-ness of the sum $H$ does not survive an
indefinite split); and (2) the composed scheme is an **impulse / RESPA**
multiple-time-step method, which is stable at any stiffness *between*
resonances $\omega_L \Delta t \approx k\pi$ but is **not** unconditionally
A-stable — replacing the earlier overstated "no wall at all" claim. §30 is now
the as-built description (frozen/detached mode geometry, mode curvature
$\kappa_j = \lambda_j$ of $L$ rather than a Rayleigh quotient of $H$, and the
soft $D_a$ + nonlinear residual demoted to the explicit kick).
