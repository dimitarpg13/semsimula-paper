# Evidence for Second-Order ODE Governing Hidden-State Evolution in Transformers

> Discussion between Dimitar Gueorguiev and Claude, April 27, 2026.  
> Based on: *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference* (Gueorguiev, 2026).

> **Update 2026-04-27 (post-experiment) — overdamped synthesis.** A
> pre-registered Markov-order regression test of the dynamical-order claim
> ([`first_order_ODE_rejection_pre-registered_protocol.md`](./first_order_ODE_rejection_pre-registered_protocol.md))
> was run on GPT-2 small (primary) and Pythia-160m (replication). Outcome:
> **C — first-order ODE not rejected** in the kernel-ridge / linear-ridge /
> MLP function classes at PCA dim 50, robustly across the
> 4-class × 3-PCA-dim × 2-architecture grid (21 / 24 cells C; 3 poly-2
> over-fitting artefacts). The strong dynamical-exclusion claim
> (*"definitely no first-order ODE"*) is **not** empirically supported
> at one-token resolution.
>
> The synthesis the data *does* support is **overdamped second order**:
> the full Euler–Lagrange equation
> $w_t\ddot{h}\_t + \gamma(h_t)\dot{h}\_t = -\nabla V(h_t)$
> (Eq. 67) in the regime $\gamma \gg \omega_0$, where the inertial term
> $w_t\ddot{h}\_t$ is small relative to the dissipation $\gamma\dot{h}\_t$
> and the potential $\nabla V$. In that limit the EOM collapses to
> $\dot{h} \approx -\nabla V / \gamma$ — observationally indistinguishable
> from a first-order gradient flow at one-token resolution, but
> *generatively* still a Lagrangian system with kinetic, potential and
> Rayleigh terms. What the test rejects is the *underdamped* version of
> the second-order ODE — the one in which a velocity slot would carry
> detectable predictive information beyond what $h_t$ already encodes.
> It does not reject the second-order Lagrangian *as a generative
> account*. The promotion argument (§1 below) and Theorem 49 (§2 below)
> remain valid as a kinematic and theoretical scaffold.
>
> The §14 acceleration statistics ($a_\parallel<0$ on 97.9 % of triplets,
> $|a_\parallel|/|a_\perp|\approx 2$, permutation-null $z=23$) are exactly
> the trajectory-shape signature an overdamped attractive Lagrangian
> produces: the trajectory decelerates along its tangent because the
> velocity is being both damped and aligned with $-\nabla V$. They
> therefore remain valid *descriptive* evidence consistent with the
> framework — but they are silent on whether the inertial term is
> dynamically dominant or vanishing, and they cannot, by themselves,
> exclude first-order alternatives.
>
> Read in this light, the framework connects naturally to gradient-flow
> neural ODEs, score-matching and overdamped Langevin dynamics — all
> first-order in the same observational sense, and all derivable from
> the overdamped limit of the same Lagrangian. The sections below
> develop the *generative* second-order account; the *predictive* test
> does not detect the inertial term at one-token resolution. Full write-up:
> [`notebooks/dynamics_order_test/results/RESULTS.md`](../notebooks/dynamics_order_test/results/RESULTS.md).

> **Update 2026-04-28 — E4 damping sweep (SPLM with controlled $\gamma$).**
> A six-cell sweep of the SPLM (`splm_sarfmass_logfreq`) with $\gamma$
> fixed at 0, 0.10, 0.30, 0.85, 2.00, and 5.00 provides two further
> results that sharpen the overdamped synthesis.
>
> **Outcome β replicated across architecture and damping setting.**
> The Markov-order regression returns Decision C — first-order not
> rejected — at *every* cell, including $\gamma = 0$ (no architectural
> damping at all). At $\gamma = 0$ the ratio $\rho_{12}$ (defined as $R_1/R_2$) equals $0.906$
> and the lag-1 model is preferred with Wilcoxon $p = 4.9 \times 10^{-11}$.
> Training dynamics suppress the velocity slot regardless of how the
> architectural damping constant is set. The overdamped observational
> basin is a training-dynamics property, not a free-$\gamma$ artefact.
>
> **$\gamma^* \approx 0.30$ — 29 % PPL gain over the freely-trained model.**
> The sweep locates an empirical optimum at $\gamma = 0.30$ (val PPL 144)
> vs. the freely-trained convergence point $\gamma \approx 0.85$ (PPL 203).
> This confirms the framework's prediction (companion theory document §7)
> that joint optimisation of $\gamma$ is suboptimal: the optimizer drives
> $\gamma$ upward for training stability, overshooting the inference optimum.
> There is also a critical lower bound $\gamma_{\mathrm{crit}} \in (0, 0.10)$
> below which the Euler integrator becomes energy-non-conservative
> ($H_L / H_0 = 95$ at $\gamma = 0$) and performance collapses (PPL 202).
> Full write-up: [`docs/E4_sweep_results_and_discussion.md`](./E4_sweep_results_and_discussion.md).

---

## Background: First-Order vs. Second-Order Dynamical Systems

In a first-order system $\dot{x} = f(x, t)$, the evolution is fully determined by position alone. Acceleration exists as a derived quantity — $\ddot{x} = \frac{\partial f}{\partial x}\dot{x} + \frac{\partial f}{\partial t}$ — but it plays **no dynamical role**; it is entirely slaved to the state $x$ and carries no independent information.

In a second-order system (e.g., $m\ddot{x} = F(x, \dot{x}, t)$), acceleration is a **genuine dynamical quantity**: the phase space is $(x, \dot{x})$, both position and velocity are needed as initial conditions, and the system has inertia in a physically meaningful sense.

| Property | 1st Order | 2nd Order |
|---|---|---|
| Phase space dimension | $n$ | $2n$ |
| Initial data needed | $x(0)$ | $x(0), \dot{x}(0)$ |
| Acceleration's role | Derived / redundant | Independent dynamical variable |
| Physical archetype | Diffusion, gradient flow | Mechanics, oscillators |
| Inertia | Absent | Present |

---

## How the Paper Frames the Second-Order Question

The paper's argument is structured as a **two-part case**: a *theoretical promotion argument* (why the dynamics *must* be second-order) and an *empirical verification* (evidence that it *is* second-order).

---

## 1. The Main Reason: The "Promotion" Argument (Section 12)

The core claim is that **both prior dynamical accounts of transformers are first-order, and both are recovered as shallow limits of the same second-order Lagrangian**.

### Huang et al. (STP) — First-Order in Token Position

STP specifies the dynamics as a first-order ODE (their eq. 2):

$$dx_{\leq t} = \mathring{u} \circ \mathring{f}(x_{\leq t}) dt$$

Here the initial velocity $v_0$ is **not an independent degree of freedom** — it is slaved to the initial position via the flow operator. STP collapses the two-dimensional phase space $(x_0, v_0)$ into a one-dimensional one.

### Lu et al. — First-Order in Layer Depth (Convection–Diffusion)

$$\dot{x}_i = F_{\text{conv}}(x_i) + F_{\text{diff}}\left(x_i, \{x_j\}_{j \neq i}\right)$$

Again, no independent velocity slot. $\dot{x}_i$ is determined pointwise by current positions. This is the layer-as-time counterpart of the position-as-time flow of Huang et al.

### The Full Euler–Lagrange Equation — Genuinely Second-Order

The paper shows (Eq. 67) that the **full dissipation-adjusted Euler–Lagrange equation** is:

$$w_t \ddot{h}_t + \gamma(h_t) \dot{h}_t = -\nabla V(h_t)$$

a genuine second-order ODE. Both prior equations are recovered by **dropping the inertial term** $w_t \ddot{h}_t$ and fixing velocity algebraically as a function of position. They are **sibling shallow limits** of the same Lagrangian — not independent models.

The current paper (v3, §12) states this as follows:

> "The promotion from first to second order is therefore generative — it changes the kinematic state space and licenses the kinematic identity [Theorem 49] that is invisible to first-order accounts — without committing to the strictly underdamped observational claim that the data does not support."

> **Historical note.** An earlier draft of the paper used stronger language: "The promotion from first to second order is not cosmetic: the defining observable of a second-order system is acceleration, and it is precisely acceleration that Section 14 measures on GPT-2 hidden-state trajectories and that neither (105) nor (106) can host." This framing was revised in paper v3 after a pre-registered Markov-order regression test (Outcome C) demonstrated that the observed trajectory behaviour is also consistent with an overdamped first-order reduction of the same Lagrangian, so the observational second-order claim cannot be made from these data alone. See `first_order_ODE_rejection_pre-registered_protocol.md` and the §12 paragraph "Generative second-order, observational first-order."

### Architectural Grounding

The **first transformer block acts as a launching apparatus** that sets $v_0$ as a function of $x_0$ *and its neighbors* via attention — a function depending on the full sequence context, not just the current token. This context-dependence cannot be collapsed into a pointwise first-order flow. As the paper states, any contribution of $\text{Attn}^{(0)}$ that draws on neighbors $h_s$, $s \neq t$, is captured by the second-order term and is **invisible to the first-order model in principle**.

---

## 2. The Definitive Proof: Theorem 49 — The STP–Acceleration Identity (Section 13)

This is the linchpin theorem. It proves that the STP loss is *algebraically identical* to a function of the normal acceleration:

$$\mathcal{L}_{\text{STP}}(h_{t-1}, h_t, h_{t+1}) = 1 - \sqrt{1 - \frac{|\vec{a}_\perp|^2}{\lVert\vec{d}_2\rVert^2}}$$

with the immediate consequence (**Corollary 51**) that:

- STP measures **only** $|\vec{a}_\perp|$ — the normal (centripetal) component of acceleration.
- The **tangential acceleration** $a_\parallel$ is **algebraically invisible** to STP.

This is both a proof and a conceptual sharpening: since the STP loss equals a function of $|\vec{a}_\perp|$, the existence of STP-loss as a meaningful training signal is itself *direct evidence* that acceleration exists and is structured. A first-order system cannot have $\vec{a}$ as an independent observable — here it demonstrably is one.

**The companion total-acceleration decomposition identity** (Eq. 113):

$$\lVert\vec{a}_t\rVert^2 = \left(\lVert\vec{d}_2\rVert - \lVert\vec{d}_1\rVert\right)^2 + 2\lVert\vec{d}_1\rVert\lVert\vec{d}_2\rVert \cdot \mathcal{L}_{\text{STP}}$$

makes explicit that any nonzero STP loss implies acceleration — specifically normal acceleration — and cleanly separates tangential and normal contributions.

### Resolution of the Geodesic Paradox

A natural objection: *"if trajectories are geodesics, then $\mathcal{L}_{\text{STP}} = 0$, and there is no acceleration."* Theorem 49 resolves this: $\mathcal{L}_{\text{STP}} = 0$ forces only $\vec{a}_\perp = 0$. The tangential acceleration $a_\parallel$ is **unconstrained by STP**.

A hidden-state trajectory with $\mathcal{L}_{\text{STP}} \approx 0$ can still exhibit rich **tangential deceleration** — exactly what the Gaussian well predicts for a particle falling into an attractive potential. STP regularization neither detects this nor suppresses it; the two quantities ($\mathcal{L}_{\text{STP}}$ and $a_\parallel$) are **algebraically orthogonal**.

---

## 3. Empirical Evidence — GPT-2 Validation (Section 14)

The paper validates the second-order reading with four results on GPT-2 last-layer hidden states across **1,314 consecutive triplets**:

| Result | Finding |
|---|---|
| STP–acceleration identity | Holds to **machine precision** (exact verification of Theorem 49) |
| Tangential vs. normal acceleration | $\lVert a_\parallel \rVert$ is approximately **twice** $\lVert a_\perp \rVert$ on average |
| Sign of $a_\parallel$ | Negative (deceleration) on **97.9%** of triplets — consistent with Gaussian well attraction |
| Permutation null test | Natural token orderings produce **significantly less acceleration** than random permutations, quantifying the near-geodesic character of learned trajectories |

Results 2–4 are empirical support for the second-order dynamical reading — these kinematics signatures (tangential dominance, systematic deceleration, permutation null) are consistent with and motivated by the second-order framework.

> **Update (post Outcome C).** A subsequent pre-registered Markov-order regression test demonstrated that these signatures are also consistent with an overdamped first-order gradient flow $\dot{h} \approx -\nabla V / \gamma$ — the predicted reduction of the full EL equation when $\gamma \gg \omega_0$. Results 2–4 therefore support the **generative** second-order reading (the Lagrangian has an inertial term; acceleration is the primitive the STP identity measures) but do not by themselves exclude a first-order ODE as an effective description at the trained inference fixed point. The current paper (v3, §12) draws this distinction explicitly.

---

## Logical Structure of the Full Argument

```
Both STP (Huang et al.) and Lu et al. are first-order ODEs
        ↓
Both recovered as shallow limits of the full EL equation (drop wₜḧₜ)
        ↓
Theorem 49: STP loss ≡ f(|a⊥|) — acceleration is the primitive, not a derivative
        ↓
Corollary 51: a∥ is invisible to STP — first-order model cannot capture it in principle
        ↓
EL equation (67): genuine second-order ODE with inertial term wₜḧₜ
        ↓
GPT-2 validation: a∥ ≈ 2|a⊥|, 97.9% deceleration — empirical confirmation
```

---

## Summary

The **main reason** the paper argues for the generative second-order framework is that the transformer's first block computes the initial velocity $v_0$ as a function of both $x_0$ and its sequence neighbors (via cross-position attention) — a context-dependent computation that cannot be collapsed into a pointwise first-order flow. The framework treats $(x_0, v_0)$ as an independent pair, making velocity an architectural initial-condition slot. This is the **generative** second-order claim; the current paper (v3, §12) commits to this claim.

The **definitive proof** is Theorem 49, which shows the STP loss is normal acceleration — a quantity that only has meaning, and only can be measured, in a second-order dynamical system.

The **empirical confirmation** comes from GPT-2 experiments establishing that (i) the identity holds to machine precision, (ii) tangential deceleration dominates and is systematically negative, and (iii) natural orderings produce smoother trajectories than permuted ones — all consistent with second-order inertial dynamics.

> **Update (post Outcome C).** A pre-registered Markov-order regression test subsequently found that the observed trajectory behavior at inference is **not** exclusively second-order (Outcome C — first-order not rejected). The current paper explains this as the predicted observational consequence of the overdamped regime $\gamma \gg \omega_0$: the full EL equation reduces to a first-order gradient flow at the trained inference fixed point, so a Markov-order regression cannot distinguish the two. The **observational** second-order claim — that velocity carries independent predictive power beyond $h_t$ alone at inference — is therefore not supported, and the current paper (v3, §12 "Generative second-order, observational first-order") does not make it. The summary above describes only the generative claim, which the paper does make and which the Markov-order test does not refute.
