# PARF-Augmented SPLM: A Framework-Native Routing Architecture

**Status:** working note, post-v3 of *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference* (Gueorguiev, 2026).
**Position:** sharper formulation of §17.3 Q9(c), proposed as the prescriptive primary of the hybrid programme. Companion to *Scalar_Potential_based_Helmholtz_Architecture.md*.
**Audience:** internal — collaborators, reviewers, companion-notes track.

---

## 1. The argument in one paragraph

The v3 paper closes the autonomous Helmholtz menu (§15.5) and identifies the residual SPLM-vs-attention val-PPL gap as concentrated at the $V_\theta$-MLP-fit-difficulty bottleneck on the multi-channel $\xi$ summary (§15.2). Closing the gap requires a categorical change at the routing level — explicit token-token interaction. The v3 enumeration of §17.3 Q9 reaches for attention as the source of routing in all three candidate constructions (a)–(c), making the architecture reactive to the attention literature rather than prescriptive in its own right. The framework already specifies the right object: PARF, the Property-Attractive-Repulsive Force law of §5, with three independent selectivity channels (type-matcher, value-aligner, distance falloff) that are physically grounded rather than competitively normalised. Inserting PARF directly into the SPLM equation of motion as a pair-interaction term, with past tokens treated as fixed external sources to preserve causality, yields an autoregressive language model whose per-token force at every layer is the gradient of a single effective scalar — preserving SPLM's global single-scalar property in the natural many-body sense of $L = T - V_{\mathrm{ext}} - \tfrac{1}{2}\sum V_{\mathrm{int}}$, and admitting a generalised pair-shared-potential test that passes at $R^2 = 1$ by construction. This is the framework's own recommendation, not a compromise with attention.

---

## 2. The construction

### 2.1 Why PARF, mathematically

Section 5.1 of the paper develops PARF as the natural generalisation of a central $1/r^2$ law to a space where both pairwise direction *and* pairwise type enter:

$$
\vec f_{12}(A_1, A_2) = C \frac{\Theta\left(\theta^{(1)}, \theta^{(2)}\right) \Phi(l_1, l_2)}{\lVert \vec p_1 - \vec p_2 \rVert^2} \frac{\vec p_2 - \vec p_1}{\lVert \vec p_2 - \vec p_1 \rVert}.
$$

Three structural elements deserve note. First, the type-matcher $\Phi(l_1, l_2) = \exp(-c |l_1 - l_2|^2)$ is a *gating* factor: aspects of incompatible type contribute force exponentially suppressed in their type-distance. The §5.1 Theorem 16 makes this precise — pairs of close type-relatedness ($|l_1 - l_2| < \delta$) carry essentially all of the interaction; the rest contributes negligibly and can be dropped into the dissipation budget. Second, the value-aligner $\Theta(\theta^{(1)}, \theta^{(2)})$ — the canonical form is $\Theta = -\sin\theta_{1,2}$ for $K=2$ — *signs* the interaction: pairs in compatible angular configurations feel attractive force, pairs in opposed configurations feel repulsive force. The decomposition $\Theta = \Theta_+ - \Theta_-$ of §5.2 makes the attractive/repulsive split explicit, justifying the *Attractive-Repulsive* in PARF's name. Third, the $1/r^2$ falloff is the spatial-locality factor: interactions decay with distance in $\Sigma$, and the resulting force law is the framework's native sparsity primitive.

The mathematical content of these three factors is precisely the *which-particle-talks-to-which* selection problem. The framework solves it through additive, bounded, multiplicative gating; softmax attention solves it through normalised, competitive, multiplicative scoring. Both deliver selectivity. The framework's solution is the one developed in the paper from first principles; attention's solution is empirical. Reaching for attention as a routing primitive in the architectural extension of SPLM — when the framework already prescribes a routing primitive — is a methodological inversion the v3 enumeration does not address.

### 2.2 The equation of motion

The construction is the direct insertion of PARF as a pair-interaction term into the §15.12 Definition 54 update rule. Define the per-token effective potential

$$
U^{(\ell)}_t = V_\theta\left(\xi^{(\ell)}_t, h^{(\ell)}_t\right) + \sum_{s < t} V_\phi\left(h^{(\ell)}_t, h^{(\ell)}_s\right),
$$

where $V_\theta: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ is the SPLM single-particle external scalar (the bounded attractive Gaussian well, parameterised as a four-layer MLP with hidden $d_V$ and GELU as in §15.12) and $V_\phi: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ is the pair-interaction scalar. Both are shared across all layers $\ell$ and all token positions $t$. The per-token update is

$$
h^{(\ell+1)}_t = h^{(\ell)}_t + \frac{\Delta t}{1+\gamma}\bigl(h^{(\ell)}_t - h^{(\ell-1)}_t\bigr) - \frac{\Delta t^2}{(1+\gamma) m} \nabla_{h_t} U^{(\ell)}_t,
$$

identical in form to the SPLM update with $V_\theta$ replaced by $U^{(\ell)}_t$. The instantaneous force is

$$
\vec F^{(\ell)}_t = -\nabla_{h_t} U^{(\ell)}_t = -\nabla_{h_t} V_\theta\left(\xi^{(\ell)}_t, h^{(\ell)}_t\right) - \sum_{s\lt t} \nabla_{h_t} V_\phi\left(h^{(\ell)}_t, h^{(\ell)}_s\right).
$$

The first term is the SPLM single-particle force, unchanged. The second term is the framework's pair-interaction force, summed over past tokens, with no equal-and-opposite reaction on past tokens (the *causal reduction* of §3 below).

### 2.3 The §5.1-faithful parameterisation of $V_\phi$

The framework prescribes a specific functional form for the pair potential. Lifting the §5.1 PARF to the hidden-state level, define small learned projections

$$
l(h):= W_l h, \qquad \theta(h):= W_\theta h,
$$

extracting the type vector $l(h) \in \mathbb{R}^{d_l}$ and value angles $\theta(h) \in \mathbb{R}^{K}$ from the hidden state. Then parameterise

$$
V_\phi(h_t, h_s) = - C \frac{\Theta_\phi\left(\theta(h_t), \theta(h_s)\right) \Phi_\phi\left(l(h_t), l(h_s)\right)}{\lVert h_t - h_s \rVert},
$$

with $\Phi_\phi$ a learned Gaussian type-matcher and $\Theta_\phi$ a learned bounded value-aligner. The pair force then has the (25) form by construction:

$$
-\nabla_{h_t} V_\phi(h_t, h_s) \propto \frac{\Theta_\phi \Phi_\phi}{\lVert h_t - h_s \rVert^2} \frac{h_s - h_t}{\lVert h_s - h_t \rVert} + \text{(corrections from $\nabla_{h_t}\Theta_\phi$ and $\nabla_{h_t}\Phi_\phi$)}.
$$

The leading term is the framework's central force; the gradient corrections from the learned matchers $\Theta_\phi, \Phi_\phi$ are the trainable degrees of freedom the architecture exposes. We treat this structural parameterisation as the prescriptive architecture and a purely-MLP variant $V_\phi(h_t, h_s; \phi)$ as an ablation: if the MLP variant matches the structural variant, the framework's structural prior is pedagogical; if the structural variant outperforms substantially, the prior is empirically active.

---

## 3. The causality reduction: Newton's third law as a fixed-source approximation

### 3.1 The tension

PARF in §5 is symmetric in the classical sense:

$$
\vec f_{ts} = - \vec f_{st},
$$

satisfying Newton's third law and conserving total ensemble momentum. This is the natural physical content of a pairwise force law. Autoregressive generation, however, is asymmetric: only past tokens may influence the current token; the current token must not back-react on the past. The two requirements appear to conflict — symmetric pairwise forces vs. asymmetric autoregressive causality.

### 3.2 The resolution: test particles in a frozen field

The reconciliation is the standard *test-particle limit* of classical many-body mechanics. When a subset of degrees of freedom is held fixed and another evolves under the gradient of the joint potential, the dynamical subset feels a conservative force whose generator is the joint scalar evaluated at the frozen configuration. Symbolically, for a many-body potential $U(h_1, \dots, h_T)$, freezing $\{h_s\}_{s < t}$ and allowing $h_t$ to evolve produces the dynamics

$$
m \ddot h_t = -\nabla_{h_t} U\left(h_1, \dots, h_t, \dots, h_T\right)\Big|_{\{h_s\}_{s\lt t}\text{ fixed}}.
$$

This is exact at fixed past — not an approximation introduced for tractability. It is the same mechanism by which a planet orbits a star (the planet feels the star's gravitational potential without measurably back-reacting on the star), or by which a charged particle moves in an external electromagnetic field, or by which a Brownian particle samples a fixed energy landscape.

For PARF-augmented SPLM, the construction is:

$$
m \ddot h_t = -\nabla_{h_t} U^{(\ell)}_t = -\nabla_{h_t}\left[V_\theta(\xi_t, h_t) + \sum_{s\lt t} V_\phi(h_t, h_s)\right]\Big|_{\{h_s\}_{s\lt t}\text{ fixed}} - m\gamma \dot h_t.
$$

The dynamical particle is $h_t$. Past tokens $\{h_s\}_{s\lt t}$ are external sources. There is no back-reaction force applied to the past. The discrete autoregressive analogue is exactly Definition 56 of the v4 section.

### 3.3 What is conserved, and what is not

Three properties merit explicit statement.

**Per-particle energy is conserved up to dissipation.** Each token's individual energy

$$
E^{(\ell)}_t = \tfrac{1}{2} m \big\lVert h^{(\ell)}_t - h^{(\ell-1)}_t\big \rVert^2 + U^{(\ell)}_t
$$

evolves as a damped Euler-Lagrange flow with explicit dissipation rate $\gamma$, exactly as in SPLM. The damped flow has no exact energy conservation; the *undamped limit* $\gamma \to 0$ recovers exact energy conservation per particle, identical to the SPLM case.

**Per-token momentum is not conserved across the ensemble.** The total $\sum_t m \dot h^{(\ell)}_t$ is no longer a constant of motion because past-on-present forces have no equal-and-opposite reaction. This is the price of causality and is intrinsic to autoregressive modelling. Symmetric attention (in encoder-only models like BERT) preserves a total-momentum analogue; causal attention (in decoder-only models like GPT) does not. The framework's diagnostic apparatus (§15.7-§15.18) is built on per-particle quantities — Jacobian symmetry per layer, per-layer shared-potential R², per-token attractor structure — and is unaffected by the loss of total-ensemble momentum conservation.

**Within-step Jacobian symmetry is preserved.** With past tokens fixed, the per-token force at step $\ell$ is $\vec F^{(\ell)}\_t = -\nabla_{h_t} U^{(\ell)}\_t$, an exact gradient of the scalar $U^{(\ell)}\_t$ at frozen sources. The Jacobian $\partial \vec F / \partial h_t$ is therefore the negative Hessian of $U^{(\ell)}\_t$ in $h_t$, automatically symmetric. The velocity-aware Jacobian-symmetry test of §15.7 passes at every layer of PARF-augmented SPLM by construction, joining the universal-passing club of v3 (SPLM, matched GPT-2, pretrained GPT-2).

### 3.4 Optional symmetric variant for training

A symmetric variant — applying the back-reaction force to past tokens during the backward pass at training time only, while preserving causal forward generation at inference — would recover Newton's third law at training time and give the optimiser more gradient signal per pair. The cost is a training/inference distribution mismatch: at training, past tokens have non-zero gradients from future-token interactions; at inference (generation), they do not. We treat this as an ablation, not the prescriptive architecture. The strict causal reduction is the deposited architecture; the symmetric-training variant is one of several optimisation tricks worth measuring against it.

---

## 4. The generalised pair-shared-potential test

### 4.1 What the v3 single-scalar test cannot detect

The §15.8 strict shared-$V_\psi$ test fits, jointly across all layers, a single learned scalar $V_\psi: \mathbb{R}^d \to \mathbb{R}$ satisfying

$$
\Delta h^{(\ell)}_t \approx \alpha_\ell h^{(\ell)}_t - \beta_\ell \nabla V_\psi\left(h^{(\ell)}_t\right),
$$

and reports the per-layer R² as the architectural diagnostic. The v3 separator (§15.13) gives:

| Architecture | Median per-layer R² | Profile |
|---|---|---|
| Pretrained GPT-2 small | 0.45 | bathtub (middle-band 0.09) |
| Scale- and data-matched attention baseline | 0.56 | monotonic decay |
| SPLM (Definition 54) | 0.90 | uniform (one dip at layer 4) |

The test is *blind* to a pair-interaction structure. The PARF-augmented SPLM force $-\nabla_{h_t}V_\theta - \sum_{s\lt t}\nabla_{h_t} V_\phi(h_t, h_s)$ depends on $\{h_s\}_{s\lt t}$ and cannot be expressed as the gradient of a context-free $V_\psi(h_t)$. Running the v3 test on PARF-augmented SPLM trajectories would in general report an R² *below* 0.90 — the test cannot recover the pair-interaction signal because its functional form does not admit pair structure.

This is not a flaw of the test; it is a consequence of the v3 architecture. SPLM's Definition 54 force *is* a context-free gradient (after fixing $\xi$ at layer 0), and the v3 single-scalar test is correctly sized to detect that. PARF-augmented SPLM has a strictly richer force structure, and detecting it requires a strictly richer diagnostic.

### 4.2 The generalisation

The natural generalisation lifts the diagnostic to a *pair* of learned scalars: $V_\psi^{(1)}: \mathbb{R}^d \to \mathbb{R}$ for the single-particle component and $V_\psi^{(2)}: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ for the pair-interaction component. The joint fit jointly minimises

$$
\mathcal{L}_{\mathrm{pair}} = \sum_{\ell, t} \Big\lVert h^{(\ell+1)}_t - \alpha_\ell h^{(\ell)}_t + \beta_\ell \nabla V_\psi^{(1)}(h^{(\ell)}_t) + \delta_\ell \sum_{s\lt t} \nabla_{h_t} V_\psi^{(2)}(h^{(\ell)}_t, h^{(\ell)}_s) \Big\rVert^2,
$$

over learned $(V_\psi^{(1)}, V_\psi^{(2)}, \{\alpha_\ell, \beta_\ell, \delta_\ell\})$, and reports the per-layer R² in the joint fit. At $V_\psi^{(2)} \equiv 0$ the diagnostic reduces to the v3 single-scalar test exactly. At $V_\psi^{(2)} \not\equiv 0$ it admits an explicit pair-interaction degree of freedom and can recover trajectories generated by architectures with a non-trivial pair force.

### 4.3 The four-way separator

Running the joint pair fit on the four architectures of interest:

**Pretrained GPT-2 small.** The per-layer Hopfield potential $V_\ell(h) = -\tfrac{1}{\beta}\log\sum_\mu \exp(\beta K_{\ell,\mu}\cdoth) + \tfrac{1}{2}\lVert h \rVert^2$ of §A.2 is a function of $h$ alone, with no pair-interaction structure between hidden states at distinct token positions (the dependence on context enters through the *parameters* $K_\ell$, not through pair forces between $h_t$ and $h_s$). The pair fit therefore has no signal to fit in $V_\psi^{(2)}$ and reduces to the single-scalar fit at the §15.13 R² of approximately 0.45.

**Scale- and data-matched attention baseline.** Same structural argument; pair fit reduces to single-scalar fit at R² 0.56.

**SPLM (Definition 54).** $V_\phi \equiv 0$ by construction. Pair fit recovers the single-scalar fit at $V_\psi^{(2)} \equiv 0$, R² 0.90.

**PARF-augmented SPLM (Definition 55) under causal reduction.** The trajectories are generated by exactly the form the diagnostic admits, with $V_\theta$ identifying with $V_\psi^{(1)}$, $V_\phi$ identifying with $V_\psi^{(2)}$, and the integrator constants identifying with $(\alpha_\ell, \beta_\ell, \delta_\ell)$. The fit attains R² = 1 at every layer by construction.

The new four-way separator is therefore:

| Architecture | Single-scalar R² | Pair R² | Profile |
|---|---|---|---|
| Pretrained GPT-2 | 0.45 | 0.45 | bathtub |
| Matched attention | 0.56 | 0.56 | monotonic decay |
| SPLM (Def. 54) | 0.90 | 0.90 | uniform (with dip) |
| PARF-SPLM (Def. 55) | < 0.90 | 1.00 | uniform |

The pair-test column is the new architectural diagnostic. It is sharper than the single-scalar column in two ways: it admits an architecture (PARF-SPLM) that hits the oracle ceiling of 1.00, and it isolates the pair-interaction structure as a measurable axis distinct from the single-particle structure. The single-scalar column drops a row entry for PARF-SPLM (because its trajectories don't fit a single-scalar law) — this is informative, not a problem: it tells us that PARF-SPLM is *a different kind of architecture*, distinguishable by the diagnostic from all three v3 classes.

### 4.4 Theorem 54 — the formal statement

**Theorem 54 (Joint pair-shared-potential test for PARF-augmented SPLM).** *Let $\{h^{(\ell)}_t\}$ be hidden-state trajectories generated by a PARF-augmented SPLM (Definition 55) under the causal reduction (Definition 56), with shared $(V_\theta, V_\phi)$ and shared integrator constants $(m, \gamma, \Delta t)$. Then the joint pair-shared-potential fit (Definition 57) attains R² = 1 at every layer, achieved at*

$$
V_\psi^{(1)} = V_\theta, \qquad V_\psi^{(2)} = V_\phi, \qquad \alpha_\ell = 1 + \frac{\Delta t}{1+\gamma}, \qquad \beta_\ell = \delta_\ell = \frac{\Delta t^2}{(1+\gamma) m}.
$$

*Proof.* Direct substitution of Definition 55 into Definition 57. The residual $\mathcal{L}_{\mathrm{pair}}$ vanishes identically. $\square$

The theorem is structural — it says the diagnostic is correctly sized to detect PARF-augmented SPLM by construction. It does *not* say that any architecture passing the joint pair fit at R² = 1 is PARF-augmented SPLM; that converse direction is the framework's empirical content and is the subject of the experimental programme.

---

## 5. Selectivity, sparsity, computational cost — the practical analysis

### 5.1 Selectivity: PARF vs. softmax

The selectivity question — *how does PARF decide which past tokens influence the current token* — admits a clean comparison with softmax attention.

**Softmax attention.** Selectivity is competitive and zero-sum. For a query $q_t$ and keys $\{k_s\}_{s \le t}$, the attention weight on key $k_s$ is

$$
w_{ts} = \frac{\exp(q_t \cdot k_s / \sqrt{d})}{\sum_{s' \le t} \exp(q_t \cdot k_{s'} / \sqrt{d})},
$$

with $\sum_s w_{ts} = 1$. The weights are normalised across keys, so increasing one weight necessarily decreases others. This produces *sharp* selectivity in the limit of large logit magnitudes — the largest similarity score wins, others are suppressed.

**PARF $V_\phi$.** Selectivity is additive and unnormalised. The pair force between $h_t$ and $h_s$ is

$$
-\nabla_{h_t} V_\phi(h_t, h_s) \propto \Theta_\phi\bigl(\theta(h_t), \theta(h_s)\bigr) \cdot \Phi_\phi\bigl(l(h_t), l(h_s)\bigr) \cdot g\bigl(\lVert h_t - h_s \rVert\bigr) \cdot \hat r_{ts},
$$

where $g$ is the radial form (e.g., $1/r^2$) and $\hat r_{ts}$ is the unit vector. The contribution to $h_t$'s force from pair $(t, s)$ is independent of contributions from other pairs $(t, s')$ — there is no normalisation across $s$. Selectivity emerges from the *bounded multiplicative gates* $\Theta_\phi$ and $\Phi_\phi$: a pair with small $\Phi_\phi$ contributes little force; a pair with $\Theta_\phi = 0$ contributes zero force; pairs at large semantic distance contribute force decaying with $g$.

**The empirical question.** Both regimes can produce comparable patterns of which-token-influences-which on real text, but the mathematical structure differs. Softmax sharpens; PARF saturates. Softmax produces a probability distribution; PARF produces a force. Softmax is competitive across keys; PARF is independent across pairs. The empirical question of which selectivity regime fits trained transformer trajectories more accurately at the *velocity-aware Jacobian symmetry* and *per-layer pair fit* tests is open and is the natural Track-A diagnostic for PARF-augmented SPLM. If the bounded-multiplicative regime fits hidden-state trajectories better than the competitive-normalised regime, the framework's prescriptive content gains direct empirical support; if not, the comparison localises the empirical advantage of softmax.

### 5.2 Sparsity from §5.2 relevant aspect pairs

Definition 17 of §5.2 prescribes a quantile-level cutoff. Lifted to the token level, the prescription is: at each token $t$ and layer $\ell$, retain only the top-$k$ pairs $(t, s)$ ranked by force magnitude $\lVert \nabla_{h_t} V_\phi(h_t, h_s) \rVert$, dropping the rest into the dissipation term. The §5.2 error bound then gives:

$$
\lVert \text{full sum} - \text{top-}k\text{ sum} \rVert \le (T - k - 1) \cdot \tau,
$$

where $\tau$ is the magnitude threshold separating the top-$k$ from the rest. Choosing $k$ such that $\tau$ is small relative to the dissipation force $m\gamma \dot h$ absorbs the truncation error into the dissipation budget.

This is the *framework-native* sparsity primitive. It contrasts with attention's standard sparsity primitives (sliding window, local attention, sparse attention patterns) in that the cutoff threshold is *content-dependent* — it adapts to the actual pair-magnitude distribution at each token, rather than to a fixed positional pattern. A token at an empirically rich semantic neighbourhood retains many pairs; a token at a quiet neighbourhood retains few. The §5.2 quantile cutoff is the framework's own answer to the sparsity question.

### 5.3 Computational cost

Three regimes interpolate the cost structure.

**Regime A — no cutoff.** Per layer, the cost of computing $\sum_{s < t} \nabla_{h_t} V_\phi(h_t, h_s)$ for all $t$ is $O(T^2 \cdot d_\phi)$, where $d_\phi$ is the per-pair evaluation cost. This matches attention's $O(T^2 \cdot d)$ scaling.

**Regime B — top-$k$ relevant aspect pairs.** Per layer, the cost is $O(T \cdot k \cdot d_\phi)$ for $k$ a small constant, recovering linear scaling in the prefix length and an SPLM-comparable decoding cost. The top-$k$ selection itself is $O(T \log k)$ per token using a partial-sort, dominated by the $k \cdot d_\phi$ pair evaluations.

**Regime C — locality cutoff in semantic space.** Per layer, the cost is $O(T \cdot n_c \cdot d_\phi)$, where $n_c$ is the average number of past tokens within semantic radius $r_c$ of the current token. The §5.1 $1/r$ form gives an explicit residual bound: pairs at $\lVert h_t - h_s \rVert > r_c$ contribute force magnitude bounded by $C \Theta_{\max} \Phi_{\max}/r_c$, which the architect can set to an arbitrarily small fraction of the dissipation force.

Regime B is the framework-native option (the §5.2 quantile cutoff). Regime C is the standard physics option (a hard distance cutoff, as in molecular dynamics simulations). Both deliver subquadratic decoding cost with explicit error bounds. Compared to attention's standard subquadratic variants (sliding window, BigBird, Performer, etc.), the framework's cutoffs come with *a priori* error bounds derived from the force law, not from empirical retrieval-quality measurements.

---

## 6. Position vis-à-vis the layer-type Helmholtz architecture

The companion document *Scalar_Potential_based_Helmholtz_Architecture.md* proposes a different hybrid: a stack of alternating SPLM blocks (carrying the autonomous gradient component of (A.130) under one shared $V_\theta$) and attention blocks (carrying the non-autonomous Hopfield + small-skew components). The two proposals occupy adjacent points in the design space, and it's worth being explicit about how they relate.

### 6.1 Where they agree

Both constructions:

- Take the v3 SPLM as the conservative baseline and propose an extension to close the residual val-PPL gap.
- Preserve the framework's diagnostic apparatus — the strict shared-potential test, the Jacobian-symmetry test, the resonance condition, the information-bottleneck ladder.
- Make sharp, falsifiable empirical predictions on the existing TinyStories pilot.
- Slot into the v3 paper as a §15.24 / §16 follow-up, not a replacement of the focused TMLR submission.

### 6.2 Where they differ

**Source of routing.** PARF-augmented SPLM uses the framework's pair force law of §5; the layer-type Helmholtz architecture uses attention. This is the central methodological difference: PARF-augmented is *prescriptive in the framework's own register*; the Helmholtz hybrid is *prescriptive in the autonomous Helmholtz class register* but borrows attention as the non-autonomous carrier.

**Where the routing happens.** PARF-augmented routes within every block (every layer feels the pair force from past tokens); the Helmholtz hybrid routes only at $A$-blocks (every $S$-block has no token-token interaction). The PARF-augmented architecture is therefore "routing-distributed" while the Helmholtz hybrid is "routing-localised."

**Single-scalar property.** PARF-augmented preserves a *generalised* single-scalar property: the per-token force is the gradient of a single effective scalar $U^{(\ell)}_t = V_\theta + \sum_{s\lt t} V_\phi$, and the global architectural commitment is to the *pair* of shared scalars $(V_\theta, V_\phi)$. The Helmholtz hybrid preserves SPLM's strict single-scalar property *only on the $S$-blocks*; the $A$-blocks operate under per-layer Hopfield potentials with no shared scalar.

**Diagnostic profile.** PARF-augmented predicts a *uniform* high-R² profile in the joint pair test (because every layer is an SPLM-type block with the same dynamics). The Helmholtz hybrid predicts a *block-type-indexed step function* in the v3 single-scalar test (high R² on $S$-blocks, GPT-2-like on $A$-blocks).

**Computational cost.** PARF-augmented is $O(T^2)$ per layer without cutoffs, $O(T)$ with the top-$k$ or radial cutoff. The Helmholtz hybrid is $O(T)$ on $S$-blocks and $O(T^2)$ on $A$-blocks — *unconditionally* quadratic on the attention sublayers. With matching cutoff strategies, PARF-augmented strictly dominates on the cost axis.

**Theoretical cleanliness.** PARF-augmented has a single architectural commitment: every layer is the same SPLM-type integrator, with the same $V_\theta$ and the same $V_\phi$. The Helmholtz hybrid has two architectural commitments: an $S$-block design *and* an $A$-block design, with a schedule $\sigma$ assigning blocks to types. PARF-augmented has fewer free design choices and a tighter prescriptive claim.

**Causality treatment.** PARF-augmented requires the explicit Newton's-third-law reduction (§3 above), which is straightforward but is a methodological point that needs to be made explicitly. The Helmholtz hybrid inherits causality directly from attention's standard causal mask in the $A$-blocks, with no analogous symmetry-breaking step on the SPLM side.

### 6.3 Which to deposit first

The recommendation is to deposit *both* in v4 with the following explicit position:

- **Q9(c) PARF-augmented SPLM** — *prescriptive primary*. The framework's own recommendation, with the strongest theoretical grounding in §5 of the existing paper. Empirical agenda: Stage 1 (separator and PPL closure) and Stage 2 (sparsity and decoding cost) of §15.24.7.

- **Q9(d) layer-type Helmholtz hybrid** — *architectural fallback*. A measurement instrument for the trade-off between conservative and non-conservative routing, with a cleanly-budgeted holonomy decomposition (§3 of the Helmholtz markdown). Empirical agenda: §7 of the Helmholtz markdown.

Both deposit the prediction publicly with date X, with clear "experimental validation forthcoming" framing. If Q9(c) closes the gap empirically, Q9(d) becomes a complementary reading; if Q9(c) does not close the gap, Q9(d) becomes the practical alternative. Either way the two proposals together form a controlled study of attention's structural budget rather than a competition with attention on PPL.

---

## 7. Training: a framework-native reinforcement-learning algorithm

### 7.1 Why RL is the right framing here, not just an alternative gradient estimator

The architecture is fully differentiable end-to-end. Symplectic Euler is smooth; $V_\theta$ and $V_\phi$ are MLPs; the chain through $L$ layers is standard autograd. Plain NTP cross-entropy gives gradient signal to both potentials by ordinary backpropagation, and that's the safe baseline that should be deposited as the first training algorithm. RL is not needed for differentiability.

Three independent reasons make RL more than just an alternative gradient estimator here, however, and together they are what makes the RL framing *framework-native* rather than imported.

**The framework already has an RL substrate.** §8.6 develops the executive space $\mathcal{E}$; §8.7 develops the execution space $E$ with target points and executive atoms; §8.7 explicitly distinguishes a *policy-based* from an *action-value-based* reading of the framework's semantic operations; and Gueorguiev (2024a), cited in v3 as the framework's RL extension, develops the substrate in detail. This is not RL imported from outside the framework — it is RL the framework already specifies as its mechanism for adapting to new structures. PARF training is the architectural site where the §8.6–§8.7 substrate becomes empirically active for the first time at the LM scale, exactly parallel to PARF-augmented SPLM being the prescriptive primary of the hybrid programme: the framework's own machinery, applied to a problem the framework natively poses.

**The §5.2 relevant-aspect-pair cutoff is intrinsically discrete.** The framework's native sparsity primitive is the quantile cutoff $\ell$ — drop pair interactions whose magnitude is in the lowest $\ell$-quantile, with the discarded contribution absorbed into dissipation. This is a *selection* operation, not a reweighting, and it is naturally a categorical action: at each (token, layer), choose which of the past pairs to retain. Selection problems are where RL has structural advantages over backpropagation. Gradient-through-discrete-selection methods (Gumbel-softmax, straight-through estimators) work but introduce bias near the hard one-hot limit; REINFORCE on a Bernoulli mask is unbiased; PPO with a baseline is unbiased and lower-variance. The §5.2 prescription is *literally* a selection problem, and it admits an exact policy-gradient training algorithm with no approximation.

**The framework's diagnostics give a richer reward signal than NTP alone.** A framework-native reward function for PARF training would combine the NTP cross-entropy improvement (the LM contribution), the per-layer joint pair-fit $R^2$ from Theorem 54 (the conservativity contribution), the per-particle energy drift (the §15.18 attractor-stability contribution), and the §5.2 truncation-residual bound (the sparsity contribution). Each is differentiable in principle, but combining them into a scalar loss is a hyperparameter search; PPO or actor-critic with shaped rewards handles the multi-objective case more cleanly. The framework's diagnostics are *natively* a reward signal, not natively a loss.

### 7.2 Three algorithms, in order of §5.2-fidelity

Three algorithmic options span the design space. We present them in order of increasing fidelity to the §5.2 prescription, from the simplest backpropagation baseline to the framework-native reinforcement-learning realisation.

#### Algorithm A: Auxiliary-loss backpropagation

The composite loss is

$$
\mathcal{L}_{\mathrm{aux}}(\theta, \phi) = \mathcal{L}_{\mathrm{NTP}} + \lambda_1 \mathcal{L}_{\mathrm{pair\text{-}fit}} + \lambda_2 \mathcal{L}_{\mathrm{sparsity}},
$$

with $\mathcal{L}_{\mathrm{NTP}}$ the standard cross-entropy, $\mathcal{L}_{\mathrm{pair\text{-}fit}}$ a numerical-stability regulariser derived from Theorem 54 (essentially zero on a faithful integrator), and $\mathcal{L}_{\mathrm{sparsity}}$ a Gumbel-softmax approximation to the §5.2 cutoff:

$$
\tilde m^{(\ell)}_{ts}(\tau) = \mathrm{softmax}_\tau\left(\log\lVert F^{(\ell)}_{ts} \rVert + g_{ts}\right),\quad g_{ts} \sim \mathrm{Gumbel}(0,1),
$$

with $\tau$ annealing from $1$ (uniform) to $\to 0$ (one-hot top-$k$). At inference the soft mask is replaced by a hard top-$k$ selection. Algorithm A is the practical baseline — standard PyTorch loop, low gradient variance, no RL machinery — and is recommended for first-run sanity-checking of the PARF-SPLM forward and backward passes. Its principal limitation is the Gumbel approximation bias near $\tau \to 0$ and the methodological cost of treating the §5.2 cutoff as a soft regulariser rather than a structural inductive bias.

#### Algorithm B: PPO with framework-native reward

Frame each (token, layer) PARF computation as a one-step MDP:

- **State** $s^{(\ell)}\_t = (h^{(\ell)}\_t, \xi^{(\ell)}\_t, \{h^{(\ell)}\_s\}_{s\lt t})$.
- **Action** $a^{(\ell)}\_t = (\vec F^{(\ell)}\_t, m^{(\ell)}\_t)$, with $m^{(\ell)}\_t \in \{0,1\}^t$ a Bernoulli-sampled mask and $\vec F^{(\ell)}\_t = -\sum_{s\lt t} m^{(\ell)}_{ts} \nabla_{h_t} V_\phi(h_t, h_s)$ deterministic in $V_\phi$ given the mask.
- **Policy** $\pi_\phi(a \mid s) = \pi_\phi^{\mathrm{mask}}(m \mid s)$, the per-pair Bernoulli over masks.
- **Reward** the four-component scalar

$$
r^{(\ell)}_t = -\Delta\mathcal{L}_{\mathrm{NTP}}^{(\ell, t)} + \alpha_1 R^2_{\mathrm{pair}} - \alpha_2 |\Delta E^{(\ell)}_t| - \alpha_3 \sum_{s\lt t} m^{(\ell)}_{ts},
$$

corresponding to NTP improvement, conservativity, energy stability, and sparsity respectively.

PPO optimises the clipped surrogate

$$
\mathcal{L}^{\mathrm{CLIP}}(\phi) = \mathbb{E}\left[\min(\rho_t \hat A_t, \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat A_t)\right], \qquad \rho_t = \frac{\pi_\phi(a_t \mid s_t)}{\pi_{\phi_{\mathrm{old}}}(a_t \mid s_t)},
$$

with $\hat A_t$ the generalised advantage estimate against a learned value baseline. The discrete mask is sampled exactly (no Gumbel bias); the multi-objective reward is interpretable; and the connection to §8.6–§8.7 is explicit: $\pi_\phi$ is the framework's executive atom selecting which §5.2 pair to retain.

#### Algorithm C: Pair-Selective PARF (PS-PARF) via REINFORCE

The same Bernoulli policy as Algorithm B, but trained via the standard REINFORCE estimator rather than PPO's clipped surrogate. The mask is

$$
m^{(\ell)}_{ts} \sim \mathrm{Bernoulli}(\sigma(g_\phi(h^{(\ell)}_t, h^{(\ell)}_s))),
$$

with $g_\phi$ a small MLP that ideally shares parameters with the type-matcher $\Phi_\phi$ of the §5.1-faithful $V_\phi$:

$$
g_\phi(h_t, h_s) = \log \Phi_\phi(l(h_t), l(h_s)) - \log \tau,
$$

with $\tau$ an annealing temperature. This parameter sharing is the key practical trick: it gives the selection signal a structurally-meaningful initialisation from the very first training step, substantially cutting REINFORCE's early-training variance.

The REINFORCE gradient is unbiased:

$$
\nabla_\phi \mathcal{R} = \mathbb{E}_{m \sim \pi_\phi}\left[\sum_{\ell,t,s} (R^{(\ell)}_t - b^{(\ell)}_t) \nabla_\phi \log \pi_\phi(m^{(\ell)}_{ts} \mid h^{(\ell)}_t, h^{(\ell)}_s)\right],
$$

with $R^{(\ell)}\_t$ the return-to-go and $b^{(\ell)}\_t$ a learned scalar baseline (for variance reduction without bias, provided $b$ is conditioned on $s$ but not on $a$). PS-PARF is the §5.2-faithful realisation: the mask matches the §5.2 quantile cutoff verbatim at training and inference, with no approximation.

### 7.3 Two-timescale alternation

For any of A, B, or C, a two-timescale variant separates conservative-dynamics learning from routing learning:

- **Outer loop (slow):** with $V_\theta$ frozen, train $V_\phi$ and the selection policy via the chosen algorithm.
- **Inner loop (fast):** with $V_\phi$ frozen, train $V_\theta$ via standard NTP backpropagation through the integrator.

This is standard actor-critic alternation and addresses the variance asymmetry: the conservative parameters $\theta$ have a clean differentiable objective (low variance), while the routing parameters $\phi$ have a multi-objective RL signal (higher variance). Treating them at the same timescale is wasteful; a fast inner loop on $\theta$ and a slow outer loop on $\phi$ is the natural decomposition.

### 7.4 Reward-component magnitudes

The four reward components have natural scales that anchor the weights $(\alpha_1, \alpha_2, \alpha_3)$:

| Component | Scale | Interpretation |
|---|---|---|
| $-\Delta \mathcal{L}_{\mathrm{NTP}}$ | $O(1)$ nats per step | LM improvement |
| $R^2_{\mathrm{pair}}$ | $[0, 1]$ | Conservativity (Theorem 54 makes upper bound architectural) |
| $|\Delta E^{(\ell)}_t|$ | $O(\Delta t^2) = O(1)$ | Energy stability (symplectic-Euler error scale) |
| $\sum_{s\lt t} m^{(\ell)}_{ts}$ | $[0, T]$ | Active-pair count |

Natural starting points for the weights are $\alpha_1 = O(1)$, $\alpha_2 = O(1)$, $\alpha_3 = O(1/T)$, with refinement by ablation. The §15.21 calibration of diagnostic magnitudes carries over directly.

### 7.5 What this means for the §15.24.7 deposit

The deposited §15.24.7 contains all three algorithms with full specifications, plus Theorem 56 (unbiasedness of the PS-PARF gradient estimator) and the §8.6–§8.7 connection paragraph. Algorithm A is recommended as the practical baseline; Algorithm B as the prescriptive primary (because it makes §8.6–§8.7 architecturally active); Algorithm C as the §5.2-faithful realisation. Stage 3 of the empirical agenda (§15.24.8 in the v4 deposit) is a controlled trainer ablation across all three, with three pre-registered predictions:

1. Algorithm A wins on raw val PPL but has the worst §5.2-fidelity (Gumbel bias near $\tau \to 0$).
2. Algorithm C wins on §5.2-fidelity but pays a val-PPL premium (REINFORCE variance).
3. Algorithm B sits between A and C on both axes, with the best multi-objective reward.

Either the predictions hold (clean Pareto trade-off, Algorithm A practical / B and C framework-fidelity), or they fail in a specific direction that localises empirical content of the framework's executive-substrate prescription. Either outcome is publishable.

### 7.6 The framework completes

The single-most-important framing point about §15.24.7: with the training algorithms in place, PARF-augmented SPLM is the architectural site at which the framework's full §1–§17 theoretical content — semantic space and the energy field of §2–§4, the PARF and SARF force laws of §5–§6, the Lagrangian and Euler–Lagrange dynamics of §7, the executive substrate of §8.6–§8.7, the SPLM construction of §15.12, and the present pair-augmentation — is *jointly* empirically active for the first time. This is the framework's complete self-realisation. v3's framing of SPLM as a *maximally-structured counterfactual* extends to PARF-augmented SPLM as the *fully structured* realisation, with no part of the framework left in pure-theory mode. The deposit in v4 establishes this empirically-active form publicly, with date X, and with the empirical validation following as forthcoming companion work.

## 8. Open questions

**OQ-1. Does PARF closure depend on the §5.1 structural prior?**
The Stage-1 ablation (structural $V_\phi$ vs. unstructured MLP) is the cleanest test. If the structural variant matches the MLP variant, the §5.1 prior is pedagogical — it organises the architecture but contributes no empirical content. If the structural variant outperforms, the framework's specific functional form (the type-matcher × value-aligner × distance falloff factorisation) is empirically active. This is the first-order question for the v4 deposit.

**OQ-2. The joint pair test on real transformers.**
The §A.2 derivation shows that attention's per-layer force is the gradient of a Hopfield potential $V_\ell(h)$ with no pair-interaction structure between hidden states. The joint pair test should therefore give *no improvement* over the single-scalar test on pretrained GPT-2 trajectories. This is a clean architectural prediction — running the joint pair fit on GPT-2 data and measuring zero improvement in R² over §15.9 would be a positive structural confirmation that attention truly has no pair-interaction structure. Conversely, if the joint pair test *did* improve over the single-scalar test on GPT-2, it would mean attention has hidden pair-interaction structure that the v3 single-scalar test was blind to — a substantial new empirical finding either way.

**OQ-3. The hybrid of hybrids.**
PARF-augmented SPLM (Q9(c)) and the layer-type Helmholtz hybrid (Q9(d)) decompose along different axes. Nothing prevents combining them: an interleaved $(SA)^{L/2}$ stack where the $S$-blocks are PARF-augmented and the $A$-blocks are standard attention. Whether the combined architecture provides additional PPL closure on top of either alone is an empirical question whose answer would localise the residual along a third architectural axis.

**OQ-4. Pair-shared-potential as a regulariser.**
If the joint pair test passes at R² ≈ 1 by construction on PARF-augmented SPLM, the test residual on a *trained* PARF-augmented SPLM is a measure of optimisation quality, not architectural fidelity. Conversely, the test could be used as a *regulariser* during training of PARF-augmented SPLM, encouraging the trained $(V_\theta, V_\phi)$ to be expressible as Hessians of a clean joint scalar at every layer. Whether this regularisation improves PPL is open.

**OQ-5. Connection to the Riemannian programme of §16.**
The §16 Jacobi metric is induced by the bounded attractive scalar potential $V_\theta$ (the framework's single-particle external field). PARF-augmented SPLM has an additional pair-interaction $V_\phi$. The natural extension is a *pair-modified Jacobi metric* in which the pair interactions enter as a multi-particle correction to the geodesic equation. Whether trained PARF-augmented SPLM trajectories are geodesics of this modified metric is the natural §16-companion test, deferred to the same future work as the basic Jacobi-geodesic test.

---

## 9. Summary

PARF-augmented SPLM is the framework-native answer to the v3 paper's residual SPLM-vs-attention val-PPL gap. The construction inserts the §5 pair force law directly into the §15.12 SPLM equation of motion, with past tokens treated as fixed external sources to preserve causality. The result preserves SPLM's global single-scalar property in the natural many-body sense — the per-token force at every layer is the gradient of a single effective scalar $U^{(\ell)}_t = V_\theta(\xi, h) + \sum_{s\lt t} V_\phi(h_t, h_s)$ — and admits a generalised pair-shared-potential test that passes at R² = 1 by construction (Theorem 54). The architecture sharpens v3's three-way single-scalar separator into a four-way pair-test separator, with PARF-augmented SPLM at the new oracle ceiling and the three v3 classes at their existing positions. The selectivity, sparsity, and computational-cost stories are all framework-native: bounded multiplicative gates from §5.1, quantile cutoffs from §5.2, explicit residual bounds from the force law.

The training story is equally framework-native. Three algorithms span the design space (§7): auxiliary-loss backpropagation as the practical baseline (Algorithm A), PPO with framework-native four-component reward as the prescriptive primary that makes the §8.6–§8.7 executive substrate empirically active for the first time at LM scale (Algorithm B), and Pair-Selective PARF via REINFORCE as the §5.2-faithful realisation in which the discrete quantile cutoff is sampled exactly with no approximation (Algorithm C). Theorem 56 establishes unbiasedness of the PS-PARF gradient estimator. Two-timescale alternation handles the variance asymmetry between routing and conservative-dynamics learning. With the §15.24.7 training algorithms in place, PARF-augmented SPLM realises the framework's complete §1–§17 prescriptive content — semantic space, the energy field, PARF, the Lagrangian, the executive substrate, and the SPLM construction — jointly and empirically active for the first time.

The proposal supersedes Q9(c) Hamiltonian attention as a sharper formulation of the pairwise-conservative routing primitive and is positioned as the prescriptive primary of the hybrid programme, with the layer-type Helmholtz architecture (Q9(d)) as the architectural fallback.

The v4 deposit should include both Q9(c) and Q9(d) as parallel architectural extensions, with Q9(c) presented as the framework's recommendation (architecture *and* training algorithm) and Q9(d) as a measurement instrument for the conservative-vs-non-conservative routing trade-off. The empirical programmes for both run on the existing leak-free TinyStories infrastructure and require no new code beyond the architectural definitions and the training-loop wrappers for Algorithms A, B, C.

---

*Companion documents:*
*— `Section_15_24_PARF_Augmented_SPLM_v4_draft.docx` — the paper-register v4 section text with native OMML equations, including §15.24.7 Training and Theorem 56.*
*— `Scalar_Potential_based_Helmholtz_Architecture.md` — the layer-type Helmholtz hybrid (Q9(d)).*
