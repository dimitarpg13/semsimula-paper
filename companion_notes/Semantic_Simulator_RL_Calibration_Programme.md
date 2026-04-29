# Semantic Simulator with RL-calibrated Force Fields: a programme memo

An internal planning document for a radical alternative to the
SPLM/transformer line: implement the Semantic Simulation framework
*directly* as a particle-mechanics simulator in semantic space,
with force-field parameters calibrated by reinforcement learning
rather than learned end-to-end inside a neural network.

Written 2026-04-24, parallel in role to
`docs/SPLM_Path_Toward_SOTA.md` but for the simulator branch of the
research programme. The two branches are complementary, not
substitutes.

---

## 1. Short answer

**Pursue this as a parallel research line, starting with a
toy-language behavior-cloning experiment that costs 2–4 weeks.**

If the toy experiment succeeds, this becomes a defensible branch of
physics-informed ML applied to language, in the same family as
neural force fields for molecular dynamics (ANI, NequIP, MACE) and
structured energy-based models. If it fails, the failure mode itself
will tell us *which* of the framework's load-bearing assumptions
(force forms, coordinate system, non-autonomy structure) need
revision — information that no transformer-based experiment can
extract.

The expected upside is **not** SOTA perplexity. It is (a) a
direct, executable realisation of the Semantic Simulation framework
that doesn't rely on the transformer scaffold, (b) an interpretability
substrate that exposes every causal force by name, and (c) the
strongest possible falsification path for the framework: either the
forces simulate language, or they don't.

**Important:** v0 (M0–M5 below) deliberately restricts the simulator
to a closed-vocabulary, fixed-anchor system. This bounds expressivity
at the static parameter count and is *not* the framework's full
intent. The programme also names three deferred extensions —
**destruction / salience decay (v1.5)**, **creation of new semantic
structures (v2)**, and **operator-valued execution (v3)** — that the
framework needs to model the full productive expressivity of human
language. They are specified in §10 below and in
`Semantic_Simulator_EOM.md` §13. The v0 ceiling is an artefact of v0,
not of the framework.

---

## 2. Why this is worth pursuing now

The SPLM line establishes that the framework's dynamics are
*empirically realised* in trained transformers (§13–14, R²≈0.99 in
shared-potential separator, attractor structure in damped flow). It
does **not** establish that the framework's dynamics are *sufficient*
on their own — the transformer is doing the heavy lifting of
discovering the force functions from data.

A simulator branch tests the stronger claim:

> The framework's force forms (SARF + PARF + Gaussian wells +
> semantic mass), with appropriate calibration, are sufficient to
> generate non-trivial linguistic behaviour without learning the
> architecture from scratch.

If true, this is a Newton-of-language scientific contribution
independent of perplexity benchmarks. If false, we learn precisely
which force forms are missing and which assumptions in §A1 are too
strong.

This is the kind of binary scientific question the SPLM line cannot
ask, because SPLM assumes a transformer backbone exists.

---

## 3. Equations of motion: minimum specification

The simulator's state at integration step $\ell$ is

$$
s_\ell = (x_\ell, \dot x_\ell, \xi_\ell, \mathfrak{m}_\ell, \theta_\ell),
$$

where $x_\ell \in \mathbb{R}^d$ is the particle's semantic-space
position, $\dot{x}_\ell$ its velocity, $\xi_\ell$ the cumulative
semantic context (causal mean of preceding semantic structures), and
$\mathfrak{m}_\ell$ the per-step semantic mass. $\theta_\ell$ collects
the layer-indexed parameters of any non-autonomous fields.

The dynamics is the damped Euler–Lagrange flow of the paper's §7:

$$
\mathfrak{m}_\ell \ddot x_\ell  =  -\nabla_x V(\xi_\ell, x_\ell)
                                  -  \gamma \dot x_\ell,
$$

with the potential decomposed into named force terms:

$$
V(\xi, x)  =  \underbrace{V_{\text{wells}}(x)}_{\text{lexical attractors}}
             +  \underbrace{V_{\text{SARF}}(\xi, x)}_{\text{semantic attractor-repellor field}}
             +  \underbrace{V_{\text{PARF}}(x)}_{\text{property field}}
             +  \underbrace{V_{\text{ctx}}(\xi, x)}_{\text{context coupling}}.
$$

Specific parametric forms (a starting point, to be refined):

- **Wells** (one per concept anchor $k$):
  $V_{\text{wells}}(x) = \sum_k \mathfrak{m}\_k \upsilon_k^2 \bigl(1 - e^{-\kappa_k^2 \lVertx - x_{c,k}\rVert^2}\bigr)$.
  Parameters per well: centre $x_{c,k}$, width $\kappa_k$, depth
  $\upsilon_k$, mass $\mathfrak{m}_k$.

- **SARF** (semantic attractor-repellor field): pairwise interactions
  with anisotropy, parameters control coupling strength, range,
  directional alignment.

- **PARF** (property field): scalar-valued properties (POS, syntactic
  role, sentiment, register) modulate force magnitudes via couplings
  $\lambda_p$.

- **Context coupling** $V_{\text{ctx}}(\xi, x)$: explicit non-autonomy
  through $\xi$. Simplest form: a quadratic-bilinear coupling
  $\frac{1}{2}(x-\xi)^\top \Lambda (x-\xi)$ with learnable $\Lambda$.

The integrator is fixed to **damped semi-implicit Euler** at the
training step size; the velocity-Verlet finding of §14.15 (better
integrator → coarser, less content-bearing attractors) is a *prior*
that says we should not over-engineer the integrator.

The **readout** $x_L \mapsto p(\text{token})$ is, at v0, a tied
nearest-neighbour decoder over a fixed vocabulary embedding
$\{e_v\}_v \subset \mathbb{R}^d$:
$p(v \mid x_L) \propto \exp(\beta  e_v^\top x_L)$.
Whether $\{e_v\}$ is fixed (corpus-derived) or RL-calibrated is
itself a parameter-classification decision (§4).

This is the **complete v0 EOM**. Writing it in pseudocode is the
first concrete deliverable of the programme (§9, milestone M0).

---

## 4. Parameter classification: static / RL-calibrated / hyperparameter

A healthy v0 design should classify parameters into three buckets,
with explicit estimators or calibration paths for each:

### 4.1 Static (corpus statistics, no learning)

These are fixed by closed-form estimators on the corpus, calibrated
once and never updated:

- **Vocabulary embeddings** $\{e_v\}$: PMI-derived or Laplacian-eigenmap
  derived, frozen after computation.
- **Per-token semantic mass** $\mathfrak{m}_v = -\log p(v)$ (corpus surprisal),
  frozen after counting.
- **Attractor anchor positions** $x_{c,k}$: cluster centroids of
  embeddings (top-K modes of the corpus distribution), frozen after
  clustering.
- **Property-tag couplings** $\lambda_p$: per-property mutual-information
  with next-token, frozen after estimation.

Target: ~70–85% of parameters in this bucket. The *more* of the
simulator we can fix from corpus statistics, the less work RL has
to do, and the lower the sample complexity.

### 4.2 RL-calibrated (learned)

These are parameters whose role is structural rather than
distributional, where corpus statistics are insufficient or
ambiguous, and where the calibration target is naturally
trajectory-level:

- **Well depths and widths** $\upsilon_k, \kappa_k$: the *strength*
  of an attractor, not its location.
- **SARF coupling tensors**: pairwise force magnitudes and
  anisotropies.
- **Damping coefficient** $\gamma$ (or per-mass-class $\gamma_v$).
- **Context-coupling matrix** $\Lambda$ in $V_{\text{ctx}}$.
- **Readout temperature** $\beta$.
- **Optionally:** small corrections to $\{e_v\}$ on top of the
  corpus-statistics initialisation.

Target: ~10–25% of parameters in this bucket. RL signal is what
calibrates these.

### 4.3 Hyperparameters (swept, not learned)

- Integration step size $\Delta t$ and horizon $L$.
- Number of wells $K$, number of SARF anchors.
- Embedding dimension $d$.
- RL algorithm choices (entropy bonus, exploration schedule).

Target: ~5% of parameters, swept on a coarse grid.

**The classification itself is a research artifact.** Half the
intellectual content of the programme is justifying *which* parameters
fall into (4.1) vs (4.2). A defensible classification table is
deliverable M1.

---

## 5. Toy-task design (the first experiment)

The first experiment must be **constructive, ground-truth-known, and
small enough to debug end-to-end in days, not weeks.**

Recommendation: a **probabilistic context-free grammar (PCFG) with
explicit semantic-role assignments**.

- **Vocabulary:** 50–200 tokens partitioned into syntactic categories
  (DET, NOUN, VERB, ADJ, ADV, PUNCT) and semantic categories (AGENT,
  PATIENT, INSTRUMENT, TIME, PLACE).
- **Production rules:** ~10 rules producing sentences of depth 3–5,
  with non-trivial agreement and selectional constraints (verbs select
  for agent/patient categories).
- **Corpus size:** $10^4$–$10^5$ generated sentences.
- **Ground truth:** the PCFG production probabilities, recoverable
  exactly, give a closed-form upper bound on next-token prediction.

This is a setting where attention-transformers are known to
*not* recover the true grammar without inductive bias — see the
generalisation literature on PCFGs in transformers. A successful
physics simulator on this task is therefore a clean differentiator,
not just a perplexity comparison.

**Why a PCFG and not Tiny Shakespeare?** Two reasons:

1. The corpus distribution is known exactly; calibration error can
   be measured directly against ground truth.
2. The required force-form complexity is bounded by the grammar
   complexity, so a failure to fit doesn't conflate with "the
   simulator needs more capacity."

If the simulator fits the PCFG cleanly, we widen to: regular
grammar → CFG → CFG-with-semantics → real-text-fragments. The
scaling axis is **linguistic richness**, not parameter count.

---

## 6. RL flavour sequencing

Four distinct RL substrates, in order of recommended deployment:

1. **Behavior cloning on corpus trajectories (M2).** Treat token
   sequences as expert demonstrations, embed them in semantic space,
   minimise a divergence between simulator trajectories and embedded
   corpus trajectories. This is the closest analog to supervised
   pretraining and the highest-SNR regime. Equivalent to GAIL/DAgger
   in framing. Cheapest first; if this fails, the proposal fails.

2. **Intrinsic-reward RL for trajectory properties (M3).** Reward the
   simulator for satisfying framework-internal regularities the
   paper argues should hold: trajectory predictability, energy
   conservation modulo damping, attractor-stability at the training
   horizon, low residual STP loss. This directly calibrates the
   simulator to satisfy R1–R6 of §14.8 as explicit reward signals.

3. **Task-reward RL (M4).** Define reward as task performance
   (next-token NLL on held-out, classification accuracy on linguistic
   probes). Higher variance, sparser signal, but allows reward-shaped
   calibration that supervised loss can't easily express. Use as a
   fine-tuning layer on top of (1)+(2).

4. **Distillation from a trained transformer (optional, M5).** Use a
   small pretrained transformer's hidden-state flow as ground truth
   and calibrate the simulator to imitate it. Closest to existing
   matched-baseline experiments (§13), but framed as RL. *Highest
   SNR but doesn't free us from the transformer*; useful as a sanity
   check that the simulator architecture is expressive enough, not
   as a primary calibration path.

The discipline is to **stop at the lowest-flavour calibration that
works.** If behavior cloning alone calibrates the simulator to
within 2× of baseline perplexity on the toy task, declare victory
and move to scaling the linguistic richness, not the RL
sophistication.

---

## 7. Baselines

At each scale, three baselines at matched parameter count:

- **Trigram model.** The pure-statistics floor. If we can't beat
  trigram, the physics is adding nothing.
- **Tiny SPLM.** Our own architecture from §13–14. The "internal
  baseline" — does the explicit simulator match the implicit
  simulator?
- **Tiny transformer.** The external baseline. Closes the question
  "is structure helping vs hurting?"

**The claim we need to demonstrate is not "physics simulator > all
baselines on perplexity."** That claim almost certainly fails. The
demonstrable claim is:

> At matched parameters and matched compute, the physics simulator
> achieves perplexity ≥ trigram and within 2× of tiny-SPLM/transformer,
> *while dominating them on interpretability metrics*.

Interpretability metrics include: number of decodable attractors,
basin-purity by content-token fraction, force-decomposition
attribution stability, ablation cleanliness (zero-out one force
term, observe predictable behaviour change).

---

## 8. Evaluation metrics

**Quantitative (calibration quality):**

- KL divergence between simulator and corpus next-token
  distribution.
- Trajectory match: average squared distance between simulator
  trajectory and embedded corpus trajectory at each step.
- Energy drift over $L$ integration steps (should be bounded by
  damping, not unbounded).
- Sample complexity: number of trajectory samples to reach a
  perplexity threshold.

**Qualitative (interpretability):**

- Attractor decomposition: for each prompt, list the basins reached
  by the damped flow at horizon $L$, decoded through the readout.
  This should reproduce the §14.15 finding that prompt-dependent
  multi-basin structure exists.
- Force-decomposition: at each step, report the contribution of
  each named force term to the gradient. A well-calibrated
  simulator should have all terms making non-trivial contributions
  (no single force dominating, no force collapsing to zero).
- Counterfactual: zero out one force term, re-run, observe
  predictable degradation (e.g., zeroing PARF removes
  property-conditioning sensitivity).

**Falsification (the most important):**

- Where does the simulator *systematically fail* on the toy task?
  Specific syntactic constructions, specific semantic-role
  assignments, specific dependency lengths. A clean failure mode
  is more scientifically informative than a partial success.

---

## 9. Milestone schedule

| Milestone | Deliverable | Effort | Decision gate |
| --- | --- | --- | --- |
| M0 | `Semantic_Simulator_EOM.md`: complete v0 EOM as one block of equations + pseudocode | 1–2 days | Are the equations under-specified? |
| M1 | Parameter-classification table: every parameter assigned to static / RL / hyper, with named estimator | 2–3 days | Is the (4.1)/(4.2) ratio defensible? |
| M2 | Behavior-cloning experiment on a 100-token PCFG | 1–2 weeks | Does the simulator fit the PCFG to within 2× of trigram perplexity? |
| M3 | Intrinsic-reward RL refinement on the same task | 1 week | Does intrinsic-reward calibration reduce KL further without overfitting? |
| M4 | Scaling to CFG-with-semantics, then to a 1000-token sub-domain corpus (e.g., Tiny Shakespeare cleaned to fixed vocabulary) | 2–4 weeks | Does the calibration scale, or does sample complexity explode? |
| M5 | Comparison against tiny-SPLM and tiny-transformer at matched params on the same sub-domain | 1 week | Within 2× perplexity? Dominant on interpretability? |
| M6 | Programme document v1.0 (this file → expanded) and a companion arXiv preprint draft | 2 weeks | Go/no-go for full paper |

**Total time-to-decision: 2–4 weeks for go/no-go on M2; 8–12 weeks
for full programme assessment through M5.**

The programme is paused or pivoted at any milestone where the
gate fails. Specifically: failure at M2 (behavior cloning can't fit
a PCFG) means the proposal as currently formed is wrong, and the
specific failure mode dictates the pivot — usually toward enriching
the force-form vocabulary (4.2) or relaxing the static-coordinates
assumption (4.1).

---

## 10. Beyond M5: structure-lifecycle extensions (v1.5 / v2 / v3)

The M0–M5 schedule above defines a **closed-vocabulary,
fixed-anchor** simulator. v0 deliberately bounds expressivity at the
static parameter count. The framework as currently described in the
paper is *not* a closed-vocabulary system in its full intent — it
has explicit room for three structural mechanisms that v0 omits and
that are needed to model the productive expressivity of human
language. They are named here as deferred-but-planned extensions, so
the programme has somewhere to go after the toy demonstration and so
v0's known expressivity ceiling does not get mistaken for the
framework's ceiling.

The full specification of each mechanism lives in
`docs/Semantic_Simulator_EOM.md` §13.

### 10.1 v1.5 — Destruction / salience decay (cheapest, prototype-first)

**Why first.** Single scalar per particle, large computational
payoff (bounds state growth), can be added to v1 without redesigning
the integrator. Tests the framework's claim to handle long-range
discourse phenomena.

**Concrete test.** A deliberately long-context toy task (a
generated discourse where topics shift and return) where v1 (no
destruction) would collapse but v1.5 (destruction + re-promotion)
should handle cleanly. If v1.5 succeeds, that is strong evidence
that the framework's "structures decay and re-promote" picture
matches a real linguistic phenomenon, and the v2 / v3 work becomes
much more justified.

**Effort.** ~1–2 weeks on top of a working v1.

### 10.2 v2 — Creation of new semantic structures

**What it adds.** Binding rules: when two particles enter a specific
configuration, they fuse into a composite particle with inherited
attractors and an RL-calibrated correction. Enables productive
combinatorics (compounding, novel metaphor, idiomatic
crystallisation).

**Why this is the heavier lift.** Triggering rules and inheritance
rules are discrete and non-differentiable; calibration shifts from
gradient descent to policy gradient.

**Effort.** ~4–8 weeks of focused work, conditional on v1.5
succeeding. Requires a v2 EOM specification document analogous to
`Semantic_Simulator_EOM.md`.

### 10.3 v3 — Operator-valued execution

**What it adds.** Particles carry transformation operators
$\hat{O}$ that act on other particles' states when they enter
operating range. The simulator's analogue of function application.
Encodes non-commutative composition (essential for
verb-argument structure).

**Mathematical infrastructure already in place.** Lie-group
machinery cited in the paper (`Gueorguiev2025LieGroups`,
`BaezMuniain1994`, `Nakahara2003`) is the natural toolkit. The
framework inherits a half-century of operator-algebra mathematics
once it goes this route.

**Effort.** ~8–12 weeks of focused work, conditional on v2
succeeding.

### 10.4 What v0 → v3 buys, and what it doesn't

If all three mechanisms are added cleanly, the simulator becomes a
**dynamical system on a state space that itself evolves**.
Expressivity is no longer bounded by the static parameter count, and
the upper bound moves from finite to potentially Turing-complete.

This shifts the honest probability of the simulator being a
viable frontier-scale alternative to transformers from **~zero
(v0)** to **~5–15% (v0 + v1.5 + v2 + v3)**. That is a real
update, not a rhetorical one. It is also not a high probability — it
is "open question, defensible programme" rather than "this will
work."

What v0 → v3 does *not* buy is matching the **memorisation**
fraction of transformer behaviour — verbatim factoids, idioms,
training-data formats. A productivity-focused simulator will likely
cap below transformers on benchmarks dominated by recall (TriviaQA,
factual MMLU) regardless of mechanism quality. This is structural,
not a calibration artefact.

### 10.5 Decision discipline

The v1.5 / v2 / v3 staging is **strictly conditional** on the
preceding stage succeeding. If v0 fails (M2 gate), v1.5 / v2 / v3
are not pursued — the right move is to revise v0 first. If v1
succeeds, v1.5 is the natural prototype. v2 is launched only on a
successful v1.5; v3 only on a successful v2. At every stage, the
go / no-go is a milestone-decision-gate, not enthusiasm.

---

## 11. Honest framing and known risks

**What this is:** a structured-prior, RL-calibrated, physics-informed
generative model of language. It belongs to the same broad family as
neural force fields for molecular dynamics, physics-informed neural
networks, and energy-based models with explicit functional forms.
What's *novel* is the specific Lagrangian content (SARF + PARF +
wells + semantic mass) inherited from the framework as the structural
prior.

**What this is not:** a replacement for transformer-based language
models at frontier scale. Effectively zero chance, regardless of
calibration sophistication. The corpus encodes too much idiosyncratic,
non-physical regularity for a fixed-functional-form simulator to
match.

**Known risks the programme must confront:**

1. **Sample complexity at scale.** Toy-language calibration with
   $10^4$ samples does not imply real-corpus calibration with $10^9$
   samples. RL-calibrated systems are typically *more* sample-hungry
   than supervised. M4 is the milestone that exposes this.

2. **Structural-rule sample complexity (v2 / v3 specific).**
   Discrete creation/destruction events and operator-valued
   transformations are non-differentiable; they can only be
   calibrated by policy-gradient-style RL. RL of structural
   (discrete) rules is dramatically more sample-hungry than RL of
   continuous coefficients. The 3–4 orders-of-magnitude
   parameter-count advantage of the v0 simulator over a transformer
   probably erodes to 1–2 orders of magnitude once v2 / v3
   mechanisms come in. Whether that residual advantage is enough is
   the central empirical question of the v2 / v3 stages.

3. **Reward design.** "Match corpus trajectory in semantic space"
   requires defining the embedding *and* the trajectory metric.
   Both choices have ML-content; neither is pure physics. The
   programme should treat reward design as a first-class research
   question, not a hyperparameter.

4. **MDP framing of language is not airtight.** Long-range
   dependencies, hierarchical structure, and discrete-symbolic
   content do not always fit cleanly into "state + action + reward."
   Specific linguistic phenomena (anaphora, ellipsis,
   centre-embedding) may require non-RL calibration substrates.

5. **The Lagrangian content may be incomplete.** SARF + PARF +
   wells + mass may not span the space of force forms language
   actually requires. M2/M3/M4 will surface specific missing
   capabilities; the programme must be willing to either extend
   the force vocabulary (and incur parameter explosion) or restrict
   the linguistic scope.

6. **Compositional rule coverage is open-ended (v2 specific).**
   Language has many composition rules — syntactic, semantic,
   pragmatic, idiomatic. There is a real risk of a "Tower of Babel"
   failure where each new linguistic phenomenon requires its own
   binding rule and the rule count grows linearly with corpus
   coverage. v2 must monitor for this and be willing to bound
   linguistic scope rather than let the rule count explode.

7. **Frontier scale is partly memorisation, not productivity.** A
   non-trivial fraction of frontier-transformer behaviour is
   verbatim memorisation of training data — common phrases, formats,
   factoids. A productivity-focused simulator will replicate the
   *generative* part of language but probably not the memorised
   part. The simulator may cap below transformers on benchmarks
   dominated by recall (TriviaQA, factual MMLU) regardless of
   mechanism quality. This is structural, not a calibration
   artefact, and should be acknowledged in any honest write-up of
   the v2 / v3 results.

8. **The interpretability claim must hold under stress.** A
   simulator with $10^5$ RL-calibrated parameters that produces
   uninterpretable trajectories is no better than a small
   transformer. The interpretability metrics (§8) are the second
   gate alongside perplexity, and failure on either is a programme
   failure.

---

## 12. Relationship to the SPLM/TMLR line

This programme is **parallel**, not competing:

- The SPLM/TMLR line establishes that the framework is empirically
  realised in trained transformers. It is the existence proof.
- The simulator line tests whether the framework is *directly
  executable* without learning the architecture from scratch. It is
  the constructive proof.

Either alone is a contribution. Together they would constitute
"the dynamics is real, and here are two independent ways to access
it" — a substantially stronger scientific position than either
alone.

The submission strategy is:

1. **TMLR submission** of the existing 126-page paper proceeds on
   its current timeline. No reordering.
2. **Programme M0–M2** runs in parallel (2–3 weeks of focused
   work, can be evening/weekend if the day job is the SPLM line).
3. **Companion arXiv preprint** at M5/M6 if the toy experiments
   succeed; otherwise the negative result becomes a section in a
   future paper rather than a standalone submission.
4. The two papers cite each other and build a coherent two-track
   programme.

---

## 13. First action

**M0 is done.** `docs/Semantic_Simulator_EOM.md` exists; it contains
the complete v0 equations of motion as one block of equations and
pseudocode, separate from the paper's prose. The act of writing it
surfaced seven under-specifications (§12 of that document) that the
framework's narrative form left implicit. It also drove the
discussion that produced the structure-lifecycle extensions of §10.

The artifact is in the repository regardless of whether the larger
programme is pursued. **Next action is M1**:

> Verify the parameter-classification table of
> `Semantic_Simulator_EOM.md` §10 by running each static estimator on
> a small corpus and confirming the output shapes and ranges.
> Fixed cost, ~1–2 days.

After M1, M2 (behaviour cloning on a 100-token PCFG) becomes the
first real experiment. M2's outcome — within 2× of trigram
perplexity, or not — is the first programme-level decision gate.

---

*End of programme memo. The v0 EOM is specified in
`docs/Semantic_Simulator_EOM.md` (M0, complete). Next action:
M1 — verify the parameter-classification table on a small corpus
(~1–2 days). The structure-lifecycle extensions (v1.5 / v2 / v3) of
§10 are on record as deferred-but-planned milestones, not
last-minute additions; their go/no-go gates are conditional on the
preceding v# stage succeeding.*
