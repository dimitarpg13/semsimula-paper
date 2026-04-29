# Advancing the Dynamic-Simulation Model: Expressivity Bounds and the Mathematical Apparatuses for v1.5 / v2 / v3

Companion to `docs/Semantic_Simulator_EOM.md` and `docs/Semantic_Simulator_RL_Calibration_Programme.md`. Drafted 2026-04-26 as the conceptual scaffold underpinning the v1.5 / v2 / v3 extensions named in the programme memo §10. The technical 5-page note `docs/Expressivity_Bounds_For_v0_Simulator.md` is the formal companion to this report.

---

## Abstract

The v0 dynamic-simulation model specified in `Semantic_Simulator_EOM.md` is a continuous-state, finite-dimensional, smooth-dynamics system with damping. The programme memo (`Semantic_Simulator_RL_Calibration_Programme.md` §10) names three structural extensions — *salient decay* (v1.5), *creation/destruction of semantic structures* (v2), and *operator-valued execution* (v3) — as deferred milestones whose purpose is to lift expressivity beyond what v0 alone can deliver. The case for each was, in the programme memo, motivated qualitatively. This report sharpens the argument in four ways. First, it gives a formal four-step proof that the v0 class is at most a finite automaton: phase-space capacity is bounded by $\dim M \cdot \log_2(L/\epsilon)$ bits; damping at rate $\gamma$ is information-destroying with exponential contraction; the simulator's potential $V$ is smooth and non-chaotic, ruling out the Siegelmann–Sontag construction of universal computation in continuous flows; consequently the v0 simulator accepts at most regular languages. Second, it specifies a decisive experiment — language recognition on $\mathrm{Dyck}_n$ at controlled nesting depth — that produces a *predicted, measurable* collapse depth $D^\ast$ from these formal bounds, and constructs nested variants of the experiment (Dyck + topic-shift, Dyck + let-binding, cross-serial $a^n b^n c^n$, bounded copy $ww$, and a deliberately-too-hard 2-counter task) that *separately* falsify v0, v0+v2-without-v1.5, v0+v2+v1.5-without-v3, the MCS-reach claim itself, and an over-strong v3. Third, it maps each of the three extensions onto a mature mathematical apparatus: salient decay onto dissipative semigroups and discounted Markov decision processes; creation/destruction onto Fock space and the canonical creation/annihilation operator algebra; execution onto Lie groups acting via non-abelian gauge fields. Fourth, and centrally, it argues that the composite v0+v1.5+v2+v3 reaches **exactly** the **mildly context-sensitive** (MCS) class — the empirically established class for human language (Joshi 1985, Shieber 1985) — by an explicit reduction to Linear Context-Free Rewriting Systems of bounded fan-out and rank. The reduction is sketched here; the full proof is the subject of the companion document `docs/MCS_Reduction_For_v3_Composite.md`. The composite system is also identified, in the algebraic-physics tradition, as a classical-mechanical analogue of a Haag–Kastler-style local operator algebra. Cellular automata, considered separately, supply a complementary lower-bound existence proof (locality is sufficient for universal computation provided state space is unbounded) but are not the right primary apparatus for a continuous-state system.

---

## 1. Background and motivation

`Semantic_Simulator_EOM.md` specifies the v0 simulator as a damped Euler–Lagrange flow on a finite-dimensional semantic space $\mathbb{R}^d$:

$$
\mathfrak{m}_\ell \ddot{x}_\ell = -\nabla_x V(\xi_\ell, x_\ell) - \gamma \dot{x}_\ell
\qquad(1)
$$

with $V = V_{\mathrm{wells}} + V_{\mathrm{SARF}} + V_{\mathrm{PARF}} + V_{\mathrm{ctx}}$, fixed-vocabulary readout via tied nearest-neighbour decoder, and parameter classification into static (~98%) / RL-calibrated (~2%) buckets at v0 toy scale.

The programme memo (`Semantic_Simulator_RL_Calibration_Programme.md` §10) identifies three structural extensions as needed for the framework to model the productive expressivity of human language:

- **v1.5: salient decay / destruction.** A salience scalar per particle, with multiplicative decay and re-promotion, formalising the framework's prediction that semantic structures decay and re-promote across discourse.
- **v2: creation of new semantic structures.** Binding rules that fuse particles into composite entities, formalising productive compounding, novel metaphor, and idiomatic crystallisation.
- **v3: operator-valued execution.** Particles that carry transformation operators acting on other particles' states, formalising verb-argument structure and predicate composition.

The programme memo presents these as plausible mechanisms whose justification is qualitative. This report supplies the formal scaffold the programme memo defers: a proof that v0 is intrinsically bounded, a measurable falsifier, and a mathematical apparatus for each extension.

The motivation for sharpening the argument now, before any of v1.5 / v2 / v3 is implemented, is twofold. First, an experimentally falsifiable expressivity ceiling is a stronger basis for the staged programme than a qualitative one — it tells us *exactly* where each extension is required and lets us measure, rather than declare, the sufficiency of any candidate v0+. Second, naming the mathematical apparatus for each extension up front lets the programme inherit existing theorems (stability, ergodicity, mean-field limits, conservation laws, gauge-equivariance) rather than rederiving them.

---

## 2. Formal class of the v0 simulator

The v0 state $s_\ell$ — with components position $x_\ell$, velocity $\dot{x}_\ell$, context $\xi_\ell$, mass $\mathfrak{m}_\ell$, and parameters $\theta_\ell$ — lives in a fixed manifold $M \subseteq \mathbb{R}^{\dim M}$ of bounded volume, where $\dim M = O(d) + O(K) + O(N_S) + O(P)$ with all dimensions fixed at calibration time. The vector field $F$ on $M$ — the right-hand side of (1) — is $C^\infty$ and a sum of fixed-form smooth functions: Gaussian potential wells, anisotropic SARF terms with bounded gradients, bilinear context coupling $\frac{1}{2}(x-\xi)^\top\Lambda(x-\xi)$, and linear PARF couplings (see `Semantic_Simulator_EOM.md` for the full expressions). In particular:

- $\dim M$ is fixed and does not grow with input length.
- The cast of particles is fixed at the vocabulary $V$ plus the calibration-time anchors; no inference-time mechanism creates or destroys particles.
- The functional form of $V$ has bounded Hessian almost everywhere; the local Lyapunov exponents are bounded, and $\gamma > 0$ damping makes the global Lyapunov spectrum negative-on-average.
- The non-autonomy enters only through $\xi_\ell$, which is itself an element of $\mathbb{R}^d$ updated by causal averaging. $\xi$ adds no new degrees of freedom.

The integrator is fixed to damped semi-implicit Euler at the training step size; the integrator itself is a contraction at any $\gamma > 0$.

This is the formal class. Its expressivity is what §3 bounds.

---

## 3. The expressivity ceiling: a four-step formal argument

### 3.1 Phase-space capacity

At any working precision $\epsilon > 0$, the number of distinguishable states in $M$ is bounded by the metric entropy:

$$
N_\epsilon(M) \le \mathrm{vol}(M) / \epsilon^{\dim M},
\qquad
\log_2 N_\epsilon(M) \le \dim M \cdot \log_2(L_M/\epsilon)
\qquad(2)
$$

where $L_M$ is the diameter of $M$. The total bits available to represent the simulator's instantaneous state is $O(\dim M \cdot \log(1/\epsilon))$. For the v0 toy scale ($d=64$, $L=8$, $K=32$, $N_S=16$, $P=5$), and at $\epsilon = 10^{-6}$, this gives roughly $\dim M \approx 200$ and a state capacity of $\approx 4000$ bits. This is *fixed*; it does not grow with input length.

### 3.2 Damping is anti-memory

Under the flow defined by (1), phase-space volume contracts. The divergence of $F$ is $\nabla \cdot F = -\dim M \cdot \gamma + O(\nabla^2 V)$, and at typical operating points the trace of $\nabla^2 V$ is small relative to the damping term, so

$$
\dot V_M \approx -\dim M \cdot \gamma \cdot V_M
\qquad\Longrightarrow\qquad
V_M(t) \approx V_M(0) \cdot e^{-\dim M \cdot \gamma \cdot t}.
\qquad(3)
$$

The accessible phase volume after $L$ integration steps shrinks exponentially. The maximum information about the initial condition that can be retrieved at step $\ell$ is

$$
I(s_0; s_\ell) \le \dim M \cdot \log_2(L_M/\epsilon) - \frac{\ell \cdot \dim M \cdot \gamma}{\ln 2}
\qquad(4)
$$

bits. For our toy ($\dim M \approx 200$, $\gamma \approx 1$, $\ell = L = 8$) this leaves roughly $4000 - 2300 = 1700$ effective bits at horizon — and crucially, the second term grows linearly in $\ell$ while the first does not. The simulator is structurally a **finite-memory device with active forgetting**.

The context variable $\xi_\ell$ partially compensates by aggregating earlier states into a causal mean, but $\xi_\ell \in \mathbb{R}^d$ has $O(d)$ capacity and the averaging operation is itself a contraction.

### 3.3 The functional class of $V$ is non-chaotic

Siegelmann and Sontag (1991, 1995) showed that a continuous-time recurrent system over $\mathbb{R}^n$ with rational weights and arbitrary precision is Turing-complete — but the construction relies critically on (i) genuinely chaotic dynamics with positive Lyapunov exponent and (ii) infinite-precision real-valued state. Our $V$ is a sum of Gaussian wells, anisotropic exponentials, and bilinear forms; its largest Lyapunov exponent is bounded, and it is made smaller still by damping. The Siegelmann–Sontag construction does not apply.

Even setting damping aside, smooth non-chaotic flows on a compact manifold have $\omega$-limit sets that are unions of equilibria, periodic orbits, and recurrent invariant sets of bounded complexity (Smale–Palis–Pugh structural-stability theorems for generic Morse–Smale systems). They do not encode unbounded counters or stacks.

### 3.4 Chomsky-hierarchy placement

Combining (3.1)–(3.3): at any practical precision $\epsilon$ and any practical $V$ in our class, the v0 simulator implements a deterministic finite automaton with at most $N_\epsilon(M) = 2^{O(\dim M \log(1/\epsilon))}$ states. By Kleene's theorem this accepts exactly the regular languages.

> **Theorem (informal).** *The v0 simulator class accepts at most regular languages. Any language strictly above the regular tier in the Chomsky hierarchy — context-free, context-sensitive, recursively enumerable — is structurally outside its expressivity range.*

This is a *formal* claim, not an empirical one. v0 cannot be made context-free by adding parameters, by adding RL signal, or by enriching the force vocabulary, because none of those operations grow the state-space dimension during inference.

### 3.5 The empirical target: mildly context-sensitive grammars

The v0 ceiling (regular) is a *lower bound* on what we need. The *upper bound* is fixed not by mathematics but by linguistic empirics. Two foundational results constrain the target:

- **Shieber (1985)** demonstrated that Swiss German cross-serial dependencies require strictly more than context-free power. Natural language is *not* context-free.
- **Joshi (1985)** and successors (TAGs, CCGs, LCFRSs, MCFGs, head grammars) identified a class — **mildly context-sensitive (MCS)** — that captures the cross-serial constructions while remaining (i) closed under intersection with regular languages, (ii) parsable in polynomial time, and (iii) constant-growth (the lengths of derivable sentences grow as a constant-rate semilinear set). MCS is the empirically established class for human language and the de facto target for any computational model of natural-language generation.

This sets the framework's target. The composite v0+v1.5+v2+v3 system should reach **exactly** MCS — not less (which would fail Shieber's argument and many concrete linguistic phenomena), and not more (which would silently admit Turing-complete pathologies that human language does not exhibit). §5.5 below sketches the argument that MCS is, in fact, where the composite system lands; the full reduction is the subject of `docs/MCS_Reduction_For_v3_Composite.md`. §4.5 specifies the empirical falsifiers — cross-serial $a^n b^n c^n$, bounded copy $ww$, and a Minsky 2-counter task — that together test both the lower-bound staircase and the MCS upper bound.

The expressivity story for the framework is therefore **a four-tier ladder**:

| Tier              | Class            | Mechanism                          | Falsifier        |
| ----------------- | ---------------- | ---------------------------------- | ---------------- |
| v0                | regular          | smooth dynamics, finite phase space | F1 (Dyck base)  |
| v0 + v2          | context-free     | particle creation                  | F2 (Dyck+topic) |
| v0 + v2 + v1.5   | context-free     | + decay (orthogonal to class)      | F3 (Dyck+let)   |
| v0 + v2 + v1.5 + v3 | mildly context-sensitive | + operator action $G$ on particles | F4, F5 (cross-serial, copy) |
| (over-strong v3)  | Turing-complete  | unbounded operator algebra         | F6 (2-counter)  |

The bottom tier is established (§3); the top tier is the object of the next section.

---

## 4. A decisive experiment: $\mathrm{Dyck}_n$ at controlled depth

### 4.1 Why $\mathrm{Dyck}_n$ is the right falsifier

The claim of §3 is the v0 simulator is at most regular. The cleanest empirical falsifier is a language that is *strictly* above regular and in which the gap is parameterised by a single, controllable variable:

- $\mathrm{Dyck}_n$ for $n \ge 2$ — strings of balanced parentheses over $n$ bracket types — is **deterministic context-free**.
- Recognising $\mathrm{Dyck}_n$ requires a stack of depth at least equal to the maximum nesting depth $D$ of the string.
- A finite automaton with $|Q|$ states accepts $\mathrm{Dyck}_n$ correctly only up to depth $D \le \log_n |Q|$. Beyond that depth, the automaton's accuracy falls to chance.
- The minimal computational requirement is therefore *exactly* a stack of varying depth — no other context-free phenomenon (e.g. anaphora, agreement) is conflated with it.

Tiny transformers and tiny LSTMs solve $\mathrm{Dyck}_n$ to substantial depth; this is a well-replicated finding (Hewitt et al. 2020; Yao, Peng, Papadimitriou & Steinhardt 2021). The failure of v0 at depth $D^\ast$, while transformer/LSTM baselines at matched parameters succeed past $D^\ast$, isolates the bottleneck to the v0 *class*, not to the task or the parameter budget.

### 4.2 Experimental setup

- **Vocabulary.** $\{(_1, )_1, (_2, )_2, \ldots, (_n, )_n, \text{BOS}, \text{EOS}\}$, with $n \in \{2, 3\}$.
- **Generator.** A stochastic grammar producing balanced strings of controlled maximum depth $D \in \{1, 2, 4, 8, 16, 32\}$, with at least $10^4$ training and $10^3$ held-out strings per depth.
- **Task.** At each closing-bracket position, predict the type of the closing bracket (the only context-free decision; positional decisions are regular and trivially solvable).
- **Models at matched parameters and matched compute.**
  1. v0 simulator (the static + RL-calibrated configuration of `notebooks/semsim_simulator/`).
  2. Tiny transformer (single block, attention + MLP, parameter count matched to v0).
  3. Tiny LSTM (single layer, hidden size matched).
- **Scoring.** Per-position accuracy as a function of nesting depth $D$.

### 4.3 The collapse-depth formula

Combining the bit-counting argument of §3.1 (capacity) with the damping argument of §3.2 (forgetting), the maximum nesting depth a v0 simulator can correctly handle is

$$
D^\ast(d, L, \gamma, n) \sim \frac{\dim M \cdot \log_2(L_M/\epsilon)}{\log_2 n} - \frac{L \cdot \dim M \cdot \gamma}{\ln 2 \cdot \log_2 n}
\qquad(5)
$$

— the available bits, divided by the bits per stack frame ($\log_2 n$). For our toy ($\dim M \approx 200$, $\epsilon = 10^{-6}$, $L = 8$, $\gamma \approx 1$, $n = 2$), this gives $D^\ast \in [3, 6]$. For $n = 3$, $D^\ast$ is correspondingly smaller.

The experiment is decisive at the level of *both the qualitative claim* (v0 collapses, transformer/LSTM does not) *and the quantitative claim* (the collapse depth matches the prediction within constant factors).

### 4.4 Predicted outcome (the falsifiable table)

| Depth $D$ | Tiny transformer | Tiny LSTM | v0 simulator |
| --------- | ---------------- | --------- | ------------ |
| 1         | ~100%            | ~100%      | ~100%        |
| 2         | ~100%            | ~100%      | ~100%        |
| 4         | ~100%            | ~100%      | ~95%         |
| 8         | ~99%             | ~99%       | chance (1/n) |
| 16        | ~95%             | ~90%       | chance       |
| 32        | ~80%             | ~70%       | chance       |

If observed v0 accuracy collapses to chance at $D \approx D^\ast$ from (5), the formal expressivity bound of §3 is *empirically confirmed*. If v0 surprisingly succeeds at $D \gg D^\ast$, the formal bound is wrong and §3 must be revised — the experiment is structured so that *either outcome is informative*.

### 4.5 Nested falsifiers separating v1.5 / v2 / v3

The base $\mathrm{Dyck}_n$ experiment falsifies v0 alone. Three nested variants separately falsify partial v0+ designs:

- **F1 — base $\mathrm{Dyck}_n$** falsifies v0 → motivates **v2**. With creation/destruction, each $($ instantiates a new "open-bracket" particle whose state encodes the bracket type; each $)$ removes the most recently created. The state-space dimension grows linearly with depth, and the automaton class lifts from regular to deterministic context-free.

- **F2 — $\mathrm{Dyck}_n$ + topic-shift** falsifies v0+v2 (without decay). The grammar is augmented with long stretches of "filler" tokens between matching brackets that should not interfere with bracket-matching. v2 alone proliferates particles indefinitely; without a decay or destruction mechanism, the active-particle set grows unbounded and competes for the simulator's compute budget. Salient decay (v1.5) gates dormant particles out of the active dynamics — formalising the framework's prediction that semantic structures retire when not in active context.

- **F3 — $\mathrm{Dyck}_n$ + let-binding** falsifies v0+v2+v1.5 (without operators). The grammar admits *variable substitution*: a let-statement reassigns one bracket type to another mid-string ("hereafter $(_1$ means $(_2$"). A purely particle-based v2+v1.5 has no mechanism to *transform* the state of an existing particle in response to such a statement; only operator-valued execution (v3) supplies the needed action. The let-statement is realised as an operator $\hat O_{1\to 2} \in G$ acting on the open-bracket particle's representation.

The first three falsifiers establish the v0 → v0+v2 → v0+v2+v1.5 → v0+v2+v1.5+v3 staircase **from below**: each tier has a stricter linguistic phenomenon that the previous tier cannot do. The next two falsifiers test the **upper-bound claim** that the composite system reaches *exactly* mildly context-sensitive (MCS) and not above. Both are well-known formal-language tasks for which MCS-class machines succeed and CF-class machines fail.

- **F4 — cross-serial dependency $a^n b^n c^n$** confirms v0+v2+v1.5+v3 reaches MCS. The language $\{a^n b^n c^n : n \ge 1\}$ is the canonical non-context-free, mildly context-sensitive language. A pushdown automaton (CF) cannot recognise it; a tree-adjoining grammar, an LCFRS, or a 2-stack machine can. The framework's prediction is that v0+v2+v1.5+v3 succeeds at $a^n b^n c^n$ up to depth comparable to the Dyck collapse depth $D^\ast$, while v0+v2+v1.5 (without operator action) collapses to chance for $n \ge 2$. The mechanism is that the v3 operator action reads the count of "$a$" particles and *re-uses* it to license the corresponding $b$ and $c$ counts via group-element coupling, mirroring the LCFRS rule that emits paired fragments from a single non-terminal.

- **F5 — bounded copy language $ww$** confirms MCS reach via a different construction. $\{ww : w \in \Sigma^\ast, |w| \le L\}$ is non-CF for $|\Sigma| \ge 2$ but is in MCS (it is the canonical example for LCFRS-style reduplication). The framework's prediction is that v0+v2+v1.5+v3 succeeds up to length $L$ matching the operator-bandwidth bound, again via v3-mediated re-use of the encoded prefix.

The composite system passing F4 and F5 — at depths bounded by the same $D^\ast$ formula adjusted for the MCS-class capacity — is the empirical confirmation that the framework reaches MCS.

- **F6 — 2-counter machine simulation (intentional over-strong)** is the upper-bound falsifier. A Minsky 2-counter machine is Turing-complete; the corresponding language family is recursively enumerable. The prediction is that v0+v2+v1.5+v3 *fails* at this task — specifically, that it cannot reliably increment, decrement, and zero-test two unbounded counters in arbitrary sequence. **A success at F6 is a failure of the MCS claim**: it would mean the design space of v3 operators is *too large* and the framework over-generates, admitting languages humans do not exhibit. In that case the v3 specification (the choice of $G$, its rank, the operator-arity bound) must be tightened. F6 is the falsifier that protects the *upper bound*, not the lower one — and is therefore the most important formal-language experiment for the long-term theoretical health of the framework.

This nested falsifier sequence is the empirical analogue of the v0 → v1.5 → v2 → v3 staging in the programme memo. Each tier has its own measurable failure mode and its own measurable resolution. *The order of necessity (v2 before v1.5, v1.5 before v3) is fixed by which tier of the falsifier sequence the partial v0+ design fails at.* F4 and F5 add the *upper-bound confirmation* that the composite system reaches MCS, and F6 enforces that it stops there.

---

## 5. Mathematical apparatuses for the three extensions

A central scientific advantage of the proposed staging is that each of the three extensions corresponds to a mature mathematical formalism with inherited theorems. This section names each apparatus and identifies the principal theorems that become available.

### 5.1 Salient decay (v1.5) → dissipative semigroups and discounted Markov decision processes

The cleanest formalisation of v1.5 is to augment the per-particle state with a salience scalar $\sigma_\ell^{(i)} \in [0,1]$, evolving as

$$
\sigma_\ell^{(i)} = (1 - \Delta t/\tau) \cdot \sigma_{\ell-1}^{(i)} + r_\ell^{(i)}
\qquad(6)
$$

where $r_\ell^{(i)} \ge 0$ is the re-promotion signal at step $\ell$. Particles with $\sigma_\ell^{(i)} < \theta_{\min}$ are gated out of the force computation. The map $\sigma \mapsto (1 - \Delta t/\tau)\sigma$ is a strict contraction; the augmented v0+v1.5 dynamics is a **dissipative semigroup** on the extended state space.

The mature mathematical apparatuses available are:

- **Lyapunov stability theory.** The salience-augmented system admits a Lyapunov function $L = L_{\mathrm{kin}} + V + \mathrm{const}\cdot\sum_i\sigma_\ell^{(i)}$; the dynamics is exponentially stable in the limit $r_\ell^{(i)}\to 0$.
- **Attractor characterisation for dissipative systems.** Milnor attractors, $\omega$-limit sets, and global-attractor theorems (Conley index, Morse decompositions) apply directly.
- **Discounted MDPs.** Equation (6) is mathematically identical to the discount-factor equation $V(s) = r + \gamma_{\mathrm{RL}} V(s')$ of an MDP with discount $\gamma_{\mathrm{RL}} = 1 - \Delta t/\tau$. The full apparatus of dynamic programming, Bellman equations, and policy-gradient theorems is available.

Critically, salient decay does *not* change the simulator's computational class — it does not lift v0 above the regular tier. Its purpose is to make the bounded memory *usable* for long-context phenomena: a discourse where Topic A is established, drops out, and returns ten sentences later. The combinatorial improvement is large; the formal expressivity-class change is zero. This is the right v1.5 to commit to: a clean, well-understood mechanism whose role is operational rather than class-jumping.

### 5.2 Creation and destruction (v2) → Fock space and second quantisation

Variable particle number is the natural setting of mathematical physics. For a single-particle Hilbert space $\mathcal{H}$, the **Fock space**

$$
\mathcal{F}(\mathcal{H}) = \bigoplus_{n=0}^{\infty} \mathcal{H}^{\otimes n}
\qquad(7)
$$

is the direct sum over all particle counts. State sizes grow with input length. The **creation operator** $a^\dagger_v$ instantiates a particle of type $v$, the **annihilation operator** $a_v$ destroys one. Canonical (anti-)commutation relations $[a_v, a^\dagger_w] = \delta_{vw}$ encode whether particles are bosons or fermions (linguistically, whether two distinct discourse referents can co-occur or whether they must be distinguished — both have natural readings).

The mapping to v2 is direct:

| v2 mechanism                          | Fock-space object                                                        |
| ------------------------------------- | ------------------------------------------------------------------------ |
| Introduce an entity into discourse    | $a^\dagger_v \lvert\psi\rangle$                                          |
| Topic drops out of discourse          | $a_v \lvert\psi\rangle$                                                  |
| Count of currently-live entities      | number operator $N = \sum_v a^\dagger_v a_v$                             |
| Field at semantic position $x$        | $\hat\phi(x) = \sum_v \phi_v(x) a_v$                                     |
| Two-particle interaction $V(x_1,x_2)$ | $\int dx_1 dx_2 \cdot V \cdot a^\dagger(x_1)a^\dagger(x_2)a(x_2)a(x_1)$ |

The principal payoff is that **creation/destruction dynamics on a Fock space is a 90-year-old branch of mathematical physics**. Mean-field approximations, Hartree–Fock self-consistency, variational Monte Carlo, Bogoliubov transformations, the entire BBGKY hierarchy, second-quantised path integrals — all available off-the-shelf for v0+v2.

A particularly clean classical instantiation is the **Doi–Peliti formalism** (Doi 1976; Peliti 1985), originally derived for classical reaction–diffusion systems. Doi–Peliti maps stochastic configuration-number dynamics — particles at sites, reactions $A + B \to C$ — onto a Fock-space-style operator algebra without invoking quantum mechanics. This is precisely the formal setting for v2 if particles are kept classical (no superposition): each semantic-particle creation is a classical reaction, the state on $\mathcal{F}$ is a classical generating-function representation of the configuration distribution, and the field equations are classical Hamilton equations on the resulting symplectic manifold.

### 5.3 Execution (v3) → Lie groups and non-abelian gauge theory

For execution we want particles that carry transformation operators which act on other particles' states. The natural apparatus:

- A particle $i$ carries a Lie-group element $g_i \in G$.
- When particle $i$ comes within "operating range" of particle $j$, $j$'s state transforms as $x_j \mapsto g_i \cdot x_j$.
- Composition of operators is group multiplication: $(g_1 g_2) \cdot x = g_1 \cdot (g_2 \cdot x)$, with $g_1 g_2 \neq g_2 g_1$ in the non-abelian case.

The non-commutativity is exactly what verb-argument structure requires: "John gave Mary a book" $\neq$ "Mary gave John a book", even though the sets of participants are identical. Operator composition order is meaning-bearing.

The mature apparatus is **non-abelian gauge theory** — fibre bundles, principal $G$-bundles, connections, curvature tensors, holonomy. The companion notes `companion_notes/P-rot-6_transformer_dynamics.md`, `geometric_deep_learning/docs/Lie_Groups_for_Gauge_Theory_Tutorial.md`, and `geometric_deep_learning/docs/Gauge_Theory_Tutorial.md` (cited in the SPLM paper's bibliography) already lay this groundwork in the context of attention transformers. Re-using the apparatus for v3 is a one-step inheritance: instead of "the gauge field $B(x)$ is implicit in attention", we have "each particle carries an explicit $G$-valued tag, and the gauge field is the manifest interaction operator". The connection between the two readings is one of the more interesting unifications the framework makes available.

The principal theorems that become available include the **Yang–Mills equations** (governing the dynamics of the gauge field itself), **Wilson-loop holonomy** (path-dependent transformations of state along trajectories — exactly the right object for "the meaning of a sentence depends on the order of words"), and **gauge equivariance** (the simulator is invariant under a class of meaning-preserving relabellings). All of these have well-developed mathematical theory at varying levels of sophistication.

### 5.4 The combined formalism: a classical analogue of algebraic quantum field theory

Composed, the three extensions specify a system that is mathematically very specific:

> **A dissipative classical Hamiltonian dynamics on a Fock space, with a non-abelian gauge action on the underlying single-particle Hilbert space, and a salience-weighted local operator algebra on the Fock space itself.**

This is a classical-mechanical analogue of **algebraic quantum field theory** in the Haag–Kastler tradition: a system in which (i) particle number is not conserved, (ii) the algebra of observables is local (each spacetime region has its own subalgebra), (iii) symmetry groups (Lie groups) act on the algebra, and (iv) the dynamics is generated by a Hamiltonian (here, the v0 potential $V$ promoted to an operator on $\mathcal{F}$).

The framework's identification of *semantic structures* as *classical particles with mass and force interactions* maps onto a mature physical formalism whose semantic content (as a model of anything) is normally quantum-mechanical. The classical-mechanical specialisation we propose here is mathematically simpler but inherits a meaningful slice of the underlying machinery. This is the precise sense in which the v0 → v3 staging is "Newton-of-language": the framework's commitment is to the non-quantum reading of the same formal apparatus.

This identification has consequences. It is not merely rhetorical:

1. The v0 → v3 system inherits **exact and approximate conservation laws** (number, Lie-charge, energy, Lyapunov functions) and **renormalisation-group reasoning** (what happens at long discourse vs short utterance scales).
2. The interpretability metrics of §8 of the programme memo gain a precise formulation: the active particle algebra at any layer is a *named, computable, decomposable object*.
3. The v1.5 / v2 / v3 staging is no longer a list of engineering proposals; it is the natural three-step extension of v0 along three orthogonal axes in a known mathematical landscape.
4. The composite system's *generative class* — what languages it can produce — is, under modest assumptions on the operator algebra, exactly **mildly context-sensitive** (§5.5 below). This identifies the framework with the empirically established class for human language without requiring Turing-completeness as either a feature or an unintended side-effect.

### 5.5 The composite reaches MCS, by reduction to LCFRS (sketch)

This subsection states the central theoretical claim of the report and sketches the reduction. The full proof is the subject of `docs/MCS_Reduction_For_v3_Composite.md`. Familiarity with mildly context-sensitive grammars (Joshi 1985; Kallmeyer 2010) is assumed; a self-contained primer is in §2 of the companion note.

**Claim.** *Under bounded-fan-out and bounded-rank assumptions on the v3 operator algebra (made precise below), the composite v0+v1.5+v2+v3 simulator generates exactly the **mildly context-sensitive** class of languages, equivalently the class of LCFRS / MCFG languages of bounded fan-out.*

**Setup.** Recall LCFRS / MCFG (Vijay-Shanker, Weir & Joshi 1987; Seki et al. 1991). An LCFRS rule has the form

$$
A(x_1, \ldots, x_k) \to f[B_1(\vec y_1), \ldots, B_r(\vec y_r)]
\qquad(8)
$$

where each non-terminal $A, B_j$ carries a fixed-arity tuple of strings (its *fan-out* $k$ counts the number of independent string fragments associated with the non-terminal), each rule has a fixed *rank* $r$ (number of non-terminals on the right-hand side), and the function $f$ is a fixed concatenation/permutation/copy operation on the fragments. LCFRS of bounded fan-out and bounded rank generates exactly the mildly context-sensitive class.

**The reduction.** Each LCFRS object maps to a v0+v1.5+v2+v3 object as follows.

| LCFRS object                       | Composite-simulator object                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------------ |
| Non-terminal $A$ of fan-out $k$    | A v2-particle of type $A$ whose state $x_\ell^A \in \mathbb{R}^{k \cdot d}$ encodes $k$ fragments of a fixed dimension $d$. |
| Production rule $A \to f[B_1, \ldots, B_r]$ | A v3 operator $\hat f \in G$ of arity $r$ that creates the $A$-particle from $r$ existing $B_j$-particles via $a^\dagger_A \cdot \hat f \cdot a_{B_1} \cdots a_{B_r}$. |
| Concatenation/permutation $f$      | A specific $G$-action on the concatenation of fragment-tuples; $G$ is finite-rank.        |
| Yield function on a derivation tree | Iterated v3 operator application along the order of v2-creation events.                   |
| Termination at terminals $\sigma_i$ | Atomic v2-particle creation events emitting $\sigma_i$.                                   |
| Salient decay (v1.5)               | Tree-pruning of completed sub-derivations whose particles are no longer referenced.       |

**Verification of the four MCS criteria.**

1. **Constant growth.** Every LCFRS rule increases the total fragment length by a bounded amount. In the composite simulator, every v2-creation event emits a bounded number of terminals and every v3-operator action is fixed-arity; the total derivation length grows linearly in the number of particle-creation events, which is the constant-growth condition.

2. **Polynomial parsing.** LCFRS of fixed fan-out $k$ and fixed rank $r$ is parsable in time $O(n^{(r+1)k})$ (Seki et al. 1991). In the composite simulator, this corresponds to: (i) bounded particle dimension $d$ (assumption on v2 states), (ii) bounded operator arity (assumption on v3 actions). Under these, the per-step inference cost of the simulator is polynomial in the active-particle count $N(t)$, and $N(t)$ grows at most linearly in input length by constant growth.

3. **Beyond context-free.** Cross-serial dependencies $a^n b^n c^n$ are produced by an LCFRS of fan-out 2 (one non-terminal carries the pair "future $b$-string, future $c$-string"). In the composite simulator, this corresponds to a v2-particle of state-dim $2d$ (two fragments) generated by a v3 rule of rank 1 that copies the $a$-count into both fragments. A pure CF (i.e. v0+v2 without v3) cannot do this because creation alone is fan-out 1; the operator action of v3 is required to *couple* fan-out-2 fragments. This is the formal counterpart of the F4 falsifier of §4.5.

4. **Below context-sensitive / Turing.** Three structural restrictions on the v3 operator algebra are sufficient to enforce the MCS upper bound:
   - Finite rank $G$. The Lie group $G$ is finite-dimensional with finite rank. This bounds the operator-arity of v3 rules.
   - *Bounded fragment dimension.* Each v2-particle carries a fragment tuple of bounded total dimension $k \cdot d$; this bounds LCFRS fan-out.
   - *No general word-equation composition.* Operator composition $g_1 g_2$ in $G$ does not solve arbitrary word equations; the available operations are concatenation, permutation, and copy of bounded-length fragments — the LCFRS function-set $f$.

   Under these three restrictions, the simulator is bounded above by LCFRS of bounded fan-out and rank, hence by MCS. Without all three, the simulator can encode arbitrary register machines and is Turing-complete — this is the failure mode that F6 of §4.5 exposes.

**The MCS-design point.** The framework therefore lands at MCS *by design choice*, not by structural necessity. The natural design point — finite-rank $G$, bounded particle state dimension, fixed operator-arity vocabulary — yields exactly MCS. Deliberate over-engineering (allowing $G$ to be infinite-dimensional, allowing unbounded operator arity, allowing arbitrary operator-composition word equations) lifts the framework to Turing-complete and therefore over-generates linguistically. The MCS target is the natural, defensible commitment.

This places the v0+v1.5+v2+v3 simulator on the same shelf as TAGs, CCGs, and LCFRSs — alongside the well-understood mildly context-sensitive grammar formalisms — but with the additional advantages that (i) its dynamics is continuous and differentiable end-to-end, (ii) its operator algebra inherits the apparatus of §5.3, and (iii) its parsing complexity is exactly that of LCFRS rather than worse.

---

## 6. Cellular automata as complementary motivation

A natural question — and the one that motivated this report — is whether **cellular automata** (CA), in particular Wolfram's analysis of elementary CA in `A New Kind of Science` (Wolfram 2002), provide the right primary apparatus for the three extensions. The answer is: CA is a useful complementary apparatus but not the right primary one. This section explains why.

### 6.1 What CA establish positively

Wolfram (2002) classifies elementary 1D CA into four classes: I (fixed-point), II (periodic), III (chaotic), IV (complex). Class IV CA — most famously Rule 110 — were proved Turing-complete by Cook (2004). Cook's proof works by exhibiting "particles" (coherent moving patterns) on the Rule 110 grid and showing that their collisions implement the operations of a cyclic tag system, which is universal.

Two consequences are immediately relevant to our setting:

- **Locality is not the bottleneck.** A strictly local update rule on a 1D infinite lattice suffices for universal computation. Restricting interactions to nearest neighbours is *not* what limits expressivity. Our v0 simulator's force forms are also local in semantic space (Gaussian wells, anisotropic SARF terms, bilinear context coupling); locality is therefore *not* the obstacle to lifting v0 above regular.

- **Finite cell count IS the bottleneck.** Restrict any 1D CA — including Rule 110 — to a finite grid of $N$ cells and the system instantly collapses to a finite automaton with at most $|S|^N$ states. *This is the formal mirror of the v0 expressivity bound:* when state space cannot grow, the system is regular.

The **Principle of Computational Equivalence** (Wolfram 2002, Ch. 12) — the conjecture that almost any "complex enough" rule system reaches universal computation — is consistent with this diagnosis. The bottleneck for v0 is not the complexity or richness of the rules; it is the inability of the state space to grow during inference.

This is a strong rhetorical lever:

> Locality is sufficient for universal computation provided the state space is unbounded (Cook 2004 on Rule 110). The v0 simulator has the locality but lacks the unboundedness. The minimal modification that recovers the unboundedness, without abandoning the continuous-state framework, is the introduction of creation/annihilation operators on a Fock space — i.e. v2.

This pre-empts the natural objection "but maybe a richer $V$ alone is enough": Wolfram tells us that's not what's missing. What's missing is *unbounded state*.

### 6.2 Why CA is not the right primary apparatus

Despite the above, three structural mismatches prevent CA from serving as the primary mathematical apparatus for the three extensions:

1. **Discrete vs continuous.** CA are discrete-state, discrete-time. The v0 simulator is continuous-state (so that wells, attractors, gradients, and damped flow are first-class), continuous-time (or finely discretised). Switching to a CA grid would force discarding the entire "force-and-flow" content of the framework — a very large trade.

2. **No native creation/destruction.** CA cells are immutable in count; the grid does not grow or shrink. "Creation" in CA is a quiescent region lighting up. This is *expressible* but awkward, and obscures the Fock-space structure that maps onto v2 cleanly. Doi–Peliti and second quantisation are the natural formalism for variable particle count; CA is not.

3. **No native operator-valued composition.** "Particle $i$ carries operator $\hat O_i$ that acts on particle $j$" is hard to express in CA. It would require encoding the operator as a propagating pattern that interacts with the target pattern, an unstable and combinatorially expensive construction. Lie groups acting on a fibre bundle is the natural formalism; CA is not.

### 6.3 The rhetorical role of CA

The proper place for CA in the v0 → v3 narrative is as a **lower-bound existence proof** and a **rhetorical diagnostic**, not as the primary apparatus:

- *Existence proof:* Cook (2004) on Rule 110 establishes that a system with locality plus unbounded state space can be Turing-complete with strictly local update rules. This is the cleanest possible statement that the issue is unboundedness, not rule complexity.
- *Diagnostic:* Wolfram-class analysis of v0's force forms can in principle classify them by long-term behaviour (fixed-point / periodic / chaotic / complex), giving a cross-check on §3.3's claim that v0 dynamics is non-chaotic.

CA is therefore a *complementary* mathematical apparatus to the primary triple of dissipative semigroups + Fock space + Lie groups. It is referenced in the v0+ programme as motivation, not built into the simulator.

---

## 7. Comparison to transformer expressivity

A natural comparison for the framework is to attention-based transformers — currently the dominant neural architecture for language. The relevant question is: where do transformers sit in the same Chomsky-hierarchy / formal-language landscape that places the v0+v1.5+v2+v3 simulator at MCS, and what does that comparison say about the framework's scientific positioning? This section summarises the transformer-expressivity literature and locates the framework against it.

### 7.1 Vanilla transformers are in TC$^0$, empirically below MCS

A sequence of theoretical results, refined over several papers, places vanilla transformers (no chain-of-thought, no external memory) inside the **TC$^0$** circuit class — constant-depth, polynomial-size threshold circuits. The cleanest tight characterisation is in Merrill & Sabharwal (2023): log-precision transformers, which is the realistic finite-precision regime, capture exactly uniform TC$^0$. Earlier and complementary results: Hahn (2020) showed that soft-attention transformers cannot recognise PARITY or unbounded-depth Dyck-1 with constant probability as input length grows; Hao, Angluin & Frank (2022) placed hard-attention transformers in AC$^0$. The comprehensive synthesis is the survey of Strobl, Merrill, Weiss, Chiang & Angluin (2023).

The implication for our framework is sharp. TC$^0$ is widely believed to be properly inside NC$^1 \subseteq \mathrm{P} \subseteq \mathrm{PSPACE}$, while many CSL-complete problems are PSPACE-hard. Vanilla transformers therefore *cannot recognise CSL-complete languages*, and they cannot reliably recognise the full context-free class either. Empirically, Hewitt et al. (2020) and Yao et al. (2021) showed tiny transformers handle *bounded-depth* $\mathrm{Dyck}_n$ to roughly $D \approx \log_n |Q|$ — exactly the same finite-state collapse depth predicted for v0 by §3 of this report — and length-generalisation past training depth fails consistently. Bhattamishra, Ahuja & Goyal (2020) extended the empirical picture: transformers handle some star-free regular and bounded counter languages but degrade sharply outside those classes. The DeepMind Chomsky-hierarchy benchmark of Delétang et al. (2023) is the most directly comparable empirical study to F1–F6 of §4.5: vanilla transformers fail length-generalisation on every CFL or CSL task in the benchmark.

In Chomsky-hierarchy terms, vanilla transformers are therefore *empirically below MCS* — and below CFL with length generalisation. They fit natural-language data well in-distribution because human language rarely exhibits deep nesting in practice; the formal class they implement is well below what humans actually do.

### 7.2 Chain-of-thought lifts the class — at the cost of serial compute

The picture changes substantially when transformers are permitted to emit intermediate tokens before committing to an answer. Each output token is another forward pass, so chain-of-thought (CoT) is *amortised serial compute on top of parallel attention*. Merrill & Sabharwal (2024) gave the first tight theoretical characterisation: polynomial-length CoT lifts transformers to **P** (polynomial time); linear-length CoT, with the right precision and decoding, captures **NL** (non-deterministic log-space) and approaches CSL = NSPACE$(n)$. Earlier empirical evidence (Feng et al. 2023) is consistent with this picture, and follow-up work (Sanford, Hsu & Telgarsky 2024) refines the precise depth/length trade-offs.

The interesting consequence for our discussion is that *transformer + linear CoT* lives in the same complexity neighbourhood as CSL, the class strictly above MCS. But the cost is structural: the model is no longer doing parallel constant-depth computation in a single pass; it is doing serial computation amortised over many autoregressive steps. The interpretation of "what the model represents at any single step" becomes muddled, the per-step parallelism that defines transformers is given up, and the linguistic interpretation of the architecture's expressivity ceases to be a property of one forward pass and becomes a property of an extended autoregressive trajectory.

### 7.3 Memory- and stack-augmented transformers reach higher classes

The architectural-augmentation literature confirms the diagnosis. To get above CFL with length generalisation, transformers need explicit, structured memory augmentation — and the exact form of the augmentation determines the class reached:

- **Stack-augmented transformers** (DuSell & Chiang 2020, 2023) handle CFL with length generalisation; the differentiable stack is the structural addition that permits unbounded depth.
- **Universal Transformers** with adaptive computation time (Dehghani et al. 2018) plus position encodings are Turing-complete in principle (Pérez, Marinković & Barceló 2019; Bhattamishra, Patel & Goyal 2020), at the cost of unbounded precision and unbounded position encodings.
- **Neural Turing Machines / Differentiable Neural Computers** (Graves et al. 2014, 2016) are in principle Turing-complete and empirically reach CSL-class tasks (sequence reversal, copying, duplication) — but training stability is the documented practical limitation.
- **External-memory and retrieval-augmented transformers** (Memorizing Transformers, Compressive Transformers, retrieval-augmented variants) increase effective context but do not change the formal class on length-generalisation benchmarks.

The conclusion of Delétang et al. (2023) is sharp on this point: *recognising CSL-class languages with length generalisation requires explicit, structured memory augmentation; raw transformer attention does not get there*.

### 7.4 Where the framework sits, by contrast

The transformer literature has been working *backward*: starting from a tractable parallel architecture, characterising its expressivity, and then bolting on additional mechanisms (CoT, stacks, external memory) to compensate for the gap to natural language. The result is a stack of architectural retrofits whose composite formal class depends on which retrofits are included and how.

The v0+v1.5+v2+v3 framework works *forward*: starting from the linguistic empirics — human language is mildly context-sensitive — and asking what natural mathematical home lands the model exactly there. The Fock-space formalism of v2 is the structured memory that transformers acquire only via augmentation; the Lie-group operator algebra of v3 is the rule-application mechanism that transformers approximate only via attention plus serial CoT. The framework's MCS commitment is therefore *native*: it falls out of the bounded operator algebra and bounded fragment dimension by the reduction in §5.5, without architectural retrofits, and without giving up the parallelism or interpretability of the per-layer state.

Concretely, the contrast organises along three axes:

| Architecture                              | Native formal class       | Path to MCS                                    | Path to CSL                                    |
| ----------------------------------------- | ------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| Vanilla transformer                       | TC$^0$ (below CFL with length generalisation) | not native — requires CoT or memory augmentation | linear-CoT approaches NL/CSL, with serial-compute cost |
| Stack-augmented / NTM / DNC               | CFL or higher              | possible empirically, training-unstable        | possible empirically, training-unstable        |
| v0+v1.5+v2+v3 (this framework)            | MCS (native, by §5.5)      | **native**                                     | requires unnatural relaxations (§5.4 of `MCS_Reduction_For_v3_Composite.md`) |

This is the principal scientific positioning of the framework against the dominant neural architecture. The framework is *not* a neural-network architecture with formal-language ceilings to be discovered post hoc; it is a generative formalism whose expressivity is determined by its mathematical structure and is provably exactly MCS — the empirically established class for human language. Whether this advantage translates into empirical wins at scale is the subject of the experimental programme of §4 and §8; what the present section establishes is that the *theoretical positioning* of the framework is in a different regime from the dominant transformer line, with the mathematical home and inherited theorems documented in §5 supplying the structural backbone.

### 7.5 Chain-of-thought is native to v2+v3

§7.2 noted that chain-of-thought (CoT) lifts vanilla transformers from TC$^0$ toward P or NL via amortised serial compute on top of parallel constant-depth attention. A natural follow-up is whether a CoT-like mechanism is *also* available in the v0+v1.5+v2+v3 framework — and if so, whether it is built-in or has to be retrofitted. The answer, which follows directly from the LCFRS reduction of §5.5, is that **chain-of-thought is structurally built into v2+v3 by their natural mathematical form: the LCFRS derivation tree of a single inference is, in effect, the chain of thought**, with non-terminal v2-particles as intermediate steps and v3-operator events as reasoning operations. This subsection makes the claim precise, identifies three structural advantages over transformer CoT, explains why the native CoT does not lift the framework's expressivity above MCS (and why that is the right outcome), and notes the small interface addition required to *expose* the chain of thought to downstream tools.

**The translation.** Transformer CoT operates as a sequence of forward passes, each emitting one token, with each emitted token rejoining the input context for the next pass. The composite simulator at inference time operates as a sequence of v0-evolution-plus-v3-operator events, each potentially emitting a non-terminal particle, with each new particle joining the active particle set for the next operator event to read. The mapping between the two is tight:

| Transformer + CoT element                          | Composite simulator analogue                                  |
| -------------------------------------------------- | ------------------------------------------------------------- |
| Intermediate tokens $t_1, t_2, \ldots$              | Non-terminal v2-particles (LCFRS internal nodes)              |
| Each forward pass                                   | One v3-operator event preceded by a v0 evolution interval     |
| Context window (input + previously emitted tokens) | Active-particle set above the v1.5 salience threshold          |
| Final answer tokens                                 | Terminal v2-creation events emitting to the yield channel     |
| CoT length / total intermediate tokens             | Derivation tree depth (number of non-terminal v2 events)       |
| Token-by-token autoregressive constraint           | Causal ordering of v2/v3 events on the derivation tree        |
| "The model can read its own previous tokens"        | "v3 operators read existing particles in the active set"      |

Every element of transformer CoT has a clean translation in the framework's mechanisms; the framework therefore has, by construction, the structural ingredients of chain-of-thought.

**Three structural advantages of the framework's native CoT.** The translation also reveals three respects in which the framework's CoT is structurally *richer* than transformer CoT, not merely equivalent:

1. **Tree-shaped reasoning, not linear-chain.** Transformer CoT is a strict linear sequence of tokens. The framework's "chain" is an LCFRS derivation tree, in which one v3 operator can take multiple existing particles as inputs (rank $r > 1$). The dependency structure of the reasoning is then explicitly encoded by which particles each operator reads, rather than being implicit in attention weights.

2. **Typed, structured "thoughts", not raw tokens.** Transformer CoT tokens are linear sequences in the same vocabulary as the final answer. The framework's intermediate particles carry typed states (a non-terminal label $A$, a fan-out, and a state vector in a $k_A \cdot d$-dimensional real space), and operator-labelled provenance (which v3 operator $\hat{f}_p$ produced them). This is essentially "CoT with grammar": each thought is a non-terminal in a derivation, with a position in a known LCFRS rule schema, not just a token.

3. **Internal reasoning structurally separated from external yield.** In transformer CoT, intermediate tokens *are part of the output stream* — they are emitted into the response and then conventionally hidden by the application layer. In the framework, non-terminal particles *never enter the yield* by construction (only terminal v2-creation events emit to yield). Internal reasoning is a private process, the external answer is a separate emission, and no convention or post-processing is needed to distinguish them.

**Expressivity is preserved at MCS — and that is the right outcome.** In transformers, CoT lifts the formal class because each individual forward pass is constant-depth (TC$^0$); chaining $T$ forward passes amortises serial compute and gives $O(T)$ sequential depth, lifting to P with polynomial CoT or toward NL with linear CoT (Merrill & Sabharwal 2024). In the framework, by contrast, *each individual v3-operator event is already at MCS* in the §5.5 reduction; the deep serial compute of the LCFRS derivation tree is already factored into the class. Adding more "thinking steps" to the simulator therefore does not lift the class — it produces longer derivations within the same class. The framework has the *practical advantages* of CoT (interpretable serial reasoning, intermediate structured representations, the "scratchpad" pattern) without the *expressivity-class costs* (transformer CoT overshoots from TC$^0$ to P or NL, which exceeds MCS and admits non-linguistic constructions).

This is the empirically correct outcome for a model of human language: *more steps, same class*. Human language reasoning involves arbitrarily many internal cognitive steps within the same MCS-bounded grammatical structure; the framework natively supports this without the constant-vigilance argument transformer CoT requires to avoid silent over-generation.

**A minor interface addition: the thought-trace channel.** The framework already supports CoT-like inference *internally*. To *expose* the chain of thought to downstream tools (interpretability, debugging, auditing, training-signal extraction, comparison against transformer scratchpads), one expressivity-neutral interface addition is useful: at every v3-operator event and every non-terminal v2-creation event, log the event (operator type, inputs read, new particle's state, timestamp) to a designated trace channel. This is purely diagnostic — it does not change the simulator's dynamics, the boundedness assumptions of §5.5, the MCS reduction, or the training procedure. The trace channel produces, for each yield, a structured record of the reasoning that produced it. This is the framework's analogue of a transformer "scratchpad" output, with substantially more structure: a derivation tree rather than a flat token sequence; typed particles with named operators rather than bare tokens.

The implementation cost is minor (event hooks in the v0+v2+v3 inference loop). The scientific payoff is large: each yield produced by the framework comes with a *certified, interpretable, grammar-aligned* thought trace, against which transformer CoT scratchpads can be directly compared on the F1–F6 falsifiers and on the broader linguistic benchmarks.

**The take-away.** The transformer literature has had to *retrofit* CoT to lift expressivity from TC$^0$ toward P — a serial-compute amortisation that gives up parallel constant-depth attention as the inference primitive and that overshoots the empirically correct MCS class. The composite v0+v1.5+v2+v3 simulator natively implements an analogue of CoT *as part of its generative mechanism*, with three structural advantages (tree-shaped, typed, separated-from-yield) and with the expressivity class held at MCS by the same boundedness assumptions that govern the LCFRS reduction. This is, in our view, one of the strongest scientific contrasts between the framework and the dominant attention-based architecture line: the framework gets CoT *for free*, with *more structure*, at *the right expressivity class*, by the natural form of v2 and v3 alone.

---

## 8. Scientific payoff

The framework articulated in §§3–7 promotes the v0 → v3 staging from a list of engineering proposals to a structured scientific programme with the following falsifiable, formally-grounded properties.

1. **A formally provable expressivity ceiling for v0.** Theorem of §3.4: v0 accepts at most regular languages. The bit-counting and damping arguments are quantitative and constructive.

2. **A predicted, measurable falsifier.** $D^\ast$ from equation (5) is a function of $\dim M$, $L$, $\gamma$, $\epsilon$, $n$ alone, computable at calibration time. The $\mathrm{Dyck}_n$ experiment of §4 measures it directly.

3. **A nested falsifier sequence aligning empirically with the v1.5/v2/v3 staging.** F1 → F2 → F3 of §4.5 separately rule out v0, v0+v2, and v0+v2+v1.5 respectively. The order of necessity (v2, then v1.5, then v3) is empirically forced by which falsifier each partial design passes.

4. **A named mathematical apparatus per extension.** Salient decay = dissipative semigroup / discounted MDP. Creation/destruction = Fock space / second quantisation / Doi–Peliti. Execution = Lie group action / non-abelian gauge field. Each apparatus carries inherited theorems on stability, ergodicity, mean-field limits, conservation laws, gauge equivariance.

5. **A specific composite formalism.** v0+v1.5+v2+v3 is a classical-mechanical analogue of an algebraic-QFT-style local operator algebra in the Haag–Kastler tradition. This is mathematically respectable, is novel as a model of language, and inherits a mature toolkit.

6. **A clear rhetorical role for cellular automata.** Locality is sufficient for universal computation provided state space is unbounded (Cook 2004, Wolfram 2002). The pre-empted objection "but maybe richer rules suffice" is closed.

7. **An empirically calibrated upper bound: the composite reaches MCS, not Turing.** Under bounded-fan-out, bounded-rank, and bounded-arity assumptions on the v3 operator algebra, v0+v1.5+v2+v3 generates exactly the mildly context-sensitive class — the empirically established class for human language (Joshi 1985; Shieber 1985). The explicit reduction to LCFRS in §5.5, with the four MCS criteria verified, places the framework on the same shelf as TAGs, CCGs, and LCFRSs. The MCS reach is then *empirically tested* by F4 (cross-serial $a^n b^n c^n$) and F5 (bounded copy $ww$); the upper bound is *empirically protected* by F6 (a Minsky 2-counter task that the framework should *fail*). The full reduction proof is in `docs/MCS_Reduction_For_v3_Composite.md`.

8. **Native chain-of-thought via the LCFRS derivation tree.** The composite v2+v3 mechanisms supply, by their natural mathematical form, a structured analogue of chain-of-thought (§7.5): non-terminal v2-particles act as intermediate "thoughts", v3-operator events act as reasoning operations, and the active-particle set acts as the context-window of currently-readable thoughts. This native CoT is structurally richer than transformer CoT (tree-shaped rather than linear-chain, typed particles rather than bare tokens, internal reasoning structurally separated from external yield) and — critically — it preserves the MCS expressivity class regardless of derivation depth, avoiding the TC$^0 \to$ P or NL overshoot of transformer CoT. The framework gets CoT *for free*, with *more structure*, at *the right expressivity class*. A minor expressivity-neutral interface addition (the thought-trace channel) exposes this native CoT to downstream interpretability and interpretation-comparison tools.

The qualitative version of the v0 → v3 programme in the programme memo is replaced by a tier-structured formal argument with one experiment per tier (Dyck for v0; F2/F3 for v2/v1.5; F4/F5 for the v3 MCS reach; F6 for the upper-bound), three mappings to mature mathematical apparatuses, one composite formalism, one named target generative class (MCS), and a native CoT analogue inherited from the LCFRS derivation structure. This is the strongest form of the case for the v1.5 / v2 / v3 staging that we are aware of.

---

## 9. Recommended roadmap

The deliverables of the present note frame three concrete next actions, in increasing order of effort.

| # | Deliverable | Effort | Decision gate |
|---|-------------|--------|---------------|
| 1 | `docs/Expressivity_Bounds_For_v0_Simulator.md`: 5-page formal note tightening §3 with the constants worked out and $D^\ast$ derived in full. | ~1 day | Are the constants in (4) and (5) consistent with the standard metric-entropy / damping calculations? |
| 2 | $\mathrm{Dyck}_n$ experiment in `notebooks/semsim_simulator/`: corpus generator, integrator + readout, comparison vs tiny transformer / LSTM at matched parameters and matched compute. | ~1 week | Does the observed $D^\ast$ match (5) within a factor of 2? Do tiny transformer / LSTM exceed v0's $D^\ast$ by at least a factor of 4? |
| 3 | **Thought-trace channel** in `notebooks/semsim_simulator/`: event hooks on v3-operator and non-terminal v2-creation events that emit a structured trace per yield (operator type, inputs read, particle states, timestamps). Expressivity-neutral; logging-only. | ~1–2 days | Does the trace round-trip into a well-formed LCFRS derivation tree? Can it be read back to reproduce the yield? |
| 4 | `docs/Semantic_Simulator_v2_EOM.md`: v2 EOM specification as a Fock-space + creation/annihilation extension of `Semantic_Simulator_EOM.md`, analogous in role to the M0 deliverable for v0. | ~1–2 weeks | Is the v2 EOM well-typed (number-conserving where expected, gauge-equivariant where required)? Does the Doi–Peliti specialisation reduce to v0 in the zero-creation limit? |

Action 1 produces the formal companion this report depends on. Action 2 produces the empirical confirmation of (or correction to) the formal bound. Action 3 is a low-effort, high-leverage instrumentation step that exposes the framework's native chain-of-thought (§7.5) for interpretability and side-by-side comparison against transformer CoT scratchpads on the F1–F6 falsifier suite. Action 4 is the M0-class deliverable for the v2 stage, deferred until 2 has confirmed (or empirically corrected) the bound.

The corresponding extensions of the programme memo's milestone schedule are:

- **M1.5:** complete actions 1–3 of the table above.
- **M1.6:** if action 2's results are consistent with (5), commit to action 4; if not, revise §3 and §4.3 of this note before proceeding.
- **M2 onward:** continues as specified in the programme memo.

---

## 10. Conclusions

The v0 simulator class — continuous-state, finite-dimensional, smooth, damped — is at most a finite automaton. This is provable by a four-step argument from phase-space capacity, damping-induced volume contraction, the non-chaotic functional class of $V$, and the Chomsky-hierarchy placement that follows. The corresponding empirical falsifier — $\mathrm{Dyck}_n$ at controlled depth — is a measurement, not a metaphor; the predicted collapse depth is a closed-form function of architecture parameters.

Each of the three structural extensions named in the programme memo (v1.5, v2, v3) corresponds to a mature mathematical formalism: dissipative semigroups, Fock space and second quantisation, Lie groups and non-abelian gauge theory. The composite v0+v1.5+v2+v3 specifies a particular point in the algebraic-QFT landscape — a classical-mechanical local operator algebra on a Fock space with a salience-weighted, gauge-equivariant Hamiltonian.

Generatively, under bounded-fan-out, bounded-rank, and bounded-arity assumptions on the v3 operator algebra, this composite system reaches **exactly the mildly context-sensitive class** — the empirically established class for human language. The reduction to LCFRS sketched in §5.5 (and proved in detail in `docs/MCS_Reduction_For_v3_Composite.md`) places the framework alongside TAGs, CCGs, and LCFRS-style grammars, with the additional advantages of continuous, differentiable dynamics and the operator-algebraic apparatus of §5.3. This is, to our knowledge, the first explicit reduction of a force-and-flow continuous-state simulator to a mildly context-sensitive grammar formalism.

Compared to the dominant attention-based architecture line, the framework occupies a structurally different regime (§7). Vanilla transformers sit in TC$^0$ — well below MCS — and reach P or NL only via amortised serial chain-of-thought, at the cost of overshooting the empirically correct class. The composite simulator, by contrast, lands at MCS *natively* by §5.5, and supplies a *native* analogue of chain-of-thought through the LCFRS derivation tree of v2+v3 (§7.5): non-terminal particles act as intermediate "thoughts", v3-operator events as reasoning steps, the active-particle set as the context-window of currently-readable thoughts. The framework's CoT is tree-shaped rather than linear-chain, typed rather than bare-token, structurally separated from the external yield rather than inlined into the output stream — and, critically, the expressivity class is held at MCS regardless of derivation depth, so the framework gains the practical advantages of CoT without the expressivity-class cost.

Cellular automata, examined as a candidate primary apparatus, are the wrong fit: their discreteness, fixed cell count, and absence of operator-valued composition make them a poor home for the framework. They are, however, the right complementary tool: Cook's universality proof for Rule 110 establishes that locality is sufficient for universal computation given unbounded state space, which is the cleanest possible argument that v0's bottleneck is unboundedness rather than rule complexity.

The qualitative case for v1.5 / v2 / v3 in the programme memo is replaced by a structured formal programme. The v0 → v3 staging is no longer a list of plausible mechanisms; it is the natural three-step extension of v0 along three orthogonal axes in a known mathematical landscape, with a decisive experiment per tier (F1–F3 for the lower-bound staircase, F4–F5 for the MCS-reach, F6 for the upper-bound), a named theorem-rich apparatus per axis, a single named generative target — mildly context-sensitive — that holds the framework empirically accountable to the linguistic record, and a native analogue of chain-of-thought delivered for free by the LCFRS derivation structure. The framework gets, by its natural mathematical form, what transformers retrofit at expressivity-class cost.

---

## References

- Bhattamishra, S., Ahuja, K., and Goyal, N. (2020). On the ability and limitations of transformers to recognize formal languages. *EMNLP*.
- Bhattamishra, S., Patel, A., and Goyal, N. (2020). On the computational power of transformers and its implications in sequence modeling. *CoNLL*.
- Cook, M. (2004). Universality in elementary cellular automata. *Complex Systems*, 15(1).
- Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., and Kaiser, Ł. (2018). Universal Transformers. *ICLR 2019*.
- Delétang, G., Ruoss, A., Grau-Moya, J., Genewein, T., Wenliang, L. K., Catt, E., Cundy, C., Hutter, M., Legg, S., Veness, J., and Ortega, P. A. (2023). Neural networks and the Chomsky hierarchy. *ICLR*.
- Doi, M. (1976). Second quantization representation for classical many-particle system. *Journal of Physics A: Mathematical and General*, 9(9).
- DuSell, B. and Chiang, D. (2020). Learning context-free languages with nondeterministic stack RNNs. *CoNLL*.
- DuSell, B. and Chiang, D. (2023). The surprising computational power of nondeterministic stack RNNs. *ICLR*.
- Feng, G., Zhang, Y., Zhang, R., Sun, M., and Wang, L. (2023). Towards revealing the mystery behind chain of thought: a theoretical perspective. *NeurIPS*.
- Graves, A., Wayne, G., and Danihelka, I. (2014). Neural Turing Machines. *arXiv:1410.5401*.
- Graves, A., Wayne, G., Reynolds, M., et al. (2016). Hybrid computing using a neural network with dynamic external memory. *Nature*, 538(7626).
- Haag, R. and Kastler, D. (1964). An algebraic approach to quantum field theory. *Journal of Mathematical Physics*, 5(7).
- Hahn, M. (2020). Theoretical limitations of self-attention in neural sequence models. *TACL*, 8.
- Hao, Y., Angluin, D., and Frank, R. (2022). Formal language recognition by hard attention transformers: perspectives from circuit complexity. *TACL*, 10.
- Hewitt, J., Hahn, M., Ganguli, S., Liang, P., and Manning, C. D. (2020). RNNs can generate bounded hierarchical languages with optimal memory. *EMNLP*.
- Joshi, A. K. (1985). Tree-adjoining grammars: how much context-sensitivity is required to provide reasonable structural descriptions? In *Natural Language Parsing*, Cambridge University Press.
- Kallmeyer, L. (2010). *Parsing Beyond Context-Free Grammars.* Springer.
- Merrill, W. and Sabharwal, A. (2023). The parallelism tradeoff: limitations of log-precision transformers. *TACL*, 11.
- Merrill, W. and Sabharwal, A. (2024). The expressive power of transformers with chain of thought. *ICLR*.
- Peliti, L. (1985). Path integral approach to birth–death processes on a lattice. *Journal de Physique*, 46(9).
- Pérez, J., Marinković, J., and Barceló, P. (2019). On the Turing completeness of modern neural network architectures. *ICLR*.
- Sanford, C., Hsu, D., and Telgarsky, M. (2024). Representational strengths and limitations of transformers. *ICML*.
- Seki, H., Matsumura, T., Fujii, M. and Kasami, T. (1991). On multiple context-free grammars. *Theoretical Computer Science*, 88(2).
- Shieber, S. M. (1985). Evidence against the context-freeness of natural language. *Linguistics and Philosophy*, 8(3).
- Siegelmann, H. T. and Sontag, E. D. (1991, 1995). Turing computability with neural nets. *Applied Mathematics Letters / Journal of Computer and System Sciences*.
- Strobl, L., Merrill, W., Weiss, G., Chiang, D., and Angluin, D. (2023). Transformers as recognizers of formal languages: a survey on expressivity. *TACL*, 12.
- Vijay-Shanker, K., Weir, D. J., and Joshi, A. K. (1987). Characterizing structural descriptions produced by various grammatical formalisms. In *Proceedings of ACL*.
- Wolfram, S. (2002). *A New Kind of Science.* Wolfram Media.
- Yao, S., Peng, B., Papadimitriou, C., and Steinhardt, J. (2021). Self-attention networks can process bounded hierarchical languages. *ACL*.
- Gueorguiev, D. P. (2025). *Lie Groups for Gauge Theory — A Graduate Tutorial from Matrix Groups to the Structure of U(1), SU(2), SU(3).*
- Gueorguiev, D. P. (2026a). *A Full Tutorial on Gauge Theory — From First Principles to Non-Abelian Fields and Transformer Connections.*
- Gueorguiev, D. P. (2026b). *Semantic Simulator EOM* (`docs/Semantic_Simulator_EOM.md`).
- Gueorguiev, D. P. (2026c). *Semantic Simulator with RL-calibrated Force Fields: a programme memo* (`docs/Semantic_Simulator_RL_Calibration_Programme.md`).
- Gueorguiev, D. P. (2026d). *MCS Reduction for the v0+v1.5+v2+v3 composite simulator* (`docs/MCS_Reduction_For_v3_Composite.md`).
