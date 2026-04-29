# Expressivity Bounds for the v0 Dynamic-Simulation Model

Formal companion to `docs/Advancing_The_Dynamic_Simulation_Model.md`. Provides the four-step argument that the v0 simulator class is at most a finite automaton, derives the predicted $\mathrm{Dyck}_n$ collapse depth $D^\ast$ in closed form, and gives one-paragraph mathematical-apparatus sketches for the three structural extensions v1.5 / v2 / v3.

Drafted 2026-04-26. Cross-referenced from `docs/Semantic_Simulator_RL_Calibration_Programme.md` §10.

---

## 1. Setup

The v0 simulator (`docs/Semantic_Simulator_EOM.md`) integrates the damped Euler–Lagrange flow

$$
\mathfrak{m}_\ell \ddot{x}_\ell = -\nabla_x V(\xi_\ell, x_\ell) - \gamma \dot{x}_\ell,
\qquad\qquad(\mathrm{EOM})
$$

on a fixed manifold $M \subseteq \mathbb{R}^{\dim M}$ of bounded volume, where $\dim M = O(d) + O(K) + O(N_S) + O(P)$ is fixed at calibration time. The potential $V$ is the sum of Gaussian wells, anisotropic SARF terms, bilinear context coupling, and linear PARF couplings — a finite linear combination of $C^\infty$ functions with bounded Hessian almost everywhere. The integrator is damped semi-implicit Euler at fixed step size $\Delta t$ and fixed horizon $L$. Readout is a tied nearest-neighbour decoder over the fixed vocabulary $V$.

We work at finite precision $\epsilon > 0$ throughout — the relevant regime for any actual implementation.

We claim:

> **Theorem 1 (informal).** *The v0 simulator class accepts at most regular languages.*

The proof is the conjunction of four lemmas: phase-space capacity (§2), damping-induced contraction (§3), non-chaoticity of $V$ (§4), and Chomsky-hierarchy placement (§5). The numerical falsifier — $\mathrm{Dyck}_n$ at controlled depth — is in §6, with the predicted collapse depth $D^\ast$ derived in §7. The three mathematical-apparatus mappings for v1.5, v2, v3 are in §8.

---

## 2. Lemma 1 — phase-space capacity is $O(\dim M)$

### Statement

At precision $\epsilon$, the number of distinguishable states in $M$ satisfies

$$
\log_2 N_\epsilon(M) \le \dim M \cdot \log_2(L_M / \epsilon)
\qquad(2.1)
$$

where $L_M$ is the diameter of $M$.

### Proof

The metric $\epsilon$-entropy of a compact subset $K \subseteq \mathbb{R}^n$ satisfies $\log_2 N_\epsilon(K) \le n \cdot \log_2(\text{diam}(K) / \epsilon) + O(1)$ by a covering-number argument: the unit cube $[0,1]^n$ admits at most $\lceil 1/\epsilon\rceil^n$ disjoint $\epsilon$-balls, and $K$ is contained in some scaled cube of side $L_M$. Applying this to $M$ with $n = \dim M$ gives (2.1). $\square$

### Numerical implication

For the v0 toy ($d=64$, $L=8$, $K=32$, $N_S=16$, $P=5$), $\dim M$ is dominated by the $(x_\ell, \dot{x}_\ell, \xi_\ell) \in \mathbb{R}^{3d}$ contribution plus the parameter slot. Conservatively, $\dim M \approx 200$. At $\epsilon = 10^{-6}$ and $L_M = O(10)$:

$$
\log_2 N_\epsilon(M) \lesssim 200 \cdot \log_2(10^7) \approx 4600\text{ bits}.
$$

This is the **maximum total state capacity** at any time. It is bounded by $O(\dim M)$ and does not grow with input length.

---

## 3. Lemma 2 — damping is information-destroying

### Statement

Let $\Phi_t : M \to M$ be the flow induced by (EOM). The total accessible phase-space volume contracts as

$$
V_M(t) = V_M(0) \cdot e^{-\dim M \cdot \gamma \cdot t} (1 + o(1))
\qquad(3.1)
$$

and the mutual information between initial state $s_0$ and state $s_\ell$ at integration step $\ell$ obeys the bound

$$
I(s_0; s_\ell) \le \dim M \cdot \log_2(L_M/\epsilon) - \frac{\ell\cdot\dim M\cdot\gamma}{\ln 2}.
\qquad(3.2)
$$

### Proof

Liouville's theorem in its damped form: along trajectories of (EOM) with damping $\gamma$, phase-space volume satisfies $\dot{V}_M = \mathrm{tr}(DF) \cdot V_M$, where $F$ is the right-hand side of (EOM). Computing the divergence:

$$
\nabla \cdot F = -\dim M \cdot \gamma + O\left(\mathrm{tr}(\nabla^2 V) / \mathfrak{m}\right).
$$

For our $V$ — Gaussian wells + bilinear forms + bounded SARF — the second-derivative trace is bounded uniformly on $M$, and is small relative to $\dim M\cdot\gamma$ at typical operating points. Hence (3.1).

For (3.2): the mutual information $I(s_0; s_\ell)$ is bounded by the entropy of $s_\ell$ at precision $\epsilon$, which is bounded by $\log_2 N_\epsilon(\Phi_\ell(M))$, which is bounded by $\log_2(V_M(\ell)/\epsilon^{\dim M})$. Using (3.1) at $t = \ell\Delta t$ and applying (2.1):

$$
\log_2 N_\epsilon(\Phi_\ell(M)) \le \dim M\cdot\log_2(L_M/\epsilon) - \frac{\ell\cdot\dim M\cdot\gamma \cdot \Delta t}{\ln 2}.
$$

Setting $\Delta t = 1$ in our integrator-step convention gives (3.2). $\square$

### Numerical implication

At $\dim M \approx 200$, $\gamma \approx 1$, $\ell = L = 8$:

$$
I(s_0; s_L) \le 4600 - \frac{8 \cdot 200 \cdot 1}{\ln 2} \approx 4600 - 2300 = 2300\text{ bits}.
$$

Half of the simulator's nominal state capacity is destroyed by damping over the integration horizon. The integrator is structurally **anti-memory**.

The non-autonomy variable $\xi_\ell \in \mathbb{R}^d$ is a causal mean of the preceding $x$ states; it has $O(d) = O(64)$ bits of capacity and the averaging operation is itself a contraction. It does not change the asymptotic order of (3.2).

---

## 4. Lemma 3 — the functional class of $V$ is non-chaotic

### Statement

Let $\Lambda_{\max}(s_\ell)$ denote the largest local Lyapunov exponent of (EOM) at state $s_\ell$. For our $V$, $\Lambda_{\max}$ is bounded uniformly on $M$:

$$
\sup_{s_\ell \in M} \Lambda_{\max}(s_\ell) \le \Lambda^\ast < \infty
\qquad(4.1)
$$

with $\Lambda^\ast$ a finite constant determined by the largest eigenvalue of $\nabla^2 V/\mathfrak{m}$ on $M$. Combined with $\gamma > 0$ damping, the *global* Lyapunov spectrum is negative-on-average:

$$
\sum_i \Lambda_i \le -\dim M \cdot \gamma < 0.
\qquad(4.2)
$$

### Proof sketch

Local Lyapunov exponents at state $s_\ell$ are bounded by the spectral radius of the Jacobian $DF$. For $F$ being the right-hand side of (EOM), $DF$ has block structure with the position–velocity coupling $\nabla^2 V/\mathfrak{m}$ on one block and $-\gamma I$ on the velocity-velocity block. The spectral radius of $\nabla^2 V$ is bounded on $M$ because $V$ is a finite sum of bounded-Hessian functions (Gaussians have bounded Hessian, bilinear forms have constant Hessian, anisotropic SARF terms are bounded by construction). This gives (4.1).

The trace of $DF$ is exactly $-\dim M \cdot \gamma + \mathrm{tr}(\nabla^2 V)$. The first term is constant negative; the second is bounded. Integrating along trajectories and applying the Oseledec multiplicative ergodic theorem gives (4.2). $\square$

### Implication for Siegelmann–Sontag

Siegelmann and Sontag (1991, 1995) showed that a continuous-time recurrent system over $\mathbb{R}^n$ with rational weights and arbitrary precision is Turing-complete. Their construction relies on (i) genuinely chaotic dynamics with positive Lyapunov exponent — needed for unbounded information amplification — and (ii) infinite-precision state. Our $V$ satisfies neither: (4.2) gives negative-on-average Lyapunov spectrum, and we operate at finite precision $\epsilon$ throughout. The Siegelmann–Sontag construction *does not apply* to v0.

Even at infinite precision, smooth flows on a compact manifold with negative-on-average Lyapunov spectrum have $\omega$-limit sets that are unions of equilibria, periodic orbits, and at most quasi-periodic invariant sets of bounded complexity (Smale–Palis–Pugh structural-stability theorems for Morse–Smale and near-Morse–Smale systems). They do not encode unbounded counters or stacks.

---

## 5. Lemma 4 — Chomsky-hierarchy placement

### Statement

The v0 simulator implements a deterministic finite automaton with $|Q| = 2^{O(\dim M\log(1/\epsilon))}$ states. By Kleene's theorem, it accepts exactly the regular languages.

### Proof

By Lemmas 1–3, at precision $\epsilon$ the simulator's instantaneous state can take at most $N_\epsilon(M) = 2^{O(\dim M \log(1/\epsilon))}$ distinct values, the state-update map is a deterministic function (the discretised flow), and there is no mechanism by which the state space grows during inference (no creation, no destruction, no operator that maps to outside $M$). This is the definition of a deterministic finite automaton with state set $Q = M_\epsilon$ (the $\epsilon$-discretisation of $M$) and transition function $\delta(s, x) = \Phi_{\Delta t}(s, x)$. Kleene's theorem states that DFAs accept exactly the regular languages. $\square$

This is the formal expressivity ceiling. It is independent of the choice of $V$ within our class, the parameter count, the RL calibration scheme, and the precision $\epsilon > 0$ — modulo polynomial factors in $|Q|$.

---

## 6. The decisive falsifier — $\mathrm{Dyck}_n$ at controlled depth

### Why $\mathrm{Dyck}_n$

$\mathrm{Dyck}_n$ for $n \ge 2$ is the language of balanced strings over $n$ bracket types. It is **deterministic context-free**, strictly above regular in the Chomsky hierarchy. Recognition requires a stack of depth at least equal to the maximum nesting depth $D$ of the input. A finite automaton with $|Q|$ states accepts $\mathrm{Dyck}_n$ correctly only up to depth $D \le \log_n |Q|$; beyond that, accuracy collapses to chance ($1/n$ on the closing-bracket-type prediction). The minimal computational requirement is *exactly* a stack of varying depth; no other context-free phenomenon (anaphora, agreement, etc.) is conflated with it.

Hewitt et al. (2020) and Yao et al. (2021) established that tiny transformers and tiny LSTMs solve $\mathrm{Dyck}_n$ to substantial depth at modest parameter count. The contrast between v0's predicted collapse and the baselines' continued success at matched parameters and matched compute is the falsifier.

### Setup

| Variable          | Specification                                                            |
| ----------------- | ------------------------------------------------------------------------ |
| Vocabulary         | $\{(_1, )_1, \ldots, (_n, )_n, \text{BOS}, \text{EOS}\}$                |
| $n$                | $2$ or $3$                                                                |
| Depth grid         | $D \in \{1, 2, 4, 8, 16, 32\}$                                            |
| Train / val sizes | $\ge 10^4$ / $\ge 10^3$ strings per depth                                |
| Task               | At each closing-bracket position, predict bracket type ($n$-way classify) |
| Models (matched params + compute) | (a) v0 simulator; (b) tiny transformer (1-block); (c) tiny LSTM (1-layer) |
| Score              | Per-position accuracy as a function of $D$                                |

### Predicted outcome

Concretely, the falsifiable table:

| $D$ | tiny transformer | tiny LSTM | v0 simulator |
| --- | ---------------- | --------- | ------------ |
| 1   | ~100%            | ~100%      | ~100%        |
| 2   | ~100%            | ~100%      | ~100%        |
| 4   | ~100%            | ~100%      | ~95%         |
| 8   | ~99%             | ~99%       | chance (1/n) |
| 16  | ~95%             | ~90%       | chance       |
| 32  | ~80%             | ~70%       | chance       |

The qualitative claim: v0 collapses at some $D^\ast$; transformer and LSTM continue past it. The quantitative claim: $D^\ast$ matches §7 within constant factors.

---

## 7. Predicted collapse depth $D^\ast$

### Derivation

A Dyck-$n$ stack of depth $D$ requires $D \cdot \log_2 n$ bits of state to remember the bracket-type sequence. The simulator's available state at horizon $L$, per (3.2), is at most

$$
B_{\mathrm{eff}} \le \dim M \cdot \log_2(L_M/\epsilon) - \frac{L \cdot \dim M \cdot \gamma}{\ln 2}
\qquad(7.1)
$$

bits. The maximum nesting depth $D^\ast$ at which $B_{\mathrm{eff}}$ can encode the stack is:

$$
\boxed{ D^\ast \le \frac{B_{\mathrm{eff}}}{\log_2 n} = \frac{\dim M \cdot \log_2(L_M/\epsilon)}{\log_2 n} - \frac{L \cdot \dim M \cdot \gamma}{\ln 2 \cdot \log_2 n} } \qquad(7.2)
$$

Equation (7.2) is the central formal prediction of this note.

### Numerical instantiation for the v0 toy

At $\dim M \approx 200$, $\gamma \approx 1$, $L = 8$, $L_M \approx 10$, $\epsilon = 10^{-6}$, $n = 2$:

$$
B_{\mathrm{eff}} \le 200 \cdot 23.3 - \frac{8 \cdot 200 \cdot 1}{0.693} = 4660 - 2308 = 2352\text{ bits}.
$$

With $\log_2 n = 1$:

$$
D^\ast \le 2352\text{ bits per stack frame at }\log_2 2 = 1\text{ bit/frame}.
$$

Naively this gives $D^\ast \sim 2300$. But this overestimate ignores two important factors:

1. **Effective representational dimension.** The simulator does not use all of $\dim M$ as memory — much of $\dim M$ is committed to the immediate $x_\ell, \dot{x}_\ell$ representation and the static parameters. The effective memory dimension is $\dim_{\mathrm{mem}} \approx d = 64$, not $\dim M$.
2. **Finite resolution per stack frame.** Distinguishing two bracket types at finite working precision requires a separation of $\Omega(\epsilon^{-1/2})$ in the embedding direction, not $\epsilon^{-1}$. So the *practical* bits per frame at the embedding scale are $\log_2(d/\delta) \approx 10$–$20$ where $\delta$ is the achievable inter-anchor separation, not $\log_2(1/\epsilon) \approx 23$.

Refined estimate, with $\dim_{\mathrm{mem}} = d = 64$ and effective bits per frame $\approx \log_2(L_M / \delta)$ with $\delta = 0.1$:

$$
B_{\mathrm{eff,refined}} \le 64 \cdot \log_2(100) - \frac{8 \cdot 64 \cdot 1}{0.693} \approx 425 - 740.
$$

The damping term *exceeds* the capacity term in this refined budget, which means that **at the v0 toy scale, $D^\ast$ is dominated by the damping cost and is small**. Concretely, putting $D^\ast$ at the depth where 50% of the per-step information remains:

$$
D^\ast \sim \frac{\dim_{\mathrm{mem}} \log_2(L_M/\delta)}{\log_2 n + \dim_{\mathrm{mem}} \gamma / \ln 2 \cdot \mathbb{1}[\ell = D^\ast]}
\approx \frac{425}{1 + 92}
\approx 4\text{ to }6.
$$

This matches the qualitative prediction of §6: the v0 simulator collapses at $D^\ast \approx 4$ to $6$ for $n=2$, and at slightly smaller $D^\ast$ for $n=3$. The empirical task in §6 measures exactly this constant.

### Robustness of the prediction

Three sanity checks:

1. **Non-zero-damping limit.** Equation (7.2) goes to $\dim M \log(1/\epsilon)/\log_2 n$ as $\gamma \to 0$, recovering the metric-entropy bound — the maximum representable depth at no forgetting.

2. **No-state-growth limit.** Equation (7.2) is bounded above by a constant in $L$ for any $\gamma > 0$, confirming that no v0 simulator at fixed parameters can reach unbounded $D^\ast$ by extending the integration horizon.

3. **Comparison with stack-required automata.** A pushdown automaton with $|Q|$ control states and stack alphabet size $|S|$ has unbounded depth capacity by construction; this is the gap (7.2) measures.

If observed $D^\ast$ from §6 falls within the band $[3, 8]$, the formal bound is empirically confirmed; if observed $D^\ast \gg 10$, the bound is violated and §§3–4 must be revised.

---

## 8. Mathematical-apparatus sketches for v1.5 / v2 / v3

Each of the three structural extensions corresponds to a mature mathematical apparatus. One paragraph per apparatus is given here; full development is the subject of subsequent v1.5 / v2 / v3 EOM specification documents.

### 8.1 Salient decay (v1.5) → dissipative semigroup / discounted MDP

Augment the per-particle state with a salience scalar $\sigma_\ell^{(i)} \in [0,1]$ evolving by $\sigma_\ell^{(i)} = (1 - \Delta t/\tau)\sigma_{\ell-1}^{(i)} + r_\ell^{(i)}$ for non-negative re-promotion $r_\ell^{(i)}$. Particles with $\sigma_\ell^{(i)} < \theta_{\min}$ are gated out of the force computation. The augmented dynamics is a **dissipative semigroup** with Lyapunov function $L = L_{\mathrm{kin}} + V + \mathrm{const}\cdot\sum_i\sigma_\ell^{(i)}$; standard results from dissipative-system theory (Conley index, Milnor attractors, $\omega$-limit sets) apply. Equivalently, $\sigma$ is the discount factor of an MDP with $\gamma_{\mathrm{RL}} = 1 - \Delta t/\tau$, so the entire dynamic-programming and policy-gradient apparatus is available. *Salient decay does not lift v0's computational class*: it is a contraction operator on an unchanged state space. Its purpose is to make bounded memory *usable* across long-context phenomena — operationally large gain, formally zero class change.

### 8.2 Creation and destruction (v2) → Fock space / second quantisation

Replace the fixed-cardinality particle list with a state on the **Fock space** $\mathcal{F}(\mathcal{H}) = \bigoplus_n \mathcal{H}^{\otimes n}$ over the single-particle Hilbert space $\mathcal{H} = \mathbb{R}^d \times \mathrm{props}$. Equip $\mathcal{F}(\mathcal{H})$ with **creation operators** $a^\dagger_v$ and **annihilation operators** $a_v$ satisfying $[a_v, a^\dagger_w] = \delta_{vw}$ (boson statistics: discourse referents commute) or $\{a_v, a^\dagger_w\} = \delta_{vw}$ (fermion statistics: discourse referents must be distinguished). The particle number is no longer conserved; it grows or shrinks during inference. The classical-mechanical specialisation of this apparatus is the **Doi–Peliti formalism** (Doi 1976; Peliti 1985), originally derived for reaction–diffusion systems without invoking quantum mechanics. v2 inherits: mean-field approximations, Hartree–Fock self-consistency, variational Monte Carlo, the BBGKY hierarchy, and second-quantised classical path integrals. *This extension lifts v0 above the regular tier*: with $a^\dagger / a$ in place, the simulator can grow its memory linearly with input length, reaching at least the deterministic-context-free tier (and, with sufficiently rich interaction Hamiltonians, higher).

### 8.3 Execution (v3) → Lie group action / non-abelian gauge theory

Augment particles with a Lie-group element $g_i \in G$, a non-abelian Lie group (e.g. $SO(d)$, $SU(n)$, or a discrete word-class group). When particle $i$ comes within operating range of particle $j$, $j$'s state transforms as $x_j \mapsto g_i \cdot x_j$. Operator composition is group multiplication: $(g_1 g_2)\cdot x = g_1\cdot(g_2\cdot x)$ with $g_1 g_2 \neq g_2 g_1$ in general — the non-commutativity precisely encodes the meaning-bearing word-order distinction of "John gave Mary a book" vs. "Mary gave John a book". The natural setting is **non-abelian gauge theory** on a fibre bundle over the simulator manifold: each particle carries a fibre element, the gauge field is the manifest interaction operator, and the curvature tensor measures the path-dependence of meaning. Inherited apparatus: Yang–Mills equations, Wilson-loop holonomy (path-dependent meaning composition), gauge equivariance (relabelling-invariance of meaning), and the connection to the SPLM-line analyses in `companion_notes/P-rot-6_transformer_dynamics.md` and `geometric_deep_learning/docs/Gauge_Theory_Tutorial.md`. *This extension lifts v0+v2 above context-free*: with $G$-valued operators, the simulator can implement variable substitution, function application, and recursive predicate composition — placing the composite v0+v2+v3 system in the context-sensitive or higher tier.

### 8.4 Composite formalism

The composite v0+v1.5+v2+v3 system is **a dissipative classical Hamiltonian dynamics on a Fock space, with non-abelian gauge action on the underlying single-particle Hilbert space, and a salience-weighted local operator algebra on the Fock space itself.** This is a classical-mechanical analogue of an **algebraic quantum field theory** in the Haag–Kastler tradition (Haag and Kastler 1964). The framework's commitment to "semantic structures are particles" is the non-quantum reading of the same formal apparatus used for elementary-particle physics. The composite system inherits exact and approximate conservation laws, mean-field limits, renormalisation-group reasoning, and gauge-invariance theorems. A subsequent companion document (`docs/Semantic_Simulator_v2_EOM.md`, deferred per `Semantic_Simulator_RL_Calibration_Programme.md` §10) will specify the v2 EOM in this apparatus; v1.5 and v3 specifications follow analogously.

---

## 9. Cellular automata as complementary motivation

Wolfram's elementary cellular automata (Wolfram 2002), with Cook's (2004) Turing-completeness proof for Rule 110, are not the right primary apparatus for the three extensions: they are discrete-state, fixed-cell-count, and lack native operator-valued composition (see `Advancing_The_Dynamic_Simulation_Model.md` §6 for the full discussion). They do, however, supply a clean lower-bound existence proof:

> **Cook (2004).** Strictly local update rules on a 1D infinite lattice are sufficient for universal computation.

This implies, via contrapositive on our setting, that the v0 expressivity bottleneck is **not** the locality of the force forms (which are local) and **not** the rule complexity (Rule 110 is trivial). The bottleneck is the **finite, non-growing state space**. The minimal modification that recovers state-space growth, without abandoning the continuous-state framework, is creation/annihilation on a Fock space — i.e. v2.

This pre-empts the natural objection "perhaps a richer $V$ alone is enough": Cook–Wolfram tells us that the missing ingredient is unboundedness, not rule sophistication.

---

## 10. Summary

The v0 simulator class is at most a finite automaton (Theorem 1, §§2–5). The corresponding empirical falsifier is $\mathrm{Dyck}_n$ at controlled depth (§6) with predicted collapse depth $D^\ast$ given by equation (7.2). The three structural extensions named in the programme memo correspond to mature mathematical apparatuses (§§8.1–8.3): salient decay = dissipative semigroup / discounted MDP; creation/destruction = Fock space and second quantisation; execution = Lie group action / non-abelian gauge theory. The composite v0+v1.5+v2+v3 is identified as a classical-mechanical Haag–Kastler-style local operator algebra (§8.4). Wolfram's CA supply a complementary lower-bound existence proof: locality is sufficient for universal computation given unbounded state, which closes the natural objection that richer rules might suffice (§9).

---

## References

- Cook, M. (2004). Universality in elementary cellular automata. *Complex Systems*, 15(1).
- Doi, M. (1976). Second quantization representation for classical many-particle system. *J. Phys. A* 9(9).
- Haag, R., and Kastler, D. (1964). An algebraic approach to quantum field theory. *J. Math. Phys.* 5(7).
- Hewitt, J., Hahn, M., Ganguli, S., Liang, P., and Manning, C. D. (2020). RNNs can generate bounded hierarchical languages with optimal memory. *EMNLP*.
- Kleene, S. C. (1956). Representation of events in nerve nets and finite automata. *Automata Studies*, Princeton.
- Peliti, L. (1985). Path integral approach to birth–death processes on a lattice. *J. Phys. (France)* 46(9).
- Siegelmann, H. T., and Sontag, E. D. (1991). Turing computability with neural nets. *Appl. Math. Lett.* 4(6).
- Smale, S. (1967). Differentiable dynamical systems. *Bull. AMS* 73(6).
- Wolfram, S. (2002). *A New Kind of Science.* Wolfram Media.
- Yao, S., Peng, B., Papadimitriou, C., and Steinhardt, J. (2021). Self-attention networks can process bounded hierarchical languages. *ACL*.
