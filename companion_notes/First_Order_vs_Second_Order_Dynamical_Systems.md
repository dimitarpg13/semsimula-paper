# First-Order vs Second-Order Dynamical Systems: Dynamical Order and State-Space Extension Are Independent Design Choices

> **Summary.** The Semantic Simulation framework adopts a damped
> second-order Lagrangian dynamics on the token subsystem. This note
> separates that base-dynamics choice from a distinct design lever —
> state-space extension via auxiliary Fock register particles — and shows
> that the two are orthogonal. The Conservative Obstruction Theorem
> applies to any conservative force field on the token particles
> regardless of dynamical order; expressivity is unlocked by extending
> the state space, not by adding inertia. The framework's second-order
> base dynamics is justified by trajectory-geometry content (tangential
> acceleration, interior damping optimum, Jacobi geodesics, mass
> protocol, overdamped synthesis) rather than by expressivity necessity.

**Paper reference:** §7 of `paper_v4/main.tex`
(Lagrangian framework, subsection on first-order vs second-order
dynamics), with cross-pointers to §1 (introduction preview), §12 (mass
protocol), §13 (STP acceleration), §15 (resonance predictor and
overdamped synthesis), §17c (Fock-augmented PARFLM), §18 (Jacobi
geodesic programme), and §19 (overdamped synthesis conclusion).

**Related documents:**
- [`Conservative_Obstruction_and_Virtual_Particle_Necessity.md`](Conservative_Obstruction_and_Virtual_Particle_Necessity.md)
  — proof of the no-go theorem that motivates state-space extension
- [`Improving_the_Fock_Mechanism_to_match_Attention.md`](Improving_the_Fock_Mechanism_to_match_Attention.md)
  — Q/K/V creation protocol design rationale for FockPARFLM v2

---

## 1. The question

The Semantic Simulation programme models language dynamics via a
damped second-order Euler-Lagrange equation. The natural challenge from
a dynamical-systems standpoint is whether second order is necessary at
all: most modern generative models (score-based diffusion, Langevin
samplers) and most analytic accounts of transformer hidden states (STP,
Lu et al. 2020) are first order. A related challenge follows: if we
need an attention-like inter-token routing mechanism (the v2 Fock
creation/destruction stage on the expressivity ladder), does that
require a second-order base, or would a first-order dynamics suffice?

The answer is that two distinct design axes have been conflated in the
question, and once they are separated the choices become clearer.

---

## 2. Three independent design axes

The architectural choices that determine what a particle-based
language-model dynamics can do are not one knob but three:

| Axis | First option | Second option |
|---|---|---|
| **A1 — Force field** | Conservative ($F = -\nabla V$) | Non-conservative (Helmholtz, gauge, gauge-like exchange) |
| **A2 — Dynamical order** | First-order ($\dot h = F$) | Second-order ($m\ddot h = F - m\gamma\dot h$) |
| **A3 — State space** | Token particles only ($h_1,\ldots,h_T$) | Extended (tokens + auxiliary registers) |

The three axes are independent. The Conservative Obstruction Theorem
sits on axis A1; expressivity beyond the regular class is unlocked by
axis A3; the second-order base-dynamics content (acceleration, mass,
resonance, geodesics) sits on axis A2. Treating any pair of these axes
as a single choice obscures the design landscape.

---

## 3. The Conservative Obstruction Theorem is order-independent

Recall the theorem (see
[`Conservative_Obstruction_and_Virtual_Particle_Necessity.md`](Conservative_Obstruction_and_Virtual_Particle_Necessity.md)):

> Let $h_1, \ldots, h_T$ in $\mathbb{R}^d$ be a system of token
> particles with forces $F_i = -\nabla_{h_i} V$ derived from a $C^2$
> scalar potential. Then the force map cannot simultaneously satisfy
> the three structural properties of scaled dot-product attention
> (P1 asymmetric coupling, P2 coupling-content decoupling, P3 normalised
> budget).

The proof of Lemma 1 (Jacobian symmetry $\Rightarrow$ no P1) invokes
Schwarz's theorem on $V$:

$$
\frac{\partial F_i^\alpha}{\partial h_j^\beta} = -\frac{\partial^2 V}{\partial h_j^\beta \partial h_i^\alpha} = -\frac{\partial^2 V}{\partial h_i^\alpha \partial h_j^\beta} = \frac{\partial F_j^\beta}{\partial h_i^\alpha}.
$$

This identity is a statement about $V$. It mentions $\dot h$ nowhere
and $\ddot h$ nowhere. The off-diagonal Jacobian of the force field is
symmetric whenever the force is a gradient of a scalar — regardless of
whether the force is then plugged into a first-order ODE, a Langevin
equation with thermal noise, or a second-order Newton equation with
damping. Concretely, the same obstruction applies to all three of:

- First-order gradient flow: $\dot h_i = -\nabla_{h_i} V$.
- Langevin diffusion: $\dot h_i = -\nabla_{h_i} V + \sqrt{2T}\eta_i$.
- Damped second-order Newton: $m\ddot h_i + m\gamma\dot h_i = -\nabla_{h_i} V$.

In each case the off-diagonal force-Jacobian on the token subsystem is
symmetric, so P1 fails. Lemmas 2 and 3 then rule out P2 and P3 by
arguments that likewise depend only on the gradient structure, not on
the order of the ODE. The theorem therefore lives entirely on axis A1.

**Corollary.** Any choice on axis A2 alone — first- or second-order — is
insufficient to escape the obstruction. The escape route is axis A3,
state-space extension.

---

## 4. A first-order Fock architecture is fully coherent

Once axis A3 is exercised, the obstruction lifts. The Fock register
mechanism extends the state from $(h_1,\ldots,h_T)$ to
$(h_1,\ldots,h_T,\ r_1,\ldots,r_M,\ \sigma_1,\ldots,\sigma_M)$. On this
extended state a perfectly valid first-order autonomous vector field is

$$
\dot h_i = -\nabla_{h_i} V_\theta(h_i) + \sum_{j \neq i} F_{ij}^{(\mathrm{PARF})} + \sum_{k \in \mathrm{active}} \alpha_{ik} v_k^{(\mathrm{reg})},
$$

$$
\dot r_k = -\lambda r_k + \sum_{j=1}^{T} \alpha_{kj} W_V h_j,
$$

$$
\dot \sigma_k = -\lambda \sigma_k + (1-\lambda) \max_j \alpha_{kj}.
$$

The structural argument that this restores P1, P2, P3 is identical to
the argument in §5 of the companion obstruction note:

- **P1 (asymmetry).** Creation uses
  $(q_k^{(\mathrm{reg})}, k_j^{(\mathrm{tok})})$ while the reverse
  channel uses $(q_i^{(\mathrm{tok})}, k_k^{(\mathrm{reg})})$. These
  involve independent projection matrices.
- **P2 (decoupling).** Softmax coefficients $\alpha_{kj}$, $\alpha_{ik}$
  determine coupling; independent value projections $W_V h_j$,
  $W_V^{(\mathrm{reg})} r_k$ carry the content.
- **P3 (budget).** Both softmaxes are normalised in their summation
  index; the total reverse-channel force on each token is $O(1)$ in $T$.

None of these arguments invoke inertia or velocity. The asymmetric
attention-like routing is enabled by **the Q/K/V projections, the
softmax normalisation, and the auxiliary register state** — not by
the order of the ODE on the tokens.

In fact, most existing transformer variants — Lu et al. (2020)'s
multiparticle reading, the Universal Transformer's ODE limit, and most
diffusion-based language models — already implement attention-like
routing inside a first-order ODE on an extended state (the residual
stream). Experiment A's per-layer fits in the present paper confirm that
attention transformer inference is non-autonomous and effectively
first-order at every layer. A first-order Fock-augmented architecture
would inherit the same expressivity class lift (regular to context-free)
that FockPARFLM v2 enjoys.

---

## 5. The QED parallel: hybrid first-order field, second-order particles

The cleanest natural analogy is electromagnetism. In QED:

- The electromagnetic field $(E, B)$ obeys Maxwell's equations, which
  are first-order in time:
  $\partial_t E = \nabla \times B - J$, $\partial_t B = -\nabla \times E$.
- Charged particles obey the second-order Lorentz force equation:
  $m\ddot x = q(E + \dot x \times B)$.

The gauge field has no inertia; the charges do. Force is mediated
between charges by exchange of virtual photons (perturbative-QED
language), and the dynamical orders of the two subsystems differ.

FockPARFLM v2 has the same hybrid structure already:

| QED | FockPARFLM v2 | Dynamical order |
|---|---|---|
| Charged particle $e^-$ | Token particle $h_i$ | Second-order |
| Gauge field / virtual photon $\gamma^*$ | Register particle $r_k$ with salience $\sigma_k$ | First-order (overwrite-with-decay) |
| Vertex coupling | Q/K/V creation gate and reverse channel | n/a |
| Photon lifetime | Salience lifetime $\tau \approx 1/(1-\lambda)$ | n/a |

The registers carry content and salience, both first-order quantities
updated by overwrite-with-decay; there is no $\ddot r_k$. The tokens
carry position and velocity. The hybrid structure is not accidental —
it mirrors the canonical physical realisation of the same conservative
obstruction and its gauge-mediated resolution.

A *fully* first-order Fock variant (tokens also first-order) would
correspond to the non-relativistic, instantaneous-Coulomb-only limit
of QED: still well-defined, but stripped of the dynamical content that
makes the gauge picture quantitatively predictive.

---

## 6. What second-order base dynamics buys you

The second-order choice for the token subsystem is justified by what it
adds to the framework's predictive content, not by what it enables on
the expressivity ladder. The genuine second-order content is:

**(B1) Tangential acceleration content.** Acceleration decomposes as
$\vec a = a_\parallel \hat t + \vec a_\perp$. A first-order theory has
no acceleration at all. The STP regulariser
of Huang, LeCun, and Balestriero (2026) measures only $\lVert\vec a_\perp\rVert$,
via the framework's STP-acceleration identity
$\mathrm{STP} = 1 - \sqrt{1 - \lVert\vec a_\perp\rVert^2 / \lVert\vec d_2\rVert^2}$.
On GPT-2 the measurement
$\lvert a_\parallel\rvert \approx 2 \lVert\vec a_\perp\rVert$
with $a_\parallel \lt 0$ on 97.9% of consecutive triplets shows that
half of the empirical acceleration content is invisible to STP's
first-order ansatz. This is the precise sense in which §15 says STP is
an incomplete description of the trajectory.

**(B2) Interior damping optimum and resonance predictor.** The E4
damping sweep finds an interior optimum
$\gamma^\ast = 0.10$ on TinyStories — a critical-damping signature with
no analogue in first-order systems. The closed-form predictor
$\gamma^\ast = (m/(L \Delta t)) \ln(1/\rho)$ is verified at two
distinct operating points and requires $m$ as a real degree of freedom.

**(B3) Jacobi geodesic / Riemannian structure.** The Jacobi metric
$g_{ij} = 2(E - V) \delta_{ij}$ is intrinsic to a Lagrangian with a
kinetic energy term. Free motion under $L = T - V$ is a geodesic of
this metric. In a first-order theory there is no kinetic term, no
metric, no curvature, no parallel transport. §18 (Riemannian
geometry) is therefore structurally impossible in a first-order
framework.

**(B4) Mass-velocity protocol (E-init).** The protocol of §12 predicts
the entire trajectory $\hat h_t^{(\ell)}$ for $\ell = 2, \ldots, L$
from a measured triple $(h_t^{(0)}, \dot h_t^{(0)}, w_t)$. A first-order
theory has no independent $\dot h_t^{(0)}$ — the initial condition is
just the position — so the protocol does not exist.

**(B5) Symplectic structure and Verlet stability.** The conservative
limit ($\gamma = 0$) of the second-order dynamics is a Hamiltonian
flow. Verlet integration is symplectic, has bounded energy drift over
long horizons, and admits conserved-quantity diagnostics
(Noether-style). First-order flow has none of this — its only invariant
is the level set of $V$.

**(B6) Asymmetric superiority via the overdamped reduction.** In the
limit $\gamma \to \infty$ the damped second-order equation reduces to

$$
\gamma \dot h = -\nabla V,
$$

which is exactly first-order gradient flow. So every first-order
content in the literature — STP, Lu et al. (2020), score-based
diffusion, Langevin — is recovered as the overdamped limit of the
present second-order framework. The converse is false: B1 through B5
cannot be reconstructed from a first-order ODE no matter how it is
parameterised. The second-order framework is strictly more expressive
in the dynamical-content sense.

This is the overdamped-synthesis argument of §19: STP and Lu are not
competitor models; they are predicted shallow limits of the same
Lagrangian.

---

## 7. What first-order would give you (honest counter-case)

For completeness, the cases in which a first-order base dynamics is
attractive:

**(C1) Direct match to score-based generative modelling.** Langevin
dynamics $\dot h = -\nabla V + \sqrt{2T} \eta$ converges to the Gibbs
distribution $p \propto e^{-V/T}$. The Fokker-Planck equation gives a
clean probabilistic interpretation. Diffusion-language-model lineages
(SEDD, DiffusionLM, etc.) sit naturally here.

**(C2) Fewer state variables, no mass parameter.** A first-order
system has half the state-space dimension of its second-order analogue
($h$ only, not $(h, \dot h)$). No mass parameter; no learnable
damping; simpler integrators.

**(C3) Direct fit to first-order accounts of transformers.** STP
(Huang, LeCun, Balestriero 2026), Lu et al. (2020), Universal
Transformer ODE readings, and most score-based language models are all
first-order. A first-order base dynamics maps to them without going
through an overdamped reduction.

**(C4) Experiment A's observational signature.** The per-layer
trajectory-fitting experiment in this paper shows that pretrained
attention-transformer inference is non-autonomous and effectively
first-order at every layer. If the goal were purely descriptive (model
what attention does), first-order would be the parsimonious choice.

The framework's response to (C4) is that its mission is prescriptive
rather than descriptive: the goal is not to fit attention but to
*replace* it with a principled second-order system whose geometric
properties are accessible by construction. This is the mission
statement of §1 and §19 in the v4.5 framing.

---

## 8. Why the second-order choice is kept for FockPARFLM v2

Given that (i) the Fock mechanism does not require second-order
dynamics and (ii) attention transformer inference is observably
first-order, why does FockPARFLM v2 keep a second-order base for the
tokens? Three reasons:

**(D1) Consistency with PARFLM.** PARFLM is second-order by
construction. Fock-augmenting it with reverse-channel forces is a
strictly additive extension that preserves the rest of the Lagrangian
framework. A first-order Fock variant would require parallel
re-derivation of every PARFLM result on the same task suite.

**(D2) The generalised-force slot $Q_i$ has a designated home.** In
the Euler-Lagrange equation with non-conservative forces,

$$
\frac{d}{dt} \frac{\partial L}{\partial \dot h_i} - \frac{\partial L}{\partial h_i} = Q_i,
$$

the reverse-channel exchange force lands cleanly in the $Q_i$ slot, with
the conservative pieces $V_\theta$ and $V_\phi$ unchanged. The
decomposition between "what comes from the scalar potential"
(conservative half) and "what is non-conservative Fock exchange" is
manifest. In a first-order ODE the non-conservative term is just an
additive drift; coherent but less structurally clean.

**(D3) The shared-potential $R^2$ separator stays diagnostic.** The
separator of §15 measures whether the conservative pieces explain the
per-layer dynamics. Keeping the base second-order means the same
diagnostic continues to apply to FockPARFLM v2, and the model is
predicted to land in the attention quadrant precisely because of the
deliberate non-conservative reverse channel — quantitatively
distinguishable from a conservative architecture that happens to be
expressive enough.

---

## 9. Decision principle

A single-line statement of the design logic:

> Dynamical order (axis A2) and state-space extension (axis A3) are
> independent design choices. The Fock mechanism (axis A3) is what
> lifts the expressivity class from regular to context-free; second
> order (axis A2) is what gives the base trajectory the tangential
> acceleration, resonance, geodesic, mass-protocol, and overdamped-
> synthesis content that no first-order framework can generate.

The framework chooses second order on axis A2 *and* Fock extension on
axis A3 because the two choices answer different design questions, and
each pays in a different empirical ledger.

---

## 10. Summary

| Axis | Choice | Justified by | Independent of |
|---|---|---|---|
| A1 (force field) | Conservative base + non-conservative Fock exchange | Conservative Obstruction Theorem (§17c) | Choice on A2 and A3 — obstruction applies uniformly |
| A2 (dynamical order) | Second-order base for tokens | Tangential acceleration content (§13), resonance (§15), geodesics (§18), mass protocol (§12), overdamped synthesis (§19), asymmetric superiority over first-order frameworks | Choice on A3 — first-order Fock variant is coherent but loses B1–B5 |
| A3 (state space) | Token + Fock register extension | Conservative Obstruction Corollary — expressivity beyond regular class requires auxiliary state | Choice on A2 — extension works under either dynamical order |

The natural shape of the framework is the **(conservative base + Fock
extension, second-order on tokens, first-order on registers)** corner —
exactly the configuration of FockPARFLM v2. The design follows from
treating the three axes as independent and choosing each on its own
empirical and theoretical merits.
