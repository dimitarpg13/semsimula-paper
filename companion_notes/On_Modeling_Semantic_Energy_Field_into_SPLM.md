# On Modeling the Semantic Energy Field into SPLM

## Question

*Did we somehow model the semantic energy field in the SPLM model?*

Short answer: **Yes — but only in part, and the partial coverage is
deliberate.** The framework's "semantic energy field" is a *composite*
of four distinct ingredients. Of those four, two have dedicated, named
slots in the SPLM architecture, one is absorbed implicitly into the
shared scalar potential $V_\theta$, and one is effectively absent as
the price of preserving SPLM's inference-efficiency claim. This
document lays out that correspondence honestly, piece by piece, and
flags the gaps as candidates for future work.

Supporting material for this note lives in the companion repo under
`notebooks/conservative_arch/` (main-text SPLM, §14.2–§14.12),
`notebooks/conservative_arch/sarf_variant/` (SARF-faithful $\xi$,
§14.13) and
`notebooks/conservative_arch/sarf_mass_variant/` (per-token semantic
mass, §14.14).

---

## 1. What the framework calls the "semantic energy field"

The field under discussion is built up across four chapters of the
main text (or, equivalently, four independent markdowns in this
`docs/` folder):

| ingredient | defined in | role |
|---|---|---|
| **Gaussian well** $V_{\mathrm{well}}(h)$ | §4 / `On_Gaussian_Inverse_Semantic_Energy_Well.*` | single-particle global attractor; pulls every hidden state toward the manifold of meaningful configurations |
| **PARF** — Property-Attractive–Repulsive Force | §5 / `Modeling_Attractive_and_Repulsive_Forces_in_Semantic_Properties.*` | *pairwise* forces between particles at the **property** level; depends on type/value matching between two tokens' property vectors |
| **SARF** — Structure-Attractive–Repulsive Force | §6 / `Modeling_Attractive_and_Repulsive_Forces_between_Semantic_Structures.*` | *pairwise* forces at the **structure** level, plus a time-dependent reinforcement field $\mathcal{E}(t)$ that aggregates the activation history of every structure active so far |
| **Semantic mass** $m_t$ | §7 and §11 / `On_the_Interpretation_of_Semantic_Mass.md` | per-token inertia that weights how strongly the kinetic term resists force-driven acceleration |

The full prescribed Lagrangian therefore reads

$$\mathcal{L} = \sum_t \frac{1}{2} m_t \lVert \dot{h}_t \rVert^2 - V_{\mathrm{well}}(h_t) - V_{\mathrm{PARF}}(h_t, h_{\ne t}) - V_{\mathrm{SARF}}(h_t, \mathcal{E}(t)),$$

where $h_{\ne t}$ denotes the set of hidden states at all positions other than $t$.

and the force on each particle is minus the gradient of all four
potential contributions at once. The *semantic energy field* the
question refers to is the second bracket on the right-hand side --
everything minus the kinetic term.

---

## 2. How SPLM represents each piece

SPLM's Euler–Lagrange update (Eq. (14.2) of the paper) is

$$h_t^{(\ell+1)} = h_t^{(\ell)} + \frac{dt}{1+\gamma} (h_t^{(\ell)} - h_t^{(\ell-1)}) - \frac{dt^2}{(1+\gamma) m} \nabla_h V_\theta(\xi_t^{(\ell)}, h_t^{(\ell)}).$$

The only scalar potential that appears anywhere in the architecture is
the single shared MLP $V_\theta(\xi, h)$. Everything the framework
calls an energy contribution must either live inside that MLP or in
its arguments $(\xi, h)$ — there is no second scalar, and no
per-layer offset. Against that constraint:

| ingredient | representation in SPLM | explicit / implicit / absent |
|---|---|---|
| **Reinforcement field $\mathcal{E}(t)$** (SARF time-dependent part) | the causal cumulative-mean pool $\xi_t^{(\ell)} = \frac{1}{t}\sum_{s\le t}h_s^{(\ell)}$, with layer $\ell$ playing the role of time | **explicit and named as such.** §14.2 of the paper states: *"$\xi^{(\ell)}$ is the SPLM-native analog of the reinforcement field $\mathcal{E}$ of §6"*. In the SARF-faithful variant of §14.13 the identification is *literal* — $\mathcal{E}$ is recomputed at every integration step. |
| **Semantic mass $m_t$** | the learnable scalar $m$ in the integrator; promoted to per-token $m_t^{(A)}$ (embed-head) and $m_t^{(B)}$ (Shannon-surprisal) in §14.14, with variant (B) $m_t \propto -\log\hat p(x_t)$ implementing the framework's information-theoretic prior directly | **explicit.** Variant (B) achieves 44 % Tiny-Shakespeare ppl reduction vs. fixed-$\xi$ SPLM. |
| **Gaussian well $V_{\mathrm{well}}(h)$** | not instantiated as a named Gaussian; its role is absorbed into the shared scalar MLP $V_\theta(\xi, h)$, which is free to learn whatever attractor geometry the training objective demands | **implicit.** The MLP can in principle approximate any smooth well (including the predicted Gaussian shape), but the functional form is not constrained. The Gaussian E-init validation at `notebooks/conservative_arch/e_init_validation.py` verifies *empirically* that $V_\theta$ converges to a Gaussian-like well, but this is a measured outcome, not an inductive bias. |
| **SARF structure-level pairwise forces** $V_{\mathrm{SARF}}(h_t, \mathcal{E}(t))$ | the $\xi$-dependence of $V_\theta$ captures the *field* a particle sees from other structures, but only in mean-field form — the aggregation $\xi$ throws away which token contributed what | **partially implicit.** The reinforcement field is explicit (via $\xi$); the bilinear structure $\times$ structure coupling is not disentangled from it and lives inside $V_\theta$'s weights. |
| **PARF pairwise property-level forces** $V_{\mathrm{PARF}}(h_t, \{h_s\})$ | no pairwise interaction term at all; token $t$ sees token $s$ only through the mean-field $\xi$, never directly as a pair $(h_t, h_s)$ | **effectively absent.** A deliberate simplification — restoring explicit pairwise PARF would reintroduce a $O(T^2)$ cost and forfeit the inference-efficiency claim. |

### 2.1 One-sentence summary of the mapping

> SPLM models the framework's semantic energy field as **one shared
> scalar MLP** $V_\theta(\xi, h)$, where $\xi$ is the SARF
> reinforcement field recomputed at every integration step; the
> Gaussian well and SARF structural forces live inside $V_\theta$
> implicitly, the reinforcement field and semantic mass are
> explicit, and explicit pairwise PARF is deferred to preserve the
> $O(T)$ inference-efficiency claim.

---

## 3. Why the architecture is structured this way

Three reasons, all documented in the paper:

1. **Conservativity demands a single scalar.** The whole point of SPLM
   is that the force is $-\nabla_h V_\theta$ for **one** scalar
   $V_\theta$. Decomposing $V$ into
   $V_{\mathrm{well}} + V_{\mathrm{PARF}} + V_{\mathrm{SARF}}$ is
   mathematically equivalent to having one scalar (the sum), so the
   decomposition adds no expressive power, only structural commitment.
   As long as the MLP is large enough to approximate the sum, nothing
   is lost.

2. **PARF as explicit pairwise forces breaks the inference-efficiency
   claim.** Every pairwise interaction between tokens resurrects a
   $O(T^2)$ term in the forward cost — exactly what the cumulative-
   mean pool $\xi$ was designed to avoid. The framework *allows* PARF
   in principle, but instantiating it faithfully in an attention-free
   circuit is a non-trivial architectural problem that would have
   pushed the paper's programme back to $O(T^2)$. Appendix B makes the
   efficiency argument precise; PARF in its naive form would
   invalidate it.

3. **The Gaussian well is a statement about shape, not about a
   named term.** §4 of the paper claims the basin is *approximately*
   Gaussian, not that the model must contain a literal Gaussian term.
   The empirical check that $V_\theta$ actually learns a Gaussian-
   like well at convergence lives in the E-init validation pipeline
   (`notebooks/conservative_arch/e_init_validation.py`) and is
   reported as a *positive* sanity check. The framework therefore
   does not force the architecture to contain a hand-crafted Gaussian;
   it asks the model to end up there.

---

## 4. What is still missing

If one pushes on "does SPLM implement *all* of the semantic energy
field?", the honest caveats are three:

### 4.1 Pairwise PARF is not represented at all

The mean-field $\xi$ is a strict aggregation: token identities and
positions inside the pool are lost. A token at position $t$ sees the
*average* of $\{h_s\}_{s\le t}$, not any particular partner. This is
the main gap between the framework's PARF machinery and SPLM. It is
also the single biggest barrier to lifting SPLM out of mean-field
physics into the genuinely pairwise regime the framework prescribes
for property-level interactions.

### 4.2 PARF and SARF are not disentangled

Even for the parts that *are* representable via $V_\theta$, we have no
way to ask "which fraction of the learned $V_\theta$ is PARF and
which is SARF?" They co-inhabit one MLP. Structurally this is fine —
the force is the sum of their gradients, and the sum is all that
appears in the dynamics — but diagnostically it prevents us from
making separate quantitative claims about either one.

### 4.3 The Gaussian-well shape is a learned outcome, not an inductive bias

Nothing in SPLM's parameterisation biases $V_\theta(\xi, h)$ toward
the $\lVert h \rVert^2$-quadratic form at small displacements; the
E-init validation just empirically confirms that it tends there.
Stronger claims (e.g. "SPLM *provably* has a Gaussian basin") would
require either an architectural constraint (add a hand-crafted
$\lambda \lVert h \rVert^2$ term to the MLP output) or a theoretical
argument about what $V_\theta$ must converge to.

---

## 5. Where these gaps would live as open questions

These three gaps are natural candidates for **Q11, Q12, Q13** in the
open-questions list of §14.17 / §16 of `paper_v2`. Framed that way
they would extend the existing programme from "*what more of the
mass prescription can we recover?*" (Q10) to "*what more of the
field prescription can we recover?*":

- **Q11 (low-rank PARF).** Can we add a rank-$r$ pairwise term
  $\sum_{s<t}\phi(h_t)^\top\psi(h_s)$ evaluated via associative
  scans, giving an *efficient* PARF at $O(T r d)$ cost? If so, does it
  close any of the remaining 13 % perplexity gap between SARF +
  logfreq SPLM and the matched attention baseline, and does it keep
  the shared-$V_\psi$ separator intact?
- **Q12 (named SARF / PARF decomposition).** Can we force the shared
  $V_\theta$ to admit a factorisation
  $V_\theta(\xi, h) = V_{\mathrm{well}}(h) + V_{\mathrm{PARF}}(h, \phi(\xi)) + V_{\mathrm{SARF}}(h, \xi)$
  through architectural priors, and measure the relative contribution
  of each term at convergence?
- **Q13 (architectural Gaussian-well prior).** Does biasing
  $V_\theta$ toward a quadratic-at-origin shape (e.g. residual
  Gaussian + learned correction) improve LM quality, shared-$V_\psi$
  fidelity, or both, compared to the unconstrained MLP?

Each of these is a one-ablation experiment comparable in size to the
SARF-faithful and per-token-mass studies already in §14.13 and
§14.14, and each would be reported in the same format (four-way
`compare.py`, per-layer diagnostic tables, interpretation).

---

## 6. One-paragraph summary for external use

> The Scalar-Potential Language Model (SPLM) represents the Semantic
> Simulation framework's semantic energy field as a single shared
> scalar MLP $V_\theta(\xi, h)$ trained end-to-end by backprop. Of
> the four ingredients the framework prescribes for that field — the
> Gaussian well, the property-level pairwise force PARF, the
> structure-level pairwise force SARF, and the time-dependent
> reinforcement field $\mathcal{E}(t)$ — the reinforcement field is
> modelled *literally* as the causal cumulative-mean pool $\xi^{(\ell)}$
> (recomputed at every integration step in the SARF-faithful variant
> of §14.13), and the semantic-mass prescription from §7 and §11 is
> modelled *literally* by the Shannon-surprisal variant of §14.14
> ($m_t \propto -\log\hat p(x_t)$). The Gaussian well and the
> structure-level SARF potential are absorbed *implicitly* into
> $V_\theta$, which is free to learn whatever geometry training
> demands; the Gaussian shape is verified empirically (E-init
> validation, `notebooks/conservative_arch/e_init_validation.py`)
> rather than built in. Explicit pairwise PARF is not represented —
> the mean-field $\xi$ is a strict aggregation that loses token
> identities inside the pool. This is a deliberate simplification:
> adding pairwise PARF naively would reintroduce a $O(T^2)$ cost and
> forfeit the inference-efficiency claim that Appendix B is built on.
> Recovering an *efficient* PARF, disentangling the learned
> $V_\theta$ into named components, and biasing the architecture
> toward the Gaussian-well shape are natural Q11 / Q12 / Q13 follow-
> ups to the Q8–Q10 programme of the current paper.
