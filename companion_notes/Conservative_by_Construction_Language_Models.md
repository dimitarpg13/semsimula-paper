# Conservative-by-Construction Language Models

*A prescriptive reading of the Semantic Simulation framework in light of
the empirical failures of conservative and linear-Helmholtz fits to
attention-transformer hidden-state trajectories.*

**Status:** design document / research plan for v2 of the paper.
**Companion:** [`The_Failure_of_Conservative_Models_to_explain_hidden_state_trajectories.md`](./The_Failure_of_Conservative_Models_to_explain_hidden_state_trajectories.md)
(hereafter "**Failure doc**") contains the five negative experiments that
motivate this document.

---

## 0. TL;DR

The five negative experiments of the Failure doc -- scalar potentials
(§1.1-1.3), linear position-coupled skew $\Omega\,x$ (§1.4), constant and
affine-in-$x$ velocity-coupled skew $B(x)\,\dot x$ (§1.5), and the
combined $\Omega x + B \dot x$ linear U(1) gauge (§1.5) -- do **not**
refute the Semantic Simulation framework. They demonstrate a different,
stronger claim:

> **Current decoder-only attention transformers are not optimally
> designed for the semantic-inference task.**  A circuit whose
> inference dynamics obeys a damped Lagrangian flow on a learned scalar
> energy $V(h)$ is structurally simpler, admits a closed-form physical
> interpretation, and is the natural prescription of the Semantic
> Simulation theory.  The theory was always meant to be **prescriptive**
> (telling us how to design the circuit), not merely **descriptive**
> (retro-fitting whichever architecture happens to exist).

Under this reframing, the Failure doc's five negative results become the
first half of a two-sided argument:

1. **Negative control (done):** attention transformers **do not**
   admit a Lagrangian description of their hidden-state trajectories.
2. **Positive control (this document's experimental programme):** a
   conservative-by-construction circuit *does* admit such a description,
   is trainable, and is competitive per parameter on language-modelling
   tasks.

This document captures the logic of that reframing, enumerates concrete
conservative-by-construction architectures, proposes the minimal
experimental protocol to validate the approach, and outlines the
resulting v2 narrative.

---

## 1. Reframing the five negative experiments

A physical theory of semantic inference has two jobs:

1. **Identify the minimal dynamical law that a correctly-designed
   semantic-inference circuit should obey.**
2. **Use that law as a design criterion for new circuits.**

Read that way, §1.1--§1.5 of the Failure doc are a **prediction of the
framework**, not a refutation of it: *if* semantic inference is conservative
(or damped-Lagrangian) motion on a semantic manifold, and *if* current
attention transformers are not that, *then* current attention transformers
are doing something more complicated and less interpretable than what the
inference task actually requires.

The Semantic Simulation framework is therefore a **prescriptive theory**:
it tells us which simpler circuits *would* admit a clean
Lagrangian / geodesic description, and predicts those should be
competitive or better per parameter on the inference task -- with the
added bonus of interpretable semantic trajectories, provable stability,
and direct access to the full Noether / Hamilton--Jacobi toolkit for
analysis.

## 2. Why attention is structurally non-conservative

Each of the following architectural choices in the standard
decoder-only transformer block violates one condition for conservatism.
Taken together they guarantee that the hidden-state flow cannot be
written as $-\nabla V(h)$ or even as $-\nabla V(h) + F(h)\dot h$ with
antisymmetric $F$:

| transformer feature | non-conservative consequence |
|---|---|
| asymmetric $W_Q \neq W_K^{\top}$ attention | similarity $q^{\top}k$ is not the gradient of any scalar $V(h)$ -- exactly the non-integrability of $A \to B$ vs. $B \to A$ moves in §3 of the Failure doc. |
| multi-head concatenation | a sum of per-head torques cannot in general be written as $\nabla V$; the Yang--Mills / non-abelian obstruction. |
| causal mask + position-dependent $h_{<t}$ context | force at $h_t$ depends on extrinsic history, violating $\vec F = \vec F(h)$. |
| LayerNorm after residual | projects onto a non-Euclidean manifold whose induced metric breaks Hamiltonian volume preservation. |
| distinct $W^{(\ell)}$ per layer | the "force law" changes every step, so the layer-indexed evolution is not a time integration of any single Lagrangian. |
| softmax in attention | exponentiation + normalisation is not a volume-preserving map on $h$. |

Remove or symmetrise any one of these and one recovers a small **island
of conservatism**; remove them all and the resulting architecture is
Lagrangian by construction. The standard decoder-only transformer
violates all six simultaneously, which is why the five Failure-doc
experiments fail uniformly.

## 3. Conservative-by-construction architectures

All of the following exist in the literature or are within reach, and
none has had E-init-style hidden-state-trajectory-fit diagnostics run on
it. Together they define a natural experimental programme for v2.

| architecture | conservative structure | prior art |
|---|---|---|
| **Modern Hopfield network (symmetric, continuous)** | $\dot h = -\nabla E(h)$ with $E(h) = -\mathrm{lse}(\beta X h)$ stored-pattern energy; one update = one gradient step | Ramsauer et al. 2020 *"Hopfield Networks Is All You Need"*; Krotov & Hopfield 2016, 2020 |
| **Symmetric attention (tied $Q=K$), no softmax, weight-tied across layers** | $h \leftarrow h - \eta\,\nabla_h (\tfrac12 h^{\top} W h)$ with $W=W^{\top}$; pure gradient flow on a quadratic energy | Katharopoulos et al. 2020 *"Transformers are RNNs"*; Tay et al. 2021 *"Synthesizer"* |
| **Hamiltonian Neural Networks (HNN)** applied to token dynamics | learn a scalar $H(h, p)$ and evolve via $\dot h = \partial_p H$, $\dot p = -\partial_h H$; energy-conserving up to integrator error | Greydanus et al. 2019; Finzi et al. 2020 |
| **Lagrangian Neural Networks (LNN)** applied to token dynamics | learn $L(h, \dot h)$ as a neural scalar; integrate the Euler--Lagrange equations | Cranmer et al. 2020 |
| **Deep Equilibrium Models with monotone / symmetric fixed-point map** | the equilibrium $h^{*} = \arg\min_h V(h)$ is by definition the minimum of an implicit scalar $V$ | Bai, Kolter, Koltun 2019 (DEQ); Winston & Kolter 2020 (monotone DEQ -- provably conservative) |
| **Neural scalar-potential language model** *(this document's canonical v2 proposal)* | a single learned scalar $V_\theta : \mathbb{R}^d \to \mathbb{R}$ of the pooled context; inference = damped Lagrangian flow on $V_\theta$; next-token head reads off basin of attraction | no direct prior art -- this is what Semantic Simulation theory directly prescribes |

### 3.1 The canonical v2 proposal in plain terms

The last row is the sharp, minimal form of the architectural thesis.
Concretely:

- **Context pooling (permutation- / symmetry-invariant):** the context
  $h_{<t}$ is summarised into a single pooled state
  $\xi \in \mathbb{R}^d$, e.g. via a set-pooling that is manifestly
  symmetric under permutation of context tokens. This replaces
  asymmetric attention with an explicit invariant.

- **Single scalar energy field $V_\theta(\xi, h)$:** parameterised by
  a small neural network (e.g. MLP or shallow residual stack ending
  in a scalar), smooth in $h$.

- **Inference = deterministic damped Lagrangian flow:** for $L$ integration steps,
  $$\mathfrak m\,\ddot h = -\nabla_h V_\theta(\xi, h) - \mathfrak m\,\gamma\,\dot h,$$
  with fixed $(\mathfrak m, \gamma)$ (or learnable scalars). The "depth"
  of the network is the number of integration steps on a **single
  shared** force law -- i.e. weight-tied over layers.

- **Output head:** read off the next-token class from the final state
  $h_L$ (or, equivalently, from the attractor basin that
  $h_0 \mapsto h_L$ falls into).

- **Training:** fully differentiable. Maximise log-likelihood of the
  ground-truth next token at the end of this deterministic flow; all
  parameters live in $V_\theta$, the pooling module, and
  $(\mathfrak m, \gamma)$.

This circuit is **conservative by construction** (one scalar field,
smooth in $h$); **weight-tied across layers** (the whole "depth" is one
force law repeated); **permutation-symmetric in the context**; and has
**no softmax, no multi-head split, no per-layer asymmetric weights**.
The entire inference dynamics fits inside a single scalar field plus a
pooled-context summary.

### 3.2 Trade-offs and lightly-relaxed variants

Two natural relaxations remain strictly inside the conservative-by-design
envelope:

- **Scalar potential with context-conditioning on a single manifold
  chart.** $V_\theta(\xi, h)$ above is already context-conditioned; one
  can also let $\mathfrak m$ or $\gamma$ depend on $\xi$ (the "effective
  mass" per semantic region), which preserves the Lagrangian structure
  but gives more expressive landscapes.

- **Non-flat kinetic metric on $h$.** Replace the Euclidean kinetic term
  $\tfrac12\,\mathfrak m\,\lVert\dot h\rVert^2$ with
  $\tfrac12\,g_{ij}(h)\,\dot h^i \dot h^j$ where $g$ is a learned
  positive-definite metric.  Dynamics are then geodesic + potential on
  $(M, g, V)$, which is the §14 Jacobi form.  Still conservative, no
  non-integrable rotation.

## 4. Empirical validation plan

What turns this from a manifesto into a paper is a concrete experimental
protocol that can succeed or fail in well-defined ways.

### 4.1 Minimal positive-control experiment

1. **Train a small instance** of row 1, 2, or 6 of the table in §3 to
   a non-trivial LM task at GPT-2-small parameter count or smaller.
   Practical first-pass targets:
   - **Dataset**: Tiny Stories (narrative, short-context) or WikiText-2
     (standard, slightly harder).
   - **Parameter budget**: ~10--50 M parameters, comparable to GPT-2-small (124M) but slanted towards the scalar-potential network rather than attention.
   - **Baselines**: a matched-parameter GPT-2-small; a matched-parameter
     linear / performer-style transformer; optional: a matched-parameter
     S4 / Mamba.

2. **Extract hidden-state trajectories** from the trained
   conservative-by-design model using exactly the §1 E-init
   methodology of the Failure doc: same per-sentence per-layer
   centering, same train/test split size, same 12-step symplectic
   integration.

3. **Run the same five fits** (scalar $V$, $\Omega x$, $B v$,
   $B(x)v$ affine, combined) and measure the layer-$L$ residual
   relative to the static null.

4. **Decision criterion.** If any of the fits -- most cleanly the
   pure scalar $V$ fit -- goes *below* the static null on this
   architecture with the same diagnostics, this is a
   **quantitative positive result** that the conservative
   description works on a correctly-designed circuit. The separation
   between that residual and the GPT-2 static-null floor is the main
   number that will appear in v2's main table.

5. **Bonus analysis (if step 4 succeeds).** Extract $V_\theta$ along
   observed trajectories. Visualise the energy landscape. Check
   whether its local minima cluster by semantic category, domain, or
   syntactic context. This is the paper's original interpretability
   programme, finally operational because the potential is now a
   real learned object rather than a post-hoc fit.

### 4.2 Falsification paths

The experiment as designed can fail in three informative ways, each
with its own follow-up:

- **Architecture trains fine, trajectory fit succeeds (expected):**
  v2 has its positive pillar.
- **Architecture trains fine, trajectory fit fails:** the theoretical
  picture was wrong at the prescriptive level too. Important negative
  result; directly suggests moving to the Riemannian / non-flat-kinetic
  variant of §3.2.
- **Architecture does not train competitively at all:** non-conservative
  parts of attention are buying computational capacity the theory
  hasn't accounted for. Still publishable -- quantifies the
  performance penalty of strict conservatism and gives a lower
  bound on how much "richness" a successful inference circuit needs
  beyond the minimal Lagrangian.

### 4.3 Scaffolding (deliverable layout)

Planned directory structure under `notebooks/conservative_arch/` for
the v2 work:

```
notebooks/conservative_arch/
  scalar_potential_lm.py       # row-6 architecture + training loop
  trajectory_extraction.py     # reuses §1 infrastructure unchanged
  e_init_validation.py         # runs the same five fits on the new LM
  results/
    scalar_potential_lm_training_log.md
    scalar_potential_lm_e_init_summary.md
    fig_residual_vs_gamma_conservative_arch.png
    fig_residual_vs_layer_at_gamma_star_conservative_arch.png
```

Parallel directories for the other architectures
(`modern_hopfield_lm`, `symmetric_linear_attn_lm`, etc.) are
structurally identical and can share the trajectory-extraction and
E-init validation modules unchanged.

## 5. The resulting v2 narrative

Under this re-framing, the v2 paper tells a much sharper story than v1
can:

1. **Theory (unchanged from v1).** Semantic inference is Lagrangian
   motion of semantic particles on an energy manifold.

2. **Prediction (sharpened).** Circuits that obey this theory should
   admit clean scalar-potential fits of their hidden-state
   trajectories. Circuits that do not obey it should not.

3. **Experiment 1 -- negative control (the five Failure-doc experiments).**
   Standard GPT-2-style decoder-only attention does not admit such
   fits; its hidden-state flow is neither conservative nor
   linear-Helmholtz. This falsifies the view that current attention
   transformers are already optimal instantiations of semantic inference.

4. **Experiment 2 -- positive control (this document's §4 programme).**
   A minimal conservative-by-construction circuit (e.g. the
   scalar-potential LM or the symmetric-attention Hopfield) trained
   to the same task admits clean scalar-potential fits of its
   hidden-state trajectories, and achieves competitive perplexity at
   matched or smaller parameter count.

5. **Prescriptive conclusion.** Well-designed semantic-inference
   circuits are *simpler* than attention transformers, are directly
   characterised by the Semantic Simulation framework, and expose
   interpretable energy landscapes by construction. Attention can be
   understood as an over-parameterised surrogate for what is
   fundamentally a scalar-gradient-flow problem; its
   non-conservativity is cost, not value.

## 6. Open questions and honest caveats

Before committing to v2, these are the known risks and unknowns.

- **Scale-up feasibility of conservative-by-construction LMs has not
  been established.** Hopfield-style, HNN-style, and weight-tied
  symmetric-attention architectures have been competitive on small
  and medium tasks but have not been scaled to GPT-2 parameter count
  on standard LM corpora. There is real risk that the conservative
  architecture fits its own trajectories beautifully but loses on
  perplexity per parameter. That outcome is still publishable
  (§4.2, third bullet), but should be anticipated.

- **Expressive capacity of a single scalar field is unproven for
  natural language.** A sufficiently rich $V_\theta$ is a universal
  function approximator, but the practical question is whether finite
  damping on a finite-depth integration step can sort token basins
  with the granularity natural language requires. The Hopfield-capacity
  results (Ramsauer et al.) say $\exp(d)$ stored patterns are
  retrievable with $\log d$ steps, which is optimistic but not a
  proof.

- **Causal asymmetry at the token level.** Autoregressive LM is
  causally asymmetric: the future conditions on the past but not
  vice versa. Can a conservative circuit express this? Yes, but
  only if the asymmetry is externalised to the graph structure
  (one integration per token, conditioning only on past $\xi$)
  rather than built into the per-step force. The canonical v2
  proposal handles this cleanly: the pooled context $\xi$ is
  recomputed for every new token, but each per-token inference is a
  conservative flow.

- **"Conservative by construction" vs. "empirically conservative".**
  Even a conservative-by-design circuit may show small non-conservative
  residuals at inference due to numerical integration error, finite
  precision, or regularisation. The E-init diagnostic must therefore
  be applied with clear expectations: we expect test residuals
  **below** the static null, not exactly zero.

- **Relationship to the Jacobi / Riemannian formulation (§14 of the
  paper).** The canonical v2 proposal is the flat-metric case of the
  Jacobi programme; the non-flat-metric variant (§3.2 here) is the
  full one. Both are strictly inside the "conservative-by-design"
  envelope. If the flat case fits well, the flat case is the paper.
  If not, the non-flat version is the natural escalation and the
  Christoffel structure directly accommodates the symmetric
  non-Hessian component that §1.5 isolated as the dominant residual
  structure.

## 7. Immediate next step

Build `notebooks/conservative_arch/scalar_potential_lm.py` and a
matched-size GPT-2-small baseline trainer, and get both to run on a
small corpus (Tiny Stories / WikiText-2) to convergence. Once both are
trained, the trajectory-extraction and E-init-validation scripts can be
ported over in an afternoon from the existing §1 infrastructure. The
decision criterion of §4.1 step 4 then gives us the first quantitative
answer to the question that motivates this document: **does a circuit
whose inference dynamics is conservative-by-design, in fact, admit a
cleaner hidden-state-trajectory fit than a standard attention
transformer of matched parameter count?**

If the answer is yes, the Semantic Simulation framework stops being a
post-hoc phenomenology and becomes an architectural design principle.
If the answer is no, we learn exactly what additional structure
(non-flat kinetic metric, multi-scalar potentials, non-abelian gauge)
is needed to recover a complete theory.

Either way, the result is the sharp, testable, architectural claim that
v2 needs.

## 8. Status update (2026-04)

The step-1 prototype (`notebooks/conservative_arch/`) is built and
converged on Tiny Shakespeare ($d=128, L=8$, val ppl 287), and two
quantitative diagnostics have been run against pretrained GPT-2 small
as a negative control.

**Step 1 -- velocity-aware Jacobian symmetry test (§4.1 step 4, local
level).** The per-step linear operator $M_\ell$ on the PCA-16 subspace
is approximately symmetric for both architectures: full-vs-symmetric
TEST $R^2$ gaps are $\le 0.04$ for SPLM and $\le 0.08$ for GPT-2. This
rules out a non-trivial skew-symmetric (curl / Helmholtz) force
component in either model -- consistent with §1.5 of the Failure doc
-- but does *not* rule out the existence of some scalar potential whose
Hessian matches the per-layer $M_\ell$'s. The local-linear level is
therefore not a clean separator between the two architectures.

**Step 2 -- strict shared-potential fit (the promised quantitative
separator).** We next asked the stronger question: *does a single
smooth scalar $V_\psi(h)$ -- a 2-layer MLP with hidden 256 -- whose
gradient reproduces $\Delta x_\ell$ across **every** layer and every
held-out sentence exist?* Step-2 + step-3 results at a glance:

| metric | SPLM (PC, 7.1 M) | Matched GPT-style (NC, 8.0 M) | Pretrained GPT-2 (NC, 124 M) |
|---|--:|--:|--:|
| median per-layer TEST $R^2$ | **+0.90** | **+0.56** | **+0.45** |
| min    per-layer TEST $R^2$ | +0.28 | +0.27 | +0.04 |
| middle-band mean $R^2$$^{*}$ | **+0.86** | +0.48 | **+0.09** |
| # layers with $R^2 \ge 0.5$ | 6 / 7 | 5 / 7 | 5 / 11 |
| median gain over velocity-only | **+0.46** | **+0.41** | **+0.09** |

$^{*}$ middle band: SPLM $\ell=3..5$, matched $\ell=3..5$, GPT-2 $\ell=6..10$.

The three models exhibit **three different curve shapes** on the shared-
potential test (one-pane each in `sharedV_three_way.png`):

- **SPLM:** uniform high R², as the construction prescribes.
- **Matched GPT-2-style:** monotonic decay from 0.81 at $\ell=1$ to 0.27
  at $\ell=7$.  The early part of the stack is well-approximated by a
  single scalar; the later part is not.
- **Pretrained GPT-2 small:** "bathtub" -- boundary layers (1, 2, 11)
  approach $R^2 \approx 1$, but middle layers ($\ell = 6..10$) collapse to
  $R^2 \le 0.2$.

**$V_\psi$ capacity is not the bottleneck.** A 6-config sweep on GPT-2
(hidden $\in \{128, 256, 512, 1024\}$, depth $\in \{2, 3, 4\}$, up to
1.84 M $V_\psi$ parameters, 3 k AdamW steps each) yields **middle-layer
$R^2$ values identical to 3 decimals across all configurations**.  The
failure is structural, not representational -- see
`sharedV_capacity_sweep_summary.md` and `sharedV_capacity_sweep_saturation.png`.

**Upper-bound (oracle) reference on SPLM.** Replacing the learned
$V_\psi(h)$ with SPLM's own $V_\theta(\xi, h)$ recovers the exact
integrator: TRAIN = TEST $R^2 = 1.0000$ on every layer, with
$\alpha_\ell = 0.5099, \beta_\ell = 0.5202$ (matching $dt/(1+\gamma)$
and $dt^2/((1+\gamma)m)$ to 4 decimals).  So the ~0.10 gap from the
oracle to the learned $V_\psi(h)$ for SPLM quantifies the *context
drop* $\xi \to \emptyset$ in the $V_\psi$ ansatz -- not a limitation
of the shared-scalar hypothesis itself.

Details and figures:
[`notebooks/conservative_arch/results/step3_comparative_summary.md`](../notebooks/conservative_arch/results/step3_comparative_summary.md),
[`sharedV_three_way.png`](../notebooks/conservative_arch/results/sharedV_three_way.png),
[`sharedV_capacity_sweep_summary.md`](../notebooks/conservative_arch/results/sharedV_capacity_sweep_summary.md).

**Step 4 -- coordinate-system robustness (token-direction replication).**
The step-2/3 test uses *depth-as-time* at fixed token.  We repeated
both diagnostics along the orthogonal axis -- *token-as-time at fixed
layer* -- which is the natural coordinate of autoregressive inference
and the one STP's Geodesic Hypothesis most directly addresses:

| metric | SPLM | Matched GPT-style | Pretrained GPT-2 |
|---|--:|--:|--:|
| median per-layer TEST $R^2$ (token axis) | **+0.508** | +0.114 | +0.216 |
| pooled TRAIN $R^2$ | +0.683 | +0.796 | +0.492 |
| pooled TEST $R^2$ | +0.518 | +0.136 | +0.185 |
| TRAIN$-$TEST gap | 0.16 | **0.66** | 0.31 |
| **median gain over velocity-only** | **+0.27** | **$-$0.08** | **+0.03** |
| Jacobian max gap (full - sym, TEST) | 0.020 | 0.016 | 0.050 |

The separator reproduces in the new coordinate system and *sharpens*
as the gain over velocity-only: only SPLM's token-direction forces
genuinely derive from a shared scalar on held-out data; the attention
baselines' shared-$V_\psi$ fits are either memorisation (Matched,
$-0.08$ held-out gain over velocity-only) or indistinguishable from
velocity persistence (GPT-2, $+0.03$).  Crucially, SPLM was *not*
designed to be conservative along the token axis -- its integrator
is layer-to-layer -- yet it emergently is.  Conservative-by-
construction depth dynamics propagate structure to the orthogonal
time axis.  Details:
[`notebooks/conservative_arch/results/token_direction_summary.md`](../notebooks/conservative_arch/results/token_direction_summary.md)
and [`sharedV_layer_vs_token.png`](../notebooks/conservative_arch/results/sharedV_layer_vs_token.png).

**What this means.** The structural property that the Semantic
Simulation framework prescribes -- all per-layer hidden-state forces
derive from a single shared scalar energy landscape -- is **empirically
achievable** by an actually-trained language model (SPLM) and is
**not** satisfied by any attention transformer we have tested, whether
at matched scale and matched training data (median TEST $R^2 = 0.56$
layer direction, $+0.11$ token direction) or at $\sim 16 \times$ greater
scale with large-corpus pretraining (median TEST $R^2 = 0.45$ layer
direction, $+0.22$ token direction).  The separator is robust to the
choice of dynamical axis.  The §4.1 decision criterion, when applied
at the shared-potential level, gives the answer v2 needs:

> *Yes -- a circuit whose inference dynamics is conservative-by-design
> admits a quantitatively cleaner hidden-state-trajectory fit than any
> attention transformer we have measured, including a scale- and
> data-matched control, and this advantage holds in both coordinate
> systems we examined (depth and token).  The Semantic Simulation
> framework is prescriptive, not merely descriptive.*

The remaining open question -- a SPLM scale sweep (does the +0.90
depth-direction / +0.51 token-direction median hold at larger $d$,
longer training, and larger corpora? does a context-aware
$V_\psi(\xi, h)$ oracle close the token-direction residual?) -- is
listed in [`notebooks/conservative_arch/results/v2_comparison_summary.md`](../notebooks/conservative_arch/results/v2_comparison_summary.md) §6.
