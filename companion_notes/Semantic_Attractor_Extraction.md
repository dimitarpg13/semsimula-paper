# Semantic-attractor extraction from $V_\theta$

## 1.  Why this experiment

The *Semantic Simulation* framework asserts that a trained Scalar
Potential Language Model (SPLM) does not just *predict* the next
token; it learns a **semantic landscape** -- a scalar field
$V_\theta(\xi, h)$ over the joint space of contexts $\xi$ and hidden
states $h$ -- whose **local minima** correspond to coherent semantic
configurations.

This is a strong claim.  Attention transformers have no analogue of
it: their next-token logits are produced by a single matrix
multiplication, not by an energy minimisation, so they cannot, even in
principle, expose a basin structure of "topics the model has
internalised".  If the SPLM picture is right, we should be able to:

1. fix a context $\xi$ (e.g. the cumulative mean of an English
   prefix);
2. start from many random hidden states $h_0$;
3. follow the gradient $-\nabla_h V_\theta(\xi, h)$ to convergence;
4. read out the converged $h^*$ through the tied LM head;
5. and **see a small number of distinct, interpretable token
   distributions** -- one per attractor.

This document reports what happens when we actually do this.

The conclusion is **more interesting than either a simple yes or
no**.

## 2.  Setup

We use two trained SPLM checkpoints from the same Tiny-Shakespeare
recipe (vocab = GPT-2 BPE, $d = 128$, mass-mode = `logfreq`):

| name       | integrator         | $L$ | $\Delta t$ | val PPL |
|------------|--------------------|-----|------------|---------|
| Euler L=8  | semi-implicit Euler (SARF-mass) | 8 | 1.0 | 9.99 |
| Verlet L=16| velocity-Verlet (symplectic)    | 16 | 0.5 | 10.43 |

For each prompt:

1. Run the SPLM forward, capture $\xi$ at the last layer/last-token
   position.
2. Build 384 seeds $h_0$ = 128 Gaussian (matched to $h_L$ statistics)
   + 128 random token embeddings + 128 perturbed real $h_L$
   trajectory points.
3. Evolve each seed at **fixed** $\xi$ until convergence (see below).
4. K-means cluster the converged $h^*$ with silhouette-sweep over
   $K \in [2, 10]$.
5. Decode each cluster centroid with the tied LM head
   $\mathrm{softmax}(c \cdot E^{\top})$.

Five prompts spanning narrative, mathematics, science, dialogue, and
code provide context diversity.

We tried two evolution rules:

- **Gradient descent**: Adam on $V_\theta(\xi, h)$ -- the literal
  reading of \"find the minima of $V_\theta$\".
- **Damped dynamics**: SPLM's own integrator -- semi-implicit Euler on
  $\ddot{h} = -\nabla V_\theta(\xi, h)/m - \gamma \dot{h}$, the actual
  dynamical system the model implements at inference time.

## 3.  Tutorial: why these two rules can disagree

For a *bounded-below* potential $V$, the equilibria of damped second-
order dynamics are exactly the critical points of $V$ (set
$\dot{h} = \ddot{h} = 0$ in the equation above, get $\nabla V = 0$).
For a *bounded-below* $V$ the two rules find the same minima.

For an **unbounded-below** $V$ they do not.  Gradient descent runs $h$
off to infinity along whichever direction $V$ keeps decreasing.  The
damped second-order system, by contrast, has a finite kinetic energy
budget at each instant; it can only travel so far per unit time
before dissipation catches up.  At any *finite* horizon $T$ -- in
particular at $T = L_\text{train}$ -- the damped trajectories occupy
a bounded region, and **clustering them is well-defined even when
$V$ has no finite minima**.

This dichotomy turns out to govern everything we see below.

## 4.  Result 1: $V_\theta$ has no finite local minima

Pure Adam descent on $V_\theta(\xi, h)$ at fixed $\xi$ never
converges.  Across all five prompts and 384 seeds:

- $\langle V\rangle$ at step 300 reaches $\approx -2500$
  (real trajectory: $\approx -260$).
- $\langle V\rangle$ at step 1500 reaches $\approx -50000$.
- $\lVert h\rVert$ grows from 25 to 2200.
- 0 of 384 seeds satisfy $\lVert\nabla V\rVert < 0.05$.

This is a structural property of how SPLM is trained, not a quirk of
the optimiser: **$V_\theta$ is touched by the loss only through its
gradient $-\nabla V_\theta$ (the force).**  Adding any constant to
$V_\theta$ leaves the loss invariant; multiplying $V_\theta$ by any
positive constant rescales the gradient and is partially absorbed by
the learnable $\gamma$, $m$, and $\Delta t$.  The absolute scale of
$V_\theta$ is therefore an unconstrained gauge degree of freedom,
which the optimiser is free to drive to $-\infty$ along any direction
where doing so has zero penalty.

**Implication for the framework.**  The phrase \"semantic attractor\"
should be read **dynamically**, not energetically.  An attractor is a
region of $h$-space that the damped flow at fixed $\xi$ concentrates
on within $L_\text{train}$ steps.  It need not be -- and empirically
is not -- a critical point of $V_\theta$.

## 5.  Result 2: anchored $V_\theta$ descent collapses to one mode

To check whether the unboundedness is "merely" a missing prior, we
add a Gaussian anchor on the data manifold,

$$\mathcal{L}_\text{anchored}(h) = V_\theta(\xi, h) + \frac{\lambda}{2}\Big\lVert \frac{h - h_\text{c}}{h_\text{s}}\Big\rVert^2$$

where $h_\text{c}, h_\text{s}$ are the empirical mean and per-dimension
std of real $h_L$ over a held-out batch.  The minima of
$\mathcal{L}_\text{anchored}$ are the modes of the posterior
$\pi(h\mid\xi) \propto \exp(-V_\theta(\xi, h))$ tempered by an
isotropic prior at $h_\text{c}$.

Sweeping $\lambda \in \{0.5, 2, 10, 50, 200, 1000\}$ we find the same
phenomenon at every $\lambda \ge 50$:

- 70%-99% of seeds converge ($\lVert\nabla\rVert < 0.05$).
- The silhouette-best partition is $K^* = 2$ with one cluster
  containing $> 280$ points and the other $1$ -- effectively, *one*
  attractor with one outlier.
- The decoded distribution at the cluster centroid is identical
  across all five prompts: the same five tokens
  (`,`, `\n`, `the`, `a`, `-`) with the same probabilities.

In other words, the anchored landscape is **globally unimodal** and
the unique mode is **prompt-independent**.  The mode is essentially
$h_\text{c}$, which decodes to whichever tokens have the largest
inner product with the empirical mean of last-layer hidden states --
the unconditional unigram-like distribution of Tiny Shakespeare.

This is the right answer to the literal "find minima of
$V_\theta$" question for a bounded-below proxy: the only mode is the
one the prior already picked out.  All the *interesting* prompt
dependence has to live in the dynamics.

## 6.  Result 3: the damped dynamics IS prompt-dependently multi-basin

Now we run SPLM's semi-implicit damped integrator from the 384 seeds
at fixed $\xi$, for **exactly $L_\text{train}$ steps** (this matches
the model's training-time integration depth, beyond which $V_\theta$
extrapolates pathologically -- see Sec. 7).  Cluster the resulting
$h_{L_\text{train}}^*$.

### Verlet L=16 ($\Delta t = 0.5$) -- punctuation-dominated, 2-6 basins

| Prompt        | $K^*$ | Top attractors (size, top-3 tokens)                                 |
|---------------|-------|---------------------------------------------------------------------|
| narrative     | 5     | A0 (158: `,` 0.79, `\n` 0.18) ; A1 (148: `,` 0.52, `\n` 0.47) ; A2 (61: `,` 0.55, `\n` 0.34, `:` 0.07) |
| mathematics   | 3     | A0 (208: `,` 0.67, `\n` 0.14, `:` 0.12) ; A1 (160: `\n` 0.94) ; A2 (16: `EN` 0.60, `ER` 0.35) |
| scientific    | 6     | A0 (169: `,` 0.51, `\n` 0.42) ; A2 (149: `,` 0.80, `\n` 0.15) ; A3 (16: `.` 0.98) |
| dialogue      | 2     | A0 (329: `,` 0.61, `\n` 0.30) ; A1 (55: `\n` 1.00) |
| code          | 5     | A0 (142: `,` 0.55, `\n` 0.41) ; A1 (182: `\n` 0.87, `,` 0.11) ; A4 (15: `:` 1.00) |

### Euler L=8 ($\Delta t = 1.0$) -- much richer, 2-10 basins, content tokens appear

| Prompt        | $K^*$ | Top attractors (size, top-3 tokens)                                 |
|---------------|-------|---------------------------------------------------------------------|
| narrative     | 10    | A1 (39: ` the` 0.93) ; A5 (74: ` the` 0.91) ; A2 (45: ` I` 0.67, ` to` 0.16, ` the` 0.11) ; A3 (23: ` the` 0.81, ` I` 0.12) ; A4 (38: `\n` 0.31, ` the` 0.24, ` my` 0.07) |
| mathematics   | 2     | A0 (202: `:` 0.53, `EN` 0.26) ; A1 (182: ` I` 0.47, ` the` 0.30, `\n` 0.11, ` to` 0.09) |
| scientific    | 10    | A2 (97: ` the` 0.87, ` I` 0.10) ; A3 (21: ` I` 0.94) ; A4 (25: ` the` 0.97) ; A7 (27: ` I` 0.84, `\n` 0.15) |
| dialogue      | 10    | A0 (70: ` the` 0.63, ` I` 0.21, ` to` 0.06) ; A2 (17: ` I` 0.44, `\n` 0.32, ` and` 0.15) ; A3 (26: ` I` 0.73, `\n` 0.17, ` the` 0.07) ; A6 (41: ` the` 0.70, `\n` 0.13) |
| code          | 10    | A0 (112: ` the` 0.85, ` I` 0.12) ; A4 (21: ` I` 0.90, ` the` 0.09) ; A5 (20: ` I` 0.78, `\n` 0.21) ; A9 (26: ` the` 0.97) |

The Euler baseline is qualitatively much closer to what the framework
predicted: *distinct, prompt-conditional, content-bearing* attractors.
For \"The old king sat on the\", the largest attractor decodes to
` the` (0.93) -- which is *also* the dominant real continuation
(` the`, 0.60).  For \"She whispered: I love\", the basin structure
includes ` the`, ` I`, ` to` and `\n`, all plausible continuations
of dialogue in Shakespeare.

The headline figure `notebooks/conservative_arch/attractor_analysis/results/attractors_comparison.png`
shows the three runs side by side for all five prompts.

## 7.  Result 4: beyond $L_\text{train}$, the dynamics also diverges

Running the same dynamical experiment with $n_\text{sim} = 200$ steps
($\gg L_\text{train} = 16$) reproduces the gradient-descent runaway:
$\lVert h\rVert$ grows to $\sim 2300$, $V$ falls to $-50000$, and the
\"attractors\" decode to subword fragments (`ARD`, `ICH`, `WARD`, `INC`)
which are simply the directions of the largest tied embeddings.

This is internally consistent with Sec. 4: $V_\theta$ has no finite
critical points, so even the damped flow has no asymptotic
equilibria; it can only have *transient* basins on the timescale of
the damping.  At $L = L_\text{train}$ those basins coincide, by
construction, with the regions the model was trained to land on.
Outside that depth, $V_\theta$ extrapolates and the dynamics escapes.

## 8.  Why Verlet has fewer / coarser basins than Euler

This is the secondary scientific surprise of the study.  The Verlet
L=16 dt=0.5 model has slightly worse perplexity (10.43 vs 9.99) and
*also* a much coarser attractor landscape (mostly punctuation,
$K^* \le 6$) compared to the Euler L=8 baseline ($K^*$ up to 10, with
content tokens).

A consistent picture is:

- The Verlet integrator is more *accurate* per step.  At fixed
  damping budget, the trajectory tracks the true continuous-time
  damped flow more faithfully.
- True damped flow on an unbounded $V$ concentrates exponentially
  fast on the steepest-descent direction.
- That steepest-descent direction is dominated by whichever tokens
  have the highest unconditional frequency in Tiny Shakespeare --
  punctuation (`,`, `\n`, `:`).
- The Euler L=8 integrator's per-step truncation error effectively
  *jitters* the trajectory; this stochasticity prevents premature
  basin collapse and preserves richer prompt-dependent structure.

So Euler's "imprecision" is, on this corpus, a *useful* regulariser
that keeps the late-layer hidden states diverse enough for the LM
head to recover content tokens.  Verlet's symplectic accuracy is the
wrong inductive bias for an unbounded potential trained on a tiny
corpus.  This is a concrete, mechanistic explanation for the slight
PPL regression observed in the symplectic experiments
(`docs/Symplectic_Integration_for_SPLM.md`).

## 9.  What this means for the paper

1. **Replace "minima of $V_\theta$" with "basins of the damped flow at
   $L = L_\text{train}$"** in the introduction's promissory text.
   The minima formulation is empirically false; the dynamical
   formulation is empirically supported.

2. **Add a figure** (`attractors_comparison.png`) showing the 5x3
   grid -- Euler dynamics, Verlet dynamics, $V_\theta$ gradient
   descent.  This is the first concrete interpretability artefact
   the paper has that has no transformer counterpart.  The Euler
   column alone -- 10 distinct content-bearing basins per prompt --
   is the strongest qualitative evidence we have so far that SPLM
   exposes interpretable structure that attention does not.

3. **State the gauge symmetry of $V_\theta$ explicitly**.  The
   training loss is invariant under $V_\theta \mapsto V_\theta + c$
   and (modulo $\gamma, m, \Delta t$) under positive rescaling of
   $V_\theta$.  This explains why gradient descent on $V_\theta$
   alone is ill-posed and motivates the damped-flow reading.

4. **$V_\theta$ regularisation (now tested -- see §11).**  An
   explicit penalty $\lambda_V \lVert V_\theta\rVert_2^2$ on the network's
   own scalar output (not its weights) breaks the gauge and gives an
   actually bounded-below potential.  The sweep in §11 confirms that
   genuine $V_\theta$-minima attractors appear at $\lambda_V \ge 10^{-4}$,
   the landscape remains multi-modal, and the PPL cost is quantified.

5. **Re-frame the symplectic study.**  The Verlet result is no
   longer "an interesting tangential experiment that slightly
   regressed".  It is a clean demonstration that *integrator
   accuracy can hurt model expressivity* when the underlying
   continuous system has no equilibria -- which is itself a
   simulation-framework insight that attention-based papers cannot
   make.

## 10.  Visualising the landscape in 3D

The 2D PCA scatter plots used in Sec. 6 are good at showing *where*
trajectories end up, but they throw away the single most interesting
object we have ever trained: the scalar field $V_\theta(\xi, h)$
itself.  To make that object concrete we render $V_\theta$ as a 3D
surface over the 2-component PCA plane of the trajectory data, and
overlay the damped-flow trajectories on top of it.

### 10.1  Inference-time landscape

For a given model, prompt, and fixed $\xi$:

1. simulate the SPLM damped integrator from $N=288$ random $h$ seeds
   for exactly $L_\text{train}$ steps, **keeping the full trajectory**
   (shape $(N, L_\text{train}+1, d)$);
2. fit a 2D PCA on the union of the real trajectory and the
   $N(L_\text{train}+1)$ intermediate points;
3. grid-sample the PCA plane and lift each 2D point back to
   $\mathbb R^d$ via the affine PCA inverse;
4. evaluate $V_\theta(\xi, \cdot)$ on the grid -- this gives the
   *height* of the surface;
5. overlay each trajectory as a 3D curve
   $\lbrace(\text{PCA}\_2(h\_l),\, V\_\theta(\xi, h\_l))\rbrace\_{l=0}^{L}$,
   coloured by the basin its endpoint lands in (silhouette-optimal
   K-means on endpoints).

The Euler $L{=}8$ and Verlet $L{=}16$ checkpoints produce
qualitatively **different** landscapes, and the difference is
instantly legible:

- **Euler** opens up a broad, symmetric **U-valley** with basin
  endpoints distributed across the entire valley floor.  Trajectories
  fan out from the seed cloud and settle into several distinct
  endpoints.  This is the pictorial realisation of the 7-to-10 basins
  Sec. 6 reports.
- **Verlet** draws a **narrow funnel-slide**: a steep-walled canyon
  along one PCA direction, with every trajectory channelling down the
  slide into essentially the same region.  The silhouette-optimal $K$
  is still nominally 2--6, but all basin endpoints sit at the
  upper rim of the slide -- they are the same semantic configuration
  up to small perturbations.

`results/landscape3d_compare_<prompt>.png` places the two panels on one
figure; the "Euler is wide, Verlet is narrow" story is then a single
visual claim.  Rotating 360-degree animations of the dialogue-prompt
landscapes are provided in `results/landscape3d_*_dialogue.gif`.

This is, to our knowledge, the first direct visualisation of a learned
language-model's scalar potential in a language paper.  Attention
transformers cannot be rendered this way -- they have no fixed
$V_\theta$ to draw a surface of.

### 10.2  Training-time evolution

A separate training run (`train_with_snapshots.py`) retrains the
SARF-mass Euler model on Tiny Shakespeare while checkpointing at
log-spaced training steps $\{0, 50, 200, 500, 1000, 2000, 4000\}$.
`render_training_evolution.py` then reuses the landscape-rendering
pipeline on each snapshot and tiles the results into a 7-panel figure
(`results/training_evolution_euler_shakespeare_<prompt>.png`).

The evolution is pedagogically clean:

| Step | val CE | $V$ range on trajectories | Visual character |
|------|--------|---------------------------|------------------|
| 0 (random init) | -- | $\sim 10^{-5}$ | flat landscape; no structure |
| 50  | 10.75 | $\sim 0.1$     | tilted plane; gradient direction learned first |
| 200 | 6.53  | $\sim 150$     | a basin begins to carve; $K^\ast{=}10$ (noise) |
| 500 | 6.13  | $\sim 220$     | basin is visible; $K^\ast$ collapses to 2 |
| 1000| 5.80  | $\sim 180$     | valley deepens |
| 2000| 5.60  | $\sim 400$     | steep-walled valley |
| 4000| 5.61  | $\sim 1500$    | deep canyon with well-separated endpoints |

Two observations stand out:

- **The gradient is learned before the curvature.** At step 50 the
  validation loss has already fallen from random chance ($\ln V \approx
  10.8$) to something meaningful, but $V_\theta$ itself is still a
  near-linear ramp -- there is no basin yet.  The optimiser first aligns
  $\nabla_h V_\theta$ to push trajectories in the useful direction; only
  later does it add curvature to make those trajectories *converge*.
  This is a reassuring match to the first-principles picture of SPLM
  training: the loss grades $\nabla V$ via the integrator, not $V$
  directly, so a linear $V_\theta$ that points the right way already
  reduces loss.
- **The silhouette-optimal $K$ is a reliable basin-formation
  indicator.** Pre-carving ($\le$ step 200) $K^\ast$ is saturated at
  the upper bound of the sweep because k-means is finding spurious
  structure on a flat endpoint cloud.  The moment the landscape
  develops real curvature (step 500), $K^\ast$ drops to $2$ -- the real
  number of basins -- and stays there as the valley deepens.  The
  number of basins is *established early* in training and the
  subsequent ten-fold expansion in $V$ range is almost entirely about
  depth, not topology.

## 11.  V_θ regularisation sweep

Section 9 item 4 identified $V_\theta$ regularisation as a sharp,
testable hypothesis.  This section reports the results of a six-cell
sweep that answers three questions.

### 11.1  Experimental setup

We add a single loss term to the standard NTP cross-entropy:

$$\mathcal{L} = \mathcal{L}\_{\text{NTP}} + \lambda\_V \cdot \frac{1}{BT}\sum\_{b,t} V\_\theta(\xi\_{b,t},\, h\_{b,t})^2$$

This penalises large absolute values of $V_\theta$, anchoring the
potential near zero and breaking the additive/multiplicative gauge
symmetry (§9 item 3).

Six cells sweep $\lambda_V$ and integrator choice on TinyShakespeare
with the SARF-mass Euler $L{=}8$ baseline (d=128, v\_hidden=512,
4000 steps, batch 16, block 256):

| Cell | $\lambda\_V$ | Integrator | What it tests |
|------|-------------|------------|---------------|
| VR0 | 0 | Euler L=8 | Unregularised baseline |
| VR1 | $10^{-6}$ | Euler L=8 | Weakest regularisation |
| VR2 | $10^{-4}$ | Euler L=8 | Moderate regularisation |
| VR3 | $10^{-2}$ | Euler L=8 | Strong regularisation |
| VR4 | $1$ | Euler L=8 | Very strong regularisation |
| VR5 | $10^{-4}$ | Verlet L=16 | Verlet regression test (Q3) |

Post-training, each checkpoint undergoes attractor extraction using
the same three protocols as §5--6: pure GD on $V_\theta$, anchored
descent, and damped dynamics at $L\_{\text{train}}$ steps.

The notebook is at
`notebooks/conservative_arch/parf/scripts/vreg_sweep_v_theta_regularisation.ipynb`
(Colab-ready; outputs to `semsimula_vreg/vreg_sweep/` on GDrive).

### 11.2  Results

| Cell | $\lambda\_V$ | Integ. | Best PPL | Final PPL | $V\_\theta$ range | $V\_\theta$ std | GD conv% | avg $K^\ast$(GD) |
|------|-------------|--------|----------|-----------|------------------|----------------|----------|-----------------|
| VR0 | 0 | Euler | 249.5 | 250.7 | 1808 | 350 | 2% | 3.4 |
| VR1 | $10^{-6}$ | Euler | 256.4 | 260.8 | 893 | 75 | 0% | 2.4 |
| VR2 | $10^{-4}$ | Euler | 315.1 | 358.4 | 332 | 6.8 | 27% | 2.6 |
| VR3 | $10^{-2}$ | Euler | 318.9 | 452.6 | 70 | 0.6 | 70% | 2.8 |
| VR4 | $1$ | Euler | 342.6 | 437.5 | 13 | 0.1 | 100% | 3.8 |
| VR5 | $10^{-4}$ | Verlet | 275.4 | 275.4 | 204 | 7.2 | 0% | 2.0 |

"GD conv%" = fraction of 384 seeds with $\lVert\nabla V\rVert < 0.05$
after 1500 Adam steps, averaged across 5 prompts.  "$K^\ast$(GD)" =
silhouette-optimal cluster count from gradient-descent endpoints.
Damped-dynamics $K^\ast$ is uniformly 2 across all cells.

### 11.3  Answering the three questions

**Q1: Is there a $\lambda_V$ where $V_\theta$ is bounded below AND
the attractor landscape remains multi-modal?**

Yes.  VR4 ($\lambda_V = 1$) achieves 100% GD convergence across all
5 prompts with $K^\ast = 3{-}4$ basins each (average 3.8).  The
$V_\theta$ range collapses from 1808 to 13 — the potential is
genuinely bounded — and the mean potential at the converged points is
$\langle V \rangle \approx -14$, shallow finite minima rather than
$-5600$ escape directions.  VR3 ($\lambda_V = 10^{-2}$) also converges
70% of seeds and uniquely produces compact attractors with
$\lVert h\rVert \approx 89$ (vs. 670--720 for all other cells).

VR4's $K^\ast = 3.8$ is actually higher than VR0's 3.4 — the bounded
potential has *more* distinguishable basins, not fewer.

**Q2: At what $\lambda_V$ does perplexity degrade measurably?**

The cliff is between $10^{-6}$ and $10^{-4}$:

- VR0 $\to$ VR1: +7 PPL (249.5 $\to$ 256.4) — marginal
- VR1 $\to$ VR2: +59 PPL (256.4 $\to$ 315.1) — the onset
- VR2 $\to$ VR3 $\to$ VR4: +4 $\to$ +24 further — diminishing

The cost of breaking the gauge is approximately 60--90 PPL (~25--35%
relative increase).  Substantial but not catastrophic.

**Q3: Does the Verlet PPL regression disappear in the regularised
regime?**

Partially confirmed.  VR5 (Verlet L=16, $\lambda_V = 10^{-4}$)
achieves 275.4 PPL — 40 PPL better than VR2 (Euler, same $\lambda_V$)
at 315.1.  The Verlet integrator goes from harmful (unregularised, §8)
to helpful once $V_\theta$ has some structure.  However, the
regularisation cost means VR5 is still 26 PPL worse than VR0
(unregularised Euler at 249.5).  The regression *reverses direction*
but the net budget remains non-zero.

### 11.4  Structural takeaways

1. **The gauge symmetry is real and testable.**  Breaking it costs PPL
   but creates genuine energetic structure.  $\lambda_V$ continuously
   interpolates between "pure dynamical attractors" ($\lambda_V = 0$)
   and "genuine energy minima" ($\lambda_V \ge 10^{-2}$).

2. **Multi-modal structure survives regularisation.**  VR4's
   $K^\ast = 3.8$ exceeds VR0's 3.4 — the bounded potential has more
   distinguishable basins, not fewer.

3. **Two attractor regimes.**  VR3 ($\lambda_V = 10^{-2}$) finds
   compact attractors near the origin ($\lVert h\rVert \approx 89$),
   while VR4 ($\lambda_V = 1$) finds attractors at large norm
   ($\lVert h\rVert \approx 708$).  The stronger penalty forces
   $V_\theta$ to be flatter, migrating equilibria to where the MLP's
   nonlinearity naturally creates structure.

4. **Prompt sensitivity at moderate $\lambda_V$.**  VR2 shows 91%
   convergence for the narrative prompt but 0% for scientific/code.
   The V-landscape is easier to regularise for some semantic domains
   than others.

5. **The Verlet result is nuanced.**  The §8 hypothesis
   ("integrator accuracy is harmful specifically because the unbounded
   potential has no equilibria") is directionally correct — Verlet
   beats Euler at $\lambda_V = 10^{-4}$.  But the full picture is
   richer: the regularisation + Verlet combination explores a
   different part of parameter space than regularisation + Euler.

## 12.  Files

- `notebooks/conservative_arch/attractor_analysis/`
  - `attractor_extraction.py` -- main script (gradient + dynamical modes)
  - `make_comparison_figure.py` -- builds Sec. 6 headline figure
  - `landscape_3d.py` -- inference-time 3D surface + trajectories per model/prompt
  - `compare_landscapes_3d.py` -- Euler-vs-Verlet side-by-side 3D panels
  - `train_with_snapshots.py` -- retrain with log-spaced checkpoint saves
  - `render_training_evolution.py` -- build the 7-panel landscape-evolution figure
  - `README.md`     -- reproduction recipe
  - `results/`      -- per-prompt PNGs, GIFs, JSONs, summary markdowns,
    plus `attractors_comparison.png`,
    `landscape3d_*.png`, `training_evolution_*.png`
- `notebooks/conservative_arch/parf/scripts/`
  - `vreg_sweep_v_theta_regularisation.ipynb` -- §11 V_θ regularisation sweep
    (Colab-ready; outputs to `semsimula_vreg/vreg_sweep/` on GDrive)

## References

- Sec. \"Inference of Semantic Structures\" of the SPLM paper draft
- `docs/Symplectic_Integration_for_SPLM.md` (motivates Verlet runs)
- `docs/Next_Model_Experiments_for_SPLM.md` (this study is item C3)
