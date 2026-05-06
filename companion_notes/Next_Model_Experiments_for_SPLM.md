# Next Model Experiments for SPLM

A concrete, prioritised catalogue of experiments that would either
strengthen the Semantic Simulation framework's applicability or
measurably improve SPLM's performance.  Each item is actionable (what
to run, what to measure, why it matters) rather than abstract.

The starting point is the current state of the companion repo:

- Tiny Shakespeare, $d=128$, $L=8$, Euler integrator.
- Best variant: `sarf_mass_variant` with `logfreq` per-token mass —
  val ppl **160.55**.
- Integrator ablation ([Symplectic_Integration_for_SPLM.md](Symplectic_Integration_for_SPLM.md)):
  Euler wins on ppl; Verlet wins on the token-axis shared-$V_\psi$
  diagnostic at every setting.
- Two structural diagnostics running in `notebooks/conservative_arch/`:
  depth-axis and token-axis shared-$V_\psi$ fits.

The goal of the experiments below is to move beyond "novel framework,
modest Shakespeare results" to **"novel framework with at least one
measurable advantage over attention"** and to provide the paper with
independent diagnostic axes that attention models demonstrably lack.

---

## 0. Multi-seed variance — the variance bar all subsequent items inherit

**Motivation.** The paper currently reports a single seed on Tiny
Shakespeare (`sarf_mass_variant logfreq` → val ppl $160.55$). The TMLR
review summary
([tmlr_review_summary_chatgpt.md](tmlr_review_summary_chatgpt.md))
correctly flags this as underpowered: there is no $\sigma$ on the
headline number, no significance test against the matched baseline,
and no reproducibility evidence at all. Every quality experiment in
sections A, B, D, F below inherits this problem until a variance bar
is established for the *current* model.

**What to run.** A non-invasive multi-seed harness lives in
[`notebooks/conservative_arch/multi_seed/`](../notebooks/conservative_arch/multi_seed/):

- `multi_seed_runner.py` — `subprocess`-driven launcher that runs the
  existing `train_splm.py` / `train_splm_sarf_mass.py` /
  `train_matched.py` trainers **unmodified** with `--seed s`, then
  relocates the per-seed artefacts
  (`*_training_log.jsonl`, `*_ckpt_latest.pt`, `*_loss_curve.png`,
  `*_summary.md`) into seed-namespaced subdirectories
  `results/<tag>/<model_label>/seed_<s>/`. This pattern lets us add
  multi-seed coverage to any future trainer without touching upstream
  code.
- `multi_seed_aggregator.py` — parses the per-seed JSONL logs, computes
  mean / std / min / max of final val ppl and val loss, runs Welch's
  $t$-test for the SPLM-vs-baseline mean difference with $95\%$ CI,
  plots overlay loss-curves, and emits a markdown report.

**Round-1 production run** (currently executing, ~5–5.5 h MPS time
for the first phase): $N=5$ seeds for each of `sarf_mass_variant
logfreq` (paper's currently-stated headline, val ppl 160.55 at seed=0)
and `matched_baseline` on Tiny Shakespeare, $4000$ steps each.

A second phase will then run $N=5$ seeds of the
`energetic_minima --variant ln` SPLM (LayerNorm-after-step, val ppl
**88.63** at seed=0 -- the strongest result in the repo) on the same
schedule (~2.5 h additional MPS time).  All three models land in the
same E1 report at
`notebooks/conservative_arch/multi_seed/results/E1_shakespeare/E1_report.md`,
giving a multi-seed comparison of LN-after-step vs SARF+logfreq vs
matched baseline.

**Architectural relationship of the two SPLM variants.**  LN-after-step
is *not* an alternative architecture to SARF+logfreq: it is the same
model with one additional projection step.  Concretely
(`notebooks/conservative_arch/energetic_minima/model_ln.py`),
`SPLMSARFMassLNConfig` extends `SPLMSARFMassConfig`,
`ScalarPotentialLMSARFMassLN` extends `ScalarPotentialLMSARFMass`, the
trainer hard-codes `mass_mode="logfreq"`, and the integration loop is
the SARF+mass loop with one extra line `h_{l+1} ← LayerNorm(h_{l+1})`
after each damped step.  The framework's prescriptions
(causal-cumulative-mean $\xi$, SARF re-pooling, logfreq per-token mass,
free MLP $V_\theta$, damped Euler-Lagrange flow) are unchanged.  Phase
1 vs phase 2 is therefore a clean one-switch ablation -- it isolates
the effect of the unit-shell projection alone.

The corrected ablation chain at seed $=0$ on Tiny Shakespeare,
$d=128,\ L=8$:

| variant                                  | val ppl | what was added |
|---|---:|---|
| §1 fixed-$\xi$ SPLM                       | 287.4   | (baseline)            |
| $\quad+$ SARF re-pooling                  | 192.2   | causal cumulative $\xi$ |
| $\quad+$ logfreq per-token mass           | **160.6** | (paper §14.13 flagship) |
| $\quad+$ LayerNorm-after-step             | **88.63** | (paper §14.16 F5(i)) |
| Matched GPT-2 transformer (8 M params)   | 141.7   | (different model)     |

**Paper-position consequence (flagged for triage after E1 completes).**
The paper currently presents the SARF+logfreq variant (160.6) as the
flagship in §14.13 and the LN-after-step variant (88.63) as a
falsification side-finding in §14.16(F5(i)), with the explicit hedge
*"a future flagship could productively move toward LN-after-step."*
The paper's introduction and §14 figure caption further state, of the
matched baseline, *"the matched attention baseline is a better LM than
SPLM by validation perplexity --- a deliberate choice."*

That framing is **already empirically inconsistent** with the paper's
own §14.16(F5(i)) number ($88.63 < 141.7$), but the paper is currently
written as if it were not.  E1's phase 2 multi-seed measurement of
LN-after-step is therefore not just a hardening of the F5(i) claim:
its outcome decides whether the paper's current ppl-superiority hedge
needs to be removed at the next revision.  See the *"Conditional paper
edits"* paragraph at the end of §G's Round-1 execution plan for the
specific edits this would entail.

**Expected outcome.** A first defensible "$\mathrm{ppl} \pm \sigma$"
number for SPLM, the matched baseline, and the gap between them. This
turns the headline claim from a single point estimate into a properly
reported measurement.

**Standing requirement going forward.** Every quality, efficiency, and
domain-transfer item below (sections A, B, D, F) should be reported
with $N \geq 3$ seeds and an explicit $\sigma$ (or $95\%$ CI) on every
headline number. The pure-eval diagnostics (sections C, E) inherit
this only to the extent that they consume training checkpoints — but
because (e.g.) the energy-drift extractor (C2) operates on a fixed
checkpoint, *its* variance is exactly the cross-seed variance of the
upstream training run.

---

## A. LM quality — scaling axes that are untested

Tiny Shakespeare at $d=128,\ L=8$ is a single data point.  The paper's
quality numbers are essentially unknown outside this regime.

### A1. Width sweep at fixed $L$: $d \in \{128, 256, 512\}$

**Motivation.** Attention-transformer FLOP scaling is $O(T d^{2})$;
SPLM's is $O(T dd_V)$.  The FLOP gap **grows with $d$**, so the
crossover is at larger $d$, not smaller.  Shakespeare at $d=128$ is
arguably the worst regime for SPLM relative to attention.

**What to run.** Retrain `sarf_mass_variant` at $d=256$ and $d=512$
on TinyStories (or a subset of OpenWebText), matched params + FLOPs
against a transformer baseline.  Report the ppl gap as a function of
$d$.

**Expected outcome.** The gap closes or inverts as $d$ grows — the
paper's efficiency story becomes more credible.

### A2. Depth sweep at fixed width: $L \in \{4, 8, 12, 16, 24\}$

**Motivation.** Verlet at $L=4$ collapsed to ppl 280; at $L=8$ gives
160; we do not know what the plateau looks like.

**What to run.** Sweep $L$ with the Euler integrator (cheapest),
$d=128$, Tiny Shakespeare.

**Expected outcome.** Monotone improvement plateauing around
$L \approx 10$–$12$; provides a principled lower bound on "flow
distance" to put in the paper.

### A3. $V_\theta$ capacity sweep

**Motivation.** $V_\theta$ is a 3-layer MLP with hidden 512.  This is
the entire non-embedding parameter budget — it is plausibly the
bottleneck.

**What to run.** Sweep $v\_\text{hidden} \in \{256, 512, 1024, 2048\}$
and $v\_\text{depth} \in \{2, 3, 4, 6\}$ at $d=256$.

**Expected outcome.** Perplexity curve as a function of non-embedding
params; directly addresses "how much of the $160.55$ is architecture
vs tuning".

### A4. Non-constant $\gamma$ and $\mathfrak m$

**Motivation.** $\gamma$ is a single learnable scalar; $\mathfrak m$
only varies by token.  Per-layer or per-$(token, layer)$ parameters
would let the model place "fast layers" early and "slow layers" late
— a known useful inductive bias in neural-ODE literature.

**What to run.** Two small ablations: (a) $\gamma_\ell$ per layer;
(b) $\mathfrak m(x_t, \ell)$ layer-conditioned.

**Expected outcome.** Modest perplexity gains (3–8 %) and a richer
interpretability story.

---

## B. Efficiency — turn the paper's FLOP story into a measured win

Appendix B has an analytical FLOP comparison.  The paper would be
significantly strengthened by matching wall-clock numbers in at least
one regime where SPLM wins.

### B1. KV-cache-free autoregressive decoding benchmark

**Motivation.** SPLM has no KV cache by construction.  At long context
this is a real throughput win that is invisible in training-loss
metrics.

**What to run.** Measure tokens/s and peak memory for SPLM vs a
FLOP-matched transformer at context lengths
$T \in \{512, 1024, 2048, 4096\}$ on an A100 or MPS.  Report a
throughput-vs-$T$ curve.

**Expected outcome.** SPLM crosses over the transformer around
$T \approx 512$–$1024$ and is dramatically better at $T = 4096$.
This is the headline efficiency chart the paper currently lacks.

### B2. Long-context language modelling

**Motivation.** Transformer long-context is expensive in both FLOPs
*and* memory; SPLM should be competitive on ppl at a fraction of the
cost.

**What to run.** Train SPLM and a FLOP-matched transformer on PG19 or
ProofPile at $T=2048$.  Compare ppl on held-out long documents.

**Expected outcome.** SPLM within a small margin on ppl, large margin
on wall-clock / memory.

### B3. Adaptive-$\Delta t$ inference

**Motivation.** Today's experiments showed that most of the "flow
distance" $L\Delta t$ matters; the discretization granularity
matters less for LM quality.  One can **train at
$L=8,\ \Delta t=1$** and **infer at $L=4,\ \Delta t=2$** or
**$L=16,\ \Delta t=0.5$** without retraining, trading compute vs
trajectory fidelity at deploy time.

**What to run.** Measure ppl degradation as a function of
$\Delta t$-change between train and infer, with and without refitting
$\gamma$.

**Expected outcome.** A knob users can turn — cheap inference becomes
a deployment-time choice.

### B4. Early-exit inference

**Motivation.** If $V_\theta$ has converged to a semantic attractor by
layer $\ell^\* < L$, later layers are redundant.

**What to run.** Add a lightweight convergence criterion
($\lVertv_\ell\rVert$ or $|V_\theta(\xi_\ell, h_\ell) - V_\theta(\xi_{\ell-1},
h_{\ell-1})|$) and sample-adaptively stop.  Measure average
layers-per-token on held-out data and the ppl cost.

**Expected outcome.** A further $\sim 1.5\times$ throughput win at
negligible ppl cost, reinforcing the "physics-inspired inference"
narrative.

---

## C. Diagnostic and interpretability — tighten the framework's identification

The symplectic-integration study
([Symplectic_Integration_for_SPLM.md](Symplectic_Integration_for_SPLM.md))
showed the depth-axis shared-$V_\psi$ fit has a coarse-$\Delta t$
ansatz bias.  That is a methodological bug in the paper that we now
know how to fix; several parallel upgrades are natural.

### C1. Two-point (integrator-matched) shared-$V_\psi$ ansatz

**Motivation.** The current ansatz
$\Delta h = \alpha v - \beta\nabla V_\psi(h_\ell)$ rewards
Euler-style one-step updates.  A Verlet-matched ansatz
$\Delta h = \alpha v - \tfrac{1}{2}\beta
\bigl[\nabla V_\psi(h_\ell) + \nabla V_\psi(h_{\ell+1})\bigr]$
would remove the bias.

**What to run.** Drop into `shared_potential_fit.py`; re-run on
Euler, Verlet, and GPT-2 trajectories.

**Expected outcome.** Verlet's depth-axis $R^2$ catches up to or
passes Euler's at coarse $\Delta t$.  The paper can then report an
unbiased diagnostic.

### C2. Energy-drift diagnostic

**Motivation.** The direct test of "is the flow Hamiltonian?" is to
compute the total energy
$H_\ell = \tfrac{1}{2}\mathfrak m\|v_\ell\|^{2}
+ V_\theta(\xi_\ell, h_\ell)$
across depth and check for drift vs bounded oscillation.

**What to run.** Plot $H_\ell$ vs $\ell$ for Euler, Verlet $L=8$,
Verlet $L=16,\Delta t=0.5$, and GPT-2.  Fit a linear drift and report
slope-per-layer and oscillation bandwidth.

**Expected outcome.** Verlet $L=16,\Delta t=0.5$ shows bounded
oscillation; Euler shows a systematic drift; GPT-2 shows no
structure.  This is a new, publishable diagnostic.

**Status (Round-1, implementation complete, run queued).**
Implementation lives in
[`notebooks/conservative_arch/energy_drift/`](../notebooks/conservative_arch/energy_drift/):

- `extract_energy_states.py` — re-implements the integration
  loops of the three SPLM variants (parent Euler,
  `sarf_mass_variant`, `symplectic_variant`) inside dedicated
  adapter classes that capture full per-layer state
  $(h_\ell, v_\ell, \xi_\ell, V_\theta(\xi_\ell, h_\ell))$.
  This re-implementation is necessary because the existing
  `trajectory_extraction.py` only stores positions $h_\ell$, not
  velocities $v_\ell$, and kinetic energy is unrecoverable from
  positions at a single layer.  The adapter pattern is
  non-invasive: upstream model code remains untouched.
- `energy_drift_diagnostic.py` — loads the `.npz` artefacts
  produced by the extractor, fits OLS drift slopes per variant
  with $t$-distribution $95\%$ CI, computes detrended oscillation
  bandwidth, and produces overlay plots of $H_\ell$,
  $\tfrac{1}{2}\mathfrak m\lVertv_\ell\rVert^{2}$, and
  $V_\theta(\xi_\ell, h_\ell)$ vs normalised layer index.

End-to-end CPU smoke validated on idle checkpoints.  Production
run on the existing trained checkpoints (Euler $L=8$, `sarfmass
logfreq` $L=8$, Verlet $L=16,\Delta t=0.5$) is queued behind the
multi-seed production run (Section 0) to avoid MPS contention.
Expected production compute: $\sim 1$ minute of MPS time.

### C3. Semantic-attractor extraction from $V_\theta$  *(COMPLETED)*

**Motivation.** The paper's "semantic simulation" story predicts that
trained $V_\theta$ should have isolated local minima corresponding to
meaningful semantic configurations.  This is empirically untested.

**What to run.** From a trained SPLM, run gradient descent on
$V_\theta(\xi, \cdot)$ at fixed $\xi$ for many random initializations.
Cluster the converged points; decode them with the LM head; inspect
whether they correspond to coherent token distributions / topics.

**Outcome (see `docs/Semantic_Attractor_Extraction.md`).**
Implemented in `notebooks/conservative_arch/attractor_analysis/`.
Three sub-results:

1. Pure $V_\theta$ gradient descent diverges — $V_\theta$ has no
   finite local minima because its absolute scale is an unconstrained
   gauge degree of freedom (the loss only sees $-\nabla V_\theta$).
2. *Anchored descent is unimodal and prompt-independent* — the
   posterior $\propto \exp(-V_\theta - \tfrac{\lambda}{2}\lVert(h-h_c)/h_s\rVert^2)$
   collapses to one mode that decodes to the unconditional unigram
   over Tiny Shakespeare for every prompt.
3. *The damped flow at $L = L_\text{train}$ does have prompt-dependent
   multi-basin structure* — Euler L=8 produces up to $K^*=10$ distinct
   basins per prompt, decoded to real content tokens (` the`, ` I`,
   ` to`, `\n`, `:`).  This is the strongest qualitative
   interpretability artefact SPLM has so far that attention
   transformers have no analogue of.

A side-finding: the symplectic Verlet model has *coarser* attractors
than Euler (mostly punctuation, $K^*$ up to 6), giving a mechanistic
explanation for its slight PPL regression — the more accurate
integrator concentrates probability mass too aggressively on the
unconditional-frequency directions.

Headline figure: `notebooks/conservative_arch/attractor_analysis/results/attractors_comparison.png`.

**3D visualisation follow-up** (also completed).  To make the learned
$V_\theta$ directly visible, the landscape is rendered as a 3D surface
over the 2D PCA plane of the trajectory data, with damped-flow
trajectories overlaid and colour-coded by endpoint basin:

- `results/landscape3d_euler_L8_<prompt>.png` and
  `results/landscape3d_verlet_L16_dt05_<prompt>.png` -- per-model 3D
  landscapes, 3 camera views each.
- `results/landscape3d_compare_<prompt>.png` -- Euler vs Verlet side
  by side on one figure.  Makes the "wide U-valley vs narrow funnel"
  story of the two integrators a single readable claim.
- `results/landscape3d_*_dialogue.gif` -- 360-degree rotating
  animations.
- `results/training_evolution_euler_shakespeare_<prompt>.png` --
  7-panel figure showing $V_\theta$ forming from flat (step 0, $V$
  range $\sim 10^{-5}$) to a deep canyon (step 4000, $V$ range $\sim
  10^3$). $K^*$ drops from 10 (noise) to 2 (real basins) at step 500,
  showing that basin topology is established early; subsequent
  training deepens the wells, it does not create new ones.

Details and rendering scripts in `docs/Semantic_Attractor_Extraction.md`
Sec. 10 and `notebooks/conservative_arch/attractor_analysis/`
(`landscape_3d.py`, `compare_landscapes_3d.py`, `train_with_snapshots.py`,
`render_training_evolution.py`).

### C4. Cross-variant $V_\psi$ transfer

**Motivation.** If the framework is right, $V_\psi$ fitted on SPLM
Euler trajectories should predict SPLM Verlet trajectories reasonably
well, and vice versa, but both should fail to predict GPT-2
trajectories.

**What to run.** Fit $V_\psi$ on one variant, evaluate fit $R^2$ on
others.

**Expected outcome.** SPLM ↔ SPLM transfer $\geq 0.7$ pooled TEST,
SPLM ↔ GPT-2 transfer $\leq 0.3$.  Provides a stronger separator than
the within-model $R^2$ numbers alone.

---

## D. Framework applicability — domains beyond Tiny Shakespeare

The Semantic Simulation framework claims to be a *general* theory.
Currently it is instantiated on one corpus of one domain.

### D1. TinyStories + WikiText-103 + code (the llm.c triplet)

**What to run.** Train SPLM at $d=256,\ L=12$ on each corpus in turn.
Publish the ppl/FLOP Pareto curve against a matched transformer.

**Expected outcome.** Three data points on the quality/efficiency
plot instead of one.

### D2. Domain-transfer perplexity

**Motivation.** Train SPLM on one domain, evaluate on another.  Does
the learned $V_\theta$ encode domain-specific semantics or
general-purpose semantics?

**What to run.** Cross-domain ppl matrix on the 5-domain corpus used
in the diagnostics.

**Expected outcome.** Establishes whether $V_\theta$ captures
universal or domain-local structure — important for the framework's
claimed generality.

### D3. In-context learning test

**Motivation.** SPLM should in principle support in-context learning
if $V_\theta$ depends on the context-aggregated $\xi$.  This is the
core of the "semantic simulation" picture but has not been
demonstrated.

**What to run.** LAMBADA or a simple ICL induction-head task at
$d \geq 256$.

**Expected outcome.** Even a weak positive result here is publishable
on its own — it would show the framework scales to behaviour
attention models are known for.

---

## E. Continuous-time and higher-order integrators (follow-on to today's study)

### E1. Neural-ODE variant

**Motivation.** The $L \to \infty,\ L\Delta t$ fixed limit is a
continuous-time neural ODE.  Using an adaptive ODE solver
(Dormand-Prince, or the `torchdiffeq` family) eliminates the
discretization bias entirely.

**What to run.** Replace the discrete Verlet loop with `odeint`;
compare ppl and diagnostic $R^2$ against the best Verlet variant.

**Expected outcome.** Diagnostics lift further; ppl may or may not
(depends on stiffness); either outcome is informative.

### E2. Higher-order symplectic: Forest-Ruth / Yoshida-4

**Motivation.** If the token-axis Verlet win really comes from
symplectic structure, a 4th-order symplectic integrator should push
it further.

**What to run.** Implement Forest-Ruth ($O(\Delta t^{4})$, 3 force
evaluations per step); retrain at $L=4$ (12 force evals, matched to
$L=16$ Verlet cost).

**Expected outcome.** Highest token-axis $R^2$ we've seen; possibly
matches or beats Euler on ppl because the discretization error is
now tiny.

### E3. Stochastic Langevin variant

**Motivation.** Adding noise $\mathrm{d}W$ to the velocity update
turns the flow into a Langevin SDE, which equilibrates to
$\pi(h) \propto e^{-V_\theta(h)/k_B T}$ — the paper's "thermal" /
semantic-sampling picture.

**What to run.** Train with a small noise amplitude; at inference,
run multiple noisy trajectories and ensemble the outputs.

**Expected outcome.** Improved calibration and sample diversity;
directly realizes the paper's semantic-sampling discussion as a
working inference procedure.

---

## F. Positioning against attention (comparison studies)

### F1. Hybrid SPLM + attention architecture

**Motivation.** Real systems rarely replace components wholesale.  A
layer stack of "$k$ attention blocks + $L$ SPLM layers" is the
natural first deployed variant.

**What to run.** $k=2$ attention blocks feeding an $L=8$ SPLM body.
Compare against pure-attention and pure-SPLM at matched parameters.

**Expected outcome.** Hybrid beats both at some $k^\*$; provides a
concrete "how would you deploy this?" answer for the paper's
discussion.

### F2. SPLM as a tail replacement

**Motivation.** Use a pretrained GPT-2 / Pythia model, chop off its
top $k$ layers, replace with an SPLM tail, and fine-tune only the
SPLM block.

**What to run.** Start with GPT-2 small, replace last 2–4 transformer
blocks with SPLM, fine-tune on C4 subset.

**Expected outcome.** Small ppl degradation, meaningful
inference-cost reduction — a practitioner-relevant result.

---

## G. Suggested prioritisation

For the **paper v2**, the two highest-leverage items are probably
**B1 (KV-cache-free decoding benchmark)** and **C2 (energy-drift
diagnostic)** — one closes the efficiency claim, one gives the
framework a new first-class diagnostic that attention models
demonstrably lack.  Adding either would meaningfully shift the
paper's profile from "novel framework, modest Shakespeare results"
to "novel framework with at least one measurable advantage over
attention".

For the **framework's long-term credibility**, **C3
(attractor extraction)** and **E1 (neural-ODE variant)** are the
most principled — the first makes $V_\theta$ interpretable in a way
no transformer analogue is, the second removes the discretization
story as a source of epistemic uncertainty.

For **SPLM's raw performance numbers**, **A1 (width sweep)** and
**A3 ($V_\theta$ capacity sweep)** are the most likely to
substantially lower perplexity, since current numbers are
essentially unscaled.

### Top-3 shortlist

If only three experiments fit in the next iteration, pick:

1. **B1 — KV-cache-free decoding benchmark.**  One afternoon of
   implementation, one day of measurement.  Highest chart-worthy
   payoff.
2. **A1 — Width sweep at fixed $L$, on TinyStories.**  Three
   training runs at $d \in \{128, 256, 512\}$ plus matched
   transformer baselines.  Establishes the scaling story.
3. **C2 — Energy-drift diagnostic.**  Pure-eval on existing
   checkpoints (SPLM Euler, SPLM Verlet L=16 dt=0.5, GPT-2).  No
   new training required.

Together these give the paper: (a) an efficiency win, (b) a
scaling curve, (c) a new diagnostic axis — the three ingredients
for a convincing positional claim.

### Round-1 execution plan (current state)

The following is the actual plan currently being executed against
the catalogue above.  It does not replace the prioritisation; it
records what is in flight, what is queued, and why this ordering
was chosen given today's compute budget.

1. **Section 0 — multi-seed variance bar.**  *In flight (phase 1
   of 2).*  $N=5$ seeds on Tiny Shakespeare via
   `notebooks/conservative_arch/multi_seed/`.

   - **Phase 1** ($\sim 5$-$5.5$ h MPS, in progress):
     `sarf_mass_variant logfreq` (paper headline) +
     `matched_baseline` transformer.
   - **Phase 2** ($\sim 2.5$ h MPS, queued behind phase 1):
     `energetic_minima --variant ln` (LayerNorm-after-step,
     val ppl 88.63 at seed=0 -- the strongest single-seed result
     in the repo).  Adding this avoids the failure mode where E1
     reports a variance bar around the *second-best* SPLM
     configuration only.

   Output: `results/E1_shakespeare/E1_report.md` covering all three
   models with mean $\pm$ std, pairwise Welch $t$-tests, and overlay
   loss curves.  Selected first because it is the cheapest item that
   defends the paper's actual best-known SPLM perplexity, and because
   every quality experiment below inherits this variance bar.

2. **C2 — energy-drift diagnostic.**  *Implementation
   complete, run queued.*  Pure-eval extract + diagnostic on the
   existing trained checkpoints (Euler $L=8$, `sarfmass logfreq`
   $L=8$, Verlet $L=16,\Delta t=0.5$).  Implementation in
   `notebooks/conservative_arch/energy_drift/`.  Compute is
   trivial ($\sim 1$ min MPS); the queue is purely to avoid MPS
   contention with item 1.

3. **A1 — width sweep on TinyStories.**  *Next.*  Three
   matched training pairs at $d \in \{128, 256, 512\}$, run
   under the harness from Section 0 with $N \geq 3$ seeds each.
   Estimated 1-2 days of MPS time.  Pushed to last because it
   is by far the most expensive of the three; running it before
   items 1-2 complete would mean any aggregate findings are
   still single-seed.

**Interpretation framework for E1 outcomes.**  The single-seed
number $88.63$ (LN-after-step) already inverts the baseline
ordering ($88.63 < 141.7$ matched).  E1 phase 2 turns this from a
single data point into a properly-reported $\mathrm{ppl} \pm \sigma$
measurement.  Three outcome regimes are possible at $N=5$:

- **Regime A: LN cleanly beats matched (LN mean $+ \sigma$ below
  matched mean $- \sigma$).**  This constitutes statistically
  significant evidence that LN-after-step is the stronger SPLM
  configuration.  The SPLM-vs-matched perplexity gap with 95% CI
  from Welch's $t$-test is the primary reportable quantity.
  The result also shows that the matched baseline is the better LM
  *at the SARF+logfreq configuration* but not at the strongest
  tested SPLM configuration.

- **Regime B: LN beats matched but not cleanly (overlapping CIs).**
  The LN-after-step configuration represents the first known SPLM
  setting where the perplexity gap to the matched baseline closes
  within $1\sigma$; the gap should be presented with appropriate
  uncertainty quantification.

- **Regime C: $88.63$ does not survive multi-seed (LN mean
  comparable to or worse than matched mean).**  The seed=0 number
  was unrepresentative; the multi-seed result recovers the original
  theoretical prediction that LN-after-step leaves validation
  perplexity essentially unchanged relative to the baseline
  configuration.

Pre-registering the regime-to-conclusion mapping before the data
lands protects the interpretation against ex-post rationalisation.

**Deferred to Round-2 (with reasoning).**  B1 (KV-cache-free
decoding benchmark) is the original top-3 item but its payoff
depends on long-context evaluation data ($T \geq 512$) that the
current Tiny Shakespeare / TinyStories corpora do not exercise
meaningfully.  Running it before A1 lands a TinyStories-trained
checkpoint at $d \geq 256$ would produce a chart that is
technically correct but qualitatively uninformative, so we run
it after A1.

---

## H. Relationship to the symplectic-integration study

The integrator ablation described in
[Symplectic_Integration_for_SPLM.md](Symplectic_Integration_for_SPLM.md)
is best read as a *completed axis* in this catalogue:

- **Axis covered.** Integrator order (Euler vs velocity-Verlet +
  Strang damping).
- **Axes opened.** Two of the experiments above (C1, C2, E1, E2)
  are direct follow-ons to that study and would not have been
  motivated without it.
- **Axes untouched.** Everything in sections A, B, D, F — the
  scaling, efficiency, domain-transfer, and positioning-against-
  attention stories — is orthogonal to the integrator ablation and
  is where the next round of experimental effort is most likely to
  change the paper's empirical profile.
