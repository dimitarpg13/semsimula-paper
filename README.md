# Semantic Simulation — Paper companion

Companion repository for the paper

> **Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference**
> *Conservative-by-Construction Language Models and the Shared-Potential Separator, with a Correspondence to Joint Embedding Predictive Architectures.*
> Dimitar P. Gueorguiev (Independent Researcher), 2026.
> Zenodo preprint: [10.5281/zenodo.19819861](https://doi.org/10.5281/zenodo.19819861)

[![DOI — paper](https://zenodo.org/badge/DOI/10.5281/zenodo.19819861.svg)](https://doi.org/10.5281/zenodo.19819861)
[![DOI — companion code](https://zenodo.org/badge/DOI/10.5281/zenodo.19708205.svg)](https://doi.org/10.5281/zenodo.19708205)

This repository collects the **reproducibility artifacts** and the **unpublished
background manuscripts** cited in the paper. Its scope is deliberately narrow:
everything here is either directly required to resolve a citation in the paper
or needed to reproduce a figure or experimental claim. The repository covers
both the descriptive experiments of §13 (STP–acceleration identity and the
Pythia / GPT-2 deceleration analysis) and the prescriptive experiments of §14
and Appendix A (the negative-results chain on attention transformers, the
scalar-potential language model, and the three-way shared-potential
separator).

> **Git LFS.** Some large binary artefacts under
> `notebooks/conservative_arch/` (trajectory pickles up to ~110 MB, SPLM and
> matched-GPT-2 checkpoints at ~30 MB each, and several PNG figures) are
> stored via **Git LFS**. After cloning, run `git lfs pull` once to download
> them; without this step the large files will appear as short text pointers.
> Git LFS is not required for the v1 artefacts under `notebooks/stp_loss/`
> and `notebooks/cross_model/`.

> **Rendering note.** Several markdown files under `companion_notes/` in this repository contain LaTeX math (inline `$...$` and display `$$...$$` blocks, with macros such as `\mathfrak{...}`, `\boldsymbol{...}`, `\mathcal{...}`, etc.). The math has been verified to render correctly in **Safari**. In **Chrome** some symbols — notably calligraphic and fraktur letters, e.g. `\mathfrak{C}` rendering as a plain `C` instead of $\mathfrak{C}$ — appear to render incorrectly. **Firefox** has not been tested. If symbols look wrong while viewing a companion note on GitHub, please open the file in Safari or consult the main paper's PDF, where the same symbols are typeset by LaTeX directly. Each affected companion note repeats this warning in its own header.

---

## Repository contents

### `manuscripts/` — cited unpublished manuscripts

These are the unpublished technical notes cited in the paper's bibliography
(`@unpublished{Gueorguiev…}` entries in `references.bib`). They document the
author's 2021–2026 work on Semantic Simulation, whose material is subsumed and
extended by the paper. Both PDF and Word (`.docx`) versions are provided where
available.

| BibTeX key                      | File                                                                                             | Paper section(s) where subsumed |
| ------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------- |
| `Gueorguiev2021TreeOps`         | `Semantic_Tree_Operations.{pdf,docx}`                                                            | §8                              |
| `Gueorguiev2022Foundations`     | `The_Foundations_of_Semantic_Simulation.{pdf,docx}`                                              | §§1–2                           |
| `Gueorguiev2022PARF`            | `Modeling_Attractive_and_Repulsive_Forces_in_Semantic_Properties.{pdf,docx}`                     | §5                              |
| `Gueorguiev2022DynSim`          | `On_The_Need_of_Dynamic_Simulation_when_Modeling_Interactions_of_Semantic_Structures.{pdf,docx}` | §6                              |
| `Gueorguiev2022Signature`       | `On_the_Signature_Matrix_of_Semantic_Property.{pdf,docx}`                                        | §3                              |
| `Gueorguiev2022SARF`            | `Modeling_Attractive_and_Repulsive_Forces_between_Semantic_Structures.{pdf,docx}`                | §6                              |
| `Gueorguiev2022Well`            | `On_Gaussian_Inverse_Semantic_Energy_Well.{pdf,docx}`                                            | §4                              |
| `Gueorguiev2022Execution`       | `Execution_Of_Semantic_Structures.{pdf,docx}`                                                    | §8.6 (summary only)             |
| `Gueorguiev2024SemSim`          | `Semantic_Simulation.{pdf,docx}`                                                                 | §§1–2                           |
| `Gueorguiev2026Lagrangian`      | `Constructing_Langrangian_for_Semantic_Space.{pdf,docx}`                                         | §7                              |

These manuscripts are the **historical record** of the work. The paper is the
canonical statement of the framework as of 2026; the notes above are preserved
because they are cited in the paper.

**Additional background manuscript (not cited in the paper).** For
completeness, the `manuscripts/` folder also includes
`Semantic_Templates.{pdf,docx}`, a 2022 background note from the same line of
work. It is **not** referenced in the paper and has no BibTeX entry; it is
retained here only as context for readers who want to trace the evolution of
the framework. Its earliest git commit in `aiconcepts` is recorded in
[`manuscripts/PROVENANCE.md`](manuscripts/PROVENANCE.md) for completeness.

The authorship dates asserted in the paper's bibliography (`note` fields) are
the primary record for each manuscript. An independent external anchor —
showing the earliest commit of the original `.docx` source in the author's
long-running research repository
[`dimitarpg13/aiconcepts`](https://github.com/dimitarpg13/aiconcepts) — is
provided in [`manuscripts/PROVENANCE.md`](manuscripts/PROVENANCE.md).

> **Note on `Gueorguiev2024ReadMe`.** The citation "Semantic Simulation with
> Reinforcement Learning — README" (cited in §8) refers to unpublished
> project documentation dated 30 September 2024. The document is not
> publicly distributed and is not included in this companion repository;
> the bibliography entry records it as an `@unpublished` note whose
> material is subsumed and extended in §8 of the paper.

### `companion_notes/` — 2026 companion notes (work in progress)

These are informal companion notes and living documents developed alongside
the paper. They capture material that the paper **does not subsume** but that
is either summarized briefly in the main text (with a pointer here for the
detailed treatment) or is the target of explicit deferral to future work.
They are **not** peer-reviewable standalone artifacts; they are included so
that readers of the paper can trace the reasoning behind claims that are
summarized (but not fully developed) in the main text.

| BibTeX key                       | File                                                                                             | Cited around                |
| -------------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------- |
| `SemSimNotes2026Mass`            | `On_the_Interpretation_of_Semantic_Mass.md`                                                      | §11                         |
| `SemSimNotes2026Hidden`          | `On_the_Interpretation_of_Hidden_State.md`                                                       | §10                         |
| `SemSimNotes2026Accel`           | `On_The_Existence_of_Acceleration_in_Semantic_Structures.md`                                     | §12                         |
| `SemSimNotes2026Emergent`        | `STP_Loss_Is_An_Emergent_Property_Of_The_Energy_Landscape_Defined_By_Gaussian_Well_Potential.md` | §12                         |
| `Gueorguiev2026ExecutionProblem` | `The_Execution_Problem.md`                                                                       | §8.6 (deferred to companion)|

In addition, the following three living documents back the §14 / Appendix A
discussion of attention-transformer conservativity and the conservative-by-
construction language model. They are not cited by BibTeX key in the paper's
bibliography (the paper develops their content in-line) but are included
here as the author's working notes for readers who want the longer
exposition:

| File                                                                                | Cited around                              |
| ----------------------------------------------------------------------------------- | ----------------------------------------- |
| `The_Failure_of_Conservative_Models_to_explain_hidden_state_trajectories.md`        | §14.1 (negative results)                  |
| `P-rot-6_transformer_dynamics.md`                                                   | §14.1 (theoretical motivation for the velocity-coupled gauge fit: derives the zero-free-parameter prediction $B_{\text{theory}} = \Omega(\bar x)$ from the K$\ne$V antisymmetry in scaled dot-product attention that the E5 experiment in `notebooks/e_init/velocity_coupled_gauge.py` empirically tests) |
| `Conservative_by_Construction_Language_Models.md`                                   | §14 (motivation for SPLM)                 |
| `Considered_Non-Autonomous_Conservative_Mechanisms.md`                              | Appendix A (non-autonomous framework)     |
| `Addendum_Non_Autonomous_Fields_For_Appendix_A.md`                                  | Appendix A, §14 (short reader’s guide: Class F equation, Hopfield / Tracks A–B, integrability; points to the full appendix in the PDF) |
| `On_Modeling_Semantic_Energy_Field_into_SPLM.md`                                    | §14.2 (mapping framework energy field onto $V_\theta$, $\xi$, $m_t$; candidate Q11–Q13) |
| `On_The_Smoothness_of_Scaled_Dot_Product_Attention.md`                              | §14, Theorem 46 (smoothness of attention in $h$; Poincaré prerequisites; $\Omega^{\mathrm{att}}$; LayerNorm / ReLU / mask caveats) |
| `Training_and_Inference_with_SPLM.md`                                               | §14.2, §14.13, §14.14 (training loop, nested-autograd force computation, inference pipeline, and summary of the fixed-$\xi$ / SARF-faithful / per-token-mass ablations) |

#### Forthcoming-work planning artifacts (deferred to a future companion paper)

The two documents below are categorically different from the working
notes above. They do **not** support a claim made in the present paper;
they specify a separate, deliberately deferred research programme — a
direct, RL-calibrated particle-mechanics simulator in semantic space —
that the paper points to as forthcoming work at the end of the theory
chapter (§8.8, "Salience and destruction of semantic structures",
formalising the (D1)–(D5) framework requirements), with a bridging
sentence in §14.17 (Open follow-ups) and a final empirical-scope
forward-pointer in §16 ("Empirical scope: the structure lifecycle").
They are included here so that readers who follow the paper's
`\path{...}` pointers can locate them; they are explicitly **not** part
of the SPLM/conservative-architectures experimental record of §14, and
a self-contained future companion paper covering this branch is the
natural home for the empirical validation of their content.

| File                                              | Pointed to from                                                                                                                                 |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `Semantic_Simulator_RL_Calibration_Programme.md`  | §8.8 ((D1)–(D5) destruction requirements), §14.17 (Open follow-ups bridge), §16 ("Empirical scope: the structure lifecycle") — programme-level memo, milestones M0–M6 + v1.5/v2/v3 lifecycle extensions |
| `Semantic_Simulator_EOM.md`                       | §8.8 ((D1)–(D5) destruction requirements), §14.17 (Open follow-ups bridge), §16 ("Empirical scope: the structure lifecycle") — v0 equations of motion, parameter classification, pseudocode |

#### Dynamical-simulation expressivity programme — formal proofs and prioritised catalogue

The four documents below extend the v0 dynamical-simulation programme of
`Semantic_Simulator_RL_Calibration_Programme.md` and
`Semantic_Simulator_EOM.md` with **formal expressivity bounds** that pin
the framework on both sides of the Chomsky hierarchy and a **prioritised
experimental catalogue** that operationalises them. Two are short
technical notes that prove specific theorems; two are longer planning
documents that organise the staged research programme. Together they
back the §16 ("Empirical scope: the structure lifecycle") forward-pointer
and motivate the multi-seed (E1) and energy-drift (E3) experiments
catalogued under
[`notebooks/conservative_arch/multi_seed/`](notebooks/conservative_arch/multi_seed/)
and
[`notebooks/conservative_arch/energy_drift/`](notebooks/conservative_arch/energy_drift/)
below.

| File                                              | Role                                                                                                                                                                                                                                                                              |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Expressivity_Bounds_For_v0_Simulator.md`         | Formal short note: the v0 simulator class accepts **at most regular languages**, by a four-step argument (phase-space-capacity bound $\dim M \cdot \log_2(L/\epsilon)$ bits, exponential information contraction at damping rate $\gamma$, smooth non-chaotic $V$ ruling out the Siegelmann–Sontag continuous-flow construction, and consequent acceptance class $\subseteq$ REG). Derives the predicted $\mathrm{Dyck}_n$ collapse depth $D^\ast$ in closed form; supplies one-paragraph mathematical-apparatus sketches for the v1.5 / v2 / v3 extensions. |
| `MCS_Reduction_For_v3_Composite.md`               | Formal proof of the framework's *upper* expressivity boundary: the composite v0+v1.5+v2+v3 simulator, under explicit boundedness assumptions on its operator algebra, generates **exactly** the **mildly context-sensitive** class — equivalently LCFRS / MCFG of bounded fan-out and bounded rank, the empirically established class for human language (Joshi 1985, Shieber 1985). The reduction is constructive in both directions and verifies the four classical MCS criteria of Joshi (1985); §5.5 catalogues the structural and architectural reasons v3 cannot be eliminated. |
| `Advancing_The_Dynamic_Simulation_Model.md`       | Conceptual scaffold for the v1.5 / v2 / v3 extensions: maps each onto a mature mathematical apparatus (salient decay → dissipative semigroups and discounted MDPs; creation/destruction → Fock space and the canonical creation/annihilation algebra; execution → Lie groups acting via non-abelian gauge fields), specifies the falsifying-experiment battery ($\mathrm{Dyck}\_n$ + topic-shift, $\mathrm{Dyck}\_n$ + let-binding, cross-serial $a^n b^n c^n$, bounded copy $ww$, 2-counter), and identifies the composite as a classical-mechanical analogue of a Haag–Kastler-style local operator algebra. |
| `Next_Model_Experiments_for_SPLM.md`              | Prioritised, actionable catalogue of experiments that either strengthen the framework's applicability (sections A–F: multi-seed variance, scaling, integrator ablation, expressivity falsifiers, capacity sweeps, energy-drift diagnostic) or measurably improve SPLM's performance. Each item is concrete (what to run, what to measure, why it matters). The source of truth for the **E1** multi-seed harness and the **E3** energy-drift diagnostic shipped under `notebooks/conservative_arch/`. |

`Expressivity_Bounds_For_v0_Simulator.md` and
`MCS_Reduction_For_v3_Composite.md` are formal proofs whose claims
directly back the framework's expressivity statements;
`Advancing_The_Dynamic_Simulation_Model.md` and
`Next_Model_Experiments_for_SPLM.md` are programme-level planning
artifacts in the same family as the two notes immediately above and are
included so that readers who follow the paper's `\path{...}` pointers,
or want to trace the design rationale of the experiments shipped under
[`notebooks/conservative_arch/multi_seed/`](notebooks/conservative_arch/multi_seed/)
and
[`notebooks/conservative_arch/energy_drift/`](notebooks/conservative_arch/energy_drift/),
can locate them.

### `notebooks/` — reproducibility

#### `notebooks/stp_loss/energy_landscape_validation.ipynb`

The notebook behind the GPT-2 STP-acceleration analysis reported in §13 of the
paper. It loads a pretrained GPT-2 checkpoint, extracts last-layer hidden
states over a corpus of natural-language text, computes discrete
displacement/acceleration triplets, and evaluates the relationships predicted
by the Gaussian-well / Lagrangian model of §§4–7 and the STP–acceleration
identity of §12.

An `energy_landscape_validation_executed.ipynb` is provided with all cell
outputs intact for reviewers who prefer a static rendering. The
`results/` subfolder contains the serialized outputs (`experiment_results.json`)
and rendered figures (`stage2–stage6_*.png`) that back the corresponding
figures in §13 of the paper.

#### `notebooks/cross_model/pythia_tangential_acceleration.ipynb`

The notebook behind the cross-architecture replication reported as
**Result 5** in §13 of the paper. It runs the tangential / normal
acceleration analysis on **both** GPT-2 small and Pythia-160M, extracting
last-layer hidden-state trajectories, decomposing discrete acceleration
into tangential and normal components, and comparing the observed
distribution against a token-swap permutation null. The result is that
both architectures exhibit the same qualitative signature --- net
deceleration on essentially every triplet (≈ 98% for GPT-2, 100% for
Pythia-160M), with $|\vec a_{\parallel}|$ and $|\vec a_{\perp}|$ both
significantly smaller than the permutation null --- showing that the
"bounded attractive well plus deceleration" picture is not an idiosyncrasy
of GPT-2's specific training trajectory.

The `results/` subfolder contains the serialized summary
(`cross_model_summary.json`), the raw per-triplet samples
(`cross_model_samples.npz`), and the two figures used in §13 of the paper
(`cross_model_obs_vs_null_bars.png`, `cross_model_a_par_hist.png`).

#### `notebooks/e_init/` — scalar, Helmholtz, and gauge-field fits on GPT-2

The **negative-results chain** of §14.1 ("Retrospective: five negative
experiments on scalar, linear Helmholtz, and velocity-coupled gauge fits").
This folder closes the natural classical-Lagrangian menu on pretrained
GPT-2 small hidden-state trajectories over the 50-sentence, five-domain
E-init corpus and shows that every ansatz considered ties or loses to the
static-null baseline on held-out data. Contents:

- `e_init_validation.ipynb` — the reference run (§1 Gaussian-well fit,
  per-sentence per-layer centered, produces the `e_init_results*.npz` and
  `well_params*.json` artefacts consumed by the follow-up scripts);
- `extended_gamma_and_first_order.py` — E1 (extended damping sweep
  $\gamma \in \{0,\ldots,50\}$) and E2 (first-order overdamped gradient
  flow, $\eta \in \{10^{-4},\ldots,10^{-1}\}$);
- `well_functional_form_comparison.py` — E3 (seven scalar-well functional
  forms: harmonic, Gaussian, Morse, Lorentzian-saturation, log-saturation,
  Weibull, power);
- `helmholtz_curl_augmented.py` — E4 (linear Helmholtz augmentation with a
  position-coupled skew term $\Omega x$);
- `velocity_coupled_gauge.py` — E5 (velocity-coupled gauge
  $F(x)\dot{x}$ with constant, affine-rank-1, and affine-rank-2 $F$).

Each script writes a markdown summary, an `.npz` of numerical results, and
one or more `.png` figures to `results/`. See
[`notebooks/e_init/README.md`](notebooks/e_init/README.md) for the full
catalogue and the reproduction commands.

#### `notebooks/conservative_arch/` — the SPLM prototype and the three-way separator

The **prescriptive experiments** of §14.2 ff. and Appendix A. This folder
implements the scalar-potential language model (SPLM), trains it and a
scale-matched attention baseline on Tiny Shakespeare, extracts hidden-
state trajectories from SPLM / matched GPT-2 / pretrained GPT-2 small on
the identical corpus, and runs the strict shared-potential fit, the
velocity-aware Jacobian-symmetry test, the $V_{\psi}$ capacity sweep, the
SPLM oracle fit, and the token-direction replication. Contents (see
[`notebooks/conservative_arch/README.md`](notebooks/conservative_arch/README.md)
for the full list):

- **Models.** `model.py` (SPLM), `matched_baseline_model.py` (matched GPT-2);
- **Training.** `train_splm.py`, `train_matched.py` (Tiny Shakespeare
  loaders in `data_module.py`, BPE-tokenised in `data/`);
- **Trajectory extraction.** `trajectory_extraction.py`,
  `extract_gpt2_baseline.py`, `extract_matched_baseline.py`;
- **Diagnostics.** `e_init_validation.py` (Gaussian-well E-init on SPLM),
  `jacobian_symmetry.py` (velocity-aware PCA-16 test),
  `shared_potential_fit.py` (strict shared-$V_{\psi}$ across all layers),
  `sharedV_capacity_sweep.py` (6-config $V_{\psi}$ capacity band),
  `splm_oracle_fit.py` (oracle upper bound using SPLM's own $V_{\theta}$),
  `token_direction_fit.py` (token-axis replication);
- **Plots.** `plot_sharedV_comparison.py`,
  `plot_three_way_comparison.py`,
  `plot_token_vs_layer_three_way.py`;
- **Pipeline driver.** `run_full_pipeline.py`;
- **`results/`.** Serialised pickles (SPLM / matched / GPT-2 trajectories),
  model checkpoints (`.pt`), `.npz` result archives, markdown summaries,
  and rendered figures — the raw material behind the separator plot
  (Fig. 8, "SPLM vs.\ matched GPT-2 vs.\ pretrained GPT-2"), the
  capacity-sweep saturation plot, the oracle ceiling, and the
  token-direction robustness check.
- **`sarf_variant/`.** Controlled ablation of §14.13: a SARF-faithful
  SPLM that recomputes the reinforcement-field pool $\xi^{(\ell)}$ at
  every integration step instead of freezing it at the input layer.
  Ships its own `model_sarf.py`, `train_splm_sarf.py`,
  `trajectory_extraction_sarf.py`, `compare.py`, comparison plots, and
  [`comparison_report.md`](notebooks/conservative_arch/sarf_variant/comparison_report.md);
  the parent `shared_potential_fit.py` and `token_direction_fit.py`
  diagnostics are reused verbatim and their SARF outputs
  (`sharedV_sarf_*`, `tokdir_sarf_*`) live alongside the baseline
  results in `results/`. Headline finding: a single-line change to the
  context pool yields a **33 % Tiny-Shakespeare perplexity
  reduction** at identical parameter count, wall-clock, and compute,
  while preserving the shared-potential separator.
- **`sarf_mass_variant/`.** Follow-up ablation of §14.14 stacking
  per-token semantic mass on top of the SARF-faithful $\xi$ of
  §14.13 --- the first paper experiment that directly targets
  Open Question Q10 (the prescribed per-token mass of §7). Ships a
  `model_sarf_mass.py` that exposes three mass modes
  (`global`, `embed_head`, `logfreq`), a
  `compute_unigram_frequencies.py` that builds the frozen
  Shannon-surprisal lookup $-\log\hat p(x_t)$ on the Tiny
  Shakespeare training split, a `train_splm_sarf_mass.py` selecting
  the mass mode via `--mass-mode`, a
  `trajectory_extraction_sarf_mass.py`, a four-way `compare.py`
  covering fixed-$\xi$ / SARF / SARF+embed-head / SARF+logfreq, and
  [`comparison_report.md`](notebooks/conservative_arch/sarf_mass_variant/comparison_report.md);
  the parent `shared_potential_fit.py` and `token_direction_fit.py`
  diagnostics are reused verbatim and their mass-variant outputs
  (`sharedV_sarfmass_*`, `tokdir_sarfmass_*`) live alongside the
  baseline and SARF results in `results/`. Headline finding: the
  framework-prescribed surprisal mass
  $m_t \propto -\log\hat p(x_t)$ (variant (B)) yields a
  **44 % Tiny-Shakespeare perplexity reduction** vs. fixed-$\xi$
  SPLM (and **17 % vs. SARF-faithful SPLM**) at the cost of a
  single extra scalar parameter and a frozen vocabulary-sized
  surprisal tensor, and *simultaneously* raises the depth-axis
  pooled shared-$V_\psi$ $R^2$ from $+0.79$ (fixed-$\xi$) to
  $+0.84$ — the first configuration in which LM perplexity and
  strict shared-potential fidelity improve in the same direction.
  A free learned linear head (variant (A)) underperforms variant
  (B) by ~27 % val ppl at this scale, an inductive-bias-vs-data-
  efficiency result flagged as the Q10 open follow-up in §14.17 and
  §16. All four training logs, checkpoints, trajectory pickles, and
  per-layer diagnostic tables live under
  `sarf_mass_variant/results/`; `comparison_*.png` are mirrored at
  the folder root for direct figure inclusion in the paper.
- **`attractor_analysis/`.** Direct test of one of the load-bearing
  predictions of the *Semantic Simulation* framework, reported in
  §14.15 (`subsec:cba-attractors`): that a trained scalar potential
  $V_\theta(\xi, h)$ exhibits prompt-dependent localised basins
  corresponding to coherent semantic configurations. Attention
  transformers have no analogue of this prediction. Ships
  `attractor_extraction.py` (two modes — `gradient`: Adam descent on
  $V_\theta(\xi, h)$ plus a data-manifold anchor
  $\tfrac{\lambda}{2}\lVert(h - h_c)/h_s\rVert^2$; `dynamical`: SPLM's own
  semi-implicit damped Euler from random $h$ seeds at fixed $\xi$ for
  exactly $L_\text{train}$ steps), `landscape_3d.py` and
  `compare_landscapes_3d.py` (3D rendering of $V_\theta$ as a surface
  over the 2D PCA plane of trajectory data, with damped-flow
  trajectories overlaid; Euler-vs-Verlet side-by-side comparison
  including
  [`landscape3d_compare_dialogue.png`](notebooks/conservative_arch/attractor_analysis/results/landscape3d_compare_dialogue.png)
  and rotating 360° GIFs), and the during-training pair
  `train_with_snapshots.py` + `render_training_evolution.py` (retrain
  SARF+mass SPLM with checkpoints at log-spaced steps
  $\{0, 50, 200, 500, 1000, 2000, 4000\}$, render the seven-panel
  landscape-evolution grid showing how $V_\theta$ carves a basin from
  flat). Headline findings: (1) pure Adam descent on $V_\theta$ runs
  $h$ off to infinity ($\langle V\rangle = -2500$ at step 300 vs.
  $-260$ on the real trajectory) — $V_\theta$ is **unbounded below**
  along multiple directions because training only ever sees its
  gradient; (2) anchored descent (any $\lambda \in [10, 10^3]$)
  collapses to **one prompt-independent attractor** decoding to the
  same five stopwords (`,`, `\n`, `the`, `a`, `-`); (3) the **damped
  dynamics at $L = L_\text{train}$** *does* exhibit prompt-dependent
  multi-basin structure with silhouette-optimal
  $K^\ast \in \{2, \dots, 10\}$ basins decoding to real tokens
  (`the`, `I`, `to`, `\n`, `:`, `,`). The "semantic attractors" of
  SPLM are therefore **time-bounded basins of the damped flow**, not
  minima of $V_\theta$ — consistent with the framework's
  pullback-attractor mathematics but distinct from the narrower
  energetic reading. The Verlet $L = 16$ ablation produces fewer,
  more punctuation-dominated basins, which tracks its slight
  perplexity regression: more accurate but heavier-damped integrators
  concentrate probability mass on the global "punctuation manifold"
  of Tiny Shakespeare. The consolidated paper-style write-up is
  [`companion_notes/Semantic_Attractor_Extraction.md`](companion_notes/Semantic_Attractor_Extraction.md);
  the in-folder
  [`attractor_analysis/README.md`](notebooks/conservative_arch/attractor_analysis/README.md)
  documents the per-prompt JSON/PNG/MD outputs.
- **`energetic_minima/`.** Three falsification experiments motivated
  by the §14.15 design rationale (R1–R6) and reported as Q11 of
  §14.17: structural alternatives to a free $V_\theta$ that *should*
  buy a finite energetic minimum at zero or modest expressivity cost,
  if R5/R6 are correctly framed. Implements all three in a unified
  pipeline: `model_ln.py` (LayerNorm-after-step — project $h_{l+1}$
  back onto the unit-LayerNorm shell after every damped step;
  compactness of $S^{d-1}$ guarantees a finite minimum without
  changing $V_\theta$ itself), `model_gm.py` (Gaussian-mixture head
  $V_\theta(\xi,h) = \sum_{k=1}^{K} \mathrm{amp}_k (1 - e^{-\kappa_k^2 \lVertz - c_k\rVert^2})$,
  the **honest test** of the framework's prescribed well form at
  full SPLM scale), a unified `train.py --variant {ln, sg, gm}` (the
  scale-gauge `sg` is a loss-side regulariser
  $\lambda_{V_0} \mathbb{E}_{b,t} V_\theta(\xi_0, h_0)^2$ on the
  baseline model, anchoring $V_\theta$ at the input embedding),
  `run_attractor_pipeline.sh` driving `attractor_analysis/` over all
  four checkpoints (baseline + three alternatives), `compare.py`
  building [`results/comparison_report.md`](notebooks/conservative_arch/energetic_minima/results/comparison_report.md),
  and `make_compare_figure.py` assembling
  [`results/landscape3d_compare_four_variants_dialogue.png`](notebooks/conservative_arch/energetic_minima/results/landscape3d_compare_four_variants_dialogue.png).
  Headline findings:
  **(i) LayerNorm-after-step** drops val ppl from baseline's
  $160.55$ to **$88.63$** — a 45 % relative improvement at
  zero additional parameters — and narrows the $V$ range from
  $[-1916, +1445]$ to $[-84, -60]$ (about $30\times$ narrower)
  while keeping $K^\ast$ prompt-dependent; the neutrality prediction
  is **refuted in the positive direction**, R6 ("unbounded below is
  a gauge, not a pathology") is **strengthened by a second witness**,
  and the cost is a shift toward punctuation-dominated attractors
  (content-basin fraction drops from $0.58$ to $0.23$).
  **(ii) Scale-gauge** at $\lambda_{V_0} = 10^{-3}$ gives val ppl
  $191.0$ (+19 % vs. baseline), $V$ range $[-2332, -186]$ (not
  narrower), and $K^\ast$ collapse to $2$ on four of five prompts —
  neither a useful regulariser nor a decisive falsifier at this
  $\lambda$; full $\lambda_{V_0}$ sweep noted but not promising
  given the basin-collapse signal.
  **(iii) Gaussian-mixture head** ($K = 64$) plateaus at val ppl
  **$677.67$** ($4.2 \times$ worse), the $V$ range collapses to a
  $0.05$-wide spike at $\approx +60.3$, $196$ of $288$ seeds
  structurally converge to the same two basins for **every one of
  the five prompts**, and both basins decode to the Tiny-Shakespeare
  unigram distribution (content-basin fraction $0.00$) — the model
  has collapsed to a context-free unigram predictor. The
  pre-registered falsification criterion (a Gaussian-mixture SPLM
  matching the SARF-faithful val ppl $160.6$) **fails by a factor of
  four**; **R5** ("structurally bounded $V$ is expressivity-limited
  at this scale") is **vindicated both statically and dynamically**.
  Net effect: R5 must be *sharpened* from "structurally bounded
  scalar potentials are expressivity-limited" to "structurally
  bounded $V_\theta$ (GM) is expressivity-limited; compactifying the
  state space while keeping $V_\theta$ a free MLP (LN) is not." The
  consolidated paper-style write-up is
  [`companion_notes/Energetic_Minima_Alternatives.md`](companion_notes/Energetic_Minima_Alternatives.md);
  the in-folder
  [`energetic_minima/README.md`](notebooks/conservative_arch/energetic_minima/README.md)
  documents the variant flags, training schedule, and full attractor
  pipeline.
- **`multi_seed/`.** Multi-seed variance harness — the **E1** experiment
  of the
  [`Next_Model_Experiments_for_SPLM.md`](companion_notes/Next_Model_Experiments_for_SPLM.md)
  programme — that closes the *"no error bars"* gap inherited by every
  earlier single-seed SPLM perplexity number. Re-runs three trainers
  (the matched GPT-2-micro baseline `matched_baseline`, the previous
  SPLM flagship `splm_sarfmass_logfreq`, and the new
  LayerNorm-after-step variant `splm_em_ln`) at five distinct random
  seeds each on Tiny Shakespeare and aggregates per-seed training logs
  into a curated multi-seed report with mean ± std, pairwise Welch
  t-tests with 95 % CIs, per-seed loss-curve overlays, and a
  divergence-rate diagnostic that stratifies architectures by
  gradient-norm trajectory. Ships
  [`multi_seed_runner.py`](notebooks/conservative_arch/multi_seed/multi_seed_runner.py)
  (subprocess-driven N-seed launcher; one model spec per row in
  `MODEL_SPECS`, model-agnostic — adding a variant is a 5-line entry),
  [`multi_seed_aggregator.py`](notebooks/conservative_arch/multi_seed/multi_seed_aggregator.py)
  (NaN-aware mean / std / min / max + Welch's t-test + overlay plots
  that draw diverged seeds as dotted exclusions from the mean), and
  [`e1_divergence_diagnostic.py`](notebooks/conservative_arch/multi_seed/e1_divergence_diagnostic.py)
  (per-seed first-NaN step + grad-norm trajectory tabulation, with a
  stratified figure overlaying training NLL and grad-norm for all three
  architectures). Headline finding (`results/E1_shakespeare/`,
  N = 5 seeds): `splm_em_ln` reaches val ppl **$95.33 \pm 4.44$**
  versus `matched_baseline` **$149.80 \pm 7.21$** — a **36.4 %**
  relative improvement at **11.5 %** fewer parameters
  ($7.12$ M vs $8.05$ M), Welch's $t = 14.4$, two-sided p-value
  $< 10^{-5}$, with 95 % CI on the gap **$[+45.4,+63.5]$ ppl**
  (well-separated from zero); the worst `em_ln` seed (98.78) still
  beats the best baseline seed (141.80) by ~30 %. The previous
  flagship `splm_sarfmass_logfreq` is **structurally falsified** at this
  corpus scale: 2 of 3 seeds NaN-diverge at training steps 1250 and
  3250 with *modest* gradient norms (5–13), which together with the
  stable LN-after-step run-mate establishes that the failure mode is
  **integrator-side state-space drift**, not gradient blow-up, and that
  the LayerNorm-after-step projection onto $S^{d-1}$ is the minimal
  intervention that restores stability without sacrificing perplexity.
  The curated narrative report
  [`results/E1_shakespeare/E1_report.md`](notebooks/conservative_arch/multi_seed/results/E1_shakespeare/E1_report.md)
  is the canonical write-up; the auto-generated machine-friendly
  companion `E1_shakespeare_report.md` is regenerable from per-seed
  logs in ~20 seconds. The in-folder
  [`multi_seed/README.md`](notebooks/conservative_arch/multi_seed/README.md)
  documents the runner / aggregator / diagnostic interface and is the
  template for adding additional model specs (E2 width sweep,
  E3 integrator ablation, etc.). All 13 per-seed checkpoints, training
  logs, and loss curves (11 finite + 2 NaN-diverged seeds — the latter
  retained because they are themselves the falsifying evidence for the
  divergence-rate diagnostic) are committed under Git LFS so that the
  aggregator and diagnostic steps alone reproduce the headline table
  from shipped artifacts.
- **`energy_drift/`.** Eval-only diagnostic that opens a new
  *architecture-discriminating* axis for the SPLM — the **E3**
  experiment of the
  [`Next_Model_Experiments_for_SPLM.md`](companion_notes/Next_Model_Experiments_for_SPLM.md)
  programme (section C2). Computes the SPLM Hamiltonian energy
  $H_\ell = \tfrac{1}{2}\mathfrak{m}\lVertv_\ell\rVert^{2} + V_\theta(\xi_\ell, h_\ell)$
  at every layer of an SPLM forward pass and reports the linear drift
  slope $\partial H/\partial \ell$ across depth and the oscillation
  bandwidth $\max_\ell H_\ell - \min_\ell H_\ell$ around the layer-mean.
  The expectation, derived directly from the integrator class, is
  three-way separable: a **velocity-Verlet** flow ($L=16,\Delta t=0.5$)
  is symplectic at $\gamma = 0$ and $O(\Delta t^4)$-bounded in energy
  at finite damping, so $H_\ell$ should oscillate around an
  exponentially-damped envelope; an **explicit Euler** flow ($L=8$) is
  first-order and should exhibit a systematic drift growing linearly
  with depth; a **transformer** is not derived from any potential and
  should show no structure at all when its hidden-state flow is forced
  through the same diagnostic with a fitted $V_\psi$ proxy. Ships
  [`extract_energy_states.py`](notebooks/conservative_arch/energy_drift/extract_energy_states.py)
  (re-runs the SPLM forward pass on the §1 e-init test corpus and
  saves $(h_\ell, v_\ell, V_\theta(\xi_\ell, h_\ell), \tfrac{1}{2}m\lVertv_\ell\rVert^2)$
  per layer for one checkpoint at a time; supports parent-SPLM Euler,
  `sarf_mass_variant` Euler + per-token mass, `symplectic_variant`
  velocity-Verlet, and the production `energetic_minima/model_ln.py`
  Euler + per-token mass + LayerNorm-after-step) and
  [`energy_drift_diagnostic.py`](notebooks/conservative_arch/energy_drift/energy_drift_diagnostic.py)
  (per-variant linear drift fit with 95 % CI, oscillation-bandwidth
  tabulation, overlay plots of $H_\ell$, $\tfrac{1}{2}m\lVertv\rVert^2$ and
  $V_\theta$, and a markdown comparison report). The diagnostic is
  forward-pass-only on existing checkpoints and complements the
  *fixed-point* analysis of [`attractor_analysis/`](notebooks/conservative_arch/attractor_analysis/)
  (where are the basins of $V_\theta$?) and the *depth-axis existence*
  question of `shared_potential_fit.py` (is there *some* scalar
  potential whose gradient explains the layer updates?) with a
  *flow-conservation* analysis (does the *learned* $V_\theta$ obey the
  conservation law it was designed to?). The in-folder
  [`energy_drift/README.md`](notebooks/conservative_arch/energy_drift/README.md)
  documents the variant flags, the comparison output layout, and the
  expected energy-drift signatures for each integrator. The production
  E3 run ships under
  [`energy_drift/results/E3_splm_em_ln_compare/`](notebooks/conservative_arch/energy_drift/results/E3_splm_em_ln_compare/)
  and is a 3-way comparison `parent_euler_L8` × `verlet_L16_dt05` ×
  `em_ln_L8_seed0` (the LayerNorm-after-step production SPLM, val ppl
  88.63 at seed 0). The originally-planned `sarfmass logfreq` (no-LN)
  column was dropped because the multi-seed E1 sweep falsified its
  stability (2/3 NaN-divergent seeds at modest gradient norms); using a
  single-seed energy trace from a model the rest of the repo has
  invalidated would not be informative. The headline finding is that
  `em_ln` uses the explicit-Euler integrator yet exhibits a Verlet-like
  energy-conservation signature: bandwidth-to-scale ratio $7.0 / 10.0 =
  70\%$, versus $145.7 / 76.6 = 190\%$ for the bare Euler model and
  $91.4 / 205.5 = 45\%$ for the genuine Verlet integrator. The
  mechanism is the LayerNorm projection
  $h_{l+1} \leftarrow \mathrm{LN}(h_l + \Delta tv_{l+1})$, which clips
  the trajectory's dynamic range without contributing any potential
  gradient; the production SPLM is consequently *not* a clean
  Hamiltonian flow but a "cheating" symplectic integrator whose
  stability comes from compactification of the state space rather than
  from symplectic structure of the integrator. See
  [`E3_splm_em_ln_compare_report.md`](notebooks/conservative_arch/energy_drift/results/E3_splm_em_ln_compare/E3_splm_em_ln_compare_report.md)
  for the full per-variant table, overlay figures, and caveats.

---

## Reproducing the paper's experiments

### 0. Prerequisites

```bash
# 1. Clone and enter the repository
git clone https://github.com/dimitarpg13/semsimula-paper.git
cd semsimula-paper

# 2. Create and activate a Python 3.10+ virtual environment
python -m venv .venv && source .venv/bin/activate   # Linux / macOS
# python -m venv .venv && .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt
# For GPU training, replace the torch line first:
#   pip install torch==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# 4. Pull Git LFS artefacts (trajectory pickles, checkpoints, figures)
git lfs pull
# Without this step, large files under notebooks/conservative_arch/results/
# appear as short text pointers rather than real binary data.
```

All results committed to this repository were produced on a **MacBook Pro,
Intel Core i9 2.3 GHz (8-core), 64 GB RAM**, running Python 3.12.11,
between 18 April and 25 April 2026. GPU is **not required**; every script
runs on CPU (or MPS on Apple Silicon). The shipped `results/` artefacts
mean that any figure in the paper can be replotted without retraining.

---

### §13 — descriptive experiments (Results 1–5)

```bash
# Results 1–4 and Figures 4–6: STP–acceleration identity and
# Gaussian-well analysis on GPT-2 small.
jupyter lab notebooks/stp_loss/energy_landscape_validation.ipynb
# Static rendering with all outputs pre-executed:
#   notebooks/stp_loss/energy_landscape_validation_executed.ipynb

# Result 5: cross-architecture replication on GPT-2 small + Pythia-160M.
jupyter lab notebooks/cross_model/pythia_tangential_acceleration.ipynb
```

Expected runtime: **5–15 minutes** per notebook on the reference hardware
(GPT-2 small hidden-state extraction over a ~200-sentence corpus is the
bottleneck; the Pythia notebook adds a second model load).

---

### §14.1 — negative-results chain on attention transformers (E1–E5)

```bash
# Baseline Gaussian-well E-init on GPT-2 (prerequisite for E1–E5):
jupyter lab notebooks/e_init/e_init_validation.ipynb

# E1 (damping sweep) + E2 (first-order gradient flow):
python notebooks/e_init/extended_gamma_and_first_order.py

# E3 (seven scalar-well functional forms):
python notebooks/e_init/well_functional_form_comparison.py

# E4 (linear Helmholtz position-coupled skew augmentation):
python notebooks/e_init/helmholtz_curl_augmented.py

# E5 (velocity-coupled gauge, constant / affine-rank-1 / affine-rank-2):
python notebooks/e_init/velocity_coupled_gauge.py
```

See [`notebooks/e_init/README.md`](notebooks/e_init/README.md) for the
exact command sequence and the mapping from experiment IDs E1–E5 to
scripts. Each script writes a markdown summary, an `.npz` result archive,
and one or more figures to `notebooks/e_init/results/`.

Expected runtime: **2–10 minutes** per script on the reference hardware.

---

### §14.2 ff. and Appendix A — prescriptive experiments (SPLM pipeline)

The full end-to-end pipeline is documented step by step in
[`notebooks/conservative_arch/README.md`](notebooks/conservative_arch/README.md).
The quick summary:

```bash
cd notebooks/conservative_arch

# 1. Train SPLM and the scale-matched attention baseline
python train_splm.py
python train_matched.py

# 2. Extract hidden-state trajectories from all three models
python trajectory_extraction.py          # SPLM
python extract_matched_baseline.py       # matched GPT-2
python extract_gpt2_baseline.py          # pretrained GPT-2 small

# 3. Run the full diagnostic suite
python shared_potential_fit.py           # strict shared-V_psi separator
python jacobian_symmetry.py              # velocity-aware Jacobian-symmetry test
python sharedV_capacity_sweep.py         # 6-config V_psi capacity band
python splm_oracle_fit.py                # oracle upper bound (SPLM's own V_theta)
python token_direction_fit.py            # token-direction replication

# 4. Produce paper figures
python plot_three_way_comparison.py      # Fig. 8: SPLM / matched / pretrained GPT-2
python plot_token_vs_layer_three_way.py  # token-direction two-panel figure
python plot_sharedV_comparison.py        # shared-V_psi profile plot
```

**Shortcut:** checkpoints and trajectory pickles are shipped via Git LFS
in `results/`. Steps 1–2 can be skipped and steps 3–4 run in
**~5 minutes** on the reference hardware using the precomputed artefacts.
End-to-end including training takes **~35–45 minutes** (SPLM ~20 min,
matched baseline ~15 min, extraction ~5 min, diagnostics ~5 min).

#### SARF-faithful ablation (§14.13)

```bash
cd sarf_variant
python train_splm_sarf.py
python trajectory_extraction_sarf.py
python compare.py
# Re-runs shared_potential_fit.py and token_direction_fit.py from the
# parent directory automatically; outputs go to ../results/.
```

#### Per-token semantic-mass ablation (§14.14)

```bash
cd sarf_mass_variant
python compute_unigram_frequencies.py    # build frozen surprisal lookup (once)
python train_splm_sarf_mass.py --mass-mode global
python train_splm_sarf_mass.py --mass-mode embed_head
python train_splm_sarf_mass.py --mass-mode logfreq
python trajectory_extraction_sarf_mass.py
python compare.py
```

#### Attractor analysis (§14.15)

```bash
cd attractor_analysis
python attractor_extraction.py --mode gradient    # Adam descent on V_theta
python attractor_extraction.py --mode dynamical   # damped Euler from random seeds
python compare_landscapes_3d.py                    # Euler-vs-Verlet 3D comparison
python train_with_snapshots.py                     # retrain with log-spaced checkpoints
python render_training_evolution.py                # seven-panel landscape-evolution grid
```

See [`notebooks/conservative_arch/attractor_analysis/README.md`](notebooks/conservative_arch/attractor_analysis/README.md)
for the per-prompt JSON/PNG/MD output catalogue.

#### Energetic-minima alternatives (F5 of §14.17)

```bash
cd energetic_minima
python train.py --variant ln   # LayerNorm-after-step
python train.py --variant sg   # scale-gauge regulariser
python train.py --variant gm   # Gaussian-mixture V_theta head
bash run_attractor_pipeline.sh  # attractor extraction over all four checkpoints
python compare.py               # produces results/comparison_report.md
python make_compare_figure.py   # produces results/landscape3d_compare_four_variants_dialogue.png
```

See [`notebooks/conservative_arch/energetic_minima/README.md`](notebooks/conservative_arch/energetic_minima/README.md)
for variant flags, training schedule, and expected outputs.

#### Multi-seed variance harness (E1 of `Next_Model_Experiments_for_SPLM.md`)

```bash
cd notebooks/conservative_arch

# 0. (Once) Precompute the surprisal lookup table for SPLM logfreq mass.
python sarf_mass_variant/compute_unigram_frequencies.py

# 1. Smoke test (single seed, ~1-2 minutes total).
python multi_seed/multi_seed_runner.py \
    --mode smoke --n-seeds 1 --models splm_sarfmass_logfreq

# 2. E1 production: 5 seeds x 3 models on Tiny Shakespeare
#    (~7-8 hours wall-clock on Apple MPS; runs sequentially).
python multi_seed/multi_seed_runner.py \
    --mode shakespeare --n-seeds 5 \
    --models splm_em_ln,splm_sarfmass_logfreq,matched_baseline

# 3. Aggregate logs into report + overlay plots + divergence diagnostic.
python multi_seed/multi_seed_aggregator.py --tag E1_shakespeare
python multi_seed/e1_divergence_diagnostic.py --tag E1_shakespeare
```

The shipped `results/E1_shakespeare/` includes 13 per-seed checkpoints,
training logs, and loss curves under Git LFS (5 seeds for
`matched_baseline` and `splm_em_ln`, 3 for `splm_sarfmass_logfreq`
before the divergence-rate diagnostic short-circuited the sweep), plus
the curated [`E1_report.md`](notebooks/conservative_arch/multi_seed/results/E1_shakespeare/E1_report.md)
narrative. Re-running step 3 alone reproduces the mean / std /
Welch-t table and the overlay figures from the shipped per-seed logs in
**~20 seconds**. See
[`notebooks/conservative_arch/multi_seed/README.md`](notebooks/conservative_arch/multi_seed/README.md)
for the model-spec interface and the recipe for adding new variants
(E2 width sweep, E3 integrator ablation, etc.).

#### Energy-drift diagnostic (E3 of `Next_Model_Experiments_for_SPLM.md`)

The production E3 comparison is `parent_euler_L8` × `verlet_L16_dt05` ×
`em_ln_L8_seed0` (LayerNorm-after-step SPLM, val ppl 88.63 at seed 0,
the production-best variant of the multi-seed E1 sweep). The originally
planned `sarfmass logfreq` (no-LN) column is omitted: E1 multi-seed
falsified its stability (2/3 NaN-divergent seeds), so a single-seed
energy trace from it is not representative.

```bash
cd notebooks/conservative_arch

# 1. Extract energy states for the three production SPLM checkpoints.
python energy_drift/extract_energy_states.py \
    --variant euler \
    --ckpt results/splm_shakespeare_ckpt_latest.pt \
    --label splm_euler_L8 \
    --out_npz energy_drift/results/splm_euler_L8.npz

python energy_drift/extract_energy_states.py \
    --variant symplectic \
    --ckpt symplectic_variant/results/splm_sym_logfreq_shakespeare_L16_dt05_ckpt_latest.pt \
    --label splm_verlet_L16_dt05 \
    --logfreq sarf_mass_variant/results/logfreq_surprisal.npy \
    --out_npz energy_drift/results/splm_verlet_L16_dt05.npz

python energy_drift/extract_energy_states.py \
    --variant em_ln \
    --ckpt multi_seed/results/E1_shakespeare/splm_em_ln/seed_0/em_ln_shakespeare_ckpt_latest.pt \
    --label splm_em_ln_L8_seed0 \
    --logfreq sarf_mass_variant/results/logfreq_surprisal.npy \
    --out_npz energy_drift/results/splm_em_ln_L8_seed0.npz

# 2. Cross-variant comparison: drift slope, oscillation bandwidth,
#    overlay plots of H_l, kinetic, and potential per layer.
python energy_drift/energy_drift_diagnostic.py \
    --inputs splm_euler_L8.npz,splm_verlet_L16_dt05.npz,splm_em_ln_L8_seed0.npz \
    --tag E3_splm_em_ln_compare
```

The full pipeline finishes in under three minutes on MPS. The diagnostic
is forward-pass-only on existing checkpoints; no retraining is required.
See
[`notebooks/conservative_arch/energy_drift/README.md`](notebooks/conservative_arch/energy_drift/README.md)
for the variant flags, the expected drift signatures for each
integrator, the production-result interpretation, and the relationship
to `shared_potential_fit.py` and `attractor_analysis/`. If you want to
run the diagnostic on the original `sarfmass logfreq` no-LN checkpoint
as an ablation, the syntax is documented in the in-folder README; any
conclusion drawn from such a single-seed run must be qualified by the
E1 multi-seed instability finding.

---

## Citing this work

See [`CITATION.bib`](CITATION.bib) for the full BibTeX file. The short form:

```bibtex
@misc{Gueorguiev2026SemSim,
  author    = {Gueorguiev, Dimitar P.},
  title     = {Semantic Simulation: A Prescriptive {L}agrangian Framework
               for Efficient Semantic Inference --- Conservative-by-
               Construction Language Models and the Shared-Potential
               Separator, with a Correspondence to Joint Embedding
               Predictive Architectures},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19819861},
  url       = {https://doi.org/10.5281/zenodo.19819861},
  note      = {Companion code repository (DOI 10.5281/zenodo.19708205):
               \url{https://github.com/dimitarpg13/semsimula-paper}}
}
```

The companion code archive (this repository, tagged `v1.0-arxiv`) has its own
citable DOI: [10.5281/zenodo.19708205](https://doi.org/10.5281/zenodo.19708205).
If you re-run the experiments or build on the code specifically, please cite
both DOIs.

If you use or build on material from this companion repository specifically
(e.g., re-run the notebook, cite one of the background manuscripts), please
also cite the paper above as the canonical source of the framework.

---

## License

This repository is released under a **dual license** that reflects the
difference in character between its code and its prose content.

**Code.** The Jupyter notebooks under `notebooks/`, the Python scripts,
`requirements.txt`, and `pyproject.toml` are licensed under the MIT
License; see [`LICENSE`](LICENSE).

**Prose content.** The manuscripts under `manuscripts/`, the companion notes
under `companion_notes/`, this `README.md`, and `CITATION.bib` are licensed
under the Creative Commons Attribution 4.0 International License (CC BY 4.0);
see [`LICENSE-CC-BY-4.0`](LICENSE-CC-BY-4.0). A human-readable summary of
CC BY 4.0 is available at
[creativecommons.org/licenses/by/4.0](https://creativecommons.org/licenses/by/4.0/).

Both licenses allow broad reuse, including commercial use and the creation of
derivative works, and require only that the original author be credited.
When reusing any material from this repository, please cite the paper (see
[`CITATION.bib`](CITATION.bib)) as the canonical source of the framework.

---

## Open items (to resolve before public release)

1. **arXiv identifier.** The paper is currently available as a Zenodo preprint
   ([10.5281/zenodo.19819861](https://doi.org/10.5281/zenodo.19819861)).
   Once submitted to arXiv, fill in the arXiv identifier and propagate to both
   `README.md` and `CITATION.bib`.
