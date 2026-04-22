# Semantic Simulation — Paper companion

Companion repository for the paper

> **Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference**
> *Conservative-by-Construction Language Models and the Shared-Potential Separator, with a Correspondence to Joint Embedding Predictive Architectures.*
> Dimitar P. Gueorguiev (Independent Researcher), 2026.
> arXiv: `TODO: arXiv:XXXX.XXXXX` (link to be added after submission).

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
| `Conservative_by_Construction_Language_Models.md`                                   | §14 (motivation for SPLM)                 |
| `Considered_Non-Autonomous_Conservative_Mechanisms.md`                              | Appendix A (non-autonomous framework)     |
| `On_Modeling_Semantic_Energy_Field_into_SPLM.md`                                    | §14.2 (mapping framework energy field onto $V_\theta$, $\xi$, $m_t$; candidate Q11–Q13) |
| `Training_and_Inference_with_SPLM.md`                                               | §14.2, §14.13, §14.14 (training loop, nested-autograd force computation, inference pipeline, and summary of the fixed-$\xi$ / SARF-faithful / per-token-mass ablations) |

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
  $F(x)\,\dot x$ with constant, affine-rank-1, and affine-rank-2 $F$).

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
  efficiency result flagged as the Q10 open follow-up in §14.16 and
  §16. All four training logs, checkpoints, trajectory pickles, and
  per-layer diagnostic tables live under
  `sarf_mass_variant/results/`; `comparison_*.png` are mirrored at
  the folder root for direct figure inclusion in the paper.

---

## Reproducing the paper's experiments

> **TODO — fill in before release.** Concretely:
>
> 1. Clone this repo and create a fresh Python environment (e.g., `python -m
>    venv .venv && source .venv/bin/activate`).
> 2. Install dependencies. A `requirements.txt` will be added before the
>    companion repo goes public; key dependencies are `torch`,
>    `transformers`, `numpy`, `scipy`, `matplotlib`, `pandas`, `jupyter`.
> 3. Pull the Git LFS artefacts: `git lfs pull`. Without this step, the
>    large trajectory pickles and checkpoints under
>    `notebooks/conservative_arch/results/` will appear as short pointer
>    files rather than the real binary data.
>
> **§13 — descriptive experiments (Results 1–5).**
>
> 4. Launch `jupyter lab` and open one of:
>    - `notebooks/stp_loss/energy_landscape_validation.ipynb` (GPT-2
>      STP-acceleration and Gaussian-well analysis, backing Results 1–4
>      and Figures 4–6 in §13);
>    - `notebooks/cross_model/pythia_tangential_acceleration.ipynb`
>      (cross-architecture replication on GPT-2 and Pythia-160M, backing
>      Result 5 in §13).
>
> **§14.1 — negative-results chain on attention transformers.**
>
> 5. Run the notebook then the four companion scripts under
>    `notebooks/e_init/` (see
>    [`notebooks/e_init/README.md`](notebooks/e_init/README.md) for the
>    exact commands and the mapping from experiment IDs E1–E5 to scripts).
>    Each script writes a markdown summary, an `.npz` of numerical
>    results, and figures to `notebooks/e_init/results/`.
>
> **§14.2 ff. and Appendix A — prescriptive experiments.**
>
> 6. Follow the step-by-step pipeline in
>    [`notebooks/conservative_arch/README.md`](notebooks/conservative_arch/README.md):
>    train SPLM and the matched GPT-2 baseline, extract hidden-state
>    trajectories (SPLM, matched GPT-2, pretrained GPT-2 small), then
>    run the shared-$V_{\psi}$ fit, Jacobian-symmetry test, capacity
>    sweep, oracle fit, and token-direction replication. The separator
>    figure (SPLM vs.\ matched GPT-2 vs.\ pretrained GPT-2, median per-
>    layer $R^{2} \in \{0.90, 0.45, 0.56\}$) is produced by
>    `plot_three_way_comparison.py`; the token-direction two-panel figure
>    by `plot_token_vs_layer_three_way.py`. Full checkpoints and
>    trajectories are shipped under `results/` (via Git LFS) so that any
>    result in §14 and Appendix A can be **replotted without retraining**.
>
> Expected end-to-end runtime on an M-series Mac with 16 GB unified memory:
> §13 notebooks **TODO (measure before release)**; §14.1 e_init scripts
> 2–10 minutes each; §14.2 ff. conservative_arch pipeline ~30–40 minutes
> including SPLM training and trajectory extraction, or ~5 minutes for
> fit-and-plot only when starting from the shipped checkpoints. GPU is not
> required; everything runs on CPU or MPS.
>
> Each notebook / script writes its outputs into the sibling `results/`
> subfolder. The versions already committed there were produced by the
> author on **TODO (date + hardware)** and are included so that every
> figure in the paper can be inspected without re-executing the pipeline.

---

## Citing this work

See [`CITATION.bib`](CITATION.bib) for a BibTeX entry. The short form:

```bibtex
@article{Gueorguiev2026SemSim,
  author  = {Gueorguiev, Dimitar P.},
  title   = {Semantic Simulation: A Prescriptive Lagrangian Framework for
             Efficient Semantic Inference --- Conservative-by-Construction
             Language Models and the Shared-Potential Separator, with a
             Correspondence to Joint Embedding Predictive Architectures},
  journal = {arXiv preprint},
  volume  = {TODO: arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

If you use or build on material from this companion repository specifically
(e.g., re-run the notebook, cite one of the background manuscripts), please
also cite the paper above as the canonical source of the framework.

---

## License

This repository is released under a **dual license** that reflects the
difference in character between its code and its prose content.

**Code.** The Jupyter notebooks under `notebooks/` and any configuration
files (e.g., a future `requirements.txt` or `environment.yml`) are licensed
under the MIT License; see [`LICENSE`](LICENSE).

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

1. **arXiv identifier.** Fill in once v1 is submitted; propagate to both
   `README.md` and `CITATION.bib`.
2. **`requirements.txt`.** Produce from the actual environment used to run
   the notebooks; commit with pinned versions.
3. **Reproducibility runtime.** Measure and document notebook runtime and
   hardware.
4. **Zenodo archive (optional).** If you want a DOI for citation stability,
   mint one via Zenodo once the repo is public and tag the first release as
   `v1.0-arxiv`.
