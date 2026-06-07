# Semantic Simulation — Paper companion

Companion repository for the paper

> **Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference**
> *A Conservative-by-Construction Language Model and the Shared-Potential Separator, with a Correspondence to Joint Embedding Predictive Architectures.*
> Dimitar P. Gueorguiev (Independent Researcher), 2026.
> Zenodo preprint (v14, Jun 7 2026): [10.5281/zenodo.20531415](https://doi.org/10.5281/zenodo.20531415). Supersedes v13 ([10.5281/zenodo.20478543](https://doi.org/10.5281/zenodo.20478543), May 31 2026).
> Companion code latest release: **v4.9** (Jun 7, 2026) — Fock v2.1 routing-fix results (B1+B2+B3 = PPL 9.30, new best, surpasses Fock Attention 9.42 at O(1) memory). Adds B1/B2/B3 mechanism illustrations; rewrites §19.5 with full per-fix intuition and super-additivity analysis; adds §19.9 with complete v2.1 results table and 7 key findings; routing deficit → routing surplus confirmed. Rebuilt `semsimula_paper.pdf` (298 pages) and `semsimula_paper_source.zip`. Supersedes v4.8 (May 31, 2026).

[![DOI — paper](https://zenodo.org/badge/DOI/10.5281/zenodo.20531415.svg)](https://doi.org/10.5281/zenodo.20531415)
[![DOI — companion code](https://zenodo.org/badge/DOI/10.5281/zenodo.20577388.svg)](https://doi.org/10.5281/zenodo.20577388)

This repository collects the **reproducibility artifacts** and the **unpublished
background manuscripts** cited in the paper. Its scope is deliberately narrow:
everything here is either directly required to resolve a citation in the paper
or needed to reproduce a figure or experimental claim. The repository covers
both the descriptive experiments of §13 (STP–acceleration identity and the
Pythia / GPT-2 deceleration analysis) and the prescriptive experiments of §14
and Appendix A (the negative-results chain on attention transformers, the
scalar-potential language model, and the three-way shared-potential
separator).

> **Git LFS.** Several PNG figures, NPZ result archives, NPY frozen
> surprisal tensors, and a couple of GIF landscape rotations under
> `notebooks/conservative_arch/` are stored via **Git LFS** for
> bandwidth economy. After cloning, run `git lfs pull` once to
> download them; without this step the large files will appear as
> short text pointers. Git LFS is not required for the v1 artefacts
> under `notebooks/stp_loss/` and `notebooks/cross_model/`.

> **Checkpoint policy (v3 release, May 2026).** Following the
> causal-leak audit (see the *v3 update* section below), **no model
> checkpoints (`*.pt`) or hidden-state trajectory pickles (`*.pkl`)
> are committed** to this repository. Every SPLM checkpoint that
> existed in earlier releases was trained under the v2 leaky
> integrator and is therefore an empirical casualty of the bug; the
> matched-GPT-2 / pretrained-GPT-2 trajectory pickles existed only to
> back the three-way separator comparison against those SPLMs and
> have no standalone reproducibility value. All claims in the
> paper that previously cited a shipped checkpoint are now backed by
> training logs, summary markdown, and loss-curve PNGs that *are*
> committed. Reproducing any v2 ablation or any v3 R6-ladder cell
> from scratch requires re-running the relevant
> `train_*.py` script — runtimes are documented per cell in the
> reproduction recipe sections below.

> **Rendering note.** Several markdown files under `companion_notes/` in this repository contain LaTeX math (inline `$...$` and display `$$...$$` blocks, with macros such as `\mathfrak{...}`, `\boldsymbol{...}`, `\mathcal{...}`, etc.). The math has been verified to render correctly in **Safari**. In **Chrome** some symbols — notably calligraphic and fraktur letters, e.g. `\mathfrak{C}` rendering as a plain `C` instead of $\mathfrak{C}$ — appear to render incorrectly. **Firefox** has not been tested. If symbols look wrong while viewing a companion note on GitHub, please open the file in Safari or consult the main paper's PDF, where the same symbols are typeset by LaTeX directly. Each affected companion note repeats this warning in its own header.

> **AI-assisted research disclosure.** This work was developed through an
> extensive human--AI collaborative workflow using Anthropic's Claude. A full
> disclosure of the domains of AI contribution and the boundaries of human
> scientific judgment is provided in
> [`AI_Assisted_Collaborative_Research_Disclosure.md`](AI_Assisted_Collaborative_Research_Disclosure.md).

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

Working notes developed alongside the paper. They capture material the paper does not subsume — either summarised briefly in the main text with a pointer here, or deferred to future work. Each note's header identifies the paper sections it backs.

#### Notes cited by BibTeX key

| File | Role |
|------|------|
| `On_the_Interpretation_of_Semantic_Mass.md` | §11 — physical interpretation of per-token mass in the Lagrangian framework. |
| `On_the_Interpretation_of_Hidden_State.md` | §10 — hidden-state ontology: phase-space coordinates vs. latent features. |
| `On_The_Existence_of_Acceleration_in_Semantic_Structures.md` | §12 — empirical evidence for deceleration and the STP–acceleration identity. |
| `STP_Loss_Is_An_Emergent_Property_Of_The_Energy_Landscape_Defined_By_Gaussian_Well_Potential.md` | §12 — STP loss as an emergent property of the Gaussian-well energy landscape. |
| `The_Execution_Problem.md` | §8.6 — deferred treatment of structure execution in semantic simulation. |

#### Notes backing §14 / Appendix A (not BibTeX-cited)

| File | Role |
|------|------|
| `The_Failure_of_Conservative_Models_to_explain_hidden_state_trajectories.md` | §14.1 — why conservative models fail on pretrained transformer trajectories. |
| `P-rot-6_transformer_dynamics.md` | §14.1 — derives the E5 zero-parameter prediction from K≠V antisymmetry. |
| `Conservative_by_Construction_Language_Models.md` | §14 — motivation and design rationale for SPLM. |
| `Considered_Non-Autonomous_Conservative_Mechanisms.md` | Appendix A — non-autonomous conservative framework alternatives. |
| `Addendum_Non_Autonomous_Fields_For_Appendix_A.md` | Appendix A — Class F equation, Hopfield / Tracks A–B, integrability guide. |
| `On_Modeling_Semantic_Energy_Field_into_SPLM.md` | §14.2 — mapping the framework energy field onto V_θ, ξ, m_t. |
| `On_The_Smoothness_of_Scaled_Dot_Product_Attention.md` | §14, Theorem 46 — smoothness of attention; Poincaré prerequisites. |
| `Training_and_Inference_with_SPLM.md` | §14.2, §14.13 — training loop, nested-autograd forces, inference pipeline. |

#### Forthcoming-work planning and expressivity bounds

| File | Role |
|------|------|
| `Semantic_Simulator_RL_Calibration_Programme.md` | Programme-level memo for the deferred RL-calibrated simulator (§8.8, §16). |
| `Semantic_Simulator_EOM.md` | v0 equations of motion, parameter classification, pseudocode (§8.8, §16). |
| `Expressivity_Bounds_For_v0_Simulator.md` | Formal proof: v0 simulator accepts at most regular languages (§16). |
| `MCS_Reduction_For_v3_Composite.md` | Formal proof: v0+v1.5+v2+v3 composite generates exactly the MCS class (§16). |
| `Advancing_The_Dynamic_Simulation_Model.md` | Conceptual scaffold mapping v1.5/v2/v3 onto mature mathematical apparatus. |
| `Next_Model_Experiments_for_SPLM.md` | Prioritised experiment catalogue; source of truth for the E1 and E3 programmes. |
| `Semantic_Simulator_v15_EOM.md` | Forthcoming — v1.5 dynamics (dissipative semigroups). |
| `Semantic_Simulator_v2_EOM.md` | Forthcoming — v2 dynamics (Fock-space second quantisation). |
| `Semantic_Simulator_v3_EOM.md` | Forthcoming — v3 dynamics (non-abelian gauge theory). |

#### v3 leak-correction and information-bottleneck programme (May 2026)

| File | Role |
|------|------|
| `Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md` | Forensic detail of the v2 anti-causal autograd path, the `h.detach()` fix, and the `causal_probe.py` regression-test framework. |
| `Causal_Leak_Empirical_Comparison_Report.md` | Closed-loop leak-free re-evaluation of every v2 SPLM checkpoint (headline: 777× inflation for TinyStories `splm_em_ln_multixi`). |
| `Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md` | The R6 information-bottleneck programme: K-EMA / HiPPO-LegT / S4D basis experiments and four information-theoretic diagnostics. |
| `Determining_optimal_gamma_for_SPLM.md` | Four-estimator γ*-prediction framework and the resonance-predictor double success across the leak-correction boundary. |
| `Energetic_Minima_Alternatives.md` | Leak-free retrains of the three energetic-minima alternatives (LN / SG / GM); recommendation shift from LN to scale-gauge when attractor diversity matters. |
| `Semantic_Attractor_Extraction.md` | Leak-free dynamical-mode attractor extractions (Tiers 1 + 2b); F3 multi-basin structure survives the fix. |

#### PARFLM and FockPARFLM programme

| File | Role |
|------|------|
| `PARF_Augmented_SPLM_Architecture_v2.md` | Architecture design for the PARFLM V_φ augmentation (Q9c path). |
| `PARF-SPLM_Path_Forward_and_Experiments.md` | Experiment plan for the P1–P10 PARF scale-up ladder. |
| `Augmenting_PARFLM_to_handle_MCS_Languages.md` | Fock-space augmentation design for v2 (context-free) expressivity. |
| `Improving_the_Fock_Mechanism_to_match_Attention.md` | Fock Attention experiments, register diagnostics, causal leak discovery and fix, routing resolution hierarchy. |
| `Fock_PARFLM_Conservativity_Diagnostic.md` | Five-arm diagnostic battery proving the conservative-by-construction property of FockPARFLM v2.1: Jacobian symmetry, ablation R², energy budget decomposition, conservativity dial, four-way separator. |

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed discussion of each experiment and its results.

### `notebooks/` — reproducibility

Each notebook or folder below is summarised in one or two sentences. See [COLAB_NOTEBOOKS.md](COLAB_NOTEBOOKS.md) for full descriptions, Colab links, and runtime estimates. See [EXPERIMENTS.md](EXPERIMENTS.md) for per-experiment methodology and results.

| Notebook / folder | Role |
|-------------------|------|
| `stp_loss/energy_landscape_validation.ipynb` | §13 — GPT-2 STP–acceleration analysis and Gaussian-well validation. A pre-executed version with all outputs is included. |
| `cross_model/pythia_tangential_acceleration.ipynb` | §13 Result 5 — cross-architecture deceleration replication on GPT-2 small and Pythia-160M. |
| `e_init/` | §14.1 — five negative experiments (E1–E5) on scalar, Helmholtz, and gauge-field fits on pretrained GPT-2 trajectories. See [`e_init/README.md`](notebooks/e_init/README.md). |
| `conservative_arch/` | §14.2 ff. and Appendix A — SPLM prototype, three-way shared-potential separator, and all prescriptive experiments. See [`conservative_arch/README.md`](notebooks/conservative_arch/README.md). |
| `conservative_arch/sarf_variant/` | §14.13 — SARF-faithful SPLM ablation (33% perplexity reduction via single-line context-pool change). |
| `conservative_arch/sarf_mass_variant/` | §14.14 — per-token semantic mass ablation (44% perplexity reduction with framework-prescribed surprisal mass). |
| `conservative_arch/attractor_analysis/` | §14.15 — prompt-dependent semantic attractor extraction via damped dynamics and gradient descent. |
| `conservative_arch/energetic_minima/` | §14.17 Q11 — three structural alternatives to free V_θ (LN-after-step, scale-gauge, Gaussian-mixture). |
| `conservative_arch/multi_seed/` | E1 multi-seed variance harness — 5-seed runs of three architectures on Tiny Shakespeare with Welch t-tests. |
| `conservative_arch/energy_drift/` | E3 energy-drift diagnostic — per-layer Hamiltonian energy and conservation-bandwidth analysis across integrator types. |
| `conservative_arch/multixi/` | R6 ladder — multi-channel-ξ model implementations (K-EMA / HiPPO-LegT / S4D) and channel-correlation diagnostics. |
| `conservative_arch/scaleup/` | R6 TinyStories scale-up — ~16M-parameter training scripts and per-pilot result logs for the R6 ladder. |
| `conservative_arch/first_order_ablation/` | SPLM-1 first-order ablation — pre-registered v2 baseline and leak-free 3-seed retrain with forensic re-eval. |
| `conservative_arch/ln_damping_sweep/` | Controlled-γ damping sweep — v2 6-cell sweep, leak-free 4-point U-curve, and 5-seed S=5 confirmation sweep. |
| `conservative_arch/helmholtz/` | Q9d — Helmholtz-SPLM hybrid architecture with Dyck-language expressivity falsification. |
| `conservative_arch/hybrid/` | Variant A — hybrid two-stage SPLM combining frozen-ξ first stage with SARF-faithful second stage. |
| `conservative_arch/parf/` | §17 — PARFLM and FockPARFLM: V_φ pairwise-interaction augmentation with Gumbel-softmax sparse routing and Fock-space registers. Includes the P10 TinyStories ladder (PPL 26.4 architectural ceiling) and Phase 1 Dyck₂ falsifier. |

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

# 4. Pull Git LFS artefacts (PNG figures, NPZ result archives, NPY surprisal
#    tensors, GIF landscape rotations). Note: as of the v3 release no model
#    checkpoints or trajectory pickles are shipped -- see the "Checkpoint
#    policy" notice at the top of this README.
git lfs pull
```

All results committed to this repository were produced on a **MacBook Pro,
Intel Core i9 2.3 GHz (8-core), 64 GB RAM**, running Python 3.12.11,
between 18 April and 25 April 2026. GPU is **not required**; every script
runs on CPU (or MPS on Apple Silicon). The shipped `results/` artefacts
mean that any figure in the paper can be replotted without retraining.

For full reproduction commands — including the §13 descriptive experiments, §14 negative-results chain, SPLM pipeline, multi-seed harness, energy-drift diagnostic, v3 leak-correction audit, R6 ladder, and §17 PARFLM/FockPARFLM experiments — see **[EXPERIMENTS.md](EXPERIMENTS.md)**.

---

## Citing this work

See [`CITATION.bib`](CITATION.bib) for the full BibTeX file. The short form:

```bibtex
@misc{Gueorguiev2026SemSim,
  author    = {Gueorguiev, Dimitar P.},
  title     = {Semantic Simulation: A Prescriptive {L}agrangian Framework
               for Efficient Semantic Inference --- A Conservative-by-
               Construction Language Model and the Shared-Potential
               Separator, with a Correspondence to Joint Embedding
               Predictive Architectures},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20531415},
  url       = {https://doi.org/10.5281/zenodo.20531415},
  note      = {Version v14 (Jun 7, 2026); supersedes v13
               (DOI 10.5281/zenodo.20478543, May 31, 2026).
               Companion code repository (DOI 10.5281/zenodo.20577388):
               \url{https://github.com/dimitarpg13/semsimula-paper}}
}
```

The companion code archive at the v4.9 release has its own citable DOI:
[10.5281/zenodo.20577388](https://doi.org/10.5281/zenodo.20577388).
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
   ([10.5281/zenodo.20531415](https://doi.org/10.5281/zenodo.20531415), v14,
   Jun 7, 2026). Once submitted to arXiv, fill in the arXiv identifier and
   propagate to both `README.md` and `CITATION.bib`.
2. **Forthcoming-work EOM stubs.** `companion_notes/Semantic_Simulator_v15_EOM.md`,
   `Semantic_Simulator_v2_EOM.md`, and `Semantic_Simulator_v3_EOM.md` are
   short placeholders flagged "*Forthcoming*" in their headers. They exist
   so that the §9 / §16 Q8 `\path{...}` references in the paper resolve
   to a real file in the public companion repository. Each will be filled
   in by a future companion paper covering the v1.5 / v2 / v3 dynamics.

## Version history

| Release | Date | Paper DOI | Code DOI | Notes |
|---|---|---|---|---|
| **v4.9** | **Jun 7, 2026** | [10.5281/zenodo.20531415](https://doi.org/10.5281/zenodo.20531415) | [10.5281/zenodo.20577388](https://doi.org/10.5281/zenodo.20577388) | **Fock v2.1 routing-fix results and §19 update.** B1 (per-register temperature), B2 (per-register key subspaces), B3 (orthogonal init) mechanism illustrations added to `companion_notes/images/`. §19.5 rewritten with full intuition for each fix, super-additivity analysis, and embedded diagrams including creation gate overview. §19.9 added with complete v2.1 routing-fix results table (PPL 9.30 with B1+B2+B3, new best — surpasses Fock Attention 9.42 at O(1) memory) and 7 key findings; Q2 and tainted-experiment reruns marked RESOLVED; PPL ladder updated: routing deficit → routing surplus confirmed. Rebuilt `semsimula_paper.pdf` (298 pages) and `semsimula_paper_source.zip`. Supersedes v4.8. |
| **v4.8** | **May 31, 2026** | [10.5281/zenodo.20478543](https://doi.org/10.5281/zenodo.20478543) | [10.5281/zenodo.20478517](https://doi.org/10.5281/zenodo.20478517) | **§17c paragraph reordering for logical flow.** Reordered the five key paragraphs of `17c_fock_parflm.tex` in both paper v4 and v5 to eliminate forward references: (1) Q/K/V creation protocol → (2) asymmetric non-conservative reverse channel (now defines `eq:fock-eom` and $Q_i$ before first use) → (3) transfer mechanism (spring-bundle / force-field picture, refers backward to `eq:fock-eom`) → (4) controlled non-conservatism capstone (shape / magnitude / localisation) → (5) novelty beyond attention (memory lifetime). Light de-duplication of closing sentences. Both paper v4 and v5 build with exit code 0 and 0 undefined references. Rebuilt `semsimula_paper.pdf` (291 pages) and `semsimula_paper_source.zip`. Supersedes v4.7-zenodo. |
| **v4.7-zenodo** | **May 30, 2026** | [10.5281/zenodo.20469626](https://doi.org/10.5281/zenodo.20469626) | [10.5281/zenodo.20469618](https://doi.org/10.5281/zenodo.20469618) | **Fock-PARFLM section and caption fix.** Added cache-free inference paragraph to the introduction motivating Fock registers from a KV-cache-free linear-time decoding perspective. Fixed fragile `\hyperlink` inside `\caption{}` in `17c_fock_parflm.tex` that was corrupting the `.aux` file and causing 15 labels to appear undefined. Both paper v4 and v5 now build with exit code 0 and 0 undefined references. Rebuilt `semsimula_paper.pdf` (291 pages) and `semsimula_paper_source.zip`. Supersedes v4.6-zenodo. |
| v4.6-zenodo | May 27, 2026 | [10.5281/zenodo.20421901](https://doi.org/10.5281/zenodo.20421901) | [10.5281/zenodo.20421845](https://doi.org/10.5281/zenodo.20421845) | Pre-release audit and remediation. Six-priority systematic audit of the 287-page paper v4: fixed critical consistency issues (C-item ranges, orphan refs, edition history, deferred-scope narrowing, section bridges, wrong crefs); added roadmap coverage for sections 16–17c and a sixth intro movement; cleaned ~35 version-specific framing instances; normalised notation (`vh`→`d_V`, bare code identifiers wrapped in `\texttt{}`); fixed LaTeX quality (running header overflow, severe body overflows, table widths, BibTeX entry type); grammar typo. Rebuilt `semsimula_paper.pdf` (287 pages, 0 undefined references) and `semsimula_paper_source.zip`. Supersedes v4.5-zenodo. |
| v4.5-zenodo | May 24, 2026 | [10.5281/zenodo.20370417](https://doi.org/10.5281/zenodo.20370417) | [10.5281/zenodo.20370370](https://doi.org/10.5281/zenodo.20370370) | Second-order framing audit and Experiment A integration. Experiment A (direct trajectory fitting of first-order vs. second-order autonomous ODEs to GPT-2 hidden states) confirms that inference-time dynamics is non-autonomous and effectively first-order at every layer. Systematic audit of 21 passages across 12 LaTeX files to distinguish prescriptive second-order claims (SPLM/PARFLM by construction) from descriptive claims about attention transformers. SP-HSPLM Stage 1 and Stage 2 notebooks and results synced. 3 missing companion notes and 3 missing notebooks added. Rebuilt `semsimula_paper.pdf` (268 pages). Supersedes v4.4-zenodo. |
| v4.4-zenodo | May 24, 2026 | [10.5281/zenodo.20368349](https://doi.org/10.5281/zenodo.20368349) | [10.5281/zenodo.20368266](https://doi.org/10.5281/zenodo.20368266) | Comprehensive source audit and post-audit paper release. Full audit of paper v4: 498 labels, 320 cross-references, 44 figures, 89 citations — all clean (0 unresolved, 0 duplicates, 0 missing). Three typo fixes applied to both paper v4 and v5. Rebuilt `semsimula_paper.pdf` (260 pages) and `semsimula_paper_source.zip` (82 files, build artifacts excluded). |
| **v4.3-zenodo** | **May 22, 2026** | [10.5281/zenodo.20347855](https://doi.org/10.5281/zenodo.20347855) | [10.5281/zenodo.20347828](https://doi.org/10.5281/zenodo.20347828) | **Abstract condensation, keyword expansion, and reference audit.** Condensed the 2,584-word release-log abstract to ~500 words (4 focused paragraphs: framework, descriptive validation, prescriptive contribution, v4 summary). Expanded keyword list from 13 to 35 terms covering energy-based models, state-space models, physics-informed ML, attention alternatives, mechanistic interpretability, and key acronyms (SPLM, PARF, JEPA, STP, HiPPO, S4D). Full audit of all 73 `\path{}` references in paper v4/v5 — 2 stale `docs/` paths corrected to `companion_notes/`; `Augmenting_PARFLM_to_handle_MCS_Languages.md` synced to latest 367-line active version. Added `zenodo_keywords_v4.txt` and `zenodo_keywords_v5.txt` for copy/paste into Zenodo/SSRN upload forms. |
| **v4.2-zenodo** | **May 19, 2026** | [10.5281/zenodo.20289294](https://doi.org/10.5281/zenodo.20289294) | [10.5281/zenodo.20289174](https://doi.org/10.5281/zenodo.20289174) | **Companion-notes cleanup and paper audit.** Removed stale `.docx` files (`Section_15_24_PARF_Augmented_SPLM_v4_draft.docx`, `SPLM_Experiments.docx`); cleaned dangling markdown cross-references in 4 companion notes/READMEs; updated `PARF-SPLM_Path_Forward_and_Experiments.md` scope pointer to `paper_v4/main.tex §17`; rebuilt `semsimula_paper.pdf` from audited paper v4 source with leak-free R² numbers, Pareto-frontier hybrid narrative, and corrected companion-repo paths. |
| **v4.1-zenodo** | **May 12, 2026** | [10.5281/zenodo.20138055](https://doi.org/10.5281/zenodo.20138055) | [10.5281/zenodo.20138172](https://doi.org/10.5281/zenodo.20138172) | **AI disclosure update.** Added dedicated "Statement on AI-Assisted Research" section to paper v4 with explicit enumeration of AI contribution domains; added `AI_Assisted_Collaborative_Research_Disclosure.md` to companion repo; updated README with disclosure callout. |
| v4-zenodo | May 10, 2026 | [10.5281/zenodo.20114821](https://doi.org/10.5281/zenodo.20114821) | [10.5281/zenodo.20114898](https://doi.org/10.5281/zenodo.20114898) | PARFLM P10 ladder complete (architectural ceiling at PPL ≈ 26.4 confirmed); FockPARFLM with Phase 1 Dyck₂ falsifier; warm-start bridge to EOM simulator; attention expressivity limits discussion; table of contents; incremental shard tokenization. |
| v3 | May 3, 2026 | [10.5281/zenodo.20014411](https://doi.org/10.5281/zenodo.20014411) | [10.5281/zenodo.20014131](https://doi.org/10.5281/zenodo.20014131) | Causal-leak audit, leak-corrected re-evaluation of every v2 SPLM result, multi-channel-ξ R6 information-bottleneck programme, reframing of SPLM as a Lagrangian counterfactual. See the *v3 update* block at the top of this README. |
| v2 | April 27, 2026 | [10.5281/zenodo.19819861](https://doi.org/10.5281/zenodo.19819861) | — | Multi-seed E1 release of the paper (also uploaded to SSRN). **Note:** every SPLM perplexity number in this version is an empirical casualty of the causal-leak bug discovered after release; the descriptive findings on pretrained GPT-2 / Pythia survive the fix unchanged. v2 is preserved as a historical record; new readers should land on v3. |
| v1.0-arxiv | April 2026 | — | [10.5281/zenodo.19708205](https://doi.org/10.5281/zenodo.19708205) | First publicly archived snapshot of this companion repository. Superseded by v3 code archive above; the v1.0 record remains accessible as a historical reference. |
