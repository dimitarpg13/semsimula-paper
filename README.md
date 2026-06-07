# Semantic Simulation — Paper companion

Companion repository for the paper

> **Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference**
> *A Conservative-by-Construction Language Model and the Shared-Potential Separator, with a Correspondence to Joint Embedding Predictive Architectures.*
> Dimitar P. Gueorguiev (Independent Researcher), 2026.
> Zenodo preprint (v15, Jun 7 2026): [10.5281/zenodo.20579593](https://doi.org/10.5281/zenodo.20579593). Supersedes v14 ([10.5281/zenodo.20531415](https://doi.org/10.5281/zenodo.20531415), Jun 7 2026).
> Companion code latest release: **v4.10** (Jun 7, 2026) — Comprehensive paper audit: right-margin overflow fixes (24 → 2), "this work" self-reference replacements, companion-repo reference integrity check (80 links, 0 dangling), figure/label audit (56 figures, 0 mismatches). Rebuilt `semsimula_paper.pdf` (308 pages) and `semsimula_paper_source.zip` (32 files, incl. new §18d). Supersedes v4.9 (Jun 7, 2026).

[![DOI — paper](https://zenodo.org/badge/DOI/10.5281/zenodo.20579593.svg)](https://doi.org/10.5281/zenodo.20579593)
[![DOI — companion code](https://zenodo.org/badge/DOI/10.5281/zenodo.20579561.svg)](https://doi.org/10.5281/zenodo.20579561)

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

| BibTeX key | Title | §§ subsumed |
|------------|-------|-------------|
| `Gueorguiev2021TreeOps` | Semantic Tree Operations | §8 |
| `Gueorguiev2022Foundations` | Foundations of Semantic Simulation | §§1–2 |
| `Gueorguiev2022PARF` | Attractive and Repulsive Forces in Semantic Properties | §5 |
| `Gueorguiev2022DynSim` | Need for Dynamic Simulation in Semantic Structures | §6 |
| `Gueorguiev2022Signature` | Signature Matrix of Semantic Property | §3 |
| `Gueorguiev2022SARF` | Attractive and Repulsive Forces between Semantic Structures | §6 |
| `Gueorguiev2022Well` | Gaussian Inverse Semantic Energy Well | §4 |
| `Gueorguiev2022Execution` | Execution of Semantic Structures | §8.6 (summary only) |
| `Gueorguiev2024SemSim` | Semantic Simulation | §§1–2 |
| `Gueorguiev2026Lagrangian` | Constructing the Lagrangian for Semantic Space | §7 |

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

| Note | Role |
|------|------|
| [Semantic Mass interpretation](companion_notes/On_the_Interpretation_of_Semantic_Mass.md) | §11 — physical interpretation of per-token mass in the Lagrangian framework. |
| [Hidden State interpretation](companion_notes/On_the_Interpretation_of_Hidden_State.md) | §10 — hidden-state ontology: phase-space coordinates vs. latent features. |
| [Acceleration in Semantic Structures](companion_notes/On_The_Existence_of_Acceleration_in_Semantic_Structures.md) | §12 — empirical evidence for deceleration and the STP–acceleration identity. |
| [STP loss and Gaussian-well energy landscape](companion_notes/STP_Loss_Is_An_Emergent_Property_Of_The_Energy_Landscape_Defined_By_Gaussian_Well_Potential.md) | §12 — STP loss as an emergent property of the Gaussian-well energy landscape. |
| [The Execution Problem](companion_notes/The_Execution_Problem.md) | §8.6 — deferred treatment of structure execution in semantic simulation. |

#### Notes backing §14 / Appendix A (not BibTeX-cited)

| Note | Role |
|------|------|
| [Failure of conservative models on transformer trajectories](companion_notes/The_Failure_of_Conservative_Models_to_explain_hidden_state_trajectories.md) | §14.1 — why conservative models fail on pretrained transformer trajectories. |
| [P-rot-6 transformer dynamics](companion_notes/P-rot-6_transformer_dynamics.md) | §14.1 — derives the E5 zero-parameter prediction from K≠V antisymmetry. |
| [Conservative-by-construction language models](companion_notes/Conservative_by_Construction_Language_Models.md) | §14 — motivation and design rationale for SPLM. |
| [Non-autonomous conservative mechanisms](companion_notes/Considered_Non-Autonomous_Conservative_Mechanisms.md) | Appendix A — non-autonomous conservative framework alternatives. |
| [Non-autonomous fields addendum](companion_notes/Addendum_Non_Autonomous_Fields_For_Appendix_A.md) | Appendix A — Class F equation, Hopfield / Tracks A–B, integrability guide. |
| [Semantic energy field in SPLM](companion_notes/On_Modeling_Semantic_Energy_Field_into_SPLM.md) | §14.2 — mapping the framework energy field onto V_θ, ξ, m_t. |
| [Smoothness of scaled-dot-product attention](companion_notes/On_The_Smoothness_of_Scaled_Dot_Product_Attention.md) | §14, Theorem 46 — smoothness of attention; Poincaré prerequisites. |
| [Training and inference with SPLM](companion_notes/Training_and_Inference_with_SPLM.md) | §14.2, §14.13 — training loop, nested-autograd forces, inference pipeline. |

#### Forthcoming-work planning and expressivity bounds

| Note | Role |
|------|------|
| [RL calibration programme](companion_notes/Semantic_Simulator_RL_Calibration_Programme.md) | Programme-level memo for the deferred RL-calibrated simulator (§8.8, §16). |
| [Simulator v0 EOM](companion_notes/Semantic_Simulator_EOM.md) | v0 equations of motion, parameter classification, pseudocode (§8.8, §16). |
| [v0 expressivity bounds](companion_notes/Expressivity_Bounds_For_v0_Simulator.md) | Formal proof: v0 simulator accepts at most regular languages (§16). |
| [MCS reduction for v3 composite](companion_notes/MCS_Reduction_For_v3_Composite.md) | Formal proof: v0+v1.5+v2+v3 composite generates exactly the MCS class (§16). |
| [Advancing the simulation model](companion_notes/Advancing_The_Dynamic_Simulation_Model.md) | Conceptual scaffold mapping v1.5/v2/v3 onto mature mathematical apparatus. |
| [Next model experiments for SPLM](companion_notes/Next_Model_Experiments_for_SPLM.md) | Prioritised experiment catalogue; source of truth for the E1 and E3 programmes. |
| [Simulator v1.5 EOM](companion_notes/Semantic_Simulator_v15_EOM.md) | Forthcoming — v1.5 dynamics (dissipative semigroups). |
| [Simulator v2 EOM](companion_notes/Semantic_Simulator_v2_EOM.md) | Forthcoming — v2 dynamics (Fock-space second quantisation). |
| [Simulator v3 EOM](companion_notes/Semantic_Simulator_v3_EOM.md) | Forthcoming — v3 dynamics (non-abelian gauge theory). |

#### v3 leak-correction and information-bottleneck programme (May 2026)

| Note | Role |
|------|------|
| [Causal leak: bug and fix](companion_notes/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md) | Forensic detail of the v2 anti-causal autograd path, the `h.detach()` fix, and the `causal_probe.py` regression-test framework. |
| [Causal leak empirical comparison report](companion_notes/Causal_Leak_Empirical_Comparison_Report.md) | Closed-loop leak-free re-evaluation of every v2 SPLM checkpoint (headline: 777× inflation for TinyStories `splm_em_ln_multixi`). |
| [R6 information-bottleneck programme](companion_notes/Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md) | K-EMA / HiPPO-LegT / S4D basis experiments and four information-theoretic diagnostics. |
| [Determining optimal γ for SPLM](companion_notes/Determining_optimal_gamma_for_SPLM.md) | Four-estimator γ*-prediction framework and the resonance-predictor double success across the leak-correction boundary. |
| [Energetic minima alternatives](companion_notes/Energetic_Minima_Alternatives.md) | Leak-free retrains of the three energetic-minima alternatives (LN / SG / GM); recommendation shift from LN to scale-gauge when attractor diversity matters. |
| [Semantic attractor extraction](companion_notes/Semantic_Attractor_Extraction.md) | Leak-free dynamical-mode attractor extractions (Tiers 1 + 2b); F3 multi-basin structure survives the fix. |

#### PARFLM and FockPARFLM programme

| Note | Role |
|------|------|
| [PARFLM architecture v2](companion_notes/PARF_Augmented_SPLM_Architecture_v2.md) | Architecture design for the PARFLM V_φ augmentation (Q9c path). |
| [PARF-SPLM path and experiments](companion_notes/PARF-SPLM_Path_Forward_and_Experiments.md) | Experiment plan for the P1–P10 PARF scale-up ladder. |
| [Augmenting PARFLM for MCS languages](companion_notes/Augmenting_PARFLM_to_handle_MCS_Languages.md) | Fock-space augmentation design for v2 (context-free) expressivity. |
| [Improving Fock to match attention](companion_notes/Improving_the_Fock_Mechanism_to_match_Attention.md) | Fock Attention experiments, register diagnostics, causal leak discovery and fix, routing resolution hierarchy. |
| [FockPARFLM conservativity diagnostic](companion_notes/Fock_PARFLM_Conservativity_Diagnostic.md) | Five-arm diagnostic battery proving the conservative-by-construction property of FockPARFLM v2.1: Jacobian symmetry, ablation R², energy budget decomposition, conservativity dial, four-way separator. |

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed discussion of each experiment and its results.

### `notebooks/` — reproducibility

The experiments are organised into five categories. See
[EXPERIMENTS.md](EXPERIMENTS.md) for per-experiment methodology, result
tables, and key findings. See [COLAB_NOTEBOOKS.md](COLAB_NOTEBOOKS.md)
for Colab-ready notebook descriptions, GPU requirements, and runtime
estimates.

| Category | Paper §§ | Scope | Key folders |
|----------|----------|-------|-------------|
| **Descriptive validation** | §13 | STP–acceleration identity, Gaussian-well validation, and cross-architecture (GPT-2 / Pythia-160M) deceleration replication. | `stp_loss/`, `cross_model/` |
| **Negative-results chain** | §14.1 | Five scalar, Helmholtz, and gauge-field fits (E1–E5) on pretrained GPT-2 trajectories — all fail, motivating the prescriptive programme. | `e_init/` |
| **Conservative SPLM programme** | §§14–16 | SPLM prototype, SARF / mass ablations, attractor analysis, energetic-minima alternatives, multi-seed E1, energy-drift E3, controlled-γ damping sweep, v3 leak-correction audit, R6 multi-channel-ξ ladder (K-EMA / HiPPO / S4D), Helmholtz Q9d, Hybrid VA, and TinyStories scale-up pilots. | `conservative_arch/` and subfolders |
| **PARFLM / FockPARFLM programme** | §17, §17b–c | V_φ pairwise-interaction augmentation (P1–P10 ladder, PPL 26.4 ceiling), V_θ regularisation sweeps, structured V_θ (SQ1–SQ5), Fock-space registers (v1/v2 gates, QFT v2.1 creation-gate ablation, Dyck₂ falsifier), multi-ξ PARF at H=128, **Fock v2.1 routing fix (PPL 9.30)**, **Fock Attention direct exchange (PPL 9.42)**, and **CONS1–5 controlled-conservativity diagnostic**. | `conservative_arch/parf/`, `conservative_arch/scaleup/` |
| **Non-conservative / SP-HSPLM** | — | Per-token gyroscopic and solenoidal forces (Stage 1, Class B/C), Q9(e) pair-skew cell ladder (Stage 2). | `conservative_arch/non_conservative/`, `conservative_arch/sphsplm/` |

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
  doi       = {10.5281/zenodo.20579593},
  url       = {https://doi.org/10.5281/zenodo.20579593},
  note      = {Version v15 (Jun 7, 2026); supersedes v14
               (DOI 10.5281/zenodo.20531415, Jun 7, 2026).
               Companion code repository (DOI 10.5281/zenodo.20579561):
               \url{https://github.com/dimitarpg13/semsimula-paper}}
}
```

The companion code archive at the v4.10 release has its own citable DOI:
[10.5281/zenodo.20579561](https://doi.org/10.5281/zenodo.20579561).
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
   ([10.5281/zenodo.20579593](https://doi.org/10.5281/zenodo.20579593), v15,
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
| **v4.10** | **Jun 7, 2026** | [10.5281/zenodo.20579593](https://doi.org/10.5281/zenodo.20579593) | [10.5281/zenodo.20579561](https://doi.org/10.5281/zenodo.20579561) | **Comprehensive paper audit.** Right-margin overflow fixes (24 → 2 residuals): `\emergencystretch`, `\allowbreak` in new §23, two table-width corrections (§16 p.183, §17b p.229), shortened `\item[(R5)…]` label, `\path{}` for long identifiers. Self-reference pass: replaced every "the paper"/"present paper" with "this work" across all sections (§1×5, §4, §8, §9, §10, §15×2, §18, §18b, §19×3, A0, A3, `main.tex`×4). Companion-repo reference integrity: 80 `\path{}` links verified, 0 dangling. Figure/label audit: 56 figures confirmed, 0 undefined refs/citations, 0 duplicate labels, 0 label-prefix mismatches. Rebuilt `semsimula_paper.pdf` (308 pages) and `semsimula_paper_source.zip` (32 files incl. `18d_geometric_capabilities.tex`). Supersedes v4.9. |
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
