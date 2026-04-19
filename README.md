# Semantic Simulation — Paper companion

Companion repository for the paper

> **Semantic Simulation: A Dynamical Framework for Language Model Representations, and its Connection to Joint Embedding Predictive Architectures**
> Dimitar P. Gueorguiev (Independent Researcher), 2026.
> arXiv: `TODO: arXiv:XXXX.XXXXX` (link to be added after v1 submission).

This repository collects the **reproducibility artifacts** and the **unpublished
background manuscripts** cited in the paper. Its scope is deliberately narrow:
everything here is either directly required to resolve a citation in the paper
or needed to reproduce a figure or experimental claim.

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

> **Note on `Gueorguiev2024ReadMe`.** The citation "Semantic Simulation with
> Reinforcement Learning — README" (cited in §8) does not have a copy in this
> companion repository; it is pinned to commit
> [`38b1be1`](https://github.com/dimitarpg13/semsimula/blob/38b1be1/README.md)
> (2024-09-30) of the main research repository. This is the only external
> citation in the paper's bibliography that is not resolved inside this
> companion repository.

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

---

## Reproducing the §13 experiments

> **TODO — fill in before release.** Concretely:
>
> 1. Clone this repo and create a fresh Python environment (e.g., `python -m
>    venv .venv && source .venv/bin/activate`).
> 2. Install dependencies. A `requirements.txt` will be added before the
>    companion repo goes public; key dependencies are `torch`,
>    `transformers`, `numpy`, `scipy`, `matplotlib`, `pandas`, `jupyter`.
> 3. Launch `jupyter lab` and open `notebooks/stp_loss/energy_landscape_validation.ipynb`.
> 4. Execute top-to-bottom. Expected runtime on an M-series Mac with 16 GB
>    unified memory: **TODO (measure before release)**. GPU is not required.
>
> The notebook writes its outputs into `notebooks/stp_loss/results/`. The
> versions already committed there were produced by the author on **TODO
> (date + hardware)** and are included so that the figures in §13 can be
> inspected without re-executing the notebook.

---

## Citing this work

See [`CITATION.bib`](CITATION.bib) for a BibTeX entry. The short form:

```bibtex
@article{Gueorguiev2026SemSim,
  author  = {Gueorguiev, Dimitar P.},
  title   = {Semantic Simulation: A Dynamical Framework for Language Model
             Representations, and its Connection to Joint Embedding Predictive
             Architectures},
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
