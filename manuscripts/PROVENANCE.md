# Provenance of the Unpublished Manuscripts

This file records the **historical provenance** of each unpublished manuscript
included in this `manuscripts/` folder. For every manuscript it gives:

1. The BibTeX key under which the paper cites it
   (`paper/references.bib` of the main paper repository).
2. The **authorship date** as asserted in the bibliography `note` field
   — this is the date the author first wrote or substantively revised the
   document.
3. The **earliest git-observable commit** of the original source file (`.pdf`
   whenever the earliest form was a PDF, otherwise `.docx`) in the author's
   long-running research repository
   [`dimitarpg13/aiconcepts`](https://github.com/dimitarpg13/aiconcepts),
   together with that commit's date, short SHA, the path the file had at
   that commit, and a pinned URL.

The commit history in `dimitarpg13/aiconcepts` provides an independent,
cryptographically-anchored attestation of the existence of each manuscript at
or before the stated git timestamp. For eight of the ten manuscripts, the
earliest git commit date falls **within a few days** of the authorship date
asserted in the paper's bibliography, closely corroborating the bibliography
record through external git evidence.

---

## Provenance table

| Manuscript file (this repo)                                                                              | BibTeX key                        | Authorship date (from bib `note`)                                 | Earliest aiconcepts commit                                                                                                                                                                          |
| -------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Semantic_Tree_Operations.{pdf,docx}`                                                                    | `Gueorguiev2021TreeOps`           | 6 January 2021                                                    | [`f89d2a7` — 2021-01-07](https://github.com/dimitarpg13/aiconcepts/blob/f89d2a7/SemanticTreeOperations.docx) (as `.docx`; first `.pdf` was `4d0684b`, 2022-01-09)                                 |
| `Modeling_Attractive_and_Repulsive_Forces_in_Semantic_Properties.{pdf,docx}`                             | `Gueorguiev2022PARF`              | 8 February 2022                                                   | [`17717cf` — 2022-02-18](https://github.com/dimitarpg13/aiconcepts/blob/17717cf/docs/ModelingAttractiveRepulsiveForcesInSemanticProperties.pdf)                                                  |
| `On_Gaussian_Inverse_Semantic_Energy_Well.{pdf,docx}`                                                    | `Gueorguiev2022Well`              | 2022 (month unspecified)                                          | [`a7ee493` — 2022-03-02](https://github.com/dimitarpg13/aiconcepts/blob/a7ee493/docs/OnGaussianInverseSemanticEnergyWell.pdf)                                                                    |
| `On_The_Need_of_Dynamic_Simulation_when_Modeling_Interactions_of_Semantic_Structures.{pdf,docx}`         | `Gueorguiev2022DynSim`            | 20 March 2022                                                     | [`ed97151` — 2022-03-24](https://github.com/dimitarpg13/aiconcepts/blob/ed97151/docs/OnTheNeedofDynamicSimulationWhenModelingAttractiveRepulsiveForcesinSemanticStructures.pdf) (original filename, later shortened; see note below) |
| `Execution_Of_Semantic_Structures.{pdf,docx}`                                                            | `Gueorguiev2022Execution`         | 6 May 2022                                                        | [`c4d98e0` — 2022-05-07](https://github.com/dimitarpg13/aiconcepts/blob/c4d98e0/docs/ExecutionOfSemanticStructures.pdf)                                                                          |
| `On_the_Signature_Matrix_of_Semantic_Property.{pdf,docx}`                                                | `Gueorguiev2022Signature`         | 19 May 2022                                                       | [`24df96a` — 2022-05-21](https://github.com/dimitarpg13/aiconcepts/blob/24df96a/docs/OntheSignatureMatrixofSemanticProperty.pdf)                                                                 |
| `Modeling_Attractive_and_Repulsive_Forces_between_Semantic_Structures.{pdf,docx}`                        | `Gueorguiev2022SARF`              | 6 June 2022                                                       | [`5e49b23` — 2022-06-12](https://github.com/dimitarpg13/aiconcepts/blob/5e49b23/docs/ModelingAttractiveandRepulsiveForcesBetweenSemanticStructures.pdf)                                          |
| `Semantic_Simulation.{pdf,docx}`                                                                         | `Gueorguiev2024SemSim`            | first revision 11 February 2023; second revision 10 March 2024    | [`1bdbd47` — 2023-02-11](https://github.com/dimitarpg13/aiconcepts/blob/1bdbd47/docs/SemanticSimulation.pdf)                                                                                     |
| `The_Foundations_of_Semantic_Simulation.{pdf,docx}`                                                      | `Gueorguiev2022Foundations`       | 2022 (per bib `year` field)                                       | [`b9b1ceb` — 2024-12-30](https://github.com/dimitarpg13/aiconcepts/blob/b9b1ceb/docs/SemanticStructures/TheFoundationsOfSemanticSimulation.docx) (no earlier `.pdf` in aiconcepts)                |
| `Constructing_Langrangian_for_Semantic_Space.{pdf,docx}`                                                 | `Gueorguiev2026Lagrangian`        | 2026                                                              | [`de90d71` — 2026-04-12](https://github.com/dimitarpg13/aiconcepts/blob/de90d71/docs/SemanticStructures/ConstructingLangrangianForSemanticSpace.docx) (no earlier `.pdf` in aiconcepts)          |

---

## How closely does git corroborate the bib dates?

For eight of the ten manuscripts, the earliest git-observable timestamp is
**within two days to three weeks** of the authorship date asserted in
`paper/references.bib`:

| Manuscript              | Bib date              | Earliest git commit | Gap      |
| ----------------------- | --------------------- | ------------------- | -------- |
| Semantic Tree Operations | 6 Jan 2021            | 2021-01-07          | +1 day   |
| PARF (properties)       | 8 Feb 2022            | 2022-02-18          | +10 days |
| Gaussian Well           | 2022 (unspecified)    | 2022-03-02          | n/a      |
| DynSim                  | 20 Mar 2022           | 2022-03-24          | +4 days  |
| Execution               | 6 May 2022            | 2022-05-07          | +1 day   |
| Signature Matrix        | 19 May 2022           | 2022-05-21          | +2 days  |
| SARF (structures)       | 6 Jun 2022            | 2022-06-12          | +6 days  |
| Semantic Simulation     | 11 Feb 2023 (rev. 1)  | 2023-02-11          | **same day** |

The remaining two manuscripts, `Gueorguiev2022Foundations` and
`Gueorguiev2026Lagrangian`, appear in aiconcepts only as `.docx` (no PDF
intermediate); their earliest commit dates are 2024-12-30 and 2026-04-12
respectively. For `Gueorguiev2022Foundations` the git record is therefore
roughly two years later than the bibliography's `year = 2022` field, and
the bib remains the primary authorship record in that case.

---

## Notes on individual entries

### `Gueorguiev2022DynSim` — filename change

The earliest PDF at `ed97151` (2022-03-24) is stored under the original
longer filename
`docs/OnTheNeedofDynamicSimulationWhenModelingAttractiveRepulsiveForcesinSemanticStructures.pdf`,
which was later shortened to
`docs/OnTheNeedofDynamicSimulationWhenModelingInteractionsOfSemanticStructures.pdf`
(first appearance on 2022-03-27 at commit `8133c16`). Both names refer to
the same manuscript; the 2022-03-24 filename preserves the original wording
of the title as first written.

### The 2024-01-05 "rm pdfs in docs" reorganisation

On 2024-01-05 the aiconcepts repository was reorganised: the PDF versions
of these manuscripts were removed from `docs/` (commit `6602c55`, message
*"rm pdfs in docs"*) and the `.docx` sources were bulk-committed into
`docs/` (commit `5e29662`, message *"add docs"*), later moved to
`docs/SemanticStructures/`. The `5e29662` commit is therefore **not** the
earliest provenance evidence for these documents — the 2022 PDFs at the
commits listed in the table above precede the reorganisation by roughly
two years and are retained in git history.

### `Gueorguiev2021TreeOps`

The earliest form is the `.docx` at commit `f89d2a7` on 2021-01-07, initially
stored at repo root (path `SemanticTreeOperations.docx`) and later moved
into `docs/SemanticStructures/`. A `.pdf` version appears later at commit
`4d0684b` on 2022-01-09. The 2021-01-07 `.docx` timestamp is within a day
of the authorship date asserted in the bibliography ("6 January 2021").

### `Gueorguiev2024SemSim`

The earliest PDF at `1bdbd47` on 2023-02-11 carries the commit message
*"add new doc"*, corresponding exactly to the bibliography's first-revision
date ("first revision 11 February 2023"). Two subsequent revisions occur
at `2bffb23` (2023-02-12) and `9267a14` (2023-02-14), with the second
major revision date noted in the bib (10 March 2024) reflected in further
commits during 2024.

---

## How to update this file

When a new manuscript is added to `manuscripts/`, run (from the aiconcepts
working copy) the following to find the earliest PDF-or-docx commit under
any path:

```bash
git log --all --diff-filter=A --follow --date=iso \
    --pretty=format:'%h %ad' -- '*<short-name-pattern>*' | tail -1
```

If that command does not return the earliest commit — for example when
the file was renamed outside the detection threshold of `--follow` —
enumerate all `--diff-filter=A` commits without `--follow` using a broad
path pattern and inspect the list:

```bash
git log --all --diff-filter=A --date=iso \
    --pretty=format:'%h %ad %s' --name-only -- '*<keyword>*' 2>&1 \
    | less
```

The pinned URL format is
`https://github.com/dimitarpg13/aiconcepts/blob/<short-sha>/<path-at-that-commit>`
where `<path-at-that-commit>` is the path the file held at the listed commit
(not necessarily its current path).
