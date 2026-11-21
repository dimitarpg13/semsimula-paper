# Curvature Geometry Instruments, and the Plan for Splitting the Note

**What this note is.** §52 of the training-instabilities record: the curvature-geometry instruments, and the plan for splitting the CfC note that this folder carries out. It was split on 2026-11-21 from [`CfC_BAOAB_Integrator_and_Mitigations.md`](https://github.com/dimitarpg13/semsimula-paper/blob/5551315/companion_notes/CfC_BAOAB_Integrator_and_Mitigations.md) (written August to October 2026). The text is that document's, unchanged except for its links, and keeps its section numbers, so a reference to "§N" anywhere in the record or in other notes finds its note through [the index](README.md).

## Contents

- [52. Curvature Geometry Instruments, and a Plan for Splitting This Note](#52-curvature-geometry-instruments-and-a-plan-for-splitting-this-note)

---

## 52. Curvature Geometry Instruments, and a Plan for Splitting This Note

Two unrelated items, both administrative rather than investigative, recorded
here so the chronological record stays complete.

### 52.1 Three new curvature instruments (2026-09-11)

§41-§42 established that the low-rank channel is chronically dominant and that
capping it collapses the spikes, and `PRECISION_LR_MAX = 1.0` has been live
since step 47,121 as a result. Three instruments were added to answer the
questions that a *binding* cap opens up, which §41-§42 could not have asked:

| instrument | cell | what it adds |
|---|---|---|
| `sigma_lr_spectrum_report` | 6b-4 | keeps the **full** singular-value spectrum of $B_k$, which `sigma_lr_report` had been computing and then discarding at `svdvals(B)[..., 0]` |
| `spectrum_across_checkpoints` | 6b-4 | the above across best/spike/prereload checkpoints, with fallback to the permanent archive |
| `replay_curvature_rebalance_ablation` | 6d | sweeps `precision_max` **and** `precision_lr_max` together, rather than only the latter as `replay_precision_cap_ablation` does |

The reasoning, the measurements they are pointed at, and a staged procedure for
choosing the rank $r$ from measurement are in
[`Curvature_Diagnostics_and_Rank_Selection_for_Aniso_Gaussian_Vtheta.md`](Curvature_Diagnostics_and_Rank_Selection_for_Aniso_Gaussian_Vtheta.md).
Two points from it bear directly on sections above:

- **§42.2's exponent minima make §9's dead-well finding stale.** The
  ">99.9% of well-token pairs are numerically dead" measurement predates the
  cap. §42.2 shows the minima moving from the −257,630 to −150,076 range up to
  −146.8 to −51.8 once the cap was on, and fp32 underflows around $e^{-104}$ —
  so occupancy has probably improved and has not been re-measured.
- **The diagonal cap is on the wrong channel.** Cell 5's
  `_prec_max = 2.0 / d` sums over $d$ coordinates to exactly 2, the
  $\omega \Delta t \lt 2$ Verlet wall — a constraint that stopped applying to the
  diagonal the moment §24's propagator made that channel exactly integrated,
  while the low-rank channel that *does* still have the wall went uncapped until
  step 47,121.

### 52.2 A plan for splitting this note

At the time this section was first written this document was 5,780 lines —
past the point where it could be read or maintained comfortably. A naive
split was not available: it contained **566 intra-document section
references**, and every candidate boundary broke a large number of them.

| split before section | references crossing the boundary |
|---|---|
| §35 | 84 |
| §40 | 123 |
| §43 | 89 |
| §45 | 80 |
| §47 | 94 |
| §49 | 105 |

So the strategy was **not** to cut the chronological record in half. It was to
keep doing what had already been working — spinning settled investigations out
into standalone reports — and then to *stub* the sections whose content had
been superseded, preserving the heading (so every `§N.x` reference still
resolves) while removing the bulk.

Four spin-outs exist, all originating in sections of this note:
[`Diagnostic_Programme_in_CfC_BAOAB_Integrator.md`](Diagnostic_Programme_in_CfC_BAOAB_Integrator.md)
(from §35-§39),
[`Progressive_Curvature_Confinement_for_Aniso_Gaussian_Vtheta.md`](Progressive_Curvature_Confinement_for_Aniso_Gaussian_Vtheta.md)
(from §28-§29),
[`Register_Temperature_Instability_in_the_Fock_Creation_Gate.md`](Register_Temperature_Instability_in_the_Fock_Creation_Gate.md)
(from §49-§51), and the curvature/rank note above.

**Tier 1 — the changelog trailer (done, zero risk).** The provenance-plus-
changelog block at the foot of this file had grown to 440 lines and 26
"Previously updated" entries, 7.6% of the document, in a single paragraph. It
contains no section anchors, so nothing links into it. It now lives in
[`CfC_BAOAB_Integrator_and_Mitigations_CHANGELOG.md`](CfC_BAOAB_Integrator_and_Mitigations_CHANGELOG.md).

**Tier 2 — stub §49-§51 (done, 2026-09-11).** These three sections measured
**759 lines**, not the ~1,200 first estimated here — corrected before the cut
was made. Their settled version was already a standalone report, whose §49
preamble told readers to read that report first and treat these sections as
the record of the order in which conclusions were reached; every heading
(`§49.x`, `§50.x` and `§51.x`, 25 in total) was preserved verbatim; every
citation into them was verified beforehand to be a plain-text `§N.x` mention,
never a clickable cross-document anchor, so nothing outside this file could
break. Each body became a 2-4 sentence summary citing the report's own
section number, with corrections noted inline where a later section revised
an earlier one (§49.5's "self-reinforcing loop" reading, revised by §51.3's
decay accounting, is the clearest case) — the one exception is §49.7, whose
token-degeneracy finding has no analogue in the report and was kept in fuller
form rather than force a pointer that does not exist. Net: 759 → 262 lines
(−495, a 65% cut of the stubbed span).

**Tier 3 — §24-§48 (deferred).** This is the genuinely interlinked core, and
the table above says any cut through it breaks 80 or more references. It should
not be split until there is tooling to rewrite cross-document references
mechanically — at which point the boundary before §40 or §43 is the natural
one, since §24-§39 is the integrator-and-taxonomy story and §40 onward is the
remediation-and-validation story.

**Current size, after both completed tiers:** 4,987 lines.
