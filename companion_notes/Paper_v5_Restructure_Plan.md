# Restructuring paper v5 around the forced-Lagrangian thesis — a triage

> **Status.** Skeleton, opened **2026-09-25**. Section-by-section verdicts
> for the 470-page `paper_v5`, with a *decision point* column naming the
> experiment that settles any verdict not yet settled. **No edit in this
> plan should be made before E5 and F1 have run** — they decide the wording
> of the central thesis. The thesis itself is stated in
> [`Forced_Lagrangian_Reformulation.md`](Forced_Lagrangian_Reformulation.md);
> the measurements behind it are in
> [`Geodesic_Experiments_with_CfC_BAOAB.md`](Geodesic_Experiments_with_CfC_BAOAB.md).
> Edits already applied to the paper: Remark 52 (§10), its footnote (§17c),
> and the pointer in §18.

---

## 1. Principles

1. **Retreat to what was measured, not to what feels safe.** Every
   withdrawn claim is replaced by the narrower claim the measurement
   supports, with the measurement cited. Nothing is softened by adjective.
2. **Keep the pre-registration visible.** The paper's credibility with all
   three audiences rests on the fact that predictions were recorded before
   results. The forecast record (checklist §8, three misses) is an asset in
   the text, not an embarrassment to hide.
3. **The geometry is untouched.** §§2–6 are not edited for content.
4. **One thesis sentence, everywhere the same.** Drafted in the
   reformulation note §0, finalised after F1, then propagated to the
   abstract, §1, §7, §18, §19, §20.
5. **Depth and corpus caveats are explicit.** Every trajectory-level claim
   carries its depth and its corpus: "at L=2, on OpenWebText" until F2 and
   F6 say otherwise. The Verlet-era geodesic work mixed TinyStories and
   OWT; the CfC+BAOAB work is OWT only (reformulation note §2.7).

---

## 2. The headline result: the mechanism ladder

The restructured paper's central experimental table is not a single PPL but
an **ordering**, pre-registered before the runs that complete it. Five arms,
each removing exactly one mechanism from the one above, all sharing
tokenizer, corpus, batches, width, depth, learning rate, schedule and token
budget:

| arm | mechanism removed relative to the arm above | settled PPL |
| --- | --- | ---: |
| matched GPT-2 L=8 | — (the reference architecture) | **49.81** |
| L=2 `'attention'` | the transformer itself | 63 (59–68), run 9 |
| L=2 `'attention_potential'` | **conservativity**: the same xi-routed attention, but entering as a potential so the force stays a gradient | 66 (62–72), run 5 |
| L=2 `'none'` | the exchange field | **66.98** |
| L=2 `'none'`, reverse channel off | **the Fock mechanism**: the register-to-token path | 105 (85–140), run 8 |

Measured values in bold; the rest are pre-registered bands from protocol
§5.3. The *ordering* is the prediction and any inversion is a result.

**Why this is the headline.** Each gap prices one thing the paper argues
about, and prices it by construction rather than by attribution:

| gap | prices | predicted |
| --- | --- | ---: |
| GPT-2 → `'attention'` | what the transformer has that this architecture does not | ~13 PPL |
| `'attention'` → `'attention_potential'` | **the price of conservativity** | ~3 PPL |
| `'attention_potential'` → `'none'` | the exchange field | ~1 PPL |
| `'none'` → no reverse channel | **the Fock mechanism** | ~38 PPL |

If the predictions hold, the paper's two most quotable sentences fall out of
one table: **conservativity is cheap** — about 5% — and **the
non-conservative memory mechanism is what carries the model**, worth more
than the entire remaining gap to a matched transformer.

**And it converges with the geometry.** E1 and E5 measured the same
statement in the dynamics: the reverse channel is ~90% of the per-layer
deflection, and removing it costs 3.9× in PPL at inference (master doc
§4.7, §11.6). The ladder says it again in trained PPL, from scratch, with
every parameter free to compensate. Two independent measurements — one
geometric on a fixed model, one predictive across trained models — landing
on the same conclusion is worth more than either alone, and the paper
should present them as a pair.

The uncomfortable half is the point, not a problem to be managed: the
conservative machinery is nearly free *and* nearly inert without the
non-conservative driver. That is the forced-Lagrangian thesis
([`Forced_Lagrangian_Reformulation.md`](Forced_Lagrangian_Reformulation.md))
stated in perplexity.

**Caveats that travel with the table.** L=2 only (F2/L=4 pending),
OpenWebText only (F6 pending), and `'attention'`'s 3e-04 predecessor
(68.33, §5.1) must not be quoted in the same table — mixing learning rates
is exactly what §5.2's suspended attribution records.

---

## 3. The triage

Verdicts: **keep** (no content change), **reframe** (same results,
rewritten around the forced system), **retreat** (claims withdrawn and
replaced), **promote** (moves toward the centre), **rewrite**, **read
first** (the section may already hold what the reframing needs).

| § | section | verdict | what changes | decision point |
| --- | --- | --- | --- | --- |
| 00–06 | notation, semantic space, signature matrix, Gaussian well, PARF, SARF | **keep** | notation gets F_rc and κ_g | — |
| 07 | Lagrangian | **reframe** | Lagrange–d'Alembert with an explicit forcing term; "geodesic" becomes "unforced motion" | none; exact |
| 07a | position-dependent damping | **reframe** | add the tangential-damping point (master doc §4.8); the Verlet-era γ_eff discussion stands as a statement about speed | none |
| 08 | tree operations | **keep** | — | — |
| 09 | expressivity / MCS | **keep, review** | check for any reliance on refinement | none expected |
| 10 | JEPA connection | **keep** | Remark 52 already in place | — |
| 11 | hidden-state interpretation | **review** | E3 §6.8: at L=2 the layer-1 velocity is the embedding projection; any interpretation of v as a semantic velocity needs L ≥ 3 | E3 at L ≥ 3 |
| 12 | semantic mass | **keep** | mass enters the forced equation unchanged | — |
| 13 | STP loss as normal acceleration | **promote** | this *is* κ_g; becomes the forcing measurement rather than a diagnostic to minimise | F1 (its distribution) |
| 14 | experiments | **reframe** | add the ladder, the matched baseline, E1–E5, F-series; the forecast record | E5, F1 |
| 15, 15a | conservative architectures, causal integrity | **keep** | — | — |
| 16 | hybrid SPLM | **keep, review** | — | — |
| 17 | PARF-augmented SPLM | **keep** | — | — |
| **17c** | **Fock-PARFLM** | **promote** | from an add-on chapter to the central mechanism: the register bank is the driver; +275% ablation (with its OOD caveat), the L=1 static-bank finding, the reverse channel as ~90% of the step | E5, F1, F2 |
| 17b, 17d–17g | cross-arch v-reg, structured potential, scaling, context mixing, continuous learning | **keep, review** | 17e scaling: the ladder numbers replace any older ratios | — |
| 17h | first-order sufficiency | **promote** | E3's L=2 finding is direct evidence for the depth-memory mismatch | E3 at L ≥ 3 |
| **18** | **Riemannian geometry** | **retreat, largest** | the residual diagnostic is re-based as the forcing measurement; "the trajectory is a geodesic" withdrawn and replaced by the three-layer table; the continuum footnote stays; refinement language removed | F1, F2 |
| 18b, 18f, 18g, 18h, 18i | relations: EBMs, AlphaFold, Langevin completion, portable potentials, optimiser-inspired transformers | **keep, light** | remove any "geodesic" phrasing that refers to the trained path | — |
| 18c, 18d | memorisation capacity, geometric capabilities | **review** | any capability argued *from* geodesicity needs re-basing | F1 |
| 18e | relation to liquid neural networks | **promote** | CfC is that lineage and is now the production integrator | — |
| 18j | relation to flow matching | **retreat, small** | any refinement / continuous-time equivalence language goes | none |
| 19 | conclusion | **rewrite** | last | after everything |
| 20 | dynamical simulator | **reframe** | "simulation" survives (it is a numerical integrator); of a *driven* system; the exact propagators are the strongest part and stay | E5 |
| A0 | edition history | **update** | this edition's entry | — |
| A1 | non-autonomous framework | **read first** | a time-dependent forcing from a slowly changing register bank is a non-autonomous system with an adiabatic parameter; the reformulation should be written *in* this appendix's language, and A1 may move into the main text | — |
| A2 | inference efficiency | **keep** | — | — |
| A3 | experiment index | **update** | E1–E5, F1–F5, ladder runs | as results land |

---

## 4. The three audiences, and what each keeps

**Language-modelling researchers.** The exact per-token decomposition of
every step into prior and memory (reformulation §2.4) is the thing no
transformer offers, and the matched baselines (same tokenizer, data,
batches, width, token budget) are what make the 66.98 vs 49.81 comparison
mean something. The Fock register bank as a working-memory mechanism with a
measured contribution is the mechanism story.

**Physicists.** A driven, damped mechanical system with exact propagators
per substep; Jacobi's theorem removing the Christoffel symbols; a concrete
reason the symplectic integrator failed ($\omega \Delta t \gt 2$) and what
replaced it (the exact harmonic flow); damping that is tangential and a
forcing that is transverse; the Lagrange–d'Alembert form. The retreat from
"geodesic" to "forced" is a retreat *toward* standard mechanics, not away
from it.

**The general reader.** A hypothesis stated, pre-registered, measured, and
retreated in the open, with the forecast misses recorded. That is rarer
than a confirmed one and reads better.

---

## 5. Title and subtitle

**Current:**

> **Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient
> Semantic Inference**
> *Conservative-by-Construction Language Models and the Shared-Potential
> Separator, with a Correspondence to Joint Embedding Predictive
> Architectures*

**Agreed 2026-09-25:**

> **Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient
> Semantic Inference**
> *Conservative Potentials, Non-Conservative Memory, and the
> Shared-Potential Separator*

### 5.1 Why the old subtitle cannot stand

"Conservative-by-Construction" is **true of SPLM and PARFLM and false of
Fock-PARFLM**, which is the model every result in the restructured paper
comes from. The Obstruction Theorem forced auxiliary degrees of freedom
in; the reverse channel that carries them is non-conservative by design;
E1 and E5 then measured it as ~90% of the per-layer deflection and 3.9x in
PPL. So the subtitle advertises the property the paper's own theorem
proves insufficient and the paper's own experiments show is nearly inert
without the non-conservative addition.

**This change does not wait on any pending result.** It is wrong
independently of F1, F2, F6 and the two remaining ladder arms.

### 5.2 Why the new one is a strengthening, not a retreat

The arc it names is the paper's real one, and it is unusually clean:

1. The **Shared-Potential Separator** and the **Conservative Obstruction
   Theorem** prove that no scalar potential on the token subsystem
   reproduces attention's structural properties without auxiliary state.
2. **Fock-augmented PARFLM** is identified as the minimal extension that
   supplies it.
3. E1, E5 and the mechanism ladder (§2) **measure** that the auxiliary,
   non-conservative part does the bulk of the work — conservativity costs
   about 5%, the register-to-token path is worth more than the entire
   remaining gap to a matched transformer.

The theorem predicted the conservative part would be insufficient; the
experiments say by how much. A paper that confirms its own impossibility
result empirically is in a stronger position than one that only proves it,
and the subtitle should carry both halves rather than only the half that
turned out to be the smaller term.

### 5.3 What was dropped, and why

"with a Correspondence to Joint Embedding Predictive Architectures" moves
to the body and the keywords. The JEPA correspondence (§10) is one section
of forty, descriptive rather than load-bearing, and giving it equal billing
with the Separator was already generous. Remark 52 lives there and stays.

### 5.4 Runner-up, kept on the record

> *The Shared-Potential Separator and the Measured Price of Each Mechanism*

Foregrounds the measurement instead of the tension. Rejected because
"non-conservative memory" is the more arresting phrase and the one nobody
else is claiming.

### 5.5 Open question on the main title

**"Semantic Simulation" survives** — the model *is* a numerical integrator
of a mechanical system in semantic space, with exact propagators per
substep. What the reformulation drops is *free*, not *simulation*.

The word to re-examine is **"Efficient"**. That claim is about inference
cost (Appendix A2), while every headline number now in play is a quality
ratio (1.345x the matched GPT-2 at L=2). Left open: does it still earn its
place in the title, or does it invite the comparison the paper then has to
spend a section qualifying? **Decide with the thesis sentence, last**, once
the ladder is complete.

---

## 6. Order of edits, once the decision points are in

1. A1 read; decide whether it moves into the main text (it probably does,
   as the formal home of the forcing).
2. §7 and §07a: the forced Lagrangian, damping is tangential.
3. §18: the retreat. Three-layer table, residual re-based, refinement
   language out.
4. §13: promote; connect the STP loss to κ_g and to F1's
   distribution.
5. §17c: promote; the register bank as the driver.
6. §14: the experimental record, including the misses.
7. §11, §17h, §18c, §18d, §20: reframe as their decision points allow.
8. §1, abstract, §19: the thesis sentence, last — **and the title/subtitle
   with it** (§5), including the open "Efficient" question.
9. A0, A3.

---

## 7. Open decision points

| decision | settled by | changes |
| --- | --- | --- |
| "memory steers" vs "memory steers occasionally" | **F1** | the thesis sentence; §13, §18 |
| is L=2 a floor or the verdict on the geodesic share? | **F2** (needs L=4) | every "at L=2" caveat |
| can the forecastability claim be made at all? | E3 at L ≥ 3 | §11, §17h |
| the price of the pure geodesic in PPL | **E5 — settled 2026-09-25**: 3.91×, no knee; layer 1's output direction is the register readout | §17c, §20 — and §17c must describe the last layer as a memory read, not a forced step |
| is V_φ ever active? | F5 | §5, §17 (the pairwise potential's role) |
| does the hallucination claim survive re-basing? | F4 | §18d and wherever hallucination is discussed |
| is the dominance of the forcing a property of the corpus? | F6 (TinyStories) | every "the reverse channel dominates" sentence gains "on OpenWebText" until then; §14, §17c |

---

## 8. Ledger

| date | item |
| --- | --- |
| 2026-09-25 | skeleton opened; triage table drafted; no paper edits beyond Remark 52 / footnote / pointer |
| 2026-09-25 | E5 settled its decision point (§5); F1 remains the gate on the thesis sentence |
| 2026-09-25 | mechanism ladder named as the headline result (§2); `attention_potential` band revised to 66 (62–72) after re-reading the detach semantics — protocol §5.3a |
| 2026-09-25 | subtitle agreed (§5): *Conservative Potentials, Non-Conservative Memory, and the Shared-Potential Separator*; JEPA correspondence demoted to the body; "Efficient" in the main title left open |
