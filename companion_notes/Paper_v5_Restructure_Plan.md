# Restructuring paper v5 around the forced-Lagrangian thesis — a triage

> **Status.** Skeleton, opened **2026-09-25**. Section-by-section verdicts
> for the 470-page `paper_v5`, with a *decision point* column naming the
> experiment that settles any verdict not yet settled. **No edit in this
> plan should be made before E5 and F1 have run** — they decide the wording
> of the central thesis. The thesis itself is stated in
> [`Forced_Lagrangian_Reformulation.md`](Forced_Lagrangian_Reformulation.md);
> the measurements behind it are in
> [`Geodesic_Experiments_with_CfC_BAOAB.md`](Geodesic_Experiments_with_CfC_BAOAB.md).
> Edits already applied to the paper: Remark 52 (§10), its footnote (§20),
> and the pointer in §18.

---

## 0. Framing decision — this is a book, not a paper — **2026-09-27**

> **Read this before acting on anything below.** Every verdict in this
> document, and every entry in
> [`Paper_v6_Section_Audit.md`](Paper_v6_Section_Audit.md), was written as
> though the object were a paper. It is not.

**The author's framing, recorded verbatim in substance:** the work started
as a paper, but at 476 pages it has evolved into a **book** — intended as
relevant reading for a broad scientific audience: physicists curious about
language modelling, mechanical engineers, chemists working on particle
dynamics, scientists generally. Not only machine-learning researchers.

**The measurable mismatch.** The document is 476 pages, 23 numbered
sections spanning pp. 20–416, built on `\documentclass{article}` with a
995-word `abstract` environment. None of those three facts belongs to the
same form:

| | what it is | what a book of this size would have |
| --- | --- | --- |
| front matter | a 995-word abstract | a preface, a reader's guide, a short abstract |
| top level | `\section`, 23 of them, flat | `\chapter`, grouped into `\part`s |
| class | `article` (via `jmlr2e`) | `book` or `memoir` |
| opening | dense results, perplexity tables | the question the work started from |

**The constraint that shapes any solution.** SSRN and Zenodo both require
an abstract as *platform metadata*, so an abstract cannot simply be
removed. The question is not whether one exists but **what goes in the PDF
and how long it is**.

**Consequences for the plan below.** Three verdicts in §3's triage change
character under a book framing, and should be re-read before use:

- "Keep / reframe / retreat" was judged by *claim correctness*. A book
  must also be judged by *accessibility and narrative order* — a section
  can be entirely correct and still be in the wrong place for a chemist
  reading linearly.
- §4's "three audiences" was written as a positioning argument for a
  paper's framing. For a book it becomes a **structural requirement**: the
  reader's guide has to route each audience to a different entry point.
- §6's edit order is a *correctness* order. A book needs that work done,
  but the front matter and part structure should be decided **first**,
  because they determine where the corrected material lands.

**Nothing in the audit is invalidated** — the section-level findings are
about claims and remain true. What changes is what "restructuring" means:
not only fixing what is wrong, but deciding the form the corrected
material takes.

**Two outputs, not one — recorded 2026-09-27.** A separate **TMLR paper**
is planned, much shorter, to be started once the ladder experiments are
complete and the CfC+BAOAB models are understood. That changes what this
document is for:

| | **this document → the book (paper v6)** | the TMLR paper (later) |
| --- | --- | --- |
| audience | broad scientific — physicists, engineers, chemists, ML | ML reviewers |
| length | ~476pp | conference/journal length |
| carries | the whole programme, including the retreat and the misses | one result, cleanly argued |
| likely core | the framework, the architectures, the full measurement record | **the mechanism ladder** — matched arms, each gap pricing one component |
| front matter | preface, reader's guide, short abstract | a conventional abstract |

**The book is the current focus.** The TMLR paper is deferred until the
ladder is complete, and nothing in this plan should be shaped around its
needs. One practical consequence worth holding on to: the ladder is the
most obviously extractable unit — matched arms, pre-registered
predictions, a single ordering — so material written for it in the book
should be self-contained enough to lift, but the book's version should not
be trimmed to paper length in anticipation.

**Open, and the author's call:**

1. Short abstract (~200 words, accessible) in the PDF, with the present
   995 words relocated to a *Summary of principal results*?
2. A **preface** — why this exists, the question it started from, and the
   honest arc of a hypothesis stated, measured and partly retreated. This
   is the book's most distinctive feature and it is written in the
   author's voice, so it is not something to draft unasked.
3. A **reader's guide** routing physicist / engineer / LM researcher /
   general scientist to different entry points.
4. `\part` grouping — and eventually whether to move from `article` to a
   book class, which is mechanical but touches all 40 input files.

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
   abstract, **§1**, **§7**, **§27**, **§38**, **§20** (rendered numbers;
   see the audit's file-to-section map).
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
| L=2 `'attention'` | the transformer itself | **63.51**, run 9 |
| L=2 `'attention_potential'` | **conservativity**: the same xi-routed attention, but entering as a potential so the force stays a gradient | **80.90**, run 5 |
| L=2 `'none'` | the exchange field | **66.98** |
| L=2 `'none'`, reverse channel off | **the Fock mechanism**: the register-to-token path | **87.93**, run 8 |

Measured values in bold; the rest are pre-registered bands from protocol
§5.3. The *ordering* is the prediction and any inversion is a result.

**Why this is the headline.** Each gap prices one thing the paper argues
about, and prices it by construction rather than by attribution:

| gap | prices | predicted |
| --- | --- | ---: |
| GPT-2 → `'attention'` | what the transformer has that this architecture does not | **13.70 PPL (+27.5%), measured** |
| `'attention'` → `'attention_potential'` | **the price of conservativity** | **17.39 PPL (+27.4%), measured** |
| `'attention_potential'` → `'none'` | the conservative field is **worse than none**: adding it costs +20.8% | **−13.92 PPL, measured** |
| `'none'` → no reverse channel | **the Fock mechanism** | **20.95 PPL, measured** |

**The prediction that conservativity is cheap was wrong, and the measured
answer is better for the paper.** At matched parameters — 77,360,081 in
both arms, same routing source, same heads, same gate — making the
exchange field conservative costs **+27.4%**, and the conservative field
is **+20.8% worse than having no exchange field at all**. Conservativity
does not tax the mechanism; it reverses its sign. The non-conservative
twin *gains* 5.2% over `'none'`; the conservative twin *loses* 20.8%.

This is the Conservative Obstruction Theorem measured on a matched pair.
The theorem proves that no scalar potential on the token subsystem can
reproduce attention's structural properties; here the same routing, the
same capacity and the same parameter count, differing only in whether the
field enters as a force or as the gradient of a potential, differ by 27%
in perplexity. The ordering the ladder measures is therefore:

> matched GPT-2 **49.81** · attention **63.51** · none **66.98** ·
> attention_potential **80.90** · no reverse channel **87.93**

with the two conservative-leaning arms at the bottom.

The exchange field's own price is now measured too, and it halved under
retuning: `'attention'` against `'none'` is **+5.2%** with both arms at
1.2e-03, where the same comparison with both arms at 3e-04 read +9.9%
(protocol §5.2, suspension lifted; §5.7). One caveat travels with it — the
`'attention'` arm clips on 29.2% of steps at this LR against `'none'`'s
0.0%, so +5.2% is a lower bound.

The bottom row is now measured (protocol §5.6) and it carries a
methodological result of its own: the *inference-time ablations* of the
same mechanism said 3.75x and 3.91x, where training without it says
**1.31x**. Removing a trained component overstated its value by nearly
threefold in PPL ratio and fivefold in nats. Any version of this table
built from ablations rather than trained arms would have been wrong by
that factor, and §14 should say so — it is the clearest instance in the
programme of why the ladder is built from independently trained models.

**The second headline, and the one the framework needs.** The bottom arm of
the ladder is the only fully conservative trained model in the programme,
and on it the geometry is *exact*: at the clean layer the step is the
damped Vθ geodesic step followed by LayerNorm, to R = 0.0003 (master doc
§4.9). So the paper can say both of these, measured, in the same chapter:

> Riemannian geodesics are real in this architecture — the trained step is
> the projected geodesic step to three decimal places — and they cost
> 31.3% in perplexity.

That is a far better position than either "the framework is geometric"
(unmeasured) or "the geodesic reading failed" (which is true only of the
forced model). §18's retreat and §37's value proposition become two halves
of one measured statement.

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

## 2a. Objective: contain and label Verlet, do not reduce it — **2026-09-27**

**The author's objective**, and the strategy agreed for it. Verlet is
genuinely superseded, and for a demonstrated reason rather than a
preference: the trained stiffness pushes $\omega \Delta t$ past Verlet's
stability bound of 2, which is what produced the E/P spikes and motivated
the closed-form CfC/BAOAB propagator. Verlet-era material should therefore
be **contained**, not scattered through the book unmarked.

**But not deleted, and the distinction matters.** The Verlet failure *is*
the argument for the closed-form propagator. Remove it and the book's
integrator looks like an arbitrary choice rather than a forced one. The
instability checkpoints published on Hugging Face exist for the same
reason.

**The measured footprint** (2026-09-27): **120** Verlet mentions against
**157** CfC/BAOAB across 18 files. So the problem is not quantity. The
failure mode actually hit was a Verlet-era *measurement*
($\gamma_{\text{geo}} \approx 0.9$) read as a property of the
architecture, which then propagated into §8's motivation — cured by
scoping and withdrawal, not by reducing mentions.

### Terminology, settled **2026-09-27** — say "the explicit-integrator era"

**"Verlet-era" is retired.** It was never accurate for the whole period, and
the question was settled by reading the integrators rather than the labels.
Three distinct schemes ran, in two lineages:

| lineage | file | update | name |
| --- | --- | --- | --- |
| SPLM family | `model.py`, `multixi/model_multixi.py`, `energetic_minima/model_ln.py` | `v = (v + dt·f/m)/(1+dt·γ)` then `h += dt·v` | **damped semi-implicit Euler** — explicit velocity state, `v_0 = 0` |
| first-order ablation | `first_order_ablation/model_first_order.py` | `h_new = h + dt·f/m` | **gradient flow** — no velocity at all |
| hybrid + PARF/Fock | `helmholtz/model_helmholtz.py`, `parf/model_parf_multixi.py` | `h_new = h + (h−h_prev)/(1+γ·dt) + dt²·f/(m(1+γ·dt))` | **damped Störmer–Verlet** — no velocity state; the velocity is the position difference |

**Nothing in this programme ever ran velocity-Verlet.** Velocity-Verlet
carries an explicit velocity through half-kick / drift / half-kick. The third
row is the *position* form: undamped it is
$h_{n+1} = 2h_n - h_{n-1} + dt^2 f/m$. The code says so against itself — it
calls `delta` a "velocity proxy", and the CfC/BAOAB branch exists partly
because those integrators "need to return an outgoing velocity as well as a
position", which the Störmer form has none of. Every docstring and every
paper sentence saying "velocity-Verlet" is wrong by one word, consistently.

**Why the distinction earns its keep.** The $\omega \cdot dt < 2$ wall is the
Störmer/leapfrog bound. It governs the hybrid and PARF/Fock lineages. It was
never the stability condition on the SPLM family, which is semi-implicit
Euler. Any sentence generalising that bound across the whole period is
overreaching, and the E/P-spike story that motivated the propagator belongs
to the Störmer lineage alone.

**The rule for the rewrite.**

- Period, collectively: **"the explicit-integrator era"** — what the three
  schemes share is being explicit, and being superseded by the closed-form
  propagator. Not "Verlet-era", not "Euler-era".
- Any claim that depends on the scheme: name it. **damped semi-implicit
  Euler** (SPLM family), **damped Störmer–Verlet** (hybrid, PARF, Fock),
  **gradient flow** (the first-order ablation).
- Never write "velocity-Verlet" again.

**§18's equation was already right** (`eq:helmholtz-update`, the S-block
branch): it is the position-difference form, which is what
`model_helmholtz.py` computes. Only the surrounding prose calls it
velocity-Verlet. So §20 is a relabelling job plus one word, not a correction
of the mathematics — which is what the pass needed to know before starting.

**Twelve published cards were wrong, and are now fixed — 2026-09-27.**
The `semi_implicit_euler` string appears nowhere in the code, so every
`integrator` field on the older cards was hand-written metadata. Checking
`model_type` against the class that defines `_layer_step` settled each one.
Only `model_parf.py`, `model_parf_multixi.py`, `model_parf_sparse.py` and
`helmholtz/model_helmholtz.py` define a step at all; every Fock, attention
and structured-Vθ class inherits from them, so **the entire PARF / Fock /
hybrid lineage is Störmer**.

| card lineage | was | now |
| --- | --- | --- |
| `ScalarPotentialLM*` — 2 repos | `semi_implicit_euler` | unchanged, it was right |
| `FockG1MultiXiPARFLM` — 1 repo | `first_order_gradient_flow` | unchanged, it was right |
| `HybridSPLM`, `MultiXiPARFLM`, `FockMultiXiPARFLM`, `FockAttentionPARFLM` — 12 repos | `semi_implicit_euler` (8), absent (3), `verlet` (1) | **`damped_stormer_verlet`**, plus an `integrator_note` giving the update |

Eleven of the twelve also carried the wrong *update* in their README, not
just the wrong name: an ASCII architecture line reading
`+-- Damped Euler step: v += dt*f/m; v /= (1+dt*gamma); h += dt*v`, which is
the SPLM family's step, on a PARF/Fock card. Replaced with the Störmer line
and a dated correction note explaining what changed and that no measurement
moves. The hybrid card additionally claimed its S-blocks "use the same damped
Euler integration as the purely conservative variants", which was wrong twice
over — wrong scheme, and not the same as all three linked models. Rewritten.

Verified after upload: all twelve read `damped_stormer_verlet`, none still
contains the Euler pseudocode, and the two genuinely-Euler SPLM cards were
left alone.

**Second wave, same day: the sixteen "velocity-Verlet" mentions.** A separate
sweep of the collection found the term in six cards, including the one whose
whole subject is the instability. Nine of the sixteen were on that card alone,
and one of them carried the Euler update *labelled* Velocity-Verlet — both
errors in a single line, which the first pass had missed because its pattern
matched only "Damped Euler step". All are now "Störmer–Verlet", with the
stability-bound sentences left intact: the \(\omega \cdot dt < 2\) wall is the
Störmer bound, so the instability finding is unaffected by the renaming. The
`velocity-verlet` YAML tag became `stormer-verlet`. The gradient-flow
ablation's card needed a tailored note, since the wrong name was applied to
the second-order *anchor* it compares against, not to the ablation itself.

**Final state, verified against the live files: 15 of 15 cards correct.** The
two `ScalarPotentialLM*` cards keep the `v += dt*f/m; v /= (1+dt*gamma);
h += dt*v` pseudocode, which is right for them and only for them. The repo
slug `...-verlet-instability` is unchanged: renaming it would break the
collection item and the URL, and "Verlet instability" remains true of the
Störmer family.

---

### The three-tier policy

| tier | Verlet is… | policy |
| --- | --- | --- |
| 1 | **the subject** — the stability bound, the spikes, why the propagator exists | **contain**: told once, properly, in one home (**§28**, the Direct Dynamical Simulator, which is `20_dynamical_simulator.tex` — *not* §20, which is Fock-PARFLM) |
| 2 | **the provenance of a number** — any Verlet-era measurement | **label**: a scope marker on every one, and re-check any inference drawn from a Verlet-only diagnostic. §27's `rem:riemannian-verlet-scope` and §8's two withdrawals are the template |
| 3 | **a description of current machinery** — "the reverse-channel increment is applied after the Verlet step" | **rewrite**: this is the only real problem, and it is concentrated in the sections the plan wants to promote |

### Priority queue — explicit-integrator mentions with **zero** CfC/BAOAB

*(Counts below were taken by grepping "Verlet", the word the text currently
uses. Per the terminology decision above, each one is either a
**Störmer–Verlet** reference that needs the qualifier corrected, or a
period reference that becomes "the explicit-integrator era".)*

These are where a reader meets an unsignposted Verlet-era claim.

| file | renders as | verlet | cfc | tier | note |
| --- | --- | ---: | ---: | --- | --- |
| `17c_fock_parflm` | **§20** Fock-Augmented PARFLM | **9** | **0** | **3** | **highest priority.** Slated for promotion to the book's centre, and its mentions are structural: `eq:fock-gamma-verlet`, "the reverse-channel increment is applied *after* the Verlet step", "the residual is dominated by the discrete velocity-Verlet truncation error". Describes today's mechanism with yesterday's integrator |
| `17_parf_augmented_splm` | **§19** PARF-Augmented SPLM | 8 | 0 | **3** | "a depth-$L$ stack of damped velocity-Verlet integrators", "per-layer constants matching the velocity-Verlet integrator" — the architecture is defined in Verlet terms throughout |
| `18h_portable_potentials` | **§35** Portable Learned Potentials | 10 | 0 | **mostly 2** | a taxonomy column, "minimiser (Verlet) or sampler (O-step Langevin)" — defensible as a category; check whether the port table's integrator column is current |
| `16_hybrid_splm` | **§18** Hybrid SPLM | 2 | 0 | 3 | "damped velocity-Verlet update under the causal-flow invariant"; one table row is a labelled control and can stay |
| `17b_cross_architecture_vreg` | **§21** Cross-Architecture Analysis | 2 | 0 | 2 | "only the velocity-Verlet damped dynamics…" — a claim whose scope needs checking |
| `15a_causal_integrity` | **§17** Causal integrity | 2 | 0 | 1–2 | one is a genuine Verlet-stiffness instability — tier 1 material, keep and cross-reference |
| `18j_relation_to_flow_matching` | **§34** Flow Matching | 1 | 0 | 3 | "Euler--Lagrange equation yields the damped Verlet…" |
| `19_conclusion` | **§38** Conclusion | 1 | 0 | 3 | "unrolled at every layer as a velocity-Verlet integrator" — the conclusion should describe the current model |

**Sequencing.** This runs alongside the correctness audit rather than
replacing it: **§20** (`17c_fock_parflm`) is both the top of this queue and a section the
plan already wants promoted, so it is the natural next target and the two jobs
are done in one pass.

---

## 2b. Optimisation disclosures — a missing thread, opened **2026-09-28**

**The gap.** The book discloses "AdamW with grad-clip $1.0$" (§16 twice, §22
in passing) and **nowhere mentions the nine per-group overrides**. A reader
told only "grad-clip 1.0" infers something far milder than what runs: the
tightest override is 0.1 on the reverse channel, and on every arm that has
one it is the largest pre-clip group on essentially every step, at a median
10–20× its threshold, maximum 88×. The thresholds came from L=8/L=16
forensics and were never re-derived for the ladder's depths.

**Done so far:** `rem:gate-clipping` in **§27**, next to "What the geodesics
cost" — the disclosure plus the three bounds on what it can cost, and an
explicit statement that the direction of any residual bias flatters the
mechanism rather than its critics. Build clean at 483 pp.

**Still owed, and it is a thread rather than a remark.** One remark inside
the geometry chapter is the minimum, not the treatment. What a careful
reader — or a TMLR referee — will actually ask is two questions the book
currently cannot answer:

1. **What do the clips cost in accuracy?** Partly bounded (C-series C0–C3:
   the gate falls rather than being held up; Adam absorbs a persistent
   rescale; the group is scalars so allocation is untouched). Not bounded
   for the step-to-step variation in the clip factor. **The measurement that
   would close it is C1 and it is free.**
2. **What do the clips cost in CONVERGENCE?** *Entirely unexamined.* Every
   argument assembled so far is about where training ends, not how long it
   takes to get there — and variance reduction on a parameter's updates is
   exactly the kind of thing that changes the second without changing the
   first. Nothing in the C-series addresses it. The data to start is already
   on disk: the per-group pre-clip norms are in every run log, so the
   fraction of steps on which each group binds can be plotted against the
   loss curve for all five completed arms, at no compute cost.

**Where it should live in the book.** Not in §27. The natural home is the
experimental protocol, beside the other training details, with §27's remark
cross-referencing it rather than carrying the argument alone. Candidate: a
short "Optimisation and its confounds" subsection in **§16**, which is where
grad-clip 1.0 is first stated and where the prescriptive-test setup is laid
out.

**Sequencing.** Write it after the C-series reports, so the subsection states
measurements rather than caveats. Until then §27's remark is the honest
placeholder and says so.

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
| 18c | memorisation capacity | **review** | any capability argued *from* geodesicity needs re-basing | F1 |
| **18d** | **geometric capabilities (§37)** | **keep, retreat, retitle** | second-largest retreat after §18; splits into three groups, one of which inverts. See §3.1 | F1, F4, and the ladder's conservativity gap |
| 18e | relation to liquid neural networks | **promote** | CfC is that lineage and is now the production integrator | — |
| 18j | relation to flow matching | **retreat, small** | any refinement / continuous-time equivalence language goes | none |
| 19 | conclusion | **rewrite** | last | after everything |
| 20 | dynamical simulator | **reframe** | "simulation" survives (it is a numerical integrator); of a *driven* system; the exact propagators are the strongest part and stay | E5 |
| A0 | edition history | **update** | this edition's entry | — |
| A1 | non-autonomous framework | **read first** | a time-dependent forcing from a slowly changing register bank is a non-autonomous system with an adiabatic parameter; the reformulation should be written *in* this appendix's language, and A1 may move into the main text | — |
| A2 | inference efficiency | **keep** | — | — |
| A3 | experiment index | **update** | E1–E5, F1–F5, ladder runs | as results land |

### 3.1 §37 "Capabilities Unique to the Conservative Design" — keep, retreat, retitle

**Keep.** Once the mechanism ladder (§2) prices conservativity at roughly
3 PPL, the reader's next question is what that price buys. A paper that
measures a cost precisely and states no corresponding benefit is weaker,
not more honest. §37 is where the benefit lives, and it is also where the
conservative members of the family (SPLM, PARFLM, `attention_potential`)
have their value stated at all.

**But three things in it are now wrong, and they separate cleanly** — along
exactly the line the new subtitle draws.

*Group A — from the conservative potential. Survives, and is now
**established** rather than assumed.* E1 on the conservative-only arm
(master doc §4.9) measures the trained step at the clean layer as the
damped Vθ geodesic step followed by LayerNorm, to R = 0.0003. §37's first
two rows — "well-defined Riemannian metric" and "computable geodesics" —
are no longer claims resting on Verlet-era cosines of 0.52–0.75; they are
exact for this architecture, with a measured price of 31.3%. **This is the
strongest single result available to §37 and it should open the section.*** The
Jacobi metric (row 1), sectional curvature from the Hessian of
$V_\theta$ as a native uncertainty measure (row 4), energy bookkeeping
(row 3), and the mechanistic reading of the trajectory (row 5). These are
properties of $V_\theta$ and the integrator, exact by construction, and
E1 does not touch them. E1's own `geo` arm — gate-0 validated bit-exact —
*is* the metric being used.

*Group B — from the auxiliary register degrees of freedom. Survives, but
it is the non-conservative half.* The register lifecycle (row 6) and
native chain-of-thought (§37.4). **The section's title claims these for
"the conservative design", and they are not.** They exist because the
Conservative Obstruction Theorem proved conservativity alone insufficient
and forced auxiliary state in. This is the retitle: the capabilities are
unique to *this architecture*, and they come from both halves — which is
the paper's thesis, not a concession.

*Missing entries to add, and they now come as a matched pair.*
`'attention'` and `'attention_potential'` differ in exactly one thing —
whether the same 589,824-parameter exchange field enters as a force or as
the gradient of a potential — so the trade-off can be stated without any
cross-arm hand-waving (flow/maps §8.4):

| | `'attention'` | `'attention_potential'` |
| --- | ---: | ---: |
| settled PPL | **63.51** | 80.90 (**+27.4%**) |
| refinement at N = 8 | 31.0× | **6.46×** (**4.8× more robust**) |
| depth extrapolation at N = 8 | 62.1× | **39.5×** |

**The price of conservativity is prediction quality; what it buys is
dynamical robustness.** That is the single strongest entry available to
§37, and it is measured rather than argued. The conservative arm is the
*second worst* on perplexity and the *second best* on refinement — so the
section's organising claim should be the trade-off, not a list of
capabilities.

The wider fact behind it: refinement brittleness tracks **non-conservative
content**, not capacity or parameter count. Adding a whole extra context
mechanism as a gradient costs nothing in brittleness (6.46× against
`'none'`'s 6.35×); adding the same mechanism as a force multiplies it
fivefold.

*Group C — presupposed geodesic compliance of the trajectory. Broken.*
Row 2's directional cosine 0.52–0.75, the G1 directed geodesic analogy,
G3/G4 asymmetric geodesic distance, and the geodesic tiers of the
leak-audit kit. The section states the presupposition explicitly — that
the models sit in a weakly-damped regime where
$\gamma_{\mathrm{eff}} \approx 0.13$ "preserves approximate geodesic
compliance" — and under CfC+BAOAB that is false: R(geo) = 1.09, and the
mechanism is not damping at all but the transverse reverse-channel force
(master doc §4.8).

**The inversion, which is the interesting part.** Group C's *instruments*
survive with their interpretation turned around. The leak-audit kit
measures deviation from the geodesic and reads it as a fault signal; the
reformulation says deviation from the geodesic **is the forcing** — the
informative quantity, not the error (reformulation §2.4). The kit was
measuring the right thing and calling it the wrong name. Re-based, it
becomes F4, and the three-tier hierarchy keeps its structure. Likewise
G3/G4's asymmetry: an asymmetric distance is what a *forced* system
produces, and the Tversky connection is if anything better motivated by a
forcing term than by a damped geodesic.

**Stale numbers to fix.** The section opens on "9.04 PPL versus ~7.8 matched attention" — **TinyStories at 16k steps**, a
different corpus and scale from the OWT ladder's 66.98 versus 49.81. Both
belong, but they must be labelled, and the framing gap should be the
ladder's. Row 2's cosine figures come from the Verlet-era §18 battery and
need the same label.

**The corpus point, which the section should absorb rather than dodge.**
Which capabilities are exhibitable depends on what the corpus rewards:
curvature-as-uncertainty and asymmetric distance are claims about semantic
structure, and a corpus whose structure is mostly local may not exercise
them. That is F6 (reformulation §3.6) reaching §37, and it argues for
keeping the section and conditioning it, not for dropping it.

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

**Agreed 2026-09-25, and APPLIED to `paper_v6` the same day** (branch
`paper_v6_9-25-26`; `main.tex` title block, `abstract.txt` title line,
`A0_edition_history` v6 entry; rebuilt clean at 475 pages):

> **Semantic Simulation: A Prescriptive Lagrangian Framework for
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

**"Efficient" was dropped, 2026-09-25.** That claim is about inference
cost (Appendix A2), while every headline number now in play is a quality
ratio (1.345x the matched GPT-2 at L=2), so the word invited exactly the
comparison the paper then has to spend a section qualifying. The
efficiency argument itself is untouched and stays where it belongs, in
A2 --- it is the title that no longer advertises it.

`\ShortHeadings` needs no change: it reads "Semantic Simulation:
Prescriptive Lagrangian Framework" and never carried the word.

---

## 6. Order of edits, once the decision points are in

**Rendered section numbers throughout, with the filename stem beside each,
because the two do not match.** See the audit's file-to-section map.

1. A1 read; decide whether it moves into the main text (**audited
   2026-09-27: it stays an appendix**).
2. **§7** (`07_lagrangian`) and **§8** (`07a_position_dependent_damping`): the forced Lagrangian, damping is tangential.
3. **§27** (`18_riemannian_geometry`): the retreat. Three-layer table,
   residual re-based, refinement language out.
4. **§14** (`13_stp_acceleration`): promote; connect the STP loss to κ_g and
   to F1's distribution.
5. **§20** (`17c_fock_parflm`): promote; the register bank as the driver.
6. **§15** (`14_experiments`): the experimental record, including the misses.
6b. **§16** (`15_conservative_architectures`): "Optimisation and its
   confounds" — the per-group clips, what they cost in accuracy and in
   convergence (§2b). After the C-series reports.
7. **§11**, **§26**, **§36**, **§37**, **§28**: reframe as their decision
   points allow.
8. **§1**, abstract, **§38** (conclusion): the thesis sentence, last — **and
   the title/subtitle with it** (§5), including the open "Efficient" question.
9. A0, A3.

---

## 7. Open decision points

| decision | settled by | changes |
| --- | --- | --- |
| "memory steers" vs "memory steers occasionally" | **F1 — settled 2026-09-27**: all four L=2 arms read UNIFORM, so **"memory steers"**, no hedge | the thesis sentence; §14, §27 |
| is L=2 a floor or the verdict on the geodesic share? | **F2** (needs L=4) | every "at L=2" caveat |
| can the forecastability claim be made at all? | E3 at L ≥ 3 | §11, §26 |
| the price of the pure geodesic in PPL | **E5 — settled 2026-09-25**: 3.91×, no knee; layer 1's output direction is the register readout | §20, §28 — and §20 must describe the last layer as a memory read, not a forced step |
| is V_φ ever active? | **F5 + E1 — settled 2026-09-25: no.** Trained with no competition it still moves the step by −0.0015. Its inertness is a V_φ/PARF matter | §5, §19 must stop describing V_φ as load-bearing |
| does the hallucination claim survive re-basing? | F4 | §37 and wherever hallucination is discussed |
| is the dominance of the forcing a property of the corpus? | F6 (TinyStories) | every "the reverse channel dominates" sentence gains "on OpenWebText" until then; §15, §20 |

---

## 8. Ledger

| date | item |
| --- | --- |
| 2026-09-25 | skeleton opened; triage table drafted; no paper edits beyond Remark 52 / footnote / pointer |
| 2026-09-25 | E5 settled its decision point (§5); F1 remains the gate on the thesis sentence |
| 2026-09-25 | mechanism ladder named as the headline result (§2); `attention_potential` band revised to 66 (62–72) after re-reading the detach semantics — protocol §5.3a |
| 2026-09-25 | §37 (geometric capabilities) triaged: **keep, retreat, retitle** — three groups, one inverted; §3.1 |
| 2026-09-25 | **`paper_v6` forked from v5** (branch `paper_v6_9-25-26`), build artifacts cleared, new A0 edition entry, title and subtitle applied, rebuilt clean: 475 pages, 0 undefined refs/cites, 0 errors |
| 2026-09-25 | subtitle agreed (§5): *Conservative Potentials, Non-Conservative Memory, and the Shared-Potential Separator*; JEPA correspondence demoted to the body; "Efficient" in the main title left open |
