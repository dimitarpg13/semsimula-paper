# The Template Programme: Where It Stands and What Comes Next

**What this note is.** A synthesis of the measurement side of the semantic-template programme, written on 2026-11-27 at the author's request: what has been tested, what is established and how firmly, what is open, and the plan for the next experiments and investigations. It adds no new measurement; every claim links to the report that holds it. The design side of the programme (how templates would be built into the model) is in [Implementing_Semantic_Templates_A_Path.md](../Implementing_Semantic_Templates_A_Path.md); the terminology is in [README.md](README.md). Update this note when a test lands.

**In brief.**
- **A working template matcher exists.** One machine, the geometry matcher, fires a logical form's template from the geometry of a premise's particles, in a subspace of about ten dimensions, with no word list. On verbatim premises it identifies the four valid forms 92 to 97% of the time with 1% false fires.
- **On verbatim premises its success is mostly string matching.** Token identity alone scores 0.906 on unseen wordings; the embedding's lead over it is word similarity.
- **Where the words cannot help, the layer states carry what the embedding cannot.** Read at the final token "Therefore", the states identify the form (54 to 62% against 20%, chance 25%) and which clause was restated (AUROC 0.75 against 0.52); and when only a context sentence makes a premise valid, they carry which clause the context bound the signal to (0.57 to 0.97 against the foil, 0.69 to 0.84 against the twin, in a control that names both clauses).
- **Templates rank; they do not yet fire** outside verbatim premises. A firing rule for the summary-position reader is the next engineering step.
- **Depth looks like it helps, but no clean depth comparison exists yet.** SR2-NP-L8 (training, due 2026-11-28) and the matched GPT-2 (to be read with the same tests) are the two runs that decide the next claims.

---

## Contents

1. [The question](#1-the-question)
2. [What has been tested, in order](#2-what-has-been-tested-in-order)
3. [What is established](#3-what-is-established)
4. [What is open](#4-what-is-open)
5. [The plan](#5-the-plan)
6. [Rules the programme follows](#6-rules-the-programme-follows)

---

## 1. The question

A semantic template is a recogniser for a pattern of meaning; the first templates are the logical forms modus ponens, modus tollens and the two disjunctive syllogisms, which must fire on valid premises and stay silent on their fallacy twins (affirming the consequent, denying the antecedent, the affirmed disjunct). The programme asks two things: can a template be read from a trained conservative model's states, and can it be built into the model's landscape. The measurements so far answer the first.

## 2. What has been tested, in order

| date | test | main result | report |
| --- | --- | --- | --- |
| 2026-11-25 | logical forms as frozen figures of seven node states | a form is recognised 85 to 88% of the time from 21 distances; the embedding does better than the final state | [Logical_Forms_as_Geometric_Figures.md](Logical_Forms_as_Geometric_Figures.md) |
| 2026-11-25 | the geometry matcher on four checkpoints | 92 to 97% on known wordings, about 10 dimensions; the embedding as good on unseen wordings | [Template_Matcher_From_Geometry.md](Template_Matcher_From_Geometry.md) |
| 2026-11-25 | the matcher on the dynamics (SR2-2B) | the displacement and the well frame fall below the embedding on unseen wordings; the states catch 2 to 8 points more paraphrases; "unless" misfiled | [Template_Matchers_on_the_Dynamics_SR2_2B.md](Template_Matchers_on_the_Dynamics_SR2_2B.md) |
| 2026-11-25 | depth: G2 and L4 | the early states catch more paraphrases; the flow gap does not fall with depth | [Template_Matchers_and_Depth_L4.md](Template_Matchers_and_Depth_L4.md) |
| 2026-11-26 | hybrids, steps 0, 1a, 1b | the states carry extra separable information, small when held out | [Hybrid_Step1b_Null_Check.md](Hybrid_Step1b_Null_Check.md) |
| 2026-11-26 | the runner-up | the early state is level with the embedding on wordings and ahead on paraphrases (+0.05 to +0.16) | [Early_State_Matcher_Runner_Up.md](Early_State_Matcher_Runner_Up.md) |
| 2026-11-26 | test 1: the lexical baseline | token identity reaches 0.906; the verbatim test is mostly string matching | [Lexical_Baseline.md](Lexical_Baseline.md) |
| 2026-11-26 | test 3: a metric fitted per reading | run; report pending | (pending) |
| 2026-11-26 | test 2: analogy and in-context meaning | the embedding stays above chance on every analogy level; the states add nothing on two layers; nothing above chance on C1 | [Analogy_Test_Results.md](Analogy_Test_Results.md) |
| 2026-11-27 | task A: the form at "Therefore" | the states carry the form and which clause was restated; the embedding does not | [Summary_State_Test.md](Summary_State_Test.md) |
| 2026-11-27 | task B-logic and its control | a context sentence alone sets the form of an identical premise; the states at "Therefore" carry the binding, the matcher's particle coordinates only the mention | [BLogic_Test.md](BLogic_Test.md) |
| 2026-11-27 | the matched GPT-2 on every test | GA1, GA2, GA4 hit: a transformer's states carry form and binding to "Therefore" at SR2-NP's level, but only from the third block | [GPT2_Template_Battery_Results.md](GPT2_Template_Battery_Results.md) |

## 3. What is established

Each line states its standing: **measured** (read from a confirmed or exploratory run, with the run named), **derived** (follows from measurements by an argument given in the report), or **hypothesis**.

1. **The geometry matcher works on verbatim premises** (measured, four checkpoints): one cloud per kind in a small subspace, about ten dimensions, 92 to 97% detection at 1% false fires on known wordings.
2. **On verbatim premises it is mostly reading repeated words** (measured, test 1): token identity, with no learned vector, reaches 0.906 on unseen wordings; the embedding's lead is morphology and synonymy. The frame of a new wording is read by the rule "contains *or*".
3. **The embedding's word similarity carries most analogy** (measured, test 2): with string identity removed, the embedding stays above chance on paraphrases, entailed cases, pronouns, restructured clauses and new domains, because "storm" is near "rain" and "he" near "king".
4. **The layer states add something on paraphrases** (measured, three of four models): +0.05 to +0.16 caught at 1% near misses; level on identifying the form in new wordings.
5. **Information that must travel to one position is carried by the states and not by the embedding** (measured, task A, three models): at "Therefore" the form ranks first 54 to 62% (embedding 20 to 22%, length 41%), valid against twin AUROC 0.74 to 0.76.
6. **A context sentence's binding reaches the summary position** (measured, B-logic control, three models, both word orders): with token identity and the embedding tied at exactly 0.5 by construction, the states at "Therefore" rank the version the context makes valid above its foil and its twin. The matcher's particle coordinates show only which clause was mentioned.
7. **Templates rank; they do not fire** outside verbatim premises (measured): the 1% rule fires on 0 to 21% of analogy premises and on none of task A's single-vector readings.
8. **Depth helps on G2's recipe** (measured, one model per depth, confounded): G2 to L4 (the same recipe, two and four layers) lifts the B-logic control's foil reading from 0.57 to 0.97; within L4 the middle layers carry the most. Not yet measured on a clean recipe.
9. **The well frames and displacements are not better readers than the states** (measured, every test so far): the well frames fall near chance on the analogy items.

## 4. What is open

- **Does depth sharpen form and binding on a clean recipe?** SR2-NP against SR2-NP-L8 (only the depth changed).
- **Are the templates architecture-specific?** Partly answered 2026-11-27: the matched GPT-2 reads form and binding at "Therefore" at SR2-NP's level, so the readings are not specific to the conservative architecture; the difference is that GPT-2 needs three blocks where the ladder models need one ([GPT2_Template_Battery_Results.md](GPT2_Template_Battery_Results.md)). GA3 (SR2-NP-L8 against GPT-2) is open.
- **Can templates fire on context-defined premises?** A threshold on the summary reader's log odds, set on development premises.
- **Does binding extend to roles** ("the man was bitten by the dog" against "the dog was bitten by the man")? Task C.
- **Why did C1 fail** while B-logic succeeded? C1 was read only through the matcher's particle coordinates; it has not been read at the summary position.
- **What does the pair force, the reverse channel or the exact damped flow contribute?** SR2-NP reaches 0.90 at two layers where G2 reaches 0.57; the recipes differ in all three.
- **Can a template be installed** as a well whose basin is the premises of one form? The design question of [Implementing_Semantic_Templates_A_Path.md](../Implementing_Semantic_Templates_A_Path.md), untouched by measurement.

## 5. The plan

### 5.1 Running

- **SR2-NP-L8 on the GPU** (A100, layer checkpointing on): SR2-NP with eight layers; 76,594,057 parameters; launched 2026-11-27, step 2,000 at 2.7% above SR2-NP's validation perplexity, closing; about 3.3 s per step; finish expected around midday 2026-11-28, with one resume. Record: [Template_Matchers_and_Depth.md](Template_Matchers_and_Depth.md), §2.

### 5.2 Next, on the CPU, in order

1. ~~**Calls for the depth-and-architecture battery**~~ *confirmed by the author 2026-11-27* (SR2-NP-L8 and the matched GPT-2; predictions against SR2-NP as the two-layer reference).
2. **The matched GPT-2 extraction.** The matched GPT-2 has the same width (384), the same number of layers as SR2-NP-L8 (eight) and, as W2, the ladder's schedule, learning rate and 0.5B tokens (settled 48.37), so it compares with SR2-NP-L8 at equal depth. The extraction scripts build ladder models; a GPT-2 path (the matched baseline already used by the frequency-fifth script) has to be added and checked on a few premises before the full run. *Built and checked 2026-11-27; the battery ran on GPT-2 the same day; GA1, GA2, GA4 hit* ([GPT2_Template_Battery_Results.md](GPT2_Template_Battery_Results.md)). Calls: [Depth_and_Architecture_Battery.md](Depth_and_Architecture_Battery.md).
3. **The battery on SR2-NP-L8 and GPT-2**, when L8 lands: the depth premises with every state kept, test 2, task A, B-logic and its control; flow gaps for L8.
4. **The firing test** for the summary reader: log odds against rivals, threshold at 1% false fires on development premises, detection on test clauses and on B-logic premises; detection reported at 1%, 5% and 10%.
5. **Task C** (roles), drafted for the author's review as B-logic was.
6. **The report of test 3** (the fitted metric), run on 2026-11-26.
7. **C1 at the summary position**: the extracted C1 premises read by task A's reader (no new extraction).
8. **The equal-length rule** for the analogy items (the author's decision pending).

### 5.3 On the GPU, after SR2-NP-L8

- **SR2-L8**, the pair-force twin at eight layers (the author asked to settle the pair force's part sooner), then **SR2-NP-L4**, which completes the clean depth series.
- **Twin PA** (productization), after SR2-NP-L8, its slot set against the productization order and the 2B-token priorities.
- The productization order continues on the main GPU: PMX-K16-NP, then the tied arms of the grid.

### 5.4 Later investigations

- **Installing a template**: a well trained so that its basin holds the premises of one form, then read by the same tests. This is where the conservative architecture has a claim a transformer cannot make.
- **More frames per level** (60 instead of 20) for the levels that carry the main claims, so that a gain of 0.1 is resolved on one level.
- **The factorised hybrid** (approach F): the particle matcher for the frame and the clause match, the summary reader for the binding.

## 6. Rules the programme follows

- Calls are written before a test is read and confirmed by the author; the programme is exploratory, so readings without a call are leads to be confirmed by a later run.
- Items are drafted by the agent and reviewed item by item by the author; corrections go into the item source and the review log.
- Gates (token identity at chance, exact ties, extraction checks) are read before any model reading; a failed gate stops the level until it is fixed.
- The claim is stated at the strength the measurement allows: "rank" is not "fire", "mention" is not "binding", and a confounded comparison is named as such.
