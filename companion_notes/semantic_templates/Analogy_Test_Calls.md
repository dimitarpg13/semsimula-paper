# Test 2, the Analogy and In-Context Set: What Is Scored, and the Calls

**Status (2026-11-26).** The calls below were drafted by the agent and **await the author's confirmation**. The extraction is running, since it reads no score. Scoring waits until the calls are confirmed. The design is [Analogy_and_Context_Test_Set.md](Analogy_and_Context_Test_Set.md); the items, as reviewed by the author, are in [analogy_items/](analogy_items/README.md).

**In brief.**
- **Items.** 11 levels, 20 frames each, six known wordings (if-then, when, provided; either-or, or, or else): 6,720 premises per model, on SR2-NP, G2 and L4.
- **Matcher.** The matcher is unchanged. The clouds are fitted on the 60 verbatim development clauses of the earlier tests and never see an analogy premise. Nothing is tuned: each reading keeps the particle count and dimension of its earlier primary run.
- **Main reading.** For each pair of a restatement and its matched foil, does the restatement get the larger log odds for its template? Chance is 0.5. Token identity is at chance by construction from L1 on.
- **Uncertainty.** With 20 clauses per level, the 95% intervals are roughly 0.1 either side.
- **Calls.**
  - Three gates on the item set and the pipeline.
  - One anchor on the verbatim level.
  - Five calls on the embedding.
  - Four calls on the layer states, the question this test was built for.

---

## Contents

1. [What is scored](#1-what-is-scored)
2. [Gates, checked before any model reading is looked at](#2-gates-checked-before-any-model-reading-is-looked-at)
3. [The calls](#3-the-calls)
4. [What is not called](#4-what-is-not-called)
5. [Departures from the design](#5-departures-from-the-design)
6. [Reproduce and files](#6-reproduce-and-files)

---

## 1. What is scored

**Premises.** A premise is as in every earlier test: the two clauses joined by a connective, the second sentence, then "Therefore". The second sentence is the level's restatement, its matched foil, or a fallacy twin.

| frame | valid kinds (template) | matched foils | fallacy twins |
| --- | --- | --- | --- |
| conditional | MP affirms P; MT denies Q | one per valid kind | AC affirms Q; DA denies P |
| disjunctive | DS1 denies P; DS2 denies Q | one per valid kind | the affirmed P; the affirmed Q |

At L0 to L6 every kind is present. The C levels carry only the forms their items allow:
- **C1 and C2** have MP, DS1, their foils and the affirmed P. C2 also has the C2′ paraphrase.
- **C3** has all eight forms. Its foil is the other clause's pronoun, so "She does." is AC and "She does not." is MT.
- **C4** has MT, DS1, DS2, their foils and DA.

In C1 the context sentence and a blank line come before the premise. Its tokens are not particles.

**Development set (what the clouds learn from).**
- **Primary:** the depth analysis's own development set. It has 60 clauses, only verbatim kinds, and "unless" left out: 5,790 clause premises plus 600 of ordinary text.
- **Secondary, a check on shared nouns.** Many development clauses share a subject noun with an analogy item (39 of the 60; for example "the road is closed" is both a development clause and the L0 clause of frame 1). The clouds read relations between particles, never which word a particle is, so a shared noun gives them nothing to remember. To check this, a second fit uses only premises whose subjects share no noun with any analogy item: 404 development premises, 3,957 premises of the earlier test set, and the 1,200 of ordinary text.

**Matchers.**
- **lex:** token identity, as in [Lexical_Baseline.md](Lexical_Baseline.md).
- **h0:** the embedding.
- **The early state:** h1, the state after layer 0. On L4 also h2. This is the runner-up of [Early_State_Matcher_Runner_Up.md](Early_State_Matcher_Runner_Up.md).
- **states:** all states after layer 0.
- **path:** the displacements.
- **Also reported:** every other single state, displacement and well frame.

**Readings, per level.** Each is a mean over the level's 20 clauses, with a 95% interval from 4,000 resamples of the clauses.

| reading | what is compared | chance |
| --- | --- | ---: |
| foil | restatement against its matched foil, within clause and wording: the share of pairs in which the restatement has the larger log odds of its template (ties count half) | 0.5 |
| twin | the same, a valid form against its fallacy twin (MP and AC, MT and DA, DS1 and the affirmed P, DS2 and the affirmed Q) | 0.5 |
| foil C2′ | C2 only: the paraphrase in the clause's sense against the false friend | 0.5 |
| rank | the right template has the largest density of the four | 0.25 (see §4) |

The log odds of a template are the log of its clouds' summed densities over the summed densities of all other clouds. Every difference between two readings is paired by clause.

## 2. Gates, checked before any model reading is looked at

A gate that fails stops the scoring of that level until it is fixed.

- **G1 (the foils are matched).** On every level from L1 on, lex's foil reading has a point estimate from 0.40 to 0.60, and its interval contains 0.5. The exception is C2′, which is G2.
- **G2 (the false friend works).** On C2′, lex's foil reading is below 0.5.
- **G3 (the development premises are what this extraction gives).** The 96 check premises, rebuilt and extracted with the analogy items, have cosines within 0.001 of the stored ones on every kind. On the smoke run of SR2-NP the difference was exactly 0.

## 3. The calls

Each call names its model scope. "Above 0.5" means the interval lies above 0.5. "At chance" means the interval contains 0.5.

**Anchor.**
- **A1.** On L0 (verbatim, near-miss foils), lex and h0 have foil and rank readings of at least 0.90 on every model. This checks the pipeline, not a hypothesis.

**The embedding.**
- **E1.** On L1, h0's foil reading is at least 0.65 and above 0.5 on every model. One-word synonyms have nearby word vectors.
- **E2.** On L2 and L6, h0's foil reading is above 0.5 on at least two of the three models, and below its own L1 value on every model.
- **E3.** On C1, h0 is at chance on every model. Only the context links a name to a role.
- **E4.** On C2′, h0's foil reading is below 0.5 (point estimate) on every model. The false friend repeats the clause's words.
- **E5.** On L4 and C3, h0's foil reading is above 0.5 on every model, because the word vectors carry gender ("he" nearer "king" than "she" is). This is the call held with the least confidence.

**The layer states.**
- **S1.** On L0 and L1, the early state's foil reading is within 0.05 of h0's on every model. It keeps the repetition signal.
- **S2.** On C1, none of h0, the early state, states and path is above 0.5 on SR2-NP or G2. Two-layer models trained on 0.5B tokens are not expected to bind a name to a role from one context sentence.
- **S3 (the pattern the test was built to show).** The early state's gain over h0 in the foil reading is larger on the analogy levels than on L0 and L1. The analogy levels here are L2 to L6, C1, C2 and C3; C4 needs knowledge of the world and is left out. The call: on every model the difference of the two mean gains is above 0, as a point estimate; on at least one model its interval lies above 0. Levels are resampled independently.
- **S4.** On L2, h0's twin reading is at least 0.60 on every model. A paraphrase's subject is nearer its own clause's subject than the other clause's.

## 4. What is not called

- **The rank reading.** Its chance value is 0.25, but a reader that sees only "not" already decides two of the four choices, and lex reads "not". Its real floor is lex's own score, so it is reported against lex, not against 0.25.
- **L4's model on C1,** and every level not named above for the states. The deeper model may or may not bind names; there is no basis for a call.
- **Displacements, well frames and the single states other than the early state.** Reported, not called. The earlier tests put them below the embedding on the verbatim test.
- **The secondary readings:** the clean development set, and detection and false fires at the 1% rule's threshold.
- **The fitted metric of test 3.** Its report is pending, and it was fitted on test paraphrases, which this test may not use. The extraction keeps the particle vectors so that a metric fitted on development premises can be read later.

## 5. Departures from the design

- **20 frames per level, not 60.** The design asked for 60 to reach intervals of about 0.05 to 0.07. With 20, intervals are about 0.1, so a gain of 0.1 over chance may not be resolved on a single level. S3 pools levels to compensate. Drafting 40 more frames per level would need another review round.
- **Six wordings,** as the design proposed. All are known to the clouds: the question is the clause, not the connective.
- **Forms per C level** are those the items allow (§1). C1 and C2 test only the sentence about P, as their pages say.

## 6. Reproduce and files

```bash
cd notebooks/conservative_arch/scaleup
python3 debug/template_analogy_items.py                       # builds the 6,720 premises and checks that every pair is complete
python3 debug/template_analogy_test.py RUN FOLDER name=np vec=1   # extraction, about 15 minutes on SR2-NP; also g2, l4
python3 debug/template_analogy_analysis.py DEPTHDIR RUN results/template_matcher np,g2,l4   # after the calls are confirmed
```

Scripts in `notebooks/conservative_arch/scaleup/debug/`: `template_analogy_items.py` (premises and pairs), `template_analogy_test.py` (extraction: the depth extraction with the premise list replaced), `template_analogy_analysis.py` (scoring). Outputs will be `results/template_matcher/template_analogy_{np,g2,l4}.json`.
