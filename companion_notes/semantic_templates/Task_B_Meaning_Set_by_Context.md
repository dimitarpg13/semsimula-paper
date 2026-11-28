# Task B: Meaning Set Only by Context

**Status (2026-11-27).** Design, items and calls drafted by the agent at the author's request ("let us run Task B on its own"), **for the author's review. Nothing has been extracted or scored.** The items are on [bcontext_items/](bcontext_items/README.md); the calls in §5 wait for the author's confirmation, and the gate of §4 is read before any model reading.

**In brief.**
- **The question.** Can a template recognise a valid logical form when the second sentence of the premise is ambiguous, and only a context sentence before the premise says what it means? This is task B as first proposed on 2026-11-27, alongside task A (the form at "Therefore") and task C (roles). Until now it ran only in its hybrid form, B-logic ([BLogic_Test.md](BLogic_Test.md)).
- **Two kinds of ambiguity.** A **word sense**: in "The bank is closed." the bank is the lender or the embankment, and the clauses say each in other words. A **pronoun**: in "He is tired." he is John or Bill, and the clauses name both.
- **Why the words cannot help.** Within a frame the premise is the same string at the same position in every version; only the context differs, and the context is not one of the matcher's particles. Token identity and the embedding tie at exactly 0.5 by construction.
- **Four flavours of context per kind.** Explicit, natural, and two controls that name both meanings, so that binding can be told from mention, as the B-logic control did.
- **The difference from B-logic.** There, an arbitrary signal ("the blue flag is up") was given a meaning by definition. Here the second sentence already has meanings, and the context has to choose between them, as in ordinary text.
- **Size.** 3,600 premises per model: 20 word-sense and 20 pronoun frames, six wordings, eight levels. To run on SR2-NP, G2, L4 and the matched GPT-2 W2, then on SR2-NP-L8.

---

## Contents

1. [Definitions](#1-definitions)
2. [The design](#2-the-design)
3. [What is read and scored](#3-what-is-read-and-scored)
4. [The gate](#4-the-gate)
5. [Draft calls](#5-draft-calls)
6. [What the results would mean](#6-what-the-results-would-mean)
7. [Hybrids with other test sets](#7-hybrids-with-other-test-sets)
8. [How it would be run](#8-how-it-would-be-run)
9. [Caveats](#9-caveats)

---

## 1. Definitions

- **Premise, templates, rivals.** As in every template test ([Template_Matcher_From_Geometry.md](Template_Matcher_From_Geometry.md), §1): two clauses P and Q joined by a connective, a second sentence S, then "Therefore"; the templates are MP, MT, DS1 and DS2.
- **Frame.** The fixed parts of a set of premises: the two clauses and the ambiguous second sentence, written in each of six wordings (if-then, when, provided; either-or, or, or else).
- **Version.** One premise of a frame with one context sentence before it. Versions of a frame differ only in the context.
- **Word-sense frame.** S contains an ambiguous phrase W ("the bank"); P says the predicate of one sense in other words ("the lender is closed"), Q of the other sense ("the embankment is closed").
- **Pronoun frame.** S has a pronoun ("he is tired"); P and Q say the predicate of two people of the same gender ("John is tired", "Bill is tired"). A third person ("Tom") gives the foil.
- **Binding** is what the context fixes: which meaning S takes. **Mention** is which meanings the context names. A state that marks mentions can separate versions whose contexts name different meanings; only a state that carries the binding can separate versions whose contexts name the same ones.

## 2. The design

### 2.1 What the context makes of a premise

| the second sentence | conditional frame | disjunctive frame |
| --- | --- | --- |
| affirmed, meaning P | MP, valid | not used |
| affirmed, meaning Q | AC, twin of MP | not used |
| denied, meaning Q | MT, valid | DS2, valid |
| denied, meaning P | DA, twin of MT | DS1, valid |
| about the third person (pronoun frames only) | foil | foil |

Example, word sense, conditional, affirmed:

```text
We were talking about money and loans.

If the lender is closed, then the embankment is closed. The bank is closed. Therefore      -> MP
```

```text
We were walking by the water.

If the lender is closed, then the embankment is closed. The bank is closed. Therefore      -> AC
```

The premise lines are identical; the first context makes it modus ponens, the second its twin.

### 2.2 The flavours of context

| level | kind | flavour | example | what it names |
| --- | --- | --- | --- | --- |
| SX | word sense | explicit | `Here, "the bank" means the lender.` | the selected meaning |
| SN | word sense | natural | `We were talking about money and loans.` | no clause word: a topic |
| SCa | word sense | control, bound first | `Here, "the bank" means the lender, not the embankment.` | both meanings |
| SCb | word sense | control, bound last | `Here, "the bank" is not the embankment; it means the lender.` | both meanings |
| PX | pronoun | explicit | `Here, "he" means John.` | the selected person |
| PN | pronoun | natural | `John was the only one in the room.` | the selected person |
| PCa | pronoun | control, bound first | `John came into the room while Bill stayed outside.` | both people |
| PCb | pronoun | control, bound last | `While Bill stayed outside, John came into the room.` | both people |

**SN is the cleanest test of meaning in context.** Its contexts name no word of the clauses, so neither mention nor binding by definition is available: the state must connect "money and loans" to "the lender" through what the words mean. The pronoun levels have no such flavour, since a referent cannot be chosen without naming someone; their control is the natural sentence that names both people.

### 2.3 The items

20 word-sense frames and 20 pronoun frames, listed with every natural context on [bcontext_items/](bcontext_items/README.md). The word-sense frames use common ambiguous nouns (bank, bat, pitcher, crane, mouse, seal, glasses, plant, deck, spring, pupil, letter, organ, trunk, fan, note, board, jam, tank, match), with a predicate that fits both senses. The pronoun frames alternate "he" and "she", with a different trio of names in each. The review page checks that no natural word-sense context shares a content word with its clauses, that the two senses share no word, and that every clause can be denied.

## 3. What is read and scored

- **Readings**, exactly as in B-logic ([BLogic_Test.md](BLogic_Test.md), §1):
  - the geometry matcher on every vector, with its clouds fitted on the 60 verbatim development clauses;
  - the state at "Therefore" read by task A's clouds ([Summary_State_Test.md](Summary_State_Test.md)), the reading that carried the binding in B-logic.
- **Pair readings, per level**, means over the 20 frames, intervals from 4,000 resamples of the frames:
  - **twin**: MP against AC on the MP template, MT against DA on the MT template; chance 0.5;
  - **other**: DS1 against DS2 on the DS1 template and the reverse; the two are both valid, so this asks whether the right one of the two templates scores higher; chance 0.5;
  - **foil** (pronoun levels only): each valid form against the version about the third person; chance 0.5.
- **The best reading at "Therefore"** is the best of the states after layers 0 and 1 for the two-layer models, as in the B-logic calls; for L4, SR2-NP-L8 and GPT-2, the best single state, named in the report.
- **The templates rank; they do not fire.** As in B-logic, the scores ask whether the version the context makes valid scores higher on its template, not whether it crosses a threshold. Task B's premises are also a natural calibration set for the firing test ([Template_Firing_Mechanisms.md](Template_Firing_Mechanisms.md), §4.2).

## 4. The gate

**TG1.** On every model and level, token identity's and the embedding's twin, other and foil readings are exactly 0.500 (every pair tied), and the check premises match the stored premises to within 0.001. A failure stops the scoring of that level. The build already checks the construction without a model: within every frame, level, wording and polarity of the second sentence, the particle tokens and their positions are identical (1,440 groups, all identical; `template_bcontext_items.py`).

## 5. Draft calls

For the author's confirmation; not binding until confirmed. A call hits when the point estimate is above 0.5 **and** the lower end of its interval is above 0.5.

| id | call | probability |
| --- | --- | ---: |
| TB1 | Pronoun, explicit (PX): on at least one model, the reader at "Therefore" has a foil reading above 0.5. | 60% |
| TB2 | Pronoun binding (PCa and PCb): on at least one model, the reader at "Therefore" has a twin reading above 0.5 in both. | 40% |
| TB3 | Word sense in natural context (SN): on at least one model, the reader at "Therefore" has a twin reading above 0.5. | 30% |
| TB4 | Word-sense binding (SCa and SCb): on at least one model, the reader at "Therefore" has a twin reading above 0.5 in both. | 40% |
| TB5 | The matched GPT-2's best twin reading at "Therefore", averaged over SN, SCa, SCb, PCa and PCb, is above SR2-NP's. | 55% |

**Reasoning.**
- **TB1.** The explicit definition worked in B-logic (its foil readings were the strongest), and the foil's context names a person who is not in the clauses, so even a mention would separate it.
- **TB2.** A natural sentence that names both people leaves the referent to a preference (the subject of the main clause), which a two-layer model may not have.
- **TB3.** Linking "money and loans" to "the lender" needs word meaning that small models trained on 0.5B tokens may hold only weakly; test 2's word-sense level C2 followed the false friend on every reading ([Analogy_Test_Results.md](Analogy_Test_Results.md)).
- **TB4.** The B-logic control, also an explicit definition naming both clauses, hit with twin readings of 0.69 to 0.84; here the definition also has to override the phrase's own meaning, so lower.
- **TB5.** GPT-2 is the better language model at these tokens (48.37 against 58.51), and word sense and reference resolution grow with language-model quality; at its best it was level with SR2-NP on B-logic, hence an even call leaning to GPT-2.

**Not called:** the other readings (DS1 against DS2), the natural pronoun level PN (mention and binding coincide there), the geometry matcher's readings, and differences between the two-layer models. They are reported as leads.

## 6. What the results would mean

| outcome | reading |
| --- | --- |
| TB1 hits, TB2 misses | the states carry who was named, not who "he" is |
| TB2 hits | the states resolve a pronoun from a natural sentence and carry the referent to the summary position |
| TB3 hits | the states choose a word's sense from a topic that names none of the clause words: meaning in context, with no definition and no mention |
| TB4 hits, TB3 misses | an explicit definition can override a phrase's meaning, but a topic cannot choose it |
| everything misses at two layers | the question goes to depth: SR2-NP-L8 and L4 |

## 7. Hybrids with other test sets

The author suggested combining task B with other test sets. Each hybrid keeps task B's construction (identical premise strings, meaning set by context) and adds the structure of the other test.

| hybrid | what it adds | status |
| --- | --- | --- |
| B with the logical forms | an arbitrary signal given a meaning by context | done: B-logic and its control |
| B with task C (roles) | a pronoun whose referent, set by context, decides who did what: "John pushed Bill. He fell." | after task C is drafted |
| B with test 2's paraphrases | a second sentence that paraphrases a clause only under one sense ("The financial institution is shut.") | needs new items |
| B with the firing test | task B's premises as the calibration set of a firing rule set on context-defined premises | when the firing test is drafted |
| B with depth | the battery on SR2-NP-L8 (eight layers, the same recipe as SR2-NP) | when L8 lands |

## 8. How it would be run

Nothing below has been run. The extraction reuses the B-logic path with a new item bank (`bank=bcontext`):

```bash
cd notebooks/conservative_arch/scaleup
python3 debug/template_bcontext_items.py                                       # builds the 3,600 premises; checks the particles are identical within each group
python3 debug/template_analogy_test.py RUN FOLDER name=np vec=1 bank=bcontext   # extraction; also g2, l4 and the matched GPT-2
python3 debug/template_analogy_analysis.py DEPTHDIR RUN results/template_matcher np,g2,l4 gates items=template_bcontext_items tag=bcontext
python3 debug/template_analogy_analysis.py DEPTHDIR RUN results/template_matcher np,g2,l4 items=template_bcontext_items tag=bcontext
python3 debug/template_summary_state.py VECDIR results/template_matcher np,g2,l4 analogy=RUN items=template_bcontext_items tag=bcontext       # task A's reader
```

These are the B-logic commands of [BLogic_Test.md](BLogic_Test.md), §4, with the new item bank; the extraction script's `bank=bcontext` option is registered. A dry run on a subset comes first, with no score printed. Expected extraction time: about 40 minutes per model on eight threads (3,600 premises against B-logic's 1,200 in 12 to 20 minutes).

## 9. Caveats

- **English premises drafted by the agent;** the word-sense frames need the author's eye on whether each predicate fits both senses and whether each topic selects the intended sense.
- **The pronoun frames use names;** names carry frequency and gender cues that differ between frames, though not between versions of a frame.
- **One checkpoint and one seed per model.**
- **The word-sense paraphrases are not synonyms of the ambiguous phrase** ("the lender" for "the bank"), so even a perfect reader must connect the paraphrase to the right sense; that is part of what is tested.
