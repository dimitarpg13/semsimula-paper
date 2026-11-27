# Analogy and In-Context Test Set: Draft Items

**What this folder is.** The first 20 frames per level of the test set designed in [Analogy_and_Context_Test_Set.md](../Analogy_and_Context_Test_Set.md), drafted by the agent on 2026-11-26 for the author's review. Nothing has been extracted or scored. The source of every sentence is `notebooks/conservative_arch/scaleup/debug/template_analogy_clauses.py`; these pages are generated from it by `notebooks/conservative_arch/scaleup/debug/template_analogy_review.py`, so corrections go into the source and the pages are regenerated.

**How the premises are built.** Each frame gives the clauses and, at its level, the restatements and foils of the second sentence. The six wordings (three conditional, three disjunctive) and the forms are composed as in the current test: modus ponens is a conditional with a restatement of P, modus tollens a conditional with a denial of Q, the disjunctive syllogisms a disjunctive with a denial of P or of Q, and the fallacy twins use the other clause. For the clause-level banks (L0 to L3, L5, L6) the second clause is another frame of the same bank.

**The matched-overlap rule.** Each restatement has a foil that shares exactly as many tokens with the clause (GPT-2 tokens, lower-cased), so token identity cannot prefer the restatement. The check below is computed, not judged by eye.

| level | page | frames | pairs checked | pairs that break the rule |
| --- | --- | ---: | ---: | ---: |
| L0 verbatim, L1 one word, L2 full paraphrase | [L0_L1_L2_base.md](L0_L1_L2_base.md) | 20 | 80 | 0 |
| L3 entailed case | [L3_entailed.md](L3_entailed.md) | 20 | 40 | 0 |
| L4 reference | [L4_reference.md](L4_reference.md) | 20 | 60 | 0 |
| L5 restructured | [L5_restructured.md](L5_restructured.md) | 20 | 40 | 0 |
| L6 new domain | [L6_new_domain.md](L6_new_domain.md) | 20 | 40 | 0 |
| C1 alias defined in context | [C1_alias.md](C1_alias.md) | 20 | 40 | 0 |
| C2 same word, sense differs; C2′ paraphrase against a false friend | [C2_sense.md](C2_sense.md) | 20 | 40 | 0 |
| C3 ellipsis | [C3_ellipsis.md](C3_ellipsis.md) | 20 | 40 | 0 |
| C4 denial by an incompatible value | [C4_denial_by_context.md](C4_denial_by_context.md) | 20 | 40 | 0 |

**What the author is asked to check, for every level:** that each restatement is the same statement as its clause (or implies it, at L3), that each foil is clearly a different statement and is neither implied nor excluded by the clause, and that nothing reads unnaturally. Items can be struck, replaced or rewritten; the overlap check reruns on the new text.

**Review log** (the author's corrections, applied in the source):

| date | item | change |
| --- | --- | --- |
| 2026-11-26 | L0 to L2, frame 3 | "the entrance is unbarred" replaced by "the entrance is unclosed": unbarred means unlocked, not open (author). |
| 2026-11-26 | L0 to L2, frame 11 | frame "the bread is stale" (L1 "dry", L2 "the loaf is dried out") replaced by "the merchant is wealthy" (L1 "rich", L2 "the trader is affluent", foils "tall", "talkative"): dry is a texture, bread can be dried on purpose and be fresh, and stale has no one-word synonym (author). |
| 2026-11-26 | L0 to L2, frame 14 | frame "the river is wide" (L2 "the waterway is expansive") replaced by "the sofa is wide" (L1 "broad", L2 "the couch is broad", foils "old", "soft"): expansive speaks of extent (length or area) rather than width (author), and a waterway is a broader term than a river (agent). |
| 2026-11-26 | L0 to L2, frame 4 | frame "the patient is asleep" (L2 "the sick man is dozing") replaced by "the infant is asleep" (L1 "sleeping", L2 "the baby is slumbering", foils "hungry", "swaddled"): a sick man is not the same as a patient (author); patient has no exact synonym, and dozing is a light sleep (agent). |
| 2026-11-26 | L3, frame 2 | denial case and foil "has just run a marathon / a bath" replaced by "has just finished a marathon / a novel": "to run a bath" is an idiom whose meaning is not clear to every reader (author), and it plays on two senses of "run" (agent). |
| 2026-11-26 | L3, frame 7 | case foil "scored three goals and conceded four" replaced by "scored three goals and received two yellow cards": three against four is a loss, so the old foil denied the clause instead of deciding nothing (agent). The denial case "lost every one of its games" stays: a denial case need only exclude the clause, not restate its denial (author). |
| 2026-11-26 | L3, frame 19 | kept: "the window has just passed its inspection" excludes "the window is broken" in the ordinary reading (a window that passed inspection was not broken, unless the inspector failed) (author). |
| 2026-11-26 | L4, all frames | redrafted: he and she only, unambiguous genders, predicates plausible for either person, P "he" in 10 frames and "she" in 10. The eight frames that paired "it" with a person (3, 5, 7, 9, 13, 15, 17, 20) gave absurd foils ("She is barking.") that a reader rejects as nonsense without resolving the reference; "the waiter" replaced (either gender); frames 12 and 16 reversed for balance (author and agent). |
| 2026-11-26 | C3, all frames | redrafted as L4 was: one man and one woman per frame, he and she only, unambiguous genders, verbs plausible for either person, P "he" in 10 frames and "she" in 10. Thirteen frames paired "it" with a person, so animacy alone told the restatement from the foil without recovering the clause (author and agent). |
| 2026-11-26 | C4, frame 2 | Q "the door is unlocked" with its denial "the door is bolted" and foil "the door is wooden" replaced by "the shutters are up", "the shutters are down", "the shutters are metal": a door can be unlocked and still bolted, so "bolted" does not deny "unlocked" (author); "bolted" also means fastened with bolts or ran off (agent). |
| 2026-11-26 | C4, frame 12 | foil of Q "the gauge points north" replaced by "the gauge points steadily": a fuel gauge does not point to a compass direction, so the old foil read as nonsense (agent). |
| 2026-11-26 | L0 to L2, frame 16 | frame "the meal is ready" (L1 "prepared", L2 "dinner is done") replaced by "the task is difficult" (L1 "hard", L2 "the job is tough", foils "urgent", "boring"): dinner is narrower than a meal and lacked its article, and "done" can mean finished eating (agent; author agreed). |
| 2026-11-26 | L0 to L2, frame 17 | frame "the hill is steep" (L1 "sheer", L2 "the slope is precipitous") replaced by "the film is boring" (L1 "dull", L2 "the movie is tedious", foils "long", "French"): sheer means vertical, a slope is part of a hill, and precipitous is rare and stronger than steep (agent; author agreed). |
| 2026-11-26 | L0 to L2, frame 18 | frame "the box is sealed" (L1 "shut", L2 "the crate is taped closed") replaced by "the gift is expensive" (L1 "costly", L2 "the present is pricey", foils "small", "wrapped"): a crate is not a box, and shut is not sealed and repeated frame 1 (agent; author agreed). |
| 2026-11-26 | L6 | subjects "the infection / the illness" replaced by "the disease is contagious / the illness is catching": the subjects are not synonyms (an infection is the invasion by a germ, an illness the state of being unwell) (author). |
| 2026-11-26 | L6 | subjects "the patient / the sufferer" replaced by "the pill is tiny / the tablet is minuscule" (foil "white"): the subjects are not synonyms (a patient is under care, a sufferer is anyone with an affliction) (author). |
| 2026-11-26 | L6 | subjects "the wound / the cut" replaced by "the operation is risky / the surgery is dangerous" (foil "lengthy"): the subjects are not synonyms (a cut is one kind of wound) (author). |
| 2026-11-26 | L6 | subjects "the valve / the tap" replaced by "the photocopier is broken / the copier is out of order" (foil "beige"): the subjects are not synonyms (a tap is one kind of valve) (author). The first replacement, the lift and the elevator, was dropped because "the elevator is stuck" was taken for a development clause; it is in fact a clause of the earlier robust-matcher test set (template_robust_clauses.py, TEST), not of the development set, so the reason was wrong. The photocopier frame is kept, as it serves equally (agent, corrected the same day). |
| 2026-11-26 | L6 | subjects "the battery / the cell" replaced by "the lorry is overloaded / the truck is overburdened" (foil "muddy"): the subjects are not synonyms (a battery is made of cells, and "cell" has other senses) (author). |
| 2026-11-26 | L6 | subjects "the circuit / the wiring" replaced by "the bicycle is rusty / the bike is corroded" (foil "folding"): the subjects are not synonyms (wiring is part of a circuit) (author). |
| 2026-11-26 | C4, frame 14 | kept: the denial "the race is called off" is a common phrasal verb and reads clearly; the proposed "the race is cancelled" is not needed (author). |
| 2026-11-26 | L3, frame 1 | foil of the denial case "the stands are crowded and the ground is hard" replaced by "... the ground is level": hard ground hints that it is dry, so the old foil leaned toward deciding (agent; author agreed). |
| 2026-11-26 | L3, frame 4 | clause "the metal is valuable" with denial case "the metal is scrap" replaced by "the metal is precious" with denial case "the metal is iron" (case gold and foils grey and shiny kept): scrap metal can be valuable, and iron is not a precious metal by definition (agent; author agreed). |
| 2026-11-26 | L3, frame 11 | foil of the case "the prisoner climbed onto the wall and sang" replaced by "the prisoner climbed a wall in the gym": the old foil was odd; a gym climbing wall says nothing about an escape (agent; author agreed). |
| 2026-11-26 | L5, frame 18 | restatement "the gate's opening was the guard's doing" and its foil replaced by "the opening of the gate was done by the guard" and "the painting of the gate was done by the guard": "was X's doing" is an idiom (agent; author agreed). |
| 2026-11-26 | C2, frame 2 | C2′ paraphrase "the willow paddle is stored in the outbuilding" replaced by "the batsman's willow is kept in the hut": a paddle is not a bat (agent; author agreed). |
| 2026-11-26 | C2, frame 9 | C2′ paraphrase "the iron he swings has just been bought" replaced by "the golfer's driver was bought last week": "he" had no antecedent, and an iron is only one kind of club (agent; author agreed). |
| 2026-11-26 | C2, frame 19 | foil "the dog bark is rough" replaced by "the dog's bark is rough": "dog bark" is not a natural compound (agent; author agreed). |
| 2026-11-26 | L6 | subjects "the appeal / the challenge" replaced by "the will is forged / the testament is faked" (foil "handwritten"): the subjects are not synonyms (a challenge is broader than an appeal) (agent; author agreed). |
| 2026-11-26 | L6 | subjects "the tumour / the growth" replaced by "the tumour is benign / the neoplasm is harmless" (foil "round"): the subjects are not synonyms (a growth is broader than a tumour; neoplasm is the medical synonym) (agent; author agreed). |
| 2026-11-26 | L6 | subjects "the turbine / the generator" replaced by "the aircraft is grounded / the plane is barred from flying" (foil "enormous"): the subjects are not synonyms (a turbine drives a generator; they are different machines) (agent; author agreed). |
| 2026-11-26 | L6 | subjects "the knight / the warrior" replaced by "the wizard is slain / the sorcerer is killed" (foil "bearded"): the subjects are not synonyms (a knight is one kind of warrior) (agent; author agreed). |
| 2026-11-26 | L3, 17 foils | every foil whose GPT-2 length differed from its restatement's was redrafted to the same length, overlap with the clause kept equal (frames 1, 3, 5, 6, 7, 8, 11, 12, 15, 17, 19, 20; frame 7's case foil is now "the team scored three goals and travelled home"): token identity preferred the foils (gate G1 of Analogy_Test_Calls.md failed, 0.385, interval 0.32 to 0.45), because the matcher's coordinates count unmatched particles, so length alone moved the score (agent; author approved). |

**Known limits of the draft.**
- C1 and C4 test the second sentence about one clause; the other forms of the frame (those that restate or deny the other clause) fall back on verbatim sentences.
- C4 relies on general knowledge (green denies red). A variant in which a context sentence sets the incompatibility is not drafted yet.
- L4 and C3 use gendered pronouns, which need the model to know the gender of "the king" or "the bride"; the embedding may carry part of that, which the readings would then show.
- All items are by one drafter, so the foils may share a style the restatements lack. The author's check and a later lexical check on style words limit this.
