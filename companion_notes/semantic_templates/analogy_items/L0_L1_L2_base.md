# L0 verbatim, L1 one word, L2 full paraphrase

**Draft items for review (2026-11-26).** Level of [Analogy_and_Context_Test_Set.md](../Analogy_and_Context_Test_Set.md). Source: `notebooks/conservative_arch/scaleup/debug/template_analogy_clauses.py`. Back to the [index](README.md).

**What this level tests.** One bank of 20 clauses read at three distances from the words, so the fall from L0 to L2 is paired by clause. L0 restates the clause word for word against a near miss (the current test). L1 changes one word of the predicate (subject kept) against the same near miss. L2 replaces the subject and the predicate by paraphrases, against a foil with the same paraphrased subject and an unrelated predicate: neither shares a content word with the clause.

**What to check:**
- Is each L1 paraphrase the same statement as the clause (not stronger, not weaker)?
- Is each L2 paraphrase the same statement, with no word of the clause left?
- Is each foil clearly a different statement, and not implied or excluded by the clause?

**Matched-overlap check:** 80 of 80 restatement and foil pairs share the same number of tokens with the clause.

| # | clause | near miss (L0, L1 foil) | L1 paraphrase | L2 paraphrase | L2 foil |
| --- | --- | --- | --- | --- | --- |
| 1 | The road is closed. | The road is long. | The road is shut. | The street is blocked. | The street is wide. |
| 2 | The coffee is cold. | The coffee is strong. | The coffee is chilly. | The brew is cool. | The brew is bitter. |
| 3 | The door is open. | The door is red. | The door is ajar. | The entrance is unbarred. | The entrance is narrow. |
| 4 | The patient is asleep. | The patient is tall. | The patient is sleeping. | The sick man is dozing. | The sick man is bearded. |
| 5 | The shop is busy. | The shop is new. | The shop is crowded. | The store is packed. | The store is painted. |
| 6 | The soldier is brave. | The soldier is young. | The soldier is fearless. | The fighter is courageous. | The fighter is famous. |
| 7 | The lake is frozen. | The lake is deep. | The lake is icy. | The pond is iced over. | The pond is shallow. |
| 8 | The car is fast. | The car is blue. | The car is quick. | The vehicle is speedy. | The vehicle is rusty. |
| 9 | The child is happy. | The child is short. | The child is cheerful. | The youngster is joyful. | The youngster is clever. |
| 10 | The room is dark. | The room is large. | The room is dim. | The chamber is unlit. | The chamber is square. |
| 11 | The bread is stale. | The bread is sliced. | The bread is dry. | The loaf is dried out. | The loaf is cut in half. |
| 12 | The singer is famous. | The singer is hungry. | The singer is renowned. | The vocalist is celebrated. | The vocalist is seated. |
| 13 | The bridge is unsafe. | The bridge is old. | The bridge is dangerous. | The crossing is hazardous. | The crossing is wooden. |
| 14 | The river is wide. | The river is cold. | The river is broad. | The waterway is expansive. | The waterway is muddy. |
| 15 | The teacher is angry. | The teacher is tall. | The teacher is furious. | The instructor is cross. | The instructor is bald. |
| 16 | The meal is ready. | The meal is spicy. | The meal is prepared. | Dinner is done. | Dinner is late. |
| 17 | The hill is steep. | The hill is green. | The hill is sheer. | The slope is precipitous. | The slope is grassy. |
| 18 | The box is sealed. | The box is heavy. | The box is shut. | The crate is taped closed. | The crate is painted green. |
| 19 | The lawyer is honest. | The lawyer is busy. | The lawyer is truthful. | The attorney is trustworthy. | The attorney is wealthy. |
| 20 | The engine is noisy. | The engine is small. | The engine is loud. | The motor is roaring. | The motor is rusty. |
