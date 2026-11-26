# C2 same word, sense differs; C2′ paraphrase against a false friend

**Draft items for review (2026-11-26).** Level of [Analogy_and_Context_Test_Set.md](../Analogy_and_Context_Test_Set.md). Source: `notebooks/conservative_arch/scaleup/debug/template_analogy_clauses.py`. Back to the [index](README.md).

**What this level tests.** The clause uses a word with two senses; its consequent fixes the sense. The restatement adds a word of that sense, the foil a word of the other sense, the same number of tokens: token identity ties. In C2′ the restatement is a paraphrase in the clause's sense that shares no content word with it, against the same foil, which repeats the clause's words (a false friend): token identity prefers the foil.

**What to check:**
- Does the consequent fix the sense of the clause?
- Does each restatement keep that sense, and each foil clearly take the other?
- Is each C2′ paraphrase the same statement in the clause's sense?

**Matched-overlap check:** 40 of 40 restatement and foil pairs share the same number of tokens with the clause.

C2′ by design: the false friend shares 5, 6, 7, 6, 4 ... tokens with the clause, the paraphrase 2, 4, 2, 2, 1 ... (first five items).

| # | clause P | consequent Q (fixes the sense) | restatement (same sense) | foil (other sense) | C2′ paraphrase |
| --- | --- | --- | --- | --- | --- |
| 1 | The bank is closed. | We cannot withdraw cash. | The bank on the high street is closed. | The bank of the river is closed. | The branch on the high street has shut. |
| 2 | The bat is in the shed. | The game cannot start. | The cricket bat is in the shed. | The fruit bat is in the shed. | The willow paddle is stored in the outbuilding. |
| 3 | The crane is standing by the lake. | The birdwatchers gather. | The white crane is standing by the lake. | The steel crane is standing by the lake. | The long-legged wader waits at the water's edge. |
| 4 | The mouse is under the desk. | The cat is excited. | The field mouse is under the desk. | The computer mouse is under the desk. | The little rodent hides beneath the table. |
| 5 | The letter is missing. | The post office apologises. | The posted letter is missing. | The capital letter is missing. | The mailed envelope has gone astray. |
| 6 | The plant is dying. | We water it more often. | The potted plant is dying. | The steel plant is dying. | The house fern is wilting. |
| 7 | The court is empty. | The trial cannot go ahead. | The criminal court is empty. | The tennis court is empty. | The hall of justice has nobody in it. |
| 8 | The organ is damaged. | The surgeon cannot transplant it. | The donor organ is damaged. | The church organ is damaged. | The harvested kidney is injured. |
| 9 | The club is new. | The golfer practises more. | The golf club is new. | The youth club is new. | The iron he swings has just been bought. |
| 10 | The ring is too small. | The bride cannot wear it. | The wedding ring is too small. | The boxing ring is too small. | The gold band will not fit her finger. |
| 11 | The nail is bent. | The carpenter pulls it out. | The steel nail is bent. | The toe nail is bent. | The iron spike is crooked. |
| 12 | The palm is dry. | The gardener waters it. | The date palm is dry. | The left palm is dry. | The date tree is parched. |
| 13 | The glasses are broken. | The reader cannot see the page. | The reading glasses are broken. | The wine glasses are broken. | Her spectacles have cracked. |
| 14 | The key is lost. | We cannot open the front door. | The door key is lost. | The answer key is lost. | The latchkey has gone missing. |
| 15 | The jam is terrible. | We will be late for work. | The traffic jam is terrible. | The strawberry jam is terrible. | The congestion on the motorway is awful. |
| 16 | The fan is broken. | The room is too hot. | The ceiling fan is broken. | The football fan is broken. | The overhead cooler has stopped working. |
| 17 | The deck is wet. | The sailors slip. | The upper deck is wet. | The card deck is wet. | The boat's planking is soaked. |
| 18 | The pupil is small. | The light is too bright. | The eye pupil is small. | The school pupil is small. | The iris opening has narrowed. |
| 19 | The bark is rough. | The climber grips it easily. | The tree bark is rough. | The dog bark is rough. | The oak's skin is coarse. |
| 20 | The record is broken. | The music stops. | The vinyl record is broken. | The world record is broken. | The old LP has cracked. |
