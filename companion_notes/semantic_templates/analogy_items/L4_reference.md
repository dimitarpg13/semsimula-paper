# L4 reference

**Draft items for review (2026-11-26).** Level of [Analogy_and_Context_Test_Set.md](../Analogy_and_Context_Test_Set.md). Source: `notebooks/conservative_arch/scaleup/debug/template_analogy_clauses.py`. Back to the [index](README.md).

**What this level tests.** The second sentence names its subject by a pronoun only. Each frame pairs a man and a woman, so "he" or "she" says which clause is meant. The foil is the same sentence with the other pronoun: it says P's predicate about Q's subject ("He is ill." restates "the king is ill"; "She is ill." says the queen is ill). Restatement and foil differ in one token, which neither clause contains. Every predicate is plausible for either person, so a foil cannot be rejected as nonsense, and P is "he" in half the frames. The cue is the gender of the subject word, which is knowledge the model may or may not link to the pronoun.

**What to check:**
- Is each subject unambiguously male or female?
- Is every predicate plausible for both people, so that no foil reads as nonsense?
- Is each foil a different statement (about the wrong person), not a restatement?

**Matched-overlap check:** 60 of 60 restatement and foil pairs share the same number of tokens with the clause.

| # | clause P | clause Q | restatement of P | foil (Q's pronoun) | restatement of Q (AC twin) |
| --- | --- | --- | --- | --- | --- |
| 1 | The king is ill. | The queen is in charge. | He is ill. | She is ill. | She is in charge. |
| 2 | The waitress is late. | The landlord is busy. | She is late. | He is late. | He is busy. |
| 3 | The girl is frightened. | The old man is cheerful. | She is frightened. | He is frightened. | He is cheerful. |
| 4 | The uncle is asleep. | The aunt is cooking. | He is asleep. | She is asleep. | She is cooking. |
| 5 | The widow is worried. | The priest is hopeful. | She is worried. | He is worried. | He is hopeful. |
| 6 | The boy is lost. | The mother is searching. | He is lost. | She is lost. | She is searching. |
| 7 | The grandmother is pleased. | The grandson is shy. | She is pleased. | He is pleased. | He is shy. |
| 8 | The nun is silent. | The bishop is speaking. | She is silent. | He is silent. | He is speaking. |
| 9 | The husband is away. | The wife is anxious. | He is away. | She is away. | She is anxious. |
| 10 | The bride is nervous. | The groom is calm. | She is nervous. | He is nervous. | He is calm. |
| 11 | Anna is tired. | Tom is driving. | She is tired. | He is tired. | He is driving. |
| 12 | The prince is abroad. | The princess is unhappy. | He is abroad. | She is abroad. | She is unhappy. |
| 13 | The landlady is upset. | The postman is early. | She is upset. | He is upset. | He is early. |
| 14 | The father is proud. | The daughter is graduating. | He is proud. | She is proud. | She is graduating. |
| 15 | The schoolboy is crying. | The headmistress is stern. | He is crying. | She is crying. | She is stern. |
| 16 | The duke is bankrupt. | The duchess is generous. | He is bankrupt. | She is bankrupt. | She is generous. |
| 17 | The grandfather is asleep. | The granddaughter is bored. | He is asleep. | She is asleep. | She is bored. |
| 18 | Maria is angry. | Peter is apologising. | She is angry. | He is angry. | He is apologising. |
| 19 | The policeman is suspicious. | The woman is innocent. | He is suspicious. | She is suspicious. | She is innocent. |
| 20 | The actress is impatient. | The cameraman is tired. | She is impatient. | He is impatient. | He is tired. |
