# L4 reference

**Draft items for review (2026-11-26).** Level of [Analogy_and_Context_Test_Set.md](../Analogy_and_Context_Test_Set.md). Source: `notebooks/conservative_arch/scaleup/debug/template_analogy_clauses.py`. Back to the [index](README.md).

**What this level tests.** The second sentence names its subject by a pronoun only. The two clauses have subjects that take different pronouns, so the pronoun says which clause is meant. The foil is the same sentence with the other clause's pronoun: it says P's predicate about Q's subject. Restatement and foil differ in one token, which neither clause contains.

**What to check:**
- Is the pronoun unambiguous for each pair (he, she, it)?
- Is each foil a different statement (the wrong person or thing), not a restatement?

**Matched-overlap check:** 60 of 60 restatement and foil pairs share the same number of tokens with the clause.

| # | clause P | clause Q | restatement of P | foil (Q's pronoun) | restatement of Q (AC twin) |
| --- | --- | --- | --- | --- | --- |
| 1 | The king is ill. | The queen is in charge. | He is ill. | She is ill. | She is in charge. |
| 2 | The waiter is late. | The actress is hungry. | He is late. | She is late. | She is hungry. |
| 3 | The dog is barking. | The girl is frightened. | It is barking. | She is barking. | She is frightened. |
| 4 | The uncle is asleep. | The aunt is cooking. | He is asleep. | She is asleep. | She is cooking. |
| 5 | The bull is angry. | The farmer's wife is worried. | It is angry. | She is angry. | She is worried. |
| 6 | The boy is lost. | The mother is searching. | He is lost. | She is lost. | She is searching. |
| 7 | The kettle is boiling. | The grandmother is pleased. | It is boiling. | She is boiling. | She is pleased. |
| 8 | The nun is silent. | The bishop is speaking. | She is silent. | He is silent. | He is speaking. |
| 9 | The car is broken. | The mechanic is away. | It is broken. | He is broken. | He is away. |
| 10 | The bride is nervous. | The groom is calm. | She is nervous. | He is nervous. | He is calm. |
| 11 | Anna is tired. | Tom is driving. | She is tired. | He is tired. | He is driving. |
| 12 | The princess is unhappy. | The prince is abroad. | She is unhappy. | He is unhappy. | He is abroad. |
| 13 | The boiler is leaking. | The landlady is upset. | It is leaking. | She is leaking. | She is upset. |
| 14 | The father is proud. | The daughter is graduating. | He is proud. | She is proud. | She is graduating. |
| 15 | The cat is missing. | The boy is crying. | It is missing. | He is missing. | He is crying. |
| 16 | The duchess is generous. | The duke is bankrupt. | She is generous. | He is generous. | He is bankrupt. |
| 17 | The phone is ringing. | The grandfather is asleep. | It is ringing. | He is ringing. | He is asleep. |
| 18 | Maria is angry. | Peter is apologising. | She is angry. | He is angry. | He is apologising. |
| 19 | The policeman is suspicious. | The woman is innocent. | He is suspicious. | She is suspicious. | She is innocent. |
| 20 | The clock is fast. | The headmistress is impatient. | It is fast. | She is fast. | She is impatient. |
