# C3 ellipsis

**Draft items for review (2026-11-26).** Level of [Analogy_and_Context_Test_Set.md](../Analogy_and_Context_Test_Set.md). Source: `notebooks/conservative_arch/scaleup/debug/template_analogy_clauses.py`. Back to the [index](README.md).

**What this level tests.** The second sentence keeps only a pronoun and "does": "He does." The clause's subject and verb must be recovered from the premise. The foil is the other clause's pronoun ("She does."), which affirms Q: the AC twin. Neither shares a token with the clauses. Each frame has one man and one woman, so the pronoun alone picks the clause, and P is "he" in 10 frames and "she" in 10.

**What to check:**
- Is the gender of each person unambiguous, so the pronoun picks one clause?
- Is each verb plausible for either person, so the verb does not tell which one is meant?
- Does "does" read naturally as standing for the clause's verb?

**Matched-overlap check:** 40 of 40 restatement and foil pairs share the same number of tokens with the clause.

| # | clause P | clause Q | restatement | foil (AC twin) | denial |
| --- | --- | --- | --- | --- | --- |
| 1 | The uncle snores. | The aunt complains. | He does. | She does. | He does not. |
| 2 | The grandmother sings. | The grandfather smiles. | She does. | He does. | She does not. |
| 3 | Tom oversleeps. | Anna leaves. | He does. | She does. | He does not. |
| 4 | The princess dances. | The king weeps. | She does. | He does. | She does not. |
| 5 | The father swears. | The daughter blushes. | He does. | She does. | He does not. |
| 6 | The nun prays. | The bishop listens. | She does. | He does. | She does not. |
| 7 | The boy cheats. | The mother worries. | He does. | She does. | He does not. |
| 8 | The landlady shouts. | The postman apologises. | She does. | He does. | She does not. |
| 9 | The bride cries. | The groom laughs. | She does. | He does. | She does not. |
| 10 | The duke gambles. | The duchess moves out. | He does. | She does. | He does not. |
| 11 | Maria wins. | Peter cheers. | She does. | He does. | She does not. |
| 12 | The grandson whistles. | The granddaughter hums. | He does. | She does. | He does not. |
| 13 | The policeman hesitates. | The woman escapes. | He does. | She does. | He does not. |
| 14 | The waitress stumbles. | The landlord frowns. | She does. | He does. | She does not. |
| 15 | The husband travels. | The wife stays home. | He does. | She does. | He does not. |
| 16 | The queen abdicates. | The prince rules. | She does. | He does. | She does not. |
| 17 | The actress improvises. | The cameraman keeps filming. | She does. | He does. | She does not. |
| 18 | The schoolboy misbehaves. | The headmistress intervenes. | He does. | She does. | He does not. |
| 19 | The widow remarries. | The son objects. | She does. | He does. | She does not. |
| 20 | The brother calls. | The sister answers. | He does. | She does. | He does not. |
