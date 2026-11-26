# C1 alias defined in context

**Draft items for review (2026-11-26).** Level of [Analogy_and_Context_Test_Set.md](../Analogy_and_Context_Test_Set.md). Source: `notebooks/conservative_arch/scaleup/debug/template_analogy_clauses.py`. Back to the [index](README.md).

**What this level tests.** A context sentence before the premise names two people. The clause uses a role ("the baker"), the restatement the person who holds it, the foil the other person, with the same predicate. Only the context links name and role; the names occur nowhere in the premise.

**What to check:**
- Does each context assign the role to exactly one of the two names?
- Is the role in the clause the one the context assigns (no ambiguity)?

**Matched-overlap check:** 40 of 40 restatement and foil pairs share the same number of tokens with the clause.

| # | context | clause P | clause Q | restatement | foil |
| --- | --- | --- | --- | --- | --- |
| 1 | Mr Hale runs the only bakery in town, and Mrs Roe runs the flower shop. | The baker is ill. | The bakery stays shut. | Mr Hale is ill. | Mrs Roe is ill. |
| 2 | Dr Patel is the town's only dentist; Dr Okafor is its vet. | The dentist is away. | The clinic cancels its appointments. | Dr Patel is away. | Dr Okafor is away. |
| 3 | Lena captains the hockey team; Sara coaches it. | The captain is injured. | The team forfeits the match. | Lena is injured. | Sara is injured. |
| 4 | Mr Brook drives the school bus; Mr Finch teaches music. | The driver is sick. | The children walk to school. | Mr Brook is sick. | Mr Finch is sick. |
| 5 | Our neighbour Ruth is the mayor; her brother Ian is the treasurer. | The mayor is abroad. | The council meeting is postponed. | Ruth is abroad. | Ian is abroad. |
| 6 | Mr Moss commands the ship; Mr Reed is its cook. | The commander is drunk. | The voyage is delayed. | Mr Moss is drunk. | Mr Reed is drunk. |
| 7 | Professor Lind chairs the panel; Dr Hart takes the minutes. | The chair is late. | The hearing is delayed. | Professor Lind is late. | Dr Hart is late. |
| 8 | Pedro owns the café on the corner; Marco works at the bank. | The café owner is on holiday. | The café is closed. | Pedro is on holiday. | Marco is on holiday. |
| 9 | The goalkeeper is a woman called Joy; the striker is called Kim. | The goalkeeper is tired. | The defence is weak. | Joy is tired. | Kim is tired. |
| 10 | Mr Grant is the night guard at the museum; Mr Adler is its curator. | The guard is asleep. | The alarm goes unanswered. | Mr Grant is asleep. | Mr Adler is asleep. |
| 11 | Nora manages the hotel; Vera cleans the rooms. | The manager is unavailable. | The complaint is ignored. | Nora is unavailable. | Vera is unavailable. |
| 12 | Ivan flies the morning plane; Oleg sits beside him as co-pilot. | The pilot is unwell. | The flight is cancelled. | Ivan is unwell. | Oleg is unwell. |
| 13 | Mrs Kahn is the head teacher; Miss Doyle teaches art. | The head teacher is absent. | The assembly is cancelled. | Mrs Kahn is absent. | Miss Doyle is absent. |
| 14 | Bruno is the band's drummer; Felix sings. | The drummer is ill. | The concert is called off. | Bruno is ill. | Felix is ill. |
| 15 | Mrs Ward hears the case; Mr Slade defends the accused. | The judge is delayed. | The trial is adjourned. | Mrs Ward is delayed. | Mr Slade is delayed. |
| 16 | Tom is the village blacksmith; Will is the miller. | The blacksmith is busy. | The horses wait unshod. | Tom is busy. | Will is busy. |
| 17 | Amira is the expedition's doctor; Jonas is its guide. | The doctor is lost. | The expedition turns back. | Amira is lost. | Jonas is lost. |
| 18 | Mr Cole keeps the pub; Mr Dunn is one of its regulars. | The landlord is angry. | The pub closes early. | Mr Cole is angry. | Mr Dunn is angry. |
| 19 | Hana conducts the orchestra; Mei plays first violin. | The conductor is late. | The rehearsal is cut short. | Hana is late. | Mei is late. |
| 20 | Mr Price owns the shop; Mr Shah keeps its accounts. | The owner is worried. | The shop is sold. | Mr Price is worried. | Mr Shah is worried. |
