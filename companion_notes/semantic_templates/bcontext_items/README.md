# Task B: Draft Items (meaning set only by context)

**What this folder is.** The draft of task B, for the author's review (2026-11-27): the restatement and its rival are the same string, and only a context sentence before the premise says what it means. The source of every sentence is `notebooks/conservative_arch/scaleup/debug/template_bcontext_clauses.py`; this page is generated from it by `notebooks/conservative_arch/scaleup/debug/template_bcontext_review.py`, so corrections go into the source and the page is regenerated. Nothing has been extracted or scored. Design and draft calls: [Task_B_Meaning_Set_by_Context.md](../Task_B_Meaning_Set_by_Context.md).

**The design.** Two sublevels, each built on the usual premise ("If P, then Q. S." or "Either P or Q. S."):

- **Word sense.** The second sentence S contains an ambiguous word ("The bank is closed."). Clause P says the same of one sense in other words ("the lender is closed"), clause Q of the other sense ("the embankment is closed"). The context picks the sense.
- **Pronoun.** S has a pronoun ("He is tired."). P and Q say it of two people of the same gender ("John is tired", "Bill is tired"). The context picks the referent; a third person gives a foil.

What the context makes of the premise:

| the second sentence | conditional frame | disjunctive frame |
| --- | --- | --- |
| affirmed, meaning P | MP, valid | (not used) |
| affirmed, meaning Q | AC, twin of MP | (not used) |
| denied, meaning Q | MT, valid | DS2, valid |
| denied, meaning P | DA, twin of MT | DS1, valid |
| about the third person (pronoun only) | foil | foil |

**Why the words cannot help.** Within a frame, wording and polarity, the matcher's particles (connective, clauses, second sentence, "Therefore") are the same tokens at the same positions in every version, and the context's tokens are not particles. Token identity and the embedding therefore tie at exactly 0.5 on every pair, by construction.

**Four flavours of context per sublevel.**

| flavour | word sense (example) | pronoun (example) | what it names |
| --- | --- | --- | --- |
| explicit | `Here, "the bank" means the lender.` | `Here, "he" means John.` | the selected meaning only |
| natural | `We were talking about money and loans.` | `John was the only one in the room.` | word sense: no clause word; pronoun: the selected person |
| control a | `Here, "the bank" means the lender, not the embankment.` | `John came into the room while Bill stayed outside.` | both meanings, the bound one first |
| control b | `Here, "the bank" is not the embankment; it means the lender.` | `While Bill stayed outside, John came into the room.` | both meanings, the bound one last |

The controls separate binding from mention, as in B-logic: a state that only marks which meaning was named, or which was named last, cannot separate versions that name both.

## Checks

- Every natural word-sense context shares no content word with its clauses; the two senses share no word; every clause can be denied; no name is reused across pronoun frames.
- Context lengths: word sense 5 to 21 tokens, pronoun 8 to 14 tokens (the filler before them makes up the difference).

## Word-sense frames

| frame | second sentence | clause P (sense 1) | clause Q (sense 2) | natural context, sense 1 | natural context, sense 2 |
| ---: | --- | --- | --- | --- | --- |
| 1 | The bank is closed. | the lender is closed | the embankment is closed | We were talking about money and loans. | We were walking by the water. |
| 2 | The bat is in the shed. | the baseball club is in the shed | the flying mammal is in the shed | We were getting ready for the game. | We were watching animals at dusk. |
| 3 | The pitcher is in the kitchen. | the water jug is in the kitchen | the baseball thrower is in the kitchen | We were setting the table for lunch. | We were watching the team practise. |
| 4 | The crane is by the lake. | the lifting machine is by the lake | the tall wading bird is by the lake | We were watching the builders at work. | We were looking at the wildlife in the marsh. |
| 5 | The mouse is under the desk. | the computer pointer is under the desk | the small rodent is under the desk | We were fixing the laptop. | We were trying to catch the pest. |
| 6 | The seal is grey. | the sea animal is grey | the wax stamp is grey | We were watching wildlife at the coast. | We were reading an old royal letter. |
| 7 | The glasses are on the shelf. | the spectacles are on the shelf | the drinking cups are on the shelf | She could not read the menu. | We were washing up after the party. |
| 8 | The plant is near the road. | the factory is near the road | the green shrub is near the road | The workers were on strike. | We were watering the garden. |
| 9 | The deck is wet. | the ship floor is wet | the pack of cards is wet | The sailors were scrubbing all morning. | We were playing poker. |
| 10 | The spring is hidden. | the coiled wire is hidden | the natural fountain is hidden | We were repairing the old mattress. | We were hiking up the hill, thirsty. |
| 11 | The pupil is small. | the student is small | the dark centre of the iris is small | We were visiting the school. | The doctor shone a light in her face. |
| 12 | The letter is on the page. | the written message is on the page | the alphabet symbol is on the page | We were waiting for the post. | We were learning to spell. |
| 13 | The organ is large. | the musical instrument is large | the body part is large | We were at the church concert. | We were studying anatomy. |
| 14 | The trunk is heavy. | the elephant nose is heavy | the storage chest is heavy | We were at the zoo. | We were packing for the long voyage. |
| 15 | The fan is in the hall. | the cooling device is in the hall | the sports supporter is in the hall | It was a very hot day. | The team had just won the cup. |
| 16 | The note is short. | the written reminder is short | the musical tone is short | She left a message on the fridge. | The violinist was tuning up. |
| 17 | The board is new. | the wooden plank is new | the governing committee is new | We were building a shed. | The shareholders met on monday. |
| 18 | The jam is bad today. | the fruit spread is bad today | the traffic queue is bad today | We were making breakfast. | We were driving to the airport. |
| 19 | The tank is in the yard. | the water container is in the yard | the armoured vehicle is in the yard | We were fixing the garden irrigation. | The soldiers were on parade. |
| 20 | The match is short. | the small fire stick is short | the sports game is short | We were lighting the candles. | We were watching the football. |

The explicit and control contexts follow the patterns of the flavour table with the frame's phrase and senses.

## Pronoun frames

| frame | second sentence | clause P | clause Q | third person (foil) |
| ---: | --- | --- | --- | --- |
| 1 | He is tired. | John is tired | Bill is tired | Tom |
| 2 | She is hungry. | Mary is hungry | Anne is hungry | Kate |
| 3 | He is late. | Peter is late | Mark is late | Sam |
| 4 | She is at home. | Lucy is at home | Emma is at home | Rose |
| 5 | He is angry. | David is angry | Paul is angry | Jack |
| 6 | She is ill. | Sarah is ill | Helen is ill | Jane |
| 7 | He is asleep. | James is asleep | Henry is asleep | Luke |
| 8 | She is happy. | Alice is happy | Grace is happy | Ruth |
| 9 | He is outside. | Robert is outside | Frank is outside | Adam |
| 10 | She is busy. | Laura is busy | Clare is busy | Beth |
| 11 | He is cold. | George is cold | Simon is cold | Ben |
| 12 | She is upstairs. | Emily is upstairs | Megan is upstairs | Holly |
| 13 | He is lost. | Michael is lost | Steven is lost | Ryan |
| 14 | She is worried. | Julia is worried | Fiona is worried | Amy |
| 15 | He is ready. | Daniel is ready | Oliver is ready | Ian |
| 16 | She is awake. | Hannah is awake | Chloe is awake | Ella |
| 17 | He is wet. | Thomas is wet | Edward is wet | Carl |
| 18 | She is bored. | Sophie is bored | Molly is bored | Zoe |
| 19 | He is early. | Andrew is early | Philip is early | Neil |
| 20 | She is sad. | Rachel is sad | Diana is sad | Eve |

The contexts follow the patterns of the flavour table with the frame's names; the foil contexts name the third person (control: "Tom came into the room while John and Bill stayed outside").

## Review log

- Not yet reviewed.
