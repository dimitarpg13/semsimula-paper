# Task B-logic: Draft Items

**What this folder is.** The draft of task B-logic, for the author's review (2026-11-27): a logical form set by context. The source of every sentence is `notebooks/conservative_arch/scaleup/debug/template_blogic_clauses.py`; this page is generated from it by `notebooks/conservative_arch/scaleup/debug/template_blogic_review.py`, so corrections go into the source and the page is regenerated. Nothing has been extracted or scored. Background: task B in [Summary_State_Test.md](../Summary_State_Test.md) and the discussion that led to combining it with the logical forms.

**The design.** Each frame has two clauses P and Q, a signal S and an unrelated clause U. Every premise of a frame has S as its second sentence, so within a frame the premise is the same string; only a context sentence before it differs, saying what S means. The context makes the premise a valid form, its fallacy twin, or a foil:

| S means | conditional ("If P, then Q. S.") | disjunctive ("Either P or Q. S.") |
| --- | --- | --- |
| P | MP, valid | the affirmed P, twin of DS1 |
| Q | AC, twin of MP | the affirmed Q, twin of DS2 |
| not Q | MT, valid | DS2, valid |
| not P | DA, twin of MT | DS1, valid |
| U | foil: decides nothing | foil: decides nothing |

**Why the words cannot help.** Within a frame the matcher's particles (connective, clauses, second sentence, "Therefore") are identical in every version, and the context's tokens are not particles. The extraction will also fix the position at which the premise starts (the ordinary text before the context is shortened by the context's length), so that the position vectors are identical too. The embedding and token identity then give identical readings for MP, AC and the foil of a frame: they tie at exactly 0.5 by construction. Only states that took in the context can separate them.

**Two flavours of context.** *Explicit*: the meaning of the signal is stated (`Here, "the blue flag is up" means that the bank is closed.`). *Natural*: a regularity in the world is stated (`The blue flag is up only when the bank is closed.`). Both are generated from the frame by one template each, so only the frame needs review. The signal is arbitrary (a flag, a lamp, a bell) so that it favours no clause on its own.

**Example, frame 1, natural flavour, MP:** `The blue flag is up only when the bank is closed.` / `If the bank is closed, then the high street is quiet. The blue flag is up. Therefore`

**What to check, per frame.**

- Does "If P, then Q" read as a sensible conditional, and "Either P or Q" as a sensible disjunction?
- Is the signal free of any meaning that would link it to P or Q without the context?
- Is U unrelated to P, Q and the signal?
- Do the ten context sentences read naturally, including the denied ones ("only when the bank is not closed")?

**Automatic checks.** 20 frames; 0 problems found.

Context sentences run from 11 to 20 GPT-2 tokens.

**Size.** 20 frames × 2 frames of premise × 5 meanings × 3 wordings × 2 flavours = 1,200 premises per model.

**The author's corrections.**

| date | item | change |
| --- | --- | --- |
| 2026-11-27 | all frames | reviewed by the author: "makes sense"; no change. |

**The control** (added 2026-11-27, BLogic_Test.md §5.4): contexts that name both clauses, in two orders; all of them on [control.md](control.md). Example, frame 1, MP, variant a: `Here, "the blue flag is up" means that the bank is closed, and nothing about whether the high street is quiet.`

## The frames

| # | P | Q | signal S | unrelated U |
| --- | --- | --- | --- | --- |
| 1 | the bank is closed | the high street is quiet | the blue flag is up | the cleaners are inside |
| 2 | the road is icy | the buses are late | the paper lantern is lit | the radio is on |
| 3 | the river is high | the ferry is cancelled | the yellow kite is flying | the fair is in town |
| 4 | the oven is hot | the bread is ready | the blue dot is showing | the kitchen is tidy |
| 5 | the match is on | the pub is full | the tower bell is ringing | the car park is new |
| 6 | the baby is asleep | the house is quiet | the pink ribbon is tied | the curtains are blue |
| 7 | the museum is open | the guides are busy | the round sign is turned | the roof is old |
| 8 | the boiler is on | the radiators are warm | the chequered flag is up | the pipes are copper |
| 9 | the teacher is ill | the lesson is cancelled | the white card is showing | the window is open |
| 10 | the harbour is calm | the boats are out | the white pennant is flying | the gulls are noisy |
| 11 | the printer is jammed | the report is late | the brass bell is ringing | the paper is recycled |
| 12 | the goalkeeper is injured | the coach is worried | the purple flag is raised | the referee is tall |
| 13 | the library is shut | the students are outside | the chalk mark is green | the books are heavy |
| 14 | the soup is ready | the guests are seated | the silver ribbon is up | the napkins are white |
| 15 | the lift is broken | the stairs are crowded | the green ribbon is up | the lobby is bright |
| 16 | the pool is closed | the children are bored | the striped flag is up | the lifeguard is young |
| 17 | the train is delayed | the platform is packed | the orange kite is flying | the station is old |
| 18 | the market is busy | the prices are high | the grey flag is up | the fountain is dry |
| 19 | the volcano is active | the village is empty | the copper bell is ringing | the hills are green |
| 20 | the concert is sold out | the touts are busy | the poster is red | the hall is warm |

## The context sentences

### Frame 1

Premises: "If the bank is closed, then the high street is quiet. The blue flag is up." and "Either the bank is closed or the high street is quiet. The blue flag is up."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the blue flag is up" means that the bank is closed. | The blue flag is up only when the bank is closed. |
| Q | AC | Qaff | Here, "the blue flag is up" means that the high street is quiet. | The blue flag is up only when the high street is quiet. |
| not P | DA | DS1 | Here, "the blue flag is up" means that the bank is not closed. | The blue flag is up only when the bank is not closed. |
| not Q | MT | DS2 | Here, "the blue flag is up" means that the high street is not quiet. | The blue flag is up only when the high street is not quiet. |
| U | foil | foil | Here, "the blue flag is up" means that the cleaners are inside. | The blue flag is up only when the cleaners are inside. |

### Frame 2

Premises: "If the road is icy, then the buses are late. The paper lantern is lit." and "Either the road is icy or the buses are late. The paper lantern is lit."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the paper lantern is lit" means that the road is icy. | The paper lantern is lit only when the road is icy. |
| Q | AC | Qaff | Here, "the paper lantern is lit" means that the buses are late. | The paper lantern is lit only when the buses are late. |
| not P | DA | DS1 | Here, "the paper lantern is lit" means that the road is not icy. | The paper lantern is lit only when the road is not icy. |
| not Q | MT | DS2 | Here, "the paper lantern is lit" means that the buses are not late. | The paper lantern is lit only when the buses are not late. |
| U | foil | foil | Here, "the paper lantern is lit" means that the radio is on. | The paper lantern is lit only when the radio is on. |

### Frame 3

Premises: "If the river is high, then the ferry is cancelled. The yellow kite is flying." and "Either the river is high or the ferry is cancelled. The yellow kite is flying."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the yellow kite is flying" means that the river is high. | The yellow kite is flying only when the river is high. |
| Q | AC | Qaff | Here, "the yellow kite is flying" means that the ferry is cancelled. | The yellow kite is flying only when the ferry is cancelled. |
| not P | DA | DS1 | Here, "the yellow kite is flying" means that the river is not high. | The yellow kite is flying only when the river is not high. |
| not Q | MT | DS2 | Here, "the yellow kite is flying" means that the ferry is not cancelled. | The yellow kite is flying only when the ferry is not cancelled. |
| U | foil | foil | Here, "the yellow kite is flying" means that the fair is in town. | The yellow kite is flying only when the fair is in town. |

### Frame 4

Premises: "If the oven is hot, then the bread is ready. The blue dot is showing." and "Either the oven is hot or the bread is ready. The blue dot is showing."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the blue dot is showing" means that the oven is hot. | The blue dot is showing only when the oven is hot. |
| Q | AC | Qaff | Here, "the blue dot is showing" means that the bread is ready. | The blue dot is showing only when the bread is ready. |
| not P | DA | DS1 | Here, "the blue dot is showing" means that the oven is not hot. | The blue dot is showing only when the oven is not hot. |
| not Q | MT | DS2 | Here, "the blue dot is showing" means that the bread is not ready. | The blue dot is showing only when the bread is not ready. |
| U | foil | foil | Here, "the blue dot is showing" means that the kitchen is tidy. | The blue dot is showing only when the kitchen is tidy. |

### Frame 5

Premises: "If the match is on, then the pub is full. The tower bell is ringing." and "Either the match is on or the pub is full. The tower bell is ringing."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the tower bell is ringing" means that the match is on. | The tower bell is ringing only when the match is on. |
| Q | AC | Qaff | Here, "the tower bell is ringing" means that the pub is full. | The tower bell is ringing only when the pub is full. |
| not P | DA | DS1 | Here, "the tower bell is ringing" means that the match is not on. | The tower bell is ringing only when the match is not on. |
| not Q | MT | DS2 | Here, "the tower bell is ringing" means that the pub is not full. | The tower bell is ringing only when the pub is not full. |
| U | foil | foil | Here, "the tower bell is ringing" means that the car park is new. | The tower bell is ringing only when the car park is new. |

### Frame 6

Premises: "If the baby is asleep, then the house is quiet. The pink ribbon is tied." and "Either the baby is asleep or the house is quiet. The pink ribbon is tied."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the pink ribbon is tied" means that the baby is asleep. | The pink ribbon is tied only when the baby is asleep. |
| Q | AC | Qaff | Here, "the pink ribbon is tied" means that the house is quiet. | The pink ribbon is tied only when the house is quiet. |
| not P | DA | DS1 | Here, "the pink ribbon is tied" means that the baby is not asleep. | The pink ribbon is tied only when the baby is not asleep. |
| not Q | MT | DS2 | Here, "the pink ribbon is tied" means that the house is not quiet. | The pink ribbon is tied only when the house is not quiet. |
| U | foil | foil | Here, "the pink ribbon is tied" means that the curtains are blue. | The pink ribbon is tied only when the curtains are blue. |

### Frame 7

Premises: "If the museum is open, then the guides are busy. The round sign is turned." and "Either the museum is open or the guides are busy. The round sign is turned."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the round sign is turned" means that the museum is open. | The round sign is turned only when the museum is open. |
| Q | AC | Qaff | Here, "the round sign is turned" means that the guides are busy. | The round sign is turned only when the guides are busy. |
| not P | DA | DS1 | Here, "the round sign is turned" means that the museum is not open. | The round sign is turned only when the museum is not open. |
| not Q | MT | DS2 | Here, "the round sign is turned" means that the guides are not busy. | The round sign is turned only when the guides are not busy. |
| U | foil | foil | Here, "the round sign is turned" means that the roof is old. | The round sign is turned only when the roof is old. |

### Frame 8

Premises: "If the boiler is on, then the radiators are warm. The chequered flag is up." and "Either the boiler is on or the radiators are warm. The chequered flag is up."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the chequered flag is up" means that the boiler is on. | The chequered flag is up only when the boiler is on. |
| Q | AC | Qaff | Here, "the chequered flag is up" means that the radiators are warm. | The chequered flag is up only when the radiators are warm. |
| not P | DA | DS1 | Here, "the chequered flag is up" means that the boiler is not on. | The chequered flag is up only when the boiler is not on. |
| not Q | MT | DS2 | Here, "the chequered flag is up" means that the radiators are not warm. | The chequered flag is up only when the radiators are not warm. |
| U | foil | foil | Here, "the chequered flag is up" means that the pipes are copper. | The chequered flag is up only when the pipes are copper. |

### Frame 9

Premises: "If the teacher is ill, then the lesson is cancelled. The white card is showing." and "Either the teacher is ill or the lesson is cancelled. The white card is showing."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the white card is showing" means that the teacher is ill. | The white card is showing only when the teacher is ill. |
| Q | AC | Qaff | Here, "the white card is showing" means that the lesson is cancelled. | The white card is showing only when the lesson is cancelled. |
| not P | DA | DS1 | Here, "the white card is showing" means that the teacher is not ill. | The white card is showing only when the teacher is not ill. |
| not Q | MT | DS2 | Here, "the white card is showing" means that the lesson is not cancelled. | The white card is showing only when the lesson is not cancelled. |
| U | foil | foil | Here, "the white card is showing" means that the window is open. | The white card is showing only when the window is open. |

### Frame 10

Premises: "If the harbour is calm, then the boats are out. The white pennant is flying." and "Either the harbour is calm or the boats are out. The white pennant is flying."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the white pennant is flying" means that the harbour is calm. | The white pennant is flying only when the harbour is calm. |
| Q | AC | Qaff | Here, "the white pennant is flying" means that the boats are out. | The white pennant is flying only when the boats are out. |
| not P | DA | DS1 | Here, "the white pennant is flying" means that the harbour is not calm. | The white pennant is flying only when the harbour is not calm. |
| not Q | MT | DS2 | Here, "the white pennant is flying" means that the boats are not out. | The white pennant is flying only when the boats are not out. |
| U | foil | foil | Here, "the white pennant is flying" means that the gulls are noisy. | The white pennant is flying only when the gulls are noisy. |

### Frame 11

Premises: "If the printer is jammed, then the report is late. The brass bell is ringing." and "Either the printer is jammed or the report is late. The brass bell is ringing."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the brass bell is ringing" means that the printer is jammed. | The brass bell is ringing only when the printer is jammed. |
| Q | AC | Qaff | Here, "the brass bell is ringing" means that the report is late. | The brass bell is ringing only when the report is late. |
| not P | DA | DS1 | Here, "the brass bell is ringing" means that the printer is not jammed. | The brass bell is ringing only when the printer is not jammed. |
| not Q | MT | DS2 | Here, "the brass bell is ringing" means that the report is not late. | The brass bell is ringing only when the report is not late. |
| U | foil | foil | Here, "the brass bell is ringing" means that the paper is recycled. | The brass bell is ringing only when the paper is recycled. |

### Frame 12

Premises: "If the goalkeeper is injured, then the coach is worried. The purple flag is raised." and "Either the goalkeeper is injured or the coach is worried. The purple flag is raised."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the purple flag is raised" means that the goalkeeper is injured. | The purple flag is raised only when the goalkeeper is injured. |
| Q | AC | Qaff | Here, "the purple flag is raised" means that the coach is worried. | The purple flag is raised only when the coach is worried. |
| not P | DA | DS1 | Here, "the purple flag is raised" means that the goalkeeper is not injured. | The purple flag is raised only when the goalkeeper is not injured. |
| not Q | MT | DS2 | Here, "the purple flag is raised" means that the coach is not worried. | The purple flag is raised only when the coach is not worried. |
| U | foil | foil | Here, "the purple flag is raised" means that the referee is tall. | The purple flag is raised only when the referee is tall. |

### Frame 13

Premises: "If the library is shut, then the students are outside. The chalk mark is green." and "Either the library is shut or the students are outside. The chalk mark is green."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the chalk mark is green" means that the library is shut. | The chalk mark is green only when the library is shut. |
| Q | AC | Qaff | Here, "the chalk mark is green" means that the students are outside. | The chalk mark is green only when the students are outside. |
| not P | DA | DS1 | Here, "the chalk mark is green" means that the library is not shut. | The chalk mark is green only when the library is not shut. |
| not Q | MT | DS2 | Here, "the chalk mark is green" means that the students are not outside. | The chalk mark is green only when the students are not outside. |
| U | foil | foil | Here, "the chalk mark is green" means that the books are heavy. | The chalk mark is green only when the books are heavy. |

### Frame 14

Premises: "If the soup is ready, then the guests are seated. The silver ribbon is up." and "Either the soup is ready or the guests are seated. The silver ribbon is up."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the silver ribbon is up" means that the soup is ready. | The silver ribbon is up only when the soup is ready. |
| Q | AC | Qaff | Here, "the silver ribbon is up" means that the guests are seated. | The silver ribbon is up only when the guests are seated. |
| not P | DA | DS1 | Here, "the silver ribbon is up" means that the soup is not ready. | The silver ribbon is up only when the soup is not ready. |
| not Q | MT | DS2 | Here, "the silver ribbon is up" means that the guests are not seated. | The silver ribbon is up only when the guests are not seated. |
| U | foil | foil | Here, "the silver ribbon is up" means that the napkins are white. | The silver ribbon is up only when the napkins are white. |

### Frame 15

Premises: "If the lift is broken, then the stairs are crowded. The green ribbon is up." and "Either the lift is broken or the stairs are crowded. The green ribbon is up."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the green ribbon is up" means that the lift is broken. | The green ribbon is up only when the lift is broken. |
| Q | AC | Qaff | Here, "the green ribbon is up" means that the stairs are crowded. | The green ribbon is up only when the stairs are crowded. |
| not P | DA | DS1 | Here, "the green ribbon is up" means that the lift is not broken. | The green ribbon is up only when the lift is not broken. |
| not Q | MT | DS2 | Here, "the green ribbon is up" means that the stairs are not crowded. | The green ribbon is up only when the stairs are not crowded. |
| U | foil | foil | Here, "the green ribbon is up" means that the lobby is bright. | The green ribbon is up only when the lobby is bright. |

### Frame 16

Premises: "If the pool is closed, then the children are bored. The striped flag is up." and "Either the pool is closed or the children are bored. The striped flag is up."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the striped flag is up" means that the pool is closed. | The striped flag is up only when the pool is closed. |
| Q | AC | Qaff | Here, "the striped flag is up" means that the children are bored. | The striped flag is up only when the children are bored. |
| not P | DA | DS1 | Here, "the striped flag is up" means that the pool is not closed. | The striped flag is up only when the pool is not closed. |
| not Q | MT | DS2 | Here, "the striped flag is up" means that the children are not bored. | The striped flag is up only when the children are not bored. |
| U | foil | foil | Here, "the striped flag is up" means that the lifeguard is young. | The striped flag is up only when the lifeguard is young. |

### Frame 17

Premises: "If the train is delayed, then the platform is packed. The orange kite is flying." and "Either the train is delayed or the platform is packed. The orange kite is flying."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the orange kite is flying" means that the train is delayed. | The orange kite is flying only when the train is delayed. |
| Q | AC | Qaff | Here, "the orange kite is flying" means that the platform is packed. | The orange kite is flying only when the platform is packed. |
| not P | DA | DS1 | Here, "the orange kite is flying" means that the train is not delayed. | The orange kite is flying only when the train is not delayed. |
| not Q | MT | DS2 | Here, "the orange kite is flying" means that the platform is not packed. | The orange kite is flying only when the platform is not packed. |
| U | foil | foil | Here, "the orange kite is flying" means that the station is old. | The orange kite is flying only when the station is old. |

### Frame 18

Premises: "If the market is busy, then the prices are high. The grey flag is up." and "Either the market is busy or the prices are high. The grey flag is up."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the grey flag is up" means that the market is busy. | The grey flag is up only when the market is busy. |
| Q | AC | Qaff | Here, "the grey flag is up" means that the prices are high. | The grey flag is up only when the prices are high. |
| not P | DA | DS1 | Here, "the grey flag is up" means that the market is not busy. | The grey flag is up only when the market is not busy. |
| not Q | MT | DS2 | Here, "the grey flag is up" means that the prices are not high. | The grey flag is up only when the prices are not high. |
| U | foil | foil | Here, "the grey flag is up" means that the fountain is dry. | The grey flag is up only when the fountain is dry. |

### Frame 19

Premises: "If the volcano is active, then the village is empty. The copper bell is ringing." and "Either the volcano is active or the village is empty. The copper bell is ringing."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the copper bell is ringing" means that the volcano is active. | The copper bell is ringing only when the volcano is active. |
| Q | AC | Qaff | Here, "the copper bell is ringing" means that the village is empty. | The copper bell is ringing only when the village is empty. |
| not P | DA | DS1 | Here, "the copper bell is ringing" means that the volcano is not active. | The copper bell is ringing only when the volcano is not active. |
| not Q | MT | DS2 | Here, "the copper bell is ringing" means that the village is not empty. | The copper bell is ringing only when the village is not empty. |
| U | foil | foil | Here, "the copper bell is ringing" means that the hills are green. | The copper bell is ringing only when the hills are green. |

### Frame 20

Premises: "If the concert is sold out, then the touts are busy. The poster is red." and "Either the concert is sold out or the touts are busy. The poster is red."

| S means | conditional | disjunctive | explicit | natural |
| --- | --- | --- | --- | --- |
| P | MP | Paff | Here, "the poster is red" means that the concert is sold out. | The poster is red only when the concert is sold out. |
| Q | AC | Qaff | Here, "the poster is red" means that the touts are busy. | The poster is red only when the touts are busy. |
| not P | DA | DS1 | Here, "the poster is red" means that the concert is not sold out. | The poster is red only when the concert is not sold out. |
| not Q | MT | DS2 | Here, "the poster is red" means that the touts are not busy. | The poster is red only when the touts are not busy. |
| U | foil | foil | Here, "the poster is red" means that the hall is warm. | The poster is red only when the hall is warm. |
