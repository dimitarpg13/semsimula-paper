# C4 denial by an incompatible value

**Draft items for review (2026-11-26).** Level of [Analogy_and_Context_Test_Set.md](../Analogy_and_Context_Test_Set.md). Source: `notebooks/conservative_arch/scaleup/debug/template_analogy_clauses.py`. Back to the [index](README.md).

**What this level tests.** The second sentence denies a clause only through what is known about it: "the light is green" denies "the light is red". The foil keeps the subject and names a value that is compatible with the clause. The denial of Q gives the MT analogue; the denial of P the DS1 analogue and the DA twin. Morphological antonyms ("armed", "disarmed") are avoided, since they share tokens.

**What to check:**
- Is each incompatible value really incompatible with the clause?
- Is each foil really compatible (it neither denies nor implies the clause)?
- Should some items fix the incompatibility by a context sentence instead of general knowledge?

**Matched-overlap check:** 40 of 40 restatement and foil pairs share the same number of tokens with the clause.

| # | clause P | clause Q | denies Q | foil | denies P | foil |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | The alarm is on. | The light is red. | The light is green. | The light is bright. | The alarm is off. | The alarm is loud. |
| 2 | The shop is open. | The shutters are up. | The shutters are down. | The shutters are metal. | The shop is shut. | The shop is small. |
| 3 | The patient is alive. | His heart is beating. | His heart is still. | His heart is large. | The patient is dead. | The patient is old. |
| 4 | The glass is full. | The jug is empty. | The jug is brimming. | The jug is ceramic. | The glass is drained. | The glass is cold. |
| 5 | The meeting is on Monday. | The room is booked. | The room is free. | The room is large. | The meeting is on Friday. | The meeting is on schedule. |
| 6 | The signal is red. | The cars stop. | The cars keep moving. | The cars keep honking. | The signal is green. | The signal is bright. |
| 7 | The child is awake. | The nursery is noisy. | The nursery is silent. | The nursery is blue. | The child is sleeping. | The child is small. |
| 8 | The coin landed heads. | Anna wins the bet. | Anna loses the bet. | Anna doubles the bet. | The coin landed tails. | The coin landed quickly. |
| 9 | The oven is hot. | The bread is baking. | The bread is raw. | The bread is wholemeal. | The oven is cold. | The oven is new. |
| 10 | The student passed. | Her parents are proud. | Her parents are ashamed. | Her parents are tall. | The student failed. | The student studied. |
| 11 | The door is closed. | The cat is inside. | The cat is outside. | The cat is ginger. | The door is ajar. | The door is painted. |
| 12 | The tank is full. | The gauge points high. | The gauge points low. | The gauge points steadily. | The tank is dry. | The tank is metal. |
| 13 | The river is high. | The ford is dangerous. | The ford is safe. | The ford is wide. | The river is low. | The river is brown. |
| 14 | The road is dry. | The race goes ahead. | The race is called off. | The race is televised. | The road is flooded. | The road is long. |
| 15 | The flag is up. | The castle is occupied. | The castle is deserted. | The castle is ancient. | The flag is down. | The flag is blue. |
| 16 | The tide is in. | The beach is covered. | The beach is exposed. | The beach is sandy. | The tide is out. | The tide is strong. |
| 17 | The fridge is working. | The milk is cold. | The milk is warm. | The milk is fresh. | The fridge is broken. | The fridge is white. |
| 18 | The train is early. | The platform is crowded. | The platform is deserted. | The platform is long. | The train is late. | The train is new. |
| 19 | The soup is hot. | The bowl is warm. | The bowl is icy. | The bowl is round. | The soup is cold. | The soup is thick. |
| 20 | The switch is up. | The lamp is lit. | The lamp is dark. | The lamp is brass. | The switch is down. | The switch is plastic. |
