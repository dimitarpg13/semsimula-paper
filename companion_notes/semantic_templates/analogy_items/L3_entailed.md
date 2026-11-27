# L3 entailed case

**Draft items for review (2026-11-26).** Level of [Analogy_and_Context_Test_Set.md](../Analogy_and_Context_Test_Set.md). Source: `notebooks/conservative_arch/scaleup/debug/template_analogy_clauses.py`. Back to the [index](README.md).

**What this level tests.** The second sentence states a specific case that implies the clause (the MP analogue), or one that implies its denial (the DS1 and DA analogues), in other words. Each case has a foil that shares as many tokens with the clause and decides nothing about it. Where possible the foil copies its case word for word except the one word that carries the meaning ("the guest has a high fever" against "the guest has a high collar"), so only knowing what that word means separates them. A denial case need only exclude the clause; it need not say the same as the clause's denial ("the team lost every one of its games" excludes "the team won") (author, 2026-11-26).

**What to check:**
- Does each case really imply the clause, for an ordinary reader?
- Does each denial case really exclude the clause (it need not restate the denial)?
- Is each foil neutral: it neither implies the clause nor excludes it?

**Matched-overlap check:** 40 of 40 restatement and foil pairs share the same number of tokens with the clause.

| # | clause | case (implies the clause) | foil of the case (decides nothing) | denial case (implies the clause is false) | foil of the denial case (decides nothing) |
| --- | --- | --- | --- | --- | --- |
| 1 | It rains. | A storm is pouring down on the pitch. | A crowd is pouring down onto the pitch. | The sky is cloudless and the ground is dry. | The stands are very crowded and the ground is level. |
| 2 | The guest is ill. | The guest has a high fever. | The guest has a high collar. | The guest has just finished a marathon. | The guest has just finished a novel. |
| 3 | The animal is a mammal. | The animal is a whale. | The animal is a pet. | The animal is a shark. | The animal is a female. |
| 4 | The metal is precious. | The metal is gold. | The metal is grey. | The metal is iron. | The metal is shiny. |
| 5 | The container holds liquid. | The container holds milk. | The container holds plenty. | The container holds rice. | The container holds enough. |
| 6 | The house is empty. | All the tenants have moved out of the house. | All the tenants have often argued about the house. | All the tenants are at home in the house. | All the tenants are very fond of the house. |
| 7 | The team won. | The team scored three goals and conceded none. | The team scored three goals and travelled home. | The team lost every one of its games. | The team travelled for four hours by coach. |
| 8 | The plant is dead. | The plant has withered to brown stalks. | The plant has been moved out onto a porch. | The plant has burst into bloom. | The plant has been watered daily. |
| 9 | The bird can fly. | The bird is soaring over the valley. | The bird is singing over the valley. | The bird is a penguin. | The bird is in a cage. |
| 10 | The water is hot. | The water is boiling. | The water is murky. | The water has frozen solid. | The water has turned green. |
| 11 | The prisoner escaped. | The prisoner climbed over the wall and fled. | The prisoner climbed a wall in the gym. | The prisoner is still in his cell. | The prisoner is still very quiet today. |
| 12 | The shop is closed. | The shutters of the shop are down and locked. | The shutters of the shop are green and dusty. | Customers are queuing at the shop's tills. | Customers are talking at length about the shop's prices. |
| 13 | The man is married. | The man has a wife. | The man has a wallet. | The man is a bachelor. | The man is a baker. |
| 14 | The film was long. | The film ran for four hours. | The film ran for four weeks. | The film was over in ten minutes. | The film was shown in ten cities. |
| 15 | The number is even. | The number is twelve. | The number is large. | The number is seven. | The number is printed. |
| 16 | The road is wet. | Puddles cover the road. | Potholes cover the road. | Dust is blowing across the road. | Smoke is blowing across the road. |
| 17 | The drawer is locked. | The drawer will not open without the key. | The drawer will not close without the latch. | The drawer slides open at a touch. | The drawer squeaks at a touch. |
| 18 | The soup is salty. | The soup tastes like seawater. | The soup looks like seawater. | The cook added no salt at all to the soup. | The cook added no pepper at all to the soup. |
| 19 | The window is broken. | Shards of the window lie on the floor. | Photos of the window lie on the floor. | The window has just passed its inspection. | The window has just been cleaned again. |
| 20 | The horse is tired. | The horse is panting and stumbling. | The horse is chestnut and white. | The horse is galloping eagerly. | The horse is grazing very quietly. |
