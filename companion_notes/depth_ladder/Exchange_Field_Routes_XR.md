# §5.25 XR: the exchange field against the register path (parts A and B)

**What this note is.** A short, readable summary of §5.25 of the depth-ladder protocol, "XR: the exchange field against the reverse channel", written 2026-11-17. Part A measured the two token-to-token routes of the Gen 3 models on their trained checkpoints (pre-registered 2026-10-17, before any measurement; scored the same day). Part B trained the missing cell, an exchange-field-only model called XRB-F (pre-registered 2026-10-17, binding from the start of the run; scored 2026-10-18). This note gives the pre-registrations, the scoring, and the per-route kernels and capacities. Every number, verdict and date is copied from the [protocol](../Depth_Ladder_and_Matched_Baseline_Protocol.md), which keeps the full running log and remains the record of truth. Part B's results block is [protocol, `p-xrb-f`](../Depth_Ladder_and_Matched_Baseline_Protocol.md#p-xrb-f).

**Status in one line.** Part A hit 2 of 8 calls: the register path is a broad channel and the exchange field a copying specialist. Part B hit all six: the field alone is worth 6.0% to the conservative core, three quarters of what the register path is worth, and almost nothing beside it.

## Contents

1. [The two routes and the terms](#1-the-two-routes-and-the-terms)
2. [Part A: the routes on the trained models](#2-part-a-the-routes-on-the-trained-models)
3. [Part B: the exchange-field-only model, XRB-F](#3-part-b-the-exchange-field-only-model-xrb-f)
4. [XRB-F's route measurements](#4-xrb-fs-route-measurements)
5. [Synchronization readings on XRB-F](#5-synchronization-readings-on-xrb-f)
6. [What remains open](#6-what-remains-open)

## 1. The two routes and the terms

The Gen 3 models carry two attention-like routes between tokens, both treated as measuring instruments:

- **The exchange field**: one hop, a force inside the integrator, recomputed every layer. It has 8 heads.
- **The register path**: two hops. Tokens write to 32 registers (the write port), and the reverse channel reads them back after the step (the only read port).

Before XR they had been compared only by their value as trained. The register path is worth 8.0% over the conservative core (F3.1, 57.76, to G2, 53.12). The exchange field adds nothing measurable on top of it (G3 54.21, G3′ 52.90). G3′ is G3 with hardened routing (QK-norm and a 0.3 field clip). The field had never been trained without the registers, so "weak" could not be told from "redundant".

**Terms.**

- **Removal cost**: the rise in per-token loss, in nats, when a route is switched off at inference on the trained weights. It measures **reliance**, not the value of training without the route (§5.20).
- **Overlap**: $1 - c_{\mathrm{both}} / (c_{\mathrm{field}} + c_{\mathrm{read}})$, from the removal costs. It is 0 when the routes carry separate loss.
- **Target classes**, by the next token. **Induction**: the current token occurred earlier, and the next token is the one that followed it then. **Repeated**: the next token occurred earlier otherwise. **Novel**: it did not occur earlier.
- **The kernel** of a route is how much each earlier token feeds each later one (rows normalised to sum to 1). From it: the share of routing mass more than 64 tokens back; the **effective number of sources** (the exponential of the row entropy); the **effective rank**; and the **induction score** (mass on induction targets).
- **Retained value** at $k$: the share of a route's removal cost recovered by keeping only its top $k$ heads or registers.

## 2. Part A: the routes on the trained models

**Arms and data.** G3 and G3′ (both routes) and G2 (registers only), on their best checkpoints, loaded with the corrected builder (§5.22). The SY validation set, 200 sequences of 512 tokens; 100 sequences for removal, 20 for kernels and capacity. All switches restore the model bit for bit.

**Amendment, before any scored run.** Removing registers by masking them renormalises the read over those left, and in a smoke run that did worse than no read at all. For a like-for-like measure, a removed register's terms are dropped and the rest keep their trained weights; XR6a is scored this way.

**Removal at inference** (nats per token):

| | G3 | G3′ | G2 |
| --- | ---: | ---: | ---: |
| full model, PPL on these sequences | 51.36 | 50.26 | 49.95 |
| read off (register path removed) | 1.56 [1.52, 1.60] | 1.34 [1.30, 1.38] | 1.42 [1.34, 1.50] |
| field off | 0.52 [0.45, 0.60] | 0.47 [0.40, 0.55] | — |
| both off | 1.61 | 1.66 | — |
| overlap | 0.225 | 0.084 | — |

**Cost by target class** (induction 8.3% of positions, repeated 40.3%, novel 51.4%):

| | induction | repeated | novel | induction ÷ novel |
| --- | ---: | ---: | ---: | ---: |
| G2, read off | 2.42 | 1.33 | 1.25 | 1.93 |
| G3, read off | 1.06 | 1.52 | 1.65 | 0.64 |
| G3′, read off | 0.94 | 1.39 | 1.36 | 0.69 |
| G3, field off | 1.75 | 0.51 | 0.27 | 6.56 |
| G3′, field off | 1.74 | 0.44 | 0.22 | 7.84 |

**The kernels at the last layer** (rows from position 64 on; at layer 0 the register path transfers nothing between tokens):

| | mass > 64 back | mean distance | effective sources | effective rank | induction score |
| --- | ---: | ---: | ---: | ---: | ---: |
| G2 register path | 0.53 | 96 | 69 | 28.7 | 0.075 |
| G3 register path | 0.55 | 103 | 61 | 32.3 | 0.057 |
| G3 exchange field | 0.58 | 111 | 79 | 7.6 | 0.132 |
| G3′ register path | 0.52 | 108 | 58 | 10.2 | 0.033 |
| G3′ exchange field | 0.64 | 126 | 79 | 7.4 | 0.158 |

Row overlap of the two kernels: cosine 0.29 (G3) and 0.49 (G3′).

**Capacity** (retained value keeping the top $k$):

| | k = 1 | 2 | 4 | 8 | 16 | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G2 registers | 0.10 | 0.48 | 0.67 | 0.85 | 0.95 | 1 (32) |
| G3 registers | 0.08 | −0.08 | 0.25 | 0.50 | 0.96 | 1 (32) |
| G3′ registers | 0.06 | 0.02 | 0.06 | 0.52 | 0.96 | 1 (32) |

| | k = 1 | 2 | 4 | 6 | all |
| --- | ---: | ---: | ---: | ---: | ---: |
| G3 heads | 0.43 | 0.57 | 0.80 | 0.95 | 1 (8) |
| G3′ heads | 0.44 | 0.60 | 0.84 | 0.95 | 1 (8) |

| call | prediction | called | result |
| --- | --- | ---: | --- |
| XR1 | overlap ≥ 0.20 in G3 and G3′ | 60% | MISS (0.225, 0.084) |
| XR2 | the read's removal cost exceeds the field's | 75% | HIT (1.56 vs 0.52; 1.34 vs 0.47) |
| XR3 | the register path reaches further (mass beyond 64 tokens) | 55% | MISS: the field reaches slightly further |
| XR4 | row cosine between the kernels below 0.3 | 55% | MISS (0.29, 0.49) |
| XR5a | the read's induction ÷ novel cost ≥ 1.5 in G2, G3 and G3′ | 50% | MISS: 1.93 in G2, 0.64 and 0.69 beside the field |
| XR5b | the field's induction ÷ novel cost ≥ 1.5 | 45% | HIT (6.56, 7.84) |
| XR6a | the top 8 registers retain ≥ 90% | 40% | MISS (0.50, 0.52) |
| XR6b | the top 4 heads retain ≥ 90% | 50% | MISS (0.80, 0.84) |

**Reading.**

1. **The register path is the broad channel.** It carries most of the loss at every kind of position, and removing it costs three times as much as removing the field. It needs about 16 of its 32 registers to keep 95% of its value.
2. **The exchange field is a copying specialist**: 1.75 nats at induction positions against 0.22–0.27 at novel ones. This is the job of attention's induction heads.
3. **Beside the field, the register path gives induction up.** Alone in G2, the read serves induction (2.42 nats there). Beside the field its induction cost falls to 0.94–1.06. The total copying capacity is about the same; it is divided differently.
4. **"The field adds nothing" does not mean it is idle.** It takes over a job the register path does when the field is absent.
5. **Both routes are wider than predicted**, and in G3′ they carry nearly separate loss (overlap 0.08 against 0.23 in G3).

## 3. Part B: the exchange-field-only model, XRB-F

**The arm.** F3.1's configuration exactly, plus G3′'s hardened exchange field. Seed 0, 532.5M tokens, the ladder's schedule. **A caveat for all four cells:** "without registers" means as in F3.1. The 32 registers are still built and written, and their repulsion term (coefficient 0.05) still acts in training, but nothing reads them. A fully register-free control is a separate question.

**The factorial, completed** (settled PPL, $L = 2$, Gen 3; gain against F3.1):

| | no exchange field | exchange field (hardened) |
| --- | ---: | ---: |
| no register read | 57.76 (F3.1) | 54.27 (XRB-F, −6.0%) |
| register path | 53.12 (G2, −8.0%) | 52.90 (G3′, −8.4%) |

The run finished cleanly at step 32,500, with every causal-leak audit CLEAN. Settled 54.27 is the mean of 54.20, 54.39 and 54.22; the best single evaluation was 52.50, at step 31,000.

| call | prediction | called | result |
| --- | --- | ---: | --- |
| XRB1 | the field alone helps: at most 56.60 | 55% | HIT (54.27, 6.0% below F3.1) |
| XRB2 | it does not match the register path: above 53.12 | 70% | HIT (54.27, 2.2% above G2) |
| XRB3 | it is worth more alone than beside the registers, by at least 2 points | 55% | HIT (6.0% against 0.4%) |
| XRB4 | its removal cost at novel positions is at least 0.50 nats | 50% | HIT (1.16) |
| XRB5 | its removal cost at induction positions is at least 1.74 nats | 60% | HIT (2.61) |
| XRB6 | Gate 3 at 1.5× below F3.1's +143% | 40% | HIT, narrowly (+134%) |

**Reading, by the rule fixed in advance** (XRB1 and XRB2 both hit): the field is a real but weaker route, and the register path stays the stronger instrument. The two largely substitute for each other: with both, the model is only 0.4 points better than with the register path alone. One seed per arm; settled values have an evaluation spread of about 1 PPL either way.

**Refinement** (Cell 6b-7, best checkpoint; Gate 0 passes):

| | XRB-F | G3′ | G2 | F3.1 |
| --- | ---: | ---: | ---: | ---: |
| Gate 1, velocity reset | +113% | +34% | +33% | +34% |
| Gate 2, N = 3 at fixed Δt | +112% | +441% | +51% | +92% |
| Gate 3, 1.5× refinement | +134% | +115% | +1,274% | +143% |

- **The exchange field does not cause the large refinement failures.** Without the register path the model refines about as well as F3.1 and G3′. Gate 3 still grows at every step count, so the model is still a map, not a flow.
- **Gate 1 is the surprise:** three times the cost in every other two-layer arm. A plausible account is that the velocity carries forward what the register path otherwise supplies. It is untested and was not predicted, so it is descriptive.

## 4. XRB-F's route measurements

Same script and sequences as part A. The local build reproduces Colab (52.7508 against 52.7510).

| removal cost (nats per token) | induction | repeated | novel | all positions | induction / novel |
| --- | ---: | ---: | ---: | ---: | ---: |
| XRB-F: exchange field (no register read) | 2.61 | 1.17 | 1.16 | 1.32 [1.24, 1.41] | 2.25 |
| G2: register path (no field) | 2.42 | 1.33 | 1.25 | 1.42 | 1.94 |
| G3′: exchange field beside the register path | 1.74 | 0.44 | 0.22 | 0.47 | 7.84 |
| G3′: register path beside the field | 0.94 | 1.39 | 1.36 | 1.34 | 0.69 |

| the route's kernel at layer 1 | effective rank | effective sources | weight on induction targets | weight beyond 64 tokens |
| --- | ---: | ---: | ---: | ---: |
| XRB-F field | 25.1 | 107 | 0.142 | 0.57 |
| G3′ field | 7.4 | 79 | 0.158 | 0.64 |
| G2 register path | 28.7 | 69 | 0.075 | 0.53 |

**Capacity.** Two of the 8 heads carry most of the field (single-head removal costs 0.68 and 0.43 nats; the next is 0.10). The top 2 heads keep 53% of its value, the top 4 keep 80%, the top 6 keep 93%. Beside the registers no head stood out (largest single cost 0.074).

**Reading.**

1. **Whichever route is alone becomes the broad channel.** The field alone has almost the same profile as the register path alone in G2, within 0.2 nats at every kind of position. Its kernel widens to match (effective rank 25 against 7.4 beside the registers). It keeps its preference for induction targets. So part A's division of labour is not fixed: it arises when the two routes are trained together.
2. **Removal costs measure reliance, not value.** Removing the field costs 1.32 nats, but training with it gained only 0.062 nats over F3.1. G2's register path likewise: 1.42 against 0.084.

## 5. Synchronization readings on XRB-F

The eight SY scripts (§5.22) were run on 2026-10-18, descriptive since part B fixed no SY calls. Every check is exact. See [Synchronization_Tests_SY.md](Synchronization_Tests_SY.md) for the measures.

- **The field draws no tokens together** by its token-specific part. At layer 0, read before normalisation, the two forms agree (mean $-0.009$, below the floor). At layer 1 it spreads the tokens strongly.
- **Its shared push is six times larger than beside the registers**: it moves every token by 1.7 times the tokens' spread at layer 0 and 3.2 times at layer 1 (G3′: 0.28 and 0.71). This is a common-mode effect, not synchronization.
- **Without registers the field separates related words** at the last layer: noun-chunk members apart ($-0.046$), heads from dependents ($-0.029$). The context field $\xi$ still groups noun chunks (+0.049), so the two nearly cancel on chunks.
- **No reference binding**, as everywhere; the coreference subspace check reads LEXICAL.
- **Extra depth over-synchronizes** (one extra step: perplexity 49.6 to 117.7, $S$ +0.30, rank down 41%).

## 6. What remains open

- **A second run without the hardening** would separate the hardening's effect; the pre-registration lists it as optional.
- **A fully register-free control** (no register writes and no repulsion term) is a separate question.
- **The Gate 1 account** (velocity standing in for the register path) is untested.
- **Seeds.** Every arm has one seed; the protocol gives the settled values an evaluation spread of about 1 PPL either way. All of XR is at $L = 2$ and 532.5M tokens.
