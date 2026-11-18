# Dead wells on code: XI6-CODE, the two screens and their five-channel twins

**What this note is.** A short, readable summary of the protocol entries on the dead-wells episode on the code corpus, written 2026-11-17. It covers XI6-CODE, the root cause found in it, the two screens that kept its wells alive (screen A and screen B, each at 3,000 steps and at full length), and the two SO-CODE twins that test each fix on a healthy model (SO-CODE-VTLR and SO-CODE-VTUN). Every number, verdict and date is copied from the [protocol](../Depth_Ladder_and_Matched_Baseline_Protocol.md), which remains the full log. The full technical report, with figures and the debugging steps, is [Dead_Wells_in_the_Joint_V_theta_Bank.md](../Dead_Wells_in_the_Joint_V_theta_Bank.md). Each heading below names the protocol anchor it summarises, so a reader of a model card can match the two.

**Status in one line.** Done and scored. A sixth long-range context channel kills layer 1's wells at full $V_\theta$ speed. Either of two switches keeps them alive, and XI6 then ends level with SO-CODE. Neither switch is free on a healthy five-channel model.

---

## Contents

1. [Terms used](#1-terms-used)
2. [XI6-CODE and the root cause](#2-xi6-code-and-the-root-cause)
3. [The two screens at 3,000 steps](#3-the-two-screens-at-3000-steps)
4. [Screen B and its twin](#4-screen-b-and-its-twin)
5. [Screen A and its twin](#5-screen-a-and-its-twin)
6. [What the episode established](#6-what-the-episode-established)

---

## 1. Terms used

- **SO-CODE** is the reference: the second-order conservative model on Python code, $L = 2$ layers, five context channels. It settled at perplexity (PPL) 6.00. The matched GPT-2 on code, GPT-2-CODE, settled at 4.96. Lower perplexity is better. See [Code_Set_First_vs_Second_Order.md](Code_Set_First_vs_Second_Order.md).
- **Settled perplexity** is the mean of the evaluations at steps 18,000, 18,500 and 19,000. Every arm here trains from step 0 on the same batches, so evaluations at the same step are **paired**. Paired gaps are in nats; 0.01 nats is about 1% in perplexity.
- **$V_\theta$'s well bank.** The potential $V_\theta$ is a bank of 8 Gaussian wells. One bank serves both layers; a learned depth code tells the layers apart. Layer-1 states lie on a sphere of radius $\lvert h \rvert = \sqrt{384} \approx 19.6$.
- **Dead well.** A well whose centre has drifted far from every state. Its **occupancy** (how much it weighs on a state) is zero, so it exerts no force and receives no gradient, and it cannot come back.
- **Nearest-well ratio.** The distance to the nearest well divided by $\lvert h \rvert$. Near 1 is healthy; the dead layer in XI6-CODE read 3.43.
- **Removal cost.** The perplexity change when every well of a layer is switched off on the trained model. A large cost means the layer relies on its wells.
- **The two switches.** `VTHETA_LR_MULT = 0.25` (tag `vtlr0p25`) trains every $V_\theta$ parameter at a quarter of the learning rate. `VTHETA_INPUT_UNIT_NORM = True` (tag `vtun`) scales every context channel to unit norm at the bank's input.
- **Gates.** Cell 6b-7: Gate 1 resets the velocity between layers; Gate 3 refines the step at fixed total time, with its penalty in nats. Cell 6b-13 reads $\omega \Delta t$, a mode's turn per layer step.

## 2. XI6-CODE and the root cause

### 2.1 XI6-CODE (protocol `p-xi6-code`)

Proposed 2026-10-27 after the copy split; the author decided it should run. It is SO-CODE's Cell 0 with `XI_OVERRIDE = '6long'`: a sixth content-routed context channel, initialised like the fifth. Finished 2026-10-28 in 6.0 h, clean; 83,690,702 parameters. The bank's input grows with the channel count, which is a disclosed confound: +9.3% parameters, mostly in the bank.

| reading | XI6-CODE | SO-CODE |
| --- | --- | --- |
| settled PPL (evals 8.45, 8.58, 8.62) | 8.55 | 6.00 |
| against SO-CODE / GPT-2-CODE | +42.7% / +72% | |
| paired gap, steps 16,000–19,000 | 0.352 nats (SD 0.009) | |
| Gate 1, velocity reset | +11.8% | +4.0% |
| Gate 3 at N = 3 | +18.3% (0.168 nats) | +89% (0.639) |
| ω·Δt median (layer 0 / layer 1) | 2.61 (2.84 / 0.000) | 7.55 (9.58 / 4.94) |

The two arms are level to step 2,000, then XI6-CODE falls to a fixed deficit of 40% to 45% from step 5,500 on. Its training loss is as far behind, so it underfits; it does not overfit.

**Calls** (registered 2026-10-27, binding: the author started training without changes):

- **XI1** (more than 2% below SO-CODE, called 50%): **MISS** (+42.7%).
- **XI2** (the gain is long-range, called 60%): **MISS.** There is no gain to locate: XI6-CODE widens the gap in every group of the copy split, most at long range.

### 2.2 Root cause: the layer-1 wells died (protocol `p-dead-wells`)

Found 2026-10-28 on CPU with `debug/vtheta_well_probe.py`.

- **Every layer-1 well left the region the states live in.** The nearest well is 67 away against SO-CODE's 20 (with $\lvert h \rvert \approx 19.6$). Its occupancy is 0.000 against 0.18, and layer 1's $\omega \Delta t$ is 0. Layer 1 runs without its potential.
- **Not the duplicated channel.** Channels 5 and 6 had separated by step 500.
- **Not present at step 500,** where the two arms are indistinguishable.
- **The depth code is the counterweight that failed.** In every healthy arm the depth-code term shrinks the layer-1 centres; in XI6-CODE it does not.
- **Layer 1 is marginal in every arm, and XI6-CODE is the one that crossed.** Layer 1 works on one or two wells everywhere, so there is no spare.
- **The long channels in layer 1 stopped retrieving,** turning into near one-position lookups. Whether this followed from the dead wells or caused them was left open here.

### 2.3 A first fix that failed (protocol `p-well-reach`, `p-xi6-code-wr`)

A well-reachability penalty pulled the nearest well towards the states when it was far. XI6-CODE-WR (2026-10-29) settled at 8.87; layer 1 stayed dead (calls WR1, WR3, WR4 MISS; WR2 not scored, since the author dropped SO-CODE-WR). The penalty's threshold was set against the healthy arms' tails, not against the wells' width. The run dated the deaths: layer 1's wells died within about 350 steps, just after the learning rate reached its peak (step 812).

## 3. The two screens at 3,000 steps

### 3.1 Setup (protocol `p-dead-wells-screens`)

Two short screens on the death window, 2026-10-29. The leading hypothesis was that layer 1 loses a race: its well centres share a weight matrix with layer 0, the matrix grows about fourfold, and layer-1 centres inflate with it before layer 1 uses its wells. Each screen is XI6-CODE-WR's Cell 0 plus one switch, stopped at step 3,000 (about 1 h). Screen A is the `vtun` switch and screen B the `vtlr0p25` switch. The control is XI6-CODE-WR.

| id | reading | proposed |
| --- | --- | ---: |
| DW1 | screen A keeps layer 1 alive to step 3,000: ratio at or below 1.2 from step 1,000 on, strongest-well occupancy at least 0.05 at step 3,000 | 55% |
| DW2 | screen B delays but does not prevent the drift: ratio rises over steps 1,000–3,000 and passes 1.5 | 60% |
| DW3 | screen A costs no more than 5% in perplexity against the control while layer 1 is alive in both | 50% |

### 3.2 Result: both keep layer 1 alive (protocol `p-dead-wells-screens-result`)

Run 2026-10-29, about 59 min each, clean. The DW calls were proposed and **not confirmed by the author** before the runs; they are scored as proposed.

| step | PPL: control / A / B | layer-1 nearest-well ratio: control / A / B |
| ---: | --- | --- |
| 500 | 43.35 / 35.75 / 50.22 | 0.95 / 0.99 / 1.04 |
| 1,000 | 25.15 / 22.03 / 28.29 | 1.35 / 0.99 / 0.96 |
| 2,000 | 17.32 / 16.21 / 15.87 | 2.45 / 0.98 / 0.87 |
| 3,000 | 14.80 / 13.54 / 11.27 | 2.55 / 0.97 / 0.82 |

- **DW1 HIT.** Screen A's ratio is at most 0.99 from step 1,000 on; occupancy 0.49 at step 3,000.
- **DW2 MISS.** Screen B's ratio falls (0.96 to 0.82). It is not only a delay: at equal cumulative learning rate, B's wells are alive where the control's were dying.
- **DW3 HIT,** by more than called: screen A is 12.4% better than the control at step 1,000.
- **The race reading holds from both sides.** Screen A fixes the channel size at 1, so the centres stay small. Screen B slows the weight's growth. In both, layer 1 starts to use its wells before the centres can leave.
- **Two layer-1 geometries.** Screen B's layer 1 works like SO-CODE's, on one well. Screen A's spreads over five to eight wide wells near the origin.

## 4. Screen B and its twin

### 4.1 Screen B at full length (protocol `p-dw-screen-b-full`)

Resumed 2026-10-30 from the step-3,000 checkpoint; steps 3,001 to 19,250 in 5.2 h, clean.

| reading | screen B | SO-CODE | XI6-CODE |
| --- | --- | --- | --- |
| settled PPL (evals 5.99, 5.91, 6.04) | 5.98 | 6.00 | 8.55 |
| best | 5.86 at step 16,000 | | |
| paired gap to SO-CODE, steps 16,000–19,000 | +0.002 nats (SD 0.034) | | +0.352 |
| settling check | −0.022 against SD 0.040: settled | | |
| layer-1 removal cost | +242% | +202% | — |
| Gate 1, velocity reset | +2.5% | +4.0% | +11.8% |
| Gate 3 at N = 3 | +34% (0.290 nats) | +89% (0.639) | +18.3% (0.168) |

Layer 1's ratio stays at 0.76 to 0.83 to the end, with no late death. The layer works like SO-CODE's: one well carries the force. With live wells there are no one-position lookups, so the lookups in XI6-CODE followed from the dead wells.

**Reading.** The root cause is confirmed by a fix: XI6-CODE's whole +43% deficit came from the dead layer-1 wells. The sixth channel buys nothing on code. Not yet separated at this point: the fix from a general gain of a slower $V_\theta$. That needed the SO-CODE twin.

### 4.2 SO-CODE-VTLR: registered (protocol `p-so-code-vtlr`)

Registered 2026-10-30 with the author's go-ahead. SO-CODE's Cell 0 plus `VTHETA_LR_MULT = 0.25` and early snapshots; fresh from step 0. Its question: does a slower $V_\theta$ alone move SO-CODE? Calls proposed, **not confirmed by the author**:

| id | prediction | called |
| --- | --- | ---: |
| VT1 | settles within ±2% of SO-CODE (5.88–6.12) | 55% |
| VT2 | more than 5% ahead of SO-CODE at step 3,000 (paired) | 65% |
| VT3 | layer-1 wells alive: ratio below 1.2 throughout, removal cost at the end above +100% | 90% |
| VT4 | screen B and SO-CODE-VTLR within ±2% of each other | 60% |

### 4.3 SO-CODE-VTLR: result (protocol `p-so-code-vtlr-result`)

Run 2026-10-30, 6.0 h, clean.

| reading | SO-CODE-VTLR | screen B | SO-CODE |
| --- | --- | --- | --- |
| settled PPL (evals 5.96, 6.11, 6.20) | 6.09 | 5.98 | 6.00 |
| against SO-CODE | +1.6% | | |
| paired gap to SO-CODE, steps 2,000–4,000 | −0.121 nats | | |
| paired gap to SO-CODE, steps 16,000–19,000 | +0.015 nats (SD 0.002) | | |
| layer-1 removal cost, end | +272% | +242% | +202% |
| Gate 3 at N = 3 | +37% (0.312 nats) | +34% (0.290) | +89% (0.639) |

| call | result | verdict |
| --- | --- | --- |
| VT1 | 6.09, +1.6% | HIT, near the edge |
| VT2 | −12.7% at step 3,000 (11.32 against 12.96) | HIT |
| VT3 | ratio at most 1.03 at all seven checkpoints; +272% | HIT |
| VT4 | 5.98 against 6.09, −1.8% | HIT, near the edge |

**Reading.** The slower $V_\theta$ alone gives screen B's early lead and the halved refinement penalty. On SO-CODE, whose wells never died, it ends 1.5% behind, consistently from step 14,000 on (one seed). So screen B is a dead-well fix that costs nothing on XI6, not a general gain, and `VTHETA_LR_MULT = 0.25` is not a setting to adopt for healthy arms. Net, the sixth channel neither helps nor hurts.

## 5. Screen A and its twin

### 5.1 Screen A at full length: launched (protocol `p-dw-screen-a-full`)

Launched 2026-10-31, resumed from the step-3,000 checkpoint. The question: unit-norm input caps the channels' size rather than slowing the optimizer, so if it is free it could be a default. The calls were proposed 2026-10-30 and Cell 6 started without changes, so they are registered as written; **the author has not confirmed them**.

| id | reading | proposed |
| --- | --- | ---: |
| VA1 | settles within ±2% of SO-CODE (5.88–6.12) | 55% |
| VA2 | layer 1 alive at the best checkpoint: ratio below 1, strongest-well occupancy above 0.05 | 60% |
| VA3 | Gate 3 (N = 3) above 0.45 nats, near SO-CODE's 0.639 | 60% |
| VA4 | paired gap to SO-CODE over steps 16,000–19,000 within ±0.02 nats | 55% |

### 5.2 Screen A at full length: result (protocol `p-dw-screen-a-full-result`)

Steps 3,001 to 19,250 in 5.3 h, clean.

| reading | screen A | screen B | SO-CODE |
| --- | --- | --- | --- |
| settled PPL (evals 5.96, 5.88, 6.03) | 5.95 | 5.98 | 6.00 |
| best | 5.85 at step 16,000 | | |
| paired gap to SO-CODE, steps 16,000–19,000 | −0.002 nats (SD 0.035) | | |
| layer-1 removal cost | +136% | +242% | +202% |
| Gate 1, velocity reset | +11.5% | +2.5% | +4.0% |
| Gate 3 at N = 3 | +80% (0.587 nats) | +34% (0.290) | +89% (0.639) |

| call | result | verdict |
| --- | --- | --- |
| VA1 | 5.95, −0.7% | HIT |
| VA2 | ratio 0.68, occupancy 0.289; removal cost +136% | HIT (carried by occupancy and removal cost) |
| VA3 | 0.587 | HIT |
| VA4 | −0.002 (SD 0.035) | HIT |

- **The wide wells did not stay wide.** By step 16,000 one well carries 72% of layer 1's force.
- **The nearest-well ratio alone is a weak test here.** Wells near the origin read near 1 whatever their use. Layer 1 is shown alive by occupancy and removal cost.
- **Resetting the velocity costs more** (+11.5%), not explained by these readings.
- **Reading.** Two fixes, both free on XI6. Screen A keeps SO-CODE's refinement penalty, so the halved penalty of the two arms with the slower $V_\theta$ belongs to the slower $V_\theta$. Unit-norm input is a candidate default only if it costs SO-CODE nothing.

### 5.3 SO-CODE-VTUN: registered (protocol `p-so-code-vtun`)

Registered 2026-10-31 with the author's go-ahead. SO-CODE's Cell 0 plus `VTHETA_INPUT_UNIT_NORM = True` and early snapshots; fresh from step 0. Calls proposed, **not confirmed by the author**; launched the same day with the calls unchanged.

| id | prediction | called |
| --- | --- | ---: |
| VU1 | settles within ±2% of SO-CODE (5.88–6.12) | 65% |
| VU2 | paired gap to SO-CODE over steps 16,000–19,000 below +0.015 nats | 55% |
| VU3 | layer-1 wells alive: occupancy above 0.05 and removal cost above +100% at the best checkpoint | 85% |
| VU4 | Gate 3 at N = 3 above 0.45 nats | 75% |

### 5.4 SO-CODE-VTUN: result (protocol `p-so-code-vtun-result`)

Run 2026-10-31, 6.1 h, clean.

| reading | SO-CODE-VTUN | screen A | SO-CODE |
| --- | --- | --- | --- |
| settled PPL (evals 6.23, 6.37, 6.43) | 6.34 | 5.95 | 6.00 |
| against SO-CODE | +5.8% | | |
| paired gap to SO-CODE, steps 16,000–19,000 | +0.056 nats (SD 0.002) | | |
| layer-1 removal cost, end | +83% | +136% | +202% |
| Gate 1, velocity reset | +25.3% | +11.5% | +4.0% |
| Gate 3 at N = 3 | +60% (0.473 nats) | +80% (0.587) | +89% (0.639) |

| call | result | verdict |
| --- | --- | --- |
| VU1 | 6.34, +5.8% | MISS |
| VU2 | +0.056 (SD 0.002) | MISS |
| VU3 | occupancy 0.25; removal cost +83% | MISS on the removal cost (the wells are alive; layer 1 carries less) |
| VU4 | 0.473 | HIT, near the edge |

**Reading, by the registered table.** Unit-norm input hurts a healthy arm: 5.8% on SO-CODE, nearly four times the slower $V_\theta$'s 1.5%. It is a fix for XI6's dead wells, not a default. Screen A's parity is the sixth channel making up for it: with unit-norm input in both, the sixth channel is worth about 6%. Why unit-norm input costs five channels so much is not shown. The protocol offers one untested possibility: scaling to norm 1 discards each channel's size, which may carry information.

## 6. What the episode established

| bank input | five channels | six channels | six against five |
| --- | ---: | ---: | --- |
| as trained (SO-CODE, XI6-CODE) | 6.00 | 8.55 | +42.7%: layer 1's wells die |
| V_θ learning rate × 0.25 (SO-CODE-VTLR, screen B) | 6.09 | 5.98 | −1.8%, within the paired noise |
| unit-norm channels (SO-CODE-VTUN, screen A) | 6.34 | 5.95 | −6.1% |

- **No configuration beats SO-CODE.** The best code arms (SO-CODE and screens A and B) all settle at 5.95 to 6.00.
- **The sixth channel adds nothing net.** With its wells kept alive by either switch, it recovers what the switch costs, no more.
- **Neither switch is free** on a five-channel arm (+1.5% and +5.8%).
- **Consequence for depth.** L4-CODE ran on SO-CODE's Cell 0 with neither switch. If a deeper arm ever needs a guard, the protocol proposes the slower $V_\theta$ as the cheaper fix. That choice is **proposed, not confirmed by the author**. See [Code_Depth_Runs.md](Code_Depth_Runs.md).
- **Caveats.** One seed per arm. Calls within the $\pm 2\%$ band are read as "no effect detected", not as equality.

On Hugging Face, screen A sits in the dead-wells collection beside screen B, and SO-CODE-VTUN is the collection's fifth item, after screen A (both published 2026-10-31 at the author's request). The SO-CODE, screen B and SO-CODE-VTLR cards point to both.
