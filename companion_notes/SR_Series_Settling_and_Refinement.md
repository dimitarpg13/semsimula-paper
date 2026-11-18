# The SR series: settling, refinement and depth extension

**What this note is.** A short, readable summary of the SR series of the depth ladder, written 2026-11-17. It gives, for each arm, the question it asks, what was registered before it ran, and what happened. Every number, verdict and date below is copied from the [protocol note](Depth_Ladder_and_Matched_Baseline_Protocol.md), which remains the record of truth. The registration is [protocol, `p-sr`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr) (§5.9, pre-registered 2026-10-03, before any run) and [protocol, `p-sr5`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr5). The refinement tests themselves are explained with figures in [Refinement_Experiments_and_Variable_Step_Training.md](Refinement_Experiments_and_Variable_Step_Training.md).

**Status in one line.** SR2, SR3 and SR4a have run and are scored. SR1, SR4b, SR5a and SR5b have not run. SR5c was conditional on SR4b and was never scheduled.

---

## 1. The problem the series addresses

A conservative layer moves each token's hidden state $h$ for a time $\Delta t$ under a learned potential, with a velocity carried from layer to layer and a friction $\gamma$. A stack of $L$ layers covers a total time $T = L \Delta t$. The ladder models use $L = 2$ and $T = 8$, so $\Delta t = 4$.

If the layers really sample a continuous flow, the model should keep working when the same time $T$ is cut into more, smaller steps. Let $N$ be the number of steps the stack walks at inference. At $N = L$ it is the trained model. Two properties are wanted:

- **Refinement:** at fixed $T$, taking $N \gt L$ steps of $\Delta t = T/N$ should not hurt.
- **Settling:** the velocity at the end of the trajectory should be small, so the state has come to rest.

The protocol reads these with three gates (Cell 6b-7 of the ladder notebook). Each compares perplexity (PPL) against the trained model:

| gate | what changes at inference | what it measures |
| --- | --- | --- |
| Gate 1 | the velocity entering each layer is reset to zero | how much the model relies on inertia |
| Gate 2 | more steps at the trained step size, so T grows | depth extension |
| Gate 3 | more steps at fixed T, so the step size shrinks | refinement |

The **refinement penalty** is $\ln(\mathrm{PPL_N} / \mathrm{PPL_L})$ in nats, the Gate 3 cost at $N$ steps. A penalty of 0.095 nats is a 10% rise in perplexity.

Two more quantities recur. The **damping ratio** $\zeta$ of a mode compares its friction with the friction that would make it critically damped: $\zeta = 1$ is critical (no oscillation), and $\zeta$ near 0 is a lightly damped oscillation. The product $\omega \Delta t$ is how far, in radians, a stiff mode of angular frequency $\omega$ rotates in one layer step.

**The baseline.** Every arm starts from F3.1, the $L = 2$ conservative-only configuration with live gradients, retrained from scratch for 32,500 steps. Its readings in [protocol, `p-sr`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr):

| F3.1 reading | value |
| --- | ---: |
| settled PPL | 57.76 |
| Gate 1, velocity reset | +34.3% |
| Gate 2, N = 3 at fixed Δt | +92% |
| Gate 3, N = 3 at fixed T | +143% |
| ω·Δt median | 4.32 |

**The hypothesis.** The book's settling-and-refinement subsection (`ssec:settling-refinement`, Propositions 44 and 45 at registration) shows that the split $A \cdot O \cdot A$ step makes a stiff mode's dissipation depend on the step count, through the factor $\theta / \sin\theta$ with $\theta = \omega \Delta t$. The registration names a common cause behind the failures: training at a single discretisation with weak damping. Each arm attacks one part of that cause.

## 2. The arms

Thresholds are on the F3.1 readings above, at $N = 3$ against the trained $N = 2$ (1.5 times the steps).

| arm | what it changes | registered thresholds | status | headline |
| --- | --- | --- | --- | --- |
| SR1 | palindromic step order A·O½·B·O½·A | all gates within ±25% of F3.1; PPL within ±2% (no change predicted) | not run | — |
| SR2 | exact damped flow on the stiff subspace, constant γ = 0.1 | Gate 3 ≤ +100%; Gate 1 within ±25%; PPL within ±3% | scored 2026-10-09 | Gate 3 at N = 3 +50% (0.405 nats), PPL +1.0% |
| SR3 | SR2 plus constant-ratio friction ζ* = 1 on the stiff modes | Gate 3 ≤ +70%; Gate 2 ≤ +50%; Gate 1 ≤ +25%; PPL within +5% | run as SR3-crit, scored 2026-10-20 | PPL 63.28, +8.5% against SR2 |
| SR4a | train with N drawn from {2, 3, 4} at fixed T = 8 | Gate 3 ≤ +10% on N in {2, 3, 4}; Gate 1 ≥ +27%; PPL within +5% | run on code as SR4a-CODE, scored 2026-11-04 | refines inside {2, 3, 4}, fails at N = 6 and 8 |
| SR4b | train with N drawn from {2, 3} at fixed Δt = 4 | Gate 2 ≤ +10% at N = 3; PPL within +5% | not run | — |
| SR5a | thermal training, LANGEVIN_T = 0.00025 (r ≈ 0.1) | all gates within ±25% of F3.1; PPL within +2% | not run | — |
| SR5b | thermal training, LANGEVIN_T = 0.0022 (r = 0.3) | Gate 3 ≤ +100% (45%); Gate 2 ≤ +70% (40%); Gate 1 ≤ +25% (60%); PPL +2% to +8% | not run | — |

**The registered decision rule** (§5.9, to be read once SR2 to SR4a were in):

- If SR2 or SR3 meets its Gate 3 threshold and its PPL band, the split step or the underdamping is a cause. Adopt it in the integrator.
- If only SR4a meets it, the single-discretisation objective is the cause. Variable-step training becomes the default for any model meant to have depth as an inference-time knob.
- If both meet it, the effects add, and SR2 and SR4a run together.
- If neither does, the cause lies in what no arm touches. Record a negative result.
- **The common-cause check.** Gate 1 and Gate 3 had been perfectly rank-ordered across the $L = 2$ arms. SR4a was predicted to break that ordering (keep inertia, become refinable). SR3 and SR5 were predicted to keep it.

**SR5's logic** ([protocol, `p-sr5`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr5)). Velocity noise in training only randomises the phase of the one discrete trajectory, so the readout cannot rely on it. The temperatures were calibrated on F3.1's checkpoint so that the thermal speed is $r$ times the typical speed. SR5 is not a settling route at inference: with $\gamma T = 0.8$ the relaxation time exceeds the trajectory.

## 3. SR2: the exact damped stiff-mode flow

[protocol, `p-sr2`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr2). SR2 replaces the split step on the stiff subspace with the closed-form flow of a forced damped oscillator, which removes the $\theta / \sin\theta$ phase term. Before the run, §5.19 of the protocol (2026-10-06) added two calls: E1, the $N = 3$ penalty halves to 0.45 nats or below (55%), and E2, settled PPL at or below 59.5 (60%).

| reading | SR2 | F3.1 |
| --- | ---: | ---: |
| settled PPL | 58.32 | 57.76 |
| Gate 1, velocity reset | +3.9% | +34.3% |
| Gate 2, N = 3 | +35% | +92% |
| Gate 3, N = 3 (nats) | 0.405 (+50%) | 0.889 (+143%) |
| Gate 3, N = 8 (nats) | 2.110 (+725%) | 2.275 (+873%) |
| seconds per step | 1.72 | 1.28 |

- **E1: HIT** (0.405 nats). **E2: HIT** (+1.0%). Both scored 2026-10-09.
- The cut is largest near the trained step (54% at $N = 3$) and shrinks at fine steps (7% at $N = 8$). By the gate's own classification SR2 is still a map at the trained $\Delta t$, not a flow.
- Inertia nearly stopped mattering: Gate 1 fell from +34% to +3.9%.
- Under the exact flow the stiff modes stiffened: the median $\omega \Delta t$ rose from 4.32 to 7.55 (8.93 at layer 0).

**Against the §5.9 table** (this note's reading; the protocol scored E1 and E2 rather than this table): Gate 3 +50% meets the $\le +100\%$ line and PPL +1.0% is inside $\pm 3\%$. Gate 1 left its predicted band, downwards.

**Where the remaining penalty lives** (the two blocks after `p-sr2`, 2026-10-09). A CPU diagnostic, not pre-registered, re-ran Gate 3 with three switches applied once per trained layer instead of at every substep: one LayerNorm projection per layer, the low-rank quadratic of the potential frozen at the layer's start, and the context frozen at the layer's start. With all three on, SR2's penalty shrinks as steps get finer (0.331, 0.269, 0.143 nats at $N = 3, 4, 8$). No single switch does it. FLOW-R then replicated this per-layer flow on fresh batches, pre-registered the same day:

| call | prediction | called | result |
| --- | --- | --- | --- |
| R1 | the per-layer flow converges by N = 12–16 | 80% | HIT: 0.0001 and 0.0012 |
| R2 | its limit is within 0.25 nats of the trained step | 80% | HIT: 0.104 |
| R3 | the gap replicates FLOW-C's 0.129 within 0.05 | 70% | HIT |
| R4 | standard refinement still diverges, pen(8) ≥ 1.5 | 90% | HIT: 2.004 |
| R5 | the explicit kick carries at least 10% of the step | 90% | HIT: 0.369 and 0.505 |

So SR2's layer step is a coarse sample of a convergent per-layer flow, 0.104 nats from the trained step. The standard refinement, which recomputes the context and projects at every substep, still fails. This became the inference mode `substeps_per_layer`.

**At four times the data** ([protocol, `p-sr2-2b-result`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr2-2b-result), 2026-11-15). RT3 predicted that SR2 trained on 2.13B tokens would read Gate 3 at $N = 3$ at or below 0.30 nats. **MISS:** 0.772 nats (0.405 at 0.53B). Its per-layer flow converges by the letter, but to 0.920 nats (0.104 at 0.53B). Settled PPL fell from 58.32 to 48.58.

## 4. SR3: critical damping of the stiff modes (SR3-crit)

[protocol, `p-sr3-crit`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr3-crit). SR3 was run as SR3-crit on SR2's settings: each stiff mode gets friction $\gamma_i = \gamma_0 + 2\zeta\omega_i$ with $\zeta = 1$, which removes its oscillation but keeps the velocity between layers. By then the question had shifted to the first-versus-second-order comparison, so its calls (SC1 to SC5) were re-registered 2026-10-20 and amended the same day after the first-order twin FO-b-OWT fell behind SR2. They are scored against SR2 and FO-b, not against the §5.9 table.

| reading | SR2 | SR3-crit | FO-b |
| --- | ---: | ---: | ---: |
| settled PPL | 58.32 | 63.28 | 61.16 |
| Gate 1, velocity reset | +3.9% | +3.6% | 0 |
| Gate 2, one extra layer | +35% | +132% | +0.6% |
| Gate 3, N = 3 (nats) | 0.405 | 0.447 | 0.178 |
| ω·Δt median | 7.55 | 2.06 | 0.74 |
| per-layer flow at N = 16 (nats) | 0.104, converged | 0.300, drifting | 0.904, not converging |

- **SC3′** (3% or more behind SR2, toward FO-b; 25%): **HIT**. SC1′ and SC2′: MISS. The superseded SC1 and SC3: MISS.
- **SC4** (Gate 1 at most SR2's +3.9%; 60%): **HIT, narrowly** (+3.6%).
- **SC5** (the per-layer flow converges within 0.15 nats; 60%): **MISS**.
- **Reading, as recorded.** Critical damping is worse than removing inertia altogether. What the second-order model gains from is the oscillation of its stiff modes, not inertia as such.

**Against the §5.9 table** (this note's reading, not a scored call): Gate 1 meets $\le +25\%$. Gate 2 (+132%) misses $\le +50\%$. PPL is +9.6% against F3.1's 57.76, outside $+5\%$. A follow-up, SR3-crit on the well base PMX (call MS3), is registered; the protocol records no result as of 2026-11-17.

**The learned damping arm** ([protocol, `p-gamma-h`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-gamma-h), 2026-10-21). The same axis, with $\zeta(h)$ learned per token by a small head starting at $\zeta = 0.005$. The model chose moderate damping (median $\zeta$ 0.56 at layer 0, 0.36 at layer 1) and settled at 59.29, +1.7% against SR2. GH7 (within 2% of SR2) HIT; GH6, GH8 and GH9 MISS. Its Gate 3 at $N = 3$ is the worst of the second-order arms, 0.747 nats. A pairing caveat applies: the head shifts the initial weights, so the difference contains an unmeasured seed spread.

## 5. SR4a: variable-step training

[protocol, `p-sr4a-code`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr4a-code). SR4a moved to the code corpus after FLOW-R on code showed every model's whole stack failing refinement by 1.4 to 3 nats while its per-layer flow settles. Its base is SO-CODE, the second-order conservative model on code ($L = 2$, already with SR2's exact flow). Each optimizer step draws $N$ uniformly from $\lbrace 2, 3, 4 \rbrace$ at $T = 8$. The calls VS1 to VS5 were **proposed 2026-11-01 and not confirmed by the author before launch**; they are scored as proposed. VS1 and VS3 carry the §5.9 thresholds over unchanged.

Result, 2026-11-04 ([protocol, `p-sr4a-code-result`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr4a-code-result)). Gate 3 penalty in nats, $\ln(\mathrm{PPL_N} / \mathrm{PPL_2})$:

| N | 2 | 3 | 4 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SR4a-CODE | 0 | −0.039 | −0.027 | +0.450 | +1.394 |
| SO-CODE | 0 | +0.639 | +1.418 | +2.311 | +2.727 |

| call | prediction | measured | verdict |
| --- | --- | --- | --- |
| VS1 (70%) | Gate 3 at N = 3 and 4 both ≤ 0.095 nats | −0.039 and −0.027 | hit |
| VS2 (65%) | settled PPL at N = L ≤ 6.295 | 6.14 (+2.5%) | hit |
| VS3 (70%) | Gate 1 ≥ +3.2% (inertia kept) | +8.1% | hit |
| VS4 (40%) | Gate 3 at N = 8 ≤ 0.50 nats | 1.394 | miss |
| VS5 (45%) | per-layer flow limit ≤ 0.20 nats | 0.840 (SO-CODE 0.369) | miss |

- Inside the trained range the model is better at $N = 3$ and 4 than at $N = 2$. Beyond it, refinement fails, and the $N = 8$ penalty grew with training (0.43 at step 2,000, 1.39 at 18,000).
- Inertia is worth twice as much as in SO-CODE (Gate 1 +8.1% against +4.0%), while the stack refines inside its band. This is the break in the Gate 1 and Gate 3 ordering that the common-cause check predicted for SR4a.
- Variable steps moved each layer's step further from its own continuous limit (0.840 against 0.369 nats). The model learned three maps, not a flow that refines further.

**At eight layers** ([protocol, `p-k2-l8-r6-vs-result`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-k2-l8-r6-vs-result), 2026-11-08). The same design on an $L = 8$ code model, K2-L8-R6-VS-CODE, trained at $N$ in $\lbrace 8, 12, 16 \rbrace$. Calls RV-1 to RV-6 were proposed and not re-confirmed by the author. The outcome was the opposite of SR4a-CODE beyond the band. The model refines at every $N$ from 8 to 16, trained or not (largest penalty +0.006 nats), and nearly to $N = 32$ (+0.038 at 24, +0.081 at 32). It costs +0.020 nats at $N = 8$ against its fixed-step parent. Its per-layer flow sits 0.034 nats from the trained step, closer than the parent's 0.059. RV-1, RV-2, RV-3 and RV-6 hit. RV-4 and RV-5 miss, both in the favourable direction. One seed each, and the two models differ in depth and wells, so the protocol does not attribute the difference.

## 6. The SR-π side tests (2026-10-04 to 2026-10-05)

[protocol, `p-sr`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr) also holds four evaluation-only tests on the Fock arms, which tried to explain their much larger refinement failure before any GPU run. Each named a single cause. The three that ran each missed.

| test | single cause tested | verdict |
| --- | --- | --- |
| SR-π.3 | stiff modes crossing θ = π under refinement | MISS: ratio 1.21 against ≥ 2 predicted |
| SR-π.3b | register bookkeeping repeated per substep | MISS: Gate 3 +994% against ≤ +300% predicted |
| SR-π.4 | the LayerNorm projection rewriting large jumps | MISS: rank orders do not match |
| SR-π.2 | SR2 on the Fock arm G2 | withdrawn untested (author, 2026-10-05) |
| SR-π.1 | an L = 8 Fock arm trained below π | pending; the call fell from 60% to about 30% |

## 7. What the series established, and what is open

**Established (one seed each, about 0.5B tokens unless stated):**

- The split step's stiff-mode phase error is real and removable. SR2 halves the penalty at $N = 3$ at no perplexity cost. Later conservative arms (SR3-crit, SO-CODE, SR2-2B) build on it.
- What remains at fine steps is not integration error on the stiff modes. It comes from recomputing the context and projecting at every substep. Defined per layer, SR2's step is a coarse sample of a convergent flow (FLOW-R, 0.104 nats).
- Damping the stiff modes critically costs perplexity (+8.5%) and depth extension, and does not buy refinement. The oscillation does useful work.
- Variable-step training makes a model work at the step counts it was trained on, and keeps its inertia. At $L = 2$ it did not extrapolate; at $L = 8$ it did.

**Open:**

- SR1, SR4b, SR5a and SR5b have not run. The decision rule's SR5 branches and SR5c are therefore unscored.
- The §5.9 decision rule was never formally applied as a whole. The arms were re-aimed as the programme moved, and SR3 and SR4a were scored on later calls.
- RT3 missed: at 2.13B tokens SR2 refines worse (0.772 nats at $N = 3$), and its per-layer flow limit rose to 0.920 nats. Whether refinement improves or degrades with training is unresolved.
- Why variable-step training extrapolates at $L = 8$ but not at $L = 2$.
- SR3-crit on PMX (MS3) and SR-π.1 have no result yet.
