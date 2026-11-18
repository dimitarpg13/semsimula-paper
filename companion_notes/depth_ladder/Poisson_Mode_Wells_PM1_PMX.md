# §5.15 Poisson-mode wells: PM1, PM1-cap and PMX

**What this note is.** A short, readable summary of §5.15 of the depth-ladder protocol, "PM1: bosonic Poisson-mode registers", written 2026-11-17. It covers the three arms of that section (PM1, PM1-cap and PMX), what was pre-registered for each before it ran, how each was scored, and the calls. Every number, verdict and date is copied from the [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md), which keeps the full running log and remains the record of truth. Two scoring blocks there have anchors: the PM1 full run, [protocol, `p-pm1`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-pm1), and the PMX full run, [protocol, `p-pmx`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-pmx). The mechanism and its proofs are in [Poisson_Mode_Registers_PM1.md](../Poisson_Mode_Registers_PM1.md) and [Exact_Integration_of_Poisson_Mode_Wells_PMX.md](../Exact_Integration_of_Poisson_Mode_Wells_PMX.md).

**Status in one line.** PM1 and PMX ran to 32,500 steps and are scored. PM1-cap was stopped by its probe gate at step 8,000. PMX is the best conservative model on the ladder by perplexity (settled 54.63), but it missed the per-layer-flow calls CX3 and CX3′, so SR2 remains the refinement-ready arm.

## Contents

1. [Terms used below](#1-terms-used-below)
2. [The mechanism](#2-the-mechanism)
3. [The arms at a glance](#3-the-arms-at-a-glance)
4. [PM1](#4-pm1)
5. [PM1-cap](#5-pm1-cap)
6. [PMX: the wells integrated exactly](#6-pmx-the-wells-integrated-exactly)
7. [What remains open](#7-what-remains-open)

## 1. Terms used below

- **The comparison arms.** F3.1 is the two-layer ($L = 2$) conservative-only configuration (no reverse channel; the pair potential $V_\phi$ and the context field $\xi$ trained with live gradients), settled at 57.76. G2 is F3.1 plus slot registers read through the Fock reverse channel, settled at 53.12. SR2 is F3.1 with the stiff part of the single-token potential $V_\theta$ integrated by an exact damped flow (§5.19), settled at 58.32.
- **Settled PPL** is the mean validation perplexity of the last three evaluations.
- **A probe** stops training at step 8,000 and scores it against a gate fixed in advance. A passing probe continues from its own checkpoint, so the stop costs nothing.
- **The refinement gates** (Cell 6b-7). Gate 1 resets the velocity entering each layer. Gate 2 adds steps at the trained step size. Gate 3 takes $N$ steps over the same total time, so each step is smaller. The trained model is $N = 2$. A penalty is given in percent of PPL, or in nats as $\ln(\mathrm{PPL_N} / \mathrm{PPL_2})$.
- **The per-layer flow** is the refinement mode in which the context and the LayerNorm projection are held once per layer (FLOW-R, §5.19). It converges by FLOW-R's criterion if the penalties at 16 and 12 steps differ by at most 0.01 nats and those at 12 and 8 by at most 0.02.
- **The kick share** is the part of a layer step applied as an explicit kick rather than inside the exact flow.

## 2. The mechanism

**Why it was built.** DP1–DP3 (§5.14) showed that the slot registers are exclusion objects, with constant number and salience acting as retention. So the book's bosonic, Poisson-mean Doi–Peliti v2 describes no trained model. PM1 is a register mechanism that has both properties by construction, to find out whether the bosonic v2 can be trained and what it is worth.

**How it works.** There are $K = 64$ shared mode prototypes $\mu_v$, one per mode $v$.

- **Creation.** Token $s$ creates particles in mode $v$ at a rate given by its overlap with the mode, $E_v(s) = \exp(-\kappa_v^2 \lVert h_s - \mu_v \rVert^2)$, where $h_s$ is the token's hidden state.
- **Survival.** Each particle survives to the next token with probability $\lambda_v$. Half-lives start log-spaced from 4 to 128 tokens and are learnable.
- **Occupation.** The occupation $\phi_v(t)$ is the sum over earlier tokens $s \lt t$ of $\lambda_v^{t-1-s} E_v(s)$. This is exactly the Poisson mean of an immigration–death process, which is unbounded and bosonic: shared modes, no cap on occupation.
- **Force.** The force is minus the gradient of a sum of Gaussian wells centred on the prototypes, each with depth $a$ (one per layer and mode) weighted by the occupation. It is linear in the occupation, so carrying only the mean is exact.
- **Conservativity.** The occupation uses only the strict past, so it is constant in the current state and the force is a gradient. The depths start at 0, so the force is exactly zero at step 0.

**Recorded in advance.** With a coupling linear in the occupation, no measurement can show bosonic fluctuations. What is bosonic is the structure. A sampled variant, drawing particle numbers from a Poisson law in training, is the follow-up that would make the statistics matter.

**Verified before the run** (2026-10-05): off, bit-identical to HEAD on G2, F3.1 and G3′; on, the force equals minus the gradient (relative error 2.4e-7), a Monte Carlo simulation reproduces the occupation within 0.5% with variance/mean 1.001, and causality is exact.

## 3. The arms at a glance

| arm | what changes | base | outcome | date scored |
| --- | --- | --- | --- | --- |
| PM1 | 64 Poisson modes, explicit wells, pm_ clip 0.3 | F3.1 | settled 55.17 | 2026-10-08 |
| PM1, clip 1.0 | the same with the pm_ clip at 1.0 | F3.1 | probe 81.67; stopped by the rule | 2026-10-07 |
| PM1-cap, first form | well depths capped at 0.3 | F3.1 | superseded before running | — |
| PM1-cap on SR2 | cap 0.3 on the SR2 integrator | SR2 | probe 92.89; the gate stops the arm | 2026-10-10 |
| PMX (pmx16) | the wells integrated exactly in the SR2 flow, no cap | SR2 | settled 54.63 | 2026-10-11 |

## 4. PM1

**The probe gate.** It was revised on 2026-10-06, before any PM1 data existed, from 3,000 steps to 8,000. At 3,000 steps no memory mechanism has yet shown itself: G2's lead over F3.1 is 3.2% at step 3,000 and 11.4% at step 8,000. The gate at step 8,000 was 90.8 or lower (at least 3% better than F3.1's 93.57), with pm_ clip hits under 5% of logged steps.

**Probe scored, 2026-10-06: 79.78, perplexity PASS, clip MISS on the letter.** PM1 was below G2 at every evaluation from step 3,000 on, which no prediction anticipated. The pm_ group's pre-clip norm was 0.4 or more on 77% of logged steps, a steady throttle with no divergence. The author chose a fresh arm with the clip at 1.0, with its own probe.

**Clip 1.0, scored 2026-10-07: 81.67, so the 0.3 arm continues.** PC1 (clip hits under 5%) HIT; PC2 (81.4 or lower) MISS by 0.27 PPL; PC3 (78.2 or lower) MISS. Clip 1.0 trailed at every evaluation from step 2,000, by 1.4–4.4%. The reading: under AdamW a constant rescaling cancels in the update, so the 0.3 clip was not a learning-rate cut; it made every step's pm_ gradient count equally. Cell 6b-15 (tuning diagnostics) then flagged no knob on either checkpoint. Both showed the same picture: PM1 acts at layer 1, with attractive wells whose force is 5–6 times the conservative force, and has switched itself off at layer 0.

**Full run scored, 2026-10-08** ([protocol, `p-pm1`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-pm1)):

| | PM1 | F3.1 | G2 | matched GPT-2 |
| --- | ---: | ---: | ---: | ---: |
| settled PPL | 55.17 | 57.76 | 53.12 | 49.81 |
| best | 53.67 (step 30,500) | 57.35 | 51.27 | — |
| PPL at step 21,000 | 67.29 | 71.81 | 63.61 | — |
| ω·Δt median at the end | 2.84 | 4.32 | 3.40 | — |

| call | prediction | called | result |
| --- | --- | ---: | --- |
| — | settled at least 1% better than F3.1 | 50% | HIT (−4.5%) |
| — | settled between G2 and F3.1 | 45% | HIT |
| PC4 | settled below G2's 53.12 | 50% | MISS (+3.9%) |
| RR-PM1 | Gate 3 at N = 3 at or below F3.1's +143% | 60% | MISS (+149%) |
| RR-PM2 | Gate 3 at N = 3 below +600% | 85% | HIT |
| — | most trained depths positive | 60% | HIT on the letter (57%) |
| — | repetition: Spearman above 0.5 | 55% | MISS (+0.086) |
| — | DP3 on the modes: Spearman above 0.5 | 80% | HIT (+0.815) |

By the decision rule, the bosonic Doi–Peliti v2 is trainable and worth something. PM1 was the best conservative model at $L = 2$ at the time: it recovers 56% of the F3.1 to G2 gap (2.59 of 4.64 PPL). The modes track regions of semantic space the recent context visited, not token identity. The step is a gradient flow on the trained weights, cleaner than F3.1's, and causal. A small prefix-only discrepancy at positions 3–15 was traced to the top-k router term of $V_\phi$, not a leak.

**Refinement is the weak point.** The N = 3 point hides the shape:

| | F3.1 | G2 | PM1 |
| --- | ---: | ---: | ---: |
| Gate 1, velocity reset | +34% | +33% | +10,600% |
| Gate 2, N = 3 | +92% | +51% | +951% |
| Gate 3, N = 3 | +143% | +1,274% | +149% |
| Gate 3, N = 4 | +374% | +611% | +16,700% |
| Gate 3, N = 8 | +873% | +1,565% | +3,700% |

A CPU diagnostic (not pre-registered) scaled the trained well depths by a factor $\alpha$. With the wells at half strength, Gate 3 is +85% at N = 3 and +133% at N = 4, both below F3.1's. With the wells off, PPL rises to 122.29: the rest of the model co-adapted to the wells. So the layer-1 wells cause both the refinement failure and the momentum dependence.

## 5. PM1-cap

**Mechanism.** Each well's effective depth is $c \tanh(a_{\mathrm{raw}} / c)$, so its size stays below $c$, and the force stays an exact gradient. The cap was $c = 0.3$, which leaves the median layer-1 well almost intact (0.24 to 0.20) and bounds the tails. A post-hoc preview on PM1's weights removed the momentum lock (Gate 1 at $-19\%$) at a 65% perplexity cost (88.53).

**First registration superseded** (2026-10-09, before the run, author's decision). Its calls CAP0–CAP6 were never scored. FLOW-C and FLOW-R had shown that the per-layer flow converges on SR2 but not on F3.1's split step, so a capped arm on F3.1 could not become refinement-ready. The arm moved to the SR2 base, and its memory gain is read against SR2.

**Probe scored, 2026-10-10: 92.89, the gate stops the arm.** The gate was 90.1 or lower against SR2's 92.93. The capped arm tracked SR2 within 1% at every evaluation. CS0 (84.0 or lower) MISS; CS1–CS6 not scored, since no full run happened.

| at step 8,000 | PM1-cap on SR2 | PM1 |
| --- | ---: | ---: |
| layer-1 PM force / conservative force | 3.13 | 6.29 |
| layer-1 force share, short / long half-life quartile | 10% / 57% | 34% / 11% |
| κ²·d p50 / p95 | 1.15 / 3.03 | 0.80 / 1.13 |
| loss direction on pm_depth | −0.03 | +0.42 (outward) |

**Reading.** The cap binds, the model compensates (force moved to long-lived, sharp, shallow wells), and the force it rebuilds buys nothing. PM1's perplexity gain and its refinement failure come from the same deep layer-1 wells. Of three options, the author chose (b) on 2026-10-10: integrate the wells exactly.

## 6. PMX: the wells integrated exactly

**Design.** A check on PM1's weights found the wells' curvature stiff (median $\omega \Delta t$ 2.10 for its isotropic part) and well described by a quadratic (4.7% of the force missed). PMX takes that quadratic at the start of each step and folds it into SR2's exact low-rank flow, which can handle negative curvature too. The kick keeps only the remainder. The forward value of every force is unchanged; only the integration differs. There is no depth cap.

**Amendment, `pmx16`** (2026-10-10, before any data). The first launch ran above 26 s per step and was stopped. cuSOLVER's batched eigensolvers handle matrices only up to 32 by 32, and PMX's were 80 by 80. The fix keeps every well's force and isotropic curvature in the flow, plus the rank part of each token's 16 strongest wells (32 wide in all). That misses 13.2% of the force at the kick point. On PM1's weights the layer-1 kick share fell to 0.119 (explicit wells 0.938). The probe launched at 2.92 s per step.

**Probe scored, 2026-10-11: 79.43, the gate passes** (90.1 or lower). PMX tracked PM1 within 1.2% at every evaluation. Integrating the deep wells exactly kept the memory gain that the cap erased. At the probe, Gate 1 was +30,600%.

**Full run scored, 2026-10-11** ([protocol, `p-pmx`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-pmx)), best checkpoint at step 30,500:

| call | prediction | called | result |
| --- | --- | ---: | --- |
| CX0 | step 8,000 at or below 84.0 | 50% | HIT (79.43) |
| CX1 | settled at or below 57.74 (1% better than SR2) | 55% | HIT (54.63) |
| CX1′ | settled at or below PM1's 55.17 | 30% | HIT (54.63) |
| CX2 | Gate 3 at N = 3 at or below SR2's 0.405 nats | 40% | MISS (1.097) |
| CX3 | the per-layer flow converges (FLOW-R criterion) | 50% | MISS on the letter (0.029 against 0.02) |
| CX3′ | its limit within 0.25 nats of the trained step | 40% | MISS (0.387) |
| CX4 | Gate 1 at or below +100% | 60% | HIT (+91%) |
| CX5 | the conservativity test passes | 90% | HIT |
| CX6 | layer-1 kick share at most 0.3 | 75% | HIT (0.288) |
| RT1 | Gate 3 at N = 3 falls from the probe's 0.584 | — | MISS (rises to 1.097) |

Against the other $L = 2$ arms, PMX settles 6.3% below SR2, 5.4% below F3.1 and 1.0% below PM1, and 2.8% above G2. The per-layer flow penalties, in nats:

| steps N | 3 | 4 | 6 | 8 | 12 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PMX, per-layer flow | 1.711 | 0.841 | 0.292 | 0.415 | 0.386 | 0.387 |
| SR2, per-layer flow | 0.286 | 0.206 | 0.084 | 0.103 | 0.104 | 0.104 |

The flow does reach a limit between 12 and 16 steps, more slowly than SR2's, which had settled by 8. Gate 1 fell from +30,600% at the probe to +91%: the momentum lock was a mid-training state. Causality is CLEAN.

**Decision rule** (pre-registered, unchanged). CX1 hits and CX3 misses, so "the gain survives but the flow does not converge". The first branch (an HF card and the arm named in the book) is not met. SR2 remains the refinement-ready arm; PMX is the best conservative model by perplexity. **Scope:** PMX is trained on 532.5M tokens, a third of compute-optimal, so this is a statement about PMX at this budget, not about exactly integrated wells.

## 7. What remains open

- **The clip-1.0 continuation.** After PC4 missed, a queued conditional sent the clip-1.0 arm on from its step-8,000 checkpoint, before the seed pair. §5.15 does not record its result.
- **SR3 on PMX.** Its old rationale (the momentum lock) is gone. It is re-aimed at settling, to be re-registered after CX3.
- **Candidates, not scheduled:** a clip group of its own for pm_depth; an initial half-life range of 4–32 tokens; removing the layer-0 depths; PMX at layer 1 only; 8 wells in the eigensolve; the sampled Poisson variant.
- **Related:** the removal costs of the wells and $V_\phi$ (§5.20, [Coupling_Removal_Costs_CD.md](Coupling_Removal_Costs_CD.md)) withdrew PM1's reading that "the wells have taken over $V_\phi$'s role".
