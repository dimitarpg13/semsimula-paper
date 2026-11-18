# Depth on code: L4-CODE, L8-CODE and the two-well eight-layer arms

**What this note is.** A short, readable summary of the protocol entries on depth on the code corpus, written 2026-11-17. It covers L4-CODE, the rule that decided whether L8-CODE would run, L8-CODE itself, and three eight-layer arms with two wells: K2-L8-CODE, its rank-6 version K2-L8-R6-CODE, and the variable-step version K2-L8-R6-VS-CODE. Every number, verdict and date is copied from the [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md), which remains the full log. Each heading below names the protocol anchor it summarises, so a reader of a model card can match the two. Two related topics have their own notes. Variable-step training at $L = 2$ (SR4a-CODE) is in [SR_Series_Settling_and_Refinement.md](SR_Series_Settling_and_Refinement.md). The memory switch that the two-well eight-layer arms train with is in [VPhi_Force_Checkpointing.md](VPhi_Force_Checkpointing.md).

**Status in one line.** All six runs are done. Depth at fixed integration time keeps paying on code, with sharply diminishing returns. Training at several step counts makes an eight-layer model refine at every step count from 8 to 16.

---

## Contents

1. [Terms used](#1-terms-used)
2. [The depth axis at a glance](#2-the-depth-axis-at-a-glance)
3. [L4-CODE](#3-l4-code)
4. [The rule for L8-CODE](#4-the-rule-for-l8-code)
5. [L8-CODE](#5-l8-code)
6. [K2-L8-CODE: two wells at eight layers](#6-k2-l8-code-two-wells-at-eight-layers)
7. [K2-L8-R6-CODE: rank 6](#7-k2-l8-r6-code-rank-6)
8. [K2-L8-R6-VS-CODE: variable step](#8-k2-l8-r6-vs-code-variable-step)
9. [Open points](#9-open-points)

---

## 1. Terms used

- **Depth at fixed time.** Each model integrates over a total time $T = 8$ in $L$ layer steps of $\Delta t = T / L$. SO-CODE has $L = 2$ and $\Delta t = 4$; L4-CODE has $\Delta t = 2$; L8-CODE has $\Delta t = 1$. Every arm starts from SO-CODE, the second-order conservative model on Python code (see [Code_Set_First_vs_Second_Order.md](Code_Set_First_vs_Second_Order.md)).
- **Settled perplexity (PPL).** The mean over the evaluations at steps 18,000, 18,500 and 19,000; lower is better. GPT-2-CODE, the matched attention model ($L = 8$), settled at 4.96 (4.959). **Share of the gap closed** is measured in log perplexity, against SO-CODE's gap to GPT-2-CODE.
- **Paired by step.** Every arm trains from step 0 on the same batches, so evaluations at the same step compare directly. Paired gaps are in nats; 0.01 nats is about 1% in perplexity.
- **Gates** (Cell 6b-7). **Gate 1** resets the velocity between layers, which measures how much the model leans on inertia. **Gate 2** adds layers at the trained $\Delta t$. **Gate 3** walks $N$ steps of $T / N$ at fixed $T$. Its **penalty** is the loss at $N$ minus the loss at the trained $N = L$, in nats. **MAPS** means the loss has its minimum at the trained step and rises on both sides: the layers are maps tied to their step, not samples of a continuous flow.
- **FLOW-R** is a CPU check (`refinement_flow_confirmation.py`). It splits each trained layer into $k$ substeps and records the penalty. If the penalty stops changing as $k$ grows, the per-layer flow has settled, and its value is the **limit**: how far the trained step sits from its own continuous flow. The **whole stack** reading is the ordinary refinement of all layers together.
- **Wells and rank.** $V_\theta$ is one bank of Gaussian wells shared by all layers. SO-CODE has 8 wells (`dcvt5x8`); "K2" means 2 wells (`V_THETA_WELLS_PER_HEAD = 2`, tag `dcvt5x2`). Each well's curvature has rank 4 by default; "R6" means rank 6 (`ANISO_RANK = 6`, tag `ar6`).
- $\omega \Delta t$ (Cell 6b-13) is how far a stiff mode turns in one layer step; past 2 is the explicit integrator's wall.

## 2. The depth axis at a glance

| arm | L | wells, rank | settled PPL | share of SO-CODE's gap closed | Gate 1 cost | FLOW-R limit (nats) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| SO-CODE | 2 | 8, 4 | 5.99 | — | +4.0% | 0.369 |
| L4-CODE | 4 | 8, 4 | 5.48 | 47% | +76% | 0.306 |
| L8-CODE | 8 | 8, 4 | 5.35 | 60% | +172% | 0.035 |
| K2-L8-CODE | 8 | 2, 4 | 5.667 | 30% | factor 3.71 | 0.098 |
| K2-L8-R6-CODE | 8 | 2, 6 | 5.543 | 41% | factor 2.28 | 0.059 |
| K2-L8-R6-VS-CODE | 8 | 2, 6 | 5.651 | 31% | factor 2.96 | 0.034 |
| GPT-2-CODE | 8 | attention | 4.96 | | | |

The FLOW-R limit is the penalty at $k = 8$ substeps per layer. Gate 1 is given as the protocol gives it: a percentage rise for the first three, the ratio of reset to carried perplexity for the last three.

## 3. L4-CODE

### 3.1 Launch (protocol `p-l4-code-launch`)

Launched 2026-10-31/11-01 on a Colab A100 80 GB. The calls L4a and L4b were registered on 2026-10-27 and are binding; the L8 rule is the one in §4. Cell 0 is SO-CODE's nine changes plus `LADDER_L = 4` ($\Delta t = 2$, $T = 8$), early snapshots and `EVAL_LAYER_CHECKPOINT = True` (12 changes).

- **The first attempt ran out of memory** at the step-500 evaluation: training peaks at 60.5 GB, and evaluation needs about 15 GB more. The fix (`7da8776`) is the switch `EVAL_LAYER_CHECKPOINT`, which turns layer checkpointing on during evaluation only. Its loss is bit-identical. The run restarted from step 0, so the pairing with the other arms is kept.
- **The second attempt** ran at 2.0 s per step, about 10.9 h. At step 500 the perplexity was 46.79 (SO-CODE 44.02).

| id | prediction | called |
| --- | --- | ---: |
| L4a | depth helps on code at least as much as on OpenWebText: L4-CODE settles more than 5% below SO-CODE (below 5.70) | 60% |
| L4b | its gain is long-range: in the copy split, the gap on targets last seen 65–512 tokens back shrinks by a larger fraction than on those last seen 1–16 back | 55% |

### 3.2 Result (protocol `p-l4-code-result`)

2026-11-01. 19,250 steps in 10.9 h, one session, clean (no watchdog trigger, leak monitor CLEAN).

| reading | L4-CODE | SO-CODE |
| --- | --- | --- |
| settled PPL (evals 5.37, 5.50, 5.59) | 5.48 | 5.99 |
| against SO-CODE / FOCK-CODE / GPT-2-CODE | −8.5% / −3.0% / +10.6% | |
| paired gap to SO-CODE, steps 16,000–19,000 | −0.088 nats (SD 0.004) | |
| Gate 1, velocity reset | +76% (4.66 → 8.21) | +4.0% |
| Gate 3 penalty at half the trained step (N = 8) | 1.132 nats | 1.42 (N = 4) |
| ω·Δt median; past 2 | 2.20; 53.6% | 7.55; 99.97% |

- **Gate 3 is MAPS,** as for every second-order arm, but flatter near the trained step than SO-CODE's.
- **No dead wells at any snapshot.** The deep layers (1 to 3) run on the same two wells. Layer 0 uses all eight, and removing them costs +4,127%.
- **Retrieval is repeated, not sharper.** Each deep layer has one long channel with SO-CODE's per-layer reach.
- **The copy split:** depth closes 47.4% of SO-CODE's gap overall, against 30.4% for the register path (FOCK-CODE). It closes about half in every group, first occurrences as much as induction targets. Far against near: 48.1% against 45.2%, +2.9 points (95% interval $-1.8$ to $+7.9$).

| call | result | verdict |
| --- | --- | --- |
| L4a | 5.48, −8.5% (OpenWebText, with the register path: −5.7%) | HIT |
| L4b | +2.9 points (−1.8 to +7.9) | HIT on the letter, not distinguishable from no difference |

**Reading.** On code, depth at fixed integration time does what the register path does, but more. The gain is spread over every kind of target, and the gap keeps its shape. The protocol reads this as fitting Conjecture 1 of the Dead Wells note (readout, not retrieval). It is an interpretation, with one seed per model. Published 2026-11-01 at the author's request as `semsimula-ladder-live-code-d384-l4-none-norc-sr2`, in the "Gen 3: depth on code" collection.

## 4. The rule for L8-CODE

### 4.1 The rule (protocol `p-l8-code-rule`)

Fixed by the author on 2026-10-31, before L4-CODE trained. L8-CODE is SO-CODE's Cell 0 plus `LADDER_L = 8`, which matches GPT-2-CODE's depth. It runs only on L4-CODE's result:

| L4-CODE settles | L8-CODE |
| --- | --- |
| more than 2% below SO-CODE (below 5.88) | runs |
| within ±2% of SO-CODE (5.88–6.12) | skipped |
| more than 2% above SO-CODE | skipped |

**Amendment, also fixed by the author on 2026-10-31: a convergence trigger.** L8-CODE also runs if L4-CODE's Gate 3 shows real progress towards a flow, by either test:

- **(a)** the penalty at half the trained step ($N = 8$ at $L = 4$) is below 0.71 nats, half of SO-CODE's 1.42 at its own half step;
- **(b)** over $N$ = 4, 6, 8, 12 and 16 the successive loss changes shrink at every step, and the penalty at $N = 16$ is within 0.1 nats of that at $N = 8$.

For this, Cell 6b-7 was extended to run $N$ up to $\max(8, 4L)$.

Proposed alongside the rule and **not confirmed by the author**: early snapshots for L4-CODE, and settling the dead-wells fix before L8-CODE if L4-CODE lost any layer's wells. Later, after `p-so-code-vtun-result`, the slower $V_\theta$ was proposed as that guard; that choice is not confirmed either.

### 4.2 How the rule fired

Read in `p-l4-code-result`: the perplexity trigger fires (5.48 < 5.88), so **L8-CODE runs.** Neither convergence trigger fires: (a) the half-step penalty is 1.13 nats, above 0.71; (b) the successive changes grow from $N = 6$ on. L4b reads a general gain, so L8-CODE is read on its settled perplexity and the share of the gap closed, not on the long-distance gap.

## 5. L8-CODE

### 5.1 Setup (protocol `p-l8-code-setup`)

Set up and launched 2026-11-01. Cell 0 is SO-CODE's changes, keeping the default layer checkpointing on, plus `LADDER_L = 8` ($\Delta t = 1$) and early snapshots (10 changes). It has 76,731,865 parameters, ran at 5.6 s per step and was estimated at about 29.9 h, so more than one Colab session.

- **Resume pairing fixed (`63ce0d2`).** Before this, a resumed run repeated the batch sequence from step 0, so its evaluations after the resume were not paired with any other arm's. Checkpoints now save the batch generator's state and the resume restores it. A simulation confirmed that a restored run draws exactly the uninterrupted run's batches.
- At step 500 the perplexity was 50.93 (L4-CODE 46.79, SO-CODE 44.02). In warmup the deeper models trail, in depth order.

### 5.2 Result (protocol `p-l8-code-result`)

2026-11-02. Two sessions, 29.5 h in all, resumed at step 15,500 with the batch sampler restored. 5.5 s per step, peak memory 28.9 GB. Clean apart from the skipped updates below.

| reading | L8-CODE |
| --- | --- |
| settled PPL (evals 5.26, 5.36, 5.45) | 5.35 |
| against L4-CODE / FOCK-CODE / SO-CODE / GPT-2-CODE | −2.4% / −5.3% / −10.7% / +8.0% |
| share of SO-CODE's gap closed | 60% (L4-CODE 47%) |
| paired gap to L4-CODE, steps 16,000–19,000 | −0.024 nats (SD 0.0016) |
| Gate 1, velocity reset | +172% (4.59 → 12.49) |
| Gate 3 penalty at N = 16 (half step) / N = 12 | 0.783 / 0.416 nats |
| ω·Δt median; past 2 | 1.75; 33.8% |
| FLOW-R limit at k = 8 (L4-CODE / SO-CODE) | 0.035 (0.306 / 0.369) |

- **The 2.4% gain over L4-CODE is just outside the $\pm 2\%$ band.** L8-CODE trails both shallower models early, is level with L4-CODE by step 8,000, and leads from step 10,500.
- **Skipped updates, steps 1,400 to 3,600.** 34 of the 45 logged steps there show a non-finite gradient norm, and Cell 6 skips such steps. That is roughly 1,650 skipped updates if the logged steps are typical; the exact count was not logged. The cause was not diagnosed. So **the 2.4% gain over L4-CODE is a lower bound** on the eight-layer gain at this budget, as far as the skipped steps go.
- **The resume kept the pairing.** Evaluation changes of L8-CODE and L4-CODE correlate at 0.991 after the resume and 0.994 before it.
- **Gate 3 is MAPS.** The half-step penalty falls with depth (1.42, 1.13, 0.78 nats), but the curve still turns up past $N = 12$.
- **FLOW-R (2026-11-03):** at eight layers the per-layer flow settles 0.035 nats from the trained step. That is ten times closer than at two or four layers, and closer than SR2 on prose (0.10). The whole stack still fails (2.695 nats at $k = 4$).

**Reading.** Depth at fixed integration time keeps paying on code, with sharply diminishing returns. The loss falls by 0.090 nats from $L = 2$ to 4 and by 0.024 from 4 to 8, with some of L8-CODE's early steps skipped. Matching GPT-2-CODE's depth leaves an 8.0% gap. An interpretation; one seed per model. The entry's list of CPU jobs still to run marks the eight-layer well probe "done, below", but no such result appears in the entry. This is unclear in the log.

## 6. K2-L8-CODE: two wells at eight layers

### 6.1 Setup (protocol `p-k2-l8-code-setup`)

Set up and launched 2026-11-02. L8-CODE with two $V_\theta$ wells instead of eight, paired with L8-CODE, so $K$ is the only difference in the arithmetic. **No calls were registered.** It uses the $V_\varphi$-force checkpoint (`VPHI_FORCE_CHECKPOINT = True`) in place of layer checkpointing, with the same arithmetic. It has 50,164,435 parameters. At step 50 it ran at 3.46 s per step with a 35.0 GB peak.

**A stop rule, fixed by the author on 2026-11-02** after step 50 and before any evaluation was read. Let $d$ be the mean paired gap to L8-CODE over the evaluations at 5,000, 5,500 and 6,000. Stop if $d$ is above +0.10 nats; run to the end if $d$ is at or below +0.05; in between, re-read at step 10,000. At step 6,000, $d = -0.066$ nats, so the run continued to the end. The setup warned that this early reading still carried L8-CODE's skipped-update lag. From step 8,000 the paired gap to L8-CODE turned positive and grew.

### 6.2 Result (protocol `p-k2-l8-code-result`)

2026-11-03. One session, 19.0 h, 3.5 s per step, peak 35.0 GB in training and 51.6 GB in evaluation. **No non-finite gradient norm at any logged step**, where L8-CODE had many on the same batches.

| window (paired, nats) | against L8-CODE | against L4-CODE | against SO-CODE |
| --- | ---: | ---: | ---: |
| 2,000–4,000 | −0.340 | +0.021 | −0.272 |
| 8,000–13,000 | +0.028 | +0.029 | −0.070 |
| 16,000–19,000 | +0.056 (SD 0.0015) | +0.032 | −0.056 |

- **Settled 5.667:** 5.8% behind L8-CODE, 3.3% behind L4-CODE and 5.5% ahead of SO-CODE. It closes 30% of SO-CODE's gap.
- **At $L = 8$ the six extra wells are worth 0.056 nats,** more than doubling depth from four to eight layers (0.024). Two wells at $L = 8$ still beat eight wells at $L = 2$, with 26.4 M fewer parameters. Outside the embeddings it has 11.6 M parameters against GPT-2-CODE's 14.2 M, and trails it by 14.3%.
- **Cell 6b-4 (the wells' curvature spectrum):** participation ratio median 3.03 of 4. The Frobenius cap binds, so rank 6 would redistribute the same budget rather than add curvature. Reading it found and fixed a bug: Cells 6b-2 and 6b-4 crashed on joint-bank arms at $K = 2$ and $K = 8$ and were silently wrong at $K = 4$ (the fix was uncommitted at the time of reading).
- **Gates:** two wells refine no better than eight, and somewhat worse near the trained step. Gate 1's reset costs a factor of 3.71 (L8-CODE 2.72).
- **FLOW-R (2026-11-04):** the limit is 0.098 nats, about three times L8-CODE's 0.035. At eight layers, six fewer wells move the limit by 0.06 nats.

## 7. K2-L8-R6-CODE: rank 6

### 7.1 Setup (protocol `p-k2-l8-r6-setup`)

Decided 2026-11-04 by the author, and launched that evening. SR4a-CODE's result had selected a variable-step version by the order of 2026-11-03. The author chose this fixed-step version first, and the variable-step one followed later (§8). Against K2-L8-CODE the arm changes one thing: the wells' rank, from 4 to 6, on the same batches.

- **The switch.** `ANISO_RANK` was not in the tag, so a rank-6 run would have resumed K2-L8-CODE's checkpoint. Cell 0 now appends `ar<rank>` when the rank is not 4, and every existing tag is unchanged.
- **The parameter match.** 53,115,092 parameters; 14,517,715 outside the embeddings, +2.3% against GPT-2-CODE's 14,196,480.

Calls proposed 2026-11-04, **not confirmed by the author**:

| call | prediction | probability |
| --- | --- | ---: |
| R6-1 | rank 6 helps: paired difference to K2-L8-CODE ≤ −0.010 nats | 40% |
| R6-2 | no effect: within ±0.010 nats | 45% |
| R6-3 | near matched GPT-2: settled ≤ 5.45 | 10% |
| R6-4 | all six directions in use: participation ratio median ≥ 4.5 of 6 | 50% |
| R6-5 | stable: no non-finite gradient norm | 85% |

### 7.2 Result (protocol `p-k2-l8-r6-result`)

2026-11-05. One session, 19.5 h, 3.65 s per step; no non-finite gradient norm in 385 logged steps.

| window (paired, nats) | against K2-L8-CODE | against L8-CODE | against L4-CODE | against SO-CODE |
| --- | ---: | ---: | ---: | ---: |
| 16,000–19,000 | −0.023 (SD 0.0016) | +0.033 | +0.009 | −0.079 |

| call | result | verdict |
| --- | --- | --- |
| R6-1 | −0.023, SD 0.0016 | hit |
| R6-2 | −0.023 | miss |
| R6-3 | 5.543 (one eval, step 18,000, reads 5.44) | miss |
| R6-4 | 4.87 | hit |
| R6-5 | none in 385 logged steps | hit |

- **Settled 5.543,** 2.2% below K2-L8-CODE. It trails GPT-2-CODE by 11.8% with 2.3% more parameters outside the embeddings, and closes 41% of SO-CODE's gap.
- **Rank 6 recovers 41% of what the six extra wells are worth** at $L = 8$ (0.023 of 0.056 nats), with 2.95 M parameters against their 26.6 M.
- **The setup's reading was wrong.** The cap still binds and the budget is redistributed, not added, but the redistribution helps.
- **Refinement improves.** Finer steps cost about half of what they cost K2-L8-CODE, but the whole stack still fails: maps, not a flow.
- **FLOW-R:** the limit is 0.059 nats. Rank 6 closes 62% of the gap in the limit between two wells (0.098) and eight (0.035).

## 8. K2-L8-R6-VS-CODE: variable step

### 8.1 Setup (protocol `p-k2-l8-r6-vs-setup`)

The next GPU arm by the author's decision of 2026-11-05. It applies SR4a-CODE's design at $L = 8$ on the rank-6 model: every optimizer step draws $N$ uniformly from $\lbrace 8, 12, 16 \rbrace$ and walks $N$ steps of $T / N$. Evaluation runs at $N = 8$, so the arm pairs with K2-L8-R6-CODE step by step. The author's question: does a model trained at $N$ = 8, 12 and 16 also refine at the counts it never saw, 9 to 11 and 13 to 15? When $N$ is not a multiple of $L$, layers get uneven shares of the time, so "interpolation" here means new layer-time patterns as well as new step sizes. The tag switch now accepts multi-digit counts (`vs8-12-16`); a CPU check passed 35 of 35 tests.

Calls proposed 2026-11-05, **not re-confirmed by the author**:

| call | prediction | probability |
| --- | --- | ---: |
| RV-1 | cost at N = L: paired difference to K2-L8-R6-CODE at most +0.030 nats | 60% |
| RV-2 | refinement inside the trained set: penalty at N = 12 and 16 both ≤ 0.02 nats | 65% |
| RV-3 | interpolation: penalty at every untrained N from 9 to 15 ≤ 0.05 nats | 45% |
| RV-4 | extrapolation fails: penalty at N = 24 above 0.20 nats | 70% |
| RV-5 | the per-layer flow moves away: FLOW-R limit at k = 8 above 0.059 | 65% |
| RV-6 | stable: no non-finite gradient norm | 85% |

### 8.2 Result (protocol `p-k2-l8-r6-vs-result`)

2026-11-08. The calls are scored on the relaunch.

**The first launch failed, and found a bug.** From step 1,000 the gradient norm was NaN at almost every logged step, so about 98% of optimizer steps were skipped and nothing raised an alarm. The cause: with $K = 2$, a large gap between the two well logits underflows the softmax weight to exactly 0, and the backward of its square root gives NaN. The fix (`15c236d`) gives such a well gradient 0 with a bit-identical forward. A guard now raises an error after 25 non-finite gradient norms in a row. The relaunch (2026-11-07) resumed from the clean step-500 checkpoint, on the same batches and step-count draws.

**The run.** Two sessions, 5.5 s per step, peak 63.6 GB in training. No non-finite gradient norm and no skipped step in 385 logged steps.

| window (paired, nats, at N = 8) | against K2-L8-R6-CODE | against K2-L8-CODE | against L8-CODE |
| --- | ---: | ---: | ---: |
| 2,000–4,000 | +0.142 | +0.126 | −0.214 |
| 16,000–19,000 | +0.020 (SD 0.0014) | −0.003 | +0.053 |

| N | 4 | 6 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 24 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| penalty, this arm (nats) | +2.568 | +0.175 | +0.005 | −0.001 | −0.004 | −0.008 | +0.006 | −0.000 | +0.003 | +0.001 | +0.038 | +0.081 |
| K2-L8-R6-CODE (Cell 6b-7) | +3.68 | +0.690 | | | | +0.251 | | | | +0.499 | +1.279 | +1.726 |

| call | result | verdict |
| --- | --- | --- |
| RV-1 | +0.020, SD 0.0014 | hit |
| RV-2 | −0.008 and +0.001 | hit |
| RV-3 | largest +0.006 (N = 13) | hit |
| RV-4 | +0.038 | miss |
| RV-5 | 0.034 | miss (closer, not farther) |
| RV-6 | none in 385 logged steps of the relaunch | hit (relaunch) |

- **Settled 5.651.** The cost at $N = 8$ is +0.020 nats against the fixed-step parent. Variable-step training gives back 87% of what rank 6 had gained over rank 4.
- **It refines at every $N$ from 8 to 16,** trained or not, and nearly to $N = 32$. Every earlier eight-layer code model failed Gate 3 at every $N$ above 8.
- **Extrapolation held,** against the setup's expectation, so RV-4 misses in the favourable direction. SR4a-CODE at $L = 2$ failed beyond its band at the same ratio of substeps per layer. The two models differ in depth and wells, and there is one seed each, so the difference is not attributed to either.
- **Below the trained $N$ it fails as before** ($N = 4$ costs 2.57 nats).
- **FLOW-R:** the limit is 0.034 nats, level with eight wells (0.035), so RV-5 misses in the favourable direction. The whole stack at $k = 4$ costs 0.085 nats, against 1.77 for the parent.
- Published 2026-11-08 at the author's request as `dimitarpg13/semsimula-ladder-live-code-d384-l8-none-norc-sr2-k2-r6-vs8-12-16`, in the depth-on-code collection.

## 9. Open points

- **The cause of L8-CODE's non-finite gradients** over steps 1,400 to 3,600 was not diagnosed in `p-l8-code-result`. The later NaN bug found in the variable-step arm occurred at $K = 2$, while L8-CODE has eight wells; the protocol does not link the two.
- **Why variable-step training sharpened the well selection** six-fold in 500 steps, which triggered that bug, is open.
- **Still to run on the eight-layer arms**, per the entries: Cell 8, the well probe on all eight layers, `xi_reach.py` and the copy split. For K2-L8-R6-VS-CODE there is also the final model's well-logit gap.
- **The proposed dead-wells guard** for deeper arms (the slower $V_\theta$) is not confirmed by the author; see [Code_Dead_Wells_and_Screens.md](Code_Dead_Wells_and_Screens.md).
- **One seed per model throughout.**
