# The two-billion-token runs: SR2-2B, the decay gain and well occupancy

**What this note is.** A short, readable summary of the 2B-token series of the depth ladder, written 2026-11-17. It covers the plan and its calls (the 2B-token block of §5.13, pre-registered 2026-10-22), the first 2B run, SR2-2B ([protocol, `p-sr2-2b-setup`](../Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr2-2b-setup) and [protocol, `p-sr2-2b-result`](../Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr2-2b-result)), the decay-gain calls DG1 and DG2 and the decay branches DB1 to DB3 ([protocol, `p-decay-branch-setup`](../Depth_Ladder_and_Matched_Baseline_Protocol.md#p-decay-branch-setup)), and the well-occupancy calls OC1 to OC3 ([protocol, `p-well-occupancy-setup`](../Depth_Ladder_and_Matched_Baseline_Protocol.md#p-well-occupancy-setup)). Every number, verdict and date is copied from the [protocol](../Depth_Ladder_and_Matched_Baseline_Protocol.md), which keeps the full log and remains the record of truth. The matched GPT-2 at 2B tokens (G2B) appears only as the comparison number.

**Status in one line.** SR2-2B has run and is scored: RT3 MISS, DG1 refuted, DG2 HIT, OC1 MISS, OC2 MISS, OC3 HIT. The other 2B arms (PMX-2B, FO-b-PMX-2B, FO-b-2B) and the decay branches have no result in the entries read here, so most TL calls are still open.

---

## Contents

1. [Terms used](#terms)
2. [The plan: the 2B-token series and its calls](#plan)
3. [SR2-2B: setup](#sr2-2b-setup)
4. [SR2-2B: result](#sr2-2b-result)
5. [The decay gain and the decay branches](#decay-gain)
6. [Well occupancy against training tokens](#well-occupancy)
7. [Scorecard](#scorecard)
8. [What stands and what is open](#open)

---

<a name="terms"></a>
## 1. Terms used

- **SR2.** The conservative core of the ladder at $L = 2$ layers, trained on 0.53B tokens of OpenWebText. Settled perplexity 58.32.
- **SR2-2B.** The same model trained for 130,000 steps, which is 2.13B tokens, four times SR2's budget.
- **G2B.** The matched GPT-2 trained on the same 2.13B tokens. It has 8 layers, against SR2's 2. Settled 42.69. **W2** is the same GPT-2 at 0.5B tokens, settled 48.37.
- **Perplexity (PPL).** Lower is better. **Settled** is the mean of the last three evaluations.
- **WSD schedule.** Warm-up, then a stable phase at the peak learning rate, then a decay. For 130,000 steps: warm-up to 6,500, stable to 84,500, decay to 130,000. Peak $1.2 \times 10^{-3}$, floor $6 \times 10^{-5}$.
- **Decay gain.** How much the decay phase improves the model, in nats (natural-log units of perplexity). Its exact definition is in [section 5](#decay-gain).
- **Gate 3 at N = 3.** The whole two-layer stack is run in three smaller steps at the same total time, at inference. The cost, in nats, is the refinement penalty. A small penalty means the model behaves like a continuous flow.
- **FLOW-R penalty at N = 16.** The same idea applied layer by layer: each trained layer is cut into substeps, 16 in total. The penalty is how far that finer path lands from the trained step, in nats.
- **Kick share.** The fraction of a layer step carried by the explicit kick (the force term), measured by replaying the step with the kick set to zero.
- **Wells needed.** The model's learned potential $V_\theta$ has eight wells per layer. "Wells needed" is the smallest $k$ such that keeping only the $k$ strongest wells gives a perplexity within 0.5% of all eight.

---

<a name="plan"></a>
## 2. The plan: the 2B-token series and its calls

**Pre-registered 2026-10-22, before any 2B run (the author's decision on the calls).** Two claims the book is to rest on had been measured only at 0.53B tokens. The first is memory substitution: does the first-order twin stay level with PMX at 2B tokens (call MS4)? The second is law-governed evolution: does the potential fall along every trained layer step, and does the trained map stay close to its continuous law?

**The runs, in the registered order.**

1. PMX-2B and FO-b-PMX-2B (for MS4).
2. SR2-2B and FO-b-2B. SR2-2B is RT3's run (registered 2026-10-11, its call kept). FO-b-2B is its first-order twin.

Each arm is its 0.5B configuration plus 130,000 steps: 2.13B tokens, G2B's budget, about 1.07 passes over the 2.0B-token pool. Each run saves weights-only **snapshots** at steps 6,500, 16,500, 32,500, 48,500, 65,000, 84,500, 97,500, 113,500 and 130,000, so one run gives a trend over training.

**Calls carried in** (as registered): MS4, the first-order twin within the seed spread of PMX at 2B, 50%. RT3, SR2-2B's Gate 3 at $N = 3$ at or below 0.30 nats with its per-layer flow still converging, 45%.

**The TL calls:**

| id | prediction (short form) | called |
| --- | --- | ---: |
| TL1 | the potential falls along the trained step for at least 99% of tokens at both layers, every snapshot from 16,500 on, all four arms | 80% |
| TL2 | the per-layer flow penalty at N = 16 is larger at 84,500 than at 32,500, in at least three of the four arms | 60% |
| TL3 | from 32,500 to 130,000 the first-order arm's N = 16 penalty rises more than its twin's, in both pairs | 55% |
| TL4 | the first-order arms' median direction cosine at layer 0 is at least 0.95 at every snapshot | 70% |
| TL5 | in both pairs the second-order arm gains more than its first-order twin from 84,500 to 130,000 | 65% |
| TL6 | SR2-2B is more than 4.9% better than FO-b-2B at the end | 50% |
| TL7 | in both K = 64 arms, N_eff is larger at 84,500 than at 32,500 | 55% |
| TL8 | in both K = 64 arms, N_eff at 130,000 exceeds the same arm's 0.5B best checkpoint | 60% |
| TL9 | at 130,000, FO-b-PMX-2B's N_eff exceeds PMX-2B's at both layers | 55% |
| TL10 | at 130,000, swapping a sequence's occupations costs more than 0.5% perplexity in at least one K = 64 arm (registered 2026-10-23) | 35% |

Here $N_{\mathrm{eff}}$ is the effective number of PMX modes in use, from the entropy of each mode's share of the wells' force.

---

<a name="sr2-2b-setup"></a>
## 3. SR2-2B: setup (protocol `p-sr2-2b-setup`)

**Set up 2026-11-12, the first 2B run (the author's decision, 2026-11-12).** The registered order put the PMX pair first. SR2-2B runs first instead, for three reasons:

- every 2B comparison is read against the core's own 2B number;
- its change from 0.5B to 2B is the core's data-scaling slope, comparable with G2B's fall of 11.7% over the same fourfold budget;
- at about 1.1 s per step it is the cheapest arm with a registered call (RT3).

The calls are the registered ones: RT3, TL1, TL2 and TL4 on this arm, and its half of TL3, TL5 and TL6, scored when FO-b-2B has run. Nothing is added.

**Configuration.** SR2's six changes from the notebook defaults, plus 130,000 steps and the nine snapshot steps. Float32, seed 0, OpenWebText. The run tag was checked locally against HEAD `1eef209`.

**Data provenance (2026-11-12).** The Drive folder holding the ladder's OpenWebText pool was deleted before launch. The pool was rebuilt with `debug/rebuild_owt_pool.py`, repeating the July procedure.

- The rebuilt stream reproduces: its first 200M and first 1B tokens are identical to the June pools kept locally.
- The ladder's validation file appears in none of the rebuilt 2B training tokens, so there is no overlap.
- **Not checked:** whether the deleted training file was the first 2B tokens of this stream. A streaming check to 4.2B tokens settles it.
- SR2-2B trains on the rebuilt `openwebtext_train_2000M.npy` and evaluates on the ladder's own validation file (md5 `a4e8b4af…`), so its validation numbers compare with every earlier ladder arm.

**Launch, 2026-11-12 (late evening), from GitHub `d968e75`, on an A100 80 GB.** 76,572,110 parameters, 35,427,080 of them in the wells. Batch `16 x accum=2`. Step 50: training loss 7.6356 at a learning rate of 9.23e-06. This is not comparable with SR2's 7.6064, whose step 50 ran at four times the rate. About 1.05 s per step, 38 to 39 h projected, two Colab sessions.

<a name="sr2-2b-result"></a>
## 4. SR2-2B: result (protocol `p-sr2-2b-result`)

**Filed 2026-11-15; the run finished 2026-11-14.** Model card: [`semsimula-ladder-live-owt-d384-l2-none-norc-sr2-2b`](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-none-norc-sr2-2b).

| | SR2 (0.53B) | SR2-2B (2.13B) | G2B, matched GPT-2 at 2.13B |
| --- | ---: | ---: | ---: |
| settled (mean of last three evaluations) | 58.32 | 48.58 | 42.69 |
| final (130,000) | | 48.21 | 42.68 |
| best | 56.49 | 47.56 (126,500) | |
| end of stable phase (84,500) | | 59.81 | 61.84 |
| decay gain (nats) | 0.193 | 0.220 | 0.368 |

**The gap to the transformer narrows with data and opens in the decay.**

- From 0.53B to 2.13B tokens, SR2's settled perplexity falls 16.7%. G2B's falls 11.7% (from W2's 48.37).
- So the ratio to the matched GPT-2 goes from 1.206 to 1.138.
- At the end of the stable phase SR2-2B is 3.3% ahead (59.81 against 61.84). The decay turns that into 13.8% behind.

**RT3: MISS.** Gate 3 at $N = 3$ reads 0.772 nats (0.405 at 0.5B), against the call's 0.30 or less. FLOW-R's per-layer flow converges by the letter (successive differences 0.006 and 0.005 nats), but to a penalty of 0.920 nats (0.104 at 0.5B). Also recorded: Gate 1 +42.2 and Gate 2 at $N = 3$ +39.7 perplexity. *Unclear in the source:* the unit of the Gate 1 and Gate 2 figures (percent or perplexity points) is not stated beyond "perplexity".

**The map departs from the flow with training.** FLOW-R's penalty at $N = 16$ on every snapshot:

| step | 6,500 | 16,500 | 32,500 | 48,500 | 65,000 | 84,500 | 97,500 | 113,500 | 130,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| penalty at N = 16, nats | 0.008 | 0.079 | 0.486 | 0.619 | 0.691 | 0.797 | 0.807 | 0.879 | 0.922 |

Step 6,500 is the end of warm-up, 84,500 the start of the decay, 130,000 the end. Layer 1's kick share rises from 0.36 to 0.86 over the run. *Note:* RT3's 0.920 and the 130,000 snapshot's 0.922 differ slightly. The setup planned FLOW-R for RT3 on the best checkpoint, which would explain it, but the result entry does not say which checkpoint gave 0.920.

**TL calls on this arm.** TL2's condition holds here: 0.486 at 32,500 against 0.797 at 84,500, both at the peak rate. TL2 itself is scored when three of its four arms have run. TL1 waits for `descent_readout.py` to support SR2. TL4 applies to first-order arms only.

**Directions per token** (`vtheta_direction_count.py`, every snapshot). Layer 0 fills its budget at every snapshot: it needs more than 24 of its 32 directions. Projecting onto 24 costs 3.9% at 130,000, and the need is flat after 16,500. Layer 1 needs 8 directions from 16,500 to 97,500, and 12 at 113,500 and 130,000.

**Stiffness.** The median of $\omega \Delta t$ (how far, in radians, a stiff mode turns in one layer step) is 4.087 at $\Delta t = 4$ at 130,000, against 7.554 for SR2 at 0.5B.

**Why the decay gain is smaller.** The decay gain follows the gradient noise the optimizer carries at the peak rate, and G2B carries about 4.5 times more per coordinate. That measurement is summarised in [SR2_L1_and_Decay_Noise.md](SR2_L1_and_Decay_Noise.md#decay-noise) (protocol `p-decay-noise-cpu`).

<a name="decay-gain"></a>
## 5. The decay gain and the decay branches (protocol `p-decay-branch-setup`)

**Set up 2026-11-13, before SR2-2B's decay starts.** Running them was the author's decision. DG1 and DG2 were confirmed by the author on 2026-11-13 at the values below, before step 84,500. DB1 to DB3 were confirmed the same day, before any branch ran.

**Why.** At step 71,650, SR2-2B was level with G2B: mean validation perplexity 61.2 against 62.2 over steps 67,000 to 72,000. At 0.5B the two were also close before the decay (70.73 against 68.01) and far apart after it (58.32 against 48.37). So the gap opens in the decay. Both models use the same schedule, so the smaller gain belongs to the model. Two readings were open: the cosine decay to a 5% floor does not suit our model, or our model carries less rate-driven excess loss at the peak rate, which no decay shape can recover.

**Decay gain, defined.** The mean validation log-perplexity over the last five evaluations at or before the decay start, minus the settled value. The windows are steps 19,000 to 21,000 at 0.5B and 82,500 to 84,500 at 2B. Measured: SR2 at 0.5B 0.193 nats, GPT-2 at 0.5B 0.341, GPT-2 at 2B 0.368. With SR2-2B's evaluation spread of 0.020 nats, the standard error of its gain is about 0.015 nats.

*Two definitions.* The G2B entry (§5.17, 2026-10-18) gives G2B's decay gain as 0.371, from the single evaluation at 84,500 (61.84) to the settled 42.69. The 0.368 used here comes from the five-evaluation definition above.

**The clipping conjecture.** Clipping might cap SR2's updates in the stable phase, leaving the decay less noise to remove. What the logs showed:

- At 0.5B neither of SR2's clips acted (the global norm exceeded 1.0 in 0.2% of logged steps), and the gap was there all the same.
- GPT-2 was clipped in most steps of its 2B decay and still gained 0.368.
- Under AdamW a steady clip factor largely cancels. So clipping is not the cause of the 0.5B gap.
- At 2B it might add to it: from step 57,000 SR2-2B's $V_\phi$ gradient grows (global norm above 1.0 in 65% of steps over 60,000 to 72,000). SR2 at 0.5B never showed this.

| id | prediction | called | outcome |
| --- | --- | ---: | --- |
| DG1 | SR2-2B's decay gain is below 0.15 nats: the clipping active at 2B adds to the gap (refuted at 0.18 or more; 0.15 to 0.18 undecided) | 30% | **refuted** (0.220) |
| DG2 | SR2-2B's decay gain is smaller than GPT-2's 2B gain (0.368) by 0.10 nats or more | 75% | **HIT** (0.148 below) |

**The decay branches.** Each model reruns the same decay window (84,500 to 130,000) with only the shape and floor changed. B0 is the control (the current cosine to $6 \times 10^{-5}$). B1 is a cosine to zero. B2 is $1 - \sqrt{p}$ to zero, where $p$ is the fraction of the decay elapsed. SR2's branches start from SR2-2B's checkpoint at step 82,500, GPT-2's from its checkpoint at step 75,000. The score is the narrowing: SR2's gain over its B0 minus GPT-2's gain over its B0, in settled log-perplexity.

| id | prediction | called | outcome |
| --- | --- | ---: | --- |
| DB1 | neither B1 nor B2 narrows the SR2 to GPT-2 gap by 0.03 nats or more: the gap is not a schedule effect | 65% | no result recorded |
| DB2 | B1 or B2 improves GPT-2's settled perplexity by 1% or more over its B0 | 50% | no result recorded |
| DB3 | B1 or B2 improves SR2's settled perplexity by 1% or more over its B0 | 50% | no result recorded |

The estimated cost is about 43 GPU-hours after SR2-2B. As of the setup (checked 2026-11-13), the notebook changes the branches need were not yet written, and where the branches go in the queue is the author's decision. The entries read for this note record no branch result.

<a name="well-occupancy"></a>
## 6. Well occupancy against training tokens (protocol `p-well-occupancy-setup`)

**Set up 2026-11-13, while SR2-2B trained and before any of its snapshots was read.** It is the author's question, for the book. OC1 to OC3 were confirmed by the author on 2026-11-13 at the values below, before any snapshot was read.

**The question.** Does a layer use more of its eight $V_\theta$ wells as the training budget grows? At 0.5B, layer 1 runs on one or two wells in every healthy arm, and layer 0 needs all eight. If layer 1 recruits more wells with more data, "two wells are enough" is a property of the 0.5B budget, and the $K = 2$ arms would lose ground at 2B. If not, the narrowness is structural.

**The measure.** Wells needed per layer (see [Terms](#terms)), on 16 validation windows, with `debug/keep_k_curves.py`. It was to be read on all nine SR2-2B snapshots and on SR2's 0.5B best checkpoint. The snapshots in between are read at the peak rate or part-way through the decay, so the curve mixes the budget with the schedule. One seed, one corpus.

| id | prediction | called | outcome |
| --- | --- | ---: | --- |
| OC1 | at step 130,000, layer 1 needs at most two wells | 65% | **MISS** (three wells) |
| OC2 | layer 1's wells needed at 130,000 equals its count at the 32,500 snapshot | 60% | **MISS** (two at 32,500, three at 130,000) |
| OC3 | layer 0 still needs all eight wells at 130,000 | 70% | **HIT** |

**Occupancy per snapshot, as recorded.** The result entry gives layer 1's count at two snapshots: two wells at 32,500 and three at 130,000. It says the third well enters in the decay. Layer 0 needs all eight at 130,000. *Not in the protocol text:* a wells-needed count for each of the other snapshots, and the SR2 0.5B reading under the same measure.

*Unclear in the source:* the setup names SR2's best checkpoint as step 32,500, while the SR2-2B result table gives SR2's best as 56.49 without a step, and other entries place it at step 31,000.

An optional decayed 1B point (a branch from step 37,500 with a 65,000-step schedule) was **proposed, not agreed**.

<a name="scorecard"></a>
## 7. Scorecard

| call | called | outcome |
| --- | ---: | --- |
| RT3 | 45% | MISS (Gate 3 at N = 3: 0.772 nats) |
| DG1 | 30% | refuted (0.220) |
| DG2 | 75% | HIT (0.148 below G2B) |
| OC1 | 65% | MISS |
| OC2 | 60% | MISS |
| OC3 | 70% | HIT |
| TL2 | 60% | condition holds on SR2-2B; scored when three of four arms have run |
| TL1 | 80% | waits for descent_readout.py to support SR2 |
| TL3, TL5, TL6 | 55%, 65%, 50% | wait for FO-b-2B |
| TL4, TL7 to TL10, MS4 | as registered | wait for the first-order and PMX 2B arms |
| DB1 to DB3 | 65%, 50%, 50% | no branch result recorded |

<a name="open"></a>
## 8. What stands and what is open

**Stands.**

- Four times the data narrows SR2's gap to the matched GPT-2: the ratio goes from 1.206 to 1.138.
- SR2-2B is ahead of G2B at the end of the stable phase and behind after the decay. The gap forms in the decay.
- The decay gain is not held down by clipping at 2B (DG1 refuted) and is well below GPT-2's (DG2 HIT).
- Training moves the two-layer map away from a continuous flow: the per-layer penalty at $N = 16$ grows from 0.008 to 0.922 nats across the snapshots (RT3 MISS).
- Layer 1 recruits a third well in the decay at 2B; layer 0 keeps all eight.

**Open.**

- The first-order twin FO-b-2B and the PMX pair at 2B, and with them most TL calls and MS4.
- The decay branches DB1 to DB3.
- Whether the deleted training file was the first 2B tokens of the rebuilt stream.
