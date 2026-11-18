# SR2-L1 and the decay gain against the gradient noise

**What this note is.** A short, readable summary of three entries of the depth-ladder protocol, written 2026-11-17: the one-layer conservative model SR2-L1, with its setup, calls, result and diagnostics ([protocol, `p-sr2-l1-setup`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr2-l1-setup) and [protocol, `p-sr2-l1-result`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr2-l1-result)); the depth-against-wells call A1, scored on K2-R6 ([protocol, `p-k2-r6-result`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-k2-r6-result)); and the measurement of the decay gain against the optimizer's gradient noise on eleven checkpoints ([protocol, `p-decay-noise-cpu`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-decay-noise-cpu)). Every number, verdict and date is copied from the [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md), which keeps the full log and remains the record of truth. The 2B-token run these entries refer to is summarised in [Two_Billion_Token_Runs.md](Two_Billion_Token_Runs.md).

**Status in one line.** SR2-L1 settled 69.15, 18.6% above SR2: L1a HIT, L1b MISS, L1c HIT, all three proposed but not confirmed by the author. A1 HIT: K2-R6 settled 65.12, below SR2-L1's 69.15. The decay gain follows the optimizer's gradient noise with one constant, about 0.73, across three models.

---

## Contents

1. [Terms used](#terms)
2. [SR2-L1: setup and calls](#sr2-l1-setup)
3. [SR2-L1: result and diagnostics](#sr2-l1-result)
4. [A1: depth against wells](#a1)
5. [The decay gain against the gradient noise](#decay-noise)
6. [What stands and what is open](#open)

---

<a name="terms"></a>
## 1. Terms used

- **SR2.** The conservative core of the ladder at $L = 2$ layers, 0.5B tokens of OpenWebText (32,500 steps). Settled 58.32, decay gain 0.193 nats.
- **SR2-L1.** The same model at $L = 1$. **Run 7** is the ladder's earlier one-layer arm, and **run 3** its two-layer twin.
- **Settled perplexity.** The mean of the last three evaluations. Lower is better.
- **Decay gain.** The mean validation log-perplexity over the last five evaluations at or before the decay start (steps 19,000 to 21,000 at 0.5B), minus the settled value, in nats. It measures how much the final phase of the learning-rate schedule improves the model.
- **Wells.** The learned potential $V_\theta$ is a bank of wells, shared by every layer. SR2 has eight wells of rank 4 per head. **K2-R6** has two wells of rank 6, at $L = 2$.

---

<a name="sr2-l1-setup"></a>
## 2. SR2-L1: setup and calls (protocol `p-sr2-l1-setup`)

**Set up 2026-11-15, before the decay branches. The run is the author's decision of 2026-11-15. L1a to L1c were proposed, not yet confirmed by the author.**

**Why.** Run 7 differs from its two-layer twin in two ways: depth, and a register bank that is read but never updated. SR2 has the reverse channel off, so the bank has no path to the tokens at any depth. SR2 at $L = 1$ therefore differs from SR2 only in depth. It also bears on SR2-2B's decay gap: at 0.5B run 7 gained 0.135 nats in the decay and run 3 gained 0.201. If the decay gain grows with depth, part of SR2-2B's shortfall against the 8-layer GPT-2 may be depth.

**What it cannot test.** At $L = 1$ the velocity entering the only layer is zero, so no inertia carries between layers. The arm says nothing about second order as such.

**Configuration.** SR2's six changes plus `LADDER_L = 1`. The total time is held at 8, so the one step has $\Delta t = 8$. 32,500 steps, the ladder's schedule, float32, seed 0. The CPU build gives 76,545,484 parameters (SR2: 76,572,110), of which 35,425,160 are in $V_\theta$.

**Data caveat.** SR2-L1 trains on the training file rebuilt on 2026-11-12. SR2 trained on the deleted file. Whether the two are the same tokens is unchecked, so the comparison with SR2 carries that caveat.

**Calls,** against SR2 at 0.5B (settled 58.32, decay gain 0.193 nats):

| id | prediction | called |
| --- | --- | ---: |
| L1a | SR2-L1's settled perplexity is at least 15% above SR2's (67.07 or more) | 80% |
| L1b | the gap is at least run 7's 30.0% (75.82 or more): removing the static bank does not shrink the one-layer gap | 50% |
| L1c | SR2-L1's decay gain is smaller than SR2's 0.193 by 0.03 nats or more (0.163 or less) | 70% |

<a name="sr2-l1-result"></a>
## 3. SR2-L1: result and diagnostics (protocol `p-sr2-l1-result`)

**Result, 2026-11-15.** 32,500 steps in 5.65 h at 0.63 s per step. Model card (published 2026-11-15, Gen 3 collection): [`semsimula-ladder-live-owt-d384-l1-none-norc-sr2`](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l1-none-norc-sr2).

| | SR2 (L = 2) | SR2-L1 (L = 1) | difference |
| --- | ---: | ---: | ---: |
| settled | 58.32 | 69.15 | +18.6%, +0.170 nats |
| best | 56.49 (31,000) | 67.77 (31,000) | |
| mean over 19,000 to 21,000 (before the decay) | 70.73 | 79.67 | |
| decay gain (nats) | 0.193 | 0.142 | −0.051 |

**Scored.** The calls were not confirmed by the author before the decay started, so they are scored on the letter and marked unconfirmed.

| id | called | outcome |
| --- | ---: | --- |
| L1a | 80% | **HIT** (69.15, +18.6%) |
| L1b | 50% | **MISS** (+18.6%) |
| L1c | 70% | **HIT** (0.142) |

**The second layer is worth nothing early, and its value builds with training.** Gap of SR2-L1 against SR2, as window means of validation perplexity:

| steps | 500–1,500 | 2,000–4,000 | 4,500–7,000 | 7,500–10,000 | 10,500–15,000 | 15,500–21,000 | 21,500–26,000 | 26,500–30,000 | 30,500–32,500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gap | −2.0% | −3.0% | −1.4% | +1.4% | +6.6% | +11.4% | +14.1% | +17.4% | +18.9% |

One layer leads for the first 7,000 steps. The two cross near 8,000, and the gap widens from there through the decay.

**Readings.**

- **L1b:** the conservative model loses 18.6% at one layer, against run 7's 30.0% with the register bank frozen. So part of run 7's gap was the static bank, not depth. The split is not exact, since SR2 also differs from run 3 in its gradient paths and integrator.
- **L1c:** one layer gains less in the decay here too (0.142 against 0.193 nats, about 2.4 standard errors), as in run 7. Depth is a candidate cause of SR2-2B's decay shortfall. The decay branches decide whether the schedule is another.
- **The one-layer potential is softer.** The median of $\omega \Delta t$ (how far a stiff mode turns in one step, in radians) is 3.12 at $\Delta t = 8$, so $\omega \approx 0.39$. SR2 read 7.55 at $\Delta t = 4$, so $\omega \approx 1.89$. The one-layer model's stiff modes are about five times slower.
- **Run health.** No skipped steps. SCAF CLEAN at all seven audits.

**Diagnostic cells on the best checkpoint (step 31,000).**

- **Gate 1** (velocity reset) reads +0.00, as it must at $L = 1$.
- **Gate 2** (more hops at $\Delta t = 8$): 71.88 trained, 74.33 at two hops (+3.4%). SR2's one extra hop costs +34.8%. The one-layer step lands near a fixed point of itself.
- **Gate 3** (refining the one step): 91.57 at $N = 2$ (+27.4%), 158.31 at $N = 3$. A map, not a flow, as SR2.
- **What the step is made of.** The single step moves the state 7.2 times its own norm. SR2's layer 0 moves it 6.0 times and its layer 1 0.72 times. The $V_\phi$ part of the step has median 0.17 at $L = 1$, against 0.053 at SR2's layer 0 and 0.000 at its layer 1.
- **Not yet read:** well occupancy, keep-k and the direction count on SR2-L1's best checkpoint, and FLOW-R at $L = 1$.

<a name="a1"></a>
## 4. A1: depth against wells (protocol `p-sr2-l1-setup`, scored in `p-k2-r6-result`)

**Filed 2026-11-15 at the author's request; the author shares the bet.** Two ways to spend the same architecture:

- (a) $L = 2$ with two wells at rank 6: the K2-R6 arm of the well screen, about 52.96M parameters, 11.8M of them in the wells.
- (b) $L = 1$ with eight wells at rank 4: SR2-L1, 76.55M parameters, 35.4M in the wells.

The case for (a): one layer cost run 7 0.263 nats, and that gap widened through training. Two wells at rank 6 give 12 directions per token, enough for layer 1 but not for layer 0. The case for (b) is layer 0's direction need and the larger bank. Both arms train on the rebuilt pool and are scored on settled perplexity.

| id | prediction | called | outcome |
| --- | --- | ---: | --- |
| A1 | at 0.5B, (a) K2-R6 at L = 2 settles below (b) SR2-L1 | 65% | **HIT** |

**Outcome (K2-R6 scored 2026-11-17).** K2-R6 settled 65.12, below SR2-L1's 69.15, so A1 hits. K2-R6 is still 11.7% behind SR2's 58.32. The same bet at 2B (about 75%) is not registered: no 2B run of either arm is planned.

<a name="decay-noise"></a>
## 5. The decay gain against the gradient noise (protocol `p-decay-noise-cpu`)

**Measured 2026-11-15 on eleven checkpoints, on the CPU.** The author's question: is there a theoretical reason the decay gain depends on $L$?

**The argument.** At a constant peak rate the optimizer bounces in the steep directions, and the loss carries an excess that the decay drains. Take one coordinate with curvature $\lambda$ and gradient noise $\sigma$, with the noise dominating. AdamW's second moment is then $v \approx \sigma^2$, and the excess loss is $\eta \sigma / 4$, whatever $\lambda$ is. Summed over the coordinates, the predicted decay gain is

$$G_{\mathrm{pred}} = \frac{\eta_{\mathrm{peak}}}{4} \sum_i \sqrt{v_i}.$$

In SR2 the layers share parameters: the well bank is 93% of the non-embedding parameters and serves every layer. Each layer adds its own gradient to the shared parameters, so their noise grows with depth, by between $\sqrt{L}$ and $L$. That is a reason for the decay gain to grow with depth.

**The test.** `debug/adam_noise_decay_gain.py` sums $\sqrt{v_i}$ over every coordinate of a checkpoint's AdamW state. It also reads the median of $\lvert m \rvert / \sqrt{v}$, which is about 0.15 for pure noise. The reading that matches the argument is at the peak rate just before the decay. SR2 at 0.5B has no such checkpoint left, so it is read at the end of its decay.

| checkpoint | rate at the checkpoint | measured gain (nats) | sum of sqrt(v) | predicted gain | measured over predicted | median abs(m) over sqrt(v) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **SR2-L1, step 22,500** | 97% of peak | 0.142 | 661.1 | 0.198 | **0.72** | 0.098 |
| **SR2-2B, step 82,500** | peak | 0.220 | 1,010.3 | 0.303 | **0.73** | 0.119 |
| **GPT-2 2B, step 75,000** | peak | 0.368 | 1,654.9 | 0.496 | **0.74** | 0.146 |
| SR2-L1, best (31,000) | end of decay | 0.142 | 683.1 | 0.205 | 0.69 | 0.106 |
| SR2, best (31,000) | end of decay | 0.193 | 836.1 | 0.251 | 0.77 | 0.143 |
| SR2-2B, best (126,500) | end of decay | 0.220 | 917.6 | 0.275 | 0.80 | 0.137 |
| SR2-2B, step 37,500 | peak | 0.220 | 907.6 | 0.272 | 0.81 | 0.126 |
| GPT-2 2B, step 25,000 | peak | 0.368 | 1,568.9 | 0.471 | 0.78 | 0.145 |
| GPT-2 2B, step 100,000 | 75% of peak | 0.368 | 1,861.3 | 0.558 | 0.66 | 0.150 |
| GPT-2 2B, step 125,000 | 8% of peak | 0.368 | 2,487.9 | 0.746 | 0.49 | 0.153 |
| GPT-2 2B, best (130,000) | end of decay | 0.368 | 2,504.1 | 0.751 | 0.49 | 0.151 |

**Readings.**

1. **The coordinates are noise-dominated** (median of $\lvert m \rvert / \sqrt{v}$ 0.11 to 0.15 everywhere), as the argument assumes.
2. **One constant fits both architectures, both depths and both budgets.** At the three pre-decay checkpoints, measured over predicted is 0.72, 0.73 and 0.74, while the gains span 0.142 to 0.368 nats. So $G \approx 0.73 \cdot G_{\mathrm{pred}}$ across a weight-shared one-layer model, a weight-shared two-layer model and an unshared eight-layer transformer.
3. **The second layer's extra noise is all in the reused parameters.** By block, SR2 against SR2-L1 at step 31,000: the unshared embedding and readout carry the same noise (ratio 1.00). The bank's blocks carry 1.48 to 2.15 times more at $L = 2$, inside the range the shared-gradient argument allows. The well weights (2.67) and the $\xi$ content routing (3.50) carry more than two summed contributions can give; composition through the second layer is the candidate.
4. **GPT-2.** Before the decay it fits the same constant (0.74). During its decay its gradient noise grows instead of falling. Per coordinate outside the vocabulary matrices, GPT-2 carries about 4.5 times SR2-2B's noise before the decay, which is where its larger gain comes from.

**What this settles and what not.**

- It supports the reading that SR2's smaller decay gain is less rate-driven excess at the peak rate, not a badly suited decay shape: SR2 stores less noise, so the decay has less to drain. It also gives the depth dependence a mechanism.
- Not settled: the measured gain also contains progress along the valley floor during the decay, which the argument ignores. The constant is fitted on three points, not derived.
- A consequence worth testing, not adopted: a higher peak rate may buy more progress for little extra excess.

**Later readings on the smaller banks** (recorded in `p-k4-r6-result` and `p-k2-r6-result`, at step 22,500):

| checkpoint | sum of sqrt(v) | predicted gain | measured gain | measured over predicted |
| --- | ---: | ---: | ---: | ---: |
| K4-R6, step 22,500 | 861.2 | 0.258 | 0.175 | 0.68 |
| K2-R6, step 22,500 | 868.6 | 0.261 | 0.176 | 0.67 |

Both fall below the 0.72 to 0.74 band: the smaller banks drain less than their noise predicts. For K4-R6 the protocol calls 0.68 "a first point outside the band, not a refutation".

<a name="open"></a>
## 6. What stands and what is open

**Stands.**

- A second layer is worth 18.6% in settled perplexity to the conservative core at 0.5B, and its value builds after step 8,000.
- One layer gains less in the decay (0.142 against 0.193 nats).
- At this budget, two layers with a small bank beat one layer with the full bank (A1 HIT).
- Before the decay, the decay gain is about 0.73 times the noise prediction in three models.

**Open.**

- L1a to L1c remain unconfirmed by the author.
- Well occupancy, keep-k, the direction count and FLOW-R on SR2-L1.
- Whether the training files of SR2 and SR2-L1 hold the same tokens.
- Why the smaller banks fall below the band.
