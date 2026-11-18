# The matched GPT-2 baselines: cosine, W1, W2 and G2B

**What this note is.** A short, readable summary, written 2026-11-17, of the GPT-2 models that every Fock-PARFLM and SR-series perplexity in the depth ladder is compared with. It covers the token-matched cosine baseline of protocol §5.4 and the three runs of protocol §5.17: W1 (the same GPT-2 on the ladder's learning-rate schedule), W2 (the same at the ladder's peak learning rate) and G2B (W2's setup at 2B tokens). For each it gives the configuration, what was registered before the run, the verdicts, and the results, including the stable-phase fits. It also says plainly what is and is not matched between these baselines and the Fock arms. Every number, verdict and date is copied from the [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md), which keeps the full log and remains the record of truth.

**Status in one line.** All four runs are done and scored: the cosine baseline settles at 49.81, W1 at 48.82, W2 at 48.37, and G2B at 42.69.

## Contents

1. [Terms used](#1-terms-used)
2. [The four baselines at a glance](#2-the-four-baselines-at-a-glance)
3. [What is and is not matched](#3-what-is-and-is-not-matched)
4. [The cosine baseline (protocol §5.4)](#4-the-cosine-baseline-protocol-54)
5. [W1: GPT-2 on the ladder's WSD schedule (protocol §5.17)](#5-w1-gpt-2-on-the-ladders-wsd-schedule-protocol-517)
6. [W2: GPT-2 at the ladder's peak learning rate (protocol §5.17)](#6-w2-gpt-2-at-the-ladders-peak-learning-rate-protocol-517)
7. [G2B: the matched GPT-2 at 2B tokens (protocol §5.17)](#7-g2b-the-matched-gpt-2-at-2b-tokens-protocol-517)
8. [Other matched GPT-2 runs, for orientation](#8-other-matched-gpt-2-runs-for-orientation)
9. [Unclear items](#9-unclear-items)

---

## 1. Terms used

- **PPL** is validation perplexity. **Settled** is the mean of the last three evaluations, the rule used everywhere in the ladder.
- **WSD** (warmup, stable, decay) is the ladder's learning-rate schedule: a warmup over the first 5% of steps, a constant rate until 65%, then a cosine decay to a floor. At 32,500 steps the decay starts at step 21,125, so step 21,000 is the end of the stable phase.
- **Cosine** is the schedule of the published GPT-2 baseline: a warmup to step 2,000, then a decay from 6e-4 to 6e-5 for the rest of the run, with no constant phase.
- **Decay gain** is how much the final decay improves the loss, in nats: $\ln P_{\mathrm{stable}} - \ln P_{\mathrm{settled}}$, where $P_{\mathrm{stable}}$ is the perplexity at the end of the stable phase and $P_{\mathrm{settled}}$ the settled perplexity. In W1, for example, the decay takes perplexity from 64.01 to 48.82, a gain of 0.271 nats.
- **Stable-phase fit** (from protocol §5.16, F0): the curve $L(t) = L_\infty + A t^{-\alpha}$, fitted to the validation loss $L$ (in nats) at step $t$ over the constant-rate phase, steps 3,000 to 21,000. $L_\infty$ is the level the loss would approach if the rate stayed constant. The tables' PPL∞ is $e^{L_\infty}$, the same level in perplexity. A 90% interval comes from 2,000 bootstrap refits.
- **Fock arms** are the Fock-PARFLM models of the ladder. Their depth $L$ is the number of layer steps (2 or 4 here); their width is $d = 384$.

## 2. The four baselines at a glance

All four are the same GPT-2 architecture from `colab_matched_gpt2_baseline_openwebtext.ipynb`, trained on OpenWebText from scratch at seed 0.

| | cosine baseline | W1 | W2 | G2B |
| --- | --- | --- | --- | --- |
| protocol | §5.4, `p-gpt2-matched` | §5.17, `p-w1` | §5.17, `p-w2` | §5.17, `p-g2b` |
| date scored | 2026-09-21 | 2026-10-06 | 2026-10-18 | 2026-10-18 |
| width d, depth L | 384, 8 | 384, 8 | 384, 8 | 384, 8 |
| heads × d_k | 6 × 64 | as cosine | as cosine | as cosine |
| embeddings | tied | tied | tied | tied |
| parameters | 33.7M | 33.7M | 33.7M | 33.7M |
| schedule | cosine, warmup 2,000 | WSD | WSD | WSD |
| peak / floor learning rate | 6e-4 / 6e-5 | 6e-4 / 6e-5 | 1.2e-3 / 6e-5 | 1.2e-3 / 6e-5 |
| weight decay | 0.1 | 0.1 | 0.1 | 0.1 |
| batch | 32 × 512 | 32 × 512 | 32 × 512 | 32 × 512 (16,384 tokens per step) |
| steps | 32,500 | 32,500 | 32,500 | 130,000 |
| tokens | 532.5M | 532.5M | 0.532B | 2.13B |
| tag | (published default) | `_wsd` | `_wsd_lr0p0012` | `_wsd_lr0p0012_st130000` |
| settled PPL | **49.81** | **48.82** | **48.37** | **42.69** |

The width, depth, weight decay and batch of the cosine baseline are those listed in W1's run description ("everything else at the published baseline's values"). The 33.7M parameter count is the one quoted for the published baseline in protocol §5.10 and for W2 in §5.17. The published baseline is on the Hugging Face Hub as `dimitarpg13/semsimula-ladder-owt-d384-l2-gpt2-matched` (uploaded 2026-09-27, protocol §7.3a).

## 3. What is and is not matched

**Matched in every run:**

- **Width.** $d = 384$, the same semantic-space dimension as every Fock arm. Width is held fixed by design (protocol §5.10): a Fock model at $d = 512$ would be a different dynamical system.
- **Data and tokens.** OpenWebText, 532.5M tokens at 32,500 steps, the same budget as every ladder point. G2B and the 2B-token Fock arms read the same 2.0B-token pool.
- **Evaluation.** The same validation tokens and the same settled rule. The GPT-2 notebook draws its own evaluation batches from those tokens (protocol `p-gpt2-ts`).

**Matched only in some runs:**

- **Schedule.** The cosine baseline does not use WSD. W1, W2 and G2B do.
- **Learning rate.** The cosine baseline and W1 peak at 6e-4. The Fock ladder's rate is 1.2e-3 (`LADDER_LR`). Only W2 and G2B run at 1.2e-3, and W2's schedule was verified equal to a ladder run's at all 650 logged steps.

**Not matched in any run:**

- **Depth.** GPT-2 has $L = 8$ layers; the Fock arms compared with it have $L = 2$ or $L = 4$. The protocol notes this as "the depth mismatch the G2B card states (L = 8 against L = 2)".
- **Parameter count.** GPT-2 has 33.7M parameters. The $L = 4$ live Fock arm has 76.8M and G2 76,770,256. A parameter-matched GPT-2 (G1, $L = 22$) was registered but deferred; see the [abstract-gating note](Abstract_Gating_Runs.md).
- **Attention heads.** GPT-2 uses 6 heads of width 64; the Fock models use 8 of width 48, chosen for width parity.
- **Hyperparameter effort.** GPT-2's settings are nanoGPT defaults, "known-good across a wide range". At §5.4's date the Fock side's settings were inherited and unswept. Section 4 gives the table.

## 4. The cosine baseline (protocol §5.4)

**Done 2026-09-21.** Batch 32 throughout, 32,500 steps, 532.5M tokens: token-matched to every ladder point for the first time. The earlier GPT-2 figure was measured at 360.4M tokens.

| | value |
| --- | ---: |
| final (step 32,500) | **49.76** |
| settled (mean of last three) | **49.81** |

**Prediction scored: HIT**, the first in the programme. Registered: 49.5, band 49.0 to 50.0. Actual 49.76, error +0.26. The method that worked was an analogue, not a fit: the published run's behaviour over the same learning-rate range, applied to the current value. Three fits to the log of the learning rate gave 38.87, 47.32 and 49.00 depending on the window.

**Consequences.** The extra 172M tokens are worth 4.86 PPL to GPT-2, so every ratio widens:

| | tokens | settled | ratio |
| --- | ---: | ---: | ---: |
| **GPT-2 matched** | 532.5M | **49.81** | 1.000 |
| Fock L=2 + exchange force | 532.5M | 68.33 | **1.372** |
| Fock L=2, conservative only | 532.5M | 75.09 | **1.508** |
| Fock L=8 warm-start (reference) | 532.5M | 81.58 | 1.638 |

The 1.49x figure then in circulation for the $L = 8$ arm was computed against the 360.4M-token baseline. Against the token-matched one it is 1.638x.

**Tuned against untuned.** The comparison is matched on data and evaluation, not on hyperparameter effort:

| | GPT-2 | Fock ladder (at §5.4's date) |
| --- | --- | --- |
| learning rate | 6e-04, nanoGPT default | 3e-04, inherited from the L=8 arm, never swept at this depth |
| schedule | cosine, warmup 2,000 | WSD 5/60/35, inherited |
| lambda on the exchange field | n/a | 1.0, chosen on structural argument, not evidence |
| heads × d_k | 6 × 64, standard | 8 × 48, chosen for width parity |
| T = L·dt, gamma, clip overrides | n/a | inherited or conventional, all unswept |

So every ratio of that date is "at these hyperparameters": an upper bound on the gap, not a measurement of it. The full inventory of unswept settings is in [Hyperparameter_Tuning_Checklist.md](../Hyperparameter_Tuning_Checklist.md).

**The learning-rate sweep that followed.** $L = 2$ `'none'` at 1.2e-3 settles at 66.98 against 75.09 at 3e-4, 10.8% better, moving its ratio from 1.507 to 1.345. 2.4e-3 came back at 69.59, 3.9% worse, so the optimum is bracketed (a quadratic through the three points puts it at 1.13e-3). **1.2e-3 became the ladder learning rate.** The protocol adds that closing 68.33 to 49.81 needs 27%, while learning-rate tuning typically buys 5 to 15%.

## 5. W1: GPT-2 on the ladder's WSD schedule (protocol §5.17)

**Pre-registered 2026-10-06, before the run.**

**Why.** The cosine baseline and the Fock arms used different schedules, so every GPT-2 comparison mixed architecture with schedule. That included "$L = 4$ reaches parity with GPT-2" (50.10 against 49.81). The F0 floor fits (§5.16) also could not include GPT-2, because a cosine curve has no constant-rate phase.

**The run.** The cosine baseline's notebook with `LR_SCHEDULE = 'wsd'`, everything else unchanged (section 2). The schedule's shape is the only change. Verified 2026-10-06: with the default, the schedule is identical to before at all 32,500 steps; with `wsd`, it equals the Fock ladder's formula at every step.

**Predictions:**

| | prediction | called |
| --- | --- | --- |
| W1.1 | settled PPL within −3% to +2% of the cosine baseline (48.3–50.8) | 70% |
| W1.2 | settled PPL at or below the cosine baseline's 49.81 | 55% |
| W1.3 | the F0 fit on its stable phase (steps 3,000–21,000) is identified | 75% |
| W1.4 | that fit's L∞ lies below L=4 Fock's 90% interval (under 3.661 nats) | 50% |

**Decision rule.** Within 2% of 49.81, the parity reading of $L = 4$ survives the schedule change and the book's sentence stands. More than 2% better, the book's $L = 4$ comparison is restated against W1. More than 2% worse, the cosine baseline stays the reference.

**Scored 2026-10-06** ([protocol, `p-w1`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-w1)). Fit script `debug/w1_f0_fit.py`, using F0's functions unchanged on the 65 printed evaluations.

| | result | verdict |
| --- | --- | --- |
| settled | 48.82 (mean of 49.00, 48.78, 48.67); final and best 48.67 | −1.99% against 49.81 |
| W1.1 | within 48.3–50.8 | **HIT** |
| W1.2 | at or below 49.81 | **HIT** |
| W1.3 | stable-phase fit identified: L∞ 4.038 nats [4.031, 4.045], PPL∞ 56.74 [56.33, 57.14], α 0.93, rmse 0.003 | **HIT** |
| W1.4 | L∞ lies above L=4's whole interval [3.661, 3.839], and above G3's and G3′'s | **MISS**, in the opposite direction |

**Decision: within 2%, so the parity reading of $L = 4$ and the book's sentence stand.** The protocol notes the margin is 0.01 percentage points: met by the letter, on the boundary in substance. $L = 4$ Fock (50.10) is 2.6% behind W1 and 0.6% behind the cosine baseline.

**A stable-phase $L_\infty$ is not a floor.** W1 ended 0.15 nats (8 PPL) below its own stable-phase asymptote, because the decay removes the constant rate's noise. So $L_\infty$ cannot compare models trained at different peaks: GPT-2 ran at 6e-4, the Fock arms at 1.2e-3.

**Same schedule shape and token budget (descriptive):**

| model | PPL at step 21,000 (end of stable phase) | settled | decay gain (nats) | stable-phase PPL∞ |
| --- | ---: | ---: | ---: | ---: |
| GPT-2, WSD (W1), peak 6e-4 | 64.01 | 48.82 | **0.271** | 56.74 |
| L=4 Fock, peak 1.2e-3 | 60.80 | 50.10 | 0.194 | 43.07 |
| G3′ | 63.45 | 52.90 | 0.182 | 48.17 |
| G2 | 63.61 | 53.12 | 0.180 | (unidentified) |
| G3 | 65.22 | 54.21 | 0.185 | 49.31 |
| F3.1 | 71.81 | 57.76 | 0.218 | (unidentified) |

At the end of the stable phase the Fock arms lead or are level ($L = 4$ is 5% ahead). GPT-2 then gains more from the decay (0.27 nats against 0.18 to 0.22) and finishes ahead. W1 could not say whether that is the architecture or GPT-2's lower peak rate. W2 was designed to separate the two.

## 6. W2: GPT-2 at the ladder's peak learning rate (protocol §5.17)

**Pre-registered 2026-10-18, before the run.**

**Why.** To separate the two readings of GPT-2's larger decay gain, and to fix the peak rate of the 2B-token GPT-2 (Epic 3).

**The run.** W1 with `LR = 1.2e-3`; the floor stays at 6e-5, which is the ladder's floor at this peak (`WSD_LR_FLOOR = 0.05 × LR`). Verified 2026-10-18: the schedule equals the learning rate logged by a ladder run (XRB-F) at all 650 logged steps, to a relative 5e-16.

**Scored 2026-10-18** ([protocol, `p-w2`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-w2)). Colab A100-40GB, 2.7 h; 0.532B tokens, 33.7M parameters. Decay gain is measured from step 21,000.

| | prediction | called | result | verdict |
| --- | --- | ---: | --- | --- |
| W2.0 | no eval rises more than 0.2 nats above the previous one | 85% | largest rise +0.009 nats (step 15,000) | **HIT** |
| W2.1 | settled within ±2% of W1's 48.82 | 55% | **48.37** (mean of 48.65, 48.32, 48.14), −0.92% | **HIT** |
| W2.2 | settled at or below 48.82 | 50% | 48.37 | **HIT** |
| W2.3 | decay gain above W1's 0.271 | 65% | **0.338** (67.79 at step 21,000 to 48.37) | **HIT** |
| W2.4 | decay gain above every Fock arm's (0.218, F3.1's) | 60% | 0.338 | **HIT** |

**Decision: within 2% of W1.** The peak rate does not matter for GPT-2 at this budget. The published reference stands by the rule, and **the 2B-token GPT-2 runs at 1.2e-3**, so that its schedule is the ladder's in every respect.

**What it shows.**

- **The decay-gain gap is architectural.** On exactly the ladder's schedule GPT-2 gains 0.338 nats from the decay, against 0.18 to 0.22 for every Fock arm. The higher peak widened the gap rather than closing it.
- **At the end of the stable phase GPT-2 is behind** every comparable Fock arm except F3.1: GPT-2 67.79, against $L = 4$ 60.80, G3′ 63.45, G2 63.61, G3 65.22 (F3.1 71.81).
- **Against W2 (descriptive, not a change of reference):** $L = 4$ Fock (50.10) is +3.6%, G3′ (52.90) +9.4%, PMX (54.63) +12.9%. Against the cosine baseline the same arms read +0.6%, +6.2% and +9.7%. W2 is 2.9% below the cosine baseline. The rule does not act on this, but the protocol says any claim of parity should state it.

## 7. G2B: the matched GPT-2 at 2B tokens (protocol §5.17)

**Pre-registered 2026-10-18, before the run.** It is the reference for every 2B-token comparison (Epic 3), and it ran first because it does not depend on which Fock models are chosen.

**The run.** W2's settings plus `TOTAL_STEPS = 130_000`: 2.13B tokens at 16,384 per step. With the ladder's WSD fractions, warmup ends at step 6,500, the constant phase at 84,500, and the decay runs to 130,000. Evaluations every 500 steps.

- **Resume, fixed 2026-10-18.** A rolling checkpoint every 2,500 steps now stores the batch sampler's state. A test on a tiny CPU model resumed bit-identically to an uninterrupted run.
- **Data.** The 2.0B-token cache every 2B arm uses, read as random windows with replacement: about 1.07 passes on average. The protocol discloses this and does not enlarge the pool.

**Scored 2026-10-18** ([protocol, `p-g2b`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-g2b)). One uninterrupted Colab session, 38,709 s (10.75 h, 0.30 s per step), 260 evaluations. Decay gain is measured from step 84,500.

| | prediction | called | result | verdict |
| --- | --- | ---: | --- | --- |
| G2B.0 | no eval more than 0.2 nats above the previous one | 85% | largest rise +0.013 nats (step 15,000) | **HIT** |
| G2B.1 | settled between 41.0 and 45.5 | 65% | **42.69** (mean of 42.68, 42.70, 42.68) | **HIT** |
| G2B.2 | at least 5% below W2's 48.37 | 80% | −11.7% | **HIT** |
| G2B.3 | decay gain below W2's 0.338 (a longer constant phase leaves the model nearer its plateau) | 55% | **0.371** (61.84 at step 84,500 to 42.69) | **MISS** |

**The curve:**

| step | tokens | perplexity | phase |
| ---: | ---: | ---: | --- |
| 6,500 | 0.11B | 92.99 | end of warmup |
| 21,000 | 0.34B | 68.01 | constant |
| 32,500 | 0.53B | 64.97 | constant (W2 had settled at 48.37 by here, after its decay) |
| 50,000 | 0.82B | 63.17 | constant |
| 84,500 | 1.38B | 61.84 | end of the constant phase |
| 100,000 | 1.64B | 57.66 | decay |
| 120,000 | 1.97B | 45.92 | decay |
| 130,000 | 2.13B | 42.68 | end |

**Reading.**

- Four times the tokens buy 0.125 nats (48.37 to 42.69). At the end the model has seen about 63 tokens per parameter, well past compute-optimal for its size.
- **G2B.3 misses: the decay is worth more after a longer constant phase.** The constant phase flattens slowly (4.8% from step 32,500 to 84,500) and nearly all the final gain comes in the decay. At a fixed peak, the noise the constant rate holds above the floor does not shrink with time.
- **For Epic 3:** the Fock arms' 2B runs are compared against 42.69, and their constant-phase curves against this one at equal tokens.

**First use (protocol §5.13, `p-sr2-2b-result`, filed 2026-11-15).** SR2-2B, the no-wells core at 2.13B tokens, settles at 48.58 against G2B's 42.69. Its ratio to the matched GPT-2 goes from 1.206 at 0.53B tokens to 1.138. At the end of the stable phase SR2-2B is 3.3% ahead (59.81 against 61.84); the decay turns that into 13.8% behind.

## 8. Other matched GPT-2 runs, for orientation

C1 and C3 use W2's settings on other corpora and are recorded in protocol §5.13. G1 is registered in §5.10.

| run | protocol anchor | corpus | steps | settled |
| --- | --- | --- | ---: | ---: |
| C1, 2026-10-25 | `p-gpt2-ts` | TinyStories, 315M tokens | 19,250 | 4.47 |
| C3 (GPT-2-CODE), 2026-10-26 | `p-gpt2-code` | Python code (CodeParrot) | 19,250 | 4.96 |
| G1 (parameter-matched, L = 22) | §5.10 | OpenWebText | — | deferred, not run |

Both C1 and C3 report 33,691,776 parameters.

## 9. Unclear items

- **Parameter count in the CPU benchmark.** Protocol §5.13 (CPU inference benchmark, 2026-11-17) lists the matched GPT-2 as "36.6M parameters". Everywhere else it is 33.7M or 33,691,776. The source does not explain the difference.
- **G2B's decay gain.** The G2B scoring gives 0.371 nats. The SR2-2B result table gives 0.368 for G2B. The source does not reconcile the two.
- **The 3e-4 ratio for $L = 2$ `'none'`.** The §5.4 ratio table gives 1.508; the paragraph on the learning-rate sweep says "from 1.507". Both are quoted as written.
- **Hub repository name.** The published baseline's repository name contains `l2`, although the model has $L = 8$ layers. This note quotes the name as listed in protocol §7.3a.
