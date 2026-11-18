# The depth ladder protocol: an overview

**What this note is.** A short, readable summary of the rules the depth ladder was run under, written 2026-11-17. It covers why the programme exists, the configuration held fixed on every arm, the run queue and the completion plan, the token-budget correction, the pre-registered bands and how each one scored, the run-health conventions, and the publication audit of the first five arms. It distils §1, §2, §3, §3.3, §4, §5.3, §5.3a, §6 and §7 of the [protocol](../Depth_Ladder_and_Matched_Baseline_Protocol.md). Every number, verdict and date is copied from there; the protocol stays the full running log and the record of truth. The arms themselves, with their result tables, are in the companion note [Ladder_Early_Arms_Results.md](Ladder_Early_Arms_Results.md).

**Contents**

1. [Why the programme exists](#o-why) (protocol §1)
2. [The fixed configuration](#o-config) (protocol §2)
3. [The queue and the completion plan](#o-queue) (protocol §3, §3.3, §3b)
4. [The token-budget correction](#o-tokens) (protocol §4)
5. [Pre-registered bands, and how each scored](#o-bands) (protocol §5.3, §5.3a)
6. [Run health conventions](#o-health) (protocol §6 and the run-health entries of §5)
7. [The publication audit](#o-audit) (protocol §7)

**Terms used throughout.** PPL is validation perplexity; lower is better. An arm is one training run with one configuration. $L$ is the number of layers, $\Delta t$ the integration step per layer, and $T = L \Delta t$ the total integration time. **Settled** PPL is the mean of the last three evaluations, which come every 500 steps. A **pre-registered band** is a point prediction and a range, written down before the run (or before the reading it scores).

---

<a name="o-why"></a>
## 1. Why the programme exists (protocol §1)

Two failures of attribution made it necessary.

- **The warm-start arms could only screen.** Every arm grafted onto the step-28,500 checkpoint pays a consolidation cost that cannot be separated from its effect. §7.3 of the conservativity note measures that cost at about 1 PPL. Such arms answer "does adding X during the final anneal help", never "is X worth anything here".
- **The published GPT-2 baseline was not token-matched.** GPT-2 ran its first 14,000 steps at batch 8, so it saw **360.4M tokens** against Fock's **532.5M**. Every budget-fraction figure downstream inherited that.

Training from scratch fixes the first. Re-running GPT-2 at a constant batch fixes the second.

<a name="o-config"></a>
## 2. The fixed configuration (protocol §2)

Everything below is held identical on every ladder point, so that $L$ and the added mechanism are the only variables.

| setting | value |
| --- | --- |
| d / block / vocab | 384 / 512 / 50257 |
| V_θ | joint anisotropic Gaussian, K=8, ANISO_RANK=4 |
| ξ channels | 5, XI_CONTENT_ROUTE=True (Alternative E on) |
| V_φ | sparse top-k, TOP_K=16 |
| integrator | baoab_cfc_lowrank, lowrank_driver='gram' |
| total integration time | T = L × dt = 8, held fixed |
| schedule | WSD: warmup 5%, stable 60%, decay to a 1.50e-05 floor |
| steps / tokens | 32,500 / 532.5M at 16,384 tokens per step |
| learning rate | 3.0e-04 as written in §2; the ladder LR later became 1.2e-03 (see below) |
| exchange field, when present | 8 heads × d_k 48, λ pinned at 1.0, live readout |

- **WSD** is a warmup–stable–decay learning-rate schedule. At 32,500 steps it reads warmup to 1,625, stable to 21,125, decay to 32,500.
- **`lowrank_driver='gram'` is mandatory and is not the library default.** The default `'svd'` measured 3,368.9 ms per call against gram's 40.1 ms. Cell 5 asserts it.
- **The learning rate.** §2 still lists 3.0e-04. A sweep on the $L = 2$ `'none'` arm then found 1.2e-03 best (2.4e-03 was worse by 3.9%; a quadratic through the three points puts the vertex at 1.13e-03). Protocol §5.4 records the decision: "`LADDER_LR = 1.2e-03` is the ladder learning rate". The early 3e-04 arms are labelled as such.
- **Batch.** 16 sequences × gradient accumulation 2 = 32 on every arm, so the token budget is exactly 532,480,000.

<a name="o-queue"></a>
## 3. The queue and the completion plan (protocol §3, §3.3, §3b)

**The queue** numbers the runs. Run 1 went first regardless of order elsewhere: 2.7 hours, and it makes every other number defensible.

| # | run | what it settles | state as the protocol records it |
| ---: | --- | --- | --- |
| 1 | matched GPT-2, batch 32 throughout | removes the 1.48× token confound | DONE, §5.4: 49.81 settled |
| 2 | L=2 'attention' @3e-04 | — | DONE, §5.1: 68.33 (old LR; see run 9) |
| 3 | L=2 'none' @3e-04, then @1.2e-03 | what the exchange field contributes at fixed depth | DONE: 75.09 (§5.2) and 66.98 |
| 4 | L=4, matched | hops, versus "the L=8 arm was handicapped" | queued in the §3 table; settled 71.75 on 2026-09-29 (L=4 section) |
| 5 | L=2 'attention_potential' @1.2e-03 | the price of conservativity, parameter-matched | DONE, §5.8: 80.90 |
| 6 | L=2 'nonconservative', λ pinned @1.2e-03 | an unconstrained pointwise map | queued; deferred in §3.3 |
| 7 | L=1 'none' | one hop; but see §6.1 | DONE, §5.5: 87.09 |
| 8 | L=2 'none', reverse channel off | the conservative-only baseline (an architecture control, not a ladder point) | DONE, §5.6: 87.93 |
| 9 | L=2 'attention' @1.2e-03, a re-run | repairs the ladder's central comparison | DONE, §5.7: 63.51 |
| 10 | L=2 multi-ξ SPLM: no V_φ, no Fock | the PARF rung | queued; unblocked 2026-10-01 |
| 11 | L=2 Fock-SPLM: no V_φ, Fock on | replicates the Fock price on a V_φ-free base | queued; unblocked 2026-10-01 |

The queue cites run 3's 1.2e-03 result as "§5.4a". No heading of that name exists; the entry sits at the end of protocol §5.4 (anchor `p-gen2-none`).

Runs 10 and 11 form a two-by-two factorial with runs 3 and 8. Protocol §3.2 was **superseded 2026-10-11 by §5.21 (VF)**, the same factorial under live gradients. This note does not summarise §5.21.

**The completion plan, agreed 2026-09-27 (§3.3).** Four runs remained. Two address headline claims that each rested on a single measurement:

| headline claim | rests on | what removes the single point of failure |
| --- | --- | --- |
| geo+LN = 0.0003: geodesics are exact on the conservative arm | one dynamically clean layer at L=2 | run 4, L=4: three clean layers |
| the Fock mechanism costs +31.3% | one pair (run 3 against run 8) | runs 11 against 10 |

- The sequencing is by risk, not value. Run 4 needed one Cell 0 knob and started at once (launched 2026-09-27). Runs 10 and 11 needed new shared model code (`pair_potential='none'`), written and tested while run 4 trained.
- §3.3 corrected an earlier priority note that ranked runs 10 and 11 third. A per-step or ablation measure does not predict a trained-without outcome (§5.6), so they were promoted to follow run 4.
- Run 6 was deferred as a function-class bound that unblocks nothing.

**The D-series decision, 2026-09-28 (§3b).** When run 4 came in behind $L = 2$, the protocol opened four repairs, D1 to D4. Only D1 ($L = 4$ at fixed $\Delta t = 4$, so $T$ grows to 16) was authorised: "run D1, then return to the factorial unless D1 comes back positive". Details are in the L=4 section of the [results note](Ladder_Early_Arms_Results.md#e-l4). Whether D1 ran, and how it scored, is not recorded in the sections summarised here.

<a name="o-tokens"></a>
## 4. The token-budget correction (protocol §4)

| arm | steps | batch | tokens |
| --- | ---: | ---: | ---: |
| Fock, any ladder point | 32,500 | 32 throughout | 532.5M |
| GPT-2 as published | 32,500 | 8 then 32 | 360.4M |
| GPT-2, matched rerun | 32,500 | 32 throughout | 532.5M |

- GPT-2 crossed Fock's settled endpoint at **14.3%** of Fock's token budget, not the 46.7% recorded in checklist §9.2. That figure was a step fraction.
- Fock used **1.48×** the tokens to reach 81.58 and still lost. Correcting the budget makes the comparison **worse** for Fock.
- `training_log.jsonl` cannot detect this: it computes tokens as step × 16,384 whatever the batch. Only the console banner records the truth, which is why console logs are archived beside the JSONL.
- The notebook now **raises** on a batch mismatch, with an explicit `ALLOW_BATCH_MISMATCH` escape hatch, and prints a token-budget line on every run.
- The matched rerun settled at **49.81**. Protocol §5.4 adds that the comparison is matched on data and evaluation but not on hyperparameter effort, so every ratio is quoted "at these hyperparameters".

<a name="o-bands"></a>
## 5. Pre-registered bands, and how each scored (protocol §5.3, §5.3a)

**The idea.** Before an arm runs, the protocol records a point prediction and a band, the reasoning behind them, and "the quantity that could turn it": the one thing that, if it moved, would send the result outside the band. The settled value alone scores the band. Mid-run readings are recorded as readings, not revisions. §5.3 also states the expected ordering of the arms at the tuned learning rate, each arm removing one mechanism from the one above it: "Any inversion is a result."

**Revising a band.** §5.3a (2026-09-25) revised the `'attention_potential'` band *before* the run, because the stated mechanism was factually wrong: the arm's detaches cut the backward path, not the forward flow of context. Both bands stay on the record. The revision made the prediction worse (see the table), and §5.8 drew the rule: when correcting the reasoning behind a band, re-derive the number from the corrected reasoning rather than adjusting the old number, and record what the superseded band would have predicted.

**Lessons the scoring added.** Linear extrapolation of a monotone quantity has overshot every time; fit a saturating form, then treat even that as an upper bound (§5.2). A gap measured off-optimum is an upper bound (§5.7). An inference-time ablation overstates a mechanism (§5.6).

**The ledger of bands.** "Results note" links point to the [results note](Ladder_Early_Arms_Results.md).

| arm or reading | registered | result | verdict as recorded | details |
| --- | --- | --- | --- | --- |
| matched GPT-2 | 49.5, band 49.0–50.0 | 49.76 final, 49.81 settled | HIT, error +0.26 | protocol §5.4 |
| L=2 'attention' @3e-04 (run 2) | 65, band 63–68 | 68.33 | MISS, error +3.33 | [§5.1](Ladder_Early_Arms_Results.md#e-attn-3e4) |
| L=2 'none' @3e-04 (run 3), gap to 'attention' | gap holds at 20–30% | 9.9% | MISS | [§5.2](Ladder_Early_Arms_Results.md#e-none-3e4) |
| L=1 'none' @1.2e-03 (run 7) | 74–80 | 87.09 | MISS | [§5.5](Ladder_Early_Arms_Results.md#e-l1) |
| L=2 'none', reverse channel off (run 8) | 105, band 85–140 | 87.93 | BAND HIT, point high; the first band hit in the programme | [§5.6](Ladder_Early_Arms_Results.md#e-norc) |
| L=2 'attention' @1.2e-03 (run 9) | 63, band 59–68 | 63.51 | HIT, error +0.51 (+0.8%); the first point-accurate one | [§5.7](Ladder_Early_Arms_Results.md#e-attn) |
| L=2 'attention_potential' @1.2e-03 (run 5) | 66, band 62–72 (revised from 75–82, point 78) | 80.90 | MISS, error +14.90 (+22.6%); the superseded band would have hit | [§5.8](Ladder_Early_Arms_Results.md#e-attnpot) |
| L=4 'none' (run 4) | 59, band 55–65 | 71.75 | MISS | [L=4](Ladder_Early_Arms_Results.md#e-l4) |
| ω·dt median at L=2, 6b-13 | 3.3–4.2 | 3.796 | HIT; previewed twice, so not blind | [L=4](Ladder_Early_Arms_Results.md#e-l4) |
| probe (b) rglive at step 3,000 | full recovery 127.3–135.1; revised at step 1,300: at most 137.1 | 133.74 | HIT on both | [§5.8](Ladder_Early_Arms_Results.md#e-attnpot) |
| probe (b) rglive, settled | 67, band 63–72 | 61.11 settled (recorded in later tables) | no explicit scoring found in §5.8 | [§5.8](Ladder_Early_Arms_Results.md#e-attnpot) |
| P2.1 at step 3,000 | move if at most 154.1 | 127.73 | MOVE, by eight times the threshold | [§5.6](Ladder_Early_Arms_Results.md#e-norc) |
| F3.1 (P2.1 run to 32,500) | 67, band 60–79 | 57.76 | MISS on the good side | [§5.6](Ladder_Early_Arms_Results.md#e-norc) |
| L=4 live at step 3,000 | move if at most 135.5; about 125, band 112–135 | 119.77 | MOVE and HIT | [L=4](Ladder_Early_Arms_Results.md#e-l4) |
| L=4 live, settled | 57, band 52–63 | 50.10 | MISS on the good side | [L=4](Ladder_Early_Arms_Results.md#e-l4) |
| CG3 forecastability on L=4 live | frozen verdict rule | (b) 0.759, (a⊥) +0.526 | FORECASTABLE | [L=4](Ladder_Early_Arms_Results.md#e-l4) |
| C1: gate scaled above λ = 1 | PPL rises monotonically above λ = 1 | rises on both sides of 1.0 | HIT; question closed (2026-10-05) | [L=4](Ladder_Early_Arms_Results.md#e-l4) |
| D1: L=4 at fixed dt | 62–70 | not in the summarised sections | — | [L=4](Ladder_Early_Arms_Results.md#e-l4) |

<a name="o-health"></a>
## 6. Run health conventions (protocol §6 and the run-health entries of §5)

Each completed arm reports the same health items. What they mean, and their known blind spots:

- **Clip-hit rate.** The share of logged steps whose gradient norm exceeded the global clip, `GRAD_CLIP = 1.0` (logged every 50 steps, so 650 logged steps per full run). §6 calls the clip a depth-dependent intervention: it fired on 4.2% of steps at $L = 2$ `'none'` and 36.2% at $L = 8$. **From $L = 4$ onward every ladder point must report it.** The earlier arms report it too.
- **Per-group clip overrides.** Training uses the global clip plus nine per-group overrides. The tightest is 0.1 on the reverse-channel gate `reverse_channel_scale`, which is clipped by a median 10–20× on essentially every step of every arm that has it. The thresholds were set on $L = 8$ and $L = 16$ runs and never re-derived for the ladder. The C-series closed the question on 2026-10-05: the trained gate sits at a sharp optimum (C1), and Adam absorbs the rescale, so the clip is cosmetic for the endpoint (C2).
- **Watchdog and spike captures.** Counted per run. The watchdog is structurally blind to `reverse_channel_scale` and `reverse_ch`, which are excluded from its aggregate (C6). State this wherever run health is claimed.
- **Stiffness, $\omega \Delta t$.** The resonance monitor reports how far, in radians, a stiff mode of angular frequency $\omega$ rotates in one layer step. Above 2 an explicit leapfrog step would amplify; the low-rank integrator handles those modes exactly, so this is a stiffness reading, not an instability. **The monitor was blind for every `baoab_cfc_lowrank` run until the 2026-09-28 fix (`semsimula-diag` commit `409175b`).** Earlier arms show "EMPTY SUMMARY": a missing diagnostic, not a passing one.
- **Causal-leak audits (SCAF).** The live-gradient arms report seven audits (5k, 10k, …, 30k, 32.5k). Each CLEAN audit is reported with its future perturbation (exactly 0.0 on the arms summarised here) and its leak tax in nats.
- **Weight-space proxies.** `bproj_sig` is a largest-singular-value proxy on the low-rank factor. On run 4 it tracked the tail of the $\omega \Delta t$ distribution, not the median. `share_max` is the exchange field's share of the force.
- **Single seed.** Every ladder arm is one seed.

Operational rules learned during the runs:

- The console output is the only home of the per-step telemetry, the clip-group lines and the resonance readings. Save it before every restart.
- Every knob that changes an arm must reach the variant tag. An untagged arm shares a Drive folder and silently resumes from the wrong checkpoint. `GRAD_CLIP_OVERRIDES` does not reach the tag, which is a live trap for any run that varies a clip threshold.
- A completed-run guard was added 2026-09-29, after Cell 6 resumed the finished run 3 by accident. Cell 6 now refuses to extend a run whose log holds an eval at or past the final step, unless `ALLOW_EXTEND_COMPLETED_RUN` is set. Run 3's folder was repaired byte-exact and its 66.98 stands.

<a name="o-audit"></a>
## 7. The publication audit (protocol §7)

**Audit of 2026-09-27**, over the five completed arms: the matched GPT-2 and the four $L = 2$ Fock arms at 1.2e-03.

**Every headline number is recomputable from the shipped artefacts.** The mean of the last three `val_ppl` entries in each training log reproduces the settled value in `hf_model_cards/_ladder/ladder.json` to the second decimal:

| arm | best step / PPL in checkpoint | last three evals | settled |
| --- | --- | --- | ---: |
| gpt2-matched | 32,500 / 49.76 | 49.87, 49.80, 49.76 | 49.81 |
| attention | 31,000 / 61.49 | 63.52, 63.58, 63.42 | 63.51 |
| attention_potential | 31,000 / 79.14 | 80.74, 80.46, 81.50 | 80.90 |
| none | 31,500 / 66.56 | 66.56, 66.75, 67.63 | 66.98 |
| none no-RC | 31,000 / 85.90 | 87.53, 87.44, 88.82 | 87.93 |

- **What each repo ships:** two stripped checkpoints (the endpoint and step 500, no optimizer state), the full model and training configuration inside every Fock checkpoint, complete JSONL training logs, console logs, and probe logs.
- **Parameter counts on the cards are right.** Summing a state dict over-counts because it carries buffers: +50,257 plus three scalars on every Fock arm, +2,097,152 on GPT-2.
- **Four gaps were found.** Gaps 1 and 2, missing probe logs on the `'none'` arm, were **closed 2026-09-27** in one probe session with no retraining. Gap 3, a stale cross-reference in `gpt2_baseline_summary.json`, and gap 4, GPT-2's second checkpoint being step 25,000 rather than step 500, were handled on the cards.
- **Uploaded 2026-09-27:** five public repos under `dimitarpg13/` (`semsimula-ladder-owt-d384-l2-gpt2-matched`, `-attention`, `-attention-potential`, `-none`, `-none-norc`), plus the collection *Semantic Simulation — CfC+BAOAB Mechanism Ladder*, with the Verlet-instability repo as its prologue. The total is about 2.6 GB. The flagship checkpoint was downloaded back and confirmed at step 31,500, PPL 66.56.
- **Hub limits found:** a collection description is capped at 150 characters, an item note at 500, and the collection slug gets a 24-hex suffix.
- **The "Verlet-era" label is too narrow** (§7.3b): fourteen of the fifteen models in the older SPLM family are not Verlet (thirteen semi-implicit Euler, one first-order gradient flow). What they share is being pre-CfC. Recorded as unreviewed; no text was changed then.
