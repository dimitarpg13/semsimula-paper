# The well screen at 0.5B and what the wells' precision needs

**What this note is.** A short, readable summary of five entries of the depth-ladder protocol, written 2026-11-17. Three are the well screen at 0.5B tokens on OpenWebText: its setup and calls ([protocol, `p-well-screen-setup`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-well-screen-setup)), the K4-R6 result ([protocol, `p-k4-r6-result`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-k4-r6-result)) and the K2-R6 result ([protocol, `p-k2-r6-result`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-k2-r6-result)). Two are earlier measurements on SR2's well bank without retraining: what its projections need, with two proposed smaller twins ([protocol, `p-precision-twins-setup`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-precision-twins-setup)), and whether the wells can share one precision core ([protocol, `p-precision-cpu-result`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-precision-cpu-result)). Every number, verdict and date is copied from the [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md), which keeps the full log and remains the record of truth.

Related notes: the conjecture that the bank is a memory store of the second-order model, and the probes that test it, are in [Second_Order_Memory_Probes.md](Second_Order_Memory_Probes.md). The CPU speed benchmarks of these models are in [CPU_Inference_Benchmark.md](CPU_Inference_Benchmark.md). The decay-gain argument and the call A1 are in [SR2_L1_and_Decay_Noise.md](SR2_L1_and_Decay_Noise.md). SR2-2B is in [Two_Billion_Token_Runs.md](Two_Billion_Token_Runs.md). The written-up shared-core reading is in [the well-count note, section 8](../Minimum_Well_Count_for_Conservative_Models.md#n-shared-core).

**Status in one line.** K4-R6 settled 59.62 (+2.2% against SR2; KS1 MISS by 0.13, KS2 to KS5 HIT); K2-R6 settled 65.12 (+11.7%; KS6 to KS8 MISS, KS9 and A1 HIT); K4-R6 is the candidate smaller bank; no shared-core twin is filed, and the twins PA and PB stay proposed, unconfirmed and not run.

---

## Contents

1. [Terms used](#terms)
2. [The well screen: setup and calls](#screen-setup)
3. [K4-R6: result](#k4-r6)
4. [K2-R6: result](#k2-r6)
5. [What the bank's projections need, and the twins PA and PB](#precision-twins)
6. [Can the wells share one precision core?](#shared-core)
7. [What stands and what is open](#open)

---

<a name="terms"></a>
## 1. Terms used

- **SR2.** The conservative second-order core at $L = 2$ layers, trained on 0.5B tokens of OpenWebText (32,500 steps). It settled at perplexity 58.32 and has 76,572,110 parameters.
- **The well bank.** The learned potential $V_\theta$ is one bank of Gaussian wells shared by both layers. SR2 has 8 wells of rank 4, and the bank holds 35,427,080 of its parameters. Each well has a centre $\mu_k$, a diagonal precision $a_k$, a low-rank precision factor $B_k$ and a weight $w_k$, all computed from the context by dense maps (`mu_proj`, `a_proj`, `B_proj`, `w_proj`).
- **Screen arms.** "K4-R6" has 4 wells of rank 6 (`V_THETA_WELLS_PER_HEAD = 4`, `ANISO_RANK = 6`); K2-R6 and K8-R6 have 2 and 8.
- **Directions.** A well of rank $R$ pushes along $R$ directions at a token, so a bank of $K$ wells pushes along at most $K R$: 32 for SR2, 24 for K4-R6, 12 for K2-R6. The **direction count** projects a layer onto $s$ shared directions and records the cost; $r^{\ast}$ is the smallest $s$ that stays within 0.5%.
- **Keep-k.** Remove all but the $k$ strongest wells of one layer, without retraining, and record the perplexity change. Removal overstates what a model trained with the smaller bank loses.
- **Settled perplexity.** The mean of the evaluations at steps 31,500 to 32,500. **Paired gap:** both runs score the same validation batches, so the difference in validation loss, in nats, cancels the evaluation scatter. 0.01 nats is about 1%.
- **Phases.** The learning-rate schedule (WSD) warms up to step 1,625, holds a stable phase to 21,125 and decays to 32,500. The **decay gain** is the mean validation log perplexity of the last five evaluations up to the decay start, minus the settled value.
- **Weight entropy** $n_{\mathrm{eff}}$: the effective number of wells in use, from the force weights.
- **FO-b** is SR2's first-order twin: the same model trained as an overdamped gradient flow, without inertia.

---

<a name="screen-setup"></a>
## 2. The well screen: setup and calls (protocol `p-well-screen-setup`)

**Set up 2026-11-15.** The order was agreed on 2026-11-14: the pool check, then K4-R6, then the next arm by the rule below. **KS1 to KS5 and the next-arm rule were confirmed by the author on 2026-11-15, before K4-R6 trained.**

**The question.** Can a bank of fewer wells of higher rank replace SR2's eight wells of rank 4 on OpenWebText? It is the first of two questions to settle before any 4B run; the second is the decay schedule. K4-R6 cuts the bank from 35,427,080 parameters to 23,616,772.

**What the calls rest on.**

- On SR2's trained checkpoints, layer 0 fills its 32 directions. Projecting it onto 24 costs +5.0% at 0.5B and +3.9% at 2B, and the need is flat after step 16,500. Layer 1 needs 8 directions at 0.5B and 12 at the end of the 2B decay.
- Removal overstates what a model trained with the smaller budget loses. On code, K2-L8-CODE, trained with 8 directions per token, settled 0.057 nats behind L8-CODE, where removing all but two wells cost about 20 times as much. Rank 6 closed part of the rest (K2-L8-R6-CODE, +3.4% against L8-CODE; see [Code_Depth_Runs.md](Code_Depth_Runs.md)).
- Layer 0's need is the same at 0.5B and 2B, so a 0.5B screen should pick a well count for layer 0 that carries to larger budgets. Layer 1's need grew in the 2B decay, so an arm that passes with little margin is checked again at the end of a longer decay.

**Step 1, the pool check.** SR2 trained on a training file (`openwebtext_train_2000M.npy`) that has since been deleted. Every screen reads the file rebuilt on 2026-11-12. If the two hold the same tokens, the screens draw SR2's batches step for step and every comparison is paired. The check runs SR2's settings for 250 steps and compares the training loss with SR2's log. Registered thresholds: all five losses within 0.005 means the same pool; any off by 0.02 or more, or anything between the two thresholds, means a different pool.

| step | 50 | 100 | 150 | 200 | 250 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SR2's log | 7.6064 | 7.4470 | 7.2886 | 7.0446 | 6.8386 |
| the check (2026-11-15, A100 40 GB) | 7.6116 | 7.4505 | 7.2912 | 7.0469 | 6.8409 |
| difference | +0.0052 | +0.0035 | +0.0026 | +0.0023 | +0.0023 |

- **Scored by the letter: different pool.** Step 50 is 0.0002 past the 0.005 threshold. The author decided on 2026-11-15, after seeing the numbers, to score by the letter, keep the caveat on every comparison with SR2, and not rerun SR2 (about 9.5 h).
- **The evidence, recorded beside the caveat, is that the draws are the same.** Over steps 250 to 1,500, SR2's logged training loss scatters by 0.043 nats about a smooth fit, so two different draws would differ by about 0.06, with either sign. All five differences have the same sign, stay below 0.0055 and shrink with training. That is what the same batches under slightly different arithmetic give. The internal readings separate more than the losses (`v_reg` 0.2233 against 0.1599 at step 200; gradient norm 0.46 against 0.69 at step 250). So the code and hardware changes since October move the arithmetic by more than the threshold assumed; it was set too tight for a check across code versions.
- The same caveat applies to SR2-L1 and to the open item in `p-sr2-2b-setup`.

**Step 2, K4-R6.** Cell 0 is SR2's six deltas plus `V_THETA_WELLS_PER_HEAD = 4` and `ANISO_RANK = 6`. 32,500 steps, the ladder's WSD fractions, peak rate 1.2e-3, floor 6e-5, float32, seed 0, OpenWebText. Built locally from HEAD `ae280af`: tag `…aniso_dcvt5x4_ar6_vtjoint_cgqk_norc_vplive_xilive_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn`, 64,761,802 parameters, 23,616,772 of them in $V_\theta$; everything outside the bank is unchanged (41,145,030). One forward and backward pass gave loss 7.2881 and gradient norm 1.06, and took 0.86 times as long as SR2's on the CPU. Expected 8 to 9.5 h, one session.

**Launch, 2026-11-15,** on an A100 40 GB from GitHub `ae280af`. Training peaked at 32.0 GB at batch $16 \times 2$. Evaluation without layer recompute needs more than this card holds, so Cell 0 gained `EVAL_LAYER_CHECKPOINT = True` (recompute during evaluation only; training arithmetic and tag unchanged; nine changes from the defaults). A first start without it was stopped before step 500, so before any evaluation or checkpoint, and the run started fresh.

**Calls** against SR2 at 0.5B (settled 58.32; evaluation scatter about 2%):

| id | prediction | called |
| --- | --- | --- |
| KS1 | K4-R6's settled perplexity is within +2% of SR2's (59.49 or less) | 45% |
| KS2 | within +5% (61.24 or less): training with 24 directions costs less than projecting SR2 onto 24 does | 80% |
| KS3 | on K4-R6's best checkpoint, layer 0 needs all four wells (keeping its three strongest costs more than 0.5%) | 85% |
| KS4 | layer 0 needs more than 20 of its 24 directions (projecting onto 20 costs more than 0.5%) | 75% |
| KS5 | layer 1 needs at most two wells (keeping its two strongest costs at most 0.5%) | 70% |

**The next-arm rule (confirmed 2026-11-15).** If KS1 hits: K2-R6 (12 directions; 52,955,336 parameters, tag `…dcvt5x2_ar6…`), the lower bound and the arm A1 is scored on. If KS1 misses: K8-R6 (48 directions; 88,374,734 parameters, tag `…dcvt5x8_ar6…`), to ask whether layer 0 uses more than SR2's 32. K3-R6 is not run in that case: projecting SR2's layer 0 onto 18 directions costs about +10%.

**What the screen cannot settle.** One seed per arm. The decay branches run on SR2-2B's checkpoints, which have eight wells of rank 4, so a schedule chosen there carries to a new well count only by assumption.

---

<a name="k4-r6"></a>
## 3. K4-R6: result (protocol `p-k4-r6-result`)

**Scored 2026-11-16: settled 59.62, +2.2% against SR2. KS1 misses by 0.13; KS2 to KS5 hit. Next arm: K2-R6, by the author's decision after the result, in place of the rule's K8-R6.**

**The run.** 32,500 steps on the A100 40 GB in 35,813 s (1.10 s per step); training peak 31.2 GB, evaluation peak 30.57 GB. No watchdog trigger, no spike capture, SCAF (the leak audit) CLEAN at every audit. Output `L2_SR2_K4_R6_OWT_32500steps_output.txt` with the cell outputs, in [the arm's results folder](../../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x4_ar6_vtjoint_cgqk_norc_vplive_xilive_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/).

| step | phase | SR2 | K4-R6 | K4-R6 − SR2 (nats) |
| ---: | --- | ---: | ---: | ---: |
| 500 | warm-up | 461.8 | 452.5 | −0.021 |
| 5,000 | stable | 108.4 | 107.2 | −0.011 |
| 10,000 | stable | 86.0 | 82.8 | −0.037 |
| 20,000 | stable | 70.8 | 70.7 | −0.002 |
| 25,000 | decay | 65.9 | 66.6 | +0.010 |
| 30,000 | decay | 58.1 | 59.4 | +0.023 |
| settled (31,500–32,500) | | 58.32 | 59.62 | +0.022 |

- **The gap is measured, not noise.** Over the last seven evaluations it is +0.0219 nats with a standard deviation of 0.0007. Both runs settled (fitted change +0.001 nats against a residual of 0.013). Best 57.71 at step 31,000 (SR2 56.49, also at 31,000).
- **Where the gap forms.** K4-R6 leads in the stable phase ($-0.017$ nats on average over steps 5,000 to 21,000, SD 0.014). Fitted per phase, in nats per 1,000 steps, the gap falls at $-0.0061$ over steps 2,000 to 10,000, rises at +0.0037 over 10,000 to 21,000 and rises at +0.0013 in the decay (residuals 0.004; lag-1 autocorrelation 0.93 in the stable phase, so it drifts rather than scatters). SR2 overtakes K4-R6 from about step 10,000, and the decay continues that trend at about a third of the late-stable rate. The decay gain is 0.175 nats against SR2's 0.193; that is the trend carried through the decay, not a loss the decay adds. **The deficit does not start at the decay** (corrected the same day; the first filing said it did). For comparison, over steps 10,000 to 21,000 and over the decay: SR2-L1 against SR2 +0.0081 and +0.0054; FO-b against SR2 +0.0012 and +0.0027, the one arm whose deficit grows faster in the decay than before it.

| id | prediction | called | result | scored |
| --- | --- | --- | --- | --- |
| KS1 | settled within +2% of SR2 (59.49 or less) | 45% | 59.62, +2.2% | MISS (by 0.13) |
| KS2 | settled within +5% of SR2 (61.24 or less) | 80% | 59.62 | HIT |
| KS3 | layer 0 needs all four wells (keep-3 costs more than 0.5%) | 85% | keep-3 +29.5% | HIT |
| KS4 | layer 0 needs more than 20 of 24 directions | 75% | 20 directions +3.7% | HIT |
| KS5 | layer 1 needs at most two wells (keep-2 at most 0.5%) | 70% | keep-2 −0.0%, keep-1 −0.0% | HIT |

**The pool caveat holds.** The training loss agrees with the same draws: at step 50 it is 7.5981 against SR2's 7.6064, and over the first 1,000 steps the training gap has a standard deviation of 0.005, below the 0.06 that different draws would give.

**The wells on the CPU** (best checkpoint, step 31,000; 16 validation windows; trained perplexity 59.49 on them against SR2's 58.28; files `*k4r6_owt*` in [`results/well_screen/`](../../notebooks/conservative_arch/scaleup/results/well_screen/)):

| reading | layer 0 | layer 1 |
| --- | --- | --- |
| keep the 1 / 2 / 3 strongest wells | +1,293% / +249% / +29.5% | keep-1 −0.0% |
| force weight | well 0 89.7%, the others 3.4–3.5% (SR2 81% and 2.3–3.7%) | one well 100.0% |
| project onto shared directions | 16 / 18 / 20 / 22: +12.3% / +7.0% / +3.7% / +1.4%, so r* above 22 | 4: +17.2%; 6: −0.0%, so r* = 6 (SR2 8) |
| weight entropy n_eff | 2.18 (SR2 3.8) | 1.01 (SR2 1.5) |

Trained with 24 directions, layer 0 fills all of them, as SR2's layer 0 fills its 32. Layer 1's directions fit inside one well when the well is wide enough.

**The cells** (best checkpoint; SR2 in brackets). Cell 6b-13: median $\omega \Delta t$ (how far a stiff mode turns in one step) 6.83 [7.55]; layer 0 7.80 [8.93], layer 1 4.69 [5.46]; 99.4% past the explicit wall. Fewer, wider wells are a little softer. The cell's "pre-registered band" MISS is its depth-ladder band, not a call of this screen. Cell 6b-7: Gate 0 PASS; Gate 1 (velocity reset) +2.83 perplexity [+2.22]; Gate 2 at $N = 3$ +0.243 nats [+0.299]; Gate 3 at $N = 3$ +0.653 nats [+0.405]; MAPS, as for SR2. Cell 6b-9: R(geo) 0.601 [0.620]. Cell 6b-12: layer 0 median deflection 0.060 [0.053], layer 1 none. Cell 7: the registers ablate to +0.00. FLOW-R has not been run on K4-R6.

**Reading.** A bank of 23.6M parameters, against SR2's 35.4M, costs 2.2% at 0.5B. Layer 0 uses all 24 directions and loses 29.5% without its fourth well, so 24 is below what layer 0 would use. Layer 1 needs six directions and one well. Whether layer 0's shortfall grows with the budget is left open: SR2-2B's layer 0 need was flat from step 16,500, while K4-R6 lost ground steadily from step 10,000.

**Next arm: K2-R6 (the author's decision, 2026-11-16).** The rule sends a KS1 miss to K8-R6; the author chose K2-R6 instead. It is the lower bound on the well count and the arm A1 is scored on. By removal, projecting K4-R6's layer 0 onto 12 directions costs +38.6% and keeping its two strongest wells +249%, both upper bounds. K8-R6 is deferred, not dropped.

**K2-R6 calls (proposed and confirmed by the author on 2026-11-16, before K2-R6 trained):**

| id | prediction | called |
| --- | --- | --- |
| KS6 | K2-R6 settles within +5% of SR2 (61.24 or less) | 35% |
| KS7 | within +10% of SR2 (64.15 or less) | 80% |
| KS8 | the K2-R6 − SR2 gap in the stable phase (mean over steps 5,000 to 21,000) is at most +0.01 nats: like K4-R6, it loses in the decay | 55% |
| KS9 | layer 1 of K2-R6 needs one well (keep-1 costs at most 0.5%) | 75% |

**K2-R6 launch, 2026-11-16,** on an A100 80 GB from GitHub `6ae745f` (notebook and scaleup code unchanged since `ae280af`). K4-R6's nine changes with 2 wells in place of 4; tag `…dcvt5x2_ar6…`; 52,955,336 parameters (11,810,306 in $V_\theta$); step 50 loss 7.5949 (K4-R6 7.5981, SR2 7.6064); about 0.8 s per step.

**The decay gain against the gradient noise** (read 2026-11-16; the argument is in [SR2_L1_and_Decay_Noise.md](SR2_L1_and_Decay_Noise.md)). The step-22,500 checkpoint is 1,375 steps into the decay at 97% of the peak rate. Output: [`results/decay_noise/k4r6/adam_noise_decay_gain.txt`](../../notebooks/conservative_arch/scaleup/results/decay_noise/k4r6/adam_noise_decay_gain.txt).

| checkpoint | sum of sqrt(v) | predicted gain | measured gain | measured over predicted | median abs(m) over sqrt(v) |
| --- | ---: | ---: | ---: | ---: | ---: |
| K4-R6, step 22,500 (97% of peak) | 861.2 | 0.258 | 0.175 | 0.68 | 0.131 |
| K4-R6, best (31,000; end of decay) | 849.9 | 0.255 | 0.175 | 0.69 | 0.131 |
| SR2, best (31,000; end of decay) | 836.1 | 0.251 | 0.193 | 0.77 | 0.143 |

The noise argument does not explain K4-R6's smaller decay gain: it carries slightly more optimizer noise than SR2 yet drains less. Its ratio, 0.68, is below the 0.72 to 0.74 of SR2-L1, SR2-2B and GPT-2 2B. The pre-decay constant was fitted on three points, so 0.68 is a first point outside the band, not a refutation. The 0.02-nat shortfall is read as the continuation of the slower learning that began at step 10,000. By block, the precision factors (`B_proj`) carry 186 against SR2's 182, and the centres and diagonals 119 against 132: per coordinate, the smaller bank's maps are noisier.

**Does the first-order model need its wells as SR2 does?** Read 2026-11-16 on FO-b-OWT's best checkpoint (the author's question: would a smaller bank cost FO-b more than SR2?). Same probes and windows; trained perplexity 60.09 against SR2's 58.28; files `*fob_owt*` in `results/well_screen/`.

| reading, layer 0 (8 wells of rank 4) | SR2 | FO-b |
| --- | ---: | ---: |
| force weight of the strongest well | 81.0% | 60.7% |
| weight entropy n_eff | 3.8 | 6.9 |
| occupancy of the strongest well (median) | 0.99 | 0.50 |
| distance to that well (median; the state's norm is about 3) | 3.1 | 6.4 |
| keep the 1 / 2 / 3 strongest wells | +1,279% / +729% / +276% | +730% / +398% / +217% |
| keep 4 / 5 / 6 / 7 | +130% / +53% / +28% / +10% | +128% / +60% / +26% / +7.1% |
| project onto 8 / 12 / 16 shared directions | +157% / +46% / +19% | +48% / +22% / +12% |
| project onto 20 / 24 / 28 | +9.8% / +5.0% / +1.8% | +6.8% / +3.5% / +1.4% |
| energy in the top 8 directions (median) | 79% | 66% |

FO-b's layer 0 also fills its budget but degrades more gently: at 8 and 12 directions removal costs SR2 3.3 and 2.1 times what it costs FO-b. FO-b's layer 1 runs on one well and four directions or fewer. By removal, at K2-R6's 12 directions the first-order model should lose less than SR2, not more; this is a removal reading on two models trained with 32 directions.

---

<a name="k2-r6"></a>
## 4. K2-R6: result (protocol `p-k2-r6-result`)

**Scored 2026-11-17: settled 65.12, +11.7% against SR2. KS6, KS7 and KS8 miss; KS9 and A1 hit. Two wells of rank 6 cost a model five times what four wells did.** 32,500 steps in 7.85 h (0.87 s per step on an A100 80 GB); training peak 29.4 GB, evaluation peak 28.8 GB; clean finish, SCAF CLEAN. Log and diagnostics in [the arm's results folder](../../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x2_ar6_vtjoint_cgqk_norc_vplive_xilive_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/).

| step | SR2 | K4-R6 | K2-R6 |
| ---: | ---: | ---: | ---: |
| 2,000 | 152.73 | 152.90 | 157.81 |
| 5,000 | 108.36 | 107.17 | 113.86 |
| 10,000 | 85.96 | 82.84 | 92.23 |
| 21,000 (decay starts) | 70.94 | 71.44 | 77.99 |
| 28,000 | 60.72 | 61.73 | 67.40 |
| settled (31,500 to 32,500) | 58.32 | 59.61 | 65.12 |
| best | 56.49 at 31,000 | 57.71 at 31,000 | 63.07 at 31,000 |

K4-R6's settled value is 59.61 in this table and 59.62 in its own entry; both are as the protocol gives them.

- **The gap opened early and widened in every phase:** +0.0037 nats per 1,000 steps over steps 2,000 to 10,000, +0.0021 over 10,000 to 21,000 and +0.0016 in the decay. The last seven evaluations' gap is +0.1106 (SD 0.0009). K2-R6 was never ahead.
- **The decay is not where it loses.** Its decay gain is 0.176, as K4-R6's 0.175, against SR2's 0.193. A call KS10 would have hit, but it was never filed.

| id | call | result | verdict |
| --- | --- | --- | --- |
| KS6 | settles within +5% of SR2 (61.24 or less), 35% | 65.12, +11.7% | MISS |
| KS7 | within +10% (64.15 or less), 80% | 65.12 | MISS |
| KS8 | stable-phase gap (mean over 5,000 to 21,000) at most +0.01 nats, 55% | +0.0750 | MISS |
| KS9 | layer 1 needs one well (keep-1 at most 0.5%), 75% | +0.0% | HIT |
| A1 | K2-R6 at L = 2 settles below SR2-L1's 69.15, 65% | 65.12 | HIT |

| cell reading | K2-R6 | K4-R6 | SR2 |
| --- | --- | --- | --- |
| 6b-13, ω·Δt median (layer 0, layer 1) | 6.11 (7.55, 4.11) | 6.83 (7.80, 4.69) | 7.55 |
| 6b-7, Gate 1 | +4.92 | +2.83 | |
| 6b-7, Gate 3 at N = 3 (nats) | +0.790 | 0.653 | 0.405 |
| 6b-9, R(geo) at layer 0 | 0.760 | 0.601 | |

The refinement penalty grows as the bank shrinks.

**The wells on the CPU (2026-11-17;** files `*k2r6_owt*` in `results/well_screen/`; 16 validation windows, base 65.72). Layer 0: the heavier well has 94.3% of the force weight, and keeping only it costs +495%; $\omega \Delta t$ 7.54, occupancy 0.95, $n_{\mathrm{eff}}$ 1.58. All 12 directions are used: 10 cost +10.9%, 8 cost +40%, 6 cost +145%. Layer 1: one well, keep-1 +0.0%, 6 directions suffice; $\omega \Delta t$ 4.13.

**Decay gain against the gradient noise** (`results/decay_noise/k2r6/`). At step 22,500: sum of $\sqrt{v}$ 868.6, predicted 0.261, measured 0.176, ratio **0.67**. At the best checkpoint: 844.5, 0.253, ratio 0.70. As for K4-R6 (0.68 and 0.69), below the 0.72 to 0.74 band: the smaller banks drain less than their noise predicts.

**Speed.** Training speed is not comparable across these runs: they ran on different GPUs, and SR2's run predates the speed switches (1.72 s per step). On the CPU, redone with the analytic force, K2-R6 is 1.30 times faster than SR2 at 512 tokens and K4-R6 1.14 times, and the gap closes at long contexts. A first run's 1.84 and 1.62 timed the models without their pair force. Details in [CPU_Inference_Benchmark.md](CPU_Inference_Benchmark.md).

**Reading.** The bank's size is not free. Going from 8 wells of rank 4 (32 directions) to 4 of rank 6 (24) cost +2.2%; going to 2 of rank 6 (12) cost +11.7%. This is in line with the direction counts: projecting SR2's layer 0 onto 24 of its 32 directions costs +5.0% and onto 12 costs +46%. Fewer wells cannot carry the bank's work, even with a higher rank. The bank's matrix products are about 45% of SR2's CPU time, but a cut to 12 directions is too expensive in perplexity. **K4-R6 is the candidate**; its same-machine speed was queued.

---

<a name="precision-twins"></a>
## 5. What the bank's projections need, and the twins PA and PB (protocol `p-precision-twins-setup`)

**Measured and set up 2026-11-13** (the author's question: can the model shrink?). **The twins and their calls are proposed and await the author's confirmation.**

**Where the bank's parameters are.** The dense maps read the concatenated context channels (1,920 wide): `B_proj` to the low-rank precision factors (23,605,248), `mu_proj` to the centres (5,901,312), `a_proj` to the diagonal precisions (5,901,312), `w_proj` to the weights (15,368), plus the depth code (3,840). The matched GPT-2 has 33.7M parameters in all.

**Measured without retraining** on SR2's best checkpoint (16 validation windows to estimate, 16 others to score; trained perplexity 59.63 on the latter; `debug/vtheta_precision_ablation.py` and `debug/vtheta_projection_svd.py`, outputs in `results/precision_ablation/`):

| replacement (both layers unless stated) | perplexity change |
| --- | ---: |
| B fixed per well and layer (best rank-4 fit to its force-weighted mean) | +1,296% |
| the same, rescaled to the trained norm | +1,118% |
| B removed (diagonal precision only) | +1,173% |
| every well takes the strongest well's B, layer 0 / layer 1 only | +1,252% / +5.7% |
| B fixed, layer 0 / layer 1 only | +846% / +73% |
| a fixed per well and layer | +1.2% |
| B_proj truncated to rank 256 / 512 / 1,024 (SVD) | +102% / +41% / +7.0% |
| mu_proj truncated to rank 512 / 1,024 (input-aware) | +12% / +1.6% |
| a_proj truncated to rank 64 | +1.2% |

- **$B$ carries the context.** Fixing its orientation costs about as much as removing it, so all of $B$'s value lies in its dependence on the context. The term $B_k B_k^{\top}(h - \mu_k)$ projects the state's offset from the well onto rank-4 directions chosen per token: the bank's main path from the context into the state.
- **Layer 0 needs $B$ per well and per token**; layer 1 is far more compressible (a shared $B$ costs +5.7%).
- **`B_proj` is close to full rank:** half its squared singular values lie outside the top 256 directions.
- **$a$ is nearly context-free.**
- Truncation and replacement without retraining are upper bounds on the cost.

**The proposed twins** (both: SR2's Cell 0 plus a switch, 32,500 steps, OpenWebText, seed 0, paired with SR2; about 10 h each):

- **PA:** `a_proj` replaced by one learned $a_k$ per well and layer ($2 \times 8 \times 384$ = 6,144 parameters). 70,676,942 parameters, 5,894,986 fewer than SR2.
- **PB:** PA plus a bottleneck in `B_proj` (1,920 to 512 to 12,288; no bias in the first layer), 7,286,784 parameters instead of 23,605,248. 54,358,478 parameters, 22,213,632 fewer than SR2. Width 512 is the narrowest that still saves most of `B_proj`; its truncation cost (+41%) is the risk it measures. Width 1,024 would be safe but saves only 9.1M.

| id | prediction | called |
| --- | --- | --- |
| PA1 | PA settles within +1% of SR2 (58.32) | 75% |
| PB1 | PB settles within +2% of SR2 | 35% |
| PB2 | PB settles within +5% of SR2 | 65% |

**Moved onto the productization base (2026-11-22).** Twin PB's pre-registration (calls confirmed) on SR2-NP or SR2-NP-T, with SR2-NP's own untrained costs and a timing proxy, is [productization/Stage3c_Twin_PB_Precision_Bottleneck.md](../productization/Stage3c_Twin_PB_Precision_Bottleneck.md).

**What the notebooks need** (checked 2026-11-13 against HEAD `d968e75`; not yet written): a Cell 0 switch for each twin with a conditional tag component, so existing tags stay the same; the per-layer $a_k$ in `AnisotropicDepthConditionedGaussianVTheta`; the bottleneck in `B_proj`'s construction; and a parameter-count check.

---

<a name="shared-core"></a>
## 6. Can the wells share one precision core? (protocol `p-precision-cpu-result`)

**Measured 2026-11-14 on SR2's best checkpoint, without retraining (the author's question). No shared-core twin is filed; PA and PB stay proposed, unconfirmed and not run.**

**The question.** `B_proj` gives each of the eight wells its own rank-4 factor $B_k$ for every token. Could the wells share one core? Three versions were tried, from the most generous to one a model could compute: one shared $B$; a shared subspace read off the trained factors per token (an **oracle**: the model could not compute it more cheaply than the factors themselves); and a **fitted head** $\widehat{B_k}(x) = \mathrm{cap}(U(x) C_k(x))$, with a basis $U(x)$ of $s$ directions and a per-well mix $C_k(x)$, fitted to the trained force on 128 training windows with the rest of the model frozen.

| replacement | layer 0 only | layer 1 only | both layers |
| --- | ---: | ---: | ---: |
| every well takes the strongest well's B_k | +1,252% | +5.7% | +1,261% |
| one shared B (oracle) | +1,021% | +8.1% | +1,047% |
| each B_k in a shared 4-dimensional subspace (oracle) | +975% | +8.0% | +979% |
| each B_k in a shared 8-dimensional subspace (oracle) | +121% | −0.0% | +121% |

The fitted head (`debug/vtheta_shared_subspace_fit.py`; 16 evenly spaced validation windows, trained perplexity 58.28):

| head | weights (share of B_proj) | force error, layer 0 / layer 1 | layer 0 only | layer 1 only | both layers |
| --- | ---: | --- | ---: | ---: | ---: |
| s = 8 | 6,393,088 (27.1%) | 13.0% / 1.9% | +663% | +35% | +731% |
| s = 16 | 12,786,176 (54.2%) | 7.3% / 1.2% | +513% | +32% | +578% |

For comparison, plain SVD truncation of `B_proj` to rank 512 (7,274,496 weights, 30.8%) costs +41% with no fitting at all.

- **Layer 0 has no shared core.** Well 3 carries 81.5% of the force weight and the other seven 2.1–3.7% each, yet replacing their factors with the strong well's costs +1,252%. Their orientations change from token to token, and the wells do not agree on a few common directions. The direction count agrees: 8 shared directions cost +157%, 24 cost +5.0% and 28 cost +1.8%.
- **Layer 1's success is close to automatic.** Its two working wells (80.7% and 19.3% of the weight) span at most 8 directions, so a shared 8-dimensional subspace is exact by construction.
- **The head fails where the oracle works.** Doubling $s$ from 8 to 16 lowers the cost only from +731% to +578%, so the limit is predicting the basis, not its width. The frozen model is also sensitive to the force: a 1.9% force error costs +35% at layer 1.
- All numbers are without retraining and bound the cost from above. The fitted head is worse than a plain bottleneck with fewer weights, so a training run was not judged worth its GPU time.

**Decision (2026-11-14).** No shared-core twin. The compression candidates stay PA (+1.2% without retraining) and PB (+41% without retraining); PA1, PB1 and PB2 await the author's confirmation. Scripts are in [`notebooks/conservative_arch/scaleup/debug/`](../../notebooks/conservative_arch/scaleup/debug/), outputs in [`results/precision_ablation/`](../../notebooks/conservative_arch/scaleup/results/precision_ablation/). The head was fitted on an OpenWebText training file, never on the validation set.

---

<a name="open"></a>
## 7. What stands and what is open

- **Stands.** Shrinking the bank costs perplexity: +2.2% at 24 directions (K4-R6), +11.7% at 12 (K2-R6). Layer 0 fills whatever direction budget it is given; layer 1 runs on one well of rank 6. The smaller banks drain less in the decay than their gradient noise predicts (ratios 0.67 to 0.70). Layer 0's wells share no precision core.
- **Caveats.** One seed per arm. Every screen comparison with SR2 carries the pool caveat: the registered check scored "different pool", with evidence that the draws are the same.
- **Open.** K8-R6 is deferred. FLOW-R on K4-R6 is still to come, if the author wants it. The twins PA and PB are proposed, unconfirmed and not run. K4-R6's same-machine speed was queued.
