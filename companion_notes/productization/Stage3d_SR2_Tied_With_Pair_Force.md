# Stage 3d: SR2 With Tied Embeddings and the Pair Force (SR2-T)

**What this note is.** The setup and pre-registration of SR2-T: [SR2](Stage3_SR2_Without_Pair_Force.md), which keeps the routed pair force $V_\varphi$, trained from scratch with its input embedding and output head tied, as [SR2-NP-T](Stage3b_SR2_NP_Tied_Embeddings.md) was. It completes the 2×2 of pair force (on, off) by embeddings (untied, tied), and its first call is a trigger fixed in advance for whether the mean-field programme for the pair force should reopen. Written 2026-11-22.

**Status.** Trained 2026-11-22 on an A100 40 GB (16 × 2 microbatches, pair-force checkpoint on; §2): all 32,500 steps in 12.2 hours with evaluations, training peak 25.0 GB, leak monitor clean. **Settled 61.27, +5.1% against SR2 and 3.6% above the additive prediction. All four calls miss** (§8). The calls in §4 were confirmed by the author on 2026-11-22, before the first evaluation, and are binding. The diagnostics show a different regime from the other three corners. With tied embeddings and the pair force, SR2 trained into the operating point of F3.1, the model without the exact stiff-mode flow: soft stiff modes, a pair force that does a third of the step's work, and a heavy reliance on the velocity.

---

## Contents

1. [Why this arm](#1-why-this-arm)
2. [The configuration](#2-the-configuration)
3. [The build check](#3-the-build-check)
4. [Calls](#4-calls)
5. [What the result decides](#5-what-the-result-decides)
6. [Launch and readings](#6-launch-and-readings)
7. [Caveats](#7-caveats)
8. [Result](#8-result)

---

## 1. Why this arm

Three corners of the 2×2 are trained, all at the same budget on OpenWebText:

| | untied | tied |
| --- | ---: | ---: |
| pair force on | SR2, 58.32 | **SR2-T, this arm** |
| pair force off | SR2-NP, 58.51 | SR2-NP-T, 59.35 |

Without the pair force, tying costs 1.44%, all of it on the most frequent fifth of the vocabulary. The [embedding–head alignment test](Embedding_Head_Alignment_Test.md) found that for the most frequent types the shared matrix kept the input embedding's geometry, and the dynamics work harder: larger steps, twice the reliance on the velocity. The pair force and the velocity already stand in for each other in part ([Velocity_and_Pair_Force_as_Competing_Mechanisms.md](../depth_ladder/Velocity_and_Pair_Force_as_Competing_Mechanisms.md)), so the pair force may take part of the tie's extra load.

**The additive prediction.** If the two changes act independently, their costs add in log-perplexity:

$$\ln P_{\mathrm{SR2\text{-}T}} = \ln P_{\mathrm{SR2}} + \left( \ln P_{\mathrm{SR2\text{-}NP\text{-}T}} - \ln P_{\mathrm{SR2\text{-}NP}} \right),$$

so $P_{\mathrm{SR2\text{-}T}} = 58.32 \times 59.35 / 58.51 \approx 59.16$. The arm measures the interaction

$$I = \ln P_{\mathrm{SR2\text{-}T}} - \ln P_{\mathrm{SR2}} - \ln P_{\mathrm{SR2\text{-}NP\text{-}T}} + \ln P_{\mathrm{SR2\text{-}NP}},$$

zero under additivity and negative if the pair force absorbs part of what the tie takes away.

**Why it matters beyond the 2×2.** The mean-field programme for the pair force ([Mean_Field_PARFLM_Programme.md](../Mean_Field_PARFLM_Programme.md)) was moved off the productization path when SR2-NP showed that training without the pair force costs only 0.3%: there was little worth preserving cheaply. If the pair force is worth much more in a tied model, a cheap stand-in for it becomes the piece that would let the deployment model tie its embeddings, 25% fewer parameters at near-untied quality. A promising result would reopen the mean-field question with that concrete payoff; it would not answer it by itself (§5).

**Not a productization arm.** The deployment model has no pair force, and SR2-NP-T settled the untied head. The arm is a measurement of competing mechanisms whose first call doubles as a decision trigger.

## 2. The configuration

SR2's Cell 0, plus `TIE_EMBEDDINGS = True` and the pair-force checkpoint. Against the notebook's defaults, eight settings differ:

```python
REVERSE_CHANNEL     = False     # SR2
VPHI_GRAD_PATH      = 'live'    # SR2
XI_GRAD_PATH        = 'live'    # SR2
LOWRANK_DAMPED_FLOW = True      # SR2
VPHI_ANALYTIC_FORCE = True      # SR2
LAYER_CHECKPOINT    = False     # SR2 (speed switch; same arithmetic)
VPHI_FORCE_CHECKPOINT = True    # memory switch; same arithmetic (see below)
TIE_EMBEDDINGS      = True      # this arm; USE_OUTPUT_BIAS stays True
```

`PAIR_POTENTIAL` stays at its default (`'sparse_topk'`), the routed pair force of SR2.

**Why the pair-force checkpoint (2026-11-22, at launch).** Without it, Cell 5's memory probe on a 40 GB A100 found batch 16 at 33.6 GB plus a 0.6 GB optimizer reserve, over the notebook's 80% budget of 33.9 GB, and fell back to 8 sequences × 4 accumulations. SR2, SR2-NP, SR2-NP-T, K4-R6 and K2-R6 all ran at 16 × 2; at 8 × 4 the per-microbatch clip of the positional table acts four times per step instead of two, a second difference inside the 2×2. `VPHI_FORCE_CHECKPOINT = True` recomputes only the closed-form pair force in the backward, which frees most of the pair force's saved intermediates; the loss and every gradient are bit-identical on the CPU (`debug/verify_vphi_force_checkpoint.py`), it is not in the tag, and K2-L8-CODE used it in place of layer checkpointing. Cell 5 builds the model with it, so the probe sees the saving: with it on, batch 16 peaked at 21.7 GB (33.6 GB without), and the probe chose 16 × 2. The tag lines and the gradient clip of the shared matrix are handled as for SR2-NP-T (Stage3b, §2): Cell 0 appends `tied`, and Cell 6 sets `CLIP_THEN_SUM_GROUPS = {'P'}` when the embeddings are tied, so the shared matrix is clipped as SR2's separate head was.

## 3. The build check

2026-11-22, the notebook's own Cells 0 to 5 executed on the CPU (no checkpoint), with the folder tag's switches applied by `debug/copy_split.py`'s `tag_repl`:

| | tag | parameters | output head |
| --- | --- | ---: | --- |
| SR2 | `…_norc_vplive_xilive_sr2_L2probe_ob_untied_wsd_…` (its own folder's tag) | 76,572,110 | separate, plus bias |
| SR2-T | `…_norc_vplive_xilive_sr2_L2probe_ob_tied_wsd_…` | 57,273,422 | the input embedding, plus bias |

The difference is exactly the separate head, 19,298,688 parameters. In both, the tag Cell 0 builds equals the one `tag_repl` expects, and Cell 5 builds the pair force (`PAIR_POTENTIAL = 'sparse_topk'`).

## 4. Calls

**Proposed and confirmed by the author on 2026-11-22, at the probabilities below, before the run's first evaluation; binding.** Settled perplexity as for every OpenWebText arm (the mean of the last three evaluations once the trend has flattened). The tie's cost without the pair force is $\ln(59.35 / 58.51) = 0.0143$ nats.

| id | call | probability |
| --- | --- | ---: |
| ST1 | **the trigger:** SR2-T settles at or below 58.75, so the pair force recovers at least half of the tie's cost (its value in the tied model, against SR2-NP-T, at least 1.0%, three times its untied 0.33%) | 25% |
| ST2 | the interaction is negative beyond seed scatter: SR2-T settles at or below 58.98, 0.3% under the additive 59.16 | 40% |
| ST3 | the pair force takes part of the velocity's load: resetting the velocity (Cell 6b-7, gate 1) costs SR2-T less than SR2-NP-T's +5.80 in perplexity | 70% |
| ST4 | the recovery is where the tie's cost sits: on the most frequent fifth, SR2-T's paired loss difference from SR2 is at most +0.021 nats, half of SR2-NP-T's +0.043 against SR2-NP | 30% |

- **The threshold of ST1.** Half the tie's cost is $0.0072$ nats, and $58.32 \times e^{0.0072} \approx 58.74$; 58.75 rounds it.
- **ST1 implies ST2;** ST2 alone records a negative interaction too small to matter for the mean-field question.
- **ST3 and ST4** read the mechanism: whether the pair force relieves the velocity, and whether it helps the tokens that pay for the tie.

## 5. What the result decides

**ST1 hits: the mean-field question reopens, with two checks before any mean-field implementation.**

1. **The cheapest competitor first:** SR2-NP tied, with one learned $d \times d$ linear map applied to the final state before the tied read-out (147,456 parameters in place of the 19.3M head). It sits outside the dynamics and is the "shared linear map" the alignment test allowed for; untrained, such a map left most of each output row unexplained, but a trained one may do better. One ordinary GPU arm, with its own pre-registration. If it recovers the tie's cost, the motive for the mean field disappears again.
2. **Probe A on SR2-T's weights:** whether the pair force's contribution in the tied model is captured by its mean field (Mean_Field_PARFLM_Programme.md; calls A1 to A5 were confirmed for F3.1 and would need restating for this model). If the tied model's pair force is pair-specific routing, a mean field will not replace it.

The mean field is implemented only if the competitor falls short and Probe A says a mean field can stand in.

**ST1 misses, ST2 hits:** the interaction is recorded as a measured case of competing mechanisms (a candidate for the book's §28); no mean-field motive.

**Both miss:** the tie and the pair force act independently at this scale; recorded, and the 2×2 is closed.

## 6. Launch and readings

Open `colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb` from GitHub, set the eight values of §2 in Cell 0, and run Cell 0 → 1 → 1b → 1c → 1d → 2 → 3 → 4 → 5 → 5b → 6. Check that Cell 0 prints the SR2-T tag of §3 (with `_vplive_` and `_ob_tied_`) and the line `[V_phi] VPHI_FORCE_CHECKPOINT=True`, Cell 5 reports 57,273,422 parameters and `Auto batch: 16 x accum=2`, and Cell 6's header reads `[clip-then-sum] active for groups ['P'] (1 param(s) total)`.

**Expected time:** the pair force makes each step slower than SR2-NP's (0.68 s). K4-R6, the most recent run with the pair force, took 1.10 s per step on an A100 40 GB, so about 10 hours. If evaluation runs out of memory, `EVAL_LAYER_CHECKPOINT = True` (recompute during evaluation only; the training arithmetic and the tag are unchanged).

**After training, on the best checkpoint:**

- **On the GPU:** Cells 6b-7 (gate 1 for ST3), 6b-9 and 6b-12 (here they read the pair force's share of the step), 6b-13, 7 and 8.
- **On the CPU,** one job at a time:
  - the frequency quintiles of SR2 and SR2-T on the same windows (`debug/frequency_quintile_loss.py`; SR2's per-token losses are not cached yet), for ST4;
  - the causality check (`debug/causality_check_checkpoint.py OUT FOLDER False live live none default sr2 tied`);
  - the embedding–head alignment test with SR2 and SR2-T (`debug/embedding_head_alignment.py`), to see whether the shared matrix takes the same role with the pair force present;
  - the layer-1 well screen.

## 7. Caveats

- **One seed per corner.** A difference near 0.3% is of the size seed scatter can produce; ST2's margin is that size, ST1's twice it.
- **SR2 read the earlier OpenWebText training file;** the other three corners read the file rebuilt on 2026-11-12. The registered pool check scored "different pool" by 0.0002 nats at one step, with evidence that the draws are the same (Stage3, caveats). The interaction mixes the two.
- **The additive prediction assumes costs add in log-perplexity,** the convention of the competing-mechanisms analysis; additivity in perplexity itself would give 59.16 as well to two decimals.
- **The trigger is a reason to look, not a result about the mean field.** ST1 says the pair force is worth more when tied; whether a mean field keeps that value is what Probe A and the competitor arm decide.

## 8. Result

Scored 2026-11-22 from the run's log and the diagnostic cells, read on the best checkpoint (step 31,000, an evaluation dip all four corners share). "Settled" is the mean of the last three evaluations: 61.38, 61.31, 61.13.

**The 2 × 2, settled perplexity:**

| | untied | tied | cost of the tie |
| --- | ---: | ---: | ---: |
| with the pair force | SR2 58.32 | **SR2-T 61.27** | +4.9% (0.049 nats) |
| without the pair force | SR2-NP 58.51 | SR2-NP-T 59.35 | +1.4% (0.014 nats) |
| value of the pair force | +0.3% | **−3.2%** | |

The interaction is 0.035 nats, ten times the pair force's untied value and of the opposite sign from the one ST1 and ST2 asked about. With tied embeddings, the pair force makes the model worse: SR2-T settles 3.2% above SR2-NP-T.

**The calls:**
- **ST1 (settles at 58.75 or below; the mean-field trigger), 25%: MISS.** 61.27.
- **ST2 (at or below 58.98, a negative interaction), 40%: MISS.** The interaction is positive and large: 3.6% above the additive 59.16.
- **ST3 (resetting the velocity costs SR2-T less than SR2-NP-T's +5.80), 70%: MISS.** +24.17 (+40.9%), four times SR2-NP-T's.
- **ST4 (most frequent fifth, SR2-T's paired loss difference from SR2 at most +0.021 nats), 30%: MISS.** +0.061 (0.004) nats, larger than SR2-NP-T's +0.043 against SR2-NP. The pair force does not recover the tie's cost on the frequent tokens; it adds to it.

**Where the cost sits** (`debug/frequency_quintile_loss.py`, 128 validation windows of 512 tokens, the same 65,536 targets for all four corners; `results/frequency_quintiles/frequency_quintiles_sr2_tie_2x2.txt`). Mean loss per frequency fifth, in nats; Q4, the most frequent fifth of the token types, carries 89.2% of the targets:

| quintile | SR2 | SR2-NP | SR2-NP-T | SR2-T | tie, without the pair force | tie, with the pair force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q0 (rarest) | 11.397 | 11.473 | 11.115 | 11.302 | −0.358 | −0.096 (0.162) |
| Q1 | 10.015 | 10.056 | 9.669 | 9.890 | −0.387 | −0.125 (0.083) |
| Q2 | 8.347 | 8.375 | 8.208 | 8.323 | −0.167 | −0.024 (0.052) |
| Q3 | 7.127 | 7.137 | 7.042 | 7.095 | −0.095 | −0.032 (0.032) |
| Q4 (most frequent) | 3.614 | 3.611 | 3.654 | 3.676 | +0.043 | **+0.061 (0.004)** |

The last two columns are the tie's paired effect: SR2-NP-T minus SR2-NP, and SR2-T minus SR2 (standard error clustered by window). Without the pair force, the tie helps every rarer fifth and costs the frequent one. With the pair force, it costs the frequent fifth more and keeps only about a third of the gain on the rarer ones. The pair force spoils the tie at both ends.

**The diagnostics: SR2-T sits in F3.1's regime.** Cell outputs on each run's best checkpoint; the gates of Cell 6b-7 as a share of the carried perplexity:

| reading | SR2 | SR2-NP | SR2-NP-T | **SR2-T** | F3.1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| velocity reset (gate 1) | +3.9% | +5.6% | +10.1% | **+40.9%** | +34% |
| one hop more, same step (gate 2, N = 3) | +35% | +29% | +23% | **+134%** | +92% |
| refinement, same span (gate 3, N = 3) | +50% | +50% | +219% | **+123%** | +143% |
| median $\omega \Delta t$ (Cell 6b-13) | 7.55 | 6.18 | 6.25 | **4.21** | 4.32 |
| step's distance from the $V_\theta$ geodesic, R(geo) (Cell 6b-9) | 0.62 | 0.50 | 0.42 | **0.90** | |
| the pair force's share of it (R moved, geo → cons) | −0.002 | 0 | 0 | **−0.326** | |

- **The stiff modes are soft.** SR2's median $\omega \Delta t$ is 7.55, and 6.2 in both models without the pair force. SR2-T's is 4.21, F3.1's 4.32. The exact stiff-mode flow is still there, but the trained landscape no longer puts much in the stiff directions.
- **The pair force does a third of the step's work.** In SR2 the pair force moves the step 0.002 away from the $V_\theta$ geodesic; in SR2-T, 0.326, the same order as SR3-crit's 0.30, the other model where the pair force took over part of the step. Cell 6b-12, which in these conservative-only models reads the pair force's per-token contribution, finds 92.8% of tokens strongly forced at layer 1 (deflection past 0.75); in SR2 it finds none (all tokens under 0.25).
- **The velocity is carrying.** Resetting it costs +41%, more than in F3.1 (+34%) and ten times SR2's.
- **No flow.** Gates 2 and 3 grow away from the trained step, as in F3.1. SR2-NP-T is also poor under refinement (+219%), so the tie alone already costs SR2 its flow under refinement; the pair force adds the loss under more hops.

**Reading.** SR2's defining property in the earlier arms was that the exact stiff-mode flow idles the pair force (reliance 1.8%, value 0.3%). That property does not survive tying the embeddings. Tied, the same configuration trains into the regime of the split-step model F3.1, where the pair force and the velocity do much of the work, and that regime is 3.2% worse here than the same tied model without the pair force. The tie and the pair force compete: each is affordable alone, and together they push training to a worse operating point. That is a measured case of competing mechanisms for the book's §28, and a stronger one than ST2 contemplated. Why the tie has this effect is not known. The embedding–head alignment test on SR2-T (§6) is the first CPU reading of it: whether, with the pair force present, the shared matrix gave up the input geometry it kept in SR2-NP-T.

**What it decides.**
- **No mean-field motive** (ST1 missed); the 2 × 2 is closed.
- **For deployment:** a tied model must drop the pair force. That is the premise of the tied arms in the plan's grid (VF0-T, PMX-K16-NP-T, FO-b-PMX-K16-NP-T; [Conservative_Core_Productization_Plan.md](Conservative_Core_Productization_Plan.md), §6).
- **§5's decision table did not foresee this outcome.** Its "both miss" row reads "the tie and the pair force act independently"; they do not, in the opposite direction.

**The CPU readings (2026-11-22).**

- **Which role the shared matrix took** (`debug/embedding_head_alignment.py`, SR2 against SR2-T; `results/frequency_quintiles/embedding_head_alignment_sr2_vs_tied.txt`; method in [Embedding_Head_Alignment_Test.md](Embedding_Head_Alignment_Test.md)). The same role as in SR2-NP-T. For the 100 most frequent types (45% of the training targets), a linear map from the untied model's input embedding predicts SR2-T's shared rows with $R^2 = 0.395$, and from its output head with 0.219. SR2-NP-T read 0.419 and 0.222. At every frequency band the two tied models agree to within 0.03, and their shared rows have the same norms to within 2%. With or without the pair force, the shared matrix keeps the input geometry for the frequent tokens. So the tie's extra cost with the pair force is not in how the shared matrix is used. It is in the dynamics, the regime change above.
- **Causality** (`debug/causality_check_checkpoint.py … sr2 tied`): changing future tokens moves nothing (exactly 0) and the batches are independent. Prefix-only scoring agrees to within 0.037 in the logits, with a leak tax of −3.3e-5 nats; the script rates it clean. The logit difference is eight times SR2's (4.6e-3, leak tax −4.6e-6) and well under PM1's 9.8e-2. A model whose pair force does more work amplifies the float rounding of a shorter prefix more; the negative leak tax says no future information gets in.
- **Well screen** (`debug/vtheta_well_probe.py`, `debug/keep_k_curves.py`, 16 windows of OpenWebText; `results/well_screen/*sr2t*`). Layer 1 runs on a single well (well 0) with 100% of the force weight, and keeping only it changes nothing (+0.0%). The benchmark's layer-1 pruning for this checkpoint is `prune=1:0`. At layer 0 the strongest well carries 49.0% of the force weight, against 86.3% in SR2-NP-T and 77.4% in SR2-NP: tied and with the pair force, layer 0 spreads its work over its wells. A first run defaulted to the code validation set and is discarded.
- **How much SR2-T leans on its pair force** (the trained per-layer scale set to zero; 16 windows of OpenWebText). Loss 4.050 to 4.840: **+0.790 nats, +120% in perplexity.** Removing the pair force costs SR2 0.0135 nats (+1.4%), F3.1 0.282 and PM1 0.324 (the CD measurements). SR2-T relies on its pair force about 60 times as much as SR2, the most of any OpenWebText model measured. The trained scales are 0.063 at layer 0 and 1.694 at layer 1, against SR2's 0.020 and 0.006.

**A caveat specific to this result.** A switch of operating point from one seed is the kind of result a different seed could undo. The size of the change (four readings moving by factors of two to ten) argues against noise, but it is one run.

