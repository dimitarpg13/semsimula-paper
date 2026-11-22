# Stage 3b: SR2-NP With Tied Embeddings (SR2-NP-T)

**What this note is.** The setup and pre-registration of the second GPU arm of stage 3 of [Conservative_Core_Productization_Plan.md](Conservative_Core_Productization_Plan.md): [SR2-NP](Stage3_SR2_Without_Pair_Force.md) trained from scratch with its input embedding and output head tied, and the output bias kept. Its question: is the separate output head, 19.3M of SR2-NP's 76.4M parameters, needed? Written 2026-11-21.

**Status.** Trained 2026-11-22 (A100 40 GB, 0.72 s per step, 6.5 h, all 32,500 steps; leak monitor clean; clip groups as intended). Settled **59.35** against SR2-NP's 58.51 (+1.44%): **TE1 misses**, TE2 and TE3 hit (§3). By the decision rule the untied head stays. The cost is not in the rare tokens: tied, the four rarer frequency fifths improve and only the most frequent fifth gets worse (§5). The calls were confirmed by the author on 2026-11-21 (TE2's threshold amended on 2026-11-22, before any result) and were binding. A first launch was stopped at step 50 because of the gradient clip of the shared matrix (§2).

---

## Contents

1. [Why this arm](#1-why-this-arm)
2. [The configuration](#2-the-configuration)
3. [Calls](#3-calls)
4. [Launch](#4-launch)
5. [Results](#5-results)
6. [Caveats](#6-caveats)

---

## 1. Why this arm

Every Gen 3 model runs with two read-out fixes, adopted together and never separated ([training_instabilities/Scaling_Outlook_Tied_Embeddings_and_the_d1024_Cascade.md](../training_instabilities/Scaling_Outlook_Tied_Embeddings_and_the_d1024_Cascade.md), §22):

- **an output bias** initialised to the log-unigram frequency (`USE_OUTPUT_BIAS = True`);
- **an untied output head**, a separate 384 × 50,257 matrix (`TIE_EMBEDDINGS = False`).

They answered diagnostic D0.4 ([Xi_Bottleneck_Diagnosis_Phase5.md](../Xi_Bottleneck_Diagnosis_Phase5.md), §8.4): on the Gen 1 Fock-PARFLM v2.1, with a tied head and no output bias, the rarest fifth of the vocabulary received 13.16 nats per token, worse than the uniform guess of $\ln 50257 = 10.83$. The note recommended the bias alone first and the untied head only if the rare tokens did not recover. Both went in at once, and the bias alone was never tried. §22 itself records that the matched GPT-2, tied and without an output bias, never showed the pathology.

The untied head costs SR2-NP no speed (the output product has the same size either way) and almost no memory at inference (about 77 MB of weights in float32). It costs parameters: tied, SR2-NP has 57.1M against 76.4M, which changes every comparison with the matched GPT-2 (33.7M) and the model cards.

## 2. The configuration

SR2-NP's Cell 0, plus `TIE_EMBEDDINGS = True`. Against the notebook's defaults, six settings differ:

```python
REVERSE_CHANNEL     = False     # SR2
XI_GRAD_PATH        = 'live'    # SR2
LOWRANK_DAMPED_FLOW = True      # SR2
LAYER_CHECKPOINT    = False     # SR2 (speed switch; same arithmetic)
PAIR_POTENTIAL      = 'none'    # SR2-NP
TIE_EMBEDDINGS      = True      # this arm; USE_OUTPUT_BIAS stays True
```

**A change to the notebook was needed first.** Until 2026-11-21 Cell 0 appended `ob` and `untied` to the run tag whatever the two switches said, so this arm would have received SR2-NP's tag and run folder. The two tag lines now follow the switches (`ob` or `nob`, `tied` or `untied`); with the defaults the tag is unchanged. `debug/copy_split.py`'s `tag_repl` reads `tied` and `nob` back from a folder's tag, so the CPU tools rebuild the arm correctly.

**The build check** (2026-11-21, the notebook's own Cells 0 to 5, executed on the CPU against the patched notebook):

| | tag | parameters | output head |
| --- | --- | ---: | --- |
| SR2-NP | `…_nophi_xilive_sr2_L2probe_ob_untied_wsd_…` (unchanged) | 76,434,307 | separate, plus bias |
| SR2-NP-T | `…_nophi_xilive_sr2_L2probe_ob_tied_wsd_…` | 57,135,619 | the input embedding, plus bias |

The difference is exactly the separate head, 19,298,688 parameters. In both, the tag built by Cell 0 equals the one `tag_repl` expects.

**The gradient clip of the shared matrix** (found at the first launch, 2026-11-22). Clip groups are keyed by the first part of a parameter's name. In SR2-NP the separate head `lm_head` is its own group: summed over the two microbatches, then clipped once at 1.0, a threshold it never reached (the whole model's norm was 0.32 at step 50). The input embedding `E` is one of the two groups clipped *per microbatch* at 0.3 before summing (`CLIP_THEN_SUM_GROUPS = {'E', 'P'}`). Tied, `E` carries the head's gradient too, and the first launch read `top[E]=0.6` at step 50, exactly twice 0.3: the per-microbatch clip was binding on the head, so the arm would have tested tying and a tighter clip at once. That launch was stopped at step 50 and its Drive folder renamed with the suffix `_aborted_clip`. Cell 6 now sets `CLIP_THEN_SUM_GROUPS = {'P'} if TIE_EMBEDDINGS else {'E', 'P'}`: tied, the shared matrix is clipped as SR2-NP's head was, and untied runs are unchanged. The rule follows from `tied` in the tag.

## 3. Calls

Confirmed by the author on 2026-11-21, at the probabilities below; binding. Settled perplexity is defined as for every OpenWebText arm (the mean of the last three evaluations once the trend has flattened). The frequency quintiles follow D0.4: token types ranked by their frequency in the training split, in five groups of equal size; the reading is the mean loss on the validation targets of each group.

| id | call | probability | result |
| --- | --- | ---: | --- |
| TE1 | SR2-NP-T settles within 1% of SR2-NP's 58.51 (at or below 59.10) | 50% | **miss**: 59.35 (59.42, 59.44, 59.20), +1.44% |
| TE2 | the rare-token collapse does not return: in SR2-NP-T every quintile's mean loss is below the unigram prior's on the same targets (amended 2026-11-22; see below) | 80% | **hit**: every quintile beats it by 2.84 to 5.03 nats |
| TE3 | the rarest quintile's mean loss in SR2-NP-T is within 0.5 nats of SR2-NP's | 55% | **hit**: 11.115 against 11.473 nats (−0.36) |

**TE2 amended before any result (2026-11-22, confirmed by the author).** As first written, TE2 compared each quintile with the uniform guess $\ln V = 10.83$ nats. The D0.4 script (`debug/frequency_quintile_loss.py`) showed that this threshold says nothing about a collapse: a calibrated model must score above $\ln V$ on types rarer than $1/V$, and the matched GPT-2 itself reads 10.84 nats on the rarest quintile, SR2-NP 11.47. The reference that does mark a collapse is the unigram prior, the loss of a model that knows only the training frequencies (13.96 nats on the rarest quintile of the same validation targets; `results/frequency_quintiles/frequency_quintiles_sr2np_gpt2.txt`). The amendment was made while SR2-NP-T was still training and before any of its outputs were seen; the probability stays at 80%.

**Decision rule** (the 1% tolerance confirmed by the author). TE1 hits: later productization arms run tied, and the cards and the book quote 57.1M parameters. TE1 misses: the untied head stays, now as a measured requirement on the Gen 3 core, and TE2 and TE3 say whether the cost sits in the rare tokens, as D0.4 found on Gen 1, or elsewhere.

## 4. Launch

Open `colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb` from GitHub after the tag change of §2 is pushed. In Cell 0, set the six values of §2. Run Cell 0 → 1 → 1b → 1c → 1d → 2 → 3 → 4 → 5 → 5b → 6. Check that Cell 0 prints the SR2-NP-T tag of §2 (with `_ob_tied_`), Cell 5 reports 57,135,619 parameters, and Cell 6's header reads `[clip-then-sum] active for groups ['P'] (1 param(s) total)`.

After training, on the best checkpoint, the same diagnostic cells as SR2-NP: 6b-7 (flow or maps; its Gate 1 reads the velocity reset), 6b-9 (the damped $V_\theta$ geodesic step), 6b-12 (is the forcing sparse), 6b-13 ($\omega \Delta t$ at the endpoint), 7 (component diagnostics) and 8 (the training curve). Cells 6b-8, 6b-10 and 6b-11 concern the reverse channel or the register path and do not apply. Expected time: SR2-NP's, about 6 hours on an A100, perhaps slightly less (a smaller optimizer state); if evaluation runs out of memory, `EVAL_LAYER_CHECKPOINT = True` as before.

## 5. Results

**Perplexity.** Settled 59.35 against SR2-NP's 58.51, +1.44%; the best evaluation 57.22 against 56.71, both at step 31,000. The gap is steady through the decay phase: on each of the 14 evaluations the two runs share from step 26,000 on, the tied run is 0.9% to 2.1% worse, and the paired mean is +1.49% (standard error 0.09%). It is a real cost of the tie, not seed noise, and it is 19,298,688 parameters (25%) cheaper.

**Where the cost sits** (`debug/frequency_quintile_loss.py`, 128 validation windows of 512 tokens, the same targets for every model; `results/frequency_quintiles/frequency_quintiles_sr2np_tied_gpt2.txt`). Mean loss per frequency fifth, in nats, and the tied run's paired difference from SR2-NP with its standard error clustered by window:

| fifth (share of targets) | unigram prior | SR2-NP | SR2-NP-T | tied minus untied | matched GPT-2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q0 rarest (0.9%) | 13.958 | 11.473 | 11.115 | −0.358 (0.180) | 10.838 |
| Q1 (1.6%) | 13.316 | 10.056 | 9.669 | −0.387 (0.089) | 9.774 |
| Q2 (2.6%) | 12.825 | 8.375 | 8.208 | −0.168 (0.050) | 8.171 |
| Q3 (5.7%) | 12.073 | 7.137 | 7.042 | −0.095 (0.037) | 6.825 |
| Q4 most frequent (89.2%) | 7.017 | 3.611 | 3.654 | +0.043 (0.004) | 3.481 |
| all | 7.620 | 4.109 | 4.129 | +0.020 | 3.960 |

- **The tie helps the rare tokens and costs the frequent ones.** Weighted by their shares, the four rarer fifths gain 0.019 nats and the most frequent fifth loses 0.038; the net is the +0.020 nats (+2.0% in perplexity on these windows) of the settled gap.
- **No sign of the Gen 1 collapse.** The tied run is closer to the matched GPT-2 on the rarest fifth than SR2-NP is (11.12 against 10.84 and 11.47), and better than GPT-2 on Q1.
- **A reading, not a measurement:** tied, a rare token's row is trained by the output softmax at every step, not only when the token appears in the input, which is the usual reason tying helps rare tokens. The cost on the frequent tokens is what the caveat of §6 anticipated: one matrix now serves as the tokens' starting positions in the potential landscape and as the read-out, and the frequent tokens, which dominate both roles, pay for the compromise.

**Which role the shared matrix took** (`debug/embedding_head_alignment.py`, 2026-11-22; `results/frequency_quintiles/embedding_head_alignment_sr2np_vs_tied.txt`; problem statement, mathematics and figures in [Embedding_Head_Alignment_Test.md](Embedding_Head_Alignment_Test.md)). Tied, one matrix is both the input embedding $E$ (each token's starting position) and the head $W$. Rows are compared after removing their mean, since a vector common to every head row shifts every logit equally:

- **In SR2-NP the two roles want different rows.** The raw cosine between a token's input and output rows is about 0.02 to 0.05. Even after the best shared linear map (which a network could absorb, since $W = EA$ gives $h W^\top = (h A^\top) E^\top$), the median cosine is only 0.28 for the rarest fifth and 0.56 for the most frequent; shuffled rows give 0.00. A map fitted to favour the frequent tokens reconciles them (frequency-weighted cosine 0.82 on the most frequent fifth) only by fitting the rarer fifths worse than no map at all.
- **For the most frequent tokens, the shared matrix kept the input role.** A linear map from SR2-NP's rows predicts the tied model's rows for the 100 most frequent types (45% of the training tokens) with $R^2$ 0.42 from the input embedding against 0.22 from the head. From rank 100 on the two are level (0.41 against 0.39, then 0.33 against 0.34, then 0.21 against 0.21).
- **So the read-out yielded where the cost sits,** and the dynamics must turn an input-like starting position into the output direction. That fits the larger steps and the heavier use of the velocity below. All $R^2$ are modest: the tied matrix is largely its own solution, resembling neither role closely.

**The dynamics** (Colab cells on the best checkpoint, against SR2-NP's):

- **The velocity carries more.** Gate 1 of Cell 6b-7 (velocity reset between the layers): +5.80 in perplexity (57.20 to 63.00, +10.1%) against +3.21 (+5.6%) in SR2-NP.
- **Less tolerant of a changed step.** Under the standard refinement (Cell 6b-7, gate 3: three steps of 8/3 in place of two of 4) the loss rises by 1.16 nats, against 0.40 in SR2-NP; one step of 8 costs 3.80 nats against 1.94. One extra step at the trained size (gate 2) costs about the same (0.21 against 0.26 nats). Both models are maps, not flows; the tied one is fitted more tightly to its trained step.
- **Larger steps.** Median step size relative to the state, Cell 6b-12: 6.56 and 1.08 at layers 0 and 1, against 6.06 and 0.76.
- **Layer 1 closer to the bank's geodesic.** The bare $V_\theta$ geodesic deviates from the step by 0.17 at layer 1 (Cell 6b-9), against 0.31; layer 0 is unchanged (0.66 against 0.69).
- **Stiffer at layer 0.** Median $\omega \Delta t$ 8.61 at layer 0 and 3.99 at layer 1 (Cell 6b-13), against 7.29 and 4.31; both outside the pre-registered band of the L = 2 ladder, as SR2-NP was.

**Checks.**

- **Causality** (`debug/causality_check_checkpoint.py` with the new option `tied`): future perturbation and batch independence exact; prefix-only scoring within 4.0e-5 in the logits (leak tax +1.3e-7 nats). Clean.
- **Well screen** (`debug/vtheta_well_probe.py`, `debug/keep_k_curves.py`, 16 windows; `results/well_screen/*sr2npt*`): layer 1 again runs on one well, now well 1, with 100% of the force weight, and keeping only it changes nothing; at layer 0 the strongest well carries 86.3% of the force weight, against 77.4% in SR2-NP. The benchmark's layer-1 pruning for this checkpoint is `prune=1:1`.

**What this decides.** By the confirmed rule, the untied head stays as a measured requirement of the Gen 3 core: 19.3M parameters for 1.4% in perplexity. SR2-NP-T is exactly the configuration D0.4 recommended trying first (the output bias with a tied head). On the Gen 3 core it leaves the rare tokens better off, as D0.4 hoped, but it costs the frequent tokens, which D0.4 did not measure; TE2 and TE3 place the cost there. The next productization arm, twin PB ([Stage3c_Twin_PB_Precision_Bottleneck.md](Stage3c_Twin_PB_Precision_Bottleneck.md)), runs on SR2-NP, untied. The tie with the pair force present, the fourth corner of the 2×2, is set up as SR2-T ([Stage3d_SR2_Tied_With_Pair_Force.md](Stage3d_SR2_Tied_With_Pair_Force.md)).

**Model card.** [semsimula-ladder-live-owt-d384-l2-none-norc-nophi-sr2-tied](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-none-norc-nophi-sr2-tied), published 2026-11-22 in the Gen 3 collection (position 9 of 14, by settled perplexity).

**Outputs.** The run's logs and cell outputs: [`results/cfc_baoab_…_ob_tied_…/`](../../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_nophi_xilive_sr2_L2probe_ob_tied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/). The quintiles and the well screen as named above.

## 6. Caveats

- **One seed per arm.** A difference near 0.3% is of the size the spread between training seeds can produce; TE1's 1% is well above that.
- **The tie couples two roles in this model.** The input embedding is also each token's starting position in the potential landscape, so tying binds the landscape's geometry to the read-out's. The theory note in the CfC+BAOAB deep dive argues this costs a conservative model more than a transformer; GPT-2's record points the other way. The arm measures it.
- **D0.4 was read on a different model** (Gen 1, explicit integrator, a run later found to carry the causal leak). Its finding is the motivation, not the prediction.
