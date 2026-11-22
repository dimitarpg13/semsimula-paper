# Stage 3b: SR2-NP With Tied Embeddings (SR2-NP-T)

**What this note is.** The setup and pre-registration of the second GPU arm of stage 3 of [Conservative_Core_Productization_Plan.md](Conservative_Core_Productization_Plan.md): [SR2-NP](Stage3_SR2_Without_Pair_Force.md) trained from scratch with its input embedding and output head tied, and the output bias kept. Its question: is the separate output head, 19.3M of SR2-NP's 76.4M parameters, needed? Written 2026-11-21.

**Status.** Set up 2026-11-21. A first launch on 2026-11-22 was stopped at step 50 because of the gradient clip of the shared matrix (§2); relaunched with the fix. The calls in §3 were **confirmed by the author on 2026-11-21**, with the 1% tolerance of TE1, and are binding.

---

## Contents

1. [Why this arm](#1-why-this-arm)
2. [The configuration](#2-the-configuration)
3. [Calls](#3-calls)
4. [Launch](#4-launch)
5. [After the run](#5-after-the-run)
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

| id | call | probability |
| --- | --- | ---: |
| TE1 | SR2-NP-T settles within 1% of SR2-NP's 58.51 (at or below 59.10) | 50% |
| TE2 | the rare-token collapse does not return: in SR2-NP-T every quintile's mean loss is below the uniform guess of 10.83 nats | 80% |
| TE3 | the rarest quintile's mean loss in SR2-NP-T is within 0.5 nats of SR2-NP's | 55% |

**Decision rule** (the 1% tolerance confirmed by the author). TE1 hits: later productization arms run tied, and the cards and the book quote 57.1M parameters. TE1 misses: the untied head stays, now as a measured requirement on the Gen 3 core, and TE2 and TE3 say whether the cost sits in the rare tokens, as D0.4 found on Gen 1, or elsewhere.

## 4. Launch

Open `colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb` from GitHub after the tag change of §2 is pushed. In Cell 0, set the six values of §2. Run Cell 0 → 1 → 1b → 1c → 1d → 2 → 3 → 4 → 5 → 5b → 6. Check that Cell 0 prints the SR2-NP-T tag of §2 (with `_ob_tied_`), Cell 5 reports 57,135,619 parameters, and Cell 6's header reads `[clip-then-sum] active for groups ['P'] (1 param(s) total)`.

After training, on the best checkpoint, the same diagnostic cells as SR2-NP: 6b-7 (flow or maps; its Gate 1 reads the velocity reset), 6b-9 (the damped $V_\theta$ geodesic step), 6b-12 (is the forcing sparse), 6b-13 ($\omega \Delta t$ at the endpoint), 7 (component diagnostics) and 8 (the training curve). Cells 6b-8, 6b-10 and 6b-11 concern the reverse channel or the register path and do not apply. Expected time: SR2-NP's, about 6 hours on an A100, perhaps slightly less (a smaller optimizer state); if evaluation runs out of memory, `EVAL_LAYER_CHECKPOINT = True` as before.

## 5. After the run

On the CPU, one job at a time:

- **The quintile breakdown** for SR2-NP and SR2-NP-T on the same validation windows (TE2, TE3). The script does not exist yet; it is a few dozen lines on top of the per-token loss the recall pathway probe already computes.
- **The causality check** on the final weights, as for SR2-NP: the shared matrix touches no force term, so it should read clean.
- **The well screen** of layer 1 (one well in SR2-NP), to see whether the read-out change moves the bank.

No CPU speed benchmark is needed: the forward pass does the same arithmetic.

## 6. Caveats

- **One seed per arm.** A difference near 0.3% is of the size the spread between training seeds can produce; TE1's 1% is well above that.
- **The tie couples two roles in this model.** The input embedding is also each token's starting position in the potential landscape, so tying binds the landscape's geometry to the read-out's. The theory note in the CfC+BAOAB deep dive argues this costs a conservative model more than a transformer; GPT-2's record points the other way. The arm measures it.
- **D0.4 was read on a different model** (Gen 1, explicit integrator, a run later found to carry the causal leak). Its finding is the motivation, not the prediction.
