# Stage 3e: FO-b-PMX-K16 Trained Without the Pair Force (FO-b-PMX-K16-NP)

**What this note is.** The pre-registration of the second arm of the PMX line in the productization plan ([Conservative_Core_Productization_Plan.md](Conservative_Core_Productization_Plan.md), §6). The arm is the first-order model with sixteen Poisson-mode wells, trained without the pair force $V_\varphi$: the analogue, on the PMX line, of SR2-NP ([Stage3_SR2_Without_Pair_Force.md](Stage3_SR2_Without_Pair_Force.md)). Written 2026-11-22.

**Status.** Trained and scored (results downloaded 2026-11-24; §7). The calls in §4 were confirmed by the author on 2026-11-22 and are binding; **the tolerance is +2%,** set by the author on 2026-11-24 before launch. **FN1, FN2, FN3 and FN4 all hit.** The model settled at 53.39, 0.63% *below* FO-b-PMX-K16: training without the pair force made the first-order model better. Why is the subject of a separate probe, [Pair_Force_in_the_First_Order_Step.md](../depth_ladder/Pair_Force_in_the_First_Order_Step.md).

---

## Contents

1. [Why this arm](#1-why-this-arm)
2. [The configuration](#2-the-configuration)
3. [The build check](#3-the-build-check)
4. [Calls](#4-calls)
5. [What the result decides](#5-what-the-result-decides)
6. [Launch](#6-launch)
7. [Results](#7-results)
8. [Caveats](#8-caveats)

---

## 1. Why this arm

**Reliance is not value.** Removing the pair force from the trained FO-b-PMX-K16 costs +3.60% on the `mechanism_ablations` windows and 0.026 nats per token on the recall-probe windows. That is reliance: what the trained model routes through the term. Its value is what a model trained without the term loses. On SR2 the two differ by a factor of about six: removing the pair force costs 1.8%, training without it 0.3%, because the velocity took over part of its work.

**Why the first-order model may differ.** FO-b-PMX-K16 has no velocity to take the work over. When the velocity was removed, the first-order twins leaned harder on the pair force (1.2 to 2.1 times their second-order twins on OpenWebText, 3 times on code) and on a stiffer second layer ([Velocity_and_Pair_Force_as_Competing_Mechanisms.md](../depth_ladder/Velocity_and_Pair_Force_as_Competing_Mechanisms.md)). Without the pair force as well, what is left to substitute is the wells and the landscape. The arm measures how much of the reliance they can absorb.

**What it is for.** The CPU test ([Pair_Force_Recall_Split_FO_b_PMX_K16.md](Pair_Force_Recall_Split_FO_b_PMX_K16.md)) found that the pair force does no addressing in this model, so a mean field could replace it. This arm asks whether anything needs replacing.

## 2. The configuration

FO-b-PMX-K16's Cell 0 with the pair force switched off, following the SR2-NP rule. `VPHI_GRAD_PATH` returns to its default, since Cell 5b refuses a live pair-force path when there is no pair force, and `VPHI_ANALYTIC_FORCE` is inert because both of its paths return a zero force.

| setting | default | FO-b-PMX-K16 | this arm |
| --- | --- | --- | --- |
| `REVERSE_CHANNEL` | True | False | False |
| `XI_GRAD_PATH` | 'default' | 'live' | 'live' |
| `POISSON_MODES` | 0 | 16 | 16 |
| `LOWRANK_DAMPED_FLOW` | False | True | True |
| `POISSON_WELLS_EXACT` | False | True | True |
| `FIRST_ORDER_OVERDAMPED` | False | True | True |
| `LAYER_CHECKPOINT` | True | False | False (speed switch, same arithmetic) |
| `VPHI_GRAD_PATH` | 'default' | 'live' | 'default' |
| `VPHI_ANALYTIC_FORCE` | False | True | default (inert) |
| `PAIR_POTENTIAL` | 'sparse_topk' | 'sparse_topk' | **'none'** |

Same corpus, budget and schedule as FO-b-PMX-K16: 32,500 steps × 32 × 512 = 532,480,000 OpenWebText tokens, peak learning rate 1.2e-3, seed 0. The two runs see the same training and evaluation batches, so their difference is measured paired.

**Tag:**

```text
xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_nophi_xilive_pm16_pmx16_sr2_fob_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn
```

FO-b-PMX-K16's tag with `vplive` replaced by `nophi`.

## 3. The build check

Run 2026-11-22 on the CPU. Cells 0 to 5b of the notebook were executed through `debug/verify_pm_switch.py`'s `build`, with the Cell 0 replacements derived from each tag by `debug/copy_split.py`'s `tag_repl`, at seed 0:

| | FO-b-PMX-K16 (rebuilt) | this arm |
| --- | ---: | ---: |
| tag printed by Cell 0 equals the folder tag | yes | yes |
| trainable parameters | 76,578,317 | **76,440,514** |
| pair force present | yes | no |
| first-order flow | yes | yes |
| loss at initialisation on a random batch of 2 × 64 | 12.7244 | 12.7244 |

The difference, 137,803 parameters, is the pair force, its score head and its per-layer scale, exactly the difference between SR2 and SR2-NP. FO-b-PMX-K16's card lists 76,578,318 parameters; the build counts trainable parameters only.

## 4. Calls

Proposed 2026-11-22, before the arm runs; **confirmed by the author on 2026-11-22, as written and at the probabilities below; binding.** "Settled" is the mean of the last three evaluations, as for every ladder model. FO-b-PMX-K16 settled at 53.73 and PMX-K16, its second-order twin with the pair force, at 54.24.

| id | prediction | called |
| --- | --- | --- |
| FN1 | settles within +2% of FO-b-PMX-K16 (54.80 or less) | 70% |
| FN2 | settles within +1% of FO-b-PMX-K16 (54.27 or less): training without the pair force costs less than a third of removing it | 45% |
| FN3 | settles at or below PMX-K16 (54.24): the first-order model without the pair force stays level with or ahead of the second-order model with it | 40% |
| FN4 | the wells take up the work: removing the wells costs more than FO-b-PMX-K16's +343% (`mechanism_ablations.py`, same windows) | 60% |

**Reasoning.** If the trained-out cost kept SR2's ratio to reliance (about 1 to 6), it would be about +0.6%, inside FN2. Two things argue for a larger cost. There is no velocity to stand in for the pair force, and FO-b-PMX-K16 already relies on it twice as much as its twin. On the other hand, the first-order models absorbed the velocity's work into the wells (8 times the reliance at two wells), so the wells may absorb this too, which FN4 tests.

## 5. What the result decides

- **At or below +1% (FN2 passes).** The pair force is not worth its cost on the PMX line in first order. The PMX line's production candidate drops it, as SR2-NP did, and the mean field is not needed there.
- **Above +2% (FN1 fails).** The pair force carries value that nothing else in the model can absorb. The mean field returns to the productization path, with Probe A and the field arm run on FO-b-PMX-K16 beside F3.1.
- **Between +1% and +2%.** Within the author's tolerance of +2% (set 2026-11-24, before launch): the PMX line's production candidate drops the pair force, as at or below +1%.
- **Since written (2026-11-22):** Probe A has run and the author ruled that no field arm runs ([Probe_A_Field_Pair_Force.md](../mean_field/Probe_A_Field_Pair_Force.md), §7). Above +2%, the pair force therefore stays in the PMX line's candidate, and any stand-in would need its own registration.

**A companion arm, not pre-registered here.** PMX-K16 trained without the pair force would complete a 2 × 2: {first order, second order} × {with, without the pair force}. With trained-out costs at both orders, it would answer directly whether the pair force is valuable only when the model has no velocity. Its interaction term is defined in [Velocity_and_Pair_Force_as_Competing_Mechanisms.md](../depth_ladder/Velocity_and_Pair_Force_as_Competing_Mechanisms.md).

## 6. Launch

In Cell 0 of `colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`, set:

```python
REVERSE_CHANNEL=False, XI_GRAD_PATH='live',
POISSON_MODES=16, LOWRANK_DAMPED_FLOW=True, POISSON_WELLS_EXACT=True,   # PMX with 16 wells
FIRST_ORDER_OVERDAMPED=True,              # tag 'fob'
LAYER_CHECKPOINT=False,                   # speed switch, same arithmetic
PAIR_POTENTIAL='none'                     # the one new change; VPHI_GRAD_PATH and VPHI_ANALYTIC_FORCE stay at their defaults
```

Then run Cell 0 → 1 → 1b → 1c → 1d → 2 → 3 → 4 → 5 → 5b → 6. Check that Cell 0 prints the tag of §2 and Cell 5 reports 76,440,514 parameters.

**Expected time.** FO-b-PMX-K16 took 1.33 s per step with the pair force, about 12 hours. Removing the pair force took SR2 from 1.28 to 0.68 s per step. The wells cost FO-b-PMX-K16 about 0.05 s per step over SR2, and they stay. So expect about 0.7 to 0.8 s per step: 6.5 to 7.5 hours of training on an A100, one session, plus about an hour for the diagnostic cells.

**After training,** on the best checkpoint:
- the diagnostic cells FO-b-PMX-K16 ran: 6b-7 (flow or maps), 6b-9 (the damped $V_\theta$ geodesic step), 6b-13 ($\omega \Delta t$ at the endpoint), 6b-15 (the wells), 7 and 8;
- on the CPU: `mechanism_ablations.py` (FN4, and the context-swap reading), `vtheta_omega_dt.py` (layer stiffness), `recall_pathway_probe.py` (recall through the context channels, against FO-b-PMX-K16), and the causality check.

## 7. Results

Results downloaded 2026-11-24; the run folder, the CPU readings and the copied run outputs are in `results/<tag>/` (the tag of §2 with the `cfc_baoab_owt_` prefix).

### 7.1 The training result

| model | settled | best (step 31,000) | last three evaluations |
| --- | ---: | ---: | --- |
| this arm | **53.39** | 51.53 | 53.37, 53.55, 53.26 |
| FO-b-PMX-K16 | 53.73 | 51.82 | 53.77, 53.81, 53.61 |
| PMX-K16 | 54.24 | 52.28 | |

Against FO-b-PMX-K16 the settled difference is −0.63%. Paired by step over the 23 evaluations of the decay, the mean gap is −0.0074 nats (standard deviation 0.0015), negative at all 23. In the stable phase (steps 5,500 to 21,000) it is −0.0051 nats, negative at 28 of 32. Between steps 2,000 and 5,500 the pair force helped, by up to 0.015 nats at step 4,000. Both models gain 0.171 nats over the decay.

### 7.2 The calls

| id | prediction | called | outcome |
| --- | --- | --- | --- |
| FN1 | settles within +2% of FO-b-PMX-K16 (54.80 or less) | 70% | **HIT** (53.39, −0.63%) |
| FN2 | settles within +1% of FO-b-PMX-K16 (54.27 or less) | 45% | **HIT** (53.39) |
| FN3 | settles at or below PMX-K16 (54.24) | 40% | **HIT** (53.39, −1.57%) |
| FN4 | removing the wells costs more than FO-b-PMX-K16's +343% | 60% | **HIT** (+382.5% on the same windows; per-sequence spread 29%) |

**What it decides (§5).** At or below +1%, the PMX line's production candidate drops the pair force, and the mean field is not needed there. The result goes further than the decision rule anticipated: the pair force has negative value in this model.

### 7.3 Run health and cost

- 0.974 s per training step (median), 8.8 hours for 32,500 steps, against 1.33 s for FO-b-PMX-K16: 27% faster, though slower than the 0.7 to 0.8 s expected in §6. The card was an A100 40 GB: Cell 7 reports 42.4 GB of device memory.
- Training peak memory 17.5 GB (24.6 GB with evaluation). No skipped steps; the gradient norm exceeded 1.0 at 15 of 650 logged steps, 11 of them in the warm-up (largest 1.50, step 1,000).
- SCAF leak monitor CLEAN at all seven audits (5k to 30k and 32.5k; future perturbation exactly 0; leak tax at most 1.5e-4 nats).

### 7.4 The CPU readings

On the best checkpoint, against FO-b-PMX-K16 on the same windows where both were read:

| reading | this arm | FO-b-PMX-K16 |
| --- | ---: | ---: |
| base perplexity, mechanism-ablation windows | 54.39 | 54.76 |
| removing the wells | +382.5% | +343.4% |
| swapping the context channels with another sequence's | +349.4% | +337.5% |
| swapping the wells' occupations with another sequence's | −0.00% | +0.01% |
| stiffness of the context potential, median angle per step of its stiffest mode, layer 0 / layer 1 | 2.35 / 6.43 | 2.13 / 5.31 |
| wells' force against the rest of the force, layer 0 / layer 1 (Cell 6b-15) | 0.535 / 3.81 | 0.559 / 3.24 |
| wells' depth positive, layer 0 / layer 1 (Cell 6b-15) | 50% / 88% | 31% / 75% |
| freezing the wells' clock, first 16 positions | +61.8% | +28.8% |
| layer 1: share of the step made by the explicit kick | 0.585 | 0.417 |
| per-layer flow, refinement penalty at 16 substeps (nats) | 1.430 | 1.414 |
| far recall: earlier occurrence removed from the layer-1 context channels (nats) | +0.456 (0.052) | +0.463 (0.055) |

- **The wells and the layer-1 landscape took the pair force's work.** Removing the wells costs more, their force share and depth at layer 1 are larger, the context potential is stiffer at layer 1, and the explicit kick makes more of layer 1's step. The context channels change little.
- **The wells are still positional:** swapping their occupations between sequences changes nothing, and position explains their force with R² 0.996 at layer 0 and 0.9999 at layer 1 (`pm_mode_usage.py`).
- **Recall runs through the same pathway,** the layer-1 context channels; the paired difference on far recall is −0.004 nats (standard error 0.002).
- **The per-layer flow still does not settle,** as in every first-order model; as trained, eight steps cost 2.21 nats against FO-b-PMX-K16's 2.18.
- **Causality, on the best weights:** future perturbation at five cut points exactly 0, batch independence exactly 0, prefix-only scoring leak tax +7.0e-7 nats. Clean.
- **Conservativity, on the trained weights:** every term conservative at both layers (Jacobian asymmetry at most 3.3e-7, closed-loop work at most 1.3e-7; rerun 2026-11-25). The first run read layer 0 with layer 1's depth code and gave $V_\theta$ a force of order $10^{-6}$ there; an earlier version of this note concluded that the force of $V_\theta$ is negligible at layer 0. That is withdrawn: with the right code its force is 0.10, two thirds of the total ([Depth_Code_Correction.md](../depth_ladder/Depth_Code_Correction.md)).

### 7.5 Why: the step probe

The probe and its reading are in [Pair_Force_in_the_First_Order_Step.md](../depth_ladder/Pair_Force_in_the_First_Order_Step.md). In short: in the trained FO-b-PMX-K16 the pair force at layer 0 has grown into a push along the landscape force (cosine 0.63, against 0.16 in PMX-K16; about 0.1 in both at step 500). The model leans on it, and the wells and the layer-1 landscape do less. Trained without it, they do that work themselves, and do it better. The explicit update of the pair force is not the main cause: the force is not rougher in first order, and re-evaluating it in sub-kicks costs a quarter to a third of its reliance.

### 7.6 Not run, and a correction

- Cell 8, the training curve, did run: its plot is `training_curve_aniso_gaussian_owt.png` in the run folder (corrected 2026-11-24; an earlier version of this note said it had not run).
- `recall_pathway_probe.py` has no pair-force arms here (the model has none); the context-channel cuts were run.

## 8. Caveats

- **One seed.** The pairing removes the evaluation noise, not the training-seed spread, which is unmeasured.
- **The hyperparameters were tuned for the second-order models** and are inherited unchanged.
- **L = 2 and 532M tokens.** The pair force's value could grow with depth or training length; this arm measures it at the ladder's scale only.
