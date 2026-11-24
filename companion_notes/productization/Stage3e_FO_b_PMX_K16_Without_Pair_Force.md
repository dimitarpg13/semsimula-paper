# Stage 3e: FO-b-PMX-K16 Trained Without the Pair Force (FO-b-PMX-K16-NP)

**What this note is.** The pre-registration of the second arm of the PMX line in the productization plan ([Conservative_Core_Productization_Plan.md](Conservative_Core_Productization_Plan.md), §6). The arm is the first-order model with sixteen Poisson-mode wells, trained without the pair force $V_\varphi$: the analogue, on the PMX line, of SR2-NP ([Stage3_SR2_Without_Pair_Force.md](Stage3_SR2_Without_Pair_Force.md)). Written 2026-11-22.

**Status.** Set up; not launched. The calls in §4 were confirmed by the author on 2026-11-22 and are binding. **The tolerance is +2%,** set by the author on 2026-11-24 before launch, as for twin PB and the earlier first- against second-order comparisons (§5).

---

## Contents

1. [Why this arm](#1-why-this-arm)
2. [The configuration](#2-the-configuration)
3. [The build check](#3-the-build-check)
4. [Calls](#4-calls)
5. [What the result decides](#5-what-the-result-decides)
6. [Launch](#6-launch)
7. [Caveats](#7-caveats)

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

## 7. Caveats

- **One seed.** The pairing removes the evaluation noise, not the training-seed spread, which is unmeasured.
- **The hyperparameters were tuned for the second-order models** and are inherited unchanged.
- **L = 2 and 532M tokens.** The pair force's value could grow with depth or training length; this arm measures it at the ladder's scale only.
