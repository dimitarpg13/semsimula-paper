# Stage 3f: PMX-K16 Trained Without the Pair Force (PMX-K16-NP)

**What this note is.** The pre-registration of PMX-K16-NP: the second-order model with sixteen Poisson-mode wells, PMX-K16, trained without the pair force $V_\varphi$. It is the fourth corner of order × pair force on the line with sixteen wells, and the second-order companion of [FO-b-PMX-K16-NP](Stage3e_FO_b_PMX_K16_Without_Pair_Force.md). The author added it to the plan on 2026-11-22 ([Conservative_Core_Productization_Plan.md](Conservative_Core_Productization_Plan.md), §6). Written 2026-11-24.

**Status.** Set up; build check passed (§3); not launched. Third in the GPU queue, after VF0 and twin PB (author, 2026-11-24). **The calls in §4 were confirmed by the author on 2026-11-24 and are binding.** The tolerance is the author's to set.

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

Three corners of the grid are trained, each on the same 532,480,000 OpenWebText tokens with the same schedule, batches and seed:

| model | order | pair force | settled perplexity |
| --- | --- | --- | ---: |
| PMX-K16 | second | trained with it | 54.24 |
| FO-b-PMX-K16 | first | trained with it | 53.73 |
| FO-b-PMX-K16-NP | first | none | 53.39 |
| **PMX-K16-NP** (this arm) | second | none | |

In first order, training without the pair force made the model 0.63% better. The step probe reads this as the pair force crowding out the wells and the layer-1 landscape: at layer 0 the trained first-order pair force pushes along the rest of the force, and the model leans on it instead of strengthening its landscape ([Pair_Force_in_the_First_Order_Step.md](../depth_ladder/Pair_Force_in_the_First_Order_Step.md)). On SR2, which has no wells, training without the pair force cost the second-order model 0.33%.

Whether the pair force's value depends on the order on this line is open until this arm runs. With PMX-K16 as the base, write $V$ for the value of training a mechanism out, as in [the competing-mechanisms note](../depth_ladder/Velocity_and_Pair_Force_as_Competing_Mechanisms.md), §1. The two first-order corners give

$$V_v = \ln \frac{53.73}{54.24} = -0.0094, \qquad V_{v\varphi} = \ln \frac{53.39}{54.24} = -0.0158 .$$

This arm gives the pair force's value in second order,

$$V_\varphi = \ln \frac{\mathrm{PPL}(\text{PMX-K16-NP})}{54.24},$$

and with it the interaction

$$I_{v\varphi} = V_{v\varphi} - V_v - V_\varphi = -0.0063 - V_\varphi .$$

If the pair force's value were the same in both orders, $V_\varphi = -0.0063$ and the arm would settle at

$$54.24 e^{-0.0063} = 53.90 .$$

## 2. The configuration

PMX-K16's Cell 0 with the pair force switched off, following the rule of SR2-NP and FO-b-PMX-K16-NP. `VPHI_GRAD_PATH` returns to its default, since Cell 5b refuses a live pair-force path when there is no pair force; `VPHI_ANALYTIC_FORCE` returns to its default, since it is inert without a pair force.

| setting | default | PMX-K16 | this arm |
| --- | --- | --- | --- |
| `REVERSE_CHANNEL` | True | False | False |
| `XI_GRAD_PATH` | 'default' | 'live' | 'live' |
| `POISSON_MODES` | 0 | 16 | 16 |
| `LOWRANK_DAMPED_FLOW` | False | True | True |
| `POISSON_WELLS_EXACT` | False | True | True |
| `FIRST_ORDER_OVERDAMPED` | False | False | False |
| `LAYER_CHECKPOINT` | True | False | False (speed switch, same arithmetic) |
| `VPHI_GRAD_PATH` | 'default' | 'live' | 'default' |
| `VPHI_ANALYTIC_FORCE` | False | True | default (inert) |
| `PAIR_POTENTIAL` | 'sparse_topk' | 'sparse_topk' | **'none'** |
| `SEED` | 0 | 0 | 0 |

It differs from FO-b-PMX-K16-NP in `FIRST_ORDER_OVERDAMPED` alone.

**Tag:**

```text
xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_nophi_xilive_pm16_pmx16_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn
```

PMX-K16's tag with `vplive` replaced by `nophi`.

## 3. The build check

Run 2026-11-24 on the CPU. Cells 0 to 5b of the notebook were executed through `debug/verify_pm_switch.py`'s `build`, with the Cell 0 replacements derived from each tag by `debug/copy_split.py`'s `tag_repl`, from fresh weights at seed 0:

| | PMX-K16 (rebuilt) | this arm |
| --- | ---: | ---: |
| tag printed by Cell 0 equals the folder tag | yes | yes |
| trainable parameters | 76,578,317 | **76,440,514** |
| pair force present | yes | no |
| first-order flow | no | no |
| loss at initialisation on a random batch of 2 × 64 | 12.7244 | 12.7244 |

The 137,803 parameters removed are the pair force, its score head and its per-layer scale, as for SR2-NP and FO-b-PMX-K16-NP. The count equals FO-b-PMX-K16-NP's, since first order adds no parameters.

## 4. Calls

**Proposed 2026-11-24, before the arm runs; confirmed by the author the same day, as written and at the probabilities below; binding.** "Settled" is the mean of the last three evaluations, as for every ladder model.

| id | prediction | called |
| --- | --- | ---: |
| PN1 | settles within +2% of PMX-K16 (55.32 or less): the second-order model on this line can drop the pair force | 85% |
| PN2 | settles at or above 53.90: training the pair force out helps the second-order model less than it helped the first-order one (the interaction of §1 at or below zero) | 65% |
| PN3 | settles at or above PMX-K16 (54.24): in second order the pair force has zero or positive value, as on SR2 | 45% |
| PN4 | settles above FO-b-PMX-K16-NP (53.39): without the pair force, first order stays ahead, as it is with it (−0.9%) | 75% |
| PN5 | removing the wells costs this arm more than PMX-K16's +155.5% (`mechanism_ablations.py`, same windows): the wells take up the work in second order too | 55% |

**Reasoning.**
- **PN1.** In both precedents training the pair force out cost at most a third of a percent, and here the reliance is only +1.7%.
- **PN2 and PN3.** The step probe found the second-order pair force nearly orthogonal to the rest of the force at layer 0 (cosine 0.16, against 0.63 in first order), so the layer-0 crowding seen in first order should be weaker. But at layer 1 the second-order pair force does align with the rest of the force (0.39), so some crowding there is possible; hence PN3 at a coin toss.
- **PN4.** With wells, first order matched or beat second order at 2, 16 and 64 wells. If the pair force's value is higher in second order (PN2), the gap widens.
- **PN5.** In first order, removing the wells cost 39 points more without the pair force. The second-order model also has the velocity to absorb work, so the rise may be smaller.

## 5. What the result decides

- **PN1 hits.** The second-order PMX line also drops the pair force from its production candidate.
- **PN2 hits.** The pair force's value depends on the order on this line, and the crowding-out reading of the step probe gains its training-side support. The book can state the order dependence as measured.
- **PN2 misses.** Training the pair force out helps both orders about equally; the effect belongs to the wells, not to the order. The step probe's "why first order" reading is withdrawn.
- **PN3 hits.** As on SR2: in second order the pair force is worth a little, in first order it costs.

## 6. Launch

In Cell 0 of `colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`, set:

```python
REVERSE_CHANNEL=False, XI_GRAD_PATH='live',
POISSON_MODES=16, LOWRANK_DAMPED_FLOW=True, POISSON_WELLS_EXACT=True,   # PMX with 16 wells
LAYER_CHECKPOINT=False,                   # speed switch, same arithmetic
PAIR_POTENTIAL='none'                     # the one change; VPHI_GRAD_PATH and VPHI_ANALYTIC_FORCE stay at their defaults
```

Leave `FIRST_ORDER_OVERDAMPED` off and `SEED` at 0. Then run Cell 0 → 1 → 1b → 1c → 1d → 2 → 3 → 4 → 5 → 5b → 6. Check that Cell 0 prints the tag of §2 and Cell 5 reports 76,440,514 parameters.

**Expected time.** PMX-K16 took 1.39 s per step. Removing the pair force took the first-order model from 1.33 to 0.97 s per step. Expect about 1.0 s per step: about 9 hours of training on an A100, one session, plus about an hour for the diagnostic cells.

**After training,** on the best checkpoint:
- the diagnostic cells PMX-K16 ran: 6b-7 (flow or maps), 6b-9 (the damped $V_\theta$ geodesic step), 6b-13 ($\omega \Delta t$ at the endpoint), 6b-15 (the wells), 7 and 8;
- on the CPU:
  - `mechanism_ablations.py` (PN5, the context swap and the velocity reset);
  - `vtheta_omega_dt.py` (layer stiffness);
  - `pair_force_order_probe.py` (this arm's step readings, against the three other corners);
  - `recall_pathway_probe.py` (recall through the context channels, against PMX-K16);
  - `refinement_flow_confirmation.py` (the per-layer flow and the kick share);
  - the causality and conservativity checks (options `nophi`, `pm16`, `pmx16`, `sr2`).

## 7. Caveats

- **One seed.** The pairing removes the evaluation noise, not the training-seed spread, which is unmeasured. The thresholds of PN2 and PN3 lie 0.6% apart, of the order of a seed spread.
- **The hyperparameters were tuned for the second-order models** and are inherited unchanged.
- **L = 2 and 532M tokens.** The balance could change with depth or training length.
- **Wording corrected on 2026-11-25:** §1 and §4 said "the wells' force" where the probe measured the rest of the force (the context potential and the wells together). The calls are unchanged.
