# VF0: F3.1 Trained Without the Pair Force

**What this note is.** The launch record and pre-registration of VF0, the first cell of the trained-without factorial (protocol §5.21, [Coupling_Removal_Costs_CD.md](Coupling_Removal_Costs_CD.md), §5). The arm is F3.1, the conservative-only model with the split stiff-mode step, trained without the pair force $V_\varphi$. Written 2026-11-22.

**Status.** **Trained and scored (2026-11-26): VF0 settled at 61.06, $v_0 = 0.056$. VF-a and VN1 hit, VN2 misses** (§7). The run finished on 2026-11-25; the calls were binding before launch (VF-a since 2026-10-11, VN1 and VN2 since 2026-11-22). The causality check is clean; the other CPU readings of §6 are not yet run. It ran first in the GPU order the author set on 2026-11-24 (then twin PB, then PMX-K16-NP).

---

## Contents

1. [Why this arm](#1-why-this-arm)
2. [The configuration](#2-the-configuration)
3. [The checks](#3-the-checks)
4. [Calls](#4-calls)
5. [What the result decides](#5-what-the-result-decides)
6. [Launch](#6-launch)
7. [Results](#7-results)
8. [Caveats](#8-caveats)

---

## 1. Why this arm

**Reliance is not value.** Switching the pair force off in the trained F3.1 costs 0.282 nats per token (CD, +33% in perplexity). That is reliance: what the trained model routes through the term. Its value is what a model trained without it loses. In every model where both were measured, the value was far smaller than the reliance:
- SR2: reliance 1.8%, value 0.3% (SR2-NP, [Stage3_SR2_Without_Pair_Force.md](../productization/Stage3_SR2_Without_Pair_Force.md));
- the Fock channel: reliance 1.49 nats, value 0.084.

F3.1 is where the pair force carries the most. It is the model's only token-to-token force besides the context channels.

**Why now.** Three threads meet here.
- **The pair force as a mechanism that stands in for others.** Where the exact stiff-mode flow, the registers or the velocity are absent, the pair force takes up work ([Velocity_and_Pair_Force_as_Competing_Mechanisms.md](Velocity_and_Pair_Force_as_Competing_Mechanisms.md)). VF0 measures how much of that work is irreplaceable when training can route around it. It is the value the book's pair-force batch needs.
- **The F3.1-against-SR2 design choice.** F3.1 trains about a third faster than SR2 at the same perplexity (57.76 against 58.32), but pays for the pair force at inference. If VF0 loses little, F3.1 without the pair force is a fast model in its own right.
- **Probe A.** In F3.1 the pair force acts almost entirely as a fixed landscape in the token's own state: a context-free stand-in with one centre recovers 94% of its removal cost ([Probe_A_Field_Pair_Force.md](../mean_field/Probe_A_Field_Pair_Force.md)). A landscape is the kind of term the context potential $V_\theta$ can learn, so training without the pair force may move much of it into $V_\theta$. Probe A's call A2 missed, which stops the mean-field programme's GPU stages, VF0 among them. The author chose to run VF0 under this protocol entry instead, because its question does not depend on any field.

## 2. The configuration

F3.1's Cell 0 with the pair force switched off, following the SR2-NP rule. `VPHI_GRAD_PATH` returns to its default, because Cell 5b refuses a live pair-force path when there is no pair force.

| setting | default | F3.1 | this arm |
| --- | --- | --- | --- |
| `REVERSE_CHANNEL` | True | False | False |
| `XI_GRAD_PATH` | 'default' | 'live' | 'live' |
| `VPHI_GRAD_PATH` | 'default' | 'live' | 'default' |
| `PAIR_POTENTIAL` | 'sparse_topk' | 'sparse_topk' | **'none'** |
| `LAYER_CHECKPOINT` | True | default | False (speed switch, same arithmetic) |

Every other setting stays at its default, as in F3.1: no exact stiff-mode flow, no wells, second order. Same corpus, budget and schedule as F3.1: 32,500 steps × 32 × 512 = 532,480,000 OpenWebText tokens, WSD schedule, peak learning rate 1.2e-3, seed 0. VF0 sees the same training and evaluation batches as F3.1, so their difference is paired.

**Tag:**

```text
xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_nophi_xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn
```

F3.1's tag with `vplive` replaced by `nophi`.

## 3. The checks

Run 2026-11-22 on the CPU, from the current notebook.

**Build check.** Cells 0 to 5b were run through `debug/verify_pm_switch.py`'s `build`, with the Cell 0 replacements derived from each tag by `debug/copy_split.py`'s `tag_repl`, at seed 0:

| | F3.1 (rebuilt) | this arm |
| --- | ---: | ---: |
| tag printed by Cell 0 equals the folder tag | yes | yes |
| trainable parameters | 76,572,109 | **76,434,306** |
| pair force present | yes | no |
| loss at initialisation on a random batch of 2 × 64 | 12.9388 | 12.9388 |

The difference, 137,803 parameters, is the pair force, its score head and its per-layer scale, the same difference as between SR2 and SR2-NP. Cell 5b counts one non-trainable parameter more (76,434,307). Probe A rebuilt F3.1 from the same notebook and loaded its trained weights, so the notebook still reproduces F3.1.

**Switch check.** `debug/verify_pair_potential_none.py`, first run 2026-10-01, rerun 2026-11-22. Its multi-ξ case with `XI_GRAD_PATH = 'live'` is this arm's build:
- 76,434,307 parameters, 137,803 fewer than the same configuration with the pair force;
- the forward pass and backward pass run;
- the gradient into earlier tokens is exactly 0 with the context channels' default gradient path and positive with the live path (2.22 at layer 0, 0.34 at layer 1): with the pair force gone, the context channels are the only path between tokens, and it is live.

## 4. Calls

"Settled" is the mean of the last three evaluations, as for every ladder model. F3.1 settled at 57.76. The value is $v_0 = \ln(\mathrm{PPL}_{\mathrm{VF0}} / 57.76)$.

| id | prediction | called | status |
| --- | --- | ---: | --- |
| VF-a | the pair force is worth something when nothing else competes: $v_0 \ge 0.03$ (VF0 settles at 59.52 or above) | 70% | pre-registered 2026-10-11, binding |
| VN1 | training recovers at least half of the reliance: $v_0 \le 0.141$ (VF0 settles at 66.5 or below) | 75% | proposed and confirmed 2026-11-22, binding |
| VN2 | VF0 settles at or below SR2-NP (58.51): F3.1 without the pair force matches the productization model | 20% | proposed and confirmed 2026-11-22, binding |

**Reasoning for VN1 and VN2.** In both precedents the value was a sixth or less of the reliance, and Probe A says F3.1's pair force is mostly a landscape that $V_\theta$ could absorb. Against that, F3.1 has no exact flow, no registers and no wells, so less is left to absorb it, and its reliance is twenty times SR2's. Matching SR2-NP needs $v_0 \le 0.013$, within VF-a's miss region.

## 5. What the result decides

- **VF-a misses ($v_0 \lt 0.03$).** The pair force has little value even in the model that leans on it most. In the book it is a mechanism the trained model uses where it is present but can do without. F3.1 without the pair force is a candidate fast model, and the CPU benchmark decides how fast. No stand-in is needed.
- **VF-a hits, VN1 hits.** The pair force has real value, but training recovers most of its reliance. A context-free landscape arm on F3.1 becomes worth registering, as post hoc and limited to the two-layer model ([Probe_A_Field_Pair_Force.md](../mean_field/Probe_A_Field_Pair_Force.md), §7). It is scored against VF0.
- **VN1 misses.** The pair force carries value that nothing else in F3.1 can absorb in training. The book states that plainly, and the stand-in arm becomes the main route to a fast F3.1.

Whatever the outcome, VF0 fills the first cell of the VF factorial. VF1 (PM1 without the pair force) is the next cell, and it needs its own switch check with the wells.

## 6. Launch

In Cell 0 of `colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`, set:

```python
REVERSE_CHANNEL=False, XI_GRAD_PATH='live',
LAYER_CHECKPOINT=False,                   # speed switch, same arithmetic
PAIR_POTENTIAL='none'                     # the one change from F3.1; VPHI_GRAD_PATH stays at its default
```

Leave every other switch at its default (in particular `LOWRANK_DAMPED_FLOW`, `POISSON_MODES`, `FIRST_ORDER_OVERDAMPED` and `TIE_EMBEDDINGS`). Then run Cell 0 → 1 → 1b → 1c → 1d → 2 → 3 → 4 → 5 → 5b → 6.
- **Cell 0** prints the tag of §2.
- **Cell 5b** prints two banners: "NO PAIR POTENTIAL (PAIR_POTENTIAL='none') -- NOT A LADDER POINT" and the banner every arm with the live context-channel gradient prints ("SOURCE-GRADIENT PROBE: XI_GRAD_PATH=live -- NOT A LADDER POINT"). Both are expected. It reports 76,434,307 parameters.

**Expected time.** SR2-NP, SR2 without the pair force, ran at 0.68 to 0.72 s per step on an A100 40 GB with a training peak of 15.1 GB. F3.1's split step is cheaper than SR2's exact one. Expect about 0.5 to 0.7 s per step: 4.5 to 6.5 hours, one session, plus about an hour for the diagnostic cells.

**After training,** on the best checkpoint:
- the diagnostic cells F3.1 ran: 6b-7 (flow or maps), 6b-9 (the $V_\theta$ geodesic step; its pair-force rows read zero by construction), 6b-13 ($\omega \Delta t$ at the endpoint), 7 and 8;
- on the CPU:
  - `mechanism_ablations.py` (the velocity reset and the context swap, against F3.1's readings; its velocity reset costs +34%);
  - `vtheta_omega_dt.py` (did the landscape stiffen to take the pair force's work?);
  - `recall_pathway_probe.py` (recall through the context channels);
  - the causality check;
  - the CPU inference benchmark against F3.1 and SR2-NP, the measure of the fast-model claim.

## 7. Results

Scored 2026-11-26 from the run's own log and diagnostic cells (`results/<tag>/`, the tag of §2). Nothing below changed a call; the calls are scored on the letter of §4.

### 7.1 The calls

The run went 32,500 steps without a restart, a spike or a watchdog trigger, and ended with the leak monitor clean. The last three evaluations read 61.01, 61.10 and 61.08, so VF0 **settles at 61.06**, and

$$
v_0 = \ln (61.06 / 57.76) = 0.056 .
$$

| id | threshold | measured | outcome |
| --- | --- | --- | --- |
| VF-a | settles at 59.52 or above | 61.06 | **hit** |
| VN1 | settles at 66.5 or below | 61.06 | **hit** |
| VN2 | settles at 58.51 or below | 61.06 | **miss** |

**How firm VF-a is.** The evaluations scatter by about 0.015 nats around their trend, in VF0 and in F3.1 alike, so a mean of three has a standard error near 0.009 and the difference of two such means near 0.012. VF-a's margin is 0.026, about two standard errors. Two other readings agree:

- a straight-line fit of the last twelve evaluations (steps 27,000 to 32,500), the same fit for both models, gives 59.89 against 56.78 at step 32,500, so 0.053;
- best checkpoint against best checkpoint (59.23 at step 31,000 against 57.35) gives 0.032, still above 0.03. VF0's best is one low evaluation: the next three are all near 61.

**The value against the reliance.** Switching the pair force off in the trained F3.1 costs 0.282 nats. Training without it costs 0.056, a fifth of that. The two precedents of §1 were a sixth (SR2) and an eighteenth (the Fock channel). So training recovers most of the reliance here too, but less of it than in SR2, as the reasoning of §4 expected for a model with nothing else to absorb it.

### 7.2 The two-by-two: the stiff-mode step and the pair force

VF0 completes a square. F3.1 and SR2 differ only in the stiff-mode step (split in F3.1, exact flow in SR2, `LOWRANK_DAMPED_FLOW`); SR2-NP and VF0 are the same two models trained without the pair force.

| stiff-mode step | with the pair force | without | value of the pair force, nats |
| --- | ---: | ---: | ---: |
| split (F3.1, VF0) | 57.76 | 61.06 | 0.056 |
| exact flow (SR2, SR2-NP) | 58.32 | 58.51 | 0.003 |

- **The two mechanisms stand in for each other.** The pair force is worth 0.056 nats on the split step and 0.003 on the exact flow. The exact flow costs 0.010 nats with the pair force and gains 0.043 without it. The interaction is 0.052 nats: almost all of each mechanism's value is the work the other one would otherwise do.
- **For the productization line.** Without the pair force the exact flow is the better design by 4.4% (VN2's miss). F3.1's advantage in training time does not buy a model without the pair force that matches SR2-NP.

One seed per cell, so the interaction carries the training-seed spread of four runs, which is unmeasured.

### 7.3 What the diagnostic cells show

Best checkpoint (step 31,000), against F3.1 and SR2-NP read by the same cells:

| reading | VF0 | F3.1 | SR2-NP |
| --- | ---: | ---: | ---: |
| stiffness of the landscape, layer 0 (median of the stiffest mode's frequency times the step) | 16.98 | 4.34 | 7.29 |
| the same, layer 1 | 4.85 | 4.28 | 4.31 |
| velocity reset at inference, change in perplexity | +5.7% | +34.3% | +5.6% |
| refinement, three substeps in place of two, change in perplexity | +51.0 | +80.7 | +28.6 |
| distance of the step from the landscape's damped geodesic, mean of the two layers | 0.767 | 0.665 | 0.498 |
| step size relative to the state, layer 0 | 5.62 | 6.27 | 5.97 |
| step size relative to the state, layer 1 | 0.77 | 1.07 | 0.75 |

- **The landscape at layer 0 took the pair force's place.** Its stiffest modes are four times stiffer than F3.1's ($\omega \Delta t$ median 17.0 against 4.3) and stiffer than SR2-NP's (7.3). This fits Probe A, where F3.1's pair force acted almost entirely as a fixed landscape in the token's own state: trained without it, $V_\theta$ grows that landscape itself. Layer 1 is unchanged. The cell's automatic verdict ("band [3.3, 4.2], MISS") is the ladder's band for models with the pair force and is not a call of this note. The split step integrates these modes exactly, so the stiffness is not an instability.
- **The model leans on the velocity far less.** Resetting the velocity between the layers costs 5.7%, against 34% in F3.1 and the same 5.6% as SR2-NP. This is reliance, not value: in F3.1 the carried velocity was used together with the pair force, and without it the model does not route as much through inertia.
- **It is still a map, not a flow.** Refining the step at a fixed total time makes it worse, as for F3.1 and SR2-NP (Gate 3 of 6b-7).
- **The geodesic reading has nothing left but the LayerNorm.** Without the pair force every force is the landscape's, so the step's distance from the landscape's geodesic (0.767) is the layer-end LayerNorm's alone; in F3.1 the 0.665 includes the pair force.
- Cell 7: no register or reverse-channel term is active, as configured. Cell 8 and 6b-12 were not run.

**Speed.** 0.81 s per step including evaluations, on a GPU with 102 GB of memory, not the A100 40 GB that SR2-NP ran on, so the step times cannot be compared (§8). The CPU benchmark of §6 is the like-for-like measure.

### 7.4 What the result decides

By §5, **VF-a and VN1 hit**: the pair force has real value in F3.1, and training recovers most of its reliance. §5 makes a context-free landscape arm on F3.1 worth registering, post hoc and limited to two layers, scored against VF0. Two facts weigh on that choice, and it is the author's:

- the landscape already grew stiffer to take the pair force's work, so a context-free stand-in may add little that $V_\theta$ has not absorbed;
- to be useful for productization the stand-in would have to recover the 4.4% gap to SR2-NP, which has no pair force and no stand-in.

**Causality check (CPU, 2026-11-26): clean.** Perturbing the future changes nothing (0 to machine precision), and scoring on the prefix alone moves a logit by at most $3.7 \times 10^{-5}$ (leak tax $7 \times 10^{-7}$ nats). Output: `causality_check_output.txt` in the run's results folder.

**Not yet done.** The other CPU readings of §6: the mechanism ablations (velocity reset and context swap), the landscape stiffness by `vtheta_omega_dt.py`, the recall pathway and the CPU benchmark against F3.1 and SR2-NP. `mechanism_ablations.py` and `vtheta_omega_dt.py` are now written for the PMX models and need an F3.1 path first.

## 8. Caveats

- **One seed.** The pairing removes the evaluation noise, not the training-seed spread, which is unmeasured.
- **L = 2 and 532M tokens.** The pair force's value could change with depth or training length; this arm measures it at the ladder's scale only.
- **F3.1's step time is not comparable.** Its run predates the speed switches, so its 1.28 s per step cannot be set against VF0's. The CPU benchmark of both checkpoints is the like-for-like speed comparison.
