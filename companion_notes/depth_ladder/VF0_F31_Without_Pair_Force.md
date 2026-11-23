# VF0: F3.1 Trained Without the Pair Force

**What this note is.** The launch record and pre-registration of VF0, the first cell of the trained-without factorial (protocol §5.21, [Coupling_Removal_Costs_CD.md](Coupling_Removal_Costs_CD.md), §5). The arm is F3.1, the conservative-only model with the split stiff-mode step, trained without the pair force $V_\varphi$. Written 2026-11-22.

**Status.** Set up; build and switch checks passed; not launched. The author decided on 2026-11-22 to run it ("for the sake of completing the investigation on the pair force as a valid mechanism for replacement of others and updating the book"). The queue position is the author's call. VF-a has been binding since 2026-10-11; VN1 and VN2 (§4) were confirmed by the author on 2026-11-22 and are binding.

---

## Contents

1. [Why this arm](#1-why-this-arm)
2. [The configuration](#2-the-configuration)
3. [The checks](#3-the-checks)
4. [Calls](#4-calls)
5. [What the result decides](#5-what-the-result-decides)
6. [Launch](#6-launch)
7. [Caveats](#7-caveats)

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

## 7. Caveats

- **One seed.** The pairing removes the evaluation noise, not the training-seed spread, which is unmeasured.
- **L = 2 and 532M tokens.** The pair force's value could change with depth or training length; this arm measures it at the ladder's scale only.
- **F3.1's step time is not comparable.** Its run predates the speed switches, so its 1.28 s per step cannot be set against VF0's. The CPU benchmark of both checkpoints is the like-for-like speed comparison.
