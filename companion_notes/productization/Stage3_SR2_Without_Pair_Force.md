# Stage 3: SR2 Trained Without the Pair Force (SR2-NP)

**What this note is.** The setup and pre-registration of the first GPU arm of [Conservative_Core_Productization_Plan.md](Conservative_Core_Productization_Plan.md) (lever T1): SR2 trained from scratch without the routed pair force $V_\varphi$, on SR2's recipe and data, so that it pairs with SR2. Written 2026-11-21.

**Status.** Set up and checked on the CPU. **The calls in §4 are proposed and await the author's confirmation before the run starts.**

---

## Contents

1. [Why this arm](#1-why-this-arm)
2. [The configuration](#2-the-configuration)
3. [The build check](#3-the-build-check)
4. [Calls](#4-calls)
5. [What is read after the run](#5-what-is-read-after-the-run)
6. [Launch](#6-launch)
7. [Caveats](#7-caveats)

---

## 1. Why this arm

The productization work disabled the pair force on 2026-11-17, and every lever since was measured on models with $V_\varphi$ removed at inference. Removal measures how much a trained model relies on a term, not what a model trained without it loses: SR2 loses 1.8% in perplexity when its pair force is removed (59.36 against 58.28 on 16 validation windows), and a model that never had the term may lose less, or more. This arm measures it, and it becomes the base every later lever is measured on.

The pair force carries little in SR2. Its share of the layer step is 0.053 at layer 0 and nothing at layer 1 (Cell 6b-12), and on code it carries none of the recall: removing the earlier occurrence from layer 1's pair force changes the far-recall loss by +0.000 nats ([Second_Order_Memory_Probes.md](../depth_ladder/Second_Order_Memory_Probes.md), probe 6). It is also the largest cost at long context: 38% to 69% of SR2's inference time ([Stage1_CPU_Levers.md](Stage1_CPU_Levers.md)).

This arm comes before VF0 (F3.1 trained without $V_\varphi$, pre-registered in [Coupling_Removal_Costs_CD.md](../depth_ladder/Coupling_Removal_Costs_CD.md)). VF0 measures the pair force's value on F3.1, where it carries 0.28 nats; that is what the postponed mean-field agenda needs (the plan's §5.1), not productization.

## 2. The configuration

SR2's Cell 0 with the pair force switched off. With no $V_\varphi$, Cell 0 requires `VPHI_GRAD_PATH` to stay at its default, so the arm keeps five of SR2's six changes and adds one:

| knob | default | SR2 | this arm |
| --- | --- | --- | --- |
| `REVERSE_CHANNEL` | True | False | False |
| `VPHI_GRAD_PATH` | 'default' | 'live' | 'default' (no pair force to make live) |
| `XI_GRAD_PATH` | 'default' | 'live' | 'live' |
| `LOWRANK_DAMPED_FLOW` | False | True | True |
| `VPHI_ANALYTIC_FORCE` | False | True | True (the closed-form branch returns a zero force when there is no pair force) |
| `LAYER_CHECKPOINT` | True | False | False |
| `PAIR_POTENTIAL` | 'sparse_topk' | 'sparse_topk' | 'none' |

Everything else is SR2's: $L = 2$, $d = 384$, eight wells of rank 4, five content-routed context channels, the exact damped flow for the stiff modes, 32,500 steps of WSD (peak 1.2e-3, floor 6e-5), batch 16 × 2 × 512, float32, seed 0, OpenWebText. With the reverse channel off and no pair force, this is the architecture of run 10 of the Gen 2 factorial (multi-ξ SPLM), which was planned but never trained, here with SR2's integrator and live gradients.

## 3. The build check

Built locally through the notebook's own Cells 0–5b at a fresh initialisation, from HEAD `2309849` (`debug/verify_sr2_nophi_build.py`, output `results/productization/verify_sr2_nophi_build_output.txt`):

| | SR2 | this arm |
| --- | --- | --- |
| tag | `…cgqk_norc_vplive_xilive_sr2_L2probe_…` | `…cgqk_norc_nophi_xilive_sr2_L2probe_…` |
| parameters | 76,572,110 | 76,434,307 (the pair force, its score head and its per-layer scale removed) |
| loss and gradient norm at initialisation | 7.5145, 1.067 | 7.5145, 1.067 |
| one forward and backward on the CPU (2 × 512) | 6.70 s | 3.09 s (0.46 times) |
| gradient into earlier tokens, layer 0 / layer 1 step | 2.27 / 0.283 | 2.26 / 0.283 |

The full tag is `xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_nophi_xilive_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn`. Cell 5b prints "NO PAIR POTENTIAL … NOT A LADDER POINT" and "multi-xi SPLM (run 10)"; both are expected. The gradient into earlier tokens stays live through the context channels, the only pathway left between tokens.

## 4. Calls

Against SR2 at 0.53B (settled 58.32; evaluation scatter about 2%). **Proposed 2026-11-21; not yet confirmed.**

| id | prediction | called |
| --- | --- | --- |
| NP1 | settles within +2% of SR2 (59.49 or less): training without the pair force costs no more than removing it from SR2 did | 65% |
| NP2 | settles within +1% of SR2 (58.90 or less) | 40% |
| NP3 | the gap to SR2 in the stable phase (mean over steps 5,000 to 21,000) is at most +0.01 nats | 55% |

The case for NP1: the pair force's share of SR2's step is small, a model trained without it can move that work to the wells and the context channels, and a removal overstates the cost. The case against: the pair force may shape what the wells learn during training, which a removal at the end cannot show.

## 5. What is read after the run

- **Settled perplexity** (mean of the last three evaluations) and the paired gap to SR2 by phase, as for the well-screen arms.
- **On the CPU, with the stage-1 and stage-2 benchmarks:** prefill and decoding speed and the cache size against SR2 with its pair force removed, the matched GPT-2 timed in the same session, and the validation perplexity on the same 16 windows. The architecture is fixed, so the speed should match the bound measured by removal; what changes is the perplexity it comes with.
- **The standard cells** 6b-7, 6b-9, 6b-12, 6b-13 and Cell 7, as for every ladder arm.
- **The recall pathway, if wanted:** whether the context channels carry more of the recall when the pair force was never there (probe 6 of the memory probes, on OpenWebText windows).

## 6. Launch

In Cell 0 of `colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`, set:

```python
REVERSE_CHANNEL=False, XI_GRAD_PATH='live',
LOWRANK_DAMPED_FLOW=True, VPHI_ANALYTIC_FORCE=True, LAYER_CHECKPOINT=False,   # SR2's deltas, VPHI_GRAD_PATH left 'default'
PAIR_POTENTIAL='none'                                                         # the one change
```

Then run Cell 0 → 1 → 1b → 1c → 1d → 2 → 3 → 4 → 5 → 5b → 6. Check that Cell 0 prints the tag above and Cell 5 reports 76,434,307 parameters. Without the pair force the step should be faster and lighter than SR2's; if evaluation runs out of memory on a 40 GB card, set `EVAL_LAYER_CHECKPOINT = True` (recompute during evaluation only; the training arithmetic and the tag are unchanged), as K4-R6 did. Expected: well under SR2's time per step, about 6 to 8 hours on an A100, one session.

## 7. Caveats

- **One seed.** The seed spread has not been measured.
- **The training file differs from SR2's.** Like the well-screen arms, this arm reads the OpenWebText training file rebuilt on 2026-11-12; the registered pool check scored "different pool" by 0.0002 nats at one step, with evidence that the draws are the same ([Well_Screen_and_Precision.md](../depth_ladder/Well_Screen_and_Precision.md)).
- **Not parameter-matched:** 137,803 fewer parameters (0.18%).
- **OpenWebText only.** On code, where SR2's recall runs through the context channels and the pair force carries none of it, the arm would be a separate run.
