# Stage 3: SR2 Trained Without the Pair Force (SR2-NP)

**What this note is.** The setup and pre-registration of the first GPU arm of [Conservative_Core_Productization_Plan.md](Conservative_Core_Productization_Plan.md) (lever T1): SR2 trained from scratch without the routed pair force $V_\varphi$, on SR2's recipe and data, so that it pairs with SR2. Written 2026-11-21.

**Status.** Trained on 2026-11-21 (A100 40 GB, about 0.72 s per step), all 32,500 steps; SCAF leak checks clean. Settled 58.51 against SR2's 58.32 (+0.33%); all three calls in §4 pass. The calls were proposed before the run and not confirmed by the author before it started. Diagnostics and CPU readings in §5; model card published the same day.

---

## Contents

1. [Why this arm](#1-why-this-arm)
2. [The configuration](#2-the-configuration)
3. [The build check](#3-the-build-check)
4. [Calls](#4-calls)
5. [What was read after the run](#5-what-was-read-after-the-run)
6. [Launch](#6-launch)
7. [Caveats](#7-caveats)

---

## 1. Why this arm

The productization work disabled the pair force on 2026-11-17, and every lever since was measured on models with $V_\varphi$ removed at inference. Removal measures how much a trained model relies on a term, not what a model trained without it loses: SR2 loses 1.8% in perplexity when its pair force is removed (59.36 against 58.28 on 16 validation windows), and a model that never had the term may lose less, or more. This arm measures it, and it becomes the base every later lever is measured on.

The pair force carries little in SR2. Its share of the layer step is 0.053 at layer 0 and nothing at layer 1 (Cell 6b-12), and on code it carries none of the recall: removing the earlier occurrence from layer 1's pair force changes the far-recall loss by +0.000 nats ([Second_Order_Memory_Probes.md](../depth_ladder/Second_Order_Memory_Probes.md), probe 6). It is also the largest cost at long context: 38% to 69% of SR2's inference time ([Stage1_CPU_Levers.md](Stage1_CPU_Levers.md)).

This arm comes before VF0 (F3.1 trained without $V_\varphi$, pre-registered in [Coupling_Removal_Costs_CD.md](../depth_ladder/Coupling_Removal_Costs_CD.md)). VF0 measures the pair force's value on F3.1, where it carries 0.28 nats; that is what the postponed mean-field agenda needs (the plan's §5.1), not productization.

## 2. The configuration

SR2's Cell 0 with the pair force switched off. Two of SR2's six changes concern only $V_\varphi$ and stay at their defaults: Cell 5b refuses `VPHI_GRAD_PATH = 'live'` when there is no $V_\varphi$, and `VPHI_ANALYTIC_FORCE` is inert, because with no pair force both of its paths return a zero force. So the arm keeps four of SR2's changes and adds one, five changes from the defaults:

| knob | default | SR2 | this arm |
| --- | --- | --- | --- |
| `REVERSE_CHANNEL` | True | False | False |
| `VPHI_GRAD_PATH` | 'default' | 'live' | 'default' (no pair force to make live) |
| `XI_GRAD_PATH` | 'default' | 'live' | 'live' |
| `LOWRANK_DAMPED_FLOW` | False | True | True |
| `VPHI_ANALYTIC_FORCE` | False | True | True in the run as launched; inert (both paths return a zero force), so False builds the same model |
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
| one forward and backward on the CPU (2 × 512) | 6.2 to 6.7 s | 3.1 to 3.4 s (0.46 to 0.54 times; two runs) |
| gradient into earlier tokens, layer 0 / layer 1 step | 2.27 / 0.283 | 2.26 / 0.283 |

The full tag is `xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_nophi_xilive_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn`. Cell 5b prints "NO PAIR POTENTIAL … NOT A LADDER POINT" and "multi-xi SPLM (run 10)"; both are expected. The gradient into earlier tokens stays live through the context channels, the only pathway left between tokens.

## 4. Calls

Against SR2 at 0.53B (settled 58.32; evaluation scatter about 2%). **Proposed 2026-11-21, before the run; not confirmed by the author before it started.** Scored the same day from the run's `training_log.jsonl` and SR2's Cell 6 output, on the 65 evaluations the two runs share.

| id | prediction | called | result |
| --- | --- | --- | --- |
| NP1 | settles within +2% of SR2 (59.49 or less): training without the pair force costs no more than removing it from SR2 did | 65% | **pass**: 58.51 (58.45, 58.54, 58.55), +0.33% |
| NP2 | settles within +1% of SR2 (58.90 or less) | 40% | **pass** |
| NP3 | the gap to SR2 in the stable phase (mean over steps 5,000 to 21,000) is at most +0.01 nats | 55% | **pass**: −0.0041 nats; this arm is lower at 30 of the 33 evaluations |

The paired gap by phase, in nats (this arm minus SR2; mean over the shared evaluations):

| phase | evaluations | mean gap | this arm higher at |
| --- | ---: | ---: | ---: |
| warm-up (to step 1,625) | 3 | −0.0017 | 1 |
| stable (1,625 to 21,125) | 39 | −0.0035 | 6 |
| decay (21,125 to 32,500) | 23 | +0.0005 | 15 |
| last three | 3 | +0.0033 | 3 |

Both arms reach their best at step 31,000: 56.49 for SR2 and 56.71 here. The arm without the pair force is slightly lower through the stable phase and slightly higher at the end, and every gap is a small fraction of the evaluation scatter. Training without the pair force costs about 0.3% in perplexity, against 1.8% for removing it from SR2 after training: the model moves the pair force's small share of the work elsewhere.

$\omega \Delta t$ at the endpoint (Cell 6b-13, best checkpoint, same 16,384 tokens as SR2): median 6.18 against SR2's 7.55 (layer 0: 7.29 against 8.93; layer 1: 4.31 against 5.46); 99.7% of tokens over the explicit-integrator bound of 2 in both.

The case for NP1: the pair force's share of SR2's step is small, a model trained without it can move that work to the wells and the context channels, and a removal overstates the cost. The case against: the pair force may shape what the wells learn during training, which a removal at the end cannot show.

## 5. What was read after the run

Read 2026-11-21. The model card holds the full tables: [`semsimula-ladder-live-owt-d384-l2-none-norc-nophi-sr2`](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-none-norc-nophi-sr2), in the Gen 3 collection.

**Training cost.** 0.68 s per step on one A100 40 GB, training peak 15.1 GB (19.4 GB with evaluation); K4-R6 took 1.10 s and 31.2 GB on the same card type.

**On 16 validation windows of 512 tokens** (stage 1's windows): 58.41, against 58.28 for SR2 as trained and 59.36 for SR2 with its pair force removed.

**The wells** (`results/well_screen/*sr2np*`). Layer 1 runs on one well: keeping only well 5 costs +0.0% (SR2 needs two). Layer 0 needs all eight: keeping seven costs +15.5%; its weight entropy is 4.51 against SR2's 3.8.

**The decay gain** (`results/decay_noise/sr2np/`): 0.187 nats against SR2's 0.193. Measured over predicted is 0.74 at step 22,500, inside the band of SR2-L1, SR2-2B and GPT-2 2B.

**The standard cells** (results folder of the run):
- Gate 3 at N = 3: +0.404 nats, against SR2's +0.405.
- $\omega \Delta t$ median: 6.18, against SR2's 7.55.
- With no pair force, 6b-9 and 6b-12 read zero deflection from the $V_\theta$ geodesic plus LayerNorm, by construction.

**On the CPU** (one session, mostly unthrottled; GPT-2 timed before and after; `results/productization/cpu_bench_g_*`, `decode_bench_e_*`, `thermal_stage3.log`). Tokens per second at 512, 2,048 and 8,192 tokens:

| configuration | prefill | decoding | perplexity, 16 windows |
| --- | --- | --- | ---: |
| matched GPT-2 (fused attention; key–value cache) | 4,601 / 4,330 / 2,563 | 118 / 105 / 72 | 49.81 (settled) |
| this arm | 865 / 671 / 413 | 29.5 / 28.2 / 23.6 | 58.41 |
| this arm, layer 1 pruned to well 5 + `eigh` (+ folded attention in prefill) | 1,455 / 1,089 / 634 | 38.5 / 36.5 / 27.0 | 58.41 |
| … + compiled layer-step core (decoding) | | 63.9 / 60.1 / 39.1 | |
| SR2, pair force removed, the same levers (wells 3 and 4) | 1,334 / 1,056 / 593 | 60.8 / 57.6 / 43.0 (compiled) | 59.35 |

- **Against GPT-2, with all the levers:** 3.2 to 4.0 times slower in prefill and 1.7 to 1.9 times slower in decoding. The cache is a fifth the size (1,248 numbers per token against 6,144).
- **The decoder** matches the full forward to 5.5e-5, with top-1 agreement 100%.
- **The speed is SR2's with its pair force removed,** within the session's spread. What training without the pair force bought is 1.6% in perplexity at the same speed.

**On the final weights** (read the same day on the CPU; outputs in the run's results folder and `results/recall_pathway/*_owt*`):
- **Causality:** clean. Future perturbation and batch independence are exact; the prefix-only leak tax is $-2.2 \times 10^{-7}$ nats.
- **Conservativity:** every force term passes at both layers (Jacobian asymmetry at most $2.9 \times 10^{-7}$, closed-loop work at most $1.6 \times 10^{-6}$; rerun 2026-11-25 with each layer's own depth code, [Depth_Code_Correction.md](../depth_ladder/Depth_Code_Correction.md), the first run gave $6.7 \times 10^{-7}$).
- **Recall pathway** (probe 6, 16 fresh OpenWebText windows, SR2 on the same windows; the probe in full: [Recall_Pathway_Probe.md](../depth_ladder/Recall_Pathway_Probe.md)):
  - Far recall runs through the layer-1 context channels as in SR2: +0.364 nats when the earlier occurrence is removed, against SR2's +0.376.
  - SR2's pair force carries none of it on OpenWebText either (+0.000).
  - This arm leans more on the velocity. Zeroing the velocity into layer 1 costs +0.111 nats on control tokens, against SR2's +0.064 (paired difference +0.046, se 0.006).
  - Gate 1 of 6b-7 agrees (+3.21 in perplexity against +2.22).
- **What it means for the competing mechanisms** (book §28): the velocity partly stands in for the pair force; the mathematics, the definitions and the interaction test that would settle it are in [Velocity_and_Pair_Force_as_Competing_Mechanisms.md](../depth_ladder/Velocity_and_Pair_Force_as_Competing_Mechanisms.md).

## 6. Launch

In Cell 0 of `colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`, set:

```python
REVERSE_CHANNEL=False, XI_GRAD_PATH='live',
LOWRANK_DAMPED_FLOW=True, LAYER_CHECKPOINT=False,   # SR2's deltas; VPHI_GRAD_PATH and VPHI_ANALYTIC_FORCE stay at their defaults
PAIR_POTENTIAL='none'                               # the one new change
```

These five are the only settings that differ from the notebook's defaults (checked 2026-11-21 by executing Cell 0 both ways at `eda2c35`; with `VPHI_ANALYTIC_FORCE` left at its default, the build check gives the same tag, parameter count, loss and gradients as with it on). Then run Cell 0 → 1 → 1b → 1c → 1d → 2 → 3 → 4 → 5 → 5b → 6. Check that Cell 0 prints the tag above and Cell 5 reports 76,434,307 parameters.

After training, on the best checkpoint, the diagnostic cells of the recent arms: 6b-7 (flow or maps), 6b-9 (the damped $V_\theta$ geodesic step), 6b-12 (is the forcing sparse), 6b-13 ($\omega \Delta t$ at the endpoint) and 7 (component diagnostics), then 8 (the training curve). Cells 6b-9 and 6b-12 switch the pair force off by zeroing its output, which is already zero here, so they run and their $V_\varphi$ rows read zero by construction. Cells 6b-8, 6b-10 and 6b-11 concern the reverse channel or the register path and do not apply. Without the pair force the step should be faster and lighter than SR2's; if evaluation runs out of memory on a 40 GB card, set `EVAL_LAYER_CHECKPOINT = True` (recompute during evaluation only; the training arithmetic and the tag are unchanged), as K4-R6 did. Expected: well under SR2's time per step, about 6 to 8 hours on an A100, one session.

## 7. Caveats

- **One seed.** The seed spread has not been measured.
- **The training file differs from SR2's.** Like the well-screen arms, this arm reads the OpenWebText training file rebuilt on 2026-11-12; the registered pool check scored "different pool" by 0.0002 nats at one step, with evidence that the draws are the same ([Well_Screen_and_Precision.md](../depth_ladder/Well_Screen_and_Precision.md)).
- **Not parameter-matched:** 137,803 fewer parameters (0.18%).
- **OpenWebText only.** On code, where SR2's recall runs through the context channels and the pair force carries none of it, the arm would be a separate run.
