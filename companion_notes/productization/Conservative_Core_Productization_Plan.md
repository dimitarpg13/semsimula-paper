# The Conservative Core: Productization Plan

**What this note is.** The plan for closing the speed and memory gap between the Gen 3 conservative core (SR2, K4-R6, K2-R6 and their first-order and code twins) and the matched GPT-2 at inference. It starts from what was measured on 2026-11-17, lists every lever in two groups (those that leave the trained model unchanged, and those that need a new training run), and sets the order of work. Written 2026-11-17.

**Status.** A plan. The measurements in §1 and §2 are done ([CPU_Inference_Benchmark.md](../depth_ladder/CPU_Inference_Benchmark.md)); no lever in §4 or §5 has been tried. The calls for the GPU arms are not yet written; each gets a pre-registration, confirmed by the author, before it launches.

**Predecessor.** [Fock_Inference_Productionization_Plan.md](Fock_Inference_Productionization_Plan.md) (2026-09-18) did the same for the Gen 2 Fock arm at eight layers: a cost model by multiply–accumulate counts, A100 measurements, and the training-speed levers of the 2B-token runs. §3 below says which of its levers carry over.

---

## Contents

1. [The gap today](#1-the-gap-today)
2. [Where the time and the memory go](#2-where-the-time-and-the-memory-go)
3. [What carries over from the Gen 2 plan](#3-what-carries-over-from-the-gen-2-plan)
4. [Levers that leave the trained model unchanged](#4-levers-that-leave-the-trained-model-unchanged)
5. [Levers that need a new training run](#5-levers-that-need-a-new-training-run)
6. [The order of work](#6-the-order-of-work)
7. [What this plan does not claim](#7-what-this-plan-does-not-claim)
8. [Files and related notes](#8-files-and-related-notes)

---

## 1. The gap today

One forward pass over one sequence (prefill) on an Intel Core i9-9880H, 8 threads, float32, PyTorch 2.2.2, with the closed-form pair force; tokens per second, and in brackets how many times slower than the matched GPT-2 with fused attention:

| model | 512 | 2,048 | 8,192 | peak GB at 8,192 | perplexity, 16 validation windows of 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| matched GPT-2, fused attention | 2,504 | 2,516 | 1,652 | 6.2 | |
| SR2, as trained | 503 (5.0) | 289 (8.7) | 116 (14.2) | 40.1 | 58.28 |
| SR2, memory switches on | 420 (6.0) | 229 (11.0) | 99 (16.7) | 11.7 | 58.28 |
| K4-R6, memory switches on | 615 (4.1) | 351 (7.2) | 118 (14.0) | 11.8 | not measured (settled 59.62) |
| SR2, pair force removed, switches on | 886 (2.8) | 687 (3.7) | 385 (4.3) | 9.6 | 59.36 |
| K4-R6, pair force removed, switches on | 1,190 (2.1) | 861 (2.9) | 434 (3.8) | 9.2 | 60.66 |

The last two rows are the bound: what removing the pair force $V_\varphi$ from the trained model could save, at the perplexity cost of the removal. The book reports it in Appendix C.10 as the bound the work aims to come close to. **Even at the bound the model is 2.1 to 3.8 times slower than fused attention**; closing that remainder is what §4 and §5 are for.

**Decoding is not measured.** No incremental decoder exists for these models. Read from the code, a decoder would keep about 1,040 numbers per token and layer with the pair force and 624 without it (keys 240, states 384): 2,080 and 1,248 per token at two layers, against $2 L d = 6{,}144$ for the matched GPT-2 at eight layers. So the Gen 3 models are not constant-memory: their decoding state grows with the context, as a key–value cache does, at about a fifth to a third of GPT-2's size ([CPU_Inference_Benchmark.md](../depth_ladder/CPU_Inference_Benchmark.md), §1; the book, Appendix C.10).

## 2. Where the time and the memory go

Shares of the CPU profile for SR2 as trained:

| length | elementwise, copies, reductions | matrix products | decompositions | softmax, top-k |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 47.0% | 36.1% | 15.8% | 1.1% |
| 2,048 | 53.3% | 33.3% | 12.4% | 1.1% |
| 8,192 | 66.2% | 24.9% | 7.7% | 1.3% |
| 8,192, pair force removed | 33.7% | 54.7% | 9.6% | 2.0% |

Read with the code, the costs come from four places:

- **The pair force's score network.** It runs on every causal pair of positions with a hidden width of 32. It holds most of the elementwise share at long context (removing it takes that share from 66% to 34% at 8,192 tokens) and, before the memory switches, most of the 40 GB.
- **The well bank $V_\theta$.** A hypernetwork: at each layer and each token it generates the wells' centres, scales and stiff directions from the token's own state and context channels, then evaluates the force. Its projections are the widest matrix products in the model: `B_proj` maps 1,920 numbers to 12,288 per token and layer.
- **The context channels $\xi$.** Five content-routed channels per layer: a softmax over a distance term plus a query–key content term, with the earlier states as values. This is attention with a distance bias, so its cost grows as $T^2$ in prefill. Its share is not separated in the profile.
- **The stiff-mode decompositions.** One batched float64 decomposition of a $32 \times 32$ Gram matrix per token and layer (the `gram` driver, which every ladder model already uses). 8% to 16% of the time.

The Fock registers cost almost nothing once skipped: with the reverse channel off nothing reads them (`skip_unread_registers`).

## 3. What carries over from the Gen 2 plan

The Gen 2 plan found 87.6% of the multiply–accumulates in $V_\theta$'s parameter generation and proposed five levers. Measured since:

| Gen 2 lever | what it was | status on the Gen 3 core |
| --- | --- | --- |
| A, factorize W_B | a bottleneck in `B_proj` | measured on SR2 without retraining: width 512 costs +41%; trained as twin PB, proposed and not run ([Well_Screen_and_Precision.md](../depth_ladder/Well_Screen_and_Precision.md), §5) |
| B, factorize W_μ, W_a | bottlenecks in the centre and scale maps | W_μ cannot be factorized faithfully (Gen 2 plan, §6.1); replacing `a_proj` by one learned scale per well costs +1.2% without retraining (twin PA, proposed and not run) |
| C, context-independent B | fixed or shared stiff directions | refuted at layer 0: a fixed B costs +846%, every well taking the strongest well's B +1,252%; layer 1 is compressible (a shared B costs +5.7%) |
| D, ξ as a recurrence | the channels as exponential averages | no longer exact: the channels are content-routed in every ladder model; see lever T3 |
| E, fuse the score network, cache its projection | tiling the pair force | done in part: the memory switches build the pair force a block of queries at a time |

The closed-form pair force of the Gen 2 plan's §10 is the trained configuration of every Gen 3 model, and the benchmark uses it.

## 4. Levers that leave the trained model unchanged

Each one gives the same logits up to float rounding, so it needs an exactness check, not a training run. The check is the one the memory switches passed: logits and loss against the unmodified model on the same windows, with counters confirming the new path ran (`verify_inference_chunking.py`).

**N1. Fused attention for the context channels.** The channels compute

$$\xi_{k,t} = \sum_{s \le t} \mathrm{softmax}_s\left(b_k(t - s) + \frac{q_{k,t} \cdot \kappa_{k,s}}{\sqrt{d_\kappa}}\right) h_s ,$$

with $b_k$ the channel's distance term, $q$ and $\kappa$ its query and key (width $d_\kappa = 48$) and $h$ the states. This is exactly `scaled_dot_product_attention` with the distance term as an additive mask and the states as values (value width 384, shared by the five channels). On a GPU the distance term can be generated inside the kernel (FlexAttention's score modification), so no $(T, T)$ tensor is ever built. Whether PyTorch 2.2.2 on the CPU takes a fused kernel with an additive mask has to be checked; if it does not, the gain is only what the block-of-queries switch already gives. The matched GPT-2 gains 3.7 times at 8,192 tokens from the same change (443 against 1,652 tokens per second).

**N2. Compiling the layer step.** `torch.compile` on one layer's step, to fuse the elementwise work of the integrator, the bank's force and the pair force's score network. Elementwise work is 47% to 66% of the CPU time, and about half the GPU time in training (the Gen 2 plan, §10.3b). The pair force's top-k selection and the decomposition may break the compiled graph; how much it fuses is what the test measures.

**N3. Pruning layer 1's bank to one well.** Layer 1 runs on one well: keeping only its strongest well costs +0.0% on K4-R6 and on K2-R6 (calls KS5 and KS9, both hits), and six directions suffice. At inference the other wells can be dropped from layer 1's projections. That is a removal, so it changes the logits slightly; its check is the loss on the validation windows, not bit-exactness. Layer 0 cannot be pruned this way: it fills every direction it is given.

**N4. An incremental decoder.** Without one, decoding speed cannot be measured at all, for these models or against GPT-2 with its key–value cache. With the pair force it caches the states and the score network's projections; without it, the channels' keys and the states. It is exact by construction and a prerequisite for any claim about generation speed.

**N5. The decompositions.** 8% to 16% of the time, in float64. A float32 decomposition at inference would be cheaper, but the first float32 version of this solver failed on real matrices (2026-09-17, the note in `_gram_eigh`), so it would need checking on every model. Last in priority.

## 5. Levers that need a new training run

Each changes what the model can express, so each needs a trained arm on one A100, with its calls written and confirmed before launch. A removal from a trained model measures how much the model relies on a term, not what a model trained without it would lose.

**T1. Train without the pair force.** The largest lever: it gives the bound in §1. The pair force carries none of the recall on code: cutting the earlier occurrence from layer 1's pair force changes the far-recall loss by +0.000 nats in both orders ([Second_Order_Memory_Probes.md](../depth_ladder/Second_Order_Memory_Probes.md), probe 6). VF0 (F3.1 trained without $V_\varphi$) is pre-registered and not run ([Coupling_Removal_Costs_CD.md](../depth_ladder/Coupling_Removal_Costs_CD.md), §5); it answers the value question on F3.1. The productization arm is SR2, or K4-R6, trained without it.

**T2. A field form of the pair force.** Only if T1 costs more than the author's tolerance. The routed force is replaced by its interaction with a mixture of a few components, carried by content-gated exponential averages, so its cost per token no longer grows with the context ([Mean_Field_PARFLM_Programme.md](../Mean_Field_PARFLM_Programme.md), §1 and the field arm FA-OWT).

**T3. Content-gated context channels, layer by layer.** The same mixture used for the channels gives a fixed decoding state and prefill linear in $T$ (the mean-field note, §1.1). But **recall runs through layer 1's content-routed channels**: removing the earlier occurrence from them costs +0.794 nats at far-recall positions in SO-CODE and +0.740 in FO-b-CODE, against +0.000 at control positions (probe 6). Replacing layer 1's channels risks exactly that, since a mixture cannot single out one earlier token. The candidate is therefore a hybrid: content-gated channels at layer 0, content-routed at layer 1, decoded with a cache at layer 1 only. What layer 0's content term carries has not been measured. Its first reading is a removal on the trained model, where setting layer 0's query map to zero turns its channels into exact exponential averages (§6, stage 1).

**T4. A smaller bank.** K4-R6 is measured: +2.2% in perplexity, 1.14 times faster than SR2 at 512 tokens and level at 8,192 ([Well_Screen_and_Precision.md](../depth_ladder/Well_Screen_and_Precision.md)). Further candidates are a per-layer bank (layer 0 as now, layer 1 with one well of rank 6, which N3 measures without retraining first) and the twins PA and PB, which shrink the bank's projections. The bank is a small speed lever on its own; it matters most once the pair force is gone and matrix products are 55% of the time.

## 6. The order of work

1. **Stage 1, CPU, no retraining.** First separate the context channels' share of the profile. Then measure N1, N2 and N3, alone and together, on SR2 and K4-R6, with and without the pair force, at 512, 2,048 and 8,192 tokens, each with its exactness check. In the same session, take the layer-0 content-term removal for T3. One job at a time.
2. **Stage 2, CPU.** Build the incremental decoder (N4) for the configuration stage 1 leaves best, and measure decoding speed and state size against the matched GPT-2 with its key–value cache.
3. **Stage 3, GPU, one arm at a time.** First T1 (VF0, then the SR2-class arm without the pair force, with the leanest bank stage 1 supports). Then T3's hybrid, if stage 1 shows layer 0's content term carries little. T2 only if T1 costs more than the tolerance. The author sets the perplexity tolerance before each arm.
4. **After each stage.** Update §1's table, the benchmark note, and the book's Appendix C.10 where the bound or the measured state changes.

## 7. What this plan does not claim

- That the gap closes. The bound in §1 is an upper bound on one lever, read by removal; stage 1 measures how much of the remainder N1 and N2 recover.
- That any of these numbers hold on a GPU. They are CPU prefill numbers; the GPU training profile (the Gen 2 plan, §10.3b) also finds about half the time in elementwise work, but nothing in this note has been timed on a GPU at inference.
- That a model with all levers applied predicts as well as SR2. Each trained lever carries a perplexity cost that only its arm measures, and the costs need not add.

## 8. Files and related notes

- Benchmark and checks: [`cpu_throughput_benchmark.py`](../../notebooks/conservative_arch/scaleup/debug/cpu_throughput_benchmark.py), [`verify_inference_chunking.py`](../../notebooks/conservative_arch/scaleup/debug/verify_inference_chunking.py); outputs in [`results/cpu_bench/`](../../notebooks/conservative_arch/scaleup/results/cpu_bench/).
- The code each lever touches: the context channels in `notebooks/conservative_arch/multixi/model_multixi.py` (`MultiChannelXi.forward`); the pair force and the bank in `notebooks/conservative_arch/parf/model_parf_multixi.py`; the decomposition in `notebooks/conservative_arch/parf/cfc_baoab.py` (`_gram_eigh`); the register skip in `notebooks/conservative_arch/parf/model_fock_parf_multixi.py`.
- The measured state: [CPU_Inference_Benchmark.md](../depth_ladder/CPU_Inference_Benchmark.md). The bank sizes and the twins PA and PB: [Well_Screen_and_Precision.md](../depth_ladder/Well_Screen_and_Precision.md). The recall pathway: [Second_Order_Memory_Probes.md](../depth_ladder/Second_Order_Memory_Probes.md). VF0: [Coupling_Removal_Costs_CD.md](../depth_ladder/Coupling_Removal_Costs_CD.md). The field forms: [Mean_Field_PARFLM_Programme.md](../Mean_Field_PARFLM_Programme.md).
