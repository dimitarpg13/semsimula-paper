# CPU Inference: Cost, Memory Switches and the Bound

**What this note is.** A readable account of the inference-cost work of 2026-11-17 on the Gen 3 conservative models: what their pathways between tokens cost, how they compare with the matched GPT-2 on a CPU, three inference-only memory switches, and the upper bound on what replacing the pair force $V_\varphi$ could save. The full log is in the [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md) (the CPU-benchmark entries after `p-bank-memory-conjecture`); the book reports the same numbers in its Appendix C.10, "Gen 3 models: what is and is not constant-memory".

**Status.** Measured; nothing here is a pre-registered test. The numbers describe the models as they are, not the target.

---

## Contents

1. [What the models keep between tokens](#1-what-the-models-keep-between-tokens)
2. [The benchmark](#2-the-benchmark)
3. [A trap, and the guard against it](#3-a-trap-and-the-guard-against-it)
4. [Speed and memory](#4-speed-and-memory)
5. [Where the time goes](#5-where-the-time-goes)
6. [The memory switches](#6-the-memory-switches)
7. [The upper bound without the pair force](#7-the-upper-bound-without-the-pair-force)
8. [What would restore constant memory](#8-what-would-restore-constant-memory)
9. [Files](#9-files)

---

## 1. What the models keep between tokens

A code audit of SR2 (the conservative-only model with the exact damped flow, two layers, $d = 384$) found two pathways that carry information from one position to another and reach the prediction:

- **The context channels $\xi$.** Five channels, each a softmax over an exponential-decay term plus a content term: the query of the current state against learned keys (width 48) of the earlier states, with the earlier states as values. This is five-head attention with a distance bias (`XI_CONTENT_ROUTE = True`, on in every ladder model). Without the content term the channels are exponential moving averages, a recurrence with a fixed state.
- **The routed pair force $V_\varphi$.** Its score network runs on every causal pair of positions, with a hidden layer of width 32, before the 16 highest-scoring earlier tokens are selected; its router term reads the soft weights of every earlier token.

Everything else acts within one position: the well bank $V_\theta$ reads only the position's own state and context channels, and the integrator, the mass and the velocity are per position. The Fock registers are computed at every layer, but with the reverse channel off nothing reads them.

| pathway | prefill tensors per layer | decoding state per token and layer |
| --- | --- | ---: |
| context channels, content-routed | (K, T, T) weights and (B, K, T, T) scores, K = 5 | 624 numbers (keys 240, state 384) |
| context channels without the content term | none needed (a recurrence) | fixed: 5 × 384 per layer |
| pair force | (B, T, T, 32) score-network tensors, twice | 416 numbers (state 384, projection 32) |
| Fock registers | (B, T, 32, 384) tensors | none: unread |
| well bank, integrator, mass | per position | none |

So a decoder would keep about 1,040 numbers per token and layer, 2,080 per token at two layers, against $2 L d = 6{,}144$ for the matched GPT-2 ($L = 8$). The state grows with the context, as a key–value cache does, with about a third of its size. No incremental decoder exists for these models; the counts are read from the code.

## 2. The benchmark

One forward pass over one sequence (prefill), at 512, 2,048 and 8,192 tokens, on an Intel Core i9-9880H with 8 threads, float32, PyTorch 2.2.2; median of three timed passes after a warm-up, and the process's peak resident memory. Past the trained context the position tables are tiled, which is valid for timing only. The baselines are the matched GPT-2 (eight layers, $d = 384$) with the attention it was trained with, and the same weights with the fused kernel `scaled_dot_product_attention`.

## 3. A trap, and the guard against it

The first run timed the ladder models without their pair force. The probe scripts build a model from its folder tag, which leaves `VPHI_ANALYTIC_FORCE` at its notebook default (off), and under `torch.no_grad` the autograd pair force silently dropped out: SR2's loss was 4.1637 instead of 4.1476 on four windows of 1,024 tokens. Every probe runs with gradients enabled and was unaffected; only the first benchmark was wrong, and it was redone with the closed-form force, the trained configuration. The model now raises an error in that case instead of returning a zero force (`model_parf_multixi.py`, the guard on the autograd pair-force branch).

## 4. Speed and memory

Tokens per second, with the closed-form pair force:

| model | 512 | 2,048 | 8,192 | peak GB at 8,192 |
| --- | ---: | ---: | ---: | ---: |
| matched GPT-2, fused attention | 2,504 | 2,516 | 1,652 | 6.2 |
| matched GPT-2, attention as trained | 2,887 | 1,529 | 443 | 8.0 |
| SR2 | 503 | 289 | 116 | 40.1 |
| SR2, memory switches on | 420 | 229 | 99 | 11.7 |
| K4-R6 (4 wells of rank 6) | 572 | 302 | 113 | 41.1 |
| K4-R6, memory switches on | 615 | 351 | 118 | 11.8 |
| K2-R6 (2 wells of rank 6) | 656 | 339 | 124 | 39.7 |

SR2 is 5.0, 8.7 and 14 times slower than fused attention at the three lengths. The well bank's size is a small speed lever: K4-R6 is 1.14 times faster than SR2 at 512 tokens and level at 8,192, and K2-R6 1.30 times at 512, for 2.2% and 11.7% in perplexity. (The first run reported 1.6 to 1.8 times; it timed the models without their pair force, which made the bank a larger share.)

## 5. Where the time goes

At 512 tokens most of SR2's time goes to the well bank's matrix products (43% to 47% of the profile at 512 and 2,048 tokens) and to the per-token decompositions that give the stiff modes (18% to 21%). At 8,192 tokens two thirds is elementwise work (66%), most of it the pair force's score network over the $(T, T, 32)$ tensors; matrix products 25%, decompositions 8%.

Sharing the stiff basis across neighbouring tokens does not reduce the decomposition cost: a basis a causal model can use costs 0.03 nats on layer 1 alone and 0.64 on both layers (Probe B, in [the mean-field programme note](../Mean_Field_PARFLM_Programme.md)).

## 6. The memory switches

Three inference-only switches, off by default and active only under `no_grad`, so training and the probes are untouched:

- `cfg.vphi_query_chunk`: the closed-form pair force built a block of query positions at a time against all sources;
- `xi_module.query_chunk`: the content-routed context channels built the same way;
- `cfg.skip_unread_registers`: the Fock registers skipped when nothing reads them.

Every term is a sum over one query's own row, so the results are unchanged: on SR2 the logits differ by at most $2.3 \times 10^{-5}$ and the loss not at all, with counters confirming each path ran, and with gradients enabled the outputs are bit-identical (`verify_inference_chunking.py`). With blocks of 512, SR2's peak memory at 8,192 tokens falls from 40.1 to 11.7 GB, for 16% to 21% in speed on SR2; on K4-R6 the switches are 4% to 16% faster.

## 7. The upper bound without the pair force

The pair force carries little in the trained model, and it dominates the long-context cost. Removing it entirely (score network, top-k, router and force) gives the upper bound on what replacing it could save:

| configuration | 512 | 2,048 | 8,192 | peak GB | validation perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| SR2, as trained | 503 | 289 | 116 | 40.1 | 58.28 |
| SR2, pair force removed | 757 | 615 | 386 | 14.0 | 59.36 |
| SR2, pair force removed, switches on | 886 | 687 | 385 | 9.6 | 59.36 |
| K4-R6, pair force removed, switches on | 1,190 | 861 | 434 | 9.2 | 60.66 |

Removing the pair force makes SR2 1.5, 2.1 and 3.3 times faster and costs 1.8% in perplexity; with K4-R6's bank and the switches the bound is 2.4 to 3.7 times SR2's speed at 9.2 GB. This is the bound the work aims to come close to, not an achievement. A removal from a trained model measures how much the model relies on the term, not what a model trained without it would lose; the pre-registered arm without the pair force (VF0) measures that. Even at the bound the model is 2.1 to 3.8 times slower than fused attention, and with the pair force gone most of its time is matrix products (55% at 8,192 tokens): the bank and the context channels.

## 8. What would restore constant memory

Both pathways have a fixed-size form. A mixture of $M$ components whose weights and means are content-gated exponential averages of the earlier states is a linear recurrence. Used for the context channels it keeps a dependence on content; used as the source of a field form of $V_\varphi$ it replaces the routed pair force. Together they would make every coupling a fixed-size recurrence and the decoding state independent of the context length ([the mean-field programme note](../Mean_Field_PARFLM_Programme.md), §1.1). What either costs in perplexity is untested.

## 9. Files

- Scripts in `notebooks/conservative_arch/scaleup/debug/`: [`cpu_throughput_benchmark.py`](../../notebooks/conservative_arch/scaleup/debug/cpu_throughput_benchmark.py) (options `analytic`, `chunk`, `skipreg`, `novphi`, `loss`), [`verify_inference_chunking.py`](../../notebooks/conservative_arch/scaleup/debug/verify_inference_chunking.py), [`shared_stiff_basis_probe.py`](../../notebooks/conservative_arch/scaleup/debug/shared_stiff_basis_probe.py).
- Outputs: [`results/cpu_bench/`](../../notebooks/conservative_arch/scaleup/results/cpu_bench/), [`results/shared_basis/`](../../notebooks/conservative_arch/scaleup/results/shared_basis/).
- The switches and the guard: `notebooks/conservative_arch/parf/model_parf_multixi.py`, `notebooks/conservative_arch/multixi/model_multixi.py`, `notebooks/conservative_arch/parf/model_fock_parf_multixi.py`.
- The well-bank sizes behind K4-R6 and K2-R6: [Well_Screen_and_Precision.md](Well_Screen_and_Precision.md).
