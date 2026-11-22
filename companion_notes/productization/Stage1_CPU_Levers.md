# Stage 1: The Levers That Need No Retraining, on the CPU

**What this note is.** The results of stage 1 of [Conservative_Core_Productization_Plan.md](Conservative_Core_Productization_Plan.md): where SR2's inference time goes component by component, and what the levers that leave the trained model unchanged recover, each checked against the unmodified model's logits and validation loss. It also takes the first reading for lever T3, the content term of the context channels removed one layer at a time. Measured 2026-11-17 on an Intel Core i9-9880H with 8 threads, float32, PyTorch 2.2.2, one sequence per forward (prefill).

**Status.** Measured. Lever N2 (`torch.compile`) could not run in this environment (§6); it was measured afterwards with decoding (N4), in [Stage2_Decoding.md](Stage2_Decoding.md).

---

## Contents

1. [Where the time goes, by component](#1-where-the-time-goes-by-component)
2. [The levers and their checks](#2-the-levers-and-their-checks)
3. [Speed-ups, within one session](#3-speed-ups-within-one-session)
4. [Against the matched GPT-2](#4-against-the-matched-gpt-2)
5. [Lever T3: what the content term carries at each layer](#5-lever-t3-what-the-content-term-carries-at-each-layer)
6. [What did not run, and what this changes in the plan](#6-what-did-not-run-and-what-this-changes-in-the-plan)
7. [Files](#7-files)
8. [Lever N6: flushing subnormal floats (added 2026-11-22)](#8-lever-n6-flushing-subnormal-floats-added-2026-11-22)

---

## 1. Where the time goes, by component

Wall time of each component, summed over its calls in one forward (two calls each, one per layer), SR2 with the memory switches on (one repetition each):

| component | 512 | 2,048 | 8,192 | 512, pair force removed | 2,048, removed | 8,192, removed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pair force (score network, top-k, router, force) | 38.3% | 54.9% | 69.1% | | | |
| context channels | 5.7% | 14.4% | 19.3% | 9.1% | 31.6% | 63.2% |
| stiff-mode decomposition | 21.3% | 12.4% | 4.4% | 34.3% | 27.8% | 14.5% |
| well bank: generating the wells | 13.8% | 8.0% | 3.1% | 22.8% | 18.2% | 9.7% |
| well bank: harmonic terms and force | 6.0% | 3.5% | 1.4% | 9.2% | 7.3% | 4.3% |
| the rest (integrator, embeddings, output head) | 14.9% | 6.7% | 2.7% | 24.6% | 15.1% | 8.3% |

Two things the operator profile of [CPU_Inference_Benchmark.md](../depth_ladder/CPU_Inference_Benchmark.md) did not show. The stiff-mode decomposition is the largest single cost at short context once the pair force is gone (34% at 512 tokens): the profile counted only the SVD call (8% to 16%), not the Gram products and the set-up around it. And at long context without the pair force, the context channels take almost two thirds of the time.

## 2. The levers and their checks

Each lever was compared with the unmodified model on one fixed validation window of 512 tokens (logits) and on 16 windows of 512 (perplexity; SR2 as trained 58.28, without the pair force 59.36).

| lever | what it does | logits against the reference | perplexity |
| --- | --- | --- | ---: |
| N3, layer 1 pruned | layer 1 computes only the wells it uses (SR2: wells 3 and 4 of 8; K4-R6: well 0 of 4), with the weights from the full softmax; the stiff-mode decomposition shrinks to their columns | against the keep-k screen's masking: max 2.3e-5 (exact); against the unmodified model: max 0.27, mean 0.013, top-1 agreement 100% | 58.27 |
| N5, float64 `eigh` | the decomposition of the 32 × 32 Gram matrix by a symmetric eigensolver instead of an SVD, in float64, same hardenings | max 1.9e-5 (exact) | 58.28 |
| N1, folded fused attention | the context channels by `scaled_dot_product_attention`: the distance term (t − s) log a_k splits into a row constant, which cancels in the softmax, and the per-key term −s log a_k, carried by one extra query and key component; then plain causal attention, no mask | max 0.077, mean 2.2e-4, top-1 agreement 100% | 58.28 |
| N1, masked fused attention | the same kernel with the distance term as an additive mask | max 4.8e-7 | |
| context-channel chunking off | the memory switch kept for the pair force only | exact by construction | |

The folded form is not bit-exact: in float32 the per-key term grows with the position, up to about 6,000 at 8,192 tokens, and loses about $5 \times 10^{-4}$ of resolution there. Neither the perplexity nor the top-1 predictions moved.

**The context channels alone** (SR2's trained query and key maps, `xi_sdpa_microbench.py`), seconds per call:

| variant | 512 | 2,048 | 8,192 |
| --- | ---: | ---: | ---: |
| as trained | 0.033 | 0.393 | 4.60 |
| block-of-queries switch | 0.034 | 0.449 | 5.68 |
| masked fused attention | 0.032 | 0.368 | 4.39 |
| folded fused attention | 0.028 | 0.297 | 3.22 |

With an additive mask, PyTorch 2.2.2 on the CPU falls back to the unfused computation, so the masked form gains nothing. The folded form is 1.17, 1.32 and 1.43 times faster. The block-of-queries switch costs speed on the channels; it is worth having for the memory it saves elsewhere.

## 3. Speed-ups, within one session

Tokens per second, three repetitions each, all in one queue so the configurations share one machine state (memory switches on throughout; in brackets the speed-up over the row above it, or over the stated base):

| configuration | 512 | 2,048 | 8,192 | peak GB at 8,192 | perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| SR2, as trained | 412 | 243 | 91 | 11.6 | 58.28 |
| SR2, layer 1 pruned + `eigh` | 510 (1.24) | 264 (1.09) | 95 (1.04) | 11.7 | 58.27 |
| SR2, pair force removed | 670 | 504 | 287 | 9.6 | 59.36 |
| … + layer 1 pruned + `eigh` | 1,081 (1.61) | 667 (1.32) | 332 (1.16) | 9.6 | 59.35 |
| … + context-channel chunking off | 1,042 (0.96) | 727 (1.09) | 374 (1.12) | 12.5 | 59.35 |
| … + folded fused attention instead | 1,077 (1.00) | 803 (1.20) | 465 (1.40) | 9.6 | 59.35 |
| K4-R6, pair force removed | 930 | 646 | 333 | 9.3 | 60.66 |
| … + layer 1 pruned + `eigh` + folded attention | 1,443 (1.55) | 1,028 (1.59) | 420 (1.26) | 9.2 | 60.65 |

The three levers together (pruning, `eigh`, folded attention) make SR2 without the pair force 1.61 times faster at every length. With the pair force in place they gain only 1.04 to 1.24 times, because the pair force dominates the time.

## 4. Against the matched GPT-2

The CPU was thermally throttled throughout (macOS `CPU_Speed_Limit` 47% to 64%, sampled every 30 s), and absolute speeds drift with it: this morning's benchmark found SR2 5.0 to 14.2 times slower than fused GPT-2, this session finds it 10 to 22 times slower. Only ratios timed back to back are comparable, so GPT-2 with fused attention was timed before and after the ladder models, and the table divides by the mean of its two runs (3,874, 3,264 and 1,977 tokens per second; the two runs differ by 4% to 17%):

| configuration | 512 | 2,048 | 8,192 | how many times slower than GPT-2 |
| --- | ---: | ---: | ---: | --- |
| SR2, as trained (switches on) | 381 | 221 | 91 | 10.2, 14.8, 21.8 |
| SR2, pair force removed | 693 | 496 | 287 | 5.6, 6.6, 6.9 |
| SR2, pair force removed, three levers | 1,067 | 803 | 452 | 3.6, 4.1, 4.4 |
| K4-R6, pair force removed, three levers | 1,460 | 995 | 411 | 2.7, 3.3, 4.8 |

Against SR2 as trained in the same session, the best configurations are 2.8 to 5.0 times faster (SR2) and 3.8 to 4.5 times (K4-R6), at 1.8% and 4.1% in perplexity: SR2's 1.8% is the removal of the pair force; K4-R6's 4.1% adds its smaller bank. The gap to fused attention remains 2.7 to 4.8 times.

## 5. Lever T3: what the content term carries at each layer

The plan's candidate for a fixed decoding state keeps content routing at layer 1, where the recall runs, and replaces it at layer 0. The first reading removes the content term from the trained model on one layer at a time, which turns that layer's channels into exact moving averages:

| content term removed | perplexity | logits against the reference |
| --- | ---: | --- |
| none | 58.28 | |
| layer 0 | 99.14 (+0.53 nats) | top-1 agreement 60.2% |
| layer 1 | 82.43 (+0.35 nats) | top-1 agreement 68.0% |

The trained model relies on content routing at layer 0 even more than at layer 1. This is a removal, which measures reliance, not what a model trained without the term would lose; but it makes the layer-0 replacement a high-risk arm rather than the safe half of a hybrid.

## 6. What did not run, and what this changes in the plan

**`torch.compile` (N2) cannot run under Python 3.12.** PyTorch 2.2.2 does not support it there, and 2.2.2 is the last PyTorch release with wheels for Intel Macs, so a newer PyTorch is not an option either; the same limit rules out the newer CPU attention kernels that accept a mask. Measured later the same day in a Python 3.11 environment ([Stage2_Decoding.md](Stage2_Decoding.md), §5): compiling the layer step gives 0.96 to 1.13 times in prefill; compiling its core gives 1.28 to 1.40 times in decoding.

**What changes.**
- Without the pair force, the short-context cost is now the stiff-mode decomposition and the well bank, and pruning plus `eigh` take 38% off the time at 512 tokens. At long context it is the context channels, and the folded form takes 1.4 times off them.
- With the pair force, nothing in stage 1 matters much: T1 (training without it) stays the first GPU arm.
- T3 as planned (layer 0 content-gated) is weakened by §5. A fixed decoding state would need either a trained arm that shows the loss recovers, or a different form for layer 0.
- Stage 2 (the incremental decoder) is unchanged: it is the only way to measure generation speed.

## 7. Files

- Script: [`cpu_throughput_benchmark.py`](../../notebooks/conservative_arch/scaleup/debug/cpu_throughput_benchmark.py), with the stage-1 options `timers=1`, `prune=<layer>:<wells>` and `mask=1`, `gram=eigh64`, `xi=fold`, `xichunk=0`, `xiplain=<layers>`, `compile=`, `dump=` and `compare=`; [`xi_sdpa_microbench.py`](../../notebooks/conservative_arch/scaleup/debug/xi_sdpa_microbench.py).
- Outputs in [`results/productization/`](../../notebooks/conservative_arch/scaleup/results/productization/): component timers `cpu_bench_sr2_memopt_timers.txt` and `cpu_bench_sr2_novphi_memopt_timers.txt`; checks `cpu_bench_sr2_prune*.txt`, `cpu_bench_sr2_eigh64.txt`, `cpu_bench_sr2_fold_check.txt`; §3 `cpu_bench_f_*.txt`; §4 `cpu_bench_b_*.txt` and `thermal_bracket.log`; §5 `cpu_bench_sr2_xiplain0.txt` and `cpu_bench_sr2_xiplain1.txt`; the microbenchmark `xi_sdpa_microbench_*.txt`.
- The wells each layer uses: [Well_Screen_and_Precision.md](../depth_ladder/Well_Screen_and_Precision.md) and the keep-k curves in `results/dead_wells/` and `results/well_screen/`.

## 8. Lever N6: flushing subnormal floats (added 2026-11-22)

**Where the long-context time went.** Component timers on SR2-NP with every lever on (`prune=1:5 gram=eigh64 xi=fold`, the memory switches, 2026-11-22; `cpu_bench_np_lean_fold_timers.txt`) put the context channels at 13% of the forward at 512 tokens and 63% at 8,192 (9.2 s of 14.7 s). By arithmetic they need fewer multiply-adds than the matched GPT-2's attention: 5 channels of width $48 + 1$ in the scores and 384 in the values, at two layers, against 48 heads of 64 and 64 at eight layers, about 4,330 against 6,144 times $T^2$.

**The cause.** In isolation at 8,192 tokens, the folded fused-attention call took 3.36 s per layer with the model's queries and keys, but 1.37 s with random ones of the same shapes and layout. The folded keys carry the distance term $-s \log \alpha$, which grows with the position $s$; inside the fused softmax the exponentials of the older keys fall into the subnormal range, where x86 arithmetic is slow. With `torch.set_flush_denormal(True)` the same call takes 1.25 s per layer, and its output is bit-identical.

**The lever.** `denormal=flush` in `cpu_throughput_benchmark.py` flushes subnormals for the whole run. One session, matched GPT-2 timed before and after, tokens per second at 512 / 2,048 / 8,192:

| configuration | without the flush | with the flush |
| --- | --- | --- |
| SR2-NP, every lever | 1,485 / 1,093 / 594 | 1,473 / 1,510 / 1,096 |
| matched GPT-2 (fused attention) | 4,628 / 4,216 / 2,905 | 5,185 / 5,482 / 3,245 |
| GPT-2 against SR2-NP | 3.2 / 3.6 / 4.9 times faster | 3.5 / 3.6 / 3.0 times faster |

- SR2-NP gains 1.0, 1.38 and 1.85 times; the gain grows with the context, as the distance term does. Perplexity on the 16 windows is unchanged (58.41) and the logits are identical to the run without the flush. Peak memory is unchanged.
- GPT-2 gains 12% to 30% from the same switch, so the gap narrows only at long context, from 4.9 to 3.0 times at 8,192 tokens.

**The context channels as 30 heads of 64** (`xi=fold64`, measured 2026-11-22). Queries and keys (48 plus the folded distance component) are zero-padded to 64, which leaves every score unchanged, and the 384-wide value is split into six blocks of 64, each attended with its channel's scores: the same computation with equal head widths. Against a float64 reference at 2,048 tokens it is off by 2.3e-4, as the folded form is (2.2e-4); in the model the perplexity and the logits check are unchanged. Interleaved runs, subnormals flushed, every other lever on (`cpu_bench_j_*`):

| | folded (two runs) | 30 heads of 64 (two runs) |
| --- | --- | --- |
| context channels at 8,192 tokens | 2.42, 2.34 s | 1.32, 1.31 s |
| forward at 8,192 tokens, tokens per second | 1,131, 1,176 | 1,348, 1,391 |
| context channels at 512 tokens | 0.012, 0.013 s | 0.022, 0.019 s |
| forward at 512 tokens, tokens per second | 1,422, 1,594 | 1,508, 1,553 |

At long context the 30 heads take the channels 1.8 times faster and the forward 1.18 times; at 512 tokens they are 8 ms slower in the channels, inside the noise of the forward. Use them above about 1,000 tokens. The derivation, the kernel dispatch rule that explains the gain, and the cost and error analysis are in [Stage1b_Context_Channels_as_30_Heads.md](Stage1b_Context_Channels_as_30_Heads.md). With both, the matched GPT-2 (flushed, 3,168 to 3,245 tokens per second at 8,192) is about 2.3 times faster at 8,192 tokens and about 3.5 times at 512. The component timers at 8,192 tokens then read: context channels 23%, well bank 20% plus 8% for its quadratic and force, stiff-mode decomposition 23%, the rest 26%. The remaining time is no longer in the channels. Outputs: `cpu_bench_h_*`, `cpu_bench_i_*`, `cpu_bench_j_*`, `queue_denormal.log` and `queue_fold64.log` in `results/productization/`.
