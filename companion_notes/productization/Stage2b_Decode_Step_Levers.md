# Stage 2b: Levers on the Decode Step

**What this note is.** The design, and as they come in the results, of three controlled experiments on the cost of one decoding step of the conservative core. They follow from the attribution in [Stage2_Decoding.md](Stage2_Decoding.md), §4: a decode step issues about 2,900 operator calls (dispatch level; the profiler counts 5,854 with nested calls), against about 1,000 for the matched GPT-2, and the step is bound by per-call overhead. The levers are taken one at a time, each against the step before it. Written 2026-11-23.

**Status.** Step 1 done (2026-11-23, §3.5): bit-identical logits, and the decode step 29% faster at context 512 in the productization configuration (19% at 8,192), prefill unchanged. Step 2 designed; its calls P1 to P4 were confirmed by the author before any timing (§4). **Step 1 is not pre-registered:** a harness smoke test meant as the null comparison (base against base) was launched as base against step 1, so a first reading was seen before any call was written (§3.4). Steps 2 and 3 will be pre-registered: calls written and confirmed by the author before their first timing, and every smoke test run as a null comparison.

---

## Contents

1. [The problem](#1-the-problem)
2. [Common protocol](#2-common-protocol)
3. [Step 1: the Taylor series only where it is used](#3-step-1-the-taylor-series-only-where-it-is-used)
4. [Step 2: single-token products as matrix products](#4-step-2-single-token-products-as-matrix-products)
5. [Step 3: fusing the rest](#5-step-3-fusing-the-rest)
6. [Files](#6-files)

---

## 1. The problem

What the decoder is and how a decode step works: [Incremental_Decoder.md](Incremental_Decoder.md).

In a decode step every operation acts on one token's tensors: 16 stiff modes, or vectors of width 384. Each operation then costs its fixed overhead, whatever its arithmetic. In the step of SR2 (pair force removed, layer 1 pruned, `eigh`) at context 512, 58% of the dispatch-level calls come from one function, `damped_mode_coefficients` in [`parf/cfc_baoab.py`](../../notebooks/conservative_arch/parf/cfc_baoab.py). It returns, in float64, the coefficients of the exact flow of each stiff mode over a half step, for every damping regime at once. Its position coefficient $Q = (1 - E_{11})/\omega_0^2$ cancels where $\omega_0^2 t^2$ is tiny, so it also sums a 40-term Taylor series, and `torch.where` keeps the series only where $\lvert \omega_0^2 \rvert t^2 \lt 10^{-4}$. The series is computed for every mode on every call: about 9 operations per term, 360 per call, 4 calls per decode step (two half steps in each of two layers). Another 25% of the calls are `einsum` on one token's tensors, which PyTorch splits into `permute`, `view`, `unsqueeze` and `bmm`.

**What reaches the function** (recorded 2026-11-23 on 8 decode steps at context 512, `t` the half step):

| model | $t$ | $\gamma t$ | modes per call | share of modes in the series branch |
| --- | ---: | ---: | --- | --- |
| SR2-NP, layer 1 pruned to well 5 | 2.0 | 0.2 | 16 at layer 0, 4 at layer 1 | **0** in every call |
| SR2, pair force removed, not pruned | 2.0 | 0.2 | 16 | 0 at layer 0; up to 50% at layer 1, mostly modes of zero stiffness |
| K4-R6, pair force removed, not pruned | 2.0 | 0.2 | 16 | 0 at layer 0; up to 62% at layer 1 |

In the productization configuration the series is never used, and all of it is discarded.

## 2. Common protocol

- **Model and configuration.** SR2-NP, the productization base, with every current inference lever on: query chunk 512, unread registers skipped, layer 1 pruned to well 5 (`prune=1:5`), `eigh` in float64 (`gram=eigh64`). Secondary: the same without pruning, where layer 1 has modes in the series branch. Batch 1, float32, 8 threads, PyTorch 2.2.2, the machine of stage 2.
- **Correctness.** Logits on one OpenWebText validation window of 512 tokens, from the full forward and from the decoder (prefill 448, then 64 single steps), under the variant against the base: the largest absolute difference. A lever that claims bit-identity must give exactly 0.
- **Operator calls.** Dispatch-level calls per decode step at context 512, averaged over 8 steps.
- **Timing: paired, within one process.** Per context length (512, 2,048, 8,192): one prefill, 4 warm-up steps, then blocks of four steps in the order A B B A on the same cache. Within a block, drift in the machine's state and the context's growth by one token per step affect A and B alike. Per block: the ratio of B's summed step time to A's, and the saving $(A - B)/2$ per step. Reported: the medians over blocks with bootstrap 95% intervals (2,000 resamples). 32 blocks per length.
- **Null comparison.** The same harness with A = B = base, run once per session: its ratio interval must contain 1. It calibrates the harness, and it is the only allowed smoke test.
- **Harness.** `debug/cpu_throughput_benchmark.py ... decode=1 ab=A,B loss=<val>` (§6).

## 3. Step 1: the Taylor series only where it is used

### 3.1 The lever

Two inference switches in `cfc_baoab.py`, both off by default, so training is unchanged:
- **`SERIES_ONLY_WHERE_USED`** (variant `masked`). The recurrence runs only on the modes in the series branch, elementwise, and not at all when there are none. Each mode's arithmetic is the same IEEE operations in the same order as before, so the result is bit-identical. It costs one check per call (`small.any()`), a host synchronisation that is free on the CPU and costly on a GPU, hence off for training.
- **`SERIES_TERMS_AUTO`** (variant `auto`). `lowrank_damped_substep` takes the term count from the existing `series_terms_for` at the largest $\gamma t$, instead of the default 40. PMX has used this helper since 2026-10-11 (12 terms at $\gamma t = 0.2$). It matters only where modes are in the series branch.

### 3.2 Function-level check (done, 2026-11-23)

- **`masked` is bit-identical** to the current function on 15 input sets: $t \in \lbrace 0.5, 2, 8 \rbrace$; $\gamma$ scalar 0, 0.1 or 3, or a per-mode tensor; stiffness exactly zero, tiny positive, tiny negative, ordinary and negative, and the two sides of the $10^{-4}$ threshold; 5,265 modes in the series branch. All four outputs are equal (`torch.equal`).
- **`auto`:** at $\gamma t$ = 0, 0.2, 1 and 3.99 (8, 12, 18 and 32 terms), $Q$ equals the 40-term sum exactly on 1,010 modes in the series branch.

### 3.3 The measurements to make

| comparison | configuration | what it shows |
| --- | --- | --- |
| base against base | pruned | the null |
| base against `masked` | pruned | the lever where no mode uses the series |
| base against `masked` | not pruned | the lever where layer 1 has modes in the series branch |
| `masked` against `masked+auto` | not pruned | what the term count adds there |

Correctness for each. The prefill time at 512, 2,048 and 8,192 tokens with and without `masked`, to check that prefill does not slow down.

### 3.4 Not pre-registered

The first smoke test (6 blocks at context 512, pruned) ran base against `masked`. It read 2,914 against 1,478 calls per step, and 28.4 against 20.3 ms per step (ratio 0.72, interval 0.70 to 0.75). The logits check did not run (the validation file was not passed). A saving of 7.9 ms is more than the 4.7 ms the function's own timing predicted (1.43 ms per call, 0.26 ms without the series). With that reading seen, step 1's calls cannot be written blind, so step 1 is reported as a measurement against the protocol of §2, without calls.

### 3.5 Result (2026-11-23)

Run by `results/productization/queue_stage2b_step1.sh`, one job at a time; outputs `decode_ab_s1_*.txt`. A first launch failed at 8,192 tokens in the harness (the position table was extended for 32 decode steps, not the 132 of the paired blocks) and was rerun after the fix; its readings at 512 and 2,048 agreed with the rerun.

**Correctness.** Every variant gives logits identical to the base, maximum difference exactly 0, in the full forward and in the decoder. That holds for `auto` too: on these models its shorter series changes no float64 bit that survives to the logits.

**Calls per decode step (context 512):** pruned 2,914 → 1,478 with `masked`; not pruned 2,915 → 2,211 with `masked`, → 1,715 with `masked+auto`.

**Decode step time,** paired: medians over 32 blocks, bootstrap 95% intervals.

| comparison | context | A, ms | B, ms | B/A | saving, ms |
| --- | ---: | ---: | ---: | --- | --- |
| null, pruned | 512 | 29.24 | 29.61 | 1.024 [0.996, 1.038] | −0.71 [−1.08, 0.02] |
| | 2,048 | 29.27 | 29.35 | 1.004 [0.990, 1.015] | −0.13 [−0.46, 0.30] |
| | 8,192 | 37.13 | 37.17 | 0.994 [0.982, 1.012] | 0.21 [−0.43, 0.76] |
| base → `masked`, pruned | 512 | 27.97 | **19.82** | **0.710** [0.701, 0.720] | 7.98 [7.77, 8.40] |
| | 2,048 | 29.38 | 21.43 | 0.728 [0.719, 0.742] | 7.88 [7.51, 8.32] |
| | 8,192 | 37.55 | 30.33 | 0.809 [0.805, 0.818] | 7.10 [6.76, 7.48] |
| null, not pruned | 512 | 32.63 | 32.40 | 0.997 [0.991, 1.008] | 0.10 [−0.25, 0.29] |
| | 2,048 | 34.26 | 34.15 | 0.996 [0.993, 1.008] | 0.13 [−0.26, 0.26] |
| | 8,192 | 42.62 | 42.93 | 1.000 [0.993, 1.011] | −0.00 [−0.42, 0.29] |
| base → `masked`, not pruned | 512 | 33.13 | 29.70 | 0.889 [0.881, 0.908] | 3.76 [3.03, 3.91] |
| | 2,048 | 34.79 | 31.12 | 0.899 [0.883, 0.909] | 3.55 [3.15, 3.98] |
| | 8,192 | 43.82 | 40.52 | 0.925 [0.906, 0.946] | 3.30 [2.38, 4.12] |
| `masked` → `masked+auto`, not pruned | 512 | 29.58 | 26.76 | 0.901 [0.896, 0.908] | 2.94 [2.67, 3.08] |
| | 2,048 | 31.07 | 28.42 | 0.908 [0.901, 0.919] | 2.84 [2.52, 3.07] |
| | 8,192 | 40.00 | 36.83 | 0.923 [0.914, 0.933] | 3.10 [2.74, 3.44] |

**Prefill,** 3 blocks A B B A per length, ratios B/A: pruned 1.007, 1.010, 0.967 (512); 0.986, 0.982, 0.970 (2,048); 0.982, 0.995, 0.987 (8,192). Not pruned: 0.970, 1.018, 0.976; 0.987, 0.992, 1.004; 0.996, 1.018, 1.006. No slowdown; any gain is within the spread.

**Reading.**
- **The harness holds.** All three null intervals at 2,048 and 8,192, and the not-pruned one at 512, contain 1. The pruned null at 512 reads 1.024 with an interval that only just contains 1: the floor of the paired design is about ±2% at the shortest context.
- **The lever saves a fixed time per step.** In the productization configuration, 7 to 8 ms per step at every context: the step is 29% faster at 512 and 19% at 8,192, where the context-channel read is a larger part of the step. Decoding at context 512 goes from 35.8 to 50.5 tokens per second.
- **More than the function's own timing predicted.** 8.0 ms against 4.7 from timing the function alone. The likely cause is that inside the model the discarded series also allocated about 400 small float64 temporaries per call, a cost the isolated loop paid in a warm allocator. Not tested.
- **Where the series is used, both switches count.** Not pruned, `masked` saves 3.3 to 3.8 ms (the series still runs at layer 1, on fewer modes) and `masked+auto` another 2.8 to 3.1 ms: together about 6.5 ms, close to the pruned saving.
- **For deployment:** both switches on. They are exact on these models and only affect inference.

## 4. Step 2: single-token products as matrix products

### 4.1 What is left

After step 1 (`series=masked+auto`, pruned), a decode step issues 1,487 calls (`decode_bench_decode_attrib_sr2np_lean_step1.txt`). Fifty of them per step are `einsum` on one token's tensors, in four shapes (counted by call site over 8 steps):

| products per step | form | shapes | where |
| ---: | --- | --- | --- |
| 20 | $U^\top h$, a $d$-vector onto $q$ modes | $384 \times 16$, $384 \times 4$ | `lowrank_damped_substep` (3 per call), the layer step |
| 14 | $U z$, $q$ mode amplitudes back to $d$ | $384 \times 16$, $384 \times 4$ | the same |
| 12 | $G (G^\top x)$ and $G \mu$ | $384 \times 32$, $384 \times 4$ | `_lowrank_matvec`, the layer step |
| 4 | the bank's per-well projections, $\sum_d x_{kd} W_{kdr}$ and back | $8 \times 384 \times 4$, $1 \times 384 \times 4$ | `harmonic_terms_lowrank`, `analytical_grad` |

An `einsum` of this kind costs 11 dispatch-level operations (permutes and views around one `bmm`). The same product as an elementwise multiply followed by a sum costs 3, and as `matmul` 8.

### 4.2 The lever

An inference switch (variant `products`), off by default: where the leading dimensions of the operands hold a single token, these 50 products are computed as multiply-and-sum; otherwise, and always when the switch is off, by the same `einsum` as now. Multiply-and-sum builds a $d \times q$ temporary per product, which is small for one token and would be large in prefill, hence the single-token condition. The prefill is therefore unchanged bit for bit. The decode step's results change by float32 rounding, since the sum runs in a different order.

**Design input** (timed alone, one token, 8 threads, median of 3,000 calls): $U^\top h$ at $384 \times 16$ 29.5 µs by `einsum`, 9.6 by multiply-and-sum, 11.8 by `matmul`; $U z$ 25.5, 11.5, 12.2; $384 \times 32$ 27.2, 10.4, 11.7; the bank's $8 \times 384 \times 4$ 54.0, 22.0, 33.1. Over the 50 products that predicts about 0.9 ms per step. Step 1's model-level saving was 1.7 times its function-level prediction.

### 4.3 Measurements

Same protocol as §2: A = step 1 (`masked+auto`), B = step 1 + `products`; pruned (the productization configuration); null comparison A = B = step 1 first; correctness against step 1; prefill check; calls per step.

### 4.4 Calls (proposed 2026-11-23; confirmed by the author the same day, before any timing of the lever; binding)

| id | prediction | called |
| --- | --- | ---: |
| P1 | correctness: the full forward's logits identical to step 1's (difference exactly 0); the decoder's within $10^{-3}$ of step 1's, with the same top-1 token at every one of the 64 positions | 85% |
| P2 | calls per decode step fall by at least 300 (from 1,487) | 90% |
| P3 | speed at context 512: the median ratio B/A at most 0.97 (a saving of 3% or more) | 70% |
| P4 | the saving is fixed per step: at 8,192 tokens it is within ±40% of the saving at 512, in ms | 70% |

**What the result decides.** If P1 holds, the switch joins step 1's in the deployment configuration whatever the speed, unless the step is slower (ratio interval above 1). If P1 fails, the products stay as they are and the reason is reported. Step 3 is designed against whichever configuration results.

## 5. Step 3: fusing the rest

To be designed after step 2. The options: `torch.compile` of the decode step's core, measured in stage 2 at 1.28 to 1.40 times, now on top of steps 1 and 2; or a hand-written single-token step.

## 6. Files

- Switches: [`parf/cfc_baoab.py`](../../notebooks/conservative_arch/parf/cfc_baoab.py), `SERIES_ONLY_WHERE_USED` and `SERIES_TERMS_AUTO` above `damped_mode_coefficients`.
- Harness: [`debug/cpu_throughput_benchmark.py`](../../notebooks/conservative_arch/scaleup/debug/cpu_throughput_benchmark.py), options `decode_attrib=1` (calls by source function) and `ab=A,B` (the paired comparison), with `decode=1`.
- Outputs (to come): `results/productization/decode_ab_*.{txt,json}`; the attribution `decode_bench_decode_attrib_sr2_nv_lean.txt`.
