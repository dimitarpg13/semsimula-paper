# Stage 2b: Levers on the Decode Step

**What this note is.** The design, and as they come in the results, of three controlled experiments on the cost of one decoding step of the conservative core. They follow from the attribution in [Stage2_Decoding.md](Stage2_Decoding.md), §4: a decode step issues about 2,900 operator calls (dispatch level; the profiler counts 5,854 with nested calls), against about 1,000 for the matched GPT-2, and the step is bound by per-call overhead. The levers are taken one at a time, each against the step before it. Written 2026-11-23.

**Status.** Step 1 done (2026-11-23, §3.5): bit-identical logits, and the decode step 29% faster at context 512 in the productization configuration (19% at 8,192), prefill unchanged. Step 2 done (§4.5): its calls P1 to P4, confirmed by the author before any timing, all hit; the decode step another 10% faster at 512 (6% at 8,192), the prefill unchanged bit for bit. Step 3 done (§5.4): calls Q1 to Q4, confirmed before any timing, all hit; compiled, the core is fastest with the series switches off (B2), 23% faster than steps 1 and 2 eager at 512 and 14% at 8,192. End to end (§5.5): the decode step 0.516 of the original at 512 and 0.653 at 8,192, prefill 8 to 16% faster. Training: the series costs nothing measurable (§6), so no GPU session; step 4 done (§5.7): reading the cached states once (`gemm`) makes the step 31% faster at 8,192 tokens and leaves it nearly flat in the context; the folded read is slower and misses its correctness bound at 8,192. R1, R3, R4 hit; R2 misses. **Step 1 is not pre-registered:** a harness smoke test meant as the null comparison (base against base) was launched as base against step 1, so a first reading was seen before any call was written (§3.4). Steps 2 and 3 will be pre-registered: calls written and confirmed by the author before their first timing, and every smoke test run as a null comparison.

---

## Contents

1. [The problem](#1-the-problem)
2. [Common protocol](#2-common-protocol)
3. [Step 1: the Taylor series only where it is used](#3-step-1-the-taylor-series-only-where-it-is-used)
4. [Step 2: single-token products as matrix products](#4-step-2-single-token-products-as-matrix-products)
5. [Step 3: fusing the rest](#5-step-3-fusing-the-rest)
6. [Does the series length matter in training?](#6-does-the-series-length-matter-in-training)
7. [Files](#7-files)

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
- **Harness.** `debug/cpu_throughput_benchmark.py ... decode=1 ab=A,B loss=<val>` (§7).

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

### 4.5 Result (2026-11-23)

Run by `results/productization/queue_stage2b_step2.sh`; outputs `decode_ab_s2_null.txt` and `decode_ab_s2_products.txt`. A is step 1 (`masked+auto`), B adds `products`; SR2-NP, layer 1 pruned.

**Correctness.** The full forward's logits are identical to step 1's (difference exactly 0): the prefill path is unchanged. The decoder's differ by at most $4.0 \times 10^{-5}$, the same size as the decoder's own difference from the full forward ($3.8 \times 10^{-5}$), and the top token agrees at all 64 single steps.

**Calls per decode step (context 512):** 1,486 → 1,086.

**Decode step time,** paired: medians over 32 blocks, bootstrap 95% intervals.

| comparison | context | A, ms | B, ms | B/A | saving, ms |
| --- | ---: | ---: | ---: | --- | --- |
| null (step 1 against itself) | 512 | 22.31 | 21.91 | 0.984 [0.968, 0.993] | 0.36 [0.15, 0.74] |
| | 2,048 | 24.67 | 24.00 | 0.983 [0.953, 1.006] | 0.39 [−0.17, 1.17] |
| | 8,192 | 33.09 | 33.50 | 1.010 [0.989, 1.025] | −0.34 [−0.82, 0.26] |
| step 1 → + `products` | 512 | 21.56 | **19.43** | **0.897** [0.882, 0.919] | 2.26 [1.76, 2.56] |
| | 2,048 | 22.96 | 20.99 | 0.919 [0.903, 0.936] | 1.88 [1.46, 2.24] |
| | 8,192 | 32.45 | 30.94 | 0.940 [0.925, 0.966] | 1.95 [1.02, 2.49] |

**Prefill,** 3 blocks, ratios B/A: 1.028, 0.893, 1.052 (512); 0.969, 1.002, 0.940 (2,048); 1.013, 0.988, 1.081 (8,192). The prefill runs the same code with or without the switch (its logits are bit-identical), so these ratios measure the spread of three blocks, not the lever.

**The calls.**
- **P1 (correctness), 85%: HIT.** Full forward identical; decoder within $4.0 \times 10^{-5}$ (threshold $10^{-3}$); top token the same at 64 of 64 steps.
- **P2 (calls fall by at least 300), 90%: HIT.** By 400.
- **P3 (at context 512, median ratio at most 0.97), 70%: HIT.** 0.897.
- **P4 (the saving at 8,192 within ±40% of the saving at 512), 70%: HIT.** 1.95 against 2.26 ms, 86%.

**A failure of the null at 512.** The protocol requires the null's interval to contain 1. At 512 tokens it reads 0.984 [0.968, 0.993], so this time it did not. With step 1's null at 512 (1.024 [0.996, 1.038]), two nulls have now landed about 2% from 1 at the shortest context, in opposite directions. The bootstrap interval over blocks does not capture all the spread between sessions there: at 512 tokens, differences of about 2% are below what this design resolves. P3's threshold of 3%, and the measured 10%, lie above that floor, so the reading stands. The nulls at 2,048 and 8,192 contain 1.

**Reading.**
- **A fixed saving of about 2 ms per step,** 10% at 512 and 6% at 8,192. Decoding at context 512 goes from 46.4 to 51.5 tokens per second in this session.
- **Again more than the isolated timing predicted:** 2.3 ms against 0.9. With step 1 (8.0 against 4.7 ms), the pattern is consistent: in the model, each operation removed saves more than its own time measured alone. The likely reason is the same, fewer temporaries and less allocator traffic, still untested.
- **Sessions differ.** Step 1's configuration took 19.82 ms per step at 512 in step 1's session and 21.56 to 22.31 ms in this one. Absolute times are comparable only within a session; the paired ratios are the measurement. Taken together, steps 1 and 2 give a ratio of about $0.710 \times 0.897 = 0.64$ at 512 and $0.809 \times 0.940 = 0.76$ at 8,192 against the original step, a product of two sessions' ratios. One end-to-end paired run, original against all levers, will measure it directly after step 3.
- **For deployment:** `products` joins step 1's switches.

## 5. Step 3: fusing the rest

### 5.1 The lever and the design question

After steps 1 and 2 a decode step still issues 1,086 calls. The general remedy for many small operations is fusion: `torch.compile` traces the layer step's core (the stiff-mode decomposition, the forces and the integrator, from the linearisation on) and generates kernels that do several operations in one pass. In stage 2, before any lever, compiling the core made decoding 1.28 to 1.40 times faster ([Stage2_Decoding.md](Stage2_Decoding.md), §3). The alternative, a hand-written single-token step, is kept for later, in case compiling gains little.

**One design question comes from step 1.** Its switches add a data-dependent branch (`small.any()`) and boolean indexing, and `auto` reads $\max \gamma$ on the host. The compiler cannot fuse across these: it splits the graph there and runs the pieces separately. Compiled, the original form, the full series discarded by `torch.where`, has no such break, and its 360 elementwise operations might fuse into a few kernels at almost no cost. So step 3 compares two compiled variants:
- **B1** = steps 1 and 2 + compiled core (`masked+auto+products+compiled`);
- **B2** = step 2 alone + compiled core, the series switches off (`products+compiled`).

### 5.2 Measurements

Same protocol as §2, in the Python 3.11 environment that `torch.compile` needs with PyTorch 2.2.2 on this machine (`~/.venvs/py311-torch222`, `OMP_PREFIX` set; [Stage2_Decoding.md](Stage2_Decoding.md), §7). Every arm of a comparison runs in that environment, so the pairing holds. A is steps 1 and 2, eager (`masked+auto+products`). The harness swaps the compiled core in and out per step (`compiled` in a variant name); the compilation itself happens in the warm-up block.
- null: A against A;
- A against B1; A against B2;
- B1 against B2, directly;
- correctness of B1 and B2 against the base; operator calls are not counted for compiled variants (fused kernels).

**Feasibility, checked blind (2026-11-23).** Both compiled variants were run once at 128 tokens with 2 decode blocks, to see that they compile and run; only the exit status and the error lines were read (exit 0, no errors), and the output was deleted unread.

### 5.3 Calls (proposed 2026-11-23; confirmed by the author the same day, before any timing of the lever; binding)

| id | prediction | called |
| --- | --- | ---: |
| Q1 | correctness: for B1 and B2, the logits of the full forward and of the decoder within $10^{-3}$ of the base, with the same top-1 token at all 64 single steps | 80% |
| Q2 | the faster of B1 and B2 has a median ratio against A of at most 0.90 at context 512 | 60% |
| Q3 | B2 is faster than B1: in their direct comparison at context 512, the interval of the ratio B2/B1 lies below 1 | 45% |
| Q4 | for the faster variant, the saving at 8,192 tokens is within ±40% of the saving at 512, in ms | 65% |

**What the result decides.** If Q1 holds and the faster variant's ratio interval lies below 0.95, the compiled core joins the deployment configuration, with its costs stated: a compiler toolchain on the deployment machine and a compilation of about a minute on first use. If the gain is smaller, compiling is not worth those costs for decoding, and a hand-written single-token step is designed as step 3b. Either way, one end-to-end paired run, the original step against the final configuration, closes the series.

### 5.4 Result (2026-11-23)

Run by `results/productization/queue_stage2b_step3.sh` in the Python 3.11 environment; outputs `decode_ab_s3_*.txt`. A is steps 1 and 2, eager (`masked+auto+products`); B1 adds the compiled core; B2 is `products` with the compiled core and the series switches off.

**Correctness.** B1 and B2 give the same logits as each other: the full forward within $4.2 \times 10^{-5}$ of the base, the decoder within $3.4 \times 10^{-5}$, the top token the same at all 64 single steps. The compiled kernels reorder the arithmetic, so the full forward is no longer bit-identical.

**Decode step time,** paired: medians over 32 blocks, bootstrap 95% intervals.

| comparison | context | A, ms | B, ms | B/A | saving, ms |
| --- | ---: | ---: | ---: | --- | --- |
| null (A against A) | 512 | 25.96 | 26.34 | 1.019 [0.980, 1.056] | −0.45 [−1.38, 0.55] |
| | 2,048 | 23.52 | 23.42 | 0.992 [0.941, 1.012] | 0.20 [−0.26, 1.52] |
| | 8,192 | 32.14 | 31.64 | 0.997 [0.981, 1.038] | 0.11 [−1.27, 0.61] |
| A → B1 | 512 | 18.70 | 16.32 | 0.874 [0.862, 0.882] | 2.35 [2.21, 2.57] |
| | 2,048 | 19.84 | 17.61 | 0.877 [0.867, 0.892] | 2.44 [2.14, 2.65] |
| | 8,192 | 29.16 | 26.70 | 0.925 [0.918, 0.929] | 2.19 [2.06, 2.45] |
| A → B2 | 512 | 18.90 | **14.41** | **0.768** [0.753, 0.783] | 4.48 [4.25, 4.79] |
| | 2,048 | 19.72 | 15.34 | 0.773 [0.765, 0.787] | 4.44 [4.17, 4.68] |
| | 8,192 | 29.07 | 25.02 | 0.860 [0.847, 0.863] | 4.07 [3.88, 4.51] |
| B1 → B2 | 512 | 16.32 | 14.90 | 0.907 [0.897, 0.925] | 1.52 [1.20, 1.74] |
| | 2,048 | 17.57 | 16.23 | 0.922 [0.914, 0.930] | 1.36 [1.27, 1.52] |
| | 8,192 | 26.73 | 25.29 | 0.949 [0.939, 0.957] | 1.33 [1.11, 1.68] |

**The calls.**
- **Q1 (correctness of B1 and B2), 80%: HIT.** Within $4.2 \times 10^{-5}$ (threshold $10^{-3}$), top token the same at 64 of 64 steps.
- **Q2 (the faster variant at most 0.90 against A at 512), 60%: HIT.** B2, 0.768.
- **Q3 (B2 faster than B1: the interval of B2/B1 below 1), 45%: HIT.** 0.907 [0.897, 0.925].
- **Q4 (the faster variant's saving at 8,192 within ±40% of its saving at 512), 70%: HIT.** 4.07 against 4.48 ms, 91%.

The three nulls contain 1. The first run of the session (the null) read 26 ms per step at 512 against 18.7 to 18.9 ms for the same configuration in the next two runs, the largest drift between runs seen in this series; the paired ratios are unaffected by it, which is what the design is for.

**Reading.**
- **Compiled, the series switches hurt.** Under compilation the original form, the full series discarded by `torch.where`, fuses into a few kernels and costs almost nothing, while step 1's data-dependent check and boolean indexing split the compiled graph. B2 is 9% faster than B1. So step 1 is the right lever for an eager step and the wrong one for a compiled step. The deployment configuration takes one or the other, not both.
- **The compiled core saves a fixed 4.1 to 4.5 ms per step** over steps 1 and 2 eager: 23% at 512, 14% at 8,192.
- **Taken together** (a product of three sessions' paired ratios, against the original step): $0.710 \times 0.897 \times 0.768 \approx 0.49$ at context 512 and $0.809 \times 0.940 \times 0.860 \approx 0.65$ at 8,192. The end-to-end run (§5.5) measures it directly.
- **Against the matched GPT-2,** across sessions and therefore indicative only: 14.4 ms per step at 512 against GPT-2's 8.2 (1.8 times), 25.0 against 14.7 at 8,192 (1.7 times). In stage 2 the eager step (pruned, `eigh`) was 3.3 and 2.4 times slower than GPT-2, and 2.6 and 1.9 times with its compiled core ([Stage2_Decoding.md](Stage2_Decoding.md), §3).

**What it decides (§5.3).** Q1 holds and B2's interval lies below 0.95, so the compiled core joins the deployment configuration, as B2: `products` on, the series switches off, the core compiled. Its costs: a compiler toolchain on the deployment machine (here, the Python 3.11 environment with LLVM's OpenMP header) and a compilation on first use, not timed here. Without a compiler, the eager configuration stays steps 1 and 2. A hand-written single-token step (step 3b) is not needed for now.

### 5.5 End-to-end check

One paired run of the original step (`base`, eager) against the final configuration (`products+compiled`), with its null, the correctness check and the prefill check, in the Python 3.11 environment. No calls: it measures the combined effect that §5.4 can only estimate as a product. Run by `results/productization/queue_stage2b_e2e.sh`; outputs `decode_ab_e2e_{null,final}.txt`.

**Correctness.** The final configuration's logits are within $4.2 \times 10^{-5}$ of the original's (full forward) and $3.4 \times 10^{-5}$ (decoder), the top token the same at all 64 single steps.

| comparison | context | original, ms | final, ms | ratio | saving, ms |
| --- | ---: | ---: | ---: | --- | --- |
| null | 512 | 26.54 | 26.40 | 1.000 [0.990, 1.004] | 0.01 [−0.10, 0.27] |
| | 2,048 | 29.48 | 29.35 | 0.993 [0.979, 1.013] | 0.19 [−0.39, 0.62] |
| | 8,192 | 39.40 | 39.51 | 1.024 [0.971, 1.043] | −0.92 [−1.62, 1.10] |
| original → final | 512 | 28.92 | **15.12** | **0.516** [0.504, 0.524] | 13.66 [13.44, 14.29] |
| | 2,048 | 28.68 | 15.54 | 0.545 [0.537, 0.557] | 12.93 [12.62, 13.26] |
| | 8,192 | 37.07 | 24.20 | 0.653 [0.647, 0.657] | 12.72 [12.60, 13.28] |

**Prefill,** 3 blocks, ratios final/original: 0.853, 0.832, 0.850 (512); 0.869, 0.863, 0.909 (2,048); 0.935, 0.913, 0.923 (8,192). The compiled core speeds up the prompt too, by 8 to 16%.

**Reading.**
- **Measured directly, the levers halve the decode step at context 512** (0.516) and take a third off at 8,192 (0.653). The product estimate of §5.4 (0.49 and 0.65) was close.
- **A fixed saving of about 13 ms per step.** Decoding at context 512 goes from 34.6 to 66.1 tokens per second in this session.
- **What remains grows with the context:** 15.1 ms at 512, 24.2 at 8,192. Everything in the step except the context-channel read acts within one position, so the 9 ms of growth is the read. The matched GPT-2's step grows by 6.5 ms over the same range (stage 2), on a cache that is five times larger. A folded form of the read (step 4, §5.6) is the candidate lever.
- **Against the matched GPT-2,** across sessions: 15.1 against 8.2 ms per token at 512 (1.8 times slower), 24.2 against 14.7 at 8,192 (1.6 times).

### 5.6 Step 4: the context-channel read

**What grows with the context.** After step 3 the decode step takes 15.1 ms at context 512 and 24.2 ms at 8,192 (§5.5). Only the context-channel read depends on the context, so the 9 ms of growth is the read. At layer $\ell$ it computes, for the new token, one row of weights per channel over the $n$ cached positions and the weighted sum of the cached states.

**Two candidate causes, two levers.**
- **The states are read once per channel.** The decoder writes the weighted sum as the weights, shape $(B, K, 1, n)$, times the states broadcast over the $K = 5$ channels. PyTorch runs that as five row-times-matrix products, each streaming all cached states: at 8,192 tokens, $8{,}192 \times 384$ numbers (12.6 MB) per layer, read five times per step. Written as one $(K \times n)(n \times d)$ product, the same arithmetic reads them once (variant `gemm`). Checked at the operation level on small tensors: no copy is made in either form, and the results agree exactly.
- **The distance logits are rebuilt each step.** The decoder recomputes $(t - s) \log a_k$ for every cached position and then runs the logits, the softmax and the weighted sum as separate operations. Folding the distance term into the cached keys (Stage1b, §3: one extra key component $\lambda_k (s - s_0)$ with $\lambda_k = -\log a_k$, centred at $s_0 = (n_{\max} - 1)/2$, and a query component $\sqrt{d_{\mathrm{k}}}$) gives the full logits from one product, and one `scaled_dot_product_attention` call does the rest (variant `fold`, the author's proposal). Its float32 error grows with the context (Stage1b, §7). By count, the distance row is $K n$ multiplications against $K n d$ for the weighted sum, so the fold alone is expected to save little unless the fused call also reads the states more efficiently.

Both are switches of the decoder (`XI_READ`, `FOLD_CACHE` in `parf/incremental_decoder.py`), off by default. They change the decode step only; the prefill runs the module's own forward.

**Measurements.** Protocol of §2 in the Python 3.11 environment, on top of the final configuration of §5.5 (A = `products+compiled`):
- null: A against A;
- A against A + `gemm`; A against A + `fold`; A + `gemm` against A + `fold`, directly;
- correctness: the logit check of §2, and a new check at each timed length: the same new token decoded under A and under B with the cache rewound in between, so that their logits are compared at contexts of 512, 2,048 and 8,192.

**Calls (proposed 2026-11-23; confirmed by the author the same day, before any timing of the lever; binding).**

| id | prediction | called |
| --- | --- | ---: |
| R1 | `gemm` is exact to float rounding: decoder logits within $10^{-4}$ of A on the check window and at every timed context, top token the same at all 64 steps | 90% |
| R2 | `fold` stays usable: decoder logits within $10^{-3}$ of A on the check window and at every timed context, top token the same at all 64 steps | 70% |
| R3 | `gemm` at 8,192 tokens: median ratio against A at most 0.85 (it removes at least 15% of the step, about 3.6 ms of the 9 ms the read adds) | 55% |
| R4 | `gemm` is faster than `fold` at 8,192 tokens: in their direct comparison the ratio's interval lies away from 1 in `gemm`'s favour | 65% |

**What the result decides.** A variant that passes its correctness call and whose interval at 8,192 lies below 0.95 joins the deployment configuration; if both qualify, the faster. If neither does, the read stays as it is and the remaining gap to GPT-2 at long contexts is recorded as the channels' cost.

### 5.7 Step 4: result (2026-11-23)

Run by `results/productization/queue_stage2b_step4.sh` in the Python 3.11 environment; outputs `decode_ab_s4_*.txt`. A is the final configuration of §5.5 (`products+compiled`).

**Correctness.** On the check window, the decoder's logits against the base: A and `gemm` $3.4 \times 10^{-5}$, `fold` $4.5 \times 10^{-4}$; top token the same at all 64 single steps for all three. The same new token decoded under A and under B at each timed context:

| B | context 512 | 2,048 | 8,192 |
| --- | ---: | ---: | ---: |
| A (null) | 0 | 0 | 0 |
| `gemm` | $1.5 \times 10^{-5}$ | $7.6 \times 10^{-6}$ | $1.5 \times 10^{-5}$ |
| `fold` | $1.1 \times 10^{-4}$ | $3.9 \times 10^{-4}$ | $1.9 \times 10^{-3}$ |

**Decode step time,** paired: medians over 32 blocks, bootstrap 95% intervals.

| comparison | context | A, ms | B, ms | B/A | saving, ms |
| --- | ---: | ---: | ---: | --- | --- |
| null | 512 | 15.31 | 15.17 | 0.998 [0.979, 1.005] | 0.03 [−0.08, 0.37] |
| | 2,048 | 15.92 | 16.06 | 1.002 [0.998, 1.009] | −0.04 [−0.16, 0.03] |
| | 8,192 | 24.44 | 24.27 | 0.995 [0.987, 1.002] | 0.13 [−0.04, 0.31] |
| A → `gemm` | 512 | 15.16 | 15.14 | 0.988 [0.969, 1.013] | 0.18 [−0.19, 0.49] |
| | 2,048 | 16.07 | 15.05 | 0.931 [0.901, 0.977] | 1.09 [0.40, 1.63] |
| | 8,192 | 24.44 | **16.74** | **0.685** [0.679, 0.695] | 7.66 [7.48, 7.84] |
| A → `fold` | 512 | 15.63 | 15.69 | 1.001 [0.997, 1.012] | −0.02 [−0.22, 0.05] |
| | 2,048 | 16.36 | 16.79 | 1.016 [1.007, 1.034] | −0.27 [−0.58, −0.12] |
| | 8,192 | 24.26 | 25.51 | 1.048 [1.042, 1.060] | −1.16 [−1.45, −1.03] |
| `gemm` → `fold` | 512 | 15.43 | 15.54 | 1.009 [1.002, 1.031] | −0.13 [−0.48, −0.03] |
| | 2,048 | 15.53 | 16.94 | 1.097 [1.079, 1.102] | −1.51 [−1.58, −1.25] |
| | 8,192 | 16.90 | 25.82 | 1.526 [1.515, 1.536] | −8.87 [−9.07, −8.67] |

**The calls.**
- **R1 (`gemm` exact to float rounding), 90%: HIT.** At most $1.5 \times 10^{-5}$ at every context (threshold $10^{-4}$); top token the same at 64 of 64 steps.
- **R2 (`fold` within $10^{-3}$ on the check window and at every context), 70%: MISS.** Within the bound on the check window ($4.5 \times 10^{-4}$) and at 512 and 2,048, but $1.9 \times 10^{-3}$ at 8,192: the error grows with the context, as Stage1b's analysis predicted, and centring did not keep it under $10^{-3}$.
- **R3 (`gemm` at 8,192: ratio at most 0.85), 55%: HIT.** 0.685.
- **R4 (`gemm` faster than `fold` at 8,192), 65%: HIT.** `fold` takes 1.53 times as long.

**Reading.**
- **The read's cost was reading the states once per channel.** Written as one product, the read removes 7.7 ms at 8,192 tokens, 85% of the 9 ms the read added over context 512. The step now grows by only 1.6 ms between 512 and 8,192 (15.1 to 16.7 ms), against 6.5 ms for the matched GPT-2.
- **The fold is slower and less accurate.** It removes the cheap part (the distance row) and keeps the expensive one: the fused call is given the states broadcast over the five channels and reads them five times, like the original. It is also 5% slower at 8,192, and its error passes $10^{-3}$ there.
- **Against the matched GPT-2,** across sessions: 15.1 against 8.2 ms per token at 512 (1.8 times slower), 16.7 against 14.7 at 8,192 (1.14 times).
- **Decision (§5.6):** `gemm` passes R1 and its interval at 8,192 lies below 0.95, so it joins the deployment configuration: **`products`, the compiled core and the `gemm` read**. `fold` does not.

## 6. Does the series length matter in training?

The step 1 lever's term count may also save time in training, where the 40-term series runs over every token's modes and autograd keeps its intermediates for the backward pass. A CPU estimate first (the author's stop rule: under about 2% of a training step, no GPU session): one training step of SR2-NP, unpruned, 4 sequences of 512 validation tokens, forward and backward, paired A B B A in 6 blocks; 40 terms against 12 (the count at $\gamma t = 0.2$) and against 0 (the series removed, an upper bound with wrong numbers), with a null; per variant the loss and the bytes autograd saves for the backward. For a GPU, `SERIES_TERMS_FIXED` in `cfc_baoab.py` fixes the count once, so no call reads $\gamma$ on the host. Harness: `train_ab=A,B` (`results/productization/queue_train_series.sh`).
**Result (2026-11-23).** Outputs `results/productization/train_ab_ts_{null,auto,zero}.txt`. One training step takes about 12.5 s on this CPU.

| comparison | step-time ratio B/A | saving, s | saved for the backward, MB | loss |
| --- | --- | --- | --- | --- |
| null, 40 against 40 | 1.000 [0.976, 1.009] | −0.005 [−0.116, 0.303] | 7,518 / 7,518 | 4.077451 / 4.077451 |
| 40 against 12 (the lever) | 0.993 [0.967, 1.046] | 0.096 [−0.585, 0.433] | 7,518 / 7,462 (−0.7%) | 4.077451 / 4.077451 |
| 40 against 0 (series removed) | 1.000 [0.980, 1.022] | 0.001 [−0.268, 0.260] | 7,518 / 7,440 (−1.0%) | 4.077451 / 4.077451 |

**Reading.** Even removing the series altogether leaves the training step unchanged within ±2%, the floor of this design, and saves 1% of the activation memory. In training every operation covers $B \times T \times 16$ modes, so the 360 elementwise operations are a small part of a step dominated by the large products of the bank and the head; their per-call overhead, which dominated the decode step, is spread over 2,048 tokens. The loss is identical to the printed digits under every variant: no mode in this model's training step uses the series in a way that reaches the loss. **Decision, by the rule agreed with the author:** under 2%, no GPU session. `SERIES_TERMS_FIXED` stays in the code, off by default, for a GPU decoder.

## 7. Files

- Switches: [`parf/cfc_baoab.py`](../../notebooks/conservative_arch/parf/cfc_baoab.py), `SERIES_ONLY_WHERE_USED` and `SERIES_TERMS_AUTO` above `damped_mode_coefficients`.
- Harness: [`debug/cpu_throughput_benchmark.py`](../../notebooks/conservative_arch/scaleup/debug/cpu_throughput_benchmark.py), options `decode_attrib=1` (calls by source function) and `ab=A,B` (the paired comparison), with `decode=1`.
- Outputs (to come): `results/productization/decode_ab_*.{txt,json}`; the attribution `decode_bench_decode_attrib_sr2_nv_lean.txt`.
