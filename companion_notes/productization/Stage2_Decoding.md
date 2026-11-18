# Stage 2: Token-by-Token Decoding

**What this note is.** The results of stage 2 of [Conservative_Core_Productization_Plan.md](Conservative_Core_Productization_Plan.md): an incremental decoder for the Gen 3 conservative core with the pair force disabled (lever N4), checked against the full forward, and its generation speed and cache size against the matched GPT-2 decoding with a key–value cache. It also reports `torch.compile` (lever N2), which stage 1 could not run. Measured 2026-11-17 on an Intel Core i9-9880H with 8 threads, float32, batch 1, PyTorch 2.2.2 under Python 3.11.

**Status.** Measured. The decoder covers the configuration the plan has fixed for productization: no pair force, unread registers skipped.

---

## Contents

1. [How the decoder works](#1-how-the-decoder-works)
2. [The check](#2-the-check)
3. [Decoding speed and cache size](#3-decoding-speed-and-cache-size)
4. [Where a decode step's time goes](#4-where-a-decode-steps-time-goes)
5. [torch.compile](#5-torchcompile)
6. [What this changes in the plan](#6-what-this-changes-in-the-plan)
7. [Files and environment](#7-files-and-environment)

---

## 1. How the decoder works

With the pair force disabled and the unread registers skipped, the only pathway between positions is the context channels $\xi$. At layer $\ell$, position $t$ reads

$$\xi_{k,t} = \sum_{s \le t} \underset{s}{\mathrm{softmax}}\left((t - s) \log a_k + \frac{q_{k,t} \cdot \kappa_{k,s}}{\sqrt{d_\kappa}}\right) h_s$$

over that layer's input states $h_s$. Everything else (the well bank, the stiff-mode decomposition, the integrator, the mass, the output head) acts within one position. So the decoder runs the model's own forward on the new token alone, at its position, and replaces the context-channel module with one that reads a cache of the earlier keys $\kappa_s$ and states $h_s$, one cache per layer. The prefill is an ordinary forward over the prompt that fills the cache on the way.

The cache holds $K d_\kappa + d = 5 \cdot 48 + 384 = 624$ numbers per token and layer, 1,248 per token at two layers, against 6,144 for the matched GPT-2's keys and values (eight layers of $2 d$). The decoder refuses any configuration in which something else crosses positions: the routed pair force, an exchange field, Poisson-mode or particle-store terms, a reverse channel, read registers, or a walk with substeps or a variable number of steps.

## 2. The check

On 256 validation tokens, the first 128 as the prefill and the rest one at a time, against the full forward's logits:

| model | max difference, prefill | max difference, single steps | top-1 agreement |
| --- | ---: | ---: | ---: |
| SR2, pair force removed | 3.4e-5 | 3.8e-5 | 100% |
| SR2, + layer 1 pruned + `eigh` | 3.8e-5 | 3.8e-5 | 100% |
| SR2, + compiled core | 4.0e-5 | 4.6e-5 | 100% |
| K4-R6, the same levers | 4.4e-5 | 5.0e-5 | 100% |
| matched GPT-2, key–value cache | 2.6e-5 | 6.0e-5 | 100% |

The decoder reproduces the full forward to float rounding.

## 3. Decoding speed and cache size

Tokens per second when generating one token at a time after a prefill of 512, 2,048 and 8,192 tokens (median of 30 steps; GPT-2 is the mean of three runs that bracket the ladder runs and agree within 5%):

| configuration | 512 | 2,048 | 8,192 | cache at 8,192 |
| --- | ---: | ---: | ---: | ---: |
| matched GPT-2, key–value cache | 122 | 107 | 68 | 193 MB |
| SR2, pair force removed | 32.5 | 31.2 | 26.2 | 39 MB |
| … + layer 1 pruned + `eigh` | 36.8 | 33.9 | 27.9 | 39 MB |
| … + compiled layer-step core | 47.0 | 47.6 | 35.0 | 39 MB |
| K4-R6, pair force removed, + pruned + `eigh` | 41.5 | 39.8 | 31.2 | 39 MB |
| … + compiled layer-step core | 59.3 | 53.0 | 37.3 | 39 MB |

With all the levers, SR2 decodes 2.6, 2.2 and 1.9 times slower than GPT-2 and K4-R6 2.1, 2.0 and 1.8 times, with a cache a fifth of GPT-2's. The gap narrows with the context: per cached token, GPT-2's eight layers of attention do more work than the ladder's two five-channel reads. The perplexity cost is that of the configurations, measured in stage 1: 59.35 for SR2 and 60.65 for K4-R6, against 58.28 for SR2 as trained.

## 4. Where a decode step's time goes

A profile of eight decode steps at context 512 (SR2, pair force removed, layer 1 pruned, `eigh`):

| | SR2 | matched GPT-2 |
| --- | ---: | ---: |
| operator calls per step | 5,854 | 1,036 |
| step, without the profiler | 27 ms | 8 ms |

The large matrix products (`addmm` and `mm`: the bank's projections and the output head) take about 8.7 ms of a step. The rest is thousands of operations on single-token tensors, among them about 425 dtype conversions per step for the float64 decomposition. A decode step is bound by per-operation overhead, not by arithmetic: removing 75% of layer 1's bank (pruning) speeds decoding by only 1.13 times, while compiling the step's core, which fuses the small operations, speeds it by 1.28 to 1.40 times.

## 5. torch.compile

PyTorch 2.2.2, the last release with wheels for Intel Macs, supports `torch.compile` on Python 3.11 at most. With a Python 3.11 environment and an OpenMP header for Apple's compiler (§7), three targets were tried, each checked against the reference logits:

| target | prefill, 512 / 2,048 / 8,192 tokens | decoding | logits |
| --- | --- | --- | --- |
| the layer step | 0.96 / 1.13 / 1.09 times eager | not measured | as eager |
| the layer step's core (from the linearisation on) | not measured | 1.28 / 1.40 times (512 / 2,048 context) | as eager (§2) |
| the whole model | failed inside PyTorch 2.2's tracer | | |

The prefill comparison ran eager, compiled, eager in one session (eager 1,407 and 1,410 tokens per second at 512 before and after). In prefill the operations are already large, so fusion gains little; in decoding they are tiny, and it gains more. Compiling adds a one-off cost on the first call of each shape (in the 8,192-token prefill, about 50 s).

## 6. What this changes in the plan

- **Decoding is overhead-bound.** The next decode lever is fusing more of the per-token step: the bank's well generation and the context-channel read sit outside the compiled core today, and a hand-written single-token step could remove most of the 5,854 operator calls. Bank compression (the twins PA and PB) mainly shortens the large matrix products, about a third of a step.
- **The cache is already small.** A fifth of GPT-2's at every context, and it grows with the context as a key–value cache does. A fixed decoding state still needs a content-gated form of the channels, which stage 1 found risky at layer 0 (T3).
- **Prefill and decoding trade places.** In prefill the ladder is 2.7 to 4.8 times slower than fused attention (stage 1); in decoding, with the compiled core, 1.8 to 2.6 times.

## 7. Files and environment

- The decoder: [`parf/incremental_decoder.py`](../../notebooks/conservative_arch/parf/incremental_decoder.py) (`LadderDecoder`, and `GPT2Decoder` for the matched GPT-2).
- The benchmark: [`cpu_throughput_benchmark.py`](../../notebooks/conservative_arch/scaleup/debug/cpu_throughput_benchmark.py) with `decode=1`, `decode_tokens=`, `decode_check=` and `decode_profile=1`, and `compile=layer`, `compile=core` or `compile=model`.
- Outputs in [`results/productization/`](../../notebooks/conservative_arch/scaleup/results/productization/): `decode_bench_d_*.txt` and `.json` (§2 to §4), `cpu_bench_c2_*.txt` (§5, prefill).
- **Environment for `torch.compile`.** A Python 3.11 virtual environment with PyTorch 2.2.2, NumPy 1.26 and setuptools. Apple's clang has no OpenMP: set `OMP_PREFIX` to a folder holding LLVM's `include/omp.h` (taken from conda-forge's `llvm-openmp` 15.0.5 package). The benchmark then links the compiled kernels against the `libiomp5` that PyTorch already loads, so one OpenMP runtime is in the process.
