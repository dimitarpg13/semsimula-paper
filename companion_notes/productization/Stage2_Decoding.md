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

A reader's account, with what a decoder is, its role in text generation, what it requires of a model, and figures, is in [Incremental_Decoder.md](Incremental_Decoder.md).

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

**Which code issues the calls (2026-11-23).** The benchmark's `decode_attrib=1` attributes every operator call of 8 decode steps to the innermost source function that issues it, on the same configuration at context 512 (`decode_bench_decode_attrib_sr2_nv_lean.txt`). It counts at the dispatch level, where composite operators such as `einsum` and `to` are already split into their parts and their nested calls are not counted again, so its total is 2,918 calls per step against the profiler's 5,854. The shares:

| source | calls per step | share | what they are |
| --- | ---: | ---: | --- |
| `cfc_baoab.py: damped_mode_coefficients` | 1,692 | 58% | the exact damped-oscillator coefficients of the stiff modes, 4 calls per step |
| the low-rank substep, the low-rank matrix–vector product, the layer step, the bank's harmonic terms and gradient | 742 | 25% | mostly `permute`, `view`, `unsqueeze` and `bmm`: `einsum` on one token's tensors, split into its parts |
| the context-channel read (`incremental_decoder.py: _xi`) | 94 | 3% | |
| the float64 `eigh` path | 82 | 3% | |
| everything else | 308 | 11% | |

The first row is the root cause. `damped_mode_coefficients` computes, in float64, the coefficients of the exact flow of each stiff mode over the step, for every damping regime at once (`torch.where` evaluates every branch). Its position coefficient $Q = (1 - E_{11})/\omega_0^2$ cancels where $\omega_0^2 t^2$ is tiny, so it also sums a 40-term Taylor series. The series runs on every mode on every call, although only modes with $\omega_0^2 t^2 \lt 10^{-4}$ use it: about 9 operations per term, some 360 per call. In prefill each of these operations covers $T \times 16$ modes and costs little. In decoding each one acts on 16 numbers and costs its full overhead. Timed alone on one token's 16 modes (8 threads), a call takes 1.43 ms, and 0.26 ms without the series: about 5.7 ms of a 27 ms step, 4.7 ms of it the series. A transformer layer, by contrast, is a dozen large operations on full-width vectors.

**Levers this points to (not done).** (1) Replace the series by its first terms in $\omega_0^2$ about the closed form at $\omega_0 = 0$ (relative error of order $(\omega_0^2 t^2)^2 \le 10^{-8}$ in the branch that uses it), or evaluate it only on the modes that need it: about 1,450 calls and 4.5 ms per step. (2) Write the single-token products as plain matrix products instead of `einsum`. (3) Fuse the rest, by compiling or by a hand-written single-token step.

## 5. torch.compile

PyTorch 2.2.2, the last release with wheels for Intel Macs, supports `torch.compile` on Python 3.11 at most. With a Python 3.11 environment and an OpenMP header for Apple's compiler (§7), three targets were tried, each checked against the reference logits:

| target | prefill, 512 / 2,048 / 8,192 tokens | decoding | logits |
| --- | --- | --- | --- |
| the layer step | 0.96 / 1.13 / 1.09 times eager | not measured | as eager |
| the layer step's core (from the linearisation on) | 1.02 / 1.39 / 1.17 times eager (SR2-NP, measured 2026-11-22, below) | 1.28 / 1.40 times (512 / 2,048 context) | as eager (§2) |
| the whole model | failed inside PyTorch 2.2's tracer | | |

The prefill comparison ran eager, compiled, eager in one session (eager 1,407 and 1,410 tokens per second at 512 before and after). In prefill the operations are already large, so fusion gains little; in decoding they are tiny, and it gains more. Compiling adds a one-off cost on the first call of each shape (in the 8,192-token prefill, about 50 s).

**The core in prefill (2026-11-22, item 3 of the plan's remaining speed push).** On SR2-NP with every current lever on (`chunk=512 skipreg=1 prune=1:5 gram=eigh64 xi=fold64 denormal=flush`), one session in the Python 3.11 environment, in the order eager, core compiled, eager, layer step compiled; tokens per second at 512 / 2,048 / 8,192 tokens:

| run | 512 | 2,048 | 8,192 | peak memory at 8,192, GB |
| --- | ---: | ---: | ---: | ---: |
| eager | 1,384 | 1,189 | 1,257 | 9.55 |
| core compiled | 1,390 | 1,763 | 1,461 | 11.10 |
| eager | 1,339 | 1,340 | 1,244 | 9.53 |
| layer step compiled | 1,134 | 1,738 | 1,437 | 11.09 |

- Against the mean of the two eager runs, compiling the core gives 1.02, 1.39 and 1.17 times. The eager runs differ by 13% at 2,048 tokens, so that reading carries the most noise.
- Compiling the whole layer step gains as much at 2,048 and 8,192 tokens but loses 17% at 512.
- The perplexity on 16 windows (58.41) and the logits check against the reference agree with eager to the printed digits in every run.
- The compiled runs hold about 1.5 GB more at 8,192 tokens.
- These eager runs are slower than the Python 3.12 runs of Stage1_CPU_Levers.md, §8 (1,250 against about 1,370 tokens per second at 8,192), so compare within this table only.

So the core is worth compiling for prefill from about 2,048 tokens. At 512 it gains nothing: there the time sits in the large matrix products, which compiling does not change. Outputs: `cpu_bench_l_np_*` and `queue_compile_prefill.log` in `results/productization/`.

## 6. What this changes in the plan

- **Decoding is overhead-bound.** The next decode lever is fusing more of the per-token step: the bank's well generation and the context-channel read sit outside the compiled core today, and a hand-written single-token step could remove most of the 5,854 operator calls. Bank compression (the twins PA and PB) mainly shortens the large matrix products, about a third of a step.
- **The cache is already small.** A fifth of GPT-2's at every context, and it grows with the context as a key–value cache does. A fixed decoding state still needs a content-gated form of the channels, which stage 1 found risky at layer 0 (T3).
- **Prefill and decoding trade places.** In prefill the ladder is 2.7 to 4.8 times slower than fused attention (stage 1); in decoding, with the compiled core, 1.8 to 2.6 times.

## 7. Files and environment

- The decoder: [`parf/incremental_decoder.py`](../../notebooks/conservative_arch/parf/incremental_decoder.py) (`LadderDecoder`, and `GPT2Decoder` for the matched GPT-2).
- The benchmark: [`cpu_throughput_benchmark.py`](../../notebooks/conservative_arch/scaleup/debug/cpu_throughput_benchmark.py) with `decode=1`, `decode_tokens=`, `decode_check=` and `decode_profile=1`, and `compile=layer`, `compile=core` or `compile=model`.
- Outputs in [`results/productization/`](../../notebooks/conservative_arch/scaleup/results/productization/): `decode_bench_d_*.txt` and `.json` (§2 to §4), `cpu_bench_c2_*.txt` (§5, prefill).
- **Environment for `torch.compile`.** A Python 3.11 virtual environment with PyTorch 2.2.2, NumPy 1.26 and setuptools. Apple's clang has no OpenMP: set `OMP_PREFIX` to a folder holding LLVM's `include/omp.h` (taken from conda-forge's `llvm-openmp` 15.0.5 package). The benchmark then links the compiled kernels against the `libiomp5` that PyTorch already loads, so one OpenMP runtime is in the process.
