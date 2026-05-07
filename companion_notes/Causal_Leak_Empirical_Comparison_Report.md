# Causal Leak — Empirical Comparison Report

> **Status.** Drafted **2026-05-02**, immediately after the leak-corrected K = 4 multi-ξ 4000-step pilot completed (`multixi_pilot_fixed`, 05:45 EDT). All four core runs cited below have terminated; their trained weights, training logs, and forensic outputs are on disk and reproducible.
>
> **Companion documents.**
>
> - [`Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md) — the bug, the mechanism, the fix, and the per-run §4.x deep dives that this report aggregates.
> - [`Multi-Channel_vs_Single_Channel_Xi_SPLM_Design.md`](Multi-Channel_vs_Single_Channel_Xi_SPLM_Design.md) — the multi-ξ architecture rationale, now annotated with §0 about how the leak corrupted prior measurements.
> - [`Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md`](Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md) — information-theoretic case for HiPPO/S4 as the foundational generalisation of K-channel ξ.
>
> **Purpose.** Aggregate the four post-bug-discovery runs (one buggy, three leak-free) into a single side-by-side comparison so that conclusions about the SPLM architecture, the multi-channel ξ extension, and the gap to attention can be drawn cleanly *and* the boundary between "real architectural lift" and "leak amplification" is unambiguous.
>
> **TL;DR (the four numbers to remember).**
>
> | run | training integrator | val_ppl @ training mode | val_ppl @ leak-free eval | leak signature |
> |---|---|---:|---:|---|
> | E9 single-ξ, 8000 steps (`scaleup_seed0`) | buggy | 8.85 | 6,843.40 | inflation **777×** |
> | E11 multi-ξ, 2000 steps (`multixi_buggy_2k`) | buggy | 1.05 | 408.12 | inflation **389×** |
> | E9 single-ξ, 4000 steps (`pilot_splm_fixed`) | **fixed** | 33.55 | 33.55 | inflation **1.00×** |
> | E11 multi-ξ, 4000 steps (`multixi_pilot_fixed`) | **fixed** | **14.78** | 14.86 | ratio **0.002×** (inverted) |
> | MatchedGPT, 8000 steps (`seed0_attn`, ref) | n/a | **7.81** | 7.81 | n/a |

---

## 1. Executive summary

Three findings, in decreasing order of strength of evidence:

1. **The pre-fix SPLM PPL claims were dominated by an anti-causal autograd leak**, which inflated apparent perplexity by **389× to 777×** depending on the architecture. Under leak-free evaluation the same trained weights deliver perplexities **two to three orders of magnitude worse** than what was reported. The bug is mechanistically understood, fixed in a single one-line change (`h.detach()` before computing ξ), and verified by a permanent CI-grade probe that confirms forward-causal Δ ≡ 0 to numerical precision in the fixed integrator and Δ ≈ 0.1–0.6 in the buggy one.

2. **Multi-channel ξ is a genuine architectural improvement, not a leak artefact.** Under the fix, K = 4 multi-ξ at 4000 steps achieves **val_ppl = 14.78**, vs. **33.55** for single-ξ at the same step count — a **2.27× honest improvement**. This was not predicted in advance (Prediction B in `Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md` §4.4 estimated 28–32 PPL); the actual lift is approximately twice the predicted magnitude.

3. **MatchedGPT remains roughly 2× lower in PPL than the leak-free K = 4 multi-ξ** (7.81 vs 14.78) at this corpus and budget, despite an asymmetry that *favours* SPLM (4000 vs 8000 training steps). The architecture-vs-architecture gap is now reportable cleanly and is roughly **half** the apparent gap once accounting for the budget asymmetry — but it is still material. The honest separator between SPLM and attention on TinyStories at this scale is approximately a factor of two in PPL, not the four orders of magnitude that the buggy numbers had implied in the *opposite* direction.

The remainder of this report walks through the data behind each finding, organised by metric (trajectories, inflation, causal probe, α_k drift, compute budgets) and then synthesised in §9 against the `paper_v3` restructure plan.

---

## 2. Run inventory

All five tracked runs share corpus (TinyStories, 5 M train tokens), block size 512, batch size 16, and MPS hardware. They differ in integrator, architecture, parameter count, and step budget:

| run id | architecture | integrator | params | steps | batch×block tokens/step | wall time | result dir |
|---|---|---|---:|---:|---:|---:|---|
| `scaleup_seed0` (E9, pre-fix) | SPLM single-ξ (`em_ln`) | buggy | 15.75 M | 8000 | 8,192 | ~14 h | `notebooks/.../scaleup/results/seed0_splm/` |
| **`multixi_buggy_2k`** | SPLM multi-ξ K=4 (`em_ln_multixi`) | **buggy** | 16.54 M | 2000 | 8,192 | 4.86 h | `notebooks/.../scaleup/results/multixi_buggy_2k/` |
| **`pilot_splm_fixed`** | SPLM single-ξ (`em_ln`) | **fixed** | 15.75 M | 4000 | 8,192 | 7.17 h | `notebooks/.../scaleup/results/pilot_splm_fixed/` |
| **`multixi_pilot_fixed`** | SPLM multi-ξ K=4 (`em_ln_multixi`) | **fixed** | 16.54 M | 4000 | 8,192 | 9.25 h | `notebooks/.../scaleup/results/multixi_pilot_fixed/` |
| `seed0_attn` (E9 baseline) | MatchedGPT | n/a | 19.45 M | 8000 | 8,192 | 6.75 h | `notebooks/.../scaleup/results/seed0_attn/` |

**Notes on asymmetries.**

- **Step budgets.** The leak-free SPLM runs were truncated at 4000 steps to fit within the post-bug schedule. MatchedGPT was already trained to 8000 steps before the bug was discovered. To do a strict apples-to-apples PPL comparison the SPLM runs would need to be re-launched at 8000 steps; this is captured as a follow-up P1 item. The 4000-step val PPL trajectory below shows that single-ξ has plateaued by step 2000 (gain of ≤ 2 PPL points expected from doubling budget) while multi-ξ is **still descending** at step 4000 (gain of perhaps 2–4 PPL points expected from doubling budget). The current gap of `14.78 / 7.81 ≈ 1.9×` is therefore likely an *upper bound* on the honest multi-ξ-vs-MatchedGPT gap.
- **Parameter counts.** MatchedGPT has 19.45 M params (extra 2.9 M vs. multi-ξ's 16.54 M). This is the standard "matched-attention" baseline used throughout E9; the slight excess is intentional and was set during the pre-bug protocol so that MatchedGPT is *favoured* by a small margin. We retain that asymmetry here and flag it explicitly.
- **Wall-clock.** SPLM step time is ~6.4 s/step on MPS for single-ξ and ~8.3 s/step for multi-ξ; MatchedGPT is ~3.0 s/step. So MatchedGPT runs roughly 2.7× faster per step and 2× cheaper per training-token at this scale on MPS. This is consistent with the architectural FLOP analysis in `notebooks/conservative_arch/inference_efficiency/` and is unaffected by the leak (the leak is in the autograd graph of the integrator update, not in the per-step compute).

---

## 3. Val PPL trajectories

### 3.1 Buggy multi-ξ (`multixi_buggy_2k`, 2000 steps)

The buggy multi-ξ run collapses to near-perfect val performance by step 1000 — a clear leak-exploit signature, since no model can achieve PPL ≈ 1 on TinyStories without seeing future tokens.

| step | val_loss | val_ppl |
|---:|---:|---:|
|  600 | 0.227 | 1.26 |
|  800 | 0.107 | 1.11 |
| 1000 | 0.069 | 1.07 |
| 1200 | 0.054 | 1.06 |
| 1400 | 0.049 | 1.05 |
| 1600 | 0.045 | 1.05 |
| 1800 | 0.044 | 1.04 |
| 2000 | 0.041 | 1.04 |

(Run halted at step 2000 to preserve compute once the leak signature was unambiguous.)

### 3.2 Fixed multi-ξ (`multixi_pilot_fixed`, 4000 steps)

Smooth, monotone descent — no leak collapse, no plateau yet at the LR floor. The trajectory is the cleanest signature of a healthy SPLM training run we have on record.

| step | val_loss | val_ppl |
|---:|---:|---:|
|  200 | 4.7239 | 112.60 |
|  600 | 3.4062 |  30.15 |
| 1000 | 3.1095 |  22.41 |
| 1400 | 2.9579 |  19.26 |
| 1800 | 2.8722 |  17.68 |
| 2200 | 2.8106 |  16.62 |
| 2600 | 2.7338 |  15.39 |
| 3000 | 2.7259 |  15.27 |
| 3400 | 2.7202 |  15.18 |
| 3800 | 2.7026 |  14.92 |
| **4000** | **2.6934** | **14.78** |

The curve is still descending at step 4000 with `lr → 8.5e-11` (the cosine schedule has fully cooled). Extrapolating modestly, an 8000-step continuation would likely reach the **12–13 PPL** range — closing perhaps half of the remaining gap to MatchedGPT but not closing it fully.

### 3.3 Fixed single-ξ (`pilot_splm_fixed`, 4000 steps)

Plateaus from step 2000 onward at val PPL ≈ 33.55. The plateau is not a learning-rate artefact — even with the LR fully at floor by step 3500 the loss remains within ±0.02 of 3.51.

| step | val_loss | val_ppl |
|---:|---:|---:|
| (smoke 300) | 5.018 | 151.13 |
| 2000 (approx) | ~3.575 | ~35.7 |
| 3000 (approx) | ~3.523 | ~33.9 |
| **4000** | **3.5131** | **33.55** |

(The fixed single-ξ pilot's `train_stdout.log` was not retained on disk; trajectory points above are reconstructed from the deep-dive narrative in `Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md` §4.3 and from the per-step training-loss curve in `splm_em_ln_scaleup_pilot_fixed_loss_curve.png`. The final value is authoritative — it comes from the run summary file.)

### 3.4 MatchedGPT (`seed0_attn`, 8000 steps)

Final val PPL = **7.81** at step 8000. Per-step val measurements were not logged in the JSONL during training (only summary final), so trajectory is not available for plotting; only the endpoint is authoritative.

### 3.5 Side-by-side — final PPL at the natural endpoint of each run

| run | steps trained | tokens trained | final val_ppl | gap to MatchedGPT |
|---|---:|---:|---:|---:|
| `multixi_buggy_2k` | 2000 | 16.4 M | 1.05 (artefact) | +∞ in correct direction; meaningless |
| `pilot_splm_fixed` | 4000 | 32.8 M | 33.55 | **4.30×** |
| `multixi_pilot_fixed` | 4000 | 32.8 M | **14.78** | **1.89×** |
| `seed0_attn` | 8000 | 65.5 M | **7.81** | 1.00× (reference) |

---

## 4. Cross-mode evaluation: the inflation factor as a leak signature

Every leak-corrected ckpt should evaluate to the *same* val PPL whether the buggy or the fixed integrator is used at inference time, because the trained $V_\theta$ never learned to use the leak channel. Conversely, every buggy-trained ckpt should evaluate dramatically worse under the fixed integrator, because severing the leak takes away the channel through which it routed prediction signal.

Inflation factor is defined as `inflation = val_ppl_fixed_eval / val_ppl_buggy_eval`. We measured it on 40,960 val tokens (20 batches × 8 × 256, identical batches under both modes) for every available SPLM ckpt:

| ckpt | trained-mode val_ppl | buggy-eval val_ppl | fixed-eval val_ppl | inflation factor |
|---|---:|---:|---:|---:|
| `scaleup_seed0` (E9, pre-fix, 8000 steps) | 8.85 | 8.85 | **6,843.40** | **777×** |
| `multixi_buggy_2k` (buggy, 2000 steps) | 1.05 | 1.05 | **408.12** | **389×** |
| `pilot_splm_fixed` (fixed, 4000 steps) | 33.55 | 33.55 | 33.55 | 1.00× |
| `multixi_pilot_fixed` (fixed, 4000 steps) | 14.78 | **7,215.34** | 14.86 | **0.002× (inverted)** |

Three regimes are visible:

1. **Buggy-trained ckpts (`scaleup_seed0`, `multixi_buggy_2k`):** inflation ≫ 1 because trained $V_\theta$ has learned to lean on the leak. Multi-ξ has stronger absolute leak fidelity (PPL 1.05 vs 8.85) but a *smaller* inflation ratio (389× vs 777×) because its long-range α = 0.985 channel preserves a fraction of *real* causal structure that survives the fix; the single-ξ has only one channel, which collapsed entirely onto the leak.

2. **Leak-free trained ckpts that retain useful causal structure (`pilot_splm_fixed`):** inflation ≈ 1× — both evaluators see the same V_θ. This is the expected baseline.

3. **Leak-free trained ckpts whose forward step is *destroyed* by the buggy integrator at inference (`multixi_pilot_fixed`):** inflation ≪ 1 ("inverted"). This was *not* predicted in advance, but the mechanism is clean: the buggy integrator mechanically injects information from $h\_{\gt t}$ into the update of $h_t$ via $\partial V/\partial h$ even at inference, regardless of whether $V_\theta$ knows what to do with it. A leak-trained $V_\theta$ uses that injected info; a leak-free $V_\theta$ treats it as destructive noise. The 7,215 PPL "inflation" of a leak-free multi-ξ ckpt under the buggy integrator is therefore a *positive* signature of leak-free training, complementary to the 389× signature of leak-trained training.

The corresponding mechanistic table:

| trained $V_\theta$ knows leak? | injected future-info at inference | result under buggy eval | inflation factor |
|---|---|---|---:|
| Yes (buggy training) | helpful — known channel | val PPL drops dramatically | ≫ 1 |
| No  (leak-free training) | harmful — unknown noise | val PPL rises dramatically | ≪ 1 |
| No  (leak-free training, single-ξ collapse) | neutral — V_θ can't propagate it through the cumulative-mean channel | unchanged | ≈ 1 |

The third row is what `pilot_splm_fixed` shows: the cumulative-mean ξ has no learnable parameters, so the buggy-vs-fixed difference at inference is purely the autograd-derived $f_t$ direction in the integrator update — and on a single-channel cumulative-mean architecture that direction is so impoverished that V_θ-trained-leak-free can't be visibly perturbed by it. Multi-ξ has learnable α_k that interact non-trivially with the integrator update, which is why its leak-free ckpt is sensitive to the eval-mode change.

---

## 5. Causal violation probe results

The probe (`notebooks/conservative_arch/causal_probe.py`) perturbs token at position `t_pert = 40` and measures the maximum absolute change in logits at every other position. For a properly causal model, $\Delta\_{\text{causal}} = \max\_{t \lt t\_{\text{pert}}} |\Delta \text{logits}\_t| \equiv 0$.

| ckpt | evaluator | causal-side Δ | after-side Δ |
|---|---|---:|---:|
| `scaleup_seed0` (buggy) | buggy | 6.20e-01 | 4.61e-01 |
| `scaleup_seed0` (buggy) | fixed | 0.0000     | 4.61e-01 |
| `multixi_buggy_2k`      | buggy | 6.20e-01 | 4.99e-01 |
| `multixi_buggy_2k`      | fixed | 0.0000     | 4.99e-01 |
| `pilot_splm_fixed`      | buggy | 0.0000     | 1.94e-01 |
| `pilot_splm_fixed`      | fixed | 0.0000     | 1.94e-01 |
| `multixi_pilot_fixed`   | buggy | 9.58e-02 | 1.95e-01 |
| `multixi_pilot_fixed`   | fixed | 0.0000     | 2.14e-01 |
| MatchedGPT              | buggy (n/a) | 0.0000  | 1.99e-01 |

**Reading the table.**

- The *fixed-mode* causal-side Δ is the correctness check on the integrator. It is **0.0000 in every leak-free run** (and in MatchedGPT) and **0.0000 even when re-evaluating buggy-trained ckpts**, because the post-fix integrator is causal regardless of how V_θ was trained. This is the per-ckpt verification that the fix took.
- The *buggy-mode* causal-side Δ measures the *integrator's* forward-noncausality, not the trained weights'. It is large (~0.6) for both ckpts that were *trained with the buggy integrator* — because there V_θ has *amplified* the directions in which the leak couples — and small but non-zero (0.096) for `multixi_pilot_fixed`, where the leak-free V_θ provides only the geometric baseline of the integrator's anti-causal autograd path. For `pilot_splm_fixed` (single-ξ leak-free) the buggy-mode Δ is exactly 0, again because the cumulative-mean channel has no non-trivial learnable structure for the leak to act on.

The probe is now a **permanent regression test**: any future SPLM checkpoint must satisfy fixed-mode Δ_causal ≡ 0 to be admissible to a paper or companion-repo release.

---

## 6. α_k drift in the multi-ξ runs

The K = 4 EMA decay rates were initialised at `α = (0, 0.5, 0.9, 0.99)` and made learnable in both buggy and fixed multi-ξ runs. The drift is the cleanest mechanistic signature of whether the optimiser is "harvesting the leak":

| channel | $\alpha\_{\text{init}}$ | buggy final (step 2000) | drift (buggy) | fixed final (step 4000) | drift (fixed) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 (1e-6 floor) | locked | 0.000 (1e-6 floor) | locked |
| 1 | 0.500 | **0.414** | **−0.086** | 0.519 | +0.020 |
| 2 | 0.900 | **0.851** | **−0.049** | 0.855 | −0.045 |
| 3 | 0.990 | 0.985 | −0.005 | 0.979 | −0.011 |

**What this shows.**

- Channel 0 is locked at machine zero in both runs because that is its initialisation and the only structure available to it; the learnable parameter has hit the lower-bound floor. (In the buggy run this is the *highest-fidelity* leak channel, since α = 0 propagates the next-token h directly into the position-t ξ-input.)
- Channel 1 (mid-horizon EMA) shows **strongly opposite drift directions** between the two runs: the buggy run pulls it *down* toward shorter horizons (more future-token weight, harvesting the leak); the fixed run lets it drift slightly *up* toward longer horizons because there is no leak to harvest and the optimiser is just finding the EMA timescale that best matches the corpus statistics.
- Channel 2 happens to drift downward in both runs, but for *different reasons*: in the buggy run it is also harvesting leak; in the fixed run the −0.045 drift simply reflects that an α slightly below 0.9 fits TinyStories better than 0.9 itself. The magnitudes are similar (−0.049 vs −0.045) but the *interpretation* is fundamentally different.
- Channel 3 (long-horizon, $\tau \approx 100$) is approximately stable in both runs, consistent with this being the architecturally protected long-range summary channel.

The K = 4 multi-ξ pre-fix paper narrative ("the optimiser learns task-appropriate timescales") is therefore *partially* salvaged: under the fix, channels 2 and 3 still settle into corpus-appropriate values that look like the predicted "spread of timescales" picture — but channel 1's behaviour is reversed, and channel 0's saturation has a much less interesting interpretation than was originally written.

---

## 7. Compute budgets and parameter efficiency

| run | params | training steps | training tokens | wall time (MPS) | tokens / sec | val_ppl | nats / param-step |
|---|---:|---:|---:|---:|---:|---:|---:|
| `scaleup_seed0` (E9, buggy)   | 15.75 M | 8000 | 65.5 M | ~14 h    | ~1300 | 8.85   | (artefact) |
| `multixi_buggy_2k`            | 16.54 M | 2000 | 16.4 M | 4.86 h   | ~937  | 1.05   | (artefact) |
| `pilot_splm_fixed`            | 15.75 M | 4000 | 32.8 M | 7.17 h   | ~1270 | 33.55  | 0.0531    |
| `multixi_pilot_fixed`         | 16.54 M | 4000 | 32.8 M | 9.25 h   |  ~985 | 14.78  | 0.0407    |
| `seed0_attn` (MatchedGPT)     | 19.45 M | 8000 | 65.5 M | 6.75 h   | ~2700 | 7.81   | 0.0132    |

(`nats / param-step` $= \log(\text{val\\_ppl}) / (\text{params} \cdot \text{steps})$, normalised so smaller is better; this is a crude "efficiency" measure and is included only as a coarse summary, not as a paper-grade metric.)

**Observations.**

- MatchedGPT is roughly **2.7× faster per step** on MPS at this scale, primarily because its forward+backward pass does not include the autograd-derived integrator update (`V.sum().backward()` inside `integrate()`). This is a structural property of SPLM, not an optimisation issue.
- The leak-free multi-ξ uses **4.5% more params** than single-ξ for the K = 4 EMA bank and **+30% wall time per training token** (985 vs 1270 tokens/sec), but achieves **2.27× lower PPL**. The trade-off is favourable.
- All SPLM compute budgets here are *below* what MatchedGPT received (4000 vs 8000 steps; 32.8 M vs 65.5 M tokens). Closing the budget gap is the first item on the P1 re-run list.

---

## 8. The honest SPLM-vs-MatchedGPT separator

This was the single most contested claim in `paper_v3` pre-bug-discovery and is the most important quantitative result in the restructured paper:

| comparison | metric | result | confidence |
|---|---|---|---|
| Buggy single-ξ vs MatchedGPT | val_ppl ratio | 8.85 / 7.81 = 1.13× (SPLM nominally close) | discredited by leak |
| Buggy multi-ξ vs MatchedGPT  | val_ppl ratio | 1.05 / 7.81 = 0.13× (SPLM apparently better!) | discredited by leak |
| **Fixed single-ξ vs MatchedGPT** | val_ppl ratio | 33.55 / 7.81 = **4.30×** | clean, but budget-asymmetric |
| **Fixed multi-ξ vs MatchedGPT**  | val_ppl ratio | 14.78 / 7.81 = **1.89×** | clean, but budget-asymmetric |
| Fixed multi-ξ vs MatchedGPT (predicted at matched budget) | val_ppl ratio | ~12 / 7.81 ≈ **1.5×** | extrapolation, untested |

The honest SPLM-vs-attention separator on TinyStories at 16–20 M params is therefore approximately a factor of two in PPL, with the multi-channel ξ extension. This is a material, non-trivial gap, but it is **roughly the size of the gap from a moderately undertrained transformer to a well-trained one**, not the orders-of-magnitude separation that would justify a "competitive new architecture" claim. The paper's thesis must be re-anchored accordingly (see §9).

The architectural FLOP advantages of SPLM (asymptotic $O(L \cdot d)$ per token vs. attention's $O(T \cdot L \cdot d)$, established in `notebooks/conservative_arch/inference_efficiency/results/RESULTS.md`) are *unaffected* by the leak — they measure forward-pass structure, which depends only on the integrator's shape, not on whether ξ is computed from `h` or `h.detach()`. The "SPLM is asymptotically cheaper at long contexts" claim survives intact and remains the most defensible quantitative SPLM advantage.

---

## 9. Implications for the paper

This section ties the results above to the v4 restructure of the paper.

### 9.1 What survives

- **The Semantic Simulation framework** (the conceptual core of `paper_v3`) is unaffected by the leak. The framework is a theoretical proposal about how to think about language as evolution on a semantic manifold; its validity does not depend on a particular SPLM PPL beating attention.
- **The shared-potential separator methodology** is unaffected. It is a diagnostic that distinguishes architectures based on their adherence to a single scalar potential, and its results on the four available ckpts (single-ξ buggy, single-ξ fixed, multi-ξ fixed, MatchedGPT) can be re-run cleanly under the fix. (P1 in §5 of the restructure plan.)
- **The architectural FLOP asymptotic argument** (SPLM $O(L \cdot d)$ vs attention $O(T \cdot L \cdot d)$) is unaffected. This is now the headline quantitative SPLM advantage.
- **The multi-channel ξ design** is *strengthened* — it earns 2.27× honest PPL improvement over single-ξ. The architectural extension is now empirically motivated, not just theoretically motivated.
- **The HiPPO/S4 generalisation** (`Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md`) is now a natural §A1 architectural appendix: K = 4 EMAs leave $\sim$2 nats of mutual information on the table that orthogonal Legendre projection would recover. This is no longer a speculative future-work item; it is a clear next architectural lever.

### 9.2 What is invalidated

- **Every pre-fix val_ppl claim** in `paper_v3` Phase-3 sections, Appendix §A2.5, and the E9/E10/E11 result tables. These were inflated by 389× to 777× and must be retracted or re-anchored to the leak-free numbers.
- **The "SPLM beats MatchedGPT" or "SPLM is competitive with MatchedGPT" framing.** Under the fix, MatchedGPT is roughly 2× lower in PPL than the strongest SPLM variant we have measured. The honest framing is "MatchedGPT remains the strongest sequence-modelling architecture for next-token PPL on TinyStories at this scale, but SPLM offers a structurally different inductive bias with markedly better asymptotic compute properties at long contexts."
- **The multi-channel ξ "8.7×–28.1× over single-ξ" gain claim** (cited in pre-fix design notes) is mostly leak-amplified. The honest gain is **2.27×**, which is still substantial and worth a §A1 architectural appendix, but not the dominant story.

### 9.3 What this report unlocks

With the four leak-free numbers in hand, the following downstream documents can now be written with confidence:

- A revised `paper_v3` §1 introduction that opens with the architectural thesis ("Semantic Simulation framework, with SPLM as a representative implementation") rather than the empirical PPL comparison.
- A revised `paper_v3` §3 results table that uses the four leak-free PPLs (8.85 → discarded; 33.55, 14.78, 7.81 reported) and acknowledges the budget asymmetry honestly.
- A revised `paper_v3` §4 architectural appendix (now §A1) that introduces K-channel ξ as a 2.27× honest improvement and motivates the HiPPO/S4 generalisation as the natural next step.
- A revised `paper_v3` §A2 forensic appendix that includes the bug, the fix, the inflation table, and the causal probe as a transparency mechanism for readers.
- A pruned public companion repo (`semsimula-paper`) per the plan in `Companion_Repo_Restructure_Plan.md`, with the four leak-free runs as canonical artefacts and the buggy runs preserved in an `archive/` subtree.

### 9.4 What still needs to be measured

The minimal P1 re-run list is now:

1. **Leak-free single-ξ at 8000 steps** — extends `pilot_splm_fixed` to MatchedGPT's training budget. Predicted gain: ≤ 2 PPL points beyond 33.55, since the curve plateaued by step 2000. Cost: ~7 h MPS.
2. **Leak-free multi-ξ at 8000 steps** — extends `multixi_pilot_fixed`. Predicted gain: 2–4 PPL points (to the 11–13 range). Cost: ~9 h MPS.
3. **Leak-free multi-seed (5 seeds)** for the 8000-step multi-ξ run, to put error bars on the "1.5× gap to MatchedGPT" claim. Cost: 5 × ~9 h MPS, parallel-launchable.
4. **Causal probe + inflation eval on the MatchedGPT ckpts** at every step boundary archived in `seed0_attn/`, as a sanity check that no analogous bug exists in the attention path. Cost: ~30 min MPS.

(Items 5+ in §5 — e.g., HiPPO/S4 ablations, longer-context corpora — are P2/P3 and not blockers for the restructured paper draft.)

---

## 10. Provenance and reproducibility

**Source code.**

- Training: `notebooks/conservative_arch/scaleup/train_splm_em_ln_scaleup.py` (single-ξ), `notebooks/conservative_arch/scaleup/train_splm_em_ln_multixi_scaleup.py` (multi-ξ), both with `--causal-force {true,false}` flag.
- Forensics: `notebooks/conservative_arch/causal_probe.py`, `notebooks/conservative_arch/eval_ppl_under_fix.py`, `notebooks/conservative_arch/post_fixed_pilot.py`.
- Models: `notebooks/conservative_arch/scaleup/model_*.py`, all six SPLM classes patched with `if cfg.causal_force: xi_input = h.detach() else: xi_input = h`.

**Per-run artefacts (all on disk, not in version control).**

- `multixi_buggy_2k/{ckpt_latest.pt, training_log.jsonl, summary.md, loss_curve.png}` — 4.86 h on MPS, completed 2026-05-01 morning.
- `pilot_splm_fixed/{ckpt_latest.pt, training_log.jsonl, summary.md, loss_curve.png}` — 7.17 h on MPS, completed 2026-05-01 afternoon.
- `multixi_pilot_fixed/{ckpt_latest.pt, training_log.jsonl, summary.md, loss_curve.png, post_fixed_pilot_report.{md,json}, train_stdout.log}` — 9.25 h on MPS, completed 2026-05-02 05:45 EDT.
- `seed0_attn/{ckpt_latest.pt, training_log.jsonl, summary.md, loss_curve.png}` — 6.75 h on MPS, completed pre-bug-discovery.
- `seed0_splm/{ckpt_latest.pt, ...}` — pre-fix E9 single-ξ ckpt, retained for forensic purposes; *not* a quality reference.

**Git commit at time of report drafting.** Pre-merge of the leak-fix PR; current working tree contains the fix applied to all six model classes plus the new probe/eval scripts. Commit hash will be added when the PR lands.

**Replication instructions.**

```bash
git clone https://github.com/dggueorguiev/semsimula.git
git checkout <leak-fix-commit>
cd notebooks/conservative_arch
python scaleup/train_splm_em_ln_scaleup.py \
    --mode pilot --max-steps 4000 --causal-force true \
    --seed 0 --tag-suffix fixed \
    --results-dir scaleup/results/pilot_splm_fixed_repro
python causal_probe.py --ckpt scaleup/results/pilot_splm_fixed_repro/<ckpt>.pt
python eval_ppl_under_fix.py --ckpt scaleup/results/pilot_splm_fixed_repro/<ckpt>.pt
```

(Multi-ξ replication is identical with `train_splm_em_ln_multixi_scaleup.py`. For exact reproduction of the buggy comparison, pass `--causal-force false`. Wall time on MPS: see §7.)

**Hardware.** All five runs used the same Apple Silicon MPS device. Step times above are MPS-specific; we have no CUDA measurements at this writing.

---

*End of report.*
