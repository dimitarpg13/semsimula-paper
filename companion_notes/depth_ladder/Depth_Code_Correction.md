# Correction: Layer 0 Read With Layer 1's Depth Code

**What this note is.** A correction record, 2026-11-25. Five scripts evaluated a layer's force or step outside the model's forward pass and, at layer 0, used the depth code of layer 1. It lists what was affected, the corrected numbers, and what was checked and found right.

**In brief.**
- The forward pass, training, every perplexity, and every reading taken inside a forward pass are **not** affected.
- Nine conservativity runs, their router checks, and four gradient and verification checks were rerun. Layer 1 readings of the conservativity test are unchanged.
- The qualitative error was one: the force of $V_\theta$ at layer 0 was reported as negligible in several models. It is not, in any of them.
- No trained model is reclassified. Of the conservativity verdicts one changes: the total conservative force of G2 passes, where its control said it failed.

---

## Contents

1. [The error](#1-the-error)
2. [What was affected](#2-what-was-affected)
3. [Corrected numbers](#3-corrected-numbers)
4. [Statements withdrawn](#4-statements-withdrawn)
5. [What was audited and found right](#5-what-was-audited-and-found-right)
6. [The rule](#6-the-rule)

---

## 1. The error

One bank of Gaussian wells serves all layers. Layer $\ell$ is told apart by a learned depth code $c_{\ell}$ added to the context channels before the bank reads them, so the centre of well $k$ is

$$
\mu_k = W_k (\xi + c_{\ell}) + b_k ,
$$

and the precisions and weights depend on the same shifted context. The bank holds one active layer at a time. A wrapper around the layer step sets it, and the bank keeps the last value when the step ends.

A script that captures the layer inputs in a forward pass and evaluates the force afterwards therefore finds the bank at the **last** layer. It must select the layer itself. Five scripts did not. At the last layer the reading is right. At layer 0 it is the force of a landscape that the model never applies to those states. It is a well-formed force, so nothing failed: it is the gradient of a potential, and the conservativity test passed.

## 2. What was affected

| script | published where | effect |
| --- | --- | --- |
| `conservativity_test_checkpoint.py` (nine models, router mode included) | results folders, the PM1 note, the SR2-NP and FO-b-PMX-K16-NP notes, the pair-force note, four model cards | layer-0 rows of $V_\theta$ and of the total; shares of the total at layer 0 |
| `gradcheck_vphi_xi_paths.py` | the gradient-starvation note, Tier 0 | layer-0 force of $V_\theta$, and the live gradient through the context channels |
| `verify_vphi_xi_grad_path.py` (and the exchange and no-pair-force checks that use it) | the same note, Tier 1 | layer-0 step gradients |
| `verify_fob_switch.py` | the damping report (pass or fail) | layer-1 convergence numbers; both sides of the comparison used the same code |
| `verify_sr3_switch.py` | the same | none: the rerun reproduces the output exactly |

All five now select the layer. The conservativity test asserts that the force values match the forward pass.

## 3. Corrected numbers

### 3.1 The force of the context potential at layer 0

Median over the test's five tokens, from the conservativity test.

| model | first run | corrected | total force, corrected |
| --- | ---: | ---: | ---: |
| SR2 | 0.090 | 0.071 | 0.072 |
| SR2-NP | 0.121 | 0.074 | 0.074 |
| PM1 | 0.0003 | 0.197 | 0.238 |
| PMX64 | 0.0006 | 0.157 | 0.227 |
| PMX-K16 | 0.000005 | 0.168 | 0.200 |
| FO-b-PMX-K16 | 0.0000002 | 0.089 | 0.139 |
| FO-b-PMX-K16-NP | 0.0000007 | 0.100 | 0.157 |
| F3.1 | 0.010 | 0.160 | 0.161 |
| G2 | 0.0006 | 0.236 | 0.236 |

### 3.2 Conservativity

Worst Jacobian asymmetry and closed-loop work over both layers for the force terms of the step (the reverse channel excepted).

| model | asymmetry | closed-loop work | verdicts |
| --- | ---: | ---: | --- |
| SR2-NP | 2.9e-7 (was 6.7e-7) | 1.6e-6 | all conservative, unchanged |
| FO-b-PMX-K16-NP | 3.3e-7 (was 4.0e-7) | 1.3e-7 (was 2.0e-7) | all conservative, unchanged |
| SR2 | 1.9e-6 | 3.4e-5 for the acting terms | unchanged; the pair force exerts no force at layer 1, so its verdict there stays undefined |
| FO-b-PMX-K16 | 1.8e-5 | 1.7e-4 | unchanged |
| PMX-K16 | 4.5e-6 | 4.7e-5 | unchanged |
| PMX64 | 2.4e-4 | 2.4e-4 (was 3.3e-4) | all conservative, unchanged |
| PM1 | 2.6e-6 | 7.0e-4 | all conservative, unchanged |
| F3.1 (control) | 1.9e-4 | 7.5e-4 | all conservative; the first control was run before the verdict rule was settled |
| G2 (control) | 4.6e-6 | see right | total now conservative (was not); its collapsed pair force and the reverse channel still fail |

G2's total failed because $V_\theta$ was read as 0.0006 and left the pair force's router term dominant. With the right code $V_\theta$ is 400 times larger and the router is under 3% of the total.

### 3.3 The router term

Share of the total conservative force, 95th percentile over tokens.

| model | first run | corrected |
| --- | ---: | ---: |
| SR2, layer 0, position 16 and later | 0.009% | 0.010% |
| FO-b-PMX-K16, the same | 0.437% | 0.288% |
| PMX-K16, the same | 0.054% | 0.026% |
| PMX64, the same | 2.17% | 1.25% |
| PM1, layer 0, positions 3 to 15 | 0.98% | 0.37% |
| F3.1, layer 0, position 16 and later | 2.4% | 0.22% |
| G2, the same | 89% (median 10.6%) | 2.7% (median 0.44%) |
| F3.1, layer 1, positions 3 to 15 | 15% (unaffected) | 15% |

### 3.4 Gradient starvation

| reading | first run | corrected |
| --- | ---: | ---: |
| no-exchange, layer 0, RMS force of the context potential | 0.0000 | 0.0056 |
| no-exchange, layer 0, live gradient through the context channels, force only | 4.3e-5 | 0.56 |
| no-exchange, layer 0, the same through one layer step | 0.023 | 70.8 |
| conservative-only, layer 0, RMS force | 0.0056 | 0.0040 |
| conservative-only, layer 0, live gradient, force only | 0.10 | 0.35 |
| conservative-only, layer 0, the same through one layer step | 20.1 | 98.2 |

The findings that the programme rests on are unchanged: as trained in Gen 2, exactly zero gradient reaches earlier tokens through $V_\varphi$ and through $\xi$; the live construction leaves the forward pass bit-identical.

## 4. Statements withdrawn

- "The force of $V_\theta$ is negligible at layer 0" (FO-b-PMX-K16-NP and the pair-force note). First withdrawn the same day with a wrong reason, a misread row. The output itself was wrong.
- "In no-exchange layer 0 the V_θ force is about 0, so live ξ has almost nothing to act through there", and "live ξ offers 4–20× more source gradient than live V_φ" (the gradient-starvation note). It is 12–25×, and it holds at both layers of no-exchange and at layer 0 of conservative-only. The note's reading that ξ is the more likely cause is strengthened.
- "PM1's step is a gradient flow to within 1% for its worst 5% of tokens": to within 0.4%.
- "F3.1: at layer 0 the router is p95 2.4% of the total": 0.22%.
- "The total conservative force of G2 is not conservative" (control): it is.

## 5. What was audited and found right

- **Taken inside a forward pass, where the wrapper sets the layer:** all notebook cells, the refinement tests and the per-layer flow (FLOW-R), the settling map, the well probes, the pair-force probe, `layer_force_split.py`, `vphi_ste_term.py`, mechanism ablations, the recall probe, the synchronization tests, the well-force and descent readings, `fobpmx_omega_dt.py` and `vtheta_omega_dt.py`.
- **Set the layer themselves:** `geodesic_residual.py`, `d1_zeta_readout.py`, `stage0_models.py`, the SCAF stiffness and well probes, the HF speed benchmark.
- **Evaluate the last layer only:** `verify_pm_wells_exact.py`.
- **The wells of the Poisson modes, the pair force and the reverse channel carry no depth code in $V_\theta$.** Their own rows are unchanged.
- **The template tests** read states and the bank inside the forward pass.
- **Models without depth-conditioned wells** (the 2026-06 conservativity diagnostic and earlier): not affected.

## 6. The rule

A script that calls the bank, a force or a layer step outside the forward pass selects the layer first,

```python
if hasattr(model.V_theta, 'set_active_layer'):
    model.V_theta.set_active_layer(layer)
```

and asserts that its replay equals what the forward pass produced. The test that would have caught this compares the replayed force with the captured one, at every layer. It is in the conservativity test now.
