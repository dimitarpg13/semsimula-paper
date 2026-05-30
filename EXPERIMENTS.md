# Experiment Index

The paper uses short experiment codes (P10g, SQ3, VR2, etc.) throughout.
This file is the master index: look up any code to find what it tests,
where it appears in the paper, which notebook runs it, and where the
result artefacts live.

**Convention.** Most notebooks are multi-arm: a `CELL` variable at the
top selects the experiment.  Set `CELL`, then execute all cells
sequentially.  See [COLAB_NOTEBOOKS.md](COLAB_NOTEBOOKS.md) for Colab
run instructions.

---

## Quick-lookup table

| Code | Paper ref | One-line description | Status |
|------|-----------|---------------------|--------|
| [A1–A10](#structured-v_theta-tinystories-sweep-a1a10-b1b2) | §17.10, Tab 27 | Structured V_theta TinyStories sweep (SQ1–SQ4 + MLP variants) | completed |
| [B1–B2](#structured-v_theta-tinystories-sweep-a1a10-b1b2) | §17.10, Tab 27 | MLP V_theta baselines for structured V_theta comparison | completed |
| [D1–D9](#fockparf-v2-debug-d1d9) | §17c | FockPARF v2 debug cells on TinyStories | completed |
| [D6–D10](#qft-v21-planned-series-d6d10) | §17c.4 | Planned QFT v2.1 incremental tests | planned |
| [E-init / E15](#e-init-mass-validation-e15) | §12.7 | Forward integration of Euler–Lagrange from GPT-2 initial conditions | completed |
| [E1](#multi-seed-validation-and-divergence-diagnostic-e1) | §15.12 | Multi-seed SPLM validation with LN-after-step rescue | completed |
| [E1–E3](#five-negative-scalar-potential-fits-e1e5) | §15.5 | Scalar-potential fits (seven functional forms) | completed |
| [E4](#controlled-damping-sweep-e4) | §15.13, Tab 23 | Controlled-γ damping sweep on SPLM | completed |
| [E5](#five-negative-scalar-potential-fits-e1e5) | §15.5 | Velocity-coupled electromagnetic-analogue gauge sweep | completed |
| [E8](#inference-efficiency-benchmark-e8) | App A2, Tab 38–40 | Pre-registered inference benchmark (quality, wall-clock, Pareto) | completed |
| [E9](#splm-scale-up-on-h100-e9) | §15.14, Tab 24 | SPLM scale-up validation on H100 (TinyStories d=256, L=8) | completed |
| [E10](#-transfer-diagnostic-e10) | §15.14.2, Tab 25 | γ-transfer diagnostic: γ* at scale-up | completed |
| [F1–F6](#f1f6-falsifier-programme) | §9.7 | Six-falsifier programme testing v0 staircase and MCS bounds | partial |
| [F2-baseline / F2-fock-v1 / F2-fock-v2](#dyck-2-falsifier-f2) | §17c.2, Tab 32 | FockPARF v2 Dyck-2 controlled comparison | completed |
| [FR0–FR4](#fockparf-v_theta-regularisation-fr0fr4) | §17b, Tab 26 | FockPARF V_theta regularisation sweep | completed |
| [G1–G3](#planned-confound-resolution-sweeps-g1g3) | §17b | Planned PARF γ / d_V / em_ln confound sweeps | planned |
| [H0 + H1](#h0-h1-schedule-sweep) | §16.3.2, Tab 18 | Hybrid schedule sweep at S=1 | completed |
| [H1.5](#h15-v_theta-narrow-ablation) | §16.3.3, Tab 19 | V_theta-narrow FLOP-arm ablation | completed |
| [H2](#h2-paired-confirmation) | §16.3.4, Tab 20 | Hybrid paired confirmation at n=5 | completed |
| [H6](#h6-substack-restricted-v_-separator) | §16.3.5, Tab 21 | Substack-restricted shared-V_ψ separator test | completed |
| [HR0–HR4](#hybrid-v_theta-regularisation-hr0hr4) | §17b, Tab 26 | Hybrid V_theta regularisation sweep | completed |
| [L1–L3](#architecture-comparison-phase-2-l1l3) | — | TinyStories Phase 2 large-scale architecture comparison | completed |
| [M1–M4](#experiment-a-trajectory-fitting-m1m4) | §18.7, Tab 30 | First-order vs second-order ODE trajectory fitting on GPT-2 | completed |
| [multixi_buggy_2k / multixi_pilot_fixed](#multi-leak-fix-forensics) | §15.14.3, Tab 26 | Multi-ξ K-channel SPLM leak-fix forensic runs | completed |
| [N5](#scale-up-n5-confirmation) | §16.5.2, Tab 22 | Five-seed paired confirmation at scale-up | completed |
| [OQ-1](#oq-1-structural-vs-mlp-ablation) | §17.8.2 | Pre-registered structural-vs-MLP V_phi ablation | completed |
| [P0](#p0-smoke-verification) | §17.8.1 | PARF smoke verification and causal-violation probe | completed |
| [P1](#p1-structural-v_phi-at-seed-0) | §17.8.2, Tab 17 | PARF structural V_phi at seed 0 | completed |
| [P1.5 / P1.5a](#p15-p15a-oq-1-ablation) | §17.8.2, Fig 18 | OQ-1 ablation: structural vs unstructured MLP V_phi | completed |
| [P1.6](#p16-capacity-disambiguation) | §17.8.3, Fig 19 | Wider structural V_phi capacity disambiguation | completed |
| [P5](#p5-gumbel-sparsity-at-top-k4) | §17.8.4 | Stage 1.5 Gumbel-softmax sparsity at top-k=4 | completed |
| [P6](#p6-channel-level-diagnostic) | §17 | Channel-level gate-selectivity diagnostic | completed |
| [P7 / P8](#p7-p8-composite-patches) | §17 | Force-balance and saturation-fix patches | completed |
| [P10d–P10j](#p10-ablation-ladder-p10dp10j) | §17.12, Tab 28 | P10 ablation ladder on TinyStories | completed |
| [P1–P5 (Fock)](#fockparf-improvement-sweep-p1p5-fock-g1g3) | — | FockPARF improvement sweep closing ~40 PPL gap | completed |
| [PR0–PR4](#parflm-v_theta-regularisation-pr0pr4) | §17b, Tab 26, Fig 21–22 | PARFLM V_theta regularisation sweep | completed |
| [Q0–Q8](#qft-v21-experimental-ladder-q0q8) | §17c.5, Tab 33 | QFT v2.1 incremental creation-gate improvements | completed |
| [Q9c / Q9d](#q9c-q9d-architecture-definitions) | §16.1, §17.9 | PARF-augmented SPLM (Q9c) and Helmholtz hybrid (Q9d) | completed |
| [R6.*](#r6-information-bottleneck-ladder) | §15.3, Tab 14 | R6 information-bottleneck ladder | completed |
| [S1–S4](#architecture-comparison-s1s4) | — | TinyStories v3: PARF vs FockPARF vs Hybrids with SQ3 V_theta | completed |
| [Scope 3](#scope-3-leak-free-multi-seed-retrain) | §15.11, Tab 15–17 | Leak-free S=5 retrain of v2 SPLM experiments | completed |
| [SQ1–SQ5](#structured-v_theta-sweep-sq1sq5) | §17.10, Tab 27 | Structured V_theta expressivity test on TinyShakespeare | completed |
| [MXP-H16](#multi-xi-parf-h16-pilot) | §17 | Multi-ξ PARF pilot at H=16 (pre-memory-fix) — 3 arms | completed |
| [MXP-H128](#multi-xi-parf-h128-scaleup) | §17 | Multi-ξ PARF at H=128 (Level-2 ckpt + gathered V_φ) — 6 arms | completed |
| [FMXP-H128](#fock-multi-xi-parf-h128-scaleup) | §17 | Fock Multi-ξ PARF at H=128 (v1/v2 gates, register sweep) — 13 arms | in progress |
| [Stage-1.5a](#stage-15a-stage-15b-v_phi-memory-variants) | §17.9, Tab 29 | Dense V_phi forward with post-masking | completed |
| [Stage-1.5b](#stage-15a-stage-15b-v_phi-memory-variants) | §17.9 | Gathered V_phi (top-k source gathering) | design only |
| [VR0–VR5](#splm-v_theta-regularisation-vr0vr5) | §17b, Tab 26 | Standalone SPLM V_theta regularisation sweep | completed |
| [e0–e5 (SP-HSPLM Stage 1)](#sp-hsplm-stage-1-e0e5) | — | Per-token non-conservative force experiment (Class B/C) | completed |
| [q9e_a–q9e_n (SP-HSPLM Stage 2)](#sp-hsplm-stage-2-q9e_aq9e_n) | — | Q9(e) pair-skew cell ladder | completed |
| [Pilot Arms 1–5b](#scale-up-pilot-arms-15b) | §16.5.1, Tab 22 | Five-arm SPLM-family scale-up comparison | completed |
| [H1–H3 (hypotheses)](#predictions-and-hypotheses) | §13, §18 | Jacobi / overdamped dynamics hypotheses | hypothesis |
| [M1–M5 (mass axioms)](#mass-axioms-m1m5) | §12.1 | Five axioms any mass candidate must satisfy | axiom |
| [P1–P4 (predictions)](#predictions-and-hypotheses) | §11 | Hidden-state testable predictions (geodesic, clustering, wells) | prediction |
| [Q1–Q9 (open questions)](#open-questions-q1q9-19) | §19 | Research-agenda open questions | open question |

---

## Experiment A — Trajectory fitting (§18)

### Experiment A: trajectory fitting (M1–M4)

Tests whether GPT-2 hidden-state trajectories follow first-order or
second-order autonomous ODE dynamics.

| Code | Description | Best test R² (K=8) |
|------|-------------|-------------------|
| **M1** | First-order physics: gradient descent on learned V_ψ | −0.47 |
| **M2** | Second-order physics: damped Velocity-Verlet on V_ψ | −0.43 |
| **M3** | General lag-1 MLP (no physics structure) | −0.93 |
| **M4** | General lag-2 MLP | −0.81 |

- **Paper:** §18.7 (Experiment A), Table 30, Figure 30–31
- **Per-layer extension:** §18.8, Table 31, Figure 32–33
- **Notebook:** [`notebooks/dynamics_order_test/scripts/experiment_a_trajectory_fitting.ipynb`](notebooks/dynamics_order_test/scripts/experiment_a_trajectory_fitting.ipynb)
- **Per-layer notebook:** [`notebooks/dynamics_order_test/scripts/experiment_a_per_layer_sweep.ipynb`](notebooks/dynamics_order_test/scripts/experiment_a_per_layer_sweep.ipynb)
- **Results:** [`notebooks/dynamics_order_test/results/experiment_a/`](notebooks/dynamics_order_test/results/experiment_a/), [`notebooks/dynamics_order_test/results/experiment_a_per_layer/`](notebooks/dynamics_order_test/results/experiment_a_per_layer/)
- **Key finding:** All four models have negative test R² (worse than predicting the mean). Physics-structured models (M1, M2) degrade more gracefully at long rollout horizons. M2 learns γ=2.40 (velocity retention 29%), independently confirming the overdamped regime. Per-layer sweep shows M2 fails catastrophically at every layer (R²=−35 at layer 3), confirming the non-autonomous regime.

---

## E-init mass validation (§12)

### E-init: mass validation (E15)

Forward integration of the Euler–Lagrange equation from GPT-2
first-block initial conditions. Fits per-layer Gaussian-well potentials
and tests symplectic Euler integration with γ calibration.

- **Paper:** §12.7 (The E-init experimental protocol)
- **Notebook:** [`notebooks/e_init/e_init_validation.ipynb`](notebooks/e_init/e_init_validation.ipynb) (local only)
- **Results:** [`notebooks/e_init/results/`](notebooks/e_init/results/)
- **Key finding:** Registered as experiment E15 of the mass-validation programme (E1–E14). Tests whether fitted well parameters suffice to predict the full trajectory from first-layer boundary data.

### Mass axioms (M1–M5)

Five axioms any mass candidate must satisfy. These are validation
targets, not experiment runs.

| Code | Axiom |
|------|-------|
| **M1** | Inertia: mass resists change |
| **M2** | Gravitational centering: higher-mass entities pull centroid |
| **M3** | Well depth: V_∞ = m·v² = E_t |
| **M4** | Bound-state proximity: heavier properties closer to centroid |
| **M5** | Information × valence decomposition |

- **Paper:** §12.1

---

## Conservative SPLM programme (§15)

### Five negative scalar-potential fits (E1–E5)

| Code | Description |
|------|-------------|
| **E1–E3** | Scalar-potential sweep across seven functional forms |
| **E4** | Linear Helmholtz augmentation: skew-symmetric solenoidal term |
| **E5** | Velocity-coupled electromagnetic-analogue gauge sweep |

- **Paper:** §15.5
- **Results:** [`notebooks/conservative_arch/results/`](notebooks/conservative_arch/results/)
- **Key finding:** Five prior fitting experiments all tie the static null on held-out data. The rotational/solenoidal component exists in principle but the fit is dominated by per-layer potential variation.

### R6 information-bottleneck ladder

Multi-ξ K-channel SPLM variants tested in a progressive ladder.

| Code | Description |
|------|-------------|
| **R6.h.0**, **R6.h.1** | Baseline and first rung |
| **R6.a**, **R6.e**, **R6.i** | Completed ladder rungs |
| **R6.b**, **R6.c**, **R6.f**, **R6.h.2**, **R6.h.3** | Deferred rungs |

- **Paper:** §15.3, Table 14
- **Results:** [`notebooks/conservative_arch/results/`](notebooks/conservative_arch/results/)

### Multi-seed validation and divergence diagnostic (E1)

Multi-seed SPLM validation with the LN-after-step rescue.

- **Paper:** §15.12, Table 23, Figure 25–26
- **Results:** [`notebooks/conservative_arch/ln_damping_sweep/results/leakfree_3seed/`](notebooks/conservative_arch/ln_damping_sweep/results/leakfree_3seed/)
- **Key finding:** LN-after-step is a necessary rescue for training stability; without it, most seeds diverge or NaN.

### Controlled-γ damping sweep (E4)

- **Paper:** §15.13, Table 23, Figure 24
- **Notebook:** included in the `ln_damping_sweep` scripts
- **Results:** [`notebooks/conservative_arch/ln_damping_sweep/results/`](notebooks/conservative_arch/ln_damping_sweep/results/) (gamma subdirs + `leakfree_5seed_confirmation/`)
- **Key finding:** U-shaped PPL-vs-γ curve; overdamped basin at γ∈[0.10, 0.15]; five-seed confirmation locks optimal γ.

### Scope 3: leak-free multi-seed retrain

Re-runs v2 SPLM-family TinyShakespeare experiments under the v4
leak-free integrator.

| Cell | Description |
|------|-------------|
| **splm_baseline** | SPLM baseline (5 seeds) |
| **splm_sarf** | SARF variant |
| **splm_sarfmass_embed_head** | SARF-mass embed head |
| **splm_sarfmass_logfreq** | SARF-mass log-frequency |
| **matched_baseline** | Matched-attention baseline |

- **Paper:** §15.11, Tables 15–17, Figures 27–28
- **Notebook:** [`notebooks/conservative_arch/multi_seed/colab_scope3.ipynb`](notebooks/conservative_arch/multi_seed/colab_scope3.ipynb)
- **Results:** [`notebooks/conservative_arch/multi_seed/results/scope3_shakespeare/`](notebooks/conservative_arch/multi_seed/results/scope3_shakespeare/)
- **Key finding:** Under leak-free training the SARF xi-recomputation fix reduces the SARF advantage by 1.34×. The per-token mass head still provides a statistically significant benefit.

### SPLM scale-up on H100 (E9)

- **Paper:** §15.14, Table 24
- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_pilot.ipynb`](notebooks/conservative_arch/scaleup/colab_pilot.ipynb), [`notebooks/conservative_arch/scaleup/colab_gamma_sweep.ipynb`](notebooks/conservative_arch/scaleup/colab_gamma_sweep.ipynb)
- **Results:** [`notebooks/conservative_arch/scaleup/results/pilot/`](notebooks/conservative_arch/scaleup/results/pilot/), [`notebooks/conservative_arch/scaleup/results/gamma_sweep/`](notebooks/conservative_arch/scaleup/results/gamma_sweep/)
- **Key finding:** Matched-attention beats SPLM em_ln by ~18–31 PPL at the prototype scale (Phase 1 grade Q3).

### γ-transfer diagnostic (E10)

Tests whether small-scale optimal γ (0.166 from E5) holds at scale-up.

- **Paper:** §15.14.2, Table 25
- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_gamma_sweep.ipynb`](notebooks/conservative_arch/scaleup/colab_gamma_sweep.ipynb)
- **Results:** [`notebooks/conservative_arch/scaleup/results/gamma_sweep/`](notebooks/conservative_arch/scaleup/results/gamma_sweep/)
- **Key finding:** γ*=0.30 at scale-up vs 0.166 at small scale; the transfer is imperfect but the U-shape is preserved.

### Multi-ξ leak-fix forensics

| Code | Description |
|------|-------------|
| **multixi_buggy_2k** | Pre-fix multi-ξ run (leaked integrator) |
| **multixi_pilot_fixed** | Post-fix multi-ξ run (leak-free) |
| **multihippo**, **multis4d** | Planned multi-ξ sweep IDs |

- **Paper:** §15.14.3, Table 26
- **Results:** [`notebooks/conservative_arch/scaleup/results/multixi_buggy_2k/`](notebooks/conservative_arch/scaleup/results/multixi_buggy_2k/), [`notebooks/conservative_arch/scaleup/results/multixi_pilot_fixed/`](notebooks/conservative_arch/scaleup/results/multixi_pilot_fixed/)

---

## Hybrid SPLM (§16)

### H0 + H1 schedule sweep

Tests six hybrid SPLM+attention schedules at S=1.

- **Paper:** §16.3.2, Table 18, Figure 17
- **Results:** [`notebooks/conservative_arch/hybrid/results/h1_sweep/`](notebooks/conservative_arch/hybrid/results/h1_sweep/)
- **Key finding:** Best hybrid (Variant A, k=4, m=4) reaches 133.01 PPL — a 15 PPL improvement over matched attention with identical parameter count.

### H1.5: V_theta-narrow ablation

- **Paper:** §16.3.3, Table 19, Figure 18
- **Key finding:** V_theta-narrow prevents any cell from clearing the FLOP bar at long context; confirms the need for V_theta capacity.

### H2: paired confirmation

- **Paper:** §16.3.4, Table 20
- **Results:** [`notebooks/conservative_arch/hybrid/results/h2_paired_confirmation/`](notebooks/conservative_arch/hybrid/results/h2_paired_confirmation/)
- **Key finding:** Hybrid advantage over matched attention confirmed at n=5 seeds.

### H6: substack-restricted V_ψ separator

- **Paper:** §16.3.5, Table 21
- **Key finding:** The substack-restricted shared-V_ψ separator test is empirically confirmed for the Helmholtz hybrid.

### Q9c / Q9d architecture definitions

| Code | Description |
|------|-------------|
| **Q9c** | PARF-augmented SPLM |
| **Q9d** | Helmholtz hybrid (layer-type decomposition) with AAAASSSS schedule |

- **Paper:** §16.1, §17.9
- **Notebook (Q9d scale-up):** [`notebooks/conservative_arch/scaleup/colab_pilot.ipynb`](notebooks/conservative_arch/scaleup/colab_pilot.ipynb)

### Scale-up pilot (Arms 1–5b)

Five-arm SPLM-family comparison at the E9 scale-up configuration.

| Arm | Description |
|-----|-------------|
| **Arm 1** | Matched-attention baseline |
| **Arm 2** | SPLM em_ln (fixed γ=0.30) |
| **Arm 2b** | TF32-on sub-arm |
| **Arm 3** | Helmholtz Q9d |
| **Arm 4** | Hybrid VA (k=4, m=4) |
| **Arm 5 / 5b** | PARF Q9c sparse (k=4); H100 scale-up |

- **Paper:** §16.5.1, Table 22
- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_pilot.ipynb`](notebooks/conservative_arch/scaleup/colab_pilot.ipynb)
- **Results:** [`notebooks/conservative_arch/scaleup/results/pilot/`](notebooks/conservative_arch/scaleup/results/pilot/)

### Scale-up N5 confirmation

Five-seed paired confirmation of pilot results with 95% CIs.

- **Paper:** §16.5.2, Table 22
- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_n5_confirmation.ipynb`](notebooks/conservative_arch/scaleup/colab_n5_confirmation.ipynb)
- **Results:** [`notebooks/conservative_arch/scaleup/results/n5_confirmation/`](notebooks/conservative_arch/scaleup/results/n5_confirmation/)

---

## PARF empirical programme (§17)

### P0: smoke verification

- **Paper:** §17.8.1
- **Key finding:** PARF passes the causal-violation probe; no signal leakage from future tokens.

### P1: structural V_phi at seed 0

- **Paper:** §17.8.2, Table 17
- **Key finding:** Structural V_phi achieves 210.5 PPL with ~4,000 V_phi params.

### OQ-1: structural vs MLP ablation

Pre-registered ablation testing whether the structural prior on V_phi
matters.

- **Paper:** §17.8.2 (Theorem OQ-1)

### P1.5 / P1.5a: OQ-1 ablation

| Code | V_phi type | Val PPL |
|------|-----------|---------|
| **P1** | Structural | 210.5 |
| **P1.5a** | Unstructured MLP | 297.2 |

- **Paper:** §17.8.2, Figure 18
- **Key finding:** Structural V_phi outperforms unstructured MLP by 87 PPL — OQ-1 verdict confirms the structural prior.

### P1.6: capacity disambiguation

Wider structural V_phi to disentangle capacity from structural prior.

- **Paper:** §17.8.3, Figure 19
- **Key finding:** PPL gain comes from the structural prior, not from V_phi width.

### P5: Gumbel sparsity at top-k=4

Stage 1.5 Gumbel-softmax sparsity.

- **Paper:** §17.8.4
- **Key finding:** Large PPL gain vs P1.6; k∈{4,8,16} ordering preserved at TinyStories scale.

### P6: channel-level diagnostic

- **Paper:** §17 (paired with P10i/P10j)
- **Key finding:** P10j attains better gate selectivity than P10i yet worse PPL — the v0 ceiling binds.

### P7 / P8: composite patches

Four P8 patches: LN-before-distance, per-layer V_phi scale, softsign
theta, bilinear theta.

- **Paper:** §17 (referenced as components of P10d baseline)
- **Notebook:** [`notebooks/conservative_arch/parf/scripts/p8_cell_a100_h100.ipynb`](notebooks/conservative_arch/parf/scripts/p8_cell_a100_h100.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/structural/`](notebooks/conservative_arch/parf/results/structural/)
- **Key finding:** Fixes P6 Layer-1 force imbalance and theta saturation.

### P10 ablation ladder (P10d–P10j)

Eight-cell ablation ladder on TinyStories, progressively adding
structural improvements.

| Code | Change from previous | k | Steps | Corpus | Best PPL |
|------|---------------------|---|-------|--------|----------|
| **P10d** | Full P5+P7+P8 baseline | 4 | 8,000 | 5M | 28.67 |
| **P10f** | V_theta: 1024→2048 | 4 | 8,000 | 5M | 28.50 |
| **P10g** | + 16k steps | 4 | 16,000 | 5M | **26.42** |
| **P10h** | + 20M tokens (4×) | 4 | 16,000 | 20M | 26.43 |
| **P10i** | top-k Gumbel routing | 8 | 16,000 | 5M | 27.10 |
| **P10j** | top-k Gumbel routing | 16 | 16,000 | 5M | 27.73 |

- **Paper:** §17.12, Table 28
- **Notebook:** [`notebooks/conservative_arch/parf/scripts/p10_tinystories_a100_h100.ipynb`](notebooks/conservative_arch/parf/scripts/p10_tinystories_a100_h100.ipynb) (P10a–P10h), [`notebooks/conservative_arch/parf/scripts/p10_sparsity_ladder_k8_k16.ipynb`](notebooks/conservative_arch/parf/scripts/p10_sparsity_ladder_k8_k16.ipynb) (P10i–P10j)
- **Results:** [`notebooks/conservative_arch/scaleup/results/semsimula_parflm/p10_tinystories/`](notebooks/conservative_arch/scaleup/results/semsimula_parflm/p10_tinystories/) (P10e–P10g), [`notebooks/conservative_arch/scaleup/results/parflm/p10_tinystories/`](notebooks/conservative_arch/scaleup/results/parflm/p10_tinystories/) (P10h–P10j)
- **Key finding:** P10h null result (zero improvement from 4× corpus) is the empirical signature of the v0 expressivity ceiling. Monotone degradation in k confirms k=4 as optimal. The ~26.4 PPL ceiling is confirmed across both corpus and sparsity axes.

### Stage-1.5a / Stage-1.5b: V_phi memory variants

| Code | Description |
|------|-------------|
| **Stage-1.5a** | Dense V_phi evaluation with post-masking (current implementation) |
| **Stage-1.5b** | Gathered V_phi (evaluating only top-k pairs) — proposed |

- **Paper:** §17.9, Table 29
- **Design doc:** [`companion_notes/PARF_Stage_1_5b_design.md`](companion_notes/PARF_Stage_1_5b_design.md)
- **POC notebook:** [`notebooks/conservative_arch/parf/scripts/gradient_checkpoint_gathered_vphi_poc.ipynb`](notebooks/conservative_arch/parf/scripts/gradient_checkpoint_gathered_vphi_poc.ipynb)
- **Key finding:** Stage-1.5b reduces V_phi intermediates from O(T²) to O(T·k); combined with Level 2 checkpointing, peak V_phi activation memory drops from O(L·B·T²·H) to O(B·T·k·H).

### Multi-Xi PARF H=16 pilot

Three-arm pilot combining multi-channel K-EMA ξ (K=4) with sparse PARF
pair forces (MultiXiPARFLM) at the memory-constrained setting (H=16,
grad-accum=2).

| Arm | V_φ kind | k | PPL |
|-----|----------|---|-----|
| **competitive_k8** | structural_competitive | 8 | 15.44 |
| **competitive_k4** | structural_competitive | 4 | 16.03 |
| **structural_k8** | structural | 8 | — |

- **Paper:** §17 (multi-ξ results subsection)
- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_parf_multixi.ipynb`](notebooks/conservative_arch/scaleup/colab_parf_multixi.ipynb)
- **Results:** GDrive `semsimula_parf_multixi/`
- **Key finding:** Multi-ξ dramatically improves PARF (15.44 vs 28 PPL with single ξ) but V_φ at H=16 adds ~0.75 PPL overhead vs multi-ξ SPLM alone (14.69). V_φ capacity is the binding constraint.

### Multi-Xi PARF H=128 scaleup

Re-runs multi-ξ PARF at full V_φ capacity (H=128) enabled by Level-2
per-layer checkpointing + Stage-1.5b gathered V_φ. Six arms sweep
channel count (K=2, 4, 8), α-init strategy (hand-picked, log-spaced),
routing density (k=4, 8), and V_φ kind (competitive, structural).

| Arm | K | α-init | V_φ | k | PPL |
|-----|---|--------|-----|---|-----|
| **comp_K4_best_alpha** | 4 | [0.25,0.50,0.75,0.95] | competitive | 8 | 13.19 |
| **comp_K4_k4** | 4 | [0.25,0.50,0.75,0.95] | competitive | 4 | 13.11 |
| **comp_K4_log_spaced** | 4 | log_spaced | competitive | 8 | 12.47 |
| **comp_K2** | 2 | [0.50,0.95] | competitive | 8 | 13.48 |
| **comp_K8** | 8 | log_spaced | competitive | 8 | **12.06** |
| **struct_K4** | 4 | [0.25,0.50,0.75,0.95] | structural | 8 | — |

- **Paper:** §17 (multi-ξ results subsection, Table `tab:parf-multixi-h128`)
- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_parf_multixi_h128.ipynb`](notebooks/conservative_arch/scaleup/colab_parf_multixi_h128.ipynb)
- **Results:** GDrive `semsimula_parf_multixi_h128/`
- **Key finding:** K=8 log-spaced α achieves **12.06 PPL** — within 1.55× of attention (7.81), closing 77% of the single-ξ-to-attention gap. The K=2→4→8 progression (13.48→12.47→12.06) shows clear diminishing returns, confirming the remaining gap is not addressable by widening ξ alone. The 2.80-PPL improvement over multi-ξ SPLM (14.86) is the first direct evidence that the pair force actively helps beyond multi-ξ V_θ.

### Fock Multi-Xi PARF H=128 scaleup

Adds Fock-space latent register pools (v1 and v2 gates) on top of the
multi-ξ PARF H=128 architecture. Thirteen arms sweep Fock version (v1/v2),
register count (M=4, 8, 16, 32), activation discipline (LIFO vs free),
reverse channel (on/off), routing density (k=4, 8), V_φ kind
(competitive, structural), channel count (K=2, 4), and training schedule
(8k / 16k / 32k steps).

| Arm | Fock | K | M | Disc | Rev | k | V_φ | Steps | PPL |
|-----|------|---|---|------|-----|---|-----|-------|-----|
| **v1_K4_M16_lifo** | v1 | 4 | 16 | LIFO | — | 8 | competitive | 8k | — |
| **v1_K4_M32_lifo** | v1 | 4 | 32 | LIFO | — | 8 | competitive | 8k | — |
| **v1_K4_M16_free** | v1 | 4 | 16 | free | — | 8 | competitive | 8k | — |
| **v2_K4_M16_lifo** | v2 | 4 | 16 | LIFO | ✓ | 8 | competitive | 8k | **14.21** |
| **v2_K4_M16_no_rev** | v2 | 4 | 16 | LIFO | ✗ | 8 | competitive | 8k | — |
| **v2_K4_M32_lifo** | v2 | 4 | 32 | LIFO | ✓ | 8 | competitive | 8k | — |
| **v2_K2_M16_lifo** | v2 | 2 | 16 | LIFO | ✓ | 8 | competitive | 8k | — |
| **v2_K4_M16_k4** | v2 | 4 | 16 | LIFO | ✓ | 4 | competitive | 8k | — |
| **v1_K4_M16_struct** | v1 | 4 | 16 | LIFO | — | 8 | structural | 8k | — |
| **v2_K4_M8_lifo** | v2 | 4 | 8 | LIFO | ✓ | 8 | competitive | 8k | — |
| **v2_K4_M4_lifo** | v2 | 4 | 4 | LIFO | ✓ | 8 | competitive | 8k | — |
| **v2_K4_M16_lifo_16k** | v2 | 4 | 16 | LIFO | ✓ | 8 | competitive | **16k** | — |
| **v2_K4_M16_lifo_32k** | v2 | 4 | 16 | LIFO | ✓ | 8 | competitive | **32k** | — |

- **Paper:** §17 (first result: v2_K4_M16_lifo = 14.21 PPL)
- **Model:** [`notebooks/conservative_arch/parf/model_fock_parf_multixi.py`](notebooks/conservative_arch/parf/model_fock_parf_multixi.py)
- **Training script:** [`notebooks/conservative_arch/scaleup/train_fock_multixi_scaleup.py`](notebooks/conservative_arch/scaleup/train_fock_multixi_scaleup.py)
- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_fock_multixi_h128.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_multixi_h128.ipynb)
- **Results:** [`notebooks/conservative_arch/scaleup/results/semsimula_fock_multixi_h128/`](notebooks/conservative_arch/scaleup/results/semsimula_fock_multixi_h128/) + GDrive
- **Key question:** Can Fock registers close the remaining gap between multi-ξ PARF (12.06 PPL) and attention (7.81 PPL)?
- **First finding:** v2_K4_M16_lifo achieves 14.21 PPL (best 13.76 at step 7600) — 2.15 PPL above non-Fock baseline. Fock registers currently add interference rather than value. Two diagnostic arms added: M=4 (interference test) and 16k steps (convergence test).
- **Status:** in progress — 1/13 arms completed

### Scale-up PARF OOM picture

Five distinct CUDA OOM failure modes catalogued at scale-up.

- **Paper:** §17.9.2, Table 29

---

## V_theta regularisation sweeps (§17b)

### SPLM V_theta regularisation (VR0–VR5)

| Code | λ_V | Description |
|------|-----|-------------|
| **VR0** | 0 | Unregularised |
| **VR1** | 1e-6 | |
| **VR2** | 1e-4 | |
| **VR3** | 1e-2 | |
| **VR4** | 1 | |
| **VR5** | — | Verlet integrator variant (L=16, dt=0.5) |

- **Paper:** §17b, Table 26
- **Notebook:** [`notebooks/conservative_arch/parf/scripts/vreg_sweep_v_theta_regularisation.ipynb`](notebooks/conservative_arch/parf/scripts/vreg_sweep_v_theta_regularisation.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/vreg_sweep/VR0/`](notebooks/conservative_arch/parf/results/vreg_sweep/) through `VR5/`

### PARFLM V_theta regularisation (PR0–PR4)

| Code | λ_V | Best PPL |
|------|-----|----------|
| **PR0** | 0 | 246.4 |
| **PR1** | 1e-6 | 191 |
| **PR2** | 1e-4 | **186.0** |
| **PR3** | 1e-2 | |
| **PR4** | 1 | |

- **Paper:** §17b, Table 26, Figure 21 (PR2 attractors), Figure 22 (V_theta histograms)
- **Notebook:** [`notebooks/conservative_arch/parf/scripts/vreg_sweep_parf.ipynb`](notebooks/conservative_arch/parf/scripts/vreg_sweep_parf.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/vreg_sweep_parf/PR0/`](notebooks/conservative_arch/parf/results/vreg_sweep_parf/) through `PR4/`
- **Key finding:** PR2 (λ_V=10⁻⁴) achieves the largest PPL improvement of any conservative architecture: −60.4 PPL vs PR0. GD attractor extraction converges at 99.9% (1919/1920 prompts). V_theta range collapses from 59 (PR0) to 20 (PR2). This is the canonical gauge-breaking recipe.

### FockPARF V_theta regularisation (FR0–FR4)

| Code | λ_V |
|------|-----|
| **FR0** | 0 |
| **FR1** | 1e-6 |
| **FR2** | 1e-4 |
| **FR3** | 1e-2 |
| **FR4** | 1 |

- **Paper:** §17b, Table 26
- **Notebook:** [`notebooks/conservative_arch/parf/scripts/vreg_sweep_fockparf.ipynb`](notebooks/conservative_arch/parf/scripts/vreg_sweep_fockparf.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/vreg_sweep_fockparf/FR0/`](notebooks/conservative_arch/parf/results/vreg_sweep_fockparf/) through `FR4/`
- **Key finding:** FockPARF is most tolerant of V_theta regularisation via V_phi and register lifecycle (M=16 registers, d=128, L=8).

### Hybrid V_theta regularisation (HR0–HR4)

| Code | λ_V |
|------|-----|
| **HR0** | 0 |
| **HR1** | 1e-6 |
| **HR2** | 1e-4 |
| **HR3** | 1e-2 |
| **HR4** | 1 |

- **Paper:** §17b, Table 26
- **Notebook:** [`notebooks/conservative_arch/hybrid/scripts/vreg_sweep_hybrid.ipynb`](notebooks/conservative_arch/hybrid/scripts/vreg_sweep_hybrid.ipynb)
- **Results:** [`notebooks/conservative_arch/hybrid/results/vreg_sweep_hybrid/HR0/`](notebooks/conservative_arch/hybrid/results/vreg_sweep_hybrid/) through `HR4/`
- **Key finding:** Hybrid PPL stays within 0.7–1.6 PPL of unregularised HR0 (140.4) at all λ_V — the attention front-end pre-normalises the state, making V_theta regularisation effectively free.

### Planned confound-resolution sweeps (G1–G3)

| Code | Description |
|------|-------------|
| **G1** | PARF γ sweep at λ_V=10⁻⁴, d_V=128, γ∈{0.05, 0.10, 0.15} |
| **G2** | PARF d_V sweep at λ_V=10⁻⁴, γ=0.10, d_V∈{128, 256, 512} |
| **G3** | SPLM em_ln at d_V=512, γ=0.10, λ_V=10⁻⁴ |

- **Paper:** §17b
- **Notebook:** [`notebooks/conservative_arch/parf/scripts/fockparf_improvement_sweep.ipynb`](notebooks/conservative_arch/parf/scripts/fockparf_improvement_sweep.ipynb) (G1–G3 cells)
- **Results:** planned — no results yet
- **Status:** planned

---

## Structured V_theta (§17)

### Structured V_theta sweep (SQ1–SQ5)

Tests whether structured V_theta (analytical gradient, fewer params)
can match the PR2 MLP baseline on TinyShakespeare.

| Code | V_theta variant | Description |
|------|----------------|-------------|
| **SQ1** | Diagonal quadratic | Cheapest; axis-aligned wells |
| **SQ2** | Low-rank quadratic (rank=8) | Rotated wells |
| **SQ3** | Mixture K=4 | K Gaussian-well basins |
| **SQ4** | Quadratic + small MLP residual | Hybrid interpretable + learned |
| **SQ5** | MLP reference | Full MLP rerun |

- **Paper:** §17.10, Table 27
- **Notebook:** [`notebooks/conservative_arch/parf/scripts/structured_vtheta_sweep.ipynb`](notebooks/conservative_arch/parf/scripts/structured_vtheta_sweep.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/structured_vtheta/SQ3/`](notebooks/conservative_arch/parf/results/structured_vtheta/) through `SQ5/`
- **Key finding:** SQ3 (mixture K=4) matches the PR2 MLP baseline on PPL with analytical ∇_h V_theta, zero autograd cost, and interpretable basins.

### Structured V_theta TinyStories sweep (A1–A10, B1–B2)

Systematic comparison at TinyStories scale. Fixed SparsePARFLM backbone
(d=256, L=8); only V_theta is swapped.

| Code | V_theta variant |
|------|----------------|
| **A1** | SQ1 diagonal quadratic |
| **A2** | SQ2 low-rank (rank=4) |
| **A3** | SQ2 low-rank (rank=16) |
| **A4** | SQ3 mixture K=4 |
| **A5** | SQ3 mixture K=8 |
| **A6** | SQ3 mixture K=16 |
| **A7** | SQ3 mixture K=4 (tau variant) |
| **A8** | SQ3 mixture K=8 (tau variant) |
| **A9** | SQ4 small MLP residual |
| **A10** | SQ4 large MLP residual |
| **B1** | MLP baseline (v_hidden=2048) |
| **B2** | MLP baseline (v_hidden=512) |

- **Paper:** §17.10, Table 27, Figure 23
- **Notebook:** [`notebooks/conservative_arch/parf/scripts/structured_vtheta_tinystories_sweep.ipynb`](notebooks/conservative_arch/parf/scripts/structured_vtheta_tinystories_sweep.ipynb)
- **Key finding:** SQ3 matches the MLP reference at TinyStories scale. SQ5 (MLP rerun) diverges due to large initial ‖V_theta‖² destabilising next-token loss.

---

## FockPARF v2 / QFT (§17c)

### FockPARF v2 debug (D1–D9)

Debug and tune FockPARF v2 on TinyStories before full P10g scale. Tests
six bottleneck hypotheses.

| Code | Description |
|------|-------------|
| **D1** | Baseline FockPARF v2 (M=16) |
| **D2** | M=32 registers |
| **D3** | d_k=128 query width |
| **D4** | Increased blend scale |
| **D5** | tau_create temperature |
| **D6** | Learnable τ |
| **D7** | + Gumbel-Softmax |
| **D8** | + per-register keys |
| **D9** | + orthogonal init |

- **Paper:** §17c (referenced; detailed subsections use Q0–Q8 for the controlled ladder)
- **Notebook:** [`notebooks/conservative_arch/parf/scripts/fockparf_v2_tinystories_debug.ipynb`](notebooks/conservative_arch/parf/scripts/fockparf_v2_tinystories_debug.ipynb)

### QFT v2.1 experimental ladder (Q0–Q8)

Controlled 9-arm experiment testing four QFT-motivated creation-gate
improvements.

| Code | Description | Best PPL | vs Q0 |
|------|-------------|----------|-------|
| **Q0** | FockPARF v2 baseline | 58.84 | — |
| **Q1** | + Gumbel-Softmax creation | 58.83 | −0.0% |
| **Q2** | + per-register key subspaces | 59.38 | +0.9% |
| **Q3** | + orthogonal W_Q init | 56.35 | −4.2% |
| **Q4** | + canonical destruction | 55.89 | −5.0% |
| **Q5** | Full QFT v2.1 (all four) | 58.48 | −0.6% |
| **Q6** | Q5 + learnable τ (τ_0=1.0) | **53.47** | **−9.1%** |
| **Q7** | Q5 + M=32 registers | 122.24 | +108% |
| **Q8** | PARFLM baseline (no registers) | 56.43 | −4.1% |

- **Paper:** §17c.5, Table 33
- **Notebook:** [`notebooks/conservative_arch/parf/scripts/fockparf_v2_qft_improvements.ipynb`](notebooks/conservative_arch/parf/scripts/fockparf_v2_qft_improvements.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/fock_v2_qft/Q0/`](notebooks/conservative_arch/parf/results/fock_v2_qft/) through `Q8/`
- **Key finding:** Q6 (full QFT stack + learnable temperature) achieves 53.47 PPL — 9.1% below FockPARF v2 baseline and 5.2% below plain PARFLM. The four QFT improvements are not optional refinements — they are what makes the register mechanism beneficial. Q7 (M=32) diverges: diversity collapses at layers 5–7.

### QFT v2.1 planned series (D6–D10)

Planned incremental tests isolating each QFT-motivated change.

| Code | Description |
|------|-------------|
| **D6** | Learnable τ only (baseline) |
| **D7** | + Gumbel-Softmax |
| **D8** | + per-register keys |
| **D9** | + orthogonal initialisation |
| **D10** | + canonical destruction |

- **Paper:** §17c.4
- **Status:** planned — no notebook yet

### Dyck-2 falsifier (F2)

Three-arm controlled comparison on synthetic bracket-matching.

| Code | Description | Best loss | Deep-test acc |
|------|-------------|-----------|---------------|
| **F2-baseline** | PARFLM (no registers) | 3.078 | 43.64% |
| **F2-fock-v1** | Mean-conditioned creation | 3.037 | 45.12% |
| **F2-fock-v2** | Q/K/V creation + gated reverse | **2.856** | **49.01%** |

- **Paper:** §17c.2, Table 32
- **Notebook:** [`notebooks/conservative_arch/parf/scripts/fockparf_v2_dyck2_falsifier.ipynb`](notebooks/conservative_arch/parf/scripts/fockparf_v2_dyck2_falsifier.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/fock_v2/F2-baseline_seed0/`](notebooks/conservative_arch/parf/results/fock_v2/) through `F2-fock-v2_seed0/`
- **Key finding:** FockPARF v2 achieves the best Dyck-2 performance at 49.01% deep-test accuracy, approaching the 50% success criterion.

### FockPARF improvement sweep (P1–P5 Fock, G1–G3)

Eight cells targeting the ~40 PPL gap between FockPARF+λ_V=1 (~190 PPL)
and attention (~150) on TinyShakespeare.

| Code | Strategy |
|------|----------|
| **P1** | Hybrid FockPARF + Attention |
| **P2** | Wider V_theta |
| **P3** | More registers |
| **P4** | Width scale-up |
| **P5** | Phased gates |
| **G1–G3** | γ/v_hidden confound resolution |

- **Paper:** not yet in paper (notebook-only experiments)
- **Notebook:** [`notebooks/conservative_arch/parf/scripts/fockparf_improvement_sweep.ipynb`](notebooks/conservative_arch/parf/scripts/fockparf_improvement_sweep.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/fockparf_improvement/P1/`](notebooks/conservative_arch/parf/results/fockparf_improvement/) through `P5/`

---

## Architecture comparison

### Architecture comparison (S1–S4)

TinyStories v3 scale-up: replaces MLP V_theta with structured SQ3
(K=4) across four architectures at d=256.

| Code | Architecture |
|------|-------------|
| **S1** | Regularised PARF |
| **S2** | Regularised FockPARF (M=32 registers) |
| **S3** | Hybrid FockPARF + Attention |
| **S4** | Hybrid SPLM + Attention reference |

- **Paper:** not yet in paper (notebook-only)
- **Notebook:** [`notebooks/conservative_arch/parf/scripts/tinystories_parf_vs_fockparf.ipynb`](notebooks/conservative_arch/parf/scripts/tinystories_parf_vs_fockparf.ipynb) (also at [`notebooks/conservative_arch/tinystories_parf_vs_fockparf.ipynb`](notebooks/conservative_arch/tinystories_parf_vs_fockparf.ipynb))
- **Results:** [`notebooks/conservative_arch/parf/results/tinystories_scaleup/`](notebooks/conservative_arch/parf/results/tinystories_scaleup/) (S1, S2), [`notebooks/conservative_arch/parf/results/tinystories_v2/`](notebooks/conservative_arch/parf/results/tinystories_v2/) (S2, S3)

### Architecture comparison Phase 2 (L1–L3)

TinyStories Phase 2 large-scale comparison.

| Code | Architecture |
|------|-------------|
| **L1** | Large PARF |
| **L2** | Large FockPARF |
| **L3** | Large Hybrid |

- **Paper:** not yet in paper (notebook-only)
- **Notebook:** [`notebooks/conservative_arch/parf/scripts/tinystories_parf_phase2_large.ipynb`](notebooks/conservative_arch/parf/scripts/tinystories_parf_phase2_large.ipynb)
- **Results:** not yet available

---

## Non-conservative / SP-HSPLM

### SP-HSPLM Stage 1 (e0–e5)

Per-token non-conservative terms (gyroscopic and solenoidal forces)
testing whether they close the SPLM-vs-attention gap.

| Code | Description |
|------|-------------|
| **e0_baseline** | em_ln comparator |
| **e1_const_skew** | Full-rank gyroscopic |
| **e2_affine_rank1** | Affine rank-1 |
| **e3_lowrank_rank2** | Low-rank rank-2 |
| **e4_solenoidal_rank4** | Solenoidal rank-4 |
| **e5_lowrank_rank4** | Low-rank rank-4 |

- **Paper:** not yet in paper (notebook-only)
- **Notebook:** [`notebooks/conservative_arch/non_conservative/scripts/sp_hsplm_stage1_a100_h100.ipynb`](notebooks/conservative_arch/non_conservative/scripts/sp_hsplm_stage1_a100_h100.ipynb)
- **Results:** [`notebooks/conservative_arch/non_conservative/results/sp_hsplm/stage1/`](notebooks/conservative_arch/non_conservative/results/sp_hsplm/stage1/)

### SP-HSPLM Stage 2 (q9e_a–q9e_n)

Q9(e) pair-skew cell ladder covering mechanism-2 autonomous force law,
mechanism-1 extensions, additivity tests, and full Class F.

| Code | Description |
|------|-------------|
| **q9e_a** | Mechanism-2 autonomous |
| **q9e_b–q9e_n** | Progressive additions through full Class F |

- **Paper:** not yet in paper (notebook-only)
- **Notebook:** [`notebooks/conservative_arch/sphsplm/scripts/sp_hsplm_stage2_a100_h100.ipynb`](notebooks/conservative_arch/sphsplm/scripts/sp_hsplm_stage2_a100_h100.ipynb)
- **Results:** [`notebooks/conservative_arch/sphsplm/results/sp_hsplm/stage2/`](notebooks/conservative_arch/sphsplm/results/sp_hsplm/stage2/)

---

## Inference efficiency (Appendix A2)

### Inference efficiency benchmark (E8)

Pre-registered three-phase inference benchmark.

| Phase | Description | Grade |
|-------|-------------|-------|
| **Phase 1** | Matched-compute quality | Q3 (matched-attn beats SPLM by ~18–31 PPL) |
| **Phase 2** | Per-token wall-clock and FLOP | C (2 CONFIRMED, 2 MARGINAL, 2 REFUTED) |
| **Phase 3** | Quality-adjusted Pareto | Attention dominates at prototype scale |

- **Paper:** Appendix A2, Tables 38–40
- **Results:** [`notebooks/conservative_arch/inference_efficiency/results/`](notebooks/conservative_arch/inference_efficiency/results/)
- **Key finding:** SPLM achieves ~38% FLOP advantage at $T\geq 512$ but 2.5–3× wall-clock overhead due to `create_graph=True`. Phase 2 structured V_theta (SQ3) path is expected to recover 30–40% of the wall-clock overhead.

---

## Formal-language falsifiers (§9)

### F1–F6 falsifier programme

Six experiments testing the v0 lower-bound staircase and MCS bounds.

| Code | Target language | Tests |
|------|----------------|-------|
| **F1** | Dyck_n (base) | Falsifies v0 alone; motivates v2. Predicted collapse depth D* |
| **F2** | Dyck_n + topic-shift | Falsifies v0+v2 without decay; motivates v1.5 |
| **F3** | Dyck_n + let-binding | Falsifies v0+v2+v1.5 without operators; motivates v3 |
| **F4** | Cross-serial a^n b^n c^n | Confirms v0+v2+v1.5+v3 reaches MCS |
| **F5** | Bounded copy ww | Confirms MCS reach via different construction |
| **F6** | 2-counter machine simulation | Falsifies the upper bound (intentional over-strong) |

- **Paper:** §9.7
- **Status:** F2 partially run via the Dyck-2 falsifier notebook (see [F2 arms](#dyck-2-falsifier-f2) above). F1 prediction is closed-form. F3–F6 are post-M2 deliverables — planned, no notebooks yet.

---

## PARFLM scaling optimisations

Engineering proof-of-concept notebooks validating memory-reduction
techniques for PARFLM. These address the `create_graph=True` memory
bottleneck.

| Notebook | Description |
|----------|-------------|
| [gradient_checkpoint_poc](notebooks/conservative_arch/parf/scripts/gradient_checkpoint_poc.ipynb) | Level 2 per-layer-step gradient checkpointing POC |
| [gradient_checkpoint_gathered_vphi_poc](notebooks/conservative_arch/parf/scripts/gradient_checkpoint_gathered_vphi_poc.ipynb) | Integrated Level 2 checkpointing + Stage-1.5b gathered V_phi POC |

- **Paper:** §17.12 (scaling discussion)
- **Design docs:** [`companion_notes/Gradient_Checkpointing_for_PARF.md`](companion_notes/Gradient_Checkpointing_for_PARF.md), [`companion_notes/PARF_Stage_1_5b_design.md`](companion_notes/PARF_Stage_1_5b_design.md)
- **Key finding:** Combined Level 2 checkpointing + Stage-1.5b reduces peak V_phi activation memory from O(L·B·T²·H) to O(B·T·k·H) — constant in L, linear in T. At B=16, T=512, L=8, k=4: ~10–14 GB → ~8 MB (~1000× reduction).

---

## Predictions and hypotheses

These are not experiment runs but testable claims and research
questions used throughout the paper.

### Predictions (P1–P4, §11)

| Code | Prediction |
|------|-----------|
| **P1** | Geodesic: hidden-state trajectories approximate near-geodesics |
| **P2** | Clustering: semantically related tokens cluster in potential wells |
| **P3** | Well depth: potential well depth correlates with attention mass |
| **P4** | Bound-state proximity: heavier tokens sit closer to trajectory centroid |

### Hypotheses (H1–H3, §13/§18)

| Code | Hypothesis |
|------|-----------|
| **H1** | Jacobi metric governs hidden-state geometry |
| **H2** | Overdamped regime: dynamics is effectively first-order |
| **H3** | Per-layer potential variation dominates over autonomous dynamics |

### Open questions (Q1–Q9, §19)

| Code | Question |
|------|---------|
| **Q1** | Is the hidden-state dynamics first-order or second-order? |
| **Q2** | Can the framework match attention-level PPL? |
| **Q3** | What is the optimal γ at scale? |
| **Q4** | Can Fock-space structure close the remaining PPL gap? |
| **Q5** | Does the F1 falsifier collapse prediction hold empirically? |
| **Q6** | (a) Can v3 operators be learned end-to-end? (b) Do they improve PPL? |
| **Q7** | Is the MCS upper bound tight? |
| **Q8** | Can the framework scale to GPT-2-level parameter counts? |
| **Q9** | Which hybrid design (a–d) is the right architecture for practice? |

---

## Overloaded codes

Some short codes are reused across different contexts. This table
disambiguates them.

| Code | Context 1 | Context 2 | Context 3 |
|------|-----------|-----------|-----------|
| **P1** | PARF Stage 1 cell (§17.8.2) | Hidden-state prediction (§11) | E8 inference grade (App A2) |
| **P1–P5** | PARF cells P1, P1.5, P1.6, P5 (§17) | FockPARF improvement sweep cells (notebook-only) | — |
| **M1–M4** | Experiment A dynamics models (§18.7) | — | — |
| **M1–M5** | Mass axioms (§12.1) | — | — |
| **H1–H3** | Dynamics hypotheses (§13, §18) | — | — |
| **H0–H6** | Hybrid SPLM experiment cells (§16) | — | — |
| **Q1–Q9** | Open research questions (§19) | — | — |
| **Q0–Q8** | QFT v2.1 experiment cells (§17c) | — | — |
| **E1** | Divergence diagnostic (§15.12) | Scalar-potential fit (§15.5) | — |
| **F2** | Falsifier programme (§9.7) | Dyck-2 arms F2-baseline etc. (§17c) | — |

---

## Other local-only notebooks

These notebooks have no Colab setup and no short experiment codes. They
run locally and auto-detect CUDA/MPS/CPU.

| Notebook | Path | Description |
|----------|------|-------------|
| e_init_validation | `notebooks/e_init/e_init_validation.ipynb` | Forward integration of Euler–Lagrange from GPT-2 first-block initial conditions |
| pythia_tangential_acceleration | `notebooks/cross_model/pythia_tangential_acceleration.ipynb` | Cross-model tangential acceleration test (GPT-2 vs Pythia-160M) |
| energy_landscape_validation | `notebooks/stp_loss/energy_landscape_validation.ipynb` | Gaussian-well energy landscape hypothesis validation |
| energy_landscape_validation_executed | `notebooks/stp_loss/energy_landscape_validation_executed.ipynb` | Same as above with frozen execution outputs |
