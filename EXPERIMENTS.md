# Experiment Index

The paper uses short experiment codes (P10g, SQ3, VR2, etc.) throughout.
This file is the master index: look up any code to find what it tests,
where it appears in the paper, which notebook runs it, and where the
result artefacts live.

**Convention.** Most notebooks are multi-arm: a `CELL` variable at the
top selects the experiment.  Set `CELL`, then execute all cells
sequentially.  See [COLAB_NOTEBOOKS.md](COLAB_NOTEBOOKS.md) for Colab
run instructions.

**Links from the book.** Entries that start with an HTML anchor, such as
`<a name="sr3-crit"></a>**SR3-crit**`, are the targets of the experiment
names linked in the book (the `\expt` macro). Keep each anchor and the
bold name after it unchanged; the book's build reads them from this file.
To add an entry, give it a new anchor (lower case, letters, digits and
hyphens) and the name exactly as the book writes it.

---

## Quick-lookup table

| Code | Book (current; "June 2026" = old numbering) | One-line description | Status |
|------|-----------|---------------------|--------|
| [A1–A10](#a1a10) | Table 26 | Structured V_theta TinyStories sweep (SQ1–SQ4 + MLP variants) | completed |
| [B1–B2](#a1a10) | Table 26 | MLP V_theta baselines for structured V_theta comparison | completed |
| [SPLM-SVT-A1–A5](#splmsvt) | §22.4.2 | Structured V_theta (SQ3) sweep on Multi-Xi SPLM — 5 arms, best A2 **13.33 PPL** | completed |
| [PARF-SVT-A2](#parfsvt) | §22.4.3 | Structured V_theta (SQ3) on Multi-Xi PARFLM — A2 arm, **12.27 PPL**, 0.17 PPL gap | completed |
| [Fock-SVT-A2](#focksvt) | §22.4.4 | Structured V_theta (SQ3) on Fock-PARFLM v2.1 — A2 arm, **10.36 PPL**, Fock paradox | completed |
| [D1–D9](#d1d9) | Table 98 | FockPARF v2 debug cells on TinyStories | completed |
| [D6–D10](#d6d10) | Table 98 | Planned QFT v2.1 incremental tests | planned |
| [E-init / E15](#einit) | §13.7 | Forward integration of Euler–Lagrange from GPT-2 initial conditions | completed |
| [E1](#e1ms) | Table 7 | Multi-seed SPLM validation with LN-after-step rescue | completed |
| [E1–E3](#e1e3) | §16.6 | Scalar-potential fits (seven functional forms) | completed |
| [E4](#e4damp) | Table 8 | Controlled-γ damping sweep on SPLM | completed |
| [E5](#e5) | §16.6 | Velocity-coupled electromagnetic-analogue gauge sweep | completed |
| [E8](#e8) | Table 86 | Pre-registered inference benchmark (quality, wall-clock, Pareto) | completed |
| [E9](#e9) | Table 11 | SPLM scale-up validation on H100 (TinyStories d=256, L=8) | completed |
| [E10](#e10) | Table 12 | γ-transfer diagnostic: γ* at scale-up | completed |
| [F1–F6](#f1) | §10.8 | Six-falsifier programme testing v0 staircase and MCS bounds | partial |
| [F2-baseline / F2-fock-v1 / F2-fock-v2](#f2baseline) | Table 98 | FockPARF v2 Dyck-2 controlled comparison | completed |
| [FR0–FR4](#fr) | Table 33 | FockPARF V_theta regularisation sweep | completed |
| [G1–G3](#g1) | Table 33 | Planned PARF γ / d_V / em_ln confound sweeps | planned |
| [H0 + H1](#h0h1) | Table 18 | Hybrid schedule sweep at S=1 | completed |
| [H1.5](#h1p5) | Table 19 | V_theta-narrow FLOP-arm ablation | completed |
| [H2](#h2) | Table 20 | Hybrid paired confirmation at n=5 | completed |
| [H6](#h6) | Table 21 | Substack-restricted shared-V_ψ separator test | completed |
| [HR0–HR4](#hr) | Table 33 | Hybrid V_theta regularisation sweep | completed |
| [L1–L3](#l1l3) | Table 99 | TinyStories Phase 2 large-scale architecture comparison | completed |
| [M1–M4](#m1) | Table 56 | First-order vs second-order ODE trajectory fitting on GPT-2 | completed |
| [multixi_buggy_2k / multixi_pilot_fixed](#multi-ξ-leak-fix-forensics) | §15.14.3, Tab 26 (June 2026) | Multi-ξ K-channel SPLM leak-fix forensic runs | completed |
| [N5](#n5) | §18.3 | Five-seed paired confirmation at scale-up | completed |
| [OQ-1](#oq1) | §19.9 | Pre-registered structural-vs-MLP V_phi ablation | completed |
| [P0](#p0) | §19.9 | PARF smoke verification and causal-violation probe | completed |
| [P1](#p1) | Table 24 | PARF structural V_phi at seed 0 | completed |
| [P1.5 / P1.5a](#p1p5) | Table 24 | OQ-1 ablation: structural vs unstructured MLP V_phi | completed |
| [P1.6](#p1p6) | §19.9 | Wider structural V_phi capacity disambiguation | completed |
| [P5](#p5) | §19.9 | Stage 1.5 Gumbel-softmax sparsity at top-k=4 | completed |
| [P6](#p6) | Table 24 | Channel-level gate-selectivity diagnostic | completed |
| [P7 / P8](#p7p8) | Table 24 | Force-balance and saturation-fix patches | completed |
| [P10d–P10j](#p10d) | Table 24 | P10 ablation ladder on TinyStories | completed |
| [P1–P5 (Fock)](#p1p5fock) | Table 98 | FockPARF improvement sweep closing ~40 PPL gap | completed |
| [PR0–PR4](#pr) | Table 33 | PARFLM V_theta regularisation sweep | completed |
| [Q0–Q8](#q0) | Table 98 | QFT v2.1 incremental creation-gate improvements | completed |
| [Q9c / Q9d](#q9c) | §19.9; §18.3 | PARF-augmented SPLM (Q9c) and Helmholtz hybrid (Q9d) | completed |
| [R6.*](#r6h0) | Table 2 | R6 information-bottleneck ladder | completed |
| [S1–S4](#s1s4) | Table 99 | TinyStories v3: PARF vs FockPARF vs Hybrids with SQ3 V_theta | completed |
| [Scope 3](#scope3) | Table 4 | Leak-free S=5 retrain of v2 SPLM experiments | completed |
| [SQ1–SQ5](#sq1) | Table 26 | Structured V_theta expressivity test on TinyShakespeare | completed |
| [MXP-H16](#multi-xi-parf-h16-pilot) | §17 (June 2026) | Multi-ξ PARF pilot at H=16 (pre-memory-fix) — 3 arms | completed |
| [MXP-H128](#multi-xi-parf-h128-scaleup) | §17 (June 2026) | Multi-ξ PARF at H=128 (Level-2 ckpt + gathered V_φ) — 6 arms | completed |
| [FMXP-H128](#fock-multi-xi-parf-h128-scaleup) | §17c (June 2026) | Fock Multi-ξ PARF at H=128 (v1/v2 gates, register sweep) — 13 arms | completed |
| [FMXP-v21](#fock-v21-routing-fix) | §17c, Tab 34 (June 2026) | Fock v2.1 routing-fix arms (B1, B1+B2+B3) — PPL **9.30** | completed |
| [FATTN-H128](#fock-attention-h128) | §17c, Tab 34 (June 2026) | Fock Attention direct exchange — 4 arms, best PPL 9.42 | completed |
| [CONS1–5](#cons) | §20.8 | Controlled-conservativity 5-arm diagnostic on trained v2.1 checkpoint | completed |
| [Stage-1.5a](#stage15a) | §19.9; §19.10 | Dense V_phi forward with post-masking | completed |
| [Stage-1.5b](#stage15b) | §19.9; §19.10 | Gathered V_phi (top-k source gathering) | design only |
| [VR0–VR5](#vr) | Table 33 | Standalone SPLM V_theta regularisation sweep | completed |
| [Fock-Gauss-Aniso, Fock-G1, RD, SU-iso, SU-aniso, SCAF-stiff](#gaussian-scaleup) | Appendix D | Gaussian V_θ on Fock-PARFLM, the OpenWebText scale-up and the Riemannian diagnostics | completed |
| [CG1–CG8, SR1–SR5, CB1–CB3, DP1–DP3](#ladder-diagnostics) | Appendix D | Geometry, settling, balance and register diagnostics on the ladder | mixed |
| [F3.1, G2, G3, G3′, XRB-F, PM1, Gen 2, GPT-2 …](#ladder) | Table 101 | The CfC/BAOAB depth ladder on OpenWebText: 18 arms, Gen 2 and Gen 3, and the matched GPT-2 references | completed |
| [SR2, FO-b-OWT, SR3-crit, PMX, …](#fo-series) | §26.9, Tab 45, 47–48, 51 | First- vs second-order twins on the exact integrator: 19 arms on OpenWebText, TinyStories and code | completed |
| [e0–e5 (SP-HSPLM Stage 1)](#e0e5) | Table 99 | Per-token non-conservative force experiment (Class B/C) | completed |
| [q9e_a–q9e_n (SP-HSPLM Stage 2)](#q9e) | Table 99 | Q9(e) pair-skew cell ladder | completed |
| [Pilot Arms 1–5b](#pilot) | §18.3 | Five-arm SPLM-family scale-up comparison | completed |
| [H1–H3 (hypotheses)](#predictions-and-hypotheses) | §13, §18 (June 2026) | Jacobi / overdamped dynamics hypotheses | hypothesis |
| [M1–M5 (mass axioms)](#mass-axioms-m1m5) | §12.1 (June 2026) | Five axioms any mass candidate must satisfy | axiom |
| [P1–P4 (predictions)](#predictions-and-hypotheses) | §11 (June 2026) | Hidden-state testable predictions (geodesic, clustering, wells) | prediction |
| [Q1–Q9 (open questions)](#open-questions-q1q9-19) | §19 (June 2026) | Research-agenda open questions | open question |

---

## Experiment A — Trajectory fitting (§18)

### Experiment A: trajectory fitting (M1–M4)

<a name="m1"></a>**M1** · <a name="m2"></a>**M2** · <a name="m3"></a>**M3** · <a name="m4"></a>**M4** — Book: Table 56 (`tab:experiment-a`)

Tests whether GPT-2 hidden-state trajectories follow first-order or
second-order autonomous ODE dynamics.

| Code | Description | Best test R² (K=8) |
|------|-------------|-------------------|
| **M1** | First-order physics: gradient descent on learned V_ψ | −0.47 |
| **M2** | Second-order physics: damped Velocity-Verlet on V_ψ | −0.43 |
| **M3** | General lag-1 MLP (no physics structure) | −0.93 |
| **M4** | General lag-2 MLP | −0.81 |

- **Per-layer extension:** §18.8, Table 31, Figure 32–33
- **Notebook:** [`notebooks/dynamics_order_test/scripts/experiment_a_trajectory_fitting.ipynb`](notebooks/dynamics_order_test/scripts/experiment_a_trajectory_fitting.ipynb)
- **Per-layer notebook:** [`notebooks/dynamics_order_test/scripts/experiment_a_per_layer_sweep.ipynb`](notebooks/dynamics_order_test/scripts/experiment_a_per_layer_sweep.ipynb)
- **Results:** [`notebooks/dynamics_order_test/results/experiment_a/`](notebooks/dynamics_order_test/results/experiment_a/), [`notebooks/dynamics_order_test/results/experiment_a_per_layer/`](notebooks/dynamics_order_test/results/experiment_a_per_layer/)
- **Key finding:** All four models have negative test R² (worse than predicting the mean). Physics-structured models (M1, M2) degrade more gracefully at long rollout horizons. M2 learns γ=2.40 (velocity retention 29%), independently confirming the overdamped regime. Per-layer sweep shows M2 fails catastrophically at every layer (R²=−35 at layer 3), confirming the non-autonomous regime.

---

## E-init mass validation (§12)

### E-init: mass validation (E15)

<a name="einit"></a>**E-init** — Book: §13.7 (`subsec:e-init-protocol`)

Forward integration of the Euler–Lagrange equation from GPT-2
first-block initial conditions. Fits per-layer Gaussian-well potentials
and tests symplectic Euler integration with γ calibration.

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

- **Paper (June 2026 numbering):** §12.1

---

## Conservative SPLM programme (§15)

### Five negative scalar-potential fits (E1–E5)

<a name="e1e3"></a>**E1–E3** · <a name="e4"></a>**E4** · <a name="e5"></a>**E5** — Book: §16.6 (`subsec:five-negatives`)

| Code | Description |
|------|-------------|
| **E1–E3** | Scalar-potential sweep across seven functional forms |
| **E4** | Linear Helmholtz augmentation: skew-symmetric solenoidal term |
| **E5** | Velocity-coupled electromagnetic-analogue gauge sweep |

- **Results:** [`notebooks/conservative_arch/results/`](notebooks/conservative_arch/results/)
- **Key finding:** Five prior fitting experiments all tie the static null on held-out data. The rotational/solenoidal component exists in principle but the fit is dominated by per-layer potential variation.

### R6 information-bottleneck ladder

<a name="r6h0"></a>**R6.h.0** · <a name="r6h1"></a>**R6.h.1** · <a name="r6a"></a>**R6.a**<!-- aka: R6a --> · <a name="r6e"></a>**R6.e**<!-- aka: R6e --> · <a name="r6i"></a>**R6.i**<!-- aka: R6i --> — Book: Table 2 (`tab:r6-ladder`)

Multi-ξ K-channel SPLM variants tested in a progressive ladder.

| Code | Description |
|------|-------------|
| **R6.h.0**, **R6.h.1** | Baseline and first rung |
| **R6.a**, **R6.e**, **R6.i** | Completed ladder rungs |
| **R6.b**, **R6.c**, **R6.f**, **R6.h.2**, **R6.h.3** | Deferred rungs |

- **Results:** [`notebooks/conservative_arch/results/`](notebooks/conservative_arch/results/)

### Multi-seed validation and divergence diagnostic (E1)

<a name="e1ms"></a>**E1** (multi-seed) — Book: Table 7 (`tab:e1-leakfree-summary`)

Multi-seed SPLM validation with the LN-after-step rescue.

- **Results:** [`notebooks/conservative_arch/ln_damping_sweep/results/leakfree_3seed/`](notebooks/conservative_arch/ln_damping_sweep/results/leakfree_3seed/)
- **Key finding:** LN-after-step is a necessary rescue for training stability; without it, most seeds diverge or NaN.

### Controlled-γ damping sweep (E4)

<a name="e4damp"></a>**E4** — Book: Table 8 (`tab:e4-leakfree-Ucurve`)

- **Notebook:** included in the `ln_damping_sweep` scripts
- **Results:** [`notebooks/conservative_arch/ln_damping_sweep/results/`](notebooks/conservative_arch/ln_damping_sweep/results/) (gamma subdirs + `leakfree_5seed_confirmation/`)
- **Key finding:** U-shaped PPL-vs-γ curve; overdamped basin at γ∈[0.10, 0.15]; five-seed confirmation locks optimal γ.

### Scope 3: leak-free multi-seed retrain

<a name="scope3"></a>**Scope 3** — Book: Table 4 (`tab:scope3-headline`)

Re-runs v2 SPLM-family TinyShakespeare experiments under the v4
leak-free integrator.

| Cell | Description |
|------|-------------|
| **splm_baseline** | SPLM baseline (5 seeds) |
| **splm_sarf** | SARF variant |
| **splm_sarfmass_embed_head** | SARF-mass embed head |
| **splm_sarfmass_logfreq** | SARF-mass log-frequency |
| **matched_baseline** | Matched-attention baseline |

- **Notebook:** [`notebooks/conservative_arch/multi_seed/colab_scope3.ipynb`](notebooks/conservative_arch/multi_seed/colab_scope3.ipynb)
- **Results:** [`notebooks/conservative_arch/multi_seed/results/scope3_shakespeare/`](notebooks/conservative_arch/multi_seed/results/scope3_shakespeare/)
- **Key finding:** Under leak-free training the SARF xi-recomputation fix reduces the SARF advantage by 1.34×. The per-token mass head still provides a statistically significant benefit.

### SPLM scale-up on H100 (E9)

<a name="e9"></a>**E9** — Book: Table 11 (`tab:e9-phase1`)

- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_pilot.ipynb`](notebooks/conservative_arch/scaleup/colab_pilot.ipynb), [`notebooks/conservative_arch/scaleup/colab_gamma_sweep.ipynb`](notebooks/conservative_arch/scaleup/colab_gamma_sweep.ipynb)
- **Results:** [`notebooks/conservative_arch/scaleup/results/pilot/`](notebooks/conservative_arch/scaleup/results/pilot/), [`notebooks/conservative_arch/scaleup/results/gamma_sweep/`](notebooks/conservative_arch/scaleup/results/gamma_sweep/)
- **Key finding:** Matched-attention beats SPLM em_ln by ~18–31 PPL at the prototype scale (Phase 1 grade Q3).

### γ-transfer diagnostic (E10)

<a name="e10"></a>**E10** — Book: Table 12 (`tab:e10-gamma-sweep`)

Tests whether small-scale optimal γ (0.166 from E5) holds at scale-up.

- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_gamma_sweep.ipynb`](notebooks/conservative_arch/scaleup/colab_gamma_sweep.ipynb)
- **Results:** [`notebooks/conservative_arch/scaleup/results/gamma_sweep/`](notebooks/conservative_arch/scaleup/results/gamma_sweep/)
- **Key finding:** γ*=0.30 at scale-up vs 0.166 at small scale; the transfer is imperfect but the U-shape is preserved.

### Multi-ξ leak-fix forensics

| Code | Description |
|------|-------------|
| **multixi_buggy_2k** | Pre-fix multi-ξ run (leaked integrator) |
| **multixi_pilot_fixed** | Post-fix multi-ξ run (leak-free) |
| **multihippo**, **multis4d** | Planned multi-ξ sweep IDs |

- **Paper (June 2026 numbering):** §15.14.3, Table 26
- **Results:** [`notebooks/conservative_arch/scaleup/results/multixi_buggy_2k/`](notebooks/conservative_arch/scaleup/results/multixi_buggy_2k/), [`notebooks/conservative_arch/scaleup/results/multixi_pilot_fixed/`](notebooks/conservative_arch/scaleup/results/multixi_pilot_fixed/)

---

## Hybrid SPLM (§16)

### H0 + H1 schedule sweep

<a name="h0h1"></a>**H0+H1** — Book: Table 18 (`tab:hybrid-H1`)

Tests six hybrid SPLM+attention schedules at S=1.

- **Results:** [`notebooks/conservative_arch/hybrid/results/h1_sweep/`](notebooks/conservative_arch/hybrid/results/h1_sweep/)
- **Key finding:** Best hybrid (Variant A, k=4, m=4) reaches 133.01 PPL — a 15 PPL improvement over matched attention with identical parameter count.

### H1.5: V_theta-narrow ablation

<a name="h1p5"></a>**H1.5** — Book: Table 19 (`tab:hybrid-H1p5`)

- **Key finding:** V_theta-narrow prevents any cell from clearing the FLOP bar at long context; confirms the need for V_theta capacity.

### H2: paired confirmation

<a name="h2"></a>**H2** — Book: Table 20 (`tab:hybrid-H2-paired`)

- **Results:** [`notebooks/conservative_arch/hybrid/results/h2_paired_confirmation/`](notebooks/conservative_arch/hybrid/results/h2_paired_confirmation/)
- **Key finding:** Hybrid advantage over matched attention confirmed at n=5 seeds.

### H6: substack-restricted V_ψ separator

<a name="h6"></a>**H6** — Book: Table 21 (`tab:hybrid-H6`)

- **Key finding:** The substack-restricted shared-V_ψ separator test is empirically confirmed for the Helmholtz hybrid.

### Q9c / Q9d architecture definitions

<a name="q9c"></a>**Q9c** · <a name="q9d"></a>**Q9d** — Book: §19.9 (`ssec:parf-experiments`); §18.3 (`ssec:hybrid-experiments`)

| Code | Description |
|------|-------------|
| **Q9c** | PARF-augmented SPLM |
| **Q9d** | Helmholtz hybrid (layer-type decomposition) with AAAASSSS schedule |

- **Notebook (Q9d scale-up):** [`notebooks/conservative_arch/scaleup/colab_pilot.ipynb`](notebooks/conservative_arch/scaleup/colab_pilot.ipynb)

### Scale-up pilot (Arms 1–5b)

<a name="pilot"></a>**Pilot** — Book: §18.3 (`ssec:hybrid-experiments`)

Five-arm SPLM-family comparison at the E9 scale-up configuration.

| Arm | Description |
|-----|-------------|
| **Arm 1** | Matched-attention baseline |
| **Arm 2** | SPLM em_ln (fixed γ=0.30) |
| **Arm 2b** | TF32-on sub-arm |
| **Arm 3** | Helmholtz Q9d |
| **Arm 4** | Hybrid VA (k=4, m=4) |
| **Arm 5 / 5b** | PARF Q9c sparse (k=4); H100 scale-up |

- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_pilot.ipynb`](notebooks/conservative_arch/scaleup/colab_pilot.ipynb)
- **Results:** [`notebooks/conservative_arch/scaleup/results/pilot/`](notebooks/conservative_arch/scaleup/results/pilot/)

### Scale-up N5 confirmation

<a name="n5"></a>**N5** — Book: §18.3 (`ssec:hybrid-experiments`)

Five-seed paired confirmation of pilot results with 95% CIs.

- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_n5_confirmation.ipynb`](notebooks/conservative_arch/scaleup/colab_n5_confirmation.ipynb)
- **Results:** [`notebooks/conservative_arch/scaleup/results/n5_confirmation/`](notebooks/conservative_arch/scaleup/results/n5_confirmation/)

---

## PARF empirical programme (§17)

### P0: smoke verification

<a name="p0"></a>**P0** — Book: §19.9 (`ssec:parf-experiments`)

- **Key finding:** PARF passes the causal-violation probe; no signal leakage from future tokens.

### P1: structural V_phi at seed 0

<a name="p1"></a>**P1** — Book: Table 24 (`tab:parf-single-xi-retrospective`)

- **Key finding:** Structural V_phi achieves 210.5 PPL with ~4,000 V_phi params.

### OQ-1: structural vs MLP ablation

<a name="oq1"></a>**OQ-1**<!-- aka: OQ1 --> — Book: §19.9 (`ssec:parf-experiments`)

Pre-registered ablation testing whether the structural prior on V_phi
matters.


### P1.5 / P1.5a: OQ-1 ablation

<a name="p1p5"></a>**P1.5** · <a name="p1p5a"></a>**P1.5a** — Book: Table 24 (`tab:parf-single-xi-retrospective`)

| Code | V_phi type | Val PPL |
|------|-----------|---------|
| **P1** | Structural | 210.5 |
| **P1.5a** | Unstructured MLP | 297.2 |

- **Key finding:** Structural V_phi outperforms unstructured MLP by 87 PPL — OQ-1 verdict confirms the structural prior.

### P1.6: capacity disambiguation

<a name="p1p6"></a>**P1.6** — Book: §19.9 (`ssec:parf-experiments`)

Wider structural V_phi to disentangle capacity from structural prior.

- **Key finding:** PPL gain comes from the structural prior, not from V_phi width.

### P5: Gumbel sparsity at top-k=4

<a name="p5"></a>**P5** — Book: §19.9 (`ssec:parf-experiments`)

Stage 1.5 Gumbel-softmax sparsity.

- **Key finding:** Large PPL gain vs P1.6; k∈{4,8,16} ordering preserved at TinyStories scale.

### P6: channel-level diagnostic

<a name="p6"></a>**P6** — Book: Table 24 (`tab:parf-single-xi-retrospective`)

- **Key finding:** P10j attains better gate selectivity than P10i yet worse PPL — the v0 ceiling binds.

### P7 / P8: composite patches

<a name="p7p8"></a>**P7, P8** — Book: Table 24 (`tab:parf-single-xi-retrospective`)

Four P8 patches: LN-before-distance, per-layer V_phi scale, softsign
theta, bilinear theta.

- **Notebook:** [`notebooks/conservative_arch/parf/scripts/p8_cell_a100_h100.ipynb`](notebooks/conservative_arch/parf/scripts/p8_cell_a100_h100.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/structural/`](notebooks/conservative_arch/parf/results/structural/)
- **Key finding:** Fixes P6 Layer-1 force imbalance and theta saturation.

### P10 ablation ladder (P10d–P10j)

<a name="p10d"></a>**P10d** · <a name="p10f"></a>**P10f** · <a name="p10g"></a>**P10g** · <a name="p10h"></a>**P10h** · <a name="p10i"></a>**P10i** · <a name="p10j"></a>**P10j** — Book: Table 24 (`tab:parf-single-xi-retrospective`)

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

- **Notebook:** [`notebooks/conservative_arch/parf/scripts/p10_tinystories_a100_h100.ipynb`](notebooks/conservative_arch/parf/scripts/p10_tinystories_a100_h100.ipynb) (P10a–P10h), [`notebooks/conservative_arch/parf/scripts/p10_sparsity_ladder_k8_k16.ipynb`](notebooks/conservative_arch/parf/scripts/p10_sparsity_ladder_k8_k16.ipynb) (P10i–P10j)
- **Results:** [`notebooks/conservative_arch/scaleup/results/semsimula_parflm/p10_tinystories/`](notebooks/conservative_arch/scaleup/results/semsimula_parflm/p10_tinystories/) (P10e–P10g), [`notebooks/conservative_arch/scaleup/results/parflm/p10_tinystories/`](notebooks/conservative_arch/scaleup/results/parflm/p10_tinystories/) (P10h–P10j)
- **Key finding:** P10h null result (zero improvement from 4× corpus) is the empirical signature of the v0 expressivity ceiling. Monotone degradation in k confirms k=4 as optimal. The ~26.4 PPL ceiling is confirmed across both corpus and sparsity axes.

### Stage-1.5a / Stage-1.5b: V_phi memory variants

<a name="stage15a"></a>**Stage-1.5a** · <a name="stage15b"></a>**Stage-1.5b** — Book: §19.9 (`ssec:parf-experiments`); §19.10 (`ssec:parf-multixi-results`)

| Code | Description |
|------|-------------|
| **Stage-1.5a** | Dense V_phi evaluation with post-masking (current implementation) |
| **Stage-1.5b** | Gathered V_phi (evaluating only top-k pairs) — proposed |

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

- **Paper (June 2026 numbering):** §17 (multi-ξ results subsection)
- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_parf_multixi.ipynb`](notebooks/conservative_arch/scaleup/colab_parf_multixi.ipynb)
- **Results:** GDrive `semsimula_parf_multixi/`
- **Key finding:** Multi-ξ dramatically improves PARF (15.44 vs 28 PPL with single ξ) but V_φ at H=16 adds ~0.75 PPL overhead vs multi-ξ SPLM alone (14.69 pilot; 11.51 at 16k). V_φ capacity is the binding constraint.

### Multi-Xi SPLM full training rerun (8k / 16k)

Full training rerun of the α-sweep winner (`learned_from_uniform`,
α=[0.25, 0.50, 0.75, 0.95]) at the standard 8k scaleup and extended
16k schedules, to determine the converged Multi-Xi SPLM PPL for the
HuggingFace model card.

| Arm | Steps | PPL |
|-----|-------|-----|
| **scaleup_8k** | 8,000 | 12.49 |
| **extended_16k** | 16,000 | **11.51** |

- **Paper (June 2026 numbering):** §17 (multi-ξ results subsection)
- **Model:** `ScalarPotentialLMSARFMassLNMultiXi` (16.5M params)
- **Training script:** [`notebooks/conservative_arch/scaleup/train_splm_em_ln_multixi_scaleup.py`](notebooks/conservative_arch/scaleup/train_splm_em_ln_multixi_scaleup.py)
- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_splm_multixi_rerun.ipynb`](notebooks/conservative_arch/scaleup/colab_splm_multixi_rerun.ipynb)
- **Results:** [`notebooks/conservative_arch/scaleup/results/splm_multixi_rerun/`](notebooks/conservative_arch/scaleup/results/splm_multixi_rerun/)
- **Key finding:** Extended training from the pilot's 4000 steps (14.69 PPL) to 16,000 steps closes 3.18 PPL, reaching **11.51** — within 3.70 PPL of the attention baseline (7.81) despite no pairwise token interactions. The α channels converge to similar values regardless of training length: [0.25, 0.60, 0.81, 0.96] at 16k.
- **HuggingFace:** Both checkpoints uploaded to [`dimitarpg13/semsimula-splm-multixi`](https://huggingface.co/dimitarpg13/semsimula-splm-multixi)

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

- **Paper (June 2026 numbering):** §17 (multi-ξ results subsection, Table `tab:parf-multixi-h128`)
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
| **v2_K4_M16_lifo_16k** | v2 | 4 | 16 | LIFO | ✓ | 8 | competitive | **16k** | **12.31** (best 12.00 @ 14.4k) |
| **v2_K4_M16_lifo_32k** | v2 | 4 | 16 | LIFO | ✓ | 8 | competitive | **32k** | — |

- **Paper (June 2026 numbering):** §17 (first result: v2_K4_M16_lifo = 14.21 PPL)
- **Model:** [`notebooks/conservative_arch/parf/model_fock_parf_multixi.py`](notebooks/conservative_arch/parf/model_fock_parf_multixi.py)
- **Training script:** [`notebooks/conservative_arch/scaleup/train_fock_multixi_scaleup.py`](notebooks/conservative_arch/scaleup/train_fock_multixi_scaleup.py)
- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_fock_multixi_h128.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_multixi_h128.ipynb)
- **Results:** [`notebooks/conservative_arch/scaleup/results/semsimula_fock_multixi_h128/`](notebooks/conservative_arch/scaleup/results/semsimula_fock_multixi_h128/) + GDrive
- **Key question:** Can Fock registers close the remaining gap between multi-ξ PARF (12.06 PPL) and attention (7.81 PPL)?
- **First finding:** v2_K4_M16_lifo achieves 14.21 PPL at 8k steps (best 13.76 at step 7600) — 2.15 PPL above non-Fock baseline, suggesting Fock registers add interference at short schedules.
- **Second finding:** v2_K4_M16_lifo_16k achieves 12.31 PPL at 16k steps (best **12.00** at step 14,400) — crossing below the non-Fock K=8 baseline (12.06). The Fock mechanism is beneficial but requires ~1.8× more training steps. Three convergence phases observed: rapid descent (0–6k), plateau (6k–12k), resumed improvement (12k–16k). See [convergence analysis](companion_notes/Improving_the_Fock_Mechanism_to_match_Attention.md#18-fock-v2-convergence-slowdown-diagnosis-and-future-work).
- **Status:** completed (2/13 arms ran at this scale; superseded by the v2.1 routing fix below)

### Fock v2.1 routing fix

The v2.1 routing fix addresses the creation-gate routing collapse diagnosed
in §17c.4 via three fixes: B1 (per-register learnable temperature),
B2 (per-register key subspaces), and B3 (orthogonal register init).

| Arm | Fixes | K | M | Steps | PPL |
|-----|-------|---|---|-------|-----|
| **v21_tau_only** | B1 | 4 | 16 | 16k | 11.18 |
| **v21_tau_perK_ortho** | B1+B2+B3 | 4 | 16 | 16k | **9.30** |

- **Paper (June 2026 numbering):** §17c.5, Table 34
- **Model:** [`notebooks/conservative_arch/parf/model_fock_parf_multixi.py`](notebooks/conservative_arch/parf/model_fock_parf_multixi.py), [`notebooks/conservative_arch/parf/model_fock_parf_v2.py`](notebooks/conservative_arch/parf/model_fock_parf_v2.py)
- **Training script:** [`notebooks/conservative_arch/scaleup/train_fock_multixi_scaleup.py`](notebooks/conservative_arch/scaleup/train_fock_multixi_scaleup.py)
- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_fock_v21_routing_fix.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_v21_routing_fix.ipynb)
- **Results:** [`notebooks/conservative_arch/scaleup/results/semsimula_fock_v21_routing_fix/`](notebooks/conservative_arch/scaleup/results/semsimula_fock_v21_routing_fix/) + GDrive
- **Key finding:** The full B1+B2+B3 routing fix achieves **9.30 PPL** — the best result in the SPLM programme without an attention primitive. B1 alone gives only 0.19 PPL improvement (11.37 → 11.18); adding B2+B3 drops a further 1.88. The routing diagnostics confirm the mechanism: normalised entropy rises from 0.04 to 0.45, cross-register diversity from 0.37 to 0.58. The learned reverse-channel scale converges to tanh(s_ex) = −0.227 (repulsive).
- **Design doc:** [`companion_notes/Improving_the_Fock_Mechanism_to_match_Attention.md`](companion_notes/Improving_the_Fock_Mechanism_to_match_Attention.md)

### Fock Attention (H=128)

Direct token-to-token exchange force (Route 2 of §17c), injecting the
bare non-conservative force with no register mediator and no
cross-layer persistence. The $\lambda=0$, $M=0$ limit of the register
mechanism.

| Arm | K | Heads | d_k | Steps | PPL |
|-----|---|-------|-----|-------|-----|
| **direct_K4_h1_8k** | 4 | 1 | 64 | 8k | 11.48 |
| **direct_K4_h4_8k** | 4 | 4 | 32 | 8k | 10.93 |
| **direct_K8_h4_8k** | 8 | 4 | 32 | 8k | 11.65 |
| **direct_K4_h4_16k** | 4 | 4 | 32 | 16k | **9.42** |

- **Paper (June 2026 numbering):** §17c.5, Table 34
- **Model:** [`notebooks/conservative_arch/parf/model_fock_attention.py`](notebooks/conservative_arch/parf/model_fock_attention.py)
- **Training script:** [`notebooks/conservative_arch/scaleup/train_fock_attention_scaleup.py`](notebooks/conservative_arch/scaleup/train_fock_attention_scaleup.py)
- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_fock_attention_h128.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_attention_h128.ipynb)
- **Results:** [`notebooks/conservative_arch/scaleup/results/semsimula_fock_attention_h128/`](notebooks/conservative_arch/scaleup/results/semsimula_fock_attention_h128/)
- **Key finding:** The direct route reaches **9.42 PPL** at 16k steps — both routes cross below 10 PPL. The mediated route (9.30) slightly outperforms the direct route, confirming that once routing is solved, the register mechanism's cross-layer persistence is an advantage. The learned exchange scale is repulsive (tanh(s_ex) ∈ [−0.14, −0.32]), independently confirming the same sign discovered by the mediated route (−0.227). Costs $O(T^2)$ per layer (vs $O(1)$ for registers).

### Conservativity diagnostic (CONS1–5)

<a name="cons"></a>**CONS1–5** — Book: §20.8 (`ssec:fock-conservativity-diagnostic`)

Five-arm empirical verification of FockPARFLM v2.1 controlled
conservativity, run on the trained 9.30 PPL checkpoint
(`v21_tau_perK_ortho`). Validates the three-axis "design knob" claim
of §17c (shape, magnitude, localisation).

| Arm | Test | Result |
|-----|------|--------|
| **CONS1** | Jacobian symmetry — conservative channel curl-free | antisymmetry 1.4% (PASS), Q_i curl 25.8% (PASS) |
| **CONS2** | Conservative ablation — Q_i=0 recovers high R² | deep-layer R² +0.43 with Q_i disabled |
| **CONS3** | Energy budget — ΔH = W_damp + W_Q + residual | all named sources accounted for |
| **CONS4** | Conservativity dial — sweep tanh(s_ex) 0 → −0.227 | PPL 15.95 → 11.46 (28% improvement) |
| **CONS5** | Architectural separator — extend §15 R² diagnostic | GPT-2 0.46 < Fock 0.62 < Fock(Q=0) 0.74 < SPLM 0.96 |

- **Script:** [`notebooks/conservative_arch/parf/conservativity_diagnostic.py`](notebooks/conservative_arch/parf/conservativity_diagnostic.py)
- **Results:** [`notebooks/conservative_arch/parf/results/conservativity_v21/`](notebooks/conservative_arch/parf/results/conservativity_v21/) (5 PNGs, JSON report, summary)
- **Companion note:** [`companion_notes/Fock_PARFLM_Conservativity_Diagnostic.md`](companion_notes/Fock_PARFLM_Conservativity_Diagnostic.md)
- **Key finding:** The conservative channel is curl-free at trained weights (Hessian antisymmetry 1.4%, vs 25.8% for the reverse-channel force Q_i). Disabling Q_i recovers high R² at all layers (Arm 2). The conservativity dial (Arm 4) quantifies the central trade: a 28% PPL reduction is bought by controlled non-conservatism, while R² stays above 0.97 throughout the sweep. FockPARFLM sits at R² = 0.615 on the architectural separator — between unconstrained GPT-2 (0.46) and fully conservative SPLM (0.957), exactly where its hybrid architecture predicts.

### Scale-up PARF OOM picture

Five distinct CUDA OOM failure modes catalogued at scale-up.

- **Paper (June 2026 numbering):** §17.9.2, Table 29

---

## V_theta regularisation sweeps (§17b)

### SPLM V_theta regularisation (VR0–VR5)

<a name="vr"></a>**VR0–VR5** — Book: Table 33 (`tab:vreg-cross-arch`)

| Code | λ_V | Description |
|------|-----|-------------|
| **VR0** | 0 | Unregularised |
| **VR1** | 1e-6 | |
| **VR2** | 1e-4 | |
| **VR3** | 1e-2 | |
| **VR4** | 1 | |
| **VR5** | — | Verlet integrator variant (L=16, dt=0.5) |

- **Notebook:** [`notebooks/conservative_arch/parf/scripts/vreg_sweep_v_theta_regularisation.ipynb`](notebooks/conservative_arch/parf/scripts/vreg_sweep_v_theta_regularisation.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/vreg_sweep/VR0/`](notebooks/conservative_arch/parf/results/vreg_sweep/) through `VR5/`

### PARFLM V_theta regularisation (PR0–PR4)

<a name="pr"></a>**PR0–PR4** — Book: Table 33 (`tab:vreg-cross-arch`)

| Code | λ_V | Best PPL |
|------|-----|----------|
| **PR0** | 0 | 246.4 |
| **PR1** | 1e-6 | 191 |
| **PR2** | 1e-4 | **186.0** |
| **PR3** | 1e-2 | |
| **PR4** | 1 | |

- **Notebook:** [`notebooks/conservative_arch/parf/scripts/vreg_sweep_parf.ipynb`](notebooks/conservative_arch/parf/scripts/vreg_sweep_parf.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/vreg_sweep_parf/PR0/`](notebooks/conservative_arch/parf/results/vreg_sweep_parf/) through `PR4/`
- **Key finding:** PR2 (λ_V=10⁻⁴) achieves the largest PPL improvement of any conservative architecture: −60.4 PPL vs PR0. GD attractor extraction converges at 99.9% (1919/1920 prompts). V_theta range collapses from 59 (PR0) to 20 (PR2). This is the canonical gauge-breaking recipe.

### FockPARF V_theta regularisation (FR0–FR4)

<a name="fr"></a>**FR0–FR4** — Book: Table 33 (`tab:vreg-cross-arch`)

| Code | λ_V |
|------|-----|
| **FR0** | 0 |
| **FR1** | 1e-6 |
| **FR2** | 1e-4 |
| **FR3** | 1e-2 |
| **FR4** | 1 |

- **Notebook:** [`notebooks/conservative_arch/parf/scripts/vreg_sweep_fockparf.ipynb`](notebooks/conservative_arch/parf/scripts/vreg_sweep_fockparf.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/vreg_sweep_fockparf/FR0/`](notebooks/conservative_arch/parf/results/vreg_sweep_fockparf/) through `FR4/`
- **Key finding:** FockPARF is most tolerant of V_theta regularisation via V_phi and register lifecycle (M=16 registers, d=128, L=8).

### Hybrid V_theta regularisation (HR0–HR4)

<a name="hr"></a>**HR0–HR4** — Book: Table 33 (`tab:vreg-cross-arch`)

| Code | λ_V |
|------|-----|
| **HR0** | 0 |
| **HR1** | 1e-6 |
| **HR2** | 1e-4 |
| **HR3** | 1e-2 |
| **HR4** | 1 |

- **Notebook:** [`notebooks/conservative_arch/hybrid/scripts/vreg_sweep_hybrid.ipynb`](notebooks/conservative_arch/hybrid/scripts/vreg_sweep_hybrid.ipynb)
- **Results:** [`notebooks/conservative_arch/hybrid/results/vreg_sweep_hybrid/HR0/`](notebooks/conservative_arch/hybrid/results/vreg_sweep_hybrid/) through `HR4/`
- **Key finding:** Hybrid PPL stays within 0.7–1.6 PPL of unregularised HR0 (140.4) at all λ_V — the attention front-end pre-normalises the state, making V_theta regularisation effectively free.

### Planned confound-resolution sweeps (G1–G3)

<a name="g1"></a>**G1** · <a name="g2"></a>**G2** · <a name="g3"></a>**G3** — Book: Table 33 (`tab:vreg-cross-arch`)

| Code | Description |
|------|-------------|
| **G1** | PARF γ sweep at λ_V=10⁻⁴, d_V=128, γ∈{0.05, 0.10, 0.15} |
| **G2** | PARF d_V sweep at λ_V=10⁻⁴, γ=0.10, d_V∈{128, 256, 512} |
| **G3** | SPLM em_ln at d_V=512, γ=0.10, λ_V=10⁻⁴ |

- **Notebook:** [`notebooks/conservative_arch/parf/scripts/fockparf_improvement_sweep.ipynb`](notebooks/conservative_arch/parf/scripts/fockparf_improvement_sweep.ipynb) (G1–G3 cells)
- **Results:** planned — no results yet
- **Status:** planned

---

## Structured V_theta (§17)

### Structured V_theta sweep (SQ1–SQ5)

<a name="sq1"></a>**SQ1** · <a name="sq2"></a>**SQ2** · <a name="sq3"></a>**SQ3** · <a name="sq4"></a>**SQ4** · <a name="sq5"></a>**SQ5** — Book: Table 26 (`tab:structured-vtheta-summary`)

Tests whether structured V_theta (analytical gradient, fewer params)
can match the PR2 MLP baseline on TinyShakespeare.

| Code | V_theta variant | Description |
|------|----------------|-------------|
| **SQ1** | Diagonal quadratic | Cheapest; axis-aligned wells |
| **SQ2** | Low-rank quadratic (rank=8) | Rotated wells |
| **SQ3** | Mixture K=4 | K Gaussian-well basins |
| **SQ4** | Quadratic + small MLP residual | Hybrid interpretable + learned |
| **SQ5** | MLP reference | Full MLP rerun |

- **Notebook:** [`notebooks/conservative_arch/parf/scripts/structured_vtheta_sweep.ipynb`](notebooks/conservative_arch/parf/scripts/structured_vtheta_sweep.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/structured_vtheta/SQ3/`](notebooks/conservative_arch/parf/results/structured_vtheta/) through `SQ5/`
- **Key finding:** SQ3 (mixture K=4) matches the PR2 MLP baseline on PPL with analytical ∇_h V_theta, zero autograd cost, and interpretable basins.

### Structured V_theta TinyStories sweep (A1–A10, B1–B2)

<a name="a1a10"></a>**A1–A10** · <a name="b1b2"></a>**B1–B2** — Book: Table 26 (`tab:structured-vtheta-summary`)

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

- **Notebook:** [`notebooks/conservative_arch/parf/scripts/structured_vtheta_tinystories_sweep.ipynb`](notebooks/conservative_arch/parf/scripts/structured_vtheta_tinystories_sweep.ipynb)
- **Key finding:** SQ3 matches the MLP reference at TinyStories scale. SQ5 (MLP rerun) diverges due to large initial ‖V_theta‖² destabilising next-token loss.


### Multi-Xi SPLM structured V_theta (A1–A5)

<a name="splmsvt"></a>**SPLM-SVT A1–A5** — Book: §22.4.2 (`sssec:ssp-results-multixi-splm`)

Five-arm SQ3 hyperparameter sweep on the Multi-Xi SPLM, replacing the MLP V_theta
with `StructuredVThetaMultiXiAdapter(MixtureQuadraticVTheta(...))`. Arms vary K_mix
(4 or 8), tau (1.0 or 0.5), and K_xi (4 or 8), all with lambda_V=0.01. Run at the
standard scaleup config (d=256, L=8, 16k steps, TinyStories 5M tokens).

| Arm | K_mix | tau | K_xi | PPL | V_theta params |
|-----|-------|-----|------|-----|----------------|
| A1 | 4 | 1.0 | 4 | 14.10 | 2.1M |
| **A2** | **8** | **1.0** | **4** | **13.33** | **4.2M** |
| A3 | 4 | 0.5 | 4 | 14.10 | 2.1M |
| A4 | 4 | 1.0 | 8 | 14.16 | 4.2M |
| A5 | 8 | 1.0 | 8 | 14.25 | 8.4M |
| B1 (MLP ref.) | — | — | 4 | 11.51 | — |

- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_splm_multixi_structured_vtheta.ipynb`](notebooks/conservative_arch/scaleup/colab_splm_multixi_structured_vtheta.ipynb)
- **Model file:** [`notebooks/conservative_arch/parf/model_structured_vtheta_multixi.py`](notebooks/conservative_arch/parf/model_structured_vtheta_multixi.py)
- **Results:** GDrive `semsimula_splm_multixi_structured_vtheta/`
- **HuggingFace (A2):** [`dimitarpg13/semsimula-splm-multixi-structured-vtheta`](https://huggingface.co/dimitarpg13/semsimula-splm-multixi-structured-vtheta)
- **Key finding:** A2 (K_mix=8, K_xi=4) achieves **13.33 PPL** — a 1.82 PPL expressivity gap vs the MLP reference (11.51). Six of eight attractor basins are active at convergence. V_theta landscape: mean 0.31, range 47.2. Learned xi-channel alphas: [0.25, 0.60, 0.81, 0.96].

### Multi-Xi PARFLM structured V_theta (A2)

<a name="parfsvt"></a>**PARF-SVT A2** — Book: §22.4.3 (`sssec:ssp-results-multixi-parf`)

Single-arm structured V_theta experiment on the Multi-Xi PARFLM using the SPLM
winner configuration (A2: SQ3, K_mix=8, tau=1.0, K_xi=4, lambda_V=0.01). Reveals
V_theta landscape compression: the pairwise V_phi potential absorbs the bulk of the
nonlinear force budget, leaving V_theta operating as a near-flat bias field and
drastically shrinking the structured-vs-MLP PPL gap.

| Arm | K_mix | K_xi | PPL | PPL gap vs MLP |
|-----|-------|------|-----|----------------|
| **A2** | **8** | **4** | **12.27** | **0.17 PPL (0.6%)** |
| MLP ref. | — | 4 | 12.10 | — |

- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_parf_multixi_structured_vtheta.ipynb`](notebooks/conservative_arch/scaleup/colab_parf_multixi_structured_vtheta.ipynb)
- **Model file:** [`notebooks/conservative_arch/parf/model_structured_vtheta_multixi.py`](notebooks/conservative_arch/parf/model_structured_vtheta_multixi.py)
- **Results:** GDrive `semsimula_parf_multixi_structured_vtheta/`
- **HuggingFace (A2):** [`dimitarpg13/semsimula-parflm-multixi-structured-vtheta`](https://huggingface.co/dimitarpg13/semsimula-parflm-multixi-structured-vtheta)
- **Key finding:** V_phi in PARFLM compresses the V_theta landscape (mean 0.02, range 31.4 vs 47.2 in SPLM), reducing the expressivity gap from 1.82 PPL (SPLM) to just 0.17 PPL (0.6%). Learned xi-channel alphas: [0.24, 0.49, 0.74, 0.95].

### Fock-PARFLM structured V_theta (A2)

<a name="focksvt"></a>**Fock-SVT A2** — Book: §22.4.4 (`sssec:ssp-results-multixi-fock`)

Structured V_theta experiment on Fock-PARFLM v2.1 (`FockMultiXiPARFLM`, d=256, L=8,
M=16 registers, LIFO, reverse channel, per-register learnable temperature, per-register
key subspaces, orthogonal init). Uses the A2 arm (SQ3, K_mix=8, K_xi=4, lambda_V=0.01)
from the same hyperparameter sweep as SPLM and PARFLM experiments.

| Arm | K_mix | K_xi | PPL | PPL gap vs MLP |
|-----|-------|------|-----|----------------|
| **A2** | **8** | **4** | **10.36** | **1.06 PPL (11.4%)** |
| MLP ref. (B1) | — | 4 | 9.30 | — |

- **Notebook:** [`notebooks/conservative_arch/scaleup/colab_fock_multixi_structured_vtheta.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_multixi_structured_vtheta.ipynb)
- **Model file:** [`notebooks/conservative_arch/parf/model_structured_vtheta_multixi.py`](notebooks/conservative_arch/parf/model_structured_vtheta_multixi.py)
- **Results:** GDrive `semsimula_fock_multixi_structured_vtheta/`
- **HuggingFace (A2):** [`dimitarpg13/semsimula-fock-parflm-structured-vtheta`](https://huggingface.co/dimitarpg13/semsimula-fock-parflm-structured-vtheta)
- **Key finding:** The "Fock precision paradox" — despite the most compressed V_theta landscape of all three architectures (mean 0.008, range 19.1), the structured-vs-MLP PPL gap is *larger* than PARFLM's (1.06 PPL vs 0.17 PPL). Attributed to the Fock model operating closer to the TinyStories entropy floor, where residual V_theta imprecision has amplified marginal impact. Learned xi-channel alphas: [0.24, 0.51, 0.76, 0.97].
---

## FockPARF v2 / QFT (§17c)

### FockPARF v2 debug (D1–D9)

<a name="d1d9"></a>**D1–D9** — Book: Table 98 (`tab:expidx-fock`)

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

- **Notebook:** [`notebooks/conservative_arch/parf/scripts/fockparf_v2_tinystories_debug.ipynb`](notebooks/conservative_arch/parf/scripts/fockparf_v2_tinystories_debug.ipynb)

### QFT v2.1 experimental ladder (Q0–Q8)

<a name="q0"></a>**Q0** · <a name="q1fock"></a>**Q1** · <a name="q2fock"></a>**Q2** · <a name="q3fock"></a>**Q3** · <a name="q4fock"></a>**Q4** · <a name="q5fock"></a>**Q5** · <a name="q6fock"></a>**Q6** · <a name="q7fock"></a>**Q7** · <a name="q8fock"></a>**Q8** — Book: Table 98 (`tab:expidx-fock`)

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

- **Notebook:** [`notebooks/conservative_arch/parf/scripts/fockparf_v2_qft_improvements.ipynb`](notebooks/conservative_arch/parf/scripts/fockparf_v2_qft_improvements.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/fock_v2_qft/Q0/`](notebooks/conservative_arch/parf/results/fock_v2_qft/) through `Q8/`
- **Key finding:** Q6 (full QFT stack + learnable temperature) achieves 53.47 PPL — 9.1% below FockPARF v2 baseline and 5.2% below plain PARFLM. The four QFT improvements are not optional refinements — they are what makes the register mechanism beneficial. Q7 (M=32) diverges: diversity collapses at layers 5–7.

### QFT v2.1 planned series (D6–D10)

<a name="d6d10"></a>**D6–D10** — Book: Table 98 (`tab:expidx-fock`)

Planned incremental tests isolating each QFT-motivated change.

| Code | Description |
|------|-------------|
| **D6** | Learnable τ only (baseline) |
| **D7** | + Gumbel-Softmax |
| **D8** | + per-register keys |
| **D9** | + orthogonal initialisation |
| **D10** | + canonical destruction |

- **Status:** planned — no notebook yet

### Dyck-2 falsifier (F2)

<a name="f2baseline"></a>**F2-baseline** · <a name="f2fockv2"></a>**F2-fock-v2** — Book: Table 98 (`tab:expidx-fock`)

Three-arm controlled comparison on synthetic bracket-matching.

| Code | Description | Best loss | Deep-test acc |
|------|-------------|-----------|---------------|
| **F2-baseline** | PARFLM (no registers) | 3.078 | 43.64% |
| **F2-fock-v1** | Mean-conditioned creation | 3.037 | 45.12% |
| **F2-fock-v2** | Q/K/V creation + gated reverse | **2.856** | **49.01%** |

- **Notebook:** [`notebooks/conservative_arch/parf/scripts/fockparf_v2_dyck2_falsifier.ipynb`](notebooks/conservative_arch/parf/scripts/fockparf_v2_dyck2_falsifier.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/fock_v2/F2-baseline_seed0/`](notebooks/conservative_arch/parf/results/fock_v2/) through `F2-fock-v2_seed0/`
- **Key finding:** FockPARF v2 achieves the best Dyck-2 performance at 49.01% deep-test accuracy, approaching the 50% success criterion.

### FockPARF improvement sweep (P1–P5 Fock, G1–G3)

<a name="p1p5fock"></a>**P1–P5** — Book: Table 98 (`tab:expidx-fock`)

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

- **Notebook:** [`notebooks/conservative_arch/parf/scripts/fockparf_improvement_sweep.ipynb`](notebooks/conservative_arch/parf/scripts/fockparf_improvement_sweep.ipynb)
- **Results:** [`notebooks/conservative_arch/parf/results/fockparf_improvement/P1/`](notebooks/conservative_arch/parf/results/fockparf_improvement/) through `P5/`

---

## Architecture comparison

### Architecture comparison (S1–S4)

<a name="s1s4"></a>**S1–S4** — Book: Table 99 (`tab:expidx-nonconservative`)

TinyStories v3 scale-up: replaces MLP V_theta with structured SQ3
(K=4) across four architectures at d=256.

| Code | Architecture |
|------|-------------|
| **S1** | Regularised PARF |
| **S2** | Regularised FockPARF (M=32 registers) |
| **S3** | Hybrid FockPARF + Attention |
| **S4** | Hybrid SPLM + Attention reference |

- **Notebook:** [`notebooks/conservative_arch/parf/scripts/tinystories_parf_vs_fockparf.ipynb`](notebooks/conservative_arch/parf/scripts/tinystories_parf_vs_fockparf.ipynb) (also at [`notebooks/conservative_arch/tinystories_parf_vs_fockparf.ipynb`](notebooks/conservative_arch/tinystories_parf_vs_fockparf.ipynb))
- **Results:** [`notebooks/conservative_arch/parf/results/tinystories_scaleup/`](notebooks/conservative_arch/parf/results/tinystories_scaleup/) (S1, S2), [`notebooks/conservative_arch/parf/results/tinystories_v2/`](notebooks/conservative_arch/parf/results/tinystories_v2/) (S2, S3)

### Architecture comparison Phase 2 (L1–L3)

<a name="l1l3"></a>**L1–L3** — Book: Table 99 (`tab:expidx-nonconservative`)

TinyStories Phase 2 large-scale comparison.

| Code | Architecture |
|------|-------------|
| **L1** | Large PARF |
| **L2** | Large FockPARF |
| **L3** | Large Hybrid |

- **Notebook:** [`notebooks/conservative_arch/parf/scripts/tinystories_parf_phase2_large.ipynb`](notebooks/conservative_arch/parf/scripts/tinystories_parf_phase2_large.ipynb)
- **Results:** not yet available

---

## Non-conservative / SP-HSPLM

### SP-HSPLM Stage 1 (e0–e5)

<a name="e0e5"></a>**e0–e5** — Book: Table 99 (`tab:expidx-nonconservative`)

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

- **Notebook:** [`notebooks/conservative_arch/non_conservative/scripts/sp_hsplm_stage1_a100_h100.ipynb`](notebooks/conservative_arch/non_conservative/scripts/sp_hsplm_stage1_a100_h100.ipynb)
- **Results:** [`notebooks/conservative_arch/non_conservative/results/sp_hsplm/stage1/`](notebooks/conservative_arch/non_conservative/results/sp_hsplm/stage1/)

### SP-HSPLM Stage 2 (q9e_a–q9e_n)

<a name="q9e"></a>**q9e** — Book: Table 99 (`tab:expidx-nonconservative`)

Q9(e) pair-skew cell ladder covering mechanism-2 autonomous force law,
mechanism-1 extensions, additivity tests, and full Class F.

| Code | Description |
|------|-------------|
| **q9e_a** | Mechanism-2 autonomous |
| **q9e_b–q9e_n** | Progressive additions through full Class F |

- **Notebook:** [`notebooks/conservative_arch/sphsplm/scripts/sp_hsplm_stage2_a100_h100.ipynb`](notebooks/conservative_arch/sphsplm/scripts/sp_hsplm_stage2_a100_h100.ipynb)
- **Results:** [`notebooks/conservative_arch/sphsplm/results/sp_hsplm/stage2/`](notebooks/conservative_arch/sphsplm/results/sp_hsplm/stage2/)

---

## Inference efficiency (Appendix A2)

### Inference efficiency benchmark (E8)

<a name="e8"></a>**E8** — Book: Table 86 (`tab:e8-phase2-verdict`)

Pre-registered three-phase inference benchmark.

| Phase | Description | Grade |
|-------|-------------|-------|
| **Phase 1** | Matched-compute quality | Q3 (matched-attn beats SPLM by ~18–31 PPL) |
| **Phase 2** | Per-token wall-clock and FLOP | C (2 CONFIRMED, 2 MARGINAL, 2 REFUTED) |
| **Phase 3** | Quality-adjusted Pareto | Attention dominates at prototype scale |

- **Results:** [`notebooks/conservative_arch/inference_efficiency/results/`](notebooks/conservative_arch/inference_efficiency/results/)
- **Key finding:** SPLM achieves ~38% FLOP advantage at $T\geq 512$ but 2.5–3× wall-clock overhead due to `create_graph=True`. Phase 2 structured V_theta (SQ3) path is expected to recover 30–40% of the wall-clock overhead.

---

## Formal-language falsifiers (§9)

### F1–F6 falsifier programme

<a name="f1"></a>**F1** · <a name="f2"></a>**F2** · <a name="f3"></a>**F3** · <a name="f4"></a>**F4** · <a name="f5"></a>**F5** · <a name="f6"></a>**F6** — Book: §10.8 (`subsec:falsifiers`)

Six experiments testing the v0 lower-bound staircase and MCS bounds.

| Code | Target language | Tests |
|------|----------------|-------|
| **F1** | Dyck_n (base) | Falsifies v0 alone; motivates v2. Predicted collapse depth D* |
| **F2** | Dyck_n + topic-shift | Falsifies v0+v2 without decay; motivates v1.5 |
| **F3** | Dyck_n + let-binding | Falsifies v0+v2+v1.5 without operators; motivates v3 |
| **F4** | Cross-serial a^n b^n c^n | Confirms v0+v2+v1.5+v3 reaches MCS |
| **F5** | Bounded copy ww | Confirms MCS reach via different construction |
| **F6** | 2-counter machine simulation | Falsifies the upper bound (intentional over-strong) |

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

- **Paper (June 2026 numbering):** §17.12 (scaling discussion)
- **Design docs:** [`companion_notes/Gradient_Checkpointing_for_PARF.md`](companion_notes/Gradient_Checkpointing_for_PARF.md), [`companion_notes/PARF_Stage_1_5b_design.md`](companion_notes/PARF_Stage_1_5b_design.md)
- **Key finding:** Combined Level 2 checkpointing + Stage-1.5b reduces peak V_phi activation memory from O(L·B·T²·H) to O(B·T·k·H) — constant in L, linear in T. At B=16, T=512, L=8, k=4: ~10–14 GB → ~8 MB (~1000× reduction).

---

## Gaussian V_θ on Fock-PARFLM, the OpenWebText scale-up, and the Riemannian diagnostics (book Appendix D)

<a name="gaussian-scaleup"></a>Models and diagnostics of August 2026 that the book's Appendix D lists
without an earlier entry here. Where a notebook writes to Google Drive,
the run's artefacts are on its Hugging Face card, not in the repository.

| Code | What it is | Status | Notebook, cell or script | Results | Checkpoint | Written up in |
|---|---|---|---|---|---|---|
| <a name="fockgaussaniso"></a>**Fock-Gauss-Aniso** | Bounded anisotropic-Gaussian V_θ (diagonal plus rank-4 precision) with the Fock-coupling regulariser on Fock-PARFLM v2.1; TinyStories, d = 256, L = 8, γ = 0.30, 20,000 steps, leak-free; PPL 9.04 (`Fock_Mechanism_Engagement…` §8.2 reports 9.14 for its Experiment D) — Book: §20.7 (`ssec:fock-multixi-results`) | completed (2026-08-06) | [`colab_fock_aniso_gaussian_fockreg_tinystories.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_aniso_gaussian_fockreg_tinystories.ipynb); γ sweep [`colab_fock_gamma_sweep_geodesic_aniso_gaussian_fockreg_tinystories.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_gamma_sweep_geodesic_aniso_gaussian_fockreg_tinystories.ipynb) | on the card (the notebook writes to Drive) | [card](https://huggingface.co/dimitarpg13/semsimula-fock-parflm-anisogaussian-vtheta) | [γ note §11](companion_notes/Determining_optimal_gamma_for_Fock-PARFLM.md#n-fockgaussaniso) |
| <a name="fockg1"></a>**Fock-G1** | First-order (gradient-flow) ablation of Fock-Gauss-Aniso: same architecture, data and γ; PPL 8.95 against 9.04, one seed — Book: §20.7 (`ssec:fock-multixi-results`) | completed (2026-08-16) | [`colab_fock_g1_aniso_gaussian_fockreg_tinystories.ipynb`](notebooks/conservative_arch/first_order_ablation/colab_fock_g1_aniso_gaussian_fockreg_tinystories.ipynb) | on the card (`fock_g1_vs_second_order_report.json`, `training_log.jsonl`) | [card](https://huggingface.co/dimitarpg13/semsimula-fock-parflm-anisogaussian-vtheta-fock-g1) | [first- vs second-order note §6](companion_notes/Implicit_vs_Explicit_Damping_and_the_First_vs_Second_Order_Dynamics_Hypothesis.md#n-fockg1) |
| <a name="fockgaussiso"></a>**Fock-Gauss-Iso** | Isotropic, depth-conditioned multi-context Gaussian V_θ on Fock-PARFLM v2.1; TinyStories, d = 256, leak-free; PPL 16.33, the reference for the anisotropy premium — Book: §22.7.9 (`sssec:ssp-gw-status`) | completed | [`colab_fock_gaussian_vtheta_tinystories.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_gaussian_vtheta_tinystories.ipynb) | on the card | [card](https://huggingface.co/dimitarpg13/semsimula-fock-parflm-depthcond-vtheta) | [Fock-engagement note §4.2](companion_notes/Fock_Mechanism_Engagement_MLP_vs_Gaussian_VTheta.md#n-fockgaussiso) |
| <a name="rd"></a>**RD1–5** | Riemannian diagnostic battery on Multi-ξ SPLM, Fock v2.1 and Fock-Attention checkpoints: metric validity, geodesic compliance (damped and undamped), curvature proxy, energy dissipation, conservativity separator — Book: §27.12 (`subsec:diagnostic-battery`) | completed (June 2026) | [`colab_riemannian_diagnostic.ipynb`](notebooks/conservative_arch/scaleup/colab_riemannian_diagnostic.ipynb) | [damped](notebooks/conservative_arch/scaleup/results/semsimula_riemannian_diagnostic/), [undamped](notebooks/conservative_arch/scaleup/results/semsimula_riemannian_diagnostic_undamped/) | evaluation only | [Riemannian note §3.5](companion_notes/Exploiting_the_Riemannian_geometry_of_conservative_language_models.md#n-rd) |
| <a name="rdgamma"></a>**RD-γ** | Learned-γ diagnostic: Arm 2 re-run with the effective γ on six checkpoints; resolves the negative-R² artefact — Book: §27.12 (`par:learned-gamma-diagnostic`) | completed | [`colab_gamma_diagnostic.ipynb`](notebooks/conservative_arch/scaleup/colab_gamma_diagnostic.ipynb) | [`semsimula_gamma_diagnostic`](notebooks/conservative_arch/scaleup/results/semsimula_gamma_diagnostic/) | evaluation only | [Riemannian note, learned-γ diagnostic](companion_notes/Exploiting_the_Riemannian_geometry_of_conservative_language_models.md#n-rdgamma) |
| <a name="su-iso"></a>**SU-iso** | OpenWebText scale-up pilot, isotropic depth-conditioned Gaussian V_θ, d = 384, L = 16 (tag `e5c_plgate`): 27.23 PPL, apparent (L3 leak); reverse-channel-off ablation `e5a` 125.94 — Book: §23.7 (`subsec:su-payoff`) | retired (L3 leak) | [`colab_fock_depthcond_vtheta_openwebtext.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_depthcond_vtheta_openwebtext.ipynb), [`colab_fock_depthcond_vtheta_openwebtext_ext.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_depthcond_vtheta_openwebtext_ext.ipynb) | [training logs](companion_notes/data/d384_e5c_plgate/) | [card](https://huggingface.co/dimitarpg13/semsimula-fock-parflm-depthcond-vtheta-openwebtext) (leak banner) | [d384 results note §1](companion_notes/Fock-PARFLM_d384_Training_Results_OpenWebText.md#n-su-iso) |
| <a name="su-aniso"></a>**SU-aniso** | OpenWebText anisotropic-Gaussian V_θ with the Fock regulariser: damping sweeps (8 values of γ, 3,000 steps), leak-free; γ* = 0.10 at d = 384 (184.11), 0.05 at d = 768 (326.97) and d = 1024 (244.23). The d = 384 full run reached its best checkpoint, 184.11, at step 10,000 and stalled at step 17,100 of 100,000 (card) — Book: §23.9 (`subsec:su-damping`) | completed | [d384](notebooks/conservative_arch/scaleup/colab_fock_gamma_sweep_geodesic_aniso_gaussian_fockreg_d384.ipynb), [d768](notebooks/conservative_arch/scaleup/colab_fock_gamma_sweep_geodesic_aniso_gaussian_fockreg_d768.ipynb), [d1024](notebooks/conservative_arch/scaleup/colab_fock_gamma_sweep_geodesic_aniso_gaussian_fockreg_d1024.ipynb) sweeps; full run [`colab_fock_aniso_gaussian_fockreg_openwebtext.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_aniso_gaussian_fockreg_openwebtext.ipynb) | on the cards | [d384](https://huggingface.co/dimitarpg13/semsimula-fock-parflm-anisogaussian-vtheta-owt-d384-gammasweep), [d768](https://huggingface.co/dimitarpg13/semsimula-fock-parflm-anisogaussian-vtheta-owt-d768-gammasweep), [d1024](https://huggingface.co/dimitarpg13/semsimula-fock-parflm-anisogaussian-vtheta-owt-d1024-gammasweep), [d384 full run](https://huggingface.co/dimitarpg13/semsimula-fock-parflm-anisogaussian-vtheta-owt-d384-verlet-instability) | [γ note §12–14](companion_notes/Determining_optimal_gamma_for_Fock-PARFLM.md#n-su-aniso) |
| <a name="scaf-stiff"></a>**SCAF-stiff** | SCAF `StiffnessProbe` envelope audit: the off-diagonal σ_max(B_k)² is unbounded and pushes ω·Δt past the Verlet ceiling of 2; motivates CfC/BAOAB — Book: Table 40 (`tab:su-envelope`) | completed (L = 8 bracket; L = 16 not started) | [`scaf_checkpoint_analysis_stiffness_probe.ipynb`](notebooks/conservative_arch/scaleup/debug/scaf_checkpoint_analysis_stiffness_probe.ipynb) (Phase 7), [`scaf_checkpoint_analysis.ipynb`](notebooks/conservative_arch/scaleup/debug/scaf_checkpoint_analysis.ipynb) (7b/7c); probe code in the separate `semsimula-scaf` repository | [L = 8 bracket](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_L8probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc/) | evaluation only | [CfC/BAOAB note §28, §31](companion_notes/CfC_BAOAB_Integrator_and_Mitigations.md#n-scaf-stiff) |

---

## The CfC/BAOAB depth ladder on OpenWebText (book §D.10)

<a name="ladder"></a>The models of the book's CfC/BAOAB programme: d = 384,
532,480,000 training tokens per arm (32,500 steps of 32 × 512), the
CfC/BAOAB integrator, a joint V_θ bank. **Gen 2** arms were trained with
detached V_φ and ξ sources, **Gen 3** arms with live source gradients and
the same forward pass. Registered and scored in the protocol, §5.

- **Book:** Table 101 (`tab:expidx-ladder`) lists the arms; they are
  discussed from §27 on.
- **Notebook (every CfC/BAOAB arm):**
  [`colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb);
  the GPT-2 references use
  [`colab_matched_gpt2_baseline_openwebtext.ipynb`](notebooks/conservative_arch/scaleup/colab_matched_gpt2_baseline_openwebtext.ipynb). The full Cell 0
  of each published arm is printed on its Hugging Face card.
- **Settled PPL** is the mean of the last three evaluations.
- **SR2, PMX** and the first- versus second-order twins are in the next
  section ([SR2](#sr2), [PMX](#pmx)).

### Gen 3: live source gradients

| Arm | What it is: Cell 0 against its base | Settled PPL | Run folder and training log | Checkpoint | Scored in |
|---|---|---:|---|---|---|
| <a name="f3-1"></a>**F3.1** | **Conservative-only**, no Fock mechanism (`LADDER_MECHANISM = 'none'`, `REVERSE_CHANNEL = False`), with live V_φ and ξ source gradients (`VPHI_GRAD_PATH = XI_GRAD_PATH = 'live'`); the base of SR2, PM1, PMX and XRB-F | 57.76 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_idt4_lr0p0012_norc_vplive_xilive_noattn_32500_result.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-none-norc) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-f3-1) |
| <a name="g2-ladder"></a>**G2** | F3.1 with the **Fock mechanism** on (register path, `REVERSE_CHANNEL = True`), no exchange field | 53.12 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_vplive_xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_vplive_xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_arm_none_live_grads_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-none) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-g2-ladder) |
| <a name="l4-live"></a>**L = 4 live** | G2 at **depth 4** (`LADDER_L = 4`, Δt = 2, T = 8) | 50.10 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_vplive_xilive_L4probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt2_lr0p0012_noattn/), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_vplive_xilive_L4probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt2_lr0p0012_noattn/L4_idt2_lr0p0012_vplive_xilive_noattn_32500_result.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l4-none) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-l4-live) |
| <a name="g3-ladder"></a>**G3** | G2 plus the conservative **exchange field** (`LADDER_MECHANISM = 'attention_potential'`, `RELAX_GRAD_PATH = 'live'`): everything live | 54.21 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_rglive_vplive_xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot/), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_rglive_vplive_xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot/L2_arm_attnpot_live_grads_32500steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-attention-potential) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-g3-ladder) |
| <a name="g3p"></a>**G3′** | G3 with **hardened routing**: QK-normalised exchange-field routing (`RELAX_ATTN_QK_NORM = True`) and its own clip group (`RELAX_FIELD_CLIP = 0.3`) | 52.90 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_rglive_vplive_xilive_rfqk_rfclip0p3_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot/), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_rglive_vplive_xilive_rfqk_rfclip0p3_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot/L2_arm_attnpot_rfqk_32500steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-attention-potential-rfqk) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-g3p) |
| <a name="xrb-f"></a>**XRB-F** | F3.1 plus G3′'s exchange field, **no register read** (`REVERSE_CHANNEL = False`) | 54.27 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_rglive_vplive_xilive_rfqk_rfclip0p3_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot/), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_rglive_vplive_xilive_rfqk_rfclip0p3_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot/L2_attnpot_norc_F3.1prim_32500steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-attention-potential-rfqk-norc) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-xrb-f) |
| <a name="pm1"></a>**PM1** | F3.1 plus 64 **Poisson-mode registers** with explicit wells (`POISSON_MODES = 64`, own clip group at 0.3); probe to 8,000 steps, then resumed to 32,500 | 55.17 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/), [log 1](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2probe_arm_none_pm64_clip0.3_8000steps_output.txt), [log 2](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_arm_none_pm64_clip0.3_32500steps_output.txt) | not published | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-pm1) |

### Gen 2: detached sources

| Arm | What it is: Cell 0 against its base | Settled PPL | Run folder and training log | Checkpoint | Scored in |
|---|---|---:|---|---|---|
| <a name="gen2-attention"></a>**attention (Gen 2)** | Non-conservative **exchange field** (`LADDER_MECHANISM = 'attention'`), detached sources | 63.51 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attn/), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attn/L2_idt4_lr0p0012_attn_altE_fromscratch_32500_result.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-owt-d384-l2-attention) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-gen2-attention) |
| <a name="gen2-none"></a>**no-exchange (Gen 2)** | **Fock mechanism** on, no exchange field (`LADDER_MECHANISM = 'none'`), detached sources | 66.98 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_idt4_lr0p0012_noattn_altE_fromscratch_32500_result.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-owt-d384-l2-none) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-gen2-none) |
| <a name="gen2-attnpot"></a>**attention_potential (Gen 2)** | Conservative **exchange field** (`LADDER_MECHANISM = 'attention_potential'`), detached sources | 80.90 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot/), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot/L2_idt4_lr0p0012_attnpot_altE_fromscratch_32500_result.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-owt-d384-l2-attention-potential) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-gen2-attnpot) |
| <a name="gen2-none-norc"></a>**conservative-only (Gen 2)** | No-exchange without the reverse channel (`REVERSE_CHANNEL = False`), detached sources | 87.93 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_idt4_lr0p0012_norc_noattn_altE_fromscratch_32500_result.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-owt-d384-l2-none-norc) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-gen2-none-norc) |
| <a name="gen2-none-l4"></a>**no-exchange L = 4 (Gen 2)** | No-exchange at depth 4 (`LADDER_L = 4`, Δt = 2), detached sources (run 4) | 71.75 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L4probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt2_lr0p0012_noattn/), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L4probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt2_lr0p0012_noattn/L4_idt2_lr0p0012_noattn_altE_fromscratch_32500_result.txt) | not published | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-gen2-none-l4) |
| <a name="gen2-none-l1"></a>**no-exchange L = 1 (Gen 2)** | No-exchange at depth 1 (`LADDER_L = 1`, Δt = 8), detached sources (run 7) | 87.09 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L1probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt8_lr0p0012_noattn/), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L1probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt8_lr0p0012_noattn/L1_idt8_lr0p0012_noattn_altE_fromscratch_32500_result.txt) | not published | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-gen2-none-l1) |
| <a name="gen2-attnpot-rglive"></a>**attention_potential, live field (probe)** | The gradient-path probe: attention_potential with a live exchange-field gradient only (`RELAX_GRAD_PATH = 'live'`), extended from 3,000 to 32,500 steps | 61.11 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_rglive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot/), [log 1](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_rglive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot/L2_idt4_lr0p0012_attnpot_rglive_probe3000_result.txt), [log 2](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_rglive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot/L2_attnpot_live_grad_exchange_field_32500steps.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-owt-d384-l2-attention-potential-rglive) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-gen2-attnpot-rglive) |

### Matched GPT-2 references

| Arm | What it is: Cell 0 against its base | Settled PPL | Run folder and training log | Checkpoint | Scored in |
|---|---|---:|---|---|---|
| <a name="gpt2-matched"></a>**matched GPT-2** | The **reference**: d = 384, L = 8, 33.7M parameters, cosine schedule (peak 6e-4), batch 32 × 512, 32,500 steps | 49.81 | [folder](notebooks/conservative_arch/scaleup/results/gpt2_baseline_d384_L8/), [log](notebooks/conservative_arch/scaleup/results/gpt2_baseline_d384_L8/matched_batch32_32500_result.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-owt-d384-l2-gpt2-matched) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-gpt2-matched) |
| <a name="w1"></a>**W1** | The same GPT-2 on the ladder's WSD schedule (`LR_SCHEDULE = 'wsd'`), peak 6e-4 | 48.82 | [folder](notebooks/conservative_arch/scaleup/results/gpt2_baseline_wsd/), [log](notebooks/conservative_arch/scaleup/results/gpt2_baseline_wsd/gpt2_matched_baseline_wsd_schedule_output.txt) | not published | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-w1) |
| <a name="w2"></a>**W2** | W1 at the ladder's peak learning rate (`LR = 1.2e-3`) | 48.37 | [folder](notebooks/conservative_arch/scaleup/results/gpt2_baseline_wsd_lr0p0012/), [log](notebooks/conservative_arch/scaleup/results/gpt2_baseline_wsd_lr0p0012/W2_matched_gpt2_wsd_lr1.2e-3_32500steps_training_output.txt) | not published | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-w2) |
| <a name="g2b"></a>**G2B** | W2 for **2.13B tokens** (`TOTAL_STEPS = 130_000`) | 42.69 | [folder](notebooks/conservative_arch/scaleup/results/gpt2_baseline_wsd_lr0p0012_st130000/), [log](notebooks/conservative_arch/scaleup/results/gpt2_baseline_wsd_lr0p0012_st130000/matched_gpt2_LR=1.2e-03_130000steps_training_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-owt-d384-gpt2-matched-2b) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-g2b) |

---

## Geometry, settling, balance and register diagnostics on the CfC/BAOAB ladder (book §D.10)

<a name="ladder-diagnostics"></a>The CG, SR, CB and DP codes of the book's Appendix D. The CG codes are
cells of the ladder notebook run on trained arms; their outputs sit in
each arm's results folder (see [the ladder](#ladder)). SR2 and SR3 ran as
[SR2](#sr2) and [SR3-crit](#sr3-crit).

| Code | What it is | Status | Notebook, cell or script | Results | Checkpoint | Written up in |
|---|---|---|---|---|---|---|
| <a name="cg1"></a>**CG1** | One-step replay: how far each layer step deflects from the damped V_θ geodesic step, with arms that switch off V_φ, LayerNorm and the reverse channel (note label E1) — Book: Table 58 (`tab:cg1-arms`) | done (2026-09-25; re-run on later arms) | [`colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb), Cell 6b-9 | `Cell-6b-9_*_output.txt` in each arm's folder, e.g. [F3.1](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/); the first run's output is not in the repository | evaluation only | [geodesic note §4](companion_notes/Geodesic_Experiments_with_CfC_BAOAB.md#n-cg1) |
| <a name="cg2"></a>**CG2** | Refinement at fixed integration time T: is the layer stack a flow or a sequence of maps? (Gates 0–3; the note's decomposed refinement E2 was designed but not run) — Book: Remark 67 (`rem:refinement-invariance`) | done (Gate 3, 2026-09-24 on; re-run on later arms) | [`colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb), Cell 6b-7 | `Cell-6b-7_flow_or_maps*_output.txt` in each arm's folder, e.g. [F3.1](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) | evaluation only | [flow-or-maps note §8](companion_notes/Composing_Single_Layer_Inferences_Flow_or_Maps.md#n-cg2-gate3); design [geodesic note §5](companion_notes/Geodesic_Experiments_with_CfC_BAOAB.md#n-cg2) |
| <a name="cg3"></a>**CG3** | Is the trajectory more forecastable than the matched GPT-2's? Three metrics (note label E3) — Book: Proposition 115 (`prop:cg-sphere`) | done (null at L = 2; scored on L = 4 live, 2026-10-03) | [`colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb), Cell 6b-10 | [F3.1](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/Cell-6b-10_forecastable_trajectory_output.txt), [L = 4 live](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_vplive_xilive_L4probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt2_lr0p0012_noattn/Cell-6b-10_forecastable_trajectory_output.txt) | evaluation only | [geodesic note §6](companion_notes/Geodesic_Experiments_with_CfC_BAOAB.md#n-cg3); [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-cg3) |
| <a name="cg4"></a>**CG4** | LayerNorm acts as a position-only constraint, not a perturbation (note label E4), read from CG1's LayerNorm-off arm — Book: Proposition 113 (`prop:cg-ln-velocity`) | analysis | [`colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb), Cell 6b-9 (LayerNorm-off arm) | in the Cell 6b-9 outputs | evaluation only | [geodesic note §7](companion_notes/Geodesic_Experiments_with_CfC_BAOAB.md#n-cg4) |
| <a name="cg5"></a>**CG5** | Reverse-channel slider: scale the reverse channel by λ from 1 to 0 and trade perplexity for geodesicity (note label E5) — Book: Figure 70 (`fig:e5-slider`) | done (2026-09-25, re-run 09-27: 69.5 → 271.8 PPL, no knee) | [`colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb), Cell 6b-11 | no Cell 6b-11 log in the repository; figure [slider](companion_notes/figures/geodesic_cfc/gcfc_e5_slider_measured.png) | evaluation only | [geodesic note §11](companion_notes/Geodesic_Experiments_with_CfC_BAOAB.md#n-cg5) |
| <a name="cg6"></a>**CG6** | Per-token forcing: is the non-conservative deflection sparse? Uniform on every arm (note label F1) — Book: §27.14 (`par:forcing-is-uniform`) | done (2026-09-26/27) | [`colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb), Cell 6b-12 | [Gen 2 conservative-only](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/F1_6b12_per_token_deflection.txt); `Cell-6b-12_sparse_forcing_output.txt` in later arms' folders | evaluation only | [forced-Lagrangian note §3.1](companion_notes/Forced_Lagrangian_Reformulation.md#n-cg6) |
| <a name="cg7"></a>**CG7** | ω·Δt at the end of training on fixed tokens, compared across arms (note label D2) — Book: Proposition 119 (`prop:cfp-no-wall`) | done (scored 2026-09-29; run on later arms) | [`colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb), Cell 6b-13 | [Gen 2 L = 2](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/D2_6b13_omega_dt_endpoint.txt); `Cell-6b-13_*_output.txt` in later arms' folders | evaluation only | [diagnostic note §18.3](companion_notes/Diagnostic_Programme_in_CfC_BAOAB_Integrator.md#n-cg7); [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-cg7) |
| <a name="cg8"></a>**CG8** | Does the geometry predict the model's own next-token errors beyond its confidence? — Book: §38.6 (`subsec:geom-requirements`) | planned (Cell 6b-14 not yet written) | ladder notebook, Cell 6b-14 (planned) | — | — | [protocol §5.12](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-cg8) |
| <a name="sr1"></a>**SR1** | Palindromic step order, the friction split around the kick (A·O½·B·O½·A); F3.1 base — Book: Table 103 (`tab:expidx-sr`) | planned | ladder notebook (to be added) | — | — | [protocol §5.9](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr) |
| <a name="sr4"></a>**SR4** | Variable-step training: (a) N ~ U{2, 3, 4} at fixed T (refinement axis), (b) N ~ U{2, 3} at Δt = 4 (extension axis) — Book: Table 103 (`tab:expidx-sr`) | planned | ladder notebook (to be added) | — | — | [protocol §5.9](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr) |
| <a name="sr5"></a>**SR5** | Thermal training: an O-step thermostat in training only, `LANGEVIN_T` ∈ {0.00025, 0.0022}; annealed variant only if SR4(b) passes — Book: Table 103 (`tab:expidx-sr`) | planned (temperatures calibrated 2026-10-03) | [`colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb), Cell 0 `LANGEVIN_T`; [`calibrate_langevin_T.py`](notebooks/conservative_arch/scaleup/debug/calibrate_langevin_T.py) | [calibration](notebooks/conservative_arch/scaleup/debug/calibrate_langevin_T_output.txt) | — | [protocol §5.9](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr5) |
| <a name="cb1"></a>**CB1** | Register path ramped in late: `REVERSE_CHANNEL_WARMUP_STEPS = 20000` (tag `rcw20000`) instead of 4,000 — Book: Table 104 (`tab:expidx-cb`) | planned (switch verified) | [`colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb), Cell 0; [`verify_cb_switches.py`](notebooks/conservative_arch/scaleup/debug/verify_cb_switches.py) | — | — | [protocol §5.11](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-cb) |
| <a name="cb2"></a>**CB2** | Per-token cap on the register increment at ρ times the conservative step: `FOCK_BUDGET` = 1.0 (CB2a) or 0.3 (CB2b) — Book: Table 104 (`tab:expidx-cb`) | planned (switch verified) | [`colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb), Cell 0; [`verify_cb_switches.py`](notebooks/conservative_arch/scaleup/debug/verify_cb_switches.py) | — | — | [protocol §5.11](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-cb) |
| <a name="cb3"></a>**CB3** | Learned per-token gate on the register increment, L1-penalised: `FOCK_GATE_L1 = 0.02` (tag `fg0p02`) — Book: Table 104 (`tab:expidx-cb`) | planned (switch verified) | [`colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb), Cell 0; [`verify_cb_switches.py`](notebooks/conservative_arch/scaleup/debug/verify_cb_switches.py) | — | — | [protocol §5.11](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-cb) |
| <a name="dp1"></a>**DP1** | Is particle number dynamic? Active fraction of the registers from layer 1 on — Book: Table 105 (`tab:expidx-dp`) | done (2026-10-05): 0.9995–1.000, no number dynamics | [`dp_register_statistics.py`](notebooks/conservative_arch/scaleup/debug/dp_register_statistics.py) (CPU) | [output](notebooks/conservative_arch/scaleup/debug/dp_register_statistics_output.txt), [JSON](notebooks/conservative_arch/scaleup/debug/dp_register_statistics.json) | evaluation only (G2, L = 4 live, G3) | [protocol §5.14](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-dp-scored); [Doi–Peliti note §7](companion_notes/Doi_Peliti_Dynamics_of_Semantic_Particles_and_Registers.md#n-dp) |
| <a name="dp2"></a>**DP2** | Do registers share content? Near-duplicate pairs (cosine > 0.9) among active registers — Book: Table 105 (`tab:expidx-dp`) | done: at most 1.3%, distinct slots | [`dp_register_statistics.py`](notebooks/conservative_arch/scaleup/debug/dp_register_statistics.py) | [output](notebooks/conservative_arch/scaleup/debug/dp_register_statistics_output.txt) | evaluation only | [protocol §5.14](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-dp-scored) |
| <a name="dp3"></a>**DP3** | Does salience act as intensity? Spearman correlation of salience with the register's own leave-one-out force — Book: Table 105 (`tab:expidx-dp`) | done: −0.41 to 0.0, retention not intensity | [`dp_register_statistics.py`](notebooks/conservative_arch/scaleup/debug/dp_register_statistics.py) | [output](notebooks/conservative_arch/scaleup/debug/dp_register_statistics_output.txt) | evaluation only | [protocol §5.14](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-dp-scored) |

---

## First- versus second-order twins on the exact integrator (book §26.9)

<a name="fo-series"></a>Does a model need second-order (inertial) dynamics, or does a first-order
gradient flow of the same potential do as well? Each arm below is trained
from scratch beside its twin, at the same seed and on the same batches, so
each pair is also compared step by step. Pre-registered in the protocol,
[§5.13 FO series](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-series) and the memory-substitution block that
follows it.

- **Book:** §26.9 (`subsec:fos-memory-substitution`); Table 45
  (`tab:ms-twobytwo`), Table 51 (`tab:ms-tinystories`), Tables 47–48
  (`tab:ms-code`, `tab:ms-code-split`). The numbers are those of the
  build of 2026-10-27; the labels in brackets are what to search for in
  the book's source if they move.
- **Notebook (every arm except the two GPT-2 references):**
  [`colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb).
  Set Cell 0 as listed, then run the cells in order. Common settings:
  L = 2, d = 384, seed 0, register path off (`REVERSE_CHANNEL = False`,
  tag `norc`), speed switches on (`VPHI_ANALYTIC_FORCE = True`,
  `LAYER_CHECKPOINT = False`; same arithmetic). The full Cell 0 of each
  arm is printed on its Hugging Face card.
- **Two first-order forms.** <a name="fo-a"></a>**FO-a**
  ([definition](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-a)) resets the velocity where each layer begins
  and keeps the stiff modes' rotation within the layer.
  <a name="fo-b"></a>**FO-b** ([definition](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b)) replaces each
  layer's second-order step by the gradient flow of the same potential,
  with the stiff modes relaxed exactly and no velocity anywhere.
- **Settled PPL.** On OpenWebText, the mean of the last three
  evaluations once a line fitted to the last 3,000 steps moves by less
  than its residual scatter (book, caption of Table 45). On TinyStories
  and code, the mean of the evaluations at steps 18,000, 18,500 and
  19,000 (caption of Table 51).

### OpenWebText (32,500 steps, 0.53B tokens)

| Arm | What it is: Cell 0 against the base | Settled PPL | Run folder and training log | Checkpoint | Scored in |
|---|---|---:|---|---|---|
| <a name="sr2"></a>**SR2** | **Second order**, the conservative core: the stiff modes integrated as exact damped oscillators (`LOWRANK_DAMPED_FLOW = True`); the base of this table | 58.32 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…sr2…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_SR2_for_noattn_none_32500steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-none-norc-sr2) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr2) |
| <a name="gamma-h"></a>**learned ζ(h)** | SR2 plus a learned per-token damping ratio, ζ(h) = 0.005·exp(head(h)), capped at 10 (`LEARNED_DAMPING_RATIO = True`, `DAMPING_RATIO_INIT = 0.005`) | 59.29 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_gzh_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…sr2_gzh…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_gzh_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_learned-gamma-h_on_SR2_OWT_32500steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-none-norc-sr2-gzh) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-gamma-h) |
| <a name="fo-b-owt"></a>**FO-b-OWT** | SR2 trained as a true **first-order** gradient flow (`FIRST_ORDER_OVERDAMPED = True`) | 61.16 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_fob_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…sr2_fob…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_fob_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_FO-b-SR2_OWT_32500steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-none-norc-sr2-fob) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b-owt) |
| <a name="sr3-crit"></a>**SR3-crit** | SR2 with its stiff modes **critically damped**, γᵢ = γ₀ + 2ωᵢ; keeps the velocity between layers, removes the oscillation (`STIFF_MODE_DAMPING_RATIO = 1.0`) | 63.28 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_sr3_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…sr2_sr3…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_sr3_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_SR3-crit_on_SR2_OWT_32500steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-none-norc-sr2-sr3) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr3-crit) |
| <a name="pmx"></a>**PMX**<!-- aka: PMX-PARFLM --> | SR2 plus **64 Poisson-mode wells** integrated exactly (`POISSON_MODES = 64`, `POISSON_WELLS_EXACT = True`), second order | 54.63 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64_pmx16_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…pm64_pmx16_sr2…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64_pmx16_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_PMX_32500steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-none-norc-pm64-pmx16-sr2) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-pmx) |
| <a name="fo-b-pmx"></a>**FO-b-PMX** | PMX trained first order (`FIRST_ORDER_OVERDAMPED = True`) | 54.02 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64_pmx16_sr2_fob_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…pm64_pmx16_sr2_fob…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64_pmx16_sr2_fob_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_FO-b-PMX_OWT_32500steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-none-norc-pm64-pmx16-sr2-fob) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b-pmx) |
| <a name="pmx-k16"></a>**PMX-K16** | PMX with **16 wells** (`POISSON_MODES = 16`), second order | 54.24 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm16_pmx16_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…pm16_pmx16_sr2…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm16_pmx16_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_PM16-PMX16_on_OWT_32500steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-none-norc-pm16-pmx16-sr2) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-pmx-k16) |
| <a name="fo-b-pmx-k16"></a>**FO-b-PMX-K16** | PMX-K16 trained first order; protocol name FO-b-PM16-PMX16 | 53.73 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm16_pmx16_sr2_fob_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…pm16_pmx16_sr2_fob…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm16_pmx16_sr2_fob_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_FO-b-PM16-PMX16_on_OWT_32500steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-none-norc-pm16-pmx16-sr2-fob) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b-pmx-k16) |
| <a name="pm2-pmx16"></a>**PM2-PMX16** | PMX with **2 wells** (`POISSON_MODES = 2`), second order | 54.71 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm2_pmx16_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…pm2_pmx16_sr2…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm2_pmx16_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_PM2-PMX16_on_OWT_32500steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-none-norc-pm2-pmx16-sr2) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-pm2-pmx16) |
| <a name="fo-b-pm2-pmx16"></a>**FO-b-PM2-PMX16** | PM2-PMX16 trained first order | 53.81 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm2_pmx16_sr2_fob_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…pm2_pmx16_sr2_fob…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm2_pmx16_sr2_fob_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_FO-b-PM2-PMX16_on_OWT_32500steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-owt-d384-l2-none-norc-pm2-pmx16-sr2-fob) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b-pm2-pmx16) |

### TinyStories (16,250 steps plus 3,000 at the floor rate, 315M tokens)

Cell 0 adds `CORPUS = 'tinystories'`, `LADDER_STEPS = 16_250`,
`EXTRA_FLOOR_STEPS = 3_000`. SO-TS and FO-TS were trained to 16,250 steps
and then extended, so each has two logs.

| Arm | What it is: Cell 0 against the base | Settled PPL | Run folder and training log | Checkpoint | Scored in |
|---|---|---:|---|---|---|
| <a name="gpt2-ts"></a>**GPT-2-TS** | The **matched GPT-2** (d = 384, L = 8, 33.7M parameters) on the same tokens and schedule; notebook [`colab_matched_gpt2_baseline_openwebtext.ipynb`](notebooks/conservative_arch/scaleup/colab_matched_gpt2_baseline_openwebtext.ipynb) with `CORPUS = 'tinystories'` | 4.47 | [folder](notebooks/conservative_arch/scaleup/results/gpt2_baseline_wsd_lr0p0012_st19250_sch16250_ts/), [log](notebooks/conservative_arch/scaleup/results/gpt2_baseline_wsd_lr0p0012_st19250_sch16250_ts/matched_GPT2_TS_19250steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-ts-d384-gpt2-matched) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-gpt2-ts) |
| <a name="so-ts"></a>**SO-TS** | SR2 on TinyStories, **second order**, no wells; the base of this table | 5.16 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_ts_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…sr2…`), [log 1](notebooks/conservative_arch/scaleup/results/cfc_baoab_ts_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_SO-TS_16250steps_output.txt), [log 2](notebooks/conservative_arch/scaleup/results/cfc_baoab_ts_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_SO-TS_19250steps_ext_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-ts-d384-l2-none-norc-sr2) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-so-ts) |
| <a name="fo-ts"></a>**FO-TS** | SO-TS with the velocity reset where each layer begins (**FO-a**, `FIRST_ORDER_MEMORYLESS = True`); keeps the rotation within a layer | 5.13 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_ts_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_fo_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…sr2_fo…`), [log 1](notebooks/conservative_arch/scaleup/results/cfc_baoab_ts_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_fo_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_FO-TS_16250steps_output.txt), [log 2](notebooks/conservative_arch/scaleup/results/cfc_baoab_ts_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_fo_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_FO-TS_19250steps_ext_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-ts-d384-l2-none-norc-sr2-fo) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-ts) |
| <a name="fo-b-ts"></a>**FO-b-TS** | SO-TS trained as a true **first-order** gradient flow (**FO-b**) | 5.28 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_ts_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_fob_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…sr2_fob…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_ts_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_fob_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_FO-b-noattn_SR2_on_TS_19250steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-ts-d384-l2-none-norc-sr2-fob) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b-ts) |
| <a name="pm2-ts"></a>**PM2-TS** | SO-TS plus **2 wells** (PM2-PMX16's settings), second order | 4.99 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_ts_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm2_pmx16_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…pm2_pmx16_sr2…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_ts_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm2_pmx16_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_PM2-PMX16_on_TS_19250steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-ts-d384-l2-none-norc-pm2-pmx16-sr2) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-pm2-ts) |
| <a name="fo-b-pm2-ts"></a>**FO-b-PM2-TS** | PM2-TS trained first order | 4.96 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_ts_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm2_pmx16_sr2_fob_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…pm2_pmx16_sr2_fob…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_ts_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm2_pmx16_sr2_fob_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_FO-b-PM2-PMX16_on_TS_19250steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-ts-d384-l2-none-norc-pm2-pmx16-sr2-fob) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b-pm2-ts) |

### Python code (CodeParrot; 16,250 steps plus 3,000 at the floor rate, 315M tokens)

Cell 0 adds `CORPUS = 'code'`, `LADDER_STEPS = 16_250`,
`EXTRA_FLOOR_STEPS = 3_000`. The copy split of every arm (where the gap
sits, by kind of token and by distance to an earlier occurrence) is in
[`results/copy_split_code/`](notebooks/conservative_arch/scaleup/results/copy_split_code/).

| Arm | What it is: Cell 0 against the base | Settled PPL | Run folder and training log | Checkpoint | Scored in |
|---|---|---:|---|---|---|
| <a name="gpt2-code"></a>**GPT-2-CODE** | The **matched GPT-2** on the same Python tokens and schedule; notebook [`colab_matched_gpt2_baseline_openwebtext.ipynb`](notebooks/conservative_arch/scaleup/colab_matched_gpt2_baseline_openwebtext.ipynb) with `CORPUS = 'code'` | 4.96 | [folder](notebooks/conservative_arch/scaleup/results/gpt2_baseline_wsd_lr0p0012_st19250_sch16250_code/), [log](notebooks/conservative_arch/scaleup/results/gpt2_baseline_wsd_lr0p0012_st19250_sch16250_code/matched_GPT2_Code_19250steps_output.txt) | not published | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-gpt2-code) |
| <a name="so-code"></a>**SO-CODE** | SR2 on Python code, **second order**, no wells; the base of this table | 6.00 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_code_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…sr2…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_code_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_noattn-sr2_on_Code_19250steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-code-d384-l2-none-norc-sr2) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-so-code) |
| <a name="fo-b-code"></a>**FO-b-CODE** | SO-CODE trained as a true **first-order** gradient flow (**FO-b**) | 6.55 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_code_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_fob_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…sr2_fob…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_code_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_fob_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_FO-b-noattn-sr2_on_Code_19250steps_output.txt) | [card](https://huggingface.co/dimitarpg13/semsimula-ladder-live-code-d384-l2-none-norc-sr2-fob) | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b-code) |
| <a name="fock-code"></a>**FOCK-CODE** | SO-CODE with the **register path** on (`REVERSE_CHANNEL = True`), second order; exploratory, outside MS5 | 5.66 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_code_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_vplive_xilive_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…xilive_sr2…`, no `norc`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_code_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_vplive_xilive_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_Fock-rc-noattn_on_Code_19250steps_output.txt) | not published | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fock-code) |
| <a name="xi6-code"></a>**XI6-CODE** | SO-CODE with a **sixth content-routed context channel** (`XI_OVERRIDE = '6long'`), second order; exploratory, outside MS5; its layer-1 wells died ([root cause](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-dead-wells)) | 8.55 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_code_xi6long_topk16_dt32da16_mh4_aniso_dcvt6x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`xi6long_…dcvt6x8…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_code_xi6long_topk16_dt32da16_mh4_aniso_dcvt6x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_XI6_on_Code_19250steps_output.txt) | not published | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-xi6-code) |
| <a name="xi6-code-wr"></a>**XI6-CODE-WR** | XI6-CODE plus the **well-reachability penalty** (`WELL_REACH_RHO = 3`); exploratory; layer 1 died again ([result](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-xi6-code-wr)) | 8.87 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_code_xi6long_topk16_dt32da16_mh4_aniso_dcvt6x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_wr3l0p1_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…wr3l0p1…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_code_xi6long_topk16_dt32da16_mh4_aniso_dcvt6x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_wr3l0p1_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_XI6-WR_on_Code_19250steps_output.txt) | not published | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-xi6-code-wr) |
| <a name="dw-screen-a"></a>**DW-SCREEN-A** | XI6-CODE-WR plus **unit-norm context channels at V_θ's bank input** (`VTHETA_INPUT_UNIT_NORM = True`), 3,000 steps; dead-wells screen A; layer 1 alive ([result](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-dead-wells-screens-result)) | 13.54 at step 3,000 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_code_xi6long_topk16_dt32da16_mh4_aniso_dcvt6x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_wr3l0p1_vtun_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…wr3l0p1_vtun…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_code_xi6long_topk16_dt32da16_mh4_aniso_dcvt6x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_wr3l0p1_vtun_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2probe_XI6-WR-Scan_A_on_Code_3000steps_output.txt) | not published | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-dead-wells-screens-result) |
| <a name="dw-screen-b"></a>**DW-SCREEN-B** | XI6-CODE-WR plus **V_θ's learning rate × 0.25** (`VTHETA_LR_MULT = 0.25`); dead-wells screen B, screened to step 3,000 ([screen](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-dead-wells-screens-result)) and resumed to full length; layer 1 alive at the end, level with SO-CODE ([result](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-dw-screen-b-full)) | 5.98 | [folder](notebooks/conservative_arch/scaleup/results/cfc_baoab_code_xi6long_topk16_dt32da16_mh4_aniso_dcvt6x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_wr3l0p1_vtlr0p25_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/) (`…wr3l0p1_vtlr0p25…`), [log](notebooks/conservative_arch/scaleup/results/cfc_baoab_code_xi6long_topk16_dt32da16_mh4_aniso_dcvt6x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_wr3l0p1_vtlr0p25_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_XI6-WR-Scan_B_on_Code_19250steps_output.txt) (steps 3,001–19,250; the first 3,000 in [the screen's log](notebooks/conservative_arch/scaleup/results/cfc_baoab_code_xi6long_topk16_dt32da16_mh4_aniso_dcvt6x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_wr3l0p1_vtlr0p25_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2probe_XI6-WR-Scan_B_on_Code_3000steps_output.txt)) | not published | [protocol](companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-dw-screen-b-full) |

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

---

## Reproducing the paper's experiments

This section contains step-by-step reproduction commands for every experiment in the paper.
See the [README.md](README.md) for prerequisites (clone, venv, dependencies, Git LFS).

### 0. Prerequisites

```bash
# 1. Clone and enter the repository
git clone https://github.com/dimitarpg13/semsimula-paper.git
cd semsimula-paper

# 2. Create and activate a Python 3.10+ virtual environment
python -m venv .venv && source .venv/bin/activate   # Linux / macOS
# python -m venv .venv && .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt
# For GPU training, replace the torch line first:
#   pip install torch==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# 4. Pull Git LFS artefacts (PNG figures, NPZ result archives, NPY surprisal
#    tensors, GIF landscape rotations). Note: as of the v3 release no model
#    checkpoints or trajectory pickles are shipped -- see the "Checkpoint
#    policy" notice at the top of this README.
git lfs pull
```

All results committed to this repository were produced on a **MacBook Pro,
Intel Core i9 2.3 GHz (8-core), 64 GB RAM**, running Python 3.12.11,
between 18 April and 25 April 2026. GPU is **not required**; every script
runs on CPU (or MPS on Apple Silicon). The shipped `results/` artefacts
mean that any figure in the paper can be replotted without retraining.

---

### §13 — descriptive experiments (Results 1–5)

```bash
# Results 1–4 and Figures 4–6: STP–acceleration identity and
# Gaussian-well analysis on GPT-2 small.
jupyter lab notebooks/stp_loss/energy_landscape_validation.ipynb
# Static rendering with all outputs pre-executed:
#   notebooks/stp_loss/energy_landscape_validation_executed.ipynb

# Result 5: cross-architecture replication on GPT-2 small + Pythia-160M.
jupyter lab notebooks/cross_model/pythia_tangential_acceleration.ipynb
```

Expected runtime: **5–15 minutes** per notebook on the reference hardware
(GPT-2 small hidden-state extraction over a ~200-sentence corpus is the
bottleneck; the Pythia notebook adds a second model load).

---

### §14.1 — negative-results chain on attention transformers (E1–E5)

```bash
# Baseline Gaussian-well E-init on GPT-2 (prerequisite for E1–E5):
jupyter lab notebooks/e_init/e_init_validation.ipynb

# E1 (damping sweep) + E2 (first-order gradient flow):
python notebooks/e_init/extended_gamma_and_first_order.py

# E3 (seven scalar-well functional forms):
python notebooks/e_init/well_functional_form_comparison.py

# E4 (linear Helmholtz position-coupled skew augmentation):
python notebooks/e_init/helmholtz_curl_augmented.py

# E5 (velocity-coupled gauge, constant / affine-rank-1 / affine-rank-2):
python notebooks/e_init/velocity_coupled_gauge.py
```

See [`notebooks/e_init/README.md`](notebooks/e_init/README.md) for the
exact command sequence and the mapping from experiment IDs E1–E5 to
scripts. Each script writes a markdown summary, an `.npz` result archive,
and one or more figures to `notebooks/e_init/results/`.

Expected runtime: **2–10 minutes** per script on the reference hardware.

---

### §14.2 ff. and Appendix A — prescriptive experiments (SPLM pipeline)

The full end-to-end pipeline is documented step by step in
[`notebooks/conservative_arch/README.md`](notebooks/conservative_arch/README.md).
The quick summary:

```bash
cd notebooks/conservative_arch

# 1. Train SPLM and the scale-matched attention baseline
python train_splm.py
python train_matched.py

# 2. Extract hidden-state trajectories from all three models
python trajectory_extraction.py          # SPLM
python extract_matched_baseline.py       # matched GPT-2
python extract_gpt2_baseline.py          # pretrained GPT-2 small

# 3. Run the full diagnostic suite
python shared_potential_fit.py           # strict shared-V_psi separator
python jacobian_symmetry.py              # velocity-aware Jacobian-symmetry test
python sharedV_capacity_sweep.py         # 6-config V_psi capacity band
python splm_oracle_fit.py                # oracle upper bound (SPLM's own V_theta)
python token_direction_fit.py            # token-direction replication

# 4. Produce paper figures
python plot_three_way_comparison.py      # Fig. 8: SPLM / matched / pretrained GPT-2
python plot_token_vs_layer_three_way.py  # token-direction two-panel figure
python plot_sharedV_comparison.py        # shared-V_psi profile plot
```

**Note (v3 release, May 2026):** earlier releases shipped the
checkpoint files (`*.pt`) and trajectory pickles (`*.pkl`) via Git
LFS, which made it possible to skip steps 1–2 and run steps 3–4 in
~5 minutes. As of v3 those artefacts are no longer shipped (see the
*Checkpoint policy* notice at the top of this README); the full
pipeline must be re-run end-to-end. Total **~35–45 minutes** (SPLM
~20 min, matched baseline ~15 min, extraction ~5 min, diagnostics
~5 min) on the reference hardware. The training logs, summary
markdown, and loss-curve PNGs remain committed and back the
quantitative claims that depend on the historical (v2-buggy) runs;
the v3 paper documents which of those numbers survive the leak fix
and which are flagged as casualties of the bug.

#### SARF-faithful ablation (§14.13)

```bash
cd sarf_variant
python train_splm_sarf.py
python trajectory_extraction_sarf.py
python compare.py
# Re-runs shared_potential_fit.py and token_direction_fit.py from the
# parent directory automatically; outputs go to ../results/.
```

#### Per-token semantic-mass ablation (§14.14)

```bash
cd sarf_mass_variant
python compute_unigram_frequencies.py    # build frozen surprisal lookup (once)
python train_splm_sarf_mass.py --mass-mode global
python train_splm_sarf_mass.py --mass-mode embed_head
python train_splm_sarf_mass.py --mass-mode logfreq
python trajectory_extraction_sarf_mass.py
python compare.py
```

#### Attractor analysis (§14.15)

```bash
cd attractor_analysis
python attractor_extraction.py --mode gradient    # Adam descent on V_theta
python attractor_extraction.py --mode dynamical   # damped Euler from random seeds
python compare_landscapes_3d.py                    # Euler-vs-Verlet 3D comparison
python train_with_snapshots.py                     # retrain with log-spaced checkpoints
python render_training_evolution.py                # seven-panel landscape-evolution grid
```

See [`notebooks/conservative_arch/attractor_analysis/README.md`](notebooks/conservative_arch/attractor_analysis/README.md)
for the per-prompt JSON/PNG/MD output catalogue.

#### Energetic-minima alternatives (F5 of §14.17)

```bash
cd energetic_minima
python train.py --variant ln   # LayerNorm-after-step
python train.py --variant sg   # scale-gauge regulariser
python train.py --variant gm   # Gaussian-mixture V_theta head
bash run_attractor_pipeline.sh  # attractor extraction over all four checkpoints
python compare.py               # produces results/comparison_report.md
python make_compare_figure.py   # produces results/landscape3d_compare_four_variants_dialogue.png
```

See [`notebooks/conservative_arch/energetic_minima/README.md`](notebooks/conservative_arch/energetic_minima/README.md)
for variant flags, training schedule, and expected outputs.

#### Multi-seed variance harness (E1 of `Next_Model_Experiments_for_SPLM.md`)

```bash
cd notebooks/conservative_arch

# 0. (Once) Precompute the surprisal lookup table for SPLM logfreq mass.
python sarf_mass_variant/compute_unigram_frequencies.py

# 1. Smoke test (single seed, ~1-2 minutes total).
python multi_seed/multi_seed_runner.py \
    --mode smoke --n-seeds 1 --models splm_sarfmass_logfreq

# 2. E1 production: 5 seeds x 3 models on Tiny Shakespeare
#    (~7-8 hours wall-clock on Apple MPS; runs sequentially).
python multi_seed/multi_seed_runner.py \
    --mode shakespeare --n-seeds 5 \
    --models splm_em_ln,splm_sarfmass_logfreq,matched_baseline

# 3. Aggregate logs into report + overlay plots + divergence diagnostic.
python multi_seed/multi_seed_aggregator.py --tag E1_shakespeare
python multi_seed/e1_divergence_diagnostic.py --tag E1_shakespeare
```

The shipped `results/E1_shakespeare/` includes 13 per-seed training
logs (JSONL + summary markdown) and loss-curve PNGs (5 seeds for
`matched_baseline` and `splm_em_ln`, 3 for `splm_sarfmass_logfreq`
before the divergence-rate diagnostic short-circuited the sweep), plus
the curated [`E1_report.md`](notebooks/conservative_arch/multi_seed/results/E1_shakespeare/E1_report.md)
narrative. **Per-seed `*.pt` checkpoints are not shipped** as of the
v3 release (see the *Checkpoint policy* notice at the top of this
README) — re-run step 2 above to regenerate them. Re-running step 3
alone, however, *does* reproduce the mean / std / Welch-t table and
the overlay figures from the shipped per-seed JSONL training logs in
**~20 seconds**, since the aggregator and the divergence diagnostic
only read the logs (not the checkpoints). See
[`notebooks/conservative_arch/multi_seed/README.md`](notebooks/conservative_arch/multi_seed/README.md)
for the model-spec interface and the recipe for adding new variants
(E2 width sweep, E3 integrator ablation, etc.).

#### Energy-drift diagnostic (E3 of `Next_Model_Experiments_for_SPLM.md`)

The production E3 comparison is `parent_euler_L8` × `verlet_L16_dt05` ×
`em_ln_L8_seed0` (LayerNorm-after-step SPLM, val ppl 88.63 at seed 0,
the production-best variant of the multi-seed E1 sweep). The originally
planned `sarfmass logfreq` (no-LN) column is omitted: E1 multi-seed
falsified its stability (2/3 NaN-divergent seeds), so a single-seed
energy trace from it is not representative.

> **Note (v3 release).** As of the v3 release the three SPLM
> checkpoints referenced below are *not* shipped (see the *Checkpoint
> policy* notice at the top of this README). To re-run the E3
> diagnostic from scratch, first regenerate the relevant checkpoints
> by running the appropriate `train_*.py` scripts under
> `notebooks/conservative_arch/`,
> `notebooks/conservative_arch/symplectic_variant/`, and
> `notebooks/conservative_arch/multi_seed/multi_seed_runner.py`
> (`splm_em_ln` model spec, seed 0). The example commands below
> assume the regenerated checkpoint paths.

```bash
cd notebooks/conservative_arch

# 1. Extract energy states for the three production SPLM checkpoints
#    (must be regenerated locally; see the v3 note above).
python energy_drift/extract_energy_states.py \
    --variant euler \
    --ckpt results/splm_shakespeare_ckpt_latest.pt \
    --label splm_euler_L8 \
    --out_npz energy_drift/results/splm_euler_L8.npz

python energy_drift/extract_energy_states.py \
    --variant symplectic \
    --ckpt symplectic_variant/results/splm_sym_logfreq_shakespeare_L16_dt05_ckpt_latest.pt \
    --label splm_verlet_L16_dt05 \
    --logfreq sarf_mass_variant/results/logfreq_surprisal.npy \
    --out_npz energy_drift/results/splm_verlet_L16_dt05.npz

python energy_drift/extract_energy_states.py \
    --variant em_ln \
    --ckpt multi_seed/results/E1_shakespeare/splm_em_ln/seed_0/em_ln_shakespeare_ckpt_latest.pt \
    --label splm_em_ln_L8_seed0 \
    --logfreq sarf_mass_variant/results/logfreq_surprisal.npy \
    --out_npz energy_drift/results/splm_em_ln_L8_seed0.npz

# 2. Cross-variant comparison: drift slope, oscillation bandwidth,
#    overlay plots of H_l, kinetic, and potential per layer.
python energy_drift/energy_drift_diagnostic.py \
    --inputs splm_euler_L8.npz,splm_verlet_L16_dt05.npz,splm_em_ln_L8_seed0.npz \
    --tag E3_splm_em_ln_compare
```

The full pipeline finishes in under three minutes on MPS. The diagnostic
is forward-pass-only on existing checkpoints; no retraining is required.
See
[`notebooks/conservative_arch/energy_drift/README.md`](notebooks/conservative_arch/energy_drift/README.md)
for the variant flags, the expected drift signatures for each
integrator, the production-result interpretation, and the relationship
to `shared_potential_fit.py` and `attractor_analysis/`. If you want to
run the diagnostic on the original `sarfmass logfreq` no-LN checkpoint
as an ablation, the syntax is documented in the in-folder README; any
conclusion drawn from such a single-seed run must be qualified by the
E1 multi-seed instability finding.

---

### v3 — leak-correction audit and the R6 ladder

The v3 paper revision (May 2026) adds a regression-tested causality
audit, a leak-corrected re-evaluation of every v2 SPLM checkpoint, a
leak-corrected single-channel pilot, an information-bottleneck
programme over the multi-channel-ξ basis (the *R6 ladder*: K-EMA /
log-spaced K-EMA / HiPPO-LegT / learnable-Δ HiPPO / S4D), and
leak-free 3-seed retrains of the v2 SPLM-1 first-order ablation
and the controlled-$\gamma$ damping sweep — both of whose v2
absolute and paired numbers do not survive the leak-fix (see the
*v3 leak-correction and information-bottleneck programme* table
above). The full programme is documented in
[`companion_notes/Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md`](companion_notes/Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md);
the audit framework is documented in
[`companion_notes/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](companion_notes/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md).
Quick-start commands:

```bash
cd notebooks/conservative_arch

# 1. Causality regression test (≤ 5 s; runs on every model variant
#    registered in the file — fails loudly if a future change
#    re-introduces the v2 leak).
python causal_probe.py --strict

# 2. Leak-free re-evaluation of any v2 SPLM checkpoint
#    (closed-loop, fixed-graph; produces (PPL_buggy, PPL_fixed)).
python eval_ppl_under_fix.py \
    --ckpt scaleup/results/multixi_buggy_2k/em_ln_multixi_tinystories_ckpt_latest.pt

# 3. Leak-corrected single-channel SPLM pilot on TinyStories
#    (~30 min on Apple MPS; ~10 min on a single A100).
python post_fixed_pilot.py

# 4. R6 ladder pilots (~30 min – 4 h each on Apple MPS, depending on
#    cell — see `scaleup/README.md` for per-cell runtimes).
python scaleup/train_splm_em_ln_multixi_scaleup.py        \
    --xi-channels 4 --xi-alpha-init-mode log_spaced       \
    --xi-tau-max 100  --tag multixi_pilot_logspaced_taumax100   # R6.h.1
python scaleup/train_splm_em_ln_multixi_hippo_scaleup.py  \
    --xi-channels 4 --tag multihippo_pilot_fixed                # R6.a
python scaleup/train_splm_em_ln_multixi_hippo_scaleup.py  \
    --xi-channels 4 --learnable-dt --tag multihippo_pilot_learndt   # R6.e
python scaleup/train_splm_em_ln_multixi_s4d_scaleup.py    \
    --xi-channels 4 --xi-eigval-init legt --tag multis4d_pilot_legtinit  # R6.i

# 5. Forensic re-eval of the buggy SPLM-1 / SPLM-2 ckpts under the
#    corrected integrator (loads the v2 ckpts, runs them through
#    causal_force=True; reports asymmetric inflation 3.89× / 14.83×
#    and predicts the asymmetric collapse the leak-free retrain
#    in step 6 confirms).
python first_order_ablation/splm1_leakfree_re_eval.py \
    --out-md   first_order_ablation/results/LEAKFREE_RE_EVAL.md \
    --out-json first_order_ablation/results/leakfree_re_eval.json

# 6. Leak-free 3-seed retrains of the v2 ablation grid:
#      (a) SPLM-1 vs SPLM-2 6-cell at γ=0.30 (~3 h 12 min on MPS)
#      (b) γ-sweep 9-cell at γ ∈ {0.00, 0.10, 0.85} (~4 h 47 min on MPS)
#    Combined with the existing γ=0.30 leak-free 3-seed retrain from
#    (a) above, (b) gives a 4-point leak-free U-curve that puts the
#    new γ* at 0.10 (vs the buggy γ* = 0.30); see the
#    *v3 leak-correction* paragraph above for the full headline.
bash first_order_ablation/scripts/run_ablation_leakfree.sh
bash ln_damping_sweep/scripts/run_gamma_sweep_leakfree.sh
# Per-cell artefacts land under
#   first_order_ablation/results/{splm1_leakfree,splm2_gamma0p30_leakfree}/
#   ln_damping_sweep/results/leakfree_3seed/gamma{0p00,0p10,0p85}/
# (training logs + summary mds + loss-curve PNGs; per-cell .pt
#  checkpoints are .gitignored and stay local-only).

# 7. S=5 confirmation sweep. Refine the γ-grid to {0.05, 0.10, 0.15,
#    0.20} at 5 paired SPLM-1/SPLM-2 seeds per γ to firmly establish
#    or refute the +4.71-PPL second-order lift from step 6 against
#    the pre-registered Δ_min = 5.0 PPL. Reuses the leak-free 3-seed
#    γ=0.10 cells from step 6 above; runs ~11 h of wall-clock on MPS
#    for the remaining 16 cells.
bash ln_damping_sweep/scripts/run_confirmation_5seed_sweep.sh
python ln_damping_sweep/aggregate_confirmation_5seed.py
# Per-cell artefacts land under
#   ln_damping_sweep/results/leakfree_5seed_confirmation/gamma{0p05,...,0p20}/seed{0..4}/
# Aggregated paired-t / d_z / sign-consistency table:
#   ln_damping_sweep/results/leakfree_5seed_confirmation/RESULTS_CONFIRMATION_S5.md
#   ln_damping_sweep/results/leakfree_5seed_confirmation/results_confirmation_s5.json

# 8. Tier 0.5 — γ*-prediction reanchored on a leak-corrected ckpt.
#    Runs the four-estimator framework of
#    companion_notes/Determining_optimal_gamma_for_SPLM.md on a
#    leak-free SPLM-2 checkpoint at the inference-optimal γ = 0.10.
#    Validates the *resonance-predictor double success*: the same
#    closed form predicts both v2 buggy γ* = 0.30 and leak-free
#    γ* = 0.10, with only ρ shifting (0.18 → 0.565).
python scaleup/gamma_transfer/predict_gamma_hessian.py \
    --ckpt ln_damping_sweep/results/leakfree_3seed/gamma0p10/seed0/splm_em_ln_shakespeare_gamma0p10_seed0_ckpt_latest.pt \
    --mode shakespeare \
    --tag leakfree_gamma0p10_seed0
# Output: scaleup/gamma_transfer/results/leakfree_gamma0p10_seed0/predict_gamma_summary.md
#       + scaleup/gamma_transfer/results/leakfree_gamma0p10_seed0/predict_gamma.json

# 9. Tier 0.6 — shared-potential separator on a leak-corrected ckpt.
#    Re-fits the §14.7 separator regression on a leak-free em_ln
#    checkpoint (the same Tier-0.5 ckpt). Yields median per-layer
#    test R^2 = 0.949, range [0.925, 0.960] — *higher* than the v2
#    buggy R^2 = 0.90. Used to fill the placeholders in TMLR1 §A.3.3.
python energetic_minima/trajectory_extraction_em_ln.py \
    --ckpt ln_damping_sweep/results/leakfree_3seed/gamma0p10/seed0/splm_em_ln_shakespeare_gamma0p10_seed0_ckpt_latest.pt \
    --tag em_ln_leakfree_gamma0p10_seed0
python shared_potential_fit.py \
    --traj results/splm_em_ln_leakfree_gamma0p10_seed0.trajectories.pkl \
    --tag em_ln_leakfree_gamma0p10_seed0
# Output: results/sharedV_em_ln_leakfree_gamma0p10_seed0_{fig.png, results.npz, summary.md}

# 10. Tier 1 / Tier 2a–b / Tier 3a–b retrains of the energetic-minima
#     alternatives + attractor extractions on leak-free ckpts.
#     Tier 1 — leak-free fixed-γ=0.10 attractor (~5 min on CPU):
python attractor_analysis/attractor_extraction.py \
    --ckpt ln_damping_sweep/results/leakfree_3seed/gamma0p10/seed0/splm_em_ln_shakespeare_gamma0p10_seed0_ckpt_latest.pt \
    --tag em_ln_leakfree_gamma0p10_seed0 \
    --mode dynamical --device cpu --seed 0
#     Tier 2a + 3a + 3b — em_ln / em_sg / em_gm leak-free retrains
#     on Tiny Shakespeare (~3 h on Apple MPS):
bash energetic_minima/scripts/run_leakfree_tiers_2_3.sh
#     Tier 2b — attractor extraction on the em_ln free-γ ckpt
#     produced by Tier 2a (~5 min on CPU):
bash energetic_minima/scripts/run_tier2b_attractor.sh
#     Cross-variant 4-way comparison + 3D landscape grid:
bash energetic_minima/run_attractor_pipeline.sh
python energetic_minima/compare.py
python energetic_minima/make_compare_figure.py
# Per-tier artefacts land under
#   attractor_analysis/results/attractors_em_ln_leakfree_gamma0p10_seed0_*  (Tier 1)
#   energetic_minima/results/em_{ln, sg_lam1e-03, gm_K64}_shakespeare_*     (Tiers 2a, 3a, 3b)
#   attractor_analysis/results/attractors_em_ln_leakfree_freegamma_seed0_*  (Tier 2b)
#   attractor_analysis/results/attractors_em_{ln, sg, gm}_*                 (pipeline n_sim_steps=8)
#   energetic_minima/results/comparison_{report.md, table.json}
#   energetic_minima/results/landscape3d_compare_four_variants_dialogue.png
# Canonical synthesis report:
#   energetic_minima/results/leakfree_tiers_2_3_summary.md

# 11. Information-theoretic diagnostics on a trained ξ trajectory:
#     pairwise correlation matrix, mean off-diagonal |corr|,
#     total correlation TC, entropy-power K_eff. Same script for
#     every R6-ladder variant — dispatched on the checkpoint config.
python multixi/diagnose_xi_channel_correlations.py \
    --ckpt scaleup/results/multihippo_pilot_learndt/em_ln_multixi_hippo_tinystories_ckpt_latest.pt
```

Output: each pilot writes a per-step training log, val\_ppl trajectory,
and a final-summary `.md` to `scaleup/results/<tag>/`. The
`diagnose_xi_channel_correlations.py` script writes a
`channel_correlations.json` next to the checkpoint and an overlay PNG
of the K×K correlation matrix; these are the source of the
`mean |corr|`, `TC`, and `K_eff/K` numbers in the §14 R6-ladder table
of the paper. Steps 5–10 land their own report markdowns
([`LEAKFREE_RE_EVAL.md`](notebooks/conservative_arch/first_order_ablation/results/LEAKFREE_RE_EVAL.md),
[`RESULTS_LEAKFREE.md`](notebooks/conservative_arch/first_order_ablation/results/RESULTS_LEAKFREE.md),
[`RESULTS_LEAKFREE_GAMMA_SWEEP.md`](notebooks/conservative_arch/ln_damping_sweep/results/RESULTS_LEAKFREE_GAMMA_SWEEP.md),
[`RESULTS_CONFIRMATION_S5.md`](notebooks/conservative_arch/ln_damping_sweep/results/leakfree_5seed_confirmation/RESULTS_CONFIRMATION_S5.md),
[`predict_gamma_summary.md`](notebooks/conservative_arch/scaleup/gamma_transfer/results/leakfree_gamma0p10_seed0/predict_gamma_summary.md),
[`sharedV_em_ln_leakfree_gamma0p10_seed0_summary.md`](notebooks/conservative_arch/results/sharedV_em_ln_leakfree_gamma0p10_seed0_summary.md),
[`leakfree_tiers_2_3_summary.md`](notebooks/conservative_arch/energetic_minima/results/leakfree_tiers_2_3_summary.md))
that the v3 paper's §15 leak-free addenda and the prospective TMLR1
paper's §A.3 leak-correction note cite directly.

---

### §17 — PARFLM and FockPARFLM experiments

The PARFLM (Property-Attractive-Repulsive Force Language Model) adds a
learnable pairwise potential $V_\phi$ on top of the SPLM's single-particle
$V_\theta$. The FockPARFLM further augments this with Fock-space
creation/annihilation registers to reach context-free expressivity.

```bash
cd notebooks/conservative_arch

# --- Shakespeare-scale PARF prototyping (P1–P5, ~5 min each on CPU/MPS) ---
# P1 dense structural V_phi:
python parf/train_parf.py --v-phi-kind structural --seed 0
# P5 winner — Gumbel-softmax sparse k=4:
python parf/train_parf.py --v-phi-kind structural --sparse-top-k 4 --seed 0

# --- P6 V_phi channel diagnostic on any trained PARF checkpoint ---
python parf/diagnostics/diagnose_v_phi_channels.py \
    --ckpt parf/results/structural_sparse/seed0_k4/ckpt.pt

# --- P10 TinyStories ladder (A100/H100 on Google Colab) ---
# Open parf/scripts/p10_tinystories_a100_h100.ipynb, set CELL='P10h',
# and run all cells.  Artefacts route to GDrive under
# semsimula_parflm/p10_tinystories/P10h/seed0/.

# --- FockPARFLM: Dyck_2 expressivity falsifier (Phase 1, local MPS/CPU) ---
OMP_NUM_THREADS=1 python parf/train_fock_parf.py \
    --corpus dyck --arch fock --discipline stack \
    --d 64 --L 4 --M 16 --steps 4000 --seed 0
# Baseline comparison (same scale, no registers):
OMP_NUM_THREADS=1 python parf/train_fock_parf.py \
    --corpus dyck --arch parflm \
    --d 64 --L 4 --steps 4000 --seed 0
```

**P10 results (committed training logs):**

| Cell | Val PPL (best) | What it tests |
| ---- | -------------- | ------------- |
| P10a | 32.60 | Anchor — broke the SPLM em\_ln 33.55 wall |
| P10e | 31.12 | V\_φ capacity ablation (4× wider V\_φ) |
| P10f | 28.67 | V\_θ ceiling test (v\_hidden 1024 → 2048) |
| P10g | 26.42 | Training-budget disambiguator (16k steps) |
| **P10h** | **26.43** | **Corpus scale-up (5M → 20M tokens) — architectural ceiling confirmed** |

**FockPARFLM Dyck₂ falsifier (Phase 1, seed 0):**

| Arm | Val PPL | Deep-test acc. (depth 5–12) |
| --- | ------- | --------------------------- |
| PARFLM baseline (no registers) | 3.50 | 37.9% |
| FockPARFLM (bag, M=16) | 3.57 | 37.4% |
| FockPARFLM (LIFO stack, M=16) | **3.43** | **39.2%** |

Design documents:
[`companion_notes/PARF_Augmented_SPLM_Architecture_v2.md`](companion_notes/PARF_Augmented_SPLM_Architecture_v2.md),
[`companion_notes/PARF-SPLM_Path_Forward_and_Experiments.md`](companion_notes/PARF-SPLM_Path_Forward_and_Experiments.md),
[`companion_notes/Augmenting_PARFLM_to_handle_MCS_Languages.md`](companion_notes/Augmenting_PARFLM_to_handle_MCS_Languages.md).

**Fock v2.1 routing-fix (B1+B2+B3) — best SPLM-family result (PPL 9.30):**

```bash
# Open colab_fock_v21_routing_fix.ipynb, set CELL to the desired arm,
# and run all cells.  Artefacts route to GDrive under
# semsimula_fock_v21_routing_fix/<arm_tag>/.
# Locally: results land under
# notebooks/conservative_arch/scaleup/results/semsimula_fock_v21_routing_fix/
```

**Fock Attention direct exchange (Route 2):**

```bash
# Open colab_fock_attention_h128.ipynb, set CELL to the desired arm,
# and run all cells.  Artefacts route to GDrive under
# semsimula_fock_attention_h128/<arm_tag>/.
# Locally: results land under
# notebooks/conservative_arch/scaleup/results/semsimula_fock_attention_h128/
```

**Controlled-conservativity diagnostic (CONS1–5):**

```bash
cd notebooks/conservative_arch/parf

# Run the 5-arm diagnostic on the trained v2.1 checkpoint.
# Replace <CHECKPOINT_PATH> with the path to the
# v21_tau_perK_ortho ckpt_latest.pt file.
python conservativity_diagnostic.py \
    --checkpoint <CHECKPOINT_PATH> \
    --output-dir results/conservativity_v21
# Produces: arm{1..5}_*.png, conservativity_report.json,
#           conservativity_summary.md
```

---

