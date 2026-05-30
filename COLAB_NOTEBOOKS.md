# Colab Notebooks

All notebooks clone from `https://github.com/dimitarpg13/semsimula-paper.git`, mount
Google Drive for persistent output, and auto-detect the available GPU (CUDA preferred,
MPS/CPU fallback). Each notebook is designed to run **one cell (experiment arm) per
Colab session** — set the `CELL` variable at the top and execute all cells sequentially.

> **Looking up an experiment code from the paper?** See
> [EXPERIMENTS.md](EXPERIMENTS.md) for a master index mapping every
> designation (P10g, SQ3, VR2, etc.) to its paper section, notebook,
> and result artefacts.

---

## Scale-Up Pilot and Confirmation

| Notebook | Path | Description |
|----------|------|-------------|
| [colab_pilot](#colab_pilot) | `notebooks/conservative_arch/scaleup/colab_pilot.ipynb` | 5-arm pilot comparing SPLM family vs matched-attention baseline at scale |
| [colab_gamma_sweep](#colab_gamma_sweep) | `notebooks/conservative_arch/scaleup/colab_gamma_sweep.ipynb` | SPLM em_ln gamma sweep to find optimal damping at scale |
| [colab_n5_confirmation](#colab_n5_confirmation) | `notebooks/conservative_arch/scaleup/colab_n5_confirmation.ipynb` | 5-seed paired confirmation of pilot results with 95% CIs |

### colab_pilot

Five-arm SPLM-family comparison at the E9 scale-up config (d=256, L=8, 8k steps)
on TinyStories (5M tokens). Arms: matched-attention baseline, SPLM em_ln (fixed
gamma=0.30), Helmholtz Q9d, Hybrid VA (k=4, m=4), and PARF Q9c sparse (k=4).
Includes optional TF32-on and full-V_phi sub-arms. Decision rule: arm wins if
PPL delta vs baseline exceeds 5 PPL.

- **Dataset:** TinyStories (5M tokens)
- **GPU:** A100 40GB primary (~13-15h total); H100 for optional arms

### colab_gamma_sweep

Tests whether small-scale optimal gamma (0.166 from E5) holds at scale-up.
Sweeps SPLM em_ln over gamma in {0.166, 0.20, 0.25, 0.30, 0.35} and identifies
the argmin gamma for val PPL.

- **Dataset:** TinyStories (5M tokens)
- **GPU:** H100 recommended (~60 min total); A100/L4 work but slower

### colab_n5_confirmation

Repeats three pilot arms across seeds 0-4 to upgrade seed-0 point estimates to
paired comparisons with 95% confidence intervals and hypothesis tests. Arms:
matched-attention baseline, Helmholtz Q9d, Hybrid VA.

- **Dataset:** TinyStories (5M tokens)
- **GPU:** H100 recommended (~95 min total); A100/L4 also work

---

## Multi-Channel ξ and PARF Memory Scaling

| Notebook | Path | Description |
|----------|------|-------------|
| [colab_alpha_init_sweep](#colab_alpha_init_sweep) | `notebooks/conservative_arch/scaleup/colab_alpha_init_sweep.ipynb` | K-EMA α-initialisation sweep for multi-channel ξ SPLM |
| [colab_parf_multixi](#colab_parf_multixi) | `notebooks/conservative_arch/scaleup/colab_parf_multixi.ipynb` | Multi-ξ PARF hybrid at H=16 (pre-memory-fix baseline) |
| [colab_parf_multixi_h128](#colab_parf_multixi_h128) | `notebooks/conservative_arch/scaleup/colab_parf_multixi_h128.ipynb` | Multi-ξ PARF at full V_φ capacity (H=128) with Level-2 checkpointing + gathered V_φ |
| [colab_fock_multixi_h128](#colab_fock_multixi_h128) | `notebooks/conservative_arch/scaleup/colab_fock_multixi_h128.ipynb` | Fock Multi-ξ PARF at full V_φ capacity (H=128) — 12-arm sweep over Fock v1/v2, register count, discipline, reverse channel, and schedule |

### colab_alpha_init_sweep

Sweeps K-EMA α-initialisation strategies for the multi-channel ξ SPLM
(no PARF). Tests hand-picked, log-spaced, uniform, and learned-from-X
strategies at K=4 channels. Winner (`learned_from_uniform`, α≈[0.25,
0.50, 0.75, 0.95]) achieved 14.69 PPL at 4000 steps — the multi-ξ
SPLM baseline.

- **Dataset:** TinyStories (5M tokens)
- **GPU:** A100 40GB (~45 min per arm, 8000 steps)

### colab_parf_multixi

Three-arm pilot combining multi-channel K-EMA ξ with sparse PARF pair
forces (MultiXiPARFLM). Tests competitive V_φ (k=8, k=4) and plain
structural V_φ (k=8), all at H=16 and grad-accum=2 due to V_φ memory
constraints. Best arm reached 15.44 PPL — better than single-ξ PARF
(28 PPL) but above multi-ξ SPLM (14.69), suggesting V_φ capacity is
the binding constraint.

- **Arms:** competitive k=8, competitive k=4, structural k=8
- **Dataset:** TinyStories (5M tokens)
- **GPU:** A100 40GB (~1h per arm, pilot 4000 steps, grad-accum=2)

### colab_parf_multixi_h128

Re-runs multi-ξ PARF at full V_φ capacity (H=128) enabled by two
memory optimisations:
- **Level-2 per-layer checkpointing** (`--use-layer-checkpoint`):
  wraps each `_layer_step` in `checkpoint(use_reentrant=False)`,
  reducing peak V_φ memory from O(L) to O(1) layers.
- **Stage-1.5b gathered V_φ** (`--use-gathered-v-phi`): evaluates
  V_φ only at top-k indices, reducing intermediates from O(T²) to
  O(T·k) — 128× at k=4, T=512.

Combined effect: peak V_φ activation memory drops from ~8 GB to ~8 MB,
enabling H=128 without grad-accum on A100 40GB. Six arms sweep channel
count (K=2, 4, 8), α-init strategy (hand-picked, log-spaced), top-k
(4, 8), and V_φ kind (competitive, structural).

- **Arms:** comp_K4_best_alpha, comp_K4_k4, comp_K4_log_spaced,
  comp_K2, comp_K8, struct_K4
- **Dataset:** TinyStories (5M tokens)
- **GPU:** A100 40GB (~2-3h per arm, pilot 4000 steps, no grad-accum)

### colab_fock_multixi_h128

Adds Fock-space latent register pools (v1 and v2 gates) on top of the
multi-ξ PARF H=128 architecture. Tests whether M latent register particles
with creation/destruction gates can close the remaining PPL gap between
multi-ξ PARF (12.06 PPL) and the attention baseline (7.81 PPL).

Two gate variants are swept:
- **v1**: Mean-conditioned creation gate (one MLP per layer)
- **v2**: Q/K/V cross-attention creation + optional non-conservative
  reverse channel force Q_i on tokens

Twelve arms cover a full sweep over Fock version (v1/v2), register count
(M=4, 8, 16, 32), activation discipline (LIFO vs free), reverse channel
(on/off), routing density (k=4, 8), V_φ kind (competitive, structural),
channel count (K=2, 4), and training schedule (8k vs 16k steps).

All arms use the same memory optimisations as `colab_parf_multixi_h128`:
Level-2 per-layer checkpointing + Stage-1.5b gathered V_φ, H=128,
no grad-accum.

- **Arms:** v1_K4_M16_lifo, v1_K4_M32_lifo, v1_K4_M16_free,
  v2_K4_M16_lifo, v2_K4_M16_no_rev, v2_K4_M32_lifo, v2_K2_M16_lifo,
  v2_K4_M16_k4, v1_K4_M16_struct, v2_K4_M8_lifo, v2_K4_M4_lifo,
  v2_K4_M16_lifo_16k
- **Dataset:** TinyStories (5M tokens)
- **GPU:** A100 40GB (~2-4h per arm, scaleup 8000 steps, no grad-accum)

---

## Multi-Seed Retrain

| Notebook | Path | Description |
|----------|------|-------------|
| [colab_scope3](#colab_scope3) | `notebooks/conservative_arch/multi_seed/colab_scope3.ipynb` | Scope-3 leak-free retrain of v2 SPLM experiments |

### colab_scope3

Re-runs v2 SPLM-family TinyShakespeare experiments under the v4 leak-free
integrator (`causal_force=True`). Tests whether the SARF xi-recomputation leak
inflated v2 results. Four Phase A cells (splm_baseline, splm_sarf,
splm_sarfmass_embed_head, splm_sarfmass_logfreq) x 5 seeds, plus Phase B
matched_baseline x 5 seeds.

- **Dataset:** TinyShakespeare
- **GPU:** Any Colab GPU (T4 ~3.5h, L4 ~2.5h, A100 ~1.5h for 25 runs)

---

## PARFLM P8-P10 Ladder

| Notebook | Path | Description |
|----------|------|-------------|
| [p8_cell_a100_h100](#p8_cell) | `notebooks/conservative_arch/parf/scripts/p8_cell_a100_h100.ipynb` | P8 composite patches for force-balance and saturation fixes |
| [p10_tinystories_a100_h100](#p10_tinystories) | `notebooks/conservative_arch/parf/scripts/p10_tinystories_a100_h100.ipynb` | P10 ablation ladder targeting val PPL <= 20 on TinyStories |
| [p10_sparsity_ladder_k8_k16](#p10_sparsity) | `notebooks/conservative_arch/parf/scripts/p10_sparsity_ladder_k8_k16.ipynb` | P10 sparsity ladder testing top-k=8 and k=16 vs k=4 ceiling |

### p8_cell

Single composite experiment applying four P8 patches (LN-before-distance,
per-layer V_phi scale, softsign theta, bilinear theta) to fix P6 Layer-1 force
imbalance and theta saturation. Pre-registered predictions: flatter R(l), less
theta saturation, PPL improvement vs P1 structural baseline.

- **Dataset:** TinyShakespeare
- **GPU:** A100/H100 (~25 min A100, ~14 min H100 for 4k steps)

### p10_tinystories

Eight-cell ablation ladder (P10a-P10h) on TinyStories, progressively adding
structural improvements, competitive V_phi, wider V_theta, longer training, and
corpus scale. Isolates contributions of V_phi capacity (P10e), V_theta width
(P10f), training budget (P10g at 16k steps), and corpus scale (P10h at 20M
tokens).

- **Dataset:** TinyStories (5M tokens for P10a-g; 20M for P10h)
- **GPU:** A100/H100 (~6-9h H100, ~16-20h A100 per 8k-step cell)

### p10_sparsity

Extends P10g (k=4, ~26.42 PPL) to test whether the top-k sparsity optimum holds
at TinyStories scale. Two cells: P10i (top_k=8) and P10j (top_k=16), locked to
the P10g architecture.

- **Dataset:** TinyStories (5M tokens)
- **GPU:** A100/H100 (~6-9h per cell on H100)

---

## V_theta Regularisation Sweeps

| Notebook | Path | Description |
|----------|------|-------------|
| [vreg_sweep_v_theta_regularisation](#vreg_splm) | `notebooks/conservative_arch/parf/scripts/vreg_sweep_v_theta_regularisation.ipynb` | Standalone SPLM V_theta regularisation sweep |
| [vreg_sweep_parf](#vreg_parf) | `notebooks/conservative_arch/parf/scripts/vreg_sweep_parf.ipynb` | PARF V_theta regularisation sweep |
| [vreg_sweep_fockparf](#vreg_fockparf) | `notebooks/conservative_arch/parf/scripts/vreg_sweep_fockparf.ipynb` | FockPARF V_theta regularisation sweep |
| [vreg_sweep_hybrid](#vreg_hybrid) | `notebooks/conservative_arch/hybrid/scripts/vreg_sweep_hybrid.ipynb` | Hybrid SPLM+Attention V_theta regularisation sweep |

### vreg_splm

Standalone SPLM (ScalarPotentialLMSARFMass) V_theta regularisation sweep. Breaks
gauge symmetry of unbounded V_theta. Six cells (VR0-VR5) sweeping lambda_V from
0 to 1, plus a Verlet integrator variant (VR5 with L=16, dt=0.5). Includes
post-training attractor extraction and V_theta landscape diagnostics.

- **Cells:** VR0 (lambda=0), VR1 (1e-6), VR2 (1e-4), VR3 (1e-2), VR4 (1), VR5 (Verlet)
- **Dataset:** TinyShakespeare
- **GPU:** Any CUDA (4k steps each)

### vreg_parf

V_theta regularisation sweep for PARF (pair interactions + V_phi). Tests whether
V_phi absorbs expressivity lost when penalising V_theta. Five cells (PR0-PR4)
sweeping lambda_V from 0 to 1, comparing val PPL, V_theta/V_phi energy split, and
attractor structure.

- **Cells:** PR0 (lambda=0), PR1 (1e-6), PR2 (1e-4), PR3 (1e-2), PR4 (1)
- **Dataset:** TinyShakespeare
- **GPU:** Any CUDA (4k steps each)

### vreg_fockparf

V_theta regularisation sweep for FockPARF (registers + sparse routing + Gumbel
tau annealing). Hypothesis: FockPARF is most tolerant of V_theta reg via V_phi
and register lifecycle. Five cells (FR0-FR4) with M=16 registers, d=128, L=8.

- **Cells:** FR0 (lambda=0), FR1 (1e-6), FR2 (1e-4), FR3 (1e-2), FR4 (1)
- **Dataset:** TinyShakespeare
- **GPU:** Any CUDA (4k steps each)

### vreg_hybrid

V_theta regularisation sweep for Hybrid SPLM+Attention architecture. Tests
whether attention layers absorb the PPL cost of V_theta reg, making it effectively
"free". Five cells (HR0-HR4) with n_attn=4, n_splm=4.

- **Cells:** HR0 (lambda=0), HR1 (1e-6), HR2 (1e-4), HR3 (1e-2), HR4 (1)
- **Dataset:** TinyShakespeare
- **GPU:** Any CUDA (4k steps each)

---

## Structured V_theta Experiments

| Notebook | Path | Description |
|----------|------|-------------|
| [structured_vtheta_sweep](#svtheta_sweep) | `notebooks/conservative_arch/parf/scripts/structured_vtheta_sweep.ipynb` | Structured V_theta expressivity test (SQ1-SQ5) on TinyShakespeare |
| [structured_vtheta_tinystories_sweep](#svtheta_tinystories) | `notebooks/conservative_arch/parf/scripts/structured_vtheta_tinystories_sweep.ipynb` | Structured V_theta landscape sweep (SQ1-SQ4 + MLP) on TinyStories |

### svtheta_sweep

Tests whether structured V_theta (analytical gradient, fewer params) can match the
PR2 MLP baseline (~186 PPL, K*=4 basins) on TinyShakespeare. Five cells: SQ1
(diagonal quadratic), SQ2 (low-rank rank=8), SQ3 (mixture K=4), SQ4 (hybrid
quad+MLP), SQ5 (MLP reference). Compares analytical attractors vs gradient-descent
extraction.

- **Cells:** SQ1-SQ5
- **Dataset:** TinyShakespeare
- **GPU:** Any CUDA (4k steps each)

### svtheta_tinystories

Systematic comparison of structured V_theta variants vs MLP baseline at TinyStories
scale. Twelve cells covering SQ1, SQ2 (rank 4 and 16), SQ3 (K=4, 8, 16 with tau
variations), SQ4 (small and large MLP), and two MLP baselines (v_hidden=2048 and
512). Fixed SparsePARFLM backbone (d=256, L=8); only V_theta is swapped. Includes
CUDA step-timing benchmark and cross-cell comparison dashboard.

- **Cells:** A1-A10, B1, B2
- **Dataset:** TinyStories (5M tokens)
- **GPU:** Any CUDA (16k steps each)

---

## PARFLM Scaling Optimisations

Proof-of-concept notebooks validating engineering techniques to reduce
PARFLM's GPU memory footprint.  These address the V_φ `create_graph=True`
memory bottleneck catalogued in the paper's OOM forensics table (§17,
Table 28).  Both notebooks are self-contained (no Colab clone cell) and
auto-detect CUDA/MPS/CPU.

| Notebook | Path | Description |
|----------|------|-------------|
| [gradient_checkpoint_poc](#grad_ckpt_poc) | `notebooks/conservative_arch/parf/scripts/gradient_checkpoint_poc.ipynb` | Level 2 per-layer-step gradient checkpointing POC |
| [gradient_checkpoint_gathered_vphi_poc](#grad_ckpt_gathered_poc) | `notebooks/conservative_arch/parf/scripts/gradient_checkpoint_gathered_vphi_poc.ipynb` | Integrated Level 2 checkpointing + Stage-1.5b gathered V_φ POC |

### grad_ckpt_poc

Validates that per-layer-step checkpointing with `use_reentrant=False`
produces correct parameter gradients for a PARF-like model whose
`_layer_step` contains `autograd.grad(create_graph=True)`.  Three modes
tested: no checkpointing (baseline), Level 1 V_φ-only checkpoint, and
Level 2 full layer-step checkpoint.  Compares parameter gradients across
all modes (< 10⁻⁵ relative error) and measures peak GPU memory.

Level 2 reduces peak V_φ activation memory from O(L·B·T²·H) to
O(B·T²·H) — constant in the number of layers L — at ~50% wall-clock
overhead per training step.

- **Experiments:** Gradient correctness validation + GPU memory measurement
- **Design doc:** `companion_notes/Gradient_Checkpointing_for_PARF.md`
- **GPU:** Any CUDA for correctness; A100/H100 for meaningful memory numbers

### grad_ckpt_gathered_poc

Validates the compound effect of Level 2 checkpointing **and** Stage-1.5b
gathered V_φ (top-k source gathering before V_φ forward).  Four modes
tested in a 2×2 grid: (dense vs gathered) × (no checkpoint vs layer
checkpoint).  Includes a minimal score head for Gumbel top-k routing.

The integrated design reduces peak V_φ activation memory from
O(L·B·T²·H) to O(B·T·k·H) — constant in L, linear in T, proportional
to k rather than T.  At B=16, T=512, L=8, k=4: ~10–14 GB → ~8 MB
(~1000× reduction).

- **Experiments:** 4-mode gradient correctness + 4-mode GPU memory comparison
- **Design docs:** `companion_notes/Gradient_Checkpointing_for_PARF.md`,
  `companion_notes/PARF_Stage_1_5b_design.md`
- **GPU:** Any CUDA for correctness; A100/H100 for meaningful memory numbers

---

## FockPARF v2 Development

| Notebook | Path | Description |
|----------|------|-------------|
| [fockparf_v2_tinystories_debug](#fock_debug) | `notebooks/conservative_arch/parf/scripts/fockparf_v2_tinystories_debug.ipynb` | FockPARF v2 debug/tuning on TinyStories |
| [fockparf_v2_qft_improvements](#fock_qft) | `notebooks/conservative_arch/parf/scripts/fockparf_v2_qft_improvements.ipynb` | QFT-motivated FockPARF v2.1 creation-gate ablation |
| [fockparf_v2_dyck2_falsifier](#fock_dyck2) | `notebooks/conservative_arch/parf/scripts/fockparf_v2_dyck2_falsifier.ipynb` | FockPARF v2 falsifier on synthetic Dyck-2 |
| [fockparf_improvement_sweep](#fock_improvement) | `notebooks/conservative_arch/parf/scripts/fockparf_improvement_sweep.ipynb` | FockPARF improvement sweep to close the PPL gap vs attention |

### fock_debug

Debug and tune FockPARF v2 on TinyStories before full P10g scale. Tests six
bottleneck hypotheses (register compression, shared gate, salience staleness,
register-token path, reverse channel, V_theta ceiling) with deep per-layer
diagnostics. Nine cells (D1-D9) varying register count (M=16/32), query width
(d_k=64/128), blend scale, and tau_create temperature.

- **Cells:** D1-D9
- **Dataset:** TinyStories (1M tokens subset)
- **GPU:** T4 or A100 (~20-40 min per arm)

### fock_qft

Validates four QFT-motivated FockPARF v2.1 creation-gate improvements: Gumbel-Softmax
creation, per-register W_K, orthogonal W_Q init, and canonical destruction. Nine cells
(Q0-Q8) from D1 baseline replica through incremental additions to full QFT v2.1, plus
a SparsePARFLM control.

- **Cells:** Q0-Q8
- **Dataset:** TinyStories (1M tokens subset)
- **GPU:** A100/H100 (~20-40 min per arm)

### fock_dyck2

Three-arm Dyck-2 comparison for FockPARF v2 vs v1 vs PARF baseline on a synthetic
bracket-matching task. Success criterion: FockPARF v2 achieves >50% deep-test accuracy
at depth 5-12. Small model (d=64, L=4) with synthetic data.

- **Cells:** F2-baseline, F2-fock-v1, F2-fock-v2
- **Dataset:** Synthetic Dyck-2 (20k train, 2k val, 2k deep test)
- **GPU:** Any CUDA (4k steps, small model)

### fock_improvement

Eight cells targeting the ~40 PPL gap between FockPARF+lambda_V=1 (~190 PPL) and
attention (~150) on TinyShakespeare. Strategies: hybrid FockPARF+Attention (P1),
wider V_theta (P2), more registers (P3), width scale-up (P4), phased gates (P5),
and G-series gamma/v_hidden confound resolution (G1-G3).

- **Cells:** P1-P5, G1-G3
- **Dataset:** TinyShakespeare
- **GPU:** Any CUDA (4k-8k steps per arm)

---

## Architecture Comparison

| Notebook | Path | Description |
|----------|------|-------------|
| [tinystories_parf_vs_fockparf (top-level)](#ts_compare_top) | `notebooks/conservative_arch/tinystories_parf_vs_fockparf.ipynb` | TinyStories v3 scale-up: PARF vs FockPARF vs Hybrids with SQ3 V_theta |
| [tinystories_parf_vs_fockparf (scripts)](#ts_compare_scripts) | `notebooks/conservative_arch/parf/scripts/tinystories_parf_vs_fockparf.ipynb` | Same notebook (scripts location) |

### ts_compare_top

TinyStories scale-up v3: replaces MLP V_theta with structured
MixtureQuadraticVTheta (SQ3, K=4) across four architectures at d=256. Compares
regularised PARF (S1), regularised FockPARF with M=32 registers (S2), hybrid
FockPARF+Attention (S3), and hybrid SPLM+Attention reference (S4). Run S3 first.

- **Cells:** S1-S4
- **Dataset:** TinyStories (5M tokens)
- **GPU:** Any CUDA (16k steps, d=256)

### ts_compare_scripts

Same notebook as above, located under `parf/scripts/` for organisational
convenience. Identical cells and experiments.

---

## Non-Conservative / SP-HSPLM

| Notebook | Path | Description |
|----------|------|-------------|
| [sp_hsplm_stage1](#sphsplm_s1) | `notebooks/conservative_arch/non_conservative/scripts/sp_hsplm_stage1_a100_h100.ipynb` | SP-HSPLM Stage 1: per-token Class B/C force rerun |
| [sp_hsplm_stage2](#sphsplm_s2) | `notebooks/conservative_arch/sphsplm/scripts/sp_hsplm_stage2_a100_h100.ipynb` | SP-HSPLM Stage 2: Q9(e) pair-skew cell ladder |

### sphsplm_s1

Pre-registered Stage 1 protocol testing whether per-token non-conservative terms
(gyroscopic and solenoidal forces) close the SPLM-vs-attention gap on TinyStories.
Six cells: e0_baseline (em_ln comparator), e1_const_skew (full-rank gyroscopic),
e2_affine_rank1, e3_lowrank_rank2, e4_solenoidal_rank4, e5_lowrank_rank4. Includes
causal-leak probe and nonconservative norm diagnostics.

- **Cells:** e0_baseline, e1-e5
- **Dataset:** TinyStories (5M tokens)
- **GPU:** A100/H100 (~6-9h per cell on H100)

### sphsplm_s2

Pre-registered Stage 2 protocol with the Q9(e) pair-skew cell ladder. Fourteen
cells from q9e_a through q9e_n, covering mechanism-2 autonomous force law,
mechanism-1 extensions (per-layer J_phi, V_phi, alpha_phi), additivity tests,
maximal mechanism-1, and full Class F. Compares against SparsePARFLM P10g (26.42
PPL) and Stage 1 E4-fix (24.58 PPL).

- **Cells:** q9e_a through q9e_n
- **Dataset:** TinyStories (5M tokens)
- **GPU:** H100/A100 (16k steps per cell)

---

## Dynamics Order Tests

| Notebook | Path | Description |
|----------|------|-------------|
| [experiment_a_trajectory_fitting](#exp_a_traj) | `notebooks/dynamics_order_test/scripts/experiment_a_trajectory_fitting.ipynb` | First-order vs second-order trajectory fitting on GPT-2 |
| [experiment_a_per_layer_sweep](#exp_a_layer) | `notebooks/dynamics_order_test/scripts/experiment_a_per_layer_sweep.ipynb` | Per-layer ODE trajectory fitting across all GPT-2 layers |

### exp_a_traj

Tests four physics-structured ODE models for predicting GPT-2 last-layer hidden-state
trajectories: M1 (1st-order gradient descent on V), M2 (2nd-order damped Verlet),
M3 (general lag-1 MLP), M4 (general lag-2 MLP). Evaluates single-step R-squared and
multi-step rollout (K=1,3,5,8,10). Includes paired Wilcoxon tests.

- **Dataset:** Custom 50-sentence corpus (5 domains) with pretrained GPT-2
- **GPU:** Colab T4 sufficient (~20 min)

### exp_a_layer

Extends Experiment A by fitting M1-M4 independently at each of GPT-2's 13 layers
(embedding + 12 transformer layers). Tests the hypothesis that early layers show
positive R-squared while middle layers are worst (bathtub profile).

- **Dataset:** Custom 50-sentence corpus (5 domains) with pretrained GPT-2
- **GPU:** Colab A100 recommended (~30 min, 13 layers)

---

## Local-Only Notebooks

The following notebooks do **not** have Colab setup (no `REPO_URL`, no
`google.colab` imports). They run locally and auto-detect CUDA/MPS/CPU.

| Notebook | Path | Description |
|----------|------|-------------|
| e_init_validation | `notebooks/e_init/e_init_validation.ipynb` | Forward integration of the Euler-Lagrange equation from GPT-2 first-block initial conditions. Fits per-layer Gaussian-well potentials and tests symplectic Euler integration with gamma calibration. |
| pythia_tangential_acceleration | `notebooks/cross_model/pythia_tangential_acceleration.ipynb` | Cross-model tangential acceleration test comparing GPT-2 small vs Pythia-160M. Replicates deceleration statistics for cross-model robustness. |
| energy_landscape_validation | `notebooks/stp_loss/energy_landscape_validation.ipynb` | Validates the Gaussian-well energy landscape hypothesis in GPT-2 hidden-state space. Tests potential fits, Lagrangian reconstruction, and acceleration-STP equivalence. |
| energy_landscape_validation_executed | `notebooks/stp_loss/energy_landscape_validation_executed.ipynb` | Same as above with frozen execution outputs (cell outputs preserved). |
