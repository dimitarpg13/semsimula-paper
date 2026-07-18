# Fock-PARFLM d=384 Training Results on OpenWebText

**Author:** Dimitar P. Gueorguiev
**Date:** July 2026
**Status:** Complete — Phase 1 (100K steps) and Phase 2 (150K steps) finished; Phase 3 (250K steps) planned

---

## 1. Summary

This document records the complete training history of the Fock-PARFLM v2.1 model at d=384, L=16 on OpenWebText (run tag: `e5c_plgate`). Training was conducted across two phases using a graduated token-pool strategy:

| | Phase 1 | Phase 2 | Combined |
|---|:---:|:---:|:---:|
| Steps | 100,000 | 150,000 | 250,000 |
| Token pool | 1B | 2B | — |
| Tokens consumed | 0.82B | 1.23B | 2.05B |
| Wall time | 5.8 h | 20.8 h | 26.6 h |
| Best PPL | 63.69 (step 99K) | **27.23** (step 150K) | **27.23** |
| Hardware | Colab A100 / H100 | Colab A100 / H100 | — |

The model achieved **PPL 27.23** — a new best for Fock-PARFLM at any scale — on the final evaluation step, indicating that capacity was not exhausted and further extension is warranted.

---

## 2. Architecture

| Parameter | Value |
|-----------|-------|
| Model | Fock-PARFLM v2.1 (untied embeddings) |
| Hidden dimension ($d$) | 384 |
| Depth ($L$) | 16 |
| Total parameters | 53,378,075 (~53M) |
| Integrator | Velocity-Verlet with O-step Langevin friction |
| Damping ($\gamma$) | 0.30 (fixed) |
| $V_\theta$ | Depth-conditioned multi-context Gaussian wells (5 heads x 8 wells) |
| $V_\phi$ | Structural-competitive pairwise potential |
| $\xi$ channels | 5 ("5long" override) |
| Registers | 32 |
| Fock mechanism | Per-layer creation/destruction gates + reverse channel (per-layer gated) |
| Embeddings | Untied (`E` separate from `lm_head`) |
| Output bias | Log-unigram frequency initialised |
| Positional encoding | Learned (`P`, 1024 x 384) |
| $d_k$ (register attention) | 64 |
| Memory optimisations | `use_layer_checkpoint=True`, `use_gathered_v_phi=True` |

See [Fock-PARFLM\_Scale-Up\_Comparative\_Experiments.md](Fock-PARFLM_Scale-Up_Comparative_Experiments.md) §2.1 for the full per-component parameter breakdown.

---

## 3. Training Configuration

### 3.1 Common settings (both phases)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW ($\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$) |
| Weight decay | 0.01 |
| Sequence length (`BLOCK_SIZE`) | 512 |
| Effective batch size | 16 (batch x grad\_accum) |
| Learning rate | $3 \times 10^{-4}$ (peak) |
| LR schedule | WSD (Warmup-Stable-Decay) |
| Gradient clip (global) | 1.0 |
| Gradient clip ($V_\phi$) | 0.5 |
| Evaluation interval | 500 steps |
| Evaluation batches | 5 |
| Data | OpenWebText (Skylion007/openwebtext), GPT-2 BPE tokenisation |
| Sampling | Random with replacement from token pool |

### 3.2 Phase-specific settings

| Parameter | Phase 1 | Phase 2 |
|-----------|:-------:|:-------:|
| Steps | 100,000 | 150,000 |
| Token pool (`MAX_TRAIN_TOKENS`) | 1B | 2B |
| WSD warmup | 0 → 300 | 0 → 150 |
| WSD stable end | ~65,000 | 100,000 |
| WSD decay end | 100,000 | 150,000 |
| `FRESH_SCHEDULE` | — | True |
| `SKIP_OPTIMIZER_STATE` | — | True |
| `INIT_BEST_PPL` | — | 63.69 |
| Resume from | — | Phase 1 best checkpoint (step 99K) |
| Batch size | auto-probed (~8) | auto-probed (~8) |
| Gradient accumulation | 2 | 2 |

---

## 4. Learning Curve

### 4.1 Phase 1: 100K steps on 1B tokens

| Step | PPL | Notes |
|-----:|----:|-------|
| 500 | 1481.64 | Initial random performance |
| 1,000 | 990.14 | |
| 5,000 | 251.98 | Rapid early descent |
| 10,000 | 182.39 | |
| 20,000 | 135.97 | |
| 30,000 | 126.40 | |
| 40,000 | 116.44 | |
| 50,000 | 105.70 | Midpoint |
| 60,000 | 101.23 | |
| 70,000 | 92.40 | |
| 80,000 | 78.16 | WSD decay accelerates improvement |
| 90,000 | 72.59 | |
| 99,000 | **63.69** | **Phase 1 best** |
| 100,000 | 68.70 | Slight regression at final step |

**Phase 1 observations:**
- Smooth, monotonic descent from random initialisation (PPL 1481) to PPL 63.69
- The WSD decay phase (steps ~65K–100K) produced the strongest gains: PPL dropped from ~101 to 64 — a 37% reduction in the final 35% of training
- The maximum gradient spike was 757.2 (15 spikes with grad > 100), all recovered without watchdog intervention

### 4.2 Phase 2: 150K steps on 2B tokens (resuming from Phase 1 best)

| Step | PPL | Notes |
|-----:|----:|-------|
| 500 | 68.39 | Warm-restart regression (+7.4% vs Phase 1 best) |
| 5,000 | 73.34 | LR ramping up, temporary regression |
| 10,000 | 74.40 | Peak regression (16.8% above Phase 1 best) |
| 20,000 | 66.53 | Recovery begins |
| 25,000 | 64.38 | Surpasses Phase 1 best |
| 30,000 | 64.85 | |
| 40,000 | 59.07 | |
| 50,000 | 50.79 | |
| 60,000 | 46.88 | |
| 70,000 | 44.36 | |
| 75,000 | 41.85 | Phase 1 WSD stable-end equivalent |
| 100,000 | 41.27 | Stable-phase plateau |
| 101,000 | 37.19 | WSD decay begins — sharp drop |
| 110,000 | 34.23 | |
| 120,000 | 32.60 | |
| 130,000 | 30.86 | |
| 135,000 | 29.00 | |
| 140,500 | 27.90 | |
| 150,000 | **27.23** | **Overall best — NEW BEST on final step** |

**Phase 2 observations:**
1. **Warm-restart regression:** PPL regressed from 63.69 to 74.40 over the first 10K steps as the fresh WSD schedule ramped the LR back to peak. Recovery to Phase 1 levels took ~25K steps.
2. **WSD decay is the learning engine:** the stable phase (steps 0–100K) brought PPL from 68 to 41 (40% reduction). The decay phase (steps 100K–150K) brought PPL from 41 to 27 (34% reduction in 1/3 the steps).
3. **Final-step best:** The model hit its best PPL on the very last evaluation step, demonstrating that capacity was not exhausted. Further extension is justified.
4. **Moderate spikes:** 14 spikes with grad > 100 (max 1702.9), all recovered. No watchdog triggers. This is qualitatively different from the catastrophic spikes at d=768/d=1024 (grad > 60,000).

---

## 5. Gradient Health

### 5.1 Spike profile

| | Phase 1 | Phase 2 |
|---|:---:|:---:|
| Spikes (grad > 100) | 15 | 14 |
| Maximum gradient norm | 757.2 | 1702.9 |
| Watchdog triggers | 0 | 0 |
| Watchdog reloads | 0 | 0 |
| Top spike groups | `E`, `P`, `creation_gate`, `reverse_channel_scale` | `E`, `P`, `creation_gate`, `reverse_channel_scale` |

All spikes were intermittent and self-correcting: the gradient clip truncated the update, and AdamW momentum smoothed over the transient. No spike caused a persistent PPL regression.

### 5.2 Steady-state gradient profile

During non-spike steps, the dominant gradient group is `reverse_channel_scale`, with typical norms of 3–15 (post-clip). This reflects the reverse channel's role as the primary information conduit between registers and the hidden stream — it carries the largest learning signal because it mediates the Fock mechanism's core function.

### 5.3 Comparison with larger scales

| Scale | Max grad | Spikes > 100 | Watchdog reloads | Regime |
|-------|:--------:|:------------:|:----------------:|--------|
| d=384, L=16 | 1,703 | 29 (across 250K steps) | 0 | Manageable |
| d=768, L=12 | 81,019 | multiple | 1+ | Catastrophic (late-training) |
| d=1024, L=16 | 63,949 | frequent | multiple | Catastrophic (early-onset) |

The d=384 model's spike regime is qualitatively different: spikes are moderate (3 orders of magnitude smaller than d=768/d=1024) and never trigger the watchdog. This confirms that the catastrophic second-order gradient cascade (documented in [Training\_Instabilities\_in\_Fock-PARFLM\_with\_structured\_V\_theta.md](Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) §23–25) is a scale-dependent phenomenon that does not manifest at d=384.

---

## 6. Dynamics Diagnostics

### 6.1 $v_{\text{reg}}$ (regularisation loss) trajectory

| Step (cumulative) | Phase | $v_{\text{reg}}$ |
|------------------:|:-----:|:-----------------:|
| 25,000 | 1 | 0.0008 |
| 50,000 | 1 | 0.0025 |
| 75,000 | 1 | 0.0048 |
| 100,000 | 1 | 0.0060 |
| 125,000 | 2 (step 25K) | 0.0119 |
| 150,000 | 2 (step 50K) | 0.0213 |
| 175,000 | 2 (step 75K) | 0.0207 |
| 225,000 | 2 (step 125K) | 0.0162 |
| 250,000 | 2 (step 150K) | 0.0145 |

$v_{\text{reg}}$ rises through Phase 1 and the first half of Phase 2 (peaking at 0.0213 around step 50K of Phase 2), then declines as the model settles into the WSD decay phase. This is the expected pattern: the velocity regulariser penalises large hidden-state displacements, and the model initially explores aggressively (high velocity) before converging to a low-velocity, low-loss regime.

### 6.2 $\xi$ channel utilisation ($\alpha$ trajectory)

The $\xi$ routing mechanism assigns a learned scalar $\alpha_k$ to each of the 5 channels, controlling their contribution to the force field. Values near 1.0 indicate full utilisation; values near 0.0 indicate suppression.

| Step | $\alpha_1$ | $\alpha_2$ | $\alpha_3$ | $\alpha_4$ | $\alpha_5$ |
|-----:|:----------:|:----------:|:----------:|:----------:|:----------:|
| 25,000 (P1) | 0.415 | 0.525 | 0.766 | 0.911 | 0.989 |
| 50,000 (P1) | 0.395 | 0.503 | 0.714 | 0.876 | 0.985 |
| 75,000 (P1) | 0.364 | 0.470 | 0.672 | 0.849 | 0.981 |
| 100,000 (P1) | 0.332 | 0.439 | 0.643 | 0.828 | 0.980 |
| 150,000 (P2) | 0.275 | 0.389 | 0.588 | 0.751 | 0.974 |
| 250,000 (P2) | 0.172 | 0.322 | 0.498 | 0.639 | 0.965 |

All five channels remain active throughout training ($\alpha > 0.17$). The monotonic decrease in lower-channel $\alpha$ values over time suggests the model is learning to specialise channels hierarchically — concentrating force-field influence on fewer, more refined channels while retaining broad coverage through the higher-indexed channels.

---

## 7. Structural Health (Post-Training Probe)

The structural health probe at step 150,000 (PPL 27.23) confirms a well-functioning Fock mechanism:

| Metric | Value | Interpretation |
|--------|:-----:|----------------|
| `active_frac` | 1.000 | All registers active at every layer |
| `salience_mean` | 0.517 | Moderate register influence |
| `reg_cos_sim` | 0.062 | Low redundancy — registers are diverse |
| `destroy_mean` | 0.262 | ~26% destruction rate per layer |
| `qforce_ratio` | 0.222 | Force field contributing ~22% of updates |
| `rev_scale` | 0.027 | Reverse channel active but not dominant |
| `create_alpha_max` | 0.62 | Creation gate partially open |
| `create_entropy` | 1.931 | High entropy — creation is distributed |
| `rev_entropy` | 1.856 | High entropy — reverse channel is distributed |
| `rev_alpha_max` | 0.381 | No single layer dominates reverse flow |

### 7.1 Ablation at step 150K

| Configuration | PPL | $\Delta$ PPL |
|---------------|----:|:------------:|
| Full model | 32.21 | — |
| Without reverse channel | 2085.52 | +2053.31 |
| Without registers | 2085.50 | +2053.29 |

Removing either the reverse channel or the registers produces catastrophic failure (PPL → 2085). This confirms the Fock mechanism is not merely helpful but architecturally essential. See [Fock\_Mechanism\_Ablation\_Study\_d384\_OpenWebText.md](Fock_Mechanism_Ablation_Study_d384_OpenWebText.md) for the detailed ablation analysis, including comparison with the e5a run (reverse channel disabled from scratch, plateau at PPL 125.94).

---

## 8. Gamma Sweep and Geodesic Analysis

A dedicated gamma sweep (8 candidates, 3,000 steps each) with integrated geodesic residual analysis was conducted using `colab_fock_gamma_sweep_geodesic_d384.ipynb`. Full results are documented in [Fock-PARFLM\_Scale-Up\_Gamma\_Sweep\_Results\_and\_Damping\_Regime\_Analysis.md](Fock-PARFLM_Scale-Up_Gamma_Sweep_Results_and_Damping_Regime_Analysis.md) §3. Key findings:

- **Optimal $\gamma$ for PPL:** 0.250 (PPL 342.02 at 3K steps)
- **Optimal $\gamma$ for geodesic residual $\bar{R}$:** 0.050 ($\bar{R}$ = 1.041)
- **PPL-geodesic coincidence:** **Breaks** at d=384 (gap = 0.200)
- **Training $\gamma$ (0.30):** ranks #3 (PPL 353.69), only 3.4% behind optimal

The E5c model's training gamma of 0.30 was near-optimal. The PPL-geodesic coincidence breakdown at d=384 (vs. confirmed coincidence at d=1024) reveals a dimension-dependent phase transition in the optimal dynamical regime. See [Geodesic\_Preservation\_Experiment.md](Geodesic_Preservation_Experiment.md) §4.5–4.6 for the full analysis.

---

## 9. Checkpoints

Retained checkpoints in Google Drive (and local backup):

| Checkpoint | Step | PPL | Purpose |
|------------|-----:|----:|---------|
| `..._best.pt` | 150,000 | 27.23 | Canonical best — use for inference and further extension |
| `..._step150000_best.pt` | 150,000 | 27.23 | Step-stamped copy |
| `..._step140500_best.pt` | 140,500 | 27.90 | Previous best |
| `..._step135000_best.pt` | 135,000 | 29.00 | Decay-phase checkpoint |
| `..._step99000_best.pt` | 99,000 | 63.69 | Phase 1 best |
| `..._step93000_best.pt` | 93,000 | 65.79 | Early Phase 1 checkpoint |

All checkpoints include model weights, optimizer state, scheduler state, and training metadata.

---

## 10. Token Budget Analysis

| Phase | Pool | Consumed | Pool coverage | Repetition rate |
|-------|-----:|---------:|--------------:|----------------:|
| Phase 1 | 1B | 0.82B | 82% | ~1.0× |
| Phase 2 | 2B | 1.23B | 62% | ~0.6× |
| **Total** | — | **2.05B** | — | — |
| Phase 3 (planned) | 4B | ~2.05B | ~51% | ~0.5× |

The graduated token-pool strategy (1B → 2B → 4B) provides increasing data diversity while maintaining repetition in early phases for stable learning. Phase 1's near-complete pool coverage (82%) acts as implicit regularisation — the model sees most sequences more than once, reinforcing early-stage patterns.

---

## 11. Planned Extension: Phase 3

Phase 3 will extend training for 250K additional steps on a 4B token pool, using the `colab_fock_depthcond_vtheta_openwebtext_ext2.ipynb` notebook.

| Parameter | Value |
|-----------|-------|
| Steps | 250,000 |
| Token pool | 4B |
| Resume from | Phase 2 best (step 150K, PPL 27.23) |
| WSD stable end | 175,000 |
| `FRESH_SCHEDULE` | True |
| `INIT_BEST_PPL` | 27.23 |
| Expected warm-restart regression | PPL ~30–40 for first 10–25K steps |
| Projected final PPL | 20–24 (extrapolation from Phase 1→2 trend) |

---

## 12. Data Archive

Training logs are archived in the companion repo:

- `data/d384_e5c_plgate/training_log_phase1.jsonl` — Phase 1 (100K steps, 2171 entries)
- `data/d384_e5c_plgate/training_log_phase1_phase2.jsonl` — Phase 2 (150K steps, 3053 entries)

Each JSONL entry contains: `step`, `train_loss`, `v_reg`, `lr`, `grad_norm`, `gamma`, `xi_alphas`, `reg_repulsion`, `elapsed_sec`, `sec_per_step`. Evaluation entries additionally contain `val_loss` and `val_ppl`.

---

## 13. Cross-References

- [Fock-PARFLM\_Scale-Up\_Comparative\_Experiments.md](Fock-PARFLM_Scale-Up_Comparative_Experiments.md) — parameter counts and architecture comparison with GPT-2
- [Fock\_Mechanism\_Ablation\_Study\_d384\_OpenWebText.md](Fock_Mechanism_Ablation_Study_d384_OpenWebText.md) — ablation study comparing e5c (full Fock) vs e5a (no reverse channel)
- [Training\_Instabilities\_in\_Fock-PARFLM\_with\_structured\_V\_theta.md](Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) — gradient spike analysis and cascade mechanism
- [Fock-PARFLM\_Scale-Up\_Gamma\_Sweep\_Results\_and\_Damping\_Regime\_Analysis.md](Fock-PARFLM_Scale-Up_Gamma_Sweep_Results_and_Damping_Regime_Analysis.md) — gamma sweep results and dimension-dependent phase transition
- [Geodesic\_Preservation\_Experiment.md](Geodesic_Preservation_Experiment.md) — geodesic residual analysis and PPL-geodesic coincidence breakdown
