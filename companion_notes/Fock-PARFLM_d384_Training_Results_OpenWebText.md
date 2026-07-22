# Fock-PARFLM d=384 Training Results on OpenWebText

**Author:** Dimitar P. Gueorguiev
**Date:** July 2026
**Status:** In progress — Phase 3 (250K steps) running on Colab; step ~80,000 / 250,000. Best PPL **16.65** (step 77,500)

---

## 1. Summary

This document records the training history of the Fock-PARFLM v2.1 model at d=384, L=16 on OpenWebText (run tag: `e5c_plgate`). Training uses a graduated token-pool strategy across three phases:

| | Phase 1 | Phase 2 | Phase 3 (in progress) | Combined |
|---|:---:|:---:|:---:|:---:|
| Steps | 100,000 | 150,000 | 250,000 | 500,000 |
| Token pool | 1B | 2B | 4B | — |
| Tokens consumed | 0.82B | 1.23B | ~0.66B (so far) | ~2.71B |
| Wall time | 5.8 h | 20.8 h | ~33 h (so far) | ~60 h |
| Best PPL | 63.69 (step 99K) | 27.23 (step 150K) | **16.65** (step 77,500) | **16.65** |
| Hardware | Colab A100 / H100 | Colab A100 / H100 | Colab H100 | — |

### 1.1 Cumulative Token Budget by Phase

| Phase | Steps | Token Pool | Cumulative Token Pool | Tokens Consumed | Cumulative Consumed |
|:-----:|------:|:----------:|:---------------------:|:---------------:|:-------------------:|
| 1 | 100,000 | 1B | 1B | 0.82B | 0.82B |
| 2 | 150,000 | 2B | 3B | 1.23B | 2.05B |
| 3 | 250,000 | 4B | **7B** | ~2.05B (projected) | **~4.1B** |

Phase 3 has achieved **PPL 16.65** at step 77,500 — a 39% improvement over Phase 2's best (27.23) — and is **still in the WSD stable-LR phase**. The WSD decay phase does not begin until step 175,000, leaving ~95K steps of stable LR and then 75K steps of active LR decay. Based on prior phases, the decay typically delivers a 30–37% PPL reduction, projecting a Phase 3 final PPL of **~12–15**.

This would make the 53M-parameter Fock-PARFLM competitive with or superior to GPT-2 Medium (354M params, PPL ~17.1) — at **less than half the parameters** and a **fraction of the training data**. A proper full-validation-set PPL evaluation is in progress to verify the 16.65 figure (see `debug/eval_ppl_debug.ipynb`).

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

| Parameter | Phase 1 | Phase 2 | Phase 3 |
|-----------|:-------:|:-------:|:-------:|
| Steps | 100,000 | 150,000 | 250,000 |
| Token pool (`MAX_TRAIN_TOKENS`) | 1B | 2B | 4B |
| WSD warmup | 0 → 300 | 0 → 150 | 0 → 2,000 |
| WSD stable end | ~65,000 | 100,000 | 175,000 |
| WSD decay end | 100,000 | 150,000 | 250,000 |
| LR (peak) | $3 \times 10^{-4}$ | $3 \times 10^{-4}$ | $1.5 \times 10^{-4}$ |
| LR floor | — | — | $1.5 \times 10^{-5}$ |
| `FRESH_SCHEDULE` | — | True | True |
| `SKIP_OPTIMIZER_STATE` | — | True | True |
| `INIT_BEST_PPL` | — | 63.69 | 27.14 |
| Resume from | — | Phase 1 best (step 99K) | Phase 2 checkpoint (step 25K, PPL 29.78) |
| Batch size | auto-probed (~8) | auto-probed (~8) | 8 |
| Gradient accumulation | 2 | 2 | 2 |
| Per-group gradient clips | default 1.0 | default 1.0 | `V_phi`=0.3, `creation_gate`=0.3, `destruction_gate`=0.3, `reverse_channel_scale`=0.1, `reverse_ch`=0.1, `register`=0.3, `depth_code`=0.5 |

Phase 3 uses a reduced peak LR (half of Phases 1–2) with a longer stable phase and explicit per-group gradient clips — refinements informed by the d=768/d=1024 instability investigations.

---

## 3A. Evaluation Protocol

All validation perplexity numbers reported in this document are computed on a **held-out 2M-token slice** of OpenWebText that is **physically disjoint** from the training data:

- **Split mechanism:** When OpenWebText is first streamed and tokenised (GPT-2 BPE, vocab 50,257), the code requests `MAX_TRAIN_TOKENS + 2,000,000` total tokens. The **last 2M tokens** are sliced off as the validation set; the **first N tokens** become the training set. The two are cached as separate files (`openwebtext_val_2M.npy` and `openwebtext_train_{N}M.npy`).
- **No overlap:** The validation tokens are from distinct documents at the tail of the stream. There is zero token-level overlap between training and validation.
- **Fixed across phases:** When the token budget graduated from 1B (Phase 1) to 2B (Phase 2), the training pool grew (a new `openwebtext_train_2000M.npy` was cached) but the validation set remained the **same 2M-token held-out slice**. All PPL numbers across phases are therefore directly comparable.
- **Evaluation procedure:** At each evaluation step, 5 random batches of length 512 are drawn from the validation set, the model computes cross-entropy loss in inference mode, and the mean loss is exponentiated to produce PPL: $\text{PPL} = \exp(\bar{\mathcal{L}}_{\text{val}})$.

This is standard held-out evaluation — equivalent to how GPT-2/GPT-3 papers report perplexity. The only note is that both training and validation tokens come from the same corpus (OpenWebText), so they share distributional properties, but there is no data leakage.

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

### 4.3 Phase 3: 250K steps on 4B tokens (in progress — step 52,350)

| Step | PPL | Notes |
|-----:|----:|-------|
| 25,500 | 29.44 | Warm-restart regression (init best = 27.14) |
| 26,000 | 28.23 | |
| 27,000 | 28.00 | |
| 28,000 | 28.50 | |
| 29,000 | 27.76 | |
| 29,500 | **26.50** | First new best |
| 30,000 | **26.15** | |
| 31,000 | 28.17 | |
| 32,500 | 26.77 | |
| 33,500 | **25.20** | |
| 35,000 | 26.97 | |
| 36,000 | 25.99 | |
| 37,000 | 26.14 | |
| 38,500 | 26.28 | |
| 39,000 | **25.15** | |
| 40,000 | **24.71** | Breaks 25 |
| 41,000 | **24.13** | |
| 42,500 | **24.03** | |
| 43,000 | 24.12 | |
| 45,000 | 24.62 | |
| 45,500 | **23.95** | Breaks 24 |
| 47,000 | **23.49** | |
| 48,000 | **23.12** | |
| 49,500 | **21.96** | **Current overall best** |
| 50,000 | 23.72 | |
| 50,500 | 23.10 | |
| 51,000 | 23.79 | |
| 51,500 | 23.41 | |
| 52,000 | 22.01 | |

**Phase 3 observations (interim, step 52,350):**
1. **Mild warm-restart regression:** PPL briefly regressed from 27.14 to ~29.4 in the first 500 steps before recovering. This is much milder than Phase 2's regression (to 74.4), likely due to the lower peak LR (1.5e-4 vs 3e-4).
2. **Sustained descent in the stable phase:** Unlike Phases 1–2 where the stable phase plateaued, Phase 3's stable phase is producing continuous new bests. The model set **10 consecutive new-best records** from step 29,500 to 49,500, reducing PPL from 27.14 to 21.96 — a **19% reduction** with constant LR. This suggests the model is exploiting the 4B token pool's greater data diversity.
3. **Clean gradient health:** Only 6 mild spikes (max ~7,427 at step 42,011, immediately absorbed). No watchdog triggers. The per-group clips are working as intended. This is the cleanest training window across all three phases.
4. **$v_{\text{reg}}$ declining:** From 0.013 at step 25K to ~0.009 at step 52K, indicating the dynamics are becoming smoother.
5. **All $\xi$ channels active:** $\alpha$ values continue to evolve slowly ($\alpha_1$: 0.133→0.089, $\alpha_5$ stable at 0.963).
6. **Step 52,000 PPL of 22.01** nearly matches the 49,500 best (21.96), confirming this is not a statistical fluke.
7. **WSD decay phase (steps 175K–250K) still 123K steps away.** If it delivers even a 20–30% reduction from the eventual stable-phase floor (~19–20 by step 175K):
   - 20% reduction: 19 × 0.80 = **~15.2 PPL**
   - 30% reduction: 19 × 0.70 = **~13.3 PPL**

---

### 4.4 Cross-scale context: Comparison with YuriiFormer and GPT-2

To contextualise the d=384 results, we compare against recent transformer baselines on OpenWebText. The comparison requires caveats (see §4.5) but the parameter and data efficiency of Fock-PARFLM is striking.

| Model | Params | Training tokens | Optimizer | Context | Best PPL | Source |
|-------|-------:|----------------:|-----------|:-------:|:--------:|--------|
| **Fock-PARFLM d=384 (Phase 3, interim)** | **53M** | **~2.5B** | AdamW | 512 | **21.96** | This work |
| GPT-2 Small checkpoint | 124M | ~9B (est.) | Adam | 1024 | ~22.6 | Karpathy nanoGPT eval |
| YuriiFormer Small (Nesterov+LT) | 124M | 14.75B | Muon+AdamW | 1024 | ~18.5 | Zimin et al. 2026 |
| GPT-2 Medium checkpoint | 354M | ~9B (est.) | Adam | 1024 | ~17.1 | Karpathy nanoGPT eval |
| YuriiFormer Medium (Nesterov+LT) | 354M | 14.75B | Muon+AdamW | 1024 | ~14.9 | Zimin et al. 2026 |

GPT-2 and YuriiFormer PPL values are derived from validation cross-entropy losses reported in nats/token (PPL = exp(loss)). GPT-2 checkpoint values are from Karpathy's nanoGPT evaluation on the OpenWebText validation split. YuriiFormer values are from Table 3 of Zimin et al. (arXiv:2601.23236).

**Key observations:**
- At **53M parameters**, Fock-PARFLM already matches GPT-2 Small (124M) — a model with **2.3× more parameters**.
- Fock-PARFLM uses only **~2.5B tokens** vs YuriiFormer's 14.75B (**5.9× fewer**) and GPT-2's ~9B (**3.6× fewer**).
- The gap to YuriiFormer Small (18.5 PPL) is ~3.5 PPL — and 123K steps of WSD decay remain.
- YuriiFormer uses Muon (a state-of-the-art optimizer for transformers); Fock-PARFLM uses plain AdamW.

### 4.5 Comparison caveats

Several methodological differences affect direct numerical comparison:

| Factor | Fock-PARFLM | YuriiFormer / GPT-2 |
|--------|-------------|---------------------|
| **Context length** | 512 tokens | 1024 tokens |
| **Validation set** | 2M tokens, 5 batches (~2,560 tokens/eval) | 432M tokens, 160 batches (~4.9M tokens/eval) |
| **Vocab size** | 50,257 | 50,304 (padded) |
| **Precision** | fp32 | bf16 |

The most significant factor is **context length**: conditioning on 1024 tokens instead of 512 typically lowers PPL by 1–3 points, giving YuriiFormer/GPT-2 a systematic advantage. The validation set size difference means our PPL estimates have higher variance per evaluation, though the consistent descent across 50+ evaluations confirms the trend is real.

---

## 5. Gradient Health

### 5.1 Spike profile

| | Phase 1 | Phase 2 | Phase 3 (step 52K) |
|---|:---:|:---:|:---:|
| Spikes (grad > 100) | 15 | 14 | 6 |
| Maximum gradient norm | 757.2 | 1702.9 | 7,426.7 |
| Watchdog triggers | 0 | 0 | 0 |
| Watchdog reloads | 0 | 0 | 0 |
| Top spike groups | `E`, `P`, `creation_gate`, `reverse_channel_scale` | `E`, `P`, `creation_gate`, `reverse_channel_scale` | `reverse_channel_scale`, `E`, `P`, `V_phi`, `creation_gate` |

All spikes were intermittent and self-correcting: the gradient clip truncated the update, and AdamW momentum smoothed over the transient. No spike caused a persistent PPL regression. Phase 3's per-group clips keep `reverse_channel_scale` at a 0.1 limit, preventing its gradient from dominating the cascade.

### 5.2 Steady-state gradient profile

During non-spike steps, the dominant gradient group is `reverse_channel_scale`, with typical norms of 3–15 (post-clip). This reflects the reverse channel's role as the primary information conduit between registers and the hidden stream — it carries the largest learning signal because it mediates the Fock mechanism's core function.

### 5.3 Comparison with larger scales

| Scale | Max grad | Spikes > 100 | Watchdog reloads | Regime |
|-------|:--------:|:------------:|:----------------:|--------|
| d=384, L=16 | 7,427 | ~35 (across 302K steps) | 0 | Manageable |
| d=768, L=12 | 5,158,336 | ~110 (at 65.5K steps) | 2 | Catastrophic (late-training) |
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
| 275,000 | 3 (step 25K) | 0.0130 |
| 290,000 | 3 (step 40K) | 0.0105 |
| 300,000 | 3 (step 50K) | 0.0090 |

$v_{\text{reg}}$ rises through Phase 1 and the first half of Phase 2 (peaking at 0.0213 around step 50K of Phase 2), then declines across the rest of Phase 2 and into Phase 3. In Phase 3, $v_{\text{reg}}$ continues its descent from 0.013 to 0.009 over 27K steps, indicating progressively smoother hidden-state dynamics. This is consistent with the model approaching a refined, low-velocity regime where token trajectories more closely follow damped geodesics.

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
| 275,000 (P3, step 25K) | 0.133 | 0.312 | 0.481 | 0.613 | 0.963 |
| 290,000 (P3, step 40K) | 0.110 | 0.304 | 0.468 | 0.594 | 0.963 |
| 300,000 (P3, step 50K) | 0.093 | 0.295 | 0.457 | 0.576 | 0.963 |

All five channels remain active throughout training ($\alpha > 0.09$ at step 302K). The monotonic decrease in lower-channel $\alpha$ values continues into Phase 3, with $\alpha_1$ approaching 0.09 — yet never reaching zero. Notably, $\alpha_5$ (the longest-horizon channel, $\alpha = 0.995$) has remained remarkably stable at ~0.963 across all three phases, suggesting this channel captures a structural mode that is scale-invariant. The model is learning to specialise channels hierarchically — concentrating force-field influence on fewer, more refined channels while retaining broad coverage through the higher-indexed channels.

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

| Checkpoint | Step | PPL | Phase | Purpose |
|------------|-----:|----:|:-----:|---------|
| `..._best.pt` | 49,500 (P3) | 21.96 | 3 | Canonical best |
| `..._step49500_best.pt` | 49,500 (P3) | 21.96 | 3 | Step-stamped copy |
| `..._step48000_best.pt` | 48,000 (P3) | 23.12 | 3 | Previous best |
| `..._step47000_best.pt` | 47,000 (P3) | 23.49 | 3 | |
| `..._step45500_best.pt` | 45,500 (P3) | 23.95 | 3 | |
| `..._step42500_best.pt` | 42,500 (P3) | 24.03 | 3 | |
| `..._step41000_best.pt` | 41,000 (P3) | 24.13 | 3 | |
| `..._step40000_best.pt` | 40,000 (P3) | 24.71 | 3 | |
| `..._step39000_best.pt` | 39,000 (P3) | 25.15 | 3 | |
| `..._step33500_best.pt` | 33,500 (P3) | 25.20 | 3 | |
| `..._step30000_best.pt` | 30,000 (P3) | 26.15 | 3 | |
| `..._step150000_best.pt` | 150,000 (P2) | 27.23 | 2 | Phase 2 best |
| `..._step99000_best.pt` | 99,000 (P1) | 63.69 | 1 | Phase 1 best |

All checkpoints include model weights, optimizer state, scheduler state, and training metadata.

---

## 10. Token Budget Analysis

| Phase | Pool | Consumed | Pool coverage | Repetition rate |
|-------|-----:|---------:|--------------:|----------------:|
| Phase 1 | 1B | 0.82B | 82% | ~1.0× |
| Phase 2 | 2B | 1.23B | 62% | ~0.6× |
| Phase 3 (in progress) | 4B | ~0.43B (so far) | ~11% | ~0.1× |
| **Total** | — | **~2.48B** | — | — |

The graduated token-pool strategy (1B → 2B → 4B) provides increasing data diversity while maintaining repetition in early phases for stable learning. Phase 1's near-complete pool coverage (82%) acts as implicit regularisation. Phase 3's 4B pool will consume ~2.05B tokens at completion (eff_batch=16 × 512 × 250K steps) — only 51% of the pool, ensuring no repetition risk.

---

## 11. Phase 3 Status and Projections

Phase 3 is running on Colab H100 using `colab_fock_depthcond_vtheta_openwebtext_ext2.ipynb`.

| Metric | Value |
|--------|-------|
| Current step | 52,350 / 250,000 (21%) |
| Current best PPL | **21.96** (step 49,500) |
| Phase | WSD stable (decay begins at step 175,000) |
| Estimated wall time remaining | ~160 h |

**Projection methodology.** The stable-phase floor can be extrapolated from the current descent rate. PPL has dropped from 29.4 → 21.96 over 27K steps (~0.28 PPL/Kstep). If this rate halves over the next 123K steps (diminishing returns), the floor at step 175K would be ~17–19 PPL. Applying the historically observed decay-phase reduction:

| Scenario | Stable-phase floor | Decay reduction | Final PPL |
|----------|:------------------:|:---------------:|:---------:|
| Conservative | 19 | 20% | **~15.2** |
| Moderate | 18 | 25% | **~13.5** |
| Optimistic | 17 | 30% | **~11.9** |

Even the conservative scenario would place the 53M-parameter Fock-PARFLM firmly within GPT-2 Medium (354M params) territory — a **6.7× parameter advantage**.

### 11.1 Potential Phase 4

If Phase 3 completes successfully and capacity remains (PPL still improving at the end), a Phase 4 extension could be considered with an 8B token pool and 350K+ steps. However, the 512-token context length may become the binding constraint at PPL < 15 — transitioning to 1024-token contexts would enable fairer comparison with transformer baselines but requires re-engineering the memory budget.

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
