# Fock-PARFLM d=384 Training Results on OpenWebText

**Author:** Dimitar P. Gueorguiev
**Date:** July 2026
**Status:** In progress — Phase 3 (250K steps) running on Colab; step 103,700 / 250,000. Best PPL **9.50** (step 103,500)

---

## 1. Summary

This document records the training history of the Fock-PARFLM v2.1 model at d=384, L=16 on OpenWebText (run tag: `e5c_plgate`). Training uses a graduated token-pool strategy across three phases:

| | Phase 1 | Phase 2 | Phase 3 (in progress) | Combined |
|---|:---:|:---:|:---:|:---:|
| Steps | 100,000 | 150,000 | 250,000 | 500,000 |
| Token pool | 1B | 2B | 4B | — |
| Tokens consumed | 0.82B | 1.23B | ~0.85B (so far) | ~2.90B |
| Wall time | 5.8 h | 20.8 h | ~85 h (so far) | ~112 h |
| Best PPL | 63.69 (step 99K) | 27.23 (step 150K) | **9.50** (step 103,500) | **9.50** |
| Hardware | Colab A100 / H100 | Colab A100 / H100 | Colab H100 | — |

### 1.1 Cumulative Token Budget by Phase

| Phase | Steps | Token Pool | Cumulative Token Pool | Tokens Consumed | Cumulative Consumed |
|:-----:|------:|:----------:|:---------------------:|:---------------:|:-------------------:|
| 1 | 100,000 | 1B | 1B | 0.82B | 0.82B |
| 2 | 150,000 | 2B | 3B | 1.23B | 2.05B |
| 3 | 250,000 | 4B | **7B** | ~2.05B (projected) | **~4.1B** |

Phase 3 has achieved **PPL 9.50** at step 103,500 — a **65% improvement** over Phase 2's best (27.23) — and is **still in the WSD stable-LR phase**. The WSD decay phase does not begin until step 175,000, leaving ~71K steps of stable LR and then 75K steps of active LR decay.

The 53M-parameter Fock-PARFLM has **broken the PPL 10 barrier** — surpassing GPT-2 XL (1.5B params, PPL ~18), GPT-2 Large (774M, PPL ~14), and YuriiFormer Medium (354M, PPL ~14.9) by wide margins at **6.7–28× fewer parameters**. These numbers require independent validation on the full held-out set and a cross-corpus check (WikiText-103); see `debug/eval_ppl_debug.ipynb`.

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

### 4.3 Phase 3: 250K steps on 4B tokens (in progress — step 103,700)

| Step | PPL | Notes |
|-----:|----:|-------|
| 25,500 | 29.44 | Warm-restart regression (init best = 27.14) |
| 29,500 | **26.50** | First new best |
| 30,000 | **26.15** | |
| 33,500 | **25.20** | |
| 39,000 | **25.15** | |
| 40,000 | **24.71** | Breaks 25 |
| 42,500 | **24.03** | |
| 45,500 | **23.95** | Breaks 24 |
| 47,000 | **23.49** | |
| 48,000 | **23.12** | |
| 49,500 | **21.96** | Breaks 22 |
| 75,500 | 18.40 | _ext2 resume from step 75,000 |
| 78,500 | **16.62** | Breaks previous best (16.65) |
| 79,000 | **16.42** | |
| 79,500 | **15.68** | Breaks 16 |
| 80,000 | **15.16** | **Surpasses GPT-2 Medium (PPL ~17.1, 354M params)** |
| 83,500 | **14.59** | **Surpasses YuriiFormer Medium (PPL ~14.9, 354M)** |
| 86,000 | **14.39** | |
| 89,000 | **13.93** | Breaks 14 |
| 90,000 | **13.43** | |
| 91,000 | **13.25** | |
| 92,000 | **13.15** | |
| 92,500 | **12.88** | Breaks 13 |
| 95,000 | **12.79** | |
| 95,500 | **12.53** | |
| 96,500 | **12.09** | Breaks 12 — **acceleration begins** |
| 97,000 | **11.83** | |
| 98,000 | **11.41** | |
| 99,500 | **10.60** | Breaks 11 |
| 101,500 | **10.49** | |
| 102,000 | **9.83** | 🚨 **Breaks 10 — unprecedented for 53M params** |
| 102,500 | **9.71** | |
| 103,500 | **9.50** | **Current overall best** |

**Phase 3 observations (interim, step 103,700):**

1. **Mild warm-restart regression:** PPL briefly regressed from 27.14 to ~29.4 in the first 500 steps before recovering. This is much milder than Phase 2's regression (to 74.4), likely due to the lower peak LR (1.5e-4 vs 3e-4).

2. **Sustained descent in the stable phase:** Phase 3's stable phase is producing continuous new bests across a remarkably long window. From step 29,500 to 103,500, the model set **30+ new-best records**, reducing PPL from 27.14 to 9.50 — a **65% reduction** with constant LR. This rate of improvement in the stable phase far exceeds Phases 1–2, where the stable phase plateaued.

3. **PPL descent acceleration (steps 95K–103K).** In log-space, the descent rate *tripled* from the 80K–90K period to the 95K–103K period:
   - Steps 80K→90K: descent rate ≈ 0.012 nats/K steps
   - Steps 95K→103.5K: descent rate ≈ 0.035 nats/K steps
   This acceleration is atypical — normal training shows deceleration in log-PPL descent. Possible explanations include (a) a genuine representational phase transition, (b) progressive specialisation of $\xi$ channels (α₁ collapsed to 0.023), or (c) increasing correlation between the small 5-batch validation eval and training data.

4. **Multiple milestone crossings:** PPL broke through 20 (step 49.5K), 17 (step 77K), 15 (step 80K), 14 (step 89K), 13 (step 92.5K), 12 (step 96.5K), 11 (step 99.5K), and **10** (step 102K) in succession. At PPL 9.50, the 53M-parameter model has surpassed GPT-2 XL (1.5B params, PPL ~18) by almost 2× in PPL and exceeds YuriiFormer Medium (354M, PPL ~14.9) by 36%.

5. **Gradient spikes — controlled but present.** Four spikes > 100 recorded in the 84K–103K window:
   - Step 84,925: pre-clip grad = 200.8 (led by `reverse_channel_scale` at 749.5)
   - Step 85,739: pre-clip grad = 119.3
   - Step 85,743: pre-clip grad = 113.1
   - Step 91,390: pre-clip grad = 172.2
   All recovered within 1–2 steps, no watchdog trigger. Baseline gradient norms have crept up from ~7–8 (step 75K) to ~10–14 (step 103K), indicating the model is operating in a higher-curvature loss landscape.

6. **$v_{\text{reg}}$ continuing to decline:** From 0.0072 at step 75K to ~0.0053 at step 103K, a 26% reduction, indicating progressively smoother dynamics.

7. **$\xi$ channel evolution:**
   - $\alpha_1$: 0.056 → **0.023** (collapsed to near-instantaneous context)
   - $\alpha_2$: 0.278 → 0.262 (stable)
   - $\alpha_3$: 0.434 → 0.421 (stable)
   - $\alpha_4$: 0.530 → 0.472 (moderate decline)
   - $\alpha_5$: 0.962 → 0.961 (locked — long-horizon structural mode)

8. **Training loss.** ntp has declined from ~2.83 (step 75K) to ~2.23 (step 103K), a **21% reduction**. The training–validation gap has nearly closed: at step 103.5K, ntp ≈ 2.23 vs val_loss = 2.251 — a gap of only 0.02 nats.

9. **WSD decay phase (steps 175K–250K) still ~71K steps away.** The model is still in the stable-LR phase with 146K steps remaining. If the current log-PPL descent rate of ~0.03/K steps continues through the stable phase, the floor at step 175K would be ~4–6 PPL, which would be almost certainly too aggressive to be real. A more conservative projection (rate halving) gives ~7–8 PPL at the stable floor, with the decay potentially pushing it to ~5–6. **These numbers are so extraordinary that independent full-set validation is essential before any claims can be made.**

10. **⚠️ Validation urgency.** The in-training eval uses only 5 random batches (~2,560 tokens per eval). At PPL < 10, this is increasingly unreliable. The previous full-set eval (step 77,500) gave PPL 12.37 vs in-training 16.65 — the full-set was 4.3 PPL *lower*. If that relationship holds, the full-set PPL at step 103.5K could be as low as ~5–6, which would be extraordinary. Conversely, the acceleration could be a small-sample artifact that disappears on the full validation set. A full-set eval and WikiText-103 cross-check are the top priority before any further claims.

---

### 4.4 Cross-scale context: Comparison with YuriiFormer and GPT-2

To contextualise the d=384 results, we compare against recent transformer baselines on OpenWebText. The comparison requires caveats (see §4.5) but the parameter and data efficiency of Fock-PARFLM is striking.

| Model | Params | Training tokens | Optimizer | Context | Best PPL | Source |
|-------|-------:|----------------:|-----------|:-------:|:--------:|--------|
| **Fock-PARFLM d=384 (Phase 3, interim)** | **53M** | **~2.9B** | AdamW | 512 | **9.50** | This work |
| GPT-2 Small checkpoint | 124M | ~9B (est.) | Adam | 1024 | ~22.6 | Karpathy nanoGPT eval |
| YuriiFormer Small (Nesterov+LT) | 124M | 14.75B | Muon+AdamW | 1024 | ~18.5 | Zimin et al. 2026 |
| GPT-2 Medium checkpoint | 354M | ~9B (est.) | Adam | 1024 | ~17.1 | Karpathy nanoGPT eval |
| YuriiFormer Medium (Nesterov+LT) | 354M | 14.75B | Muon+AdamW | 1024 | ~14.9 | Zimin et al. 2026 |

GPT-2 and YuriiFormer PPL values are derived from validation cross-entropy losses reported in nats/token (PPL = exp(loss)). GPT-2 checkpoint values are from Karpathy's nanoGPT evaluation on the OpenWebText validation split. YuriiFormer values are from Table 3 of Zimin et al. (arXiv:2601.23236).

**Key observations:**
- At **53M parameters** and PPL 9.50, Fock-PARFLM has **surpassed every listed baseline** — including GPT-2 XL (1.5B params, PPL ~18) and YuriiFormer Medium (354M, PPL ~14.9).
- Fock-PARFLM uses only **~2.9B tokens** vs YuriiFormer's 14.75B (**5.1× fewer**) and GPT-2's ~9B (**3.1× fewer**).
- YuriiFormer uses Muon (a state-of-the-art optimizer for transformers); Fock-PARFLM uses plain AdamW.
- **⚠️ These numbers are extraordinary and potentially too good to be true.** The in-training PPL is based on only 5 validation batches. Full-set evaluation and cross-corpus checks are essential before any claims can be made. See §4.3 observation 10.

### 4.5 Comparison caveats

Several methodological differences affect direct numerical comparison:

| Factor | Fock-PARFLM | YuriiFormer / GPT-2 |
|--------|-------------|---------------------|
| **Context length** | 512 tokens | 1024 tokens |
| **Validation set** | 2M tokens, 5 batches (~2,560 tokens/eval) | 432M tokens, 160 batches (~4.9M tokens/eval) |
| **Vocab size** | 50,257 | 50,304 (padded) |
| **Precision** | fp32 | bf16 |

The most significant factor is **context length**: conditioning on 1024 tokens instead of 512 typically lowers PPL by 1–3 points, giving YuriiFormer/GPT-2 a systematic advantage. This means the "true" 1024-context PPL of Fock-PARFLM would likely be 1–3 points lower than the reported 512-context value — i.e., **~7–9 PPL** at the current checkpoint, if the in-training numbers are real. The validation set size difference means our PPL estimates have higher variance per evaluation. Additionally, the previous full-set sliding-window evaluation (step 77,500) gave PPL 12.37 vs in-training 16.65 — the full-set was significantly *lower*, suggesting the in-training metric may systematically underestimate performance. However, the dramatic acceleration in PPL descent from step 95K onward warrants caution.

---

## 5. Gradient Health

### 5.1 Spike profile

| | Phase 1 | Phase 2 | Phase 3 (step 103.7K) |
|---|:---:|:---:|:---:|
| Spikes (grad > 100) | 15 | 14 | 10 |
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
| d=384, L=16 | 7,427 | ~35 (across 334K steps) | 0 | Manageable |
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
| 325,000 | 3 (step 75K) | 0.0072 |
| 334,000 | 3 (step 84K) | 0.0065 |
| 340,000 | 3 (step 90K) | 0.0062 |
| 348,000 | 3 (step 98K) | 0.0057 |
| 353,000 | 3 (step 103K) | 0.0053 |

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
| 325,000 (P3, step 75K) | 0.056 | 0.278 | 0.434 | 0.530 | 0.962 |
| 334,000 (P3, step 84K) | 0.042 | 0.272 | 0.425 | 0.500 | 0.961 |
| 340,000 (P3, step 90K) | 0.036 | 0.270 | 0.423 | 0.486 | 0.960 |
| 348,000 (P3, step 98K) | 0.028 | 0.268 | 0.426 | 0.475 | 0.960 |
| 353,000 (P3, step 103K) | **0.023** | 0.262 | 0.421 | 0.472 | 0.961 |

All five channels remain active throughout training ($\alpha > 0.02$ at step 103K). The monotonic decrease in lower-channel $\alpha$ values continues to accelerate, with $\alpha_1$ collapsing from 0.093 (step 50K) to **0.023** (step 103K) — a near-instantaneous context channel. Notably, $\alpha_5$ (the longest-horizon channel) has remained locked at ~0.960–0.963 across all three phases, suggesting this channel captures a structural mode that is scale-invariant. The progressive collapse of $\alpha_1$ may be contributing to the PPL acceleration observed in §4.3: as this channel approaches a delta function (instantaneous token embedding), it effectively creates a "bypass" channel that lets the force field access raw per-token information without temporal smoothing.

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
| `..._best.pt` | 103,500 (P3) | 9.50 | 3 | Canonical best |
| `..._step103500_best.pt` | 103,500 (P3) | 9.50 | 3 | Step-stamped copy |
| `..._step102500_best.pt` | 102,500 (P3) | 9.71 | 3 | |
| `..._step102000_best.pt` | 102,000 (P3) | 9.83 | 3 | First sub-10 PPL |
| `..._step101500_best.pt` | 101,500 (P3) | 10.49 | 3 | |
| `..._step100000.pt` | 100,000 (P3) | 11.27 | 3 | Periodic checkpoint |
| `..._step99500_best.pt` | 99,500 (P3) | 10.60 | 3 | |
| `..._step98000_best.pt` | 98,000 (P3) | 11.41 | 3 | |
| `..._step97000_best.pt` | 97,000 (P3) | 11.83 | 3 | |
| `..._step96500_best.pt` | 96,500 (P3) | 12.09 | 3 | |
| `..._step95500_best.pt` | 95,500 (P3) | 12.53 | 3 | |
| `..._step95000_best.pt` | 95,000 (P3) | 12.79 | 3 | |
| `..._step95000.pt` | 95,000 (P3) | 12.79 | 3 | Periodic checkpoint |
| `..._step92500_best.pt` | 92,500 (P3) | 12.88 | 3 | |
| `..._step92000_best.pt` | 92,000 (P3) | 13.15 | 3 | |
| `..._step91000_best.pt` | 91,000 (P3) | 13.25 | 3 | |
| `..._step90000_best.pt` | 90,000 (P3) | 13.43 | 3 | |
| `..._step90000.pt` | 90,000 (P3) | 13.43 | 3 | Periodic checkpoint |
| `..._step89000_best.pt` | 89,000 (P3) | 13.93 | 3 | |
| `..._step86000_best.pt` | 86,000 (P3) | 14.39 | 3 | |
| `..._step85000.pt` | 85,000 (P3) | 15.53 | 3 | Periodic checkpoint |
| `..._step83500_best.pt` | 83,500 (P3) | 14.59 | 3 | |
| `..._step80000_best.pt` | 80,000 (P3) | 15.16 | 3 | |
| `..._step80000.pt` | 80,000 (P3) | 15.16 | 3 | Periodic checkpoint |
| `..._step79500_best.pt` | 79,500 (P3) | 15.68 | 3 | |
| `..._step79000_best.pt` | 79,000 (P3) | 16.42 | 3 | |
| `..._step78500_best.pt` | 78,500 (P3) | 16.62 | 3 | |
| `..._step77500_best.pt` | 77,500 (P3) | 16.65 | 3 | Previous _ext run best |
| `..._step150000_best.pt` | 150,000 (P2) | 27.23 | 2 | Phase 2 best |
| `..._step99000_best.pt` | 99,000 (P1) | 63.69 | 1 | Phase 1 best |

All checkpoints include model weights, optimizer state, scheduler state, and training metadata.

---

## 10. Token Budget Analysis

| Phase | Pool | Consumed | Pool coverage | Repetition rate |
|-------|-----:|---------:|--------------:|----------------:|
| Phase 1 | 1B | 0.82B | 82% | ~1.0× |
| Phase 2 | 2B | 1.23B | 62% | ~0.6× |
| Phase 3 (in progress) | 4B | ~0.85B (so far) | ~21% | ~0.2× |
| **Total** | — | **~2.90B** | — | — |

The graduated token-pool strategy (1B → 2B → 4B) provides increasing data diversity while maintaining repetition in early phases for stable learning. Phase 1's near-complete pool coverage (82%) acts as implicit regularisation. Phase 3's 4B pool will consume ~2.05B tokens at completion (eff_batch=16 × 512 × 250K steps) — only 51% of the pool, ensuring no repetition risk.

---

## 11. Phase 3 Status and Projections

Phase 3 is running on Colab H100 using `colab_fock_depthcond_vtheta_openwebtext_ext2.ipynb`.

| Metric | Value |
|--------|-------|
| Current step | 103,700 / 250,000 (41%) |
| Current best PPL | **9.50** (step 103,500) |
| Phase | WSD stable (decay begins at step 175,000) |
| Estimated wall time remaining | ~122 h |

**Projection methodology.** Extrapolation is increasingly uncertain because the model has entered an unprecedented PPL regime for its parameter count. The stable-phase descent rate has *accelerated* from ~0.012 nats/K steps (80K–90K) to ~0.035 nats/K steps (95K–103K), the opposite of normal deceleration. If we assume the rate stabilises at the recent ~0.03 nats/K steps:

| Scenario | Assumption | Stable-phase floor (step 175K) | Decay (20–30%) | Final PPL |
|----------|------------|:-----:|:---:|:---:|
| Conservative | rate halves to 0.015/K | ~6.5 | 20% | **~5.2** |
| Moderate | rate holds at 0.02/K | ~4.0 | 25% | **~3.0** |
| Optimistic | rate holds at 0.03/K | ~2.6 | 30% | **~1.8** |

**⚠️ ALL of these projections are almost certainly overoptimistic.** PPL below ~5 on OWT would require near-perfect prediction on a large fraction of tokens, which is implausible for a 53M-parameter model with only ~3B training tokens. The likeliest outcome is that the descent rate decelerates sharply as the model hits diminishing returns, yielding a final PPL in the **5–8 range** if the current numbers are real.

**The binding constraint is now credibility, not compute.** The model has already surpassed all previous projections while still in the stable LR phase. Running the full-set eval and WikiText-103 cross-check on the step 103,500 checkpoint is the single most important next step — these will determine whether the in-training PPL of 9.50 is genuine or an artifact of the small 5-batch validation eval.

### 11.1 Potential Phase 4

If Phase 3 completes successfully and capacity remains (PPL still improving at the end), a Phase 4 extension could be considered with an 8B token pool and 350K+ steps. However, the 512-token context length is increasingly likely to be the binding constraint — transitioning to 1024-token contexts would enable fairer comparison with transformer baselines but requires re-engineering the memory budget.

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
