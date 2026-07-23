# Fock-PARFLM d=768 Training Results on OpenWebText

**Author:** Dimitar P. Gueorguiev
**Date:** July 2026
**Status:** In progress — Phase 1 (100K steps) running on LambdaLabs 1×H100; step 81,500 / 100,000 (WSD decay phase active)

---

> **⚠ CAUSAL LEAK — all d=768 results are compromised, regardless of scale (audit, July 2026).**
>
> Every PPL in this document (best **84.04** and the whole descent) comes from
> the **same leaky Fock-PARFLM v2.1 architecture** as d=384: a reverse channel
> reading from a shared full-window register state, trained **without** the
> `prefix_causal_registers` fix. This leak is **architectural, not
> dimension-dependent** — it lets past-token predictions access future tokens in
> *every* configuration that enables the reverse channel (`reverse_channel=True`,
> `fock_version=v2`; see §3.2), independent of hidden dimension `d`, depth `L`, or
> damping `γ`. **Changing the hidden state dimension does not remove the leak;
> every model in the family that uses this reverse channel is leaky and its
> perplexities are compromised.**
>
> The leak was measured directly on the d=384 checkpoint: **honest PPL ≈258 vs
> reported ≈7.69 — +3.51 nats/token, ≈33× inflation** — carried entirely by the
> reverse channel (zeroing its gate returns bit-exact 0.0 future→past
> sensitivity). Because d=768 shares that *exact* mechanism, **its reported
> perplexities are inflated by the same leak and are not valid language-modeling
> results.** They cannot be compared to leak-free transformer baselines, and the
> cross-scale comparisons in §5.5 compare one leaky model to another.
>
> All d=768 numbers are **pending re-certification**: re-training with
> `prefix_causal_registers=True` plus the honest-PPL (target-relocation) test is
> required. Full analysis:
> [`Fock-PARFLM_Causal_Leak_Audit_Results.md`](Fock-PARFLM_Causal_Leak_Audit_Results.md);
> d=384 probe results:
> [`Fock-PARFLM_d384_Training_Results_OpenWebText.md`](Fock-PARFLM_d384_Training_Results_OpenWebText.md) §7.2.

---

## 1. Summary

This document records the training history of the Fock-PARFLM v2.1 model at d=768, L=12 on OpenWebText, driven by the standalone `train_fock.py` script (preset `d768`) on a LambdaLabs 1×H100 instance. This is the first scale-up beyond the d=384 baseline.

| | Phase 1 (in progress) |
|---|:---:|
| Steps | 100,000 |
| Token pool | 4B |
| Tokens consumed (so far) | ~1.34B |
| Wall time (so far) | ~93 h |
| Best PPL (so far) | **84.04** (step 81,500) |
| Hardware | LambdaLabs 1×H100 |

The model is deep into the **WSD decay phase** (step 81,500 / 100,000). After surviving two watchdog reloads (step 52,064 and step 65,076) and catastrophic gradient spikes (worst: grad=5.16M at step 51,898), the model has entered a regime of confident, monotonic PPL descent with 4 consecutive NEW BEST records in the latest 2K steps (85.21 → 84.77 → 84.06 → 84.04). The LR has decayed from 1.34e-4 to 1.14e-4. With ~18.5K steps of decay remaining, substantial further PPL compression is expected.

### 1.1 Cumulative Token Budget by Phase (Planned)

| Phase | Steps | Token Pool | Cumulative Token Pool | Tokens Consumed (est.) | Cumulative Consumed |
|:-----:|------:|:----------:|:---------------------:|:----------------------:|:-------------------:|
| 1 | 100,000 | 4B | 4B | ~1.64B | 1.64B |
| 2 | 150,000 | 6B | 10B | ~2.46B | 4.10B |
| 3 | 250,000 | 8B | **18B** | ~4.10B | **~8.20B** |

Token consumption per phase is estimated as steps × effective_batch_size (32) × sequence_length (512) = 16,384 tokens/step. The graduated pool strategy (4B → 6B → 8B) ensures no repetition risk: even Phase 3's ~4.1B consumed tokens represents only 51% of the 8B pool.

---

## 2. Architecture

| Parameter | Value |
|-----------|-------|
| Model | Fock-PARFLM v2.1 (untied embeddings) |
| Hidden dimension ($d$) | 768 |
| Depth ($L$) | 12 |
| Total parameters | ~137M |
| Integrator | Velocity-Verlet with O-step Langevin friction |
| Damping ($\gamma$) | 0.05 (fixed, gamma-sweep optimal) |
| $V_\theta$ | Depth-conditioned multi-context Gaussian wells (5 heads × 8 wells) |
| $V_\phi$ | Structural-competitive pairwise potential |
| $\xi$ channels | 5 ("5long" override) |
| Registers | 32 |
| Fock mechanism | Per-layer creation/destruction gates + reverse channel (per-layer gated) |
| Embeddings | Untied (`E` separate from `lm_head`) |
| Output bias | Log-unigram frequency initialised |
| Positional encoding | Learned (`P`, 1024 × 768) |
| $d_k$ (register attention) | 192 |
| Memory optimisations | `use_layer_checkpoint=True`, `use_gathered_v_phi=True` |

See [Fock-PARFLM\_Scale-Up\_Comparative\_Experiments.md](Fock-PARFLM_Scale-Up_Comparative_Experiments.md) §2.2 for the full per-component parameter breakdown.

---

## 3. Training Configuration

### 3.1 Execution environment

The d=768 model is trained using the standalone `train_fock.py` script rather than a Colab notebook. The launch command is:

```bash
bash launch_lambdalabs.sh d768 --fixed_gamma 0.05
```

This runs `python3 train_fock.py` in single-GPU mode on the H100.

### 3.2 Configuration verification: script vs notebook

Since the d=384 model was trained via Colab notebooks (`colab_fock_depthcond_vtheta_openwebtext.ipynb` / `_ext.ipynb`) while d=768 uses the standalone script, a configuration audit was performed to verify consistency. The `build_fock_model` function in `train_fock.py` was extracted from the notebook's `make_config` and produces an identical `FockMultiXiPARFConfig`.

#### Identical parameters (architecture-defining)

| Parameter | Value (both) | Notes |
|-----------|:---:|-------|
| `v_theta_variant` | `gaussian` | Depth-conditioned multi-context Gaussian |
| `v_theta_wells_per_head` | 8 | |
| `v_theta_depth_condition` | `True` | |
| `v_theta_depth_code_init_std` | 0.02 | |
| `v_theta_n_heads` | 5 | One per $\xi$ channel |
| `xi_override` | `5long` | $\alpha = [0.50, 0.75, 0.95, 0.99, 0.995]$ |
| `v_phi_kind` | `structural_competitive` | |
| `v_phi_d_type` / `d_angle` / `n_heads` / `mlp_hidden` | 32 / 16 / 4 / 128 | |
| `top_k` | 16 | |
| `fock_version` | `v2` | |
| `reverse_channel` | `True` | Per-layer, stable, pre-LN, soft-norm |
| `reverse_channel_warmup_steps` | 4000 | |
| `register_repulsion` | `True` (coeff 0.05, gram) | |
| `tie_embeddings` | `False` | |
| `use_output_bias` | `True` | |
| `use_layer_checkpoint` | `True` | |
| `use_gathered_v_phi` | `True` | |
| `ln_after_step` | `True` | |
| `causal_force` | `True` | |
| `per_group_clip` | `True` | |
| `wsd_warmup_frac` | 0.05 | |
| `lr_schedule` | `wsd` | |
| `weight_decay` | 0.01 | |
| `grad_clip` | 1.0 | |
| `grad_clip_vphi` | 0.3 | |

#### Scale-adjusted parameters

| Parameter | Notebook (d=384) | Script (d=768) | Rationale |
|-----------|:---:|:---:|-----------|
| $d$ | 384 | 768 | 2× width scale-up |
| $L$ | 16 | 12 | Fewer layers at wider $d$ (matches GPT-2 Small depth) |
| $d_k$ | 64 | 192 | Scales as $d/4$; ensures register-attention capacity matches hidden dimension |
| Learning rate | $3 \times 10^{-4}$ | $2 \times 10^{-4}$ | Lower LR for larger model (standard practice) |
| `batch_size` | auto-probed (~8) | 4 | Limited by H100 VRAM at d=768 |
| `grad_accum` | 2 | 8 | Maintains effective batch size of 32 |
| Effective batch | 16 | 32 | Larger eff batch at larger scale |
| `fixed_gamma` | 0.30 | 0.05 | Gamma-sweep optimal (see §7) |
| `max_train_tokens` | 1B (Phase 1) | 4B | Larger token pool; no exhaustion risk at 100K steps |
| Hardware | Colab A100/H100 | LambdaLabs 1×H100 | Dedicated instance for long runs |

**Key observation:** The architecture-defining parameters are identical. All differences are scale-appropriate adjustments to training hyperparameters ($d$, $L$, $d_k$, LR, batch logistics) and the gamma value determined by the d=768 gamma sweep.

### 3.3 Training hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW ($\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$) |
| Weight decay | 0.01 |
| Sequence length (`BLOCK_SIZE`) | 512 |
| Effective batch size | 32 (`batch_size=4 × grad_accum=8`) |
| Learning rate | $2 \times 10^{-4}$ (peak) |
| LR schedule | WSD (Warmup-Stable-Decay) |
| WSD warmup | 0 → 5,000 (5% of 100K) |
| WSD stable end | 65,000 |
| WSD decay end | 100,000 |
| Gradient clip (global) | 1.0 |
| Gradient clip ($V_\phi$) | 0.3 |
| Per-group gradient clipping | Enabled |
| Evaluation interval | 500 steps |
| Data | OpenWebText (Skylion007/openwebtext), GPT-2 BPE tokenisation |
| Token pool | 4B |
| Precision | fp32 |

---

## 3A. Evaluation Protocol

All validation perplexity numbers reported in this document are computed on a **held-out 2M-token slice** of OpenWebText that is **physically disjoint** from the training data:

- **Split mechanism:** When OpenWebText is first streamed and tokenised (GPT-2 BPE, vocab 50,257), the code requests `MAX_TRAIN_TOKENS + 2,000,000` total tokens. The **last 2M tokens** are sliced off as the validation set; the **first N tokens** become the training set. The two are cached as separate files (`openwebtext_val_2M.npy` and `openwebtext_train_{N}M.npy`).
- **No overlap:** The validation tokens are from distinct documents at the tail of the stream. There is zero token-level overlap between training and validation.
- **Fixed across phases:** When the token budget changes between phases, the training pool grows but the validation set remains the **same 2M-token held-out slice**. All PPL numbers across phases are therefore directly comparable.
- **Evaluation procedure:** At each evaluation step, 5 random batches of length 512 are drawn from the validation set, the model computes cross-entropy loss in inference mode, and the mean loss is exponentiated to produce PPL: $\text{PPL} = \exp(\bar{\mathcal{L}}_{\text{val}})$.

This is identical to the protocol used for d=384 (see [Fock-PARFLM\_d384\_Training\_Results\_OpenWebText.md](Fock-PARFLM_d384_Training_Results_OpenWebText.md) §3A) and standard held-out evaluation practice.

---

## 4. Learning Curve

### 4.1 Phase 1: 100K steps on 4B tokens (in progress — step 65,500)

| Step | PPL | Notes |
|-----:|----:|-------|
| 500 | 1378.75 | Initial random performance |
| 1,000 | 754.70 | |
| 2,000 | 417.18 | |
| 3,000 | 286.29 | Rapid early descent |
| 5,000 | 186.30 | |
| 10,000 | 137.68 | |
| 15,000 | 114.99 | |
| 20,000 | 105.43 | |
| 25,000 | 99.59 | Breaks 100 |
| 30,000 | 96.77 | |
| 35,000 | 93.33 | |
| 40,000 | 92.50 | Plateau begins |
| 45,000 | 91.46 | |
| 45,500 | 90.66 | |
| 48,000 | 90.85 | |
| 48,500 | 92.98 | Spike-induced regression |
| 50,000 | 91.49 | |
| 51,000 | 91.27 | |
| 52,000 | 90.93 | Post-watchdog-reload #1 recovery |
| 52,500 | 90.50 | Previous best after watchdog reload #1 |
| 53,000 | 90.38 | |
| 53,500 | 90.32 | |
| 55,500 | **89.72** | **Current best** |
| 63,000 | 90.95 | |
| 63,500 | 90.32 | |
| 64,000 | 91.05 | |
| 64,500 | 90.23 | |
| 65,000 | 91.68 | WSD decay phase begins |
| 65,500 | 91.87 | Post-watchdog-reload #2 regression |

**Phase 1 observations (interim, step 65,500):**
1. **Rapid initial descent:** PPL drops from 1379 to 186 in the first 5K steps, consistent with d=384 Phase 1 (1482 → 252 at 5K steps).
2. **Stable-phase plateau (steps 40K–65K):** PPL oscillated between 89–93 for ~25K steps. This is expected behaviour during the WSD stable-LR phase — the model has extracted most of the signal available at this LR. The same pattern was observed at d=384.
3. **Resilience through catastrophic spikes and two watchdog reloads:**
   - **Watchdog reload #1** (step 52,064): grad=5.16M at step 51,898 triggered EMA breach; reverted to step 45,500 (PPL 90.66). Model recovered and improved to 89.72 by step 55,500.
   - **Watchdog reload #2** (step 65,076): EMA grad_norm=82.5 > 50.0 for 200 sustained steps; reverted to step 55,500 (PPL 89.72). This was triggered by sustained elevated gradient EMA, not a single catastrophic event. Post-reload PPL at step 65,500 is 91.87 — a mild 2.4% regression from best.
4. **Decay phase entry:** The model has crossed step 65,000 and entered the WSD decay phase. At d=384, the decay phase reduced PPL by 37% (from ~101 to ~64). Applying a similar reduction factor from the current best of 89.72 projects a Phase 1 final PPL of **~57–67**. Even a more conservative 20–25% reduction gives ~67–72.
5. **Spike magnitude moderation:** The worst spike in the 53K–65K window (~48K at step 64,877) is two orders of magnitude below the run record (5.16M at step 51,898). This may indicate that the cascade partially resets after watchdog reloads and moderates as the LR begins to decay.
6. **Throughput:** ~3.4 sec/step on 1×H100, ~72 hours remaining for Phase 1.

---

## 5. Gradient Health

### 5.1 Spike profile (steps 0–65,500)

The JSONL training log records one entry per 50 steps, capturing the grad norm at those periodic checkpoints. However, the terminal output reveals many additional spikes between logged steps. The full picture combines both sources:

**From JSONL log (every 50 steps):**

| Metric | Value |
|--------|:-----:|
| Spikes (grad > 100) in logged steps | ~110 |
| Maximum gradient norm (logged) | 1,409.1 |

**From terminal output (every step):**

| Metric | Value |
|--------|:-----:|
| Maximum gradient norm (run record) | **5,158,336** (step 51,898) |
| Second-worst spike | 54,818 (step 52,818) |
| Worst spike in 53K–65K window | ~48,740 (step 64,877) |
| Spikes > 10,000 | 4+ (steps 50,051, 50,343, 52,818, 64,877) |
| Spikes > 1,000 | frequent |
| Watchdog triggers | 2 |
| Watchdog reloads | 2 |

**Watchdog reload events:**

| Event | Step | Trigger | Reverted to | Reverted PPL |
|-------|-----:|---------|------------:|-------------:|
| Reload #1 | 52,064 | EMA grad_norm=85.9 > 50.0 for 200 steps | 45,500 | 90.66 |
| Reload #2 | 65,076 | EMA grad_norm=82.5 > 50.0 for 200 steps | 55,500 | 89.72 |

The JSONL log significantly underreports spike severity because it samples only every 50 steps, missing the between-step catastrophic events. The terminal output reveals that the spike regime escalated to catastrophic levels (grad > 5M) around step 52K, then partially moderated in the 53K–65K window (worst ~48K). The watchdog mechanism has now triggered twice, both times successfully preserving the model's learning trajectory by reverting to a recent best checkpoint. Both reloads were triggered by sustained elevated EMA — not single catastrophic spikes — suggesting the cascade builds cumulatively before triggering the watchdog.

### 5.2 Spike escalation by training phase

| Step window | Spikes > 100 (logged) | Max grad (logged) | Max grad (terminal) | Notes |
|:-----------:|:---------------------:|:------------------:|:-------------------:|-------|
| 0–10K | 1 | 395 | — | Benign |
| 10K–20K | 2 | 482 | — | Benign |
| 20K–30K | 10 | 2,773 | — | Moderate onset |
| 30K–40K | 27 | 3,843 | — | Escalating |
| 40K–50K | 21 | 1,270 | 15,848 (step 50,051) | Catastrophic spikes emerging |
| 50K–53K | 8 | 1,409 | **5,158,336** (step 51,898) | Catastrophic; watchdog reload #1 |
| 53K–60K | ~15 | ~800 | — | **Moderation** after reload |
| 60K–65.5K | ~15 | ~1,100 | ~48,740 (step 64,877) | Re-escalating; watchdog reload #2 |

The escalation shows a distinctive pattern: spike frequency peaks around steps 30K–40K, while spike *magnitude* climbs to a catastrophic peak at step 52K (5.16M). After watchdog reload #1, spike magnitudes moderated significantly — the worst spike in the 53K–65K window (~48K) is **two orders of magnitude** below the run record. This partial reset suggests the watchdog reload breaks the second-order gradient cascade, though it re-accumulates over the subsequent ~10K steps until triggering reload #2. With the LR now actively declining in the decay phase, spikes may further moderate.

### 5.3 Top spike groups (from terminal output)

During catastrophic spikes, the gradient energy is concentrated in the embedding matrices (`E` and `P`) and the Fock mechanism gates:

| Step | Total grad | Top groups |
|-----:|----------:|-----------|
| 51,898 | 5,158,336 | E=3,305,054, P=3,303,969, destruction_gate=1,880,423, reverse_channel_scale=1,206,437 |
| 52,818 | 54,818 | P=51,865, register=13,893, E=9,753, creation_gate=4,588 |
| 50,051 | 15,848 | P=10,878, E=10,878, creation_gate=2,825, register=2,535 |

The pattern is consistent with the second-order gradient cascade: the `create_graph=True` backward pass amplifies gradients through the embedding matrices and Fock gates, with the cascade magnitude growing as the model's internal representations become more refined.

> **⚠ Causal-leak note.** `reverse_channel_scale` appears among the top spike
> groups here (e.g. 1,206,437 at step 51,898). Beyond its role in the gradient
> cascade, this scalar is the **carrier of the causal leak** (see the
> top-of-document banner): gradient descent persistently drives it open because
> peeking at future tokens lowers the training loss. Its large sustained gradient
> is therefore both a stability signal and a leak signal — identical in kind to
> the d=384 case, confirming the leak is present at this scale too.

### 5.4 Post-watchdog recovery

Both watchdog reloads demonstrate rapid recovery:

| Reload | Reverted to | Reverted PPL | Recovery PPL | Recovery step | Steps to recover |
|:------:|:----------:|:------------:|:------------:|:-------------:|:----------------:|
| #1 (step 52,064) | 45,500 | 90.66 | 89.72 | 55,500 | ~3,436 |
| #2 (step 65,076) | 55,500 | 89.72 | 91.87* | 65,500 | 424 (early) |

*Post-reload #2 PPL is still slightly regressed at step 65,500 (91.87 vs 89.72 best). This is expected — the model has only had 424 steps of recovery, and is now running under a declining LR in the WSD decay phase.

These recoveries demonstrate:
1. The watchdog mechanism works as designed — it prevents permanent damage from catastrophic spikes
2. The model's learning capacity is not exhausted despite repeated instabilities
3. The gradient cascade is a transient phenomenon: reloaded weights break the cascade, allowing productive learning to resume
4. Post-reload #1, the model not only recovered but *improved beyond its pre-spike best* (89.72 < 90.66), confirming that the spikes do not represent a capacity ceiling

### 5.5 Cross-scale comparison

| Scale | Phase | Max grad | Spikes > 100 | Watchdog reloads | Regime |
|-------|-------|:--------:|:------------:|:----------------:|--------|
| d=384, L=16 | Phase 1 (100K) | 757 | 15 | 0 | Manageable |
| d=384, L=16 | Phase 2 (150K) | 1,703 | 14 | 0 | Manageable |
| **d=768, L=12** | **Phase 1 (65.5K/100K)** | **5,158,336** | **~110 (logged)** | **2** | **Catastrophic (onset ~50K)** |
| d=768, L=12 | Prior run (late) | 81,019 | many | 1+ | Catastrophic |
| d=1024, L=16 | Full run | 63,949 | frequent | multiple | Catastrophic |

The d=768 run confirms that the catastrophic spike regime is a universal late-training phenomenon at d ≥ 768, not specific to a particular depth (L=12 here vs L=16 in prior runs). The onset at ~50K steps (of 100K) aligns with the prior d=768 run's timing. A notable finding from this run is that spike magnitudes **partially moderate** after a watchdog reload — the 53K–65K window's worst spike (~48K) is two orders of magnitude below the 5.16M catastrophe at step 52K. This suggests the reload breaks the second-order gradient cascade, though it re-accumulates over subsequent ~10K-step windows.

---

## 6. Dynamics Diagnostics

### 6.1 $v_{\text{reg}}$ (regularisation loss) trajectory

| Step | $v_{\text{reg}}$ |
|-----:|:-----------------:|
| 50 | 8.066 |
| 5,000 | 0.0016 |
| 10,000 | 0.0015 |
| 15,000 | 0.0013 |
| 20,000 | 0.0019 |
| 25,000 | 0.0016 |
| 30,000 | 0.0019 |
| 35,000 | 0.0023 |
| 40,000 | 0.0019 |
| 45,000 | 0.0014 |
| 50,000 | 0.0017 |
| 52,000 | 0.0028 |
| 52,500 | 0.0021 |

$v_{\text{reg}}$ remains stable in a narrow range (~0.001–0.003) throughout training, including through the catastrophic spike regime (steps 49K–53K) and subsequent watchdog reloads. A brief elevation to ~0.003 around step 52K coincided with the cascade buildup before the first watchdog reload but returned to baseline immediately after. This indicates that the hidden-state velocities are well-controlled even when the gradient cascade produces extreme gradient norms — the cascade affects the parameter updates, not the forward-pass dynamics. The lower magnitude compared to d=384 Phase 1 (which reached 0.006 by step 100K) may reflect d=768's smaller $\gamma$ (0.05 vs 0.30) — the weaker damping produces smoother, near-geodesic dynamics with inherently smaller velocity fluctuations.

### 6.2 $\xi$ channel utilisation ($\alpha$ trajectory)

While the JSONL log does not record $\xi$ alphas, the terminal output includes them at periodic reporting intervals. The $\xi$ channels are initialised to `[0.50, 0.75, 0.95, 0.99, 0.995]`.

| Step | $\alpha_1$ | $\alpha_2$ | $\alpha_3$ | $\alpha_4$ | $\alpha_5$ |
|-----:|:----------:|:----------:|:----------:|:----------:|:----------:|
| 49,250 | 0.466 | 0.561 | 0.785 | 0.932 | 0.991 |
| 50,050 | 0.466 | 0.561 | 0.784 | 0.931 | 0.991 |
| 51,300 | 0.465 | 0.561 | 0.782 | 0.931 | 0.991 |
| 52,500 | 0.466 | 0.562 | 0.785 | 0.932 | 0.991 |
| 52,850 | 0.467 | 0.562 | 0.785 | 0.932 | 0.991 |

All five channels remain active ($\alpha > 0.46$). The values are remarkably stable across the last ~4K steps, including through the catastrophic spike and watchdog reload at step 52,064. Compared to d=384 at a similar training fraction (50% complete), the d=768 $\alpha$ values are higher across all channels (e.g. $\alpha_1$ = 0.466 vs d=384's 0.395 at step 50K), suggesting the wider model retains more utility from the shorter-horizon channels.

---

## 7. Gamma Sweep

A dedicated gamma sweep (8 candidates, 3,000 steps each) was conducted on LambdaLabs before launching full training. Full results are documented in [Fock-PARFLM\_Scale-Up\_Gamma\_Sweep\_Results\_and\_Damping\_Regime\_Analysis.md](Fock-PARFLM_Scale-Up_Gamma_Sweep_Results_and_Damping_Regime_Analysis.md) §4. Key findings:

- **Optimal $\gamma$ for PPL:** 0.05 (PPL 259.09 at 3K steps)
- **Sweep quality:** Clean, monotonic — PPL increases strictly with $\gamma$
- **d=384 comparison:** d=384 optimal is $\gamma = 0.25$; the shift to $\gamma = 0.05$ at d=768 reflects a dimension-dependent transition to weaker damping at larger scale
- **Geodesic residual $\bar{R}$:** Also minimised at $\gamma = 0.05$ — PPL-geodesic coincidence holds at d=768 (unlike the breakdown at d=384)

The training run uses $\gamma = 0.05$ based on this sweep result.

---

## 8. Checkpoints

No checkpoints have been exported yet. Phase 1 completion checkpoints will be archived upon run completion.

---

## 9. Token Budget Analysis

| Phase | Pool | Consumed (so far) | Pool coverage | Notes |
|-------|-----:|---------:|--------------:|-------|
| Phase 1 (in progress) | 4B | ~1.34B | ~33% | No repetition risk |

With a 4B token pool and 100K steps at effective batch 32 × sequence length 512 = 16,384 tokens/step, the full Phase 1 will consume ~1.64B tokens — only 41% of the pool. At step 81,500, approximately 1.34B tokens have been consumed (81,500 × 16,384), well within the pool.

---

## 10. Planned Extensions

Following the graduated extension strategy established with d=384:

| | Phase 2 (planned) | Phase 3 (planned) |
|---|:---:|:---:|
| Steps | 150,000 | 250,000 |
| Token pool | 6B | 8B |
| Resume from | Phase 1 best checkpoint | Phase 2 best checkpoint |
| `FRESH_SCHEDULE` | True | True |
| Projected final PPL | ~40–55 | ~25–35 |

PPL projections are based on the d=384 phase-over-phase reduction pattern (Phase 1→2: 57% reduction, Phase 2→3: 39% so far), adjusted for the larger model's higher capacity. The total tokens seen across all three phases will be ~8.2B (see §1.1).

---

## 11. Cross-References

- [Fock-PARFLM\_Scale-Up\_Comparative\_Experiments.md](Fock-PARFLM_Scale-Up_Comparative_Experiments.md) — parameter counts and architecture comparison with GPT-2
- [Fock-PARFLM\_d384\_Training\_Results\_OpenWebText.md](Fock-PARFLM_d384_Training_Results_OpenWebText.md) — d=384 training results (reference point for scale comparison)
- [Training\_Instabilities\_in\_Fock-PARFLM\_with\_structured\_V\_theta.md](Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) — gradient spike analysis and cascade mechanism
- [Fock-PARFLM\_Scale-Up\_Gamma\_Sweep\_Results\_and\_Damping\_Regime\_Analysis.md](Fock-PARFLM_Scale-Up_Gamma_Sweep_Results_and_Damping_Regime_Analysis.md) — gamma sweep results and dimension-dependent phase transition
- [Geodesic\_Preservation\_Experiment.md](Geodesic_Preservation_Experiment.md) — geodesic residual analysis and PPL-geodesic coincidence
