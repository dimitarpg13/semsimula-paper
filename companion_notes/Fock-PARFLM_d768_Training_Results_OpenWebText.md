# Fock-PARFLM d=768 Training Results on OpenWebText

**Author:** Dimitar P. Gueorguiev
**Date:** July 2026
**Status:** In progress — Phase 1 (100K steps) running on LambdaLabs 1×H100; step 52,950 / 100,000

---

## 1. Summary

This document records the training history of the Fock-PARFLM v2.1 model at d=768, L=12 on OpenWebText, driven by the standalone `train_fock.py` script (preset `d768`) on a LambdaLabs 1×H100 instance. This is the first scale-up beyond the d=384 baseline.

| | Phase 1 (in progress) |
|---|:---:|
| Steps | 100,000 |
| Token pool | 4B |
| Tokens consumed (so far) | ~0.87B |
| Wall time (so far) | 36.5 h |
| Best PPL (so far) | **90.50** (step 52,500) |
| Hardware | LambdaLabs 1×H100 |

The model is currently at step 52,950 in the WSD stable-LR phase (decay begins at step 65,000). PPL continues to improve slowly despite escalating gradient spikes (including a catastrophic grad=5.16M at step 51,898 and a watchdog reload at step 52,064). The new best of 90.50 was set *after* the watchdog reload, confirming the model is still learning. The main PPL gains are expected during the WSD decay phase (steps 65K–100K), consistent with the d=384 training pattern where the decay phase delivered a 37% PPL reduction.

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

## 4. Learning Curve

### 4.1 Phase 1: 100K steps on 4B tokens (in progress — step 52,950)

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
| 45,500 | 90.66 | Previous best |
| 48,000 | 90.85 | |
| 48,500 | 92.98 | Spike-induced regression |
| 50,000 | 91.49 | |
| 51,000 | 91.27 | |
| 52,000 | 90.93 | Post-watchdog-reload recovery |
| 52,500 | **90.50** | **Current best — NEW BEST after watchdog reload** |

**Phase 1 observations (interim, step 52,950):**
1. **Rapid initial descent:** PPL drops from 1379 to 186 in the first 5K steps, consistent with d=384 Phase 1 (1482 → 252 at 5K steps).
2. **Stable-phase plateau:** PPL has been oscillating between 90–93 for the last ~10K steps. This is expected behaviour during the WSD stable-LR phase — the model is at a steady learning rate and has extracted most of the signal available at this LR. The same pattern was observed at d=384, where the stable phase (steps ~5K–65K) saw progressively slower gains.
3. **Resilience through catastrophic spikes:** Despite a grad=5.16M spike at step 51,898 and a watchdog reload at step 52,064 (reverting to step 45,500 checkpoint), the model recovered and set a new best PPL of 90.50 just 436 steps later at step 52,500. This confirms the watchdog mechanism works as designed and the model's learning capacity is not exhausted.
4. **WSD decay phase (steps 65K–100K) is the critical window:** At d=384, the decay phase reduced PPL by 37% (from ~101 to ~64). Applying a similar reduction factor to d=768's current ~90.5 PPL projects a Phase 1 final PPL of **~57–65**.
5. **Throughput:** 7.53 sec/step on 1×H100, with an estimated ~98 hours remaining for Phase 1.

---

## 5. Gradient Health

### 5.1 Spike profile (steps 0–52,950)

The JSONL training log records one entry per 50 steps, capturing the grad norm at those periodic checkpoints. However, the terminal output reveals many additional spikes between logged steps. The full picture combines both sources:

**From JSONL log (every 50 steps):**

| Metric | Value |
|--------|:-----:|
| Spikes (grad > 100) in logged steps | 69 |
| Maximum gradient norm (logged) | 1,409.1 |

**From terminal output (every step, steps 49K–53K):**

| Metric | Value |
|--------|:-----:|
| Maximum gradient norm | **5,158,336** (step 51,898) |
| Second-worst spike | 54,818 (step 52,818) |
| Spikes > 10,000 | 3 (steps 50,051, 50,343, 52,818) |
| Spikes > 1,000 | frequent |
| Watchdog triggers | 1 (step 52,064, EMA=85.9 > 50.0 for 200 steps) |
| Watchdog reloads | 1 (reverted to step 45,500, PPL 90.66) |

The JSONL log significantly underreports spike severity because it samples only every 50 steps, missing the between-step catastrophic events. The terminal output reveals that the spike regime has escalated to catastrophic levels (grad > 5M), confirming the onset of the late-training second-order gradient cascade documented in [Training\_Instabilities\_in\_Fock-PARFLM\_with\_structured\_V\_theta.md](Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) §23–25.

### 5.2 Spike escalation by training phase

| Step window | Spikes > 100 (logged) | Max grad (logged) | Max grad (terminal) | Notes |
|:-----------:|:---------------------:|:------------------:|:-------------------:|-------|
| 0–10K | 1 | 395 | — | Benign |
| 10K–20K | 2 | 482 | — | Benign |
| 20K–30K | 10 | 2,773 | — | Moderate onset |
| 30K–40K | 27 | 3,843 | — | Escalating |
| 40K–50K | 21 | 1,270 | 15,848 (step 50,051) | Catastrophic spikes emerging |
| 50K–53K | 8 | 1,409 | **5,158,336** (step 51,898) | Catastrophic; watchdog reload |

The escalation is clear: spike frequency peaks around steps 30K–40K, while spike *magnitude* continues to climb, with the truly catastrophic events (grad > 10K) appearing only after step 50K.

### 5.3 Top spike groups (from terminal output)

During catastrophic spikes, the gradient energy is concentrated in the embedding matrices (`E` and `P`) and the Fock mechanism gates:

| Step | Total grad | Top groups |
|-----:|----------:|-----------|
| 51,898 | 5,158,336 | E=3,305,054, P=3,303,969, destruction_gate=1,880,423, reverse_channel_scale=1,206,437 |
| 52,818 | 54,818 | P=51,865, register=13,893, E=9,753, creation_gate=4,588 |
| 50,051 | 15,848 | P=10,878, E=10,878, creation_gate=2,825, register=2,535 |

The pattern is consistent with the second-order gradient cascade: the `create_graph=True` backward pass amplifies gradients through the embedding matrices and Fock gates, with the cascade magnitude growing as the model's internal representations become more refined.

### 5.4 Post-watchdog recovery

The watchdog reload at step 52,064 reverted the model to its best checkpoint (step 45,500, PPL 90.66). Remarkably, the model recovered and set a **new best PPL of 90.50** just 436 steps later at step 52,500. This demonstrates:
1. The watchdog mechanism works as designed — it prevents permanent damage from catastrophic spikes
2. The model's learning capacity is not exhausted despite the instability
3. The gradient cascade is a transient phenomenon: the reloaded weights break the cascade, and the model can resume productive learning

### 5.5 Cross-scale comparison

| Scale | Phase | Max grad | Spikes > 100 | Watchdog reloads | Regime |
|-------|-------|:--------:|:------------:|:----------------:|--------|
| d=384, L=16 | Phase 1 (100K) | 757 | 15 | 0 | Manageable |
| d=384, L=16 | Phase 2 (150K) | 1,703 | 14 | 0 | Manageable |
| **d=768, L=12** | **Phase 1 (53K/100K)** | **5,158,336** | **69+ (logged)** | **1** | **Catastrophic (onset ~50K)** |
| d=768, L=12 | Prior run (late) | 81,019 | many | 1+ | Catastrophic |
| d=1024, L=16 | Full run | 63,949 | frequent | multiple | Catastrophic |

The d=768 run confirms that the catastrophic spike regime is a universal late-training phenomenon at d ≥ 768, not specific to a particular depth (L=12 here vs L=16 in prior runs). The onset at ~50K steps (of 100K) aligns with the prior d=768 run's timing, suggesting the cascade triggers when the model's representations reach a critical level of refinement.

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

$v_{\text{reg}}$ remains stable in a narrow range (~0.001–0.002) throughout training, including through the catastrophic spike regime (steps 49K–53K). This indicates that the hidden-state velocities are well-controlled even when the gradient cascade produces extreme gradient norms — the cascade affects the parameter updates, not the forward-pass dynamics. The lower magnitude compared to d=384 Phase 1 (which reached 0.006 by step 100K) may reflect d=768's smaller $\gamma$ (0.05 vs 0.30) — the weaker damping produces smoother, near-geodesic dynamics with inherently smaller velocity fluctuations.

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
| Phase 1 (in progress) | 4B | ~0.87B | ~22% | No repetition risk |

With a 4B token pool and 100K steps at effective batch 32 × sequence length 512 = 16,384 tokens/step, the full Phase 1 will consume ~1.64B tokens — only 41% of the pool. This ensures no data repetition during Phase 1.

---

## 10. Planned Extensions

Following the graduated extension strategy established with d=384:

| | Phase 2 (planned) | Phase 3 (planned) |
|---|:---:|:---:|
| Steps | 150,000 | 250,000 |
| Token pool | 4B | 4B (or 8B) |
| Resume from | Phase 1 best checkpoint | Phase 2 best checkpoint |
| `FRESH_SCHEDULE` | True | True |
| Projected final PPL | ~40–55 | ~25–35 |

PPL projections are based on the d=384 phase-over-phase reduction pattern (Phase 1→2: 37% reduction, Phase 2→3: TBD), adjusted for the larger model's higher capacity.

---

## 11. Cross-References

- [Fock-PARFLM\_Scale-Up\_Comparative\_Experiments.md](Fock-PARFLM_Scale-Up_Comparative_Experiments.md) — parameter counts and architecture comparison with GPT-2
- [Fock-PARFLM\_d384\_Training\_Results\_OpenWebText.md](Fock-PARFLM_d384_Training_Results_OpenWebText.md) — d=384 training results (reference point for scale comparison)
- [Training\_Instabilities\_in\_Fock-PARFLM\_with\_structured\_V\_theta.md](Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) — gradient spike analysis and cascade mechanism
- [Fock-PARFLM\_Scale-Up\_Gamma\_Sweep\_Results\_and\_Damping\_Regime\_Analysis.md](Fock-PARFLM_Scale-Up_Gamma_Sweep_Results_and_Damping_Regime_Analysis.md) — gamma sweep results and dimension-dependent phase transition
- [Geodesic\_Preservation\_Experiment.md](Geodesic_Preservation_Experiment.md) — geodesic residual analysis and PPL-geodesic coincidence
