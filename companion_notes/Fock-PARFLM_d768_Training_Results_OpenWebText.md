# Fock-PARFLM d=768 Training Results on OpenWebText

**Author:** Dimitar P. Gueorguiev
**Date:** July 2026
**Status:** In progress — Phase 1 (100K steps) running on LambdaLabs 2×H100; step 48,400 / 100,000

---

## 1. Summary

This document records the training history of the Fock-PARFLM v2.1 model at d=768, L=12 on OpenWebText, driven by the standalone `train_fock.py` script (preset `d768`) on a LambdaLabs 2×H100 instance. This is the first scale-up beyond the d=384 baseline.

| | Phase 1 (in progress) |
|---|:---:|
| Steps | 100,000 |
| Token pool | 4B |
| Tokens consumed (so far) | ~0.79B |
| Wall time (so far) | 27.0 h |
| Best PPL (so far) | **90.66** (step 45,500) |
| Hardware | LambdaLabs 2×H100 (DDP) |

The model is currently at step 48,400 in the WSD stable-LR phase (decay begins at step 65,000). PPL has plateaued around 90–91 during the stable phase; the main gains are expected during the WSD decay phase (steps 65K–100K), consistent with the d=384 training pattern where the decay phase delivered a 37% PPL reduction.

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
bash launch_lambdalabs.sh d768 --multi-gpu --fixed_gamma 0.05
```

This invokes `torchrun --nproc_per_node=2` for DDP across the two H100 GPUs.

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
| Hardware | Colab A100/H100 | LambdaLabs 2×H100 (DDP) | Multi-GPU for throughput |

**Key observation:** The architecture-defining parameters are identical. All differences are scale-appropriate adjustments to training hyperparameters ($d$, $L$, $d_k$, LR, batch logistics) and the gamma value determined by the d=768 gamma sweep.

### 3.3 Training hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW ($\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$) |
| Weight decay | 0.01 |
| Sequence length (`BLOCK_SIZE`) | 512 |
| Effective batch size | 32 (`batch_size=4 × grad_accum=8`) on each GPU, DDP-averaged |
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

### 4.1 Phase 1: 100K steps on 4B tokens (in progress — step 48,400)

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
| 45,500 | **90.66** | **Current best** |
| 48,000 | 90.85 | Latest evaluation |

**Phase 1 observations (interim):**
1. **Rapid initial descent:** PPL drops from 1379 to 186 in the first 5K steps, consistent with d=384 Phase 1 (1482 → 252 at 5K steps).
2. **Stable-phase plateau:** PPL has flattened at ~90–91 for the last ~10K steps. This is expected behaviour during the WSD stable-LR phase — the model is at a steady learning rate and has extracted most of the signal available at this LR. The same pattern was observed at d=384, where the stable phase (steps ~5K–65K) saw progressively slower gains.
3. **WSD decay phase (steps 65K–100K) is the critical window:** At d=384, the decay phase reduced PPL by 37% (from ~101 to ~64). Applying a similar reduction factor to d=768's current ~91 PPL projects a Phase 1 final PPL of **~57–65**.
4. **Throughput:** 7.55 sec/step on 2×H100 (DDP), with an estimated ~108 hours remaining for Phase 1.

---

## 5. Gradient Health

### 5.1 Spike profile (steps 0–48,400)

| Metric | Value |
|--------|:-----:|
| Spikes (grad > 100) | 58 |
| Maximum gradient norm | 3,843.2 |
| Watchdog triggers | 0 |
| Watchdog reloads | 0 |

| Spike band | Count |
|:----------:|:-----:|
| 100–500 | 49 |
| 500–1,000 | 6 |
| 1,000–5,000 | 3 |
| 5,000+ | 0 |

The spike profile at d=768 is moderate and contained. All 58 spikes were self-correcting — per-group gradient clipping truncated the offending component without corrupting the global update. No watchdog intervention was needed.

### 5.2 Comparison with d=384

At the equivalent training stage (step 48K out of 100K), d=768 has ~2× the spike count of d=384 (58 vs ~15 at Phase 1 completion) and a higher max spike magnitude (3,843 vs 757). However, the spikes are still 1–2 orders of magnitude below the catastrophic regime observed during late-training instabilities at d=768 (grad > 80,000) and d=1024 (grad > 60,000). The question is whether the spike regime will escalate as training enters the WSD decay phase — this was where catastrophic spikes emerged in the prior d=768 run.

### 5.3 Cross-scale comparison

| Scale | Phase | Max grad | Spikes > 100 | Watchdog reloads | Regime |
|-------|-------|:--------:|:------------:|:----------------:|--------|
| d=384, L=16 | Phase 1 (100K) | 757 | 15 | 0 | Manageable |
| d=384, L=16 | Phase 2 (150K) | 1,703 | 14 | 0 | Manageable |
| **d=768, L=12** | **Phase 1 (48K/100K)** | **3,843** | **58** | **0** | **Moderate** |
| d=768, L=12 | Prior run (late) | 81,019 | many | 1+ | Catastrophic |
| d=1024, L=16 | Full run | 63,949 | frequent | multiple | Catastrophic |

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

$v_{\text{reg}}$ settles to a narrow range (~0.001–0.002) after initial transients, indicating stable hidden-state velocities. This is consistent with the d=384 pattern but at a lower magnitude (d=384 Phase 1 reached 0.006 by step 100K). The lower $v_{\text{reg}}$ may reflect d=768's smaller $\gamma$ (0.05 vs 0.30) — less damping means the velocity penalty is less actively penalised but the velocities are also inherently smaller due to the smoother, near-geodesic dynamics.

### 6.2 $\xi$ channel utilisation

The training log for the d=768 run (driven by `train_fock.py`) does not record per-step $\xi$ alpha values — this data is only available from the Colab notebooks which log the `alpha` vector at each evaluation step. The $\xi$ channels are initialised to `[0.50, 0.75, 0.95, 0.99, 0.995]` and are expected to follow the same hierarchical specialisation pattern observed at d=384.

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
| Phase 1 (in progress) | 4B | ~0.79B | ~20% | No repetition risk |

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
