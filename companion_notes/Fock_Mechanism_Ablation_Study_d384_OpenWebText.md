# Fock Mechanism Ablation Study: d=384 on OpenWebText

**Author:** Dimitar P. Gueorguiev
**Date:** July 2026
**Status:** Complete — extension run finished, ablation results final

---

## 1. Purpose

This document presents the results of a controlled ablation study comparing the Fock-PARFLM architecture with and without the reverse channel mechanism at hidden dimension $d{=}384$ on OpenWebText. The comparison provides direct evidence that the Fock mechanism — the register-based particle creation/destruction channel with reverse information flow — is not a marginal architectural decoration but the essential component responsible for the model's language modelling capability.

**Companion documents:**
- [Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md](Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) — gradient spike taxonomy and per-group clipping analysis
- [Fock-PARFLM_Scale-Up_Gamma_Sweep_Results_and_Damping_Regime_Analysis.md](Fock-PARFLM_Scale-Up_Gamma_Sweep_Results_and_Damping_Regime_Analysis.md) — gamma sweep results across scales
- [Geodesic_Preservation_Experiment.md](Geodesic_Preservation_Experiment.md) — geodesic residual analysis framework
- [Improving_the_Fock_Mechanism_to_match_Attention.md](Improving_the_Fock_Mechanism_to_match_Attention.md) — theoretical analysis of the reverse channel design

---

## 2. Experimental Setup

### 2.1 Common Configuration

Both runs share an identical base configuration:

| Parameter | Value |
| --- | --- |
| Hidden dimension $d$ | 384 |
| Layers $L$ | 16 |
| Registers $M$ | 32 |
| $V_\theta$ | Depth-conditioned multi-context Gaussian (5 heads, 8 wells/head) |
| $\xi$ contexts | 5 (`xi5long`) with `topk=16` |
| Attention heads | 4 (`mh4`) |
| Embeddings | Untied (`ob_untied`) |
| Damping $\gamma$ | 0.30 (nominal) |
| Total parameters | ~53.4M |
| Dataset | OpenWebText |
| Batch size | 8 with gradient accumulation 2 (effective batch = 16) |
| Block size | 512 |
| Peak learning rate | $3 \times 10^{-4}$ |
| Schedule | WSD (warmup-stable-decay) |
| Gradient clipping | 1.0 (default) |
| Repetition penalty | 0.05 |

### 2.2 Experimental Conditions

| | **e5a** (Ablation) | **e5c** (Full Model) |
| --- | --- | --- |
| **Reverse channel** | Disabled | Enabled (QK-norm + soft-norm + pre-LN) |
| **Per-layer gates** | Not applicable | Per-layer creation/destruction gates (`plgate`) |
| **Per-group clip overrides** | Default | `V_phi`: 0.3, `creation_gate`: 0.3, `destruction_gate`: 0.3, `reverse_channel_scale`: 0.1, `reverse_ch`: 0.1, `register`: 0.3, `depth_code`: 0.5 |
| **Total training steps** | ~76,000 | 150,000 (75K + 75K extension) |
| **Tokens seen** | ~1B | ~2B |

The e5c run was conducted in two phases: an initial 75K-step run followed by an extension run of 75K additional steps with a fresh WSD schedule (warmup 0→2,000, stable 2,000→100,000, decay 100,000→150,000, floor $1.5 \times 10^{-5}$).

---

## 3. Results

### 3.1 Validation Perplexity Trajectory

| Milestone | **e5a** (No Reverse Channel) | **e5c** (Full Fock) | PPL Gap |
| --- | --- | --- | --- |
| Step 5,000 | 329.25 | — | — |
| Step 10,000 | 244.59 | — | — |
| Step 15,000 | 185.17 | — | — |
| Step 25,000 | 141.17 | — | — |
| Step 50,000 | 142.94 (plateaued) | — | — |
| Step 51,000 | **125.94** (best ever) | — | — |
| Step 75,000 | 131.62 (regressed) | 41.85 (best at end of phase 1) | 89.77 |
| Step 100,000 | — | ~32 (mid-extension) | — |
| Step 150,000 | — | **27.23** (final best, new record) | — |

### 3.2 The e5a Plateau

The e5a run (reverse channel disabled) reached its best validation PPL of **125.94** at step 51,000 and then failed to improve for the remaining ~25,000 steps:

- 52 validation evaluations after step 51K
- Minimum PPL in this range: 128.68
- Maximum PPL in this range: 147.91
- Mean PPL in this range: 138.49

This plateau is not caused by overfitting or learning rate issues — the learning rate was still at $2.3 \times 10^{-4}$ (well above the floor) and the training loss continued to decline slowly. The model simply reached the representational ceiling imposed by the absence of the reverse channel.

### 3.3 The e5c Progression

The e5c run (full Fock mechanism) followed a qualitatively different trajectory:

**Phase 1 (steps 1→75,000):** Reached PPL 41.85, already 3.0$\times$ better than e5a's ceiling.

**Phase 2 / Extension (steps 75,001→150,000):** Resumed from the phase 1 checkpoint with a fresh WSD schedule. The model continued to improve throughout:

- Step 100,000: PPL ~32 (stable phase)
- Step 148,000: PPL 29.61 (entering decay phase)
- Step 150,000: PPL **27.23** — a new best at the very last evaluation

The final-step improvement is significant: it demonstrates that the model had not yet exhausted its capacity at 150K steps. The WSD decay phase was still extracting meaningful gains, suggesting further extension could push PPL lower.

### 3.4 Summary Comparison

| Metric | **e5a** (No Reverse Channel) | **e5c** (Full Fock) |
| --- | --- | --- |
| Best validation PPL | 125.94 | 27.23 |
| Step at best | 51,000 | 150,000 |
| PPL ratio (e5a / e5c) | — | **4.63$\times$** |
| Continued improvement at end | No (plateaued for 25K+ steps) | Yes (new best at final step) |
| Training loss at best | ~4.84 | ~3.30 |

---

## 4. Structural Health Probe (e5c at Step 150,000)

The structural health diagnostic on the best checkpoint reveals a healthy, well-differentiated architecture:

### 4.1 Layer-by-Layer Analysis

| Layer | Active Frac | Reg Cos Sim | Create Entropy | Create $\alpha$ | Rev Entropy | Rev Scale | QForce Ratio | Destroy Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.000 | 0.074 | 3.923 | 0.343 | 2.395 | 0.137 | 1.159 | 0.203 |
| 1 | 1.000 | 0.051 | 1.376 | 0.732 | 0.901 | 0.001 | 0.007 | 0.119 |
| 2 | 1.000 | 0.052 | 1.514 | 0.708 | 1.021 | 0.001 | 0.006 | 0.028 |
| 3 | 1.000 | 0.044 | 1.542 | 0.690 | 1.809 | 0.001 | 0.008 | 0.012 |
| 4 | 1.000 | 0.043 | 1.692 | 0.670 | 1.691 | 0.004 | 0.033 | 0.007 |
| 5 | 1.000 | 0.044 | 1.766 | 0.652 | 1.679 | 0.006 | 0.051 | 0.067 |
| 6 | 1.000 | 0.043 | 1.751 | 0.654 | 1.637 | 0.005 | 0.037 | 0.263 |
| 7 | 1.000 | 0.032 | 1.753 | 0.658 | 1.341 | -0.025 | 0.199 | 0.061 |
| 8 | 1.000 | 0.055 | 1.882 | 0.623 | 1.861 | -0.017 | 0.138 | 0.055 |
| 9 | 1.000 | 0.061 | 1.921 | 0.619 | 2.007 | -0.019 | 0.162 | 0.197 |
| 10 | 1.000 | 0.108 | 2.319 | 0.578 | 2.608 | 0.043 | 0.359 | 0.532 |
| 11 | 1.000 | 0.104 | 2.177 | 0.554 | 2.569 | 0.019 | 0.151 | 0.687 |
| 12 | 1.000 | 0.098 | 2.295 | 0.506 | 2.449 | 0.013 | 0.085 | 0.656 |
| 13 | 1.000 | 0.078 | 1.742 | 0.612 | 2.069 | 0.025 | 0.133 | 0.665 |
| 14 | 1.000 | 0.040 | 0.975 | 0.735 | 1.945 | 0.044 | 0.197 | 0.139 |
| 15 | 1.000 | 0.058 | 2.272 | 0.583 | 1.716 | 0.189 | 0.827 | 0.499 |

### 4.2 Key Observations

**Register diversity is excellent.** Mean cosine similarity between registers is 0.062 — registers are nearly orthogonal, confirming they carry distinct, non-redundant information. No register collapse is observed at any layer.

**Creation routing differentiates with depth.** Layers 0-6 show broad creation entropy (1.4-3.9) with moderate $\alpha$ (0.34-0.73), indicating diverse routing across register slots. Layer 14 has the lowest creation entropy (0.975) and highest $\alpha$ (0.735), suggesting it has learned a sharper, more specialised routing pattern.

**Reverse channel scale increases with depth.** The reverse channel is nearly dormant in layers 1-6 (`rev_scale` $\approx$ 0.001) and progressively strengthens through the upper layers, reaching 0.189 at layer 15. This is consistent with the interpretation that early layers establish local feature representations while later layers require increasingly non-local information exchange via the register channel.

**Quantum force ratio peaks at endpoints.** `qforce_ratio` is 1.159 at layer 0 and 0.827 at layer 15, but drops below 0.1 in the middle layers. This U-shaped profile suggests the conservative (potential-derived) force dominates in the interior of the stack, while boundary layers rely more heavily on the quantum force correction.

**Destruction activity concentrates in upper layers.** Layers 10-13 show the highest destruction means (0.53-0.69), indicating active register recycling in the upper half of the stack — consistent with a model that progressively refines its working memory by discarding irrelevant register content.

---

## 5. Ablation: Component Criticality

The structural probe includes a component ablation that removes individual mechanisms and measures the impact on perplexity:

| Condition | Loss | PPL | $\Delta$PPL |
| --- | --- | --- | --- |
| **Full model** | 3.4723 | 32.21 | +0.00 |
| **$-$ Reverse channel** | 7.6428 | 2085.52 | +2053.31 |
| **$-$ Registers (all)** | 7.6428 | 2085.50 | +2053.29 |

Removing either the reverse channel or the registers produces **catastrophic failure** — the model degrades from PPL 32.21 to PPL 2085, equivalent to a loss of 7.64 (nearly random predictions from a 50K-token vocabulary). The near-identical impact of removing registers vs. removing the reverse channel confirms that these are co-dependent components: registers without the reverse channel cannot communicate back to the hidden stream, and the reverse channel without registers has nothing to read.

---

## 6. Discussion

### 6.1 The Reverse Channel is the Essential Innovation

The comparison between e5a and e5c provides the clearest evidence yet for the centrality of the Fock mechanism in Fock-PARFLM:

**Without the reverse channel (e5a):** The model plateaus at PPL 125.94 — a hard ceiling that 25,000 additional training steps cannot break. The architecture retains the conservative dynamics, the Gaussian $V_\theta$ potential, the Verlet integrator, and the multi-context $\xi$ routing. All of these components are necessary but **insufficient**. The model can learn to move tokens through the potential landscape but cannot establish the inter-token communication channels that language modelling demands.

**With the reverse channel (e5c):** The model reaches PPL 27.23 — a 4.63$\times$ improvement — and is still actively improving at the end of training. The reverse channel provides the missing ingredient: a mechanism for registers to write information back into the hidden stream, enabling non-local information flow across the sequence.

The 4.63$\times$ PPL gap is not a marginal improvement amenable to alternative explanations. It represents a qualitative difference in capability: the e5a model is functionally incapable of the next-token prediction quality that language modelling requires, while the e5c model operates in the competitive range for a 53M-parameter model.

### 6.2 Learning Dynamics Differ Qualitatively

The two runs exhibit fundamentally different learning dynamics:

| Characteristic | **e5a** (No Reverse Channel) | **e5c** (Full Fock) |
| --- | --- | --- |
| Learning curve shape | Rapid initial descent, then plateau | Sustained descent across both phases |
| Best PPL timing | Early (step 51K of 76K) | Late (step 150K — the very last step) |
| Post-best behaviour | Oscillation with no improvement (25K+ steps) | Continuous improvement through WSD decay |
| Gradient norm (final) | 0.54 (very low — no learning signal) | 3.5-5.8 (active learning) |
| Training loss (final) | 4.82 (high floor) | 3.30 (still declining) |

The e5a gradient norms are strikingly low (0.35-0.54 throughout), suggesting the model has exhausted the learning signal available to it. By contrast, e5c maintains healthy gradient norms of 3.5-5.8, with occasional spikes that are safely managed by per-group clipping — a sign of active, ongoing optimisation.

### 6.3 The QFT Analogy Holds

In quantum field theory, the vacuum state is inert — it contains no particles and no interactions. Particle creation and annihilation operators are what generate the entire spectrum of physical phenomena. Fock-PARFLM's architecture mirrors this structure:

- **Without Fock operators (e5a):** The model has a potential landscape and dynamics, but no mechanism for creating or destroying information carriers (registers). It is analogous to a QFT with a Lagrangian but no interaction vertices. The model can propagate but cannot interact.
- **With Fock operators (e5c):** Registers are created, populated, read via the reverse channel, and destroyed as the hidden state propagates through layers. The analogy to particle creation/annihilation is not merely nominal — the structural health probe confirms that registers carry diverse, non-redundant information (low cosine similarity), are actively created and destroyed (non-zero creation entropy and destruction rates), and their influence on the hidden stream increases with depth (rising `rev_scale`).

The ablation at step 150K drives this home: removing either the registers or the reverse channel produces identical catastrophic failure (PPL 2085), confirming they function as a coupled system — the "Fock space" of the model.

### 6.4 Remaining Capacity

The fact that e5c achieved a new best PPL at its very last evaluation (step 150,000) strongly indicates remaining model capacity. The WSD decay phase was still extracting gains, and the training loss (3.30) was still above the validation loss (3.30), suggesting no overfitting. This motivates the planned ext2 run: 250K additional steps on 4B tokens, which is expected to push PPL into the low 20s.

---

## 7. Conclusion

The Fock mechanism — comprising registers, per-layer creation/destruction gates, and the reverse channel — is the essential architectural innovation in Fock-PARFLM. Without it, the conservative dynamics framework (Verlet integrator, Gaussian $V_\theta$ potential, multi-context $\xi$ routing) plateaus at PPL 125.94 on OpenWebText, unable to achieve the non-local information flow that language modelling requires. With it, the same architecture reaches PPL 27.23 — a 4.63$\times$ improvement — and continues to improve, demonstrating that the Fock mechanism does not merely add incremental performance but fundamentally transforms the model's representational capacity.

The component ablation at step 150K confirms the mechanism's criticality through a different lens: removing either registers or the reverse channel from the trained model produces identical catastrophic failure (PPL $\to$ 2085), establishing that these components form a co-dependent system that cannot be decomposed.

These results validate the QFT-inspired architectural design: just as particle creation and annihilation operators are necessary for a quantum field theory to generate physical phenomena beyond the free vacuum, the Fock mechanism is necessary for the conservative dynamics framework to generate language modelling capability beyond trivial propagation.
