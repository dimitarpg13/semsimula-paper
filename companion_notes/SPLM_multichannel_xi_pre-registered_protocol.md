# Pre-registered protocol — E11 multi-channel-ξ SPLM scale-up

> **Status.** Drafted **April 30, 2026**, by Dimitar Gueorguiev with Claude. Pre-registers the first architectural extension to the E9 SPLM scale-up baseline. The motivation is the E9 Phase-1 single-seed result (Δ ≈ −1.04 PPL in favour of MatchedGPT at fixed γ = 0.30 on TinyStories) plus the γ\*-prediction framework's verdict that γ = 0.30 was already near-optimal for the corpus (so the gap is **not** a tuning artefact). E11 attacks the most likely structural cause: the rank-1 information bottleneck of the single causal cumulative-mean ξ.
>
> **Companion documents:**
> - **E9** scale-up RESULTS: [`notebooks/conservative_arch/scaleup/results/RESULTS.md`](../notebooks/conservative_arch/scaleup/results/RESULTS.md)
> - **E10** γ-transfer pre-registration: [`Gamma_transfer_pre-registered_protocol.md`](Gamma_transfer_pre-registered_protocol.md)
> - **γ\*-prediction framework:** [`Determining_optimal_gamma_for_SPLM.md`](Determining_optimal_gamma_for_SPLM.md)
> - **Architectural background:** [`notebooks/conservative_arch/multixi/model_multixi.py`](../notebooks/conservative_arch/multixi/model_multixi.py)

---

## 1. Motivation

E9 Phase-1 single-seed comparison at scale-up (TinyStories, ~5 M BPE train tokens, $L=8$, $d=256$, fixed γ = 0.30) produced

| Arm | Final val PPL | Wall-clock |
|---|---:|---:|
| `splm_em_ln` (E9) | 8.85 | 13.08 h |
| `matched_baseline` (MatchedGPT) | 7.81 | 6.75 h |

Δ⁽⁰⁾ = −1.04 PPL, classified as outcome **B** (|Δ| < §5 effect-size threshold of 5 PPL) by the E9 pre-registration.

The γ\*-prediction framework (`Determining_optimal_gamma_for_SPLM.md`) gives independent evidence that γ = 0.30 was already near-optimal for TinyStories at this configuration:

| Estimator | Predicted γ\*\_TS |
|---|---:|
| §2.1 depth-scaling closed form | 0.299 |
| §2.3 corpus-surprisal scaling | 0.285 |

So the −1.04 PPL gap is **not** explainable by suboptimal γ. The next most likely structural cause is that SPLM's per-token semantic context is encoded in a **single causal cumulative-mean ξ_t = (h_1 + … + h_t) / t**: a *rank-1 summary* of the past in which any two prefixes with the same arithmetic mean are indistinguishable to the energy potential V_θ.

By contrast, attention transformers route information from any past token to any present token via *learned, content-conditioned* weights — effectively giving V_θ-equivalent layers access to a high-rank summary of the past.

E11 tests whether replacing the single ξ with a **multi-channel weighted causal EMA** at multiple decay scales materially closes the −1.04 PPL gap.

---

## 2. Question

**Q1 (primary).** Does multi-channel ξ on a 4-resolution grid (effective horizons ≈ 1, 2, 10, 100 tokens) improve val PPL over the E9 baseline at the same scale-up configuration?

**Q2 (architectural diagnostic).** What do the learned decays $\alpha_k$ converge to? Do they spread across the multi-resolution scale (evidence of multi-resolution use), or do they collapse near 0 or near 1 (evidence that one channel is sufficient)?

**Q3 (gap closure).** Conditional on a positive Q1, does the closed-gap-fraction $1 - (\text{PPL}\_\text{E11} - \text{PPL}\_\text{matched}) / (\text{PPL}\_\text{E9} - \text{PPL}\_\text{matched})$ exceed 0.5? (≥ 50 % gap closure ⇒ "T1 partially closes the gap"; ≥ 0.9 ⇒ "T1 essentially matches"; ≤ 0.1 ⇒ "T1 does not close the gap, the cause lies elsewhere".)

---

## 3. Architecture (locked)

The model is `ScalarPotentialLMSARFMassLNMultiXi` defined in `notebooks/conservative_arch/multixi/model_multixi.py`. Differences from the E9 baseline are summarised here.

### 3.1 Multi-channel ξ

For each of $K$ channels with decay $\alpha_k \in (0, 1)$ the per-token context is

$$\xi^{(k)}_t = \sum_{s \le t} W_k[t, s] \cdot h_s, \qquad W_k[t, s] = \frac{\alpha_k^{(t-s)}}{\sum_{r \le t} \alpha_k^{(t-r)}}.$$

Boundary cases:

- $\alpha_k \to 0$: $\xi^{(k)}_t = h_t$ (instant, no past).
- $\alpha_k \to 1$: $\xi^{(k)}_t = (h_1 + \dots + h_t)/t$ (causal cumulative mean — exactly the E9 baseline ξ).
- intermediate: weighted causal mean with effective horizon $\approx 1/(1 - \alpha_k)$.

The model can therefore *recover the E9 baseline* by driving any one $\alpha_k \to 1$ and zeroing V_θ's dependence on the other channels; this is a graceful-fallback property of the parameterisation.

### 3.2 Locked configuration

| Parameter | Value |
|---|---|
| $K$ | **4** |
| $\alpha$-init | **(0.0, 0.5, 0.9, 0.99)** |
| α-learnable | **yes** (sigmoid parameterisation; decays via $\alpha_k = \sigma(\text{raw}\_\alpha\_k)$) |
| V_θ input dim | $(K+1) \cdot d = 1280$ (vs 512 in E9) |
| V_θ hidden / depth | 1024 / 3 (unchanged from E9) |
| All other architecture | identical to E9 (logfreq mass, $L=8$, $d=256$, LN-after-step, γ fixed at 0.30) |

The 4-channel grid was chosen by reasoning about TinyStories' typical sentence/paragraph structure (sentences ~10–30 tokens, paragraphs ~50–150 tokens; the (1, 2, 10, 100) horizon grid covers all four scales without large overlap). The choice is locked here; we do not search over $K$ or α-init in this experiment.

### 3.3 Parameter cost

| Component | E9 | E11 | Δ |
|---|---:|---:|---:|
| Embedding + position | 12.93 M | 12.93 M | 0 |
| V_θ first layer | 0.53 M | 1.32 M | **+0.79 M** |
| V_θ hidden + output | 2.10 M | 2.10 M | 0 |
| α_k (raw) | 0 | 4 | +4 |
| Mass + γ | 2 | 2 | 0 |
| Other | 0.19 M | 0.19 M | 0 |
| **Total** | **15.75 M** | **16.54 M** | **+0.79 M** |

Confirmed empirically by the smoke run: `params: 16,539,911`. Still 2.91 M below MatchedGPT's 19.45 M, so the comparison is not capacity-skewed in E11's favour.

---

## 4. Anchors and arms

### 4.1 Anchors (no new compute)

| Anchor | Source | Final val PPL | Wall-clock |
|---|---|---:|---:|
| **A. E9 SPLM-em_ln** (single causal mean ξ, fixed γ = 0.30, seed 0) | `notebooks/conservative_arch/scaleup/results/seed0_splm/splm_em_ln_scaleup_scaleup_seed0_summary.md` | 8.85 | 13.08 h |
| **B. E9 MatchedGPT** (matched-attention baseline, seed 0) | `notebooks/conservative_arch/scaleup/results/seed0_attn/matched_baseline_scaleup_scaleup_seed0_summary.md` | 7.81 | 6.75 h |

Both anchor numbers are **locked at this protocol's commit**; no re-running.

### 4.2 New arm

**E11 SPLM-em_ln-multiξ.** Identical to E9 SPLM-em_ln except for the architectural changes in §3. Single seed at the protocol's commit (Stage 1); multi-seed in Stage 2 if Stage 1 lift is material.

---

## 5. Decision rule

Locked at protocol commit. **No retroactive changes.**

### 5.1 Materiality threshold

Define the per-seed paired delta against each anchor:

$$\Delta^{(0)}\_\text{vsE9} = \text{PPL}\_{\text{E9}} - \text{PPL}\_{\text{E11}}, \qquad \Delta^{(0)}\_\text{vsMatched} = \text{PPL}\_{\text{matched}} - \text{PPL}\_{\text{E11}}.$$

(positive ⇒ E11 is better.)

We pre-register **Δ\_min = 0.30 PPL** as the materiality threshold against the E9 anchor. This is calibrated against (i) the seed-1 single-seed measurement uncertainty observed in E5 (~0.15 PPL standard deviation around the optimum on Tiny Shakespeare), (ii) the §3.3 added-parameter cost (~0.79 M extra params buys at most ~0.20 PPL by capacity alone in the matched-param literature for transformers at this scale).

### 5.2 Outcomes

- **Outcome A — material lift over E9.** $\Delta^{(0)}\_\text{vsE9} \ge +0.30$ PPL. Multi-channel ξ is the structural improvement. Trigger Stage 2 (multi-seed paired band). Update `docs/CoT_modeled_via_Semantic_Simulation.md` and `Determining_optimal_gamma_for_SPLM.md` to reflect the new baseline.
- **Outcome B — no material lift.** $|\Delta^{(0)}\_\text{vsE9}| \lt 0.30$ PPL. Multi-channel ξ at this configuration does not move the gap. Move to T3 (state-dependent γ, `docs/Finding_optimal_gamma_for_SPLM.md` §4.2) as the next architectural test. Stage 2 not triggered.
- **Outcome C — material *regression*.** $\Delta^{(0)}\_\text{vsE9} \le -0.30$ PPL. The added V_θ-input width hurts at this training budget (under-trained, capacity-mismatch) and/or multi-channel ξ harms the energy landscape. Investigate (Q2 diagnostic on learned $\alpha_k$) before any further architectural extension.
- **Outcome D — gap-closure milestone.** Conditional on Outcome A, additionally check: $\Delta^{(0)}\_\text{vsMatched} \ge 0$ at any seed in Stage 2 ⇒ "E11 matches MatchedGPT on at least one seed" — promotion-tier headline.

### 5.3 Stage 2 (conditional on Outcome A)

If Stage 1 fires Outcome A, run **two additional seeds (seeds 1 and 2)** of the E11 arm at the same configuration. Compute the paired band $\bar\Delta\_\text{vsE9} \pm s_\Delta / \sqrt{n}$ across $n=3$ seeds.

- $\bar\Delta\_\text{vsE9} - s\_\Delta/\sqrt{3} \ge 0.30$ ⇒ confirmed material lift; promote.
- $\bar\Delta\_\text{vsE9} - s\_\Delta/\sqrt{3} \lt 0.30$ ⇒ Stage 1 lift was a single-seed artefact; do **not** promote.

Stage 2 cost: 2 × ~13 h ≈ 26 h on MPS. Trigger only if Stage 1 fires Outcome A.

---

## 6. Pre-registered subjective predictions

For honesty against the locked decision rule, the team's prior beliefs at protocol commit:

| Outcome | Prior probability |
|---|---:|
| A — material lift (Δ ≥ 0.30 PPL over E9) | **0.55** |
| B — no material lift (\|Δ\| < 0.30) | 0.30 |
| C — material regression (Δ ≤ −0.30) | 0.10 |
| D — additionally matches MatchedGPT (conditional on A) | 0.20 |

The 0.55 prior on A reflects the strength of the rank-1-bottleneck hypothesis (the −1.04 PPL gap is large relative to E5's optimum-flatness; rank-1 → rank-K is a rare *qualitative* architectural change that typically moves PPL > 0.30 in transformer ablations). The 0.30 prior on B reflects the alternative possibility that V_θ as currently sized cannot exploit the higher-rank ξ within 8000 steps. The 0.10 prior on C reflects training-budget mismatch.

---

## 7. Configuration and reproducibility

### 7.1 Trainer

```bash
python3 notebooks/conservative_arch/scaleup/train_splm_em_ln_multixi_scaleup.py \
    --mode scaleup --seed 0 --tag-suffix seed0 \
    --fixed-gamma 0.30 \
    --xi-channels 4 \
    --xi-alpha-inits 0.0,0.5,0.9,0.99 \
    --results-dir notebooks/conservative_arch/scaleup/results/seed0_multixi
```

(Optional `--xi-frozen` to disable α-learning; reserved for ablation, not Stage 1.)

### 7.2 Data and infrastructure

| Item | Value |
|---|---|
| Corpus | TinyStories (HuggingFace `noanabeshima/TinyStoriesV2`), GPT-2 BPE tokeniser, 5 M training tokens, ~4.79 M validation tokens (cached locally at `notebooks/conservative_arch/data/tinystories_gpt2_1files_5000000toks.npz`) |
| Hardware | Apple M-series MPS, 64 GB unified memory, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` |
| Seeds | 0 for Stage 1; 1, 2 added for Stage 2 |

### 7.3 Locked training schedule (identical to E9)

| Parameter | Value |
|---|---|
| Steps | 8000 |
| Warmup | 400 |
| LR (peak, cosine to 0) | 5 × 10⁻⁴ |
| Weight decay | 0.01 |
| Optimiser | AdamW, β = (0.9, 0.95) |
| Gradient clip | 1.0 |
| Batch size / block size | 16 / 512 |
| Eval interval | every 400 steps, 40 batches × batch 16 × block 512 |

### 7.4 Wall-clock estimate

Smoke run measured 0.83 s/step at smoke config (B=8, block=256). Scaling to scale-up (B=16, block=512) gives ~5–6 s/step ⇒ **~12–13 h per run on MPS**. Stage 1: 1 run ≈ 13 h. Stage 2: 2 runs ≈ 26 h.

---

## 8. Reporting plan

When Stage 1 (and, if triggered, Stage 2) finishes, write
`notebooks/conservative_arch/scaleup/results/RESULTS_E11.md` with

1. Final val PPL, wall-clock, full hyper-parameter footprint.
2. Realised α_k trajectory (logged every 50 steps in the JSONL training log).
3. Realised Δ\_vsE9 and Δ\_vsMatched, classified per §5.
4. Headline call (Outcome A / B / C / D).
5. Companion plot: val PPL vs step, three curves overlaid (E9, E11 multi-ξ, MatchedGPT).
6. Diagnostic plot: α_k(step) for the four learnable decays, to inspect Q2 (do they spread, collapse, or migrate?).

If Outcome A and Outcome D both fire, additionally:

- Update `Determining_optimal_gamma_for_SPLM.md` §6 to reflect the new SPLM baseline.
- Update `docs/CoT_modeled_via_Semantic_Simulation.md` §"Looped SPLM" to incorporate multi-channel ξ.
- Draft a `docs/T3_state_dependent_gamma_pre-registered_protocol.md` to chain the next architectural test.

If Outcome B fires, append a brief negative-results note to this file (not a separate doc) with the realised α_k pattern and the team's interpretation, then move to T3.

---

## 9. What this protocol does **not** do

- **Does not search over $K$ or α-init.** Both are locked at the values in §3.2; ablations on these are explicitly out of scope for E11.
- **Does not re-tune γ.** γ is fixed at 0.30 — the same value E10 will report on. If E10 reports γ\*\_TS far from 0.30, a follow-up may rerun E11 at the new γ\*; that is **not** part of this protocol.
- **Does not match parameter count to MatchedGPT.** E11 has 16.54 M; MatchedGPT has 19.45 M. We accept the resulting capacity-asymmetry rather than introducing a parameter-balancing complication; if Outcome A fires *and* the gap is closed, we will run an explicit param-matched control as a follow-up.
- **Does not compare against E9 Phase-2 multi-seed bands.** E9 Phase 2 was deferred (see `SPLM_scaleup_pre-registered_protocol.md` §5 and `notebooks/conservative_arch/scaleup/results/RESULTS.md` §4). The single-seed E9 anchor is the comparison baseline.

---

## 10. Pre-registration commit

This document will be committed before Stage 1 launches. The commit hash is recorded in `notebooks/conservative_arch/scaleup/results/RESULTS_E11.md` once that file is generated.

---

*End of pre-registered protocol.*
