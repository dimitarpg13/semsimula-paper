# E9 — SPLM scale-up de-risking experiment

**Status:** Phase 1 in progress (single seed, both arms sequential)
**Pre-registered protocol:** [docs/SPLM_scaleup_pre-registered_protocol.md](../../../docs/SPLM_scaleup_pre-registered_protocol.md)
**Pre-registration commit:** `17a3795` (April 29, 2026)

## Question

Does the +25 PPL gap that SPLM em_ln (γ⋆=0.30) showed over a parameter-matched
GPT-2 attention baseline on Tiny Shakespeare at ~7-8 M params survive a
**2.2× model scale-up** to ~16-19 M params on a **16× larger corpus** at a
**4× longer context length**?

## Arms

| arm                | model                        | params  | architecture                                            |
| ------------------ | ---------------------------- | ------- | ------------------------------------------------------- |
| `splm_em_ln`       | `ScalarPotentialLMSARFMassLN` | 15.75 M | d=256, L=8, v_hidden=1024, max_len=1024, mass=`logfreq`, ln_after_step, fixed γ=0.30 |
| `matched_baseline` | `MatchedGPT`                  | 19.45 M | d=256, L=8, n_head=4, mlp_mult=4, max_len=1024, tied embeddings |

## Configuration

- **Corpus:** TinyStories, GPT-2 BPE, ~5 M training tokens, ~140 k validation tokens
- **Context:** `max_len=1024` for both models, `block_size=512` for training samples
- **Optimisation:** AdamW lr=5e-4 (cosine, 400-step warmup), weight_decay=0.01, betas=(0.9, 0.95), grad_clip=1.0
- **Training budget:** 8000 steps × batch 16 × block 512 = ~65.5 M tokens seen (~13 epochs over 5 M token train split)
- **Evaluation:** every 400 steps, 40 batches × 16 × 512 = ~327 k tokens
- **Hardware:** Apple M-series MPS, 64 GB unified memory, single seed per arm

## Files

| file                                       | purpose                                                                                       |
| ------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `compute_unigram_frequencies_tinystories.py` | One-off: precompute -log p_hat(v) over the GPT-2 BPE vocabulary using the TinyStories train split. Saves to `results/logfreq_surprisal_tinystories.npy`. |
| `train_splm_em_ln_scaleup.py`              | Adapted SPLM em_ln trainer: single TinyStories scale-up mode, fixed γ, MPS-friendly.          |
| `train_matched_baseline_scaleup.py`        | Adapted MatchedGPT trainer: single TinyStories scale-up mode, MPS-friendly.                   |
| `train_helmholtz_scaleup.py`               | Pilot-extension: Helmholtz Q9d (AAAASSSS) at the E9 scale-up config. Used by `paper_tmlr_1`.  |
| `train_hybrid_scaleup.py`                  | Pilot-extension: Hybrid Variant A (k=4, m=4) at the E9 scale-up config. Used by `paper_tmlr_1`. |
| `train_parf_scaleup.py`                    | Pilot-extension: PARF Q9c sparse top-k=4 at the E9 scale-up config. Used by `paper_tmlr_1`.   |
| `aggregate_pilot_results.py`               | Aggregator: writes `PILOT_RESULTS.md`, `pilot_loss_curves.png`, `pilot_pareto.png` from per-arm artifacts. |
| `colab_pilot.ipynb`                        | Colab driver notebook for the 5-cell pilot (mount Drive, clone repo, smoke + full pilot, aggregate). |
| `PILOT_README.md`                          | Pilot-specific README: configuration, decision rule, wall-clock estimates, Colab usage.       |
| `results/`                                 | Per-arm training logs, checkpoints, loss curves, summary md.                                  |

## `paper_tmlr_1` 5-cell pilot extension

A focused 5-cell pilot extending E9 with three additional architectural arms
(Helmholtz Q9d, Hybrid Variant A, PARF Q9c sparse k=4) at the **same** E9
scale-up configuration (d=256, L=8, max_len=1024, TinyStories, 8000 steps).
Designed to run end-to-end in one Colab Pro+ A100 session (~18 h total).

See [`PILOT_README.md`](./PILOT_README.md) for full details and
[`colab_pilot.ipynb`](./colab_pilot.ipynb) for the Colab driver notebook.

## Decision rule (locked at pre-registration)

Let Δ = PPL(`matched_baseline`) − PPL(`splm_em_ln`) (pooled over completed seeds).
- **Outcome A** — Δ > +Δ_min ⇒ SPLM beats matched-attention at scale. ✅ paper claim survives.
- **Outcome B** — |Δ| ≤ Δ_min ⇒ tie at scale. Paper still publishable; gap softens to "matched, not superior".
- **Outcome C** — Δ < −Δ_min ⇒ baseline wins. Paper claim narrows to "small-scale only"; honest disclosure.

**Δ_min = 5.0 PPL**, selected before any training run.

## Adaptive seed plan

- **Phase 1:** seed 0, both arms sequentially (~24-30 h wall-clock).
  - If `|Δ⁽⁰⁾| ≥ 20 PPL` → stop (single-seed evidence is decisive).
  - Else proceed to Phase 2.
- **Phase 2:** seeds 1 + 2, both arms (~48-60 h additional). Pool with seed 0 for the final paired comparison.

Pre-registered subjective probabilities (informal): A: 0.65, B: 0.25, C: 0.10.
Pre-registered point prediction: Outcome A, Δ ∈ [+10, +30] PPL, most likely +20 PPL.

## Reporting

Final results are written into `RESULTS.md` here. Once both arms have completed
seed 0, a "Phase 1 outcome" decision rule application is recorded; if Phase 2
is triggered the post-Phase-2 outcome supersedes it.
