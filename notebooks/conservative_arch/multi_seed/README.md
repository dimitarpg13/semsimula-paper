# `multi_seed/` -- Multi-seed variance harness for SPLM experiments

Reusable harness that re-runs an existing trainer (SPLM or matched
baseline) with N different random seeds and aggregates the per-seed
training logs into a mean ± std summary plus an overlay loss-curve
plot.

This is **E1** of the SPLM-strengthening programme captured in
[`docs/Next_Model_Experiments_for_SPLM.md`](../../../docs/Next_Model_Experiments_for_SPLM.md)
(extended with multi-seed variance, which the catalogue takes for
granted but every existing experiment is missing). It directly addresses
the *"Multiple seeds / variance"* gap flagged in
[`docs/tmlr_review_summary_chatgpt.md`](../../../docs/tmlr_review_summary_chatgpt.md).

## Why this exists

Every published SPLM perplexity number to date (val ppl 287 on the §1
SPLM, 160.55 on the `sarf_mass_variant logfreq`, etc.) comes from a
single seed=0 run. A TMLR reviewer can correctly say "these numbers
have no error bars; we have no way to know if the gap to baseline is
statistically meaningful". The harness here closes that gap.

## Files

| file | purpose |
|---|---|
| [`multi_seed_runner.py`](multi_seed_runner.py) | Subprocess-driven N-seed launcher; calls an existing trainer once per seed and moves its outputs into a seed-namespaced subdir. |
| [`multi_seed_aggregator.py`](multi_seed_aggregator.py) | Reads each seed's training log, computes mean/std/min/max of final val ppl + intermediate eval points, writes a markdown report and a multi-seed loss-curve overlay plot. |
| `results/` | Populated by runs. One subdir per `(model_label,)` pair; one inner subdir per seed. |

## Supported model specs

The harness understands six model specs (the original three plus the
three Scope-3 cells added 2026-05-09):

- `splm_baseline` — the §1 fixed-$\xi$ SPLM (no SARF $\xi$
  recomputation, single global mass scalar). v2 single-seed val ppl
  287.43; leak-immune by construction. Trainer:
  `notebooks/conservative_arch/train_splm.py`.
- `splm_sarf` — SARF-faithful SPLM with $\xi$ recomputed per layer
  and a single global mass scalar. v2 single-seed val ppl 192.21.
  Trainer: `notebooks/conservative_arch/sarf_variant/train_splm_sarf.py`.
- `splm_sarfmass_embed_head` — SARF-faithful SPLM with per-token mass
  from a learned linear head over the token embedding (variant A of the
  per-token mass ablation). v2 single-seed val ppl 222.91. Trainer:
  `notebooks/conservative_arch/sarf_mass_variant/train_splm_sarf_mass.py --mass-mode embed_head`.
- `splm_sarfmass_logfreq` — SARF-faithful SPLM with per-token mass
  from the Shannon-surprisal prior (variant B). v2 single-seed val
  ppl 160.55. Trainer:
  `notebooks/conservative_arch/sarf_mass_variant/train_splm_sarf_mass.py --mass-mode logfreq`.
- `splm_em_ln` — SPLM with LayerNorm-after-step (variant `ln` in
  `notebooks/conservative_arch/energetic_minima/`). Each semi-implicit
  damped step is followed by an affine-free LayerNorm projection back
  to the unit shell, which gives $V_\theta$ a finite minimum on a
  compact manifold. **The strongest Tiny Shakespeare result in the
  repo so far: val ppl 88.63 at seed=0** (under the v2 buggy
  integrator; v4 leak-free 5-seed mean is ~178–181 PPL at fixed
  $\gamma \in [0.10, 0.20]$, see
  [`../ln_damping_sweep/results/leakfree_5seed_confirmation/`](../ln_damping_sweep/results/leakfree_5seed_confirmation/)).
  Trainer: `notebooks/conservative_arch/energetic_minima/train.py --variant ln`.
- `matched_baseline` — 8 M-param tiny GPT-2-style decoder matched to
  SPLM on $d$, $L$, $V$, data budget. v2 5-seed mean val ppl
  149.80 ± 7.21; leak-immune by construction (no SPLM integrator).
  Trainer: `notebooks/conservative_arch/train_matched.py`.

Adding more specs is a 10-line entry in `MODEL_SPECS` of
[`multi_seed_runner.py`](multi_seed_runner.py); the harness itself is
model-agnostic.

## Scope-3 retrain (paper\_v4)

The first four `splm_*` specs above plus `matched_baseline` are
collectively the **Scope-3 retrain** of `paper_v4`: the v2 SPLM-family
experiments rerun under the leak-free integrator
(`cfg.causal_force = True`, the post-fix default). See
[`SCOPE3_README.md`](SCOPE3_README.md) for the full story; the
turn-key driver is [`colab_scope3.ipynb`](colab_scope3.ipynb), which
runs all 5 cells × 5 seeds in 2–4 h on a Colab GPU and emits both the
standard multi-seed report and a Scope-3-specific
v2-vs-v4 headline table via [`scope3_comparison.py`](scope3_comparison.py).

## Quick reference

```bash
# 0. (Once) Precompute the surprisal lookup table for SPLM logfreq mode.
python3 notebooks/conservative_arch/sarf_mass_variant/compute_unigram_frequencies.py

# 1. Smoke test (single seed, smoke mode -- ~1-2 minutes total).
python3 notebooks/conservative_arch/multi_seed/multi_seed_runner.py \
    --mode smoke --n-seeds 1 --models splm_sarfmass_logfreq

# 2. Production run for E1 (5 seeds x 3 models on Tiny Shakespeare,
#    expect ~7-8 hours wall-clock on MPS).
python3 notebooks/conservative_arch/multi_seed/multi_seed_runner.py \
    --mode shakespeare --n-seeds 5 \
    --models splm_em_ln,splm_sarfmass_logfreq,matched_baseline

# 3. Aggregate logs into a markdown report + overlay loss curves.
python3 notebooks/conservative_arch/multi_seed/multi_seed_aggregator.py \
    --tag E1_shakespeare
```

## Output layout

```
multi_seed/results/
└── E1_shakespeare/                    # name passed via --tag
    ├── splm_em_ln/
    │   ├── seed_0/
    │   │   ├── em_ln_shakespeare_training_log.jsonl
    │   │   ├── em_ln_shakespeare_ckpt_latest.pt
    │   │   ├── em_ln_shakespeare_loss_curve.png
    │   │   └── em_ln_shakespeare_summary.md
    │   └── seed_{1..4}/...
    ├── splm_sarfmass_logfreq/
    │   ├── seed_0/
    │   │   ├── splm_sarfmass_logfreq_shakespeare_training_log.jsonl
    │   │   ├── splm_sarfmass_logfreq_shakespeare_ckpt_latest.pt
    │   │   ├── splm_sarfmass_logfreq_shakespeare_loss_curve.png
    │   │   └── splm_sarfmass_logfreq_shakespeare_summary.md
    │   └── seed_{1..4}/...
    ├── matched_baseline/
    │   ├── seed_0/...
    │   └── ...
    ├── E1_report.md                   # main deliverable: table + interpretation
    ├── E1_loss_curves_splm_em_ln.png  # overlay of all seeds, LN variant
    ├── E1_loss_curves_splm.png        # overlay of all seeds, SARF+mass
    └── E1_loss_curves_matched.png     # overlay of all seeds, matched
```

## What "E1 done" looks like

A markdown report `multi_seed/results/E1_shakespeare/E1_report.md` with:

| model | n seeds | val ppl mean ± std | val ppl min | val ppl max | gap vs matched (95% CI) |
|---|---|---|---|---|---|
| SPLM `em_ln` (LayerNorm-after-step), $d=128$, $L=8$ | 5 | … | … | … | (vs baseline) |
| SPLM `sarf_mass_variant logfreq`, $d=128$, $L=8$ | 5 | … | … | … | (vs baseline) |
| Matched transformer, $d=128$, $L=8$ | 5 | … | … | … | (reference) |

…plus a short interpretation: do the headline ppl numbers (88.63 for
LN, 160.55 for SARF+mass) fall inside one std of their respective
means? Are the SPLM-vs-baseline and LN-vs-SARF+mass gaps statistically
meaningful at n=5? The pairwise Welch t-test results are reported with
95% CIs.

## Non-goals at v0

- **No new model architectures.** The harness only re-runs existing
  trainers. Adding new variants (E2 width sweep, E3 energy-drift) goes
  in adjacent directories that *use* this harness.
- **No HPC orchestration.** All runs are local subprocesses on the
  configured device (MPS / CUDA / CPU). Cluster-style launching (Slurm,
  Ray, etc.) is out of scope at v0.
- **No automatic statistical testing.** The aggregator reports
  descriptive statistics and a t-test gap with 95% CI; deciding what
  counts as "meaningful" stays with the human reviewer.
