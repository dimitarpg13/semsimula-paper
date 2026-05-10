# Scope-3 retrain (paper\_v4) — multi-seed harness for Colab GPU

This README accompanies [`colab_scope3.ipynb`](colab_scope3.ipynb): the
Google-Colab notebook that re-runs the **Scope-3 SPLM-family experiments**
of `paper_v4` under the **leak-free integrator**
(`cfg.causal_force = True`, the post-fix default of every SPLM-family
config).

The motivation is captured in
[`docs/Plan_for_first_TMLR_paper_v1.md`](../../../docs/Plan_for_first_TMLR_paper_v1.md)
§1.5 ("Scope 3 retrain — 60–100 h MPS"). Running this on Colab GPU
collapses that estimate to **2–4 h** and unblocks the `paper_v4` freeze.

## What is Scope 3?

Five Tiny Shakespeare cells need new multi-seed numbers under v4:

| cell | trainer | leak-immune by construction? | role |
|---|---|:--:|---|
| `splm_baseline` | `train_splm.py` | yes (fixed-$\xi$, no recomputation from $h$) | leak-immunity sanity check + 4-cell mass-ablation row 1 |
| `splm_sarf` | `sarf_variant/train_splm_sarf.py` | no (SARF $\xi$ recomputed per layer) | 4-cell mass-ablation row 2; the v2 numbers reported "33 % PPL reduction over fixed-$\xi$" needs to be re-verified |
| `splm_sarfmass_embed_head` | `sarf_mass_variant/train_splm_sarf_mass.py --mass-mode embed_head` | no | variant A of the per-token mass ablation |
| `splm_sarfmass_logfreq` | `sarf_mass_variant/train_splm_sarf_mass.py --mass-mode logfreq` | no | variant B (Shannon-surprisal prior); the previous flagship at val PPL 160.55 |
| `matched_baseline` | `train_matched.py` | yes (no SPLM integrator at all) | n=5 pairing target for SPLM-vs-baseline absolute-quality claim |

The `splm_em_ln` flagship (LayerNorm-after-step) is **not** part of
Scope 3 — it has already been retrained leak-free at $S=5$ in
[`notebooks/conservative_arch/ln_damping_sweep/results/leakfree_5seed_confirmation/`](../ln_damping_sweep/results/leakfree_5seed_confirmation/).

## What gets produced

After the notebook completes, the run directory
`results/scope3_shakespeare/` (default tag) contains:

```
results/scope3_shakespeare/
├── splm_baseline/
│   ├── seed_0/   ├── seed_1/   ├── seed_2/   ├── seed_3/   ├── seed_4/
├── splm_sarf/                ... (same per-seed layout)
├── splm_sarfmass_embed_head/ ...
├── splm_sarfmass_logfreq/    ...
├── matched_baseline/         ...
├── run_log.jsonl                            # per-(model,seed) launch log
├── scope3_shakespeare_report.md             # multi_seed_aggregator output
├── scope3_shakespeare_loss_curves_*.png     # per-cell overlay plots
└── scope3_comparison.md                     # v2-vs-v4 headline table
```

Each `seed_<s>/` subdir contains the trainer's four fixed-name artefacts,
namespaced into the seed slot:

* `<tag>_training_log.jsonl` — per-step train + every-200-step eval rows
* `<tag>_ckpt_latest.pt` — model + optimiser checkpoint
* `<tag>_loss_curve.png` — train / val loss curve
* `<tag>_summary.md` — human-readable convergence report

The two top-level reports are the deliverables that go into `paper_v4`
§15:

1. **`scope3_shakespeare_report.md`** — per-cell mean ± std + Welch's
   pairwise table on final val PPL (the standard multi-seed report).
2. **`scope3_comparison.md`** — Scope-3-specific narrative: v2 buggy vs
   v4 leak-free PPL, asymmetric inflation factor per cell, qualitative
   direction preservation check, and four explicit decision rules for
   updating `paper_v4` §15.17 / §15.19.

## How to run on Colab

1. Open `colab_scope3.ipynb` in Google Colab (File → Upload notebook, or
   point Colab at the GitHub URL).
2. Runtime → Change runtime type → **GPU**. Any of the available tiers
   works; expected wall-clock for the full $5 \times 5 = 25$ runs:

| GPU | wall-clock for 25 runs | recommended |
|---|---|:--:|
| T4 (free Colab) | ~3.5 h | yes — safest free option |
| L4 (Colab Pro) | ~2.5 h | yes — best free-tier-Pro pick |
| A100 / H100 (Pro+) | 1–1.5 h | overkill but fastest |
| MPS (laptop, reference) | 60–100 h | the reason this notebook exists |

(Wall-clock estimates already include the ~2× matmul slowdown from
forcing TF32 off; see the next section.)

3. Run cells top-to-bottom. The smoke cell (~3 min) validates the entire
   pipeline before the production cells launch.
4. The aggregation cell at the bottom is **idempotent**: re-running it
   refreshes the reports without re-training.

## Numerics: TF32 forced off

On Ampere+ GPUs (L4 / A100 / H100) the default fp32 matmul path uses
**TF32**, a 19-bit format with only **10 mantissa bits** (vs true
fp32's 23). The SPLM forward computes its force field via
`torch.autograd.grad(…, create_graph=True)`, which is the most
numerically sensitive part of the pipeline; under TF32 we previously
observed PPL inflation in the H100 scale-up pilot (the
TF32-on/TF32-off A/B reported in `paper_v4` §15.21 and the headline
`paper_tmlr_3` discussion).

For the Scope-3 retrain to be a clean precision-controlled comparison
against the v2 numbers (which were trained on MPS without TF32), TF32
is forced off in two redundant ways:

1. **Driver-level.** [`colab_scope3.ipynb`](colab_scope3.ipynb) sets
   `NVIDIA_TF32_OVERRIDE=0` in the notebook environment **before** any
   subprocess is spawned, so cuBLAS / cuDNN refuse the TF32 path even
   if Python forgets to ask.
2. **PyTorch-level.** Every Scope-3 trainer
   ([`train_splm.py`](../train_splm.py),
   [`sarf_variant/train_splm_sarf.py`](../sarf_variant/train_splm_sarf.py),
   [`sarf_mass_variant/train_splm_sarf_mass.py`](../sarf_mass_variant/train_splm_sarf_mass.py),
   [`train_matched.py`](../train_matched.py)) now exposes a
   `--allow-tf32` flag that defaults to **False** and explicitly sets
   `torch.backends.cuda.matmul.allow_tf32 = False` and
   `torch.backends.cudnn.allow_tf32 = False` on entry. Each trainer
   prints a `TF32 disabled (default)` line that the per-seed
   `stdout.log` will record for audit.

T4 GPUs (free Colab) are pre-Ampere and have no TF32 path at all, so
on the free tier this is documented no-op. On L4 / A100 / H100 it
costs roughly a **2× slowdown** in the matmul kernels (already
included in the wall-clock estimates above) in exchange for clean
fp32 second-order numerics that match the v2 baseline regime.

Pass `--allow-tf32` to any individual trainer **only** to produce a
precision-artifact reference run for comparison; never use it in the
production Scope-3 sweep.

## How to run locally (laptop, MPS)

```bash
# 0. (Once) Pre-compute the surprisal table the SPLM-mass cells need.
cd notebooks/conservative_arch/sarf_mass_variant
python3 compute_unigram_frequencies.py
cd -

# 1. Production retrain (S=5 seeds across 5 cells).
python3 notebooks/conservative_arch/multi_seed/multi_seed_runner.py \
    --mode shakespeare --n-seeds 5 \
    --models splm_baseline,splm_sarf,splm_sarfmass_embed_head,splm_sarfmass_logfreq,matched_baseline \
    --tag scope3_shakespeare

# 2. Aggregate per-cell mean / std / Welch (multi-seed report).
python3 notebooks/conservative_arch/multi_seed/multi_seed_aggregator.py \
    --tag scope3_shakespeare

# 3. v2-vs-v4 headline comparison (Scope-3-specific narrative).
python3 notebooks/conservative_arch/multi_seed/scope3_comparison.py \
    --tag scope3_shakespeare
```

The runner is **resumable**: pass `--skip-existing` and any seed whose
`*_ckpt_latest.pt` already lives in the destination directory is
skipped, so you can re-launch after a Colab disconnect without losing
any progress.

## Decision rules for `paper_v4`

The four rules emitted by `scope3_comparison.py` are (verbatim):

1. **Leak-immunity controls** — `splm_baseline` and `matched_baseline`
   should reproduce v2 PPL within seed noise (inflation factor in
   `[0.95, 1.05]`). If they do, the v4 retrain is self-consistent and
   the SARF-cell inflations are attributable to the leak fix.
2. **SARF asymmetric inflation** — Each SARF cell's inflation factor
   replaces the `~2x asymmetric inflation` placeholder in
   `paper_v4` §15. Inflation `> 1` is the predicted direction (the leak
   gave v2 SARF models access to future tokens via $\xi$).
3. **Qualitative direction** — If the v4 ordering matches the v2
   ordering across the 4 SPLM cells, the `v2-historical` caveat blocks
   in §15.17 / §15.19 can be retired. Any rank inversion warrants a
   footnote.
4. **Matched-baseline pairing** — The `matched_baseline` 5-seed PPL is
   the new pairing target for the SPLM-vs-baseline absolute-quality
   claim previously retired in §15.

## Cross-references

* Setup pattern reused from
  [`notebooks/conservative_arch/scaleup/colab_pilot.ipynb`](../scaleup/colab_pilot.ipynb)
  and
  [`colab_n5_confirmation.ipynb`](../scaleup/colab_n5_confirmation.ipynb).
* Multi-seed harness: this directory's
  [`multi_seed_runner.py`](multi_seed_runner.py) +
  [`multi_seed_aggregator.py`](multi_seed_aggregator.py) +
  [`scope3_comparison.py`](scope3_comparison.py).
* Leak-fix forensics:
  [`docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](../../../docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md),
  [`docs/Causal_Leak_Empirical_Comparison_Report.md`](../../../docs/Causal_Leak_Empirical_Comparison_Report.md).
* `paper_v4` open follow-ups (the Scope-3 anchor):
  `paper_v4/sections/15_conservative_architectures.tex`, `subsec:cba-open`.
