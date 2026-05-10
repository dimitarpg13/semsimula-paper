# Scope-3 retrain comparison (`scope3_shakespeare`)

Re-runs the v2 SPLM-family experiments under the v4 leak-free
integrator (`cfg.causal_force = True`, the post-fix default).
All cells share the v2 hyperparameters (Tiny Shakespeare,
`d=128, L=8, max_len=512, block_size=128, batch_size=16,`
`lr=5e-4, 4000 steps`); the only systematic change vs v2 is the
integrator's causal-honesty flag.

Run root: `notebooks/conservative_arch/multi_seed/results/scope3_shakespeare/`

## Headline table -- v2 buggy vs v4 leak-free

Inflation factor = v4 mean PPL / v2 reference PPL. A factor
near `1.00` for `splm_baseline` and `matched_baseline` is the
leak-immunity sanity check (those two cells are leak-free by
construction); for the SARF cells the inflation quantifies how
much the v2 numbers were under-reported by the bug.

| cell | v2 ref PPL | v4 mean ± std PPL | n_finite / n_seeds | diverged | inflation x | leak-immune? |
|---|---:|---:|---:|---:|---:|:--:|
| `splm_baseline` | 287.43 | 288.80 ± 6.36 | 5 / 5 | 0 | 1.00x | yes |
| `splm_sarf` | 192.21 | 256.60 ± 5.92 | 5 / 5 | 0 | 1.34x | no |
| `splm_sarfmass_embed_head` | 222.91 | 291.83 ± 16.35 | 5 / 5 | 0 | 1.31x | no |
| `splm_sarfmass_logfreq` | 160.55 | 254.87 ± 17.13 | 5 / 5 | 0 | 1.59x | no |
| `matched_baseline` | 149.80 ± 7.21 (5s) | 149.81 ± 7.19 | 5 / 5 | 0 | 1.00x | yes |

## Per-seed PPL ladder

Empty cells = seed not run yet.

| cell | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std |
|---|---:|---:|---:|---:|---:|---:|---:|
| `splm_baseline` | 291.35 | 295.94 | 287.45 | 278.81 | 290.43 | 288.80 | 6.36 |
| `splm_sarf` | 255.23 | 255.18 | 266.20 | 249.96 | 256.44 | 256.60 | 5.92 |
| `splm_sarfmass_embed_head` | 284.42 | 291.95 | 294.74 | 271.70 | 316.34 | 291.83 | 16.35 |
| `splm_sarfmass_logfreq` | 229.45 | 263.82 | 250.95 | 254.48 | 275.65 | 254.87 | 17.13 |
| `matched_baseline` | 141.71 | 154.84 | 159.49 | 146.76 | 146.25 | 149.81 | 7.19 |

## Qualitative-direction check

The v2 `compare.py` reading (single-seed) ordered the four
SPLM cells from best to worst as:

> matched (1) -> sarfmass_logfreq (2) -> sarf (3) -> sarfmass_embed_head (4) -> splm_baseline (5)

This corresponds to the framework-internal prediction that
(a) SARF-faithful $\xi$ recomputation beats fixed-$\xi$,
(b) the surprisal-prior mass beats the unconstrained-MLP mass,
and (c) per-token mass with a useful prior beats global mass.
If the v4 leak-free retrain preserves this order, the
framework's qualitative claims survive causal honesty.

| cell | v2 rank | v4 rank | Δ rank | preserved? |
|---|---:|---:|---:|:--:|
| `splm_baseline` | 5 | 4 | -1 | shifted (-1) |
| `splm_sarf` | 3 | 3 | +0 | yes |
| `splm_sarfmass_embed_head` | 4 | 5 | +1 | shifted (+1) |
| `splm_sarfmass_logfreq` | 2 | 2 | +0 | yes |
| `matched_baseline` | 1 | 1 | +0 | yes |

## Decision rules for `paper_v4`

1. **Leak-immunity controls**. If `splm_baseline` and
   `matched_baseline` reproduce v2 PPL within seed noise
   (inflation factor in `[0.95, 1.05]`), the v4 retrain is
   self-consistent and the SARF-cell inflations are
   attributable to the leak fix, not to confound.
2. **SARF asymmetric inflation**. Report each SARF cell's
   inflation factor in `paper_v4` §15 (replacing the
   `~2x asymmetric inflation` placeholder). Inflation is
   expected to be `>1` because the leak gave the v2 SARF
   models access to future tokens through $\xi$.
3. **Qualitative direction**. If the v4 ordering matches the
   v2 ordering across the 4 SPLM cells, all `v2-historical`
   caveat blocks in `paper_v4` §15.17 / §15.19 can be
   retired. Any rank inversion warrants a footnote in the
   relevant subsection.
4. **Matched-baseline pairing**. The `matched_baseline` 5-seed
   PPL is the new pairing target for the SPLM-vs-baseline
   absolute-quality claim previously retired in §15.
