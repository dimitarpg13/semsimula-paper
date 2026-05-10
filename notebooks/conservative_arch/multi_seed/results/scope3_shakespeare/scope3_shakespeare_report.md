# Multi-seed report: `scope3_shakespeare`

Aggregator: `notebooks/conservative_arch/multi_seed/multi_seed_aggregator.py`
Run root: `notebooks/conservative_arch/multi_seed/results/scope3_shakespeare/`

## Per-model summary (final val loss / val ppl)

Stats are computed over **finite** seeds only; the `diverged` column reports seeds whose final eval was NaN / inf (these are excluded from mean/std/min/max).

| model | n seeds | diverged | val loss mean | val loss std | val ppl mean | val ppl std | val ppl min | val ppl max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `matched_baseline` | 5 | 0 | 5.0085 | 0.0477 | 149.81 | 7.19 | 141.71 | 159.49 |
| `splm_baseline` | 5 | 0 | 5.6655 | 0.0222 | 288.80 | 6.36 | 278.81 | 295.94 |
| `splm_sarf` | 5 | 0 | 5.5473 | 0.0229 | 256.60 | 5.92 | 249.96 | 266.20 |
| `splm_sarfmass_embed_head` | 5 | 0 | 5.6749 | 0.0556 | 291.83 | 16.35 | 271.70 | 316.34 |
| `splm_sarfmass_logfreq` | 5 | 0 | 5.5389 | 0.0682 | 254.87 | 17.13 | 229.45 | 275.65 |

## Per-seed final eval points

| model | seed | step | train loss eval | val loss | val ppl |
|---|---:|---:|---:|---:|---:|
| `matched_baseline` | 0 | 4000 | 3.5479 | 4.9538 | 141.71 |
| `matched_baseline` | 1 | 4000 | 3.5110 | 5.0424 | 154.84 |
| `matched_baseline` | 2 | 4000 | 3.5053 | 5.0720 | 159.49 |
| `matched_baseline` | 3 | 4000 | 3.5053 | 4.9888 | 146.76 |
| `matched_baseline` | 4 | 4000 | 3.5586 | 4.9853 | 146.25 |
| `splm_baseline` | 0 | 4000 | 5.0679 | 5.6745 | 291.35 |
| `splm_baseline` | 1 | 4000 | 5.0690 | 5.6902 | 295.94 |
| `splm_baseline` | 2 | 4000 | 5.0228 | 5.6611 | 287.45 |
| `splm_baseline` | 3 | 4000 | 5.0671 | 5.6305 | 278.81 |
| `splm_baseline` | 4 | 4000 | 5.0230 | 5.6714 | 290.43 |
| `splm_sarf` | 0 | 4000 | 4.9267 | 5.5421 | 255.23 |
| `splm_sarf` | 1 | 4000 | 4.8254 | 5.5420 | 255.18 |
| `splm_sarf` | 2 | 4000 | 4.8154 | 5.5842 | 266.20 |
| `splm_sarf` | 3 | 4000 | 4.7655 | 5.5213 | 249.96 |
| `splm_sarf` | 4 | 4000 | 4.8698 | 5.5469 | 256.44 |
| `splm_sarfmass_embed_head` | 0 | 4000 | 4.7215 | 5.6505 | 284.42 |
| `splm_sarfmass_embed_head` | 1 | 4000 | 4.7290 | 5.6766 | 291.95 |
| `splm_sarfmass_embed_head` | 2 | 4000 | 4.6722 | 5.6861 | 294.74 |
| `splm_sarfmass_embed_head` | 3 | 4000 | 4.6883 | 5.6047 | 271.70 |
| `splm_sarfmass_embed_head` | 4 | 4000 | 4.7609 | 5.7568 | 316.34 |
| `splm_sarfmass_logfreq` | 0 | 4000 | 4.7737 | 5.4357 | 229.45 |
| `splm_sarfmass_logfreq` | 1 | 4000 | 4.8526 | 5.5753 | 263.82 |
| `splm_sarfmass_logfreq` | 2 | 4000 | 4.7542 | 5.5252 | 250.95 |
| `splm_sarfmass_logfreq` | 3 | 4000 | 4.8646 | 5.5392 | 254.48 |
| `splm_sarfmass_logfreq` | 4 | 4000 | 4.8743 | 5.6191 | 275.65 |

## Pairwise gap (Welch's t-test on final val ppl)

Welch's t-test is applied to the **finite** final-ppl values only; pairs with fewer than 2 finite seeds in either group are reported as `n/a`.

| model A | model B | n_A | n_B | A mean - B mean | 95% CI half-width | t | dof |
|---|---|---:|---:|---:|---:|---:|---:|
| `matched_baseline` | `splm_baseline` | 5 | 5 | -138.98 | 9.92 | -32.38 | 7.9 |
| `matched_baseline` | `splm_sarf` | 5 | 5 | -106.79 | 9.66 | -25.65 | 7.7 |
| `matched_baseline` | `splm_sarfmass_embed_head` | 5 | 5 | -142.02 | 19.99 | -17.78 | 5.5 |
| `matched_baseline` | `splm_sarfmass_logfreq` | 5 | 5 | -105.06 | 20.93 | -12.64 | 5.4 |
| `splm_baseline` | `splm_sarf` | 5 | 5 | +32.19 | 8.97 | +8.29 | 8.0 |
| `splm_baseline` | `splm_sarfmass_embed_head` | 5 | 5 | -3.03 | 19.95 | -0.39 | 5.2 |
| `splm_baseline` | `splm_sarfmass_logfreq` | 5 | 5 | +33.93 | 20.91 | +4.15 | 5.1 |
| `splm_sarf` | `splm_sarfmass_embed_head` | 5 | 5 | -35.23 | 19.95 | -4.53 | 5.0 |
| `splm_sarf` | `splm_sarfmass_logfreq` | 5 | 5 | +1.73 | 20.91 | +0.21 | 4.9 |
| `splm_sarfmass_embed_head` | `splm_sarfmass_logfreq` | 5 | 5 | +36.96 | 24.43 | +3.49 | 8.0 |

## Loss-curve overlays

### `matched_baseline`

![matched_baseline](scope3_shakespeare_loss_curves_matched_baseline.png)

### `splm_baseline`

![splm_baseline](scope3_shakespeare_loss_curves_splm_baseline.png)

### `splm_sarf`

![splm_sarf](scope3_shakespeare_loss_curves_splm_sarf.png)

### `splm_sarfmass_embed_head`

![splm_sarfmass_embed_head](scope3_shakespeare_loss_curves_splm_sarfmass_embed_head.png)

### `splm_sarfmass_logfreq`

![splm_sarfmass_logfreq](scope3_shakespeare_loss_curves_splm_sarfmass_logfreq.png)

## Interpretation (manual)

> **TODO (human reviewer):** Inspect the table and overlay plots and answer:
>
> 1. Does the previously-reported single-seed perplexity fall within one std of the multi-seed mean? If not, was that single run an outlier and the headline number needs to be revised.
> 2. Is the SPLM-vs-baseline gap statistically meaningful at this n? Compare the 95% CI half-width of the difference of means against the absolute gap.
> 3. Are any seeds catastrophic (val ppl >> mean+3*std)? If so, investigate before reporting; do not silently discard.
