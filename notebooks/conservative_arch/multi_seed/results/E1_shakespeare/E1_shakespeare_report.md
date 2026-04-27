# Multi-seed report: `E1_shakespeare`

Aggregator: `notebooks/conservative_arch/multi_seed/multi_seed_aggregator.py`
Run root: `notebooks/conservative_arch/multi_seed/results/E1_shakespeare/`

## Per-model summary (final val loss / val ppl)

Stats are computed over **finite** seeds only; the `diverged` column reports seeds whose final eval was NaN / inf (these are excluded from mean/std/min/max).

| model | n seeds | diverged | val loss mean | val loss std | val ppl mean | val ppl std | val ppl min | val ppl max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `matched_baseline` | 5 | 0 | 5.0084 | 0.0478 | 149.80 | 7.21 | 141.80 | 159.59 |
| `splm_em_ln` | 5 | 0 | 4.5564 | 0.0473 | 95.33 | 4.44 | 88.91 | 98.78 |
| `splm_sarfmass_logfreq` | 3 | 2 | 5.0948 | n/a | 163.18 | n/a | 163.18 | 163.18 |

## Per-seed final eval points

| model | seed | step | train loss eval | val loss | val ppl |
|---|---:|---:|---:|---:|---:|
| `matched_baseline` | 0 | 4000 | 3.5479 | 4.9544 | 141.80 |
| `matched_baseline` | 1 | 4000 | 3.5109 | 5.0421 | 154.79 |
| `matched_baseline` | 2 | 4000 | 3.5056 | 5.0726 | 159.59 |
| `matched_baseline` | 3 | 4000 | 3.5053 | 4.9894 | 146.85 |
| `matched_baseline` | 4 | 4000 | 3.5580 | 4.9835 | 145.99 |
| `splm_em_ln` | 0 | 4000 | 3.6015 | 4.4876 | 88.91 |
| `splm_em_ln` | 1 | 4000 | 3.6043 | 4.5929 | 98.78 |
| `splm_em_ln` | 2 | 4000 | 3.5250 | 4.5263 | 92.41 |
| `splm_em_ln` | 3 | 4000 | 3.6362 | 4.5887 | 98.37 |
| `splm_em_ln` | 4 | 4000 | 3.6418 | 4.5867 | 98.17 |
| `splm_sarfmass_logfreq` | 0 | 4000 | 4.3325 | 5.0948 | 163.18 |
| `splm_sarfmass_logfreq` | 1 | 4000 | n/a | n/a | n/a |
| `splm_sarfmass_logfreq` | 2 | 4000 | n/a | n/a | n/a |

## Pairwise gap (Welch's t-test on final val ppl)

Welch's t-test is applied to the **finite** final-ppl values only; pairs with fewer than 2 finite seeds in either group are reported as `n/a`.

| model A | model B | n_A | n_B | A mean - B mean | 95% CI half-width | t | dof |
|---|---|---:|---:|---:|---:|---:|---:|
| `matched_baseline` | `splm_em_ln` | 5 | 5 | +54.48 | 9.05 | +14.38 | 6.7 |
| `matched_baseline` | `splm_sarfmass_logfreq` | 5 | 1 | n/a | n/a | n/a | n/a |
| `splm_em_ln` | `splm_sarfmass_logfreq` | 5 | 1 | n/a | n/a | n/a | n/a |

## Loss-curve overlays

### `matched_baseline`

![matched_baseline](E1_shakespeare_loss_curves_matched_baseline.png)

### `splm_em_ln`

![splm_em_ln](E1_shakespeare_loss_curves_splm_em_ln.png)

### `splm_sarfmass_logfreq`

![splm_sarfmass_logfreq](E1_shakespeare_loss_curves_splm_sarfmass_logfreq.png)

## Interpretation (manual)

> **TODO (human reviewer):** Inspect the table and overlay plots and answer:
>
> 1. Does the previously-reported single-seed perplexity fall within one std of the multi-seed mean? If not, was that single run an outlier and the headline number needs to be revised.
> 2. Is the SPLM-vs-baseline gap statistically meaningful at this n? Compare the 95% CI half-width of the difference of means against the absolute gap.
> 3. Are any seeds catastrophic (val ppl >> mean+3*std)? If so, investigate before reporting; do not silently discard.
