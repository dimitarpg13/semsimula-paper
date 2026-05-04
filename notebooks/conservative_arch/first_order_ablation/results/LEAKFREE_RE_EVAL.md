# SPLM-1 vs SPLM-2 — leak-free re-evaluation of the TinyShakespeare 6-cell sweep

> **Forensic re-evaluation only.** The trained weights are those produced by training under the *buggy* SPLM `integrate()` loop. We evaluate each ckpt under the leak-free integrator (`cfg.causal_force = True`) on the same random val batches as the buggy integrator. The resulting (`PPL_buggy`, `PPL_fixed`) pair quantifies how much of the trained-time PPL claim was an autograd-leak artefact for *this* checkpoint. A definitive replication requires leak-free retraining of the 6-cell sweep.

## 1. Eval configuration

- corpus: `shakespeare`
- n_batches: 40
- batch: 16
- block: 128
- device: `cpu`
- val-batch RNG seed: 0 (same seed used for the buggy and fixed evaluators, so they see the identical val tokens for a paired comparison).

## 2. Per-ckpt paired (buggy, fixed) PPL

Inflation factor = `PPL_fixed / PPL_buggy`. Values > 1 mean the buggy training-time integrator overstated this ckpt's val performance.

| arm | seed | ckpt | PPL (buggy) | PPL (fixed) | inflation × |
|---|---:|---|---:|---:|---:|
| splm1 | 0 | `splm_first_order_shakespeare_seed0_ckpt_latest.pt` | 110.16 | 602.34 | 5.47× |
| splm1 | 1 | `splm_first_order_shakespeare_seed1_ckpt_latest.pt` | 116.91 | 365.66 | 3.13× |
| splm1 | 2 | `splm_first_order_shakespeare_seed2_ckpt_latest.pt` | 116.51 | 357.27 | 3.07× |
| splm2 | 0 | `splm_em_ln_shakespeare_seed0_ckpt_latest.pt` | 90.04 | 1559.19 | 17.32× |
| splm2 | 1 | `splm_em_ln_shakespeare_seed1_ckpt_latest.pt` | 94.64 | 1233.17 | 13.03× |
| splm2 | 2 | `splm_em_ln_shakespeare_seed2_ckpt_latest.pt` | 88.35 | 1249.13 | 14.14× |

## 3. Paired SPLM-1 − SPLM-2 deltas

| seed | Δ_PPL (buggy) | Δ_PPL (fixed) |
|---:|---:|---:|
| 0 | +20.12 | -956.85 |
| 1 | +22.27 | -867.51 |
| 2 | +28.16 | -891.86 |
| **mean** | **+23.52** | **-905.41** |
| **std**  | 4.16 | 46.18 |

- **buggy integrator** (matches published RESULTS.md headline of +23.18 PPL): paired-t = 9.79, df = 2, Cohen's d_z = 5.65.
- **fixed integrator** (leak-free): paired-t = -33.96, df = 2, Cohen's d_z = -19.60.

## 4. Interpretation guide

Three coarse outcomes are possible:

- **(a) Both arms re-evaluate close to their published numbers** (inflation ≈ 1× for both). The TinyShakespeare comparison is robust to the leak-fix; the published 23.18-PPL gap is a clean second-order-vs-first-order architectural lift. Paper v3 §15 may stand as written, with a footnote citing this re-evaluation.
- **(b) Both arms inflate roughly equally**, the paired delta under the fixed integrator is similar in sign and magnitude. Published gap is *qualitatively* preserved but the absolute PPLs are not safe to claim. Paper v3 §15 needs the absolute numbers replaced (after a leak-free retrain) but the qualitative conclusion stands.
- **(c) Asymmetric inflation** — one arm exploits the leak more than the other, so Δ(fixed) ≠ Δ(buggy). If SPLM-2's lead shrinks or inverts under the fix, the published paper-v3 §15 conclusion is at risk. Definitive resolution requires a 6-cell leak-free retrain.

These outcome categories are pre-registered to remove post-hoc reading of the data. Inspect the table in §3 and assign one of (a), (b), (c) before drawing further implications.
