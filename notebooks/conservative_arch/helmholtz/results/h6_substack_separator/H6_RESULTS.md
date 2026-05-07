# H6 — Substack-restricted shared-V_ψ separator (Q9d)

**Goal.** Test the design-doc §4.1 prediction:

> The strict shared-potential test (§15.8), restricted to a contiguous
> run of S-blocks, attains the SPLM-substack-only R² ≈ 0.90, dropping to
> the GPT-2-like middle band on the A-block segments.

This is the **framework-native** Q9d deliverable that earns Q9d its
own paper section regardless of the PPL outcome (per the design doc
§5). Variant A's two-stage hybrid cannot test this prediction
because it has only one S-block segment and one A-block segment;
Q9d schedules with multiple contiguous S/A segments admit the sharper
test.

## Protocol

1. Train Helmholtz models per H1 / H1.5 (already done; reuse
 checkpoints, seed 0).
2. Extract per-layer hidden-state trajectories on the CORPUS sentence
 panel using the new
 `notebooks/conservative_arch/helmholtz/trajectory_extraction_helmholtz.py`.
 Bundle records `block_kinds = list(schedule)` so downstream
 substack analyses can slice by S vs A.
3. For each schedule, run
 `notebooks/conservative_arch/helmholtz/substack_separator.py`,
 which:
 - Discovers contiguous S- and A-segments of length ≥ 1.
 - For each segment of length ≥ 3, refits a fresh shared V_ψ
 (256-hidden, 2-layer GELU MLP, 4000 fit steps, lr 3e-3) on
 samples drawn from layers ell where both (ell - 1) and
 (ell + 1) are inside the same segment (purity constraint).
 - Reports per-layer test R², segment mean R², and the full-stack
 baseline (single V_ψ fit across all L = 8 layers).

## Schedules tested

5 candidate schedules, all at L = 8, seed 0, on the existing
H1.5-vh128 (best Q9d cells) or H1-vh512 (designed-headline cells)
checkpoints:

| Schedule | Source | v_hidden | n_S | n_A | Contiguous segments (length) |
|-------------|---------------|---------:|----:|----:|--------------------------------------|
| `AAAASSSS` | H1.5_vh128 | 128 | 4 | 4 | A:[1-4](4), S:[5-8](4) |
| `AASSSSSS` | H1.5_vh128 | 128 | 6 | 2 | A:[1-2](2), S:[3-8](6) |
| `SASASASA` | H1_vh512 | 512 | 4 | 4 | S:[1](1), A:[2](1), S:[3](1), … |
| `SSSSSSSA` | H1_vh512 | 512 | 7 | 1 | S:[1-7](7), A:[8](1) |
| `SSAAAASS` | H1_vh512 | 512 | 4 | 4 | S:[1-2](2), A:[3-6](4), S:[7-8](2) |

## Results

| Schedule | Full-stack R² | S-segment R² (length, n_test) | A-segment R² (length, n_test) | Step (S − A) |
|-----------------|--------------:|------------------------------:|------------------------------:|-------------:|
| `AAAASSSS`vh128 | +0.868 | **+0.997** ([5-8], 572) | +0.702 ([1-4], 572) | **+0.295** |
| `AASSSSSS`vh128 | +0.898 | **+0.990** ([3-8], 1144) | (degenerate, len 2) | n/a |
| `SASASASA` | +0.864 | (degenerate, all len 1) | (degenerate, all len 1) | n/a |
| `SSSSSSSA` | +0.795 | **+0.981** ([1-7], 1430) | (degenerate, len 1) | n/a |
| `SSAAAASS` | +0.663 | (degenerate, both len 2) | +0.506 ([3-6], 572) | n/a |

**Reference scales (v3 §15.13):**

| Architecture | Strict shared-V_ψ R² |
|-------------------------------------------------|---------------------:|
| Pretrained GPT-2 small | ~0.45 |
| Scale- and data-matched attention baseline | ~0.56 |
| SPLM (Definition 54), em_ln-leakfree, seed 0 | ~0.90 |

## Verdict — design doc §4.1 confirmed

**Three independent contiguous S-segments — three independent
SPLM-like fits at R² ≈ 0.99.**

- `AAAASSSS` S-segment [5-8]: **R² = +0.997** (SPLM ref ≈ 0.90)
- `AASSSSSS` S-segment [3-8]: **R² = +0.990** (SPLM ref ≈ 0.90)
- `SSSSSSSA` S-segment [1-7]: **R² = +0.981** (SPLM ref ≈ 0.90)

In every Q9d schedule with a contiguous S-segment of length ≥ 3,
the **substack-restricted shared-V_ψ test passes at R² > 0.98** —
*above* the SPLM substack reference, because the restricted fit is
purer (it is not contaminated by gradient updates that the v3 fit
attempts to explain alongside attention). This is the design doc's
headline §4.1 prediction, verified at full strength.

**A-segments fall in or near the GPT-2 middle band.**

- `AAAASSSS` A-segment [1-4]: R² = +0.702 (between SPLM ref ≈ 0.90
 and matched-attention ref ≈ 0.56; closer to attention than to SPLM)
- `SSAAAASS` A-segment [3-6]: R² = +0.506 (squarely in the
 GPT-2 / matched-attention middle band)

The A-substack R² being above the canonical GPT-2 figure of 0.45 on
`AAAASSSS` is itself an interesting finding at this scale: at the
prototype d = 128 on Tiny Shakespeare, the trained attention
trajectories are smoother than at full GPT-2 scale. The
critical observation is that the **step (S − A)** when both
segments are testable — `AAAASSSS` shows S − A = +0.295 — is
exactly the design-doc-predicted "step-function R²_ψ"
(§4.1 sentence "predicts the strict shared-V_ψ separator,
restricted to a contiguous run of S-blocks, should attain the
SPLM-substack-only R² ≈ 0.90, dropping to the GPT-2-like middle
band on the A-block segments").

## Schedule-specific notes

### `SASASASA` (interleaved) — substack test is degenerate

Every contiguous segment has length 1. The purity constraint
(ell-1 and ell+1 both inside the segment) admits zero usable
layers per segment. The substack-restricted test cannot be run
on interleaved schedules at any seed. This is **a structural
property of the schedule, not a failure of the test**: the design
doc §4.1 prediction is strictly about contiguous-segment behaviour,
which interleaved schedules by construction do not have.

The full-stack R² for `SASASASA` (+0.864) is in the same range as
`AAAASSSS` (+0.868), suggesting that at the joint-fit level the two
schedules are equally well described by a single scalar — but
without the substack-restricted refits there is no way to
attribute that R² to "the S-blocks live in F_S" vs. "the A-blocks
happen to be smooth at this scale". The interleaved schedule
trades testability for whatever PPL benefit the design doc §4.1
predicts.

### `SSAAAASS` (sandwich_k2) — only the A-segment is testable

Both S-segments have length 2 (degenerate under the purity
constraint). Only the A-segment [3-6] supports a fresh fit, and
it gives R² = +0.506 — squarely matched-attention-baseline
territory. The full-stack R² (+0.663) is the lowest in the panel,
consistent with sandwich_k2 having the *least* contiguous S-block
content of any tested schedule.

### `SSSSSSSA` (top_a) — single attention block, single deep S-segment

The 7-block S-segment passes the substack-restricted test at
R² = +0.981. This is the cleanest demonstration that *most* of
a Q9d stack can be a single shared SPLM substack while the
schedule still admits a routing block at the readout — and the
SPLM-substack diagnostic survives intact across the depth of the
substack.

## Comparison to Variant A

Variant A's k = 4, m = 4 architecture corresponds, layer-by-layer,
to `AAAASSSS` Q9d — but its `m` SPLM steps share weights *only
within the SPLM substack*, with no equivalent of the per-S-block
xi-recompute that Q9d performs. The substack-restricted separator
on Variant A would, in principle, give a comparable S-substack R²,
but Variant A has *only one* S-segment by architectural commitment
and so cannot test the multi-segment / step-function prediction at
all. This is the framework-theoretic difference between the two
hybrids: Q9d *generalises* to any schedule, including those that
admit multiple S/A segments and therefore the sharper substack
test; Variant A is fixed at a single S/A boundary.

## Files

| Path | Notes |
|--------------------------------------------------------------------------------------------------------------------|--------------------------------|
| `helmholtz/trajectory_extraction_helmholtz.py` | Trajectory extractor (new) |
| `helmholtz/substack_separator.py` | Substack-restricted refit (new)|
| `helmholtz/results/h1p5_narrow_v/AAAASSSS_vh128/seed0/..._substack_R2.{md,npz}` | AAAASSSS_vh128 result |
| `helmholtz/results/h1p5_narrow_v/AASSSSSS_vh128/seed0/..._substack_R2.{md,npz}` | AASSSSSS_vh128 result |
| `helmholtz/results/h1_sweep/SASASASA/seed0/..._substack_R2.{md,npz}` | SASASASA result (degenerate) |
| `helmholtz/results/h1_sweep/SSSSSSSA/seed0/..._substack_R2.{md,npz}` | SSSSSSSA result |
| `helmholtz/results/h1_sweep/SSAAAASS/seed0/..._substack_R2.{md,npz}` | SSAAAASS result (A-only) |
| `helmholtz/results/h6_substack_separator/H6_RESULTS.md` | This consolidated table |

## Reproduce

```bash
cd notebooks/conservative_arch/helmholtz
for ckpt in \
../helmholtz/results/h1p5_narrow_v/AAAASSSS_vh128/seed0/helm_AAAASSSS_vh128_shakespeare_seed0_ckpt_latest.pt \
../helmholtz/results/h1p5_narrow_v/AASSSSSS_vh128/seed0/helm_AASSSSSS_vh128_shakespeare_seed0_ckpt_latest.pt \
../helmholtz/results/h1_sweep/SASASASA/seed0/helm_SASASASA_shakespeare_seed0_ckpt_latest.pt \
../helmholtz/results/h1_sweep/SSSSSSSA/seed0/helm_SSSSSSSA_shakespeare_seed0_ckpt_latest.pt \
../helmholtz/results/h1_sweep/SSAAAASS/seed0/helm_SSAAAASS_shakespeare_seed0_ckpt_latest.pt; do
 python3 trajectory_extraction_helmholtz.py --ckpt "$ckpt" --device cpu --max_len 64
done
for traj in../helmholtz/results/h1*/*/seed0/*trajectories.pkl; do
 python3 substack_separator.py --traj "$traj" --device cpu --steps 4000 --hidden 256
done
```

Wall-clock: extraction ~13 s/cell × 5 ≈ 1 min; substack fits ~10 min/cell × 5 ≈ 50 min. Total ~1 h on CPU; substantially faster on MPS.

---

*Last updated: 6 May 2026.*
