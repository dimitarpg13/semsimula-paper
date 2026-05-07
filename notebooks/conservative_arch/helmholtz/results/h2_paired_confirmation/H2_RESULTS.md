# H2 - Helmholtz (Q9d) S=3 paired confirmation

Setup: 5 seeds × 2 schedules at vh=128, 4000-step Tiny Shakespeare config (d=128, L=8, mass_mode='logfreq', AdamW 5e-4, batch 16 × block 128, free gamma, causal_force=True, ln_after_s_step=True). Seeds present (Q9d): [0, 1, 2, 3, 4]. Seed 0 sourced from H1.5; seeds ≥ 1 from H2 / H2 power-up. Cells parsed: **10**. Variant A H2 paired baseline at seeds [0, 1, 2, 3, 4] (5 cells).

Schedules under test:

- `AAAASSSS` vh=128 - **quality lead** (best PPL at S=1; +1.88 PPL vs Variant A best at seed 0).
- `AASSSSSS` vh=128 - **joint quality+FLOP lead** (only Q9d cell that beats Variant A outright AND clears the 30% decode-FLOP-reduction bar at T=4096).

## Per-cell results (5 seeds × 2 schedules, vh=128)

| schedule | v_hidden | seed | val PPL | val loss | Δ vs all-attn(seed) | Δ vs VA(seed) | Δ vs VA best |
|----------|----------|------|---------|----------|---------------------|---------------|--------------|
| `AAAASSSS` | 128 | 0 | 134.89 | 4.9045 | -6.91 | +1.88 | +1.88 |
| `AAAASSSS` | 128 | 1 | 152.16 | 5.0249 | -2.63 | -0.09 | +19.15 |
| `AAAASSSS` | 128 | 2 | 151.37 | 5.0197 | -8.22 | -0.73 | +18.36 |
| `AAAASSSS` | 128 | 3 | 139.69 | 4.9394 | -7.16 | -1.98 | +6.68 |
| `AAAASSSS` | 128 | 4 | 151.20 | 5.0186 | +5.21 | -6.77 | +18.19 |
| `AASSSSSS` | 128 | 0 | 139.63 | 4.9390 | -2.17 | +6.62 | +6.62 |
| `AASSSSSS` | 128 | 1 | 157.26 | 5.0579 | +2.47 | +5.01 | +24.25 |
| `AASSSSSS` | 128 | 2 | 151.82 | 5.0227 | -7.77 | -0.28 | +18.81 |
| `AASSSSSS` | 128 | 3 | 140.87 | 4.9478 | -5.98 | -0.80 | +7.86 |
| `AASSSSSS` | 128 | 4 | 155.94 | 5.0495 | +9.95 | -2.03 | +22.93 |

## Paired-t statistics vs all-attention 5-seed E1 baseline

All-attention E1 PPL by seed: seed0=141.80, seed1=154.79, seed2=159.59, seed3=146.85, seed4=145.99 (mean 149.80)

| schedule | n pairs | Q9d mean | Δ̄ vs attn | std Δ | sign (Δ<0) | paired-t | two-sided p | meets +5 PPL bar? |
|----------|---------|----------|------------|-------|------------|----------|-------------|--------------------|
| `AAAASSSS` | 5 | 145.86 | -3.94 | 5.54 | 4/5 | -1.590 | 0.1871 | YES |
| `AASSSSSS` | 5 | 149.10 | -0.70 | 7.13 | 3/5 | -0.219 | 0.8370 | YES |
| `k4_m4` (VA) | 5 | 147.40 | -2.40 | 8.39 | 4/5 | -0.641 | 0.5564 | YES |

## Paired-t statistics vs Variant A 5-seed H2 baseline (k=4, m=4)

Variant A k=4, m=4 PPL by seed: seed0=133.01, seed1=152.25, seed2=152.10, seed3=141.67, seed4=157.97 (mean 147.40)

| schedule | n pairs | Q9d mean | Δ̄ vs VA | std Δ | sign (Δ<0) | paired-t | two-sided p | Q9d-vs-VA win? |
|----------|---------|----------|---------|-------|------------|----------|-------------|---------------|
| `AAAASSSS` | 5 | 145.86 | -1.54 | 3.24 | 4/5 | -1.061 | 0.3484 | weak win |
| `AASSSSSS` | 5 | 149.10 | +1.70 | 3.85 | 3/5 | +0.990 | 0.3782 | ON PAR |

## H2 decision verdict

Pre-registered title rule (per `companion_notes/Helmholtz-HSPLM_Path_Forward_and_Experiments.md` §8.3):

- **PASS quality arm** iff at the best Q9d schedule: mean Δ̄ ≥ -5 PPL vs all-attention AND sign-consistency n/n AND paired-t two-sided p < 0.05.
- **Q9d-vs-Variant-A win** iff at the best Q9d schedule: mean Δ̄ ≤ -5 PPL vs Variant A best AND sign-consistency n/n.
- Current sample sizes: Q9d n = 5, all-attn n = 5, VA n = 5.

### Quality arm (vs all-attention 5-seed E1)

- Q9d `AAAASSSS`: Δ̄ = -3.94 PPL, sign 4/5, p = 0.1871 → **MARGINAL**
- Q9d `AASSSSSS`: Δ̄ = -0.70 PPL, sign 3/5, p = 0.8370 → **MARGINAL**
- Variant A (k=4, m=4): Δ̄ = -2.40 PPL, sign 4/5, p = 0.5564 → **MARGINAL**

**Effect-size summary**: Q9d's best schedule (`AAAASSSS`) and Variant A are both directionally better than all-attention (Δ̄ = -3.9 PPL for Q9d, sign 4/5; Δ̄ = -2.4 PPL for VA, sign 4/5), but the per-seed dispersion is high enough that **at least one seed reverses the sign on at least one hybrid arm**. The earlier n=3 "3/3" sign-consistency claim was a small-sample artifact: with seeds 3 and 4 added, the hybrid advantage shrinks and is not robust across the full seed panel. Treat the apparent PPL gap as a noisy small-effect signal, not a reliable architectural win at this scale.

### Q9d-vs-Variant-A arm (paired-t vs VA k=4, m=4 5-seed mean 147.40)

- `AAAASSSS`: Δ̄ = -1.54 PPL, sign 4/5 (Q9d better than VA), p = 0.3484 → **ON PAR**
- `AASSSSSS`: Δ̄ = +1.70 PPL, sign 3/5 (Q9d better than VA), p = 0.3782 → **ON PAR**

