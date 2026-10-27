# Variant B stand-in: SPLM-first stage order (2026-10-27)

Question: does the hybrid need the attention stage first? Strict Variant B
(`S^m A^k`, ξ frozen from the embeddings) was defined in the book (§18.2) but
never trained. With the SPLM stage at the bottom, Q9d also starts from v = 0,
so the Q9d schedules `SSSSAAAA` / `SSSSSSAA` differ from Variant B only in
recomputing ξ at every S-block. They are used as the stand-in.

Setup: H1.5 config unchanged — Tiny Shakespeare, d = 128, L = 8, d_V = 128,
4,000 steps, seed 0, `mass_mode='logfreq'`, free γ, `causal_force=True`,
`ln_after_s_step=True`; local Mac (MPS), torch 2.2.2. Run serially by
`train_helmholtz.py --mode shakespeare --schedule <σ> --v-hidden 128 --seed 0`.

## Control (reproduction)

`AAAASSSS` vh128 seed 0 rerun: val loss 4.9010 (PPL 134.43) against the
recorded 4.9045 (134.89, `h1p5_narrow_v/`). Largest difference over the 20
evaluations 0.012 nats. Same parameter count (7,324,804), same token counts
(train 321,124 / val 16,901). The setup reproduces.

## Results

| schedule | mirror of | params | val loss | val PPL | best val (step) | final train loss | γ |
|---|---|---:|---:|---:|---:|---:|---:|
| `AAAASSSS` (control) | — | 7,324,804 | 4.9010 | 134.43 | 4.9010 (4000) | 3.769 | 0.164 |
| `SSSSAAAA` | `AAAASSSS` 134.89 | 7,324,804 | 5.1599 | **174.15** | 5.1519 (1600) | 3.539 | 0.149 |
| `SSSSSSAA` | `AASSSSSS` 139.63 | 6,928,260 | 5.1740 | **176.61** | 5.1556 (1600) | 3.658 | 0.145 |

- Paired at seed 0: +39.3 and +37.0 PPL. Seed 0 is the best of five seeds
  for both attention-first cells (H2: 134.89–152.16 and 139.63–157.26), so
  against the worst seed the mirrors are still 22 and 19 PPL behind.
- The mirrors land at the all-SPLM level (173.59) and near `SSSSSSSA` (179.06,
  d_V = 512). Attention after the SPLM stage brings no gain at this seed.
- Same pattern as `SASASASA` and the S-heavy schedules in H1: lower training
  loss, higher validation loss; validation stops improving after ~1,600 steps.
- Cost per token is identical to the counterparts (FLOPs depend only on
  (n_A, n_S)): 18.21 / 17.13 MFLOPs at T = 1024.

Reading: the attention stage must come first; the SPLM stage works as a
refinement of an attention-mixed representation, not as a front end.
One seed only.

Book: `paper_v6/sections/16_hybrid_splm.tex` — §18.2 stand-in sentence,
tab:hybrid-H1p5 middle block, paragraph `par:hybrid-stage-order`, synthesis
item 2.
