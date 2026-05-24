# Training summary — splm_nonconservative_e2_affine_rank1_scaleup_seed0

- experiment: SP-HSPLM Stage 1 (per-token Class B/C rerun on leak-fixed v3 codebase)
- protocol: docs/SP_HSPLM_Stage_1_pre-registered_protocol.md
- model: ScalarPotentialLMNonConservative (em_ln + per-token g_l)
- cell: e2_affine_rank1
- corpus: TinyStories (5,000,000 train tokens)
- params: total 15,753,732  V_theta 2,625,537  nonconservative 256
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True  causal_force=True
- block_size: 512  batch_size: 16  steps: 16000
- seed: 0
- elapsed: 1090 s (0.30 h)

## Results

- Best val PPL: **25.60** @ step 14400
- Final val PPL: 26.37
- Final gamma: 0.9263

## Causal-leak probe history (H3)

| step | label | max_logit_delta_past | verdict |
|---|---|---:|---|
| 1 | init | 0.00e+00 | leak-clean |
| 8000 | mid | 0.00e+00 | leak-clean |
| 16000 | final | 0.00e+00 | leak-clean |

## Nonconservative norms (final eval interval)

(step 16000)

| layer | ||f|| | ||g|| | ||g||/||f|| |
|---:|---:|---:|---:|
| 0 | 2817.238 | 0.000 | 0.000 |
| 1 | 1087.446 | 445.705 | 0.410 |
| 2 | 792.692 | 221.451 | 0.279 |
| 3 | 478.420 | 168.553 | 0.352 |
| 4 | 338.943 | 123.997 | 0.366 |
| 5 | 326.810 | 134.346 | 0.411 |
| 6 | 659.834 | 140.250 | 0.213 |
| 7 | 1775.755 | 127.108 | 0.072 |
