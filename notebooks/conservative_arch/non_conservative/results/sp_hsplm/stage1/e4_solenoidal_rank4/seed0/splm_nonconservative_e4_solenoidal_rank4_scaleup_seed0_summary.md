# Training summary — splm_nonconservative_e4_solenoidal_rank4_scaleup_seed0

- experiment: SP-HSPLM Stage 1 (per-token Class B/C rerun on leak-fixed v3 codebase)
- protocol: docs/SP_HSPLM_Stage_1_pre-registered_protocol.md
- model: ScalarPotentialLMNonConservative (em_ln + per-token g_l)
- cell: e4_solenoidal_rank4
- corpus: TinyStories (5,000,000 train tokens)
- params: total 16,049,732  V_theta 2,625,537  nonconservative 296,256
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True  causal_force=True
- block_size: 512  batch_size: 16  steps: 16000
- seed: 0
- elapsed: 1183 s (0.33 h)

## Results

- Best val PPL: **24.58** @ step 14400
- Final val PPL: 25.29
- Final gamma: 0.9868

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
| 0 | 2410.914 | 975.717 | 0.405 |
| 1 | 432.664 | 377.232 | 0.872 |
| 2 | 356.747 | 346.411 | 0.971 |
| 3 | 315.663 | 375.611 | 1.190 |
| 4 | 319.740 | 435.604 | 1.362 |
| 5 | 383.977 | 517.153 | 1.347 |
| 6 | 630.394 | 636.448 | 1.010 |
| 7 | 1114.260 | 741.125 | 0.665 |
