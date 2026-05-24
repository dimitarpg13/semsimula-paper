# Training summary — splm_nonconservative_e3_lowrank_rank2_scaleup_seed0

- experiment: SP-HSPLM Stage 1 (per-token Class B/C rerun on leak-fixed v3 codebase)
- protocol: docs/SP_HSPLM_Stage_1_pre-registered_protocol.md
- model: ScalarPotentialLMNonConservative (em_ln + per-token g_l)
- cell: e3_lowrank_rank2
- corpus: TinyStories (5,000,000 train tokens)
- params: total 15,885,060  V_theta 2,625,537  nonconservative 131,584
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True  causal_force=True
- block_size: 512  batch_size: 16  steps: 16000
- seed: 0
- elapsed: 1145 s (0.32 h)

## Results

- Best val PPL: **25.33** @ step 14400
- Final val PPL: 25.99
- Final gamma: 0.8991

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
| 0 | 2926.401 | 0.000 | 0.000 |
| 1 | 895.078 | 937.960 | 1.048 |
| 2 | 683.161 | 897.803 | 1.314 |
| 3 | 617.636 | 654.303 | 1.059 |
| 4 | 700.015 | 451.139 | 0.644 |
| 5 | 881.463 | 376.708 | 0.427 |
| 6 | 1096.022 | 459.767 | 0.419 |
| 7 | 1729.129 | 565.424 | 0.327 |
