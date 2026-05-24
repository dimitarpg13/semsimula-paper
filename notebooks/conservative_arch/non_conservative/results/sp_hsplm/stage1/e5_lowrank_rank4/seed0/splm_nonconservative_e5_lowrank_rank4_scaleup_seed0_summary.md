# Training summary — splm_nonconservative_e5_lowrank_rank4_scaleup_seed0

- experiment: SP-HSPLM Stage 1 (per-token Class B/C rerun on leak-fixed v3 codebase)
- protocol: docs/SP_HSPLM_Stage_1_pre-registered_protocol.md
- model: ScalarPotentialLMNonConservative (em_ln + per-token g_l)
- cell: e5_lowrank_rank4
- corpus: TinyStories (5,000,000 train tokens)
- params: total 16,016,644  V_theta 2,625,537  nonconservative 263,168
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True  causal_force=True
- block_size: 512  batch_size: 16  steps: 16000
- seed: 0
- elapsed: 1162 s (0.32 h)

## Results

- Best val PPL: **24.77** @ step 14400
- Final val PPL: 25.39
- Final gamma: 0.9224

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
| 0 | 2684.480 | 0.000 | 0.000 |
| 1 | 1035.477 | 826.470 | 0.798 |
| 2 | 808.271 | 864.876 | 1.070 |
| 3 | 777.323 | 711.184 | 0.915 |
| 4 | 864.426 | 580.913 | 0.672 |
| 5 | 1002.001 | 548.893 | 0.548 |
| 6 | 1164.792 | 622.444 | 0.534 |
| 7 | 1538.544 | 712.878 | 0.463 |
