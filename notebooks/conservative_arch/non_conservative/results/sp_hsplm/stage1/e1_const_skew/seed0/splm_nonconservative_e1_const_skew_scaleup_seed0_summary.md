# Training summary — splm_nonconservative_e1_const_skew_scaleup_seed0

- experiment: SP-HSPLM Stage 1 (per-token Class B/C rerun on leak-fixed v3 codebase)
- protocol: docs/SP_HSPLM_Stage_1_pre-registered_protocol.md
- model: ScalarPotentialLMNonConservative (em_ln + per-token g_l)
- cell: e1_const_skew
- corpus: TinyStories (5,000,000 train tokens)
- params: total 15,819,012  V_theta 2,625,537  nonconservative 65,536
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True  causal_force=True
- block_size: 512  batch_size: 16  steps: 16000
- seed: 0
- elapsed: 1106 s (0.31 h)

## Results

- Best val PPL: **25.76** @ step 14400
- Final val PPL: 26.43
- Final gamma: 0.9594

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
| 0 | 3178.214 | 0.000 | 0.000 |
| 1 | 681.824 | 1199.718 | 1.760 |
| 2 | 641.620 | 957.517 | 1.492 |
| 3 | 915.735 | 838.112 | 0.915 |
| 4 | 1178.687 | 771.294 | 0.654 |
| 5 | 707.655 | 924.986 | 1.307 |
| 6 | 659.049 | 880.349 | 1.336 |
| 7 | 1720.881 | 833.882 | 0.485 |
