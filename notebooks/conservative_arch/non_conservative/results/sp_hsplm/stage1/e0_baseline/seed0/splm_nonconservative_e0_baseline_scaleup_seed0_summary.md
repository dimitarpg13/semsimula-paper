# Training summary — splm_nonconservative_e0_baseline_scaleup_seed0

- experiment: SP-HSPLM Stage 1 (per-token Class B/C rerun on leak-fixed v3 codebase)
- protocol: docs/SP_HSPLM_Stage_1_pre-registered_protocol.md
- model: ScalarPotentialLMNonConservative (em_ln + per-token g_l)
- cell: e0_baseline
- corpus: TinyStories (5,000,000 train tokens)
- params: total 15,753,476  V_theta 2,625,537  nonconservative 0
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True  causal_force=True
- block_size: 512  batch_size: 16  steps: 16000
- seed: 0
- elapsed: 1068 s (0.30 h)

## Results

- Best val PPL: **26.31** @ step 14400
- Final val PPL: 27.07
- Final gamma: 0.9024

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
| 0 | 3209.341 | 0.000 | 0.000 |
| 1 | 1095.518 | 0.000 | 0.000 |
| 2 | 906.477 | 0.000 | 0.000 |
| 3 | 694.645 | 0.000 | 0.000 |
| 4 | 466.442 | 0.000 | 0.000 |
| 5 | 446.732 | 0.000 | 0.000 |
| 6 | 655.815 | 0.000 | 0.000 |
| 7 | 1250.355 | 0.000 | 0.000 |
