# Training summary — sphsplm_q9e_a_scaleup_seed0

- experiment: SP-HSPLM Stage 2 (Q9(e) pair-skew cell ladder)
- protocol: docs/SP_HSPLM_Stage_2_pre-registered_protocol.md
- model: ScalarPotentialLMSPHSPLM (SparsePARFLM S-block + low-rank pair-skew C-block)
- cell: q9e_a  schedule: SCSCSCSC
- mode: scaleup
- corpus: TinyStories (cap 5,000,000 train tokens)
- params: total 15,789,927  skew 8,192  gyro 0
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True  k=4  r=16  gyro=False
- block_size: 512  batch_size: 16  steps: 16000
- seed: 0
- elapsed: 4674 s (1.30 h)

Final val loss: 3.317082 (ppl 27.58)
Final gamma: 0.8948

## Causal-leak probe history

| step | label | max_logit_delta_past | verdict |
|---|---|---:|---|
| 1 | init | 0.00e+00 | leak-clean |
| 8000 | mid | 0.00e+00 | leak-clean |
| 16000 | final | 0.00e+00 | leak-clean |

## Pair-kernel norms (final)

| quantity | value |
|---|---:|
| J_phi_fro | 2.0435 |
| U_fro | 2.2542 |
| V_fro | 2.2287 |
