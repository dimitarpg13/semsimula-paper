# Training summary — splm_em_ln_scaleup_smoke_fixed

- experiment: E9 scale-up (SPLM arm)
- model: ScalarPotentialLMSARFMassLN (em_ln)
- mode: smoke
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 15,753,475
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- block_size: 256  batch_size: 8  steps: 300
- seed: 0
- elapsed: 245 s (0.07 h)

Final val loss: 5.018151 (ppl 151.13)
Final gamma: 0.3000
