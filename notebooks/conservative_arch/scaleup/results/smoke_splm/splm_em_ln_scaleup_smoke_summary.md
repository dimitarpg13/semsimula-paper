# Training summary — splm_em_ln_scaleup_smoke

- experiment: E9 scale-up (SPLM arm)
- model: ScalarPotentialLMSARFMassLN (em_ln)
- mode: smoke
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 15,753,475
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- block_size: 256  batch_size: 8  steps: 300
- seed: 0
- elapsed: 191 s (0.05 h)

Final val loss: 5.385293 (ppl 218.17)
Final gamma: 0.3000
