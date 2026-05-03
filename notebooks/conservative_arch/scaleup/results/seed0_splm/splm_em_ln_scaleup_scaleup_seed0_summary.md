# Training summary — splm_em_ln_scaleup_scaleup_seed0

- experiment: E9 scale-up (SPLM arm)
- model: ScalarPotentialLMSARFMassLN (em_ln)
- mode: scaleup
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 15,753,475
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- block_size: 512  batch_size: 16  steps: 8000
- seed: 0
- elapsed: 47102 s (13.08 h)

Final val loss: 2.179907 (ppl 8.85)
Final gamma: 0.3000
