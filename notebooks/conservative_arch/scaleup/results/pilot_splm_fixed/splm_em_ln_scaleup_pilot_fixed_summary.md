# Training summary — splm_em_ln_scaleup_pilot_fixed

- experiment: E9 scale-up (SPLM arm)
- model: ScalarPotentialLMSARFMassLN (em_ln)
- mode: pilot
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 15,753,475
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- block_size: 512  batch_size: 16  steps: 4000
- seed: 0
- elapsed: 25828 s (7.17 h)

Final val loss: 3.513138 (ppl 33.55)
Final gamma: 0.3000
