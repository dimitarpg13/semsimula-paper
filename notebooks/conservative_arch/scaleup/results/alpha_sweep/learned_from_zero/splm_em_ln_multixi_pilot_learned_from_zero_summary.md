# Training summary — splm_em_ln_multixi_pilot_learned_from_zero

- experiment: E11 multi-channel-ξ SPLM scale-up
- model: ScalarPotentialLMSARFMassLNMultiXi (em_ln + multi-ξ)
- mode: pilot
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,539,911
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- xi_channels: 4  learnable=True  α_init=[0.0, 0.0, 0.0, 0.0]
- block_size: 512  batch_size: 16  steps: 4000
- seed: 0
- elapsed: 1029 s (0.29 h)

Final val loss: 3.718527 (ppl 41.20)
Final gamma: 0.3000
Final α_k: [1.5697065691711032e-06, 1.5268801689671818e-06, 1.5262206716215587e-06, 1.4471440863417229e-06]
