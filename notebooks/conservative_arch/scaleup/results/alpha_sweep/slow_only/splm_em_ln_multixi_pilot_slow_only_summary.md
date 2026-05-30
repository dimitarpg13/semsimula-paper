# Training summary — splm_em_ln_multixi_pilot_slow_only

- experiment: E11 multi-channel-ξ SPLM scale-up
- model: ScalarPotentialLMSARFMassLNMultiXi (em_ln + multi-ξ)
- mode: pilot
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,539,911
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- xi_channels: 4  learnable=True  α_init=[0.95, 0.97, 0.99, 0.999]
- block_size: 512  batch_size: 16  steps: 4000
- seed: 0
- elapsed: 1030 s (0.29 h)

Final val loss: 2.822752 (ppl 16.82)
Final gamma: 0.3000
Final α_k: [0.8784022331237793, 0.9235950112342834, 0.9790964722633362, 0.9981166124343872]
