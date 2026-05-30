# Training summary — splm_em_ln_multixi_pilot_fast_only

- experiment: E11 multi-channel-ξ SPLM scale-up
- model: ScalarPotentialLMSARFMassLNMultiXi (em_ln + multi-ξ)
- mode: pilot
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,539,911
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- xi_channels: 4  learnable=True  α_init=[0.0, 0.0, 0.1, 0.2]
- block_size: 512  batch_size: 16  steps: 4000
- seed: 0
- elapsed: 1027 s (0.29 h)

Final val loss: 3.079959 (ppl 21.76)
Final gamma: 0.3000
Final α_k: [1.1881825230375398e-06, 1.1453118986537447e-06, 0.21329258382320404, 0.38079383969306946]
