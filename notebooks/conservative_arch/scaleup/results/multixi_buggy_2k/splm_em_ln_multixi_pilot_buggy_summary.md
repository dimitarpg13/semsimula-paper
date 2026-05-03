# Training summary — splm_em_ln_multixi_pilot_buggy

- experiment: E11 multi-channel-ξ SPLM scale-up
- model: ScalarPotentialLMSARFMassLNMultiXi (em_ln + multi-ξ)
- mode: pilot
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,539,911
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- xi_channels: 4  learnable=True  α_init=[0.0, 0.5, 0.9, 0.99]
- block_size: 512  batch_size: 16  steps: 2000
- seed: 0
- elapsed: 17513 s (4.86 h)

Final val loss: 0.044048 (ppl 1.05)
Final gamma: 0.3000
Final α_k: [9.999964731832733e-07, 0.41416192054748535, 0.8513372540473938, 0.9846636056900024]
