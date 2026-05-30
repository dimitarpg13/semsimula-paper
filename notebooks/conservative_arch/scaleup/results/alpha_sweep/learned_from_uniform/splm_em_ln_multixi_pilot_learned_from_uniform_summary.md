# Training summary — splm_em_ln_multixi_pilot_learned_from_uniform

- experiment: E11 multi-channel-ξ SPLM scale-up
- model: ScalarPotentialLMSARFMassLNMultiXi (em_ln + multi-ξ)
- mode: pilot
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,539,911
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- xi_channels: 4  learnable=True  α_init=[0.25, 0.5, 0.75, 0.95]
- block_size: 512  batch_size: 16  steps: 4000
- seed: 0
- elapsed: 1028 s (0.29 h)

Final val loss: 2.686827 (ppl 14.69)
Final gamma: 0.3000
Final α_k: [0.22953851521015167, 0.5342720150947571, 0.7627133727073669, 0.955636203289032]
