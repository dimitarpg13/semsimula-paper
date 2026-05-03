# Training summary — splm_em_ln_multixi_pilot_fixed

- experiment: E11 multi-channel-ξ SPLM scale-up
- model: ScalarPotentialLMSARFMassLNMultiXi (em_ln + multi-ξ)
- mode: pilot
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,539,911
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- xi_channels: 4  learnable=True  α_init=[0.0, 0.5, 0.9, 0.99]
- block_size: 512  batch_size: 16  steps: 4000
- seed: 0
- elapsed: 33285 s (9.25 h)

Final val loss: 2.692503 (ppl 14.77)
Final gamma: 0.3000
Final α_k: [1.2085392881999724e-06, 0.5190631151199341, 0.8546685576438904, 0.9793670773506165]
