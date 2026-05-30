# Training summary — splm_em_ln_multixi_pilot_uniform_narrow

- experiment: E11 multi-channel-ξ SPLM scale-up
- model: ScalarPotentialLMSARFMassLNMultiXi (em_ln + multi-ξ)
- mode: pilot
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,539,911
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- xi_channels: 4  learnable=True  α_init=[0.85, 0.9, 0.95, 0.99]
- block_size: 512  batch_size: 16  steps: 4000
- seed: 0
- elapsed: 1030 s (0.29 h)

Final val loss: 2.703253 (ppl 14.93)
Final gamma: 0.3000
Final α_k: [0.7039780616760254, 0.7917513847351074, 0.9325516819953918, 0.9861944913864136]
