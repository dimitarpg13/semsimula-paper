# Training summary — splm_em_ln_multixi_pilot_uniform_wide

- experiment: E11 multi-channel-ξ SPLM scale-up
- model: ScalarPotentialLMSARFMassLNMultiXi (em_ln + multi-ξ)
- mode: pilot
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,539,911
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- xi_channels: 4  learnable=True  α_init=[0.1, 0.4, 0.7, 0.95]
- block_size: 512  batch_size: 16  steps: 4000
- seed: 0
- elapsed: 1029 s (0.29 h)

Final val loss: 2.699341 (ppl 14.87)
Final gamma: 0.3000
Final α_k: [0.09186152368783951, 0.474901020526886, 0.7214352488517761, 0.9506804943084717]
