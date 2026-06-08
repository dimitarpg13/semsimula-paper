# Training summary — splm_em_ln_multixi_scaleup_scaleup_8k

- experiment: E11 multi-channel-ξ SPLM scale-up
- model: ScalarPotentialLMSARFMassLNMultiXi (em_ln + multi-ξ)
- mode: scaleup
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,539,911
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- xi_channels: 4  learnable=True  α_init=[0.25, 0.5, 0.75, 0.95]
- block_size: 512  batch_size: 16  steps: 8000
- seed: 0
- elapsed: 1997 s (0.55 h)

Final val loss: 2.524788 (ppl 12.49)
Final gamma: 0.3000
Final α_k: [0.23349103331565857, 0.5611025094985962, 0.781069278717041, 0.9583898186683655]
