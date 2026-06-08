# Training summary — splm_em_ln_multixi_scaleup_extended_16k

- experiment: E11 multi-channel-ξ SPLM scale-up
- model: ScalarPotentialLMSARFMassLNMultiXi (em_ln + multi-ξ)
- mode: scaleup
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,539,911
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- xi_channels: 4  learnable=True  α_init=[0.25, 0.5, 0.75, 0.95]
- block_size: 512  batch_size: 16  steps: 16000
- seed: 0
- elapsed: 3979 s (1.11 h)

Final val loss: 2.443249 (ppl 11.51)
Final gamma: 0.3000
Final α_k: [0.24926145374774933, 0.5983831286430359, 0.8079418540000916, 0.9606558084487915]
