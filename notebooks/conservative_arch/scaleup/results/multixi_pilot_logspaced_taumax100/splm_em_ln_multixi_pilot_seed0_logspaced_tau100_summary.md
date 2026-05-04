# Training summary — splm_em_ln_multixi_pilot_seed0_logspaced_tau100

- experiment: E11 multi-channel-ξ SPLM scale-up
- model: ScalarPotentialLMSARFMassLNMultiXi (em_ln + multi-ξ)
- mode: pilot
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,539,911
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- xi_channels: 4  learnable=True  α_init=[0.0, 0.7845565309968117, 0.9535841116638722, 0.99]
- block_size: 512  batch_size: 16  steps: 4000
- seed: 0
- elapsed: 29110 s (8.09 h)

Final val loss: 2.709861 (ppl 15.03)
Final gamma: 0.3000
Final α_k: [1.7069863815777353e-06, 0.6416216492652893, 0.919352114200592, 0.9844106435775757]
