# Training summary — splm_em_ln_multixi_pilot_R6h1_logspaced

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
- elapsed: 1028 s (0.29 h)

Final val loss: 2.708494 (ppl 15.01)
Final gamma: 0.3000
Final α_k: [1.7208413964908686e-06, 0.6429312229156494, 0.9175439476966858, 0.9843553304672241]
