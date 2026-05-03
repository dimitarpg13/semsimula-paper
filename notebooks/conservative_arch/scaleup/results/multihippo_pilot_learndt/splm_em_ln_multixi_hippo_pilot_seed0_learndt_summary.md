# Training summary — splm_em_ln_multixi_hippo_pilot_seed0_learndt

- experiment: E12 HiPPO multi-channel-ξ SPLM scale-up
- model: ScalarPotentialLMSARFMassLNMultiHiPPO (em_ln + multi-ξ HiPPO)
- mode: pilot
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,539,908
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- xi_channels: 4  basis=legt  theta=200.0  learnable_dt=True
- block_size: 512  batch_size: 16  steps: 4000
- causal_force: True
- seed: 0
- elapsed: 29186 s (8.11 h)

Final val loss: 2.859483 (ppl 17.45)
Final gamma: 0.3000
Final HiPPO dt: 0.0133194
