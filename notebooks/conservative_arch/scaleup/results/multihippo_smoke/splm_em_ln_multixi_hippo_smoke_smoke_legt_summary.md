# Training summary — splm_em_ln_multixi_hippo_smoke_smoke_legt

- experiment: E12 HiPPO multi-channel-ξ SPLM scale-up
- model: ScalarPotentialLMSARFMassLNMultiHiPPO (em_ln + multi-ξ HiPPO)
- mode: smoke
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,539,907
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- xi_channels: 4  basis=legt  theta=200.0  learnable_dt=False
- block_size: 256  batch_size: 8  steps: 300
- causal_force: True
- seed: 0
- elapsed: 188 s (0.05 h)

Final val loss: 5.017250 (ppl 151.00)
Final gamma: 0.3000
Final HiPPO dt: 0.005
