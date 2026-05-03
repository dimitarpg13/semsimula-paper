# Training summary — splm_em_ln_multixi_s4d_pilot_seed0_legtinit

- experiment: E13 S4D multi-channel-ξ SPLM scale-up
- model: ScalarPotentialLMSARFMassLNMultiS4D (em_ln + multi-ξ S4D)
- mode: pilot
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,539,920
- d=256  L=8  v_hidden=1024  max_len=1024  ln_after_step=True
- xi_channels: 4  init=legt  theta(init)=200.0  learnable_dt=True  learnable_B=True
- block_size: 512  batch_size: 16  steps: 4000
- causal_force: True
- final_val_loss: 2.8249
- final_val_ppl:  16.86
- final_gamma:    0.3000
- final_dt:       0.01372
- final_eigvals:  (-8.005+5.325j) (-8.038-5.306j) (-4.754+1.108j) (-5.076-1.134j)
- final_b_proj:   ['+2.7560', '+3.1328', '+1.1613', '+1.8556']
- elapsed_sec:    31545
- ckpt:           splm_em_ln_multixi_s4d_pilot_seed0_legtinit_ckpt_latest.pt
