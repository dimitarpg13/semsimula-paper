# Training summary — parf_multixi_K8_structural_competitive_vphi128_lc_gv_g0.300_lnD-pls_sparse_k8_scaleup_comp_K8

- experiment: multi-channel ξ PARF scale-up
- model: MultiXiPARFLM (K-EMA ξ + sparse PARF)
- v_phi_kind: structural_competitive
- top_k: 8
- gumbel_tau: 1.0 -> 0.1
- mode: scaleup
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 17,632,215  V_theta=4,460,545  V_phi=19,074  score_head=24,641  xi_module=8
- d=256  L=8  v_hidden=1024  v_depth=3  max_len=1024
- xi: K=8  learnable=True  α_init_mode=log_spaced  α_init=[0.0, 0.4820525320768788, 0.7317304204720274, 0.8610504505626863, 0.9280314326998848, 0.9627240627968506, 0.9806930227111675, 0.99]
- block_size: 512  batch_size: 16  steps: 8000
- seed: 0
- elapsed: 6885 s (1.91 h)

Final val loss: 2.489790 (ppl 12.06)
Final gamma: 0.3000
Final α_k: [1.0000741212934372e-06, 0.4828068017959595, 0.6404229998588562, 0.7880547642707825, 0.8876540660858154, 0.9446097612380981, 0.9680293202400208, 0.9857261180877686]
