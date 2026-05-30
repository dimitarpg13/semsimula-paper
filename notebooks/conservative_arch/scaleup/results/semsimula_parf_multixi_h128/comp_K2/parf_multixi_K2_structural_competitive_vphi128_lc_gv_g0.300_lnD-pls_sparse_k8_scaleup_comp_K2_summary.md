# Training summary — parf_multixi_K2_structural_competitive_vphi128_lc_gv_g0.300_lnD-pls_sparse_k8_scaleup_comp_K2

- experiment: multi-channel ξ PARF scale-up
- model: MultiXiPARFLM (K-EMA ξ + sparse PARF)
- v_phi_kind: structural_competitive
- top_k: 8
- gumbel_tau: 1.0 -> 0.1
- mode: scaleup
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,059,345  V_theta=2,887,681  V_phi=19,074  score_head=24,641  xi_module=2
- d=256  L=8  v_hidden=1024  v_depth=3  max_len=1024
- xi: K=2  learnable=True  α_init_mode=explicit  α_init=[0.5, 0.95]
- block_size: 512  batch_size: 16  steps: 8000
- seed: 0
- elapsed: 5395 s (1.50 h)

Final val loss: 2.601028 (ppl 13.48)
Final gamma: 0.3000
Final α_k: [0.5208654999732971, 0.9210228323936462]
