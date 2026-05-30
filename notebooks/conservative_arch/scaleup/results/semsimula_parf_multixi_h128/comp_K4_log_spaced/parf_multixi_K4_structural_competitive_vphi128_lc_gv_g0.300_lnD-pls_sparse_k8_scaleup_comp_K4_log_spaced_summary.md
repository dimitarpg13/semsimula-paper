# Training summary — parf_multixi_K4_structural_competitive_vphi128_lc_gv_g0.300_lnD-pls_sparse_k8_scaleup_comp_K4_log_spaced

- experiment: multi-channel ξ PARF scale-up
- model: MultiXiPARFLM (K-EMA ξ + sparse PARF)
- v_phi_kind: structural_competitive
- top_k: 8
- gumbel_tau: 1.0 -> 0.1
- mode: scaleup
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,583,635  V_theta=3,411,969  V_phi=19,074  score_head=24,641  xi_module=4
- d=256  L=8  v_hidden=1024  v_depth=3  max_len=1024
- xi: K=4  learnable=True  α_init_mode=log_spaced  α_init=[0.0, 0.7845565309968117, 0.9535841116638722, 0.99]
- block_size: 512  batch_size: 16  steps: 8000
- seed: 0
- elapsed: 5971 s (1.66 h)

Final val loss: 2.523326 (ppl 12.47)
Final gamma: 0.3000
Final α_k: [2.372547669438063e-06, 0.603515625, 0.8985836505889893, 0.9763656854629517]
