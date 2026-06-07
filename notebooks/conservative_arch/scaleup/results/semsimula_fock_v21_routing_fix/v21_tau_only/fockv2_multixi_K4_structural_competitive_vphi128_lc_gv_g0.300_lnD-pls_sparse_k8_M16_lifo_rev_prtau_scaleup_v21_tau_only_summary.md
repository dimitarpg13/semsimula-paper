# Training summary — fockv2_multixi_K4_structural_competitive_vphi128_lc_gv_g0.300_lnD-pls_sparse_k8_M16_lifo_rev_prtau_scaleup_v21_tau_only

- experiment: Fock multi-channel ξ PARF scale-up
- model: FockMultiXiPARFLM (v2 gates + K-EMA ξ + sparse PARF)
- v_phi_kind: structural_competitive
- top_k: 8
- gumbel_tau: 1.0 -> 0.3
- mode: scaleup
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 17,162,220  V_theta=3,411,969  V_phi=19,074  score_head=24,641  xi_module=4  fock_oh=578,585
- d=256  L=8  v_hidden=1024  v_depth=3  max_len=1024
- xi: K=4  learnable=True  α_init_mode=log_spaced  α_init=[0.0, 0.7845565309968117, 0.9535841116638722, 0.99]
- fock: version=v2  M=16  discipline=LIFO  decay=0.5  thresh=0.005
- fock-v2: d_k=64  reverse_channel=True  tau_create_init=8.0  v2.1=[per_reg_tau]
- block_size: 512  batch_size: 16  steps: 16000
- seed: 0
- elapsed: 14481 s (4.02 h)

Final val loss: 2.413969 (ppl 11.18)
Final gamma: 0.3000
Final α_k: [3.02076296065934e-06, 0.5850334763526917, 0.8803706169128418, 0.9696952700614929]
