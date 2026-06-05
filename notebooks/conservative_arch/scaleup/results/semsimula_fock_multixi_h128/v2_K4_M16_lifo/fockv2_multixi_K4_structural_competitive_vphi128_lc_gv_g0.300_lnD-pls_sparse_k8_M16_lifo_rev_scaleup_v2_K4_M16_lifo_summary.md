# Training summary — fockv2_multixi_K4_structural_competitive_vphi128_lc_gv_g0.300_lnD-pls_sparse_k8_M16_lifo_rev_scaleup_v2_K4_M16_lifo

- experiment: Fock multi-channel ξ PARF scale-up
- model: FockMultiXiPARFLM (v2 gates + K-EMA ξ + sparse PARF)
- v_phi_kind: structural_competitive
- top_k: 8
- gumbel_tau: 1.0 -> 0.3
- mode: scaleup
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 17,162,205  V_theta=3,411,969  V_phi=19,074  score_head=24,641  xi_module=4  fock_oh=578,570
- d=256  L=8  v_hidden=1024  v_depth=3  max_len=1024
- xi: K=4  learnable=True  α_init_mode=log_spaced  α_init=[0.0, 0.7845565309968117, 0.9535841116638722, 0.99]
- fock: version=v2  M=16  discipline=LIFO  decay=0.5  thresh=0.005
- fock-v2: d_k=64  reverse_channel=True  tau_create_init=0.1
- block_size: 512  batch_size: 16  steps: 8000
- seed: 0
- elapsed: 7659 s (2.13 h)

Final val loss: 2.540570 (ppl 12.69)
Final gamma: 0.3000
Final α_k: [2.3443276404577773e-06, 0.601327121257782, 0.8999636769294739, 0.9772133827209473]
