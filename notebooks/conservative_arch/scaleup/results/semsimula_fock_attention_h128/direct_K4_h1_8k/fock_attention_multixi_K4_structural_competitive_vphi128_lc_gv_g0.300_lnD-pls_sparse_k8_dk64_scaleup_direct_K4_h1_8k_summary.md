# Training summary — fock_attention_multixi_K4_structural_competitive_vphi128_lc_gv_g0.300_lnD-pls_sparse_k8_dk64_scaleup_direct_K4_h1_8k

- experiment: Fock Attention (direct exchange force) scale-up
- model: FockAttentionPARFLM (K-EMA ξ + sparse PARF + §5.1 exchange)
- v_phi_kind: structural_competitive
- top_k: 8
- gumbel_tau: 1.0 -> 0.3
- mode: scaleup
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,649,172  V_theta=3,411,969  V_phi=19,074  score_head=24,641  xi_module=4  exchange_oh=65,537
- d=256  L=8  v_hidden=1024  v_depth=3  max_len=1024
- xi: K=4  learnable=True  α_init_mode=log_spaced  α_init=[0.0, 0.7845565309968117, 0.9535841116638722, 0.99]
- exchange: n_heads=1  d_k=64  scale_init=0.0
- block_size: 512  batch_size: 16  steps: 8000
- seed: 0
- elapsed: 6044 s (1.68 h)

Final val loss: 2.440676 (ppl 11.48)
Final gamma: 0.3000
Final α_k: [2.3737652554700617e-06, 0.5969452857971191, 0.8906123042106628, 0.9748552441596985]
Final exchange scale: -0.2547
