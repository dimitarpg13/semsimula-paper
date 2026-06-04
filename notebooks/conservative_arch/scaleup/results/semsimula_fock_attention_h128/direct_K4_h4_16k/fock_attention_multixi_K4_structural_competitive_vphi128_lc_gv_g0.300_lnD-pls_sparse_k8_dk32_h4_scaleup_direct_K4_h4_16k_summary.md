# Training summary — fock_attention_multixi_K4_structural_competitive_vphi128_lc_gv_g0.300_lnD-pls_sparse_k8_dk32_h4_scaleup_direct_K4_h4_16k

- experiment: Fock Attention (direct exchange force) scale-up
- model: FockAttentionPARFLM (K-EMA ξ + sparse PARF + §5.1 exchange)
- v_phi_kind: structural_competitive
- top_k: 8
- gumbel_tau: 1.0 -> 0.3
- mode: scaleup
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,714,708  V_theta=3,411,969  V_phi=19,074  score_head=24,641  xi_module=4  exchange_oh=131,073
- d=256  L=8  v_hidden=1024  v_depth=3  max_len=1024
- xi: K=4  learnable=True  α_init_mode=log_spaced  α_init=[0.0, 0.7845565309968117, 0.9535841116638722, 0.99]
- exchange: n_heads=4  d_k=32  scale_init=0.0
- block_size: 512  batch_size: 16  steps: 16000
- seed: 0
- elapsed: 12290 s (3.41 h)

Final val loss: 2.242890 (ppl 9.42)
Final gamma: 0.3000
Final α_k: [2.3293428057513665e-06, 0.5741551518440247, 0.8592199087142944, 0.9709038138389587]
Final exchange scale: -0.3184
