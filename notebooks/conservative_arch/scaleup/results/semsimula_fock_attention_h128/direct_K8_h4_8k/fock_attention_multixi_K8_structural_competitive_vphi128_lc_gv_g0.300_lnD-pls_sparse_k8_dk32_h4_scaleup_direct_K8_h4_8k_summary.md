# Training summary — fock_attention_multixi_K8_structural_competitive_vphi128_lc_gv_g0.300_lnD-pls_sparse_k8_dk32_h4_scaleup_direct_K8_h4_8k

- experiment: Fock Attention (direct exchange force) scale-up
- model: FockAttentionPARFLM (K-EMA ξ + sparse PARF + §5.1 exchange)
- v_phi_kind: structural_competitive
- top_k: 8
- gumbel_tau: 1.0 -> 0.3
- mode: scaleup
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 17,763,288  V_theta=4,460,545  V_phi=19,074  score_head=24,641  xi_module=8  exchange_oh=131,073
- d=256  L=8  v_hidden=1024  v_depth=3  max_len=1024
- xi: K=8  learnable=True  α_init_mode=log_spaced  α_init=[0.0, 0.4820525320768788, 0.7317304204720274, 0.8610504505626863, 0.9280314326998848, 0.9627240627968506, 0.9806930227111675, 0.99]
- exchange: n_heads=4  d_k=32  scale_init=0.0
- block_size: 512  batch_size: 16  steps: 8000
- seed: 0
- elapsed: 7180 s (1.99 h)

Final val loss: 2.455365 (ppl 11.65)
Final gamma: 0.3000
Final α_k: [9.999891972256592e-07, 0.47860732674598694, 0.6394316554069519, 0.7902877926826477, 0.8840280771255493, 0.9430862665176392, 0.9658975601196289, 0.9845645427703857]
Final exchange scale: -0.1407
