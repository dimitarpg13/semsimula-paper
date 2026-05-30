# Training summary — parf_multixi_K4_structural_competitive_vphi128_lc_gv_g0.300_lnD-pls_sparse_k4_scaleup_comp_K4_k4

- experiment: multi-channel ξ PARF scale-up
- model: MultiXiPARFLM (K-EMA ξ + sparse PARF)
- v_phi_kind: structural_competitive
- top_k: 4
- gumbel_tau: 1.0 -> 0.1
- mode: scaleup
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: 0.3
- params: 16,583,635  V_theta=3,411,969  V_phi=19,074  score_head=24,641  xi_module=4
- d=256  L=8  v_hidden=1024  v_depth=3  max_len=1024
- xi: K=4  learnable=True  α_init_mode=explicit  α_init=[0.25, 0.5, 0.75, 0.95]
- block_size: 512  batch_size: 16  steps: 8000
- seed: 0
- elapsed: 5680 s (1.58 h)

Final val loss: 2.573590 (ppl 13.11)
Final gamma: 0.3000
Final α_k: [0.2508236765861511, 0.5628144145011902, 0.7744423747062683, 0.9559810161590576]
