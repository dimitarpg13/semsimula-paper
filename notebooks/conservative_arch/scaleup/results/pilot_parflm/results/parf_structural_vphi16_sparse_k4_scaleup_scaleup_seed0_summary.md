# Training summary — parf_structural_vphi16_sparse_k4_scaleup_scaleup_seed0

- experiment: paper_tmlr_1 scale-up pilot (PARF Q9c arm)
- model: SparsePARFLM
- v_phi_kind: structural
- top_k (Gumbel-softmax sparse): 4
- gumbel_tau: 1.0 -> 0.1  (anneal 0.8)
- mode: scaleup
- corpus: TinyStories (cap 5,000,000 train tokens)
- fixed_gamma: None
- params: 15,791,255  V_theta=2,625,537  V_phi=13,138
- d=256  L=8  v_hidden=1024  v_depth=3  max_len=1024
- v_phi inner: phi_hidden=16  theta_hidden=16  d_type=32  d_angle=16
- block_size: 512  batch_size: 16  steps: 8000
- seed: 0
- elapsed: 5030 s (1.40 h)

Final val loss: 3.484170 (ppl 32.60)
Final gamma: 0.1500
