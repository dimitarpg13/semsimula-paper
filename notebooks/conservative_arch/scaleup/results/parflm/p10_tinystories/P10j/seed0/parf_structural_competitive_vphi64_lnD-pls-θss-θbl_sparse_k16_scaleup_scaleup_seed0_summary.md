# Training summary — parf_structural_competitive_vphi64_lnD-pls-θss-θbl_sparse_k16_scaleup_scaleup_seed0

- experiment: P10 sparsity ladder at TinyStories scale
- cell: P10j
- model: SparsePARFLM (full P5+P7+P8 stack)
- v_phi_kind: structural_competitive
- top_k: **16** (Gumbel-softmax sparse routing)
- gumbel_tau: 1.0 -> 0.1
- corpus: TinyStories (5,000,000 train tokens)
- params: 22,610,703  V_theta=9,445,377  V_phi=12,738  score_head=24,641
- d=256  L=8  v_hidden=2048  v_depth=3  max_len=1024
- v_phi inner: phi_hidden=64  theta_hidden=64
- block_size: 512  batch_size: 16  steps: 16000
- seed: 0
- elapsed: 10411 s (2.89 h)

## Results

- Best val PPL: **27.73** @ step 14400
- Final val PPL: 28.66
- Final gamma: 0.1954
- Final gumbel_tau: 0.1001

## Comparison to P10g (k=4 baseline)

| Cell | top_k | Best val PPL | Final val PPL | Final γ |
| --- | ---: | ---: | ---: | ---: |
| P10g | 4 | 26.42 | 27.16 | 0.134 |
| P10j | 16 | 27.73 | 28.66 | 0.195 |

Δ best PPL vs P10g: +1.31

**Verdict:** k=16 is worse than k=4 at TinyStories scale (+1.31 PPL). Consistent with the Shakespeare-scale sparsity-ladder ordering.
