# Training summary — matched_baseline_scaleup_scaleup_seed0

- experiment: E9 scale-up (matched-attention arm)
- model: MatchedGPT
- mode: scaleup
- corpus: TinyStories (cap 5,000,000 train tokens)
- variant: matched_attention_baseline
- params: 19,446,528
- d=256  L=8  n_head=4  mlp_mult=4  max_len=1024  tie_embeddings=True
- block_size: 512  batch_size: 16  steps: 8000
- seed: 0
- elapsed: 24296 s (6.75 h)

Final val loss: 2.055737 (ppl 7.81)
