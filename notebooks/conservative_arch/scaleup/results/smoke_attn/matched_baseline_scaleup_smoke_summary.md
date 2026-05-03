# Training summary — matched_baseline_scaleup_smoke

- experiment: E9 scale-up (matched-attention arm)
- model: MatchedGPT
- mode: smoke
- corpus: TinyStories (cap 5,000,000 train tokens)
- variant: matched_attention_baseline
- params: 19,446,528
- d=256  L=8  n_head=4  mlp_mult=4  max_len=1024  tie_embeddings=True
- block_size: 256  batch_size: 8  steps: 300
- seed: 0
- elapsed: 107 s (0.03 h)

Final val loss: 4.289235 (ppl 72.91)
