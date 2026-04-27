# Matched GPT-2-style baseline -- shakespeare

- Device: `mps`
- Parameters: **8,052,096**
- Model config: `{'vocab_size': 50257, 'd': 128, 'max_len': 256, 'n_layer': 8, 'n_head': 4, 'mlp_mult': 4, 'dropout': 0.0, 'tie_embeddings': True}`
- Train config: `{'batch_size': 16, 'block_size': 128, 'steps': 4000, 'lr': 0.0005, 'weight_decay': 0.01, 'warmup_steps': 200, 'grad_clip': 1.0, 'eval_interval': 200, 'eval_iters': 40, 'log_interval': 50}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 1742s
- Final train loss: 3.5056
- Final val loss: 5.0726 (ppl 159.59)
- Loss curve: `matched_shakespeare_loss_curve.png`
- Checkpoint: `matched_shakespeare_ckpt_latest.pt`
