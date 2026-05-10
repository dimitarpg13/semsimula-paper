# Scalar-potential LM -- shakespeare training summary

- Device: `cuda`
- Parameters: **7,123,075**
- Model config: `{'vocab_size': 50257, 'd': 128, 'max_len': 256, 'v_hidden': 512, 'v_depth': 3, 'L': 8, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 1.0, 'learn_mgamma': True}`
- Train config: `{'batch_size': 16, 'block_size': 128, 'steps': 4000, 'lr': 0.0005, 'weight_decay': 0.01, 'warmup_steps': 200, 'grad_clip': 1.0, 'eval_interval': 200, 'eval_iters': 40, 'log_interval': 50}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 421s
- Final train loss: 5.0671
- Final val loss: 5.6305 (ppl 278.81)
- Final m = 0.9806, gamma = 0.9620
- Loss curve: `splm_shakespeare_loss_curve.png`
- Checkpoint: `splm_shakespeare_ckpt_latest.pt`
- Log: `splm_shakespeare_training_log.jsonl`
