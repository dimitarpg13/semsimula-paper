# SARF scalar-potential LM -- shakespeare training summary

Variant: **SARF-faithful** (xi recomputed at every layer).

- Device: `mps`
- Parameters: **7,123,075**
- Model config: `{'vocab_size': 50257, 'd': 128, 'max_len': 256, 'v_hidden': 512, 'v_depth': 3, 'L': 8, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 1.0, 'learn_mgamma': True}`
- Train config: `{'batch_size': 16, 'block_size': 128, 'steps': 4000, 'lr': 0.0005, 'weight_decay': 0.01, 'warmup_steps': 200, 'grad_clip': 1.0, 'eval_interval': 200, 'eval_iters': 40, 'log_interval': 50}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 1879s
- Final train loss: 4.5789
- Final val loss: 5.2586 (ppl 192.21)
- Final m = 0.9592, gamma = 0.9031
- Loss curve: `splm_sarf_shakespeare_loss_curve.png`
- Checkpoint: `splm_sarf_shakespeare_ckpt_latest.pt`
- Log: `splm_sarf_shakespeare_training_log.jsonl`
