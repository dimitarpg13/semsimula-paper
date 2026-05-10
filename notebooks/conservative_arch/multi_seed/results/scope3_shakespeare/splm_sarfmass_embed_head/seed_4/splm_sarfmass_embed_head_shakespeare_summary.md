# SARF+mass(embed_head) -- shakespeare training summary

Variant: **SARF-faithful with mass_mode = `embed_head`**.

- Device: `cuda`
- Parameters: **7,123,204**
- Model config: `{'vocab_size': 50257, 'd': 128, 'max_len': 256, 'v_hidden': 512, 'v_depth': 3, 'L': 8, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 1.0, 'learn_mgamma': True, 'mass_mode': 'embed_head', 'logfreq_init_alpha': 0.1, 'logfreq_path': None, 'fixed_gamma': None, 'causal_force': True}`
- Train config: `{'batch_size': 16, 'block_size': 128, 'steps': 4000, 'lr': 0.0005, 'weight_decay': 0.01, 'warmup_steps': 200, 'grad_clip': 1.0, 'eval_interval': 200, 'eval_iters': 40, 'log_interval': 50}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 424s
- Final train loss: 4.7609
- Final val loss: 5.7568 (ppl 316.34)
- Final mass: mean=0.8980, std=0.2523, min=0.1875, max=2.4023
- Final gamma: 0.9559
- Loss curve: `splm_sarfmass_embed_head_shakespeare_loss_curve.png`
- Checkpoint: `splm_sarfmass_embed_head_shakespeare_ckpt_latest.pt`
- Log: `splm_sarfmass_embed_head_shakespeare_training_log.jsonl`
