# SARF+mass(embed_head) -- shakespeare training summary

Variant: **SARF-faithful with mass_mode = `embed_head`**.

- Device: `mps`
- Parameters: **7,123,204**
- Model config: `{'vocab_size': 50257, 'd': 128, 'max_len': 256, 'v_hidden': 512, 'v_depth': 3, 'L': 8, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 1.0, 'learn_mgamma': True, 'mass_mode': 'embed_head', 'logfreq_init_alpha': 0.1, 'logfreq_path': None}`
- Train config: `{'batch_size': 16, 'block_size': 128, 'steps': 4000, 'lr': 0.0005, 'weight_decay': 0.01, 'warmup_steps': 200, 'grad_clip': 1.0, 'eval_interval': 200, 'eval_iters': 40, 'log_interval': 50}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 2367s
- Final train loss: 4.5001
- Final val loss: 5.4068 (ppl 222.91)
- Final mass: mean=0.8406, std=0.2120, min=0.3736, max=1.7970
- Final gamma: 0.9132
- Loss curve: `splm_sarfmass_embed_head_shakespeare_loss_curve.png`
- Checkpoint: `splm_sarfmass_embed_head_shakespeare_ckpt_latest.pt`
- Log: `splm_sarfmass_embed_head_shakespeare_training_log.jsonl`
