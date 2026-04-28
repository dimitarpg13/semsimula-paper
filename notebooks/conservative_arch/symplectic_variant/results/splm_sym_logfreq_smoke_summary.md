# Symplectic SPLM (Verlet + logfreq) -- smoke training summary

Variant: **velocity-Verlet + Strang-split damping**,  `mass_mode = logfreq`.

- Device: `mps`
- Parameters: **3,257,796**
- Model config: `{'vocab_size': 50257, 'd': 64, 'max_len': 128, 'v_hidden': 128, 'v_depth': 2, 'L': 4, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 1.0, 'learn_mgamma': True, 'mass_mode': 'logfreq', 'logfreq_init_alpha': 0.1, 'logfreq_path': '/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/symplectic_variant/results/logfreq_surprisal.npy'}`
- Train config: `{'batch_size': 8, 'block_size': 64, 'steps': 300, 'lr': 0.001, 'weight_decay': 0.0, 'warmup_steps': 20, 'grad_clip': 1.0, 'eval_interval': 50, 'eval_iters': 20, 'log_interval': 10}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 45s
- Final train loss: 6.4660
- Final val loss: 6.5598 (ppl 706.16)
- Final mass: mean=1.4078, std=0.1891, min=1.1220, max=1.8127
- Final gamma: 0.9759
- Loss curve: `splm_sym_logfreq_smoke_loss_curve.png`
- Checkpoint: `splm_sym_logfreq_smoke_ckpt_latest.pt`
- Log: `splm_sym_logfreq_smoke_training_log.jsonl`
