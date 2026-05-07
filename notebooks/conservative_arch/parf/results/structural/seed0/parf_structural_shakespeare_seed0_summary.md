# PARF Q9c -- shakespeare training summary

- Tag: `parf_structural_shakespeare_seed0`
- V_phi kind: `structural` (structural = §5.1-faithful; mlp = unstructured ablation)
- Architecture: PARF-augmented SPLM (Q9c) -- single shared V_theta (4-layer MLP) plus single shared V_phi pair interaction with causal reduction (past tokens = fixed external sources via.detach), velocity-Verlet damped Euler-Lagrange step at every layer, Algorithm A (NTP-only backprop, no Gumbel sparsity).
- L (depth): 8
- Device: `mps`
- Parameters: **6,535,718** (target ~8.0 M, delta -1.464 M; V_theta=66,049, V_phi=4,002)
- Model config: `{'vocab_size': 50257, 'd': 128, 'max_len': 256, 'L': 8, 'v_hidden': 128, 'v_depth': 3, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 0.15, 'learn_mgamma': True, 'fixed_gamma': None, 'v_phi_kind': 'structural', 'v_phi_d_type': 16, 'v_phi_d_angle': 8, 'v_phi_phi_hidden': 32, 'v_phi_theta_hidden': 32, 'v_phi_mlp_hidden': 64, 'v_phi_C': 1.0, 'v_phi_eps': 0.01, 'v_phi_init_scale': 0.02, 'mass_mode': 'logfreq', 'logfreq_init_alpha': 0.1, 'logfreq_path': '/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy', 'ln_after_step': True, 'ln_eps': 1e-05, 'causal_force': True, 'tie_embeddings': True}`
- Train config: `{'batch_size': 16, 'block_size': 128, 'steps': 4000, 'lr': 0.0005, 'weight_decay': 0.01, 'warmup_steps': 200, 'grad_clip': 1.0, 'eval_interval': 200, 'eval_iters': 40, 'log_interval': 50}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 15554s
- Final train loss: 4.7590
- Final val loss: 5.3497 (ppl 210.54)
- Final gamma: 0.0883
- Loss curve: `parf_structural_shakespeare_seed0_loss_curve.png`
- Checkpoint: `parf_structural_shakespeare_seed0_ckpt_latest.pt`
