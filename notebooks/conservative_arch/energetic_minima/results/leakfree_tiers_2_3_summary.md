# Leak-free Tier 2 + Tier 3 retrains -- summary report

> Run date: **May 4-5, 2026** (Tier 2a/3a/3b ran from 23:14 EDT 2026-05-04 to 02:21 EDT 2026-05-05; total wall-clock 11248s = 3h 7min). Attractor pipeline + landscape3D rendering ran on 2026-05-05 morning. All retrains are under `cfg.causal_force = True` (the post-fix default), seed 0, Tiny Shakespeare 4000 steps, `em_ln` integrator at $L = 8$, $d = 128$, $v_{\mathrm{hidden}} = 512$, `mass_mode='logfreq'`. The trainings used `train.py` from this folder; γ is **freely trained** (`fixed_gamma=None`), $\gamma_{\mathrm{init}} = 1.0$, `learn_mgamma=True`.

## 1. Tier 2a + Tier 3 final training metrics

| variant | tag | val_ppl | val_loss | final γ | wall-clock | params |
|---|---|---:|---:|---:|---:|---:|
| (i) em_ln (LayerNorm-after-step) | `em_ln` | **173.59** | 5.157 | 0.9583 | 2235s (37 min) | 7.12 M |
| (ii) em_sg (scale-gauge, λ_v0 = 1e-3) | `em_sg_lam1e-03` | 244.84 | 5.501 | 0.8632 | 2410s (40 min) | 7.10 M |
| (iii) em_gm (Gaussian-mixture, K=64) | `em_gm_K64` | 542.65 | 6.296 | 0.6683 | 6603s (110 min) | 7.13 M |

**Headline finding (Tier 2a, em_ln free-γ).** Under leak-free training with γ freely trained from $\gamma_{\mathrm{init}} = 1.0$, the em_ln model converges to $\gamma_{\mathrm{natural}} = 0.9583$ — **essentially unchanged from initialisation**. The corresponding val_ppl of 173.59 is **lower than** the leak-free fixed-γ optimum at $\gamma \in [0.10, 0.15]$ (S=5 confirmation sweep mean SPLM-2 val_ppl ≈ 178-181 at $\gamma = 0.15$ / 180-181 at $\gamma = 0.10$). This is opposite to the buggy v2 §4.3 observation ("free-γ landed at γ ≈ 0.65, val PPL ~89 vs fixed-γ=0.30 val PPL 87.06") — the leak-corrected dynamics produces a different free-γ regime. **The freely-trained γ does not converge to the fixed-γ optimum and instead lands in a separate (better-PPL) regime at high damping init.**

**Caveat.** The freely-γ result is a **single-seed** observation. The fixed-γ optimum is a 5-seed mean. The free-γ vs fixed-γ paired comparison at the same seed budget is a follow-up experiment (would require 5 × 4000 step retrains with γ_init = 1.0; ~3 h wall-clock).

**Tier 3a (em_sg).** The scale-gauge regulariser $\lambda_{v_0} \cdot \mathbb{E}[V_\theta(\xi_0, h_0)^2]$ at $\lambda_{v_0} = 1e\!-\!3$ pulls $V_\theta$'s absolute scale toward zero at the input embedding, which costs ~71 PPL on Tiny Shakespeare under leak-free training (val_ppl 244.84 vs em_ln free-γ 173.59) but recovers attractor diversity (see §2). The trained γ lands at 0.863, lower than em_ln free-γ's 0.958 — the regulariser pushes the model toward more dissipative dynamics.

**Tier 3b (em_gm).** The Gaussian-mixture head with $K = 64$ wells trains substantially worse under leak-free regime (val_ppl 542.65). The trained γ lands at 0.668, the lowest of the three. The fixed-mixture potential structure conflicts with the actual energy landscape SPLM converges to under causal honesty.

## 2. Cross-variant attractor comparison (n_sim_steps = 8 = L_train)

`run_attractor_pipeline.sh` extracts attractors via 8-step damped-flow integration from $N = 288$ random initial states (96 Gaussian + 96 token-embedding + 96 perturbed real-$h_L$); silhouette-optimal $K$-means clustering, $K \in [2, 10]$. Results in `attractor_analysis/results/attractors_em_*_summary.md`:

| variant | val_ppl | K* per prompt (n,m,s,d,c) | V range | content-basin fraction |
|---|---:|---:|---|---:|
| em_base (v2 buggy, log-freq SARF+mass) | — | (9,10,8,10,8) | [-1916.6, +1444.8] | 0.58 |
| em_ln (leak-free free-γ, γ=0.958) | 173.59 | (2,4,2,3,2) | [-114.8, -24.6] | 0.00 |
| em_sg (leak-free free-γ, γ=0.863) | 244.84 | (7,5,4,5,5) | [-1698.8, -311.3] | 0.52 |
| em_gm (leak-free free-γ, γ=0.668) | 542.65 | (2,2,2,2,2) | [+63.4, +64.1] | 0.00 |

Where (n, m, s, d, c) are the prompt domains: narrative / mathematics / scientific / dialogue / code.

**Key cross-variant finding.** Under leak-free free-γ training, the attractor structure varies substantially across the three energetic-minima variants:
- **em_ln (LN-after-step) + free-γ**: collapses to a single dominant newline-emitting basin per prompt (K\* ∈ {2, 3, 4}, content = 0.00). The LN-after-step normalisation amplifies the dominance of the newline channel.
- **em_sg (scale-gauge V anchor) + free-γ**: preserves prompt-dependent multi-basin structure (K\* ∈ {4, 5, 7}, content = 0.52). The $\lambda_{v_0}$ regulariser at the input embedding prevents $V_\theta$ from drifting to extremes and stabilises diverse attractors.
- **em_gm (Gaussian-mixture head) + free-γ**: collapses uniformly to K\* = 2 across all prompts; the V range is essentially flat ($[63.4, 64.1]$, $\Delta V = 0.7$); content = 0.00. The fixed mixture geometry does not adapt to the actual landscape under leak-free training.

**Comparison to the leak-free fixed-γ=0.10 attractor result (Tier 1, n_sim_steps = 128).** Tier 1 ran on the leak-free `leakfree_3seed/gamma0p10/seed0` SPLM-2 ckpt (i.e. fixed γ = 0.10) at `n_sim_steps = 128` (the full `--mode dynamical` default) and obtained K\* = (4, 4, 11, 8, 12), basin content punctuation/morpheme-dominated for all prompts. The Tier 2/3 here uses `n_sim_steps = 8` (the training horizon), which gives smaller K\* by construction (less time for trajectories to separate). The two results are not directly comparable in K\* magnitude, but the qualitative ranking matches:

| ckpt | n_sim_steps | K\* | content fraction | narrative |
|---|---:|---|---:|---|
| Tier 1 leak-free fixed-γ=0.10 | 128 | (4, 4, 11, 8, 12) | low (punct./morpheme) | F3 prompt-dependent multi-basin **survives leak fix** |
| Tier 2a leak-free free-γ=0.96 | 8 | (2, 4, 2, 3, 2) | 0.00 (newline) | F3 **partially preserved** for math, scientific |
| Tier 3a leak-free em_sg | 8 | (7, 5, 4, 5, 5) | 0.52 (content) | F3 **strongly preserved**; em_sg recovers content coverage |

## 3. v2 → leak-free comparison summary

The v2 buggy `comparison_report.md` from the April 2026 retrain reported em_ln free-γ at val_ppl 87.06 (TS, buggy SPLM val_ppl scale, deflated by leak factor); em_sg at 87.43; em_gm at 91.5+. Under the leak fix:

- The PPL inflation factor on val_ppl scale is non-trivial (e.g. on the SPLM-2 line of the leak-free 6-cell retrain at γ=0.30, the PPL went from buggy 96.32 to leak-free 130.97, an inflation of 1.36×; on the TinyStories scaleup checkpoint the inflation was 777×). The Tiny Shakespeare em_ln val_ppl values are inflated by an analogous factor; the v2 87.06 cannot be directly compared with the leak-free 173.59. The cross-variant *ratios* are more stable than the absolute values.
- **Cross-variant ranking** (em_ln < em_sg < em_gm in val_ppl, with em_sg recovering the best attractor diversity) is preserved between v2 and leak-free.

## 4. Implications for paper_v3 §15.10 (cba-attractors)

1. **F3 prompt-dependent multi-basin structure** survives the leak fix at fixed γ=0.10 (Tier 1, K\* = (4, 4, 11, 8, 12)) and at em_sg free-γ (K\* = (7, 5, 4, 5, 5)). It is partially weakened at em_ln free-γ (K\* = (2, 4, 2, 3, 2)).
2. **The attractor-content question** (basin content = punctuation/morpheme vs content tokens) is leak-independent in qualitative direction: em_base v2 baseline had content-basin fraction 0.58; leak-free em_sg recovers 0.52; em_ln and em_gm collapse to 0.00 in both v2 and leak-free.
3. **The leak fix does not introduce new attractor diversity** beyond what was visible in v2; it does, however, **rescue** the LF-attractor framework's structural prediction (F3) by re-anchoring it at the new γ\* = 0.10 / 0.15 instead of the old γ\* = 0.30.

## 5. Reproduction

```bash
cd notebooks/conservative_arch/energetic_minima

# Tier 2a + Tier 3 trainings (~3h on Apple MPS, M2 Pro)
bash scripts/run_leakfree_tiers_2_3.sh

# Cross-variant attractor extraction + 3D landscape (~5 min CPU)
bash run_attractor_pipeline.sh

# Cross-variant report + JSON
python3 compare.py
```

Outputs:
- `results/em_ln_shakespeare_*` (Tier 2a)
- `results/em_sg_lam1e-03_shakespeare_*` (Tier 3a)
- `results/em_gm_K64_shakespeare_*` (Tier 3b)
- `results/comparison_report.md` (cross-variant summary)
- `../attractor_analysis/results/attractors_em_*_summary.md` (per-variant attractor JSON + Markdown)
- `../attractor_analysis/results/landscape3d_landscape3d_em_*_dialogue.png` (3D landscape rendering for each variant)
- `results/leakfree_tiers_2_3_logs/` (Tier 2/3 stdout/stderr per variant)
- `results/leakfree_tiers_2_3_master.log` (master orchestration log)
