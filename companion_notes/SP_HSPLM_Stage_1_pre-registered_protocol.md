# Pre-Registered Protocol — SP-HSPLM Stage 1: Leak-fixed Per-Token Class B/C Rerun

> **Status:** Pre-registered, not yet executed.
> **Date locked:** 2026-05-15.
> **Author:** Dimitar Gueorguiev (with Claude as scribe / sanity-checker).
> **Pre-registration commit:** to be filled in at lock-in commit.
> **Companion documents:**
> - [`Scalar_Potential_based_Helmholtz_Architecture_v2.md`](./Scalar_Potential_based_Helmholtz_Architecture_v2.md) — the SP-HSPLM design doc that motivates this Stage 1 rerun (section 9.1).
> - [`SP_HSPLM_Stage_0_Literature_Check.md`](./SP_HSPLM_Stage_0_Literature_Check.md) — the originality assessment that recommends Stage 1 as the cheapest entry point.
> - [`Scalar_Potential_based_Helmholtz_Architecture.md`](./Scalar_Potential_based_Helmholtz_Architecture.md) — the predecessor Q9(d) Helmholtz hybrid (uses attention).
> - [`SPLM_scaleup_pre-registered_protocol.md`](./SPLM_scaleup_pre-registered_protocol.md) — the E9 scale-up protocol that establishes the leak-fixed SPLM em\_ln baseline this Stage 1 builds on.
> - [`PARF-SPLM_Path_Forward_and_Experiments.md`](./PARF-SPLM_Path_Forward_and_Experiments.md) — the P10 ladder; P10g is the SparsePARFLM scale-up against which Stage 2 will eventually compare.
> - [`Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](./Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md) — the leak-fix invariant. Every Stage 1 cell must satisfy it.
> **Status of executed write-up:** the executed write-up will live alongside the implementation in the main repo (under the non-conservative SPLM results directory) once complete.

---

## 1. One-paragraph motivation

The v3 paper (section 15.5) closed the autonomous Helmholtz menu with five negative-result fitting experiments E1–E5 on **GPT-2 hidden-state trajectories**: scalar-potential sweeps, a linear Helmholtz position-coupled solenoidal augmentation, and a velocity-coupled electromagnetic-analogue gauge sweep — each of which tied or undershot the static-null floor on held-out data. The companion claim — repeated at the close of the abstract — is that **per-token non-conservative additions cannot close the SPLM-vs-attention gap on TinyStories**, and that closing the residual *requires a categorical change* — token-token routing comparable to attention. SP-HSPLM (Q9(e), see [`Scalar_Potential_based_Helmholtz_Architecture_v2.md`](./Scalar_Potential_based_Helmholtz_Architecture_v2.md)) takes that prescription and replaces attention with a **causal pair-interaction skew/solenoidal field**. Before paying the engineering cost of Stage 2, however, we must establish that the per-token negative result *actually reproduces under the leak-fixed v3 codebase*. The original SPLM training experiments were leak-conditioned on the SPLM side; their *fitting-on-GPT-2* counterparts (the E1–E5 of section 15.5) are leak-immune by construction (the probe does not re-run the buggy integrator), but **the matched claim that a per-token Class-B/C augmentation, trained end-to-end inside SPLM, does not close the gap on TinyStories has never been measured on the leak-fixed code**. Stage 1 is that measurement.

---

## 2. Hypotheses

Stated as falsifiable predictions, against which the executed run will be compared in the section 6 decision matrix.

**H1 (negative result reproduces).** Every Stage 1 cell ties the leak-fixed SPLM em\_ln baseline within a 2-sigma seed-variance band. Formally, for each cell `c` in `{E1-fix, E2-fix, E3-fix, E4-fix, E5-fix}`,
$$\lvert \mathrm{PPL}(c) - \mathrm{PPL}(\mathrm{splm\_em\_ln, leakfree}) \rvert \lt 2\sigma_{\mathrm{seed}},$$
with $\sigma_{\mathrm{seed}}$ taken from the multi-seed E1 reference (`splm_em_ln`, $\sigma \approx 1.8$ PPL). This corresponds to the v3 abstract's "per-token additions cannot close the gap" claim under leak-fixed training.

**H2 (no negative outcome).** No Stage 1 cell *worsens* the leak-fixed SPLM em\_ln baseline by more than $5\sigma_{\mathrm{seed}}$ ($\approx 9$ PPL). A worsening larger than this would indicate either a training-stability failure or an architectural pathology (e.g., velocity-coupled feedback divergence as flagged in the v3 paper E5 with $s = 1$, section 15.5), and must be diagnosed before any reading is published.

**H3 (causal-leak invariant).** The standalone causal-leak probe in the main repo returns leak floor $\le 10^{-6}$ on every Stage 1 cell at every checkpoint. Any non-zero leak floor invalidates the cell and triggers a code review of the integrator before re-running.

**H4 (Jacobian-symmetry signature, soft prediction).** For Class-B/C cells with a non-zero non-conservative coefficient at convergence, the velocity-aware Jacobian-symmetry probe (section 15.7 of the v3 paper, run on the corresponding checkpoint trajectory) returns an asymmetric Jacobian whose Frobenius norm is bounded above by the Frobenius norm of the cell's parametrised non-conservative coefficient at convergence. This is a soft prediction — it confirms the cell is *expressing* the non-conservative force rather than driving its coefficient to zero. The shape of this prediction is the same as the v3 paper section 15.7 finding for SPLM (symmetric Jacobian) but inverted in sign for any cell where the non-conservative term is doing real work.

---

## 3. The five cells

Each cell augments the SPLM Euler-Lagrange flow with one specific per-token non-conservative force. The integrator step changes from
$$v_{\ell+1} = (v_\ell + \mathrm{d}t \cdot f_\ell / m) / (1 + \mathrm{d}t \cdot \gamma), \qquad h_{\ell+1} = h_\ell + \mathrm{d}t \cdot v_{\ell+1}$$
to
$$v_{\ell+1} = (v_\ell + \mathrm{d}t \cdot (f_\ell + g_\ell) / m) / (1 + \mathrm{d}t \cdot \gamma), \qquad h_{\ell+1} = h_\ell + \mathrm{d}t \cdot v_{\ell+1}$$
where $f_\ell = -\nabla\_h V_\theta(\xi, h_\ell)$ is the conservative force (unchanged) and $g_\ell$ is the cell-specific per-token non-conservative force defined below. Layer normalisation (em\_ln configuration) is applied after the step in every cell.

### 3.1 E1-fix: per-token constant skew (Class B, gyroscopic, simplest)

A learned, position-independent skew matrix $\Omega \in \mathbb{R}^{d \times d}$ couples to the token velocity:
$$g^{\mathrm{E1}}_\ell = \Omega \dot{h}_\ell, \qquad \Omega = J - J^\top, \qquad J \in \mathbb{R}^{d \times d} \ \text{learned}.$$
**Parameters:** $d^2$ (one full-rank $J$). Skew enforced structurally: only $J - J^\top$ is used in the integrator, so the trace of $\Omega$ is zero and the term is divergence-free in $h$.

### 3.2 E2-fix: per-token affine-rank-1 skew (Class B, h-dependent)

The skew matrix becomes a rank-1 outer-product expression of the local hidden state:
$$g^{\mathrm{E2}}_\ell = \Omega(h_\ell) \dot{h}_\ell, \qquad \Omega(h) = u h^\top - h u^\top, \qquad u \in \mathbb{R}^d \ \text{learned}.$$
**Parameters:** $d$ (one $u$ vector). The skew is automatic by construction; the trace of $\Omega(h)$ is zero for every $h$.

### 3.3 E3-fix: per-token affine-rank-2 skew (Class B, h-dependent, low-rank, r=2)

Generalising E2 to learnable rank $r$:
$$g^{\mathrm{E3}}_\ell = \Omega(h_\ell) \dot{h}_\ell, \qquad \Omega(h) = U H(h)^\top - H(h) U^\top,$$
where $U \in \mathbb{R}^{d \times r}$ is learned and $H(h) \in \mathbb{R}^{d \times r}$ is a learned linear projection $H(h) = W h$ with $W \in \mathbb{R}^{(d \cdot r) \times d}$ reshaped to $d \times r$. **Parameters:** $d \cdot r + d^2 \cdot r$. **Locked $r = 2$** (the minimal "affine-rank-2" extension of E2 — directly corresponds to the v3 paper section 15.5 E5b "rank-2 affine" gauge). E3 forms a rank-ablation pair with E5 below.

### 3.4 E4-fix: per-token solenoidal field (Class C, position-only, low-rank)

The position-only solenoidal force is the per-token analogue of the SP-HSPLM C-block but **without pair structure** — it is the cleanest test of whether a per-token solenoidal field is what's missing:
$$g^{\mathrm{E4}}_\ell = (J_+(h_\ell) - J_+(h_\ell)^\top) \rho(h_\ell),$$
where $J_+(h) = U V(h)^\top$ with $U \in \mathbb{R}^{d \times r}$ learned and $V(h) = W h$ for $W \in \mathbb{R}^{(d \cdot r) \times d}$ reshaped, and $\rho(h)$ is a small learned MLP `Linear(d, h_rho) -> GELU -> Linear(h_rho, d)` with $h_\rho = 64$. The $J_+ - J_+^\top$ construction makes the term divergence-free in $h$ at every $h_\ell$. **Parameters:** $d \cdot r + d^2 \cdot r + (d \cdot h_\rho + h_\rho \cdot d)$. Locked $r = 4$, $h_\rho = 64$.

### 3.5 E5-fix: per-token gauge (Class B, h-dependent, low-rank, r=4)

The richer rank-ablation companion of E3, with the same parameterisation but at higher rank:
$$g^{\mathrm{E5}}_\ell = \Omega(h_\ell) \dot{h}_\ell, \qquad \Omega(h) = U V(h)^\top - V(h) U^\top,$$
with $U \in \mathbb{R}^{d \times r}$ learned and $V(h) = W h$ for $W \in \mathbb{R}^{(d \cdot r) \times d}$ reshaped. **Parameters:** $d \cdot r + d^2 \cdot r$. **Locked $r = 4$** (twice the rank of E3). The pair E3 (r=2) / E5 (r=4) is a clean rank ablation: if E5 closes the gap and E3 does not, the v3 paper's reading "rank-2 is enough" is wrong; if both tie Cell 0, the rank ablation closes the rank axis as well as the form axis.

### 3.6 Coverage of the v3 paper E1–E5 menu

The Stage 1 cells do not cover Class D (dissipative). This is intentional — the v3 paper's Class-D term ($\gamma$, the Rayleigh damping) is *already* in the SPLM baseline as the learned `raw_gamma`. Adding a per-token Class-D augmentation is the [`E4_damping_sweep_pre-registered_protocol.md`](./E4_damping_sweep_pre-registered_protocol.md) experiment, already executed and with results in [`E4_sweep_results_and_discussion.md`](./E4_sweep_results_and_discussion.md). Stage 1 therefore covers Class B (E1–E3, E5) and Class C (E4); Class D is excluded as already-covered.

---

## 4. Locked configuration

### 4.1 Hyperparameters (locked at pre-registration)

All five cells share the following configuration. Anything not listed inherits from the SPLM em\_ln scale-up training script in the main repo, mode `scaleup`.

| Quantity | Value |
|---|---|
| Tokenizer | GPT-2 BPE (vocab 50 257) |
| Corpus | TinyStories (HF: `roneneldan/TinyStories`), shard 0, ~5 M training BPE tokens, canonical validation shard |
| Hidden dim `d` | 256 |
| Layers `L` (integration steps) | 8 |
| Per-token mass mode | `logfreq` (frozen, computed from TinyStories train) |
| `V_theta` MLP | `v_hidden = 1024`, `v_depth = 3`, GELU |
| `init_gamma` | 1.0 with `learn_mgamma = True` (matches E9 scale-up; gamma is **learned**, not fixed) |
| Tied embeddings | True |
| `max_len` | 1024 |
| `block_size` | 512 |
| Steps | **16 000** (matched to P10g for forward compatibility with Stage 2) |
| Batch size | 16 |
| Optimiser | AdamW, beta = (0.9, 0.95), weight decay 0.01, grad clip 1.0 |
| LR schedule | 5e-4 peak, cosine decay, 800 warmup steps |
| Eval | `eval_iters = 40` every 800 steps; final eval at step 16 000 |
| Causal-force flag | `cfg.causal_force = True` (leak fix on, mandatory) |
| TF32 | **disabled** (`torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`) for autograd numerical stability |
| Seeds | 1 seed per cell first cut (seed = 0); 3 seeds for any cell that breaks the floor by more than 2 sigma |
| Hardware | H100 or A100 single-GPU per cell, runs sequential or parallel as available |

The token budget per cell is therefore $16000 \cdot 16 \cdot 512 = 131$ M token-passes, double the E9 scale-up budget. This is consistent with the P10 ladder (which also runs 16k steps) and ensures Stage 1 cells can be compared directly against Stage 2 SP-HSPLM cells without a budget asterisk.

### 4.2 Baselines (already executed; numbers re-cited here)

| Baseline | val PPL | Source |
|---|---:|---|
| `splm_em_ln` leak-free, K = 1, 4000 steps | ~30 | section 15 of paper v3, R6 ladder |
| `splm_em_ln` leak-free, K = 4 K-EMA, 4000 steps | 14.78 | section 15 of paper v3, R6.h.1 |
| `matched_baseline` attention, 8000 steps | ~8 | section 15 of paper v3 |
| `splm_em_ln` leak-free, 8000 steps | (executed; number to be retrieved at run time) | E9 scale-up |
| **`splm_em_ln` leak-free, 16 000 steps (Stage 1 baseline)** | **must be executed** | **new — Cell 0 of this protocol** |

The 16k-step `splm_em_ln` leak-free run does not exist yet at the locked config. To make Stage 1 a clean apples-to-apples test, **a Cell-0 baseline run is added to the protocol**: identical config to the five cells, with `g_l = 0` (no non-conservative term). All five Stage 1 cells are compared against Cell 0, not against the existing 8k em\_ln number.

### 4.3 Causal-leak invariant (mandatory across all cells)

Each cell's training loop must run the causal-leak probe (the standalone probe utility in the main repo) at three checkpoints: step 1 (initialisation), step 8000 (mid-training), and step 16 000 (final). The leak floor at every checkpoint must be at or below `1e-6`. Any cell that violates this invariant is invalidated and re-run after a code review.

The invariant is preserved by construction in every cell because the non-conservative force $g_\ell$ in cells E1–E5 is computed from the **local** $h_\ell$ and $\dot{h}_\ell$ at each layer — no information from $h_s$ for $s \neq t$ enters $g_\ell$ at token $t$, so the existing `.detach()` points in the integrator are not bypassed. This is the *structural* reason Class B/C per-token forces preserve the leak fix, and it is a key contrast with the upcoming SP-HSPLM Stage 2 cells, which will need *additional* `.detach()` points to preserve the invariant under pair coupling.

---

## 5. Outputs per cell

Each cell, after training, produces a named-tag output bundle in the main repo (under the non-conservative SPLM results directory):
```
<results_root>/<tag>/
  splm_<cell>_summary.md
  splm_<cell>_training_log.jsonl
  splm_<cell>_loss_curve.png
  splm_<cell>_ckpt_latest.pt

  causal_probe.json                # leak floors at steps 1, 8000, 16000
  jacobian_symmetry.json           # velocity-aware Jacobian-symmetry probe at step 16000
  nonconservative_norms.json       # || g_l || statistics across layers and steps
```
with `<tag>` in `{e0_baseline, e1_const_skew, e2_affine_rank1, e3_lowrank_rank2, e4_solenoidal_rank4, e5_lowrank_rank4}`. The `nonconservative_norms.json` reports the mean and standard deviation of $\lVert g_\ell \rVert_2$ at every layer and every 800-step eval interval — this is the diagnostic for H4 and for detecting "non-conservative coefficient went to zero" degenerate solutions.

---

## 6. Statistical and decision protocol

### 6.1 PPL — single-seed first cut, multi-seed if borderline

We train **one seed (= 0)** per cell. If the resulting PPL is within $2\sigma_{\mathrm{seed}}$ of Cell 0 (the new 16k em\_ln baseline), we accept H1 for that cell and stop. If the PPL beats Cell 0 by more than $2\sigma_{\mathrm{seed}}$, we run two additional seeds (1, 2) on that cell and on Cell 0 before deciding.

The reference seed-variance is $\sigma_{\mathrm{seed}} \approx 1.8$ PPL from the E1 multi-seed `splm_em_ln` row. Under the longer 16k schedule the seed variance is expected to be smaller, not larger, so the 2-sigma threshold of $\Delta \approx 3.6$ PPL is conservative.

### 6.2 Decision matrix

For each cell, three independent verdicts are reported.

| Diagnostic | Verdict |
|---|---|
| PPL vs Cell 0 | "ties" if `\|delta_PPL\| < 2 sigma_seed`; "improves" if delta_PPL <= -2 sigma_seed; "worsens" if delta_PPL >= +2 sigma_seed |
| Causal probe | "leak-clean" if all three checkpoint floors are at or below 1e-6; "leak-fail" otherwise |
| Non-conservative norm | "active" if mean `\|g_l\| / \|f_l\|` at step 16000 is at least 0.05 across all layers; "collapsed" otherwise |

The headline reading of Stage 1 is the *joint* outcome across the three verdicts and across the five cells.

### 6.3 Hard go / no-go gates for Stage 2

- **Outcome ALPHA — negative result reproduces (expected):** every cell ties Cell 0, `causal probe` is leak-clean, and the non-conservative norms are non-trivial (the cells *are* expressing the force; they just fail to translate it into a quality improvement). This is the v3 paper's expected outcome and the green light for Stage 2 (SP-HSPLM C-block).
- **Outcome BETA — surprise positive on E4-fix (per-token solenoidal):** if the leak fix flips the result and a per-token solenoidal cell beats Cell 0 by more than 2 sigma, **pause Stage 2** and re-design it around the per-token solenoidal cell rather than the pair-skew cell. The SP-HSPLM construction still applies, but the per-token term may be enough on its own and Stage 2's pair structure becomes optional rather than load-bearing.
- **Outcome GAMMA — surprise positive on E1, E2, E3, or E5:** if a per-token Class-B cell (rather than a Class-C cell) closes the gap, the v3 paper's reading of "Class C is the missing ingredient" is wrong. The §17.3 discussion needs revision before Stage 2; SP-HSPLM remains a candidate but the motivation tree changes.
- **Outcome DELTA — non-conservative collapse (degenerate):** if every cell's $\lVert g_\ell\rVert$ collapses to zero by step 16 000, the cells trained but learned to set the non-conservative coefficient to zero — the v3 paper's "v3 abstract closing sentence" is reproduced under leak-fixed training, and Stage 2 is justified.
- **Outcome EPSILON — instability:** if a cell diverges (loss spike, NaN, or causal-probe floor violation), the cell is invalidated and the issue diagnosed. This is the analogue of the v3 paper's E5 $s = 1$ velocity-coupled divergence finding.

---

## 7. What Stage 1 can show

Three qualitative outcomes are possible at the level of the entire grid.

**Outcome A — clean confirmation of the v3 reading.** All five cells tie Cell 0 within 2 sigma; the non-conservative coefficients are non-zero but the quality result is unchanged. This validates the v3 abstract's framing — per-token non-conservative additions are routing-poor — and motivates Stage 2 (SP-HSPLM, pair-skew force) as the principled next step.

**Outcome B — surprise on Class C only.** E4-fix beats Cell 0 by 2 sigma or more; the four Class-B cells tie. This would indicate that the per-token solenoidal field is the missing ingredient and Stage 2's pair-skew construction is a strict generalisation — the prior on Stage 2 working improves, but the per-token solenoidal cell becomes the new architectural baseline to beat.

**Outcome C — surprise on Class B.** A Class-B cell beats Cell 0 by 2 sigma or more. This contradicts the v3 paper's reading that "the symmetric, non-Hessian stretching dominates" and the velocity-coupled gauge "collapses to zero". The §17.3 hybrid programme needs to be rewritten before Stage 2 launches.

In every case the verdict is what the data says, with the decision rule pre-committed in this protocol.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Velocity-coupled feedback divergence (the v3 paper E5 with s = 1 pathology) | Initialise the non-conservative parameters small (std 0.002) so the initial ratio of `\|g_l\|` to `\|f_l\|` is at most 0.05; cosine warmup the LR over 800 steps; grad-clip at 1.0 (already in the E9 config). If a cell diverges, the protocol's Outcome EPSILON captures this case. |
| Causal-leak regression | Mandatory probe at three checkpoints (section 4.3). The structural argument (section 4.3 paragraph 2) makes a leak structurally impossible for Class B/C per-token cells; the probe enforces it. |
| Non-conservative coefficient collapse to zero (degenerate solution) | The `nonconservative_norms.json` output captures this; Outcome DELTA in the decision matrix interprets it as a *positive* confirmation of the v3 reading rather than a failure. |
| Compute over-spend | The 5 cells (plus Cell 0 baseline) at 16k steps each on H100 is ~6 GPU-days at the existing P10g per-cell wall time. The full multi-seed escalation (3 seeds for any cell that breaks the floor) is at most 3 cells extra at 2 seeds each, or 12 GPU-days worst case. |
| Over-counting parameters | Each cell's non-conservative force adds at most d^2*r + d*r + (E4 only: 2*d*h_rho) parameters at d = 256, r in {2, 4}, h_rho = 64. Per-cell counts: E1 ~66 k, E2 ~256, E3 (r=2) ~131 k, E4 (r=4) ~296 k, E5 (r=4) ~263 k. Worst case ~1.9% of the 15.75 M-parameter SPLM em\_ln baseline. The PPL comparison is therefore not biased by parameter count at single-seed resolution; the protocol still reports each cell's parameter count alongside its PPL for transparency. |

---

## 9. Schedule and cost estimate

| Phase | Work | Calendar | GPU-days |
|---|---|---|---:|
| A | Pre-registration (this document); peer-read by collaborator(s); commit-lock | 1 day | 0 |
| B | Implementation in the main repo: a non-conservative SPLM model module with the 5 force terms; modified integrator; unit tests; causal\_probe verification on a 200-step smoke run per cell | 3-5 days | 0.2 (smoke) |
| C | Scale-up training script with `--cell` flag (in the main repo); H100/A100 notebooks for each cell | 1-2 days | 0 |
| D | Run the 6 cells (Cell 0 + E1-fix through E5-fix), 16k steps, 1 seed | 1 day calendar (parallel) | 6 |
| E | Aggregation, RESULTS.md, decision-matrix evaluation; if any cell breaks the floor, run the 2 additional seeds | 1-2 days plus 0-12 GPU-days | 0-12 |
| F | Write up; cite in section 17 of paper v4 / v5 if the result changes the framing | 1-2 days | 0 |

**Total:** 7-14 calendar days; 6-18 GPU-days. Cheap relative to Stage 2.

---

## 10. References

### Primary references (this repository)

- [`Scalar_Potential_based_Helmholtz_Architecture_v2.md`](./Scalar_Potential_based_Helmholtz_Architecture_v2.md) — section 9.1 specifies the Stage 1 cells; this protocol locks the implementation and decision rule.
- [`SP_HSPLM_Stage_0_Literature_Check.md`](./SP_HSPLM_Stage_0_Literature_Check.md) — the originality assessment that recommends Stage 1.
- [`SPLM_scaleup_pre-registered_protocol.md`](./SPLM_scaleup_pre-registered_protocol.md) — E9 scale-up; the SPLM em\_ln baseline structure this Stage 1 inherits.
- [`E4_damping_sweep_pre-registered_protocol.md`](./E4_damping_sweep_pre-registered_protocol.md) — Class-D coverage (excluded from Stage 1 because already executed).
- [`Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](./Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md) — leak-fix invariant, mandatory across all cells.
- [`PARF-SPLM_Path_Forward_and_Experiments.md`](./PARF-SPLM_Path_Forward_and_Experiments.md) — P10g matching; Stage 2 will compare to P10g, so Stage 1 matches the same training schedule.

### Paper reference (this conversation's reading)

- *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference*, Gueorguiev (2026), v3, section 15.5 ("five-negatives") and section 17.3 (Q9 hybrid programme). The Stage 1 cells are the leak-fixed reruns of the per-token Class B/C analogues of the section 15.5 fitting experiments.

---

*Last updated: 15 May 2026. Pre-registered, awaiting commit-lock and Phase B implementation.*
