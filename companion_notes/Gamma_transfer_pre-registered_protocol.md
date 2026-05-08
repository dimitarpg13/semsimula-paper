# Pre-Registered Protocol — γ-Transfer Re-Tuning Experiment (E10)

> Pre-registration document, drafted **April 30, 2026**, by Dimitar Gueorguiev with Claude.
> Companion to:
> *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference* (Gueorguiev, 2026), v3.
> Companion experiments:
> - **E4** plain Euler damping sweep: [`notebooks/conservative_arch/damping_sweep/results/RESULTS.md`](../notebooks/conservative_arch/damping_sweep/results/RESULTS.md) — Tiny Shakespeare optimum at $\gamma^{\ast}=0.30$ (val PPL 18.8).
> - **E5** LN-after-step damping sweep: [`notebooks/conservative_arch/ln_damping_sweep/results/RESULTS.md`](../notebooks/conservative_arch/ln_damping_sweep/results/RESULTS.md) — Tiny Shakespeare LN-after-step optimum at $\gamma^{\ast}=0.30$ (val PPL 87.06; clean parabolic shape, $\gamma=0.10$ at 91.33, $\gamma=0.85$ at 93.93).
> - **E9** scale-up de-risking pre-registration: [`SPLM_scaleup_pre-registered_protocol.md`](SPLM_scaleup_pre-registered_protocol.md) — locks the TinyStories scale-up configuration; E9 Phase 1 SPLM arm result at the *transferred* $\gamma=0.30$: val PPL **8.85** at step 8 000 (single seed).

> **Status.** Pre-registered, not yet executed. This document fixes the experimental design, the analysis pipeline, and the decision rule **before** any γ-sweep training is launched. All γ values, step counts, decision thresholds, and stage triggers are locked at pre-registration. The committing commit hash is the timestamp of pre-registration.
>
> **Why pre-register a γ-transfer experiment.** The E9 SPLM scale-up arm was run at the small-scale optimum $\gamma^{\ast}_{\mathrm{E5}}=0.30$ inherited from E4 / E5 (Tiny Shakespeare, ~7 M params, max_len = 256). The transfer of that optimum to TinyStories at the E9 scale (15.75 M params, max_len = 1024, 5 M training tokens) was a *protocol assumption*, not a tested fact. This experiment tests the assumption directly. Locking the γ-grid, the truncation horizon, and the decision rule in advance prevents post-hoc cherry-picking of γ values, especially in the case where γ-transfer fails and a different $\gamma^{\ast}$ is found.

---

## 1. Question

The pre-registered E9 SPLM arm used $\gamma = 0.30$, fixed at small-scale's E4 / E5 optimum.  This is the *γ-transfer hypothesis*: that the damping coefficient that minimises validation PPL on Tiny Shakespeare also minimises it on TinyStories at the E9 scale-up configuration.  E9's pre-registration explicitly notes this assumption and defers the test to a separate experiment — this one.

The question this experiment answers, in two parts:

1. **Q1 (γ-transfer).** Does the E5 optimum $\gamma^{\ast}_{\mathrm{E5}}=0.30$ remain the optimum on TinyStories at the E9 scale-up configuration? Equivalently: where is the TinyStories $\gamma^{\ast}_{\mathrm{TS}}$?
2. **Q2 (quality).** At $\gamma^{\ast}_{\mathrm{TS}}$, is the SPLM em_ln scale-up val PPL materially better (smaller) than E9's $\gamma=0.30$ result of $8.85$?

These questions are *independent* of the E9 outcome (Outcomes A / B / C of the SPLM-vs-matched comparison): even if E9 lands as Outcome A, finding a better $\gamma^{\ast}_{\mathrm{TS}}$ would *strengthen* the SPLM ranking; even if E9 lands as Outcome B or C, finding a better $\gamma^{\ast}_{\mathrm{TS}}$ might invert the verdict.

---

## 2. The single arm and the comparison

This is a **single-arm sweep** over $\gamma$. Only the SPLM em_ln architecture is trained at multiple $\gamma$ values. The **matched-baseline arm is NOT re-trained**: it is $\gamma$-independent and the E9 result(s) provide the comparison anchor.

| Arm | Architecture | Ranges over | Source |
|---|---|---|---|
| **`splm_em_ln_gamma`** (sole arm) | SARF-faithful causal cumulative-mean $\xi$, per-token logfreq mass, second-order semi-implicit damped Euler integrator, LayerNorm projection after every step, $\gamma$ **fixed** at the swept value (no scheduling, no learning) | $\gamma \in \{0.10, 0.30, 0.60\}$ at Stage 1; $\gamma^{\ast}_{\mathrm{TS}}$ at Stages 2 and 3 (and optionally one neighbour). | new `notebooks/conservative_arch/scaleup/gamma_transfer/train_splm_em_ln_gamma_sweep.py`, modes `pilot` / `confirmation` |

**Anchor for the comparison (re-used from E9, not re-run).**

| Quantity | Source | Value |
|---|---|---|
| Matched-baseline (`MatchedGPT`) at TinyStories E9 config, seed 0 | E9 Phase 1 (in progress at the time of this pre-registration; result will be locked when E9 reports its outcome) | TBD (pre-registered as "the value E9 reports for `matched_baseline` at seed 0" — no post-hoc selection) |
| Matched-baseline at seeds 1 + 2 | from E9 Phase 2 if triggered, else run as part of this protocol's Stage 3 (see §5.3) | TBD |
| SPLM em_ln at $\gamma=0.30$, scale-up, seed 0 | E9 Phase 1 SPLM arm | **val PPL 8.85, val loss 2.1799, elapsed 47 102 s** |

### 2.1 Detailed configuration (locked at pre-registration)

Identical to the E9 SPLM arm pre-registration except for the $\gamma$ override and the truncation horizon at Stage 1.

| Quantity | Value |
|---|---|
| Tokenizer | GPT-2 BPE (vocab 50 257) |
| Corpus | TinyStories (HF: `roneneldan/TinyStories`), shard 0, first ~5 M BPE training tokens (E9-locked split) |
| logfreq surprisal cache | reused from E9: `notebooks/conservative_arch/scaleup/results/logfreq_surprisal_tinystories.npy` (no recomputation) |
| Hidden dim $d$ | 256 |
| Layers $L$ | 8 |
| $V_{\theta}$ MLP | `v_hidden=1024`, `v_depth=3`, GELU |
| Per-token mass mode | `logfreq` (frozen unigram surprisal lookup) |
| $\gamma$ schedule | **fixed at the swept value, NO learning** (`learn_mgamma = False`, `init_gamma = γ_swept`, `fixed_gamma = γ_swept`) |
| Tied embeddings | True |
| `max_len` | 1024 |
| Training `block_size` | 512 |
| Batch size | 16 |
| Optimiser | AdamW, $\beta=(0.9, 0.95)$, weight decay $0.01$, gradient clip $1.0$ |
| Peak LR / schedule | $5\times 10^{-4}$, cosine decay |
| Hardware | MacBook (Intel i9, 64 GB), MPS backend, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` |
| Devices for the γ-sweep arms | sequential, same machine, same Python venv as E9 |

**Stage-specific schedule overrides (locked at pre-registration):**

| Quantity | Stage 1 (pilot) | Stages 2 + 3 (confirmation, multi-seed) |
|---|---|---|
| Steps | **4 000** | **8 000** |
| Warmup steps | **200** (5% of total, matching E9's 400 / 8000 = 5 %) | **400** (E9 lock) |
| Eval interval | **200** steps | **400** steps (E9 lock) |
| Eval iters | 40 (E9 lock, unchanged) | 40 |

The Stage-1 truncation horizon is set at **half** of the E9 schedule. The 4000-step PPL is being used as a proxy for the 8000-step PPL; the inference is valid only if the *ranking* of γ values at step 4000 matches the ranking at step 8000. This is the central statistical assumption of the γ-pilot stage and is empirically supported by E4 / E5 where the γ-ordering is established by step 1000 and remains stable through end-of-training.

### 2.2 Token budget

Stage 1: $4000 \cdot 16 \cdot 512 = 32.8$ M tokens / arm $\Rightarrow$ ${\sim}6.5$ epochs over the 5 M-token train split.

Stages 2 + 3 (per arm): $8000 \cdot 16 \cdot 512 = 65.5$ M tokens / arm $\Rightarrow$ ${\sim}13$ epochs over the train split (matches E9 exactly).

---

## 3. Anchor evidence — small-scale γ-curves

Reproduced verbatim from E4 (plain Euler) and E5 (LN-after-step), Tiny Shakespeare, ~7 M params:

| γ | E4 val PPL | E5 val PPL | rank |
|---:|---:|---:|---|
| 0.00 | 22.4 | 113.0 | poor (no damping) |
| 0.10 | 19.7 |  91.3 | second-best |
| **0.30** | **18.8** | **87.1** | **best** (both sweeps) |
| 0.85 | 19.6 |  93.9 | second-best on the high side |
| 2.00 | 21.9 | 103.8 | over-damped |
| 5.00 | div. | 121.9 | over-damped |

E5's curve has a clean parabolic shape with a global minimum at $\gamma=0.30$ and ~6 PPL of curvature between 0.10 and 0.85. The two sweeps agree on the location of $\gamma^{\ast}$ to within the resolution of the grid.

The hypothesis-band for the TinyStories sweep is anchored to this curve: the most likely outcome is that $\gamma^{\ast}_{\mathrm{TS}}$ is also at $0.30$ but the curve may have shifted, especially because (a) the matched-attention task at E9 scale is significantly less "saturated" than at E1 scale, so the inertial term may be less needed, and (b) the longer context length ($T=512$ vs $T=128$) compounds per-step integration error, which generically favours slightly higher damping.

---

## 4. Hypotheses

### Q1 — γ-transfer

Let $\hat{\gamma}^{\ast}_{\mathrm{TS}}$ denote the argmin of validation PPL over the Stage-1 grid (or its boundary-expanded extension; see §5.1.1). Define:

| Hypothesis | Operational form | Theoretical reading |
|---|---|---|
| $H_T$ (γ-transfer holds) | $\hat{\gamma}^{\ast}_{\mathrm{TS}} = 0.30$ | The damping optimum is scale-invariant in the regime explored; the E9 SPLM arm was at the right γ. |
| $H_{¬T}$ (γ-transfer fails — interior) | $\hat{\gamma}^{\ast}_{\mathrm{TS}} \in \{0.10, 0.60\}$ (Stage 1 grid) | Damping optimum shifts with scale; E9 was at a sub-optimal γ. |
| $H_{¬T,\mathrm{boundary}}$ (γ-transfer fails — outside grid) | $\hat{\gamma}^{\ast}_{\mathrm{TS}}$ at the Stage-1 grid boundary, requiring boundary expansion to a value $\not\in\{0.10, 0.30, 0.60\}$ | Damping optimum has shifted outside the bracket inferred from E5 — strong qualitative effect; warrants further analysis. |

### Q2 — Quality of γ\*

Let $P_{\mathrm{TS}}^{(0)}(\gamma)$ denote the val PPL at step 8000 for SPLM em_ln on TinyStories at seed 0. Define:

$$\Delta_{\gamma^{\ast}} = P_{\mathrm{TS}}^{(0)}(\gamma=0.30) - P_{\mathrm{TS}}^{(0)}(\hat{\gamma}^{\ast}_{\mathrm{TS}}) = 8.85 - P_{\mathrm{TS}}^{(0)}(\hat{\gamma}^{\ast}_{\mathrm{TS}}).$$

| Hypothesis | Operational form | Reading |
|---|---|---|
| $H_Q$ (γ\* gives a material PPL improvement) | $\Delta_{\gamma^{\ast}} \ge +0.5$ PPL | Re-tuning produces a defensible PPL improvement worth re-running E9 Phase 2. |
| $H_{¬Q}$ (γ\* gives no material PPL improvement) | $\Delta_{\gamma^{\ast}} \lt +0.5$ PPL | The γ-transfer assumption was harmless in practice; E9 result stands as-is. |

The 0.5-PPL threshold is justified by the per-seed standard deviation of E1 (`splm_em_ln` $\sigma_A=4.44$ rescaled by ~$8.85/95.33 \approx 0.09$ to the new PPL regime gives $\sigma_A^{\mathrm{TS}} \approx 0.4$ as a coarse single-seed estimate). A 0.5-PPL improvement is just above this single-seed noise floor.

---

## 5. Decision rule (locked at pre-registration)

The protocol is a **three-stage adaptive plan**:

### 5.1 Stage 1 — γ-grid pilot

**Grid:** $\gamma \in \{0.10,\ 0.30,\ 0.60\}$, evaluated at **seed 0 only**, for **4 000 steps** each.

**Sequencing:** the three runs are launched sequentially on the same MPS device. Wall-clock per arm is empirically known from E9 to be $\sim 5.96$ s/step × 4000 = **~6.6 h**, total Stage 1 ${\sim}19.8\text{ h}$.

**Outcome:** $\hat{\gamma}^{\ast}_{\mathrm{TS}} := \arg\min_{\gamma \in G} P_{\mathrm{TS}}^{(0, \text{step}=4000)}(\gamma)$, where $G$ is the Stage-1 grid (or its boundary-expanded extension).

**Sanity check:** the curve must show a **clear minimum** (i.e., $P^{(0)}(\hat{\gamma}^{\ast})$ at least 0.5 PPL below both neighbours). If the three values are all within 0.5 PPL of each other, $\hat{\gamma}^{\ast}_{\mathrm{TS}}$ is **declared "indistinguishable from $0.30$"** and the experiment terminates at Stage 1 with conclusion $H_T$ (γ-transfer holds, weakly).

#### 5.1.1 Boundary expansion (locked, contingent)

If $\hat{\gamma}^{\ast}_{\mathrm{TS}} = 0.10$ (low boundary), run **one additional pilot at $\gamma = 0.05$**, 4000 steps, seed 0. Update $\hat{\gamma}^{\ast}_{\mathrm{TS}}$ to the new argmin over $\{0.05, 0.10, 0.30, 0.60\}$.

If $\hat{\gamma}^{\ast}_{\mathrm{TS}} = 0.60$ (high boundary), run **one additional pilot at $\gamma = 0.85$**, 4000 steps, seed 0. Update $\hat{\gamma}^{\ast}_{\mathrm{TS}}$ to the new argmin over $\{0.10, 0.30, 0.60, 0.85\}$.

**No further boundary expansion is permitted.** If even after one boundary expansion $\hat{\gamma}^{\ast}_{\mathrm{TS}}$ is still at the (extended) boundary, the experiment is reported as boundary-saturated ($H_{¬T,\mathrm{boundary}}$) and a separate follow-on protocol is required to extend further.

### 5.2 Stage 2 — confirmation at full schedule

**Trigger:**
- If $\hat{\gamma}^{\ast}_{\mathrm{TS}} = 0.30$ and the Stage-1 sanity check passed: **skip Stage 2 entirely.** The E9 Phase 1 SPLM result (val PPL 8.85) *is* the $\gamma^{\ast}_{\mathrm{TS}}$ confirmation. Conclusion: $H_T$ + $H_{¬Q}$.
- If $\hat{\gamma}^{\ast}_{\mathrm{TS}} \neq 0.30$: run **one full 8000-step training at $\hat{\gamma}^{\ast}_{\mathrm{TS}}$, seed 0**. Wall-clock ${\sim}13.1$ h.

**Output:** $P_{\mathrm{TS}}^{(0)}(\hat{\gamma}^{\ast}_{\mathrm{TS}})$ at step 8000, single seed. Compute $\Delta_{\gamma^{\ast}} = 8.85 - P_{\mathrm{TS}}^{(0)}(\hat{\gamma}^{\ast}_{\mathrm{TS}})$.

**Decision:**
- If $\Delta_{\gamma^{\ast}} \lt 0.5$ PPL: $H_T$ effectively holds (the γ-shift is real but immaterial in PPL). Conclusion: $H_T$ + $H_{¬Q}$. **Stage 3 is NOT triggered.**
- If $\Delta_{\gamma^{\ast}} \ge 0.5$ PPL: γ-transfer fails materially. Conclusion: $H_{¬T}$ + $H_Q$. **Stage 3 is triggered.**

### 5.3 Stage 3 — multi-seed paired band at γ\*

**Trigger:** Stage 2 returns $\Delta_{\gamma^{\ast}} \ge 0.5$ PPL (E9-relative material improvement at γ\*).

**Action:** run SPLM em_ln at $\hat{\gamma}^{\ast}_{\mathrm{TS}}$, **seeds 1 and 2**, 8000 steps each. Wall-clock ${\sim}26.2$ h total.

**Matched-baseline reference for the paired Δ comparison:**
- If E9 Phase 2 was triggered (i.e. E9's Phase 1 returned $|\Delta^{(0)}|\lt20$ PPL), reuse the matched-baseline values at seeds 0 / 1 / 2 from E9 Phase 2 directly. **Do NOT re-train them.**
- If E9 Phase 2 was *not* triggered (E9's Phase 1 returned $|\Delta^{(0)}|\ge 20$ PPL), then run matched-baseline at seeds 1 + 2 as part of Stage 3. Wall-clock ${\sim}13.4$ h additional (matched-baseline runs at $\sim 3.05$ s/step × 8000 = $\sim 6.7$ h per seed).

**Output:** $\overline{\Delta}_{\gamma^{\ast}} = \frac{1}{3}\sum_{s=0}^{2}\bigl(P_B^{(s)} - P_A^{(s)}(\hat{\gamma}^{\ast}_{\mathrm{TS}})\bigr)$, where $P_A^{(s)}(\hat{\gamma}^{\ast}_{\mathrm{TS}})$ is the SPLM em_ln val PPL at seed $s$ and γ = γ\*, and $P_B^{(s)}$ is the matched-baseline val PPL at seed $s$ (reused from E9 if available).

### 5.4 Outcomes (locked)

The realised outcome is one of:

| Outcome | Stage(s) reached | Operational form | Reading |
|---|---|---|---|
| **T0** | Stage 1 only | All three γ values within 0.5 PPL at step 4000, OR $\hat{\gamma}^{\ast}_{\mathrm{TS}}=0.30$. | γ-transfer holds (in the weak sense that 0.30 is indistinguishable from the optimum). E9 stands. |
| **T1** | Stages 1 + 2, $\Delta_{\gamma^{\ast}} \lt 0.5$ | $\hat{\gamma}^{\ast}_{\mathrm{TS}}\neq 0.30$ but PPL gap is <0.5. | γ-transfer technically fails but the E9 number is within the single-seed noise floor of the optimum. E9 stands. |
| **NT-material** | Stages 1 + 2 + 3, $\Delta_{\gamma^{\ast}} \ge 0.5$ | $\hat{\gamma}^{\ast}_{\mathrm{TS}}\neq 0.30$ and PPL improvement $\ge 0.5$ PPL. | γ-transfer fails materially. E9's pre-registered conclusion is *augmented* (not overwritten — E9's Phase-1 result is at γ=0.30, the pre-registered γ; an updated Phase-1 result at γ\* is reported as a separate, complementary finding). |
| **NT-boundary** | Stages 1 + boundary expansion + 2 + 3 | Stage-1 boundary-expanded grid still has γ\* at boundary | γ\* lies outside the explored bracket; reported as a qualitative finding requiring follow-up. |

**No post-hoc threshold adjustment, training-budget adjustment, learning-rate retuning, eval-interval change, or per-seed exclusion is permitted.** The protocol explicitly anticipates each branch; deviation from the locked branch list invalidates the result.

### 5.5 Auxiliary observations (recorded but not part of the decision rule)

The following are NOT part of the locked decision rule but are recorded for the eventual write-up:
- Final train loss per γ-pilot.
- $\gamma$ trajectory across training (degenerate — fixed γ — but recorded for completeness; the field will be constant).
- Wall-clock per arm.
- Loss-curve overlay: all γ-pilot arms on the same axes, train and val.
- Memory peak (per arm).
- For Stages 2 + 3, the per-step gradient norm and energy-drift diagnostics computed in the same convention as E4 / E5 (so the new γ\* run can be added to the E5 PPL-vs-γ + drift-vs-γ figures).

---

## 6. Pre-registered prediction

The author predicts, before any γ-sweep training is run:

### 6.1 Q1 (where is $\gamma^{\ast}_{\mathrm{TS}}$?)

| Sub-outcome | Subjective probability |
|---|---:|
| $\hat{\gamma}^{\ast}_{\mathrm{TS}} = 0.30$ (γ-transfer holds exactly) | $0.45$ |
| $\hat{\gamma}^{\ast}_{\mathrm{TS}} = 0.60$ (slightly higher damping favoured at longer context) | $0.25$ |
| Stage-1 sanity check fails (all within 0.5 PPL) | $0.15$ |
| $\hat{\gamma}^{\ast}_{\mathrm{TS}} = 0.10$ (lower damping favoured) | $0.10$ |
| Boundary expansion needed (γ\* > 0.60 or γ\* < 0.10) | $0.05$ |

**Reasoning.** E5's curve is shallow with a clear minimum at $0.30$, with the $0.60$-side of the parabola slightly less steep than the $0.10$-side. At the longer context length of E9, per-step integration error compounds more, which generically rewards slightly higher damping; this nudges the prior toward $\gamma^{\ast}_{\mathrm{TS}}=0.60$ being the most likely *non-transfer* outcome. The 0.45 mass on exact transfer reflects two arguments pulling in opposite directions: (a) the E5 / E4 curves agree to within their grid resolution, suggesting the optimum is robust to architectural details (LN vs no-LN); (b) but the corpus, model size, and context length are all simultaneously different in E9, so structural transfer cannot be assumed.

### 6.2 Q2 (how much PPL improvement at γ\*?)

Conditional on $H_{¬T}$ (i.e., conditional on $\hat{\gamma}^{\ast}_{\mathrm{TS}}\neq 0.30$):

$$\Delta_{\gamma^{\ast}} \in [+0.2, +0.8]\ \text{PPL},\qquad\text{most-likely }\overline{\Delta}_{\gamma^{\ast}}\approx +0.4\ \text{PPL}.$$

Reasoning: E5's shallowness across the $\{0.10, 0.30, 0.85\}$ bracket implies the curve is locally flat near the optimum; even a non-transferring optimum should be $\le 1$ PPL improvement at the new scale. A strong lift of $\ge 1$ PPL would be surprising and would warrant further investigation.

### 6.3 Joint outcome prediction

| Outcome label (§5.4) | Subjective probability |
|---|---:|
| **T0** (γ-transfer in the weak sense) | $0.60$ |
| **T1** (γ-shift detected, but PPL gap < 0.5) | $0.25$ |
| **NT-material** | $0.10$ |
| **NT-boundary** | $0.05$ |

A confirming outcome (T0 or T1) leaves the E9 conclusion *intact* and is the cleanest possible follow-up to the E9 paper, demonstrating that the γ-transfer assumption was harmless. A falsifying outcome (NT-material) would meaningfully change the SPLM-vs-matched ranking on TinyStories — in SPLM's favour — and would augment the E9 / TMLR write-up with an explicit "γ-tuned SPLM beats matched-attention by ${\sim}1$ PPL more than the γ-transferred SPLM" sentence.

---

## 7. Reporting plan

Regardless of which outcome is realised, the experiment is written up as follows:

- **Code lives at:** `notebooks/conservative_arch/scaleup/gamma_transfer/`. The directory contains:
  - `train_splm_em_ln_gamma_sweep.py` — re-uses the SPLM em_ln scaleup model & data path; adds modes `pilot` (4000-step truncated) and `confirmation` (8000-step full-schedule) with a required `--fixed-gamma` argument.
  - `run_stage1.sh` — driver that launches the three Stage-1 pilots sequentially.
  - `run_stage2.sh` — driver that runs the Stage-2 confirmation at $\hat{\gamma}^{\ast}_{\mathrm{TS}}$ (no-op if $\hat{\gamma}^{\ast}_{\mathrm{TS}}=0.30$).
  - `run_stage3.sh` — driver that runs Stage-3 multi-seed confirmation, including matched-baseline replication if E9 Phase 2 was not triggered.
  - `README.md` — short experiment overview.
  - `results/` — per-stage, per-γ subdirectories with checkpoints (`*.pt`), training logs (`*.jsonl`), summary `.md`, loss-curves, plus the canonical `RESULTS.md` write-up.

- **Headline `RESULTS.md`** in `notebooks/conservative_arch/scaleup/gamma_transfer/results/RESULTS.md` will report:
  - The locked decision rule, the realised outcome (T0 / T1 / NT-material / NT-boundary), the realised $\hat{\gamma}^{\ast}_{\mathrm{TS}}$, and the realised $\Delta_{\gamma^{\ast}}$.
  - Stage-1 PPL-vs-γ table (and parabolic fit if 4 points are available).
  - Stage-2 row: $P_{\mathrm{TS}}^{(0)}(\hat{\gamma}^{\ast})$ at step 8000.
  - Stage-3 paired table (S=3 seeds): per-seed $P_A$, $P_B$, $\Delta^{(s)}$, and $\overline{\Delta}_{\gamma^{\ast}}$.
  - Loss-curve overlay (Stage 1: 3-or-4 γ trajectories truncated at step 4000; Stage 2: γ\* full-length curve overlaid on E9's γ=0.30 curve).
  - Cross-reference back to E4 / E5 / E9.
  - One paragraph on auxiliary observations.

- **Paper-side consequence (TMLR submission, conditional on outcome):**
  - **T0 / T1:** add one short paragraph to the SPLM scale-up section: "We additionally verified γ-transfer from E5 to TinyStories: the small-scale optimum γ\*=0.30 remains optimal at the E9 scale to within $0.5$ PPL of the single-seed noise floor."
  - **NT-material:** report the E9 result at γ=0.30 *and* the γ-tuned result at γ\* as the *primary* TinyStories number, with the γ-transfer-failure being a notable finding worth its own subsection. This lifts SPLM's empirical headline by $\sim 0.5$–$1$ PPL.
  - **NT-boundary:** flag for a separate follow-up γ-sweep at smaller / larger γ; the present γ-transfer experiment reports only what was tested.

- **No post-submission revision of the realised outcome is permitted.** As with E9, downstream experiments (different corpora, different scales) become *new* pre-registered protocols.

---

## 8. Pre-registration metadata

- **Drafted:** 2026-04-30.
- **Pre-registered (committed):** the commit hash that introduces this file is the timestamp of pre-registration.
- **Author:** Dimitar Gueorguiev (with Claude as drafting assistant).
- **No γ-sweep data exists at the time of drafting.** The matched-baseline reference is currently still in training (E9 Phase 1 matched-baseline arm at step ~2100/8000 as of 15:00 EDT 2026-04-30). The SPLM γ=0.30 anchor (8.85 PPL) is the only post-hoc-fixed quantity, and it is fixed *before* the γ-sweep is launched.
- **Companion implementation files** (`notebooks/conservative_arch/scaleup/gamma_transfer/`) will be added in a *separate* commit after this protocol is reviewed and approved.
- **Stage 1 launch trigger:** Stage 1 will be launched only after E9 Phase 1 (matched-baseline arm) completes and its outcome is recorded. If E9 Phase 2 is triggered, Stage 1 of E10 may launch in parallel with E9 Phase 2 only if the user confirms this in writing; otherwise E10 waits for E9 Phase 2 to complete (so no MPS-device contention on the same machine).

---

## Appendix A — Wall-clock budget summary

Estimated, based on E9 Phase 1 empirical rates (5.96 s/step SPLM, 3.05 s/step matched-baseline at the locked configuration):

| Phase | Description | Hours | Cumulative |
|---|---|---:|---:|
| Stage 1 (3 γ values × 4000 steps) | γ-grid pilot, seed 0 | ~19.8 | 19.8 |
| (Stage 1a, contingent: 1 boundary expansion × 4000 steps) | only if γ\* at boundary | +6.6 | 26.4 |
| Stage 2 (γ\* at 8000 steps × 1 seed) | confirmation at full schedule, seed 0 | +13.1 | 39.5 |
| Stage 3 (γ\* at 8000 steps × 2 seeds) | multi-seed band at γ\* | +26.2 | 65.7 |
| (Stage 3 contingent: matched-baseline at seeds 1 + 2 × 8000 steps) | only if E9 Phase 2 not triggered | +13.4 | 79.1 |

Best case (T0, no expansion, no Stage 2/3): **~19.8 h ≈ 1 day**.

Modal case (T1, expansion, full Stages 2 + 3, E9 Phase 2 already done): **~65.7 h ≈ 2.7 days**.

Worst case (NT-boundary, full expansion, Stages 2 + 3, E9 Phase 2 not done): **~79.1 h ≈ 3.3 days**.

The user-facing "around 3 days" estimate is the modal-to-worst-case wall-clock budget.
