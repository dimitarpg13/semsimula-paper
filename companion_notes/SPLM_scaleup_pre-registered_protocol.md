# Pre-Registered Protocol — SPLM Scale-Up De-Risking Experiment (E9)

> Pre-registration document, drafted **April 29, 2026**, by Dimitar Gueorguiev with Claude.
> Companion to:
> *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference* (Gueorguiev, 2026), v3.
> Companion experiments:
> - E1 multi-seed (small-scale matched-parameter comparison): [`notebooks/conservative_arch/multi_seed/results/E1_shakespeare/E1_report.md`](../notebooks/conservative_arch/multi_seed/results/E1_shakespeare/E1_report.md) — Outcome: `splm_em_ln` 95.33 ± 4.44 PPL beats `matched_baseline` 149.80 ± 7.21 PPL by $\overline{\Delta} = +54.47$ PPL across $S=5$ seeds (Welch $t=14.4$, $p\lt10^{-5}$).
> - E5 LN-after-step damping sweep: [`notebooks/conservative_arch/ln_damping_sweep/results/RESULTS.md`](../notebooks/conservative_arch/ln_damping_sweep/results/RESULTS.md) — establishes $\gamma^{\ast} \approx 0.30$ at small scale.
> - SPLM-1 first-order ablation: [`notebooks/conservative_arch/first_order_ablation/results/RESULTS.md`](../notebooks/conservative_arch/first_order_ablation/results/RESULTS.md) — Outcome A confirms training-time value of the inertial term.

> **Status.** Pre-registered, not yet executed. This document fixes the experimental design, the analysis pipeline, and the decision rule **before** any training launches. The committing commit hash is the timestamp of pre-registration.
>
> **Why pre-register a "de-risking" experiment.** The express purpose of this experiment is to find out whether the small-scale (~7–8 M, Tiny Shakespeare) SPLM-vs-matched-attention quality gap survives a meaningful scale-up. Outcomes B (gap shrinks) or C (gap inverts) directly affect the paper's headline framing for the TMLR submission. Locking the decision rule, threshold, and reporting plan in advance prevents post-hoc reframing of the result regardless of which direction it goes.

---

## 1. Question

The paper's empirical headline at small scale is:

> At matched-parameter / matched-compute on Tiny Shakespeare, an SPLM with LayerNorm-after-step (`splm_em_ln`) achieves validation perplexity $95.33 \pm 4.44$ vs. a vanilla GPT-2-style matched baseline at $149.80 \pm 7.21$ — a $+54$ PPL improvement at $11.5\%$ fewer parameters (E1 multi-seed result).

A reasonable sceptic — and any reviewer at a peer-reviewed ML venue — will ask whether this gap is an artefact of the small-corpus / small-model regime. Three concrete generalisation axes are *simultaneously* small in the existing experiments:

1. **Model size.** SPLM 7.1 M params, matched 8.0 M params. Below typical "interesting LM" thresholds.
2. **Corpus size.** Tiny Shakespeare ~321 k BPE training tokens. Two orders of magnitude below standard LM benchmarks (WikiText-103 ~100 M, enwik8 ~30 M).
3. **Context length.** `max_len = 256`, training `block_size = 128`. SPLM's headline architectural claim — constant per-token decode cost via streaming-$\xi$ — is structurally most relevant at long $T$, which is barely exercised at $T=128$.

This experiment scales all three axes simultaneously at a controlled multiple (2.2 × params, 16 × tokens, 4 × context) and asks **whether the matched-parameter PPL gap of E1 survives the scale-up**.

The question is *not* "what is SPLM's absolute PPL on TinyStories?" — that depends on training budget and is less informative for a relative comparison. The question is "**does the paper's headline ranking (SPLM beats matched-attention) hold at the next level of scale, on the same hardware, in the same training framework?**"

---

## 2. The two arms

| Family | Architecture | Layer update at the integrator | Param count | Source |
|---|---|---|---:|---|
| **Arm A — `splm_em_ln`** (SPLM with LayerNorm-after-step, $\gamma$ learned, the E1/E5 winner) | SARF-faithful causal cumulative-mean $\xi$, per-token logfreq mass, second-order semi-implicit damped Euler integrator, LayerNorm projection after every step | $v_{l+1} = (v_l + \mathrm{d}t\cdot f/m)/(1 + \mathrm{d}t \cdot \gamma)$, then $h_{l+1} = \mathrm{LN}(h_l + \mathrm{d}t\cdot v_{l+1})$ | **15.75 M** | new `notebooks/conservative_arch/scaleup/train_splm_em_ln_scaleup.py`, mode `scaleup` |
| **Arm B — `matched_baseline`** (vanilla pre-LN GPT-2 decoder, the same baseline used in E1 / E8 Phase 1) | Multi-head causal self-attention, GELU MLP at $4\times$ width, tied embeddings | standard pre-LN transformer block | **19.45 M** | new `notebooks/conservative_arch/scaleup/train_matched_baseline_scaleup.py`, mode `scaleup` |

Param-count ratio: arm B / arm A = $1.235$, slightly larger than E1's small-scale ratio of $8.05 / 7.12 = 1.131$ but in the same regime. The matched-attention arm therefore has *more* parameters than the SPLM arm; any quality gap in SPLM's favour cannot be attributed to a parameter-count advantage.

### 2.1 Detailed configuration (locked at pre-registration)

| Quantity | Value (both arms) |
|---|---|
| Tokenizer | GPT-2 BPE (vocab 50 257) |
| Corpus | TinyStories (HF: `roneneldan/TinyStories`), shard 0 only — first ~5 M BPE tokens of the train split, plus the canonical validation shard |
| Hidden dim $d$ | 256 |
| Layers $L$ / $n_{\mathrm{layer}}$ | 8 |
| Attention heads (arm B) | 8 (head_dim = 32) |
| $V_{\theta}$ MLP (arm A) | `v_hidden = 1024`, `v_depth = 3`, GELU |
| Per-token mass mode (arm A) | `logfreq` (frozen unigram surprisal lookup, computed from the same TinyStories train split as the corpus, with add-one Laplace smoothing) |
| $\gamma$ initialisation (arm A) | `init_gamma = 1.0`, `learn_mgamma = True` (the $\gamma$ value is *learned*, not fixed at $\gamma^{\ast}=0.30$) — same setting as E1 `splm_em_ln` |
| Tied embeddings | True (both arms) |
| `max_len` (positional capacity) | **1024** |
| Training `block_size` | **512** |
| Steps | **8000** |
| Batch size | **16** |
| Optimiser | AdamW, $\beta = (0.9, 0.95)$, weight decay $0.01$, gradient clip $1.0$ |
| Learning rate / schedule | $5 \times 10^{-4}$ peak, cosine decay, $400$ warmup steps |
| Eval | `eval_iters = 40` every $400$ steps; final eval at step $8000$ |
| Seeds | **see §5 — adaptive 1-seed / 3-seed plan** |
| Hardware | MacBook (Intel i9, 64 GB), MPS backend |
| Devices for the two arms | sequential, same machine, same Python venv |

Everything not in the table — initialisation scheme, loss function (cross-entropy on next-token, ignore-index `-100`), tokeniser, batch sampler RNG protocol — is identical to the E1 `multi_seed/multi_seed_runner.py` recipe.

### 2.2 Token budget and Chinchilla-style ratio

At the locked configuration, total token-passes $= 8000\cdot 16 \cdot 512 = 65.5\text{ M}$, which is

$$\frac{\text{tokens seen}}{\text{params}} \in \left\{\frac{65.5\text{M}}{15.75\text{M}},\ \frac{65.5\text{M}}{19.45\text{M}}\right\} \approx \{4.2,3.4\} \text{ tokens / param}.$$

This is **deliberately below** the Chinchilla optimal of $\sim 20$ tokens / param. **The protocol is intentionally training-budget-limited rather than data-limited.** This is appropriate for a relative comparison (both arms see the same token budget; the comparison is unbiased), but it means absolute PPL numbers are *not* in the high-quality / well-converged regime. Reporting will explicitly acknowledge this.

The corpus itself contains $\sim 5$ M training tokens, so each token is sampled ${\sim}13$ times across the 8 000 steps, well above the "single epoch" boundary.

---

## 3. The comparison anchor

The reference point is the E1 multi-seed result at small scale:

| Quantity | `splm_em_ln` (5 seeds, mean ± std) | `matched_baseline` (5 seeds, mean ± std) | $\Delta = $ matched $-$ SPLM |
|---|---:|---:|---:|
| Final val PPL (E1, Tiny Shakespeare, 4 000 steps, 7.1 M / 8.0 M params, max_len=256) | $95.33 \pm 4.44$ | $149.80 \pm 7.21$ | **$+54.47$ PPL** in SPLM's favour |

We do **not** re-evaluate the E1 numbers in the present experiment; the E1 result is the published anchor.

The pre-registered prediction (§6 below) is *not* that the small-scale gap of $+54$ PPL is preserved at scale — small-scale gaps in LM training typically shrink, often substantially, as scale grows. The prediction is about the *direction* of the gap and a *plausible band* for its magnitude.

---

## 4. Hypotheses

Let $P_A$ be the final validation perplexity of arm A (SPLM em\_ln scale-up) at the chosen seed configuration and $P_B$ the final validation perplexity of arm B (matched-baseline scale-up) at the same seed configuration. Both are evaluated on the same held-out TinyStories validation shard with `eval_iters = 40` and the same `block_size = 512`.

Define the matched-parameter quality gap

$$\Delta = P_B - P_A,$$

so $\Delta \gt 0$ means SPLM beats matched-attention.

| Hypothesis | Operational form | Theoretical reading |
|---|---|---|
| $H_1$ (the paper's claim survives scale-up) | $\Delta \ge +\Delta_{\min}$, sign-consistent across all seeds run | The matched-parameter SPLM-vs-attention quality ranking established in E1 generalises to a 2.2× param / 16× token / 4× context-length scale-up |
| $H_0$ (gap shrinks to ambiguous) | $\lvert \Delta \rvert \lt \Delta_{\min}$, or sign-inconsistent across seeds | The SPLM advantage at small scale is regime-dependent; at scale-up the two architectures are quality-comparable. The paper's headline narrows from "SPLM beats matched-attention" to "SPLM exhibits the predicted dynamical signatures and is competitive with matched-attention at small scale" |
| $H_{-1}$ (gap inverts) | $\Delta \le -\Delta_{\min}$, sign-consistent across all seeds run | Matched-attention overtakes SPLM at the scale-up regime. The paper's empirical PPL-win is restricted to the small-scale window; the architectural / dynamical-systems contributions (E4, E5, E7, E8 efficiency) remain valid claims, but the headline "SPLM beats matched-attention" cannot be made without a "at the small-corpus scale" qualifier |

---

## 5. Decision rule (locked at pre-registration)

### 5.1 Effect-size threshold

The minimum effect size is fixed at:

$$\Delta_{\min} = 5.0 \text{ perplexity units}.$$

**Justification.** In the E1 multi-seed sweep at small scale, the per-seed standard deviation of `splm_em_ln` PPL was $4.44$ and of `matched_baseline` was $7.21$. A difference smaller than ${\sim}5$ PPL is at the edge of the single-seed measurement uncertainty inherited from E1; a difference larger than $5$ PPL exceeds it. The threshold is symmetrical (Outcomes A and C use the same $\lvert\Delta\rvert \ge 5$ requirement) so the protocol is equally rigorous against confirming and refuting the paper's claim.

### 5.2 Adaptive seed plan

Single-seed scale-up runs on the reference hardware are expensive (${\sim}9.5$ h SPLM + ${\sim}14$ h matched ≈ 24 h end-to-end). The protocol uses an adaptive two-phase seed allocation, *both phases pre-registered*:

**Phase 1 — single seed, mandatory.** Run arm A and arm B at `seed = 0` exactly. Compute $\Delta^{(0)} = P_B^{(0)} - P_A^{(0)}$.

**Phase 2 trigger — locked at pre-registration.**
- **If $\lvert \Delta^{(0)} \rvert \ge 20$ PPL** (clearly decisive in either direction, well above E1's seed-induced PPL uncertainty), **the protocol terminates at $S=1$**. The headline is reported with an explicit single-seed disclaimer of "${\pm 5}$ PPL inherited uncertainty from E1" and the outcome (A / B / C) is locked.
- **If $\lvert \Delta^{(0)} \rvert \lt 20$ PPL** (within the ambiguous zone), run **two additional seeds** at `seed = 1` and `seed = 2` for a $S=3$ paired band. The outcome is then determined from the per-seed mean $\overline{\Delta}$ as in §5.3.

The rationale for the $\lvert \Delta^{(0)} \rvert \ge 20$ threshold is that, at small scale, E1 had a $+54$ PPL gap with $\sigma_A = 4.44$ — i.e., the small-scale signal is roughly $12\sigma$ above noise. Even a $20$-PPL gap at scale-up would be ${\sim}4\sigma$ if noise scales linearly with PPL, which is far enough from the threshold that two additional seeds would not change the qualitative outcome. A gap inside ${\pm}20$ PPL is genuinely ambiguous at $S=1$ and *does* warrant additional seeds.

**No other seed configuration is permitted.** If a phase-1 cell crashes (e.g., NaN-divergence as observed in E1's `sarfmass_logfreq` arm), the failure is logged and a single replacement seed is run; the cell is replaced strictly in the order $1, 2, 3, \ldots$.

### 5.3 Outcome determination

Let $\overline{\Delta}$ denote $\Delta^{(0)}$ (Phase-1-only termination) or the per-seed mean $\tfrac{1}{3}\sum_{s=0}^{2} \Delta^{(s)}$ (Phase-2 termination), with the per-seed sign function $\mathrm{sgn}(\Delta^{(s)})$ defined for each.

- **Outcome A ($H_1$ confirmed; SPLM beats matched-attention at scale-up):**
  $\overline{\Delta} \ge +5.0$ **and** all per-seed signs are $\gt 0$ (sign-consistency across whichever seeds were run).
- **Outcome B ($H_0$; ambiguous / scale-dependent):**
  $\lvert \overline{\Delta} \rvert \lt 5.0$, *or* per-seed signs are inconsistent.
- **Outcome C ($H_{-1}$; gap inverts; matched-attention beats SPLM at scale-up):**
  $\overline{\Delta} \le -5.0$ **and** all per-seed signs are $\lt 0$ (sign-consistency).

The outcome (A / B / C) is determined **only** from $\overline{\Delta}$ and the per-seed sign-consistency on the seeds that completed, with the seed count locked by §5.2. No post-hoc threshold adjustment, training-budget adjustment, learning-rate retuning, or per-seed exclusion is permitted.

### 5.4 Auxiliary observations (recorded but not part of the decision rule)

The following are **not** part of the locked decision rule but are recorded for the eventual write-up:

- Final train loss (both arms).
- $\gamma$ trajectory across training (arm A only).
- Wall-clock per arm.
- Loss-curve overlay: arm A vs arm B on the same axes, train and val.
- Memory-usage peak (per arm), to reality-check the SPLM streaming-$\xi$ memory-cost claim at long context.

---

## 6. Pre-registered prediction

The author predicts **Outcome A** (the gap survives scale-up), with a substantially smaller numerical effect size than at small scale, written before any scale-up training is run:

$$\Delta_{\mathrm{predicted}} \in [+10,+30] \text{ PPL}, \qquad \text{with most likely value } \overline{\Delta} \approx +20 \text{ PPL}.$$

**Reasoning.** Three considerations narrow the prediction band:
1. The small-scale gap of $+54$ PPL was at a regime where the matched-attention arm is severely under-parameterised relative to the corpus complexity (8 M params on Tiny Shakespeare — within the memorisation-vs-generalisation transition zone). At 19 M params on 5 M tokens, both arms are still well below Chinchilla but the matched-attention disadvantage from the small-corpus regime should weaken.
2. SPLM's architectural advantages — conservativity by construction, prescribed per-token mass, the LayerNorm-after-step regulariser — are largely scale-invariant in their formulation and should retain at least *some* of their value at the new scale.
3. There is a 2.2× model-size scale-up but only a 16× corpus scale-up; the ratio of model-to-data has *worsened* slightly, which on the Chinchilla scaling law would marginally favour SPLM (which has a stronger inductive bias).

A confirming prediction in the predicted band would be the cleanest possible scale-up evidence for the TMLR submission. A prediction-falsifying outcome (Outcome C, $\overline{\Delta} \le -5$) would require honest re-framing of the paper's empirical headline.

The author assigns the following subjective probabilities to the three outcomes, **before** any scale-up training is executed:

| Outcome | Probability (subjective, pre-registered) |
|---|---:|
| A — $\Delta \ge +5$, gap survives | $0.65$ |
| B — $\lvert\Delta\rvert \lt 5$ or sign-inconsistent | $0.25$ |
| C — $\Delta \le -5$, gap inverts | $0.10$ |

---

## 7. Reporting plan

Regardless of which outcome is realised, the experiment is written up as follows:

- **Code lives at:** `notebooks/conservative_arch/scaleup/`. The directory contains:
  - `compute_unigram_frequencies_tinystories.py` — adapts the SARF-mass surprisal computation for TinyStories (one-off; re-runnable).
  - `train_splm_em_ln_scaleup.py` — SPLM em\_ln trainer in `--mode scaleup`.
  - `train_matched_baseline_scaleup.py` — matched-baseline trainer in `--mode scaleup`.
  - `README.md` — short experiment overview.
  - `results/` — per-seed checkpoints (`*.pt`), training logs (`*.jsonl`), summary `.md` files, loss-curve `.png` overlays, plus the canonical `RESULTS.md` write-up.

- **Headline `RESULTS.md`** in `notebooks/conservative_arch/scaleup/results/RESULTS.md` will report:
  - The locked decision rule, the realised $\overline{\Delta}$, and the locked outcome (A / B / C).
  - A two-row paired table: arm A vs arm B per seed.
  - The loss-curve overlay figure.
  - $\gamma$ trajectory plot (arm A).
  - Per-arm wall-clock + memory peak.
  - Cross-reference back to the E1 small-scale anchor.
  - One paragraph on the auxiliary observations.

- **Paper-side consequence (TMLR submission, conditional on outcome):**
  - **Outcome A:** add a "Scale-up corroboration" subsection (~half a page) to the experimental section of the cut-down SPLM paper, citing $\overline{\Delta}$ and the matched-parameter point. The section title can promote SPLM's ranking from "demonstrated at the Tiny Shakespeare scale" to "demonstrated at two scales: Tiny Shakespeare ($S=5$ seeds) and TinyStories ($S=$ realised seed count)".
  - **Outcome B:** the cut-down SPLM paper retains the E1 small-scale headline as the primary empirical result, and adds a one-paragraph "Scale-up: results are competitive at TinyStories, with the small-scale gap not preserved" subsection that is honest about scope. The architectural / dynamical-systems / efficiency contributions (E4, E5, E7, E8) are unaffected.
  - **Outcome C:** the cut-down SPLM paper's headline is rewritten. The empirical PPL-win is restricted to the small-scale window with explicit caveat. The paper's primary contribution shifts from "SPLM beats matched-attention" to "SPLM exhibits the predicted dynamical signatures, is competitive at small scale, and provides constant per-token decode cost". This is still a publishable TMLR submission but its narrative arc is meaningfully different from the Outcome-A / B versions.

- **No post-submission revision of the realised outcome is permitted.** If, after the scale-up experiment is run and recorded, the author chooses to extend with additional scale runs (e.g., enwik8, WikiText-103, a third-tier 50 M+ model), those become *new* pre-registered protocols with their own decision rules; they cannot retroactively change the outcome of E9.

---

## 8. Pre-registration metadata

- **Drafted:** 2026-04-29.
- **Pre-registered (committed):** the commit hash that introduces this file is the timestamp of pre-registration.
- **Author:** Dimitar Gueorguiev (with Claude as drafting assistant).
- **No data-dependent decisions before this commit.** No scale-up training has been launched, no smoke test at the scale-up configuration has been run, and no preliminary PPL numbers exist at the time of drafting. The 300-step smoke test referenced by the SPLM-1 protocol is at a *different* configuration and provides no information about scale-up PPL.

The companion implementation files (`notebooks/conservative_arch/scaleup/`) will be added in a *separate* commit after the user has reviewed and approved this protocol. The smoke test itself (a 300-step verification that gradient flow works at the scale-up configuration) is allowed *between* protocol commit and the production run, and its sole purpose is to confirm pipeline correctness; its training loss is not used to set any threshold above.
