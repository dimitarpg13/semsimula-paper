# Pre-Registered Protocol — SPLM Inference-Efficiency Benchmark

> Pre-registration document, drafted **April 29, 2026**, by Dimitar Gueorguiev with Claude.
> Companion to:
> *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference* (Gueorguiev, 2026), v3, **Appendix A2** "Inference efficiency: FLOP and parameter-count comparison between SPLM and attention transformers."
> Companion experiments:
> - SPLM-1 first-order ablation: [`companion_notes/SPLM-1_ablation_pre-registered_protocol.md`](./SPLM-1_ablation_pre-registered_protocol.md) — pre-registered concurrently; in flight at time of draft.
> - SPLM inference Markov-order test (E7): [`companion_notes/SPLM_inference_first_order_pre-registered_protocol.md`](./SPLM_inference_first_order_pre-registered_protocol.md) — pre-registered concurrently.

> **Status.** Pre-registered, not yet executed. This document fixes the experimental design, the analysis pipeline, and the decision rule **before** any wall-clock measurement, FLOP count, or matched-baseline retraining is run with the intention of comparing against SPLM. The committing commit hash is the timestamp of pre-registration.

---

## 1. Question

the paper Appendix A2 advances four quantitative claims about SPLM's inference efficiency relative to a parameter-matched attention transformer:

- **A2.C1 — No $T^2$ attention term.** SPLM's per-forward-pass cost is $O(LTd\,d_V)$; attention is $O(LTd^2 + LT^2d)$. As $T \to \infty$ at fixed $d$, the ratio $F_{\text{attn}} / F_{\text{splm}}$ grows linearly in $T$.
- **A2.C2 — FLOP crossover at $T^{\ast} = 34d$** (eq. A2.crossover, derived for the prototype $d_V = 4d$, $v_{\text{depth}} = 3$). For $d = 128$, $T^{\ast} \approx 4\,352$.
- **A2.C3 — KV-cache-free decoding.** Per-new-token autoregressive decode cost at prefix $T$ is $O(L\,d\,d_V)$ for SPLM (constant in $T$) vs. $O(LTd)$ for attention with KV cache. Cache memory shrinks from $2LTd$ to $Ld$ — a factor of $2T$.
- **A2.C4 — Depth-independent parameters.** SPLM's non-embedding parameter count is independent of $L$; attention's grows as $12Ld^2$.

The paper itself flags one variant of these claims as **explicitly open** in A2's last paragraph:

> "A quality-adjusted statement — *SPLM reaches cross-entropy $\mathcal{L}^{\ast}$ at lower total inference FLOPs than an attention transformer* — is empirically open and is listed among the explicit open follow-ups in §17."

This is "Q6" of the paper's open-questions list. Closing it is the primary goal of this experiment. Verifying the unconditional FLOP/parameter claims (A2.C1–C4) is the secondary goal.

---

## 2. The two architectures under test

| Symbol | Architecture | Source |
|---|---|---|
| **SPLM\_2** | SPLM em\_ln at fixed $\gamma^{\ast} = 0.30$ (the E5 winner) | `notebooks/conservative_arch/energetic_minima/model_ln.py:ScalarPotentialLMSARFMassLN` + checkpoints from the SPLM-1 ablation sweep arm B |
| **SPLM\_1** | First-order ablation (no v-buffer, no $\gamma$) | `notebooks/conservative_arch/first_order_ablation/model_first_order.py:ScalarPotentialLMFirstOrder` + checkpoints from the SPLM-1 ablation sweep arm A |
| **ATTN\_match** | GPT-2-style decoder, parameter-matched to SPLM\_2 | `notebooks/conservative_arch/matched_baseline_model.py` (model class exists; trainer is **new** for this experiment, see Phase 1) |

ATTN\_match is the natural inference-efficiency comparison. SPLM\_1 is included because it is structurally cheaper than SPLM\_2 (no v-buffer update per Euler step) and because it provides a Pareto-frontier point — the question "is the inference advantage stronger if we accept SPLM\_1's quality penalty?" only has a sharp answer if we measure SPLM\_1.

The configuration triple, locked at pre-registration:

| | $d$ | $L$ | $d_V$ | $n_{\text{head}}$ | non-emb params | val PPL @ Shakespeare |
|---|---:|---:|---:|---:|---:|---:|
| SPLM\_2 (E5 winner) | 128 | 8 | 512 | — | $\sim 0.66$ M | 87.06 (seed 0; multi-seed mean from SPLM-1 ablation arm B) |
| SPLM\_1 | 128 | 8 | 512 | — | $\sim 0.66$ M | $\sim 109$ (seed 0; multi-seed mean from arm A) |
| ATTN\_match | 128 | 8 | — | 4 | $\sim 1.6$ M | **TBD by Phase 1** |

ATTN\_match's val PPL is *not* known at pre-registration — that's the point of Phase 1.

---

## 3. Three-phase design

### Phase 1 — Quality re-baseline of ATTN\_match (E8.1)

The paper currently quotes ATTN\_match at val PPL 142 (the paper §6 / A2.eq:flops-attn instantiated paragraph). That number predates the LN-after-step + damping-sweep work that brought SPLM\_2 to 87.06 PPL. To fairly compare quality vs. inference cost, we re-train ATTN\_match at the **identical** training protocol used by the SPLM-1 ablation sweep arm B:

- 4 000 optimisation steps, batch size 16, block size 128.
- AdamW, $\text{lr} = 5\times10^{-4}$ with cosine decay and 200-step warmup.
- Weight decay 0.01, betas $(0.9, 0.95)$, grad-clip 1.0.
- Tiny Shakespeare data, GPT-2 BPE, deterministic batch RNG.
- Three seeds: $\{0, 1, 2\}$.
- Per-seed final val PPL evaluated with `eval_iters = 40` on the same held-out validation split as the SPLM-1 ablation.

Code: `notebooks/conservative_arch/inference_efficiency/train_matched_baseline.py` (new file; structurally a delta of `notebooks/conservative_arch/first_order_ablation/train_splm_first_order.py` with the model class swapped to `MatchedBaselineLM`).

#### Phase 1 hypotheses

Let $P_{\text{ATTN}}^{(s)}$ be the per-seed val PPL at seed $s$, and let $\overline{P}_{\text{ATTN}} = \tfrac{1}{3}\sum_s P_{\text{ATTN}}^{(s)}$. Define $\Delta_{\text{quality}} = \overline{P}_{\text{ATTN}} - \overline{P}_{\text{SPLM\_2}}$ (negative means ATTN\_match wins on quality).

| Outcome | Operational form | Paper consequence |
|---|---|---|
| **Q1 — quality parity** | $\lvert \Delta_{\text{quality}} \rvert < 5.0$ PPL | A2.C1–C3 inference-efficiency claims gain force; quality is not a confounding factor. |
| **Q2 — SPLM\_2 better** | $\Delta_{\text{quality}} \ge +5.0$ PPL | the paper has a *stronger* claim than v3 currently makes; A2 paragraph 1 should be updated to claim SPLM matches *or beats* attention at this scale. |
| **Q3 — ATTN better by a margin** | $\Delta_{\text{quality}} \le -5.0$ PPL | the inference-efficiency story is now conditional on quality parity at scale, which is not observed at the prototype scale. A2 must be rewritten to note that SPLM is FLOP-cheaper *given* quality parity at larger scale, and the small-scale prototype gap is a known limitation. |

5.0 PPL is identical to the $\Delta_{\min}$ used in the SPLM-1 ablation, for consistency.

### Phase 2 — Unconditional inference benchmarks (E8.2)

Phase 2 verifies A2.C1, A2.C2, A2.C3, and A2.C4 *without conditioning on quality*. It compares the three architectures (SPLM\_2 seed 0, SPLM\_1 seed 0, ATTN\_match seed 0) across a sequence-length grid:

$$T \in \{128, 256, 512, 1024, 2048, 4096, 8192, 16384\}.$$

The grid is extended past the paper's $T = 4096$ analytic prediction so the practical wall-clock crossover (which lags FLOP crossover because of kernel maturity asymmetry — see §7 caveat) has room to manifest.

For each $(T, \text{architecture})$ cell, four numbers are recorded:

1. **$F^{\text{fwd}}(T)$** — forward-pass FLOPs computed analytically from the closed-form expressions in A2.eq:flops-attn and A2.eq:flops-splm. **No measurement; pure arithmetic.**
2. **$W^{\text{fwd}}(T)$** — forward-pass wall-clock time in seconds, measured on the local MPS device, after a warm-up of 5 calls, then median of 20 timed calls, batch size 1.
3. **$F^{\text{dec}}(T)$** — per-new-token autoregressive-decode FLOPs at prefix $T$ (analytic, from A2.eq:decoding terms).
4. **$W^{\text{dec}}(T)$** — per-new-token autoregressive-decode wall-clock at prefix $T$. Median of 20 calls, after warm-up.

Implementation requirements (locked):

- **For ATTN\_match.** Standard pre-LN GPT-2 block. KV-cache implemented in the matched-baseline trainer's inference path (delta of HuggingFace's standard `past_key_values` mechanism).
- **For SPLM (both flavours).** A streaming-$\xi$ inference path. The current `model.forward` recomputes $\xi_t$ from scratch on every call, which artificially inflates SPLM's per-new-token decode cost from $O(Ld\,d_V)$ to $O(LTd\,d_V)$. The paper's A2.C3 claim explicitly assumes streaming. Implementing streaming-$\xi$ is part of this experiment; we add `model.generate_streaming(prefix, max_new_tokens)` that maintains a single $L$-vector of running cumulative means and updates them incrementally.
- **No torch.compile, no flash-attention.** Both architectures run in eager-mode PyTorch, with the same kernel-maturity baseline. The protocol explicitly does *not* claim wall-clock speedups under fused / compiled kernels — that would require a separate "production engineering" study.

Code: `notebooks/conservative_arch/inference_efficiency/{count_flops.py, benchmark_wallclock.py, model_streaming.py}` (all new).

#### Phase 2 hypotheses (locked)

| Sub-claim | Operational form | Pre-registered prediction |
|---|---|---|
| **A2.C2** — FLOP crossover at $T^{\ast} = 34d$ | for $d = 128$, $T^{\ast} \in [128, 4096]$ such that $F_{\text{attn}}^{\text{fwd}}(T^{\ast}) = F_{\text{splm}}^{\text{fwd}}(T^{\ast})$ | $T^{\ast} \in [4\,000, 4\,700]$ (i.e. paper's $34d \approx 4\,352$ ± 8 %), with $T^{\ast}$ between $T = 1\,024$ and $T = 4\,096$. |
| **A2.C1** — long-context FLOP scaling | $F_{\text{attn}}^{\text{fwd}}(T) / F_{\text{splm}}^{\text{fwd}}(T)$ grows linearly in $T$ at $T \ge T^{\ast}$ | for $T \in \{8\,192, 16\,384\}$ the ratio should approximately double, consistent with linear-in-$T$ growth. |
| **A2.C3** — constant-cost decoding for SPLM | $W_{\text{splm}}^{\text{dec}}(T) - W_{\text{splm}}^{\text{dec}}(128) < 5\%$ of $W_{\text{splm}}^{\text{dec}}(128)$ for all $T \le 16\,384$ | confirmed if streaming-$\xi$ is implemented correctly; this is essentially a unit test on the streaming path. |
| **A2.C3** — linear-cost decoding for attention | $W_{\text{attn}}^{\text{dec}}(T)$ scales as $\alpha + \beta T$ with $\beta > 0$ on a regression of the 8 grid points | confirmed if KV-cache is implemented correctly; also a unit test. |
| **A2.C4** — depth-independent parameters for SPLM | report counted parameters for SPLM $L \in \{4, 8, 16\}$ and ATTN $L \in \{4, 8, 16\}$ | trivial; SPLM should be flat in $L$ (modulo the v-buffer for SPLM\_2 which is layer-state, not parameter), ATTN should be linear in $L$. |
| **Wall-clock crossover (new claim, not in v3)** | $T_{\text{wc}}^{\ast}$ such that $W_{\text{attn}}^{\text{fwd}}(T_{\text{wc}}^{\ast}) = W_{\text{splm}}^{\text{fwd}}(T_{\text{wc}}^{\ast})$ | The paper's A2 acknowledges kernel-maturity asymmetry as a known caveat. The pre-registered prediction is that $T_{\text{wc}}^{\ast}$ exists but lies above $T^{\ast}$ (i.e. above $\sim 4\,352$). If $T_{\text{wc}}^{\ast} \le 16\,384$, the practical wall-clock advantage is realisable at long contexts; if not, the FLOP advantage is conceded as theoretical-only at the prototype kernel. |

#### Phase 2 decision rule (locked)

Each sub-claim is independently classified as **CONFIRMED** / **MARGINAL** / **REFUTED**:

| Sub-claim | CONFIRMED if | MARGINAL if | REFUTED if |
|---|---|---|---|
| A2.C2 | $T^{\ast}$ within ± 8 % of $34d$ | within ± 20 % | outside ± 20 % |
| A2.C1 | ratio grows by a factor $\ge 1.8$ when $T$ doubles in the long-context regime | $\ge 1.4$ but $< 1.8$ | $< 1.4$ |
| A2.C3 SPLM | $\le 5$ % drift | $\le 20$ % | $> 20$ % |
| A2.C3 ATTN | $R^2 \ge 0.95$ on linear fit | $\ge 0.85$ | $< 0.85$ |
| A2.C4 | flat to within 1 % of SPLM params, linear $\pm 5$ % for ATTN | flat to within 5 %, linear $\pm 15$ % | otherwise |
| WC crossover | $T_{\text{wc}}^{\ast} \le 16\,384$ | $T_{\text{wc}}^{\ast} > 16\,384$ but ratio decreasing monotonically | ratio not decreasing |

The headline conclusion of Phase 2 is determined by the count of CONFIRMED / MARGINAL / REFUTED sub-claims:

- **5 or 6 CONFIRMED, 0 REFUTED** → Phase 2 outcome **A**: A2 verified end-to-end.
- **3 or 4 CONFIRMED, 0 REFUTED** → outcome **B**: A2 broadly correct; soften the affected sub-claims.
- **any REFUTED, or fewer than 3 CONFIRMED** → outcome **C**: A2 is materially wrong; rewrite the affected sub-claim with the measured number replacing the asymptotic argument.

### Phase 3 — Quality-adjusted comparison (the "Q6" question, E8.3)

For each architecture, plot val PPL on the validation split (a **fixed** number per architecture, computed at evaluation $T = 128$ to match the training configuration) against forward-pass FLOPs at multiple inference $T$. The quality number is fixed; the FLOP number is the inference cost the user pays per evaluation context of length $T$.

This produces a "PPL-vs-inference-FLOPs at length-$T$" plot for $T \in \{128, 1\,024, 4\,096, 16\,384\}$. The Pareto-dominant architecture at each $T$ is the headline.

#### Phase 3 hypotheses (locked)

| Outcome | Operational form | Conclusion |
|---|---|---|
| **P1 — Q6 confirmed** | At $T = 4\,096$ and $T = 16\,384$, SPLM\_2 lies on the Pareto frontier of (PPL, FLOPs); SPLM\_2 is FLOP-cheaper than ATTN\_match by $\ge 1.5\times$ at PPL within $\pm 5$ of ATTN\_match's. | A2.last-paragraph "Q6 unresolved" wording is replaced by a positive empirical statement. |
| **P2 — Q6 partially confirmed** | SPLM\_2 is FLOP-cheaper at $T = 16\,384$ but not at $T = 4\,096$. | A2 is updated to note that the quality-adjusted advantage materialises at $T \ge T_{\text{Q6}}^{\ast}$, with $T_{\text{Q6}}^{\ast}$ given by the data. |
| **P3 — Q6 falsified** | SPLM\_2 is not FLOP-cheaper at any $T \le 16\,384$ when conditioned on PPL parity (Phase 1 outcome Q1) or SPLM\_2 is not Pareto-dominant under outcome Q3. | A2 is materially weakened; the paper is rewritten to acknowledge that the quality-adjusted version of the inference-efficiency claim does not hold at this scale, and the asymptotic claim is reported as theoretical-only. |

The decision in Phase 3 is conditional on Phase 1: if Phase 1 returns Q3 (ATTN beats SPLM by $\ge 5$ PPL), the Pareto frontier at low $T$ is dominated by ATTN, and the question becomes "is SPLM still Pareto-dominant at $T = 16\,384$?" The locked threshold $1.5\times$ FLOP saving compensates for up to a $\sim 10$-PPL quality gap.

---

## 4. Pre-registered predictions (separate from the decision rule)

The author's predictions, locked in writing for accountability:

- **Phase 1**: Q1 (quality parity, $|\Delta| < 5$). Belief: ATTN\_match at the new protocol reaches $\sim 90 \pm 10$ PPL, similar to SPLM\_2's 87. Caveat: this is the noisiest of the three predictions; the matched baseline at the *old* protocol reached 142, so the improvement to $\sim 90$ assumes the LR/schedule/grad-clip choices that helped SPLM also help ATTN.
- **Phase 2**: outcome **A** for everything except wall-clock crossover; outcome **B** for the wall-clock crossover (i.e. crossover exists at some $T \le 16\,384$ but is well above the FLOP $T^{\ast}$).
- **Phase 3**: outcome **P1** if Phase 1 returns Q1 or Q2, outcome **P2** if Phase 1 returns Q3.

---

## 5. What this experiment is **not** claiming

- **Not a kernel-optimisation study.** Both architectures are run in eager-mode PyTorch with no fused kernels. The wall-clock crossover is a function of the reference implementations only; a production SPLM implementation (custom kernel for the Euler step, fused with the cumulative-mean scan; cache-aware $\nabla_h V_\theta$ computation) would shift $T_{\text{wc}}^{\ast}$ left, possibly substantially. The paper acknowledges this caveat (A2.where-it-does-not, point 2); we do not over-claim wall-clock speedups under conditions that haven't been measured.
- **Not a scaling study.** All three architectures are at the v3 prototype configuration ($d = 128$, $L = 8$). At larger $d$, the FLOP crossover $T^{\ast} = 34d$ moves to longer contexts in absolute tokens but shorter contexts as a fraction of $d$; how the wall-clock crossover scales with $d$ is unknown and would require a separate experiment.
- **Not a benchmark on long-context tasks.** The PPL number is from Tiny Shakespeare's validation split, computed at $T = 128$ to match training. We do *not* claim that SPLM is *better* at long-context understanding; we claim only that it is *cheaper at inference* on long contexts at matched (or close-to-matched) quality.
- **Not a comparison against optimised attention variants.** The matched baseline is a vanilla GPT-2-style transformer. We do not benchmark against linear-attention, state-space models (Mamba/S4), Hyena, RetNet, or other linear-in-$T$ alternatives. SPLM is positioned in v3 as a Lagrangian-derived architecture; the practical comparison against contemporary linear-time architectures is a separate research question that v3 does not raise.

---

## 6. Pitfalls and how each is handled

| Pitfall | Mitigation |
|---|---|
| Wall-clock noise at small $T$ on MPS | Median over 20 timed calls after 5-call warm-up; report inter-quartile range; pin the device to MPS only (no CPU fallback for any op). |
| ATTN\_match's KV-cache implementation is harder than streaming-$\xi$ — implementation asymmetry could bias wall-clock | Both implementations are reviewed for matching levels of optimisation: identical batch size, identical evaluation mode, identical no\_grad scope. Code is committed and reviewable; a sceptic can replicate. |
| Tiny Shakespeare validation split is too short for robust PPL at large $T$ | PPL is computed at $T = 128$ (training-time value), held fixed; the inference FLOPs are computed at the larger $T$ values without re-evaluating PPL. The Phase 3 plot conflates evaluation $T$ with training $T$ explicitly to highlight that the PPL number is from $T = 128$; the FLOP number is from each grid point. |
| Researcher degrees of freedom in choosing $T$ values | Grid is locked: $\{128, 256, 512, 1024, 2048, 4096, 8192, 16384\}$. No post-hoc cell selection. |
| Multiple comparisons across 8 $T$ values × 3 architectures × 4 metrics = 96 cells | The headline conclusion is a 6-row decision rule (Phase 2) and a 3-row Pareto verdict (Phase 3); no per-cell statistical inference is involved. The 96 numbers are reported in full but not multiplicity-corrected. |

---

## 7. Reporting plan

After all three phases run, the following artefacts are committed:

- `notebooks/conservative_arch/inference_efficiency/RESULTS.md` — headline (Phase 1 outcome × Phase 2 outcome × Phase 3 outcome), Pareto plot, FLOP and wall-clock tables.
- `notebooks/conservative_arch/inference_efficiency/results/phase1_matched_baseline/` — per-seed training logs and final PPLs.
- `notebooks/conservative_arch/inference_efficiency/results/phase2_benchmarks.csv` — 96-cell raw measurements.
- `notebooks/conservative_arch/inference_efficiency/results/figures/` — at minimum: $F$ vs $T$ log-log plot (FLOP), $W$ vs $T$ log-log plot (wall-clock), per-new-token decoding cost vs prefix length, Pareto-frontier scatter at each $T$.

### 7.1 Paper consequences

| Phase 1 | Phase 2 | Phase 3 | the paper consequence |
|---|---|---|---|
| Q1 or Q2 | **A** | **P1** | Add a §6.5 (or §A2.6) sub-section "Empirical verification of the inference-efficiency claims" with the headline numbers inline. Replace A2.last-paragraph "Q6 unresolved" wording with the measured FLOP-saving factor at $T = 16\,384$ at matched PPL. The paper now claims A2 end-to-end with empirical backing. |
| Q1 or Q2 | **B** | **P1** or **P2** | Same as above but soften the sub-claims rated MARGINAL in Phase 2; report the actual measured constants in place of the predicted ones. |
| Q3 | any | any | A2 is rewritten conditional on $\Delta_{\text{quality}}$ measured: SPLM is FLOP-cheaper *despite* a $\Delta_{\text{quality}}$-PPL quality penalty; the paper is honest that this is the price of the prototype scale. |
| any | **C** | any | The REFUTED Phase-2 sub-claim has its A2 paragraph rewritten with the measured number replacing the analytical argument. |
| any | any | **P3** | The "Q6" line in A2's last paragraph is replaced with: *"At the prototype scale, SPLM does not achieve a quality-adjusted inference-FLOP saving; the asymptotic claim of A2 is empirically realised only when matched on quality at larger scale, which is left to future work."* |

---

## 8. Compute estimate

| Step | Hardware | Time |
|---|---|---|
| Phase 1: 3-seed ATTN\_match training | local MPS | ~30 min/seed × 3 = 1.5 h |
| Phase 2: streaming-$\xi$ implementation (one-off) | local | ~2 h dev |
| Phase 2: 96-cell wall-clock benchmark | local MPS | ~30 min |
| Phase 3: PPL evaluation at $T = 128$ + Pareto plotting | local | ~10 min |
| RESULTS.md + figures | local | ~30 min |

**Total wall-clock: ~5 h**, distributed: ~1.5 h training + ~2 h dev + ~1.5 h benchmarking + analysis. Phase 1 and Phase 2 streaming-$\xi$ implementation are independent and can run in parallel.

---

## 9. Pre-registered authors' beliefs (separate from the decision rule)

The author believes the most likely composite outcome is **(Q1, A, P1)**: quality parity between ATTN\_match and SPLM\_2 at the improved protocol, all six A2 sub-claims confirmed in the unconditional benchmark, and a quality-adjusted FLOP saving of $\ge 2 \times$ at $T = 16\,384$. This belief does **not** modify the decision rule of §3; the locked thresholds stand regardless.

If the actual outcome is (Q3, *, *) — ATTN beats SPLM on quality at this scale — the paper consequence is the most consequential rewrite: A2 is reframed as "asymptotically SPLM is FLOP-cheaper, but at the prototype scale it pays a quality cost; closing this gap is the v4 scaling experiment". This would be a partial concession but not a falsification of the framework.

---
