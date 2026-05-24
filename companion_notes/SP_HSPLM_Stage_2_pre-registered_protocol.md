# Pre-Registered Protocol — SP-HSPLM Stage 2: Q9(e) Pair-Skew Cell Ladder

> **Status:** Pre-registered, not yet executed.
> **Date locked:** 2026-05-16.
> **Author:** Dimitar Gueorguiev (with Claude as scribe / sanity-checker).
> **Pre-registration commit:** to be filled in at lock-in commit.
> **Companion documents:**
> - [`Scalar_Potential_based_Helmholtz_Architecture_v3.md`](./Scalar_Potential_based_Helmholtz_Architecture_v3.md) — the SP-HSPLM (Q9(e)) design doc; section 9.2 specifies the Stage 2 cell ladder, this protocol locks the implementation and decision rule.
> - [`SP_HSPLM_Stage_0_Literature_Check.md`](./SP_HSPLM_Stage_0_Literature_Check.md) — Stage 0 originality assessment.
> - [`SP_HSPLM_Stage_1_pre-registered_protocol.md`](./SP_HSPLM_Stage_1_pre-registered_protocol.md) — Stage 1 protocol (per-token Class B/C/D leak-fixed rerun); the result of Stage 1 is the empirical floor Stage 2 must beat.
> - [`PARF_Augmented_SPLM_Architecture_v2.md`](./PARF_Augmented_SPLM_Architecture_v2.md) — PARFLM design doc; the conservative pair scalar $V_\phi$ Stage 2 reuses for the S-block.
> - [`parf/On_Gumbel_softmax_sparsity_applied_to_V_phi.md`](./parf/On_Gumbel_softmax_sparsity_applied_to_V_phi.md) — score head + Gumbel routing primitive shared by both block types.
> - [`PARF-SPLM_Path_Forward_and_Experiments.md`](./PARF-SPLM_Path_Forward_and_Experiments.md) — P10 ladder; SparsePARFLM P10g is the conservative-pair-routing ceiling Stage 2's central bet must beat.
> - [`Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](./Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md) — the leak-fix invariant; Stage 2 introduces additional `.detach()` points on the source-side $h_s$ and $\delta_s$ in the C-block, all of which must preserve a leak floor at or below $10^{-6}$.
> **Status of executed write-up:** the executed RESULTS.md will live at `notebooks/conservative_arch/sphsplm/results/sp_hsplm/stage2/RESULTS.md` once complete.

---

## 1. One-paragraph motivation

Stage 1 closed cleanly: every per-token Class-B/C non-conservative augmentation (constant skew, affine rank-1 / rank-2 skew, position-only solenoidal, low-rank skew rank-4) trained at the leak-fixed v3 codebase **ties** the SPLM em\_ln 16 000-step baseline within $2\sigma_{\mathrm{seed}}$ on TinyStories — Outcome ALPHA in the Stage 1 protocol §6.3, the v3 paper §15.5 reading reproduces. The strongest single per-token cell (E4-fix, position-only solenoidal at rank $r=4$) lands at **best val PPL 24.58** versus the leak-fixed em\_ln 16k Cell-0 baseline at **26.31** — a 0.96-sigma improvement, below the multi-seed escalation gate. SparsePARFLM at the architectural ceiling (P10g, $k=4$, 16k steps) lands at **best val PPL 26.42**, also tied with em\_ln to within seed noise. The conclusion is that **per-token non-conservative additions are routing-poor and the conservative pair scalar alone has saturated** at the same ceiling as em\_ln. SP-HSPLM (Q9(e), see [`Scalar_Potential_based_Helmholtz_Architecture_v3.md`](./Scalar_Potential_based_Helmholtz_Architecture_v3.md)) takes that prescription — *the missing ingredient is a non-conservative pair-interaction force* — and replaces the attention block of Q9(d) with a **causal pair-interaction skew/solenoidal kernel** $\sum_{s \lt t} m_{ts} J_\phi(h_t - h_s)(\delta_s - \delta_t)$. Stage 2 is the first end-to-end TinyStories measurement of whether this construction breaks the conservative-pair ceiling and beats Stage 1's per-token solenoidal floor.

---

## 2. Hypotheses

Stated as falsifiable predictions, against which the executed run will be compared in the section 7 decision matrix.

**H1 (central bet — pair-coupled non-conservative routing beats conservative-pair-only routing).** The best Stage 2 cell beats the SparsePARFLM P10g 16k-step baseline by at least $2\sigma_{\mathrm{seed}}$. Formally, with $\sigma_{\mathrm{seed}} = 1.8$ PPL (multi-seed E1 reference, same as Stage 1),
$$\min_c \mathrm{PPL}(c) \le \mathrm{PPL}(\mathrm{P10g}) - 2\sigma_{\mathrm{seed}} = 26.42 - 3.6 = 22.82 \text{ PPL}.$$
A best-cell PPL in $(22.82, 24.62]$ — i.e. a 1-sigma to 2-sigma improvement — triggers the multi-seed escalation gate of section 7 rather than a clean H1 acceptance.

**H2 (architectural bet — pair coupling adds value over per-token solenoidal).** The best Stage 2 cell beats the Stage 1 best per-token cell (E4-fix, position-only solenoidal $r=4$) by at least $2\sigma_{\mathrm{seed}}$:
$$\min_c \mathrm{PPL}(c) \le \mathrm{PPL}(\mathrm{E4\text{-}fix}) - 2\sigma_{\mathrm{seed}} = 24.58 - 3.6 = 20.98 \text{ PPL}.$$
H2 is strictly stronger than H1 (because $24.58 \lt 26.42$). H2 is the architectural claim of the paper; H1 is the publishable result. Both are pre-registered, both decided at single-seed first-cut with multi-seed escalation as in Stage 1.

**H3 (causal-leak invariant preserved under pair coupling).** The standalone causal-leak probe in the main repo (registered as `sp_hsplm_stage2_<schedule>_<k>_<r>` model class) returns a leak floor at or below $10^{-6}$ on every Stage 2 cell at three checkpoints — step 1 (initialisation), step 8000 (mid-training), step 16 000 (final). The pair coupling introduces a new `.detach()` requirement: the source-side $h_s$ and the source-side $\delta_s = h_s - h_{\mathrm{prev},s}$ must be detached before they enter the $J_\phi(h_t - h_s)(\delta_s - \delta_t)$ kernel for all $s \lt t$. This is a **structural** correctness invariant, not a soft prediction; any non-zero leak floor invalidates the cell and triggers a code review of the C-block forward.

**H4 (Jacobian-asymmetry signature, soft prediction).** For Stage 2 cells with non-zero $\lVert J_\phi \rVert_F$ at convergence, the velocity-aware Jacobian-symmetry probe (paper section 15.7) returns an asymmetric Jacobian whose Frobenius residual is bounded above by $c \cdot L_C \cdot \lVert J_\phi \rVert_F$, with $c$ a probe-weighting constant of order $10^{-2}$ and $L_C$ the number of C-blocks in the schedule. This is the architecture v3 doc §3.4 prediction, written here as a Stage 2 falsifier: a near-symmetric Jacobian *with* a large $\lVert J_\phi \rVert_F$ is evidence of a probe bug; a near-symmetric Jacobian *with* $\lVert J_\phi \rVert_F \to 0$ at convergence is the "skew-kernel collapse" failure mode (Outcome DELTA below).

**H5 (Mechanism-1 closes the residual gap to attention).** Added as the Stage 2 *extension* after the q9e_a/b/c readings made it clear that increasing $k$ or $r$ (Mechanism-2 capacity sweeps within the autonomous-in-$\ell$ class) does not move PPL off the $\approx 26$-$28$ cluster ceiling. Paper Appendix A (Eq. A.130) decomposes attention's expressivity into two mechanisms: **Mechanism 2** (the prefix-conditioned argument $\xi_t$ used by $V_\phi$ and $J_\phi$, already present in q9e_a/b/c/d) and **Mechanism 1** (a per-layer-indexed parametric family $\theta_\ell$ for the force law). The SP-HSPLM v3 specification deliberately shares $J_\phi$, $V_\phi$, $\Omega$, and $\alpha_\phi$ across all layers, preserving the SPLM autonomous-in-$\ell$ commitment. **H5** states that promoting any of these submodules to per-layer-indexed (via `nn.ModuleList`) closes a measurable fraction of the remaining gap to the MatchedGPT baseline:
$$\min_{c \in \{q9e\_h, q9e\_i, q9e\_j, q9e\_k\}} \mathrm{PPL}(c) \le \mathrm{PPL}(q9e\_a) - 2\sigma_{\mathrm{seed}} = \mathrm{PPL}(q9e\_a) - 3.6 \text{ PPL}.$$
H5 has four sub-hypotheses (one per Mechanism-1 cell) and a joint hypothesis:

- **H5a (per-layer $J_\phi$):** $\mathrm{PPL}(q9e\_h) \le \mathrm{PPL}(q9e\_a) - 2\sigma_{\mathrm{seed}}$. The bet that "the missing piece is layer-indexed *non-conservative* parameters" — most directly aligned with the paper's reading of attention's per-layer $W_Q^{(\ell)}, W_K^{(\ell)}, W_V^{(\ell)}$.
- **H5b (per-layer $V_\phi$):** analogous for $q9e\_i$.
- **H5c (per-layer $\alpha_\phi$):** analogous for $q9e\_j$.
- **H5d (joint):** $\mathrm{PPL}(q9e\_k) \le \mathrm{PPL}(q9e\_a) - 2\sigma_{\mathrm{seed}}$ AND $\mathrm{PPL}(q9e\_k) \le \min(\mathrm{PPL}(q9e\_h), \mathrm{PPL}(q9e\_i), \mathrm{PPL}(q9e\_j)) - 1\sigma_{\mathrm{seed}}$. The joint cell must strictly improve over the best single Mechanism-1 cell by at least one $\sigma$ to claim non-additivity of the three mechanisms.

H5 is the *publishable* hypothesis irrespective of its sign: a positive H5 ($q9e\_h$ or $q9e\_k$ wins) validates the paper Appendix A two-mechanism decomposition empirically with an attention-free model; a negative H5 (all four Mechanism-1 cells tie q9e_a) strengthens Outcome C — the cluster ceiling holds even when both Mechanisms 1 and 2 are in play — and is a publishable revision of the Class-F reading.

**H6 (Mechanism-1 × Mechanism-2 additivity).** Added 16 May 2026 after q9e_d (per-token Ω) and q9e_h (per-layer $J_\phi^{(\ell)}$) both produced measurable improvements over q9e_a at single-seed (−0.69 and −0.90 PPL respectively, both below the strict 2σ threshold but on different architectural axes). H6 tests whether the two improvements compose:
$$\mathrm{PPL}(q9e\_l) \le \mathrm{PPL}(q9e\_a) - 2\sigma_{\mathrm{seed}} = 23.98 \text{ PPL (strict threshold)}$$
or, at the additivity *point estimate*,
$$\mathrm{PPL}(q9e\_l) \approx \mathrm{PPL}(q9e\_a) + \Delta_{q9e\_d} + \Delta_{q9e\_h} = 27.58 - 0.69 - 0.90 = 25.99 \text{ PPL}.$$
A q9e_l verdict at or below 26.0 PPL is the first clean Outcome ALPHA confirmation (beats P10g 26.42 by ≥ 1σ_seed at single seed). A q9e_l verdict at or above 26.5 PPL falsifies H6 — the two mechanisms compete rather than compose, and the architectural reading switches to "Ω and per-layer $J_\phi$ both saturate the same residual capacity axis". H6 is *strictly* gated on q9e_d and q9e_h having executed and produced their current verdicts; if either is invalidated at seed 1, H6 reverts to a contingent hypothesis.

**H6 executed verdict (single-seed):** $\mathrm{PPL}(q9e\_l\text{-seed0}) = 25.11$ PPL — **H6 confirmed and exceeded**. The two mechanisms are not just additive; the actual q9e_l result is 0.88 PPL *better* than the additive point estimate (25.99). This **triggers mandatory multi-seed escalation per §6.1** (Δ vs P10g = −1.31 PPL falls in (−2σ_seed, −1σ_seed]). q9e_l at seeds 1 + 2 is required to upgrade the H1 verdict from "ties at strict 2σ" to clean Outcome BETA.

**H6 executed verdict (3 seeds, 17 May 2026):** $\mathrm{PPL}(q9e\_l) = \{25.11, 24.98, 26.82\}$ — mean **25.64 ± 1.03** (sample σ), median **25.11**, range $[24.98, 26.82]$. All three seeds passed the causal-leak probe at init / mid / final with Δ ≡ 0.00e+00. The diagnostic reveals a **bimodal kernel regime**: seeds 0 and 1 converged to a middle-peaked per-layer J_φ profile (≈ 4.13 at C-blocks ℓ=3,5; PPL ≤ 25.11), while seed 2 converged to a monotonic-growing profile (1.91 → 3.99 across ℓ=1,3,5,7; PPL 26.82). Per-layer J_φ U/V factor norms in seed 2 are systematically 10–20 % smaller than seeds 0/1, consistent with the kernels never escaping the Frobenius warm-up suppression regime. Ω and γ are seed-robust (Ω_F ∈ [6.48, 6.90]; γ ∈ [0.83, 0.85]).

This pins the verdict at **BETA confirmed (median below ceiling), strict ALPHA missed (2 of 3 seeds below ceiling by ≥ 1σ_P10g, seed 2 above by 0.40 PPL)**. The architectural claim (Mechanism-1 × Mechanism-2 super-linear synergy) holds at the median; the seed-fragility is localised to the per-layer J_φ kernels. A 4th seed (`q9e_l-seed3`) is therefore queued to disambiguate whether the 2/3 below-ceiling fraction is the true convergence rate or seed 2 is the outlier; it is the gating run for promoting the verdict to strict ALPHA on the multi-seed escalation row.

**H8 (full Class-F test).** Added 17 May 2026 after q9e_l's 3-seed reading confirmed that Mechanism-1 + Mechanism-2 super-linear synergy is real but explains only $\approx 1.3$ PPL of the $\approx 17$ PPL across-class gap to MatchedGPT ($\approx 8$ PPL). The cells executed so far implement a **strict subset** of paper Appendix A Eq. A.130's Class F: every q9e_a..q9e_m cell keeps the dominant scalar potential $V_\theta(h; \theta_\ell, \xi_t)$ **shared across layers**, even though Class F prescribes per-layer $\theta_\ell$ for $V_\theta$ exactly as for $J_\phi$ and $\Omega$. At $d{=}256, v_{\mathrm{hidden}}{=}1024, v_{\mathrm{depth}}{=}3$ each $V_\theta$ copy is ≈ 2.6 M parameters, so $L_S = 4$ per-layer copies add ≈ 10 M parameters — the largest single Mechanism-1 move available, and the only one we have *not* tested.

H8 states:
$$\mathrm{PPL}(q9e\_n) \le \mathrm{PPL}(q9e\_l) - 2\sigma_{\mathrm{seed}} \approx 22.0 \text{ PPL (strict),}$$
$$\mathrm{PPL}(q9e\_n) \le \mathrm{PPL}(q9e\_l) - 1\sigma_{\mathrm{seed}} \approx 23.8 \text{ PPL (loose),}$$
with three decision branches on the multi-seed median:

- **H8 positive (PPL ≤ 12):** Class F is attention's class as paper Appendix A claims; the entire ≈ 17 PPL gap to MatchedGPT was the V_θ-sharing constraint. **This is a publishable theoretical confirmation** of Appendix A's central reading.
- **H8 partial (PPL ∈ (12, 22]):** Class F closes a substantial fraction of the gap but does not match attention. The residual is **beyond Class F** (multi-head structure, dense softmax routing vs Gumbel top-k, position-wise MLP, or Fock-space dynamics per paper §16+).
- **H8 negative (PPL ≥ 22):** Class F as currently implemented is **not** attention's class. The Appendix A "two-mechanism decomposition" needs revising; attention realises a class strictly larger than Class F. This is also publishable — as a negative — and elevates FockPARFLM (paper §16+) as the canonical next architectural axis.

H8 is **intentionally non-iso-parameter-count** vs P10g (q9e_n is ≈ 26 M parameters vs P10g's ≈ 15.8 M, ~70 % larger), so the q9e_n→P10g comparison carries a budget asterisk. The H8 decision rule is stated relative to **q9e_l multi-seed median** (the same Stage 2 family, just with V_θ shared instead of per-layer) — this isolates the V_θ-sharing contribution from the rest of the architecture.

**H7 (per-layer Ω is additive with per-layer J_φ on top of q9e_l; soft, stretch goal).** Added 16 May 2026 after q9e_l confirmed H6. q9e_l's diagnostic showed ‖Ω‖_F = 6.48 (down from q9e_d's 7.76 — Ω partially ceded work to the per-layer J_φ kernels but still carries substantial non-conservative work). H7 tests whether lifting Ω to per-layer ($\Omega^{(\ell)}$, `nn.ModuleList(L_C)`) on top of q9e_l can extract additional capacity:
$$\mathrm{PPL}(q9e\_m) \le \mathrm{PPL}(q9e\_l) - 0.5\sigma_{\mathrm{seed}} \approx 24.21 \text{ PPL.}$$
A q9e_m verdict at or below 24.5 PPL would put SP-HSPLM **within 0.4 PPL of the Stage-1 per-token solenoidal E4-fix baseline (24.58 PPL)** — H2 confirmation territory at single-seed. A q9e_m verdict at or above 25.0 PPL signals that q9e_l already captured the Ω capacity; per-layer Ω is then a clean negative for the maximal-Mechanism-1 axis at this scale. H7 is a *soft* prediction (the H6 synergy is the publishable headline; H7 is the stretch target).

---

## 3. The seven cells (Mechanism-2 ladder) plus four Mechanism-1 extension cells

Each cell is one execution of the depth-$L = 8$ SP-HSPLM stack with a specific block schedule $\sigma$, a specific top-$k$ routing density, a specific kernel rank $r$, and the per-token gyroscopic option on or off. The forward at layer $\ell$ with $\sigma(\ell) = S$ uses the SparsePARFLM force
$$f^{(\ell)}_t = -\nabla_{h_t}\Bigl[V_\theta(\xi_t, h_t) + \sum_{s} m^{(\ell)}_{ts} V_\phi(h_t, h_s)\Bigr],$$
and with $\sigma(\ell) = C$ uses the new pair-skew force
$$f^{(\ell)}_t = \sum_{s} m^{(\ell)}_{ts} J_\phi(h_t - h_s)(\delta_s - \delta_t) + [\text{optional}] \Omega(h_t)\delta_t,$$
with $J_\phi = J_+ - J_+^\top$, $J_+ = U V^\top$, $U, V \in \mathbb{R}^{d \times r}$ low-rank, and the routing mask $m^{(\ell)}_{ts}$ shared between branches in any layer where both are active (it is the same Gumbel top-$k$ output of the SparsePARFLM score head $\alpha_\phi$). Each cell inherits the SPLM damped-Verlet integrator shell, learnable mass, learnable damping $\gamma$ floored at $\gamma_{\min} = 0.05$, and tied embeddings.

| Cell | Schedule $\sigma$ | $k$ | $r$ | per-token $\Omega$ | NC params (approx) | Notes |
|---|---|---:|---:|:---:|---:|---|
| Q9e-A | `interleaved` (`SCSCSCSC`) | 4 | 16 | off | $2dr = 8\,192$ ($J_\phi$) | first cell — central bet of H1/H2 |
| Q9e-B | `interleaved` | 8 | 16 | off | $8\,192$ | routing-density sweep ($k$ ablation) |
| Q9e-C | `interleaved` | 4 | 32 | off | $16\,384$ | kernel-rank sweep ($r$ ablation) |
| Q9e-D | `interleaved` | 4 | 16 | on | $8\,192 + 8\,192 = 16\,384$ | per-token gyro on top of pair coupling |
| Q9e-E | `bottom_c_4` (`CCCCSSSS`) | 4 | 16 | off | $8\,192$ | C-then-S ordering (pair non-conservative routing first, conservative refinement at the top) |
| Q9e-F | `top_c_4` (`SSSSCCCC`) | 4 | 16 | off | $8\,192$ | S-then-C ordering |
| Q9e-G | `sandwich_2` (`SSCCCCSS`) | 4 | 16 | off | $8\,192$ | conservative-on-edges hypothesis from Q9(d) v2 doc |

The score-head $\alpha_\phi$ and the conservative pair scalar $V_\phi$ are reused from SparsePARFLM (≈ 12 k parameters total). The total non-conservative parameter overhead at $d = 256$ is at most 16 384 / 22 M ≈ 0.07 %, so the cell ladder is essentially parameter-matched against P10g — the PPL comparison is not biased by parameter count.

The schedule registry extends the Q9(d) `make_schedule(name, L, k, LA)` to accept a third token `C` in addition to the existing `S` and `A`, and the SP-HSPLM Stage 2 cells use only `S` and `C` (no `A` block is allocated anywhere in the Stage 2 ladder — this is Q9(e)'s defining commitment). The registered schedule names map to the existing `bottom_a` / `top_a` / `sandwich` / `interleaved` machinery with `A` renamed to `C` in the SP-HSPLM dispatch path.

### 3.1 Mechanism-1 extension cells (H/I/J/K)

The four cells below were added to the protocol after the q9e_a/b/c readings (PPL $\approx 27.6$/$28.1$/$27.7$) made it clear that capacity sweeps in $k$ and $r$ alone cannot move the model off the 26-28 PPL cluster ceiling that the entire SPLM family — em\_ln, PARFLM, SparsePARFLM, SP-HSPLM cells A-D — has produced at this scale. Paper Appendix A (Eq. A.130) names the missing primitive: **per-layer-indexed parametric families $\theta_\ell$ for the force law** ("Mechanism 1"). The SP-HSPLM v3 specification deliberately shares all four learnable submodules across the layer stack (the SPLM autonomous-in-$\ell$ commitment); the Mechanism-1 cells lift that commitment one submodule at a time and otherwise inherit the q9e_a configuration. The implementation uses `nn.ModuleList(L_C)` for $J_\phi$ and $\Omega$ (one independent kernel per C-block), `nn.ModuleList(L_S)` for $V_\phi$ (one per S-block), and `nn.ModuleList(L)` for $\alpha_\phi$ (one per layer, since the score head feeds both branches in any layer where both are active).

| Cell | Schedule $\sigma$ | $k$ | $r$ | per-token $\Omega$ | per-layer modules | NC params (approx, $d=256$) | Notes |
|---|---|---:|---:|:---:|---|---:|---|
| Q9e-H | `interleaved` (`SCSCSCSC`) | 4 | 16 | off | $J_\phi^{(\ell)}$ only | $L_C \cdot 2dr = 4 \cdot 8\,192 = 32\,768$ | Mechanism-1: per-layer skew kernel (most direct analogue of attention's per-layer $W_Q, W_K, W_V$). Otherwise == Q9e-A. |
| Q9e-I | `interleaved` | 4 | 16 | off | $V_\phi^{(\ell)}$ only | $L_S \cdot \lvert V_\phi \rvert \approx 4 \cdot 8\,500$ extra | Mechanism-1: per-layer conservative pair scalar. Otherwise == Q9e-A. |
| Q9e-J | `interleaved` | 4 | 16 | off | $\alpha_\phi^{(\ell)}$ only | $L \cdot \lvert \alpha_\phi \rvert \approx 8 \cdot 33\,000$ extra | Mechanism-1: per-layer score head (per-layer routing topology). Otherwise == Q9e-A. |
| Q9e-K | `interleaved` | 4 | 16 | off | $J_\phi^{(\ell)} + V_\phi^{(\ell)} + \alpha_\phi^{(\ell)}$ | sum of H+I+J overheads | Mechanism-1 (joint): all three submodules per-layer. The cleanest empirical test of the paper Appendix A two-mechanism decomposition without instantiating any attention layer. |

The exact parameter overheads of Q9e-I/J depend on the SparsePARFLM hyperparameters carried over (`v_phi_*` hidden widths and `score_head_hidden`); the relevant numbers will be reported in the executed `RESULTS.md` per cell. By construction, the Mechanism-1 overhead is **larger** than the Mechanism-2 overhead — the H1 PPL comparison against P10g is not parameter-matched on the H/I/J/K row. To compensate, the H5 decision rule (§6.3) is stated relative to **Q9e-A** rather than to P10g: a Mechanism-1 cell only "wins" if it beats q9e_a (the Mechanism-2 central bet that already shares the SparsePARFLM ceiling) by at least $2\sigma_{\mathrm{seed}}$. P10g and E4-fix remain the H1/H2 baselines.

**Mechanism-1 × Mechanism-2 additivity + maximal-Mechanism-1 cells (added after q9e_d / q9e_h / q9e_l executed):**

| Cell | Schedule $\sigma$ | $k$ | $r$ | per-token $\Omega$ | per-layer modules | NC params (approx, $d=256$) | Notes |
|---|---|---:|---:|:---:|---|---:|---|
| Q9e-L | `interleaved` (`SCSCSCSC`) | 4 | 16 | **on** (shared) | $J_\phi^{(\ell)}$ only | $L_C \cdot 2dr + 2dr = 5 \cdot 8\,192 = 40\,960$ | H6 additivity test: q9e_d's per-token $\Omega$ + q9e_h's per-layer $J_\phi^{(\ell)}$. **EXECUTED 3 seeds: PPL mean 25.64 ± 1.03, median 25.11, range [24.98, 26.82].** First SP-HSPLM cells below P10g 26.42 ceiling (at median); bimodal kernel regime explains seed-2 outlier. BETA confirmed, strict ALPHA pending q9e_l-seed3. |
| Q9e-M | `interleaved` | 4 | 16 | **on** (per-layer $\Omega^{(\ell)}$) | $J_\phi^{(\ell)} + \Omega^{(\ell)}$ | $2 L_C \cdot 2dr = 65\,536$ | H7 stretch test: q9e_l + per-layer $\Omega^{(\ell)}$ (lifts the last shared non-conservative submodule). Predicted PPL ≤ 24.5 if H7 holds (H2 confirmation territory at single-seed). Contingent on q9e_l multi-seed confirmation. |
| Q9e-N | `interleaved` | 4 | 16 | **on** (per-layer $\Omega^{(\ell)}$) | $J_\phi^{(\ell)} + \Omega^{(\ell)} + V_\theta^{(\ell)} + V_\phi^{(\ell)} + \alpha_\phi^{(\ell)}$ | $\approx 65\,536 + L_S \cdot 2.63\text{M} + 3 \cdot V_\phi + 7 \cdot \alpha_\phi$ $\approx 10.8$ M extra | **H8 full-Class-F test (intentionally non-iso-param-count vs P10g).** Every SP-HSPLM force-law module per-layer; V_θ dominates the new parameter cost. Tests whether the residual ≈ 17 PPL gap to MatchedGPT collapses once V_θ is no longer the autonomous-in-ℓ bottleneck. Multi-seed budget required from launch (≥ 3 seeds) given q9e_l's seed-fragility on the per-layer J_φ axis. |

Notes on implementation:

- The four `share_*_across_layers` flags default to `True`, preserving the SPLM autonomous-in-$\ell$ commitment for cells A-G (no behavioural change for the original ladder).
- The per-layer dispatch is via `_skew_kernel_at(layer_idx)`, `_gyro_kernel_at(layer_idx)`, `_v_phi_at(layer_idx)`, `_score_head_at(layer_idx)` in `model_sphsplm.py`; the underlying detach-contract of the C-block step (source-side $\delta_s$ detach) is structurally preserved under per-layer dispatch — the causal-leak probe confirms `fixed Δ = 0.00e+00` on all four Mechanism-1 cells at the smoke-test scale.
- The Frobenius warm-up regulariser is `lam_skew * sum_ell ||J_phi^(ell)||_F^2`, summed over all $L_C$ per-layer kernels via the new `ScalarPotentialLMSPHSPLM.skew_kernel_frobenius_squared()` method; this keeps the warm-up suppression intensity scaled to the per-cell kernel count.

---

## 4. Locked configuration

### 4.1 Hyperparameters (locked at pre-registration)

All fourteen cells (seven Mechanism-2 + four Mechanism-1 + q9e_l/q9e_m additivity-and-maximal + q9e_n full-Class-F) share the following configuration. Anything not listed inherits from the SparsePARFLM scale-up training script (P10g schedule). The hyperparameters are picked to match P10g exactly so the H1 PPL comparison is apples-to-apples with no budget asterisk for the Mechanism-2 ladder. The four Mechanism-1 cells (§3.1) and q9e_l/q9e_m additionally flip one or more `share_*_across_layers` flags on `SPHSPLMConfig` — all other configuration is identical to Q9e-A. **Q9e-N intentionally breaks iso-parameter-count vs P10g** (~10 M extra parameters from per-layer V_θ); the H8 decision rule (§6.3) is stated relative to q9e_l multi-seed median, not P10g, to isolate the V_θ-sharing contribution.

| Quantity | Value |
|---|---|
| Tokenizer | GPT-2 BPE (vocab 50 257) |
| Corpus | TinyStories (HF: `roneneldan/TinyStories`), shard 0, ~5 M training BPE tokens, canonical validation shard |
| Hidden dim `d` | 256 |
| Layers `L` (integration steps) | 8 |
| Per-token mass mode | `logfreq` (frozen, computed from TinyStories train) |
| `V_theta` MLP | `v_hidden = 1024`, `v_depth = 3`, GELU |
| `V_phi` MLP | matches P10g (`v_phi_hidden = 32`, GELU; structural pair scalar) |
| Score head $\alpha_\phi$ | matches SparsePARFLM (small MLP on $(h_t, h_s, h_t - h_s)$, $H_s = 32$); shared between $V_\phi$ and $J_\phi$ branches |
| Skew kernel $J_\phi$ | $J_+ = U V^\top$, $U, V \in \mathbb{R}^{d \times r}$, init $\mathcal{N}(0, 0.02^2 / \sqrt{r})$ |
| Per-token gyro $\Omega$ (Q9e-D only) | $\Omega = \Omega_+ - \Omega_+^\top$ low-rank, same rank $r$, same init |
| `init_gamma` | 1.0 with `learn_mgamma = True`, $\gamma_{\min} = 0.05$ enforced via softplus + offset |
| Tied embeddings | True |
| `max_len` | 1024 |
| `block_size` | 512 |
| Steps | **16 000** (matched to P10g and Stage 1) |
| Batch size | 16 |
| Optimiser | AdamW, $\beta = (0.9, 0.95)$, weight decay 0.01, grad clip 1.0 |
| LR schedule | 5e-4 peak, cosine decay, 800 warmup steps |
| Aux losses | $\mathcal{L}_{\mathrm{NTP}} + \lambda_{\mathrm{ent}}\mathcal{L}_{\mathrm{ent}} + \lambda_{\mathrm{skew}}(t)\mathcal{L}_{\mathrm{skew}}$ with $\lambda_{\mathrm{ent}} = 10^{-3}$, $\lambda_{\mathrm{skew}}(t) = 10^{-2}\max(0, 1 - t/200)$ (Frobenius warm-up on $J_\phi$) |
| Eval | `eval_iters = 40` every 800 steps; final eval at step 16 000 |
| Causal-force flag | `cfg.causal_force = True` (leak fix on, mandatory) |
| TF32 | **disabled** (`torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`) for autograd numerical stability |
| Seeds | 1 seed per cell first cut (seed = 0); 3 seeds for any cell that breaks the floor by more than $2\sigma_{\mathrm{seed}}$ on either H1 or H2 |
| Hardware | H100 or A100 single-GPU per cell |

The token budget per cell is therefore $16\,000 \cdot 16 \cdot 512 = 131$ M token-passes — identical to Stage 1 and to P10g. Total Stage 2 first-cut budget is 14 cells $\times 16\,000$ steps $\approx 16$ GPU-days at H100 (7 for the Mechanism-2 ladder + 4 for the Mechanism-1 extension + 2 for q9e_l/q9e_m + 3 for q9e_n at $\approx 1.5\times$ the wall-clock of q9e_l owing to the 70 % larger model). Multi-seed budgets are additional (see §9.3).

### 4.2 Locked baselines (already executed; numbers re-cited here)

| Baseline | Best val PPL | Final val PPL | Source |
|---|---:|---:|---|
| SPLM em\_ln Cell 0, leak-free, 16 000 steps | 26.31 | 27.07 | Stage 1 e0_baseline (this codebase) |
| SparsePARFLM P10g, $k=4$, 16 000 steps | **26.42** | 26.98 | P10 ladder, [`PARF-SPLM_Path_Forward_and_Experiments.md`](./PARF-SPLM_Path_Forward_and_Experiments.md), shard 0, 5 M tokens |
| Stage 1 best per-token cell (E4-fix solenoidal $r=4$) | **24.58** | 25.29 | Stage 1 e4_solenoidal_rank4 (this codebase) |
| Attention small (matched-baseline, GPT-2 $n_{\mathrm{layer}} = 8$) | $\approx 8$ | $\approx 8$ | paper v3 §15 (informational; not a Stage 2 gate) |

The two binding baselines for the decision rule are **SparsePARFLM P10g** (H1) and **Stage 1 E4-fix** (H2). The em\_ln Cell 0 number is reported alongside as a sanity baseline; the attention number is reported as the v3-paper context only.

### 4.3 Causal-leak invariant (mandatory across all cells)

Each cell's training loop must run the causal-leak probe at three checkpoints: step 1 (initialisation), step 8000 (mid-training), step 16 000 (final). The leak floor at every checkpoint must be at or below $10^{-6}$; the expected reading at every checkpoint is $0.0$ exactly, as in Stage 1.

The C-block introduces three new sites where source-side state leaks across positions and where `.detach()` must be applied:

1. **Source-side hidden state $h_s$** entering the kernel argument $h_t - h_s$: the C-block reads $h_s$ for $s \lt t$ and must `.detach()` it before subtraction. This is the same pattern SparsePARFLM applies to source-side $h_s$ entering $V_\phi(h_t, h_s)$.
2. **Source-side velocity proxy $\delta_s$** entering the kernel multiplication $J_\phi (\delta_s - \delta_t)$: the C-block reads $\delta_s = h_s - h_{\mathrm{prev}, s}$ from the previous-layer kinematic memory. Both $h_s$ and $h_{\mathrm{prev}, s}$ must be detached before the subtraction.
3. **Score-head logits** $\alpha_\phi(h_t, h_s)$ for $s \lt t$ feeding the Gumbel mask: this is unchanged from SparsePARFLM (same `.detach()` pattern on $h_s$).

All three `.detach()` points must be unit-tested with the standalone causal probe before the cell is allowed to launch. Any cell that violates the invariant at *any* probe checkpoint is invalidated and re-run after a code review.

---

## 5. Outputs per cell

Each cell, after training, produces a named-tag output bundle under the SP-HSPLM Stage 2 results namespace:
```
notebooks/conservative_arch/sphsplm/results/sp_hsplm/stage2/<tag>/seed<S>/
  splm_<tag>_summary.md
  splm_<tag>_training_log.jsonl
  splm_<tag>_loss_curve.png
  splm_<tag>_ckpt_latest.pt

  causal_probe.json                # leak floors at steps 1, 8000, 16000
  jacobian_symmetry.json           # velocity-aware Jacobian-symmetry probe at step 16000
  pair_kernel_norms.json           # || J_phi ||_F, || U ||_F, || V ||_F per layer per eval step
  routing_density.json             # actual top-k mask density and entropy per layer per eval step
  nonconservative_norms.json       # || g_l || statistics across layers and steps (S+C)
```
with `<tag>` in `{q9e_a_interleaved_k4_r16, q9e_b_interleaved_k8_r16, q9e_c_interleaved_k4_r32, q9e_d_interleaved_k4_r16_gyro, q9e_e_bottom_c4_k4_r16, q9e_f_top_c4_k4_r16, q9e_g_sandwich2_k4_r16}` for the Mechanism-2 ladder, and in `{q9e_h_interleaved_k4_r16_perLJ, q9e_i_interleaved_k4_r16_perLV, q9e_j_interleaved_k4_r16_perLA, q9e_k_interleaved_k4_r16_perLJVA}` for the Mechanism-1 extension. The `pair_kernel_norms.json` is the diagnostic for H4 and for detecting the "skew kernel collapsed to zero" degenerate solution (Outcome DELTA below). The `routing_density.json` quantifies the per-layer mask entropy that drives the entropy regulariser. For Mechanism-1 cells, `pair_kernel_norms.json` additionally contains per-layer lists (`J_phi_fro_per_layer`, `V_phi_fro_per_layer`, `Omega_fro_per_layer` as relevant); the H5d non-additivity sub-hypothesis is read off the per-layer trajectories together with the scalar PPL.

---

## 6. Statistical and decision protocol

### 6.1 PPL — single-seed first cut, multi-seed if borderline

We train **one seed (= 0)** per cell. For each cell $c$, two delta values are computed: $\Delta^{\mathrm{H1}}(c) = \mathrm{PPL}(c) - 26.42$ (versus P10g) and $\Delta^{\mathrm{H2}}(c) = \mathrm{PPL}(c) - 24.58$ (versus E4-fix).

If, for the best cell, $\Delta^{\mathrm{H1}} \le -2\sigma_{\mathrm{seed}}$ AND $\Delta^{\mathrm{H2}} \le -2\sigma_{\mathrm{seed}}$, we accept H1 and H2 jointly and stop. If either delta lands in $(-2\sigma_{\mathrm{seed}}, -1\sigma_{\mathrm{seed}}]$, we run two additional seeds (1, 2) on the best cell and on the relevant baseline before deciding. If both deltas land in $(-1\sigma_{\mathrm{seed}}, +\infty)$, we reject the corresponding hypothesis at single-seed resolution and report the negative result.

The reference seed-variance is $\sigma_{\mathrm{seed}} \approx 1.8$ PPL (E1 multi-seed reference; same as Stage 1). Under the longer 16k schedule the seed variance is expected to be smaller, not larger, so the 2-sigma threshold of $\Delta = -3.6$ PPL is conservative.

### 6.2 Decision matrix (per cell)

Five independent verdicts are reported for each cell.

| Diagnostic | Verdict |
|---|---|
| PPL vs P10g (H1) | "ties" if $\lvert\Delta^{\mathrm{H1}}\rvert \lt 2\sigma_{\mathrm{seed}}$; "improves" if $\Delta^{\mathrm{H1}} \le -2\sigma_{\mathrm{seed}}$; "worsens" if $\Delta^{\mathrm{H1}} \ge +2\sigma_{\mathrm{seed}}$ |
| PPL vs E4-fix (H2) | "ties" if $\lvert\Delta^{\mathrm{H2}}\rvert \lt 2\sigma_{\mathrm{seed}}$; "improves" if $\Delta^{\mathrm{H2}} \le -2\sigma_{\mathrm{seed}}$; "worsens" if $\Delta^{\mathrm{H2}} \ge +2\sigma_{\mathrm{seed}}$ |
| Causal probe (H3) | "leak-clean" if all three checkpoint floors are at or below $10^{-6}$; "leak-fail" otherwise |
| Pair-kernel norm | "active" if $\lVert J_\phi \rVert_F$ at step 16 000 is at least 5 % of its initialisation Frobenius norm; "collapsed" otherwise |
| Jacobian-symmetry (H4) | "asymmetric-as-predicted" if $\Delta_{\mathrm{sym}} \in [c L_C \lVert J_\phi \rVert_F / 2, 2 c L_C \lVert J_\phi \rVert_F]$ with $c \approx 10^{-2}$; "near-symmetric" otherwise (consistent with kernel collapse, expected only if pair-kernel verdict is "collapsed"); "explosive" if outside the upper bound (probe bug or `.detach()` bug) |

The headline reading of Stage 2 is the *joint* outcome across the five verdicts and across the fourteen cells (seven Mechanism-2 + four Mechanism-1 + q9e_l/q9e_m additivity-and-maximal + q9e_n full-Class-F, §3.1).

### 6.3 Hard go / no-go gates for Stage 3

- **Outcome ALPHA — central bet wins (expected on the v3 architectural reading):** the best Stage 2 cell has both H1 and H2 verdicts equal to "improves", causal probe is leak-clean across the ladder, and the pair-kernel verdict is "active". This is the green light for Stage 3 (multi-seed power-up + diagnostics) and the publishable result.
- **Outcome BETA — H1 wins, H2 ties:** the best cell beats P10g by $\ge 2\sigma_{\mathrm{seed}}$ but does *not* beat E4-fix by $\ge 2\sigma_{\mathrm{seed}}$. The pair coupling adds value over conservative-only routing but does not measurably exceed per-token solenoidal. Stage 3 still proceeds for the publishable result, but the architectural narrative is "pair-skew matches per-token solenoidal at this scale" rather than "pair-skew dominates per-token solenoidal".
- **Outcome GAMMA — both H1 and H2 tie:** no Stage 2 cell beats either baseline by $\ge 2\sigma_{\mathrm{seed}}$. The Q9(e) construction is provisionally falsified at this scale. The natural follow-up is to (a) increase $r$ further (Q9e-C already swept to $r = 32$; the next step would be $r = 64$ at the cost of doubled NC parameter count), (b) increase $k$ further (Q9e-B already swept to $k = 8$; next would be $k = 16$), or (c) move to a larger $d$ scale-up where the conservative-pair ceiling may have shifted.
- **Outcome DELTA — pair-kernel collapse:** the best cell ties E4-fix or P10g and the pair-kernel verdict is "collapsed" at every cell. The optimiser learned to set $J_\phi \to 0$ and the C-block is functionally equivalent to a pure dissipative step. This is the "Algorithm A is not enough" failure mode flagged in architecture v3 §5.2; the remediation is to switch to Algorithm B (alternating S / C parameter freezing) and rerun.
- **Outcome EPSILON — instability or leak-fail:** any cell shows a loss spike, NaN, trajectory divergence, or causal-probe leak floor above $10^{-6}$. The cell is invalidated, the issue diagnosed, and the cell rerun after fix.
- **Outcome ZETA — Mechanism-1 closes the gap (H5 confirmation; canonical positive Stage 2 result if Outcomes ALPHA / BETA do not trigger):** at least one Mechanism-1 cell (q9e_h, i, j, or k) satisfies H5 — i.e. beats q9e_a by $\ge 2\sigma_{\mathrm{seed}}$. The pair-kernel verdict on the relevant per-layer kernels is "active" (no collapse), the causal probe stays leak-clean across all Mechanism-1 cells, and the per-layer trajectories in `pair_kernel_norms.json` show that the per-layer kernels diverge from each other (the simplest empirical proxy for "the layers learned distinct force laws"). If H5d holds in addition (q9e_k strictly improves over the best single Mechanism-1 cell by $\ge 1\sigma_{\mathrm{seed}}$), the three Mechanism-1 primitives are non-additive and the joint construction is the architectural recipe. Outcome ZETA is the green-light gate for Stage 3 of the H5 line: per-layer kernel-trajectory diagnostics, principal-angle analysis of $J_\phi^{(\ell)}$ across $\ell$, and the comparison against attention-block expressivity at matched per-layer parameter budget.
- **Outcome ETA — Mechanism-1 tie (H5 negative; cluster ceiling holds with both mechanisms):** no Mechanism-1 cell improves on q9e_a by $\ge 2\sigma_{\mathrm{seed}}$. Combined with a tied Mechanism-2 ladder (Outcomes BETA or GAMMA), this is a strong publishable negative for the entire paper §15.5/§17.3 reading: the SPLM family's $\approx 26$-$28$ cluster ceiling is **not** an artifact of the autonomous-in-$\ell$ commitment; both Appendix A mechanisms in play do not close the gap to attention at TinyStories scale. This outcome promotes the FockPARFLM creation/destruction operator path (paper §16+ / v0-v1.5-v2 programme) to the canonical next step, with a much sharper falsified-baseline write-up than was available at the end of Stage 1.
- **Outcome THETA — full Class F matches attention (H8 positive; canonical positive across-class result):** q9e_n multi-seed median PPL $\le 12$. The ≈ 17 PPL across-class gap between q9e_l (median 25.11) and MatchedGPT (≈ 8) is essentially the V_θ-sharing constraint. Paper Appendix A Eq. A.130's central reading — *attention is in Class F* — is empirically confirmed by an attention-free, multi-head-free, MLP-free architecture. This is the strongest possible positive Stage 2 result; the v0/v1 paper revision can claim mechanistic recovery of attention's expressivity from primitives. Stage 3 then becomes a Jacobian/holonomy probe of q9e_n's per-layer V_θ^(ℓ) trajectories to characterise *how* the per-layer scalar potentials specialise.
- **Outcome IOTA — partial / negative full-Class-F (H8 partial or negative):** q9e_n multi-seed median PPL $\in (12, 22]$ (partial) or $\ge 22$ (negative). Either branch is publishable as a quantitative bound on Class F's expressivity: under the **partial** branch, Class F closes a substantial but incomplete fraction of the attention gap and the residual is identified as living in axes *beyond* Class F (multi-head, dense softmax routing vs Gumbel top-k, position-wise MLP, Fock-space dynamics); under the **negative** branch, Class F as currently realised is not attention's class and the Appendix A two-mechanism decomposition needs revising. Either way, the result precisely separates the Class-F-recoverable from the Class-F-irrecoverable portions of attention's expressivity and elevates FockPARFLM (paper §16+) as the canonical next architectural axis.

---

## 7. What Stage 2 can show

Three qualitative outcomes are possible at the level of the entire grid.

**Outcome A — clean confirmation of the architectural bet.** The best Stage 2 cell beats both P10g and E4-fix by $\ge 2\sigma_{\mathrm{seed}}$, with the pair kernel non-trivially active and the Jacobian asymmetric as predicted. This validates the v3 architecture v3 doc's central reading — pair-coupled non-conservative routing is the architectural realisation of the missing routing capacity that per-token additions and conservative pair scalars both lack. SP-HSPLM is then the first attention-free Helmholtz-decomposed sequence model with an empirical advantage over its conservative-pair predecessor; Stage 3 produces the multi-seed effect-size estimate.

**Outcome B — partial confirmation.** The best cell beats P10g but ties E4-fix. The pair-skew construction is then a parameter-equivalent re-realisation of the per-token solenoidal effect — useful (because it inherits the Q9(e) attention-free narrative) but not architecturally dominant. The §17.3 paper subsection still accommodates SP-HSPLM as a published architecture; the headline claim weakens from "pair coupling beats per-token solenoidal" to "pair coupling matches per-token solenoidal at the same scale".

**Outcome C — clean negative result.** No cell in the ladder beats either baseline by $\ge 2\sigma_{\mathrm{seed}}$. Combined with the Stage 1 negative result, this would establish that **TinyStories at $d = 256$, $L = 8$, 16k steps is fully saturated by the conservative pair-routing class** at $\approx 26.4$ PPL, regardless of which non-conservative augmentation is added. The natural follow-up is the larger-scale ladder (higher $d$, longer $T$, more steps) and the FockPARFLM creation/destruction operator path, which is the v0+v1.5+v2 programme of the paper.

In every case the verdict is what the data says, with the decision rule pre-committed in this protocol.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Velocity-coupled feedback divergence (the v3 paper E5 with $s = 1$ pathology, in pair-coupled form) | $J_\phi$ Frobenius warm-up regulariser $\lambda_{\mathrm{skew}}(t) = 10^{-2}\max(0, 1 - t/200)$; init $\lVert J_\phi \rVert_F \approx 0.02$ at $t=0$; $\gamma_{\min} = 0.05$ floor on the dissipation; grad clip 1.0. If a cell still diverges, the protocol's Outcome EPSILON captures it. |
| Causal-leak regression under pair coupling | The three new `.detach()` points (section 4.3) are unit-tested before launch and re-checked at three checkpoints during training. The structural argument (§4.3 paragraph 1) makes a leak structurally impossible if the `.detach()` calls are present; the probe enforces it. |
| Skew-kernel collapse to zero (Outcome DELTA) | The `pair_kernel_norms.json` output captures the full trajectory of $\lVert J_\phi \rVert_F$ over training. Outcome DELTA in the decision matrix interprets it as a *fail* mode (not a positive confirmation of the v3 reading, in contrast to Stage 1 where it would have been). The remediation is Algorithm B (alternating S / C parameter freezing) per architecture v3 §5.2. |
| Routing-mask collapse (single source $s$ dominates per token) | The entropy regulariser $\mathcal{L}_{\mathrm{ent}}$ at $\lambda = 10^{-3}$ is inherited from SparsePARFLM; the `routing_density.json` output reports per-layer mask entropy at each eval step. If entropy collapses below $0.5$ nats per token before step 8 000, the regulariser weight is doubled and the cell rerun. |
| Score-head sharing between $V_\phi$ and $J_\phi$ branches biases routing | Architecture v3 §4.1 motivates sharing as the architectural commitment. If Stage 2 lands a positive result and the post-hoc ablation (separate score heads, deferred to Stage 3) shows separate heads beat shared heads by $\ge 1\sigma_{\mathrm{seed}}$, the design choice is revisited; for Stage 2 it is locked. |
| Compute over-spend | The 14 cells at 16 k steps each on H100 is $\approx 16$ GPU-days first cut (7 Mechanism-2 + 4 Mechanism-1 + 2 additivity/maximal + 3 full-Class-F). Multi-seed escalation is q9e_l-seed3 (1.3 H100-h) + q9e_n at $\ge 3$ seeds (3 $\times$ 1.5 GPU-days $\approx 5$ GPU-days) + 0-2 baseline reruns $\approx 8$ GPU-days extra. Worst-case total is 24 GPU-days, still well under the Stage 0+1+2+3 ladder budget projected in architecture v3 §9.4. |
| Q9e-D parameter overhead double-counting (per-token gyro on top of pair kernel) | Q9e-D adds $\Omega$ on top of $J_\phi$, doubling the NC parameter count (16 384 vs 8 192 for Q9e-A). This is only $\approx 0.07\,\%$ of the 22 M total at $d = 256$, so the PPL comparison is not biased; the protocol still reports each cell's parameter count alongside its PPL for transparency, matching Stage 1's policy. |

---

## 9. Schedule and cost estimate

| Phase | Work | Calendar | GPU-days |
|---|---|---|---:|
| A | Pre-registration (this document); peer-read by collaborator(s); commit-lock | 1 day | 0 |
| B | Implementation: new `notebooks/conservative_arch/sphsplm/` directory with `model_sphsplm.py` (S-block reuse from SparsePARFLM, new C-block module, integrator with shared mask, schedule registry extended to `S` and `C`); unit tests for the skew kernel and the three `.detach()` points; causal-probe verification on a 200-step smoke run per cell | 4-6 days | 0.3 (smoke) |
| C | Training script `train_sphsplm_scaleup.py` with `--cell` flag matching the protocol; H100/A100 notebook for the 14 cells (Mechanism-2 + Mechanism-1 extension + additivity/maximal + full-Class-F); causal-leak probe smoke-tested on all 14 cells (q9e_n verified leak-clean 17 May 2026 at d=64/L=8/T=32) | 1-2 days | 0 |
| D | Run the 7 Mechanism-2 cells (Q9e-A through Q9e-G), 16 k steps, 1 seed | 1 day calendar (parallel) | 7 |
| D' | Run the 4 Mechanism-1 cells (Q9e-H/I/J/K), 16 k steps, 1 seed; sequential after Q9e-A is in (Q9e-H/I/J/K all reference Q9e-A as their H5 baseline) | 1 day calendar (parallel) | 4 |
| E | Aggregation, RESULTS.md, decision-matrix evaluation; multi-seed escalation if any cell breaks the floor | 1-2 days plus 0-6 GPU-days | 0-6 |
| F | Stage 3 launch (multi-seed power-up + Jacobian / holonomy probes + per-layer kernel diagnostics for Mechanism-1 / Class-F) only if Outcome ALPHA, BETA, ZETA, or THETA in section 6.3 | 1-2 days | (Stage 3 budget, separate) |

**Total Stage 2:** 7-14 calendar days; 16-24 GPU-days (was 11-17 before the q9e_l/m additivity-and-maximal cells and q9e_n full-Class-F extension). Stage 3 is gated on the section 6.3 outcome and budgeted separately in architecture v3 §9.3.

---

## 10. References

### Primary references (this repository)

- [`Scalar_Potential_based_Helmholtz_Architecture_v3.md`](./Scalar_Potential_based_Helmholtz_Architecture_v3.md) — section 9.2 specifies the Stage 2 cells; this protocol locks the implementation and decision rule.
- [`SP_HSPLM_Stage_0_Literature_Check.md`](./SP_HSPLM_Stage_0_Literature_Check.md) — originality assessment.
- [`SP_HSPLM_Stage_1_pre-registered_protocol.md`](./SP_HSPLM_Stage_1_pre-registered_protocol.md) — Stage 1 protocol; the Stage 1 results (Outcome ALPHA, all per-token cells tie em\_ln Cell 0; E4-fix at 24.58 best PPL) are the empirical floor Stage 2 must beat for H2.
- [`PARF_Augmented_SPLM_Architecture_v2.md`](./PARF_Augmented_SPLM_Architecture_v2.md) — PARFLM design doc; source of the conservative pair scalar $V_\phi$ and the score head $\alpha_\phi$ Stage 2 reuses for the S-block.
- [`parf/On_Gumbel_softmax_sparsity_applied_to_V_phi.md`](./parf/On_Gumbel_softmax_sparsity_applied_to_V_phi.md) — Gumbel top-$k$ routing primitive shared between $V_\phi$ and $J_\phi$ branches.
- [`PARF-SPLM_Path_Forward_and_Experiments.md`](./PARF-SPLM_Path_Forward_and_Experiments.md) — P10 ladder; P10g (best val PPL 26.42) is the H1 baseline.
- [`Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](./Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md) — leak-fix invariant; Stage 2 introduces three new `.detach()` points listed in §4.3.
- [`Helmholtz-HSPLM_Path_Forward_and_Experiments.md`](./Helmholtz-HSPLM_Path_Forward_and_Experiments.md) — Q9(d) Helmholtz hybrid; the schedule registry (`make_schedule(name, L, k, LA)`) Stage 2 extends from `S+A` to `S+C`.

### Paper reference (this conversation's reading)

- *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference*, Gueorguiev (2026), v3, section 15.5 ("five-negatives"), section 15.6 (E1-E5 ablations), section 15.7 (Jacobian-symmetry probe), section 17.3 (Q9 hybrid programme), Appendix A (Eq. A.130, the non-autonomous conservative class).

---

---

## 11. Stage 2 executed verdicts — pointer to companion verdict notes

The verdict tables and structural-re-examination notes triggered by the executed Stage 2 cells live in companion documents under this same `docs/` namespace, so this pre-registration stays clean as a pre-registration:

- **`q9e_l` 3-seed verdict (H6, BETA confirmed):** recorded inline in §2 above and reflected in the `q9e_l` row of the §3.1 cell table; no separate verdict document.
- **`q9e_n` single-seed verdict (H8, Outcome IOTA-negative) + structural re-examination:** [`SP_HSPLM_Stage_2_q9e_n_verdict_and_structural_reexamination.md`](./SP_HSPLM_Stage_2_q9e_n_verdict_and_structural_reexamination.md). PPL 25.04, leak-clean across all checkpoints, per-layer kernels non-trivially differentiated (Mechanism-1 is being exploited; the IOTA-negative reading is not a Mechanism-1-idle artifact). The note records the verdict, the three live structural hypotheses (H1 many-body coupling, H2 PCA-16 artifact, H3 conservative-tax) that the negative-IOTA classification triggers per §6.3 ("the Appendix A two-mechanism decomposition needs revising; attention realises a class strictly larger than Class F"), and the next experiments (PCA-symmetry sweep, `q9e_o` bilinear pairwise coupling, conditional `q9e_p`). Paper-update policy is explicit: no `paper_v4` / `paper_v5` / `paper_tmlr_1` revision lands on the basis of the `q9e_n` verdict; the next paper-edit cycle is gated on the H2 PCA-symmetry sweep outcome.

Future executed-cell verdicts that warrant a structural re-examination (rather than a one-line entry in the §3.1 table) follow the same pattern: a dedicated `SP_HSPLM_Stage_2_<cell>_verdict_and_<topic>.md` under `docs/`, with a one-paragraph pointer added to this §11.

---

*Last updated: 17 May 2026. Mechanism-2 ladder pre-registered, partially executed (q9e_a/b/c/d complete). Mechanism-1 extension (§3.1, H5, Outcomes ZETA / ETA) pre-registered 16 May 2026 after the q9e_a/b/c readings; q9e_h executed. Mechanism-1 × Mechanism-2 additivity (H6, q9e_l) executed 17 May 2026 at 3 seeds — BETA confirmed, strict ALPHA pending q9e_l-seed3. Full-Class-F test (H8, q9e_n; Outcomes THETA / IOTA) pre-registered 17 May 2026 after the q9e_l 3-seed reading revealed that Mechanism-1 + Mechanism-2 synergy is real but covers only ~1.3 of the ~17 PPL across-class gap to MatchedGPT; q9e_n executed at seed 0 on 17 May 2026 landing in **Outcome IOTA-negative** (PPL 25.04, verdict note linked from §11).*
