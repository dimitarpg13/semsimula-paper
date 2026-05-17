# SP-HSPLM Stage 2 — `q9e_n` Verdict and Structural Re-examination

> **Companion to:** [`SP_HSPLM_Stage_2_pre-registered_protocol.md`](./SP_HSPLM_Stage_2_pre-registered_protocol.md) — this note is the verdict + structural re-examination triggered by the `q9e_n` result; the protocol holds the pre-registration of the H8 decision rule and Outcomes THETA / IOTA.
> **Status:** Draft, locked at `q9e_n-seed0` verdict.
> **Date:** 2026-05-17.
> **Author:** Dimitar Gueorguiev (with Claude as scribe / structural-reading sanity-checker).
> **Paper-update policy:** No paper revision (`paper_v4`, `paper_v5`, `paper_tmlr_1`) lands on the basis of this note. See §6 for the rationale and the explicit gate the next paper-edit cycle is waiting on.

---

## 1. Verdict

### 1.1 `q9e_n` configuration (as executed)

| Field | Value |
|---|---|
| Experiment | SP-HSPLM Stage 2 (Q9(e) pair-skew cell ladder) |
| Cell | `q9e_n` (full Class-F: all five Mechanism-1 modules per-layer indexed) |
| Schedule | `SCSCSCSC` |
| Mode | scaleup |
| Corpus | TinyStories (cap 5,000,000 train tokens) |
| `d` / `L` / `v_hidden` / `max_len` | 256 / 8 / 1024 / 1024 |
| `k` / `r` / `gyro` / `ln_after_step` | 4 / 16 / on / True |
| Mechanism-1 per-layer modules | `J_phi`, `Omega`, `V_theta`, `V_phi`, `alpha_phi` |
| Module param breakdown | `V_theta` = 10,502,148 ; `V_phi` = 14,472 ; `alpha_phi` = 197,128 ; `skew` = 32,768 ; `gyro` = 32,768 |
| **Params (total)** | **23,907,223** |
| Steps / batch / block | 16,000 / 16 / 512 |
| Seed | 0 |
| Elapsed | 4,686 s (1.30 h, single H100) |
| **Final val loss** | **3.220635** |
| **Final val PPL** | **25.04** |
| Final $\gamma$ | 0.9909 |

### 1.2 Causal-leak probe history

| Step | Label | `max_logit_delta_past` | Verdict |
|---:|---|---:|---|
| 1 | init | 0.00e+00 | leak-clean |
| 8,000 | mid | 0.00e+00 | leak-clean |
| 16,000 | final | 0.00e+00 | leak-clean |

H3 (causal-leak invariant) holds across all three checkpoints. The `q9e_n` per-layer dispatch and the V_θ per-layer indexing preserved the leak-fix invariant of `SP_HSPLM_Stage_2_pre-registered_protocol.md` §4.3 by construction.

### 1.3 Per-layer pair-kernel norms — Mechanism-1 is being exploited

The per-layer Frobenius norms reported below are non-trivially differentiated across layers, which is the simplest empirical proxy for "the per-layer kernels learned distinct force laws" (the auxiliary condition on Outcomes ZETA / THETA in the pre-registration). Mechanism-1 is therefore not sitting idle — the PPL plateau cannot be attributed to per-layer indexing failing to engage.

| Norm | Aggregate | Per-layer (len = $L_C = 4$) |
|---|---:|---|
| $\lVert J_\phi \rVert_F$ | 2.4987 | [1.8402, 1.7913, 2.9778, 3.3854] |
| $\lVert U \rVert_F$ | 2.2857 | [2.0413, 2.0757, 2.3601, 2.6656] |
| $\lVert V \rVert_F$ | 2.2843 | [2.0516, 2.0396, 2.3604, 2.6856] |
| $\lVert J_\phi \rVert_F^{\mathrm{total}}$ | 5.1888 | — |
| $\lVert \Omega \rVert_F$ | 4.5600 | [6.0823, 2.3101, 4.3144, 5.5332] |
| $\lVert \Omega_U \rVert_F$ | 2.3413 | [2.8397, 1.5711, 2.3512, 2.6029] |
| $\lVert \Omega_V \rVert_F$ | 2.3520 | [2.8015, 1.7192, 2.3341, 2.5531] |
| $\lVert \Omega \rVert_F^{\mathrm{total}}$ | 9.5687 | — |

Reading: $J_\phi^{(\ell)}$ ramps monotonically with depth (1.84 → 3.39 across the four C-blocks at $\ell \in \{1, 3, 5, 7\}$); $\Omega^{(\ell)}$ is bimodal (high at $\ell = 1$ at 6.08, drops sharply at $\ell = 3$ to 2.31, then grows). Both the U/V low-rank factor norms and the per-layer scalar potentials carry layer-specific structure. None of the per-layer kernels collapsed.

### 1.4 Pre-registered outcome — IOTA-negative

Per protocol §6.3 the H8 decision rule has three branches on multi-seed median PPL:

- **Outcome THETA** — H8 positive: PPL $\le 12$.
- **Outcome IOTA-partial** — H8 partial: PPL $\in (12, 22]$.
- **Outcome IOTA-negative** — H8 negative: PPL $\ge 22$.

`q9e_n-seed0` lands at **PPL 25.04**, in the **IOTA-negative** branch. The publishable reading per protocol §6.3 Outcome IOTA-negative is:

> *Class F as currently realised is not attention's class. The Appendix A two-mechanism decomposition needs revising; attention realises a class strictly larger than Class F. … This precisely separates the Class-F-recoverable from the Class-F-irrecoverable portions of attention's expressivity and elevates FockPARFLM (paper §16+) as the canonical next architectural axis.*

Multi-seed escalation note: per §6.1 the multi-seed-escalation trigger is a $\Delta$ in $(-2\sigma_{\mathrm{seed}}, -1\sigma_{\mathrm{seed}}]$ vs the relevant baseline. The protocol's H8 decision rule is stated relative to **q9e_l multi-seed median (25.11)**, against which $\Delta = 25.04 - 25.11 = -0.07$ PPL — well within the "ties" band of $\pm 2\sigma_{\mathrm{seed}} = \pm 3.6$ PPL. A multi-seed `q9e_n` run is therefore not required to change the verdict's classification (negative-IOTA is robust to any plausible seed variance), and is treated here as an optional precision-improvement run rather than a verdict-gating run.

### 1.5 Cell-ladder comparison

The full SP-HSPLM Stage 2 single-seed ladder, with `q9e_n` appended:

| Cell | Mechanism-1 scope | Mechanism-2 scope | Params | Best val PPL | Δ vs P10g (26.42) |
|---|---|---|---:|---:|---:|
| P10g (baseline) | — | — | ~15.8 M | 26.42 | 0 |
| `q9e_a` | none | $\xi_t$ (prefix arg) | ~22 M | 27.58 | +1.16 |
| `q9e_h` | $J_\phi^{(\ell)}$ | $\xi_t$ | ~22 M | 26.68 | +0.26 |
| `q9e_d` | none | $\xi_t$ + per-token $\Omega$ | ~22 M | 26.89 | +0.47 |
| `q9e_l` (seed 0) | $J_\phi^{(\ell)}$ | $\xi_t$ + per-token $\Omega$ | ~22 M | **25.11** | −1.31 |
| `q9e_l` (3-seed median) | $J_\phi^{(\ell)}$ | $\xi_t$ + per-token $\Omega$ | ~22 M | **25.11** | −1.31 |
| `q9e_n` (seed 0) | $J_\phi^{(\ell)} + \Omega^{(\ell)} + V_\theta^{(\ell)} + V_\phi^{(\ell)} + \alpha_\phi^{(\ell)}$ | $\xi_t$ + per-token $\Omega$ | **23.9 M** | **25.04** | **−1.38** |
| MatchedGPT (informational, paper §15) | per-layer $W_Q^{(\ell)} W_K^{(\ell)} W_V^{(\ell)}$ | softmax routing | ~22 M | $\approx 8$ | $\approx -18.4$ |

Δ between `q9e_l` median and `q9e_n` is $-0.07$ PPL on $+\approx 11$ M extra parameters (V_θ-per-layer is the dominant new cost). The remaining gap to MatchedGPT is $\approx 17$ PPL on essentially matched parameter count.

---

## 2. What `q9e_n` rules out

Two consequences follow directly from §1.4 and §1.5.

### 2.1 The Mechanism-1 program is exhausted as a PPL lever

The maximally aggressive Mechanism-1 configuration on the SP-HSPLM stack — every learnable submodule of the force law per-layer indexed, with the prefix-conditioned argument and per-token $\Omega$ already enabled by Mechanism-2 — moves the headline by 0.07 PPL on the seed-0 reading versus the same architecture with shared V_θ. There is no remaining low-hanging fruit inside "make more things depth-non-autonomous on this stack". The four Mechanism-1 cells of §3.1 of the protocol (Q9e-H / I / J / K), `q9e_l`'s additivity confirmation, `q9e_m`'s per-layer $\Omega$ stretch, and `q9e_n`'s full Class-F together exhaust the Mechanism-1 axis at this scale.

### 2.2 The 17-PPL residual to attention is structural, not capacity

The matched-attention baseline (MatchedGPT, paper §15) sits at $\approx 8$ PPL at one-third of the parameter budget of `q9e_n`. The gap is therefore not parameter-budget-limited on the attention side and not parameter-budget-limited on the SP-HSPLM side. By the Outcome IOTA-negative reading, the gap reflects an **expressivity-class deficit**: SP-HSPLM Class F as built is in a strictly smaller structural class than the class realised by trained attention transformers.

---

## 3. Scope of the `paper_tmlr_1` "locally conservative" framing

`paper_tmlr_1` §7–§8 reports that the velocity-aware per-layer Jacobian-symmetry test passes universally across all three tested architectures (pretrained GPT-2, matched-attention 8M baseline, scalar-potential probe) with maximum symmetric-vs-unconstrained gap $\le 0.079$ on any layer of any architecture. The headline reading is "trained attention transformers are locally conservative, globally not".

That reading is correct as stated, **but** the per-layer Jacobian-symmetry test as run in `paper_tmlr_1` §7.2 has two non-trivial scope-of-claim conditions that bear on the present discussion. First, the test is conducted at PCA-16 — i.e. on the projection of each architecture's hidden-state operator onto its top-16 principal-component subspace. Second, the test is **single-token-internal**: it inspects the symmetric part of $\partial y_i / \partial h_i$ at fixed-token-$i$, freezing $h_{j \ne i}$ at their realised values, rather than the full sequence-level Jacobian $\partial y_i / \partial h_j$ at $j \ne i$.

A pair-coupled non-conservative operator on the whole sequence, such as attention's $y_i = \sum_j A_{ij} V_j$, can pass the single-token-internal test (the per-token-$i$ map looks locally like a gradient *in $h_i$'s own coordinate*, holding $h_{j \ne i}$ fixed) while being structurally non-conservative as an operator on $(h_1, \ldots, h_T)$. The off-diagonal blocks $\partial y_i / \partial h_j$ for $j \ne i$ encode the many-body coupling that the per-token-$i$-internal test never inspects.

For the present note's purposes the operational reading is: the `paper_tmlr_1` local-conservativity finding is robust within its tested scope, and the structural gap between "passes the single-token-internal test at PCA-16" and "is in Class F as an operator on the full sequence" is exactly the class of phenomena `q9e_n` was supposed to probe and that, on the IOTA-negative reading, are responsible for the 17-PPL residual.

A formal sharpening of the `paper_tmlr_1` §7–§8 scope-of-claim text is therefore a candidate follow-up edit, conditional on §5.1 below; it is not committed to in this note.

---

## 4. Three live hypotheses for the IOTA-negative residual

Listed by current credence, highest first. All three are falsifiable; each has a single cheap experiment that distinguishes it from the other two.

### 4.1 H1 — Many-body coupling, not conservativity, is the missing axis

The structurally hard thing trained attention does is data-dependent pairwise token coupling: $y_i = \sum_j A_{ij}(h, \mathrm{prefix}) V_j(h_j)$, where the routing weights $A_{ij}$ and the value vectors $V_j$ both depend on the *current values* of the other tokens $h_{j \ne i}$. Mechanism-2's prefix-conditioning $\Omega(\xi_t)$ summarises the entire prefix into one rank-deficient vector and applies it as a velocity coupling on $h_i$ alone — it cannot encode "the force on token $i$ depends on which token $j$ is at distance $r$ from it in current state, separately for every $j$".

Class F as built in SP-HSPLM is **mean-field**: each token's update is a function of its own state $h_i$ plus a prefix-summary vector $\xi_t$. The class realised by attention is **many-body**: each token's update is a function of the full $(h_1, \ldots, h_T)$ tuple. Even a Class F with all submodules per-layer indexed remains in the mean-field class; the IOTA-negative result is what mean-field theory costs you on a sequence model whose ground truth is many-body.

**Falsifier:** add an explicit pairwise coupling term $F_i^{\mathrm{pair}} = \sum_j W_{\mathrm{pair}} (h_j - h_i)$ to the SP-HSPLM force law (bilinear, *no* softmax routing — just a learnable pairwise tensor) and measure the PPL delta against `q9e_n` at matched parameter count. If PPL drops materially (say to $\le 18$), H1 is confirmed.

### 4.2 H2 — Local conservativity fails at full $d$ (the PCA-16 finding does not extend)

The `paper_tmlr_1` single-token-internal Jacobian-symmetry test is conducted at PCA-16. At $d_{\mathrm{model}} = 768$ on GPT-2-small the test inspects roughly $16/768 \approx 2\%$ of the operator's variance. The possibility H2 entertains is that the remaining $98\%$ of the operator (the 752-d residual) is where the *non-conservative* dynamics live — i.e. that GPT-2's per-layer update is non-conservative even per-token, but the non-conservative residual is mostly orthogonal to the top-16 principal components and is therefore invisible to the PCA-16 test.

If H2 holds, the entire framing of Class F as "the structural class GPT-2 is in" is wrong: GPT-2 is in the strictly larger non-autonomous *non-conservative* class, and the SP-HSPLM IOTA-negative result reflects the cost of insisting that the per-layer update be a gradient at all (a constraint GPT-2 never accepted).

**Falsifier:** rerun the velocity-aware Jacobian-symmetry test of `paper_tmlr_1` §7.2 at PCA-32, PCA-64, PCA-128, PCA-256, and full $d = 768$ on the same frozen GPT-2-small checkpoints, with no retraining. If the symmetric-vs-unconstrained gap stays flat ($\le 0.08$) across the sweep, the local finding is robust and H2 is rejected. If the gap grows monotonically with PCA dimension and is large at full $d$, H2 is confirmed and the "locally conservative" claim of `paper_tmlr_1` §7–§8 needs a scope-of-claim sharpening *and* the SP-HSPLM Class-F design needs a structural rethink.

### 4.3 H3 — Conservative force fields have an intrinsic expressivity tax at high $d$

A scalar potential $V: \mathbb{R}^d \to \mathbb{R}$ has $\sim p$ degrees of freedom where $p$ is its parameter count; a general per-layer linear operator on $\mathbb{R}^d$ has $\sim d^2$ degrees of freedom. For $d_{\mathrm{model}} = 1024$ in SP-HSPLM's V_θ this is a $\sim 1024 : 1$ ratio per layer, of which Mechanism-1 recovers a factor of $L_S = 4$ (one V_θ per S-block). The remaining gap is the cost of insisting the per-layer update be a gradient at all.

H3 is the limiting case of H1 with no pairwise coupling: even if H1 holds, H3 says that no purely-conservative architecture can match attention at this $d$ by parameter count alone, because the conservative constraint costs $\Omega(d)$ degrees of freedom per layer. H3 cannot be isolated by a single experiment — it is what is *left over* after H1 is falsified or confirmed — but it sets a floor on what a Class-F-style architecture can achieve regardless of how many submodules are per-layer indexed.

---

## 5. Next experiments (proposed, not pre-registered)

Listed in execution order, cheapest first.

### 5.1 H2 PCA-symmetry sweep (no training, ~1 GPU-day)

Rerun the velocity-aware Jacobian-symmetry test of `paper_tmlr_1` §7.2 on the same frozen GPT-2-small checkpoint used in `paper_tmlr_1` §8, sweeping the PCA dimension across $\{16, 32, 64, 128, 256, 768\}$. Report the per-layer symmetric-vs-unconstrained gap $\Delta_{\mathrm{sym}}(\ell, d_{\mathrm{PCA}})$ as a function of $d_{\mathrm{PCA}}$ on every layer. Also re-run on Pythia-160M (already used in `paper_tmlr_1` §6) for cross-architecture confirmation.

**Decision rule:**
- If $\Delta_{\mathrm{sym}}$ stays $\le 0.08$ across the sweep on every layer of both architectures, H2 is **rejected** and the `paper_tmlr_1` local-conservativity finding is robust at all reasonable PCA dimensions. Diagnosis is then H1 (with H3 as residual).
- If $\Delta_{\mathrm{sym}}$ grows monotonically with $d_{\mathrm{PCA}}$ and exceeds $0.20$ at $d_{\mathrm{PCA}} \ge 128$, H2 is **confirmed** and the `paper_tmlr_1` §7–§8 scope-of-claim text needs a substantive sharpening; the SP-HSPLM Class-F design needs revisiting at the architectural level.
- Intermediate behaviour (mild monotone growth, $\Delta_{\mathrm{sym}} \in [0.08, 0.20]$ at full $d$) is read as "H2 partially holds"; the scope-of-claim sharpening is still warranted but the Class-F design is not invalidated.

This is the cheapest possible experiment in the structural-re-examination programme and gates everything downstream.

### 5.2 `q9e_o` — Class F + bilinear pairwise coupling (conditional on H1 surviving §5.1)

If §5.1 rejects H2 (or only partially confirms it), `q9e_o` adds an explicit pairwise coupling term to the SP-HSPLM force law that is **not** softmax-routed (to isolate the pairwise-coupling effect from the routing effect):

$$f_i^{\mathrm{(C, pair)}} = \sum_{j < i} W_{\mathrm{pair}}\,(h_j - h_i)$$

with $W_{\mathrm{pair}} \in \mathbb{R}^{d \times d}$ shared across layers (Mechanism-1 disabled for this term so the pairwise effect is isolated from the per-layer-indexing effect; per-layer indexing of $W_{\mathrm{pair}}$ is the follow-up `q9e_p`).

**Configuration:** same as `q9e_n` except $W_{\mathrm{pair}}$ is added and one of the existing per-layer submodules has its parameter budget reduced to keep total params matched. Concretely: drop V_θ-per-layer (i.e., share V_θ across layers, recovering the q9e_l-style baseline) and use the freed ~10 M parameters for $W_{\mathrm{pair}}$ (a single $1024 \times 1024$ tensor is $\approx 1$ M params, so this gives multiple per-pair coupling heads or a much higher pairwise rank).

**Decision rule:** if `q9e_o` PPL $\le 18$, H1 is **confirmed** and the next axis is identified (pairwise coupling, the missing primitive). If `q9e_o` PPL $\ge 22$ (still IOTA-negative against the q9e_l baseline), H1 is **rejected**: pairwise coupling alone — without softmax / data-dependent routing — is not the missing primitive, and the residual is concentrated in attention's *routing* component (data-dependent $A_{ij}$, not just data-independent $W_{\mathrm{pair}}$).

### 5.3 `q9e_p` — per-layer-indexed pairwise coupling (conditional on `q9e_o` landing well)

Only if `q9e_o` confirms H1. Lifts $W_{\mathrm{pair}}$ to a per-layer `nn.ModuleList(L_C)` and re-runs at matched parameter count by reducing the per-layer Mechanism-1 budget further. Tests whether the H6-style super-linear synergy of Mechanism-1 × Mechanism-2 reappears on the Mechanism-3-style pairwise-coupling axis.

### 5.4 (Optional) `q9e_n` multi-seed precision (not gate-changing)

A 3-seed `q9e_n` re-run would sharpen the IOTA-negative classification's seed-variance interval. Per §1.4 this is not gate-changing — the verdict's classification is robust to any plausible seed variance — but is logged here as an available precision-improvement run if compute is cheap.

---

## 6. Paper-update policy

**No paper revision (`paper_v4`, `paper_v5`, `paper_tmlr_1`) lands on the basis of this note.** Rationale:

1. **The story is incomplete.** What we have is a clean negative result plus three live structural hypotheses (§4). Writing the negative result into any of the three papers without picking a diagnosis would commit the paper to a framing that the §5.1 H2 PCA-symmetry sweep — which is cheap and uncommitted — could materially revise. If §5.1 confirms H2, the framing of `paper_tmlr_1` §7–§8 changes substantially; if §5.1 holds the line, the framing stays and H1 becomes the named residual.

2. **The papers are coherent as they stand.** `paper_tmlr_1` is a positive story about the STP–acceleration identity and the shared-V three-way separator; `paper_v4` and `paper_v5` cover the SP-HSPLM language-modelling programme through `q9e_l` and its 3-seed BETA verdict. Inserting `q9e_n` plus a "we now believe Class F is mean-field" caveat into any of these mid-investigation dilutes their focus and risks a revision pass two weeks later.

3. **The pre-registered protocol already anticipated this branch.** `q9e_n` was set up with three outcome scenarios (THETA / IOTA-partial / IOTA-negative) precisely so that landing in IOTA-negative would trigger structural re-examination, not paper revision. This note is the structural re-examination; the papers are where the eventually-resolved conclusion lives.

**Trigger for the next paper-edit cycle:** the §5.1 H2 PCA-symmetry sweep landing. Both branches of its outcome unblock a specific paper-edit:

- §5.1 rejects H2 → `paper_tmlr_1` §7–§8 stays as is; a one-paragraph addition to the discussion can summarise the H1 reading and the `q9e_n` IOTA-negative classification as supporting evidence for the "globally not" half of the title claim.
- §5.1 confirms H2 → a scope-of-claim sharpening to `paper_tmlr_1` §7–§8 is mandatory (the local-symmetry pass is at PCA-16 single-token-internal only; the full-$d$ symmetric residual is large; the architectural class GPT-2 is in is strictly larger than Class F); `paper_v4` / `paper_v5` get a corresponding revision of the SP-HSPLM ceiling discussion.

In both cases the edit is one-paragraph-scale, not section-scale.

---

## 7. References

### Primary (this repository)

- [`SP_HSPLM_Stage_2_pre-registered_protocol.md`](./SP_HSPLM_Stage_2_pre-registered_protocol.md) — pre-registration of `q9e_n` and the H8 decision rule with Outcomes THETA / IOTA-partial / IOTA-negative (§6.3).
- [`Scalar_Potential_based_Helmholtz_Architecture_v3.md`](./Scalar_Potential_based_Helmholtz_Architecture_v3.md) — SP-HSPLM design doc; §9.2 specifies the Stage 2 cell ladder including the Mechanism-1 extension and the full Class-F test.
- [`SP_HSPLM_Stage_1_pre-registered_protocol.md`](./SP_HSPLM_Stage_1_pre-registered_protocol.md) — Stage 1 protocol; E4-fix solenoidal $r=4$ at PPL 24.58 is the H2 baseline.
- [`Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](./Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md) — leak-fix invariant verified across all `q9e_n` checkpoints (§1.2 above).

### Companion repository (mirror of this note)

- [`semsimula-paper/companion_notes/SP_HSPLM_Stage_2_q9e_n_verdict_and_structural_reexamination.md`](https://github.com/dimitarpg13/semsimula-paper/blob/main/companion_notes/SP_HSPLM_Stage_2_q9e_n_verdict_and_structural_reexamination.md)

### Paper references

- `paper_tmlr_1` §7.2 — velocity-aware Jacobian-symmetry test, single-token-internal at PCA-16 (the scope-of-claim subject of §3 above).
- `paper_tmlr_1` §8 — shared-V three-way separator ($R^2 = 0.45 / 0.56 / 0.949$); the "globally not" half of the title claim.
- `paper_v4` §15 — SP-HSPLM Stage 2 ladder write-up through `q9e_l`; the `q9e_n` row is *not* added to this paper on the basis of this note (see §6).

---

*Last updated: 17 May 2026. Locked at `q9e_n-seed0` verdict (PPL 25.04, IOTA-negative). Next experiment is the §5.1 H2 PCA-symmetry sweep on frozen GPT-2-small; the paper-edit cycle is gated on its outcome.*
