# Restructuring `paper_v3` After the Causal-Leak Bug Discovery

> **Status.** Drafted **2026-05-01**, immediately after the discovery and forensic characterisation of the anti-causal autograd leak in the SPLM `integrate()` method ([`docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md)). This document records the strategic discussion of how the bug discovery should reshape `paper_v3`. It is descriptive, not prescriptive: a record of the reasoning that led to the restructure, not a final manuscript plan.
>
> **Companion documents:**
> - **The bug itself:** [`docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md)
> - **Multi-channel ξ design (now updated for the leak):** [`docs/Multi-Channel_vs_Single_Channel_Xi_SPLM_Design.md`](Multi-Channel_vs_Single_Channel_Xi_SPLM_Design.md)
> - **Current paper:** [`paper_v3/main.tex`](../paper_v3/main.tex)

---

## 1. The precipitating event

On 2026-04-30 a project-wide anti-causal autograd leak was discovered in every per-step `integrate()` site of the SPLM family. At training time the integrator was computing the conservative force as

$$
f_t = -\Big[ \frac{\partial V_t}{\partial h_t} + \sum_{s > t} \frac{\partial V_s}{\partial \xi_s} \cdot \frac{\partial \xi_s}{\partial h_t} \Big],
$$

where the second sum is anti-causal: token $h_t$ feels a force that depends on hidden states at *future* positions $s > t$. After 8000 training steps the trained $V_\theta$ had learned to actively route prediction signal through this leak channel. The published E9 SPLM val_ppl of $8.85$ on TinyStories evaluates to **6843 PPL under a leak-free integrator** — an inflation factor of **777×**. A 2000-step buggy multi-ξ run (E11) reached **val_ppl = 1.26 at step 600** (essentially perfect prediction via the leak channel), confirming that the multi-channel architecture provides an even more potent leak path than the single-channel cumulative mean.

The fix (one line: compute ξ from `h.detach()`) restores causality. Under the fix, the same E9 architecture trains to **val_ppl = 33.55 at 4000 steps** — a real, but not leak-driven, perplexity ~4.3× larger than the matched-attention baseline (MatchedGPT, val_ppl = 7.81 at 8000 steps).

The empirical state of the SPLM-quality story therefore changed dramatically between the morning of 2026-04-30 and the evening of 2026-05-01.

---

## 2. What this changes in `paper_v3`'s claim stack

The current `paper_v3` abstract makes five distinct empirical / structural claims. The leak fix hits them very unevenly.

| # | claim (abstract paragraph) | hit by leak? | post-fix status |
|---|---|---|---|
| 1 | **Descriptive validation on pretrained transformers** — STP=acceleration identity to machine precision on $1{,}314$ GPT-2 triplets; tangential ≈ 2× normal; 97.9% deceleration; permutation null $z<-11$; replicates on Pythia-160M | no | **untouched** |
| 2 | **Negative result on conservative retrofitting of attention** — seven scalar-potential forms + Helmholtz position-coupled gauge + EM velocity-coupled gauge all tie static-null on held-out GPT-2 trajectories; six attention features each independently obstruct conservativity | no | **untouched** |
| 3 | **Shared-potential separator** — SPLM $R^2 \approx 0.90$ (uniform), matched 8M GPT-2-style $R^2 = 0.56$ (monotonic decay), pretrained GPT-2 small $R^2 = 0.45$ (bathtub) | partial | **structurally intact**; the SPLM number must be re-fit on leak-corrected weights, but the architectural fact "SPLM is $\nabla V_\theta$ flow by construction, attention is not" is robust |
| 4 | **Multi-seed LM-quality** — splm_em_ln val_ppl $95.33 \pm 4.44$ vs GPT-2-micro $149.80 \pm 7.21$, 36% reduction, Welch's $t = 14.4$ | yes | **gone**. E1 was run on the buggy integrator. The 36% reduction is almost certainly the leak; multi-ξ collapsing to PPL $1.26$ at step 600 is the same mechanism on steroids |
| 5 | **γ-damping sweep** — optimal $\gamma^{\ast} \approx 0.30$, val PPL $144$, vs freely-trained $\gamma \approx 0.85$ at val PPL $203$; Markov-order regression returns Decision β at all six γ values | partial | the SPLM side **needs re-running**; specific numbers are casualties; the structural claim "γ-tuning matters" survives. The natural-transformer reproductions on GPT-2 / Pythia ($21/24$ and $24/24$ cells) are independent and survive |

Net effect: claims 1, 2, and 5 (natural-transformer half) are **completely untouched**. Claim 3 is structurally intact but requires re-running one experiment. Claims 4 and 5 (SPLM half) require new experiments; the specific PPL numbers in the abstract are dead.

The two strongest empirical claims of the paper — the descriptive validation on pretrained transformers, and the negative-result analysis of attention's non-conservativity — are both entirely measurements on transformers the author did not train. They are robust *by construction* against any training-side bug. What the leak fix kills is precisely the bridge from those analyses to "and therefore SPLM as a built thing is also competitive on the LM task itself."

---

## 3. Why SPLM-as-an-independent-paper is no longer viable

Pre-fix, the previously-discussed plan was to spin SPLM off as one of multiple journal submissions: a stand-alone "scalar-potential language model" paper at a top ML venue. That submission needed three legs to stand:

1. **Quality leg.** "SPLM is competitive with scale-matched attention on PPL." — under the buggy integrator, supported by E1 and E9. **Now: gone.** Honest fixed-SPLM PPL is ~4–5× worse than MatchedGPT; the multi-ξ extension doesn't change this materially (Prediction B in [`Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md) §4.4).

2. **Inference-efficiency leg.** "SPLM has $O(L \cdot d)$ per-token decode FLOPs, attention has $O(T \cdot L \cdot d)$." — the architectural FLOP and wall-clock measurements are leak-immune (`inference_efficiency/results/RESULTS.md` Phase 2 §2.6 grades A2.C1–A2.C4 and WC-cross are robust). **Status: survives, but is a workshop / niche claim.** "$O(L \cdot d)$ per token at the cost of 4× PPL" is publishable as a focused note, not as a general LM paper.

3. **Theoretical-novelty leg.** "Conservative-by-construction LM circuit derived from a Lagrangian framework." **Status: derivative.** SPLM realises the framework, but the framework's *interesting* validation is descriptive (on GPT-2/Pythia), not prescriptive. Without the quality leg, the theoretical novelty is "we built the simplest possible thing that the framework predicts and showed it trains stably" — true and useful, but not headline material.

There is no SPLM paper that can stand on its own legs after the fix. The architecture's right place is as **the simplest possible existence proof of the framework** — one figure, one table, one paragraph in a larger paper — not as a co-headline contribution.

---

## 4. Concrete restructuring proposal

The following is in rough order of edit-effort. None of this is a rewrite from scratch; the framework theory (sections 02–13) is essentially untouched, and the descriptive findings (sections 14 first half, 16) are unaffected.

### 4.1 Title and framing shift (small edit, large effect)

**Current title:**

> **Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference**
> *Conservative-by-Construction Language Models and the Shared-Potential Separator, with a Correspondence to Joint Embedding Predictive Architectures*

**Proposed new title:**

> **Semantic Simulation: A Lagrangian Account of Transformer Inference Dynamics**
> *JEPA Correspondence, the Shared-Potential Separator, and a Constructive Architectural Realisation*

The substantive shift is from *"we propose a new architecture"* to *"we explain what attention is doing, and demonstrate constructively that the framework is implementable."*

The word "efficient" is removed because — without the quality claim — the inference-efficiency story is at best a side benefit, not a headline. "Conservative-by-Construction Language Models" is replaced by "Constructive Architectural Realisation" because the new framing positions SPLM as a single working example, not as a model class to be promoted.

### 4.2 Restructured contribution stack

| tier | contribution | sections | post-fix status |
|---|---|---|---|
| **primary** | Descriptive — semantic simulation explains the geometry of trained-attention trajectories. STP=acceleration identity, near-geodesic trajectories, locally-conservative-but-globally-not-shared-potential character. | §§02–13, §10 (JEPA), §13 (STP=acceleration), §14 first half (GPT-2 / Pythia), §16 (Riemannian) | unaffected |
| **secondary** | The shared-potential separator as a **methodology** for diagnosing architectural conservativity. The construction — fitting per-layer updates jointly with a single shared $V_\psi$ across all layers — is novel and reusable beyond `paper_v3`. | §15, restructured around the methodology | structurally intact; SPLM number re-fits on leak-corrected weights; attention numbers survive |
| **tertiary** | SPLM as a **minimal working realisation** that the dynamics is trainable as a circuit. Performance is honestly reported as not competitive with attention; the value is structural. | §14 second half + §15, condensed | reframed as proof-of-concept |

### 4.3 New "Causal-leak correction" subsection

A new prominent subsection (probably §14.0 or §14.1) documents the bug, the fix, the inflation factors, and what specifically changed in the paper. This is honest scientific practice; reviewers will reward visible disclosure far more than they would punish a quietly-revised paper. Concrete content:

- The bug: anti-causal autograd path from $\xi$ back to $h$.
- The fix: `h.detach()` before computing $\xi$, exposed as `causal_force=True` (default) on every affected config.
- The empirical impact:
    - E9 single-ξ ckpt published at val_ppl $8.85$, leak-free PPL = $6843$ ($777\times$ inflation).
    - Buggy multi-ξ at 600 steps reached val_ppl $1.26$ — a manifestly implausible number that would have been a smoking-gun in any peer review.
    - Fixed E9 single-ξ at 4000 steps converges to val_ppl $33.55$, plateauing.
- Pointer to [`docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md) for full forensic detail and the regression test (`notebooks/conservative_arch/causal_probe.py`).

### 4.4 Rewrite the multi-seed LM-quality paragraph (abstract + §14)

**Current abstract paragraph** (~paragraph 4):

> "An $n=5$-seed sweep on Tiny Shakespeare with identical optimiser configuration shows that the LayerNorm-after-step variant of SPLM reaches val perplexity $95.33 \pm 4.44$ at $7.12$M parameters with $0/5$ NaN divergence against a scale- and data-matched GPT-2-micro baseline at $149.80 \pm 7.21$ — a $36.4\%$ mean-perplexity reduction at $11.5\%$ fewer parameters, Welch's $t = 14.4$..."

**Proposed replacement:**

> "Under the leak-corrected integrator, SPLM trains stably (no NaN divergences in any of $n=5$ seeds) and learns a non-trivial causal language model: val_ppl ${\sim}30$ at the 4000-step pilot scale on TinyStories. SPLM is **not** competitive on PPL with scale-matched attention at this configuration ($\sim$4–5× larger PPL than a parameter-matched MatchedGPT baseline). The contribution is structural — the dynamics is implementable and trains cleanly — not a quality competition. Pre-fix numbers (val_ppl $95.33$ on Shakespeare, val_ppl $8.85$ on TinyStories) are documented with their leak-correction factors in §14.1; the corresponding leak-corrected re-runs are reported in §14.4."

This is a much easier number to defend than the pre-fix $95.33$-vs-$149.80$ claim, and it does not invite the question "if SPLM is so much better, why hasn't anyone deployed it?".

### 4.5 §A2 inference efficiency: keep architectural claims, drop quality Pareto

The Phase 2 architectural FLOP and wall-clock measurements are entirely independent of the leak. Specifically:

- **A2.C1** — $F_{\rm attn}^{\rm fwd} / F_{\rm splm}^{\rm fwd}$ growth ratio under $T$-doubling: *unaffected* (forward FLOP counts).
- **A2.C2** — forward-FLOP crossover at finite $T$: *unaffected*.
- **A2.C3-SPLM** — streaming-ξ wall-clock per-step constant in $T$: *unaffected* (per-step FLOPs are exactly constant).
- **A2.C3-ATTN** — KV-cached wall-clock linear in $T$: *unaffected*.
- **A2.C4** — SPLM params flat in $L$, ATTN params linear in $L$: *unaffected* (parameter counting).
- **WC-cross** — empirical wall-clock crossover $T_{\rm wc} \le 16{,}384$: *unaffected* (wall-clock measurement).

What does *not* survive: the **Phase 3 quality-axis Pareto frontier** ("SPLM substantially lower val PPL on Shakespeare and on TinyStories"). The PPL numbers used to construct that Pareto frontier were leak-driven; the frontier itself is now meaningless.

Recommendation: keep §A2 with all six architectural sub-claims and their grades, drop the Phase 3 Pareto subsection entirely (or replace with a single paragraph saying "no quality claim is made on this configuration; see §14.1 leak-correction").

### 4.6 §15 separator: re-fit on leak-corrected SPLM, keep the methodology

The shared-potential separator's key claim is structural: a *single* learned $V_\psi$ jointly across all layers fits SPLM's per-layer updates well (R² ≈ 0.9), fits a parameter-matched GPT-2-style decoder poorly (R² = 0.56), and fits pretrained GPT-2 worse (R² = 0.45). The 0.56 / 0.45 numbers are independent of the SPLM bug — they're measured on attention checkpoints that have never used SPLM's `integrate()` method.

What needs to be re-run: the SPLM R² ≈ 0.9 number, on a leak-corrected SPLM checkpoint. By construction a fixed SPLM is *literally* a $\nabla V_\theta$ flow (the $\nabla V_\theta$ is computed from `h.detach()` + `h_in`; the integrator is $h \mathrel{+}= dt \cdot v$; $v$ accumulates $-\nabla V_\theta / m$ with damping). So the structural fit is mechanical: the per-layer updates *are* generated by a single $V_\theta$. The R² should still be ≈ 1.0, with the residuals coming only from the integrator's discretisation choices (semi-implicit Euler).

The methodology is the contribution. The specific R² values are confirmation. The methodology slice is potentially extractable as a separate "tools" submission to a venue like TMLR.

### 4.7 γ-damping sweep: shorten or re-run

The current §14 γ-sweep narrative reports specific numbers (optimal $\gamma^* \approx 0.30$, val PPL $144$ at the optimum) that came from buggy training runs. Two options:

**Option A — re-run a small γ-sweep under the fix.** Cheap-ish (3 γ values × 4000 steps × ~7h each = ~21h on MPS, parallelisable across nodes if available). Yields new specific numbers and confirms (or revises) the optimal-γ claim.

**Option B — drop the specific γ-optimum, keep the structural claim.** Reframe as "γ-tuning matters and is non-trivial; the specific optimum is left to future work for the leak-corrected configuration." The Markov-order Decision β finding **on natural transformers** (GPT-2 / Pythia, $21/24$ and $24/24$ cells) is independent of SPLM and survives in either case.

Recommendation: Option A if MPS budget permits; Option B as the fallback. See §5.6 below for the experimental plan if Option A is chosen.

---

## 5. Targeted re-runs to support the restructure

The structural changes in §4 each require either (a) a leak-corrected re-run to replace a now-obsolete number, or (b) a forensic eval-only measurement that backs a specific claim in the rewritten paper. This section enumerates those experiments by priority. The list is intentionally small — not a "redo the whole experimental record" plan, but the **minimum set of runs needed to make the restructured paper defensible**. Each item is tagged with the §4 placeholder it fills, the companion-repo deliverable it serves, and the falsifiable prediction it tests (where applicable).

### 5.1 Priority tiers

The experiments fall into three tiers by compute cost / leverage:

| tier | criterion | total compute |
|---|---|---:|
| **P1** | cheap (eval-only or trivial); directly fixes a specific number quoted in the paper / restructure plan; high leverage per minute of compute | ~2.5 h MPS |
| **P2** | medium cost (single overnight run); useful but not strictly required | ~14 h MPS |
| **P3** | expensive multi-cell sweep; only needed if the paper takes Option A on §14 γ-sweep | ~21 h MPS |

P1 is the **mandatory** set. P2 is **recommended** for a stronger §14 stability claim. P3 is **optional** and can be replaced by Option B (drop the specific γ-optimum, keep the structural γ-tuning narrative).

### 5.2 P1.1 — §15 separator R² re-fit on leak-corrected E9 single-ξ (HIGHEST leverage)

**What:** Load the leak-corrected E9 single-ξ pilot ckpt (`pilot_splm_fixed/`, val_ppl = 33.55 at 4000 steps) and run `shared_potential_fit.py` against its hidden-state trajectories. Get the new SPLM R² number for the §15 three-way separator.

**Cost:** ~30–60 min on MPS, eval-only. No retrain.

**Fills:** §4.6 of this doc; specifically the abstract claim "SPLM at median per-layer test $R^2 = 0.90$ (uniform profile)" must be re-quoted on the leak-corrected ckpt.

**Falsifiable prediction:** $R^2 \approx 1.0$. The leak-corrected SPLM is by construction a single-$V_\theta$ flow at every step (the integrator is $h \mathrel{+}= dt \cdot v$, with $v$ accumulating $-\nabla V_\theta / m$ under damping; both the buggy and the fixed integrators have this structural property — the bug affected *what the trained $V_\theta$ learned to value*, not whether the dynamics is gradient-flow). Residuals should come only from the semi-implicit Euler discretisation. **If $R^2$ is materially below ~1.0**, that's a substantive new finding worth a paragraph in §15 of its own.

**Why this is highest priority:** of all the leak-corrected numbers the paper needs, this is the cheapest, the most cited (it appears in the abstract), and the one with the strongest mechanical prediction. If $R^2$ comes out at 0.99–1.00 as expected, the §15 separator argument is rock-solid post-fix; if it doesn't, the surprise has its own scientific value. Either outcome materially helps the paper.

**Companion-repo deliverable:** new figure / numerical record under `notebooks/conservative_arch/results/sharedV_splm_postfix_*.{png,npz,md}`, replacing the old `sharedV_sarfmass_logfreq_shakespeare_*` artefacts in `paper_v3/figures/sharedV_splm_vs_gpt2.png` (per §4.6 of this doc).

### 5.3 P1.2 — Causal probe on every natural-transformer ckpt used in the paper (hygiene check)

**What:** Run `causal_probe.py` in ckpt mode on each MatchedGPT, GPT-2 small, Pythia-160M ckpt cited in the descriptive-validation half of §14 and §15. Confirm causal-side $\Delta = 0$ on all of them.

**Cost:** ~5 min total (it's a single forward pass per ckpt).

**Fills:** §4.3 ("New 'Causal-leak correction' subsection") — adds a one-sentence paragraph to the rewritten §14.0 / §14.1 saying "we ran the causal-violation probe on every model in this paper; all natural-transformer ckpts give $\Delta = 0$ on the causal side, ruling out any analogous hidden bug in the descriptive-validation half of the paper."

**Why this matters:** this is the **single most defusing-of-reviewer-question paragraph** the paper can add. The obvious challenge after a bug-disclosure section is "you found a causal-violation bug in your own model — is something analogous lurking in the natural transformers you used for descriptive validation?" Without this measurement, the paper has no answer. With it, the answer is one sentence and a reference to a regression test that ships in the companion repo.

**Falsifiable prediction:** $\Delta = 0$ for every natural-transformer ckpt — they're standard causal autoregressive decoders with no SPLM-style integrator. If any non-zero $\Delta$ appears, that's a serious finding requiring its own investigation, but it would be extraordinary.

**Companion-repo deliverable:** small results file `notebooks/conservative_arch/causal_probe_natural_transformer_results.{md,json}` recording the probe output for every cited ckpt. Strengthens the public companion's "we checked our work" story.

### 5.4 P1.3 — E1 buggy multi-seed inflation forensic (Shakespeare)

**What:** Run `eval_ppl_under_fix.py` on each of the 5 seeds of the E1 buggy Shakespeare multi-seed run (`multi_seed/results/E1_shakespeare/splm_em_ln/seed_{0..4}/`). Record the (buggy-mode PPL, fixed-mode PPL, inflation factor) tuple for each seed.

**Cost:** ~10 min per seed × 5 seeds = ~50 min total.

**Fills:** the inflation table in §4.3 ("the empirical impact" bullets) currently has E9 (777×) and E11 multi-ξ (389×). Adding E1 makes the bug-disclosure narrative **forensically complete across every architecture/corpus combination** the paper has ever published a leak-driven number for.

**Why this matters:** the public companion repo Phase 4 plan (`docs/Companion_Repo_Restructure_Plan.md` §6) specifies that the `legacy/buggy_pre_2026-05-01/multi_seed_E1/` directory should be retained as forensic evidence of the inflation factor. **Without measuring the inflation factor on E1**, that legacy directory is uncalibrated — a reader sees the buggy ckpts but can't quantify how much of the published $36.4\%$ PPL reduction was leak-driven. The measurement is the one thing that makes the legacy archive useful as evidence rather than as a shrug.

**Companion-repo deliverable:** populates the `legacy/buggy_pre_2026-05-01/multi_seed_E1/README.md` with a 5-row inflation table; strengthens both this restructure doc and the bug doc.

**Falsifiable prediction:** all 5 seeds show inflation factors in the same ballpark (qualitatively similar to E9's 777× since both are single-ξ on small corpora; possibly higher because Shakespeare has more local repetition the leak channel can exploit). The seed-to-seed variance of the inflation factor itself is independently informative.

### 5.5 P1.4 — §A2 wall-clock benchmark on leak-corrected ckpt

**What:** Re-run the `inference_efficiency/run_inference_benchmark.py` and `run_inference_benchmark_longctx.py` micro-benchmarks using the leak-corrected E9 single-ξ ckpt (`pilot_splm_fixed/`) as the SPLM side, comparing against the unchanged MatchedGPT ckpt.

**Cost:** ~30 min on MPS / CPU.

**Fills:** §4.5 ("§A2 inference efficiency: keep architectural claims, drop quality Pareto"). Right now the §A2 architectural sub-claims (A2.C1–C4 and WC-cross) are leak-immune *by construction*, but the cleanest way to make that unambiguous in the rewritten §A2 is to re-anchor the numbers on a leak-corrected ckpt and confirm the FLOP / wall-clock measurements come out identical.

**Why this matters:** this is cheap insurance. The argument "leak-immune by construction, take our word for it" is technically correct but textually weaker than "we re-ran it; same numbers." The rewritten §A2 becomes a one-line addition: "all measurements re-confirmed on the leak-corrected ckpt; no values changed at the precision quoted."

**Falsifiable prediction:** every architectural FLOP-count and wall-clock number in the existing `inference_efficiency/results/` reproduces to within sampling noise (~5%). The structural FLOP scaling laws ($O(L \cdot d)$ vs $O(T \cdot L \cdot d)$) are mathematical identities and must reproduce exactly.

**Companion-repo deliverable:** updated `inference_efficiency/results/RESULTS.md` with a footnote "all measurements re-confirmed against leak-corrected ckpt on YYYY-MM-DD; per-cell numbers identical to the precision quoted."

### 5.6 P2.1 — Multi-seed leak-corrected single-ξ E9 pilot (seeds 1, 2)

**What:** Two additional seeds of the leak-corrected single-ξ E9 4000-step pilot, seeds 1 and 2, otherwise identical config to seed 0 (val_ppl = 33.55).

**Cost:** ~7 h per seed × 2 seeds = ~14 h on MPS, single overnight run.

**Fills:** §4.4 (rewritten LM-quality paragraph). The placeholder text quotes "val_ppl ${\sim}30$ at the 4000-step pilot scale" and adds "no NaN divergences in any of $n=5$ seeds". Without multi-seed runs the rewritten paragraph has to qualify with "single seed; multi-seed left to future work", which weakens the stability claim materially.

**Why this matters:** stability is a real property that needs $n \ge 2$ to demonstrate. The current pre-fix abstract leaned on $n = 5$ for E1's quality claim; the rewritten paragraph aims for a more modest **stability** claim, but it still needs at least $n = 2$ to be a claim at all. Three seeds (0, 1, 2) is the minimum that gives a defensible "no divergences and val_ppl reproducible to within noise" statement.

**Falsifiable prediction:** seeds 1 and 2 land within $\pm$ 2 PPL of 33.55 (so val_ppl in the 31.5–35.5 range). No NaN divergences. If either seed diverges, that's a stability concern worth investigating before claiming "trains stably".

**Companion-repo deliverable:** populates `notebooks/conservative_arch/scaleup/results/pilot_splm_fixed_seed{1,2}/` and updates the §14.4 number in the rewritten paper to a $\bar{\text{ppl}} \pm \sigma$ form.

**Sequencing note:** can launch the moment the in-flight fixed multi-ξ pilot finishes (so MPS runs both back-to-back without gap).

### 5.7 P3.1 — Mini γ-sweep under the fix (only if §4.7 Option A is chosen)

**What:** Re-run the §14 γ-sweep at three γ values (e.g. $\gamma \in \{0.10, 0.30, 0.85\}$ to bracket the previous optimum) under the leak-corrected integrator. Each cell runs the same 4000-step pilot config as the fixed single-ξ baseline.

**Cost:** ~7 h per cell × 3 cells = ~21 h on MPS.

**Fills:** §4.7 of this doc, Option A (re-run a small γ-sweep under the fix). Yields new specific numbers for the γ-optimum claim and either confirms or revises the pre-fix $\gamma^{\ast} \approx 0.30$ result.

**Why this is P3 and not P1/P2:** the γ-sweep narrative on natural transformers (the GPT-2 / Pythia $21/24$ and $24/24$ Decision-β cells) is **independent** of the SPLM γ-sweep and survives the leak fix unchanged. The rewritten §14 can take Option B (drop the specific γ-optimum on the SPLM side, keep the structural "γ-tuning matters" claim) without losing any descriptive content. The specific γ-optimum number on SPLM was never the headline; it was a corroborating detail.

**Recommendation:** **default to Option B**. Run P3.1 only if the reviewer-feedback / submission-target context makes a specific γ-optimum number a hard requirement.

**Falsifiable prediction (if executed):** the optimum γ stays in the $0.10$–$0.30$ neighbourhood (the structural prediction from the corpus-surprisal estimator is unchanged by the fix; what's changed is the absolute val PPL at every γ). The minimum val PPL is in the $30$–$50$ range, an order of magnitude above MatchedGPT, and significantly higher than the leak-driven $144$ that appeared in the buggy γ-sweep.

### 5.8 Recommended day-of ordering after the in-flight fixed multi-ξ pilot finishes

Assuming the fixed multi-ξ 4000-step pilot finishes around 6 AM EDT (≈9.7 h from launch at 20:30 EDT 2026-05-01), the ordering that minimises both wall-clock time and inter-experiment dependencies is:

```text
T+0    fixed multi-xi pilot finishes; run causal_probe + eval_ppl_under_fix
       on its ckpt (~5 min); update §4.6 of bug doc with Prediction B result
T+5m   P1.2  causal probe on natural transformers      (5 min)
T+10m  P1.3  E1 buggy multi-seed inflation forensic    (50 min)
T+1h   P1.1  §15 separator R^2 re-fit on E9 fixed      (30-60 min)
T+2h   P1.4  §A2 wall-clock benchmark on E9 fixed      (30 min)
T+2.5h All P1 done; write Buggy_vs_Fixed_SPLM_Comparison_Report
       (no compute; ~1 h writing)
       --- machine free for the rest of the day ---
T+~10h Launch P2.1 multi-seed (seeds 1, 2) overnight   (14 h on MPS)
T+24h  P2.1 finishes; multi-seed numbers landed in §4.4 placeholder
```

Total elapsed: ~24 h from when the fixed multi-ξ pilot finishes; total active MPS-bound time: ~17 h, with ~14 h of that being the unattended overnight P2.1 run. P3.1 is *not* in this plan; if Option A becomes preferred, add it as a separate ~21 h overnight run.

### 5.9 Summary table

| ID | what | cost | fills | priority |
|---|---|---:|---|---|
| P1.1 | §15 separator R² re-fit on leak-corrected E9 | ~1 h | §4.6, abstract R² claim | mandatory |
| P1.2 | causal probe on natural transformers (MatchedGPT, GPT-2, Pythia) | ~5 min | §4.3 reviewer-defusing paragraph | mandatory |
| P1.3 | E1 buggy multi-seed inflation forensic (5 seeds) | ~50 min | §4.3 inflation table, companion repo `legacy/` archive | mandatory |
| P1.4 | §A2 wall-clock benchmark on leak-corrected ckpt | ~30 min | §4.5 §A2 cleanup | mandatory |
| P2.1 | multi-seed leak-corrected E9 (seeds 1, 2) | ~14 h overnight | §4.4 rewritten LM-quality paragraph | recommended |
| P3.1 | mini γ-sweep under fix (3 γ cells) | ~21 h | §4.7 Option A (only if chosen) | optional |

After P1 (mandatory) lands, the paper restructure has every leak-corrected number it strictly needs to be defensible. After P2 lands, the rewritten §14 stability claim is a real claim. P3 is the only experiment whose value depends on a separate decision (Option A vs B).

---

## 6. Journal strategy: from three slices to one paper plus optional spin-offs

The original "preprint, then cut into slices" plan envisioned three target journal submissions. Post-fix, the slicing changes:

| slice | pre-fix viability | post-fix viability | venue |
|---|---|---|---|
| **Framework + descriptive validation** (§§02–13 + §14 first half + §16) | strong | **strongest of the three now** — the paper's intellectual core is unchanged | NeurIPS / ICML / JMLR (main paper) |
| **Shared-potential separator methodology** (§15 restructured) | viable as separate methods note | viable; SPLM number re-fits on leak-corrected weights but methodology and attention numbers survive | TMLR (tools) or ICLR methods track |
| **SPLM as architecture in its own right** | viable as quality-driven submission | **not viable as standalone** | folded back into main paper as a single supporting section |

The third slice is dead as a standalone submission. The first two slices become **stronger** in some respects — the framework paper no longer carries the implicit aspirational subtext "and our architecture is also a competitive replacement", which was always the most fragile claim in the abstract. The methodology slice is unchanged, and gains a clearer lane (the separator works *better* as a diagnostic tool than as a competitive comparator).

A useful by-product: there is no longer a concrete "SPLM efficiency" workshop submission either. The *FLOP and wall-clock* claim survives, but presenting it as a "$O(L \cdot d)$ per-token decode is X× cheaper than attention at $T \ge 8000$" workshop note is a thin reed without the quality story. It can fold into the main paper's §A2 instead.

---

## 7. What gets stronger after the restructure

Three things become genuinely better in the post-fix paper:

### 7.1 Tension resolution between prescriptive and descriptive

The pre-fix paper had a latent tension. If SPLM (built to be conservative-by-construction) is competitive with attention on PPL, AND attention is locally conservative (descriptive finding §14, §16), then *why* does paper_v3 advocate for SPLM specifically? The answer the pre-fix paper gave was "global vs. local conservativity" — but that was a subtle distinction to defend in a quality-comparison setting where the two architectures had similar numbers.

Post-fix, the tension is gone: attention is locally conservative but globally not shared-potential, and is genuinely better on PPL because of that very fact (the global non-conservativity is what allows attention's content-conditioned mixing). SPLM is the minimal globally-shared-potential architecture, and it pays a quality price for that constraint. The framework explains both architectures and the quality gap between them.

### 7.2 Sharper interpretation of attention's "almost-conservative" character

§16 of `paper_v3` argues that attention's trained trajectories are "near-geodesic" — i.e., that attention almost behaves as if it followed a damped Euler-Lagrange flow on a shared potential, but doesn't quite. Pre-fix, this had a slightly awkward implication: if attention is *almost* there, why not push it the rest of the way (i.e., to SPLM), and what would we lose? The answer post-fix is clear: pushing all the way to a globally-shared-potential architecture costs ~4× PPL because content-conditioned mixing is doing real work. The "almost" matters.

### 7.3 The separator becomes a clean diagnostic tool

Pre-fix, the shared-potential separator was framed simultaneously as a methodology (§15) and as part of an argument that SPLM is structurally distinguished from attention (§§14–15). Post-fix, the methodology framing is the cleaner one: the separator is a tool for *measuring* how close any given architecture is to a globally-shared-potential dynamics, on a continuous scale from $R^2 = 0.45$ (pretrained GPT-2) to $R^2 \approx 1.0$ (SPLM by construction). This is a generalisable diagnostic, useful for any future architecture comparison; the SPLM "score" is just one calibration point, not the headline.

---

## 8. What gets weaker after the restructure

Two things get weaker, and they deserve honest acknowledgement:

### 8.1 The implicit aspirational sub-text of the paper is gone

Pre-fix, the paper carried (without quite stating) the implication "and so a long-term direction for the field is to replace attention with conservative-by-construction architectures." Post-fix, that direction is not supported by the data. Attention works; SPLM works less well. The paper becomes a more constrained theoretical / methodological contribution and loses the speculative-architectural-ambition flavour. This is intellectually honest but reduces the paper's emotional/strategic weight.

### 8.2 Some empirical paragraphs in the abstract become less compact

The "$36.4\%$ mean-perplexity reduction, Welch's $t = 14.4$, $95\%$ CI on the gap $[+45.43, +63.53]$ ppl" is a genuinely punchy claim. The replacement — "SPLM trains stably and learns; quantitative PPL is not competitive with attention at this scale, by a factor of ~4×" — is honest but less rhetorically crisp. There's no way around this.

---

## 9. Open decisions

A few specific calls remain for the user to make once the current empirical runs (buggy multi-ξ short run + fixed multi-ξ pilot) finish:

1. **Option A vs Option B for §14 γ-sweep** (§4.7, with the experimental cost laid out as P3.1 in §5.7). Default recommendation: **Option B** — drop the specific γ-optimum on the SPLM side, keep the structural γ-tuning narrative; the natural-transformer Decision-β reproduction is independent and survives. Pick Option A only if the submission target makes a specific γ-optimum number a hard requirement.
2. **Which subset of §5 to commit to running?** Default recommendation: **all of P1 (mandatory) + P2.1 (recommended overnight)**, skip P3.1 unless the §14 γ-sweep choice changes.
3. **Should the methodology slice (§15) be split into a separate TMLR submission?** Cleaner story, but reduces the main paper's empirical surface area. Defensible either way.
4. **Should the new "causal-leak correction" subsection appear as §14.0/§14.1, or as an appendix?** §14.0 is more honest (front-and-centre disclosure); appendix is more conventional. I'd argue for §14.0.
5. **Once all four runs (single-ξ buggy, single-ξ fixed, multi-ξ buggy, multi-ξ fixed) are in, should we publish the comparison report as a separate technical note?** Pro: forensic completeness. Con: more documents to maintain.

---

## 10. Summary

The leak fix kills one specific story the paper was telling (SPLM as a competitive LM architecture). It does **not** kill the paper. It actually makes the paper more coherent: the framework explains attention (descriptive primary contribution), the separator diagnoses architectures (methodological secondary contribution), and SPLM is one calibration point demonstrating that the dynamics is trainable as a circuit (constructive footnote). The original "preprint then slice into three submissions" plan collapses into "one main paper, one optional methodology spin-off, and the SPLM-architecture submission is dead."

The user's instinct — "SPLM is now a curiosity supporting the main idea, nothing more" — is exactly right. This document records the supporting reasoning and the concrete restructuring proposal that follows from it. The restructure should proceed once the multi-ξ runs in flight finish, the comparison report ([`docs/Buggy_vs_Fixed_SPLM_Comparison_Report.md`](Buggy_vs_Fixed_SPLM_Comparison_Report.md), TBD) is written, and the targeted re-runs catalogued in §5 — at minimum the P1 mandatory set — have landed.

---

## 11. Provenance

This document is the written form of the strategic discussion held on 2026-05-01 between the user and the assistant, immediately after the multi-channel buggy training run produced its dramatic val_ppl = 1.26 result at step 600. The discussion was triggered by the user's observation:

> "I no longer see the need of independent paper on SPLM design as this architecture is nothing more than curiosity which could support the main idea of the paper on Semantic Simulation but nothing more. If this is the case we will need to redesign paper_v3 in a very substantial way. What is your take?"

This document records the take, in detail, for future reference and for any reviewer who asks why the paper was restructured the way it was.

---

## 12. Follow-up (added 2026-05-03): narrative options after the R6 ladder matures

> **Status.** Drafted 2026-05-03, after the R6.a / R6.e / R6.i / R6.h.1 ladder of multi-channel ξ basis-class experiments matured. §1–§11 above were written on 2026-05-01 immediately after the leak fix, when the post-fix empirical bound was the single E9 single-ξ ckpt at val_ppl = 33.55. The R6 program subsequently established that with multi-channel ξ + LayerNorm-after-step + leak-free integrator, the *best* SPLM variant (vanilla K-EMA at K = 4) reaches val_ppl ≈ 14.78 on TinyStories at the 4000-step pilot scale. This is materially better than the §1 33.55 number but still ~4–5× MatchedGPT's val_ppl ≈ 8. §12 records the strategic reframing implications now that the *post-R6* ceiling is known.
>
> **Companion documents (added since §1):**
> - **R6 ladder design + results:** [`docs/Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md`](Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md) §10–§12
> - **Empirical evidence for the second-order-ODE structure:** [`docs/Evidence_for_second_order_ODE_governing_evolution.md`](Evidence_for_second_order_ODE_governing_evolution.md) — promote from doc to §-level paper section per §12.4 below
> - **Causal-leak empirical comparison:** [`docs/Causal_Leak_Empirical_Comparison_Report.md`](Causal_Leak_Empirical_Comparison_Report.md)
>
> **Relation to §1–§11:** The R6 results do not invalidate any §1–§11 conclusion. They *sharpen* the empirical bound used by §4.4 ("val_ppl ~30 at the 4000-step pilot scale") to ~14 PPL with the multi-channel architecture, and they confirm the §3 conclusion that SPLM-as-an-independent-paper is dead. What §12 adds is a more careful framing for the role SPLM plays *inside* paper v3 once we know the architecture's true (leak-free, fully-tuned) PPL ceiling.

### 12.1 The post-R6 empirical state

Five leak-free SPLM variants and one matched baseline have been run at the same pilot config (K = 4 where applicable, 4000 steps, batch 16, block 512, fixed γ = 0.30, identical seed and LR schedule):

| run | basis | val PPL @ 4000 | gap to MatchedGPT (~8) |
|---|---|---:|---:|
| R6.h.0 = K-EMA pilot | learnable $\{\alpha\_k\}$, hand-picked init | 14.78 | +6.78 (~85 %) |
| R6.h.1 = K-EMA log-spaced (in flight) | learnable $\{\alpha\_k\}$, log-spaced init τ_max = 100 | predicted ≈ 14.5–15.0 | +6.5–7.0 |
| R6.i = S4D legt-init | learnable diagonal complex $A$, $B$, $\Delta$ | 16.85 | +8.85 (~110 %) |
| R6.e = HiPPO-LegT learn-Δ | LegT spectrum, learnable $\Delta$ | 17.45 | +9.45 (~118 %) |
| R6.a = HiPPO-LegT fixed-Δ | LegT spectrum, fixed | 19.82 | +11.82 (~148 %) |
| MatchedGPT | attention | ~8 | — |

Two structural observations crystallised in [`Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md`](Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md) §12 (the *V_θ-fit-difficulty* synthesis):

1. **The intra-SPLM spread (~5 PPL) is much smaller than the SPLM-vs-GPT gap (~7 PPL).** Switching basis class moves val PPL by ≤ 5 PPL; switching architecture class (gradient flow on a scalar potential ↔ attention) moves it by ≥ 7 PPL.
2. **The SPLM ceiling at K = 4 is bounded below by V_θ's MLP-fit difficulty, not by ξ's information content.** R6.c showed orthogonal bases carry strictly more information per ξ vector; they still lose to K-EMA because V_θ-as-MLP cannot extract that information at the 4000-step budget. This argues that the SPLM ceiling isn't moved much by *any* basis-class search — DPLR, S4D-Lin, K-EMA Fix 3, etc. would all cap near ~14 PPL.

The honest conclusion: **the leak-free SPLM family, as currently architected (scalar V_θ, MLP head, no token-token routing), caps around val_ppl 13–15 on TinyStories at 16 M params**. Closing the gap to MatchedGPT's ~8 PPL would require a categorical change — most likely adding token-token routing in some form, which the SARF / Lagrangian framework can accommodate but the *current* SPLM realisation does not.

### 12.2 The framing question

Given that ceiling, the paper-v3 authoring question is:

> If the best leak-free SPLM design caps at ~14 PPL on TinyStories and a matched-parameter GPT reaches ~8 PPL, **what narrative arc keeps the paper honest, defensible, and citable** without depending on a quality-competition claim?

§4 of this doc gave a partial answer (reframe SPLM as a constructive existence-proof, not a competitor). §12 sharpens that answer with three concrete narrative options now that the ceiling is empirically pinned.

### 12.3 Three narrative options

#### 12.3.1 Option 1 (recommended) — SPLM as measurement device, not competitor

**Thesis.** "What inductive biases does a transformer encode that a physically-grounded sequence model lacks? Quantifying this requires a counterfactual — a *maximally-structured* LM trained on identical data, params, and compute. We construct SPLM as that counterfactual and use the GPT-vs-SPLM gap as a measurement."

The 6–7 PPL gap stops being a liability and becomes the central scientific result: **the gap *is* the contribution, not the SPLM val_ppl number**. The §10–§12 R6 ladder becomes a *mechanistic decomposition of the gap*: which architectural primitives in attention is each PPL of the 6–7 PPL gap attributable to?

**Concrete consequences:**
- §1–§5 (theoretical SPLM construction) stay essentially as written.
- §6–§9 (descriptive validation on pretrained transformers) stay as written — they are independent of the SPLM-quality story.
- §10–§12 multi-channel ξ work, R6 ladder, V_θ-fit-difficulty synthesis (§12 of the design doc) become §s of the paper, framed as a mechanistic decomposition of the SPLM-vs-GPT gap.
- §13–§14 (causal-leak forensic + diagnostic toolkit per §4.3 above) become a stand-alone methodology section.
- §15 (separator) gets its leak-free re-fit per §4.6 above; framing remains diagnostic.
- §16 (paper-v2's conclusion, currently being rewritten — file under git modification) leads with the measurement framing, not the architecture-proposal framing.

**Audience:** NeurIPS / ICML methods, ICLR representation-learning, scaling-laws community. This is a top-tier methods venue paper, not a top-tier system / architecture paper.

**Strengths:** honest (no over-claiming), defensible (even reviewers who think "SPLM doesn't beat GPT, so what?" must engage with the measurement framing), citable ("the 6–7 PPL gap localized to V_θ-fit-difficulty" is a clean handle the rest of the field can pick up).

**Weaknesses:** loses the implicit aspirational sub-text already noted in §8.1 above; less rhetorically punchy than a "we beat the baseline by X%" frame.

#### 12.3.2 Option 2 — Honest catalog: scope and limitations of physics-grounded LMs at GPT-2 scale

**Thesis.** "We rigorously test the conservative-flow LM paradigm against transformers at matched scale. We find a structural ~6–7 PPL gap and trace it to two specific limitations: (a) scalar V_θ has no token-token routing analogue to attention, (b) the multi-channel ξ context summary has a V_θ-fit-difficulty bottleneck. We document the gap, the diagnostics, and the family of attempted fixes."

Less ambitious than Option 1; doesn't try to repurpose SPLM as a measurement device. Just reports what was learned.

**Audience:** ML reproducibility / negative-results tracks, physics-ML workshops, JMLR-style methodology venues.

**Strengths:** lowest revision effort (closest to §4's existing plan); easy to defend; reproducible-research community will reward the transparent reporting.

**Weaknesses:** lower impact than Option 1; harder to position at a top-tier venue.

#### 12.3.3 Option 3 — Reframe as benchmark / toolkit paper

**Thesis.** "Here is a leak-free, instrumented, physics-inspired LM testbed at GPT-2 scale, plus the diagnostic framework (causal-leak audit, channel-correlation diagnostic, R6 falsifiability ladder, second-order-ODE evidence pipeline). Any future physics-LM paper can use this as a baseline." Empirical results become *use-cases*, not the thesis.

**Audience:** ICLR datasets-and-benchmarks track, NeurIPS datasets and benchmarks, MLSys.

**Strengths:** lowest-effort rewrite if the codebase is what gets shipped; benefits from the substantial diagnostic infrastructure already built (`causal_probe.py`, `diagnose_xi_channel_correlations.py`, `eval_ppl_under_fix.py`, `splm1_leakfree_re_eval.py`); cleanly orthogonal to any quality-comparison claim.

**Weaknesses:** discards the §1–§5 theoretical content that is the paper's intellectual core; benchmark venues usually want a different rhetorical structure than the descriptive framework chapters provide.

#### 12.3.4 Recommendation

**Pursue Option 1.** It preserves §1–§5 (theory) and §6–§9 (descriptive validation on pretrained transformers) — which constitute the paper's intellectual core — while reframing §14–§16 (the LM-quality empirical content) as measurement rather than competition. Options 2 and 3 are viable fallbacks if reviewer feedback at first submission suggests the measurement framing is not landing.

The remainder of §12 elaborates Option 1.

### 12.4 What SPLM uniquely has that GPT doesn't (do not bury this in v3)

Even at val_ppl ~14 vs ~8, SPLM has *qualitative* properties that matter independently of the PPL number. Paper v3 should foreground these as primary contributions, not as caveats:

1. **Provable causality.** Post-fix SPLM is exactly causal under the regression test in `notebooks/conservative_arch/causal_probe.py`. GPT happens to be causal because attention masking enforces it as a coincidence of architecture, not as a guarantee. SPLM-style causality auditing is generically useful for any sequence-LM that introduces non-trivial autograd structure (state-space models, retrieval, diffusion-as-LM, etc.).

2. **Approximate energy conservation.** The SARF / second-order-ODE invariant. The currently-open companion document [`docs/Evidence_for_second_order_ODE_governing_evolution.md`](Evidence_for_second_order_ODE_governing_evolution.md) is *exactly* this contribution. **Recommendation: promote it from a doc to a §-level section in paper v3.** This is the strongest qualitative claim SPLM has and is currently buried in `docs/`.

3. **Mechanical interpretability of token trajectories.** Each token has a position $h\_t$, a velocity $v\_t$, a learned per-token mass $m\_t$, and an explicit potential $V\_\theta(\xi\_t, h\_t)$ governing its evolution. This is *much* more interpretable than a residual stream and is a property GPT does not have at all. For downstream applications where interpretability matters (scientific computing, physics simulation, model auditing), this is a real advantage even at worse PPL.

4. **The V_θ-fit-difficulty mechanism as a transferable finding.** "Smooth, multi-scale, partially-redundant context summaries are easier to fit than orthogonal ones" applies to *any* downstream MLP, not just SPLM's V_θ. This is a research-level finding — see [`Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md`](Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md) §12.3 for the four candidate mechanisms. Worth its own paper-v3 subsection.

5. **The basis-class hierarchy at LM scale.** The R6 ladder (K-EMA / HiPPO-LegT / S4D / DPLR) is the first careful comparison of these context-summary primitives *as drivers of LM val_ppl* at GPT-2-class scale. Most prior SSM work was on Long Range Arena, not LM. This is independently valuable to the SSM community.

### 12.5 Concrete structural changes for v3 (Option 1 implementation)

Building on §4 above, with the post-R6 ceiling locked in:

1. **Abstract + §1 positioning paragraph.** Rewrite to lead with "we use SPLM as a counterfactual to measure attention's contribution to LM quality at matched scale" rather than "we propose SPLM as an architecture". Single sentence stating "we do not claim PPL parity with attention at this scale" near the top of §1, once.

2. **§§02–13 framework + descriptive validation.** Untouched (already untouched per §2 of this doc). Reviewers reading the framework chapters will see the most polished material.

3. **§14 (current LM-quality section).** Replace with three subsections:
    - §14.0 / §14.1 causal-leak correction (per §4.3 above).
    - §14.2 controlled-comparison setup: SPLM-best (K-EMA log-spaced or whichever R6.h pilot wins) vs MatchedGPT, identical training conditions, single side-by-side trajectory plot.
    - §14.3 mechanistic decomposition of the gap: R6 ladder summary, V_θ-fit-difficulty mechanism. References the §12 of [`Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md`](Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md).

4. **§15 (separator).** Per §4.6 above, methodology + leak-free re-fit. Unchanged structural plan.

5. **§16 (paper-v2's conclusion, currently under git modification).** Lead paragraph reframes around the measurement contribution, not the architecture proposal. Concrete edit pass deferred until after the in-flight R6.h.1 lands and §12.6 below is acted on.

6. **New §17 (or §14.4): the second-order-ODE evidence pipeline.** Promote [`docs/Evidence_for_second_order_ODE_governing_evolution.md`](Evidence_for_second_order_ODE_governing_evolution.md) from a doc to a §. This is the strongest qualitative claim SPLM has and is currently invisible to anyone reading only the paper.

7. **New §18 (or appendix-promoted): causal-leak audit methodology.** The diagnostic framework (`causal_probe.py`, `eval_ppl_under_fix.py`, `splm1_leakfree_re_eval.py`) is itself citable. Currently this material is split between [`docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md), §4.3 of this doc, and the regression-test code; v3 should consolidate into one §.

8. **New §19 (or appendix): the SPLM-1 / SPLM-2 leak-free re-evaluation.** Once `splm1_leakfree_re_eval.py` runs the 6 ckpts (deferred per §11.x of the parent restructure plan, scheduled after R6.i + R6.h finish), the resulting (PPL_buggy, PPL_fixed) per-seed comparison closes the SPLM-1 vs SPLM-2 chapter cleanly.

Items 6, 7, 8 are *additions* relative to §4's plan, motivated by the post-R6 framing in Option 1. Items 1–5 are refinements of §4's existing plan.

### 12.6 Publication versioning: do not delete, version

A practical concern raised by the user: paper v2 is already published on SSRN and Zenodo with the title *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference*. Does the paper-v3 reframing require taking down or invalidating those publications?

**No. Deletion is unnecessary and would be the wrong move.** Both platforms are explicitly designed for iterative scientific work and have first-class versioning support:

- **Zenodo:** new version on the existing record creates a new DOI for the new version while preserving the original DOI in perpetuity. A "concept DOI" auto-resolves to the latest. Workflow: log in → existing record → "New version" → upload v3 PDF → fill in changelog → publish. Citations to the v2 DOI continue to resolve to v2; new citers naturally land on v3.
- **SSRN:** revisions are a first-class feature on a single record. Workflow: log in → My Papers → Revise/Update → upload v3 PDF → save. Revision history is preserved.

This is **standard practice**, not a workaround. Most major ML preprints (GPT-3, BERT, Llama, attention-is-all-you-need) have multiple preprint versions with substantive body changes between versions. The community fully expects iterative revision, and reviewers / citers reward visibly self-corrective authorship.

The only situation where deletion is on the table is when a previous version contains an *actively misleading* claim (e.g., a fabricated number) — and even then the standard remedy is *retraction with notice*, not deletion. The causal-leak inflation documented in v2 is *not* in that category: it was an honest scientific issue discovered post-hoc and reported transparently. That's normal scientific practice, not retraction-grade misconduct.

### 12.7 The existing v2 title already accommodates Option 1

The published title:

> ***Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference***

makes *zero* claims about beating any baseline. It claims a *prescriptive framework*. That's exactly the Option 1 frame: the Lagrangian framework is the contribution, and SPLM's empirical val_ppl vs GPT is a *test of* the framework, not a *competitive claim by* the framework.

**The title can survive into v3 essentially unchanged.** §4.1 of this doc proposed a title shift from "*Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference*" → "*Semantic Simulation: A Lagrangian Account of Transformer Inference Dynamics*". Under Option 1 the original title actually fits *better* than the proposed shift because "Prescriptive Lagrangian Framework" is exactly what Option 1 needs. **Recommendation: revert the §4.1 title change and keep the original title for v3.** What changes is §1's positioning paragraph and §6–§19, not the title.

#### 12.7.1 Subtitle: minimal plural→singular edit (decided 2026-05-03)

The v2 subtitle decomposes into three intellectual pillars; the post-R6 / Option 1 framing leaves two of them untouched and only requires a one-character edit on the third.

> ***Original v2 / current v3 subtitle:** Conservative-by-Construction Language Models and the Shared-Potential Separator, with a Correspondence to Joint Embedding Predictive Architectures*

| pillar | content | post-Option-1 status |
|---|---|---|
| 1 | "Conservative-by-Construction Language Model**s**" | structurally intact (the conservativity *is* the property that makes SPLM useful as a counterfactual), but the **plural** implies a *class to promote* — exactly the framing Option 1 rejects |
| 2 | "the Shared-Potential Separator" | *strengthens* — becomes the core architecture-distance diagnostic. Keep verbatim. |
| 3 | "Correspondence to Joint Embedding Predictive Architectures" | independent of every empirical surprise (leak fix, R6 ceiling). Keep verbatim. |

Three subtitle options were considered:

| version | pillar-1 wording | rhetorical effect |
|---|---|---|
| **A — minimal** | "Conservative-by-Construction Language Model**s**" → "**A** Conservative-by-Construction Language Model" | singular indefinite signals *one* working realisation, not a class. Zero structural impact on body. |
| **B — measurement-frame** | "Conservative-by-Construction Language Models" → "A Conservative-by-Construction **Counterfactual**" | aligns hardest with Option 1, but "counterfactual" is jargon-y without a §1 setup. |
| **C — full Option-1 banner** | "Conservative-by-Construction Language Models" → "**Quantifying Attention via** a Conservative-by-Construction Counterfactual" | most aligned with Option 1; reads like a sentence; subtitle becomes long. |

**Decision: Option A** (minimal plural → singular).

Rationale:

1. The singular indefinite article correctly signals that the paper presents *a single working construction*, not an architectural class to promote — the exact Option 1 stance.
2. *"Conservative-by-construction"* should stay, because that property is precisely what makes SPLM useful as the measurement counterfactual. The whole logic of "use SPLM as a counterfactual to quantify what attention provides" relies on SPLM being the *most-structured-possible* LM, and "conservative-by-construction" is what makes that true.
3. No structural changes to §§02–13 or §15 or §10 are implied. B and C would require §1's positioning paragraph to introduce "counterfactual" as a defined term, which is more revision work for marginal gain.
4. The plural→singular swap is a one-character fix that reviewers will read as honest self-correction, not as a major repositioning. That is exactly the kind of low-cost, high-trust signal Option 1 needs.

**Final v3 title block (decided 2026-05-03):**

> ***Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference***
> *A Conservative-by-Construction Language Model and the Shared-Potential Separator, with a Correspondence to Joint Embedding Predictive Architectures*

### 12.8 Transparent-changelog approach for v3

Two text additions, neither of which is hard:

1. **Footnote at the top of v3's title block:** "This is v3 of the manuscript. Previous versions: v2 (DOI:…), v1 (DOI:…). v3 incorporates a controlled comparison against transformer baselines at matched parameter count, a mechanistic decomposition of the resulting performance gap, and a leak-free re-evaluation of v2's empirical claims (§14.0). Empirical claims have been retained where they survive the leak-free re-eval and have been updated where they do not."

2. **One-page changelog appendix at the end of v3,** summarising: causal-leak fix and its effect on previously published numbers, R6 ladder additions, V_θ-fit-difficulty section, leak-free SPLM-1 vs SPLM-2 re-eval. Increasingly common in revised preprints and is well-received.

This is the most credibility-preserving move: be visibly self-corrective with full provenance. In ML, papers whose authors revise *with full provenance* tend to be cited *more*, not less, because the revision itself becomes a piece of methodological evidence.

### 12.9 Actionable workflow

Sequencing for paper v3 publication, assuming Option 1:

1. **Do not touch the existing SSRN / Zenodo records.** Leave them as historical record.
2. **Wait for the in-flight R6.h.1** (4000-step pilot, eta ~21:00 EDT 2026-05-03) and decide on R6.h.2 / R6.h.3 / R6.i_v2 next steps. The post-R6 SPLM-best val PPL needs to be locked before §14.2 can be written.
3. **Run the deferred SPLM-1 leak-free re-eval** ([`splm1_leakfree_re_eval.py`](../notebooks/conservative_arch/first_order_ablation/splm1_leakfree_re_eval.py)) once the user is ready. Result feeds §19 and any §15 paragraph touching SPLM-1.
4. **Write paper v3 freely** with the Option 1 framing. The §12.5 structural changes are the concrete edit list.
5. **When v3 is ready,** push it as a new version on Zenodo (gets a new DOI auto-assigned) and as a revision on SSRN. Add the §12.8 footnote and changelog appendix.
6. **Update CV / website / citations** to point to the latest version. Old DOI remains a valid pointer for anyone citing v2.

### 12.10 Open decisions added by §12

A few concrete calls for the user:

1. **Confirm Option 1 over Options 2 / 3** as the paper-v3 framing. Default recommendation: **Option 1**. Pick Option 2 if reviewer feedback at first submission suggests the measurement framing is not landing; pick Option 3 only if the paper would benefit from being repositioned as a benchmark / toolkit submission.
2. **Title revert.** Keep v2's title (*Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference*) for v3 unchanged, replacing §4.1's proposed shift. Default recommendation: **keep**. **Subtitle decided 2026-05-03 (§12.7.1): change "Conservative-by-Construction Language Models" → "A Conservative-by-Construction Language Model" (plural → singular indefinite); pillars 2 and 3 of the subtitle unchanged.**
3. **§17 / §18 / §19 promotion calls.** Promote the second-order-ODE evidence (currently in `docs/`) to a paper §; promote the causal-leak audit methodology to a paper §; add the SPLM-1 leak-free re-eval as a §. Default recommendation: **all three**. Each is a distinct contribution and benefits from § visibility.
4. **Changelog format.** Footnote-in-§1 + appendix-changelog (per §12.8), or §1-only with an inline narrative paragraph. Default recommendation: **both**.
5. **When to start writing v3?** §12.9 step 4. The §14.2 numbers for the R6.h.1 / R6.i_v2 cells are not strictly required to start §1–§13 + §17–§18 of v3; only §14 needs the post-R6 numbers locked.

### 12.11 What §12 changes about §1–§11

§1–§11 of this doc were written with one specific empirical bound in mind: the leak-free single-ξ E9 ckpt at val_ppl 33.55. §12's R6 ladder reduces that bound to ~14 PPL with multi-channel ξ. The strategic conclusions of §1–§11 still hold:

- **§3 (SPLM as standalone paper is dead) — reinforced.** Even at val_ppl 14, SPLM trails MatchedGPT's val_ppl ≈ 8 by ~75 %. The §3 conclusion that no SPLM-quality submission can stand on its own is *more* robust at the post-R6 ceiling, not less.
- **§4 (concrete restructuring) — refined.** §12.5 above adds 3 new § promotions (§17 / §18 / §19) that §4 did not anticipate. §4.4's placeholder val_ppl ~30 should be updated to "val_ppl ~14 (multi-channel ξ at K = 4) or ~33 (single-ξ baseline) depending on the §14.2 baseline of choice".
- **§5 (targeted re-runs) — partially obviated.** P1.1 / P1.2 / P1.4 still apply. P1.3 (E1 forensic) is still mandatory. P2.1 (multi-seed leak-corrected E9) is now arguably *less* important than a multi-seed multi-channel ξ run (the new SPLM-best baseline). P3.1 (γ-sweep) recommendation is unchanged: default to Option B.
- **§§6–10 (journal strategy, what gets stronger / weaker, summary) — unchanged.** §12 does not alter the slicing argument or the strengths/weaknesses analysis. It only makes the post-fix bound concrete enough to be quotable.

The sense in which §12 *refines* rather than *replaces* §1–§11: §1–§11 was written before the R6 ladder existed, with a placeholder val_ppl in mind. §12 fills that placeholder with the empirically-determined ceiling and resolves a few open decisions that depended on it.

### 12.12 Provenance for §12

§12 records the strategic discussion held on 2026-05-03 between the user and the assistant, after the R6.i pilot completed and the R6.h.1 pilot launched. The discussion was triggered by the user's question:

> "So if we find out that our best SPLM design cannot go below 13–14 ppl (vs 8 ppl from matched GPT) how this would shape our narrative in paper v3?"

followed by the practical concern:

> "But I have uploaded a paper with the title Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference to SSRN and Zenodo — wouldn't such reframing require me to delete the paper publication from those?"

§12 records the answers, in detail, for future reference and for any reviewer who asks why paper v3 took the framing it did and what its relation to v2 is.
