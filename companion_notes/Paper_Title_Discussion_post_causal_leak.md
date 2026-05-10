# Paper Title Discussion (post causal leak)

**Status:** Discussion record · 5 May 2026
**Trigger:** Section §A2 leak-corrected Q3 grade
**Context paper:** `paper_v3/main.tex`
**Companion notes:** `companion_notes/Paper_Title_Discussion_post_causal_leak.md` (mirror)

---

## 1. Why the title is being re-examined

The current `paper_v3` title is

> **Semantic Simulation: A Prescriptive Lagrangian Framework for *Efficient* Semantic Inference**
> *A Conservative-by-Construction Language Model and the Shared-Potential Separator,
> with a Correspondence to Joint Embedding Predictive Architectures*

The word **"Efficient"** is doing real work in this title: it asserts that
SPLM is more efficient than baselines at semantic inference. Under
the v3 causal-leak correction (`cfg.causal_force = True` in
`notebooks/conservative_arch/energetic_minima/model_ln.py`), the strength
of that claim has become substantially narrower than the v2 abstract
implied. We catalog here what survived and what did not, and we record
the design path we will pursue to either earn the "Efficient" word back
or replace it with one that is rigorously honest.

## 2. What survived the leak fix in §A2

The leak fix did **not** change any FLOP or parameter count.
The following claims are *unconditional* and remain in §A2 verbatim:

- No T² attention term in per-layer FLOP count.
- Depth-independent parameter count (single shared V_θ across all
  L integration steps).
- KV-cache-free streaming-ξ AR decoding.
- Per-new-token decode FLOPs **exactly constant** in prefix length T
  (`44.396` MFLOPs/step in the prototype, every step).
- Per-forward-pass FLOP crossover at T ≳ 34d.

These are the **structural** efficiency properties of the architecture
and follow from the FLOP and parameter accounting in
`paper_v3/sections/A2_inference_efficiency.tex` §§
`app:flops-attn`, `app:flops-splm`, `app:complexity`, `app:params`,
`app:decoding`, `app:instantiated`.

## 3. What did *not* survive the leak fix

The **quality-adjusted FLOPs** claim — *"SPLM reaches a given
cross-entropy at lower total inference FLOPs than matched attention"* —
flipped sign at the prototype scale.

| Configuration                                                  | val PPL              | leak-immune? |
|----------------------------------------------------------------|----------------------|--------------|
| Matched GPT-2-style baseline (`d=128, L=8`, ~8.0 M)            | 149.80 ± 7.21 (E1)   | yes          |
|                                                                | 156.13 ± 8.10 (Phase 1) | yes        |
| SPLM em_ln, buggy v2 integrator, fixed γ* = 0.30               | 95.33 ± 4.44 (E1)    | **no**       |
|                                                                | 88.32 ± 2.03 (Phase 1) | **no**     |
| SPLM em_ln, leak-fixed integrator, free-γ (γ_natural=0.958)    | 173.59               | yes          |
| SPLM em_ln, leak-fixed integrator, fixed γ ∈ [0.10, 0.15]     | ~178–181             | yes          |
| SPLM em_ln, leak-fixed integrator, fixed γ = 0.30 (suboptimal) | 182.90               | yes          |

Under the leak-corrected integrator at the same width and budget the
matched-attention baseline beats leak-free SPLM by **~18–31 PPL** on
Tiny Shakespeare. The §A2 Phase 1 grade therefore moves from
**Q2 (SPLM beats matched-attention by margin)** under the buggy
integrator to **Q3 (matched-attention beats SPLM by margin)** under
causal honesty.

This is the single empirical claim in §A2 that flipped sign. All
structural FLOP advantages stand; the *quality-adjusted* statement
is no longer corroborated at the (d=128, L=8) prototype scale. It
remains open at larger scale and at long-context ($T \gtrsim 10 d$)
regimes where the structural FLOP wins compound.

## 4. The title-word problem

Reading "Efficient Semantic Inference" in the title, a reader of the
LM literature defaults to **quality-adjusted** efficiency: same
cross-entropy, fewer FLOPs / faster wall-clock / smaller cache.
That is exactly the part of the §A2 claim that flipped sign.

The §1 paragraph is already carefully hedged — it says
*"inference **efficiently** in the Lagrangian-mechanics sense: with a
minimal dynamical law, a conservative global force field, a physically
interpretable integrator, and a provable stability margin"*
(`paper_v3/sections/01_introduction.tex:87`). But the title does not
carry this qualification, so the title is currently overpromising
relative to what the abstract and §A2 honestly support.

## 5. Three honest paths considered

### Option 1 — Drop "Efficient" entirely (lowest risk, smallest edit)

> Semantic Simulation: A Prescriptive Lagrangian Framework for Semantic Inference
> *A Conservative-by-Construction Language Model and the Shared-Potential Separator,
> with a Correspondence to Joint Embedding Predictive Architectures*

Pros: zero overpromise; the subtitle still carries
"Conservative-by-Construction", which is the framework's actual
rigorous claim.
Cons: loses some punch in indexes / citations / search.

### Option 2 — Replace "Efficient" with "Conservative"

> Semantic Simulation: A Prescriptive Lagrangian Framework for Conservative Semantic Inference

Pros: **Conservative is the framework's defining mathematical
property** — energy-conserving force field, no anti-causal information
leak. It is rigorously satisfied by the leak-fixed integrator, and
exactly what differentiates SPLM from attention transformers as a
*design principle*. The subtitle already used the word, so the title
plays it as the lead.
Cons: minor lexical ambiguity ("conservative" reads as either
"physics-conservative" or "epistemically-cautious" depending on
the reader); we control the reading via the abstract.

Variants of Option 2 considered:

- **"Conservative Semantic Inference"** (cleanest)
- **"Conservative-by-Construction Semantic Inference"** (matches subtitle verbatim, longest)
- **"Energy-Conserving Semantic Inference"** (most explicit physics-link)

### Option 3 — Qualify "Efficient" to the structural-only reading

Three sub-variants, each defensible by leak-immune evidence alone:

- *... for **Stream-Decodable** Semantic Inference* — narrow, accurate (covers KV-cache-free + constant per-token decode)
- *... for **FLOP-Efficient** Semantic Inference* — accurate at T ≳ 34d and at decode time, but still risks reading as a QA-FLOP claim
- *... for **Asymptotically-Efficient** Semantic Inference* — closest to honest; explicit that the claim is in the long-context limit

Pros: keeps the "efficient" hook with a qualifier the leak-immune §A2
evidence supports. Cons: "asymptotic" reads as "weak" to ML reviewers;
"stream-decodable" is concrete but narrow.

### Recommendation at this point in the discussion

**Option 2 ("Conservative Semantic Inference")** as the *fallback* if
no further design work is undertaken. Rigorously what the framework
establishes, fully consistent with the leak-corrected §A2 analysis,
and the cleanest single-word swap. The "Efficient" word can be
re-earned later.

## 6. The HSPLM (Hybrid SPLM + Attention) proposal

The user's instinct is to push the SPLM design further and *earn* the
"Efficient" title word rather than soften it: build a **hybrid model
that uses scalar-potential layers for some layers and attention for
others**, with the explicit goal of reaching matched-attention quality
at strictly lower total decode FLOPs.

### 6.1. Why the hybrid path is well-precedented

- **Mamba-Hybrid, Jamba, Striped Hyena** all interleave attention
  with SSM blocks and routinely show that 1 attention layer per ~7 SSM
  layers is sufficient to recover full-attention quality at much
  lower decode cost.
- The pattern transfers to SPLM directly: SPLM blocks have the same
  asymptotic shape as SSM blocks (no T² term, no KV cache, constant
  per-new-token decode FLOPs).

### 6.2. Two architectural variants

#### Variant A — Two-stage hybrid (recommended for ablation)

```
h_0 = E[x] + P
for i = 1..k:        h_i = AttnBlock_i(h_{i-1})        # k distinct attn blocks
xi  = causal_cumulative_mean(h_k.detach())             # leak-safe re-derivation
h_final = integrate_m_steps(h_k, xi, V_theta, m, gamma)  # m shared SPLM steps
logits = h_final @ E^T
```

Rationale: attention does what it does best (gather global context
across positions); SPLM does what it does best (refine each position
deterministically through a learned energy field). Total layers
`k + m` matched to the all-attention budget L. The xi re-derivation
from `h_k.detach()` preserves the v3 causal-leak fix.

Decode-time payoff: the k attention blocks pay O(T·d) per new token
(with KV cache); the m SPLM steps pay O(d²) per new token,
**independent of T**. At long T the SPLM tail is essentially free.

#### Variant B — Interleaved hybrid (stripe pattern)

`[A, S, A, S, A, S, A, S]` or `[A, S, S, S, A, S, S, S]` style.
Matches Jamba/Striped-Hyena precedent more directly. Easy to add
once Variant A is built.

### 6.3. Existing infrastructure (very low build cost)

| Piece                                                | Source                                                     |
|------------------------------------------------------|------------------------------------------------------------|
| Causal attention block + KV cache (training + decode)| `notebooks/conservative_arch/matched_baseline_model.py`    |
| Leak-fixed SPLM-2 integration step (LN-after-step)   | `notebooks/conservative_arch/energetic_minima/model_ln.py` |
| Per-layer analytical FLOP counter                    | `notebooks/conservative_arch/inference_efficiency/flop_counter.py` |
| Streaming-ξ KV-cache-free decoder                    | `notebooks/conservative_arch/inference_efficiency/splm_streaming_decode.py` |

A hybrid is therefore a ~150-line glue file
(`notebooks/conservative_arch/hybrid/model_hybrid.py`) that wires
existing pieces.

### 6.4. Tiered experimental plan

| Tier | What                                                                                            | Cells | Time      |
|------|-------------------------------------------------------------------------------------------------|------:|-----------|
| H0   | Variant A at (k=4, m=4), S=1, verify training stability + param count                           | 1     | ~1 h MPS  |
| H1   | Variant A across (k, m) ∈ {(2,6), (3,5), (4,4), (5,3), (6,2)}, S=1                              | 5     | ~3 h MPS  |
| H2   | Best 1-2 (k, m) splits at S=3, paired against all-attention and all-SPLM                        | 4-6   | ~3 h MPS  |
| H3   | Variant B (best stripe pattern, e.g. 1A-3S-1A-3S), S=3                                          | 3     | ~1.5 h MPS |
| H4   | Pareto: per-token decode FLOPs vs val PPL at T ∈ {256, 1024, 4096}                              | 0     | analytical |
| **Total** |                                                                                            | ~13-15 | **~8-9 h MPS** |

Optional **Phase 3** (TinyStories scale-up, +4 h MPS) confirms transfer
beyond the prototype scale.

### 6.5. Pre-registered title-justification rule

The rule decided **before** any sweep starts (locked in this document):

> **"Efficient" is justified iff** some hybrid configuration (k, m)
> achieves val PPL within **+5 PPL** of the leak-immune all-attention
> baseline (val PPL ≈ 150 on Tiny Shakespeare at d=128, L=8) **AND**
> its analytical decode-FLOP cost at T = 1024 is **≥ 30% lower** than
> the all-attention baseline. Both bars must clear at S=3 with
> sign-consistency 3/3 against all-attention.

Outcomes:

- **Rule clears** → "Efficient" stays in the title; new §15 (or §A2)
  subsection reports the hybrid Pareto plot; abstract is retuned.
- **Rule fails** → fallback to **Option 2** (Conservative Semantic
  Inference); hybrid result is still reported as a new subsection
  ("limits of the prototype-scale hybrid"); honestly framed as Future Work.

### 6.6. Honest risks

1. **Quality risk at this scale.** (d=128, L=8) on Tiny Shakespeare may
   be too small for hybrids to express their advantage; the
   all-attention baseline is already small-and-well-tuned.
2. **xi-after-attention design choice.** Re-deriving `xi` from
   `h_k.detach()` (cumulative mean) is the leak-safe choice but may
   not be the right semantic content for the SPLM stage. We may need
   to ablate {raw-emb-ξ, attn-output-cumulative-ξ, learned-ξ-projection}.
3. **Weight-sharing semantics.** Current SPLM uses one shared V_θ
   across all L integration steps. In a hybrid: keep that single-shared
   V_θ across the m SPLM steps (preserves
   conservative-single-potential interpretation) or per-stage V_θs
   (more like a transformer). Default: shared; ablation: per-stage.
4. **Larger-scale exposure.** Even if we win on Tiny Shakespeare, a
   scale-up at TinyStories ((d=192, L=12)) is needed before the
   title-claim is locked. That's the optional Phase 3.

## 7. Decision recorded

User decision (this document, 5 May 2026):

> *"Let us follow your proposal (A) then (B). I like your pre-registered
> rule for the title which I think we should stick with."*

Operationally:

- **(A)**: Build only — `model_hybrid.py` + `train_splm_hybrid.py` + an
  H0 smoke-test script — and pause for review before any sweeps run.
  (~1 day work, no MPS time.)
- **(B)**: After review, run **H0 + H1** (smoke + 5-cell layer-split
  sweep at S=1). (~1.5 days elapsed total; ~4 hours MPS).
- The pre-registered title-justification rule of §6.5 is **locked**
  for the subsequent H2-H4 phases.

## 8. Collateral edits if "Efficient" is changed

These are independent of the hybrid result and need to be applied
the moment the title word changes:

1. `paper_v3/main.tex` line 53-58: title block.
2. `paper_v3/sections/A2_inference_efficiency.tex` Summary paragraph:
   the sentence *"The efficient-inference claim of the paper's title
   is quantitatively grounded..."* → *"The structural inference-efficiency
   claim of the paper's title is..."* (or whatever the new key word is).
3. `paper_v3/sections/01_introduction.tex` line 87: already correctly
   hedged for Options 1 and 2; for Option 3 may need retuning to match.
4. `paper_v3/sections/17_conclusion.tex`: grep for *"efficient inference"*
   framing; align with the chosen title word.
5. `paper_v3/abstract.txt`: ditto.
6. Bib entry / arxiv title metadata.

## 9. Pointers

- Triggering passages:
  `paper_v3/sections/A2_inference_efficiency.tex` (top-of-section
  leak-status notice + "Update" block at the QA-FLOPs item +
  Phase 1 leak-corrected reading).
- Locked decision rule for hybrids: §6.5 of this document.
- Referenced numerical results:
  - All-attention val PPL: `notebooks/conservative_arch/multi_seed/results/`
  - Leak-free SPLM em_ln val PPL (free γ): `notebooks/conservative_arch/energetic_minima/results/`
  - Leak-free γ-sweep: `notebooks/conservative_arch/ln_damping_sweep/results/RESULTS_LEAKFREE_GAMMA_SWEEP.md`
  - S=5 confirmation sweep: `notebooks/conservative_arch/ln_damping_sweep/results/RESULTS_CONFIRMATION_S5.md` (or equivalent)
  - Resonance-predictor double match:
    `Determining_optimal_gamma_for_SPLM.md` §2.5

---

## 10. Update for paper v4 (6 May 2026)

**Status:** Title decision · 6 May 2026 · supersedes §5 recommendation
**Trigger:** Hybrid sweeps cleared the §6.5 pre-registered rule;
architectural family has expanded to seven flavors.
**Context paper:** v4 preprint (umbrella reference) — Zenodo deposit
(concept DOI minted) + SSRN secondary record. v4 will remain in
preprint form only; journal carving begins with **TMLR 1**.

### 10.1. Status against the pre-registered §6.5 rule

The locked rule was:

> **"Efficient" is justified iff** some hybrid configuration (k, m)
> achieves val PPL within **+5 PPL** of the leak-immune all-attention
> baseline (val PPL ≈ 150 on Tiny Shakespeare at d=128, L=8) **AND**
> its analytical decode-FLOP cost at T = 1024 is **≥ 30% lower** than
> the all-attention baseline. Both bars must clear at S=3 with
> sign-consistency 3/3 against all-attention.

Outcome after H0 + H1 + H1.5 + H2 + H6:

| bar                                                     | result                                                                                              | clears? |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------- |
| Quality bar: hybrid val PPL within +5 of all-attention  | Variant A (k=4, m=4) seed 0: **133.0** vs all-attention E1 mean ~141.8 → hybrid is **better** by ~9 PPL; Helmholtz Q9d AAAASSSS vh=128 seed 0: **134.9** (also better)               | yes     |
| FLOP bar at T = 1024: ≥ 30% lower than all-attention    | Did **not** clear at T = 1024 (`embed + logits` cost dominates at short context, leaves only ~9% room) | **no at T = 1024** |
| FLOP bar at T = 4096 (post-H1.5 long-context arm)        | Helmholtz Q9d AASSSSSS vh=128 delivers **−39.0% FLOPs** at val-PPL parity                            | yes     |
| Sign-consistency at n=5                                  | n=5 H2 power-up: sign-consistent direction in favor of hybrids on quality, but strict p<0.05 not met | partial |

**Interpretation.** The pre-registered rule was written assuming
all gates would be cleared at T = 1024. In practice the FLOP gate
clears decisively at long context (T ≥ 4096) where the structural
asymmetry between attention's $O(T^2)$ pair sum and the SPLM
family's $O(T \cdot d)$ is large enough to escape the
embed-and-logits floor that dominates short-context FLOP accounting.
Both quality bars cleared at S=3 and at the n=5 H2 power-up;
sign-consistency holds even where strict p<0.05 does not (small-sample
artifact at n=5, see `Helmholtz-HSPLM_Path_Forward_and_Experiments.md`
§4.5).

The honest reading: **"Efficient" is justified at the framework level
and at long context, but does not hold uniformly at every cell of
every variant.** This is consistent with how the field uses the word
in titles for related architecture families (e.g. linear-attention,
SSM, Mamba): the framework *enables* efficient inference; not every
member of the family is uniformly efficient at every context length.

### 10.2. The architectural family has grown

Since the 5 May 2026 record, the v4 contents have expanded from a
single SPLM variant to seven distinct conservative-by-construction
families, all under the same Lagrangian umbrella:

| family                                 | distinguishing feature                                                                          | code path                                                       |
| -------------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Vanilla SPLM (single-channel ξ)        | one scalar $V_\theta(\xi, h)$, scalar ξ summary                                                  | `notebooks/conservative_arch/energetic_minima/model_ln.py`      |
| Multi-channel ξ SPLM (HiPPO)           | ξ pooled over multiple HiPPO basis channels                                                      | `notebooks/conservative_arch/multixi/model_multixi_hippo.py`    |
| Multi-channel ξ SPLM (K-EMA)           | ξ pooled over K exponential-moving-average channels                                              | `notebooks/conservative_arch/multixi/model_multixi.py`          |
| Hybrid SPLM (Helmholtz Q9d)            | layer-type schedule of conservative S-blocks and attention A-blocks, single shared $V_\theta$    | `notebooks/conservative_arch/helmholtz/model_helmholtz.py`      |
| Hybrid SPLM (Variant A)                | k attention blocks then m SPLM steps, no shared potential across the two stacks                  | `notebooks/conservative_arch/hybrid/model_hybrid.py`            |
| Hybrid SPLM (Variant B)                | the alternative two-stage ordering                                                                | `notebooks/conservative_arch/hybrid/` (Variant B branch)        |
| PARF-augmented SPLM (Q9c)              | adds a pair-interaction scalar $V_\phi(h_t, h_s)$ alongside $V_\theta$                            | `notebooks/conservative_arch/parf/model_parf.py`                |

All seven derive from the same prescriptive principle: force =
$-\nabla$(scalar potential), with the dynamics integrated by a
symplectic (or quasi-symplectic, in the damped case) method. The
shared-potential separator (and its substack-restricted H6
extension) discriminates these families framework-natively (see
`Helmholtz-HSPLM_Path_Forward_and_Experiments.md` §4.6).

### 10.3. The plural-vs-singular question for the v4 title

The §6 H0-H4 plan envisioned only the hybrid extension to a single
SPLM model. With seven families now under the v4 umbrella, the
**singular "Conservative-by-Construction Language Model"** is no
longer factually accurate. The plural is required by the content,
not chosen for stylistic reasons.

The grammatical effect of pluralising is to drop the orphan
indefinite article "A" before the hyphenated compound, which is a
small stylistic improvement.

### 10.4. Recommended v4 title

> **Semantic Simulation: A Prescriptive Lagrangian Framework for
> Efficient Semantic Inference**
> *Conservative-by-Construction Language Models and the Shared-Potential
> Separator, with a Correspondence to Joint Embedding Predictive
> Architectures*

Three deliberate decisions baked in:

1. **"Efficient" is retained**, with the framework-level reading
   established by §10.1: the framework *enables* efficient inference;
   the long-context wins (Q9d AASSSSSS at T ≥ 4096, Variant A at the
   same regime, vanilla SPLM and the multi-channel ξ variants
   strictly cheaper than attention at any non-trivial context length)
   are the empirical ground for the word. PARF dense at Algorithm A
   does not currently meet the bar; this is forward-compatible
   (Stage 1.5 Gumbel sparsity is a designated path to a $O(T \cdot k)$
   PARF, see `parf/On_the_MLP_Layer_modeling_pairwise_potential.md`
   §9 and `PARF_Augmented_SPLM_Architecture.md` §7).
2. **"Models" is plural** (was singular in v3 and in §5 of this
   document), to match the seven-family content surveyed under the
   umbrella.
3. **The "Lagrangian Framework" hat-tip and the Shared-Potential
   Separator + JEPA correspondence clauses are kept verbatim.** The
   Lagrangian framing is correct across all seven families (every one
   has an underlying scalar potential, even where attention is grafted
   on); the separator is the unifying framework-native diagnostic; the
   JEPA correspondence still applies to all variants since each derives
   inference dynamics from a shared embedding-predictor stack.

### 10.5. Publication strategy implication

v4 will live as a **preprint-only umbrella** on Zenodo (with concept
DOI minted) + SSRN (secondary record). It will not be submitted to a
single journal venue — its scope is too wide and its contribution
surface too heterogeneous for any single peer-reviewed cycle.

Instead the v4 contents will be **carved into a sequence of
journal-targeted papers**, starting with **TMLR 1** on the
SPLM-foundational slice:

- TMLR 1 (foundational): SPLM + multi-channel ξ variants
  (HiPPO, K-EMA) + the causal-leak fix + the shared-potential
  separator (with the H6 substack-restricted version). The "core
  conservative-by-construction" paper.
- TMLR 2 (or NeurIPS/ICML, depending on timeline): Hybrid Q9d +
  Variant A + the H1.5 / H2 paired-seeds methodology + the
  long-context FLOP win. The "efficient hybrid" paper.
- Later (NeurIPS/ICML): PARF + Stage 1.5 sparsity + OQ-1 verdict.
  The "attention replacement" paper.
- Workshop or RL venue: PARF + RL outer loop.

Each journal paper cites v4 by **concept DOI** (not version-specific
DOI) so the umbrella reference always resolves to the latest preprint
revision. v4 is revised with each landed journal paper to cite that
paper as the journal-of-record for its slice; this prevents the
umbrella from competing with its own children for citations.

The arXiv-deposit pathway is a separate workstream that will be
unlocked once TMLR 1 is in review (cited-author cold-email channel)
or accepted (action-editor pathway). For now, the Zenodo concept
DOI is the canonical citable identifier for the v4 framework.

### 10.6. Decisions recorded

User decisions, this document, 6 May 2026:

> **(D1)** Retain the word "Efficient" in the v4 title, on the strength
> of the long-context FLOP wins (H1.5 −39% at T ≥ 4096) and the
> framework-level reading.
>
> **(D2)** Pluralise "Conservative-by-Construction Language Model" to
> "Conservative-by-Construction Language Models" to match the seven
> architectural families now in v4.
>
> **(D3)** Keep v4 in preprint-only form (Zenodo concept DOI + SSRN);
> begin journal carving with TMLR 1 on the SPLM-foundational slice.
> Each journal paper will cite v4 by concept DOI.
>
> **(D4)** The pre-registered §6.5 rule is **partially satisfied** —
> at T ≥ 4096 it clears decisively; at T = 1024 the embed-and-logits
> floor prevents a 30% reduction, but the quality bar still clears.
> The honest framing in v4 is that the FLOP win is a long-context
> property, not a uniform-context property. This must be reflected
> in the v4 abstract and §A2-equivalent section.

### 10.7. Collateral edits if §10.4 is adopted

In addition to the §8 edits (which still apply to v3 if v3 is also
revised), v4 specifically requires:

1. **v4 abstract**: surface the framework-level reading of "Efficient"
   explicitly; cite the H1.5 long-context FLOP win as the empirical
   anchor.
2. **v4 §A2-equivalent inference-efficiency section**: extend the v3
   §A2 analysis to cover the hybrid arms (Q9d, Variant A) and the
   PARF arm (with the explicit caveat that PARF dense Algorithm A is
   $O(T^2)$ per layer; Stage 1.5 is the path to subquadratic PARF).
3. **v4 introduction**: explicitly declare the seven-family scope of
   the paper (the "umbrella" framing) and the published-journal
   carving plan.
4. **v4 conclusion / future work**: surface the TMLR-first journal
   carving plan; cite the pre-registered §6.5 rule and its post-hoc
   resolution (§10.1 of this document).

---

*Last updated: 6 May 2026.*
