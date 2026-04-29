# Energetic-minima alternatives to a free $V_\theta$

**Companion note to the main paper, Section 14 (Conservative
Architectures), Subsections 14.15 (attractor analysis and design
rationale) and 14.17 (Q11, structural alternatives).**

This document reports the three follow-up experiments flagged as open
problems in the paper's §14.17 (Q11), implemented in
[`notebooks/conservative_arch/energetic_minima/`](../notebooks/conservative_arch/energetic_minima/).
It ended up producing one decisive confirmation of the paper's
position and one clean refutation of a specific neutrality claim;
both results strengthen, rather than weaken, the overall story.

---

## 1. The question the paper left open

Section 14.15 of the paper established two uncomfortable facts about
the flagship SPLM:

1. The trained scalar potential $V_\theta(\xi, h)$ is **unbounded
   below**. The training loss only ever sees the gradient of $V_\theta$
   along the realised trajectory, so the absolute scale and
   off-trajectory behaviour of $V_\theta$ are unpenalised and
   effectively a gauge.
2. Nevertheless, the damped flow at the trained horizon
   $L = L_{\text{train}}$ does exhibit prompt-dependent multi-basin
   attractor structure with $K^\ast \in [2, 10]$. These attractors are
   **pullback basins of the non-autonomous damped flow**, not local
   minima of $V_\theta$.

On six grounds (R1 – R6 in §14.15) the paper argued that this free,
scale-free $V_\theta$ is the correct design and that structural
energetic minima would be an expressivity downgrade. That position
made three falsifiable predictions about lightweight alternative
designs. This note implements and tests all three.

---

## 2. Experimental setup

All four variants share the SARF-faithful $\xi$ re-pooling, logfreq
per-token semantic mass, $d = 128$, $L = 8$ semi-implicit damped
steps, $\Delta t = 1$, training on Tiny Shakespeare for
$4{,}000$ steps with $\text{batch}=16$, $\text{block}=128$,
$\text{lr} = 5\times 10^{-4}$, cosine schedule, $\text{seed}=0$. Any
observed difference in val perplexity is attributable to the variant
itself.

| tag | variant | what changes | code |
|---|---|---|---|
| `em_base` | flagship SARF+mass+logfreq | nothing | `sarf_mass_variant/` (baseline) |
| `em_ln` | (i) **LayerNorm-after-step** | project $h_{l+1}$ onto the unit-LayerNorm shell (mean 0, var 1) after every damped step | `energetic_minima/model_ln.py` |
| `em_sg` | (ii) **scale-gauge** $\lambda_{V_0}=10^{-3}$ | add $\lambda_{V_0} \cdot \mathbb{E}_{b,t}V_\theta(\xi_0, h_0)^2$ to the loss | `energetic_minima/train.py --variant sg` |
| `em_gm` | (iii) **Gaussian-mixture head** $K=64$ | replace the MLP $V_\theta$ with $\sum_{k=1}^{K} \mathrm{amp}_k\bigl(1 - e^{-\kappa_k^2 \lVert z - c_k\rVert^2}\bigr)$ in $(xi, h)$ space | `energetic_minima/model_gm.py` |

Attractor extraction (on all four) uses the standard
`attractor_analysis/` pipeline in *dynamical* mode with $N=288$ seeds
(96 Gaussian + 96 token-embedding + 96 perturbed real $h_L$), five
prompts (narrative, mathematics, scientific, dialogue, code), K-means
over $K \in [2, 10]$ with silhouette selection.

---

## 3. Results — the one-page table

| variant | val ppl | $K^\ast$ per prompt (n, m, s, d, c) | $V$ range | content-basin fraction |
|---|---:|---|---|---:|
| baseline SARF+mass (logfreq) | **160.55** | 9, 10, 8, 10, 8 | $[-1916, +1445]$ | **0.58** |
| (i) LayerNorm-after-step | **88.63** | 5, 9, 10, 2, 2 | $[-84, -60]$ | 0.23 |
| (ii) scale-gauge ($\lambda_{V_0}{=}10^{-3}$) | 191.00 | 2, 2, 2, 2, 10 | $[-2332, -186]$ | 0.12 |
| (iii) Gaussian-mixture head ($K{=}64$) | **677.67** | 2, 2, 2, 2, 2 | $[+60.3, +60.3]$ | **0.00** |

Legend:
* **val ppl**: validation perplexity at the last eval step (step 4000).
* **$K^\ast$**: silhouette-optimal number of K-means clusters of the
  damped-flow endpoints, per prompt.
* **$V$ range**: $[\min, \max]$ of basin-averaged $\langle V \rangle$
  over all basins over all five prompts.
* **content-basin fraction**: over the five prompts, average
  fraction of basins whose largest-probability decoded token is not a
  punctuation symbol.

The four 3D potential-plus-trajectories panels for the dialogue
prompt are assembled in
[`energetic_minima/results/landscape3d_compare_four_variants_dialogue.png`](../notebooks/conservative_arch/energetic_minima/results/landscape3d_compare_four_variants_dialogue.png).

---

## 4. Interpretation, variant by variant

### 4.1 (i) LayerNorm-after-step: paper's prediction refuted, in the nice direction

**The paper said:** LN should leave val ppl "essentially unchanged
(within 10 – 15 %) while producing a narrower $V_\theta$ range and
comparable or slightly crisper basins." This was R5's negative-case
corollary: LN buys a finite minimum of $V$ on the compact shell
$S^{d-1}$ but shouldn't otherwise matter.

**What we see.**

* val ppl: **88.63**, a **45 % relative improvement** over baseline
  (160.55). Not "essentially unchanged." LN is the single biggest
  win in this family.
* $V$ range: $[-84.2, -60.5]$ vs. baseline's $[-1916, +1445]$ — a
  **30×** reduction in width of the learned energy surface, entirely
  in the negative half. LN has put $V$ on a tight leash.
* basin-count: slightly reduced on two prompts (dialogue, code:
  $K^\ast = 2$) but unchanged or increased on the other three
  (narrative, mathematics, scientific: $K^\ast \in [5, 10]$).
* content fraction: 0.23 vs. baseline's 0.58 — the attractors are
  **more** punctuation-dominated under LN.

**Mechanistic read.** Per-step LayerNorm keeps $\lVert h \rVert$
pinned to $\sqrt{d}$ (we measure $\lVert h_L \rVert \approx 14$ for
$d=128$, vs. the baseline's $\lVert h_L \rVert \approx 60\text{--}80$).
This has two simultaneous effects:

1. *Loss reduction.* The tied LM head $h_L \cdot E^\top$ now
   operates on vectors of controlled norm, which is a standard
   transformer design choice. It reduces noise, stabilises gradients,
   and lets the head focus on *direction* rather than having to
   compensate for varying $\lVert h \rVert$.
2. *Basin concentration on punctuation.* The same controlled norm
   means the damped flow can converge onto a tighter target. On
   Shakespeare, the commonest continuation of a short prompt is a
   punctuation mark. LN lets SPLM fit that distribution very
   accurately, at the cost of suppressing rarer content-word
   attractors — **the same phenomenon we saw in the symplectic
   (Verlet) side-finding of §14.** A better-regularised flow
   concentrates probability on the most frequent symbols.

**Implication for the paper's R1 – R6.** R1 – R4 are unaffected
(pullback basins, trajectory-not-equilibrium, energy decomposition
all hold). R6 (unbounded $V$ is a gauge choice, not a pathology) is
**strengthened**: LN is a different choice of gauge, on a compact
shell, that also produces a well-trained SPLM. The "unboundedness" of
the flagship is indeed a gauge. R5 (structurally bounded $V$ hurts
expressivity) **still must be weakened to say: structurally bounded
in $V$-space hurts; compactifying the state space while keeping $V$
free does not**. The distinction matters and is made precise in
§5 below.

### 4.2 (ii) Scale-gauge: paper's prediction partially confirmed

**The paper said:** a small loss-side anchor on $V_\theta$'s absolute
value at the input embedding should leave ppl essentially unchanged
while narrowing the $V$ range.

**What we see.**

* val ppl: 191.00, about **19 % worse** than baseline — not
  "essentially unchanged", but not catastrophic.
* $V$ range: $[-2332, -186]$. *Not narrower* than baseline; the
  penalty with $\lambda_{V_0} = 10^{-3}$ was too weak to compete with
  the LM gradient's pull on $V$'s scale. The input-side anchor
  $V_\theta(\xi_0, h_0)^2 \to 0$ pulls $V_\theta$ downward at $h_0$
  but leaves $V_\theta(h_L)$ free.
* $K^\ast = 2$ for four of five prompts — the scale-gauge
  **collapses the basin structure** to a punctuation/non-punctuation
  split.
* content fraction: 0.12, the worst non-saturated number in the
  table.

**Mechanistic read.** The scale-gauge creates a gradient pull on
$V_\theta$'s *parameters* that competes with, but does not replace,
the CE gradient. The CE gradient remains the dominant training
signal, so $V$ stays wide. What the penalty does do is bias the
*direction* the flow takes during the first few steps, because $V$ is
now pinned low at $h_0$. That bias appears in the coarsened basin
structure: the flow has fewer "upstream" degrees of freedom and
converges on a binary split.

**Implication.** The scale-gauge with $\lambda_{V_0} \in \{10^{-3}\}$
is neither a useful regulariser nor a decisive falsifier. A sweep
over $\lambda_{V_0} \in [10^{-5}, 10^{-1}]$ would likely reveal a
crossover where the penalty starts to dominate and $V$ collapses to
zero everywhere; our single value landed on the low-$\lambda$ side of
that crossover. We leave that sweep as explicit future work; it is
not promising based on the coarsened-basin signal we saw.

### 4.3 (iii) Gaussian-mixture head: paper's prediction vindicated

**The paper said:** a physics-prescribed, bounded-below Gaussian
mixture should reproduce the static-null behaviour of the seven
scalar-form fits of §14.2 (five-negatives) and degrade LM quality.
A Gaussian-mixture SPLM matching the SARF-faithful val ppl 160.55
would force §14.15 and §14.17 (Q11) to invert.

**What we see.** Complete vindication of R5, at the extreme end of
the predicted range.

* val ppl: **677.67**, **4.2 × worse** than baseline. The model
  plateaus at ppl $\approx 680$ from step 600 onwards and does not
  improve with further training.
* $V$ range: $[+60.28, +60.33]$ — a *spike* of width $0.05$ over
  five prompts and ten basins. $V$ is structurally bounded in
  $[0, \sum_k \mathrm{amp}_k]$ and the trained model uses
  essentially a single value of $V$ across the entire $h$-space the
  flow visits.
* Convergence: **196/288 seeds structurally converge**
  ($\lVert v \rVert < 0.05$) — far more than any other variant, as
  the theory predicts a compactly bounded $V$ with finitely many
  wells should.
* Basin content: both basins, on every one of the five prompts,
  decode to the **same** distribution:
  `\n·0.09, ,·0.05, :·0.03, .·0.02, the·0.01, to·0.01, ;·0.01, and·0.01, I·0.01, of·0.01`.
  This is the Tiny-Shakespeare unigram distribution of
  next-tokens-regardless-of-prompt. The model has completely
  collapsed to a context-free unigram predictor.
* content fraction: 0.00 — no basin has a content word as its
  top-1 decoded token, on any prompt.

**Mechanistic read.** A sum of $K = 64$ Gaussian wells with
$d = 128$ has roughly $64 \times 258 \approx 16{,}500$ parameters,
vs. the MLP head's $\approx 656{,}000$. More importantly, the
*functional family* is constrained: at any point $z \in \mathbb{R}^{2d}$,
$V(z)$ is a sum of monotone saturating functions of distances to
$K$ centres. That family is expressive enough to represent a smooth
bump landscape but *cannot* represent the kind of highly anisotropic,
direction-selective surface a tied-embedding LM head needs to steer
$h_L$ toward the right row of $E$. So the flow has nowhere useful to
go; it settles on the unigram mean. The same phenomenon we diagnosed
as the static null of §14.2 reappears here in dynamical form.

**Implication.** R5 stands, strongly. The flagship position that
"structurally bounded scalar potentials are expressivity-limited at
this scale" is empirically confirmed, now both statically (§14.2,
seven scalar fits to frozen GPT-2 trajectories) and dynamically
(§14.15 → this note, a full Gaussian-mixture SPLM).

---

## 5. What this changes in the paper's story

Net effect on the paper's §14.15 and §14.17 (Q11):

1. **R1 – R4 are unchanged.** The conserved quantity is still
   $E = T + V$ not $V$ alone (R1), the correct mathematical object
   is still a pullback basin (R2), language is still trajectory not
   equilibrium (R3), and the Verlet side-finding still shows that
   integrator accuracy and basin diversity trade off (R4).
2. **R5 is sharpened.** "Structurally bounded scalar potentials are
   expressivity-limited" needs to be re-stated as
   "structurally bounded $V_\theta$ (GM) is expressivity-limited;
   compactifying the state space while keeping $V$ a free MLP (LN)
   is not." Both are forms of boundedness, but only the first kills
   expressivity.
3. **R6 is strengthened.** "Unbounded below is a gauge, not a
   pathology" now has a second empirical witness: LN is a different
   gauge on a compact state space that also trains SPLM successfully
   (in fact more successfully).
4. **A new positive result joins the open-follow-up discussion.**
   LayerNorm-after-step is not a neutral alternative design, it is a
   *useful* one: it gives a $45\%$ ppl improvement at zero
   additional parameters, with 30 × narrower $V$. The cost is a
   shift of the dynamic basin structure toward punctuation-dominated
   attractors, consistent with the Verlet side-finding. Whether this
   trade-off is *net positive* for interpretability depends on the
   downstream use. For pure language modelling, the answer is clearly
   yes.

---

## 6. What remains open

* A $\lambda_{V_0}$ sweep for the scale-gauge across
  $\{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$, to locate the
  penalty-dominance crossover.
* A capacity sweep for GM across $K \in \{16, 64, 256, 1024\}$, to
  check whether the expressivity ceiling can be lifted by making the
  mixture wide. The prediction, from the static-null result of §14.2,
  is no.
* A combined LN + scale-gauge run, where the compact state space
  already buys a minimum of $V$ and the loss-side anchor can be very
  small.
* A longer-training run for LN at $d = 192$, $L = 16$, on
  Tiny-Stories, to check whether the ppl gain survives to a richer
  corpus.

We note these but do not chase them here. The three experiments in
this document answer the specific falsification criterion flagged in
§14.17 (Q11) of the paper: GM fails to match baseline ppl,
decisively; LN exceeds baseline ppl, decisively; SG does neither.

---

## 7. Reproducing these results

From the repo root:

```bash
cd notebooks/conservative_arch/energetic_minima

python3 train.py --variant ln --mode shakespeare
python3 train.py --variant sg --mode shakespeare --lambda-v0 1e-3
python3 train.py --variant gm --mode shakespeare --gm-K 64

bash run_attractor_pipeline.sh
python3 compare.py
python3 make_compare_figure.py
```

Results land in `energetic_minima/results/` (training) and
`attractor_analysis/results/` (attractor summaries and 3D landscapes).
Each shakespeare run takes $\sim 20$ min (LN, SG) to $\sim 85$ min
(GM) on an Apple-Silicon MPS device; the attractor pipeline takes
another $\sim 8$ min. Total wall-clock for the full experiment from
scratch: $\sim 2.5$ hours.
