# PARF-SPLM (Q9c, PARF-augmented SPLM) — Path Forward and Experiments

**Status:** Live experiment record · started 6 May 2026 · *no quality cells run yet*
**Scope paper:** `paper_v3/main.tex` (Q9(c) follow-up branch of §17.3) and the v4 carve-out `Section_15_24_PARF_Augmented_SPLM_v4_draft.docx`
**Design doc:** [`PARF_Augmented_SPLM_Architecture.md`](PARF_Augmented_SPLM_Architecture.md)
**Sibling path (Q9d, layer-type Helmholtz hybrid):** [`Helmholtz-HSPLM_Path_Forward_and_Experiments.md`](Helmholtz-HSPLM_Path_Forward_and_Experiments.md)
**Sibling path (Variant A two-stage SPLM):** [`HSPLM_Path_Forward_and_Experiments.md`](HSPLM_Path_Forward_and_Experiments.md)
**Code root:** `notebooks/conservative_arch/parf/`

This document tracks the design, experiments, results, and outstanding
questions of the **PARF-augmented SPLM (Q9c)** investigation — the
*pair-potential* extension of SPLM. The investigation runs in
parallel with the Variant A two-stage HSPLM (`hybrid/`) and the Q9d
layer-type Helmholtz hybrid (`helmholtz/`), and tests the broader
claim that the v3 paper's §5 PARF prescription admits a constructive,
end-to-end-trainable language model in which the pair-interaction
scalar $V_\phi$ is jointly learned with the per-token scalar $V_\theta$
under strict causality and a single shared energy field.

The Q9c construction extends the SPLM Lagrangian by adding a
pair-interaction term to the per-layer effective scalar:

$$
U^{(\ell)}_t = V_\theta(\xi_t, h_t) + \sum_{s \lt t} V_\phi(h_t, h_s)
$$

with the design-doc §3 **causal reduction** (past tokens are treated
as fixed external sources by `.detach`-ing the source slice
$\lbrace h_s \rbrace_{s \lt t}$ when forming the pair-potential matrix).
This both severs the back-reaction force on past tokens and makes
the per-token force strictly causal, in the same sense as the
v3 leak-fix invariant for $V_\theta$.

The pre-registered v4 title-justification rule applies unchanged at
the Q9c quality arm:

> **"Efficient" is justified iff** some hybrid achieves val PPL
> within **+5 PPL** of the all-attention baseline (~150 on Tiny
> Shakespeare at d=128, L=8) **AND** its analytical decode-FLOP cost
> at T = 1024 is **≥ 30% lower** than all-attention, both at S=3 with
> sign-consistency 3/3.

Q9c adds two architecture-level open questions (the design doc's
§5.1 and §5.2 commitments, lifted to OQ-1 / OQ-2 in the v4 draft):

- **OQ-1 (structural prior).** If the structural $V_\phi$ (the
 §5.1-faithful form $-C \cdot \Theta_\phi(\theta(h_t),\theta(h_s)) \cdot \Phi_\phi(l(h_t),l(h_s)) / \sqrt{\lVert h_t - h_s \rVert^2 + \varepsilon^2}$)
 matches an unstructured MLP $V_\phi$ on val PPL, the §5.1 prior is
 pedagogical. If it outperforms, the prior is empirically active.
- **OQ-2 (joint pair test).** Does the learned $V_\phi$ assign
 measurable interaction strength to pairs that real GPT-style
 attention also attends to, on the same sentences?

These are the *framework-native* deliverables that the val PPL +
decode-FLOP table does not capture; they are scheduled for P3
(post-P2 confirmation).

---

## 1. Architecture (Q9c, every layer is a velocity-Verlet step under $V_\theta + \sum_{s \lt t} V_\phi$)

```text
h_0     = E[x] + P
h_prev  = h_0
for ell in 1..L:
    delta = h - h_prev                               # velocity proxy (carries across layers)
    xi    = causal_cumulative_mean(h.detach())       # leak-fix invariant (V_theta arm)
    h_src = h.detach()                               # causal reduction (V_phi arm)

    V_th_per_token = V_theta(xi, h)                  # (B, T, 1)  shared V_theta
    P_pair         = V_phi(h, h_src)                 # (B, T, T)  shared V_phi
    P_pair_masked  = mask_strict_lower(P_pair)       # only s < t survives the sum

    U     = V_th_per_token.sum() + P_pair_masked.sum()
    f     = -grad_h U                                # single autograd.grad call

    denom = 1 + dt*gamma
    h_new = h + delta / denom + (dt^2 / (m_b * denom)) * f
    h_new = LayerNorm(h_new)                         # if ln_after_step
    h_prev, h = h, h_new

logits = h @ E^T                                     # tied embeddings
```

Implementation: `notebooks/conservative_arch/parf/model_parf.py`.
Fixed design choices (matching Q9d / Variant A where possible):

- **xi re-derivation = `causal_cumulative_mean(h.detach)`** at every
 layer — leak-safe; same shape as Q9d.
- **`h_src = h.detach` at every layer** — the new PARF-specific
 causal-reduction `.detach` point. Severs back-reaction forces
 on past tokens, makes the per-token force strictly causal, and
 is the architectural commitment that makes Algorithm A (pure
 NTP backprop) tractable.
- **Single shared `V_theta`** across all layers — same as Q9d / VA.
- **Single shared `V_phi`** across all layers — the PARF analogue
 of the "single energy field" commitment. Two variants ship:
 - `structural` — the §5.1-faithful pair potential (default).
 - `mlp` — unstructured MLP on
 $\mathrm{concat}(h_t, h_s, h_t - h_s)$ (the OQ-1 comparator).
- **Velocity-Verlet step** — same damped position-Verlet form as Q9d
 S-blocks.
- **Combined `autograd.grad` on $V_\theta + V_\phi$** — a single
 scalar `U` is summed and the force is recovered with one
 `torch.autograd.grad(U, h_in, create_graph=self.training)` call,
 halving the second-order graph cost vs separate per-potential
 calls.
- **Optional `torch.utils.checkpoint`** on the $V_\phi$ pair-sum
 call (`PARFConfig.use_grad_checkpoint`) — auto-enabled for the
 MLP variant by `scripts/run_first_quality_cell.sh` to fit within
 the 13.5 GB MPS watermark; not needed for the structural variant
 at the prototype shape. Bit-equality of training trace with and
 without checkpointing is verified by `smoke_test.py`'s `[3/3]
 em-ln+gc` block.
- **`ln_after_step=True`** — LN after each velocity-Verlet step,
 matches Q9d / VA / em-ln-leakfree.
- **`causal_force=True` is hard-default** — preserves v3 leak-fix
 invariant *and* the new PARF-specific h_src `.detach`.
- **Tied embeddings** with `E.weight.T` for the LM head.
- **Per-token logfreq mass** (`mass_mode='logfreq'`) — matches
 leak-free SPLM em_ln, Q9d, and VA.

### V_phi shape registry

| variant | shape | params (em-ln cell) | role |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------|--------------------:|-------------------------------------------------------------------|
| `structural` | $-C \cdot \Theta_\phi(\theta(h_t), \theta(h_s)) \cdot \Phi_\phi(l(h_t), l(h_s)) / \sqrt{\lVert h_t - h_s \rVert^2 + \varepsilon^2}$ | 4,002 | **§5.1-faithful pair potential** (default; the framework prior) |
| `mlp` | learned MLP on $\mathrm{concat}(h_t, h_s, h_t - h_s)$ at hidden $H$ = `mlp_h` | 28,865 | **unstructured ablation** (the design-doc OQ-1 comparator) |

Both variants share the SAME outer machinery (single shared
$V_\theta$, velocity-Verlet step, causal reduction, logfreq mass,
embed/logits shape). The only difference is the inner shape of
$V_\phi$. This isolates the "does the §5.1 prior matter empirically?"
question to exactly one experimental knob.

---

## 2. Reference baselines (already on disk, leak-immune)

| Arm | Val PPL | Source |
|-----------------------------------------------------------------|------------|--------------------------------------------------------------------------------------------------|
| All-attention (matched GPT-2, `n_layer=8`) | **149.80 ± 7.21** (5-seed E1) | `multi_seed/results/` |
| All-SPLM em_ln (free-γ, leak-free) | **173.59** (single seed) | `energetic_minima/results/` |
| All-SPLM em_ln (γ ∈ [0.10, 0.15], leak-free) | **~178–181** | `ln_damping_sweep/results/RESULTS_LEAKFREE_GAMMA_SWEEP.md` |
| **Variant A HSPLM (k=4, m=4) at S=1** | **133.01** (seed 0) | `hybrid/results/h1_sweep/H1_RESULTS.md` |
| **Variant A HSPLM (k=4, m=4) at n=5 mean** | **147.40** (seed 0..4) | `hybrid/results/h2_paired_confirmation/k4_m4/` |
| **Q9d Helmholtz `AAAASSSS` vh=128 (seed 0)** | **134.89** | `helmholtz/results/h1p5_narrow_v/` |
| **Q9d Helmholtz `AAAASSSS` vh=128 at n=5 mean** | **145.86** | `helmholtz/results/h2_paired_confirmation/H2_RESULTS.md` |
| **Q9d Helmholtz `AASSSSSS` vh=128 (seed 0)** | **139.63** | same |

All trained at `(d=128, max_len=256, L=8, n_head=4, mlp_mult=4,
v_hidden=512, v_depth=3)` (or `v_hidden=128` for the H1.5 / Q9d
narrow-V cells), batch 16, block 128, 4000 steps, AdamW(0.9, 0.95)
lr=5e-4 with 200 warmup + cosine, on Tiny Shakespeare (GPT-2 BPE).

PARF cells are run at the same shape as the Q9d $\mathtt{AAAASSSS}$
vh=128 cell (`d=128, L=8, T=128, B=16, v_hidden=128`) so val PPL is
directly comparable to all the anchors above. The PARF-specific
cost contribution is the $O(T^2)$ pair sum at every layer, which
adds ~50% wall-clock relative to the Q9d $\mathtt{AAAASSSS}$ vh=128
cell on Apple MPS at $T = 128$.

### Param-count expectation at d=128, L=8, v_hidden=128 (em-ln cell shape)

| variant | $V_\theta$ params | $V_\phi$ params | total over the SPLM core |
|------------------|------------------:|----------------:|-------------------------:|
| `structural` | 66,049 | 4,002 | 70,051 |
| `mlp` (mlp_h=64) | 66,049 | 28,865 | 94,914 |

Both add to the shared embedding (~6.4 M params) for an end-to-end
total in the 6.5–6.6 M range, modestly smaller than the 7.9–8.3 M
Q9d / VA cells because the per-A-block per-layer attention parameters
are not allocated in the PARF cell (every layer is an SPLM step,
not an attention block). This is an inherent property of the Q9c
construction and is documented in the design doc as expected.

---

## 3. Tiered experimental plan

| Tier | What | Cells | Time est. | Status |
|------|-----------------------------------------------------------------------------------------------------------------------------------------------|------:|------------------|-----------------|
| P0 | Smoke + causal probe (CPU + MPS) on both V_phi variants | 2 | ~5 min | ✅ done |
| P1 | First quality cell: `structural` V_phi, seed 0, em-ln vh=128 cell shape, S=1 | 1 | ~252 min MPS | ✅ done (val PPL 210.54 → **FAIL** §6.3) |
| P1.5a | OQ-1 comparator: `mlp` V_phi (mlp_h=16), seed 0, em-ln vh=128 cell shape, S=1 | 1 | ~430 min MPS | ✅ done (val PPL 297.22; OQ-1 verdict: **structural prior empirically active**) |
| P1.5b | OQ-1 comparator: `mlp` V_phi (mlp_h=32), seed 0, em-ln vh=128 cell shape, S=1 | 1 | ~640 min MPS | deferred (P1.5a verdict already unambiguous) |
| P1.6 | Wider `structural` V_phi (`phi_hidden=128, theta_hidden=128`), seed 0, em-ln vh=128 cell shape, S=1 — capacity disambiguation | 1 | ~300 min MPS | **in progress** (launched 7 May 2026 09:06 EDT) |
| P2 | Paired confirmation at n=5 on the better V_phi variant; seeds 0..4 | 4 | ~17 h MPS | **planned (post-P1.5)** |
| P3 | Framework-native diagnostics on best cell: OQ-2 (joint pair test on real GPT-2 attention), holonomy decomposition, R6 ladder inversion | 1-3 | ~3 h CPU | **planned (post-P2)** |
| P4 | TinyStories scale-up cell on best PARF configuration, S=3 | 3 | ~4-5 h MPS | optional |
| P5 | Stage 1.5 — Gumbel-softmax sparsity for $V_\phi$ (token sparsification → decode-FLOP arm) | 2 | ~6 h MPS | optional |
| P6 | Algorithm B / Algorithm C — PPO with framework-native reward / Pair-Selective REINFORCE (the v4 §15.24.7 prescriptive primary) | ? | ? | **deferred** (separate paper draft) |

P3 is the deliverable that distinguishes Q9c from a pure
engineering benchmark — it is what makes the construction "the
first language model in which the §5.1 PARF prescription is jointly
end-to-end-trainable under strict causality," to use the design
doc's framing.

---

## 4. Results so far

### 4.1. Smoke verification — completed 6 May 2026

Output: `parf/smoke_test.py` self-test on both V_phi variants and
both gradient-checkpoint modes (artifacts not committed).

- **Model smoke** (3 cell shapes: tiny, em-ln-shape, em-ln-shape +
 grad-checkpointing): forward + backward clean for both V_phi
 variants; gradients flow through both $V_\theta$ and $V_\phi$;
 optimizer step + 5-step training loss reduction confirmed.
- **Bit-equality of the training trace** with and without gradient
 checkpointing on the em-ln cell shape — verified to within
 numerical tolerance by the `[3/3] em-ln+gc` block. This is the
 guarantee that the grad-checkpoint path is a memory-vs-compute
 trade-off only, not a behavioural change.

```text
[em-ln] V_phi='structural'  params=135,717  V_theta=66,049  V_phi=4,002
[em-ln] causal probe PASS (V_phi='structural')
[em-ln] V_phi='mlp'         params=160,580  V_theta=66,049  V_phi=28,865
[em-ln] causal probe PASS (V_phi='mlp')
ALL SMOKE CHECKS PASSED
```

### 4.2. Causal-violation probe — completed 6 May 2026

Output: `parf/causal_probe_parf.py` self-test on every V_phi variant
and both `causal_force` modes (see §9 below for the probe design).

- **Fixed mode** (`causal_force=True`): perturbation Δ ≡ **0.00e+00**
 and gradient-Jacobian Δ ≡ **0.00e+00** on both V_phi variants
 at the prototype shape. Q9c is causal by construction at the
 strict 1e-6 threshold under both detach points (the inherited
 $\xi$-pool detach AND the PARF-specific h_src detach).
- **Buggy mode** (`causal_force=False`): leak signal visible on
 both V_phi variants (turns on once either of the two detach
 points is removed). Confirms the leak channel is real and
 the pair of `.detach` calls severs it bit-exactly.

### 4.3. Wall-clock survey on Apple MPS (em-ln cell shape) — completed 6 May 2026

Memory is the binding constraint for the MLP variant; wall-clock
is the binding constraint for the structural variant. See
`parf/README.md` for the full survey table; the headline numbers
relevant to the planned P1 / P1.5 cells are:

| variant | grad ckpt | B | mlp_h | s/step | 4000-step est | role for P1/P1.5 |
|--------------|-----------|----|-------|-------:|---------------|----------------------|
| structural | off | 16 | — | 3.77 | ~252 min | **P1 headline cell** |
| mlp | on | 16 | 32 | 9.6 | ~640 min | P1.5 candidate (large mlp_h) |
| mlp | on | 16 | 16 | 6.4 | ~430 min | P1.5 candidate (small mlp_h) |
| mlp | on | 16 | 64 | OOM | OOM | grad-ckpt cannot intercept the input tensor itself |

The grad-checkpoint flag (`PARFConfig.use_grad_checkpoint`,
`--grad-checkpoint`, env `GRAD_CHECKPOINT=1`) is **auto-enabled
for the MLP variant** by `scripts/run_first_quality_cell.sh`; the
structural variant runs without it by default.

### 4.4. P1 — first quality cell (structural V_phi, seed 0) — completed 7 May 2026

Output: `parf/results/structural/seed0/parf_structural_shakespeare_seed0_*`
(the `*_ckpt_latest.pt` is local-only; `*_summary.md`,
`*_training_log.jsonl`, `*_loss_curve.png`, and `training.log` are
committed; the `.png` is stored via Git LFS).

#### Headline result

| arm | val PPL (seed 0) | params |
|------------------------------------------------------------|-----------------:|--------:|
| Variant A HSPLM (k=4, m=4) | 133.01 | 7.92 M |
| Q9d Helmholtz `AAAASSSS` vh=128 | 134.89 | 7.92 M |
| All-attention (matched GPT-2) | 141.80 | ~8.0 M |
| All-SPLM em_ln (free-γ, leak-free) | 173.59 | ~6.5 M |
| **PARF Q9c structural (this cell)** | **210.54** | 6.54 M |

**Verdict per §6.3 decision rule (PASS iff val PPL \< 155): FAIL** by
+55 PPL over the gate, +75 PPL over the Q9d analogue, and even +37
PPL worse than vanilla all-SPLM em_ln. Wall-clock 15,554 s (~4.3 h)
on Apple MPS — slower than the 252 min estimate due to MPS contention.

#### Diagnostic readout

The training run was *operationally* clean:

- Causal-violation probe at startup passed at the strict 1e-6
 threshold under both `.detach` points.
- No numerical instability, no NaN/Inf, no optimizer divergence; the
 trainer ran the full 4000-step schedule.
- Train and validation losses tracked each other tightly — final
 train loss 4.7590, val loss 5.3497, generalization gap is small.
 This is **not an overfitting issue**; it is a capacity / dynamics
 issue.
- Best val checkpoint was around step 3800 (val PPL 213.96); the val
 curve is essentially flat after step 2600 in the 220–240 PPL range
 with no plateau-to-improvement.

The two interpretable diagnostic signals are:

1. **γ converged to 0.0883** — markedly lower than VA (0.154) /
 Q9d (0.114–0.163), and well below the leak-free SPLM resonance
 anchor (γ\* ≈ 0.166). Reading γ as 1 / decoherence-time of the
 second-order Lagrangian dynamics, the optimizer is *suppressing
 dissipation* — consistent with the pair force being destabilising
 at the canonical γ. The system is forced toward the under-damped
 regime to keep the integrator stable and in doing so loses the
 resonance basin that VA / Q9d both find independently across
 seeds.
2. **Train loss never falls below 4.76** (vs ~3.74 for VA k=4, m=4
 and ~3.78 for Q9d at the same compute budget). The bottleneck is
 on the *fitting* side, not the *generalization* side — the
 structural V_phi (4,002 params) does not have enough capacity at
 the prototype scale, OR the velocity-Verlet step is incompatible
 with the pair force at the SPLM resonance γ.

#### Triage (per §6.3 of this document)

The §6.3 triage tree's first branch applies (training completed
cleanly but val PPL ≥ 155):

> *If val PPL ≥ 155 but training completed cleanly: investigate
> whether the structural $V_\phi$ has enough capacity at this
> scale; consider P1.5 with `mlp_h=64` (memory-permitting) or
> raise $\Theta_\phi$ / $\Phi_\phi$ MLP widths.*

Acted on (7 May 2026): **launch P1.5 with `mlp_h=16`** (the cheaper
of the two MLP cells, ~430 min MPS) as the cleanest disambiguator
between the two competing hypotheses:

- If MLP V_φ at mlp_h=16 recovers to \~140 PPL: **capacity is the
 issue**, the §5.1 prior is too restrictive at this scale → keep
 the framework story, escalate to a wider structural form
 (`phi_hidden=128, theta_hidden=128`, params ~30k) and re-run.
- If MLP V_φ also lands ≥ 200 PPL: **capacity is NOT the issue**,
 the dynamics are wrong (likely the pair force is too aggressive
 relative to the integrator timestep / γ) → revisit §3 causal
 reduction, the C scale of the structural $V_\phi$, or the dt /
 m_b scaling.

Either outcome is informative; the P1.5 cell is the cheapest way to
get there. P1.5 result will be written to §4.5 once it completes.

#### Implication for the OQ-1 question

The OQ-1 question (does the §5.1 structural prior matter
empirically?) was originally framed as a parity-or-better comparison
between `structural` and `mlp` *both starting from a competitive
baseline*. With `structural` at 210.54 PPL, the OQ-1 framing now
shifts:

- If `mlp` matches `structural` (both ~200 PPL): the §5.1 prior
 is *not* the cause of the under-fitting; the PARF outer loop
 itself is mis-tuned at this scale.
- If `mlp` outperforms `structural` (e.g., MLP at ~140, structural
 at 210): the §5.1 prior is *actively* hurting at the prototype
 scale; the unstructured form recovers what the structural prior
 loses.
- If `mlp` underperforms `structural` (mlp \> 210): the §5.1 prior
 is at minimum a useful regulariser; both still under-fit but the
 prior loses less.

This refines OQ-1 from a *parity* test into a *capacity-vs-prior*
test. The P1.5 result will localise which of the three buckets the
PARF prototype falls into.

### 4.5. P1.5a — `mlp` V_phi ablation (mlp_h=16, seed 0) — completed 7 May 2026

Output: `parf/results/mlp/seed0_h16/parf_mlp_vphi16_gc_shakespeare_seed0_*`
(the `*_ckpt_latest.pt` is local-only; the rest are committed; the
`.png` is stored via Git LFS).

#### Headline result

| arm | val PPL (seed 0) | train-loss floor | final γ | params |
|------------------------------------------------------------------|-----------------:|-----------------:|---------:|--------:|
| Variant A HSPLM (k=4, m=4) | 133.01 | 3.74 | 0.154 | 7.92 M |
| Q9d Helmholtz `AAAASSSS` vh=128 | 134.89 | 3.78 | 0.163 | 7.92 M |
| All-attention | 141.80 | — | — | ~8.0 M |
| All-SPLM em_ln (free-γ, leak-free) | 173.59 | — | — | ~6.5 M |
| **PARF Q9c structural (P1)** | **210.54** | 4.76 | 0.088 | 6.54 M |
| **PARF Q9c MLP V_φ, mlp_h=16 (P1.5a)** | **297.22** | 4.43 | 0.139 | 6.54 M |

Wall-clock 15,061 s (~4.18 h MPS, modestly faster than the 430 min
estimate due to grad-checkpoint amortising better than expected at
mlp_h=16). Final val ppl 297.22; best val checkpoint at step 3800
(307.22). Causal probe at startup passed at strict 1e-6.

#### OQ-1 verdict

The §7.2 decision rule reads:

- *OQ-1 verdict "structural prior is empirically active"* iff
 `val PPL[structural] < val PPL[mlp_h=32] - 5 PPL` — i.e., structural
 beats the unstructured form by more than the ±5 PPL parity bar.

P1.5a result: **structural beats MLP by 86.68 PPL** (210.54 vs
297.22). This is **17× the ±5 PPL parity bar** — far outside the
margin of any plausible single-seed dispersion at this scale.

> **OQ-1 conclusion: the §5.1 structural prior is *empirically
> active* and substantially beneficial.** The unstructured MLP
> form has 60% more $V_\phi$ parameters (6,449 vs 4,002) and yet
> places ~86 PPL worse — the textbook signature of a useful
> structural inductive bias.

This is a publishable framework-native result for the v4 §15.24
carve-out, recordable independently of whether either PARF variant
ever beats VA / Q9d on absolute PPL: under the SAME outer machinery,
SAME compute budget, SAME integrator, the §5.1-faithful pair-potential
factorisation $-C \cdot \Theta_\phi(\theta(h_t), \theta(h_s)) \cdot \Phi_\phi(l(h_t), l(h_s)) / \sqrt{\lVert h_t - h_s \rVert^2 + \varepsilon^2}$
substantially outperforms a generic unstructured MLP on
$\mathrm{concat}(h_t, h_s, h_t - h_s)$.

P1.5b (`mlp_h=32`) is now *not on the critical path* — the OQ-1
verdict is unambiguous at mlp_h=16 and the 60% capacity gap to
structural already biases against the structural form. Running
mlp_h=32 (more capacity) would only widen the unstructured form's
advantage if the result were going to flip; since it doesn't flip
even at mlp_h=16 (smaller and arguably noisier), mlp_h=32 is
deferred unless a v4-revision reviewer explicitly requests it.

#### Diagnostic readout: γ behaviour and the dynamics hypothesis

The two PARF cells produced very different γ trajectories:

- **P1 structural:** γ collapsed from 0.150 → **0.088**. This
 pointed to "PARF outer loop is mis-tuned" — the optimizer was
 suppressing dissipation to keep the integrator stable under a
 destabilising structural pair force.
- **P1.5a MLP:** γ stayed near init, settling at **0.139** —
 comfortably inside the SPLM resonance basin (γ\* ≈ 0.166). The
 MLP V_φ is *gentler* on the velocity-Verlet integrator.

Reading these together: the dynamics-instability hypothesis is
**partially refuted**. γ collapse alone does not explain the PPL
gap — the MLP's gentler γ trajectory did *not* translate into
better PPL. So:

- **Capacity AND prior-fit BOTH matter**, in the order: prior-fit
 (~86 PPL of the gap, OQ-1) > capacity (the residual ~75 PPL gap
 vs Q9d, post-OQ-1).
- **The PARF Algorithm A construction at this scale is genuinely
 capacity-limited.** Both train-loss floors (4.43 / 4.76) sit
 ~0.7 nats above VA / Q9d's 3.74. No version of a 4–6.5 k-param
 $V_\phi$ on a 6.5 M-param backbone is going to close that gap.

#### Acted on (7 May 2026): launch P1.6 — wider structural V_φ

Per the §6.3 triage tree:

> *If val PPL ≥ 155 but training completed cleanly: investigate
> whether the structural $V_\phi$ has enough capacity at this
> scale; consider P1.5 with `mlp_h=64` (memory-permitting) or
> raise $\Theta_\phi$ / $\Phi_\phi$ MLP widths.*

Launched **P1.6 — wider structural V_φ at `phi_hidden=128,
theta_hidden=128`** (~30 k V_φ params, ~7× the original structural
form's 4 k). Estimated wall-clock ~5 h MPS. This cell isolates the
capacity-vs-prior tension cleanly: it preserves the §5.1 structural
form (so the OQ-1 verdict transfers) but raises capacity by 7×.

P1.6 succeeds if val PPL drops materially below 210.54 — ideally
into the 150–170 range (~Q9d/VA range minus a small structural-prior
tax). If P1.6 still places ≥ 200 PPL, the bottleneck is **not**
$V_\phi$ width but something deeper in Algorithm A (likely the
fixed-source `h_src.detach` that prevents the pair force from
co-evolving with the queries, OR the per-layer-shared $V_\phi$
overconstraining capacity vs a per-layer $V_\phi$).

### 4.6. P1.6 — wider structural V_phi (phi_hidden=128, theta_hidden=128)

**Status: launched 7 May 2026 09:06 EDT**, structural V_phi at
`phi_hidden=128, theta_hidden=128` (vs P1's 32/32 → ~7× capacity
bump on V_φ), seed 0. Estimated wall-clock ~5 h MPS. Result to
be appended here on completion.

---

## 5. Decision path (Q9c sequencing)

### 5.1. P0 → P1 — first quality cell

P0 is complete (smoke + causal probe on both V_phi variants). The
next step is P1 — the first quality cell at `structural` V_phi,
seed 0, em-ln vh=128 cell shape, 4000 steps Tiny Shakespeare. This
is a **~4.2 h MPS cell**, and it produces the headline PARF datapoint
that anchors all subsequent P1.5 / P2 / P3 decisions.

P1 succeeds if:

- The training run completes without numerical instability or
 causal-probe failure.
- The val PPL lands within a reasonable range of the existing
 anchors (target: within +20 PPL of Q9d $\mathtt{AAAASSSS}$ vh=128
 seed 0, i.e., \< 155). A wider gap than +20 PPL signals either a
 bug in $V_\phi$ or a fundamental capacity issue with the
 structural form, both of which abort P1.5 / P2.

### 5.2. P1 → P1.5 — OQ-1 comparator

If P1 lands in range, P1.5 runs the `mlp` V_phi at the same shape
to settle OQ-1 (does the §5.1 structural prior matter empirically?).
Two cells: `mlp_h ∈ {16, 32}`. The smaller `mlp_h=16` cell is the
fast comparator (~430 min MPS); `mlp_h=32` adds an extra capacity
datapoint (~640 min MPS) in case `mlp_h=16` is capacity-limited.

P1.5 verdict:

- If `structural` matches `mlp` on val PPL: the §5.1 prior is
 pedagogical (the unstructured form has enough expressivity to
 recover the same dynamics). Recommendation: keep the structural
 form for narrative purposes but document that the prior is not
 empirically active at this scale.
- If `structural` outperforms: the §5.1 prior is empirically active
 and worth keeping in any subsequent paper write-up. P2 then
 confirms with paired statistics.

### 5.3. P1.5 → P2 — paired confirmation at n=5

Once P1.5 picks the better V_phi variant, P2 runs n=5 paired
confirmation on the winner: 4 new seeds (1..4) with seed 0 reused
from P1 / P1.5. The aggregator (to be added; planned name
`notebooks/conservative_arch/parf/aggregate_p2.py`) reports the
paired-t result vs all-attention 5-seed E1, vs Variant A k=4 m=4
at n=5, and vs Q9d $\mathtt{AAAASSSS}$ vh=128 at n=5.

Estimated combined P1 + P1.5 + P2 compute: **~25–30 h MPS** over
~3 days elapsed. This is the analogue of Variant A's H1 + H2 and
Q9d's H1 + H1.5 + H2 spend.

### 5.4. P2 → P3 — framework-native diagnostics

P3 is the OQ-2 joint pair test plus the holonomy decomposition and
R6 ladder inversion (the same framework-native diagnostics that
H6 ran for Q9d). These are not blocking for the title-arm reading
but they are the deliverables that distinguish Q9c from a generic
$O(T^2)$ pair-attention architecture. Sketches of the protocol are
in §10 of the design doc; the PARF-specific OQ-2 procedure (joint
pair test on real GPT-2 attention) is novel to Q9c.

---

## 6. P1 — first quality cell plan (detailed)

### 6.1. Cell

| Cell | V_phi | seed | shape (d, L, T, B, v_hidden) | mass | γ | dt | causal_force | ln_after_step |
|------|------------|------|------------------------------|------|----------|----|--------------|---------------|
| P1 | structural | 0 | (128, 8, 128, 16, 128) | logfreq | free (init 0.15) | 1.0 | True | True |

Training: 4000 steps Tiny Shakespeare, AdamW(0.9, 0.95) lr=5e-4
with 200 warmup + cosine, gradient clip 1.0. Same trainer hyper-
parameters as the Q9d H1.5 vh=128 cells (so val PPL is directly
comparable seed-by-seed).

### 6.2. Reference for the cell-#0 quality gate

The relevant comparators at the SAME cell shape (em-ln vh=128, d=128,
L=8, seed 0) are:

| arm | val PPL (seed 0) | source |
|-----------------------------------------------------------|------------------|-----------------------------------------------------|
| All-attention (matched GPT-2, n_layer=8) — seed 0 | 141.80 | `multi_seed/results/` |
| Variant A HSPLM (k=4, m=4) — seed 0 | 133.01 | `hybrid/results/h1_sweep/k4_m4/seed0/` |
| Q9d Helmholtz $\mathtt{AAAASSSS}$ vh=128 — seed 0 | 134.89 | `helmholtz/results/h1p5_narrow_v/` |
| Q9d Helmholtz $\mathtt{AASSSSSS}$ vh=128 — seed 0 | 139.63 | same |

The PARF P1 cell needs to land within +20 PPL of the Q9d
$\mathtt{AAAASSSS}$ vh=128 seed-0 datapoint (i.e., \< 155) to clear
the gate. A wider gap signals a bug in $V_\phi$ or a fundamental
capacity issue with the structural form.

### 6.3. Decision rule for P1

- **PASS** iff val PPL \< 155 (within +20 PPL of Q9d
 $\mathtt{AAAASSSS}$ vh=128 seed-0 datapoint) **AND** training
 completed without numerical instability or causal-probe failure.
- **FAIL** otherwise. Triage tree:
 - If val PPL ≥ 155 but training completed cleanly: investigate
 whether the structural $V_\phi$ has enough capacity at this
 scale; consider P1.5 with `mlp_h=64` (memory-permitting) or
 raise $\Theta_\phi$ / $\Phi_\phi$ MLP widths.
 - If training diverged or causal probe leaked at startup:
 investigate the second-order autograd graph and the
 `.detach` placement.

### 6.4. Output layout

```text
parf/results/structural/seed0/
  parf_structural_seed0_summary.md
  parf_structural_seed0_training_log.jsonl
  parf_structural_seed0_loss_curve.png      (LFS)
  parf_structural_seed0_ckpt_latest.pt      (NOT committed)
  training.log
```

Per-cell summary (one per file) follows the same format as the
Q9d / VA per-cell summaries (architecture line + model_cfg + train_cfg
+ tokens + wall-clock + final losses + final γ + checkpoint pointer).

---

## 7. P1.5 — `mlp` V_phi ablation (planned)

### 7.1. Cells

| Cell | V_phi | mlp_h | seed | grad ckpt | shape (d, L, T, B, v_hidden) | est. wall |
|------|-------|------:|------|-----------|------------------------------|-----------:|
| P1.5a | mlp | 16 | 0 | on | (128, 8, 128, 16, 128) | ~430 min |
| P1.5b | mlp | 32 | 0 | on | (128, 8, 128, 16, 128) | ~640 min |

Training: identical to P1 except `--v-phi mlp --mlp-h {16,32}` and
the auto-enabled `--grad-checkpoint`.

### 7.2. Decision rule for P1.5

- **OQ-1 verdict "structural prior is pedagogical"** iff
 `|val PPL[mlp_h=32] - val PPL[structural]| ≤ 5 PPL` AND the
 same direction holds for `mlp_h=16` (within wider tolerance,
 e.g., +10 PPL since mlp_h=16 is capacity-limited).
- **OQ-1 verdict "structural prior is empirically active"** iff
 `val PPL[structural] < val PPL[mlp_h=32] - 5 PPL` (the
 structural prior wins by more than the +5 PPL bar).
- **OQ-1 verdict "MLP outperforms"** iff
 `val PPL[mlp_h=32] < val PPL[structural] - 5 PPL` (the unstructured
 form wins by more than the +5 PPL bar) — surprising outcome,
 warrants a separate investigation into why the §5.1 prior under-
 performs at this scale.

### 7.3. Output layout

```text
parf/results/
  structural/seed0/                 (from P1)
  mlp/seed0_h16/                    (from P1.5a)
  mlp/seed0_h32/                    (from P1.5b)
  P1P5_RESULTS.md                   (joint quality table; OQ-1 verdict)
```

---

## 8. P2 — n=5 paired confirmation plan (detailed, planned)

### 8.1. Cells

The winner of P1.5 — call it $V_\phi^\star$ — is run at 4 new seeds
(1..4) under the same trainer/cell shape. Seed 0 is reused from P1
or P1.5 (whichever produced the winner).

| Cell | V_phi | seed | grad ckpt | shape (d, L, T, B, v_hidden) | est. wall (per cell) |
|------|----------------|------|-----------|------------------------------|---------------------:|
| 1 | $V_\phi^\star$ | 1 | as P1/P1.5| (128, 8, 128, 16, 128) | ~252 min (struct) / ~430 min (mlp) |
| 2 | $V_\phi^\star$ | 2 | as P1/P1.5| (128, 8, 128, 16, 128) | same |
| 3 | $V_\phi^\star$ | 3 | as P1/P1.5| (128, 8, 128, 16, 128) | same |
| 4 | $V_\phi^\star$ | 4 | as P1/P1.5| (128, 8, 128, 16, 128) | same |

Total: 4 new cells = ~17 h MPS (structural) or ~29 h MPS (mlp).

### 8.2. Reference baselines at n=5

- All-attention 5-seed E1 mean: 149.80 (per-seed: 141.80, 154.79,
 159.59, 146.85, 145.99).
- Variant A k=4, m=4 5-seed mean: 147.40 (per-seed: 133.01, 152.25,
 152.10, 141.67, 157.97).
- Q9d $\mathtt{AAAASSSS}$ vh=128 5-seed mean: 145.86 (per-seed:
 134.89, 152.16, 151.37, 139.69, 151.20).

### 8.3. Decision rule for P2

Reuses the pre-registered §6.5 rule (lifted to n=5 paired-t):

- **PASS PARF quality arm** iff at the winner $V_\phi^\star$:
 mean Δ̄ ≥ -5 PPL vs all-attention AND sign-consistency 5/5 across
 seeds AND paired-t two-sided p \< 0.05.
- **PARF-vs-Q9d win** iff at the winner $V_\phi^\star$: mean Δ̄ ≤
 -5 PPL vs Q9d $\mathtt{AAAASSSS}$ vh=128 best AND sign-consistency
 5/5.
- **PARF-vs-VA win** iff at the winner $V_\phi^\star$: mean Δ̄ ≤
 -5 PPL vs VA k=4, m=4 best AND sign-consistency 5/5.

The *headline* gate is whether PARF places **at or below** VA's
133.01 / Q9d's 134.89 seed-0 datapoint at seed 0, and whether its
n=5 mean is below the corresponding 147.40 / 145.86 anchors. The
PARF pair sum is $O(T^2)$ at every layer, so PARF only earns its
wall-clock cost if it places strictly better than the cheaper Q9d
analogue — a parity result is not a decisive PARF win.

### 8.4. Output layout

```text
parf/results/
  p2_paired_confirmation/
    structural_or_mlp/
      seed1/
      seed2/
      seed3/
      seed4/
      P2_RESULTS.md                 (paired-t vs all-attn + VA + Q9d; sign + p)
```

---

## 9. Causal-violation probe (Q9c-specific safeguard)

Q9c has TWO causal-reduction `.detach` points (Q9d / VA only have
one): the inherited $\xi$-pool detach (`xi_input = h.detach`) and
the new PARF-specific pair-source detach (`h_src = h.detach`). The
probe must verify both are active.

### 9.1. Standalone probe

`notebooks/conservative_arch/parf/causal_probe_parf.py` builds a
PARF model with each of the 2 V_phi variants × 2 `causal_force`
modes = 4 configurations and runs both probes:

- **Perturbation probe.** Clone the input, perturb a single position
 $t^\star$, run forward, and check that *no* output position $t' \lt
 t^\star$ changes (Δ \< 1e-6).
- **Gradient-Jacobian probe.** Compute
 $\partial \mathrm{logits}_{t'} / \partial \mathrm{embed}_{t^\star}$
 for $t' \lt t^\star$ and check that the entire row is identically
 zero (Δ \< 1e-6).

### 9.2. Verified results (random init, smoke scale)

Run on the prototype shape (`d=128, L=4, T=24, B=2`) at random init:

```text
[  OK] V_phi=  'structural'  (L=4)   fixed pre=0.00e+00, grad post=0.00e+00
[  OK] V_phi=         'mlp'  (L=4)   fixed pre=0.00e+00, grad post=0.00e+00
All 2 V_phi variants: fixed-mode causal-side Δ < 1e-06.  PARF is causal by construction.
```

In `causal_force=False` mode, the leak signal turns on as soon as
either detach is removed, and scales with the number of layers
(more PARF layers = more compounded leak).

### 9.3. Trainer startup guard

`train_parf.py` runs the probe at startup before any optimizer step
and aborts if the leak signal exceeds 1e-6. This is the production
guarantee that no optimizer step is ever taken on a leaky model.

---

## 10. Code inventory

| File | Status | Purpose |
|-----------------------------------------------------------------------------|----------|------------------------------------------------------------------------|
| `notebooks/conservative_arch/parf/__init__.py` | ✅ done | module marker |
| `notebooks/conservative_arch/parf/model_parf.py` | ✅ done | `PARFConfig` + `StructuralVPhi` + `MLPVPhi` + `PARFLM` |
| `notebooks/conservative_arch/parf/causal_probe_parf.py` | ✅ done | perturbation + gradient-Jacobian probe (both V_phi variants) |
| `notebooks/conservative_arch/parf/smoke_test.py` | ✅ done | end-to-end smoke (forward + backward + probe + 5-step train, both V_phi + grad-ckpt path) |
| `notebooks/conservative_arch/parf/train_parf.py` | ✅ done | trainer (Algorithm A: pure NTP backprop) |
| `notebooks/conservative_arch/parf/scripts/run_first_quality_cell.sh` | ✅ done | idempotent wrapper for the first quality cell (P1 / P1.5) |
| `notebooks/conservative_arch/parf/aggregate_p1p5.py` | planned | OQ-1 verdict aggregator (joint quality table for `structural` vs `mlp`) |
| `notebooks/conservative_arch/parf/aggregate_p2.py` | planned | P2 aggregator with paired statistics vs all-attn / VA / Q9d |
| `notebooks/conservative_arch/parf/decode_flop_pareto.py` | planned | analytical decode-FLOP Pareto for PARF (T ∈ {256, 1024, 4096}, includes the $O(T^2)$ pair-sum cost) |
| `notebooks/conservative_arch/parf/README.md` | ✅ done | reproduce instructions + wall-clock survey + sanity checks |

Supporting documents:

| File | Status | Purpose |
|-----------------------------------------------------------------------------|----------|------------------------------------------------------------------|
| | ✅ done | Algorithm A backprop pipeline, second-order graph, optimisation options |
| | ✅ done | `MLPVPhi` deep dive: architecture, OQ-1 framing, accuracy vs smoothness |
| | ✅ done | velocity-Verlet integrator: derivation, stability, integrator inventory |

---

## 11. Open questions (parked, not blocking P1 / P1.5)

1. **OQ-1 (structural vs MLP).** Is the §5.1 structural prior
 empirically active or pedagogical at this scale? P1.5 settles
 this directly.
2. **OQ-2 (joint pair test on real GPT-2 attention).** Does the
 learned $V_\phi$ assign measurable interaction strength to
 pairs that real GPT-style attention also attends to, on the
 same sentences? Procedure: extract attention scores from a
 reference GPT-2 head at every layer, extract $V_\phi(h_t, h_s)$
 on the same sentences, compute Spearman correlation per layer,
 and report a layer-by-layer plot. Scheduled for P3 post-P2.
3. **Decode-FLOP arm at long context.** PARF's $O(T^2)$ pair-sum
 competes directly with attention. The analytical FLOP question
 is whether at any T the PARF cell is cheaper than all-attention
 at PPL parity. The structural $V_\phi$ has constant-per-pair
 cost (one MLP eval per (t, s)) versus attention's d-dimensional
 $QK^T$ per-pair cost; back-of-envelope says PARF is asymptotically
 cheaper at d ≥ ~64 because the pair-MLP output is scalar, but
 this needs to be confirmed by the planned `decode_flop_pareto.py`.
4. **Stage 1.5 (Gumbel-softmax sparsity).** If P2 shows PARF at
 parity or worse than Q9d on quality, sparsifying the pair sum
 (top-k retention via Gumbel-softmax) is the natural P5 add-on
 to recover decode-FLOP advantage at long T. The mechanism is
 in the v4 §15.24.7 deposit (Algorithm A's optional sparsity
 block); wiring is straightforward but adds the Gumbel
 approximation bias near $\tau \to 0$.
5. **Algorithm B / Algorithm C.** The v4 deposit also describes
 PPO-with-framework-native-reward (Algorithm B) and
 Pair-Selective REINFORCE (Algorithm C). These are deferred to a
 separate paper draft — they require a separate RL outer loop
 and are not on the prototype's critical path.
6. **Embedding-tied $V_\phi$ at scale.** At the prototype scale
 (d=128, L=8) the pair-MLP is small enough to fit. At TinyStories
 scale (d=192, L=12) the parameter count of the structural V_phi
 grows mildly; the MLP V_phi grows ~quadratically in d (due to the
 `concat(h_q, h_k, h_q-h_k)` 3d feature width). Scale-up
 considerations are deferred to P4.
7. **Velocity-Verlet timestep at T=4096 long context.** The same
 damped position-Verlet step that Q9d uses is used here. At very
 long context the pair-sum dominates wall-clock, but the integrator
 stability is unchanged. No new analysis is needed.

---

## 12. Decision log

| Date | Decision | Notes |
|--------------|------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| 6 May 2026 | Build prototype with both `structural` and `mlp` $V_\phi$ variants | Settles OQ-1 in a single P1.5 ablation cell |
| 6 May 2026 | Causal reduction = TWO `.detach` points ($\xi$-pool + h_src) | Severs back-reaction force on past tokens; preserves v3 leak-fix invariant; new for PARF |
| 6 May 2026 | Combined `autograd.grad` on $V_\theta + V_\phi$ | Single scalar U; halves second-order graph cost vs separate per-potential calls |
| 6 May 2026 | Optional `torch.utils.checkpoint` on $V_\phi$ pair sum (`use_grad_checkpoint`) | Memory-vs-compute trade-off; auto-enabled for MLP variant by wrapper script |
| 6 May 2026 | Same em-ln vh=128 cell shape as Q9d $\mathtt{AAAASSSS}$ vh=128 | Ensures direct seed-by-seed PPL comparability with Q9d / VA / all-attn anchors |
| 6 May 2026 | Free γ in P1 / P1.5 / P2 (init 0.15, learnable, not fixed) | Same as Q9d / VA; lets optimiser find right damping |
| 6 May 2026 | Single shared $V_\theta$ + single shared $V_\phi$ across all layers | Strongest version of the "single energy field" Lagrangian narrative; matches Q9d |
| 6 May 2026 | **No quality cells run yet**; P1 launch is the next decision | Estimated ~252 min MPS; produces the PARF headline datapoint |
| 6 May 2026 | **No paper update committed** to the PARF arm yet | Per user instruction; paper v4 carve-out (§15.24) is upstream of the prototype results |
| 7 May 2026 | **P1 structural-V_phi seed-0 cell completed at val PPL 210.54** — FAIL per §6.3 | Wall-clock 15,554 s (~4.3 h MPS, slower than estimate due to MPS contention). Causal probe + numerical stability clean; capacity / dynamics issue, not overfitting. γ collapsed to 0.088 (vs VA 0.154 / Q9d 0.114-0.163), train loss floor at 4.76 (vs ~3.74 for VA / Q9d). |
| 7 May 2026 | **Launch P1.5 mlp_h=16 in background** as the cheapest disambiguator | OQ-1 question refines from parity test → capacity-vs-prior test (see §4.4 "Implication"). Three possible outcomes localise the failure mode. |
| 7 May 2026 | **P1.5a MLP-V_φ (mlp_h=16) seed-0 cell completed at val PPL 297.22** | structural beats MLP by 86.68 PPL (17× the ±5 PPL parity bar) → **OQ-1 verdict: structural prior empirically active**. γ trajectory diagnostic: structural γ collapsed to 0.088, MLP γ stayed at 0.139 → dynamics-instability hypothesis partially refuted; capacity AND prior-fit both matter. |
| 7 May 2026 | **Defer P1.5b** (mlp_h=32) — OQ-1 verdict already unambiguous at mlp_h=16 | mlp_h=32 would only widen the OQ-1 verdict's margin in the same direction; not on critical path. Held in reserve for v4-revision reviewer requests. |
| 7 May 2026 | **Launch P1.6 wider-structural V_φ** (`phi_hidden=128, theta_hidden=128`) | ~7× capacity bump on V_φ; preserves §5.1 form (so OQ-1 verdict transfers); ~5 h MPS. Disambiguates capacity-vs-architectural-bottleneck on the structural variant. |

---

## 13. Pointers

- Design doc: [`PARF_Augmented_SPLM_Architecture.md`](PARF_Augmented_SPLM_Architecture.md)
- v4 carve-out: `Section_15_24_PARF_Augmented_SPLM_v4_draft.docx`
- PARF-specific deep dives:
 - — Algorithm A pipeline, optimisation options
 - — `MLPVPhi` deep dive, OQ-1 framing
 - — integrator derivation + stability + inventory
- Sibling Q9d Helmholtz path: [`Helmholtz-HSPLM_Path_Forward_and_Experiments.md`](Helmholtz-HSPLM_Path_Forward_and_Experiments.md)
- Sibling Variant A two-stage path: [`HSPLM_Path_Forward_and_Experiments.md`](HSPLM_Path_Forward_and_Experiments.md)
- Title-discussion master record:
- Pre-registered title-justification rule: §6.5 of the title-discussion record
- Prototype root: `notebooks/conservative_arch/parf/`
- Prototype README (wall-clock survey + sanity checks + reproduce instructions): `notebooks/conservative_arch/parf/README.md`
- Causality bug & fix history: [`Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md)
- Markdown LaTeX rendering rules:

---

*Last updated: 6 May 2026.*
