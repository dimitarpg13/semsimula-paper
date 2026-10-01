# Gradient Starvation Across the Ladder: an Investigation Programme

**Status:** opened 2026-09-30 as the programme's top priority. Tier 0 done
(2026-09-30): the V_φ and ξ source gradients are exactly zero in both trained arms.
Tier 1 done (2026-10-01): `vphi_grad_path` / `xi_grad_path`
built, forward bit-identical, source gradient opened. Next: P2.1.
**Companion to:**
[`Depth_Ladder_and_Matched_Baseline_Protocol.md`](Depth_Ladder_and_Matched_Baseline_Protocol.md)
§5.8 (the gradient-path probes and the full `rglive` run),
[`Paper_v6_Section_Audit.md`](Paper_v6_Section_Audit.md),
[`Hyperparameter_Tuning_Checklist.md`](Hyperparameter_Tuning_Checklist.md) (the queue).

---

## 0. Why this is the top priority

The `attention_potential` arm detached its sources so that its exchange force
would stay a causal gradient. Detaching changes no forward value, but it cut
every loss gradient from the exchange field into the hidden states. With that
learning signal restored and the forward pass bit-identical
(`relax_grad_path='live'`), the same arm went from the ladder's second-worst to
its best Fock arm:

| arm | settled PPL |
| --- | ---: |
| `attention_potential`, starved (as trained) | 80.90 |
| `attention` (non-conservative field) | 63.51 |
| **`attention_potential`, live gradients** | **61.11** |

The same detach pattern is used in **two more channels, in every arm the
programme has trained** (`parf/model_parf_multixi.py`):

| channel | code | what the backward pass loses |
| --- | --- | --- |
| **V_φ**, the pair potential | `_pair_potential`: `h_src = h_in.detach()` (l.717); the top-k score head's source input when `score_head_use_detached_h_src` (l.719) | earlier tokens never learn to be useful pair *sources* |
| **ξ**, the context channels | `xi_input = h.detach()` (l.897, l.959) | earlier tokens never learn what to write into the running context that V_θ(ξ, h) reads |
| exchange field | fixed by `rglive` | — |
| reverse channel, creation gate | already live | — |

So in every trained arm, the only token-to-token paths that carried a learning
signal were the Fock register path and, in `attention`, the exchange field.
Three ladder conclusions rest directly on the starved convention:

1. **"V_φ is inert"** (E1: −0.0015; F5). The same detach produced the same
   symptom as the exchange field's.
2. **The price of the Fock mechanism, +31.3%** (conservative-only 87.93
   against no-exchange 66.98). With the reverse channel off, the
   conservative-only arm's only inter-token channels are V_φ and ξ — both
   starved.
3. **The queued factorial** (`splm-multixi`, `fock-splm`), which exists to
   price V_φ.

**Paused until this programme decides the gradient convention:** D1, the two
factorial arms, and any restatement of the ladder in the cards or the book.
**Not paused:** the 6b-7/9/12/13 probes on the finished `rglive` run.

---

## 1. Design principles

- **Forward identical, backward live.** Every switch keeps the forward force
  exactly as trained — the partial gradient in the token's own coordinates, with
  sources held fixed — and changes only which tensors the loss gradient reaches.
  The `rglive` construction is the template.
- **Verify before any GPU time.** Each switch is checked on a real checkpoint:
  forward bit-identical (or float-reordering only) to the parent, source gradient
  non-zero, SCAF CLEAN.
- **Cheapest first, with a stop rule at every tier.** A tier runs only if the
  one before it says it is worth running.
- **Pre-register before launch.** Bands below are *provisional*; each is frozen
  in the protocol note before its run starts.
- **Every run gets its own tag component,** so no probe can resume from or
  overwrite a finished arm.

---

## 2. The experiments, in order of cost

### Tier 0 — offline, free (laptop, existing checkpoints)

Checkpoints: `…_noattn_best.pt` (no-exchange, 66.98) and `…_norc_…_noattn_best.pt`
(conservative-only, 87.93), both mirrored locally. Harness:
`scaleup/debug/gradcheck_exchange_paths.py` extended.

| id | question | measurement | expected under starvation |
| --- | --- | --- | --- |
| **G0.1** | Is the source gradient through **V_φ** exactly zero? | backprop a last-position cotangent through the V_φ pair force only; gradient reaching earlier tokens | exactly 0 (autograd: no path) |
| **G0.2** | Is it zero through **ξ**? | the same through the V_θ(ξ, h) force, via ξ | exactly 0 |
| **G0.3** | Does the **score head** routing receive source gradient? | gradient into earlier tokens via the top-k scores | 0 if `score_head_use_detached_h_src` is set in these runs |
| **G0.4** | How large is each channel's forward force? | RMS of V_θ, V_φ and reverse-channel forces per layer | V_φ small (consistent with "inert"); establishes what "waking up" would look like |

**Stop rule.** If G0.1–G0.3 are *not* zero, the starvation hypothesis is wrong
for these channels; the programme stops and the report says so.

#### Tier 0 result (2026-09-30): starvation confirmed; the stop rule does not fire

Harness: [`scaleup/debug/gradcheck_vphi_xi_paths.py`](../notebooks/conservative_arch/scaleup/debug/gradcheck_vphi_xi_paths.py).
Output:
[`scaleup/results/gradient_starvation/`](../notebooks/conservative_arch/scaleup/results/gradient_starvation/).
Inputs: the two `_best.pt` checkpoints (no-exchange step 31,500; conservative-only step 31,000), 2 × 512 validation tokens (seed 20260930), and a random cotangent on the last position.
The harness propagates it back through each layer's force at the trained layer inputs.

| arm | layer | RMS F_θ | RMS F_φ | RMS F_rev | V_φ → earlier, as trained | V_φ → earlier, live | ξ → earlier, as trained | ξ → earlier, live |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-exchange | 0 | 0.0000 | 0.0006 | 0.0173 | **0** | 2.2e-2 | **0** | 4.3e-5 |
| no-exchange | 1 | 0.0109 | 0.0008 | 0.0147 | **0** | 1.7e-2 | **0** | 3.8e-1 |
| conservative-only | 0 | 0.0056 | 0.0005 | — | **0** | 2.8e-2 | **0** | 1.0e-1 |
| conservative-only | 1 | 0.0048 | 0.0000 | — | **0** | 7.8e-8 | **0** | 1.4e-1 |

"Live" is the Tier 1 construction applied offline. The forward force is unchanged: its maximum difference from the trained force is **exactly 0** in every row, for both V_φ and ξ. Only the source tensor is left live for backprop.

- **G0.1, G0.2: exactly zero, as the hypothesis predicts.**
  - No gradient reaches an earlier token through V_φ or through ξ, in either arm, at either layer.
  - Autograd has no path. This is not a small number.
  - The target token's own gradient through V_φ is non-zero (0.04–0.09), so the channel is wired and working. Only its sources are cut off.
- **G0.3: the score head is not starved, only blind to its sources.**
  - `score_head_use_detached_h_src = True` in both runs, so its source input is detached. G0.1's zero already includes that route.
  - Its parameters do learn, through the straight-through top-k mask inside the force: parameter-gradient norm 0.5–1.8 at every layer.
  - So the routing trains, but earlier tokens never learn to make themselves worth routing to.
- **The Tier 1 construction works as designed.**
  - The forward pass is bit-identical, and the source gradient becomes non-zero wherever the channel's force is non-zero.
  - That clears I1.1 and I1.2 in principle. The model-code switches still need their own checks on the real training path.
- **G0.4: what the forces look like.**
  - V_φ is small everywhere: 7–9% of F_θ's RMS where both are active.
  - In conservative-only layer 1, V_φ is **effectively off**: force below 5e-5 RMS, and the live gradient only 8e-8. The cause (per-layer scale, gate, or learned flatness) has not been checked.
  - In no-exchange layer 0, the V_θ force is about 0. The reverse channel (0.017) carries that layer's inter-token force, so live ξ has almost nothing to act through there (4e-5).
- **Reading for Tier 2.**
  - Where there is a force to act through, **live ξ offers 4–20× more source gradient than live V_φ (3.6× and 22×)**.
  - In the conservative-only arm, the "only inter-token channels" turn out to be ξ at both layers plus a weak V_φ at layer 0 alone.
  - So if P2.1 moves, ξ is the more likely cause. P2.4 (ξ only) is the more informative decomposition.
  - Gradient norm under one random cotangent measures what could be learned. It does not predict PPL. The bands stay as written.

**Decision:** proceed to Tier 1.

### Tier 1 — implementation and verification (laptop, no training)

| id | work | verification |
| --- | --- | --- |
| **I1.1** | `vphi_grad_path` (`'default'` / `'live'`): the pair force stays ∂V_φ(h_t, h_s)/∂h_t with sources held fixed in the forward, but h_s (and the score head's source input) are live for backprop. Implementation: build the target slot from an alias node and differentiate only with respect to it, so no reaction force appears. | forward Δlogit ≈ 0 vs parent; source gradient > 0; SCAF CLEAN |
| **I1.2** | `xi_grad_path` (`'default'` / `'live'`): ξ built from live h for backprop, but the V_θ force stays the partial in h with ξ held fixed — ξ enters the force computation through a separate node so ∂V/∂ξ · ∂ξ/∂h_t never enters the forward force. **Care:** the causal EMA at position t includes h_t itself. | as I1.1, on both checkpoints |
| **I1.3** | Notebook wiring: Cell 0 settings, tag components (`vplive`, `xilive`), Cell 5 pass-through, Cell 5b guard and banner — as for `rglive`. | tags distinct from every finished arm; guard fires on a missing component |

#### Tier 1 result (2026-10-01): switches built and verified; ready for Tier 2

**Code** (uncommitted):

- `parf/model_parf_multixi.py`: config `vphi_grad_path`, `xi_grad_path` (`'default'` / `'live'`). `'live'` is refused without `causal_force`, and for V_φ on the `xi_attention` pair potential.
  - **V_φ live.** `_layer_forces` passes `h_in` to `_pair_potential` as `h_src_live`, which becomes the V_φ sources and the score head's source input. Every force is then taken w.r.t. `h_in.view_as(h_in)`, an alias node. `autograd.grad(·, alias)` follows only the target slot, so the force has no reaction term.
    - The live source is gathered from (B, T, d) directly, not from the (B, T, T, d) expansion. Same values, but its backward is a scatter into (B, T, d).
  - **ξ live.** ξ is built from live `h`. In the CfC/BAOAB step the force is taken w.r.t. `h_mid`, which is downstream of `h`, so no path through ξ can enter it and no alias is needed.
    - The Verlet step differentiates w.r.t. `h` itself, so there the target is an alias, keeping ∂V/∂ξ·∂ξ/∂h_t out of the force.
    - Live ξ reaches `h` both through the force and through the CfC/low-rank linearisation, which is built from ξ.
- Ladder notebook:
  - Cell 0: `VPHI_GRAD_PATH`, `XI_GRAD_PATH`, and tag components `vplive` / `xilive`, placed after `rgdet`/`rglive` and before `L{L}probe`.
  - Cell 5: pass-through.
  - Cell 5b: tag guard and a "SOURCE-GRADIENT PROBE — NOT A LADDER POINT" banner.

**Verification.** Script: [`scaleup/debug/verify_vphi_xi_grad_path.py`](../notebooks/conservative_arch/scaleup/debug/verify_vphi_xi_grad_path.py). Output: [`results/gradient_starvation/tier1_verify_output.txt`](../notebooks/conservative_arch/scaleup/results/gradient_starvation/tier1_verify_output.txt).

Each arm was built through the notebook's own Cells 0–5b on the parent's `_best.pt`. The gradient is measured through one real layer step (`_layer_step_ex`, train mode), from the last position into earlier tokens.

| arm | switches | forward vs parent (eval / train) | layer 0 → earlier | layer 1 → earlier |
| --- | --- | --- | ---: | ---: |
| no-exchange | default | 0 / 0 | **0** | **0** |
| no-exchange | V_φ live | 0 / 0 | 23.9 | 0.36 |
| no-exchange | ξ live | 0 / 0 | 0.023 | 8.8 |
| no-exchange | both | 0 / 0 | 23.9 | 8.8 |
| conservative-only | default | 0 / 0 | **0** | **0** |
| conservative-only | V_φ live | 0 / 0 | 41.1 | 1.8e-17 |
| conservative-only | ξ live | 0 / 0 | 20.1 | 13.1 |
| conservative-only | both | 0 / 0 | 45.8 | 13.1 |

- **The forward pass is bit-identical** (max |Δlogit| = 0) in all eight builds, in eval and in train mode with the same Gumbel seed. The tiny random Verlet model, where the alias is required, is bit-identical too.
- **As trained, a layer step sends exactly 0 gradient to earlier tokens.** Within a step, V_φ and ξ are the only inter-token paths. The live switches open them.
- The pattern matches Tier 0:
  - V_φ carries nothing at conservative-only layer 1.
  - ξ carries little at no-exchange layer 0.
- The step-level gradients are larger than Tier 0's force-only ones because they include the CfC linearisation.
- **Cost:**
  - Train forward+backward time on CPU is unchanged (11.6–13.8 s for every build).
  - The `_smoke` tests of `model_parf_multixi.py` and `model_fock_parf_multixi.py` pass.
  - GPU memory is not measured yet: watch the first log lines of P2.1.
- **Causality.** No forward value changes, so the SCAF leak audit's result is unchanged by construction. The periodic audits in Cell 6 still run.
- **Tags:**
  - no-exchange: `…cgqk_vplive_xilive_L2probe…`;
  - conservative-only: `…cgqk_norc_vplive_xilive_L2probe…`.
  - Each switch alone gets its own tag (`vplive` or `xilive`), so P2.3 and P2.4 cannot collide with P2.1 or P2.2.

**Launch recipe for P2.1** (Colab, fresh session, notebook from GitHub). Cell 0:

```python
LADDER_L         = 2
LADDER_MECHANISM = 'none'
REVERSE_CHANNEL              = False     # conservative-only
PROBE_MAX_STEPS = 3_000
VPHI_GRAD_PATH         = 'live'
XI_GRAD_PATH           = 'live'
```

Cell 5b must print the SOURCE-GRADIENT PROBE banner and a tag containing `norc_vplive_xilive_L2probe`. Only then run Cell 6.

### Tier 2 — 3,000-step probes (Colab, about 1.5 GPU h each)

Same protocol as probe (b): `PROBE_MAX_STEPS = 3_000`, everything else at the
ladder defaults, scored at step 3,000 against the parent's own evals. Noise
band ±2.5% (the earlier probes on the exchange pair moved step 3,000 by 0.6–1.7%).

Parent trajectories (validation PPL):

| step | conservative-only | no-exchange | `attention` | `rglive` |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 267.41 | 257.74 | 248.81 | 255.98 |
| 2,000 | 183.10 | 167.87 | 158.76 | 162.27 |
| 3,000 | **158.09** | **140.61** | 131.20 | 133.74 |

| id | arm | switches | provisional prediction under starvation (freeze before launch) |
| --- | --- | --- | --- |
| **P2.1** | conservative-only | V_φ live + ξ live | **≤ 154.1** (> 2.5% below its parent's 158.09). Approaching no-exchange's 140.61 would mean the "price of the Fock mechanism" is largely starvation. **Sharpest test: the only inter-token channels are the starved ones.** |
| **P2.2** | no-exchange | V_φ live + ξ live | **≤ 137.1** (> 2.5% below 140.61) means V_φ/ξ add something once they can train, alongside the Fock path |
| P2.3 | whichever of P2.1/P2.2 moved | V_φ live only | decomposition: how much of the effect is V_φ |
| P2.4 | the same arm | ξ live only | decomposition: how much is ξ |

**Order:** P2.1, then P2.2. P2.3 and P2.4 run only if at least one of those
moves beyond noise.

**Stop rule.** If both P2.1 and P2.2 land within noise of their parents, the
detach costs nothing in these channels: the ladder stands as measured, the
queue resumes, and the report records a clean null.

### Tier 3 — full runs (Colab, about 15 GPU h each), only if Tier 2 shows an effect

| id | run | purpose |
| --- | --- | --- |
| F3.1 | the Tier 2 arm with the largest effect, full 32,500 steps | a settled number comparable with the ladder |
| F3.2 | **everything live**: `attention_potential` + `rglive` + live V_φ + live ξ | the strongest conservative-forward model the programme can build |
| F3.3 | the factorial arms (`splm-multixi`, `fock-splm`) in the chosen convention | the V_φ × Fock 2×2, measured on the right convention |

### Tier 4 — consolidation

- Re-run 6b-9 (E1), 6b-12 (F1) and 6b-13 on each live arm. Key question: **does
  V_φ's E1 attribution move away from −0.0015?**
- Probe (a) — `attention` with detached inputs — as the mirror-image control, if
  budget allows.
- Restate the ladder, the Fock price and the conservativity price in the
  protocol note, the model cards and the book (§3.5 of the audit item on the
  Conservative Obstruction hypothesis already expects this).

---

## 3. Budget

| tier | cost | cumulative |
| --- | --- | --- |
| 0 | about an hour of laptop CPU | ~0 GPU h |
| 1 | a few hours of implementation and checks | ~0 GPU h |
| 2 | 2 probes (+2 conditional) × ~1.5 h | 3–6 GPU h |
| 3 | 1–4 full runs × ~15 h | 15–60 GPU h |

---

## 4. Ledger

| date | item |
| --- | --- |
| 2026-09-30 | Programme opened. Motivating result: `attention_potential` + `rglive`, settled 61.11 (pre-registered 63–72; better than predicted). V_φ and ξ source detaches identified at `model_parf_multixi.py` l.717/719 and l.897/959. |
| 2026-09-30 | **Tier 0 done.** In no-exchange and conservative-only, the gradient reaching earlier tokens through V_φ and through ξ is exactly 0 at both layers; the score-head parameters still train. The offline live-source construction leaves the forward force bit-identical, and its source gradient is non-zero. V_φ is small (7–9% of F_θ) and effectively off at conservative-only layer 1; live ξ carries 4–20× more source gradient than live V_φ. Stop rule does not fire → Tier 1. |
| 2026-10-01 | **Tier 1 done.** Added `vphi_grad_path` / `xi_grad_path` to `model_parf_multixi.py` (alias-node target, live sources; flat gather) and wired `VPHI_GRAD_PATH` / `XI_GRAD_PATH` (tags `vplive` / `xilive`) through Cells 0, 5 and 5b. Across 8 builds on the parent checkpoints the forward is bit-identical; a layer step's gradient into earlier tokens goes from exactly 0 to non-zero; CPU step time is unchanged. Next: P2.1 (conservative-only, both live), band ≤ 154.1 at step 3,000. |
