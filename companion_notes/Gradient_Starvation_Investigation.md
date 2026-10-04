# Gradient Starvation Across the Ladder: an Investigation Programme

**Status:** opened 2026-09-30 as the programme's top priority. Tier 0 done
(2026-09-30): the V_φ and ξ source gradients are exactly zero in both trained arms.
Tier 1 done (2026-10-01): `vphi_grad_path` / `xi_grad_path`
built, forward bit-identical, source gradient opened. **P2.1 (2026-10-01): 127.73 vs 158.09, MOVE (−19.2%)**; **F3.1 (2026-10-02): 57.76 settled**, the best L=2 model in the programme. Vφ woke up (E1 −0.0015 → −0.291). Next: P2.2 / Fock-PARF live (the Fock mechanism's real value), then the rest of the live ladder.
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
| **P2.1** ✅ **127.73, MOVE (−19.2%)** | conservative-only | V_φ live + ξ live | **≤ 154.1** (> 2.5% below its parent's 158.09). Approaching no-exchange's 140.61 would mean the "price of the Fock mechanism" is largely starvation. **Sharpest test: the only inter-token channels are the starved ones.** |
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

#### The live-gradient ladder — planned 2026-10-01

F3.1's mid-run readings, a move at 3k and about 57–59 projected settled, make the live convention the programme's convention, not just a probe. Every ladder rung needs a live counterpart so the comparisons are made on one convention. Each arm is its parent's config plus every switch that applies to it:

| arm | base | Fock | exchange | live switches | tag | state |
| --- | --- | --- | --- | --- | --- | --- |
| PARF only | V_θ + V_φ + ξ | off | — | `vplive`, `xilive` | `…norc_vplive_xilive…` | **F3.1, running** |
| Fock-PARF `none` | V_θ + V_φ + ξ | on | — | `vplive`, `xilive` | `…vplive_xilive…_noattn` | queued (P2.2 → full) |
| Fock-PARF `attention` | V_θ + V_φ + ξ | on | non-conservative field | `vplive`, `xilive` | `…vplive_xilive…_attn` | queued |
| Fock-PARF `attention_potential` | V_θ + V_φ + ξ | on | conservative field | `rglive`, `vplive`, `xilive` | `…rglive_vplive_xilive…_attnpot` | queued (F3.2, everything live) |
| multi-ξ SPLM (run 10) | V_θ + ξ, **no V_φ** | off | — | `xilive` | `…norc_nophi_xilive…` | queued |
| Fock-SPLM (run 11) | V_θ + ξ, **no V_φ** | on | — | `xilive` | `…nophi_xilive…` | queued |
| L=4 Fock-PARF `none` | as above, L=4 | on | — | `vplive`, `xilive` | `…vplive_xilive_L4probe…` | queued: may also explain L=4 < L=2 |

**Verified before any GPU time** (all on the parent's `_best.pt`, or at fresh init where no parent exists):

- **Exchange-field arms**
  - Script: `verify_vphi_xi_grad_path.py --exchange`.
  - Output: [`results/gradient_starvation/tier1_verify_exchange_arms_output.txt`](../notebooks/conservative_arch/scaleup/results/gradient_starvation/tier1_verify_exchange_arms_output.txt).
  - Coverage: `attention`, `attention_potential` and `attention_potential` + `rglive`, each with all four switch combinations.
  - The forward pass is bit-identical in all 12 builds.
  - The live switches add gradient into earlier tokens on top of what the exchange field already carries. In starved `attention_potential` that gradient was exactly 0.
- **`pair_potential='none'`**
  - New in `model_parf_multixi.py`, wired as `PAIR_POTENTIAL` in Cells 0, 5 and 5b; the tag gains `nophi`.
  - Script: `verify_pair_potential_none.py`.
  - Output: [`results/gradient_starvation/pair_potential_none_verify_output.txt`](../notebooks/conservative_arch/scaleup/results/gradient_starvation/pair_potential_none_verify_output.txt).
  - V_φ, the score head and the per-layer scale are gone from the module and the `state_dict`: 137,803 parameters (0.18%).
  - Both arms build, train-step and backprop.
  - In multi-ξ SPLM, ξ is the only inter-token channel. A layer step sends exactly 0 gradient to earlier tokens by default (2.2 at layer 0 with `xilive`), and the forward pass is identical.
  - Regression: every existing-arm reading in `verify_vphi_xi_grad_path.py` is unchanged, and both `_smoke` suites pass.
  - `VPHI_GRAD_PATH = 'live'` with no V_φ is refused, in the model and in Cell 5b.

**Cost:** about 14 GPU h per L=2 arm and about 27 h at L=4. The full table is roughly 110 GPU h. Suggested order:

1. F3.1 (running).
2. L=4 Fock-PARF `none`, live.
3. Fock-PARF `none`, live.
4. F3.2, everything live.
5. Fock-PARF `attention`, live.
6. Run 10 and run 11, live.

Run 10 and run 11 go last because, with V_φ absent, the PARF-only arm and the Fock-PARF arm already bound them.

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
| 2026-10-01 | **P2.1 launched.** Cell 5b confirmed tag `…cgqk_norc_vplive_xilive_L2probe…`, fresh run, both switches in the banner. Pre-registration frozen in the protocol note §5.6 (move ≤ 154.1 / null 154.1–162.0 / worse > 162.0). That entry also withdraws §5.6's "V_φ grew" reading: V_φ is 9% of F_θ at layer 0 and ~0 at layer 1. |
| 2026-10-01 | **P2.1 scored: 127.73 at step 3,000, a MOVE** (parent 158.09, −19.2%; criterion ≤ 154.1). The gap widened at every eval (−5.6% at step 500 → −19.2% at step 3,000). It is below no-exchange (140.61, −9.2%), `attention` (131.20) and `rglive` (133.74), with no reverse channel at all: at 3k the measured Fock price is entirely starvation. Clip-hits 2/60 vs the parent's 17/60. Scored in the protocol note §5.6, where every Fock-price and conservativity-price statement is suspended. Log filed under `results/…cgqk_norc_vplive_xilive_L2probe…_noattn/`. |
| 2026-10-01 | **P2.1 extended to the full run (F3.1)** at the author's call, ahead of P2.2: it continues from `_step3000_probe_stop.pt`. Pre-registered in protocol §5.6 at step 3,000: settled point 67, band 60–79. Key line: below no-exchange's 66.98, called even odds. Above 79.1 = transient. P2.2, P2.3 and P2.4 stay queued. |
| 2026-10-01 | **F3.1 at step 15,000: 77.31** (parent 107.30, −27.9%). Below no-exchange (86.57), level with `rglive` (77.46). Settled-ratio projection 59.8–63.3. SCAF CLEAN ×3; clip-hits 0.7%. The ω·dt median rose to 6.41 (no-exchange ended at 3.80), a stiffness watch item. Corrected the P2.1 ξ reading: smaller α = shorter memory; the fast channels became more local. |
| 2026-10-01 | **F3.1 at step 29,500:** best 58.84 (step 29,000), below every L=2 arm at matched steps since 25,500 (`rglive` 62.11, `attention` 65.02, no-exchange 68.82, parent 89.55). Projected settled 57–59, below the pre-registered band on the good side and below `rglive`'s 61.11. The ω·dt median eased to 4.45. SCAF CLEAN ×5. |
| 2026-10-01 | **Live-gradient ladder planned; switches verified on every arm it needs.** `vplive`/`xilive` on `attention`, `attention_potential` and `rglive`: 12 builds, forward bit-identical. New `pair_potential='none'` (`PAIR_POTENTIAL`, tag `nophi`) unblocks protocol runs 10 (multi-ξ SPLM) and 11 (Fock-SPLM); verified, with no change to existing arms. |
| 2026-10-02 | **F3.1 scored: 57.76 settled** (best = final 57.35), −34.3% against its parent (87.93), below no-exchange (66.98, −13.8%), `attention` (63.51) and `rglive` (61.11); 1.160× the matched GPT-2. Pre-registration (point 67, band 60–79) missed on the good side; key line (< 66.98) YES. Clip-hit 0.6%, SCAF CLEAN ×7. **Vφ woke up**: E1 attribution −0.0015 → −0.291, and at layer 1 it now carries most of the step's departure from the Vθ geodesic. "Vφ is inert" withdrawn. Inertia 4× (+4.72 → +19.33). Still MAPS. 6b-10 'results void' banner fixed for no-reverse-channel arms. |
| 2026-10-02 | **F3.1 published as the first Gen 3 model**, `dimitarpg13/semsimula-ladder-live-owt-d384-l2-none-norc`, in the new Gen 3 collection (with its Gen 2 twin as paired control and the matched GPT-2 as shared reference). Before publishing, an independent causality check on the final weights (`scaleup/debug/causality_check_checkpoint.py`; output in the run's results folder): future perturbation at five cuts exactly 0, batch independence exactly 0, prefix-only leak tax −7.7e-8 nats, local val PPL 57.11. |
| 2026-10-02 | **L=4 Fock-PARF live launched** as a 3k probe (tag `…cgqk_vplive_xilive_L4probe…idt2…noattn`). The switches were verified at L=4 from fresh init. Pre-registered in protocol (run 4 section): move ≤ 135.5 against run 4's 138.94; point about 125. Depth is only read against L=2 once P2.2 (L=2 Fock live) has run. |
| 2026-10-02 | **L=4 live probe: 119.77 at step 3,000**, against run 4's 138.94 (−13.8%). MOVE and band HIT (point about 125). The gap widened at every eval; this is the best 3k number in the programme. 0 clip-hits. Extension pre-registered: settled point 57, band 52–63. The depth answer still needs P2.2. |
| 2026-10-03 | **L=4 live scored: 50.10 settled** (best = final 49.48), −30.2% against run 4 (71.75), −13.3% against F3.1; **1.006× the matched GPT-2 (49.81)**, i.e. parity within eval noise, but with 2.3× its parameters (76.8M vs 33.7M) and half its depth. Pre-registration (point 57, band 52–63) missed on the good side. SCAF CLEAN ×7; first in-flight Tier A/B zeros on a joint-bank model. The run-4 L=4 < L=2 inversion does not survive the gradient fix. Depth answer still needs P2.2. |
| 2026-10-03 | **L=4 live published as the second Gen 3 model**, `dimitarpg13/semsimula-ladder-live-owt-d384-l4-none`, first in the Gen 3 collection (order: L=4 Fock live, L=2 conservative-only live, the Gen 2 paired control, matched GPT-2). Run 4, its Gen 2 twin, is not published, so the card names it without linking it. Before publishing, an independent causality check on the final weights (`causality_check_output.txt`) found future perturbation exactly 0 at five cut points, batch independence exactly 0 and a prefix-only leak tax of −1.1e-4 nats; the prefix-only logits differ by up to 0.07 (reduction order, not information). The card follows the F3.1 card's layout and adds a geometry section with the CG1/CG2/CG7 comparison against F3.1 and the CG3 forecastability table. The repo carries the stripped best and step-500 checkpoints, 11 results files and the code snapshot. |
| 2026-10-03 | **v6 abstract-gating runs pre-registered** (protocol §5.10), before any run. G1: a parameter-matched GPT-2, d=512, L=8, untied, 76.9M; point 44, band 40–47; the parity key line is 47.6 (revised the same day, before any run: d=384, L=22, untied, 77.8M; point 45, band 41–48; deferred until G2–G4 are in). G2 = P2.2 → full, L=2 Fock live; point 53, band 50–57; key lines < 57.76 (the register mechanism's value) and > 50.10 (depth). G3 = F3.2, L=2 `attention_potential` everything live; point 51, band 46–56. G4: 6b-10 on run 4. The GPT-2 notebook now writes non-default configs to their own `checkpoints_<tag>`/`results_<tag>` folders. Before this change, a 77M run would have resumed from, and appended to, the published baseline's files. |
| 2026-10-03 | **CB series (PARF–Fock balance) implemented and pre-registered** (protocol §5.11). Switches: `REVERSE_CHANNEL_WARMUP_STEPS` now reaches the tag (`rcw<N>`); before this, CB1 would have resumed from G2's folder. `FOCK_BUDGET` (`fb<ρ>`, a per-token cap on the Fock increment relative to the conservative step) and `FOCK_GATE_L1` (`fg<λ>`, a learned per-token gate with an L1 penalty). Verified bit-identical to G2 at the neutral settings, and G2's default build is bit-identical to the committed model file, gradients included. On Gen 2 no-exchange weights, η (Fock increment / conservative step) is 1.62 at layer 0 and 3.05 at layer 1: the model is Fock-dominated. Key arm CB2b (ρ = 0.3): ≥ 50% of the F3.1→G2 gap recovered, called at 50%. |
| 2026-10-04 | **FO series pre-registered** (protocol §5.13) and **Stage 0 run.**<br>• **Corpus side:** OWT has the long-range dependence the note expected. The corrected C4 tail slope is −0.19 against TinyStories' −0.44, and κ(G) is 1,243 against 1,022. Total predictive information is equal (5.24 vs 5.38 bits), but only 33% of OWT's is available from the previous token, against 58% for TinyStories (post hoc). My contrarian C3–C5 predictions failed.<br>• **Model side:** on the TinyStories models the V_θ force is constant across interior layer steps (‖Δf‖/‖f‖ ≈ 0, ε = 0.000), so inertia is absorbable. That explains the Fock-G1 null mechanistically, despite an inertial share of 0.56–0.92. On the OWT CfC models the force changes by 60–100% per step (ε 0.3–6).<br>• **Confound:** the CfC architecture's stiff low-rank wells. The full 2×2 runs, with ε on SO-TS as the architecture-vs-corpus discriminator. |
| 2026-10-04 | **FO §5.13 amended before any Stage-1 run.** SO-TS and FO-TS train for 16,250 steps (266M tokens, about 7 h each), not 32,500. The comparison is within each corpus, and TinyStories saturates sooner. Settling check: a linear fit over the last 3,000 steps must be within eval noise for both TinyStories arms; if not, extend both equally. FO-OWT stays at 32,500 to match G2. Total about 28 GPU-h serial. |
| 2026-10-04 | **G2 (= P2.2 run in full; L=2 Fock live) scored: 53.12 settled** (best 51.27). The pre-registered point was 53, band 50–57: hit.<br>• **Register mechanism, live gradients:** −8.0% against F3.1 (57.76).<br>• **Depth, live gradients:** L=4 (50.10) is 5.7% better, at 1.9× the per-step cost.<br>• **Against the Gen 2 twin (66.98):** −20.7%.<br>• **Against GPT-2:** 1.067×.<br>• **SCAF:** CLEAN at all audits; Tier A and Tier B 0.<br>• **CB series:** live (G2 < 56.6).<br>Next: the causality check, 6b-9/7/13/8 (and 12), the HF upload as the third Gen 3 model, and the CB0 η reading on G2's checkpoint. |
| 2026-10-04 | **G2 6b readings.**<br>• **V_φ inert: CG1 −0.0002, HIT** (|·| < 0.05). Substitution confirmed at L=2.<br>• **Gate 1:** +32.7%.<br>• **Gate 3:** **+1,274%** at 1.5× (F3.1 +143%, L=4 +216%). This breaks §8.9's momentum–refinement rank order: the register path drives refinement failure on its own.<br>• **ω·Δt p50 3.40:** HIT.<br>• **Forcing:** uniform.<br>• **Inference ablation of the reverse channel:** +300%, against 8.7% trained without (F3.1). Ablation overstates far more under live gradients (4.00× vs 1.087×). |
| 2026-10-04 | **G2 published as the third Gen 3 model**, `dimitarpg13/semsimula-ladder-live-owt-d384-l2-none`, second in the Gen 3 collection (after L=4 live), with its Gen 2 twin (no-exchange, 66.98) as a paired control.<br>• **Independent causality check on the final weights:** future perturbation 0, batch independence 0, prefix-only leak tax +1.5e-4 nats.<br>• **README-only pushes:**<br>&nbsp;&nbsp;– the L=4 card, whose depth line is now answered (5.7% better than L=2 at 1.9× the per-step cost) and whose V_φ reading is now at two depths;<br>&nbsp;&nbsp;– the four Gen 2 arm cards, whose banner now lists the no-exchange arm's live retrain (66.98 → 53.12). |
| 2026-10-04 | **Refinement readiness: L=4 against L=2 Fock live, added to both cards and pre-registered as SR-π** (protocol §5.9).<br>• **Gate 3 at 1.5× the steps:** +216% at L=4, against +1,274% at L=2.<br>• **Trained stiff-mode θ = ω·Δt:** 2.35 at L=4 (θ/sin θ = 3.3; stays below π when refined), against 3.40 at L=2 (−13.3; refinement crosses π and flips the sign of the phase term).<br>• **Not the whole story:** F3.1 crosses π but fails less (+143%), so the register path amplifies.<br>• **Predictions:** an L=8 Fock live arm reads Gate 3 ≤ +100%; SR2 more than halves Gate 3 on G2's configuration; free first, a per-token Gate 3 split by π crossing on G2's checkpoint.<br>• **Pushes:** README-only, to both repos. |
| 2026-10-04 | **Cards brought up to date with G2** (README-only pushes).<br>• **F3.1 card:** the L=4 and L=2 Fock live models added to its Result table. "Not yet known: whether the Fock mechanism adds value" replaced by the answer (8.0%, at the price of an inert V_φ and 9× worse refinement). Gate 3 now carries numbers (+32% for the twin, +143%).<br>• **Four Gen 2 arm cards:** in the banner, the retrained values now link to their Gen 3 models (53.12 → `l2-none`, 57.76 → `l2-none-norc`).<br>• **Held back:** the refinement-terms paragraph sits behind `BOOK_V22_LIVE = False` in `build_cards.py`, so no card cites the book's §8.9 before v22 is on Zenodo. |
| 2026-10-04 | **Definitions on the Gen 3 cards** (pushed to all three). In the geometry section, a **Terms** paragraph: Gate 1 inertia, Gate 2 extension, Gate 3 refinement, FLOW vs MAPS, and why refinement invariance is the precondition for a geodesic reading. A **Glossary** section covering settled PPL, Gen 1/2/3, live vs detached gradients, Vθ, Vφ, ξ, the Fock mechanism / registers / reverse channel, the exchange field, conservative-only, the CfC/BAOAB propagator, ω·Δt and θ/sin θ, geodesic step / R(geo) / share, CG and 6b codes, SCAF Tier A/B, leak tax, clip-hit, and pre-registered. The citation of the book's sections stays held until v22. |
