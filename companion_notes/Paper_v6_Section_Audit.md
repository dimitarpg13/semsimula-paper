# Paper v6 — section-by-section audit

> **Status.** Opened **2026-09-27**. A live audit of
> `/Users/dimitargueorguiev/git/ml/semsimula/paper_v6` against
> [`Paper_v5_Restructure_Plan.md`](Paper_v5_Restructure_Plan.md) and the
> CfC+BAOAB findings. The plan's triage table was written *before* most
> sections were read; this file records what reading them actually found,
> and corrects the plan where it was guessing.
>
> **Method.** One section at a time, in the plan's §6 edit order. For each:
> read it, list the specific claims the findings touch, give a verdict, and
> record the edit. A section is not "audited" until specific line-level
> findings are written here.

## 0. This is a book, not a paper — **2026-09-27**

> **Read before using any verdict below.** The author's framing: the work
> began as a paper and at 476 pages has become a **book**, intended for a
> broad scientific audience — physicists curious about language modelling,
> mechanical engineers, chemists working on particle dynamics, scientists
> generally — not only ML researchers. Full statement and its consequences
> in [`Paper_v5_Restructure_Plan.md`](Paper_v5_Restructure_Plan.md) §0.

**What this changes for this audit.** The findings recorded here are about
**claim correctness** — is a statement still true given the CfC+BAOAB
measurements — and they stand unchanged. What they do *not* yet assess is
**placement and accessibility**: whether a correct section is in the right
chapter, at the right depth, for a reader who knows mechanics but not
language modelling.

So each section now needs two verdicts, and only the first has been given
so far:

| verdict | question | status |
| --- | --- | --- |
| **correctness** | does this claim survive the measurements? | being audited, §A1 and §7 done |
| **placement** | is this in the right part, at the right depth, for a non-ML scientist reading linearly? | **not started** — awaits the part structure and front-matter decision |

**Sequencing.** The front matter and `\part` grouping should be settled
before the correctness edits are written, because they determine where the
corrected material lands. The audit can continue in the meantime: finding
what is wrong does not depend on knowing where it will live.

**Scope note — 2026-09-27.** A separate, much shorter **TMLR paper** is
planned once the ladder experiments finish (plan §0). It is **not** in
scope here and this audit should not be shaped around it. The book is the
current object.

---

## File names versus rendered section numbers

**They differ, and the difference has already caused one confusion.** The
`sections/NN*.tex` filenames are historical; `\cref` and the table of
contents use the rendered numbers. This audit uses **`file` -> §rendered**
throughout, and quotes page numbers from the current build.

**Complete map, generated from `main.toc` of the current build
(regenerated 2026-09-30 after the velocity-Verlet sweep and Remark 72: 484 pp.,
0 errors, 0 undefined references or citations). Always
cite the rendered number — it is what the reader sees in the PDF.**

**The rule.** The rendered number is the position in `main.tex`'s
`\input` order (ll. 342–379): each file opens with exactly one `\section`,
counted in input order. The filename prefix plays no part. `\appendix`
(l. 381) switches the counter to letters for the four `A*` files.

| file | renders as | p. |
| --- | --- | ---: |
| `01_introduction.tex` | **§1** Introduction | 21 |
| `02_semantic_space.tex` | **§2** Semantic Space: Metric Structure and Hierarchy | 27 |
| `03_signature_matrix.tex` | **§3** The Signature Matrix of a Semantic Property | 30 |
| `04_gaussian_well.tex` | **§4** The Gaussian Semantic Energy Well | 34 |
| `05_parf.tex` | **§5** Dynamics of Semantic Properties: PARF | 39 |
| `06_sarf.tex` | **§6** Dynamics of Semantic Structures: SARF | 47 |
| `07_lagrangian.tex` | **§7** The Lagrangian for Semantic Space | 57 |
| `07a_position_dependent_damping.tex` | **§8** Position-Dependent Damping and the Reinforcement Field | 69 |
| `08_tree_operations.tex` | **§9** Semantic Tree Operations and Executive Space | 76 |
| `09_expressivity_mcs.tex` | **§10** Expressivity, Mechanism Justification, MCS | 84 |
| `10_jepa_connection.tex` | **§11** Connection to JEPA | 104 |
| `11_hidden_state_interpretation.tex` | **§12** Hidden States as Semantic Properties | 108 |
| `12_semantic_mass.tex` | **§13** Semantic Mass in Transformers | 111 |
| `13_stp_acceleration.tex` | **§14** The STP Loss as Normalized Normal Acceleration | 122 |
| `14_experiments.tex` | **§15** Experimental Validation on GPT-2 and Pythia | 127 |
| `15_conservative_architectures.tex` | **§16** The Prescriptive Test: A Conservative-by-Construction LM | 133 |
| `15a_causal_integrity.tex` | **§17** Causal integrity: leak taxonomy, detection, remediation | 199 |
| `16_hybrid_splm.tex` | **§18** Hybrid SPLM: Helmholtz, Variant A, Variant B | 208 |
| `17_parf_augmented_splm.tex` | **§19** PARF-Augmented SPLM and the Generalised PSP Test | 223 |
| `17c_fock_parflm.tex` | **§20** Fock-Augmented PARFLM: Non-Conservative Virtual Particle Exchange | 247 |
| `17b_cross_architecture_vreg.tex` | **§21** Cross-Architecture Analysis: Vφ Output Regularisation | 274 |
| `17d_structured_scalar_potential.tex` | **§22** Structured Scalar Potential | 278 |
| `17e_scaling_up.tex` | **§23** Scaling Up the SPLM Family | 301 |
| `17f_context_mixing_design_space.tex` | **§24** The Conservative Context-Mixing Design Space | 317 |
| `17g_continuous_learning.tex` | **§25** Continuous Learning | 322 |
| `17h_first_order_sufficiency.tex` | **§26** When First-Order Dynamics Suffices | 327 |
| `18_riemannian_geometry.tex` | **§27** Riemannian Geometry of Hidden-State Space | 339 |
| `20_dynamical_simulator.tex` | **§28** The Direct Dynamical Simulator | 368 |
| `18b_relation_to_energy_based_models.tex` | **§29** Relation to Energy-Based Models | 382 |
| `18e_relation_to_liquid_neural_networks.tex` | **§30** Relation to Liquid Neural Networks | 384 |
| `18i_relation_to_optimizer_inspired_transformers.tex` | **§31** Relation to Optimizer-Inspired Transformers | 389 |
| `18f_relation_to_alphafold.tex` | **§32** Relation to AlphaFold | 394 |
| `18g_langevin_completion.tex` | **§33** The Thermal Langevin Completion | 397 |
| `18j_relation_to_flow_matching.tex` | **§34** Relation to Flow Matching and CNFs | 400 |
| `18h_portable_potentials.tex` | **§35** Portable Learned Potentials | 403 |
| `18c_memorization_capacity.tex` | **§36** Memorization Capacity | 407 |
| `18d_geometric_capabilities.tex` | **§37** Capabilities Unique to the Conservative Design | 411 |
| `19_conclusion.tex` | **§38** Conclusion and Open Questions | 424 |

| `A0_edition_history.tex` | **Appendix A** Edition history | 438 |
| `A1_non_autonomous_framework.tex` | **Appendix B** The non-autonomous conservative framework | 440 |
| `A2_inference_efficiency.tex` | **Appendix C** Inference efficiency: FLOP and parameter counts | 446 |
| `A3_experiment_index.tex` | **Appendix D** Experiment quick-reference index | 462 |

Unnumbered: `00_notation.tex` only (`\section*`, p. 15). The four `A*`
files are **lettered, not unnumbered** — `\appendix` in `main.tex` switches
the counter to letters. (Corrected 2026-09-28; the earlier version of this
table listed them as unnumbered.)

**Note the three traps.** The filename ordering is not the rendered ordering:
`20_dynamical_simulator.tex` renders as **§28**, between `18_riemannian`
(§27) and `18b_energy_based` (§29); and `17b_cross_architecture_vreg.tex`
renders as **§21**, *after* `17c_fock_parflm.tex` at **§20**. And a third:
`A1_non_autonomous_framework.tex` is **Appendix B**, not Appendix A —
`A0_edition_history` takes A. Never infer a section number from a filename.

Regenerate this table from `main.toc` after any change that adds or removes
a section.



---

## Ladder numbers this audit checks against

| arm | settled | what it removes |
| --- | ---: | --- |
| matched GPT-2 L=8 | 49.81 | — |
| L=2 `'attention'` | 63.51 | the transformer |
| L=2 `'none'` | 66.98 | the exchange field |
| L=2 `'attention_potential'` | 80.90 | conservativity (matched capacity) |
| L=2 `'none'`, no reverse channel | 87.93 | the Fock mechanism |

Geometry: R(geo) = 1.09 on the flagship, **0.0003** on the conservative
arm; forcing is UNIFORM per token, not sparse; refinement fails in every
arm and the severity tracks non-conservative content; inertia is worth
+50.5% / +27.2% / +5.2% by how many non-conservative routes exist.

> **Superseded as readings, 2026-10-02.** The table above is the **Gen 2
> (detached-source)** ladder: V_φ and ξ never sent gradient to earlier tokens
> in any of those arms, and `attention_potential` sent none at all. The
> perplexities stand as measurements of that convention. Every reading of a
> gap as a *price* is withdrawn or pending. See the live-gradient re-review
> below.
>
> | Gen 3 (live gradients) | settled |
> | --- | ---: |
> | `attention_potential`, live exchange field only (Gen 2 probe) | 61.11 |
> | conservative-only, live Vφ + ξ (first Gen 3 model) | **57.76** (1.160× GPT-2) |
> | Fock arm live (P2.2), L=4 live, everything-live, `attention` live | not yet run (L=4 live running) |

---

## Live-gradient re-review — **opened 2026-10-02**

**Why.** The gradient-starvation finding
([`Gradient_Starvation_Investigation.md`](Gradient_Starvation_Investigation.md))
changes the reading of every ladder gap. The forward pass is the same, so
the measurements stand; the *interpretations* do not.

**Audit (read-only, 2026-10-02; spot-checked against source).** The Gen 2
ladder numbers appear in only three places. Everything else that is affected
rests on the same detached-ξ/Vφ convention in older, Verlet-era or TinyStories
models.

| where | what rests on a withdrawn reading | class |
| --- | --- | --- |
| **`main.tex` abstract, l.194–267** | "The obstruction, measured"; "every gap prices one component"; 31.3% Fock price; 27.4% conservativity price and "reverses its sign"; "Geodesics, established and **priced**", "cost 31.3%"; "conduit, not stored momentum"; bold "**the price of conservativity is prediction quality; what it buys is dynamical robustness**" | **W** (withdraw) + R (label Gen 2, add Gen 3) |
| **§18 `subsec:geometry-under-cfc`, l.1581–1813** (+ fwd refs l.85–89, 1123–1127) | "Vφ … inert" (l.1621, 1677); "What the geodesics cost" (l.1681–1688); "The same price, measured inside one model" (l.1724); "the smaller one is the honest one" (1.31×, l.1743); "latent rather than absent. It does not" (l.1765); the closing trade-off (l.1810) | **W** + R; the geometry measurements stay, labelled Gen 2 |
| **`A0_edition_history` v6 entry, l.35–72** | "each gap prices one component"; "every result in this edition comes from the Fock-augmented model" (now false: the best result is conservative-only Gen 3) | W + P |
| **§17e scaling-up, l.18–29, 545–565, 1040–1048** | "deficit is representational capacity (not optimisation…)"; "reverse channel load-bearing / necessary"; "price of conservativity made qualitative" | **W** (the "not optimisation" claim) + P |
| §17c Fock-PARFLM (TinyStories, Verlet) l.9, 1010–1020, 1136–1146, 1570 | "single-ξ PPL floor"; "residual gap is the price of conservativity"; the conservativity dial; "conservativity premium" | P: detached-ξ convention, not re-measured |
| §18e l.119–123, 297–299; §20 l.6–10, 928–936; §18d l.908, 937; intro l.180–187 | PPL floors / "expressivity plateau floored by the Obstruction" / "price of geometric structure" / "conservativity premium 1.23 PPL" | P: starvation is a competing cause |
| §17 l.420–430; §17f l.32–45 (+ §15, §16 ξ-from-`h.detach()` mentions) | "conservativity is implemented by a single `.detach()`" | R: the detach also cuts training; a forward stop with a live backward (alias node) keeps the force and causality |
| subtitle (`main.tex` l.130–131) "Conservative Potentials, Non-Conservative Memory" | the framing assumes the Fock memory carries the value | P: revisit after P2.2 |

**Stands:**
- the theorem statement (pure mathematics);
- the GPT-2 numbers;
- the integrator facts;
- the leak history;
- refinement failure (MAPS), which Gen 3 shares;
- the Gen 2 measurements themselves, once labelled.

**Plan, staged by what is known:**

1. **Now (facts known).**
   - Withdraw the W claims and label the Gen 2 numbers.
   - Add the two Gen 3 results.
   - State that the Fock mechanism's value, the depth question and the conservativity price are **open**.
   - Text changes only, no restructuring. Scope: the abstract, the §18 pricing arc, A0, and the §17e "not optimisation" sentence.
2. **After P2.2 (Fock live) and L=4 live.** Rewrite the abstract's experimental half and the §18 arc around what the live ladder actually ranks; revisit the subtitle.
   - **Gating arms (decided 2026-10-03):** L=4 Fock-PARF live (finishing; at step 27,000 it reads 53.11 and projects to about 48–50, near the matched GPT-2's 49.81); **P2.2, L=2 Fock-PARF live** (needed for both the register mechanism's value and the depth question); **F3.2, `attention_potential` everything-live** (the only live-convention price of conservativity, read against `attention` live). Runs 10–11 and `attention` live refine the ladder but do not gate the abstract.
   - **Naming fix for that pass:** the abstract calls the detached-source ladder its "first generation"; the Hub collections and `rem:gen2-gen3` call it **Gen 2** (Gen 1 = explicit integrators). Use Gen 2 / Gen 3 throughout.
   - Any parity claim against the matched GPT-2 must carry its caveats in the same sentence: one seed, an untuned baseline, unmatched depth (L=8 vs L=4).
3. **After the rest of the live ladder** (everything-live, `attention` live, runs 10–11). Restate the ladder as Gen 3 and decide the P items in §17c/§17e/§18e/§20.
   - Each of those either gets a live re-run or a "measured under the detached convention" caveat.
4. **Structural restructuring** (the v6 restructure plan) stays deferred until stage 3. The ranking it would organise the book around is still moving.

**Scope note.** `xi_input = h.detach() if causal_force` (or equivalent) is in every SPLM-family model, not only the ladder:
- `multixi/model_multixi.py`;
- `sarf_variant/model_sarf.py`;
- `parf/model_parf.py`;
- `parf/model_parf_multixi.py`.

So the TinyStories/Verlet "floors" and "premiums" share the convention. That is why they are P, not S.

---

## Progress

| § | plan verdict | audit verdict | status |
| --- | --- | --- | --- |
| A1 non-autonomous framework | read first; *"probably moves into the main text as the formal home of the forcing"* | **stays an appendix** — plan was wrong, see A1 below | **audited 2026-09-27** |
| `07_lagrangian` -> **§7** | reframe | **augment, don't reframe** — the three-force decomposition is already correct | **audited 2026-09-27** |
| `07a_position_dependent_damping` -> **§8** | reframe | **augment + two withdrawals** — see below | **edited 2026-09-27** |
| `18_riemannian_geometry` -> **§27** | retreat, largest | **confirmed — and it was 100% Verlet-era** | **edited 2026-09-27** |
| `13_stp_acceleration` -> **§14** STP loss | promote | not yet read | pending |
| `17c_fock_parflm` -> **§20** Fock-Augmented PARFLM | promote | not yet read | pending |
| `14_experiments` -> **§15** Experimental Validation | reframe | not yet read | pending |
| **§11**, **§26**, **§36**, **§37**, **§28** | reframe as decision points allow | not yet read | pending |
| `09_expressivity_mcs` -> **§10** Expressivity / MCS | (plan: keep, review) | **real gap** — no built model placed; v2.1's register lifecycle runs over layers, not tokens, so it does not realise v2; needs a closing "Where the implemented models sit" subsection | **audited 2026-09-28, corrected from code same day**; Phase 1b pending |
| `18d_geometric_capabilities` -> **§37** | keep, retreat, retitle | plan §3.1 written; not line-audited | pending |
| **§1**, abstract, **§38** conclusion | thesis sentence, last | **abstract updated 2026-09-27** (obstruction + geodesics paragraphs); §1 and §38 pending | partial |
| A0 edition history | update | v6 entry written 2026-09-25 | **done** |

---

## `A1_non_autonomous_framework.tex` -> Appendix B — **audited 2026-09-27**

**Plan said:** *"A1 read; decide whether it moves into the main text (it
probably does, as the formal home of the forcing)."*

**Audit: it does not move, and the plan's reason was a guess made without
reading it.** A1 is about **transformers'** non-autonomy — why attention
fails the shared-$V_\psi$ diagnostic, the two non-autonomy mechanisms
(layer-varying $\theta_\ell$, context-varying $\xi_t$), the integrability
certificate, and the layer-11 anomaly at $R^2 \approx 0.99$. It is
descriptive-side analysis of GPT-2 and Pythia. It is not about
Lagrange--d'Alembert forcing in the prescriptive models, and moving it
into the main text would relocate a transformer post-mortem into the
architecture chapter.

**The formal home of the forcing is §7**, which already has most of it
(below).

**What A1 should gain instead:** a forward reference. Its *Mechanism 2 —
context-varying $\xi_t$* is precisely what makes $V_\theta(\xi, h)$
non-autonomous in the prescriptive model too, and the register bank is a
slowly-varying context of exactly that kind. A1's adiabaticity and
effective-autonomy machinery (§A1.7) is therefore the natural language for
asking when the register coupling can be treated as autonomous. One
subsection, not a relocation.

**Action:** keep as an appendix; add a closing subsection linking
Mechanism 2 to the register bank and the forcing. **Not yet written.**

---

## `07_lagrangian.tex` -> §7, pp. 57–68 — the Lagrangian for semantic space — **audited 2026-09-27, not yet edited**

**Plan said:** *"reframe — Lagrange--d'Alembert with an explicit forcing
term; 'geodesic' becomes 'unforced motion'."*

**Audit: augment, don't reframe.** `\begin{remark}[Terminology:
``conservative'' in this paper]` (`rem:conservative-terminology`) already
decomposes the dynamics into three additive forces and already gets the
hard part right:

1. conservative potential force $-\nabla_h V_\theta$ — "the component the
   term *conservative* refers to";
2. Rayleigh dissipation $-\gamma v$ — "non-conservative in the energy
   sense but a *known, quantified* departure";
3. **"Designed non-conservative generalised force (Fock-PARFLM only): the
   Fock exchange force $Q_i$ ... violates Newton's Third Law, and cannot
   be derived from any scalar."**

The remark then says "conservative by construction" means (1) is enforced
and (3) is absent. **That is already the v6 thesis**, written before any
of it was measured. The section does not need its frame changed; it needs
the measurements it anticipated.

**Four specific gaps:**

- **Lagrange--d'Alembert is never named.** §7.4 uses the Rayleigh
  formalism, which is correct for the velocity-proportional term (2) but
  does not cover term (3): the Fock force is not velocity-proportional and
  has no Rayleigh potential. The correct general statement is
  $\delta \int L dt + \int F \cdot \delta h dt = 0$. One paragraph
  after `eq:lagrange-rayleigh`.
- **Term (3)'s magnitude is now measured and absent here.** It is ~90% of
  the per-layer deflection (E1) and worth **+31.3%** in trained perplexity
  (ladder §5.6). The remark currently reads as though (3) were a small
  designed extra; it is the dominant term.
- **The tangential-damping point is missing.** Term (2) is parallel to the
  velocity and therefore *never bends the path* — it changes speed along an
  unchanged geodesic. Only term (3) has a transverse component, so only (3)
  produces geodesic curvature $\kappa_g = \lVert F_\perp \rVert / \lVert \dot{h} \rVert^2$. This is what separates "heavily damped" from
  "non-geodesic", and it belongs immediately after the three-force list.
- **A fourth force is now known and unlisted:** the LayerNorm projection.
  It is a *constraint*, not a force — SHAKE without RATTLE — and E1
  measures its deflection at $-0.669$ on the conservative arm, where it is
  the *only* non-gradient element. The three-force decomposition should
  become three forces plus one constraint.

**Action:** four inserts, no restructuring. **Not yet written.**

---

---

## `07a_position_dependent_damping.tex` -> §8, pp. 69–75 — position-dependent damping — **EDITED 2026-09-27**

**Plan said:** reframe. **Audit: augment, plus two withdrawals** — one of
which was load-bearing for the section's motivation.

**Withdrawal 1 (§8.5, p. 72).** The "Impact on the Jacobi metric"
paragraph read the gap between the explicit dial ($\gamma = 0.10$) and the
recovered $\gamma_{\text{geo}} \approx 0.9$ as *"precisely the signature of
implicit position-dependent damping through the channels that $\gamma(h)$
would make explicit."* That inference is withdrawn: the residual
diagnostic returns $\gamma_{\text{geo}} = 0.93 / 0.82 / 0.75$ on **known**
trajectories whose true damping is $0.05 / 0.10 / 0.30$, so the recovered
value measures the instrument. Added as `rem:gamma-geo-artefact`, which is
explicit that the motivation for γ(h) survives on its other grounds.

**Withdrawal 2 (§8.3, p. 70).** Eq. (8.4)'s $\gamma_{Q}r_{Q}(h)$ term was
motivated as *"regions with strong reverse-channel injection receive higher
damping to compensate."* Sound for the energy budget, wrong geometrically —
and measurably so. The reverse-channel increment carries no $v$ while the
conservative displacement does, so damping harder **raises** the channel's
share of the step: 0.915 / 0.931 / 0.941 / 0.952 at
$\gamma = 0.10 / 0.20 / 0.30 / 0.50$, the increment itself constant to four
decimals (replayed layer step, toy at the live config). Added as
`rem:gamma-q-does-not-steer`.

**Augmentation: new §8.6 "What position-dependent damping cannot do".**
Friction is tangential *at every position* — $-\gamma(h)\dot{h}$ is
parallel to the velocity whatever $\gamma$ does, so its normal component is
zero. γ(h) is a throttle, not a steering wheel. Only a transverse
non-gradient force produces $\kappa_g$, and in Fock-PARFLM that is the
reverse channel.

**Still open for this section:** whether a limiting subsection belongs in a
chapter that *proposes* the mechanism (author's call), and whether §8.6
should fold into §8.5 once the part structure is settled.

**Build:** 477pp, 0 undefined refs/cites, 0 errors.

---

---

## `18_riemannian_geometry.tex` -> §27, pp. 339–367 — Riemannian geometry — **EDITED 2026-09-27**

**Plan said:** "retreat, largest". **Audit: correct, and the reason is
starker than the plan knew.**

**The headline finding: §27 contained zero mentions of CfC or BAOAB.**
1,608 lines carrying the paper's central geometric claims, entirely
Verlet-era, on an integrator the production models left behind. A reader
arriving at p. 337 had no way to know that.

**Three edits.**

1. **Scope remark at the head of the section**
   (`rem:riemannian-verlet-scope`). States that every measurement below is
   Verlet-era, separates what is unaffected (the Jacobi metric, the
   Christoffel symbols, conformal invariance — mathematics, integrator
   independent) from what is not, and routes a reader who wants only the
   current state directly to the new final subsection.
2. **Result 3 withdrawn** (`rem:residual-cannot-detect`). The section read
   $\gamma_{\text{geo}} \approx 0.93$, "remarkably stable" across trained
   $\gamma$, as proof of a structural constant — "the effective damping is
   not the nominal 0.3 but a larger intrinsic value set by LayerNorm, the
   potential curvature, and the reverse channel". The calibration returns
   0.93 / 0.82 / 0.75 for *known* trajectories whose true damping is
   0.05 / 0.10 / 0.30, so stability across the dial is the artefact
   signature, not evidence. Withdrawn with it: the two-channel
   decomposition of `par:two-damping-channels`, which rested on it.
   **Retained:** Results 1 and 2, which are claims about *where* minima
   fall and survive a constant offset.
3. **New §27.19 "The same question under CfC/BAOAB"**
   (`subsec:geometry-under-cfc`). The retreat and the positive result in
   one place: the replay instrument (no derivatives, no Christoffel
   symbols, no parametrisation); R(geo) = 1.09 on the flagship with the
   reverse channel at −0.90; damping is tangential so it is not the cause;
   UNIFORM per-token forcing closing the "geodesic between events" escape;
   **0.0003** on the conservative arm, with LayerNorm named as a constraint
   and its −0.669 deflection; the 31.3% price; and refinement failure
   scoping everything at the trained step size.

**Fourth edit, and a correction to my own flag** (`rem:gamma-eff-scope`,
Arm 2 rescoped). I first recorded Arm 2's "fully resolved by the
learned-$\gamma$ diagnostic" as resting on the withdrawn machinery. **That
was wrong, and the two must not be conflated:**

| quantity | how it is obtained | status |
| --- | --- | --- |
| $\gamma_{\text{geo}}$ | least-squares fit *inside the Jacobi residual*, minimising over $\gamma_{\text{eval}}$ | **withdrawn** — the residual cannot detect a geodesic at one step per layer |
| $\gamma_{\text{eff}} \approx 0.13$ | empirical T-ratio, $\mathrm{median}(T_{\ell+1}/T_\ell)$ measured directly on held-out trajectories | **stands** — kinematic, independent of the residual |

So the LayerNorm counter-damping finding is real and downstream
quantities should keep using $\gamma_{\text{eff}}$.

**But reading the paragraph properly surfaced a different, real
overclaim.** Its own table reports, at $\gamma_{\text{eff}} = 0.13$:
cos(dmp) 0.645 against cos(und) 0.643, and $R^2$(dmp) **−2.54** against
$R^2$(und) **−2.15**. The text concluded that the family "*is* following
the damped geodesic equation at both the directional and magnitude
levels". It does not follow: at $\gamma_{\text{eff}}$ the damping term is
*negligible*, so damped and undamped coincide — and they coincide at
$R^2 \approx -2.5$, far worse than a constant predictor. **Agreement
between two failing predictors is not compliance.**

Withdrawn: the magnitude-compliance inference. Retained: the
$\gamma_{\text{eff}}$ measurement and its use downstream. Arm 2 now reads
as a *directional* result only and points forward to
`subsec:geometry-under-cfc`, where the magnitude question is answered by
replaying trajectories rather than by fitting an $R^2$ to an observed
acceleration.

**Build:** 481pp, 0 undefined refs/cites, 0 errors.

---

## `09_expressivity_mcs.tex` -> §10, pp. 84–103 — where do the built models sit? — **asked 2026-09-28**

**The chapter names no implemented model.** Zero occurrences of
"Fock-PARFLM", "PARFLM", "SPLM" or "v2.1" in §10. It develops a staircase

$$\mathcal{L}(\text{v0}) \subseteq \mathrm{REG} \subsetneq
  \mathcal{L}(\text{v0+v2}) \subseteq \mathrm{CFL} \subsetneq
  \mathcal{L}(\text{v0+v1.5+v2+v3}) = \mathrm{MCS}$$

and never says which rung the architecture the rest of the book builds
actually occupies. A reader finishing §10 meets trained models from §16
onward with no statement connecting them.

**By component inventory, Fock-PARFLM v2.1 is v0 + v1.5 + v2, with no v3.**

| mechanism | in Fock-PARFLM v2.1? | what implements it |
| --- | --- | --- |
| v0 field | yes | Vθ pointwise + Vφ pairwise potentials |
| v1.5 salient decay | yes | `register_salience_decay`, `register_salience_threshold`, the destruction gate |
| v2 creation / Fock | yes | the M-register pool, creation gates, LIFO stack discipline |
| **v3 execution** | **no** | nothing. No operator-valued particles, no non-abelian composition; grep finds no execution/gauge machinery anywhere in the model code |

So it sits on the **middle rung** — and **no model in this programme reaches
MCS**, because none implements v3. That is a sentence the book should say out
loud, before a referee says it first.

**But the theorem that puts that rung at CFL does not cover the model as
built.** §10's bound argues v0+v2 is a multi-type branching process, and it
is explicit about the hypothesis: *"Each creation event is conditioned only
on the parent's type and the local field configuration around the parent; the
offspring distribution depends on no other particle's state."* Fock-PARFLM's
creation gate is `QKVCreationGate_v21.forward_prefix`: register queries
scored against **keys and values from every token at or before position t**,
per-register key subspaces, per-register temperatures. Creation content is a
function of the whole causal prefix, not of a parent's local field. The
branching-process reduction therefore does not apply.

**SHARPENED 2026-09-28 — the prefix-conditioning worry is a distraction, and
the earlier three options were all downstream of a wrong premise.** They
assumed the model sits on the CFL rung and asked whether the bound survives.
It does not sit on that rung, for a more basic reason.

**§10 defines v2 by UNBOUNDED cardinality. The implementation fixes M.**

> "The cardinality of the active particle set therefore grows during
> inference, in contrast to v0's fixed-cast assumption." (§10,
> `subsec:mech-restate`)

> "each $($ instantiates a new open-bracket particle ... **The state-space
> dimension grows linearly with depth**, lifting from regular to
> deterministic context-free." (§10, F1 falsifier)

Fock-PARFLM v2.1 has `n_registers` = 16 (TinyStories family) or 32 (ladder),
fixed at construction, with slots **recycled** by LIFO discipline, salience
decay and the destruction gate. Nothing grows with depth. A fixed pool at
bounded precision is a bounded register machine — finite state — so the
implemented model sits **below** the middle rung, not on it. It has v2's
*mechanism* (creation, destruction, stack discipline, Fock bookkeeping) with
v2's *defining property* absent.

> **Corrected later the same day — see "CORRECTED 2026-09-28" below.** The
> "bounded register machine, finite state" step assumes the registers carry
> state from token to token. In the prefix-causal lifecycle they do not: the
> bank is rebuilt from the whole prefix at every layer. The first half of the
> paragraph (M is fixed, nothing grows with bracket depth) stands; the
> finite-state conclusion does not.

**The programme's own experimental plan already encodes this without naming
it.** `Augmenting_PARFLM_to_handle_MCS_Languages.md` §Phase 1:

| experiment | architecture | expected |
| --- | --- | --- |
| F1-fock-nostack | FockPARFLM, **M=16**, no stack | extends past D* (**≥ 8–10**) |
| F1-fock-stack | FockPARFLM, **M=16**, LIFO stack | extends further (**≥ 12–15**) |
| F1-attention | matched GPT-2 | **succeeds to arbitrary depth (TC⁰)** |

A collapse depth just under M is the signature of a depth-M stack, i.e. of
bounded memory. And the fourth row expects the transformer to **beat** the
Fock mechanism on the very falsifier meant to demonstrate its expressivity
advantage. Both were written down and neither was read back.

**Why this dissolves the prefix question.** Prefix conditioning does break
the branching-process hypothesis, so the CFG argument genuinely does not
apply — but that only matters for proving an upper bound of CFL, and a
tighter bound already binds: whatever the gate reads, it writes into M slots.
Bounded memory dominates.

> **Withdrawn, 2026-09-28 — see "CORRECTED" below.** The gate writes into M
> slots, but it re-reads the full prefix at every layer, so the M slots are
> not where history is stored. Prefix conditioning is not a distraction; it
> is the mechanism.

**What replaces it is worse for the chapter.** §10 carries two measuring
instruments: the Chomsky ladder for the formalism, whose rungs are defined by
what grows without bound, and circuit complexity (TC⁰/NC¹) for transformers.
The implemented model belongs on the **second** axis. Yet
`subsubsec:framework-vs-transformers` credits the Fock formalism as *"the
structured memory that transformers acquire only via augmentation"* while, two
paragraphs earlier, quoting Deletang et al. that length generalisation
*"requires explicit, structured memory augmentation"*. A fixed 32-slot pool
is the bolt-on that paragraph criticises, not the unbounded stack it credits.
The forward-vs-backward rhetoric is about the formalism; the built model is on
the backward side of that line.

**The honest trilemma.**

1. **The formalism is MCS-capable and the implementation is a bounded
   truncation of it.** Then §10's expressivity claims are about a system
   nobody has built, and the chapter must say so in those words.
2. **The bounded pool is the design and unbounded cardinality was never the
   plan.** Then v2's definition in §10 is wrong as written, the middle rung
   changes, and the MCS claim needs a different route or is withdrawn.
3. **Recycling plus continuous register content recovers effective
   unboundedness at high precision.** This is exactly the unbounded-precision
   escape §10 denies Universal Transformers. It cannot be claimed here and
   denied there.

**Designed and pre-registered 2026-09-28** as Phase 1b of
`Augmenting_PARFLM_to_handle_MCS_Languages.md`: M ∈ {2,4,8,16,32,64} × 3
seeds on v2.1, close-type accuracy at exact stack depth, collapse depth
$D^\ast(M)$, four controls (bag, v0, parameter-matched, tiny transformer).
Stated prior: limb 1 — **revised the same day, before any run, to limb (b)**;
see "CORRECTED 2026-09-28" below. Refutation conditions named. ~2 h Colab
GPU, after L=4.

**The decisive experiment is cheap and already half-built.**
`parf/dyck_data.py` exists and its docstring names this exact purpose. Sweep
**M ∈ {4, 8, 16, 32}** at otherwise fixed configuration and measure the
collapse depth. Under the bounded-pool reading D* scales with M, roughly
linearly. One plot settles which limb of the trilemma the programme is on,
at d=64, L=4 — a few small runs, not an OpenWebText rung.
**Pre-register before running.**

### CORRECTED 2026-09-28 — the register lifecycle runs over layers, not tokens

Read against the code, not the design notes. Three findings; the first two
revise the placement above, the third removes the May Dyck evidence.

**1. The lifecycle clock is depth, not discourse time.** With
`prefix_causal_registers=True` — the default, and the configuration of every
ladder arm — `FockMultiXiPARFLM._fock_layer_step`
(`model_fock_parf_multixi.py`, the `elif prefix_causal:` branch) does, at
**each layer**:

```python
readout, alpha_max = self.creation_gate_qkv.forward_prefix(h, r)
r = blend * r + (1.0 - blend) * readout                  # (B, T, M, d)
salience = salience * decay + alpha_max * (1.0 - decay)  # (B, T, M)
```

`QKVCreationGate_v21.forward_prefix` (`model_fock_parf_v2.py`) scores each
register's query — taken from that register's *previous-layer* state at
position t — against keys from tokens 1…t, and returns a cumulative-softmax
readout of their values. Salience, the active mask and the destruction gate
then step once, also per layer. Consequences:

- At position t there are **at most L creation events** — 2 or 4 on the
  ladder — whatever the number of open brackets before t.
- A `(` creates nothing and a `)` destroys nothing. §10's F1 paragraph
  (`09_expressivity_mcs.tex` ≈ l.1005: "each `(` instantiates a new
  open-bracket particle … each `)` removes the most recent") describes a
  mechanism the model does not have.
- The register pool is **M learned-query attention readouts over the
  prefix, iterated L times** — a causal, per-position relative of a
  Perceiver latent array. "Creation", "destruction" and "salience" are
  accurate names for the layer-wise gating, but they are not a particle
  population evolving along the token sequence.

**2. So the model is not a finite-state truncation either.** The
finite-state argument above needed memory to pass from token to token
through M slots. It does not: every layer re-reads the full prefix, so the
store is the prefix itself, exactly as in attention. M bounds how many
readouts a layer takes, not how much history is retained. Two things follow:

- The trilemma's limb 1 ("bounded truncation") loses its mechanism. What M
  caps is readout width, and there is no reason D\* must track it — small
  transformers carry bounded Dyck depth through layers and position
  information, not through a slot count (Yao et al. 2021).
- The model's natural home is the **circuit-complexity axis beside
  transformers** — the placement the "What replaces it" paragraph above
  already reached by a different route. That paragraph's conclusion stands
  and is now the primary reading, not a consequence of bounded memory.

The staircase question ("which rung?") is therefore mis-posed for the built
model: its rungs are defined by what grows along the input, and nothing in
v2.1 grows along the input. The honest statement is that v2.1 shares v2's
*vocabulary* and none of v2's *dynamics*.

**3. The May 2026 Dyck results ran the leaky lifecycle.** Phase 1 (10 May)
and F2 (23 May) predate the causal-leak fix (23 July,
`Fock-PARFLM_Causal_Leak_Audit_Results.md`). `FockPARFLM_v2` at the time ran
the legacy lifecycle — cross-layer register state taken from the
**last position of the full window** (`_causal_creation_readout`, `r_new =
r_causal_mt[:, :, -1, :]`), which the leak audit's T2 probe certifies as
leaking future tokens backward whenever the reverse channel is on. On Dyck a
future leak can hand the model the closing bracket outright. The leak's size
at d=64 on Dyck was never measured, so the May 49.01% and the "LIFO is the
active ingredient" reading are **not evidence either way**. Phase 1b's list
of May defects did not include this; it now does.

**What changes downstream.**

- Phase 1b (`Augmenting_PARFLM_to_handle_MCS_Languages.md`) is still the
  right experiment and its four limbs still cover the outcomes. Its stated
  prior moves from limb (a) to **limb (b)**, and its "mechanism as built"
  section is rewritten — both revised **before any run**, with the original
  prior kept on the record there.
- The §10 fix below gains a sentence: the built model's creation and
  destruction are layer-wise gating over prefix attention, and the Fock
  vocabulary describes the formalism, not the trained model's dynamics.
- `subsubsec:framework-vs-transformers` and its table
  (`tab:framework-vs-transformers`) cannot contrast the built model with
  attention-based memory, because the built model's memory *is*
  attention-based. The contrast is legitimate for the formalism only.

**Verdict: a real gap, and a substantive one.** Not a placement sentence that
can be dropped in — the honest fix is a short subsection at the end of §10,
"Where the implemented models sit", that states the inventory, states plainly
that no built model has v3 and so none is claimed to reach MCS, **states that
the built model's register lifecycle runs over layers on prefix attention and
so does not realise v2's growth along the input**, names Phase 1b as the
measurement that places it, and names the hypothesis mismatch as open rather
than resolving it by assertion. Queue it behind the §20 pass; it needs
thought, not typing.

---

## Audit log

| date | section | outcome |
| --- | --- | --- |
| 2026-09-27 | abstract | updated: five-arm ladder with the ordering, the +27.4% conservativity sign-flip, UNIFORM forcing, the robustness trade-off. Rebuilt clean, 476pp |
| 2026-09-27 | A1 | audited — stays an appendix; plan corrected |
| 2026-09-27 | §7 | audited — augment not reframe; four specific inserts identified |
| 2026-09-27 | — | **Verlet objective added to the plan (§2a)**: contain and label, do not reduce; three-tier policy; priority queue of 8 files with Verlet and zero CfC |
| 2026-09-27 | §27 (`18`) | **edited** — Verlet scope remark, Result 3 + two-channels withdrawn, new §27.19 under CfC/BAOAB; 480pp clean |
| 2026-09-27 | §27 Arm 2 | **magnitude-compliance inference withdrawn** (`rem:gamma-eff-scope`). Corrects my own flag: γ_eff (T-ratio) is *not* the withdrawn γ_geo machinery and stands; the overclaim was concluding compliance from damped ≈ undamped at R² ≈ −2.5. 481pp clean |
| 2026-09-27 | §8 (`07a`) | **edited** — two withdrawals (γ_geo artefact, γ_Q sign) + new §8.6; 477pp clean |
| 2026-09-27 | — | **framing decision recorded (§0): this is a book.** Every verdict so far is a *correctness* verdict; *placement* verdicts not yet started |
| 2026-09-28 | §10 (`09`) | audited — no built model named; v2.1 = v0+v1.5+v2 by inventory, no v3, so none reaches MCS. Phase 1b (M-sweep) designed and pre-registered |
| 2026-09-28 | §10 (`09`) | **corrected from code**: in the prefix-causal lifecycle the registers are rebuilt from the prefix at every layer — creation/destruction run over L, not over tokens. Withdraws the "bounded register machine, finite state" step; places the built model on the circuit axis beside attention. May Dyck runs flagged as pre-leak-fix. Phase 1b prior moved (a) → (b) before any run |
| 2026-09-30 | all (integrator names) | **velocity-Verlet sweep.** All 40 case-insensitive occurrences audited in context (inventory in the restructure plan §2a). **30 fixed** across 13 files: 22 renamed to damped Störmer–Verlet; 8 renamed with a real correction — §27 `18` l.1369 (the Störmer-era model has **no** explicit velocity stream), §8 `07a` ("every SPLM-family integrator" was wrong; caption's v is the δ proxy), notation Δt row, §11 `10`, §21 `17b`, §20 `17c` l.1016. **10 kept** as genuinely velocity-Verlet: the symplectic SPLM variant (§16 ×4), classical MD (§32 ×3), a generic integrator list (§23), and **BAOAB's deterministic skeleton** (§28 ×2). Also §19.8 and Theorem 71 (+ new Remark 72, scope to the explicit-integrator era). Rebuilt clean: 484 pp., 0 errors, 0 undefined refs. Every section moved +1 page (notation grew); map regenerated. Remark 72 shifted the shared theorem counter: Conservative Obstruction is now **Theorem 75, p. 249**. Flagged, not changed: `eq:helmholtz-update` and `eq:gamma-h-verlet` are exact only at Δt = 1 |
| 2026-10-02 | all | **live-gradient re-review opened.** Read-only audit: Gen 2 ladder numbers live only in the abstract (`main.tex` l.194–267), §18 `subsec:geometry-under-cfc` and A0; withdrawn readings (31.3% Fock price, 27.4% conservativity price, "Vφ inert", "geodesics priced", "prediction quality vs robustness", §17e "not optimisation") itemised; older detached-ξ "floors/premiums" (§17c/§17e/§18e/§20/intro) marked pending. Four-stage plan; restructuring deferred until the live ladder settles. No book edits yet |
| 2026-10-02 | abstract, §18 (CfC geometry), A0, §17e | **live-gradient re-review, stage 1 done (text-only).**
  - New `rem:gen2-gen3` at the head of §18 `subsec:geometry-under-cfc` explains the detached-source (Gen 2) vs live-gradient (Gen 3) convention once, gives 61.11 and 57.76, and states the Fock value, the depth question and any conservativity price as **open**.
  - New `par:gen3-geometry`: the Gen 3 conservative arm is a damped geodesic step of Vθ + Vφ, with Vφ attribution −0.291 and R(geo) 0.665.
  - Withdrawn: the 31.3% and 27.4% price readings, "reverses its sign", "priced" geodesics, "Vφ inert" (now attributed to starvation), "conduit, not stored momentum", the prediction-quality-for-robustness trade-off, "each gap prices one component", and §17e's "not optimisation".
  - Gen 2 numbers kept and labelled. The abstract's code paragraph now lists the three generation collections.
  - Damping correction (same day): §7a, §18 and the abstract summary.
  - Rebuilt clean: **487 pp.**, 0 undefined references. The section map above predates both edits: pages after §7a shifted by up to 3, and Theorem 75's page needs re-reading.
  - Stages 2–4 unchanged |
| 2026-10-02 | §17d, §20, §17e | **joint coupling and the exact low-rank integrator added, with full math.** Neither had been in the book; §17d even called the additive banks "the production form".
  - **§17d:** new subsubsection `sssec:ssp-gw-joint`.
    - Props 79–82: additive banks are separable (zero cross-channel Hessian, zero mixed second difference); the analyticity hinge (closed-form ∇V and ∇²V for any ξ-only parameter map); joint contains additive (exact for unnormalised weights; the softmax caveat stated); product-of-experts closed form with stiffness bounds.
    - Also: the coupling ladder A–E, the channel-Hessian figure, and the rank P = Kr vs HKr argument.
  - **§20:** new `par:cfp-lowrank`, Props 97–100: the frozen-occupancy split (PSD by construction, vs the indefinite Hessian); the exact low-rank mode flow; no ω·dt wall (leapfrog stable iff ωΔt < 2, exact map energy-preserving); modes from a P×P Gram eigensolve.
    - Also: the cost argument (P = 32 at the batched-solver limit vs 160) and the evidence (5.74 vs 112 s/step; 1.57× calmer gradients).
  - **§17e:** the envelope table now points to it.
  - Historical fact recorded in the coupling note §11: additive + low-rank was attempted first and was impractically slow.
  - Rebuilt clean: **492 pp.**; Theorem 75 is now on p. 251 |
| 2026-10-02 | §8 (`07a`) | **position-dependent damping under the exact propagator: new §8.8 `ssec:gamma-h-exact`** (replaces the August forward-looking paragraph, which was written before CfC existed and described the blended map, not the built integrator).
  - Props 38–43 / Cor 42:
    - exact O-step, with the explicit factor's velocity excess e^x/(1+x) = 1 + x²/2 (+6.6% velocity and +13.6% KE at L=2);
    - Maxwell / Gibbs invariance for any γ(h);
    - the dissipation budget exp(−∫γ dt) is depth-invariant at fixed T (so no dissipation reading of L=4 vs L=2);
    - Lie brackets [B,O] and [A,O], whose only γ(h) term is −(v·∇γ)v.
  - **Finding:** the deployed A·B·O·A order is formally first-order in the force–friction coupling; the palindrome A·O½·B·O½·A is second-order at no extra force evaluation.
  - Smoothness of γ(h) becomes an accuracy preference, not a stability requirement.
  - Mode-resolved friction Γ = γ₀I + U diag(β)Uᵀ in closed form, with per-mode damping ratio ζ. Deployed stiff modes sit at ζ ≈ 0.05. A constant-ζ family Γ = 2ζ√(L/m) is proposed.
  - The control signals are already computed. Four pre-registered predictions.
  - Rebuilt clean: **496 pp.**; the Conservative Obstruction Theorem is now **Theorem 81, p. 255** |
| 2026-10-02 | §18, App. A3 | **replay instrument and forecastability test added: new §18 subsection `subsec:replay-forecast`.**
  - Naming: CG1–CG7 ("CfC geometry"), because the book already uses E1–E10 and F1–F6. A mapping table gives the cells and the companion-note labels.
  - CG1: the replay arms as Φ^S; R_h and R_v.
    - Prop: the shares don't add; the defect is the φ×LN interaction I, so the attribution depends on order (Shapley = s + I/2).
    - Prop: LN moves positions, not velocities (exact without a reverse channel; confirmed digit for digit).
    - Table across six arms, Gen 2 and Gen 3, with I. attention_potential's positive Vφ share is an interaction (I = −0.30).
  - CG3: three metrics; Prop on the nulls (stay-put 1; isotropic coherence mean 0, variance 1/d; √2).
    - Prop (sphere obstruction at L=2): cos(s₁, ĥ₁) = −sin(θ/2), and a radial s₀ forecasts h₁ exactly. It predicts −0.55 vs −0.566 measured.
    - Results inconclusive at L=2 in both generations.
  - App. A3: new CfC/BAOAB ladder table (Gen 2 and Gen 3, settled values, status) and CG table; the overloaded-code row for the note labels E1–E5/F1.
  - Protocol: CG3 on L=4 live pre-registered (ℓ ≥ 2 only; tangential coherence; paired with run 4; verdict rule).
  - Note `Geodesic_Experiments_with_CfC_BAOAB.md` §4.8 damping claim corrected.
  - Rebuilt clean: **501 pp.** |
| 2026-10-02 | §15a (`15a`), §37.3 (`18d`), §23.10 (`17e`) | **SCAF geometric leak kit: status corrected, results recorded.**
  - §18d said the three-tier kit "is realised" in SCAF. Tiers A and B are implemented; the geodesic-distance Tier C is designed only. SCAF's own "Tier C" is the `StiffnessProbe` (predicts integration instability, not leaks): naming collision now flagged. New status paragraph says what is built, measured and designed.
  - §15a: probe battery now names the Tier A/B diagnostics; "continuous monitoring" rewritten (full battery, not "a cheap AILE proxy") with the ladder's audit record; validation paragraph adds the independent final-weights check.
  - **Bug found:** Tier B silently failed on every joint-bank (`vtjoint`) checkpoint (adapter sliced one channel's d columns for a bank reading all H·d; monitor's `except: pass` hid it). Fixed in `semsimula-scaf` (`ae9094e`, pushed, +3 tests); monitor now records swallowed probe errors (uncommitted); ladder notebook audit print shows Tier A/B with 'absent' ≠ 0 (uncommitted).
  - **Census** on six archived ladder checkpoints: Tier A 0.0 and Tier B 0.0 at every layer (`results/gradient_starvation/scaf_geometric_tiers_census.txt`).
  - Open: the acceptance test's positive half (detect the known pre-fix leak) has never run; the leaky depthcond checkpoints are archived locally.
  - §17e stiffness-audit paragraph got `par:su-stiffness-audit`. Rebuilt clean: **502 pp.**; Theorem 81 still p. 255 |
| 2026-10-03 | §8 (`07a`), App. A3 | **new §8.9 `ssec:settling-refinement`.** Theory of refinement and extension invariance, three repair routes and their causal links.
  - **Prop 44, phase-sampled dissipation:** the split A·O·A step's dissipation carries a phase-dependent term amplified by θ/sin θ, proportional to γ/ω, so adding damping within the split does not remove it. It aliases at ωΔt > π, as in the L=2 conservative-only arm.
  - **Prop 45, exact damped-mode flow:** closed form for all damping regimes, refinement-invariant by the group property.
  - Settling argument (inertial fraction 1.41 at L=4); causal links (1a)⇒(3), (2)⇒(3), (1b)⇒extension in part, (3)⇏extension; the common cause.
  - The four-arm Gate 1 / Gate 3 rank-order table, and the SR1–SR4 experiment table with a pointer to the pre-registration.
  - Corrects my own earlier claim that critical damping removes phase sensitivity: only the joint exact flow does.
  - A3: SR table. Protocol §5.9: pre-registration and decision rule. Checklist: SR agenda.
  - Rebuilt clean: **505 pp.**; Conservative Obstruction is now **Theorem 83, p. 258** |

### 2026-10-03 — geometric capabilities: what they require of the dynamics

- **New book §18d subsection `subsec:geom-requirements`** ("What the capabilities require of the dynamics"; §37.6, p. 444).
  - It names three properties: (C) a conservative step, (R) refinement invariance, (S) settling. A table shows which capability needs which, and where the OWT models stand.
  - The Gen 2 Fock arm has η = 1.6/3.0 on (C). Every arm fails Gate 3 on (R). The L=4 live arm costs +41%/+435% on Gate 2 on (S).
  - The tension: momentum and the register path both earn perplexity and break the properties.
  - The deciding experiments are SR1–SR4, CB1–CB3, and a new reading, CG8.
- **Pointers added:** §8.9 (closing paragraph "Why this matters beyond depth"), §18d intro and §18d Summary.
- **CG table:** a CG8 row.
- **A3 index:**
  - CG8 added, and a CB1–CB3 table;
  - a new overloaded-codes row: G1–G4 means the §18d experiments in the book and the abstract-gating runs in the protocol; the book will call the latter AG1–AG4.
- **Flagged for stage 2:**
  - Experiment G2's premise in §18d (a small, calibrated reverse-channel non-conservatism, tanh = −0.227) is a TinyStories measurement. The bridge now says it must be re-measured on the OWT models.
  - The §18d intro still opens with the TinyStories 9.04-PPL framing.
- **SR5, thermal training (2026-10-03).**
  - §8.9: the SR table gains SR5 ("SR1–SR5"), plus a new paragraph, "Temperature: a training-time route, not a settling one". It covers the FDT stationary kinetic energy, 1/γ > T, the micro-state randomisation mechanism and the F3.1 calibration. It records the median speed rising 6.7 → 15.0 across the two layers, and the annealed variant as conditional on SR4b.
  - §18d `subsec:geom-requirements` and A3 updated to SR1–SR5.
  - The book builds at 510 pages, with no undefined references.
- **Margin sweep before the Zenodo upload (2026-10-03).** Every line extending past the right margin was measured on the PDF itself, and all of them at ≥ 6 pt are fixed, 27 places in total. Two had run off the page, at p. 352 and p. 496.
  - **Fixes:**
    - `xurl`, so long paths break;
    - shortened run-in titles for Remark 37, Proposition 39 and Remark 102, with the rest of each title moved into the body;
    - ragged-right `p{}` columns or narrower column gaps in about 10 tables;
    - one display equation split onto two lines (`eq:ssp-poe`);
    - inline math split at commas;
    - `\path` in place of `\texttt` for file names;
    - three short rewordings (§19 opening, Acknowledgments, the HiPPO list);
    - the §15 step figure scaled to 1.0×, not 1.1×, the text width.
  - **Result:** the build is 512 pages, with no undefined references. 16 log warnings remain, all < 6 pt and not visible.
- **Update (2026-10-04, later): definitions are live.** The gate definitions (Terms) and a Glossary are now on the three Gen 3 cards. Only the sentence citing the book's sections is held, behind `BOOK_V22_LIVE = False` in `build_cards.py`. After v22, the steps below reduce to: verify the numbers, set the flag to True, rebuild, push.
- **Pending, right after the Zenodo v22 upload (2026-10-04).** The three Gen 3 HF cards now carry a shared **Terms** paragraph in their geometry sections, built locally but not pushed. It defines Gate 1 (inertia), Gate 2 (extension), Gate 3 (refinement) and FLOW vs MAPS, and points to the book: §8.9 *Settling, refinement and depth extension* (Props 44–45), Remark 62 *Refinement invariance is a testable precondition*, and §27.15. Its link is the concept DOI 10.5281/zenodo.19712427, which resolves to the latest version, and v21 has none of this. So, after the v22 upload:
  1. verify §8.9, Prop 44, Prop 45, Remark 62 and §27.15 against the final PDF (update `GATES_TERMS` in `build_cards.py` if anything moved);
  2. rebuild;
  3. push the three Gen 3 READMEs (`l4-none`, `l2-none`, `l2-none-norc`) README-only.
- **§10.5.2 replaced (2026-10-04)** with the author's revision (`semsimula/docs/Sec_10_5_2_Fock_revised.tex`), in `09_expressivity_mcs.tex`. The new text has a single-particle space of overlapping Gaussian modes tied to the well (Def 13), the Gram matrix and induced distance, consistent (anti-)commutation relations and statistics (bosonic adopted, with three reasons), an inverse-Gram number operator and field (Löwdin equivalent), and a reworded Doi–Peliti paragraph.
  - **Adapted on integration:**
    - kept `subsubsec:apparatus-v2` (cited 11 times) and `eq:fock`;
    - new equation labels prefixed `eq:v2-…`;
    - hard-coded numbers turned into `\cref`: `def:well`, `def:semantic-space`, `sec:parf,sec:sarf`, `subsubsec:apparatus-v3`;
    - the book's bibliography keys (`Doi1976SecondQ`, `Peliti1985PathIntegral`);
    - `\semm\upsilon^2` and `\semSpace`, matching Def 13 and §2;
    - a booktabs table;
    - `\mathbb{1}` → `\mathbf{1}` (the blackboard digit renders as a broken glyph).
  - **Result:** the build is 513 pages, with no undefined or multiply defined references and no new margin overflow. The draft in `docs/` is unchanged.
  - **Also flagged:** the two other `\mathbb{1}` in the book (indicator functions, §17f l.217 and §18d l.486) render the same broken glyph.
- **Stage 2, the abstract rewritten (2026-10-05)**, after all the gating arms except G4.
  - **The mechanism-ladder paragraph:** Gen 2 / Gen 3 naming throughout, with the two conventions explained. The Gen 2 ordering and the withdrawn price readings stay. Under Gen 3, every arm improves by 21–34%. At L=2:
    - conservative-only, 57.76;
    - with the Fock register mechanism, 53.12 (+8.0%);
    - with a conservative exchange field on top, 54.21 (no gain as trained; hardened version pre-registered).
  - **Depth:** L=4 reaches 50.10 (a further 5.7%, at 1.9× the per-step cost), 1.006× GPT-2. The parity sentence carries 2.3× the parameters (76.8M against 33.7M; about 19M is the untied head), half the depth, one seed, and an untuned baseline.
  - **Methodological findings:** inference ablation overstates (Gen 3: 4.0× against 1.087×; the field +56% against −2.1%; Gen 2: 3.75–3.91× against 1.31×), and the channels substitute (V_φ −0.291 → −0.0002 / −0.0035).
  - **Geodesics:**
    - exact on conservative-only, at the trained step;
    - the Fock models are forced, with η 1.2–3;
    - refinement fails everywhere, tracks non-conservative content, and L=4 is more refinement-ready (+216% against +1,274%);
    - L=4 forecastability is stated as pending attribution (G4);
    - a pointer to §37.6.
  - **Scope:** d=384, L ∈ {2, 4}, one seed; four open, pre-registered questions.
  - **Unchanged:** the descriptive, prescriptive and expressivity paragraphs.
  - **Also:** the code-availability paragraph's "first generation" became Gen 2 / Gen 3 (all three collection URLs return 200).
  - **Build:** 513 pages, 0 undefined.
  - **When G4 lands:** update the forecastability clause.
- **Abstract revised again (2026-10-05, the author's call): no Gen 2 in the abstract.** The abstract describes only the current design: every model is trained with the loss gradient reaching every force's source tokens.
  - Removed: the Gen 2 ordering, the withdrawn price readings, the "every arm improves by 21–34%" clause, and the Gen 2 ablation parenthetical.
  - Kept as a one-clause pointer in Scope: an earlier convention, which detached the source gradients, and the readings it supported are analysed and withdrawn in `rem:gen2-gen3`. Those readings never reached a paper release or Zenodo (the ladder postdates v5.2), but the detached-source models are public in the Hugging Face Gen 2 collection, so the pointer stays.
  - Forecastability now reads "whose paired control is pending".
  - Build: 513 pages, 0 undefined.
