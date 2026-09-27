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

| file | renders as | pp. | |
| --- | --- | --- | --- |
| `07_lagrangian.tex` | **§7** The Lagrangian for Semantic Space | 56–67 |
| `07a_position_dependent_damping.tex` | **§8** Position-Dependent Damping and the Reinforcement Field | 68–74 |
| `18_riemannian_geometry.tex` | **§27** Riemannian Geometry of Hidden-State Space | 337– |
| `18d_geometric_capabilities.tex` | **§37** Capabilities Unique to the Conservative Design | 402– |
| `14_experiments.tex` | **§15** Experimental Validation on GPT-2 and Pythia | 125– |
| `17c_fock_parflm.tex` | (Fock-PARFLM) | — | **next target**: top of the Verlet queue *and* slated for promotion |
| `A1_non_autonomous_framework.tex` | appendix | — |

Numbers not listed are filled in as each section is audited.

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

---

## Progress

| § | plan verdict | audit verdict | status |
| --- | --- | --- | --- |
| A1 non-autonomous framework | read first; *"probably moves into the main text as the formal home of the forcing"* | **stays an appendix** — plan was wrong, see A1 below | **audited 2026-09-27** |
| `07_lagrangian` -> **§7** | reframe | **augment, don't reframe** — the three-force decomposition is already correct | **audited 2026-09-27** |
| `07a_position_dependent_damping` -> **§8** | reframe | **augment + two withdrawals** — see below | **edited 2026-09-27** |
| `18_riemannian_geometry` -> **§27** | retreat, largest | **confirmed — and it was 100% Verlet-era** | **edited 2026-09-27** |
| 13 STP loss | promote | not yet read | pending |
| 17c Fock-PARFLM | promote | not yet read | pending |
| 14 experiments | reframe | not yet read | pending |
| 11, 17h, 18c, 18d, 20 | reframe as decision points allow | not yet read | pending |
| 37 (18d) geometric capabilities | keep, retreat, retitle | plan §3.1 written; not line-audited | pending |
| 1, abstract, 19 | thesis sentence, last | **abstract updated 2026-09-27** (obstruction + geodesics paragraphs); §1 and §19 pending | partial |
| A0 edition history | update | v6 entry written 2026-09-25 | **done** |

---

## `A1_non_autonomous_framework.tex` -> appendix — **audited 2026-09-27**

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

## `07_lagrangian.tex` -> §7, pp. 56–67 — the Lagrangian for semantic space — **audited 2026-09-27, not yet edited**

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

## `07a_position_dependent_damping.tex` -> §8, pp. 68–74 — position-dependent damping — **EDITED 2026-09-27**

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

## `18_riemannian_geometry.tex` -> §27, pp. 337–392 — Riemannian geometry — **EDITED 2026-09-27**

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
