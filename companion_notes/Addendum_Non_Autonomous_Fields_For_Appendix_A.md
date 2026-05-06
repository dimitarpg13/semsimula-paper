# Addendum: Non-Autonomous Conservative Fields (Reader’s Guide to Appendix A)

> **Canonical copy (this file).** This path is the one indexed from the
> [`semsimula-paper` README](https://github.com/dimitarpg13/semsimula-paper#companion_notes--2026-companion-notes-work-in-progress)
> for readers of the paper. A second copy may exist in the
> [semsimula](https://github.com/dimitarpg13/semsimula) monorepo under `docs/`;
> if the two differ, **this** repository’s file is authoritative for the paper.

### Companion to *Semantic Simulation* — §14 and Appendix A (`app:non-autonomous`)

This note is **not** a substitute for Appendix A of the paper. It is a **short on-ramp**:
it fixes vocabulary, states the logical structure, and points to the objects that matter
when the paper says “non-autonomous conservative” dynamics. Full derivations, citations,
and figures remain in the PDF.

**Canonical LaTeX source** for the printable appendix is
[`paper_v2/sections/A1_non_autonomous_framework.tex`](https://github.com/dimitarpg13/semsimula/blob/main/paper_v2/sections/A1_non_autonomous_framework.tex)
in the [semsimula](https://github.com/dimitarpg13/semsimula) paper build (JMLR-style appendix).
**This** repository holds reproducibility code (`notebooks/`) and extended companion
notes; the addendum is here next to the other §14 / Appendix A reader notes.

---

## Table of Contents

1. [What “non-autonomous” means in this project](#1-what-non-autonomous-means-in-this-project)
2. [The apparent paradox Appendix A resolves](#2-the-apparent-paradox-appendix-a-resolves)
3. [The refined governing equation (Class F)](#3-the-refined-governing-equation-class-f)
4. [Why Classes A–D fail and F survives](#4-why-classes-ad-fail-and-f-survives)
5. [Hopfield energy and the two mechanisms](#5-hopfield-energy-and-the-two-mechanisms)
6. [Integrability and the shared-potential diagnostic](#6-integrability-and-the-shared-potential-diagnostic)
7. [Track A vs. Track B (dominant vs. secondary non-conservativity)](#7-track-a-vs-track-b)
8. [Optional: bundle language and adiabaticity](#8-optional-bundle-language-and-adiabaticity)
9. [Reading order in the main text](#9-reading-order-in-the-main-text)

---

## 1. What “non-autonomous” means in this project

In classical ODEs, *autonomous* often means “the right-hand side does not depend on time
$t$ explicitly.” Here the paper uses a **depth-indexed and context-indexed** family of
dynamics. The “parameters” of the per-layer energy are:

- **$\theta_\ell$** — trainable weights of layer $\ell$ (distinct across layers in a
  standard decoder: $W_Q^{(\ell)}$, $W_K^{(\ell)}$, $W_V^{(\ell)}$, MLP, etc.).
- **$\xi_t$** — a **context** that fixes how token $t$ is coupled to the prefix (in a
  causal model, the effective Hopfield “memory” depends on $h_{\lt t}$; in SPLM, the paper
  realises a concrete pool $\xi$).

A map is **autonomous in layer** if the *same* potential $V(h)$ (or the same
parameter vector $\theta$) governs *every* layer. A **non-autonomous conservative**
system, in the paper’s language, is one where **at each fixed $(\ell,\xi_t)$** the
step is still the gradient of a **scalar** $V(h;\theta_\ell,\xi_t)$ in $h$, but
**no single** scalar potential $V(h)$ **shared across** all $\ell$ and $\xi$ reproduces the full depth
dynamics. That is the sense in which “conservative” and “non-autonomous” are combined.

---

## 2. The apparent paradox Appendix A resolves

The main text of §14 (`sec:conservative-arch`) leaves three facts side by side
that *feel* incompatible until Appendix A is read:

1. **Local conservativity (Jacobian symmetry)** holds broadly — SPLM, a matched
   GPT-2–style model, and **pretrained** GPT-2 all pass a velocity-aware Jacobian
   test at the PCA scale used in the paper: the one-step map looks locally like a
   Hessian of *some* scalar.
2. The **strict shared-potential** fit $V_{\psi}$ — a *single* learned scalar, joint
   across all layers — **splits the architectures**: SPLM high, matched baseline
   intermediate, pretrained GPT-2 low in a characteristic **bathtub** profile
   in middle layers.
3. A **layer-11** anomaly: one middle layer in pretrained GPT-2 recovers
   near-oracle $R^2$ on the shared potential, an outlier the generic story must
   explain.

Appendix A’s thesis: **(1) is not optional mysticism; (2) is not a failure of the
per-layer gradient picture; (3) is a predicted boundary case.** The unifying
object is a **per-layer Hopfield potential** with parameters $(\theta_\ell,\xi_t)$,
not a globally autonomous potential on depth alone.

---

## 3. The refined governing equation (Class F)

Appendix A organises the discussion around a **refined** second-order picture (paper
eq. `eq:refined-eom`):

$$
\mathfrak{m} \ddot{h} = -\nabla_h V\bigl(h; \theta_\ell, \xi_t\bigr) + \Omega_\ell(h;\theta_\ell) \dot{h} - \mathfrak{m} \gamma \dot{h}, \qquad \theta_\ell \in \Theta, \quad \xi_t \in \Xi.
$$

- **$V(h;\theta_\ell,\xi_t)$** — scalar **Hopfield-style** energy for layer $\ell$ at
  token $t$ (the **doubly** non-autonomous part: both $\ell$ and $t$ may enter).
- **$\Omega_\ell(h;\theta_\ell) \dot{h}$** — a **solenoidal** (skew / velocity-coupled)
  correction. In the attention story, this is the $W_Q\neq W_K^\top$ (equivalently
  $K\neq V$) channel **within** a layer. Empirically it is **secondary** relative to
  the between-layer effect (see Track B below).
- **$\gamma$** — Rayleigh damping; **$\mathfrak{m}$** — semantic mass (paper notation).

**Class F** in the paper’s table (`tab:candidates`) is this **non-autonomous
conservative** class with layer-varying $\theta_\ell$ and context $\xi_t$.

---

## 4. Why Classes A–D fail and F survives

The negative experiments **E1–E5** in §14.1 test increasingly rich **Lagrangian
menus** (scalar potentials of several shapes, then linear skew in $h$, then
gyroscopic $B$ constant or affine in $h$, etc.). Appendix A’s diagnosis:

- **Classes A–D** all assume, in the way they are **fit** to data, a **single**
  autonomous structure in **depth** (the same $V$, $\Omega$, or $B$ must serve every
  layer in the fit). Attention trajectories on held-out data **do not** admit
  such a fit — they *tie the null floor*.
- The failure is **not** “attention is not smooth” (see
  [On_The_Smoothness_of_Scaled_Dot_Product_Attention.md](On_The_Smoothness_of_Scaled_Dot_Product_Attention.md)
  in this directory) and not “gradients are impossible,” but
  **structural non-autonomy**: the *right* object is a **family**
  $V(\cdot;\theta_\ell,\xi_t)$, not one $V(h)$ for all $\ell$.

**Class F** is the only row of the table that **drops** the depth-autonomy
assumption. It is the class that **matches** the experiments.

---

## 5. Hopfield energy and the two mechanisms

### 5.1 Per-layer Hopfield potential

For one layer, with fixed context, modern Hopfield network theory gives a **scalar**
(log-sum-exp) energy whose gradient reproduces a key–value-coincident attention step.
The paper writes this as $V_\ell(h)$; its Hessian is **symmetric**, which is the
algebraic backbone of the local Jacobian-symmetry tests.

### 5.2 Mechanism 1 — layer-varying $\theta_\ell$

Different layers have different $K_\ell$ (and thus different **principal
directions** of curvature). A **global** $V_{\psi}$ whose Hessian is supposed to match
*all* per-layer $M_\ell$ **cannot** exist unless those Hessian fields are **aligned
up to scale** along depth. In GPT-2 *middle* layers, they are not — that is the
**bathtub** / middle-band failure of the shared $V_{\psi}$ diagnostic.

### 5.3 Mechanism 2 — context-varying $\xi_t$

Even with $\theta_\ell$ fixed, a **causal** decoder’s effective energy at token $t$
depends on the **prefix** through memory. A diagnostic that uses only
$V_{\psi}(h)$ **averages away** that dependence — the **SPLM** single dip at a layer
in the paper is attributed to this: Mechanism 1 is **off** by construction (tied
weights), Mechanism 2 is **on** until $V$ is **conditioned** on the pool $\xi$.

**Summary:** Mechanism 2 is, in the paper’s program, *curable* by conditioning
(oracle $V_\theta(\xi,h)$); Mechanism 1 is *structural* and needs **architectural**
tying $\theta_\ell\equiv\theta$ (SPLM).

---

## 6. Integrability and the shared-potential diagnostic

Here $V_{\psi}$ denotes the **single** learned scalar used in the strict shared-
potential fit (paper notation). Appendix A states an **integrability** condition: the per-layer linearisations
$M_\ell$ are Hessians of **one** scalar $V_{\psi}$ only if (i) they agree **up to a
per-layer scale** across $\ell$, and (ii) satisfy the usual **third-order** symmetry
that turns a matrix field into a **Hessian of a potential**. SPLM obeys both by
construction; GPT-2 middle layers violate (i) when $K_\ell$ differ structurally.

The **$R^2$** of the shared $V_{\psi}$ fit is an empirical **integrability
certificate** aligned with that logic.

**Layer 11 (GPT-2 small):** at the *final* pre-logit layer, **tied embeddings** make
a **$K=V$** situation in the Hopfield picture, so the **within-layer** skew
$\Omega_\ell$ **vanishes**; the local potential is an honest single scalar, and
the shared fit **recovers** — Appendix A calls this a **boundary case** predicted
by the same formalism.

---

## 7. Track A vs. Track B

Appendix A separates two **sources** of “non-gradient” content:

| Track | Source | What it is | Lead diagnostic in §14 |
|------|--------|------------|---------------------------|
| **A (dominant)** | Between layers / context: $V(h;\theta_\ell,\xi_t)$ | Mechanisms 1 & 2 | Three-way shared $V_{\psi}$ separator, bathtub profiles |
| **B (secondary)** | Within layer: $\Omega_\ell(h) \dot{h}$ from $K\neq V$ | Skew / solenoidal | **Jacobian** asymmetry **gap** vs. SPLM (small) |

**Prescriptive claim of the paper:** Track A is what the **SPLM** architecture is
engineered to remove (Track B is a separate, second-order line of work).

---

## 8. Optional: bundle language and adiabaticity

Appendix A’s later subsections rephrase the same picture as:

- a **connection** on the trivial product bundle (hidden state $\times$ layer index)
  and **holonomy** as non-conservativity when $\theta_\ell$ varies, and
- an **adiabatic** regime when $\lVert\partial_\ell V\rVert$ is small compared to
  $\lVert\nabla^2 V \dot{h}\rVert$ (fast relaxation in the instantaneous well).

You can read the mainline argument **without** that geometry; the bundle section is
a second coordinate system, not a separate hypothesis.

---

## 9. Reading order in the main text

A productive path through the published structure:

1. **§14** — `sec:conservative-arch`: the **separator**, experiments E1–E5, SPLM, and
   the **three-way** $R^2$ comparison.
2. **Theorem** $K\neq V$ and **$\Omega$** in §14 (per-head **attention** field — not
   conflated with the full MLP+LN stack without a separate definition).
3. **This appendix** — `app:non-autonomous` / Appendix A: **Class F**, Hopfield
   potentials, Mechanisms 1 & 2, integrability, layer 11, Tracks A/B.
4. If needed, the **smoothness** addendum
   [On_The_Smoothness_of_Scaled_Dot_Product_Attention.md](On_The_Smoothness_of_Scaled_Dot_Product_Attention.md)
   (Poincaré regularity and what is in scope for $F(h)$).
5. **Conclusion** follow-ups (e.g. conditioning $V_{\psi}$ on context) as signposted
   in the paper.

---

*Version: April 2026. Aligns with `paper_v2/sections/A1_non_autonomous_framework.tex` in the
[semsimula](https://github.com/dimitarpg13/semsimula) paper tree and
main-text §14 cross-references. If Appendix A is revised, update the equation labels
and section pointers here in lockstep.*
