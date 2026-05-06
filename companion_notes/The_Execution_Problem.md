# The Execution Problem

> **Rendering note.** This document contains LaTeX math (inline `$...$` and display `$$...$$` blocks, with macros such as `\mathfrak{...}`, `\boldsymbol{...}`, `\mathcal{...}`, etc.). The math has been verified to render correctly in **Safari**. In **Chrome** some symbols — notably calligraphic and fraktur letters, e.g. `\mathfrak{C}` rendering as a plain `C` instead of $\mathfrak{C}$ — appear to render incorrectly. **Firefox** has not been tested. If symbols look wrong, please view the document in Safari or consult the main paper's PDF, where the same symbols are typeset by LaTeX directly.

## The Proposal

_Static semantic meaning_ (a structure at rest relative to an observer) can be computationally executed to produce new semantic structures. This is essentially asking: How does _meaning_ become _action_? How does _understanding_ become _computation_?

The framework so far (from [Execution_Of_Semantic_Structures.docx](https://github.com/dimitarpg13/semsimula-paper/blob/main/manuscripts/Execution_Of_Semantic_Structures.docx)):
```
Semantic Space (𝚺)                    Execution Space (𝜠)
       │                                     │
   Semantic Structure S          →      Target Point τ
       (meaning)                    (with mass 𝔪_τ and signature)
                                            │
                                            ▼
                                   Executive Particles ε
                                   aggregate around τ
                                            │
                                            ▼
                                   Executive Atoms α = (μ, ℑ, 𝔖, 𝔈, δ)
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                         𝔖 (Semantic Transfer)      𝔈 → δ (Action)
                              │                           │
                              ▼                           ▼
                    New Semantic Structure         Executive continuation
                    back in Semantic Space         (feeds next atom)
```

**Note 1**: _**On Capital Fraktur letters**_

To be precise $\mathfrak{I}$ denotes the space of all inference templates while each inference template will be denoted with lower cap fraktur letter e.g. $\mathfrak{i}$, a point in $\mathfrak{I}$. Similarly, $\mathfrak{S}$ denotes the space of the semantic transfer templates and the specific semantic transfer template which is part of the tuple of some e-atom will be denoted with lower cap fraktur letter e.g. $\mathfrak{s} \in \mathfrak{S}$.  Similarly each execution transfer template $\mathfrak{e}$ is a point in the space of the execution templates $\mathfrak{E}$.
For brevity, instead of using the notation $\alpha = \left(\mu, \mathfrak{i}, \mathfrak{s}, \mathfrak{e}, \delta\right); \mathfrak{i} \in \mathfrak{I}, \mathfrak{s} \in \mathfrak{S}, \mathfrak{e} \in \mathfrak{E}$, we will write $\alpha = \left(\mu, \mathfrak{I}, \mathfrak{S}, \mathfrak{E}, \delta\right)$.

**Note 2**: _**Latch in template matching vs execution**_

First, let us anchor the concept. In the *Semantic Templates* working note (cited in the main paper as `Gueorguiev2022Templates`; not included in this repository), the latch $\mu$ appears in the definition of template matching:

> "When the semantic latch $\mu$ associated with 𝔗 is triggered the centroid of 𝔗 is affixed to the point which has triggered the latch."

So μ is fundamentally a gate - a binary precondition that transitions from "waiting" to "triggered." Before μ triggers, the template (or atom) is inert. After μ triggers, the machinery behind it activates. This is the meaning of "firing": the complete sequence μ triggers → ℑ processes → output produced.


1. **The Mapping Problem** $\boldsymbol{\Sigma} \rightarrow \boldsymbol{E}$

How does a semantic structure $S$ determines its target point $\tau$ in Execution Space $\boldsymbol{E}$ ? How does $\tau$ inherits its mass and signature from $S$? 

$\tau$ inherits mass and signature from $S$ but this raises more questions:

* Is this a deterministic mapping?

* Does the same $S$ always execute the same way?

* How does _the context_ (the surrounding semantic structures) influence the mapping?

2. **The Aggregation Problem** : Executive Particle Assembly

Executive particles aggregate around $\tau$ based on attraction. This is analogous to the semantic structure formation but the following questions arise:

* What determines which atoms $\alpha$ are "attracted" to a given target?

* Is there a competition between different possible executions?

* This is where non-determinism enters (equivalence to Non-Deterministic Turing Machine)

3. **The Transfer Templates** $\mathfrak{S}$ and $\mathfrak{E}$

$\mathfrak{S}$ and $\mathfrak{E}$ are critical bridges which are at the heart of the whole **_semantic structure_** $\rightarrow$ **_execution_** $\rightarrow$ **_new semantic structure_**  construct.

* $\mathfrak{S}$ - how does an execution result become a new semantic structure? What determines its position, mass, and signature in Semantic Space?

* $\mathfrak{E}$ - how does one atomic execution feed into the next? This is essentially control flow, but how to model its evolution as a component of a stochastic dynamical system?

4. **The Observer Dependence**

The requirement of stationarity with respect to observer has consequences and brings up additonal questons

* Different observers may see different structures as "stationary"

* Execution may be **frame-dependent** - the same structure might execute differently (or not at all) relative to different observers

* relation to the dynamical simulation - a structure moving through semantic space isn't "ready" to execute until it settles


### Possible RL Formulation

We would like to integrate RL with the evolution of the semantic structures - that is, _RL-based semantic simulation_.

`State`: The semantic structure $S$ plus execution context

`Actions`: Choice of which executive atoms to aggregate

`Reward`: Quality / relevance of resulting semantic structures

`Policy` : $\pi\left(\alpha | \tau, \text{context}\right)$ - probability of aggregating atom $\alpha$ given target $\tau$.

**What can be resolved through learning**:

* **Which atoms aggregate** (the "program" that executes)

* **Transfer template parameters** (how results map back to semantic space)

* **Action selection** (choices of $\delta$ when $\mathfrak{E}$ feeds multiple possibilities)

### Connection with Semantic Templates: Inverse Template Matching

The mapping of $\boldsymbol{\Sigma} \rightarrow \boldsymbol{E}$ and the execution of structures in $\boldsymbol{E}$ can be interpreted as the inverse of semantic template matching.

| Semantic Templates | Execution |
|--|--|
| Pattern in $\boldsymbol{\Sigma}$ $\rightarrow$ triggers template $\rightarrow$ produces inference | Structure in $\boldsymbol{\Sigma}$ $\rightarrow$ maps to $\boldsymbol{E}$ |
| Input: patterns | Input: complete structure |
| Output : new structure | Output : new structure(s) |
| Recognition-driven | Structure-driven |

Working statement: templates and execution are dual operations

* $\underline{Template}: \text{"If I see }X\text{, produce }Y\text{"}$.

* $\underline{Execution}: \text{"Given }X\text{, compute what }Y\text{ should be"}$.

### Execution as a dual operation to Template Matching

#### The Forward Direction : Template Matching 

A semantic template $\mathfrak{t} \in \mathfrak{T}$ performs the following operation:

$M\left(\mathfrak{t}\right)\text{ scans region }\mathfrak{U}\left(\mathfrak{t}\right) \in \boldsymbol{\Sigma} \rightarrow\text{ latch }\mu\text{ triggers }\rightarrow I\left(\mathfrak{t}\right)\text{ produces }S'\text{ in } \boldsymbol{\Sigma}$

The template is an _incomplete_ semantic structure - the missing pieces are filled by template particles which are in a sense stochastic placeholders. Matching succeeds when a real semantic structure $S$ in $\boldsymbol{\Sigma}$ "fills the gaps" sufficiently well (that is, exceeding the matching threshold $\Theta$, as defined via the average trajectory weight $\bar{w}$ in the *Semantic Templates* working note referenced above).  

Thus, the template matching can be viewed as an analysis operation:

$M : \left(\text{incomplete structure in } \mathfrak{T}\right) \times \left(\text{structures in } \boldsymbol{\Sigma}\right) \rightarrow \text{new structure in } \boldsymbol{\Sigma}$

In this sense the template conveys the following meaning: _"I am missing these pieces. If the environment provides them, I can infer something new"_.

#### The Inverse Direction : Execution

In Execution context a complete semantic structure $S$, stationary in $\boldsymbol{\Sigma}$ with respect to the observer, maps to a target point $\tau$ in $\boldsymbol{E}$. Executive particles aggregate around $\tau$, each atom $\alpha$ carrying the tuple $\left(\mu, \mathfrak{I}, \mathfrak{S}, \mathfrak{E}, \delta\right)$.
 Recall, each executive atom carries an inference template $\mathfrak{I}$ - the atom takes the semantic structure as input to $\mathfrak{I}$, which is now guaranteed to match because the input is the complete structure itself. The atom then routes the result back to either $\mathfrak{S}$ (back to $\boldsymbol{\Sigma}$) or $\mathfrak{E}$ (onward in $\boldsymbol{E}$). So execution is a synthesis operation:

 $$E:\left(\text{complete structure in }\boldsymbol{\Sigma}\right)\rightarrow\left(\text{target in }\boldsymbol{E}\right)\rightarrow\text{new structures in }\boldsymbol{\Sigma}$$
 
The executive structure conveys the following meaning: _"I am complete. Here are the consequences of my existence"_.

#### The Precise Sense of _Inverse_

The duality can be stated as follows:

**Template matching** asks: Given fragments observed in $\boldsymbol{\Sigma}$, what complete structure **explains** them?

**Execution** asks: Given a complete structure in $\boldsymbol{\Sigma}$, what new fragments must **follow** from it?

In more formal terms, if we write template matching as a relation:

$$M\left(T,S\right)=S′\text{ meaning: template }T\text{ applied to }S\text{ yields }S′$$

Then execution inverts this: given $S'$, find the sequence of atomic inference templates $\mathfrak{I}\_1, \mathfrak{I}\_2, ...$ such that the composition of their outputs reconstitutes meaningful consequences of $S'$. The tree structure of executive atoms in $\boldsymbol{E}$ precisely organizes this sequential/branching inversion.

#### Three Levels of Inversion

**Level 1** - **Single Atom**: Each executive atom $\alpha$ contains $\mathfrak{I}$ (an inference template). At the atomic level, execution applies a template to a known, complete input rather than searching for a match. The latch $\mu$ in the atom determines which template fires, but unlike in template matching where the latch is triggered by environmental detection, here the latch is triggered by the executive flow (the parent atom's $\delta$ or $\mathfrak{E}$). Matching is guaranteed by construction - the input is the structure itself.

**Level 2** - **Particle Tree**: The n-ary tree of atoms in an executive particle $\epsilon$ organizes the order of inversion. The tree structure encodes which parts of $S$ must be "unfolded" first before deeper consequences can be derived. The significance vectors $\vec{\sigma}$ on the arcs govern how the output of one atom's inference feeds into the next atom's latch. This is the control flow that has no analogue in passive template matching.

**Level 3** - **Executive Structure**: The full aggregation of executive particles around τ represents the complete operational semantics of the structure $S$. This is where the NDTM equivalence enters - the tree of atoms, with branching controlled by $\delta: \mathfrak{E} → \mathbb{N}$, can represent any computable unfolding of $S$.

#### The Deep Asymmetry

While there is a clear duality, there is also an essential asymmetry:

* Template matching is local: $M(\mathfrak{t})$ watches a bounded region $\mathfrak{U}(\mathfrak{t})$ in $\boldsymbol{\Sigma}$. It reacts to what passes through.
  
* Execution is global with respect to $S$: The entire structure $S$ maps to $\tau$, and the executive structure that aggregates around τ must account for all of $S$'s content - its signature, mass, and internal composition.

Furthermore:

* Templates live in $\boldsymbol{T}$ (same dimensionality $L$ as $\boldsymbol{\Sigma}$)

* Executive structures live in $\boldsymbol{E}$ (generally different dimensionality)

This dimensional difference is significant. Template matching operates in a space isomorphic to semantic space because it needs to overlay patterns onto semantic structures. Execution needs a different geometry because it must encode computational structure (branching, sequencing, action selection) that has no natural representation in the geometry of meaning.

#### An Analogy That May Be Useful

Consider a compiler:

| | Template Matching | Execution |
|--|--|--|
| Analogue |	Parser (pattern recognition) |	Evaluator (code execution) |
| Input | Token stream (fragments in $\boldsymbol{\Sigma}$) | AST (complete structure) |
| Process | Match grammar rules (templates) | Walk the tree, execute nodes |
| Output | AST (inferred structure)  | Values / side effects (new structures) |
| Space | Same language ($\boldsymbol{T} \cong \boldsymbol{\Sigma}$ in dimension) | Runtime is different ($\boldsymbol{E} \neq \boldsymbol{\Sigma}$ in dimension) |

The parser and evaluator are precise inverses: parsing goes from surface form to meaning, evaluation goes from meaning to consequences. The Execution framework has this same architecture, but lifted into semantic space.

#### What Needs Formalization

The key missing piece in making this duality rigorous is the mapping S → τ from 𝚺 to 𝜠. The Execution of Semantic Structures document states that τ inherits mass 𝔪τ and signature ssig(τ) from S, but the questions are:

1. **What determines the executive atoms that aggregate around τ?** There is a mention of "select the one with the largest attraction force toward τ" - this attraction must depend on both the signature of τ and the latches μ of the available atoms. Formalizing this attraction law would pin down how the "inverse program" assembles itself.

2. **Is the mapping injective?** Can two different semantic structures S₁ ≠ S₂ map to the same target τ and hence execute identically? If so, execution defines equivalence classes on semantic structures - structures that are "operationally indistinguishable."

3. **How does the 𝔖 transfer template close the loop?** When an atom sends results back to 𝚺 via 𝔖, those new structures can themselves be stationary and hence executable. This creates a potentially recursive chain: execution produces meaning, meaning produces execution. The termination conditions for this recursion are not yet specified.

### The Action $\delta$ in Execution - Connection with Reinforcement Learning

In the standard RL formulation (following Sutton & Barto), the action concept is embedded in the Markov Decision Process (MDP) framework:

**Definition (MDP)**: A tuple $(S, A, P, R, \gamma)$ where:

* $S$ is the set of states

* $A$ is the action space - the set of all possible actions

* $A(s) \subseteq A$ is the set of actions available in state $s$

* $P(s'|s, a)$ is the transition probability: given state $s$ and action $a$, probability of transitioning to $s'$

* $R(s, a, s')$ is the reward

* $\gamma$ is the discount factor

The action $a \in A(s)$ is the agent's choice at state $s$. It has three essential properties:

1. It is selected by the agent (not the environment)

2. It influences the next state (through the transition dynamics $P$)

3. It is drawn from a well-defined set $A(s)$ that depends on the current state

The **policy** $\pi(a|s)$ gives the probability of selecting action $a$ in state $s$. The goal is to find ${\pi}^{*}$ that maximizes expected cumulative reward.

#### Action $\delta$ in Execution Space

From the Exeuction of Semantic Structures document, each executive atom α = (μ, ℑ, 𝔖, 𝔈, δ) contains:

$$\delta: \boldsymbol{E} \rightarrow \mathbb{N}$$

That is, δ is an integer-valued semantic function that takes the output of the executive transfer template 𝔈 and produces a natural number.

Let us make the correspondence precise-

**Establishing the Formal Relationship**

**Step 1: The Execution MDP**

Consider the execution of a semantic structure S that has been mapped to target τ in 𝜠. The executive tree rooted at τ defines a sequential decision process. We can formulate it as an MDP:

**States**. The state at each step of execution is:

$$s_t = \left(\alpha_t,I_t\right)$$

where $\alpha_t$ is the current executive atom being processed and $I_t$ is the semantic input arriving at that atom. $I_t$ is either:

* The semantic structure of the root τ (for the first atom), or
  
* The output of ℑ from a parent atom (for subsequent atoms)

**Actions**. Now here is where $\delta_t$ connects. At atom $\alpha_t$, the sequence is:


$$I_t \rightarrow \Im_t(I_t) \rightarrow \mathfrak{E}_t(\Im_t(I_t)) \rightarrow \delta_t(\mathfrak{E}_t(\mathfrak{I}_t(I_t))) = n \in \mathbb{N}$$


The integer n produced by $\delta_t$ is literally the action. It selects which child atom to activate next in the executive tree. If atom $\alpha_t$ has children $\{\alpha_{t,1}, \alpha_{t,2}, ..., \alpha_{t,k}\}$, then:

$$a_t = \delta_t\left(\mathfrak{E}_t\left(\mathfrak{I}_t\left(I_t\right)\right)\right) \in \{1, 2, \dots, k\}$$

**Mapping RL Concepts to Execution Space**

**State**

In RL: The state s captures everything the agent needs to make a decision at time t.
In Execution: At each step of execution, the relevant information is:
Which atom α is currently active in the executive tree
What semantic input I has arrived at that atom (either from the root structure S, or from a parent atom's inference)
So the execution state is the pair:

$$s_t = \left(\alpha_t, I_t\right)$$

The atom tells you where you are in the executive tree. The semantic input tells you what you're working with.

**Action**

In RL: The action a is a choice made by the agent that determines what happens next.

In Execution: The atom processes its input through a chain of three transformations:

```
Semantic input I
       │
       ▼
   ℑ(I)          ← inference template produces an intermediate result
       │
       ▼
   𝔈(ℑ(I))       ← executive transfer template reshapes it for decision-making
       │
       ▼
   δ(𝔈(ℑ(I))) = n ∈ ℕ    ← action function produces an integer
```


The final integer n is the action. It selects which child atom to activate next.

**Available Actions**

In RL: The set A(s) of actions available in state s. Not every action is possible in every state.
In Execution: Atom α has some number of children in the executive tree, say k children labeled 1 through k. The available actions are:

$$A(\alpha)=\{1,2,...,k\}$$

The action n produced by δ must index a valid child. Furthermore, each child α_n has its own latch μ_n which may reject the input - so the effectively available actions are only those children whose latches accept the current semantic input.

**Transition**

In RL: The transition P(s'|s, a) gives the probability of moving to state s' given current state s and action a.

In Execution: The transition is deterministic. When atom α_t produces action n:

The next active atom becomes the $n$-th child: ${\alpha}_{t+1} = \text{child}_n(\alpha_t)$

The semantic input for the next atom is the inference output: $I_{t+1} = \mathfrak{I}_t(I_t)$

So the next state is fully determined:

$$s_{t+1} = \left(\text{child}_n\left(\alpha_t\right), \mathrm{I}_t\left(I_t\right)\right)$$

No randomness, no probability distribution - the executive tree structure and the inference template together determine the next state completely.

**Reward**

In RL: The reward R(s, a) is the signal that tells the agent how good its choice was.

In Execution: Each atom may transfer results back to Semantic Space 𝚺 through the semantic transfer template 𝔖. The reward measures the quality of these produced structures - whether they are semantically coherent, whether they settle into stable positions in 𝚺, whether they trigger useful further inferences via templates.
This reward is not immediate in the RL sense - it is evaluated after the structures produced by 𝔖 have evolved in Semantic Space and their consequences have played out. This makes execution a delayed reward problem.

**Policy**

In RL: The policy π(a|s) is the strategy - the probability of choosing action a in state $s$. It is what the agent learns.

In Execution: The policy is implicit in the chain ℑ → 𝔈 → δ. There is no separate policy function. Instead, the action emerges from the semantic processing within the atom. Learning therefore does not update a policy table; it updates the parameters of ℑ, 𝔈, and δ so that the actions they collectively produce lead to better outcomes in Semantic Space.

**Step 2: The Key Difference - $\delta$ is Computed, Not Freely Chosen**

In standard RL, the action is chosen by the agent's policy $\pi(a|s)$, which is a free parameter to be optimized. In the Execution Space formalism, $\delta$ is a semantic function - it computes the action from the executive transfer template's output.
This means $\delta$ is closer to a deterministic policy:

$$\delta_t \equiv \pi(a \mid s_t) \text{ where } \pi \text{ is deterministic}$$

But this raises the question: where does the learning happen? In RL, the policy is updated based on experience. In the Execution Space framework, the analogous learning targets are:

1. The inference template $\mathfrak{I}$ itself - how it processes the semantic input

2. The executive transfer template $\mathfrak{T}$ - how it transforms the inference result before δ sees it

3. The mapping $\delta$ - the action function itself

**Step 3: Factoring $\delta$ to Expose the RL Structure**

To make the RL connection tighter, consider factoring δ into two parts:

$$\delta = \delta_{\text{select}} \circ \delta_{\text{eval}}$$

where: - $\delta_{\text{eval}}: \mathfrak{E}(\mathfrak{I}(I)) \rightarrow \mathbb{R}^{k}$ produces a value estimate for each of the $k$ children - $\delta_{\text{sel}} : \mathbb{R}^{k} \rightarrow \mathbb{N}$ selects based on those values (e.g., argmax, or stochastic sampling)
This factorization reveals that $\delta$ can be seen as a Q-function applied locally:

$$\delta_{\text{eval}}(j) \approx Q(s_t, a_t = j) \quad \text{for each child } j$$

$$\delta_{\text{select}} = \arg \max_{j} \delta_{\text{eval}}(j) \quad (\text{greedy policy})$$


**Step 4: The Action Space A(α) at Each Atom**

In RL, the action space can vary by state. For the executive atoms, the action space is naturally defined by the tree structure:

$$A(\alpha) = \{1, 2, \dots, \text{deg}(\alpha)\}$$

where deg(α) is the number of children of α in the executive tree. But since you also have the significance vectors σ⃗ on arcs, these could constrain the effective action space:

$$A_{\text{eff}}(\alpha, I) = \{j \in A(\alpha) : \sigma_j \cdot \text{ssig}(I) > \theta_j\}$$

Only children whose significance vector is compatible with the current semantic input are reachable - analogous to state-dependent action masking in RL.

**Step 5: The Non-Determinism**

You note that executive structures are equivalent to NDTMs. In RL terms, non-determinism enters through:
1. Stochastic δ: If δ samples from a distribution rather than computing deterministically, the execution becomes a stochastic policy. This is precisely a stochastic policy π(a|s).
2. Multiple execution paths: The NDTM accepts if any* branch accepts. The RL analogue would be: execute all branches, and the "value" of the execution is the maximum over branch outcomes.
3. This connects to the tree structure of the executive atom: The n-ary tree of atoms can represent all non-deterministic branches simultaneously, and δ at each atom selects which branches to explore.

**Summary: A Formal Definition**

Drawing from both formalisms, the Action in Execution Space can be defined as:
Definition (Executive Action): Let α = (μ, ℑ, 𝔖, 𝔈, δ) be an executive atom with children {α₁, ..., αₖ} in the executive tree. The executive action at α given semantic input I is:

$$a = \delta(\mathfrak{E}(\mathfrak{I}(I))) \in \{1, \dots, k\} \subseteq \mathbb{N}$$

This is equivalent to the RL action under the identification: - The state s = (α, I) is the atom-input pair - The action space A(s) = {1, ..., deg(α)} indexes children - The transition is deterministic: s' = (α_a, ℑ(I)) - The side-effect 𝔖(ℑ(I)) produces structures in 𝚺

**The learning problem is then***: optimize the components (ℑ, 𝔈, δ) of each atom so that the semantic structures produced by the chain of 𝔖 transfers maximize some measure of semantic quality (the "reward" in the execution MDP).
The critical difference from standard RL is that in the semantic framework, the action is not freely chosen but is computed through a chain of semantic transformations (ℑ → 𝔈 → δ). The learning thus operates on the parameters of these transformations rather than on a policy table or network directly. This is analogous to model-based RL where the agent learns the dynamics model and derives the policy from it, rather than learning the policy directly.


### The Latch in Template Matching vs Execution

#### Template Matching: μ as Stimulus Detector
In template matching, μ is passive and environment-driven:

```
Structures moving through 𝚺
        │
        ▼
    μ watches region 𝔄(𝔗) ──── "Is there something here that looks like my pattern?"
        │
        │ (threshold Θ exceeded by trajectory weight w̄)
        ▼
    μ TRIGGERS ──── template centroid locks to the triggering point
        │
        ▼
    ℑ fires ──── inference produces new structure
```

Here μ is essentially a sensory receptor. It does not choose what comes into its region; it reacts when something sufficiently compatible passes through. The relationship between μ and ℑ is:

* μ determines IF ℑ fires (the gate)
  
* ℑ determines WHAT is produced (the content)
  
Without μ, ℑ would fire indiscriminately on everything in its region. The latch provides selectivity - it ensures that ℑ only processes inputs that are semantically relevant to this template.

#### Execution: μ as Input Validator

In the executive atom α = (μ, ℑ, 𝔖, 𝔈, δ), the situation is fundamentally different. The input doesn't arrive by happenstance - it is directed by the parent atom's executive flow:

```
Parent atom α_parent
        │
        │  δ_parent selects child index n
        │  𝔈_parent packages the executive output
        ▼
    Child atom α_n receives directed input I_n
        │
        ▼
    μ_n examines I_n ──── "Is this input valid for me to process?"
        │
        │ (if validated)
        ▼
    ℑ_n fires on I_n
        │
        ├──→ 𝔖_n transfers result to 𝚺
        │
        └──→ 𝔈_n → δ_n selects next child
```


Here μ is active and flow-driven. The executive flow has already decided to route input to this atom, but μ still must validate that the input is appropriate. This is a crucial distinction:

* In template matching: μ asks "Is there anything for me?" (detection)
  
* In execution: μ asks "Is what was sent to me valid?" (validation)

### The Relationship Between μ and ℑ

Consider what happens inside an atom when input I arrives:

1. μ examines I's semantic signature against the atom's key (μ is also called the "atom key"). This is a compatibility check: does I's signature match what this atom expects?

2. If μ triggers, ℑ processes I. The inference template ℑ transforms I into something new. ℑ can assume its input is compatible because μ has already verified this.

3. If μ does not trigger, the atom is effectively a no-op for this input.

So μ and ℑ have a precondition-action relationship:

$$\text{output}(\alpha, I) = \begin{cases} I(I) & \text{if } \mu(I) = \text{true} \varnothing & \text{if } \mu(I) = \text{false} \end{cases}​$$
 
But there is a subtlety. In template matching, μ and M(𝔗) work together - μ triggers the initial lock, and then M(𝔗) expands its radius to find the optimal match. So μ is a coarse filter and ℑ performs the fine-grained processing.

In execution, this same coarse/fine structure applies:

* μ performs fast, shallow validation: does the signature of I match my key?

* ℑ performs deep processing: given that I is compatible, what can I infer from it?

This decomposition is computationally important - it means most atoms can be rejected cheaply (μ check) without invoking the expensive ℑ.

### The Duality of μ

Now consider the duality between template matching and execution through the lens of μ:

| Aspect | Template Matching μ | Execution μ |
|--|--|--|
| Triggered by | Environment (structures in 𝚺) | Executive flow (parent's δ/𝔈) |
| Direction |	Bottom-up (data → template) | Top-down (goal → atom) |
| Question asked | "Does the world match my pattern?" | "Does the directed input fit my key?" |
| Failure mode | Template stays dormant | Atom is skipped; execution must find alternative |
| Analogy | Perception | Action precondition |

The duality is precise: in template matching, the data seeks the template; in execution, the template seeks the data. μ mediates both directions but with reversed information flow.

### What Does "Firing of template" Mean?

"Firing" of a template is the irreversible transition from potential to actual. More precisely:

1. Before firing: The atom/template exists as a potentiality - it could produce an inference if given appropriate input. Its ℑ, 𝔖, 𝔈, δ are all defined but inert.

2. μ triggers: The potentiality becomes committed. The latch locks. In template matching, the centroid affixes to the triggering point. In execution, the atom accepts the directed input.

3. ℑ processes: The commitment becomes actual. New semantic content is produced. This is the firing proper - the moment at which the atom contributes to the semantic universe.

4. 𝔖/𝔈 transfer: The result propagates. The firing has consequences in 𝚺 (via 𝔖) and in 𝜠 (via 𝔈 → δ).

The reason "firing" is the right word is that it is irreversible within the current execution context - once μ triggers and ℑ processes, the inferred structure exists. It can be modified by subsequent operations but it cannot be un-inferred.

### Reinforcement Learning Connections

The latch μ maps onto several RL concepts depending on the level of analysis:

#### 1. μ as State-Dependent Action Mask

In RL with invalid actions, one defines a mask function:

$$\text{mask}(s) \subseteq A \quad (\text{valid actions in state } s)$$

In execution, μ serves exactly this role but at the receiving end. When parent's δ selects action n (child atom α_n), the child's μ_n validates:

$$\mu_n(I) = \begin{cases} 1 & \text{atom } \alpha_n \text{ can process } I 0 & \text{atom } \alpha_n \text{ rejects } I
\end{cases}$$
 
This creates a two-stage action validation:

* Stage 1: Parent's δ proposes action n

* Stage 2: Child's μ_n validates the proposal

If μ_n rejects, the execution must handle the failure. This is analogous to safe RL where proposed actions must pass constraint checks.

#### 2. μ as Learnable Gating Function

The parameters of μ (what signatures it accepts, how strict it is) can themselves be learned. This is analogous to learning the option initiation set in the Options framework (Sutton, Precup, Singh):

* An Option o = (I_o, π_o, β_o) has an initiation set I_o, a policy π_o, and a termination condition β_o

* an atom α = (μ, ℑ, 𝔖, 𝔈, δ) has μ playing the role of I_o

The RL optimization of μ would balance:

* **Too restrictive μ**: Many valid executions are blocked → low utility (the atom rarely fires)

* **Too permissive μ**: Invalid inputs pass through → ℑ produces poor inferences → low quality

The reward signal for tuning μ would come from the quality of structures produced by the downstream chain of 𝔖 transfers.

#### 3. μ as Feature Detector in the State Representation

In deep RL, the state representation often involves learned feature detectors that determine which parts of the observation are relevant. The latch μ, by examining the semantic signature of the input, is performing feature detection - it identifies whether the relevant features are present in the input.

The RL connection: μ can be seen as the first layer of a hierarchical policy, where:

* μ performs fast eligibility checking (is this atom relevant?)

* ℑ performs slow deep processing (what should be inferred?)

* δ performs action selection (where to go next?)

#### 4. The Asymmetry Worth Noting

There is a fundamental asymmetry in how μ participates in learning across the two contexts:

In **template matching**, μ adapts to the statistics of the environment - which patterns frequently appear in its region. The RL signal comes from whether the inferences triggered by μ prove useful (whether the structures produced by I(𝔗) participate in further inferences or reach stable positions in 𝚺).

In **execution**, μ adapts to the statistics of the executive flow - which inputs are typically directed to it by parent atoms. The RL signal comes from execution success: does the chain of atoms, gated by their respective μ's, ultimately produce meaningful structures in 𝚺?

This is the duality restated in learning terms: template μ learns **what to recognize**; executive μ learns **what to accept**.


## The NDTM-Equivalence Conjecture and the Path to a Formal Proof

Three earlier sections of this document have asserted, in passing, that the execution process in Semantic Space is equivalent to a non-deterministic Turing machine (NDTM):

- **"The Aggregation Problem"** states *"This is where non-determinism enters (equivalence to Non-Deterministic Turing Machine)"*.
- **"Level 3 — Executive Structure"** under *"Three Levels of Inversion"* states *"the tree of atoms, with branching controlled by δ: 𝔈 → ℕ, can represent any computable unfolding of S"*.
- **"Step 5: The Non-Determinism"** repeats the claim while discussing stochastic δ and multi-branch execution.

None of those places offers a proof, or even a precise statement of the conjecture. This section consolidates the claim into a single explicit **conjecture**, records the heuristic arguments for why the conjecture is plausible, identifies the specific gaps in the current formalism that prevent a formal proof, and lays out an action plan for closing those gaps.

### Statement of the Conjecture

> **Conjecture (NDTM-equivalence of Semantic Execution).**
> Let $\mathfrak{T}$ denote the template space of Semantic Simulation and let $\boldsymbol{E}$ denote the execution space, with executive atoms $\alpha = (\mu, \mathfrak{I}, \mathfrak{S}, \mathfrak{E}, \delta)$ defined as in *Execution of Semantic Structures*. Fix a semantic structure $S \in \boldsymbol{\Sigma}$ that is stationary with respect to the observer, together with its target point $\tau \in \boldsymbol{E}$ and the executive tree aggregated around $\tau$. Then the multiset of execution trajectories rooted at $\tau$ — under stochastic $\delta$ (or equivalently, under the non-deterministic selection of applicable atoms by $\mu$) — recognises the same class of languages as a non-deterministic Turing machine, provided that the template calculus underlying $\mathfrak{T}$ is Turing-expressive and that the energy-damping constraints of the Lagrangian do not restrict the admissible computations to a strict sub-class.

The conjecture therefore has two necessary riders:

1. **Template-calculus expressiveness.** The template space must be rich enough to express arbitrary tree-to-tree computable functions. A template calculus that admits only linear left-hand sides, or only first-order pattern matching without variable binding, is **strictly sub-Turing** (orthogonal left-linear term rewriting systems are confluent but not Turing-complete).

2. **Lagrangian admissibility.** The energy-damping constraints of the full-dynamics equation of the main paper and the bound-state arrival rule of *§"Bound-state arrival and the freeze–unfreeze scheme"* must not forcibly terminate every execution in finite time at a bounded energy. If they do, the effective computational class collapses below NDTM (at most decidable, possibly context-sensitive).

### Heuristic Argument: Why the Conjecture is Plausible

The execution process, viewed abstractly, is:

$$
\text{(template match on } S) \to \text{(select operation } \mathcal{O}) \to \text{(apply rewriting)} \to S'
$$

Formally, this is a **tree-rewriting system with side conditions** (the template-match precondition) plus a policy over applicable rules. Two standard results from rewriting theory bracket the expected position in the Chomsky hierarchy:

- **Term Rewriting Systems (TRS).** Unrestricted TRS are Turing-complete. Explicitly: any partial recursive function can be encoded as a TRS on a suitable signature (e.g. via the arithmetisation of Klop, or the Markov-algorithm encoding). Under a stochastic rule-selection policy the system becomes non-deterministic.
- **Graph/Tree Rewriting.** Rewriting systems on labelled trees or graphs with duplication and deletion are also Turing-complete (Courcelle, Plump). Non-determinism arises naturally from multiple matching rules.

**Simulation in both directions:**

- **NDTM → Execution Process.** Encode the NDTM's tape as a path-like subtree of a semantic structure S. Encode the head position as a designated subtree marker. Encode each NDTM transition δ(q, a) = {(q', a', L/R), ...} as a template whose pattern matches the current subtree-at-head and an executive atom whose ℑ rewrites the matched subtree according to (q', a', L/R). Non-determinism is implemented by having multiple templates match simultaneously and letting the δ of the atom (or multiple atoms) branch via the executive tree.
- **Execution Process → NDTM.** If template matching and operation application are themselves computable functions (which they are for any concrete template calculus), then a universal TM can simulate the execution by maintaining S as an explicit data structure, evaluating all templates at each step, and non-deterministically selecting one applicable atom to apply. The number of non-deterministic choices per step is bounded by the number of applicable atoms, which is finite.

Under the two riders (Turing-expressive templates; admissibility of unbounded computations under the Lagrangian), both directions go through, and the equivalence holds.

### Why This is a Conjecture and Not a Theorem

Four distinct gaps prevent a formal proof in the present state of the framework:

1. **The template calculus is not yet formalised.** §2.4 of the main paper gives Definition 2.8 and an explicit deferral statement. Neither this document nor *Execution of Semantic Structures* specifies a grammar for templates, a matching algorithm, or a closure/expressiveness result. Without a precise calculus, questions such as *"can this template language express arbitrary tree-to-tree computable functions?"* cannot be answered.

2. **The operation calculus is not yet formalised.** *Executive Space and Operations* (§8.9 of the main paper) says that operations are *"substitution, insertion, deletion, and more elaborate rewriting rules"* — but the phrase *"more elaborate"* is doing all the heavy lifting. A precise formal signature of operations, an application relation, and a closure claim (*"these operations generate all computable tree-to-tree functions"*) would be required.

3. **No simulation argument has been written.** Both directions above — NDTM → execution, execution → NDTM — are sketched heuristically in this document but have not been carried out as rigorous constructions. Each direction requires choosing explicit encodings, checking that the encoding preserves the relevant structure (tape, head, transitions), and verifying that the simulation faithfully reproduces the input machine's language.

4. **The impact of Lagrangian constraints is not yet characterised.** The execution process is embedded in the full Semantic Simulation dynamics: structures are created with finite energy, lose energy to the damping factor H_i, arrive at bound states, and freeze. These constraints could:
   - enforce termination at a bound state after a finite number of execution steps — reducing the class to decidable;
   - bound the per-step work by the damping schedule — potentially reducing the class to a linear-bounded-automaton level (context-sensitive);
   - admit unbounded executions only in the idealised energy-conservation limit — leaving NDTM equivalence only as an asymptotic statement.
   Determining which of these regimes actually applies requires separately studying the coupling between the executive tree's depth/branching and the Lagrangian's energy budget.

### Action Plan for a Formal Proof

The following steps, in order, constitute a concrete programme of work to turn the conjecture into a theorem.

#### Step 1. Formalise the template calculus $\mathfrak{T}$.

1.1 Fix a finite signature of node labels $\Sigma$ and a set of meta-variables $X$. Define a template as a tree over $\Sigma \cup X$ with optional side-conditions on meta-variable bindings.

1.2 Define template matching: given a template $t$ and a semantic structure $S$, a match is a substitution $\sigma: X \rightarrow (\text{subtrees of } S)$ such that $\sigma(t) \equiv (\text{a specified subtree of } S)$ up to the matching-threshold criterion already used in the matching document.

1.3 Prove (or conjecture and test on examples) a **normal-form theorem** for $\mathfrak{T}$: every template reduces to a canonical form under a specified set of template-space equivalences.

1.4 Establish the expressiveness baseline: show that $\mathfrak{T}$ can express at minimum all **first-order linear patterns**. Then show, as a separate result, that $\mathfrak{T}$ can express **non-linear patterns with variable repetition** — this is the key step that takes the calculus from TRS-style (polynomial-time matching) to Turing-expressive (undecidable matching in general).

#### Step 2. Formalise the operation calculus in the executive space $\mathcal{E}_{\mathrm{exec}}$.

2.1 Fix a signature of tree-rewriting operations: substitution (replace subtree $t$ with $t'$), insertion (add subtree $t$ as the $i$-th child of node $n$), deletion (remove subtree rooted at $n$), and relabeling (change the label of node $n$).

2.2 Prove the **completeness of tree-rewriting**: every tree-to-tree computable function can be expressed as a finite composition of the four primitives above. (This is a standard result in rewriting theory, adapted to the semantic-tree signature.)

2.3 Integrate operations with templates: define an **executive rule** as a pair $(t, \mathrm{op})$ where $t \in \mathfrak{T}$ and $\mathrm{op}$ is a tree-rewriting operation, with semantics "if $t$ matches $S$ with substitution $\sigma$, apply $\mathrm{op}$ to the matched subtree". Show that the set of executive rules equipped with concatenation forms a category whose morphisms are exactly the tree-to-tree computable functions.

#### Step 3. Execute the simulation arguments.

3.1 **NDTM → Executive system.** Fix an NDTM $M = (Q, \Gamma, \delta, q_0, F)$. Construct:
- a semantic signature $\Sigma_M$ for encoding configurations of $M$ as trees;
- a template calculus sub-language that expresses the patterns "head is at position $p$, reads symbol $a$, in state $q$";
- a set of executive rules implementing $\delta$;
- a proof that the non-deterministic execution tree of the resulting executive system, rooted at the initial-configuration tree, accepts exactly $L(M)$.

3.2 **Executive system → NDTM.** Fix an arbitrary executive system $\mathcal{X}$ with a Turing-expressive template calculus. Construct:
- a tape alphabet capable of representing semantic structures as strings;
- NDTM transition rules that implement, for each step: (i) enumeration of applicable templates, (ii) non-deterministic selection of one applicable atom, (iii) application of its operation, (iv) replacement of the current tape content with the rewritten structure;
- a proof that $L(\text{the NDTM}) = L(\text{the executive system})$.

3.3 Combine 3.1 and 3.2 into a single equivalence theorem: the class of languages recognised by executive systems equals the class of recursively enumerable languages (i.e. class RE), under a Turing-expressive template calculus and unconstrained dynamics.

#### Step 4. Characterise the admissible regime under Lagrangian constraints.

4.1 Formalise the energy budget of an executive tree: each executive atom's application contributes a quantum of energy $E(\alpha)$ (the operational cost from the operation-Lagrangian equation of the main paper), and the cumulative energy is bounded by the structure's ensemble net energy $E_t(S)$.

4.2 Prove a **trichotomy theorem** (refining the dichotomy originally envisaged). The admissible computational class depends jointly on the operation calculus and on the presence of a **structural-retirement** mechanism:
   - **(i)** No deletion operation and no structural retirement: class $\le$ context-sensitive (monotone growth only).
   - **(ii)** Explicit deletion-as-operation (Step 2.1) but no structural retirement, closed energy: class $\le$ **LBA / context-sensitive** (bounded workspace; structures are never forgotten so the addressable workspace grows with the computation length).
   - **(iii)** Explicit deletion-as-operation **plus** a structural-retirement mechanism (cf. Step 4.4 below) **plus** reinforcement-replenished energy via $\mathfrak{F}$: class $=$ **RE / NDTM-equivalent**.

4.3 Characterise which regime applies to specific classes of semantic structures (sentences, short discourses, full documents). Empirical hypothesis: short structures are in regime (i)–(ii); longer structures with reinforcement-driven context are in regime (iii).

4.4 Specify a **structural-retirement mechanism** and prove that it is necessary and sufficient for regime (iii) of the trichotomy theorem. Two concrete candidate mechanisms — **energy-threshold retirement** and **reinforcement-driven persistence** — are developed in their own section, *"Structural Retirement Mechanisms"*, below.

#### Step 5. Position the result in the Chomsky hierarchy.

5.1 State the final equivalence as a **locus theorem**: the class of languages recognised by Semantic Execution processes lies between the regular languages (lower bound: trivially, since arbitrary finite-state automata can be encoded as degenerate executive trees) and RE (upper bound: from Step 3.3).

5.2 Identify which sub-classes are natural in the framework:
- **Bounded-depth executive trees** → context-free / context-sensitive (depending on template expressiveness).
- **Unbounded depth with energy damping** → decidable sub-class.
- **Unbounded depth with reinforcement-replenished energy** → RE.
- **Non-deterministic selection among applicable atoms** → places the class strictly above the deterministic analogue at each level, except at the RE boundary where NTM = DTM in language-recognition power (though non-deterministic execution retains meaningful resource-complexity differences, cf. NP vs P).

5.3 Compare to related rewriting systems: pure TRS, combinatory logic, Plotkin-style structural operational semantics, process calculi. Identify which of these the Semantic Execution framework is closest to, and transfer known results (confluence, normalisation, complexity) where applicable.

### Dependency on Other WIP Pieces of the Framework

The NDTM-equivalence programme cannot be pursued in isolation. It depends on concurrent progress in three other active strands:

- **Templates (§2.4 of the main paper).** Step 1 is blocked until the template formalism is promoted from its current minimal definition to a full calculus.
- **Executive / execution spaces (§§8.9–8.10).** Step 2 is blocked until the operation set is formalised beyond *"substitution, insertion, deletion, and more elaborate rewriting rules"*.
- **Reinforcement-update rule (§6.2 scope remark).** Step 4.2's regime (ii) requires specifying how external reinforcement replenishes structural energy; this in turn requires the concrete `𝔉`-update rule that the main paper explicitly defers to a separate manuscript.

The three strands are logically independent but should be worked in rough parallel, since each informs the others.

### Expected Deliverable

The intended end state of this programme is a stand-alone manuscript, *"Computational power of Semantic Execution"*, containing:

- The formal template calculus of Step 1 as its first technical chapter.
- The operation calculus of Step 2 as its second.
- The bidirectional simulation theorems of Step 3 as its main result.
- The dichotomy theorem of Step 4 as a refinement that connects the computational class to the Lagrangian dynamics of the main framework.
- The locus result of Step 5 as the summary statement.

Such a manuscript would consolidate the NDTM-equivalence argument, replacing the scattered informal assertions with a single rigorous proof.

Pending that proof, the NDTM-equivalence claim has the status of a **working conjecture** and should not be used as a load-bearing assumption in downstream formal arguments.


## Structural Retirement Mechanisms

The trichotomy theorem of Step 4.2 makes **structural retirement** — the removal of semantic structures from the active arena after their usefulness has ended — a load-bearing ingredient of the full NDTM-equivalence claim. Deletion-as-operation (Step 2.1 of the action plan) is sufficient for *locally correct* rewriting but is *not* sufficient to push the computational class from LBA / context-sensitive up to RE / NDTM-equivalent. An additional mechanism is needed that clears structures which the execution process is no longer actively rewriting. This section develops two concrete candidate mechanisms, compares them, and specifies an implementation and validation plan for each.

Throughout this section we write $S$ for a semantic structure in $\boldsymbol{\Sigma}$, $t$ for the (continuous or discrete) time parameter of the dynamics, $T(S, t)$ for its kinetic energy, $V(S, t)$ for its potential energy under the Gaussian well, PARF, and SARF contributions, and $\mathfrak{F}(S, t)$ for the reinforcement flux deposited onto $S$ by the semantic energy field.

### Mechanism A. Energy-Threshold Retirement

#### Formulation

Define the **residual energy** of a semantic structure as the sum of its kinetic and potential contributions:

$$
E_{\mathrm{res}}(S, t) = T(S, t) + V(S, t).
$$

Fix a retirement threshold $\epsilon_{\min} > 0$ and a hold window $\tau_{\mathrm{hold}} > 0$. The retirement rule is:

> If $E_{\mathrm{res}}(S, t) < \epsilon_{\min}$ for all $t \in [t_0, t_0 + \tau_{\mathrm{hold}}]$, remove $S$ from $\boldsymbol{\Sigma}$ at time $t_0 + \tau_{\mathrm{hold}}$.

The hold window $\tau_{\mathrm{hold}}$ is essential. It prevents transient dips below threshold (caused by damping, collisional events, or numerical discretisation) from triggering spurious retirements. In practice $\tau_{\mathrm{hold}}$ will be chosen in proportion to the characteristic decay time-scale of the damping factor $H_i$.

#### Coupling to the Existing Lagrangian

This mechanism folds cleanly into the existing dynamics of the paper because it introduces no new state variables. The quantities $T$ and $V$ are already tracked per-structure; the threshold check is a scalar comparison applied post-update at each integration step. Three properties make the coupling natural:

1. **Consistency with bound states.** A structure that arrives at a bound state has $T \to 0$ (kinetic energy fully dissipated by $H_i$) but in general $V \ne 0$ because the Gaussian well has a non-zero floor at the centroid. Thus bound states with $V > \epsilon_{\min}$ persist — they are "interesting" frozen structures that merit continued presence in the semantic arena. Bound states that additionally drain their potential energy (e.g. through SARF-mediated coupling to other structures) drop below threshold and retire.

2. **Consistency with damping.** The damping factor $H_i$ removes kinetic energy only; without a retirement mechanism it would leave the configuration cluttered with zero-velocity but non-zero-potential relics. Energy-threshold retirement is the natural extension of damping to the structure-presence level.

3. **Emergent, not stipulated.** Retirement becomes a **consequence** of the dynamics rather than an independent primitive. This preserves the framework's editorial commitment to deriving mechanisms from the Lagrangian wherever possible.

#### Advantages

- **Minimal theoretical overhead.** No new fields, no new rate constants beyond $\epsilon_{\min}$ and $\tau_{\mathrm{hold}}$, no new rewriting primitives.
- **Physically interpretable.** Analogous to thermal dissociation in chemistry: structures whose binding energy has fallen below the background noise level dissociate.
- **Energy-conservation story is clean.** Retirement releases $E_{\mathrm{res}}(S, t) < \epsilon_{\min}$ back to the field; the total energy budget accounts for this explicitly, providing a convenient hook for verifying conservation laws in simulation.

#### Concerns

- **Threshold calibration is non-trivial.** $\epsilon_{\min}$ must be small enough to preserve meaningful bound states and large enough to retire clutter. Calibration likely depends on the scale $\mathfrak{m}\upsilon^2$ of the Gaussian well.
- **Interaction with numerical damping.** In discrete-time simulations, numerical dissipation from the integrator may artificially drive structures below threshold. The hold window mitigates this, but it must be tuned to the integrator.
- **No memory of usefulness.** A structure that has never been reinforced is treated identically to one that *has* been reinforced but now sits below threshold. If reinforcement is a better proxy for "computational usefulness" than residual energy, Mechanism B may be preferable.

#### Action Plan — Mechanism A

**A.1 Theoretical specification.**
- Fix a precise form of $V(S, t)$ that aggregates the Gaussian-well contribution of $\mathfrak{F}$ with the PARF/SARF pairwise contributions.
- State the retirement rule as a formal sentence of the dynamics, including the hold window and the release of residual energy back to $\mathfrak{F}$.
- Prove a **conservation lemma**: total system energy (field + structures + released retirement quanta) is conserved up to the damping loss, with retirement contributing an identifiable term.

**A.2 Reduction to a rewriting property.**
- Prove a **necessity lemma**: in the closed-energy, deletion-as-operation regime without retirement, the reachable configuration space from any initial $S$ is bounded by a function of the number of operations applied, hence the system is LBA-equivalent.
- Prove a **sufficiency lemma**: with energy-threshold retirement **and** reinforcement-replenished energy via $\mathfrak{F}$, the reachable configuration space is unbounded, hence the system can simulate an NDTM.

**A.3 Implementation in the STP/Lagrangian simulator.**
- Extend the existing simulator used for the STP-loss validation to track per-structure residual energy.
- Add a post-step retirement pass: iterate over active structures, apply the threshold-and-hold rule, remove structures that satisfy it, and record the released energy.
- Expose $\epsilon_{\min}$ and $\tau_{\mathrm{hold}}$ as configuration parameters.

**A.4 Empirical validation.**
- **(Sanity)** In a two-structure PARF demonstration, verify that structures with $V > \epsilon_{\min}$ at their bound state persist indefinitely, while structures driven below threshold (e.g. by SARF-mediated coupling) retire.
- **(Sensitivity)** Sweep $\epsilon_{\min}$ and $\tau_{\mathrm{hold}}$ across two orders of magnitude each; plot the retirement rate as a function of both.
- **(Consistency)** Verify the conservation lemma numerically: track total energy across retirements and confirm it is conserved to within the integrator's tolerance.
- **(Behavioural)** On a small executive-tree toy problem (3–5 atoms, 2–3 retirement events expected), confirm that the computation completes correctly and that the retired structures are indeed those that would have been "garbage" in a hand-analysed reference execution.

**A.5 Documentation deliverable.**
- A companion note *"Energy-Threshold Retirement in Semantic Simulation"* containing: the formal rule, the two lemmas, the calibration results, and the toy-executive-tree validation.

### Mechanism B. Reinforcement-Driven Persistence

#### Formulation

Attach to each active semantic structure $S$ a scalar **persistence variable** $\pi(S, t) \in \mathbb{R}_{\ge 0}$ with the following dynamics:

$$
\frac{d\pi(S, t)}{dt} = -\lambda \pi(S, t) + \rho\bigl(\mathfrak{F}(S, t)\bigr),
$$

where:
- $\lambda > 0$ is a decay rate, specifying how quickly persistence leaks away in the absence of reinforcement.
- $\rho: \mathbb{R} \to \mathbb{R}_{\ge 0}$ is a monotone-increasing rate function converting the local reinforcement flux $\mathfrak{F}(S, t)$ into a persistence-replenishment rate. A natural choice is $\rho(x) = \kappa \max(x, 0)$ with $\kappa > 0$.

Fix a persistence threshold $\pi_{\min} > 0$. The retirement rule is:

> If $\pi(S, t) < \pi_{\min}$, remove $S$ from $\boldsymbol{\Sigma}$ at time $t$.

Unlike Mechanism A, there is no hold window: the low-pass-filter dynamics of $\pi$ itself already smooth out transients, so a bare threshold is sufficient.

#### Coupling to the Deferred $\mathfrak{F}$-Update Rule

The main paper defers the specification of the $\mathfrak{F}$-update rule to a separate manuscript (cf. the "Scope: two senses of reinforcement" paragraph in §6.2 of the paper). Mechanism B couples retirement directly to that rule: a structure survives iff $\mathfrak{F}$ continues to deposit reinforcement on it. Three properties make the coupling natural:

1. **Symmetry with reinforcement.** Reinforcement creates and reshapes $\mathfrak{F}$; reinforcement-driven persistence retires structures that $\mathfrak{F}$ has stopped reinforcing. The two sides of the reinforcement loop are made symmetric: $\mathfrak{F}$ both builds up and dissolves structures depending on whether it continues to interact with them.

2. **RL-interpretable.** In the RL view of §6 of the paper, $\mathfrak{F}$ is the externally shaped reward-like field. Reinforcement-driven persistence makes structural lifetime a direct function of cumulative reinforcement received. This is the closest analogue to eligibility traces in RL: $\pi(S, t)$ *is* an eligibility trace, decaying at rate $\lambda$ and replenished by the reinforcement flux.

3. **Cognitively plausible.** The structure "forgets" itself if it is not being actively referenced by the environment (the field $\mathfrak{F}$). This matches the working-memory story: items in working memory that are not rehearsed decay and are retired.

#### Advantages

- **Semantically meaningful.** Retirement tracks *computational usefulness* (as proxied by reinforcement) rather than mere energy. A high-energy structure that no one cares about is retired just as readily as a low-energy one.
- **Natural hook for the deferred $\mathfrak{F}$ manuscript.** Any specification of the $\mathfrak{F}$-update rule — whether additive, multiplicative, or RL-policy-driven — can be paired with this mechanism without further changes to the retirement rule itself.
- **No interaction with the integrator.** Because $\pi$ is a first-order linear ODE driven by $\mathfrak{F}$, it is easy to integrate stably in discrete time and does not suffer from the transient-dip pathology of Mechanism A.

#### Concerns

- **New state variable.** Introduces $\pi(S, t)$ as a per-structure scalar field, orthogonal to the energy variables. This slightly expands the state of the dynamics.
- **Depends on the $\mathfrak{F}$-update rule being specified.** Until the deferred manuscript fixes that rule, Mechanism B cannot be fully operationalised. In the interim, one can use a placeholder rule (e.g. $\mathfrak{F}(S, t) \propto$ a count of recent template matches on $S$), but the full validation must wait.
- **Double-counting risk.** If $\mathfrak{F}$ already encodes "how much the structure is being used" somewhere in its dynamics, then $\pi$ may duplicate that information. Avoided by specifying $\pi$ as a *local time-integral* of $\mathfrak{F}(S, t)$ rather than an independent quantity.

#### Action Plan — Mechanism B

**B.1 Theoretical specification.**
- Fix a placeholder form of $\rho(\mathfrak{F})$ (e.g. $\rho(x) = \kappa \max(x, 0)$) and of the retirement threshold $\pi_{\min}$.
- State the joint dynamics (structures + $\mathfrak{F}$ + $\pi$) as a coupled ODE system.
- Prove an **eligibility-trace lemma**: in the limit $\lambda \to 0$, $\pi(S, t)$ reduces to the cumulative reinforcement received by $S$; for general $\lambda > 0$, $\pi(S, t)$ is an exponentially-weighted moving average of $\mathfrak{F}(S, \cdot)$ over a window of width $\sim 1/\lambda$.

**B.2 Reduction to a rewriting property.**
- Prove a **necessity lemma**: in the closed-energy, deletion-as-operation regime without reinforcement-driven retirement, the reachable configuration space is LBA-bounded (same as Mechanism A).
- Prove a **sufficiency lemma**: with reinforcement-driven retirement **and** an $\mathfrak{F}$ that continuously redirects reinforcement in response to the execution process, the reachable configuration space is unbounded and the system can simulate an NDTM.

**B.3 Implementation in the STP/Lagrangian simulator.**
- Extend the simulator to track per-structure $\pi(S, t)$.
- Add a placeholder $\mathfrak{F}$-update rule (e.g. deposit rate proportional to recent template-match count) to drive $\pi$ in the absence of the final $\mathfrak{F}$ manuscript.
- Add a post-step retirement pass applying the $\pi < \pi_{\min}$ rule.
- Expose $\lambda$, $\kappa$, and $\pi_{\min}$ as configuration parameters.

**B.4 Empirical validation.**
- **(Sanity)** In a toy two-structure scenario, reinforce structure $S_1$ periodically and leave $S_2$ unreinforced. Verify that $\pi(S_2, t) \to 0$ within $O(1/\lambda)$ and $S_2$ retires, while $S_1$ persists.
- **(Sensitivity)** Sweep $\lambda$ across two orders of magnitude; plot the mean structure lifetime as a function of $\lambda$. Expected: lifetime $\propto 1/\lambda$ in the unreinforced regime.
- **(Consistency with RL)** Verify that $\pi(S, t)$ behaves as an eligibility trace by comparing it against the analytical solution of the linear ODE on a known reinforcement schedule.
- **(Behavioural)** On the same small executive-tree toy problem used for Mechanism A, confirm that retirements coincide with structures that cease to receive reinforcement (i.e. whose parent atoms have fired and moved on).

**B.5 Documentation deliverable.**
- A companion note *"Reinforcement-Driven Persistence in Semantic Simulation"* containing: the formal rule, the eligibility-trace lemma, the two rewriting lemmas, the sensitivity results, and the toy-executive-tree validation.

### Comparison and Unified Formulation

Mechanisms A and B address the same problem (structural retirement for RE-regime computation) from complementary directions. Their characteristics map as follows:

| Aspect | Mechanism A (energy-threshold) | Mechanism B (reinforcement-driven) |
|--|--|--|
| New state variable per structure | None | Persistence scalar $\pi(S, t)$ |
| Driven by | Residual energy $T + V$ | Reinforcement flux $\mathfrak{F}(S, t)$ |
| Time-scale knob | Hold window $\tau_{\mathrm{hold}}$ | Decay rate $\lambda$ |
| Threshold | Energy $\epsilon_{\min}$ | Persistence $\pi_{\min}$ |
| Couples to | Existing Lagrangian | Deferred $\mathfrak{F}$-update rule |
| Retires structures that | Have lost all energy | Have lost all reinforcement |
| Interpretation | Thermal dissociation | Eligibility-trace decay |
| Dependency | Self-contained within the paper's Lagrangian | Requires $\mathfrak{F}$-update manuscript |

The two mechanisms are **not mutually exclusive**. A unified rule is straightforward:

> Retire $S$ at time $t$ when either $E_{\mathrm{res}}(S, t) < \epsilon_{\min}$ for $\tau_{\mathrm{hold}}$ seconds (Mechanism A) or $\pi(S, t) < \pi_{\min}$ (Mechanism B).

Under this **OR-combined rule**, a structure must maintain *both* non-trivial energy *and* non-trivial reinforcement to persist. Physically, this corresponds to a system in which persistence requires both internal stability (energy) and external relevance (reinforcement) — arguably the cognitively most plausible formulation.

#### Action Plan — Unified Mechanism (A ∪ B)

**U.1** Implement both Mechanism A and Mechanism B as described in their respective plans.

**U.2** In the simulator, add a flag `retirement_mode` with four settings: `{none, A_only, B_only, A_or_B}`. Run the same suite of validation experiments under all four settings.

**U.3** Compare the retirement statistics (rate, age distribution, correlation with "computational usefulness" as measured by downstream template-match activity) across the four modes. Hypothesis: `A_or_B` produces the cleanest alignment between retirement events and loss of computational usefulness, with neither mechanism alone achieving both good physical interpretability (A) and good semantic alignment (B).

**U.4** Use the outcome of U.3 to select a preferred mode for the full NDTM-equivalence proof of Step 3–5. The working expectation is that `A_or_B` will emerge as the preferred mode, but the experimental programme should treat this as a hypothesis to be tested rather than assumed.

### Dependency on Other WIP Pieces of the Framework

- **Mechanism A** is self-contained within the existing Lagrangian formalism of the paper; its companion manuscript can be written **in parallel with** or even **before** the $\mathfrak{F}$-update manuscript.
- **Mechanism B** is blocked on the $\mathfrak{F}$-update manuscript for full fidelity. A placeholder rule allows partial implementation and validation in advance; the placeholder should be swapped out for the final rule once that manuscript is complete.
- The **unified mechanism** is blocked on both, but the implementation skeleton (U.1–U.2) can be put in place while individual mechanisms mature.

### Relation to the NDTM Proof

Both mechanisms, and especially the unified rule, are the intended content of Step 4.4 of the NDTM-equivalence action plan. The proof strategy is:

1. Pick a retirement mechanism (A, B, or A ∪ B).
2. Prove the necessity lemma: without retirement, the class is LBA-bounded.
3. Prove the sufficiency lemma: with retirement plus reinforcement-replenished energy, the class is RE-bounded.
4. Combine with Step 3 (the bidirectional simulation theorems) to conclude that the executive system with retirement is exactly RE-equivalent.

Until these lemmas are proven, retirement — like the NDTM-equivalence claim itself — remains a **working design** rather than an established result. The present section specifies that design concretely enough that the proof programme can proceed.
