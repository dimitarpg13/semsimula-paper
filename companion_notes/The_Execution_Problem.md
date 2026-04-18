# The Execution Problem

## The Proposal

_Static semantic meaning_ (a structure at rest relative to an observer) can be computationally executed to produce new semantic structures. This is essentially asking: How does _meaning_ become _action_? How does _understanding_ become _computation_?

The framework so far (from [Execution_of_Semantic_Structures.docx](https://github.com/dimitarpg13/semsimula/blob/main/docs/Execution_Of_Semantic_Structures.docx)):
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

First, let us anchor the concept. From the document [Semantic_Templates.docx](https://github.com/dimitarpg13/semsimula/blob/main/docs/Semantic_Templates.docx), the latch $\mu$ appears in the definition of template matching:

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

$M\left(\mathfrak{t}\right)\text{ scans region }\mathfrak{U}\left(\mathfrak{t}\right) \in \boldsymbol{\Sigma}  \rightarrow\text{ latch }\mu\text{ triggers }\rightarrow I\left(\mathfrak{t}\right)\text{ produces }S'\text{ in } \boldsymbol{\Sigma}$

The template is an _incomplete_ semantic structure - the missing pieces are filled by template particles which are in a sense stochastic placeholders. Matching succeeds when a real semantic structure $S$ in $\boldsymbol{\Sigma}$ "fills the gaps" sufficiently well (that is, exceeding the matching threshold $\Theta$, as defined via the average trajectory weight $\bar{w}$ in the document [Semantic_Templates.docx](https://github.com/dimitarpg13/semsimula/blob/main/docs/Semantic_Templates.docx).  

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

Then execution inverts this: given $S'$, find the sequence of atomic inference templates $\mathfrak{I}_1, \mathfrak{I}_2, ...$ such that the composition of their outputs reconstitutes meaningful consequences of $S'$. The tree structure of executive atoms in $\boldsymbol{E}$ precisely organizes this sequential/branching inversion.

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
