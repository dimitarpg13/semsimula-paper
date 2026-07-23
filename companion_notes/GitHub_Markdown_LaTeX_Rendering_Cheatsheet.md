# GitHub Markdown Rendering Cheatsheet (KaTeX + Mermaid)

A reference for writing LaTeX math **and** Mermaid diagrams in Markdown files that render correctly on GitHub (and similar renderers). Every rule here was confirmed by observing an actual rendering failure on `github.com`.

The cheatsheet has two parts:

- **Part I — KaTeX math rendering** (rules §1–§13, §19): inline `$...$` and display `$$...$$` math.
- **Part II — Mermaid diagram rendering** (rules §14–§18, §20–§22): ` ```mermaid ` fenced blocks.

---

# Part I — KaTeX math

---

## 1. Spacing commands render as punctuation — remove them

| Do not use | Renders as | Use instead |
| ---------- | ---------- | ----------- |
| `\;` | `;` (semicolon) | nothing, or `\quad` for large gaps |
| `\,` | `,` (comma) | nothing (operators carry their own spacing) |

**Rule:** never use `\;` or `\,` as spacing modifiers around operators in display or inline math. Math-mode operators (`=`, `\le`, `\ge`, `\approx`, `\sim`, `-`, `+`, etc.) carry correct spacing automatically.

```latex
% Bad
I(s_0; s_\ell) \;\le\; \dim M \cdot \log_2(L_M/\epsilon) \;-\; \frac{\ell \cdot \gamma}{\ln 2}

% Good
I(s_0; s_\ell) \le \dim M \cdot \log_2(L_M/\epsilon) - \frac{\ell \cdot \gamma}{\ln 2}
```

---

## 2. `\operatorname` is blocked — use alternatives

`\operatorname{...}` is explicitly disallowed by this renderer and produces a pink error block.

| Do not use | Use instead |
| ---------- | ----------- |
| `\operatorname{div} F` | `\nabla \cdot F` (physics notation) |
| `\operatorname{tr} A` | `\mathrm{tr}(A)` (parens provide visual separation) |
| `\operatorname{diam}(K)` | `\text{diam}(K)` |
| `\operatorname{grad} f` | `\nabla f` |

**Rule:** use `\nabla \cdot`, `\nabla`, `\mathrm{...}(...)`, or `\text{...}` as replacements. Always use explicit parentheses around the argument when using `\mathrm` or `\text` so there is no ambiguity about spacing.

```latex
% Bad
\operatorname{div} F = -\dim M \cdot \gamma

% Good
\nabla \cdot F = -\dim M \cdot \gamma
```

---

## 3. `$\sim$100%` and similar in table cells — use plain Unicode

Inline math inside `| ... |` table cells is often not parsed. The `$...$` delimiters are swallowed.

| Do not use | Use instead |
| ---------- | ----------- |
| `$\sim$100%` | `~100%` |
| `$\approx$50` | `~50` or `≈50` |
| `$\le$` | `≤` (copy-paste the Unicode character) |

**Rule:** for simple symbols inside table cells, use the plain Unicode character or ASCII approximation. Reserve `$...$` in table cells for expressions that have no plain-text equivalent and are short enough not to confuse the parser.

---

## 4. Inline math inside italic spans breaks — remove the italic wrapper

GitHub's Markdown parser resolves `*...*` italic spans **before** handing text to the math renderer. Any `$...$` expression inside an italic span is therefore processed as Markdown first; the resulting mangled text is then passed to KaTeX, which cannot parse it. The math silently renders as raw literal text (e.g., `$h_t$` appears verbatim).

This applies at **both levels**:

- **Paragraph-level italic:** a whole paragraph wrapped in `*...*`.
- **Inline italic clause:** a phrase or sentence wrapped in `*...*` inside an otherwise normal paragraph. This is the most common silent failure — the clause looks fine in the source but the math inside it is not rendered.

```markdown
<!-- Bad: entire paragraph italic -->
*Companion to `docs/foo.md`. Derives $D^\ast$ in closed form.*

<!-- Good: plain text, no italic wrapper -->
Companion to `docs/foo.md`. Derives $D^\ast$ in closed form.
```

```markdown
<!-- Bad: inline italic clause containing math -->
The test asks: *is $h_t$ alone sufficient for predicting $h_{t+1}$?*

<!-- Good: remove the italic markers; the question reads clearly in plain text -->
The test asks: is $h_t$ alone sufficient for predicting $h_{t+1}$?
```

**Rule:** never wrap a clause or sentence in `*...*` (or `_..._`) if it contains `$...$` math expressions or backtick code spans. Use plain text. If emphasis is needed, restructure the sentence so the emphasised fragment contains no math (e.g., use **bold** for a keyword before or after the math).

---

## 5. Multiple underscores in one inline `$...$` expression — break it up

GitHub's Markdown parser pairs `_` characters as italic delimiters *before* the math renderer processes the expression. A long inline expression like `$s_\ell = (x_\ell, \dot x_\ell, \xi_\ell, \mathfrak{m}_\ell, \theta_\ell)$` has five underscores; the parser may pair the 4th and 5th as `_italic_`, stripping both subscripts.

**Symptoms:** subscripts disappear from the rendered output mid-expression.

**Fix options (in order of preference):**

1. Break into several shorter expressions, each with ≤ 2 underscores:
```markdown
<!-- Bad -->
The state $s_\ell = (x_\ell, \dot x_\ell, \xi_\ell, \mathfrak{m}_\ell, \theta_\ell)$ lives in...

<!-- Good -->
The state $s_\ell$ — with components position $x_\ell$, velocity $\dot{x}_\ell$,
context $\xi_\ell$, mass $\mathfrak{m}_\ell$, and parameters $\theta_\ell$ — lives in...
```

2. Move the expression to a display block `$$...$$` where the parser is less aggressive.

3. Replace complex subscript notation with plain-text description and a reference to the EOM document.

---

## 5a. A lone `*` (superscript/optimality star) inside math — use `\ast` instead

The same pre-processing GitHub applies for `_` (rule 5, rule 12) also applies to a literal `*`: GitHub's Markdown parser pairs single `*` characters as italic delimiters **before** KaTeX ever sees the content, and it does this across the whole paragraph, not just within one `$...$`/`$$...$$` span. A pattern like `R^{*}` used twice in nearby text or equations — e.g. once in `$$R^{*} = \arg\min_{R \in O(d)} \dots$$` and again a few lines later in `$x_C = R^{*} x_P$` — gives the parser two lone `*` characters to pair as an italic open/close span. Everything between them (including the intervening display equation) gets swallowed by the emphasis pass, and KaTeX receives mangled/truncated input, typically failing with:

> **Extra close brace or missing open brace**

**Symptom:** the equation renders as a pink error box, and the "raw" fallback text GitHub shows underneath has the `*` characters silently replaced by `_` (a visible fingerprint that the Markdown emphasis pass — not KaTeX — is what actually broke the input).

```latex
% Bad — two lone "*" characters (one per R^{*}) let the Markdown parser
% pair them as italic delimiters across the whole span between them
$$
R^{*} = \arg\min_{R \in O(d)} \lVert R E_P - E_C\rVert_F = U V^\top,
\qquad
x_C = R^{*} x_P.
$$

% Good — \ast is a first-class KaTeX command, renders identically,
% and contains no literal "*" for the Markdown parser to pair
$$
R^{\ast} = \arg\min_{R \in O(d)} \lVert R E_P - E_C\rVert_F = U V^\top,
\qquad
x_C = R^{\ast} x_P.
$$
```

**Rule:** never use a bare `*` inside `$...$` or `$$...$$` math (common in optimality notation like `R^*`, `w^*`, `\theta^*`, or swept-value notation like `\gamma^*`). Always write `\ast` (or `\star` if the five-pointed-star glyph is intended instead of the asterisk glyph). This is safe even when only one `*` appears in the whole document, but it is *mandatory* whenever the same starred symbol is reused more than once, since a single unpaired `*` will happily pair with any other lone `*` anywhere later in the rendered page.

---

## 6. `\|...\|` (double-bar norm) inside inline math — avoid in prose

`\|` in Markdown source looks like `\|` to the Markdown parser. Even in prose (not tables), a `\|...\|` norm like `\|x - x_{c,k}\|^2` can break the math context because `\` escapes the `|` character, making the math parser see unbalanced delimiters.

```latex
% Bad (in inline prose math)
$\sum_k V_k \cdot (1 - e^{-\kappa_k^2 \|x - x_{c,k}\|^2})$

% Good options:
$\sum_k V_k \cdot (1 - e^{-\kappa_k^2 \lVert x - x_{c,k} \rVert^2})$  % use \lVert / \rVert
% or just describe in prose and put the full formula in a display block
```

**Rule:** prefer `\lVert ... \rVert` over `\|...\|` in inline math. For complex norms with subscripted arguments, move to a display block.

---

## 7. Display math lines starting with `-` become list items

Inside a `$$...$$` block, if any line starts with `- ` (hyphen + space), GitHub's Markdown parser converts it to a bullet list item, breaking the equation.

```latex
% Bad — the "- \frac{...}" line starts with "-"
$$
\boxed{
D^\ast \le \frac{A}{\log_2 n}
= \frac{B}{\log_2 n}
- \frac{C}{\ln 2 \cdot \log_2 n}
}
$$

% Good — collapse to one line (or restructure to avoid leading "-")
$$
\boxed{ D^\ast \le \frac{A}{\log_2 n} = \frac{B}{\log_2 n} - \frac{C}{\ln 2 \cdot \log_2 n} }
$$
```

**Rule:** never let a display-math line begin with `- `. Either put the whole expression on one line, or restructure so the minus sign is not the first character on the line (e.g. carry it to the end of the previous line).

---

## 8. Thousands separators — prefer plain numbers

`4{,}600` (the LaTeX idiom for comma-separated thousands) sometimes renders correctly and sometimes produces `4,600` with extra space depending on the KaTeX version. In any case, the result is the same digit string. For clarity and compatibility, just write `4600` in rendered documents.

---

## 9. `\ddot x` vs `\ddot{x}` — always use braces

Without braces, some renderers apply `\ddot` only to the next character and the subscript may attach unexpectedly.

```latex
% Safer
\ddot{x}_\ell    % double-dot over x, subscript ell on x
\dot{x}_\ell     % single dot over x, subscript ell on x
```

---

## 10. `\tag{...}` in display blocks causes vertical rendering — remove it

Inside a `$$...$$` block, `\tag{n}` tells KaTeX to number the equation. On GitHub's specific renderer the `\tag` mechanism forces the equation into an internal "AMS-style" layout that stacks the content vertically rather than displaying it on a single horizontal line.

```latex
% Bad — causes the whole equation to display vertically
$$w_t \ddot{h}_t + \gamma(h_t) \dot{h}_t = -\nabla V(h_t) \tag{67}$$

% Good — number the equation in surrounding prose instead
The full dissipation-adjusted Euler–Lagrange equation (Eq. 67) is:

$$w_t \ddot{h}_t + \gamma(h_t) \dot{h}_t = -\nabla V(h_t)$$
```

**Rule:** never use `\tag{...}` inside `$$...$$` blocks in GitHub Markdown. Reference equation numbers in the preceding or following sentence instead.

---

## 11. `\!` (negative thin space) inside display or inline math — remove it

`\!` is a negative thin space in LaTeX. It is supported by KaTeX in principle but can interact badly with `\left(` and other delimiters, producing parse failures or broken layout in some GitHub renderer versions.

```latex
% Bad
F_{\text{diff}}\!\left(x_i, \{x_j\}_{j \neq i}\right)

% Good — just remove \! (the spacing is fine without it)
F_{\text{diff}}\left(x_i, \{x_j\}_{j \neq i}\right)
```

**Rule:** never use `\!` in GitHub Markdown math. Remove it; the visual difference is imperceptible in rendered HTML.

---

## 12. `}_x` subscripts inside inline math trigger italic — escape as `\_`

When an inline `$...$` expression contains a subscript of the form `\cmd{arg}_x` (closing brace `}` immediately followed by `_`), GitHub's Markdown parser treats the `_` as a potential italic delimiter. The `}` is punctuation (not alphanumeric), so the parser considers `_x` a valid left-flanking italic opener. When two such `}_x` patterns appear on the same line — in the same or different `$...$` spans — the parser pairs them as italic open/close markers, which:

- breaks the math expression (subscripts become plain text or disappear), and
- causes all text after the first matched `_` to render in italic until the paired closing `_` is found elsewhere on the line or page.

**Symptom:** text after an expression like `$\ddot{h}_t + \dot{h}_t$` suddenly renders in italic; the rendered math shows `\dot{h}t` with `_` and everything after in italic.

**Mechanism:** `\_` is a Markdown-level escape. The `\` is consumed by GitHub's Markdown processor, preventing `_` from acting as an italic delimiter. KaTeX then receives the bare `_` and interprets it correctly as a subscript operator.

```markdown
<!-- Bad: two }_t patterns on one line — the parser pairs them as italic -->
$w_t\ddot{h}_t + \gamma(h_t)\dot{h}_t = -\nabla V(h_t)$

<!-- Good: escape the _ that follows } -->
$w_t\ddot{h}\_t + \gamma(h_t)\dot{h}\_t = -\nabla V(h_t)$
```

**Rule:** whenever an inline `$...$` expression (or a line containing multiple such spans) has two or more `}_[alphanumeric]` subscript patterns, change each `}_x` to `}\_x`. This applies to common patterns like `\ddot{h}\_t`, `\dot{h}\_t`, `\vec{d}\_1`, `\mathfrak{m}\_i`, `\bar{R}\_1`, `\mathrm{Dyck}\_n`, etc. Alternatively, move such expressions to a display block `$$...$$` where the Markdown parser is less aggressive.

---

## 13. `<` and `>` inside math — replace with `\lt` and `\gt`

GitHub's Markdown pipeline runs an **HTML sanitiser before KaTeX**. Whenever it sees `<` followed by an alphabetic character (e.g., `<k`, `<n`, `<j`), it treats the substring as the start of an HTML tag and consumes everything up to the next `>` — even if both characters live inside `$...$` or `$$...$$`.

The consumed text is silently stripped, leaving KaTeX with truncated input and unbalanced braces. The resulting render error is the cryptic pink box:

> **Extra open brace or missing close brace**

**Symptom:** A display equation that contains `<X` (alphabetic) and a later `>` on the same line (typical: `\xi^{<k}\_t` together with `h\_{>t}`) fails to render with the brace-mismatch error, even though the source is balanced.

**Mechanism:** `<k}\_t; h\_{>` matches the HTML pattern `<tagname...>`, so the sanitiser eats it as if it were a malformed tag. KaTeX then sees a fragment with one `^{` left dangling.

```markdown
<!-- Bad: HTML preprocessor eats the substring between <k and the next > -->
$$
I(\xi\_t; h\_{>t}) = \sum\_{k=1}^{K} I(\xi^k\_t; h\_{>t} \mid \xi^{<k}\_t).
$$

<!-- Good: \lt and \gt bypass the HTML sanitiser entirely -->
$$
I(\xi\_t; h\_{\gt t}) = \sum\_{k=1}^{K} I(\xi^k\_t; h\_{\gt t} \mid \xi^{\lt k}\_t).
$$
```

**Rule:** inside any `$...$` or `$$...$$` block, replace literal `<` with `\lt` and literal `>` with `\gt`. Both are first-class KaTeX commands and render identically to the literal characters. Standalone `>` (not preceded by a `<X` pattern earlier on the line) is usually safe, but using `\gt` everywhere is the robust default.

**Scope:** this rule applies only inside math. In prose, the literal `<` is fine when followed by whitespace or digits (e.g., "`< 1024 tokens`"), since the HTML sanitiser only opens a tag context on `<` followed by an alphabetic character.

---

## 19. `\left\{ ... \middle| ... \right\}` and `\left\lVert ... \right\rVert` fail on GitHub — use plain delimiters

GitHub's KaTeX rejects several `\left ... \right`-paired constructs that are valid in standard KaTeX, with a misleading error:

> **Missing or unrecognized delimiter for \left**

The error is *always* reported on `\left`, but the actual cascade originates from one of:

- **`\middle|` between `\left\{ ... \right\}`** in a set-builder (the most common failure).
- **`\left\lVert ... \right\rVert`** when the content spans many tokens or contains nested `\left/\right`.
- **`\left\{ ... \right\}` nested inside `\underbrace{ ... }`** (the inner `\left/\right` pair confuses the renderer).

Confirmed reproduction: a clean source line like

```latex
\mathcal{F}_S = \left\{ F : \mathbb{R}^d \to \mathbb{R}^d \middle| F = -\nabla V \right\}
```

renders correctly under standalone KaTeX and inside LaTeX, but fails on GitHub with the error above. Replacing `\left/\middle/\right` with plain delimiters renders correctly.

**Mechanism (best guess):** GitHub's KaTeX is configured with a stricter delimiter pairing pass that does not handle `\middle` reliably, and that bails out on `\left` when the matching `\right` is separated from it by a nested `\left/\right` pair (which is exactly what `\underbrace{\left{...}\right}` looks like to the parser). Standalone, fixed-size delimiters never trigger this code path.

| Pattern | Bad (GitHub fails) | Good (renders everywhere) |
| ------- | ------------------ | ------------------------- |
| Set-builder | `\left\{ X \middle\| Y \right\}` | `\lbrace X \mid Y \rbrace` |
| Single-line braced group | `\left\{ X \right\}` | `\lbrace X \rbrace` |
| Braced group inside `\underbrace` | `\underbrace{\left\{ X \right\}}_{\dots}` | `\underbrace{\lbrace X \rbrace}_{\dots}` |
| Long norm | `\left\lVert \text{long} \right\rVert` | `\Big\lVert \text{long} \Big\rVert` or plain `\lVert \text{long} \rVert` |
| Short norm needing big braces | `\left\lVert h \right\rVert^2` | `\lVert h \rVert^2` (height matches naturally) |

```latex
% Bad — set-builder with \left/\middle/\right
$$
\mathcal{F}_A = \left\{ F_\ell(h) = -\nabla V_\ell(h) + \Omega_\ell(h)\dot h \middle| V_\ell(h) = \dots \right\}
$$

% Good — plain \lbrace ... \mid ... \rbrace
$$
\mathcal{F}_A = \lbrace F_\ell(h) = -\nabla V_\ell(h) + \Omega_\ell(h)\dot h \mid V_\ell(h) = \dots \rbrace
$$
```

```latex
% Bad — \underbrace wrapping \left\{ ... \right\}
$$
\underbrace{\left\{ m\ddot h = -\nabla V_\theta(h;\xi) \right\}}_{\text{Class A}} \circ \dots
$$

% Good — \underbrace wrapping plain \lbrace ... \rbrace
$$
\underbrace{\lbrace m\ddot h = -\nabla V_\theta(h;\xi) \rbrace}_{\text{Class A}} \circ \dots
$$
```

```latex
% Bad — \left\lVert wrapping a long expression
$$
\mathcal{L} = \sum_{\ell,t} \left\lVert h^{(\ell+1)}_t - \alpha_\ell h^{(\ell)}_t + \dots \right\rVert^2
$$

% Good — \Big\lVert (or plain \lVert) gives the same visual at no parsing cost
$$
\mathcal{L} = \sum_{\ell,t} \Big\lVert h^{(\ell+1)}_t - \alpha_\ell h^{(\ell)}_t + \dots \Big\rVert^2
$$
```

**Rule:** never use `\left/\middle/\right` in GitHub Markdown for set-builders, braced groups inside `\underbrace`, or norm displays whose content spans more than a few tokens. Use:

- `\lbrace ... \mid ... \rbrace` for set-builders (universal, no sizing needed for one-line definitions).
- `\Big\lVert ... \Big\rVert` (or `\bigg\lVert ... \bigg\rVert`) for norms of multi-token expressions.
- Plain `\lVert ... \rVert` when the content already has natural delimiter height (subscripts, simple variables).

Reserve `\left ... \right` for short, simple parenthetical groups like `\left( \frac{a}{b} \right)` where automatic sizing is genuinely useful and the content is small enough not to trip the parser.

---

# Part II — Mermaid diagrams

The symptom of a violated rule below is almost always the same red box from GitHub:

> Unable to render rich display
> Cannot read properties of undefined (reading 'render')

This message means the parser succeeded but the renderer rejected the AST. There is no per-line diagnostic; you have to know the failure modes.

---

## 14. `{...}` braces inside quoted node labels — escape or rephrase

Mermaid reserves `{...}` for the rhombus (decision-diamond) shape: `Node{label}`. Even *inside* a quoted label `["..."]`, the lexer can choke on `{` / `}`. Common offenders are LaTeX-style subscripts like `{l+1}` or `R^{4d}`.

```text
%% Bad — "{l+1}" inside a quoted label
Update["v_{l+1} = (v_l + dt·f/m)/(1+dt·γ)"]
V9["V_θ : R^{2d} → R"]
```

```text
%% Good — replace {l+1} with (l+1) or "l plus 1"; spell exponents out
Update["v_new = v_l plus dt f over m / 1 plus dt gamma"]
V9["V_theta : R^2d -> R"]
```

**Rule:** never put a `{` or `}` inside a mermaid node label, even when the label is quoted. Replace `_{x}` subscripts with `_x` or words; replace `^{2d}` with `^2d` or `power 2d`. If you really need a brace character, use the HTML entity `&#123;` / `&#125;`.

---

## 15. `[...]` square brackets inside quoted node labels — replace with parens

Mermaid uses `[...]` to delimit the *label itself* of a rectangular node: `Node[Label here]`. When the label is quoted (`Node["..."]`) the parser is *supposed* to treat `[` and `]` inside the quotes as literal text, but in practice GitHub's Mermaid version regularly fails on patterns like `["E[x_t]"]` or `["[ξ^1, ξ^2]"]`.

```text
%% Bad — nested [...] inside a quoted label
Embed["E[x_t] + P[t]<br/>(token + position embedding)"]
xi11["[ξ^1, ξ^2, ξ^3, ξ^4]<br/>∈ R^{4d}"]
```

```text
%% Good — use parentheses or just spaces
Embed["E of x_t plus P of t (embedding)"]
xi11["xi_1 xi_2 xi_3 xi_4 in R^4d"]
```

**Rule:** never nest `[...]` inside a `["..."]` label. Use parentheses or plain space-separated tokens.

---

## 16. `subgraph ID ["Title"]` — drop the quotes

The legacy form `subgraph ID [Plain Title]` and the modern form `subgraph "Title"` both work. The mixed form `subgraph ID ["Title"]` (quotes *inside* the brackets) is not part of the Mermaid grammar and is a confirmed failure mode on GitHub.

```text
%% Bad
subgraph LayerStep ["One layer-step (damped Euler)"]
    H_in["h_l"] --> Xi["xi_l"]
end
```

```text
%% Good — drop the quotes, drop the inner parentheses if you can
subgraph LayerStep [One layer-step - damped Euler]
    H_in["h_l"] --> Xi["xi_l"]
end
```

**Rule:** for subgraphs use `subgraph ID [Plain Title]` (no quotes, minimal punctuation). If you need richer text, use the bare `subgraph "Title"` form (no ID) — but never combine the two.

---

## 17. Dotted edge with label — use pipe syntax when the label contains a dot

Mermaid supports a *dotted* edge with optional label, but the label form is strict: there must be **a space on each side of the label**, and the arrow must close with `.->` (or `.-` for no arrowhead). The compact form `-.text.-` (no spaces, no arrowhead) is non-standard and triggers a render failure.

**Additional pitfall:** if the label text itself contains a `.` (e.g. `loss.backward`), the inline form `-. loss.backward .->` fails because the parser reads the `.` inside the label as the closing delimiter. Use the **pipe-based label syntax** `-.->|text|` instead, which is immune to dots in the label.

```text
%% Bad — non-standard dotted edge label form
MultiXi -.contrasts with.- FullAttn["Full attention"]

%% Bad — dot inside label breaks the inline form
Loss -. loss.backward .-> Logits
```

```text
%% Good — space-bounded label, .-> closing (when label has no dots)
MultiXi -. contrasts with .-> FullAttn["Full attention"]

%% Good — pipe-based label (safe for any label text, including dots)
Loss -.->|loss backward| Logits
```

**Rule:** dotted edges with labels must be written as `A -. text .-> B` (or `A -. text .- B` for no arrowhead) with spaces around the label. If the label contains a `.` character, switch to the pipe form `A -.->|text| B` to avoid a lexical error.

---

## 18. Cautionary cleanups (not always required, but recommended)

These are not confirmed-fatal on every GitHub Mermaid version, but each one has been observed contributing to instability and removing them never breaks anything that worked before:

| Pattern | Replace with | Why |
| ------- | ------------ | --- |
| `<br/>` (self-closing XHTML) | `<br>` | older Mermaid lexers occasionally trip on `/` inside the tag; `<br>` is universally accepted |
| `?` inside any label, e.g. `Xi["ξ_l = ?"]` | drop it, or use `unknown`, or use a word | the question mark interacts badly with some grammar rules in older parser versions |
| `..` inside any label, e.g. `A["xi 1..K"]` | spell out as words, e.g. `A["xi K"]` | the double-dot sequence is tokenised as a range operator in GitHub's Mermaid version and triggers a silent render failure |
| hyphen (`-`) immediately followed by a letter inside a label, e.g. `A["top-k"]` | replace with a space, e.g. `A["top k"]` | the `-letter` pattern resembles the start of a dotted edge and can corrupt tokenisation |
| `=` inside labels, e.g. `A["f = grad V"]` | replace with a word, e.g. `A["f neg grad V"]` | `=` is safe in most renderers but triggers crashes in GitHub's stricter parser when combined with certain label content |
| `_` inside quoted node labels, e.g. `A["h_t"]` | replace with a space or remove, e.g. `A["h"]` | even inside `["..."]`, GitHub's Mermaid lexer can treat `_` as a Markdown italic delimiter, corrupting the label and crashing the diagram |
| Unicode operators in labels (`−`, `∇`, `Σ`, `∈`, `→`, `≈`, `∞`) | ASCII spell-out (`-`, `grad`, `sum`, `in`, `->`, `approx`, `infty`) | confirmed-safe in 100 % of GitHub Mermaid versions; Unicode works on most but not all |
| Greek letters in labels (`α`, `β`, `γ`, `θ`, `ξ`) | ASCII spell-out (`alpha`, `beta`, `gamma`, `theta`, `xi`) | same as above; also makes the source readable in non-Unicode terminals |

**Rule:** when your KaTeX-rich document also contains Mermaid diagrams, prefer ASCII inside the diagram labels and keep the Greek / unicode in the surrounding KaTeX prose. The diagram is for structure; the math beside it is for symbols.

---

## 20. Advanced Mermaid node shapes — avoid `(("text"))` and `[/"text"/]`

GitHub's Mermaid renderer does **not** reliably support several advanced node shapes introduced in newer Mermaid versions:

| Shape syntax | Intended shape | Problem |
| ------------ | -------------- | ------- |
| `(("text"))` | double circle | parser crashes: "Cannot read properties of undefined (reading 'render')" |
| `[/"text"/]` | parallelogram (lean-right) | same crash — the `/` inside brackets is mis-parsed |
| `[\"text"\]` | parallelogram (lean-left) | same crash |
| `{{"text"}}` | hexagon | intermittent failures when combined with quotes |

**Fix:** replace with a universally supported shape:

```
# Instead of double-circle:
mul(("x"))          -->   mul("x")       # rounded rectangle (stadium)

# Instead of parallelogram:
h_t[/"h_t (query)"/]  -->   h_t["h_t (query)"]   # plain rectangle
```

Supported shapes on GitHub Mermaid: `[text]` (rectangle), `["text"]` (rectangle, quoted), `(text)` / `("text")` (rounded/stadium), `{text}` (rhombus), `>text]` (asymmetric).

---

## 21. `--` inside unquoted Mermaid node labels — always quote

A double-dash `--` inside an unquoted `[...]` node label is parsed as an edge connector, not as text. This corrupts the graph definition and produces the same "Cannot read properties of undefined (reading 'render')" crash.

```
# Broken — Mermaid sees "Lever 3" as text, then "--" as a new edge:
D1[Lever 3 -- competitive Phi]

# Fixed — quoted label treats everything as a literal string:
D1["Lever 3 -- competitive Phi"]
```

**Rule:** any node label that contains `--`, `---`, `-->`, or `-.` must be wrapped in double quotes inside its shape delimiters: `["text with -- in it"]`.

---

## 22. `$$...$$` math delimiters inside Mermaid node labels — use plain ASCII text

GitHub's Mermaid renderer attempts to parse `$$...$$` inside node labels as KaTeX math, but the interaction between the Mermaid lexer and KaTeX's parser is unreliable. The typical failure message is:

> **KaTeX parse error: Can't use function '$' in math mode at position 13: \gamma \to 0$$**

The closing `$$` is mis-parsed: once KaTeX enters math mode at the opening `$$`, it interprets the second `$` of the closing delimiter as an attempt to invoke the TeX `$` command inside an already-open math context.

```
# Broken — $$...$$ inside node label:
A["$$\\gamma \\to 0$$\nHamiltonian\nConservative\n$$\\nabla \\cdot F = 0$$"]

# Fixed — plain ASCII, no math delimiters:
A["gamma -> 0\nHamiltonian\nConservative\ndiv F = 0"]
```

**Rule:** never use `$$...$$` (or `$...$`) math delimiters inside Mermaid node labels. Replace math symbols with their ASCII equivalents (e.g., `gamma` for $\gamma$, `->` for $\to$, `infty` for $\infty$, `div F` for $\nabla \cdot F$, `grad V` for $\nabla V$). Keep the real LaTeX math in the surrounding prose where KaTeX renders it correctly.

---

## 23. Chained arrows and inline-defined targets of dotted edges — declare every node on its own line

Two patterns are **valid in upstream Mermaid** but **trigger the generic "Cannot read properties of undefined (reading 'render')" crash** on GitHub:

**Pattern A — chained arrows on a single line.** A chain of four or more nodes connected by repeated `-->` on the same source line tokenises ambiguously in GitHub's parser, particularly in `flowchart LR` layouts. The diagram silently fails with the standard render-undefined error.

```text
%% Bad — chained arrows of length >= 4
Hop1982 --> HopCont --> DenseHop --> Attn --> Transformer --> SemSimula
```

**Pattern B — inline node definition as the target of a dotted-edge label.** When a dotted edge of the form `A -. text .-> B` is closed by a target node whose label is being defined inline on the same line (`B["label text"]`), the parser tries to match `.->` against the inline label content and gets confused once the target's label brackets appear. The same diagram crashes.

```text
%% Bad — Lyap is defined inline at the end of a dotted edge with an inline label
Hop1982 -. derives from energy .-> Lyap["Lyapunov function E"]
```

**Mechanism (best guess):** GitHub's Mermaid version uses a parser that:

- treats each line as a single statement and tokenises it greedily; long arrow chains and dotted-edge labels both push it into a code path that fails when the line is "too rich",
- builds the node table from declarations encountered in lexical order, so an inline label declaration appearing after a dotted-edge label on the same line is not registered correctly.

**Fix — the robust pattern:**

1. **Declare every node on its own line first** at the top of the diagram. Each declaration is `NodeId["label"]` (or `NodeId(("label"))`, etc.) on a line by itself.
2. **Write one edge per line.** Never chain `A --> B --> C` on a single line.
3. **Use pipe-form labels for all dotted edges**: `A -.->|text| B` instead of `A -. text .-> B`. The pipe form is immune to label content (it already handles dots in labels per §17) and to the inline-target-definition failure mode of this rule.
4. **Subgraphs reference pre-declared nodes** — they should not define nodes inline. Move every `NodeId["label"]` outside the `subgraph ... end` block; inside the subgraph, list only the bare `NodeId`.

```text
%% Good — pre-declare nodes, one edge per line, pipe-form dotted edges
flowchart LR
    Hop1982["Classical Hopfield 1982<br>binary states<br>Hebbian weights"]
    HopCont["Continuous Hopfield 1984<br>tanh neurons"]
    DenseHop["Dense Hopfield 2020<br>logsumexp energy"]
    Attn["Scaled dot product attention"]
    Transformer["Transformer block"]
    SemSimula["SemSimula trajectory dynamics"]
    Lyap["Lyapunov function E"]

    Hop1982 --> HopCont
    HopCont --> DenseHop
    DenseHop --> Attn
    Attn --> Transformer
    Transformer --> SemSimula
    Hop1982 -.->|derives from energy| Lyap
    DenseHop -.->|derives from energy| Lyap
    SemSimula -.->|derives from energy| Lyap
```

```text
%% Good — subgraph with pre-declared nodes (declarations outside, references inside)
flowchart TB
    H["Hopfield network"]
    R["Restricted Boltzmann Machine"]
    DH["Dense Hopfield 2020"]
    AT["Scaled dot product attention"]

    subgraph CLAS [Classical energy based models]
        H
        R
    end
    subgraph DENSE [Modern dense Hopfield]
        DH
        AT
    end

    H -->|adds hidden layer| R
    H --> DH
    DH --> AT
```

**Rule:** for any non-trivial flowchart (≥ 4 nodes, or any subgraph, or any dotted edge), apply all four parts of the robust pattern above. The cost is a slightly longer source listing; the benefit is that the diagram renders on every GitHub version observed to date. The patterns this rule replaces (chained arrows, inline-target dotted edges, subgraphs with inline node definitions) work fine in standalone Mermaid editors and in some older GitHub renderer versions, but are not robust across the versions GitHub currently serves.

---

## Quick reference card

### KaTeX (Part I)

| Problem | Symptom | Fix |
| ------- | ------- | --- |
| `\;` in math | renders as `;` | remove it |
| `\,` in math | renders as `,` | remove it |
| `\operatorname{foo}` | pink error block | use `\text{foo}`, `\mathrm{foo}(...)`, or physics notation |
| `$\sim$` in table cell | shows raw `$\sim$` | use `~` |
| Italic block or inline `*clause*` + math | math renders as raw literal text | remove `*...*`; use plain text or bold for emphasis |
| Many `_` in one `$...$` | subscripts disappear | break into shorter expressions |
| Lone `*` in math (e.g. `R^{*}`), reused later in the doc | "Extra close brace or missing open brace"; fallback text shows `_` where `*` was | use `\ast` (or `\star`) instead of a literal `*` |
| `\|...\|` in inline math | math context broken | use `\lVert...\rVert` or display block |
| Display line starts with `- ` | becomes bullet point | collapse to one line |
| `\boxed{...}` multiline with `-` | bullet inside box | single-line `\boxed{...}` |
| `\tag{n}` in `$$...$$` | equation renders vertically | remove `\tag`, number in prose |
| `\!` near `\left(` | parse failure or broken layout | remove `\!` entirely |
| `}_x` in inline math (2+ on same line) | italic bleeds; subscript disappears | change `}_x` to `}\_x` |
| `<X` (alphabetic) in math, with later `>` on same line | "Extra open brace or missing close brace"; HTML sanitiser eats text | replace `<` with `\lt`, `>` with `\gt` |
| `\left\{ ... \middle\| ... \right\}` set-builder | "Missing or unrecognized delimiter for \left" | use `\lbrace ... \mid ... \rbrace` |
| `\left\lVert ... \right\rVert` over a long expression | same error | use `\Big\lVert ... \Big\rVert` or plain `\lVert ... \rVert` |
| `\underbrace{\left\{ X \right\}}_{...}` | same error | use `\underbrace{\lbrace X \rbrace}_{...}` |

### Mermaid (Part II)

| Problem | Symptom | Fix |
| ------- | ------- | --- |
| `{...}` inside a quoted node label, e.g. `["v_{l+1}"]` | "Cannot read properties of undefined (reading 'render')" | remove braces; rephrase as `(l+1)` or words |
| `[...]` nested inside a quoted node label, e.g. `["E[x_t]"]` | same render error | replace nested brackets with parens or spaces |
| `subgraph ID ["Title"]` (quotes inside brackets) | same render error | use `subgraph ID [Plain Title]` (no quotes) |
| `-.text.-` dotted-edge label | same render error | use `-. text .->` (spaces around label, `.->` closing); if label contains `.`, use `-.->|text|` pipe form |
| `<br/>` (self-closing) in label | intermittent render error | use `<br>` |
| `?` in label | intermittent render error | drop it or replace with a word |
| `..` in label, e.g. `"xi 1..K"` | silent render failure | spell out as words, e.g. `"xi K"` |
| `-letter` in label, e.g. `"top-k"` | silent render failure | replace hyphen with space: `"top k"` |
| `=` in label, e.g. `"f = grad V"` | silent render failure | replace with a word: `"f neg grad V"` |
| `_` in quoted label, e.g. `["h_t"]` | silent render failure | remove or replace with space: `["h"]` |
| Unicode / Greek in label | intermittent render error on some versions | spell out as ASCII (`alpha`, `xi`, `grad`, `->`, `approx`, ...) |
| `(("text"))` double-circle or `[/"text"/]` parallelogram | "Cannot read properties of undefined (reading 'render')" | use `("text")` (stadium) or `["text"]` (rectangle) instead |
| `--` inside unquoted node label, e.g. `[Lever 3 -- X]` | same render error — `--` parsed as edge | quote the label: `["Lever 3 -- X"]` |
| `(` / `[` / `{` inside a pipe-form edge label, e.g. `A -->\|f (x)\| B` | `Parse error ... got 'PS'` (the `(` token) | remove the bracket — edge labels aren't quoted; use plain words (`A -->\|f of x\| B`). Parens are only safe in quoted *node* labels |
| `$$...$$` or `$...$` math inside node label | "KaTeX parse error: Can't use function '$' in math mode" | remove math delimiters; use ASCII (`gamma`, `->`, `div F`, `grad V`) |
| Chained arrows ≥ 4 on one line, e.g. `A --> B --> C --> D --> E` | "Cannot read properties of undefined (reading 'render')" | one edge per line |
| Inline node definition as target of dotted-edge label, e.g. `A -. text .-> B["label"]` | same render error | pre-declare `B["label"]` on its own line; use pipe form `A -.->\|text\| B` |
| Subgraph with inline node definitions inside the block | intermittent render error | declare nodes outside the `subgraph ... end` block; reference bare `NodeId` inside |
