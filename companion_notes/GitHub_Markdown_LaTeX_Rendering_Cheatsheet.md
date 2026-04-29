# GitHub Markdown + KaTeX Rendering Cheatsheet

A reference for writing LaTeX math in Markdown files that render correctly on GitHub (and similar KaTeX-based renderers). Every rule here was confirmed by observing actual rendering failures.

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

## Quick reference card

| Problem | Symptom | Fix |
| ------- | ------- | --- |
| `\;` in math | renders as `;` | remove it |
| `\,` in math | renders as `,` | remove it |
| `\operatorname{foo}` | pink error block | use `\text{foo}`, `\mathrm{foo}(...)`, or physics notation |
| `$\sim$` in table cell | shows raw `$\sim$` | use `~` |
| Italic block or inline `*clause*` + math | math renders as raw literal text | remove `*...*`; use plain text or bold for emphasis |
| Many `_` in one `$...$` | subscripts disappear | break into shorter expressions |
| `\|...\|` in inline math | math context broken | use `\lVert...\rVert` or display block |
| Display line starts with `- ` | becomes bullet point | collapse to one line |
| `\boxed{...}` multiline with `-` | bullet inside box | single-line `\boxed{...}` |
| `\tag{n}` in `$$...$$` | equation renders vertically | remove `\tag`, number in prose |
| `\!` near `\left(` | parse failure or broken layout | remove `\!` entirely |
| `}_x` in inline math (2+ on same line) | italic bleeds; subscript disappears | change `}_x` to `}\_x` |
