"""
GitHub Markdown / KaTeX / Mermaid cheatsheet checker.
Checks the rules documented in GitHub_Markdown_LaTeX_Rendering_Cheatsheet.md.
Usage: python _check_cheatsheet.py file1.md [file2.md ...]
"""
import re, sys

def check(path):
    text = open(path, encoding="utf-8").read()
    lines = text.splitlines()
    issues = []

    def warn(rule, lineno, snippet):
        issues.append((path, lineno, rule, snippet[:100]))

    # ── track fenced-block state ────────────────────────────────────
    in_code   = False   # generic code fence (``` or ```lang)
    in_mermaid = False
    mermaid_lines = []
    mermaid_start = 0

    for i, raw in enumerate(lines, 1):
        # detect fence open/close
        stripped = raw.strip()
        if stripped.startswith("```"):
            lang = stripped[3:].strip().lower()
            if not in_code and not in_mermaid:
                if lang == "mermaid":
                    in_mermaid = True
                    mermaid_lines = []
                    mermaid_start = i
                else:
                    in_code = True
            elif in_code:
                in_code = False
            elif in_mermaid and stripped == "```":
                # end of mermaid block — check it
                _check_mermaid(mermaid_lines, mermaid_start, path, issues)
                in_mermaid = False
            continue

        if in_code or in_mermaid:
            if in_mermaid:
                mermaid_lines.append((i, raw))
            continue

        # ── KaTeX rules (prose lines only) ──────────────────────────

        # §1: \; or \, inside inline math
        for m in re.finditer(r'\$[^$\n]+\$', raw):
            expr = m.group()
            if r'\;' in expr or r'\,' in expr:
                warn("§1 spacing cmd (\\; or \\,) in inline math", i, expr)

        # §2: \operatorname inside any math
        for m in re.finditer(r'\$[^$\n]+\$|\$\$[^$]+\$\$', raw):
            if r'\operatorname' in m.group():
                warn("§2 \\operatorname in math", i, m.group())

        # §3: $...$ inside table cells (line has | at start/end)
        if raw.strip().startswith("|") or raw.strip().endswith("|"):
            for m in re.finditer(r'\$[^|\n]+\$', raw):
                warn("§3 $math$ inside table cell", i, raw.strip())
                break

        # §4: math inside *italic* spans  (* ... $...$ ... *)
        # Only flag single-asterisk italic delimiters, not bold (**...**).
        # Approach: strip bold spans first so they don't create false positives.
        _no_bold = re.sub(r'\*\*[^*\n]*\*\*', lambda m: ' ' * len(m.group()), raw)
        for m in re.finditer(r'(?<!\*)\*(?!\*)[^*\n]*\$[^$\n]+\$[^*\n]*(?<!\*)\*(?!\*)', _no_bold):
            warn("§4 math inside *italic* span", i, m.group())

        # §6: \| inside inline math (use \lVert instead)
        for m in re.finditer(r'\$[^$\n]+\$', raw):
            if r'\|' in m.group():
                warn("§6 \\| in inline math (use \\lVert)", i, m.group())

        # §7: display math line starting with "- "
        if raw.strip().startswith("- ") and i > 1:
            prev = lines[i-2].strip()
            if prev.startswith("$$") or (in_code and prev == ""):
                warn("§7 display math line starts with '- '", i, raw.strip())

        # §10: \tag{} in display math
        for m in re.finditer(r'\$\$[^$]*\\tag\{[^}]*\}[^$]*\$\$', raw):
            warn("§10 \\tag{} in $$...$$ block", i, m.group())

        # §11: \! in math
        for m in re.finditer(r'\$[^$\n]+\\![^$\n]*\$', raw):
            warn("§11 \\! in inline math", i, m.group())

        # §12: }_x pattern (2+ times on same line) in inline math
        inline_exprs = re.findall(r'\$[^$\n]+\$', raw)
        all_expr = " ".join(inline_exprs)
        bracket_sub = re.findall(r'\}[_^][a-zA-Z0-9]', all_expr)
        if len(bracket_sub) >= 2:
            escaped = re.findall(r'\}\\[_^][a-zA-Z0-9]', all_expr)
            unescaped = [x for x in bracket_sub if x not in escaped]
            if len(unescaped) >= 2:
                warn("§12 multiple }_x subscripts in inline math on same line", i, raw.strip())

        # §13: < followed by alphabetic inside math (HTML sanitiser)
        for m in re.finditer(r'\$[^$\n]+\$|\$\$[^$]+\$\$', raw):
            if re.search(r'<[a-zA-Z]', m.group()):
                warn("§13 <letter in math (HTML sanitiser will eat it)", i, m.group())

        # §19: \left\{...\middle|...\right\} or \left\lVert...\right\rVert
        if r'\left\{' in raw and r'\middle' in raw:
            warn("§19 \\left\\{...\\middle|...\\right\\} — use \\lbrace...\\mid...\\rbrace", i, raw.strip())
        if r'\left\lVert' in raw and r'\right\rVert' in raw:
            warn("§19 \\left\\lVert...\\right\\rVert — use \\Big\\lVert or plain \\lVert", i, raw.strip())

    return issues


def _check_mermaid(mermaid_lines, start, path, issues):
    def warn(rule, lineno, snippet):
        issues.append((path, lineno, rule, snippet[:100]))

    full_text = "\n".join(raw for _, raw in mermaid_lines)
    node_labels = re.findall(r'\["([^"]+)"\]', full_text)

    for lineno, raw in mermaid_lines:
        # §14: { or } inside quoted label
        for m in re.finditer(r'\["[^"]*[{}][^"]*"\]', raw):
            warn("§14 { or } inside quoted mermaid label", lineno, raw.strip())

        # §15: [...] nested inside quoted label
        for m in re.finditer(r'\["[^"]*\[[^"]*\][^"]*"\]', raw):
            warn("§15 [...] nested inside quoted mermaid label", lineno, raw.strip())

        # §16: subgraph ID ["Title"] — quotes inside brackets
        if re.search(r'subgraph\s+\w+\s+\["', raw):
            warn('§16 subgraph ID ["Title"] form — drop quotes', lineno, raw.strip())

        # §20: (("text")) double circle or [/"text"/] parallelogram
        if '(("' in raw or '[/"' in raw or '"/]' in raw:
            warn("§20 unsupported shape (double-circle or parallelogram)", lineno, raw.strip())

        # §21: -- inside unquoted label
        for m in re.finditer(r'\[(?!")([^\]]*--[^\]]*)\]', raw):
            warn("§21 '--' in unquoted node label", lineno, raw.strip())

        # §22: $$ inside mermaid node label
        if '$$' in raw or re.search(r'\$[^$\n]+\$', raw):
            warn("§22 $math$ inside mermaid label", lineno, raw.strip())

        # §23a: chained arrows >=4 on one line
        arrows = len(re.findall(r'-->', raw))
        if arrows >= 3:
            warn("§23 chained arrows >=4 on one line", lineno, raw.strip())

        # §23b: dotted edge with inline target definition A -. text .-> B["label"]
        if re.search(r'\.->\s*\w+\["', raw) or re.search(r'-->\|[^|]+\|\s*\w+\["', raw):
            warn("§23 inline node definition as dotted-edge target", lineno, raw.strip())


def main():
    files = sys.argv[1:]
    if not files:
        print("Usage: python _check_cheatsheet.py file1.md [file2.md ...]")
        sys.exit(1)
    total = 0
    for f in files:
        issues = check(f)
        if issues:
            print(f"\n{'='*60}")
            print(f"FILE: {f}  ({len(issues)} issue(s))")
            print('='*60)
            for _, lineno, rule, snippet in issues:
                print(f"  L{lineno:4d}  [{rule}]")
                print(f"         {snippet}")
            total += len(issues)
        else:
            print(f"OK  {f}")
    print(f"\n{'─'*60}")
    print(f"Total issues: {total}")
    return total


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
