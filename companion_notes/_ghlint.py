#!/usr/bin/env python3
"""Lint markdown against GitHub_Markdown_LaTeX_Rendering_Cheatsheet.md.

    python3 _ghlint.py FILE.md [FILE.md ...]

Read-only. Reports rule hits with line numbers. Rule ids match the
cheatsheet's section numbers; `Rglue` is an extra check for an inline `$`
touching an adjacent word character or hyphen, which GitHub does not open.

The `_` analysis implements CommonMark's flanking rules, so it flags a line
only when one underscore can genuinely OPEN an emphasis run and another can
CLOSE it -- counting underscores alone over-fires badly on intraword
subscripts like `h_t`.
"""
import re, sys
import re
PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~–—‘’“”")
def _cls(ch):
    if ch is None or ch.isspace(): return 'space'
    if ch in PUNCT: return 'punct'
    return 'alnum'
def underscore_pair_exists(line):
    """CommonMark: can any `_` open an emphasis run that another `_` closes?"""
    masked = re.sub(r'`[^`]*`', lambda m: ' '*len(m.group(0)), line)
    runs = [m for m in re.finditer(r'(?<![_\\])_+(?!_)', masked)]  # \_ is a Markdown escape (rule 12)
    openers, closers = [], []
    for m in runs:
        b = _cls(masked[m.start()-1] if m.start() else None)
        a = _cls(masked[m.end()] if m.end() < len(masked) else None)
        left  = a != 'space' and not (a == 'punct' and b not in ('space','punct'))
        right = b != 'space' and not (b == 'punct' and a not in ('space','punct'))
        # for `_`, opening also requires: not right-flanking, or preceded by punct
        can_open  = left  and ((not right) or b == 'punct')
        can_close = right and ((not left)  or a == 'punct')
        if can_open:  openers.append(m.start())
        if can_close: closers.append(m.start())
    return any(c > o for o in openers for c in closers)



def inline_spans(line):
    """[(start, end, body)] for $...$ spans, skipping $$ and code spans."""
    # blank out inline code first
    masked = re.sub(r'`[^`]*`', lambda m: ' ' * len(m.group(0)), line)
    out = []
    for m in re.finditer(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', masked):
        out.append((m.start(), m.end(), m.group(1)))
    return out

def lint(path):
    lines = open(path).read().split('\n')
    hits = []
    in_fence = False
    in_display = False
    for i, l in enumerate(lines, 1):
        if re.match(r'^\s*```', l):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = l.strip()
        if stripped.startswith('$$') and stripped.count('$$') == 1:
            in_display = not in_display
            continue

        spans = inline_spans(l)

        # R5 / R12: underscores across the line's inline math
        if spans:
            if underscore_pair_exists(l) and any('_' in b for _, _, b in spans):
                hits.append((i, 'R5', 'a real `_` emphasis pair on a line carrying inline math', l))
            for _, _, b in spans:
                if re.search(r'\}_[A-Za-z0-9]', b):
                    hits.append((i, 'R12', 'a `}_x` subscript inside inline math', l))

        # R1 spacing commands
        if re.search(r'\\[;,](?![a-zA-Z])', l):
            hits.append((i, 'R1', r'\; or \, spacing command', l))
        if '\\!' in l:
            hits.append((i, 'R11', r'\! negative thin space', l))
        if '\\operatorname' in l:
            hits.append((i, 'R2', r'\operatorname is blocked', l))
        if re.search(r'\\tag\{', l):
            hits.append((i, 'R10', r'\tag{} in display math', l))

        # R3 math inside a table cell
        if stripped.startswith('|') and spans and not re.fullmatch(r'\|[\s:|-]*\|', stripped):
            hits.append((i, 'R3', f'inline math in a table cell ({len(spans)} span(s))', l))

        # R4 inline math inside an italic span
        for m in re.finditer(r'(?<!\*)\*(?!\*)([^*\n]+?)\*(?!\*)', l):
            if '$' in m.group(1):
                hits.append((i, 'R4', 'inline math inside an italic span', l))

        # R5a lone * inside math
        for _, _, b in spans:
            if re.search(r'(?<!\\)\*', b):
                hits.append((i, 'R5a', r'a bare * inside math — use \ast', l))

        # R6 double-bar norm inline
        for _, _, b in spans:
            if '\\|' in b:
                hits.append((i, 'R6', r'\|...\| double-bar norm inside inline math', l))

        # R9 unbraced accents
        if re.search(r'\\(ddot|dot|bar|hat|vec|tilde)\s+[A-Za-z]', l):
            hits.append((i, 'R9', 'accent without braces', l))

        # R13 < or > inside math
        for _, _, b in spans:
            if re.search(r'(?<!\\)[<>]', b):
                hits.append((i, 'R13', r'< or > inside math — use \lt / \gt', l))

        # R19 fragile delimiters
        if re.search(r'\\middle\||\\left\\lVert|\\right\\rVert', l):
            hits.append((i, 'R19', 'fragile \\left/\\middle/\\right delimiters', l))

        # R27 two bare ~ on one line
        _bare_tilde = re.sub(r'~~', '', re.sub(r'`[^`]*`', '', l))
        if len(re.findall(r'(?<!\\)~', _bare_tilde)) >= 2:
            hits.append((i, 'R27', 'two bare ~ on one line reads as strikethrough', l))

        # R28 \# inside math
        for _, _, b in spans:
            if '\\#' in b:
                hits.append((i, 'R28', r'\# inside math', l))

        # R29 bracketed optional args
        if re.search(r'\\xrightarrow\[|\\xleftarrow\[', l):
            hits.append((i, 'R29', 'bracketed optional argument', l))

        # R12a \_ inside a text-mode command
        if re.search(r'\\(text|math|textrm|textbf|textit|texttt|mathrm|mathbf|mathit|mathsf|mathtt|emph)\w*\{[^}]*\\_', l):
            hits.append((i, 'R12a', r'\_ inside a \text-family command', l))

        # R7 display-math line starting with - + *
        if in_display and re.match(r'^\s*[-+*]\s', l):
            hits.append((i, 'R7', 'display-math line starts with -, + or *', l))

        # R26 bare = or - line inside display math
        if in_display and re.fullmatch(r'\s*[=-]+\s*', l) and l.strip():
            hits.append((i, 'R26', 'bare = / - line inside display math (Setext heading)', l))

        # extra: $ glued to a word character (renders literally on GitHub)
        for s, e, b in spans:
            before = l[s-1] if s > 0 else ' '
            after = l[e] if e < len(l) else ' '
            if re.match(r'[A-Za-z0-9-]', before) or re.match(r'[A-Za-z0-9]', after):
                hits.append((i, 'Rglue', 'inline $ glued to an adjacent word character', l))
    return hits

if __name__ == '__main__':
    for path in sys.argv[1:]:
        hits = lint(path)
        print(f'### {path}: {len(hits)} hit(s)')
        agg = {}
        for i, rule, msg, l in hits:
            agg.setdefault(rule, []).append((i, msg, l))
        for rule in sorted(agg, key=lambda r: -len(agg[r])):
            rows = agg[rule]
            print(f'\n  {rule}  x{len(rows)}  — {rows[0][1]}')
            for i, msg, l in rows[:6]:
                print(f'     L{i}: {l.strip()[:120]}')
            if len(rows) > 6:
                print(f'     ... and {len(rows)-6} more')
