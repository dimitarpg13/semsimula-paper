#!/usr/bin/env python3
r"""Apply the MECHANICAL cheatsheet fixes to every .md in this directory.

    python3 _ghfix.py            # dry run, prints the counts
    python3 _ghfix.py --apply    # rewrite in place

Only rules with an exact prescribed replacement are automated: 1, 5a, 9, 10,
11, 12, 12a, 13, 27. Rules needing judgement (3, 4, 6, 7, 19, 26, 29 and the
glue check) are left alone -- run `_ghlint.py` to see them.

NOT idempotent in one pass: fixing rule 9 (`\dot V` -> `\dot{V}`) creates new
rule-12 `}_x` patterns, so run it until it reports zero.

The cheatsheet itself is skipped: its bad examples are the point.
"""
import re, sys, pathlib, collections

def _cls(ch):
    PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~–—‘’“”")
    if ch is None or ch.isspace(): return 'space'
    return 'punct' if ch in PUNCT else 'alnum'

def opener_positions(line):
    masked = re.sub(r'`[^`]*`', lambda m: ' '*len(m.group(0)), line)
    op, cl = [], []
    for m in re.finditer(r'(?<![_\\])_+(?!_)', masked):
        b = _cls(masked[m.start()-1] if m.start() else None)
        a = _cls(masked[m.end()] if m.end() < len(masked) else None)
        left  = a != 'space' and not (a == 'punct' and b not in ('space','punct'))
        right = b != 'space' and not (b == 'punct' and a not in ('space','punct'))
        if left  and ((not right) or b == 'punct'): op.append(m.start())
        if right and ((not left)  or a == 'punct'): cl.append(m.start())
    return op, cl

def math_spans(line):
    masked = re.sub(r'`[^`]*`', lambda m: ' '*len(m.group(0)), line)
    return [(m.start(), m.end()) for m in
            re.finditer(r'(?<!\$)\$(?!\$)[^$\n]+?\$(?!\$)', masked)]

def in_math(pos, spans):
    return any(s < pos < e for s, e in spans)

def fix_text(t):
    counts = collections.Counter()
    lines = t.split('\n')
    out, in_fence, in_display = [], False, False
    for l in lines:
        if re.match(r'^\s*```', l):
            in_fence = not in_fence; out.append(l); continue
        if in_fence: out.append(l); continue
        st = l.strip()
        # Toggle once per `$$` on the line, wherever it sits. The old test only
        # fired when the line STARTED with `$$`, so a block written as
        #     $$a = b, \qquad
        #       c = d$$
        # opened and never closed, and every later line in the file was treated
        # as display math. That mangled 11 HTML <img> tags into \lt img ...\gt .
        n_dd = st.count('$$')
        if n_dd:
            if n_dd % 2:
                in_display = not in_display
            if st.startswith('$$') or st.endswith('$$'):
                out.append(l); continue
        math_mode = in_display

        # R5/R12 — escape the underscore that OPENS an emphasis run, when it
        # follows `}` (the cheatsheet's exact case). Only on lines with math.
        spans = math_spans(l)
        if spans:
            op, cl = opener_positions(l)
            if op and cl and any(c > o for o in op for c in cl):
                for pos in sorted(op, reverse=True):
                    if pos and l[pos-1] == '}' and in_math(pos, math_spans(l)):
                        l = l[:pos] + '\\_' + l[pos+1:]; counts['R12'] += 1

        def in_math_sub(pattern, repl, line, rule):
            nonlocal counts
            if math_mode:
                new, n = re.subn(pattern, repl, line)
                counts[rule] += n; return new
            sp = math_spans(line)
            if not sp: return line
            res, last = [], 0
            for s, e in sp:
                res.append(line[last:s])
                body, n = re.subn(pattern, repl, line[s:e])
                counts[rule] += n; res.append(body); last = e
            res.append(line[last:])
            return ''.join(res)

        l = in_math_sub(r'\^\*(?![a-zA-Z])', r'^\\ast', l, 'R5a')          # R5a
        l = in_math_sub(r'(?<![\\<>=!])<(?!=)', r'\\lt ', l, 'R13')        # R13
        l = in_math_sub(r'(?<![\\<>=!])>(?!=)', r'\\gt ', l, 'R13')
        l = in_math_sub(r'\\(lt|gt)  +', r'\\\1 ', l, 'R13')
        l = in_math_sub(r'\\;', ' ', l, 'R1')     # \; the COMMAND, not a semicolon
        l = in_math_sub(r'\\,(?![a-zA-Z])', ' ', l, 'R1')
        l = in_math_sub(r'\\!', '', l, 'R11')                              # R11
        l = in_math_sub(r'\\tag\{[^}]*\}', '', l, 'R10')                   # R10
        l = in_math_sub(r'\\(ddot|dot|bar|hat|vec|tilde)\s+([A-Za-z])',    # R9
                        r'\\\1{\2}', l, 'R9')
        # R12a — \_ inside a text-mode command is a math-mode-only command
        def r12a(m):
            counts['R12a'] += 1
            return m.group(1) + m.group(2).replace('\\_', '_') + '}'
        l = re.sub(r'(\\(?:text|mathrm|mathbf|mathit|mathsf|mathtt|textrm|textbf|'
                   r'textit|texttt|emph)\{)([^}]*\\_[^}]*)\}', r12a, l)

        # R27 — every bare `~` (not `~~`) becomes ≈ when it reads as "approximately"
        def r27(line):
            nonlocal counts
            res, i = [], 0
            code = re.sub(r'`[^`]*`', lambda m: '\x00'*len(m.group(0)), line)
            while i < len(line):
                if code[i] == '~':
                    if i+1 < len(line) and code[i+1] == '~':
                        res.append('~~'); i += 2; continue
                    if i and code[i-1] == '~':
                        res.append('~'); i += 1; continue
                    if i+1 < len(line) and (line[i+1].isdigit() or line[i+1] in '±.'):
                        res.append('≈'); counts['R27'] += 1; i += 1; continue
                res.append(line[i]); i += 1
            return ''.join(res)
        out.append(l)
    text = '\n'.join(out)

    # R27, paragraph-scoped: only when a paragraph holds two or more bare `~`.
    def bare_tilde_spans(block):
        code = re.sub(r'`[^`]*`', lambda m: '\x00'*len(m.group(0)), block)
        code = re.sub(r'~~', '\x00\x00', code)
        return [m.start() for m in re.finditer(r'~(?=[\d±.])', code)]
    paras = re.split(r'(\n\s*\n)', text)
    for i, blk in enumerate(paras):
        if blk.startswith('\n') or '```' in blk: continue
        pos = bare_tilde_spans(blk)
        if len(pos) >= 2:
            b = list(blk)
            for q in pos: b[q] = '≈'
            paras[i] = ''.join(b); counts['R27'] += len(pos)
    return ''.join(paras), counts

if __name__ == '__main__':
    apply = '--apply' in sys.argv
    total = collections.Counter(); touched = 0
    SKIP = {'GitHub_Markdown_LaTeX_Rendering_Cheatsheet.md'}  # its bad examples are the point
    for p in sorted(pathlib.Path('.').rglob('*.md')):
        if p.name in SKIP: continue
        src = p.read_text()
        new, c = fix_text(src)
        if new != src:
            touched += 1; total.update(c)
            if apply: p.write_text(new)
    print(('APPLIED' if apply else 'DRY RUN') + f" — {touched} files, {sum(total.values())} fixes")
    for r, n in total.most_common(): print(f"   {r:6s} {n:5d}")
