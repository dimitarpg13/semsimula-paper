#!/usr/bin/env python3
"""Build the Poisson-mode deep dive (.docx) in the template of the other deep dives (2026-10-23).

    python3 build.py            # deep_dive.md -> ../../Deep_Dive_in_the_Poisson_Mode_Registers_Wells_and_Semantic_Templates.docx

1. Code directives in deep_dive.md,  @@code <repo path> | <start regex> | <end regex>@@  (optionally a 4th field
   'cell=<notebook cell header>' for the ladder notebook), are replaced by the verbatim lines from the repository.
2. pandoc converts the Markdown (LaTeX math -> native Word equations, OMML) with the SPLM deep dive as the
   reference document, so the styles (fonts, headings, TOC, footnotes) are the template's own.
3. Post-processing gives the template's conventions: body paragraphs in Normal; code blocks one paragraph per line in
   Consolas 8.5 pt with the template's colour scheme; inline code in red Consolas; captions and tables restyled.
The table of contents is a TOC field: Word fills it in (update_toc_and_pdf.applescript does this and exports the PDF).
"""
import html
import io
import json
import keyword
import re
import builtins
import subprocess
import sys
import tokenize
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DD = HERE.parents[1]
REF = DD / 'Deep_Dive_in_the_SPLM_model_and_code.docx'
OUT = DD / 'Deep_Dive_in_the_Poisson_Mode_Registers_Wells_and_Semantic_Templates.docx'
NB = REPO / 'notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb'

# ------------------------------------------------------------------ 1. code directives
def excerpt(path, start, end, cell=None):
    if cell:
        src = [''.join(c['source']) for c in json.load(open(NB))['cells'] if ''.join(c['source']).startswith(cell)]
        assert len(src) == 1, cell
        lines = src[0].split('\n')
    else:
        lines = (REPO / path).read_text().split('\n')
    i = next(k for k, l in enumerate(lines) if re.search(start, l))
    j = next(k for k in range(i, len(lines)) if re.search(end, lines[k]))
    block = lines[i:j + 1]
    ind = min(len(l) - len(l.lstrip()) for l in block if l.strip())
    return '\n'.join(l[ind:] for l in block)

md = (HERE / 'deep_dive.md').read_text()
def _sub(m):
    parts = [p.strip() for p in m.group(1).split('|')]
    cell = parts[3][5:] if len(parts) > 3 and parts[3].startswith('cell=') else None
    return '```python\n' + excerpt(parts[0], parts[1], parts[2], cell) + '\n```'
md = re.sub(r'^@@code (.+?)@@$', _sub, md, flags=re.M)
(HERE / '_expanded.md').write_text(md)

# ------------------------------------------------------------------ 2. pandoc
raw = HERE / '_raw.docx'
subprocess.run(['pandoc', str(HERE / '_expanded.md'), '-f', 'markdown+tex_math_dollars+raw_attribute+footnotes+pipe_tables',
                '-t', 'docx', '--reference-doc', str(REF), '--syntax-highlighting=none',
                '--resource-path', str(HERE), '-o', str(raw)], check=True)

# ------------------------------------------------------------------ 3. post-processing
BUILTIN_TYPES = {'int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set', 'bytes', 'object', 'type', 'complex'}
BUILTINS = {n for n in dir(builtins) if not n.startswith('_')} - BUILTIN_TYPES
COL = dict(kw='C00000', cls='0B769F', fn='80340D', self='7030A0', bi='BF4E14', bt='074F6A', num='0F9ED5',
           com='3A7C22', str='4EA72E', deco='FFC000')

def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def run(text, color=None, sz=17):
    rpr = '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/>'
    if color:
        rpr += f'<w:color w:val="{color}"/>'
    rpr += f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'
    return f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'

def highlight_line_tokens(code):
    """[(line_no, col_start, col_end, colour)] for a Python snippet, PyCharm-light style (the template's)."""
    spans = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code + '\n').readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    prev_sig = None
    fdepth = 0
    line_start = True                      # no significant token yet on this logical line
    deco = False                           # inside a decorator name ('@' at a line start)
    for k, tk in enumerate(toks):
        tt, s, (r0, c0), (r1, c1) = tk.type, tk.string, tk.start, tk.end
        ttn = tokenize.tok_name[tt]
        if ttn in ('FSTRING_START', 'FSTRING_MIDDLE', 'FSTRING_END'):
            spans.append((r0, c0, r1, c1, COL['str']))
            fdepth += (ttn == 'FSTRING_START') - (ttn == 'FSTRING_END')
            prev_sig = tk; continue
        col = None
        if tt == tokenize.COMMENT:
            col = COL['com']
        elif tt == tokenize.STRING:
            col = COL['str']
        elif tt == tokenize.NUMBER:
            col = COL['num']
        elif tt == tokenize.NAME:
            nxt = next((t for t in toks[k + 1:] if t.type not in (tokenize.NL, tokenize.NEWLINE)), None)
            called = nxt is not None and nxt.string == '('
            if deco:
                col = COL['deco']
            elif keyword.iskeyword(s) or s in ('self_',):
                col = COL['kw']
            elif s in ('self', 'cls'):
                col = COL['self']
            elif prev_sig is not None and prev_sig.string == 'def':
                col = COL['fn']
            elif prev_sig is not None and prev_sig.string == 'class':
                col = COL['cls']
            elif s in BUILTIN_TYPES:
                col = COL['bt']
            elif s in BUILTINS and (prev_sig is None or prev_sig.string != '.'):
                col = COL['bi']
            elif s[:1].isupper():
                col = COL['cls']
            elif called:
                col = COL['fn']
        elif tt == tokenize.OP and s == '@' and fdepth == 0 and line_start:
            col = COL['deco']; deco = True
        if col:
            spans.append((r0, c0, r1, c1, col))
        if tt in (tokenize.NEWLINE, tokenize.NL):
            line_start, deco = True, False
        elif tt not in (tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
            line_start = False
            if not (tt == tokenize.OP and s in ('@', '.')) and not (tt == tokenize.NAME and deco):
                deco = False
        if tt not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
            prev_sig = tk
    return spans

def code_paragraphs(code):
    lines = code.split('\n')
    spans = highlight_line_tokens(code)
    colour = [[None] * (len(l) + 1) for l in lines]
    if spans is not None:
        for r0, c0, r1, c1, col in spans:
            for r in range(r0, r1 + 1):
                if r - 1 >= len(lines):
                    continue
                a = c0 if r == r0 else 0
                b = c1 if r == r1 else len(lines[r - 1])
                for c in range(a, min(b, len(lines[r - 1]))):
                    colour[r - 1][c] = col
    out = []
    for li, l in enumerate(lines):
        runs, cur, buf = [], None, ''
        for c, ch in enumerate(l):
            col = colour[li][c] if not ch.isspace() or colour[li][c] in (COL['str'], COL['com']) else None
            if col != cur and buf:
                runs.append(run(buf, cur)); buf = ''
            cur = col; buf += ch
        if buf:
            runs.append(run(buf, cur))
        if not runs:
            runs.append(run(''))
        out.append('<w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/><w:rPr>'
                   '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="17"/></w:rPr></w:pPr>' + ''.join(runs) + '</w:p>')
    out.append('<w:p><w:pPr><w:spacing w:after="0"/></w:pPr></w:p>')
    return ''.join(out)

def para_text(p):
    txt = ''
    for m in re.finditer(r'<w:t(?: [^>]*)?>([^<]*)</w:t>|<w:br/>|<w:tab/>', p):
        if m.group(0) == '<w:br/>':
            txt += '\n'
        elif m.group(0) == '<w:tab/>':
            txt += '\t'
        else:
            txt += html.unescape(m.group(1))
    return txt

z = zipfile.ZipFile(raw)
files = {n: z.read(n) for n in z.namelist()}
doc = files['word/document.xml'].decode('utf8')
doc = re.sub(r'\s+/>', '/>', doc)              # pandoc writes '<w:br />'; text content escapes '>', so this only touches tags

# code blocks
doc = re.sub(r'<w:p>(?:(?!</w:p>).)*?<w:pStyle w:val="SourceCode"/>.*?</w:p>',
             lambda m: code_paragraphs(para_text(m.group(0))), doc, flags=re.S)
# inline code: red Consolas at the body size
doc = doc.replace('<w:rStyle w:val="VerbatimChar"/>',
                  '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/><w:color w:val="C00000"/>')
# body paragraphs -> Normal with 6 pt after; list items -> Normal, 2 pt after
doc = re.sub(r'<w:pStyle w:val="(?:FirstParagraph|BodyText)"/>', '<w:spacing w:after="120"/>', doc)
doc = re.sub(r'<w:pStyle w:val="Compact"/>', '<w:spacing w:after="40"/>', doc)
# figures and captions
doc = re.sub(r'<w:pStyle w:val="(?:CaptionedFigure|Figure)"/>', '<w:jc w:val="center"/><w:spacing w:after="60"/>', doc)
doc = re.sub(r'<w:pStyle w:val="ImageCaption"/>', '<w:spacing w:after="200"/><w:rPr><w:i/><w:sz w:val="17"/></w:rPr>', doc)
# tables: thin borders, compact cells
doc = doc.replace('<w:tblStyle w:val="Table"/>',
                  '<w:tblStyle w:val="Table"/><w:tblBorders><w:top w:val="single" w:sz="4" w:color="A6A6A6"/>'
                  '<w:bottom w:val="single" w:sz="4" w:color="A6A6A6"/><w:insideH w:val="single" w:sz="2" w:color="D9D9D9"/>'
                  '</w:tblBorders>')
files['word/document.xml'] = doc.encode('utf8')

st = files['word/settings.xml'].decode('utf8')
if '<w:updateFields' not in st:
    st = st.replace('</w:settings>', '<w:updateFields w:val="true"/></w:settings>')
files['word/settings.xml'] = st.encode('utf8')

with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zo:
    for n in z.namelist():
        zo.writestr(n, files[n])
print('wrote', OUT, f'({OUT.stat().st_size / 1e6:.2f} MB)')
