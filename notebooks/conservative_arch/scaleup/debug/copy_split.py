#!/usr/bin/env python3
"""Where is the code gap? Per-token loss split by whether the target can be copied from the context (2026-10-26).

    python3 copy_split.py OUT_DIR nll NAME FOLDER [windows=512] [threads=8] [ckpt=<file>] [val=<code_val_2M.npy>]
    python3 copy_split.py OUT_DIR report [ref=gpt2] [windows=512] [val=<code_val_2M.npy>]

On code the conservative model trails the matched GPT-2 far more than on TinyStories (at step 2,000: 1.82x
against 1.14x). One reading: code rewards exact copying of names chosen afresh in each file, which attention
does by pointing at one earlier token and the conservative model can only approximate through smoothed
context channels and a pair potential. This splits both models' loss on the same validation tokens by what
the context offers for each target, so the gap can be located instead of guessed at:

  first        the target token does not occur earlier in the window: nothing to copy
  repeat       it occurs earlier, but not after the current token: copyable by identity only
  induction    the pair (current token, target) occurs earlier in the window: an induction-head target,
               the most copyable kind

plus the token's class (indentation and newlines, identifier-like, number, punctuation and other), the distance
to the target's last earlier occurrence, and the position in the window.

`nll` writes OUT_DIR/nll_<NAME>.npy, the per-token loss (windows x 512) of one model on windows of 513
validation tokens spaced evenly over the set, so every model sees the same targets. FOLDER is
a run folder in ~/Downloads: a ladder folder (semsimula_fock_cfc_baoab_code_<tag>; its Cell 0 is rebuilt from
the tag's parts) or the matched GPT-2's (semsimula_matched_gpt2_...). `report` compares every cached model with
the reference and writes copy_split_report.txt and .json. Inference only, on CPU.
"""
import contextlib, io, json, re, sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve(); MODE = sys.argv[2]     # absolute: the ladder build changes directory
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
POS = [a for a in sys.argv[3:] if '=' not in a]
W = int(OPTS.get('windows', 512)); T = 512
VAL = Path(OPTS.get('val', Path.home() / 'Downloads/semsimula_code_gpt2/data/code_val_2M.npy'))
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(int(OPTS.get('threads', 8)))


def windows():
    # Evenly spaced over the whole validation set: perplexity varies several-fold from file to file, so the
    # first W windows would measure a few files, not the set.
    v = np.load(VAL).astype(np.int64)
    assert len(v) >= W * (T + 1), (len(v), W)
    st = np.linspace(0, len(v) - (T + 1), W).astype(np.int64)
    c = np.stack([v[s:s + T + 1] for s in st])
    return c[:, :T], c[:, 1:]


def ladder_model(folder, ckpt):
    """The ladder notebook's own Cells 0-5b (verify_pm_switch.build) with the folder tag's switches."""
    sys.argv = [sys.argv[0], str(OUT / 'harness')]
    sys.path.insert(0, str(Path(__file__).parent))
    import verify_pm_switch as P
    tag = re.fullmatch(r'semsimula_fock_cfc_baoab_[a-z]+_(.+)', folder).group(1)
    parts = tag.split('_')
    repl = P.LIVE
    if 'norc' in parts:
        repl += (('REVERSE_CHANNEL              = True', 'REVERSE_CHANNEL              = False'),)
    if 'sr2' in parts:
        repl += (('LOWRANK_DAMPED_FLOW  = False', 'LOWRANK_DAMPED_FLOW  = True'),)
    if 'fob' in parts:
        repl += (('FIRST_ORDER_OVERDAMPED = False', 'FIRST_ORDER_OVERDAMPED = True'),)
    k = next((int(t[2:]) for t in parts if re.fullmatch(r'pm\d+', t)), 0)
    if k:
        repl += (('POISSON_MODES        = 0', f'POISSON_MODES        = {k}'),)
        if 'pmx16' in parts:
            repl += (('POISSON_WELLS_EXACT  = False', 'POISSON_WELLS_EXACT  = True'),)
    if 'st16250' in parts:
        repl += (('LADDER_STEPS     = 32_500', 'LADDER_STEPS     = 16_250'),)
    if not ckpt:
        ck = [p.name for p in (Path.home() / 'Downloads' / folder / 'checkpoints').glob('fock_cfc_*_best.pt')
              if '_step' not in p.name]
        assert len(ck) == 1, ck
        ckpt = ck[0]
    model, g, _ = P.build(folder, repl, ckpt=ckpt)
    built = g['_variant_tag'] if '_variant_tag' in g else None
    assert built is None or built == tag, (built, tag)
    model.to('cpu'); model.eval(); model.cfg.use_layer_checkpoint = False

    def run(x):
        with torch.enable_grad():          # the V_phi pair force is an autograd gradient
            return model(torch.from_numpy(x))[0].detach()
    return run, f'{ckpt} (built tag {built})'


def gpt2_model(folder):
    """The matched GPT-2 notebook's own model class (its Cells 0 and 3) and the run's best checkpoint."""
    nb = Path(__file__).resolve().parents[1] / 'colab_matched_gpt2_baseline_openwebtext.ipynb'
    cells = [''.join(c['source']) for c in json.load(open(nb))['cells']]
    ck = torch.load(Path.home() / 'Downloads' / folder / 'checkpoints' / 'gpt2_baseline_best.pt',
                    map_location='cpu', weights_only=False)
    g = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec('import math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np', g)
        exec(cells[1], g)                  # Cell 0, the defaults (d = 384, L = 8, 6 heads, tied)
        exec(cells[4], g)                  # Cell 3, the model definition
    m = g['GPT2'](vocab_size=g['VOCAB_SIZE'], d=g['D_MODEL'], n_layers=g['N_LAYERS'], n_heads=g['N_HEADS'],
                  d_ff=g['D_FF'], max_len=g['MAX_LEN'], dropout=0.0, tie_embeddings=g['TIE_EMBEDDINGS'])
    m.load_state_dict(ck['model_state_dict']); m.eval()

    def run(x):
        with torch.no_grad():
            return m(torch.from_numpy(x))[0]
    return run, f"gpt2_baseline_best.pt (step {ck['step']}, val {ck['val_ppl']:.3f})"


def nll(name, folder):
    x, y = windows()
    run, what = (gpt2_model(folder) if 'matched_gpt2' in folder
                 else ladder_model(folder, OPTS.get('ckpt')))
    rows = []
    for i in range(0, W, 4):
        lo = run(x[i:i + 4])
        ce = torch.nn.functional.cross_entropy(lo.reshape(-1, lo.shape[-1]).float(),
                                               torch.from_numpy(y[i:i + 4]).reshape(-1), reduction='none')
        rows.append(ce.reshape(-1, T).numpy())
        print(f'   {name}: windows {i + 4}/{W}  running ppl {np.exp(np.concatenate(rows).mean()):.3f}', flush=True)
    a = np.concatenate(rows)
    np.save(OUT / f'nll_{name}.npy', a)
    (OUT / f'nll_{name}.txt').write_text(f'{name}: {folder}\n  {what}\n  {W} windows x {T}, ppl {np.exp(a.mean()):.4f}\n')
    print(f'{name}: ppl {np.exp(a.mean()):.4f} on {W} x {T} tokens -> {OUT / f"nll_{name}.npy"}')


def categories(x, y):
    """Per target (window, i): the class of y[i] given the context x[:i + 1]."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    text = [tok.decode([i]) for i in range(tok.vocab_size)]
    cls = np.full(tok.vocab_size, 3, np.int8)                      # 3 punctuation and other
    for i, s in enumerate(text):
        if s.strip() == '':
            cls[i] = 0                                             # 0 indentation, newlines, spaces
        elif re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', s.strip()):
            cls[i] = 1                                             # 1 identifier-like (names, keywords)
        elif re.fullmatch(r'[0-9.]+', s.strip()):
            cls[i] = 2                                             # 2 number
    kind = np.zeros(x.shape, np.int8); dist = np.zeros(x.shape, np.int32)
    for w in range(x.shape[0]):
        last, pairs = {}, set()
        for i in range(T):
            last[x[w, i]] = i
            if i:
                pairs.add((x[w, i - 1], x[w, i]))
            t = y[w, i]
            if t in last:
                dist[w, i] = i + 1 - last[t]
                kind[w, i] = 2 if (x[w, i], t) in pairs else 1     # 2 induction, 1 repeat
    return kind, cls[y], dist


def report():
    ref = OPTS.get('ref', 'gpt2')
    x, y = windows()
    nl = {p.stem[4:]: np.load(p) for p in sorted(OUT.glob('nll_*.npy'))}
    assert ref in nl, f'no nll_{ref}.npy in {OUT}'
    for k, a in nl.items():
        assert a.shape == (W, T), (k, a.shape, W)
    kind, cls, dist = categories(x, y)
    pos = np.broadcast_to(np.arange(T), (W, T))
    groups = {
        'all': np.ones_like(kind, bool),
        'first occurrence': kind == 0, 'repeat (not induction)': kind == 1, 'induction': kind == 2,
        'indent/newline/space': cls == 0, 'identifier-like': cls == 1, 'number': cls == 2, 'punct/other': cls == 3,
        'identifier, first': (cls == 1) & (kind == 0), 'identifier, repeat': (cls == 1) & (kind == 1),
        'identifier, induction': (cls == 1) & (kind == 2),
        'repeat/ind., dist 1-16': (kind > 0) & (dist <= 16), 'repeat/ind., dist 17-64': (kind > 0) & (dist > 16) & (dist <= 64),
        'repeat/ind., dist 65-256': (kind > 0) & (dist > 64) & (dist <= 256), 'repeat/ind., dist 257-512': (kind > 0) & (dist > 256),
        'position 0-63': pos < 64, 'position 64-255': (pos >= 64) & (pos < 256), 'position 256-511': pos >= 256,
    }
    lines, js = [], {'windows': W, 'tokens': W * T, 'ref': ref, 'val': str(VAL), 'models': {}}
    lines.append(f'COPY SPLIT on code validation: {W} windows x {T} = {W * T:,} targets, reference {ref}')
    lines.append(f'  ppl on these tokens: ' + ', '.join(f'{k} {np.exp(a.mean()):.3f}' for k, a in nl.items()))
    for k, a in nl.items():
        if k == ref:
            continue
        gap = a - nl[ref]; tot = gap.sum()
        lines.append(f'\n{k} against {ref}: mean gap {gap.mean():+.4f} nats/token (ppl ratio {np.exp(gap.mean()):.3f})')
        lines.append(f'  {"group":28s} {"share":>6s} {ref + " nll":>9s} {k + " nll":>12s} {"gap":>8s} {"ratio":>6s} {"of gap":>7s}  seq spread')
        js['models'][k] = {}
        for gname, m in groups.items():
            n = int(m.sum())
            if not n:
                continue
            g = gap[m]
            # spread over windows: the group's mean gap per window, its standard error across windows
            pw = [gap[w][m[w]].mean() for w in range(W) if m[w].any()]
            se = float(np.std(pw, ddof=1) / np.sqrt(len(pw))) if len(pw) > 1 else float('nan')
            row = {'n': n, 'share': n / m.size, 'ref_nll': float(nl[ref][m].mean()), 'nll': float(a[m].mean()),
                   'gap': float(g.mean()), 'gap_se': se, 'gap_share': float(g.sum() / tot)}
            js['models'][k][gname] = row
            lines.append(f'  {gname:28s} {row["share"]:6.1%} {row["ref_nll"]:9.3f} {row["nll"]:12.3f} '
                         f'{row["gap"]:+8.3f} {np.exp(row["gap"]):6.2f} {row["gap_share"]:7.1%}  ±{se:.3f}')
    txt = '\n'.join(lines)
    print(txt)
    stem = 'copy_split_report' + ('' if ref == 'gpt2' else f'_vs_{ref}')
    (OUT / f'{stem}.txt').write_text(txt + '\n')
    (OUT / f'{stem}.json').write_text(json.dumps(js, indent=1))


if MODE == 'nll':
    nll(POS[0], POS[1])
elif MODE == 'report':
    report()
else:
    raise SystemExit(__doc__)
