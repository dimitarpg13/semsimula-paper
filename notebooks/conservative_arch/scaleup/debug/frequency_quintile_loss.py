#!/usr/bin/env python3
"""Loss by token-frequency quintile (diagnostic D0.4, 2026-11-22, for SR2-NP-T's calls TE2 and TE3).

    python3 frequency_quintile_loss.py OUT FOLDER [FOLDER ...] [windows=128] [threads=8]
                                       [val=<tokens.npy>] [logfreq=<surprisal.npy>] [name=<stem>]

Quintiles as in D0.4 (colab_fock_gaussian_sarf_openwebtext_phase5.ipynb, section 4): the vocabulary's token types
ranked by their frequency in the training split, ascending, and split into five groups of equal size; Q0 holds the
rarest types, Q4 the most frequent. The ranking is read from the run's unigram surprisal table
(results/logfreq_surprisal_<corpus>.npy, add-one smoothed counts of train_ids), which orders the types as the counts
do; types of equal count keep their index order.

Each FOLDER is a ladder run folder in ~/Downloads, rebuilt from its tag through the notebook's own cells
(copy_split.ladder_model), or the matched GPT-2's folder (copy_split.gpt2_model).

The reference per quintile is the unigram prior's loss on the same targets: the mean training surprisal of the
targets, which is what a model knowing only token frequencies would score. The uniform guess ln V is printed for
comparison with D0.4, but it is not a floor: a calibrated model must score above ln V on types rarer than 1/V. The per-token loss on `windows` windows of 513 validation tokens, evenly spaced over the
file, is cached per model in OUT (ce_<tag>.npy), so a later model can be added without recomputing the first.

Readings per model and quintile: the mean loss over every validation target in the quintile (D0.4's estimator),
the number of targets and their share. With two or more folders, the paired difference against the first, on the
same targets, with a standard error clustered by window. The uniform guess is ln V nats.
Writes OUT/frequency_quintiles_<stem>.txt and .json.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve()
FOLDERS = [a for a in sys.argv[2:] if '=' not in a]
OPTS = dict(a.split('=', 1) for a in sys.argv[2:] if '=' in a)
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))
W, T = int(OPTS.get('windows', 128)), 512
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(int(OPTS.get('threads', 8)))
VAL = Path(OPTS.get('val', Path.home() / 'Downloads/semsimula_fock_gamma_sweep_aniso_gaussian_fockreg_d384/data/openwebtext_val_2M.npy'))
LOGFREQ = Path(OPTS.get('logfreq', Path.home() / 'Downloads' / FOLDERS[0] / 'results/logfreq_surprisal_openwebtext.npy'))


def tag_of(folder):
    return folder.split('semsimula_fock_cfc_baoab_', 1)[-1]


def per_token_loss(folder, X, Y):
    cache = OUT / f'ce_{tag_of(folder)}.npy'
    if cache.exists():
        ce = np.load(cache)
        if ce.shape == X.shape:
            return ce, f'{folder} (cached)'
    src = (DBG / 'copy_split.py').read_text()
    src = src[:src.index("\nif MODE == 'nll':")]
    sys.argv = ['copy_split.py', str(OUT), 'probe', 'windows=1', f"threads={OPTS.get('threads', 8)}"]
    cs = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
    exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), cs)
    if folder.startswith('semsimula_matched_gpt2'):       # the matched GPT-2 (tied, no output bias): a reference
        run, desc = cs['gpt2_model'](folder)
    else:
        run, desc = cs['ladder_model'](folder, OPTS.get('ckpt'))
    ce = np.zeros(X.shape, dtype=np.float32)
    for i in range(0, len(X), 4):
        lg = run(X[i:i + 4])
        ce[i:i + 4] = torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.shape[-1]).float(), torch.from_numpy(Y[i:i + 4]).reshape(-1),
            reduction='none').reshape(lg.shape[:2]).numpy()
        print(f'  {tag_of(folder)[:40]}... windows {min(i + 4, len(X))}/{len(X)}', flush=True)
    np.save(cache, ce)
    return ce, desc


def main():
    v = np.load(VAL).astype(np.int64)
    starts = (np.arange(W) * ((len(v) - (T + 1)) / W)).astype(np.int64)
    xy = np.stack([v[s:s + T + 1] for s in starts])
    X, Y = xy[:, :T], xy[:, 1:]

    s = np.load(LOGFREQ).astype(np.float64)
    V = len(s)
    order = np.argsort(-s, kind='stable')                   # ascending training frequency (highest surprisal first)
    quint = np.empty(V, dtype=np.int64)
    for q, ids in enumerate(np.array_split(order, 5)):
        quint[ids] = q
    Qy = quint[Y]
    ln_v = float(np.log(V))

    models = []
    for f in FOLDERS:
        ce, desc = per_token_loss(f, X, Y)
        print('model:', desc, flush=True)
        models.append((f, ce))

    lines = [f'Loss by token-frequency quintile (D0.4) -- {W} validation windows x {T} = {X.size:,} targets '
             f'({VAL.name}); quintiles of the {V:,} token types by training frequency ({LOGFREQ.name}); '
             f'uniform guess ln V = {ln_v:.2f} nats', '']
    res = {'windows': W, 'T': T, 'val': str(VAL), 'logfreq': str(LOGFREQ), 'ln_V': ln_v, 'models': {}}
    n_q = [int((Qy == q).sum()) for q in range(5)]
    sy = s[Y]
    uni = [float(sy[Qy == q].mean()) for q in range(5)]
    lines.append('unigram prior (training frequencies only), mean loss per quintile: '
                 + ', '.join(f'Q{q} {uni[q]:.3f}' for q in range(5)) + f'; overall {sy.mean():.3f}')
    lines.append('')
    res['unigram_prior'] = {'quintiles': uni, 'overall': float(sy.mean())}
    for f, ce in models:
        name = tag_of(f)
        lines.append(f'{name}')
        lines.append(f'  overall mean loss {ce.mean():.4f} nats (perplexity {np.exp(ce.mean()):.2f} on these windows)')
        lines.append(f"  {'quintile':<16}{'mean loss':>10}{'unigram':>9}{'gain':>8}{'targets':>10}{'share':>8}   below ln V?")
        row = {'overall': float(ce.mean()), 'quintiles': []}
        for q in range(5):
            m = float(ce[Qy == q].mean())
            label = {0: 'Q0 rarest', 4: 'Q4 most frequent'}.get(q, f'Q{q}')
            lines.append(f'  {label:<16}{m:>10.3f}{uni[q]:>9.3f}{uni[q] - m:>8.3f}{n_q[q]:>10,}{n_q[q] / X.size:>8.1%}'
                         f'   {"yes" if m < ln_v else "no"}')
            row['quintiles'].append({'q': q, 'mean_loss': m, 'gain_over_unigram': uni[q] - m, 'targets': n_q[q]})
        row['all_below_ln_V'] = all(r['mean_loss'] < ln_v for r in row['quintiles'])
        res['models'][name] = row
        lines.append('')

    if len(models) > 1:
        f0, ce0 = models[0]
        for f, ce in models[1:]:
            d = ce - ce0
            lines.append(f'paired: {tag_of(f)}')
            lines.append(f'  minus {tag_of(f0)}, same targets; standard error clustered by window')
            pair = []
            for q in range(5):
                msk = Qy == q
                n = msk.sum()
                mean = float(d[msk].mean())
                S = (d * msk).sum(1) - mean * msk.sum(1)        # per-window residual sums
                se = float(np.sqrt((S ** 2).sum()) / n)
                lines.append(f'  Q{q}: {mean:+.3f} nats (se {se:.3f})')
                pair.append({'q': q, 'diff': mean, 'se': se})
            lines.append(f'  overall: {float(d.mean()):+.4f} nats')
            res.setdefault('paired', {})[tag_of(f)] = {'against': tag_of(f0), 'quintiles': pair,
                                                        'overall': float(d.mean())}
            lines.append('')

    stem = OPTS.get('name', 'run')
    (OUT / f'frequency_quintiles_{stem}.txt').write_text('\n'.join(lines) + '\n')
    (OUT / f'frequency_quintiles_{stem}.json').write_text(json.dumps(res, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
