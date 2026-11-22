#!/usr/bin/env python3
"""How far apart are a model's input embedding and output head, per token-frequency fifth? (2026-11-22, after SR2-NP-T.)

    python3 embedding_head_alignment.py OUT UNTIED_FOLDER [TIED_FOLDER] [name=<stem>]

SR2-NP-T (tied) settled 1.44% above SR2-NP (untied), with the whole cost in the most frequent fifth of the vocabulary and
a gain on the four rarer fifths (frequency_quintile_loss.py). The question here: is that a read-out compromise, visible
already in the untied model as input and output rows that want to be different, or not?

In both models h_0 = E[x] + P[t] and the logits are h_L W^T + b; tying sets W = E. Rows are compared after removing
the mean row (a vector common to every head row shifts every logit equally and the softmax ignores it). A network could absorb one shared
linear map between the two roles (W = E A gives h_L W^T = (h_L A^T) E^T), so alignment is read two ways per token i:
  raw      cos(E_i, W_i)
  mapped   cos(E_i A, W_i) and the relative residual |E_i A - W_i| / |W_i|, with A the least-squares 384 x 384 map
           over all token types (unweighted), and again with the fit weighted by training frequency
and against a null: the same cosine for shuffled rows (E_i A against W_j).

With TIED_FOLDER, per fifth: how well the tied model's shared matrix is predicted (best linear map, R^2 of the rows)
from the untied model's input embedding and from its head, i.e. which role the shared rows took.
Fifths as in D0.4: types ranked by training frequency (the run's logfreq_surprisal buffer), five equal groups.
Writes OUT/embedding_head_alignment_<stem>.txt and .json.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve()
FOLDERS = [a for a in sys.argv[2:] if '=' not in a]
OPTS = dict(a.split('=', 1) for a in sys.argv[2:] if '=' in a)
OUT.mkdir(parents=True, exist_ok=True)


def load(folder):
    ck = [p for p in (Path.home() / 'Downloads' / folder / 'checkpoints').glob('fock_cfc_*_best.published.pt')
          if '_step' not in p.name]
    assert len(ck) == 1, ck
    sd = torch.load(ck[0], map_location='cpu', weights_only=False)['model_state_dict']
    E = sd['E.weight'].double().numpy()
    W = sd['lm_head.weight'].double().numpy() if 'lm_head.weight' in sd else E
    # a vector common to every head row adds the same amount to every logit and the softmax ignores it, so the rows
    # are compared after removing their mean; the input rows likewise (a common shift of every starting position)
    return E - E.mean(0), W - W.mean(0), sd['logfreq_surprisal'].double().numpy()


def lstsq_map(X, Y, w=None):
    if w is not None:
        s = np.sqrt(w / w.mean())[:, None]
        X, Y = X * s, Y * s
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def rowcos(X, Y):
    return (X * Y).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12)


def r2_pooled(Yhat, Y, m, w=None):
    """R^2 pooled over the rows in mask m (rows are mean-free): 1 - sum |Y - Yhat|^2 / sum |Y|^2, optionally weighted."""
    w = np.ones(len(Y)) if w is None else w
    return 1 - (w[m] * ((Y[m] - Yhat[m]) ** 2).sum(1)).sum() / (w[m] * (Y[m] ** 2).sum(1)).sum()


def main():
    E, W, s = load(FOLDERS[0])
    V = len(s)
    order = np.argsort(-s, kind='stable')
    quint = np.empty(V, dtype=np.int64)
    for q, ids in enumerate(np.array_split(order, 5)):
        quint[ids] = q
    freq = np.exp(-s); freq /= freq.sum()
    rng = np.random.default_rng(0)
    perm = rng.permutation(V)

    A = lstsq_map(E, W)
    Aw = lstsq_map(E, W, freq)
    rows = {
        'norm E': np.linalg.norm(E, axis=1), 'norm W': np.linalg.norm(W, axis=1),
        'cos(E, W) raw': rowcos(E, W),
        'cos(EA, W) mapped': rowcos(E @ A, W),
        'residual mapped': np.linalg.norm(E @ A - W, axis=1) / np.linalg.norm(W, axis=1),
        'cos(EA, W) mapped, freq-weighted fit': rowcos(E @ Aw, W),
        'residual mapped, freq-weighted fit': np.linalg.norm(E @ Aw - W, axis=1) / np.linalg.norm(W, axis=1),
        'null: cos(E_i A, W_j) shuffled': rowcos(E @ A, W[perm]),
    }
    res = {'untied': FOLDERS[0], 'fifths': {}, 'overall_freq_weighted': {}}
    bins = np.linspace(-0.4, 1.0, 57)                       # per-token distributions, for the figures
    res['hist'] = {'bins': bins.tolist()}
    for key, v in (('mapped', rows['cos(EA, W) mapped']), ('null', rows['null: cos(E_i A, W_j) shuffled'])):
        for q in (0, 4):
            res['hist'][f'{key}_Q{q}'] = np.histogram(v[quint == q], bins=bins)[0].tolist()
    lines = [f'input embedding against output head, {FOLDERS[0]}',
             f'{V:,} token types in five frequency fifths (Q0 rarest); per fifth the median over its types, '
             'and the training-frequency-weighted mean (the loss-relevant average) in brackets', '',
             f'{"reading":<40}' + ''.join(f'{"Q" + str(q):>17}' for q in range(5))]
    for k, v in rows.items():
        cells = []
        for q in range(5):
            m = quint == q
            cells.append(f'{np.median(v[m]):>8.3f} ({np.average(v[m], weights=freq[m]):6.3f})')
            res['fifths'].setdefault(k, []).append({'median': float(np.median(v[m])),
                                                    'freq_mean': float(np.average(v[m], weights=freq[m]))})
        res['overall_freq_weighted'][k] = float(np.average(v, weights=freq))
        lines.append(f'{k:<40}' + ''.join(f'{c:>17}' for c in cells))
    lines.append('')
    lines.append('frequency-weighted over all types: ' + ', '.join(f'{k} {v:.3f}' for k, v in res['overall_freq_weighted'].items()))

    if len(FOLDERS) > 1:
        Et, _, _ = load(FOLDERS[1])
        res['tied'] = FOLDERS[1]
        lines += ['', f'the tied model\'s shared matrix ({FOLDERS[1]}), predicted by a linear map from the untied model\'s rows',
                  '(per fifth: R^2 pooled over its rows, frequency-weighted in brackets; maps fitted unweighted over all types)', '',
                  f'{"predicted from":<40}' + ''.join(f'{"Q" + str(q):>17}' for q in range(5))]
        for name, X in (('untied input embedding E', E), ('untied output head W', W)):
            Yh = X @ lstsq_map(X, Et)
            cells = []
            for q in range(5):
                m = quint == q
                a, b = r2_pooled(Yh, Et, m), r2_pooled(Yh, Et, m, freq)
                cells.append(f'{a:>8.3f} ({b:6.3f})')
                res.setdefault('tied_r2', {}).setdefault(name, []).append({'pooled': float(a), 'freq_weighted': float(b)})
            lines.append(f'{name:<40}' + ''.join(f'{c:>17}' for c in cells))
        # by frequency rank: the 100 most frequent types carry about 45% of the targets
        order_f = np.argsort(s, kind='stable')
        lines += ['', 'the same by frequency rank (pooled R^2; share of training targets)']
        YE, YW = E @ lstsq_map(E, Et), W @ lstsq_map(W, Et)
        cosm = rowcos(E @ A, W)
        for lo, hi in ((0, 100), (100, 1000), (1000, 10051), (10051, V)):
            m = np.zeros(V, bool); m[order_f[lo:hi]] = True
            b = {'ranks': [lo, hi], 'share': float(freq[m].sum()), 'r2_from_E': float(r2_pooled(YE, Et, m)),
                 'r2_from_W': float(r2_pooled(YW, Et, m)), 'untied_mapped_cos_median': float(np.median(cosm[m]))}
            res.setdefault('bands', []).append(b)
            lines.append(f'  ranks {lo:>6} to {hi:<6} share {b["share"]:.3f}   R^2 from E {b["r2_from_E"]:.3f}   '
                         f'from W {b["r2_from_W"]:.3f}   untied cos(EA, W) median {b["untied_mapped_cos_median"]:.3f}')
        res['norm_shared'] = [float(np.median(np.linalg.norm(Et[quint == q], axis=1))) for q in range(5)]
        rows_t = {'norm shared (tied)': np.linalg.norm(Et, axis=1)}
        for k, v in rows_t.items():
            lines.append(f'{k:<40}' + ''.join(f'{np.median(v[quint == q]):>8.3f} ({np.average(v[quint == q], weights=freq[quint == q]):6.3f})' for q in range(5)))

    stem = OPTS.get('name', 'run')
    (OUT / f'embedding_head_alignment_{stem}.txt').write_text('\n'.join(lines) + '\n')
    (OUT / f'embedding_head_alignment_{stem}.json').write_text(json.dumps(res, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
