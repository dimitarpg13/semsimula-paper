"""FO series, Stage 0, corpus side (protocol SS5.13, C1-C5): TinyStories vs OpenWebText.

Identical procedures on both corpora, 5M GPT-2-BPE tokens each, blocks of 512
as in training, and ONE fixed embedding (GPT-2 wte, centred) for both, so every
difference is the corpus's, not a model's.

  C1  token types used; Zipf exponent alpha (log-log fit, ranks 10..10^4)
  C2  I_pred proxy = H_unigram - H_model (bits/token), H_model from the best
      trained model on that corpus; plus a model-free check, the held-out
      drop H_unigram - H_bigram (interpolated, lambda fit on held-out)
  C3  token-repetition autocorrelation R(tau) = P(x_t = x_{t+tau}) - sum p^2
  C4  embedded-stream autocovariance C_e(tau)/C_e(0): half-decay lag, tail slope
  C5  xi filter-bank Gram for the ladder's alphas: kappa(G), effective rank

Usage: python3 stage0_corpus.py OUT_DIR
"""
import json, math, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
N, BLOCK = 5_000_000, 512
ALPHAS = (0.5, 0.75, 0.95, 0.99, 0.995)       # ladder xi decays (Cell 6 log)
H_MODEL_PPL = {'TinyStories': 9.04,           # second-order anchor (d=256, L=8)
               'OpenWebText': 49.81}          # matched GPT-2 (ladder reference)
DL = Path.home() / 'Downloads'


def load():
    ts = np.load(DL / 'semsimula_fock_g1_aniso_gaussian_fockreg_tinystories/data/'
                 'tinystories_gpt2_1files_5000000toks.npz')['train'][:N].astype(np.int64)
    owt = np.load(DL / 'semsimula_fock_gamma_sweep_aniso_gaussian_fockreg_d384/data/'
                  'openwebtext_train_1000M.npy', mmap_mode='r')[:N].astype(np.int64)
    return {'TinyStories': ts, 'OpenWebText': owt}


def zipf(x):
    c = np.array(sorted(Counter(x.tolist()).values(), reverse=True), dtype=float)
    r = np.arange(1, len(c) + 1)
    sel = (r >= 10) & (r <= 10_000)
    slope = np.polyfit(np.log(r[sel]), np.log(c[sel]), 1)[0]
    p = c / c.sum()
    return len(c), -slope, float(-(p * np.log2(p)).sum()), float((p ** 2).sum())


def bigram_drop(x, h_uni):
    """Held-out interpolated bigram cross-entropy; lambda fit on a dev split."""
    tr, dev, te = x[:3_000_000], x[3_000_000:4_000_000], x[4_000_000:]
    V = 50257
    uni = np.bincount(tr, minlength=V).astype(float) + 0.5
    uni /= uni.sum()
    pair = Counter(zip(tr[:-1].tolist(), tr[1:].tolist()))
    ctx = np.bincount(tr[:-1], minlength=V).astype(float)

    def xent(seq, lam):
        a, b = seq[:-1].tolist(), seq[1:].tolist()
        tot = 0.0
        for u, w in zip(a, b):
            pb = pair.get((u, w), 0) / ctx[u] if ctx[u] > 0 else 0.0
            tot -= math.log2(lam * pb + (1 - lam) * uni[w])
        return tot / len(b)
    lam = min((0.3, 0.5, 0.7, 0.8, 0.9), key=lambda l: xent(dev[:200_000], l))
    return xent(te, lam), lam


def repetition(x, taus, p2):
    out = []
    xb = x[: (len(x) // BLOCK) * BLOCK].reshape(-1, BLOCK)
    for t in taus:
        out.append(float((xb[:, t:] == xb[:, :-t]).mean()) - p2)
    return out


def embedded_stats(x, wte, taus):
    xb = torch.from_numpy(x[: (len(x) // BLOCK) * BLOCK].reshape(-1, BLOCK))
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(xb.shape[0], generator=g)[:2000]          # ~1M tokens
    e = wte[xb[idx]]                                              # (B, 512, 768)
    e = e - wte.mean(0)                                           # centre on the vocab mean
    c0 = (e * e).sum(-1).mean()
    ac = [float((e[:, t:] * e[:, :-t]).sum(-1).mean() / c0) for t in taus]
    # xi channels: causal EMA within each block, xi_t = a xi_{t-1} + (1-a) e_t
    xis = []
    for a in ALPHAS:
        z = torch.zeros_like(e[:, 0]); out = []
        for t in range(BLOCK):
            z = a * z + (1 - a) * e[:, t]; out.append(z)
        xis.append(torch.stack(out, 1)[:, 64:])                  # skip the warm-up
    X = torch.stack([v.reshape(-1, v.shape[-1]) for v in xis], 0)  # (C, n, d)
    X = X - X.mean(1, keepdim=True)
    G = torch.einsum('cnd,knd->ck', X, X) / X.shape[1]
    ev = torch.linalg.eigvalsh(G.double())
    corr = G / torch.sqrt(torch.diag(G)[:, None] * torch.diag(G)[None, :])
    evc = torch.linalg.eigvalsh(corr.double())
    pr = float(evc.sum() ** 2 / (evc ** 2).sum())
    return ac, float(ev.max() / ev.min()), float(evc.max() / evc.min()), pr


if __name__ == '__main__':
    from transformers import GPT2Model
    wte = GPT2Model.from_pretrained('gpt2', local_files_only=True).wte.weight.detach().float()
    taus = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    res = {}
    for name, x in load().items():
        types, alpha, h_uni, p2 = zipf(x)
        h_bi, lam = bigram_drop(x, h_uni)
        rep = repetition(x, taus, p2)
        ac, kG, kC, pr = embedded_stats(x, wte, taus)
        h_mod = math.log2(H_MODEL_PPL[name])
        half = next((t for t, v in zip(taus, ac) if v <= 0.5 * ac[0]), None)
        sl = np.polyfit(np.log(taus[3:]), np.log(np.clip(ac[3:], 1e-6, None)), 1)[0]
        res[name] = dict(types=types, zipf_alpha=alpha, H_unigram=h_uni, H_bigram_heldout=h_bi,
                         bigram_lambda=lam, H_model=h_mod, Ipred_proxy=h_uni - h_mod,
                         bigram_drop=h_uni - h_bi, repetition=dict(zip(map(str, taus), rep)),
                         emb_autocorr=dict(zip(map(str, taus), ac)), emb_half_lag=half,
                         emb_tail_slope=sl, kappa_G=kG, kappa_corr=kC, eff_rank=pr)
    json.dump(res, open(OUT / 'stage0_corpus.json', 'w'), indent=1)
    T, O = res['TinyStories'], res['OpenWebText']
    row = lambda k, f='{:.3f}': f"  {k:22s} TS {f.format(T[k]):>12s}   OWT {f.format(O[k]):>12s}"
    print('FO Stage 0, corpus side: 5M GPT-2-BPE tokens each, blocks of 512, GPT-2 wte (centred)\n')
    print('C1 (Zipf)');  print(row('types', '{:d}')); print(row('zipf_alpha'))
    print('C2 (predictive information, bits/token)')
    for k in ('H_unigram', 'H_model', 'Ipred_proxy', 'H_bigram_heldout', 'bigram_drop'): print(row(k))
    print('C3 (token repetition R(tau) = P(x_t = x_t+tau) - sum p^2)')
    for t in taus: print(f"  tau={t:<4d}             TS {T['repetition'][str(t)]:12.4f}   OWT {O['repetition'][str(t)]:12.4f}")
    print('C4 (embedded autocorrelation C_e(tau)/C_e(0))')
    for t in taus: print(f"  tau={t:<4d}             TS {T['emb_autocorr'][str(t)]:12.4f}   OWT {O['emb_autocorr'][str(t)]:12.4f}")
    print(row('emb_half_lag', '{}')); print(row('emb_tail_slope'))
    print('C5 (xi filter-bank Gram, alphas %s)' % (ALPHAS,))
    for k in ('kappa_G', 'kappa_corr', 'eff_rank'): print(row(k))
