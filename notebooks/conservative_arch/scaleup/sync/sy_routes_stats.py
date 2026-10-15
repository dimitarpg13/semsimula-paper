"""Interval estimates for the SY route readouts (the same computations as sy_routes.py's summary)."""
import numpy as np
import sy_instruments as S


def enrichment(Ssum):
    """related selections / selections expected at the unrelated rate for their distances."""
    ok = Ssum[:, 3] > 0
    exp = (Ssum[ok, 1] * Ssum[ok, 2] / Ssum[ok, 3]).sum()
    return Ssum[ok, 0].sum() / exp if exp > 0 else np.nan


def enrichment_ci(lst, seed=0):
    Sx = np.asarray(lst); I = np.random.default_rng(seed).integers(0, len(Sx), (S.N_BOOT, len(Sx)))
    b = np.array([enrichment(Sx[ii].sum(0)) for ii in I])
    return float(enrichment(Sx.sum(0))), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def nmi(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
    C = np.zeros((len(ua), len(ub))); np.add.at(C, (ia, ib), 1); C /= C.sum()
    pa, pb = C.sum(1), C.sum(0); nz = C > 0
    I = (C[nz] * np.log(C[nz] / np.outer(pa, pb)[nz])).sum()
    Ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum(); Hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return 2 * I / (Ha + Hb) if Ha + Hb > 0 else 0.0


def nmi_gain_ci(R, l, n_boot=200, seed=0):
    """NMI(dominant mode, POS) at the layer's output minus at its input; cluster bootstrap over sequences."""
    lab = R['pos_lab']; o = R[f'L{l}/pos_dom_out']; n_in = R[f'L{l}/pos_dom_in']
    cut = np.concatenate([[0], np.cumsum(R['pos_seq_len'])])
    pt = nmi(o, lab) - nmi(n_in, lab)
    I = np.random.default_rng(seed).integers(0, len(cut) - 1, (n_boot, len(cut) - 1))
    bs = []
    for ii in I:
        sl = np.concatenate([np.arange(cut[q], cut[q + 1]) for q in ii])
        bs.append(nmi(o[sl], lab[sl]) - nmi(n_in[sl], lab[sl]))
    return float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
