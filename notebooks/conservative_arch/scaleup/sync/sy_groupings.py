"""Groupings of the SY programme (protocol §5.22), shared by sy_routes.py and later scripts.

The code is sy_measure.py's, moved here unchanged (sy_measure.py keeps its own copy because it
was already running when this module was made; tested equal on the evaluation sequences).
"""
import numpy as np
import sy_instruments as S

GROUPINGS = ('coref', 'dep', 'chunk', 'ent', 'pos', 'repeat')


class Groupings:
    def __init__(self, ANN, X):
        global _ANN, _X, POS, EXCL_PUNCT, IU, BIN
        _ANN, _X = ANN, X
        POS = np.arange(X.shape[1])
        EXCL_PUNCT = {list(ANN['pos_tags']).index(t) for t in ('PUNCT', 'SPACE', 'X', 'SYM')}
        IU = np.triu_indices(len(POS), 1)
        BIN = np.searchsorted(S.EDGES, np.abs(POS[IU[0]] - POS[IU[1]]), side='right') - 1

    def __call__(self, i, rng):
        return groupings(i, rng)

    def pos_labels(self, i):
        word = _ANN['word'][i]; punct = _ANN['punct'][i] == 1
        gp = np.where((word >= 0) & ~punct, _ANN['pos'][i], -1)
        gp[np.isin(gp, list(EXCL_PUNCT))] = -1
        return gp



def rel_gid(gid):
    lab = gid >= 0
    same = (gid[:, None] == gid[None, :])
    return same & lab[:, None] & lab[None, :], (~same) & lab[:, None] & lab[None, :]

def groupings(i, rng):
    """{name: (related, unrelated)} boolean N x N, for the real labels and a shuffled control."""
    word = _ANN['word'][i]; posc = _ANN['pos'][i]; head = _ANN['head'][i]; punct = _ANN['punct'][i] == 1
    same_word = (word[:, None] == word[None, :]) & (word[:, None] >= 0)
    content = (word >= 0) & ~punct
    out, shuf = {}, {}
    def put(name, rel, unr, rel_s, unr_s):
        out[name] = (rel & ~same_word, unr & ~same_word)
        shuf[name] = (rel_s & ~same_word, unr_s & ~same_word)
    def permuted(gid):
        g2 = gid.copy(); m = gid >= 0
        g2[m] = rng.permutation(gid[m]); return g2
    # coreference: chains, pairs within one mention excluded
    cr = _ANN['coref'][i]; mn = _ANN['mention'][i]
    same_m = (mn[:, None] == mn[None, :]) & (mn[:, None] >= 0)
    r, u = rel_gid(cr); rs, us = rel_gid(permuted(cr))
    put('coref', r & ~same_m, u, rs & ~same_m, us)
    # dependencies: head-dependent word pairs among content words
    wid_ok = content
    def dep_rel(head_of_word):
        hw = np.full(word.max() + 2 if word.max() >= 0 else 1, -1)
        for t in range(len(word)):
            if word[t] >= 0: hw[word[t]] = head_of_word.get(word[t], -1)
        wt, ws = word[:, None], word[None, :]
        rel = ((hw[np.maximum(wt, 0)] == ws) | (hw[np.maximum(ws, 0)] == wt)) & (wt >= 0) & (ws >= 0) & (wt != ws)
        both = wid_ok[:, None] & wid_ok[None, :]
        return rel & both, (~rel) & both
    heads = {int(word[t]): int(head[t]) for t in range(len(word)) if word[t] >= 0 and content[t] and head[t] != word[t]}
    r, u = dep_rel(heads)
    ws = list(heads); offs = [heads[w] - w for w in ws]; perm = rng.permutation(len(ws))
    heads_s = {w: w + offs[perm[k]] for k, w in enumerate(ws)}
    rs, us = dep_rel(heads_s)
    put('dep', r, u, rs, us)
    for name, key in (('chunk', 'chunk'), ('ent', 'ent')):
        gid = _ANN[key][i]
        r, u = rel_gid(gid); rs, us = rel_gid(permuted(gid))
        put(name, r, u, rs, us)
    gp = np.where(content, posc, -1)
    gp[np.isin(gp, list(EXCL_PUNCT))] = -1
    r, u = rel_gid(gp); rs, us = rel_gid(permuted(gp))
    put('pos', r, u, rs, us)
    ids = _X[i]; vals, cnt = np.unique(ids, return_counts=True)
    rep = np.where(np.isin(ids, vals[cnt >= 2]), ids, -1)
    r, u = rel_gid(rep); rs, us = rel_gid(permuted(rep))
    put('repeat', r, u, rs, us)
    return out, shuf


def bin_stats_rel(M, rel, unr):
    r = rel[IU]; u = unr[IU]; b = BIN
    v = M[IU]
    out = np.zeros((len(S.EDGES) - 1, 4))
    np.add.at(out[:, 0], b[r], v[r]); np.add.at(out[:, 1], b[r], 1)
    np.add.at(out[:, 2], b[u], v[u]); np.add.at(out[:, 3], b[u], 1)
    return out


