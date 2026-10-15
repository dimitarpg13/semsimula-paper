#!/usr/bin/env python3
"""
SY6: the same measurements on the matched GPT-2 (protocol §5.22; additions fixed
2026-10-15 before this script was built).

    python3 sy6_gpt2.py [n_seq=200]

The model is the matched GPT-2 of the ladder (d = 384, 8 layers, 6 heads, the same
data and token budget), built from the matched-baseline notebook's own Cell 3
(executed verbatim) and its best checkpoint.

Each layer is x -> x + attn(ln1 x) -> x + mlp(ln2 x). Attention is the only term
through which another token acts on token i, so it is the coupling. Each layer is
replayed from its captured input as trained, with attention switched off, and with
attention's output replaced by its average over the tokens (the common-mode
reference). Measures as for the Lagrangian arms: contraction S (on the residual
stream, translation-invariant) and the binding contrast B on centred states, per
grouping, with shuffled controls. Added (fixed in advance): the prediction-geometry
test B on the final states the readout reads (after the final normalization). The
coreference probe is sy6_gpt2_probe.py.

SY-GPT2 (60%): HIT if, for at least two of coreference, head-dependent and noun
chunks, GPT-2's attention-made binding at its last layer is at least the largest
coupling-made binding at the last layer over all Lagrangian arms and couplings.

Checks first: the checkpoint loads strictly; replaying a layer "on" reproduces it bit
for bit; logits rebuilt from the captured trajectory equal the model's.
(Restored 2026-10-15 after a smoke-test cleanup deleted it; content unchanged.)
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

N_SEQ = next((int(a[6:]) for a in sys.argv[1:] if a.startswith('n_seq=')), 200)
HERE = Path(__file__).resolve().parent
CACHE = HERE / 'sy_cache'
sys.path.insert(0, str(HERE))
import sy_instruments as S
from sy_groupings import GROUPINGS, Groupings, bin_stats_rel

LING = ('coref', 'dep', 'chunk', 'ent', 'pos')
CKPT = Path.home() / 'Downloads/semsimula_matched_gpt2_baseline_owt/checkpoints/gpt2_baseline_best.pt'
NB = HERE.parent / 'colab_matched_gpt2_baseline_openwebtext.ipynb'
lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))

# ---------------------------------------------------------------- build from the notebook's own Cell 3
nb = json.load(open(NB))
cell3 = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).lstrip().startswith('# ── Cell 3')][0]
g = {'torch': torch, 'nn': nn, 'F': F, 'math': math, '__name__': 'cell3'}
exec(compile(cell3, 'Cell3', 'exec'), g)
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
cfg = ck['config']
model = g['GPT2'](cfg['vocab_size'], cfg['d'], cfg['n_layers'], cfg['n_heads'], cfg['d_ff'], cfg['max_len'],
                  dropout=0.0, tie_embeddings=cfg['tie_embeddings'])
model.load_state_dict(ck['model_state_dict'], strict=True)
model.eval()
NL = len(model.blocks)

TOK = np.load(CACHE / 'sy_tokens.npz'); X = TOK['x'][:N_SEQ]; Y = TOK['y'][:N_SEQ]
ANN = np.load(CACHE / 'sy_annotation.npz')
GR = Groupings(ANN, X)

def embed(xb):
    T = xb.shape[1]
    return model.tok_emb(xb) + model.pos_emb(torch.arange(T).unsqueeze(0))

def block_step(blk, x, mode):
    a = blk.attn(blk.ln1(x))
    if mode == 'off':
        a = torch.zeros_like(a)
    elif mode == 'common':
        a = a.mean(dim=1, keepdim=True).expand_as(a)
    x = x + a
    return x + blk.mlp(blk.ln2(x))

@torch.no_grad()
def trajectory(xb):
    xs = [embed(xb)]
    for blk in model.blocks:
        xs.append(blk(xs[-1]))
    return xs                                     # xs[l] is the input to layer l; xs[NL] the last output

# ---------------------------------------------------------------- checks
say(f'SY6 on the matched GPT-2: d = {cfg["d"]}, {NL} layers, {cfg["n_heads"]} heads; checkpoint step {ck["step"]}, '
    f'recorded validation perplexity {ck["val_ppl"]:.2f}; checkpoint loaded strictly')
xb = torch.from_numpy(X[:4])
with torch.no_grad():
    logits_ref, _ = model(xb)
    xs = trajectory(xb)
    lg = model.ln_f(xs[-1]) @ model.tok_emb.weight.T
    d0 = float((lg - logits_ref).abs().max())
    dl = max(float((block_step(model.blocks[l], xs[l], 'on') - xs[l + 1]).abs().max()) for l in range(NL))
say(f'CHECK  logits rebuilt from the captured trajectory vs the model: max |d| {d0:.1e}; layer replay "on" vs the forward pass: max |d| {dl:.1e}')
assert d0 == 0.0 and dl == 0.0
with torch.no_grad():
    nll = np.mean([float(F.cross_entropy(model(torch.from_numpy(X[i:i + 4]))[0].flatten(0, 1),
                                         torch.from_numpy(Y[i:i + 4]).flatten())) for i in range(0, N_SEQ, 4)])
say(f'CHECK  perplexity on the {N_SEQ} SY sequences: {np.exp(nll):.2f} (recorded on the validation split: {ck["val_ppl"]:.2f})')

# ---------------------------------------------------------------- state-level measurement
store = {}
def put(k, v): store.setdefault(k, []).append(v)
rng = np.random.default_rng(20261015)
Wt = model.tok_emb.weight.detach().double().numpy()
t0 = time.time()
for b0 in range(0, N_SEQ, 2):
    xb = torch.from_numpy(X[b0:b0 + 2])
    with torch.no_grad():
        xs = trajectory(xb)
        vers = {(l, m): block_step(model.blocks[l], xs[l], m).double().numpy() for l in range(NL) for m in ('off', 'common')}
        fin = model.ln_f(xs[-1])
        logp = torch.log_softmax(fin @ model.tok_emb.weight.T, -1)
    for j in range(xb.shape[0]):
        i = b0 + j
        G, Gs = GR(i, rng)
        for l in range(NL):
            Hin = xs[l][j].double().numpy(); Hon = xs[l + 1][j].double().numpy()
            cos_in = S.cos_matrix(S.centred(Hin)); cos_on = S.cos_matrix(S.centred(Hon))
            for name in LING:
                put(f'L{l}/{name}/B_in', bin_stats_rel(cos_in, *G[name]))
                put(f'L{l}/{name}/B_on', bin_stats_rel(cos_on, *G[name]))
                put(f'L{l}/{name}/shuf/B_on', bin_stats_rel(cos_on, *Gs[name]))
            for m in ('off', 'common'):
                Hr = vers[(l, m)][j]
                put(f'L{l}/{m}/S', S.coherence(S.contraction_matrix(Hon, Hr)))
                cos_r = S.cos_matrix(S.centred(Hr))
                for name in LING:
                    put(f'L{l}/{name}/{m}/B_ref', bin_stats_rel(cos_r, *G[name]))
                    put(f'L{l}/{name}/{m}/shuf/B_ref', bin_stats_rel(cos_r, *Gs[name]))
        # test B on the final states
        H = fin[j].double().numpy(); cosS = S.cos_matrix(S.centred(H))
        q = torch.exp(0.5 * logp[j]).double(); BC = (q @ q.T).numpy()
        tgt = S.cos_matrix(S.centred(Wt[Y[i]]))
        iu = np.triu_indices(len(BC), 1)
        coef, *_ = np.linalg.lstsq(np.stack([np.ones(len(iu[0])), BC[iu]], 1), cosS[iu], rcond=None)
        R = cosS - (coef[0] + coef[1] * BC)
        put('fit', np.array([coef[0], coef[1], np.corrcoef(BC[iu], cosS[iu])[0, 1]]))
        for name in LING:
            for tag, rel in (('', G[name]), ('shuf/', Gs[name])):
                for k, M in (('B_state', cosS), ('B_pred', BC), ('B_resid', R), ('B_target', tgt)):
                    put(f'pred/{name}/{tag}{k}', bin_stats_rel(M, *rel))
    if (b0 // 2) % 20 == 0:
        print(f'   {b0 + 2}/{N_SEQ} sequences, {(time.time() - t0) / 60:.1f} min', flush=True)
A = {k: np.asarray(v) for k, v in store.items()}
np.savez_compressed(HERE / 'sy6_gpt2.npz', **A)
D = lambda a, b: S.boot_contrast_diff(A[a], A[b])

say()
say(f'STATE LEVEL ({N_SEQ} sequences; 95% cluster-bootstrap intervals; floor {S.FLOOR}); attention as the coupling')
spec_last = {}
for l in range(NL):
    s_c = S.boot_mean(A[f'L{l}/common/S']); s_o = S.boot_mean(A[f'L{l}/off/S'])
    say(f'layer {l}: S vs common {S.fmt(s_c)}   S vs off {S.fmt(s_o)}')
    for name in LING:
        bi = S.boot_contrast(A[f'L{l}/{name}/B_in'])[0]; bo = S.boot_contrast(A[f'L{l}/{name}/B_on'])[0]
        bs = D(f'L{l}/{name}/B_on', f'L{l}/{name}/common/B_ref'); bss = D(f'L{l}/{name}/shuf/B_on', f'L{l}/{name}/common/shuf/B_ref')
        say(f'   {name:<6} B_in {bi:+.4f}  B_on {bo:+.4f}   attention-made B_spec {S.fmt(bs)}  (shuffled {S.fmt(bss)})')
        if l == NL - 1: spec_last[name] = bs

say()
f = A['fit']
say(f'TEST B, prediction geometry on the final states (median slope {np.median(f[:, 1]):+.3f}, median correlation {np.median(f[:, 2]):+.3f})')
for name in LING:
    v = {k: S.boot_contrast(A[f'pred/{name}/{k}']) for k in ('B_state', 'B_pred', 'B_resid', 'B_target')}
    say(f'   {name:<6} state {S.fmt(v["B_state"])}  prediction {S.fmt(v["B_pred"])}  residual {S.fmt(v["B_resid"])}  true next words {S.fmt(v["B_target"])}')

# ---------------------------------------------------------------- SY-GPT2: against the best Lagrangian arm
NUM = json.loads((HERE.parents[3] / 'companion_notes/figures/sy_results/_numbers.json').read_text())
say()
say('SY-GPT2: attention-made binding at GPT-2\'s last layer vs the best coupling-made binding at the last layer of any Lagrangian arm')
wins = 0; cmp = {}
for name in ('coref', 'dep', 'chunk'):
    best = max(((k, v[0]) for k, v in NUM.items() if k.endswith(f'/{name}') and '/Bspec/' in k), key=lambda t: t[1])
    gv = spec_last[name][0]; w = gv >= best[1]; wins += w
    cmp[name] = {'gpt2': spec_last[name], 'best_lagrangian': best}
    say(f'   {name:<6} GPT-2 {gv:+.4f}   best Lagrangian {best[1]:+.4f} ({best[0]})   {"GPT-2 at least as large" if w else "smaller"}')
say(f'   => SY-GPT2 {"HIT" if wins >= 2 else "MISS"} ({wins} of 3)')
(HERE / 'sy6_gpt2_output.txt').write_text('\n'.join(lines) + '\n')
(HERE / 'sy6_gpt2.json').write_text(json.dumps({'sy_gpt2': cmp, 'wins': int(wins)}, default=float, indent=1))
print('wrote sy6_gpt2_output.txt')
