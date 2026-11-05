#!/usr/bin/env python3
"""How many wells does a corpus need? Softmax-gated mixtures of K linear maps fitted on a trained model's bank input (2026-11-03).

    python3 well_count_fit.py OUT FOLDER NAME [ckpt=<file>] [windows=256] [threads=8] [val=<tokens.npy>] [layer=0]
                              [Ks=1,2,4,8] [depths=1,2] [epochs=4] [evals=2] [batch=2048] [lr=3e-4] [wd=0.05] [obj=ce|kl] [cache=<dir>]
                              [also=<layers>]

A V_theta well's centre is linear in the bank input x (mu_k = M_k x + m_k, x = the context channels plus the depth
code) and its weight is softmax(W_w x + b)_k. A layer whose wells are stiff puts the state at their weighted
centre, so what K wells can express is a softmax-gated mixture of K linear maps (a mixture of linear experts):

    h(x) = sum_k softmax(G x + g)_k (A_k x + c_k).

One well whose centre follows the context already reaches any target linear in x; more wells buy a
piecewise-linear fit of the rest. The estimate of K_needed(eps): fit this form for K in Ks, read out through
the trained model's own frozen readout (the states projected onto the sphere |h| = sqrt(d) when the model's
final states lie on it), and take the fewest K whose held-out loss is within eps of the best K's. depths=D stacks
D token-local blocks, h_{j+1} = h_j + mix_j([x, LN(h_j)]) (block 0 sees x alone), for the wells-against-depth
trade; it is a relaxation: the model's deeper layers also mix positions through their own context channels.

Capture: the bank input at layer `layer` (default 0: the context of the embeddings, the nearest thing to a
corpus statistic), the final state h_L and the targets, on `windows` validation windows of 512 tokens spaced
evenly over the set (copy_split.windows); cached as <cache>/wcf_<NAME>_L<layer>_w<windows>.pt. also=<layers> records
those layers' bank inputs in the same pass, one cache each, for later runs with layer=<one of them>. A window whose
forward fails (2026-11-03: CPU LAPACK svd non-convergence in cfc_baoab._gram_eigh on L8-CODE) is retried alone and
dropped if it fails again; the dropped windows are printed and noted in the cache. Windows are split
70/15/15 into train, validation (early stopping) and test (reported). Each fit starts from the previous K's
(every expert split in two with a small perturbation, the gate rows copied: the same function at the start),
and a deeper stack from the shallower one with the new blocks at zero output. obj=ce fits the next token,
obj=kl the trained model's own distribution (both report both). Inference of the model only; the fits on CPU.
AdamW with weight decay, the validation objective `evals` times per epoch and the best point kept (2026-11-03: without
the decay, at lr 1e-3 and one validation per epoch, the fits overfitted 358 windows from epoch 1-2 and the warm start
was kept unchanged for K >= 4). Each "vs prev K" difference carries its paired per-window standard error (the test
windows' mean differences), so the K_needed table says what the sample resolves.
Writes OUT/well_count_fit_<NAME>[_L<layer>].txt.
"""
import math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(sys.argv[1]).resolve(); FOLDER, NAME = sys.argv[2], sys.argv[3]
OPTS = dict(a.split('=', 1) for a in sys.argv[4:] if '=' in a)
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))
LAYER = int(OPTS.get('layer', 0)); WIN = int(OPTS.get('windows', 256))
KS = [int(k) for k in OPTS.get('Ks', '1,2,4,8').split(',')]
DEPTHS = [int(k) for k in OPTS.get('depths', '1,2').split(',')]
EPOCHS, BATCH, LR = int(OPTS.get('epochs', 4)), int(OPTS.get('batch', 2048)), float(OPTS.get('lr', 3e-4))
WD, EVALS = float(OPTS.get('wd', 0.05)), int(OPTS.get('evals', 2))   # 2026-11-03: AdamW weight decay; validations per epoch
OBJ = OPTS.get('obj', 'ce'); assert OBJ in ('ce', 'kl'), OBJ
CACHE = Path(OPTS.get('cache', Path.home() / 'Downloads/semsimula_wcf_cache')).resolve()
cache = CACHE / f'wcf_{NAME}_L{LAYER}_w{WIN}.pt'
assert all(b % a == 0 for a, b in zip(KS, KS[1:])), f'each K must divide the next (warm start by splitting): {KS}'

# copy_split's own Cell-0-from-tag build and validation windows, without its command dispatch
src = (DBG / 'copy_split.py').read_text()
src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', f'windows={WIN}', f"threads={OPTS.get('threads', 8)}"] \
    + ([f"val={OPTS['val']}"] if 'val' in OPTS else [])
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)


def build():
    """The trained model, as vtheta_well_probe.py builds it (a TinyStories folder through stage0_models)."""
    import verify_pm_switch as P
    _build = P.build; got = {}
    def b(*a, **k):
        r = _build(*a, **k); got['model'] = r[0]; return r
    P.build = b
    if '_baoab_ts_' in FOLDER:
        import importlib.util, tempfile
        sp = importlib.util.spec_from_file_location('stage0_models', DBG.parents[1] / 'first_order_ablation/fo_series/stage0_models.py')
        s0 = importlib.util.module_from_spec(sp)
        argv = list(sys.argv); sys.argv = [sys.argv[0], tempfile.mkdtemp()]
        sp.loader.exec_module(s0); sys.argv = argv
        model, _, sd = s0.build_cfc_ts(FOLDER, (), ckpt_name=OPTS.get('ckpt'), return_g=True)
        if 'val' not in OPTS:
            CS['VAL'] = next(v for v in (Path.home() / 'Downloads' / FOLDER / 'data/tinystories_val_2M.npy',
                                         Path.home() / 'Downloads/semsimula_tinystories_gpt2/data/tinystories_val_2M.npy') if v.exists())
        desc = f"{OPTS.get('ckpt') or '<prefix>_best.pt'} (step {sd.get('step')})"
    else:
        _, desc = CS['ladder_model'](FOLDER, OPTS.get('ckpt'))
        model = got['model']
    model.to('cpu'); model.eval(); model.cfg.use_layer_checkpoint = False
    return model, f"{desc}; validation {CS['VAL']}"


model, desc = build()
W_out = (model.lm_head.weight if model.lm_head is not None else model.E.weight).detach().float()
b_out = model.out_bias.detach().float() if model.out_bias is not None else None
d = W_out.shape[1]
readout = lambda h: F.linear(h, W_out, b_out)

if cache.exists():
    C = torch.load(cache)
    assert C['folder'] == FOLDER and C['val'] == str(CS['VAL']), (C['folder'], C['val'])
else:
    # also=<layers>: capture those layers' bank inputs in the same forwards and cache them too (one pass for several fits)
    LAYERS = sorted({LAYER} | {int(l) for l in OPTS.get('also', '').split(',') if l})
    vt = model.V_theta
    bank = vt.bank.bank if hasattr(vt.bank, 'bank') else vt.bank
    cur, got = {}, {}
    orig_cc = bank.context_components
    def cc(xi):
        g = vt._active_layer % vt.n_layers
        if g in LAYERS and g not in cur:             # the first call at the layer in this forward (later ones repeat it)
            cur[g] = xi.detach().float().reshape(-1, xi.shape[-1])
        return orig_cc(xi)
    bank.context_components = cc
    orig_lg = model.compute_logits
    def lg(h):
        cur['h'] = h.detach().float().reshape(-1, h.shape[-1])
        return orig_lg(h)
    model.compute_logits = lg
    x, y = CS['windows']()
    xs, hs, nl, kept, t0 = {g: [] for g in LAYERS}, [], [], [], time.time()

    def run(j):
        """Windows j: (bank inputs by layer, h, nll), or None where CPU LAPACK's SVD in the model's Gram step fails."""
        cur.clear()
        try:
            with torch.enable_grad():                # the V_phi pair force is an autograd gradient
                lgt = model(torch.from_numpy(x[j]))[0].detach()
        except torch.linalg.LinAlgError:
            return None
        nll = F.cross_entropy(lgt.reshape(-1, lgt.shape[-1]).float(), torch.from_numpy(y[j]).reshape(-1), reduction='none')
        return {g: cur[g] for g in LAYERS}, cur['h'], nll

    for i in range(0, len(x), 4):
        # 2026-11-03: L8-CODE hit "linalg.svd ... failed to converge" on CPU in one batch of four windows (late in the
        # set); retry such a batch one window at a time and drop only the window that fails, counted below
        parts = [(list(range(i, min(i + 4, len(x)))), run(slice(i, i + 4)))]
        if parts[0][1] is None:
            parts = [([j], run(slice(j, j + 1))) for j in range(i, min(i + 4, len(x)))]
        for idx, r in parts:
            if r is None:
                continue
            for g in LAYERS:
                xs[g].append(r[0][g].half())
            hs.append(r[1]); nl.append(r[2]); kept += idx
        if i % 64 == 0:
            print(f'capture: windows {i + 4}/{len(x)}  ({time.time() - t0:.0f} s)', flush=True)
    bank.context_components, model.compute_logits = orig_cc, orig_lg
    dropped = len(x) - len(kept)
    if dropped:
        print(f'capture: dropped {dropped} window(s) where the CPU SVD failed: {sorted(set(range(len(x))) - set(kept))}', flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    for g in LAYERS:
        Cg = dict(x=torch.cat(xs[g]), h=torch.cat(hs), y=torch.from_numpy(y[kept]).reshape(-1), nll=torch.cat(nl),
                  folder=FOLDER, val=str(CS['VAL']), T=x.shape[1],
                  desc=desc + (f'; {dropped} of {len(x)} windows dropped (CPU SVD failed)' if dropped else ''))
        torch.save(Cg, CACHE / f'wcf_{NAME}_L{g}_w{WIN}.pt')
        if g == LAYER:
            C = Cg
    print(f'captured {len(C["y"])} tokens at layers {LAYERS} in {time.time() - t0:.0f} s -> {cache}', flush=True)

torch.manual_seed(0)
X, H, Y, T = C['x'], C['h'], C['y'], C['T']
n_win = len(Y) // T
perm = torch.randperm(n_win, generator=torch.Generator().manual_seed(0))
split = {'train': perm[:int(.7 * n_win)], 'val': perm[int(.7 * n_win):int(.85 * n_win)], 'test': perm[int(.85 * n_win):]}
idx = {k: (v[:, None] * T + torch.arange(T)).reshape(-1) for k, v in split.items()}
def _moments(ii, chunk=65536):                     # 2026-11-03: in chunks; X stays half (a 1,024-window cache is 2 GB)
    n = 0; s1 = torch.zeros(X.shape[1], dtype=torch.float64); s2 = torch.zeros_like(s1)
    for i in range(0, len(ii), chunk):
        xb = X[ii[i:i + chunk]].double(); n += len(xb); s1 += xb.sum(0); s2 += (xb * xb).sum(0)
    m = s1 / n
    return m.float(), (s2 / n - m * m).clamp_min(0).sqrt().float().clamp_min(1e-6)
mean, std = _moments(idx['train'])
Xs = lambda ii: (X[ii].float() - mean) / std       # an affine change of variables: the fitted family is unchanged
hn = H.norm(dim=-1)
SPHERE = bool((hn.std() / hn.mean()) < 0.05 and abs(float(hn.mean()) / math.sqrt(d) - 1) < 0.05)
proj = (lambda h: F.layer_norm(h, (d,))) if SPHERE else (lambda h: h)


@torch.no_grad()
def model_logp(ii):
    return torch.log_softmax(readout(H[ii]), -1)


class Mix(nn.Module):
    """h = sum_k softmax(G u + g)_k (A_k u + c_k), u of size din."""
    def __init__(s, din, K, zero=False):
        super().__init__()
        s.K = K
        s.A = nn.Parameter(torch.zeros(din, K * d) if zero else torch.randn(din, K * d) / math.sqrt(din))
        s.c = nn.Parameter(torch.zeros(K, d))
        s.G = nn.Linear(din, K)
        nn.init.normal_(s.G.weight, std=0.01 / math.sqrt(din)); nn.init.zeros_(s.G.bias)

    def forward(s, u):
        w = torch.softmax(s.G(u), -1)
        e = (u @ s.A).view(-1, s.K, d) + s.c
        return torch.einsum('nk,nkd->nd', w, e), w

    def split(s, m):
        """The same function with K*m experts: each copied m times (a small perturbation breaks the tie)."""
        new = Mix(s.A.shape[0], s.K * m)
        with torch.no_grad():
            A = s.A.view(-1, s.K, d).repeat_interleave(m, 1)
            new.A.copy_((A + 1e-3 * A.std() * torch.randn_like(A)).reshape(s.A.shape[0], -1))
            new.c.copy_(s.c.repeat_interleave(m, 0))
            new.G.weight.copy_(s.G.weight.repeat_interleave(m, 0) + 1e-3 * torch.randn(s.K * m, s.A.shape[0]) / math.sqrt(s.A.shape[0]))
            new.G.bias.copy_(s.G.bias.repeat_interleave(m, 0))
        return new


class Stack(nn.Module):
    def __init__(s, blocks):
        super().__init__(); s.blocks = nn.ModuleList(blocks)

    def forward(s, x):
        h, ws = None, []
        for j, b in enumerate(s.blocks):
            u = x if j == 0 else torch.cat([x, F.layer_norm(h, (d,))], -1)
            dh, w = b(u); ws.append(w)
            h = dh if j == 0 else h + dh
        return proj(h), ws


def losses(net, ii, want=('ce', 'kl')):
    h, ws = net(Xs(ii))
    lp = torch.log_softmax(readout(h), -1)
    ce = F.nll_loss(lp, Y[ii]) if 'ce' in want else None
    if 'kl' in want:
        mlp = model_logp(ii)
        kl = (mlp.exp() * (mlp - lp)).sum(-1).mean()
    else:
        kl = None
    return ce, kl, ws


@torch.no_grad()
def evaluate(net, part, want=('ce', 'kl')):
    tot = {'ce': 0., 'kl': 0.}; ne = []
    ii_all = idx[part]
    for i in range(0, len(ii_all), 4096):
        ii = ii_all[i:i + 4096]
        ce, kl, ws = losses(net, ii, want)
        tot['ce'] += float(ce) * len(ii) if ce is not None else 0.
        tot['kl'] += float(kl) * len(ii) if kl is not None else 0.
        ne.append(torch.stack([torch.exp(-(w * w.clamp_min(1e-30).log()).sum(-1)) for w in ws], -1))
    ne = torch.cat(ne)
    return tot['ce'] / len(ii_all), tot['kl'] / len(ii_all), ne.median(0).values.tolist()


def fit(net, label):
    """AdamW, cosine schedule; the validation objective EVALS times per epoch, the best state kept (2026-11-03: the
    first version, plain Adam at lr 1e-3 validated once per epoch, overfitted from epoch 1-2 on 358 windows, so for
    K >= 4 the warm start was kept unchanged and the K-dependence was unreadable). Returns the best point in epochs."""
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    tr = idx['train']; per_ep = math.ceil(len(tr) / BATCH); steps = EPOCHS * per_ep
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    key = 0 if OBJ == 'ce' else 1
    val = lambda: evaluate(net, 'val', want=(OBJ,))[key]
    best = (val(), {k: v.clone() for k, v in net.state_dict().items()}, 0.0)
    t0 = time.time(); every = max(per_ep // EVALS, 1); s = 0
    for ep in range(1, EPOCHS + 1):
        for ii in tr[torch.randperm(len(tr))].split(BATCH):
            ce, kl, _ = losses(net, ii, want=(OBJ,))
            opt.zero_grad(); (ce if OBJ == 'ce' else kl).backward(); opt.step(); sched.step(); s += 1
            if s % every == 0 or s == steps:
                v = val()
                if v < best[0]:
                    best = (v, {k: t.clone() for k, t in net.state_dict().items()}, s / per_ep)
                print(f'  {label} epoch {s / per_ep:.2f}: val {OBJ} {v:.4f}  ({time.time() - t0:.0f} s)', flush=True)
    net.load_state_dict(best[1])
    return best[2]


@torch.no_grad()
def test_nll(net):                                  # per-token test loss, for the paired per-window SE of differences
    out = []
    for i in range(0, len(idx['test']), 4096):
        ii = idx['test'][i:i + 4096]; h, _ = net(Xs(ii)); out.append(F.cross_entropy(readout(h), Y[ii], reduction='none'))
    return torch.cat(out)


n_par = lambda net: sum(p.numel() for p in net.parameters())
def _model_nll(part):                              # 2026-11-03: in chunks (all test logits at once was the 26 GB peak)
    return torch.cat([F.nll_loss(model_logp(idx[part][i:i + 4096]), Y[idx[part][i:i + 4096]], reduction='none')
                      for i in range(0, len(idx[part]), 4096)])
mce = {p: float(_model_nll(p).mean()) for p in ('val', 'test')}
chk = float((_model_nll('test') - C['nll'][idx['test']]).abs().max())
assert chk < 1e-3, f'the frozen readout does not reproduce the model loss: max |d nll| {chk}'
din = X.shape[1]
hdr = [f'well-count fit, {NAME}: softmax-gated mixtures of K linear maps on the layer-{LAYER} bank input',
       f'{FOLDER}', f"{C['desc']}",
       f'{n_win} windows x {T} tokens (train/val/test {len(split["train"])}/{len(split["val"])}/{len(split["test"])} windows); '
       f'bank input {din} dims; d = {d}; final states |h| median {float(hn.median()):.2f} (sqrt d = {math.sqrt(d):.2f}) -> '
       f'{"projected onto the sphere" if SPHERE else "not projected"}',
       f'objective {OBJ}; {EPOCHS} epochs, batch {BATCH}, AdamW lr {LR} cosine, weight decay {WD}; validation {EVALS}x per epoch, best point kept',
       f'trained model on these tokens: test ce {mce["test"]:.4f} (ppl {math.exp(mce["test"]):.2f}), val ce {mce["val"]:.4f}; '
       f'the frozen readout on the captured final states reproduces its per-token loss to {chk:.1e}', '',
       f'{"D":>2} {"K":>3} {"params":>10} {"test ce":>8} {"ppl":>8} {"vs model":>9} {"KL->model":>9} {"vs prev K":>9} {"pair SE":>7} {"best ep":>7}  gate n_eff medians']
print('\n'.join(hdr), flush=True)
rows, res, prev_depth, ses = [], {}, {}, []
for D in DEPTHS:
    net, prev, prev_nll = None, None, None
    for K in KS:
        if net is None:
            if D == DEPTHS[0] or (DEPTHS[DEPTHS.index(D) - 1], K) not in prev_depth:
                net = Stack([Mix(din, K)] + [Mix(din + d, K, zero=True) for _ in range(D - 1)])
            else:                                     # the shallower stack at this K, new blocks at zero output
                base = prev_depth[DEPTHS[DEPTHS.index(D) - 1], K]
                net = Stack(list(base.blocks) + [Mix(din + d, K, zero=True) for _ in range(D - len(base.blocks))])
        else:
            net = Stack([b.split(K // b.K) for b in net.blocks])
        ep = fit(net, f'D={D} K={K}')
        ce, kl, ne = evaluate(net, 'test')
        nll = test_nll(net)
        res[D, K] = ce; prev_depth[D, K] = Stack([b for b in net.blocks])
        if prev is None:
            dp = ''
        else:
            se = float((nll - prev_nll).view(-1, T).mean(1).std() / math.sqrt(len(split['test'])))
            ses.append(se); dp = f'{ce - prev:+9.4f} {se:7.4f}'
        rows.append(f'{D:2d} {K:3d} {n_par(net):10,d} {ce:8.4f} {math.exp(ce):8.2f} {ce - mce["test"]:+9.4f} {kl:9.4f} {dp:>17} {ep:7.2f}  '
                    + ' '.join(f'{v:.2f}' for v in ne))
        print(rows[-1], flush=True)
        prev, prev_nll = ce, nll

tail = ['', 'K_needed(eps) at each depth: the fewest K whose test ce is within eps nats of the best K at that depth']
for D in DEPTHS:
    best = min(res[D, K] for K in KS)
    tail.append(f'  D = {D}: ' + ', '.join(f'eps {e}: K = {next(K for K in KS if res[D, K] <= best + e)}' for e in (0.005, 0.01, 0.02, 0.05)))
if ses:
    tail.append(f'paired per-window SE of the "vs prev K" differences: median {sorted(ses)[len(ses) // 2]:.4f}, max {max(ses):.4f}; '
                f'a difference under about twice the SE is not resolved by {len(split["test"])} test windows')
txt = '\n'.join(hdr + rows + tail)
print('\n'.join(tail))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"well_count_fit_{NAME}{'' if LAYER == 0 else f'_L{LAYER}'}.txt").write_text(txt + '\n')
