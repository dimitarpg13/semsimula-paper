#!/usr/bin/env python3
"""Probe B: can tokens share their stiff basis? (Macroscopic_Limits note §5.2 and §8; mean-field programme Stage 5;
2026-11-17, with a causal variant added.)

    python3 shared_stiff_basis_probe.py OUT FOLDER [ckpt=<file>] [blocks=8,64,512] [windows=16] [threads=8]
                                        [val=<tokens.npy>] [name=<stem>]

Per layer step the model computes, per token, the stiffest modes of L_t = G_t G_t^T (lowrank_modes: U_t, kappa_t). A
block basis replaces them: U_b, the top-q eigenvectors of the pooled L over a block of b consecutive tokens, and
per token the exact eigenpairs of L_t restricted to span(U_b) (a q x q eigenproblem), U_t' = U_b V_t, kappa_t'. The
part of the force outside span(U_b) goes to the kick, as the truncation to q modes already does. Two bases:
  - pooled: from the tokens of the block itself. It reads later tokens of the block, so it is NOT causal; an upper
    bound on what sharing can keep;
  - causal: from the tokens of the previous block; the first block keeps its per-token modes. A decoder could use it.
Readings on the unperturbed run, per layer: the energy fraction |U_b^T G_t|_F^2 / |G_t|_F^2 and the largest principal
angle between span(U_t) and span(U_b) (medians, 10th and 90th percentiles). Then the loss with the block basis on
layer 0, on layer 1 and on both, paired by window against the trained model, and the time spent in the mode
computation per forward. Writes OUT/shared_basis_<stem>.txt and .json.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))

src = (DBG / 'copy_split.py').read_text()
src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', f"windows={OPTS.get('windows', 16)}", f"threads={OPTS.get('threads', 8)}"] \
    + ([f"val={OPTS['val']}"] if 'val' in OPTS else [])
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)

import verify_pm_switch as P
_build = P.build; got = {}
def build(*a, **k):
    r = _build(*a, **k); got['model'] = r[0]; return r
P.build = build
_, desc = CS['ladder_model'](FOLDER, OPTS.get('ckpt'))
model = got['model']
print('model:', desc, flush=True)
BLOCKS = [int(t) for t in OPTS.get('blocks', '8,64,512').split(',')]

mod = sys.modules[type(model)._layer_step_langevin.__module__]
orig_modes = mod.lowrank_modes
st = {'layer': None, 'mode': None, 'stats': None, 't_modes': 0.0}


def block_basis(G, b, q, causal):
    """G (B, T, d, P) -> per-token (U', kappa') from block bases; U_b from the block (pooled) or the previous one"""
    B, T, d, _ = G.shape
    nb = -(-T // b)
    pad = nb * b - T
    Gp = torch.nn.functional.pad(G, (0, 0, 0, 0, 0, pad)) if pad else G
    Gr = Gp.reshape(B, nb, b, d, -1)
    Lb = torch.einsum('bnsdp,bnsep->bnde', Gr, Gr)                          # pooled L over the block's tokens
    ev, evec = torch.linalg.eigh(Lb.double())
    Ub = evec[..., -q:].to(G.dtype)                                           # (B, nb, d, q)
    if causal:
        Ub = torch.cat([Ub[:, :1], Ub[:, :-1]], 1)                            # block j uses block j-1; block 0 is replaced below
    Ut = Ub.repeat_interleave(b, 1)[:, :T]                                    # (B, T, d, q)
    A = torch.einsum('btdq,btdp->btqp', Ut, G)
    lam, V = torch.linalg.eigh(torch.einsum('btqp,btrp->btqr', A, A).double())
    return Ut, torch.einsum('btdq,btqr->btdr', Ut, V.to(G.dtype)), lam.clamp(min=0).to(G.dtype)


N_RETRY = [0]
def modes(G, *a, **k):
    t0 = time.perf_counter()
    try:
        U, kappa = orig_modes(G, *a, **k)
    except RuntimeError:                  # a non-converging CPU SVD (code 'gram' driver): jittered retry, counted
        N_RETRY[0] += 1
        gj = torch.Generator().manual_seed(N_RETRY[0])
        sc = G.detach().abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        U, kappa = orig_modes(G + 1e-10 * sc * torch.randn(G.shape, generator=gj, dtype=G.dtype), *a, **k)
    L = st['layer']; q = U.shape[-1]
    if st['stats'] is not None:                       # readings on the unperturbed run
        r = st['stats'].setdefault(L, {})
        e_tot = (G * G).sum((-2, -1)).clamp_min(1e-30)
        for b in BLOCKS:
            for causal in (False, True):
                Ub, _, _ = block_basis(G.detach(), b, q, causal)
                frac = (torch.einsum('btdq,btdp->btqp', Ub, G) ** 2).sum((-2, -1)) / e_tot
                sv = torch.linalg.svdvals(torch.einsum('btdq,btdr->btqr', U, Ub))
                ang = torch.rad2deg(torch.arccos(sv.min(-1).values.clamp(-1, 1)))
                if causal:
                    frac, ang = frac[:, b:], ang[:, b:]
                key = f'b{b}_{"causal" if causal else "pooled"}'
                r.setdefault(key, {'frac': [], 'angle': []})
                r[key]['frac'].append(frac.flatten()); r[key]['angle'].append(ang.flatten())
    if st['mode'] is not None and L in st['mode'][0]:
        _, b, causal = st['mode']
        _, Un, kn = block_basis(G.detach(), b, q, causal)
        if causal:                                    # the first block keeps its own per-token modes
            Un[:, :b], kn[:, :b] = U[:, :b], kappa[:, :b]
        U, kappa = Un, kn
    st['t_modes'] += time.perf_counter() - t0
    return U, kappa
mod.lowrank_modes = modes

orig_step = model._layer_step_langevin
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    st['layer'] = int(layer_idx)
    return orig_step(h, h_prev, m_b, gamma, dt_, layer_idx)
model._layer_step_langevin = step

x, y = CS['windows']()
ARMS = [('base', None)] + [(f'{"L0" if ls == (0,) else "L1" if ls == (1,) else "both"}_b{b}_{"causal" if c else "pooled"}', (ls, b, c))
                           for b in BLOCKS for c in (False, True) for ls in ((0,), (1,), (0, 1))]
per = {a: [] for a, _ in ARMS}; tm = {a: 0.0 for a, _ in ARMS}; stats = {}
for i in range(0, len(x), 4):
    xb, yb = x[i:i + 4], y[i:i + 4]
    for a, m in ARMS:
        st['mode'] = m; st['stats'] = stats if a == 'base' else None; st['t_modes'] = 0.0
        with torch.enable_grad():         # arms built without the analytic V_phi force need the graph
            lg = model(torch.from_numpy(xb))[0].detach()
        ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), torch.from_numpy(yb).reshape(-1),
                                               reduction='none').view(len(xb), -1).mean(1)
        per[a] += ce.tolist(); tm[a] += st['t_modes']
    print(f'windows {i + 4}/{len(x)}', flush=True)

b0 = np.array(per['base'])
lines = [f'Shared stiff basis (Probe B): {FOLDER}', f'model {desc}',
         f'{len(x)} validation windows x 512 ({Path(CS["VAL"]).name}); blocks {BLOCKS}; base loss {b0.mean():.4f} '
         f'(ppl {np.exp(b0.mean()):.2f}); the mode computation took {tm["base"]:.1f} s over the base forwards; '
         f'mode-solver retries {N_RETRY[0]}',
         'pooled = basis from the block itself (reads later tokens of the block: not causal, an upper bound); '
         'causal = basis from the previous block', '', 'readings on the trained run (median, p10, p90):']
res = {'base_loss': float(b0.mean()), 'stats': {}, 'arms': {}}
for L in sorted(stats):
    for key, r in stats[L].items():
        f_ = torch.cat(r['frac']).float(); a_ = torch.cat(r['angle']).float()
        # a causal basis at b >= T has no previous block inside the window: nothing to read (nan)
        q = lambda t, p: (float(torch.quantile(t[torch.isfinite(t)][:2_000_000], p))
                          if torch.isfinite(t).any() else float('nan'))
        res['stats'][f'{L}:{key}'] = {'frac': [q(f_, .5), q(f_, .1), q(f_, .9)], 'angle_deg': [q(a_, .5), q(a_, .1), q(a_, .9)]}
        s_ = res['stats'][f'{L}:{key}']
        lines.append(f'  layer {L}  {key:>12}: energy fraction {s_["frac"][0]:.3f} ({s_["frac"][1]:.3f}, {s_["frac"][2]:.3f}); '
                     f'largest principal angle {s_["angle_deg"][0]:.1f} deg ({s_["angle_deg"][1]:.1f}, {s_["angle_deg"][2]:.1f})')
lines += ['', f'{"arm":>20}  {"loss":>7}  {"d nats":>8}  {"se":>7}  {"ppl change":>10}  {"mode time s":>11}']
for a, _ in ARMS:
    d = np.array(per[a]) - b0
    r = {'loss': float(np.mean(per[a])), 'd_nats': float(d.mean()), 'se': float(d.std(ddof=1) / np.sqrt(len(d))),
         'mode_time_s': tm[a]}
    res['arms'][a] = r
    lines.append(f'{a:>20}  {r["loss"]:7.4f}  {r["d_nats"]:+8.4f}  {r["se"]:7.4f}  {100 * np.expm1(r["d_nats"]):+9.2f}%  {tm[a]:11.1f}')
stem = OPTS.get('name', FOLDER[-40:])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f'shared_basis_{stem}.txt').write_text('\n'.join(lines) + '\n')
(OUT / f'shared_basis_{stem}.json').write_text(json.dumps({'folder': FOLDER, 'model': desc, **res}, indent=1))
print('\n'.join(lines))
