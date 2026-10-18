#!/usr/bin/env python3
"""
How large is the router term in the trained V_phi force? (2026-10-18; prerequisite for the analytic
V_phi force, Eliminating_the_Autograd_Dependency_in_VPhi.md)

The force is f = -grad_h U_pair, taken by autograd. U_pair = s_l * sum_j V_phi(h_t, h_j) * m_g with the
straight-through gate m_g = stop_grad(m_hard - k y) + k y, where y = softmax(score_head(h_t, .) / tau).
The forward value of m_g is the hard mask, but its gradient path through y is live in eval as in
training, so the force contains a router term  -s_l * sum_j V_phi(h_t, h_j) * k * grad_h y_tj  beside
the pair term  -s_l * sum_j grad_h V_phi(h_t, h_j).  The note's section 5.1 assumed the router term
vanishes at eval. This script measures it on the trained checkpoints:

  per layer: |router term| / |pair term| and / |V_theta force|, from the model's own force code with
  the gate's gradient path on and off (m_g -> m_g.detach(), same forward value);
  end to end: logits and loss with the router term removed from the force at every layer.

    python3 vphi_ste_term.py [ARM ...] [n_seq=8]       ARM in F3.1 SR2 G2 PM1 (default: F3.1 SR2 G2)
"""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'sync'))
from sy_replay import Replayer, CACHE

args = [a for a in sys.argv[1:] if '=' not in a] or ['F3.1', 'SR2', 'G2']
N_SEQ = next((int(a[6:]) for a in sys.argv[1:] if a.startswith('n_seq=')), 8)
torch.set_num_threads(8)
X = torch.from_numpy(np.load(CACHE / 'sy_tokens.npz')['x'][:N_SEQ]).long()
lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
say(f'V_phi router (straight-through) term in the trained force; {N_SEQ} sequences of 512 tokens')

for arm in args:
    R = Replayer(arm); m = R.model; m.eval()
    m.cfg.use_layer_checkpoint = False     # forward-only measurement: same values, and the force can be taken twice
    if m.V_phi is None:
        say(f'\n{arm}: no V_phi'); continue
    topk0 = m._sparse_topk_indices
    state = {'detach': False}

    def topk(pi, causal, T):
        idx, m_g = topk0(pi, causal, T)
        return (idx, m_g.detach()) if state['detach'] else (idx, m_g)
    m._sparse_topk_indices = topk
    lf0 = m._layer_forces
    rec = {}

    def lf(h_in, xis, layer_idx, *, split=False, vtheta_comps=None):
        out = lf0(h_in, xis, layer_idx, split=True, vtheta_comps=vtheta_comps)
        if not state['detach']:                       # the model's own force, plus a twin without the router path
            state['detach'] = True
            try:
                _, fp_no = lf0(h_in, xis, layer_idx, split=True, vtheta_comps=vtheta_comps)
            finally:
                state['detach'] = False
            ft, fp = out
            r = rec.setdefault(layer_idx, {'router': [], 'pair': [], 'theta': [], 'cos': []})
            router = (fp - fp_no).detach(); pair = fp_no.detach()
            r['router'].append(router.norm(dim=-1).flatten()); r['pair'].append(pair.norm(dim=-1).flatten())
            r['theta'].append(ft.detach().norm(dim=-1).flatten() if ft is not None else torch.zeros(1))
            r['cos'].append(torch.nn.functional.cosine_similarity(router, pair, dim=-1).flatten())
        ft, fp = out
        return out if split else (fp if ft is None else ft + fp)
    m._layer_forces = lf

    def nll(detach_all):
        state['detach'] = detach_all
        tot = []
        try:
            for b in range(0, N_SEQ, 2):
                xb = X[b:b + 2]
                with torch.enable_grad():
                    lo = m(xb[:, :-1])[0].detach()
                tot.append(torch.nn.functional.cross_entropy(lo.reshape(-1, lo.shape[-1]), xb[:, 1:].reshape(-1), reduction='none'))
                if not detach_all:
                    full_logits.append(lo)
                else:
                    no_logits.append(lo)
        finally:
            state['detach'] = False
        return torch.cat(tot).mean().item()

    full_logits, no_logits = [], []
    rec.clear()
    l_full = nll(False)
    stats = {l: {k: torch.cat(v) for k, v in d.items()} for l, d in rec.items()}
    l_no = nll(True)
    d_lo = max(float((a - b).abs().max()) for a, b in zip(full_logits, no_logits))
    say(f'\n{arm}  (per-layer V_phi scale s_l: '
        f'{[round(float(m.per_layer_scale(l)), 4) if m.per_layer_scale(l) is not None else None for l in range(m.cfg.L)]})')
    for l, s in sorted(stats.items()):
        ratio = (s['router'] / s['pair'].clamp_min(1e-30))
        say(f'   layer {l}: |router| / |pair term| median {ratio.median():.3f} (p90 {ratio.quantile(0.9):.3f}); '
            f'|router| / |V_theta force| median {(s["router"] / s["theta"].clamp_min(1e-30)).median():.4f}; '
            f'cos(router, pair) median {s["cos"].median():+.3f}')
    say(f'   removing the router term from the force at every layer: loss {l_full:.4f} -> {l_no:.4f} '
        f'(perplexity x{np.exp(l_no - l_full):.3f}); max |d logit| {d_lo:.2e}')
    m._layer_forces = lf0; m._sparse_topk_indices = topk0
(HERE / 'vphi_ste_term_output.txt').write_text('\n'.join(lines) + '\n')
