"""Exchange-field probe for G3 and G3' (protocol SS5.10): routing, scale, force share, gradient.

Supersedes g3_routing_probe.py, which recomputes scores as q.k / sqrt(d_k) and
so misreads the QK-normalised routing of G3'. For every checkpoint found in the
arm's folder (step500_best, best, and any manual/periodic ones), in eval mode,
on the same validation tokens:

  routing   per layer and head: entropy / log(t) over rows with t >= 16, mean
            max weight, score std and max |score|, computed exactly as
            XiRoutedConservativeAttention._routing does (cosine x clamped
            per-head scale under qk_norm, q.k / sqrt(d_k) otherwise)
  scale     G3': the per-head logit scale sigma_h = min(exp(lambda_h), max),
            and whether any head sits at the ceiling. G3: the logit bound
            sigma(W_q) sigma(W_k) / sqrt(d_k) per unit input
  weights   spectral norms of W_q, W_k per head
  share     the field's share of the total conservative force, per layer
  gradient  OFFLINE: the pre-clip gradient norm of the relax_field group (and
            of each of its parameters) under the LM loss, train mode, on fixed
            tokens. The training log records no per-group norms, so this is the
            only like-for-like reading of G3' pre-registered prediction that the
            field's gradient stays flat; compare G3 and G3' checkpoint by
            checkpoint.

Usage: python3 exchange_field_probe.py OUT_DIR [G3|G3p|both] [--smoke]
  --smoke runs G3''s code path on G3's weights (logit scale at its init): a test
          of the probe, not a measurement.
"""
import json, math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G
import verify_pm_switch as VP

_P, _S = VP._P, VP._S
ARMS = {
    'G3': (VP.G3_F, VP.LIVE + (("LADDER_MECHANISM = 'none'", "LADDER_MECHANISM = 'attention_potential'"),
                               ("RELAX_GRAD_PATH        = 'default'", "RELAX_GRAD_PATH        = 'live'"))),
    'G3p': (_P + 'rglive_vplive_xilive_rfqk_rfclip0p3_L2probe' + _S + 'attnpot', VP.CONFIGS["G3'"][1]),
}


def scores(rf, x_route):
    B, T, _ = x_route.shape
    q = rf.W_q(x_route).view(B, T, rf.H, rf.d_k).transpose(1, 2)
    k = rf.W_k(x_route).view(B, T, rf.H, rf.d_k).transpose(1, 2)
    if getattr(rf, 'qk_norm', False):
        q, k = torch.nn.functional.normalize(q, dim=-1), torch.nn.functional.normalize(k, dim=-1)
        sig = rf.logit_scale.exp().clamp(max=rf.logit_scale_max)
        return (q @ k.transpose(-1, -2)) * sig.view(1, -1, 1, 1)
    return (q @ k.transpose(-1, -2)) * rf.d_k ** -0.5


def probe_forward(model, x):
    rf = model.relax_field
    rec, cur = [], {}
    orig_fl, orig_lf = rf.force_live, model._layer_forces

    def fl(h_in, x_route, causal):
        with torch.no_grad():
            T = x_route.shape[1]
            sc = scores(rf, x_route)
            a = rf._routing(x_route, causal)
            n_src = causal.sum(-1).clamp(min=1).float()                     # (T,)
            rows = n_src >= 16
            ent = -(a * torch.log(a.clamp(min=1e-30))).sum(-1)              # (B, H, T)
            cur['ent'] = (ent[..., rows] / torch.log(n_src[rows])).mean(dim=(0, 2)).tolist()
            cur['amax'] = a.max(-1).values[..., rows].mean(dim=(0, 2)).tolist()
            m = causal.view(1, 1, T, T).expand_as(sc)
            cur['score_std'] = [float(sc[:, h][m[:, h]].std()) for h in range(rf.H)]
            cur['score_max'] = [float(sc[:, h][m[:, h]].abs().max()) for h in range(rf.H)]
        f = orig_fl(h_in, x_route, causal)
        cur['f_field'] = f.detach()
        return f

    def lf(*a, **k):
        out = orig_lf(*a, **k)
        if 'f_field' in cur:
            tot = sum(t for t in (out if isinstance(out, tuple) else (out,)) if t is not None)
            share = cur['f_field'].norm(dim=-1) / tot.detach().norm(dim=-1).clamp(min=1e-12)
            rec.append({**{k2: v for k2, v in cur.items() if k2 != 'f_field'}, 'call': len(rec),
                        'share_median': float(share.median()),
                        'share_p90': float(share.flatten().quantile(0.9))})
            cur.clear()
        return out

    rf.force_live = fl
    model._layer_forces = lf
    try:
        model.eval()
        with torch.enable_grad():
            model(x)
    finally:
        rf.force_live = orig_fl
        del model._layer_forces
    return rec


def grad_norms(model, batches):
    out = []
    for x, y in batches:
        model.train(); torch.manual_seed(7)
        _, loss = model(x, y)
        loss.backward()
        per = {n: float(p.grad.norm()) for n, p in model.named_parameters()
               if n.startswith('relax_field.') and p.grad is not None}
        group = math.sqrt(sum(v * v for v in per.values()))
        total = math.sqrt(sum(float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None))
        out.append({'group': group, 'total': total, 'per': per})
        model.zero_grad(set_to_none=True)
    keys = out[0]['per']
    return {'group': float(np.mean([o['group'] for o in out])), 'group_sd': float(np.std([o['group'] for o in out])),
            'total': float(np.mean([o['total'] for o in out])),
            'per': {k: float(np.mean([o['per'][k] for o in out])) for k in keys}}


def checkpoints(folder):
    ck = Path.home() / 'Downloads' / folder / 'checkpoints'
    pfx = folder[len('semsimula_'):].replace('fock_cfc_baoab_owt', 'fock_cfc_owt')
    found = sorted(p for p in ck.glob(f'{pfx}_*.pt') if not p.name.endswith('.published.pt'))
    return found, pfx


def run_arm(name, smoke=False):
    folder, repl = ARMS['G3p' if smoke else name]
    wfolder = ARMS['G3'][0] if smoke else folder
    if not (Path.home() / 'Downloads' / wfolder).exists():
        print(f"\n== {name}: folder not downloaded yet ({wfolder[:60]}...)"); return None
    model, g, _ = VP.build(wfolder, repl)
    model.cfg.use_layer_checkpoint = False
    rf = model.relax_field
    val = np.load(G.VAL)
    rng = np.random.default_rng(20261005)
    st = rng.integers(0, len(val) - 513, size=6)
    seqs = [val[s:s + 513].astype(np.int64) for s in st]
    x_probe = torch.from_numpy(np.stack([q[:512] for q in seqs[:2]]))
    batches = [(torch.from_numpy(np.stack([q[:512] for q in seqs[i:i + 2]])),
                torch.from_numpy(np.stack([q[1:] for q in seqs[i:i + 2]]))) for i in (2, 4)]
    found, pfx = checkpoints(wfolder)
    res = {}
    label = f"{name}{' (SMOKE: G3 code path test on G3 weights, not a measurement)' if smoke else ''}"
    print(f"\n== {label}: tag ...{g['_variant_tag'][g['_variant_tag'].find('rglive'):]}")
    print(f"   relax_field H={rf.H} d_k={rf.d_k} qk_norm={getattr(rf, 'qk_norm', False)}; checkpoints: "
          + ', '.join(p.name[len(pfx) + 1:-3] for p in found))
    for p in found:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        r = model.load_state_dict(ck['model_state_dict'], strict=False)
        assert not r.unexpected_keys and set(r.missing_keys) <= {'relax_field.logit_scale'}, r
        if r.missing_keys:
            with torch.no_grad():
                rf.logit_scale.fill_(math.log(1 / 0.07))
        tag = p.name[len(pfx) + 1:-3]
        w = {}
        for nm in ('W_q', 'W_k'):
            blocks = getattr(rf, nm).weight.detach().view(rf.H, rf.d_k, -1)
            w[nm] = [float(torch.linalg.matrix_norm(b, ord=2)) for b in blocks]
        bound = [w['W_q'][h] * w['W_k'][h] / math.sqrt(rf.d_k) for h in range(rf.H)]
        sig = (rf.logit_scale.detach().exp().clamp(max=rf.logit_scale_max).tolist()
               if getattr(rf, 'qk_norm', False) else None)
        rec = probe_forward(model, x_probe)
        gn = grad_norms(model, batches)
        res[tag] = dict(step=ck.get('step'), val_ppl=ck.get('val_ppl'), weights=w, logit_bound=bound,
                        sigma=sig, layers=rec, grad=gn)
        f = lambda v: '[' + ', '.join(f'{u:.2f}' for u in v) + ']'
        print(f"-- {tag} (step {ck.get('step')}, val_ppl {ck.get('val_ppl')})")
        print(f"   spectral norm W_q {f(w['W_q'])}  W_k {f(w['W_k'])}")
        if sig is not None:
            print(f"   logit scale sigma_h {f(sig)}  (ceiling {rf.logit_scale_max:g}; at ceiling: "
                  f"{sum(s >= rf.logit_scale_max - 1e-6 for s in sig)} of {rf.H})")
        else:
            print(f"   logit bound sigma(W_q)sigma(W_k)/sqrt(d_k) per unit input {f(bound)}")
        for rr in rec:
            print(f"   call {rr['call']}: entropy/log(t) {f(rr['ent'])}  max weight {f(rr['amax'])}")
            print(f"           score std {f(rr['score_std'])}  max|score| {f(rr['score_max'])}  "
                  f"field share median {rr['share_median']:.3f} p90 {rr['share_p90']:.3f}")
        print(f"   OFFLINE grad (LM loss, 2 batches x 2 x 512): relax_field group {gn['group']:.4f} "
              f"(sd {gn['group_sd']:.4f}); total {gn['total']:.4f}; "
              + ', '.join(f"{k.split('.', 1)[1].replace('.weight', '')} {v:.4f}" for k, v in gn['per'].items()))
    return res


if __name__ == '__main__':
    which = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'both'
    smoke = '--smoke' in sys.argv
    out = {}
    if smoke:
        out['smoke'] = run_arm("G3'", smoke=True)
    else:
        for name in (('G3', 'G3p') if which == 'both' else (which,)):
            out[name] = run_arm(name)
    (G.OUT / f"exchange_field_probe{'_smoke' if smoke else ''}.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {G.OUT / ('exchange_field_probe' + ('_smoke' if smoke else '') + '.json')}")
