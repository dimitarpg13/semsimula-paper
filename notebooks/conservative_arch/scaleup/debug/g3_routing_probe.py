"""G3 exchange-field routing probe: did the routing sharpen? (protocol SS5.10, G3/G3').

6b-6 part B (routing entropy) skips XiRoutedConservativeAttention, and its force
share reads 0 because relax_share is computed only for the non-conservative
modes. This measures, on the step-500 and best (step 31,000) checkpoints of G3,
over the same validation tokens, in eval mode, for each layer:

  - routing entropy H(alpha_t) / log(t) per head, averaged over t >= 16
    (1 = uniform over the prefix, 0 = one-hot), and the mean max weight
  - routing logits q.k/sqrt(d_k): std and max |score| per head
  - weight norms: spectral and Frobenius norms of W_q, W_k (per head) and of
    W_uq, W_v; sigma(W_q) sigma(W_k) / sqrt(d_k) bounds the logit scale per
    unit input
  - the field's force share: ||lambda f_field|| / ||f_total|| per token, from
    force_live and _layer_forces

Usage: python3 g3_routing_probe.py OUT_DIR
"""
import math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G
from verify_vphi_xi_grad_path import build

FOLDER = ('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_rglive_vplive_'
          'xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot')
CK = Path.home() / 'Downloads' / FOLDER / 'checkpoints'
PREFIX = FOLDER[len('semsimula_'):].replace('fock_cfc_baoab_owt', 'fock_cfc_owt')


def weight_norms(rf):
    H, dk, dv = rf.H, rf.d_k, rf.d_v
    out = {}
    for name, W, dh in (('W_q', rf.W_q.weight, dk), ('W_k', rf.W_k.weight, dk),
                        ('W_uq', rf.W_uq.weight, dv), ('W_v', rf.W_v.weight, dv)):
        blocks = W.detach().view(H, dh, -1)
        out[name] = ([float(torch.linalg.matrix_norm(b, ord=2)) for b in blocks],
                     [float(b.norm()) for b in blocks])
    out['logit_bound'] = [out['W_q'][0][h] * out['W_k'][0][h] / math.sqrt(dk) for h in range(H)]
    return out


def probe(model, x):
    rf = model.relax_field
    rec = []
    orig_fl, orig_lf = rf.force_live, model._layer_forces
    cur = {}

    def fl(h_in, xi_route, causal):
        with torch.no_grad():
            B, T, d = xi_route.shape
            q = rf.W_q(xi_route).view(B, T, rf.H, rf.d_k).transpose(1, 2)
            k = rf.W_k(xi_route).view(B, T, rf.H, rf.d_k).transpose(1, 2)
            sc = torch.matmul(q, k.transpose(-1, -2)) * (rf.d_k ** -0.5)
            mask = causal.view(1, 1, T, T)
            a = rf._routing(xi_route, causal)                       # (B, H, T, T)
            t_idx = torch.arange(T)
            n_src = mask.sum(-1).clamp(min=1).float()               # sources per row
            ent = -(a * torch.log(a.clamp(min=1e-30))).sum(-1)      # (B, H, T)
            rows = (n_src[0, 0] >= 16)
            cur['ent'] = (ent[..., rows] / torch.log(n_src[..., rows])).mean(dim=(0, 2)).tolist()
            cur['amax'] = a.max(-1).values[..., rows].mean(dim=(0, 2)).tolist()
            scm = sc.masked_select(mask.expand_as(sc)).view(-1)
            cur['score_std'] = [float(sc[:, h][mask[0, 0].expand(B, T, T)].std()) for h in range(rf.H)]
            cur['score_max'] = [float(sc[:, h][mask[0, 0].expand(B, T, T)].abs().max()) for h in range(rf.H)]
        f = orig_fl(h_in, xi_route, causal)
        cur['f_field'] = f.detach()
        return f

    def lf(*a, **k):
        out = orig_lf(*a, **k)
        if 'f_field' in cur:
            layer = len(rec)                      # call order: one call per layer step
            tot = sum(t for t in (out if isinstance(out, tuple) else (out,)) if t is not None)
            lam = 1.0                             # the gate is pinned at 1 (6b-6: live lambda [1.0, 1.0])
            share = ((lam * cur['f_field']).norm(dim=-1) / tot.detach().norm(dim=-1).clamp(min=1e-12))
            rec.append({**{k2: v for k2, v in cur.items() if k2 != 'f_field'},
                        'layer': layer, 'share_median': float(share.median()),
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


if __name__ == '__main__':
    model, tag, _ = build(True, FOLDER, 'live', 'live', 'attention_potential', 'live')
    val = np.load(G.VAL)
    rng = np.random.default_rng(20261005)
    st = rng.integers(0, len(val) - 513, size=2)
    x = torch.from_numpy(np.stack([val[s:s + 512] for s in st]).astype(np.int64))
    print(f"G3 exchange-field routing probe  ...{tag[tag.find('cgqk'):]}")
    print(f"   relax_field: H={model.relax_field.H} d_k={model.relax_field.d_k} route_from={model.relax_field.route_from}\n")
    res = {}
    for name in ('step500_best', 'best'):
        ck = torch.load(CK / f'{PREFIX}_{name}.pt', map_location='cpu', weights_only=False)
        r = model.load_state_dict(ck['model_state_dict'], strict=False)
        assert not r.missing_keys and not r.unexpected_keys, r
        res[name] = (ck.get('step'), weight_norms(model.relax_field), probe(model, x))
    for name, (step, wn, rec) in res.items():
        print(f"== {name} (step {step})")
        f = lambda v: '[' + ', '.join(f'{u:.3f}' for u in v) + ']'
        print(f"   spectral norm per head  W_q {f(wn['W_q'][0])}   W_k {f(wn['W_k'][0])}")
        print(f"                           W_uq {f(wn['W_uq'][0])}   W_v {f(wn['W_v'][0])}")
        print(f"   logit bound sigma(W_q) sigma(W_k)/sqrt(d_k) per head {f(wn['logit_bound'])}"
              f"   (inputs have |h| ~ sqrt(d) = {math.sqrt(model.cfg.d):.1f})")
        for rr in rec:
            print(f"   layer {rr['layer']}: entropy/log(t) {f(rr['ent'])}   max weight {f(rr['amax'])}")
            print(f"            score std {f(rr['score_std'])}   max|score| {f(rr['score_max'])}")
            print(f"            field force share of total: median {rr['share_median']:.3f}  p90 {rr['share_p90']:.3f}")
        print()
