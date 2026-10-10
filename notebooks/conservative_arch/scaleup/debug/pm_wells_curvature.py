"""Can PM1's wells be integrated exactly? Curvature and quadratic fidelity (protocol SS5.15).

Written 2026-10-10, before design (b): fold the stiff part of the Poisson-mode
wells into the exactly integrated low-rank flow, as SR2 did for V_theta. The
design rests on two measurable conditions; this script measures them on PM1's
trained weights, evaluation only, CPU.

The wells at token t, layer l, with phi held fixed (the per-layer flow):
    U(h) = -sum_v c_v E_v(h),  c_v = phi_v(t) a_{l,v},  E_v = exp(-k_v^2 |h - mu_v|^2)
    force   F(h) = -grad U = -sum_v 2 k_v^2 c_v E_v r_v,          r_v = h - mu_v
    Hessian H(h) = sum_v 2 k_v^2 c_v E_v (I - 2 k_v^2 r_v r_v^T)  =  alpha I - W
so H is an isotropic part alpha = sum_v 2 k_v^2 c_v E_v plus a rank <= K part
W = sum_v 4 k_v^4 c_v E_v r_v r_v^T. Its eigenvalues are alpha on the
complement of span{r_v} and alpha - eig(W) on the span.

  1. STIFFNESS. omega dt = dt sqrt(lambda / m) for the wells' isotropic part
     and their stiffest positive direction; the rate of the most negative
     (unstable) direction, dt sqrt(|lambda| / m). An explicit kick
     mis-integrates curvature with omega dt above about 2 (V_theta's modes, which
     SR2 fixed, read 2.8 on PM1 in Cell 6b-13).
  2. QUADRATIC FIDELITY (and, added 2026-10-10 after the first run, the same for
     an isotropic-only variant that integrates alpha exactly and leaves W in the
     kick: |F(h_mid) - (F(h_in) - alpha (h_mid - h_in))| / |F(h_mid)|). At the point the kick is actually evaluated (h_mid,
     captured from the model's own force call), the force of the frozen
     quadratic F(h_in) - H(h_in)(h_mid - h_in) against the true F(h_mid), phi
     fixed. remainder = |F(h_mid) - F_quad| / |F(h_mid) - F(h_in)|: the share of
     the force change over the half-step that the quadratic misses. Design (b)
     leaves exactly that in the kick.

Descriptive, not pre-registered: it decides whether design (b) is worth
implementing. Rule of thumb written before the run: (b) is worthwhile if the
wells are stiff (median omega dt of the isotropic or stiffest part above 2 at
layer 1) and the quadratic captures most of the change (median remainder
below 0.5 at layer 1).

Usage: python3 pm_wells_curvature.py OUT_DIR FOLDER
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

if __name__ == '__main__':
    OUT = Path(sys.argv[1]); FOLDER = sys.argv[2]
    sys.argv = [sys.argv[0], str(OUT / 'harness')]
    sys.path.insert(0, str(Path(__file__).parent))
    import verify_pm_switch as P
    model, g, _ = P.build(FOLDER, P.CONFIGS['F3.1'][1] + P.PM_ON)
    model.to('cpu'); model.eval()
    assert getattr(model, 'pm_mu', None) is not None
    DT = model.cfg.dt

    rng = np.random.default_rng(20260920)                    # Cell 6b-7's seed: its first batch
    xb, _ = g['get_batch'](g['val_ids'], 4, 512, rng)
    x = torch.from_numpy(xb[:2])                             # 2 x 512 tokens

    # capture each layer's input (h_in) and the state the wells' force is evaluated at (h_mid)
    cap_in, cap_mid = {}, {}
    orig_step = model._fock_layer_step
    def step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
        cap_in[layer_idx] = (h.detach().clone(), m_b)
        return orig_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
    had_pmf = 'poisson_mode_force' in model.__dict__
    orig_pmf = model.poisson_mode_force
    cur = {'l': None}
    def pmf(h, layer_idx):
        cap_mid.setdefault(layer_idx, h.detach().clone())
        return orig_pmf(h, layer_idx)
    model._fock_layer_step = step
    model.poisson_mode_force = pmf
    try:
        with torch.enable_grad():
            model(x)
    finally:
        model._fock_layer_step = orig_step                   # restore by assignment (depth routing)
        if had_pmf:
            model.poisson_mode_force = orig_pmf
        else:
            model.__dict__.pop('poisson_mode_force', None)
    assert model._fock_layer_step is orig_step

    mu = model.pm_mu.detach().double()
    k2 = model.pm_log_kappa2.detach().double().exp()         # (K,)

    def wells(h, phi, a):
        """F, alpha, and the K x K pieces of W at states h (N, d), phi (N, K) fixed."""
        r = h[:, None, :] - mu[None]                          # (N, K, d)
        E = torch.exp(-k2 * (r * r).sum(-1))                  # (N, K)
        c = phi * a[None]                                     # (N, K)
        w = 2 * k2 * c * E                                    # (N, K)
        F = -(w[..., None] * r).sum(1)                        # (N, d)
        alpha = w.sum(-1)                                     # (N,)
        dW = 2 * k2 * w                                       # W = sum_v dW_v r_v r_v^T
        return F, alpha, r, dW

    res = {}
    lines = [f'PM1 wells: curvature and quadratic fidelity  ...{FOLDER[-60:]}',
             f'   2 x 512 tokens (Cell 6b-7\'s first batch), phi fixed at the layer input, dt = {DT:g}']
    for l in sorted(cap_in):
        h_in, m_b = cap_in[l]
        h_mid = cap_mid[l]
        with torch.no_grad():
            _, phi = model.poisson_mode_occupation(h_in)
            a = model.pm_effective_depth(l).double()
        B, T, d = h_in.shape
        hi = h_in.reshape(-1, d).double(); hm = h_mid.reshape(-1, d).double()
        ph = phi.reshape(-1, phi.shape[-1]).double()
        m = (m_b if torch.is_tensor(m_b) else torch.tensor(float(m_b))).double()
        m = m.expand(B, T, 1).reshape(-1) if m.dim() else m.expand(B * T)

        F_in, alpha, r, dW = wells(hi, ph, a)
        F_mid, _, _, _ = wells(hm, ph, a)
        # H (h_mid - h_in) = alpha dh - sum_v dW_v r_v (r_v . dh)
        dh = hm - hi
        Hdh = alpha[:, None] * dh - (dW[..., None] * r * (r * dh[:, None, :]).sum(-1, keepdim=True)).sum(1)
        F_quad = F_in - Hdh
        change = (F_mid - F_in).norm(dim=-1)
        remainder = (F_mid - F_quad).norm(dim=-1) / change.clamp_min(1e-12)
        rel_to_force = (F_mid - F_quad).norm(dim=-1) / F_mid.norm(dim=-1).clamp_min(1e-12)
        # top-M (added 2026-10-10 for PMX on the GPU, whose batched solvers stop at
        # 32 x 32): only each token's M wells with the largest force |w_v| |r_v| at
        # h_in go into the exact flow; the others stay whole in the kick. And the
        # hybrid: all wells' force F(h_in) and isotropic alpha (no eigensolve), plus
        # the top-M wells' rank part W
        r_in = hi[:, None, :] - mu[None]
        E_in = torch.exp(-k2 * (r_in * r_in).sum(-1)); w_in = 2 * k2 * ph * a[None] * E_in
        score = w_in.abs() * r_in.norm(dim=-1)
        topm = {}
        for M in (8, 16, 24):
            idx = score.topk(M, dim=-1).indices                              # (N, M)
            keep = torch.zeros_like(w_in).scatter_(1, idx, 1.0)
            wS = w_in * keep
            FS = -(wS[..., None] * r_in).sum(1)
            aS = wS.sum(-1)
            dWS = 2 * k2 * wS
            HdhS = aS[:, None] * dh - (dWS[..., None] * r_in * (r_in * dh[:, None, :]).sum(-1, keepdim=True)).sum(1)
            topm[M] = (F_mid - (FS - HdhS)).norm(dim=-1) / F_mid.norm(dim=-1).clamp_min(1e-12)
            # hybrid: every well's force F(h_in) and isotropic alpha go into the flow
            # (no eigensolve needed), only the top-M wells' rank part W_S does
            HdhH = alpha[:, None] * dh - (dWS[..., None] * r_in * (r_in * dh[:, None, :]).sum(-1, keepdim=True)).sum(1)
            topm[f'iso+W{M}'] = (F_mid - (F_in - HdhH)).norm(dim=-1) / F_mid.norm(dim=-1).clamp_min(1e-12)
            # the same hybrid with the top M chosen by the rank part's own weight |dW_v| |r_v|^2
            idx2 = (2 * k2 * w_in.abs() * (r_in * r_in).sum(-1)).topk(M, dim=-1).indices
            dW2 = 2 * k2 * w_in * torch.zeros_like(w_in).scatter_(1, idx2, 1.0)
            Hdh2 = alpha[:, None] * dh - (dW2[..., None] * r_in * (r_in * dh[:, None, :]).sum(-1, keepdim=True)).sum(1)
            topm[f'iso+W{M}byW'] = (F_mid - (F_in - Hdh2)).norm(dim=-1) / F_mid.norm(dim=-1).clamp_min(1e-12)
        # the isotropic-only variant: only alpha (h - h_in) goes into the exact flow
        F_iso = F_in - alpha[:, None] * dh
        rel_iso = (F_mid - F_iso).norm(dim=-1) / F_mid.norm(dim=-1).clamp_min(1e-12)

        # eigenvalues of W: nonzero eigenvalues of (R R^T) diag(dW), K x K, per token
        G = torch.einsum('nkd,njd->nkj', r, r)                # (N, K, K)
        ev = torch.linalg.eigvals(G * dW[:, None, :]).real    # (N, K)
        lam_pos = alpha - ev.min(-1).values                   # stiffest positive direction of H
        lam_neg = alpha - ev.max(-1).values                   # most negative direction of H
        q = lambda t: [float(t.quantile(p)) for p in (0.05, 0.5, 0.95)]
        odt = lambda lam: DT * torch.sqrt(lam.clamp_min(0) / m)
        rate = DT * torch.sqrt((-lam_neg).clamp_min(0) / m)
        R = dict(alpha_omega_dt=q(odt(alpha)), stiffest_omega_dt=q(odt(lam_pos)),
                 unstable_rate_dt=q(rate), share_unstable=float((lam_neg < 0).double().mean()),
                 remainder=q(remainder), remainder_rel_force=q(rel_to_force), remainder_iso_rel_force=q(rel_iso),
                 alpha_negative_share=float((alpha < 0).double().mean()),
                 remainder_topM_rel_force={M: q(v) for M, v in topm.items()},
                 step_over_state=float((dh.norm(dim=-1) / hi.norm(dim=-1)).median()))
        res[l] = R
        lines += [f'\nlayer {l}   (|h_mid - h_in| / |h_in| median {R["step_over_state"]:.2f})',
                  f'   STIFFNESS   omega dt, isotropic part      p05/p50/p95 ' + ' / '.join(f'{v:.2f}' for v in R['alpha_omega_dt']),
                  f'               omega dt, stiffest direction  p05/p50/p95 ' + ' / '.join(f'{v:.2f}' for v in R['stiffest_omega_dt']),
                  f'               unstable rate x dt            p05/p50/p95 ' + ' / '.join(f'{v:.2f}' for v in R['unstable_rate_dt'])
                  + f'   (tokens with a negative direction {100*R["share_unstable"]:.0f}%)',
                  f'   FIDELITY    remainder / force change      p05/p50/p95 ' + ' / '.join(f'{v:.3f}' for v in R['remainder']),
                  f'               remainder / force at h_mid    p05/p50/p95 ' + ' / '.join(f'{v:.3f}' for v in R['remainder_rel_force']),
                  f'               isotropic part only: remainder / force at h_mid  p05/p50/p95 ' + ' / '.join(f'{v:.3f}' for v in R['remainder_iso_rel_force'])
                  + f'   (tokens with alpha < 0: {100*R["alpha_negative_share"]:.0f}%)']
        lines += [(f'               top {M} wells only' if isinstance(M, int) else
                   f'               all wells\' alpha and force + W of the top {M[5:].replace("byW", "")}' + (' (chosen by W)' if M.endswith('byW') else ' (chosen by force)'))
                  + ': remainder / force at h_mid  p05/p50/p95 ' + ' / '.join(f'{v:.3f}' for v in vals)
                  for M, vals in R['remainder_topM_rel_force'].items()]
    L1 = res[max(res)]
    stiff = max(L1['alpha_omega_dt'][1], L1['stiffest_omega_dt'][1]) > 2
    faithful = L1['remainder'][1] < 0.5
    lines += ['', f'RULE OF THUMB (written before the run), layer {max(res)}: stiff {stiff}, quadratic faithful {faithful}'
              f'  -> design (b) {"WORTHWHILE" if stiff and faithful else "NOT SUPPORTED as written"}']
    txt = '\n'.join(lines)
    print(txt)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pm_wells_curvature_output.txt').write_text(txt + '\n')
    (OUT / 'pm_wells_curvature.json').write_text(json.dumps(res, indent=1))
