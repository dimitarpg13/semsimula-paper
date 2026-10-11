"""D-sync: two clustering mechanisms on the trained conservative arms (descriptive, 2026-10-11).

Companion note: Two_Clustering_Mechanisms_Saturation_and_Synchronization.md.

The 2026-09-17 eigensolve failures in the low-rank integrator ("ill-conditioned
or too many repeated eigenvalues") came from a rank-deficient low-rank factor
G = [sqrt(g_1) B_1, ..., sqrt(g_K) B_K]. Each column block's weight is the
well responsibility g_k = w_k exp(-e_k / 2), with two factors that correspond
to the two clustering mechanisms of the note:
  * w_k, the context gating: a softmax over the K wells (SATURATION, the
    hard-assignment mechanism of Geshkovski et al. 2023a, Theorem 2.1, and of
    the leaders);
  * exp(-e_k / 2), the token's Gaussian proximity to well k (CLUSTERING onto
    attractors, the Kuramoto-type mechanism of Geshkovski et al. 2023b and
    Karagodin et al. 2024).
This script measures both factors per token and layer, the effective rank of
G, and the inter-token geometry, on the layer inputs the model actually used.

Per layer:
  GATING        entropy (in wells, exp H) of w, of the proximity p ~ exp(-e/2),
                and of the responsibility g; which factor carries the
                concentration (H(g) against H(w) and H(p))
  RANK          participation ratio of the eigenvalues of G^T G and the count
                above the integrator's floor, against the nominal K * rank;
                Spearman of the rank against H(w), H(p), and the token's
                nearest-neighbour distance
  TOKENS        synchronization of the token states: pairwise cosine (mean,
                histogram, share above 0.9), nearest-neighbour cosine,
                geodesic distance to the FIRST token by position band
                (Karagodin et al. 2024, Theorem 4.1), and the same for the
                layer outputs, so coherence can be followed with depth
  MODES (PM1)   how many of the 64 Poisson modes the tokens are nearest to,
                and the share on the most popular one

Usage: python3 dsync_clustering_diagnostics.py OUT_DIR FOLDER {f31|sr2|pm1|pmx}
"""
import json, math, sys
from pathlib import Path

import numpy as np
import torch

N_SEQ, T = 4, 512
POS_BANDS = ((1, 8), (8, 32), (32, 128), (128, 512))


def entropy_wells(p):
    """exp of the Shannon entropy along the last axis: effective number of wells."""
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-30)
    return torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum(-1))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def q(t, ps=(0.05, 0.5, 0.95)):
    t = torch.as_tensor(t, dtype=torch.float64).flatten()
    return [float(t.quantile(p)) for p in ps]


def token_geometry(h):
    """h (B, T, d): synchronization statistics of the token directions per sequence."""
    u = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    C = torch.einsum('btd,bsd->bts', u, u)                       # cosines
    B, T_ = C.shape[:2]
    off = ~torch.eye(T_, dtype=torch.bool)[None].expand(B, T_, T_)
    cos = C[off]
    nn_cos = C.masked_fill(~off, -2.0).max(-1).values            # nearest neighbour
    hist, edges = np.histogram(cos.numpy(), bins=40, range=(-1.0, 1.0))
    # geodesic distance to the first token, by position band
    d_first = torch.acos(C[:, :, 0].clamp(-1, 1))                # (B, T)
    bands = {f'{a}-{b}': float(d_first[:, a:b].mean()) for a, b in POS_BANDS}
    return dict(cos_mean=float(cos.mean()), cos_p05_p50_p95=q(cos), share_cos_gt_0p9=float((cos > 0.9).double().mean()),
                nn_cos_p05_p50_p95=q(nn_cos), cos_hist=hist.tolist(), cos_edges=edges.tolist(),
                dist_to_first_by_band=bands, radius_p50=float(h.norm(dim=-1).median()))


if __name__ == '__main__':
    OUT = Path(sys.argv[1]); FOLDER = sys.argv[2]; KIND = sys.argv[3]
    assert KIND in ('f31', 'sr2', 'pm1', 'pmx')
    sys.argv = [sys.argv[0], str(OUT / 'harness')]
    sys.path.insert(0, str(Path(__file__).parent))
    import verify_pm_switch as P
    repl = P.CONFIGS['F3.1'][1]
    if KIND in ('sr2', 'pmx'):
        repl = repl + (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
    if KIND in ('pm1', 'pmx'):
        repl = repl + P.PM_ON
    if KIND == 'pmx':
        repl = repl + (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),)
    model, g, _ = P.build(FOLDER, repl)
    model.to('cpu'); model.eval()
    vt = model.V_theta

    # ---- capture: layer inputs/outputs, and (xis, h, comps) at the linearisation ----
    rng = np.random.default_rng(20260920)                        # Cell 6b-7's batches
    xb, _ = g['get_batch'](g['val_ids'], N_SEQ, T, rng)
    x = torch.from_numpy(xb)
    lin, steps = {}, {}
    orig_htl = vt.harmonic_terms_lowrank
    cur = {'l': None}
    def htl(xis, h, *, comps=None):
        out = orig_htl(xis, h, comps=comps)
        if cur['l'] is not None and cur['l'] not in lin:         # the first call per layer: at h_in
            lin[cur['l']] = (comps, h.detach().clone(), out[2].detach().clone())
        return out
    orig_step = model._fock_layer_step
    def step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
        cur['l'] = layer_idx
        out = orig_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
        steps[layer_idx] = (h.detach().clone(), out[0].detach().clone())
        cur['l'] = None
        return out
    vt.harmonic_terms_lowrank = htl
    model._fock_layer_step = step
    try:
        with torch.enable_grad():
            model(x)
    finally:
        vt.__dict__.pop('harmonic_terms_lowrank', None)
        model._fock_layer_step = orig_step
    assert model._fock_layer_step is orig_step and vt.harmonic_terms_lowrank == orig_htl

    res = {'kind': KIND, 'tokens': N_SEQ * T, 'layers': {}}
    lines = [f'D-sync clustering diagnostics: {KIND}  ...{FOLDER[-60:]}',
             f'   {N_SEQ} x {T} tokens (Cell 6b-7\'s first batch), layer inputs as the model used them']
    say = lambda s='': (print(s, flush=True), lines.append(s))
    for l in sorted(lin):
        comps, h_in, G = lin[l]
        mu, a, w, B = (c if isinstance(c, torch.Tensor) else c for c in comps)  # joint bank: one tuple
        # the bank's own responsibility, recomputed from its components
        with torch.no_grad():
            diff = h_in.unsqueeze(-2) - mu                                 # (B, T, K, d)
            e = (a * diff * diff).sum(-1) + torch.einsum('...kd,...kdr->...kr', diff, B).pow(2).sum(-1)
            p_prox = torch.exp(-0.5 * (e - e.min(-1, keepdim=True).values))
            gk = w * torch.exp(-0.5 * e)
            Hw, Hp, Hg = entropy_wells(w), entropy_wells(p_prox), entropy_wells(gk)
            lam = torch.linalg.eigvalsh(G.transpose(-1, -2).double() @ G.double()).clamp_min(0)  # (B, T, KR)
            pr = lam.sum(-1).pow(2) / lam.pow(2).sum(-1).clamp_min(1e-300)
            n_above = (lam > 1e-10).sum(-1).double()                       # lowrank_modes' floor
            geo_in = token_geometry(h_in)
            u = h_in / h_in.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            C = torch.einsum('btd,bsd->bts', u, u).masked_fill(torch.eye(T, dtype=torch.bool)[None], -2.0)
            nn_dist = torch.acos(C.max(-1).values.clamp(-1, 1))            # geodesic to nearest token
        K, KR = w.shape[-1], G.shape[-1]
        R = dict(K=K, nominal_rank=KR,
                 H_w=q(Hw), H_prox=q(Hp), H_g=q(Hg),
                 share_g_one_well=float((Hg < 1.5).double().mean()),
                 pr_rank=q(pr), n_above_floor=q(n_above),
                 spearman_rank_vs=dict(H_w=spearman(pr.flatten(), Hw.flatten()), H_prox=spearman(pr.flatten(), Hp.flatten()),
                                       H_g=spearman(pr.flatten(), Hg.flatten()), nn_dist=spearman(pr.flatten(), nn_dist.flatten())),
                 tokens_in=geo_in, tokens_out=token_geometry(steps[l][1]))
        if getattr(model, 'pm_mu', None) is not None:
            with torch.no_grad():
                E, phi = model.poisson_mode_occupation(h_in)
                near = E.argmax(-1)                                        # (B, T)
                counts = torch.bincount(near.flatten(), minlength=E.shape[-1]).double()
                pm_H = float(torch.exp(-((counts / counts.sum()).clamp_min(1e-30) * torch.log((counts / counts.sum()).clamp_min(1e-30))).sum()))
                R['poisson_modes'] = dict(modes_used=int((counts > 0).sum()), effective_modes=pm_H,
                                          share_top_mode=float(counts.max() / counts.sum()),
                                          E_top_p50=float(E.max(-1).values.median()))
        res['layers'][l] = R
        say(f'\nlayer {l}   V_theta: K = {K} wells, G nominal rank {KR}')
        say(f'   GATING  effective wells p05/p50/p95:  context w {R["H_w"][0]:.2f} / {R["H_w"][1]:.2f} / {R["H_w"][2]:.2f}'
            f'   proximity {R["H_prox"][0]:.2f} / {R["H_prox"][1]:.2f} / {R["H_prox"][2]:.2f}'
            f'   responsibility g {R["H_g"][0]:.2f} / {R["H_g"][1]:.2f} / {R["H_g"][2]:.2f}   (tokens on one well: {100*R["share_g_one_well"]:.0f}%)')
        say(f'   RANK    participation ratio p05/p50/p95 {R["pr_rank"][0]:.1f} / {R["pr_rank"][1]:.1f} / {R["pr_rank"][2]:.1f};  above floor {R["n_above_floor"][1]:.0f} of {KR}')
        s = R['spearman_rank_vs']
        say(f'           Spearman(rank, .): context gating {s["H_w"]:+.2f}   proximity {s["H_prox"]:+.2f}   responsibility {s["H_g"]:+.2f}   nearest-token distance {s["nn_dist"]:+.2f}')
        gi, go = geo_in, R['tokens_out']
        say(f'   TOKENS  in : pairwise cos mean {gi["cos_mean"]:+.3f} (p95 {gi["cos_p05_p50_p95"][2]:+.3f}; >0.9: {100*gi["share_cos_gt_0p9"]:.1f}%)  nn cos p50 {gi["nn_cos_p05_p50_p95"][1]:.3f}  '
            f'|h| p50 {gi["radius_p50"]:.1f};  geodesic to token 1 by position ' + ', '.join(f'{k}: {v:.2f}' for k, v in gi['dist_to_first_by_band'].items()))
        say(f'           out: pairwise cos mean {go["cos_mean"]:+.3f} (p95 {go["cos_p05_p50_p95"][2]:+.3f}; >0.9: {100*go["share_cos_gt_0p9"]:.1f}%)  nn cos p50 {go["nn_cos_p05_p50_p95"][1]:.3f}  '
            f'|h| p50 {go["radius_p50"]:.1f};  geodesic to token 1 by position ' + ', '.join(f'{k}: {v:.2f}' for k, v in go['dist_to_first_by_band'].items()))
        if 'poisson_modes' in R:
            m = R['poisson_modes']
            say(f'   MODES   nearest Poisson mode: {m["modes_used"]} of 64 used, effective {m["effective_modes"]:.1f}, top mode holds {100*m["share_top_mode"]:.0f}% of tokens; top overlap E p50 {m["E_top_p50"]:.3f}')
    txt = '\n'.join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'dsync_clustering_{KIND}_output.txt').write_text(txt + '\n')
    (OUT / f'dsync_clustering_{KIND}.json').write_text(json.dumps(res, indent=1))
    print(f'\nwrote {OUT}/dsync_clustering_{KIND}_output.txt')
