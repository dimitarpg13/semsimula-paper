"""PM1 full run: the post-run measurements pre-registered in protocol SS5.15.

Evaluation only, CPU. The model is built through the ladder notebook's own
Cells 0-5b (F3.1's Cell 0 plus POISSON_MODES) on the run's best checkpoint,
over 8 x 512 validation tokens (seed 20261007). For every layer the PM1 force
is evaluated at, the layer's input states are captured from inside the
model's own force call (the first call per layer and batch is the layer's
input state), and the measurements below are made on them.

  DEPTH SIGNS   "most of the trained well depths positive (attractive)",
                called 60%. Scored on the letter: the share of all L x K
                depths that are positive. Also reported, descriptive: the
                share per layer, and the share of the PM1 force carried by
                modes with positive depth.
  REPETITION    "Spearman of phi for the best-matching mode against the
                decay-weighted count of earlier occurrences of the same token
                above 0.5", called 55%. Operationalised here, before the full
                run is scored: for every position t, v* = argmax_v E_v(t) (the
                mode the token's own state overlaps most), phi* = phi_v*(t),
                and C(t) = sum over s < t with x_s = x_t of lambda_v*^(t-1-s).
                Spearman(phi*, C) over all positions t >= 1. Scored at the
                layer that carries the larger PM1 force share (layer 1 in both
                probes); the other layer and the repeated positions alone
                (C > 0) are reported as descriptive.
  DP3 ON MODES  "Spearman of phi times depth against each mode's leave-one-
                out force contribution above 0.5 (by construction, a sanity
                check)", called 80%. The force is linear in the modes, so a
                mode's leave-one-out contribution is exactly the norm of its
                own term, |w_v| |h - mu_v|. Per token, Spearman across the K
                modes of phi_v * a_v (signed, as written) against that norm,
                averaged over tokens; per layer. |phi_v * a_v| reported too:
                with negative depths the signed form can read low while the
                magnitudes agree.
  DESCRIPTIVE   trained half-lives, kappa^2 x d, PM1 force / conservative
                force per layer.

Usage: python3 pm1_post_run_measurements.py OUT_DIR FOLDER [pmclip<thr>]
  FOLDER is the run's Drive folder mirrored under ~/Downloads, WITH its
  checkpoints/ (the best checkpoint is loaded, all keys must match).
Writes OUT_DIR/pm1_post_run_measurements.json and prints the report.
"""
import json, math, re, sys
from pathlib import Path

import numpy as np
import torch

PM_K = 64


def _ranks(t, dim=-1):
    return t.argsort(dim=dim).argsort(dim=dim).float()


def spearman_flat(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_rows(a, b):
    """Spearman per row (last dim), returns a 1-D tensor."""
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(-1, keepdim=True), rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1) + 1e-12)


@torch.no_grad()
def measure(model, val, n_batches=8, T=512, seed=20261007):
    model.eval()
    L, K = model.cfg.L, model.pm_mu.shape[0]
    lam = torch.sigmoid(model.pm_logit_lambda.float())
    loglam = torch.log(lam)
    k2 = model.pm_log_kappa2.float().exp()
    mu = model.pm_mu.float()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(val) - T - 1, size=n_batches)

    cap = {}
    def _wrap(h, layer_idx):
        if (cur[0], layer_idx) not in cap:
            cap[(cur[0], layer_idx)] = h.detach().float().clone()
        return type(model).poisson_mode_force(model, h, layer_idx)
    cur = [0]
    xs, shares = [], {l: [] for l in range(L)}
    model.poisson_mode_force = _wrap
    try:
        for b, s0 in enumerate(starts):
            cur[0] = b
            x = torch.from_numpy(val[s0:s0 + T].astype(np.int64))[None]
            xs.append(x)
            with torch.enable_grad():           # the forces need autograd even in eval
                model(x)
            for l in range(L):
                shares[l].append(float(model.pm_share[l]))
    finally:
        model.__dict__.pop('poisson_mode_force', None)

    res = {'layers': {}}
    for l in range(L):
        a = model.pm_depth[l].float()
        rep_phi, rep_C, dp3_signed, dp3_abs, f_pos, f_all = [], [], [], [], 0.0, 0.0
        for b in range(n_batches):
            h, x = cap[(b, l)], xs[b]
            E, phi = model.poisson_mode_occupation(h)
            E, phi = E.float()[0], phi.float()[0]                         # (T, K)
            d = ((h[0, :, None, :] - mu[None]) ** 2).sum(-1).sqrt()       # (T, K)
            fnorm = (2 * k2 * phi * a * E).abs() * d                       # |own term| = leave-one-out
            dp3_signed.append(spearman_rows(phi * a, fnorm))
            dp3_abs.append(spearman_rows((phi * a).abs(), fnorm))
            f_pos += float(fnorm[:, a > 0].sum()); f_all += float(fnorm.sum())
            # repetition
            vstar = E.argmax(-1)                                            # (T,)
            tt = torch.arange(T)
            lag = (tt[:, None] - tt[None, :] - 1).float()                   # t - 1 - s
            same = (x[0][:, None] == x[0][None, :]) & (lag >= 0)
            w = torch.exp(lag.clamp_min(0) * loglam[vstar][:, None])
            C = (same.float() * w).sum(-1)
            rep_phi.append(phi[tt, vstar][1:]); rep_C.append(C[1:])
        rp, rc = torch.cat(rep_phi).numpy(), torch.cat(rep_C).numpy()
        pos = rc > 0
        res['layers'][l] = dict(
            depth_positive_share=float((a > 0).float().mean()),
            depth_p05_p50_p95=[float(a.quantile(q)) for q in (0.05, 0.5, 0.95)],
            force_share_from_positive_depths=f_pos / (f_all + 1e-30),
            pm_over_conservative_force=float(np.mean(shares[l])),
            repetition_spearman=spearman_flat(rp, rc),
            repetition_spearman_repeats_only=spearman_flat(rp[pos], rc[pos]) if pos.sum() > 10 else float('nan'),
            repeated_positions=int(pos.sum()), positions=int(len(rc)),
            dp3_signed=float(torch.cat(dp3_signed).nanmean()),
            dp3_abs=float(torch.cat(dp3_abs).nanmean()))
    allw = model.pm_depth.float()
    hl = (math.log(2) / -loglam).numpy()
    res['depth_positive_share_all'] = float((allw > 0).float().mean())
    res['halflife_p05_p50_p95'] = [float(np.quantile(hl, q)) for q in (0.05, 0.5, 0.95)]
    res['kappa2_x_d_p50'] = float(np.median(k2.numpy() * model.cfg.d))
    res['score_layer'] = int(max(range(L), key=lambda l: res['layers'][l]['pm_over_conservative_force']))
    return res


def report(res):
    out = []
    p = out.append
    sl = res['score_layer']; S = res['layers'][sl]
    p(f'PM1 post-run measurements (protocol SS5.15); scored at layer {sl}, the larger PM1 force share')
    p(f'   half-life p05/p50/p95 {res["halflife_p05_p50_p95"][0]:.1f} / {res["halflife_p05_p50_p95"][1]:.1f} / '
      f'{res["halflife_p05_p50_p95"][2]:.1f} tokens;  kappa^2 x d p50 {res["kappa2_x_d_p50"]:.2f}')
    for l, r in sorted(res['layers'].items()):
        p(f'   layer {l}: PM1 / conservative force {r["pm_over_conservative_force"]:.2f}; depths positive '
          f'{100*r["depth_positive_share"]:.0f}% (p05/p50/p95 {r["depth_p05_p50_p95"][0]:+.3f} / '
          f'{r["depth_p05_p50_p95"][1]:+.3f} / {r["depth_p05_p50_p95"][2]:+.3f}); force from positive-depth modes '
          f'{100*r["force_share_from_positive_depths"]:.0f}%')
    p('')
    hit = res['depth_positive_share_all'] > 0.5
    p(f'DEPTH SIGNS   most depths positive (60%): {100*res["depth_positive_share_all"]:.1f}% of all L x K depths '
      f'positive -> {"HIT" if hit else "MISS"}')
    hit = S['repetition_spearman'] > 0.5
    p(f'REPETITION    Spearman(phi*, decay-weighted repeat count) > 0.5 (55%): {S["repetition_spearman"]:+.3f} '
      f'at layer {sl} -> {"HIT" if hit else "MISS"}')
    for l, r in sorted(res['layers'].items()):
        p(f'              layer {l}: all positions {r["repetition_spearman"]:+.3f}; repeated positions only '
          f'{r["repetition_spearman_repeats_only"]:+.3f} ({r["repeated_positions"]} of {r["positions"]})')
    hit = S['dp3_signed'] > 0.5
    p(f'DP3 ON MODES  Spearman(phi * a, leave-one-out force) > 0.5 (80%): {S["dp3_signed"]:+.3f} at layer {sl} '
      f'-> {"HIT" if hit else "MISS"}')
    for l, r in sorted(res['layers'].items()):
        p(f'              layer {l}: signed {r["dp3_signed"]:+.3f}; |phi * a| {r["dp3_abs"]:+.3f}')
    return '\n'.join(out)


if __name__ == '__main__':
    OUT = Path(sys.argv[1]); FOLDER = sys.argv[2]; EXTRA = sys.argv[3:]
    sys.path.insert(0, str(Path(__file__).parent))
    import gradcheck_vphi_xi_paths as G                     # reads OUT from sys.argv[1]
    from verify_vphi_xi_grad_path import build
    subs = [("POISSON_MODES        = 0", f"POISSON_MODES        = {PM_K}")]
    for e in EXTRA:
        if re.fullmatch(r'pmclip[\d.p]+', e):
            subs.append(("POISSON_MODE_CLIP    = 0.3", f"POISSON_MODE_CLIP    = {float(e[6:].replace('p', '.'))}"))
        else:
            raise SystemExit(f'unknown option {e!r}')
    for old, new in subs:
        assert G.cells['Cell 0:'].count(old) == 1, old
        G.cells['Cell 0:'] = G.cells['Cell 0:'].replace(old, new)
    torch.manual_seed(0)
    model, tag, _ = build(False, FOLDER, 'live', 'live')
    assert getattr(model, 'pm_mu', None) is not None, 'model built without Poisson modes'
    print(f"model: ...{tag[tag.find('cgqk'):]}\n")
    res = measure(model, np.load(G.VAL))
    txt = report(res)
    print(txt)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pm1_post_run_measurements.json').write_text(json.dumps(res, indent=1, default=float))
    (OUT / 'pm1_post_run_measurements_output.txt').write_text(txt + '\n')
    print(f'\nwrote {OUT}/pm1_post_run_measurements.json and _output.txt')
