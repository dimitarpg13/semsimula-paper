#!/usr/bin/env python3
"""Compare the learned aniso-Gaussian V_theta potential across two checkpoints.

Purpose
-------
Test whether second-order (damped-Verlet) training leaves a structural
imprint on the learned potential V_theta that first-order (Fock-G1,
delta:=0) training does not reproduce -- i.e. compare the *shape* of the
potential, not just the validation PPL.

This is an ARCHITECTURE-ONLY comparison: it reconstructs just the
V_theta submodule (AnisotropicDepthConditionedGaussianVTheta) from each
checkpoint's `model_state_dict` and evaluates its mixture components
(centres mu, diagonal precision a, low-rank factor B, mixture weights w)
at a shared, fixed bank of probe xi contexts. It does NOT require the
full FockMultiXiPARFLM model, TinyStories data, or a GPU -- both
checkpoints only need to share the same V_theta hyperparameters
(d, heads, wells/head, rank, layers), which is true by construction for
matched Fock-G1 vs second-order-anchor comparisons.

Because mu/a/w/B are *functions* of the xi context (linear/softplus/
softmax projections), not literal per-well parameter vectors, "the
potential's shape" is necessarily evaluated at some xi. We use each
layer's own depth_code (the trained, data-independent per-layer shift)
plus small Gaussian jitter as the probe bank, so the comparison reflects
each checkpoint's own learned depth-conditioning rather than an
arbitrary external xi.

Caveat: this does not yet reflect the *data-conditional* xi a real
forward pass would produce (that requires the full model + TinyStories
val data -- a natural follow-up once this architecture-only pass is
inspected).

Usage
-----
    python compare_vtheta_profiles.py CKPT_A CKPT_B \
        --label-a "first-order (Fock-G1)" --label-b "second-order anchor" \
        --out comparison_report.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Verbatim copy of the aniso-Gaussian V_theta classes (identical in both the
# second-order scaleup notebook and the Fock-G1 first-order notebook --
# notebooks/conservative_arch/scaleup/colab_fock_aniso_gaussian_fockreg_tinystories.ipynb
# and .../first_order_ablation/colab_fock_g1_aniso_gaussian_fockreg_tinystories.ipynb).
# Copied rather than imported because the class is defined inline in both
# notebooks, not in a shared .py module.
# ---------------------------------------------------------------------------


class AnisotropicMixtureGaussianVTheta(nn.Module):

    def __init__(self, d: int, K: int = 8, rank: int = 4,
                 w_scale: float = 1.0, xi_d=None,
                 init_log_precision=None, precision_max=None,
                 force_norm_max=None):
        super().__init__()
        self.d = d
        self.K = K
        self.rank = rank
        self.w_scale = w_scale
        self._precision_max = precision_max
        self._force_norm_max = force_norm_max
        in_d = xi_d if xi_d is not None else d

        self.mu_proj = nn.Linear(in_d, K * d)
        self.a_proj = nn.Linear(in_d, K * d)
        self.w_proj = nn.Linear(in_d, K)
        self.B_proj = nn.Linear(in_d, K * d * rank)

    def _components(self, xi):
        lead = xi.shape[:-1]
        mu = self.mu_proj(xi).view(*lead, self.K, self.d)
        a = (F.softplus(self.a_proj(xi)) + 1e-4).view(*lead, self.K, self.d)
        if self._precision_max is not None:
            a = a.clamp(max=self._precision_max)
        w = F.softmax(self.w_proj(xi), dim=-1) * self.w_scale
        B = self.B_proj(xi).view(*lead, self.K, self.d, self.rank)
        return mu, a, w, B


class AnisotropicMultiContextGaussianVTheta(nn.Module):

    def __init__(self, d, K, n_ctx, rank=4, w_scale=1.0,
                 init_log_precision=None, precision_max=None,
                 force_norm_max=None):
        super().__init__()
        self.d = d
        self.K = K
        self.n_ctx = n_ctx
        self.banks = nn.ModuleList(
            AnisotropicMixtureGaussianVTheta(
                d=d, K=K, rank=rank, w_scale=w_scale, xi_d=d,
                init_log_precision=init_log_precision,
                precision_max=precision_max,
                force_norm_max=force_norm_max,
            )
            for _ in range(n_ctx)
        )


class AnisotropicDepthConditionedGaussianVTheta(nn.Module):

    def __init__(self, d, K, n_ctx, n_layers, rank=4,
                 w_scale=1.0, init_log_precision=None,
                 precision_max=None, force_norm_max=None,
                 code_init_std=0.02):
        super().__init__()
        self.d = d
        self.K = K
        self.n_ctx = n_ctx
        self.n_layers = n_layers
        self.bank = AnisotropicMultiContextGaussianVTheta(
            d=d, K=K, n_ctx=n_ctx, rank=rank, w_scale=w_scale,
            init_log_precision=init_log_precision,
            precision_max=precision_max,
            force_norm_max=force_norm_max,
        )
        self.depth_code = nn.Parameter(
            torch.randn(n_layers, n_ctx, d) * code_init_std
        )

    @property
    def banks(self):
        return self.bank.banks


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_vtheta(ckpt_path, d, K, n_ctx, n_layers, rank, precision_max=None):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    full_sd = ckpt['model_state_dict']
    prefix = 'V_theta.'
    sub_sd = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}
    if not sub_sd:
        raise RuntimeError(f'No "{prefix}*" keys found in {ckpt_path}')

    vtheta = AnisotropicDepthConditionedGaussianVTheta(
        d=d, K=K, n_ctx=n_ctx, n_layers=n_layers, rank=rank,
        precision_max=precision_max,
    )
    missing, unexpected = vtheta.load_state_dict(sub_sd, strict=False)
    if missing or unexpected:
        print(f'[warn] {ckpt_path}: missing={missing}  unexpected={unexpected}',
              file=sys.stderr)
    vtheta.eval()

    meta = {
        'step': ckpt.get('step'),
        'val_ppl': ckpt.get('val_ppl'),
        'best_val_ppl': ckpt.get('best_val_ppl'),
        'gamma': ckpt.get('gamma'),
        'first_order': ckpt.get('first_order', False),
        'gamma_star': ckpt.get('gamma_star'),
        'xi_alphas': ckpt.get('xi_alphas'),
    }
    return vtheta, meta


# ---------------------------------------------------------------------------
# Structural diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_layer(vtheta: AnisotropicDepthConditionedGaussianVTheta,
                 layer_idx: int, n_probe: int, sigma: float, gen: torch.Generator):
    """Evaluate mu/a/w/B for one layer at depth_code[layer] + jitter."""
    n_ctx, d = vtheta.n_ctx, vtheta.d
    code = vtheta.depth_code[layer_idx]                      # (n_ctx, d)
    jitter = torch.randn(n_probe, n_ctx, d, generator=gen) * sigma
    xi = code.unsqueeze(0) + jitter                           # (n_probe, n_ctx, d)

    per_head = []
    for m, bank in enumerate(vtheta.banks):
        mu, a, w, B = bank._components(xi[:, m, :])           # (n_probe, K, ...)
        per_head.append(dict(mu=mu, a=a, w=w, B=B))
    return per_head, xi


def well_curvature_stats(a, B):
    """Eigenvalues of Sigma_k^{-1} = diag(a_k) + B_k B_k^T for a batch of wells.

    a: (..., d)   B: (..., d, r)   -> returns (..., d) eigenvalues, ascending.
    """
    d = a.shape[-1]
    diagA = torch.diag_embed(a)                    # (..., d, d)
    M = diagA + B @ B.transpose(-1, -2)             # (..., d, d)
    eigvals = torch.linalg.eigvalsh(M)              # (..., d), ascending
    return eigvals


def summarize_checkpoint(vtheta, n_probe=5, sigma=1.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    n_layers, n_ctx, K = vtheta.n_layers, vtheta.n_ctx, vtheta.K

    lam_min_all, lam_max_all, trace_all = [], [], []
    entropy_all = []
    nn_dist_all = []
    depth_code_norms = []

    for ell in range(n_layers):
        per_head, xi = probe_layer(vtheta, ell, n_probe, sigma, gen)
        depth_code_norms.append(vtheta.depth_code[ell].norm().item())

        centres_this_layer = []
        for head in per_head:
            a, B, w = head['a'], head['B'], head['w']          # (n_probe,K,d) / (n_probe,K,d,r) / (n_probe,K)
            eig = well_curvature_stats(a, B)                   # (n_probe,K,d)
            lam_min_all.append(eig[..., 0].reshape(-1))
            lam_max_all.append(eig[..., -1].reshape(-1))
            trace_all.append(eig.sum(dim=-1).reshape(-1))

            wc = w.clamp_min(1e-12)
            ent = -(wc * wc.log()).sum(dim=-1)                 # (n_probe,)
            entropy_all.append(ent)

            centres_this_layer.append(head['mu'])              # (n_probe,K,d)

        # nearest-neighbour distance among all K*n_ctx centres, per probe sample
        centres = torch.cat(centres_this_layer, dim=1)          # (n_probe, K*n_ctx, d)
        dmat = torch.cdist(centres, centres)                    # (n_probe, KC, KC)
        dmat.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
        nn_dist_all.append(dmat.min(dim=-1).values.reshape(-1))

    lam_min = torch.cat(lam_min_all)
    lam_max = torch.cat(lam_max_all)
    trace = torch.cat(trace_all)
    entropy = torch.cat(entropy_all)
    nn_dist = torch.cat(nn_dist_all)
    anisotropy = (lam_max / lam_min.clamp_min(1e-12))

    max_entropy = float(np.log(K))

    def stat(t):
        return {'mean': float(t.mean()), 'std': float(t.std()),
                'min': float(t.min()), 'max': float(t.max())}

    return {
        'lambda_min': stat(lam_min),
        'lambda_max': stat(lam_max),
        'trace': stat(trace),
        'anisotropy_ratio': stat(anisotropy),
        'well_weight_entropy': stat(entropy),
        'well_weight_entropy_max_possible': max_entropy,
        'well_weight_entropy_frac_of_max': float(entropy.mean()) / max_entropy,
        'nearest_neighbour_centre_dist': stat(nn_dist),
        'depth_code_norm_per_layer': depth_code_norms,
        'depth_code_norm_variation_across_layers': float(np.std(depth_code_norms)),
    }


def raw_weight_similarity(vtheta_a, vtheta_b):
    """Coarse parameter-space comparison: per-tensor norms and cosine sim."""
    sd_a = dict(vtheta_a.named_parameters())
    sd_b = dict(vtheta_b.named_parameters())
    out = {}
    for name in sd_a:
        ta, tb = sd_a[name].flatten(), sd_b[name].flatten()
        cos = float(F.cosine_similarity(ta.unsqueeze(0), tb.unsqueeze(0)).item())
        out[name] = {
            'norm_a': float(ta.norm()), 'norm_b': float(tb.norm()),
            'l2_dist': float((ta - tb).norm()), 'cosine_sim': cos,
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ckpt_a')
    ap.add_argument('ckpt_b')
    ap.add_argument('--label-a', default='A')
    ap.add_argument('--label-b', default='B')
    ap.add_argument('--d', type=int, default=256)
    ap.add_argument('--heads', type=int, default=4, help='n_ctx / xi channels')
    ap.add_argument('--wells', type=int, default=8, help='K, wells per head')
    ap.add_argument('--rank', type=int, default=4)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--precision-max', type=float, default=None,
                     help='defaults to 2/d, matching the training notebooks')
    ap.add_argument('--n-probe', type=int, default=20)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='optional JSON report path')
    args = ap.parse_args()

    precision_max = args.precision_max if args.precision_max is not None else 2.0 / args.d

    vt_a, meta_a = load_vtheta(args.ckpt_a, args.d, args.wells, args.heads,
                                args.layers, args.rank, precision_max)
    vt_b, meta_b = load_vtheta(args.ckpt_b, args.d, args.wells, args.heads,
                                args.layers, args.rank, precision_max)

    print(f'=== {args.label_a} ===  {args.ckpt_a}')
    print(f'  meta: {meta_a}')
    print(f'=== {args.label_b} ===  {args.ckpt_b}')
    print(f'  meta: {meta_b}')

    stats_a = summarize_checkpoint(vt_a, n_probe=args.n_probe, sigma=args.sigma, seed=args.seed)
    stats_b = summarize_checkpoint(vt_b, n_probe=args.n_probe, sigma=args.sigma, seed=args.seed)
    weight_sim = raw_weight_similarity(vt_a, vt_b)

    def pct(a, b):
        if b == 0:
            return float('nan')
        return 100.0 * (a - b) / abs(b)

    print(f'\n{"metric":38s} {args.label_a:>16s} {args.label_b:>16s} {"%diff (A vs B)":>16s}')
    print('-' * 90)
    for key in ['lambda_min', 'lambda_max', 'trace', 'anisotropy_ratio',
                'well_weight_entropy', 'nearest_neighbour_centre_dist']:
        a_mean = stats_a[key]['mean']
        b_mean = stats_b[key]['mean']
        print(f'{key + " (mean)":38s} {a_mean:16.5g} {b_mean:16.5g} {pct(a_mean, b_mean):15.1f}%')
    print(f'{"well_weight_entropy_frac_of_max":38s} '
          f'{stats_a["well_weight_entropy_frac_of_max"]:16.4f} '
          f'{stats_b["well_weight_entropy_frac_of_max"]:16.4f}')
    print(f'{"depth_code_norm_variation":38s} '
          f'{stats_a["depth_code_norm_variation_across_layers"]:16.5g} '
          f'{stats_b["depth_code_norm_variation_across_layers"]:16.5g}')

    print(f'\n{"raw weight tensor":38s} {"norm_A":>10s} {"norm_B":>10s} {"cos_sim":>10s}')
    print('-' * 72)
    for name, v in weight_sim.items():
        print(f'{name:38s} {v["norm_a"]:10.3f} {v["norm_b"]:10.3f} {v["cosine_sim"]:10.4f}')

    report = {
        'ckpt_a': str(args.ckpt_a), 'ckpt_b': str(args.ckpt_b),
        'label_a': args.label_a, 'label_b': args.label_b,
        'meta_a': meta_a, 'meta_b': meta_b,
        'config': vars(args),
        'stats_a': stats_a, 'stats_b': stats_b,
        'raw_weight_similarity': weight_sim,
    }
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f'\nSaved report to {args.out}')


if __name__ == '__main__':
    main()
