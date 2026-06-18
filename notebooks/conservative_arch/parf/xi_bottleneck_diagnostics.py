"""
Xi-bottleneck diagnostics for Gaussian mixture V_theta.

Three lightweight metrics that expose whether the model is bottlenecked
by the number of Gaussian wells (K_MIX), the number/horizon of Xi
channels, or the mixing-weight projection capacity:

1. **Well collapse** — decode each well centroid mu_k(xi) through the LM
   head; if multiple wells decode to the same token, K is too small.
   Reports the number of *unique* decoded tokens vs K, and the
   per-component decode list.

2. **Xi sensitivity** — ||dV/d(xi_k)||  per channel, averaged over a
   batch.  If the longest-horizon channel dominates, the model wants
   more long-range context.

3. **Weight entropy** — H(w_k(xi)) averaged over the batch.  Near
   log(K) = uniform: the model cannot differentiate wells (context
   bottleneck or K too small).  Low entropy = sharp selection
   (healthy, or K too large with many dead wells).

All three are designed to run inside a torch.no_grad() / eval context
using a single validation batch, with minimal extra VRAM.

Usage (inside a notebook eval loop)::

    from xi_bottleneck_diagnostics import run_xi_diagnostics

    diag = run_xi_diagnostics(model, val_batch_x, embedding_weight, K_xi=4, d=384)
    print(diag['summary'])
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F


def _get_inner_gaussian(model) -> Optional[Any]:
    """Walk model.V_theta → inner to reach MixtureGaussianVTheta."""
    vt = getattr(model, 'V_theta', None)
    if vt is None:
        return None
    inner = getattr(vt, 'inner', None)
    if inner is not None and hasattr(inner, '_components'):
        return inner
    if hasattr(vt, '_components'):
        return vt
    return None


# ── 1. Well collapse ────────────────────────────────────────────────────

def well_collapse(
    model,
    xis: torch.Tensor,
    h: torch.Tensor,
    embedding_weight: torch.Tensor,
    tokenizer=None,
) -> Dict[str, Any]:
    """Decode each well centroid through the LM head.

    Parameters
    ----------
    model : FockMultiXiPARFLM (or any model with .V_theta.inner)
    xis : (B, T, K_xi, d)  — pre-computed xi channels
    h : (B, T, d) — hidden states
    embedding_weight : (V, d) — tied embedding / LM head weight
    tokenizer : optional, for decoding token ids to strings

    Returns
    -------
    dict with 'K', 'unique_tokens', 'collapse_ratio', 'decoded_ids',
    'decoded_tokens' (if tokenizer given), 'per_component_counts'.
    """
    inner = _get_inner_gaussian(model)
    if inner is None:
        return {'error': 'V_theta has no _components method'}

    B, T, K_xi, d = xis.shape
    xi_flat = xis.reshape(B, T, K_xi * d)

    with torch.no_grad():
        mu, _a, w = inner._components(xi_flat)  # mu: (B, T, K, d), w: (B, T, K)

    K = mu.shape[-2]
    mu_flat = mu.reshape(-1, d)       # (B*T*K, d)
    w_flat = w.reshape(-1, K)         # (B*T, K)

    logits = mu_flat @ embedding_weight.T   # (B*T*K, V)
    decoded_ids = logits.argmax(dim=-1)     # (B*T*K,)

    decoded_per_position = decoded_ids.reshape(-1, K)  # (B*T, K)

    unique_per_position = []
    for row in decoded_per_position:
        unique_per_position.append(len(row.unique()))
    avg_unique = sum(unique_per_position) / len(unique_per_position)

    global_decoded = decoded_ids.unique()
    n_global_unique = len(global_decoded)

    mean_resp = w_flat.mean(dim=0).cpu().tolist()

    centroid_per_k = mu.mean(dim=(0, 1))    # (K, d)
    centroid_logits = centroid_per_k @ embedding_weight.T
    centroid_ids = centroid_logits.argmax(dim=-1).cpu().tolist()

    result: Dict[str, Any] = {
        'K': K,
        'avg_unique_per_position': round(avg_unique, 2),
        'global_unique_tokens': n_global_unique,
        'collapse_ratio': round(1.0 - avg_unique / K, 3),
        'centroid_decoded_ids': centroid_ids,
        'mean_responsibility': [round(r, 4) for r in mean_resp],
    }

    if tokenizer is not None:
        result['centroid_decoded_tokens'] = [
            tokenizer.decode([tid]).strip() for tid in centroid_ids
        ]
        per_pos_decoded = []
        for row in decoded_per_position[:min(4, len(decoded_per_position))]:
            per_pos_decoded.append(
                [tokenizer.decode([tid.item()]).strip() for tid in row]
            )
        result['sample_position_decodes'] = per_pos_decoded

    return result


# ── 2. Xi sensitivity ──────────────────────────────────────────────────

def xi_sensitivity(
    model,
    x: torch.Tensor,
    K_xi: int,
    d: int,
) -> Dict[str, Any]:
    """Compute ||dV/d(xi_k)|| per channel.

    Runs one forward pass with gradients enabled on xi only.

    Parameters
    ----------
    model : the full model (needs .V_theta and .xi_module)
    x : (B, T) token ids
    K_xi : number of Xi channels
    d : hidden dim

    Returns
    -------
    dict with 'per_channel_grad_norm' (list of K_xi floats),
    'dominant_channel' index, 'dominance_ratio'.
    """
    model.eval()
    with torch.no_grad():
        h0 = model._embed(x)
        h_L, _ = model._stack_forward(h0, x, return_trajectory=False)
        h_det = h_L.detach()

    xi_input = h_det
    with torch.no_grad():
        xis_det = model.xi_module(xi_input)  # (B, T, K_xi, d)

    xis = xis_det.detach().clone().requires_grad_(True)
    B, T, _K, _d = xis.shape
    xi_flat = xis.reshape(B, T, K_xi * d)

    inner = _get_inner_gaussian(model)
    if inner is None:
        return {'error': 'V_theta has no _components method'}

    with torch.enable_grad():
        V = inner(xi_flat, h_det)  # (B, T, 1)
        V_sum = V.sum()
        grad_xis = torch.autograd.grad(V_sum, xis, create_graph=False)[0]

    per_channel_norms = []
    for k in range(K_xi):
        gk = grad_xis[:, :, k, :]  # (B, T, d)
        norm_k = gk.norm(dim=-1).mean().item()
        per_channel_norms.append(round(norm_k, 6))

    max_norm = max(per_channel_norms)
    dominant = per_channel_norms.index(max_norm)
    min_norm = min(n for n in per_channel_norms if n > 0) if any(n > 0 for n in per_channel_norms) else 1e-12
    dominance_ratio = round(max_norm / max(min_norm, 1e-12), 2)

    return {
        'per_channel_grad_norm': per_channel_norms,
        'dominant_channel': dominant,
        'dominance_ratio': dominance_ratio,
    }


# ── 3. Weight entropy ──────────────────────────────────────────────────

def weight_entropy(
    model,
    xis: torch.Tensor,
    K_xi: int,
    d: int,
) -> Dict[str, Any]:
    """Compute H(w_k(xi)) averaged over batch positions.

    Parameters
    ----------
    model : needs .V_theta.inner with w_proj
    xis : (B, T, K_xi, d)
    K_xi, d : channel count and hidden dim

    Returns
    -------
    dict with 'mean_entropy', 'max_entropy' (=log K), 'entropy_ratio'
    (0=sharp, 1=uniform), 'per_component_mean_weight'.
    """
    inner = _get_inner_gaussian(model)
    if inner is None:
        return {'error': 'V_theta has no _components method'}

    B, T, _K, _d = xis.shape
    xi_flat = xis.reshape(B, T, K_xi * d)

    with torch.no_grad():
        _mu, _a, w = inner._components(xi_flat)  # w: (B, T, K)

    K = w.shape[-1]
    w_clamped = w.clamp(min=1e-12)
    w_normed = w_clamped / w_clamped.sum(dim=-1, keepdim=True)
    H = -(w_normed * w_normed.log()).sum(dim=-1)  # (B, T)
    mean_H = H.mean().item()
    max_H = math.log(K)
    ratio = mean_H / max_H if max_H > 0 else 0.0

    mean_w = w.mean(dim=(0, 1)).cpu().tolist()

    return {
        'mean_entropy': round(mean_H, 4),
        'max_entropy': round(max_H, 4),
        'entropy_ratio': round(ratio, 4),
        'per_component_mean_weight': [round(w, 4) for w in mean_w],
    }


# ── Combined runner ─────────────────────────────────────────────────────

def run_xi_diagnostics(
    model,
    x: torch.Tensor,
    embedding_weight: torch.Tensor,
    K_xi: int,
    d: int,
    tokenizer=None,
) -> Dict[str, Any]:
    """Run all three diagnostics on a single validation batch.

    Parameters
    ----------
    model : the full model
    x : (B, T) token ids
    embedding_weight : (V, d) tied embedding weight
    K_xi : number of Xi channels
    d : hidden dim
    tokenizer : optional, for human-readable decodes

    Returns
    -------
    dict with keys 'well_collapse', 'xi_sensitivity', 'weight_entropy',
    and a formatted 'summary' string.
    """
    model.eval()

    with torch.no_grad():
        h0 = model._embed(x)
        h_L, _ = model._stack_forward(h0, x, return_trajectory=False)
        h_det = h_L.detach()
        xis = model.xi_module(h_det)  # (B, T, K_xi, d)

    wc = well_collapse(model, xis, h_det, embedding_weight, tokenizer)
    ws = xi_sensitivity(model, x, K_xi, d)
    we = weight_entropy(model, xis, K_xi, d)

    alphas = model.xi_alpha_values()
    horizons = [round(1.0 / (1.0 - a), 1) if a < 1.0 else float('inf') for a in alphas]

    lines = [
        '─── Xi Bottleneck Diagnostics ───',
        '',
        f'  Well collapse:  K={wc.get("K","?")}  '
        f'avg_unique={wc.get("avg_unique_per_position","?")}  '
        f'collapse_ratio={wc.get("collapse_ratio","?")}',
        f'    centroid decodes: {wc.get("centroid_decoded_ids", "?")}',
        f'    mean responsibility: {wc.get("mean_responsibility", "?")}',
    ]
    if 'centroid_decoded_tokens' in wc:
        lines.append(f'    centroid tokens: {wc["centroid_decoded_tokens"]}')
    if 'sample_position_decodes' in wc:
        for i, row in enumerate(wc['sample_position_decodes'][:2]):
            lines.append(f'    pos[{i}] decodes: {row}')

    lines.append('')
    lines.append(f'  Xi sensitivity (||dV/dxi_k||):')
    for k, (norm, alpha, hz) in enumerate(
        zip(ws.get('per_channel_grad_norm', []), alphas, horizons)
    ):
        marker = ' <-- dominant' if k == ws.get('dominant_channel', -1) else ''
        lines.append(f'    ch{k}: {norm:.6f}  (alpha={alpha:.3f}, ~{hz:.0f} tok){marker}')
    lines.append(f'    dominance ratio: {ws.get("dominance_ratio", "?")}x')

    lines.append('')
    lines.append(
        f'  Weight entropy:  H={we.get("mean_entropy","?"):.4f}  '
        f'H_max={we.get("max_entropy","?"):.4f}  '
        f'ratio={we.get("entropy_ratio","?")}  '
        f'(0=sharp, 1=uniform)'
    )
    lines.append(f'    per-component mean w: {we.get("per_component_mean_weight", "?")}')
    lines.append('─' * 34)

    summary = '\n'.join(lines)

    return {
        'well_collapse': wc,
        'xi_sensitivity': ws,
        'weight_entropy': we,
        'xi_alphas': alphas,
        'xi_horizons': horizons,
        'summary': summary,
    }


# ── Smoke test ──────────────────────────────────────────────────────────

def _smoke():
    """Verify diagnostics with a tiny random model."""
    import sys
    sys.path.insert(0, '.')

    d, K_mix, K_xi, V_size = 16, 4, 4, 100

    from model_gaussian_vtheta import MixtureGaussianVTheta, GaussianVThetaMultiXiAdapter

    inner = MixtureGaussianVTheta(d=d, K=K_mix, xi_d=K_xi * d)
    adapter = GaussianVThetaMultiXiAdapter(inner, K=K_xi, d=d)

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = torch.nn.Embedding(V_size, d)
            self.V_theta = adapter
            self._alpha = torch.nn.Parameter(
                torch.tensor([0.25, 0.50, 0.75, 0.95])
            )

        def _embed(self, x):
            return self.E(x)

        def _stack_forward(self, h0, x, return_trajectory=False):
            return h0, None

        def xi_module(self, h):
            B, T, _d = h.shape
            return h.unsqueeze(2).expand(B, T, K_xi, _d).clone()

        def xi_alpha_values(self):
            return torch.sigmoid(self._alpha).tolist()

    model = _FakeModel()
    model.eval()

    B, T = 2, 8
    x = torch.randint(0, V_size, (B, T))
    emb_w = model.E.weight.data

    print('Running xi_bottleneck_diagnostics smoke test ...\n')

    wc_result = well_collapse(
        model,
        model.xi_module(model._embed(x)),
        model._embed(x),
        emb_w,
    )
    assert 'K' in wc_result, f'well_collapse missing K: {wc_result}'
    assert wc_result['K'] == K_mix
    print(f'  well_collapse: OK  (K={wc_result["K"]}, '
          f'unique={wc_result["avg_unique_per_position"]})')

    ws_result = xi_sensitivity(model, x, K_xi, d)
    assert 'per_channel_grad_norm' in ws_result
    assert len(ws_result['per_channel_grad_norm']) == K_xi
    print(f'  xi_sensitivity: OK  (norms={ws_result["per_channel_grad_norm"]})')

    we_result = weight_entropy(
        model,
        model.xi_module(model._embed(x)),
        K_xi, d,
    )
    assert 'mean_entropy' in we_result
    print(f'  weight_entropy: OK  (H={we_result["mean_entropy"]:.4f}, '
          f'ratio={we_result["entropy_ratio"]:.4f})')

    full = run_xi_diagnostics(model, x, emb_w, K_xi, d)
    assert 'summary' in full
    print(f'\n{full["summary"]}')
    print('\nAll smoke tests passed.')


if __name__ == '__main__':
    _smoke()
