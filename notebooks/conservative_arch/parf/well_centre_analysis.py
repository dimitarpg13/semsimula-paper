"""
Well-centre analysis for the structured V_theta family.

Goal
----
The Gaussian variant learns *context-dependent* well centres
``mu_k(xi) = mu_proj(xi)`` (see ``model_gaussian_vtheta.MixtureGaussianVTheta``),
whereas the SARF variant uses *static* frozen anchors.  This module extracts
the converged Gaussian centre cloud ``{mu_k(xi_t)}`` over a corpus, derives the
responsibility-weighted modes of that cloud (the empirical "settled centres"),
and scores several closed-form anchor-placement rules against those modes.

The winning rule is a candidate *semi-empirical law* for choosing SARF anchors
analytically, replacing (or improving on) the current PMI-peak heuristic.

The functions here are deliberately dependency-light (torch + numpy only) and
unit-testable on a tiny random model -- see ``_smoke()`` at the bottom, which is
run when this file is executed directly.

Shared-basis note
------------------
The model uses tied embeddings (``logits = h_L @ E.weight.T``), so token
embeddings ``E[v]`` and hidden states / centres ``mu_k(xi)`` live in the *same*
``R^d`` basis.  Geometric comparison between rule anchors (token embeddings) and
empirical centres is therefore meaningful, provided both are placed on the same
scale.  We follow the Phase-5 SARF builder and row-standardise every vector
(mean 0, unit variance per dim -> ``||v|| ~ sqrt(d)``) before geometric metrics.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def standardize_rows(a: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Row-standardise to mean 0, unit variance per dim (||row|| ~ sqrt(d)).

    Matches the SARF anchor re-normalisation in the Phase-5 builder so that
    token-embedding anchors and learned centres are compared on the same scale.
    """
    mu = a.mean(dim=-1, keepdim=True)
    sd = a.std(dim=-1, keepdim=True)
    return (a - mu) / (sd + eps)


# ---------------------------------------------------------------------------
# Centre-cloud extraction from a converged model
# ---------------------------------------------------------------------------
def _sample_x(ids: np.ndarray, batch_size: int, block_size: int,
              rng: np.random.Generator) -> np.ndarray:
    """Uniform random (B, block_size) token windows (inputs only)."""
    n = len(ids) - block_size - 1
    starts = rng.integers(0, n, size=batch_size)
    x = np.stack([ids[s:s + block_size] for s in starts])
    return x.astype(np.int64)


@torch.no_grad()
def extract_centre_cloud(
    model,
    ids: np.ndarray,
    *,
    n_batches: int = 32,
    batch_size: int = 8,
    block_size: int = 512,
    device: str = "cpu",
    max_points: Optional[int] = 200_000,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Run the converged model over ``ids`` and collect the centre cloud.

    Returns a dict of CPU tensors:
      - ``mu``  (N, K, d)  per-token, per-component centres mu_k(xi)
      - ``w``   (N, K)     responsibilities (softmax of w_proj logits)
      - ``h_L`` (N, d)     final hidden states (the data manifold sample)

    where ``N = n_batches * batch_size * block_size`` (optionally subsampled to
    ``max_points``).  The model forward is wrapped in ``enable_grad`` because the
    Verlet integrator computes forces via ``torch.autograd.grad`` internally
    (mirroring the notebook ``evaluate()``); outputs are detached immediately.
    """
    model.eval()
    rng = np.random.default_rng(seed)

    mu_chunks: List[torch.Tensor] = []
    w_chunks: List[torch.Tensor] = []
    h_chunks: List[torch.Tensor] = []

    inner = model.V_theta.inner
    K_xi = model.V_theta.K

    for _ in range(n_batches):
        xb = _sample_x(ids, batch_size, block_size, rng)
        x = torch.from_numpy(xb).to(device)

        with torch.enable_grad():
            h0 = model._embed(x)
            h_L, _ = model._stack_forward(h0, x, return_trajectory=False)
        h_L = h_L.detach()

        xis = model.xi_module(h_L)                        # (B, T, K_xi, d)
        B, T, _, d = xis.shape
        centres = model.V_theta.attractor_centres(xis)    # (B, T, K, d)
        xi_flat = xis.reshape(B, T, K_xi * d)
        w = torch.softmax(inner.w_proj(xi_flat), dim=-1)  # (B, T, K)

        K = centres.shape[-2]
        mu_chunks.append(centres.reshape(B * T, K, d).cpu())
        w_chunks.append(w.reshape(B * T, K).cpu())
        h_chunks.append(h_L.reshape(B * T, d).cpu())

    mu = torch.cat(mu_chunks, dim=0)
    w = torch.cat(w_chunks, dim=0)
    h = torch.cat(h_chunks, dim=0)

    if max_points is not None and mu.shape[0] > max_points:
        sel = torch.from_numpy(
            rng.choice(mu.shape[0], size=max_points, replace=False)
        )
        mu, w, h = mu[sel], w[sel], h[sel]

    return {"mu": mu, "w": w, "h_L": h}


def flatten_weighted_cloud(cloud: Dict[str, torch.Tensor]):
    """Flatten (N, K, d) centres + (N, K) weights into (N*K, d), (N*K,).

    Each per-token, per-component centre becomes one weighted point, with the
    responsibility w_k(xi) as its importance weight.  This is the cloud the
    empirical modes and all geometric metrics operate on.
    """
    mu = cloud["mu"]
    w = cloud["w"]
    N, K, d = mu.shape
    points = mu.reshape(N * K, d)
    weights = w.reshape(N * K)
    return points, weights


# ---------------------------------------------------------------------------
# Weighted k-means (no sklearn dependency)
# ---------------------------------------------------------------------------
def weighted_kmeans(
    points: torch.Tensor,
    weights: torch.Tensor,
    n_clusters: int,
    *,
    n_iters: int = 50,
    tol: float = 1e-5,
    seed: int = 0,
) -> torch.Tensor:
    """Lloyd's algorithm with per-point importance weights.

    Returns ``(n_clusters, d)`` centroids.  Empty clusters are re-seeded to the
    highest-weight point to avoid NaNs.
    """
    points = points.float()
    weights = weights.float().clamp(min=0.0)
    N, d = points.shape
    g = torch.Generator().manual_seed(seed)

    probs = weights / weights.sum().clamp(min=1e-12)
    init_idx = torch.multinomial(
        probs, min(n_clusters, N), replacement=False, generator=g
    )
    centroids = points[init_idx].clone()
    if centroids.shape[0] < n_clusters:  # tiny-N fallback
        extra = points[torch.randint(0, N, (n_clusters - centroids.shape[0],),
                                     generator=g)]
        centroids = torch.cat([centroids, extra], dim=0)

    for _ in range(n_iters):
        d2 = torch.cdist(points, centroids)            # (N, C)
        assign = d2.argmin(dim=1)
        new_centroids = centroids.clone()
        for c in range(n_clusters):
            mask = assign == c
            if mask.any():
                w_c = weights[mask].unsqueeze(1)
                denom = w_c.sum().clamp(min=1e-12)
                new_centroids[c] = (points[mask] * w_c).sum(0) / denom
            else:
                new_centroids[c] = points[weights.argmax()]
        shift = (new_centroids - centroids).norm()
        centroids = new_centroids
        if shift < tol:
            break
    return centroids


# ---------------------------------------------------------------------------
# LM-head decode
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_via_lm_head(
    vecs: torch.Tensor, E: torch.Tensor, top_k: int = 5
):
    """Decode hidden-space vectors to tokens via the tied LM head ``vecs @ E.T``.

    Returns ``(token_ids, probs)`` each of shape ``(n_vecs, top_k)``.
    """
    logits = vecs.float() @ E.float().t()
    probs = torch.softmax(logits, dim=-1)
    top = probs.topk(top_k, dim=-1)
    return top.indices, top.values


# ---------------------------------------------------------------------------
# Candidate anchor rules
# ---------------------------------------------------------------------------
def compute_token_counts(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    return np.bincount(ids.astype(np.int64), minlength=vocab_size).astype(np.float64)


def compute_pmi_peak_tokens(
    ids: np.ndarray,
    vocab_size: int,
    n_anchors: int,
    *,
    window: int = 5,
    top_v: int = 8192,
) -> np.ndarray:
    """Return the ``n_anchors`` token ids with the highest PMI co-occurrence peak.

    Faithful port of the Phase-5 SARF builder's anchor selection.
    """
    token_counts = np.bincount(ids.astype(np.int64), minlength=vocab_size)
    top_v = min(top_v, int((token_counts > 0).sum()))
    top_v_ids = np.argsort(-token_counts)[:top_v]
    id_to_local = np.full(vocab_size, -1, dtype=np.int64)
    id_to_local[top_v_ids] = np.arange(top_v)

    cooc = np.zeros((top_v, top_v), dtype=np.float64)
    local_ids = id_to_local[ids.astype(np.int64)]
    for offset in range(1, window + 1):
        a = local_ids[:-offset]
        b = local_ids[offset:]
        valid = (a >= 0) & (b >= 0)
        np.add.at(cooc, (a[valid], b[valid]), 1.0)
    cooc = cooc + cooc.T

    row_sums = cooc.sum(axis=1, keepdims=True)
    total = cooc.sum()
    expected = row_sums * row_sums.T / max(total, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(cooc / np.maximum(expected, 1e-12))
    pmi = np.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=-20.0)

    np.fill_diagonal(pmi, -np.inf)
    pmi_peaks = pmi.max(axis=1)
    anchor_local_ids = np.argsort(-pmi_peaks)[:n_anchors]
    return top_v_ids[anchor_local_ids]


def _pca_shell_anchors(
    points: torch.Tensor, weights: torch.Tensor, n_anchors: int, c: float = 1.0
) -> torch.Tensor:
    """Place anchors on the principal-axis shell of the (weighted) cloud.

    For the top ``ceil(n_anchors/2)`` eigenvectors v_i of the weighted
    covariance, emit a +/- pair at ``mean +/- c * sqrt(lambda_i) * v_i``.
    """
    points = points.float()
    w = weights.float().clamp(min=0.0)
    w = w / w.sum().clamp(min=1e-12)
    mean = (points * w.unsqueeze(1)).sum(0)
    centred = points - mean
    cov = (centred * w.unsqueeze(1)).t() @ centred   # (d, d), weighted
    evals, evecs = torch.linalg.eigh(cov)            # ascending
    n_pairs = (n_anchors + 1) // 2
    top = list(range(cov.shape[0] - 1, cov.shape[0] - 1 - n_pairs, -1))
    anchors = []
    for i in top:
        lam = evals[i].clamp(min=0.0).sqrt()
        v = evecs[:, i]
        anchors.append(mean + c * lam * v)
        anchors.append(mean - c * lam * v)
    return torch.stack(anchors[:n_anchors], dim=0)


def build_rule_anchors(
    rule: str,
    *,
    E: torch.Tensor,
    n_anchors: int,
    ids: Optional[np.ndarray] = None,
    vocab_size: Optional[int] = None,
    cloud_points: Optional[torch.Tensor] = None,
    cloud_weights: Optional[torch.Tensor] = None,
    h_L: Optional[torch.Tensor] = None,
    surprisal: Optional[np.ndarray] = None,
    pca_scale: float = 1.0,
    seed: int = 0,
) -> torch.Tensor:
    """Construct ``(n_anchors, d)`` anchors for a candidate placement rule.

    Rules
    -----
    - ``"pmi"``   R0  token embeddings of top PMI-peak tokens (current SARF).
    - ``"freq"``  R1  token embeddings of top unigram-frequency tokens.
    - ``"hmodes"`` R2 weighted k-means centroids of the h_L manifold sample.
    - ``"pca"``   R3  principal-axis shell of the centre cloud.
    - ``"surprisal"`` R4 token embeddings of top ``count * surprisal`` tokens
                       (total information contribution; ties to the logfreq mass).

    Anchors are returned in the raw embedding/hidden basis; the caller / scorer
    row-standardises before geometric comparison.
    """
    rule = rule.lower()
    if rule == "pmi":
        assert ids is not None and vocab_size is not None
        tok = compute_pmi_peak_tokens(ids, vocab_size, n_anchors)
        return E[torch.from_numpy(tok).long()].clone()

    if rule == "freq":
        assert ids is not None and vocab_size is not None
        counts = compute_token_counts(ids, vocab_size)
        tok = np.argsort(-counts)[:n_anchors]
        return E[torch.from_numpy(tok).long()].clone()

    if rule == "surprisal":
        assert ids is not None and vocab_size is not None
        counts = compute_token_counts(ids, vocab_size)
        if surprisal is None:
            p = (counts + 1.0) / (counts.sum() + vocab_size)
            surprisal = -np.log(p)
        info = counts * surprisal
        tok = np.argsort(-info)[:n_anchors]
        return E[torch.from_numpy(tok).long()].clone()

    if rule == "hmodes":
        assert h_L is not None
        return weighted_kmeans(
            h_L, torch.ones(h_L.shape[0]), n_anchors, seed=seed
        )

    if rule == "pca":
        assert cloud_points is not None and cloud_weights is not None
        return _pca_shell_anchors(cloud_points, cloud_weights, n_anchors,
                                  c=pca_scale)

    raise ValueError(f"Unknown rule: {rule!r}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Symmetric Chamfer distance between two point sets (mean nearest-neighbour)."""
    d2 = torch.cdist(a.float(), b.float())
    a_to_b = d2.min(dim=1).values.mean()
    b_to_a = d2.min(dim=0).values.mean()
    return float(0.5 * (a_to_b + b_to_a))


def mass_coverage(
    anchors: torch.Tensor,
    cloud: torch.Tensor,
    weights: torch.Tensor,
    sigma: float,
) -> float:
    """Responsibility-weighted fraction of cloud mass within ``sigma`` of an anchor."""
    d2 = torch.cdist(cloud.float(), anchors.float())
    nearest = d2.min(dim=1).values
    covered = (nearest <= sigma).float()
    w = weights.float().clamp(min=0.0)
    return float((covered * w).sum() / w.sum().clamp(min=1e-12))


def decode_jaccard(
    anchors: torch.Tensor,
    target_modes: torch.Tensor,
    E: torch.Tensor,
    *,
    top_k: int = 1,
) -> float:
    """Jaccard overlap of the token sets the anchors and modes decode to."""
    a_tok, _ = decode_via_lm_head(anchors, E, top_k=top_k)
    m_tok, _ = decode_via_lm_head(target_modes, E, top_k=top_k)
    a_set = set(a_tok.reshape(-1).tolist())
    m_set = set(m_tok.reshape(-1).tolist())
    if not a_set and not m_set:
        return 1.0
    return len(a_set & m_set) / max(len(a_set | m_set), 1)


def default_sigma(cloud: torch.Tensor, target_modes: torch.Tensor) -> float:
    """Median distance from cloud points to their nearest target mode.

    Computed once from the TARGET so the coverage threshold is identical across
    all candidate rules.
    """
    d2 = torch.cdist(cloud.float(), target_modes.float())
    return float(d2.min(dim=1).values.median())


def score_rule(
    anchors: torch.Tensor,
    target_modes: torch.Tensor,
    cloud_points: torch.Tensor,
    cloud_weights: torch.Tensor,
    E: torch.Tensor,
    *,
    sigma: Optional[float] = None,
    decode_top_k: int = 1,
    standardize: bool = True,
) -> Dict[str, float]:
    """Score one rule's anchors against the empirical target modes.

    All geometric quantities are computed in the row-standardised basis so that
    token embeddings and learned centres share the ``||v|| ~ sqrt(d)`` scale.
    """
    if standardize:
        anchors_n = standardize_rows(anchors)
        modes_n = standardize_rows(target_modes)
        cloud_n = standardize_rows(cloud_points)
        E_n = standardize_rows(E)
    else:
        anchors_n, modes_n, cloud_n, E_n = anchors, target_modes, cloud_points, E

    if sigma is None:
        sigma = default_sigma(cloud_n, modes_n)

    return {
        "chamfer": chamfer_distance(anchors_n, modes_n),
        "coverage": mass_coverage(anchors_n, cloud_n, cloud_weights, sigma),
        "decode_jaccard": decode_jaccard(anchors_n, modes_n, E_n,
                                         top_k=decode_top_k),
        "sigma": float(sigma),
    }


def evaluate_all_rules(
    *,
    E: torch.Tensor,
    cloud: Dict[str, torch.Tensor],
    target_modes: torch.Tensor,
    n_anchors: int,
    ids: Optional[np.ndarray] = None,
    vocab_size: Optional[int] = None,
    surprisal: Optional[np.ndarray] = None,
    rules: Sequence[str] = ("pmi", "freq", "hmodes", "pca", "surprisal"),
    decode_top_k: int = 1,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Build anchors for each rule and score them against the shared TARGET.

    The coverage ``sigma`` is fixed once from the TARGET modes so the threshold
    is identical across rules.
    """
    points, weights = flatten_weighted_cloud(cloud)
    cloud_n = standardize_rows(points)
    modes_n = standardize_rows(target_modes)
    sigma = default_sigma(cloud_n, modes_n)

    out: Dict[str, Dict[str, float]] = {}
    for rule in rules:
        anchors = build_rule_anchors(
            rule, E=E, n_anchors=n_anchors, ids=ids, vocab_size=vocab_size,
            cloud_points=points, cloud_weights=weights, h_L=cloud["h_L"],
            surprisal=surprisal, seed=seed,
        )
        out[rule] = score_rule(
            anchors, target_modes, points, weights, E,
            sigma=sigma, decode_top_k=decode_top_k,
        )
    return out


# ---------------------------------------------------------------------------
# Smoke test (tiny random model, no checkpoint needed)
# ---------------------------------------------------------------------------
def _smoke() -> None:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    ca_dir = repo_root / "notebooks" / "conservative_arch"
    for sub in ["", "parf", "multixi", "scaleup"]:
        p = str(ca_dir / sub) if sub else str(ca_dir)
        if p not in sys.path:
            sys.path.insert(0, p)

    import math
    from model_fock_parf_multixi import FockMultiXiPARFLM, FockMultiXiPARFConfig
    from model_gaussian_vtheta import (
        MixtureGaussianVTheta, GaussianVThetaMultiXiAdapter,
    )

    VOCAB, d, K_xi = 256, 64, 4
    cfg = FockMultiXiPARFConfig(
        vocab_size=VOCAB, d=d, max_len=128, L=2, v_hidden=64, v_depth=2, dt=1.0,
        mass_mode="global", init_gamma=1.0, fixed_gamma=0.30,
        causal_force=True, ln_after_step=True,
        xi_channels=K_xi, xi_alpha_inits=[0.25, 0.5, 0.75, 0.95],
        xi_learnable=True, xi_alpha_init_mode="explicit",
        v_phi_kind="structural_competitive", v_phi_phi_hidden=32,
        v_phi_theta_hidden=32, top_k=4, score_head_hidden=16,
        gumbel_tau_init=1.0, gumbel_tau_min=0.3, gumbel_noise=True,
        use_gathered_v_phi=True, use_layer_checkpoint=False,
        ln_before_distance=True, per_layer_v_phi_scale=True, fock_version="v2",
        n_registers=4, register_salience_decay=0.5,
        register_salience_threshold=0.005, creation_gate_hidden=32,
        stack_discipline=True, d_k=32, tau_create_init=8.0,
        reverse_channel=True, per_register_tau=True, per_register_keys=True,
        ortho_register_init=True,
    )
    model = FockMultiXiPARFLM(cfg)
    K_MIX = 8
    inner = MixtureGaussianVTheta(
        d=d, K=K_MIX, w_scale=1.0, xi_d=K_xi * d,
        init_log_precision=-math.log(d), precision_max=2.0 / d,
    )
    model.V_theta = GaussianVThetaMultiXiAdapter(inner, K=K_xi, d=d)

    rng = np.random.default_rng(0)
    ids = rng.integers(0, VOCAB, size=50_000).astype(np.int64)

    print("extracting centre cloud ...")
    cloud = extract_centre_cloud(
        model, ids, n_batches=4, batch_size=4, block_size=32,
        device="cpu", max_points=5_000, seed=0,
    )
    print(f"  mu={tuple(cloud['mu'].shape)} w={tuple(cloud['w'].shape)} "
          f"h_L={tuple(cloud['h_L'].shape)}")

    points, weights = flatten_weighted_cloud(cloud)
    N_S = 16
    target = weighted_kmeans(points, weights, N_S, seed=0)
    assert target.shape == (N_S, d)
    print(f"  target modes: {tuple(target.shape)}")

    E = model.E.weight.data.detach()
    counts = compute_token_counts(ids, VOCAB)
    p = (counts + 1.0) / (counts.sum() + VOCAB)
    surprisal = -np.log(p)

    results = evaluate_all_rules(
        E=E, cloud=cloud, target_modes=target, n_anchors=N_S,
        ids=ids, vocab_size=VOCAB, surprisal=surprisal,
    )

    print("\n  rule scores (chamfer lower=better, coverage/jaccard higher=better):")
    print(f"  {'rule':<11}{'chamfer':>10}{'coverage':>10}{'jaccard':>10}")
    for rule, s in results.items():
        print(f"  {rule:<11}{s['chamfer']:>10.4f}{s['coverage']:>10.3f}"
              f"{s['decode_jaccard']:>10.3f}")
        for key in ("chamfer", "coverage", "decode_jaccard"):
            assert np.isfinite(s[key]), f"{rule}.{key} not finite"

    print("\n[PASS] well_centre_analysis smoke test")


if __name__ == "__main__":
    _smoke()
