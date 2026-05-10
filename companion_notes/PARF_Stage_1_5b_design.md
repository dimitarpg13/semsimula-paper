# PARF Stage-1.5b: Gathered Top-k V_φ Design

**Status**: Pending implementation. Pick up after `paper_tmlr_1` submission.

**Owner**: tbd

**Estimated effort**: 2–3 days (core refactor + smoke validation + small-scale re-validation).

**Dependencies**:
- Current sparse PARF: `notebooks/conservative_arch/parf/model_parf_sparse.py` (Stage-1.5a, dense eval).
- Reference design: `parf/On_Gumbel_softmax_sparsity_applied_to_V_phi.md` §4.3 (gathered form).
- Pilot results: `notebooks/conservative_arch/scaleup/results/parf_structural_vphi16_sparse_k4_*` (the H=16 / grad-accum=2 pilot run that motivated this work).

---

## 1. Motivation

The current sparse PARF (referred to as *Stage-1.5a* in the code) computes the pair potential **densely** at all O(T²) positions and then multiplies by a sparse (k-of-T) Gumbel mask before aggregation:

```python
# notebooks/conservative_arch/parf/model_parf_sparse.py:386
P = self.V_phi(h_in, h_src)                  # (B, T, T)  — dense eval
P_masked = (P * tilde_m).masked_fill(~causal, 0.0)
U = V_th_per_token.sum() + P_masked.sum()
```

The mask `tilde_m` has only k=4 nonzeros per query row, so most entries of `P` contribute nothing to `U` and nothing to its gradient. **All the dense V_φ computation is wasted**, including the (B, T, T, H) intermediate tensors inside V_φ that dominate the activation memory at scaleup.

At our scaleup config (B=16, T=512, V_φ widths H=16) each dense (B, T, T, H) intermediate is 256 MiB; V_φ retains **four** such intermediates per layer (the φ-branch pre/post-activation and the θ-branch pre/post-activation, all kept live for the second-order graph that `autograd.grad(create_graph=True)` builds). That is ~1 GiB of V_φ activation memory per layer, ~8 GiB across 8 layers — and is what forces grad-accum=2 to fit on a 40 GiB A100 (see `paper_tmlr_1` Arm 5 commit `519f4b5`) and also OOMs Arm 5b on H100 96 GB at H=128 unless grad-accum is also enabled there (commit `840a824`). With Stage-1.5b's gathered V_φ, each intermediate is (B, T, k, H) = (16, 512, 4, 16) × 4 bytes = **2 MiB per intermediate, ~8 MiB per layer** — a **128× reduction** at k=4, T=512, preserved per-intermediate and across the per-layer working set.

### Memory savings table

At our scaleup config (B=16, T=512, L=8, k=4); accounting for the 4 retained `(B, T, T, H)` intermediates per V_φ layer that `autograd.grad(create_graph=True)` keeps live:

| Tensor | Stage-1.5a (current) | Stage-1.5b (gathered) | Savings |
|---|---|---|---|
| `(B, T, T, H)` per intermediate | 256 MiB | **2 MiB** | 128× |
| Per-layer V_φ working set (×4 intermediates) | ~1024 MiB | **~8 MiB** | 128× |
| 8 layers V_φ working set | ~8 GiB | **~64 MiB** | 128× |

**Implication**: Stage-1.5b would eliminate the V_φ memory pressure entirely. We could go back to H=128 (full V_φ capacity, ~19 K params) and still have ample memory for the rest of the graph. We could also drop `--grad-accum` and run at the full B=16 in a single pass on **both** A100 40 GB **and** H100 80/96 GB (~1.7× wall-clock speed-up on A100; an even larger speed-up on H100, where the current Arm 5b cell is grad-accum-bottlenecked rather than compute-bottlenecked).

### Compute savings

| Operation | Stage-1.5a | Stage-1.5b | Savings |
|---|---|---|---|
| V_φ matmuls per layer | O(B·T²·H) | O(B·T·k·H) | T/k = 128× |
| V_φ wall-clock (back-of-envelope) | ~5–10 s/step | ~0.05 s/step | ~100× |

**Implication**: PARF would become competitive with attention in compute too. The dominant cost would shift to the score-head (still O(T²)) and V_θ (per-token, no T² scaling).

---

## 2. Math: gradient-equivalence proof

Let `m_hard ∈ {0, 1}^{B×T×T}` be the hard top-k mask (k nonzeros per row), `y = softmax(z) ∈ ℝ^{B×T×T}` be the soft mask used by the straight-through estimator, and `~m = stop_grad(m_hard - k·y) + k·y` be the composite STE mask. Define:

```
P[b, t, s] = V_φ(h_in[b, t], h_src[b, s])    # (B, T, T)
U_pair = Σ_{b,t,s} P[b, t, s] · ~m[b, t, s]   # scalar
```

**Claim**: For any choice of indices `{idx[b, t, j]}_{j=0..k-1}` such that `m_hard[b, t, idx[b, t, j]] = 1` and `m_hard[b, t, s] = 0` for `s ∉ idx[b, t, :]`, the following are mathematically equivalent:

### Stage-1.5a (current dense form)

```
U_pair^(a) = Σ_{b,t,s} V_φ(h_in[b, t], h_src[b, s]) · ~m[b, t, s]
```

### Stage-1.5b (gathered form)

```
~m_g[b, t, j] = ~m[b, t, idx[b, t, j]]                    # (B, T, k)  — gathered STE mask
h_src_g[b, t, j] = h_src[b, idx[b, t, j]]                 # (B, T, k, d)
V_φ_g[b, t, j] = V_φ(h_in[b, t], h_src_g[b, t, j])        # (B, T, k)

U_pair^(b) = Σ_{b,t,j} V_φ_g[b, t, j] · ~m_g[b, t, j]
           + Σ_{b,t,s ∉ idx[b,t,:]} 0 · ~m[b, t, s]       # ← these terms are 0 by construction of m_hard
```

### Proof sketch

The forward equality follows directly from the construction of `m_hard`:

```
U_pair^(a) = Σ_{b,t,s} V_φ(h_in[b,t], h_src[b,s]) · ~m[b,t,s]
           = Σ_{b,t} [ Σ_{s ∈ idx[b,t,:]} V_φ(h_in[b,t], h_src[b,s]) · ~m[b,t,s]
                    + Σ_{s ∉ idx[b,t,:]} V_φ(h_in[b,t], h_src[b,s]) · ~m[b,t,s] ]
           = Σ_{b,t,j} V_φ(h_in[b,t], h_src[b,idx[b,t,j]]) · ~m[b,t,idx[b,t,j]]
                    + Σ_{b,t} Σ_{s ∉ idx[b,t,:]} V_φ(...) · ~m[b,t,s]                  (*)
```

The second sum (*) is **non-zero in Stage-1.5a only because of the soft-mask STE branch** — recall `~m = m_hard - k·y).detach() + k·y`. The `k·y` part has support over all causal s, not just the top-k. So in the dense form, those off-top-k entries DO contribute to `U_pair^(a)` via the `k·y` branch.

But here's the key observation: **at forward time, the value of `~m` at any non-top-k position is exactly zero**, by the STE construction:

```
~m[b, t, s] = (m_hard[b, t, s] - k·y[b, t, s]).detach() + k·y[b, t, s]
            = m_hard[b, t, s] - k·y[b, t, s] + k·y[b, t, s]   (in forward eval)
            = m_hard[b, t, s]
```

So in **forward** the (*) sum is zero. ✓

In **backward**, the STE construction makes `∂~m/∂z = k·∂y/∂z` everywhere (including non-top-k positions). The gradient of `U_pair` w.r.t. the score logits `z[b, t, s]` for `s ∉ idx[b, t, :]` flows through the `(P · k·y)` term:

```
∂U_pair^(a)/∂z[b, t, s] = Σ_{b',t',s'} V_φ(h_in[b',t'], h_src[b',s']) · k · ∂y[b',t',s']/∂z[b,t,s]
```

For `(b', t') ≠ (b, t)`, `∂y/∂z = 0` (softmax is row-wise). For `(b', t') = (b, t)`, the softmax derivative is non-zero for ALL s' in the row. So the gradient on `z[b, t, s]` (s ∉ top-k) depends on V_φ values at top-k positions only:

```
∂U_pair/∂z[b, t, s] = k · Σ_{s' ∈ row} V_φ(h_in[b,t], h_src[b,s']) · ∂y[b,t,s']/∂z[b,t,s]
                    = k · Σ_{s' ∈ row} V_φ(h_in[b,t], h_src[b,s']) · y[b,t,s'] · (δ_{s,s'} - y[b,t,s])
```

**Key step**: at non-top-k positions s, `y[b,t,s]` is small (because z[b,t,s] is unselected by top-k → it has low logit → low softmax probability). At equilibrium training, y is concentrated on the top-k positions, so the V_φ values at non-top-k positions enter the gradient only weighted by `y[b,t,s'] · y[b,t,s]` which is doubly-suppressed. **The gathered form computes V_φ only at top-k positions, which captures the dominant gradient terms.**

**Strict equivalence in the limit**: as the Gumbel temperature τ → 0, `y` becomes a hard one-hot over the top-k. Then `y[b,t,s] = 0` for s ∉ top-k, so the off-top-k gradient terms become exactly zero. Stage-1.5a and Stage-1.5b produce **bit-identical gradients** in this limit.

**Approximate equivalence at τ > 0**: at finite τ, Stage-1.5b drops the off-top-k contributions to the score-head gradient. This is a **finite-temperature approximation** that becomes exact as τ → 0 and is the standard trade-off in straight-through Gumbel-softmax routing (see Jang et al. 2017 §3.4). Empirically this approximation is acceptable when the routing has converged (i.e. y is concentrated), which is exactly when Stage-1.5b is most beneficial.

### Implication for V_φ parameter gradients

The V_φ parameter gradients flow through `V_φ(h_in[b,t], h_src[b,s])` only for s ∈ top-k. So even in the dense Stage-1.5a form, V_φ parameters only receive gradients from the k positions per query. **V_φ parameter gradients are bit-identical between Stage-1.5a and Stage-1.5b.** ✓

### Implication for h_in / V_θ gradients

`h_in` enters V_φ via `h_in[b,t]` (broadcast across all s in the row). The gradient back through `h_in[b,t]` is the sum over s of `∂V_φ/∂h_in[b,t] · ~m[b,t,s]`. By the same argument as before, this sum is concentrated on the top-k positions in forward (because `~m` is zero off-top-k). **`h_in` gradients are bit-identical between Stage-1.5a and Stage-1.5b.** ✓

### Summary

| Gradient route | Stage-1.5a vs Stage-1.5b |
|---|---|
| V_φ parameter gradients | **Bit-identical** |
| `h_in` gradients (force on query) | **Bit-identical** |
| Score-head gradients via top-k positions | **Bit-identical** |
| Score-head gradients via non-top-k positions | **Approximate** (exact at τ → 0, near-zero at concentrated y) |

This is a clean approximation that matches the standard straight-through Gumbel-softmax usage in the literature. The `paper_tmlr_1` discussion section can frame Stage-1.5b as "the natural top-k gathered form, equivalent to Stage-1.5a in the limit τ → 0".

---

## 3. API changes

### 3.1 `ScoreHead` (no API change, internal change only)

Currently returns `pi: (B, T, T)`. Stage-1.5b additionally needs the top-k indices, but those can be computed from `pi` separately by `_sparse_mask_indices` (see below). No change to `ScoreHead`.

### 3.2 New `_sparse_topk_indices` method on `SparsePARFLM`

Replaces the dense-mask construction in `_sparse_mask` with index-based output:

```python
def _sparse_topk_indices(
    self,
    pi: torch.Tensor,                # (B, T, T) — score logits
    causal: torch.Tensor,            # (B, T, T) — strict-causal bool mask
    T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k Gumbel-softmax routing in gathered form.

    Returns:
      idx     : (B, T, k_eff) int64 — index of each top-k source per query
      m_g     : (B, T, k_eff) float — straight-through STE composite mask
                                      gathered at the top-k positions

    Construction parallels `_sparse_mask`: idx is the top-k of the
    Gumbel-perturbed scores; m_g is built from the corresponding gathered
    entries of m_hard and softmax(z).  Rows with fewer than k_eff valid
    causal sources (only row t=0) get padded indices and zero mask
    weight (no aggregation will pick them up).

    Strict-causal enforcement: idx values are clamped to the causal set
    via masked top-k; m_g entries for non-causal indices are zeroed.
    """
    cfg = self.cfg
    k_eff = max(1, min(cfg.top_k, T - 1))
    tau = float(self._gumbel_tau)

    # Gumbel + temperature scaling (same as Stage-1.5a)
    if self.training and cfg.gumbel_noise:
        u = torch.rand_like(pi).clamp_min_(1e-9)
        g = -torch.log(-torch.log(u))
        z_unmasked = (pi + g) / tau
    else:
        z_unmasked = pi / tau

    # Top-k indices over causal sources
    z_topk = z_unmasked.masked_fill(~causal, float("-inf"))
    _, idx = z_topk.topk(k_eff, dim=-1)                # (B, T, k_eff)

    # Gather m_hard and y at idx
    # m_hard at idx is always 1.0 by construction; we still gather to
    # handle row-t=0 padding consistently.
    m_hard_g = torch.ones_like(idx, dtype=pi.dtype)    # (B, T, k_eff)

    # Soft mask: gather from the (B, T, T) softmax
    z_soft = z_unmasked.masked_fill(~causal, -1e9)
    y = torch.softmax(z_soft, dim=-1)                  # (B, T, T)
    y_g = y.gather(-1, idx)                            # (B, T, k_eff)

    # Mask invalid (non-causal) gathered indices to zero weight
    causal_g = causal.gather(-1, idx)                  # (B, T, k_eff) bool
    m_hard_g = m_hard_g * causal_g.to(m_hard_g.dtype)
    y_g = y_g * causal_g.to(y_g.dtype)

    # Composite STE: gathered analog of (m_hard - k*y).detach() + k*y
    kf = float(k_eff)
    m_g = (m_hard_g - kf * y_g).detach() + kf * y_g    # (B, T, k_eff)

    return idx, m_g
```

### 3.3 New `StructuralVPhi.forward_gathered` method

Adds a parallel forward path that takes pre-gathered source vectors:

```python
def forward_gathered(
    self,
    h: torch.Tensor,           # (B, T, d) — query side
    h_src_g: torch.Tensor,     # (B, T, k, d) — gathered source vectors
) -> torch.Tensor:             # (B, T, k) — V_φ at the gathered pairs
    """Gathered-eval form of V_φ.  Mathematically identical to

        forward(h, h_src).gather(-1, idx)

    where idx is the (B, T, k) gather index used to build h_src_g, but
    materialises only (B, T, k, H) intermediates instead of (B, T, T, H).

    This is the Stage-1.5b optimisation; see
    PARF_Stage_1_5b_design.md for the equivalence proof and
    memory math.
    """
    B, T, d = h.shape
    k = h_src_g.shape[2]

    # Type and angle projections (per-token, no T² broadcast)
    l_q = self.W_l(h)                          # (B, T, dl)
    l_s = self.W_l(h_src_g)                    # (B, T, k, dl)
    th_q = self.W_theta(h)                     # (B, T, K)
    th_s = self.W_theta(h_src_g)               # (B, T, k, K)

    # Pairwise type distance: gathered version
    # ||l_q[b,t] - l_s[b,t,j]||^2  for each (b, t, j)
    diff = l_q.unsqueeze(2) - l_s              # (B, T, k, dl)
    l_dist2 = (diff * diff).sum(dim=-1)        # (B, T, k)
    # NOTE: at small k the squared-norm trick saves less than the
    # explicit-diff form here (k=4 is comparable to dl=32), so we use
    # the simpler explicit form.

    c = F.softplus(self.phi_c_net(l_dist2.unsqueeze(-1)).squeeze(-1))
    Phi = torch.exp(-c * l_dist2)              # (B, T, k)

    # Angle Θ_φ — gathered broadcast
    proj_q = self.theta_w_q(th_q)              # (B, T, H)
    proj_s = self.theta_w_s(th_s)              # (B, T, k, H)
    proj_qd = self.theta_w_d(th_q)             # (B, T, H)
    proj_sd = self.theta_w_d(th_s)             # (B, T, k, H)

    proj_t = (proj_q + proj_qd + self.theta_b1).unsqueeze(2)  # (B, T, 1, H)
    proj_u = proj_s - proj_sd                                  # (B, T, k, H)
    hidden = proj_t + proj_u                                   # (B, T, k, H)
    hidden = F.gelu(hidden)
    Theta = torch.tanh(self.theta_w2(hidden).squeeze(-1))      # (B, T, k)

    # Distance kernel (gathered)
    h_diff = h.unsqueeze(2) - h_src_g          # (B, T, k, d)
    h_dist2 = (h_diff * h_diff).sum(dim=-1)    # (B, T, k)
    r = torch.sqrt(h_dist2 + self.eps2)        # (B, T, k)

    return -self.C * Theta * Phi / r           # (B, T, k)
```

### 3.4 `_layer_step` change

Replace the dense-eval path with gathered-eval when `cfg.use_gathered_v_phi=True`:

```python
def _layer_step(self, h, h_prev, m_b, gamma, dt):
    cfg = self.cfg
    B, T, d = h.shape
    delta = h - h_prev

    xi_input = h.detach() if cfg.causal_force else h
    xi_now = causal_cumulative_mean(xi_input)

    h_in = h
    if not h_in.requires_grad:
        h_in = h_in.requires_grad_(True)

    h_src = h_in.detach() if cfg.causal_force else h_in
    h_src_for_score = h_in.detach() if cfg.score_head_use_detached_h_src else h_in

    V_th_per_token = self.V_theta(xi_now, h_in)              # (B, T, 1)

    pi = self.score_head(h_in, h_src_for_score)              # (B, T, T)
    causal = self._pair_mask_for(T, h_in.device)             # (B, T, T) bool

    if cfg.use_gathered_v_phi:
        # Stage-1.5b: gathered top-k V_phi
        idx, m_g = self._sparse_topk_indices(pi, causal, T)  # (B, T, k), (B, T, k)
        # Gather source vectors at top-k indices
        idx_for_gather = idx.unsqueeze(-1).expand(-1, -1, -1, d)  # (B, T, k, d)
        h_src_expanded = h_src.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, T, d)
        h_src_g = h_src_expanded.gather(2, idx_for_gather)    # (B, T, k, d)
        V_phi_g = self.V_phi.forward_gathered(h_in, h_src_g)  # (B, T, k)
        U_pair = (V_phi_g * m_g).sum()
    else:
        # Stage-1.5a: dense eval, sparse aggregation (legacy path)
        tilde_m = self._sparse_mask(pi, causal, T)
        if cfg.use_grad_checkpoint and self.training:
            P = torch.utils.checkpoint.checkpoint(
                self.V_phi, h_in, h_src, use_reentrant=False,
            )
        else:
            P = self.V_phi(h_in, h_src)
        P_masked = (P * tilde_m).masked_fill(~causal, 0.0)
        U_pair = P_masked.sum()

    U = V_th_per_token.sum() + U_pair

    grad_U, = torch.autograd.grad(
        U, h_in,
        create_graph=self.training,
        retain_graph=True,
    )
    f = -grad_U

    denom = 1.0 + dt * gamma
    h_new = h_in + delta / denom + (dt * dt / (m_b * denom)) * f

    if cfg.ln_after_step:
        h_new = self._project(h_new)
    return h_new
```

### 3.5 `SparsePARFConfig` change

Add a single boolean flag, default False to preserve current behaviour:

```python
@dataclass
class SparsePARFConfig(PARFConfig):
    # ... existing fields ...
    use_gathered_v_phi: bool = False
    """If True, route through Stage-1.5b gathered V_phi
    (`forward_gathered`).  Memory drops O(T/k) per layer; gradients
    are equivalent to Stage-1.5a in the limit gumbel_tau -> 0 and
    near-equivalent at finite tau when routing has converged.
    See PARF_Stage_1_5b_design.md.
    """
```

### 3.6 Trainer CLI

Add `--use-gathered-v-phi` to `train_parf_scaleup.py` and `train_parf.py`:

```python
ap.add_argument("--use-gathered-v-phi", action="store_true",
                dest="use_gathered_v_phi",
                help="Route V_phi through the Stage-1.5b gathered path "
                     "(O(T*k) memory and compute instead of O(T^2)).")
```

Wire to `build_config(... use_gathered_v_phi=args.use_gathered_v_phi ...)` and into `SparsePARFConfig(... use_gathered_v_phi=use_gathered_v_phi ...)`.

---

## 4. Validation plan

### 4.1 Smoke test: bit-identity at τ → 0

In `notebooks/conservative_arch/parf/smoke_test_sparse.py`, add a new test:

```python
def test_stage_1_5_a_b_equivalence_at_low_tau():
    """Stage-1.5a and Stage-1.5b produce bit-identical loss + parameter
    gradients when gumbel_tau is small (hard one-hot routing)."""
    cfg_a = SparsePARFConfig(
        # ... small config ...
        use_gathered_v_phi=False,
        gumbel_noise=False,    # deterministic top-k for bit-identity
        gumbel_tau_init=0.01,  # near-zero tau -> y is hard one-hot
    )
    cfg_b = replace(cfg_a, use_gathered_v_phi=True)

    torch.manual_seed(0)
    model_a = SparsePARFLM(cfg_a)
    torch.manual_seed(0)
    model_b = SparsePARFLM(cfg_b)

    x = torch.randint(0, cfg_a.vocab_size, (2, 16))
    y = torch.randint(0, cfg_a.vocab_size, (2, 16))

    _, loss_a = model_a(x, y)
    _, loss_b = model_b(x, y)
    assert torch.allclose(loss_a, loss_b, atol=1e-5), (loss_a, loss_b)

    loss_a.backward()
    loss_b.backward()
    for (n_a, p_a), (n_b, p_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert n_a == n_b
        if p_a.grad is None and p_b.grad is None:
            continue
        assert torch.allclose(p_a.grad, p_b.grad, atol=1e-5), \
            f"grad mismatch at {n_a}"
```

### 4.2 Approximate equivalence at training τ

Run the small-scale Shakespeare protocol (`notebooks/conservative_arch/parf/sweep_parf_sparse.sh`) for k ∈ {4, 8, 16, 32} with both `use_gathered_v_phi=False` and `use_gathered_v_phi=True`. Expect:
- Final PPL within 1–2 PPL across the two paths (within seed noise)
- Wall-clock for Stage-1.5b: 5–10× faster than Stage-1.5a at small scale (less dramatic than the predicted 100× because small-scale T is small; the savings grow with T)
- Same convergence trajectory (loss curves overlap modulo noise)

### 4.3 Memory verification

Add a `--report-mem` flag to the trainer that calls `torch.cuda.memory_summary()` after the first training step. Run Arm 5 with `use_gathered_v_phi=True` and verify:
- Per-layer V_φ peak ~2 MiB (was 256 MiB at H=16)
- Total working set ~5 GiB at B=16 with NO grad-accum (was ~38 GiB)
- Can re-enable H=128 (full V_φ capacity) and still fit

### 4.4 Causal-violation probe

The existing `causal_probe_parf.py` should pass with `use_gathered_v_phi=True` unchanged, because:
- The gather indices come from a causal-masked top-k
- The `causal_g.to(m_hard_g.dtype)` mask zeroes any gathered entries that hit non-causal positions (only row t=0)
- V_φ_g is computed only at causal pairs

Add an explicit assertion to the smoke test that the Stage-1.5b path passes the causal probe.

---

## 5. Implementation plan (2–3 days)

### Day 1: Core refactor

- [ ] Add `forward_gathered` method to `StructuralVPhi` (and `MLPVPhi`).
- [ ] Add `_sparse_topk_indices` method to `SparsePARFLM`.
- [ ] Add `use_gathered_v_phi` field to `SparsePARFConfig`.
- [ ] Branch `_layer_step` on `cfg.use_gathered_v_phi`.
- [ ] Smoke-run on CPU with the existing tiny config to confirm forward + backward execute without errors.

### Day 2: Validation

- [ ] Implement the bit-identity smoke test (§4.1) and confirm it passes at τ → 0.
- [ ] Implement the memory verification (§4.3) and confirm the per-layer peak drops as predicted.
- [ ] Run the small-scale Shakespeare sweep with both paths (§4.2). Compare loss curves and wall-clock.
- [ ] Confirm causal-violation probe passes for Stage-1.5b (§4.4).

### Day 3: Integration + documentation

- [ ] Wire `--use-gathered-v-phi` CLI flag into `train_parf.py` and `train_parf_scaleup.py`.
- [ ] Update Arm 5 cell in `colab_pilot.ipynb` to use Stage-1.5b: bump V_φ widths back to H=128, drop `--grad-accum`, expect wall-clock ~25 min instead of ~80 min.
- [ ] Re-run Arm 5 on Colab with the new config and confirm the result matches (or improves on) the H=16 / Stage-1.5a baseline.
- [ ] Update `paper_tmlr_1` discussion section to note Stage-1.5b as the standard form, with Stage-1.5a kept only as a reference implementation for the equivalence proof.
- [ ] Update `parf/On_Gumbel_softmax_sparsity_applied_to_V_phi.md` §4.3 to point at this design doc and the implementing commit.

### Stretch (Day 4 if needed)

- [ ] Larger-k sweep enabled by Stage-1.5b's memory savings: k ∈ {32, 64, 128} at scaleup. May reveal that PARF benefits from larger k now that the memory cost doesn't scale.
- [ ] Larger-T sweep: T ∈ {1024, 2048} with Stage-1.5b at k=4. Tests whether the long-context regime is where PARF shines.

---

## 6. Risks and open questions

### Risk 1: gather kernel performance

`torch.gather` along a non-contiguous axis can be slower than a strided view. We may need to:
- Materialise `h_src_expanded = h_src.unsqueeze(1).expand(-1, T, -1, -1)` lazily — note this is a view, not a copy, so memory cost is zero
- The actual gather creates `(B, T, k, d)` which IS a copy, ~512 KiB at our scaleup config — negligible
- Profile vs the dense path on small-T to confirm Stage-1.5b is faster end-to-end (it should be, but verify)

**Mitigation**: profile in §4.2; if the gather is a hot spot, consider `torch.index_select` per-row or a custom triton kernel.

### Risk 2: finite-τ approximation degrades training

The proof shows Stage-1.5b drops some score-head gradient terms at finite τ. If these terms are essential for routing convergence, Stage-1.5b would underperform Stage-1.5a in the early training phase (when y is still spread).

**Mitigation**: §4.2 sweep compares loss curves head-to-head. If early-training divergence is observed, options are:
- Start with a small τ (e.g. 0.5 instead of 1.0) so y is concentrated from step 1
- Hybrid: use Stage-1.5a for the first ~10% of training (warmup) and switch to Stage-1.5b after
- Increase k during the warmup, then anneal down

### Risk 3: `forward_gathered` and `forward` numerical equivalence

The gathered form uses an explicit `(a - b).norm()²` instead of the `||a||² + ||b||² - 2<a,b>` squared-norm trick. At small d, the difference is negligible; at large d, the explicit form is more numerically stable but slower. We're using small d_type (32) and d_angle (16), so this shouldn't matter — but verify in the bit-identity test (§4.1) at scaled-up dimensions.

**Mitigation**: if §4.1 fails by a small numerical margin at scaleup d, switch the gathered form to the squared-norm trick too. The gathered `||a-b||²` is just `a²[b,t,1,:] + b²[b,t,j,:] - 2 <a[b,t,1,:], b[b,t,j,:]>` — same trick, same memory, just at the gathered shape.

### Risk 4: parameter count drift

Stage-1.5b doesn't change V_φ parameters (same `phi_c_net`, `theta_w_q/s/d/2`, `W_l`, `W_theta`). But the score-head's `b1` parameter currently has shape `(score_head_hidden,)` and is added to the `(B, T, T, H)` broadcast. With Stage-1.5b we don't change the score-head — it still produces dense `(B, T, T)` logits — so this is a non-issue. Just noting for completeness.

### Empirical justification: Stage-1.5a NaN failure at k=32 (8 May 2026)

Stage-1.5b's value is not purely a memory and compute story — it also resolves a concrete training-time failure mode of Stage-1.5a observed during the small-scale P5 sparsity ladder.

**Failure observation.** On 8 May 2026 the P5 sweep over `k ∈ {4, 8, 16, 32}` ran on local MPS (Tiny Shakespeare, d=128, L=8, V_φ widths H=128, all other hyperparameters held constant). Cells k=4 / k=8 / k=16 completed cleanly. The k=32 cell **diverged to NaN at step 50** — train loss, grad norm, γ all NaN — and stayed NaN for ~5 hours of continued training before being killed manually. No checkpoint or summary was produced; only `notebooks/conservative_arch/parf/results/structural_sparse/seed0_k32/training.log` was preserved as forensic evidence. Full hyperparameter set and trace recorded in `PARF-SPLM_Path_Forward_and_Experiments.md` §4.8.

**Mechanism.** With the score-head initialised at `score_head_init_scale=0.02` and `gumbel_tau_init=1.0`, the soft-mask `y = softmax(z/τ)` at step 0 is approximately uniform over all causal sources. The straight-through composite mask `~m = stop_grad(m_hard - k·y) + k·y` then has its *backward* term `k·y` scaled by `k=32`, distributed across ~32 active source positions per query. This makes the effective backward gradient on every score-head logit ~32× larger than the equivalent k=4 cell. In Stage-1.5a (dense V_φ eval) this gradient amplification multiplies through the *entire* (B, T, T) pair-potential field, including all the off-top-k entries that would otherwise contribute zero in forward. The first ~few AdamW steps under the 200-step linear warm-up (step 50 sits at `lr=1.25e-4` toward the `lr=5e-4` peak) are enough to drive `phi_c_net.softplus(c)` into a regime where `Phi = exp(-c · l_dist2)` underflows or `r = √(h_dist2 + ε²)` produces NaN downstream.

**Why Stage-1.5b should not reproduce this.** In the gathered form, V_φ is evaluated only at the top-k indices, and the off-top-k positions contribute *zero V_φ to U* in both forward and backward. Concretely, the gradient on a non-top-k logit `z[b, t, s]` in Stage-1.5b is:

```
∂U_pair^(b)/∂z[b, t, s ∉ idx]   (Stage-1.5b)
  = k · Σ_{j ∈ idx} V_φ_g[b, t, j] · y_g[b, t, j] · ∂y_g[b, t, j]/∂z[b, t, s]
```

where the sum runs over the **k** gathered positions only. Compare to Stage-1.5a, where the analogous sum runs over **all T** positions:

```
∂U_pair^(a)/∂z[b, t, s ∉ idx]   (Stage-1.5a)
  = k · Σ_{s' ∈ row} V_φ[b, t, s'] · y[b, t, s'] · ∂y[b, t, s']/∂z[b, t, s]
                  (T positions, including non-top-k whose V_φ is freshly evaluated)
```

Stage-1.5a's sum is dominated at training start by the (T − k) "spurious" V_φ values at non-top-k positions, which are arbitrary outputs of an uninitialised V_φ broadcast to every pair. These spurious values are exactly what the k·y amplification multiplies through. Stage-1.5b drops these terms by construction.

**Predicted Stage-1.5b behaviour at k=32.** Should train cleanly under the same hyperparameters (`gumbel_tau_init=1.0`, `score_head_init_scale=0.02`, `init_gamma=0.15`, lr=5e-4 cosine, etc.). If Stage-1.5b *also* NaNs at k=32, the failure is in a different mechanism (probably the score-head's own gradient route, which is independent of V_φ) and we should chase that separately. The §4.2 validation sweep should include a k=32 cell explicitly to test this prediction.

**Consequence for the implementation plan.** Day 2 validation (§4.2) gains an additional cell:

- [ ] **k=32 stability test**: run Stage-1.5b at the *exact* hyperparameter set that NaN'd Stage-1.5a (above) and confirm clean convergence. If Stage-1.5b reaches a finite val PPL at k=32, this is concrete evidence that Stage-1.5b is not just an efficiency optimisation but a *stability fix* for the large-k regime.

**Mitigations available within Stage-1.5a (not pursued).** For completeness, three knobs reduce the k=32 NaN risk in Stage-1.5a without switching to Stage-1.5b:
1. `--gumbel-tau-init 0.5` (or smaller) — concentrate y from step 1
2. Anneal k upward (e.g. start at k=4, ramp to k=32 over 25% of training)
3. `--score-head-init-scale 0.005` (4× smaller) — flatter initial logits

These are documented in `PARF-SPLM_Path_Forward_and_Experiments.md` §4.8 as "deferred — Stage-1.5b is the cleaner fix".

### Open question: sparse autograd for the score head

If we want to drop the score head's O(T²) cost too, we'd need a sparse score head (only compute logits at the candidate top-k positions). This is harder because we don't know the top-k until we compute the logits. Possible approaches:
- LSH-based candidate selection (out of scope for this design)
- Random feature approximation
- Block-sparse score head with learned routing

This is outside the scope of Stage-1.5b. Leave as future work; Stage-1.5b alone is a 100× win on V_φ which is the dominant cost.

---

## 7. Paper implications

Once Stage-1.5b lands, the `paper_tmlr_1` PARF arm story changes meaningfully:

- **Pilot table footnote**: remove "(H=16, grad-accum=2 due to memory)" caveat. Run at H=128, B=16 single-pass.
- **Discussion §3.2 (or wherever PARF lives)**: lead with "Stage-1.5b makes PARF competitive with attention in compute and memory". This is the architectural claim that justifies the existence of PARF beyond a curiosity.
- **Table comparison**: include "wall-clock per step" and "peak memory" columns. PARF Stage-1.5b should be within 2× of attention on both, while delivering the structured pair-interaction inductive bias.
- **Companion paper `paper_v4` §17 (PARF)**: note Stage-1.5b as the production form, with the Stage-1.5a derivation kept as the design-time equivalence-proof scaffolding.

---

## 8. References

- `notebooks/conservative_arch/parf/model_parf.py` — dense PARF (Stage-1) and structural V_φ
- `notebooks/conservative_arch/parf/model_parf_sparse.py` — current Stage-1.5a implementation
- `parf/On_Gumbel_softmax_sparsity_applied_to_V_phi.md` §4.3 — original Stage-1.5b sketch
- Jang, Gu, Poole 2017, "Categorical Reparameterization with Gumbel-Softmax", §3.4 (straight-through estimator) — foundation of the τ → 0 equivalence argument
- Maddison, Mnih, Teh 2017, "The Concrete Distribution" — companion result on continuous relaxations of categorical distributions
