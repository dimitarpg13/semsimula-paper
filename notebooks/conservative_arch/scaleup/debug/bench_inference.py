# Paste into a Colab cell. Measures INFERENCE (forward-only) cost for the
# Fock arm vs a matched GPT-2, and quantifies the xi-channel dense-matmul
# overhead.  ~2-4 min.  No training, no checkpoint.
#
# WHERE TO PASTE
#   Part B (full Fock forward) needs the notebook's already-built `model`,
#   so paste this AFTER Cell 5 of any Fock notebook (the cell that prints
#   "Model: FockMultiXiPARFLM v2.1 + Anisotropic Gaussian V_theta").
#   Reusing that object rather than rebuilding it is deliberate: the config
#   has ~50 fields and any drift would silently benchmark a different model.
#   Parts A, C and D are self-contained and run anywhere with a GPU --
#   including the GPT-2 baseline notebook, where Part B will simply skip.
#
# READ THE RATIOS, NOT THE ABSOLUTE MILLISECONDS.
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), 'needs a GPU runtime'
DEV = 'cuda'
print(f'GPU: {torch.cuda.get_device_name(0)}   torch {torch.__version__}')
print()


def bench(fn, n=5, warmup=2):
    """Median-of-n wall time in ms, with OOM reported rather than raised."""
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        return sorted(ts)[len(ts) // 2]
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            return float('nan')
        raise


def slope(xs, ys):
    """Least-squares log-log slope: ys ~ xs**slope."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys)
           if y == y and y > 0]
    if len(pts) < 2:
        return float('nan')
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return num / den if den else float('nan')


# ===================================================================
# PART A -- xi channels: dense (T,T) matmul vs the exact recurrence
# ===================================================================
# model_multixi.causal_ema_weights builds W[t,s] = alpha^(t-s) / Z_t as a
# dense (T,T) matrix and does W @ h, K times per layer.  That is a
# normalised EMA, i.e. a first-order linear recurrence:
#     u_t = alpha * u_{t-1} + h_t          (u_{-1} = 0)
#     Z_t = (1 - alpha^(t+1)) / (1 - alpha)
#     xi_t = u_t / Z_t
# so the same values are reachable in O(d) per token instead of O(T*d).
# This part checks the two agree numerically, then times both.
print('=' * 72)
print('PART A -- xi channels: dense (T,T) matmul vs exact recurrence')
print('=' * 72)

_ALPHA_EPS = 1e-6


def ema_weights_dense(T, alpha, dtype, device):
    """Local copy of model_multixi.causal_ema_weights (log-space form)."""
    a = alpha.clamp(min=_ALPHA_EPS, max=1.0 - _ALPHA_EPS)
    s = torch.arange(T, dtype=dtype, device=device)
    diffs = s.view(T, 1) - s.view(1, T)
    causal = diffs >= 0
    logW = torch.log(a) * diffs.clamp(min=0.0)
    logW = logW.masked_fill(~causal, float('-inf'))
    return torch.exp(logW - torch.logsumexp(logW, dim=1, keepdim=True))


def xi_dense(h, alphas):
    """What the model does today: K dense (T,T) matmuls."""
    B, T, d = h.shape
    return torch.stack(
        [ema_weights_dense(T, a, h.dtype, h.device).unsqueeze(0) @ h
         for a in alphas], dim=2)


def xi_recurrent(h, alphas):
    """Exact same values via the first-order recurrence, O(d) per token."""
    B, T, d = h.shape
    out = []
    for a in alphas:
        av = a.clamp(min=_ALPHA_EPS, max=1.0 - _ALPHA_EPS)
        u = torch.zeros(B, d, dtype=h.dtype, device=h.device)
        xs = []
        for t in range(T):
            u = av * u + h[:, t]
            Z = (1.0 - av ** (t + 1)) / (1.0 - av)
            xs.append(u / Z)
        out.append(torch.stack(xs, dim=1))
    return torch.stack(out, dim=2)


# The live run's learned alphas (from the step-32,500 log).
ALPHAS = torch.tensor([0.278, 0.602, 0.809, 0.964, 0.998], device=DEV)
D_REF = 384

# --- equivalence check (float64, small T so the dense form is exact) ---
_h = torch.randn(2, 64, D_REF, device=DEV, dtype=torch.float64)
_a = ALPHAS.double()
_err = (xi_dense(_h, _a) - xi_recurrent(_h, _a)).abs().max().item()
print(f'equivalence check (B=2, T=64, float64): max abs diff = {_err:.2e}')
assert _err < 1e-9, (
    f'recurrence does NOT reproduce the dense EMA (max diff {_err:.2e}). '
    'Do not trust the timings below -- the two forms differ.')

# Cross-check against the REAL module if it is importable, so this cell
# cannot drift away from what the model actually runs.
try:
    from model_multixi import causal_ema_weights as _real_w
    _rd = (_real_w(64, _a[3], torch.float64, torch.device(DEV))
           - ema_weights_dense(64, _a[3], torch.float64, torch.device(DEV)))
    print(f'cross-check vs model_multixi.causal_ema_weights: '
          f'max abs diff = {_rd.abs().max().item():.2e}')
    assert _rd.abs().max().item() < 1e-12, 'local copy has drifted'
except ImportError:
    print('cross-check vs model_multixi: SKIPPED (not on sys.path)')
print()

print(f'{"T":>6}{"dense ms":>12}{"recurrent ms":>15}{"speedup":>10}'
      f'{"dense (T,T) mem":>18}')
Ts = [128, 256, 512, 1024, 2048]
dense_ms, rec_ms = [], []
for T in Ts:
    h = torch.randn(4, T, D_REF, device=DEV)
    dm = bench(lambda: xi_dense(h, ALPHAS), n=3)
    rm = bench(lambda: xi_recurrent(h, ALPHAS), n=2, warmup=1)
    dense_ms.append(dm)
    rec_ms.append(rm)
    mem = len(ALPHAS) * T * T * 4 / 2**20
    print(f'{T:>6}{dm:>12.2f}{rm:>15.2f}{dm / rm:>9.2f}x{mem:>15.1f} MiB')
    del h
    torch.cuda.empty_cache()

print(f'\n  dense scaling in T:      O(T^{slope(Ts, dense_ms):.2f})   '
      f'(expect ~2: the (T,T) matmul)')
print(f'  recurrent scaling in T: O(T^{slope(Ts, rec_ms):.2f})   '
      f'(expect ~1: one launch per token)')
print('\n  NOTE: this sequential loop is launch-bound, so for a FULL-SEQUENCE')
print('  forward it may well LOSE to the dense matmul on a GPU. That is a')
print('  property of the naive loop, not of the recurrence -- a chunked or')
print('  associative-scan form gets O(T*d) FLOPs at matmul throughput.')
print('  The decisive case is single-token DECODE, measured next.')

# --- decode step: the case that actually matters for inference ---
print(f'\n{"decode step (1 new token at context T)":<46}'
      f'{"dense":>12}{"recurrent":>12}{"speedup":>10}')
for T in [128, 512, 1024, 2048]:
    h = torch.randn(1, T, D_REF, device=DEV)
    u = torch.randn(1, len(ALPHAS), D_REF, device=DEV)
    h_new = torch.randn(1, D_REF, device=DEV)

    def dense_step():
        # No incremental path exists, so a new token means redoing the
        # whole (T,T) construction and matmul.
        return xi_dense(h, ALPHAS)

    def rec_step():
        # Carry u forward: one multiply-add per channel.
        Z = (1.0 - ALPHAS ** (T + 1)) / (1.0 - ALPHAS)
        return (ALPHAS.view(1, -1, 1) * u + h_new.unsqueeze(1)) / Z.view(1, -1, 1)

    dm = bench(dense_step, n=3)
    rm = bench(rec_step, n=20, warmup=5)
    print(f'{"  T = " + str(T):<46}{dm:>10.2f} ms{rm:>10.3f} ms'
          f'{dm / rm:>9.0f}x')
    del h, u, h_new
    torch.cuda.empty_cache()
print()


# ===================================================================
# PART A2 -- the ScoreHead: the term the paper's cost table omits
# ===================================================================
# paper_v5 sections/17f table tab:cm-comparison lists the deployed
# "Baseline: sparse V_phi (top-k)" at O(Tkd) compute / O(Tk) memory.
# That is the cost of evaluating V_phi on the k SELECTED pairs. Choosing
# which k requires scoring all pairs first, and ScoreHead.forward
# (model_parf_sparse.py) materialises
#     hidden = proj_t.unsqueeze(2) + proj_u.unsqueeze(1)   # (B, T, T, H)
#     return w2(gelu(hidden)).squeeze(-1)                  # (B, T, T)
# The GELU sits between the two token indices, so this does NOT factor
# into q(h_t)^T k(h_s) the way attention logits do -- there is no linear
# -attention kernel trick available for it. This part measures it alone.
print('=' * 72)
print("PART A2 -- ScoreHead (the O(T^2) routing term)")
print('=' * 72)

H_SCORE = 32          # cfg.score_head_hidden in the live run
w_q = nn.Linear(D_REF, H_SCORE, bias=False).to(DEV)
w_s = nn.Linear(D_REF, H_SCORE, bias=False).to(DEV)
w_d = nn.Linear(D_REF, H_SCORE, bias=False).to(DEV)
b1 = torch.zeros(H_SCORE, device=DEV)
w2 = nn.Linear(H_SCORE, 1).to(DEV)


def score_head(h_q, h_s):
    """Replica of model_parf_sparse.ScoreHead.forward."""
    proj_t = w_q(h_q) + w_d(h_q) + b1
    proj_u = w_s(h_s) - w_d(h_s)
    hidden = proj_t.unsqueeze(2) + proj_u.unsqueeze(1)    # (B, T, T, H)
    return w2(F.gelu(hidden)).squeeze(-1)                 # (B, T, T)


print(f'{"T":>6}{"score ms":>11}{"(B,T,T,H) MiB":>16}{"V_phi@k=16 ms":>16}'
      f'{"score/V_phi":>13}')
sh_ms = []
Ts_sh = [128, 256, 512, 1024]
for T in Ts_sh:
    h = torch.randn(4, T, D_REF, device=DEV)
    with torch.no_grad():
        ms = bench(lambda: score_head(h, h), n=3, warmup=1)
        # the term the paper DOES count: V_phi on k gathered pairs
        g = torch.randn(4, T, 16, D_REF, device=DEV)
        vms = bench(lambda: (g * h.unsqueeze(2)).sum(-1), n=3, warmup=1)
    sh_ms.append(ms)
    mib = 4 * T * T * H_SCORE * 4 / 2**20
    print(f'{T:>6}{ms:>11.2f}{mib:>16.1f}{vms:>16.3f}'
          f'{ms / vms:>12.0f}x')
    del h, g
    torch.cuda.empty_cache()
print(f'\n  ScoreHead scaling in T: O(T^{slope(Ts_sh, sh_ms):.2f})   (expect ~2)')
print('  The (B,T,T,H) intermediate is the memory story: at B=16, T=512,')
print('  H=32, fp32 that is 537 MiB per layer per forward.')
print()


# ===================================================================
# PART C -- matched GPT-2 reference forward (built inline)
# ===================================================================
print('=' * 72)
print('PART C -- GPT-2 reference forward (d=384, L=8, h=6, tied)')
print('=' * 72)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (t.view(B, T, self.h, d // self.h).transpose(1, 2)
                   for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT2Ref(nn.Module):
    def __init__(self, V=50257, d=384, L=8, h=6, ctx=2048):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(V, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.wte.weight          # tied

    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


gpt2 = GPT2Ref().to(DEV).eval()
n_gpt2 = sum(p.numel() for p in gpt2.parameters())
print(f'params: {n_gpt2:,}  (uses flash SDPA -- what a real deployment runs)')

gpt2_ms = {}
print(f'\n{"T":>6}{"B=1 ms":>11}{"us/token":>11}{"B=4 ms":>11}{"us/token":>11}')
for T in Ts:
    row = [T]
    for B in (1, 4):
        x = torch.randint(0, 50257, (B, T), device=DEV)
        with torch.no_grad():
            ms = bench(lambda: gpt2(x), n=5)
        gpt2_ms[(B, T)] = ms
        row += [ms, ms * 1000 / (B * T)]
        del x
    print(f'{row[0]:>6}{row[1]:>11.2f}{row[2]:>11.1f}'
          f'{row[3]:>11.2f}{row[4]:>11.1f}')
    torch.cuda.empty_cache()
print(f'\n  scaling in T (B=1): O(T^{slope(Ts, [gpt2_ms[(1, t)] for t in Ts]):.2f})'
      f'  -- sublinear at these sizes because the 19.3M-param lm_head dominates')
print()


# ===================================================================
# PART B -- Fock full forward (needs the notebook's `model`)
# ===================================================================
print('=' * 72)
print('PART B -- Fock forward')
print('=' * 72)

fock_ms = {}
_m = globals().get('model', None)
_cfg = globals().get('model_cfg', None)

if _m is None or _cfg is None:
    print('SKIPPED: no `model` / `model_cfg` in globals.')
    print('To run this part, paste this cell into a Fock notebook AFTER')
    print('Cell 5 (the one that prints "Model: FockMultiXiPARFLM v2.1 ...").')
else:
    _was_training = _m.training
    _m.eval()

    # The force term uses autograd.grad unless vtheta_analytic_force is on
    # (this is why the notebook's own evaluate() wraps its forward in
    # torch.enable_grad()). Under no_grad that path raises, so pick the
    # context the model actually supports and say which one was used --
    # enable_grad builds a graph and therefore OVERSTATES inference cost.
    import contextlib
    _analytic = bool(getattr(_cfg, 'vtheta_analytic_force', False))
    _ctx, _ctx_name = (torch.no_grad, 'no_grad') if _analytic else (
        torch.enable_grad, 'enable_grad')
    _probe = torch.randint(0, _cfg.vocab_size, (1, 64), device=DEV)
    try:
        with _ctx():
            _m(_probe, torch.roll(_probe, -1, dims=1))
    except RuntimeError as _e:
        if _ctx_name == 'no_grad':
            _ctx, _ctx_name = torch.enable_grad, 'enable_grad'
            print(f'  no_grad forward failed ({str(_e)[:60]}...) -- '
                  f'falling back to enable_grad')
        else:
            raise
    del _probe
    torch.cuda.empty_cache()
    print(f'  grad mode: {_ctx_name}'
          + ('' if _ctx_name == 'no_grad' else
             '  <-- builds a graph; these timings are an UPPER bound'))

    _maxT = int(getattr(_cfg, 'max_len', 1024))
    Tf = [t for t in Ts if t <= _maxT]
    print(f'model: d={_cfg.d} L={_cfg.L} M={_cfg.n_registers} '
          f'max_len={_maxT}  params={sum(p.numel() for p in _m.parameters()):,}')
    print(f'\n{"T":>6}{"B=1 ms":>11}{"us/token":>11}{"B=4 ms":>11}'
          f'{"us/token":>11}{"peak GB":>10}')
    for T in Tf:
        row = [T]
        peak = 0.0
        for B in (1, 4):
            x = torch.randint(0, _cfg.vocab_size, (B, T), device=DEV)
            y = torch.roll(x, -1, dims=1)
            torch.cuda.reset_peak_memory_stats()
            with _ctx():
                ms = bench(lambda: _m(x, y), n=3, warmup=1)
            peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
            fock_ms[(B, T)] = ms
            row += [ms, ms * 1000 / (B * T) if ms == ms else float('nan')]
            del x, y
            torch.cuda.empty_cache()
        print(f'{row[0]:>6}{row[1]:>11.2f}{row[2]:>11.1f}'
              f'{row[3]:>11.2f}{row[4]:>11.1f}{peak:>10.2f}')
    _sl = slope(Tf, [fock_ms[(1, t)] for t in Tf])
    print(f'\n  scaling in T (B=1): O(T^{_sl:.2f})')
    print('  Compare with the GPT-2 exponent above. A materially larger')
    print('  exponent is the (B,T,T) pi + softmax + topk and the K dense')
    print('  (T,T) xi matmuls showing up.')
    if _was_training:
        _m.train()
print()


# ===================================================================
# PART D -- what this means for inference
# ===================================================================
print('=' * 72)
print('PART D -- inference read-out')
print('=' * 72)

if fock_ms:
    print(f'{"T":>6}{"Fock ms":>11}{"GPT-2 ms":>11}{"ratio":>9}   (B=1 forward)')
    for T in [t for t in Ts if (1, t) in fock_ms]:
        f, g = fock_ms[(1, T)], gpt2_ms[(1, T)]
        print(f'{T:>6}{f:>11.2f}{g:>11.2f}{f / g:>8.1f}x')
    print('\nThat ratio is for PREFILL / batch SCORING -- one forward over a')
    print('full sequence. It is the number to quote for evaluation cost.')
else:
    print('(no Fock timings -- ratio table needs Part B)')

print("""
GENERATION is a different and worse story, and no measurement here covers
it, because the Fock model has no incremental path to measure:

  * no generate(), no KV cache, no state carry-forward in
    model_parf_multixi.py / model_parf_sparse.py / model_parf.py
  * every forward rebuilds the K dense (T,T) xi matrices and the
    (B,T,T) routing scores, softmax and top-k from scratch

So naive autoregressive decoding re-runs the whole forward per token:

    Fock,  no cache :  sum over t of O(t^2)  =  O(T^3)
    GPT-2, KV cache :  sum over t of O(t)    =  O(T^2)

Before any inference claim can be made, two things have to be built:

  1. xi as the recurrence (Part A) -- exact, not an approximation, and it
     helps training too.
  2. a decode path for the routing. ScoreHead DOES admit one:
     hidden[t,s] = proj_t[t] + proj_u[s], and proj_u[s] depends only on
     the source token, so it caches at H_s = 32 floats per token per
     layer. A new token then needs one row, O(T*H_s) work. That is
     ~24x CHEAPER per token than attention's O(2*T*d) = 768*T, on a
     cache about half the size of K+V. What the GELU between the two
     indices blocks is collapsing the sum over s into a FIXED-SIZE
     recurrent state -- but top-k ranking needs per-source scores
     anyway, so that was never available. The runtime state is O(T),
     not the O(1) claimed for the SPLM core, but "O(T) at half a KV
     cache" is a much softer correction than it first appears.

What actually sets the floor is V_theta, not the integrator. ABOBA takes
ONE force evaluation per layer (not 8 substeps), over 40 wells at rank 4.
Measured, V_theta costs 35.429 MMAC/token/layer, of which 35.405 is
GENERATING the well bank from xi and 0.025 is using it -- a 1441:1 ratio,
and 87.6% of the whole model. B_proj alone (1920 -> 12288) is 58.3% of
Fock's total inference cost. That is the term to attack; see
companion_notes/Fock_Inference_Productionization_Plan.md.
""")
print('done.')
