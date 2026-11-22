#!/usr/bin/env python3
"""CPU inference throughput: the conservative model against the matched GPT-2, and where the conservative model's
time goes (2026-11-17; the author's question: does the closed-form CfC + BAOAB design pay on a CPU?).

    python3 cpu_throughput_benchmark.py OUT MODEL FOLDER [lengths=512,2048,8192] [reps=3] [threads=8] [profile=512,2048]
                                        [name=<stem>]

MODEL is `ladder` (FOLDER a ladder run in ~/Downloads, built from its tag as the probes build it), `gpt2` (the matched
GPT-2 notebook's own class and checkpoint, naive attention as trained) or `gpt2_sdpa` (the same weights with
F.scaled_dot_product_attention, the fused kernel a deployed transformer would use). One sequence per forward, float32,
both under no_grad (the ladder model's analytic V_phi force needs no graph; grad=1 times it under enable_grad,
as the training notebook's evaluate() runs it). Timing only: past the trained context, the position tables are extended by tiling (the speed does
not depend on those values; the outputs there mean nothing). Per length: median wall time over `reps` forwards after
one warm-up, tokens per second, and the process's peak resident memory so far (lengths run in increasing order).
`profile=` lengths get one profiled forward (ladder only): the top operators by self CPU time and a summary by kind.
Writes OUT/cpu_bench_<stem>.txt and .json.

Stage 1 of the productization plan (2026-11-17; companion_notes/productization/Conservative_Core_Productization_Plan.md)
adds, for ladder models: timers=1 (wall time per component), bank=pb<width> (twin PB's arithmetic, timing only),
prune=<layer>:<wells> [mask=1] (lever N3), compile=layer
or compile=model (lever N2), and dump=<file.npy> / compare=<file.npy> (the logits on one fixed validation window,
with loss=, to check a variant against the unmodified model).
denormal=flush (2026-11-22) flushes subnormal floats to zero for the whole run (see the note at set_flush_denormal).
"""
import json, os, resource, sys, time
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve(); MODEL = sys.argv[2]; FOLDER = sys.argv[3]
OPTS = dict(a.split('=', 1) for a in sys.argv[4:] if '=' in a)
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))
LENGTHS = [int(t) for t in OPTS.get('lengths', '512,2048,8192').split(',')]
REPS = int(OPTS.get('reps', 3)); THREADS = int(OPTS.get('threads', 8))
PROF = [int(t) for t in OPTS.get('profile', '512,2048').split(',') if t] if MODEL == 'ladder' else []
NDEC = int(OPTS.get('decode_tokens', 32)) if OPTS.get('decode') == '1' else 0
TMAX = max(LENGTHS) + NDEC + (8 if NDEC else 0)

src = (DBG / 'copy_split.py').read_text()
src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', 'windows=1', f'threads={THREADS}']
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)
torch.set_num_threads(THREADS)
# 2026-11-22: denormal=flush flushes subnormal floats to zero. The folded context channels' keys carry -s log a, which
# grows with s, so the fused attention's softmax produces many subnormal exponentials; on x86 these are slow. On SR2-NP's
# folded channels at 8,192 tokens: 3.36 s -> 1.25 s per layer, output bit-identical. Applies to every model.
FLUSH_DENORMAL = OPTS.get('denormal') == 'flush' and torch.set_flush_denormal(True)


def tile(w, n):
    reps = -(-n // w.shape[0])
    return w.detach().repeat(reps, 1)[:n].clone()


if MODEL == 'ladder':
    import verify_pm_switch as P
    _build = P.build; got = {}
    def build(*a, **k):
        r = _build(*a, **k); got['model'] = r[0]; return r
    P.build = build
    _, desc = CS['ladder_model'](FOLDER, OPTS.get('ckpt'))
    m = got['model']
    if m.P.shape[0] < TMAX:
        m.P = torch.nn.Parameter(tile(m.P, TMAX)); m.cfg.max_len = TMAX
        if hasattr(m, 'xi_module'):
            m.xi_module.max_len = TMAX
    # 2026-11-17: the tag-based build leaves VPHI_ANALYTIC_FORCE at its notebook default (False), and the autograd
    # V_phi force silently drops out under no_grad (loss +0.016 nats on SR2). analytic=1 (default) restores the
    # trained configuration. chunk=<n> sets the inference-only query chunking of V_phi and xi; skipreg=1 skips the
    # unread Fock registers (debug/verify_inference_chunking.py).
    m.cfg.vphi_analytic_force = OPTS.get('analytic', '1') == '1'
    if OPTS.get('chunk'):
        m.cfg.vphi_query_chunk = int(OPTS['chunk']); m.xi_module.query_chunk = int(OPTS['chunk'])
    m.cfg.skip_unread_registers = OPTS.get('skipreg', '0') == '1'
    desc += (f"; analytic V_phi {m.cfg.vphi_analytic_force}; query chunk {OPTS.get('chunk', 'off')}; "
             f"skip unread registers {m.cfg.skip_unread_registers}")
    if OPTS.get('novphi', '0') == '1':
        # 2026-11-17: V_phi removed entirely (no score network, top-k, router or pair force): the upper bound on
        # what dropping or replacing it saves. Timing only; the loss reading below shows what it costs as trained.
        assert m.cfg.vphi_analytic_force, 'novphi replaces the analytic pair force'
        m._pair_force_analytic = lambda h_in, layer_idx, h_src_live=None, q_slice=None: torch.zeros_like(h_in)
        desc += '; V_phi removed'
    if OPTS.get('bank', '').startswith('pb'):
        assert m.V_theta.bank.banks[0].a_proj is not None and not hasattr(m.V_theta.bank.banks[0].B_proj, 'down'), \
            'bank=pb times a dense bank as twin PB; this model is already built with the twin PB switches'
        # 2026-11-22 (productization plan, item 4): bank=pb<width> times twin PB without training it. B_proj becomes
        # two maps through a bottleneck of <width> (its truncated SVD, no bias in the first map) and a_proj a fixed
        # value per well (its bias): the arithmetic of the trained twin, not its predictions (the SVD at 512 costs
        # +41% untrained; Well_Screen_and_Precision.md, section 5). Timing only. Before prune=, which copies rows.
        nn_ = torch.nn
        full = m.V_theta.bank.banks[0]; WB = int(OPTS['bank'][2:])
        class _Bottleneck(nn_.Module):
            def __init__(self, first, second):
                super().__init__(); self.first, self.second = first, second
            def forward(self, x):
                return self.second(self.first(x))
        class _Fixed(nn_.Module):
            def __init__(self, value):
                super().__init__(); self.value = nn_.Parameter(value)
            def forward(self, x):
                return self.value.expand(*x.shape[:-1], -1)
        U_, S_, Vh_ = torch.linalg.svd(full.B_proj.weight.data, full_matrices=False)
        first = nn_.Linear(full.B_proj.in_features, WB, bias=False); first.weight.data = (S_[:WB, None] * Vh_[:WB]).clone()
        second = nn_.Linear(WB, full.B_proj.out_features); second.weight.data = U_[:, :WB].clone()
        second.bias.data = full.B_proj.bias.data.clone()
        full.B_proj = _Bottleneck(first, second); full.a_proj = _Fixed(full.a_proj.bias.data.clone())
        desc += f'; bank timed as twin PB (B_proj through {WB}, a fixed per well; timing proxy, predictions not trained)'
    if OPTS.get('prune'):
        # 2026-11-17 (lever N3): prune=<layer>:<wells> keeps only those wells of the shared bank while that layer is
        # active, e.g. prune=1:3,4. The dropped wells get weight 0 without renormalisation, as in the keep-k screen
        # (vtheta_well_probe.py keep=, keep_layers=): mask=1 does exactly that on the full bank (the reference);
        # otherwise a copy of the bank with only the kept wells' rows of mu_proj, a_proj and B_proj is swapped in, so
        # the dropped wells cost nothing and the stiff-mode decomposition shrinks from K*r to len(wells)*r columns.
        # The weights still come from the full softmax over all K wells.
        import copy
        lay_s, ws = OPTS['prune'].split(':'); PL = int(lay_s); KEEPW = [int(c) for c in ws.split(',')]
        vt = m.V_theta; joint = vt.bank; full = joint.banks[0]; d_, r_ = full.d, full.rank
        if OPTS.get('mask', '0') == '1':
            _orig = full._components
            _drop = torch.tensor([k not in KEEPW for k in range(full.K)])
            def _masked(xi):
                mu, a, w, B = _orig(xi)
                return (mu, a, w.masked_fill(_drop, 0.0), B) if vt._active_layer == PL else (mu, a, w, B)
            full._components = _masked
            desc += f'; layer {PL} wells {KEEPW} kept by masking the others (reference)'
        else:
            sub = copy.deepcopy(full); sub.K = len(KEEPW)
            def _rows(lin, width):
                idx = torch.cat([torch.arange(k * width, (k + 1) * width) for k in KEEPW])
                if hasattr(lin, 'second'):                           # bank=pb: keep the shared first map
                    new = copy.deepcopy(lin); new.second = _rows(lin.second, width); return new
                if hasattr(lin, 'up'):                               # a trained twin PB: keep the shared down map
                    new = copy.deepcopy(lin); new.up = _rows(lin.up, width); new.out_features = new.up.out_features
                    return new
                if hasattr(lin, 'value'):
                    new = copy.deepcopy(lin); new.value.data = lin.value.data[idx].clone(); return new
                new = torch.nn.Linear(lin.in_features, idx.numel())
                new.weight.data = lin.weight.data[idx].clone(); new.bias.data = lin.bias.data[idx].clone()
                return new
            sub.mu_proj = _rows(full.mu_proj, d_)
            if full.a_proj is not None:
                sub.a_proj = _rows(full.a_proj, d_)
            else:                                                    # twin PA/PB: the kept wells' context-free values
                sub.a_free = torch.nn.Parameter(
                    full.a_free.data.view(full.n_a_layers, full.K, d_)[:, KEEPW].reshape(-1).clone())
            sub.B_proj = _rows(full.B_proj, d_ * r_); sub.w_proj = None
            _kw = torch.tensor(KEEPW)
            F_ = torch.nn.functional
            def _sub_components(xi):
                lead = xi.shape[:-1]
                mu = sub.mu_proj(xi).view(*lead, sub.K, d_)
                if sub.a_proj is not None:
                    a = (F_.softplus(sub.a_proj(xi)) + 1e-4).view(*lead, sub.K, d_)
                else:
                    a_l = sub.a_free.view(sub.n_a_layers, sub.K, d_)[sub.active_a_layer % sub.n_a_layers]
                    a = (F_.softplus(a_l) + 1e-4).expand(*lead, sub.K, d_)
                if sub._precision_max is not None:
                    a = a.clamp(max=sub._precision_max)
                w = (F_.softmax(full.w_proj(xi), dim=-1) * full.w_scale)[..., _kw]
                B = sub._bound_lowrank(sub.B_proj(xi).view(*lead, sub.K, d_, r_))
                return mu, a, w, B
            sub._components = _sub_components
            _set = vt.set_active_layer
            def _set_active(i):
                _set(i); joint.banks[0] = sub if vt._active_layer == PL else full
                joint.banks[0].active_a_layer = vt._active_layer % vt.n_layers      # twin PA/PB: after the swap
            vt.set_active_layer = _set_active
            desc += f'; layer {PL} pruned to wells {KEEPW}'
    if 'xichunk' in OPTS:
        # 2026-11-17: the context channels' query chunking set apart from the pair force's (0 = off): on its own the
        # channels' chunking saves memory but costs speed (results/productization/xi_sdpa_microbench.txt)
        m.xi_module.query_chunk = int(OPTS['xichunk']) or None
        desc += f"; context-channel query chunk {OPTS['xichunk']}"
    if OPTS.get('xiplain'):
        # 2026-11-17 (lever T3, first reading): xiplain=<layers> drops the content term of the shared context channels
        # on those layers, which turns them into the exact causal moving averages (the W_q = None branch of
        # MultiChannelXi.forward). A removal from the trained model: it reads reliance, not what training without it
        # would cost. The channels are called inside each layer's step, after the active layer is set.
        xm, vt_ = m.xi_module, m.V_theta
        XP = [int(c) for c in OPTS['xiplain'].split(',')]
        _xf = xm.forward
        def _xi_plain(h):
            if vt_._active_layer not in XP:
                return _xf(h)
            wq = xm.W_q; xm.W_q = None
            try:
                return _xf(h)
            finally:
                xm.W_q = wq
        xm.forward = _xi_plain
        desc += f'; content term of the context channels removed on layers {XP}'
    if OPTS.get('xi') == 'fold':
        # 2026-11-17 (lever N1): the content-routed channels by F.scaled_dot_product_attention with the distance term
        # folded into the keys: (t - s) log a = t log a - s log a, the row constant cancels in the softmax, so one
        # extra query component sqrt(d_k) and one extra key component -s log a give plain causal attention, no mask.
        # Not bit-exact: in float32 the key term grows with s (xi_sdpa_microbench.py; compare= reads the logits).
        xm = m.xi_module
        assert xm.W_q is not None, 'xi=fold needs the content-routed channels'
        def _xi_fold(h):
            B, T, _ = h.shape; K_, dk_ = xm.K, xm.content_d_k
            la = torch.log(xm.alpha.clamp(min=1e-6, max=1 - 1e-6))
            q = xm.W_q(h).view(B, T, K_, dk_).permute(0, 2, 1, 3)
            kk = xm.W_k(h).view(B, T, K_, dk_).permute(0, 2, 1, 3)
            s = torch.arange(T, dtype=h.dtype, device=h.device)
            q = torch.cat([q, torch.full((B, K_, T, 1), dk_ ** 0.5, dtype=h.dtype, device=h.device)], -1)
            kk = torch.cat([kk, (-s.view(1, 1, T, 1) * la.view(1, K_, 1, 1)).expand(B, K_, T, 1)], -1)
            o = torch.nn.functional.scaled_dot_product_attention(
                q, kk, h.unsqueeze(1).expand(-1, K_, -1, -1), is_causal=True, scale=dk_ ** -0.5)
            return o.permute(0, 2, 1, 3).contiguous()                  # (B, T, K, d)
        xm.forward = _xi_fold
        desc += '; context channels by folded fused attention'
    if OPTS.get('xi') == 'fold64':
        # 2026-11-22 (after lever N6): the folded channels as 5 x 6 = 30 heads of width 64. Queries and keys (48 + the
        # folded distance component = 49) are zero-padded to 64, which leaves every score unchanged; the 384-wide value
        # is split into six blocks of 64, each attended with its channel's scores. Same computation as xi=fold, with
        # equal head widths, which the fused CPU kernel runs faster (sdpa_bench: 1.29 -> 0.63 s per layer at 8,192).
        xm = m.xi_module
        assert xm.W_q is not None, 'xi=fold64 needs the content-routed channels'
        def _xi_fold64(h):
            B, T, D = h.shape; K_, dk_ = xm.K, xm.content_d_k
            nb = D // 64
            assert nb * 64 == D and dk_ + 1 <= 64
            la = torch.log(xm.alpha.clamp(min=1e-6, max=1 - 1e-6))
            q = xm.W_q(h).view(B, T, K_, dk_).permute(0, 2, 1, 3)
            kk = xm.W_k(h).view(B, T, K_, dk_).permute(0, 2, 1, 3)
            s = torch.arange(T, dtype=h.dtype, device=h.device)
            pad = torch.zeros((B, K_, T, 64 - dk_ - 1), dtype=h.dtype, device=h.device)
            q = torch.cat([q, torch.full((B, K_, T, 1), dk_ ** 0.5, dtype=h.dtype, device=h.device), pad], -1)
            kk = torch.cat([kk, (-s.view(1, 1, T, 1) * la.view(1, K_, 1, 1)).expand(B, K_, T, 1), pad], -1)
            q = q.unsqueeze(2).expand(B, K_, nb, T, 64).reshape(B, K_ * nb, T, 64)
            kk = kk.unsqueeze(2).expand(B, K_, nb, T, 64).reshape(B, K_ * nb, T, 64)
            v = h.view(B, T, nb, 64).permute(0, 2, 1, 3).unsqueeze(1).expand(B, K_, nb, T, 64).reshape(B, K_ * nb, T, 64)
            o = torch.nn.functional.scaled_dot_product_attention(q, kk, v, is_causal=True, scale=dk_ ** -0.5)
            return o.view(B, K_, nb, T, 64).permute(0, 3, 1, 2, 4).reshape(B, T, K_, D)   # (B, T, K, d)
        xm.forward = _xi_fold64
        desc += '; context channels by folded fused attention, 30 heads of 64'
    if OPTS.get('gram') == 'eigh64':
        # 2026-11-17 (lever N5): the gram driver's float64 SVD of the symmetric PSD Gram matrix replaced by a float64
        # symmetric eigensolve, same hardenings (symmetrise, float64, ramp), values put in the descending order the
        # callers expect. The float32 eigh of 2026-09-17 failed on real matrices; this one stays in float64.
        import cfc_baoab as _cb
        def _gram_eigh64(Gd):
            dt_in, P = Gd.dtype, Gd.shape[-1]
            M = Gd.transpose(-1, -2) @ Gd
            M = (0.5 * (M + M.transpose(-1, -2))).double()
            scale = M.diagonal(dim1=-2, dim2=-1).amax(-1).clamp(min=1e-300)
            ramp = torch.arange(P, device=M.device, dtype=M.dtype) * (1e-12 / max(P, 1))
            M = M + torch.diag_embed(scale.unsqueeze(-1) * ramp)
            lam, V = torch.linalg.eigh(M)                    # ascending
            lam, V = lam.flip(-1).clamp(min=0.0), V.flip(-1)
            inv_sqrt = torch.where(lam > 0, lam.clamp(min=1e-300).rsqrt(), torch.zeros_like(lam))
            U = (Gd.double() @ V) * inv_sqrt.unsqueeze(-2)
            return U.to(dt_in), lam.sqrt().to(dt_in)
        assert _cb._gram_eigh.__module__ == _cb.__name__
        _cb._gram_eigh = _gram_eigh64
        desc += '; gram driver by float64 eigh'
    if OPTS.get('compile'):
        # 2026-11-17 (lever N2): compile=layer compiles one layer's step (the integrator, the bank's force, the pair
        # force); compile=core its part from the linearisation on; compile=model the whole forward. Inductor on the CPU; graph breaks fall back to eager.
        # PyTorch 2.2.2 supports torch.compile on Python <= 3.11 only. On an Intel Mac, Apple clang has no OpenMP:
        # set OMP_PREFIX to a folder holding include/omp.h (LLVM's), and the kernels are linked here against the
        # libiomp5 that libtorch_cpu already loads (functorch/.dylibs), so one OpenMP runtime is in the process.
        if sys.platform == 'darwin' and os.environ.get('OMP_PREFIX'):
            import functorch, torch._inductor.codecache as _cc
            _omp_dir = os.path.join(os.path.dirname(functorch.__file__), '.dylibs')
            _paths = _cc.get_include_and_linking_paths
            def _with_iomp5(*a, **k):
                ip, lp, libs, mac, arch = _paths(*a, **k)
                return ip, f'{lp} -L{_omp_dir} -Wl,-rpath,{_omp_dir}', f'{libs} -liomp5', mac, arch
            _cc.get_include_and_linking_paths = _with_iomp5
        # The layer step asks torch.is_autocast_enabled(device), which PyTorch 2.2's Dynamo cannot trace; autocast is
        # never on at CPU inference, so the check is a constant here.
        sys.modules[type(m)._layer_step_langevin.__module__]._autocast_active = lambda *a, **k: False
        if OPTS['compile'] == 'layer':
            m._layer_step_langevin = torch.compile(m._layer_step_langevin)
        elif OPTS['compile'] == 'core':
            # the step from the linearisation on (integrator, forces, decomposition): its inputs are per position and
            # never carry the position, so under decoding (one token, fixed shapes) it compiles once
            m._layer_step_langevin_core = torch.compile(m._layer_step_langevin_core)
        else:
            m.forward = torch.compile(m.forward)
        desc += f"; torch.compile ({OPTS['compile']})"
    TIMERS = {}
    if OPTS.get('timers', '0') == '1':
        # 2026-11-17 (productization plan, stage 1): wall time per component, inclusive, summed over the calls of one
        # forward (CPU ops are synchronous, so perf_counter around each call is exact). The rest of the forward is
        # the integrator's own work, the embeddings and the output head.
        import functools
        _depth = {}
        def timed(name, fn):                # outermost calls only: the chunked pair force calls itself per block
            @functools.wraps(fn)
            def w(*a, **k):
                _depth[name] = _depth.get(name, 0) + 1
                t0 = time.perf_counter()
                try:
                    return fn(*a, **k)
                finally:
                    _depth[name] -= 1
                    if _depth[name] == 0:
                        c = TIMERS.setdefault(name, [0.0, 0]); c[0] += time.perf_counter() - t0; c[1] += 1
            return w
        m.xi_module.forward = timed('context channels (xi)', m.xi_module.forward)
        m._pair_force_analytic = timed('pair force (V_phi, score network, top-k, router)', m._pair_force_analytic)
        for nm in ('context_components', 'harmonic_terms_lowrank', 'analytical_grad'):
            if hasattr(m.V_theta, nm):
                setattr(m.V_theta, nm, timed(f'well bank: {nm}', getattr(m.V_theta, nm)))
        _mod = sys.modules[type(m)._layer_step_langevin.__module__]
        _mod.lowrank_modes = timed('stiff-mode decomposition (lowrank_modes)', _mod.lowrank_modes)
    GRAD = OPTS.get('grad', '0') == '1'
    def run(x):                             # (verify_vphi_analytic_output.txt: no_grad, max |dlogit| 0)
        with (torch.enable_grad() if GRAD else torch.no_grad()):
            return m(x)[0]
else:
    import contextlib, io
    nb = DBG.parent / 'colab_matched_gpt2_baseline_openwebtext.ipynb'
    cells = [''.join(c['source']) for c in json.load(open(nb))['cells']]
    ck = torch.load(Path.home() / 'Downloads' / FOLDER / 'checkpoints' / 'gpt2_baseline_best.pt', map_location='cpu',
                    weights_only=False)
    g = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec('import math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np', g)
        exec(cells[1], g); exec(cells[4], g)
    m = g['GPT2'](vocab_size=g['VOCAB_SIZE'], d=g['D_MODEL'], n_layers=g['N_LAYERS'], n_heads=g['N_HEADS'],
                  d_ff=g['D_FF'], max_len=g['MAX_LEN'], dropout=0.0, tie_embeddings=g['TIE_EMBEDDINGS'])
    m.load_state_dict(ck['model_state_dict']); m.eval()
    desc = f"gpt2_baseline_best.pt (step {ck['step']}, val {ck['val_ppl']:.3f}); L {g['N_LAYERS']}, d {g['D_MODEL']}"
    if m.pos_emb.weight.shape[0] < TMAX:
        pe = torch.nn.Embedding(TMAX, m.pos_emb.weight.shape[1]); pe.weight.data = tile(m.pos_emb.weight, TMAX)
        m.pos_emb = pe
        mask = torch.tril(torch.ones(TMAX, TMAX)).view(1, 1, TMAX, TMAX)
        for b in m.blocks:
            b.attn.mask = mask                      # one shared buffer
    if MODEL == 'gpt2_sdpa':
        F = torch.nn.functional
        def fwd(self, x):
            B, T, C = x.shape
            q, k, v = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
            return self.proj(out)
        for b in m.blocks:
            b.attn.forward = fwd.__get__(b.attn)
        desc += '; attention by F.scaled_dot_product_attention'
    def run(x):
        with torch.no_grad():
            return m(x)[0]

m.eval()
n_par = sum(p.numel() for p in m.parameters())
print('model:', desc, f'{n_par:,} parameters', flush=True)
rng = np.random.default_rng(0)
res = {'model': MODEL, 'folder': FOLDER, 'desc': desc + ('; subnormals flushed' if FLUSH_DENORMAL else ''),
       'params': n_par, 'threads': THREADS, 'flush_denormal': bool(FLUSH_DENORMAL), 'lengths': {}}
lines = [f'CPU inference benchmark: {MODEL}  {FOLDER}', f'{desc}; {n_par:,} parameters; {THREADS} threads; float32; '
         'one sequence per forward', '', f'{"length":>6}  {"median s":>9}  {"tokens/s":>9}  {"peak RSS GB":>11}  times']
if NDEC:
    # 2026-11-17 (stage 2, lever N4): token-by-token decoding with a cache (parf/incremental_decoder.py). Ladder models
    # decode with the pair force dropped (novphi=1 required). decode_check=<n> first compares the decoder's logits with
    # the full forward's on one validation window of n tokens (needs loss=): the first half as the prefill, the rest
    # one token at a time. Then, per length T: a prefill of T random tokens, and decode_tokens single-token steps;
    # median seconds per step, tokens per second, the cache size and the peak resident memory.
    sys.path.insert(0, str(DBG.parents[1] / 'parf'))
    from incremental_decoder import GPT2Decoder, LadderDecoder
    def make_decoder(n):
        if MODEL == 'ladder':
            if OPTS.get('novphi', '0') != '1' and m.V_phi is not None:   # SR2-NP (2026-11-21) has none to drop
                sys.exit('decode=1 on a ladder model needs novphi=1: the decoder has no cache for the pair force')
            return LadderDecoder(m, n, drop_pair_force=True)
        return GPT2Decoder(m, n)
    lines = [f'CPU decoding benchmark: {MODEL}  {FOLDER}', f'{desc}; {n_par:,} parameters; {THREADS} threads; float32; '
             f'batch 1; {NDEC} decoded tokens per length', '']
    if OPTS.get('decode_check'):
        nchk = int(OPTS['decode_check']); vv = np.load(OPTS['loss']).astype(np.int64)
        xc = torch.from_numpy(vv[None, 1000:1000 + nchk])
        full_lg = run(xc)[0]
        dec_ = make_decoder(nchk); half = nchk // 2
        parts = [dec_.forward(xc[:, :half])[0]] + [dec_.forward(xc[:, j:j + 1])[0] for j in range(half, nchk)]
        if hasattr(dec_, 'close'):
            dec_.close()
        got = torch.cat(parts, 0)
        dif = (got - full_lg).abs()
        res['decode_check'] = {'tokens': nchk, 'max_abs_prefill': float(dif[:half].max()),
                               'max_abs_steps': float(dif[half:].max()), 'mean_abs_steps': float(dif[half:].mean()),
                               'argmax_agree': float((got.argmax(-1) == full_lg.argmax(-1)).float().mean())}
        lines += [f'decoder against the full forward on {nchk} validation tokens (prefill {half}, then single steps): '
                  f'max |dlogit| prefill {dif[:half].max():.3e}, steps {dif[half:].max():.3e} '
                  f'(mean {dif[half:].mean():.3e}); top-1 agreement {100 * res["decode_check"]["argmax_agree"]:.2f}%', '']
        print(lines[-2], flush=True)
    lines.append(f'{"context":>7}  {"prefill s":>9}  {"s/token":>9}  {"tokens/s":>9}  {"cache MB":>8}  {"peak RSS GB":>11}')
    res['decode'] = {}
    for T in LENGTHS:
        dec_ = make_decoder(T + NDEC + 8)                # 8 spare positions for decode_profile
        x = torch.from_numpy(rng.integers(0, 50257, size=(1, T)).astype(np.int64))
        t0 = time.perf_counter(); dec_.forward(x); tp = time.perf_counter() - t0
        toks = torch.from_numpy(rng.integers(0, 50257, size=(1, NDEC)).astype(np.int64))
        ts = []
        for j in range(NDEC):
            t0 = time.perf_counter(); o = dec_.forward(toks[:, j:j + 1]); ts.append(time.perf_counter() - t0)
            assert torch.isfinite(o).all()
        if OPTS.get('decode_profile') == '1' and T == LENGTHS[0]:
            # where a decode step's time goes: 8 profiled steps; operator calls per step, and self CPU time by kind
            from torch.profiler import profile, ProfilerActivity
            more = torch.from_numpy(rng.integers(0, 50257, size=(1, 8)).astype(np.int64))
            t0 = time.perf_counter()
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                for j in range(8):
                    dec_.forward(more[:, j:j + 1])
            wall = (time.perf_counter() - t0) / 8
            ka = prof.key_averages(); tot = sum(e.self_cpu_time_total for e in ka) / 8; n_ops = sum(e.count for e in ka) / 8
            lines += ['', f'decode profile at context {T}: {n_ops:.0f} operator calls per step; wall {1e3 * wall:.1f} ms '
                      f'per step under the profiler, operator self time {tot / 1e3:.1f} ms (summed over threads)',
                      '  top operators by self CPU time:'] + \
                     ['  ' + l for l in ka.table(sort_by='self_cpu_time_total', row_limit=15).split('\n')] + ['']
        if hasattr(dec_, 'close'):
            dec_.close()
        med = float(np.median(ts[2:] if NDEC > 4 else ts))          # the first steps warm the allocator
        cache_mb = dec_.cache_numbers_per_token() * (T + NDEC) * 4 / 2 ** 20
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
        res['decode'][T] = {'prefill_s': tp, 'step_times': ts, 'median_step_s': med, 'tokens_per_s': 1 / med,
                            'cache_mb': cache_mb, 'cache_numbers_per_token': dec_.cache_numbers_per_token(),
                            'peak_rss_gb': rss}
        lines.append(f'{T:>7}  {tp:9.3f}  {med:9.4f}  {1 / med:9.1f}  {cache_mb:8.1f}  {rss:11.2f}')
        print(lines[-1], flush=True)
    lines.append(f'cache: {dec_.cache_numbers_per_token():,} numbers per token')
    stem = OPTS.get('name', MODEL)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'decode_bench_{stem}.txt').write_text('\n'.join(lines) + '\n')
    (OUT / f'decode_bench_{stem}.json').write_text(json.dumps(res, indent=1, default=str))
    print('\n'.join(lines))
    sys.exit(0)

for T in LENGTHS:
    x = torch.from_numpy(rng.integers(0, 50257, size=(1, T)).astype(np.int64))
    run(x)                                           # warm-up
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); out = run(x); ts.append(time.perf_counter() - t0)
        assert torch.isfinite(out).all()
    med = float(np.median(ts)); rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30   # macOS: bytes
    res['lengths'][T] = {'times': ts, 'median_s': med, 'tokens_per_s': T / med, 'peak_rss_gb': rss}
    lines.append(f'{T:>6}  {med:9.3f}  {T / med:9.1f}  {rss:11.2f}  ' + ' '.join(f'{t:.3f}' for t in ts))
    print(lines[-1], flush=True)
    if MODEL == 'ladder' and TIMERS is not None and OPTS.get('timers', '0') == '1':
        TIMERS.clear(); t0 = time.perf_counter(); run(x); tot = time.perf_counter() - t0
        res['lengths'][T]['components'] = {k: {'s': v[0], 'calls': v[1]} for k, v in TIMERS.items()}
        lines += [f'        components at {T} (one timed forward, {tot:.3f} s; inclusive wall time, calls):'] + \
                 [f'          {100 * v[0] / tot:5.1f}%  {v[0]:8.3f} s  {v[1]:3d}  {k}' for k, v in TIMERS.items()] + \
                 [f'          {100 * (1 - sum(v[0] for v in TIMERS.values()) / tot):5.1f}%  the rest (integrator, embeddings, output head)']
        print('\n'.join(lines[-len(TIMERS) - 2:]), flush=True)

if OPTS.get('loss'):
    # 2026-11-17: the validation loss of the configuration as timed, on evenly spaced windows of 512 tokens
    vv = np.load(OPTS['loss']).astype(np.int64); nw = int(OPTS.get('loss_windows', 16))
    stl = np.linspace(0, len(vv) - 513, nw).astype(np.int64)
    xyl = torch.from_numpy(np.stack([vv[s_:s_ + 513] for s_ in stl]))
    ces = []
    for j in range(0, nw, 4):
        lgl = run(xyl[j:j + 4, :512])
        ces.append(torch.nn.functional.cross_entropy(lgl.reshape(-1, lgl.shape[-1]).float(), xyl[j:j + 4, 1:].reshape(-1)))
    lv = float(torch.stack(ces).mean())
    res['val_loss'] = lv
    lines += ['', f'validation loss on {nw} windows of 512 ({Path(OPTS["loss"]).name}): {lv:.4f} (ppl {np.exp(lv):.2f})']
    print(lines[-1], flush=True)

if OPTS.get('dump') or OPTS.get('compare'):
    # 2026-11-17 (productization stage 1): the logits on one fixed validation window of 512 tokens (needs loss=),
    # saved with dump=<file.npy>, or compared with compare=<file.npy> saved from the unmodified model
    vv = np.load(OPTS['loss']).astype(np.int64)
    lg = run(torch.from_numpy(vv[None, 1000:1512]))[0].float().numpy()
    if OPTS.get('dump'):
        np.save(OPTS['dump'], lg)
        lines += [f'logits on validation tokens 1000..1511 saved to {Path(OPTS["dump"]).name}']
    if OPTS.get('compare'):
        ref = np.load(OPTS['compare'])
        dif = np.abs(lg - ref)
        res['compare'] = {'max_abs': float(dif.max()), 'mean_abs': float(dif.mean()),
                          'argmax_agree': float((lg.argmax(-1) == ref.argmax(-1)).mean())}
        lines += [f'logits against {Path(OPTS["compare"]).name}: max |diff| {dif.max():.3e}, mean |diff| {dif.mean():.3e}, '
                  f'top-1 agreement {100 * res["compare"]["argmax_agree"]:.2f}%']
    print(lines[-1], flush=True)

KINDS = [('decompositions (svd, eigh, qr, cholesky, solve)', ('svd', 'eigh', 'qr', 'cholesky', 'solve', 'lu')),
         ('backward (autograd)', ('backward',)),
         ('matrix products (mm, bmm, matmul, einsum, addmm, linear)', ('mm', 'bmm', 'matmul', 'einsum', 'linear', 'addmm', 'baddbmm')),
         ('softmax', ('softmax',)), ('top-k, sort, gather, scatter', ('topk', 'sort', 'gather', 'scatter', 'index'))]
for T in PROF:
    x = torch.from_numpy(rng.integers(0, 50257, size=(1, T)).astype(np.int64))
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        run(x)
    ka = prof.key_averages()
    tot = sum(e.self_cpu_time_total for e in ka)
    by = {k: 0.0 for k, _ in KINDS}; other = 0.0
    for e in ka:
        nm = e.key.lower()
        for k, pats in KINDS:
            if any(p in nm for p in pats):
                by[k] += e.self_cpu_time_total; break
        else:
            other += e.self_cpu_time_total
    by['other (elementwise, copies, reductions)'] = other
    res.setdefault('profile', {})[T] = {k: v / tot for k, v in by.items()}
    lines += ['', f'profile at length {T}: self CPU time {tot / 1e6:.2f} s summed over threads and operators'] + \
             [f'  {100 * v / tot:5.1f}%  {k}' for k, v in by.items()] + ['', '  top operators by self CPU time:'] + \
             ['  ' + l for l in ka.table(sort_by='self_cpu_time_total', row_limit=20).split('\n')]
stem = OPTS.get('name', MODEL)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f'cpu_bench_{stem}.txt').write_text('\n'.join(lines) + '\n')
(OUT / f'cpu_bench_{stem}.json').write_text(json.dumps(res, indent=1, default=str))
print('\n'.join(lines[-40:]))
