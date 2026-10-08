"""Is the trained layer force a gradient? A direct test on a checkpoint.

Written 2026-10-07 for PM1 (protocol SS5.15, Poisson_Mode_Registers_PM1.md
SS4), with F3.1 and G2 as controls. Conservativity of the PM1 step rests on
three claims; this script tests the third on the trained weights:

  A. the step contains only V_theta, V_phi, the PM1 wells and LayerNorm
     (Cell 6b-9, GATE 0 with the reverse channel off);
  B. each of those forces is a gradient by construction (V_theta analytic,
     V_phi the gradient of a pair potential, PM1 proved in the note SS4);
  C. the TRAINED forces are gradients: for each token, at fixed context (the
     other tokens' states and the context channels xi held fixed), the force
     on token t as a function of its own state h_t has a SYMMETRIC Jacobian,
     and zero work around closed loops.

The sense is per token at fixed context, the book's property (C). Causal
forces are one-way between tokens, so no single energy exists for the whole
sequence; that is not tested and not claimed (note SS4.4).

METHOD. The model is built through the ladder notebook's own Cells 0-5b on the
best checkpoint. One forward pass over 256 validation tokens captures every
layer's input state (and, for a model with a reverse channel, the reverse
channel's inputs). Then, per layer and for 5 token positions, each force term
F(x) is evaluated with row t of the captured state replaced by x:

  V_theta      the analytic V_theta force
  V_phi        f_phi with the PM1 wells switched off
  PM1          the Poisson-mode well force                    (if POISSON_MODES)
  total        V_theta + V_phi + PM1, the conservative force the step uses
  RC           the reverse channel at its captured registers  (if present)

  1. Jacobian symmetry, with 12 random probe pairs (u, v): u'Jv against v'Ju,
     each from one vector-Jacobian product. asym = RMS(u'Jv - v'Ju) /
     RMS(u'Jv, v'Ju). A symmetric J gives float noise.
  1b. The same symmetry by FINITE DIFFERENCES of the force itself (added
     2026-10-07 after the first G2 control, see below): u'(F(x+ev)-F(x-ev))/2e
     against v'(F(x+eu)-F(x-eu))/2e, e = 1e-3 |x|. fd_asym as asym.
  2. Closed-loop work around 2 circles per token in random planes through
     h_t, radius 2% of |h_t|, 48 points (periodic trapezoid, spectrally
     accurate for a smooth field). loop = |closed work| / (sum |F . dx|).

The forces are differentiated through V_phi's own autograd force, which needs
create_graph, which the model ties to training mode. Training mode is used
with V_phi's Gumbel selection noise switched off, and the force VALUES are
checked bit-identical to eval mode at the captured states before anything is
measured.

WHY 1b (added after the first controls; the thresholds are unchanged). V_phi
selects its top-k sources with a score head whose query input is the token's
own state, through a straight-through mask m = (m_hard - k y).detach() + k y,
y a softmax over the scores. The force VALUE uses m_hard; its autograd
derivative treats the detached part as a constant. So autograd differentiates
a surrogate scalar whose Hessian is symmetric by construction, while the real
force field F(x) = -sum m_hard grad V - k sum V grad y carries the router
term, which need not be a gradient. On G2 the first run showed exactly that
signature: V_phi symmetric by autograd (asym <= 5e-6) but closed-loop work up
to 0.09 at layer 0. Finite differences evaluate the real force and see the
router term; autograd does not.

OUTCOME OF THE CONTROLS, and what changed (2026-10-07, before any PM1 run):
  - float32 finite differences have a floor that scales inversely with the
    force: V_theta, an exact analytic gradient, reads fd_asym 3e-4 to 1e-2.
    The 1e-3 threshold cannot be applied to that column. fd_asym is therefore
    DESCRIPTIVE, read against the V_theta row of the same layer, and the
    verdict uses asym and loop only, with the thresholds as pre-stated.
  - the router term is measured EXACTLY instead (router mode, below) and is
    reported in every full run. F3.1: 0 at layer 1; at layer 0 median 0.006%,
    p95 2.4% of the total conservative force. G2: 11% median at layer 0 (its
    V_phi has nearly collapsed, so the router dominates what is left).
  - a second, separate caveat is not measured: hard top-k selection makes
    V_phi's potential switch when the selected set changes, so the field is a
    gradient piecewise; a loop that crosses a switch has non-zero work. The
    2%-radius loops rarely cross one.

PRE-STATED READING (before any run, 2026-10-07):
  conservative   asym <= 1e-3 and loop <= 1e-3 for the term, at every layer
                 (worst token). float32 noise sits around 1e-6 to 1e-4.
  not            asym >= 0.1 or loop >= 0.1.
  VALIDITY       the test counts only if G2's RC term reads "not" and F3.1's
                 total reads "conservative". A PM1 verdict is quoted only on a
                 valid test.

ROUTER MODE (option 'router', added 2026-10-07). Measures V_phi's straight-
through router term EXACTLY, without finite differences: the force is
evaluated once as trained and once with the score head's query input
detached, which removes the router gradient and leaves -sum m_hard grad V, a
gradient at fixed source selection. The difference IS the router term. Per
layer: its norm as a share of the V_phi force and of the total conservative
force on the token (the share of the step's force that is not a gradient).

Usage: python3 conservativity_test_checkpoint.py OUT_DIR FOLDER RC VPHI XI [pm<K>] [pmclip<thr>] [router]
  F3.1: ... OUT semsimula_..._norc_vplive_xilive_L2probe_..._noattn False live live
  G2:   ... OUT semsimula_..._vplive_xilive_L2probe_..._noattn True live live
  PM1:  ... OUT semsimula_..._norc_vplive_xilive_pm64_L2probe_..._noattn False live live pm64
  FOLDER is mirrored under ~/Downloads WITH checkpoints/.
"""
import json, math, re, sys
from pathlib import Path

import numpy as np
import torch

T_SEQ, POSITIONS, N_PROBES, N_LOOPS, N_PTS, RADIUS = 256, (16, 64, 128, 200, 255), 12, 2, 48, 0.02
CONS, NOT = 1e-3, 0.1


def verdict(asym, loop, fd=0.0):
    # fd is descriptive (its float32 floor exceeds the threshold); see docstring
    if asym <= CONS and loop <= CONS:
        return 'conservative'
    if asym >= NOT or loop >= NOT:
        return 'NOT conservative'
    return 'unclear'


def replace_row(h, t, x):
    return torch.cat([h[:, :t], x[None, None], h[:, t + 1:]], 1)


def term_functions(model, h, xis, l, t, rc_in):
    """name -> F(x): force on token t with its state replaced by x."""
    has_pm = getattr(model, 'pm_mu', None) is not None

    def forces(x, pm_on=True):
        H = replace_row(h, t, x)
        if not pm_on and has_pm:
            model.poisson_mode_force = lambda h_, li_: torch.zeros_like(h_)
        try:
            ft, fp = model._layer_forces(H, xis, l, split=True)
        finally:
            model.__dict__.pop('poisson_mode_force', None)
        return ft[0, t], fp[0, t]

    fns = {'V_theta': lambda x: forces(x)[0],
           'V_phi': lambda x: forces(x, pm_on=False)[1]}
    if has_pm:
        fns['PM1'] = lambda x: type(model).poisson_mode_force(model, replace_row(h, t, x), l)[0, t]
    fns['total'] = lambda x: sum(forces(x))
    if rc_in is not None:
        hn, r, act = rc_in
        fns['RC'] = lambda x: model.reverse_ch(replace_row(hn, t, x), r, act)[0, t]
    return fns


def asymmetry(F, x0, gen):
    x = x0.clone().requires_grad_(True)
    y = F(x)
    a, b = [], []
    for _ in range(N_PROBES):
        u = torch.randn(x.shape, generator=gen, dtype=x.dtype)
        v = torch.randn(x.shape, generator=gen, dtype=x.dtype)
        gu, = torch.autograd.grad(y @ u, x, retain_graph=True)     # J^T u
        gv, = torch.autograd.grad(y @ v, x, retain_graph=True)     # J^T v
        a.append(float(v @ gu)); b.append(float(u @ gv))           # u'Jv, v'Ju
    a, b = np.array(a), np.array(b)
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean((a ** 2 + b ** 2) / 2)) + 1e-30))


def fd_asymmetry(F, x0, gen, rel_eps=1e-3):
    eps = rel_eps * float(x0.norm())

    def Fv(x):
        with torch.enable_grad():
            return F(x.clone().requires_grad_(True)).detach()
    a, b = [], []
    for _ in range(N_PROBES):
        u = torch.randn(x0.shape, generator=gen, dtype=x0.dtype); u /= u.norm()
        v = torch.randn(x0.shape, generator=gen, dtype=x0.dtype); v /= v.norm()
        jv = (Fv(x0 + eps * v) - Fv(x0 - eps * v)) / (2 * eps)     # J v
        ju = (Fv(x0 + eps * u) - Fv(x0 - eps * u)) / (2 * eps)     # J u
        a.append(float(u @ jv)); b.append(float(v @ ju))
    a, b = np.array(a), np.array(b)
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean((a ** 2 + b ** 2) / 2)) + 1e-30))


def loop_ratio(F, x0, gen):
    r = RADIUS * float(x0.norm())
    th = torch.linspace(0, 2 * math.pi, N_PTS + 1, dtype=x0.dtype)[:-1]
    worst = 0.0
    for _ in range(N_LOOPS):
        e1 = torch.randn(x0.shape, generator=gen, dtype=x0.dtype); e1 /= e1.norm()
        e2 = torch.randn(x0.shape, generator=gen, dtype=x0.dtype); e2 -= (e2 @ e1) * e1; e2 /= e2.norm()
        work = absw = 0.0
        for c, s in zip(torch.cos(th), torch.sin(th)):
            x = (x0 + r * (c * e1 + s * e2)).requires_grad_(True)
            dx = r * (-s * e1 + c * e2) * (2 * math.pi / N_PTS)
            with torch.enable_grad():
                f = F(x).detach()
            work += float(f @ dx); absw += abs(float(f @ dx))
        worst = max(worst, abs(work) / (absw + 1e-30))
    return worst


def router_shares(model, val):
    """Exact router term of V_phi, per layer, at many token positions."""
    L = model.cfg.L
    rng = np.random.default_rng(20261007)
    s0 = int(rng.integers(0, len(val) - T_SEQ - 1))
    x = torch.from_numpy(val[s0:s0 + T_SEQ].astype(np.int64))[None]
    model.eval()
    caps, _ = capture(model, x)
    has_pm = getattr(model, 'pm_mu', None) is not None
    sh = model.score_head
    orig_fwd = sh.forward
    out = {}
    for l in range(L):
        h = caps[l].clone().requires_grad_(True)
        with torch.no_grad():
            xis = model.xi_module(h)

        def forces(detach_query):
            if detach_query:
                sh.forward = lambda h_q, h_s: orig_fwd(h_q.detach(), h_s)
            if has_pm:
                model.poisson_mode_force = lambda h_, li_: torch.zeros_like(h_)
            try:
                with torch.enable_grad():
                    ft, fp = model._layer_forces(h, xis, l, split=True)
            finally:
                sh.__dict__.pop('forward', None)
                model.__dict__.pop('poisson_mode_force', None)
            return ft.detach()[0], fp.detach()[0]
        ft, fp = forces(False)
        _, fp_det = forces(True)
        pm = (type(model).poisson_mode_force(model, h, l).detach()[0] if has_pm else torch.zeros_like(fp))
        dr = (fp - fp_det).norm(dim=-1)                       # router term, per token
        tot = (ft + fp + pm).norm(dim=-1)
        # 2026-10-08: report both bands. At t < top_k the straight-through
        # mask's k = min(top_k, T-1) differs between a short prefix and the
        # full sequence, and the router term is not negligible there.
        tt = torch.arange(T_SEQ)
        out[l] = {}
        for band, keep in (('t>=16', tt >= 16), ('3<=t<16', (tt >= 3) & (tt < 16))):
            r_phi = (dr / (fp.norm(dim=-1) + 1e-30))[keep]
            r_tot = (dr / (tot + 1e-30))[keep]
            out[l][band] = dict(router_over_vphi_median=float(r_phi.median()), router_over_vphi_p95=float(r_phi.quantile(0.95)),
                                router_over_total_median=float(r_tot.median()), router_over_total_p95=float(r_tot.quantile(0.95)),
                                vphi_norm_median=float(fp.norm(dim=-1)[keep].median()), total_norm_median=float(tot[keep].median()))
    return out


def capture(model, x):
    caps, rcs = {}, []
    # _fock_layer_step may itself be an instance-level wrapper (the aniso
    # depth routing installs one), so restore by assignment, never by pop.
    orig_step = model._fock_layer_step

    def spy(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0, **kw):   # Cell 6b-9's signature
        caps.setdefault(layer_idx, h.detach().clone())
        return orig_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx, **kw)
    hook = None
    if getattr(model, 'reverse_ch', None) is not None:
        hook = model.reverse_ch.register_forward_hook(
            lambda m, inp, out: rcs.append(tuple(i.detach().clone() for i in inp)))
    model._fock_layer_step = spy
    try:
        with torch.enable_grad():
            model(x)
    finally:
        model._fock_layer_step = orig_step
        if hook is not None:
            hook.remove()
    return caps, (rcs if rcs else None)


def run(model, val):
    L = model.cfg.L
    rng = np.random.default_rng(20261007)
    s0 = int(rng.integers(0, len(val) - T_SEQ - 1))
    x = torch.from_numpy(val[s0:s0 + T_SEQ].astype(np.int64))[None]
    model.eval()
    caps, rcs = capture(model, x)
    rc_by_layer = {l: rcs[l] for l in range(L)} if rcs else {l: None for l in range(L)}

    # values in train mode (Gumbel off) must equal eval mode at the captured states
    gum = model.cfg.gumbel_noise
    model.cfg.gumbel_noise = False
    ev = {}
    for l in range(L):
        h = caps[l].clone().requires_grad_(True)       # the force differentiates w.r.t. it
        with torch.enable_grad():
            ev[l] = tuple(f.detach() for f in model._layer_forces(h, model.xi_module(h), l, split=True))
    model.train()
    check = 0.0
    for l in range(L):
        h = caps[l].clone().requires_grad_(True)
        with torch.enable_grad():
            tr = tuple(f.detach() for f in model._layer_forces(h, model.xi_module(h), l, split=True))
        check = max(check, float((tr[0] - ev[l][0]).abs().max()), float((tr[1] - ev[l][1]).abs().max()))
    res = {'eval_train_value_check': check, 'layers': {}}
    if check != 0.0:
        print(f'   WARNING: train-mode force values differ from eval by {check:.3e}')

    gen = torch.Generator().manual_seed(7)
    try:
        for l in range(L):
            h = caps[l].detach()
            with torch.no_grad():
                xis = model.xi_module(h)            # context held fixed
            rows = {}
            for t in POSITIONS:
                for name, F in term_functions(model, h, xis, l, t, rc_by_layer[l]).items():
                    x0 = (rc_by_layer[l][0] if name == 'RC' else h)[0, t].detach()
                    a = asymmetry(F, x0, gen)
                    fd = fd_asymmetry(F, x0, gen)
                    lp = loop_ratio(F, x0, gen)
                    with torch.enable_grad():
                        fn = float(F(x0.clone().requires_grad_(True)).detach().norm())
                    rows.setdefault(name, []).append((a, lp, fn, fd))
            res['layers'][l] = {n: dict(asym_worst=max(r[0] for r in v), asym_median=float(np.median([r[0] for r in v])),
                                        fd_worst=max(r[3] for r in v), fd_median=float(np.median([r[3] for r in v])),
                                        loop_worst=max(r[1] for r in v), loop_median=float(np.median([r[1] for r in v])),
                                        force_norm_median=float(np.median([r[2] for r in v])))
                                for n, v in rows.items()}
    finally:
        model.cfg.gumbel_noise = gum
        model.eval()
    return res


def report(res, label):
    out = [f'CONSERVATIVITY TEST on trained weights: {label}',
           f'   train-mode vs eval force values at the captured states: max |d| = {res["eval_train_value_check"]:.3e}'
           + ('  (identical)' if res['eval_train_value_check'] == 0 else ''),
           f'   per token at fixed context; {len(POSITIONS)} tokens per layer; worst token shown (median in brackets)',
           f'   verdict from asym and loop; fd_asym is descriptive (float32 floor = the V_theta row)',
           f'   {"layer":<6}{"term":<9}{"asym (autograd)":>21}{"fd_asym":>21}{"loop":>21}{"|F|":>10}   verdict']
    terms_all = {}
    for l, terms in sorted(res['layers'].items()):
        for n, r in terms.items():
            v = verdict(r['asym_worst'], r['loop_worst'], r['fd_worst'])
            terms_all.setdefault(n, []).append(v)
            out.append(f'   {l:<6}{n:<9}{r["asym_worst"]:>10.2e} ({r["asym_median"]:.1e}){r["fd_worst"]:>10.2e} ({r["fd_median"]:.1e})'
                       f'{r["loop_worst"]:>10.2e} ({r["loop_median"]:.1e})'
                       f'{r["force_norm_median"]:>10.3g}   {v}')
    out.append('   SUMMARY: ' + '; '.join(f'{n} {"conservative" if all(v == "conservative" for v in vs) else "NOT conservative" if any(v.startswith("NOT") for v in vs) else "unclear"}'
                                         for n, vs in terms_all.items()))
    return '\n'.join(out), terms_all


if __name__ == '__main__':
    OUT = Path(sys.argv[1]); FOLDER = sys.argv[2]
    RC, VP, XI = sys.argv[3] == 'True', sys.argv[4], sys.argv[5]
    sys.path.insert(0, str(Path(__file__).parent))
    import gradcheck_vphi_xi_paths as G
    from verify_vphi_xi_grad_path import build
    ROUTER = 'router' in sys.argv[6:]
    for e in [a for a in sys.argv[6:] if a != 'router']:
        if re.fullmatch(r'pm\d+', e):
            subs = (("POISSON_MODES        = 0", f"POISSON_MODES        = {int(e[2:])}"),)
        elif re.fullmatch(r'pmclip[\d.p]+', e):
            subs = (("POISSON_MODE_CLIP    = 0.3", f"POISSON_MODE_CLIP    = {float(e[6:].replace('p', '.'))}"),)
        else:
            raise SystemExit(f'unknown option {e!r}')
        for old, new in subs:
            assert G.cells['Cell 0:'].count(old) == 1, old
            G.cells['Cell 0:'] = G.cells['Cell 0:'].replace(old, new)
    torch.manual_seed(0)
    model, tag, _ = build(RC, FOLDER, VP, XI)
    label = tag[tag.find('cgqk'):]
    if ROUTER:
        rs = router_shares(model, np.load(G.VAL))
        lines = [f'V_phi ROUTER TERM (exact: score-head query detached vs as trained): {label}',
                 f'   {T_SEQ - 16} token positions per layer; share = |router term| / |force|']
        for l, bands in sorted(rs.items()):
            for band, r in bands.items():
                lines.append(f'   layer {l} {band:>8}: of the V_phi force median {100*r["router_over_vphi_median"]:.2f}% (p95 {100*r["router_over_vphi_p95"]:.2f}%);  '
                             f'of the total conservative force median {100*r["router_over_total_median"]:.3f}% (p95 {100*r["router_over_total_p95"]:.3f}%);  '
                             f'|V_phi| {r["vphi_norm_median"]:.3g}, |total| {r["total_norm_median"]:.3g}')
        print('\n'.join(lines))
        OUT.mkdir(parents=True, exist_ok=True)
        name = re.sub(r'[^A-Za-z0-9]+', '_', label)[:80]
        (OUT / f'router_{name}.json').write_text(json.dumps(rs, indent=1))
        (OUT / f'router_{name}_output.txt').write_text('\n'.join(lines) + '\n')
        raise SystemExit(0)
    res = run(model, np.load(G.VAL))
    txt, _ = report(res, label)
    rs = router_shares(model, np.load(G.VAL))
    res['router'] = rs
    txt += '\n   V_phi ROUTER TERM (exact), share of the total conservative force on the token:'
    for l, bands in sorted(rs.items()):
        for band, r in bands.items():
            txt += (f'\n     layer {l} {band:>8}: median {100*r["router_over_total_median"]:.3f}%  p95 {100*r["router_over_total_p95"]:.3f}%'
                    f'   (of V_phi: median {100*r["router_over_vphi_median"]:.2f}%, p95 {100*r["router_over_vphi_p95"]:.2f}%)')
    print(txt)
    OUT.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[^A-Za-z0-9]+', '_', label)[:80]
    (OUT / f'conservativity_{name}.json').write_text(json.dumps(res, indent=1, default=float))
    (OUT / f'conservativity_{name}_output.txt').write_text(txt + '\n')
    print(f'\nwrote {OUT}/conservativity_{name}.json and _output.txt')
