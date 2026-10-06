"""DP series: what Doi-Peliti process do the trained registers implement? (protocol SS5.14)

Pre-registered 2026-10-05 before any measurement. Evaluation only, CPU.
For G2 (L=2 Fock, live), the L=4 Fock live arm and G3 (L=2 + exchange field),
each built through the ladder notebook's own Cells 0-5b on its best checkpoint,
over 8 x 512 validation tokens (seed 20261005), recording at every layer,
position and register:

  - the salience that sets that layer's mask (after creation, before
    destruction) and the mask itself            (hook on _active_mask)
  - the destruction gate g                       (hook on destruction_gates)
  - the register content the reverse channel reads, and each active register's
    leave-one-out contribution to the reverse-channel force
                                                 (hook on reverse_ch)

  DP1  active fraction per position, cells below threshold, g, from layer 1 up
  DP2  salience-weighted mean |cos| between active register contents, and the
       share of active pairs with cos > 0.9; earlier positions vs the last
  DP3  Spearman rho(salience, leave-one-out contribution) across active
       registers, averaged over positions and layers
  plus (descriptive, not pre-registered) the share of the initial salience
  still present after the last layer, prod_l lambda (1 - g_l).

Usage: python3 dp_register_statistics.py OUT_DIR
Writes OUT_DIR/dp_register_statistics.json and the figure
companion_notes/figures/doi_peliti/dp_register_diagnostics.png.
"""
import contextlib, io, json, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G
import verify_vphi_xi_grad_path as V

_P = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_'
_S = '_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_'
ARMS = {  # name: (folder, Cell 0 switches)
    'G2 (L=2 Fock, live)': (_P + 'vplive_xilive_L2probe' + _S + 'idt4_lr0p0012_noattn',
                            dict(vp='live', xi='live', mech='none', rg='default', L=2)),
    'L=4 Fock, live': (_P + 'vplive_xilive_L4probe' + _S + 'idt2_lr0p0012_noattn',
                       dict(vp='live', xi='live', mech='none', rg='default', L=4)),
    'G3 (L=2 + exchange field)': (_P + 'rglive_vplive_xilive_L2probe' + _S + 'idt4_lr0p0012_attnpot',
                                  dict(vp='live', xi='live', mech='attention_potential', rg='live', L=2)),
    # RR-A (protocol SS5.18, 2026-10-06): G3', the hardened exchange field
    "G3' (L=2 + hardened field)": (_P + 'rglive_vplive_xilive_rfqk_rfclip0p3_L2probe' + _S + 'idt4_lr0p0012_attnpot',
                                   dict(vp='live', xi='live', mech='attention_potential', rg='live', L=2, rfqk=True)),
}
N_SEQ, T, SEED = 8, 512, 20261005


def build(folder, vp, xi, mech, rg, L, rfqk=False):
    c0 = G.cells['Cell 0:']
    extra = ((("RELAX_ATTN_QK_NORM          = False", "RELAX_ATTN_QK_NORM          = True"),
              ("RELAX_FIELD_CLIP            = None", "RELAX_FIELD_CLIP            = 0.3")) if rfqk else ())
    for old, new in extra + (("VPHI_GRAD_PATH         = 'default'", f"VPHI_GRAD_PATH         = {vp!r}"),
                     ("XI_GRAD_PATH           = 'default'", f"XI_GRAD_PATH           = {xi!r}"),
                     ("LADDER_MECHANISM = 'none'", f"LADDER_MECHANISM = {mech!r}"),
                     ("RELAX_GRAD_PATH        = 'default'", f"RELAX_GRAD_PATH        = {rg!r}"),
                     ("LADDER_L         = 2 ", f"LADDER_L         = {L} ")):
        assert c0.count(old) == 1, old
        c0 = c0.replace(old, new)
    os.chdir(G.REPO / 'notebooks/conservative_arch/scaleup')
    g = {'__name__': '__main__'}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(G.strip(c0), 'Cell0', 'exec'), g)
        exec(compile(G.strip(G.cells['Cell 1:']), 'Cell1', 'exec'), g)
        g['CKPT_DIR'] = Path.home() / 'Downloads' / folder / 'checkpoints'
        g['RESULTS_DIR'] = G.OUT / 'results'; g['RESULTS_DIR'].mkdir(parents=True, exist_ok=True)
        g['DATA_DIR'] = G.OUT / 'data'; g['GDRIVE_ROOT'] = G.OUT
        exec(compile(G.strip(G.cells['Cell 1b:']), 'Cell1b', 'exec'), g)
        g['DATA_DIR'] = G.OUT / 'data'
        exec(compile(G.strip(G.cells['Cell 2:']), 'Cell2', 'exec'), g)
        exec('from data_module import get_batch', g)
        g['val_ids'] = np.load(G.VAL); g['train_ids'] = g['val_ids']
        exec(compile(G.strip(G.cells['Cell 4:']), 'Cell4', 'exec'), g)
        exec(compile(G.strip(G.cells['Cell 5:']), 'Cell5', 'exec'), g)
        g['PROBE_MAX_STEPS'] = g.get('PROBE_MAX_STEPS')
        exec(compile(G.strip(V.c5b), 'Cell5b', 'exec'), g)
    model, tag = g['model'], g['_variant_tag']
    assert ('_' + tag) in folder.replace('semsimula_fock_cfc_baoab_owt', ''), (tag, folder)
    pfx = folder[len('semsimula_'):].replace('fock_cfc_baoab_owt', 'fock_cfc_owt')
    ck = torch.load(g['CKPT_DIR'] / f'{pfx}_best.pt', map_location='cpu', weights_only=False)
    r = model.load_state_dict(ck['model_state_dict'], strict=True)
    # Layer checkpointing recomputes each layer step during the force's autograd
    # pass, which would call every hook twice. It changes memory, not values.
    model.cfg.use_layer_checkpoint = False
    return model, ck['step']


def spearman(a, b):
    ra = a.argsort(-1).argsort(-1).float(); rb = b.argsort(-1).argsort(-1).float()
    ra = ra - ra.mean(-1, keepdim=True); rb = rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp(min=1e-12)


def measure(model, x):
    """One forward on x (1, T); returns per-layer records."""
    cfg = model.cfg
    rec, cur = [], {}
    orig_mask = model._active_mask

    def mask_hook(sal):
        act = orig_mask(sal)
        cur.clear()
        cur['sal'] = sal.detach().clone(); cur['act'] = act.detach().clone()
        rec.append(cur.copy())
        return act

    def rev_hook(mod, args, out):
        h_new, r_rev, active = args
        with torch.no_grad():
            Q = out.detach()
            M = active.shape[-1]
            loo = torch.zeros(active.shape)
            for k in range(M):
                if not active[..., k].any():
                    continue
                a2 = active.clone(); a2[..., k] = False
                Qk = mod.forward(h_new.detach(), r_rev.detach(), a2)
                loo[..., k] = (Q - Qk).norm(dim=-1)
            rec[-1]['loo'] = loo
            rec[-1]['r'] = r_rev.detach()

    def dg_hook(mod, args, out):
        rec[-1]['g'] = out.detach()

    hs = [model.reverse_ch.register_forward_hook(rev_hook)]
    hs += [dg.register_forward_hook(dg_hook) for dg in model.destruction_gates]
    model._active_mask = mask_hook
    try:
        model.eval()
        with torch.enable_grad():
            model(x)
    finally:
        del model._active_mask
        for h in hs:
            h.remove()
    return rec


def summarise(recs, lam, thr):
    """recs: list over sequences of per-layer record lists."""
    L = len(recs[0])
    out = {'layers': []}
    init_share = None
    for l in range(L):
        sal = torch.cat([r[l]['sal'] for r in recs]).reshape(-1, recs[0][l]['sal'].shape[-1])     # (N, M)
        act = torch.cat([r[l]['act'] for r in recs]).reshape(sal.shape)
        g = torch.cat([r[l]['g'] for r in recs]).reshape(sal.shape)
        loo = torch.cat([r[l]['loo'] for r in recs]).reshape(sal.shape) if 'loo' in recs[0][l] else None
        rr = torch.cat([r[l]['r'] for r in recs])                                                # (S, T, M, d)
        rr = rr.reshape(-1, rr.shape[-2], rr.shape[-1])                                          # (N, M, d)
        f = 1 - g * act.float()
        init_share = (lam * f) if init_share is None else init_share * lam * f
        lay = {'layer': l,
               'active_fraction_mean': float(act.float().mean()),
               'active_fraction_min_position': float(act.float().mean(-1).min()),
               'cells_below_threshold': float((sal <= thr).float().mean()),
               'salience_quantiles': [float(q) for q in torch.quantile(sal.flatten(), torch.tensor([.01, .1, .5, .9, .99]))],
               'g_quantiles': [float(q) for q in torch.quantile(g.flatten(), torch.tensor([.01, .1, .5, .9, .99]))],
               'g_mean': float(g.mean())}
        # DP2
        rn = torch.nn.functional.normalize(rr, dim=-1)
        C = rn @ rn.transpose(-1, -2)                                                            # (N, M, M)
        A = act.float(); W = (sal * A).unsqueeze(-1) * (sal * A).unsqueeze(-2)
        off = ~torch.eye(C.shape[-1], dtype=torch.bool)
        pairm = (A.unsqueeze(-1) * A.unsqueeze(-2)).bool() & off
        Tn = recs[0][l]['sal'].shape[1]
        last = torch.zeros(len(recs), Tn, dtype=torch.bool); last[:, -1] = True
        last = last.flatten()
        for name, sel in (('earlier', ~last), ('last', last)):
            Wm = (W * off)[sel]; Cm = C[sel].abs()
            pm = pairm[sel]
            lay[f'dp2_wmean_abs_cos_{name}'] = float((Wm * Cm).sum() / Wm.sum().clamp(min=1e-12))
            lay[f'dp2_dup_share_{name}'] = float(((C[sel] > 0.9) & pm).sum() / pm.sum().clamp(min=1))
        # DP3
        if loo is not None:
            rhos = []
            for i in range(sal.shape[0]):
                m = act[i]
                if m.sum() >= 4:
                    rhos.append(float(spearman(sal[i][m], loo[i][m])))
            lay['dp3_spearman_mean'] = float(np.mean(rhos)); lay['dp3_spearman_median'] = float(np.median(rhos))
            lay['dp3_n_positions'] = len(rhos)
        lay['_sal'] = sal.flatten().numpy(); lay['_g'] = g.flatten().numpy()
        out['layers'].append(lay)
    out['init_share_quantiles'] = [float(q) for q in torch.quantile(init_share.flatten(), torch.tensor([.01, .1, .5, .9, .99]))]
    out['init_share_mean'] = float(init_share.mean())
    return out


if __name__ == '__main__':
    val = np.load(G.VAL)
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(val) - T - 1, size=N_SEQ)
    xs = [torch.from_numpy(val[s:s + T].astype(np.int64))[None] for s in starts]
    results = {}
    sel = [a for a in sys.argv[2:]]                      # optional: run only these arms
    for name, (folder, sw) in ARMS.items():
        if sel and not any(name.startswith(x) for x in sel):
            continue
        model, step = build(folder, **sw)
        cfg = model.cfg
        lam, thr = cfg.register_salience_decay, cfg.register_salience_threshold
        recs = [measure(model, x) for x in xs]
        res = summarise(recs, lam, thr)
        res.update(step=step, L=cfg.L, M=cfg.n_registers, decay=lam, threshold=thr,
                   stack_discipline=getattr(cfg, 'stack_discipline', None))
        results[name] = res
        print(f"\n== {name}: best step {step}, L={cfg.L}, M={cfg.n_registers}, decay {lam}, threshold {thr}, "
              f"LIFO {res['stack_discipline']}")
        for lay in res['layers']:
            print(f"  layer {lay['layer']}: active fraction {lay['active_fraction_mean']:.4f} "
                  f"(min over positions {lay['active_fraction_min_position']:.3f}); "
                  f"below threshold {lay['cells_below_threshold']:.4f}")
            print(f"           salience q01/q10/q50/q90/q99 " + ' / '.join(f'{v:.3f}' for v in lay['salience_quantiles']))
            print(f"           g        q01/q10/q50/q90/q99 " + ' / '.join(f'{v:.3f}' for v in lay['g_quantiles']) +
                  f"   mean {lay['g_mean']:.3f}")
            print(f"           DP2 weighted |cos| earlier {lay['dp2_wmean_abs_cos_earlier']:.3f}  last "
                  f"{lay['dp2_wmean_abs_cos_last']:.3f};  near-duplicate share earlier "
                  f"{lay['dp2_dup_share_earlier']:.4f}  last {lay['dp2_dup_share_last']:.4f}")
            if 'dp3_spearman_mean' in lay:
                print(f"           DP3 Spearman(salience, leave-one-out force) mean {lay['dp3_spearman_mean']:+.3f} "
                      f"median {lay['dp3_spearman_median']:+.3f} over {lay['dp3_n_positions']} positions")
        print(f"  share of the initial salience left after layer {cfg.L - 1}: mean {res['init_share_mean']:.4f}; "
              f"q01/q10/q50/q90/q99 " + ' / '.join(f'{v:.4f}' for v in res['init_share_quantiles']))
        del model

    # figure: one entry per (arm, layer)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 150})
    arm_col = dict(zip(results, ("#3B82F6", "#22C55E", "#EF4444")))
    entries = [(n, lay) for n, res in results.items() for lay in res['layers']]
    labels = [f"{n.split(' (')[0].replace(' Fock, live', '')}\nlayer {lay['layer']}" for n, lay in entries]
    cs = [arm_col[n] for n, _ in entries]
    xp = np.arange(len(entries))
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    bp = ax[0].boxplot([lay['_sal'] for _, lay in entries], positions=xp, whis=(1, 99), showfliers=False,
                       patch_artist=True, widths=0.6)
    for patch, c in zip(bp['boxes'], cs):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax[0].set_yscale('log'); ax[0].axhline(0.005, color='k', lw=1, ls='--')
    ax[0].text(len(xp) - 0.5, 0.0065, 'activity threshold 0.005', ha='right', fontsize=8.5)
    for x0, (_, lay) in zip(xp, entries):
        ax[0].text(x0, 1.25, f"{lay['active_fraction_mean']:.4f}", ha='center', fontsize=7.5, rotation=90)
    ax[0].set_ylim(2e-3, 3); ax[0].set_title('DP1: salience (1-99%), active fraction on top')
    bp = ax[1].boxplot([lay['_g'] for _, lay in entries], positions=xp, whis=(1, 99), showfliers=False,
                       patch_artist=True, widths=0.6)
    for patch, c in zip(bp['boxes'], cs):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for x0, (n, lay) in zip(xp, entries):
        if lay['layer'] == results[n]['L'] - 1:
            ax[1].text(x0, 0.56, 'no\ngradient', ha='center', fontsize=7.5)
    ax[1].set_ylim(-0.03, 1.03); ax[1].set_title('DP1: destruction gate g (1-99%)')
    w = 0.38
    ax[2].bar(xp - w / 2, [lay['dp2_dup_share_earlier'] * 100 for _, lay in entries], w, color=cs, alpha=0.9,
              label='earlier positions')
    ax[2].bar(xp + w / 2, [lay['dp2_dup_share_last'] * 100 for _, lay in entries], w, color=cs, alpha=0.4,
              hatch='//', label='last position')
    ax[2].axhline(1, color='k', lw=0.8, ls=':'); ax[2].axhline(5, color='k', lw=0.8, ls='--')
    ax[2].text(len(xp) - 0.5, 1.1, 'prediction line 1%', ha='right', fontsize=8)
    ax[2].text(len(xp) - 0.5, 5.1, 'decision line 5%', ha='right', fontsize=8)
    ax[2].set_ylabel('% of active pairs with cos > 0.9'); ax[2].set_title('DP2: near-duplicate register content')
    ax[2].legend(fontsize=8, frameon=False, loc='upper left'); ax[2].set_ylim(0, 6)
    ax[3].bar(xp, [lay.get('dp3_spearman_mean', np.nan) for _, lay in entries], 0.6, color=cs, alpha=0.8)
    ax[3].axhline(0, color='k', lw=0.6)
    ax[3].axhline(0.3, color='k', lw=0.8, ls=':'); ax[3].axhline(0.5, color='k', lw=0.8, ls='--')
    ax[3].text(len(xp) - 0.5, 0.31, 'prediction line 0.3', ha='right', fontsize=8)
    ax[3].text(len(xp) - 0.5, 0.51, 'intensity line 0.5', ha='right', fontsize=8)
    ax[3].set_ylim(-0.5, 0.6); ax[3].set_title('DP3: Spearman(salience, own force contribution)')
    for a_ in ax:
        a_.set_xticks(xp, labels, fontsize=7.5)
    figp = G.REPO / 'companion_notes/figures/doi_peliti/dp_register_diagnostics.png'
    fig.savefig(figp, bbox_inches='tight'); print(f"\nwrote {figp}")
    for res in results.values():
        for lay in res['layers']:
            lay.pop('_sal'); lay.pop('_g')
    (G.OUT / 'dp_register_statistics.json').write_text(json.dumps(results, indent=1))
    print(f"wrote {G.OUT / 'dp_register_statistics.json'}")
