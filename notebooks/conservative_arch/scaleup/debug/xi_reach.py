"""Where do a code arm's content-routed context channels put their weight? (2026-10-27)

    python3 xi_reach.py SCRATCH_DIR [FOLDER]

For targets at positions 256-511 whose token occurred earlier in the window, per layer and channel: the
softmax weight on the earlier occurrences, against their uniform share; the weight more than 64 tokens back;
and the effective number of positions (exp entropy). 16 of copy_split.py's 512 windows, the best checkpoint of FOLDER (default SO-CODE's).
A second table (2026-10-28): per layer and channel, the median and p90 of the largest weight a query puts on one position,
the median distance back to that position, and how often it is the window's first.
"""
import sys, types, numpy as np, torch
from pathlib import Path
D = str(Path(__file__).resolve().parent)
ARGS = sys.argv[1:]
sys.argv = ['copy_split.py', ARGS[0], 'none']
sys.path.insert(0, D)
src = open(D + '/copy_split.py').read().split("\nif MODE == 'nll':")[0]
cs = types.ModuleType('cs'); cs.__file__ = D + '/copy_split.py'; exec(compile(src, cs.__file__, 'exec'), cs.__dict__)
F = 'semsimula_fock_cfc_baoab_code_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
F = ARGS[1] if len(ARGS) > 1 else F
run, _ = cs.ladder_model(F, None)
import gc
model = [o for o in gc.get_objects() if isinstance(o, torch.nn.Module) and hasattr(o, 'xi_module') and o.xi_module is not None][0]
xm = model.xi_module
cap = []
xm.register_forward_hook(lambda m, i, o: cap.append(i[0].detach()))
x, y = cs.windows()
idx = np.linspace(0, cs.W - 1, 16).astype(int)
from multixi.model_multixi import causal_ema_logits
K, dk = xm.K, xm.content_d_k
acc = {}
for w in idx:
    cap.clear(); run(x[w:w + 1])
    for L, h in enumerate(cap):
        T = h.shape[1]
        base = torch.stack([causal_ema_logits(T, xm.alpha[k], h.dtype, h.device) for k in range(K)])
        q = xm.W_q(h).view(1, T, K, dk).permute(0, 2, 1, 3); kk = xm.W_k(h).view(1, T, K, dk).permute(0, 2, 1, 3)
        wts = torch.softmax(base[None] + (q @ kk.transpose(-1, -2)) * dk ** -0.5, -1)[0].detach().numpy()   # K,T,T
        for k in range(K):                       # peak weight per query, positions 256-511
            pk = acc.setdefault(('peak', L, k), [[], []])
            pk[0].append(wts[k, 256:].max(-1)); pk[1].append(wts[k, 256:].argmax(-1))
        xs, ys = x[w], y[w]
        for t in range(256, T):
            prev = np.nonzero(xs[:t + 1] == ys[t])[0]
            if len(prev) == 0: continue
            d = t - prev[-1] + 1
            b = 'near' if d <= 16 else ('mid' if d <= 64 else 'far')
            for k in range(K):
                m = wts[k, t, prev].sum(); base_share = len(prev) / (t + 1)
                far = wts[k, t, :max(t - 63, 0)].sum()
                ent = -(wts[k, t, :t + 1] * np.log(wts[k, t, :t + 1] + 1e-12)).sum()
                a = acc.setdefault((L, k, b), [0, 0, 0, 0, 0]); a[0] += m; a[1] += base_share; a[2] += far; a[3] += np.exp(ent); a[4] += 1
print('layer chan  dist   n   mass on earlier copies   uniform share   lift   mass >64 back   eff. #positions')
for (L, k, b), a in sorted((key, v) for key, v in acc.items() if key[0] != 'peak'):
    n = a[4]; print(f'{L:5d} {k:4d}  {b:4s} {n:5d}   {a[0]/n:20.3f}   {a[1]/n:13.3f}  {a[0]/a[1]:5.1f}   {a[2]/n:13.3f}   {a[3]/n:10.1f}')
# 2026-10-28 (protocol p-dead-wells): how peaked each channel's weight is, over all queries at 256-511
print('\nlayer chan   median peak weight   p90 peak weight   median distance to the peak   peak at position 0')
for (_, L, k), (tops, ams) in sorted((key, v) for key, v in acc.items() if key[0] == 'peak'):
    top = np.concatenate(tops); am = np.concatenate(ams); dist = np.concatenate([np.arange(256, 256 + len(a_)) - a_ for a_ in ams])
    print(f'{L:5d} {k:4d}   {np.median(top):18.3f}   {np.quantile(top, 0.9):15.3f}   {np.median(dist):27.0f}   {np.mean(am == 0):17.1%}')
