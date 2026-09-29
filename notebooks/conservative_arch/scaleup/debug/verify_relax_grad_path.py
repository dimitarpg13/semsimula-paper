import sys, io, contextlib, json
from pathlib import Path
import numpy as np, torch
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open(Path(__file__).with_name('gradcheck_exchange_paths.py')).read().split("if __name__ == '__main__':")[0])

def build2(mech, suffix, gpath):
    g = {'__name__': '__main__'}
    c0 = cells['Cell 0:'].replace("LADDER_MECHANISM = 'none'", f"LADDER_MECHANISM = {mech!r}")
    assert c0.count("RELAX_GRAD_PATH        = 'default'") == 1
    c0 = c0.replace("RELAX_GRAD_PATH        = 'default'", f"RELAX_GRAD_PATH        = {gpath!r}")
    c5b = [''.join(c['source']) for c in json.load(open(NB))['cells'] if ''.join(c['source']).startswith('# == Cell 5b')][0]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(strip(c0), 'Cell0', 'exec'), g)
        exec(compile(strip(cells['Cell 1:']), 'Cell1', 'exec'), g)
        g['CKPT_DIR'] = Path.home() / 'Downloads' / (TAG + suffix) / 'checkpoints'   # the PARENT arm's weights
        g['RESULTS_DIR'] = SCR / (suffix + gpath); g['RESULTS_DIR'].mkdir(parents=True, exist_ok=True)
        g['DATA_DIR'] = SCR / 'data'; g['GDRIVE_ROOT'] = SCR
        exec(compile(strip(cells['Cell 1b:']), 'Cell1b', 'exec'), g)
        g['DATA_DIR'] = SCR / 'data'
        exec(compile(strip(cells['Cell 2:']), 'Cell2', 'exec'), g)
        sys.path.insert(0, str(REPO / 'notebooks/conservative_arch'))
        exec('from data_module import get_batch', g)
        g['val_ids'] = np.load(VAL); g['train_ids'] = g['val_ids']
        exec(compile(strip(cells['Cell 4:']), 'Cell4', 'exec'), g)
        exec(compile(strip(cells['Cell 5:']), 'Cell5', 'exec'), g)
        g['PROBE_MAX_STEPS'] = g.get('PROBE_MAX_STEPS')
        exec(compile(strip(c5b), 'Cell5b', 'exec'), g)
    model = g['model']
    ck = torch.load(g['CKPT_DIR'] / f"{TAG[len('semsimula_'):] .replace('fock_cfc_baoab_owt','fock_cfc_owt')}{suffix}_best.pt", map_location='cpu', weights_only=False)
    r = model.load_state_dict(ck['model_state_dict'], strict=False)
    assert not r.missing_keys and not r.unexpected_keys, r
    model.eval()
    banner = [l for l in out.getvalue().splitlines() if 'GRADIENT-PATH PROBE' in l]
    return model, g['_variant_tag'], banner

val = np.load(VAL); rng = np.random.default_rng(20260929)
starts = rng.integers(0, len(val) - 513, size=2)
x = torch.from_numpy(np.stack([val[s:s + 512] for s in starts]).astype(np.int64))
y = torch.from_numpy(np.stack([val[s + 1:s + 513] for s in starts]).astype(np.int64))

for mech, suffix, gpath in (('attention', 'attn', 'detached'), ('attention_potential', 'attnpot', 'live')):
    m0, tag0, _ = build2(mech, suffix, 'default')
    m1, tag1, banner = build2(mech, suffix, gpath)
    with torch.enable_grad():
        lo0, l0 = m0(x, y); lo1, l1 = m1(x, y)
    dlog = (lo0 - lo1).abs().max().item()
    # the exchange field's own force and its backward path, at layer 1's input
    with torch.enable_grad():
        _, traj = m1._stack_forward(m1._embed(x), x, return_trajectory=True)
        h = traj[1].clone().float().requires_grad_(True)
        lam = m1._relax_gate(1)
        T = h.shape[1]
        if gpath == 'detached':
            F1 = lam * m1.relax_field(h.detach()); F0 = lam * m0.relax_field(h)
        else:
            F1 = lam * m1.relax_field.force_live(h, h, m1._pair_mask_for(T, h.device))
            hh = h.detach().clone().requires_grad_(True)
            U = lam * m0.relax_field.potential(hh, hh.detach(), hh.detach(), m0._pair_mask_for(T, h.device))
            F0, = torch.autograd.grad(U, hh); F0 = -F0
        dF = (F0.detach() - F1.detach()).abs().max().item()
        cot = torch.zeros_like(F1); cot[:, -1] = torch.randn(F1.shape[0], F1.shape[2], generator=torch.Generator().manual_seed(0))
        gh, = torch.autograd.grad((cot * F1).sum(), h, allow_unused=True)
        gsrc = 0.0 if gh is None else gh[:, :-1].norm().item()
    print(f"\n== {mech} + RELAX_GRAD_PATH={gpath!r}")
    print(f"   tag   : ...{tag1[-40:]}   (parent ...{tag0[-30:]})")
    print(f"   5b    : {banner[0].strip() if banner else 'NO BANNER'}")
    print(f"   forward vs parent: max|dlogit| = {dlog:.3e}   loss {l0.item():.6f} vs {l1.item():.6f}")
    print(f"   exchange force vs parent's: max|dF| = {dF:.3e}   (force rms {F1.detach().pow(2).mean().sqrt().item():.4f})")
    print(f"   gradient into earlier tokens through the field: {gsrc:.4e}")
