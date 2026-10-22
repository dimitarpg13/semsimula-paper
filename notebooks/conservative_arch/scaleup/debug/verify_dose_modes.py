#!/usr/bin/env python3
# Dose-response (MS2) build check, 2026-10-22: python3 verify_dose_modes.py OUT_DIR
"""Does the ladder notebook build and train at K = 16 and K = 256 Poisson modes, second order and FO-b?
Builds each model from the notebook's own cells (no checkpoint), runs a training forward + backward on
2 x 64 validation tokens, and reports the tag, parameter count, loss, gradient norms of the pm_ tensors
and the half-life initialisation. Then checks that an eval forward of the K = 16 FO-b model is finite."""
import sys, torch, time
from pathlib import Path
OUT = Path(sys.argv[1]); sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, '/Users/dimitargueorguiev/git/ml/semsimula-paper/notebooks/conservative_arch/scaleup/debug')
import verify_pm_switch as P
torch.set_num_threads(8)
class Stop(Exception): pass
_load = torch.load
def build_init(repl):
    torch.load = lambda *a, **k: (_ for _ in ()).throw(Stop())
    try:
        P.build(P.F31_F, repl)
    except Stop:
        tb = sys.exc_info()[2]; m = g = None
        while tb:
            fl = tb.tb_frame.f_locals
            if tb.tb_frame.f_code.co_name == 'build' and 'model' in fl: m, g = fl['model'], fl['g']
            tb = tb.tb_next
        return m, g
    finally:
        torch.load = _load
BASE = P.CONFIGS['F3.1'][1] + (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),
                               ("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"))
FOB = (("FIRST_ORDER_OVERDAMPED = False", "FIRST_ORDER_OVERDAMPED = True"),)
lines = []
say = lambda s: (print(s, flush=True), lines.append(s))
for K in (16, 256):
    for order, extra in (('second order', ()), ('FO-b', FOB)):
        t0 = time.time()
        repl = BASE + (("POISSON_MODES        = 0", f"POISSON_MODES        = {K}"),) + extra
        model, g = build_init(repl)
        model.to('cpu'); model.train()
        tag = g['_variant_tag']
        x = torch.from_numpy(g['val_ids'][:2 * 65].astype('int64')).view(2, 65)
        xb, yb = x[:, :64], x[:, 1:]
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            _, loss = model(xb, targets=yb)
        loss.backward()
        pm = {n: (float(p.grad.norm()) if p.grad is not None else None) for n, p in model.named_parameters() if n.startswith('pm_')}
        hl = (-torch.log(torch.tensor(2.0)) / torch.nn.functional.logsigmoid(model.pm_logit_lambda.detach())).numpy()
        n_par = sum(p.numel() for p in model.parameters())
        fin = bool(torch.isfinite(loss)) and all(v is not None and v == v for v in pm.values())
        say(f'K={K:3d} {order:13s} tag ...{tag[tag.find("xilive"):tag.find("L2probe")]}  params {n_par:,}  pm_mu {tuple(model.pm_mu.shape)}  '
            f'loss {float(loss):.4f}  finite {fin}  |grad| pm_depth {pm["pm_depth"]:.2e} pm_mu {pm["pm_mu"]:.2e}  '
            f'half-lives {hl.min():.1f}-{hl.max():.1f} tokens  exact_modes {getattr(model.cfg, "poisson_wells_exact_modes", None)}  '
            f'fob {bool(getattr(model.cfg, "first_order_overdamped", False))}  [{time.time() - t0:.0f}s]')
        if K == 16 and order == 'FO-b':
            model.eval()
            with torch.enable_grad():
                lo = model(xb)[0]
            say(f'      eval forward finite: {bool(torch.isfinite(lo).all())}')
        del model
(OUT / 'dose_check_output.txt').write_text('\n'.join(lines) + '\n')
print('DONE')
