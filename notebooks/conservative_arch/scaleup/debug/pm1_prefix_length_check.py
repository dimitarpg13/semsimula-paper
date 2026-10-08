"""Why does PM1 fail the prefix-only causality check while the future test is exact?

Written 2026-10-08 after `causality_check_checkpoint.py ... pm64` on the PM1 full
run gave: future perturbation exactly 0, batch independence exactly 0, but
prefix-only scoring max |d logit| 9.8e-2, leak tax +1.7e-3 nats ("CHECK"), where
F3.1, the same model without the modes, gives 3.1e-5.

Hypothesis: a sequence-LENGTH effect, not a leak. Feeding x[:t+1] instead of the
full sequence changes tensor shapes, so float32 reductions run in a different
order; PM1's occupation phi is a (K, T, T) contraction whose length changes
with T, and a model as sensitive as PM1 (Cell 6b-7: Gate 1 +10,600%, Gate 2
+951%) magnifies the rounding.

  1. SAME LENGTH: logits at t with real future tokens vs random future tokens.
     Must be exactly 0 (re-states the future test at several lengths).
  2. LENGTH SWEEP: logits at t from x[:t+1+k], for k = 0 (the prefix) up to the
     full length, with REAL and with RANDOM tokens after t. If the change from
     the full-length logit depends on k but not on whether the appended tokens
     are real or random, it is a length effect.
  3. WHERE: at layer 0 (identical inputs: the embeddings), phi at position t
     computed from h[:, :t+1] vs from the full h. Relative difference.
  4. FIX TEST: phi computed in float64 (patched in for this script only), and
     the prefix-only comparison repeated. If the discrepancy falls to F3.1's
     level, the cause and a remedy are both confirmed.

Usage: python3 pm1_prefix_length_check.py OUT_DIR FOLDER
"""
import math, sys
from pathlib import Path

import numpy as np
import torch

T = 512
POS = (40, 127, 255, 383, 500)
KS = (0, 1, 8, 64)                       # tokens appended after t; plus the full length


def logits_at(model, x, t):
    with torch.enable_grad():            # the forces use autograd even in eval
        lo, _ = model(x)
    return lo.detach()[0, t]


def occupation64(self, h):
    """poisson_mode_occupation in float64: same formula, rounding-robust."""
    B, T_, d = h.shape
    hf = h.double(); mu = self.pm_mu.double()
    k2 = self.pm_log_kappa2.double().exp()
    d2 = ((hf * hf).sum(-1, keepdim=True) + (mu * mu).sum(-1) - 2.0 * hf @ mu.t()).clamp_min(0.0)
    E = torch.exp(-k2 * d2)
    log_lam = torch.nn.functional.logsigmoid(self.pm_logit_lambda.double())
    t = torch.arange(T_, device=h.device)
    lag = (t[:, None] - t[None, :] - 1)
    Lam = torch.where(lag >= 0, torch.exp(lag.clamp_min(0)[None].double() * log_lam[:, None, None]),
                      torch.zeros((), dtype=torch.float64, device=h.device))
    phi = torch.einsum("kts,bsk->btk", Lam, E)
    return E.float(), phi.float()


def prefix_gap(model, x, positions):
    """max |logit(x[:t+1])[t] - logit(x)[t]| and the leak tax, over positions."""
    full = None
    with torch.enable_grad():
        full, _ = model(x)
    full = full.detach()[0]
    dmax, tax = 0.0, []
    for t in positions:
        lp = logits_at(model, x[:, :t + 1], t)
        dmax = max(dmax, float((lp - full[t]).abs().max()))
        y = x[0, t + 1]
        tax.append(float(torch.log_softmax(full[t], -1)[y] - torch.log_softmax(lp, -1)[y]))
    return dmax, float(np.mean(tax))


if __name__ == '__main__':
    OUT = Path(sys.argv[1]); FOLDER = sys.argv[2]
    sys.path.insert(0, str(Path(__file__).parent))
    import gradcheck_vphi_xi_paths as G
    from verify_vphi_xi_grad_path import build
    old = "POISSON_MODES        = 0"
    assert G.cells['Cell 0:'].count(old) == 1
    G.cells['Cell 0:'] = G.cells['Cell 0:'].replace(old, "POISSON_MODES        = 64")
    torch.manual_seed(0)
    model, tag, _ = build(False, FOLDER, 'live', 'live')
    model.eval()
    V = model.cfg.vocab_size
    val = np.load(G.VAL)
    rng = np.random.default_rng(20261008)
    s0 = int(rng.integers(0, len(val) - T - 1))
    x = torch.from_numpy(val[s0:s0 + T].astype(np.int64))[None]
    g = torch.Generator().manual_seed(3)
    out = []
    p = out.append
    p(f'PM1 prefix-only check, decomposed: ...{tag[tag.find("cgqk"):]}')

    p('\n1. SAME LENGTH, real vs random tokens after t (must be exactly 0)')
    for t in POS:
        for L_ in (t + 2, t + 65, T):
            if L_ > T:
                continue
            xr = x[:, :L_].clone()
            xz = xr.clone(); xz[:, t + 1:] = torch.randint(0, V, xz[:, t + 1:].shape, generator=g)
            d = float((logits_at(model, xr, t) - logits_at(model, xz, t)).abs().max())
            p(f'   t={t:>3}  length {L_:>3}: max |d logit| = {d:.3e}')

    p('\n2. LENGTH SWEEP: |logit(length t+1+k)[t] - logit(full)[t]|, real vs random appended tokens')
    with torch.enable_grad():
        full = model(x)[0].detach()[0]
    p(f'   {"t":>4} ' + ''.join(f'{"k="+str(k)+" real":>13}{"rand":>10}' for k in KS))
    for t in POS:
        row = []
        for k in KS:
            L_ = min(t + 1 + k, T)
            xr = x[:, :L_].clone()
            xz = xr.clone()
            if L_ > t + 1:
                xz[:, t + 1:] = torch.randint(0, V, xz[:, t + 1:].shape, generator=g)
            dr = float((logits_at(model, xr, t) - full[t]).abs().max())
            dz = float((logits_at(model, xz, t) - full[t]).abs().max())
            row.append(f'{dr:>13.3e}{dz:>10.3e}')
        p(f'   {t:>4} ' + ''.join(row))

    p('\n3. WHERE: phi at position t on IDENTICAL layer-0 inputs, short vs full sequence')
    with torch.no_grad():
        h0 = model._embed(x)
        _, phi_full = model.poisson_mode_occupation(h0)
        for t in POS:
            _, phi_pre = model.poisson_mode_occupation(h0[:, :t + 1])
            rel = float(((phi_pre[0, t] - phi_full[0, t]).abs() / (phi_full[0, t].abs() + 1e-12)).max())
            p(f'   t={t:>3}: max relative |d phi| = {rel:.3e}')

    p('\n4. FIX TEST: the prefix-only comparison with phi in float32 (as trained) and in float64')
    d32, tax32 = prefix_gap(model, x, POS)
    orig = type(model).poisson_mode_occupation
    type(model).poisson_mode_occupation = occupation64
    try:
        d64, tax64 = prefix_gap(model, x, POS)
    finally:
        type(model).poisson_mode_occupation = orig
    p(f'   float32 phi: max |d logit| {d32:.3e}, leak tax {tax32:+.2e} nats')
    p(f'   float64 phi: max |d logit| {d64:.3e}, leak tax {tax64:+.2e} nats')
    p('   (F3.1, the same model without modes: max |d logit| 3.05e-05, leak tax -7.7e-08)')

    txt = '\n'.join(out)
    print(txt)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pm1_prefix_length_check_output.txt').write_text(txt + '\n')
