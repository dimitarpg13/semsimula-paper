"""Shared replay machinery for the SY follow-ups (sy_subspace.py, sy_coref_probe.py --ablate).

The same arms, switches and common-mode references as sy_measure.py (which keeps its own,
already-scored copy). Every script that uses this module checks, before measuring, that the
"on" replay reproduces the trained step bit for bit and that every patch restores.
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'sy_cache'
_argv = list(sys.argv)
sys.argv = [sys.argv[0], str(CACHE / 'harness')]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / 'debug'))   # shared model-building helpers (verify_pm_switch, ...)
import verify_pm_switch as P
sys.argv = _argv

SR2 = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
PMXS = (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),)
_T = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
_E = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
F31 = P.CONFIGS['F3.1'][1]
ARMS = {   # name: (folder, Cell 0 replacements, couplings, common reference adopted by sy_measure)
    'F3.1': (P.F31_F, F31, ['vphi'], 'fcom'),
    'SR2': (_T + 'sr2' + _E, F31 + SR2, ['vphi'], 'dcom'),
    'G2': (P.G2_F, P.CONFIGS['G2'][1], ['vphi', 'fock'], 'dcom'),
    'PM1': (_T + 'pm64' + _E, F31 + P.PM_ON, ['vphi', 'wells'], 'dcom'),
    'PMX': (_T + 'pm64_pmx16_sr2' + _E, F31 + P.PM_ON + SR2 + PMXS, ['vphi', 'wells'], 'dcom'),
    'L4F': (P._P + 'vplive_xilive_L4probe' + P._S.replace('idt4_', 'idt2_') + 'noattn',
            P.CONFIGS['G2'][1] + (("LADDER_L         = 2 ", "LADDER_L         = 4 "),), ['vphi', 'fock'], 'fcom'),
}


def _tokmean(F):
    return F.mean(dim=1, keepdim=True).expand_as(F)


class Replayer:
    """Builds an arm, records every layer step of a forward pass, replays steps under patches.

    Versions: None (as trained); ('off', c); ('fcom', c) coupling c's force replaced by its
    cross-token mean; ('xi', 'own' | 'common') the context xi replaced. The displacement-level
    common reference ('dcom', c) is computed from the 'off' replay by dcom()."""

    def __init__(self, arm):
        self.arm = arm
        folder, repl, self.couplings, self.ref = ARMS[arm]
        self.model, _, _ = P.build(folder, repl)
        m = self.model
        m.to('cpu'); m.eval()
        self.L = m.cfg.L
        self._pair = m._pair_potential
        self._lf = m._layer_forces
        self._pmf = getattr(m, 'poisson_mode_force', None)
        self._proj = m._project
        self._step = m._fock_layer_step
        self._xi = m.xi_module.forward
        self.proj_in = []
        m._project = self._spy_proj
        self.records = {}
        m._fock_layer_step = self._spy_step

    def _spy_proj(self, h):
        self.proj_in.append(h.detach())
        return self._proj(h)

    def _spy_step(self, h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
        out = self._step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
        cl = lambda a: a.detach().clone() if torch.is_tensor(a) else a
        self.records[layer_idx] = {'args': [cl(a) for a in (h, h_prev, r, sal, m_b, gamma, dt)],
                                   'out': out[0].detach().clone()}
        return out

    def forward(self, xb):
        with torch.enable_grad():
            logits, _ = self.model(xb)
        return logits.detach()

    def _zero_pair(self, h_in, layer_idx, xis=None, h_src_live=None):
        return self.model._add_relax_potential(h_in.new_zeros(()), xis, h_in, layer_idx)

    def _patch(self, version):
        m = self.model; saved = {}
        if version is None:
            return saved
        k, c = version
        if k == 'off':
            if c == 'vphi':
                m._pair_potential = self._zero_pair
            elif c == 'wells':
                saved['w'] = m.pm_depth.detach().clone()
                with torch.no_grad(): m.pm_depth.zero_()
            elif c == 'fock':
                saved['f'] = m.reverse_channel_scale.detach().clone()
                with torch.no_grad(): m.reverse_channel_scale.zero_()
        elif k == 'fcom':
            if c == 'vphi':
                lf0, zp = self._lf, self._zero_pair
                def lf(h_in, xis, layer_idx, *, split=False, vtheta_comps=None):
                    ft, fp = lf0(h_in, xis, layer_idx, split=True, vtheta_comps=vtheta_comps)
                    m._pair_potential = zp
                    try:
                        _, fp0 = lf0(h_in, xis, layer_idx, split=True, vtheta_comps=vtheta_comps)
                    finally:
                        del m.__dict__['_pair_potential']
                    fp = fp0 + _tokmean(fp - fp0)
                    return (ft, fp) if split else (fp if ft is None else ft + fp)
                m._layer_forces = lf
            elif c == 'wells':
                pmf = self._pmf
                m.poisson_mode_force = lambda h, layer_idx: _tokmean(pmf(h, layer_idx))
            elif c == 'fock':
                rc = m.reverse_ch; f0 = rc.forward
                rc.forward = lambda *a, **kw: _tokmean(f0(*a, **kw))
                saved['rc'] = True
        elif k == 'xi':
            x0 = self._xi
            if c == 'own':
                def xf(x):
                    B, T, d = x.shape
                    o = x0(x.reshape(B * T, 1, d))
                    return o.reshape(B, T, *o.shape[2:])
            else:
                def xf(x):
                    o = x0(x)
                    return o.mean(dim=1, keepdim=True).expand_as(o)
            m.xi_module.forward = xf
            saved['xi'] = True
        return saved

    def _unpatch(self, saved):
        m = self.model
        for name in ('_pair_potential', '_layer_forces', 'poisson_mode_force'):
            m.__dict__.pop(name, None)
        with torch.no_grad():
            if 'w' in saved: m.pm_depth.copy_(saved['w'])
            if 'f' in saved: m.reverse_channel_scale.copy_(saved['f'])
        if 'rc' in saved: del m.reverse_ch.__dict__['forward']
        if 'xi' in saved: del m.xi_module.__dict__['forward']

    def replay(self, l, version):
        """(pre-LayerNorm state, output state) of layer l's step under `version`, float64 numpy."""
        rec = self.records[l]
        args = [a.clone() if torch.is_tensor(a) else a for a in rec['args']]
        args[0] = args[0].requires_grad_(True)
        self.proj_in.clear()
        saved = self._patch(version)
        try:
            with torch.enable_grad():
                out = self._step(*args, layer_idx=l)
        finally:
            self._unpatch(saved)
        return self.proj_in[-1].double().numpy(), out[0].detach().double().numpy()

    def ln(self, H):
        return self._proj(torch.from_numpy(H).float()).double().numpy()

    def common_ref(self, l, c, on):
        """The coupling's common-mode reference (pre, post) as sy_measure adopted it for this arm."""
        if self.ref == 'fcom' and not (c == 'wells' and getattr(self.model.cfg, 'poisson_wells_exact', False)):
            return self.replay(l, ('fcom', c))
        pre_off, _ = self.replay(l, ('off', c))
        pre = pre_off + (on[0] - pre_off).mean(axis=1, keepdims=True)
        return pre, self.ln(pre)

    def check(self, xb, say):
        """Bit-exact replay and restoration checks; raises on failure."""
        ref = self.forward(xb)
        for l in range(self.L):
            pre, post = self.replay(l, None)
            d = float(np.abs(post - self.records[l]['out'].double().numpy()).max())
            say(f'CHECK  layer {l}: replay "on" vs the forward pass, max |d| {d:.1e}')
            assert d == 0.0
            for c in self.couplings:
                self.common_ref(l, c, (pre, post))
            self.replay(l, ('xi', 'own')); self.replay(l, ('xi', 'common'))
        assert torch.equal(self.forward(xb), ref)
        say('CHECK  every patch restored (logits identical after all replays)')
