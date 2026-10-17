#!/usr/bin/env python3
"""
Can the Fock registers affect a prediction without the reverse channel? (2026-10-16)

The reverse channel is the only operation that reads the registers into the tokens. This script
tests that claim directly, by scrambling every register-side parameter (the register embeddings,
the write port and the destruction gate) and measuring how far the logits move:

  1. the v1 design (model_fock_parf.FockPARFLM), which never had a reverse channel: a small model at
     the smoke-test configuration; the logit change and the gradient reaching the register-side
     parameters must both be exactly 0 if the registers are inert;
  2. the trained G3' (Fock v2.1, prefix-causal registers), with the read scales set to zero and then
     as trained.

    python3 fock_register_read_dependence.py      -> fock_register_read_dependence_output.txt

Why the v1 registers are inert: they are appended after the tokens in the extended state, and the
pair-force mask is strictly lower-triangular (s < t), so a token can feel only earlier rows, which
are tokens. That was already so at the commit of the v1 Dyck-2 runs (6c1d074, 2026-05-10).
"""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PARF = HERE.parents[1] / 'parf'
SYNC = HERE.parent / 'sync'
lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))


def scramble(model, keys, scale=3.0):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if any(k in n for k in keys):
                p.add_(torch.randn_like(p) * (p.detach().std() if p.numel() > 1 else 1.0) * scale)


# ---------------------------------------------------------------- 1. the v1 design
sys.path.insert(0, str(PARF))
from model_fock_parf import FockPARFConfig, FockPARFLM
cfg = FockPARFConfig(vocab_size=257, d=64, max_len=64, L=4, v_hidden=64, v_depth=2, v_phi_d_type=4,
                     v_phi_d_angle=2, v_phi_phi_hidden=8, v_phi_theta_hidden=8, v_phi_mlp_hidden=16,
                     mass_mode='global', top_k=8, score_head_hidden=8, n_registers=16,
                     creation_gate_hidden=32, stack_discipline=True)
torch.manual_seed(0)
net = FockPARFLM(cfg); net.eval()
x = torch.randint(0, cfg.vocab_size, (2, 16))
V1_KEYS = ('register', 'creation', 'destruction')
with torch.enable_grad():
    a, _ = net(x)
scramble(net, V1_KEYS)
with torch.enable_grad():
    b, _ = net(x)
net.train(); net.zero_grad()
_, loss = net(x, targets=torch.randint(0, cfg.vocab_size, (2, 16))); loss.backward()
g = max(float(p.grad.abs().max()) if p.grad is not None else 0.0
        for n, p in net.named_parameters() if any(k in n for k in V1_KEYS))
say('1. v1 design (no reverse channel), smoke-test configuration, M = 16, LIFO on')
say(f'   max |logit change| after scrambling every register-side parameter: {float((b - a).abs().max()):.1e}')
say(f'   max |gradient| reaching any register-side parameter:               {g:.1e}')

# ---------------------------------------------------------------- 2. the trained G3'
sys.path.insert(0, str(SYNC))
from sy_replay import Replayer, CACHE
torch.set_num_threads(1)
R = Replayer('G3p'); m = R.model
x = torch.from_numpy(np.load(CACHE / 'sy_tokens.npz')['x'][:1]).long()
KEYS = ('register_embed', 'creation_gate', 'destruction')
saved = {n: p.detach().clone() for n, p in m.named_parameters() if any(k in n for k in KEYS)}
s0 = m.reverse_channel_scale.detach().clone()


def restore():
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n in saved:
                p.copy_(saved[n])


def logits():
    with torch.enable_grad():
        return m(x)[0].detach()


say()
say("2. trained G3' (Fock v2.1, prefix-causal registers), one validation sequence of 512 tokens")
for label, s in (('read scales set to 0', torch.zeros_like(s0)), ('read as trained', s0)):
    with torch.no_grad():
        m.reverse_channel_scale.copy_(s)
    a = logits(); a2 = logits()
    scramble(m, KEYS); b = logits(); restore()
    say(f'   {label:22s} repeat run {float((a2 - a).abs().max()):.1e};  registers, write port and destruction '
        f'gate scrambled: max |logit change| {float((b - a).abs().max()):.1e}')
act = []
am0 = m._active_mask
def am(s):
    out = am0(s); act.append(round(float(out.float().mean()), 3)); return out
m._active_mask = am
with torch.no_grad():
    m.reverse_channel_scale.zero_()
act.clear(); a = logits(); base_act = list(act)
torch.manual_seed(0)
for k in KEYS:
    act.clear(); scramble(m, (k,)); b = logits(); restore()
    say(f'   read off, only {k:15s} scrambled: max |logit change| {float((b - a).abs().max()):.1e}   '
        f'(active share per call {act}, trained {base_act})')
def am_off(s):                                   # no register active anywhere
    out = am0(s).clone().fill_(False); act.append(0.0); return out
m._active_mask = am_off
act.clear(); b = logits()
say(f'   read off, every register forced inactive: max |logit change| {float((b - a).abs().max()):.1e}   '
    f'(active share per call {act}): the second LayerNorm is skipped')
m._active_mask = am0
with torch.no_grad():
    m.reverse_channel_scale.copy_(s0)
say()
say('Reading. Without the reverse channel the registers cannot affect a prediction. In the v1 design the')
say('effect is exactly zero and no gradient reaches the registers. In G3\' with the read off, register')
say('content never reaches the tokens. One residue is possible and carries no content: the second LayerNorm')
say('sits inside the reverse-channel block, which runs only if some register is active. If no register is')
say('active at a layer (a strongly perturbed destruction gate can do this), that LayerNorm is skipped. It')
say('acts on an already normalised state, so skipping it only rescales each token\'s state slightly (by a')
say('factor set by the LayerNorm epsilon and the state\'s variance): up to about 0.04 in a logit with every')
say('register off at both layers. With the trained gate every register is always active.')
(HERE / 'fock_register_read_dependence_output.txt').write_text('\n'.join(lines) + '\n')
