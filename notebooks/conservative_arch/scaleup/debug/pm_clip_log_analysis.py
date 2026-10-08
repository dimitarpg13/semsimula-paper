"""PM1 clip question from the logs: does a large pm_ gradient step hurt?

Protocol SS5.15. The two 8,000-step PM1 probes differ only in the pm_ clip
(0.3 against 1.0) and share the seed and the data order, so at every logged
step both arms trained on the SAME batch: their training losses compare
pairwise, step by step. Under AdamW a constant gradient rescaling cancels, so
if the tight clip helps it must be by limiting how much the occasional large
pm_ gradient counts. The prediction, if that is the mechanism:

  H-clip  In the clip-1.0 arm, the 50 steps that follow a large pm_ gradient
          (pre-clip norm above the 0.3 arm's typical value) are the steps on
          which the clip-1.0 arm loses ground against the 0.3 arm on the same
          batch; after a typical pm_ gradient it does not.

Measured: the paired gap D(t) = loss_1.0(t) - loss_0.3(t) on the logged steps,
its 50-step change dD(t) = D(t+50) - D(t), and the pm_ pre-clip norm g(t) of
the clip-1.0 arm, read from its Cell 6 log (`top[override:pm_]=x`, present on
the steps where pm_ is the largest group; the other steps are left out).

Caveats stated before running: the logs sample one step in 50, the printed
norm has one decimal, the training loss is a single batch, and the clip-1.0
printed log starts at step 1,550. So this is a screening test: a clear signal
supports H-clip, a null does not refute it.

    python3 pm_clip_log_analysis.py        # writes pm_clip_log_analysis_output.txt
"""
import json, math, re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RES = HERE.parent / 'results'
TAG = ('cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64{}'
       '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn')
DIR03, DIR10 = RES / TAG.format(''), RES / TAG.format('_pmclip1')
OUT = HERE / 'pm_clip_log_analysis_output.txt'
lines = []


def say(s=''):
    print(s)
    lines.append(s)


def train_loss(path):
    d = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            if 'train_loss' in r:
                d[r['step']] = (r['train_loss'], r['grad_norm'])
    return d


def pm_norms(path):
    d = {}
    for l in open(path, errors='replace'):
        m = re.match(r'\s*step\s+(\d+)/32500.*top\[override:pm_\]=([\d.]+)', l)
        if m:
            d[int(m.group(1))] = float(m.group(2))
    return d


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_p(x, y, n=20000, seed=0):
    """Two-sided permutation p-value for a difference of means."""
    rng = np.random.default_rng(seed)
    obs = abs(np.mean(x) - np.mean(y))
    pool = np.concatenate([x, y]); k = len(x); c = 0
    for _ in range(n):
        rng.shuffle(pool)
        c += abs(pool[:k].mean() - pool[k:].mean()) >= obs
    return (c + 1) / (n + 1)


L03, L10 = train_loss(DIR03 / 'training_log.jsonl'), train_loss(DIR10 / 'training_log.jsonl')
G10 = pm_norms(DIR10 / 'L2probe_arm_none_pm64_clip1.0_8000steps_output.txt')
G03 = pm_norms(DIR03 / 'L2probe_arm_none_pm64_clip0.3_8000steps_output.txt')
steps = sorted(set(L03) & set(L10))
D = {s: L10[s][0] - L03[s][0] for s in steps}

say('PM1 clip question from the logs (protocol SS5.15)')
say(f'paired logged steps: {len(steps)} ({steps[0]}..{steps[-1]}); clip-1.0 pm_ norms read on {len(G10)} steps')
say('')
say('1. THE PAIRED GAP on the same batches, D = loss(clip 1.0) - loss(clip 0.3), nats')
for lo, hi in ((0, 2000), (2000, 4000), (4000, 6000), (6000, 8001)):
    v = [D[s] for s in steps if lo < s <= hi]
    say(f'   steps {lo:>5}-{hi:<5}: mean {np.mean(v):+.4f}  median {np.median(v):+.4f}  '
        f'share of steps with clip 1.0 behind {100 * np.mean(np.array(v) > 0):.0f}%  (n={len(v)})')
v = np.array([D[s] for s in steps if s > 2000])
say(f'   after step 2,000: mean {v.mean():+.4f} nats (exp: {100 * (math.exp(v.mean()) - 1):+.2f}% in PPL terms), '
    f'behind on {100 * np.mean(v > 0):.0f}% of steps')

say('')
say('2. DOES clip 1.0 LOSE GROUND AFTER A LARGE pm_ GRADIENT?')
pairs = [(G10[s], D[s + 50] - D[s]) for s in steps if s in G10 and s + 50 in D and s > 2000]
g = np.array([p[0] for p in pairs]); dD = np.array([p[1] for p in pairs])
med03 = float(np.median([v for s, v in G03.items() if s > 2000]))
big = g > med03 + 0.15
say(f'   n = {len(pairs)} logged steps after 2,000 with a pm_ norm; clip-0.3 arm median pm_ norm {med03:.2f}')
say(f'   "large" = pm_ norm >= {med03 + 0.2:.1f} (at least 0.2 above that median): {int(big.sum())} steps; typical: {int((~big).sum())}')
say(f'   mean dD over the next 50 steps: after large {dD[big].mean():+.4f}   after typical {dD[~big].mean():+.4f}   '
    f'difference {dD[big].mean() - dD[~big].mean():+.4f} nats')
p = perm_p(dD[big], dD[~big])
say(f'   permutation p (two-sided, 20,000 shuffles): {p:.3f}')
rho = spearman(g, dD)
say(f'   Spearman(pm_ norm, dD) over all {len(pairs)}: {rho:+.3f}')

say('')
say('3. THE SAME TEST WITH THE GLOBAL GRADIENT NORM (both arms log it on every step)')
gg = np.array([L10[s][1] for s in steps if s > 2000 and s + 50 in D])
dg = np.array([D[s + 50] - D[s] for s in steps if s > 2000 and s + 50 in D])
say(f'   Spearman(global grad norm of clip 1.0, dD): {spearman(gg, dg):+.3f}  (n={len(gg)})')

say('')
say('4. ADAM ON pm_depth AT STEP 8,000 (Cell 6b-15 outputs): did the clip enlarge the depth steps?')
def adam_rows(path):
    d = {}
    for l in open(path):
        m = re.match(r'\s*(pm_\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.e+-]+)\s+([\d.]+)\s*$', l)
        if m:
            d[m.group(1)] = (float(m.group(2)), float(m.group(4)))
    return d
A03 = adam_rows(DIR03 / 'Cell-6b-15_PM1_tuning_what_limits_Poisson_modes_clip=0.3_output.txt')
A10 = adam_rows(DIR10 / 'Cell-6b-15_PM1_tuning_what_limits_Poisson_modes_clip=1.0_output.txt')
say(f'   {"tensor":>16} {"c, clip 1.0":>12} {"c, clip 0.3":>12} {"rel step 1.0":>13} {"rel step 0.3":>13} {"0.3 / 1.0":>10}')
for t in ('pm_depth', 'pm_mu', 'pm_log_kappa2', 'pm_logit_lambda'):
    if t in A03 and t in A10:
        say(f'   {t:>16} {A10[t][0]:12.3f} {A03[t][0]:12.3f} {A10[t][1]:13.2e} {A03[t][1]:13.2e} {A03[t][1]/A10[t][1]:10.2f}')
say('   (c = |m_hat| / |sqrt v_hat|, the consistency of the step; rel step = Adam step / |theta|.)')

say('')
say('READING')
supports = (dD[big].mean() - dD[~big].mean() > 0) and p < 0.05
say('   H-clip (single large steps hurt) ' + ('SUPPORTED: after a large pm_ gradient the clip-1.0 arm loses ground on the same batches'
                    if supports else
                    'NOT SUPPORTED at this resolution: no reliable loss of ground after large pm_ gradients. '
                    'With one logged step in 50 and one-decimal norms this does not refute it; the decisive '
                    'test is the depth-only clip arm (pm_depth at 0.3, the rest at 1.0).'))
ratio = A03['pm_depth'][1] / A10['pm_depth'][1]
say('   H-adam (the clip removes norm variance from Adam\'s second moment, so the depth steps')
say('   are larger and steadier): ' + (f'CONSISTENT -- at step 8,000 the 0.3 arm takes {100*(ratio-1):.0f}% larger relative steps '
    f'on pm_depth (consistency {A03["pm_depth"][0]:.3f} against {A10["pm_depth"][0]:.3f}), and the paired gap is continuous, '
    'not tied to individual steps.' if ratio > 1 and A03['pm_depth'][0] > A10['pm_depth'][0] else 'NOT consistent with the step sizes.'))
say('   Its prediction: clip 1.0 with a larger learning rate on pm_depth alone should recover the gap, and a')
say('   depth-only clip at 0.3 (the rest at 1.0) should match the 0.3 arm. Both are untested.')
OUT.write_text('\n'.join(lines) + '\n')
print(f'\nwrote {OUT}')
