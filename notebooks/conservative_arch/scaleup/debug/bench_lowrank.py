# Paste into a Colab cell. Times the ops lowrank_modes actually performs,
# at the shapes the joint arm produces. ~2 min, no training, no model.
#
# Runs on T4 as well as A100. The cuSOLVER gesvdjBatched 32x32 limit is an
# API constraint, not a hardware one, so the batched-vs-loop cliff shows up
# on both -- but a T4 UNDERSTATES it (looping is launch-bound and roughly
# GPU-independent, while the batched path is throughput-bound and slower on
# a T4). Read the RATIOS, not the absolute milliseconds.
import torch, time

assert torch.cuda.is_available(), 'needs a GPU runtime'
DEV = 'cuda'
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
print(f'GPU: {name}   CUDA {torch.version.cuda}   torch {torch.__version__}')
print(f'free {free/2**30:.1f} GiB / {total/2**30:.1f} GiB')

d, P_joint, P_add, q_over = 384, 32, 160, 20

# Scale the token batch to fit. Full size is 16*512; the P=160 tensor alone
# is ~2 GiB there, which is tight on a 16 GiB T4 alongside SVD workspaces.
B_T = 16 * 512
while B_T > 512 and (B_T * d * P_add * 4) * 3 > free * 0.45:
    B_T //= 2
SCALE = (16 * 512) / B_T
print(f'B_T = {B_T:,} tokens' + (f'  (scaled down {SCALE:.0f}x to fit; '
      f'multiply timings by {SCALE:.0f} for full size)' if SCALE > 1 else ''))
print()

def bench(fn, *a, n=3, warmup=1):
    for _ in range(warmup): fn(*a)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): fn(*a)
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000

# Reference matmul: lets you compare ACROSS GPUs. Everything below is also
# reported as a multiple of this, which cancels most of the hardware gap.
ref_a = torch.randn(B_T, d, q_over, device=DEV)
ref_b = torch.randn(B_T, q_over, q_over, device=DEV)
REF = bench(lambda x, y: x @ y, ref_a, ref_b)
print(f'reference batched matmul ({B_T}x{d}x{q_over} @ {q_over}x{q_over}): {REF:.2f} ms')
print()

rows = []
def row(lbl, shape, fn, *a):
    try:
        ms = bench(fn, *a)
        rows.append((lbl, shape, ms))
        print(f'{lbl:<42}{shape:>20}{ms:>10.1f} ms{ms/REF:>9.0f}x ref')
    except Exception as e:
        print(f'{lbl:<42}{shape:>20}{"FAILED":>10}   {str(e)[:40]}')

print(f'{"op":<42}{"shape":>20}{"time":>13}{"vs ref":>13}')
g32   = torch.randn(B_T, d, P_joint, device=DEV)
qb    = torch.randn(B_T, d, q_over,  device=DEV)
sm32  = torch.randn(B_T, q_over, P_joint, device=DEV)
sm160 = torch.randn(B_T, q_over, P_add,   device=DEV)

svd = lambda x: torch.linalg.svd(x, full_matrices=False)
row('QR  (randomised path; x3 per call)', f'{d}x{q_over}', torch.linalg.qr, qb)
row('SVD small  JOINT    (P=32)',   f'{q_over}x{P_joint}', svd, sm32)
row('SVD small  ADDITIVE (P=160)',  f'{q_over}x{P_add}',   svd, sm160)
row('SVD full   JOINT  (max_modes=None)', f'{d}x{P_joint}', svd, g32)

# Time the REAL driver, not a hand-rolled copy -- it was hardened on
# 2026-09-17 after the fp32 eigh version died on real G with
# "_LinAlgError: ... too many repeated eigenvalues". It now symmetrises,
# works in float64 and uses svd rather than eigh, so the earlier 14.1 ms
# figure does NOT carry over and must be re-measured.
import sys, glob, os
_lm = None
try:                                  # already on sys.path (e.g. after Cell 4)
    from cfc_baoab import lowrank_modes as _lm
except ImportError:
    _cands = ['/content/semsimula-paper/notebooks/conservative_arch/parf']
    _cands += sorted(glob.glob('/content/*/notebooks/conservative_arch/parf'))
    _cands += sorted(glob.glob(os.path.expanduser(
        '~/**/notebooks/conservative_arch/parf'), recursive=True))[:3]
    for _c in _cands:
        if os.path.exists(os.path.join(_c, 'cfc_baoab.py')):
            sys.path.insert(0, _c)
            try:
                from cfc_baoab import lowrank_modes as _lm
                print(f'[import] cfc_baoab from {_c}')
                break
            except Exception:
                pass

if _lm is None:
    print(f'{"REAL lowrank_modes -- NOT FOUND":<42}{"":>20}{"skipped":>10}')
    print('    Run this cell AFTER Cell 4 (which puts the repo on sys.path),')
    print('    or set the path by hand. Without this row the benchmark does')
    print('    NOT tell you whether the hardened gram driver is affordable.')
else:
    row('GRAM driver, REAL lowrank_modes (all 32)', f'{d}x{P_joint}',
        lambda x: _lm(x, max_modes=None, driver='gram'), g32)
    row('SVD driver, REAL lowrank_modes (q=16)',    f'{d}x{P_joint}',
        lambda x: _lm(x, max_modes=16, driver='svd'), g32)

del g32, qb, sm32, sm160, ref_a, ref_b
torch.cuda.empty_cache()

print()
print('HOW TO READ THIS')
print('  * The decisive pair is SVD small JOINT (20x32) vs ADDITIVE (20x160).')
print('    cuSOLVER batches <=32x32 and LOOPS above it. A 10x+ gap confirms')
print('    the cliff -- and means the joint arm avoids it for free.')
print('  * Per optimiser step there are 16 calls (8 layers x 2 microbatches),')
print(f'    roughly doubled by checkpoint recompute: multiply by ~32{"" if SCALE==1 else f" x {SCALE:.0f} (batch scaling)"}.')
print('    The randomised path costs 3x the QR row + 1x the SVD-small row.')
print('  * On a T4 these ratios are a LOWER bound on the A100 gap.')
print('  * The GRAM row times the REAL lowrank_modes. Its fp32-eigh ancestor')
print('    raised _LinAlgError on real G at the first training step, so the')
print('    driver was hardened (symmetrise + float64 + ramp + svd). The old')
print('    14.1 ms figure does NOT carry over -- this row is the live number.')
print('  * A random-G benchmark cannot prove the driver survives REAL G.')
print('    The 1,000-step pilot is the actual test; it fails in step 1 if not.')
