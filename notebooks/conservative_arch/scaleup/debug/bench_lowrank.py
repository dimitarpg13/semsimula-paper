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

def gram_eigh(G):
    M = G.transpose(-1, -2) @ G
    lam, V = torch.linalg.eigh(M)
    return G @ V, lam
row('GRAM+EIGH  JOINT  (proposed fix)', f'{P_joint}x{P_joint}', gram_eigh, g32)

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
print('  * If GRAM+EIGH failed or was slow, re-check on the A100 before')
print('    committing -- this session has already seen cuSOLVER reject small')
print('    batched eigh on one CUDA version but not another.')
