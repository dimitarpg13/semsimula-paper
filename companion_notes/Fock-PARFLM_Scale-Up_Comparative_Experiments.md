# Fock-PARFLM Scale-Up: Comparative Parameter Analysis vs GPT-2

Exact, code-verified parameter counts for the three scale-up tiers used in
the OpenWebText scale-up experiments (`d=384`, `d=768`, `d=1024`), for both
Fock-PARFLM v2.1 (untied embeddings) and a matched GPT-2-style baseline
(tied embeddings, canonical architecture) at the same hidden dimension `d`
and depth `L`.

All figures below were obtained by directly instantiating
`FockMultiXiPARFLM` (`model_fock_parf_multixi.py`) and `MatchedGPT`
(`matched_baseline_model.py`) with the exact configs used in
`train_fock.py` (presets `d384`, `d768`, `d1024`, `gpt2-small`,
`gpt2-medium`) and calling `.num_params()` / summing
`named_parameters()`, rather than estimated from formulas. See §5 for the
reproduction script.

---

## 1. Why "same `d`, same `L`" and not "same total params"

Fock-PARFLM and GPT-2 distribute their parameter budget very differently:
GPT-2's cost is dominated by attention + MLP blocks (`O(L * d^2)`), while
Fock-PARFLM's cost is dominated by the embedding table, the V_theta energy
landscape, and the Fock creation/destruction gates. Trying to hand-pick
`L` for one architecture to exactly match the other's total parameter
count would require arbitrary, non-round layer counts and would obscure
the actual per-component cost structure.

Instead we fix `d` and `L` to be identical between the two architectures
(mirroring standard GPT-2 Small/Medium depths) and simply report and
discuss the resulting parameter gap. This keeps both the representational
depth (`L`) and the residual-stream width (`d`) — the two quantities that
matter most for capacity and compute — matched exactly, at the cost of a
parameter delta of a few percent to ~2x, discussed per-tier below.

---

## 2. Fock-PARFLM v2.1 configurations (untied embeddings)

**Key architectural fact:** Fock-PARFLM here uses **untied** input/output
embeddings (`tie_embeddings=False`). This is a hard requirement, not an
arbitrary choice — see
[Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md](Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md)
§22 and the D0.4 diagnostic in
[Xi_Bottleneck_Diagnosis_Phase5.md](Xi_Bottleneck_Diagnosis_Phase5.md) §8.4–§9.1:
tied embeddings caused a long-tail pathology where the rarest-token
quintile received a mean cross-entropy of 13.16 nats — *worse* than the
uniform-distribution baseline of $\ln V \approx 10.83$ nats. Untying `E`
(input) from `lm_head` (output) fixed this but doubles the
embedding-table cost relative to a tied-embedding architecture — this
"untied tax" is quantified per-tier below.

Common config across all three tiers: `n_registers=32`, `xi_channels=5`
("5long" override), `V_theta` = depth-conditioned Gaussian wells
(5 heads x 8 wells/head = 40 attractors), `use_output_bias=True`
(log-unigram-frequency initialised), `mass_mode="logfreq"`,
`vocab_size=50,257` (GPT-2 BPE).

### 2.1. `d=384`, `L=16` (current running config, reproduces Colab notebook)

| Component | Params | % of total |
|-----------|-------:|-----------:|
| `E` (input embedding, 50,257 x 384) | 19,298,688 | 36.2% |
| `lm_head` (output projection, untied, 384 x 50,257) | 19,298,688 | 36.2% |
| `V_theta` (depth-conditioned Gaussian wells) | 11,873,320 | 22.2% |
| `creation_gate_qkv` | 1,720,352 | 3.2% |
| `destruction_gates` | 395,280 | 0.7% |
| `P` (positional embedding) | 393,216 | 0.7% |
| `reverse_ch` (reverse channel) | 198,145 | 0.4% |
| `V_phi` (structural-competitive) | 100,872 | 0.2% |
| `out_bias` (log-unigram init) | 50,257 | 0.1% |
| `score_head` | 36,929 | 0.1% |
| `register_embed` | 12,288 | <0.1% |
| scalars (`gamma`, `m_bias`, `logfreq_alpha`, ...) | ~40 | ~0.0% |
| **Total** | **53,378,075** | 100% |

Embeddings (`E` + `lm_head` combined): **38,597,376 (72.3% of total)**.
Core Fock dynamics (`V_theta` + gates + `V_phi` + reverse channel +
registers): **14,300,257 (26.8% of total)**.

### 2.2. `d=768`, `L=12` (matches GPT-2 Small depth)

| Component | Params | % of total |
|-----------|-------:|-----------:|
| `V_theta` (depth-conditioned Gaussian wells) | 47,324,200 | 34.5% |
| `E` (input embedding, 50,257 x 768) | 38,597,376 | 28.1% |
| `lm_head` (output projection, untied, 768 x 50,257) | 38,597,376 | 28.1% |
| `creation_gate_qkv` | 10,027,040 | 7.3% |
| `reverse_ch` | 887,809 | 0.6% |
| `P` (positional embedding) | 786,432 | 0.6% |
| `destruction_gates` | 591,372 | 0.4% |
| `V_phi` | 174,600 | 0.1% |
| `score_head` | 73,793 | 0.1% |
| `out_bias` | 50,257 | 0.0% |
| `register_embed` | 24,576 | 0.0% |
| scalars | ~40 | ~0.0% |
| **Total** | **137,134,863** | 100% |

Embeddings combined: **77,194,752 (56.3% of total)**. `V_theta` has
overtaken the embeddings as the single largest component at this width,
since `V_theta` scales faster than linearly in `d` (well centres +
precisions live in `d`-dimensional space per head).

### 2.3. `d=1024`, `L=24` (matches GPT-2 Medium depth)

| Component | Params | % of total |
|-----------|-------:|-----------:|
| `V_theta` (depth-conditioned Gaussian wells) | 84,131,880 | 40.2% |
| `E` (input embedding, 50,257 x 1024) | 51,463,168 | 24.6% |
| `lm_head` (output projection, untied, 1024 x 50,257) | 51,463,168 | 24.6% |
| `creation_gate_qkv` | 17,825,824 | 8.5% |
| `reverse_ch` | 1,576,961 | 0.8% |
| `destruction_gates` | 1,575,960 | 0.8% |
| `P` (positional embedding) | 1,048,576 | 0.5% |
| `V_phi` | 223,752 | 0.1% |
| `score_head` | 98,369 | 0.0% |
| `out_bias` | 50,257 | 0.0% |
| `register_embed` | 32,768 | 0.0% |
| scalars | ~55 | ~0.0% |
| **Total** | **209,490,739** | 100% |

Embeddings combined: **102,926,336 (49.1% of total)**. `V_theta` is now
the clearly dominant component (40.2%), reflecting its super-linear growth
in `d` relative to the embedding tables' linear growth.

---

## 3. Matched GPT-2 baselines (tied embeddings, same `d` and `L`)

`MatchedGPT` (`matched_baseline_model.py`) hardcodes weight tying at the
forward-pass level (`logits = h @ self.E.weight.T`) — there is no untied
mode available for this baseline, matching the canonical GPT-2 design.
Standard pre-LN GPT-2 block: causal self-attention (QKVO, bias=True) +
4x-expansion GELU MLP (bias=True) + 2 LayerNorms per block, plus a final
LayerNorm.

### 3.1. `d=384`, `L=16`, `n_head=6`

| Component | Params | % of total |
|-----------|-------:|-----------:|
| Transformer blocks (x16) | 28,391,424 | 59.0% |
| `E` (embedding, tied with output head) | 19,298,688 | 40.1% |
| `P` (positional embedding) | 393,216 | 0.8% |
| Final LayerNorm | 768 | 0.0% |
| **Total** | **48,084,096** | 100% |

### 3.2. `d=768`, `L=12`, `n_head=12` (GPT-2 Small depth/width)

| Component | Params | % of total |
|-----------|-------:|-----------:|
| Transformer blocks (x12) | 85,054,464 | 68.3% |
| `E` (embedding, tied) | 38,597,376 | 31.0% |
| `P` (positional embedding) | 786,432 | 0.6% |
| Final LayerNorm | 1,536 | 0.0% |
| **Total** | **124,439,808** | 100% |

### 3.3. `d=1024`, `L=24`, `n_head=16` (GPT-2 Medium depth/width)

| Component | Params | % of total |
|-----------|-------:|-----------:|
| Transformer blocks (x24) | 302,309,376 | 85.2% |
| `E` (embedding, tied) | 51,463,168 | 14.5% |
| `P` (positional embedding) | 1,048,576 | 0.3% |
| Final LayerNorm | 2,048 | 0.0% |
| **Total** | **354,823,168** | 100% |

Note how the embedding table's *share* of GPT-2's total shrinks sharply
with scale (40.1% -> 31.0% -> 14.5%) because transformer blocks grow as
`O(L * d^2)` while the (tied, single) embedding table grows only as
`O(d)`. Fock-PARFLM's embedding share shrinks much more slowly
(72.3% -> 56.3% -> 49.1%) because it pays the embedding cost *twice*
(untied) and its `V_theta` growth, while super-linear, is still gentler
than `O(L * d^2)`.

---

## 4. Side-by-side comparison and ratios

| Tier | Fock-PARFLM (untied) | GPT-2 (tied) | Ratio (Fock / GPT-2) | Delta |
|------|----------------------:|-------------:|----------------------:|------:|
| `d=384`, `L=16` | 53,378,075 | 48,084,096 | **1.110x** | Fock +5,293,979 (+11.0%) |
| `d=768`, `L=12` | 137,134,863 | 124,439,808 | **1.102x** | Fock +12,695,055 (+10.2%) |
| `d=1024`, `L=24` | 209,490,739 | 354,823,168 | **0.590x** | Fock -145,332,429 (GPT-2 +69.4%, i.e. 1.694x more) |

At `d=384` and `d=768`, Fock-PARFLM is only ~10-11% heavier than the
matched GPT-2 baseline — essentially parameter-parity given the
architectural differences. At `d=1024`, the picture flips: GPT-2's
`O(L * d^2)` block cost overtakes Fock-PARFLM's more moderate growth, and
GPT-2 ends up with **1.69x** as many parameters as Fock-PARFLM for the
same `d` and `L`. This crossover is expected and favourable for
Fock-PARFLM: it means the architecture's core mechanism (V_theta +
Fock gates) scales more parameter-efficiently with width than dense
self-attention + MLP blocks do.

### 4.1. Isolating the "untied-embedding tax"

Since `MatchedGPT` cannot use untied embeddings (see §3), it is also
useful to ask: how would Fock-PARFLM compare if we back out the *extra*
cost of its untied output head, to isolate the core-architecture
comparison from the embedding-tying choice?

| Tier | Fock total | Fock minus `lm_head` (tied-equivalent) | GPT-2 total | Ratio (tied-equiv Fock / GPT-2) |
|------|-----------:|----------------------------------------:|------------:|----------------------------------:|
| `d=384`, `L=16` | 53,378,075 | 34,079,387 | 48,084,096 | **0.709x** (GPT-2 has 1.41x more) |
| `d=768`, `L=12` | 137,134,863 | 98,537,487 | 124,439,808 | **0.792x** (GPT-2 has 1.26x more) |
| `d=1024`, `L=24` | 209,490,739 | 158,027,571 | 354,823,168 | **0.445x** (GPT-2 has 2.25x more) |

This confirms the underlying trend seen in the earlier 31.5M-parameter
`d=384` study
([Fock-PARFLM_vs_GPT-2_on_OpenWebText_Next_Steps.md](Fock-PARFLM_vs_GPT-2_on_OpenWebText_Next_Steps.md)):
Fock-PARFLM's *core dynamical mechanism* is substantially leaner than
GPT-2's attention+MLP stack at every scale tested (30-55% fewer core
parameters), and the untied-embedding requirement is what brings the
*total* parameter counts back up to near-parity (or, at `d=384`/`d=768`,
slightly above parity). The untied tax itself is constant in absolute
terms at a given `d` (exactly one extra `d x vocab_size` matrix — 19.3M,
38.6M, 51.5M at `d=384/768/1024` respectively) but shrinks as a
*fraction* of the growing total as `d` increases.

---

## 5. Reproduction

The exact figures above were produced by instantiating both architectures
with `train_fock.py`'s presets and calling `.num_params()` /
`named_parameters()`. Example for the Fock-PARFLM side (run from
`notebooks/conservative_arch/scaleup/`):

```python
import tempfile, numpy as np
import train_fock as T

cfg = T.TrainConfig()
for k, v in T.PRESETS["d768"].items():   # or "d384", "d1024"
    setattr(cfg, k, v)

T._resolve_paths(cfg, __import__("pathlib").Path(".").resolve())

logfreq = np.random.rand(cfg.vocab_size).astype(np.float32)
tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
np.save(tmp.name, logfreq)

model, model_cfg, _ = T.build_fock_model(cfg, "cpu", tmp.name)
print(model.num_params())

# Per-component breakdown:
prefixes = {}
for n, p in model.named_parameters():
    top = n.split(".")[0]
    prefixes[top] = prefixes.get(top, 0) + p.numel()
for k, v in sorted(prefixes.items(), key=lambda kv: -kv[1]):
    print(f"{k:26s} {v:>12,}")
```

And for the GPT-2 baseline (run from `notebooks/conservative_arch/`):

```python
from matched_baseline_model import MatchedGPT, MatchedConfig

cfg = MatchedConfig(vocab_size=50257, d=768, n_layer=12, n_head=12,
                    mlp_mult=4, tie_embeddings=True, max_len=1024)
model = MatchedGPT(cfg)
print(model.num_params())
```

Note the `matched_baseline_model.py`
[caveat](#3-matched-gpt-2-baselines-tied-embeddings-same-d-and-l):
`tie_embeddings` in `MatchedConfig` is not actually read by
`MatchedGPT.forward()` — the output head is *always* tied
(`logits = h @ self.E.weight.T`). This is intentional (canonical GPT-2
behaviour) but worth remembering if extending this baseline.

---

## 6. GPU memory / OOM considerations for `d=768` and `d=1024` scale-up (LambdaLabs H100)

This section documents the VRAM ceiling encountered when moving the
`d=768`/`d=1024` tiers of §2 from a single-GPU Colab pilot to full
OpenWebText scale-up runs on LambdaLabs single-H100 (80 GB) nodes, the
fixes applied to `train_fock.py`, and the levers still on the table.
Unlike the PARF Q9c OOM catalogue in
[CUDA_Memory_Errors_with_SPLM_Designs.md](CUDA_Memory_Errors_with_SPLM_Designs.md),
none of these events involve `torch.autograd.grad(..., create_graph=True)`
or a $(B, T, T, H)$ pair-interaction term — the plain Fock-PARFLM causal-LM
training path in `train_fock.py` has neither. The mechanism here is
simpler: ordinary forward-activation memory growing faster than linearly
in `d` and `L`.

### 6.1. The incident: `d=1024` OOMs at `batch_size=2` on a single 80 GB H100

Running the `d1024` preset (`d=1024, L=24`) at its original
`batch_size=4` (and even at `batch_size=2`) reliably OOM'd on a single
H100 80 GB, while `d=768` was comfortably stable at `batch_size=4` (and
ran fine even at `batch_size=2` when swept). Approximate activation-memory
scaling across the three tiers, normalised to the `d=384` baseline:

| Tier | `d` | `L` | Relative activation cost (approx.) | Est. peak VRAM at `batch_size=2` |
|------|-----|-----|---:|---:|
| `d=384` | 384 | 16 | 1.0x (baseline) | ~5 GB |
| `d=768` | 768 | 12 | ~3.0x | ~19 GB |
| `d=1024` | 1024 | 24 | ~10.7x | ~65 GB |

At `d=1024`, `batch_size=2` (~65 GB of activations alone) leaves no
headroom once model weights, AdamW optimiser state (2x fp32 moments +
fp32 master weights), and the cross-entropy logits gradient
($B \cdot T \cdot V \cdot 4$ bytes, ~unavoidable) are added on top of an
80 GB budget. `batch_size=1` (~33 GB activations) is the largest that
reliably fits.

### 6.2. Fix: updated presets in `train_fock.py`

All six presets that instantiate Fock-PARFLM or the matched GPT-2
baseline at `d=768`/`d=1024` were revised to conservative, empirically
safe `(batch_size, grad_accum)` pairs, holding the effective batch
(`batch_size x grad_accum x world_size`) fixed at 32:

| Preset | `batch_size` (was) | `grad_accum` (was) | `effective_batch` |
|---|---:|---:|---:|
| `d768` | 4 (was 8) | 8 (was 4) | 32 |
| `d1024` | **1** (was 4) | **32** (was 8) | 32 |
| `gpt2-small` | 4 (was 8) | 8 (was 4) | 32 |
| `gpt2-medium` | 2 (was 4) | 16 (was 8) | 32 |
| `sweep-d768` | 4 (was 8) | 8 (was 4) | — |
| `sweep-d1024` | **1** (was 4) | **32** (was 8) | — |

At `batch_size=1, grad_accum=32` for `d=1024`, each optimiser step
requires 32 sequential micro-steps, so wall-clock per step is roughly
4x that of a hypothetical `batch_size=4` run — the price of fitting on
a single 80 GB card.

### 6.3. Safety net already in place: auto batch-size probing

`train_fock.py` (`probe_batch_size()`, defined immediately before the
main training loop) runs a real forward+backward micro-step at the
preset's `batch_size` before training starts and halves it on OOM until
one fits, scaling `grad_accum` up to preserve the requested effective
batch:

```python
orig_bs, orig_accum = cfg.batch_size, cfg.grad_accum
safe_bs = probe_batch_size(model, model_cfg, forward_fn, cfg, train_ids, device)
if safe_bs != orig_bs:
    cfg.grad_accum = max(1, round(orig_accum * orig_bs / safe_bs))
    cfg.batch_size = safe_bs
```

This means the §6.2 preset values are a *starting point tuned to avoid
wasted OOM retries*, not a hard requirement — a smaller/larger GPU than
the reference 80 GB H100 will self-adjust. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
is also exported in `launch_lambdalabs.sh` as defence-in-depth against
allocator fragmentation (same rationale as
[CUDA_Memory_Errors_with_SPLM_Designs.md §3.1](CUDA_Memory_Errors_with_SPLM_Designs.md#31-what-every-splm-family-scaleup-configuration-must-check)).

### 6.4. Why a second GPU does not raise the per-GPU ceiling

`train_fock.py` already supports multi-GPU via plain
`torch.nn.parallel.DistributedDataParallel` (DDP). It is tempting to
read "OOM at `batch_size=2`" as an argument for a 2xH100 node, but plain
DDP is *data*-parallel only: every GPU holds a full replica of the
model, optimiser state, and its own activations. Adding a second H100
does not shrink what any single GPU must hold — `batch_size=1` per GPU
is still the ceiling for `d=1024` with 2 GPUs, just as with 1.

| | 1xH100 (`bs=1`, `accum=32`) | 2xH100 DDP (`bs=1`, `accum=16` each) |
|---|---|---|
| Per-GPU VRAM ceiling | unchanged | unchanged |
| Effective batch | 32 | 32, in **half the wall-clock time** |

So a second GPU (via plain DDP) is a **throughput** lever, not a
**memory** lever. Raising the actual per-GPU ceiling would require
sharding optimiser state / gradients / parameters across GPUs (FSDP or
DeepSpeed ZeRO-2/3) — not yet implemented in `train_fock.py`.

### 6.5. Proposed fix (not yet implemented): bf16 mixed precision

`train_fock.py` currently trains in pure fp32 — no `torch.autocast`,
`bfloat16`, or `GradScaler` anywhere in the file. This is the largest
unclaimed memory/speed lever on an H100:

- Switching the forward/backward to `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`
  (AdamW master weights and optimiser state stay fp32) typically cuts
  activation memory by ~40–50% and gives a real speedup from the H100's
  bf16 tensor cores, on top of whatever DDP throughput gain is available.
  Plausibly enough to move `d=1024` from `batch_size=1` to
  `batch_size=2-3` on a single GPU.
- **bf16, not fp16:** bf16 keeps fp32's 8-bit exponent (same dynamic
  range) and only shrinks the mantissa (7 bits vs fp32's 23), so it does
  not need loss scaling / `GradScaler` and does not carry fp16's
  overflow/underflow failure mode. `torch.autocast`'s default op policy
  already keeps softmax, layer-norm reductions, and the cross-entropy
  loss in fp32; only matmul/conv-heavy ops are downcast.
- **Fock-PARFLM-specific caveat.** This architecture has more
  numerically sharp components than a vanilla transformer — Gumbel-softmax
  creation gates (`gumbel_noise`, `gumbel_tau_init`/`gumbel_tau_min`),
  register salience decay/threshold (`register_salience_decay`,
  `register_salience_threshold`), and `tau_create_init` — plus a
  documented gradient-spike history on the `reverse_channel_scale`
  parameter group (see
  [Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md §20](Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md#20-remediation-per-group-clipping-vs-global-norm-clipping)).
  None of these are in `autocast`'s default fp32-forced op list, so if
  any turn out to be precision-sensitive under bf16, the fix is to wrap
  just those submodules in `torch.autocast(..., enabled=False)` to force
  local fp32 while leaving the large matmuls (`V_theta`, the projection
  stacks) in bf16.

**Recommended validation protocol before rolling out:** run one
gamma-sweep candidate (e.g. `gamma=0.2`, 3000 steps, fixed seed) once in
fp32 and once in bf16, and compare (a) the `val_ppl` trajectory — should
overlay within noise — and (b) per-group gradient norms, especially
`reverse_channel_scale`, watching for any *new* spike pattern introduced
by the reduced mantissa. If both match, roll bf16 out to the full
`d768`/`d1024` presets; if not, force fp32 locally around whichever
submodule diverges.

**Status:** proposed, not yet implemented in `train_fock.py`.

### 6.6. Parallelising the sweep itself across instances (horizontal scaling)

§6.4/§6.5's `--multi-gpu` flag speeds up *one* gamma candidate by
splitting its `grad_accum` across the 2 GPUs on a single node via DDP —
useful, but it does not shrink the *sweep's* total wall-clock much
below `n_candidates x candidate_time`, because DDP already keeps both
GPUs on that node ~100% busy on the one candidate that's currently
running. Measured on a `d=1024`, 2xH100 node: ~22.9 s/step, so a
3000-step candidate takes ~19.1 h, and the default 8-candidate sweep
takes `8 x 19.1h ≈ 153h ≈ 6.4 days` end to end on a single node.

Gamma-sweep candidates are **embarrassingly parallel across each
other** (each is an independent short training run), unlike the
DDP split *within* one candidate. The only way to actually shrink
sweep wall-clock without touching the training code is to add more
GPU-nodes and give each a disjoint subset of the candidate list.
`train_fock.py`'s `--sweep_gammas` CLI flag (comma-separated, overrides
the 8-value `GAMMA_CANDIDATES` default) and `launch_lambdalabs.sh`'s
matching `--sweep-gammas` passthrough exist for exactly this:

```bash
# Instance A (2xH100) — first 4 candidates:
bash launch_lambdalabs.sh sweep-d1024 --gamma-sweep --multi-gpu \
    --sweep-gammas 0.05,0.10,0.15,0.20

# Instance B (2xH100) — last 4 candidates:
bash launch_lambdalabs.sh sweep-d1024 --gamma-sweep --multi-gpu \
    --sweep-gammas 0.25,0.30,0.40,0.50
```

With 2 nodes x 2 GPUs each (4 GPUs total, double the single-node
GPU count), the two 4-candidate sweeps run concurrently and the full
8-candidate sweep finishes in `4 x 19.1h ≈ 76.4h ≈ 3.2 days` instead
of ~6.4 days — a real ~2x wall-clock reduction because it doubles the
total GPU count in use, not because it rearranges the existing 2 GPUs.
Each additional 2xH100 node given a disjoint quarter/eighth of the
candidate list scales this further (e.g. 4 nodes -> 2 candidates/node
-> ~38h).

**Avoiding wasted data-prep time on the second node:** each node
independently streams+tokenises OpenWebText into
`openwebtext_train_{tokens_in_M}M.npy` /
`openwebtext_val_{tokens_in_M}M.npy` under `--data_dir` (default
`~/data`) on first run, which for the 4B-token cache takes real wall
time on top of training. If a first node already has the cache from an
earlier run, copy it directly to the new node instead of re-streaming:

```bash
# from your local machine, relaying through node A's cache:
scp -i ~/.ssh/id_ed25519_lambda \
    ubuntu@<node-A-ip>:~/data/openwebtext_train_4000M.npy \
    ubuntu@<node-A-ip>:~/data/openwebtext_val_2M.npy \
    /tmp/
scp -i ~/.ssh/id_ed25519_lambda /tmp/openwebtext_train_4000M.npy \
    /tmp/openwebtext_val_2M.npy \
    ubuntu@<node-B-ip>:~/data/
```

(or `rsync`/direct node-to-node `scp` if the LambdaLabs instances can
reach each other's public IPs — check with `ssh -A` agent forwarding
so node A can `scp` straight to node B without round-tripping through
the local machine). Run this **before** launching training on node B
so `train_fock.py` finds the cache and skips straight to training.

**Caveat — restarting a node mid-candidate loses that candidate's
progress.** `gamma_sweep()` sets `ckpt_interval = sweep_steps + 1`
(no periodic checkpoints within a candidate; see
[train_fock.py](https://github.com/dimitarpg13/semsimula-paper/blob/main/notebooks/conservative_arch/scaleup/train_fock.py)
`gamma_sweep()`), so killing a running sweep to relaunch with a
`--sweep-gammas` subset discards whatever fraction of the *current*
candidate had completed (the next candidate in the original 8-value
list has not started yet and loses nothing). This is cheap early in a
candidate (e.g. discarding ~40 min out of a ~19h candidate) but should
be done promptly once the decision to split is made, not after a
candidate is mostly finished.

---

## 7. Relation to other companion notes

- [Fock-PARFLM_vs_GPT-2_on_OpenWebText_Next_Steps.md](Fock-PARFLM_vs_GPT-2_on_OpenWebText_Next_Steps.md) —
  the original `d=384`, 31.5M-parameter comparative study (pre-scale-up,
  4 Xi channels, K=8 Gaussian wells, no depth-conditioning). This document
  supersedes those parameter counts for the *current* running
  configuration (5 Xi channels, depth-conditioned V_theta, output bias)
  but the qualitative conclusions about relative parameter efficiency
  carry over.
- [Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md](Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) —
  documents the tied-embedding long-tail pathology that motivates the
  untied-embedding requirement quantified in §2 and §4.1 above.
- [Xi_Bottleneck_Diagnosis_Phase5.md](Xi_Bottleneck_Diagnosis_Phase5.md) §8.4 —
  the D0.4 diagnostic that first identified the tied-embedding failure
  mode.
- [CUDA_Memory_Errors_with_SPLM_Designs.md](CUDA_Memory_Errors_with_SPLM_Designs.md) —
  a forensic OOM catalogue for a different failure mechanism (PARF Q9c's
  $(B, T, T, H)$ pair-interaction V_φ term composed with
  `torch.autograd.grad(create_graph=True)`). §6 above documents a
  simpler, more common mechanism — ordinary forward-activation growth
  with `d` and `L` — that applies to the plain Fock-PARFLM causal-LM
  training path in `train_fock.py`, which has no second-order autograd
  graph in scope.
- [Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md](Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) §20 —
  per-group gradient clipping and the `reverse_channel_scale` spike
  history referenced in §6.5's bf16 precision-sensitivity discussion.
