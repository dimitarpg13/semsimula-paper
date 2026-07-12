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

## 6. Relation to other companion notes

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
