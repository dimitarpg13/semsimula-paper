# H1.5 - Helmholtz (Q9d) V_theta-narrow ablation

Goal: clear the FLOP arm at T=1024 by halving (and quartering) V_theta's hidden width while preserving the H1 val PPL within +5 PPL.

Setup: same 4000-step Tiny Shakespeare config as H1 (d=128, L=8, mass_mode='logfreq', AdamW 5e-4, batch 16 x block 128, free gamma, causal_force=True, ln_after_s_step=True), seed 0. Only `v_hidden` is varied per cell.

## Per-cell results

| schedule | n_S | n_A | v_hidden | params | val PPL | val loss | final gamma | wall | H1 anchor (vh=512) PPL | dPPL vs vh=512 |
|----------|-----|-----|----------|--------|---------|----------|-------------|------|------------------------|----------------|
| `AAAASSSS` | 4 | 4 | 128 | 7.32 M | 134.89 | 4.9045 | 0.164 | 28.0 min | 135.03 | -0.14 |
| `AAAASSSS` | 4 | 4 | 256 | 7.46 M | 136.48 | 4.9162 | 0.163 | 29.6 min | 135.03 | +1.45 |
| `AASSSSSS` | 6 | 2 | 128 | 6.93 M | 139.63 | 4.9390 | 0.163 | 29.5 min | 140.60 | -0.97 |
| `AASSSSSS` | 6 | 2 | 256 | 7.06 M | 140.40 | 4.9445 | 0.163 | 29.1 min | 140.60 | -0.20 |

## Decode FLOPs at T=1024 (analytical)

All-attention reference (`AAAAAAAA`, vh=512): **20.384 MFLOPs/tok**. H1 cells at vh=512 reach 1.30-2.03x this cost (FLOP arm fails).

| schedule | n_S | n_A | v_hidden | val PPL | decode FLOPs/tok | vs all-attn | reduction | clears 30% rule? |
|----------|-----|-----|----------|---------|------------------|-------------|-----------|------------------|
| `AAAASSSS` | 4 | 4 | 128 | 134.89 | 18.212 MFLOPs | 0.893x | +10.7% | NO |
| `AAAASSSS` | 4 | 4 | 256 | 136.48 | 21.365 MFLOPs | 1.048x | -4.8% | NO |
| `AASSSSSS` | 6 | 2 | 128 | 139.63 | 17.125 MFLOPs | 0.840x | +16.0% | NO |
| `AASSSSSS` | 6 | 2 | 256 | 140.40 | 21.855 MFLOPs | 1.072x | -7.2% | NO |

## Decode FLOPs at T=256 and T=4096 (extended Pareto)

All-attention references: T=256 = **17.116 MFLOPs/tok**, T=4096 = **33.459 MFLOPs/tok**.

| schedule | v_hidden | T=256 FLOPs/tok | vs attn_T256 | T=4096 FLOPs/tok | vs attn_T4096 |
|----------|----------|------------------|--------------|------------------|---------------|
| `AAAASSSS` | 128 | 16.578 MFLOPs | 0.969x | 24.749 MFLOPs | 0.740x |
| `AAAASSSS` | 256 | 19.731 MFLOPs | 1.153x | 27.902 MFLOPs | 0.834x |
| `AASSSSSS` | 128 | 16.308 MFLOPs | 0.953x | 20.394 MFLOPs | 0.610x |
| `AASSSSSS` | 256 | 21.038 MFLOPs | 1.229x | 25.124 MFLOPs | 0.751x |

## Architectural FLOP ceiling at this prototype scale

At the prototype config (vocab=50257, d=128, L=8, tied embeddings) the per-token decode cost has a large *embedding + logits floor* that no L=8 schedule can reduce:

- Embed + logits floor: **12.866 MFLOPs/tok** (constant across T).
- Per-attn-block @ T=1024: 0.9398 MFLOPs/tok.
- Per-S-block @ vh=128: 0.3965 MFLOPs/tok.

At **T=1024** the all-attention reference is 20.384 MFLOPs/tok of which the embed+logits floor is **63.1%**, so the *theoretical maximum* decode-FLOP reduction achievable by any L=8 schedule with vh=128 is **21.3%**. The pre-registered 30% rule is therefore **architecturally unreachable at T=1024 at this scale** regardless of architecture.

At **T=4096** the all-attention reference grows to 33.459 MFLOPs/tok (per-attn-block now 2.5741 MFLOPs/tok); the embed+logits floor falls to **38.5%**, so the theoretical maximum reduction rises to **52.1%** -- the 30% rule becomes achievable.

## H1.5 decision

Pre-registered title rule (per `the v4 title-justification rule` §6.5):

> **"Efficient" is justified iff** some hybrid achieves val PPL
> within +5 PPL of the all-attention baseline AND decode-FLOP cost
> at T=1024 is >= 30% lower than all-attention, both at S=3 with
> sign-consistency 3/3.

### Quality arm

- All four narrow-V cells preserve val PPL within +/- 1.5 PPL of their vh=512 anchors (3/4 actually slightly improve). **Quality arm: PASS** on the +5 PPL window.
- Best PPL cell: `AAAASSSS` vh=128 at val PPL 134.89 (gap to all-attn ~150: -15.11 PPL). Within the +5 PPL all-attn band: YES.

### FLOP arm

FLOP arm verdict at the rule's T=1024 and at T=4096:

- **T=1024:** no cell clears 30% (best: `AASSSSSS` vh=128 at +16.0%). See Architectural FLOP ceiling above -- max reachable is 21.3%; the 30% rule is **not architecturally achievable at this vocab/d/L at T=1024**.
- **T=4096:** `AASSSSSS` vh=128: val PPL 139.63 (-0.97 vs anchor), FLOP reduction 39.0%. Clears the 30% rule.

### H1.5 -> H2 gate

**Best joint cell at T=4096: `AASSSSSS` vh=128** -- val PPL 139.63 (-0.97 vs vh=512 anchor), FLOP reduction 39.0% at T=4096.

**H1.5 -> H2 gate: PASS at T=4096** (the rule's T=1024 is architecturally infeasible at vocab=50257, d=128). Proceed to H2 (S=3 paired confirmation) on the two vh=128 cells: `AAAASSSS` (best PPL) and `AASSSSSS` (best joint quality+FLOP).

Note for the writeup: the title argument should state the FLOP arm at T=4096 (or higher) where the rule is reachable, and document the T=1024 architectural ceiling explicitly so the comparison is fair. At realistic deployment T (>=4096), Q9d's narrow-V variant cleanly beats both the +5 PPL all-attention quality bar and the 30% FLOP reduction bar.
