# GPT-2 baseline, d=384 / L=8 / 6 heads — two runs, one superseded

Kept together because the older run is the **provenance for the 54.59 that
appears in `paper_v5`, checklist §9 and the productionization plan**, and it
is the only evidence for the correction recorded in checklist §9.

| file | run | batch | tokens | settled |
| --- | --- | --- | ---: | ---: |
| `superseded_batch8_then_32_*` | 2026-09-19 | **8 for steps 0-14,000, then 32** | **360.4M** | 54.67 |
| `matched_batch32_*` | 2026-09-21 | 32 throughout | **532.5M** | **49.81** |

## Why the older run is superseded, not deleted

Its console banner is the **only** artifact recording the real batch. The
`.jsonl` computes its `tokens` field as `step * 16,384` throughout and so
reports 532.5M for a run that saw 360.4M — the mismatch is invisible there.
Deleting the log would remove the evidence for the §9 correction while
leaving the corrected claim standing on assertion.

Consequences of the mismatch, all of which flattered Fock:

- Fock used **1.48x** the tokens to reach 81.58 and still lost.
- GPT-2 crossed Fock's settled endpoint at **14.3%** of Fock's token budget,
  not the 46.7% recorded in §9.2 — that figure was a *step* fraction.

## Result of the matched run

Final **49.76** at step 32,500, settled **49.81** (mean of the last three:
49.87, 49.80, 49.76), monotone to the end with no late reversal.

| | tokens | settled | ratio to matched |
| --- | ---: | ---: | ---: |
| GPT-2 superseded | 360.4M | 54.67 | 1.098 |
| **GPT-2 matched** | 532.5M | **49.81** | 1.000 |
| Fock L=2 + exchange force | 532.5M | 68.33 | **1.372** |
| Fock L=2, conservative only | 532.5M | 75.09 | **1.508** |

The extra 172M tokens are worth **4.86 PPL** to GPT-2. Every Fock-vs-GPT-2
ratio therefore widens: the attention arm goes 1.252 to 1.372 and the
conservative arm 1.373 to 1.508. The 1.49x figure quoted for the L=8 arm was
flattered by the token mismatch.

## Quote these ratios only with the tuning caveat

Matched on data and on evaluation; **not** matched on hyperparameter effort.
GPT-2 runs nanoGPT defaults, known-good across a wide range. The Fock side
inherits its learning rate from the L=8 arm and has never been swept at this
depth, and its exchange-field lambda was chosen on structural argument rather
than evidence. The ratios are therefore an **upper bound on the gap**. See
[`Depth_Ladder_and_Matched_Baseline_Protocol.md`](../../../../../companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md)
SS5.4.

## Do not quote the superseded numbers

The matched run replaces them. Its endpoint is the correct denominator for
every Fock-versus-GPT-2 ratio. The notebook now **raises** on a batch
mismatch rather than printing a warning, so this cannot recur silently.

See [`Depth_Ladder_and_Matched_Baseline_Protocol.md`](../../../../../companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md) §4.
