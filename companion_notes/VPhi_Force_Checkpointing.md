# Checkpointing the $V_\varphi$ pair force

> **Status: built and verified 2026-11-02; used by every $L = 8$ code arm since.** This note distils one
> entry of the running log, [`Depth_Ladder_and_Matched_Baseline_Protocol.md`, entry `p-vphi-force-ckpt`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-vphi-force-ckpt),
> and the later entries that used it. Every number below is quoted from those entries or from the
> verification output. Numbers marked "expected" were scaled from CPU counts and not measured.

## What it is

A memory switch for training, not an experiment arm: the ladder notebook's Cell 0 setting
`VPHI_FORCE_CHECKPOINT` (model config `vphi_force_checkpoint`, default off). It changes memory and time,
not arithmetic: the loss and every gradient are bit-identical with it on or off. It is not in the variant
tag, so it creates no new run folder. Three terms used below:

- **Checkpointing** (gradient checkpointing) means: in the forward pass, do not keep some intermediate
  tensors for the backward pass; recompute them in the backward pass instead. It trades compute for memory.
- **Layer checkpointing** (Cell 0 `LAYER_CHECKPOINT`) does this for a whole layer: the backward recomputes
  the entire layer's forward.
- **Microbatch**: the slice of a batch that goes through one forward and backward pass. Gradients are
  summed over several microbatches before one optimizer step. "$16 \times 2$" means 16 sequences per
  microbatch and 2 microbatches per step.

The switch needs the closed-form pair force (Cell 0 `VPHI_ANALYTIC_FORCE = True`), which computes the
$V_\varphi$ force $f_\varphi = -\nabla_h U_{\text{pair}}$ without autograd. That force is described in
[`Eliminating_the_Autograd_Dependency_in_VPhi.md`](Eliminating_the_Autograd_Dependency_in_VPhi.md).

## The problem it solves

The goal was to train an eight-layer ($L = 8$) model on the code corpus without layer checkpointing,
at microbatch $16 \times 2$, the microbatch of every code arm. Without layer checkpointing, $L = 8$ did
not fit an 80 GB card, even with $K = 2$ wells per head ($K$ is the number of $V_\theta$ wells per head).

The log measured on CPU the bytes one training forward saves for the backward. The build was SO-CODE's
(the two-layer second-order code model), on $2 \times 512$ code tokens, with the closed-form force on, as on
the GPU.

| Quantity (recorded 2026-11-02) | Value |
|---|---|
| Saved per layer, K = 8 | 1.611 GiB |
| Saved per layer, K = 2 | 1.403 GiB |
| Scale to the GPU microbatch (batch 16 / 2, GiB to GB) | × 8.59 |
| Reproduced training peaks, no checkpointing, 16 × 2 | SO-CODE 32.9 GB, L4-CODE 60.5 GB |
| Expected need at L = 8, 16 × 2 | about 116 GB at K = 8, about 100 GB at K = 2 |
| Card limit (A100; the H100 has the same 80 GB) | 85.1 GB |

So the notebook's batch probe would have fallen back to a smaller microbatch ($8 \times 4$). That is not
a neutral change: the update rule `clip_then_sum` clips two parameter groups (called E and P in the log)
per microbatch, so a different microbatch gives a different update.

Where the memory goes, per layer: about three quarters is the analytic pair force's intermediates of
shape (B, T, k, H). These are batch, position, the $k$ selected source tokens per query, and the
$V_\varphi$ heads (the axis names are read from the code). The Fock register path is about 18%, and
$V_\theta$ about 6% at $K = 2$. The cross-entropy and logits add about 2 GB once. Tying the output head
to the embeddings would save only about 0.3 GB (19.3M parameters with gradients and Adam state) and no
activations.

## What it does

With the closed-form force on and gradients enabled, the layer-force routine (`_layer_forces`) calls the
pair force (`_pair_force_analytic`) inside `torch.utils.checkpoint`, non-reentrant. So:

- **Not saved:** the pair force's (B, T, k, H) intermediates. The rest of the layer is saved as usual.
- **Recomputed in the backward:** one pair-force forward per layer, not the whole layer as layer
  checkpointing does. The random-number state is restored for the recompute.
- Under `torch.no_grad` (evaluation) nothing is checkpointed.
- Cell 0 refuses the switch without `VPHI_ANALYTIC_FORCE = True` and prints a banner; Cell 5 passes it
  to the model; the timing cell 6b-17 gained the row `float32, analytic, V_phi ckpt`.

## How it was verified

Script `verify_vphi_force_checkpoint.py`: CPU, trained weights, the notebook's own Cells 0–5b.

- **Switch off:** bit-identical to the repository at commit `d330f1e`. Compared: eval logits, train
  logits, loss and every gradient, on five arms (SO-CODE, XI6-CODE, FOCK-CODE, F3.1, G3′). Each arm was
  run with the notebook defaults and with the speed switches (closed-form force on, layer checkpointing
  off), on $2 \times 256$ OpenWebText validation tokens. Every difference was $0$.
- **Switch on:** 31 of 31 checks pass on three code arms (SO-CODE, XI6-CODE, FOCK-CODE). The tag is
  unchanged and the banner prints. Eval logits, train logits, loss and every gradient are bit-identical
  to the switch off, with layer checkpointing off and with it on (checkpoints nested). The random-number
  state after forward and backward is identical. Each layer's pair force runs once in the forward and
  once more in the backward. The new 6b-17 row runs on CPU (a smoke test at a small batch, not a speed
  measurement).
- **Memory, SO-CODE at $K = 8$, $L = 2$:** saved tensors fall from 3.580 to 1.505 GiB. In an earlier
  prototype at $K = 2$ ($L = 2$ and 4), the per-layer figure fell from 1.403 to 0.367 GiB.

## Memory and speed on the GPU

The 2026-11-02 entry expected about 29 GB at $K = 2$ and about 45 GB at $K = 8$ for $L = 8$ at
$16 \times 2$, and a step time between about 3.8 s (no checkpointing, extrapolated) and layer
checkpointing's time. All rows below are A100 80 GB runs at $16 \times 2$.

| Run (L = 8, code) | Checkpointing | Wells K | s per step | Training peak | Eval peak |
|---|---|---|---|---|---|
| L8-CODE | whole layer | 8 | 5.5 | 28.9 GB | not recorded |
| K2-L8-CODE | pair force only | 2 | 3.5 (3.46 at step 50) | 35.0 GB | 51.6 GB |
| K2-L8-R6-CODE | pair force only | 2, rank 6 | 3.65 | 37.1 GB | 55.4 GB |
| K2-L8-R6-VS-CODE | pair force only | 2, rank 6, N of 8, 12 or 16 | 5.5 | 63.6 GB | 55.4 GB |

The log reads the K2-L8-CODE launch this way: the pair-force checkpoint trains $L = 8$ at 62% of layer
checkpointing's time per step, in about 6 GB more memory. The measured 35.0 GB was above the CPU estimate
of about 29 GB. The comparison mixes $K = 2$ with $K = 8$, so the gain at $K = 8$ is not measured. (The
switch entry quoted L8-CODE at 5.6 s per step while it was running; its final result gives 5.5 s.)

## How it has been used

- **K2-L8-CODE** (launched 2026-11-02, result 2026-11-03): the first use. It paired with L8-CODE, so the
  two runs differ only in $K$. The pair-force checkpoint replaced layer checkpointing with the same
  arithmetic. The batch probe chose $16 \times 2$ (probe peak 36.1 GB), as required. One session, 19.0 h.
- **K2-L8-R6-CODE** (result 2026-11-05) inherits K2-L8-CODE's Cell 0 with the switch on. One session, 19.5 h.
- **K2-L8-R6-VS-CODE** (result 2026-11-08) inherits that Cell 0 again. Each step draws $N$ from
  $\lbrace 8, 12, 16 \rbrace$ integration steps; it peaked at 63.6 GB on the $N = 16$ walks (estimate about 64 GB),
  still without layer checkpointing. Two sessions.

**Verdict:** it did what it was built for. Every $L = 8$ code arm recorded after L8-CODE trained at
$16 \times 2$ without layer checkpointing, below the 80 GB limit.

**Open point.** The entry planned to run Cell 6b-17 on L8-CODE's checkpoint ($K = 8$) to decide
whether the switch beats layer checkpointing there. The log still lists it as "still to run", and no result is recorded.

## Where to look

- Log entries: [`p-vphi-force-ckpt`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-vphi-force-ckpt), [`p-k2-l8-code-setup`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-k2-l8-code-setup), [`p-k2-l8-code-result`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-k2-l8-code-result), [`p-k2-l8-r6-result`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-k2-l8-r6-result), [`p-k2-l8-r6-vs-result`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-k2-l8-r6-vs-result).
- Code: [`notebooks/conservative_arch/parf/model_parf_multixi.py`](../notebooks/conservative_arch/parf/model_parf_multixi.py),
  config field `vphi_force_checkpoint` and its call site in `_layer_forces`.
- Check: [`notebooks/conservative_arch/scaleup/debug/verify_vphi_force_checkpoint.py`](../notebooks/conservative_arch/scaleup/debug/verify_vphi_force_checkpoint.py),
  output [`verify_vphi_force_checkpoint_output.txt`](../notebooks/conservative_arch/scaleup/debug/verify_vphi_force_checkpoint_output.txt).
