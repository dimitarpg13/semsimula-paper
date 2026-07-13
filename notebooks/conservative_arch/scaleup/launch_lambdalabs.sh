#!/usr/bin/env bash
#
# Launch FockPARFLM or GPT-2 baseline training on a LambdaLabs instance.
#
# Usage:
#   1. Spin up a LambdaLabs GPU instance (1x or 2x H100/A100).
#   2. SSH in and run:
#        curl -sL https://raw.githubusercontent.com/dimitarpg13/semsimula-paper/main/notebooks/conservative_arch/scaleup/launch_lambdalabs.sh | bash -s -- d768
#      Or clone the repo first and run locally:
#        bash launch_lambdalabs.sh d768
#
#   Available presets: d384, d768, d1024, gpt2-small, gpt2-medium
#
#   For multi-GPU (2x GPUs):
#        bash launch_lambdalabs.sh d768 --multi-gpu
#
#   For Google Drive sync (checkpoints + logs pushed after each save):
#        bash launch_lambdalabs.sh d768 --gdrive
#        bash launch_lambdalabs.sh d768 --multi-gpu --gdrive
#
#   GAMMA SWEEP (run before full training at a new d):
#        bash launch_lambdalabs.sh sweep-d768 --gamma-sweep
#        bash launch_lambdalabs.sh sweep-d1024 --gamma-sweep
#        bash launch_lambdalabs.sh sweep-d768 --gamma-sweep --sweep-steps 5000
#
#   SPLIT GAMMA SWEEP ACROSS MULTIPLE INSTANCES (--sweep-gammas):
#   Run a disjoint subset of the 8 default candidates on each instance
#   so two (or more) machines finish the full sweep in parallel instead
#   of one machine running all 8 sequentially:
#        # instance A:
#        bash launch_lambdalabs.sh sweep-d1024 --gamma-sweep --multi-gpu \
#            --sweep-gammas 0.05,0.10,0.15,0.20
#        # instance B:
#        bash launch_lambdalabs.sh sweep-d1024 --gamma-sweep --multi-gpu \
#            --sweep-gammas 0.25,0.30,0.40,0.50
#   See companion_notes/Fock-PARFLM_Scale-Up_Comparative_Experiments.md §6.6.
#
#   BF16 MIXED PRECISION (recommended for d=1024):
#        bash launch_lambdalabs.sh d1024 --multi-gpu --bf16
#        bash launch_lambdalabs.sh sweep-d1024 --gamma-sweep --multi-gpu --bf16
#
#   MULTI-GPU GAMMA SWEEP (sweep-d1024 only — see companion_notes/
#   Fock-PARFLM_Scale-Up_Comparative_Experiments.md §6.4/§6.5):
#   sweep-d1024's batch_size=1/grad_accum=32 preset makes each 3000-step
#   candidate ~20-30h on 1 GPU; --multi-gpu splits grad_accum across
#   GPUs via DDP (torchrun) for a ~2x wall-clock speedup at the same
#   effective batch. sweep-d768 stays single-GPU-only even with
#   --multi-gpu (its candidates are already fast enough that DDP sync
#   overhead isn't worth it):
#        bash launch_lambdalabs.sh sweep-d1024 --gamma-sweep --multi-gpu
#
#   First time --gdrive: you'll be prompted to authorize rclone.
#   On repeat runs the token is cached in ~/.config/rclone/rclone.conf.
#
set -euo pipefail

# Reduces CUDA allocator fragmentation on long multi-candidate runs
# (gamma sweep trains 8 models sequentially in one process).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PRESET="${1:-d768}"
MULTI_GPU=""
GDRIVE=""
GAMMA_SWEEP=""
SWEEP_STEPS="3000"
SWEEP_GAMMAS=""
BF16=""

shift || true
for arg in "$@"; do
    case "$arg" in
        --multi-gpu)      MULTI_GPU="yes" ;;
        --gdrive)         GDRIVE="yes" ;;
        --gamma-sweep)    GAMMA_SWEEP="yes" ;;
        --bf16)           BF16="yes" ;;
        --sweep-steps)    SWEEP_STEPS_NEXT="yes" ;;
        --sweep-gammas)   SWEEP_GAMMAS_NEXT="yes" ;;
        *)
            if [ "${SWEEP_STEPS_NEXT:-}" = "yes" ]; then
                SWEEP_STEPS="$arg"
                SWEEP_STEPS_NEXT=""
            elif [ "${SWEEP_GAMMAS_NEXT:-}" = "yes" ]; then
                SWEEP_GAMMAS="$arg"
                SWEEP_GAMMAS_NEXT=""
            else
                echo "Unknown arg: $arg"; exit 1
            fi
            ;;
    esac
done

MODE="Training"
[ "$GAMMA_SWEEP" = "yes" ] && MODE="Gamma Sweep"

echo "=================================================="
echo "  FockPARFLM $MODE — LambdaLabs Setup"
echo "  Preset: $PRESET"
[ "$GAMMA_SWEEP" = "yes" ] && echo "  Sweep steps: $SWEEP_STEPS per candidate"
[ -n "$SWEEP_GAMMAS" ] && echo "  Sweep gammas (subset): $SWEEP_GAMMAS"
[ "$BF16" = "yes" ] && echo "  Precision: bf16 mixed"
echo "=================================================="

# ── 1. Clone repo if not present ──
REPO_DIR="$HOME/semsimula-paper"
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "[1/6] Cloning repository..."
    git clone --depth 1 https://github.com/dimitarpg13/semsimula-paper.git "$REPO_DIR"
else
    echo "[1/6] Repository already cloned, pulling latest..."
    cd "$REPO_DIR" && git pull --ff-only || true
fi

# ── 2. Install Python dependencies ──
echo "[2/6] Installing Python dependencies..."
cd "$REPO_DIR"
pip install -q torch numpy transformers tokenizers datasets huggingface_hub pyarrow

# ── 3. Set up output directory ──
SCRIPT_DIR="$REPO_DIR/notebooks/conservative_arch/scaleup"
if [ "$GAMMA_SWEEP" = "yes" ]; then
    OUTPUT_DIR="$HOME/runs/sweep_${PRESET}_$(date +%Y%m%d_%H%M%S)"
else
    OUTPUT_DIR="$HOME/runs/${PRESET}_$(date +%Y%m%d_%H%M%S)"
fi
DATA_DIR="$HOME/data"
mkdir -p "$OUTPUT_DIR/checkpoints" "$DATA_DIR"

echo "[3/6] Output: $OUTPUT_DIR"
echo "       Data:   $DATA_DIR"

# ── 4. GPU info ──
echo "[4/6] GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  No GPU detected"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
echo "  GPUs available: $NUM_GPUS"

# ── 5. Google Drive setup via rclone ──
SYNC_REMOTE=""
if [ "$GDRIVE" = "yes" ]; then
    echo "[5/6] Setting up Google Drive sync via rclone..."
    if ! command -v rclone &>/dev/null; then
        echo "  Installing rclone..."
        curl -sL https://rclone.org/install.sh | sudo bash
    fi
    if ! rclone listremotes 2>/dev/null | grep -q "^gdrive:"; then
        echo ""
        echo "  ┌──────────────────────────────────────────────────┐"
        echo "  │  rclone needs a one-time Google Drive auth.      │"
        echo "  │  Since LambdaLabs has no browser, use:           │"
        echo "  │                                                  │"
        echo "  │  Option A (headless / recommended):              │"
        echo "  │    On your LOCAL machine run:                    │"
        echo "  │      rclone authorize \"drive\"                    │"
        echo "  │    Copy the resulting token, then on this        │"
        echo "  │    server run:                                   │"
        echo "  │      rclone config                               │"
        echo "  │    and paste the token when prompted.            │"
        echo "  │                                                  │"
        echo "  │  Option B (pre-copy config):                     │"
        echo "  │    scp ~/.config/rclone/rclone.conf to this      │"
        echo "  │    server's ~/.config/rclone/rclone.conf         │"
        echo "  │                                                  │"
        echo "  └──────────────────────────────────────────────────┘"
        echo ""
        rclone config
    fi
    SYNC_REMOTE="gdrive:semsimula_runs/${PRESET}_$(date +%Y%m%d_%H%M%S)"
    echo "  Sync target: $SYNC_REMOTE"
else
    echo "[5/6] Google Drive sync: disabled (use --gdrive to enable)"
fi

# ── 6. Launch ──
echo "[6/6] Starting $MODE..."
echo ""

cd "$SCRIPT_DIR"

SYNC_ARG=""
if [ -n "$SYNC_REMOTE" ]; then
    SYNC_ARG="--sync_remote $SYNC_REMOTE"
fi

SWEEP_ARGS=""
if [ "$GAMMA_SWEEP" = "yes" ]; then
    SWEEP_ARGS="--gamma_sweep --sweep_steps $SWEEP_STEPS"
    if [ -n "$SWEEP_GAMMAS" ]; then
        SWEEP_ARGS="$SWEEP_ARGS --sweep_gammas $SWEEP_GAMMAS"
    fi
fi

BF16_ARG=""
if [ "$BF16" = "yes" ]; then
    BF16_ARG="--bf16 true"
fi

# Multi-GPU gamma sweep is only enabled for sweep-d1024: at
# batch_size=1/grad_accum=32 each d=1024 candidate is ~20-30h
# single-GPU, so splitting grad_accum across GPUs via DDP is worth
# the added complexity. sweep-d768 candidates are fast enough
# single-GPU that DDP sync overhead isn't worth it, so it stays
# excluded (same as before) to keep that path unchanged.
USE_MULTI_GPU="no"
EFFBATCH_ARG=""
if [ "$MULTI_GPU" = "yes" ] && [ "$NUM_GPUS" -gt 1 ]; then
    if [ "$GAMMA_SWEEP" != "yes" ]; then
        USE_MULTI_GPU="yes"
    elif [ "$PRESET" = "sweep-d1024" ]; then
        USE_MULTI_GPU="yes"
        # effective_batch = batch_size * grad_accum * world_size, and
        # world_size just went from 1 to $NUM_GPUS. Divide grad_accum
        # by $NUM_GPUS so the effective batch (32) — and thus the LR /
        # gradient-noise regime each gamma candidate is evaluated under
        # — stays identical to the single-GPU sweep-d1024 preset.
        EFFBATCH_ARG="--grad_accum $((32 / NUM_GPUS))"
    fi
fi

if [ "$USE_MULTI_GPU" = "yes" ]; then
    echo ">>> Multi-GPU mode: $NUM_GPUS GPUs via torchrun"
    torchrun --nproc_per_node="$NUM_GPUS" \
        train_fock.py \
        --preset "$PRESET" \
        --output_dir "$OUTPUT_DIR" \
        --data_dir "$DATA_DIR" \
        $SYNC_ARG \
        $SWEEP_ARGS \
        $EFFBATCH_ARG \
        $BF16_ARG
else
    if [ "$GAMMA_SWEEP" = "yes" ]; then
        echo ">>> Gamma sweep mode (single-GPU)"
    else
        echo ">>> Single-GPU mode"
    fi
    python3 train_fock.py \
        --preset "$PRESET" \
        --output_dir "$OUTPUT_DIR" \
        --data_dir "$DATA_DIR" \
        $SYNC_ARG \
        $SWEEP_ARGS \
        $BF16_ARG
fi
