#!/usr/bin/env bash
# =============================================================================
# SWE-Agent Quick Validation Script (8.92.9.152)
#
# One-shot script: cleanup + data gen + training launch.
# Hardcoded for 152 machine (8x RTX 3090), used to quickly verify refactoring.
#
# Usage:
#   bash recipe/swe_agent/example/quick_test.sh            # default: 1 epoch, max_turns=3
#   bash recipe/swe_agent/example/quick_test.sh --epochs 2  # 2 epochs
#   bash recipe/swe_agent/example/quick_test.sh --turns 5   # more turns
#   bash recipe/swe_agent/example/quick_test.sh --full       # 2 epochs, 15 turns (production-like)
# =============================================================================
set -euo pipefail

# ======================== 152 Machine Config ========================
WORK_BASE=/data1/lmy/workspace
MODEL_PATH=/data1/models/Qwen/Qwen3-4B-Instruct-2507
EXPERIMENT_NAME=qwen3-4b-swe-train-v1

# ======================== Quick Test Defaults ========================
TOTAL_EPOCHS=1
MAX_TURNS=3      # 3 turns is enough for simple tasks, keeps rollout fast

# ======================== Parse Args ========================
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
        --turns)  MAX_TURNS="$2";    shift 2 ;;
        --full)   TOTAL_EPOCHS=2; MAX_TURNS=15; shift ;;
        *)        break ;;  # pass remaining args to training script
    esac
done

# ======================== Derived Paths ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$VERL_ROOT/data/swe_agent_test"
CKPT_DIR="$WORK_BASE/checkpoints/$EXPERIMENT_NAME"
LOG_DIR="$WORK_BASE/logs"
LOG_FILE="$LOG_DIR/swe_train_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  SWE-Agent Quick Validation (152)"
echo "============================================"
echo "  Model:       $MODEL_PATH"
echo "  Workspace:   $WORK_BASE"
echo "  Epochs:      $TOTAL_EPOCHS"
echo "  Max turns:   $MAX_TURNS"
echo "  Log file:    $LOG_FILE"
echo "============================================"

# ======================== Step 1: Cleanup ========================
echo ""
echo "[1/4] Cleaning up environment..."

# Kill any lingering training process
pkill -9 -f "verl.trainer.main_ppo" 2>/dev/null && echo "  Killed old training process" || true
sleep 1

# Stop Ray
ray stop --force 2>/dev/null && echo "  Ray stopped" || true

# Stop swerex Docker containers
SWEREX_CONTAINERS=$(docker ps --filter "ancestor=swerex-python:3.11" -q 2>/dev/null)
if [ -n "$SWEREX_CONTAINERS" ]; then
    echo "$SWEREX_CONTAINERS" | xargs docker stop 2>/dev/null
    echo "  Stopped $(echo "$SWEREX_CONTAINERS" | wc -l) swerex containers"
else
    echo "  No swerex containers running"
fi

# Remove old checkpoints (CRITICAL: prevents auto-resume skipping all training)
if [ -d "$CKPT_DIR" ]; then
    rm -rf "$CKPT_DIR"
    echo "  Removed old checkpoints: $CKPT_DIR"
else
    echo "  No old checkpoints found"
fi

# ======================== Step 2: Verify Prerequisites ========================
echo ""
echo "[2/4] Verifying prerequisites..."

FAIL=0
[ -f "$MODEL_PATH/config.json" ]          && echo "  Model:    OK" || { echo "  Model:    MISSING ($MODEL_PATH)"; FAIL=1; }
which sweagent >/dev/null 2>&1            && echo "  sweagent: OK" || { echo "  sweagent: MISSING (pip install sweagent)"; FAIL=1; }
docker images swerex-python:3.11 --format '{{.Repository}}' 2>/dev/null | grep -q swerex && echo "  Docker:   OK" || { echo "  Docker:   swerex-python:3.11 image missing"; FAIL=1; }

# Check GPUs
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -ge 8 ]; then
    echo "  GPUs:     OK ($GPU_COUNT available)"
else
    echo "  GPUs:     WARN (expected 8, got $GPU_COUNT)"
fi

[ $FAIL -eq 1 ] && { echo ""; echo "FATAL: Prerequisites check failed. Aborting."; exit 1; }

# ======================== Step 3: Generate Test Data (if needed) ========================
echo ""
echo "[3/4] Checking test data..."

if [ -f "$DATA_DIR/train.parquet" ] && [ -f "$DATA_DIR/test.parquet" ]; then
    echo "  Test data exists at $DATA_DIR"
else
    echo "  Generating test data..."
    cd "$VERL_ROOT"
    python3.12 recipe/swe_agent/prepare/prepare_data.py \
        --mode simple --train_size 8 --test_size 2 \
        --output_dir data/swe_agent_test
    echo "  Generated 8 train + 2 test samples"
fi

# ======================== Step 4: Launch Training ========================
echo ""
echo "[4/4] Launching training..."
echo "  Log: $LOG_FILE"
echo "  Tail with: tail -f $LOG_FILE"
echo ""

mkdir -p "$LOG_DIR"
cd "$VERL_ROOT"

export WORK_BASE MODEL_PATH

# Launch with reduced turns/epochs for quick validation, pass extra args through
bash recipe/swe_agent/example/run_qwen3_4b_instruct.sh \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_TURNS \
    trainer.total_epochs=$TOTAL_EPOCHS \
    "$@" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# ======================== Step 5: Post-Training Cleanup ========================
echo ""
echo "============================================"
echo "  Training finished (exit code: $EXIT_CODE)"
echo "  Log:        $LOG_FILE"
echo "  Checkpoint: $CKPT_DIR/"
echo "============================================"

# Cleanup leaked Docker containers
docker ps --filter "ancestor=swerex-python:3.11" -q 2>/dev/null | xargs -r docker stop 2>/dev/null
ray stop --force 2>/dev/null

exit $EXIT_CODE
