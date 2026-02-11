#!/usr/bin/env bash
# =============================================================================
# SWE-bench VERL Training - 4-Node (32 GPU)
#
# Thin wrapper: sets 4-node defaults, then delegates to run_swebench.sh.
# Cluster topology: 152 (head) + 150/154/155 (workers), 8 GPUs each.
#
# Prerequisites:
#   1. Start Ray cluster: bash setup_ray_cluster.sh
#   2. Prepare data:      see run_swebench.sh header
#
# Usage:
#   bash run_swebench_4node.sh
#   bash run_swebench_4node.sh trainer.total_epochs=5   # Hydra overrides pass through
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ================= 4-node defaults =================
export NNODES=${NNODES:-4}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-40}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-4b-swebench-4node-v1}

# Multi-node networking
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp96s0f0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp96s0f0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-enp96s0f0}
export MASTER_ADDR=${MASTER_ADDR:-8.92.9.152}
export RAY_ADDRESS=${RAY_ADDRESS:-auto}

# Data: use swe_bench_small for multi-node validation
export DATA_DIR=${DATA_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)/data/swe_bench_small}

echo "=========================================="
echo "SWE-bench 4-Node Launch"
echo "  Nodes:      $NNODES × $GPUS_PER_NODE GPUs"
echo "  Batch size: $TRAIN_BATCH_SIZE"
echo "  Data:       $DATA_DIR"
echo "=========================================="

exec bash "$SCRIPT_DIR/run_swebench.sh" "$@"
