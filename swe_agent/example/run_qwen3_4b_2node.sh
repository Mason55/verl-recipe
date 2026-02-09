#!/usr/bin/env bash
# =============================================================================
# SWE Agent VERL Training Script - 2-Node (16 GPU) Version
#
# Based on run_qwen3_4b_instruct.sh, adapted for distributed training across:
#   - Head:   8.92.9.152 (8x GPU)
#   - Worker: 8.92.9.150 (8x GPU)
#
# Prerequisites:
#   1. Ray cluster already running (run setup_ray_cluster.sh first)
#   2. Same model/data/code on both nodes
#   3. swerex-python:3.11 Docker image on both nodes
#
# Usage:
#   # Step 1: Start Ray cluster
#   bash recipe/swe_agent/example/setup_ray_cluster.sh
#
#   # Step 2: Run training
#   bash recipe/swe_agent/example/run_qwen3_4b_2node.sh
#
#   # Step 3: Stop cluster when done
#   bash recipe/swe_agent/example/stop_ray_cluster.sh
# =============================================================================

set -xeuo pipefail

# ================= Work directories =================
WORK_BASE=${WORK_BASE:-/data1/lmy/workspace}
export TMPDIR=$WORK_BASE/tmp
export TEMP=$WORK_BASE/tmp
export TMP=$WORK_BASE/tmp
export RAY_TMPDIR=$WORK_BASE/ray_tmp
export TRITON_CACHE_DIR=$WORK_BASE/triton_cache
export TORCH_EXTENSIONS_DIR=$WORK_BASE/torch_extensions
export HF_HOME=$WORK_BASE/hf_cache
export XDG_CACHE_HOME=$WORK_BASE/cache
mkdir -p $TMPDIR $RAY_TMPDIR $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $HF_HOME $XDG_CACHE_HOME

# ================= Cluster topology =================
# 2 nodes x 8 GPUs = 16 GPUs total
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${NNODES:-2}
export RAY_NUM_NODES=$NNODES

echo "=========================================="
echo "SWE Agent 2-Node Training"
echo "=========================================="
echo "  Nodes: $NNODES x $GPUS_PER_NODE GPUs = $(( NNODES * GPUS_PER_NODE )) GPUs total"
echo "=========================================="

# ================= Verify Ray cluster =================
echo "Verifying Ray cluster..."
NODE_COUNT=$(ray status 2>/dev/null | grep -c "node_" || echo "0")
if [ "$NODE_COUNT" -lt "$NNODES" ]; then
    echo "[ERROR] Ray cluster has only ${NODE_COUNT} nodes, expected ${NNODES}"
    echo "  Run: bash recipe/swe_agent/example/setup_ray_cluster.sh"
    exit 1
fi
echo "  Ray cluster OK: ${NODE_COUNT} nodes"

# ================= Paths =================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERL_ROOT="$(cd "$RECIPE_DIR/../.." && pwd)"

# ========== Model config ==========
model_path=${MODEL_PATH:-/data1/models/Qwen/Qwen3-4B-Instruct-2507}

# ========== Data config ==========
DATA_DIR=$VERL_ROOT/data/swe_agent_test
train_files=$DATA_DIR/train.parquet
test_files=$DATA_DIR/test.parquet

if [ ! -f "$train_files" ]; then
    echo "[ERROR] Training data not found at $train_files"
    echo "Run: python3.12 recipe/swe_agent/prepare/prepare_data.py --mode simple --train_size 64 --test_size 8 --output_dir data/swe_agent_test"
    exit 1
fi

# ========== Agent Loop config ==========
agent_loop_config_path=recipe/swe_agent/config/swe_agent_config.yaml

# =================== wandb ===================
project_name=swe_agent_2node
experiment_name=qwen3-4b-swe-2node-v2-optimized
default_local_dir=$WORK_BASE/checkpoints/$experiment_name

# ================= Algorithm =================
adv_estimator=grpo
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# ========== Training parameters ==========
# Optimized for 2 nodes (16 GPUs total) with higher concurrency
max_turns=8      # Reduced from 15 — most patches in 2~5 turns; cuts long-tail blocking
max_prompt_length=4096
max_response_length=8192    # Reduced from 16384 — 8 turns * ~200 tokens is enough
actor_lr=5e-6

# Batch size: 4x per GPU → 64 total
# Larger batch = more concurrent agent interactions = better GPU utilization
train_batch_size=64
ppo_mini_batch_size=16
n_resp_per_prompt=1
n_resp_per_prompt_val=1

# Agent loop workers: match batch_size for max concurrency
# Each worker handles batch_size/num_workers samples concurrently
# Default is 8 (from rollout.yaml); we override to 16 for 2-node setup
num_agent_loop_workers=16

# =================== Logging ===================
export RAY_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1

# ================= NCCL for cross-node communication =================
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# CRITICAL: Use real NIC for cross-node NCCL communication
# enp96s0f0 is the interface bound to 8.92.x.x on both nodes
export NCCL_SOCKET_IFNAME=enp96s0f0
# Do NOT disable SHM or P2P for multi-node (they were disabled for single-node /dev/shm workaround)
# export NCCL_SHM_DISABLE=1    # removed
# export NCCL_P2P_DISABLE=1    # removed
# export NCCL_SOCKET_IFNAME=lo # removed — lo only works for single-node

# TP/SP stay within a single node (8 GPUs each)
infer_tp=8
train_sp=8

# ================= FSDP Optimization =================
fsdp_strategy=fsdp2
offload_policy=true
param_offload=false
optimizer_offload=false

# ================= vLLM Memory Optimization =================
gpu_memory_utilization=0.5  # Increased from 0.4 — more KV cache for concurrent requests
max_model_len=12288         # Reduced from 16384 — 4096 prompt + 8192 response = 12288

rollout_prompt_length=$max_prompt_length
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 2 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

train_files="['$train_files']"
test_files="['$test_files']"

echo "=========================================="
echo "Configuration:"
echo "  Model: $model_path"
echo "  Train data: $train_files"
echo "  Test data: $test_files"
echo "  Agent config: $agent_loop_config_path"
echo "  Nodes: $NNODES x $GPUS_PER_NODE GPUs"
echo "  Batch size: $train_batch_size"
echo "  Mini-batch: $ppo_mini_batch_size"
echo "  Max turns: $max_turns"
echo "  TP: $infer_tp, SP: $train_sp"
echo "  Agent workers: $num_agent_loop_workers"
echo "  NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "=========================================="

# ================= Pre-install SWE-Agent tools =================
SWE_AGENT_TOOLS_SRC="$VERL_ROOT/../SWE-agent/tools"
SWE_AGENT_TOOLS_DST="/root/tools"

echo "Checking SWE-Agent tools pre-installation..."
if [ -d "$SWE_AGENT_TOOLS_SRC" ]; then
    if mkdir -p "$SWE_AGENT_TOOLS_DST" 2>/dev/null; then
        rm -rf "$SWE_AGENT_TOOLS_DST"/* 2>/dev/null || true
        if cp -r "$SWE_AGENT_TOOLS_SRC"/* "$SWE_AGENT_TOOLS_DST"/ 2>/dev/null; then
            echo "  SWE-Agent tools pre-installed to $SWE_AGENT_TOOLS_DST"
        fi
    fi
fi
echo ""

# ================= Launch training via ray job submit =================
# IMPORTANT: Must use "ray job submit" for multi-node training.
# Direct "python3.12 -m verl.trainer.main_ppo" only sees the local node's GPUs.
# "ray job submit" runs the driver within the Ray cluster context, seeing all nodes.
HEAD_IP=${HEAD_IP:-8.92.9.152}
RAY_DASHBOARD="http://${HEAD_IP}:8265"

echo "Submitting training job to Ray cluster at ${RAY_DASHBOARD}..."

# Both nodes have identical code at same paths, so no need for --working-dir upload.
# Use --runtime-env-json to pass NCCL env vars to all Ray workers.
RUNTIME_ENV=$(cat <<'RENV'
{
  "env_vars": {
    "NCCL_SOCKET_IFNAME": "enp96s0f0",
    "NCCL_DEBUG": "WARN",
    "VLLM_USE_V1": "1",
    "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    "HYDRA_FULL_ERROR": "1",
    "RAY_LOGGING_LEVEL": "INFO"
  }
}
RENV
)

ray job submit --address="${RAY_DASHBOARD}" \
    --runtime-env-json="${RUNTIME_ENV}" \
    --no-wait \
    -- \
    python3.12 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=true \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=true \
    data.truncation='error' \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.strategy=$fsdp_strategy \
    actor_rollout_ref.actor.fsdp_config.offload_policy=$offload_policy \
    actor_rollout_ref.actor.fsdp_config.param_offload=$param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    actor_rollout_ref.rollout.agent.num_workers=$num_agent_loop_workers \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.prompt_length=$rollout_prompt_length \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    custom_reward_function.path="${RECIPE_DIR}/reward/compute_score.py" \
    custom_reward_function.name=compute_score \
    'trainer.logger=["console"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.val_before_train=false \
    trainer.log_val_generations=10 \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=5 \
    trainer.default_local_dir="$default_local_dir" \
    trainer.test_freq=5 \
    trainer.total_epochs=3

echo ""
echo "=========================================="
echo "  Job submitted! Monitor with:"
echo "    ray job list"
echo "    ray job logs <SUBMISSION_ID> --follow"
echo "  Dashboard: ${RAY_DASHBOARD}"
echo "=========================================="
