#!/usr/bin/env bash
# =============================================================================
# SWE Agent VERL Training Script
# 
# 阶段 B: 完整训练流程
# 
# 使用方法:
#   # 1. 先生成数据
#   python3.12 recipe/swe_agent/prepare/prepare_data.py --mode simple
#   
#   # 2. 运行训练
#   bash recipe/swe_agent/example/run_qwen2.5_3b.sh
#
# =============================================================================
#SBATCH --job-name=rl-swe-agent-3B
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -xeuo pipefail

# ================= Work directories =================
export TMPDIR=/data1/lmy/tmp
export TEMP=/data1/lmy/tmp
export TMP=/data1/lmy/tmp
export RAY_TMPDIR=/data1/lmy/ray_tmp
export TRITON_CACHE_DIR=/data1/lmy/triton_cache
export TORCH_EXTENSIONS_DIR=/data1/lmy/torch_extensions
export HF_HOME=/data1/lmy/hf_cache
export XDG_CACHE_HOME=/data1/lmy/cache
mkdir -p $TMPDIR $RAY_TMPDIR $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $HF_HOME $XDG_CACHE_HOME

# ================= cluster topology =================
export GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${GPUS_PER_NODE:-4}}
NNODES=${SLURM_JOB_NUM_NODES:-${NNODES:-1}}
export NNODES
export RAY_NUM_NODES=$NNODES

TOTAL_GPUS=$((GPUS_PER_NODE * NNODES))
if [ "$TOTAL_GPUS" -lt 2 ]; then
  echo "Error: at least 2 GPUs are required, detected $TOTAL_GPUS." >&2
  exit 1
fi

echo "=========================================="
echo "SWE Agent VERL Training - Phase B"
echo "=========================================="
echo "Using $NNODES nodes and $GPUS_PER_NODE GPUs per node..."

# ================= data/model/tool =================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERL_ROOT="$(cd "$RECIPE_DIR/../.." && pwd)"

HDFS_ROOT=${HDFS_ROOT:-$VERL_ROOT}
DATA_ROOT=${DATA_ROOT:-$VERL_ROOT}

# ========== 模型配置 ==========
# VERL 使用的模型，SWE-Agent 的请求通过 ModelProxy 转发到此模型
model_path=${MODEL_PATH:-/data1/lll/workspace/rl/data/models/Qwen/Qwen2___5-3B-Instruct}

# ========== 数据配置 ==========
# 数据目录
DATA_DIR=${DATA_DIR:-$DATA_ROOT/data/swe_agent}
train_files=$DATA_DIR/train.parquet
test_files=$DATA_DIR/test.parquet

# 检查数据是否存在，如果不存在则生成
if [ ! -f "$train_files" ] || [ ! -f "$test_files" ]; then
    echo "[INFO] Data not found, generating..."
    cd "$VERL_ROOT"
    python3.12 recipe/swe_agent/prepare/prepare_data.py \
        --mode simple \
        --train_size 100 \
        --test_size 10 \
        --output_dir "$DATA_DIR"
    echo "[INFO] Data generation complete."
fi

# ========== Agent Loop 配置 ==========
agent_loop_config_path=recipe/swe_agent/config/swe_agent_config.yaml

# =================== wandb ===================
project_name=swe_agent_rl
experiment_name=qwen2.5-3b-swe-agent
default_local_dir=$DATA_ROOT/checkpoint/$experiment_name

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# SWE-Agent 特定配置
max_turns=50           # SWE-Agent 通常需要更多轮次
max_prompt_length=4096 # SWE 任务需要更长的上下文
max_response_length=2048
actor_lr=1e-6

train_batch_size=4     # SWE 任务较重，减小 batch size
ppo_mini_batch_size=1
n_resp_per_prompt=1    # 每个问题只生成一个 trajectory
n_resp_per_prompt_val=1

# =================== logging ===================
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1

# ================= performance =================
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN

infer_tp=4  # vLLM tensor parallel size
train_sp=4  # Ulysses sequence parallel size

# ================= FSDP Optimization =================
fsdp_strategy=fsdp2
offload_policy=true
param_offload=false
optimizer_offload=false

# ================= vLLM Memory Optimization =================
gpu_memory_utilization=0.4
max_model_len=8192  # SWE 任务需要更长的上下文

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

train_files="['$train_files']"
test_files="['$test_files']"

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
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger='["console"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.val_before_train=true \
    trainer.log_val_generations=50 \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=-1 \
    trainer.default_local_dir="$default_local_dir" \
    trainer.test_freq=5 \
    trainer.total_epochs=1 "$@"
