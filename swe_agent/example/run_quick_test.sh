#!/usr/bin/env bash
# =============================================================================
# SWE Agent 快速验证脚本 - 用于测试重构后的代码
#
# 使用最小配置快速验证所有组件能正常工作
# =============================================================================

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
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export NNODES=${NNODES:-1}
export RAY_NUM_NODES=$NNODES

echo "=========================================="
echo "SWE Agent Quick Test - Refactored Code"
echo "=========================================="
echo "Using $NNODES nodes and $GPUS_PER_NODE GPUs per node..."

# ================= paths =================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERL_ROOT="$(cd "$RECIPE_DIR/../.." && pwd)"

# ========== 模型配置 ==========
model_path=${MODEL_PATH:-/data1/lll/workspace/rl/data/models/Qwen/Qwen2___5-3B-Instruct}

# ========== 测试数据配置 ==========
DATA_DIR=$VERL_ROOT/data/swe_agent_test
train_files=$DATA_DIR/train.parquet
test_files=$DATA_DIR/test.parquet

if [ ! -f "$train_files" ]; then
    echo "[ERROR] Test data not found at $train_files"
    echo "Run: python3.12 recipe/swe_agent/prepare/prepare_data.py --mode simple --train_size 2 --test_size 1 --output_dir data/swe_agent_test"
    exit 1
fi

# ========== Agent Loop 配置 ==========
agent_loop_config_path=recipe/swe_agent/config/swe_agent_config.yaml

# =================== wandb ===================
project_name=swe_agent_test
experiment_name=quick-refactor-test
default_local_dir=$VERL_ROOT/checkpoint/$experiment_name

# ================= algorithm =================
adv_estimator=grpo
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

# ========== 最小化配置用于快速测试 ==========
max_turns=10              # 减少轮次
max_prompt_length=2048    # 减少上下文长度
max_response_length=512   # 减少响应长度
actor_lr=1e-6

train_batch_size=4        # 增加以适配多GPU
ppo_mini_batch_size=4
n_resp_per_prompt=1
n_resp_per_prompt_val=1

# =================== logging ===================
export RAY_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1

# ================= performance =================
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

infer_tp=2  # 减少 tensor parallel
train_sp=2  # 减少 sequence parallel

# ================= FSDP Optimization =================
fsdp_strategy=fsdp2
offload_policy=true
param_offload=false
optimizer_offload=false

# ================= vLLM Memory Optimization =================
gpu_memory_utilization=0.3  # 减少内存使用
max_model_len=4096

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

train_files="['$train_files']"
test_files="['$test_files']"

echo "=========================================="
echo "Configuration:"
echo "  Model: $model_path"
echo "  Train data: $train_files"
echo "  Test data: $test_files"
echo "  Agent config: $agent_loop_config_path"
echo "  Max turns: $max_turns"
echo "  Batch size: $train_batch_size"
echo "=========================================="

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
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
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
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger='["console"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.val_before_train=false \
    trainer.log_val_generations=10 \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=-1 \
    trainer.default_local_dir="$default_local_dir" \
    trainer.test_freq=1 \
    trainer.total_epochs=1 "$@"
