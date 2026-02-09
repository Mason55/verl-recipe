#!/usr/bin/env bash
# =============================================================================
# SWE Agent VERL 验证脚本 (精简版)
# 
# 用于在 RTX 3090 上快速验证完整训练流程
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
export NNODES=1
export RAY_NUM_NODES=$NNODES

echo "=========================================="
echo "SWE Agent VERL Verification"
echo "=========================================="
echo "Using $GPUS_PER_NODE GPUs"

# ================= paths =================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERL_ROOT="$(cd "$RECIPE_DIR/../.." && pwd)"

cd "$VERL_ROOT"

# ========== 模型配置 ==========
model_path=/data1/lll/workspace/rl/data/models/Qwen/Qwen2___5-3B-Instruct

# ========== 数据配置 ==========
DATA_DIR=$VERL_ROOT/data/swe_agent
train_files=$DATA_DIR/train.parquet
test_files=$DATA_DIR/test.parquet

# 检查数据是否存在
if [ ! -f "$train_files" ]; then
    echo "[INFO] Generating test data..."
    python3.12 recipe/swe_agent/prepare/prepare_data.py \
        --mode simple \
        --train_size 5 \
        --test_size 2 \
        --output_dir "$DATA_DIR"
fi

# ========== Agent Loop 配置 ==========
agent_loop_config_path=recipe/swe_agent/config/swe_agent_config.yaml

# =================== wandb ===================
project_name=swe_agent_verify
experiment_name=qwen2.5-3b-verify

# ================= algorithm =================
adv_estimator=grpo
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

# 精简配置 - 适合验证
max_turns=10              # 减少轮次
max_prompt_length=2048    # 减少上下文
max_response_length=512   # 减少响应长度
train_batch_size=2        # 小 batch
ppo_mini_batch_size=1
n_resp_per_prompt=1

# ================= logging =================
export RAY_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=INFO

# ================= performance =================
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

infer_tp=2   # RTX 3090 使用 2 卡推理
train_sp=2   # 2 卡训练

# ================= Memory Optimization =================
gpu_memory_utilization=0.6
max_model_len=4096

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

train_files="['$train_files']"
test_files="['$test_files']"

echo "[INFO] Starting VERL training..."
echo "[INFO] Model: $model_path"
echo "[INFO] Data: $DATA_DIR"

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
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    trainer.logger='["console"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.val_before_train=false \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 "$@"

echo "[INFO] Verification completed!"
