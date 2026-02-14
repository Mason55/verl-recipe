#!/usr/bin/env bash
# =============================================================================
# SWE-bench Mini Test: 2 instances, prebuilt images, end-to-end validation
# =============================================================================
# Quick sanity test before launching full training.
# Uses django__django-15375 and django__django-15973.
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

# ================= cluster topology =================
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${NNODES:-1}
export RAY_NUM_NODES=$NNODES

echo "=========================================="
echo "SWE-bench Mini Test (2 instances)"
echo "=========================================="

# ================= paths =================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERL_ROOT="$(cd "$RECIPE_DIR/../.." && pwd)"

# ========== Model config ==========
model_path=${MODEL_PATH:-/data1/models/Qwen/Qwen3-4B-Instruct-2507}

# ========== Mini data: just 2 instances ==========
DATA_DIR=$VERL_ROOT/data/swe_bench
train_files=$DATA_DIR/mini_train.parquet
test_files=$DATA_DIR/mini_test.parquet

if [ ! -f "$train_files" ]; then
    echo "[ERROR] Mini data not found at $train_files"
    exit 1
fi

# ========== Agent Loop config ==========
agent_loop_config_path=recipe/swe_agent/config/swe_agent_config_swebench.yaml

# =================== wandb ===================
project_name=swe_bench_mini_test
experiment_name=qwen3-4b-swebench-mini
default_local_dir=$WORK_BASE/checkpoints/$experiment_name

# ================= algorithm =================
adv_estimator=grpo
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# ========== Training parameters ==========
# 2 instances, batch_size=2, must be divisible by n_agent_loop_workers
# With 8 GPUs and 2 instances, we need to match the chunk requirement.
# The agent loop workers = GPUS_PER_NODE, so batch_size must be divisible by 8.
# BUT we only have 2 data points. Set n_agent_loop_workers explicitly.
max_turns=5               # Keep it short for testing
max_prompt_length=8192
max_response_length=4096
actor_lr=5e-6

train_batch_size=2
ppo_mini_batch_size=2
n_resp_per_prompt=1
n_resp_per_prompt_val=1

# =================== logging ===================
export RAY_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1

# ================= performance =================
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

unset NCCL_SHM_DISABLE 2>/dev/null || true
unset NCCL_P2P_DISABLE 2>/dev/null || true
unset NCCL_SOCKET_IFNAME 2>/dev/null || true

# TP/SP: use all GPUs for model but only 2 agent loop workers
infer_tp=8
train_sp=8

# ================= FSDP Optimization =================
fsdp_strategy=fsdp2
offload_policy=true
param_offload=false
optimizer_offload=false

# ================= vLLM Memory Optimization =================
gpu_memory_utilization=0.5
max_model_len=16384

# Token length calculations
rollout_prompt_length=$max_prompt_length
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 2 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

train_files="['$train_files']"
test_files="['$test_files']"

echo "=========================================="
echo "Configuration:"
echo "  Model: $model_path"
echo "  Train data: $train_files (2 instances)"
echo "  Agent config: $agent_loop_config_path"
echo "  Max turns: $max_turns"
echo "  Batch size: $train_batch_size"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.rollout.agent.n_agent_loop_workers=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.prompt_length=$rollout_prompt_length \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    custom_reward_function.path="${RECIPE_DIR}/reward/reward.py" \
    custom_reward_function.name=compute_score \
    trainer.logger='["console"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.val_before_train=false \
    trainer.log_val_generations=2 \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=100 \
    trainer.default_local_dir="$default_local_dir" \
    trainer.test_freq=1 \
    trainer.total_epochs=1 "$@"
