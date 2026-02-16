#!/bin/bash
#SBATCH --job-name=loka-grpo
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=900G
#SBATCH --time=72:00:00
#SBATCH --partition=main

# =============================================================================
# GRPO Training on 2×8 H100 GPUs — native worker env + Ray + Verl
# Uses the SLURM worker's pre-installed PyTorch/Ray/Transformers stack,
# with additional RL packages installed to /data shared storage.
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "SLURM Job ID : ${SLURM_JOB_ID}"
echo "Nodes        : ${SLURM_NODELIST}"
echo "Tasks        : ${SLURM_NTASKS}"
echo "GPUs / node  : 8"
echo "Start time   : $(date)"
echo "=============================================="

# ── Paths ────────────────────────────────────────────────────────────
LOKA_ROOT="${LOKA_ROOT:-/data/users/${USER}/loka}"
export LOKA_DATA_DIR="${LOKA_ROOT}/data"
export LOKA_MODEL_DIR="${LOKA_ROOT}/checkpoints"
export LOKA_CACHE_DIR="${LOKA_ROOT}/cache"
export HF_HOME="${LOKA_CACHE_DIR}/huggingface"
# HF_TOKEN: set via environment or `huggingface-cli login` before submission
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before submitting this job}"

# WANDB_API_KEY: set via environment or `wandb login` before submission
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY before submitting this job}"
export WANDB_PROJECT="${WANDB_PROJECT:-orbital_rl}"

# site-packages FIRST so verl/vllm-compatible transformers 4.57.x
# takes priority over system's transformers 5.1.0 (which removed
# AutoModelForVision2Seq that verl requires).
export PYTHONPATH="${LOKA_ROOT}/site-packages:${LOKA_ROOT}/src:${PYTHONPATH:-}"

# ── NCCL / CUDA / vLLM ──────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CROSS_NIC=1
export NCCL_IB_DISABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_DEBUG=INFO

mkdir -p "${LOKA_DATA_DIR}/orbital" "${LOKA_MODEL_DIR}" "${LOKA_CACHE_DIR}" "${LOKA_ROOT}/logs"

nvidia-smi

# ── Generate training data if missing ────────────────────────────────
if [ ! -f "${LOKA_DATA_DIR}/orbital/train.parquet" ]; then
    echo "Generating training data ..."
    python3 "${LOKA_ROOT}/scripts/generate_training_data.py" \
        --n-train 10000 --n-val 1000 \
        --output-dir "${LOKA_DATA_DIR}/orbital" \
        --seed 42
fi
echo "Training data:"
ls -lh "${LOKA_DATA_DIR}/orbital/"

# ── Resolve node addresses ───────────────────────────────────────────
NODELIST=($(scontrol show hostnames "${SLURM_NODELIST}"))
HEAD_NODE=${NODELIST[0]}
HEAD_ADDR=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -i 2>/dev/null | awk '{print $1}')
RAY_PORT=6379

echo "Head node: ${HEAD_NODE} (${HEAD_ADDR}:${RAY_PORT})"

# ── Start Ray cluster ───────────────────────────────────────────────
# Head
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" --gres=gpu:8 \
    ray start --head --port=${RAY_PORT} \
    --num-cpus=96 --num-gpus=8 \
    --dashboard-host=0.0.0.0 --include-dashboard=true --block &

echo "Ray Dashboard: http://${HEAD_ADDR}:8265"
sleep 20  # give head time to bind

# Workers
for node in "${NODELIST[@]:1}"; do
    echo "Starting Ray worker on ${node} ..."
    srun --nodes=1 --ntasks=1 -w "${node}" --gres=gpu:8 \
        ray start --address="${HEAD_ADDR}:${RAY_PORT}" \
        --num-cpus=96 --num-gpus=8 --block &
done

sleep 25  # let workers join

echo "Ray cluster status:"
ray status || true

# ── Launch Verl GRPO training ────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="${LOKA_DATA_DIR}/orbital/train.parquet" \
    data.val_files="${LOKA_DATA_DIR}/orbital/val.parquet" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    \
    reward_model.enable=False \
    custom_reward_function.path="${LOKA_ROOT}/src/loka/rl/reward.py" \
    custom_reward_function.name=compute_score \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.project_name=orbital_rl \
    trainer.experiment_name=leo_to_geo_grpo \
    trainer.logger=wandb \
    trainer.save_freq=50 \
    trainer.val_before_train=True \
    trainer.total_epochs=3 \
    trainer.default_local_dir="${LOKA_MODEL_DIR}"

echo "=============================================="
echo "GRPO training completed at: $(date)"
echo "=============================================="
