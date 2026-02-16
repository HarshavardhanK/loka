#!/bin/bash
#SBATCH --job-name=loka-grpo
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu

# =============================================================================
# GRPO Training on 2×8 H100 GPUs via Ray + Verl
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "SLURM Job ID : $SLURM_JOB_ID"
echo "Nodes        : $SLURM_NODELIST"
echo "GPUs / node  : 8"
echo "Start time   : $(date)"
echo "=============================================="

# ── Modules ───────────────────────────────────────────────────────────
module load cuda/12.1
module load anaconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate loka

# ── Environment ───────────────────────────────────────────────────────
export LOKA_DATA_DIR="${LOKA_DATA_DIR:-/scratch/$USER/loka/data}"
export LOKA_MODEL_DIR="${LOKA_MODEL_DIR:-/scratch/$USER/loka/models}"
export LOKA_CACHE_DIR="${LOKA_CACHE_DIR:-/scratch/$USER/loka/cache}"
export HF_HOME="$LOKA_CACHE_DIR/huggingface"

# Weights & Biases
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY before submitting this job}"
export WANDB_PROJECT="${WANDB_PROJECT:-orbital_rl}"

# NCCL / CUDA / vLLM
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CROSS_NIC=1
export NCCL_IB_DISABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

mkdir -p "$LOKA_DATA_DIR" "$LOKA_MODEL_DIR" "$LOKA_CACHE_DIR" logs

cd "$SLURM_SUBMIT_DIR"
nvidia-smi

# ── Resolve node addresses ────────────────────────────────────────────
NODELIST=($(scontrol show hostnames "$SLURM_NODELIST"))
HEAD_NODE=${NODELIST[0]}
HEAD_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -i | awk '{print $1}')
RAY_PORT=6379

echo "Head node: $HEAD_NODE ($HEAD_ADDR:$RAY_PORT)"

# ── Start Ray cluster ────────────────────────────────────────────────
# Head
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    ray start --head --port=$RAY_PORT \
    --num-cpus=64 --num-gpus=8 \
    --dashboard-host=0.0.0.0 --include-dashboard=true --block &

echo "Ray Dashboard: http://${HEAD_ADDR}:8265"

sleep 15  # give head time to bind

# Workers
for node in "${NODELIST[@]:1}"; do
    echo "Starting Ray worker on $node ..."
    srun --nodes=1 --ntasks=1 -w "$node" \
        ray start --address="$HEAD_ADDR:$RAY_PORT" \
        --num-cpus=64 --num-gpus=8 --block &
done

sleep 20  # let workers join

# ── Start checkpoint manager (background) ────────────────────────────
python3 -m loka.rl.checkpoint \
    --checkpoint-dir "$LOKA_MODEL_DIR" \
    --keep-last 2 --keep-best 3 \
    --watch --interval 120 &
CKPT_PID=$!
echo "Checkpoint manager PID: ${CKPT_PID}"

# ── Launch Verl GRPO training ─────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="$LOKA_DATA_DIR/orbital/train.parquet" \
    data.val_files="$LOKA_DATA_DIR/orbital/val.parquet" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
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
    \
    reward_model.enable=False \
    custom_reward_function.path=src/loka/rl/reward.py \
    custom_reward_function.name=compute_score \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.project_name=orbital_rl \
    trainer.experiment_name=leo_to_geo_grpo \
    trainer.logger=wandb \
    trainer.save_freq=50 \
    trainer.val_before_train=True \
    trainer.total_epochs=3

kill ${CKPT_PID} 2>/dev/null || true

python3 -m loka.rl.checkpoint \
    --checkpoint-dir "$LOKA_MODEL_DIR" \
    --keep-last 2 --keep-best 3 \
    --summary

echo "=============================================="
echo "GRPO training completed at: $(date)"
echo "=============================================="
