#!/bin/bash
set -euo pipefail

echo "=============================================="
echo "LOKA END-TO-END GRPO TEST"
echo "SLURM Job ID : ${SLURM_JOB_ID}"
echo "Node         : ${SLURM_NODELIST}"
echo "GPUs / node  : 8"
echo "Start time   : $(date)"
echo "=============================================="

LOKA_ROOT="/data/users/harsha/loka"
export LOKA_DATA_DIR="${LOKA_ROOT}/data"
export LOKA_MODEL_DIR="${LOKA_ROOT}/checkpoints/e2e_test"
export LOKA_CACHE_DIR="${LOKA_ROOT}/cache"
export HF_HOME="${LOKA_CACHE_DIR}/huggingface"
export PYTHONPATH="${LOKA_ROOT}/src:${PYTHONPATH:-}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CROSS_NIC=1
export NCCL_IB_DISABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_DEBUG=WARN

# Unset AMD ROCm env vars that conflict with verl's CUDA device setup
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

mkdir -p "${LOKA_MODEL_DIR}" "${LOKA_CACHE_DIR}" "${LOKA_ROOT}/logs"

# ── Step 1: GPU check ────────────────────────────────────────────────
echo ""
echo ">>> Step 1/6: GPU health"
nvidia-smi -L
echo ""

# ── Step 2: Generate tiny e2e dataset ────────────────────────────────
echo ">>> Step 2/6: Generate tiny dataset (64 train / 16 val)"
E2E_DATA="${LOKA_DATA_DIR}/e2e_test"
mkdir -p "${E2E_DATA}"
python3 "${LOKA_ROOT}/scripts/generate_training_data.py" \
    --n-train 64 --n-val 16 \
    --output-dir "${E2E_DATA}" \
    --seed 123
ls -lh "${E2E_DATA}/"
echo ""

# ── Step 3: Start Ray ───────────────────────────────────────────────
echo ">>> Step 3/6: Starting Ray cluster (single-node, 8 GPUs)"
ray start --head --port=6379 \
    --num-cpus=104 --num-gpus=8 \
    --include-dashboard=false
sleep 5
ray status
echo ""

# ── Step 4: Download model weights (if not cached) ───────────────────
echo ">>> Step 4/6: Ensuring Qwen2.5-7B-Instruct weights are cached"
python3 << 'PYEOF'
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

model_name = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading tokenizer for {model_name} ...")
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f"Tokenizer OK (vocab_size={tok.vocab_size})")
print("Checking model weights are available (download if needed)...")
path = snapshot_download(model_name, ignore_patterns=["*.gguf", "*.ggml"])
print(f"Model weights cached at: {path}")
PYEOF
echo ""

# ── Step 5: Launch Verl GRPO (tiny config, 1 epoch) ─────────────────
echo ">>> Step 5/6: Launching Verl GRPO trainer (tiny config, 1 epoch)"
echo "  batch_size=32, mini_batch=16, micro_batch=2, rollout.n=4"
echo "  total_epochs=1, save_freq=1"
echo ""

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="${E2E_DATA}/train.parquet" \
    data.val_files="${E2E_DATA}/val.parquet" \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
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
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    \
    reward_model.enable=False \
    custom_reward_function.path="${LOKA_ROOT}/src/loka/rl/reward.py" \
    custom_reward_function.name=compute_score \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.project_name=orbital_rl \
    trainer.experiment_name=e2e_test \
    trainer.logger=console \
    trainer.save_freq=1 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${LOKA_MODEL_DIR}"

E2E_EXIT=$?

# ── Step 6: Verify outputs ──────────────────────────────────────────
echo ""
echo ">>> Step 6/6: Verifying outputs"
echo "Verl exit code: ${E2E_EXIT}"

echo "Checkpoint directory:"
find "${LOKA_MODEL_DIR}" -type f 2>/dev/null | head -20 || echo "(empty)"

ray stop
echo ""
echo "=============================================="
if [ ${E2E_EXIT} -eq 0 ]; then
    echo "E2E TEST PASSED at $(date)"
else
    echo "E2E TEST FAILED (exit=${E2E_EXIT}) at $(date)"
fi
echo "=============================================="
exit ${E2E_EXIT}
