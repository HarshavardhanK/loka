#!/bin/bash
#SBATCH --job-name=loka-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# =============================================================================
# Loka Training Script for SLURM
# =============================================================================

set -e

echo "=============================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=============================================="

# Load modules (adjust based on your cluster)
module load cuda/12.1
module load anaconda3

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate loka

# Set environment variables
export LOKA_DATA_DIR="${LOKA_DATA_DIR:-/scratch/$USER/loka/data}"
export LOKA_MODEL_DIR="${LOKA_MODEL_DIR:-/scratch/$USER/loka/models}"
export LOKA_CACHE_DIR="${LOKA_CACHE_DIR:-/scratch/$USER/loka/cache}"
export HF_HOME="$LOKA_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$LOKA_CACHE_DIR/transformers"

# NCCL settings for distributed training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Create directories
mkdir -p "$LOKA_DATA_DIR" "$LOKA_MODEL_DIR" "$LOKA_CACHE_DIR" logs

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Print GPU info
nvidia-smi

# Run training with accelerate for multi-GPU
accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    scripts/train.py \
    --config configs/train_config.yaml \
    --output_dir "$LOKA_MODEL_DIR/loka-$(date +%Y%m%d-%H%M%S)"

echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="
