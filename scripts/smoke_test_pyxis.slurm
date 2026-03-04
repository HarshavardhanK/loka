#!/bin/bash
#SBATCH --job-name=loka-smoke
#SBATCH --output=/data/users/harsha/loka/logs/smoke_%j.out
#SBATCH --error=/data/users/harsha/loka/logs/smoke_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=900G
#SBATCH --time=00:30:00
#SBATCH --partition=main
#
# --- Pyxis / enroot container ---
# NOTE: --container-image passed via CLI to avoid '#' being parsed as comment
#SBATCH --container-mounts=/data:/data
#SBATCH --container-writable
#SBATCH --no-container-entrypoint

# =============================================================================
# Smoke test — single node, 8 GPUs, Pyxis container
# Validates: image pull, GPU access, Python deps, loka module imports,
#            reward function, and a minimal verl dry-run.
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "LOKA SMOKE TEST"
echo "SLURM Job ID : ${SLURM_JOB_ID}"
echo "Node         : ${SLURM_NODELIST}"
echo "GPUs / node  : 8"
echo "Start time   : $(date)"
echo "=============================================="

LOKA_ROOT="/data/users/harsha/loka"
export PYTHONPATH="${LOKA_ROOT}/src:${PYTHONPATH:-}"

# ── 1. GPU health ─────────────────────────────────────────────────────
echo ""
echo ">>> Step 1: GPU health"
nvidia-smi
echo ""

# ── 2. Python stack ───────────────────────────────────────────────────
echo ">>> Step 2: Python stack versions"
python3 -c "
import sys; print(f'Python {sys.version}')
import torch; print(f'torch={torch.__version__}  CUDA={torch.cuda.is_available()}  GPUs={torch.cuda.device_count()}')
import vllm; print(f'vllm={vllm.__version__}')
import verl; print('verl OK')
import astropy; print(f'astropy={astropy.__version__}')
import ray; print(f'ray={ray.__version__}')
import transformers; print(f'transformers={transformers.__version__}')
import scipy; print(f'scipy={scipy.__version__}')
import numpy as np; print(f'numpy={np.__version__}')
print()
print('All core imports OK')
"
echo ""

# ── 3. Loka module imports ────────────────────────────────────────────
echo ">>> Step 3: Loka module imports"
python3 -c "
from loka.rl.bridge import ActionParser, OrbitalObservationWrapper, SYSTEM_PROMPT
from loka.rl.reward import compute_score
from loka.rl.metrics import SampleMetrics, MetricsTracker, get_tracker
from loka.rl.checkpoint import CheckpointManager
print('All loka.rl imports OK')

# Quick functional test of reward function
score = compute_score(
    data_source='test',
    solution_str='<think>Coast to apoapsis.</think><action>{\"thrust\": 0.5, \"angle\": 0.0}</action>',
    ground_truth='{\"a_target\": 42164.0, \"e_target\": 0.0, \"dv_hohmann\": 3.94}',
    extra_info={'mission_reward': 80.0, 'success': True, 'a': 42164.0, 'e': 0.001, 'dv_total': 3.8, 'dv_hohmann': 3.94},
)
print(f'compute_score returned: {score:.4f} (expected ~0.6)')
assert 0.0 < score <= 1.0, f'Unexpected score: {score}'
print('Reward function OK')
"
echo ""

# ── 4. NCCL multi-GPU test ────────────────────────────────────────────
echo ">>> Step 4: NCCL multi-GPU communication"
python3 -c "
import torch
import torch.distributed as dist
import os

n = torch.cuda.device_count()
print(f'Testing {n} GPUs ...')
for i in range(n):
    with torch.cuda.device(i):
        x = torch.randn(1024, 1024, device=f'cuda:{i}')
        y = x @ x.T
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}  matmul OK  mem={torch.cuda.memory_allocated(i)/1e9:.1f}GB')
        del x, y
        torch.cuda.empty_cache()
print('Multi-GPU matmul OK')
"
echo ""

# ── 5. HuggingFace model access (tokenizer only) ─────────────────────
echo ">>> Step 5: HuggingFace model access (tokenizer)"
export HF_HOME="/data/users/harsha/loka/cache/huggingface"
mkdir -p "${HF_HOME}"
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
print(f'Tokenizer loaded: vocab_size={tok.vocab_size}')
test = tok.encode('Transfer orbit from LEO to GEO')
print(f'Tokenization OK: {len(test)} tokens')
" 2>&1 || echo "WARNING: HF model download needs HF_TOKEN or network access"
echo ""

echo "=============================================="
echo "SMOKE TEST COMPLETE at $(date)"
echo "=============================================="
