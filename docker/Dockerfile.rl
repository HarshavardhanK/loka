# =============================================================================
# Loka RL Training — Thin overlay on the SLURM worker image
# =============================================================================
# The cluster worker image already ships:
#   PyTorch 2.10+cu129, Ray 2.53, Transformers 5.1, Accelerate 1.12,
#   numba, pandas, pyarrow, wandb, datasets, CUDA 12.9
#
# We only add the RL-specific packages and the loka source tree.
# =============================================================================

# Base image: override via --build-arg if your cluster uses a different worker image
ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
FROM ${BASE_IMAGE}

USER root

# --- Install non-conflicting packages normally ------------------------------
RUN pip install --no-cache-dir \
    "gymnasium>=1.0" \
    "astropy>=6.0,<7" \
    "jplephem>=2.18" \
    "scipy>=1.13" \
    "pyyaml>=6.0" \
    "packaging>=25.0"

# --- Install verl + vllm WITHOUT transitive deps ---------------------------
# The base image already has torch, transformers, huggingface_hub, ray, etc.
# Letting pip resolve verl/vllm deps causes resolution-too-deep against
# the base image's newer versions.  We install them --no-deps and then
# add only the genuinely missing sub-dependencies below.
RUN pip install --no-cache-dir --no-deps verl vllm

# --- Install missing sub-dependencies of verl / vllm -----------------------
# flashinfer-python is omitted — vllm bundles its own attention kernels,
# and flashinfer requires CUDA compilation that is impractical under QEMU.
# It will be installed on the cluster at runtime if needed.
RUN pip install --no-cache-dir \
    codetiming \
    hydra-core \
    omegaconf \
    openai \
    msgspec \
    partial-json-parser \
    compressed-tensors \
    depyf \
    gguf \
    mistral-common \
    py-cpuinfo \
    blake3 \
    uvloop \
    watchfiles \
    starlette \
    uvicorn \
    fastapi \
    outlines_core \
    interegular \
    diskcache \
    grpcio \
    protobuf \
    tabulate \
    tensordict \
    "lark>=1.2.2" \
    "prometheus-client"

# --- Copy loka source and install in-place ----------------------------------
WORKDIR /code/loka
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

# --- Environment for multi-node GRPO ----------------------------------------
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV NCCL_CROSS_NIC=1
ENV NCCL_IB_DISABLE=0
ENV TORCH_NCCL_AVOID_RECORD_STREAMS=1
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN
ENV NCCL_DEBUG=INFO
ENV PYTHONUNBUFFERED=1

CMD ["bash"]
