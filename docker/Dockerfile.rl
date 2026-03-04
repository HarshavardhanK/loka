# =============================================================================
# Loka RL Training — Overlay on cluster worker base image
# =============================================================================
# Base: ghcr.io/voltagepark/slurm-worker (ships torch 2.10, transformers 5.x,
#        ray 2.53, CUDA 12.6, numpy 2.2, wandb, accelerate, etc.)
#
# vllm 0.16.0 pins torch==2.9.1 and transformers<5, so we downgrade those
# from the base image's versions before installing verl + vllm.
#
# BASE_IMAGE must be provided via --build-arg at build time.
# =============================================================================

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root

# --- Downgrade torch to 2.9.1 (vllm 0.16.0 ABI requirement) ----------------
RUN pip install --no-cache-dir \
    torch==2.9.1+cu126 torchvision==0.24.1+cu126 torchaudio==2.9.1+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# --- Downgrade transformers <5 (vllm 0.16.0 / verl requirement) -------------
RUN pip install --no-cache-dir "transformers>=4.56,<5"

# --- Downgrade numba to 0.61.2 (vllm 0.16.0 pins it exactly) ---------------
RUN pip install --no-cache-dir "numba==0.61.2"

# --- Install non-conflicting science packages --------------------------------
RUN pip install --no-cache-dir \
    "gymnasium>=1.0" \
    "astropy>=7.0" \
    "jplephem>=2.18" \
    "pyyaml>=6.0" \
    "packaging>=25.0"

# --- Install verl + vllm WITHOUT transitive deps ----------------------------
# Letting pip resolve verl/vllm deps causes resolution-too-deep against
# the base image's packages. Install --no-deps, then add missing sub-deps.
RUN pip install --no-cache-dir --no-deps \
    "git+https://github.com/volcengine/verl.git@main" \
    vllm==0.16.0

# --- Install missing sub-dependencies of verl / vllm ------------------------
RUN pip install --no-cache-dir \
    codetiming \
    hydra-core \
    omegaconf \
    openai \
    msgspec \
    partial-json-parser \
    "compressed-tensors==0.13.0" \
    "depyf==0.20.0" \
    "gguf>=0.17.0" \
    "mistral-common[image]>=1.9.0" \
    py-cpuinfo \
    blake3 \
    uvloop \
    watchfiles \
    starlette \
    uvicorn \
    "fastapi[standard]>=0.115.0" \
    "outlines_core==0.2.11" \
    "diskcache==5.6.3" \
    grpcio \
    grpcio-reflection \
    "protobuf>=5.29.6" \
    tabulate \
    tensordict \
    torchdata \
    peft \
    "openai-harmony>=0.0.3" \
    "llguidance>=1.3.0,<1.4.0" \
    "xgrammar==0.1.29" \
    "lm-format-enforcer==0.11.3" \
    prometheus-fastapi-instrumentator \
    "model-hosting-container-standards>=0.1.13,<1.0.0" \
    pybase64 \
    sentencepiece \
    setproctitle \
    einops \
    cachetools \
    cbor2 \
    ijson \
    "lark==1.2.2" \
    "prometheus-client" \
    cloudpickle \
    pyzmq \
    python-json-logger \
    "opencv-python-headless>=4.13.0" \
    "anthropic>=0.71.0" \
    "tiktoken>=0.6.0" \
    "flashinfer-python==0.6.3"

# --- Copy loka source and install in-place -----------------------------------
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
