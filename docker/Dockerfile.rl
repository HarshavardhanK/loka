# =============================================================================
# Loka RL Training — Overlay on cluster worker base image
# =============================================================================
# The cluster worker base image ships PyTorch, Ray, Transformers, CUDA, etc.
# We overlay RL-specific packages (verl, vllm) and the loka source tree.
#
# BASE_IMAGE must be provided via --build-arg at build time.
# =============================================================================

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root

# --- Downgrade torch to 2.9.1 (vllm 0.15.1 ABI requirement) ----------------
RUN pip install --no-cache-dir \
    torch==2.9.1+cu126 torchvision==0.24.1+cu126 torchaudio==2.9.1+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# --- Downgrade transformers <5 (verl requirement) ----------------------------
RUN pip install --no-cache-dir "transformers>=4.56,<5"

# --- Install non-conflicting science packages --------------------------------
RUN pip install --no-cache-dir \
    "gymnasium>=1.0" \
    "astropy>=6.0,<7" \
    "jplephem>=2.18" \
    "scipy>=1.13" \
    "pyyaml>=6.0" \
    "packaging>=25.0"

# --- Install verl + vllm WITHOUT transitive deps ----------------------------
# Letting pip resolve verl/vllm deps causes resolution-too-deep against
# the base image's packages. Install --no-deps, then add missing sub-deps.
# verl 0.8.0.dev0 from main is needed for vllm 0.15.1 API compatibility.
RUN pip install --no-cache-dir --no-deps \
    "git+https://github.com/volcengine/verl.git@main" \
    vllm==0.15.1

# --- Install missing sub-dependencies of verl / vllm ------------------------
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
    torchdata \
    peft \
    openai-harmony \
    llguidance \
    xgrammar \
    "lm-format-enforcer==0.11.3" \
    prometheus-fastapi-instrumentator \
    model-hosting-container-standards \
    pybase64 \
    sentencepiece \
    setproctitle \
    einops \
    cachetools \
    cbor2 \
    ijson \
    "lark>=1.2.2" \
    "prometheus-client"

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
