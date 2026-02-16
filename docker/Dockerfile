# Loka - Agentic Astrophysics Model
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Base with CUDA and Python
# =============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install miniforge (conda)
ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p $CONDA_DIR \
    && rm /tmp/miniforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# =============================================================================
# Stage 2: Dependencies
# =============================================================================
FROM base AS dependencies

WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Activate environment in shell
SHELL ["conda", "run", "-n", "loka", "/bin/bash", "-c"]

# =============================================================================
# Stage 3: Production
# =============================================================================
FROM dependencies AS production

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/

# Install loka package
RUN conda run -n loka pip install -e .

# Create directories for data and models
RUN mkdir -p /data /models /cache

# Environment variables
ENV LOKA_DATA_DIR=/data
ENV LOKA_MODEL_DIR=/models
ENV LOKA_CACHE_DIR=/cache
ENV JPL_EPHEMERIS_PATH=/data/ephemeris

# Default command
ENTRYPOINT ["conda", "run", "-n", "loka"]
CMD ["python", "-c", "from loka import LokaAgent; print('Loka ready')"]

# =============================================================================
# Stage 4: Development
# =============================================================================
FROM dependencies AS development

WORKDIR /app

# Install additional dev tools
RUN conda run -n loka pip install \
    jupyterlab \
    ipdb \
    nvitop

# Copy all project files
COPY . .

# Install in editable mode with dev dependencies
RUN conda run -n loka pip install -e ".[dev,all]"

# Expose Jupyter port
EXPOSE 8888

ENTRYPOINT ["conda", "run", "-n", "loka"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Stage 5: Training (SFT)
# =============================================================================
FROM dependencies AS training

WORKDIR /app

# Copy project files
COPY . .

# Install with all dependencies
RUN conda run -n loka pip install -e ".[all]"

# Install distributed training dependencies
RUN conda run -n loka pip install \
    deepspeed \
    flash-attn --no-build-isolation

# Environment for distributed training
ENV NCCL_DEBUG=INFO
ENV TORCH_DISTRIBUTED_DEBUG=DETAIL

ENTRYPOINT ["conda", "run", "-n", "loka"]
CMD ["python", "-m", "loka.train"]

# =============================================================================
# Stage 6: RL Training (GRPO via Verl on multi-node H100s)
# =============================================================================
FROM dependencies AS rl-training

WORKDIR /app

# Copy project files
COPY . .

# Install loka with RL dependencies
RUN conda run -n loka pip install -e ".[rl,all]"

# Install RL-specific heavy dependencies
# Note: flash-attn is omitted here — vLLM bundles its own FlashAttention
# kernels.  If a standalone flash-attn build is needed, install it on a
# CUDA-capable host:  pip install flash-attn --no-build-isolation
RUN conda run -n loka pip install \
    verl \
    vllm \
    "ray[default]" \
    gymnasium \
    numba

# NCCL / CUDA / vLLM environment for 2-node FSDP2 + GRPO
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV NCCL_CROSS_NIC=1
ENV NCCL_IB_DISABLE=0
ENV TORCH_NCCL_AVOID_RECORD_STREAMS=1
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN
ENV NCCL_DEBUG=INFO

ENTRYPOINT ["conda", "run", "-n", "loka"]
CMD ["python3", "-m", "verl.trainer.main_ppo"]
