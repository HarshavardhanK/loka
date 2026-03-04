# Loka

**Loka** (Sanskrit for "world/realm") is an agentic AI model for astrophysics, designed to navigate celestial bodies and plan optimal trajectories across the solar system.

## Overview

Loka is a specialized small language model (SLM) trained for:

- **Celestial Navigation**: Understanding and computing positions of planets, moons, asteroids, and other solar system bodies
- **Trajectory Planning**: Generating fuel-efficient transfer orbits (Hohmann transfers, gravity assists, low-thrust trajectories)
- **Mission Planning**: End-to-end mission design from Earth departure to target arrival
- **Ephemeris Queries**: Real-time position and velocity calculations using JPL ephemeris data

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Loka Agent                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   LLM Core  │  │   Astropy   │  │  Trajectory Planner │  │
│  │  (Trained)  │◄─┤  Integration│◄─┤    (Optimizer)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  JPL SPICE  │  │   Orbital   │  │   Physics Engine    │  │
│  │  Ephemeris  │  │  Mechanics  │  │   (N-body Sim)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Agentic Reasoning**: Multi-step planning with tool use for complex navigation scenarios
- **Astropy Integration**: Leverages astropy for coordinate transformations, time handling, and astronomical calculations
- **SPICE Kernel Support**: Direct access to NASA/JPL ephemeris data
- **Multi-body Optimization**: Accounts for gravitational influences from multiple bodies
- **Delta-V Budgeting**: Automatic propellant requirement calculations

## Installation

### Prerequisites

- Python 3.10+ (3.11+ recommended for local dev)
- Docker (for containerized training/deployment)
- Access to a Slurm cluster with Pyxis/Enroot (for GPU training)

### Quick Start

```bash
git clone https://github.com/HarshavardhanK/loka.git
cd loka
pip install -e ".[dev]"

# Set up pre-commit hook
./scripts/install-hooks.sh
```

### Docker (RL Training Image)

The RL training image is built via GitHub Actions (`docker-train.yml`) and
published to GHCR. It includes PyTorch, vLLM, Verl, Ray, and all
scientific dependencies on top of the Slurm worker base image.

```bash
# Trigger the build via GitHub Actions
gh workflow run docker-train.yml

# Or build locally
docker build -t loka:rl -f docker/Dockerfile.rl .
```

## Training

Loka uses **GRPO** (Group Relative Policy Optimization) via
[Verl](https://github.com/volcengine/verl) to train `Qwen/Qwen2.5-7B-Instruct`
on a custom orbital mechanics environment (`OrbitalTransferEnv`).

### Slurm + Pyxis (Primary)

The preferred method runs containerized training on a Slurm cluster using
Pyxis/Enroot. The `#` in the image URI separates registry from path and
must be passed on the CLI (not in `#SBATCH` directives):

```bash
# 1. Generate training data (runs inside the container)
sbatch --container-image="ghcr.io#harshavardhank/loka:rl-fix-docker-rl-base-image" \
       --container-mounts=/data:/data --container-writable --no-container-entrypoint \
       --wrap="python3 scripts/generate_training_data.py --n-train 10000 --n-val 1000"

# 2. Smoke test (GPU + imports + reward function)
sbatch --container-image="ghcr.io#harshavardhank/loka:rl-fix-docker-rl-base-image" \
       scripts/smoke_test_pyxis.slurm

# 3. End-to-end test (tiny data, 1 epoch, no W&B)
sbatch --container-image="ghcr.io#harshavardhank/loka:rl-fix-docker-rl-base-image" \
       --gres=gpu:h100:8 --mem=900G --nodes=1 --cpus-per-task=104 \
       scripts/e2e_test.sh

# 4. Full training (with W&B)
WANDB_API_KEY=<key> sbatch \
       --container-image="ghcr.io#harshavardhank/loka:rl-fix-docker-rl-base-image" \
       scripts/train_grpo_pyxis.slurm
```

### Kubernetes (Alternative)

```bash
kubectl apply -f k8s/training-job-rl.yaml
```

### W&B Metrics

When `WANDB_API_KEY` is set, training logs comprehensive metrics to
Weights & Biases including reward decomposition, GRPO group variance,
termination analysis, action distributions, and curriculum stage breakdowns.
See `src/loka/rl/metrics.py` for details.

## Usage

```python
from loka import LokaAgent

# Initialize agent
agent = LokaAgent.from_pretrained("loka-v1")

# Plan a mission to Mars
result = agent.plan_mission(
    origin="Earth",
    destination="Mars", 
    departure_window=("2026-07-01", "2026-09-30"),
    optimize_for="fuel"
)

print(result.trajectory)
print(f"Delta-V: {result.delta_v} km/s")
print(f"Transfer time: {result.transfer_time} days")
```

## Project Structure

```
loka/
├── src/loka/
│   ├── agent/          # Agentic reasoning
│   ├── astro/          # Astrophysics utilities
│   ├── envs/           # Gymnasium environments (OrbitalTransferEnv)
│   ├── model/          # LLM architecture
│   ├── rl/             # RL training (bridge, reward, metrics, checkpoint, curriculum)
│   └── tools/          # Agent tools
├── configs/            # Training & inference configs
├── docker/             # Dockerfiles (Dockerfile.rl for training)
├── k8s/                # Kubernetes manifests
├── scripts/            # Slurm jobs, data gen, smoke tests
│   ├── train_grpo_pyxis.slurm   # Primary training script
│   ├── smoke_test_pyxis.slurm   # Container smoke test
│   ├── e2e_test.sh              # Full pipeline e2e test
│   └── generate_training_data.py
├── .github/workflows/  # CI/CD and Docker build
└── tests/              # Test suite
```

## Dependencies

Core dependencies managed via pip (`pyproject.toml`):

- **astropy** / **jplephem** / **scipy** — Astronomical calculations
- **torch** / **transformers** — Model architecture and training
- **vllm** — Fast LLM inference (rollouts)
- **verl** — GRPO / PPO trainer
- **ray** — Distributed orchestration
- **gymnasium** / **numba** — RL environment
- **wandb** — Experiment tracking

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT

## References

- [Astropy Documentation](https://docs.astropy.org/)
- [JPL HORIZONS](https://ssd.jpl.nasa.gov/horizons/)
- [SPICE Toolkit](https://naif.jpl.nasa.gov/naif/toolkit.html)
