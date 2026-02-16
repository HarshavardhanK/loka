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

- Python 3.11+
- Conda (via miniforge recommended)
- Docker (for containerized deployment)
- kubectl (for Kubernetes deployment)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/HarshavardhanK/loka.git
cd loka

# Create conda environment
conda env create -f environment.yml
conda activate loka

# Install in development mode
pip install -e ".[dev]"
```

### Docker

```bash
# Build the image
docker build -t loka:latest -f docker/Dockerfile .

# Run container
docker run -it --gpus all loka:latest
```

### Kubernetes Deployment

```bash
# Deploy to cluster
kubectl apply -f k8s/
```

## Training

Training is performed on a SLURM cluster. See [Training Guide](docs/training.md) for details.

```bash
# Submit training job
sbatch scripts/train.slurm
```

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
├── src/loka/           # Core package
│   ├── agent/          # Agentic reasoning
│   ├── astro/          # Astrophysics utilities
│   ├── model/          # LLM architecture
│   └── tools/          # Agent tools
├── configs/            # Training & inference configs
├── data/               # Datasets and ephemeris
├── docker/             # Docker configurations
├── k8s/                # Kubernetes manifests
├── models/             # Saved model checkpoints
├── notebooks/          # Jupyter notebooks
├── scripts/            # Training and utility scripts
└── tests/              # Test suite
```

## Dependencies

Core dependencies managed via conda/pip:

- **astropy**: Astronomical calculations and coordinate systems
- **jplephem**: JPL ephemeris access
- **scipy**: Numerical optimization
- **torch**: Deep learning framework
- **transformers**: Model architecture base

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT

## References

- [Astropy Documentation](https://docs.astropy.org/)
- [JPL HORIZONS](https://ssd.jpl.nasa.gov/horizons/)
- [SPICE Toolkit](https://naif.jpl.nasa.gov/naif/toolkit.html)
