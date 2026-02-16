# AGENTS.md - Loka Project Guidelines

## Project Overview

Loka is an agentic AI model for astrophysics navigation and trajectory planning. This document provides context for AI agents working on this codebase.

## Architecture

### Core Components

1. **LokaAgent** (`src/loka/agent/`) - Main agentic interface
   - Orchestrates multi-step reasoning
   - Manages tool calls and context
   - Handles mission planning workflows

2. **Model** (`src/loka/model/`) - Small LLM architecture
   - Custom transformer optimized for numerical reasoning
   - Fine-tuned on astrophysics corpus
   - Supports tool-use tokens

3. **Astro Module** (`src/loka/astro/`) - Astrophysics computations
   - Wraps astropy for coordinate systems
   - Ephemeris data access via jplephem
   - Orbital mechanics calculations

4. **Tools** (`src/loka/tools/`) - Agent tool implementations
   - `ephemeris_tool`: Query celestial body positions
   - `trajectory_tool`: Compute transfer orbits
   - `delta_v_tool`: Calculate fuel requirements

### Key Patterns

- **Tool-augmented generation**: Model outputs special tokens to invoke tools
- **Chain-of-thought**: Multi-step reasoning with intermediate verification
- **Numerical grounding**: All outputs validated against physical constraints

## Development Guidelines

### Pre-commit Hook (MANDATORY)

A pre-commit hook runs **linting, unit tests, and integration tests** before
every commit. All agents and developers MUST have this hook active.

```bash
# One-time setup (run after cloning)
./scripts/install-hooks.sh
```

This sets `git core.hooksPath` to `.githooks/` so the tracked hook is used
automatically. The hook runs:

1. `ruff check src/ tests/` — lint (fails fast)
2. `pytest tests/test_coordinates.py tests/test_tools.py tests/test_orbital_env.py` — unit tests
3. `pytest tests/test_integration.py` — integration tests

If any step fails, the commit is **aborted**. Fix the issues first.

To skip in an emergency (use sparingly):
```bash
SKIP_PRE_COMMIT=1 git commit -m "emergency fix"
```

### Code Style

- Follow PEP 8 with 88-character line limit (ruff / Black formatter)
- Use type hints for all function signatures
- Docstrings in NumPy format
- Use `X | None` instead of `Optional[X]` (PEP 604)
- Use `list[T]` / `dict[K, V]` instead of `List[T]` / `Dict[K, V]` (PEP 585)
- Physics variable names (`R_E`, `G`, `MU`) are exempt from naming rules

### Linting

```bash
# Check
ruff check src/ tests/

# Auto-fix
ruff check src/ tests/ --fix

# Config is in pyproject.toml [tool.ruff.lint]
```

### Testing

```bash
# Run all tests
pytest tests/

# Just unit tests (fast — what the pre-commit hook runs)
pytest tests/test_coordinates.py tests/test_tools.py tests/test_orbital_env.py -v

# Integration tests
pytest tests/test_integration.py -v

# With coverage
pytest --cov=loka tests/
```

### Common Tasks

#### Adding a New Tool

1. Create tool class in `src/loka/tools/`
2. Implement `execute()` method with typed inputs/outputs
3. Register in `ToolRegistry`
4. Add corresponding tokens to tokenizer

#### Updating Ephemeris Data

```bash
# Download latest SPK files
python scripts/download_ephemeris.py --target de440s
```

#### Training

Training runs on SLURM cluster or Kubernetes. Key files:
- `configs/grpo_config.yaml` - GRPO training hyperparameters
- `configs/train_config.yaml` - Base model training hyperparameters
- `scripts/train_grpo.slurm` - GRPO job submission (generic)
- `scripts/train_grpo_native.slurm` - GRPO on native SLURM workers
- `scripts/train_grpo_pyxis.slurm` - GRPO via Pyxis container
- `k8s/training-job-rl.yaml` - Kubernetes GRPO training job
- Kubeconfig: set via `KUBECONFIG` environment variable (not committed to repo)
- Model: `Qwen/Qwen2.5-7B-Instruct`

### Dependencies

Core scientific stack:
- `astropy>=7.0` - Astronomical calculations
- `jplephem>=2.18` - JPL ephemeris
- `scipy>=1.13` - Optimization
- `poliastro` - Orbital mechanics (optional)

ML stack:
- `torch>=2.0` - Training framework
- `transformers>=4.40` - Model architecture
- `datasets` - Data loading
- `accelerate` - Distributed training

### Environment Variables

```bash
LOKA_DATA_DIR       # Path to ephemeris and training data
LOKA_MODEL_DIR      # Path to model checkpoints
LOKA_CACHE_DIR      # Caching directory
JPL_EPHEMERIS_PATH  # Path to SPK kernel files
WANDB_API_KEY       # Weights & Biases API key (training only)
HF_TOKEN            # HuggingFace token for gated models (training only)
```

Local credentials are stored in `.env` (gitignored). See `.github/SECRETS.md`
for full secret setup instructions.

## Important Files

| File | Purpose |
|------|---------|
| `src/loka/agent/base.py` | Main agent implementation |
| `src/loka/model/loka_model.py` | Model architecture |
| `src/loka/astro/coordinates.py` | Coordinate transformations |
| `src/loka/rl/bridge.py` | LLM ↔ environment bridge (system prompt, action parsing) |
| `src/loka/rl/reward.py` | Verl-compatible reward function |
| `src/loka/rl/metrics.py` | Domain-specific wandb metrics tracker |
| `src/loka/rl/checkpoint.py` | Hybrid checkpoint manager (last-N + best-K) |
| `configs/grpo_config.yaml` | GRPO training configuration |
| `configs/train_config.yaml` | Base model training configuration |
| `k8s/deployment.yaml` | Kubernetes inference deployment |
| `k8s/training-job-rl.yaml` | Kubernetes GRPO training job |
| `.githooks/pre-commit` | Pre-commit hook (lint + tests) |
| `.env` | Local credentials (gitignored) |

## Physical Constants

The codebase uses astropy constants. Key values:
- `G` - Gravitational constant
- `M_sun`, `M_earth`, etc. - Body masses
- `au` - Astronomical unit

## Coordinate Systems

Primary coordinate frames (via astropy):
- **ICRS**: International Celestial Reference System (default)
- **GCRS**: Geocentric Celestial Reference System
- **Heliocentric**: Sun-centered for trajectory planning

## Common Pitfalls

1. **Time handling**: Always use astropy `Time` objects, not raw datetime
2. **Units**: Use astropy units; never assume implicit units
3. **Frame conversions**: Specify epoch when converting between frames
4. **Numerical precision**: Use `np.float64` for orbital calculations

## References

- [Astropy Coordinates](https://docs.astropy.org/en/stable/coordinates/)
- [JPL SPICE](https://naif.jpl.nasa.gov/naif/)
- [Orbital Mechanics for Engineering Students](https://www.elsevier.com/books/orbital-mechanics-for-engineering-students/curtis/978-0-08-102133-0)
