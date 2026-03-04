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

### Commit Messages

- Keep commit messages to **5–6 words** (concise, imperative mood)
- Examples: `Fix prompt seek overwriting timeline`, `Add asteroid belt rendering logic`

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

Training runs on a Slurm cluster (preferred) or Kubernetes. The base model
is `Qwen/Qwen2.5-7B-Instruct`, trained with GRPO via Verl on multi-GPU
nodes.

Key files:
- `configs/grpo_config.yaml` - GRPO training hyperparameters
- `configs/train_config.yaml` - Base model training hyperparameters
- `scripts/train_grpo_pyxis.slurm` - GRPO via Pyxis/Enroot container (primary)
- `scripts/train_grpo_native.slurm` - GRPO on native Slurm workers
- `scripts/train_grpo.slurm` - GRPO job submission (generic / module-load)
- `scripts/smoke_test_pyxis.slurm` - Container + GPU + import smoke test
- `scripts/e2e_test.sh` - Full pipeline e2e test (tiny data, 1 epoch)
- `scripts/generate_training_data.py` - Verl-format parquet data generator
- `k8s/training-job-rl.yaml` - Kubernetes GRPO training job (alternative)

#### Slurm + Pyxis/Enroot

The preferred deployment method uses **Pyxis** (Slurm spank plugin) with
**Enroot** to run containerized GPU jobs. This avoids dependency hell on
the host and guarantees reproducible environments.

**Container image syntax** — the `#` separates registry from path. Because
`#` is a comment character in `#SBATCH` directives, always pass the image
on the `sbatch` CLI:

```bash
sbatch --container-image="ghcr.io#user/repo:tag" \
       --container-mounts=/shared:/shared \
       --container-writable \
       --no-container-entrypoint \
       script.slurm
```

**Enroot credentials** must be placed on each worker node (not just the
login node) at `~/.config/enroot/.credentials`:

```
machine ghcr.io login <user> password <token>
```

**Topology** — if workers are missing from `scontrol show topology`, run
`scontrol reconfigure` to reload `/etc/slurm/topology.conf`. Drained nodes
can be resumed with `scontrol update NodeName=X State=RESUME`.

**Known environment quirks:**
- `ROCR_VISIBLE_DEVICES` (AMD ROCm) may be set by the base image; `unset` it
  before running Verl to avoid CUDA conflict
- `flash-attn` may not be installed; use `sdpa` attention instead
- Qwen2.5-7B has a large embedding layer; set
  `rollout.update_weights_bucket_megabytes=4096`
- Start Ray with `--include-dashboard=false` if dashboard deps are missing

**Workflow:**
1. Build and push the RL Docker image via GitHub Actions
2. Sync source code to shared NFS on the cluster
3. Generate training data (`scripts/generate_training_data.py`)
4. Run smoke test → e2e test → full training
5. Set `WANDB_API_KEY` to enable W&B logging

See `docs/DEPLOYMENT.md` (gitignored) for cluster-specific details,
credentials, and operational runbooks.

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
| `scripts/train_grpo_pyxis.slurm` | Primary Slurm GRPO training script |
| `scripts/e2e_test.sh` | End-to-end pipeline smoke test |
| `scripts/generate_training_data.py` | Verl-format parquet data generator |
| `docker/Dockerfile.rl` | RL training container image |
| `k8s/deployment.yaml` | Kubernetes inference deployment |
| `k8s/training-job-rl.yaml` | Kubernetes GRPO training job |
| `.githooks/pre-commit` | Pre-commit hook (lint + tests) |
| `.env` | Local credentials (gitignored) |
| `docs/DEPLOYMENT.md` | Cluster-specific ops notes (gitignored) |

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
