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

### Code Style

- Follow PEP 8 with 88-character line limit (Black formatter)
- Use type hints for all function signatures
- Docstrings in NumPy format

### Testing

```bash
# Run tests
pytest tests/

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

Training runs on SLURM cluster. Key files:
- `configs/train_config.yaml` - Hyperparameters
- `scripts/train.slurm` - Job submission script
- Kubeconfig: set via `KUBECONFIG` environment variable (not committed to repo)

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
```

## Important Files

| File | Purpose |
|------|---------|
| `src/loka/agent/base.py` | Main agent implementation |
| `src/loka/model/loka_model.py` | Model architecture |
| `src/loka/astro/coordinates.py` | Coordinate transformations |
| `configs/train_config.yaml` | Training configuration |
| `k8s/deployment.yaml` | Kubernetes deployment |

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
