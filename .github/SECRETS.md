# GitHub Secrets Configuration

The CI/CD pipeline requires the following secrets to push Docker images.

## Required Secrets

Set these in your GitHub repo: **Settings → Secrets and variables → Actions → New repository secret**

| Secret Name       | Description                            | Example                    |
|-------------------|----------------------------------------|----------------------------|
| `DOCKER_REGISTRY` | Container registry hostname            | `ghcr.io` or `docker.io`  |
| `DOCKER_USERNAME` | Registry username                      | `your-github-username`     |
| `DOCKER_PASSWORD` | Registry token / password              | (see below)                |

## Setup by Registry

### Option A: GitHub Container Registry (ghcr.io) — Recommended

1. Go to GitHub → **Settings → Developer settings → Personal access tokens → Tokens (classic)**
2. Generate a token with scopes: `write:packages`, `read:packages`, `delete:packages`
3. Set secrets:
   - `DOCKER_REGISTRY` = `ghcr.io`
   - `DOCKER_USERNAME` = your GitHub username
   - `DOCKER_PASSWORD` = the PAT you just created

Images will be pushed to: `ghcr.io/<username>/loka:latest`

### Option B: Docker Hub

1. Go to Docker Hub → **Account Settings → Security → Access Tokens**
2. Create a token with **Read & Write** permissions
3. Set secrets:
   - `DOCKER_REGISTRY` = `docker.io`
   - `DOCKER_USERNAME` = your Docker Hub username
   - `DOCKER_PASSWORD` = the access token

Images will be pushed to: `docker.io/<username>/loka:latest`

## Additional Secrets (Training)

These are **not** used by CI/CD but are needed when running training jobs on the cluster.

| Secret / Env Var     | Where                       | Description                                    |
|----------------------|-----------------------------|------------------------------------------------|
| `WANDB_API_KEY`      | SLURM env / K8s secret      | Weights & Biases API key for experiment logging |
| `HF_TOKEN`           | SLURM env / K8s secret      | HuggingFace token for gated model downloads    |

### Weights & Biases (wandb)

1. Go to [wandb.ai/authorize](https://wandb.ai/authorize) and copy your API key.
2. **For SLURM:** Export it before submitting the job:
   ```bash
   export WANDB_API_KEY="your-key-here"
   export WANDB_PROJECT="orbital_rl"        # optional, defaults to orbital_rl
   sbatch scripts/train_grpo_native.slurm
   ```
3. **For Kubernetes:** Create the secret:
   ```bash
   kubectl -n loka create secret generic loka-secrets \
     --from-literal=wandb-api-key="your-key-here" \
     --from-literal=hf-token="your-hf-token"
   ```
4. **For GitHub Actions** (if you add wandb to CI tests later): add `WANDB_API_KEY` as a repository secret.

#### What gets logged

**Verl built-in metrics** (logged by `trainer.logger=wandb`):

| Metric | Description |
|--------|-------------|
| `train/policy_loss` | GRPO policy gradient loss |
| `train/kl_divergence` | KL between policy and reference model |
| `train/entropy` | Policy entropy (exploration measure) |
| `train/mean_reward` | Mean scalar reward across batch |
| `train/gradient_norm` | Global gradient norm (pre-clipping) |
| `train/learning_rate` | Current LR (cosine schedule) |
| `train/clip_fraction` | Fraction of samples clipped by PPO/GRPO |
| `train/response_length` | Mean token length of LLM responses |

**Loka domain-specific metrics** (logged via `src/loka/rl/metrics.py`):

| Metric | Description |
|--------|-------------|
| **Reward decomposition** | |
| `loka/reward/total_mean` | Mean total reward |
| `loka/reward/total_std` | Reward standard deviation |
| `loka/reward/total_min` | Min reward in batch |
| `loka/reward/total_max` | Max reward in batch |
| `loka/reward/format_mean` | Mean format compliance reward (0–0.2) |
| `loka/reward/physics_mean` | Mean physics simulation reward (0–0.8) |
| **Parsing quality** | |
| `loka/parse/xml_json_rate` | Fraction using perfect XML+JSON format |
| `loka/parse/bare_json_rate` | Fraction with bare JSON (no XML tags) |
| `loka/parse/regex_rate` | Fraction needing regex fallback |
| `loka/parse/fallback_rate` | Fraction with unparseable output (coast) |
| **Format compliance** | |
| `loka/format/think_rate` | Fraction of responses with `<think>` tags |
| `loka/format/action_rate` | Fraction of responses with `<action>` tags |
| `loka/format/perfect_rate` | Fraction with both `<think>` + valid XML+JSON |
| **Orbital mechanics** | |
| `loka/orbital/success_rate` | Mission success rate |
| `loka/orbital/dv_efficiency_mean` | ΔV efficiency (Hohmann / actual), 1.0 = optimal |
| `loka/orbital/dv_efficiency_std` | Efficiency std dev |
| `loka/orbital/dv_total_mean_kms` | Mean actual ΔV used (km/s) |
| `loka/orbital/final_a_mean_km` | Mean final semi-major axis (km) |
| `loka/orbital/final_e_mean` | Mean final eccentricity |
| `loka/orbital/mass_ratio_mean` | Mean remaining mass fraction |
| `loka/orbital/steps_mean` | Mean episode length |
| `loka/orbital/steps_max` | Max episode length in batch |
| **Throughput** | |
| `loka/response/length_mean` | Mean response character length |
| `loka/response/length_max` | Max response character length |
| `loka/throughput_samples_per_min` | Training throughput |

### Ray Dashboard Access

The Ray dashboard runs on port **8265** on the head node. All SLURM scripts start it with `--dashboard-host=0.0.0.0 --include-dashboard=true`.

#### What the dashboard shows

| Tab | Metrics |
|-----|---------|
| **Cluster** | Node count, CPU/GPU utilization, memory usage, object store |
| **Jobs** | Running/pending/finished jobs, duration, error logs |
| **Actors** | Verl workers (actor, rollout, ref), placement groups, restarts |
| **Logs** | Live streaming logs from all workers |
| **Metrics** | System metrics: GPU util%, GPU memory, network I/O, disk I/O |

#### SLURM — SSH tunnel

```bash
# The head node IP is printed in the job stdout:
#   "Ray Dashboard: http://<HEAD_IP>:8265"
# You can also find the head from SLURM:
HEAD_NODE=$(squeue -j <JOB_ID> -o "%N" -h | cut -d',' -f1)

# Open an SSH tunnel through the login node
ssh -L 8265:${HEAD_NODE}:8265 login-0
```

Then open [http://localhost:8265](http://localhost:8265) in your browser.

#### Kubernetes — port-forward

```bash
kubectl -n loka port-forward svc/loka-rl-head-svc 8265:8265
```

Then open [http://localhost:8265](http://localhost:8265) in your browser.

---

## Image Tagging (Automated)

Tags are generated automatically by `docker/metadata-action` — no manual versioning needed.

### How to release a new version

```bash
git tag v1.2.3
git push origin v1.2.3
```

That's it. The pipeline will produce all the tags below.

### Production image (`docker/Dockerfile`)

| You push...         | Image tags created                          |
|----------------------|---------------------------------------------|
| Commit to `main`     | `:main`, `:sha-abc1234`, `:latest`          |
| Tag `v1.2.3`         | `:1.2.3`, `:1.2`, `:1`, `:latest`           |
| Tag `v0.3.0-rc.1`    | `:0.3.0-rc.1`                               |

### RL training image (`docker/Dockerfile.rl`)

| You push...         | Image tags created                          |
|----------------------|---------------------------------------------|
| Commit to `main`     | `:rl-main`, `:rl-sha-abc1234`, `:rl-train`  |
| Tag `v1.2.3`         | `:rl-1.2.3`, `:rl-1.2`                      |

### Referencing images

In your SLURM scripts or K8s manifests, use:
- **Stable (recommended):** `<registry>/<user>/loka:rl-train` — always points to latest main
- **Pinned:** `<registry>/<user>/loka:rl-1.2.3` — locked to a specific release
- **Debugging:** `<registry>/<user>/loka:rl-sha-abc1234` — exact commit
