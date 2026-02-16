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
