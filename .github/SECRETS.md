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

## Images Produced

| Image Tag                              | When Built             | Dockerfile          |
|----------------------------------------|------------------------|---------------------|
| `<registry>/<user>/loka:latest`        | Every push to `main`   | `docker/Dockerfile` |
| `<registry>/<user>/loka:<sha>`         | Every push to `main`   | `docker/Dockerfile` |
| `<registry>/<user>/loka:v1.0.0`        | Git tag `v1.0.0`       | `docker/Dockerfile` |
| `<registry>/<user>/loka:rl-train`      | Every push to `main`   | `docker/Dockerfile.rl` |
| `<registry>/<user>/loka:rl-<sha>`      | Every push to `main`   | `docker/Dockerfile.rl` |
