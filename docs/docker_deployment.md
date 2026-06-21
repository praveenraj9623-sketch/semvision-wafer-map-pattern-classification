# Docker Deployment

SemVision is packaged as a single, non-root Streamlit container. The image contains only the dashboard source, non-secret Streamlit theme configuration, precomputed demo samples, and JSON reports required by the UI.

## Prerequisites

- Install Docker Desktop for Windows.
- Start Docker Desktop and wait until its engine is running.
- Open PowerShell in the SemVision project directory.

## Build and run with Docker Compose

```powershell
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501).

## Run in detached mode

```powershell
docker compose up --build -d
```

## View logs

```powershell
docker compose logs -f
```

## Verify container health

```powershell
docker inspect --format='{{.State.Health.Status}}' semvision
```

The expected result is `healthy`. The image checks `http://localhost:8501/_stcore/health` from inside the container.

## Stop the application

```powershell
docker compose down
```

## Rebuild after code changes

```powershell
docker compose up --build
```

Add `-d` when you prefer detached mode.

## What the image contains

- `app.py`, `config.py`, and runtime modules under `src/`
- `.streamlit/config.toml`
- `data/demo/demo_samples.pkl`
- `artifacts/reports/*.json`
- Minimal dependencies from `requirements.app.txt`

The image intentionally excludes raw and processed WM-811K data, trained model binaries, MLflow runs, virtual environments, tests, caches, logs, audit output, secrets, and environment files. The Streamlit dashboard uses precomputed predictions and reports; it does not load TensorFlow models at runtime.

## Deployment scope

Docker makes SemVision portable and reproducible across machines with a compatible container runtime. Docker alone does not publish the application publicly. A Docker-compatible cloud host can run this image later and provide public routing, TLS, and any desired access controls.

Kubernetes, Helm, Terraform, databases, cloud credentials, and additional services are intentionally not included. SemVision is a single-service portfolio dashboard, so a standalone Docker image and one-service Compose file are the appropriate deployment boundary.
