# Docker Setup Guide

This guide explains how to run the Cognitive Anomaly Detector using Docker and Docker Compose. This ensures a consistent environment and simplifies deployment.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed

## Quick Start

### 1. Build Containers

Build the Docker image for all services:

```bash
docker-compose build
```

### 2. Run Services

You can run services individually or together.

#### Run Dashboard
Starts the Streamlit dashboard on port 8501.

```bash
docker-compose up -d dashboard
```
Access at: [http://localhost:8501](http://localhost:8501)

#### Run Anomaly Detector
Starts the real-time packet monitoring.
*Note: Requires privileged access for packet capture.*

```bash
docker-compose up -d detector
```
View logs:
```bash
docker-compose logs -f detector
```

#### Run Model Trainer
Runs a one-off training job.

```bash
# Train on 60 seconds of synthetic traffic
docker-compose run --rm trainer --duration 60 --version 1

# Train on existing file
docker-compose run --rm trainer --from-file data/training/baseline.csv --version 1
```

### 3. Manage Environment

Stop all running services:
```bash
docker-compose down
```

## Configuration

### Environment Variables
The services use the `.env` file for configuration. Ensure you have created it from the example:

```bash
cp .env.example .env
# Edit .env to add MLflow/MinIO credentials if needed
```

### Volumes
- `./models`: Persists trained models.
- `./data`: Persists training data.
- `./.mlflow`: Persists local MLflow experiments (if remote not configured).

## Architecture

The `docker-compose.yml` defines three services sharing the same base image:

1.  **trainer**: Execute `train_model.py`. Ephemeral container.
2.  **detector**: Execute `main.py`. Uses `network_mode: host` to capture host traffic.
3.  **dashboard**: Execute `run_dashboard.sh`. Exposes web UI.
