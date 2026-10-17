# Setup Guide

## Local Install

### Prerequisites
- Python 3.10+ (Docker image uses 3.11)
- `libpcap` (for packet capture): `sudo apt install libpcap-dev` / `brew install libpcap`
- Root/sudo access for live traffic capture

### Steps

```bash
git clone https://github.com/RobertoDeLaCamara/CognitiveNetworkAnomalyDetector.git
cd CognitiveNetworkAnomalyDetector

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train and run

```bash
# Generate synthetic baseline data and train
python scripts/generate_synthetic_data.py
python scripts/train_model.py --from-file data/training/synthetic_baseline.csv --version 1

# Start detection (requires root for packet capture)
sudo venv/bin/python main.py
```

### Troubleshooting

**Permission denied on `sudo python`**
Always use the venv interpreter with sudo:
```bash
sudo venv/bin/python main.py     # correct
sudo python main.py              # wrong — uses system Python
```

**Too few training samples**
```bash
# Use synthetic data instead of live capture
python scripts/generate_synthetic_data.py
python scripts/train_model.py --from-file data/training/synthetic_baseline.csv
```

**Model not found**
```bash
ls models/    # check that isolation_forest_v5.joblib and scaler_v5.joblib exist
# The detector loads v{MODEL_VERSION} (MODEL_VERSION = 5 in src/config/ml_config.py).
# If missing, train with --version 5 (or change MODEL_VERSION)
```

---

## Docker

### Build and run

```bash
# Build image
docker-compose build

# Run dashboard → http://localhost:8501
docker-compose up -d dashboard

# Run detector (uses host networking for packet capture)
docker-compose up -d detector
docker-compose logs -f detector

# One-off training job
docker-compose run --rm trainer --from-file data/training/synthetic_baseline.csv --version 1
docker-compose run --rm trainer --duration 60 --version 1

# Stop everything
docker-compose down
```

### Notes
- The `detector` service uses `network_mode: host` to capture host traffic.
- The whole project directory is bind-mounted (`.:/app`), so `models/`, `data/`, `.mlflow/` and `anomalies.db` persist on the host.
- `docker-compose.yml` loads `.env` (`env_file`): copy `.env.example` to `.env` before starting, even if you do not use remote MLflow.
- Known issue: the `trainer` and `detector` services use entrypoints `train_model.py` / `main.py` relative to `/app`, but `train_model.py` lives in `scripts/`, so `docker-compose run --rm trainer ...` fails until the entrypoint is fixed.

---

## Remote MLflow + S3-compatible storage

The project supports a remote MLflow tracking server with S3-compatible artifact storage (MinIO). This is optional — without it, experiments are tracked locally in `.mlflow/mlflow.db` (SQLite).

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

```bash
MLFLOW_TRACKING_URI=http://<mlflow-server>:5050
MLFLOW_S3_ENDPOINT_URL=http://<minio-server>:9000
MLFLOW_S3_BUCKET=mlflow-artifacts
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

> MinIO uses port **9000** for the S3 API and **9001** for the web UI. `MLFLOW_S3_ENDPOINT_URL` must match `http(s)://host:port` exactly.

### Get MinIO credentials
1. Open the MinIO UI: `http://<minio-server>:9001`
2. Go to **Access Keys** → create or copy an existing key

### Test the connection

```bash
python scripts/test_mlflow_connection.py
```

The script tests the MLflow server, MinIO storage and an end-to-end run, and prints a PASS/FAIL line for each (`MLflow Server`, `MinIO Storage`, `End-to-End`; MinIO shows `NOT CONFIGURED` if no S3 endpoint is set).

### Use remote vs local

```bash
# Remote (default when .env is configured)
python scripts/train_model.py --duration 60 --version 1

# Force local for a single run
MLFLOW_TRACKING_URI="" python scripts/train_model.py --duration 60 --version 1

# View local experiments
mlflow ui --backend-store-uri sqlite:///$(pwd)/.mlflow/mlflow.db
# → http://localhost:5000
```

A non-empty `MLFLOW_TRACKING_URI` always takes precedence over the local SQLite default.

To import history from an existing file-store directory (`.mlflow/mlruns`) into the SQLite database:

```bash
mlflow migrate-filestore --source .mlflow --target sqlite:///$(pwd)/.mlflow/mlflow.db
```

The original `mlruns/` data is left untouched.

### Troubleshooting

| Error | Fix |
|---|---|
| Cannot connect to MLflow | Check server is up: `curl http://<server>:5050` |
| Cannot connect to S3 storage | Use port 9000, not 9001 |
| Bucket not found | Create the `mlflow-artifacts` bucket in the MinIO UI |
| Credentials error | Verify `.env` has no typos; re-run `test_mlflow_connection.py` |
