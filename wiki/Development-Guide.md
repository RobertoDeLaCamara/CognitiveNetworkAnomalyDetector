# Development Guide

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ (CPU)
- Root / sudo for live packet capture
- MLflow server + MinIO/S3 (optional, for remote tracking and model registry)

## Local Setup

```bash
git clone <repo>
cd cognitive-anomaly-detector
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Optionally set MLFLOW_TRACKING_URI, MLFLOW_S3_ENDPOINT_URL, AWS_* etc. in .env
```

## Training Workflow

```bash
# 1. Generate synthetic baseline
python scripts/generate_synthetic_data.py
# Creates: data/training/synthetic_baseline.csv

# 2. Train Isolation Forest (required)
python scripts/train_model.py \
  --from-file data/training/synthetic_baseline.csv \
  --version 5
# Creates: models/isolation_forest_v1.joblib, models/scaler_v1.joblib

# 3. Train LSTM Autoencoder (optional but recommended)
python scripts/train_lstm_model.py \
  --from-file data/training/synthetic_baseline.csv
# Creates: models/lstm_autoencoder.pt, models/lstm_config.json

# 4. Verify models
ls models/
# isolation_forest_v1.joblib  scaler_v1.joblib
# lstm_autoencoder.pt  lstm_config.json  lstm_scaler.joblib
```

`main.py` loads `isolation_forest_v{MODEL_VERSION}.joblib` / `scaler_v{MODEL_VERSION}.joblib` with `MODEL_VERSION = 5` (`src/config/ml_config.py`), so train with `--version 5` (or change the constant) for the detector to use the model.

## Running

```bash
# Live detection (root required)
sudo venv/bin/python main.py

# Specify interface and duration
sudo venv/bin/python main.py --interface eth0 --duration 300

# Dashboard only (no capture)
streamlit run dashboard.py --server.port 8501
# or
./run_dashboard.sh
```

## Docker Compose

```bash
# Full stack
docker-compose up -d detector

# One-off training
docker-compose run --rm trainer --duration 60
docker-compose run --rm trainer --from-file /app/data/training/synthetic_baseline.csv

# Dashboard
docker-compose up -d dashboard
```

## Testing

```bash
# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Single test
pytest tests/test_integration.py::TestEndToEndDetection::test_packet_to_features_to_detection -v

# Last failed
pytest tests/ --lf
```

## Key Environment Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `PACKET_WORKERS` | 2 | Worker thread count |
| `PACKET_QUEUE_SIZE` | 10000 | Bounded queue capacity |
| `ENSEMBLE_ANOMALY_THRESHOLD` | 0.6 | Alert firing threshold |
| `ENSEMBLE_WEIGHT_IF` | 0.4 | Isolation Forest weight |
| `ENSEMBLE_WEIGHT_LSTM` | 0.4 | LSTM weight |
| `ENSEMBLE_WEIGHT_RULES` | 0.2 | Rules weight |
| `THRESHOLD_MULTIPLIER` | 2.0 | Traffic spike multiplier |
| `ICMP_THRESHOLD` | 50 | ICMP flood count |
| `MLFLOW_TRACKING_URI` | (empty -> local SQLite) | Remote tracking server |
| `MLFLOW_ENABLE_REMOTE_LOADING` | false | Load model from the MLflow registry |
| `LSTM_SEQUENCE_LENGTH` | 50 | LSTM buffer length |
| `LSTM_EPOCHS` | 50 | Training epochs |

`MIN_PACKETS_FOR_ML` (10) and `MODEL_VERSION` (5) are constants in `src/config/ml_config.py`, not environment variables. See [docs/CONFIGURATION.md](../docs/CONFIGURATION.md) for the full list.

## Project Structure

```
src/
├── core/
│   ├── db_manager.py          SQLite anomalies table
│   ├── logger_setup.py        Logging configuration
│   ├── packet_queue.py        Bounded queue + worker threads
│   ├── resource_monitor.py    CPU/memory throttling
│   └── utils.py               Entropy calculation
├── detection/
│   ├── anomaly_detector.py    PacketAnalyzer — main class
│   ├── ensemble_scorer.py     Weighted fusion, EnsembleResult
│   └── payload_analyzer.py    Regex patterns + ReDoS timeout
├── ml/
│   ├── feature_extractor.py   18-feature per-IP extraction
│   ├── isolation_forest_detector.py
│   ├── lstm_autoencoder_detector.py
│   └── model_trainer.py       Training orchestration
└── config/
    ├── config.py              Detection thresholds
    ├── ml_config.py           Feature/model settings
    ├── mlflow_config.py       MLflow tracking + S3 settings
    ├── security_config.py     Input validation limits
    └── dashboard_config.py    Streamlit settings
```

## CI/CD (Jenkins)

`Jenkinsfile` on agent `agent-45`. Stages: Checkout → Build Image → Code Quality Checks (flake8 lint, pip-audit, in parallel) → Run Tests (pytest with coverage inside the built image) → SonarQube Analysis → Push to Registry.
