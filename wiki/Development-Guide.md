# Development Guide

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ (CPU)
- Root / sudo for live packet capture
- MLflow + MinIO (optional, for remote model registry)

## Local Setup

```bash
git clone <repo>
cd cognitive-anomaly-detector
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set CAPTURE_INTERFACE, MLFLOW_TRACKING_URI, etc. in .env
```

## Training Workflow

```bash
# 1. Generate synthetic baseline
python scripts/generate_synthetic_data.py
# Creates: data/training/synthetic_baseline.csv

# 2. Train Isolation Forest (required)
python scripts/train_model.py \
  --from-file data/training/synthetic_baseline.csv \
  --version 1
# Creates: models/isolation_forest_v1.joblib, models/scaler_v1.joblib

# 3. Train LSTM Autoencoder (optional but recommended)
python scripts/train_lstm_model.py \
  --from-file data/training/synthetic_baseline.csv
# Creates: models/lstm_autoencoder.pt, models/lstm_config.json

# 4. Verify models
ls models/
# isolation_forest_v1.joblib  scaler_v1.joblib
# lstm_autoencoder.pt  lstm_config.json
```

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
pytest tests/test_integration.py::TestEndToEndDetection::test_normal_traffic_detection -v

# Last failed
pytest tests/ --lf
```

## Key Environment Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `CAPTURE_INTERFACE` | eth0 | Network interface |
| `PACKET_WORKERS` | 2 | Worker thread count |
| `PACKET_QUEUE_SIZE` | 10000 | Bounded queue capacity |
| `MIN_PACKETS_FOR_ML` | 10 | Min per-IP packets before ML runs |
| `ENSEMBLE_ANOMALY_THRESHOLD` | 0.6 | Alert firing threshold |
| `ENSEMBLE_WEIGHT_IF` | 0.4 | Isolation Forest weight |
| `ENSEMBLE_WEIGHT_LSTM` | 0.4 | LSTM weight |
| `ENSEMBLE_WEIGHT_RULES` | 0.2 | Rules weight |
| `THRESHOLD_MULTIPLIER` | 2.0 | Traffic spike multiplier |
| `ICMP_THRESHOLD` | 50 | ICMP flood count |
| `MODEL_VERSION` | 5 | Active model version number |
| `MLFLOW_TRACKING_URI` | (empty) | Enable remote registry |
| `MLFLOW_ENABLE_REMOTE_LOADING` | false | Load model from MLflow |
| `LSTM_SEQUENCE_LENGTH` | 50 | LSTM buffer length |
| `LSTM_EPOCHS` | 50 | Training epochs |

## Project Structure

```
src/
├── core/
│   ├── db_manager.py          SQLite anomalies table
│   ├── logger_setup.py        Logging configuration
│   ├── packet_queue.py        Async bounded queue + workers
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
    ├── mlflow_config.py       MLflow tracking
    ├── security_config.py     Input validation limits
    └── dashboard_config.py    Streamlit settings
```

## CI/CD (Jenkins)

Stages: Setup → Run Tests → Stop (no Docker push for this project). Virtual environment managed per-build; PID-based process management for mock services during integration tests.
