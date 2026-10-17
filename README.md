# Cognitive Anomaly Detector

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

Network anomaly detection with triple-engine ensemble: Isolation Forest (40%) + LSTM Autoencoder/PyTorch (40%) + rule-based engine (20%). Scapy packet capture, 18 per-IP feature extraction, MLflow tracking, Streamlit dashboard.

## Architecture

```
[Scapy capture] --> [Packet queue] --> [Feature extraction (18 features/IP)]
                                              |
                          +-------------------+-------------------+
                     Rule-based        Isolation Forest     LSTM Autoencoder
                      (20%)               (40%)                (40%)
                          +-------------------+-------------------+
                                       Ensemble scorer
                                     (threshold: 0.6)
                                              |
                                       [Alert / SQLite]
```

### Detection Engines

| Engine | Type | Input | Technique |
|--------|------|-------|-----------|
| **Rule-based** | Heuristic | Raw packets | Traffic spikes, ICMP floods, port scans, payload patterns (SQLi, XSS, shell) |
| **Isolation Forest** | Unsupervised ML | 18 per-IP features | Statistical outlier detection on feature vectors |
| **LSTM Autoencoder** | Deep Learning | Sliding windows | Sequential anomaly detection via reconstruction error |

### Feature Extraction (18 features per IP)

Order and names follow `FEATURE_NAMES` in `src/config/ml_config.py`:

- **Statistical (6)**: `packets_per_second`, `bytes_per_second`, `avg_packet_size`, `packet_size_variance`, `total_packets`, `total_bytes`
- **Temporal (4)**: `inter_arrival_mean`, `inter_arrival_std`, `burst_rate`, `session_duration`
- **Protocol (3)**: `tcp_ratio`, `udp_ratio`, `icmp_ratio`
- **Port (2)**: `unique_ports`, `uncommon_port_ratio`
- **Payload (3)**: `payload_entropy`, `avg_payload_size`, `payload_size_variance`

## Quick Start

```bash
# 1. Install
git clone https://github.com/RobertoDeLaCamara/CognitiveNetworkAnomalyDetector.git
cd CognitiveNetworkAnomalyDetector
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Generate synthetic training data
python scripts/generate_synthetic_data.py

# 3. Train models
python scripts/train_model.py --from-file data/training/synthetic_baseline.csv --version 1
python scripts/train_lstm_model.py --from-file data/training/synthetic_baseline.csv

# 4. Run detection (requires root for packet capture)
sudo venv/bin/python main.py

# 5. Dashboard (optional)
./run_dashboard.sh          # --> http://localhost:8501
```

> `main.py` loads `models/isolation_forest_v{MODEL_VERSION}.joblib` and `models/scaler_v{MODEL_VERSION}.joblib`,
> where `MODEL_VERSION = 5` in `src/config/ml_config.py`. Train with `--version 5` (or change `MODEL_VERSION`)
> so the detector picks the model up; otherwise ML detection is disabled and only rules run.

For Docker, full install options, and remote MLflow setup see [SETUP.md](docs/SETUP.md).

## Project Structure

```
cognitive-anomaly-detector/
├── src/
│   ├── core/                   # Infrastructure (db_manager, logger_setup, packet_queue, resource_monitor, utils)
│   ├── detection/              # Triple detection engine + ensemble scorer
│   │   ├── anomaly_detector.py # PacketAnalyzer: rule checks + orchestration of IF/LSTM/ensemble
│   │   ├── payload_analyzer.py # Malicious payload pattern matching
│   │   └── ensemble_scorer.py  # Weighted confidence fusion
│   ├── ml/                     # Feature extraction, model training, detectors
│   │   ├── feature_extractor.py          # 18-feature pipeline
│   │   ├── isolation_forest_detector.py  # Isolation Forest wrapper
│   │   ├── lstm_autoencoder_detector.py  # LSTM Autoencoder (PyTorch) + training
│   │   └── model_trainer.py              # IF training with MLflow tracking
│   ├── config/                 # config, ml_config, mlflow_config, security_config, dashboard_config
│   └── dashboard/              # Dashboard-specific logic
├── scripts/
│   ├── generate_synthetic_data.py  # Synthetic training data generation
│   ├── train_model.py              # Isolation Forest training
│   ├── train_lstm_model.py         # LSTM Autoencoder training
│   ├── auto_train_lstm.sh          # Scheduled LSTM retraining
│   ├── setup_mlflow.py             # Initialise local MLflow store
│   ├── test_mlflow_connection.py   # Check remote MLflow + S3 connectivity
│   ├── promote_latest.py           # Promote latest registered model version
│   └── inject_synthetic_traffic.py # Generate traffic that triggers alerts
├── tests/                      # Organized by component (core, detection, ml) + test_integration.py
├── models/                     # Trained model files (.joblib, .pt)
├── data/                       # Training data
├── main.py                     # Detection entry point (sudo required)
├── dashboard.py                # Streamlit dashboard
├── run_dashboard.sh            # Dashboard launcher
├── docker-compose.yml
├── Dockerfile
├── Jenkinsfile                 # CI pipeline
└── requirements.txt
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ENSEMBLE_WEIGHT_IF` | Isolation Forest weight | `0.4` |
| `ENSEMBLE_WEIGHT_LSTM` | LSTM Autoencoder weight | `0.4` |
| `ENSEMBLE_WEIGHT_RULES` | Rule engine weight | `0.2` |
| `ENSEMBLE_ANOMALY_THRESHOLD` | Confidence threshold for alerts | `0.6` |
| `MLFLOW_TRACKING_URI` | MLflow tracking URI | `sqlite:///.mlflow/mlflow.db` (local; absolute path at runtime) |
| `MLFLOW_ENABLED` | Enable/disable MLflow tracking (`true`/`1`/`yes`) | `true` |

The capture interface is a CLI flag (`main.py --interface IFACE`), not an environment variable. The experiment name is set with `train_model.py --experiment-name` (default `cognitive-anomaly-detector-v2`). Alerts are stored in `anomalies.db` at the project root (not configurable).

See [CONFIGURATION.md](docs/CONFIGURATION.md) for all options.

### Experiment tracking backend

MLflow 3.x rejects the file-store backend, so local tracking uses SQLite
(`sqlite:///.mlflow/mlflow.db`). A non-empty `MLFLOW_TRACKING_URI` overrides
this default (for example a remote `http://` server); an empty value falls
back to SQLite.

To keep history from an existing `.mlflow/mlruns` file store, migrate it once:

```bash
mlflow migrate-filestore --source .mlflow --target sqlite:///.mlflow/mlflow.db
```

The old `mlruns/` data is not deleted.

## Training Options

```bash
# From synthetic data (fast, no sudo)
python scripts/train_model.py --from-file data/training/synthetic_baseline.csv --version 1

# From live traffic
sudo venv/bin/python scripts/train_model.py --duration 60 --version 1

# LSTM Autoencoder
python scripts/train_lstm_model.py --from-file data/training/synthetic_baseline.csv

# Disable MLflow for a run
python scripts/train_model.py --from-file data.csv --no-mlflow
```

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

## CI/CD

Jenkins pipeline (`Jenkinsfile`, agent label `agent-45`):
- **Build** Docker image
- **Lint** (flake8) + security scan (pip-audit)
- **Test** with coverage
- **SonarQube** analysis
- **Push** image to the private registry

## Documentation

- [SETUP.md](docs/SETUP.md) -- Installation, Docker, remote MLflow setup
- [CONFIGURATION.md](docs/CONFIGURATION.md) -- All config options and environment variables
- [DASHBOARD.md](docs/DASHBOARD.md) -- Streamlit dashboard guide
- [SECURITY.md](docs/SECURITY.md) -- Security notes and hardening applied

## Requirements

- Python 3.10+ (required by the pinned `mlflow` 3.x, `torch` and `streamlit`); the Docker image and CI use 3.11
- Root/sudo for packet capture (`main.py`, live training)
- Optional: remote MLflow server + S3-compatible storage for experiment tracking
