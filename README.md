# Cognitive Anomaly Detector

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

Network anomaly detection with triple-engine ensemble: Isolation Forest (40%) + LSTM Autoencoder/PyTorch (40%) + rule-based engine (20%). Scapy packet capture, 18 per-IP feature extraction, ML Tracking tracking, Streamlit dashboard.

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

- **Statistical**: packet count, byte volume, mean/std packet size
- **Temporal**: inter-arrival time stats, burst detection
- **Protocol**: TCP/UDP/ICMP ratios, SYN/FIN/RST flag counts
- **Port**: unique destination ports, port scan indicators
- **Payload**: average payload size, entropy, pattern match scores

## Quick Start

```bash
# 1. Install
git clone https://github.com/RobertoDeLaCamara/CognitiveNetworkAnomalyDetector.git
cd cognitive-anomaly-detector
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

For Docker, full install options, and ML Tracking remote setup see [SETUP.md](docs/SETUP.md).

## Project Structure

```
cognitive-anomaly-detector/
├── src/
│   ├── core/                   # Infrastructure (DB, Queue, Logging)
│   ├── detection/              # Triple detection engine + ensemble scorer
│   │   ├── rule_engine.py      # Pattern matching, threshold rules
│   │   ├── ml_detector.py      # Isolation Forest wrapper
│   │   ├── lstm_detector.py    # LSTM Autoencoder wrapper
│   │   └── ensemble.py         # Weighted confidence fusion
│   ├── ml/                     # Feature extraction, model training, detectors
│   │   ├── feature_extractor.py  # 18-feature pipeline
│   │   ├── model_trainer.py    # IF training with ML Tracking tracking
│   │   └── lstm_trainer.py     # LSTM Autoencoder training
│   ├── config/                 # Rule thresholds and ML settings
│   └── dashboard/              # Dashboard-specific logic
├── scripts/
│   ├── generate_synthetic_data.py  # Synthetic training data generation
│   ├── train_model.py              # Isolation Forest training
│   ├── train_lstm_model.py         # LSTM Autoencoder training
│   └── auto_train_lstm.sh          # Scheduled retraining
├── tests/                      # Organized by component (core, detection, ml)
├── models/                     # Trained model files (.pkl, .pt)
├── data/                       # Training data
├── main.py                     # Detection entry point (sudo required)
├── dashboard.py                # Streamlit dashboard
├── docker-compose.yml
├── CI/CDfile                 # CI pipeline
└── requirements.txt
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `CAPTURE_INTERFACE` | Network interface for Scapy | `eth0` |
| `ENSEMBLE_IF_WEIGHT` | Isolation Forest weight | `0.4` |
| `ENSEMBLE_LSTM_WEIGHT` | LSTM Autoencoder weight | `0.4` |
| `ENSEMBLE_RULES_WEIGHT` | Rule engine weight | `0.2` |
| `ALERT_THRESHOLD` | Confidence threshold for alerts | `0.6` |
| `ML Tracking_TRACKING_URI` | ML Tracking server URL | `None` (local) |
| `ML Tracking_EXPERIMENT_NAME` | Experiment name | `anomaly-detection` |
| `DB_PATH` | SQLite database path | `data/alerts.db` |

See [CONFIGURATION.md](docs/CONFIGURATION.md) for all options.

## Training Options

```bash
# From synthetic data (fast, no sudo)
python scripts/train_model.py --from-file data/training/synthetic_baseline.csv --version 1

# From live traffic
sudo venv/bin/python scripts/train_model.py --duration 60 --version 1

# LSTM Autoencoder
python scripts/train_lstm_model.py --from-file data/training/synthetic_baseline.csv

# Disable ML Tracking for a run
python scripts/train_model.py --from-file data.csv --no-ML Tracking
```

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

## CI/CD

CI/CD multibranch pipeline (Git Server SCM source):
- **Build** Docker image
- **Lint** + security scan
- **Test** with coverage
- **Quality Analysis** analysis

## Documentation

- [SETUP.md](docs/SETUP.md) -- Installation, Docker, ML Tracking remote setup
- [CONFIGURATION.md](docs/CONFIGURATION.md) -- All config options and environment variables
- [DASHBOARD.md](docs/DASHBOARD.md) -- Streamlit dashboard guide
- [SECURITY.md](docs/SECURITY.md) -- Security notes and hardening applied

## Requirements

- Python 3.8+
- Root/sudo for packet capture (`main.py`, live training)
- Optional: remote ML Tracking server + S3-compatible storage for experiment tracking
