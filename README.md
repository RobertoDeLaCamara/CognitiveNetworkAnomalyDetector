# Cognitive Anomaly Detector

Network anomaly detection using a triple-engine approach: rule-based heuristics, Isolation Forest, and LSTM Autoencoder. All three engines feed into a weighted ensemble that produces a single 0–1 confidence score.

## Architecture

```
[Scapy capture] → [Packet queue] → [Feature extraction (18 features/IP)]
                                          │
                          ┌───────────────┼───────────────┐
                     Rule-based     Isolation Forest   LSTM Autoencoder
                          └───────────────┼───────────────┘
                                   Ensemble scorer
                                          │
                                   [Alert / SQLite]
```

Detection engines:
- **Rule-based** — traffic spikes, ICMP floods, port scans, payload pattern matching (SQLi, XSS, shell commands)
- **Isolation Forest** — unsupervised ML on 18 per-IP features (statistical, temporal, protocol, port, payload)
- **LSTM Autoencoder** — sequential anomaly detection via reconstruction error on sliding windows

Default ensemble weights: IF 40%, LSTM 40%, rules 20%. Alerts fire above a 0.6 confidence threshold.

## Quick Start

```bash
# 1. Install
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Train (synthetic data, no root needed)
python scripts/generate_synthetic_data.py
python scripts/train_model.py --from-file data/training/synthetic_baseline.csv --version 1

# 3. Detect
sudo venv/bin/python main.py

# 4. Dashboard (optional)
./run_dashboard.sh          # → http://localhost:8501
```

For Docker, full install options, and MLflow remote setup see [SETUP.md](SETUP.md).

## Project Structure

```
├── src/
│   ├── core/              # Infrastructure (DB, Queue, Logging)
│   ├── detection/         # Triple detection engine + ensemble
│   ├── ml/                # Machine Learning (Feature extraction, detectors, trainer)
│   ├── config/            # Rule thresholds and ML settings
│   └── dashboard/         # Dashboard-specific logic
├── scripts/               # Training & utility scripts
├── tests/                 # Organized by component (core, detection, ml)
├── models/                # Trained model files
├── data/                  # Training data
├── main.py                # Detection entry point
├── dashboard.py           # Streamlit dashboard
└── docker-compose.yml
```

## Training Options

```bash
# From synthetic data (fast, no sudo)
python scripts/train_model.py --from-file data/training/synthetic_baseline.csv --version 1

# From live traffic
sudo venv/bin/python scripts/train_model.py --duration 60 --version 1

# LSTM Autoencoder
python scripts/train_lstm_model.py --from-file data/training/synthetic_baseline.csv

# Auto-train LSTM daily (example)
sudo scripts/auto_train_lstm.sh

# Disable MLflow for a run
python scripts/train_model.py --from-file data.csv --no-mlflow
```

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

## Documentation

- [SETUP.md](SETUP.md) — Installation, Docker, MLflow remote setup
- [CONFIGURATION.md](CONFIGURATION.md) — All config options and environment variables
- [DASHBOARD.md](DASHBOARD.md) — Streamlit dashboard guide
- [SECURITY.md](SECURITY.md) — Security notes and hardening applied

## Requirements

- Python 3.8+
- Root/sudo for packet capture (`main.py`, live training)
- Optional: remote MLflow server + MinIO for experiment tracking
