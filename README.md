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
python generate_synthetic_data.py
python train_model.py --from-file data/training/synthetic_baseline.csv --version 1

# 3. Detect
sudo venv/bin/python main.py

# 4. Dashboard (optional)
./run_dashboard.sh          # → http://localhost:8501
```

For Docker, full install options, and MLflow remote setup see [SETUP.md](SETUP.md).

## Project Structure

```
├── src/
│   ├── anomaly_detector.py         # Triple detection engine + ensemble
│   ├── ensemble_scorer.py          # Weighted scoring
│   ├── feature_extractor.py        # 18-feature extraction per IP
│   ├── isolation_forest_detector.py
│   ├── lstm_autoencoder_detector.py
│   ├── packet_queue.py             # Async packet processing
│   ├── payload_analyzer.py         # Malicious pattern matching
│   ├── db_manager.py               # SQLite persistence
│   ├── model_trainer.py            # Training pipeline (MLflow integration)
│   ├── config.py                   # Rule thresholds
│   ├── ml_config.py                # ML & ensemble settings
│   └── mlflow_config.py            # MLflow / MinIO config
├── tests/                          # 14 test files (unit + integration)
├── models/                         # Trained model files
├── data/                           # Training data
├── main.py                         # Detection entry point
├── train_model.py                  # Isolation Forest training CLI
├── train_lstm_model.py             # LSTM training CLI
├── dashboard.py                    # Streamlit dashboard
├── generate_synthetic_data.py
└── docker-compose.yml
```

## Training Options

```bash
# From synthetic data (fast, no sudo)
python train_model.py --from-file data/training/synthetic_baseline.csv --version 1

# From live traffic
sudo venv/bin/python train_model.py --duration 60 --version 1

# LSTM Autoencoder
python train_lstm_model.py --from-file data/training/synthetic_baseline.csv

# Disable MLflow for a run
python train_model.py --from-file data.csv --no-mlflow
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
