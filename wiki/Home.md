# Cognitive Anomaly Detector

Triple-engine ensemble network anomaly detector: Isolation Forest (40%) + LSTM Autoencoder (40%) + rule-based (20%) over 18 per-IP features. Scapy capture → async bounded queue → worker threads → ensemble scorer (threshold 0.6) → SQLite alerts.

## Quick Start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Generate training data (baseline normal traffic)
python scripts/generate_synthetic_data.py

# Train Isolation Forest
python scripts/train_model.py --from-file data/training/synthetic_baseline.csv --version 1

# Train LSTM Autoencoder
python scripts/train_lstm_model.py --from-file data/training/synthetic_baseline.csv

# Run detector (requires root for Scapy)
sudo venv/bin/python main.py

# Dashboard
./run_dashboard.sh    # Streamlit on http://localhost:8501

# Docker
docker-compose up -d detector
docker-compose run --rm trainer --duration 60
```

## Stack

| Component | Technology | Port |
|-----------|-----------|------|
| Packet capture | Scapy (root required) | — |
| Dashboard | Streamlit | 8501 |
| Database | SQLite (`anomalies.db`) | — |
| Model registry | ML Tracking + S3-compatible storage (optional) | — |

## Wiki Pages

1. [Architecture and Data Flow](Architecture-and-Data-Flow.md)
2. [ML Pipeline](ML-Pipeline.md)
3. [Dashboard](Dashboard.md)
4. [Security Hardening](Security-Hardening.md)
5. [Development Guide](Development-Guide.md)

## Key Layout

```
main.py                          entry point — packet capture, CLI flags
dashboard.py / run_dashboard.sh  Streamlit app
src/detection/anomaly_detector.py   PacketAnalyzer — main detection class
src/detection/ensemble_scorer.py    weighted fusion, EnsembleResult
src/detection/payload_analyzer.py   regex patterns + ReDoS timeout
src/ml/feature_extractor.py         18-feature per-IP extraction
src/ml/isolation_forest_detector.py IF wrapper + hot reload
src/ml/lstm_autoencoder_detector.py PyTorch AE + hot reload
src/core/db_manager.py              SQLite anomalies table
src/config/config.py                detection thresholds
src/config/ml_config.py             model/ensemble settings
scripts/                            training and data generation
models/                             isolation_forest_v5.joblib, lstm_autoencoder.pt
```

## Non-Obvious Facts

- ML Tracking is optional. Without it, models load from local joblib files. Set `ML Tracking_TRACKING_URI` + S3/S3-compatible storage credentials in `.env` for remote registry.
- LSTM hot reload: model file mtime checked every 60s; changed file triggers in-place reload without restart.
- Alert cooldown: max 3 alerts per (IP, alert_type) per 60s — prevents alert flooding while ML inference continues per-packet.
- Capture interface: set `CAPTURE_INTERFACE` env var (default: `eth0`).
