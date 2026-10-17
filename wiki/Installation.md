# Installation

## Prerequisites

- Python 3.10+ (the pinned `mlflow` 3.x, `torch` and `streamlit` need it; the Docker image and CI use 3.11)
- `libpcap` for packet capture: `sudo apt install libpcap-dev` (Linux) or `brew install libpcap` (macOS)
- Root / sudo for live capture

## Local install

```bash
git clone https://github.com/RobertoDeLaCamara/CognitiveNetworkAnomalyDetector.git
cd CognitiveNetworkAnomalyDetector

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train a model and run the detector

```bash
# 1. Synthetic baseline data, then train the Isolation Forest
python scripts/generate_synthetic_data.py
python scripts/train_model.py --from-file data/training/synthetic_baseline.csv --version 5

# 2. Detector (root is needed for packet capture)
sudo venv/bin/python main.py

# 3. Dashboard at http://localhost:8501
./run_dashboard.sh
```

**Use `--version 5`.** The detector loads `models/isolation_forest_v{MODEL_VERSION}.joblib` and the matching scaler, and `MODEL_VERSION` is `5` in `src/config/ml_config.py`. A model trained with another version is never picked up and ML detection stays disabled (rule-based detection keeps working).

The LSTM autoencoder is trained separately with `python scripts/train_lstm_model.py --from-file ...`. See [Configuration](Configuration.md) for every option.

## Docker

```bash
cp .env.example .env            # required: docker-compose.yml loads .env
docker-compose build

docker-compose up -d dashboard  # http://localhost:8501
docker-compose up -d detector   # host networking, privileged, runs as root
docker-compose logs -f detector

# One-off training job
docker-compose run --rm trainer --from-file data/training/synthetic_baseline.csv --version 5
```

The whole project directory is bind-mounted (`.:/app`), so `models/`, `data/`, `.mlflow/` and `anomalies.db` persist on the host. The detector service uses `network_mode: host` and `privileged: true` to capture host traffic.

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `sudo python main.py` cannot import packages | `sudo python` uses the system interpreter. Use `sudo venv/bin/python main.py`. |
| ML detection is off / "model not found" | `ls models/` must contain `isolation_forest_v5.joblib` and `scaler_v5.joblib`. Train with `--version 5`. |
| Too few samples when training from live capture | Use the synthetic baseline: `scripts/generate_synthetic_data.py`. |
| Dashboard shows no alerts | Alerts are read from `anomalies.db` in the project root; the detector must have run from that directory. |

Next: [Configuration](Configuration.md), [MLflow and Model Tracking](MLflow-and-Model-Tracking.md).
