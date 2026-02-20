# Configuration Reference

## Rule-Based Detection (`src/config/config.py`)

| Setting | Default | Description |
|---|---|---|
| `THRESHOLD_MULTIPLIER` | `2.0` | Traffic rate multiplier to trigger spike alert |
| `MONITORING_INTERVAL` | `60` | Analysis window in seconds |
| `ICMP_THRESHOLD` | `50` | ICMP packet count to trigger flood alert |
| `PAYLOAD_THRESHOLD` | `100` | Payload size (bytes) to flag as large |

## ML & Ensemble Settings (`src/config/ml_config.py`)

| Setting | Default | Description |
|---|---|---|
| `ML_ENABLED` | `True` | Enable/disable ML detection |
| `MIN_PACKETS_FOR_ML` | `10` | Min packets per IP before running ML inference |
| `CONTAMINATION` | `0.01` | Expected anomaly fraction for Isolation Forest (1%) |
| `N_ESTIMATORS` | `100` | Number of trees in Isolation Forest |
| `ENSEMBLE_WEIGHT_IF` | `0.4` | Isolation Forest weight in ensemble |
| `ENSEMBLE_WEIGHT_LSTM` | `0.4` | LSTM Autoencoder weight in ensemble |
| `ENSEMBLE_WEIGHT_RULES` | `0.2` | Rule-based weight in ensemble |
| `ENSEMBLE_ANOMALY_THRESHOLD` | `0.6` | Combined score threshold to fire alert |

When an engine has no data (e.g. LSTM buffer not yet full), its weight is redistributed proportionally to the active engines.

## Environment Variables

Ensemble weights can also be set via environment variables, which override the values in `ml_config.py`:

```bash
ENSEMBLE_WEIGHT_IF=0.5
ENSEMBLE_WEIGHT_LSTM=0.3
ENSEMBLE_WEIGHT_RULES=0.2
```

Packet processing tuning:

```bash
PACKET_WORKERS=2          # Worker threads consuming the packet queue
PACKET_QUEUE_SIZE=10000   # Max packets buffered before dropping
```

MLflow / MinIO (see [SETUP.md](SETUP.md) for full setup):

```bash
MLFLOW_TRACKING_URI=http://<server>:5050
MLFLOW_S3_ENDPOINT_URL=http://<server>:9000
MLFLOW_S3_BUCKET=mlflow-artifacts
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

## Dashboard Settings (`src/config/dashboard_config.py`)

```python
DEFAULT_REFRESH_INTERVAL = 2    # seconds between auto-refresh
MAX_REALTIME_POINTS = 100       # data points in live charts
MAX_LOG_LINES = 10000           # lines loaded from log file
MAX_ANOMALIES_DISPLAY = 500     # rows shown in anomaly tables
CHART_HEIGHT = 400

COLORS = {
    "normal":  "#2ecc71",   # green
    "anomaly": "#e74c3c",   # red
    "warning": "#f39c12",   # orange
    "info":    "#3498db",   # blue
}
```

## Training CLI Parameters

```bash
python scripts/train_model.py [options]
```

| Parameter | Description | Example |
|---|---|---|
| `--duration N` | Capture live traffic for N seconds | `--duration 300` |
| `--from-file PATH` | Train from CSV file | `--from-file data/baseline.csv` |
| `--contamination X` | Expected anomaly rate | `--contamination 0.01` |
| `--version V` | Model version number | `--version 2` |
| `--experiment-name NAME` | MLflow experiment name | `--experiment-name prod-v1` |
| `--run-name NAME` | MLflow run name | `--run-name baseline` |
| `--no-mlflow` | Disable MLflow for this run | |

```bash
python scripts/train_lstm_model.py [options]
```

| Parameter | Description | Default |
|---|---|---|
| `--from-file PATH` | Train from CSV | — |
| `--duration N` | Live capture seconds | — |
| `--epochs N` | Training epochs | `50` |
| `--hidden-dim N` | LSTM hidden size | `64` |
| `--sequence-length N` | Sliding window size | `20` |

## Detection CLI Parameters

```bash
sudo venv/bin/python main.py [--duration N] [--interface IFACE]
```

- `--duration N` — run for N seconds then exit (default: continuous until Ctrl+C)
- `--interface IFACE` — network interface to capture on (default: auto-detected)
