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

MLflow / S3-compatible storage (see [SETUP.md](SETUP.md) for full setup):

```bash
MLFLOW_TRACKING_URI=http://<server>:5050   # empty/unset -> local SQLite sqlite:///<cwd>/.mlflow/mlflow.db
MLFLOW_S3_ENDPOINT_URL=http://<server>:9000
MLFLOW_S3_BUCKET=mlflow-artifacts
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
MLFLOW_ENABLED=true                        # true/1/yes (case-insensitive); anything else disables tracking
```

Remote model loading (read in `src/config/ml_config.py` / `anomaly_detector.py`):

```bash
MLFLOW_ENABLE_REMOTE_LOADING=false   # true -> load the IF model from the MLflow registry at startup
MLFLOW_MODEL_STAGE=Production        # registry stage used when no version is set
MLFLOW_MODEL_VERSION=                # specific registry version; overrides the stage
```

### All environment variables read by the code

| Variable | Default | Where |
|---|---|---|
| `ANOMALY_LOG_FILE` | `anomaly_detection.log` | `config.py` |
| `LOG_MAX_SIZE` | `5242880` (5 MB) | `config.py` |
| `LOG_BACKUP_COUNT` | `3` | `config.py` |
| `THRESHOLD_MULTIPLIER` | `2.0` (must be > 1.0) | `config.py` |
| `MONITORING_INTERVAL` | `60` | `config.py` |
| `ICMP_THRESHOLD` | `50` | `config.py` |
| `PAYLOAD_THRESHOLD` | `100` | `config.py` |
| `TRUSTED_SUBNETS` | `192.168.65.0/24` (comma-separated CIDRs, ignored by the analyzer) | `config.py` |
| `ML_RANDOM_STATE` | random per process | `ml_config.py` |
| `ENSEMBLE_WEIGHT_IF` / `_LSTM` / `_RULES` | `0.4` / `0.4` / `0.2` | `ml_config.py`, `ensemble_scorer.py` |
| `ENSEMBLE_ANOMALY_THRESHOLD` | `0.6` | `ml_config.py`, `ensemble_scorer.py` |
| `PACKET_WORKERS` / `PACKET_QUEUE_SIZE` | `2` / `10000` | `packet_queue.py` |
| `LSTM_SEQUENCE_LENGTH` | `50` | `lstm_autoencoder_detector.py` |
| `LSTM_HIDDEN_DIM` / `LSTM_LATENT_DIM` / `LSTM_NUM_LAYERS` | `64` / `32` / `2` | `lstm_autoencoder_detector.py` |
| `LSTM_DROPOUT` | `0.2` | `lstm_autoencoder_detector.py` |
| `LSTM_BATCH_SIZE` / `LSTM_LEARNING_RATE` / `LSTM_EPOCHS` | `32` / `0.001` / `50` | `lstm_autoencoder_detector.py` |
| `LSTM_EARLY_STOPPING` | `10` (patience) | `lstm_autoencoder_detector.py` |
| `LSTM_ANOMALY_PERCENTILE` | `95` | `lstm_autoencoder_detector.py` |
| `MAX_IP_TRACKING`, `MAX_PAYLOAD_SIZE`, `MAX_PATTERN_MATCHES`, `ALERT_COOLDOWN_SECONDS`, `MAX_ALERTS_PER_IP` | `10000`, `1024`, `10`, `60`, `3` | `security_config.py` (not imported by the runtime; see [SECURITY.md](SECURITY.md)) |
| `STREAMLIT_SERVER_ADDRESS` / `STREAMLIT_SERVER_PORT` | `localhost` / `8501` | `run_dashboard.sh` |

`MIN_PACKETS_FOR_ML`, `CONTAMINATION`, `N_ESTIMATORS`, `MODEL_VERSION` and `ML_ENABLED` are plain constants in `ml_config.py`, not environment variables.

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
| `--experiment-name NAME` | MLflow experiment name (default `cognitive-anomaly-detector-v2`) | `--experiment-name prod-v1` |
| `--run-name NAME` | MLflow run name | `--run-name baseline` |
| `--no-mlflow` | Disable MLflow for this run | |
| `--no-save-data` | Do not save collected live-capture data to CSV | |
| `--interface IFACE` | Interface for `--duration` capture (default: all) | `--interface eth0` |

`--duration` (1-3600) and `--from-file` are mutually exclusive and one is required; `--contamination` accepts 0.001-0.5 (default 0.01); `--version` accepts 1-999 (default 1).

```bash
python scripts/train_lstm_model.py [options]
```

| Parameter | Description | Default |
|---|---|---|
| `--from-file PATH` | Train from CSV | — |
| `--duration N` | Live capture seconds | — |
| `--epochs N` | Training epochs | `50` |
| `--hidden-dim N` | LSTM hidden size | `64` |
| `--sequence-length N` | Sliding window size (saved to `lstm_config.json` and restored at load time; the env default `LSTM_SEQUENCE_LENGTH` is 50) | `20` |
| `--latent-dim N` / `--num-layers N` / `--dropout X` | Model shape | `32` / `2` / `0.2` |
| `--batch-size N` / `--learning-rate X` | Optimisation | `32` / `0.001` |
| `--output-dir DIR` | Where model files are written | `models` |
| `--no-mlflow` | Disable MLflow for this run | |

## Detection CLI Parameters

```bash
sudo venv/bin/python main.py [--duration N] [--interface IFACE]
```

- `--duration N` — length in seconds of each capture window, 1-3600 (default: `MONITORING_INTERVAL`, 60). After each window a traffic summary is printed and capture restarts; the process runs until Ctrl+C / SIGTERM.
- `--interface IFACE` — network interface to capture on (default: Scapy's default interface). Must be alphanumeric/`_`/`-`, at most 15 chars, and exist on the host.
