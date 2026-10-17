# Configuration

Settings come from three places: constants in `src/config/` (`config.py`, `ml_config.py`, `dashboard_config.py`), environment variables (read with `os.getenv`, usually from `.env`; see `.env.example`), and CLI flags. The authoritative, complete tables live in [docs/CONFIGURATION.md](https://github.com/RobertoDeLaCamara/CognitiveNetworkAnomalyDetector/blob/master/docs/CONFIGURATION.md); this page summarises what you change most often.

## Detection

| Variable | Default | Meaning |
|----------|---------|---------|
| `MONITORING_INTERVAL` | `60` | Analysis window in seconds (also the default `main.py --duration`) |
| `THRESHOLD_MULTIPLIER` | `2.0` | Traffic-rate multiplier that triggers a spike alert (must be > 1.0) |
| `ICMP_THRESHOLD` | `50` | ICMP packets per window that trigger a flood alert |
| `PAYLOAD_THRESHOLD` | `100` | Payload size in bytes flagged as large |
| `TRUSTED_SUBNETS` | `192.168.65.0/24` | Comma-separated CIDRs the analyzer ignores |
| `PACKET_WORKERS` / `PACKET_QUEUE_SIZE` | `2` / `10000` | Worker threads and queue size; packets beyond the queue are dropped |

## Ensemble

| Variable | Default | Meaning |
|----------|---------|---------|
| `ENSEMBLE_WEIGHT_IF` | `0.4` | Isolation Forest weight |
| `ENSEMBLE_WEIGHT_LSTM` | `0.4` | LSTM autoencoder weight |
| `ENSEMBLE_WEIGHT_RULES` | `0.2` | Rule-based weight |
| `ENSEMBLE_ANOMALY_THRESHOLD` | `0.6` | Combined score that fires an alert |

When an engine has no data yet (for example the LSTM buffer is not full), its weight is redistributed over the active engines.

`ML_ENABLED`, `MIN_PACKETS_FOR_ML` (10), `CONTAMINATION` (0.01), `N_ESTIMATORS` (100) and `MODEL_VERSION` (5) are plain constants in `ml_config.py`, not environment variables.

## LSTM autoencoder

`LSTM_SEQUENCE_LENGTH` (50), `LSTM_HIDDEN_DIM` (64), `LSTM_LATENT_DIM` (32), `LSTM_NUM_LAYERS` (2), `LSTM_DROPOUT` (0.2), `LSTM_BATCH_SIZE` (32), `LSTM_LEARNING_RATE` (0.001), `LSTM_EPOCHS` (50), `LSTM_EARLY_STOPPING` (10) and `LSTM_ANOMALY_PERCENTILE` (95). The sequence length used at inference is the one saved in `lstm_config.json` by the training script.

## Logging and storage

| Variable | Default |
|----------|---------|
| `ANOMALY_LOG_FILE` | `anomaly_detection.log` |
| `LOG_MAX_SIZE` / `LOG_BACKUP_COUNT` | 5 MB / 3 backups |
| `ML_RANDOM_STATE` | random per process (set it for reproducible training) |

Alerts are stored in `anomalies.db` at the project root. The path is not configurable.

## MLflow

See [MLflow and Model Tracking](MLflow-and-Model-Tracking.md): `MLFLOW_TRACKING_URI`, `MLFLOW_ENABLED`, `MLFLOW_S3_ENDPOINT_URL`, `MLFLOW_S3_BUCKET`, `MLFLOW_ENABLE_REMOTE_LOADING`, `MLFLOW_MODEL_STAGE`, `MLFLOW_MODEL_VERSION`.

## Command-line flags

```bash
sudo venv/bin/python main.py [--duration N] [--interface IFACE]
```

`--duration` is the length in seconds of each capture window (1-3600, default `MONITORING_INTERVAL`). After each window a summary is printed and capture restarts; the process runs until Ctrl+C or SIGTERM. `--interface` defaults to Scapy's default interface.

```bash
python scripts/train_model.py (--duration N | --from-file PATH) [--contamination X] [--version V]
                              [--experiment-name NAME] [--run-name NAME] [--interface IFACE]
                              [--no-mlflow] [--no-save-data]
```

`--duration` (1-3600) and `--from-file` are mutually exclusive and one is required. `--contamination` accepts 0.001-0.5 and `--version` 1-999.

`scripts/train_lstm_model.py` takes `--from-file` or `--duration`, plus `--epochs`, `--hidden-dim`, `--latent-dim`, `--num-layers`, `--dropout`, `--sequence-length`, `--batch-size`, `--learning-rate`, `--output-dir` and `--no-mlflow`.

## Dashboard

`STREAMLIT_SERVER_ADDRESS` (`localhost`) and `STREAMLIT_SERVER_PORT` (`8501`) are read by `run_dashboard.sh`. Refresh interval and row limits are constants in `src/config/dashboard_config.py`.
