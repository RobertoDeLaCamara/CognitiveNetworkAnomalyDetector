# MLflow and Model Tracking

MLflow tracks training experiments and, optionally, serves the Isolation Forest model from its registry. It is optional: without it the detector loads models from local `.joblib` files.

## Where runs are stored

| `MLFLOW_TRACKING_URI` | Backend |
|-----------------------|---------|
| unset or empty | Local SQLite: `sqlite:///<cwd>/.mlflow/mlflow.db` (artifacts in `.mlflow/artifacts`) |
| `http://host:5050` | Remote tracking server |

A non-empty `MLFLOW_TRACKING_URI` always takes precedence over the local default. MLflow 3.x rejects the old file-store backend (`./mlruns`), which is why the default is SQLite.

```bash
# Force local tracking for one run even if .env points to a server
MLFLOW_TRACKING_URI="" python scripts/train_model.py --duration 60 --version 5

# Browse local experiments (http://localhost:5000)
mlflow ui --backend-store-uri sqlite:///$(pwd)/.mlflow/mlflow.db
```

`MLFLOW_ENABLED` (`true`/`1`/`yes`, case-insensitive; anything else disables tracking) switches tracking on or off globally. `--no-mlflow` disables it for a single training run.

The default experiment is `cognitive-anomaly-detector-v2`; override it with `train_model.py --experiment-name`. Models are registered as `isolation-forest-anomaly-detector`.

## Migrating an existing file store

Older versions wrote to `mlruns/`. It is not deleted, and it is not migrated automatically. To keep the history:

```bash
mlflow migrate-filestore --source .mlflow --target sqlite:///.mlflow/mlflow.db
```

## Remote server with S3-compatible artifacts (MinIO)

```bash
MLFLOW_TRACKING_URI=http://<mlflow-server>:5050
MLFLOW_S3_ENDPOINT_URL=http://<minio-server>:9000   # S3 API is 9000, the web UI is 9001
MLFLOW_S3_BUCKET=mlflow-artifacts
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

`python scripts/test_mlflow_connection.py` checks the MLflow server, the MinIO storage and an end-to-end run, and prints a PASS/FAIL line for each (`MLflow Server`, `MinIO Storage`, `End-to-End`; MinIO shows `NOT CONFIGURED` when no S3 endpoint is set). `python scripts/setup_mlflow.py` creates the local directories and the default experiment.

## Loading the model from the registry

By default `main.py` loads the local `models/isolation_forest_v5.joblib`. To load from the registry at startup:

| Variable | Default | Meaning |
|----------|---------|---------|
| `MLFLOW_ENABLE_REMOTE_LOADING` | `false` | `true` loads the Isolation Forest from the registry |
| `MLFLOW_MODEL_STAGE` | `Production` | Registry stage to load when no version is given |
| `MLFLOW_MODEL_VERSION` | unset | A specific registry version; overrides the stage |

`python scripts/promote_latest.py` moves the newest registered version to the `Production` stage and archives the previous ones.

Note: `docker-compose.yml` sets `MLFLOW_MODEL_VERSION=4` for the detector; it only has an effect when `MLFLOW_ENABLE_REMOTE_LOADING=true`.

## Related

- [Installation](Installation.md) and [Configuration](Configuration.md)
- [ML Pipeline](ML-Pipeline.md) for the models themselves
