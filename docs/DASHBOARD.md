# Dashboard

Streamlit-based visualization for real-time and historical anomaly data.

## Launch

```bash
./run_dashboard.sh          # → http://localhost:8501

# Or directly
source venv/bin/activate
streamlit run dashboard.py
```

Via Docker:
```bash
docker-compose up -d dashboard
```

## Pages

| Page | Description |
|---|---|
| **Home** | Live metrics, anomaly timeline, score distribution, top IPs |
| **Historical Analysis** | Custom date range, trend charts, CSV export |
| **Anomaly Inspector** | Search by IP, filter by type/score, detailed view |
| **Model Info** | Model type, version, features, configuration |
| **Traffic Insights** | Port distribution, day/hour heatmap |
| **System Config** | Read-only view of current detection thresholds |
| **Reports** | Generate and download an HTML incident summary for the last 1-30 days |
| **Live Logs** | Tail `anomaly_detection.log` (50-2000 lines); no keyword filter |

## Data Sources

- **`anomalies.db`** (SQLite, project root) — all alert pages read from it via `DBManager`
- **`anomaly_detection.log`** — shown on the Live Logs page only
- **`models/isolation_forest_v*.joblib`** — highest version is used for the Model Info page
- **MLflow server** — experiment/registry data (only if `MLFLOW_TRACKING_URI` is set in the environment or `.env`)

## Generating Test Data

If the dashboard shows no anomalies:

```bash
# Option 1: run the detector to capture real traffic (stop with Ctrl+C;
# --duration only sets the length of each capture window)
sudo venv/bin/python main.py

# Option 2: inject synthetic traffic
python scripts/inject_synthetic_traffic.py
```

## Configuration

Edit `src/config/dashboard_config.py` to adjust display limits and colors. See [CONFIGURATION.md](CONFIGURATION.md).

## Troubleshooting

| Problem | Fix |
|---|---|
| Import errors | `pip install -r requirements.txt` |
| No anomalies shown | Run detector first to populate `anomalies.db` |
| MLflow data missing | Set `MLFLOW_TRACKING_URI` in `.env` and verify server is up |
| Slow with large tables | Reduce `MAX_LOG_LINES` (max records loaded from the DB) in `dashboard_config.py` |
