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
| **Reports** | Generate and export HTML incident summaries |
| **Live Logs** | Tail `anomaly_detection.log` with keyword filter |

## Data Sources

- **`anomaly_detection.log`** — parsed for all ML and rule-based alerts
- **`models/isolation_forest_v1.joblib`** — metadata for Model Info page
- **MLflow server** — experiment data (only if `MLFLOW_TRACKING_URI` is set in `.env`)

## Generating Test Data

If the dashboard shows no anomalies:

```bash
# Option 1: run detector for a minute to capture real traffic
sudo venv/bin/python main.py --duration 60

# Option 2: inject synthetic traffic
python scripts/inject_synthetic_traffic.py
```

## Configuration

Edit `src/config/dashboard_config.py` to adjust display limits and colors. See [CONFIGURATION.md](CONFIGURATION.md).

## Troubleshooting

| Problem | Fix |
|---|---|
| Import errors | `pip install -r requirements.txt` |
| No anomalies shown | Run detector first to populate the log |
| MLflow page error | Set `MLFLOW_TRACKING_URI` in `.env` and verify server is up |
| Slow with large logs | Reduce `MAX_LOG_LINES` in `dashboard_config.py` or archive old logs |
