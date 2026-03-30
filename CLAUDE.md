# cognitive-anomaly-detector

Network anomaly detection: Isolation Forest + rule-based. MLflow experiment tracking, Streamlit dashboard, Scapy packet capture. 18-feature extraction pipeline.

## Key Commands

```bash
# Setup
python generate_synthetic_data.py
python train_model.py --from-file data/training/synthetic_baseline.csv

# Run (needs root for packet capture)
sudo python main.py

# Dashboard
python dashboard.py

# Tests
pytest tests/
```

## CI — Jenkins Multibranch

- GiteaSCMSource
- Stages include SonarQube analysis

## Database

- `anomalies.db` — SQLite, root of project
- `src/anomalies.db` — was merged into root DB (use root DB, not src/)
- Schema: `anomalies(timestamp, ip_address, alert_type, description, anomaly_score, raw_data, is_reviewed)`
- `alert_type` values: `ML`, `ML_ENSEMBLE`, rule-based types — use `str.startswith('ML')` to filter ML-based

## Remotes

- `origin` → Gitea (192.168.1.62:9090)
- `github` → GitHub (RobertoDeLaCamara/CognitiveNetworkAnomalyDetector) — branch: `master`
- License: AGPL v3
