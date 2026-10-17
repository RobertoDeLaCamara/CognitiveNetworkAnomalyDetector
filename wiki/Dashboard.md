# Streamlit Dashboard

The dashboard reads directly from `anomalies.db` (SQLite) and `anomaly_detection.log`. No API layer — direct file access.

## Starting the Dashboard

```bash
./run_dashboard.sh           # recommended
# or
streamlit run dashboard.py --server.port 8501
```

Open http://localhost:8501

## Eight Pages

### 1. Home

Real-time overview updated on page refresh (or auto-refresh if enabled; sidebar checkbox, 1-60 s slider). A slider selects the window: last 5-120 minutes (default 30).

- **Metric cards**: four `st.metric` cards (total, ML, rule-based anomalies, unique IPs)
- **Timeline chart**: anomalies over time
- **Score distribution**: histogram of `anomaly_score` values
- **Top IPs**: bar chart of top 10 IPs by alert count
- **Alert type distribution**
- **Recent anomalies** table (at most `MAX_ANOMALIES_DISPLAY` = 500 rows)

### 2. Historical Analysis

- Date range selector (start/end datetime)
- Aggregation interval selector (5min, 15min, 1H (default), 4H, 1D)
- Anomaly rate trend line
- CSV export button (filtered by selected date range)

### 3. Anomaly Inspector

- Search by IP address
- Filter by alert type (RULE, ML, ML_ENSEMBLE, ICMP_FLOOD)
- Filter by anomaly score threshold
- Detailed view of a selected alert
- The `is_reviewed` column exists in the schema but the dashboard has no review toggle

### 4. Model Info

- Model type, number of features, contamination, number of estimators
- Model version (from the model file metadata; the highest `isolation_forest_v*.joblib` is loaded)
- Trained date and file size (KB)
- Feature names (from `dashboard_config.py`; note it names feature 16 `avg_entropy`, whereas `ml_config.py` calls it `payload_entropy`)

### 5. Traffic Insights

- **Port distribution**: pie/bar chart of destination port frequencies
- **Hourly heatmap**: alert density by hour of day × day of week
- **IP categorization**: groups IPs into local (RFC1918), public, known services (DNS, NTP, etc.)

### 6. System Config

Read-only view of active configuration values:

- Detection rules: THRESHOLD_MULTIPLIER, ICMP_THRESHOLD, HIGH_TRAFFIC_PORTS, PAYLOAD_THRESHOLD
- ML settings: ML_ENABLED, MIN_PACKETS_FOR_ML, ML_ANOMALY_THRESHOLD, MODEL_DIR (ensemble weights are not shown)

### 7. Reports

Generate downloadable HTML incident summary reports:

- Select a report period of 1-30 days (no severity filter)
- Preview in the page, then download as a single-file HTML

### 8. Live Logs

- Tail `anomaly_detection.log`
- Configurable line count (50-2000, default 200)
- "Refresh Logs" button; follows the global sidebar auto-refresh interval
- Rendered as a plain code block (no severity colouring)

## Data Sources

| Page | Source |
|------|--------|
| All pages | `anomalies.db` — SQLite via `src/core/db_manager.py` |
| Live Logs | `anomaly_detection.log` — direct file read |
| Model Info | `models/isolation_forest_v*.joblib` — file metadata |
| System Config | `src/config/config.py` + `src/config/ml_config.py` — imported at runtime |
| MLflow loader | Only created when `MLFLOW_TRACKING_URI` is set (even to an empty string, as in `.env.example`) |

## Database Schema

```sql
CREATE TABLE anomalies (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
  ip_address  TEXT NOT NULL,
  alert_type  TEXT NOT NULL,   -- RULE | ML | ML_ENSEMBLE | ICMP_FLOOD (else UNKNOWN)
  description TEXT,
  anomaly_score REAL,          -- ML_ENSEMBLE: confidence [0,1]; ML: raw IF score; RULE/ICMP_FLOOD: 0.0
  raw_data    TEXT,            -- JSON or NULL: ML_ENSEMBLE -> {engine: {raw_score, normalized_score, prediction}}; ML -> {"features": [...]}
  is_reviewed BOOLEAN DEFAULT 0
);
CREATE INDEX idx_timestamp ON anomalies(timestamp);
CREATE INDEX idx_ip       ON anomalies(ip_address);
```

## Visualization Stack

- **Plotly**: all interactive charts (timeline, heatmap, distribution, bar)
- **Streamlit**: layout, widgets, state management
- No external API calls — all data local
