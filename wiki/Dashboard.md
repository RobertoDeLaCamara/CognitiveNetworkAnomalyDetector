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

Real-time overview updated on page refresh (or auto-refresh if enabled):

- **Metric cards**: total anomalies, ML anomalies, rule-based anomalies, unique IPs affected
- **Timeline chart**: anomalies over time (Plotly line chart, configurable granularity)
- **Score distribution**: histogram of confidence_score values across all alerts
- **Top attacking IPs**: bar chart of top 10 IPs by alert count

### 2. Historical Analysis

- Date range selector (start/end datetime)
- Aggregation interval selector (hour, day, week)
- Anomaly rate trend line
- CSV export button (filtered by selected date range)

### 3. Anomaly Inspector

- Search by IP address
- Filter by alert type (RULE, ML, ML_ENSEMBLE, ICMP_FLOOD)
- Filter by anomaly score threshold
- Detailed view: shows all alert fields including `raw_data` JSON (features + patterns matched)
- `is_reviewed` toggle for manual review tracking

### 4. Model Info

- Loaded model type (IsolationForest)
- Model version (from filename: `isolation_forest_v5.joblib` → v5)
- Feature names (from `ml_config.py`)
- Training date (file mtime)
- File size in MB
- ML Tracking experiment link (if `ML Tracking_TRACKING_URI` set)

### 5. Traffic Insights

- **Port distribution**: pie/bar chart of destination port frequencies
- **Hourly heatmap**: alert density by hour of day × day of week
- **IP categorization**: groups IPs into local (RFC1918), public, known services (DNS, NTP, etc.)

### 6. System Config

Read-only view of active configuration values:

- Detection thresholds (THRESHOLD_MULTIPLIER, ICMP_THRESHOLD, etc.)
- ML settings (contamination, n_estimators, sequence length, etc.)
- Ensemble weights and threshold

### 7. Reports

Generate downloadable HTML incident summary reports:

- Select date range and minimum severity
- Report includes: timeline chart, top IPs, alert table, score distribution
- Download as single-file HTML

### 8. Live Logs

- Tail `anomaly_detection.log`
- Configurable line count (50–500)
- Auto-refresh toggle (5s interval)
- Color-coded severity levels

## Data Sources

| Page | Source |
|------|--------|
| All pages | `anomalies.db` — SQLite via `src/core/db_manager.py` |
| Live Logs | `anomaly_detection.log` — direct file read |
| Model Info | `models/isolation_forest_v*.joblib` — file metadata |
| System Config | `src/config/config.py` + `src/config/ml_config.py` — imported at runtime |

## Database Schema

```sql
CREATE TABLE anomalies (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
  ip_address  TEXT NOT NULL,
  alert_type  TEXT NOT NULL,   -- RULE | ML | ML_ENSEMBLE | ICMP_FLOOD
  description TEXT,
  anomaly_score REAL,          -- confidence_score [0,1]
  raw_data    TEXT,            -- JSON: {features, patterns_matched, engine_scores}
  is_reviewed BOOLEAN DEFAULT 0
);
CREATE INDEX idx_timestamp ON anomalies(timestamp);
CREATE INDEX idx_ip       ON anomalies(ip_address);
```

## Visualization Stack

- **Plotly**: all interactive charts (timeline, heatmap, distribution, bar)
- **Streamlit**: layout, widgets, state management
- No external API calls — all data local
