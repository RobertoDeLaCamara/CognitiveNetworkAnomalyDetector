# Anomaly Detection Dashboard

Comprehensive real-time and historical visualization dashboard for the Cognitive Anomaly Detector.

## Features

### 📊 **Five Interactive Pages**

1. **🏠 Home** - Real-time monitoring
   - Live anomaly metrics (total, ML, rule-based, unique IPs)
   - Interactive timeline chart with anomaly scores
   - Score distribution histogram
   - Top suspicious IPs
   - Alert type breakdown (pie chart)
   - Recent anomalies table
   - Auto-refresh capability

2. **📊 Historical Analysis** - Deep-dive into past detections
   - Customizable date range filters
   - Anomaly rate over time (5min, 15min, 1H, 4H, 1D intervals)
   - Interactive Plotly charts (zoom, pan, hover details)
   - Summary statistics
   - Top IPs analysis
   - CSV export functionality

3. **🔍 Anomaly Inspector** - Detailed investigation tools
   - Search by IP address
   - Filter by alert type (ML/RULE)
   - Minimum anomaly score threshold
   - Detailed anomaly view with full context
   - Side-by-side inspection

4. **🤖 Model Info** - ML model details
   - Model configuration (type, features, contamination)
   - Model metadata (version, trained date, file size)
   - Feature list (18 features)
   - Model status indicator

5. **📈 MLflow** - Experiment tracking integration
   - Recent experiments list
   - Training runs history
   - Parameters and metrics display
   - Connection status

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install dashboard dependencies (if not already installed)
pip install streamlit plotly altair umap-learn
```

### 2. Generate Test Data

```bash
# Generate synthetic data and train model
python generate_synthetic_data.py
python train_model.py --from-file data/training/synthetic_baseline.csv --version 1

# Optional: Run detector to generate some anomaly logs
sudo python main.py --duration 60 &
```

### 3. Launch Dashboard

```bash
# Using launch script (recommended)
./run_dashboard.sh

# Or manually
streamlit run dashboard.py
```

### 4. Open Browser

Navigate to: **http://localhost:8501**

## Dashboard Pages Guide

### Home Page (Real-time Monitoring)

**Key Metrics:**
- Total anomalies detected
- ML-based detections
- Rule-based detections  
- Number of unique suspicious IPs

**Visualizations:**
- **Anomaly Score Timeline**: Line chart showing ML anomaly scores over time with threshold
- **Score Distribution**: Histogram of anomaly score distribution
- **Top IPs**: Bar chart of most frequently flagged IP addresses
- **Alert Types**: Pie chart showing ML vs Rule-based detection ratio

**Controls:**
- Time range slider (5-120 minutes)
- Auto-refresh toggle
- Refresh interval selector

### Historical Analysis

**Features:**
- Date/time range picker for custom analysis periods
- Aggregation interval selector (5min, 15min, 1H, 4H, 1D)
- Anomaly rate trend chart
- Full timeline visualization
- Score distribution analysis
- Top 15 IPs chart
- Summary statistics panel
- CSV export for further analysis

**Use Cases:**
- Identify attack patterns over days/weeks
- Analyze daily/hourly trends
- Compare different time periods
- Export data for reporting

### Anomaly Inspector

**Search Capabilities:**
- Filter by IP address (partial match)
- Filter by alert type (All/ML/RULE)
- Filter by minimum anomaly score

**Detail View:**
- Timestamp
- IP address
- Alert type
- Anomaly score (for ML detections)
- Full description
- All matching anomalies in a table

**Use Cases:**
- Investigate specific IP addresses
- Find all anomalies above certain score
- Deep-dive into suspicious activity

### Model Info

**Displays:**
- Model type (IsolationForest)
- Number of features (18)
- Contamination parameter
- Number of estimators
- Training date
- Model version
- File size
- Complete feature list

**Status Indicators:**
- ✅ Model loaded successfully
- ❌ No model found (with training instructions)

### MLflow Integration

**Requirements:**
- MLflow tracking URI configured in `.env`
- Remote MLflow server running

**Features:**
- List of recent experiments
- Training runs with parameters
- Metrics display (training time, etc.)
- Connection status indicator

**If Not Configured:**
- Shows setup instructions
- Links to REMOTE_MLFLOW_SETUP.md

## Configuration

### Dashboard Settings

Edit `src/dashboard_config.py` to customize:

```python
# Display settings
DEFAULT_REFRESH_INTERVAL = 2  # seconds
MAX_REALTIME_POINTS = 100
CHART_HEIGHT = 400

# Color scheme
COLORS = {
    "normal": "#2ecc71",      # Green
    "anomaly": "#e74c3c",     # Red
    "warning": "#f39c12",     # Orange
    "info": "#3498db",        # Blue
}

# Performance limits
MAX_LOG_LINES = 10000
MAX_ANOMALIES_DISPLAY = 500
```

### MLflow Integration

To enable MLflow integration:

1. Set up remote MLflow server (see `REMOTE_MLFLOW_SETUP.md`)
2. Configure `.env`:
   ```bash
   MLFLOW_TRACKING_URI=http://your-mlflow-server:5050
   MLFLOW_S3_ENDPOINT_URL=http://your-minio-server:9000
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_secret
   ```
3. Dashboard will automatically detect and enable MLflow page

## Architecture

### Data Flow

```
Anomaly Logs ─────┐
                  ├──→ AnomalyDataLoader ──→ Pandas DataFrame ──→ Visualizations
Model Files ──────┤
                  └──→ ModelMetricsLoader ──→ Model Info Display
MLflow Server ────→ MLflowDataLoader ──→ Experiment Data
```

### Modules

- **`dashboard.py`**: Main Streamlit application
- **`src/dashboard_config.py`**: Configuration and settings
- **`src/dashboard_data.py`**: Data loading utilities
- **`src/visualization_utils.py`**: Plotly chart functions

### Data Sources

1. **Anomaly Logs**: `anomaly_detection.log`
   - Parses ML and rule-based alerts
   - Extracts timestamp, IP, score, description

2. **Model Files**: `models/isolation_forest_model.pkl`
   - Loads model metadata
   - Checks model availability

3. **MLflow Server**: Remote tracking URI
   - Experiments and runs
   - Parameters and metrics

## Performance Optimization

### Caching

Dashboard uses Streamlit caching for performance:

```python
@st.cache_resource  # For data loaders (one-time init)
@st.cache_data      # For data (TTL-based refresh)
```

### Limits

- **Max log lines**: 10,000 (most recent)
- **Max anomalies display**: 500 in table
- **Real-time points**: 100 in charts

### Auto-Refresh

- Configurable interval (1-60 seconds)
- Disabled by default to conserve resources
- Enable in sidebar when needed

## Troubleshooting

### Dashboard Won't Start

**Issue**: Import errors or missing dependencies

**Solution**:
```bash
pip install -r requirements.txt
```

### No Anomalies Showing

**Issue**: No data in `anomaly_detection.log`

**Solution**:
```bash
# Run detector to generate logs
sudo python main.py --duration 60
```

### MLflow Page Shows Error

**Issue**: MLflow not configured

**Solution**:
- Check `.env` file has `MLFLOW_TRACKING_URI`
- Verify MLflow server is running
- Test connection: `python test_mlflow_connection.py`

### Charts Not Rendering

**Issue**: Browser compatibility or Plotly issues

**Solution**:
- Use modern browser (Chrome, Firefox, Edge)
- Clear browser cache
- Check browser console for errors

### Slow Performance

**Issue**: Large log files or many anomalies

**Solution**:
- Reduce `MAX_LOG_LINES` in config
- Use date filters in Historical Analysis
- Archive old log files

## Advanced Features

### Custom Visualizations

Add new charts in `src/visualization_utils.py`:

```python
def plot_custom_chart(df: pd.DataFrame) -> go.Figure:
    """Create custom visualization."""
    fig = go.Figure(...)
    return fig
```

Then use in `dashboard.py`:

```python
from src.visualization_utils import plot_custom_chart

st.plotly_chart(plot_custom_chart(df), use_container_width=True)
```

### Export Data

Historical Analysis page includes CSV export:

1. Select date range
2. Click "📥 Download Data as CSV"
3. Save file for external analysis (Excel, R, Python)

### Embedding

Dashboard can be embedded in iframe:

```html
<iframe src="http://localhost:8501" width="100%" height="800px"></iframe>
```

## Screenshots

### Home Page
- Real-time metrics cards
- Interactive timeline with anomaly scores
- Distribution charts and top IPs

### Historical Analysis
- Custom date range selection
- Trend charts with time aggregation
- Exportable data tables

### Anomaly Inspector
- Advanced search filters
- Detailed anomaly view
- Full context inspection

## Security Considerations

- Dashboard runs on `localhost` by default
- For production deployment:
  - Use authentication (Streamlit auth or reverse proxy)
  - Enable HTTPS
  - Restrict network access
  - Sanitize displayed data

## Future Enhancements

- [ ] Feature space visualization (PCA/t-SNE/UMAP)
- [ ] Real-time WebSocket updates
- [ ] Alert configuration interface
- [ ] Model retraining interface
- [ ] Multi-language support
- [ ] Custom dashboard themes
- [ ] Email/Slack alert integration
- [ ] Anomaly clustering visualization
- [ ] Predictive analytics

## Support

For issues or feature requests:
- Check this documentation
- Review `implementation_plan.md`
- Open GitHub issue

## License

Same as main project (MIT License)
