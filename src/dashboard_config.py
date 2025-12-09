"""Configuration for the Streamlit anomaly visualization dashboard."""

import os
from pathlib import Path
from typing import Optional

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_FILE = PROJECT_ROOT / "anomaly_detection.log"

# Dashboard settings
DASHBOARD_TITLE = "Cognitive Anomaly Detector Dashboard"
DASHBOARD_ICON = "🔍"
DEFAULT_REFRESH_INTERVAL = 2  # seconds
MAX_REALTIME_POINTS = 100  # Max points to show in real-time charts

# Display settings
CHART_HEIGHT = 400
CHART_WIDTH = None  # Auto width
ANOMALY_SCORE_THRESHOLD = -0.1  # From ml_config
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Color scheme
COLORS = {
    "normal": "#2ecc71",      # Green
    "anomaly": "#e74c3c",     # Red
    "warning": "#f39c12",     # Orange
    "info": "#3498db",        # Blue
    "background": "#0e1117", # Dark background
    "text": "#fafafa",        # Light text
    "primary": "#ff4b4b",     # Streamlit red
}

# Feature names (from ml_config)
FEATURE_NAMES = [
    # Statistical (6)
    "packets_per_second",
    "bytes_per_second", 
    "avg_packet_size",
    "packet_size_variance",
    "total_packets",
    "total_bytes",
    # Temporal (4)
    "inter_arrival_mean",
    "inter_arrival_std",
    "burst_rate",
    "session_duration",
    # Protocol (3)
    "tcp_ratio",
    "udp_ratio",
    "icmp_ratio",
    # Port (2)
    "unique_ports",
    "uncommon_port_ratio",
    # Payload (3)
    "avg_entropy",
    "avg_payload_size",
    "payload_size_variance",
]

# MLflow settings
def get_mlflow_uri() -> Optional[str]:
    """Get MLflow tracking URI from environment or default."""
    return os.getenv("MLFLOW_TRACKING_URI", None)

MLFLOW_ENABLED = get_mlflow_uri() is not None
REGISTERED_MODEL_NAME = "cognitive-anomaly-detector"

# Cache settings
CACHE_TTL = 60  # seconds
DATA_CACHE_TTL = 5  # seconds for real-time data

# Anomaly log parsing
LOG_PATTERNS = {
    "ml_alert": r"\[ML\] ANOMALY: ([\d\.]+) - Score: ([-\d\.]+)",
    "rule_alert": r"\[RULE\] ALERT: (.+)",
    "timestamp": r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
    "ip_address": r"from ([\d\.]+)",
}

# Visualization settings
VIZ_CONFIG = {
    "pca_components": 2,
    "tsne_components": 2,
    "umap_neighbors": 15,
    "umap_min_dist": 0.1,
    "scatter_opacity": 0.7,
    "heatmap_colorscale": "RdYlGn_r",
}

# Performance limits
MAX_LOG_LINES = 10000  # Max log lines to read
MAX_ANOMALIES_DISPLAY = 500  # Max anomalies to display in table
