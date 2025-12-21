"""Configuration for ML-based anomaly detection components."""

import os

# ========== ML Feature Extraction Settings ==========

# Time window for feature aggregation (seconds)
FEATURE_WINDOW_SIZE = 60

# Minimum number of packets required before ML inference
MIN_PACKETS_FOR_ML = 10

# Maximum number of packets to keep in history per IP
MAX_PACKET_HISTORY = 1000

# Feature normalization method: 'standard' or 'minmax'
NORMALIZATION_METHOD = 'standard'

# ========== Isolation Forest Settings ==========

# Expected proportion of anomalies in the dataset (1%)
CONTAMINATION = 0.01

# Number of trees in the forest
N_ESTIMATORS = 100

# Random state for reproducibility (use secure random in production)
import secrets
RANDOM_STATE = int(os.getenv('ML_RANDOM_STATE', secrets.randbits(32) % 2147483647))

# Maximum samples to use for training ('auto' or int)
MAX_SAMPLES = 'auto'

# ========== Model Persistence Settings ==========

# Directory for storing trained models
MODEL_DIR = 'models'

# Current model version
MODEL_VERSION = 1

# Model file paths
ISOLATION_FOREST_MODEL_PATH = os.path.join(
    MODEL_DIR, 
    f'isolation_forest_v{MODEL_VERSION}.joblib'
)
SCALER_MODEL_PATH = os.path.join(
    MODEL_DIR, 
    f'scaler_v{MODEL_VERSION}.joblib'
)

# ========== Detection Settings ==========

# Enable/disable ML-based detection
ML_ENABLED = True

# Anomaly score threshold for alerting (-1 to 1, lower is more anomalous)
# Isolation Forest scores: -1 = anomaly, 1 = normal
# We'll alert when score < threshold
ML_ANOMALY_THRESHOLD = 0.0

# Minimum anomaly score to log (even if not alerting)
ML_LOG_THRESHOLD = 0.0

# ========== Training Data Collection ==========

# Directory for training data
TRAINING_DATA_DIR = 'data/training'

# Minimum number of samples required for training
MIN_TRAINING_SAMPLES = 2

# Maximum training samples to keep in memory
MAX_TRAINING_SAMPLES = 10000

# ========== Feature Definitions ==========

# List of all features extracted (in order)
FEATURE_NAMES = [
    # Statistical features (6)
    'packets_per_second',
    'bytes_per_second',
    'avg_packet_size',
    'packet_size_variance',
    'total_packets',
    'total_bytes',
    
    # Temporal features (4)
    'inter_arrival_mean',
    'inter_arrival_std',
    'burst_rate',
    'session_duration',
    
    # Protocol features (3)
    'tcp_ratio',
    'udp_ratio',
    'icmp_ratio',
    
    # Port features (2)
    'unique_ports',
    'uncommon_port_ratio',
    
    # Payload features (3)
    'payload_entropy',
    'avg_payload_size',
    'payload_size_variance',
]

# Expected number of features
N_FEATURES = len(FEATURE_NAMES)

# ========== Port Classification ==========

# Common/standard ports (for uncommon port ratio calculation)
COMMON_PORTS = [20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 3306, 5432, 8080, 8443]

# ========== Logging Settings ==========

# Log ML predictions to separate file
ML_LOG_FILE = 'ml_anomaly_detection.log'

# Log detailed feature vectors for debugging
LOG_FEATURE_VECTORS = False

# ========== Remote Model Loading ==========

# Enable loading model from MLflow registry at startup
MLFLOW_ENABLE_REMOTE_LOADING = os.getenv('MLFLOW_ENABLE_REMOTE_LOADING', 'false').lower() == 'true'

# Stage to load from (None, "Staging", "Production")
# If None, loads specific version if MLFLOW_MODEL_VERSION is set, else latest
MLFLOW_MODEL_STAGE = os.getenv('MLFLOW_MODEL_STAGE', 'Production')

# Specific version to load (overrides stage if set)
MLFLOW_MODEL_VERSION = os.getenv('MLFLOW_MODEL_VERSION', None)
