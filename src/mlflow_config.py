"""MLflow configuration for experiment tracking and model management.

This module centralizes all MLflow settings including:
- Tracking URI configuration
- Experiment naming
- Model registry settings
- Artifact storage locations
"""

import os
from pathlib import Path

# ========== MLflow Tracking Configuration ==========

# Remote MLflow tracking server (set via environment variable)
# Default: local file store
# Remote example: 'http://192.168.1.86:5050'
REMOTE_MLFLOW_SERVER = os.getenv('MLFLOW_TRACKING_URI')

# Base directory for local MLflow data (used when no remote server)
MLFLOW_DIR = Path('.mlflow')

# Determine if using remote or local tracking
if REMOTE_MLFLOW_SERVER:
    MLFLOW_TRACKING_URI = REMOTE_MLFLOW_SERVER
    IS_REMOTE_TRACKING = True
else:
    MLFLOW_TRACKING_URI = f'file://{Path.cwd() / MLFLOW_DIR / "mlruns"}'
    IS_REMOTE_TRACKING = False
    # Create local directories only if not using remote
    MLFLOW_DIR.mkdir(exist_ok=True)
    (MLFLOW_DIR / 'mlruns').mkdir(exist_ok=True)
    (MLFLOW_DIR / 'artifacts').mkdir(exist_ok=True)

# Enable/disable MLflow tracking globally
MLFLOW_ENABLED = True

# ========== MinIO / S3 Artifact Storage Configuration ==========

# MinIO S3 endpoint (for artifact storage)
# Set via environment variable: export MLFLOW_S3_ENDPOINT_URL=http://192.168.1.189:9000
MINIO_ENDPOINT = os.getenv('MLFLOW_S3_ENDPOINT_URL')

# S3 bucket for MLflow artifacts
# Default bucket name (can be overridden via environment)
S3_BUCKET_NAME = os.getenv('MLFLOW_S3_BUCKET', 'mlflow-artifacts')

# AWS/MinIO credentials (set via environment variables)
# export AWS_ACCESS_KEY_ID=your_access_key
# export AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# ========== Experiment Configuration ==========

# Default experiment name for anomaly detection training
DEFAULT_EXPERIMENT_NAME = 'cognitive-anomaly-detector'

# Nested runs for hyperparameter tuning (future use)
ENABLE_NESTED_RUNS = False

# ========== Model Registry Configuration ==========

# Model name in MLflow Model Registry
REGISTERED_MODEL_NAME = 'isolation-forest-anomaly-detector'

# Model artifact path within MLflow runs
MODEL_ARTIFACT_PATH = 'model'

# ========== Artifact Storage ==========

# Artifact location depends on tracking mode
if IS_REMOTE_TRACKING and MINIO_ENDPOINT:
    # Use S3/MinIO for remote storage
    ARTIFACT_LOCATION = f's3://{S3_BUCKET_NAME}/artifacts'
else:
    # Use local storage
    ARTIFACT_LOCATION = str(MLFLOW_DIR / 'artifacts')

# ========== Auto-logging Configuration ==========

# Enable automatic logging of sklearn models
AUTOLOG_ENABLED = False  # Manual logging preferred for better control

# Log system metrics (CPU, GPU, memory)
LOG_SYSTEM_METRICS = True

# ========== Logging Settings ==========

# Log model signature (input/output schema)
LOG_MODEL_SIGNATURE = True

# Log input examples for model testing
LOG_INPUT_EXAMPLE = True

# Log training data as artifacts
LOG_TRAINING_DATA = True

# Maximum training data size to log (in MB)
MAX_TRAINING_DATA_SIZE_MB = 50

# ========== Tags Configuration ==========

# Default tags to apply to all runs
DEFAULT_TAGS = {
    'project': 'cognitive-anomaly-detector',
    'model_type': 'isolation_forest',
    'framework': 'scikit-learn'
}

# ========== Helper Functions ==========

def get_tracking_uri() -> str:
    """Get the MLflow tracking URI.
    
    Prioritizes environment variable over configuration.
    
    Returns:
        MLflow tracking URI string
    """
    return os.getenv('MLFLOW_TRACKING_URI', MLFLOW_TRACKING_URI)


def is_remote_tracking() -> bool:
    """Check if using remote MLflow tracking server.
    
    Returns:
        True if using remote tracking server
    """
    uri = get_tracking_uri()
    return uri.startswith('http://') or uri.startswith('https://')


def get_experiment_name(custom_name: str = None) -> str:
    """Get experiment name with optional custom override.
    
    Args:
        custom_name: Optional custom experiment name
    
    Returns:
        Experiment name string
    """
    return custom_name if custom_name else DEFAULT_EXPERIMENT_NAME


def is_mlflow_enabled() -> bool:
    """Check if MLflow tracking is enabled.
    
    Returns:
        True if MLflow is enabled
    """
    # Allow override via environment variable
    env_enabled = os.getenv('MLFLOW_ENABLED', str(MLFLOW_ENABLED)).lower()
    return env_enabled in ('true', '1', 'yes')


def get_run_name(prefix: str = 'run', version: int = None) -> str:
    """Generate a descriptive run name.
    
    Args:
        prefix: Prefix for run name
        version: Optional version number
    
    Returns:
        Formatted run name
    """
    import time
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    if version:
        return f'{prefix}_v{version}_{timestamp}'
    return f'{prefix}_{timestamp}'


def get_s3_config() -> dict:
    """Get MinIO/S3 configuration for MLflow.
    
    Returns:
        Dictionary with S3 configuration
    """
    config = {}
    
    if MINIO_ENDPOINT:
        config['MLFLOW_S3_ENDPOINT_URL'] = MINIO_ENDPOINT
    
    if AWS_ACCESS_KEY_ID:
        config['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    
    if AWS_SECRET_ACCESS_KEY:
        config['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
    
    return config


def apply_s3_config():
    """Apply S3/MinIO configuration to environment.
    
    This sets environment variables needed for MLflow to
    connect to MinIO for artifact storage.
    """
    s3_config = get_s3_config()
    for key, value in s3_config.items():
        os.environ[key] = value


def validate_remote_config() -> tuple:
    """Validate remote MLflow and MinIO configuration.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if is_remote_tracking():
        if not REMOTE_MLFLOW_SERVER:
            issues.append("Remote tracking URI is not set")
        
        # Check if MinIO credentials are set for remote tracking
        if MINIO_ENDPOINT:
            if not AWS_ACCESS_KEY_ID:
                issues.append("AWS_ACCESS_KEY_ID not set for MinIO")
            if not AWS_SECRET_ACCESS_KEY:
                issues.append("AWS_SECRET_ACCESS_KEY not set for MinIO")
    
    return len(issues) == 0, issues
