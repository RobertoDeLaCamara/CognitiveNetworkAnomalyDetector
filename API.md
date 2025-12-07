# API Documentation

Complete reference for the Cognitive Anomaly Detector APIs.

## Table of Contents

- [Model Training API](#model-training-api)
- [Isolation Forest Detector API](#isolation-forest-detector-api)
- [Feature Extractor API](#feature-extractor-api)
- [MLflow Configuration API](#mlflow-configuration-api)
- [Anomaly Detector API](#anomaly-detector-api)

---

## Model Training API

### `ModelTrainer`

Main class for training Isolation Forest models with MLflow tracking.

**Location**: `src/model_trainer.py`

#### Constructor

```python
ModelTrainer(enable_mlflow: bool = None)
```

**Parameters:**
- `enable_mlflow` (bool, optional): Enable MLflow tracking. If None, auto-detects from config

**Example:**
```python
from src.model_trainer import ModelTrainer

# With MLflow enabled
trainer = ModelTrainer(enable_mlflow=True)

# Disabled
trainer = ModelTrainer(enable_mlflow=False)

# Auto-detect
trainer = ModelTrainer()
```

#### Methods

##### `collect_training_data(duration, interface=None)`

Collect network traffic for training.

**Parameters:**
- `duration` (int): Collection duration in seconds
- `interface` (str, optional): Network interface to monitor

**Returns:**
- `int`: Number of samples collected

**Example:**
```python
samples = trainer.collect_training_data(duration=60)
print(f"Collected {samples} samples")
```

##### `load_training_data(file_path)`

Load training data from CSV file.

**Parameters:**
- `file_path` (str): Path to CSV file

**Returns:**
- `int`: Number of samples loaded

**Example:**
```python
samples = trainer.load_training_data("data/training/baseline.csv")
```

##### `train_model(contamination=None, experiment_name=None, run_name=None)`

Train Isolation Forest model.

**Parameters:**
- `contamination` (float, optional): Expected anomaly proportion (0.0-0.5)
- `experiment_name` (str, optional): MLflow experiment name
- `run_name` (str, optional): MLflow run name

**Returns:**
- `dict`: Training statistics
  ```python
  {
      'n_samples': int,
      'training_time_seconds': float,
      'anomaly_rate': float,
      'score_mean': float,
      'score_std': float,
      'score_min': float,
      'score_max': float,
      'mlflow_run_id': str,  # If MLflow enabled
      'mlflow_experiment_id': str  # If MLflow enabled
  }
  ```

**Example:**
```python
stats = trainer.train_model(
    contamination=0.01,
    experiment_name="production-v1",
    run_name="baseline"
)
print(f"Training time: {stats['training_time_seconds']}s")
print(f"MLflow run: {stats['mlflow_run_id']}")
```

##### `save_model(version=None, register_model=True)`

Save trained model to disk and optionally register in MLflow.

**Parameters:**
- `version` (int, optional): Model version number
- `register_model` (bool): Register in MLflow Model Registry

**Example:**
```python
trainer.save_model(version=1, register_model=True)
```

---

## Isolation Forest Detector API

### `IsolationForestDetector`

Isolation Forest implementation for anomaly detection.

**Location**: `src/isolation_forest_detector.py`

#### Constructor

```python
IsolationForestDetector(
    contamination=0.01,
    n_estimators=100,
    random_state=42,
    max_samples='auto'
)
```

**Parameters:**
- `contamination` (float): Expected anomaly proportion (default: 0.01)
- `n_estimators` (int): Number of trees (default: 100)
- `random_state` (int): Random seed (default: 42)
- `max_samples` (str|int): Samples per tree (default: 'auto')

**Example:**
```python
from src.isolation_forest_detector import IsolationForestDetector

detector = IsolationForestDetector(
    contamination=0.01,
    n_estimators=100
)
```

#### Methods

##### `train(features)`

Train the model on feature data.

**Parameters:**
- `features` (np.ndarray): Feature matrix (n_samples, n_features)

**Raises:**
- `ValueError`: If feature shape is incorrect or insufficient samples

**Example:**
```python
import numpy as np

features = np.random.randn(150, 18)
detector.train(features)
```

##### `predict(features)`

Predict if sample is anomalous.

**Parameters:**
- `features` (np.ndarray): Feature vector (n_features,) or matrix (n_samples, n_features)

**Returns:**
- `tuple`: (prediction, score)
  - `prediction` (int): 1 for normal, -1 for anomaly
  - `score` (float): Anomaly score (lower = more anomalous)

**Example:**
```python
prediction, score = detector.predict(features)
if prediction == -1:
    print(f"Anomaly detected! Score: {score}")
```

##### `predict_batch(features_list)`

Predict for multiple samples.

**Parameters:**
- `features_list` (np.ndarray): Feature matrix (n_samples, n_features)

**Returns:**
- `tuple`: (predictions, scores)
  - `predictions` (np.ndarray): Array of predictions
  - `scores` (np.ndarray): Array of scores

**Example:**
```python
predictions, scores = detector.predict_batch(features_matrix)
anomalies = predictions == -1
print(f"Found {anomalies.sum()} anomalies")
```

##### `save(model_path, scaler_path)`

Save model and scaler to disk.

**Parameters:**
- `model_path` (str): Path for model file
- `scaler_path` (str): Path for scaler file

**Example:**
```python
detector.save(
    "models/isolation_forest_v1.joblib",
    "models/scaler_v1.joblib"
)
```

##### `load(model_path, scaler_path)`

Load model and scaler from disk.

**Parameters:**
- `model_path` (str): Path to model file
- `scaler_path` (str): Path to scaler file

**Example:**
```python
detector.load(
    "models/isolation_forest_v1.joblib",
    "models/scaler_v1.joblib"
)
```

##### `load_from_mlflow(model_name=None, version=None, stage=None)`

Load model from MLflow Model Registry.

**Parameters:**
- `model_name` (str, optional): Model name (default: from config)
- `version` (int, optional): Model version number
- `stage` (str, optional): Model stage ('Staging', 'Production')

**Example:**
```python
# Load latest version
detector.load_from_mlflow()

# Load specific version
detector.load_from_mlflow(version=2)

# Load production model
detector.load_from_mlflow(stage='Production')
```

##### `save_to_mlflow(experiment_name, run_name=None, register_model=True)`

Save model to MLflow.

**Parameters:**
- `experiment_name` (str): MLflow experiment name
- `run_name` (str, optional): Run name
- `register_model` (bool): Register in Model Registry

**Returns:**
- `str`: MLflow run ID

**Example:**
```python
run_id = detector.save_to_mlflow(
    experiment_name="production",
    run_name="v1.0",
    register_model=True
)
```

---

## Feature Extractor API

### `FeatureExtractor`

Extract 18 features from network traffic.

**Location**: `src/feature_extractor.py`

#### Constructor

```python
FeatureExtractor(window_size=60)
```

**Parameters:**
- `window_size` (int): Time window in seconds (default: 60)

#### Methods

##### `add_packet(packet)`

Add packet to tracking.

**Parameters:**
- `packet` (scapy.Packet): Network packet

**Example:**
```python
from src.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
extractor.add_packet(packet)
```

##### `extract_features(ip_address)`

Extract features for an IP address.

**Parameters:**
- `ip_address` (str): IP address

**Returns:**
- `np.ndarray`: Feature vector (18 features)
  1. `packets_per_second`: Packet rate
  2. `bytes_per_second`: Byte rate
  3. `avg_packet_size`: Average packet size
  4. `packet_size_variance`: Packet size variance
  5. `total_packets`: Total packets
  6. `total_bytes`: Total bytes
  7. `inter_arrival_mean`: Mean inter-arrival time
  8. `inter_arrival_std`: Std dev inter-arrival time
  9. `burst_rate`: Burst detection rate
  10. `session_duration`: Session duration
  11. `tcp_ratio`: TCP packet ratio
  12. `udp_ratio`: UDP packet ratio
  13. `icmp_ratio`: ICMP packet ratio
  14. `unique_ports`: Number of unique ports
  15. `uncommon_port_ratio`: Ratio of uncommon ports
  16. `payload_entropy`: Average payload entropy
  17. `avg_payload_size`: Average payload size
  18. `payload_size_variance`: Payload size variance

**Example:**
```python
features = extractor.extract_features("192.168.1.100")
print(f"Features: {features}")
```

##### `get_all_ips()`

Get all tracked IP addresses.

**Returns:**
- `list`: List of IP addresses

**Example:**
```python
ips = extractor.get_all_ips()
print(f"Tracking {len(ips)} IPs")
```

##### `extract_all_features()`

Extract features for all tracked IPs.

**Returns:**
- `tuple`: (features_array, ip_list, feature_names)

**Example:**
```python
features, ips, names = extractor.extract_all_features()
print(f"Extracted features for {len(ips)} IPs")
```

---

## MLflow Configuration API

### MLflow Config Module

MLflow configuration and utilities.

**Location**: `src/mlflow_config.py`

#### Constants

```python
# Tracking
MLFLOW_TRACKING_URI: str          # MLflow tracking URI
IS_REMOTE_TRACKING: bool          # Using remote server
DEFAULT_EXPERIMENT_NAME: str       # Default experiment name

# Model Registry
REGISTERED_MODEL_NAME: str        # Model name in registry
MODEL_ARTIFACT_PATH: str          # Artifact path

# Storage
MINIO_ENDPOINT: str               # MinIO S3 endpoint
S3_BUCKET_NAME: str               # S3 bucket name
ARTIFACT_LOCATION: str            # Artifact storage location

# Settings
LOG_MODEL_SIGNATURE: bool         # Log model signatures
LOG_INPUT_EXAMPLE: bool           # Log input examples
LOG_TRAINING_DATA: bool           # Log training data
```

#### Functions

##### `get_tracking_uri()`

Get MLflow tracking URI.

**Returns:**
- `str`: Tracking URI

**Example:**
```python
from src.mlflow_config import get_tracking_uri

uri = get_tracking_uri()
print(f"Tracking URI: {uri}")
```

##### `is_remote_tracking()`

Check if using remote tracking.

**Returns:**
- `bool`: True if using remote server

**Example:**
```python
from src.mlflow_config import is_remote_tracking

if is_remote_tracking():
    print("Using remote MLflow server")
```

##### `get_experiment_name(custom_name=None)`

Get experiment name.

**Parameters:**
- `custom_name` (str, optional): Custom experiment name

**Returns:**
- `str`: Experiment name

##### `is_mlflow_enabled()`

Check if MLflow is enabled.

**Returns:**
- `bool`: True if enabled

##### `get_run_name(prefix='run', version=None)`

Generate run name.

**Parameters:**
- `prefix` (str): Name prefix
- `version` (int, optional): Version number

**Returns:**
- `str`: Generated run name

**Example:**
```python
from src.mlflow_config import get_run_name

name = get_run_name('baseline', version=1)
# Returns: 'baseline_v1_20231207_120000'
```

##### `get_s3_config()`

Get S3/MinIO configuration.

**Returns:**
- `dict`: S3 config dictionary

##### `apply_s3_config()`

Apply S3 config to environment.

**Example:**
```python
from src.mlflow_config import apply_s3_config

apply_s3_config()
```

##### `validate_remote_config()`

Validate remote configuration.

**Returns:**
- `tuple`: (is_valid, issues_list)

**Example:**
```python
from src.mlflow_config import validate_remote_config

valid, issues = validate_remote_config()
if not valid:
    print(f"Configuration issues: {issues}")
```

---

## Anomaly Detector API

### `AnomalyDetector`

Dual detection system (rule-based + ML-based).

**Location**: `src/anomaly_detector.py`

#### Constructor

```python
AnomalyDetector(ml_enabled=True, ml_model_path=None)
```

**Parameters:**
- `ml_enabled` (bool): Enable ML detection
- `ml_model_path` (str, optional): Path to model file

#### Methods

##### `add_packet(packet)`

Process a packet through both detection systems.

**Parameters:**
- `packet` (scapy.Packet): Network packet

**Example:**
```python
from src.anomaly_detector import AnomalyDetector

detector = AnomalyDetector()
detector.add_packet(packet)
```

##### `check_anomalies(ip, features=None)`

Check for anomalies using both systems.

**Parameters:**
- `ip` (str): IP address
- `features` (np.ndarray, optional): Pre-extracted features

**Returns:**
- `tuple`: (has_rule_anomaly, has_ml_anomaly)

**Example:**
```python
rule_anomaly, ml_anomaly = detector.check_anomalies("192.168.1.100")
if ml_anomaly:
    print("ML anomaly detected!")
```

---

## Environment Variables

### MLflow Configuration

```bash
# Tracking server
MLFLOW_TRACKING_URI=http://server:5050

# MinIO S3
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
MLFLOW_S3_BUCKET=mlflow-artifacts
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Enable/disable
MLFLOW_ENABLED=true
```

---

## Response Codes and Errors

### Common Exceptions

#### `ValueError`
- Invalid feature shape
- Insufficient training samples
- Invalid configuration

#### `RuntimeError`
- Model not trained
- MLflow not available
- Connection failures

#### `FileNotFoundError`
- Model file not found
- Data file not found

### Example Error Handling

```python
from src.isolation_forest_detector import IsolationForestDetector

detector = IsolationForestDetector()

try:
    detector.load("models/v1.joblib", "models/scaler_v1.joblib")
except FileNotFoundError:
    print("Model not found, training new model...")
    detector.train(features)
    detector.save("models/v1.joblib", "models/scaler_v1.joblib")
except ValueError as e:
    print(f"Invalid data: {e}")
```

---

## Complete Usage Example

```python
import numpy as np
from src.model_trainer import ModelTrainer
from src.isolation_forest_detector import IsolationForestDetector
from src.feature_extractor import FeatureExtractor

# 1. Train a model
trainer = ModelTrainer(enable_mlflow=True)
trainer.collect_training_data(duration=60)
stats = trainer.train_model(
    contamination=0.01,
    experiment_name="production",
    run_name="baseline-v1"
)
trainer.save_model(version=1)

# 2. Load and use for detection
detector = IsolationForestDetector()
detector.load("models/isolation_forest_v1.joblib", "models/scaler_v1.joblib")

# Or load from MLflow
# detector.load_from_mlflow(version=1)

# 3. Extract features and detect
extractor = FeatureExtractor()
# ... add packets ...
features biểu extractor.extract_features("192.168.1.100")

prediction, score = detector.predict(features)
if prediction == -1:
    print(f"Anomaly detected! Score: {score}")

# 4. View in MLflow
print(f"View results: http://your-mlflow-server:5050/#/experiments/{stats['mlflow_experiment_id']}")
```

---

## CLI API

### Training

```bash
# Basic training
python train_model.py --duration 60 --version 1

# With MLflow
python train_model.py \
  --duration 60 \
  --experiment-name "prod" \
  --run-name "baseline" \
  --version 1

# From file
python train_model.py --from-file data/baseline.csv --version 2
```

### Detection

```bash
# Run detector
sudo python main.py

# Custom duration
sudo python main.py --duration 120
```

### MLflow Tools

```bash
# Setup MLflow
python setup_mlflow.py

# Test connection
python test_mlflow_connection.py

# Generate synthetic data
python generate_synthetic_data.py
```

---

For more examples, see the [examples/](examples/) directory.
