# Cognitive Anomaly Detector

A production-ready network anomaly detection system with **triple detection** (rule-based + Isolation Forest + LSTM Autoencoder) and **centralized experiment tracking** via MLflow.

## Features

### ✅ Phase 1: ML-Enhanced Detection (COMPLETED)
- **18-feature extraction pipeline**: Statistical, temporal, protocol, port, and payload features
- **Isolation Forest model**: Unsupervised learning for anomaly detection
- **Dual detection system**: ML and rule-based alerts work together
- **MLflow integration**: Experiment tracking and model registry
- **Remote infrastructure**: MLflow server + MinIO S3 storage support
- **Model versioning**: Track, compare, and manage model versions
- **Comprehensive testing**: 170+ tests with high coverage

### ✅ Phase 2: LSTM Autoencoder (COMPLETED)
- **LSTM Autoencoder model**: Deep learning for sequential anomaly detection
- **Temporal pattern recognition**: Captures traffic patterns over time
- **Reconstruction-based detection**: Anomalies have high reconstruction error
- **Complementary to Isolation Forest**: Better at detecting slow attacks and gradual changes
- **PyTorch implementation**: GPU-accelerated training when available

### ✅ Phase 3: Ensemble Scoring & Async Processing (COMPLETED)
- **Ensemble confidence scoring**: Weighted combination of Isolation Forest, LSTM Autoencoder, and rule-based engines into a single 0-1 confidence score
- **Configurable engine weights**: Tune via `ENSEMBLE_WEIGHT_IF`, `ENSEMBLE_WEIGHT_LSTM`, `ENSEMBLE_WEIGHT_RULES` env vars
- **Dynamic weight redistribution**: When an engine has no data (e.g. LSTM buffer not full), its weight is redistributed to active engines
- **Async packet processing**: Decouples Scapy capture from ML inference via a bounded queue with configurable worker threads (`PACKET_WORKERS`, `PACKET_QUEUE_SIZE`)
- **Thread-safe LSTM buffer**: Buffer operations protected by a lock for safe multi-threaded access
- **Drop counting**: Tracks packets dropped when the processing queue is full

### 🔬 MLflow Experiment Tracking
- **Centralized tracking**: All experiments logged to remote MLflow server
- **Artifact storage**: Models and training data stored in MinIO S3
- **Model registry**: Version management and deployment tracking
- **Team collaboration**: Shared experiment history and results
- **Production ready**: Remote server setup for scalability

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/RobertoDeLaCamara/DetectorAnomalias.git
cd cognitive-anomaly-detector

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Remote MLflow (Optional)

For centralized tracking with remote MLflow server:

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

See [REMOTE_MLFLOW_SETUP.md](REMOTE_MLFLOW_SETUP.md) for detailed setup.

### 3. Train Model

**Option A: With Synthetic Data (fastest)**
```bash
# Generate synthetic training data
python generate_synthetic_data.py

# Train model (completes in seconds)
python train_model.py --from-file data/training/synthetic_baseline.csv --version 1
```

**Option B: With Live Traffic**
```bash
# Collect 60 seconds of real network traffic and train
sudo python train_model.py --duration 60 --version 1
```

### 4. Train LSTM Autoencoder (Optional)

For enhanced temporal pattern detection:

```bash
# Train LSTM Autoencoder with synthetic data
python train_lstm_model.py --from-file data/training/synthetic_baseline.csv

# Or with live traffic (requires more samples)
sudo python train_lstm_model.py --duration 300

# Custom parameters
python train_lstm_model.py --from-file data.csv --epochs 100 --hidden-dim 128 --sequence-length 30
```

**LSTM Autoencoder advantages over Isolation Forest:**
- Captures temporal dependencies in traffic sequences
- Better at detecting slow attacks and gradual anomalies
- Learns complex sequential patterns (connection bursts, timing attacks)

### 5. Run Detector

```bash
# Start anomaly detection with trained model
sudo python main.py
```

### 6. Launch Dashboard (Optional)

Visualize anomalies in real-time with the Streamlit dashboard:

```bash
# Launch dashboard
./run_dashboard.sh

# Open browser to http://localhost:8501
```

**Dashboard Features:**
- 🏠 **Real-time monitoring** with live metrics and charts
- 📊 **Historical analysis** with custom date ranges
- 🔍 **Anomaly inspector** for detailed investigation  
- 🌐 **Traffic Insights** for network pattern analysis
- ⚙️ **System Config** for configuration verification
- 📑 **Reports** for incident reporting
- 🤖 **Model info** showing configuration and features
- 📈 **MLflow integration** (backend) for experiment tracking

See [DASHBOARD.md](DASHBOARD.md) for full documentation.

### 6. Security Updates

Recent security fixes include:
- Hardcoded credentials removal
- ReDoS protection
- Command injection prevention
- Path traversal protection

See [SECURITY_FIXES.md](SECURITY_FIXES.md) for details.

### 7. Run with Docker (Recommended)

You can run all components using Docker Compose:

```bash
# Build containers
docker-compose build

# Run Dashboard (http://localhost:8501)
docker-compose up -d dashboard

# Run Detector (background)
docker-compose up -d detector

# Run Trainer (one-off job)
docker-compose run --rm trainer --duration 60
```

See [DOCKER_SETUP.md](DOCKER_SETUP.md) for detailed instructions.

## MLflow Integration

### Local MLflow Setup

```bash
# Initialize MLflow
python setup_mlflow.py

# Train with local tracking
python train_model.py --duration 60 --version 1

# View MLflow UI
mlflow ui --backend-store-uri file://$(pwd)/.mlflow/mlruns
# Open: http://localhost:5000
```

### Remote MLflow Setup

For team collaboration and production deployments:

1. **Configure environment** (`.env`):
   ```bash
   MLFLOW_TRACKING_URI=http://your-mlflow-server:5050
   MLFLOW_S3_ENDPOINT_URL=http://your-minio-server:9000
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_secret
   ```

2. **Test connection**:
   ```bash
   python test_mlflow_connection.py
   ```

3. **Train** (automatically uses remote):
   ```bash
   python train_model.py --duration 60 --version 1
   ```

4. **View results**:
   - MLflow UI: Your configured tracking URI
   - MinIO: Your configured S3 endpoint

## Training Options

```bash
# Basic training
python train_model.py --duration 60 --version 1

# Custom experiment tracking
python train_model.py --duration 60 \
  --experiment-name "production-v1" \
  --run-name "baseline-model" \
  --version 1

# From pre-collected data
python train_model.py --from-file data/training/baseline.csv --version 2

# Disable MLflow for a run
python train_model.py --duration 60 --no-mlflow
```

### Training Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--duration N` | Collect traffic for N seconds | `--duration 300` |
| `--from-file PATH` | Train from CSV file | `--from-file data/baseline.csv` |
| `--contamination X` | Expected anomaly rate | `--contamination 0.01` |
| `--version V` | Model version number | `--version 2` |
| `--experiment-name` | MLflow experiment name | `--experiment-name prod` |
| `--run-name` | MLflow run name | `--run-name test-1` |
| `--no-mlflow` | Disable MLflow tracking | `--no-mlflow` |

## Detection Usage

### Start Monitoring

```bash
# Monitor with ML model (continuous mode, press Ctrl+C to stop)
sudo python main.py

# Monitor for custom duration (e.g. 120 seconds)
sudo python main.py --duration 120
```

### Output Example

```
Starting network monitoring...
[INFO] ML detector loaded successfully (version 1)
[ALERT] [ML_ENSEMBLE] ML ENSEMBLE ANOMALY: 192.168.1.50 - Ensemble confidence: 0.782. Engines: isolation_forest=0.800, lstm=0.650, rules=1.000
[ALERT] [RULE] Traffic spike from 192.168.1.100 - Rate: 45.2 pkt/s
[ALERT] [RULE] Uncommon port 8888 from 192.168.1.75

Traffic summary:
IP: 192.168.1.45, Packets: 345
IP: 142.250.110.81, Packets: 182

Packet processor: queue_size=0, dropped=0, workers=2
```

## Configuration

### ML Settings (`src/ml_config.py`)
```python
ML_ENABLED = True                    # Enable/disable ML detection
MIN_PACKETS_FOR_ML = 10              # Min packets before ML inference
ML_ANOMALY_THRESHOLD = 0.0          # Anomaly score threshold (adjusted for sensitivity)
CONTAMINATION = 0.01                 # Expected anomaly rate (1%)
N_ESTIMATORS = 100                   # Number of trees in forest

# Ensemble scoring weights (also configurable via env vars)
ENSEMBLE_WEIGHT_IF = 0.4             # Isolation Forest weight
ENSEMBLE_WEIGHT_LSTM = 0.4           # LSTM Autoencoder weight
ENSEMBLE_WEIGHT_RULES = 0.2          # Rule-based weight
ENSEMBLE_ANOMALY_THRESHOLD = 0.6     # Combined score threshold
```

### Rule-Based Settings (`src/config.py`)
```python
THRESHOLD_MULTIPLIER = 2.0           # Traffic spike threshold  
MONITORING_INTERVAL = 60             # Analysis interval in seconds
ICMP_THRESHOLD = 50                  # ICMP flood threshold
PAYLOAD_THRESHOLD = 100              # Large payload threshold
```

## Testing

The project includes a comprehensive test suite covering unit, component, and integration testing.

### Test Categories

| Category | Description | Key Test Files |
|----------|-------------|----------------|
| **Unit Tests** | Validates individual functions and logic | `test_feature_extractor.py`<br>`test_payload_analyzer_fixed.py`<br>`test_logger_setup.py` |
| **Component Tests** | Tests detection engines and ML models | `test_anomaly_detector_new.py`<br>`test_isolation_forest.py` |
| **Integration Tests** | Verifies end-to-end system flow | `test_integration.py`<br>`test_mlflow_integration.py` |

### Running Tests

Run specific test categories or the full suite using `pytest`:

```bash
# 1. Run All Tests
pytest tests/ -v

# 2. Run Core Detection Tests (ML & Rules)
pytest tests/test_anomaly_detector_new.py tests/test_isolation_forest.py -v

# 3. Run Integration Tests (System & MLflow)
pytest tests/test_integration.py tests/test_mlflow_integration.py -v

# 4. Generate Coverage Report
pytest tests/ --cov=src --cov-report=term-missing
```

### Key Test Suites

- **`test_anomaly_detector_new.py`**: Validates the rule-based detection engine (ICMP flood, traffic spikes, port scans).
- **`test_isolation_forest.py`**: Verifies the ML model's training, saving, loading, and prediction logic.
- **`test_integration.py`**: Simulates traffic injection to verify the entire pipeline from packet capture to alert generation.
- **`test_mlflow_integration.py`**: Ensures models are correctly logged, versioned, and retrievable from the MLflow server.

## Project Structure

```
cognitive-anomaly-detector/
├── src/
│   ├── anomaly_detector.py         # Triple detection engine (rules + IF + LSTM ensemble)
│   ├── config.py                   # General configuration
│   ├── dashboard_config.py         # Dashboard settings
│   ├── dashboard_data.py           # Dashboard data management
│   ├── dashboard_extensions.py     # Dashboard additional charts
│   ├── ensemble_scorer.py          # Weighted ensemble scoring across engines
│   ├── feature_extractor.py        # 18-feature extraction
│   ├── isolation_forest_detector.py # Isolation Forest model
│   ├── logger_setup.py             # Logging setup
│   ├── lstm_autoencoder_detector.py # LSTM Autoencoder model (thread-safe)
│   ├── ml_config.py                # ML & ensemble settings
│   ├── mlflow_config.py            # MLflow configuration
│   ├── model_trainer.py            # Training pipeline with MLflow
│   ├── packet_queue.py             # Async packet processing queue
│   ├── payload_analyzer.py         # Pattern matching
│   ├── resource_monitor.py         # Resource usage tracking
│   ├── security_config.py          # Security settings
│   ├── utils.py                    # Shared utilities
│   └── visualization_utils.py      # Plotting helpers
├── tests/
│   ├── test_anomaly_detector_new.py # Detection engine tests
│   ├── test_ensemble_scorer.py     # Ensemble scoring tests
│   ├── test_feature_extractor.py   # Feature tests
│   ├── test_integration.py         # E2E tests
│   ├── test_isolation_forest.py    # ML model tests
│   ├── test_logger_setup.py        # Logger tests
│   ├── test_mlflow_config.py       # MLflow config tests
│   ├── test_mlflow_integration.py  # MLflow tests
│   ├── test_packet_queue.py        # Async packet queue tests
│   ├── test_payload_analyzer.py    # Pattern tests
│   ├── test_payload_analyzer_fixed.py # Fixed pattern tests
│   └── test_security_config.py     # Security config tests
├── data/                           # Data directory
├── models/                         # Local model storage
├── .env.example                    # Environment template
├── API.md                          # API documentation
├── CONTRIBUTING.md                 # Contribution guidelines
├── DASHBOARD.md                    # Dashboard documentation
├── DOCKER_SETUP.md                 # Docker setup guide
├── Dockerfile                      # Docker configuration
├── INSTALL.md                      # Installation guide
├── Jenkinsfile                     # CI/CD pipeline
├── LICENSE                         # License file
├── QUICKSTART.md                   # Quick start guide
├── README.md                       # Project overview
├── REMOTE_MLFLOW_SETUP.md          # Remote setup guide
├── SECURITY.md                     # Security policy
├── SECURITY_FIXES.md               # Security fixes log
├── dashboard.py                    # Dashboard implementation
├── docker-compose.yml              # Docker Compose configuration
├── generate_synthetic_data.py      # Synthetic data generator
├── inject_synthetic_traffic.py     # Traffic injector
├── main.py                         # Detection entry point
├── promote_latest.py               # Model promotion script
├── requirements.txt                # Dependencies
├── run_dashboard.sh                # Dashboard launcher
├── setup_mlflow.py                 # MLflow initialization
├── test_mlflow_connection.py       # Connectivity test
└── train_model.py                  # Training CLI
```

## How It Works

### Triple Detection System

**1. Rule-Based Detection**
- Traffic rate spikes (2x average)
- ICMP flood detection
- Uncommon port monitoring
- Large payload detection (> 100 bytes)
- Malicious pattern matching (SQL injection, XSS, shell commands)

**2. ML-Based Detection (Isolation Forest)**

**Feature Extraction** (18 features per IP):
- **Statistical**: packet/byte rates, sizes, variance (6 features)
- **Temporal**: inter-arrival times, burst rates, duration (4 features)
- **Protocol**: TCP/UDP/ICMP ratios (3 features)
- **Port**: unique ports, uncommon port ratio (2 features)
- **Payload**: entropy, size statistics (3 features)

**3. LSTM Autoencoder Detection**
- Temporal pattern recognition on sliding windows of features
- Reconstruction-error based scoring — higher error = more anomalous
- Thread-safe sequence buffer for real-time per-packet prediction

**Ensemble Scoring**:
All three engines are combined via configurable weighted averaging:
- Isolation Forest (default 40%), LSTM Autoencoder (40%), Rules (20%)
- Combined confidence score 0–1, threshold-configurable (`ENSEMBLE_ANOMALY_THRESHOLD`, default 0.6)
- When an engine is unavailable, its weight is redistributed to active engines
- Alerts of type `ML_ENSEMBLE` include per-engine scores in raw data

**Model Lifecycle**:
1. Train on baseline normal traffic
2. Track experiments in MLflow
3. Version models in registry
4. Load for real-time detection
5. Continuous monitoring and improvement

### Async Packet Processing

The Scapy capture callback enqueues packets into a bounded queue processed by worker threads, preventing ML inference from blocking packet capture:

```
[Scapy sniff] → [PacketProcessor queue] → [Worker threads] → [analyze_packet]
```

- Configurable via `PACKET_WORKERS` (default 2) and `PACKET_QUEUE_SIZE` (default 10000)
- Drops packets when queue is full and tracks drop count
- Stats printed in traffic summary after each capture period

## MLflow Features

### Experiment Tracking
- **Parameters**: contamination, n_estimators, n_features
- **Metrics**: training_time, anomaly_rate, score statistics
- **Artifacts**: model files, training data, feature lists
- **Tags**: project, model_type, framework

### Model Registry
- **Versioning**: Automatic version management
- **Stages**: Development, Staging, Production
- **Lineage**: Track model origins and training data
- **Comparison**: Compare multiple model versions

### Remote Infrastructure
- **MLflow Server**: Centralized tracking and registry
- **MinIO Storage**: S3-compatible object storage for artifacts
- **Team Collaboration**: Shared experiment history
- **Production Ready**: Scalable deployment

## Roadmap

- ✅ **Phase 1: Foundation** (COMPLETED)
  - Isolation Forest implementation
  - Feature extraction pipeline
  - MLflow integration
  - Remote server support
  - Model versioning

- ✅ **Phase 2: Advanced Models** (COMPLETED)
  - LSTM Autoencoder for sequential analysis
  - Ensemble confidence scoring across all engines
  - Configurable weighted combination with dynamic redistribution

- ✅ **Phase 3: Production Features** (COMPLETED)
  - ✅ **Real-time dashboard**
  - ✅ **Async packet processing** with bounded queue and worker threads
  - ✅ **Thread-safe LSTM buffer** for concurrent access
  - Automated retraining
  - A/B testing framework

## Documentation

- [DOCKER_SETUP.md](DOCKER_SETUP.md) - Docker container setup guide
- [DASHBOARD.md](DASHBOARD.md) - Streamlit visualization dashboard guide
- [REMOTE_MLFLOW_SETUP.md](REMOTE_MLFLOW_SETUP.md) - Remote MLflow/MinIO setup
- [INSTALL.md](INSTALL.md) - Detailed installation guide
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [AI_ML_Enhancement_Proposal.md](AI_ML_Enhancement_Proposal.md) - Full roadmap

## Requirements

- Python 3.8+
- Root/sudo access (for packet capture)
- Network interface access
- Optional: Remote MLflow server + MinIO for team usage

### Dependencies
- `scapy>=2.5.0` - Packet capture
- `scikit-learn>=1.3.0` - ML algorithms
- `mlflow>=2.9.0` - Experiment tracking
- `boto3>=1.28.0` - S3/MinIO storage
- `numpy`, `pandas` - Data processing
- `streamlit>=1.28.0` - Real-time dashboard
- `plotly>=5.17.0` - Interactive visualizations
- `altair>=5.1.0` - Declarative charting

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Support

For issues or questions:
- GitHub Issues: [Report bugs or request features]
- Documentation: See docs/ directory
- MLflow UI: View experiment results and model versions
