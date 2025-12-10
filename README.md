# Cognitive Anomaly Detector

A production-ready network anomaly detection system with **dual detection** (rule-based + ML-based) and **centralized experiment tracking** via MLflow.

## Features

### ✅ Phase 1: ML-Enhanced Detection (COMPLETED)
- **18-feature extraction pipeline**: Statistical, temporal, protocol, port, and payload features
- **Isolation Forest model**: Unsupervised learning for anomaly detection
- **Dual detection system**: ML and rule-based alerts work together
- **MLflow integration**: Experiment tracking and model registry
- **Remote infrastructure**: MLflow server + MinIO S3 storage support
- **Model versioning**: Track, compare, and manage model versions
- **Comprehensive testing**: 36+ tests with high coverage

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

### 4. Run Detector

```bash
# Start anomaly detection with trained model
sudo python main.py
```

### 5. Launch Dashboard (Optional)

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
- 🤖 **Model info** showing configuration and features
- 📈 **MLflow integration** for experiment tracking

See [DASHBOARD.md](DASHBOARD.md) for full documentation.

### 6. Run with Docker (Recommended)

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
[ALERT] [ML] ML ANOMALY: 192.168.1.50 - Score: -0.234
[ALERT] [RULE] Traffic spike from 192.168.1.100 - Rate: 45.2 pkt/s
[ALERT] [RULE] Uncommon port 8888 from 192.168.1.75

Traffic summary:
IP: 192.168.1.45, Packets: 345
IP: 142.250.110.81, Packets: 182
```

## Configuration

### ML Settings (`src/ml_config.py`)
```python
ML_ENABLED = True                    # Enable/disable ML detection
MIN_PACKETS_FOR_ML = 10              # Min packets before ML inference
ML_ANOMALY_THRESHOLD = 0.0          # Anomaly score threshold (adjusted for sensitivity)
CONTAMINATION = 0.01                 # Expected anomaly rate (1%)
N_ESTIMATORS = 100                   # Number of trees in forest
```

### Rule-Based Settings (`src/config.py`)
```python
THRESHOLD_MULTIPLIER = 2             # Traffic spike threshold  
ICMP_THRESHOLD = 10                  # ICMP flood threshold
LARGE_PAYLOAD_SIZE = 5000            # Large payload threshold
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# MLflow integration tests
pytest tests/test_mlflow_integration.py -v

# Integration tests
pytest tests/test_integration.py -v
```

## Project Structure

```
cognitive-anomaly-detector/
├── src/
│   ├── anomaly_detector.py         # Dual detection engine
│   ├── config.py                   # General configuration
│   ├── dashboard_config.py         # Dashboard settings
│   ├── dashboard_data.py           # Dashboard data management
│   ├── feature_extractor.py        # 18-feature extraction
│   ├── isolation_forest_detector.py # ML model implementation
│   ├── logger_setup.py             # Logging setup
│   ├── ml_config.py                # ML settings
│   ├── mlflow_config.py            # MLflow configuration
│   ├── model_trainer.py            # Training pipeline with MLflow
│   ├── payload_analyzer.py         # Pattern matching
│   ├── security_config.py          # Security settings
│   ├── utils.py                    # Shared utilities
│   └── visualization_utils.py      # Plotting helpers
├── tests/
│   ├── test_feature_extractor.py   # Feature tests
│   ├── test_integration.py         # E2E tests
│   ├── test_isolation_forest.py    # ML model tests
│   ├── test_mlflow_integration.py  # MLflow tests
│   └── test_payload_analyzer.py    # Pattern tests
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
├── dashboard.py                    # Dashboard implementation
├── docker-compose.yml              # Docker Compose configuration
├── generate_synthetic_data.py      # Synthetic data generator
├── inject_synthetic_traffic.py     # Traffic injector
├── install-python-jenkins.sh       # Jenkins setup script
├── main.py                         # Detection entry point
├── promote_latest.py               # Model promotion script
├── requirements.txt                # Dependencies
├── run_dashboard.sh                # Dashboard launcher
├── setup_mlflow.py                 # MLflow initialization
├── test_mlflow_connection.py       # Connectivity test
└── train_model.py                  # Training CLI
```

## How It Works

### Dual Detection System

**1. Rule-Based Detection**
- Traffic rate spikes (2x average)
- ICMP flood detection
- Uncommon port monitoring
- Large payload detection
- Malicious pattern matching (SQL injection, XSS, shell commands)

**2. ML-Based Detection (Isolation Forest)**

**Feature Extraction** (18 features per IP):
- **Statistical**: packet/byte rates, sizes, variance (6 features)
- **Temporal**: inter-arrival times, burst rates, duration (4 features)
- **Protocol**: TCP/UDP/ICMP ratios (3 features)
- **Port**: unique ports, uncommon port ratio (2 features)
- **Payload**: entropy, size statistics (3 features)

**Anomaly Scoring**:
- Isolation Forest assigns scores (-1 to 1)
- Lower scores = more anomalous behavior
- Threshold-based alerting (configurable)

**Model Lifecycle**:
1. Train on baseline normal traffic
2. Track experiments in MLflow
3. Version models in registry
4. Load for real-time detection
5. Continuous monitoring and improvement

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

- ⏳ **Phase 2: Advanced Models**
  - LSTM networks for sequential analysis
  - Autoencoder for unsupervised detection
  - Ensemble meta-learning
  
- 📋 **Phase 3: Production Features**
  - ✅ **Real-time dashboard** (COMPLETED)
  - Automated retraining
  - A/B testing framework
  - Alert management system

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
