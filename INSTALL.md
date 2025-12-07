# Installation Guide

Complete installation guide for the Cognitive Anomaly Detector with MLflow integration.

## System Requirements

- **OS**: Linux (tested on Ubuntu 20.04+)
- **Python**: 3.8 or higher
- **Privileges**: Root/sudo access for packet capture
- **Memory**: 2GB RAM minimum
- **Disk**: 500MB free space
- **Network**: Active network interface

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### 1. Clone Repository

```bash
git clone https://github.com/RobertoDeLaCamara/DetectorAnomalias.git
cd cognitive-anomaly-detector
```

#### 2. Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Verify activation
which python
# Should show: /path/to/cognitive-anomaly-detector/venv/bin/python
```

#### 3. Install Dependencies

```bash
# Install all packages
pip install -r requirements.txt

# Verify installation
python -c "import sklearn, mlflow, boto3; print('✅ All packages installed')"
```

**Installed Dependencies:**
```
Core ML:
- scikit-learn>=1.3.0
- numpy>=1.24.0
- pandas>=2.0.0
- joblib>=1.3.0

MLflow & Storage:
- mlflow>=2.9.0
- protobuf>=3.20.0
- boto3>=1.28.0

Network & Testing:
- scapy>=2.5.0
- pytest>=7.4.0
- pytest-mock>=3.11.0
- pytest-cov>=4.1.0
```

#### 4. Verify Installation

```bash
# Check Python version
python --version

# Check key packages
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import mlflow; print(f'MLflow: {mlflow.__version__}')"
python -c "import boto3; print(f'boto3: {boto3.__version__}')"

# Run test suite
pytest tests/ -v
```

**Expected Test Results:**
```
tests/test_mlflow_integration.py .......... [ 47%]
tests/test_integration.py ........        [ 68%]
tests/test_isolation_forest.py ...........  [100%]

===================== 36 passed in 18.42s ======================
```

### Method 2: Docker Installation

```bash
# Build image
docker build -t cognitive-anomaly-detector .

# Run detector
docker run --network=host --privileged cognitive-anomaly-detector

# Train model in Docker
docker run --network=host --privileged \
  cognitive-anomaly-detector \
  python train_model.py --duration 60
```

## MLflow Setup

### Local MLflow Setup

```bash
# Initialize MLflow directories and config
python setup_mlflow.py
```

**Output:**
```
============================================================
MLflow Setup for Cognitive Anomaly Detector
============================================================

✓ Created .mlflow
✓ Created .mlflow/mlruns
✓ Created .mlflow/artifacts
✓ Created experiment: cognitive-anomaly-detector

MLflow tracking URI: file:///path/to/.mlflow/mlruns
```

### Remote MLflow Setup

For production deployments with team collaboration:

#### 1. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit with your credentials
nano .env
```

**.env file:**
```bash
# MLflow tracking server
MLFLOW_TRACKING_URI=http://192.168.1.86:5050

# MinIO S3 storage
MLFLOW_S3_ENDPOINT_URL=http://192.168.1.189:9000
MLFLOW_S3_BUCKET=mlflow-artifacts

# MinIO credentials 
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

#### 2. Test Connection

```bash
# Load environment and test
source .env
python test_mlflow_connection.py
```

**Expected Output:**
```
============================================================
MLflow and MinIO Connectivity Test
============================================================

Testing MLflow Tracking Server
✅ Connected successfully!
   Found 2 experiments

Testing MinIO S3 Storage
✅ Connected successfully!
   Bucket 'mlflow-artifacts' exists

Testing End-to-End Workflow
✅ Test run created successfully!

Test Summary
============================================================
MLflow Server:  ✅ PASS
MinIO Storage:  ✅ PASS
End-to-End:     ✅ PASS
```

See [REMOTE_MLFLOW_SETUP.md](REMOTE_MLFLOW_SETUP.md) for detailed remote setup instructions.

## Post-Installation Configuration

### 1. ML Configuration

Edit `src/ml_config.py`:

```python
# ML detection settings
ML_ENABLED = True
MIN_PACKETS_FOR_ML = 10
ML_ANOMALY_THRESHOLD = -0.1

# Model hyperparameters
CONTAMINATION = 0.01  # 1% expected anomaly rate
N_ESTIMATORS = 100    # Number of trees
RANDOM_STATE = 42     # For reproducibility
```

### 2. Rule-Based Configuration

Edit `src/config.py`:

```python
# Detection thresholds
THRESHOLD_MULTIPLIER = 2
ICMP_THRESHOLD = 10
LARGE_PAYLOAD_SIZE = 5000

# Monitoring settings
MONITORING_DURATION = 60  # seconds
```

### 3. MLflow Configuration

Edit `src/mlflow_config.py` (or use environment variables):

```python
# Tracking settings
MLFLOW_ENABLED = True
LOG_MODEL_SIGNATURE = True
LOG_TRAINING_DATA = True

# Experiment settings
DEFAULT_EXPERIMENT_NAME = 'cognitive-anomaly-detector'
```

## Training Your First Model

### Option 1: Synthetic Data (Fastest)

```bash
# Generate synthetic training data
python generate_synthetic_data.py

# Train model
python train_model.py \
  --from-file data/training/synthetic_baseline.csv \
  --version 1 \
  --experiment-name "initial-test"
```

### Option 2: Live Traffic Collection

```bash
# Collect 60 seconds of normal traffic
sudo venv/bin/python train_model.py \
  --duration 60 \
  --contamination 0.01 \
  --version 1
```

### Option 3: From Existing CSV

```bash
# Train from pre-collected data
python train_model.py \ 
  --from-file your_data.csv \
  --version 1
```

## Running the Detector

```bash
# Basic usage
sudo venv/bin/python main.py

# Custom duration
sudo venv/bin/python main.py --duration 120

# Specific interface
sudo venv/bin/python main.py --interface eth0
```

## Verification

### Check Installation

```bash
# Verify models directory
ls -lh models/
# Should show .joblib files

# Verify MLflow directory
ls -lh .mlflow/
# Should show mlruns/ and artifacts/

# Check data directory
ls -lh data/training/
# Should show CSV files if collected
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test suites
pytest tests/test_mlflow_integration.py -v  # MLflow tests
pytest tests/test_integration.py -v         # E2E tests
pytest tests/test_isolation_forest.py -v    # ML model tests

# With coverage
pytest tests/ --cov=src --cov-report=html
firefox htmlcov/index.html
```

## Troubleshooting

### Python Version Issues

```bash
# Check Python version
python3 --version

# Use specific Python version
python3.10 -m venv venv
```

### Permission Errors

```bash
# For packet capture, use sudo with venv python
sudo venv/bin/python main.py
sudo venv/bin/python train_model.py --duration 60

# Don't use just 'sudo python' as it won't use venv
```

### Import Errors

```bash
# Ensure venv is activated
source venv/bin/activate

# Reinstall dependencies if needed
pip install --force-reinstall -r requirements.txt
```

### Network Interface Issues

```bash
# List available interfaces
ip link show

# Use specific interface
sudo venv/bin/python main.py --interface wlan0
```

### MLflow Connection Issues

```bash
# Test local MLflow
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Test remote connection
python test_mlflow_connection.py

# Check environment variables
env | grep MLFLOW
```

### Insufficient Samples Error

```bash
# Increase collection time
sudo venv/bin/python train_model.py --duration 300

# Or use synthetic data
python generate_synthetic_data.py
python train_model.py --from-file data/training/synthetic_baseline.csv
```

## Uninstallation

```bash
# Remove virtual environment
rm -rf venv/

# Remove models and data (optional)
rm -rf models/ data/ .mlflow/

# Remove .env file
rm .env
```

## Next Steps

After successful installation:

1. ✅ Train your first model
2. ✅ Run the detector
3. ✅ View results in MLflow UI
4. ✅ Configure remote MLflow (optional)
5. ✅ Customize detection thresholds
6. ✅ Integrate into your workflow

## Additional Resources

- [QUICKSTART.md](QUICKSTART.md) - 5-minute quick start
- [README.md](README.md) - Complete documentation
- [API.md](API.md) - API reference
- [REMOTE_MLFLOW_SETUP.md](REMOTE_MLFLOW_SETUP.md) - Remote setup guide

## Support

For issues:
- Check [Troubleshooting](#troubleshooting) section
- Run diagnostic tests: `pytest tests/ -v`
- Test MLflow: `python test_mlflow_connection.py`
- Check logs in `logs/` directory

---

**Installation Status**: Ready for production use with MLflow integration! 🎉
