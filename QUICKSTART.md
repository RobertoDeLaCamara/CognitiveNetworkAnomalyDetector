# Quick Start Guide

Get up and running with the Cognitive Anomaly Detector in 5 minutes!

## Prerequisites

- Python 3.8+
- Root/sudo access (for packet capture)
- Virtual environment activated

## 5-Minute Setup

### 1. Install Dependencies

```bash
# Clone and setup
git clone https://github.com/RobertoDeLaCamara/DetectorAnomalias.git
cd cognitive-anomaly-detector

# Create venv and install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Installed packages:**
- `scikit-learn>=1.3.0` - ML algorithms
- `mlflow>=2.9.0` - Experiment tracking  
- `boto3>=1.28.0` - S3/MinIO storage
- `numpy`, `pandas`, `scapy` - Core dependencies

### 2. Initialize MLflow (Optional)

For experiment tracking:

```bash
# Setup local MLflow
python setup_mlflow.py

# Or configure remote (see REMOTE_MLFLOW_SETUP.md)
cp .env.example .env
# Edit .env with your credentials
```

### 3. Train Model

**Option A: Quick Test with Synthetic Data (30 seconds)**

```bash
# Generate synthetic data
python generate_synthetic_data.py

# Train model
python train_model.py --from-file data/training/synthetic_baseline.csv --version 1
```

**Option B: Real Network Traffic**

```bash
# Collect and train (requires sudo)
sudo venv/bin/python train_model.py --duration 60 --version 1
```

### 4. Run Detector

```bash
# Start monitoring (requires sudo)
sudo venv/bin/python main.py
```

## Expected Output

### Training

```
MLflow tracking enabled
Loading training data from data/training/synthetic_baseline.csv
Training Isolation Forest model...

============================================================
TRAINING COMPLETED SUCCESSFULLY  
============================================================
Samples:          150
Features:         18
Training time:    0.16s
Anomalies found:  2 (1.33%)
Score range:      [-0.575, -0.386]
============================================================

✅ Model saved successfully (version 1)
   Model path: models/isolation_forest_v1.joblib
   Scaler path: models/scaler_v1.joblib
   MLflow run ID: abc123...
   View in MLflow UI: http://localhost:5000
```

### Detection

```
Starting network monitoring...
[INFO] ML detector loaded successfully (version 1)
[ALERT] [ML] ML ANOMALY: 192.168.1.50 - Score: -0.234
[ALERT] [RULE] Traffic spike from 192.168.1.100 - Rate: 45.2 pkt/s
[ALERT] [RULE] Uncommon port 8888 from 192.168.1.75

Traffic summary:
IP: 192.168.1.45, Packets: 345
IP: 142.250.110.81, Packets: 182
Monitoring finished.
```

## View MLflow Results

### Local MLflow UI

```bash
mlflow ui --backend-store-uri file://$(pwd)/.mlflow/mlruns
# Open: http://localhost:5000
```

### Remote MLflow

If configured with remote server:
```
Open: http://your-mlflow-server:5050
```

## Common Commands

```bash
# Train with custom experiment
python train_model.py --duration 60 \
  --experiment-name "production-v1" \
  --run-name "baseline" \
  --version 1

# Train without MLflow
python train_model.py --duration 60 --no-mlflow

# Run detector for 2 minutes
sudo python main.py --duration 120

# Test MLflow connection (remote setup)
python test_mlflow_connection.py

# Generate more synthetic data
python generate_synthetic_data.py
```

## Troubleshooting

### Permission Denied

```bash
# ❌ Wrong
sudo python main.py

# ✅ Correct (use venv python with sudo)
sudo venv/bin/python main.py
```

### Insufficient Training Samples

```bash
# Error: only 12 samples collected

# Solution 1: Longer duration
sudo venv/bin/python train_model.py --duration 300

# Solution 2: Use synthetic data
python generate_synthetic_data.py
python train_model.py --from-file data/training/synthetic_baseline.csv
```

### MLflow Connection Issues

```bash
# Test connection
python test_mlflow_connection.py

# Check .env file
cat .env

# Verify environment variables
source .env && echo $MLFLOW_TRACKING_URI
```

### Model Not Found

```bash
# Check if models exist
ls -lh models/

# If missing, train first
python train_model.py --from-file data/training/synthetic_baseline.csv --version 1
```

## What's Next?

After successful setup:

1. **View Experiments**: Open MLflow UI to see training results
2. **Compare Models**: Train multiple versions and compare metrics
3. **Remote Setup**: Configure remote MLflow/MinIO for team collaboration
4. **Customize**: Adjust thresholds in `src/ml_config.py` and `src/config.py`
5. **API Usage**: See [API.md](API.md) for programmatic usage

## Quick Reference

| Task | Command |
|------|---------|
| Train (synthetic) | `python train_model.py --from-file data/training/synthetic_baseline.csv` |
| Train (live) | `sudo venv/bin/python train_model.py --duration 60` |
| Run detector | `sudo venv/bin/python main.py` |
| MLflow UI | `mlflow ui --backend-store-uri file://$(pwd)/.mlflow/mlruns` |
| Test connection | `python test_mlflow_connection.py` |
| Run tests | `pytest tests/ -v` |

## Documentation

- [README.md](README.md) - Complete overview
- [INSTALL.md](INSTALL.md) - Detailed installation guide
- [API.md](API.md) - API documentation
- [REMOTE_MLFLOW_SETUP.md](REMOTE_MLFLOW_SETUP.md) - Remote server setup

---

**Status**: ✅ Phase 1 complete - Isolation Forest + MLflow integrated and ready!

Need help? Check the [troubleshooting section](#troubleshooting) or see full documentation in [README.md](README.md).
