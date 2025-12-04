# Phase 1 Installation and Testing Guide

## Installation Steps

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show path to venv)
which python
```

### 2. Install Dependencies

```bash
# Install all dependencies including ML packages
pip install -r requirements.txt
```

Expected packages:
- scapy
- pytest
- pytest-mock
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- joblib >= 1.3.0

### 3. Verify Installation

```bash
# Check scikit-learn
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"

# Check numpy
python -c "import numpy; print(f'numpy {numpy.__version__}')"

# Check pandas
python -c "import pandas; print(f'pandas {pandas.__version__}')"
```

## Running Tests

### Run All Tests

```bash
# Activate venv first
source venv/bin/activate

# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# View coverage report
firefox htmlcov/index.html  # or your preferred browser
```

### Run Specific Tests

```bash
# Feature extraction tests
pytest tests/test_feature_extractor.py -v

# Isolation Forest tests
pytest tests/test_isolation_forest.py -v

# Existing tests (payload analyzer, etc.)
pytest tests/test_payload_analyzer.py -v
pytest tests/test_anomaly_detector.py -v
```

### Expected Test Results

```
tests/test_feature_extractor.py::TestPacketHistory::test_initialization PASSED
tests/test_feature_extractor.py::TestPacketHistory::test_entropy_calculation PASSED
tests/test_feature_extractor.py::TestFeatureExtractor::test_initialization PASSED
tests/test_feature_extractor.py::TestFeatureExtractor::test_process_packet_with_ip PASSED
tests/test_feature_extractor.py::TestFeatureExtractor::test_extract_features_valid PASSED
tests/test_feature_extractor.py::TestFeatureExtractor::test_feature_count PASSED
tests/test_feature_extractor.py::TestFeatureExtractor::test_protocol_features PASSED
tests/test_feature_extractor.py::TestFeatureExtractor::test_get_all_ips PASSED
tests/test_feature_extractor.py::TestFeatureExtractor::test_reset PASSED

tests/test_isolation_forest.py::TestIsolationForestDetector::test_initialization PASSED
tests/test_isolation_forest.py::TestIsolationForestDetector::test_train_valid_data PASSED
tests/test_isolation_forest.py::TestIsolationForestDetector::test_predict_valid PASSED
tests/test_isolation_forest.py::TestIsolationForestDetector::test_detect_anomalies PASSED
tests/test_isolation_forest.py::TestIsolationForestDetector::test_predict_batch PASSED
tests/test_isolation_forest.py::TestIsolationForestDetector::test_save_and_load PASSED
tests/test_isolation_forest.py::TestIsolationForestDetector::test_model_info PASSED
```

## Training the Model

### Collect Baseline Traffic

```bash
# Activate venv
source venv/bin/activate

# Run with sudo (required for packet capture)
sudo venv/bin/python train_model.py --duration 60 --contamination 0.01
```

**Important**: Use `sudo venv/bin/python` to ensure the virtual environment Python is used with elevated privileges.

### Training Output

```
[INFO] Collecting baseline traffic for 60 seconds...
[INFO] Collected 45 unique IPs
[INFO] Extracted features for 45 IPs
[INFO] Training Isolation Forest on 45 samples with 18 features
[INFO] Training completed in 0.12 seconds
[INFO] Anomaly rate in training set: 2.22%

============================================================
TRAINING COMPLETED SUCCESSFULLY
============================================================
Samples:          45
Features:         18
Training time:    0.12s
Anomalies found:  1 (2.22%)
Score range:      [-0.234, 0.156]
Score mean/std:   0.012 ± 0.089
============================================================

✅ Model saved successfully (version 1)
   Model path: models/isolation_forest_v1.joblib
   Scaler path: models/scaler_v1.joblib
```

## Running the Anomaly Detector

### With ML Detection (Trained Model)

```bash
# Activate venv
source venv/bin/activate

# Run detector with sudo
sudo venv/bin/python main.py
```

### Expected Output

```
Starting local network monitoring for 60 seconds...
[INFO] ML detector loaded successfully
[ALERT] [RULE] ALERT: Traffic on uncommon port 8080 from 192.168.1.100
[ALERT] [ML] ML ANOMALY DETECTED: 192.168.1.50 - Anomaly score: -0.234 (lower = more anomalous). Total packets: 45

Traffic summary:
IP: 192.168.1.100, Packets sent: 234
IP: 192.168.1.50, Packets sent: 45
IP: 192.168.1.1, Packets sent: 123
Monitoring finished.
```

### Without Trained Model

If no model is trained, the system will gracefully degrade:

```
Starting local network monitoring for 60 seconds...
[WARNING] No pre-trained model found. ML detection will be disabled. Run model training first.
[ALERT] [RULE] ALERT: Traffic spike from 192.168.1.100
```

## Troubleshooting

### Issue: ModuleNotFoundError

```
Solution: Ensure virtual environment is activated
source venv/bin/activate
```

### Issue: Permission Denied (Packet Capture)

```
Solution: Run with sudo and specify venv python
sudo venv/bin/python main.py
sudo venv/bin/python train_model.py --duration 60
```

### Issue: Insufficient Training Samples

```
Solution: Increase collection duration or ensure network has traffic
sudo venv/bin/python train_model.py --duration 120
```

### Issue: Model Not Loading

```
Check: Does models/ directory exist with .joblib files?
ls -la models/

Solution: Train model first
sudo venv/bin/python train_model.py --duration 60
```

## Deactivating Virtual Environment

```bash
deactivate
```

## Next Steps

1. ✅ Install dependencies in virtual environment
2. ✅ Run tests to verify implementation
3. ✅ Train model on baseline traffic
4. ✅ Run anomaly detector with ML detection
5. ⏭️ Proceed to Phase 2 (LSTM Networks & Ensemble Learning)
