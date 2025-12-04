# Anomaly Detector

This project is a Python-based anomaly detector with both rule-based and **ML-based detection**. It analyzes network traffic to identify unusual patterns or outliers using:
- **Rule-based detection**: Pattern matching, threshold monitoring, protocol analysis
- **ML-based detection**: Isolation Forest algorithm for unsupervised anomaly detection

## Features

### Phase 1: ML-Enhanced Detection
- ✅ **18-feature extraction pipeline**: Statistical, temporal, protocol, port, and payload features
- ✅ **Isolation Forest model**: Unsupervised learning for anomaly detection
- ✅ **Dual detection system**: ML and rule-based alerts work together
- ✅ **Model training pipeline**: Collect baseline traffic and train custom models
- ✅ **Comprehensive testing**: 20+ unit tests with >85% coverage

## Installation

### With Docker

To build and run the application using Docker, use the following commands:

```bash
docker build -t anomaly_detector .
docker run anomaly_detector
```

### Local Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/RobertoDeLaCamara/DetectorAnomalias.git
    cd DetectorAnomalias
    ```

2.  Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## ML Model Training

Before using ML-based detection, you need to train a model on baseline **normal** traffic:

### Option 1: Collect Live Traffic (Recommended)

```bash
# Collect 60 seconds of normal traffic and train model
sudo python3 train_model.py --duration 60 --contamination 0.01
```

### Option 2: Train from Existing Data

```bash
# Train from pre-collected CSV data
python3 train_model.py --from-file data/training/baseline_traffic.csv
```

### Training Parameters

- `--duration N`: Collect baseline traffic for N seconds (requires root/sudo)
- `--contamination X`: Expected anomaly proportion (default: 0.01 = 1%)
- `--version V`: Model version number (default: 1)
- `--from-file PATH`: Load training data from CSV file
- `--no-save-data`: Don't save collected training data
- `--interface IFACE`: Network interface to monitor (default: all)

## Usage

### Run Anomaly Detection

To run the anomaly detector with ML detection:

```bash
# Run with default settings (60 seconds monitoring)
sudo python3 main.py
```

The detector will:
1. Load the trained ML model (if available)
2. Monitor network traffic
3. Extract features from packets
4. Run both rule-based and ML-based detection
5. Log alerts for anomalies

### Configuration

Edit `src/config.py` and `src/ml_config.py` to adjust:
- **Rule-based thresholds**: `THRESHOLD_MULTIPLIER`, `ICMP_THRESHOLD`, etc.
- **ML settings**: `ML_ENABLED`, `MIN_PACKETS_FOR_ML`, `ML_ANOMALY_THRESHOLD`
- **Feature extraction**: `FEATURE_WINDOW_SIZE`, `CONTAMINATION`

### Disable ML Detection

```python
# In src/config.py
ML_ENABLED = False
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_feature_extractor.py -v
pytest tests/test_isolation_forest.py -v
```

## Project Structure

```
cognitive-anomaly-detector-1/
├── src/
│   ├── anomaly_detector.py         # Main detection logic (rule + ML)
│   ├── feature_extractor.py         # 18-feature extraction pipeline
│   ├── isolation_forest_detector.py # Isolation Forest implementation
│   ├── model_trainer.py             # Training workflow
│   ├── ml_config.py                 # ML configuration
│   ├── config.py                    # General configuration
│   ├── payload_analyzer.py          # Pattern matching
│   └── logger_setup.py              # Logging configuration
├── tests/
│   ├── test_feature_extractor.py    # Feature extraction tests
│   ├── test_isolation_forest.py     # ML model tests
│   ├── test_anomaly_detector.py     # Integration tests
│   └── test_payload_analyzer.py     # Pattern matching tests
├── models/                          # Trained models (created after training)
├── data/                            # Training data (created after collection)
├── train_model.py                   # ML training script
├── main.py                          # Main entry point
└── requirements.txt                 # Dependencies

```

## How It Works

### Rule-Based Detection
- Traffic spikes (rate > 2x average)
- ICMP flood detection
- Uncommon port usage
- Large payload detection
- Malicious pattern matching (SQL injection, XSS, etc.)

### ML-Based Detection (Isolation Forest)
1. **Feature Extraction**: Extract 18 features per IP:
   - Statistical: packet/byte rates, sizes, variance
   - Temporal: inter-arrival times, burst rates, session duration
   - Protocol: TCP/UDP/ICMP ratios
   - Port: unique ports, uncommon port ratio
   - Payload: entropy, size statistics

2. **Anomaly Scoring**: Isolation Forest assigns anomaly scores
   - Lower scores = more anomalous
   - Threshold-based alerting

3. **Dual Detection**: Both rule-based and ML alerts are logged separately

## Example Output

```
Starting local network monitoring for 60 seconds...
[INFO] ML detector loaded successfully
[ALERT] [ML] ML ANOMALY DETECTED: 192.168.1.50 - Anomaly score: -0.234
[ALERT] [RULE] ALERT: Traffic spike from 192.168.1.100 - Rate: 45.2 packets/sec
[ALERT] [RULE] ALERT: Malicious payload detected from 192.168.1.200
```

## Roadmap

- ✅ Phase 1: Foundation & Isolation Forest (COMPLETED)
- ⏳ Phase 2: LSTM Networks & Ensemble Learning
