# ML Pipeline

## Three-Engine Architecture

| Engine | Algorithm | Weight | Input | Minimum data |
|--------|-----------|--------|-------|-------------|
| Isolation Forest | sklearn IsolationForest | 40% | 18 host features | 10 packets/IP |
| LSTM Autoencoder | PyTorch 2-layer LSTM AE | 40% | 18-dim sequence (len 50) | 50 packets/IP |
| Rule-Based | Threshold + regex heuristics | 20% | Raw packet + features | 1 packet |

**Ensemble threshold**: 0.6 (configurable via `ENSEMBLE_ANOMALY_THRESHOLD`)

Dynamic redistribution: if LSTM buffer < 50 (warming up), its 40% redistributed proportionally to IF and Rules.

---

## 18 Per-IP Host Features

Extracted by `src/ml/feature_extractor.py` from a per-IP packet history (sliding window, last 1000 packets, 60s temporal window).

| # | Feature | Category |
|---|---------|----------|
| 0 | packets/sec | Statistical |
| 1 | bytes/sec | Statistical |
| 2 | avg packet size | Statistical |
| 3 | packet size variance | Statistical |
| 4 | total packets | Statistical |
| 5 | total bytes | Statistical |
| 6 | IAT mean | Temporal |
| 7 | IAT std | Temporal |
| 8 | burst rate (last 5s) | Temporal |
| 9 | session duration | Temporal |
| 10 | TCP ratio | Protocol |
| 11 | UDP ratio | Protocol |
| 12 | ICMP ratio | Protocol |
| 13 | unique dst ports | Port |
| 14 | uncommon port ratio | Port |
| 15 | avg entropy (Shannon) | Payload |
| 16 | avg payload size | Payload |
| 17 | payload size variance | Payload |

Minimum 3 packets required before extraction. `PacketHistory` dataclass accumulates raw packet metadata per IP.

---

## Isolation Forest Engine

File: `src/ml/isolation_forest_detector.py`

```python
# Training config (src/config/ml_config.py)
contamination = 0.01      # assume 1% of traffic is anomalous
n_estimators  = 100       # trees in forest
random_state  = seeded    # reproducibility

# Inference
decision_score = model.decision_function(features)  # typically [-0.5, 0.5]
normalized     = (0.5 - decision_score).clip(0, 1)  # → [0,1], higher = more anomalous
```

**Score normalization**: `(0.5 - d)` maps: d=0.5 (normal) → 0.0; d=−0.5 (anomalous) → 1.0.

**Loading fallback chain**:
1. MLflow registry (if `MLFLOW_ENABLE_REMOTE_LOADING=true`)
2. Local joblib file (`models/isolation_forest_v{MODEL_VERSION}.joblib`)
3. ML detection disabled (rules-only fallback)

---

## LSTM Autoencoder Engine

File: `src/ml/lstm_autoencoder_detector.py`

### Architecture

```
Input: sequence of shape (LSTM_SEQUENCE_LENGTH=50, 18)
  ↓
Encoder LSTM (hidden=64, layers=2)
  ↓
Latent projection: Dense → 32
  ↓
Decoder LSTM (hidden=64, layers=2, dropout=0.2)
  ↓
Output: reconstructed sequence of shape (50, 18)

Loss: MSE(input, output)
```

### Configuration (env vars / ml_config.py)

| Parameter | Default |
|-----------|---------|
| LSTM_SEQUENCE_LENGTH | 50 |
| LSTM_HIDDEN_DIM | 64 |
| LSTM_LATENT_DIM | 32 |
| LSTM_NUM_LAYERS | 2 |
| LSTM_DROPOUT | 0.2 |
| LSTM_BATCH_SIZE | 32 |
| LSTM_LEARNING_RATE | 0.001 |
| LSTM_EPOCHS | 50 |

### Per-IP Buffers

Each source IP gets its own `deque(maxlen=LSTM_SEQUENCE_LENGTH)`. `predict_single(vector)` appends to the buffer and runs inference only when the buffer is exactly full. Partial buffers produce no score.

### Hot Reload

```python
# Every 60s in _model_monitor_thread:
current_mtime = os.path.getmtime(model_path)
if current_mtime != self._last_mtime:
    self._load_model()   # in-place reload, no service interruption
    self._last_mtime = current_mtime
```

Supports live model replacement without restarting the detector.

---

## Rule-Based Engine

File: `src/detection/payload_analyzer.py` + `src/detection/anomaly_detector.py`

### Threshold Rules

| Rule | Condition | Default |
|------|-----------|---------|
| Traffic spike | current_rate > avg_rate × THRESHOLD_MULTIPLIER | 2.0× |
| ICMP flood | ICMP count in window > ICMP_THRESHOLD | 50 |
| Uncommon port | TCP/UDP to port not in HIGH_TRAFFIC_PORTS | see below |
| Large payload | payload size > PAYLOAD_THRESHOLD | 100 bytes |

`HIGH_TRAFFIC_PORTS` whitelist (14 ports): 80, 443, 53, 22, 21, 25, 587, 993, 995, 3306, 5432, 6379, 27017, 8080

### Malicious Pattern Categories

30+ patterns across 6 categories, matching raw payload bytes with 1-second per-pattern ReDoS timeout:

| Category | Example patterns |
|----------|----------------|
| SQL injection | `UNION SELECT`, `'; DROP TABLE`, `1=1` |
| Command injection | `; wget `, `&& chmod`, `/bin/bash -c` |
| Web shells | `<?php system(`, `eval(base64_decode(` |
| XSS | `<script>alert(`, `javascript:`, `<img onerror=` |
| Directory traversal | `../../../etc/passwd` |
| File inclusion | `php://filter`, `file:///etc/passwd` |

Patterns sorted longest-first to match specific before generic (reduces false positives).

---

## Training Workflow

```bash
# Step 1: Generate synthetic baseline (normal traffic profile)
python scripts/generate_synthetic_data.py
# Output: data/training/synthetic_baseline.csv

# Step 2: Train Isolation Forest
python scripts/train_model.py \
  --from-file data/training/synthetic_baseline.csv \
  --version 1
# Output: models/isolation_forest_v1.joblib, models/scaler_v1.joblib

# Step 3: Train LSTM Autoencoder
python scripts/train_lstm_model.py \
  --from-file data/training/synthetic_baseline.csv
# Output: models/lstm_autoencoder.pt, models/lstm_config.json
```

With MLflow enabled:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
python scripts/train_model.py --from-file data/training/synthetic_baseline.csv --version 1
```

---

## Ensemble Scorer

File: `src/detection/ensemble_scorer.py`

```python
@dataclass
class EnsembleResult:
    confidence_score: float      # combined [0,1]
    is_anomaly: bool             # score >= ENSEMBLE_ANOMALY_THRESHOLD
    engines: dict                # {"isolation_forest": {...}, "lstm": {...}, "rules": {...}}
```

### Scoring Logic

```
if IF_score is None:   redistribute IF weight to LSTM + Rules
if LSTM_score is None: redistribute LSTM weight to IF + Rules

combined = Σ (adjusted_weight[e] × score[e])

is_anomaly = combined >= ENSEMBLE_ANOMALY_THRESHOLD (default 0.6)
```

### Alert Types

| Type | Source |
|------|--------|
| `RULE` | Rule-based engine only |
| `ML` | Single ML engine |
| `ML_ENSEMBLE` | Combined ML + rules |
| `ICMP_FLOOD` | ICMP flood rule specifically |
