# ML Pipeline

## Three-Engine Architecture

| Engine | Algorithm | Weight | Input | Minimum data |
|--------|-----------|--------|-------|-------------|
| Isolation Forest | sklearn IsolationForest | 40% | 18 host features | 10 packets/IP |
| LSTM Autoencoder | PyTorch 2-layer LSTM AE | 40% | 18-dim sequence (len 50) | 50 feature vectors in one shared buffer |
| Rule-Based | Threshold + regex heuristics | 20% | Raw packet + features | 1 packet |

**Ensemble threshold**: 0.6 (configurable via `ENSEMBLE_ANOMALY_THRESHOLD`)

Dynamic redistribution: an engine that reports `prediction == 0` (no data, e.g. the LSTM buffer holds fewer than 50 vectors, or no LSTM model is loaded) is excluded and its weight is redistributed proportionally to the remaining engines.

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
| 15 | payload entropy (`payload_entropy`, mean Shannon entropy of non-zero payloads) | Payload |
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
random_state  = ML_RANDOM_STATE env, else random per process
max_samples   = 'auto'; features standardised with StandardScaler

# Inference (IsolationForestDetector.predict)
score      = model.score_samples(scaled_features)   # lower = more anomalous
normalized = clip(0.5 - score, 0, 1)                # EnsembleScorer.normalize_isolation_forest
```

**Score normalization**: `(0.5 - s)` maps s=0.5 → 0.0 and s=−0.5 (or lower) → 1.0. Note that `score_samples` is negative for normal samples too: the v5 model scored 100 synthetic baseline rows between −0.58 and −0.39 (mean normalized value ≈ 0.94). Check your model's score range before tuning weights or the threshold.

**Loading fallback chain**:
1. MLflow registry (if `MLFLOW_ENABLE_REMOTE_LOADING=true`; uses `MLFLOW_MODEL_VERSION` if set, else `MLFLOW_MODEL_STAGE`, default `Production`)
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
Decoder LSTM (hidden=64, layers=2; dropout=0.2 on both encoder and decoder LSTMs)
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
| LSTM_EARLY_STOPPING (patience) | 10 |
| LSTM_ANOMALY_PERCENTILE (threshold = this percentile of training reconstruction errors) | 95 |

These are read from environment variables in `lstm_autoencoder_detector.py` (not `ml_config.py`). Score normalization: `error / (2 × threshold)`, clipped to [0,1], so the threshold maps to 0.5.

### Sequence Buffer

The detector keeps a single `deque(maxlen=sequence_length)` shared by all source IPs (lock-protected). `predict_single(vector)` appends the vector and returns `(0, 0.0)` (no data) until the buffer is full; after that every call scores the latest 50 vectors, regardless of which IP they came from.

### Hot Reload

```python
# PacketAnalyzer._check_model_updates(), called from analyze_packet(); runs at most every 60 s
mtime = os.path.getmtime(LSTM_MODEL_PATH)
if mtime > self.lstm_model_mtime:
    self.lstm_detector.load()   # in-place reload, no restart
    self.lstm_model_mtime = mtime
```

Only the LSTM model (`models/lstm_autoencoder.pt`) is hot-reloaded; the Isolation Forest is loaded once at startup.

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
| Malicious payload | any `MALICIOUS_PATTERNS` entry found | see below |

`HIGH_TRAFFIC_PORTS` whitelist (8 ports, `src/config/config.py`): 22, 53, 80, 443, 993, 995, 587, 465. Any TCP/UDP destination port outside this list raises an "uncommon port" rule alert.

### Malicious Pattern Categories

40 byte-string patterns (`MALICIOUS_PATTERNS` in `src/config/config.py`) across 6 categories. They are matched as case-insensitive substrings on the first 1024 bytes of the payload (not regexes), in a thread with a 1-second timeout; a timeout is reported as `TIMEOUT_ATTACK`:

| Category | Example patterns |
|----------|----------------|
| SQL injection | `UNION SELECT`, `'; DROP TABLE`, `' OR 1=1--` |
| Command injection | `; wget http`, `&& chmod`, `/bin/bash -c` |
| Web application attacks | `<?php system(`, `eval(base64_decode(` |
| XSS | `<script>alert(`, `javascript:alert(`, `<img src=x onerror=` |
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
  --version 5
# Output: models/isolation_forest_v1.joblib, models/scaler_v1.joblib

# Step 3: Train LSTM Autoencoder
python scripts/train_lstm_model.py \
  --from-file data/training/synthetic_baseline.csv
# Output: models/lstm_autoencoder.pt, models/lstm_config.json
```

With a remote MLflow server (without it, runs go to the local SQLite store `.mlflow/mlflow.db`):
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
python scripts/train_model.py --from-file data/training/synthetic_baseline.csv --version 5
```

---

## Ensemble Scorer

File: `src/detection/ensemble_scorer.py`

```python
@dataclass
class EnsembleResult:
    confidence_score: float      # combined [0,1]
    is_anomaly: bool             # score >= ENSEMBLE_ANOMALY_THRESHOLD
    engines: dict                # {"isolation_forest": EngineResult, "lstm": EngineResult, "rules": EngineResult}
```

### Scoring Logic

```
engines with prediction == 0 (no data) are dropped; remaining weights are renormalised to sum to 1
rules score = 1.0 if any rule fired for the current packet, else 0.0

combined = Σ (adjusted_weight[e] × normalized_score[e])

is_anomaly = combined >= ENSEMBLE_ANOMALY_THRESHOLD (default 0.6)
```

### Alert Types

| Type | Source |
|------|--------|
| `RULE` | Rule-based engine only (traffic spike, uncommon port, large payload, malicious payload) |
| `ML` | Legacy single-engine alert: `log_ml_alert()` exists but nothing calls it, so current runs never write it (older DB rows may) |
| `ML_ENSEMBLE` | Ensemble confidence ≥ threshold (needs IF; LSTM and rules contribute when available) |
| `ICMP_FLOOD` | ICMP flood rule specifically |
