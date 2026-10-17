# Architecture & Data Flow

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Network Interface  (Scapy sniff() — root/sudo required)        │
└────────────────────────────┬────────────────────────────────────┘
                             │ raw packets
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PacketProcessor  (src/core/packet_queue.py)                    │
│  Bounded queue.Queue: 10 000 packets (PACKET_QUEUE_SIZE)        │
│  Worker threads: 2 (configurable via PACKET_WORKERS)            │
│  Dropped packets tracked; overflow does not block capture       │
└────────────────────────────┬────────────────────────────────────┘
                             │ packets
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  analyze_packet() callback   (src/detection/anomaly_detector.py)│
│                                                                  │
│  1. Feature Extractor (src/ml/feature_extractor.py)             │
│     → per-IP history (≤1000 pkts), compute 18-dim vector        │
│     → minimum 3 packets required before extraction              │
│                                                                  │
│  2. Rule-Based Engine — immediate pattern matching:             │
│     ├─ Traffic spike: rate > avg × THRESHOLD_MULTIPLIER (2.0)   │
│     ├─ ICMP flood: count > ICMP_THRESHOLD (50)                  │
│     ├─ Uncommon ports: TCP/UDP not in 8-port whitelist           │
│     ├─ Large payload: size > PAYLOAD_THRESHOLD (100 B)          │
│     └─ Malicious patterns: 40 byte substrings, case-insensitive │
│        (scan thread with 1 s timeout)                            │
│                                                                  │
│  3. Isolation Forest  (src/ml/isolation_forest_detector.py)     │
│     → predict(18-dim vector) if MIN_PACKETS_FOR_ML (10) met     │
│     → returns (prediction, raw score_samples); normalized to    │
│       [0,1] by EnsembleScorer as clip(0.5 - score)              │
│                                                                  │
│  4. LSTM Autoencoder  (src/ml/lstm_autoencoder_detector.py)     │
│     → append vector to ONE shared sequence buffer (maxlen=50)   │
│     → if buffer full: predict_single(vector)                    │
│     → reconstruction error / (2 × threshold), clipped to [0,1]  │
│                                                                  │
│  5. Ensemble Scorer  (src/detection/ensemble_scorer.py)         │
│     → weighted fusion: IF×0.4 + LSTM×0.4 + Rules×0.2          │
│     → dynamic redistribution if engine unavailable              │
│     → EnsembleResult(confidence_score, is_anomaly, engines)     │
└────────────────────────────┬────────────────────────────────────┘
                             │
         [confidence_score ≥ 0.6 AND alert cooldown not hit]
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Alert Dispatch                                                  │
│  - Rate limit: max 3 alerts per (IP, type) per 60s              │
│  - Console stdout: [ALERT] [TYPE] subject - body                │
│  - Log file: anomaly_detection.log (5 MB rolling, 3 backups)    │
│    (rule alerts are not gated on the 0.6 ensemble threshold)    │
│  - SQLite: anomalies.db (indexed on timestamp, ip_address)      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                  Streamlit Dashboard :8501
```

## Detection Decision Flow

```
For each packet from source IP X:
  Update IP history (≤1000 packets per IP; when more than max_ips (1000, capped at 5000) IPs are
  tracked, the feature extractor evicts the least-active 20% and the analyzer's counters the bottom 10%)
    ↓
  extract_features(X)  →  18-dim vector (or None if < 3 packets)
    ↓
  [Rule engine runs on raw packet regardless of packet count]
    ↓
  if packet_count(X) ≥ MIN_PACKETS_FOR_ML (10):
    IF.predict(vector)    →  (label, score_samples)
    LSTM.predict_single(vector)  →  (0, 0.0) until the shared 50-vector buffer is full, then error
    ↓
  Ensemble.combine(if, lstm, rules)   (rules score = 1.0 if any rule fired for this packet else 0.0)
    ↓
  if combined_score ≥ 0.6:
    check_alert_cooldown(X, alert_type)
      → if allowed: dispatch alert
```

## Queue Details

Plain threads and a `queue.Queue` (no asyncio):

```python
# src/core/packet_queue.py
self._queue = queue.Queue(maxsize=self.max_queue_size)  # PACKET_QUEUE_SIZE, 10 000

def enqueue(self, packet):
    try:
        self._queue.put_nowait(packet)
    except queue.Full:
        self._dropped_count += 1  # tracked, not fatal

def _worker_loop(self):
    while not self._stop_event.is_set():
        packet = self._queue.get(timeout=0.5)
        self.callback(packet)          # analyze_packet
        self._queue.task_done()
```

Two daemon worker threads by default. Increase `PACKET_WORKERS` for high-throughput networks.

## Docker Compose Services

| Service | Role | Notes |
|---------|------|-------|
| trainer | One-off training job | `--duration N` or `--from-file`; runs as root with host network; entrypoint is `scripts/train_model.py` |
| detector | Long-running detection | `restart: unless-stopped`, host network, privileged; sets `MLFLOW_MODEL_STAGE=Production`, `MLFLOW_MODEL_VERSION=4` |
| dashboard | Streamlit :8501 | Reads anomalies.db (alerts) and the log file (Live Logs page) |

All services bind-mount the project directory (`.:/app`), so they share models, data and the database. Host networking is used for packet capture; training from file requires no root.
