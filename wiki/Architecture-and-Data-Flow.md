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
│  Bounded queue: 10 000 packets                                  │
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
│     ├─ Uncommon ports: TCP/UDP not in 14-port whitelist          │
│     ├─ Large payload: size > PAYLOAD_THRESHOLD (100 B)          │
│     └─ Malicious patterns: 30+ regex (1s ReDoS timeout each)    │
│                                                                  │
│  3. Isolation Forest  (src/ml/isolation_forest_detector.py)     │
│     → predict(18-dim vector) if MIN_PACKETS_FOR_ML (10) met     │
│     → returns (is_anomaly, score [0,1])                         │
│                                                                  │
│  4. LSTM Autoencoder  (src/ml/lstm_autoencoder_detector.py)     │
│     → update per-IP sequence buffer (maxlen=50)                 │
│     → if buffer full: predict_single(vector)                    │
│     → returns reconstruction error normalized to [0,1]          │
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
│  - SQLite: anomalies.db (indexed on timestamp, ip_address)      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                  Streamlit Dashboard :8501
```

## Detection Decision Flow

```
For each packet from source IP X:
  Update IP history (≤1000 packets, evict least-active if >1000 IPs)
    ↓
  extract_features(X)  →  18-dim vector (or None if < 3 packets)
    ↓
  [Rule engine runs on raw packet regardless of packet count]
    ↓
  if packet_count(X) ≥ MIN_PACKETS_FOR_ML (10):
    IF.predict(vector)    →  (label, score)
    LSTM.update(X, vector)
    if LSTM buffer full:
      LSTM.predict_single(vector)  →  error_score
    ↓
  Ensemble.score(if_score, lstm_score, rule_score)
    ↓
  if combined_score ≥ 0.6:
    check_alert_cooldown(X, alert_type)
      → if allowed: dispatch alert
```

## Async Queue Details

```python
# src/core/packet_queue.py
queue = asyncio.Queue(maxsize=PACKET_QUEUE_SIZE)  # 10 000

async def enqueue(packet):
    try:
        queue.put_nowait(packet)
    except asyncio.QueueFull:
        _dropped_count += 1  # tracked, not fatal

async def worker():
    while True:
        packet = await queue.get()
        await analyze_packet(packet)
        queue.task_done()
```

Two workers by default. Increase `PACKET_WORKERS` for high-throughput networks.

## Docker Compose Services

| Service | Role | Notes |
|---------|------|-------|
| trainer | One-off training job | `--duration N` or `--from-file` |
| detector | Long-running detection | `restart: unless-stopped`, host network |
| dashboard | Streamlit :8501 | Reads anomalies.db and log file |

All services mount the same volumes for models and database. Host networking used for packet capture; training from file requires no root.
