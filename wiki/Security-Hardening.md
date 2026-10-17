# Security Hardening

Security measures implemented throughout the codebase.

> `src/config/security_config.py` defines limits, IP ranges and `sanitize_for_logging()`, but no runtime module imports it (only `tests/core/test_security_config.py` does). The checks below are the ones that actually run.

## Input Validation

| Input | Validation | Location |
|-------|-----------|---------|
| Source IP | `ipaddress` parse; loopback, link-local and `TRUSTED_SUBNETS` are skipped | `anomaly_detector._validate_ip_address` |
| Network interface name (`main.py --interface`) | Regex `^[a-zA-Z0-9_-]+$`, max 15 chars, no leading `-`, must exist (psutil) | `main.py` |
| `--duration` | 1-3600 seconds | `main.py`, `scripts/train_model.py` |
| Model file path | No `..` or leading `~`, extension must be `.joblib` | `IsolationForestDetector._secure_path` |
| Model file | Max 100 MB and pickle header check before load | `IsolationForestDetector._validate_file_integrity` |
| Training CSV | `.csv`, no `..`/`~`, max 100 MB | `scripts/train_model.py` (`validate_args`) |
| Experiment name | Alphanumeric, `-`, `_`, max 100 chars | `scripts/train_model.py` |
| S3 endpoint / AWS keys | Format checks | `mlflow_config.get_s3_config` |

## Payload Scan Timeout

Payload patterns are plain byte substrings (case-insensitive), not regexes. The scan runs in a daemon thread with a 1-second timeout over at most 1024 bytes; if the thread is still alive the payload is reported as `TIMEOUT_ATTACK`:

```python
# src/detection/payload_analyzer.py
thread = threading.Thread(target=pattern_match)
thread.daemon = True
thread.start()
thread.join(timeout=PATTERN_MATCH_TIMEOUT)   # 1 second
if thread.is_alive():
    return True, "TIMEOUT_ATTACK"
```

## Model Loading (pickle)

Models are loaded with plain `joblib.load` after the size and header checks above. There is no unpickler allowlist: only load model files from trusted sources.

## Memory Limits

| Resource | Limit | Enforcement |
|----------|-------|------------|
| Tracked IPs (analyzer and extractor) | `max_ips` = 1 000 by default, hard-capped at 5 000 | Eviction by lowest packet count: bottom 10% in `PacketAnalyzer`, bottom 20% in `FeatureExtractor` |
| Packets per IP (history) | 1 000 | deque maxlen |
| Packets per IP (analyzer) | 10 000 | packets beyond this are ignored |
| Total packets | 50 000 (extractor), 100 000 (analyzer) | Counter check |
| Model file size | 100 MB | Checked before loading |
| Log file size | 5 MB, 3 backups (`LOG_MAX_SIZE`, `LOG_BACKUP_COUNT`) | RotatingFileHandler |
| Training file size | 100 MB | Checked before reading |
| Packet queue | 10 000 (`PACKET_QUEUE_SIZE`) | Excess packets are dropped and counted |

Eviction removes the least-active IPs (by packet count), not the oldest, which preserves history for high-traffic IPs.

`ResourceMonitor` (`src/core/resource_monitor.py`, limits 512 MB / 80% CPU) lets `analyze_packet` skip packets while the process is over its limits.

## Thread Safety

`src/ml/feature_extractor.py` uses `threading.RLock()` on all operations that read or write the per-IP history map. The LSTM sequence buffer is guarded by its own lock.

## Logging

`ANOMALY_LOG_FILE` is validated for traversal and invalid characters, the log directory is created with mode `0o750` and the log file is set to `0o644`. Log messages are truncated (subject 200 / body 500 characters) but not redacted: IP addresses and paths appear in the log.

## Alert Rate Limiting

A per-(IP, alert_type) rate limiter prevents alert flooding (hard-coded in `PacketAnalyzer`):

```python
self.alert_cooldown = 60      # seconds
# max 3 alerts per (ip, alert_type) inside that window
```

ML inference still runs on every packet — only the alert dispatch is rate-limited. This prevents spam while maintaining detection coverage.

## Privileges

| Operation | Requires Root |
|-----------|--------------|
| Live packet capture (`main.py`) | Yes (Scapy raw sockets) |
| Training from file | No |
| Dashboard (`dashboard.py`) | No |
| Docker detector / trainer services | They run as `root` (`user: root`), the detector also `privileged: true`, with host networking |

The principle of least privilege recommendation: run `main.py` with `CAP_NET_RAW` only, not full root, in production environments that support capability-based privilege.
