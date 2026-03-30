# Security Hardening

Security measures implemented in `src/config/security_config.py` and throughout the codebase.

## Input Validation

| Input | Validation | Location |
|-------|-----------|---------|
| IP address | `ipaddress` module + reserved address check | `security_config.py` |
| Network interface name | Regex `^[a-zA-Z0-9_-]+$`, max 15 chars | `security_config.py` |
| Model file path | No `..` allowed, restricted to project directory | `security_config.py` |
| Feature values | Range check, max 10 for anomaly_score | `security_config.py` |
| Training file size | Max 500 MB | `security_config.py` |

## ReDoS Prevention

Payload pattern matching runs each regex in a daemon thread with a 1-second timeout:

```python
# src/detection/payload_analyzer.py
def _match_with_timeout(pattern, payload, timeout=1.0):
    result = []
    def _run():
        if re.search(pattern, payload):
            result.append(True)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return bool(result)
```

Without this guard, a crafted packet could trigger catastrophic backtracking, blocking all packet processing indefinitely.

## Pickle Safety

Models loaded via `joblib` use a custom safe unpickler that restricts deserializable classes to an explicit allowlist:

```python
ALLOWED_MODULES = {
    'sklearn.ensemble',
    'sklearn.preprocessing',
    'numpy',
    'numpy.core.multiarray',
}
```

Prevents arbitrary code execution via maliciously crafted model files.

## Memory Limits

| Resource | Limit | Enforcement |
|----------|-------|------------|
| Tracked IPs (analyzer) | 10 000 | Hard cap with LRU-style eviction |
| Packets per IP (history) | 1 000 | deque maxlen |
| Total packets per extractor | 50 000 | Counter check |
| Model file size | 100 MB | Checked before loading |
| Log file size | 5 MB | RotatingFileHandler (3 backups) |
| Training file size | 500 MB | Checked before reading |

When the IP limit is reached, the least-active 20% of IPs (by packet count) are removed, not the oldest. This preserves behavioral history for high-traffic IPs.

## Thread Safety

`src/ml/feature_extractor.py` uses `threading.RLock()` on all operations that read or write the per-IP history map. Deep copies of history data are made before passing to ML engines to prevent race conditions between the collector and inference threads.

## Log Sanitization

Before logging, IP addresses and file paths are redacted:

```python
# src/core/logger_setup.py
def sanitize_for_log(message: str) -> str:
    # Replace IPv4 addresses with [REDACTED-IP]
    # Replace absolute paths with [REDACTED-PATH]
    return sanitized
```

Prevents log injection and reduces exposure of network topology in log files.

## Alert Rate Limiting

A per-(IP, alert_type) rate limiter prevents alert flooding:

```python
MAX_ALERTS_PER_WINDOW = 3
ALERT_WINDOW_SECONDS  = 60
```

ML inference still runs on every packet — only the alert dispatch is rate-limited. This prevents spam while maintaining detection coverage.

## Privileges

| Operation | Requires Root |
|-----------|--------------|
| Live packet capture (`main.py`) | Yes (Scapy raw sockets) |
| Training from file | No |
| Dashboard (`dashboard.py`) | No |
| Docker detector service | No (host-mode network handles it) |

The principle of least privilege recommendation: run `main.py` with `CAP_NET_RAW` only, not full root, in production environments that support capability-based privilege.
