# Security Notes

## Hardening Applied

| Area | What was done |
|---|---|
| Credentials | Removed hardcoded credentials; `.env` uses placeholders only |
| Path traversal | Model paths validated (no `..` or leading `~`), extension restricted to `.joblib`; training CSV paths reject `..` and `~` |
| File size | 100 MB limit on model file loading to prevent DoS |
| Input validation | Experiment/run names restricted to alphanumeric; interface names validated |
| Pattern-scan DoS | Payloads are truncated to 1024 bytes and scanned in a thread with a 1-second timeout (`src/detection/payload_analyzer.py`); on timeout it reports `TIMEOUT_ATTACK` |
| Command injection | Network interface parameter validated with strict regex before use |
| Model file checks | IF model/scaler files: size limit 100 MB and a pickle-header check before `joblib.load`. There is no unpickler allowlist: loading is not sandboxed |
| Memory | Bounded packet queue and per-IP data cleanup prevent unbounded growth |
| Thread safety | RLock on shared feature extractor data structures |
| File permissions | Model and log directories created with `0o750`; log file set to `0o644` |
| Log path validation | `ANOMALY_LOG_FILE` is checked for traversal and invalid characters. Log messages are not redacted |
| Unused module | `src/config/security_config.py` (limits, IP ranges, `sanitize_for_logging`) is not imported by the runtime code; only its tests use it. The effective rate limit is hard-coded in `PacketAnalyzer` (3 alerts per 60 s per IP and alert type) |

## Operational Notes

- The `.env` file is in `.gitignore` — never commit it.
- `main.py` requires root only for packet capture. Training from file needs no sudo.
- Models loaded with joblib/pickle — only load files from trusted sources.
- IP addresses in logs may be personal data under GDPR; apply appropriate retention policies.

## Running with Least Privilege

```bash
# Detection (root needed for raw packet capture)
sudo venv/bin/python main.py

# Training from file (no root needed)
python scripts/train_model.py --from-file data/training/synthetic_baseline.csv
```

## Checklist

- [ ] `.env` not committed to version control
- [ ] Real credentials not in any committed file
- [ ] `models/` directory has restricted permissions (`chmod 750 models/`)
- [ ] MLflow server uses authentication in production
- [ ] Log files don't expose sensitive payload content
