# Security Notes

## Hardening Applied

| Area | What was done |
|---|---|
| Credentials | Removed hardcoded credentials; `.env` uses placeholders only |
| Path traversal | Model paths validated (no `..` or `~`), extension restricted to `.joblib` |
| File size | 100 MB limit on model file loading to prevent DoS |
| Input validation | Experiment/run names restricted to alphanumeric; interface names validated |
| ReDoS | 1-second timeout on payload pattern matching (`src/detection/payload_analyzer.py`) |
| Command injection | Network interface parameter validated with strict regex before use |
| Pickle safety | Custom safe unpickler with module allowlist for model loading |
| Memory | Bounded packet queue and per-IP data cleanup prevent unbounded growth |
| Thread safety | RLock on shared feature extractor data structures |
| File permissions | Model directory created with `0o750` |
| Log sanitization | File paths sanitized before inclusion in log output |

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
