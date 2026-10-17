# Adding a Rule or a Feature

## A new payload pattern

Payload rules are byte substrings in `MALICIOUS_PATTERNS` (`src/config/config.py`). `src/detection/payload_analyzer.py` sorts them longest-first and matches them case-insensitively against the first `MAX_PAYLOAD_SCAN_SIZE` (1024) bytes of each payload, with a 1 s timeout.

1. Add the pattern as `bytes` under the right comment group (SQL injection, command injection, XSS, traversal, ...). Prefer specific patterns: substrings are matched anywhere in the payload.
2. Add a positive test and at least one benign near-miss to `tests/detection/test_payload_analyzer.py`.
3. Run `pytest tests/detection -q`.

## A new rule check on packets

Rule checks live in `PacketAnalyzer.analyze_packet` (`src/detection/anomaly_detector.py`). Each check that fires sets `rule_fired = True`, which feeds the rules engine of the ensemble.

1. Add your check in `analyze_packet` and call `self.log_alert(subject, body, alert_type=..., ip_src=...)`.
2. Alert types are an allow-list in `log_alert` (`valid_types`): `RULE`, `ML`, `ML_ENSEMBLE`, `ICMP_FLOOD`. Any other type is stored as `UNKNOWN`. Add yours to that set if it needs its own type.
3. Alerts are rate-limited per IP and type by `_should_alert` (3 per 60 s).
4. Add tests under `tests/detection/`.

## A new ML feature

The 18 features are defined by `FEATURE_NAMES` in `src/config/ml_config.py`; `N_FEATURES` is their length. `FeatureExtractor.extract_features` (`src/ml/feature_extractor.py`) builds the vector in groups (statistical, temporal, protocol, port, payload), and raises if its length differs from `N_FEATURES`.

1. Append the name to `FEATURE_NAMES` and compute the value in the matching `_extract_*_features` method, **in the same position**. Order matters: models are trained on positions, not names.
2. Mirror the name in `FEATURE_NAMES` of `src/config/dashboard_config.py` (the dashboard keeps its own copy).
3. **Retrain both models.** Existing Isolation Forest and LSTM files expect the old number of features and will not load or score correctly. Run `scripts/train_model.py` and `scripts/train_lstm_model.py`, and bump `MODEL_VERSION` in `ml_config.py` so the detector picks up the new files (train with the same `--version`).
4. Update `scripts/generate_synthetic_data.py` if the synthetic baseline should include the new column.
5. Add tests in `tests/ml/` and run the full suite: `pytest tests -q`.

## Checklist before a pull request

- Tests for the new behaviour, including a benign case for rules.
- Docs: [Configuration](Configuration.md) and the README if a setting or file changed.
- Do not commit `.env`, `anomalies.db` or `.mlflow/` (all gitignored).
