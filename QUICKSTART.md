# Quick Start: Model Training and Testing

## ✅ Environment Setup Complete

All dependencies installed successfully:
- scikit-learn 1.7.2
- numpy 2.3.5  
- pandas 2.3.3
- joblib 1.5.2

**Test Results**: 22/22 ML tests passed ✅

## Next Steps

### 1. Train the Isolation Forest Model

You need baseline **normal** traffic to train the model. Run this command:

```bash
# Option A: Collect 60 seconds of live traffic (RECOMMENDED)
sudo ./venv/bin/python train_model.py --duration 60

# Option B: Longer collection for better baseline (2 minutes)
sudo ./venv/bin/python train_model.py --duration 120

# Option C: Train from existing CSV data
./venv/bin/python train_model.py --from-file data/training/baseline.csv
```

**Important**: 
- Use `sudo` for packet capture
- Use `./venv/bin/python` to run with venv python under sudo
- Make sure you have normal network traffic during collection

### 2. Expected Training Output

```
[INFO] Collecting baseline traffic for 60 seconds...
[INFO] Collected 45 unique IPs
[INFO] Extracted features for 45 IPs
[INFO] Training Isolation Forest...

============================================================
TRAINING COMPLETED SUCCESSFULLY
============================================================
Samples:          45
Features:         18
Training time:    0.12s
Anomalies found:  1 (2.22%)
Score range:      [-0.234, 0.156]
============================================================

✅ Model saved successfully (version 1)
   Model path: models/isolation_forest_v1.joblib
   Scaler path: models/scaler_v1.joblib
```

### 3. Run the Anomaly Detector

Once model is trained:

```bash
sudo ./venv/bin/python main.py
```

Expected output:
```
Starting local network monitoring for 60 seconds...
[INFO] ML detector loaded successfully
[ALERT] [ML] ML ANOMALY DETECTED: 192.168.1.50 - Anomaly score: -0.234
[ALERT] [RULE] ALERT: Traffic spike from 192.168.1.100
```

### 4. Verify Model Files

After training, check:
```bash
ls -lh models/
# Should show:
# isolation_forest_v1.joblib
# isolation_forest_v1_metadata.joblib  
# scaler_v1.joblib
```

## Troubleshooting

**No network traffic during collection?**
- Try: `ping google.com` in another terminal
- Or browse some websites during the 60 seconds

**Permission denied?**
- Make sure to use `sudo` for packet capture
- Use `sudo ./venv/bin/python` not just `sudo python`

**Insufficient samples error?**
- Increase duration: `--duration 120`
- Ensure network has active traffic

## What's Next

After successful training:
1. ✅ Model will auto-load on next run
2. ✅ ML + rule-based detection run in parallel
3. ✅ Alerts tagged with `[ML]` or `[RULE]`
4. 🎯 Ready for Phase 2 (LSTM, Ensemble Learning)

---

**Status**: Phase 1 implementation complete, ready for model training!
