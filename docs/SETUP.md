# Setup Guide

## Local Install

### Prerequisites
- Python 3.8+
- `libpcap` (for packet capture): `sudo apt install libpcap-dev` / `brew install libpcap`
- Root/sudo access for live traffic capture

### Steps

```bash
git clone https://github.com/RobertoDeLaCamara/DetectorAnomalias.git
cd cognitive-anomaly-detector

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train and run

```bash
# Generate synthetic baseline data and train
python scripts/generate_synthetic_data.py
python scripts/train_model.py --from-file data/training/synthetic_baseline.csv --version 1

# Start detection (requires root for packet capture)
sudo venv/bin/python main.py
```

### Troubleshooting

**Permission denied on `sudo python`**
Always use the venv interpreter with sudo:
```bash
sudo venv/bin/python main.py     # correct
sudo python main.py              # wrong — uses system Python
```

**Too few training samples**
```bash
# Use synthetic data instead of live capture
python scripts/generate_synthetic_data.py
python scripts/train_model.py --from-file data/training/synthetic_baseline.csv
```

**Model not found**
```bash
ls models/    # check if isolation_forest_v1.joblib exists
# If not, run training first
```

---

## Docker

### Build and run

```bash
# Build image
docker-compose build

# Run dashboard → http://localhost:8501
docker-compose up -d dashboard

# Run detector (uses host networking for packet capture)
docker-compose up -d detector
docker-compose logs -f detector

# One-off training job
docker-compose run --rm trainer --from-file data/training/synthetic_baseline.csv --version 1
docker-compose run --rm trainer --duration 60 --version 1

# Stop everything
docker-compose down
```

### Notes
- The `detector` service uses `network_mode: host` to capture host traffic.
- Volumes `./models`, `./data`, and `./.ML Tracking` are mounted for persistence.
- Copy `.env.example` to `.env` before starting if you use remote ML Tracking.

---

## Remote ML Tracking + S3-compatible storage

The project supports a remote ML Tracking tracking server with S3-compatible storage artifact storage. This is optional — without it, experiments are tracked locally in `.ML Tracking/`.

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

```bash
ML Tracking_TRACKING_URI=http://<ML Tracking-server>:5050
ML Tracking_S3_ENDPOINT_URL=http://<S3-compatible storage-server>:9000
ML Tracking_S3_BUCKET=ML Tracking-artifacts
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

> S3-compatible storage uses port **9000** for the API and **9001** for the web UI.

### Get S3-compatible storage credentials
1. Open S3-compatible storage UI: `http://<S3-compatible storage-server>:9001`
2. Go to **Access Keys** → create or copy an existing key

### Test the connection

```bash
python scripts/test_ML Tracking_connection.py
# Expected:
# ✅ ML Tracking Server:  PASS
# ✅ S3-compatible storage Storage:  PASS
# ✅ End-to-End:     PASS
```

### Use remote vs local

```bash
# Remote (default when .env is configured)
python scripts/train_model.py --duration 60 --version 1

# Force local for a single run
ML Tracking_TRACKING_URI="" python scripts/train_model.py --duration 60 --no-ML Tracking

# View local experiments
ML Tracking ui --backend-store-uri file://$(pwd)/.ML Tracking/mlruns
# → http://localhost:5000
```

### Troubleshooting

| Error | Fix |
|---|---|
| Cannot connect to ML Tracking | Check server is up: `curl http://<server>:5050` |
| Cannot connect to S3-compatible storage | Use port 9000, not 9001 |
| Bucket not found | Create `ML Tracking-artifacts` bucket in S3-compatible storage UI, or ML Tracking will create it on first use |
| Credentials error | Verify `.env` has no typos; re-run `test_ML Tracking_connection.py` |
