# Remote MLflow and MinIO Setup Guide

## Overview

This guide helps you configure the project to use your existing remote MLflow server and MinIO storage.

## Your Infrastructure

- **MLflow Server**: http://<MLFLOW_SERVER_IP>:5050
- **MinIO Web UI**: http://<MINIO_SERVER_IP>:9001/browser/mlflow-artifacts
- **MinIO API**: http://<MINIO_SERVER_IP>:9000 (not 9001!)

> **Note**: MinIO uses port 9000 for API access and port 9001 for the web UI. Replace `<MLFLOW_SERVER_IP>` and `<MINIO_SERVER_IP>` with your actual server IP addresses.

## Quick Setup (5 minutes)

### Step 1: Get MinIO Credentials

1. Open MinIO web UI: http://<MINIO_SERVER_IP>:9001
2. Log in with your MinIO administrator credentials
3. Navigate to **Access Keys** or **Service Accounts**
4. Create a new access key or use an existing one
5. Copy the **Access Key** and **Secret Key**

### Step 2: Create .env File

Create a file named `.env` in the project root:

```bash
cd /home/roberto/cognitive-anomaly-detector
cp .env.example .env
```

### Step 3: Edit .env File

Edit `.env` and fill in your MinIO credentials:

```bash
# MLflow and MinIO Configuration
MLFLOW_TRACKING_URI=http://<MLFLOW_SERVER_IP>:5050
MLFLOW_S3_ENDPOINT_URL=http://<MINIO_SERVER_IP>:9000
MLFLOW_S3_BUCKET=mlflow-artifacts

# Replace these with your actual MinIO credentials
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
```

### Step 4: Install boto3

```bash
source venv/bin/activate
pip install boto3
```

### Step 5: Test Connection

```bash
python test_mlflow_connection.py
```

You should see:
```
✅ MLflow Server:  PASS
✅ MinIO Storage:  PASS
✅ End-to-End:     PASS
```

## Usage

### Training with Remote Tracking

Once configured, simply run training as usual:

```bash
python train_model.py --duration 60 --version 1
```

The model will automatically:
- Log experiments to your MLflow server (http://<MLFLOW_SERVER_IP>:5050)
- Store artifacts in MinIO (s3://mlflow-artifacts/)

### View Results

Open your MLflow UI:
```
http://<MLFLOW_SERVER_IP>:5050
```

View artifacts in MinIO:
```
http://<MINIO_SERVER_IP>:9001/browser/mlflow-artifacts
```

## Environment Variables Reference

| Variable | Purpose | Example |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://<MLFLOW_SERVER_IP>:5050` |
| `MLFLOW_S3_ENDPOINT_URL` | MinIO API endpoint | `http://<MINIO_SERVER_IP>:9000` |
| `MLFLOW_S3_BUCKET` | S3 bucket name | `mlflow-artifacts` |
| `AWS_ACCESS_KEY_ID` | MinIO access key | (from MinIO UI) |
| `AWS_SECRET_ACCESS_KEY` | MinIO secret key | (from MinIO UI) |

## Switching Between Local and Remote

### Use Remote (default with .env file)
```bash
python train_model.py --duration 60
```

### Force Local (ignore .env)
```bash
unset MLFLOW_TRACKING_URI
python train_model.py --duration 60
```

Or temporarily:
```bash
MLFLOW_TRACKING_URI="" python train_model.py --duration 60
```

## Troubleshooting

### Connection Test Fails

**Issue**: Cannot connect to MLflow server
- Check MLflow server is running: `curl http://<MLFLOW_SERVER_IP>:5050`
- Verify firewall allows access to port 5050

**Issue**: Cannot connect to MinIO
- Verify MinIO is running: `curl http://<MINIO_SERVER_IP>:9000`
- Check credentials are correct
- Ensure using port 9000 (API) not 9001 (UI)

### Bucket Not Found

If the bucket doesn't exist:
1. Open MinIO UI: http://<MINIO_SERVER_IP>:9001
2. Create bucket named `mlflow-artifacts`
3. Or MLflow will create it automatically on first use

### Credentials Error

```
❌ AWS_ACCESS_KEY_ID not set for MinIO
```

Solution: Ensure .env file has valid credentials and no typos

## Security Notes

- ⚠️ Never commit `.env` file to git (it's in `.gitignore`)
- 🔒 Keep MinIO credentials secure
- 🌐 Consider using HTTPS for production deployments

## Next Steps

After successful setup:
1. Train your first model with remote tracking
2. Explore experiments in MLflow UI
3. Share tracking URL with your team
4. Set up model staging/production workflows

---

**Need Help?**
Run the connection test to diagnose issues:
```bash
python test_mlflow_connection.py
```
