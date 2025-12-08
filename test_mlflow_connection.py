#!/usr/bin/env python3
"""Test connectivity to remote MLflow server and MinIO storage.

This script validates:
1. MLflow tracking server connection
2. MinIO S3 connectivity and credentials
3. Ability to create experiments and log artifacts
"""

import sys
import os
from pathlib import Path

# Add src to path securely
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir / 'src'))

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    import boto3
    from botocore.exceptions import ClientError
    from src.mlflow_config import (
        get_tracking_uri,
        is_remote_tracking,
        MINIO_ENDPOINT,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        S3_BUCKET_NAME,
        validate_remote_config,
        apply_s3_config
    )
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("\nPlease install required packages:")
    print("  pip install mlflow boto3")
    sys.exit(1)


def test_mlflow_server():
    """Test connection to MLflow tracking server."""
    print("\n" + "="*60)
    print("Testing MLflow Tracking Server")
    print("="*60)
    
    tracking_uri = get_tracking_uri()
    print(f"Tracking URI: {tracking_uri}")
    
    if not is_remote_tracking():
        print("⚠️  Using local file-based tracking (not remote server)")
        return False
    
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        
        # Try to list experiments
        experiments = client.search_experiments()
        print(f"✅ Connected successfully!")
        print(f"   Found {len(experiments)} experiments")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def test_minio_connection():
    """Test connection to MinIO S3 storage."""
    print("\n" + "="*60)
    print("Testing MinIO S3 Storage")
    print("="*60)
    
    if not MINIO_ENDPOINT:
        print("⚠️  MinIO endpoint not configured")
        print("   Set MLFLOW_S3_ENDPOINT_URL environment variable")
        return False
    
    print(f"Endpoint: {MINIO_ENDPOINT}")
    print(f"Bucket: {S3_BUCKET_NAME}")
    
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("❌ MinIO credentials not set")
        print("   Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False
    
    try:
        # Create S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        # Try to list buckets
        response = s3_client.list_buckets()
        buckets = [b['Name'] for b in response['Buckets']]
        
        print(f"✅ Connected successfully!")
        print(f"   Available buckets: {', '.join(buckets)}")
        
        # Check if our bucket exists
        if S3_BUCKET_NAME in buckets:
            print(f"   ✅ Bucket '{S3_BUCKET_NAME}' exists")
        else:
            print(f"   ⚠️  Bucket '{S3_BUCKET_NAME}' not found")
            print(f"       Create it in MinIO UI or it will be created automatically")
        
        return True
        
    except ClientError as e:
        print(f"❌ Connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_end_to_end():
    """Test complete workflow: create experiment and log artifact."""
    print("\n" + "="*60)
    print("Testing End-to-End Workflow")
    print("="*60)
    
    try:
        # Apply S3 configuration
        apply_s3_config()
        
        # Set tracking URI
        mlflow.set_tracking_uri(get_tracking_uri())
        
        # Create test experiment
        experiment_name = "connection-test"
        print(f"Creating test experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        
        # Start a run and log something
        with mlflow.start_run(run_name="connectivity-test") as run:
            # Log parameter
            mlflow.log_param("test_param", "connection_test")
            
            # Log metric
            mlflow.log_metric("test_metric", 1.0)
            
            # Create and log artifact
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("MLflow connection test successful!")
                temp_file = f.name
            
            try:
                mlflow.log_artifact(temp_file, "test_artifacts")
            finally:
                Path(temp_file).unlink()  # Clean up
            
            print(f"✅ Test run created successfully!")
            print(f"   Run ID: {run.info.run_id}")
            print(f"   Experiment ID: {run.info.experiment_id}")
            
            if is_remote_tracking():
                print(f"\n   View in MLflow UI:")
                print(f"   {get_tracking_uri()}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all connectivity tests."""
    print("="*60)
    print("MLflow and MinIO Connectivity Test")
    print("="*60)
    
    # Validate configuration first
    is_valid, issues = validate_remote_config()
    if not is_valid and is_remote_tracking():
        print("\n⚠️  Configuration issues detected:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease check your environment variables.")
        print("See .env.example for required variables.\n")
    
    # Run tests
    mlflow_ok = test_mlflow_server()
    minio_ok = test_minio_connection()
    
    if mlflow_ok and (minio_ok or not MINIO_ENDPOINT):
        e2e_ok = test_end_to_end()
    else:
        e2e_ok = False
        print("\n⚠️  Skipping end-to-end test due to connection failures")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"MLflow Server:  {'✅ PASS' if mlflow_ok else '❌ FAIL'}")
    print(f"MinIO Storage:  {'✅ PASS' if minio_ok else '❌ FAIL' if MINIO_ENDPOINT else '⚠️  NOT CONFIGURED'}")
    print(f"End-to-End:     {'✅ PASS' if e2e_ok else '❌ FAIL'}")
    print("="*60)
    
    if mlflow_ok and e2e_ok:
        print("\n✅ All tests passed! You can now train models with remote tracking.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
