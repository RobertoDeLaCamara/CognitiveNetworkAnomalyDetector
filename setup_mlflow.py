#!/usr/bin/env python3
"""Setup script to initialize MLflow for the cognitive anomaly detector.

This script:
1. Creates necessary MLflow directories
2. Sets up the default experiment
3. Configures tracking URI
4. Displays helpful information
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    import mlflow
    from src.mlflow_config import (
        MLFLOW_DIR,
        get_tracking_uri,
        DEFAULT_EXPERIMENT_NAME,
        REGISTERED_MODEL_NAME
    )
except ImportError as e:
    print(f"Error: {e}")
    print("\nMLflow is not installed. Please install it first:")
    print("  pip install mlflow>=2.9.0")
    sys.exit(1)


def setup_mlflow():
    """Initialize MLflow infrastructure."""
    print("=" * 60)
    print("MLflow Setup for Cognitive Anomaly Detector")
    print("=" * 60)
    
    # Create directories
    print("\n1. Creating MLflow directories...")
    MLFLOW_DIR.mkdir(exist_ok=True)
    (MLFLOW_DIR / 'mlruns').mkdir(exist_ok=True)
    (MLFLOW_DIR / 'artifacts').mkdir(exist_ok=True)
    print(f"   ✓ Created {MLFLOW_DIR}")
    print(f"   ✓ Created {MLFLOW_DIR / 'mlruns'}")
    print(f"   ✓ Created {MLFLOW_DIR / 'artifacts'}")
    
    # Set tracking URI
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    print(f"\n2. MLflow tracking URI set to:")
    print(f"   {tracking_uri}")
    
    # Create default experiment
    print(f"\n3. Setting up default experiment...")
    try:
        experiment = mlflow.get_experiment_by_name(DEFAULT_EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                DEFAULT_EXPERIMENT_NAME,
                artifact_location=str(MLFLOW_DIR / 'artifacts')
            )
            print(f"   ✓ Created experiment: {DEFAULT_EXPERIMENT_NAME}")
            print(f"   Experiment ID: {experiment_id}")
        else:
            print(f"   ✓ Experiment already exists: {DEFAULT_EXPERIMENT_NAME}")
            print(f"   Experiment ID: {experiment.experiment_id}")
    except Exception as e:
        print(f"   ⚠ Warning: Could not create experiment: {e}")
    
    # Display configuration
    print("\n" + "=" * 60)
    print("MLflow Configuration")
    print("=" * 60)
    print(f"Tracking URI:       {tracking_uri}")
    print(f"Default Experiment: {DEFAULT_EXPERIMENT_NAME}")
    print(f"Model Registry:     {REGISTERED_MODEL_NAME}")
    print(f"Artifact Location:  {MLFLOW_DIR / 'artifacts'}")
    
    # Display next steps
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("\n1. Train a model with MLflow tracking:")
    print("   python train_model.py --duration 60 --version 1")
    
    print("\n2. View experiments in MLflow UI:")
    print(f"   mlflow ui --backend-store-uri file://{Path.cwd() / MLFLOW_DIR / 'mlruns'}")
    print("   Then open: http://localhost:5000")
    
    print("\n3. Train with custom experiment name:")
    print("   python train_model.py --duration 60 --experiment-name my-experiment")
    
    print("\n4. Disable MLflow for a run:")
    print("   python train_model.py --duration 60 --no-mlflow")
    
    print("\n" + "=" * 60)
    print("✓ MLflow setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        setup_mlflow()
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
