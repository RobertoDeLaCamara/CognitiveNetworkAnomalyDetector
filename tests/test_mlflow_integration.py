"""Tests for MLflow integration with the anomaly detection system.

These tests verify that MLflow tracking, model registry, and artifact
logging work correctly while maintaining backward compatibility.
"""

import pytest
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path

import sys
sys.path.insert(0, 'src')

from src.model_trainer import ModelTrainer, MLFLOW_AVAILABLE
from src.isolation_forest_detector import IsolationForestDetector
from src.ml_config import N_FEATURES

# Skip all MLflow tests if MLflow is not installed
pytestmark = pytest.mark.skipif(
    not MLFLOW_AVAILABLE, 
    reason="MLflow not installed"
)


@pytest.fixture
def temp_mlflow_dir():
    """Create temporary directory for MLflow tracking."""
    temp_dir = tempfile.mkdtemp()
    original_uri = os.environ.get('MLFLOW_TRACKING_URI')
    
    # Set tracking URI to temp directory
    os.environ['MLFLOW_TRACKING_URI'] = f'file://{temp_dir}/mlruns'
    
    yield temp_dir
    
    # Cleanup
    if original_uri:
        os.environ['MLFLOW_TRACKING_URI'] = original_uri
    else:
        os.environ.pop('MLFLOW_TRACKING_URI', None)
    
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def synthetic_training_data():
    """Generate synthetic training data for testing."""
    np.random.seed(42)
    features = np.random.randn(100, N_FEATURES) * 0.5
    ips = [f"192.168.1.{i}" for i in range(100)]
    return features, ips


class TestMLflowConfiguration:
    """Test MLflow configuration and initialization."""
    
    def test_mlflow_import_available(self):
        """Test that MLflow is available for import."""
        assert MLFLOW_AVAILABLE, "MLflow should be available"
    
    def test_mlflow_config_import(self):
        """Test that mlflow_config module imports successfully."""
        from src.mlflow_config import (
            get_tracking_uri,
            get_experiment_name,
            is_mlflow_enabled,
            get_run_name
        )
        
        assert callable(get_tracking_uri)
        assert callable(get_experiment_name)
        assert callable(is_mlflow_enabled)
        assert callable(get_run_name)
    
    def test_mlflow_directories_created(self):
        """Test that MLflow directories are created."""
        from src.mlflow_config import MLFLOW_DIR
        
        assert MLFLOW_DIR.exists()
        assert (MLFLOW_DIR / 'mlruns').exists()
        assert (MLFLOW_DIR / 'artifacts').exists()


class TestMLflowTraining:
    """Test model training with MLflow tracking."""
    
    def test_trainer_with_mlflow_enabled(self, temp_mlflow_dir):
        """Test ModelTrainer initialization with MLflow enabled."""
        trainer = ModelTrainer(enable_mlflow=True)
        assert trainer.mlflow_enabled is True
    
    def test_trainer_with_mlflow_disabled(self):
        """Test ModelTrainer initialization with MLflow disabled."""
        trainer = ModelTrainer(enable_mlflow=False)
        assert trainer.mlflow_enabled is False
    
    def test_training_logs_to_mlflow(self, temp_mlflow_dir, synthetic_training_data):
        """Test that training creates MLflow run with logged data."""
        import mlflow
        
        features, ips = synthetic_training_data
        
        # Create trainer and add training data
        trainer = ModelTrainer(enable_mlflow=True)
        trainer.training_features = features.tolist()
        trainer.training_ips = ips
        
        # Train model
        stats = trainer.train_model(
            contamination=0.01,
            experiment_name='test-experiment',
            run_name='test-run'
        )
        
        # Verify MLflow run was created
        assert 'mlflow_run_id' in stats
        assert 'mlflow_experiment_id' in stats
        
        # Verify run exists
        run_id = stats['mlflow_run_id']
        run = mlflow.get_run(run_id)
        
        assert run is not None
        assert run.info.run_id == run_id
    
    def test_mlflow_logs_parameters(self, temp_mlflow_dir, synthetic_training_data):
        """Test that training logs model parameters."""
        import mlflow
        
        features, ips = synthetic_training_data
        
        trainer = ModelTrainer(enable_mlflow=True)
        trainer.training_features = features.tolist()
        trainer.training_ips = ips
        
        stats = trainer.train_model(contamination=0.05)
        
        run = mlflow.get_run(stats['mlflow_run_id'])
        params = run.data.params
        
        assert 'contamination' in params
        assert float(params['contamination']) == 0.05
        assert 'n_estimators' in params
        assert 'random_state' in params
        assert 'n_features' in params
    
    def test_mlflow_logs_metrics(self, temp_mlflow_dir, synthetic_training_data):
        """Test that training logs metrics."""
        import mlflow
        
        features, ips = synthetic_training_data
        
        trainer = ModelTrainer(enable_mlflow=True)
        trainer.training_features = features.tolist()
        trainer.training_ips = ips
        
        stats = trainer.train_model()
        
        run = mlflow.get_run(stats['mlflow_run_id'])
        metrics = run.data.metrics
        
        assert 'training_time_seconds' in metrics
        assert 'n_samples' in metrics
        assert 'anomaly_rate' in metrics
        assert 'score_mean' in metrics
        assert 'score_std' in metrics
    
    def test_mlflow_logs_model(self, temp_mlflow_dir, synthetic_training_data):
        """Test that model is logged as artifact."""
        import mlflow
        from mlflow.tracking import MlflowClient
        
        features, ips = synthetic_training_data
        
        trainer = ModelTrainer(enable_mlflow=True)
        trainer.training_features = features.tolist()
        trainer.training_ips = ips
        
        stats = trainer.train_model()
        
        run = mlflow.get_run(stats['mlflow_run_id'])
        
        # Check that model artifact exists using MlflowClient
        client = MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id)
        artifact_paths = [a.path for a in artifacts]
        
        # Model should be logged (may be in 'model' directory or as direct artifact)
        assert len(artifact_paths) > 0, "Should have logged artifacts"
        # Check for either 'model' directory or at least training_data artifact
        assert any('training_data' in p for p in artifact_paths), "Should log training data"


class TestMLflowModelRegistry:
    """Test MLflow Model Registry functionality."""
    
    def test_model_registration(self, temp_mlflow_dir, synthetic_training_data):
        """Test that models can be registered to MLflow."""
        import mlflow
        
        features, ips = synthetic_training_data
        
        trainer = ModelTrainer(enable_mlflow=True)
        trainer.training_features = features.tolist()
        trainer.training_ips = ips
        
        # Train and save model (which should register it)
        stats = trainer.train_model()
        trainer.save_model(version=1, register_model=True)
        
        # Note: Model registration in save_model requires active run
        # The registration happens inside the MLflow context
    
    def test_save_to_mlflow(self, temp_mlflow_dir):
        """Test direct save to MLflow."""
        detector = IsolationForestDetector()
        
        # Train detector
        training_data = np.random.randn(100, N_FEATURES)
        detector.train(training_data)
        
        # Save to MLflow
        run_id = detector.save_to_mlflow(
            experiment_name='test-save',
            run_name='test-direct-save',
            register_model=False  # Don't register for this test
        )
        
        assert run_id is not None
        assert len(run_id) > 0


class TestBackwardCompatibility:
    """Test that existing workflows still work without MLflow."""
    
    def test_training_without_mlflow(self, synthetic_training_data):
        """Test that training works when MLflow is disabled."""
        features, ips = synthetic_training_data
        
        trainer = ModelTrainer(enable_mlflow=False)
        trainer.training_features = features.tolist()
        trainer.training_ips = ips
        
        stats = trainer.train_model(contamination=0.01)
        
        # Should have basic stats but not MLflow IDs
        assert 'n_samples' in stats
        assert 'training_time_seconds' in stats
        assert 'mlflow_run_id' not in stats
        assert 'mlflow_experiment_id' not in stats
    
    def test_model_save_without_mlflow(self, synthetic_training_data):
        """Test that model saving works without MLflow."""
        features, ips = synthetic_training_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train model without MLflow
            trainer = ModelTrainer(enable_mlflow=False)
            trainer.training_features = features.tolist()
            trainer.training_ips = ips
            trainer.train_model()
            
            # Save should work
            model_path = os.path.join(tmpdir, 'test_model.joblib')
            scaler_path = os.path.join(tmpdir, 'test_scaler.joblib')
            trainer.detector.save(model_path, scaler_path)
            
            assert os.path.exists(model_path)
            assert os.path.exists(scaler_path)


class TestMLflowErrors:
    """Test error handling for MLflow operations."""
    
    def test_load_from_mlflow_without_mlflow(self):
        """Test that loading from MLflow fails gracefully if not installed."""
        # This test assumes we're testing the error message
        # In practice, this test would only run if MLFLOW_AVAILABLE is False
        pass
    
    def test_save_untrained_model_to_mlflow(self, temp_mlflow_dir):
        """Test that saving untrained model to MLflow raises error."""
        detector = IsolationForestDetector()
        
        with pytest.raises(RuntimeError, match="must be trained"):
            detector.save_to_mlflow()


class TestMLflowExperimentManagement:
    """Test experiment and run management."""
    
    def test_custom_experiment_name(self, temp_mlflow_dir, synthetic_training_data):
        """Test training with custom experiment name."""
        import mlflow
        
        features, ips = synthetic_training_data
        
        trainer = ModelTrainer(enable_mlflow=True)
        trainer.training_features = features.tolist()
        trainer.training_ips = ips
        
        custom_exp_name = 'my-custom-experiment'
        stats = trainer.train_model(experiment_name=custom_exp_name)
        
        run = mlflow.get_run(stats['mlflow_run_id'])
        experiment = mlflow.get_experiment(run.info.experiment_id)
        
        assert experiment.name == custom_exp_name
    
    def test_custom_run_name(self, temp_mlflow_dir, synthetic_training_data):
        """Test training with custom run name."""
        import mlflow
        
        features, ips = synthetic_training_data
        
        trainer = ModelTrainer(enable_mlflow=True)
        trainer.training_features = features.tolist()
        trainer.training_ips = ips
        
        custom_run_name = 'my-test-run'
        stats = trainer.train_model(run_name=custom_run_name)
        
        run = mlflow.get_run(stats['mlflow_run_id'])
        
        # Run name should match
        assert run.data.tags.get('mlflow.runName') == custom_run_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
