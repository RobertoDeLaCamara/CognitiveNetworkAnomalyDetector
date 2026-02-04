"""Tests for LSTM Autoencoder anomaly detector."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.ml_config import N_FEATURES

# Skip all tests if PyTorch not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


@pytest.fixture
def detector():
    """Create a fresh LSTM Autoencoder detector for testing."""
    from src.lstm_autoencoder_detector import LSTMAutoencoderDetector
    return LSTMAutoencoderDetector(
        sequence_length=10,
        hidden_dim=32,
        latent_dim=16,
        num_layers=1,
        dropout=0.0,
    )


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    np.random.seed(42)
    return np.random.randn(100, N_FEATURES)


@pytest.fixture
def small_data():
    """Generate minimal training data."""
    np.random.seed(42)
    return np.random.randn(20, N_FEATURES)


class TestLSTMAutoencoder:
    """Tests for LSTM Autoencoder model."""
    
    def test_model_creation(self):
        """Test model can be created."""
        from src.lstm_autoencoder_detector import LSTMAutoencoder
        model = LSTMAutoencoder(
            input_dim=N_FEATURES,
            hidden_dim=32,
            latent_dim=16,
            seq_len=10,
        )
        assert model is not None
    
    def test_model_forward(self):
        """Test forward pass produces correct shapes."""
        from src.lstm_autoencoder_detector import LSTMAutoencoder
        model = LSTMAutoencoder(
            input_dim=N_FEATURES,
            hidden_dim=32,
            latent_dim=16,
            seq_len=10,
        )
        
        batch = torch.randn(4, 10, N_FEATURES)
        reconstructed, latent = model(batch)
        
        assert reconstructed.shape == batch.shape
        assert latent.shape == (4, 16)
    
    def test_reconstruction_error(self):
        """Test reconstruction error calculation."""
        from src.lstm_autoencoder_detector import LSTMAutoencoder
        model = LSTMAutoencoder(
            input_dim=N_FEATURES,
            hidden_dim=32,
            latent_dim=16,
            seq_len=10,
        )
        
        batch = torch.randn(4, 10, N_FEATURES)
        errors = model.get_reconstruction_error(batch)
        
        assert errors.shape == (4,)
        assert (errors >= 0).all()


class TestLSTMAutoencoderDetector:
    """Tests for LSTM Autoencoder Detector wrapper."""
    
    def test_detector_creation(self, detector):
        """Test detector can be created."""
        assert detector is not None
        assert not detector.is_trained
    
    def test_train(self, detector, small_data):
        """Test training completes successfully."""
        stats = detector.train(
            small_data,
            epochs=5,
            batch_size=4,
            log_to_mlflow=False,
        )
        
        assert detector.is_trained
        assert stats['n_samples'] == 20
        assert stats['epochs_trained'] <= 5
        assert stats['threshold'] > 0
    
    def test_train_insufficient_samples(self, detector):
        """Test training fails with too few samples."""
        insufficient_data = np.random.randn(5, N_FEATURES)
        
        with pytest.raises(ValueError, match="at least"):
            detector.train(insufficient_data, log_to_mlflow=False)
    
    def test_predict_requires_training(self, detector, sample_data):
        """Test prediction fails before training."""
        with pytest.raises(RuntimeError, match="not trained"):
            detector.predict(sample_data)
    
    def test_predict(self, detector, small_data):
        """Test prediction after training."""
        detector.train(small_data, epochs=3, log_to_mlflow=False)
        
        predictions, errors = detector.predict(small_data)
        
        assert len(predictions) == len(small_data) - detector.sequence_length + 1
        assert len(errors) == len(predictions)
        assert set(predictions).issubset({-1, 1})
    
    def test_predict_single(self, detector, small_data):
        """Test single prediction with buffer."""
        detector.train(small_data, epochs=3, log_to_mlflow=False)
        
        # First predictions should return 0 (insufficient data)
        for i in range(detector.sequence_length - 1):
            pred, error = detector.predict_single(small_data[i])
            assert pred == 0
            assert error == 0.0
        
        # After filling buffer, should get real prediction
        pred, error = detector.predict_single(small_data[detector.sequence_length - 1])
        assert pred in [-1, 1]
        assert error > 0
    
    def test_save_load(self, detector, small_data):
        """Test model persistence."""
        detector.train(small_data, epochs=3, log_to_mlflow=False)
        original_threshold = detector.threshold
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            detector.save(tmpdir)
            
            assert (Path(tmpdir) / 'lstm_autoencoder.pt').exists()
            assert (Path(tmpdir) / 'lstm_scaler.joblib').exists()
            assert (Path(tmpdir) / 'lstm_config.json').exists()
            
            # Load into new detector
            from src.lstm_autoencoder_detector import LSTMAutoencoderDetector
            new_detector = LSTMAutoencoderDetector()
            success = new_detector.load(tmpdir)
            
            assert success
            assert new_detector.is_trained
            assert abs(new_detector.threshold - original_threshold) < 1e-6
    
    def test_latent_representation(self, detector, small_data):
        """Test latent space extraction."""
        detector.train(small_data, epochs=3, log_to_mlflow=False)
        
        latent = detector.get_latent_representation(small_data)
        
        expected_sequences = len(small_data) - detector.sequence_length + 1
        assert latent.shape == (expected_sequences, detector.latent_dim)
