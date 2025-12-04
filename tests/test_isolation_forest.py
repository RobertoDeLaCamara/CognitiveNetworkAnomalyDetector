"""Unit tests for the Isolation Forest detector."""

import pytest
import numpy as np
import os
import tempfile

import sys
sys.path.insert(0, 'src')

from src.isolation_forest_detector import IsolationForestDetector
from src.ml_config import N_FEATURES


class TestIsolationForestDetector:
    """Tests for IsolationForestDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return IsolationForestDetector(contamination=0.01, n_estimators=10)
    
    @pytest.fixture
    def training_data(self):
        """Create synthetic training data."""
        np.random.seed(42)
        # Normal data clustered around origin
        normal = np.random.randn(100, N_FEATURES) * 0.5
        return normal
    
    @pytest.fixture
    def anomaly_data(self):
        """Create synthetic anomaly data."""
        np.random.seed(42)
        # Anomalies far from origin
        anomalies = np.random.randn(10, N_FEATURES) * 5 + 10
        return anomalies
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.contamination == 0.01
        assert detector.n_estimators == 10
        assert detector.is_trained == False
    
    def test_train_valid_data(self, detector, training_data):
        """Test training with valid data."""
        detector.train(training_data)
        
        assert detector.is_trained == True
        assert detector.scaler is not None
    
    def test_train_insufficient_samples(self, detector):
        """Test training with too few samples."""
        insufficient_data = np.random.randn(5, N_FEATURES)
        
        with pytest.raises(ValueError, match="at least 10 samples"):
            detector.train(insufficient_data)
    
    def test_train_wrong_shape(self, detector):
        """Test training with wrong feature count."""
        wrong_shape = np.random.randn(100, N_FEATURES + 5)
        
        with pytest.raises(ValueError, match="Expected.*features"):
            detector.train(wrong_shape)
    
    def test_predict_before_training(self, detector):
        """Test prediction without training."""
        features = np.random.randn(N_FEATURES)
        
        with pytest.raises(RuntimeError, match="must be trained"):
            detector.predict(features)
    
    def test_predict_valid(self, detector, training_data):
        """Test prediction after training."""
        detector.train(training_data)
        
        # Test on normal sample
        normal_sample = np.random.randn(N_FEATURES) * 0.5
        prediction, score = detector.predict(normal_sample)
        
        assert prediction in [1, -1]
        assert isinstance(score, (float, np.floating))
    
    def test_detect_anomalies(self, detector, training_data, anomaly_data):
        """Test that detector finds anomalies."""
        detector.train(training_data)
        
        # Test on anomaly
        anomaly_sample = anomaly_data[0]
        prediction, score = detector.predict(anomaly_sample)
        
        # Anomaly should have prediction == -1 or low score
        assert prediction == -1 or score < 0
    
    def test_predict_batch(self, detector, training_data):
        """Test batch prediction."""
        detector.train(training_data)
        
        test_data = np.random.randn(20, N_FEATURES)
        predictions, scores = detector.predict_batch(test_data)
        
        assert len(predictions) == 20
        assert len(scores) == 20
        assert all(p in [1, -1] for p in predictions)
    
    def test_save_and_load(self, detector, training_data):
        """Test model persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.joblib')
            scaler_path = os.path.join(tmpdir, 'test_scaler.joblib')
            
            # Train and save
            detector.train(training_data)
            detector.save(model_path, scaler_path)
            
            # Load into new detector
            new_detector = IsolationForestDetector()
            new_detector.load(model_path, scaler_path)
            
            assert new_detector.is_trained == True
            
            # Test that predictions match
            test_sample = np.random.randn(N_FEATURES)
            pred1, score1 = detector.predict(test_sample)
            pred2, score2 = new_detector.predict(test_sample)
            
            assert pred1 == pred2
            assert score1 == pytest.approx(score2)
    
    def test_load_nonexistent_model(self, detector):
        """Test loading model that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            detector.load('/nonexistent/model.joblib', '/nonexistent/scaler.joblib')
    
    def test_model_info(self, detector, training_data):
        """Test model info property."""
        # Before training
        info = detector.model_info
        assert info['is_trained'] == False
        
        # After training
        detector.train(training_data, feature_names=['f1', 'f2'])
        info = detector.model_info
        assert info['is_trained'] == True
        assert info['n_features'] == N_FEATURES
        assert info['feature_names'] == ['f1', 'f2']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
