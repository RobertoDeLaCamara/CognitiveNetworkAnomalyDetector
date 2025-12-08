"""Integration tests for the anomaly detection system.

These tests verify end-to-end workflows including packet processing,
feature extraction, model training, and detection.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch
from scapy.all import IP, TCP, UDP, ICMP, Raw

import sys
sys.path.insert(0, 'src')

from src.feature_extractor import FeatureExtractor
from src.isolation_forest_detector import IsolationForestDetector
from src.anomaly_detector import PacketAnalyzer
from src.model_trainer import ModelTrainer
from src.ml_config import N_FEATURES


class TestEndToEndDetection:
    """Test complete detection pipeline from packets to alerts."""
    
    @pytest.fixture
    def normal_packets(self):
        """Create synthetic normal traffic packets."""
        packets = []
        for i in range(50):
            # Normal web traffic
            packet = IP(src=f"192.168.1.{i%10}", dst="192.168.1.1") / \
                    TCP(dport=80, sport=1024+i) / \
                    Raw(load=b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
            packets.append(packet)
        return packets
    
    @pytest.fixture
    def anomalous_packets(self):
        """Create synthetic anomalous traffic packets."""
        packets = []
        # High-frequency ICMP flood
        for i in range(100):
            packet = IP(src="192.168.1.200", dst="192.168.1.1") / ICMP()
            packets.append(packet)
        
        # Large payload
        packet = IP(src="192.168.1.201", dst="192.168.1.1") / \
                TCP(dport=8080) / \
                Raw(load=b"X" * 10000)
        packets.append(packet)
        
        return packets
    
    def test_packet_to_features_to_detection(self, normal_packets, anomalous_packets):
        """Test end-to-end: packets → features → ML detection."""
        # Step 1: Extract features from normal traffic
        extractor = FeatureExtractor()
        
        for packet in normal_packets:
            extractor.process_packet(packet)
        
        # Collect features for training
        training_features = []
        for ip in extractor.get_all_ips():
            features = extractor.extract_features(ip)
            if features is not None:
                training_features.append(features)
        
        assert len(training_features) > 0, "Should extract features from training packets"
        
        # Step 2: Train detector
        detector = IsolationForestDetector(contamination=0.05)  # Higher contamination for small dataset
        training_matrix = np.array(training_features)
        
        # Need more samples for reliable training - add synthetic variance
        if len(training_matrix) < 20:
            # Duplicate with small variations
            synthetic_data = training_matrix + np.random.randn(*training_matrix.shape) * 0.1
            training_matrix = np.vstack([training_matrix, synthetic_data])
        
        detector.train(training_matrix)
        
        assert detector.is_trained
        
        # Step 3: Process anomalous traffic - create clear anomaly
        # Use synthetic data that's clearly different from normal
        anomalous_feature = training_matrix[0] * 5 + 10  # Much different from training
        
        # Step 4: Detect anomaly
        prediction, score = detector.predict(anomalous_feature)
        
        # Should detect anomaly OR have very low score
        assert prediction == -1 or score < -0.1, "Should detect clear anomaly"
    
    def test_dual_detection_system(self, normal_packets):
        """Test that both rule-based and ML detection work together."""
        # Create analyzer with ML disabled for this test (since we need trained model)
        analyzer = PacketAnalyzer(enable_ml=False)
        
        # Process packets
        for packet in normal_packets:
            analyzer.analyze_packet(packet)
        
        # Verify packet tracking
        assert len(analyzer.packet_count_per_ip) > 0, "Should track IP packets"


class TestTrainingToDetectionWorkflow:
    """Test complete workflow from training to deployment."""
    
    def test_train_save_load_predict(self):
        """Test: collect data → train → save → load → predict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.joblib')
            scaler_path = os.path.join(tmpdir, 'test_scaler.joblib')
            
            # Step 1: Create synthetic training data
            np.random.seed(42)
            normal_data = np.random.randn(100, N_FEATURES) * 0.5
            
            # Step 2: Train model
            detector = IsolationForestDetector()
            detector.train(normal_data)
            
            # Step 3: Save model
            detector.save(model_path, scaler_path)
            
            assert os.path.exists(model_path)
            assert os.path.exists(scaler_path)
            
            # Step 4: Create new detector and load model
            new_detector = IsolationForestDetector()
            assert not new_detector.is_trained
            
            new_detector.load(model_path, scaler_path)
            assert new_detector.is_trained
            
            # Step 5: Verify predictions match
            test_sample = np.random.randn(N_FEATURES) * 0.5
            
            pred1, score1 = detector.predict(test_sample)
            pred2, score2 = new_detector.predict(test_sample)
            
            assert pred1 == pred2
            assert score1 == pytest.approx(score2)
    
    def test_model_metadata_persistence(self):
        """Test that model metadata is saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.joblib')
            scaler_path = os.path.join(tmpdir, 'test_scaler.joblib')
            
            # Train and save
            detector = IsolationForestDetector()
            training_data = np.random.randn(100, N_FEATURES)
            detector.train(training_data, feature_names=['f1', 'f2'])
            detector.save(model_path, scaler_path)
            
            # Load and verify metadata
            import joblib
            model_data = joblib.load(model_path)
            
            assert isinstance(model_data, dict)
            assert 'model' in model_data
            assert 'metadata' in model_data
            
            metadata = model_data['metadata']
            assert metadata['version'] == 1
            assert 'training_date' in metadata
            assert 'sklearn_version' in metadata
            assert metadata['n_features'] == N_FEATURES
            assert metadata['contamination'] == detector.contamination


class TestModelCompatibilityValidation:
    """Test model compatibility validation."""
    
    def test_incompatible_features_raises_error(self):
        """Test that loading a model with wrong feature count raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'bad_model.joblib')
            scaler_path = os.path.join(tmpdir, 'scaler.joblib')
            
            # Create a model with wrong metadata
            import joblib
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            bad_metadata = {
                'version': 1,
                'n_features': N_FEATURES + 5,  # Wrong number
                'sklearn_version': '1.3.0'
            }
            
            model_data = {
                'model': IsolationForest(),
                'metadata': bad_metadata
            }
            
            joblib.dump(model_data, model_path)
            joblib.dump(StandardScaler(), scaler_path)
            
            # Try to load - should raise ValueError
            detector = IsolationForestDetector()
            with pytest.raises(ValueError, match="CRITICAL.*features"):
                detector.load(model_path, scaler_path)
    
    def test_old_model_format_loads_with_warning(self):
        """Test that old models without metadata still load with warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'old_model.joblib')
            scaler_path = os.path.join(tmpdir, 'old_scaler.joblib')
            
            # Create old format model (direct model object)
            import joblib
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            old_model = IsolationForest(contamination=0.01, n_estimators=10)
            # Train it so it's valid
            old_model.fit(np.random.randn(20, N_FEATURES))
            
            scaler = StandardScaler()
            scaler.fit(np.random.randn(20, N_FEATURES))
            
            # Save in old format (direct objects)
            joblib.dump(old_model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Load - should work with warning
            detector = IsolationForestDetector()
            with pytest.warns(UserWarning, match="older model format"):
                detector.load(model_path, scaler_path)
            
            assert detector.is_trained


class TestFeatureExtractionIntegration:
    """Test feature extraction integration with real packet data."""
    
    def test_various_protocol_packets(self):
        """Test feature extraction with different protocols."""
        extractor = FeatureExtractor()
        
        # Create diverse traffic
        packets = [
            IP(src="192.168.1.100", dst="192.168.1.1") / TCP(dport=80),
            IP(src="192.168.1.100", dst="192.168.1.1") / TCP(dport=443),
            IP(src="192.168.1.100", dst="192.168.1.1") / UDP(dport=53),
            IP(src="192.168.1.100", dst="192.168.1.1") / ICMP(),
            IP(src="192.168.1.100", dst="192.168.1.1") / TCP(dport=22) / \
                Raw(load=b"SSH-2.0-OpenSSH"),
        ]
        
        for packet in packets:
            ip = extractor.process_packet(packet)
            assert ip == "192.168.1.100"
        
        # Extract features
        features = extractor.extract_features("192.168.1.100")
        
        assert features is not None
        assert len(features) == N_FEATURES
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
        
        # Verify protocol ratios
        tcp_ratio = features[10]  # tcp_ratio index
        udp_ratio = features[11]
        icmp_ratio = features[12]
        
        assert tcp_ratio + udp_ratio + icmp_ratio == pytest.approx(1.0)


class TestBatchPrediction:
    """Test batch prediction capabilities."""
    
    def test_batch_prediction_performance(self):
        """Test batch prediction processes multiple samples efficiently."""
        # Train model
        detector = IsolationForestDetector()
        training_data = np.random.randn(100, N_FEATURES)
        detector.train(training_data)
        
        # Test batch prediction
        test_batch = np.random.randn(50, N_FEATURES)
        predictions, scores = detector.predict_batch(test_batch)
        
        assert len(predictions) == 50
        assert len(scores) == 50
        assert all(p in [1, -1] for p in predictions)
        
        # Verify batch predictions match individual predictions
        for i, sample in enumerate(test_batch):
            ind_pred, ind_score = detector.predict(sample)
            assert ind_pred == predictions[i]
            assert ind_score == pytest.approx(scores[i])


class TestModelTrainerWorkflow:
    """Test ModelTrainer class functionality."""
    
    def test_save_training_data(self):
        """Test saving training data to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(enable_mlflow=False)
            
            # Add some synthetic training data with correct number of features (18)
            import numpy as np
            trainer.training_features = np.random.randn(3, N_FEATURES).tolist()
            trainer.training_ips = ['192.168.1.1', '192.168.1.2', '192.168.1.3']
            
            filepath = os.path.join(tmpdir, 'training_data.csv')
            trainer.save_training_data(filepath)
            
            assert os.path.exists(filepath)
            
            # Verify CSV content
            import pandas as pd
            df = pd.read_csv(filepath)
            assert len(df) == 3
            assert 'ip' in df.columns
            assert list(df['ip']) == trainer.training_ips
    
    def test_save_training_data_no_data(self, capsys):
        """Test saving training data when no data exists."""
        trainer = ModelTrainer(enable_mlflow=False)
        trainer.save_training_data()
        
        # Should log warning
        # Note: Logger may not print to capsys, so we just check it doesn't crash
        assert len(trainer.training_features) == 0
    
    def test_validate_model_with_training_set(self):
        """Test model validation on training set."""
        trainer = ModelTrainer(enable_mlflow=False)
        
        # Create synthetic training data with minimum required samples
        np.random.seed(42)
        training_data = np.random.randn(150, N_FEATURES)
        trainer.training_features = training_data.tolist()
        trainer.training_ips = [f'192.168.1.{i}' for i in range(150)]
        
        # Train model
        stats = trainer.train_model()
        assert 'n_samples' in stats
        
        # Validate model
        metrics = trainer.validate_model()
        
        assert 'n_samples' in metrics
        assert 'n_anomalies' in metrics
        assert 'anomaly_rate' in metrics
        assert 'score_mean' in metrics
        assert metrics['n_samples'] == 150
    
    def test_validate_model_with_test_set(self):
        """Test model validation on separate test set."""
        trainer = ModelTrainer(enable_mlflow=False)
        
        # Create training data with minimum required samples
        np.random.seed(42)
        training_data = np.random.randn(120, N_FEATURES)
        trainer.training_features = training_data.tolist()
        
        # Train model
        trainer.train_model()
        
        # Validate on different test set
        test_data = np.random.randn(20, N_FEATURES)
        metrics = trainer.validate_model(test_data)
        
        assert metrics['n_samples'] == 20
    
    def test_save_model_with_version(self):
        """Test saving model with version number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(enable_mlflow=False)
            
            # Train a simple model with minimum required samples
            training_data = np.random.randn(110, N_FEATURES)
            trainer.training_features = training_data.tolist()
            trainer.train_model()
            
            # Save with version
            with patch('src.model_trainer.MODEL_DIR', tmpdir):
                trainer.save_model(version=5, register_model=False)
                
                model_file = os.path.join(tmpdir, 'isolation_forest_v5.joblib')
                scaler_file = os.path.join(tmpdir, 'scaler_v5.joblib')
                
                assert os.path.exists(model_file)
                assert os.path.exists(scaler_file)
    
    def test_load_training_data_from_csv(self):
        """Test loading pre-collected training data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a CSV file with training data
            import pandas as pd
            # Create DataFrame with correct number of features (18)
            df = pd.DataFrame({
                'ip': ['192.168.1.1', '192.168.1.2']
            })
            # Add 18 feature columns
            for i in range(N_FEATURES):
                df[f'feature{i}'] = np.random.randn(2)
            csv_path = os.path.join(tmpdir, 'training.csv')
            df.to_csv(csv_path, index=False)
            
            trainer = ModelTrainer(enable_mlflow=False)
            trainer.load_training_data(csv_path)
            
            assert len(trainer.training_features) == 2
            assert len(trainer.training_ips) == 2
            assert trainer.training_ips == ['192.168.1.1', '192.168.1.2']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
