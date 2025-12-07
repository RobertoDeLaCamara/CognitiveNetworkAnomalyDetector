"""Model training pipeline for the Isolation Forest anomaly detector.

This module provides utilities for:
- Collecting baseline traffic data
- Training the Isolation Forest model
- Validating model performance
- Saving trained models
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from scapy.all import sniff

from .feature_extractor import FeatureExtractor
from .isolation_forest_detector import IsolationForestDetector
from .ml_config import (
    MIN_TRAINING_SAMPLES,
    TRAINING_DATA_DIR,
    MODEL_DIR
)
from .logger_setup import logger

# MLflow imports (optional dependency)
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    from .mlflow_config import (
        get_tracking_uri,
        get_experiment_name,
        is_mlflow_enabled,
        get_run_name,
        REGISTERED_MODEL_NAME,
        MODEL_ARTIFACT_PATH,
        DEFAULT_TAGS,
        LOG_TRAINING_DATA
    )
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.info("MLflow not available. Experiment tracking disabled.")


class ModelTrainer:
    """Handles training workflow for the Isolation Forest model."""
    
    def __init__(self, enable_mlflow: bool = None):
        """Initialize the model trainer.
        
        Args:
            enable_mlflow: Enable MLflow tracking (None = auto-detect)
        """
        self.feature_extractor = FeatureExtractor()
        self.detector = IsolationForestDetector()
        self.training_features = []
        self.training_ips = []
        
        # MLflow configuration
        if enable_mlflow is None:
            self.mlflow_enabled = MLFLOW_AVAILABLE and is_mlflow_enabled()
        else:
            self.mlflow_enabled = enable_mlflow and MLFLOW_AVAILABLE
        
        if self.mlflow_enabled:
            mlflow.set_tracking_uri(get_tracking_uri())
            logger.info(f"MLflow tracking enabled: {get_tracking_uri()}")
        else:
            logger.info("MLflow tracking disabled")
    
    def collect_baseline_traffic(
        self,
        duration: int = 60,
        interface: Optional[str] = None
    ) -> int:
        """Collect baseline traffic for training.
        
        Args:
            duration: Collection duration in seconds
            interface: Network interface to monitor (None = all)
        
        Returns:
            Number of unique IPs collected
        """
        logger.info(f"Collecting baseline traffic for {duration} seconds...")
        
        def packet_callback(packet):
            """Process each captured packet."""
            ip = self.feature_extractor.process_packet(packet)
        
        # Capture packets
        sniff(
            prn=packet_callback,
            timeout=duration,
            iface=interface,
            store=False
        )
        
        # Extract features from all collected IPs
        ips = self.feature_extractor.get_all_ips()
        logger.info(f"Collected traffic from {len(ips)} unique IPs")
        
        # Extract features
        for ip in ips:
            features = self.feature_extractor.extract_features(ip)
            if features is not None:
                self.training_features.append(features)
                self.training_ips.append(ip)
        
        logger.info(f"Extracted features for {len(self.training_features)} IPs")
        
        return len(self.training_features)
    
    def load_training_data(self, filepath: str):
        """Load pre-collected training data from file.
        
        Args:
            filepath: Path to CSV file with training data
        """
        logger.info(f"Loading training data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Extract features (assuming all columns except 'ip' are features)
        if 'ip' in df.columns:
            self.training_ips = df['ip'].tolist()
            feature_cols = [col for col in df.columns if col != 'ip']
        else:
            self.training_ips = [f"ip_{i}" for i in range(len(df))]
            feature_cols = df.columns.tolist()
        
        self.training_features = df[feature_cols].values.tolist()
        
        logger.info(
            f"Loaded {len(self.training_features)} samples with "
            f"{len(feature_cols)} features"
        )
    
    def train_model(
        self, 
        contamination: Optional[float] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None
    ) -> dict:
        """Train the Isolation Forest model on collected data.
        
        Args:
            contamination: Override default contamination parameter
            experiment_name: Custom MLflow experiment name
            run_name: Custom MLflow run name
        
        Returns:
            Dictionary with training statistics
        
        Raises:
            ValueError: If insufficient training data
        """
        if len(self.training_features) < MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"Insufficient training data: {len(self.training_features)} samples "
                f"(minimum: {MIN_TRAINING_SAMPLES})"
            )
        
        # Convert to numpy array
        features = np.array(self.training_features)
        feature_names = self.feature_extractor.get_feature_names()
        
        logger.info(
            f"Training Isolation Forest on {features.shape[0]} samples "
            f"with {features.shape[1]} features"
        )
        
        # Update contamination if provided
        if contamination is not None:
            self.detector.contamination = contamination
            self.detector.model.contamination = contamination
        
        # Start MLflow run if enabled
        if self.mlflow_enabled:
            return self._train_with_mlflow(
                features, 
                feature_names, 
                experiment_name, 
                run_name
            )
        else:
            return self._train_without_mlflow(features, feature_names)
    
    def _train_without_mlflow(self, features: np.ndarray, feature_names: List[str]) -> dict:
        """Train model without MLflow tracking."""
        # Train model
        start_time = time.time()
        self.detector.train(features, feature_names)
        training_time = time.time() - start_time
        
        # Get predictions on training set for validation
        predictions, scores = self.detector.predict_batch(features)
        
        # Calculate statistics
        n_anomalies = np.sum(predictions == -1)
        anomaly_rate = n_anomalies / len(predictions)
        
        stats = {
            'n_samples': len(features),
            'n_features': features.shape[1],
            'training_time_seconds': training_time,
            'n_anomalies_detected': int(n_anomalies),
            'anomaly_rate': float(anomaly_rate),
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'score_min': float(np.min(scores)),
            'score_max': float(np.max(scores)),
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Anomaly rate in training set: {anomaly_rate:.2%}")
        logger.info(f"Score statistics: mean={stats['score_mean']:.3f}, "
                   f"std={stats['score_std']:.3f}")
        
        return stats
    
    def _train_with_mlflow(
        self,
        features: np.ndarray,
        feature_names: List[str],
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None
    ) -> dict:
        """Train model with MLflow tracking."""
        # Set experiment
        exp_name = get_experiment_name(experiment_name)
        mlflow.set_experiment(exp_name)
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name) as run:
            # Log tags
            for key, value in DEFAULT_TAGS.items():
                mlflow.set_tag(key, value)
            
            # Log parameters
            mlflow.log_param('contamination', self.detector.contamination)
            mlflow.log_param('n_estimators', self.detector.n_estimators)
            mlflow.log_param('random_state', self.detector.random_state)
            mlflow.log_param('max_samples', self.detector.max_samples)
            mlflow.log_param('n_features', features.shape[1])
            mlflow.log_param('n_training_samples', features.shape[0])
            
            # Train model
            start_time = time.time()
            self.detector.train(features, feature_names)
            training_time = time.time() - start_time
            
            # Get predictions on training set
            predictions, scores = self.detector.predict_batch(features)
            
            # Calculate statistics
            n_anomalies = np.sum(predictions == -1)
            anomaly_rate = n_anomalies / len(predictions)
            
            # Log metrics
            mlflow.log_metric('training_time_seconds', training_time)
            mlflow.log_metric('n_samples', len(features))
            mlflow.log_metric('n_features', features.shape[1])
            mlflow.log_metric('n_anomalies_detected', int(n_anomalies))
            mlflow.log_metric('anomaly_rate', float(anomaly_rate))
            mlflow.log_metric('score_mean', float(np.mean(scores)))
            mlflow.log_metric('score_std', float(np.std(scores)))
            mlflow.log_metric('score_min', float(np.min(scores)))
            mlflow.log_metric('score_max', float(np.max(scores)))
            
            # Log model with signature
            try:
                signature = infer_signature(features, predictions)
                mlflow.sklearn.log_model(
                    self.detector.model,
                    MODEL_ARTIFACT_PATH,
                    signature=signature
                )
            except Exception as e:
                logger.warning(f"Could not log model signature: {e}")
                mlflow.sklearn.log_model(self.detector.model, MODEL_ARTIFACT_PATH)
            
            # Log training data if enabled
            if LOG_TRAINING_DATA and len(self.training_features) > 0:
                self._log_training_data_artifact()
            
            logger.info(f"MLflow run ID: {run.info.run_id}")
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Anomaly rate in training set: {anomaly_rate:.2%}")
            
            stats = {
                'n_samples': len(features),
                'n_features': features.shape[1],
                'training_time_seconds': training_time,
                'n_anomalies_detected': int(n_anomalies),
                'anomaly_rate': float(anomaly_rate),
                'score_mean': float(np.mean(scores)),
                'score_std': float(np.std(scores)),
                'score_min': float(np.min(scores)),
                'score_max': float(np.max(scores)),
                'mlflow_run_id': run.info.run_id,
                'mlflow_experiment_id': run.info.experiment_id
            }
            
            return stats
    
    def _log_training_data_artifact(self):
        """Log training data as MLflow artifact."""
        try:
            import tempfile
            feature_names = self.feature_extractor.get_feature_names()
            df = pd.DataFrame(self.training_features, columns=feature_names)
            df.insert(0, 'ip', self.training_ips)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f.name, index=False)
                mlflow.log_artifact(f.name, 'training_data')
                Path(f.name).unlink()  # Clean up temp file
            
            logger.info("Training data logged to MLflow")
        except Exception as e:
            logger.warning(f"Could not log training data: {e}")
    
    def save_model(self, version: Optional[int] = None, register_model: bool = True):
        """Save the trained model to disk and optionally register to MLflow.
        
        Args:
            version: Model version number (optional)
            register_model: Register model to MLflow Model Registry
        """
        if version is not None:
            model_path = Path(MODEL_DIR) / f'isolation_forest_v{version}.joblib'
            scaler_path = Path(MODEL_DIR) / f'scaler_v{version}.joblib'
            self.detector.save(str(model_path), str(scaler_path))
        else:
            self.detector.save()
        
        logger.info("Model saved successfully")
        
        # Register model to MLflow if enabled and active run exists
        if self.mlflow_enabled and register_model:
            try:
                active_run = mlflow.active_run()
                if active_run:
                    model_uri = f"runs:/{active_run.info.run_id}/{MODEL_ARTIFACT_PATH}"
                    mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
                    logger.info(f"Model registered to MLflow: {REGISTERED_MODEL_NAME}")
            except Exception as e:
                logger.warning(f"Could not register model to MLflow: {e}")
    
    def save_training_data(self, filepath: Optional[str] = None):
        """Save collected training data to CSV file.
        
        Args:
            filepath: Output file path (defaults to TRAINING_DATA_DIR)
        """
        if not self.training_features:
            logger.warning("No training data to save")
            return
        
        # Create default filepath if not provided
        if filepath is None:
            Path(TRAINING_DATA_DIR).mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = Path(TRAINING_DATA_DIR) / f'baseline_traffic_{timestamp}.csv'
        
        # Create DataFrame
        feature_names = self.feature_extractor.get_feature_names()
        df = pd.DataFrame(self.training_features, columns=feature_names)
        df.insert(0, 'ip', self.training_ips)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Training data saved to {filepath}")
    
    def validate_model(self, test_features: Optional[np.ndarray] = None) -> dict:
        """Validate the trained model.
        
        Args:
            test_features: Optional test set (uses training set if None)
        
        Returns:
            Dictionary with validation metrics
        """
        if test_features is None:
            test_features = np.array(self.training_features)
            logger.info("Validating on training set (no separate test set provided)")
        
        # Get predictions
        predictions, scores = self.detector.predict_batch(test_features)
        
        # Calculate metrics
        n_anomalies = np.sum(predictions == -1)
        anomaly_rate = n_anomalies / len(predictions)
        
        metrics = {
            'n_samples': len(test_features),
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': float(anomaly_rate),
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'score_min': float(np.min(scores)),
            'score_max': float(np.max(scores)),
        }
        
        logger.info(f"Validation results: {n_anomalies}/{len(test_features)} "
                   f"anomalies ({anomaly_rate:.2%})")
        
        return metrics


def train_and_save_model(
    duration: int = 60,
    contamination: float = 0.01,
    save_data: bool = True,
    version: Optional[int] = None
) -> dict:
    """Convenience function to train and save a model.
    
    Args:
        duration: Traffic collection duration in seconds
        contamination: Expected anomaly proportion
        save_data: Whether to save training data to CSV
        version: Model version number
    
    Returns:
        Training statistics dictionary
    """
    trainer = ModelTrainer()
    
    # Collect baseline traffic
    n_samples = trainer.collect_baseline_traffic(duration=duration)
    
    if n_samples < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Collected insufficient samples: {n_samples} "
            f"(minimum: {MIN_TRAINING_SAMPLES})"
        )
    
    # Save training data if requested
    if save_data:
        trainer.save_training_data()
    
    # Train model
    stats = trainer.train_model(contamination=contamination)
    
    # Save model
    trainer.save_model(version=version)
    
    return stats
