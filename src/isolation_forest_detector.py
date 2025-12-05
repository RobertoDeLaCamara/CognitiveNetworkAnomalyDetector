"""Isolation Forest-based anomaly detector for network traffic.

This module implements the Isolation Forest algorithm for unsupervised
anomaly detection. It can train on baseline normal traffic and detect
anomalies in real-time.
"""

import os
import joblib
import warnings
from datetime import datetime
import numpy as np
from typing import Tuple, Optional, Dict, List, Any
from sklearn.ensemble import IsolationForest
import sklearn
from sklearn.preprocessing import StandardScaler

from .ml_config import (
    CONTAMINATION,
    N_ESTIMATORS,
    RANDOM_STATE,
    MAX_SAMPLES,
    ISOLATION_FOREST_MODEL_PATH,
    SCALER_MODEL_PATH,
    MODEL_DIR,
    N_FEATURES
)
from .logger_setup import logger


class IsolationForestDetector:
    """Isolation Forest anomaly detector for network traffic."""
    
    def __init__(
        self,
        contamination: float = CONTAMINATION,
        n_estimators: int = N_ESTIMATORS,
        random_state: int = RANDOM_STATE,
        max_samples: str = MAX_SAMPLES
    ):
        """Initialize the Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.0 to 0.5)
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
            max_samples: Number of samples to use per tree ('auto' or int)
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_samples = max_samples
        
        # Initialize models
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            max_samples=max_samples,
            n_jobs=-1,  # Use all CPU cores
            verbose=0
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
    
    def train(self, features: np.ndarray, feature_names: Optional[list] = None):
        """Train the Isolation Forest on baseline normal traffic.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            feature_names: Optional list of feature names for logging
        
        Raises:
            ValueError: If features have wrong shape or insufficient samples
        """
        if features.ndim != 2:
            raise ValueError(f"Features must be 2D array, got shape {features.shape}")
        
        if features.shape[1] != N_FEATURES:
            raise ValueError(
                f"Expected {N_FEATURES} features, got {features.shape[1]}"
            )
        
        if features.shape[0] < 10:
            raise ValueError(
                f"Need at least 10 samples for training, got {features.shape[0]}"
            )
        
        logger.info(f"Training Isolation Forest on {features.shape[0]} samples...")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Fit scaler
        self.scaler.fit(features)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Train model
        self.model.fit(features_scaled)
        self.is_trained = True
        
        logger.info(
            f"Isolation Forest trained successfully "
            f"(contamination={self.contamination}, n_estimators={self.n_estimators})"
        )
        
        # Log basic statistics
        scores = self.model.score_samples(features_scaled)
        logger.info(
            f"Training set anomaly scores - "
            f"Mean: {np.mean(scores):.3f}, "
            f"Std: {np.std(scores):.3f}, "
            f"Min: {np.min(scores):.3f}, "
            f"Max: {np.max(scores):.3f}"
        )
    
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """Predict if a sample is anomalous.
        
        Args:
            features: Feature vector of shape (n_features,) or (1, n_features)
        
        Returns:
            Tuple of (prediction, anomaly_score)
            - prediction: 1 for normal, -1 for anomaly
            - anomaly_score: Lower scores indicate anomalies (typically -1 to 1)
        
        Raises:
            RuntimeError: If model is not trained
            ValueError: If features have wrong shape
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Ensure 2D shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if features.shape[1] != N_FEATURES:
            raise ValueError(
                f"Expected {N_FEATURES} features, got {features.shape[1]}"
            )
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        anomaly_score = self.model.score_samples(features_scaled)[0]
        
        return prediction, anomaly_score
    
    def predict_batch(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies for multiple samples.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
        
        Returns:
            Tuple of (predictions, anomaly_scores)
            - predictions: Array of 1 (normal) or -1 (anomaly)
            - anomaly_scores: Array of anomaly scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        if features.shape[1] != N_FEATURES:
            raise ValueError(
                f"Expected {N_FEATURES} features, got {features.shape[1]}"
            )
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predictions = self.model.predict(features_scaled)
        anomaly_scores = self.model.score_samples(features_scaled)
        
        return predictions, anomaly_scores
    
    def save(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """Save the trained model and scaler to disk with metadata.
        
        Args:
            model_path: Path to save model (default: from ml_config)
            scaler_path: Path to save scaler (default: from ml_config)
        
        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        # Use default paths if not specified
        if model_path is None:
            model_path = ISOLATION_FOREST_MODEL_PATH
        if scaler_path is None:
            scaler_path = SCALER_MODEL_PATH
        
        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        try:
            # Create metadata
            metadata = {
                'version': 1,  # Model format version
                'training_date': datetime.now().isoformat(),
                'sklearn_version': sklearn.__version__,
                'n_features': N_FEATURES,
                'contamination': self.contamination,
                'n_estimators': self.n_estimators,
                'random_state': self.random_state,
                'max_samples': self.max_samples,
                'feature_names': self.feature_names if hasattr(self, 'feature_names') else None
            }
            
            # Save model with metadata
            model_data = {
                'model': self.model,
                'metadata': metadata
            }
            joblib.dump(model_data, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"Model saved to {model_path} with metadata")
            logger.info(f"Scaler saved to {scaler_path}")
            logger.info(f"Model version: {metadata['version']}, sklearn: {metadata['sklearn_version']}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """Load a trained model and scaler from disk with metadata validation.
        
        Args:
            model_path: Path to load model from (default: from ml_config)
            scaler_path: Path to load scaler from (default: from ml_config)
        
        Raises:
            FileNotFoundError: If model or scaler files don't exist
            ValueError: If model is incompatible with current configuration
        """
        # Use default paths if not specified
        if model_path is None:
            model_path = ISOLATION_FOREST_MODEL_PATH
        if scaler_path is None:
            scaler_path = SCALER_MODEL_PATH
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        try:
            # Load model data
            model_data = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Handle both old format (direct model) and new format (with metadata)
            if isinstance(model_data, dict) and 'model' in model_data:
                # New format with metadata
                self.model = model_data['model']
                metadata = model_data.get('metadata', {})
                
                # Validate metadata
                validation_issues = self._validate_model_metadata(metadata)
                
                # Log warnings for non-critical issues
                for issue in validation_issues:
                    if 'CRITICAL' in issue:
                        raise ValueError(issue)
                    else:
                        warnings.warn(issue)
                        logger.warning(issue)
                
                # Update detector's attributes from metadata if available
                self.contamination = metadata.get('contamination', self.contamination)
                self.n_estimators = metadata.get('n_estimators', self.n_estimators)
                self.random_state = metadata.get('random_state', self.random_state)
                self.max_samples = metadata.get('max_samples', self.max_samples)
                self.feature_names = metadata.get('feature_names', self.feature_names)
                
                logger.info(f"Model loaded with metadata: version={metadata.get('version', 'unknown')}")
                logger.info(f"Training date: {metadata.get('training_date', 'unknown')}")
            else:
                # Old format without metadata
                self.model = model_data
                warnings.warn("Loaded model without metadata. This is an older model format.")
                logger.warning("Model loaded without metadata (old format)")
            
            self.is_trained = True
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Scaler loaded from {scaler_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _validate_model_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Validate model metadata for compatibility.
        
        Args:
            metadata: Metadata dictionary from saved model
        
        Returns:
            List of validation issues (warnings or critical errors)
        """
        issues = []
        
        if not metadata:
            issues.append("No metadata found in model file")
            return issues
        
        # Check feature count (CRITICAL)
        if 'n_features' in metadata:
            if metadata['n_features'] != N_FEATURES:
                issues.append(
                    f"CRITICAL: Model was trained with {metadata['n_features']} features, "
                    f"but current config expects {N_FEATURES} features"
                )
        
        # Check sklearn version compatibility
        if 'sklearn_version' in metadata:
            saved_version = metadata['sklearn_version']
            current_version = sklearn.__version__
            
            # Compare major versions
            saved_major = saved_version.split('.')[0]
            current_major = current_version.split('.')[0]
            
            if saved_major != current_major:
                issues.append(
                    f"sklearn major version mismatch: model trained with {saved_version}, "
                    f"current version is {current_version}. Consider retraining."
                )
        
        # Info about model age
        if 'training_date' in metadata:
            try:
                training_date = datetime.fromisoformat(metadata['training_date'])
                age_days = (datetime.now() - training_date).days
                if age_days > 30:
                    issues.append(
                        f"Model is {age_days} days old. Consider retraining with recent data."
                    )
            except (ValueError, TypeError):
                pass
        
        return issues
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores (not directly available in Isolation Forest).
        
        Returns:
            None (Isolation Forest doesn't provide direct feature importance)
        """
        # Note: Isolation Forest doesn't have built-in feature importance
        # Could implement using permutation importance if needed
        logger.warning("Feature importance not available for Isolation Forest")
        return None
    
    @property
    def model_info(self) -> dict:
        """Get information about the current model."""
        return {
            'is_trained': self.is_trained,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'max_samples': self.max_samples,
            'n_features': N_FEATURES,
            'feature_names': self.feature_names
        }
