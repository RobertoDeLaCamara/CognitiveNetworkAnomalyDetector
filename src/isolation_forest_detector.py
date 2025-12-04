"""Isolation Forest-based anomaly detector for network traffic.

This module implements the Isolation Forest algorithm for unsupervised
anomaly detection. It can train on baseline normal traffic and detect
anomalies in real-time.
"""

import os
import joblib
import numpy as np
from typing import Tuple, Optional
from sklearn.ensemble import IsolationForest
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
        """Save the trained model and scaler to disk.
        
        Args:
            model_path: Path to save model (default: from ml_config)
            scaler_path: Path to save scaler (default: from ml_config)
        
        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        model_path = model_path or ISOLATION_FOREST_MODEL_PATH
        scaler_path = scaler_path or SCALER_MODEL_PATH
        
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Save model and scaler
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'max_samples': self.max_samples,
            'feature_names': self.feature_names,
            'n_features': N_FEATURES
        }
        metadata_path = model_path.replace('.joblib', '_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """Load a trained model and scaler from disk.
        
        Args:
            model_path: Path to load model from (default: from ml_config)
            scaler_path: Path to load scaler from (default: from ml_config)
        
        Raises:
            FileNotFoundError: If model or scaler files don't exist
        """
        model_path = model_path or ISOLATION_FOREST_MODEL_PATH
        scaler_path = scaler_path or SCALER_MODEL_PATH
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        # Load model and scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True
        
        # Load metadata if available
        metadata_path = model_path.replace('.joblib', '_metadata.joblib')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.contamination = metadata.get('contamination', self.contamination)
            self.n_estimators = metadata.get('n_estimators', self.n_estimators)
            self.feature_names = metadata.get('feature_names')
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Scaler loaded from {scaler_path}")
    
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
