"""LSTM Autoencoder-based anomaly detector for network traffic.

This module implements an LSTM Autoencoder for unsupervised anomaly detection.
Unlike Isolation Forest, it captures temporal patterns in traffic sequences.

Architecture:
    Input (seq_len × n_features) → LSTM Encoder → Latent → LSTM Decoder → Output
    Anomaly = High reconstruction error

Advantages over Isolation Forest:
    - Captures temporal dependencies (traffic patterns over time)
    - Better at detecting slow attacks, gradual changes
    - Learns complex sequential patterns
"""

import os
import json
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
from collections import deque

import numpy as np
import joblib

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. LSTM Autoencoder will not be available.")

from sklearn.preprocessing import StandardScaler

from .ml_config import (
    N_FEATURES,
    MIN_TRAINING_SAMPLES,
    MODEL_DIR,
    FEATURE_NAMES,
)
from .logger_setup import logger

# MLflow imports (optional)
try:
    import mlflow
    import mlflow.pytorch
    from .mlflow_config import (
        get_tracking_uri,
        REGISTERED_MODEL_NAME,
        MODEL_ARTIFACT_PATH,
        apply_s3_config
    )
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# ========== Model Configuration ==========

# Sequence length (number of time steps to consider)
SEQUENCE_LENGTH = int(os.getenv('LSTM_SEQUENCE_LENGTH', 50))

# LSTM hidden dimensions
HIDDEN_DIM = int(os.getenv('LSTM_HIDDEN_DIM', 64))

# Latent space dimension
LATENT_DIM = int(os.getenv('LSTM_LATENT_DIM', 32))

# Number of LSTM layers
NUM_LAYERS = int(os.getenv('LSTM_NUM_LAYERS', 2))

# Dropout rate
DROPOUT = float(os.getenv('LSTM_DROPOUT', 0.2))

# Training parameters
BATCH_SIZE = int(os.getenv('LSTM_BATCH_SIZE', 32))
LEARNING_RATE = float(os.getenv('LSTM_LEARNING_RATE', 0.001))
EPOCHS = int(os.getenv('LSTM_EPOCHS', 50))
EARLY_STOPPING_PATIENCE = int(os.getenv('LSTM_EARLY_STOPPING', 10))

# Anomaly threshold (percentile of training reconstruction errors)
ANOMALY_PERCENTILE = float(os.getenv('LSTM_ANOMALY_PERCENTILE', 95))

# Model paths
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_autoencoder.pt')
LSTM_SCALER_PATH = os.path.join(MODEL_DIR, 'lstm_scaler.joblib')
LSTM_CONFIG_PATH = os.path.join(MODEL_DIR, 'lstm_config.json')


class LSTMEncoder(nn.Module):
    """LSTM Encoder: compresses sequence to latent representation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        # x: (batch, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state for latent representation
        # hidden: (num_layers, batch, hidden_dim) -> take last layer
        latent = self.fc_latent(hidden[-1])  # (batch, latent_dim)
        
        return latent, (hidden, cell)


class LSTMDecoder(nn.Module):
    """LSTM Decoder: reconstructs sequence from latent representation."""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Expand latent to sequence
        # latent: (batch, latent_dim)
        hidden_init = self.fc_hidden(latent)  # (batch, hidden_dim)
        
        # Repeat for each timestep
        decoder_input = hidden_init.unsqueeze(1).repeat(1, self.seq_len, 1)
        # decoder_input: (batch, seq_len, hidden_dim)
        
        # Initialize hidden states
        batch_size = latent.size(0)
        h0 = hidden_init.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        lstm_out, _ = self.lstm(decoder_input, (h0, c0))
        # lstm_out: (batch, seq_len, hidden_dim)
        
        output = self.fc_output(lstm_out)
        # output: (batch, seq_len, output_dim)
        
        return output


class LSTMAutoencoder(nn.Module):
    """Complete LSTM Autoencoder for anomaly detection."""
    
    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        seq_len: int = SEQUENCE_LENGTH,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent, _ = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate per-sample reconstruction error (MSE)."""
        self.eval()
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            # MSE per sample
            error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return error


class LSTMAutoencoderDetector:
    """LSTM Autoencoder anomaly detector for network traffic.
    
    This class wraps the LSTM Autoencoder model and provides:
    - Training on normal traffic sequences
    - Anomaly detection based on reconstruction error
    - Model persistence and loading
    - MLflow integration for experiment tracking
    """
    
    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        device: Optional[str] = None
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM Autoencoder. Install with: pip install torch")
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model: Optional[LSTMAutoencoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.threshold: float = 0.0
        self.is_trained: bool = False
        
        # Sequence buffer for real-time detection
        self.sequence_buffer: deque = deque(maxlen=sequence_length)
        self._buffer_lock = threading.Lock()

        logger.info(f"LSTMAutoencoderDetector initialized (device: {self.device})")
    
    def _create_model(self) -> LSTMAutoencoder:
        """Create a new LSTM Autoencoder model."""
        model = LSTMAutoencoder(
            input_dim=N_FEATURES,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            seq_len=self.sequence_length,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        return model.to(self.device)
    
    def _prepare_sequences(self, features: np.ndarray) -> np.ndarray:
        """Convert flat features to sequences.
        
        Args:
            features: Array of shape (n_samples, n_features)
            
        Returns:
            sequences: Array of shape (n_sequences, seq_len, n_features)
        """
        n_samples = len(features)
        n_sequences = n_samples - self.sequence_length + 1
        
        if n_sequences <= 0:
            raise ValueError(f"Need at least {self.sequence_length} samples, got {n_samples}")
        
        sequences = np.zeros((n_sequences, self.sequence_length, N_FEATURES))
        for i in range(n_sequences):
            sequences[i] = features[i:i + self.sequence_length]
        
        return sequences
    
    def train(
        self,
        features: np.ndarray,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        validation_split: float = 0.2,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
        log_to_mlflow: bool = True
    ) -> Dict[str, Any]:
        """Train the LSTM Autoencoder on normal traffic.
        
        Args:
            features: Training features of shape (n_samples, n_features)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            validation_split: Fraction of data for validation
            early_stopping_patience: Stop if no improvement for N epochs
            log_to_mlflow: Whether to log metrics to MLflow
            
        Returns:
            Training statistics dictionary
        """
        n_samples = len(features)
        min_required = self.sequence_length + MIN_TRAINING_SAMPLES
        
        if n_samples < min_required:
            raise ValueError(f"Need at least {min_required} samples, got {n_samples}")
        
        logger.info(f"Training LSTM Autoencoder on {n_samples} samples...")
        
        # Fit scaler and normalize
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        sequences = self._prepare_sequences(features_scaled)
        n_sequences = len(sequences)
        
        # Train/validation split
        split_idx = int(n_sequences * (1 - validation_split))
        train_sequences = sequences[:split_idx]
        val_sequences = sequences[split_idx:]
        
        # Create data loaders
        train_tensor = torch.FloatTensor(train_sequences).to(self.device)
        val_tensor = torch.FloatTensor(val_sequences).to(self.device)
        
        train_loader = DataLoader(
            TensorDataset(train_tensor, train_tensor),
            batch_size=batch_size,
            shuffle=True
        )
        
        # Create model and optimizer
        self.model = self._create_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        # MLflow logging
        if log_to_mlflow and MLFLOW_AVAILABLE:
            try:
                apply_s3_config()
                mlflow.set_tracking_uri(get_tracking_uri())
                mlflow.set_experiment("lstm-autoencoder-anomaly-detection")
                mlflow.start_run()
                mlflow.log_params({
                    'sequence_length': self.sequence_length,
                    'hidden_dim': self.hidden_dim,
                    'latent_dim': self.latent_dim,
                    'num_layers': self.num_layers,
                    'dropout': self.dropout,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'n_samples': n_samples,
                    'n_sequences': n_sequences,
                })
            except Exception as e:
                logger.warning(f"MLflow logging disabled: {e}")
                log_to_mlflow = False
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0.0
            
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                reconstructed, _ = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_reconstructed, _ = self.model(val_tensor)
                val_loss = criterion(val_reconstructed, val_tensor).item()
            val_losses.append(val_loss)
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if log_to_mlflow and MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metrics({
                        'train_loss': epoch_train_loss,
                        'val_loss': val_loss,
                    }, step=epoch)
                except Exception:
                    pass
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        self.model.load_state_dict(best_model_state)
        
        # Calculate threshold from training reconstruction errors
        self.model.eval()
        with torch.no_grad():
            train_errors = self.model.get_reconstruction_error(train_tensor).cpu().numpy()
        
        self.threshold = np.percentile(train_errors, ANOMALY_PERCENTILE)
        self.is_trained = True
        
        # Training statistics
        stats = {
            'n_samples': n_samples,
            'n_sequences': n_sequences,
            'epochs_trained': epoch + 1,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': best_val_loss,
            'threshold': self.threshold,
            'device': str(self.device),
        }
        
        logger.info(f"Training complete. Threshold: {self.threshold:.6f}")
        
        if log_to_mlflow and MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    'final_train_loss': stats['final_train_loss'],
                    'final_val_loss': stats['final_val_loss'],
                    'threshold': self.threshold,
                })
                mlflow.end_run()
            except Exception:
                pass
        
        return stats
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies for feature sequences.
        
        Args:
            features: Features of shape (n_samples, n_features)
            
        Returns:
            predictions: Array of -1 (anomaly) or 1 (normal)
            errors: Reconstruction errors per sequence
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        if len(features) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} samples")
        
        # Normalize
        features_scaled = self.scaler.transform(features)
        
        # Create sequences
        sequences = self._prepare_sequences(features_scaled)
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        
        # Get reconstruction errors
        self.model.eval()
        errors = self.model.get_reconstruction_error(sequences_tensor).cpu().numpy()
        
        # Classify based on threshold
        predictions = np.where(errors > self.threshold, -1, 1)
        
        return predictions, errors
    
    def predict_single(self, feature_vector: np.ndarray) -> Tuple[int, float]:
        """Add a feature vector to buffer and predict if anomaly.

        Thread-safe: buffer operations are protected by a lock.

        Args:
            feature_vector: Single feature vector of shape (n_features,)

        Returns:
            prediction: -1 (anomaly), 1 (normal), or 0 (insufficient data)
            error: Reconstruction error (0 if insufficient data)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        with self._buffer_lock:
            self.sequence_buffer.append(feature_vector)
            if len(self.sequence_buffer) < self.sequence_length:
                return 0, 0.0
            sequence = np.array(list(self.sequence_buffer))

        sequence_scaled = self.scaler.transform(sequence)
        sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(self.device)

        # Predict
        self.model.eval()
        error = self.model.get_reconstruction_error(sequence_tensor).item()
        prediction = -1 if error > self.threshold else 1

        return prediction, error
    
    def save(self, path: Optional[str] = None) -> str:
        """Save the trained model to disk.
        
        Args:
            path: Directory to save the model (default: MODEL_DIR)
            
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Nothing to save.")
        
        save_dir = Path(path or MODEL_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_path = save_dir / 'lstm_autoencoder.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold,
        }, model_path)
        
        # Save scaler
        scaler_path = save_dir / 'lstm_scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        # Save config
        config_path = save_dir / 'lstm_config.json'
        config = {
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'n_features': N_FEATURES,
            'threshold': self.threshold,
            'saved_at': datetime.now().isoformat(),
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {save_dir}")
        return str(save_dir)
    
    def load(self, path: Optional[str] = None) -> bool:
        """Load a trained model from disk.
        
        Args:
            path: Directory containing the saved model
            
        Returns:
            True if successful, False otherwise
        """
        load_dir = Path(path or MODEL_DIR)
        
        try:
            # Load config
            config_path = load_dir / 'lstm_config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.sequence_length = config['sequence_length']
            self.hidden_dim = config['hidden_dim']
            self.latent_dim = config['latent_dim']
            self.num_layers = config['num_layers']
            self.dropout = config['dropout']
            self.threshold = config['threshold']
            
            # Load model
            model_path = load_dir / 'lstm_autoencoder.pt'
            self.model = self._create_model()
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load scaler
            scaler_path = load_dir / 'lstm_scaler.joblib'
            self.scaler = joblib.load(scaler_path)
            
            # Reset buffer
            self.sequence_buffer = deque(maxlen=self.sequence_length)
            
            self.is_trained = True
            logger.info(f"Model loaded from {load_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_latent_representation(self, features: np.ndarray) -> np.ndarray:
        """Get latent representations for visualization/clustering.
        
        Args:
            features: Features of shape (n_samples, n_features)
            
        Returns:
            latent: Latent vectors of shape (n_sequences, latent_dim)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        features_scaled = self.scaler.transform(features)
        sequences = self._prepare_sequences(features_scaled)
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            _, latent = self.model(sequences_tensor)
        
        return latent.cpu().numpy()


# Convenience function for comparison with Isolation Forest
def compare_with_isolation_forest(
    features: np.ndarray,
    lstm_detector: LSTMAutoencoderDetector,
    isolation_forest_detector: Any
) -> Dict[str, Any]:
    """Compare predictions between LSTM Autoencoder and Isolation Forest.
    
    Args:
        features: Features for prediction
        lstm_detector: Trained LSTMAutoencoderDetector
        isolation_forest_detector: Trained IsolationForestDetector
        
    Returns:
        Comparison statistics
    """
    # LSTM predictions
    lstm_preds, lstm_errors = lstm_detector.predict(features)
    
    # Isolation Forest predictions (on overlapping samples)
    # IF predicts on individual samples, LSTM on sequences
    offset = lstm_detector.sequence_length - 1
    if_features = features[offset:]
    if_preds = isolation_forest_detector.predict(if_features)
    
    # Align predictions
    n_compare = min(len(lstm_preds), len(if_preds))
    lstm_preds = lstm_preds[:n_compare]
    if_preds = if_preds[:n_compare]
    
    # Calculate agreement
    agreement = np.mean(lstm_preds == if_preds)
    lstm_anomalies = np.sum(lstm_preds == -1)
    if_anomalies = np.sum(if_preds == -1)
    both_anomalies = np.sum((lstm_preds == -1) & (if_preds == -1))
    
    return {
        'agreement_rate': agreement,
        'lstm_anomaly_count': lstm_anomalies,
        'isolation_forest_anomaly_count': if_anomalies,
        'both_detected_count': both_anomalies,
        'samples_compared': n_compare,
    }
