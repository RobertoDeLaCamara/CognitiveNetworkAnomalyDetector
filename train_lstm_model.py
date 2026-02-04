#!/usr/bin/env python3
"""Training script for LSTM Autoencoder anomaly detector.

Usage:
    # Train with synthetic data
    python train_lstm_model.py --from-file data/training/synthetic_baseline.csv
    
    # Train with live traffic
    sudo python train_lstm_model.py --duration 120
    
    # Custom parameters
    python train_lstm_model.py --from-file data.csv --epochs 100 --hidden-dim 128
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from src.lstm_autoencoder_detector import LSTMAutoencoderDetector
from src.ml_config import FEATURE_NAMES, N_FEATURES
from src.logger_setup import logger


def load_training_data(file_path: str) -> np.ndarray:
    """Load training data from CSV file."""
    df = pd.read_csv(file_path)
    
    # Check if all feature columns exist
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    
    features = df[FEATURE_NAMES].values
    logger.info(f"Loaded {len(features)} samples from {file_path}")
    return features


def collect_live_traffic(duration: int) -> np.ndarray:
    """Collect live network traffic for training."""
    from src.feature_extractor import FeatureExtractor
    from scapy.all import sniff, IP
    
    logger.info(f"Collecting live traffic for {duration} seconds...")
    
    extractor = FeatureExtractor()
    features_list = []
    
    def packet_callback(packet):
        if IP in packet:
            src_ip = packet[IP].src
            extractor.update(src_ip, packet)
            feature_vector = extractor.extract_features(src_ip)
            if feature_vector is not None:
                features_list.append(feature_vector)
    
    sniff(
        filter="ip",
        prn=packet_callback,
        timeout=duration,
        store=False
    )
    
    if not features_list:
        raise RuntimeError("No packets captured. Check permissions (need sudo).")
    
    features = np.array(features_list)
    logger.info(f"Collected {len(features)} feature vectors")
    return features


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder for anomaly detection")
    
    # Data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--from-file', type=str, help='Training data CSV file')
    data_group.add_argument('--duration', type=int, help='Seconds to collect live traffic')
    
    # Model parameters
    parser.add_argument('--sequence-length', type=int, default=20, help='Sequence length (default: 20)')
    parser.add_argument('--hidden-dim', type=int, default=64, help='LSTM hidden dimension (default: 64)')
    parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension (default: 32)')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='models', help='Model output directory')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow logging')
    
    args = parser.parse_args()
    
    # Load or collect training data
    if args.from_file:
        features = load_training_data(args.from_file)
    else:
        features = collect_live_traffic(args.duration)
    
    # Create detector
    detector = LSTMAutoencoderDetector(
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    
    # Train
    stats = detector.train(
        features=features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_to_mlflow=not args.no_mlflow,
    )
    
    # Save model
    detector.save(args.output_dir)
    
    # Print summary
    print("\n" + "=" * 50)
    print("LSTM Autoencoder Training Complete")
    print("=" * 50)
    print(f"Samples:           {stats['n_samples']}")
    print(f"Sequences:         {stats['n_sequences']}")
    print(f"Epochs trained:    {stats['epochs_trained']}")
    print(f"Final train loss:  {stats['final_train_loss']:.6f}")
    print(f"Final val loss:    {stats['final_val_loss']:.6f}")
    print(f"Anomaly threshold: {stats['threshold']:.6f}")
    print(f"Device:            {stats['device']}")
    print(f"Model saved to:    {args.output_dir}/")
    print("=" * 50)


if __name__ == '__main__':
    main()
